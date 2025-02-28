import fire  # type: ignore
import torch
from einops import rearrange


from model_diffing.data.model_hookpoint_dataloader import build_dataloader
from model_diffing.log import logger
from model_diffing.models.activations.topk import TopkActivation
from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.scripts.base_trainer import run_exp
from model_diffing.scripts.llms import build_llms
from model_diffing.scripts.train_l1_crosscoder.trainer import AnthropicTransposeInit
from model_diffing.scripts.train_topk_crosscoder.config import TopKExperimentConfig
from model_diffing.scripts.train_topk_crosscoder.trainer import TopKTrainer
from model_diffing.scripts.utils import build_wandb_run
from model_diffing.utils import get_device

from model_diffing.log import logger
from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.scripts.train_l1_crosscoder.trainer import AnthropicTransposeInit
from model_diffing.models.activations.topk import TopkActivation
from sleepers.scripts.train_topk_sleeper.config import TopKExperimentConfig
from sleepers.scripts.train_topk_sleeper.trainer import TopKTrainer
from model_diffing.data.model_hookpoint_dataloader import BaseModelHookpointActivationsDataloader
from model_diffing.scripts.base_trainer import run_exp
from model_diffing.scripts.utils import build_wandb_run
from model_diffing.utils import get_device, size_human_readable



def harvest_pre_pre_bias_acts(
    data_loader: BaseModelHookpointActivationsDataloader,
    W_enc_XDH: torch.Tensor,
    device: torch.device,
    n_examples_to_sample: int = 100_000,
) -> torch.Tensor:
    batch_size = data_loader._yield_batch_size

    remainder = n_examples_to_sample % batch_size
    if remainder != 0:
        logger.warning(
            f"n_examples_to_sample {n_examples_to_sample} must be divisible by the batch "
            f"size {batch_size}. Rounding up to the nearest multiple of batch_size."
        )
        # Round up to the nearest multiple of batch_size:
        n_examples_to_sample = (((n_examples_to_sample - remainder) // batch_size) + 1) * batch_size

        logger.info(f"n_examples_to_sample is now {n_examples_to_sample}")

    activations_iterator_BMPD = data_loader.get_shuffled_activations_iterator_BMPD()

    def get_batch_pre_bias_pre_act() -> torch.Tensor:
        # this is essentially the first step of the crosscoder forward pass, but not worth
        # creating a new method for it, just (easily) reimplementing it here
        batch_BMPD = next(activations_iterator_BMPD)
        x_BH = torch.einsum("b m l d, m l d h -> b h", batch_BMPD, W_enc_XDH)
        return x_BH

    first_sample_BH = get_batch_pre_bias_pre_act()
    hidden_size = first_sample_BH.shape[1]

    pre_bias_pre_act_buffer_NH = torch.empty(n_examples_to_sample, hidden_size, device=device)
    logger.info(
        f"pre_bias_pre_act_buffer_NH.shape: {pre_bias_pre_act_buffer_NH.shape}, "
        f"size: {size_human_readable(pre_bias_pre_act_buffer_NH)}"
    )

    pre_bias_pre_act_buffer_NH[:batch_size] = first_sample_BH
    examples_sampled = batch_size

    while examples_sampled < n_examples_to_sample:
        batch_pre_bias_pre_act_BH = get_batch_pre_bias_pre_act()
        pre_bias_pre_act_buffer_NH[examples_sampled : examples_sampled + batch_size] = batch_pre_bias_pre_act_BH
        examples_sampled += batch_size
    return pre_bias_pre_act_buffer_NH

def _compute_b_enc_H(
    data_loader: BaseModelHookpointActivationsDataloader,
    W_enc_XDH: torch.Tensor,
    initial_threshold_H: torch.Tensor,
    device: torch.device,
    n_examples_to_sample: int = 20_000,
) -> torch.Tensor:
    pre_bias_pre_act_buffer_NH = harvest_pre_pre_bias_acts(data_loader, W_enc_XDH, device, n_examples_to_sample)

    # find the threshold for each idx H such that 1/10_000 of the examples are above the threshold
    quantile_H = torch.quantile(pre_bias_pre_act_buffer_NH, 0.5, dim=0)

    b_enc_H = initial_threshold_H - quantile_H

    return b_enc_H

def build_trainer(cfg: TopKExperimentConfig) -> TopKTrainer:
    device = get_device()

    llms = build_llms(
        cfg.data.activations_harvester.llms,
        cfg.cache_dir,
        device,
        dtype=cfg.data.activations_harvester.inference_dtype,
    )

    dataloader = build_dataloader(
        cfg.data,
        llms,
        cfg.hookpoints,
        cfg.train.batch_size,
        cfg.cache_dir,
        device,
    )

    n_models = len(llms)
    n_hookpoints = len(cfg.hookpoints)

    cc = AcausalCrosscoder(
        crosscoding_dims=(n_models, n_hookpoints),
        d_model=llms[0].cfg.d_model,
        hidden_dim=cfg.crosscoder.hidden_dim,
        init_strategy=AnthropicTransposeInit(dec_init_norm=cfg.crosscoder.dec_init_norm),
        hidden_activation=TopkActivation(k=cfg.crosscoder.k),
    )

    with torch.no_grad():
        # parameters from the jan update doc

        n = float(n_models * n_hookpoints * llms[0].cfg.d_model)  # n is the size of the input space
        m = float(cfg.crosscoder.hidden_dim)  # m is the size of the hidden space

        # W_dec ~ U(-1/n, 1/n) (from doc)
        cc.W_dec_HXD.uniform_(-1.0 / n, 1.0 / n)

        # For now, assume we're in the X == Y case.
        # Therefore W_enc = (n/m) * W_dec^T
        cc.W_enc_XDH.copy_(
            rearrange(cc.W_dec_HXD, "hidden model layer d_model -> model layer d_model hidden")  #
            * (n / m)
        )

        calibrated_b_enc_H = _compute_b_enc_H(
            dataloader,
            cc.W_enc_XDH.to(device),
            torch.nn.Parameter(torch.ones(cfg.crosscoder.hidden_dim) * 0.1).exp().to(device),
            device,
        )
        cc.b_enc_H.copy_(calibrated_b_enc_H)

        # no data-dependent initialization of b_dec
        cc.b_dec_XD.zero_()

    crosscoder = cc
    crosscoder = crosscoder.to(device)

    wandb_run = build_wandb_run(cfg)

    return TopKTrainer(
        cfg=cfg.train,
        activations_dataloader=dataloader,
        crosscoder=crosscoder,
        wandb_run=wandb_run,
        device=device,
        hookpoints=cfg.hookpoints,
        save_dir=cfg.save_dir,
    )


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(build_trainer, TopKExperimentConfig))
