This repo is largely copied from the [Model Diffing](https://github.com/model-diffing/model-diffing) repo.
Some config and functions are changed to support training crosscoders on the Qwen 1.5B math model and the R1 distilled version of these.

Two notebooks are added to the `model_diffing/analysis folder`:
`gen_math_datasets.ipynb` generates a custom dataset of math question responses from these two models.
`cc_analysis.ipynb` contains code to analyse the crosscoder features.
