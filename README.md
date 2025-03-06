# How to run the code in this branch

### Comment

Before we start please read the **"Jax training with PyTorch dataloaders"** section for general instruction regarding running this code. This section will discuss specific details for this branch (e.g., how to run ViT with a specific recipe).

### Submitting jobs

The shell (`.sh`) files are located in the `scripts/` folder. The dependecies have been taken care of. For example, in order to run the code on the TPU v3 machine use `source scripts/run_staging.sh` (please make sure to modify these files and set your machines if you are submitting big jobs, or let Ron know if you want to use the default machines). **Notice that while the training parameters are still defind in the `config/` folder (primarily in `config/default.py`), some common parameters are overwritten in `config/tpu_vit_mae_recipe.py`, `config/tpu_vit_deit_recipe.py`, and `scripts/run_remote.sh`**. This should make running several experiments more convinient, but make sure to check these folder and to document the parameters that are used for better reproducibility. You may use this Google Sheet to document your experiments (make a copy to use it): https://docs.google.com/spreadsheets/d/1T1uZPGW-Zl__3RwfSFk3VMIs1OGNcnzq0LO-ooUvA6k/edit?usp=sharing. 

### Training

The training script for the most updated run (as imported by the `main.py` file) is in the `train_vit.py` file. The code for the ViT model is in `vit_model/models_vit.py`. Note that this repo supports two training recipes by default:

1. MAE (https://arxiv.org/abs/2111.06377)
2. DeiT (https://arxiv.org/abs/2012.12877)

In order to use one of these you need to specify the right config in the `scripts/run_remote.sh` or `scripts/run_script.sh` (if running locally). As a reminder, some of these parameters are overwritten in `config/tpu_vit_mae_recipe.py`, `config/tpu_vit_deit_recipe.py`, and `scripts/run_remote.sh`. By default, the branch is running the MAE recipe. For the DeiT recipe you also need to change the line `base_learning_rate = config.learning_rate * config.batch_size / 256` in `train_vit.py` to `base_learning_rate = config.learning_rate * config.batch_size / 512`.

### Other comments

- In `vit_model/models_vit.py`, search for `FLAG_INIT` for places where the initialization has been modified from the original JAX initialization.

- In `config/tpu_vit_mae_recipe.py` and `config/tpu_vit_deit_recipe.py` notice the different `FLAG` comments that indicate if these parameters are overwritten elsewhere. Notice that the default values in these files follow the original recipes. 

- The `augment.py` file includes augmentation code per the DeiT recipe (https://github.com/facebookresearch/deit). 

- The code for `input_pipeline.py` has been taken from the timm branch for mixup and cutmix. Special thanks to Kaiming for providing this code as well as additional contributions and comments that helped establishing this branch.