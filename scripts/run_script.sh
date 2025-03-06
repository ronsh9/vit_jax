# clean tmp
rm -rf tmp

# Efficient training:
# Note that some hyperparameters are modified here!
# use tpu_vit_mae_recipe to run the MAE recipe or
# tpu_vit_deit_recip eto run the DeiT recipe

PWD=$(pwd)
python3 main.py \
    --debug=True \
    --workdir=${PWD}/tmp --config=configs/tpu_vit_mae_recipe.py \
    --config.dataset.cache=True \
    --config.dataset.root=/kmh-nfs-ssd-eu-mount/data/imagenet \
    --config.batch_size=4096 \
    --config.dataset.prefetch_factor=2 \
    --config.dataset.num_workers=32 \
    --config.log_per_step=20 \
    --config.learning_rate=0.0001