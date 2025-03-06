# Run job in a remote TPU VM

VM_NAME=kmh-tpuvm-v3-128-2
ZONE=europe-west4-a

echo $VM_NAME $ZONE

# use tpu_vit_mae_recipe to run the MAE recipe or
# tpu_vit_deit_recip eto run the DeiT recipe
CONFIG=tpu_vit_mae_recipe 

# some of the often modified hyperparametes (uncomment to use):
# MAE recipe:
batch=4096
lr=1e-4
wd=0.3
ep=300

# DeiT recipe:
# batch=1024
# lr=0.0005
# wd=0.05
# ep=300

now=`date '+%Y%m%d_%H%M%S'`
export salt=`head /dev/urandom | tr -dc a-z0-9 | head -c6`
JOBNAME=resnet/${now}_${salt}_${VM_NAME}_${CONFIG}_b${batch}_lr${lr}_wd${wd}_ep${ep}_timm_r200_sm0.1_cutmixup_adamw

LOGDIR=/kmh-nfs-ssd-eu-mount/logs/$USER/$JOBNAME

# sudo rm -rf /tmp/libtpu_lockfile
# sudo rm -rf /tpu/tpu_logs


sudo mkdir -p ${LOGDIR}
sudo chmod 777 ${LOGDIR}

echo 'Log dir: '$LOGDIR

gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE --worker=all --command 'sudo chmod -R go+rw /tmp/tpu_logs/'

gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
    --worker=all --command "

sudo mount -o vers=3 10.26.72.146:/kmh_nfs_us /kmh-nfs-us-mount
sudo chmod go+rw /kmh-nfs-us-mount

sudo mount -o vers=3 10.150.179.250:/kmh_nfs_ssd_eu /kmh-nfs-ssd-eu-mount
sudo chmod -R go+rw /kmh-nfs-ssd-eu-mount/logs/shprintsron

sudo rm -rf /tpu/tpu_logs
sudo rm -rf /tmp/libtpu_lockfile


cd $STAGEDIR
echo Current dir: $(pwd)

python3 main.py \
    --workdir=${LOGDIR} --config=configs/${CONFIG}.py \
    --config.dataset.root='/kmh-nfs-ssd-eu-mount/data/imagenet' \
    --config.batch_size=${batch} \
    --config.num_epochs=${ep} \
    --config.learning_rate=${lr} \
    --config.weight_decay=${wd} \
    --config.dataset.prefetch_factor=2 \
    --config.dataset.num_workers=32 \
    --config.log_per_step=20 \
    --config.dataset.aug.label_smooth=0.1 \
    --config.dataset.aug.mixup_alpha=0.8 \
    --config.dataset.aug.cutmix_alpha=1.0 \
" 2>&1 | tee -a $LOGDIR/output.log
