username=`whoami`

if [ "$username" = "zzy" ]; then
    project_code_root="/home/zzy/robot/robot_zzy"
    data_root="/home/zzy/robot/data/diffusion_policy_data/data"
    is_slurm=false
elif [ "$username" = "zhangzhiyuan" ]; then
    project_code_root="/scratch/zhangzhiyuan/robot_zzy"
    data_root="/scratch/zhangzhiyuan/data/diffusion_policy_data/data"
    is_slurm=true
elif [ "$username" = "zhiyuan" ]; then
    project_code_root="$HOME/robot/robot_zzy"
    data_root="$HOME/robot/data/diffusion_policy_data/data"
    is_slurm=false
    home_path=$(echo $HOME)
    if [[ "$home_path" == *"mnt/lustre"* ]]; then
        is_slurm=true
    fi
fi

echo "*** Entering train.sh ***"
echo "username: $username"
echo "project_code_root: $project_code_root"
echo "data_root: $data_root"
echo "is_slurm: $is_slurm"

export HYDRA_FULL_ERROR=1
export DATA_ROOT=$data_root
export ZZY_DEBUG=True
export R3M_HOME=$DATA_ROOT/.r3m
export HF_HOME=$DATA_ROOT/.huggingface
export HF_ENDPOINT="https://hf-mirror.com"

if [ "$is_slurm" = "true" ]; then
    export WANDB_MODE=offline
    cmd_prefix="srun"
else
    cmd_prefix=""
fi

# $cmd_prefix python train.py \
#     --config-dir=. \
#     --config-name=train_ae_workspace.yaml \
#     "hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'" \
#     trainer.max_steps=100 \

# batch size finder
# $cmd_prefix python train.py --config-dir=. --config-name=train_latent_diffusion_workspace.yaml "hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'" datamodule.batch_size=512   trainer.fast_dev_run=True task=lift_image_abs
# $cmd_prefix python train.py --config-dir=. --config-name=train_latent_diffusion_workspace.yaml "hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'" datamodule.batch_size=1024  trainer.fast_dev_run=True task=lift_image_abs
# $cmd_prefix python train.py --config-dir=. --config-name=train_latent_diffusion_workspace.yaml "hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'" datamodule.batch_size=2048  trainer.fast_dev_run=True task=lift_image_abs
# $cmd_prefix python train.py --config-dir=. --config-name=train_latent_diffusion_workspace.yaml "hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'" datamodule.batch_size=4096  trainer.fast_dev_run=True task=lift_image_abs trainer.callbacks=null
# $cmd_prefix python train.py --config-dir=. --config-name=train_latent_diffusion_workspace.yaml "hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'" datamodule.batch_size=8192  trainer.fast_dev_run=True task=lift_image_abs trainer.callbacks=null
# $cmd_prefix python train.py --config-dir=. --config-name=train_latent_diffusion_workspace.yaml "hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'" datamodule.batch_size=16384 trainer.fast_dev_run=True task=lift_image_abs trainer.callbacks=null
# $cmd_prefix python train.py --config-dir=. --config-name=train_latent_diffusion_workspace.yaml "hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'" datamodule.batch_size=32768 trainer.fast_dev_run=True task=lift_image_abs trainer.callbacks=null

# lr finder
# $cmd_prefix python train.py --config-dir=. --config-name=train_ae_workspace.yaml "hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'" trainer.max_steps=200 policy.warmup_steps=100 trainer.val_check_interval=200 trainer.check_val_every_n_epoch=null policy.lr=1e-5
# $cmd_prefix python train.py --config-dir=. --config-name=train_ae_workspace.yaml "hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'" trainer.max_steps=200 policy.warmup_steps=100 trainer.val_check_interval=200 trainer.check_val_every_n_epoch=null policy.lr=2e-5
# $cmd_prefix python train.py --config-dir=. --config-name=train_ae_workspace.yaml "hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'" trainer.max_steps=200 policy.warmup_steps=100 trainer.val_check_interval=200 trainer.check_val_every_n_epoch=null policy.lr=4e-5
# $cmd_prefix python train.py --config-dir=. --config-name=train_ae_workspace.yaml "hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'" trainer.max_steps=200 policy.warmup_steps=100 trainer.val_check_interval=200 trainer.check_val_every_n_epoch=null policy.lr=8e-5
# $cmd_prefix python train.py --config-dir=. --config-name=train_ae_workspace.yaml "hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'" trainer.max_steps=200 policy.warmup_steps=100 trainer.val_check_interval=200 trainer.check_val_every_n_epoch=null policy.lr=1e-4
# $cmd_prefix python train.py --config-dir=. --config-name=train_ae_workspace.yaml "hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'" trainer.max_steps=200 policy.warmup_steps=100 trainer.val_check_interval=200 trainer.check_val_every_n_epoch=null policy.lr=2e-4
# $cmd_prefix python train.py --config-dir=. --config-name=train_ae_workspace.yaml "hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'" trainer.max_steps=200 policy.warmup_steps=100 trainer.val_check_interval=200 trainer.check_val_every_n_epoch=null policy.lr=4e-4
# $cmd_prefix python train.py --config-dir=. --config-name=train_ae_workspace.yaml "hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'" trainer.max_steps=200 policy.warmup_steps=100 trainer.val_check_interval=200 trainer.check_val_every_n_epoch=null policy.lr=8e-4

# run
# $cmd_prefix python train.py --config-dir=. --config-name=train_ae_workspace.yaml "hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'" task=can_lowdim_abs
# $cmd_prefix python train.py --config-dir=. --config-name=train_ae_workspace.yaml "hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'" task=square_lowdim_abs
# $cmd_prefix python train.py --config-dir=. --config-name=train_ae_workspace.yaml "hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'" task=lift_lowdim_abs
# $cmd_prefix python train.py --config-dir=. --config-name=train_ae_workspace.yaml "hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'" task=robomimic_mixed_lowdim datamodule._target_=diffusion_policy.dataset.robomimic_replay_lowdim_dataset.MixedRobomimicLowdimDatamodule trainer.max_steps=30000 trainer.val_check_interval=3000 datamodule.batch_size=256 policy.lr=1e-4

# $cmd_prefix python train.py --config-dir=. --config-name=train_latent_diffusion_workspace.yaml "hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'" task=can_image_abs

# $cmd_prefix python train.py --config-dir=. --config-name=train_latent_diffusion_workspace.yaml "hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'" task=robomimic_mixed_image datamodule._target_=diffusion_policy.dataset.robomimic_replay_image_dataset.MixedRobomimicImageDatamodule trainer.max_steps=2000 policy.warmup_steps=1000 trainer.val_check_interval=2000 datamodule.batch_size=128 policy.lr=3e-4 trainer.callbacks=null
# $cmd_prefix python train.py --config-dir=. --config-name=train_latent_diffusion_workspace.yaml "hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'" task=robomimic_mixed_image datamodule._target_=diffusion_policy.dataset.robomimic_replay_image_dataset.MixedRobomimicImageDatamodule trainer.max_steps=2000 policy.warmup_steps=1000 trainer.val_check_interval=2000 datamodule.batch_size=128 policy.lr=2e-4 trainer.callbacks=null
# $cmd_prefix python train.py --config-dir=. --config-name=train_latent_diffusion_workspace.yaml "hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'" task=robomimic_mixed_image datamodule._target_=diffusion_policy.dataset.robomimic_replay_image_dataset.MixedRobomimicImageDatamodule trainer.max_steps=2000 policy.warmup_steps=1000 trainer.val_check_interval=2000 datamodule.batch_size=128 policy.lr=1e-4 trainer.callbacks=null
$cmd_prefix python train.py --config-dir=. --config-name=train_latent_diffusion_workspace.yaml "hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'" task=robomimic_mixed_image datamodule._target_=diffusion_policy.dataset.robomimic_replay_image_dataset.MixedRobomimicImageDatamodule trainer.max_steps=400000 policy.warmup_steps=1000 trainer.val_check_interval=1000 datamodule.batch_size=128 policy.lr=1e-4


if [ "$is_slurm" = "true" ]; then
    unset WANDB_MODE
fi

unset HYDRA_FULL_ERROR
unset DATA_ROOT
unset ZZY_DEBU

echo "training finished."