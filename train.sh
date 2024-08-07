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

$cmd_prefix python train.py \
    --config-dir=. \
    --config-name=train_latent_diffusion_workspace.yaml \
    "hydra.run.dir='${DATA_ROOT}/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'" \

if [ "$is_slurm" = "true" ]; then
    unset WANDB_MODE
fi

unset HYDRA_FULL_ERROR
unset DATA_ROOT
unset ZZY_DEBUG