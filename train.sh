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

export HYDRA_FULL_ERROR=1
export DATA_ROOT=$data_root
export ZZY_DEBUG=True

if [ "$is_slurm" = "true" ]; then
    srun python train.py \
        --config-dir=. \
        --config-name=train_ae_workspace.yaml \
        "hydra.run.dir='${DATA_ROOT}/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'"
else
    python train.py \
        --config-dir=. \
        --config-name=train_ae_workspace.yaml \
        "hydra.run.dir='${DATA_ROOT}/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'"
fi