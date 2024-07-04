cwd=$(pwd)

cd /home/zzy/robot/robot_zzy/diffusion_policy
conda activate robodiff
export DATA_ROOT=/home/zzy/robot/data/diffusion_policy_data/data
export HYDRA_FULL_ERROR=1

python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='/home/zzy/robot/data/diffusion_policy_data/data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

unset DATA_ROOT
unset HYDRA_FULL_ERROR
conda deactivate
cd ${cwd}


# . /home/zzy/robot/robot_zzy/diffusion_policy/train.sh