cwd=$(pwd)
cd /home/zzy/robot/robot_zzy/diffusion_policy
conda activate robodiff
export DATA_ROOT=/home/zzy/robot/data/diffusion_policy_data/data

python eval.py --checkpoint /home/zzy/robot/data/diffusion_policy_data/data/experiments/image/lift_ph/diffusion_policy_cnn/train_0/checkpoints/epoch=0300-test_mean_score=1.000.ckpt --output_dir /home/zzy/robot/data/diffusion_policy_data/data/experiments/image/lift_ph/diffusion_policy_cnn/train_0/eval_output --device cuda:0

unset DATA_ROOT
cd ${cwd}
conda deactivate

# . /home/zzy/robot/robot_zzy/diffusion_policy/eval.sh