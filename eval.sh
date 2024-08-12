cwd=$(pwd)
cd /home/zzy/robot/robot_zzy/diffusion_policy
conda activate robodiff
export DATA_ROOT="/home/zzy/robot/data/diffusion_policy_data/data"
export ZZY_DEBUG="True"
export PYOPENGL_PLATFORM="osmesa"
export MUJOCO_GL="osmesa"
export R3M_HOME="/home/zzy/robot/data/.r3m"
python eval.py --checkpoint \
                "/home/zzy/robot/data/diffusion_policy_data/data/outputs/2024.08.07/01.36.07_train_latent_diffusion_can_image/checkpoints/latest.ckpt"  \
                --output_dir \
                "/home/zzy/robot/data/diffusion_policy_data/data/outputs/eval_output" \
                --device \
                "cuda:0" \
                --n_action_steps 16

python eval.py --checkpoint \
                "/home/zzy/robot/data/diffusion_policy_data/data/outputs/2024.08.07/01.36.07_train_latent_diffusion_can_image/checkpoints/latest.ckpt"  \
                --output_dir \
                "/home/zzy/robot/data/diffusion_policy_data/data/outputs/eval_output" \
                --device \
                "cuda:0" \
                --n_action_steps 12

python eval.py --checkpoint \
                "/home/zzy/robot/data/diffusion_policy_data/data/outputs/2024.08.07/01.36.07_train_latent_diffusion_can_image/checkpoints/latest.ckpt"  \
                --output_dir \
                "/home/zzy/robot/data/diffusion_policy_data/data/outputs/eval_output" \
                --device \
                "cuda:0" \
                --n_action_steps 8

python eval.py --checkpoint \
                "/home/zzy/robot/data/diffusion_policy_data/data/outputs/2024.08.07/01.36.07_train_latent_diffusion_can_image/checkpoints/latest.ckpt"  \
                --output_dir \
                "/home/zzy/robot/data/diffusion_policy_data/data/outputs/eval_output" \
                --device \
                "cuda:0" \
                --n_action_steps 4

python eval.py --checkpoint \
                "/home/zzy/robot/data/diffusion_policy_data/data/outputs/2024.08.07/01.36.07_train_latent_diffusion_can_image/checkpoints/latest.ckpt"  \
                --output_dir \
                "/home/zzy/robot/data/diffusion_policy_data/data/outputs/eval_output" \
                --device \
                "cuda:0" \
                --n_action_steps 1

                # "/home/zzy/robot/data/diffusion_policy_data/data/outputs/2024.08.05/20.33.27_train_ae_lift_lowdim/checkpoints/latest.ckpt"
                # "/home/zzy/robot/data/diffusion_policy_data/data/outputs/2024.08.06/22.22.49_train_ae_can_lowdim/checkpoints/latest.ckpt"
                # "/home/zzy/robot/data/diffusion_policy_data/data/outputs/2024.08.06/23.18.52_train_ae_square_lowdim/checkpoints/latest.ckpt"
                # "/home/zzy/robot/data/diffusion_policy_data/data/outputs/2024.08.07/01.36.07_train_latent_diffusion_can_image/checkpoints/latest.ckpt"
unset DATA_ROOT
unset ZZY_DEBUG
unset PYOPENGL_PLATFORM
unset MUJOCO_GL
unset R3M_HOME

cd ${cwd}
conda deactivate

# . /home/zzy/robot/robot_zzy/diffusion_policy/eval.sh