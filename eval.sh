cwd=$(pwd)
cd /home/zzy/robot/robot_zzy/diffusion_policy
export DATA_ROOT=/home/zzy/robot/data/diffusion_policy_data/data

ckpt_name="epoch=3200-test_mean_score=1.000.ckpt"
python eval.py --checkpoint ${DATA_ROOT}/${ckpt_name} --output_dir ${DATA_ROOT}/can_mh_eval_output --device cuda:0

unset DATA_ROOT
cd ${cwd}

# . /home/zzy/robot/robot_zzy/diffusion_policy/eval.sh