#!/bin/bash
#SBATCH -J rold_train                  # 作业名为 test
#SBATCH -o logs/latent_dp_train_%j.out   # 屏幕上的输出文件重定向到 test.out
#SBATCH -p SC-L40S                  # 
#SBATCH --qos=plus               # 
#SBATCH -N 1                      # 作业申请 1 个节点
#SBATCH --ntasks-per-node=1       # 单节点启动的进程数为 4
#SBATCH --cpus-per-task=32         # 单任务使用的 CPU 核心数为 32
#SBATCH -t 1-00:00:00                # 任务运行的最长时间为 1 day
#SBATCH --gres=gpu:1             # 单个节点使用 1 块 GPU 卡
#SBATCH --mem=64G                  #--mem=<size[units]>：设定每个节点的内存大小，后缀可以为[K|M|G|T]，默认为MB

# nvidia-smi
. train.sh

