#!/bin/bash
#SBATCH -J rold_train                  # 作业名为 test
#SBATCH -o logs/rold_train_%j.out               # 屏幕上的输出文件重定向到 test.out
#SBATCH -p DGX                    # 作业提交的分区为 HGX
#SBATCH --qos=lv0b               # 
#SBATCH --account=research         
#SBATCH -N 1                      # 作业申请 1 个节点
#SBATCH --ntasks-per-node=1      # 单节点启动的进程数为 4
#SBATCH --cpus-per-task=16         # 单任务使用的 CPU 核心数为 32
#SBATCH -t 10:00:00                # 任务运行的最长时间为 1 小时
#SBATCH --gres=gpu:1              # 单个节点使用 4 块 GPU 卡
#SBATCH --mem=64G                  #--mem=<size[units]>：设定每个节点的内存大小，后缀可以为[K|M|G|T]，默认为MB

. train.sh

