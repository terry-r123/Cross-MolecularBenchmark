#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH -p vip_gpu_ailab 
#SBATCH -A ai4agr


module load anaconda/2021.11 cuda/11.7.0 compilers/gcc/11.3.0 cudnn/8.8.1.3_cuda11.x

source activate /home/bingxing2/ailab/scxlab0067/.conda/envs/zhangqianyuan

python downstream/protein_tasks/train_ec.py \
    --cache_dir /home/bingxing2/ailab/group/ai4bio/hanwenwei/huggingface/  \
    --data_path /home/bingxing2/ailab/group/ai4bio/public/multi-omics/protein/downstream/ec/