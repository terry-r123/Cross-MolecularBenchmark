#!/bin/bash

# This is your argument

# 基础环境设置
gpu_device="2"
master_port=41611
nproc_per_node=1
USE_SLURM='2'
partition='ai4bio'
quotatype='vip_gpu_ailab' # vip_gpu_ailab_low
run_type='sbatch' #choice in [srun, sbatch]



task='rna-rna'
task_type='multi'
data_file_train=train.csv; data_file_val=val.csv; data_file_test=test.csv

export CUDA_LAUNCH_BLOCKING=1
if [ "$USE_SLURM" == "1" ]; then
    EXEC_PREFIX="srun --job-name=${MODEL_TYPE}_${data} --gres=gpu:$nproc_per_node --cpus-per-task=$(($nproc_per_node * 5)) --mem=50G"
elif [ "$USE_SLURM" == "2" ]; then
    module load anaconda/2021.11
    module load cuda/11.7.0
    module load cudnn/8.6.0.163_cuda11.x
    module load compilers/gcc/9.3.0
    module load llvm/triton-clang_llvm-11.0.1
    source activate dnalm_v2
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/peng_tmp_test/miniconda3/lib
    export CPATH=/usr/include:$CPATH
    export PYTHONUNBUFFERED=1
    export LD_PRELOAD=/home/bingxing2/apps/compilers/gcc/12.2.0/lib64/libstdc++.so.6
    data_root=/home/bingxing2/ailab/group/ai4bio/public/
    model_root=/home/bingxing2/ailab/group/ai4bio/
    
else
    data_root=/mnt/data/oss_beijing/   
    EXEC_PREFIX="env CUDA_VISIBLE_DEVICES=$gpu_device torchrun --num_processes=$nproc_per_node --main_process_port=$master_port"
fi
DATA_PATH=${data_root}multi-omics/multi-omics/downstream/sirnaEfficiency

omics1_model_type='beacon-b'
omics2_model_type='beacon-b'

omics1_token='single'
omics2_token='single'

omics1_pos='alibi'
omics2_pos='alibi'

batch_size=16
gradient_accumulation=2
omics1_model_max_length=30
omics2_model_max_length=1024
lr=1e-5
data=''

OMICS1_MODEL_PATH=${data_root}multi-omics/RNA/model/ours/BEACON-B
OMICS2_MODEL_PATH=${data_root}multi-omics/RNA/model/ours/BEACON-B




        
for seed in 42
do
    for lr in  3e-6 #1e-5 #3e-5 5e-5 #7e-5 9e-6 7e-6 5e-6
    do
        out_name=${task_type}/${task}/opensource/${omics1_model_type}_${omics2_model_type}_${task}_${seed}_${lr}
        OUTPUT_PATH=./outputs/ft/${out_name}
        master_port=$(shuf -i 10000-45000 -n 1)
        echo "Using port $master_port for communication."
        if [ "$run_type" == "srun" ]; then
            EXEC_PREFIX="${run_type} --nodes=1 -p ${quotatype} -A ${partition} --job-name=${omics1_model_type}_${omics2_model_type}_${task}  --gres=gpu:$nproc_per_node --cpus-per-task=4  torchrun --nproc_per_node=$nproc_per_node --master_port=$master_port"      
        else
            EXEC_PREFIX="${run_type} --nodes=1 -p ${quotatype} -A ${partition} --job-name=${omics1_model_type}_${omics2_model_type}_${task}  --gres=gpu:$nproc_per_node --cpus-per-task=4 --output=logging/${out_name}.out torchrun --nproc_per_node=$nproc_per_node --master_port=$master_port"
        fi
        echo ${MODEL_PATH}

        ${EXEC_PREFIX} \
        downstream/train_sirna_mrna_inter.py \
            --omics1_model_name_or_path ${OMICS1_MODEL_PATH} \
            --omics2_model_name_or_path ${OMICS2_MODEL_PATH} \
            --data_path  ${DATA_PATH}/${data} \
            --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test}   \
            --run_name ${MODEL_TYPE}_${data}_seed${seed} \
            --omics1_model_max_length ${omics1_model_max_length} \
            --omics2_model_max_length ${omics2_model_max_length} \
            --per_device_train_batch_size ${batch_size} \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps ${gradient_accumulation} \
            --learning_rate ${lr} \
            --num_train_epochs 30 \
            --fp16 \
            --save_steps 400 \
            --output_dir ${OUTPUT_PATH}/${data} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 200 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --omics1_token_type ${omics1_token} \
            --omics2_token_type ${omics2_token} \
            --omics1_model_type ${omics1_model_type} \
            --omics2_model_type ${omics2_model_type} \
            --use_features \
 
    done
done