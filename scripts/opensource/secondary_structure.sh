#!/bin/bash

# This is your argument

gpu_device="2"

nproc_per_node=4
partition='ai4multi'
USE_SLURM='2'
task='Secondary_structure_prediction'


export CUDA_LAUNCH_BLOCKING=1
if [ "$USE_SLURM" == "1" ]; then
    EXEC_PREFIX="srun --job-name=${MODEL_TYPE}_${data} --gres=gpu:$nproc_per_node --cpus-per-task=$(($nproc_per_node * 5)) --mem=50G"
elif [ "$USE_SLURM" == "2" ]; then
    quotatype='vip_gpu_ailab'
    run_type='sbatch'
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
    home_root=/home/bingxing2/ailab/group/ai4bio/
    
else
    data_root=/mnt/data/oss_beijing/   
    EXEC_PREFIX="env CUDA_VISIBLE_DEVICES=$gpu_device torchrun --num_processes=$nproc_per_node --main_process_port=$master_port --mixed_precision=fp16"
fi
DATA_PATH=${data_root}multi-omics/RNA/downstream/${task}/esm_data/bpRNA


# MODEL_TYPE='rna-fm'


# token='single'
# pos='ape'

# batch_size=1
# model_max_length=1024
# lr=3e-5
# data=''

# MODEL_PATH=${data_root}multi-omics/RNA/model/opensource/${MODEL_TYPE}
# OUTPUT_PATH=./outputs/ft/rna-all/${task}/rna/opensource/${MODEL_TYPE}  


        
# for seed in 42 666 3407
# do

#     master_port=$(shuf -i 10000-45000 -n 1)
#     echo "Using port $master_port for communication."
#     EXEC_PREFIX="${run_type} --nodes=1 -p ${quotatype} -A ${partition} --job-name=${MODEL_TYPE}_${task}  --gres=gpu:$nproc_per_node --gpus=$nproc_per_node --cpus-per-task=4 --output=${MODEL_TYPE}_${task}_${seed}_${lr}.out accelerate launch --num_processes=$nproc_per_node --main_process_port=$master_port --mixed_precision=fp16"           
#     echo ${MODEL_PATH}

#     ${EXEC_PREFIX} \
#     downstream/train_secondary_structure.py \
#         --model_name_or_path ${MODEL_PATH} \
#         --data_path  ${DATA_PATH}/${data} \
#         --run_name ${MODEL_TYPE}_${data}_seed${seed} \
#         --output_dir ${OUTPUT_PATH}/${seed}/${lr} \
#         --model_max_length ${model_max_length} \
#         --per_device_train_batch_size ${batch_size} \
#         --per_device_eval_batch_size 1 \
#         --gradient_accumulation_steps 8 \
#         --lr ${lr} \
#         --num_epochs 100 \
#         --patience 60 \
#         --num_workers 1 \
#         --token_type ${token} \
#         --model_type ${MODEL_TYPE} \
#         --seed ${seed} \

# done

# MODEL_TYPE='rnabert'


# token='single'
# pos='ape'

# batch_size=1
# model_max_length=440
# lr=5e-4
# data=''

# MODEL_PATH=${data_root}multi-omics/RNA/model/opensource/${MODEL_TYPE}
# OUTPUT_PATH=./outputs/ft/rna-all/${task}/rna/opensource/${MODEL_TYPE}  


        
# for seed in 42 666 3407
# do

#     master_port=$(shuf -i 10000-45000 -n 1)
#     echo "Using port $master_port for communication."
#     EXEC_PREFIX="${run_type} --nodes=1 -p ${quotatype} -A ${partition} --job-name=${MODEL_TYPE}_${task}  --gres=gpu:$nproc_per_node --gpus=$nproc_per_node --cpus-per-task=4 --output=${MODEL_TYPE}_${task}_${seed}_${lr}.out accelerate launch --num_processes=$nproc_per_node --main_process_port=$master_port --mixed_precision=fp16"           
#     echo ${MODEL_PATH}

#     ${EXEC_PREFIX} \
#     downstream/train_secondary_structure.py \
#         --model_name_or_path ${MODEL_PATH} \
#         --data_path  ${DATA_PATH}/${data} \
#         --run_name ${MODEL_TYPE}_${data}_seed${seed} \
#         --output_dir ${OUTPUT_PATH}/${seed}/${lr} \
#         --model_max_length ${model_max_length} \
#         --per_device_train_batch_size ${batch_size} \
#         --per_device_eval_batch_size 1 \
#         --gradient_accumulation_steps 8 \
#         --lr ${lr} \
#         --num_epochs 100 \
#         --patience 60 \
#         --num_workers 1 \
#         --token_type ${token} \
#         --model_type ${MODEL_TYPE} \
#         --seed ${seed} \
 
   
# done

# MODEL_TYPE='rnamsm'


# token='single'
# pos='ape'

# batch_size=1
# model_max_length=1024
# lr=3e-5
# data=''
# MODEL_PATH=${data_root}multi-omics/RNA/model/opensource/${MODEL_TYPE}
# OUTPUT_PATH=./outputs/ft/rna-all/${task}/rna/opensource/${MODEL_TYPE}  


        
# for seed in 42 666 3407
# do

#     master_port=$(shuf -i 10000-45000 -n 1)
#     echo "Using port $master_port for communication."
#     EXEC_PREFIX="${run_type} --nodes=1 -p ${quotatype} -A ${partition} --job-name=${MODEL_TYPE}_${task}  --gres=gpu:$nproc_per_node --gpus=$nproc_per_node --cpus-per-task=4 --output=${MODEL_TYPE}_${task}_${seed}_${lr}.out accelerate launch --num_processes=$nproc_per_node --main_process_port=$master_port --mixed_precision=fp16"           
#     echo ${MODEL_PATH}

#     ${EXEC_PREFIX} \
#     downstream/train_secondary_structure.py \
#         --model_name_or_path ${MODEL_PATH} \
#         --data_path  ${DATA_PATH}/${data} \
#         --run_name ${MODEL_TYPE}_${data}_seed${seed} \
#         --output_dir ${OUTPUT_PATH}/${seed}/${lr} \
#         --model_max_length ${model_max_length} \
#         --per_device_train_batch_size ${batch_size} \
#         --per_device_eval_batch_size 1 \
#         --gradient_accumulation_steps 8 \
#         --lr ${lr} \
#         --num_epochs 100 \
#         --patience 60 \
#         --num_workers 1 \
#         --token_type ${token} \
#         --model_type ${MODEL_TYPE} \
#         --seed ${seed} \
 
   
# done

# MODEL_TYPE='splicebert-human510'


# token='single'
# pos='ape'

# batch_size=1
# model_max_length=510
# #lr=5e-4
# data=''

# MODEL_PATH=${data_root}multi-omics/RNA/model/opensource/${MODEL_TYPE}
# OUTPUT_PATH=./outputs/ft/rna-all/${task}/rna/opensource/${MODEL_TYPE}  


# for lr in 3e-4 1e-4 1e-3
# do       

#     for seed in 3407 42 666 
#     do

#         master_port=$(shuf -i 10000-45000 -n 1)
#         echo "Using port $master_port for communication."
#         EXEC_PREFIX="${run_type} --nodes=1 -p ${quotatype} -A ${partition} --job-name=${MODEL_TYPE}_${task}  --gres=gpu:$nproc_per_node --gpus=$nproc_per_node --cpus-per-task=4 --output=${MODEL_TYPE}_${task}_${seed}_${lr}.out accelerate launch --num_processes=$nproc_per_node --main_process_port=$master_port --mixed_precision=fp16"           
#         echo ${MODEL_PATH}

#         ${EXEC_PREFIX} \
#         downstream/train_secondary_structure.py \
#             --model_name_or_path ${MODEL_PATH} \
#             --data_path  ${DATA_PATH}/${data} \
#             --run_name ${MODEL_TYPE}_${data}_seed${seed} \
#             --output_dir ${OUTPUT_PATH}/${seed}/${lr} \
#             --model_max_length ${model_max_length} \
#             --per_device_train_batch_size ${batch_size} \
#             --per_device_eval_batch_size 1 \
#             --gradient_accumulation_steps 8 \
#             --lr ${lr} \
#             --num_epochs 100 \
#             --patience 60 \
#             --num_workers 1 \
#             --token_type ${token} \
#             --model_type ${MODEL_TYPE} \
#             --seed ${seed} \
    
#     done
   
# done

# MODEL_TYPE='splicebert-ms510'

# token='single'
# pos='ape'

# batch_size=1
# model_max_length=510
# #lr=5e-4
# data=''
# MODEL_PATH=${data_root}multi-omics/RNA/model/opensource/${MODEL_TYPE}
# OUTPUT_PATH=./outputs/ft/rna-all/${task}/rna/opensource/${MODEL_TYPE}  


        
# for lr in 3e-4 1e-4 1e-3
# do       

#     for seed in 3407 42 666 
#     do

#         master_port=$(shuf -i 10000-45000 -n 1)
#         echo "Using port $master_port for communication."
#         EXEC_PREFIX="${run_type} --nodes=1 -p ${quotatype} -A ${partition} --job-name=${MODEL_TYPE}_${task}  --gres=gpu:$nproc_per_node --gpus=$nproc_per_node --cpus-per-task=4 --output=${MODEL_TYPE}_${task}_${seed}_${lr}.out accelerate launch --num_processes=$nproc_per_node --main_process_port=$master_port --mixed_precision=fp16"           
#         echo ${MODEL_PATH}

#         ${EXEC_PREFIX} \
#         downstream/train_secondary_structure.py \
#             --model_name_or_path ${MODEL_PATH} \
#             --data_path  ${DATA_PATH}/${data} \
#             --run_name ${MODEL_TYPE}_${data}_seed${seed} \
#             --output_dir ${OUTPUT_PATH}/${seed}/${lr} \
#             --model_max_length ${model_max_length} \
#             --per_device_train_batch_size ${batch_size} \
#             --per_device_eval_batch_size 1 \
#             --gradient_accumulation_steps 8 \
#             --lr ${lr} \
#             --num_epochs 100 \
#             --patience 60 \
#             --num_workers 1 \
#             --token_type ${token} \
#             --model_type ${MODEL_TYPE} \
#             --seed ${seed} \
    
#     done
   
# done

# MODEL_TYPE='splicebert-ms1024'


# token='single'
# pos='ape'

# batch_size=1
# model_max_length=1024
# #lr=5e-4
# data=''

# MODEL_PATH=${data_root}multi-omics/RNA/model/opensource/${MODEL_TYPE}
# OUTPUT_PATH=./outputs/ft/rna-all/${task}/rna/opensource/${MODEL_TYPE}  


        
# for lr in 3e-4 1e-4 1e-3
# do       

#     for seed in 3407 42 666 
#     do

#         master_port=$(shuf -i 10000-45000 -n 1)
#         echo "Using port $master_port for communication."
#         EXEC_PREFIX="${run_type} --nodes=1 -p ${quotatype} -A ${partition} --job-name=${MODEL_TYPE}_${task}  --gres=gpu:$nproc_per_node --gpus=$nproc_per_node --cpus-per-task=4 --output=${MODEL_TYPE}_${task}_${seed}_${lr}.out accelerate launch --num_processes=$nproc_per_node --main_process_port=$master_port --mixed_precision=fp16"           
#         echo ${MODEL_PATH}

#         ${EXEC_PREFIX} \
#         downstream/train_secondary_structure.py \
#             --model_name_or_path ${MODEL_PATH} \
#             --data_path  ${DATA_PATH}/${data} \
#             --run_name ${MODEL_TYPE}_${data}_seed${seed} \
#             --output_dir ${OUTPUT_PATH}/${seed}/${lr} \
#             --model_max_length ${model_max_length} \
#             --per_device_train_batch_size ${batch_size} \
#             --per_device_eval_batch_size 1 \
#             --gradient_accumulation_steps 8 \
#             --lr ${lr} \
#             --num_epochs 100 \
#             --patience 60 \
#             --num_workers 1 \
#             --token_type ${token} \
#             --model_type ${MODEL_TYPE} \
#             --seed ${seed} \
    
#     done
   
# done

# MODEL_TYPE='utrbert-3mer'


# token='3mer'
# pos='ape'

# batch_size=1
# model_max_length=512
# lr=3e-5
# data=''

# MODEL_PATH=${data_root}multi-omics/RNA/model/opensource/${MODEL_TYPE}
# OUTPUT_PATH=./outputs/ft/rna-all/${task}/rna/opensource/${MODEL_TYPE}  


        
# for seed in 666 42 3407
# do

#     master_port=$(shuf -i 10000-45000 -n 1)
#     echo "Using port $master_port for communication."
#     EXEC_PREFIX="${run_type} --nodes=1 -p ${quotatype} -A ${partition} --job-name=${MODEL_TYPE}_${task}  --gres=gpu:$nproc_per_node --gpus=$nproc_per_node --cpus-per-task=4 --output=${MODEL_TYPE}_${task}_${seed}_${lr}.out accelerate launch --num_processes=$nproc_per_node --main_process_port=$master_port --mixed_precision=fp16"           
#     echo ${MODEL_PATH}

#     ${EXEC_PREFIX} \
#     downstream/train_secondary_structure.py \
#         --model_name_or_path ${MODEL_PATH} \
#         --data_path  ${DATA_PATH}/${data} \
#         --run_name ${MODEL_TYPE}_${data}_seed${seed} \
#         --output_dir ${OUTPUT_PATH}/${seed}/${lr} \
#         --model_max_length ${model_max_length} \
#         --per_device_train_batch_size ${batch_size} \
#         --per_device_eval_batch_size 1 \
#         --gradient_accumulation_steps 8 \
#         --lr ${lr} \
#         --num_epochs 100 \
#         --patience 60 \
#         --num_workers 1 \
#         --token_type ${token} \
#         --model_type ${MODEL_TYPE} \
#         --seed ${seed} \
 
   
# done

# MODEL_TYPE='utrbert-4mer'


# token='4mer'
# pos='ape'

# batch_size=1
# model_max_length=512
# lr=3e-5
# data=''

# MODEL_PATH=${data_root}multi-omics/RNA/model/opensource/${MODEL_TYPE}
# OUTPUT_PATH=./outputs/ft/rna-all/${task}/rna/opensource/${MODEL_TYPE}  


        
# for seed in 666 3407 42 
# do

#     master_port=$(shuf -i 10000-45000 -n 1)
#     echo "Using port $master_port for communication."
#     EXEC_PREFIX="${run_type} --nodes=1 -p ${quotatype} -A ${partition} --job-name=${MODEL_TYPE}_${task}  --gres=gpu:$nproc_per_node --gpus=$nproc_per_node --cpus-per-task=4 --output=${MODEL_TYPE}_${task}_${seed}_${lr}.out accelerate launch --num_processes=$nproc_per_node --main_process_port=$master_port --mixed_precision=fp16"           
#     echo ${MODEL_PATH}

#     ${EXEC_PREFIX} \
#     downstream/train_secondary_structure.py \
#         --model_name_or_path ${MODEL_PATH} \
#         --data_path  ${DATA_PATH}/${data} \
#         --run_name ${MODEL_TYPE}_${data}_seed${seed} \
#         --output_dir ${OUTPUT_PATH}/${seed}/${lr} \
#         --model_max_length ${model_max_length} \
#         --per_device_train_batch_size ${batch_size} \
#         --per_device_eval_batch_size 1 \
#         --gradient_accumulation_steps 8 \
#         --lr ${lr} \
#         --num_epochs 100 \
#         --patience 60 \
#         --num_workers 1 \
#         --token_type ${token} \
#         --model_type ${MODEL_TYPE} \
#         --seed ${seed} \
 
   
# done

MODEL_TYPE='utrbert-5mer'


token='5mer'
pos='ape'

batch_size=1
model_max_length=512
#lr=3e-5
data=''

MODEL_PATH=${data_root}multi-omics/RNA/model/opensource/${MODEL_TYPE}
OUTPUT_PATH=./outputs/ft/rna-all/${task}/rna/opensource/${MODEL_TYPE}  

for lr in 1e-5 5e-5 1e-4
do
        
    for seed in 42 666 # 3407 
    do

        master_port=$(shuf -i 10000-45000 -n 1)
        echo "Using port $master_port for communication."
        EXEC_PREFIX="${run_type} --nodes=1 -p ${quotatype} -A ${partition} --job-name=${MODEL_TYPE}_${task}  --gres=gpu:$nproc_per_node --gpus=$nproc_per_node --cpus-per-task=4 --output=${MODEL_TYPE}_${task}_${seed}_${lr}.out accelerate launch --num_processes=$nproc_per_node --main_process_port=$master_port --mixed_precision=fp16"           
        echo ${MODEL_PATH}

        ${EXEC_PREFIX} \
        downstream/train_secondary_structure.py \
            --model_name_or_path ${MODEL_PATH} \
            --data_path  ${DATA_PATH}/${data} \
            --run_name ${MODEL_TYPE}_${data}_seed${seed}_${lr} \
            --output_dir ${OUTPUT_PATH}/${seed}/${lr}/${lr} \
            --model_max_length ${model_max_length} \
            --per_device_train_batch_size ${batch_size} \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps 2 \
            --lr ${lr} \
            --num_epochs 100 \
            --patience 60 \
            --num_workers 1 \
            --token_type ${token} \
            --model_type ${MODEL_TYPE} \
            --seed ${seed} \
 
    done
done

# MODEL_TYPE='utrbert-6mer'


# token='6mer'
# pos='ape'

# batch_size=1
# model_max_length=512
# lr=3e-5
# data=''

# MODEL_PATH=${data_root}multi-omics/RNA/model/opensource/${MODEL_TYPE}
# OUTPUT_PATH=./outputs/ft/rna-all/${task}/rna/opensource/${MODEL_TYPE}  


        
# for seed in 42 666 3407
# do

#     master_port=$(shuf -i 10000-45000 -n 1)
#     echo "Using port $master_port for communication."
#     EXEC_PREFIX="${run_type} --nodes=1 -p ${quotatype} -A ${partition} --job-name=${MODEL_TYPE}_${task}  --gres=gpu:$nproc_per_node --gpus=$nproc_per_node --cpus-per-task=4 --output=${MODEL_TYPE}_${task}_${seed}_${lr}.out accelerate launch --num_processes=$nproc_per_node --main_process_port=$master_port --mixed_precision=fp16"           
#     echo ${MODEL_PATH}

#     ${EXEC_PREFIX} \
#     downstream/train_secondary_structure.py \
#         --model_name_or_path ${MODEL_PATH} \
#         --data_path  ${DATA_PATH}/${data} \
#         --run_name ${MODEL_TYPE}_${data}_seed${seed} \
#         --output_dir ${OUTPUT_PATH}/${seed}/${lr} \
#         --model_max_length ${model_max_length} \
#         --per_device_train_batch_size ${batch_size} \
#         --per_device_eval_batch_size 1 \
#         --gradient_accumulation_steps 8 \
#         --lr ${lr} \
#         --num_epochs 100 \
#         --patience 60 \
#         --num_workers 1 \
#         --token_type ${token} \
#         --model_type ${MODEL_TYPE} \
#         --seed ${seed} \
 
   
# done

# MODEL_TYPE='utr-lm-mrl'


# token='single'
# pos='rope'

# batch_size=1
# model_max_length=1026
# lr=5e-3
# data=''

# MODEL_PATH=${data_root}multi-omics/RNA/model/opensource/${MODEL_TYPE}
# OUTPUT_PATH=./outputs/ft/rna-all/${task}/rna/opensource/${MODEL_TYPE}  


        
# for seed in 666 3407 42 
# do

#     master_port=$(shuf -i 10000-45000 -n 1)
#     echo "Using port $master_port for communication."
#     EXEC_PREFIX="${run_type} --nodes=1 -p ${quotatype} -A ${partition} --job-name=${MODEL_TYPE}_${task}  --gres=gpu:$nproc_per_node --gpus=$nproc_per_node --cpus-per-task=4 --output=${MODEL_TYPE}_${task}_${seed}_${lr}.out accelerate launch --num_processes=$nproc_per_node --main_process_port=$master_port --mixed_precision=fp16"           
#     echo ${MODEL_PATH}

#     ${EXEC_PREFIX} \
#     downstream/train_secondary_structure.py \
#         --model_name_or_path ${MODEL_PATH} \
#         --data_path  ${DATA_PATH}/${data} \
#         --run_name ${MODEL_TYPE}_${data}_seed${seed} \
#         --output_dir ${OUTPUT_PATH}/${seed}/${lr} \
#         --model_max_length ${model_max_length} \
#         --per_device_train_batch_size ${batch_size} \
#         --per_device_eval_batch_size 1 \
#         --gradient_accumulation_steps 8 \
#         --lr ${lr} \
#         --num_epochs 100 \
#         --patience 60 \
#         --num_workers 1 \
#         --token_type ${token} \
#         --model_type ${MODEL_TYPE} \
#         --seed ${seed} \
 
   
# done

# MODEL_TYPE='utr-lm-te-el'

# token='single'
# pos='rope'

# batch_size=1
# model_max_length=1026
# lr=5e-3
# data=''

# MODEL_PATH=${data_root}multi-omics/RNA/model/opensource/${MODEL_TYPE}
# OUTPUT_PATH=./outputs/ft/rna-all/${task}/rna/opensource/${MODEL_TYPE}  


        
# for seed in 42 666 3407
# do

#     master_port=$(shuf -i 10000-45000 -n 1)
#     echo "Using port $master_port for communication."
#     EXEC_PREFIX="${run_type} --nodes=1 -p ${quotatype} -A ${partition} --job-name=${MODEL_TYPE}_${task}  --gres=gpu:$nproc_per_node --gpus=$nproc_per_node --cpus-per-task=4 --output=${MODEL_TYPE}_${task}_${seed}_${lr}.out accelerate launch --num_processes=$nproc_per_node --main_process_port=$master_port --mixed_precision=fp16"           
#     echo ${MODEL_PATH}

#     ${EXEC_PREFIX} \
#     downstream/train_secondary_structure.py \
#         --model_name_or_path ${MODEL_PATH} \
#         --data_path  ${DATA_PATH}/${data} \
#         --run_name ${MODEL_TYPE}_${data}_seed${seed} \
#         --output_dir ${OUTPUT_PATH}/${seed}/${lr} \
#         --model_max_length ${model_max_length} \
#         --per_device_train_batch_size ${batch_size} \
#         --per_device_eval_batch_size 1 \
#         --gradient_accumulation_steps 8 \
#         --lr ${lr} \
#         --num_epochs 100 \
#         --patience 60 \
#         --num_workers 1 \
#         --token_type ${token} \
#         --model_type ${MODEL_TYPE} \
#         --seed ${seed} \
 
   
# done
