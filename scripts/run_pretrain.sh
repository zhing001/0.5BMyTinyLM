#!/bin/bash


train_type=""
gpus="" #1,2,3,4
gpu_num=""
while getopts ":t:g:n:" opt;do 
    case $opt in 
        t) train_type=$OPTARG;;
        g) gpus=$OPTARG;;
        n) gpu_num=$OPTARG;;
    esac
done

# echo $train_type
# echo $gpus
# echo $gpu_num

if [ $train_type = 'ds_pp' ]; then
    deepspeed --include localhost:$gpus --master_port 29501 src/pretrain/pretrain_ds_pp.py \
        pipe_parallel_size=1 gradient_accumulation_steps=16 \
        pretrain_seq_len=1024 per_gpu_train_batch_size=8 \
        run_name=tinylm_pt_v2 learning_rate=5e-4 save_steps=1000 save_total_limit=1 \
        tokenizer.tokenizer_path=/src/Qwen1.5-0.5b \
        log_steps=1 learning_rate=5e-4 num_workers=4 prefetch_factor=4 \
        hydra/job_logging=disabled hydra/hydra_logging=disabled

elif [ $train_type = 'hf' ]; then
    CUDA_VISIBLE_DEVICES=$gpus torchrun --nproc_per_node $gpu_num --master_port 29501 --master_port 29501 src/pretrain/pretrain_hf.py \
        gradient_accumulation_steps=32 \
        pretrain_seq_len=1024 per_gpu_train_batch_size=1 \
        run_name=tinylm_pt_v2 learning_rate=5e-4 save_steps=1000 save_total_limit=1 \
        tokenizer.tokenizer_path=/src/Qwen1.5-0.5b \
        log_steps=1 learning_rate=5e-4 num_workers=1 prefetch_factor=1 \
        hydra/job_logging=disabled hydra/hydra_logging=disabled
fi
