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

if [ $train_type = 'ds_pp' ]; then
CUDA_VISIBLE_DEVICES=$gpus torchrun --nproc_per_node $gpu_num \
            src/finetune/finetune_hf.py per_gpu_train_batch_size=2 \
            gradient_accumulation_steps=32 run_name=tinylm_finetune_v5_all_aug \
            model_init_path=outputs/tinylm_pt_v2/hf learning_rate=5e-5 num_train_epochs=3 \
            finetune_data_name=finetune_final hydra/job_logging=disabled hydra/hydra_logging=disabled

elif [ $train_type = 'hf' ]; then
# longlora for long context
CUDA_VISIBLE_DEVICES=$gpus torchrun --nproc_per_node $gpu_num \
            src/finetune/finetune_hf.py per_gpu_train_batch_size=1 \
            gradient_accumulation_steps=32 run_name=tinylm_finetune_v5_all_aug \
            model_init_path=outputs/tinylm_pt_v2/hf learning_rate=5e-5 num_train_epochs=1 \
            finetune_data_name=finetune_long hydra/job_logging=disabled hydra/hydra_logging=disabled \
            train_long_context=true low_rank_training=true finetune_seq_len=8192
fi