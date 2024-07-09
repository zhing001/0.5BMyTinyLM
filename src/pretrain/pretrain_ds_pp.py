
import glob
from pathlib import Path
import shutil
import time
import random
import warnings
from dataclasses import dataclass, field
from typing import Optional, Literal

import torch
import transformers
from transformers import Qwen2Config
import numpy as np
import deepspeed
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader, RandomSampler, SequentialSampler

import os
import hydra
from omegaconf import DictConfig, OmegaConf

from accelerate.utils import set_seed
import sys 
sys.path.append(str('.'))

from tqdm import tqdm

from src.common.logger import logger_rank0 
from loguru import logger

import wandb
from src.pretrain.data_collator import DataCollatorForSupervisedDataset
from src.data_utils.load_pt_data import load_pretrain_data
from src.pretrain.qwen_pipeline_model import get_model
from transformers.trainer_pt_utils import torch_distributed_zero_first


wandb.disabled = True
warnings.filterwarnings("ignore")

def save_model(engine, output_dir, args, tokenizer):
    engine.save_checkpoint(output_dir)
    
    if args.local_rank not in [-1, 0]:
        dist.barrier()
        
    if args.local_rank in [-1, 0]:
        tokenizer.save_pretrained(output_dir)

        OmegaConf.save(args, os.path.join(output_dir, "config.yaml"))

        if args.local_rank == 0:
            dist.barrier()


@hydra.main(config_path="../../conf", config_name="tinylm0.5b_pretrain", version_base="1.1")
def main(args: DictConfig):
    if not Path(args.output_dir).exists():
        os.makedirs(args.output_dir)
    
    deepspeed.init_distributed(dist_backend="nccl")
    args.world_size = torch.distributed.get_world_size()

    args.pp_num = args.pipe_parallel_size
    args.mp_num = args.model_parallel_size
    args.dp_num = args.world_size // args.pipe_parallel_size
    ds_config = args.deepspeed_config

    set_seed(args.seed)   
    deepspeed.runtime.utils.set_random_seed(args.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer.tokenizer_path,
        model_max_length=args.pretrain_seq_len,
        padding_side="right",
        use_fast=False,
    )
    
    model_config = Qwen2Config.from_dict(args.model_argumnts)
        
    # partition_method="type:ParallelTransformerLayerPipe" or "uniform" or "parameters"
    model = get_model(model_config, args, partition_method="uniform", activation_checkpointing=True)

    train_dataset = load_pretrain_data()
    
    total_trainig_steps = len(train_dataset) // args.dp_num // args.gradient_accumulation_steps // args.per_gpu_train_batch_size
    
    ds_config.scheduler.params.warmup_num_steps = total_trainig_steps // 100
    ds_config.scheduler.params.total_num_steps = total_trainig_steps
    
    logger_rank0.info(f'Total training steps: {total_trainig_steps}')
    logger_rank0.info(f'Warmup steps setting to: {total_trainig_steps // 100}')
    ds_config = OmegaConf.to_container(ds_config, resolve=True)
    
    engine, _, _, _ = deepspeed.initialize(
        config=ds_config,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
    )
    
    resume_step = -1
    if args.resume_from_checkpoint is not None:
        ckpt_path = args.resume_from_checkpoint
        resume_step = int(ckpt_path.split('-')[-1])
        
        engine.load_checkpoint(ckpt_path)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            ckpt_path,
            model_max_length=args.pretrain_seq_len,
            padding_side="right",
            use_fast=False,
        )
        logger_rank0.info(f'Resume from checkpoint: {args.resume_from_checkpoint} resume step:{resume_step}')

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, args=args)
    
    logger_rank0.info(f'Data parallel number: {args.dp_num}')
    logger_rank0.info(f'{args.local_rank}:{model._grid.get_data_parallel_id()}')
    
    data_sampler = DistributedSampler(train_dataset, num_replicas=engine.dp_world_size,
                    rank=model._grid.get_data_parallel_id())
    data_loader = DataLoader(dataset=train_dataset,
                                    sampler=data_sampler,
                                    batch_size=args.per_gpu_train_batch_size,
                                    collate_fn=data_collator,
                                    num_workers=args.num_workers,
                                    pin_memory=False,
                                    prefetch_factor=args.prefetch_factor,
                                    drop_last=False,
                                    )
    # epoch_update_steps = len(data_loader) // args.gradient_accumulation_steps
    epoch_update_steps = total_trainig_steps
    train_dataloader = iter(deepspeed.utils.RepeatingLoader(data_loader))
    
    start = time.time()
    bar = tqdm(range(1, epoch_update_steps + 1), disable=args.local_rank not in [-1, 0], dynamic_ncols=True)
    
    global_step = 0
    for step in bar:
        if global_step < resume_step:
            for _ in range(args.gradient_accumulation_steps):
                next(train_dataloader)
            global_step += 1
            continue

        loss = engine.train_batch(data_iter=train_dataloader)
        global_step += 1
        
        if args.local_rank == 0:
            if step % args.log_steps == 0:
                now = time.time()
                avg_time = (now-start) / args.log_steps
                bar.set_description(f"Step={step:>2}, loss={loss.item():.2f}, {avg_time:.2f} it/s")
                start = now

        if step % args.eval_steps == 0:
            # 节省时间，预训练不做eval
            pass
        
        if args.save_steps > 0 and global_step % args.save_steps == 0:
            output_dir = f'{args.output_dir}/checkpoint-{global_step}'
            logger_rank0.info(f"Saving at global step: {global_step} to {output_dir}")
            save_model(engine, output_dir, args, tokenizer)
            
            with torch_distributed_zero_first(args.local_rank):
                cur_saved_dir = glob.glob(os.path.join(args.output_dir, 'checkpoint*'))
                cur_saved_dir.sort(key=lambda x:int(x.split('-')[-1]), reverse=False)
                if len(cur_saved_dir) > args.save_total_limit:
                    if os.path.exists(cur_saved_dir[0]) and os.path.isdir(cur_saved_dir[0]):
                        shutil.rmtree(cur_saved_dir[0])
    
    output_dir = f'{args.output_dir}/final'
    save_model(engine, output_dir, args, tokenizer)


if __name__ == "__main__":
    hydra_formatted_args = []
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--"):])
        else:
            hydra_formatted_args.append(arg)
    sys.argv = hydra_formatted_args
    main()
