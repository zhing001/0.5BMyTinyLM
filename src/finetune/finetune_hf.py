import copy
import os 
import json
import sys

sys.path.append(str('.'))

import datasets 
from typing import Sequence, Dict, Union
from dataclasses import dataclass, field

import hydra
from src.data_utils.load_sft_data import load_finetune_long_data, load_belle
from src.finetune.data_loader import DataCollatorForSupervisedDataset
import transformers
from transformers import (Qwen2Tokenizer, Qwen2Config, AutoModel, AutoTokenizer, TrainingArguments,Qwen2ForCausalLM,
                          Trainer)

import torch 
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler, Dataset
from accelerate.utils import set_seed
from omegaconf import DictConfig
from src.common.logger import logger_rank0
from loguru import logger

import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from src.finetune.replace_longlora_attn import replace_qwen_attn
from src.common.callbacks import SavedPerEpochCallback
from transformers.trainer_pt_utils import torch_distributed_zero_first


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

class MyTrainer(Trainer):
    def __init__(self, model, args, tokenizer, data_collator, train_dataset, **kwargs):
        super().__init__(model=model,
                         args=args,
                         data_collator=data_collator,
                         tokenizer=tokenizer,
                         train_dataset=train_dataset,
                         **kwargs)
        
    def get_train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset)
        data_loader = DataLoader(self.train_dataset, 
                                 sampler=sampler,
                                 batch_size=self.args.per_device_train_batch_size,
                                 collate_fn=self.data_collator)
        return data_loader
    

@hydra.main(config_path="../../conf", config_name="tinylm0.5b_finetune")
def main(args: DictConfig):
    logger_rank0.info('='*100)
    # logger.info(torch.distributed.get_rank())
    training_args = TrainingArguments(**args.training_arguments)
    set_seed(args.seed)
    
    args.local_rank = os.environ["LOCAL_RANK"]
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer.tokenizer_path,
        model_max_length=args.finetune_seq_len,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )

    logger_rank0.info("Load tokenizer from {} over.".format(args.tokenizer.tokenizer_path))
    
    if args.longlora_attention:
        replace_qwen_attn()
    
    model_config = Qwen2Config.from_dict(args.model_argumnts)
    model = Qwen2ForCausalLM.from_pretrained(args.model_init_path,
                                             config=model_config,
                                             torch_dtype=torch.bfloat16)
    
    # breakpoint()
    logger_rank0.info("Load model from {} over.".format(args.model_init_path))

    if args.train_long_context:
        train_dataset = load_finetune_long_data(tokenizer, data_saved_name=args.finetune_data_name)    
    else:
        # train_dataset = load_finetune_short_data()
        train_dataset = load_belle(tokenizer, data_saved_name=args.finetune_data_name)
    
    logger_rank0.info(f"length of instruction tune set: {len(train_dataset)}.")
    logger_rank0.info("sucessfully loading datasets")
    logger.info(args.local_rank)
        
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    

    if args.low_rank_training:
        targets=["q_proj", "k_proj", "v_proj", "o_proj"]

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=targets,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        model.enable_input_require_grads()
        # enable trainable params
        
        # TODO: embeddings and norm could be trainable
        # [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")])]

    data_module=dict(train_dataset=train_dataset,
                     eval_dataset=None, data_collator=data_collator)
    
    
    # trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module,
    #                   callbacks=[SavedPerEpochCallback()])
    
    trainer = MyTrainer(model, training_args, tokenizer, data_collator, train_dataset,
                        callbacks=[SavedPerEpochCallback()])
    
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    if args.low_rank_training:
        suffix='final_lora'
    else:
        suffix='final'
        
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=os.path.join(args.output_dir, f'{suffix}'))
    
    if args.low_rank_training:
        model = model.merge_and_unload()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=os.path.join(args.output_dir, f'{suffix}'))
        model.save_pretrained(output_dir=os.path.join(args.output_dir, f'{suffix}'))


if __name__ == "__main__":
    hydra_formatted_args = []
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--"):])
        else:
            hydra_formatted_args.append(arg)
    sys.argv = hydra_formatted_args    
    main()
