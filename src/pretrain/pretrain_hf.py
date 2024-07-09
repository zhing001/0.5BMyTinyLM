import copy
import gc
import os
from pathlib import Path
import random
from dataclasses import dataclass, field
import sys
sys.path.append(str('.'))

from typing import Optional, Dict, Sequence

import torch
import torch.distributed
import transformers
from transformers import Trainer, Qwen2ForCausalLM, AutoConfig, Qwen2Config, TrainingArguments

from loguru import logger
import hydra
from accelerate.utils import set_seed
from omegaconf import DictConfig
from src.pretrain.data_collator import DataCollatorForSupervisedDataset
from src.data_utils.load_pt_data import load_pretrain_data
from loguru import logger
from torch.utils.data import DistributedSampler, DataLoader


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


class MyTrainer(Trainer):
    def __init__(self, model, args, tokenizer, data_collator, train_dataset):
        super().__init__(model=model,
                         args=args,
                         data_collator=data_collator,
                         tokenizer=tokenizer,
                         train_dataset=train_dataset)
        
    def get_train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset)
        data_loader = DataLoader(self.train_dataset, 
                                 sampler=sampler,
                                 batch_size=self.args.per_device_train_batch_size,
                                 collate_fn=self.data_collator)
        return data_loader
        
        
@hydra.main(config_path="../../conf", config_name="tinylm0.5b_pretrain")
def main(args: DictConfig):
    logger.info('='*100)
    logger.info(args)
    
    set_seed(args.seed)
    args.local_rank = os.environ["LOCAL_RANK"]
    logger.info(args.local_rank)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer.tokenizer_path,
        model_max_length=args.pretrain_seq_len,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )

    if args.local_rank == 0:
        logger.info("Load tokenizer from {} over.".format(args.tokenizer.tokenizer_path))
    
    model_config = Qwen2Config.from_dict(args.model_argumnts)
    model = Qwen2ForCausalLM(config=model_config)

    # model = model.to(torch.device("cuda"))
    if args.local_rank == 0:
        logger.info("Load model from {} over.".format(args.model_name_or_path))

    train_dataset = load_pretrain_data()

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, args=args, pp_format=False)

    training_args = TrainingArguments(**args.training_arguments)
      
    trainer = MyTrainer(model, training_args, tokenizer, data_collator, train_dataset)
    # data_module=dict(train_dataset=train_dataset,
    #                  eval_dataset=None, data_collator=data_collator)
    # trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module,
    #                   )
    
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=args.output_dir)

if __name__ == "__main__":
    
    hydra_formatted_args = []
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--"):])
        else:
            hydra_formatted_args.append(arg)
    sys.argv = hydra_formatted_args    
    main()
