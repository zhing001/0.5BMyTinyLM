import copy
import os
import datasets
import torch
import torch.utils
import transformers 
from transformers import PreTrainedTokenizer, AutoTokenizer
from itertools import chain

from omegaconf import DictConfig, OmegaConf
import gc
from pyspark.sql import SparkSession
from datasets import Dataset
import shutil
from src.common.logger import logger_rank0 as logger
from typing import List, Dict


def load_dolma():            
    data = datasets.load_dataset('src/data_utils/dolma', 
                cache_dir="cache/dolma_cache", 
                split="train",
                num_proc=24)

    data = data.select_columns('text')
    data = data.shuffle(42)
    
    return data

def load_chi(spark_path:str = 'cache/zh_data_processed/spark_preprocessed',
             removed_tmp:bool = True):
    chi_path = 'cache/zh_data_processed/chi_final'
    spark_cache_path = 'cache/zh_data_processed/chi_cache'
    try:
        chi_dataset = datasets.load_from_disk(chi_path)
    except:
        spark_session = SparkSession.builder.config('spark.executor.memory','64g')\
                        .config('spark.driver.memory','64g').getOrCreate()
        df = spark_session.read.parquet(spark_path)
        chi_dataset = datasets.Dataset.from_spark(df, cache_dir=spark_cache_path, keep_in_memory=False)
        chi_dataset.save_to_disk(chi_path, num_proc=16)
        spark_session.stop()

    if removed_tmp:
        if os.path.exists(spark_path) and os.path.isdir(spark_path):
            shutil.rmtree(spark_path)
        if os.path.exists(spark_cache_path) and os.path.isdir(spark_cache_path):
            shutil.rmtree(spark_cache_path)
        
    return chi_dataset


def grouped_pretrain_data(
    examples: Dict[str, List[str]],
    model_max_length: int,
    tokenizer: PreTrainedTokenizer,

) -> Dict[str, List[List[int]]]:
    gc.collect()
    
    token_ids_list = [
        tokenizer(item, 
                  add_special_tokens=False,
                  truncation=True,
                  padding=False)["input_ids"] + [tokenizer.eos_token_id] for item in examples["text"]
    ]  
    
    concatenated_ids = list(chain(*token_ids_list))
    
    # 长度小于1024的，会被丢弃    
    total_length = (len(concatenated_ids) // model_max_length) * model_max_length
    result = [
        concatenated_ids[i : i + model_max_length]
        for i in range(0, total_length, model_max_length)
    ]
    
    input_ids = result
    labels = copy.deepcopy(input_ids)
    return {"input_ids": input_ids, "labels": labels}


def load_pretrain_data(removed_tmp=False, tokenizer=None, grouped_len=1024, 
                       return_packing=True) -> datasets.Dataset:
    if return_packing:
        try:
            data = datasets.load_from_disk('cache/pt_packing')
        
        except:
            nl_dataset = load_dolma()
            chi_dataset = load_chi()
            
            concat_set = datasets.concatenate_datasets([chi_dataset, nl_dataset])
            concat_set = concat_set.shuffle(42)
            # concat_set = concat_set.with_format("torch")
            concat_set = concat_set.select_columns('text')
            
            tmp = 'cache/pt_packing_cache'
            data = concat_set.map(grouped_pretrain_data,
                    keep_in_memory=False,
                    batched=True,
                    batch_size=1024,
                    num_proc=32,
                    remove_columns='text',
                    load_from_cache_file=True,
                    cache_file_name='cache/pt_packing_cache/packing.arrow',
                    fn_kwargs={
                        'model_max_length': grouped_len,
                        'tokenizer': tokenizer
                    })
            
            data.save_to_disk('cache/pt_packing', num_proc=16)
            if removed_tmp:
                if os.path.exists(tmp) and os.path.isdir(tmp):
                    shutil.rmtree(tmp)
    else:
        nl_dataset = load_dolma()
        chi_dataset = load_chi()
        
        data = datasets.concatenate_datasets([chi_dataset, nl_dataset])
        data = data.shuffle(42)
        # concat_set = concat_set.with_format("torch")
        data = data.select_columns('text')
        
    data = data.shuffle(42)
    logger.info(f'dataset length: {len(data)}')
    return data


if __name__ == "__main__":
    grouped_len=1024
    tokenizer = AutoTokenizer.from_pretrained('/data/zs/LLM_Weight/Qwen1.5-0.5b',
                                               model_max_length=grouped_len)
    data = load_pretrain_data(tokenizer=tokenizer, grouped_len=grouped_len)
    
    tokenizer = AutoTokenizer.from_pretrained('/data/zs/LLM_Weight/Qwen1.5-0.5b',
                                              model_max_length=2048)
    # data = data.map(batch_grouped_pretrain_generate,
    #          keep_in_memory=False,
    #          batched=True,
    #          batch_size=1024,
    #          num_proc=32,
    #          remove_columns='text',
    #          cache_file_name='cache/pt_packing_cache/packing.arrow',
    #          fn_kwargs={
    #              'model_max_length': 2048,
    #              'tokenizer': tokenizer
    #          }
    #         )
    eos_token_id = tokenizer.eos_token_id
    test_sample = torch.LongTensor(data[0]['input_ids']).view(1, -1)
    
    bs = 1
    seq_length = 16
    mask = torch.tril(torch.ones((bs, 16, 16))).view(
        bs, 1, seq_length, seq_length
    )
    # mask = mask < 0.5
    
    eos_positions = torch.where(test_sample == eos_token_id)[1].tolist()
    eos_positions = [4,8,12]
    for stidx, enidx in zip(eos_positions[:-1], eos_positions[1:]):
        mask[:, :, stidx + 1:, :stidx + 1] = 0
    
    breakpoint()