import copy
import datasets
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Dict
import os
import shutil
import torch
import transformers
import random
from src.data_utils.instruction_aug import random_delete, random_repeat


'''
    tatsu-lab/alpaca:
        instruction: Give three tips for staying healthy.
        input: 
        output: 1.Eat a balanced diet an ...
        text: Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nGive three tips for staying healthy.\n\n### Response:\n1.Eat a balance...

    BelleGroup/train_0.5M_CN:
        instruction:
        input:
        output
    
    BelleGroup/train_1M_CN:
        instruction:
        input:
        output
    
    BelleGroup/school_math_0.25M:
        instruction:
        input:
        output
    
    YeungNLP/firefly-train-1.1M:
        input: 自然语言推理：\n前提：家里人心甘情愿地养他,还有几家想让他做女婿的\n假设：他是被家里人收养的孤儿
        target: 中立
        kind: NLI
    
    ise-uiuc/Magicoder-OSS-Instruct-75K:
        problem:
        solution:
'''

def format_conv(example, tokenizer, aug=True):
    conversations = example['conversations']
    conv_len = len(conversations)
    prompt = ""
    output = ""
    for i in range(0, conv_len, 2):
        human_val = conversations[i]['value']
        assistant_val = conversations[i + 1]['value']
        
        if aug:
            if random.random() > 0.5:
                human_val = random_delete(human_val)
            else:
                human_val = random_repeat(human_val)
        
        if i != conv_len - 2:
            prompt += f"Human:\n{human_val}\n"
            prompt += f"Assistant:\n{assistant_val}\n" + tokenizer.eos_token
        else:
            prompt += f"Human:\n{human_val}\nAssistant:\n"
            output = f"{assistant_val}\n" + tokenizer.eos_token
            
    return dict(
        input=prompt,
        output=output,
    )

def format_conv_longalign(example, tokenizer, aug=True):
    conversations = example['messages']
    conv_len = len(conversations)
    prompt = ""
    output = ""
    for i in range(0, conv_len, 2):
        human_val = conversations[i]['content']
        assistant_val = conversations[i + 1]['content']
        
        if aug:
            if random.random() > 0.5:
                human_val = random_delete(human_val)
            else:
                human_val = random_repeat(human_val)
        
        if i != conv_len - 2:
            prompt += f"Human:\n{human_val}\n"
            prompt += f"Assistant:\n{assistant_val}\n" + tokenizer.eos_token
        else:
            prompt += f"Human:\n{human_val}\nAssistant:\n"
            output = f"{assistant_val}\n" + tokenizer.eos_token
            
    return dict(
        input=prompt,
        output=output,
    )

def format_alpaca(example, tokenizer, aug=True):
    
    instruction = example['instruction'] if 'instruction' in example.keys() and example['instruction'] is not None else ''
    input_ = example['input'] if 'input' in example.keys() and example['input'] is not None else ''
    
    output = example['output']
    
    human_val = instruction + input_
    
    if aug:
        if random.random() > 0.5:
            human_val = random_delete(human_val)
        else:
            human_val = random_repeat(human_val)
            
            
    prompt = f"Human:\n{human_val}\nAssistant:\n"
    output = f"{output}\n" + tokenizer.eos_token
    return dict(
        input=prompt,
        output=output,
    )


def load_finetune_long_data(tokenizer, cache_root:str='cache/instruction_tune',
                            seed:int=42,
                            remove_tmp:bool = False,
                            data_saved_name='long_saved'):
    data_path = os.path.join(cache_root, data_saved_name)
    try:
        all_sets = datasets.load_from_disk(data_path)
    except:
        tokenizer = transformers.AutoTokenizer.from_pretrained('/data/zs/LLM_Weight/Qwen1.5-0.5b')
        def filter_too_long(example):
            tok = tokenizer([example['instruction'] + example['output']], truncation=False, padding=False)
            return len(tok['input_ids'][0]) <= 8192

        # def align_format(example):
        #     # breakpoint()
        #     instruct = example['messages'][0]['content']
        #     output = example['messages'][1]['content']
        #     return dict(instruction=instruct, output=output)
            
        data_long_align = datasets.load_dataset('THUDM/LongAlign-10k', split='train',
                                        cache_dir=f'cache/instruction_tune/tmp/THUDM_LongAlign-10k')
        data_long_align = data_long_align.filter(lambda x: x["length"] <= 8192)
        data_long_align = data_long_align.map(format_conv_longalign, num_proc=32,
                                              fn_kwargs={'tokenizer': tokenizer, 'aug': False})
        # data_long_align = data_long_align.map(align_format, num_proc=1)
        breakpoint()
        data_long_alpaca = datasets.load_dataset('Yukang/LongAlpaca-12k', split='train',
                                        cache_dir=f'cache/instruction_tune/tmp/Yukang_LongAlpaca-12k')
        data_long_alpaca = data_long_alpaca.filter(filter_too_long, num_proc=32)
        data_long_alpaca = data_long_alpaca.map(format_alpaca, num_proc=32,
                                                fn_kwargs={'tokenizer': tokenizer, 'aug': False})
        
        all_sets = datasets.concatenate_datasets([data_long_align, data_long_alpaca])
        all_sets = all_sets.select_columns(['input', 'output'])
        all_sets.save_to_disk(data_path)
    all_sets = all_sets.shuffle(seed)
    return all_sets




def load_belle(tokenizer, cache_root:str='cache/instruction_tune',
                               seed:int=42,
                            remove_tmp:bool = False,
                            data_saved_name='short_saved_all_train',
                            aug=False):
    data_names = [
        'BelleGroup/train_0.5M_CN',
        'BelleGroup/train_3.5M_CN',
        'BelleGroup/train_1M_CN',
        # 'BelleGroup/school_math_0.25M',
        # 'YeungNLP/firefly-train-1.1M',
        'tatsu-lab/alpaca',
        'flytech/python-codes-25k'
    ]
    data_path = os.path.join(cache_root, data_saved_name)
    try:
        all_train_sets = datasets.load_from_disk(data_path)
    except:
        cache_root_tmp = os.path.join(cache_root, 'tmp')  
        if not os.path.exists(cache_root_tmp):
            os.makedirs(cache_root_tmp)
              
        all_train_sets = []
        all_test_sets = []
        
        for name in data_names:
            cache_name = name.replace('/', '_')
            cur_set = datasets.load_dataset(name, cache_dir=os.path.join(cache_root_tmp, cache_name), 
                                            num_proc=8,
                                            split='train')
            if name == 'YeungNLP/firefly-train-1.1M':
                cur_set = cur_set.rename_column('input', 'instruction')
                cur_set = cur_set.rename_column('target', 'output')
            
            cur_set.shuffle(seed)
            test_ratio = 0.01 if 0.01 * len(cur_set) <= 10000 else 10000 / len(cur_set) 
            split_set = cur_set.train_test_split(test_size=test_ratio)
            test_set = split_set['test']
            train_set = split_set['train']
            
            if name == 'BelleGroup/train_3.5M_CN':
                train_set = train_set.map(format_conv,
                                      batched=False,
                                      num_proc=32,
                                      fn_kwargs={'tokenizer': tokenizer, 'aug': aug})
                test_set = test_set.map(format_conv,
                                      batched=False,
                                      num_proc=32,
                                      fn_kwargs={'tokenizer': tokenizer, 'aug': False})
            else:
                train_set = train_set.map(format_alpaca,
                                      batched=False,
                                      num_proc=32,
                                      fn_kwargs={'tokenizer': tokenizer, 'aug': aug})
                test_set = test_set.map(format_alpaca,
                                      batched=False,
                                      num_proc=32,
                                      fn_kwargs={'tokenizer': tokenizer, 'aug': False})
            
            test_set = test_set.select_columns(['input', 'output'])
            test_set.save_to_disk(os.path.join(cache_root, f'{cache_name}_test'))
            all_train_sets.append(train_set)

        all_train_sets = datasets.concatenate_datasets(all_train_sets)
        all_train_sets = all_train_sets.select_columns(['input', 'output'])
        
        all_train_sets.save_to_disk(data_path)
        if remove_tmp:
            shutil.rmtree(cache_root_tmp)
        
    all_train_sets = all_train_sets.shuffle(seed)
    return all_train_sets


if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained('/data/zs/LLM_Weight/Qwen1.5-0.5b')
    inst_set = load_finetune_long_data(tokenizer, data_saved_name='finetune_long')
    # inst_set2 = load_belle(tokenizer, data_saved_name='belle1m_wo_aug', aug=False)
    
