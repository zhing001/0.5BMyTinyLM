import copy
import datasets
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Dict
import os
import shutil
import torch
import transformers


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = []
        labels = []
        
        for item in instances:
            prompt = item['input']
            output = item['output']
            label_prompt_ids = self.tokenizer(prompt,
                                          truncation=True,
                                          padding=False,
                                          max_length=self.tokenizer.model_max_length,
                                          )['input_ids']
            input_ids_cur = self.tokenizer(prompt + output,
                                        truncation=True,
                                        padding=False,
                                        max_length=self.tokenizer.model_max_length,
                                        )['input_ids']
            
            
            input_ids_cur = torch.LongTensor(input_ids_cur)
            cur_labels = copy.deepcopy(input_ids_cur)
            cur_labels[:len(label_prompt_ids)] = -100
    
            input_ids.append(input_ids_cur)
            labels.append(cur_labels)
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

