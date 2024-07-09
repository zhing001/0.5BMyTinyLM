import copy
import datasets
from dataclasses import dataclass, field
import torch 
import transformers
from typing import Sequence, Dict, Optional, List, Union
from transformers import TrainingArguments
from omegaconf import DictConfig


IGNORE_INDEX = -100

def get_position_ids(input_ids):
    seq_length = input_ids.shape[1]
    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long)
    return position_ids.unsqueeze(0).expand_as(input_ids)


def create_attention_mask(bs, seq_length, attention_mask_2d):
    mask = torch.tril(torch.ones((bs, seq_length, seq_length))).view(
        bs, 1, seq_length, seq_length
    )
    mask = mask < 0.5
    attention_mask = attention_mask_2d[:, None, None, :].expand(bs, 1, seq_length, seq_length)
    
    # 因为padding，padding部分设为true，此时表示无需注意力
    attn_mask = mask | (~attention_mask)
    attn_mask = torch.where(attn_mask == True, float("-inf"), 0).long()
    
    return attn_mask

        
@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    args: Union[DictConfig, TrainingArguments]
    pp_format: bool = field(default=True)
    
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        input_ids = [i['input_ids'] for i in instances]
        labels = [i['labels'] for i in instances]
        
        input_ids_tensor = []
        labels_tensor = []
        
        bs = len(input_ids)
        seq_length = self.args.pretrain_seq_len
        eos_token_id = self.tokenizer.eos_token_id
        
        '''
        在数据处理的时候，将长度打包，并丢弃长度小于2048的，所以不用做padding
        '''
        causal_attention_mask = torch.tril(torch.ones((bs, seq_length, seq_length))).view(
                bs, 1, seq_length, seq_length
            ) # 生成下三角，对角线及下面都是1
        for idx, (each_input, each_label) in enumerate(zip(input_ids, labels)):
            assert len(each_input) == seq_length
            
            cur_ids_tensor = torch.LongTensor(each_input)
            cur_label_tensor = torch.LongTensor(each_label)
            
            input_ids_tensor.append(cur_ids_tensor.view(1, -1))
            labels_tensor.append(cur_label_tensor.view(1, -1))
            
            eos_positions = torch.where(cur_ids_tensor == eos_token_id)[0].tolist()
            for eos_idx in eos_positions:
                # if stidx > 0 and eos_positions[stidx] - eos_positions[stidx - 1] == 1:
                #     break
                causal_attention_mask[idx, :, eos_idx + 1:, :eos_idx + 1] = 0
                
        causal_attention_mask = causal_attention_mask < 0.5
        causal_attention_mask = torch.where(causal_attention_mask == True, float("-inf"), 0).long()
        
        input_ids = torch.cat(input_ids_tensor)
        labels = torch.cat(labels_tensor)
        if self.pp_format:
            # deepspeed流水线并行格式
            position_ids = get_position_ids(input_ids)
            
            return (
                (
                    input_ids,
                    position_ids,
                    causal_attention_mask,
                ),
                labels
            )
            
        else:
            # hf trainer格式
            return dict(input_ids=input_ids, labels=labels, attention_mask=causal_attention_mask)
        
        
        # tok_output = self.tokenizer([i['text'] + self.tokenizer.eos_token for i in instances],
        #                            max_length=self.args.pretrain_seq_len,
        #                            truncation=True,
        #                            padding='max_length',
        #                            return_tensors='pt')
        
        # input_ids = tok_output['input_ids']
        # labels = tok_output['input_ids']
        # attention_mask_2d = input_ids.ne(self.tokenizer.pad_token_id)
        
        # if self.pp_format:
        #     # deepspeed流水线并行格式
        #     position_ids = get_position_ids(input_ids)
        #     bsz, tgt_len = input_ids.size(0), input_ids.size(1)
        #     attn_mask = create_attention_mask(bsz, tgt_len, attention_mask_2d)
            
        #     return (
        #         (
        #             input_ids,
        #             position_ids,
        #             attn_mask,
        #         ),
        #         labels
        #     )
            
        # else:
        #     # hf trainer格式
        #     return dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask_2d)