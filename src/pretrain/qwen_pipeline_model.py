import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import (Qwen2Config, Qwen2DecoderLayer, Qwen2RMSNorm,
                                                      Qwen2MLP, Qwen2Model)
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec
from utils import logger_rank0 as logger


class EmbeddingPipe(torch.nn.Embedding):
    def forward(self, args):
        input_ids, position_ids, attention_mask = args
        
        # print(input_ids[0, :10])
        
        inputs_embeds = super().forward(input_ids)
        return (inputs_embeds, position_ids, attention_mask)


class ParallelTransformerLayerPipe(Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx:int, activation_checkpointing: bool = False):
        super().__init__(config, layer_idx)
        self.activation_checkpointing = activation_checkpointing
        
    def forward(self, args):
        if self.activation_checkpointing:
            return self._ckpt_forward(args)
        
        hidden_states, position_ids, mask = args
        attention_mask = torch.where(mask == True, float("-inf"), 0).long()

        # print(hidden_states[0, :10])
        outputs = Qwen2DecoderLayer.forward(self,
                                            hidden_states,
                                            attention_mask,
                                            position_ids,
        )
        # print(outputs[0].size())
        return (outputs[0], position_ids, mask)

    def _ckpt_forward(self, args):
        hidden_states, position_ids, attention_mask  = args

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return Qwen2DecoderLayer.forward(module, *inputs)

            return custom_forward

        outputs = deepspeed.checkpointing.checkpoint(
            create_custom_forward(self),
            hidden_states,
            attention_mask,
            position_ids,
            None,
        )

        return outputs, position_ids, attention_mask, 
    

class LayerNormPipe(Qwen2RMSNorm):
    def forward(self, args):
        # print(len(args))
        # for item in args:
        #     print(item.size())
            
        hidden_states, position_ids, mask = args
        last_hidden_states = super().forward(hidden_states)
        # print(last_hidden_states)
        
        return (last_hidden_states,)


class LMHidden2VocabPipe(torch.nn.Linear):
    def __init__(self, in_feature: int, out_feature: int):
        super().__init__(in_feature, out_feature)
        
    def forward(self, args):
        # for item in args:
        #     print(item.size())
        #     print(item.dtype)
        
        hidden_states, *_ = args
        logits = super().forward(hidden_states)
        return (logits,)


def loss_fn(outputs, labels):
    logits, = outputs
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
    return loss


def get_model(model_config: Qwen2Config, args, 
              partition_method="type:ParallelTransformerLayerPipe",
              activation_checkpointing = False, **kwargs):
    
    layers=[
        LayerSpec(EmbeddingPipe, model_config.vocab_size, model_config.hidden_size),
        *[LayerSpec(ParallelTransformerLayerPipe, model_config, idx, activation_checkpointing)
            for idx in range(model_config.num_hidden_layers)],
        LayerSpec(LayerNormPipe, model_config.hidden_size, model_config.rms_norm_eps),
        LayerSpec(LMHidden2VocabPipe, model_config.hidden_size, model_config.vocab_size),
    ]
            
    return PipelineModule(layers,
                        loss_fn=loss_fn,
                        num_stages=args.pipe_parallel_size,
                        base_seed=args.seed,
                        partition_method=partition_method,
                        activation_checkpoint_interval=args.gradient_checkpoint_interval,
                        **kwargs)