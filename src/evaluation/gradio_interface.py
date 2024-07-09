import gradio as gr
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaConfig,
                          StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer)
from threading import Thread



tokenizer = AutoTokenizer.from_pretrained("outputs/tinylm_finetune_v4/final",
                            use_fast=True,
                            trust_remote_code=True,
                            add_special_tokens=False)

model = AutoModelForCausalLM.from_pretrained("outputs/tinylm_finetune_v4/final", 
                                             torch_dtype=torch.float16,
                                             device_map='cuda')

eos_id = tokenizer.eos_token_id
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [29, 0, eos_id]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def predict(message, history):
    # history_transformer_format = history + [[message, ""]]
    stop = StopOnTokens()

    # messages = "".join(["".join(["\n<human>:"+item[0], "\n<bot>:"+item[1]])
    #             for item in history_transformer_format])
    
    # message_input = '#### Instruction:\n' + message + '\n### Response:'
    
    system = '''Human: \n{}\nAssistant:\n'''
    
    message_input = system.format(message.strip()).strip()
    model_inputs = tokenizer([message_input], return_tensors="pt").to("cuda")
    
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.8,
        stopping_criteria=StoppingCriteriaList([stop])
        )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    partial_message = ""
    for new_token in streamer:
        if new_token != '<':
            partial_message += new_token

        yield partial_message
    # partial_message = model.generate
    
    # partial_message = partial_message.split('Assistant:')[-1].strip() 
    

gr.ChatInterface(predict).launch()