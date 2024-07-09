import os 
from pathlib import Path  

import datasets
from datasets import Dataset
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import gc

'''
加载了小的测试版本，可以在dolma/my_dolma/my_dolma.py中调local_files
'''
nl_dataset = datasets.load_dataset("/data/zs/CodeMIA_reconstruct_v3/LLM_MIA/data/my_dolma", cache_dir="/data/zs/CodeMIA_reconstruct_v3/LLM_MIA/data/cache/cache_raw", split="train",
                                   num_proc=16)

spark_session = SparkSession.builder.config('spark.executor.memory','64g')\
            .config('spark.driver.memory','64g').getOrCreate()
chi_dataset = None 
# df = spark_session.read.parquet('cache/chi_training_data_raw/filtered_result_df')
# chi_dataset = Dataset.from_spark(df)

# breakpoint()
concat_set = nl_dataset if chi_dataset is None else datasets.concatenate_datasets([chi_dataset, nl_dataset])
concat_set = concat_set.shuffle()

def batch_iterator(batch_size=16384):
    gc.collect()
    
    tok_dataset = concat_set.select_columns("text")
    for batch in tok_dataset.iter(batch_size):
        yield batch["text"]


saved_tmp = Path('tokenizer/txt_tmp')
# data_len = len(concat_set)
# total_files_num = int(data_len / 16384 + 0.5)

# if not saved_tmp.exists():
#     saved_tmp.mkdir()
    
# for idx in range(total_files_num):
#     cur_txt = saved_tmp / f'{idx}.txt'
    
#     with open(cur_txt, 'w') as f:
#         data = next(batch_iterator())
#         for line in data:
#             f.write(line)
    
tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=50000,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=["<PAD>", "<BOS>", "<EOS>"],
)

# data = [
#     "Beautiful is better than ugly."
#     "Explicit is better than implicit."
#     "Simple is better than complex."
#     "Complex is better than complicated."
#     "Flat is better than nested."
#     "Sparse is better than dense."
#     "Readability counts."
# ]
# tokenizer.train_from_iterator(data, trainer=trainer)

# breakpoint()
files = list(saved_tmp.iterdir())
files = [str(i) for i in files]
# tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(concat_set))

tokenizer.train(files, trainer=trainer)

tokenizer.save("outputs/tokenizer-sample.json")

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    pad_token="[PAD]",
    bos_token='[BOS]',
    eos_token='[EOS]',                  
)

fast_tokenizer.save_pretrained('outputs/tokenizer-sample')
