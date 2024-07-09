from functools import partial
from pathlib import Path
import re
import shutil
import sys
import time
import datasets
from omegaconf import DictConfig
from typing import Optional, Dict, List, Union
import multiprocessing
import json 
import os
import jieba
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinHashLSH
from pyspark.ml.linalg import Vectors
from pyspark import SparkContext, SparkConf
from pyspark.sql.functions import col, split, udf
from pyspark.sql.types import StringType, ArrayType, StructType, StructField, StringType, Row
from loguru import logger
from src.data_utils.zh_data.rules import to_simplified, remove_emoji, remove_other_brackets, punc_regularized, \
                        remove_email, remove_ip, remove_html_tags, filter_too_short, jieba_tokenize


def raw_text_to_json(path, doc_spliter="", json_key="text", min_doc_length=10):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        print("No found file %s" % path)
        return 0, None

    out_filepath = path + ".jsonl"
    fout = open(out_filepath, "w", encoding="utf-8")
    len_files = 0
    with open(path, "r") as f:
        doc = ""
        line = f.readline()
        while line:
            len_files += len(line)
            if line.strip() == doc_spliter:
                # if len(list(jieba.cut(doc))) > min_doc_length:
                if len(doc) > min_doc_length:
                    fout.write(json.dumps({json_key: doc}, ensure_ascii=False) + "\n")
                doc = ""
            else:
                doc += line
            line = f.readline()

        if len(doc) > min_doc_length:
            fout.write(json.dumps({json_key: doc}, ensure_ascii=False) + "\n")
        doc = ""

    return len_files, out_filepath


def merge_file(file_paths, output_path_root):
    output_path = os.path.join(output_path_root, "raw_merged.jsonl")
        
    print("Merging files into %s" % output_path)
    with open(output_path, "wb") as wfd:
        for f in file_paths:
            if f is not None and os.path.exists(f):
                with open(f, "rb") as fd:
                    shutil.copyfileobj(fd, wfd)
                os.remove(f)
    print("File save in %s" % output_path)
    return output_path

def deduplication_minihash(spark_session: SparkSession,
                           data_path: str = 'cache/zh_data_processed/raw_merged.jsonl'):

    raw_data = spark_session.read.json(data_path)
    df = raw_data.withColumn("tokens", udf(jieba_tokenize, ArrayType(StringType()))(col("text")))
    
    start_num = df.count()
    
    hashingTF = HashingTF(inputCol="tokens", outputCol="features", numFeatures=1 << 16)
    df_tf = hashingTF.transform(df)
    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=10)
    model = mh.fit(df_tf)
    df_hashed = model.transform(df_tf)
    deduplicated_text_df = df_hashed.dropDuplicates(["hashes"])
    
    end_num = deduplicated_text_df.count()
    
    logger.info(f'minihash去重结束，初始数据量:{start_num} 结束数据量:{end_num}')
    
    final_text_df = deduplicated_text_df.select('text')        
    return final_text_df


class DataProcessPipeline:
    def __init__(self, output_path:str, worker_num=16):
        self.worker_num = worker_num
        
        self.output_path_root = output_path

    def merge_raw_text(self, file_paths:List[str],
                            ):
        '''
        args：
            file_paths：表示若干txt文件名。每个txt里表示一条条数据，每条数据由空行分开
        
        '''
        pool = multiprocessing.Pool(self.worker_num)
        
        trans_json = partial(
            raw_text_to_json, doc_spliter="", json_key="text", min_doc_length=20
        )
        encoded_files = pool.imap(trans_json, file_paths, 1)

        out_paths = []
        for i, (_, out_path) in enumerate(encoded_files, start=1):
            out_paths.append(out_path)
            print(f"Processed {i} files")

        merge_file(out_paths, self.output_path_root)
    
    def pyspark_filter(self, input_file:str = 'cache/zh_data_processed/raw_merged.jsonl', 
                       save_path:str = 'cache/zh_data_processed/preprocessed'):
        spark_session = SparkSession.builder.config('spark.executor.memory','64g')\
            .config('spark.driver.memory','64g').getOrCreate()
        
        logger.info('使用minihash去重.............................')
        text_df = deduplication_minihash(spark_session, input_file)
        raw_rdd = text_df.rdd
        
        raw_rdd = raw_rdd.map(lambda x:x['text'])
        filtered_too_short_rdd = raw_rdd.filter(filter_too_short)
        rule_rdd = filtered_too_short_rdd
        rules_func = [to_simplified, remove_emoji, remove_other_brackets, punc_regularized, \
                        remove_email, remove_ip, remove_html_tags]
        
        for rf in rules_func:
            rule_rdd = rule_rdd.map(rf)
        
        final_rdd = rule_rdd.map(lambda x:Row(text=x))
        
        schema = StructType([
            StructField("text", StringType(), True),
        ])
        
        df = spark_session.createDataFrame(final_rdd, schema)
        df.write.parquet(save_path, mode='overwrite')    
                
     
if __name__ == "__main__":
    input_path = 'cache/zh_data_raw'
    output_path_root = 'cache/zh_data_processed'
    
    file_paths = []
    if os.path.isfile(input_path):
        file_paths.append(input_path)
    else:
        for root, _, fs in os.walk(input_path):
            for f in fs:
                file_paths.append(os.path.join(root, f))
    
    data_processor = DataProcessPipeline(output_path_root)
    # data_processor.merge_raw_text(file_paths)
    data_processor.pyspark_filter(input_file='cache/zh_data_processed/raw_merged.jsonl')