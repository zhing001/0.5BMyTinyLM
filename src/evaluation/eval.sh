#!/bin/bash

task='ceval'
model_init_path=""
tokenizer_path="src/Qwen1.5"

while getopts ":t:m:tok:" opt; do
    case $opt in 
        t) task=$OPTARG ;;
        m) model_init_path=$OPTARG ;;
        tok) tokenizer_path=$OPTARG ;;
        *) usage ;;
    esac
done

echo $model_init_path
echo $task

if [[ $task = 'ceval' ]];then
    CUDA_VISIBLE_DEVICES=1 python src/evaluation/ceval_baichuan.py --model_name_or_path $model_init_path \
                        --tokenizer_path $tokenizer_path
elif
    [[ $task = 'mmlu' ]];then

    # git clone https://github.com/hendrycks/test
    # cd mmlu
    # wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
    # tar xf data.tar
    # mkdir results
    # cp ../evaluate_mmlu.py .
    # python evaluate_mmlu.py -m /path/to/Baichuan-7B
    cd evaluation/mmlu
    CUDA_VISIBLE_DEVICES=1 python mmlu_baichuan.py -m ../../$model_init_path
fi