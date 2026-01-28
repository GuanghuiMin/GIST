#!/bin/bash

if [ $# -lt 3 ]; then
    echo "Usage: bash eval.sh <TASK> <MODEL_PATH> <GPU_ID>"
    echo "Tasks: mmlu, bbh, tydiqa"
    echo "Example: bash eval.sh mmlu /path/to/checkpoint 0"
    exit 1
fi


TASK=$1
MODEL_PATH=$2
GPU_ID=$3

export PYTHONPATH=$PYTHONPATH:.:./evaluation
export DATA_DIR=./data
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPU_ID

MODEL_NAME=$(basename "$MODEL_PATH")
LOG_FILE="./logs/eval_${TASK}_${MODEL_NAME}.log"

echo "======================================================="
echo "Starting Evaluation:"
echo "  Task       : $TASK"
echo "  Model      : $MODEL_PATH"
echo "  Device     : CUDA $GPU_ID"
echo "  Log File   : $LOG_FILE"
echo "======================================================="


CHAT_ARGS="--use_chat_format --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format --convert_to_bf16"


case $TASK in
    "mmlu")
        # MMLU
        nohup python -m eval.mmlu.run_eval \
            --data_dir ./data/eval/mmlu \
            --ntrain 1 \
            --n_instances 200 \
            --save_dir "${MODEL_PATH}/mmlu" \
            --model_name_or_path "$MODEL_PATH" \
            --tokenizer_name_or_path "$MODEL_PATH" \
            --eval_batch_size 20 \
            $CHAT_ARGS \
            > "$LOG_FILE" 2>&1 &
        ;;

    "bbh")
        # BBH
        nohup python -m eval.bbh.run_eval \
            --data_dir ./data/eval/bbh/test \
            --save_dir "${MODEL_PATH}/bbh" \
            --model_name_or_path "$MODEL_PATH" \
            --tokenizer_name_or_path "$MODEL_PATH" \
            --max_num_examples_per_task 200 \
            --eval_batch_size 20 \
            $CHAT_ARGS \
            > "$LOG_FILE" 2>&1 &
        ;;

    "tydiqa")
        # TyDiQA
        nohup python -m eval.tydiqa.run_eval \
            --data_dir ./data/eval/tydiqa/test/ \
            --save_dir "${MODEL_PATH}/tydiqa" \
            --model "$MODEL_PATH" \
            --tokenizer "$MODEL_PATH" \
            --n_shot 1 \
            --max_num_examples_per_lang 200 \
            --max_context_length 512 \
            --eval_batch_size 20 \
            $CHAT_ARGS \
            > "$LOG_FILE" 2>&1 &
        ;;

    *)
        echo "Error: Unknown task '$TASK'. Supported tasks are: mmlu, bbh, tydiqa"
        exit 1
        ;;
esac

echo "Evaluation running in background. PID: $!"