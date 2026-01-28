#!/bin/bash
TASK="tydiqa"
TRAIN_FILES="./data/train/processed/cot/cot_data.jsonl ./data/train/processed/dolly/dolly_data.jsonl ./data/train/processed/flan_v2/flan_v2_data.jsonl ./data/train/processed/oasst1/oasst1_data.jsonl"
TRAIN_NAMES="cot dolly flan_v2 oasst1"

CKPT=422
DATA_DIR="./data"
MODEL_PATH="../out/llama3.2-3b-p0.05-lora-seed3/checkpoint-${CKPT}"
OUTPUT_DIR="../gist_results/grads/llama3.2/${CKPT}/${TASK}"


python3 gist/get_gist_gradients.py \
    --task $TASK \
    --train_files $TRAIN_FILES \
    --train_file_names $TRAIN_NAMES \
    --model_path $MODEL_PATH \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --target_dim 50 \
    --max_val_samples 300 \
    --proj_dtype bfloat16 \
    --use_gpu_proj