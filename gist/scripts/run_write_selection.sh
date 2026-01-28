#!/bin/bash

TARGET_TASK="tydiqa"
MODEL="qwen2.5-1.5b"

TRAIN_FILES="./data/train/processed/cot/cot_data.jsonl \
             ./data/train/processed/dolly/dolly_data.jsonl \
             ./data/train/processed/flan_v2/flan_v2_data.jsonl \
             ./data/train/processed/oasst1/oasst1_data.jsonl"
TRAIN_NAMES="cot dolly flan_v2 oasst1"
OUTPUT_PATH="../gist_results/scores/${MODEL}/"
PERCENTAGE=0.05

export PYTHONPATH=$PYTHONPATH:.

python3 -m gist.data_selection.write_selected_data \
    --target_task_names $TARGET_TASK \
    --train_file_names $TRAIN_NAMES \
    --train_files $TRAIN_FILES \
    --output_path $OUTPUT_PATH \
    --percentage $PERCENTAGE