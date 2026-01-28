TARGET_TASK_NAME="tydiqa"
MODEL="qwen2.5-1.5b"
PERCENTAGE=0.05
TRAIN_FILES=../gist_results/scores/${MODEL}/${TARGET_TASK_NAME}/top_p0.05.jsonl
MODEL_PATH=Qwen/Qwen2.5-1.5B
JOB_NAME=${MODEL}-less-p${PERCENTAGE}-lora-baseline-${TARGET_TASK_NAME}


./gist/scripts/train/lora_train.sh "$TRAIN_FILES" "$MODEL_PATH" "$JOB_NAME"