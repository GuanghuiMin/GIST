TARGET_TASK_NAME="tydiqa"
MODEL="qwen2.5-1.5b"
PERCENTAGE=0.05
TRAIN_FILES=../gist_results/scores/${MODEL}/${TARGET_TASK_NAME}/top_p0.05.jsonl
# MODEL_PATH=meta-llama/Llama-2-7b-hf
MODEL_PATH=Qwen/Qwen2.5-1.5B
JOB_NAME=${MODEL}-less-p${PERCENTAGE}-lora-baseline-${TARGET_TASK_NAME}

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="GIST"
export WANDB_NAME=${JOB_NAME}
export WANDB_DIR="/tmp/wandb_qwen"
export WANDB_RESUME="never"
export MASTER_PORT=29541


mkdir -p "$WANDB_DIR"
./gist/scripts/train/lora_train.sh "$TRAIN_FILES" "$MODEL_PATH" "$JOB_NAME"