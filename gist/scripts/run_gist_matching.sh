set -e

export CUDA_VISIBLE_DEVICES=1

TARGET_TASK="mmlu"
MODEL="qwen2.5"

GIST_TEMPLATE="../gist_results/grads/${MODEL}/{ckpt}/${TARGET_TASK}"
OUTPUT_PATH="../gist_results/scores/${MODEL}"

CKPTS="422"


TRAIN_NAMES="flan_v2 cot dolly oasst1"

echo "Checkpoints: $CKPTS"
mkdir -p "$OUTPUT_PATH"

python gist/matching_gist.py \
    --output_path "$OUTPUT_PATH" \
    --gist_path_template "$GIST_TEMPLATE" \
    --target_task "$TARGET_TASK" \
    --train_file_names $TRAIN_NAMES \
    --checkpoints $CKPTS \
    --checkpoint_weights 1 \

