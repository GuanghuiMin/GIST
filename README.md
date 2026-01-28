# GIST: Targeted Data Selection for Instruction Tuning via Coupled Optimization Geometry

This repository contains the implementation for the paper **"GIST: Targeted Data Selection for Instruction Tuning via Coupled Optimization Geometry"**.

> **Acknowledgement**: This codebase is built upon the [LESS](https://github.com/princeton-nlp/LESS) framework. We sincerely acknowledge the contributors for their excellent open-source work.

## ⚙️ Setup & Installation

### Environment
```bash
conda create -n gist_env python=3.10
conda activate gist_env

pip install -r requirements.txt

```

### Data Preparation

We utilize the same dataset format as LESS. Please refer to the original project to access the data.

---

## 🚀 Pipeline

### Step 1: Warmup Training

Perform a lightweight warmup training to obtain the necessary optimizer states.

```bash
./gist/scripts/train/warmup_lora_train.sh \
    "./data" \
    "Qwen/Qwen2.5-1.5B" \
    "0.05" \
    "3" \
    "qwen2.5-1.5b-p0.05-lora-seed3"

```

### Step 2: Calculate GIST Gradients

Compute the projected gradients for both training data and the target task validation set.

```bash
TASK="tydiqa"
MODEL="qwen2.5-1.5b"
CKPT=422
SEED="3"

# Define paths
DATA_DIR="./data"
TRAIN_FILES="./data/train/processed/cot/cot_data.jsonl ./data/train/processed/dolly/dolly_data.jsonl ./data/train/processed/flan_v2/flan_v2_data.jsonl ./data/train/processed/oasst1/oasst1_data.jsonl"
TRAIN_NAMES="cot dolly flan_v2 oasst1"

MODEL_PATH="../out/${MODEL}-p0.05-lora-seed${SEED}/checkpoint-${CKPT}"
OUTPUT_DIR="../gist_results/grads/${MODEL}/${CKPT}/${TASK}"

python3 gist/get_gist_gradients.py \
    --task $TASK \
    --train_files $TRAIN_FILES \
    --train_file_names $TRAIN_NAMES \
    --model_path $MODEL_PATH \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --target_dim 9 \
    --max_val_samples 300 \
    --proj_dtype bfloat16 \
    --use_gpu_proj

```

### Step 3: Matching and Influence Calculation

Match the gradients and calculate influence scores.

```bash
TARGET_TASK="mmlu"
MODEL="qwen2.5-1.5b"
CKPTS="422"

GIST_TEMPLATE="../gist_results/grads/${MODEL}/{ckpt}/${TARGET_TASK}"
OUTPUT_PATH="../gist_results/scores/${MODEL}"
TRAIN_NAMES="flan_v2 cot dolly oasst1"

echo "Using Checkpoints: $CKPTS"
mkdir -p "$OUTPUT_PATH"

python gist/matching_gist.py \
    --output_path "$OUTPUT_PATH" \
    --gist_path_template "$GIST_TEMPLATE" \
    --target_task "$TARGET_TASK" \
    --train_file_names $TRAIN_NAMES \
    --checkpoints $CKPTS \
    --checkpoint_weights 1

```

### Step 4: Write Selection Files

Generate the final dataset subset based on the calculated scores.

```bash
TARGET_TASK="mmlu"
MODEL="qwen2.5-1.5b"
PERCENTAGE=0.05

TRAIN_FILES="./data/train/processed/cot/cot_data.jsonl \
             ./data/train/processed/dolly/dolly_data.jsonl \
             ./data/train/processed/flan_v2/flan_v2_data.jsonl \
             ./data/train/processed/oasst1/oasst1_data.jsonl"
TRAIN_NAMES="cot dolly flan_v2 oasst1"
OUTPUT_PATH="../gist_results/scores/${MODEL}/"

export PYTHONPATH=$PYTHONPATH:.

python3 -m gist.data_selection.write_selected_data \
    --target_task_names $TARGET_TASK \
    --train_file_names $TRAIN_NAMES \
    --train_files $TRAIN_FILES \
    --output_path $OUTPUT_PATH \
    --percentage $PERCENTAGE

```

---

## 📊 Training & Evaluation

### Step 5: Fine-tune with Selected Data

Train the model using the data selected by GIST.

```bash
TARGET_TASK_NAME="mmlu"
MODEL="qwen2.5-1.5b"
PERCENTAGE=0.05

TRAIN_FILES="../gist_results/scores/${MODEL}/${TARGET_TASK_NAME}/top_p${PERCENTAGE}.jsonl"
MODEL_PATH="Qwen/Qwen2.5-1.5B"
JOB_NAME="${MODEL}-p${PERCENTAGE}-lora-gist-${TARGET_TASK_NAME}"

./gist/scripts/train/lora_train.sh "$TRAIN_FILES" "$MODEL_PATH" "$JOB_NAME"

```

### Step 6: Evaluation

Evaluate the fine-tuned model on the target task.

```bash
TARGET_TASK_NAME="mmlu"
MODEL="qwen2.5-1.5b"
PERCENTAGE=0.05

# Usage: ./eval.sh <TASK> <CHECKPOINT_PATH> <DEVICE_ID>
./eval.sh $TARGET_TASK_NAME "../out/${MODEL}-p${PERCENTAGE}-lora-gist-${TARGET_TASK_NAME}" 0

```
