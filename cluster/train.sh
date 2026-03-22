#!/bin/bash
# GR00T fine-tuning script for cluster (2x RTX 3080, 10GB each)
# Usage: bash cluster/train.sh [--dataset_path PATH] [--max_steps N] [--batch_size N]
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GROOT_DIR="$REPO_DIR/Isaac-GR00T"

# Defaults
DATASET_PATH="${DATASET_PATH:-$HOME/datasets/chess_v3/chess_demos_v3/lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/models/chess-v3-100k}"
MAX_STEPS="${MAX_STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
NUM_GPUS="${NUM_GPUS:-2}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"

# Parse CLI overrides
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_path) DATASET_PATH="$2"; shift 2;;
        --output_dir) OUTPUT_DIR="$2"; shift 2;;
        --max_steps) MAX_STEPS="$2"; shift 2;;
        --batch_size) BATCH_SIZE="$2"; shift 2;;
        --save_steps) SAVE_STEPS="$2"; shift 2;;
        --num_gpus) NUM_GPUS="$2"; shift 2;;
        --lora_rank) LORA_RANK="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

# Activate conda
eval "$(conda shell.bash hook)"
conda activate groot

echo "=== GR00T Fine-Tuning ==="
echo "Dataset:    $DATASET_PATH"
echo "Output:     $OUTPUT_DIR"
echo "Max steps:  $MAX_STEPS"
echo "Batch size: $BATCH_SIZE (per GPU) x $NUM_GPUS GPUs = $((BATCH_SIZE * NUM_GPUS)) global"
echo "LoRA:       rank=$LORA_RANK alpha=$LORA_ALPHA"
echo "Save every: $SAVE_STEPS steps"
echo ""

# Verify dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    echo "Transfer it first: scp -r user@local:/home/ray/datasets/chess_v3 ~/datasets/"
    exit 1
fi

cd "$GROOT_DIR"

WANDB_MODE=disabled \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH="$REPO_DIR:$PYTHONPATH" \
python scripts/gr00t_finetune.py \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --data_config isaaclab_arena_gr00t.chess_data_config:UnitreeG1ChessDataConfig \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --embodiment-tag new_embodiment \
    --video-backend decord \
    --num-gpus "$NUM_GPUS" \
    --batch_size "$BATCH_SIZE" \
    --max_steps "$MAX_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --report_to tensorboard

echo ""
echo "=== Training complete ==="
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "To merge LoRA weights, run: bash cluster/merge_lora.sh $OUTPUT_DIR/checkpoint-$MAX_STEPS"
