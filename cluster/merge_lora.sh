#!/bin/bash
# Merge LoRA checkpoint into full model weights
# Usage: bash cluster/merge_lora.sh <checkpoint_path> [output_path]
set -e

CHECKPOINT_PATH="${1:?Usage: bash merge_lora.sh <checkpoint_path> [output_path]}"
OUTPUT_PATH="${2:-${CHECKPOINT_PATH}-merged}"

eval "$(conda shell.bash hook)"
conda activate groot

echo "Merging LoRA weights..."
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Output:     $OUTPUT_PATH"

python -c "
from gr00t.model.gr00t_n1 import GR00T_N1_5
from peft import PeftModel
import shutil, os

base_model = GR00T_N1_5.from_pretrained('nvidia/GR00T-N1.5-3B')
model = PeftModel.from_pretrained(base_model, '$CHECKPOINT_PATH')
merged = model.merge_and_unload()
merged.save_pretrained('$OUTPUT_PATH')
shutil.copytree(
    '$CHECKPOINT_PATH/experiment_cfg',
    '$OUTPUT_PATH/experiment_cfg',
    dirs_exist_ok=True
)
print('Merged model saved to: $OUTPUT_PATH')
"
