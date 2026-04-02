#!/bin/bash
# Full training pipeline: BC-RNN visuomotor + GR00T N1.5 on right-arm chess demos
# Suspends the PC when all training is complete.
set -e

PYTHON="/home/ray/miniconda3/envs/lerobot-arena/bin/python"
DATASET="/home/ray/datasets/chess/chess_right_arm_v1.hdf5"
LEROBOT_DIR="/home/ray/datasets/chess/chess_right_arm_v1/lerobot"
ISAACLAB_DIR="/home/ray/IsaacLab-Arena/submodules/IsaacLab"
ARENA_DIR="/home/ray/IsaacLab-Arena"

echo "============================================"
echo "  Right-Arm Chess Training Pipeline"
echo "============================================"
echo "Dataset: $DATASET"
echo ""

# --------------------------------------------------
# Step 1: BC-RNN Visuomotor (SKIPPED — already trained to epoch 138)
# --------------------------------------------------
# echo "[1/3] Training BC-RNN visuomotor (1000 epochs)..."
# cd "$ISAACLAB_DIR"
# ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
#     --task Isaac-Chess-FixedBase-G1-RightArm-Abs-v0 \
#     --algo bc \
#     --dataset "$DATASET" \
#     --epochs 1000 \
#     --log_dir robomimic_right_arm
# echo "[1/3] BC-RNN training complete."
echo "[1/3] BC-RNN skipped (already done)."
echo ""

# --------------------------------------------------
# Step 2: Convert HDF5 to LeRobot format for GR00T
# --------------------------------------------------
echo "[2/3] Converting HDF5 to LeRobot format..."
cd "$ARENA_DIR"
$PYTHON -m isaaclab_arena_gr00t.data_utils.convert_hdf5_to_lerobot \
    --yaml_file isaaclab_arena_gr00t/config/g1_chess_right_arm_config.yaml

echo "[2/3] LeRobot conversion complete."
echo ""

# --------------------------------------------------
# Step 3: Train GR00T N1.5 (fine-tune, 100K steps)
# --------------------------------------------------
echo "[3/3] Training GR00T N1.5 (100K steps)..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ARENA_DIR/submodules/Isaac-GR00T"
$PYTHON scripts/gr00t_finetune.py \
    --dataset-path "$LEROBOT_DIR" \
    --output-dir /home/ray/models/chess-right-arm-groot-100k \
    --num-gpus 1 \
    --batch-size 1 \
    --max-steps 100000 \
    --lora-rank 16 \
    --data-config isaaclab_arena_gr00t.chess_data_config:UnitreeG1ChessRightArmDataConfig \
    --report-to tensorboard

echo "[3/3] GR00T training complete."
echo ""

# --------------------------------------------------
# Suspend
# --------------------------------------------------
echo "============================================"
echo "  All training complete. Suspending PC..."
echo "============================================"
sleep 5
systemctl suspend
