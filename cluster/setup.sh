#!/bin/bash
# Setup script for GR00T fine-tuning on cluster
# Tested on: 2x RTX 3080 (10GB), CUDA 12.0 driver, Ubuntu 20.04
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="groot"

echo "=== GR00T Cluster Training Setup ==="
echo "Repo dir: $REPO_DIR"

# Step 1: Create conda env
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda env '$ENV_NAME' already exists, activating..."
else
    echo "Creating conda env '$ENV_NAME' with Python 3.11..."
    conda create -n $ENV_NAME python=3.11 -y
fi

# Activate
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Step 2: Install PyTorch with CUDA 11.8 (compatible with driver 525 / CUDA 12.0)
echo "Installing PyTorch 2.5.1+cu118..."
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

# Step 3: Clone and install Isaac-GR00T
if [ ! -d "$REPO_DIR/Isaac-GR00T" ]; then
    echo "Cloning Isaac-GR00T..."
    git clone https://github.com/NVIDIA-Omniverse/Isaac-GR00T.git "$REPO_DIR/Isaac-GR00T"
fi

echo "Installing Isaac-GR00T..."
cd "$REPO_DIR/Isaac-GR00T"
pip install -e ".[base]"

# Step 4: Try flash-attn (may fail on older CUDA — that's OK, falls back to SDPA)
echo "Attempting flash-attn install (optional)..."
pip install flash-attn --no-build-isolation 2>/dev/null && echo "flash-attn installed!" || echo "flash-attn failed — will use SDPA attention (slower but works)"

# Step 5: Verify
echo ""
echo "=== Verification ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python -c "import gr00t; print('Isaac-GR00T: OK')"
python -c "
try:
    from flash_attn import flash_attn_func
    print('flash-attn: OK')
except ImportError:
    print('flash-attn: NOT AVAILABLE (will use SDPA fallback)')
"

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Transfer dataset:  scp -r user@local:/home/ray/datasets/chess_v3 ~/datasets/"
echo "  2. Run training:      bash cluster/train.sh --dataset_path ~/datasets/chess_v3/chess_demos_v3/lerobot"
