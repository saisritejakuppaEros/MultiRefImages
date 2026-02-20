#!/bin/bash
# MoviePostProduction Training Script
# 3-Stage Training Pipeline

set -e  # Exit on error

echo "=========================================="
echo "MoviePostProduction 3-Stage Training"
echo "=========================================="
echo ""

# Configuration
export CUDA_VISIBLE_DEVICES=0  # Set GPU ID
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_stage() {
    echo -e "${GREEN}=========================================="
    echo -e "$1"
    echo -e "==========================================${NC}"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if data is prepared
if [ ! -d "./output_data/latents/vae_full_images" ]; then
    print_error "VAE latents not found. Please run prepare_synthetic_dataset.py first."
    exit 1
fi

# ==========================================
# Stage 1: Layout Training (70K iterations)
# ==========================================
print_stage "Stage 1: Layout Training"
print_info "Training layout from scratch (InstanceAssemble-style)"
print_info "Duration: ~70K iterations"
print_info "Trainable: alpha_predictor, instance_fusion_mlp, layout_head, assemble_attn_lora"
echo ""

accelerate launch --mixed_precision=bf16 train.py \
    --config configs/stage1.yaml

print_info "Stage 1 completed! Checkpoint saved to: runs/checkpoints/stage1/checkpoint_70000"
echo ""

# ==========================================
# Stage 2: Modulation + Identity Training (150K iterations)
# ==========================================
print_stage "Stage 2: Modulation + Identity Training"
print_info "Adding identity on top of layout"
print_info "Duration: ~150K iterations"
print_info "Trainable: Stage 1 modules + modulation_head + per_block_adaln_projections"
print_info "Resuming from: runs/checkpoints/stage1/checkpoint_70000"
echo ""

accelerate launch --mixed_precision=bf16 train.py \
    --config configs/stage2.yaml \
    --resume_from_checkpoint runs/checkpoints/stage1/checkpoint_70000

print_info "Stage 2 completed! Checkpoint saved to: runs/checkpoints/stage2/checkpoint_150000"
echo ""

# ==========================================
# Stage 3: Feedback Bridge Fine-tuning (10K iterations)
# ==========================================
print_stage "Stage 3: Feedback Bridge Fine-tuning"
print_info "Fine-tuning feedback bridge for cross-block communication"
print_info "Duration: ~10K iterations"
print_info "Trainable: feedback_bridge ONLY"
print_info "Resuming from: runs/checkpoints/stage2/checkpoint_150000"
echo ""

accelerate launch --mixed_precision=bf16 train.py \
    --config configs/stage3.yaml \
    --resume_from_checkpoint runs/checkpoints/stage2/checkpoint_150000

print_info "Stage 3 completed! Checkpoint saved to: runs/checkpoints/stage3/checkpoint_10000"
echo ""

# ==========================================
# Training Complete
# ==========================================
print_stage "Training Complete!"
echo ""
echo "Final model checkpoint: runs/checkpoints/stage3/checkpoint_10000"
echo ""
echo "To view training progress:"
echo "  tensorboard --logdir=runs/logs --port=6006"
echo ""
echo "Checkpoints saved:"
echo "  - Stage 1: runs/checkpoints/stage1/"
echo "  - Stage 2: runs/checkpoints/stage2/"
echo "  - Stage 3: runs/checkpoints/stage3/"
echo ""
