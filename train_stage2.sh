#!/bin/bash
# Stage 2: Modulation + Identity Training (150K iterations)

echo "=========================================="
echo "Stage 2: Modulation + Identity Training"
echo "=========================================="
echo ""
echo "Goal: Add identity on top of layout"
echo "Trainable: Stage 1 modules + modulation_head + per_block_adaln_projections"
echo "Iterations: 150,000"
echo "Learning Rate: 5e-6"
echo ""

# Find the latest Stage 1 checkpoint
STAGE1_CHECKPOINT=$(ls -td runs/checkpoints/stage1/checkpoint_* 2>/dev/null | head -1)

if [ -z "$STAGE1_CHECKPOINT" ]; then
    echo "ERROR: No Stage 1 checkpoint found!"
    echo "Please run train_stage1.sh first or specify a checkpoint manually."
    echo ""
    echo "Usage: bash train_stage2.sh [checkpoint_path]"
    echo "Example: bash train_stage2.sh runs/checkpoints/stage1/checkpoint_70000"
    exit 1
fi

# Allow override via command line argument
if [ ! -z "$1" ]; then
    STAGE1_CHECKPOINT="$1"
fi

echo "Resume from: $STAGE1_CHECKPOINT"
echo ""

# Check if checkpoint exists
if [ ! -d "$STAGE1_CHECKPOINT" ]; then
    echo "ERROR: Checkpoint $STAGE1_CHECKPOINT not found!"
    exit 1
fi

# Run training
accelerate launch --mixed_precision=bf16 train.py \
    --config configs/stage2.yaml \
    --resume_from_checkpoint "$STAGE1_CHECKPOINT"

echo ""
echo "✓ Stage 2 training completed!"
echo "Checkpoint: runs/checkpoints/stage2/checkpoint_150000"
echo ""
echo "To monitor training:"
echo "  tensorboard --logdir=runs/logs --port=6006"
