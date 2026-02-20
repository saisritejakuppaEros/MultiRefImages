#!/bin/bash
# Stage 3: Feedback Bridge Fine-tuning (10K iterations)

echo "=========================================="
echo "Stage 3: Feedback Bridge Fine-tuning"
echo "=========================================="
echo ""
echo "Goal: Fine-tune feedback bridge for cross-block communication"
echo "Trainable: feedback_bridge ONLY"
echo "Iterations: 10,000"
echo "Learning Rate: 1e-5"
echo ""

# Find the latest Stage 2 checkpoint
STAGE2_CHECKPOINT=$(ls -td runs/checkpoints/stage2/checkpoint_* 2>/dev/null | head -1)

if [ -z "$STAGE2_CHECKPOINT" ]; then
    echo "ERROR: No Stage 2 checkpoint found!"
    echo "Please run train_stage2.sh first or specify a checkpoint manually."
    echo ""
    echo "Usage: bash train_stage3.sh [checkpoint_path]"
    echo "Example: bash train_stage3.sh runs/checkpoints/stage2/checkpoint_150000"
    exit 1
fi

# Allow override via command line argument
if [ ! -z "$1" ]; then
    STAGE2_CHECKPOINT="$1"
fi

echo "Resume from: $STAGE2_CHECKPOINT"
echo ""

# Check if checkpoint exists
if [ ! -d "$STAGE2_CHECKPOINT" ]; then
    echo "ERROR: Checkpoint $STAGE2_CHECKPOINT not found!"
    exit 1
fi

# Run training
accelerate launch --mixed_precision=bf16 train.py \
    --config configs/stage3.yaml \
    --resume_from_checkpoint "$STAGE2_CHECKPOINT"

echo ""
echo "✓ Stage 3 training completed!"
echo "Checkpoint: runs/checkpoints/stage3/checkpoint_10000"
echo ""
echo "Final model ready for inference!"
echo ""
echo "To monitor training:"
echo "  tensorboard --logdir=runs/logs --port=6006"
