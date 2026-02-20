#!/bin/bash
# Stage 1: Layout Training (70K iterations)

echo "=========================================="
echo "Stage 1: Layout Training"
echo "=========================================="
echo ""
echo "Goal: Learn layout from scratch"
echo "Trainable: alpha_predictor, instance_fusion_mlp, layout_head, assemble_attn_lora"
echo "Iterations: 70,000"
echo "Learning Rate: 1e-4"
echo ""

# Set CUDA memory optimization flags
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training
accelerate launch --mixed_precision=bf16 train.py \
    --config configs/stage1.yaml

echo ""
echo "✓ Stage 1 training completed!"
echo "Checkpoint: runs/checkpoints/stage1/checkpoint_70000"
echo ""
echo "To monitor training:"
echo "  tensorboard --logdir=runs/logs --port=6006"
