#!/bin/bash
# Training script for UNO hybrid layout + LoRA model
#
# Usage:
#   ./run_phases.sh          # Run all 3 phases (full training)
#   ./run_phases.sh debug    # Run Phase 1 only (for bbox debugging)
#
# Phase 1: Layout-only training with bbox functionality
#   - Trains layout parameters (layout_net + layout attention layers)
#   - Bbox-masked loss for spatial gradients
#   - Layout fires every step (current_step_ratio=0.0)
#   - TensorBoard shows bbox overlays on generated images
#
# Phase 2: LoRA-only training (no layout)
#   - Trains LoRA parameters for VAE reference conditioning
#   - Loads Phase 1 layout weights but doesn't train them
#
# Phase 3: Joint fine-tuning
#   - Trains both layout and LoRA parameters together
#   - Layout fires ~30% of time (current_step_ratio random)
#   - Separate learning rates for layout (0.3x) and LoRA (1.0x)
#
# See updates_mds/ps3_layout_fix.md for implementation details

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use only CUDA devices 2 and 4 (2 GPUs)
export CUDA_VISIBLE_DEVICES=6,7
NUM_GPUS=2

# Mode selection: Set to "debug" to run Phase 1 only with more frequent logging
# Set to "full" to run all 3 phases
MODE="${1:-full}"

# Training configuration
# See updates_mds/ps3_layout_fix.md for details on Phase 1 bbox functionality
echo "=== TRAINING CONFIGURATION ==="
echo "Mode: ${MODE}"
echo ""
echo "Phase 1: Layout-only training with bbox functionality"
echo "  - Bbox-masked loss for spatial gradients"
echo "  - Layout fires every step (current_step_ratio=0.0)"
echo "  - TensorBoard visualization with bbox overlays"
echo "Phase 2: LoRA-only training (no layout)"
echo "Phase 3: Joint fine-tuning (layout + LoRA)"
echo ""

# Set steps based on mode
if [ "$MODE" = "debug" ]; then
  # Debug mode: Phase 1 only with frequent checkpoints and logging
  PHASE1_STEPS=2000
  PHASE1_CKPT_STEPS=500
  LOG_IMAGE_FREQ=250
  echo "Debug mode: Running Phase 1 only (${PHASE1_STEPS} steps)"
  echo "Frequent logging for bbox debugging (every ${LOG_IMAGE_FREQ} steps)"
else
  # Full mode: All 3 phases
  PHASE1_STEPS=5000
  PHASE1_CKPT_STEPS=1000
  LOG_IMAGE_FREQ=500
  echo "Full mode: Running all 3 phases"
fi

PHASE1_DIR="log/phase1"
PHASE2_STEPS=8000
PHASE2_CKPT_STEPS=2000
PHASE2_DIR="log/phase2"
PHASE2_PHASE1_CKPT="${PHASE1_DIR}/checkpoint-${PHASE1_STEPS}/hybrid_lora_layout.safetensors"
PHASE3_STEPS=15000
PHASE3_CKPT_STEPS=2000
PHASE3_DIR="log/phase3"
PHASE3_PHASE1_CKPT="${PHASE1_DIR}/checkpoint-${PHASE1_STEPS}/hybrid_lora_layout.safetensors"
PHASE3_PHASE2_CKPT="${PHASE2_DIR}/checkpoint-${PHASE2_STEPS}/hybrid_lora_layout.safetensors"
PHASE3_LAYOUT_WARMUP=1000

# Create phase configs inline (overrides for test/full)
mkdir -p configs

# Phase 1: Layout-only training with bbox functionality
# - grounding_ratio: Controls when layout fires (0.3 = first 30% of denoising)
# - layout_scale: Strength of layout conditioning (2.0 for strong initial training)
# - current_step_ratio: Always 0.0 in Phase 1 (layout fires every step)
# - Bbox-masked loss: Added in train_hybrid.py for spatial gradients
# - Visualization: TensorBoard shows gen_with_layout, path_a_refs, path_b_refs
cat > configs/phase1_layout_run.json << EOF
{
  "training_phase": 1,
  "max_vae_refs": 0,
  "max_clip_objs": 50,
  "grounding_ratio": 0.3,
  "layout_scale": 2.0,
  "layout_warmup_steps": 0,
  "layout_lr_multiplier": 1.0,
  "layout_double_blocks": [0, 2, 4, 6, 8, 10, 12],
  "layout_single_blocks": [0],
  "learning_rate": 0.0002,
  "max_train_steps": ${PHASE1_STEPS},
  "checkpointing_steps": ${PHASE1_CKPT_STEPS},
  "project_dir": "${PHASE1_DIR}",
  "log_dir": "${PHASE1_DIR}",
  "log_image_freq": ${LOG_IMAGE_FREQ}
}
EOF

# Phase 2: LoRA-only training (no layout)
# - Loads Phase 1 layout weights but doesn't train them
# - Trains only LoRA parameters for VAE reference conditioning
# - layout_scale: 0.0 (layout disabled)
cat > configs/phase2_vae_run.json << EOF
{
  "training_phase": 2,
  "phase1_ckpt": "${PHASE2_PHASE1_CKPT}",
  "max_vae_refs": 4,
  "max_clip_objs": 50,
  "layout_scale": 0.0,
  "grounding_ratio": 0.0,
  "layout_double_blocks": [],
  "layout_single_blocks": [],
  "layout_lr_multiplier": 1.0,
  "learning_rate": 8e-5,
  "max_train_steps": ${PHASE2_STEPS},
  "checkpointing_steps": ${PHASE2_CKPT_STEPS},
  "project_dir": "${PHASE2_DIR}",
  "log_dir": "${PHASE2_DIR}",
  "log_image_freq": ${LOG_IMAGE_FREQ}
}
EOF

# Phase 3: Joint fine-tuning (layout + LoRA)
# - Loads both Phase 1 layout weights and Phase 2 LoRA weights
# - Trains both together with separate learning rates
# - current_step_ratio: random.uniform(0.0, 1.0) - layout fires ~30% of time
# - layout_warmup_steps: Gradually increase layout_scale from 0 to 1.0
# - layout_lr_multiplier: 0.3x lower LR for layout params (already trained in Phase 1)
# For debugging Phase 3, can set grounding_ratio=1.0 to make layout fire every step
cat > configs/phase3_joint_run.json << EOF
{
  "training_phase": 3,
  "phase1_ckpt": "${PHASE3_PHASE1_CKPT}",
  "phase2_ckpt": "${PHASE3_PHASE2_CKPT}",
  "max_vae_refs": 2,
  "max_clip_objs": 50,
  "grounding_ratio": 0.3,
  "layout_scale": 1.0,
  "mixed_precision": "bf16",
  "layout_warmup_steps": ${PHASE3_LAYOUT_WARMUP},
  "layout_lr_multiplier": 0.3,
  "layout_double_blocks": [0, 2, 4, 6, 8, 10, 12],
  "layout_single_blocks": [0],
  "learning_rate": 8e-5,
  "max_train_steps": ${PHASE3_STEPS},
  "checkpointing_steps": ${PHASE3_CKPT_STEPS},
  "project_dir": "${PHASE3_DIR}",
  "log_dir": "${PHASE3_DIR}",
  "log_image_freq": ${LOG_IMAGE_FREQ}
}
EOF

echo ""
echo "=== Phase 1: Layout only (${PHASE1_STEPS} steps) ==="
echo "Training layout parameters with bbox functionality"
echo "Monitor TensorBoard for:"
echo "  - train/loss: Global MSE loss"
echo "  - train/bbox_loss: Bbox-specific loss (should decrease faster)"
echo "  - gen_with_layout: Generated images with green bbox overlays"
echo "  - path_b_refs: CLIP crops (what should appear in boxes)"
echo ""
accelerate launch --num_processes ${NUM_GPUS} train_hybrid.py --config configs/phase1_layout_run.json

if [ "$MODE" = "debug" ]; then
  echo ""
  echo "=== Debug mode: Phase 1 complete ==="
  echo "Phase 1 logs: ${PHASE1_DIR}"
  echo ""
  echo "Next steps:"
  echo "  1. View results: tensorboard --logdir log"
  echo "  2. Check train/bbox_loss is decreasing"
  echo "  3. Compare gen_with_layout with path_b_refs"
  echo "  4. Run debug script: python debug_layout.py"
  echo ""
  echo "See updates_mds/ps3_layout_fix.md for implementation details"
  exit 0
fi

echo ""
echo "=== Phase 2: LoRA only (${PHASE2_STEPS} steps) ==="
echo "Training LoRA parameters for VAE reference conditioning"
echo ""
accelerate launch --num_processes ${NUM_GPUS} train_hybrid.py --config configs/phase2_vae_run.json

echo ""
echo "=== Phase 3: Joint fine-tuning (${PHASE3_STEPS} steps) ==="
echo "Fine-tuning both layout and LoRA parameters together"
echo ""
accelerate launch --num_processes ${NUM_GPUS} train_hybrid.py --config configs/phase3_joint_run.json

echo ""
echo "=== All 3 phases complete ==="
echo "Phase 1 logs: ${PHASE1_DIR}"
echo "Phase 2 logs: ${PHASE2_DIR}"
echo "Phase 3 logs: ${PHASE3_DIR}"
echo ""
echo "View results:"
echo "  tensorboard --logdir log"
echo ""
echo "Debug layout attention:"
echo "  python debug_layout.py"
echo ""
echo "See updates_mds/ps3_layout_fix.md for implementation details"
