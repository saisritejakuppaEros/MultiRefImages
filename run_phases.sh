#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use only CUDA devices 4, 5, 6, 7 (4 GPUs)
export CUDA_VISIBLE_DEVICES=4,5,6,7
NUM_GPUS=4

# Full run
echo "=== FULL RUN ==="
PHASE1_STEPS=5000
PHASE1_CKPT_STEPS=1000
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
LOG_IMAGE_FREQ=250

# Create phase configs inline (overrides for test/full)
mkdir -p configs
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

echo "=== Phase 1: Layout only (${PHASE1_STEPS} steps) ==="
accelerate launch --num_processes ${NUM_GPUS} train_hybrid.py --config configs/phase1_layout_run.json

echo "=== Phase 2: LoRA only (${PHASE2_STEPS} steps) ==="
accelerate launch --num_processes ${NUM_GPUS} train_hybrid.py --config configs/phase2_vae_run.json

echo "=== Phase 3: Joint fine-tuning (${PHASE3_STEPS} steps) ==="
accelerate launch --num_processes ${NUM_GPUS} train_hybrid.py --config configs/phase3_joint_run.json

echo "=== All 3 phases complete ==="
echo "Phase 1 logs: ${PHASE1_DIR}"
echo "Phase 2 logs: ${PHASE2_DIR}"
echo "Phase 3 logs: ${PHASE3_DIR}"
echo "TensorBoard: tensorboard --logdir log"
