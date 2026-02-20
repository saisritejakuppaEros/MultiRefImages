# MoviePostProduction Training Guide

## Overview

This guide explains how to train the MoviePostProduction model using the 3-stage training pipeline.

## Prerequisites

1. **Prepared Dataset**: Run `prepare_synthetic_dataset.py` to generate:
   - Images in `output_data/images/`
   - Caption JSONs in `output_data/captions/`
   - CLIP embeddings in `output_data/embedding/clip_cropped_boxes/`
   - DINO embeddings in `output_data/embedding/dino_cropped_boxes/`
   - VAE latents in `output_data/latents/vae_full_images/`

2. **Dependencies**: Install required packages:
```bash
pip install torch torchvision accelerate transformers diffusers peft tensorboard pyyaml pillow matplotlib
```

## Training Pipeline

### Quick Start - Run All 3 Stages

```bash
# Run all stages sequentially
bash train.sh
```

### Individual Stage Training

#### Stage 1: Layout Training (70K iterations)
**Goal**: Learn object placement from scratch

```bash
bash train_stage1.sh
```

**What's trained**:
- Alpha predictor (routing between layout/modulation)
- Instance encoder (spatial + visual features)
- Layout head (Assemble-Attn)
- LoRA adapters on FLUX attention (rank 128)

**Frozen**: Modulation head, feedback bridge, FLUX backbone

**Config**: `configs/stage1.yaml`
- Learning rate: 1e-4
- Batch size: 2, grad accumulation: 4 (effective batch=8)
- Checkpoints saved every 100 steps
- Inference visualization every 500 steps

---

#### Stage 2: Modulation + Identity Training (150K iterations)
**Goal**: Add identity preservation on top of layout

```bash
# Auto-detect latest Stage 1 checkpoint
bash train_stage2.sh

# Or specify checkpoint manually
bash train_stage2.sh runs/checkpoints/stage1/checkpoint_70000
```

**What's trained** (in addition to Stage 1):
- Modulation head (AdaLN offsets)
- Per-block AdaLN projections

**Config**: `configs/stage2.yaml`
- Learning rate: 5e-6 (lower for stability)
- DINO identity loss added (weight=10.0)
- Resumes from Stage 1 checkpoint

---

#### Stage 3: Feedback Bridge Fine-tuning (10K iterations)
**Goal**: Fine-tune cross-block communication

```bash
# Auto-detect latest Stage 2 checkpoint
bash train_stage3.sh

# Or specify checkpoint manually
bash train_stage3.sh runs/checkpoints/stage2/checkpoint_150000
```

**What's trained**: Feedback bridge ONLY

**Frozen**: Everything else (all Stage 1 & 2 modules)

**Config**: `configs/stage3.yaml`
- Learning rate: 1e-5
- Short fine-tuning stage
- Resumes from Stage 2 checkpoint

---

## Monitoring Training

### TensorBoard

Start TensorBoard to monitor training progress:

```bash
tensorboard --logdir=runs/logs --port=6006
```

Then open `http://localhost:6006` in your browser.

### What to Monitor

#### Stage 1
- **Loss curves**: `loss/total`, `loss/reconstruction` should decrease
- **Alpha distribution**: `metrics/alpha_mean` should stabilize around 0.5
- **FG/BG ratio**: `metrics/alpha_fg_ratio` shows routing decisions
- **Inference images**: Ground truth images with bbox overlays
  - Red boxes = Foreground (high alpha)
  - Blue boxes = Background (low alpha)

#### Stage 2
- **DINO identity loss**: `loss/dino_identity` should decrease
- **Identity similarity**: `metrics/dino_similarity_mean` should increase
- **Alpha stability**: Distribution should remain stable from Stage 1
- **Inference images**: FG objects should look more like references

#### Stage 3
- **Feedback magnitude**: `metrics/feedback_magnitude` should be small but non-zero
- **Loss stability**: Losses should remain stable or slightly improve
- **Identity quality**: Slight improvements in FG object quality

---

## Directory Structure

```
moviepostproduction/
├── configs/
│   ├── stage1.yaml          # Stage 1 hyperparameters
│   ├── stage2.yaml          # Stage 2 hyperparameters
│   └── stage3.yaml          # Stage 3 hyperparameters
├── train.py                 # Main training script
├── train.sh                 # Run all 3 stages
├── train_stage1.sh          # Run Stage 1 only
├── train_stage2.sh          # Run Stage 2 only
├── train_stage3.sh          # Run Stage 3 only
├── data_loader.py           # Dataset and DataLoader
├── apply_lora.py            # LoRA utilities
├── model.py                 # Model architecture
└── runs/
    ├── logs/                # TensorBoard logs
    │   ├── stage1_YYYYMMDD_HHMMSS/
    │   ├── stage2_YYYYMMDD_HHMMSS/
    │   └── stage3_YYYYMMDD_HHMMSS/
    └── checkpoints/         # Model checkpoints
        ├── stage1/
        │   ├── checkpoint_100/
        │   ├── checkpoint_200/
        │   └── ...
        ├── stage2/
        └── stage3/
```

---

## Checkpoints

### Checkpoint Contents

Each checkpoint directory contains:
- `model.pt` - Model state dict (all trainable parameters)
- `optimizer.pt` - Optimizer state
- `scheduler.pt` - Learning rate scheduler state
- `config.yaml` - Training configuration
- `training_state.json` - Iteration counter and metadata

### Resuming Training

If training is interrupted, resume from the latest checkpoint:

```bash
# Find latest checkpoint
ls -td runs/checkpoints/stage1/checkpoint_* | head -1

# Resume training
python train.py --config configs/stage1.yaml --resume_from_checkpoint runs/checkpoints/stage1/checkpoint_50000
```

Or use the automatic detection in the bash scripts:
```bash
bash train_stage2.sh  # Automatically finds latest Stage 1 checkpoint
```

---

## Configuration

### Key Config Parameters

Edit YAML files to adjust training:

**Training Hyperparameters**:
- `training.num_iterations` - Total training steps
- `training.learning_rate` - Learning rate
- `training.batch_size` - Batch size per GPU
- `training.gradient_accumulation_steps` - Gradient accumulation

**Logging**:
- `logging.log_every_n_steps` - Frequency for loss logging (default: 100)
- `inference.every_n_steps` - Frequency for inference visualization (default: 500)
- `inference.num_samples` - Number of validation samples to visualize (default: 4)

**Checkpointing**:
- `checkpointing.save_every_n_steps` - Checkpoint frequency (default: 100)
- `checkpointing.keep_last_n_checkpoints` - Number of checkpoints to keep (default: 5)

**Loss Weights**:
- `loss_weights.flow_matching` - Flow matching loss weight (default: 1.0)
- `loss_weights.dino_identity` - DINO identity loss weight (default: 10.0 in Stage 2/3)

---

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `training.batch_size` in config
   - Increase `training.gradient_accumulation_steps`
   - Enable `training.gradient_checkpointing: true`

2. **Corrupted Checkpoint**
   - Training will automatically skip corrupted checkpoints
   - Use a different checkpoint: `bash train_stage2.sh runs/checkpoints/stage1/checkpoint_60000`

3. **TensorBoard Not Showing Images**
   - Check that inference visualization is enabled: `inference.enabled: true`
   - Wait for next inference step (every 500 iterations)
   - Check TensorBoard logs directory: `runs/logs/`

4. **LoRA Not Applied**
   - Check that `trainable.assemble_attn_lora: true` in config
   - Verify LoRA rank/alpha settings in `lora` section

---

## Expected Training Time

With 8x H100 GPUs (80GB each):
- **Stage 1**: ~10-15 hours (70K iterations)
- **Stage 2**: ~20-30 hours (150K iterations)
- **Stage 3**: ~1-2 hours (10K iterations)
- **Total**: ~35-50 hours

With fewer GPUs or lower memory, training will take proportionally longer.

---

## Output

After training completes, you'll have:
- **Final model**: `runs/checkpoints/stage3/checkpoint_10000/model.pt`
- **Training logs**: `runs/logs/stage{1,2,3}_*/`
- **Intermediate checkpoints**: Every 100-10000 steps depending on stage

Use the final model for inference to generate images with precise object placement and identity preservation.
