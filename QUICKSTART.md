# MoviePostProduction - Quick Start

## 1. Prepare Data (One-time setup)

```bash
python prepare_synthetic_dataset.py
```

This generates:
- CLIP embeddings for all object crops
- DINO embeddings for all object crops  
- VAE latents for all full images

## 2. Train Model (3 Stages)

### Option A: Run all stages automatically
```bash
bash train.sh
```

### Option B: Run stages individually
```bash
# Stage 1: Layout training (70K iters, ~10-15 hours)
bash train_stage1.sh

# Stage 2: Identity training (150K iters, ~20-30 hours)
bash train_stage2.sh

# Stage 3: Feedback fine-tuning (10K iters, ~1-2 hours)
bash train_stage3.sh
```

## 3. Monitor Training

```bash
tensorboard --logdir=runs/logs --port=6006
```

Open: `http://localhost:6006`

## 4. Key Files

- **Configs**: `configs/stage{1,2,3}.yaml`
- **Training script**: `train.py`
- **Dataset**: `data_loader.py`
- **Model**: `model.py`
- **Checkpoints**: `runs/checkpoints/stage{1,2,3}/`
- **Logs**: `runs/logs/`

## 5. What to Watch in TensorBoard

### Stage 1
- ✅ `loss/total` decreasing
- ✅ `metrics/alpha_mean` stabilizing (~0.5)
- ✅ Bbox overlays showing correct placement

### Stage 2
- ✅ `loss/dino_identity` decreasing
- ✅ FG objects looking like references
- ✅ Alpha distribution stable

### Stage 3
- ✅ `metrics/feedback_magnitude` small but non-zero
- ✅ Slight quality improvements
- ✅ Layout remains stable

## 6. Troubleshooting

**Out of memory?**
- Edit config: reduce `batch_size` or increase `gradient_accumulation_steps`

**Checkpoint corrupted?**
- Use previous checkpoint: `bash train_stage2.sh runs/checkpoints/stage1/checkpoint_60000`

**TensorBoard empty?**
- Wait for inference step (every 500 iterations)
- Check: `ls runs/logs/stage1_*/`

## 7. Final Model

After all stages complete:
```
runs/checkpoints/stage3/checkpoint_10000/model.pt
```

This is your trained model ready for inference!
