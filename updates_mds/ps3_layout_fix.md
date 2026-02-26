# Layout Conditioning Fix Plan — Phase 1 Focus

## Overview

This document provides fixes to enable full bbox/layout functionality in **Phase 1** (layout-only training) for early debugging and validation. Phase 3 (joint fine-tuning) remains unchanged for later use.

## Current State

**Phase 1** (layout-only training):
- ✅ `current_step_ratio = 0.0` — layout fires every step
- ✅ `grounding_ratio` passed to model
- ✅ Bbox visualization in TensorBoard (green boxes on generated images)
- ❌ Missing: bbox-masked loss
- ❌ Missing: cross-attention fix
- ❌ Missing: debug script

**Phase 3** (joint fine-tuning):
- Uses `current_step_ratio = random.uniform(0.0, 1.0)` — layout fires ~30% of time
- Correct for joint training, but makes debugging hard
- Will benefit from same fixes once Phase 1 is validated

---

## Problems Found

1. **No spatial loss** — global MSE only, model has zero incentive to put correct content inside boxes
2. **Self-attention instead of cross-attention** — obj_token and region_tokens attend to each other equally, obj_token has no special conditioning role
3. **No way to debug** — no unit test to verify layout attention is actually doing anything
4. **Batch size assumption not enforced** — code assumes batch_size=1 but doesn't assert it

---

## Fix 1 — Phase 1 Setup (Already Done)

**File: `train_hybrid.py`**

Phase 1 already has the correct setup for bbox debugging:

```python
# Line 425: Layout always fires in Phase 1
if args.training_phase == 1:
    current_step_ratio = 0.0
```

```python
# Line 456: grounding_ratio passed to model
grounding_ratio=args.grounding_ratio,
```

```python
# Lines 550-554: Bbox visualization in TensorBoard
all_bboxes = [c.bbox for c in hybrid_layout.vae_conds] + [c.bbox for c in hybrid_layout.clip_conds]
if all_bboxes:
    out_with_bbox = _draw_bboxes_on_tensor(out_img.cpu(), all_bboxes)
    tb_writer.add_image("gen_with_layout", out_with_bbox, global_step)
```

**What you get in TensorBoard:**
- `gen_with_layout`: Generated images with green bbox overlays
- `path_a_refs`: VAE reference images (Path A)
- `path_b_refs`: CLIP crop images (Path B) — what should appear in each box

---

## Fix 2 — Bbox-Masked Loss for Phase 1

**File: `train_hybrid.py`**

**Location:** After line 461 (after computing main loss)

Add a second loss term computed only on tokens inside each bbox:

```python
# Compute bbox-masked loss for Phase 1
bbox_loss = 0.0
if args.training_phase == 1 and layout_kw is not None:
    layout_data = layout_kw.get("layout")
    if layout_data is not None:
        img_idxs_list = layout_data["img_idxs_list"]
        masks = layout_data["masks"][0]  # batch_size=1
        valid_objs = (masks == 1).nonzero(as_tuple=False).squeeze(-1)
        
        for obj_idx in valid_objs:
            obj_idx_val = obj_idx.item()
            if obj_idx_val < len(img_idxs_list):
                img_idxs = img_idxs_list[obj_idx_val]
                if img_idxs.numel() > 0:
                    bbox_loss += F.mse_loss(
                        model_pred[:, img_idxs], 
                        target[:, img_idxs]
                    )
        
        if len(valid_objs) > 0:
            bbox_loss = bbox_loss / len(valid_objs)
            loss = loss + 0.5 * bbox_loss
```

**Logging:** Add to the TensorBoard logging section (around line 483):

```python
# Log bbox loss separately for Phase 1
if args.training_phase == 1 and layout_kw is not None and tb_writer is not None:
    tb_writer.add_scalar("train/bbox_loss", bbox_loss, global_step)
```

**What this does:**
- Computes MSE loss only on tokens inside each bbox
- Weighted at 0.5x to balance with global loss
- Gives gradient a spatial anchor — model learns to put correct content in boxes
- Logged separately so you can track bbox convergence independently

**Expected behavior:**
- `train/bbox_loss` should decrease faster than `train/loss`
- This confirms the model is learning spatial layout

---

## Fix 3 — Cross-Attention in Layout Block

**File: `uno/flux/modules/layers.py`**

**Location:** Lines 365-382 in `DoubleStreamBlock.forward()` (and similar in `SingleStreamBlock.forward()`)

**Current (wrong):**
```python
context = torch.cat([obj_token, region_tokens], dim=0).unsqueeze(0)
# runs full self-attention over [obj_token + region_tokens]
q_h = self.layout_q(context_norm).reshape(1, L, H, d // H).transpose(1, 2)
k_h = self.layout_k(context_norm).reshape(1, L, H, d // H).transpose(1, 2)
v_h = self.layout_v(context_norm).reshape(1, L, H, d // H).transpose(1, 2)
attn_out = (torch.softmax((q_h @ k_h.transpose(-2, -1)) * scale_factor, dim=-1) @ v_h)
obj_out = attn_out[:, :1]
region_out = attn_out[:, 1:]
```

**Change to cross-attention:**

```python
# Separate obj_token and region_tokens for cross-attention
obj_token_norm = context_norm[:, :1]  # [1, 1, d]
region_tokens_norm = context_norm[:, 1:]  # [1, num_region, d]

# Q from region tokens, K/V from obj_token (cross-attention)
q_h = self.layout_q(region_tokens_norm).reshape(1, -1, H, d // H).transpose(1, 2)
k_h = self.layout_k(obj_token_norm).reshape(1, 1, H, d // H).transpose(1, 2)
v_h = self.layout_v(obj_token_norm).reshape(1, 1, H, d // H).transpose(1, 2)

# Region tokens attend TO obj_token (conditioning)
attn_out = (torch.softmax((q_h @ k_h.transpose(-2, -1)) * scale_factor, dim=-1) @ v_h)
region_out = attn_out.transpose(1, 2).reshape(1, -1, d)

# Remove obj_out computation - it's unused and wastes compute
```

**Then update the output section (lines 384-399):**

```python
# Only apply delta to region tokens
delta_region = (
    layout_mod_out.gate * layout_scale * self.layout_out(region_out.squeeze(0))
).squeeze(0)
hidden_states_add[i].scatter_add_(
    0,
    img_idxs.unsqueeze(-1).expand(-1, img.shape[-1]),
    delta_region,
)
density_map[i].scatter_add_(
    0,
    img_idxs.unsqueeze(-1),
    torch.ones(img_idxs.shape[0], 1, device=img.device, dtype=img.dtype),
)
# Remove layout_hidden_add update - obj_token doesn't need updating
```

**Why this matters:**
- Current: obj_token and region_tokens are peers in self-attention
- Fixed: region_tokens attend TO obj_token as conditioning signal
- Result: CLIP embedding properly conditions the bbox region

**Apply to both:**
- `DoubleStreamBlock.forward()` (lines 340-405)
- `SingleStreamBlock.forward()` (lines 528-595)

---

## Fix 4 — Debug Script for Layout Attention

**File: `debug_layout.py` (new file in UNO directory)**

Create a script that verifies layout attention is working:

```python
#!/usr/bin/env python3
"""
Debug script to verify layout attention is functioning.
Run this before training to establish baseline, and after cross-attention fix to verify improvement.
"""

import torch
import torch.nn.functional as F
from uno.flux.model import Flux
from uno.flux.modules.layout import HybridLayout, ObjectCond
from uno.flux.util import configs
import numpy as np

def debug_layout_attention():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    # Load model
    print("Loading model...")
    config = configs["flux-dev"]
    model = Flux(config).to(device).to(dtype)
    
    # Enable layout on only block 0 for isolated testing
    model.enable_layout(double_block_indices=[0], single_block_indices=[])
    print("Layout enabled on double_block[0] only")
    
    # Create test sample: one bbox with random CLIP embedding
    print("\nPreparing test sample...")
    batch_size = 1
    height, width = 512, 512
    latent_h, latent_w = height // 8, width // 8
    seq_len = latent_h * latent_w
    
    # Random image tokens
    img = torch.randn(batch_size, seq_len, config.hidden_size, device=device, dtype=dtype)
    
    # Random text tokens
    txt = torch.randn(batch_size, 256, config.hidden_size, device=device, dtype=dtype)
    
    # Random vec (pooled prompt embedding)
    vec = torch.randn(batch_size, config.vec_in_dim, device=device, dtype=dtype)
    
    # Create bbox: center 50% of image
    bbox = [0.25, 0.25, 0.75, 0.75]
    clip_embedding = torch.randn(768, device=device, dtype=dtype)
    
    # Build layout
    obj_cond = ObjectCond(bbox=bbox, hw=[height, width], tier="clip", embedding=clip_embedding)
    hybrid_layout = HybridLayout([obj_cond], max_objs=50)
    layout_kwargs = hybrid_layout.build_layout_kwargs(device, dtype, (latent_h, latent_w))
    
    # Extract layout data
    layout_data = layout_kwargs["layout"]
    layout_hidden_states = model.layout_net(
        boxes=layout_data["boxes"],
        masks=layout_data["masks"],
        embeddings=layout_data["embeddings"],
    )
    img_idxs_list = [layout_data["img_idxs_list"]]
    layout_masks = layout_data["masks"]
    bbox_idxs = layout_data["img_idxs_list"][0]
    
    print(f"Bbox covers {bbox_idxs.numel()} tokens out of {seq_len} total")
    
    # Capture img before layout block
    img_before = img.clone()
    
    # Run through block 0 with layout
    block = model.double_blocks[0]
    
    # Need position embeddings
    img_ids = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
    for h_idx in range(latent_h):
        for w_idx in range(latent_w):
            idx = h_idx * latent_w + w_idx
            img_ids[0, idx, 1] = h_idx
            img_ids[0, idx, 2] = w_idx
    
    txt_ids = torch.zeros(batch_size, 256, 3, device=device, dtype=dtype)
    ids = torch.cat([txt_ids, img_ids], dim=1)
    pe = model.pe_embedder(ids)
    
    with torch.no_grad():
        # Get layout_mod_out.gate value for debugging
        layout_mod_out, _ = block.layout_mod(vec)
        gate_value = layout_mod_out.gate.item()
        print(f"\nlayout_mod_out.gate = {gate_value:.6f}")
        
        # Run forward
        img_out, txt_out, _ = block(
            img=img,
            txt=txt,
            vec=vec,
            pe=pe,
            layout_hidden_states=layout_hidden_states,
            layout_masks=layout_masks,
            img_idxs_list=img_idxs_list,
            layout_scale=1.0,
        )
    
    # Compute delta at bbox indices
    delta = img_out - img_before
    delta_bbox = delta[0, bbox_idxs]  # [num_bbox_tokens, hidden_size]
    
    # Check if delta is near zero
    delta_norm = delta_bbox.norm(dim=-1).mean().item()
    print(f"Mean delta norm at bbox: {delta_norm:.6f}")
    
    if delta_norm < 1e-6:
        print("⚠️  Delta is near zero! Layout attention is not affecting the bbox region.")
        print("   Possible causes: layout_scale too low, gate suppressing, or grounding_ratio blocking")
        return
    
    # Compute cosine similarity between delta and CLIP embedding
    # Project CLIP embedding to hidden_size for comparison
    layout_hidden_obj = layout_hidden_states[0, 0]  # [hidden_size]
    
    # Average delta across bbox tokens
    delta_avg = delta_bbox.mean(dim=0)  # [hidden_size]
    
    # Cosine similarity
    similarity = F.cosine_similarity(delta_avg.unsqueeze(0), layout_hidden_obj.unsqueeze(0)).item()
    print(f"Cosine similarity (delta vs layout_hidden): {similarity:.6f}")
    
    # Interpretation
    print("\n" + "="*60)
    if similarity > 0.1:
        print("✅ Layout attention is working correctly!")
        print("   Delta correlates with CLIP embedding.")
    elif similarity < 0.05:
        print("⚠️  Layout attention is firing but embedding signal is weak.")
        print("   Cross-attention fix may be needed.")
    else:
        print("⚙️  Layout attention shows some correlation.")
        print("   Monitor bbox_loss during training to confirm.")
    print("="*60)

if __name__ == "__main__":
    debug_layout_attention()
```

**How to use:**

1. **Before any changes:** Run to establish baseline
   ```bash
   cd UNO
   python debug_layout.py
   ```

2. **After cross-attention fix:** Run again to verify improvement
   - Cosine similarity should increase (> 0.1 is good)

3. **Interpretation:**
   - Delta near zero → layout_scale too low or gate suppressing
   - Delta nonzero but similarity ~ 0 → attention firing but embedding not getting through
   - Similarity > 0.1 → layout working correctly

---

## Fix 5 — Batch Size Assert

**File: `uno/flux/model.py`**

**Location:** After line 221 (after `img_idxs_list = [lk["img_idxs_list"]]`)

Add assertion to prevent silent failures:

```python
img_idxs_list = [lk["img_idxs_list"]]
assert img.shape[0] == 1, \
    "Layout conditioning only supports batch_size=1 due to img_idxs_list wrapping"
```

**Why this matters:**
- Current code wraps `img_idxs_list` in a list: `[lk["img_idxs_list"]]`
- In `layers.py`, it accesses `img_idxs_list[i][j]` where `i` is batch index
- For batch_size > 1, `img_idxs_list[1]` doesn't exist → silent failure
- `DenseLayoutDataset` forces batch_size=1, but this makes it explicit

---

## Implementation Order

Apply fixes in this order for Phase 1:

1. **Fix 2** — Add bbox-masked loss to Phase 1
   - Gives spatial gradient signal
   - Log `train/bbox_loss` to TensorBoard

2. **Fix 4** — Create and run debug script
   - Establish baseline before changes
   - Verify layout attention is firing

3. **Fix 3** — Apply cross-attention fix
   - Change self-attention to cross-attention
   - Apply to both DoubleStreamBlock and SingleStreamBlock

4. **Fix 4 (again)** — Re-run debug script
   - Verify cosine similarity improves (> 0.1)
   - Confirms cross-attention fix worked

5. **Fix 5** — Add batch size assert
   - Low risk, prevents silent failures

6. **Train Phase 1** — Monitor TensorBoard
   - Watch `train/bbox_loss` decrease
   - Check `gen_with_layout` images for bbox alignment
   - Compare with `path_b_refs` (CLIP crops)

---

## What Success Looks Like in Phase 1

After implementing all fixes and training Phase 1:

### TensorBoard Metrics
- `train/loss`: Global MSE loss decreasing
- `train/bbox_loss`: Bbox-specific loss decreasing **faster** than global loss
- `train/layout_scale`: Effective layout scale (with warmup in Phase 3)

### TensorBoard Images
- `gen_with_layout`: Generated images with green bbox overlays
  - Content inside boxes should match CLIP crop images
- `path_a_refs`: VAE reference images (Path A)
- `path_b_refs`: CLIP crop images (Path B) — what should appear in boxes
- `target`: Ground truth image

### Debug Script Output
- Cosine similarity > 0.1 between bbox delta and CLIP embedding
- Non-zero `layout_mod_out.gate` value (not suppressing)
- Confirms layout attention is working

### Visual Validation
You can "see things coming in the box" by comparing:
1. `path_b_refs`: What should appear in each box (CLIP crops)
2. `gen_with_layout`: Generated image with bbox overlays
3. Content inside green boxes should match the corresponding CLIP crop

---

## Phase 3 Notes

Phase 3 (joint fine-tuning) remains unchanged for now:
- Uses `current_step_ratio = random.uniform(0.0, 1.0)` 
- Layout fires ~30% of time (when random ≤ grounding_ratio of 0.3)
- Correct for joint training

**When debugging Phase 3 later:**
- Can set `grounding_ratio=1.0` to make layout fire every step
- Use same debug script to verify layout attention
- Watch `train/bbox_loss` converge
- Once validated, tune back to 0.3 for final training

---

## Summary

This plan enables full bbox functionality in Phase 1 for early debugging:

✅ **Already working:** Layout fires every step, bbox visualization in TensorBoard  
🔧 **Add:** Bbox-masked loss for spatial gradients  
🔧 **Add:** Cross-attention fix for proper conditioning  
🔧 **Add:** Debug script to verify attention is working  
🔧 **Add:** Batch size assertion for safety  

After these changes, you'll be able to see bbox content in generated images and verify layout is working before moving to Phase 3.
