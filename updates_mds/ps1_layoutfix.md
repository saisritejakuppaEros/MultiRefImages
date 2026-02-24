# Fix: Layout Boxes Not Conditioning — VAE Dominance

## Root Cause Diagnosis

There are **five separate bugs** compounding to produce the symptoms you see.
They are listed in order of severity.

---

## Bug 1 — CRITICAL: The layout cross-attention is writing back the wrong thing

**File**: `uno/flux/modules/layers.py`, `DoubleStreamBlock.forward` and `SingleStreamBlock.forward`

**What your code does**:
```python
# The object token attends to the region, output is a single [1, hidden_size] vector
q = self.layout_q(context_norm[:, :1])    # [1, 1, hs] — obj token as query
k_ = self.layout_k(context_norm)          # [1, 1+R, hs]
v_ = self.layout_v(context_norm)
attn_out = (attn_w @ v_).squeeze(0)       # [1, hs] — ONE vector

# Then this single vector is EXPANDED and scattered to ALL region pixels
delta_2d = delta.reshape(1, -1).expand(img_idxs.shape[0], -1)  # [R, hs] all identical!
hidden_states_add[i].scatter_add_(...)     # same delta added to every pixel in the box
```

**What the paper says** (Section 3.2, Equation 7-8):

> We crop the image tokens h^z by bbox and get h^z_li. Then we project the cropped image
> tokens h^z_li **and** their corresponding instance tokens l_i into queries **[Q_zli, Q_li]**,
> keys **[K_zli, K_li]**, values **[V_zli, V_li]**, and apply attention. The updated tokens
> are assembled across instances via a **density map (averaging overlaps)**.

The paper's attention is **bidirectional**: both the object token AND every region pixel token
are queries. The output h^z'_li has shape **[R, hs]** — one distinct vector per pixel.
Your implementation outputs **[1, hs]** (only the object token's attended output) and clones
it across all R pixels. This means every pixel in the box gets the same undifferentiated blob
with no spatial structure, which is why the box content looks uniform and the VAE global signal
dominates.

**Fix** in `layers.py` — replace the attention block in both `DoubleStreamBlock` and
`SingleStreamBlock`:

```python
# OLD — wrong: single query, single output, expanded
q  = self.layout_q(context_norm[:, :1])    # [1, 1, hs]
k_ = self.layout_k(context_norm)
v_ = self.layout_v(context_norm)
scale = q.shape[-1] ** -0.5
attn_w  = torch.softmax((q @ k_.transpose(-2, -1)) * scale, dim=-1)
attn_out = (attn_w @ v_).squeeze(0)        # [1, hs]
delta    = layout_scale * self.layout_out(attn_out)
delta_2d = delta.reshape(1, -1).expand(img_idxs.shape[0], -1)  # BUG: [R, hs] all same

# NEW — correct: all tokens as queries, write back only the region outputs
# context = [obj_token | region_tokens]  shape [1, 1+R, hs]
context_norm = self.layout_norm(context)
q_all = self.layout_q(context_norm)        # [1, 1+R, hs]  ALL tokens query
k_all = self.layout_k(context_norm)
v_all = self.layout_v(context_norm)
scale = (q_all.shape[-1] // self.num_layout_heads) ** -0.5  # per-head scale

# Reshape for multi-head
H = self.num_layout_heads
q_h = q_all.reshape(1, q_all.shape[1], H, -1).transpose(1, 2)  # [1, H, 1+R, d]
k_h = k_all.reshape(1, k_all.shape[1], H, -1).transpose(1, 2)
v_h = v_all.reshape(1, v_all.shape[1], H, -1).transpose(1, 2)

attn_w   = torch.softmax((q_h @ k_h.transpose(-2, -1)) * scale, dim=-1)
attn_out = (attn_w @ v_h).transpose(1, 2).reshape(1, -1, q_all.shape[-1])  # [1, 1+R, hs]

# Split: obj output is first token, region outputs are the rest
obj_out    = attn_out[:, :1]               # [1, 1, hs]
region_out = attn_out[:, 1:]               # [1, R, hs]  DISTINCT per pixel

# Write region outputs back to their img positions
delta_region = layout_scale * self.layout_out(region_out.squeeze(0))  # [R, hs]
hidden_states_add[i].scatter_add_(
    0,
    img_idxs.unsqueeze(-1).expand(-1, img.shape[-1]),
    delta_region,   # [R, hs] — unique per pixel, NOT expanded
)
layout_hidden_add[i, j] = layout_scale * self.layout_out(obj_out.squeeze(0)).squeeze(0)
```

Also add `num_layout_heads` to `__init__` of both block classes (8 heads works, must divide
`hidden_size=3072`):

```python
# In __init__ of DoubleStreamBlock and SingleStreamBlock, replace the layout Q/K/V setup:
self.num_layout_heads = 8   # 3072 / 8 = 384 per head
self.layout_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
self.layout_q    = nn.Linear(hidden_size, hidden_size, bias=True)
self.layout_k    = nn.Linear(hidden_size, hidden_size, bias=True)
self.layout_v    = nn.Linear(hidden_size, hidden_size, bias=True)
self.layout_out  = zero_module(nn.Linear(hidden_size, hidden_size, bias=True))
```

---

## Bug 2 — CRITICAL: The density map averaging is missing (overlapping boxes all add up)

**File**: `uno/flux/modules/layers.py`

**What your code does**:
```python
hidden_states_add[i].scatter_add_(0, img_idxs..., delta_region)
# called for every object. If bbox A and bbox B overlap, pixel P gets delta_A + delta_B
# No normalization. Overlapping regions get 2x, 3x, Nx the signal.
```

**What the paper says** (Equation 8):
```
h^z'[:, i, j] = (1 / M[i,j]) * sum_k( h^z'_lk[:, i, j] )
```
where `M[i,j]` is the **density map** — number of objects whose bbox covers pixel (i,j).
All writes to a pixel are averaged, not summed. Without this, dense overlapping boxes
produce unbounded accumulation that swamps the VAE signal and creates artifacts.

**Fix**: Accumulate a count tensor alongside the add, then normalize after all objects
are processed:

```python
# In the layout branch of both DoubleStreamBlock and SingleStreamBlock forward:
bsz = img.shape[0]
hidden_states_add  = torch.zeros_like(img)
density_map        = torch.zeros(bsz, img.shape[1], 1, device=img.device, dtype=img.dtype)
layout_hidden_add  = torch.zeros_like(layout_hidden_states)
valid = (layout_masks == 1).nonzero(as_tuple=False)

for k in range(valid.shape[0]):
    i = valid[k, 0].item()
    j = valid[k, 1].item()
    # ... (compute region_out per Bug 1 fix) ...
    delta_region = layout_scale * self.layout_out(region_out.squeeze(0))  # [R, hs]
    hidden_states_add[i].scatter_add_(
        0, img_idxs.unsqueeze(-1).expand(-1, img.shape[-1]), delta_region
    )
    # Count how many times each pixel receives a write
    ones = torch.ones(img_idxs.shape[0], 1, device=img.device, dtype=img.dtype)
    density_map[i].scatter_add_(0, img_idxs.unsqueeze(-1), ones)

# Normalize accumulated deltas by the density map (avoid div by zero)
density_map = density_map.clamp(min=1.0)
hidden_states_add = hidden_states_add / density_map

img = img + hidden_states_add
layout_hidden_states = layout_hidden_states + layout_hidden_add
```

---

## Bug 3 — SIGNIFICANT: VAE token sequence length causes attention rank collapse

**File**: `uno/flux/model.py`, the VAE token concatenation block

**What happens**: With 5 VAE refs at 320×320, each ref produces `(320/16)^2 = 400` tokens.
Total VAE ref tokens = 2000. Target image at 1024×1024 = 4096 tokens. T5 text = 256 tokens.
Total sequence = **6352 tokens**.

The layout cross-attention in your `DoubleStreamBlock` processes the FULL `img` tensor, which
now contains `[target_tokens | ref_tokens_1 | ref_tokens_2 | ... | ref_tokens_5]`.

Your `img_end` indexing correctly slices outputs back to only the target tokens for the final
layer — but during the layout cross-attention, `img_idxs` addresses the target portion only
(0 to 4095). The problem: the FLUX global attention in the same block now has keys/values from
6352 tokens. The VAE ref tokens are semantically coherent, high-magnitude, and rich — they
dominate the key-value matrix. The model finds it easier to copy from VAE refs than to learn
the layout conditioning, because the gradient path through VAE ref tokens is shorter and
lower-loss from day 1.

**Fix — in `model.py`**: Apply layout conditioning **before** appending VAE ref tokens to
the sequence, on the target portion only. The `img_end` variable already tracks this boundary.
Change the model's forward to run layout conditioning on `img[:, :img_end]` before the ref
tokens are concatenated:

```python
# In Flux.forward, RESTRUCTURE the layout and ref-token injection order:

img = self.img_in(img)
vec = self.time_in(timestep_embedding(timesteps, 256))
if self.params.guidance_embed:
    vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
vec = vec + self.vector_in(y)
txt = self.txt_in(txt)

# ── Track target-only img end BEFORE appending refs ──────────────────────
img_end = img.shape[1]   # number of target image tokens (unchanged)

# ── Compute layout states from layout_net ────────────────────────────────
layout_hidden_states = None
img_idxs_list        = None
layout_masks         = None
effective_layout_scale = 0.0
if (layout_kwargs is not None
    and layout_kwargs.get("layout") is not None
    and current_step_ratio <= grounding_ratio):
    lk = layout_kwargs["layout"]
    layout_hidden_states = self.layout_net(
        boxes=lk["boxes"],
        masks=lk["masks"],
        embeddings=lk["embeddings"],
    )
    img_idxs_list          = [lk["img_idxs_list"]]
    layout_masks           = lk["masks"]
    effective_layout_scale = layout_scale

# ── Append ref tokens AFTER computing layout states ──────────────────────
ids = torch.cat((txt_ids, img_ids), dim=1)
if ref_img is not None:
    if isinstance(ref_img, (tuple, list)):
        img = torch.cat([img, self.img_in(torch.cat(ref_img, dim=1))], dim=1)
        img_ids_list = [ids] + list(ref_img_ids)
        ids = torch.cat(img_ids_list, dim=1)
    else:
        img = torch.cat((img, self.img_in(ref_img)), dim=1)
        ids = torch.cat((ids, ref_img_ids), dim=1)
pe = self.pe_embedder(ids)

# ── Now run double blocks ─────────────────────────────────────────────────
# img_idxs already only addresses [0, img_end), so layout attn naturally
# targets only the target portion, never the ref token tail.
```

This is actually already structurally correct in your `model.py` — `img_end` is set before
ref tokens are appended, and `img_idxs` in the layout branch address positions `[0, img_end)`.
The real fix needed is in the **block-level attention scope** (Bug 4 below).

---

## Bug 4 — SIGNIFICANT: Layout attention scope includes ref token tail

**File**: `uno/flux/modules/layers.py`

In your `DoubleStreamBlock.forward`, the layout cross-attention computes:
```python
region_tokens = img[i, img_idxs]   # img_idxs are [0, img_end) — correct
```
This part is correct. But `context = [obj_token | region_tokens]` and then the attention
`q_all, k_all, v_all` act only on this small local context — the ref tokens are not in scope
here. So Bug 4 is actually **not a code bug** but it manifests as a gradient competition:

The FLUX global self-attention in `DoubleStreamBlockProcessor.__call__` runs on the FULL
`img` (including ref tokens). The ref tokens update `img` through the normal processor path.
Then your layout branch runs afterward **as a residual** on `img` which has already been
heavily influenced by the ref tokens. So the layout delta has to overcome what the ref
attention already wrote.

**Fix**: Use **AdaLayerNorm modulation** in the layout branch so the layout signal is
scaled by the timestep-conditioned `vec`, matching how the paper's Assemble-MMDiT applies
`AdaLayerNorm` before the assembling attention. This makes the layout branch timestep-aware
and prevents it from fighting a fixed magnitude battle against the ref-updated img.

In `DoubleStreamBlock.__init__`:
```python
# Replace layout_norm with a modulated norm (same style as img_mod)
# ADD: layout modulation (single, not double)
from .layers import Modulation   # already imported
self.layout_mod  = Modulation(hidden_size, double=False)
# keep layout_norm, layout_q/k/v/out as before
```

In `DoubleStreamBlock.forward`, replace:
```python
context_norm = self.layout_norm(context)
```
with:
```python
layout_mod_out, _ = self.layout_mod(vec[i:i+1])  # [1, hs] shift/scale/gate
context_normed_raw = self.layout_norm(context)
context_norm = (1 + layout_mod_out.scale) * context_normed_raw + layout_mod_out.shift
# and gate the output:
delta_region = layout_mod_out.gate * layout_scale * self.layout_out(region_out.squeeze(0))
```

Do the same in `SingleStreamBlock`. This makes the layout branch modulated by the same
`vec` that controls the global attention — the two signals are now on the same scale at
every timestep.

---

## Bug 5 — MODERATE: `grounding_ratio=0.4` is too high; `layout_scale=1.0` is too low to start

**File**: `train_hybrid.py`, `TrainArgs`

```python
grounding_ratio: float = 0.4
layout_scale: float = 1.0
```

The paper (Section 3.4) explicitly says: **"layout-conditioned denoising is applied during
the first 30% of diffusion steps"** (`grounding_ratio=0.3`). At 40%, you're applying layout
conditioning into mid-denoising where fine detail is being refined — the layout signal
conflicts with the detail signal and produces blurring.

The `layout_scale=1.0` with a `zero_module`-initialized `layout_out` means the layout
branch starts outputting zeros and has to learn from scratch against a pretrained VAE path.
The 2x LR on layout params helps, but you need a warmup schedule that ramps `layout_scale`
from 0 → 1.5 over the first 2000 steps, then holds at 1.0. This gives the layout branch
a chance to build gradient signal before the VAE path has fully converged.

**Fix in `train_hybrid.py`** — add dynamic layout scale:

```python
# In TrainArgs:
grounding_ratio: float = 0.3        # changed from 0.4
layout_scale: float = 1.5           # changed from 1.0 (compensate for zero-init)
layout_warmup_steps: int = 2000     # new

# In the training loop, replace args.layout_scale with:
if global_step < args.layout_warmup_steps:
    effective_layout_scale = args.layout_scale * (global_step / args.layout_warmup_steps)
else:
    effective_layout_scale = args.layout_scale
```

And pass `effective_layout_scale` into `dit(... layout_scale=effective_layout_scale ...)`.

---

## Bug 6 — MODERATE: VAE crops are padded to squares, destroying aspect ratio

**File**: `uno/dataset/dense_layout.py`, `pad_to_size`

Your VAE refs are padded with black (zeros) to `320×320`. For a small object at say
`80×40` pixels, you're padding to `320×320` — 94% of the ref image is black. The VAE
encoder encodes this black padding as real latent content and the model receives a
heavily diluted signal. This makes the VAE path weaker than expected, which compounds
with the layout branch being too weak (Bug 1/2) to balance correctly.

**Fix in `dense_layout.py`**: Resize (not pad) to 320px on the long edge, then center-crop
or pad only the short edge minimally:

```python
def resize_and_pad_ref(img: Image.Image, target_size: int = 320) -> Image.Image:
    """
    Resize so the long edge = target_size, then pad the short edge.
    Preserves aspect ratio far better than square-padding from scratch.
    """
    w, h = img.size
    if w == 0 or h == 0:
        return Image.new("RGB", (target_size, target_size), (0, 0, 0))
    scale = target_size / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    # Now pad only to make it square
    return pad_to_size(img, target_size, target_size)
```

Replace `pad_to_size(crop, self.REF_SIZE, self.REF_SIZE)` → `resize_and_pad_ref(crop, self.REF_SIZE)`
for both VAE and CLIP crops.

---

## Summary — All files to change and what to change

### `uno/flux/modules/layers.py` — 3 changes

**Change A** (Bug 1 + 4): In `DoubleStreamBlock.__init__` and `SingleStreamBlock.__init__`,
add `self.num_layout_heads = 8` and `self.layout_mod = Modulation(hidden_size, double=False)`.

**Change B** (Bug 1): In both `forward` methods, replace the single-query attention block with
the full bidirectional attention that outputs `[R, hs]` distinct per-pixel deltas.

**Change C** (Bug 4): Wrap the attention output with `layout_mod.gate` scaling in both forward
methods.

Full replacement for the layout block in `DoubleStreamBlock.forward` (and mirror for `SingleStreamBlock`):

```python
if (
    self.use_layout
    and layout_hidden_states is not None
    and layout_masks is not None
    and img_idxs_list is not None
    and layout_scale != 0.0
):
    bsz = img.shape[0]
    hidden_states_add = torch.zeros_like(img)
    density_map       = torch.zeros(bsz, img.shape[1], 1, device=img.device, dtype=img.dtype)
    layout_hidden_add = torch.zeros_like(layout_hidden_states)
    valid = (layout_masks == 1).nonzero(as_tuple=False)

    for k in range(valid.shape[0]):
        i = valid[k, 0].item()
        j = valid[k, 1].item()
        if i >= len(img_idxs_list) or j >= len(img_idxs_list[i]):
            continue
        idxs = img_idxs_list[i][j]
        if idxs.numel() == 0:
            continue
        img_idxs = idxs.to(img.device, dtype=torch.long)

        # Build context: [obj_token(1) | region_tokens(R)]  → [1, 1+R, hs]
        region_tokens = img[i, img_idxs]                          # [R, hs]
        obj_token     = layout_hidden_states[i, j].unsqueeze(0)   # [1, hs]
        context       = torch.cat([obj_token, region_tokens], dim=0).unsqueeze(0)  # [1, 1+R, hs]

        # AdaLayerNorm modulated by vec (Bug 4 fix)
        layout_mod_out, _ = self.layout_mod(vec[i:i+1])
        context_norm = (1 + layout_mod_out.scale) * self.layout_norm(context) + layout_mod_out.shift

        # Bidirectional multi-head attention (Bug 1 fix)
        H    = self.num_layout_heads
        L    = context_norm.shape[1]     # 1 + R
        d    = context_norm.shape[2]
        q_h  = self.layout_q(context_norm).reshape(1, L, H, d // H).transpose(1, 2)
        k_h  = self.layout_k(context_norm).reshape(1, L, H, d // H).transpose(1, 2)
        v_h  = self.layout_v(context_norm).reshape(1, L, H, d // H).transpose(1, 2)
        scale_factor = (d // H) ** -0.5
        attn_out = (torch.softmax((q_h @ k_h.transpose(-2, -1)) * scale_factor, dim=-1) @ v_h)
        attn_out = attn_out.transpose(1, 2).reshape(1, L, d)       # [1, 1+R, hs]

        obj_out    = attn_out[:, :1]     # [1, 1, hs]
        region_out = attn_out[:, 1:]     # [1, R, hs]

        # Gate by layout_mod and accumulate (Bug 2 fix: density map)
        delta_region = layout_mod_out.gate * layout_scale * self.layout_out(region_out.squeeze(0))
        hidden_states_add[i].scatter_add_(
            0,
            img_idxs.unsqueeze(-1).expand(-1, img.shape[-1]),
            delta_region,
        )
        # Count writes for density averaging
        density_map[i].scatter_add_(
            0,
            img_idxs.unsqueeze(-1),
            torch.ones(img_idxs.shape[0], 1, device=img.device, dtype=img.dtype),
        )
        layout_hidden_add[i, j] = (
            layout_mod_out.gate * layout_scale * self.layout_out(obj_out.squeeze(0))
        ).squeeze(0)

    # Bug 2 fix: normalize by density map before adding
    density_map = density_map.clamp(min=1.0)
    img = img + hidden_states_add / density_map
    layout_hidden_states = layout_hidden_states + layout_hidden_add

return img, txt, layout_hidden_states
```

### `train_hybrid.py` — 2 changes

```python
# TrainArgs:
grounding_ratio: float = 0.3    # was 0.4
layout_scale: float = 1.5       # was 1.0
layout_warmup_steps: int = 2000 # new field

# In training loop, before calling dit():
if global_step < args.layout_warmup_steps:
    effective_layout_scale = args.layout_scale * (global_step / max(args.layout_warmup_steps, 1))
else:
    effective_layout_scale = args.layout_scale
# pass effective_layout_scale to dit() call
```

### `uno/dataset/dense_layout.py` — 1 change

Replace `pad_to_size(crop, self.REF_SIZE, self.REF_SIZE)` with `resize_and_pad_ref(crop, self.REF_SIZE)`
and add the `resize_and_pad_ref` function shown in Bug 6.

---

## Why these fixes will work together

The core failure was that the layout branch was computing a single averaged summary vector
and pasting it identically across all pixels in a box. The model correctly learned to mostly
ignore this because it had zero spatial information. After Bug 1+2 fixes, each pixel inside
the box receives a distinct vector computed from attending to both the semantic object token
and all other pixels in the region — which is exactly what the paper's Assemble-Attn does.
Bug 4 gives this signal the same timestep-sensitivity as the global attention. Bug 2 prevents
dense layout scenes from accumulating unbounded signal. Bugs 5 and 6 clean up the training
dynamics so neither signal overwhelms the other during the early phase where both are learning.



