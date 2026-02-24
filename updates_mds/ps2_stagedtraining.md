# Three-Phase Training Strategy

## Why Your Boxes Are Being Ignored Right Now

This is not a hyperparameter problem. It is a structural gradient competition that makes
the optimizer actively learn to ignore your layout branch. Here is exactly what happens:

**Step 1 of your current joint training:**
- LoRA gradient: strong and coherent — pretrained FLUX weights, low-dimensional updates
- Layout gradient: Q/K/V are random normal → random attention → `layout_out` (zero_module) outputs 0. The only useful gradient is `d(loss)/d(layout_out.weight)` which drives it to learn a **global bias** — one constant vector added identically to every pixel in the box. This is not spatial conditioning, it is a learned mean-correction term.

**Step 100:**
- LoRA has learned ~80% of the task
- Layout has learned a fixed bias per block. Q/K/V have never received signal to discriminate between different bboxes, different objects, or different positions within a box.

**Step 1000:**
- LoRA is entrenched. Layout gradient = tiny residual against a dominant converged signal.
- The optimizer finds lower expected loss by making layout weights small.
- Result: LoRA runs everything, layout is a learned-to-be-ignored bias.

**Three-phase training breaks this by giving each path exclusive gradient signal first, so both arrive at Phase 3 already trained — then Phase 3 is coordination, not competition.**

---

## Parameter Counts (From Your Actual Code)

```
layout_net (EmbedBboxProjection):           23M
layout Q/K/V/out in all 57 blocks:       2,153M   ← full Linear(3072,3072) × 4 × 57
LoRA rank-512 in all 57 blocks:             956M
Total trainable:                          3,132M

InstanceAssemble paper (Flux.1, 102M):
  layout_net:                               23M
  LoRA rank-4, 8 blocks:                   1.2M
  layout LoRA rank-4, 8 blocks:            0.8M
```

You have 2,153M randomly-initialized layout params competing with 956M LoRA params.
The paper uses 25M total across 8 blocks. The optimizer has no structural reason to prefer layout.

---

## Phase 1 — Layout Only, No VAE Refs

**Steps:** 5,000  
**Trains:** `layout_net` + layout Q/K/V/out/norm in **7 double blocks + 1 single block**  
**Freezes:** All LoRA. All other layout blocks.  
**VAE refs:** 0 — not passed. The model generates from text + bbox only.

**Which 8 blocks** (matching the InstanceAssemble paper):  
- Double blocks: `[0, 2, 4, 6, 8, 10, 12]`  
- Single block: `[0]`

FLUX builds coarse spatial structure in the early-to-mid double blocks and the first single
block. Blocks 13–18 (double) and 1–37 (single) do fine-detail refinement — injecting layout
there conflicts with texture and causes blurring. The alternating pattern (0,2,4...) prevents
monotonic gradient collapse across consecutive blocks.

**`current_step_ratio`:** Always `0.0` in Phase 1 so layout fires on every forward pass.

**Learning rate:** `2e-4` — higher because layout Q/K/V start random.

**What Phase 1 proves:** Run inference with layout active, no VAE refs. Objects must appear
in their bboxes with correct category semantics. If they do not, the bidirectional attention
fix (Bug 1 from the fix document) was not applied.

---

## Phase 2 — VAE + LoRA Only, Layout Frozen

**Steps:** 8,000  
**Trains:** All LoRA processors (rank-512, all 57 blocks)  
**Freezes:** All layout weights — loaded from Phase 1 checkpoint  
**VAE refs:** 4  
**Layout:** Pass `layout_kwargs=None` — branch does not run at all

Passing `None` (not `layout_scale=0.0`) gives clean gradient isolation. Scale=0 still runs
the dense per-object forward loop wasting compute. `None` skips it entirely.

**Learning rate:** `8e-5` — standard UNO rate.

**What Phase 2 proves:** Run inference with VAE refs, no layout. Matches original UNO
reference-following quality. This confirms the LoRA path still works independently.

---

## Phase 3 — Joint Fine-Tuning

**Steps:** 15,000  
**Trains:** All layout params (from Phase 1) + all LoRA (from Phase 2)  
**VAE refs:** 4  
**Layout active:** Yes, `grounding_ratio=0.3`

**LR ratio — layout gets 0.3× LoRA, NOT 2× like your current setup:**  
- LoRA: `8e-5`  
- Layout: `2.4e-5`

In your current joint training, layout gets `2×` because it is learning from scratch against
a stronger signal. In Phase 3, layout is already trained — high LR causes it to drift from
its Phase 1 solution. LoRA is also already trained. Both need small coordination updates.

**Layout scale warmup:** Ramp `0 → 1.0` over first 1,000 steps of Phase 3. This prevents
the layout branch from immediately asserting strong gradients before LoRA has adjusted to
having a layout co-signal.

**What Phase 3 delivers:** Objects match reference appearances (VAE controls WHAT) and land
in their bboxes (layout controls WHERE). The signals are additive not competitive because
each path pre-trained on its own objective.

---

## All Code Changes

### `uno/flux/modules/layers.py` — Q/K/V initialization

In both `DoubleStreamBlock.__init__` and `SingleStreamBlock.__init__`, after the existing
layout linear definitions, add explicit small-scale init:

```python
# ADD after self.layout_out = zero_module(...) in both classes:
self.num_layout_heads = 8   # 3072 / 8 = 384 per head — needed for bidirectional fix
nn.init.normal_(self.layout_q.weight, std=0.02)
nn.init.zeros_(self.layout_q.bias)
nn.init.normal_(self.layout_k.weight, std=0.02)
nn.init.zeros_(self.layout_k.bias)
nn.init.normal_(self.layout_v.weight, std=0.02)
nn.init.zeros_(self.layout_v.bias)
```

This makes initial attention weights near-uniform rather than chaotic, so the first Phase 1
gradient steps are coherent.

---

### `train_hybrid.py` — Phase orchestration

**Add to `TrainArgs`:**

```python
training_phase: int = 3              # 1, 2, or 3
phase1_ckpt: str | None = None
phase2_ckpt: str | None = None
layout_warmup_steps: int = 1000      # Phase 3 only: ramp layout_scale 0→target
layout_lr_multiplier: float = 2.0    # Phase 1: 2.0.  Phase 3: 0.3
```

**Replace the entire param group + optimizer block** (the section after `dit.enable_layout()`
up to `lr_scheduler = get_scheduler(...)`) with this function, called as
`optimizer = setup_phase(args, dit, logger)`:

```python
def setup_phase(args, dit, logger):
    LAYOUT_KEYS = {
        "layout_net", "layout_q", "layout_k", "layout_v",
        "layout_out", "layout_norm", "layout_mod",
    }
    active_double = set(args.layout_double_blocks) if args.layout_double_blocks else set(range(19))
    active_single = set(args.layout_single_blocks) if args.layout_single_blocks else set(range(38))

    for p in dit.parameters():
        p.requires_grad_(False)

    if args.training_phase == 1:
        for i, block in enumerate(dit.double_blocks):
            if i in active_double:
                for n, p in block.named_parameters():
                    if any(k in n for k in LAYOUT_KEYS):
                        p.requires_grad_(True)
        for i, block in enumerate(dit.single_blocks):
            if i in active_single:
                for n, p in block.named_parameters():
                    if any(k in n for k in LAYOUT_KEYS):
                        p.requires_grad_(True)
        for p in dit.layout_net.parameters():
            p.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            [p for p in dit.parameters() if p.requires_grad],
            lr=args.learning_rate, betas=tuple(args.adam_betas),
            weight_decay=args.adam_weight_decay, eps=args.adam_eps,
        )
        logger.info(f"Phase 1: {sum(p.numel() for p in dit.parameters() if p.requires_grad)/1e6:.0f}M layout params training")

    elif args.training_phase == 2:
        if args.phase1_ckpt:
            dit.load_state_dict(load_file(args.phase1_ckpt), strict=False)
            logger.info(f"Phase 2: loaded Phase 1 weights from {args.phase1_ckpt}")
        for n, p in dit.named_parameters():
            if "lora" in n.lower() or "processor" in n.lower():
                p.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            [p for p in dit.parameters() if p.requires_grad],
            lr=args.learning_rate, betas=tuple(args.adam_betas),
            weight_decay=args.adam_weight_decay, eps=args.adam_eps,
        )
        logger.info(f"Phase 2: {sum(p.numel() for p in dit.parameters() if p.requires_grad)/1e6:.0f}M LoRA params training")

    elif args.training_phase == 3:
        if args.phase1_ckpt:
            dit.load_state_dict(load_file(args.phase1_ckpt), strict=False)
            logger.info(f"Phase 3: loaded Phase 1 layout from {args.phase1_ckpt}")
        if args.phase2_ckpt:
            dit.load_state_dict(load_file(args.phase2_ckpt), strict=False)
            logger.info(f"Phase 3: loaded Phase 2 LoRA from {args.phase2_ckpt}")
        for i, block in enumerate(dit.double_blocks):
            if i in active_double:
                for n, p in block.named_parameters():
                    if any(k in n for k in LAYOUT_KEYS):
                        p.requires_grad_(True)
        for i, block in enumerate(dit.single_blocks):
            if i in active_single:
                for n, p in block.named_parameters():
                    if any(k in n for k in LAYOUT_KEYS):
                        p.requires_grad_(True)
        for p in dit.layout_net.parameters():
            p.requires_grad_(True)
        for n, p in dit.named_parameters():
            if "lora" in n.lower() or "processor" in n.lower():
                p.requires_grad_(True)

        layout_param_ids = set()
        for i, block in enumerate(list(dit.double_blocks) + list(dit.single_blocks)):
            block_idx = i if i < 19 else i - 19
            block_list = dit.double_blocks if i < 19 else dit.single_blocks
            target_set = active_double if i < 19 else active_single
            if block_idx in target_set:
                for n, p in block.named_parameters():
                    if any(k in n for k in LAYOUT_KEYS):
                        layout_param_ids.add(id(p))
        for p in dit.layout_net.parameters():
            layout_param_ids.add(id(p))

        layout_params = [p for p in dit.parameters() if p.requires_grad and id(p) in layout_param_ids]
        lora_params   = [p for p in dit.parameters() if p.requires_grad and id(p) not in layout_param_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": lora_params,   "lr": args.learning_rate},
                {"params": layout_params, "lr": args.learning_rate * args.layout_lr_multiplier},
            ],
            betas=tuple(args.adam_betas), weight_decay=args.adam_weight_decay, eps=args.adam_eps,
        )
        logger.info(f"Phase 3: {sum(p.numel() for p in lora_params)/1e6:.0f}M LoRA @ {args.learning_rate:.1e}, "
                    f"{sum(p.numel() for p in layout_params)/1e6:.0f}M layout @ {args.learning_rate * args.layout_lr_multiplier:.1e}")

    return optimizer
```

**Replace the training loop inner body** — find the `with accelerator.accumulate(dit):` block
and change the three lines computing `current_step_ratio`, `layout_kwargs`, and the `dit(...)`
call:

```python
# Phase-aware current_step_ratio
# Phase 1: layout fires on EVERY forward (not just first 30% of denoising)
if args.training_phase == 1:
    current_step_ratio = 0.0
else:
    current_step_ratio = random.uniform(0.0, 1.0)

# Phase 2: pass None so layout branch is completely skipped
if args.training_phase == 2:
    layout_kw = None
else:
    layout_kw = inp.get("layout_kwargs")

# Phase 3: ramp layout_scale from 0 to target over warmup steps
if args.training_phase == 3 and global_step < args.layout_warmup_steps:
    effective_layout_scale = args.layout_scale * (global_step / max(args.layout_warmup_steps, 1))
else:
    effective_layout_scale = args.layout_scale

model_pred = dit(
    img=x_t.to(weight_dtype),
    img_ids=inp["img_ids"].to(weight_dtype),
    ref_img=inp.get("ref_img"),
    ref_img_ids=inp.get("ref_img_ids"),
    txt=inp["txt"].to(weight_dtype),
    txt_ids=inp["txt_ids"].to(weight_dtype),
    y=inp["vec"].to(weight_dtype),
    timesteps=t_val.to(weight_dtype),
    guidance=torch.full((bs,), 1.0, device=accelerator.device, dtype=weight_dtype),
    layout_kwargs=layout_kw,
    layout_scale=effective_layout_scale,
    grounding_ratio=args.grounding_ratio,
    current_step_ratio=current_step_ratio,
)
```

**Update checkpoint save** to save only what was trained per phase:

```python
CKPT_KEYS = ("layout_net", "layout_q", "layout_k", "layout_v", "layout_out", "layout_norm", "layout_mod")
layout_keys = [k for k in state if any(x in k for x in CKPT_KEYS)]
lora_keys   = [k for k in state if "lora" in k.lower() or "processor" in k.lower()]

if args.training_phase == 1:
    to_save = {k: state[k] for k in layout_keys}
elif args.training_phase == 2:
    to_save = {k: state[k] for k in lora_keys}
else:
    to_save = {k: state[k] for k in layout_keys + lora_keys}
```

---

### `uno/dataset/dense_layout.py` — Make max_vae_refs configurable

Change `MAX_VAE_REFS = 5` from a class constant to an `__init__` parameter:

```python
def __init__(
    self,
    split: str = "train",
    cache_dir: str | None = "./data",
    max_clip_objs: int = 50,
    max_vae_refs: int = 5,     # NEW — was class constant
):
    self.max_vae_refs = max_vae_refs
    # ... rest unchanged ...
```

In `__getitem__`, replace `self.MAX_VAE_REFS` with `self.max_vae_refs` (2 occurrences).

In `train_hybrid.py` `main()`, pass it:

```python
dataset = DenseLayoutDataset(
    split=args.train_split,
    cache_dir=args.cache_dir,
    max_clip_objs=args.max_clip_objs,
    max_vae_refs=args.max_vae_refs,
)
```

---

### Three config files

**`configs/phase1_layout.json`**
```json
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
  "max_train_steps": 5000,
  "checkpointing_steps": 1000,
  "project_dir": "log/phase1",
  "log_dir": "log/phase1"
}
```

**`configs/phase2_vae.json`**
```json
{
  "training_phase": 2,
  "phase1_ckpt": "log/phase1/checkpoint-5000/checkpoint.safetensors",
  "max_vae_refs": 4,
  "max_clip_objs": 50,
  "layout_scale": 0.0,
  "grounding_ratio": 0.0,
  "layout_double_blocks": [],
  "layout_single_blocks": [],
  "layout_lr_multiplier": 1.0,
  "learning_rate": 8e-5,
  "max_train_steps": 8000,
  "checkpointing_steps": 2000,
  "project_dir": "log/phase2",
  "log_dir": "log/phase2"
}
```

**`configs/phase3_joint.json`**
```json
{
  "training_phase": 3,
  "phase1_ckpt": "log/phase1/checkpoint-5000/checkpoint.safetensors",
  "phase2_ckpt": "log/phase2/checkpoint-8000/checkpoint.safetensors",
  "max_vae_refs": 4,
  "max_clip_objs": 50,
  "grounding_ratio": 0.3,
  "layout_scale": 1.0,
  "layout_warmup_steps": 1000,
  "layout_lr_multiplier": 0.3,
  "layout_double_blocks": [0, 2, 4, 6, 8, 10, 12],
  "layout_single_blocks": [0],
  "learning_rate": 8e-5,
  "max_train_steps": 15000,
  "checkpointing_steps": 2000,
  "project_dir": "log/phase3",
  "log_dir": "log/phase3"
}
```