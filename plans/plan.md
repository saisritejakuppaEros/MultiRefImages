Now I have all the technical specifics I need. Here is the complete detailed plan.

---

**FLUX backbone facts you need to anchor dimensions**

FLUX has 19 double-stream blocks (separate text + image streams) and 38 single-stream blocks. Hidden dim = 3072. Text stream tokens: ~77 (T5) + ~77 (CLIP) ≈ 154. Image latent tokens: 128×128/2 = 8192 for 1024×1024 image (after VAE 8× + FLUX 2× patchify). Each DiT block produces 12 AdaLN parameters per stream: 6 for text (shift, scale, gate × 2 for attn and ffn), 6 for image same.

XVerse confirmed: two 3-layer resamplers, intermediate dim 3072, LoRA rank 128.

---

**MODULE 1 — α Predictor**

Purpose: predict importance per ref. Determines how much each ref goes through layout vs modulation pathway. This must converge first and stably before anything else trains.

Input: CLIP image CLS of ref → dim 768. Bbox area ratio → dim 1 (scalar, normalized 0-1). Depth label (FG/MG/BG) → one-hot → dim 3. Concatenated → dim 772.

Architecture: Linear(772 → 256) → ReLU → Linear(256 → 1) → Sigmoid → α scalar ∈ (0,1).

Output: one scalar α per ref.

Init: final linear layer initialized to output 0.5 for all inputs (zero-init the weight, 0 bias before sigmoid). This means at training start every ref gets α=0.5 — equal layout and modulation. No premature routing. The network learns to differentiate as supervision signal accumulates.

Why it converges: α is supervised indirectly by both downstream losses. If FG refs produce bad identity (DINO similarity loss fires), gradients push α upward for those refs. If BG refs misplace (bbox alignment loss fires), gradients push α downward. The predictor finds its own equilibrium. You don't hand-label α.

---

**MODULE 2 — Shared Instance Encoder (InstanceAssemble's Layout Encoder + visual extension)**

Purpose: produce one unified instance token per ref carrying both spatial and visual information.

Step 2a — Spatial encoding (identical to InstanceAssemble):
- Input: bbox (x1, y1, w, h) → DenseSample generates K²=16 uniformly spaced points inside the bbox → each point is a 2D coordinate (x, y)
- Fourier embedding of each point: each (x,y) → sin/cos at multiple frequencies → dim 64 per point → 16 points × 64 = 1024 dim total spatial feature

Step 2b — Visual encoding (your extension):
- For BG/MG refs (α < 0.5): pass ref image through frozen CLIP image encoder → take CLS token → dim 768 → Linear(768 → 1024) → visual feature
- For FG refs (α ≥ 0.5): pass ref image through frozen DINOv2-ViT-L/14 → take CLS token → dim 1024. No projection needed, already 1024.

Why CLIP for BG, DINO for FG: CLIP CLS is semantically rich but low on spatial/textural detail — good enough for "a red chair somewhere in the background." DINO CLS preserves object structure and texture far better — needed for identity-faithful FG generation. Using DINO for all 50 refs is unnecessarily expensive and doesn't help BG.

Step 2c — Fuse spatial + visual:
- Concatenate: [spatial fourier 1024, visual 1024] → dim 2048
- MLP: Linear(2048 → 3072) → LayerNorm → SiLU → Linear(3072 → 3072) → instance token of dim 3072

Why 3072: this matches FLUX's hidden dim and XVerse's resampler intermediate dim. Everything downstream works at 3072 without any dimension mismatch.

Output per ref: one token of dim 3072.

Init: linear layers with Xavier uniform. LayerNorm weight=1, bias=0. SiLU is parameterless. This gives stable gradient flow from the start.

---

**MODULE 3 — Layout Head (operates on instance token, feeds InstanceAssemble-style Assemble-Attn)**

Purpose: from the 3072-dim instance token, produce keys and values for the Assemble-Attn inside FLUX double-stream blocks.

Architecture: Linear(3072 → 3072) → this is your layout head projection.

Output: layout_kv of dim 3072 per ref.

In Assemble-Attn, this layout_kv is split into K and V (each 1536 if 2-headed, or 3072/num_heads per head). Q comes from image tokens inside the bbox region (same as InstanceAssemble).

Spatial gating: only image tokens whose position falls inside this ref's bbox participate as Q. This is a hard binary mask on the attention, same as InstanceAssemble's Assemble-Attn.

Output of Assemble-Attn: updated image tokens inside the bbox region, dim 3072 each, written back to those positions.

Scaling: layout head output is scaled by (1 - α) before entering Assemble-Attn. Low α (BG): full layout signal. High α (FG): layout signal reduced but not zero — spatial placement still needed.

Init: layout head linear initialized to identity (weight = I, bias = 0). At training start, layout head outputs the same vector as the instance token. Gradient will specialize it. Identity init ensures no sudden distribution shift in Assemble-Attn at start of training.

---

**MODULE 4 — Modulation Head (operates on instance token, feeds XVerse-style AdaLN offsets)**

Purpose: from the 3072-dim instance token, produce per-block AdaLN offset vectors for FLUX.

This is where XVerse's T-Mod Adapter logic lives, but adapted.

XVerse uses: text prompt features as query (CLIP text, 768 dim), CLIP image features as key/value, 3-layer perceiver resampler, intermediate dim 3072.

Your adaptation: your instance token (3072 dim, already fusing text region description + visual features) replaces the separate text+image query/key-value pairs. You already fused them in Module 2. So you need a simpler adapter here.

Architecture of modulation head:
- Input: instance token, dim 3072
- 2-layer MLP: Linear(3072 → 3072) → SiLU → Linear(3072 → 3072) → shared_offset, dim 3072
- This shared_offset is per-ref, shared across all DiT blocks (same as XVerse's ∆_shared)
- Then per-block projection: Linear(3072 → 3072) applied once per DiT block (19 double-stream blocks) → per_block_offset, dim 3072 per block
- Final per block: this 3072 is split into 12 AdaLN parameters (shift_pre_attn, scale_pre_attn, gate_post_attn, shift_pre_ffn, scale_pre_ffn, gate_post_ffn, × 2 for text and image streams = 12 scalars)
- So final linear inside each block: Linear(3072 → 12) → 12 scalars

Output: 12 AdaLN offset scalars per block per ref. These are added to the existing AdaLN parameters that FLUX already computed from (timestep + text prompt).

Spatial gating on modulation: the 12 AdaLN scalars are global (they affect all tokens in the stream). You spatially confine them by: after AdaLN runs with the offset, multiply the normalized image token features by the bbox mask (1 inside bbox, 0 outside). This cancels modulation outside the bbox region before attention runs.

Scaling: modulation head output scaled by α before adding to AdaLN params. High α (FG): full modulation. Low α (BG): near-zero modulation, barely affects AdaLN.

Init: both linear layers in the 2-layer MLP initialized to zero weights, zero bias. This is critical. At training start, modulation offset = 0 for all refs. The model starts as pure FLUX with no modulation — guaranteed stable baseline. Gradients slowly build up the offsets. This is exactly how XVerse initialized it and why it converges without destabilizing the backbone.

---

**MODULE 5 — Attention-to-Modulation Feedback Bridge (your novel piece)**

Purpose: after Assemble-Attn updates image tokens inside a bbox at block N, feed that spatial information back to strengthen the modulation signal at block N+1.

How it works: after Assemble-Attn output at block N, mean-pool the updated image tokens inside the bbox region → one vector of dim 3072 (the spatial summary of what just happened in attention). Pass through Linear(3072 → 3072) → this is added as a residual to the per_block_offset of block N+1's modulation head output.

Effect: if attention at block N successfully placed the FG object in its bbox region (image tokens inside bbox now carry strong object features), this signal strengthens the modulation offset for block N+1 — telling the modulation "attention has anchored this object here, now sharpen its identity in this region."

This creates a progressive loop: attention anchors placement → modulation sharpens identity → stronger identity features in those image tokens → next block's attention uses richer features → cleaner placement. Synergistic, not competing.

Init: Linear(3072 → 3072) initialized to zero weights. At training start feedback = 0, no effect. The network learns to use this bridge only when it's helpful. Zero init prevents early training instability.

Gating: only active for FG refs (α ≥ 0.5). BG/MG refs get no feedback — their layout signal is already sufficient.

---

**COMPLETE FORWARD PASS, BLOCK BY BLOCK**

Before denoising loop starts:
- Run all 50 refs through frozen CLIP/DINO → get visual features
- Run all 50 bboxes through DenseSample + Fourier → get spatial features
- Run α predictor → get α per ref
- Run Module 2 → get 50 instance tokens (3072 each)
- Run layout head → get 50 layout_kv vectors (3072 each), scaled by (1-α)
- Run modulation head 2-layer MLP → get 50 shared_offsets (3072 each)
- Compute all 50 spatial bbox masks in latent space (128×128 soft Gaussians)
- Cache all of the above — reuse across all denoising steps

At each denoising step, for each FLUX double-stream block b (1 to 19):

1. FLUX computes existing y = MLP(timestep, text_prompt_CLIP) → dim 3072

2. For each FG ref i (α≥0.5): compute per_block_offset_i_b = Linear_b(shared_offset_i + feedback_i_b) where feedback_i_b comes from block b-1's Assemble-Attn output (zero at b=1). Add offset × α_i to y → y* for that ref's token-specific modulation

3. AdaLN runs with y* → normalized image tokens (3072 each, 8192 tokens)

4. Apply spatial gating: for each FG ref i, image tokens outside bbox_i get their normalized features rolled back toward the y (non-modulated) version. Inside bbox: keep y*, outside: keep y-normalized.

5. Assemble-Attn runs: for each of the 50 refs, independently compute cross-attention between layout_kv_i (as K and V) and image tokens inside bbox_i (as Q) → updated image tokens inside bbox written back. This runs for all 50 refs but each attends to a small bbox subset of the 8192 tokens.

6. Feedback bridge: for FG refs, mean-pool the updated image tokens inside bbox → Linear → store as feedback_i_{b+1} for next block.

7. Feed-forward runs normally.

After all 19 double-stream blocks: run 38 single-stream blocks with no layout injection (modulation only for FG, very light). Then decode via VAE.

---

**TRAINING SETUP**

**What is frozen:**
- FLUX backbone entirely (all double and single stream blocks)
- CLIP image and text encoders
- DINOv2
- VAE encoder and decoder

**What is trained:**
- α predictor MLP (~0.5M params)
- Module 2 instance encoder MLPs (~12M params, two linear layers 2048→3072→3072)
- Layout head linear (~9M params, one 3072→3072 linear)
- Modulation head 2-layer MLP (~18M params)
- Per-block modulation projection: 19 × Linear(3072→12) (~0.7M params)
- Feedback bridge linear: 19 × Linear(3072→3072) (~360M params — this is large, use LoRA rank 16 on this instead: 19 × LoRA(3072, rank=16) → ~2M params)
- LoRA rank 128 on FLUX's MM-Attention Q,K,V projections for Assemble-Attn (~100M params, same as InstanceAssemble)

Total trainable: ~142M params. Comparable to InstanceAssemble.

**Loss terms:**

Primary: FLUX flow matching loss on noise prediction. Standard, all refs jointly.

Auxiliary 1 — DINO identity loss (FG refs only, weighted by α):
Crop the generated image at FG bbox locations → extract DINO CLS → cosine similarity with ref DINO CLS → minimize (1 - cosine_sim) × α. Weight = 10 (same as XVerse's region preservation weight).

Auxiliary 2 — Bbox alignment loss (all refs, weighted by (1-α)):
Use a lightweight pretrained detector (DINO-based) to find objects in generated image → L1 distance between detected center and LLM bbox center → minimize × (1-α). Weight = 1.

Auxiliary 3 — Region preservation loss (XVerse style, for preventing bleed):
During training, randomly mask out modulation for a random subset of refs → generated image in those regions should be unaffected. Penalize L2 distance of those image tokens vs no-conditioning baseline. Weight = 0.01 (same as XVerse's text-image attention loss).

**Training stages:**

Stage 1 — 70K iterations: train only α predictor + Module 2 instance encoder + layout head + Assemble-Attn LoRA. Modulation head output zeroed out (zero init means it contributes nothing). Model learns layout from scratch. LR = 1e-4. This is InstanceAssemble's training essentially.

Stage 2 — 150K iterations: unfreeze modulation head + per-block projections. Zero-initialized so they start contributing nothing and build up. Add DINO identity loss. LR = 5e-6 (same as XVerse). Model learns to add identity on top of layout.

Stage 3 — 10K iterations: unfreeze feedback bridge LoRA. LR = 1e-5. Train only the bridge, everything else frozen. Fine-tunes the cross-block communication.

**Why this converges:**

Stage 1 has no modulation — pure layout training, well-understood, stable. When Stage 2 starts, layout is already solid. Modulation starts at zero and builds up slowly (zero init), so it cannot destabilize the layout that was learned. The spatial gating prevents modulation from spreading outside bboxes — gradients from BG regions cannot corrupt FG modulation weights. The α scaling means BG refs contribute near-zero gradient to the modulation pathway — effectively the modulation head trains only on FG refs. Small effective batch, clean signal. Stage 3 trains only the feedback bridge on an already-converged model — lowest risk.

**Training data needed per sample:**
- Ground truth composited scene image
- Isolated ref crop per object (can auto-generate from any dataset with SAM2 segmentation)
- Bboxes per object (from SAM2 or ground truth annotation)
- Depth label per object (from any depth estimator — ZoeDepth, DepthAnything)
- Global caption (CogVLM or LLaVA on the scene)
- Instance caption per object (short description, from LLaVA on the crop)