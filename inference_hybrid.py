# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Hybrid UNO + Layout Conditioning inference script.

import argparse
import json
import math
import os

import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor

from uno.flux.modules.conditioner import CLIPImageEmbedder
from uno.flux.modules.layout import HybridLayout, ObjectCond
from uno.flux.sampling import denoise_hybrid, get_noise, get_schedule, prepare_hybrid_ip, unpack
from uno.flux.util import load_ae, load_clip, load_flow_model, load_t5, set_lora


def pad_to_size(img: Image.Image, target_h: int, target_w: int) -> Image.Image:
    """Pad PIL image to target size (center pad)."""
    w, h = img.size
    if h >= target_h and w >= target_w:
        left = (w - target_w) // 2
        top = (h - target_h) // 2
        return img.crop((left, top, left + target_w, top + target_h))
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    padded = Image.new(img.mode, (target_w, target_h), (0, 0, 0))
    padded.paste(img, (pad_left, pad_top))
    return padded


def main():
    parser = argparse.ArgumentParser(description="Hybrid UNO inference with layout conditioning")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to hybrid LoRA+layout checkpoint")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--vae_refs", type=str, nargs="*", default=[], help="Paths to VAE reference images")
    parser.add_argument("--vae_bboxes", type=str, default=None, help="JSON list of [x1,y1,x2,y2] per vae ref (normalized 0-1)")
    parser.add_argument("--clip_refs", type=str, nargs="*", default=[], help="Paths to CLIP reference images for layout")
    parser.add_argument("--clip_bboxes", type=str, required=False, help="JSON list of [x1,y1,x2,y2] per clip ref (normalized 0-1)")
    parser.add_argument("--output", type=str, default="output_hybrid.png", help="Output image path")
    parser.add_argument("--width", type=int, default=512, help="Output width")
    parser.add_argument("--height", type=int, default=512, help="Output height")
    parser.add_argument("--num_steps", type=int, default=25, help="Denoising steps")
    parser.add_argument("--guidance", type=float, default=4.0, help="CFG scale")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    parser.add_argument("--layout_scale", type=float, default=1.0, help="Layout conditioning scale")
    parser.add_argument("--grounding_ratio", type=float, default=0.4, help="Layout only in first X%% of steps")
    parser.add_argument("--model_type", type=str, default="flux-dev")
    parser.add_argument("--model_cache_dir", type=str, default="./models", help="Directory to cache Hugging Face models")
    parser.add_argument("--lora_rank", type=int, default=512)
    parser.add_argument("--pe", type=str, default="d", choices=["d", "h", "w", "o"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    cache_dir = os.path.abspath(args.model_cache_dir) if args.model_cache_dir else None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    t5 = load_t5(device, max_length=512, cache_dir=cache_dir)
    clip = load_clip(device, cache_dir=cache_dir)
    clip_img = CLIPImageEmbedder(device=str(device), cache_dir=cache_dir)
    model = load_flow_model(args.model_type, device="cpu", cache_dir=cache_dir)
    vae = load_ae(args.model_type, device=device, cache_dir=cache_dir)

    model = set_lora(model, args.lora_rank, None, None, device)
    model.enable_layout(None, None)
    if args.checkpoint.endswith(".pt") or args.checkpoint.endswith(".pth"):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    else:
        from safetensors.torch import load_file
        ckpt = dict(load_file(args.checkpoint, device="cpu"))
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device, dtype=dtype)
    model.eval()

    transform = Compose([ToTensor(), Normalize([0.5], [0.5])])
    ref_size = 320

    conds = []
    vae_ref_tensors = []
    if args.vae_refs:
        for path in args.vae_refs:
            img = Image.open(path).convert("RGB")
            img = pad_to_size(img, ref_size, ref_size)
            conds.append(ObjectCond(bbox=[0, 0, 1, 1], hw=[args.height, args.width], tier="vae", ref_image=img, embedding=None))
            vae_ref_tensors.append(transform(img).unsqueeze(0))

    if args.clip_refs and args.clip_bboxes:
        try:
            bboxes = json.loads(args.clip_bboxes) if args.clip_bboxes.strip().startswith("[") else json.load(open(args.clip_bboxes))
        except (json.JSONDecodeError, FileNotFoundError):
            bboxes = [[0.25, 0.25, 0.75, 0.75]] * len(args.clip_refs)
        for path, bbox in zip(args.clip_refs, bboxes):
            img = Image.open(path).convert("RGB")
            img = pad_to_size(img, ref_size, ref_size)
            emb = clip_img.encode_image([img])
            conds.append(ObjectCond(bbox=bbox, hw=[args.height, args.width], tier="clip", embedding=emb[0], ref_image=None))
    elif args.clip_refs:
        for path in args.clip_refs:
            img = Image.open(path).convert("RGB")
            img = pad_to_size(img, ref_size, ref_size)
            emb = clip_img.encode_image([img])
            conds.append(ObjectCond(bbox=[0.25, 0.25, 0.75, 0.75], hw=[args.height, args.width], tier="clip", embedding=emb[0], ref_image=None))

    hybrid_layout = HybridLayout(conds, max_objs=50)

    with torch.no_grad():
        vae_ref_latents = [vae.encode(t.to(device).float()) for t in vae_ref_tensors]
        x = get_noise(1, args.height, args.width, device, dtype, args.seed)
        latent_hw = (args.height // 16, args.width // 16)
        timesteps = get_schedule(args.num_steps, (args.height // 8) * (args.width // 8) // 4, shift=True)

        inp = prepare_hybrid_ip(
            t5=t5,
            clip=clip,
            img=x,
            prompt=[args.prompt],
            vae_ref_imgs=[v.to(dtype) for v in vae_ref_latents],
            hybrid_layout=hybrid_layout,
            latent_hw=latent_hw,
            pe=args.pe,
            device=device,
            dtype=dtype,
        )

        out = denoise_hybrid(
            model,
            **{k: v for k, v in inp.items() if k != "layout_kwargs"},
            layout_kwargs=inp["layout_kwargs"],
            layout_scale=args.layout_scale,
            grounding_ratio=args.grounding_ratio,
            timesteps=timesteps,
            guidance=args.guidance,
        )

        out_img = unpack(out.float(), args.height, args.width)
        out_img = vae.decode(out_img)
        out_img = out_img.clamp(-1, 1) * 0.5 + 0.5
        out_pil = Image.fromarray((out_img[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        out_pil.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
