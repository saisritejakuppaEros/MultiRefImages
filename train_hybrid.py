# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Hybrid UNO + Layout Conditioning training script.

import dataclasses
import gc
import logging
import math
import os
import random
from functools import partial

import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from einops import rearrange
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.utils import draw_bounding_boxes, make_grid
from tqdm import tqdm

from uno.dataset.dense_layout import DenseLayoutDataset, collate_fn as dense_layout_collate
from uno.flux.modules.conditioner import CLIPImageEmbedder
from uno.flux.sampling import (
    denoise_hybrid,
    get_noise,
    get_schedule,
    prepare_hybrid_ip,
    unpack,
)
from datasets import load_dataset

from uno.flux.util import load_ae, load_clip, load_flow_model, load_t5, set_lora

logger = get_logger(__name__)


def get_models(name: str, device: torch.device, offload: bool = False, cache_dir: str | None = None):
    t5 = load_t5(device, max_length=512, cache_dir=cache_dir)
    clip = load_clip(device, cache_dir=cache_dir)
    # use_meta=False: Flux has layout_net/layout layers not in base checkpoint; meta tensors would fail when moving to GPU
    model = load_flow_model(name, device="cpu", cache_dir=cache_dir, use_meta=False)
    vae = load_ae(name, device="cpu" if offload else device, cache_dir=cache_dir)
    clip_img = CLIPImageEmbedder(device=str(device), cache_dir=cache_dir)
    return model, vae, t5, clip, clip_img


def _latent_hw_from_pixel(H: int, W: int) -> tuple[int, int]:
    """Token grid size for layout: (H_latent//2, W_latent//2) with latent = pixel/8."""
    h_pad = math.ceil(H / 16) * 16
    w_pad = math.ceil(W / 16) * 16
    lh = h_pad // 8
    lw = w_pad // 8
    return (lh // 2, lw // 2)


def _draw_bboxes_on_tensor(img: torch.Tensor, bboxes_norm: list, color: str = "green") -> torch.Tensor:
    """Draw bboxes (normalized 0-1, [x1,y1,x2,y2]) on (C,H,W) tensor in [0,1]."""
    if not bboxes_norm:
        return img
    img_cpu = img.cpu()
    H, W = img_cpu.shape[1], img_cpu.shape[2]
    boxes = torch.tensor([
        [b[0] * W, b[1] * H, b[2] * W, b[3] * H]
        for b in bboxes_norm
    ], dtype=torch.float)
    boxes = boxes.clamp(min=0)
    boxes[:, 2] = boxes[:, 2].clamp(max=float(W) - 1)
    boxes[:, 3] = boxes[:, 3].clamp(max=float(H) - 1)
    boxes = boxes.long()
    img_uint8 = (img_cpu.clamp(0, 1) * 255).to(torch.uint8)
    drawn = draw_bounding_boxes(img_uint8, boxes, colors=color, width=2)
    return drawn.float() / 255.0


def _pil_to_display_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Convert PIL to (C,H,W) tensor in [0,1] for display."""
    t = Compose([ToTensor(), Normalize([0.5], [0.5])])(pil_img)
    return (t * 0.5 + 0.5)


@dataclasses.dataclass
class TrainArgs:
    project_dir: str | None = "log/hybrid"
    mixed_precision: str = "bf16"
    gradient_accumulation_steps: int = 8
    seed: int = 42

    model_name: str = "flux-dev"
    lora_rank: int = 512
    double_blocks_indices: list[int] | None = None
    single_blocks_indices: list[int] | None = None
    pe: str = "d"
    gradient_checkpoint: bool = True

    learning_rate: float = 8e-5
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    adam_weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 100
    max_train_steps: int = 100000

    train_data_repo: str = "FireRedTeam/DenseLayout"
    train_split: str = "test"
    cache_dir: str | None = "./data"
    model_cache_dir: str | None = "./models"
    max_vae_refs: int = 5
    max_clip_objs: int = 50
    text_dropout: float = 0.1

    layout_scale: float = 1.0
    grounding_ratio: float = 0.4
    layout_double_blocks: list[int] | None = None
    layout_single_blocks: list[int] | None = None

    log_dir: str = "log/hybrid"
    log_image_freq: int = 500
    checkpointing_steps: int = 1000
    resume_from_checkpoint: str | None = None


def main(args: TrainArgs):
    accelerator = Accelerator(
        project_dir=args.project_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed, device_specific=True)

    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "no": torch.float32}.get(
        args.mixed_precision, torch.float32
    )

    logging.basicConfig(
        format=f"[RANK {accelerator.process_index}] " + "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        force=True,
    )
    logger.info("Hybrid UNO training launched")

    model_cache = os.path.abspath(args.model_cache_dir or "./models")
    os.makedirs(model_cache, exist_ok=True)
    dit, vae, t5, clip, clip_img = get_models(args.model_name, accelerator.device, cache_dir=model_cache)
    vae.requires_grad_(False)
    t5.requires_grad_(False)
    clip.requires_grad_(False)
    clip_img.requires_grad_(False)

    dit.requires_grad_(False)
    dit = set_lora(
        dit,
        args.lora_rank,
        args.double_blocks_indices,
        args.single_blocks_indices,
        accelerator.device,
    )
    dit.enable_layout(args.layout_double_blocks, args.layout_single_blocks)
    dit.train()
    dit.gradient_checkpointing = args.gradient_checkpoint

    layout_params = [
        p
        for n, p in dit.named_parameters()
        if p.requires_grad
        and any(
            k in n
            for k in (
                "layout_net",
                "layout_q",
                "layout_k",
                "layout_v",
                "layout_out",
                "layout_norm",
            )
        )
    ]
    base_params = [
        p
        for n, p in dit.named_parameters()
        if p.requires_grad and p not in layout_params
    ]
    param_groups = [{"params": base_params, "lr": args.learning_rate}]
    if layout_params:
        param_groups.append({"params": layout_params, "lr": args.learning_rate * 2.0})

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=args.adam_betas,
        weight_decay=args.adam_weight_decay,
        eps=args.adam_eps,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    global_step = 0
    if args.resume_from_checkpoint:
        resume_path = args.resume_from_checkpoint
        accelerator.print(f"Resuming from {resume_path}")
        state = load_file(resume_path, device=str(accelerator.device))
        dit.load_state_dict(state, strict=False)
        if "global_step" in state:
            global_step = int(state["global_step"])

    try:
        _ = load_dataset(args.train_data_repo, split=args.train_split, cache_dir=args.cache_dir)
    except Exception:
        logger.warning("Split '%s' may not exist; DenseLayout often has only 'test'", args.train_split)
        args.train_split = "test"

    dataset = DenseLayoutDataset(
        split=args.train_split,
        cache_dir=args.cache_dir,
        max_clip_objs=args.max_clip_objs,
    )

    collate = partial(dense_layout_collate, clip_embedder=clip_img)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )
    dataloader = accelerator.prepare_data_loader(dataloader)

    dit, optimizer, lr_scheduler = accelerator.prepare(dit, optimizer, lr_scheduler)
    t5 = accelerator.prepare(t5)
    clip = accelerator.prepare(clip)
    clip_img = clip_img.to(accelerator.device)

    os.makedirs(args.log_dir, exist_ok=True)
    tb_writer = SummaryWriter(args.log_dir) if accelerator.is_main_process else None

    class InfiniteDataLoader:
        """Avoids 'generator already executing' with HF datasets + DataLoader."""
        def __init__(self, dl):
            self.dl = dl
            self._iter = None

        def __iter__(self):
            return self

        def __next__(self):
            if self._iter is None:
                self._iter = iter(self.dl)
            try:
                return next(self._iter)
            except StopIteration:
                gc.collect()
                self._iter = iter(self.dl)
                return next(self._iter)

    dataloader = InfiniteDataLoader(dataloader)

    num_steps_schedule = 999
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        total=args.max_train_steps,
        disable=not accelerator.is_local_main_process,
    )

    train_loss_accum = 0.0
    while global_step < args.max_train_steps:
        batch = next(dataloader)
        img_pixel = batch["img"]
        vae_ref_tensors = batch["vae_ref_tensors"]
        hybrid_layout = batch["hybrid_layout"]
        hw = batch["hw"]
        prompts = batch["txt"]

        prompt = prompts[0]
        if random.random() < args.text_dropout:
            prompt = ""

        with torch.no_grad():
            x_1 = vae.encode(img_pixel.to(accelerator.device).to(torch.float32))
            x_ref = [
                vae.encode(r.to(accelerator.device).to(torch.float32))
                for r in vae_ref_tensors
            ]
            latent_hw = _latent_hw_from_pixel(hw[0], hw[1])
            seq_len = (latent_hw[0] * 2) * (latent_hw[1] * 2)
            if seq_len < 4:
                seq_len = 4
            timesteps = get_schedule(num_steps_schedule, seq_len, shift=True)
            timesteps = torch.tensor(timesteps, device=accelerator.device)
            bs = 1
            t = torch.randint(0, len(timesteps) - 1, (bs,), device=accelerator.device)
            t_val = timesteps[t].float()

            # prepare_hybrid_ip expects 4D img (B,C,H,W); call before rearrange
            inp = prepare_hybrid_ip(
                t5=t5,
                clip=clip,
                img=x_1.to(weight_dtype),
                prompt=[prompt],
                vae_ref_imgs=[x.to(weight_dtype) for x in x_ref],
                hybrid_layout=hybrid_layout,
                latent_hw=latent_hw,
                pe=args.pe,
                device=accelerator.device,
                dtype=weight_dtype,
            )

            # Rearrange to 3D (B, seq_len, features) as expected by model - matches train.py
            x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            x_ref = [rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2) for x in x_ref]
            x_0 = torch.randn_like(x_1, device=accelerator.device)
            x_t = (1 - t_val[:, None, None]) * x_1 + t_val[:, None, None] * x_0

        with accelerator.accumulate(dit):
            current_step_ratio = random.uniform(0.0, 1.0)
            ref_img = inp.get("ref_img")
            ref_img_ids = inp.get("ref_img_ids")

            model_pred = dit(
                img=x_t.to(weight_dtype),
                img_ids=inp["img_ids"].to(weight_dtype),
                ref_img=ref_img,
                ref_img_ids=ref_img_ids,
                txt=inp["txt"].to(weight_dtype),
                txt_ids=inp["txt_ids"].to(weight_dtype),
                y=inp["vec"].to(weight_dtype),
                timesteps=t_val.to(weight_dtype),
                guidance=torch.full((bs,), 1.0, device=accelerator.device, dtype=weight_dtype),
                layout_kwargs=inp["layout_kwargs"],
                layout_scale=args.layout_scale,
                grounding_ratio=args.grounding_ratio,
                current_step_ratio=current_step_ratio,
            )

            target = (x_0 - x_1).float()
            loss = F.mse_loss(model_pred.float(), target, reduction="mean")
            avg_loss = accelerator.gather(loss.detach()).mean().item()
            train_loss_accum += avg_loss / args.gradient_accumulation_steps

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(dit.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            if tb_writer is not None:
                tb_writer.add_scalar("train_loss", train_loss_accum, global_step)
            train_loss_accum = 0.0

        if accelerator.sync_gradients and global_step % args.checkpointing_steps == 0:
            save_path = os.path.join(args.project_dir, f"checkpoint-{global_step}")
            os.makedirs(save_path, exist_ok=True)
            unwrapped = accelerator.unwrap_model(dit)
            state = unwrapped.state_dict()
            layout_keys = [k for k in state if any(x in k for x in ("layout_net", "layout_q", "layout_k", "layout_v", "layout_out", "layout_norm"))]
            lora_keys = [k for k in state if "lora" in k or "processor" in k]
            to_save = {k: state[k] for k in layout_keys + lora_keys if k in state}
            to_save["global_step"] = torch.tensor(global_step)
            save_file(to_save, os.path.join(save_path, "hybrid_lora_layout.safetensors"), safe_serialization=True)
            logger.info(f"Saved checkpoint to {save_path}")

        if accelerator.sync_gradients and tb_writer is not None and global_step % args.log_image_freq == 0 and global_step > 0:
            dit.eval()
            torch.set_grad_enabled(False)
            with torch.no_grad():
                height, width = hw[0], hw[1]
                x_noise = get_noise(1, height, width, accelerator.device, torch.bfloat16, seed=42)
                latent_hw_full = _latent_hw_from_pixel(height, width)
                timesteps = get_schedule(25, (height // 8) * (width // 8) // 4, shift=True)
                # prepare_hybrid_ip expects 4D refs; re-encode from batch (x_ref is 3D after rearrange)
                vae_ref_4d = [vae.encode(r.to(accelerator.device).to(torch.float32)) for r in vae_ref_tensors]
                inp_cond = prepare_hybrid_ip(
                    t5=t5, clip=clip, img=x_noise, prompt=[prompt],
                    vae_ref_imgs=[x.to(torch.bfloat16) for x in vae_ref_4d],
                    hybrid_layout=hybrid_layout,
                    latent_hw=latent_hw_full,
                    pe=args.pe, device=accelerator.device, dtype=torch.bfloat16,
                )
                out = denoise_hybrid(
                    dit, layout_kwargs=inp_cond["layout_kwargs"],
                    layout_scale=args.layout_scale, grounding_ratio=args.grounding_ratio,
                    **{k: v for k, v in inp_cond.items() if k != "layout_kwargs"},
                    timesteps=timesteps, guidance=4.0,
                )
                out_img = unpack(out.float(), height, width)
                out_img = vae.decode(out_img)
                out_img = (out_img.clamp(-1, 1) * 0.5 + 0.5)[0]
            tb_writer.add_image("gen", out_img.cpu(), global_step)
            tb_writer.add_image("target", (img_pixel[0] * 0.5 + 0.5).cpu(), global_step)
            # Generated image with bbox overlay
            all_bboxes = [c.bbox for c in hybrid_layout.vae_conds] + [c.bbox for c in hybrid_layout.clip_conds]
            if all_bboxes:
                out_with_bbox = _draw_bboxes_on_tensor(out_img.cpu(), all_bboxes)
                tb_writer.add_image("gen_with_layout", out_with_bbox, global_step)
            # Path A grid (VAE refs) and Path B grid (CLIP crops)
            vae_ref_pils = batch.get("vae_ref_pils", [])
            clip_conds = batch.get("clip_conds", [])
            path_a_tensors = [_pil_to_display_tensor(p) for p in vae_ref_pils]
            path_b_tensors = [_pil_to_display_tensor(c["crop_pil"]) for c in clip_conds]
            if path_a_tensors:
                grid_a = make_grid(path_a_tensors, nrow=min(5, len(path_a_tensors)), padding=2)
                tb_writer.add_image("path_a_refs", grid_a.cpu(), global_step)
            if path_b_tensors:
                grid_b = make_grid(path_b_tensors, nrow=min(5, len(path_b_tensors)), padding=2)
                tb_writer.add_image("path_b_refs", grid_b.cpu(), global_step)
            torch.set_grad_enabled(True)
            dit.train()

        progress_bar.set_postfix(loss=loss.detach().item(), lr=lr_scheduler.get_last_lr()[0])

    if tb_writer is not None:
        tb_writer.close()
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    import argparse
    import json

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    arg_parser.add_argument("--model_cache_dir", type=str, default=None, help="Directory to cache Hugging Face models")
    arg_parser.add_argument("--max_train_steps", type=int, default=None)
    arg_parser.add_argument("--log_dir", type=str, default=None)
    arg_parser.add_argument("--checkpointing_steps", type=int, default=None)
    arg_parser.add_argument("--log_image_freq", type=int, default=None)
    parsed, remaining = arg_parser.parse_known_args()

    if parsed.config and os.path.exists(parsed.config):
        with open(parsed.config) as f:
            config_dict = json.load(f)
        args = TrainArgs(**{k: v for k, v in config_dict.items() if k in TrainArgs.__dataclass_fields__})
    else:
        args = TrainArgs()

    if parsed.model_cache_dir is not None:
        args.model_cache_dir = parsed.model_cache_dir
    if parsed.max_train_steps is not None:
        args.max_train_steps = parsed.max_train_steps
    if parsed.log_dir is not None:
        args.log_dir = parsed.log_dir
    if parsed.checkpointing_steps is not None:
        args.checkpointing_steps = parsed.checkpointing_steps
    if parsed.log_image_freq is not None:
        args.log_image_freq = parsed.log_image_freq

    main(args)
