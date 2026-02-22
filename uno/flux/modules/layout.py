# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Ported and adapted from InstanceAssemble for Hybrid UNO + Layout Conditioning.

import math
import torch
import torch.nn as nn
from torch import Tensor
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Tuple


def get_fourier_embeds_from_boundingbox(
    embed_dim: int, box: Tensor, position_dim: int
) -> Tensor:
    """
    box: [B, num_objs, point_num*2]  (dense-sampled bbox points, normalized 0-1)
    Returns: [B, num_objs, position_dim]
    """
    batch_size, num_boxes = box.shape[:2]
    emb = 100 ** (torch.arange(embed_dim, device=box.device, dtype=box.dtype) / embed_dim)
    emb = emb[None, None, None] * box.unsqueeze(-1)
    emb = torch.stack((emb.sin(), emb.cos()), dim=-1)
    emb = emb.permute(0, 1, 3, 4, 2).reshape(batch_size, num_boxes, position_dim)
    return emb


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.detach().zero_()
    return module


class EmbedProjection(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, out_features: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, hidden_size, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(hidden_size, out_features, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_2(self.act(self.linear_1(x)))


class EmbedBboxProjection(nn.Module):
    """
    Projects embedding + bbox to layout token. Accepts any pooled embedding
    (CLIP image, CLIP text, or VAE-pooled).
    """
    def __init__(
        self,
        embed_dim: int = 768,
        out_dim: int = 3072,
        fourier_freqs: int = 8,
        point_num: int = 6,
    ):
        super().__init__()
        self.fourier_freqs = fourier_freqs
        self.point_num = point_num
        self.position_dim = fourier_freqs * 2 * (point_num ** 2) * 2

        self.embed_proj = EmbedProjection(embed_dim, out_dim, out_dim)
        self.fuse = EmbedProjection(out_dim + self.position_dim, out_dim // 2, out_dim)

    def forward(
        self,
        boxes: Tensor,
        masks: Tensor,
        embeddings: Tensor,
    ) -> Tensor:
        """Returns layout_hidden_states: [B, max_objs, out_dim]"""
        masks = masks.unsqueeze(-1)
        pos_emb = get_fourier_embeds_from_boundingbox(
            self.fourier_freqs, boxes, self.position_dim
        ) * masks
        proj_emb = self.embed_proj(embeddings) * masks
        fused = self.fuse(torch.cat([proj_emb, pos_emb], dim=-1))
        return fused


def bbox_to_img_idxs(
    bbox: Tensor,
    latent_h: int,
    latent_w: int,
) -> Tensor:
    """
    Convert dense-sampled bbox to flat image token indices.
    Returns 1D int64 tensor of indices into flattened [H*W] token sequence.
    """
    x1 = int(bbox[0].item() * latent_w)
    y1 = int(bbox[1].item() * latent_h)
    x2 = int(bbox[-2].item() * latent_w)
    y2 = int(bbox[-1].item() * latent_h)
    x2 = max(x2, x1 + 1)
    y2 = max(y2, y1 + 1)
    x1, x2 = max(0, x1), min(latent_w, x2)
    y1, y2 = max(0, y1), min(latent_h, y2)

    rows = torch.arange(y1, y2, dtype=torch.long)
    cols = torch.arange(x1, x2, dtype=torch.long)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
    idxs = (grid_r * latent_w + grid_c).reshape(-1)
    return idxs


def get_layout_idxs_list(
    boxes: Tensor,
    masks: Tensor,
    latent_hw: Tuple[int, int],
) -> list:
    latent_h, latent_w = latent_hw
    result = []
    for i in range(boxes.shape[0]):
        if masks[i].item() == 0:
            result.append(torch.empty(0, dtype=torch.long))
        else:
            result.append(bbox_to_img_idxs(boxes[i], latent_h, latent_w))
    return result


def get_layout_text_ids(
    boxes: Tensor,
    latent_h: int,
    latent_w: int,
) -> Tensor:
    """Center-of-bbox 3D position ids for RoPE, shape [max_objs, 3]."""
    max_objs = boxes.shape[0]
    ids = torch.zeros(max_objs, 3, device=boxes.device, dtype=boxes.dtype)
    for i in range(max_objs):
        x1 = int(boxes[i][0].item() * latent_w)
        y1 = int(boxes[i][1].item() * latent_h)
        x2 = int(boxes[i][-2].item() * latent_w)
        y2 = int(boxes[i][-1].item() * latent_h)
        ids[i, 1] = (y1 + y2) / 2.0
        ids[i, 2] = (x1 + x2) / 2.0
    return ids


@dataclass
class ObjectCond:
    """Single object condition. Exactly one of `embedding` or `ref_image` should be set."""
    bbox: list
    hw: list
    category: str = ""
    tier: str = "clip"
    embedding: Optional[Tensor] = None
    ref_image: Optional[Image.Image] = None


class HybridLayout:
    """
    Holds object conditions split into vae tier (VAE refs) and clip tier
    (layout conditioning with CLIP embedding + bbox).
    """
    def __init__(
        self,
        conds: list,
        max_objs: int = 50,
        point_num: int = 6,
    ):
        self.max_objs = max_objs
        self.point_num = point_num
        self.n_pts = point_num ** 2

        self.vae_conds = [c for c in conds if c.tier == "vae"]
        self.clip_conds = [c for c in conds if c.tier == "clip"]

        assert len(self.vae_conds) <= 6, \
            f"VAE tier supports max 6 objects, got {len(self.vae_conds)}"
        assert len(self.clip_conds) <= max_objs, \
            f"CLIP tier supports max {max_objs} objects, got {len(self.clip_conds)}"

        self.clip_masks = torch.zeros(max_objs)
        self.clip_boxes = torch.zeros(max_objs, self.n_pts * 2)
        self.clip_embeddings = []

        for i, c in enumerate(self.clip_conds):
            self.clip_masks[i] = 1.0
            self._fill_dense_box(self.clip_boxes[i], c.bbox)
            self.clip_embeddings.append(c.embedding)

    def _fill_dense_box(self, out: Tensor, bbox: list):
        """Dense-sample point_num×point_num grid inside bbox."""
        x1, y1, x2, y2 = bbox
        pts_per_edge = self.point_num
        step_x = (x2 - x1) / max(pts_per_edge - 1, 1)
        step_y = (y2 - y1) / max(pts_per_edge - 1, 1)
        idx = 0
        for u in range(pts_per_edge):
            for v in range(pts_per_edge):
                out[idx * 2] = x1 + u * step_x
                out[idx * 2 + 1] = y1 + v * step_y
                idx += 1

    def get_vae_ref_images(self) -> list:
        return [c.ref_image for c in self.vae_conds]

    def build_layout_kwargs(
        self,
        device: torch.device,
        dtype: torch.dtype,
        latent_hw: Tuple[int, int],
    ) -> dict:
        """Build layout_kwargs dict consumed by transformer forward."""
        if len(self.clip_conds) == 0:
            return {"layout": None}

        assert all(e is not None for e in self.clip_embeddings), \
            "All CLIP embeddings must be set before calling build_layout_kwargs"

        embed_dim = self.clip_embeddings[0].shape[-1]
        emb_tensor = torch.zeros(self.max_objs, embed_dim, device=device, dtype=dtype)
        for i, emb in enumerate(self.clip_embeddings):
            emb_tensor[i] = emb.to(device=device, dtype=dtype)

        boxes = self.clip_boxes.unsqueeze(0).to(device=device, dtype=dtype)
        embeddings = emb_tensor.unsqueeze(0).to(device=device, dtype=dtype)
        masks = self.clip_masks.unsqueeze(0).to(device=device, dtype=dtype)
        text_ids = get_layout_text_ids(
            self.clip_boxes, latent_hw[0], latent_hw[1]
        ).unsqueeze(0).to(device=device, dtype=dtype)
        img_idxs_list = get_layout_idxs_list(
            self.clip_boxes, self.clip_masks, latent_hw
        )

        return {
            "layout": {
                "boxes": boxes,
                "embeddings": embeddings,
                "text_ids": text_ids,
                "masks": masks,
                "img_idxs_list": img_idxs_list,
            }
        }
