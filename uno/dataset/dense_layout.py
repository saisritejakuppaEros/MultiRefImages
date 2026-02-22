# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# DenseLayout dataset for Hybrid UNO + Layout Conditioning.

import math

import numpy as np
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, ToTensor

from uno.flux.modules.layout import HybridLayout, ObjectCond


def crop_bbox_from_image(img: Image.Image, bbox: list, hw: tuple[int, int]) -> Image.Image:
    """Crop bbox region from image. bbox: [x1, y1, x2, y2] in normalized 0-1."""
    H, W = hw
    x1, y1, x2, y2 = bbox
    x1_px = max(0, int(x1 * W))
    y1_px = max(0, int(y1 * H))
    x2_px = min(W, int(x2 * W))
    y2_px = min(H, int(y2 * H))
    x2_px = max(x2_px, x1_px + 1)
    y2_px = max(y2_px, y1_px + 1)
    return img.crop((x1_px, y1_px, x2_px, y2_px))


def pad_to_size(img: Image.Image, target_h: int, target_w: int) -> Image.Image:
    """Pad PIL image to target size. Center pad, no resize."""
    w, h = img.size
    if h >= target_h and w >= target_w:
        # Crop center if larger (shouldn't happen with bbox crops)
        left = (w - target_w) // 2
        top = (h - target_h) // 2
        return img.crop((left, top, left + target_w, top + target_h))

    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    pad_right = pad_w - pad_left
    pad_bottom = pad_h - pad_top

    # Pad with zeros (black)
    padded = Image.new(img.mode, (target_w, target_h), (0, 0, 0))
    padded.paste(img, (pad_left, pad_top))
    return padded


def xyxy_pixels_to_normalized(bbox: list, hw: tuple[int, int]) -> list:
    """Convert xyxy pixel coords to normalized [0,1]. Handles both xyxy and xywh."""
    H, W = hw
    if len(bbox) == 4:
        # Assume xyxy pixels if x2 > x1 (typical)
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            # Might be xywh
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
    else:
        raise ValueError(f"bbox must have 4 elements, got {len(bbox)}")
    return [
        x1 / W, y1 / H, x2 / W, y2 / H
    ]


def pad_image_to_16(H: int, W: int) -> tuple[int, int]:
    """Return (H', W') padded to 16-divisible."""
    return (math.ceil(H / 16) * 16, math.ceil(W / 16) * 16)


class DenseLayoutDataset(Dataset):
    """
    FireRedTeam/DenseLayout dataset for hybrid layout conditioning.
    - Tier split by bbox area: largest 5 → VAE (Path A), rest → CLIP (Path B).
    - All crops from the main target image.
    - Target at native resolution, pad to 16-divisible only.
    - Ref crops pad to 320×320 (no resize).
    """

    REF_SIZE = 320
    MAX_VAE_REFS = 5

    def __init__(
        self,
        split: str = "train",
        cache_dir: str | None = "./data",
        max_clip_objs: int = 50,
    ):
        self.split = split
        self.max_clip_objs = max_clip_objs
        self.ds = load_dataset("FireRedTeam/DenseLayout", split=split, cache_dir=cache_dir)
        self.transform = Compose([
            ToTensor(),
            Normalize([0.5], [0.5]),
        ])

    def __len__(self) -> int:
        return len(self.ds)

    def _get_bbox_area(self, anno: dict, hw: tuple[int, int]) -> float:
        """Compute bbox area. bbox can be xyxy or xywh pixels."""
        bbox = anno["bbox"]
        H, W = hw
        if len(bbox) != 4:
            return 0.0
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            x1, y1, w, h = bbox
            return w * h
        return (x2 - x1) * (y2 - y1)

    def _bbox_to_normalized_xyxy(self, anno: dict, hw: tuple[int, int]) -> list:
        """Return [x1, y1, x2, y2] normalized 0-1."""
        bbox = anno["bbox"]
        H, W = hw
        if len(bbox) != 4:
            return [0, 0, 1, 1]
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
        return [x1 / W, y1 / H, x2 / W, y2 / H]

    def __getitem__(self, idx: int) -> dict:
        sample = self.ds[idx]
        image = sample["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image)).convert("RGB")
        prompt = sample["prompt"]
        H_orig = sample["height"]
        W_orig = sample["width"]
        hw = (H_orig, W_orig)
        annos = sample["annos"]

        # Sort by bbox area descending; top 5 → VAE, rest → CLIP
        with_area = [
            (a, self._get_bbox_area(a, hw)) for a in annos
        ]
        with_area.sort(key=lambda x: x[1], reverse=True)

        vae_annos = [a for a, _ in with_area[: self.MAX_VAE_REFS]]
        clip_annos = [a for a, _ in with_area[self.MAX_VAE_REFS : self.MAX_VAE_REFS + self.max_clip_objs]]

        vae_ref_pils = []
        vae_bboxes = []
        for a in vae_annos:
            bbox_norm = self._bbox_to_normalized_xyxy(a, hw)
            crop = crop_bbox_from_image(image, bbox_norm, hw)
            crop = pad_to_size(crop, self.REF_SIZE, self.REF_SIZE)
            vae_ref_pils.append(crop)
            vae_bboxes.append(bbox_norm)

        clip_conds_raw = []
        for a in clip_annos:
            bbox_norm = self._bbox_to_normalized_xyxy(a, hw)
            crop = crop_bbox_from_image(image, bbox_norm, hw)
            crop = pad_to_size(crop, self.REF_SIZE, self.REF_SIZE)
            clip_conds_raw.append({
                "bbox": bbox_norm,
                "hw": list(hw),
                "category": a.get("category_name", ""),
                "crop_pil": crop,
            })

        # Target image: native size, pad to 16-divisible
        H_pad, W_pad = pad_image_to_16(H_orig, W_orig)
        img_arr = np.array(image)
        if H_pad != H_orig or W_pad != W_orig:
            padded = np.zeros((H_pad, W_pad, 3), dtype=img_arr.dtype)
            padded[:H_orig, :W_orig] = img_arr
            image = Image.fromarray(padded)
        img_tensor = self.transform(image)

        return {
            "img": img_tensor,
            "txt": prompt,
            "vae_ref_pils": vae_ref_pils,
            "vae_bboxes": vae_bboxes,
            "clip_conds_raw": clip_conds_raw,
            "hw": (H_orig, W_orig),
        }


def collate_fn(batch: list, clip_embedder) -> dict:
    """
    Collate batch. Encodes CLIP for clip_conds and builds HybridLayout.
    Batch size must be 1 for variable resolution.
    """
    assert len(batch) == 1, "DenseLayoutDataset uses batch_size=1 for variable resolution"
    sample = batch[0]

    img = sample["img"]
    txt = sample["txt"]
    vae_ref_pils = sample["vae_ref_pils"]
    clip_conds_raw = sample["clip_conds_raw"]
    hw = sample["hw"]

    # Encode CLIP for clip objects
    clip_crops = [c["crop_pil"] for c in clip_conds_raw]
    if clip_crops:
        clip_embeddings = clip_embedder.encode_image(clip_crops)
        clip_embeddings = [clip_embeddings[i] for i in range(len(clip_crops))]
    else:
        clip_embeddings = []

    conds = []
    for i, v in enumerate(vae_ref_pils):
        conds.append(ObjectCond(
            bbox=sample["vae_bboxes"][i],
            hw=list(hw),
            category="",
            tier="vae",
            ref_image=v,
            embedding=None,
        ))
    for i, c in enumerate(clip_conds_raw):
        conds.append(ObjectCond(
            bbox=c["bbox"],
            hw=c["hw"],
            category=c["category"],
            tier="clip",
            embedding=clip_embeddings[i] if i < len(clip_embeddings) else None,
            ref_image=None,
        ))

    hybrid_layout = HybridLayout(conds, max_objs=50)

    vae_ref_tensors = [Compose([ToTensor(), Normalize([0.5], [0.5])])(p) for p in vae_ref_pils]
    vae_ref_tensors = [t.unsqueeze(0) for t in vae_ref_tensors]

    return {
        "img": img.unsqueeze(0),
        "txt": [txt],
        "vae_ref_pils": vae_ref_pils,
        "vae_ref_tensors": vae_ref_tensors,
        "vae_bboxes": sample["vae_bboxes"],
        "clip_conds": clip_conds_raw,
        "hybrid_layout": hybrid_layout,
        "hw": hw,
    }
