import torch
import torch.nn as nn
from typing import List, Tuple, Optional


def create_bbox_mask(
    img_idxs_list: List[torch.Tensor],
    num_img_tokens: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Create binary mask for spatial gating: only image tokens inside bbox participate.
    
    Args:
        img_idxs_list: List of (num_bbox_tokens,) tensors - indices of image tokens inside each bbox
        num_img_tokens: total number of image tokens
        batch_size: batch size
        device: device to create mask on
    
    Returns:
        mask: (batch_size, num_img_tokens, num_refs) binary mask
    """
    num_refs = len(img_idxs_list)
    mask = torch.zeros(batch_size, num_img_tokens, num_refs, device=device, dtype=torch.bool)
    
    for ref_idx, img_idxs in enumerate(img_idxs_list):
        if len(img_idxs) > 0:
            mask[:, img_idxs, ref_idx] = True
    
    return mask


def get_image_token_positions(
    img_ids: torch.Tensor,
    latent_h: int,
    latent_w: int,
) -> torch.Tensor:
    """
    Convert image token IDs to 2D positions for spatial gating.
    
    Args:
        img_ids: (num_img_tokens, 3) tensor with [token_id, y_pos, x_pos]
        latent_h: latent height
        latent_w: latent width
    
    Returns:
        positions: (num_img_tokens, 2) tensor with [y, x] positions normalized to [0, 1]
    """
    positions = torch.zeros(img_ids.shape[0], 2, device=img_ids.device)
    positions[:, 0] = img_ids[:, 1] / latent_h  # y normalized
    positions[:, 1] = img_ids[:, 2] / latent_w  # x normalized
    return positions


def bbox_contains_positions(
    bbox: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """
    Check which positions fall inside the bbox.
    
    Args:
        bbox: (4,) tensor with [x1, y1, x2, y2] normalized to [0, 1]
        positions: (num_positions, 2) tensor with [y, x] normalized to [0, 1]
    
    Returns:
        mask: (num_positions,) boolean tensor
    """
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    y_pos, x_pos = positions[:, 0], positions[:, 1]
    
    mask = (x_pos >= x1) & (x_pos <= x2) & (y_pos >= y1) & (y_pos <= y2)
    return mask


def compute_alpha_from_ref_type(ref_type: str) -> float:
    """
    Compute alpha value from reference type.
    BG: alpha = 0.0 (full layout signal)
    MG: alpha = 0.5 (medium layout signal)
    FG: alpha = 1.0 (minimal layout signal, but not zero)
    
    Args:
        ref_type: 'BG', 'MG', or 'FG'
    
    Returns:
        alpha: float in [0, 1]
    """
    alpha_map = {
        'BG': 0.0,
        'MG': 0.5,
        'FG': 1.0,
    }
    return alpha_map.get(ref_type.upper(), 0.5)


def prepare_layout_head_inputs(
    instance_tokens: torch.Tensor,
    alpha: torch.Tensor,
    layout_kv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare inputs for layout head and compute scaling.
    
    Args:
        instance_tokens: (batch_size, num_refs, 3072) instance tokens
        alpha: (batch_size, num_refs) alpha values
        layout_kv: optional pre-computed layout_kv
    
    Returns:
        layout_kv: (batch_size, num_refs, 3072) layout_kv
        scale: (batch_size, num_refs, 1) scaling factors (1 - alpha)
    """
    if layout_kv is None:
        # If layout_kv not provided, assume instance_tokens are already processed
        layout_kv = instance_tokens
    
    scale = (1.0 - alpha).unsqueeze(-1)  # (batch_size, num_refs, 1)
    
    return layout_kv, scale
