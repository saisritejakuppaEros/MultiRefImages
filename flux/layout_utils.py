import math
import torch
from typing import List, Tuple


def bbox_to_mask(bbox: torch.Tensor, latent_h: int, latent_w: int) -> torch.Tensor:
    """
    Create binary mask for a bbox in latent space.
    
    Args:
        bbox: (4,) or (point_num*2,) tensor with first 4 values as [x1, y1, x2, y2] normalized [0, 1]
        latent_h: latent height
        latent_w: latent width
    
    Returns:
        mask: (latent_h, latent_w) binary mask
    """
    mask = torch.zeros((latent_h, latent_w), device=bbox.device)
    (x1, y1), (x2, y2) = bbox[:2], bbox[2:4]
    if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
        return mask
    x1, y1, x2, y2 = int(x1 * latent_w), int(y1 * latent_h), int(x2 * latent_w), int(y2 * latent_h)
    if x1 == x2:
        x2 = x1 + 1
    if y1 == y2:
        y2 = y1 + 1
    mask[y1:y2, x1:x2] = 1
    return mask


def get_layout_idxslist(layout_boxes: torch.Tensor, latent_hw: Tuple[int, int]) -> List[torch.Tensor]:
    """
    Convert bboxes to list of image token indices that fall inside each bbox.
    
    Args:
        layout_boxes: (num_refs, point_num*2) or (num_refs, 4) tensor with bboxes
        latent_hw: (latent_h, latent_w) tuple
    
    Returns:
        img_idxs_list: List of (num_bbox_tokens,) tensors, one per ref
    """
    latent_h, latent_w = latent_hw
    img_idxs_list = []
    for obj_i in range(layout_boxes.shape[0]):
        mask_obj = bbox_to_mask(layout_boxes[obj_i], latent_h, latent_w)
        img_idxs = mask_obj.view(-1).nonzero().to(torch.int).view(-1)
        img_idxs_list.append(img_idxs)
    return img_idxs_list


def get_fourier_embeds_from_boundingbox(embed_dim: int, box: torch.Tensor, position_dim: int) -> torch.Tensor:
    """
    Apply Fourier embedding to bounding boxes.
    
    Args:
        embed_dim: number of frequency components
        box: (batch_size, num_boxes, point_num*2) tensor with bbox points
        position_dim: output dimension (should be embed_dim * 2 * point_num*2)
    
    Returns:
        emb: (batch_size, num_boxes, position_dim) Fourier embeddings
    """
    batch_size, num_boxes = box.shape[:2]

    emb = 100 ** (torch.arange(embed_dim) / embed_dim)
    emb = emb[None, None, None].to(device=box.device, dtype=box.dtype)
    emb = emb * box.unsqueeze(-1)

    emb = torch.stack((emb.sin(), emb.cos()), dim=-1)
    emb = emb.permute(0, 1, 3, 4, 2).reshape(batch_size, num_boxes, position_dim)

    return emb


def get_text_ids(layout_boxes: torch.Tensor, latent_h: int, latent_w: int) -> torch.Tensor:
    """
    Get text token IDs (center positions) for each bbox.
    
    Args:
        layout_boxes: (num_refs, point_num*2) or (num_refs, 4) tensor with bboxes
        latent_h: latent height
        latent_w: latent width
    
    Returns:
        text_ids: (num_refs, 3) tensor with [0, y_center, x_center]
    """
    max_objs = layout_boxes.shape[0]
    text_ids = torch.zeros((max_objs, 3))
    for obj_i in range(max_objs):
        (x1, y1), (x2, y2) = layout_boxes[obj_i][:2], layout_boxes[obj_i][2:4]
        x1, y1, x2, y2 = int(x1 * latent_w), int(y1 * latent_h), int(x2 * latent_w), int(y2 * latent_h)
        text_ids[obj_i, 1] = (y1 + y2) / 2
        text_ids[obj_i, 2] = (x1 + x2) / 2
    return text_ids
