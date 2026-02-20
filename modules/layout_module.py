import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class LayoutHead(nn.Module):
    """
    Layout Head: Projects 3072-dim instance token to layout_kv for Assemble-Attn.
    
    Architecture: Linear(3072 → 3072)
    Initialization: Identity (weight = I, bias = 0)
    Output: layout_kv of dim 3072 per ref
    """
    
    def __init__(self, dim: int = 3072):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim, bias=True)
        self._init_identity()
    
    def _init_identity(self):
        """Initialize weight to identity matrix and bias to zero."""
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, instance_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            instance_tokens: (batch_size, num_refs, 3072) or (batch_size, 3072)
        
        Returns:
            layout_kv: (batch_size, num_refs, 3072) or (batch_size, 3072)
        """
        return self.proj(instance_tokens)


def split_layout_kv(layout_kv: torch.Tensor, num_heads: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split layout_kv into K and V for multi-head attention.
    
    In this implementation, we use the same layout_kv for both K and V,
    just reshaped for multi-head attention.
    
    Args:
        layout_kv: (..., 3072) tensor
        num_heads: number of attention heads (default: 24)
    
    Returns:
        k: (..., num_heads, head_dim) tensor where head_dim = 3072 // num_heads = 128
        v: (..., num_heads, head_dim) tensor where head_dim = 3072 // num_heads = 128
    """
    head_dim = layout_kv.shape[-1] // num_heads  # 3072 // 24 = 128
    
    # Reshape for multi-head attention: (..., num_heads, head_dim)
    k = layout_kv.reshape(*layout_kv.shape[:-1], num_heads, head_dim)
    v = layout_kv.reshape(*layout_kv.shape[:-1], num_heads, head_dim)
    
    return k, v


def assemble_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Assemble-Attn: Cross-attention where Q comes from image tokens, K/V from layout_kv.
    
    Args:
        q: (batch_size, num_img_tokens, num_heads, head_dim) - queries from image tokens
        k: (batch_size, num_refs, num_heads, head_dim) - keys from layout_kv
        v: (batch_size, num_refs, num_heads, head_dim) - values from layout_kv
        scale: scaling factor (typically 1 / sqrt(head_dim))
        mask: (batch_size, num_img_tokens, num_refs) binary mask for spatial gating
    
    Returns:
        output: (batch_size, num_img_tokens, num_heads, head_dim)
    """
    # Ensure q, k, v are in correct shape (already 4D from split_layout_kv)
    if q.ndim == 3:
        batch_size, num_img_tokens, q_dim = q.shape
        num_heads = k.shape[2] if k.ndim == 4 else 24
        head_dim = q_dim // num_heads
        q = q.reshape(batch_size, num_img_tokens, num_heads, head_dim)
    else:
        batch_size, num_img_tokens, num_heads, head_dim = q.shape
    
    if k.ndim == 3:
        num_refs, k_dim = k.shape[1], k.shape[2]
        k = k.reshape(batch_size, num_refs, num_heads, head_dim)
    else:
        num_refs = k.shape[1]
    
    if v.ndim == 3:
        v = v.reshape(batch_size, num_refs, num_heads, head_dim)
    
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)
    
    # Compute attention scores: (batch_size, num_heads, num_img_tokens, num_refs)
    q_perm = q.permute(0, 2, 1, 3)  # (batch_size, num_heads, num_img_tokens, head_dim)
    k_perm = k.permute(0, 2, 3, 1)  # (batch_size, num_heads, head_dim, num_refs)
    
    attn_scores = torch.matmul(q_perm, k_perm) * scale  # (batch_size, num_heads, num_img_tokens, num_refs)
    
    # Apply spatial gating mask if provided
    if mask is not None:
        # mask can be (batch_size, num_img_tokens) or (batch_size, num_img_tokens, num_refs)
        if mask.ndim == 2:
            # (batch_size, num_img_tokens) -> expand to (batch_size, num_img_tokens, num_refs)
            mask = mask.unsqueeze(-1).expand(-1, -1, num_refs)
        # Now mask is (batch_size, num_img_tokens, num_refs)
        # Reshape to match attn_scores: (batch_size, num_heads, num_img_tokens, num_refs)
        mask = mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
    
    attn_probs = F.softmax(attn_scores, dim=-1)
    
    # Apply attention to values
    v_perm = v.permute(0, 2, 1, 3)  # (batch_size, num_heads, num_refs, head_dim)
    attn_output = torch.matmul(attn_probs, v_perm)  # (batch_size, num_heads, num_img_tokens, head_dim)
    
    # Reshape back: (batch_size, num_img_tokens, num_heads, head_dim)
    attn_output = attn_output.permute(0, 2, 1, 3)
    
    return attn_output


class AssembleAttentionBlock(nn.Module):
    """
    Assemble-Attn block that integrates LayoutHead with attention computation.
    
    This block:
    1. Takes instance tokens and produces layout_kv via LayoutHead
    2. Scales layout_kv by (1 - α)
    3. Splits layout_kv into K and V
    4. Uses Q from image tokens inside bbox region
    5. Applies spatial gating (hard binary mask)
    6. Outputs updated image tokens
    """
    
    def __init__(
        self,
        dim: int = 3072,
        num_heads: int = 24,
        head_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        # head_dim is determined by how layout_kv is split: dim // (2 * num_heads)
        # This ensures K and V each get dim/2 total, split across num_heads
        if head_dim is None:
            self.head_dim = dim // (2 * num_heads)  # 3072 // (2 * 24) = 64
        else:
            self.head_dim = head_dim
        
        # Layout head: projects instance token to layout_kv
        self.layout_head = LayoutHead(dim=dim)
        
        # Projection for Q from image tokens - output dim must match K/V
        # Q gets head_dim per head, so total is num_heads * head_dim
        self.to_q = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        
        # Output projection
        self.to_out = nn.Linear(num_heads * self.head_dim, dim, bias=False)
    
    def forward(
        self,
        instance_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        img_idxs_list: List[torch.Tensor],
        bbox_masks: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            instance_tokens: (batch_size, num_refs, 3072) - instance tokens per ref
            image_tokens: (batch_size, num_img_tokens, 3072) - all image tokens
            img_idxs_list: List of (num_bbox_tokens,) tensors - indices of image tokens inside each bbox
            bbox_masks: (batch_size, num_img_tokens, num_refs) - binary mask for spatial gating
            alpha: (batch_size, num_refs) - alpha values per ref (0=BG, 1=FG)
        
        Returns:
            updated_image_tokens: (batch_size, num_img_tokens, 3072) - updated image tokens
        """
        batch_size, num_refs, _ = instance_tokens.shape
        num_img_tokens = image_tokens.shape[1]
        
        # Step 1: Produce layout_kv from instance tokens
        layout_kv = self.layout_head(instance_tokens)  # (batch_size, num_refs, 3072)
        
        # Step 2: Scale by (1 - α)
        alpha_expanded = alpha.unsqueeze(-1)  # (batch_size, num_refs, 1)
        layout_kv_scaled = layout_kv * (1.0 - alpha_expanded)  # (batch_size, num_refs, 3072)
        
        # Step 3: Split layout_kv into K and V
        k, v = split_layout_kv(layout_kv_scaled, self.num_heads)  # (batch_size, num_refs, num_heads, head_dim)
        
        # Step 4: Process each ref's bbox region
        updated_tokens = image_tokens.clone()
        
        for ref_idx in range(num_refs):
            img_idxs = img_idxs_list[ref_idx]  # (num_bbox_tokens,)
            if len(img_idxs) == 0:
                continue
            
            # Extract Q from image tokens inside bbox
            q_tokens = image_tokens[:, img_idxs, :]  # (batch_size, num_bbox_tokens, 3072)
            q = self.to_q(q_tokens)  # (batch_size, num_bbox_tokens, num_heads * head_dim)
            q = q.reshape(batch_size, len(img_idxs), self.num_heads, self.head_dim)
            
            # Get K and V for this ref
            k_ref = k[:, ref_idx:ref_idx+1, :, :]  # (batch_size, 1, num_heads, head_dim)
            v_ref = v[:, ref_idx:ref_idx+1, :, :]  # (batch_size, 1, num_heads, head_dim)
            
            # Get mask for this ref - all tokens in bbox are valid (mask=1)
            # Since we're already filtering by img_idxs, all tokens should participate
            mask_ref_bbox = torch.ones(batch_size, len(img_idxs), 1, device=image_tokens.device, dtype=torch.bool)
            
            # Compute attention
            # Note: mask shape should be (batch_size, num_bbox_tokens) for single ref
            attn_output = assemble_attention(
                q=q,
                k=k_ref,
                v=v_ref,
                scale=1.0 / (self.head_dim ** 0.5),
                mask=mask_ref_bbox.squeeze(-1),  # (batch_size, num_bbox_tokens)
            )  # (batch_size, num_bbox_tokens, num_heads, head_dim)
            
            # Project output
            attn_output_flat = attn_output.reshape(batch_size, len(img_idxs), self.num_heads * self.head_dim)
            output = self.to_out(attn_output_flat)  # (batch_size, num_bbox_tokens, 3072)
            
            # Write back to image tokens
            updated_tokens[:, img_idxs, :] = output
        
        return updated_tokens


class LayoutIntegrator(nn.Module):
    """
    Integration wrapper that orchestrates layout pathway with FLUX.
    
    Manages:
    - img_idxs_list creation from bboxes
    - AssembleAttentionAddon orchestration
    - Alpha scaling (1-α)
    - Layout summary computation for feedback
    """
    
    def __init__(
        self,
        dim: int = 3072,
        num_heads: int = 24,
        head_dim: int = 128,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        from .assemble_attention_addon import AssembleAttentionAddon
        self.assemble_addon = AssembleAttentionAddon(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
        )
    
    def create_img_idxs_list(
        self,
        bboxes: torch.Tensor,
        latent_hw: Tuple[int, int],
    ) -> List[List[torch.Tensor]]:
        """
        Create img_idxs_list_list from bboxes.
        
        Args:
            bboxes: (batch_size, num_refs, 4) normalized [x1, y1, x2, y2]
            latent_hw: (latent_h, latent_w) tuple
        
        Returns:
            img_idxs_list_list: List[List[Tensor]] per batch and ref
        """
        from ..flux.layout_utils import get_layout_idxslist
        
        batch_size, num_refs = bboxes.shape[:2]
        img_idxs_list_list = []
        
        for batch_idx in range(batch_size):
            batch_boxes = bboxes[batch_idx]
            img_idxs_list = get_layout_idxslist(batch_boxes, latent_hw)
            img_idxs_list_list.append(img_idxs_list)
        
        return img_idxs_list_list
    
    def apply_layout(
        self,
        instance_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        alpha: torch.Tensor,
        img_idxs_list_list: List[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply layout pathway via AssembleAttentionAddon.
        
        Args:
            instance_tokens: (batch_size, num_refs, 3072)
            image_tokens: (batch_size, num_img_tokens, 3072)
            alpha: (batch_size, num_refs)
            img_idxs_list_list: List[List[Tensor]] image token indices
        
        Returns:
            updated_image_tokens: (batch_size, num_img_tokens, 3072)
            layout_summary: (batch_size, num_refs, 3072) mean-pooled bbox features
        """
        batch_size, num_refs = instance_tokens.shape[:2]
        
        layout_masks = torch.ones(batch_size, num_refs, device=instance_tokens.device, dtype=torch.bool)
        
        updated_tokens, layout_outputs = self.assemble_addon(
            instance_tokens=instance_tokens,
            image_tokens=image_tokens,
            img_idxs_list_list=img_idxs_list_list,
            layout_masks=layout_masks,
            alpha=alpha,
        )
        
        layout_summary = torch.zeros(batch_size, num_refs, self.dim, device=image_tokens.device, dtype=image_tokens.dtype)
        for batch_idx in range(batch_size):
            for ref_idx in range(num_refs):
                img_idxs = img_idxs_list_list[batch_idx][ref_idx]
                if len(img_idxs) > 0:
                    bbox_tokens = updated_tokens[batch_idx, img_idxs, :]
                    layout_summary[batch_idx, ref_idx] = bbox_tokens.mean(dim=0)
        
        return updated_tokens, layout_summary


class LayoutIntegrator(nn.Module):
    """
    Integration wrapper that orchestrates layout pathway with FLUX.
    
    Manages:
    - img_idxs_list creation from bboxes
    - AssembleAttentionAddon orchestration
    - Alpha scaling (1-α)
    - Layout summary computation for feedback
    """
    
    def __init__(
        self,
        dim: int = 3072,
        num_heads: int = 24,
        head_dim: int = 128,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        from .assemble_attention_addon import AssembleAttentionAddon
        self.assemble_addon = AssembleAttentionAddon(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
        )
    
    def create_img_idxs_list(
        self,
        bboxes: torch.Tensor,
        latent_hw: Tuple[int, int],
    ) -> List[List[torch.Tensor]]:
        """
        Create img_idxs_list_list from bboxes.
        
        Args:
            bboxes: (batch_size, num_refs, 4) normalized [x1, y1, x2, y2]
            latent_hw: (latent_h, latent_w) tuple
        
        Returns:
            img_idxs_list_list: List[List[Tensor]] per batch and ref
        """
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from flux.layout_utils import get_layout_idxslist
        
        batch_size, num_refs = bboxes.shape[:2]
        img_idxs_list_list = []
        
        for batch_idx in range(batch_size):
            batch_boxes = bboxes[batch_idx]
            img_idxs_list = get_layout_idxslist(batch_boxes, latent_hw)
            img_idxs_list_list.append(img_idxs_list)
        
        return img_idxs_list_list
    
    def apply_layout(
        self,
        instance_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        alpha: torch.Tensor,
        img_idxs_list_list: List[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply layout pathway via AssembleAttentionAddon.
        
        Args:
            instance_tokens: (batch_size, num_refs, 3072)
            image_tokens: (batch_size, num_img_tokens, 3072)
            alpha: (batch_size, num_refs)
            img_idxs_list_list: List[List[Tensor]] image token indices
        
        Returns:
            updated_image_tokens: (batch_size, num_img_tokens, 3072)
            layout_summary: (batch_size, num_refs, 3072) mean-pooled bbox features
        """
        batch_size, num_refs = instance_tokens.shape[:2]
        
        # Store original dtype for output conversion
        original_dtype = image_tokens.dtype
        
        # Convert to module dtype (float32 if trainable)
        module_dtype = next(self.assemble_addon.parameters()).dtype
        instance_tokens = instance_tokens.to(module_dtype)
        image_tokens = image_tokens.to(module_dtype)
        alpha = alpha.to(module_dtype)
        
        layout_masks = torch.ones(batch_size, num_refs, device=instance_tokens.device, dtype=torch.bool)
        
        updated_tokens, layout_outputs = self.assemble_addon(
            instance_tokens=instance_tokens,
            image_tokens=image_tokens,
            img_idxs_list_list=img_idxs_list_list,
            layout_masks=layout_masks,
            alpha=alpha,
        )
        
        layout_summary = torch.zeros(batch_size, num_refs, self.dim, device=image_tokens.device, dtype=module_dtype)
        for batch_idx in range(batch_size):
            for ref_idx in range(num_refs):
                img_idxs = img_idxs_list_list[batch_idx][ref_idx]
                if len(img_idxs) > 0:
                    bbox_tokens = updated_tokens[batch_idx, img_idxs, :]
                    layout_summary[batch_idx, ref_idx] = bbox_tokens.mean(dim=0)
        
        # Convert back to original dtype (bfloat16 for FLUX)
        return updated_tokens.to(original_dtype), layout_summary.to(original_dtype)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    batch_size = 2
    num_refs = 3
    num_img_tokens = 1024
    dim = 3072
    num_heads = 24
    head_dim = 128
    
    print("Testing LayoutHead...")
    layout_head = LayoutHead(dim=dim).to(device)
    instance_tokens = torch.randn(batch_size, num_refs, dim, device=device)
    layout_kv = layout_head(instance_tokens)
    print(f"  Input shape: {instance_tokens.shape}")
    print(f"  Output shape: {layout_kv.shape}")
    assert layout_kv.shape == (batch_size, num_refs, dim), f"Expected {(batch_size, num_refs, dim)}, got {layout_kv.shape}"
    
    # Test identity initialization
    identity_test = layout_head(torch.eye(dim, device=device).unsqueeze(0))
    assert torch.allclose(identity_test[0], torch.eye(dim, device=device), atol=1e-5), "Identity init failed"
    print("  ✓ LayoutHead passed\n")
    
    print("Testing split_layout_kv...")
    k, v = split_layout_kv(layout_kv, num_heads)
    print(f"  K shape: {k.shape}")
    print(f"  V shape: {v.shape}")
    expected_head_dim = dim // (2 * num_heads)
    assert k.shape == (batch_size, num_refs, num_heads, expected_head_dim), f"K shape mismatch"
    assert v.shape == (batch_size, num_refs, num_heads, expected_head_dim), f"V shape mismatch"
    print("  ✓ split_layout_kv passed\n")
    
    print("Testing assemble_attention...")
    # Use the actual head_dim from split_layout_kv (64, not 128)
    actual_head_dim = expected_head_dim
    q = torch.randn(batch_size, 100, num_heads, actual_head_dim, device=device)
    k_single = k[:, 0:1, :, :]  # (batch_size, 1, num_heads, head_dim)
    v_single = v[:, 0:1, :, :]
    mask = torch.ones(batch_size, 100, 1, device=device, dtype=torch.bool)
    attn_out = assemble_attention(q, k_single, v_single, mask=mask)
    print(f"  Q shape: {q.shape}")
    print(f"  K shape: {k_single.shape}")
    print(f"  V shape: {v_single.shape}")
    print(f"  Output shape: {attn_out.shape}")
    assert attn_out.shape == q.shape, f"Output shape mismatch"
    print("  ✓ assemble_attention passed\n")
    
    print("Testing AssembleAttentionBlock...")
    # Use actual head_dim from split (64) instead of 128
    actual_head_dim_for_block = expected_head_dim
    assemble_block = AssembleAttentionBlock(dim=dim, num_heads=num_heads, head_dim=actual_head_dim_for_block).to(device)
    image_tokens = torch.randn(batch_size, num_img_tokens, dim, device=device)
    
    # Create dummy img_idxs_list and bbox_masks
    img_idxs_list = [
        torch.tensor([10, 11, 12, 20, 21, 22], device=device),
        torch.tensor([30, 31, 32], device=device),
        torch.tensor([50, 51], device=device),
    ]
    
    bbox_masks = torch.zeros(batch_size, num_img_tokens, num_refs, device=device, dtype=torch.bool)
    for ref_idx, img_idxs in enumerate(img_idxs_list):
        bbox_masks[:, img_idxs, ref_idx] = True
    
    alpha = torch.tensor([[0.3, 0.7, 0.5], [0.2, 0.8, 0.4]], device=device)
    
    updated_tokens = assemble_block(
        instance_tokens=instance_tokens,
        image_tokens=image_tokens,
        img_idxs_list=img_idxs_list,
        bbox_masks=bbox_masks,
        alpha=alpha,
    )
    print(f"  Input image tokens shape: {image_tokens.shape}")
    print(f"  Output image tokens shape: {updated_tokens.shape}")
    assert updated_tokens.shape == image_tokens.shape, f"Shape mismatch"
    print("  ✓ AssembleAttentionBlock passed\n")
    
    print("All tests passed! ✓")
