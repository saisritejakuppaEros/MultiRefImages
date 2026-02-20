import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .layout_module import LayoutHead, split_layout_kv, assemble_attention
from .flux_utils import create_bbox_mask


class AssembleAttentionAddon(nn.Module):
    """
    Addon module for FLUX transformer blocks that implements Assemble-Attn using LayoutHead.
    
    This module:
    1. Takes instance tokens (3072-dim) and produces layout_kv via LayoutHead
    2. Scales layout_kv by (1 - α) before entering Assemble-Attn
    3. Splits layout_kv into K and V
    4. Uses Q from image tokens inside bbox region
    5. Applies spatial gating (hard binary mask)
    6. Outputs updated image tokens inside bbox region
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
        
        # Layout head: projects instance token to layout_kv
        self.layout_head = LayoutHead(dim=dim)
        
        # Projection for Q from image tokens
        self.to_q = nn.Linear(dim, num_heads * head_dim, bias=False)
        
        # Output projection
        self.to_out = nn.Linear(num_heads * head_dim, dim, bias=False)
    
    def forward(
        self,
        instance_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        img_idxs_list_list: List[List[torch.Tensor]],
        layout_masks: torch.Tensor,
        alpha: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            instance_tokens: (batch_size, num_refs, 3072) - instance tokens per ref
            image_tokens: (batch_size, num_img_tokens, 3072) - all image tokens
            img_idxs_list_list: List[List[torch.Tensor]] - img_idxs_list per batch item
            layout_masks: (batch_size, num_refs) - binary mask indicating valid refs
            alpha: (batch_size, num_refs) - alpha values per ref (0=BG, 1=FG)
            batch_indices: optional (batch_size,) tensor for batch mapping
        
        Returns:
            updated_image_tokens: (batch_size, num_img_tokens, 3072) - updated image tokens
            layout_outputs: (batch_size, num_refs, 3072) - layout token outputs (for compatibility)
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
        
        # Step 4: Create bbox masks for spatial gating
        if batch_indices is None:
            batch_indices = torch.arange(batch_size, device=image_tokens.device)
        
        # Process each batch item and ref
        updated_tokens = image_tokens.clone()
        layout_outputs = torch.zeros_like(layout_kv)
        
        valid_mask = (layout_masks == 1)
        valid_indices = valid_mask.nonzero(as_tuple=False)
        
        if valid_indices.size(0) > 0:
            for idx in range(valid_indices.size(0)):
                i = valid_indices[idx, 0].item()  # batch index
                j = valid_indices[idx, 1].item()  # ref index
                
                img_idxs = img_idxs_list_list[i][j]  # (num_bbox_tokens,)
                if len(img_idxs) == 0:
                    continue
                
                # Extract Q from image tokens inside bbox
                q_tokens = image_tokens[i:i+1, img_idxs, :]  # (1, num_bbox_tokens, 3072)
                q = self.to_q(q_tokens)  # (1, num_bbox_tokens, num_heads * head_dim)
                q = q.view(1, len(img_idxs), self.num_heads, self.head_dim)
                
                # Get K and V for this ref
                k_ref = k[i:i+1, j:j+1, :, :]  # (1, 1, num_heads, head_dim)
                v_ref = v[i:i+1, j:j+1, :, :]  # (1, 1, num_heads, head_dim)
                
                # Create mask for this ref's bbox (all True since we already filtered by img_idxs)
                mask_ref = torch.ones(1, len(img_idxs), 1, device=image_tokens.device, dtype=torch.bool)
                
                # Compute attention
                attn_output = assemble_attention(
                    q=q,
                    k=k_ref,
                    v=v_ref,
                    scale=1.0 / (self.head_dim ** 0.5),
                    mask=mask_ref.squeeze(-1) if mask_ref.shape[-1] == 1 else mask_ref,
                )  # (1, num_bbox_tokens, num_heads, head_dim)
                
                # Project output
                attn_output_flat = attn_output.reshape(1, len(img_idxs), self.num_heads * self.head_dim)
                output = self.to_out(attn_output_flat)  # (1, num_bbox_tokens, 3072)
                
                # Write back to image tokens (ensure dtype matches)
                updated_tokens[i:i+1, img_idxs, :] = output.to(updated_tokens.dtype)
                
                # Store layout output (mean of attended values)
                layout_outputs[i, j] = layout_kv[i, j]
        
        return updated_tokens, layout_outputs


def integrate_assemble_attention_into_block(
    transformer_block: nn.Module,
    instance_tokens: torch.Tensor,
    image_tokens: torch.Tensor,
    img_idxs_list_list: List[List[torch.Tensor]],
    layout_masks: torch.Tensor,
    alpha: torch.Tensor,
    assemble_addon: Optional[AssembleAttentionAddon] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Integrate Assemble-Attn into a FLUX transformer block.
    
    This function can be called within a transformer block's forward method to apply
    Assemble-Attn using the LayoutHead.
    
    Args:
        transformer_block: The transformer block (for compatibility, not used directly)
        instance_tokens: (batch_size, num_refs, 3072) instance tokens
        image_tokens: (batch_size, num_img_tokens, 3072) image tokens
        img_idxs_list_list: List[List[torch.Tensor]] img indices per batch/ref
        layout_masks: (batch_size, num_refs) binary mask
        alpha: (batch_size, num_refs) alpha values
        assemble_addon: Optional AssembleAttentionAddon instance
    
    Returns:
        updated_image_tokens: (batch_size, num_img_tokens, 3072)
        layout_outputs: (batch_size, num_refs, 3072)
    """
    if assemble_addon is None:
        assemble_addon = AssembleAttentionAddon(
            dim=instance_tokens.shape[-1],
            num_heads=24,
            head_dim=128,
        ).to(instance_tokens.device)
    
    return assemble_addon(
        instance_tokens=instance_tokens,
        image_tokens=image_tokens,
        img_idxs_list_list=img_idxs_list_list,
        layout_masks=layout_masks,
        alpha=alpha,
    )


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    batch_size = 2
    num_refs = 3
    num_img_tokens = 1024
    dim = 3072
    
    print("Testing AssembleAttentionAddon...")
    assemble_addon = AssembleAttentionAddon(dim=dim).to(device)
    
    instance_tokens = torch.randn(batch_size, num_refs, dim, device=device)
    image_tokens = torch.randn(batch_size, num_img_tokens, dim, device=device)
    
    img_idxs_list_list = [
        [
            torch.tensor([10, 11, 12, 20, 21, 22], device=device),
            torch.tensor([30, 31, 32], device=device),
            torch.tensor([50, 51], device=device),
        ],
        [
            torch.tensor([100, 101, 102], device=device),
            torch.tensor([200, 201], device=device),
            torch.tensor([300], device=device),
        ],
    ]
    
    layout_masks = torch.ones(batch_size, num_refs, device=device, dtype=torch.bool)
    alpha = torch.tensor([[0.3, 0.7, 0.5], [0.2, 0.8, 0.4]], device=device)
    
    updated_tokens, layout_outputs = assemble_addon(
        instance_tokens=instance_tokens,
        image_tokens=image_tokens,
        img_idxs_list_list=img_idxs_list_list,
        layout_masks=layout_masks,
        alpha=alpha,
    )
    
    print(f"  Input image tokens shape: {image_tokens.shape}")
    print(f"  Output image tokens shape: {updated_tokens.shape}")
    print(f"  Layout outputs shape: {layout_outputs.shape}")
    assert updated_tokens.shape == image_tokens.shape, f"Shape mismatch"
    assert layout_outputs.shape == instance_tokens.shape, f"Layout outputs shape mismatch"
    print("  ✓ AssembleAttentionAddon passed\n")
    
    print("Testing integrate_assemble_attention_into_block...")
    dummy_block = nn.Module()
    updated_tokens2, layout_outputs2 = integrate_assemble_attention_into_block(
        transformer_block=dummy_block,
        instance_tokens=instance_tokens,
        image_tokens=image_tokens,
        img_idxs_list_list=img_idxs_list_list,
        layout_masks=layout_masks,
        alpha=alpha,
        assemble_addon=assemble_addon,
    )
    assert torch.allclose(updated_tokens, updated_tokens2), "Integration output mismatch"
    print("  ✓ Integration function passed\n")
    
    print("All tests passed! ✓")
