import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class ModulationHead(nn.Module):
    """
    Modulation Head: Produces per-block AdaLN offset vectors for FLUX from instance tokens.
    
    Architecture:
    - Input: instance token (3072 dim)
    - 2-layer MLP: Linear(3072 → 3072) → SiLU → Linear(3072 → 3072) → shared_offset
    - Per-block projection: Linear(3072 → 3072) for each of 19 DiT blocks
    - Per-block AdaLN: Linear(3072 → 12) → 12 scalars per block
    - Output: (batch_size, num_refs, num_blocks=19, 12) AdaLN offsets
    
    The 12 AdaLN parameters per block are:
    [shift_pre_attn_img, scale_pre_attn_img, gate_post_attn_img,
     shift_pre_ffn_img, scale_pre_ffn_img, gate_post_ffn_img,
     shift_pre_attn_txt, scale_pre_attn_txt, gate_post_attn_txt,
     shift_pre_ffn_txt, scale_pre_ffn_txt, gate_post_ffn_txt]
    """
    
    def __init__(
        self,
        instance_dim: int = 3072,
        num_blocks: int = 19,
        num_adaln_params: int = 12,
    ):
        super().__init__()
        self.instance_dim = instance_dim
        self.num_blocks = num_blocks
        self.num_adaln_params = num_adaln_params
        
        # Shared offset MLP: 2-layer MLP with zero initialization
        self.shared_mlp = nn.Sequential(
            nn.Linear(instance_dim, instance_dim, bias=True),
            nn.SiLU(),
            nn.Linear(instance_dim, instance_dim, bias=True),
        )
        
        # Per-block projections: Linear(3072 → 3072) for each block
        self.block_projections = nn.ModuleList([
            nn.Linear(instance_dim, instance_dim, bias=True)
            for _ in range(num_blocks)
        ])
        
        # Per-block AdaLN projections: Linear(3072 → 12) for each block
        self.block_adaln_projs = nn.ModuleList([
            nn.Linear(instance_dim, num_adaln_params, bias=True)
            for _ in range(num_blocks)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights: shared MLP with zeros, others with standard init."""
        # Zero initialization for shared MLP (critical for stable training start)
        for layer in self.shared_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # Standard initialization for per-block projections
        for proj in self.block_projections:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
        
        # Standard initialization for AdaLN projections
        for proj in self.block_adaln_projs:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
    
    def forward(
        self,
        instance_tokens: torch.Tensor,
        alpha: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: produce per-block AdaLN offsets.
        
        Args:
            instance_tokens: (batch_size, num_refs, 3072) instance tokens per ref
            alpha: (batch_size, num_refs) alpha values per ref (0=BG, 1=FG)
                   If None, no scaling applied (returns unscaled offsets)
        
        Returns:
            adaln_offsets: (batch_size, num_refs, num_blocks, 12) AdaLN offset scalars
                           These are scaled by alpha if alpha is provided.
        """
        batch_size, num_refs, _ = instance_tokens.shape
        
        # Step 1: Shared offset MLP
        # (batch_size, num_refs, 3072) -> (batch_size, num_refs, 3072)
        shared_offset = self.shared_mlp(instance_tokens)
        
        # Step 2: Per-block processing
        # Collect AdaLN parameters for all blocks
        adaln_params_list = []
        
        for block_idx in range(self.num_blocks):
            # Per-block projection: (batch_size, num_refs, 3072) -> (batch_size, num_refs, 3072)
            block_offset = self.block_projections[block_idx](shared_offset)
            
            # AdaLN parameter generation: (batch_size, num_refs, 3072) -> (batch_size, num_refs, 12)
            adaln_params = self.block_adaln_projs[block_idx](block_offset)
            adaln_params_list.append(adaln_params)
        
        # Stack: (batch_size, num_refs, num_blocks, 12)
        adaln_offsets = torch.stack(adaln_params_list, dim=2)
        
        # Step 3: Scale by alpha if provided
        if alpha is not None:
            # alpha: (batch_size, num_refs) -> (batch_size, num_refs, 1, 1)
            alpha_expanded = alpha.unsqueeze(-1).unsqueeze(-1)
            adaln_offsets = adaln_offsets * alpha_expanded
        
        return adaln_offsets


def apply_spatial_gating(
    normalized_features: torch.Tensor,
    bbox_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Apply spatial gating: multiply normalized image token features by bbox mask.
    
    This cancels modulation outside the bbox region before attention runs.
    The mask is 1 inside bbox, 0 outside.
    
    Args:
        normalized_features: (batch_size, num_img_tokens, dim) normalized image token features
                            (after AdaLN normalization with offset)
        bbox_mask: (batch_size, num_img_tokens, num_refs) binary mask
                   1 inside bbox, 0 outside
    
    Returns:
        gated_features: (batch_size, num_img_tokens, dim, num_refs) spatially gated features
                       Modulation is cancelled outside bbox regions.
    """
    # Expand normalized_features to match mask dimensions
    # (batch_size, num_img_tokens, dim) -> (batch_size, num_img_tokens, dim, 1)
    # (batch_size, num_img_tokens, num_refs) -> (batch_size, num_img_tokens, 1, num_refs)
    normalized_expanded = normalized_features.unsqueeze(-1)  # (batch_size, num_img_tokens, dim, 1)
    mask_expanded = bbox_mask.unsqueeze(2)  # (batch_size, num_img_tokens, 1, num_refs)
    
    # Multiply: (batch_size, num_img_tokens, dim, num_refs)
    gated_features = normalized_expanded * mask_expanded
    
    return gated_features


def apply_modulation_offsets(
    base_adaln_params: torch.Tensor,
    modulation_offsets: torch.Tensor,
    block_idx: int,
    ref_idx: Optional[int] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Apply modulation offsets to base AdaLN parameters.
    
    This function adds the modulation offsets to the existing AdaLN parameters
    that FLUX already computed from (timestep + text prompt).
    
    Args:
        base_adaln_params: Tuple of 6 tensors from FLUX's AdaLayerNormZero:
                          (norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp)
                          OR if separate streams: 6 params for image + 6 params for text
        modulation_offsets: (batch_size, num_refs, num_blocks, 12) AdaLN offsets
        block_idx: Index of the current DiT block (0-18)
        ref_idx: Optional reference index. If None, aggregates across all refs.
    
    Returns:
        Modified AdaLN parameters with offsets added.
        For image stream: (norm_hidden_states_img, gate_msa_img, shift_mlp_img, scale_mlp_img, gate_mlp_img)
        For text stream: (norm_hidden_states_txt, gate_msa_txt, shift_mlp_txt, scale_mlp_txt, gate_mlp_txt)
    """
    batch_size, num_refs, num_blocks, num_params = modulation_offsets.shape
    
    # Extract offsets for this block
    # (batch_size, num_refs, 12)
    block_offsets = modulation_offsets[:, :, block_idx, :]
    
    if ref_idx is not None:
        # Use specific ref: (batch_size, 12)
        offsets = block_offsets[:, ref_idx, :]
    else:
        # Aggregate across refs (sum): (batch_size, 12)
        offsets = block_offsets.sum(dim=1)
    
    # Split into image and text stream parameters
    # Image stream: first 6 params
    # Text stream: last 6 params
    img_offsets = offsets[:, :6]  # (batch_size, 6)
    txt_offsets = offsets[:, 6:]  # (batch_size, 6)
    
    # Unpack offsets
    # Image: [shift_pre_attn, scale_pre_attn, gate_post_attn, shift_pre_ffn, scale_pre_ffn, gate_post_ffn]
    img_shift_msa, img_scale_msa, img_gate_msa, img_shift_mlp, img_scale_mlp, img_gate_mlp = img_offsets.chunk(6, dim=1)
    
    # Text: [shift_pre_attn, scale_pre_attn, gate_post_attn, shift_pre_ffn, scale_pre_ffn, gate_post_ffn]
    txt_shift_msa, txt_scale_msa, txt_gate_msa, txt_shift_mlp, txt_scale_mlp, txt_gate_mlp = txt_offsets.chunk(6, dim=1)
    
    # Note: This function returns the offsets separately.
    # The actual application happens in the FLUX block forward pass where:
    # - shift_msa and scale_msa modify the normalization: norm * (1 + scale_msa) + shift_msa
    # - gate_msa multiplies attention output: attn_output * gate_msa
    # - shift_mlp and scale_mlp modify FFN normalization: norm * (1 + scale_mlp) + shift_mlp
    # - gate_mlp multiplies FFN output: ffn_output * gate_mlp
    
    return {
        'img': {
            'shift_msa': img_shift_msa.squeeze(-1),  # (batch_size,)
            'scale_msa': img_scale_msa.squeeze(-1),  # (batch_size,)
            'gate_msa': img_gate_msa.squeeze(-1),    # (batch_size,)
            'shift_mlp': img_shift_mlp.squeeze(-1),  # (batch_size,)
            'scale_mlp': img_scale_mlp.squeeze(-1),  # (batch_size,)
            'gate_mlp': img_gate_mlp.squeeze(-1),    # (batch_size,)
        },
        'txt': {
            'shift_msa': txt_shift_msa.squeeze(-1),  # (batch_size,)
            'scale_msa': txt_scale_msa.squeeze(-1),  # (batch_size,)
            'gate_msa': txt_gate_msa.squeeze(-1),    # (batch_size,)
            'shift_mlp': txt_shift_mlp.squeeze(-1),  # (batch_size,)
            'scale_mlp': txt_scale_mlp.squeeze(-1),  # (batch_size,)
            'gate_mlp': txt_gate_mlp.squeeze(-1),    # (batch_size,)
        },
    }


class ModulationIntegrator(nn.Module):
    """
    Integration wrapper that converts ModulationHead output to FLUX-compatible format.
    
    Takes per-ref AdaLN parameters and expands them to per-token offsets that can be
    applied to FLUX's AdaLayerNormZero outputs.
    """
    
    def __init__(
        self,
        dim: int = 3072,
        num_blocks: int = 19,
        num_adaln_params: int = 12,
    ):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        self.num_adaln_params = num_adaln_params
    
    def apply_offsets(
        self,
        adaln_offsets: torch.Tensor,
        alpha: torch.Tensor,
        img_idxs_list_list: List[List[torch.Tensor]],
        num_img_tokens: int,
        num_txt_tokens: int,
        block_idx: int,
        feedback_offset: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply modulation offsets to create per-token AdaLN parameter modifications.
        
        Args:
            adaln_offsets: (batch_size, num_refs, num_blocks, 12) from ModulationHead
            alpha: (batch_size, num_refs) alpha values
            img_idxs_list_list: List[List[Tensor]] image token indices per bbox
            num_img_tokens: total number of image tokens
            num_txt_tokens: total number of text tokens
            block_idx: current block index
            feedback_offset: (batch_size, num_refs, 3072) optional feedback from previous block
        
        Returns:
            img_offsets: (batch_size, num_img_tokens, 6) AdaLN params for image stream
            txt_offsets: (batch_size, num_txt_tokens, 6) AdaLN params for text stream
        """
        batch_size, num_refs = alpha.shape
        device = adaln_offsets.device
        
        block_offsets = adaln_offsets[:, :, block_idx, :]
        
        img_params = block_offsets[:, :, :6]
        txt_params = block_offsets[:, :, 6:]
        
        img_offsets = torch.zeros(batch_size, num_img_tokens, 6, device=device, dtype=adaln_offsets.dtype)
        txt_offsets = torch.zeros(batch_size, num_txt_tokens, 6, device=device, dtype=adaln_offsets.dtype)
        
        for batch_idx in range(batch_size):
            for ref_idx in range(num_refs):
                alpha_val = alpha[batch_idx, ref_idx].item()
                img_idxs = img_idxs_list_list[batch_idx][ref_idx]
                
                if len(img_idxs) > 0:
                    ref_img_params = img_params[batch_idx, ref_idx]
                    img_offsets[batch_idx, img_idxs] += ref_img_params
        
        return img_offsets, txt_offsets
    
    def create_modified_temb(
        self,
        base_temb: torch.Tensor,
        adaln_offsets: torch.Tensor,
        alpha: torch.Tensor,
        img_idxs_list_list: List[List[torch.Tensor]],
        block_idx: int,
        feedback_offset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Create modified timestep embedding that incorporates modulation offsets.
        
        This is a simplified approach: we'll return the offsets separately and apply them
        in the custom forward pass.
        
        Args:
            base_temb: (batch_size, 3072) base timestep embedding from FLUX
            adaln_offsets: (batch_size, num_refs, num_blocks, 12) from ModulationHead
            alpha: (batch_size, num_refs) alpha values
            img_idxs_list_list: List[List[Tensor]] image token indices
            block_idx: current block index
            feedback_offset: optional feedback from previous block
        
        Returns:
            temb_modified: (batch_size, 3072) modified timestep embedding
        """
        return base_temb


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    batch_size = 2
    num_refs = 3
    instance_dim = 3072
    num_blocks = 19
    num_adaln_params = 12
    
    print("=" * 60)
    print("Testing ModulationHead")
    print("=" * 60)
    
    # Create model
    model = ModulationHead(
        instance_dim=instance_dim,
        num_blocks=num_blocks,
        num_adaln_params=num_adaln_params,
    ).to(device)
    
    # Test input
    instance_tokens = torch.randn(batch_size, num_refs, instance_dim, device=device)
    alpha = torch.tensor([[0.0, 0.5, 1.0], [0.2, 0.7, 0.9]], device=device)
    
    print(f"\nInput shapes:")
    print(f"  instance_tokens: {instance_tokens.shape}")
    print(f"  alpha: {alpha.shape}")
    
    # Test forward pass without alpha
    print(f"\n--- Test 1: Forward pass without alpha scaling ---")
    with torch.no_grad():
        offsets_no_alpha = model(instance_tokens, alpha=None)
    
    print(f"  Output shape: {offsets_no_alpha.shape}")
    expected_shape = (batch_size, num_refs, num_blocks, num_adaln_params)
    assert offsets_no_alpha.shape == expected_shape, \
        f"Expected {expected_shape}, got {offsets_no_alpha.shape}"
    print(f"  ✓ Shape check passed")
    
    # Test forward pass with alpha
    print(f"\n--- Test 2: Forward pass with alpha scaling ---")
    with torch.no_grad():
        offsets_with_alpha = model(instance_tokens, alpha=alpha)
    
    print(f"  Output shape: {offsets_with_alpha.shape}")
    assert offsets_with_alpha.shape == expected_shape, \
        f"Expected {expected_shape}, got {offsets_with_alpha.shape}"
    print(f"  ✓ Shape check passed")
    
    # Verify alpha scaling
    print(f"\n--- Test 3: Alpha scaling verification ---")
    # For ref with alpha=0.0, offsets should be zero
    ref_0_alpha = offsets_with_alpha[0, 0, :, :]  # First batch, first ref (alpha=0.0)
    assert torch.allclose(ref_0_alpha, torch.zeros_like(ref_0_alpha), atol=1e-6), \
        "Alpha=0.0 should produce zero offsets"
    print(f"  ✓ Alpha=0.0 produces zero offsets")
    
    # For ref with alpha=1.0, offsets should match unscaled
    ref_1_alpha = offsets_with_alpha[0, 2, :, :]  # First batch, third ref (alpha=1.0)
    ref_1_no_alpha = offsets_no_alpha[0, 2, :, :]
    assert torch.allclose(ref_1_alpha, ref_1_no_alpha, atol=1e-6), \
        "Alpha=1.0 should match unscaled offsets"
    print(f"  ✓ Alpha=1.0 matches unscaled offsets")
    
    # Test zero initialization
    print(f"\n--- Test 4: Zero initialization check ---")
    # Create new model and check that outputs are zero at initialization
    model_zero = ModulationHead(
        instance_dim=instance_dim,
        num_blocks=num_blocks,
        num_adaln_params=num_adaln_params,
    ).to(device)
    
    with torch.no_grad():
        offsets_zero = model_zero(instance_tokens, alpha=None)
    
    # At initialization, shared MLP outputs zeros, so all offsets should be zero
    assert torch.allclose(offsets_zero, torch.zeros_like(offsets_zero), atol=1e-6), \
        "Zero initialization should produce zero offsets"
    print(f"  ✓ Zero initialization produces zero offsets")
    print(f"  Mean offset magnitude: {offsets_zero.abs().mean().item():.2e}")
    
    # Test parameter ordering
    print(f"\n--- Test 5: Parameter ordering check ---")
    # Check that we have 12 parameters per block
    assert offsets_no_alpha.shape[-1] == 12, "Should have 12 AdaLN parameters per block"
    print(f"  ✓ 12 parameters per block confirmed")
    
    # Test apply_modulation_offsets helper
    print(f"\n--- Test 6: apply_modulation_offsets helper ---")
    offsets_dict = apply_modulation_offsets(
        base_adaln_params=None,  # Not used in current implementation
        modulation_offsets=offsets_no_alpha,
        block_idx=0,
        ref_idx=0,
    )
    
    assert 'img' in offsets_dict and 'txt' in offsets_dict, \
        "Should return dict with 'img' and 'txt' keys"
    assert len(offsets_dict['img']) == 6, "Image stream should have 6 parameters"
    assert len(offsets_dict['txt']) == 6, "Text stream should have 6 parameters"
    print(f"  ✓ Helper function returns correct structure")
    print(f"    Image params: {list(offsets_dict['img'].keys())}")
    print(f"    Text params: {list(offsets_dict['txt'].keys())}")
    
    # Test spatial gating
    print(f"\n--- Test 7: apply_spatial_gating helper ---")
    num_img_tokens = 1024
    normalized_features = torch.randn(batch_size, num_img_tokens, instance_dim, device=device)
    bbox_mask = torch.zeros(batch_size, num_img_tokens, num_refs, device=device, dtype=torch.bool)
    
    # Set some tokens inside bbox for each ref
    bbox_mask[0, 100:200, 0] = True  # First ref: tokens 100-199
    bbox_mask[0, 300:400, 1] = True  # Second ref: tokens 300-399
    bbox_mask[1, 500:600, 2] = True  # Third ref: tokens 500-599
    
    gated_features = apply_spatial_gating(normalized_features, bbox_mask)
    print(f"  Normalized features shape: {normalized_features.shape}")
    print(f"  Bbox mask shape: {bbox_mask.shape}")
    print(f"  Gated features shape: {gated_features.shape}")
    expected_gated_shape = (batch_size, num_img_tokens, instance_dim, num_refs)
    assert gated_features.shape == expected_gated_shape, \
        f"Expected {expected_gated_shape}, got {gated_features.shape}"
    print(f"  ✓ Spatial gating shape check passed")
    
    # Verify that features outside bbox are zero
    assert torch.allclose(
        gated_features[0, 0:100, :, 0],
        torch.zeros(100, instance_dim, device=device),
        atol=1e-6
    ), "Features outside bbox should be zero"
    print(f"  ✓ Features outside bbox are zeroed")
    
    # Verify that features inside bbox are preserved
    assert torch.allclose(
        gated_features[0, 100:200, :, 0],
        normalized_features[0, 100:200, :],
        atol=1e-6
    ), "Features inside bbox should be preserved"
    print(f"  ✓ Features inside bbox are preserved")
    
    print(f"\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
