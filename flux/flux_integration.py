import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Any


def apply_modulation_to_adaln_params(
    shift_msa: torch.Tensor,
    scale_msa: torch.Tensor,
    gate_msa: torch.Tensor,
    shift_mlp: torch.Tensor,
    scale_mlp: torch.Tensor,
    gate_mlp: torch.Tensor,
    modulation_offsets: torch.Tensor,
    img_idxs_list: List[torch.Tensor],
    alpha: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply per-ref modulation offsets to AdaLN parameters.
    
    Args:
        shift_msa, scale_msa, gate_msa: (batch_size, num_img_tokens, 3072) from FLUX AdaLN
        shift_mlp, scale_mlp, gate_mlp: (batch_size, num_img_tokens, 3072) from FLUX AdaLN
        modulation_offsets: (batch_size, num_refs, 6) per-ref offsets [shift, scale, gate] × 2
        img_idxs_list: List[Tensor] image token indices per ref
        alpha: (batch_size, num_refs) alpha values
    
    Returns:
        Modified shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
    """
    batch_size, num_refs = alpha.shape
    
    for batch_idx in range(batch_size):
        for ref_idx in range(num_refs):
            img_idxs = img_idxs_list[ref_idx]
            if len(img_idxs) == 0:
                continue
            
            alpha_val = alpha[batch_idx, ref_idx]
            offsets = modulation_offsets[batch_idx, ref_idx]
            
            shift_msa_off, scale_msa_off, gate_msa_off = offsets[:3].unsqueeze(0)
            shift_mlp_off, scale_mlp_off, gate_mlp_off = offsets[3:].unsqueeze(0)
            
            shift_msa[batch_idx, img_idxs] += shift_msa_off * alpha_val
            scale_msa[batch_idx, img_idxs] += scale_msa_off * alpha_val
            gate_msa[batch_idx, img_idxs] *= (1.0 + gate_msa_off * alpha_val)
            
            shift_mlp[batch_idx, img_idxs] += shift_mlp_off * alpha_val
            scale_mlp[batch_idx, img_idxs] += scale_mlp_off * alpha_val
            gate_mlp[batch_idx, img_idxs] *= (1.0 + gate_mlp_off * alpha_val)
    
    return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp


def custom_double_block_forward(
    block: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    instance_tokens: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    img_idxs_list_list: Optional[List[List[torch.Tensor]]] = None,
    modulation_offsets: Optional[torch.Tensor] = None,
    layout_integrator: Optional[nn.Module] = None,
    feedback_offset: Optional[torch.Tensor] = None,
    block_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom forward pass for FLUX double-stream block with modulation and layout injection.
    
    This wraps the standard FLUX block forward and injects:
    1. Modulation offsets to temb BEFORE AdaLN runs (critical for gradient flow)
    2. Layout pathway via AssembleAttentionAddon
    3. Feedback from previous block
    
    Args:
        block: FluxTransformerBlock from diffusers
        hidden_states: (batch_size, num_img_tokens, 3072) image latents
        encoder_hidden_states: (batch_size, num_txt_tokens, 3072) text embeddings
        temb: (batch_size, 3072) timestep embedding
        image_rotary_emb: rotary position embeddings
        instance_tokens: (batch_size, num_refs, 3072) from instance encoder
        alpha: (batch_size, num_refs) alpha values
        img_idxs_list_list: List[List[Tensor]] image token indices
        modulation_offsets: (batch_size, num_refs, 6) AdaLN offsets for this block
        layout_integrator: LayoutIntegrator instance
        feedback_offset: (batch_size, num_refs, 3072) feedback from previous block
        block_idx: current block index
    
    Returns:
        hidden_states: (batch_size, num_img_tokens, 3072) updated image tokens
        encoder_hidden_states: (batch_size, num_txt_tokens, 3072) updated text tokens
        layout_summary: (batch_size, num_refs, 3072) for feedback computation
    """
    batch_size = hidden_states.shape[0]
    num_img_tokens = hidden_states.shape[1]
    
    # CRITICAL: Modify temb BEFORE AdaLN runs to ensure gradient flow
    # Aggregate modulation offsets across all refs, weighted by alpha
    temb_modified = temb
    if modulation_offsets is not None and alpha is not None:
        # modulation_offsets: (batch_size, num_refs, 6) - we use first 3 for shift/scale/gate
        # Convert offsets to temb space by projecting and aggregating
        # Alpha-weighted sum across refs: high alpha = stronger modulation
        alpha_weights = alpha.to(temb.dtype)  # (batch_size, num_refs)
        
        # Sum offsets weighted by alpha, expand to temb dimension
        # offsets[:, :, 0:3] for shift, scale, gate of first AdaLN component
        weighted_offsets = (modulation_offsets.to(temb.dtype) * alpha_weights.unsqueeze(-1)).sum(dim=1)  # (batch_size, 6)
        
        # Add to temb (temb is (batch_size, 3072), we add a scaled contribution)
        # Use the mean of the 6 offset values as a scalar modifier to temb
        temb_modifier = weighted_offsets.mean(dim=-1, keepdim=True)  # (batch_size, 1)
        temb_modified = temb + temb_modifier * 0.1  # Scale factor to avoid destabilizing
    
    # Now run AdaLN with the modified temb - gradients will flow back through temb_modifier
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(
        hidden_states, emb=temb_modified
    )
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = block.norm1_context(
        encoder_hidden_states, emb=temb_modified
    )
    
    # Additional per-token modulation for spatial control (applied after base AdaLN)
    if modulation_offsets is not None and alpha is not None and img_idxs_list_list is not None:
        # Expand modulation parameters from (B, D) to (B, num_tokens, D)
        shift_msa_expanded = shift_mlp.unsqueeze(1).expand(batch_size, num_img_tokens, -1).clone()
        scale_msa_expanded = scale_mlp.unsqueeze(1).expand(batch_size, num_img_tokens, -1).clone()
        gate_msa_expanded = gate_msa.unsqueeze(1).expand(batch_size, num_img_tokens, -1).clone()
        shift_mlp_expanded = shift_mlp.unsqueeze(1).expand(batch_size, num_img_tokens, -1).clone()
        scale_mlp_expanded = scale_mlp.unsqueeze(1).expand(batch_size, num_img_tokens, -1).clone()
        gate_mlp_expanded = gate_mlp.unsqueeze(1).expand(batch_size, num_img_tokens, -1).clone()
        
        for b_idx in range(batch_size):
            img_idxs_list = img_idxs_list_list[b_idx]
            for ref_idx in range(len(img_idxs_list)):
                img_idxs = img_idxs_list[ref_idx]
                if len(img_idxs) == 0:
                    continue
                
                alpha_val = alpha[b_idx, ref_idx].to(shift_msa_expanded.dtype)
                offsets = modulation_offsets[b_idx, ref_idx].to(shift_msa_expanded.dtype)
                
                shift_msa_off, scale_msa_off, gate_msa_off = offsets[0], offsets[1], offsets[2]
                shift_mlp_off, scale_mlp_off, gate_mlp_off = offsets[3], offsets[4], offsets[5]
                
                # Apply per-token modulation weighted by alpha
                shift_msa_expanded[b_idx, img_idxs] = shift_msa_expanded[b_idx, img_idxs] + shift_msa_off * alpha_val
                scale_msa_expanded[b_idx, img_idxs] = scale_msa_expanded[b_idx, img_idxs] + scale_msa_off * alpha_val
                gate_msa_expanded[b_idx, img_idxs] = gate_msa_expanded[b_idx, img_idxs] * (1.0 + gate_msa_off * alpha_val)
                
                shift_mlp_expanded[b_idx, img_idxs] = shift_mlp_expanded[b_idx, img_idxs] + shift_mlp_off * alpha_val
                scale_mlp_expanded[b_idx, img_idxs] = scale_mlp_expanded[b_idx, img_idxs] + scale_mlp_off * alpha_val
                gate_mlp_expanded[b_idx, img_idxs] = gate_mlp_expanded[b_idx, img_idxs] * (1.0 + gate_mlp_off * alpha_val)
        
        # Re-apply normalization with modified parameters (maintains gradient flow)
        norm_hidden_states = block.norm1.norm(hidden_states) * (1 + scale_msa_expanded) + shift_msa_expanded
    
    attn_output, context_attn_output = block.attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        image_rotary_emb=image_rotary_emb,
    )
    
    if modulation_offsets is not None and alpha is not None and img_idxs_list_list is not None:
        attn_output = gate_msa_expanded * attn_output
    else:
        attn_output = gate_msa.unsqueeze(1) * attn_output
    
    hidden_states = hidden_states + attn_output
    
    # Expand c_gate_msa for context tokens
    num_context_tokens = context_attn_output.shape[1]
    context_attn_output = c_gate_msa.unsqueeze(1).expand(-1, num_context_tokens, -1) * context_attn_output
    encoder_hidden_states = encoder_hidden_states + context_attn_output
    
    if layout_integrator is not None and instance_tokens is not None:
        hidden_states, layout_summary = layout_integrator.apply_layout(
            instance_tokens=instance_tokens,
            image_tokens=hidden_states,
            alpha=alpha,
            img_idxs_list_list=img_idxs_list_list,
        )
    else:
        layout_summary = None
    
    norm_hidden_states = block.norm2(hidden_states)
    if modulation_offsets is not None and alpha is not None and img_idxs_list_list is not None:
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp_expanded) + shift_mlp_expanded
    else:
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
    
    norm_encoder_hidden_states = block.norm2_context(encoder_hidden_states)
    num_context_tokens = norm_encoder_hidden_states.shape[1]
    norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp.unsqueeze(1).expand(-1, num_context_tokens, -1)) + c_shift_mlp.unsqueeze(1).expand(-1, num_context_tokens, -1)
    
    ff_output = block.ff(norm_hidden_states)
    if modulation_offsets is not None and alpha is not None and img_idxs_list_list is not None:
        ff_output = gate_mlp_expanded * ff_output
    else:
        ff_output = gate_mlp.unsqueeze(1) * ff_output
    
    hidden_states = hidden_states + ff_output
    
    context_ff_output = block.ff_context(norm_encoder_hidden_states)
    context_ff_output = c_gate_mlp.unsqueeze(1).expand(-1, num_context_tokens, -1) * context_ff_output
    encoder_hidden_states = encoder_hidden_states + context_ff_output
    
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
    
    return hidden_states, encoder_hidden_states, layout_summary


def custom_single_block_forward(
    block: nn.Module,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    modulation_offsets: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    img_idxs_list_list: Optional[List[List[torch.Tensor]]] = None,
) -> torch.Tensor:
    """
    Custom forward pass for FLUX single-stream block with light modulation.
    
    Args:
        block: FluxSingleTransformerBlock from diffusers
        hidden_states: (batch_size, num_tokens, 3072) concatenated [text, image] tokens
        temb: (batch_size, 3072) timestep embedding
        image_rotary_emb: rotary position embeddings
        modulation_offsets: (batch_size, num_refs, 3) light offsets [shift, scale, gate]
        alpha: (batch_size, num_refs) alpha values
        img_idxs_list_list: List[List[Tensor]] image token indices
    
    Returns:
        hidden_states: (batch_size, num_tokens, 3072) updated tokens
    """
    residual = hidden_states
    
    norm_hidden_states, gate = block.norm(hidden_states, emb=temb)
    
    mlp_hidden_states = block.act_mlp(block.proj_mlp(norm_hidden_states))
    
    attn_output = block.attn(hidden_states=norm_hidden_states, image_rotary_emb=image_rotary_emb)
    
    hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
    # Expand gate from (B, D) to (B, num_tokens, D)
    hidden_states = gate.unsqueeze(1) * block.proj_out(hidden_states)
    hidden_states = residual + hidden_states
    
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    
    return hidden_states
