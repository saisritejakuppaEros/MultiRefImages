"""
LoRA Application Utilities for FLUX Transformer
Applies LoRA adapters to FLUX attention layers for efficient fine-tuning
"""

import torch
import torch.nn as nn
from typing import List, Optional
from peft import LoraConfig, get_peft_model, TaskType


def apply_lora_to_flux(
    flux_transformer: nn.Module,
    lora_rank: int = 128,
    lora_alpha: int = 128,
    lora_dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Apply LoRA adapters to FLUX transformer attention layers.
    
    Args:
        flux_transformer: FLUX transformer model
        lora_rank: LoRA rank (default: 128 for Assemble-Attn)
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to
    
    Returns:
        flux_transformer with LoRA adapters applied
    """
    if target_modules is None:
        # Default: apply to Q, K, V projections in attention layers
        target_modules = ["to_q", "to_k", "to_v"]
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,  # Not causal LM
    )
    
    # Apply LoRA
    flux_transformer = get_peft_model(flux_transformer, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in flux_transformer.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in flux_transformer.parameters())
    
    print(f"✓ Applied LoRA to FLUX transformer")
    print(f"  LoRA rank: {lora_rank}")
    print(f"  LoRA alpha: {lora_alpha}")
    print(f"  Target modules: {target_modules}")
    print(f"  Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return flux_transformer


def get_lora_state_dict(model: nn.Module) -> dict:
    """
    Extract only LoRA parameters from model state dict.
    
    Args:
        model: Model with LoRA adapters
    
    Returns:
        State dict containing only LoRA parameters
    """
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_state_dict[name] = param
    return lora_state_dict


def load_lora_weights(model: nn.Module, lora_state_dict: dict, strict: bool = False):
    """
    Load LoRA weights into model.
    
    Args:
        model: Model with LoRA adapters
        lora_state_dict: State dict containing LoRA parameters
        strict: Whether to strictly enforce all keys match
    """
    model.load_state_dict(lora_state_dict, strict=strict)
    print(f"✓ Loaded LoRA weights ({len(lora_state_dict)} parameters)")


def freeze_non_lora_params(model: nn.Module):
    """
    Freeze all non-LoRA parameters in the model.
    
    Args:
        model: Model with LoRA adapters
    """
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Frozen non-LoRA parameters ({trainable_params:,} trainable)")


def print_lora_info(model: nn.Module):
    """
    Print information about LoRA adapters in the model.
    
    Args:
        model: Model with LoRA adapters
    """
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_params.append((name, param.shape, param.numel()))
    
    if not lora_params:
        print("No LoRA parameters found in model")
        return
    
    print(f"\nLoRA Parameters ({len(lora_params)} layers):")
    print("-" * 80)
    total_lora_params = 0
    for name, shape, numel in lora_params[:10]:  # Show first 10
        print(f"  {name:60s} {str(shape):20s} {numel:>10,}")
        total_lora_params += numel
    
    if len(lora_params) > 10:
        print(f"  ... and {len(lora_params) - 10} more layers")
        for name, shape, numel in lora_params[10:]:
            total_lora_params += numel
    
    print("-" * 80)
    print(f"Total LoRA parameters: {total_lora_params:,}")


if __name__ == "__main__":
    print("LoRA Application Utilities for FLUX")
    print("=" * 80)
    print("\nThis module provides utilities for applying LoRA adapters to FLUX transformer.")
    print("\nUsage:")
    print("  from apply_lora import apply_lora_to_flux")
    print("  flux_transformer = apply_lora_to_flux(flux_transformer, lora_rank=128)")
    print("\nLoRA Configuration:")
    print("  - Stage 1: rank=128, alpha=128 (Assemble-Attn on Q,K,V)")
    print("  - Stage 2: Same as Stage 1")
    print("  - Stage 3: rank=16, alpha=16 (Feedback bridge only)")
