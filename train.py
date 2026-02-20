#!/usr/bin/env python3
"""
MoviePostProduction Training Script
3-Stage Training Pipeline with Accelerate, TensorBoard logging, and checkpointing
"""

import os
import sys
import json
import yaml
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from PIL import Image
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Import project modules
from model import MoviePostProductionModel
from data_loader import MoviePostProductionDataset, collate_fn
from apply_lora import apply_lora_to_flux


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model(config: Dict, accelerator: Accelerator) -> MoviePostProductionModel:
    """Initialize model and apply layer freezing based on stage."""
    model = MoviePostProductionModel(
        flux_model_name=config['model']['flux_model_name'],
        max_refs=config['model']['max_refs'],
        num_double_blocks=config['model']['num_double_blocks'],
        num_single_blocks=config['model']['num_single_blocks'],
        dim=config['model']['dim'],
        device=accelerator.device,
    )
    
    # Apply LoRA to FLUX transformer if trainable
    trainable_config = config.get('trainable', {})
    if trainable_config.get('assemble_attn_lora', False) and model.flux_transformer is not None:
        lora_config = config.get('lora', {})
        model.flux_transformer = apply_lora_to_flux(
            model.flux_transformer,
            lora_rank=lora_config.get('rank', 128),
            lora_alpha=lora_config.get('alpha', 128),
            lora_dropout=lora_config.get('dropout', 0.0),
            target_modules=lora_config.get('target_modules', ['to_q', 'to_k', 'to_v']),
        )
        # Ensure all LoRA parameters are in bfloat16
        for name, param in model.flux_transformer.named_parameters():
            if 'lora_' in name and param.dtype != torch.bfloat16:
                param.data = param.data.to(torch.bfloat16)
        accelerator.print("✓ Applied LoRA to FLUX transformer")
    
    # Enable gradient checkpointing if configured
    if config['training'].get('gradient_checkpointing', False):
        if hasattr(model.flux_transformer, 'enable_gradient_checkpointing'):
            model.flux_transformer.enable_gradient_checkpointing()
            accelerator.print("✓ Enabled gradient checkpointing")
    
    # Apply layer freezing based on stage
    freeze_config = config.get('freeze', {})
    
    # Freeze FLUX transformer
    if freeze_config.get('flux_transformer', True):
        for param in model.flux_transformer.parameters():
            param.requires_grad = False
        accelerator.print("✓ Frozen FLUX transformer")
    
    # Handle trainable modules - convert to float32 for gradient compatibility
    # This is necessary because PyTorch autograd doesn't fully support bfloat16 gradients
    if trainable_config.get('alpha_predictor', False):
        model.alpha_predictor = model.alpha_predictor.float()
        for param in model.alpha_predictor.parameters():
            param.requires_grad = True
        accelerator.print("✓ Alpha predictor trainable (float32)")
    else:
        for param in model.alpha_predictor.parameters():
            param.requires_grad = False
    
    if trainable_config.get('instance_fusion_mlp', False):
        model.instance_fusion_mlp = model.instance_fusion_mlp.float()
        for param in model.instance_fusion_mlp.parameters():
            param.requires_grad = True
        accelerator.print("✓ Instance fusion MLP trainable (float32)")
    else:
        for param in model.instance_fusion_mlp.parameters():
            param.requires_grad = False
    
    if trainable_config.get('layout_head', False):
        model.layout_head = model.layout_head.float()
        for param in model.layout_head.parameters():
            param.requires_grad = True
        accelerator.print("✓ Layout head trainable (float32)")
    else:
        for param in model.layout_head.parameters():
            param.requires_grad = False
    
    if trainable_config.get('modulation_head', False):
        model.modulation_head = model.modulation_head.float()
        for param in model.modulation_head.parameters():
            param.requires_grad = True
        accelerator.print("✓ Modulation head trainable (float32)")
    else:
        for param in model.modulation_head.parameters():
            param.requires_grad = False
    
    if trainable_config.get('feedback_bridge', False):
        model.feedback_bridge = model.feedback_bridge.float()
        for param in model.feedback_bridge.parameters():
            param.requires_grad = True
        accelerator.print("✓ Feedback bridge trainable (float32)")
    else:
        for param in model.feedback_bridge.parameters():
            param.requires_grad = False
    
    # Convert integrators and auxiliary modules to float32 for compatibility
    model.layout_integrator = model.layout_integrator.float()
    model.modulation_integrator = model.modulation_integrator.float()
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return model


def add_noise_to_latents(latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    """
    Add noise to latents for flow matching.
    
    Args:
        latents: (batch_size, 16, H, W) clean latents
        noise: (batch_size, 16, H, W) random noise
        timesteps: (batch_size,) timestep values in [0, 1]
    
    Returns:
        noisy_latents: (batch_size, 16, H, W)
    """
    # Flow matching: x_t = (1 - t) * x_0 + t * noise
    timesteps = timesteps.view(-1, 1, 1, 1)
    noisy_latents = (1 - timesteps) * latents + timesteps * noise
    return noisy_latents


def compute_flow_matching_loss(
    model_output: torch.Tensor,
    noise: torch.Tensor,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """
    Compute flow matching loss.
    
    The model predicts the velocity field v_t.
    Target velocity: v_t = noise - latents
    """
    target = noise - latents
    loss = F.mse_loss(model_output, target, reduction='mean')
    return loss


def create_position_ids(latent_h: int, latent_w: int, num_txt_tokens: int, device: torch.device) -> tuple:
    """
    Create img_ids and txt_ids for RoPE positional encoding.
    
    Args:
        latent_h: Latent height before packing (e.g., 128)
        latent_w: Latent width before packing (e.g., 128)
        num_txt_tokens: Number of text tokens
        device: Device to place tensors on
    
    Returns:
        img_ids: ((H/2)*(W/2), 3) image position IDs (after 2x2 packing)
        txt_ids: (num_txt_tokens, 3) text position IDs
    """
    # After 2x2 packing, we have (H/2) * (W/2) image tokens
    packed_h = latent_h // 2
    packed_w = latent_w // 2
    
    # Image position IDs ((H/2)*(W/2), 3)
    img_ids = torch.zeros(packed_h * packed_w, 3, device=device)
    for i in range(packed_h):
        for j in range(packed_w):
            idx = i * packed_w + j
            img_ids[idx, 1] = i
            img_ids[idx, 2] = j
    
    # Text position IDs (num_txt_tokens, 3)
    txt_ids = torch.zeros(num_txt_tokens, 3, device=device)
    for i in range(num_txt_tokens):
        txt_ids[i, 2] = i
    
    return img_ids, txt_ids


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """
    Pack latents from (B, 16, H, W) to (B, H*W, 64).
    
    FLUX uses 2x2 patching: each 2x2 spatial patch with 16 channels becomes 64 dims.
    
    Args:
        latents: (B, 16, H, W) unpacked latents
    
    Returns:
        packed_latents: (B, (H/2)*(W/2), 64) packed latents
    """
    B, C, H, W = latents.shape
    # Reshape to (B, 16, H/2, 2, W/2, 2) to group 2x2 patches
    latents = latents.view(B, C, H // 2, 2, W // 2, 2)
    # Permute to (B, H/2, W/2, 16, 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    # Reshape to (B, (H/2)*(W/2), 64) where 64 = 16 * 2 * 2
    latents = latents.reshape(B, (H // 2) * (W // 2), C * 2 * 2)
    return latents


def unpack_latents(packed_latents: torch.Tensor, latent_h: int, latent_w: int) -> torch.Tensor:
    """
    Unpack latents from (B, (H/2)*(W/2), 64) to (B, 16, H, W).
    
    Args:
        packed_latents: (B, (H/2)*(W/2), 64) packed latents
        latent_h: Original latent height (before packing)
        latent_w: Original latent width (before packing)
    
    Returns:
        latents: (B, 16, H, W) unpacked latents
    """
    B = packed_latents.shape[0]
    C = 16  # Number of channels
    # Reshape from (B, (H/2)*(W/2), 64) to (B, H/2, W/2, 16, 2, 2)
    packed_latents = packed_latents.view(B, latent_h // 2, latent_w // 2, C, 2, 2)
    # Permute to (B, 16, H/2, 2, W/2, 2)
    packed_latents = packed_latents.permute(0, 3, 1, 4, 2, 5)
    # Reshape to (B, 16, H, W)
    packed_latents = packed_latents.reshape(B, C, latent_h, latent_w)
    return packed_latents


def compute_dino_identity_loss(
    generated_latents: torch.Tensor,
    bboxes: torch.Tensor,
    alpha: torch.Tensor,
    dino_embeddings: torch.Tensor,
    vae: nn.Module,
    dino_model: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute DINO identity loss for FG refs.
    
    Args:
        generated_latents: (batch_size, 16, H, W) predicted latents
        bboxes: (batch_size, max_refs, 4) normalized bboxes
        alpha: (batch_size, max_refs) alpha values
        dino_embeddings: (batch_size, max_refs, 1024) reference DINO embeddings
        vae: VAE decoder
        dino_model: DINO model
        device: torch device
    
    Returns:
        loss: scalar tensor
    """
    batch_size, max_refs = alpha.shape
    
    # Decode latents to images
    with torch.no_grad():
        decoded = vae.decode(generated_latents / vae.config.scaling_factor).sample
        # decoded: (batch_size, 3, H, W) in [-1, 1]
        decoded = (decoded + 1.0) / 2.0  # Scale to [0, 1]
        decoded = decoded.clamp(0, 1)
    
    total_loss = 0.0
    num_fg_refs = 0
    
    # Process each FG ref
    for batch_idx in range(batch_size):
        for ref_idx in range(max_refs):
            alpha_val = alpha[batch_idx, ref_idx].item()
            
            if alpha_val < 0.5:  # Skip BG/MG refs
                continue
            
            bbox = bboxes[batch_idx, ref_idx]
            x1, y1, x2, y2 = bbox.tolist()
            
            # Convert normalized bbox to pixel coordinates
            H, W = decoded.shape[2], decoded.shape[3]
            x1_px = int(x1 * W)
            y1_px = int(y1 * H)
            x2_px = int(x2 * W)
            y2_px = int(y2 * H)
            
            if x2_px <= x1_px or y2_px <= y1_px:
                continue
            
            # Crop FG region
            crop = decoded[batch_idx:batch_idx+1, :, y1_px:y2_px, x1_px:x2_px]
            
            # Resize to 224x224 for DINO
            crop_resized = F.interpolate(crop, size=(224, 224), mode='bilinear', align_corners=False)
            
            # Normalize for DINO
            crop_normalized = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop_resized)
            
            # Extract DINO features
            with torch.no_grad():
                generated_dino = dino_model(crop_normalized)
                generated_dino = F.normalize(generated_dino, p=2, dim=-1)
            
            # Get reference DINO embedding
            ref_dino = dino_embeddings[batch_idx, ref_idx]
            ref_dino = F.normalize(ref_dino, p=2, dim=-1)
            
            # Cosine similarity loss (1 - similarity)
            similarity = (generated_dino * ref_dino).sum()
            loss = 1.0 - similarity
            
            # Weight by alpha
            total_loss += loss * alpha_val
            num_fg_refs += 1
    
    if num_fg_refs > 0:
        return total_loss / num_fg_refs
    else:
        return torch.tensor(0.0, device=device)


def save_checkpoint(
    accelerator: Accelerator,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    iteration: int,
    config: Dict,
    checkpoint_dir: str,
):
    """Save training checkpoint with atomic writes."""
    checkpoint_path = Path(checkpoint_dir) / f"checkpoint_{iteration}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save model state
        unwrapped_model = accelerator.unwrap_model(model)
        model_path = checkpoint_path / "model.pt"
        torch.save(unwrapped_model.state_dict(), model_path, _use_new_zipfile_serialization=True)
        
        # Save optimizer and scheduler
        optimizer_path = checkpoint_path / "optimizer.pt"
        torch.save(optimizer.state_dict(), optimizer_path, _use_new_zipfile_serialization=True)
        
        scheduler_path = checkpoint_path / "scheduler.pt"
        torch.save(scheduler.state_dict(), scheduler_path, _use_new_zipfile_serialization=True)
        
        # Save config
        with open(checkpoint_path / "config.yaml", 'w') as f:
            yaml.dump(config, f)
        
        # Save training state
        training_state = {
            'iteration': iteration,
            'stage': config.get('stage', 1),
        }
        with open(checkpoint_path / "training_state.json", 'w') as f:
            json.dump(training_state, f, indent=2)
        
        accelerator.print(f"✓ Saved checkpoint to {checkpoint_path}")
        
    except Exception as e:
        accelerator.print(f"Warning: Failed to save checkpoint: {e}")
        import traceback
        traceback.print_exc()


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    accelerator: Accelerator,
) -> int:
    """Load checkpoint and return starting iteration."""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        accelerator.print(f"Warning: Checkpoint {checkpoint_path} not found, starting from scratch")
        return 0
    
    try:
        # Load model state with weights_only=False for compatibility
        model_state = torch.load(
            checkpoint_path / "model.pt", 
            map_location='cpu',
            weights_only=False
        )
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(model_state, strict=False)
        accelerator.print(f"✓ Loaded model weights from {checkpoint_path}")
    except Exception as e:
        accelerator.print(f"Warning: Failed to load model weights: {e}")
        accelerator.print(f"Starting from scratch")
        return 0
    
    # Load optimizer and scheduler (optional, may fail if checkpoint is incomplete)
    try:
        if (checkpoint_path / "optimizer.pt").exists():
            optimizer.load_state_dict(torch.load(
                checkpoint_path / "optimizer.pt", 
                map_location='cpu',
                weights_only=False
            ))
            accelerator.print(f"✓ Loaded optimizer state")
    except Exception as e:
        accelerator.print(f"Warning: Failed to load optimizer state: {e}")
    
    try:
        if (checkpoint_path / "scheduler.pt").exists():
            scheduler.load_state_dict(torch.load(
                checkpoint_path / "scheduler.pt", 
                map_location='cpu',
                weights_only=False
            ))
            accelerator.print(f"✓ Loaded scheduler state")
    except Exception as e:
        accelerator.print(f"Warning: Failed to load scheduler state: {e}")
    
    # Load training state
    iteration = 0
    try:
        if (checkpoint_path / "training_state.json").exists():
            with open(checkpoint_path / "training_state.json", 'r') as f:
                training_state = json.load(f)
            iteration = training_state.get('iteration', 0)
            accelerator.print(f"✓ Resuming from iteration {iteration}")
    except Exception as e:
        accelerator.print(f"Warning: Failed to load training state: {e}")
    
    return iteration


def log_to_tensorboard(
    writer: SummaryWriter,
    metrics: Dict,
    iteration: int,
):
    """Log metrics to TensorBoard."""
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            value = value.item()
        writer.add_scalar(key, value, iteration)


def run_flow_matching_denoising(
    model: nn.Module,
    noise: torch.Tensor,
    text_embeddings: torch.Tensor,
    pooled_embeddings: torch.Tensor,
    cache: Dict[str, torch.Tensor],
    num_inference_steps: int,
    latent_h: int,
    latent_w: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Run full denoising loop for flow matching.
    
    Args:
        model: The model to use for inference
        noise: (batch_size, 16, H, W) initial noise
        text_embeddings: (batch_size, num_txt_tokens, 4096) text embeddings
        pooled_embeddings: (batch_size, 768) pooled embeddings
        cache: Preprocessed cache dict with instance_tokens, alpha, etc.
        num_inference_steps: Number of denoising steps
        latent_h: Latent height
        latent_w: Latent width
        device: Device to run on
    
    Returns:
        denoised_latents: (batch_size, 16, H, W) final denoised latents
    """
    batch_size = noise.shape[0]
    
    # Start from pure noise (t=1.0) - ensure bfloat16 for FLUX compatibility
    latents = noise.clone().to(torch.bfloat16)
    
    # Create timestep schedule: from 1.0 to 0.0
    timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.bfloat16)
    dt = 1.0 / num_inference_steps  # Step size
    
    # Create position IDs
    img_ids, txt_ids = create_position_ids(
        latent_h, latent_w,
        text_embeddings.shape[1],
        device
    )
    
    # Ensure text embeddings are in bfloat16
    text_embeddings = text_embeddings.to(torch.bfloat16)
    pooled_embeddings = pooled_embeddings.to(torch.bfloat16)
    
    for i in range(num_inference_steps):
        t = timesteps[i]  # Current timestep
        t_next = timesteps[i + 1]  # Next timestep
        
        # Pack latents for FLUX
        packed_latents = pack_latents(latents)
        
        # Expand timestep to batch dimension
        timestep_batch = t.expand(batch_size)
        
        # Predict velocity
        pred_velocity = model.forward(
            hidden_states=packed_latents,
            encoder_hidden_states=text_embeddings,
            pooled_projections=pooled_embeddings,
            timestep=timestep_batch,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=None,
            instance_tokens=cache['instance_tokens'],
            alpha=cache['alpha'],
            adaln_offsets=cache['adaln_offsets'],
            img_idxs_list_list=cache['img_idxs_list_list'],
        )
        
        # Unpack velocity
        pred_velocity = unpack_latents(pred_velocity, latent_h, latent_w)
        
        # Euler integration step: x_{t-dt} = x_t - dt * v_t
        # For flow matching: v_t points from x_t towards x_0 (clean image)
        dt_step = (t - t_next).to(torch.bfloat16)
        latents = latents - dt_step * pred_velocity
    
    return latents


def run_inference_visualization(
    model: nn.Module,
    val_dataloader: DataLoader,
    vae: nn.Module,
    writer: SummaryWriter,
    iteration: int,
    num_samples: int,
    accelerator: Accelerator,
    num_inference_steps: int = 28,
):
    """Run inference with full denoising loop and log visualizations to TensorBoard."""
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()
    
    try:
        # Get validation samples
        val_batch = next(iter(val_dataloader))
        
        with torch.no_grad():
            # Get ground truth latents
            latents_gt = val_batch['latents'][:num_samples].to(accelerator.device)
            bboxes = val_batch['bboxes'][:num_samples].to(accelerator.device)
            clip_embeddings = val_batch['clip_embeddings'][:num_samples].to(accelerator.device)
            dino_embeddings = val_batch['dino_embeddings'][:num_samples].to(accelerator.device)
            depth_labels = val_batch['depth_labels'][:num_samples]
            
            # Get text embeddings
            text_embeddings = val_batch['text_embeddings'][:num_samples].to(accelerator.device)
            pooled_embeddings = val_batch['pooled_embeddings'][:num_samples].to(accelerator.device)
            
            # Decode ground truth latents to images (ensure bfloat16 for VAE)
            latents_gt_bf16 = latents_gt.to(torch.bfloat16)
            decoded_gt = vae.decode(latents_gt_bf16 / vae.config.scaling_factor).sample
            decoded_gt = (decoded_gt + 1.0) / 2.0
            decoded_gt = decoded_gt.clamp(0, 1).float()
            
            # Log ground truth images
            writer.add_images('inference/ground_truth', decoded_gt, iteration)
            
            # Run preprocessing to get alpha values and cached tensors
            cache = unwrapped_model.preprocess(
                bboxes=bboxes,
                depth_labels=[label for sublist in depth_labels for label in sublist],
                clip_embeddings=clip_embeddings,
                dino_embeddings=dino_embeddings,
                image_width=val_batch['widths'][0],
                image_height=val_batch['heights'][0],
                latent_hw=(128, 128),
            )
            
            # Log alpha distribution for each sample
            for i in range(min(num_samples, cache['alpha'].shape[0])):
                alpha_values = cache['alpha'][i].float().cpu().numpy()
                writer.add_histogram(f'inference/alpha_sample_{i}', alpha_values, iteration)
            
            # Run full denoising loop
            batch_size = latents_gt.shape[0]
            latent_h, latent_w = latents_gt.shape[2], latents_gt.shape[3]
            
            # Start from pure noise (bfloat16 for FLUX compatibility)
            noise = torch.randn_like(latents_gt, dtype=torch.bfloat16)
            
            # Denoise
            latents_pred = run_flow_matching_denoising(
                model=unwrapped_model,
                noise=noise,
                text_embeddings=text_embeddings,
                pooled_embeddings=pooled_embeddings,
                cache=cache,
                num_inference_steps=num_inference_steps,
                latent_h=latent_h,
                latent_w=latent_w,
                device=accelerator.device,
            )
            
            # Decode predicted latents (ensure bfloat16 for VAE)
            latents_pred_bf16 = latents_pred.to(torch.bfloat16)
            decoded_pred = vae.decode(latents_pred_bf16 / vae.config.scaling_factor).sample
            decoded_pred = (decoded_pred + 1.0) / 2.0
            decoded_pred = decoded_pred.clamp(0, 1).float()
            
            # Log predicted images
            writer.add_images('inference/predicted', decoded_pred, iteration)
            
            # Create side-by-side comparison
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from io import BytesIO
            
            for i in range(min(num_samples, decoded_gt.shape[0])):
                fig, axes = plt.subplots(1, 2, figsize=(16, 8))
                
                # Ground truth
                img_gt_np = decoded_gt[i].permute(1, 2, 0).float().cpu().numpy()
                axes[0].imshow(img_gt_np)
                axes[0].set_title('Ground Truth', fontsize=12, weight='bold')
                axes[0].axis('off')
                
                # Predicted
                img_pred_np = decoded_pred[i].permute(1, 2, 0).float().cpu().numpy()
                axes[1].imshow(img_pred_np)
                axes[1].set_title('Predicted (Denoised)', fontsize=12, weight='bold')
                axes[1].axis('off')
                
                # Draw bboxes on both images
                num_refs = val_batch['num_refs'][i] if isinstance(val_batch['num_refs'], list) else val_batch['num_refs'][i].item()
                for j in range(num_refs):
                    bbox = bboxes[i, j].float().cpu().numpy()
                    x1, y1, x2, y2 = bbox
                    
                    H, W = img_gt_np.shape[:2]
                    x1_px, y1_px = int(x1 * W), int(y1 * H)
                    x2_px, y2_px = int(x2 * W), int(y2 * H)
                    
                    alpha_val = cache['alpha'][i, j].float().item()
                    color = 'red' if alpha_val >= 0.5 else 'blue'
                    
                    # Draw on both images
                    for ax in axes:
                        rect = patches.Rectangle(
                            (x1_px, y1_px), x2_px - x1_px, y2_px - y1_px,
                            linewidth=2, edgecolor=color, facecolor='none', alpha=0.8
                        )
                        ax.add_patch(rect)
                        ax.text(x1_px, y1_px - 5, f'α={alpha_val:.2f}', 
                               color=color, fontsize=8, weight='bold',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
                plt.tight_layout()
                
                # Save to buffer and log
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                buf.seek(0)
                
                from PIL import Image as PILImage
                img_pil = PILImage.open(buf)
                img_tensor = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
                
                writer.add_image(f'inference/comparison_sample_{i}', img_tensor, iteration)
                
                plt.close(fig)
                buf.close()
            
            accelerator.print(f"✓ Logged inference visualizations at iteration {iteration} (denoising steps: {num_inference_steps})")
            
    except Exception as e:
        accelerator.print(f"Warning: Inference visualization failed: {e}")
        import traceback
        traceback.print_exc()
    
    unwrapped_model.train()


def main():
    parser = argparse.ArgumentParser(description='MoviePostProduction Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override resume checkpoint if provided via CLI
    if args.resume_from_checkpoint:
        config['resume_from_checkpoint'] = args.resume_from_checkpoint
    
    # Initialize Accelerator
    accelerator = Accelerator(
        mixed_precision=config['training']['mixed_precision'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
    )
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Setup logging
    if accelerator.is_main_process:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(config['logging']['log_dir']) / f"stage{config['stage']}_{timestamp}"
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        accelerator.print(f"TensorBoard logs: {log_dir}")
    else:
        writer = None
    
    # Load datasets
    accelerator.print("Loading datasets...")
    train_dataset = MoviePostProductionDataset(
        output_dir=config['data']['output_dir'],
        max_refs=config['model']['max_refs'],
        split='train',
        train_ratio=config['data']['train_split'],
    )
    
    val_dataset = MoviePostProductionDataset(
        output_dir=config['data']['output_dir'],
        max_refs=config['model']['max_refs'],
        split='val',
        train_ratio=config['data']['train_split'],
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
    )
    
    # Initialize model
    accelerator.print("Initializing model...")
    model = setup_model(config, accelerator)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['learning_rate'],
        betas=config['optimizer']['betas'],
        eps=config['optimizer']['eps'],
        weight_decay=config['training']['weight_decay'],
    )
    
    # Setup scheduler
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['scheduler']['num_warmup_steps'],
        num_training_steps=config['training']['num_iterations'],
    )
    
    # Prepare with Accelerator
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )
    
    # Load checkpoint if resuming
    start_iteration = 0
    if config.get('resume_from_checkpoint'):
        start_iteration = load_checkpoint(
            config['resume_from_checkpoint'],
            model,
            optimizer,
            scheduler,
            accelerator,
        )
    
    # Load VAE for visualization
    if accelerator.is_main_process and config['inference']['enabled']:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(
            config['vae']['model_path'],
            torch_dtype=torch.bfloat16,
        ).to(accelerator.device)
        vae.eval()
    else:
        vae = None
    
    # Training loop
    accelerator.print(f"\nStarting Stage {config['stage']} training...")
    accelerator.print(f"Total iterations: {config['training']['num_iterations']}")
    accelerator.print(f"Batch size: {config['training']['batch_size']}")
    accelerator.print(f"Gradient accumulation steps: {config['training']['gradient_accumulation_steps']}")
    accelerator.print(f"Effective batch size: {config['training']['batch_size'] * config['training']['gradient_accumulation_steps']}")
    
    model.train()
    global_step = start_iteration
    progress_bar = tqdm(range(start_iteration, config['training']['num_iterations']), disable=not accelerator.is_local_main_process)
    
    train_iterator = iter(train_dataloader)
    
    while global_step < config['training']['num_iterations']:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)
        
        with accelerator.accumulate(model):
            # Get batch data and convert to bfloat16
            latents = batch['latents'].to(torch.bfloat16)  # (B, 16, H, W)
            bboxes = batch['bboxes'].to(torch.bfloat16)
            depth_labels = batch['depth_labels']
            clip_embeddings = batch['clip_embeddings'].to(torch.bfloat16)
            dino_embeddings = batch['dino_embeddings'].to(torch.bfloat16)
            
            # Text embeddings - choose which encoder to use and convert to bfloat16
            # Option 1: Use T5 for text, CLIP for pooled (FLUX standard)
            text_embeddings = batch['text_embeddings'].to(torch.bfloat16)  # (B, 512, 4096) - T5
            pooled_embeddings = batch['pooled_embeddings'].to(torch.bfloat16)  # (B, 768) - CLIP
            
            # Option 2: Use T5 for both (uncomment to use)
            # text_embeddings = batch['t5_text_embeddings'].to(torch.bfloat16)  # (B, 512, 4096)
            # pooled_embeddings = batch['t5_pooled_embeddings'].to(torch.bfloat16)  # (B, 4096)
            
            # Option 3: Use CLIP for both (uncomment to use)
            # text_embeddings = batch['clip_text_embeddings'].to(torch.bfloat16)  # (B, 77, 768)
            # pooled_embeddings = batch['clip_pooled_embeddings'].to(torch.bfloat16)  # (B, 768)
            
            batch_size = latents.shape[0]
            latent_h, latent_w = latents.shape[2], latents.shape[3]
            
            # 1. Sample noise and timesteps
            noise = torch.randn_like(latents)
            timesteps = torch.rand(batch_size, device=latents.device)
            
            # 2. Add noise to latents (flow matching)
            noisy_latents = add_noise_to_latents(latents, noise, timesteps)
            
            # 3. Preprocess: compute instance tokens, alpha, layout_kv, adaln_offsets
            unwrapped_model = accelerator.unwrap_model(model)
            cache = unwrapped_model.preprocess(
                bboxes=bboxes,
                depth_labels=[label for sublist in depth_labels for label in sublist],  # Flatten
                clip_embeddings=clip_embeddings,
                dino_embeddings=dino_embeddings,
                image_width=batch['widths'][0],
                image_height=batch['heights'][0],
                latent_hw=(latent_h, latent_w),
            )
            
            # 4. Prepare FLUX inputs
            packed_latents = pack_latents(noisy_latents)
            img_ids, txt_ids = create_position_ids(
                latent_h, latent_w,
                text_embeddings.shape[1],  # num_txt_tokens from cached embeddings
                accelerator.device
            )
            
            # 5. Run full FLUX forward pass with modulation & layout injection
            pred_velocity = unwrapped_model.forward(
                hidden_states=packed_latents,
                encoder_hidden_states=text_embeddings,  # Use cached text embeddings
                pooled_projections=pooled_embeddings,  # Use cached pooled embeddings
                timestep=timesteps,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=None,
                instance_tokens=cache['instance_tokens'],
                alpha=cache['alpha'],
                adaln_offsets=cache['adaln_offsets'],
                img_idxs_list_list=cache['img_idxs_list_list'],
            )
            
            # 6. Unpack predicted velocity
            pred_velocity = unpack_latents(pred_velocity, latent_h, latent_w)
            
            # 7. Compute flow matching loss in float32 for gradient compatibility
            # Convert all inputs to float32 to ensure clean autograd graph
            loss = compute_flow_matching_loss(
                pred_velocity.float(), 
                noise.float(), 
                latents.float(), 
                timesteps.float()
            )
            
            # Backward pass
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        if accelerator.sync_gradients:
            global_step += 1
            progress_bar.update(1)
            
            # Logging
            if global_step % config['logging']['log_every_n_steps'] == 0 and accelerator.is_main_process:
                # Compute additional metrics
                alpha_mean = cache['alpha'].mean().item()
                alpha_fg_ratio = (cache['alpha'] >= 0.5).float().mean().item()
                
                metrics = {
                    'loss/flow_matching': loss.detach().item(),
                    'loss/total': loss.detach().item(),
                    'metrics/alpha_mean': alpha_mean,
                    'metrics/alpha_fg_ratio': alpha_fg_ratio,
                    'metrics/learning_rate': scheduler.get_last_lr()[0],
                    'metrics/timestep_mean': timesteps.mean().item(),
                }
                
                # Log to TensorBoard
                log_to_tensorboard(writer, metrics, global_step)
                
                # Log alpha distribution histogram
                writer.add_histogram('metrics/alpha_distribution', cache['alpha'].detach().cpu(), global_step)
                writer.add_histogram('metrics/timesteps', timesteps.detach().cpu(), global_step)
                
                progress_bar.set_postfix({
                    'loss': loss.item(), 
                    'lr': scheduler.get_last_lr()[0],
                    'alpha_mean': alpha_mean,
                })
            
            # Checkpointing
            if global_step % config['checkpointing']['save_every_n_steps'] == 0 and accelerator.is_main_process:
                save_checkpoint(
                    accelerator,
                    model,
                    optimizer,
                    scheduler,
                    global_step,
                    config,
                    config['checkpointing']['checkpoint_dir'],
                )
            
            # Inference visualization
            if config['inference']['enabled'] and global_step % config['inference']['every_n_steps'] == 0 and accelerator.is_main_process:
                run_inference_visualization(
                    model,
                    val_dataloader,
                    vae,
                    writer,
                    global_step,
                    config['inference']['num_samples'],
                    accelerator,
                    num_inference_steps=config['inference'].get('num_inference_steps', 28),
                )
    
    # Save final checkpoint
    if accelerator.is_main_process:
        save_checkpoint(
            accelerator,
            model,
            optimizer,
            scheduler,
            global_step,
            config,
            config['checkpointing']['checkpoint_dir'],
        )
        writer.close()
    
    accelerator.print(f"\n✓ Training complete! Final iteration: {global_step}")


if __name__ == "__main__":
    main()
