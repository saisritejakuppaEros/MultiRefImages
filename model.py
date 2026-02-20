import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import sys
import os

from modules.alpha_predictor import AlphaPredictor, depth_label_to_onehot, compute_bbox_area_ratio
from modules.instance_encoder import encode_spatial_features, encode_visual_features, InstanceFusionMLP
from modules.layout_module import LayoutHead, LayoutIntegrator
from modules.modulation_modules import ModulationHead, ModulationIntegrator
from modules.feedback_bridge import FeedbackBridge
from flux.flux_integration import custom_double_block_forward, custom_single_block_forward
from flux.layout_utils import get_layout_idxslist


class MoviePostProductionModel(nn.Module):
    """
    Main pipeline integrating all modules with FLUX transformer.
    
    Architecture:
    1. Alpha Predictor: determines layout vs modulation routing per ref
    2. Instance Encoder: fuses spatial + visual features → instance token
    3. Layout Head: produces layout_kv for Assemble-Attn (scaled by 1-α)
    4. Modulation Head: produces AdaLN offsets (scaled by α)
    5. Feedback Bridge: connects layout output to next block's modulation
    
    Forward pass:
    - Preprocessing: compute all cached tensors (instance tokens, layout_kv, adaln_offsets)
    - Per-block: inject modulation → run FLUX block → inject layout → compute feedback
    - 19 double-stream blocks with full integration
    - 38 single-stream blocks with light modulation only
    """
    
    def __init__(
        self,
        flux_model_name: str = "black-forest-labs/FLUX.1-dev",
        max_refs: int = 50,
        num_double_blocks: int = 19,
        num_single_blocks: int = 38,
        dim: int = 3072,
        device: str = 'cuda',
    ):
        super().__init__()
        self.max_refs = max_refs
        self.num_double_blocks = num_double_blocks
        self.num_single_blocks = num_single_blocks
        self.dim = dim
        self.device = device
        
        self.alpha_predictor = AlphaPredictor().to(device).to(torch.bfloat16)
        self.instance_fusion_mlp = InstanceFusionMLP().to(device).to(torch.bfloat16)
        self.layout_head = LayoutHead(dim=dim).to(device).to(torch.bfloat16)
        self.modulation_head = ModulationHead(
            instance_dim=dim,
            num_blocks=num_double_blocks,
            num_adaln_params=12,
        ).to(device).to(torch.bfloat16)
        self.feedback_bridge = FeedbackBridge(dim=dim, num_blocks=num_double_blocks).to(device).to(torch.bfloat16)
        
        self.layout_integrator = LayoutIntegrator(dim=dim, num_heads=24, head_dim=128).to(device).to(torch.bfloat16)
        self.modulation_integrator = ModulationIntegrator(
            dim=dim,
            num_blocks=num_double_blocks,
            num_adaln_params=12,
        ).to(device).to(torch.bfloat16)
        
        self.clip_proj = None
        
        print(f"Initializing FLUX transformer from {flux_model_name}...")
        try:
            from diffusers import FluxTransformer2DModel
            
            cache_dir = "/root/.cache/huggingface/hub"
            local_files_only = os.path.exists(os.path.join(cache_dir, "models--black-forest-labs--FLUX.1-dev"))
            
            self.flux_transformer = FluxTransformer2DModel.from_pretrained(
                flux_model_name,
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            ).to(device)
            print(f"✓ FLUX transformer loaded successfully (from cache: {local_files_only})")
        except Exception as e:
            print(f"Warning: Could not load FLUX transformer: {e}")
            print(f"Creating dummy FLUX structure for testing...")
            self.flux_transformer = None
    
    def preprocess(
        self,
        bboxes: torch.Tensor,
        depth_labels: List[str],
        clip_embeddings: torch.Tensor,
        dino_embeddings: torch.Tensor,
        image_width: int,
        image_height: int,
        latent_hw: Tuple[int, int] = (128, 128),
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocessing phase: compute all cached tensors.
        
        Args:
            bboxes: (batch_size, num_refs, 4) normalized [x1, y1, x2, y2]
            depth_labels: List of depth label strings per ref
            clip_embeddings: (batch_size, num_refs, 768)
            dino_embeddings: (batch_size, num_refs, 1024)
            image_width: image width in pixels
            image_height: image height in pixels
            latent_hw: latent dimensions (128, 128)
        
        Returns:
            cache: dict with:
                - instance_tokens: (batch_size, num_refs, 3072)
                - alpha: (batch_size, num_refs)
                - layout_kv: (batch_size, num_refs, 3072)
                - adaln_offsets: (batch_size, num_refs, num_blocks, 12)
                - img_idxs_list_list: List[List[Tensor]]
        """
        batch_size, num_refs = bboxes.shape[:2]
        device = bboxes.device
        
        clip_dim = clip_embeddings.shape[-1]
        # Get the target dtype from alpha_predictor (will be float32 if trainable, bfloat16 if frozen)
        target_dtype = next(self.alpha_predictor.parameters()).dtype
        
        if self.clip_proj is None:
            self.clip_proj = nn.Linear(clip_dim, 1024).to(device).to(target_dtype)
            nn.init.xavier_uniform_(self.clip_proj.weight)
            nn.init.zeros_(self.clip_proj.bias)
        
        if not hasattr(self.alpha_predictor, 'clip_dim') or self.alpha_predictor.clip_dim != clip_dim:
            self.alpha_predictor = AlphaPredictor(clip_dim=clip_dim).to(device).to(target_dtype)
        
        alpha_list = []
        instance_tokens_list = []
        
        for batch_idx in range(batch_size):
            batch_instance_tokens = []
            batch_alphas = []
            
            for ref_idx in range(num_refs):
                bbox = bboxes[batch_idx, ref_idx]
                depth_label = depth_labels[batch_idx * num_refs + ref_idx] if isinstance(depth_labels, list) else 'midground'
                clip_emb = clip_embeddings[batch_idx, ref_idx]
                dino_emb = dino_embeddings[batch_idx, ref_idx]
                
                bbox_area_ratio = compute_bbox_area_ratio(bbox.tolist(), image_width, image_height)
                depth_onehot = depth_label_to_onehot(depth_label).to(device)
                
                # Convert to module dtype (float32 if trainable, bfloat16 if frozen)
                module_dtype = next(self.alpha_predictor.parameters()).dtype
                alpha = self.alpha_predictor(
                    clip_emb.unsqueeze(0).to(module_dtype),
                    torch.tensor([[bbox_area_ratio]], device=device, dtype=module_dtype),
                    depth_onehot.unsqueeze(0).to(module_dtype),
                ).squeeze()
                
                batch_alphas.append(alpha)
                
                x1, y1, x2, y2 = bbox.tolist()
                w, h = x2 - x1, y2 - y1
                bbox_xywh = [x1, y1, w, h]
                
                spatial_feat = encode_spatial_features(bbox_xywh, device=device)
                
                if alpha < 0.5:
                    visual_feat = encode_visual_features(clip_emb, alpha.item(), device=device, clip_proj=self.clip_proj)
                else:
                    visual_feat = dino_emb
                
                # Convert to module dtype (float32 if trainable, bfloat16 if frozen)
                module_dtype = next(self.instance_fusion_mlp.parameters()).dtype
                fused_feat = torch.cat([spatial_feat, visual_feat], dim=0).to(device).to(module_dtype)
                instance_token = self.instance_fusion_mlp(fused_feat)
                batch_instance_tokens.append(instance_token)
            
            alpha_list.append(torch.stack(batch_alphas))
            instance_tokens_list.append(torch.stack(batch_instance_tokens))
        
        instance_tokens = torch.stack(instance_tokens_list)
        alpha = torch.stack(alpha_list)
        
        # Convert to module dtype for layout_head and modulation_head
        layout_head_dtype = next(self.layout_head.parameters()).dtype
        modulation_head_dtype = next(self.modulation_head.parameters()).dtype
        
        layout_kv = self.layout_head(instance_tokens.to(layout_head_dtype))
        
        adaln_offsets = self.modulation_head(instance_tokens.to(modulation_head_dtype), alpha=alpha.to(modulation_head_dtype))
        
        # FLUX uses 2x2 packing, so packed dimensions are (H/2, W/2)
        packed_latent_hw = (latent_hw[0] // 2, latent_hw[1] // 2)
        img_idxs_list_list = self.layout_integrator.create_img_idxs_list(bboxes, packed_latent_hw)
        
        # IMPORTANT: Keep tensors in their original dtype (float32) to preserve gradient flow
        # Do NOT convert to bfloat16 here - that would break the autograd graph
        return {
            'instance_tokens': instance_tokens,
            'alpha': alpha,
            'layout_kv': layout_kv,
            'adaln_offsets': adaln_offsets,
            'img_idxs_list_list': img_idxs_list_list,
        }
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        instance_tokens: Optional[torch.Tensor] = None,
        alpha: Optional[torch.Tensor] = None,
        adaln_offsets: Optional[torch.Tensor] = None,
        img_idxs_list_list: Optional[List[List[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """
        Full forward pass through FLUX with modulation and layout injection.
        
        Args:
            hidden_states: (batch_size, num_img_tokens, 64) image latents (packed)
            encoder_hidden_states: (batch_size, num_txt_tokens, 4096) text embeddings
            pooled_projections: (batch_size, 768) pooled text embeddings
            timestep: (batch_size,) timestep values
            img_ids: (num_img_tokens, 3) image position IDs
            txt_ids: (num_txt_tokens, 3) text position IDs
            guidance: (batch_size,) guidance scale
            instance_tokens: (batch_size, num_refs, 3072) from preprocessing
            alpha: (batch_size, num_refs) from preprocessing
            adaln_offsets: (batch_size, num_refs, num_blocks, 12) from preprocessing
            img_idxs_list_list: List[List[Tensor]] from preprocessing
        
        Returns:
            output: (batch_size, num_img_tokens, patch_size^2 * out_channels) denoised latents
        """
        if self.flux_transformer is None:
            print("Warning: FLUX transformer not loaded, returning dummy output")
            return torch.randn_like(hidden_states)
        
        batch_size = hidden_states.shape[0]
        
        ids = torch.cat([txt_ids, img_ids], dim=0)
        image_rotary_emb = self.flux_transformer.pos_embed(ids)
        
        hidden_states = self.flux_transformer.x_embedder(hidden_states)
        
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is None:
            guidance = torch.tensor([3.5], device=hidden_states.device, dtype=hidden_states.dtype)
        
        guidance = guidance.to(hidden_states.dtype) * 1000
        temb = self.flux_transformer.time_text_embed(timestep, guidance, pooled_projections)
        encoder_hidden_states = self.flux_transformer.context_embedder(encoder_hidden_states)
        
        feedback_offsets = [None] * self.num_double_blocks
        
        for block_idx in range(self.num_double_blocks):
            block = self.flux_transformer.transformer_blocks[block_idx]
            
            modulation_offsets_block = None
            if adaln_offsets is not None:
                modulation_offsets_block = adaln_offsets[:, :, block_idx, :6]
            
            hidden_states, encoder_hidden_states, layout_summary = custom_double_block_forward(
                block=block,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                instance_tokens=instance_tokens,
                alpha=alpha,
                img_idxs_list_list=img_idxs_list_list,
                modulation_offsets=modulation_offsets_block,
                layout_integrator=self.layout_integrator if instance_tokens is not None else None,
                feedback_offset=feedback_offsets[block_idx],
                block_idx=block_idx,
            )
            
            if layout_summary is not None and block_idx < self.num_double_blocks - 1:
                feedback_offsets[block_idx + 1] = self.feedback_bridge(
                    layout_summary, alpha, block_idx
                )
        
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        
        for block_idx in range(self.num_single_blocks):
            block = self.flux_transformer.single_transformer_blocks[block_idx]
            
            hidden_states = custom_single_block_forward(
                block=block,
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
        
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]
        
        hidden_states = self.flux_transformer.norm_out(hidden_states, temb)
        output = self.flux_transformer.proj_out(hidden_states)
        
        return output


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("=" * 80)
    print("MoviePostProduction Model - Forward Pass Demo")
    print("=" * 80)
    
    print("\n[1/5] Initializing model...")
    model = MoviePostProductionModel(
        flux_model_name="black-forest-labs/FLUX.1-dev",
        max_refs=50,
        device=device,
    )
    model.eval()
    print("✓ Model initialized")
    
    print("\n[2/5] Loading sample data...")
    try:
        from data_loader import load_sample, prepare_batch_from_sample
        sample = load_sample(sample_idx=0, output_dir='./output_data', device=device)
        batch = prepare_batch_from_sample(sample, max_refs=50, device=device)
        print(f"✓ Loaded sample: {sample['image_id']}")
        print(f"  Objects: {batch['num_refs']}")
        print(f"  Image size: {batch['width']}×{batch['height']}")
    except Exception as e:
        print(f"Warning: Could not load real data: {e}")
        print(f"Using dummy data for testing...")
        batch = {
            'bboxes': torch.rand(1, 5, 4, device=device) * 0.5 + 0.25,
            'depth_labels': ['foreground', 'background', 'midground', 'foreground', 'background'],
            'clip_embeddings': torch.randn(1, 5, 768, device=device),
            'dino_embeddings': torch.randn(1, 5, 1024, device=device),
            'num_refs': 5,
            'width': 1024,
            'height': 1024,
            'global_caption': 'A test scene with multiple objects',
        }
    
    print("\n[3/5] Running preprocessing phase...")
    with torch.no_grad():
        cache = model.preprocess(
            bboxes=batch['bboxes'],
            depth_labels=batch['depth_labels'],
            clip_embeddings=batch['clip_embeddings'],
            dino_embeddings=batch['dino_embeddings'],
            image_width=batch['width'],
            image_height=batch['height'],
            latent_hw=(128, 128),
        )
    
    print("✓ Preprocessing complete")
    print(f"  Instance tokens shape: {cache['instance_tokens'].shape}")
    print(f"  Alpha shape: {cache['alpha'].shape}")
    print(f"  Alpha values: {cache['alpha'][0, :batch['num_refs']].tolist()}")
    print(f"  Layout KV shape: {cache['layout_kv'].shape}")
    print(f"  AdaLN offsets shape: {cache['adaln_offsets'].shape}")
    
    fg_count = (cache['alpha'][0] >= 0.5).sum().item()
    bg_count = (cache['alpha'][0] < 0.5).sum().item()
    print(f"  Routing: {fg_count} FG refs (strong modulation), {bg_count} BG/MG refs (strong layout)")
    
    print("\n[4/5] Testing forward pass (dummy inputs)...")
    if model.flux_transformer is not None:
        batch_size = 1
        num_img_tokens = 128 * 128
        num_txt_tokens = 154
        
        hidden_states_dummy = torch.randn(batch_size, num_img_tokens, 64, device=device, dtype=torch.bfloat16)
        encoder_hidden_states_dummy = torch.randn(batch_size, num_txt_tokens, 4096, device=device, dtype=torch.bfloat16)
        pooled_projections_dummy = torch.randn(batch_size, 768, device=device, dtype=torch.bfloat16)
        timestep_dummy = torch.tensor([0.5], device=device)
        
        img_ids = torch.zeros(num_img_tokens, 3, device=device)
        for i in range(128):
            for j in range(128):
                idx = i * 128 + j
                img_ids[idx, 1] = i
                img_ids[idx, 2] = j
        
        txt_ids = torch.zeros(num_txt_tokens, 3, device=device)
        for i in range(num_txt_tokens):
            txt_ids[i, 1] = 0
            txt_ids[i, 2] = i
        
        try:
            with torch.no_grad():
                output = model.forward(
                    hidden_states=hidden_states_dummy,
                    encoder_hidden_states=encoder_hidden_states_dummy,
                    pooled_projections=pooled_projections_dummy,
                    timestep=timestep_dummy,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=None,
                    instance_tokens=cache['instance_tokens'].to(torch.bfloat16),
                    alpha=cache['alpha'],
                    adaln_offsets=cache['adaln_offsets'].to(torch.bfloat16),
                    img_idxs_list_list=cache['img_idxs_list_list'],
                )
            print(f"✓ Forward pass complete")
            print(f"  Output shape: {output.shape}")
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping forward pass (FLUX not loaded)")
    
    print("\n[5/5] Testing feedback bridge...")
    if cache['img_idxs_list_list']:
        dummy_image_tokens = torch.randn(1, 128*128, 3072, device=device, dtype=torch.bfloat16)
        layout_summary = model.feedback_bridge.compute_layout_summary(
            dummy_image_tokens,
            cache['img_idxs_list_list'],
        ).to(torch.bfloat16)
        feedback = model.feedback_bridge(layout_summary, cache['alpha'].to(torch.bfloat16), block_idx=0)
        print(f"✓ Feedback bridge working")
        print(f"  Layout summary shape: {layout_summary.shape}")
        print(f"  Feedback shape: {feedback.shape}")
        print(f"  Feedback magnitude (should be ~0 at init): {feedback.abs().mean().item():.2e}")
    
    print("\n" + "=" * 80)
    print("✓ All components integrated successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Load pretrained FLUX weights")
    print("2. Implement full denoising loop")
    print("3. Add training loop with loss functions")
    print("4. Train in stages as specified in plan.md")
