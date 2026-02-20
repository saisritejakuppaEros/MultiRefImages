import math
import torch
import torch.nn as nn


def dense_sample_points(bbox, k=16):
    """
    Generate K² uniformly spaced points inside bbox.
    
    Args:
        bbox: (x1, y1, w, h) in normalized coordinates [0, 1], can be tensor or list/tuple
        k: total number of points (k² points, so k should be perfect square)
    
    Returns:
        points: (k, 2) tensor of (x, y) coordinates
    """
    if isinstance(bbox, torch.Tensor):
        x1, y1, w, h = bbox.tolist() if bbox.numel() == 4 else bbox
    else:
        x1, y1, w, h = bbox
    
    x2 = x1 + w
    y2 = y1 + h
    
    point_per_line = int(math.sqrt(k))
    points = []
    
    step_x = (x2 - x1) / (point_per_line - 1) if point_per_line > 1 else 0
    step_y = (y2 - y1) / (point_per_line - 1) if point_per_line > 1 else 0
    
    for u in range(point_per_line):
        for v in range(point_per_line):
            x = x1 + u * step_x if point_per_line > 1 else x1
            y = y1 + v * step_y if point_per_line > 1 else y1
            points.append([x, y])
    
    return torch.tensor(points, dtype=torch.float32)


def fourier_embed_points(points, embed_dim=64):
    """
    Apply Fourier embedding to 2D points.
    
    Args:
        points: (n_points, 2) tensor of (x, y) coordinates
        embed_dim: embedding dimension per point
    
    Returns:
        embedding: (n_points * embed_dim,) flattened tensor
    """
    n_points = points.shape[0]
    device = points.device
    dtype = points.dtype
    
    n_freqs = embed_dim // 4
    freqs = 100 ** (torch.arange(n_freqs, device=device, dtype=dtype) / n_freqs)
    
    emb = freqs[None, :, None] * points[:, None, :]
    emb = torch.stack([emb.sin(), emb.cos()], dim=-1)
    emb = emb.reshape(n_points, embed_dim)
    
    return emb.flatten()


def encode_spatial_features(bbox, device='cpu'):
    """
    Step 2a: Spatial encoding via DenseSample + Fourier embedding.
    
    Args:
        bbox: (x1, y1, w, h) in normalized coordinates [0, 1]
        device: device to place tensors on
    
    Returns:
        spatial_feat: (1024,) tensor
    """
    points = dense_sample_points(bbox, k=16).to(device)
    spatial_feat = fourier_embed_points(points, embed_dim=64)
    return spatial_feat


def encode_visual_features(ref_image_embedding, alpha, device='cpu', clip_proj=None):
    """
    Step 2b: Visual encoding via CLIP (BG/MG) or DINOv2 (FG).
    
    Args:
        ref_image_embedding: pre-computed embedding (CLIP: any dim or DINO: 1024-dim)
        alpha: scalar in [0, 1], threshold at 0.5
        device: device to place tensors on
        clip_proj: optional pre-initialized CLIP projection layer
    
    Returns:
        visual_feat: (1024,) tensor
    """
    if ref_image_embedding is None:
        if alpha < 0.5:
            clip_feat = torch.randn(768, device=device)
        else:
            dino_feat = torch.randn(1024, device=device)
            return dino_feat
    else:
        if alpha < 0.5:
            clip_feat = ref_image_embedding
            clip_dim = clip_feat.shape[0]
            
            if clip_proj is None:
                clip_proj = nn.Linear(clip_dim, 1024).to(device)
                nn.init.xavier_uniform_(clip_proj.weight)
                nn.init.zeros_(clip_proj.bias)
            
            visual_feat = clip_proj(clip_feat)
            return visual_feat
        else:
            dino_feat = ref_image_embedding if ref_image_embedding.shape[0] == 1024 else torch.randn(1024, device=device)
            return dino_feat


class InstanceFusionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2048, 3072)
        self.ln = nn.LayerNorm(3072)
        self.silu = nn.SiLU()
        self.linear2 = nn.Linear(3072, 3072)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        
        nn.init.ones_(self.ln.weight)
        nn.init.zeros_(self.ln.bias)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.ln(x)
        x = self.silu(x)
        x = self.linear2(x)
        return x


def fuse_instance_features(spatial_feat, visual_feat, fusion_mlp=None):
    """
    Step 2c: Fuse spatial + visual features via MLP.
    
    Args:
        spatial_feat: (1024,) tensor
        visual_feat: (1024,) tensor
        fusion_mlp: InstanceFusionMLP instance (optional, creates new if None)
    
    Returns:
        instance_token: (3072,) tensor
    """
    fused = torch.cat([spatial_feat, visual_feat], dim=0)
    if fusion_mlp is None:
        fusion_mlp = InstanceFusionMLP().to(fused.device)
    instance_token = fusion_mlp(fused)
    return instance_token


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    bbox = torch.tensor([0.1, 0.2, 0.3, 0.4])
    alpha_bg = 0.3
    alpha_fg = 0.7
    
    print("Testing Step 2a: Spatial encoding")
    spatial_feat = encode_spatial_features(bbox, device=device)
    print(f"  Spatial feature shape: {spatial_feat.shape}")
    assert spatial_feat.shape == (1024,), f"Expected (1024,), got {spatial_feat.shape}"
    print("  ✓ Step 2a passed\n")
    
    print("Testing Step 2b: Visual encoding (BG with CLIP)")
    visual_feat_bg = encode_visual_features(None, alpha_bg, device=device)
    print(f"  Visual feature shape (BG): {visual_feat_bg.shape}")
    assert visual_feat_bg.shape == (1024,), f"Expected (1024,), got {visual_feat_bg.shape}"
    
    print("Testing Step 2b: Visual encoding (FG with DINO)")
    visual_feat_fg = encode_visual_features(None, alpha_fg, device=device)
    print(f"  Visual feature shape (FG): {visual_feat_fg.shape}")
    assert visual_feat_fg.shape == (1024,), f"Expected (1024,), got {visual_feat_fg.shape}"
    print("  ✓ Step 2b passed\n")
    
    print("Testing Step 2c: Fusion (BG case)")
    instance_token_bg = fuse_instance_features(spatial_feat, visual_feat_bg)
    print(f"  Instance token shape (BG): {instance_token_bg.shape}")
    assert instance_token_bg.shape == (3072,), f"Expected (3072,), got {instance_token_bg.shape}"
    
    print("Testing Step 2c: Fusion (FG case)")
    instance_token_fg = fuse_instance_features(spatial_feat, visual_feat_fg)
    print(f"  Instance token shape (FG): {instance_token_fg.shape}")
    assert instance_token_fg.shape == (3072,), f"Expected (3072,), got {instance_token_fg.shape}"
    print("  ✓ Step 2c passed\n")
    
    print("All tests passed! ✓")
