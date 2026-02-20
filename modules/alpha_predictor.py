import torch
import torch.nn as nn
import numpy as np


class AlphaPredictor(nn.Module):
    def __init__(self, clip_dim: int = 768):
        super().__init__()
        self.clip_dim = clip_dim
        input_dim = clip_dim + 1 + 3
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, clip_embedding, bbox_area_ratio, depth_onehot):
        """
        Args:
            clip_embedding: (batch_size, clip_dim) CLIP image CLS embedding
            bbox_area_ratio: (batch_size, 1) normalized bbox area ratio [0, 1]
            depth_onehot: (batch_size, 3) one-hot encoded depth label [FG, MG, BG]
        
        Returns:
            alpha: (batch_size, 1) alpha scalar in (0, 1)
        """
        x = torch.cat([clip_embedding, bbox_area_ratio, depth_onehot], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        alpha = self.sigmoid(x)
        return alpha


def depth_label_to_onehot(depth_label):
    """
    Convert depth label string to one-hot encoding.
    
    Args:
        depth_label: str, one of "foreground", "midground", "background"
    
    Returns:
        onehot: (3,) tensor [FG, MG, BG]
    """
    mapping = {"foreground": 0, "midground": 1, "background": 2}
    idx = mapping.get(depth_label.lower(), 0)
    onehot = torch.zeros(3, dtype=torch.float32)
    onehot[idx] = 1.0
    return onehot


def compute_bbox_area_ratio(bbox, image_width, image_height):
    """
    Compute normalized bbox area ratio.
    
    Args:
        bbox: list of [xmin, ymin, xmax, ymax] in normalized coordinates [0, 1]
        image_width: int, image width in pixels
        image_height: int, image height in pixels
    
    Returns:
        area_ratio: float, normalized area ratio [0, 1]
    """
    if max(bbox) <= 1.0:
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
    else:
        xmin, ymin, xmax, ymax = bbox
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height
    area_ratio = width * height
    return max(0.0, min(1.0, area_ratio))


if __name__ == "__main__":
    batch_size = 4
    
    clip_embedding = torch.randn(batch_size, 768)
    bbox_area_ratio = torch.rand(batch_size, 1)
    depth_onehot = torch.zeros(batch_size, 3)
    depth_onehot[0, 0] = 1.0
    depth_onehot[1, 1] = 1.0
    depth_onehot[2, 2] = 1.0
    depth_onehot[3, 0] = 1.0
    
    model = AlphaPredictor()
    model.eval()
    
    with torch.no_grad():
        alpha = model(clip_embedding, bbox_area_ratio, depth_onehot)
    
    print(f"Input shapes:")
    print(f"  CLIP embedding: {clip_embedding.shape}")
    print(f"  Bbox area ratio: {bbox_area_ratio.shape}")
    print(f"  Depth onehot: {depth_onehot.shape}")
    print(f"\nOutput alpha shape: {alpha.shape}")
    print(f"Alpha values: {alpha.squeeze().tolist()}")
    print(f"\nInitialization check - all alphas should be ~0.5:")
    print(f"Mean alpha: {alpha.mean().item():.4f}")
    print(f"Std alpha: {alpha.std().item():.4f}")
