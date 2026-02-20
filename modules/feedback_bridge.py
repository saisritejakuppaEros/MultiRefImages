import torch
import torch.nn as nn
from typing import Optional, List


class FeedbackBridge(nn.Module):
    """
    Feedback Bridge: Connects layout output at block N to modulation input at block N+1.
    
    After Assemble-Attn updates image tokens at block N, this module:
    1. Mean-pools the updated image tokens inside each bbox
    2. Projects via Linear(3072 → 3072) with zero initialization
    3. Returns feedback offset to be added to modulation at block N+1
    4. Only active for FG refs (alpha >= 0.5)
    
    Architecture:
        - Per-block projections: 19 × Linear(3072 → 3072)
        - Zero initialization ensures no effect at training start
    """
    
    def __init__(self, dim: int = 3072, num_blocks: int = 19):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        
        self.projections = nn.ModuleList([
            nn.Linear(dim, dim, bias=True) for _ in range(num_blocks)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        for proj in self.projections:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
    
    def forward(
        self,
        layout_summary: torch.Tensor,
        alpha: torch.Tensor,
        block_idx: int,
    ) -> torch.Tensor:
        """
        Compute feedback offset for next block.
        
        Args:
            layout_summary: (batch_size, num_refs, 3072) - mean-pooled bbox features from layout
            alpha: (batch_size, num_refs) - alpha values per ref
            block_idx: current block index (0-18)
        
        Returns:
            feedback: (batch_size, num_refs, 3072) - feedback offset for next block
                     Zero for BG/MG refs (alpha < 0.5)
        """
        if block_idx >= self.num_blocks:
            return torch.zeros_like(layout_summary)
        
        fg_mask = (alpha >= 0.5).unsqueeze(-1)
        
        feedback = self.projections[block_idx](layout_summary)
        feedback = feedback * fg_mask
        
        return feedback
    
    def compute_layout_summary(
        self,
        image_tokens: torch.Tensor,
        img_idxs_list_list: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Mean-pool image tokens inside each bbox to create layout summary.
        
        Args:
            image_tokens: (batch_size, num_img_tokens, 3072) - updated image tokens
            img_idxs_list_list: List[List[Tensor]] - image token indices per batch/ref
        
        Returns:
            layout_summary: (batch_size, num_refs, 3072) - mean-pooled features per bbox
        """
        batch_size = image_tokens.shape[0]
        num_refs = len(img_idxs_list_list[0]) if img_idxs_list_list else 0
        
        layout_summary = torch.zeros(batch_size, num_refs, self.dim, device=image_tokens.device, dtype=image_tokens.dtype)
        
        for batch_idx in range(batch_size):
            for ref_idx in range(num_refs):
                img_idxs = img_idxs_list_list[batch_idx][ref_idx]
                if len(img_idxs) > 0:
                    bbox_tokens = image_tokens[batch_idx, img_idxs, :]
                    layout_summary[batch_idx, ref_idx] = bbox_tokens.mean(dim=0)
        
        return layout_summary


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    batch_size = 2
    num_refs = 3
    num_img_tokens = 1024
    dim = 3072
    num_blocks = 19
    
    print("=" * 60)
    print("Testing FeedbackBridge")
    print("=" * 60)
    
    model = FeedbackBridge(dim=dim, num_blocks=num_blocks).to(device)
    
    layout_summary = torch.randn(batch_size, num_refs, dim, device=device)
    alpha = torch.tensor([[0.3, 0.7, 0.9], [0.2, 0.8, 0.4]], device=device)
    
    print(f"\nInput shapes:")
    print(f"  layout_summary: {layout_summary.shape}")
    print(f"  alpha: {alpha.shape}")
    
    print(f"\n--- Test 1: Forward pass at block 0 ---")
    with torch.no_grad():
        feedback = model(layout_summary, alpha, block_idx=0)
    
    print(f"  Output shape: {feedback.shape}")
    assert feedback.shape == layout_summary.shape, f"Shape mismatch"
    print(f"  ✓ Shape check passed")
    
    print(f"\n--- Test 2: Zero initialization check ---")
    assert torch.allclose(feedback, torch.zeros_like(feedback), atol=1e-6), \
        "Zero initialization should produce zero feedback"
    print(f"  ✓ Zero initialization confirmed")
    print(f"  Mean feedback magnitude: {feedback.abs().mean().item():.2e}")
    
    print(f"\n--- Test 3: FG masking check ---")
    model_nonzero = FeedbackBridge(dim=dim, num_blocks=num_blocks).to(device)
    for proj in model_nonzero.projections:
        nn.init.xavier_uniform_(proj.weight)
    
    with torch.no_grad():
        feedback_nonzero = model_nonzero(layout_summary, alpha, block_idx=0)
    
    ref_0_feedback = feedback_nonzero[0, 0, :]
    ref_1_feedback = feedback_nonzero[0, 1, :]
    ref_2_feedback = feedback_nonzero[0, 2, :]
    
    assert torch.allclose(ref_0_feedback, torch.zeros_like(ref_0_feedback), atol=1e-6), \
        "BG ref (alpha=0.3) should have zero feedback"
    assert not torch.allclose(ref_1_feedback, torch.zeros_like(ref_1_feedback), atol=1e-6), \
        "FG ref (alpha=0.7) should have non-zero feedback"
    assert not torch.allclose(ref_2_feedback, torch.zeros_like(ref_2_feedback), atol=1e-6), \
        "FG ref (alpha=0.9) should have non-zero feedback"
    print(f"  ✓ FG masking works correctly")
    print(f"    BG ref (α=0.3) feedback mean: {ref_0_feedback.abs().mean().item():.2e}")
    print(f"    FG ref (α=0.7) feedback mean: {ref_1_feedback.abs().mean().item():.2e}")
    print(f"    FG ref (α=0.9) feedback mean: {ref_2_feedback.abs().mean().item():.2e}")
    
    print(f"\n--- Test 4: compute_layout_summary helper ---")
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
    
    layout_summary_computed = model.compute_layout_summary(image_tokens, img_idxs_list_list)
    print(f"  Image tokens shape: {image_tokens.shape}")
    print(f"  Layout summary shape: {layout_summary_computed.shape}")
    assert layout_summary_computed.shape == (batch_size, num_refs, dim), \
        f"Expected {(batch_size, num_refs, dim)}, got {layout_summary_computed.shape}"
    print(f"  ✓ Layout summary computation passed")
    
    expected_mean_ref0 = image_tokens[0, [10, 11, 12, 20, 21, 22], :].mean(dim=0)
    assert torch.allclose(layout_summary_computed[0, 0], expected_mean_ref0, atol=1e-5), \
        "Layout summary should match mean of bbox tokens"
    print(f"  ✓ Mean pooling correctness verified")
    
    print(f"\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
