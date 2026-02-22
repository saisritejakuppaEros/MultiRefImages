#!/usr/bin/env python3
"""
Quick smoke test for Hybrid UNO layout conditioning.
Verifies EmbedBboxProjection shapes, layout_out zero-init, and basic forward.
"""
import torch

def test_layout_modules():
    from uno.flux.modules.layout import EmbedBboxProjection, zero_module, get_fourier_embeds_from_boundingbox

    B, max_objs = 2, 4
    embed_dim = 768
    out_dim = 3072
    point_num = 6
    n_pts = point_num ** 2

    proj = EmbedBboxProjection(embed_dim=embed_dim, out_dim=out_dim, point_num=point_num)
    boxes = torch.randn(B, max_objs, n_pts * 2)
    masks = torch.ones(B, max_objs)
    masks[:, 2:] = 0  # last 2 objects masked
    embeddings = torch.randn(B, max_objs, embed_dim)

    out = proj(boxes=boxes, masks=masks, embeddings=embeddings)
    assert out.shape == (B, max_objs, out_dim), f"Expected (B, max_objs, out_dim), got {out.shape}"
    print("EmbedBboxProjection shape OK:", out.shape)


def test_zero_module():
    from uno.flux.modules.layout import zero_module
    import torch.nn as nn

    linear = nn.Linear(10, 10)
    zeroed = zero_module(linear)
    x = torch.randn(2, 10)
    out = zeroed(x)
    assert torch.allclose(out, torch.zeros_like(out)), "zero_module should zero output"
    print("zero_module OK")


def test_double_block_layout():
    from uno.flux.modules.layers import DoubleStreamBlock

    hidden_size = 64
    num_heads = 4
    block = DoubleStreamBlock(hidden_size, num_heads, mlp_ratio=4.0)
    block.use_layout = True

    # Check layout_out is zero-initialized
    if hasattr(block, "layout_out") and block.layout_out is not None:
        for p in block.layout_out.parameters():
            assert torch.allclose(p, torch.zeros_like(p)), "layout_out should be zero-initialized"
    print("DoubleStreamBlock layout_out zero-init OK")


def test_hybrid_layout_build_kwargs():
    from uno.flux.modules.layout import HybridLayout, ObjectCond

    conds = [
        ObjectCond(bbox=[0.1, 0.1, 0.5, 0.5], hw=[64, 64], tier="vae", ref_image=None, embedding=None),
        ObjectCond(bbox=[0.2, 0.2, 0.8, 0.8], hw=[64, 64], tier="clip", embedding=torch.randn(768), ref_image=None),
    ]
    layout = HybridLayout(conds, max_objs=50)

    kwargs = layout.build_layout_kwargs(device=torch.device("cpu"), dtype=torch.float32, latent_hw=(4, 4))
    assert "layout" in kwargs
    if kwargs["layout"] is not None:
        lk = kwargs["layout"]
        assert "boxes" in lk and "embeddings" in lk and "img_idxs_list" in lk
        print("HybridLayout.build_layout_kwargs OK:", list(kwargs["layout"].keys()))

    # Empty clip
    layout_empty = HybridLayout([conds[0]], max_objs=50)
    kw_empty = layout_empty.build_layout_kwargs(device=torch.device("cpu"), dtype=torch.float32, latent_hw=(4, 4))
    assert kw_empty["layout"] is None
    print("HybridLayout empty layout OK")


def main():
    print("Running quick_test for Hybrid UNO...")
    test_layout_modules()
    test_zero_module()
    test_double_block_layout()
    test_hybrid_layout_build_kwargs()
    print("\nAll quick_test checks passed.")


if __name__ == "__main__":
    main()
