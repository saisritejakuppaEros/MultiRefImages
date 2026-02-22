# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Copyright (c) 2024 Black Forest Labs and The XLabs-AI Team. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .modules.layers import DoubleStreamBlock, EmbedND, LastLayer, MLPEmbedder, SingleStreamBlock, timestep_embedding
from .modules.layout import EmbedBboxProjection


@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """
    _supports_gradient_checkpointing = True

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)
        self.layout_net = EmbedBboxProjection(
            embed_dim=768,
            out_dim=self.hidden_size,
            fourier_freqs=8,
            point_num=6,
        )
        self.gradient_checkpointing = False

    def enable_layout(
        self,
        double_block_indices: list[int] | None = None,
        single_block_indices: list[int] | None = None,
    ):
        """Enable layout conditioning on specified blocks. None = all blocks."""
        for i, b in enumerate(self.double_blocks):
            b.use_layout = double_block_indices is None or i in double_block_indices
        for i, b in enumerate(self.single_blocks):
            b.use_layout = single_block_indices is None or i in single_block_indices

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    @property
    def attn_processors(self):
        # set recursively
        processors = {}  # type: dict[str, nn.Module]

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
        ref_img: Tensor | None = None,
        ref_img_ids: Tensor | None = None,
        layout_kwargs: dict | None = None,
        layout_scale: float = 1.0,
        grounding_ratio: float = 0.3,
        current_step_ratio: float = 1.0,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)

        # concat ref_img/img
        img_end = img.shape[1]
        if ref_img is not None:
            if isinstance(ref_img, tuple) or isinstance(ref_img, list):
                img = torch.cat([img, self.img_in(torch.cat(ref_img, dim=1))], dim=1)
                img_ids = [ids] + [ref_ids for ref_ids in ref_img_ids]
                ids = torch.cat(img_ids, dim=1)
            else:
                img = torch.cat((img, self.img_in(ref_img)), dim=1)  
                ids = torch.cat((ids, ref_img_ids), dim=1)
        pe = self.pe_embedder(ids)

        layout_hidden_states = None
        img_idxs_list = None
        layout_masks = None
        effective_layout_scale = 0.0
        if layout_kwargs is not None and layout_kwargs.get("layout") is not None and current_step_ratio <= grounding_ratio:
            lk = layout_kwargs["layout"]
            layout_hidden_states = self.layout_net(
                boxes=lk["boxes"],
                masks=lk["masks"],
                embeddings=lk["embeddings"],
            )
            img_idxs_list = [lk["img_idxs_list"]]
            layout_masks = lk["masks"]
            effective_layout_scale = layout_scale

        for index_block, block in enumerate(self.double_blocks):
            if self.training and self.gradient_checkpointing:
                result = torch.utils.checkpoint.checkpoint(
                    block,
                    img,
                    txt,
                    vec,
                    pe,
                    layout_hidden_states=layout_hidden_states,
                    layout_masks=layout_masks,
                    img_idxs_list=img_idxs_list,
                    layout_scale=effective_layout_scale,
                    use_reentrant=False,
                )
                img, txt, layout_hidden_states = result

            else:
                img, txt, layout_hidden_states = block(
                    img=img,
                    txt=txt,
                    vec=vec,
                    pe=pe,
                    layout_hidden_states=layout_hidden_states,
                    layout_masks=layout_masks,
                    img_idxs_list=img_idxs_list,
                    layout_scale=effective_layout_scale,
                )

        img = torch.cat((txt, img), 1)
        txt_len = txt.shape[1]
        for block in self.single_blocks:
            if self.training and self.gradient_checkpointing:
                result = torch.utils.checkpoint.checkpoint(
                    block,
                    img,
                    vec,
                    pe,
                    txt_len=txt_len,
                    layout_hidden_states=layout_hidden_states,
                    layout_masks=layout_masks,
                    img_idxs_list=img_idxs_list,
                    layout_scale=effective_layout_scale,
                    use_reentrant=False,
                )
                img, layout_hidden_states = result
            else:
                img, layout_hidden_states = block(
                    img,
                    vec=vec,
                    pe=pe,
                    txt_len=txt_len,
                    layout_hidden_states=layout_hidden_states,
                    layout_masks=layout_masks,
                    img_idxs_list=img_idxs_list,
                    layout_scale=effective_layout_scale,
                )
        img = img[:, txt.shape[1] :, ...]
        # index img
        img = img[:, :img_end, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img
