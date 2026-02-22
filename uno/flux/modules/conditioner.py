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

import torch
from torch import Tensor, nn
from transformers import (CLIPModel, CLIPProcessor, CLIPTextModel, CLIPTokenizer,
                          T5EncoderModel, T5Tokenizer)


class HFEmbedder(nn.Module):
    def __init__(self, version: str, max_length: int, cache_dir: str | None = None, **hf_kwargs):
        super().__init__()
        self.is_clip = "clip" in version.lower()
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"
        hf_kwargs = dict(hf_kwargs)
        if cache_dir is not None:
            hf_kwargs["cache_dir"] = cache_dir

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length, cache_dir=cache_dir)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(version, **hf_kwargs)
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(version, max_length=max_length, cache_dir=cache_dir)
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(version, **hf_kwargs)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]


class CLIPImageEmbedder(nn.Module):
    """
    Wraps openai/clip-vit-large-patch14 to produce pooled image embeddings.
    Output: [N, 768] with L2 normalization.
    Used for Path B (layout CLIP objects).
    """
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device="cpu", cache_dir: str | None = None):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.device = device
        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def encode_image(self, images: list) -> Tensor:
        """
        images: list of PIL.Image
        Returns: [len(images), 768]
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu()

    def forward(self, images: list) -> Tensor:
        return self.encode_image(images)
