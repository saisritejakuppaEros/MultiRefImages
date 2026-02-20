# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Any, Optional, Type

try:
    from peft.tuners.tuners_utils import BaseTunerLayer
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    BaseTunerLayer = object


class enable_lora:
    def __init__(self, lora_modules: List[BaseTunerLayer], dit_activated: bool, cond_activated: bool=False, latent_sblora_weight: float=None, condition_sblora_weight: float=None) -> None:
        self.dit_activated = dit_activated
        self.cond_activated = cond_activated
        self.latent_sblora_weight = latent_sblora_weight
        self.condition_sblora_weight = condition_sblora_weight
        
        if not PEFT_AVAILABLE:
            self.lora_modules = []
            self.scales = []
            return
        
        self.lora_modules: List[BaseTunerLayer] = [
            each for each in lora_modules if isinstance(each, BaseTunerLayer)
        ]

        self.scales = [
            {
                active_adapter: lora_module.scaling[active_adapter] if active_adapter in lora_module.scaling else 1
                for active_adapter in lora_module.active_adapters
            } for lora_module in self.lora_modules
        ]


    def __enter__(self) -> None:
        if not PEFT_AVAILABLE or not self.lora_modules:
            return
        for i, lora_module in enumerate(self.lora_modules):
            if not isinstance(lora_module, BaseTunerLayer):
                continue
            for active_adapter in lora_module.active_adapters:
                if active_adapter == "default":
                    if self.dit_activated:
                        lora_module.scaling[active_adapter] = self.scales[0]["default"] if self.latent_sblora_weight is None else self.latent_sblora_weight
                    else:
                        lora_module.scaling[active_adapter] = 0
                else:
                    assert active_adapter == "condition"
                    if self.cond_activated:
                        lora_module.scaling[active_adapter] = self.scales[0]["condition"] if self.condition_sblora_weight is None else self.condition_sblora_weight
                    else:
                        lora_module.scaling[active_adapter] = 0

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        if not PEFT_AVAILABLE or not self.lora_modules:
            return
        for i, lora_module in enumerate(self.lora_modules):
            if not isinstance(lora_module, BaseTunerLayer):
                continue
            for active_adapter in lora_module.active_adapters:
                lora_module.scaling[active_adapter] = self.scales[i][active_adapter]
        
class set_lora_scale:
    def __init__(self, lora_modules: List[BaseTunerLayer], scale: float) -> None:
        if not PEFT_AVAILABLE:
            self.lora_modules = []
            self.scales = []
            self.scale = scale
            return
        
        self.lora_modules: List[BaseTunerLayer] = [
            each for each in lora_modules if isinstance(each, BaseTunerLayer)
        ]
        self.scales = [
            {
                active_adapter: lora_module.scaling[active_adapter]
                for active_adapter in lora_module.active_adapters
            }
            for lora_module in self.lora_modules
        ]
        self.scale = scale

    def __enter__(self) -> None:
        if not PEFT_AVAILABLE or not self.lora_modules:
            return
        for lora_module in self.lora_modules:
            if not isinstance(lora_module, BaseTunerLayer):
                continue
            lora_module.scale_layer(self.scale)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        if not PEFT_AVAILABLE or not self.lora_modules:
            return
        for i, lora_module in enumerate(self.lora_modules):
            if not isinstance(lora_module, BaseTunerLayer):
                continue
            for active_adapter in lora_module.active_adapters:
                lora_module.scaling[active_adapter] = self.scales[i][active_adapter]
