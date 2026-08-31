# Copyright 2026 SGLang Team
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
# ==============================================================================
"""Frozen-KV MTP 投机解码输入数据结构模块。

本模块定义 Frozen-KV MTP（Multi-Token Prediction）算法的输入数据结构，
包括上下文信息和草稿/验证输入。Frozen-KV MTP 复用 EAGLE 的调度器/注意力契约，
但使用独立的类型标识，以便未来添加算法特化逻辑。

主要类：
    - FrozenKVMTPContext:       目标 KV 池 + 草稿逻辑层→目标物理层映射
    - FrozenKVMTPDraftInput:    草稿模型前向输入
    - FrozenKVMTPVerifyInput:   目标模型验证前向输入
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.speculative.eagle_info import (
    EagleDraftInput,
    EagleVerifyInput,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType


@dataclass(frozen=True)
class FrozenKVMTPContext:
    """Frozen-KV MTP 上下文：目标 KV 池 + 草稿逻辑层→目标物理层映射。

    Frozen-KV MTP 的草稿模型复用目标模型的 KV cache（冻结），
    通过 physical_layer_ids 将草稿的逻辑层索引映射到目标的物理层索引。

    属性：
        target_token_to_kv_pool: 目标模型的 token 级 KV cache 池
        physical_layer_ids:      {草稿逻辑层 idx → 目标物理层 idx} 映射

    示例：
        >>> ctx = FrozenKVMTPContext(
        ...     target_token_to_kv_pool=kv_pool,
        ...     physical_layer_ids={0: 4, 1: 9, 2: 14, 3: 19})
        >>> ctx.get_physical_layer_id(2)
        14
        >>> ctx.get_physical_layer_id(99)  # 不存在则抛 KeyError
        KeyError: ...
    """

    target_token_to_kv_pool: KVCache
    physical_layer_ids: Dict[int, int]

    def get_physical_layer_id(self, idx: int) -> int:
        """根据草稿逻辑层索引获取目标物理层索引。

        参数：
            idx: 草稿逻辑层索引

        返回：目标物理层索引

        异常：KeyError - 当 idx 不在映射中时
        """
        if idx not in self.physical_layer_ids:
            raise KeyError(
                f"FrozenKVMTPContext has no physical layer id for assistant "
                f"logical index {idx}; available: {sorted(self.physical_layer_ids)}"
            )
        return self.physical_layer_ids[idx]


@dataclass
class FrozenKVMTPDraftInput(EagleDraftInput):
    """Frozen-KV MTP 草稿模型前向输入。

    Frozen-KV MTP 当前复用 EAGLE 的调度器/注意力契约，
    但使用独立类型标识（FROZEN_KV_MTP_DRAFT），
    以便未来将算法特化逻辑迁移至此。

    示例：
        >>> inp = FrozenKVMTPDraftInput()
        >>> inp.input_type
        <SpecInputType.FROZEN_KV_MTP_DRAFT: ...>
    """

    def __post_init__(self):
        """初始化后设置输入类型为 FROZEN_KV_MTP_DRAFT。"""
        SpecInput.__init__(self, SpecInputType.FROZEN_KV_MTP_DRAFT)


@dataclass
class FrozenKVMTPVerifyInput(EagleVerifyInput):
    """Frozen-KV MTP 目标模型验证前向输入。

    复用 EAGLE 的验证输入结构，使用独立类型标识（FROZEN_KV_MTP_VERIFY）。

    示例：
        >>> inp = FrozenKVMTPVerifyInput()
        >>> inp.input_type
        <SpecInputType.FROZEN_KV_MTP_VERIFY: ...>
    """

    def __post_init__(self):
        """初始化后设置输入类型为 FROZEN_KV_MTP_VERIFY。"""
        SpecInput.__init__(self, SpecInputType.FROZEN_KV_MTP_VERIFY)
