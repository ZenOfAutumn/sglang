# Copyright 2023-2024 SGLang Team
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
"""
Dataclasses for embedding injection.

These are placed in a separate module to avoid circular imports between
io_struct.py and schedule_batch.py.

中译：用于「嵌入注入（embedding injection）」的数据类。
      所谓嵌入注入，是指调用方直接提供某些 token 位置上的嵌入向量（而非让模型从 token id
      查 embedding 表得到），常见于多模态等场景：把图像/音频等编码出的向量放到序列中的指定位置。
      之所以单独放在本模块，是为了避免 io_struct.py 与 schedule_batch.py 之间的循环导入。
"""

from dataclasses import dataclass
from typing import List, Union

import torch


@dataclass
class PositionalEmbeds:
    """Embeddings to place at specific token positions.

    Accepts either a list of [1, hidden_dim] tensors or a pre-stacked [N, hidden_dim] tensor.
    In both cases, __post_init__ stacks into a single [N, hidden_dim] tensor to reduce
    ZMQ serialization overhead.

    Attributes:
        embeds: Stacked tensor of shape [N, hidden_dim] after __post_init__.
        positions: List of positions where embeddings should be injected.

    中译：表示「要注入到特定 token 位置上的嵌入向量」。
          - 入参既可以是 N 个 [1, hidden_dim] 张量组成的列表，也可以是已经堆叠好的
            [N, hidden_dim] 单一张量；
          - 无论哪种形式，__post_init__ 都会统一规整为单个 [N, hidden_dim] 张量，
            以减少经 ZMQ 跨进程传输时的序列化开销（单张量比张量列表更省）。
          属性：
          - embeds：__post_init__ 之后形状统一为 [N, hidden_dim] 的堆叠张量。
          - positions：长度为 N 的位置列表，指明这些嵌入应注入到序列中的哪些 token 位置。
    """

    embeds: Union[List[torch.Tensor], torch.Tensor]
    positions: List[int]

    def __post_init__(self):
        """中译：构造后处理——把 embeds 规整为单一 [N, hidden_dim] 张量，并校验长度一致。

        副作用：原地改写 self.embeds（列表 -> 堆叠后的张量）。
        异常：当 embeds 为空，或堆叠后行数 N 与 positions 数量不一致时抛错。
        """
        # Normalize list of tensors into a single [N, hidden_dim] tensor.
        # Dispatch by element rank to avoid a per-element unsqueeze.
        # 中译：若 embeds 是张量列表，则按「单个元素的维度（rank）」分派不同的合并方式，
        #       从而避免对每个元素都做一次 unsqueeze。
        if isinstance(self.embeds, list):
            if not self.embeds:
                # 中译：空列表非法——这里 torch.cat 空列表会直接抛错（原注释 "raises — empty is invalid"）。
                self.embeds = torch.cat(self.embeds, dim=0)  # raises — empty is invalid
            elif self.embeds[0].dim() == 1:
                # [hidden_dim] elements → stack adds the leading dim.
                # 中译：元素是一维 [hidden_dim]——用 stack 在前面补出新维度，得到 [N, hidden_dim]。
                self.embeds = torch.stack(self.embeds, dim=0)
            else:
                # [1, hidden_dim] (already has the leading dim) → plain concat.
                # 中译：元素已是 [1, hidden_dim]（已有前导维度）——直接 cat 拼接即可。
                self.embeds = torch.cat(self.embeds, dim=0)
        # 中译：校验嵌入行数 N 与注入位置数量必须相等，否则无法一一对应注入。
        if self.embeds.shape[0] != len(self.positions):
            raise ValueError(
                f"embeds length ({self.embeds.shape[0]}) != "
                f"positions length ({len(self.positions)})"
            )
