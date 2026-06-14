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
存储一次前向（forward）所需的信息。

一个批次（batch）的数据结构流转如下：

ScheduleBatch -> ModelWorkerBatch -> ForwardBatch

- ScheduleBatch 由 `scheduler.py::Scheduler` 管理。
  它包含高层的调度数据，大部分数据在 CPU 上。
- ModelWorkerBatch 由 `tp_worker.py::TpModelWorker` 管理。
  它是 `ScheduleBatch` 的子集，仅包含与 GPU 上模型前向相关的数据，
  会从 CPU 调度器转换给 GPU 模型运行器。
- ForwardBatch 由 `model_runner.py::ModelRunner` 管理。
  它包含底层的张量数据，大部分是 GPU 张量。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
from functools import total_ordering
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from sglang.srt.distributed.parallel_state import (
    get_moe_expert_parallel_world_size,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.attention.nsa.utils import NSAContextParallelMetadata
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    get_attention_cp_size,
    get_attention_dp_rank,
    get_attention_tp_rank,
    get_attention_tp_size,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.layers.utils.cp_utils import ContextParallelMetadata
from sglang.srt.model_executor.forward_batch_deepseek_mha_mixin import (
    ForwardBatchDeepSeekMHAMixin,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    is_cuda,
    is_hip,
    is_npu,
    support_triton,
)
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch, MultimodalInputs
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
    from sglang.srt.speculative.spec_info import SpecInput, SpeculativeAlgorithm

_is_npu = is_npu()


class ForwardMode(IntEnum):
    """前向模式：描述本次前向是 prefill、decode、混合，还是投机解码/PD/dLLM 等特殊场景。"""

    # 扩展一个序列。序列开头部分的 KV 缓存可能已计算（如系统提示词）。
    # 俗称为 "prefill"。
    EXTEND = auto()
    # 解码一个 token。
    DECODE = auto()
    # 做 chunked prefill 时同时包含 EXTEND 与 DECODE（混合分块）。
    MIXED = auto()
    # 无序列可前向。在数据并行注意力下，某些 worker 若未分配序列则为 IDLE。
    IDLE = auto()

    # 用于投机解码：在目标模型中验证一个批次。
    TARGET_VERIFY = auto()
    # 用于投机解码：在草稿模型中扩展一个批次。
    DRAFT_EXTEND = auto()

    DRAFT_EXTEND_V2 = auto()  # EAGLE v2 草稿扩展（固定形状的 logits 输出）

    # 用于 PD 分离的 decode worker：
    # 表示一批 KV 缓存已就绪、可直接开始解码的请求。
    PREBUILT = auto()

    # 用于 PD 复用（multiplexing）的拆分 prefill。
    SPLIT_PREFILL = auto()

    # 用于扩散式 LLM（dLLM）。
    DLLM_EXTEND = auto()

    def is_prefill(self):
        """是否为 prefill（等价于 is_extend）。"""
        return self.is_extend()

    def is_extend(self, include_draft_extend_v2: bool = False):
        return (
            self == ForwardMode.EXTEND
            or self == ForwardMode.MIXED
            or self == ForwardMode.DRAFT_EXTEND
            or (include_draft_extend_v2 and self == ForwardMode.DRAFT_EXTEND_V2)
            or self == ForwardMode.TARGET_VERIFY
            or self == ForwardMode.SPLIT_PREFILL
            or self == ForwardMode.DLLM_EXTEND
        )

    def is_context_parallel_extend(self, include_draft_extend_v2: bool = False):
        return (
            self == ForwardMode.EXTEND
            or self == ForwardMode.MIXED
            or (
                self == ForwardMode.DRAFT_EXTEND_V2
                if include_draft_extend_v2
                else False
            )
        )

    def is_decode(self):
        """是否为解码模式。"""
        return self == ForwardMode.DECODE

    def is_mixed(self):
        """是否为混合（prefill+decode）模式。"""
        return self == ForwardMode.MIXED

    def is_idle(self):
        """是否为空闲模式（无序列可前向）。"""
        return self == ForwardMode.IDLE

    def is_decode_or_idle(self):
        """是否为解码或空闲模式。"""
        return self == ForwardMode.DECODE or self == ForwardMode.IDLE

    def is_target_verify(self):
        """是否为投机解码的目标验证模式。"""
        return self == ForwardMode.TARGET_VERIFY

    def is_draft_extend(self, include_v2: bool = False):
        return self == ForwardMode.DRAFT_EXTEND or (
            include_v2 and self == ForwardMode.DRAFT_EXTEND_V2
        )

    def is_draft_extend_v2(self):
        # 用于 eagle v2 worker 中固定形状的 logits 输出
        return self == ForwardMode.DRAFT_EXTEND_V2

    def is_extend_or_draft_extend_or_mixed(self, include_draft_extend_v2: bool = False):
        return (
            self == ForwardMode.EXTEND
            or self == ForwardMode.DRAFT_EXTEND
            or self == ForwardMode.MIXED
            or self == ForwardMode.SPLIT_PREFILL
            or (include_draft_extend_v2 and self == ForwardMode.DRAFT_EXTEND_V2)
        )

    def is_cuda_graph(self):
        """是否可使用 CUDA Graph（仅固定形状的 decode/verify/idle/dllm 模式）。"""
        return (
            self == ForwardMode.DECODE
            or self == ForwardMode.TARGET_VERIFY
            or self == ForwardMode.IDLE
            or self == ForwardMode.DLLM_EXTEND
        )

    def is_cpu_graph(self):
        return self == ForwardMode.DECODE

    def is_split_prefill(self):
        return self == ForwardMode.SPLIT_PREFILL

    def is_extend_without_speculative(self):
        return (
            self.is_extend()
            and not self.is_target_verify()
            and not self.is_draft_extend()
        )

    def is_prebuilt(self):
        """是否为预构建模式（PD 分离 decode 端 KV 已就绪）。"""
        return self == ForwardMode.PREBUILT

    def is_dllm_extend(self):
        """是否为扩散式 LLM 的扩展模式。"""
        return self == ForwardMode.DLLM_EXTEND


@total_ordering
class CaptureHiddenMode(IntEnum):
    """隐藏状态捕获模式：控制是否以及如何保存隐藏状态（供投机解码等使用）。"""

    # 不捕获任何隐藏状态。
    NULL = 0
    # 捕获最后一个 token 的隐藏状态。
    LAST = 1
    # 捕获所有 token 的隐藏状态。
    FULL = 2

    def need_capture(self):
        """是否需要捕获隐藏状态。"""
        return self != CaptureHiddenMode.NULL

    def is_full(self):
        """是否捕获所有 token。"""
        return self == CaptureHiddenMode.FULL

    def is_last(self):
        """是否仅捕获最后一个 token。"""
        return self == CaptureHiddenMode.LAST

    def __lt__(self, other):
        return self.value < other.value


def compute_local_num_token_non_padded(
    global_num_token_non_padded: torch.Tensor,
    num_tokens_per_dp: int,
) -> torch.Tensor:
    """Compute local non-padded token count for this attention-TP rank.

    Converts a global count (across all TP ranks) to a local count for this rank.
    The "global" scope is within the current DP rank; DP is handled via num_tokens_per_dp.
    """
    attn_tp_rank = get_attention_tp_rank()
    attn_tp_size = get_attention_tp_size()
    tokens_per_rank = num_tokens_per_dp // attn_tp_size

    return torch.clamp(
        global_num_token_non_padded - tokens_per_rank * attn_tp_rank,
        0,
        tokens_per_rank,
    )


@dataclass
class NgramEmbeddingInfo:
    """LongCat 模型的 Ngram embedding 状态。"""

    token_table: torch.Tensor
    column_starts: torch.Tensor
    req_lens: torch.Tensor
    out_column_starts: torch.Tensor
    out_req_lens: torch.Tensor

    @classmethod
    def create(
        cls,
        token_table: torch.Tensor,
        batch_size: int,
        device: torch.device,
        column_starts=None,
        req_lens=None,
    ) -> NgramEmbeddingInfo:
        info = cls(
            token_table=token_table,
            column_starts=torch.empty(batch_size, dtype=torch.int32, device=device),
            req_lens=torch.empty(batch_size, dtype=torch.int32, device=device),
            out_column_starts=torch.empty(batch_size, dtype=torch.int32, device=device),
            out_req_lens=torch.empty(batch_size, dtype=torch.int32, device=device),
        )
        if column_starts is not None:
            info.column_starts[:] = column_starts
        if req_lens is not None:
            info.req_lens[:] = req_lens
        return info

    def slice(self, bs: int) -> NgramEmbeddingInfo:
        return NgramEmbeddingInfo(
            token_table=self.token_table,
            column_starts=self.column_starts[:bs],
            req_lens=self.req_lens[:bs],
            out_column_starts=self.out_column_starts[:bs],
            out_req_lens=self.out_req_lens[:bs],
        )


@dataclass
class ForwardBatch(ForwardBatchDeepSeekMHAMixin):
    """存储一次前向的所有输入（大部分为 GPU 张量）。

    最核心参数示例（假设同时处理 2 个请求 reqA / reqB）：

    场景一：Prefill（首次处理，reqA 输入 3 个 token，reqB 输入 2 个 token）
        forward_mode    = ForwardMode.EXTEND       # prefill 路径
        batch_size      = 2                        # 2 个序列
        input_ids       = [a0, a1, a2, b0, b1]     # 两请求 token 拼接，shape=[5]
        seq_lens        = [3, 2]                   # 各序列当前总长度
        seq_lens_sum    = 5                        # token 总数（=input_ids 长度）
        req_pool_indices= [0, 1]                   # reqA/reqB 在请求池中的槽位
        positions       = [0, 1, 2, 0, 1]          # 各 token 在各自序列内的位置
        out_cache_loc   = [10, 11, 12, 13, 14]     # 这 5 个 token 的 KV 写入位置
        # extend 专属：
        extend_num_tokens  = 5                     # 本次新增 token 数
        extend_seq_lens    = [3, 2]                # 各请求本次新增长度
        extend_prefix_lens = [0, 0]                # 无历史缓存前缀

    场景二：Decode（已各生成若干 token，本步每个请求各解码 1 个新 token）
        forward_mode    = ForwardMode.DECODE       # decode 路径
        batch_size      = 2
        input_ids       = [a_last, b_last]         # 每请求仅 1 个新 token，shape=[2]
        seq_lens        = [4, 3]                   # 序列已增长（含本步前的长度）
        seq_lens_sum    = 7
        req_pool_indices= [0, 1]
        positions       = [3, 2]                   # = seq_lens - 1
        out_cache_loc   = [15, 16]                 # 本步 2 个新 token 的 KV 写入位置
        # 无需 extend_* 字段

    两种场景都依赖：sampling_info（采样）、attn_backend（注意力计算）、
    req_to_token_pool / token_to_kv_pool（KV 缓存寻址）。
    """

    # ★核心：前向模式（prefill/decode/mixed/...），决定本次前向的整体执行路径
    forward_mode: ForwardMode
    # ★核心：批次大小（序列数）
    batch_size: int
    # ★核心：输入 token id（模型前向的实际输入）
    input_ids: torch.Tensor
    # ★核心：各请求在 req_to_token_pool 中的索引（定位每个请求的 KV 槽位）
    req_pool_indices: torch.Tensor
    # ★核心：各序列长度（注意力计算与位置编码的基础）
    seq_lens: torch.Tensor
    # ★核心：输出 token 在 token_to_kv_pool 中的索引（本次新 KV 的存放位置）
    out_cache_loc: torch.Tensor

    # ★核心：所有序列长度之和（即本次 token 总数，决定多数张量的第 0 维）
    seq_lens_sum: int

    # 未被分块前的原始序列长度（与 Qwen-1M 相关）
    orig_seq_lens: Optional[torch.Tensor] = None

    # 输出 token 在 token_to_kv_pool_swa（滑动窗口池）中的索引
    out_cache_loc_swa: Optional[torch.Tensor] = None
    # 用于跟踪 mamba 状态的索引
    mamba_track_indices: Optional[torch.Tensor] = None  # 形状: [b], int64
    # 跟踪 mamba 状态的掩码（如需）
    mamba_track_mask: Optional[torch.Tensor] = None  # 形状: [b], bool
    # 被掩码时跟踪 mamba 状态的序列长，仅 prefill 使用
    mamba_track_seqlens: Optional[torch.Tensor] = None  # 形状: [b], int64

    # 可选的 CPU 端 seq_lens
    seq_lens_cpu: Optional[torch.Tensor] = None

    # logprob 相关
    return_logprob: bool = False  # 是否返回 logprob
    top_logprobs_nums: Optional[List[int]] = None  # 各请求要返回的 top-k logprob 个数
    token_ids_logprobs: Optional[List[List[int]]] = None  # 指定要返回 logprob 的 token id

    # logits 与 logprob 后处理
    next_token_logits_buffer: torch.Tensor = None  # 下一个 token 的 logits 缓冲区
    temp_scaled_logprobs: bool = False  # 是否用温度缩放 logprob
    temperature: torch.Tensor = None  # 采样温度
    top_p_normalized_logprobs: bool = False  # 是否对 logprob 做 top-p 归一化
    top_p: torch.Tensor = None  # top-p 采样参数

    # 位置信息
    positions: torch.Tensor = None  # ★核心：各 token 的位置索引（旋转位置编码 RoPE 依赖）

    # extend（prefill）相关
    extend_num_tokens: Optional[int] = None  # ★核心(extend)：本次扩展的 token 总数
    extend_seq_lens: Optional[torch.Tensor] = None  # ★核心(extend)：各请求本次扩展的长度
    extend_prefix_lens: Optional[torch.Tensor] = None  # ★核心(extend)：各请求已缓存的前缀长度
    extend_start_loc: Optional[torch.Tensor] = None  # 各请求在扩展张量中的起始偏移
    extend_prefix_lens_cpu: Optional[List[int]] = None  # 前缀长度（CPU 端）
    extend_seq_lens_cpu: Optional[List[int]] = None  # 扩展长度（CPU 端）
    extend_logprob_start_lens_cpu: Optional[List[int]] = None  # 各请求 logprob 起始位置（CPU 端）
    extend_input_logprob_token_ids_gpu: Optional[torch.Tensor] = None  # 输入 logprob 对应的 token id（GPU 端）

    # 拆分 prefill 相关：拆分 prefill 的中间值
    hidden_states: torch.Tensor = None  # 隐藏状态
    residual: torch.Tensor = None  # 残差
    model_specific_states: Dict[str, any] = None  # 模型特定的中间状态
    split_index: int = 0  # 当前拆分到的层索引

    # 多模态相关
    mm_inputs: Optional[List[MultimodalInputs]] = None  # 多模态输入

    # 编码器-解码器（encoder-decoder）相关
    encoder_cached: Optional[List[bool]] = None  # 各请求的 encoder 输出是否已缓存
    encoder_lens: Optional[torch.Tensor] = None  # 各请求 encoder 部分长度
    encoder_lens_cpu: Optional[List[int]] = None  # encoder 长度（CPU 端）
    encoder_out_cache_loc: Optional[torch.Tensor] = None  # encoder 输出在 KV 池中的位置

    # LoRA 相关
    lora_ids: Optional[List[str]] = None  # 各请求使用的 LoRA 适配器 id

    # 输入 embedding（直接传入嵌入而非 token id）
    input_embeds: Optional[torch.Tensor] = None

    # 交叉编码器（cross-encoder）模型的 token 类型 id
    token_type_ids: Optional[torch.Tensor] = None

    # ★核心：采样信息（温度、top-p、top-k 等，决定如何从 logits 采样出下一个 token）
    sampling_info: SamplingBatchInfo = None

    # 注意力后端与显存池
    req_to_token_pool: ReqToTokenPool = None  # ★核心：请求->token 映射池
    token_to_kv_pool: KVCache = None  # ★核心：token->KV 缓存池
    attn_backend: AttentionBackend = None  # ★核心：注意力后端实现（实际执行 attention 计算）

    # DP（数据并行）注意力相关
    original_global_num_tokens_cpu: Optional[List[int]] = None  # 原始全局 token 数（CPU 端）
    global_num_tokens_cpu: Optional[List[int]] = None  # 全局 token 数（CPU 端）
    global_num_tokens_gpu: Optional[torch.Tensor] = None  # 全局 token 数（GPU 端）
    # 在捕获 cuda graph 时必须为 None
    global_num_tokens_for_logprob_cpu: Optional[List[int]] = None  # 用于 logprob 的全局 token 数（CPU）
    global_num_tokens_for_logprob_gpu: Optional[torch.Tensor] = None  # 用于 logprob 的全局 token 数（GPU）
    # DP 注意力的填充模式
    dp_padding_mode: Optional[DpPaddingMode] = None
    # 对于 extend，logits 处理器中的本地起始位置与 token 数不同；
    # 会在 get_dp_local_info 中计算，并在 LogitsMetadata.from_forward_batch 中重算
    dp_local_start_pos: Optional[torch.Tensor] = None  # 运行时缓存信息
    dp_local_num_tokens: Optional[torch.Tensor] = None  # 运行时缓存信息
    global_dp_buffer_len: Optional[int] = None  # 全局 DP 缓冲区长度
    is_extend_in_batch: bool = False  # 本批次是否含 extend
    all_extend_in_batch: bool = False  # 本批次是否全为 extend
    can_run_dp_cuda_graph: bool = False  # 是否可跑 DP 的 CUDA Graph
    global_forward_mode: Optional[ForwardMode] = None  # 全局前向模式

    # 本批次是否仅 prefill（无需生成 token）
    is_prefill_only: bool = False

    # 投机解码相关
    spec_info: Optional[SpecInput] = None  # 投机解码输入信息
    spec_algorithm: SpeculativeAlgorithm = None  # 投机解码算法
    mm_input_embeds: Optional[torch.Tensor] = None  # 多模态输入嵌入
    capture_hidden_mode: CaptureHiddenMode = None  # 隐藏状态捕获模式

    # 填充（padding）相关
    padded_static_len: int = -1  # 静态填充后的长度，-1 表示未填充
    num_token_non_padded: Optional[torch.Tensor] = None  # 非填充 token 数（标量张量）
    num_token_non_padded_cpu: int = None  # 非填充 token 数（CPU 端）

    # Qwen2-VL 的 mrope 位置
    mrope_positions: torch.Tensor = None

    # 双批重叠（two-batch overlap）相关
    tbo_split_seq_index: Optional[int] = None  # 拆分点所在的序列索引
    tbo_parent_token_range: Optional[Tuple[int, int]] = None  # 在父批次中的 token 区间
    tbo_padded_len: Optional[int] = None  # 填充后长度
    tbo_children: Optional[List[ForwardBatch]] = None  # 拆分出的子批次

    # Matryoshka embedding 的输出维度
    dimensions: Optional[list[int]] = None

    attn_cp_metadata: Optional[ContextParallelMetadata] = None  # 注意力上下文并行元数据
    # 记录 NSA 上下文并行的序列拆分元数据
    nsa_cp_metadata: Optional[NSAContextParallelMetadata] = None

    # 是否返回归一化前的隐藏状态
    return_hidden_states_before_norm: bool = False

    # HiSparse 协调器
    hisparse_coordinator: Optional[HiSparseCoordinator] = None

    # Ngram embedding 信息（LongCat）
    ngram_embedding_info: Optional[NgramEmbeddingInfo] = None

    # 供 dumper 使用：跨步序列跟踪的请求 id 列表
    rids: Optional[List[str]] = None

    @classmethod
    def init_new(
        cls,
        batch: ModelWorkerBatch,
        model_runner: ModelRunner,
    ):
        """从 ModelWorkerBatch 与 ModelRunner 构造一个新的 ForwardBatch（填充各字段并准备前向所需元数据）。"""
        ret = cls(
            forward_mode=batch.forward_mode,
            batch_size=len(batch.seq_lens),
            input_ids=batch.input_ids,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            out_cache_loc=batch.out_cache_loc,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_track_mask=batch.mamba_track_mask,
            mamba_track_seqlens=batch.mamba_track_seqlens,
            mm_inputs=batch.multimodal_inputs,
            encoder_cached=batch.encoder_cached,
            encoder_lens=batch.encoder_lens,
            encoder_lens_cpu=batch.encoder_lens_cpu,
            encoder_out_cache_loc=batch.encoder_out_cache_loc,
            seq_lens_sum=batch.seq_lens_sum,
            seq_lens_cpu=batch.seq_lens_cpu,
            orig_seq_lens=batch.orig_seq_lens,
            return_logprob=batch.return_logprob,
            top_logprobs_nums=batch.top_logprobs_nums,
            token_ids_logprobs=batch.token_ids_logprobs,
            is_extend_in_batch=batch.is_extend_in_batch,
            all_extend_in_batch=batch.all_extend_in_batch,
            can_run_dp_cuda_graph=batch.can_run_dp_cuda_graph,
            global_forward_mode=batch.global_forward_mode,
            is_prefill_only=batch.is_prefill_only,
            lora_ids=batch.lora_ids,
            sampling_info=batch.sampling_info,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool=model_runner.token_to_kv_pool,
            attn_backend=model_runner.attn_backend,
            spec_algorithm=batch.spec_algorithm,
            spec_info=batch.spec_info,
            capture_hidden_mode=batch.capture_hidden_mode,
            input_embeds=batch.input_embeds,
            token_type_ids=batch.token_type_ids,
            tbo_split_seq_index=batch.tbo_split_seq_index,
            dimensions=batch.dimensions,
            return_hidden_states_before_norm=batch.return_hidden_states_before_norm,
            rids=[req.rid for req in batch.reqs],
        )
        # 目标设备（如 cuda:0）；后续所有张量都会异步搬到该设备
        device = model_runner.device

        # 若需要返回输入 token 的 logprob，则把对应的 token id 异步搬到 GPU
        if batch.extend_input_logprob_token_ids is not None:
            ret.extend_input_logprob_token_ids_gpu = (
                batch.extend_input_logprob_token_ids.to(device, non_blocking=True)
            )

        # 本批次实际 token 总数（input_ids 长度）；无 input_ids 时记为 0
        num_tokens = len(batch.input_ids) if batch.input_ids is not None else 0
        # 仅当开启 MoE 专家并行（EP>1）时，才需要把非填充 token 数放到 GPU 供 kernel 使用
        if enable_num_token_non_padded(model_runner.server_args):
            ret.num_token_non_padded = torch.tensor(num_tokens, dtype=torch.int32).to(
                device, non_blocking=True
            )
        # CPU 侧始终记录非填充 token 数，供调度/统计逻辑使用
        ret.num_token_non_padded_cpu = num_tokens

        # 用于 MLP 同步（DP 注意力下各 rank 对齐 token 数）
        # global_num_tokens 记录每个 DP rank 的 token 数；仅在启用 DP 注意力时非空
        if batch.global_num_tokens is not None:
            # 二者必须同时存在：logprob 版本用于对齐计算 logprob 时的 token 数
            assert batch.global_num_tokens_for_logprob is not None

            # 处理 global_num_tokens 与 global_num_tokens_for_logprob
            if batch.spec_info is not None:
                # 投机解码下，实际跑的 token 数会受草稿 token 影响，需重新调整
                spec_info: SpecInput = batch.spec_info
                global_num_tokens, global_num_tokens_for_logprob = (
                    spec_info.get_spec_adjusted_global_num_tokens(batch)
                )
            else:
                # 非投机：直接采用批次中给定的全局 token 数
                global_num_tokens = batch.global_num_tokens
                global_num_tokens_for_logprob = batch.global_num_tokens_for_logprob

            # 保留调整前的原始全局 token 数（CPU 侧），便于后续还原/对照
            ret.original_global_num_tokens_cpu = batch.global_num_tokens
            # CPU 侧保存（可能已被投机调整后的）各 DP rank token 数
            ret.global_num_tokens_cpu = global_num_tokens
            # GPU 侧版本：用于 kernel/通信时的 token 数对齐，异步搬运
            ret.global_num_tokens_gpu = torch.tensor(
                global_num_tokens, dtype=torch.int64
            ).to(device, non_blocking=True)

            # CPU 侧保存计算 logprob 所需的各 rank token 数
            ret.global_num_tokens_for_logprob_cpu = global_num_tokens_for_logprob
            # GPU 侧版本，异步搬运
            ret.global_num_tokens_for_logprob_gpu = torch.tensor(
                global_num_tokens_for_logprob, dtype=torch.int64
            ).to(device, non_blocking=True)

        # 空闲模式（无任何请求，仅占位以保持 DP 各 rank 步调一致）
        if ret.forward_mode.is_idle():
            # 无序列可处理，位置张量置为空，并提前返回
            ret.positions = torch.empty((0,), dtype=torch.int64, device=device)
            return ret

        # 用扩散式 LLM 或投机信息覆盖 positions
        if batch.dllm_config is not None:
            # 扩散式 LLM：以固定块为单位生成位置
            block_size = batch.dllm_config.block_size
            # 使用 int64 以兼容 AMD（HIP）/NPU 的旋转位置编码 kernel，否则用 int32
            positions_dtype = torch.int64 if is_hip() or _is_npu else torch.int32
            # 对每个块的起始偏移展开出 block_size 个连续位置，拼接成完整 positions
            ret.positions = torch.tensor(
                [
                    i
                    for block_offset in batch.dllm_block_offsets
                    for i in range(block_offset, block_offset + block_size)
                ],
                dtype=positions_dtype,
            ).to(device, non_blocking=True)
        elif (
            ret.spec_info is not None
            and getattr(ret.spec_info, "positions", None) is not None
        ):
            # 投机解码：草稿/验证阶段已预先算好 positions，直接复用
            ret.positions = ret.spec_info.positions

        # 初始化位置信息（若上面未被 dllm/spec 覆盖）
        if ret.forward_mode.is_decode() or ret.forward_mode.is_target_verify():
            # 解码/目标验证：每个序列只新增 1 个 token，位置即该序列当前长度
            if ret.positions is None:
                # clamp_position 会把位置下限钳到 0，避免空序列出现负数索引
                ret.positions = clamp_position(batch.seq_lens)
        else:
            # extend（prefill/分块预填充）：需为每个新增 token 计算其绝对位置
            # 断言：extend 元数据必须是 Python list（CPU 侧原始数据）
            assert isinstance(batch.extend_seq_lens, list)
            assert isinstance(batch.extend_prefix_lens, list)
            # 各请求本次新增的 token 数，转为 int32 张量并异步搬到 GPU
            ret.extend_seq_lens = torch.tensor(
                batch.extend_seq_lens, dtype=torch.int32
            ).to(device, non_blocking=True)
            # 各请求已有的前缀长度（已在 KV cache 中的部分），同样搬到 GPU
            ret.extend_prefix_lens = torch.tensor(
                batch.extend_prefix_lens, dtype=torch.int32
            ).to(device, non_blocking=True)
            # 本批 extend 的 token 总数
            ret.extend_num_tokens = batch.extend_num_tokens
            # 由前缀长度+扩展长度计算每个 token 的位置，以及各请求在拼接序列中的起始偏移
            positions, ret.extend_start_loc = compute_position(
                model_runner.server_args.attention_backend,
                ret.extend_prefix_lens,
                ret.extend_seq_lens,
                ret.extend_num_tokens,
            )
            # 若位置未被前面的 dllm/spec 分支覆盖，则采用此处计算结果
            if ret.positions is None:
                ret.positions = positions
            # 在 CPU 侧保留前缀长度，供后续逻辑（如 logprob 计算）使用
            ret.extend_prefix_lens_cpu = batch.extend_prefix_lens
            # CPU 侧保留各请求扩展长度
            ret.extend_seq_lens_cpu = batch.extend_seq_lens
            # CPU 侧保留各请求 logprob 的起始位置
            ret.extend_logprob_start_lens_cpu = batch.extend_logprob_start_lens

        # LongCat 等使用 ngram embedding 的模型：构建本批次的 ngram embedding 信息
        if model_runner.use_ngram_embedding:
            ret._init_ngram_embedding_info(batch, model_runner, device)

        # mrope 模型（如 Qwen2-VL 多模态）：需计算多维（3D）旋转位置编码
        if model_runner.model_is_mrope:
            if (
                ret.spec_info is not None
                and getattr(ret.spec_info, "positions", None) is not None
            ):
                # 投机解码路径下的 mrope 位置计算（基于 spec_info.positions）
                ret._compute_spec_mrope_positions(model_runner, batch)
            else:
                # 常规路径下的 mrope 位置计算
                ret._compute_mrope_positions(model_runner, batch)

        # 混合 SWA（滑动窗口注意力）模型：把 full KV cache 的写入位置一次性
        # 翻译为 SWA 窗口内的位置，避免每个 SWA 层重复计算
        if model_runner.is_hybrid_swa and ret.out_cache_loc is not None:
            ret.out_cache_loc_swa = (
                model_runner.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                    ret.out_cache_loc
                )
            )

        # 初始化 LoRA 信息
        if model_runner.server_args.enable_lora:
            # 未开启「重叠加载」时，需在跑这批之前同步把所需 LoRA 适配器取入显存池
            if not model_runner.server_args.enable_lora_overlap_loading:
                model_runner.lora_manager.fetch_new_loras(set(ret.lora_ids))

            # 为本批次准备 LoRA（如构建按请求的适配器索引/批信息）
            model_runner.lora_manager.prepare_lora_batch(ret)

        # 返回构造并初始化完成的 ForwardBatch
        return ret

    def adjust_num_token_non_padded_for_attn_tp(self, server_args) -> None:
        """将 num_token_non_padded 转为当前注意力-TP rank 的本地值。"""
        from sglang.srt.utils.common import require_mlp_tp_gather

        dp_rank = get_attention_dp_rank()
        assert self.global_num_tokens_cpu is not None

        if require_mlp_tp_gather(server_args):
            num_tokens_per_dp = self.global_num_tokens_cpu[dp_rank]
        else:
            num_tokens_per_dp = self.global_num_tokens_cpu[0]

        self.num_token_non_padded = compute_local_num_token_non_padded(
            global_num_token_non_padded=self.num_token_non_padded,
            num_tokens_per_dp=num_tokens_per_dp,
        )

    def merge_mm_inputs(self) -> Optional[MultimodalInputs]:
        """将批次中所有多模态输入合并为单个 MultiModalInputs 对象。

        返回：若为 None，表示当前批次不含多模态输入。
        """
        if not self.mm_inputs or all(x is None for x in self.mm_inputs):
            return None
        # 过滤掉 None
        valid_inputs = [x for x in self.mm_inputs if x is not None]

        # TODO: 这是否开销较大？
        # 一个避免导入 `MultimodalInputs` 的权宜做法
        merged = valid_inputs[0].__class__(mm_items=[])

        # 合并其余输入
        for mm_input in valid_inputs:
            merged.merge(mm_input)

        return merged

    def contains_image_inputs(self) -> bool:
        """是否含图像输入。"""
        if self.mm_inputs is None:
            return False
        return any(
            mm_input is not None and mm_input.contains_image_inputs()
            for mm_input in self.mm_inputs
        )

    def contains_audio_inputs(self) -> bool:
        """是否含音频输入。"""
        if self.mm_inputs is None:
            return False
        return any(
            mm_input is not None and mm_input.contains_audio_inputs()
            for mm_input in self.mm_inputs
        )

    def contains_video_inputs(self) -> bool:
        """是否含视频输入。"""
        if self.mm_inputs is None:
            return False
        return any(
            mm_input is not None and mm_input.contains_video_inputs()
            for mm_input in self.mm_inputs
        )

    def contains_mm_inputs(self) -> bool:
        """是否含任意多模态（图/音/视频）输入。"""
        return (
            self.contains_audio_inputs()
            or self.contains_video_inputs()
            or self.contains_image_inputs()
        )

    def _init_ngram_embedding_info(
        self, batch: ModelWorkerBatch, model_runner: ModelRunner, device: torch.device
    ):
        if self.forward_mode.is_decode():
            column_starts, req_lens = self.seq_lens - 1, 1
        else:
            column_starts, req_lens = self.extend_prefix_lens, self.extend_seq_lens
        self.ngram_embedding_info = NgramEmbeddingInfo.create(
            batch.ne_token_table,
            self.batch_size,
            device,
            column_starts=column_starts,
            req_lens=req_lens,
        )

    def _compute_spec_mrope_positions(
        self, model_runner: ModelRunner, batch: ModelWorkerBatch
    ):
        """投机解码场景下计算 mrope（多维旋转位置编码）位置。"""
        # TODO 支持批量化的 deltas
        batch_size = self.seq_lens.shape[0]
        device = model_runner.device
        mm_inputs = batch.multimodal_inputs

        if batch.forward_mode.is_draft_extend():  # draft_extend_after_decode
            mrope_deltas = []
            extend_lens = []
            for batch_idx in range(batch_size):
                extend_seq_len = batch.extend_seq_lens[batch_idx]
                extend_lens.append(extend_seq_len)
                mrope_delta = (
                    torch.zeros(1, dtype=torch.int64)
                    if mm_inputs[batch_idx] is None
                    else mm_inputs[batch_idx].mrope_position_delta.squeeze(0)
                )
                mrope_deltas.append(mrope_delta.to(device=device))
            position_chunks = torch.split(batch.spec_info.positions, extend_lens)
            mrope_positions_list = [
                pos_chunk + delta
                for pos_chunk, delta in zip(position_chunks, mrope_deltas)
            ]
            next_input_positions = (
                torch.cat(mrope_positions_list, dim=0).unsqueeze(0).repeat(3, 1)
            )

        else:  # target_verify or draft_decode
            seq_positions = batch.spec_info.positions.view(batch_size, -1)
            # Split text-only and mixed batches here because SpecV2 text-only batches can avoid an extra D2H.
            if all(mm_input is None for mm_input in mm_inputs):
                mrope_delta_tensor = torch.zeros(
                    (batch_size, 1), dtype=torch.int64, device=device
                )
            else:
                mrope_deltas = [
                    (
                        torch.zeros(1, dtype=torch.int64)
                        if mm_inputs[i] is None
                        else mm_inputs[i].mrope_position_delta.squeeze(0)
                    )
                    for i in range(batch_size)
                ]
                mrope_delta_tensor = torch.stack(mrope_deltas, dim=0).to(device=device)
            next_input_positions = (
                (seq_positions + mrope_delta_tensor).flatten().unsqueeze(0).repeat(3, 1)
            )

        self.mrope_positions = next_input_positions

    def _expand_mrope_from_input(
        self,
        mm_input: MultimodalInputs,
        seq_len: int,
    ) -> torch.Tensor:
        """从多模态输入的 mrope_position_delta 扩展出当前序列长对应的 mrope 位置。"""
        # 在 CPU 上做以下计算，避免频繁的小 kernel 调用
        if mm_input.mrope_position_delta_repeated_cache is None:
            mm_input.mrope_position_delta_repeated_cache = (
                (mm_input.mrope_position_delta - 1).flatten().unsqueeze(0).repeat(3, 1)
            )
        mrope_positions = mm_input.mrope_position_delta_repeated_cache + seq_len
        return mrope_positions

    def _compute_mrope_positions(
        self, model_runner: ModelRunner, batch: ModelWorkerBatch
    ):
        """计算 mrope（多维旋转位置编码）位置，逐请求区分 decode 与 extend、纯文本与多模态。"""
        # 形状为 batch_size * [3 * seq_len]
        batch_size = self.seq_lens_cpu.shape[0]
        mrope_positions_list = [[]] * batch_size
        for batch_idx in range(batch_size):
            mm_input = batch.multimodal_inputs[batch_idx]
            if self.forward_mode.is_decode():
                # 3 * N
                if (
                    mm_input is None
                    or get_global_server_args().rl_on_policy_target is not None
                ):
                    mrope_positions_list[batch_idx] = torch.full(
                        (3, 1),
                        self.seq_lens_cpu[batch_idx] - 1,
                        dtype=torch.int64,
                    )
                else:
                    mrope_positions = self._expand_mrope_from_input(
                        mm_input, self.seq_lens_cpu[batch_idx]
                    )
                    mrope_positions_list[batch_idx] = mrope_positions
            elif self.forward_mode.is_extend(include_draft_extend_v2=True):
                # extend（prefill）：按前缀长度与扩展长度取位置区间
                extend_seq_len, extend_prefix_len = (
                    batch.extend_seq_lens[batch_idx],
                    batch.extend_prefix_lens[batch_idx],
                )
                if (
                    mm_input is None
                    or get_global_server_args().rl_on_policy_target is not None
                ):
                    # 纯文本
                    mrope_positions = torch.tensor(
                        [
                            [
                                pos
                                for pos in range(
                                    extend_prefix_len,
                                    extend_prefix_len + extend_seq_len,
                                )
                            ]
                        ]
                        * 3
                    )
                else:
                    mrope_positions = mm_input.mrope_positions[
                        :,
                        extend_prefix_len : extend_prefix_len + extend_seq_len,
                    ]
                    if mrope_positions.numel() == 0:
                        mrope_positions = self._expand_mrope_from_input(
                            mm_input, self.seq_lens_cpu[batch_idx]
                        )
                mrope_positions_list[batch_idx] = mrope_positions

        self.mrope_positions = torch.cat(
            [pos for pos in mrope_positions_list],
            dim=1,
        ).to(dtype=torch.int64, device=model_runner.device, non_blocking=True)

    def _pad_tensor_to_size(self, tensor: torch.Tensor, size: int, *, value: int = 0):
        """将张量在第 0 维填充到指定大小（填充值默认为 0）。"""
        if value == 0:
            return torch.cat(
                [tensor, tensor.new_zeros(size - tensor.shape[0], *tensor.shape[1:])],
                dim=0,
            )
        else:
            return torch.cat(
                [
                    tensor,
                    tensor.new_full((size - tensor.shape[0], *tensor.shape[1:]), value),
                ],
                dim=0,
            )

    def prepare_mlp_sync_batch(self, model_runner: ModelRunner):
        """为 DP 注意力下的 MLP 同步准备批次：对齐各 rank 的 token 数、确定填充模式并填充输入。"""
        from sglang.srt.batch_overlap.two_batch_overlap import TboForwardBatchPreparer

        assert self.global_num_tokens_cpu is not None
        assert self.global_num_tokens_for_logprob_cpu is not None

        global_num_tokens = self.global_num_tokens_cpu
        sync_group_size = len(global_num_tokens)
        attn_tp_size = get_attention_tp_size()

        for i in range(sync_group_size):
            # 保证填充后长度能被 attn_tp_size 整除，因为可能需要在 attn_tp 维上做 reduce-scatter。
            # LM logprob 不做 reduce-scatter，所以无需为 logprob 调整填充长度。
            global_num_tokens[i] = ceil_align(global_num_tokens[i], attn_tp_size)

        # 保证各 rank 的 token 数相同，以便集体通信。
        attn_cp_size = get_attention_cp_size()
        for i in range(sync_group_size):
            global_num_tokens[i] = ceil_align(global_num_tokens[i], attn_cp_size)

        dp_padding_mode = DpPaddingMode.get_dp_padding_mode(
            self.is_extend_in_batch, global_num_tokens
        )
        self.dp_padding_mode = dp_padding_mode

        if dp_padding_mode.is_max_len():
            # 当 DP gather 模式为 all-gather 时，会用 all_gather_into_tensor 收集隐藏状态，
            # 传输的 token 需填充到相同长度；MLP 后也会用 reduce-scatter 而非 all-reduce。
            max_num_tokens = max(global_num_tokens)
            global_num_tokens = [max_num_tokens] * sync_group_size
            buffer_len = max_num_tokens * sync_group_size
        else:
            buffer_len = sum(global_num_tokens)

        if len(global_num_tokens) > 1:
            num_tokens = global_num_tokens[get_attention_dp_rank()]
        else:
            num_tokens = global_num_tokens[0]

        self.global_dp_buffer_len = buffer_len
        set_dp_buffer_len(
            buffer_len, num_tokens, dp_padding_mode.is_max_len(), global_num_tokens
        )
        set_is_extend_in_batch(self.is_extend_in_batch)

        bs = self.batch_size

        if (
            self.forward_mode.is_decode()
            or self.forward_mode.is_target_verify()
            or self.forward_mode.is_draft_extend(include_v2=True)
            or self.forward_mode.is_idle()
        ):
            if self.is_extend_in_batch and dp_padding_mode.is_max_len():
                setattr(self, "_original_forward_mode", self.forward_mode)
                self.forward_mode = ForwardMode.EXTEND
                self.extend_num_tokens = bs
                self.extend_seq_lens = torch.full_like(self.seq_lens, 1)
                self.extend_prefix_lens = self.seq_lens - 1
                self.extend_start_loc = torch.arange(
                    bs, dtype=torch.int32, device=self.seq_lens.device
                )
                self.extend_prefix_lens_cpu = self.extend_prefix_lens.cpu()
                self.extend_seq_lens_cpu = self.extend_seq_lens.cpu()
                self.extend_logprob_start_lens_cpu = self.extend_prefix_lens_cpu
            else:
                setattr(self, "_original_batch_size", self.batch_size)
                if self.spec_info is not None:
                    bs = self.batch_size = (
                        num_tokens // self.spec_info.num_tokens_per_req
                    )
                else:
                    bs = self.batch_size = num_tokens
        elif self.forward_mode.is_extend():
            self.extend_num_tokens = num_tokens

        # padding
        self._pad_inputs_to_size(model_runner, num_tokens, bs)
        self.global_num_tokens_cpu = global_num_tokens
        global_num_tokens_pinned = torch.tensor(global_num_tokens, pin_memory=True)
        self.global_num_tokens_gpu.copy_(global_num_tokens_pinned, non_blocking=True)

        TboForwardBatchPreparer.prepare(
            batch=self, is_draft_worker=model_runner.is_draft_worker
        )
        # TODO: The following is added to make sure sub-batch input_ids are padded
        # to the multiple of attn_tp_size. It can likely be removed after this
        # function is refactored and merged into the Scheduler.
        if self.tbo_children:
            for child in self.tbo_children:
                child._pad_inputs_to_size(
                    model_runner, child.tbo_padded_len, child.batch_size
                )

    def _pad_inputs_to_size(self, model_runner: ModelRunner, num_tokens, bs):
        """将各输入张量填充到目标 token 数/批次大小（供 CUDA Graph 与 DP 同步使用）。"""
        # 填充
        self.input_ids = self._pad_tensor_to_size(self.input_ids, num_tokens)
        self.req_pool_indices = self._pad_tensor_to_size(self.req_pool_indices, bs)
        self.lora_ids.extend((bs - len(self.lora_ids)) * [None])

        seq_len_fill_value = (
            model_runner.attn_backend.get_cuda_graph_seq_len_fill_value()
        )
        self.seq_lens_sum = self.seq_lens_sum + seq_len_fill_value * (
            bs - self.seq_lens.shape[0]
        )
        self.seq_lens = self._pad_tensor_to_size(
            self.seq_lens, bs, value=seq_len_fill_value
        )
        if self.seq_lens_cpu is not None:
            self.seq_lens_cpu = self._pad_tensor_to_size(
                self.seq_lens_cpu, bs, value=seq_len_fill_value
            )

        self.out_cache_loc = self._pad_tensor_to_size(self.out_cache_loc, num_tokens)
        if self.out_cache_loc_swa is not None:
            self.out_cache_loc_swa = self._pad_tensor_to_size(
                self.out_cache_loc_swa, num_tokens
            )
        if self.encoder_lens is not None:
            self.encoder_lens = self._pad_tensor_to_size(self.encoder_lens, bs)
        self.positions = self._pad_tensor_to_size(self.positions, num_tokens)
        if self.mamba_track_indices is not None:
            self.mamba_track_indices = self._pad_tensor_to_size(
                self.mamba_track_indices, bs
            )
        if self.mamba_track_mask is not None:
            self.mamba_track_mask = self._pad_tensor_to_size(self.mamba_track_mask, bs)
        if self.mamba_track_seqlens is not None:
            self.mamba_track_seqlens = self._pad_tensor_to_size(
                self.mamba_track_seqlens, bs
            )

        if self.mrope_positions is not None:
            self.mrope_positions = torch.cat(
                [
                    self.mrope_positions,
                    self.mrope_positions.new_zeros(
                        3, num_tokens - self.mrope_positions.shape[1]
                    ),
                ],
                dim=1,
            )

        # TODO: check if we need to pad other tensors
        if self.extend_seq_lens is not None:
            self.extend_seq_lens = self._pad_tensor_to_size(self.extend_seq_lens, bs)

        if self.spec_info is not None and self.spec_info.is_draft_input():
            # 投机解码：同步填充草稿信息（topk、accept_length、隐藏状态等）
            # FIXME(lsyin): 移除这个 isinstance 逻辑
            spec_info = self.spec_info
            self.output_cache_loc_backup = self.out_cache_loc
            self.hidden_states_backup = spec_info.hidden_states
            if spec_info.topk_p is not None:
                spec_info.topk_p = self._pad_tensor_to_size(spec_info.topk_p, bs)
            if spec_info.topk_index is not None:
                spec_info.topk_index = self._pad_tensor_to_size(
                    spec_info.topk_index, bs
                )
            if spec_info.accept_length is not None:
                spec_info.accept_length = self._pad_tensor_to_size(
                    spec_info.accept_length, bs
                )
            spec_info.hidden_states = self._pad_tensor_to_size(
                spec_info.hidden_states, num_tokens
            )

    def prepare_attn_tp_scatter_input(self, model_runner: ModelRunner):
        """在注意力-TP scatter 输入模式下，将 token 填充到 rank_size 的整数倍。"""
        from sglang.srt.layers.communicator import get_attn_tp_context

        attn_tp_context = get_attn_tp_context()
        input_scattered = attn_tp_context.use_input_scattered(self)
        if not input_scattered:
            return
        assert self.forward_mode.is_extend()
        tokens = self.input_ids.shape[0]
        rank_size = get_tensor_model_parallel_world_size()
        tokens_padded = (tokens + rank_size - 1) // rank_size * rank_size
        self._pad_inputs_to_size(model_runner, tokens_padded, self.batch_size)

    def post_forward_mlp_sync_batch(self, logits_output: LogitsProcessorOutput):
        """MLP 同步的后处理：恢复原始前向模式/批次大小，并按模式裁剪 logits/隐藏状态去除填充部分。"""
        self.forward_mode = getattr(self, "_original_forward_mode", self.forward_mode)
        self.batch_size = getattr(self, "_original_batch_size", self.batch_size)
        bs = self.batch_size

        if self.spec_info is not None:
            if self.forward_mode.is_decode():  # draft
                num_tokens = self.hidden_states_backup.shape[0]
                self.positions = self.positions[:num_tokens]
                self.seq_lens = self.seq_lens[:bs]
                self.req_pool_indices = self.req_pool_indices[:bs]
                if self.seq_lens_cpu is not None:
                    self.seq_lens_cpu = self.seq_lens_cpu[:bs]
                logits_output.next_token_logits = logits_output.next_token_logits[
                    :num_tokens
                ]
                logits_output.hidden_states = logits_output.hidden_states[:num_tokens]
            elif self.forward_mode.is_target_verify():  # verify
                num_tokens = bs * self.spec_info.draft_token_num
                logits_output.next_token_logits = logits_output.next_token_logits[
                    :num_tokens
                ]
                logits_output.hidden_states = logits_output.hidden_states[:num_tokens]
            elif self.forward_mode.is_draft_extend():  # draft extend
                self.spec_info.accept_length = self.spec_info.accept_length[:bs]
                logits_output.next_token_logits = logits_output.next_token_logits[:bs]
                logits_output.hidden_states = logits_output.hidden_states[:bs]
            elif self.forward_mode.is_draft_extend_v2():  # draft extend_v2
                bs = bs * self.spec_info.num_tokens_per_req
                logits_output.next_token_logits = logits_output.next_token_logits[:bs]
                logits_output.hidden_states = logits_output.hidden_states[:bs]
            elif self.forward_mode.is_extend() or self.forward_mode.is_idle():
                logits_output.next_token_logits = logits_output.next_token_logits[:bs]
                logits_output.hidden_states = logits_output.hidden_states[:bs]

            if hasattr(self, "hidden_states_backup"):
                self.spec_info.hidden_states = self.hidden_states_backup
            if hasattr(self, "output_cache_loc_backup"):
                self.out_cache_loc = self.output_cache_loc_backup

        elif self.forward_mode.is_decode() or self.forward_mode.is_idle():
            logits_output.next_token_logits = logits_output.next_token_logits[:bs]
            if logits_output.hidden_states is not None:
                logits_output.hidden_states = logits_output.hidden_states[:bs]
        elif self.forward_mode.is_extend():
            num_tokens = self.seq_lens_sum
            logits_output.next_token_logits = logits_output.next_token_logits[
                :num_tokens
            ]
            if logits_output.hidden_states is not None:
                logits_output.hidden_states = logits_output.hidden_states[:num_tokens]

    @property
    def can_run_tbo(self):
        """是否可运行双批重叠（存在拆分点时为真）。"""
        return self.tbo_split_seq_index is not None


def enable_num_token_non_padded(server_args):
    """是否需跟踪非填充 token 数（仅当 MoE 专家并行大小>1 时）。"""
    return get_moe_expert_parallel_world_size() > 1


class PPProxyTensors:
    """流水线并行（PP）中跨 rank 传递中间张量的代理容器。"""

    # 改编自 https://github.com/vllm-project/vllm/blob/d14e98d924724b284dc5eaf8070d935e214e50c0/vllm/sequence.py#L1103
    tensors: Dict[str, torch.Tensor]

    def __init__(self, tensors):
        # 手动定义此函数，使 Dynamo 知道 `IntermediateTensors()` 来自本文件；
        # 否则 dataclass 会通过求值字符串生成此函数，从而丢失源文件信息。
        self.tensors = tensors

    def __getitem__(self, key: Union[str, slice]):
        if isinstance(key, str):
            return self.tensors[key]
        elif isinstance(key, slice):
            return self.__class__({k: v[key] for k, v in self.tensors.items()})

    def __setitem__(self, key: str, value: torch.Tensor):
        self.tensors[key] = value

    def __len__(self):
        return len(self.tensors)

    def __eq__(self, other: object):
        return isinstance(other, self.__class__) and self

    def __repr__(self) -> str:
        return f"PPProxyTensors(tensors={self.tensors})"


def compute_position(
    attn_backend: str,
    extend_prefix_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_seq_lens_sum: int,
):
    """计算 extend 的位置与起始偏移：支持 triton 时走 triton 融合 kernel，否则走 torch 实现。"""
    if support_triton(attn_backend):
        positions, extend_start_loc = compute_position_triton(
            extend_prefix_lens,
            extend_seq_lens,
            extend_seq_lens_sum,
        )
    else:
        positions, extend_start_loc = compute_position_torch(
            extend_prefix_lens, extend_seq_lens
        )
    return positions, extend_start_loc


def compute_position_triton(
    extend_prefix_lens: torch.Tensor, extend_seq_lens: torch.Tensor, extend_seq_lens_sum
):
    """Compute positions. It is a fused version of `compute_position_torch`."""
    batch_size = extend_seq_lens.shape[0]
    has_prefix = extend_prefix_lens.shape[0] == batch_size

    positions = torch.empty(
        extend_seq_lens_sum, dtype=torch.int64, device=extend_seq_lens.device
    )
    extend_start_loc = torch.empty(
        batch_size, dtype=torch.int32, device=extend_seq_lens.device
    )

    # Launch kernel
    compute_position_kernel[(batch_size,)](
        positions,
        extend_start_loc,
        extend_prefix_lens,
        extend_seq_lens,
        has_prefix,
    )

    return positions, extend_start_loc


@triton.jit
def compute_position_kernel(
    positions,
    extend_start_loc,
    extend_prefix_lens,
    extend_seq_lens,
    has_prefix: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0).to(tl.int64)

    prefix_len = tl.load(extend_prefix_lens + pid) if has_prefix else 0
    seq_len = tl.load(extend_seq_lens + pid)

    # NOTE: This can be slow for large bs
    cumsum_start = tl.cast(0, tl.int64)
    for i in range(pid):
        cumsum_start += tl.load(extend_seq_lens + i)

    num_loop = tl.cdiv(seq_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        tl.store(
            positions + cumsum_start + offset,
            prefix_len + offset,
            mask=offset < seq_len,
        )
    tl.store(extend_start_loc + pid, cumsum_start)


def compute_position_torch(
    extend_prefix_lens: torch.Tensor, extend_seq_lens: torch.Tensor
):
    """纯 torch 实现：拼接各请求的位置区间并计算各请求起始偏移。"""
    positions = torch.cat(
        [
            torch.arange(
                prefix_len, prefix_len + extend_len, device=extend_prefix_lens.device
            )
            for prefix_len, extend_len in zip(extend_prefix_lens, extend_seq_lens)
        ],
        axis=0,
    )
    extend_start_loc = torch.zeros_like(extend_seq_lens)
    extend_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0)
    return positions.to(torch.int64), extend_start_loc


def _clamp_position_native(seq_lens):
    """原生实现：返回各序列“当前长度-1”作为解码位置（下限 0）。"""
    return torch.clamp((seq_lens - 1), min=0).to(torch.int64)


if is_cuda() or is_hip():
    from sglang.jit_kernel.clamp_position import clamp_position_cuda

    clamp_position = clamp_position_cuda
else:
    clamp_position = _clamp_position_native
