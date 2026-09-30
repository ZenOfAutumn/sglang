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
"""Logits 处理模块。

本模块负责把模型主干输出的隐藏状态（hidden_states）经过语言模型头（lm_head）
投影成词表维度的 logits，并进一步在需要时计算各类对数概率（logprob）。核心内容：
  - LogitsProcessorOutput：前向产出的结果容器（logits / hidden_states / 各类 logprob）。
  - LogitsMetadata：驱动 logits 处理所需的元数据（前向模式、是否返回 logprob、
    DP attention 相关信息等），可由 ForwardBatch 转换而来。
  - LogitsProcessor：实际执行 LM head 计算、张量并行 all-gather、DP attention
    gather/scatter、softcap、以及（可选的）输入 logprob 计算的 nn.Module。

关键概念：
  - prefill / extend 阶段一次处理多个 token，需要按序列裁剪出「要采样的最后一个 token」
    与「需要 input logprob 的 token」两类位置；decode 阶段每个序列只有一个 token。
  - 当词表很大且开启分块时，输入 logprob 会按行分块计算以降低显存峰值。
"""

import dataclasses
import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from sglang.kernels.ops.activation.softcap import (
    softcap_inplace_logits as fused_softcap,
)
from sglang.srt.beam_search.logits_capture import BeamLogitsCapture
from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators import triton_symm_mem_ag
from sglang.srt.environ import envs
from sglang.srt.layers.aux_hidden_states import (
    AuxHiddenStates,
    pack_aux_hidden_states,
)
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    attn_tp_all_gather,
    attn_tp_all_gather_into_tensor,
    dp_gather_replicate,
    dp_scatter,
    get_dp_device,
    get_dp_dtype,
    get_dp_hidden_size,
)
from sglang.srt.layers.logprob_processor import (
    InputLogprobProcessor,
    LogprobStage,
    get_token_ids_logprobs_raw,
    get_top_logprobs_raw,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.sampling.sampling_observer import DeviceAuxiliaryOutput
from sglang.srt.utils.common import (
    is_cpu,
    is_npu,
    is_pin_memory_available,
    use_intel_amx_backend,
)

logger = logging.getLogger(__name__)

_is_npu = is_npu()
_is_cpu = is_cpu()

# 中译：该标志置位时，LogitsProcessor.forward 直接返回空结果，跳过 LM head 与张量并行
_UNQUANTIZED_LM_HEAD_METHODS = {
    "UnquantizedEmbeddingMethod",
    "UnquantizedLinearMethod",
    "PackWeightMethod",
}

# None outside a FlashInfer autotune pass; inside one, whether that pass runs the
# LM head. Not-None means the forward's output is discarded -- attention backends
# read that via get_in_autotune_dummy_run() to skip a cross-node exchange.
# Skipping the LM head skips its [batch * dp_size, vocab] all-gather, which OOMs
# under DP attention with a tight mem_fraction_static.
_autotune_run_lm_head: Optional[bool] = None


def _trace_e2e_logits(stage: str, **fields) -> None:
    if not envs.SGLANG_TRACE_LOGITS_E2E.get():
        return
    try:
        parallel = get_parallel()
        rank = f"dp={parallel.attn_dp_rank} " f"tp={parallel.tp_rank}"
    except Exception:
        rank = "rank=unknown"
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"SGLANG_TRACE_LOGITS_E2E {rank} stage={stage} {details}", flush=True)


def _has_lm_head_runtime_attrs(lm_head, attr_names: Tuple[str, ...]) -> bool:
    return all(hasattr(lm_head, attr_name) for attr_name in attr_names)


def should_apply_lm_head_quant_method(lm_head, quant_method) -> bool:
    if (
        quant_method is None
        or not hasattr(lm_head, "weight")
        or not callable(getattr(quant_method, "apply", None))
    ):
        return False

    method_name = type(quant_method).__name__
    if method_name in _UNQUANTIZED_LM_HEAD_METHODS:
        return False

    # A shared target lm_head can retain the draft's stale ModelOpt method; use it
    # only when the runtime tensor layout matches that method.
    if method_name == "ModelOptFp4LinearMethod":
        if lm_head.weight.dtype == torch.int32 and _has_lm_head_runtime_attrs(
            lm_head,
            (
                "weight_scale",
                "weight_global_scale",
                "workspace",
                "input_size_per_partition",
                "output_size_per_partition",
            ),
        ):
            return True
        return lm_head.weight.dtype == torch.uint8 and _has_lm_head_runtime_attrs(
            lm_head,
            (
                "weight_scale_interleaved",
                "alpha",
                "input_scale_inv",
                "input_size_per_partition",
                "output_size_per_partition",
            ),
        )
    if method_name == "ModelOptNvFp4A16LinearMethod":
        return lm_head.weight.dtype == torch.int32 and _has_lm_head_runtime_attrs(
            lm_head,
            (
                "weight_scale",
                "weight_global_scale",
                "workspace",
                "input_size_per_partition",
                "output_size_per_partition",
            ),
        )
    if method_name == "ModelOptFp8LinearMethod":
        return (
            lm_head.weight.dtype == torch.float8_e4m3fn
            and _has_lm_head_runtime_attrs(lm_head, ("weight_scale", "input_scale"))
        )

    return True


# FlashInfer autotune skips the unprofiled LM-head all-gather; its
# [batch * dp_size, vocab] output can OOM under tight DP-attention memory.
_in_autotune_dummy_run = False


def get_in_autotune_dummy_run() -> bool:
    # 中译：返回当前是否处于「autotune 空跑」模式。
    # 中译：上下文管理器——进入时置位「autotune 空跑」标志，退出时恢复，
    return _autotune_run_lm_head is not None


@contextmanager
def autotune_dummy_run_mode(*, run_lm_head: bool):
    global _autotune_run_lm_head
    _autotune_run_lm_head = run_lm_head
    try:
        yield
    finally:
        _autotune_run_lm_head = None


@dataclasses.dataclass
class LogitsProcessorOutput:
    """模型前向产出的「logits 处理器输出」容器，是 LogitsProcessor → Sampler → 调度器
    这条链路上传递的核心结果对象。

    按字段的赋值来源/用途分为 5 个部分（Part 1~5）：
      - Part 1：由 LogitsProcessor 填充——下一个 token 的 logits 与（投机解码用的）隐藏状态。
      - Part 2：由 Sampler 填充——输出 token 的各类 logprob（含 top-k、指定 token id）。
      - Part 3：仅 prefill 阶段——输入 token 的各类 logprob。
      - Part 4：仅扩散式 LLM（Diffusion LLM）使用的完整 logits。
      - Part 5：自定义附加信息与多模态输入嵌入。

    说明：很多字段是可选的，仅在请求开启了对应能力（如 return_logprob、top_logprobs_num、
    指定 token_ids_logprob、投机解码、扩散式 LLM 等）时才被填充，否则为 None。
    部分 logprob 字段可能直接持有 GPU 张量（延迟拷回 CPU 的优化），而非已转好的 list。
    """

    ## Part 1: This part will be assigned in python/sglang/srt/layers/logits_processor.py::LogitsProcessor
    # 中译：Part 1——由 LogitsProcessor 填充。
    # The logits of the next tokens.       shape: [#seq, vocab_size]
    # Can be None for certain prefill-only requests (e.g., multi-item scoring) that don't need next token generation
    # 中译：下一个 token 的 logits，形状 [序列数, 词表大小]。对某些只做 prefill、不需要生成下一个
    #       token 的请求（如 multi-item 打分），该值可能为 None。
    next_token_logits: Optional[torch.Tensor]
    # Used by speculative decoding (EAGLE)
    # The last hidden layers
    # shape: [#seq, hidden_dim] when capture_hidden_mode is LAST (only last-token
    #        states are kept), or [#token, hidden_dim] when capture_hidden_mode is FULL.
    #        When aux_hidden_states are used (multi-layer EAGLE), the last dim becomes
    #        hidden_dim * num_aux_layers (layers concatenated along dim=-1).
    # 中译：最后一层的隐藏状态，供投机解码（EAGLE）等使用；未启用时为 None。
    #       形状：capture_hidden_mode 为 LAST 时为 [序列数, hidden_dim]（仅保留每个序列最后一个
    #       token 的隐藏状态）；为 FULL 时为 [token 数, hidden_dim]。当使用 aux_hidden_states
    #       （多层 EAGLE）时，最后一维变为 hidden_dim * 辅助层数（各层沿 dim=-1 拼接）。
    hidden_states: Optional[torch.Tensor] = None

    ## Part 2: This part will be assigned in python/sglang/srt/layers/sampler.py::Sampler
    # 中译：Part 2——由 Sampler 填充（输出位置的各类 logprob）。
    # he log probs of output tokens, if SGLANG_RETURN_ORIGINAL_LOGPROB = True, will get the log probs before applying temperature. If False, will get the log probs before applying temperature.
    # 中译：输出 token 的对数概率。是否取「应用温度前」的 logprob 由 SGLANG_RETURN_ORIGINAL_LOGPROB 控制。
    next_token_logprobs: Optional[torch.Tensor] = None
    # The logprobs and ids of the top-k tokens in output positions. shape: [#seq, k]
    # 中译：输出位置 top-k token 的 logprob 值与对应 token id，形状 [序列数, k]。
    next_token_top_logprobs_val: Optional[List] = None
    next_token_top_logprobs_idx: Optional[List] = None
    # The logprobs and ids of the requested token ids in output positions. shape: [#seq, n] (n is the number of requested token ids)
    # Can contain either lists or GPU tensors (for delayed copy optimization in prefill-only requests)
    # 中译：输出位置上「调用方指定的那批 token id」的 logprob 值与 id，形状 [序列数, n]（n 为指定的
    #       token id 数）。可能是 list，也可能直接是 GPU 张量（prefill-only 请求的延迟拷贝优化）。
    next_token_token_ids_logprobs_val: Optional[
        List[Union[List[float], torch.Tensor]]
    ] = None
    next_token_token_ids_logprobs_idx: Optional[List] = None
    # Sparse top-k/top-p/min-p support ids and selected-token logprob after
    # truncation/renormalization. Only populated when requested.
    next_token_sampling_mask_idx: Optional[List[Optional[List[int]]]] = None
    next_token_sampling_logprobs: Optional[List[Optional[float]]] = None

    ## Part 3: Prefill-only. This part will be assigned in python/sglang/srt/layers/logits_processor.py::LogitsProcessor
    # 中译：Part 3——仅 prefill 阶段使用，由 LogitsProcessor 填充（输入位置的各类 logprob）。
    # The logprobs of input tokens.        shape: [#token]
    # 中译：输入 token 的对数概率，形状 [token 数]。
    input_token_logprobs: Optional[torch.Tensor] = None
    # The logprobs and ids of the top-k tokens in input positions.  shape: [#seq, #token, k]
    # 中译：输入位置 top-k token 的 logprob 值与 id，形状 [序列数, token 数, k]。
    input_top_logprobs_val: Optional[List] = None
    input_top_logprobs_idx: Optional[List] = None
    # The logprobs and ids of the requested token ids in input positions. shape: [#seq, n] (n is the number of requested token ids)
    # Can contain either lists or GPU tensors (for delayed GPU-to-CPU transfer optimization)
    # 中译：输入位置上「调用方指定的那批 token id」的 logprob 值与 id，形状 [序列数, n]。
    #       同样可能是 list 或 GPU 张量（延迟 GPU→CPU 拷贝优化）。
    input_token_ids_logprobs_val: Optional[List[Union[List[float], torch.Tensor]]] = (
        None
    )
    input_token_ids_logprobs_idx: Optional[List] = None

    ## Part 4: Diffusion LLM only.
    # 中译：Part 4——仅扩散式 LLM（Diffusion LLM）使用的完整 logits。
    full_logits: Optional[torch.Tensor] = None

    # Beam search only: raw pre-sample logits for the scheduler-side joint
    # selection; see beam_search.logits_capture.
    beam: Optional[BeamLogitsCapture] = None

    ## Part 5: Customized Info
    # 中译：Part 5——自定义附加信息（按 key 对应逐请求的值列表）。
    customized_info: Optional[Dict[str, List[Any]]] = None

    # 中译：多模态输入嵌入（多模态场景下回传的输入 embedding）。
    ## Part 6: Temporary variables
    # FIXME: These fields are not logits-related but are passed through here as a
    # workaround since ForwardBatch is local to forward_batch_generation().
    # They should be moved to GenerationBatchResult to keep this class clean.
    mm_input_embeds: Optional[torch.Tensor] = None

    # Scheduler-local output copied alongside the ordinary generation result.
    auxiliary_device_output: Optional[DeviceAuxiliaryOutput] = None


@dataclasses.dataclass
class LogitsMetadata:
    """驱动 logits 处理所需的元数据。

    它通常由 from_forward_batch 从 ForwardBatch 抽取生成，集中携带本次前向
    「该如何处理 logits / logprob」所需的全部信息，避免把庞大的 ForwardBatch
    一路透传到 LogitsProcessor 内部。主要包含三类信息：
      1. 前向模式与隐藏状态捕获模式（forward_mode / capture_hidden_mode）；
      2. 输入 logprob 相关的开关与各序列的长度/起始位置（extend_* 系列字段）；
      3. DP attention（数据并行注意力）下做 gather/scatter 所需的 token 分布信息。
    """

    # 中译：前向模式（prefill/extend、decode、target_verify、draft_extend 等）。
    forward_mode: ForwardMode
    # 中译：隐藏状态捕获模式——决定是否、以及如何保存 hidden_states（供投机解码使用）。
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL
    # 中译：可复用的 next_token_logits 输出缓冲区（若提供则原地写入，省一次分配）。
    next_token_logits_buffer: Optional[torch.Tensor] = None

    # 中译：以下 extend_* 为 prefill/extend 阶段计算输入 logprob 的相关开关与长度信息。
    extend_return_logprob: bool = False  # 是否返回输入 token 的 logprob
    extend_return_top_logprob: bool = False  # 是否返回输入位置的 top-k logprob
    extend_token_ids_logprob: bool = False  # 是否返回输入位置上指定 token id 的 logprob
    extend_seq_lens: Optional[torch.Tensor] = None  # 各序列 extend 的 token 数（GPU 张量）
    extend_seq_lens_cpu: Optional[List[int]] = None  # 同上，CPU 侧列表
    extend_logprob_start_lens_cpu: Optional[List[int]] = None  # 各序列从第几个 token 开始算 logprob
    extend_logprob_pruned_lens_cpu: Optional[List[int]] = None  # 各序列裁剪后参与 logprob 的 token 数
    top_logprobs_nums: Optional[List[int]] = None  # 各序列请求的 top-k 数量
    extend_input_logprob_token_ids_gpu: Optional[torch.Tensor] = None  # 各位置「目标 token id」（用于取其 logprob）
    token_ids_logprobs: Optional[List[List[int]]] = None  # 各序列请求 logprob 的指定 token id 列表

    # logits and logprobs post processing
    # 中译：logits / logprob 后处理参数。
    temperature: torch.Tensor = None  # 温度
    top_p: torch.Tensor = None  # top-p（核采样阈值）

    # DP attention metadata. Not needed when DP attention is not used.
    # 中译：DP attention（数据并行注意力）相关元数据；不启用 DP attention 时无需填写。
    # Number of tokens in the request.
    # 中译：本次请求（全局）的 token 数。
    global_num_tokens_gpu: Optional[torch.Tensor] = None
    # The start position of local hidden states.
    # 中译：本 DP rank 局部隐藏状态在全局缓冲区中的起始位置。
    dp_local_start_pos: Optional[torch.Tensor] = None
    dp_local_num_tokens: Optional[torch.Tensor] = None  # 本 DP rank 局部 token 数
    global_dp_buffer_len: Optional[int] = None  # 全局 DP gather 缓冲区长度
    # Number of tokens to sample per DP rank
    # 中译：每个 DP rank 需要计算 logprob 的 token 数（CPU / GPU 两份）。
    global_num_tokens_for_logprob_cpu: Optional[torch.Tensor] = None
    global_num_tokens_for_logprob_gpu: Optional[torch.Tensor] = None
    # The gather mode for DP attention
    # 中译：DP attention 的 gather 模式（如按 token 数求和对齐）。
    dp_padding_mode: Optional[DpPaddingMode] = None
    # 中译：用于 padding 的静态长度；<0 表示未启用静态 padding。

    # Whether this batch is prefill-only (no token generation needed)
    # 中译：本批是否为「仅 prefill」（不需要生成下一个 token，如打分/嵌入类请求）。
    is_prefill_only: bool = False

    # 中译：多模态输入嵌入（透传字段）。
    mm_input_embeds: Optional[torch.Tensor] = None

    # DRAFT_EXTEND_V2: when set, lm_head runs only on these rows (see
    # EagleDraftExtendInput.select_index).
    draft_extend_select_index: Optional[torch.Tensor] = None

    @classmethod
    def from_forward_batch(cls, forward_batch: ForwardBatch):
        # 中译：从 ForwardBatch 抽取构造 LogitsMetadata。
        #       仅当处于 extend 模式、请求要求返回 logprob、且不是投机解码的 target_verify
        #       时，才真正解析输入 logprob 相关字段；否则把这些开关全部置为 False/空。
        if (
            forward_batch.forward_mode.is_extend()
            and forward_batch.return_logprob
            and not forward_batch.forward_mode.is_target_verify()
        ):
            extend_return_top_logprob = any(
                x > 0 for x in forward_batch.top_logprobs_nums
            )
            extend_token_ids_logprob = any(
                x is not None for x in forward_batch.token_ids_logprobs
            )
            extend_return_logprob = False
            extend_logprob_pruned_lens_cpu = []
            # 中译：逐序列计算「裁剪后参与 input logprob 的 token 数」= extend_len - start_len；
            #       只要有任一序列该值 > 0，就说明本批确实需要返回输入 logprob。
            for extend_len, start_len in zip(
                forward_batch.extend_seq_lens_cpu,
                forward_batch.extend_logprob_start_lens_cpu,
            ):
                if extend_len - start_len > 0:
                    extend_return_logprob = True
                extend_logprob_pruned_lens_cpu.append(extend_len - start_len)
        else:
            extend_return_logprob = extend_return_top_logprob = (
                extend_token_ids_logprob
            ) = extend_logprob_pruned_lens_cpu = False

        if forward_batch.forward_mode.is_draft_extend_v2():
            draft_extend_select_index = forward_batch.spec_info.select_index
        else:
            draft_extend_select_index = None

        return cls(
            forward_mode=forward_batch.forward_mode,
            capture_hidden_mode=forward_batch.capture_hidden_mode,
            next_token_logits_buffer=forward_batch.next_token_logits_buffer,
            extend_return_logprob=extend_return_logprob,
            extend_return_top_logprob=extend_return_top_logprob,
            extend_token_ids_logprob=extend_token_ids_logprob,
            extend_seq_lens=forward_batch.extend_seq_lens,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            extend_logprob_start_lens_cpu=forward_batch.extend_logprob_start_lens_cpu,
            extend_logprob_pruned_lens_cpu=extend_logprob_pruned_lens_cpu,
            top_logprobs_nums=forward_batch.top_logprobs_nums,
            token_ids_logprobs=forward_batch.token_ids_logprobs,
            extend_input_logprob_token_ids_gpu=forward_batch.extend_input_logprob_token_ids_gpu,
            is_prefill_only=forward_batch.is_prefill_only,
            global_num_tokens_gpu=forward_batch.global_num_tokens_gpu,
            dp_local_start_pos=forward_batch.dp_local_start_pos,
            dp_local_num_tokens=forward_batch.dp_local_num_tokens,
            global_dp_buffer_len=forward_batch.global_dp_buffer_len,
            global_num_tokens_for_logprob_cpu=forward_batch.global_num_tokens_for_logprob_cpu,
            global_num_tokens_for_logprob_gpu=forward_batch.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.SUM_LEN,
            mm_input_embeds=forward_batch.mm_input_embeds,
            draft_extend_select_index=draft_extend_select_index,
        )

    def compute_dp_attention_metadata(self):
        # 中译：计算 DP attention 下本 rank 的局部起止位置，并预分配用于 all-gather 的全局缓冲区。
        #       通过对各 rank 的 token 数做前缀和，得到本 rank 在全局张量中的起始偏移与长度。
        cumtokens = torch.cumsum(self.global_num_tokens_for_logprob_gpu, dim=0)
        dp_rank = get_parallel().attn_dp_rank
        if dp_rank == 0:
            dp_local_start_pos = torch.zeros_like(
                self.global_num_tokens_for_logprob_gpu[0]
            )
        else:
            dp_local_start_pos = cumtokens[dp_rank - 1]

        self.dp_local_start_pos = dp_local_start_pos
        self.dp_local_num_tokens = self.global_num_tokens_for_logprob_gpu[dp_rank]

        hidden_size = get_dp_hidden_size()
        dtype = get_dp_dtype()
        device = get_dp_device()

        if self.global_num_tokens_for_logprob_cpu is not None:
            # create a smaller buffer to reduce peak memory usage
            # 中译：用各 rank token 数之和作为缓冲区长度，得到尽量小的缓冲区以降低显存峰值。
            self.global_dp_buffer_len = sum(self.global_num_tokens_for_logprob_cpu)
        else:
            self.global_dp_buffer_len = self.global_dp_buffer_len

        self.gathered_buffer = torch.empty(
            (
                self.global_dp_buffer_len,
                hidden_size,
            ),
            dtype=dtype,
            device=device,
        )


class LogitsProcessor(nn.Module):
    """负责「hidden_states → logits → (可选) logprob」的核心模块。

    主要职责：
      1. 根据前向模式从 hidden_states 中裁剪出「需要采样的位置」与「需要输入 logprob 的位置」；
      2. 调用 lm_head 计算 logits，并处理张量并行 all-gather、DP attention 的 gather/scatter、
         logit 缩放与 softcap 等后处理；
      3. 若请求要求，计算输入 token / top-k / 指定 token id 的 logprob（可分块以省显存）。

    构造参数：
      - config：模型配置（提供 vocab_size、final_logit_softcapping 等）。
      - skip_all_gather：是否跳过张量并行 all-gather。
      - logit_scale：可选的 logit 缩放系数。
      - return_full_logits：是否返回完整 logits（扩散式 LLM 等场景）。
    """

    def __init__(
        self,
        config,
        skip_all_gather: bool = False,
        logit_scale: Optional[float] = None,
        return_full_logits: bool = False,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.logit_scale = logit_scale
        # 中译：是否在 attention TP 组内做 LM head 并行（enable_dp_lm_head）；
        # 中译：根据是否使用 attention TP 组，决定 all-gather 的方式与是否需要叠加 DP attention 路径。
        self.use_attn_tp_group = get_parallel().enable_dp_lm_head
        self.use_tp_lm_head_all_to_all = get_parallel().enable_tp_lm_head_all_to_all
        self.use_fp32_lm_head = get_exec().features.enable_fp32_lm_head
        if self.use_attn_tp_group:
            self.attn_tp_size = get_parallel().attn_tp_size
            self.do_tensor_parallel_all_gather = (
                not skip_all_gather and self.attn_tp_size > 1
            )
            self.do_tensor_parallel_all_gather_dp_attn = False
        else:
            self.do_tensor_parallel_all_gather = (
                not skip_all_gather and get_parallel().tp_size > 1
            )
            self.do_tensor_parallel_all_gather_dp_attn = (
                self.do_tensor_parallel_all_gather and get_parallel().attn_dp_size != 1
            )
        # 中译：读取模型的 final_logit_softcapping（如 Gemma 系列）；若为负值则视为未启用。
        self.final_logit_softcapping = getattr(
            self.config, "final_logit_softcapping", None
        )
        if (
            self.final_logit_softcapping is not None
            and self.final_logit_softcapping < 0
        ):
            self.final_logit_softcapping = None

        self.return_full_logits = return_full_logits
        # 中译：是否启用 multi-item scoring（多项打分）。
        # 中译：是否启用输入 logprob 的分块计算（词表大/token 多时用以降低显存峰值）。
        # 中译：分块计算 logprob 时的每块行数。
        self.enable_mis = get_exec().features.enable_mis
        self.rl_on_policy_target = get_exec().deterministic.rl_on_policy_target

        self._logits_gatherer = triton_symm_mem_ag.MultimemAllGatherer(
            max_tokens=triton_symm_mem_ag.recommended_max_tokens(
                include_prefill=False, floor=128
            ),
            enabled=self.do_tensor_parallel_all_gather and not self.use_attn_tp_group,
            skip_entry_sync=True,
        )

        self.input_logprob_processor = InputLogprobProcessor()

    def forward(
        self,
        input_ids,
        hidden_states,
        lm_head: VocabParallelEmbedding,
        logits_metadata: Union[LogitsMetadata, ForwardBatch],
        aux_hidden_states: Optional[AuxHiddenStates] = None,
        hidden_states_before_norm: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorOutput:
        """前向入口：把 hidden_states 转为 next_token_logits（及可选的输入 logprob）。

        参数：
          - input_ids：输入 token id（主要供 multi-item scoring 使用）。
          - hidden_states：模型主干输出的隐藏状态，形状 [#token, hidden_dim]。
          - lm_head：语言模型头（词表投影）。
          - logits_metadata：驱动元数据，可是 LogitsMetadata 或直接传 ForwardBatch（会被转换）。
          - aux_hidden_states：多层 EAGLE 用的辅助隐藏状态（可选）。
          - hidden_states_before_norm：归一化前的隐藏状态（某些投机解码场景会优先保存它）。

        处理分支概述：依次处理 autotune 空跑 → multi-item scoring → 扩散式 LLM →
        通用路径（裁剪状态 → 计算 logits → 可选的输入 logprob）。
        """
        # Extract MIS indices before ForwardBatch → LogitsMetadata conversion
        # 中译：在把 ForwardBatch 转成 LogitsMetadata 之前，先取出 multi-item scoring 的分隔符位置。
        multi_item_delimiter_indices = None
        if isinstance(logits_metadata, ForwardBatch):
            multi_item_delimiter_indices = logits_metadata.multi_item_delimiter_indices
            logits_metadata = LogitsMetadata.from_forward_batch(logits_metadata)

        # 中译：autotune 空跑会丢弃本输出（详见 _in_autotune_dummy_run）。置于 MIS / DLLM /
        # Autotune dummy run discards this output. `is False` not `not`: None
        # means no autotune pass, which must not skip. Placed before the MIS /
        # DLLM / common dispatch so all three LM-head paths are skipped.
        if _autotune_run_lm_head is False:
            return LogitsProcessorOutput(next_token_logits=None)

        # Multi-item scoring only for prefill-only requests with pre-computed indices.
        # 中译：仅对「仅 prefill 且已预计算分隔符位置」的请求走 multi-item scoring 路径。
        if multi_item_delimiter_indices is not None and logits_metadata.is_prefill_only:
            return self.compute_logprobs_for_multi_item_scoring(
                input_ids,
                hidden_states,
                lm_head,
                logits_metadata,
                multi_item_delimiter_indices,
            )

        # Diffusion LLM only.
        # 中译：仅扩散式 LLM（Diffusion LLM）的 extend 路径，返回完整 logits。
        if logits_metadata.forward_mode.is_dllm_extend():
            return self._get_dllm_logits(hidden_states, lm_head, logits_metadata)

        # Get the last hidden states and last logits for the next token prediction
        # 中译：按前向模式裁剪出用于预测下一个 token 的隐藏状态，以及用于输入 logprob 的位置索引。
        (
            pruned_states,
            pruned_states_before_norm,
            aux_pruned_states,
            sample_indices,
            input_logprob_indices,
            token_to_seq_idx,
        ) = self._get_pruned_states(
            hidden_states,
            hidden_states_before_norm,
            aux_hidden_states,
            logits_metadata,
        )

        hidden_states_to_store = self._get_hidden_states_to_store(
            hidden_states,
            hidden_states_before_norm,
            aux_hidden_states,
            pruned_states,
            pruned_states_before_norm,
            aux_pruned_states,
            sample_indices,
            logits_metadata,
        )
        # 中译：hidden_states 后续不再使用，及早释放以节省显存。
        del hidden_states

        if not logits_metadata.extend_return_logprob:
            # Compute logits for both input and sampled tokens.
            # 中译：不需要返回输入 logprob 的快路——直接算 logits，并按 sample_indices 取出采样位置。
            logits = self._get_logits(pruned_states, lm_head, logits_metadata)
            sampled_logits = (
                logits[sample_indices] if sample_indices is not None else logits
            )

            # Decode mode or extend mode without return_logprob.
            # 中译：decode 模式，或不要求返回 logprob 的 extend 模式。
            return LogitsProcessorOutput(
                next_token_logits=sampled_logits,
                hidden_states=hidden_states_to_store,
                mm_input_embeds=logits_metadata.mm_input_embeds,
            )

        # 中译：开始处理输入 logprob。决定是否走分块路径；以下任一成立则跳过分块：
            # 中译：不分块路径——一次性算出全部 logits，再分别取出采样位置与输入 logprob 位置。
            # 中译：分块路径——按块计算 logits 与输入 logprob，同时填出采样 logits，以降低显存峰值。
        logprobs_result, sampled_logits = self.input_logprob_processor.forward(
            pruned_states=pruned_states,
            sample_indices=sample_indices,
            input_logprob_indices=input_logprob_indices,
            token_to_seq_idx=token_to_seq_idx,
            lm_head=lm_head,
            get_logits_fn=self._get_logits,
            logits_metadata=logits_metadata,
            skip_chunking_for_dp_attn=self.do_tensor_parallel_all_gather_dp_attn,
        )

        logits_output = LogitsProcessorOutput(
            next_token_logits=sampled_logits,
            hidden_states=hidden_states_to_store,
            mm_input_embeds=logits_metadata.mm_input_embeds,
        )
        logprobs_result.write_input_to(logits_output)
        return logits_output

    def _get_pruned_states(
        self,
        hidden_states: torch.Tensor,
        hidden_states_before_norm: Optional[torch.Tensor],
        aux_hidden_states: Optional[AuxHiddenStates],
        logits_metadata: LogitsMetadata,
    ):
        """按前向模式从 hidden_states 裁剪出后续计算所需的子集及索引。

        返回五元组（+token_to_seq_idx）：
          - pruned_states：需要计算 logits/logprob 的隐藏状态子集；
          - pruned_states_before_norm / aux_pruned_states：对应的归一化前状态与辅助状态；
          - sample_indices：在 pruned_states 中「要采样」的位置（decode 等场景为 None）；
          - input_logprob_indices：「需要输入 logprob」的位置；
          - token_to_seq_idx：每个 token 到其所属序列索引的映射（供分块计算使用）。
        三种情形：decode/target_verify/draft_extend_v2 直接用全部；不要 logprob 的 extend 只取
        每序列最后一个 token；要 logprob 的 extend 需同时算出上述四类索引。
        """
        pruned_states_before_norm: Optional[torch.Tensor] = None
        aux_pruned_states = None
        token_to_seq_idx = []

        # 中译：decode/idle、target_verify、draft_extend_v2 模式下，每序列只有一个输出位置，
        #       直接用全部 hidden_states，无需裁剪。
        if (
            logits_metadata.forward_mode.is_decode_or_idle()
            or logits_metadata.forward_mode.is_target_verify()
            or logits_metadata.forward_mode.is_draft_extend_v2()
        ):
            if logits_metadata.draft_extend_select_index is not None:
                # Only next_token_logits narrows to [bs, vocab]; the
                # FULL-capture hidden stays unpruned.
                pruned_states = hidden_states[logits_metadata.draft_extend_select_index]
            else:
                pruned_states = hidden_states
            pruned_states_before_norm = hidden_states_before_norm
            if aux_hidden_states is not None:
                aux_pruned_states = (
                    aux_hidden_states
                    if isinstance(aux_hidden_states, torch.Tensor)
                    else [hidden for hidden in aux_hidden_states]
                )
            sample_indices = None
            input_logprob_indices = None

        elif (
            logits_metadata.forward_mode.is_extend()
            and not logits_metadata.extend_return_logprob
        ):
            # Prefill without input logprobs.
            # 中译：不要求输入 logprob 的 prefill——只需取出每个序列的最后一个 token 用于采样。
                # 中译：无静态 padding：累加各序列长度减 1，即为各序列最后一个 token 的下标。
                # 中译：有静态 padding 时：若 padded_static_len=5、extend_seq_lens=[2,3]，
            last_index = torch.cumsum(logits_metadata.extend_seq_lens, dim=0) - 1
            pruned_states = hidden_states[last_index]
            if hidden_states_before_norm is not None:
                pruned_states_before_norm = hidden_states_before_norm[last_index]
            if aux_hidden_states is not None:
                aux_pruned_states = (
                    aux_hidden_states[last_index]
                    if isinstance(aux_hidden_states, torch.Tensor)
                    else [hidden[last_index] for hidden in aux_hidden_states]
                )
            sample_indices = None
            input_logprob_indices = None
        else:
            # Prefill with input logprobs.
            # Find 4 different indices.
            # 1. pruned_states: hidden states that we want logprobs from.
            # 2. sample_indices: Indices that have sampled tokens.
            # 3. input_logprob_indices: Indices that have input logprob tokens.
            # 4. token_to_seq_idx: map each token to its sequence index
            # 中译：要求返回输入 logprob 的 prefill。需要算出四类索引：
            #       1. pruned_states：需要取 logprob 的隐藏状态；
            #       2. sample_indices：有采样 token 的位置；
            #       3. input_logprob_indices：有输入 logprob token 的位置；
            #       4. token_to_seq_idx：每个 token 到其所属序列的映射。
            #
            # Example
            # -------
            # Suppose a batch (flattened by sequence):
            # [t00, t01, t02, t03, t10, t11, t12, t13, t14, t20, t21, t22, t23, t24, t25]
            # extend_seq_lens_cpu           = [4, 5, 6]
            # extend_logprob_start_lens_cpu = [0, 5, 3]
            #
            # Then, the indices are:
            # pruned_states         -> [t00, t01, t02, t03, t14, t23, t24, t25]
            # sample_indices        -> [3, 4, 7]
            # input_logprob_indices -> [0, 1, 2, 3, 5, 6, 7]
            # token_to_seq_idx      -> [0, 0, 0, 0, 1, 2, 2, 2]
            #
            # If chunk is enabled and chunk_size = 3, the chunks will be computed in a chunked manner:
            # [t00, t01, t02], [t03, t14, t23], [t24, t25]
            # 中译：上例中批按序列拼平，三个序列长为 [4,5,6]、logprob 起始为 [0,5,3]，
            #       据此算出上述四组索引；若启用分块且 chunk_size=3，则按
            #       [t00,t01,t02]、[t03,t14,t23]、[t24,t25] 逐块计算。

            sample_index_pt = -1
            sample_indices = []
            input_logprob_indices_pt = 0
            input_logprob_indices = []
            pt, pruned_states_list, pruned_states_before_norm_list = 0, [], []
            is_packed_aux_hidden_states = isinstance(aux_hidden_states, torch.Tensor)
            aux_pruned_states_lists = None
            if aux_hidden_states is not None:
                aux_pruned_states_lists = (
                    []
                    if is_packed_aux_hidden_states
                    else [[] for _ in aux_hidden_states]
                )

            for idx, (extend_logprob_start_len, extend_len) in enumerate(
                zip(
                    logits_metadata.extend_logprob_start_lens_cpu,
                    logits_metadata.extend_seq_lens_cpu,
                )
            ):
                # It can happen in chunked prefill. We still need to sample 1 token,
                # But we don't want to include it in input logprob.
                # 中译：分块 prefill 下可能出现 extend_len == start_len：仍需采样 1 个 token，
                #       但不想把它计入输入 logprob，故起始位置回退一个。
                if extend_len == extend_logprob_start_len:
                    start_len = extend_logprob_start_len - 1
                else:
                    start_len = extend_logprob_start_len

                # We always need at least 1 token to sample because that's required
                # by a caller.
                # 中译：调用方要求至少采样 1 个 token，故 extend_len 必须严格大于 start_len。
                assert extend_len > start_len
                pruned_states_list.append(
                    hidden_states[pt + start_len : pt + extend_len]
                )
                if hidden_states_before_norm is not None:
                    pruned_states_before_norm_list.append(
                        hidden_states_before_norm[pt + start_len : pt + extend_len]
                    )
                if aux_pruned_states_lists is not None:
                    if is_packed_aux_hidden_states:
                        aux_pruned_states_lists.append(
                            aux_hidden_states[pt + start_len : pt + extend_len]
                        )
                    else:
                        for j, hidden in enumerate(aux_hidden_states):
                            aux_pruned_states_lists[j].append(
                                hidden[pt + start_len : pt + extend_len]
                            )
                # Map each token to its sequence index, for chunked computation
                # of input logprobs
                # 中译：把本序列的每个 token 映射到其序列索引 idx，供分块计算输入 logprob 使用。
                token_to_seq_idx.extend([idx] * (extend_len - start_len))
                pt += extend_len
                sample_index_pt += extend_len - start_len
                sample_indices.append(sample_index_pt)
                input_logprob_indices.extend(
                    [
                        input_logprob_indices_pt + i
                        for i in range(extend_len - extend_logprob_start_len)
                    ]
                )
                input_logprob_indices_pt += extend_len - start_len

            # 中译：补上最后一个序列的末 token 对应的序列索引（供后续切片使用）。
            pruned_states = torch.cat(pruned_states_list)
            if hidden_states_before_norm is not None:
                pruned_states_before_norm = torch.cat(pruned_states_before_norm_list)
            if aux_pruned_states_lists is not None:
                aux_pruned_states = (
                    torch.cat(aux_pruned_states_lists)
                    if is_packed_aux_hidden_states
                    else [torch.cat(lst) for lst in aux_pruned_states_lists]
                )

            # Build the index tensors via pinned host memory + non-blocking H2D
            # so the small copy doesn't drain the stream.
            # 中译：用锁页内存 + 非阻塞 H2D 拷贝构造索引张量，避免这次小拷贝阻塞 CUDA 流。
            sample_indices = torch.tensor(
                sample_indices,
                dtype=torch.int64,
                pin_memory=is_pin_memory_available(),
            ).to(pruned_states.device, non_blocking=True)
            input_logprob_indices = torch.tensor(
                input_logprob_indices,
                dtype=torch.int64,
                pin_memory=is_pin_memory_available(),
            ).to(pruned_states.device, non_blocking=True)

        return (
            pruned_states,
            pruned_states_before_norm,
            aux_pruned_states,
            sample_indices,
            input_logprob_indices,
            token_to_seq_idx,
        )

    def _get_hidden_states_to_store(
        self,
        hidden_states: torch.Tensor,
        hidden_states_before_norm: Optional[torch.Tensor],
        aux_hidden_states: Optional[AuxHiddenStates],
        pruned_states: torch.Tensor,
        pruned_states_before_norm: Optional[torch.Tensor],
        aux_pruned_states: Optional[AuxHiddenStates],
        sample_indices: Optional[torch.Tensor],
        logits_metadata: LogitsMetadata,
    ) -> Optional[torch.Tensor]:
        """根据 capture_hidden_mode 决定要保存（回传）哪些隐藏状态，供投机解码（EAGLE）使用。

        - need_capture() 为 False：不保存，返回 None。
        - is_full()：保存全部 token 的隐藏状态（有辅助状态时沿最后一维拼接）。
        - is_last()：只保存每个序列最后一个 token 的隐藏状态。
        若提供了归一化前状态（hidden_states_before_norm），则优先返回它。
        """
        hidden_states_to_store: Optional[torch.Tensor] = None
        hidden_states_to_store_before_norm: Optional[torch.Tensor] = None
        if logits_metadata.capture_hidden_mode.need_capture():
            if logits_metadata.capture_hidden_mode.is_full():
                if aux_hidden_states is not None:
                    hidden_states_to_store = pack_aux_hidden_states(aux_hidden_states)
                else:
                    hidden_states_to_store = hidden_states
                hidden_states_to_store_before_norm = hidden_states_before_norm
            elif logits_metadata.capture_hidden_mode.is_last():
                # Get the last token hidden states. If sample_indices is None,
                # pruned states only contain the last tokens already.
                # 中译：取最后一个 token 的隐藏状态。若 sample_indices 为 None，
                #       说明 pruned_states 本身已只含最后一个 token。
                if aux_hidden_states is not None:
                    assert aux_pruned_states is not None
                    aux_pruned_states = pack_aux_hidden_states(aux_pruned_states)
                    hidden_states_to_store = (
                        aux_pruned_states[sample_indices]
                        if sample_indices is not None
                        else aux_pruned_states
                    )
                else:
                    hidden_states_to_store = (
                        pruned_states[sample_indices]
                        if sample_indices is not None
                        else pruned_states
                    )
                    if hidden_states_before_norm is not None:
                        hidden_states_to_store_before_norm = (
                            pruned_states_before_norm[sample_indices]
                            if sample_indices is not None
                            else pruned_states_before_norm
                        )
            else:
                assert False, "Should never reach"

        if hidden_states_to_store_before_norm is not None:
            # NOTE: when hidden_states_before_norm is provided, we always
            # prefer to return it.
            # 中译：注意——一旦提供了归一化前状态，就总是优先返回它。
            hidden_states_to_store = hidden_states_to_store_before_norm

        return hidden_states_to_store

        # 中译：log_softmax 即数值稳定版的 log(softmax(x))，沿词表维 dim=-1 计算。
        # 中译：取 top-k token 的 logprob（若请求要求）。
        # 中译：取调用方指定的那批 token id 的 logprob（若请求要求）。
        # 中译：按「每个位置的目标 token id」逐行 gather，取出输入 token 自身的 logprob。
        # 中译：显存峰值与块大小成正比；总行数除以块大小向上取整得到块数。
        # 中译：若同一序列被拆到多个块，需记录它在之前各块中已裁剪的长度（以便跨块拼接）。
            # 中译：告知 lm_head 的 LoRA 当前处于哪一块，使其切换到预计算的逐块 batch_info；
            # 中译：取出落在本块 [start_idx, end_idx) 范围内的输入 logprob 位置，并转为块内局部下标。
            # 中译：取出 chunk_mask 为 True 的原数组位置，用于正确索引 extend_input_logprob_token_ids_gpu。
            # 中译：取本块的隐藏状态并计算 logits。
            # 中译：在第一块时初始化 sampled_logits（按采样位置数 x 词表大小）。
            # 中译：处理本块内的采样位置。必须放在下面 continue 之前，以保证 sampled_logits 被填满。
            # 中译：若本块没有需要计算输入 logprob 的位置，跳过后续处理。
            # 中译：对本块做 log_softmax 得到输入 logprob。
            # 中译：取出本块对应的「序列索引」区间，用于按序列取 top-k / token_ids 等参数。
            # 中译：取本块 top-k token 的 logprob（若请求要求）。
            # 中译：取本块中指定 token id 的 logprob（若请求要求）。
            # 中译：按「每个位置的目标 token id」逐行 gather，取出本块输入 token 自身的 logprob。
        # 中译：分块迭代结束后，恢复 lm_head 的完整（未分块）batch_info。
        # 中译：拼接各块结果。
    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        logits_metadata: LogitsMetadata,
        embedding_bias: Optional[torch.Tensor] = None,
        use_logits_buffer: bool = True,
    ) -> torch.Tensor:
        """从 hidden_states 计算 logits。

        流程：DP attention gather 隐藏状态 → 计算 LM head → （可选）logit 缩放 →
        张量并行 all-gather → DP attention scatter 回局部 → 拷入输出缓冲区 → （可选）softcap。

        说明：若调用方保证传入的 hidden_states 只包含最后一个位置（如不要 logprob 的 extend），
        则输出也只对应这些位置。
        """
        # 中译：DP attention 下先把各 rank 的局部隐藏状态 all-gather 成全局缓冲区。
        _trace_e2e_logits(
            "get_logits_enter",
            hidden_shape=tuple(hidden_states.shape),
            dp_gather=self.do_tensor_parallel_all_gather_dp_attn,
            tp_gather=self.do_tensor_parallel_all_gather,
        )
        hidden_states, local_hidden_states = self._gather_dp_attn_hidden_states(
            hidden_states, logits_metadata
        )
        _trace_e2e_logits(
            "dp_hidden_gather_returned",
            global_shape=tuple(hidden_states.shape),
            local_shape=tuple(local_hidden_states.shape),
        )

        # 中译：核心一步——用 lm_head 把隐藏状态投影为词表维的 logits。
        if envs.SGLANG_TRACE_LOGITS_E2E_SYNC.get():
            _trace_e2e_logits("pre_lm_head_sync_enter")
            torch.cuda.synchronize()
            _trace_e2e_logits("pre_lm_head_sync_returned")

        _trace_e2e_logits("lm_head_enter", hidden_shape=tuple(hidden_states.shape))
        logits = self._compute_lm_head(hidden_states, lm_head, embedding_bias)
        _trace_e2e_logits("lm_head_returned", logits_shape=tuple(logits.shape))
        if envs.SGLANG_TRACE_LOGITS_E2E_SYNC.get():
            _trace_e2e_logits("post_lm_head_sync_enter")
            torch.cuda.synchronize()
            _trace_e2e_logits("post_lm_head_sync_returned")

        # 中译：若配置了 logit 缩放系数，则原地乘上。
        if self.logit_scale is not None:
            logits.mul_(self.logit_scale)

        # 中译：张量并行下把各 rank 的部分词表 logits 聚合成完整词表。
        used_tp_lm_head_all_to_all = False
        if self.do_tensor_parallel_all_gather:
            _trace_e2e_logits(
                "tp_logits_gather_enter", logits_shape=tuple(logits.shape)
            )
            if self.use_attn_tp_group:
                logits = self._gather_attn_tp_logits(logits)
            elif self._can_use_tp_lm_head_all_to_all(
                logits, local_hidden_states, lm_head, logits_metadata
            ):
                logits = self._tp_lm_head_all_to_all(logits)
                used_tp_lm_head_all_to_all = True
            else:
                logits = self._logits_gatherer(logits)
            _trace_e2e_logits(
                "tp_logits_gather_returned", logits_shape=tuple(logits.shape)
            )

        if not used_tp_lm_head_all_to_all:
            _trace_e2e_logits(
                "dp_logits_scatter_enter", logits_shape=tuple(logits.shape)
            )
            logits = self._scatter_dp_attn_logits(
                logits, local_hidden_states, logits_metadata
            )
            _trace_e2e_logits(
                "dp_logits_scatter_returned", logits_shape=tuple(logits.shape)
            )

        logits = self._copy_logits_to_buffer(
            logits, logits_metadata, use_buffer=use_logits_buffer
        )

        # 中译：截取有效词表范围并（可选）拷入复用缓冲区，统一转为 float。
        # 中译：若模型定义了 final_logit_softcapping，对 logits 做 softcap（tanh 压缩）。
        if self.final_logit_softcapping:
            if not (_is_npu or _is_cpu):
                fused_softcap(logits, self.final_logit_softcapping)
            else:
                logits = self.final_logit_softcapping * torch.tanh(
                    logits / self.final_logit_softcapping
                )

        return logits

    def _compute_lm_head(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        quant_method = getattr(lm_head, "quant_method", None)
        if hasattr(lm_head, "set_lora") and hasattr(lm_head, "apply_lora"):
            # This is a LoRA-wrapped module, use its forward method
            # 中译：LoRA 包装的 lm_head，直接调用其 forward。
            logits = lm_head(hidden_states)
        elif should_apply_lm_head_quant_method(lm_head, quant_method):
            logits = quant_method.apply(lm_head, hidden_states, embedding_bias)
        elif hasattr(lm_head, "weight"):
            # Normal linear layer
            # 中译：普通线性层。
            if self.use_fp32_lm_head:
                # Avoid materializing FP32 copies for same-dtype CUDA FP16/BF16
                # inputs. Retain explicit FP32 casts for unsupported devices or
                # dtype combinations.
                use_mm_out_dtype = (
                    hidden_states.is_cuda
                    and hidden_states.dtype == lm_head.weight.dtype
                    and hidden_states.dtype in (torch.float16, torch.bfloat16)
                )
                if use_mm_out_dtype:
                    logits = torch.mm(
                        hidden_states,
                        lm_head.weight.T,
                        out_dtype=torch.float32,
                    )
                else:
                    logits = torch.matmul(
                        hidden_states.to(torch.float32),
                        lm_head.weight.to(torch.float32).T,
                    )
            elif use_intel_amx_backend(lm_head):
                logits = torch.ops.sgl_kernel.weight_packed_linear(
                    hidden_states.to(lm_head.weight.dtype),
                    lm_head.weight,
                    None,  # bias
                    True,  # is_vnni
                )
            elif self.rl_on_policy_target is not None:
                # Due to tie-weight, we may not be able to change lm_head's weight dtype
                # 中译：RL on-policy 场景：因权重绑定（tie-weight）可能无法改变 lm_head 权重的 dtype，
                #       故这里统一转为 bfloat16 再做 matmul。
                logits = torch.matmul(
                    hidden_states.bfloat16(), lm_head.weight.T.bfloat16()
                )
            else:
                logits = torch.matmul(
                    hidden_states.to(lm_head.weight.dtype), lm_head.weight.T
                )
        else:
            # GGUF models
            # TODO: use weight_packed_linear for GGUF models
            # 中译：GGUF 等量化模型，走 quant_method.apply；TODO：后续可改用 weight_packed_linear。
            if self.use_fp32_lm_head:
                with torch.cuda.amp.autocast(enabled=False):
                    logits = lm_head.quant_method.apply(
                        lm_head, hidden_states.to(torch.float32), embedding_bias
                    )
            else:
                logits = lm_head.quant_method.apply(
                    lm_head, hidden_states, embedding_bias
                )
        return logits

    def _gather_dp_attn_hidden_states(
        self, hidden_states: torch.Tensor, logits_metadata: LogitsMetadata
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """DP attention 下把各 rank 的局部隐藏状态 gather 成全局缓冲区。

        返回 (全局隐藏状态, 本 rank 局部隐藏状态)；未启用时两者均为原输入。
        """
        if self.do_tensor_parallel_all_gather_dp_attn:
            _trace_e2e_logits(
                "dp_metadata_enter",
                local_shape=tuple(hidden_states.shape),
                global_counts_cpu=logits_metadata.global_num_tokens_for_logprob_cpu,
            )
            logits_metadata.compute_dp_attention_metadata()
            _trace_e2e_logits(
                "dp_metadata_returned",
                buffer_shape=tuple(logits_metadata.gathered_buffer.shape),
                local_start=logits_metadata.dp_local_start_pos,
                local_tokens=logits_metadata.dp_local_num_tokens,
            )
            local_hidden_states = hidden_states
            hidden_states = logits_metadata.gathered_buffer
            _trace_e2e_logits(
                "dp_hidden_gather_enter",
                global_shape=tuple(hidden_states.shape),
                local_shape=tuple(local_hidden_states.shape),
            )
            dp_gather_replicate(hidden_states, local_hidden_states, logits_metadata)
            _trace_e2e_logits("dp_hidden_gather_collective_returned")
            return hidden_states, local_hidden_states
        return hidden_states, hidden_states

    def _gather_attn_tp_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """在 attention TP 组内把各 rank 的部分词表 logits all-gather 成完整词表。

        词表能被 attn_tp_size 整除时走高效的 into_tensor 路径并 reshape；
        否则退化到按最后一维 split 的通用 all-gather 路径。
        """
        if self.vocab_size % self.attn_tp_size == 0:
            global_logits = torch.empty(
                (
                    self.attn_tp_size,
                    logits.shape[0],
                    self.vocab_size // self.attn_tp_size,
                ),
                device=logits.device,
                dtype=logits.dtype,
            )
            attn_tp_all_gather_into_tensor(global_logits, logits)
            global_logits = global_logits.permute(1, 0, 2).reshape(
                logits.shape[0], self.vocab_size
            )
        else:
            global_logits = torch.empty(
                (self.vocab_size, logits.shape[0]),
                device=logits.device,
                dtype=logits.dtype,
            )
            global_logits = global_logits.T
            attn_tp_all_gather(
                list(global_logits.tensor_split(self.attn_tp_size, dim=-1)),
                logits,
            )
        return global_logits

    def _can_use_tp_lm_head_all_to_all(
        self,
        logits: torch.Tensor,
        local_hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        logits_metadata: LogitsMetadata,
    ) -> bool:
        if not self.use_tp_lm_head_all_to_all:
            return False

        tp_size = get_parallel().tp_size
        base_lm_head = getattr(lm_head, "base_layer", lm_head)
        if getattr(base_lm_head, "tp_size", None) != tp_size:
            # Tied embeddings may be replicated across DP ranks (tp_size=1),
            # even though the logits processor runs in a larger global TP
            # group. Such logits are full-vocabulary rather than TP shards and
            # therefore do not satisfy the all-to-all layout contract.
            return False

        # Every participant must make the same collective choice. Decode CUDA
        # graphs omit CPU counts and fill every GPU count with the same padded
        # bucket size. Eager batches carry the same global CPU count list on
        # every rank, so they are also safe when all entries are equal.
        global_counts_cpu = logits_metadata.global_num_tokens_for_logprob_cpu
        is_equal_padded_graph_layout = global_counts_cpu is None and (
            logits_metadata.global_num_tokens_for_logprob_gpu is not None
        )
        is_equal_eager_layout = (
            global_counts_cpu is not None
            and len(global_counts_cpu) == tp_size
            and len(global_counts_cpu) > 0
            and all(count == global_counts_cpu[0] for count in global_counts_cpu)
        )
        if not (is_equal_padded_graph_layout or is_equal_eager_layout):
            return False

        local_rows = local_hidden_states.shape[0]
        return local_rows > 0 and logits.shape[0] == local_rows * tp_size

    def _tp_lm_head_all_to_all(self, logits: torch.Tensor) -> torch.Tensor:
        """Exchange only the row block owned by each destination DP rank."""
        logits = logits.contiguous()
        all_to_all_output = torch.empty_like(logits)
        get_tp_group().all_to_all_single(all_to_all_output.view(-1), logits.view(-1))
        return _reassemble_tp_lm_head_all_to_all_output(
            all_to_all_output, get_parallel().tp_size
        )

    def _scatter_dp_attn_logits(
        self,
        logits: torch.Tensor,
        local_hidden_states: torch.Tensor,
        logits_metadata: LogitsMetadata,
    ) -> torch.Tensor:
        """DP attention 下把全局 logits scatter 回本 rank 局部对应的那部分。"""
        if self.do_tensor_parallel_all_gather_dp_attn:
            global_logits = logits
            logits = torch.empty(
                (local_hidden_states.shape[0], global_logits.shape[1]),
                device=global_logits.device,
                dtype=global_logits.dtype,
            )
            dp_scatter(logits, global_logits, logits_metadata)
        return logits

    def _copy_logits_to_buffer(
        self,
        logits: torch.Tensor,
        logits_metadata: LogitsMetadata,
        use_buffer: bool = True,
    ) -> torch.Tensor:
        logits_buffer = logits_metadata.next_token_logits_buffer if use_buffer else None
        if logits.shape[-1] > self.vocab_size:
            logits = logits[:, : self.vocab_size]
        logits_width = logits.shape[-1]
        # The shared logits buffer is keyed by vocab width and rows; skip it
        # when this batch has a different logits shape than the graph buffer.
        if logits_buffer is not None and tuple(logits_buffer.shape) == tuple(
            logits.shape
        ):
            assert logits_buffer.dtype == torch.float
            logits_buffer.copy_(logits)
            logits = logits_buffer
        else:
            logits = logits.float()
        return logits

    def _get_dllm_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        logits_metadata: LogitsMetadata,
    ) -> LogitsProcessorOutput:
        """扩散式 LLM（Diffusion LLM）专用：返回完整 logits（而非只取下一个 token）。"""
        assert self.return_full_logits
        full_logits = self._get_logits(hidden_states, lm_head, logits_metadata)
        return LogitsProcessorOutput(
            full_logits=full_logits,
            next_token_logits=None,
        )

    def compute_logprobs_for_multi_item_scoring(
        self,
        input_ids,
        hidden_states,
        lm_head: VocabParallelEmbedding,
        logits_metadata: Union[LogitsMetadata, ForwardBatch],
        multi_item_delimiter_indices: List[torch.Tensor],
    ):
        """
        基于预计算的分隔符位置，为 multi-item scoring（多项打分）计算 logprob。

        序列格式：Query<分隔符>Item1<分隔符>Item2<分隔符>...
        打分位置：在每个 <分隔符> 之前的位置提取 logprob。

        Args:
            input_ids: 输入 token id，形状 [总序列长度]。
            hidden_states: 模型输出的隐藏状态，形状 [序列长度, hidden_dim]。
            lm_head: 用于计算 logits 的语言模型头。
            logits_metadata: 包含批信息与 logprob 规格的元数据。
            multi_item_delimiter_indices: 各请求预计算好的分隔符位置（CPU 张量）。
        """
        # Compute positions just before each delimiter.
        # Build offset-adjusted indices on CPU, then do a single CPU→GPU transfer.
        # 中译：计算每个分隔符「之前一位」的位置；先在 CPU 上构造加上偏移的索引，
        #       再一次性拷到 GPU（减少 CPU→GPU 传输次数）。
        device = input_ids.device
        all_tensors = []
        if logits_metadata.extend_seq_lens_cpu is not None:
            offset = 0
            for req_seq_len, indices_tensor in zip(
                logits_metadata.extend_seq_lens_cpu, multi_item_delimiter_indices
            ):
                if len(indices_tensor) > 0:
                    # Note: if the first delimiter is at position 0 (empty query),
                    # indices - 1 wraps to -1. This is harmless — the first
                    # delimiter entry is always discarded by
                    # _process_multi_item_scoring_results.
                    # 中译：注意——若第一个分隔符在位置 0（空 Query），indices-1 会绕到 -1；
                    #       这是无害的，因为第一个分隔符项总会被 _process_multi_item_scoring_results 丢弃。
                    all_tensors.append(indices_tensor + (offset - 1))
                offset += req_seq_len
        else:
            all_tensors.append(multi_item_delimiter_indices[0] - 1)
        multi_item_indices = torch.cat(all_tensors).to(device, non_blocking=True)

        # Extract hidden states at delimiter positions for multi-item scoring
        # 中译：取出各分隔符位置处的隐藏状态，用于多项打分。
        sliced_hidden = hidden_states[multi_item_indices]

        # 中译：算出这些位置的 logits 并做 log_softmax 得到对数概率。
        sliced_logits = self._get_logits(sliced_hidden, lm_head, logits_metadata)
        sliced_logprobs = torch.nn.functional.log_softmax(sliced_logits, dim=-1)

        # Initialize return values
        # 中译：初始化返回值。
        input_token_ids_logprobs_val = []
        input_token_ids_logprobs_idx = []
        input_top_logprobs_val = None
        input_top_logprobs_idx = None

        # Recalculate extend_logprob_pruned_lens_cpu to match delimiter counts per request
        # 中译：重算 extend_logprob_pruned_lens_cpu，使其与每个请求的分隔符个数对齐。
        if (
            logits_metadata.token_ids_logprobs
            or logits_metadata.extend_return_top_logprob
        ):
            logits_metadata.extend_logprob_pruned_lens_cpu = [
                len(t) for t in multi_item_delimiter_indices
            ]

        # Get the logprobs of specified token ids
        # 中译：取指定 token id 的 logprob（若请求要求）。
        if logits_metadata.extend_token_ids_logprob:
            (
                input_token_ids_logprobs_val,
                input_token_ids_logprobs_idx,
            ) = get_token_ids_logprobs_raw(
                sliced_logprobs,
                logits_metadata.token_ids_logprobs,
                stage=LogprobStage.PREFILL,
                extend_logprob_pruned_lens_cpu=logits_metadata.extend_logprob_pruned_lens_cpu,
                no_copy_to_cpu=True,
            )

        # Get the logprob of top-k tokens
        # 中译：取 top-k token 的 logprob（若请求要求）。
        if logits_metadata.extend_return_top_logprob:
            (
                input_top_logprobs_val,
                input_top_logprobs_idx,
            ) = get_top_logprobs_raw(
                sliced_logprobs,
                logits_metadata.top_logprobs_nums,
                stage=LogprobStage.PREFILL,
                extend_logprob_pruned_lens_cpu=logits_metadata.extend_logprob_pruned_lens_cpu,
            )

        # MIS scores come from input_token_ids_logprobs_val (label-token logprobs),
        # not from per-position input_token_logprobs. However, the shared logprob
        # pipeline (add_input_logprob_return_values) asserts input_token_logprobs is
        # non-None, converts it to a tuple, slices it, and validates its length —
        # all before score_request() ever sees the result. We can't set it to None
        # without changing those shared asserts, so we fill with zeros to satisfy
        # the pipeline. score_request() ignores this field entirely.
        input_token_logprobs = torch.zeros(multi_item_indices.shape[0], device=device)

        return LogitsProcessorOutput(
            next_token_logits=None,
            input_token_logprobs=input_token_logprobs,
            input_top_logprobs_val=input_top_logprobs_val,
            input_top_logprobs_idx=input_top_logprobs_idx,
            input_token_ids_logprobs_val=input_token_ids_logprobs_val,
            input_token_ids_logprobs_idx=input_token_ids_logprobs_idx,
            mm_input_embeds=logits_metadata.mm_input_embeds,
        )


def _reassemble_tp_lm_head_all_to_all_output(
    all_to_all_output: torch.Tensor, tp_size: int
) -> torch.Tensor:
    """Convert source-major all-to-all output to row-major full-vocab logits.

    Each source TP rank contributes ``[local_rows, vocab_shard]`` for this
    destination DP rank. ``all_to_all_single`` concatenates those contributions
    along dim 0, while the sampler expects the vocab shards concatenated along
    dim 1.
    """
    assert all_to_all_output.shape[0] % tp_size == 0
    local_rows = all_to_all_output.shape[0] // tp_size
    vocab_shard = all_to_all_output.shape[1]
    return (
        all_to_all_output.view(tp_size, local_rows, vocab_shard)
        .permute(1, 0, 2)
        .reshape(local_rows, tp_size * vocab_shard)
    )


def _has_lm_head_runtime_attrs(lm_head, attr_names: Tuple[str, ...]) -> bool:
    return all(hasattr(lm_head, attr_name) for attr_name in attr_names)


def should_apply_lm_head_quant_method(lm_head, quant_method) -> bool:
    if (
        quant_method is None
        or not hasattr(lm_head, "weight")
        or not callable(getattr(quant_method, "apply", None))
    ):
        return False

    method_name = type(quant_method).__name__
    if method_name in _UNQUANTIZED_LM_HEAD_METHODS:
        return False

    # Some draft models share an unquantized target lm_head tensor while still
    # carrying the draft model's stale ModelOpt quant_method. Only use the
    # ModelOpt lm_head kernel when the runtime quantization state matches it.
    if method_name == "ModelOptFp4LinearMethod":
        if lm_head.weight.dtype == torch.int32 and _has_lm_head_runtime_attrs(
            lm_head,
            (
                "weight_scale",
                "weight_global_scale",
                "workspace",
                "input_size_per_partition",
                "output_size_per_partition",
            ),
        ):
            return True
        return lm_head.weight.dtype == torch.uint8 and _has_lm_head_runtime_attrs(
            lm_head,
            (
                "weight_scale_interleaved",
                "alpha",
                "input_scale_inv",
                "input_size_per_partition",
                "output_size_per_partition",
            ),
        )
    if method_name == "ModelOptNvFp4A16LinearMethod":
        return lm_head.weight.dtype == torch.int32 and _has_lm_head_runtime_attrs(
            lm_head,
            (
                "weight_scale",
                "weight_global_scale",
                "workspace",
                "input_size_per_partition",
                "output_size_per_partition",
            ),
        )
    if method_name == "ModelOptFp8LinearMethod":
        return (
            lm_head.weight.dtype == torch.float8_e4m3fn
            and _has_lm_head_runtime_attrs(lm_head, ("weight_scale", "input_scale"))
        )

    return True
