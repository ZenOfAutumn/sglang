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
The definition of objects transferred between different
processes (TokenizerManager, DetokenizerManager, Scheduler).

中译：本文件定义了在 SGLang 三个核心进程之间（TokenizerManager、DetokenizerManager、
      Scheduler）通过 ZMQ 传输的全部数据结构，是「跨进程协议（IPC protocol）」的核心。
      这些对象本质上都是 dataclass，会被 pickle 序列化后在进程间收发。大致可分为几类：
      1) 请求输入类（*ReqInput）：HTTP 端 → TokenizerManager → Scheduler 的请求载荷，
         如 GenerateReqInput / EmbeddingReqInput 及其分词后的 Tokenized* 版本。
      2) 批次输出类（Batch*Output）：Scheduler → DetokenizerManager → TokenizerManager
         的批次结果，如 BatchTokenIDOutput / BatchStrOutput / BatchEmbeddingOutput。
      3) 控制/管理类：权重更新、缓存清理、暂停/恢复、性能分析、LoRA 管理、负载查询等。
      命名约定：*ReqInput 通常是「请求」，对应的 *ReqOutput 是「应答」。
"""

from __future__ import annotations

import copy
import uuid
from abc import ABC
from array import array
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Dict, List, Literal, Optional, Union

import torch
from pydantic import PlainValidator

from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.embed_types import PositionalEmbeds
from sglang.srt.managers.schedule_batch import BaseFinishReason, Modality
from sglang.srt.multimodal.mm_utils import has_valid_data
from sglang.srt.observability.req_time_stats import (
    APIServerReqTimeStats,
    DPControllerReqTimeStats,
    SchedulerReqTimeStats,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import ImageData, VideoData
from sglang.srt.utils.field_validators import validate_optional_list_i64_1d_2d

# Handle serialization of Image for pydantic
if TYPE_CHECKING:
    from PIL.Image import Image
else:
    Image = Any


@dataclass
class BaseReq(ABC):
    # 中译：所有「单请求」类消息的基类。提供两个跨进程通用字段：
    #       - rid：请求 id（request id），可为单个字符串或字符串列表（批次内多个 rid）；
    #       - http_worker_ipc：多 HTTP worker 模式下，标识结果应回送到哪个 HTTP worker 的 IPC 名。
    rid: Optional[Union[str, List[str]]] = field(default=None, kw_only=True)
    http_worker_ipc: Optional[str] = field(default=None, kw_only=True)

    def regenerate_rid(self):
        """Generate a new request ID and return it.

        中译：重新生成请求 id（rid 为列表时逐个重新生成），并返回新值。
        """
        if isinstance(self.rid, list):
            self.rid = [uuid.uuid4().hex for _ in range(len(self.rid))]
        else:
            self.rid = uuid.uuid4().hex
        return self.rid

    def _validate_rid_uniqueness(self):
        """Validate that request IDs within a batch are unique.

        中译：校验同一批次内的 rid 互不重复（重复会导致结果串台），重复则抛错。
        """
        if isinstance(self.rid, list) and len(set(self.rid)) != len(self.rid):
            counts = Counter(self.rid)
            duplicates = [rid for rid, count in counts.items() if count > 1]
            raise ValueError(
                f"Duplicate request IDs detected within the request: {duplicates}"
            )


@dataclass
class BaseBatchReq(ABC):
    # 中译：所有「批次」类消息的基类。与 BaseReq 对应，但字段是复数列表形式：
    #       - rids：批次内每个请求的 id 列表；
    #       - http_worker_ipcs：批次内每个请求对应的 HTTP worker IPC 名列表。
    rids: Optional[List[str]] = field(default=None, kw_only=True)
    http_worker_ipcs: Optional[List[str]] = field(default=None, kw_only=True)

    def regenerate_rids(self):
        """Generate new request IDs and return them.

        中译：为批次内所有请求重新生成 rid 并返回。
        """
        self.rids = [uuid.uuid4().hex for _ in range(len(self.rids))]
        return self.rids


@dataclass
class SpeculativeDecodingMetricsMixin:
    """
    Mixin class containing speculative decoding metrics.

    This class consolidates speculative decoding metrics that are shared across
    batch output types that support speculative decoding to avoid code duplication.

    中译：推测解码（speculative decoding）统计指标的 Mixin。把多个支持推测解码的批次输出
          类型共用的统计字段抽到这里，避免重复定义。
    """

    # Verify count: number of verification forward passes
    # 中译：验证次数——目标模型为校验草稿 token 而做的前向次数（逐请求计数）。
    spec_verify_ct: List[int]

    # Accepted drafts: Number of accepted draft tokens during speculative decoding
    # (strict drafts-only count, excludes the bonus token).
    # 中译：被接受的草稿 token 数（严格只数草稿，不含每步附带的 bonus token）。
    spec_num_correct_drafts: List[int]

    # Acceptance histogram: List of lists, where each inner list represents histogram counts.
    # List index = number of accepted tokens in a step, List value = count of steps with that many accepted tokens.
    # Example: histogram[0] = 5 means 5 steps with 0 accepted tokens, histogram[3] = 10 means 10 steps with 3 accepted tokens.
    # Empty list [] when speculative decoding is disabled.
    # 中译：接受数直方图（逐请求一个内层列表）。下标 = 某步接受的 token 数，值 = 出现该接受数的步数。
    #       例：histogram[0]=5 表示有 5 步接受了 0 个 token；histogram[3]=10 表示 10 步接受了 3 个。
    #       未启用推测解码时为空列表 []。
    spec_correct_drafts_histogram: List[List[int]]


# Parameters for a session
# 中译：会话（session）参数，用于「持续对话/续写」场景，让多次请求复用同一会话上下文。
@dataclass
class SessionParams:
    id: Optional[str] = None  # 会话 id
    rid: Optional[str] = None  # 关联的请求 id
    offset: Optional[int] = None  # 续写时在已有序列中的起始偏移
    replace: Optional[bool] = None  # 是否替换会话中已有内容
    drop_previous_output: Optional[bool] = None  # 是否丢弃上一轮的输出


# Type definitions for multimodal input data
# Individual data item types for each modality
# 中译：多模态输入数据的类型定义。每种模态（图像/音频/视频）允许传入对象实例、文件名、URL、
#       base64 字符串或 dict；MultimodalDataInputFormat 进一步允许单个、列表、嵌套列表
#       （用于批处理：每个请求一个列表、每个请求多个文件）。
ImageDataInputItem = Union[Image, str, ImageData, Dict]
AudioDataInputItem = Union[str, Dict]
VideoDataInputItem = Union[str, VideoData, Dict]
# Union type for any multimodal data item
MultimodalDataInputItem = Union[
    ImageDataInputItem, VideoDataInputItem, AudioDataInputItem
]
# Format types supporting single items, lists, or nested lists for batch processing
MultimodalDataInputFormat = Union[
    List[List[MultimodalDataInputItem]],
    List[MultimodalDataInputItem],
    MultimodalDataInputItem,
]


@dataclass
class GenerateReqInput(BaseReq):
    """中译：生成（文本补全）请求的「原始输入」对象。

    由 HTTP/Engine 入口构造，送入 TokenizerManager。它尚未分词，可代表单个请求或一个批次。
    TokenizerManager 会调用 normalize_batch_and_arguments() 把各字段规整成统一形态
    （单值或等长列表），完成并行采样（n>1）的展开，然后分词为 TokenizedGenerateReqInput
    再发给 Scheduler。text / input_ids / input_embeds 三者必须且只能提供其一。
    """

    # The input prompt. It can be a single prompt or a batch of prompts.
    # 中译：输入提示词。可为单个字符串，或字符串列表（批处理）。
    text: Optional[Union[List[str], str]] = None
    # The token ids for text.
    #
    # Use C-loop validator to replace Pydantic per-element type check for efficiency.
    # 中译：text 对应的 token id（已分词输入）。用 C 循环校验器替代 Pydantic 逐元素类型检查以提速。
    input_ids: Annotated[
        Optional[Union[List[List[int]], List[int]]],
        PlainValidator(validate_optional_list_i64_1d_2d),
    ] = None
    # The embeddings for input_ids; one can specify either text or input_ids or input_embeds.
    # 中译：直接以嵌入向量作为输入（绕过 embedding 层）；text/input_ids/input_embeds 三选一。
    input_embeds: Optional[Union[List[List[List[float]]], List[List[float]]]] = None
    # The image input. It can be an image instance, file name, URL, or base64 encoded string.
    # Can be formatted as:
    # - Single image for a single request
    # - List of images (one per request in a batch)
    # - List of lists of images (multiple images per request)
    # See also python/sglang/srt/utils.py:load_image for more details.
    image_data: Optional[MultimodalDataInputFormat] = None  # 中译：图像输入（多模态）
    # The video input. Like image data, it can be a file name, a url, or base64 encoded string.
    video_data: Optional[MultimodalDataInputFormat] = None  # 中译：视频输入（多模态）
    # The audio input. Like image data, it can be a file name, a url, or base64 encoded string.
    audio_data: Optional[MultimodalDataInputFormat] = None  # 中译：音频输入（多模态）
    # Optional per-image hashes the caller has already computed (hex strings,
    # one per image in `image_data`). When supplied, each MultimodalDataItem's
    # `hash` is initialised from this list and `set_pad_value` skips the
    # internal `hash_feature()` recompute, so the resulting `pad_value` is
    # deterministic from the caller's hash. Intended for external KV routers
    # that compute their own per-image hash for routing decisions and need
    # sglang's prefix-cache key to align. When unset, behavior is unchanged
    # (sglang hashes the processor feature tensor).
    # 中译：调用方预先算好的逐图哈希（十六进制字符串，每图一个）。提供后用它初始化每个多模态项的
    #       hash，并跳过内部 hash_feature() 重算，使 pad_value 可由调用方哈希确定性推导。
    #       主要给「外部 KV 路由器」使用——它们自算逐图哈希做路由，需要与 sglang 前缀缓存 key 对齐。
    mm_hashes: Optional[Union[List[str], List[List[str]]]] = None
    # Whether to extract and process audio from video inputs.
    use_audio_in_video: bool = False  # 中译：是否从视频中抽取并处理音轨
    # The sampling_params. See descriptions below.
    # 中译：采样参数（温度、top_p、max_new_tokens、n 等）；可为单个 dict 或逐请求的 dict 列表。
    sampling_params: Optional[Union[List[Dict], Dict]] = None
    # Whether to return logprobs.
    return_logprob: Optional[Union[List[bool], bool]] = None  # 中译：是否返回 logprob
    # If return logprobs, the start location in the prompt for returning logprobs.
    # By default, this value is "-1", which means it will only return logprobs for output tokens.
    # 中译：返回 logprob 时，从提示词的哪个位置开始返回。默认 -1 表示只返回输出 token 的 logprob。
    logprob_start_len: Optional[Union[List[int], int]] = None
    # If return logprobs, the number of top logprobs to return at each position.
    # 中译：返回 logprob 时，每个位置返回的 top-k 候选数量。
    top_logprobs_num: Optional[Union[List[int], int]] = None
    # If return logprobs, the token ids to return logprob for.
    # 中译：返回 logprob 时，额外指定要返回其 logprob 的特定 token id 集合。
    token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None
    # Whether to detokenize tokens in text in the returned logprobs.
    # 中译：返回的 logprob 中是否把 token 反向解码为文本一并附上。
    return_text_in_logprobs: bool = False
    # Whether to stream output.
    stream: bool = False  # 中译：是否流式返回输出
    # Whether to log metrics for this request (e.g. health_generate calls do not log metrics)
    # 中译：本请求是否记录指标（如健康检查 health_generate 不记录，避免污染统计）。
    log_metrics: bool = True
    # Whether to return hidden states
    return_hidden_states: Union[List[bool], bool] = False  # 中译：是否返回隐藏状态
    # Whether to return captured routed experts
    # 中译：是否返回 MoE 路由选中的专家（routed experts），用于分析/可观测。
    return_routed_experts: bool = False
    return_indexer_topk: bool = False  # 中译：是否返回稀疏注意力索引器选中的 top-k token
    # Absolute start position for returned routings; response covers
    # `[routed_experts_start_len, seqlen - 1)`. Must be in [0, prompt_tokens].
    # 0 = full sequence.
    # 中译：返回路由信息的绝对起始位置，覆盖区间为 [routed_experts_start_len, seqlen-1)。
    #       取值须在 [0, prompt_tokens]；0 表示返回整条序列。
    routed_experts_start_len: int = 0

    # The modalities of the image data [image, multi-images, video]
    # 中译：图像数据的模态标识（image / multi-images / video），随归一化自动填充。
    modalities: Optional[List[str]] = None
    # Session info for continual prompting
    session_params: Optional[Union[List[Dict], Dict]] = None  # 中译：持续对话/续写的会话信息

    # The path to the LoRA adaptors
    lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None  # 中译：LoRA 适配器路径
    # The uid of LoRA adaptors, should be initialized by tokenizer manager
    # 中译：LoRA 适配器的唯一 id，由 TokenizerManager 根据 lora_path 解析填充。
    lora_id: Optional[Union[List[Optional[str]], Optional[str]]] = None

    # Custom logit processor for advanced sampling control. Must be a serialized instance
    # of `CustomLogitProcessor` in python/sglang/srt/sampling/custom_logit_processor.py
    # Use the processor's `to_str()` method to generate the serialized string.
    custom_logit_processor: Optional[Union[List[Optional[str]], str]] = None
    # Embedding overrides to place at specific token positions.
    # Runtime type: Optional[Union[PositionalEmbeds, List[Optional[PositionalEmbeds]]]]
    # Typed as Any to avoid Pydantic/FastAPI schema errors (PositionalEmbeds contains torch.Tensor).
    # 中译：在指定 token 位置覆盖嵌入向量。运行时类型为 PositionalEmbeds（或其列表）；
    #       因含 torch.Tensor 会触发 Pydantic/FastAPI schema 报错，故此处标注为 Any 规避。
    positional_embed_overrides: Any = None

    # For disaggregated inference
    # 中译：以下为「PD 分离（prefill/decode 分离部署）」推理用的对接参数。
    bootstrap_host: Optional[Union[List[str], str]] = None  # 中译：对端 bootstrap 主机地址
    bootstrap_port: Optional[Union[List[Optional[int]], int]] = None  # 中译：对端 bootstrap 端口
    bootstrap_room: Optional[Union[List[int], int]] = None  # 中译：配对房间号（用于 prefill/decode 配对）
    bootstrap_pair_key: Optional[Union[List[str], str]] = None  # 中译：配对 key
    decode_tp_size: Optional[Union[List[Optional[int]], int]] = None  # 中译：decode 端的张量并行大小

    # Require reasoning for the request (hybrid reasoning model only)
    # 中译：强制本请求进入推理（thinking）模式（仅混合推理模型可用）。
    require_reasoning: bool = False

    # For DP routing — external router assigns a specific DP worker
    # 中译：DP 路由——由外部路由器指定本请求应分配到哪个 DP（数据并行）worker。
    routed_dp_rank: Optional[int] = None
    # For PD disagg — hint telling decode which prefill DP worker has the KV cache
    # 中译：PD 分离场景——提示 decode 端：KV cache 在哪个 prefill 的 DP worker 上。
    disagg_prefill_dp_rank: Optional[int] = None
    # Deprecated: use routed_dp_rank instead
    # 中译：已废弃，请改用 routed_dp_rank。
    data_parallel_rank: Optional[int] = None

    # For background responses (OpenAI responses API)
    # 中译：后台模式（OpenAI responses API），结果异步生成、稍后取回。
    background: bool = False

    # Conversation id used for tracking requests
    conversation_id: Optional[str] = None  # 中译：会话 id，用于跟踪/串联请求

    # Priority for the request
    priority: Optional[int] = None  # 中译：请求优先级（调度排序用，数值越大越优先）

    # Extra key for classifying the request (e.g. cache_salt)
    # 中译：请求分类的附加 key（如 cache_salt），可用于区分前缀缓存命名空间。
    extra_key: Optional[Union[List[str], str]] = None

    # Routing key for routing-key schedule policy
    # 中译：routing-key 调度策略所用的路由 key（同 key 倾向落到同一 worker，利于复用缓存）。
    routing_key: Optional[str] = None

    # Whether to disallow logging for this request (e.g. due to ZDR)
    # 中译：是否禁止对本请求记日志（如出于零数据保留 ZDR 合规要求）。
    no_logs: bool = False

    # For custom metric labels
    custom_labels: Optional[Dict[str, str]] = None  # 中译：自定义指标标签

    # (Internal) Whether to return bytes for image generation
    return_bytes: bool = False  # 中译：（内部）图像生成时是否返回原始字节

    # Whether to return entropy
    return_entropy: bool = False  # 中译：是否返回输出分布的熵

    # Whether to return prompt token IDs without computing logprobs
    # 中译：是否仅返回提示词 token id（不计算 logprob）。
    return_prompt_token_ids: bool = False

    # Propagates trace context via Engine.generate/async_generate
    # 中译：透传分布式追踪上下文（trace context），用于链路追踪。
    external_trace_header: Optional[Dict] = None
    received_time: Optional[float] = None  # 中译：请求被接收的时间戳（可观测/排队耗时统计）

    # For EPD-disaggregated inference
    # 中译：以下为 EPD 分离（encoder/prefill/decode 分离）推理用字段。
    need_wait_for_mm_inputs: Optional[bool] = None  # 中译：是否需等待多模态输入（由独立 encoder 计算）就绪
    num_items_assigned: Optional[Dict[Modality, List[int]]] = None  # 中译：各模态分配给本请求的条目数
    mm_data_mooncake: Optional[List] = None  # 中译：经 Mooncake 传输的多模态数据句柄
    # Snapshot of encoder URLs at the time tokenizer-side computed
    # ``num_items_assigned``.
    # 中译：tokenizer 端计算 num_items_assigned 时冻结的 encoder URL 快照，保证 encoder 索引分配一致。
    encoder_urls: Optional[List[str]] = None

    # Multimodal tiling controls (extensions)
    # 中译：多模态「动态切块（dynamic patch/tiling）」控制（扩展项，影响图像被切成多少子图）。
    max_dynamic_patch: Optional[int] = None
    min_dynamic_patch: Optional[int] = None
    image_max_dynamic_patch: Optional[int] = None
    video_max_dynamic_patch: Optional[int] = None

    # Pre-computed delimiter indices for multi-item scoring.
    # Batch-level: List[List[int]] (one per request). After __getitem__: List[int].
    # 中译：多条目打分（multi-item scoring）预计算的分隔符下标。
    #       批次级为 List[List[int]]（每请求一组）；经 __getitem__ 取单条后变为 List[int]。
    multi_item_delimiter_indices: Optional[Union[List[List[int]], List[int]]] = None

    def contains_mm_input(self) -> bool:
        # 中译：判断本请求是否携带任意多模态输入（图/视频/音频）。
        return (
            has_valid_data(self.image_data)
            or has_valid_data(self.video_data)
            or has_valid_data(self.audio_data)
        )

    def normalize_batch_and_arguments(self):
        """
        Normalize the batch size and arguments for the request.

        This method resolves various input formats and ensures all parameters
        are properly formatted as either single values or batches depending on the input.
        It also handles parallel sampling expansion and sets default values for
        unspecified parameters.

        Raises:
            ValueError: If inputs are not properly specified (e.g., none or all of
                       text, input_ids, input_embeds are provided)

        中译：归一化批次大小与各参数。把多样的输入形态统一成「单值」或「等长列表」，
              处理并行采样（n>1）的批次展开，并为未指定的参数填默认值。
              输入非法（如 text/input_ids/input_embeds 一个都没给或都给了）时抛 ValueError。
        """
        if self.data_parallel_rank is not None:
            import warnings

            warnings.warn(
                "'data_parallel_rank' is deprecated, use 'routed_dp_rank' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self.routed_dp_rank is None:
                self.routed_dp_rank = self.data_parallel_rank
            self.data_parallel_rank = None

        self._validate_inputs()
        self._determine_batch_size()
        self._handle_parallel_sampling()

        if self.is_single:
            self._normalize_single_inputs()
        else:
            self._normalize_batch_inputs()

        self._validate_rid_uniqueness()

    def _validate_inputs(self):
        """Validate that the input configuration is valid."""
        if (
            self.text is None and self.input_ids is None and self.input_embeds is None
        ) or (
            self.text is not None
            and self.input_ids is not None
            and self.input_embeds is not None
        ):
            raise ValueError(
                "Either text, input_ids or input_embeds should be provided."
            )

    def _determine_batch_size(self):
        """Determine if this is a single example or a batch and the batch size."""
        if self.text is not None:
            if isinstance(self.text, str):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.text)
            self.input_embeds = None
        elif self.input_ids is not None:
            if len(self.input_ids) == 0:
                raise ValueError("input_ids cannot be empty.")
            if isinstance(self.input_ids[0], int):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.input_ids)
            self.input_embeds = None
        else:
            if isinstance(self.input_embeds[0][0], float):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.input_embeds)

    def _handle_parallel_sampling(self):
        """Handle parallel sampling parameters and adjust batch size if needed.

        中译：处理并行采样参数（采样数 n）。当对单个样本要求 n>1 时，把它转成批次形态，
              以便后续按 batch_size * n 统一展开。要求批内各样本的 n 一致。
        """
        # Determine parallel sample count
        if self.sampling_params is None:
            self.parallel_sample_num = 1
            return
        elif isinstance(self.sampling_params, dict):
            self.parallel_sample_num = self.sampling_params.get("n", 1)
        else:  # isinstance(self.sampling_params, list):
            self.parallel_sample_num = self.sampling_params[0].get("n", 1)
            for sampling_params in self.sampling_params:
                if self.parallel_sample_num != sampling_params.get("n", 1):
                    raise ValueError(
                        "The parallel_sample_num should be the same for all samples in sample params."
                    )

        # If using parallel sampling with a single example, convert to batch
        if self.parallel_sample_num > 1 and self.is_single:
            self.is_single = False
            if self.text is not None:
                self.text = [self.text]
            if self.input_ids is not None:
                self.input_ids = [self.input_ids]
            if self.input_embeds is not None:
                self.input_embeds = [self.input_embeds]

    def _normalize_single_inputs(self):
        """Normalize inputs for a single example."""
        if self.sampling_params is None:
            self.sampling_params = {}
        if self.rid is None:
            self.rid = uuid.uuid4().hex
        if self.return_logprob is None:
            self.return_logprob = False
        if self.logprob_start_len is None:
            self.logprob_start_len = -1
        if self.top_logprobs_num is None:
            self.top_logprobs_num = 0
        if not self.token_ids_logprob:  # covers both None and []
            self.token_ids_logprob = None

    def _normalize_batch_inputs(self):
        """Normalize inputs for a batch of examples, including parallel sampling expansion."""
        # Calculate expanded batch size
        if self.parallel_sample_num == 1:
            num = self.batch_size
        else:
            # Expand parallel_sample_num
            num = self.batch_size * self.parallel_sample_num

        # Expand input based on type
        self._expand_inputs(num)
        self._normalize_rid(num)
        self._normalize_lora_paths(num)
        self._normalize_image_data(num)
        self._normalize_video_data(num)
        self._normalize_audio_data(num)
        self._normalize_sampling_params(num)
        self._normalize_logprob_params(num)
        self._normalize_custom_logit_processor(num)
        self._normalize_extra_key(num)
        self._normalize_bootstrap_params(num)

    def _expand_inputs(self, num):
        """Expand the main inputs (text, input_ids, input_embeds) for parallel sampling."""
        if self.text is not None:
            if not isinstance(self.text, list):
                raise ValueError("Text should be a list for batch processing.")
            self.text = self.text * self.parallel_sample_num
        elif self.input_ids is not None:
            if not isinstance(self.input_ids, list) or not isinstance(
                self.input_ids[0], list
            ):
                raise ValueError(
                    "input_ids should be a list of lists for batch processing."
                )
            self.input_ids = self.input_ids * self.parallel_sample_num
        elif self.input_embeds is not None:
            if not isinstance(self.input_embeds, list):
                raise ValueError("input_embeds should be a list for batch processing.")
            self.input_embeds = self.input_embeds * self.parallel_sample_num

    def _normalize_lora_paths(self, num):
        """Normalize LoRA paths for batch processing."""
        if self.lora_path is not None:
            if isinstance(self.lora_path, str):
                self.lora_path = [self.lora_path] * num
            elif isinstance(self.lora_path, list):
                self.lora_path = self.lora_path * self.parallel_sample_num
            else:
                raise ValueError("lora_path should be a list or a string.")

    def _normalize_image_data(self, num):
        """Normalize image data for batch processing."""
        if self.image_data is None:
            self.image_data = [None] * num
        elif not isinstance(self.image_data, list):
            # Single image, convert to list of single-image lists
            self.image_data = [[self.image_data]] * num
            self.modalities = ["image"] * num
        elif isinstance(self.image_data, list):
            # Handle empty list case - treat as no images
            if len(self.image_data) == 0:
                self.image_data = [None] * num
                return

            if len(self.image_data) != self.batch_size:
                raise ValueError(
                    "The length of image_data should be equal to the batch size."
                )

            self.modalities = []
            if len(self.image_data) > 0 and isinstance(self.image_data[0], list):
                # Already a list of lists, keep as is
                for i in range(len(self.image_data)):
                    if self.image_data[i] is None or self.image_data[i] == [None]:
                        self.modalities.append(None)
                    elif len(self.image_data[i]) == 1:
                        self.modalities.append("image")
                    elif len(self.image_data[i]) > 1:
                        self.modalities.append("multi-images")
                    else:
                        # Ensure len(self.modalities) == len(self.image_data)
                        self.modalities.append(None)
                # Expand parallel_sample_num
                self.image_data = self.image_data * self.parallel_sample_num
                self.modalities = self.modalities * self.parallel_sample_num
            else:
                # List of images for a batch, wrap each in a list
                wrapped_images = [[img] for img in self.image_data]
                # Expand for parallel sampling
                self.image_data = wrapped_images * self.parallel_sample_num
                self.modalities = ["image"] * num

    def _normalize_video_data(self, num):
        """Normalize video data for batch processing."""
        if self.video_data is None:
            self.video_data = [None] * num
        elif not isinstance(self.video_data, list):
            self.video_data = [self.video_data] * num
        elif isinstance(self.video_data, list):
            self.video_data = self.video_data * self.parallel_sample_num

    def _normalize_audio_data(self, num):
        """Normalize audio data for batch processing."""
        if self.audio_data is None:
            self.audio_data = [None] * num
        elif not isinstance(self.audio_data, list):
            self.audio_data = [self.audio_data] * num
        elif isinstance(self.audio_data, list):
            self.audio_data = self.audio_data * self.parallel_sample_num

    def _normalize_sampling_params(self, num):
        """Normalize sampling parameters for batch processing."""
        if self.sampling_params is None:
            self.sampling_params = [{}] * num
        elif isinstance(self.sampling_params, dict):
            self.sampling_params = [self.sampling_params] * num
        else:  # Already a list
            self.sampling_params = self.sampling_params * self.parallel_sample_num

    def _normalize_rid(self, num):
        """Normalize request IDs for batch processing."""
        if self.rid is None:
            self.rid = [uuid.uuid4().hex for _ in range(num)]
        elif isinstance(self.rid, str):
            new_rids = [f"{self.rid}_{i}" for i in range(num)]
            self.rid = new_rids
        elif isinstance(self.rid, list):
            # Note: the length of rid shall be the same as the batch_size,
            # as the rid would be expanded for parallel sampling in tokenizer_manager
            if len(self.rid) != self.batch_size:
                raise ValueError(
                    "The specified rids length mismatch with the batch_size for batch processing."
                )
        else:
            raise ValueError("The rid should be a string or a list of strings.")

    def _normalize_logprob_params(self, num):
        """Normalize logprob-related parameters for batch processing."""

        # Helper function to normalize a parameter
        def normalize_param(param, default_value, param_name):
            if param is None:
                return [default_value] * num
            elif not isinstance(param, list):
                return [param] * num
            else:
                if self.parallel_sample_num > 1:
                    raise ValueError(
                        f"Cannot use list {param_name} with parallel_sample_num > 1"
                    )
                return param

        # Normalize each logprob parameter
        self.return_logprob = normalize_param(
            self.return_logprob, False, "return_logprob"
        )
        self.logprob_start_len = normalize_param(
            self.logprob_start_len, -1, "logprob_start_len"
        )
        self.top_logprobs_num = normalize_param(
            self.top_logprobs_num, 0, "top_logprobs_num"
        )

        # Handle token_ids_logprob specially due to its nested structure
        if not self.token_ids_logprob:  # covers both None and []
            self.token_ids_logprob = [None] * num
        elif not isinstance(self.token_ids_logprob, list):
            self.token_ids_logprob = [[self.token_ids_logprob] for _ in range(num)]
        elif not isinstance(self.token_ids_logprob[0], list):
            self.token_ids_logprob = [
                copy.deepcopy(self.token_ids_logprob) for _ in range(num)
            ]
        elif self.parallel_sample_num > 1:
            raise ValueError(
                "Cannot use list token_ids_logprob with parallel_sample_num > 1"
            )

    def _normalize_custom_logit_processor(self, num):
        """Normalize custom logit processor for batch processing."""
        if self.custom_logit_processor is None:
            self.custom_logit_processor = [None] * num
        elif not isinstance(self.custom_logit_processor, list):
            self.custom_logit_processor = [self.custom_logit_processor] * num
        elif self.parallel_sample_num > 1:
            raise ValueError(
                "Cannot use list custom_logit_processor with parallel_sample_num > 1"
            )

    def _normalize_extra_key(self, num):
        """Normalize extra_key for batch processing."""
        if self.extra_key is None:
            return
        if isinstance(self.extra_key, str):
            self.extra_key = [self.extra_key] * num
        elif isinstance(self.extra_key, list):
            if len(self.extra_key) != self.batch_size:
                raise ValueError(
                    "The length of extra_key should be equal to the batch size."
                )
            self.extra_key = self.extra_key * self.parallel_sample_num
        else:
            raise ValueError("extra_key should be a list or a string.")

    def _normalize_bootstrap_params(self, num):
        """Normalize bootstrap parameters for batch processing."""
        # Normalize bootstrap_host
        if self.bootstrap_host is None:
            self.bootstrap_host = [None] * num
        elif not isinstance(self.bootstrap_host, list):
            self.bootstrap_host = [self.bootstrap_host] * num
        elif isinstance(self.bootstrap_host, list):
            self.bootstrap_host = self.bootstrap_host * self.parallel_sample_num

        # Normalize bootstrap_port
        if self.bootstrap_port is None:
            self.bootstrap_port = [None] * num
        elif not isinstance(self.bootstrap_port, list):
            self.bootstrap_port = [self.bootstrap_port] * num
        elif isinstance(self.bootstrap_port, list):
            self.bootstrap_port = self.bootstrap_port * self.parallel_sample_num

        # Normalize bootstrap_room
        if self.bootstrap_room is None:
            self.bootstrap_room = [None] * num
        elif not isinstance(self.bootstrap_room, list):
            self.bootstrap_room = [self.bootstrap_room + i for i in range(num)]
        elif isinstance(self.bootstrap_room, list):
            self.bootstrap_room = self.bootstrap_room * self.parallel_sample_num

        # Normalize bootstrap_pair_key
        if self.bootstrap_pair_key is None:
            self.bootstrap_pair_key = [None] * num
        elif not isinstance(self.bootstrap_pair_key, list):
            self.bootstrap_pair_key = [self.bootstrap_pair_key] * num
        elif isinstance(self.bootstrap_pair_key, list):
            self.bootstrap_pair_key = self.bootstrap_pair_key * self.parallel_sample_num

    def _validate_session_params(self):
        """Validate that session parameters are properly formatted."""
        if self.session_params is not None:
            if not isinstance(self.session_params, dict) and not isinstance(
                self.session_params[0], dict
            ):
                raise ValueError("Session params must be a dict or a list of dicts.")

    def _get_positional_embed_overrides_item(
        self, i: int
    ) -> Optional[PositionalEmbeds]:
        """Extract the i-th item from positional_embed_overrides."""
        if self.positional_embed_overrides is None:
            return None
        if isinstance(self.positional_embed_overrides, PositionalEmbeds):
            return self.positional_embed_overrides
        return self.positional_embed_overrides[i]

    def __getitem__(self, i):
        # Cache sub-objects so that repeated obj[i] calls return the same instance.
        # This avoids subtle bugs where different call sites get divergent objects.
        # 中译：取批次中第 i 个请求，拆成一个独立的单请求 GenerateReqInput。
        #       用缓存保证多次 obj[i] 返回同一实例，避免不同调用点拿到不一致的对象。
        cache = self.__dict__.setdefault("_sub_obj_cache", {})
        if i in cache:
            return cache[i]
        sub = GenerateReqInput(
            text=self.text[i] if self.text is not None else None,
            input_ids=self.input_ids[i] if self.input_ids is not None else None,
            input_embeds=(
                self.input_embeds[i] if self.input_embeds is not None else None
            ),
            positional_embed_overrides=self._get_positional_embed_overrides_item(i),
            image_data=self.image_data[i],
            video_data=self.video_data[i],
            audio_data=self.audio_data[i],
            sampling_params=self.sampling_params[i],
            rid=self.rid[i],
            return_logprob=self.return_logprob[i],
            logprob_start_len=self.logprob_start_len[i],
            top_logprobs_num=self.top_logprobs_num[i],
            token_ids_logprob=self.token_ids_logprob[i],
            return_text_in_logprobs=self.return_text_in_logprobs,
            stream=self.stream,
            log_metrics=self.log_metrics,
            return_hidden_states=(
                self.return_hidden_states[i]
                if isinstance(self.return_hidden_states, list)
                else self.return_hidden_states
            ),
            return_routed_experts=self.return_routed_experts,
            routed_experts_start_len=self.routed_experts_start_len,
            return_indexer_topk=self.return_indexer_topk,
            modalities=self.modalities[i] if self.modalities else None,
            session_params=self.session_params,
            lora_path=self.lora_path[i] if self.lora_path is not None else None,
            lora_id=self.lora_id[i] if self.lora_id is not None else None,
            custom_logit_processor=(
                self.custom_logit_processor[i]
                if self.custom_logit_processor is not None
                else None
            ),
            # if `__getitem__` is called, the bootstrap_host, bootstrap_port, bootstrap_room must be a list
            bootstrap_host=(
                self.bootstrap_host[i] if self.bootstrap_host is not None else None
            ),
            bootstrap_port=(
                self.bootstrap_port[i] if self.bootstrap_port is not None else None
            ),
            bootstrap_room=(
                self.bootstrap_room[i] if self.bootstrap_room is not None else None
            ),
            bootstrap_pair_key=(
                self.bootstrap_pair_key[i]
                if self.bootstrap_pair_key is not None
                else None
            ),
            decode_tp_size=(
                self.decode_tp_size[i] if self.decode_tp_size is not None else None
            ),
            routed_dp_rank=self.routed_dp_rank,
            disagg_prefill_dp_rank=self.disagg_prefill_dp_rank,
            conversation_id=self.conversation_id,
            priority=self.priority,
            extra_key=self.extra_key[i] if self.extra_key is not None else None,
            no_logs=self.no_logs,
            custom_labels=self.custom_labels,
            return_bytes=self.return_bytes,
            return_entropy=self.return_entropy,
            return_prompt_token_ids=self.return_prompt_token_ids,
            external_trace_header=self.external_trace_header,
            http_worker_ipc=self.http_worker_ipc,
            received_time=self.received_time,
            multi_item_delimiter_indices=(
                self.multi_item_delimiter_indices[i]
                if self.multi_item_delimiter_indices is not None
                else None
            ),
        )
        cache[i] = sub
        return sub


@dataclass
class TokenizedGenerateReqInput(BaseReq):
    """中译：分词后的生成请求。由 TokenizerManager 在归一化+分词 GenerateReqInput 之后产出，
    发送给 Scheduler。相比 GenerateReqInput，它一定是「单个请求」（批次已拆开），且文本已转成
    input_ids、采样参数已是结构化的 SamplingParams 对象。是调度器实际入队处理的请求形态。
    """

    # The input text
    input_text: str  # 中译：原始输入文本（保留以便调试/日志）
    # The input token ids
    input_ids: Optional[array[int]]  # 中译：分词后的输入 token id 序列
    # The multimodal inputs
    mm_inputs: object  # 中译：处理好的多模态输入（图像/音频特征等），无则为空
    # The sampling parameters
    sampling_params: SamplingParams  # 中译：结构化采样参数对象
    # Whether to return the logprobs
    return_logprob: bool  # 中译：是否返回 logprob
    # If return logprobs, the start location in the prompt for returning logprobs.
    logprob_start_len: int  # 中译：返回 logprob 的提示词起始位置
    # If return logprobs, the number of top logprobs to return at each position.
    top_logprobs_num: int  # 中译：每位置返回的 top-k logprob 数量
    # If return logprobs, the token id to return logprob for
    token_ids_logprob: List[int]  # 中译：额外指定返回 logprob 的 token id 集合
    # Whether to stream output
    stream: bool  # 中译：是否流式输出

    # Whether to return hidden states
    return_hidden_states: bool = False  # 中译：是否返回隐藏状态

    # Whether to return captured routed experts
    return_routed_experts: bool = False  # 中译：是否返回 MoE 路由选中的专家
    # See GenerateReqInput.routed_experts_start_len.
    routed_experts_start_len: int = 0  # 中译：返回路由信息的起始位置，含义见 GenerateReqInput

    return_indexer_topk: bool = False  # 中译：是否返回稀疏注意力索引器 top-k

    # The input embeds
    input_embeds: Optional[Union[List[List[List[float]]], List[List[float]]]] = None  # 中译：直接输入的嵌入向量

    # Embedding overrides to place at specific token positions.
    positional_embed_overrides: Optional[PositionalEmbeds] = None

    # Session info for continual prompting
    session_params: Optional[SessionParams] = None

    # LoRA related
    lora_id: Optional[str] = None  # None means just use the base model
    # 中译：LoRA 适配器 id；None 表示只用基座模型、不挂 LoRA。

    # Custom logit processor for advanced sampling control. Must be a serialized instance
    # of `CustomLogitProcessor` in python/sglang/srt/sampling/custom_logit_processor.py
    # Use the processor's `to_str()` method to generate the serialized string.
    # 中译：自定义 logit 处理器（高级采样控制），必须是 CustomLogitProcessor 的序列化字符串
    #       （用其 to_str() 生成）。
    custom_logit_processor: Optional[str] = None

    # For disaggregated inference
    # 中译：PD 分离推理对接参数（含义同 GenerateReqInput，但此处为单请求标量）。
    bootstrap_host: Optional[str] = None
    bootstrap_port: Optional[int] = None
    bootstrap_room: Optional[int] = None
    bootstrap_pair_key: Optional[str] = None
    decode_tp_size: Optional[int] = None

    # Require reasoning for the request (hybrid reasoning model only)
    require_reasoning: bool = False  # 中译：强制进入推理模式（仅混合推理模型）

    # For DP routing
    routed_dp_rank: Optional[int] = None  # 中译：DP 路由指定的目标 DP rank
    # For PD disagg — hint telling decode which prefill DP worker has the KV cache
    disagg_prefill_dp_rank: Optional[int] = None  # 中译：提示 KV cache 所在的 prefill DP rank

    # Priority for the request
    priority: Optional[int] = None  # 中译：请求优先级

    # Extra key for classifying the request (e.g. cache_salt)
    extra_key: Optional[str] = None  # 中译：请求分类附加 key（如 cache_salt）

    # Routing key for routing-key schedule policy
    routing_key: Optional[str] = None

    # Whether to disallow logging for this request (e.g. due to ZDR)
    no_logs: bool = False

    # (Internal) Whether to return bytes for image generation
    return_bytes: bool = False

    # Whether to return entropy
    return_entropy: bool = False

    token_type_ids: Optional[List[int]] = None  # 中译：token 类型 id（如 BERT 类双段输入区分句子）

    need_wait_for_mm_inputs: bool = False  # 中译：是否需等待独立 encoder 算好的多模态输入
    num_items_assigned: Optional[Dict[Modality, List[int]]] = None  # 中译：各模态分配条目数
    mm_data_mooncake: Optional[List] = None  # 中译：Mooncake 传输的多模态数据句柄
    # Encoder URL snapshot frozen at tokenizer-side dispatch time so that
    # encoder_idx assignments stay consistent in the scheduler subprocess.
    # Internal IPC only.
    # 中译：tokenizer 端派发时冻结的 encoder URL 快照，保证调度器子进程中 encoder_idx 分配一致。
    #       仅用于内部 IPC。
    encoder_urls: Optional[List[str]] = None

    # Pre-computed delimiter indices for multi-item scoring
    multi_item_delimiter_indices: Optional[List[int]] = None  # 中译：多条目打分的分隔符下标

    # For observability
    # 中译：可观测性——本请求在 API server / DP 控制器阶段的耗时统计。
    time_stats: Optional[Union[APIServerReqTimeStats, DPControllerReqTimeStats]] = None


@dataclass
class BatchTokenizedGenerateReqInput(BaseBatchReq):
    """中译：把多个 TokenizedGenerateReqInput 打包成一批，一次性发给 Scheduler，减少 IPC 次数。
    提供 len/索引/迭代接口，使其可像列表一样使用。
    """

    # The batch of tokenized requests
    batch: List[TokenizedGenerateReqInput]  # 中译：批次内的分词请求列表

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, i):
        return self.batch[i]

    def __iter__(self):
        return iter(self.batch)


@dataclass
class EmbeddingReqInput(BaseReq):
    """中译：嵌入（embedding）/打分请求的原始输入对象，对应 /v1/embeddings 等接口。
    与 GenerateReqInput 类似但用于「只编码不生成」的模型；sampling_params 等字段仅为占位兼容
    （max_new_tokens 会被置 0）。也支持 cross-encoder（重排/打分）请求。
    """

    # The input prompt. It can be a single prompt or a batch of prompts.
    text: Optional[Union[List[List[str]], List[str], str]] = None  # 中译：输入文本（单条或批次）
    # The image input. It can be an image instance, file name, URL, or base64 encoded string.
    # Can be formatted as:
    # - Single image for a single request
    # - List of images (one per request in a batch)
    # - List of lists of images (multiple images per request)
    # See also python/sglang/srt/utils.py:load_image for more details.
    image_data: Optional[MultimodalDataInputFormat] = None
    # The video input. Like image data, it can be a file name, a url, or base64 encoded string.
    video_data: Optional[MultimodalDataInputFormat] = None
    # The audio input. Like image data, it can be a file name, a url, or base64 encoded string.
    audio_data: Optional[MultimodalDataInputFormat] = None
    # The token ids for text; one can either specify text or input_ids.
    input_ids: Optional[Union[List[List[int]], List[int]]] = None  # 中译：分词后输入；与 text 二选一
    # Placeholder token ID used to locate embedding override positions in input token IDs.
    # 中译：占位 token id，用于在输入 token 序列中定位需要被嵌入覆盖（embed override）的位置。
    embed_override_token_id: Optional[int] = None
    # Unresolved embedding overrides: per-input list of tensors.
    # Position resolution happens in the tokenizer manager after tokenization.
    # Shape: [num_inputs][num_replacements] where each entry is a torch.Tensor of [hidden_size].
    # Per-input entry may be None when only some inputs in a batch need overrides.
    # Runtime type: Optional[List[Optional[List[torch.Tensor]]]]
    # Typed as Any to avoid Pydantic/FastAPI schema errors (contains torch.Tensor).
    # 中译：未解析的嵌入覆盖：逐输入的张量列表，形状 [输入数][替换数]，每项为 [hidden_size] 的张量。
    #       具体替换位置在 TokenizerManager 分词后才解析。因含 torch.Tensor 故标注为 Any。
    embed_overrides: Any = None
    # Resolved embedding overrides with positions (set by tokenizer manager or score mixin).
    # Runtime type: Optional[Union[PositionalEmbeds, List[Optional[PositionalEmbeds]]]]
    # 中译：已解析（带位置）的嵌入覆盖，由 TokenizerManager 或打分逻辑填充。
    positional_embed_overrides: Any = None
    # Dummy sampling params for compatibility
    sampling_params: Optional[Union[List[Dict], Dict]] = None  # 中译：占位采样参数（兼容用，编码不采样）
    # Dummy input embeds for compatibility
    input_embeds: Optional[Union[List[List[List[float]]], List[List[float]]]] = None  # 中译：占位字段（兼容用）
    # Whether to log metrics for this request (e.g. health_generate calls do not log metrics)
    log_metrics: bool = True  # 中译：是否记录指标
    # The modalities of the image data [image, multi-images, video]
    modalities: Optional[List[str]] = None  # 中译：图像数据的模态标识
    # For cross-encoder requests
    is_cross_encoder_request: bool = False  # 中译：是否为 cross-encoder（重排/打分）请求
    # Priority for the request
    priority: Optional[int] = None  # 中译：请求优先级
    # Routing key for routing-key schedule policy
    routing_key: Optional[str] = None  # 中译：routing-key 调度策略所用路由 key

    # For background responses (OpenAI responses API)
    background: bool = False

    # Propagates trace context via Engine.encode/async_encode
    external_trace_header: Optional[Dict] = None
    received_time: Optional[float] = None

    # The number of dimensions the resulting output embeddings should have. It is applicable for Matryoshka Embeddings.
    # 中译：输出嵌入的目标维度数（适用于 Matryoshka 嵌入——可按需截断到更短维度）。
    dimensions: Optional[int] = None

    # The path to the LoRA adaptors
    lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None  # 中译：LoRA 适配器路径
    # The uid of LoRA adaptors, should be initialized by tokenizer manager
    lora_id: Optional[Union[List[Optional[str]], Optional[str]]] = None  # 中译：LoRA 适配器 id

    # Whether to return pooled hidden states (pre-head transformer output)
    # 中译：是否返回池化后的隐藏状态（pooling head 之前的 transformer 输出）。
    return_pooled_hidden_states: bool = False

    # Whether to return prompt token IDs without computing logprobs
    return_prompt_token_ids: bool = False  # 中译：是否仅返回提示词 token id

    # Pre-computed delimiter indices for multi-item scoring.
    # Batch-level: List[List[int]] (one per request). After __getitem__: List[int].
    # 中译：多条目打分预计算的分隔符下标（批次级嵌套列表，取单条后变扁平列表）。
    multi_item_delimiter_indices: Optional[Union[List[List[int]], List[int]]] = None

    def normalize_batch_and_arguments(self):
        # 中译：归一化批次与参数。要求 text/input_ids/image 至少给其一且 text 与 input_ids 不并存；
        #       推导 batch_size，填默认 rid/采样参数，并把 max_new_tokens 强制置 0（编码不生成）。
        # at least one of text, input_ids, or image should be provided
        if self.text is None and self.input_ids is None and self.image_data is None:
            raise ValueError(
                "At least one of text, input_ids, or image should be provided"
            )

        # text and input_ids cannot be provided at the same time
        if self.text is not None and self.input_ids is not None:
            raise ValueError("text and input_ids cannot be provided at the same time")

        # Derive the batch size
        self.batch_size = 0
        self.is_single = True

        # check the batch size of text
        if self.text is not None:
            if isinstance(self.text, list):
                self.batch_size += len(self.text)
                self.is_single = False
            else:
                self.batch_size += 1

        # check the batch size of input_ids
        if self.input_ids is not None:
            if isinstance(self.input_ids[0], list):
                self.batch_size += len(self.input_ids)
                self.is_single = False
            else:
                self.batch_size += 1

        # Fill in default arguments
        if self.is_single:
            if self.rid is None:
                self.rid = uuid.uuid4().hex
            if self.sampling_params is None:
                self.sampling_params = {}
            self.sampling_params["max_new_tokens"] = 0
        else:
            if self.rid is None:
                self.rid = [uuid.uuid4().hex for _ in range(self.batch_size)]
            else:
                assert isinstance(self.rid, list), "The rid should be a list."

            if self.sampling_params is None:
                self.sampling_params = [{}] * self.batch_size
            elif isinstance(self.sampling_params, dict):
                self.sampling_params = [self.sampling_params] * self.batch_size
            for i in range(self.batch_size):
                self.sampling_params[i]["max_new_tokens"] = 0

            self._normalize_lora_paths(self.batch_size)

        self._validate_rid_uniqueness()

    def _normalize_lora_paths(self, num):
        """Normalize LoRA paths for batch processing."""
        if self.lora_path is not None:
            if isinstance(self.lora_path, str):
                self.lora_path = [self.lora_path] * num
            elif isinstance(self.lora_path, list):
                if len(self.lora_path) != num:
                    raise ValueError(
                        f"lora_path list length ({len(self.lora_path)}) must match batch size ({num})"
                    )
            else:
                raise ValueError("lora_path should be a list or a string.")

    def contains_mm_input(self) -> bool:
        return (
            has_valid_data(self.image_data)
            or has_valid_data(self.video_data)
            or has_valid_data(self.audio_data)
        )

    def _get_positional_embed_overrides_item(
        self, i: int
    ) -> Optional[PositionalEmbeds]:
        """Extract the i-th item from positional_embed_overrides."""
        if self.positional_embed_overrides is None:
            return None
        if isinstance(self.positional_embed_overrides, PositionalEmbeds):
            return self.positional_embed_overrides
        return self.positional_embed_overrides[i]

    def __getitem__(self, i):
        # Cache sub-objects so that repeated obj[i] calls return the same instance.
        cache = self.__dict__.setdefault("_sub_obj_cache", {})
        if i in cache:
            return cache[i]

        if self.is_cross_encoder_request:
            sub = EmbeddingReqInput(
                text=[self.text[i]] if self.text is not None else None,
                positional_embed_overrides=self._get_positional_embed_overrides_item(i),
                sampling_params=self.sampling_params[i],
                rid=self.rid[i],
                lora_path=self.lora_path[i] if self.lora_path is not None else None,
                lora_id=self.lora_id[i] if self.lora_id is not None else None,
                is_cross_encoder_request=True,
                http_worker_ipc=self.http_worker_ipc,
                return_pooled_hidden_states=self.return_pooled_hidden_states,
                return_prompt_token_ids=self.return_prompt_token_ids,
                multi_item_delimiter_indices=(
                    self.multi_item_delimiter_indices[i]
                    if self.multi_item_delimiter_indices is not None
                    else None
                ),
            )
        else:
            sub = EmbeddingReqInput(
                text=self.text[i] if self.text is not None else None,
                input_ids=self.input_ids[i] if self.input_ids is not None else None,
                embed_override_token_id=self.embed_override_token_id,
                embed_overrides=(
                    self.embed_overrides[i]
                    if self.embed_overrides is not None
                    else None
                ),
                positional_embed_overrides=self._get_positional_embed_overrides_item(i),
                image_data=self.image_data[i] if self.image_data is not None else None,
                audio_data=self.audio_data[i] if self.audio_data is not None else None,
                video_data=self.video_data[i] if self.video_data is not None else None,
                sampling_params=self.sampling_params[i],
                rid=self.rid[i],
                lora_path=self.lora_path[i] if self.lora_path is not None else None,
                lora_id=self.lora_id[i] if self.lora_id is not None else None,
                external_trace_header=self.external_trace_header,
                dimensions=self.dimensions,
                http_worker_ipc=self.http_worker_ipc,
                received_time=self.received_time,
                return_pooled_hidden_states=self.return_pooled_hidden_states,
                return_prompt_token_ids=self.return_prompt_token_ids,
                multi_item_delimiter_indices=(
                    self.multi_item_delimiter_indices[i]
                    if self.multi_item_delimiter_indices is not None
                    else None
                ),
            )
        cache[i] = sub
        return sub


@dataclass
class TokenizedEmbeddingReqInput(BaseReq):
    """中译：分词后的嵌入请求，由 TokenizerManager 产出并发往 Scheduler，是调度器处理的嵌入请求形态。"""

    # The input text
    input_text: str  # 中译：原始输入文本
    # The input token ids
    input_ids: array[int]  # 中译：分词后的 token id 序列
    # The image inputs
    image_inputs: dict  # 中译：处理好的图像输入
    # The token type ids
    token_type_ids: List[int]  # 中译：token 类型 id（双段输入区分）
    # Dummy sampling params for compatibility
    sampling_params: SamplingParams  # 中译：占位采样参数（兼容用）
    # Embedding overrides to place at specific token positions.
    positional_embed_overrides: Optional[PositionalEmbeds] = None
    # For DP routing
    routed_dp_rank: Optional[int] = None
    # Priority for the request
    priority: Optional[int] = None
    # The number of dimensions the resulting output embeddings should have. It is applicable for Matryoshka Embeddings.
    dimensions: Optional[int] = None

    # LoRA related
    lora_id: Optional[str] = None  # None means just use the base model
    # Pre-computed delimiter indices for multi-item scoring
    multi_item_delimiter_indices: Optional[List[int]] = None
    # For observability
    time_stats: Optional[Union[APIServerReqTimeStats, DPControllerReqTimeStats]] = None

    # Whether to return pooled hidden states (pre-head transformer output)
    return_pooled_hidden_states: bool = False


@dataclass
class BatchTokenizedEmbeddingReqInput(BaseBatchReq):
    """中译：把多个 TokenizedEmbeddingReqInput 打包成一批发给 Scheduler，减少 IPC 次数。"""

    # The batch of tokenized embedding requests
    batch: List[TokenizedEmbeddingReqInput]  # 中译：批次内的分词嵌入请求列表

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, i):
        return self.batch[i]

    def __iter__(self):
        return iter(self.batch)


@dataclass
class BatchTokenIDOutput(BaseBatchReq, SpeculativeDecodingMetricsMixin):
    """中译：调度器（Scheduler）发给去 token 化管理器（DetokenizerManager）的「批次 token id 输出」。

    是一个面向「一个批次」的结构：每个字段都是 List，长度等于批次内请求数，第 i 项对应第 i 个请求。
    DetokenizerManager 拿到后对 token id 做增量解码，再组装成 BatchStrOutput 继续下发。
    继承：BaseBatchReq（提供 rids / http_worker_ipcs 等批次公共字段）、
          SpeculativeDecodingMetricsMixin（提供推测解码的统计字段，如 spec_verify_ct 等）。
    """

    # The finish reason
    # 中译：每个请求的结束原因（如正常停止/达到长度上限/命中 stop 等）；None 表示尚未结束（流式中）。
    finished_reasons: List[BaseFinishReason]
    # For incremental decoding
    # 中译：以下三项供「增量解码」使用（参见 detokenizer_manager.DecodeStatus）。
    decoded_texts: List[str]  # 该请求已经解码出的文本（首包时作为初始值）
    decode_ids: List[array[int]]  # 本次新增需要解码的 token id（detokenizer 会追加到已有状态）
    read_offsets: List[int]  # 初始的读取偏移（token 下标），用于初始化 read_offset
    # Only used when `--skip-tokenizer-init` is on
    # 中译：仅在 --skip-tokenizer-init 时使用：直接输出原始 token id（不做解码）。
    output_ids: Optional[List[array[int]]]
    # Detokenization configs
    # 中译：去 token 化配置（逐请求）。
    skip_special_tokens: List[bool]  # 解码时是否跳过特殊 token
    spaces_between_special_tokens: List[bool]  # 特殊 token 之间是否加空格
    no_stop_trim: List[bool]  # 是否不裁剪命中的停止符（保留 stop 本身）

    # Token counts
    # 中译：各类 token 计数（逐请求）。
    prompt_tokens: List[int]  # 提示词（输入）token 数
    reasoning_tokens: List[int]  # 推理（thinking）阶段的 token 数
    completion_tokens: List[int]  # 生成（输出）token 数
    cached_tokens: List[int]  # 命中缓存（复用已有 KV）的 token 数

    # Logprobs
    # 中译：对数概率（logprobs）相关字段。val 为对数概率值，idx 为对应 token 下标；
    #       input_* 为输入部分，output_* 为输出部分；top_* 为每步 top-k 候选的概率。
    input_token_logprobs_val: List[float]  # 输入 token 的对数概率值
    input_token_logprobs_idx: List[int]  # 输入 token 的下标
    output_token_logprobs_val: List[float]  # 输出 token 的对数概率值
    output_token_logprobs_idx: List[int]  # 输出 token 的下标
    input_top_logprobs_val: List[List]  # 输入侧每步 top-k 概率值
    input_top_logprobs_idx: List[List]  # 输入侧每步 top-k token 下标
    output_top_logprobs_val: List[List]  # 输出侧每步 top-k 概率值
    output_top_logprobs_idx: List[List]  # 输出侧每步 top-k token 下标
    input_token_ids_logprobs_val: List[List]  # 指定输入 token id 集合的对数概率值
    input_token_ids_logprobs_idx: List[List]  # 对应的 token id
    output_token_ids_logprobs_val: List[List]  # 指定输出 token id 集合的对数概率值
    output_token_ids_logprobs_idx: List[List]  # 对应的 token id
    output_token_entropy_val: List[float]  # 输出每步分布的熵（衡量不确定性）

    # Hidden states
    # 中译：按需返回的隐藏状态（如用于推测解码/下游嵌入等）。
    output_hidden_states: List[List[float]]

    # Per-request routed experts (input + output tokens), shape
    # (token, layer, top_k). DetokenizerManager encodes to base64 into
    # BatchStrOutput; on the skip_tokenizer_init path the scheduler sends this
    # straight to TokenizerManager, which encodes on demand.
    # 中译：逐请求的「路由专家」（含输入+输出 token），形状为 (token, layer, top_k)。
    #       DetokenizerManager 会把它编码为 base64 放入 BatchStrOutput；
    #       在 skip_tokenizer_init 路径上，调度器直接发给 TokenizerManager，后者按需编码。
    routed_experts: List[Optional[torch.Tensor]]

    # 中译：逐请求的索引器 top-k（如 DeepSeek 稀疏注意力选中的 token），同样会被编码为 base64。
    indexer_topk: List[Optional[torch.Tensor]]

    # The information of placeholder tokens (e.g., image token)
    # idx is the index of the token in the prompt after expansion.
    # val is the length of padded tokens after expansion.
    # 中译：占位 token（如图像 token）的信息。
    #       idx 是扩展后该 token 在提示词中的位置；val 是扩展后填充 token 的长度。
    placeholder_tokens_idx: List[Optional[List[int]]]
    placeholder_tokens_val: List[Optional[List[int]]]

    # Number of times each request was retracted.
    # 中译：每个请求被「回撤（retract）」的次数。显存不足时请求可能被换出重跑。
    retraction_counts: List[int]

    # The trainer step id. Used to know which step's weights are used for sampling.
    # 中译：训练器的 step id（用于 RL 等场景），用来知道本次采样用的是哪一步的权重。
    token_steps: List[List[int]] = None

    # Load for DP balance
    # 中译：用于 DP（数据并行）负载均衡的负载信息。
    load: GetLoadsReqOutput = None
    # Customized info
    # 中译：自定义附加信息（按 key 对应逐请求的值列表）。
    customized_info: Optional[Dict[str, List[Any]]] = None
    # Detailed breakdown of cached tokens by source (device/host/storage)
    # 中译：缓存 token 按来源（显存/主机内存/外部存储）的细分统计。
    cached_tokens_details: Optional[List[Optional[Dict[str, Any]]]] = None
    # DP rank of the scheduler that processed each request
    # 中译：处理每个请求的调度器所在的 DP rank。
    dp_ranks: Optional[List[int]] = None

    # For observability
    # 中译：可观测性——每个请求在调度器各阶段的耗时统计。
    time_stats: Optional[List[SchedulerReqTimeStats]] = None


@dataclass
class BatchStrOutput(BaseBatchReq, SpeculativeDecodingMetricsMixin):
    """中译：DetokenizerManager 解码完成后发给 TokenizerManager 的「批次字符串输出」。

    结构与 BatchTokenIDOutput 大体对应，但核心区别是已含解码好的文本 output_strs，且
    routed_experts / indexer_topk 已被编码为 base64 字符串（而非张量）。各字段均为逐请求列表。
    """

    # The finish reason
    finished_reasons: List[dict]  # 中译：每个请求的结束原因
    # The output decoded strings
    output_strs: List[str]  # 中译：解码后的输出文本（本类核心字段，流式时为本次增量文本）
    # The token ids
    output_ids: Optional[List[int]]  # 中译：输出 token id（skip_tokenizer_init 等场景使用）

    # Token counts
    # 中译：各类 token 计数（逐请求）。
    prompt_tokens: List[int]  # 提示词 token 数
    completion_tokens: List[int]  # 生成 token 数
    reasoning_tokens: List[int]  # 推理（thinking）阶段 token 数
    cached_tokens: List[int]  # 命中缓存的 token 数

    # Logprobs
    input_token_logprobs_val: List[float]
    input_token_logprobs_idx: List[int]
    output_token_logprobs_val: List[float]
    output_token_logprobs_idx: List[int]
    input_top_logprobs_val: List[List]
    input_top_logprobs_idx: List[List]
    output_top_logprobs_val: List[List]
    output_top_logprobs_idx: List[List]
    input_token_ids_logprobs_val: List[List]
    input_token_ids_logprobs_idx: List[List]
    output_token_ids_logprobs_val: List[List]
    output_token_ids_logprobs_idx: List[List]
    output_token_entropy_val: List[float]

    # Hidden states
    output_hidden_states: List[List[float]]

    # Per-request routed experts, base64-encoded by DetokenizerManager off the
    # tokenizer hot path. Underlying tensor shape is (token, layer, top_k);
    # see BatchTokenIDOutput.routed_experts.
    # 中译：逐请求的路由专家，已由 DetokenizerManager 在热路径外编码为 base64 字符串；
    #       底层张量形状 (token, layer, top_k)，参见 BatchTokenIDOutput.routed_experts。
    routed_experts: List[Optional[str]]

    indexer_topk: List[Optional[str]]  # 中译：逐请求索引器 top-k（已 base64 编码）

    # The information of placeholder tokens (e.g., image token)
    # idx is the index of the token in the prompt after expansion.
    # val is the length of padded tokens after expansion.
    # 中译：占位 token（如图像 token）信息：idx 为扩展后位置，val 为扩展后填充长度。
    placeholder_tokens_idx: List[Optional[List[int]]]
    placeholder_tokens_val: List[Optional[List[int]]]

    # Number of times each request was retracted.
    retraction_counts: List[int]  # 中译：每个请求被回撤（retract）的次数

    # The trainer step id. Used to know which step's weights are used for sampling.
    token_steps: List[List[int]] = None  # 中译：训练器 step id（RL 场景，标识本次采样所用权重的步）

    # Load for DP balance
    load: GetLoadsReqOutput = None  # 中译：DP 负载均衡用的负载信息

    # Customized info
    customized_info: Optional[Dict[str, List[Any]]] = None  # 中译：自定义附加信息
    # Detailed breakdown of cached tokens by source (device/host/storage)
    cached_tokens_details: Optional[List[Optional[Dict[str, Any]]]] = None  # 中译：缓存 token 按来源细分
    # DP rank of the scheduler that processed each request
    dp_ranks: Optional[List[int]] = None  # 中译：处理各请求的调度器所在 DP rank

    # For observability
    time_stats: Optional[List[SchedulerReqTimeStats]] = None  # 中译：各阶段耗时统计


@dataclass
class BatchEmbeddingOutput(BaseBatchReq):
    """中译：嵌入模型的批次输出。由 Scheduler 经 DetokenizerManager（原样透传、无需解码）送回
    TokenizerManager。核心字段是 embeddings（每请求一个向量或稀疏 dict）。
    """

    # The finish reason
    finished_reasons: List[BaseFinishReason]  # 中译：每个请求的结束原因
    # The output embedding
    embeddings: Union[List[List[float]], List[Dict[int, float]]]  # 中译：输出嵌入（稠密向量或稀疏 dict）
    # Token counts
    prompt_tokens: List[int]  # 中译：提示词 token 数
    cached_tokens: List[int]  # 中译：命中缓存的 token 数
    # Placeholder token info
    placeholder_tokens_idx: List[Optional[List[int]]]  # 中译：占位 token 位置
    placeholder_tokens_val: List[Optional[List[int]]]  # 中译：占位 token 填充长度

    # Number of times each request was retracted.
    retraction_counts: List[int]  # 中译：每个请求被回撤的次数
    # Detailed breakdown of cached tokens by source (device/host/storage)
    cached_tokens_details: Optional[List[Optional[Dict[str, Any]]]] = None  # 中译：缓存 token 按来源细分

    # For observability
    time_stats: Optional[List[SchedulerReqTimeStats]] = None  # 中译：各阶段耗时统计

    # Optional pooled hidden states (pre-head transformer output).
    # Sent as a single stacked tensor to minimize pickle overhead.
    # 中译：可选的池化隐藏状态（pooling head 之前的输出）。整批堆叠成单个张量发送，减少 pickle 开销。
    pooled_hidden_states: Optional[
        Union[List[Optional[torch.Tensor]], torch.Tensor]
    ] = None


@dataclass
class ClearHiCacheReqInput(BaseReq):
    # 中译：清空分层缓存（HiCache，多级 KV 缓存）的请求。
    pass


@dataclass
class ClearHiCacheReqOutput(BaseReq):
    # 中译：清空 HiCache 的应答，success 表示是否成功。
    success: bool


@dataclass
class FlushCacheReqInput(BaseReq):
    # 中译：刷新（清空）前缀缓存（prefix cache）的请求；timeout_s 为可选超时。
    timeout_s: Optional[float] = None


@dataclass
class FlushCacheReqOutput(BaseReq):
    # 中译：刷新缓存的应答。
    success: bool
    message: str = ""


@dataclass
class AddExternalCorpusReqInput(BaseReq):
    # 中译：向引擎添加「外部语料（external corpus）」的请求，用于把外部文档预置进缓存以加速命中。
    corpus_id: Optional[str] = None  # 语料 id
    file_path: Optional[str] = None  # 语料文件路径
    documents: Optional[List[str]] = None  # 直接给定的文档文本列表
    token_chunks: Optional[List[List[int]]] = None  # 已分词的 token 块


@dataclass
class AddExternalCorpusReqOutput(BaseReq):
    # 中译：添加外部语料的应答。
    success: bool
    corpus_id: str = ""
    message: str = ""
    loaded_token_count: int = 0  # 实际载入的 token 数


@dataclass
class RemoveExternalCorpusReqInput(BaseReq):
    # 中译：移除指定外部语料的请求。
    corpus_id: str


@dataclass
class RemoveExternalCorpusReqOutput(BaseReq):
    # 中译：移除外部语料的应答。
    success: bool
    message: str = ""


@dataclass
class ListExternalCorporaReqInput(BaseReq):
    # 中译：列出当前所有外部语料的请求。
    pass


@dataclass
class ListExternalCorporaReqOutput(BaseReq):
    # 中译：列出外部语料的应答，corpus_token_counts 为各语料 id 到其 token 数的映射。
    success: bool
    corpus_token_counts: Dict[str, int] = field(default_factory=dict)
    message: str = ""


@dataclass
class AttachHiCacheStorageReqInput(BaseReq):
    """Dynamically attach (enable) HiCache storage backend at runtime.

    Note: `hicache_storage_backend_extra_config_json` is a JSON string. It may contain both:
    - backend-specific configs (e.g., mooncake master address)
    - prefetch-related knobs (prefetch_threshold, prefetch_timeout_*, hicache_storage_pass_prefix_keys)

    中译：运行时动态挂载（启用）HiCache 存储后端的请求。
          extra_config_json 是一段 JSON 字符串，可同时包含后端专属配置（如 mooncake master 地址）
          与预取相关参数（prefetch_threshold、prefetch_timeout_* 等）。
    """

    hicache_storage_backend: str  # 中译：存储后端类型
    hicache_storage_backend_extra_config_json: Optional[str] = None  # 中译：后端额外配置（JSON 字符串）
    hicache_storage_prefetch_policy: Optional[str] = None  # 中译：预取策略（best_effort/wait_complete/timeout）
    hicache_write_policy: Optional[str] = None  # 中译：写回策略（write_back/write_through/...）

    def __post_init__(self):
        if self.hicache_storage_prefetch_policy is None:
            pass
        else:
            allowed = ["best_effort", "wait_complete", "timeout"]
            if self.hicache_storage_prefetch_policy not in allowed:
                raise ValueError(
                    f"Invalid hicache_storage_prefetch_policy: {self.hicache_storage_prefetch_policy!r}. "
                    f"Expected one of {allowed}."
                )

        if self.hicache_write_policy is None:
            return
        allowed = ["write_back", "write_through", "write_through_selective"]
        if self.hicache_write_policy not in allowed:
            raise ValueError(
                f"Invalid hicache_write_policy: {self.hicache_write_policy!r}. "
                f"Expected one of {allowed}."
            )


@dataclass
class AttachHiCacheStorageReqOutput(BaseReq):
    success: bool
    message: str = ""


@dataclass
class DetachHiCacheStorageReqInput(BaseReq):
    """Dynamically detach (disable) HiCache storage backend at runtime.

    中译：运行时动态卸载（禁用）HiCache 存储后端的请求。
    """

    pass


@dataclass
class DetachHiCacheStorageReqOutput(BaseReq):
    success: bool
    message: str = ""


@dataclass
class PauseGenerationReqInput(BaseReq):
    """
    Note that the PauseGenerationRequests is only supported in SGLang Server.
    abort: Abort and return all requests currently being processed.

    in_place: Pause the scheduler's event_loop from performing inference;
            only non-inference requests (e.g., control commands) will be handled.
            The requests in the engine will be paused and stay in the event_loop,
            then continue generation after continue_generation with the old kv cache.
            Note: In 'inplace' mode, flush_cache will fail if there are any requests
            in the running_batch.

    retract: Pause the scheduler's event loop from performing inference;
            only non-inference requests will be handled, and all currently running
            requests will be retracted back to the waiting_queue.
            Note: The KV cache can be flushed in this mode and will be automatically
            recomputed after continue_generation.

    中译：暂停生成的请求（仅 SGLang Server 支持），mode 三选一：
          - abort：中止并返回当前所有正在处理的请求。
          - in_place：暂停事件循环的推理，仅处理控制类请求；运行中的请求原地保留在循环内，
            continue_generation 后用旧 KV cache 继续。注意此模式下若有运行中请求，flush_cache 会失败。
          - retract：暂停推理，并把所有运行中请求回撤到等待队列。此模式可清空 KV cache，
            continue_generation 后会自动重算。
    """

    mode: Literal["abort", "retract", "in_place"] = "abort"  # 中译：暂停模式

    def __post_init__(self):
        allowed = ["abort", "retract", "in_place"]
        if self.mode not in allowed:
            raise ValueError(
                f"Invalid mode: {self.mode!r}. " f"Expected one of {allowed}."
            )


@dataclass
class ContinueGenerationReqInput(BaseReq):
    # Call torch.cuda.empty_cache() before un-pausing. Returns blocks
    # cached by the PyTorch allocator (left over from transient allocs
    # during post-weight-update processing) back to the driver before
    # inference resumes, with no race against active streams. Set to
    # False to skip the empty_cache call.
    # 中译：恢复生成（解除暂停）的请求。torch_empty_cache=True 时在解除前先调用
    #       torch.cuda.empty_cache()，把权重更新后处理期残留的临时分配块归还给驱动，
    #       且不与活跃 stream 竞争；置 False 可跳过该调用。
    torch_empty_cache: bool = True


@dataclass
class TokenizerWorkerRegistration:
    """Sent by each TokenizerWorker on startup to register its IPC name with the router.

    中译：多 tokenizer 模式下，每个 TokenizerWorker 启动时向路由器注册自己 IPC 名的消息。
    """

    worker_ipc_name: str  # 中译：该 worker 的 IPC 名


@dataclass
class PauseContinueBroadcast:
    """Broadcast from router to all workers to set is_pause state.

    中译：路由器向所有 worker 广播暂停/恢复状态的消息。
    """

    is_pause: bool  # 中译：True 暂停 / False 恢复


@dataclass
class UpdateWeightFromDiskReqInput(BaseReq):
    """中译：从磁盘加载新权重、热更新模型的请求（如 RL 训练中周期性同步权重）。"""

    # The model path with the new weights
    model_path: str  # 中译：新权重所在的模型路径
    # The format to load the weights
    load_format: Optional[str] = None  # 中译：权重加载格式
    # Whether to abort all requests before updating weights
    abort_all_requests: bool = False  # 中译：更新前是否中止所有请求
    # Optional: Update weight version along with weights
    weight_version: Optional[str] = None  # 中译：可选，随权重一起更新版本号
    # Whether to update weights asynchronously
    is_async: bool = False  # 中译：是否异步更新
    # Whether to call torch.cuda.empty_cache() during flush
    torch_empty_cache: bool = False  # 中译：刷新时是否调用 empty_cache
    # Whether to keep the scheduler paused after weight update
    keep_pause: bool = False  # 中译：更新后是否保持调度器暂停
    # Whether to recapture cuda graph after weight update
    recapture_cuda_graph: bool = False  # 中译：更新后是否重新捕获 CUDA graph
    # The trainer step id. Used to know which step's weights are used for sampling.
    token_step: int = 0  # 中译：训练器 step id（标识本次采样所用权重的步）
    # Whether to flush the cache after updating weights
    flush_cache: bool = True  # 中译：更新后是否刷新缓存
    # Tensor metadata
    manifest: Optional[Dict[str, Any]] = None  # 中译：张量元信息清单


@dataclass
class UpdateWeightFromDiskReqOutput(BaseReq):
    # 中译：从磁盘更新权重的应答。
    success: bool
    message: str
    # Number of paused requests during weight sync.
    num_paused_requests: Optional[int] = 0  # 中译：权重同步期间被暂停的请求数


@dataclass
class UpdateWeightsFromDistributedReqInput(BaseReq):
    """中译：从分布式进程组接收新权重做热更新的请求（训练进程通过通信组把权重广播给推理进程）。"""

    names: List[str]  # 中译：待更新的参数名列表
    dtypes: List[str]  # 中译：各参数的 dtype
    shapes: List[List[int]]  # 中译：各参数的形状
    # The group name
    group_name: str = "weight_update_group"  # 中译：通信组名
    # Whether to flush the cache after updating weights
    flush_cache: bool = True
    # Whether to abort all requests before updating weights
    abort_all_requests: bool = False
    # Optional: Update weight version along with weights
    weight_version: Optional[str] = None
    # Optional format specification for loading
    load_format: Optional[str] = None
    # Whether to call torch.cuda.empty_cache() during flush
    torch_empty_cache: bool = False


@dataclass
class UpdateWeightsFromDistributedReqOutput(BaseReq):
    # 中译：从分布式组更新权重的应答。
    success: bool
    message: str


@dataclass
class UpdateWeightsFromTensorReqInput(BaseReq):
    """Update model weights from tensor input.

    - Tensors are serialized for transmission
    - Data is structured in JSON for easy transmission over HTTP

    中译：直接以张量形式更新模型权重的请求。张量被序列化以便传输，数据组织成 JSON 方便走 HTTP。
    """

    serialized_named_tensors: List[Union[str, bytes]]  # 中译：序列化后的「命名张量」列表
    # Optional format specification for loading
    load_format: Optional[str] = None
    # Whether to flush the cache after updating weights
    flush_cache: bool = True
    # Whether to abort all requests before updating weights
    abort_all_requests: bool = False
    # Optional: Update weight version along with weights
    weight_version: Optional[str] = None
    # Optional: Determine whether to disable updating the draft model
    disable_draft_model: Optional[bool] = None  # 中译：是否不更新草稿模型（推测解码的 draft model）
    # Whether to call torch.cuda.empty_cache() during flush
    torch_empty_cache: bool = False


@dataclass
class UpdateWeightsFromTensorReqOutput(BaseReq):
    # 中译：从张量更新权重的应答。
    success: bool
    message: str


@dataclass
class InitWeightsSendGroupForRemoteInstanceReqInput(BaseReq):
    """中译：为「向远程实例发送权重」初始化通信组的请求（用于把本实例权重送给另一推理实例）。"""

    # The master address
    master_address: str  # 中译：通信组 master 地址
    # The ports for each rank's communication group
    ports: str  # 中译：各 rank 通信组的端口
    # The rank in the communication group
    group_rank: int  # 中译：本进程在通信组中的 rank
    # The world size
    world_size: int  # 中译：通信组总进程数
    # The group name
    group_name: str = "weight_send_group"
    # The backend
    backend: str = "nccl"  # 中译：通信后端（默认 nccl）


# Now UpdateWeightsFromIPCReqInput and UpdateWeightsFromIPCReqOutput
# are only used by Checkpoint Engine (https://github.com/MoonshotAI/checkpoint-engine)
@dataclass
class UpdateWeightsFromIPCReqInput(BaseReq):
    """中译：通过 IPC（共享内存/ZMQ 句柄）更新权重的请求。目前仅供 Checkpoint Engine 使用。"""

    # ZMQ socket paths for each device UUID
    zmq_handles: Dict[str, str]  # 中译：各设备 UUID 对应的 ZMQ 句柄路径
    # Whether to flush cache after weight update
    flush_cache: bool = True
    # Optional: Update weight version along with weights
    weight_version: Optional[str] = None
    # Whether to call torch.cuda.empty_cache() during flush
    torch_empty_cache: bool = False


@dataclass
class UpdateWeightsFromIPCReqOutput(BaseReq):
    # 中译：通过 IPC 更新权重的应答。
    success: bool
    message: str


@dataclass
class InitWeightsSendGroupForRemoteInstanceReqOutput(BaseReq):
    # 中译：初始化权重发送组的应答。
    success: bool
    message: str


@dataclass
class SendWeightsToRemoteInstanceReqInput(BaseReq):
    """中译：把本实例权重发送给远程实例的请求（配合上面的发送组使用）。"""

    # The master address
    master_address: str
    # The ports for each rank's communication group
    ports: str
    # The group name
    group_name: str = "weight_send_group"


@dataclass
class SendWeightsToRemoteInstanceReqOutput(BaseReq):
    # 中译：发送权重到远程实例的应答。
    success: bool
    message: str


@dataclass
class UpdateExpertBackupReq(BaseReq):
    # 中译：更新专家（MoE expert）备份的请求。
    pass


@dataclass
class BackupDramReq(BaseReq):
    # 中译：把权重备份到 DRAM（主机内存）的请求。
    rank: int  # 中译：发起备份的 rank
    weight_pointer_map: Dict[str, Any]  # 中译：参数名到权重指针的映射
    session_id: str  # 中译：备份会话 id
    buffer_size: int  # 中译：缓冲区大小


@dataclass
class InitWeightsUpdateGroupReqInput(BaseReq):
    """中译：初始化「权重更新通信组」的请求（训练进程与推理进程组成 NCCL 组以同步权重）。"""

    # The master address
    master_address: str
    # The master port
    master_port: int
    # The rank offset
    rank_offset: int  # 中译：rank 偏移（推理进程在全局组中的起始 rank）
    # The world size
    world_size: int
    # The group name
    group_name: str = "weight_update_group"
    # The backend
    backend: str = "nccl"


@dataclass
class InitWeightsUpdateGroupReqOutput(BaseReq):
    # 中译：初始化权重更新组的应答。
    success: bool
    message: str


@dataclass
class DestroyWeightsUpdateGroupReqInput(BaseReq):
    # 中译：销毁权重更新通信组的请求。
    group_name: str = "weight_update_group"


@dataclass
class DestroyWeightsUpdateGroupReqOutput(BaseReq):
    # 中译：销毁权重更新组的应答。
    success: bool
    message: str


@dataclass
class UpdateWeightVersionReqInput(BaseReq):
    # 中译：仅更新权重版本号（不动权重本身）的请求。
    # The new weight version
    new_version: str  # 中译：新版本号
    # Whether to abort all running requests before updating
    abort_all_requests: bool = True


@dataclass
class GetWeightsByNameReqInput(BaseReq):
    # 中译：按参数名读取权重的请求（调试/校验用）。
    name: str  # 中译：参数名
    truncate_size: int = 100  # 中译：返回时截断的元素数（避免传输整个大张量）


@dataclass
class GetWeightsByNameReqOutput(BaseReq):
    # 中译：按名读取权重的应答，parameter 为截断后的权重值列表。
    parameter: list


@dataclass
class ReleaseMemoryOccupationReqInput(BaseReq):
    # Optional tags to identify the memory region, which is primarily used for RL
    # Currently we only support `weights` and `kv_cache`
    # 中译：释放显存占用的请求（主要用于 RL，把权重/KV cache 显存暂时让出给训练）。
    #       tags 标识要释放的内存区域，目前仅支持 `weights` 和 `kv_cache`。
    tags: Optional[List[str]] = None


@dataclass
class ReleaseMemoryOccupationReqOutput(BaseReq):
    # 中译：释放显存占用的应答。
    pass


@dataclass
class ResumeMemoryOccupationReqInput(BaseReq):
    # Optional tags to identify the memory region, which is primarily used for RL
    # Currently we only support `weights` and `kv_cache`
    # 中译：恢复（重新占用）显存的请求，与 ReleaseMemoryOccupation 配对使用。
    tags: Optional[List[str]] = None


@dataclass
class ResumeMemoryOccupationReqOutput(BaseReq):
    # 中译：恢复显存占用的应答。
    pass


@dataclass
class CheckWeightsReqInput(BaseReq):
    # 中译：校验权重的请求，action 指定校验方式（默认 checksum 校验和）。
    action: str = "checksum"


@dataclass
class CheckWeightsReqOutput(BaseReq):
    # 中译：校验权重的应答，payload 携带校验结果细节。
    success: bool
    message: str
    payload: Optional[Dict] = None


@dataclass
class SlowDownReqInput(BaseReq):
    # 中译：人为减速的请求（调试/复现时序问题用）。每次前向后睡眠 forward_sleep_time 秒。
    forward_sleep_time: Optional[float]


@dataclass
class SlowDownReqOutput(BaseReq):
    # 中译：减速请求的应答。
    pass


@dataclass
class AbortReq(BaseReq):
    """中译：中止请求。可中止指定 rid，也可 abort_all 中止全部；附带结束原因/消息。"""

    # Whether to abort all requests
    abort_all: bool = False  # 中译：是否中止所有请求
    # The finished reason data
    finished_reason: Optional[Dict[str, Any]] = None  # 中译：作为结束原因返回给客户端的数据
    abort_message: Optional[str] = None  # 中译：中止说明消息

    def __post_init__(self):
        # FIXME: This is a hack to keep the same with the old code
        # 中译：兼容旧代码的 hack——rid 为 None 时置为空串。
        if self.rid is None:
            self.rid = ""


@dataclass
class ActiveRanksOutput(BaseReq):
    # 中译：各 rank 是否活跃的状态列表。
    status: List[bool]


@dataclass
class GetInternalStateReq(BaseReq):
    # 中译：查询调度器内部状态的请求。
    pass


@dataclass
class GetInternalStateReqOutput(BaseReq):
    # 中译：内部状态查询应答，internal_state 为状态键值对。
    internal_state: Dict[Any, Any]


@dataclass
class SetInternalStateReq(BaseReq):
    # 中译：在运行时修改调度器内部参数的请求（server_args 为待覆盖的配置项）。
    server_args: Dict[str, Any]


@dataclass
class SetInternalStateReqOutput(BaseReq):
    # 中译：设置内部状态的应答，updated 表示是否生效，并回传当前 server_args。
    updated: bool
    server_args: Dict[str, Any]


@dataclass
class ProfileReqInput(BaseReq):
    """中译：性能分析（profiling）请求的输入参数（如启动 torch profiler 抓取 trace）。"""

    # The output directory
    output_dir: Optional[str] = None  # 中译：trace 输出目录
    # Specify the steps to start the profiling
    start_step: Optional[int] = None  # 中译：从第几步开始 profile
    # If set, it profile as many as this number of steps.
    # If it is set, profiling is automatically stopped after this step, and
    # the caller doesn't need to run stop_profile.
    # 中译：profile 的步数。设了它就在跑满这么多步后自动停止，调用方无需再调 stop_profile。
    num_steps: Optional[int] = None
    # The activities to record. The choices are ["CPU", "GPU", "MEM", "RPD"]
    activities: Optional[List[str]] = None  # 中译：要记录的活动类型（CPU/GPU/MEM/RPD）
    # Whether profile by stages (e.g., prefill and decode) separately
    profile_by_stage: bool = False  # 中译：是否按阶段（prefill/decode）分别 profile
    # Whether to record source information (file and line number) for the ops.
    with_stack: Optional[bool] = None  # 中译：是否记录算子的源码位置（文件/行号）
    # Whether to save information about operator’s input shapes.
    record_shapes: Optional[bool] = None  # 中译：是否记录算子输入形状
    # Merge profiles from all ranks into a single trace
    merge_profiles: bool = False  # 中译：是否把各 rank 的 profile 合并成一个 trace
    # The prefix of the profile filenames
    profile_prefix: Optional[str] = None  # 中译：trace 文件名前缀
    # Only profile these stages and ignore others
    profile_stages: Optional[List[str]] = None  # 中译：只 profile 指定的阶段


class ProfileReqType(Enum):
    # 中译：profile 请求类型——启动 / 停止。
    START_PROFILE = 1
    STOP_PROFILE = 2


@dataclass
class ProfileReq(BaseReq):
    # 中译：实际下发给调度器的 profile 控制消息（由 ProfileReqInput + 类型组装而成）。
    type: ProfileReqType  # 中译：启动还是停止
    output_dir: Optional[str] = None
    start_step: Optional[int] = None
    num_steps: Optional[int] = None
    activities: Optional[List[str]] = None
    profile_by_stage: bool = False
    with_stack: Optional[bool] = None
    record_shapes: Optional[bool] = None
    profile_id: Optional[str] = None
    merge_profiles: bool = False
    profile_prefix: Optional[str] = None
    profile_stages: Optional[List[str]] = None


@dataclass
class ProfileReqOutput(BaseReq):
    # 中译：profile 控制的应答。
    success: bool
    message: str


@dataclass
class FreezeGCReq(BaseReq):
    # 中译：冻结垃圾回收（freeze GC）的请求，减少长期对象的反复扫描开销。
    pass


@dataclass
class ConfigureLoggingReq(BaseReq):
    """中译：运行时调整日志/请求转储行为的请求。"""

    log_requests: Optional[bool] = None  # 中译：是否记录请求
    log_requests_level: Optional[int] = None  # 中译：请求日志详细级别
    log_requests_format: Optional[str] = None  # 中译：请求日志格式
    log_level: Optional[str] = None  # 中译：全局日志级别
    dump_requests_folder: Optional[str] = None  # 中译：转储请求的目录
    dump_requests_threshold: Optional[int] = None  # 中译：累计多少条后落盘
    crash_dump_folder: Optional[str] = None  # 中译：崩溃转储目录
    dump_requests_exclude_meta_keys: Optional[List[str]] = None  # 中译：转储时排除的元信息 key


@dataclass
class OpenSessionReqInput(BaseReq):
    # 中译：打开一个会话（用于持续对话）的请求。
    capacity_of_str_len: int  # 中译：会话可缓存的字符串长度上限
    session_id: Optional[str] = None  # 中译：指定会话 id（不给则自动生成）
    streaming: Optional[bool] = None  # 中译：是否流式
    timeout: Optional[float] = None  # 中译：会话超时


@dataclass
class CloseSessionReqInput(BaseReq):
    # 中译：关闭指定会话的请求。
    session_id: str


@dataclass
class OpenSessionReqOutput(BaseReq):
    # 中译：打开会话的应答，返回实际会话 id 与是否成功。
    session_id: Optional[str]
    success: bool


@dataclass
class HealthCheckOutput(BaseReq):
    # 中译：健康检查的输出占位对象。
    pass


class ExpertDistributionReqType(Enum):
    # 中译：MoE 专家分布统计的控制类型——开始记录 / 停止记录 / 导出记录。
    START_RECORD = 1
    STOP_RECORD = 2
    DUMP_RECORD = 3


@dataclass
class ExpertDistributionReq(BaseReq):
    # 中译：控制 MoE 专家分布统计的请求（分析负载是否均衡）。
    action: ExpertDistributionReqType


@dataclass
class ExpertDistributionReqOutput(BaseReq):
    # 中译：专家分布统计控制的应答。
    pass


@dataclass
class Function:
    # 中译：函数调用（function/tool calling）中的函数定义。
    description: Optional[str] = None  # 函数描述
    name: Optional[str] = None  # 函数名
    parameters: Optional[object] = None  # 参数 schema


@dataclass
class Tool:
    # 中译：可供模型调用的工具，封装一个 Function。
    function: Function
    type: Optional[str] = "function"


@dataclass
class ParseFunctionCallReq(BaseReq):
    # 中译：解析模型输出中的函数调用的请求（把文本解析成结构化的工具调用）。
    text: str  # The text to parse.
    tools: List[Tool] = field(
        default_factory=list
    )  # A list of available function tools (name, parameters, etc.).
    tool_call_parser: Optional[str] = (
        None  # Specify the parser type, e.g. 'llama3', 'qwen25', or 'mistral'. If not specified, tries all.
    )


@dataclass
class SeparateReasoningReqInput(BaseReq):
    # 中译：把模型输出中的「推理（thinking）」部分与最终答案分离的请求。
    text: str  # The text to parse.
    reasoning_parser: str  # Specify the parser type, e.g., "deepseek-r1".
    # 中译：指定推理解析器类型（如 deepseek-r1）。
    return_blocks: bool = False  # If True, also return segmented reasoning blocks.
    # 中译：为 True 时额外返回分段后的推理块。


@dataclass
class VertexGenerateReqInput(BaseReq):
    # 中译：兼容 Google Vertex AI 协议的生成请求。
    instances: List[dict]  # 中译：Vertex 格式的输入实例列表
    parameters: Optional[dict] = None  # 中译：生成参数


@dataclass
class RpcReqInput(BaseReq):
    # 中译：通用 RPC 调用请求——按方法名 method 调用调度器内部方法，parameters 为参数。
    method: str
    parameters: Optional[Dict] = None


@dataclass
class RpcReqOutput(BaseReq):
    # 中译：RPC 调用的应答。
    success: bool
    message: str


@dataclass
class LoadLoRAAdapterReqInput(BaseReq):
    """中译：运行时加载一个 LoRA 适配器的请求。"""

    # The name of the lora module to newly loaded.
    lora_name: str  # 中译：新加载的 LoRA 模块名
    # The path of loading.
    lora_path: str  # 中译：加载路径
    # Whether to pin the LoRA adapter in memory.
    pinned: bool = False  # 中译：是否把该适配器常驻显存（不被换出）
    # The unique identifier for the LoRA adapter, which automatically generated in the `TokenizerManager`.
    lora_id: Optional[str] = None  # 中译：LoRA 唯一 id，由 TokenizerManager 自动生成

    def to_ref(self) -> LoRARef:
        # 中译：转换为内部使用的 LoRARef 引用对象。
        return LoRARef(
            lora_id=self.lora_id,
            lora_name=self.lora_name,
            lora_path=self.lora_path,
            pinned=self.pinned,
        )


@dataclass
class UnloadLoRAAdapterReqInput(BaseReq):
    """中译：运行时卸载一个已加载 LoRA 适配器的请求。"""

    # The name of lora module to unload.
    lora_name: str  # 中译：要卸载的 LoRA 模块名
    # The unique identifier for the LoRA adapter, which automatically generated in the `TokenizerManager`.
    lora_id: Optional[str] = None  # 中译：LoRA 唯一 id

    def to_ref(self) -> LoRARef:
        # 中译：转换为 LoRARef 引用对象。
        return LoRARef(
            lora_id=self.lora_id,
            lora_name=self.lora_name,
        )


@dataclass
class LoadLoRAAdapterFromTensorsReqInput(BaseReq):
    """中译：直接以张量形式加载 LoRA 适配器的请求（权重以序列化张量传入，而非磁盘路径）。"""

    lora_name: str  # 中译：LoRA 模块名
    config_dict: Dict[str, Any]  # 中译：LoRA 配置
    serialized_tensors: str  # 中译：序列化后的 LoRA 权重张量
    pinned: bool = False  # 中译：是否常驻显存
    added_tokens_config: Optional[Dict[str, Any]] = None  # 中译：新增 token 的配置
    lora_id: Optional[str] = None  # 中译：LoRA 唯一 id
    load_format: Optional[str] = None  # 中译：加载格式

    def to_ref(self) -> LoRARef:
        # 中译：转换为 LoRARef；lora_path 用特殊标记 "__tensor__" 表示来自张量。
        return LoRARef(
            lora_id=self.lora_id,
            lora_name=self.lora_name,
            lora_path="__tensor__",
            pinned=self.pinned,
        )


@dataclass
class LoRAUpdateOutput(BaseReq):
    # 中译：LoRA 加载/卸载的统一应答。loaded_adapters 为当前已加载的适配器映射。
    success: bool
    error_message: Optional[str] = None
    loaded_adapters: Optional[Dict[str, LoRARef]] = None


# 中译：三种 LoRA 操作（加载/卸载/从张量加载）共用同一种应答类型 LoRAUpdateOutput。
LoadLoRAAdapterReqOutput = UnloadLoRAAdapterReqOutput = (
    LoadLoRAAdapterFromTensorsReqOutput
) = LoRAUpdateOutput


class BlockReqType(Enum):
    # 中译：阻塞控制类型——阻塞 / 解除阻塞。
    BLOCK = 1
    UNBLOCK = 2


@dataclass
class BlockReqInput(BaseReq):
    # 中译：阻塞/解除阻塞调度器的请求。
    type: BlockReqType


@dataclass
class MemoryMetrics:
    """Memory breakdown metrics.

    中译：显存占用细分指标（权重/KV cache/CUDA graph 各占多少 GB，以及 token 容量）。
    """

    weight_gb: float = field(
        metadata={"metric": ("gauge", "Model weight memory in GB")}
    )
    kv_cache_gb: float = field(metadata={"metric": ("gauge", "KV cache memory in GB")})
    graph_gb: float = field(metadata={"metric": ("gauge", "CUDA graph memory in GB")})
    token_capacity: int = field(
        metadata={"metric": ("gauge", "Max tokens in KV cache")}
    )


@dataclass
class SpeculativeMetrics:
    """Speculative decoding metrics.

    中译：推测解码指标——平均接受长度、接受率，衡量推测解码的收益。
    """

    accept_length: float = field(
        metadata={
            "metric": (
                "gauge",
                "Mean acceptance length (accepted drafts + bonus token per forward)",
            )
        }
    )
    accept_rate: float = field(
        metadata={"metric": ("gauge", "Speculative acceptance rate")}
    )


@dataclass
class LoRAMetrics:
    """LoRA adapter pool metrics.

    中译：LoRA 适配器池指标——已用槽位、总槽位、利用率。
    """

    slots_used: int = field(metadata={"metric": ("gauge", "LoRA adapter slots in use")})
    slots_total: int = field(metadata={"metric": ("gauge", "Total LoRA adapter slots")})
    utilization: float = field(
        metadata={"metric": ("gauge", "LoRA pool utilization ratio")}
    )


@dataclass
class DisaggregationMetrics:
    """PD disaggregation metrics.

    中译：PD 分离（prefill/decode 分离）相关指标——各队列长度、KV 传输速度与延迟等。
    """

    mode: str  # "prefill", "decode", or "null" - not a metric
    # 中译：本实例角色（prefill/decode/null）；它本身不是指标，仅用于区分。
    prefill_bootstrap_queue_reqs: int = field(
        default=0, metadata={"metric": ("gauge", "Prefill bootstrap queue requests")}
    )
    prefill_inflight_queue_reqs: int = field(
        default=0, metadata={"metric": ("gauge", "Prefill inflight queue requests")}
    )
    decode_prealloc_queue_reqs: int = field(
        default=0, metadata={"metric": ("gauge", "Decode prealloc queue requests")}
    )
    decode_transfer_queue_reqs: int = field(
        default=0, metadata={"metric": ("gauge", "Decode transfer queue requests")}
    )
    decode_retracted_queue_reqs: int = field(
        default=0, metadata={"metric": ("gauge", "Decode retracted queue requests")}
    )
    kv_transfer_speed_gb_s: float = field(
        default=0.0, metadata={"metric": ("gauge", "KV transfer speed in GB/s")}
    )
    kv_transfer_latency_ms: float = field(
        default=0.0, metadata={"metric": ("gauge", "KV transfer latency in ms")}
    )


@dataclass
class QueueMetrics:
    """Detailed queue breakdown.

    中译：调度器各队列的细分长度——等待队列、语法编译队列、被暂停的、被回撤的。
    """

    waiting: int = field(metadata={"metric": ("gauge", "Main waiting queue size")})
    grammar: int = field(
        metadata={"metric": ("gauge", "Grammar compilation queue size")}
    )
    paused: int = field(
        metadata={"metric": ("gauge", "Requests paused by weight sync")}
    )
    retracted: int = field(metadata={"metric": ("gauge", "Retracted requests count")})


@dataclass
class GetLoadsReqInput(BaseReq):
    """Request for /v1/loads endpoint.

    中译：/v1/loads 接口的请求——查询各 DP rank 的负载情况。
    """

    # 中译：合法的查询分区集合（核心/内存/推测解码/LoRA/PD分离/队列/全部）。
    VALID_SECTIONS = frozenset(
        {"core", "memory", "spec", "lora", "disagg", "queues", "all"}
    )

    include: List[str] = field(default_factory=lambda: ["all"])  # 中译：要返回哪些分区，默认全部
    dp_rank: Optional[int] = None  # 中译：只查指定 DP rank（不给则查全部）

    def __post_init__(self):
        """Validate include sections.

        中译：校验 include 中的分区名是否合法。
        """
        if self.include:
            invalid = set(self.include) - self.VALID_SECTIONS
            if invalid:
                raise ValueError(
                    f"Invalid include sections: {invalid}. "
                    f"Valid options: {sorted(self.VALID_SECTIONS)}"
                )


@dataclass
class GetLoadsReqOutput(BaseReq):
    """Per-DP-rank load metrics for /v1/loads endpoint.

    中译：/v1/loads 的应答——单个 DP rank 的负载指标快照。含运行/等待请求数、token 使用、
          吞吐、缓存命中率、利用率，以及可选的内存/推测解码/LoRA/PD分离/队列细分。
          其中部分字段（num_total_tokens 等）也被 DP 负载均衡用于决策。
    """

    dp_rank: int  # 中译：本快照对应的 DP rank
    timestamp: float  # 中译：采样时间戳

    num_running_reqs: int = field(
        metadata={"metric": ("gauge", "Number of running requests")}
    )
    num_waiting_reqs: int = field(
        metadata={"metric": ("gauge", "Number of waiting requests")}
    )
    num_waiting_uncached_tokens: int = field(
        metadata={
            "metric": (
                "gauge",
                "Number of uncached input tokens waiting for prefill compute",
            )
        }
    )
    num_used_tokens: int = field(
        metadata={"metric": ("gauge", "Number of tokens in use")}
    )
    # num_used_tokens + pending prefill tokens (waiting-queue seqlen, incl.
    # disagg bootstrap/prealloc/transfer queues). Used for DP balance.
    num_total_tokens: int = field(
        metadata={"metric": ("gauge", "Used tokens plus pending prefill tokens")}
    )
    max_total_num_tokens: int = field(
        metadata={"metric": ("gauge", "Maximum token capacity")}
    )
    # FIXME: token_usage is actually max usage across all pools (KV, SWA, mamba),
    # not just KV token usage. Rename requires API deprecation.
    token_usage: float = field(metadata={"metric": ("gauge", "Token pool usage ratio")})
    gen_throughput: float = field(
        metadata={"metric": ("gauge", "Generation throughput tokens/sec")}
    )
    cache_hit_rate: float = field(
        metadata={"metric": ("gauge", "Prefix cache hit rate")}
    )
    utilization: float = field(
        metadata={"metric": ("gauge", "Overall utilization ratio")}
    )
    max_running_requests: int = field(
        metadata={"metric": ("gauge", "Maximum running requests capacity")}
    )

    memory: Optional[MemoryMetrics] = None
    speculative: Optional[SpeculativeMetrics] = None
    lora: Optional[LoRAMetrics] = None
    disaggregation: Optional[DisaggregationMetrics] = None
    queues: Optional[QueueMetrics] = None


@dataclass
class WatchLoadUpdateReq(BaseReq):
    # 中译：负载更新广播消息——把各 DP rank 的最新负载推送给负载均衡/路由组件。
    loads: List[GetLoadsReqOutput]


@dataclass
class SetInjectDumpMetadataReqInput(BaseReq):
    # 中译：设置张量转储（dump）元信息的请求（调试/排查数值问题时给 dump 打标签）。
    dump_metadata: Dict[str, Any]


@dataclass
class SetInjectDumpMetadataReqOutput(BaseReq):
    # 中译：设置转储元信息的应答。
    success: bool


@dataclass
class LazyDumpTensorsReqInput(BaseReq):
    # 中译：触发「延迟张量转储」的请求。
    pass


@dataclass
class LazyDumpTensorsReqOutput(BaseReq):
    # 中译：延迟张量转储的应答。
    success: bool


@dataclass
class DumperControlReqInput(BaseReq):
    # 中译：转储器（dumper）的通用控制请求，按 method+body 调用。
    method: str
    body: Dict[str, Any]


@dataclass
class DumperControlReqOutput(BaseReq):
    # 中译：转储器控制的应答。
    success: bool
    response: List[Dict[str, Any]]
    error: str = ""


def _check_all_req_types():
    """A helper function to check all request types are defined in this file."""
    import inspect
    import sys

    all_classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    for class_type in all_classes:
        # check its name
        name = class_type[0]
        is_io_struct = (
            name.endswith("Req") or name.endswith("Input") or name.endswith("Output")
        )
        is_base_req = issubclass(class_type[1], BaseReq) or issubclass(
            class_type[1], BaseBatchReq
        )
        if is_io_struct and not is_base_req:
            raise ValueError(f"{name} is not a subclass of BaseReq or BaseBatchReq.")
        if is_base_req and not is_io_struct:
            raise ValueError(
                f"{name} is a subclass of BaseReq but not follow the naming convention."
            )


_check_all_req_types()
