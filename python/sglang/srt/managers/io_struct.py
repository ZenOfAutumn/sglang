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

Keep this file focused on IPC struct definitions so it stays concise. Put
normalizers, helper utilities, and future non-struct logic in the owning module
instead, such as sglang.srt.utils.common.
"""

from __future__ import annotations

import copy
import logging
import pickle
import uuid
from array import array
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

import msgspec
import numpy as np
import torch
import zmq
import zmq.asyncio
from pydantic import PlainValidator

from sglang.srt.beam_search.types import BeamSearchSequence
from sglang.srt.environ import envs
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.embed_types import PositionalEmbeds
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalProcessorOutput,
    ReturnHiddenStatesMode,
    get_return_hidden_states_mode,
)
from sglang.srt.multimodal.mm_utils import has_valid_data
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import ImageData, VideoData
from sglang.srt.utils.field_validators import validate_optional_list_i64_1d_2d
from sglang.srt.utils.msgpack_utils import dec_hook, enc_hook, ext_hook
from sglang.srt.utils.msgspec_utils import (
    Base64Bytes,
    msgspec_struct_pydantic_core_schema,
)
from sglang.srt.utils.weight_versions import WeightVersionSpans

# Handle serialization of Image for pydantic
if TYPE_CHECKING:
    from PIL.Image import Image
else:
    Image = Any

    # 中译：所有「单请求」类消息的基类。提供两个跨进程通用字段：
    # 中译：所有「批次」类消息的基类。与 BaseReq 对应，但字段是复数列表形式：
logger = logging.getLogger(__name__)


class BaseReq(msgspec.Struct, tag=True, kw_only=True, array_like=True):
    """Base for single-request IPC payloads."""

    rid: Optional[str] = None
    http_worker_ipc: Optional[str] = None

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return msgspec_struct_pydantic_core_schema(cls, handler)


class BaseBatchReq(msgspec.Struct, tag=True, kw_only=True, array_like=True):
    """Base for batched IPC payloads."""

    # 中译：验证次数——目标模型为校验草稿 token 而做的前向次数（逐请求计数）。
    # 中译：被接受的草稿 token 数（严格只数草稿，不含每步附带的 bonus token）。
    # 中译：接受数直方图（逐请求一个内层列表）。下标 = 某步接受的 token 数，值 = 出现该接受数的步数。
# 中译：会话（session）参数，用于「持续对话/续写」场景，让多次请求复用同一会话上下文。
    rids: Optional[List[str]] = None
    # Used by batch messages whose items are parallel arrays, such as scheduler
    # outputs. Tokenized input batches store routing on batch[i].http_worker_ipc
    # because the scheduler unpacks them into single-request handlers.
    http_worker_ipcs: Optional[List[Optional[str]]] = None

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return msgspec_struct_pydantic_core_schema(cls, handler)


class BeamSearchOutput(BaseBatchReq, kw_only=True):
    sequences: List[BeamSearchSequence]


class PickleWrapper(msgspec.Struct, tag=True, array_like=True):
    """Wraps an arbitrary Python object as pickle-serialized bytes for msgpack IPC.

    In msgpack mode, fields that carry opaque or non-msgspec-typed payloads
    (e.g. multimodal inputs, time stats, customized info) are stored as
    PickleWrapper so the outer struct can still be msgpack-encoded.  In pickle
    mode (_USE_PICKLE_IPC=True), wrap_as_pickle / unwrap_from_pickle are no-ops
    and this class is not used on the wire.
    """

    data: bytes


# Parameters for a session
class SessionParams(msgspec.Struct, kw_only=True, array_like=True):
    # The session identifier. Used by the scheduler to look up or create the
    # Session object that groups all requests in a multi-turn conversation.
    id: Optional[str] = None
    # A request identifier *within* the session. In non-streaming sessions the
    # session maintains a tree of request nodes keyed by rid; this field selects
    # which node to continue from (append) or replace. When None the default
    # branch point is used (latest node for streaming, all nodes cleared on
    # replace).
    rid: Optional[str] = None
    # Token-level insertion point. When set, the new request's tokens are
    # spliced into the accumulated context at this position instead of being
    # appended at the end (i.e. ``context[:offset] + new_tokens``).
    offset: Optional[int] = None
    # When True, the request node identified by ``rid`` (or all nodes if
    # ``rid`` is None) is aborted and its children are cleared before the new
    # request is inserted. Not supported in streaming sessions.
    replace: Optional[bool] = None
    # When True, the previous request's generated output tokens are excluded
    # from the accumulated context so the new turn sees only the original input.
    # Not supported in streaming sessions.
    drop_previous_output: Optional[bool] = None


# Type definitions for multimodal input data
# Individual data item types for each modality
# 中译：多模态输入数据的类型定义。每种模态（图像/音频/视频）允许传入对象实例、文件名、URL、
ImageDataInputItem = Union[str, bytes, Dict[str, Any], ImageData, Image]
AudioDataInputItem = Union[str, bytes, Dict[str, Any]]
VideoDataInputItem = Union[str, bytes, Dict[str, Any], VideoData]
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
class GenerateReqInput:
    # Request ID(s). If omitted, generated during normalization. For batch
    # requests, a string is expanded to per-item IDs using it as a prefix.
    rid: Optional[Union[str, List[str]]] = field(default=None, kw_only=True)
    # Stable identity shared by requests in the same session. Unlike
    # session_params, this does not alter or reconstruct the prompt.
    session_id: Optional[str] = field(default=None, kw_only=True)
    # The input prompt. It can be a single prompt or a batch of prompts.
    # 中译：输入提示词。可为单个字符串，或字符串列表（批处理）。
    text: Optional[Union[List[str], str]] = None
    # The token ids for text.
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
    audio_data: Optional[MultimodalDataInputFormat] = None
    # Optional per-image hashes the caller has already computed (hex strings).
    # Single request: one hash per image. Batch request: either one hash per
    # request when each request has one image, or one list of hashes per request.
    # When supplied, each MultimodalDataItem's
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
    # Optional `sha256:<64-hex>` identities for the original media contents. Unlike
    # mm_hashes, these identify processor inputs and never replace the
    # processor-output feature hash used by the embedding/prefix cache.
    mm_content_hashes: Optional[
        Union[List[Optional[str]], List[List[Optional[str]]]]
    ] = None
    # Whether to extract and process audio from video inputs.
    # 中译：采样参数（温度、top_p、max_new_tokens、n 等）；可为单个 dict 或逐请求的 dict 列表。
    use_audio_in_video: bool = False
    # Optional request-scoped video processor configuration.
    video_config: Optional[Dict[str, Any]] = None
    # The sampling_params. See descriptions below.
    sampling_params: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
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
    # Whether to return output-token sampling support and renormalized logprobs.
    return_sampling_mask: Optional[Union[List[bool], bool]] = None
    # Whether to detokenize tokens in text in the returned logprobs.
    # 中译：返回的 logprob 中是否把 token 反向解码为文本一并附上。
    return_text_in_logprobs: bool = False
    # Return prompt top logprobs as flat arrays plus shape metadata instead of
    # the nested per-position [logprob, token_id, text] lists.
    return_flat_raw_top_logprobs: bool = False
    # Base64-encode the flat arrays. Requires return_flat_raw_top_logprobs.
    return_flat_raw_top_logprobs_b64: bool = False
    # Whether to stream output.
    stream: bool = False  # 中译：是否流式返回输出
    # Whether to log metrics for this request (e.g. health_generate calls do not log metrics)
    # 中译：本请求是否记录指标（如健康检查 health_generate 不记录，避免污染统计）。
    log_metrics: bool = True
    # Whether to return hidden states
    return_hidden_states: Union[
        List[ReturnHiddenStatesMode], ReturnHiddenStatesMode
    ] = False
    # Whether to return captured routed experts
    # 中译：是否返回 MoE 路由选中的专家（routed experts），用于分析/可观测。
    return_routed_experts: bool = False
    # Absolute start position for returned routings; response covers
    # `[routed_experts_start_len, seqlen - 1)`. Must be in [0, prompt_tokens].
    # 0 = full sequence.
    # 中译：返回路由信息的绝对起始位置，覆盖区间为 [routed_experts_start_len, seqlen-1)。
    #       取值须在 [0, prompt_tokens]；0 表示返回整条序列。
    routed_experts_start_len: int = 0
    return_indexer_topk: bool = False

    # The modalities of the image data [image, multi-images, video]
    # 中译：图像数据的模态标识（image / multi-images / video），随归一化自动填充。
    modalities: Optional[List[str]] = None
    # Session info for continual prompting
    # 中译：LoRA 适配器的唯一 id，由 TokenizerManager 根据 lora_path 解析填充。
    session_params: Optional[Dict[str, Any]] = None

    # The path to the LoRA adaptors
    lora_path: Optional[Union[List[Optional[str]], str]] = None
    # The uid of LoRA adaptors, should be initialized by tokenizer manager
    lora_id: Optional[Union[List[Optional[str]], str]] = None

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
    # 中译：强制本请求进入推理（thinking）模式（仅混合推理模型可用）。
    bootstrap_host: Optional[Union[List[Optional[str]], str]] = None
    bootstrap_port: Optional[Union[List[Optional[int]], int]] = None
    bootstrap_room: Optional[Union[List[Optional[int]], int]] = None
    bootstrap_pair_key: Optional[Union[List[Optional[str]], str]] = None
    decode_tp_size: Optional[Union[List[Optional[int]], int]] = None

    # For DP routing — external router assigns a specific DP worker
    # 中译：DP 路由——由外部路由器指定本请求应分配到哪个 DP（数据并行）worker。
    routed_dp_rank: Optional[int] = None
    # Deprecated alias for `routed_dp_rank`, still accepted because
    # sgl-model-gateway's dp-aware mode injects this spelling into every
    # request it forwards (DPAwareWorker::prepare_request), and the OpenAI
    # entrypoints and Engine.generate() accept it as well.
    data_parallel_rank: Optional[int] = None
    # For PD disagg — hint telling decode which prefill DP worker has the KV cache
    # 中译：PD 分离场景——提示 decode 端：KV cache 在哪个 prefill 的 DP worker 上。
    disagg_prefill_dp_rank: Optional[int] = None
    # 中译：已废弃，请改用 routed_dp_rank。
    # Routing key for routing-key schedule policy
    routing_key: Optional[str] = None
    # Conversation id used for tracking requests
    conversation_id: Optional[str] = None
    # Internal IPC endpoint of the HTTP/tokenizer worker that owns this request.
    # Used to route outputs back in multi-tokenizer mode.
    http_worker_ipc: Optional[str] = field(default=None, kw_only=True)

    # For background responses (OpenAI responses API)
    # 中译：后台模式（OpenAI responses API），结果异步生成、稍后取回。
    background: bool = False
    # 中译：请求分类的附加 key（如 cache_salt），可用于区分前缀缓存命名空间。
    # 中译：routing-key 调度策略所用的路由 key（同 key 倾向落到同一 worker，利于复用缓存）。
    # Require reasoning for the request (hybrid reasoning model only)
    require_reasoning: bool = False
    # Per-request thinking budget. Requires strict thinking so the runtime can
    # enforce the limit rather than silently treating it as metadata.
    max_thinking_tokens: Optional[int] = None

    # Priority for the request
    priority: Optional[int] = None
    # Extra cache key for caller-defined request classification.
    extra_key: Optional[Union[List[str], str]] = None

    # Whether to disallow logging for this request (e.g. due to ZDR)
    # 中译：是否禁止对本请求记日志（如出于零数据保留 ZDR 合规要求）。
    no_logs: bool = False
    # For custom metric labels
    custom_labels: Optional[Dict[str, str]] = None  # 中译：自定义指标标签

    # (Internal) Whether to return bytes for image generation
    return_bytes: bool = False
    # Whether to return entropy
    return_entropy: bool = False
    # Whether to return prompt token IDs without computing logprobs
    # 中译：是否仅返回提示词 token id（不计算 logprob）。
    return_prompt_token_ids: bool = False

    # Propagates trace context via Engine.generate/async_generate
    # 中译：透传分布式追踪上下文（trace context），用于链路追踪。
    # 中译：以下为 EPD 分离（encoder/prefill/decode 分离）推理用字段。
    external_trace_header: Optional[Dict[str, Any]] = None
    received_time: Optional[float] = None

    # For EPD-disaggregated inference
    need_wait_for_mm_inputs: Optional[bool] = None
    num_items_assigned: Optional[Dict[Modality, List[int]]] = None
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

    # For Unlimited-OCR
    images_config: Optional[dict] = None

    # Pre-computed delimiter indices for multi-item scoring.
    # Batch-level: List[List[int]] (one per request). After __getitem__: List[int].
    # 中译：多条目打分（multi-item scoring）预计算的分隔符下标。
    #       批次级为 List[List[int]]（每请求一组）；经 __getitem__ 取单条后变为 List[int]。
    multi_item_delimiter_indices: Optional[Union[List[List[int]], List[int]]] = None

    # Cache namespace used to isolate otherwise-identical prefixes.
    cache_salt: Optional[Union[List[str], str]] = None

    def regenerate_rid(self):
        """Generate a new request ID and return it."""
        if isinstance(self.rid, list):
            self.rid = [uuid.uuid4().hex for _ in range(len(self.rid))]
        else:
            self.rid = uuid.uuid4().hex
        return self.rid

    def _validate_rid_uniqueness(self):
        """Validate that request IDs within a batch are unique."""
        if isinstance(self.rid, list) and len(set(self.rid)) != len(self.rid):
            counts = Counter(self.rid)
            duplicates = [rid for rid, count in counts.items() if count > 1]
            raise ValueError(
                f"Duplicate request IDs detected within the request: {duplicates}"
            )

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
        if self.session_id is not None and self.session_params is not None:
            raise ValueError("session_id and session_params cannot both be set.")
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
        if (
            self.return_flat_raw_top_logprobs
            and self.multi_item_delimiter_indices is not None
        ):
            raise ValueError(
                "return_flat_raw_top_logprobs does not support multi-item "
                "scoring: delimiter-sparse top logprob rows have no contiguous "
                "position mapping."
            )
        if (
            self.return_flat_raw_top_logprobs_b64
            and not self.return_flat_raw_top_logprobs
        ):
            raise ValueError(
                "return_flat_raw_top_logprobs_b64 requires return_flat_raw_top_logprobs."
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

    def _sampling_params_beam_width(self) -> int:
        # 1 means not a beam request.
        if isinstance(self.sampling_params, dict):
            return self.sampling_params.get("beam_width") or 1
        elif isinstance(self.sampling_params, list) and self.sampling_params:
            return self.sampling_params[0].get("beam_width") or 1
        return 1

    def _handle_beam_search_parallel_sampling(self) -> int:
        # No fan-out for beam requests: n means "number of returned sequences".
        if self._sampling_params_beam_width() > 1:
            return 1
        return self.parallel_sample_num

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

        self.parallel_sample_num = self._handle_beam_search_parallel_sampling()

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
        if self.return_sampling_mask is None:
            self.return_sampling_mask = False
        for field_name in ("extra_key", "cache_salt"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise ValueError(
                    f"{field_name} should be a string for a single request."
                )
            if value == "":
                setattr(self, field_name, None)

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
        self._normalize_mm_hashes(num)
        self._normalize_video_data(num)
        self._normalize_audio_data(num)
        self._normalize_sampling_params(num)
        self._normalize_logprob_params(num)
        self._normalize_return_hidden_states(num)
        self._normalize_custom_logit_processor(num)
        self._normalize_extra_key(num)
        self._normalize_cache_salt(num)
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

    def _normalize_mm_hashes(self, num):
        """Align per-media hashes with normalized batched image inputs."""
        for field_name in ("mm_hashes", "mm_content_hashes"):
            hashes = getattr(self, field_name)
            if hashes is None:
                setattr(self, field_name, [None] * num)
                continue
            if not isinstance(hashes, list):
                raise ValueError(f"{field_name} must be a list")
            if len(hashes) != self.batch_size:
                raise ValueError(
                    f"The length of {field_name} should equal the batch size"
                )

            normalized = []
            for request_index, request_hashes in enumerate(hashes):
                images = self.image_data[request_index]
                image_count = len(images or [])
                if isinstance(request_hashes, list):
                    per_request = request_hashes
                elif image_count == 1:
                    per_request = [request_hashes]
                else:
                    raise ValueError(
                        f"{field_name}[{request_index}] must be a list with one "
                        "entry per image"
                    )
                if len(per_request) != image_count:
                    raise ValueError(
                        f"{field_name}[{request_index}] has {len(per_request)} "
                        f"entries for {image_count} images"
                    )
                normalized.append(per_request)
            setattr(self, field_name, normalized * self.parallel_sample_num)

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
        self.return_sampling_mask = normalize_param(
            self.return_sampling_mask, False, "return_sampling_mask"
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

    def _normalize_return_hidden_states(self, num):
        """Normalize and validate per-request hidden-state return modes."""
        if isinstance(self.return_hidden_states, list):
            if len(self.return_hidden_states) != self.batch_size:
                raise ValueError(
                    "The length of return_hidden_states should be equal to the batch size."
                )
            for mode in self.return_hidden_states:
                get_return_hidden_states_mode(mode)
            self.return_hidden_states = (
                self.return_hidden_states * self.parallel_sample_num
            )
        else:
            get_return_hidden_states_mode(self.return_hidden_states)
            self.return_hidden_states = [self.return_hidden_states] * num

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
            value = self.extra_key or None
            self.extra_key = [value] * num
        elif isinstance(self.extra_key, list):
            if len(self.extra_key) != self.batch_size:
                raise ValueError(
                    "The length of extra_key should be equal to the batch size."
                )
            if any(not isinstance(value, str) for value in self.extra_key):
                raise ValueError("Every extra_key should be a string.")
            self.extra_key = [value or None for value in self.extra_key]
            self.extra_key = self.extra_key * self.parallel_sample_num
        else:
            raise ValueError("extra_key should be a list or a string.")

    def _normalize_cache_salt(self, num):
        """Normalize cache_salt for batch processing."""
        if self.cache_salt is None:
            return
        if isinstance(self.cache_salt, str):
            value = self.cache_salt or None
            self.cache_salt = [value] * num
        elif isinstance(self.cache_salt, list):
            if len(self.cache_salt) != self.batch_size:
                raise ValueError(
                    "The length of cache_salt should be equal to the batch size."
                )
            if any(not isinstance(value, str) for value in self.cache_salt):
                raise ValueError("Every cache_salt should be a string.")
            self.cache_salt = [value or None for value in self.cache_salt]
            self.cache_salt = self.cache_salt * self.parallel_sample_num
        else:
            raise ValueError("cache_salt should be a list or a string.")

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

        # Normalize decode_tp_size
        if self.decode_tp_size is None:
            self.decode_tp_size = [None] * num
        elif not isinstance(self.decode_tp_size, list):
            self.decode_tp_size = [self.decode_tp_size] * num
        elif isinstance(self.decode_tp_size, list):
            self.decode_tp_size = self.decode_tp_size * self.parallel_sample_num

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
            rid=self.rid[i],
            session_id=self.session_id,
            text=self.text[i] if self.text is not None else None,
            input_ids=self.input_ids[i] if self.input_ids is not None else None,
            input_embeds=(
                self.input_embeds[i] if self.input_embeds is not None else None
            ),
            image_data=self.image_data[i],
            video_data=self.video_data[i],
            audio_data=self.audio_data[i],
            mm_hashes=self.mm_hashes[i] if self.mm_hashes is not None else None,
            mm_content_hashes=(
                self.mm_content_hashes[i]
                if self.mm_content_hashes is not None
                else None
            ),
            sampling_params=self.sampling_params[i],
            return_logprob=self.return_logprob[i],
            logprob_start_len=self.logprob_start_len[i],
            top_logprobs_num=self.top_logprobs_num[i],
            token_ids_logprob=self.token_ids_logprob[i],
            return_sampling_mask=self.return_sampling_mask[i],
            return_text_in_logprobs=self.return_text_in_logprobs,
            return_flat_raw_top_logprobs=self.return_flat_raw_top_logprobs,
            return_flat_raw_top_logprobs_b64=self.return_flat_raw_top_logprobs_b64,
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
            positional_embed_overrides=self._get_positional_embed_overrides_item(i),
            # If `__getitem__` is called, these bootstrap fields must be lists.
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
            http_worker_ipc=self.http_worker_ipc,
            require_reasoning=self.require_reasoning,
            max_thinking_tokens=self.max_thinking_tokens,
            priority=self.priority,
            extra_key=self.extra_key[i] if self.extra_key is not None else None,
            cache_salt=(self.cache_salt[i] if self.cache_salt is not None else None),
            no_logs=self.no_logs,
            custom_labels=self.custom_labels,
            return_bytes=self.return_bytes,
            return_entropy=self.return_entropy,
            return_prompt_token_ids=self.return_prompt_token_ids,
            external_trace_header=self.external_trace_header,
            received_time=self.received_time,
            multi_item_delimiter_indices=(
                self.multi_item_delimiter_indices[i]
                if self.multi_item_delimiter_indices is not None
                else None
            ),
        )
        cache[i] = sub
        return sub


class TokenizedGenerateReqInput(BaseReq, kw_only=True):
    input_text: Optional[Union[str, List[Union[str, List[str]]]]]
    # The input token ids
    input_ids: Optional[array]  # Optional[array[int]]
    # The input embeds
    input_embeds: Optional[List[List[float]]]
    # The multimodal inputs
    mm_inputs: Optional[MultimodalProcessorOutput]
    token_type_ids: Optional[List[int]]
    # The sampling parameters
    sampling_params: SamplingParams  # 中译：结构化采样参数对象
    # Whether to return the logprobs
    return_logprob: bool  # 中译：是否返回 logprob
    # If return logprobs, the start location in the prompt for returning logprobs.
    logprob_start_len: int  # 中译：返回 logprob 的提示词起始位置
    # If return logprobs, the number of top logprobs to return at each position.
    top_logprobs_num: int  # 中译：每位置返回的 top-k logprob 数量
    # If return logprobs, the token id to return logprob for
    token_ids_logprob: Optional[List[int]]
    # Whether to stream output
    stream: bool
    # Whether to return sparse output-token support from top-k/top-p/min-p sampling.
    return_sampling_mask: bool = False
    # Assemble prompt top logprobs as flat arrays scheduler-side (see
    # GenerateReqInput.return_flat_raw_top_logprobs). The b64 flag stays
    # tokenizer-manager-side: the scheduler ships arrays either way.
    return_flat_raw_top_logprobs: bool = False

    # Whether to return hidden states
    return_hidden_states: ReturnHiddenStatesMode = False

    # Whether to return captured routed experts
    return_routed_experts: bool = False  # 中译：是否返回 MoE 路由选中的专家
    # See GenerateReqInput.routed_experts_start_len.
    routed_experts_start_len: int = 0
    return_indexer_topk: bool = False

    # Session info for continual prompting
    session_id: Optional[str] = None
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
    # Embedding overrides to place at specific token positions.
    positional_embed_overrides: Optional[PositionalEmbeds] = None

    # For disaggregated inference
    # 中译：PD 分离推理对接参数（含义同 GenerateReqInput，但此处为单请求标量）。
    bootstrap_host: Optional[str] = None
    bootstrap_port: Optional[int] = None
    bootstrap_room: Optional[int] = None
    bootstrap_pair_key: Optional[str] = None
    decode_tp_size: Optional[int] = None

    # For DP routing
    routed_dp_rank: Optional[int] = None  # 中译：DP 路由指定的目标 DP rank
    # For PD disagg — hint telling decode which prefill DP worker has the KV cache
    disagg_prefill_dp_rank: Optional[int] = None  # 中译：提示 KV cache 所在的 prefill DP rank

    # Routing key for routing-key schedule policy
    routing_key: Optional[str] = None
    # Require reasoning for the request (hybrid reasoning model only)
    require_reasoning: bool = False

    # Priority for the request
    priority: Optional[int] = None  # 中译：请求优先级

    # Extra cache key for caller-defined request classification.
    extra_key: Optional[str] = None

    # Whether to disallow logging for this request (e.g. due to ZDR)
    no_logs: bool = False

    # (Internal) Whether to return bytes for image generation
    return_bytes: bool = False
    # Whether to return entropy
    return_entropy: bool = False

    need_wait_for_mm_inputs: Optional[bool] = None
    num_items_assigned: Optional[Dict[Modality, List[int]]] = None
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
    # Pickled Optional[Union[APIServerReqTimeStats, DPControllerReqTimeStats]]
    time_stats: Optional[PickleWrapper] = None

    # Cache namespace used to isolate otherwise-identical prefixes.
    cache_salt: Optional[str] = None

    def wrap_pickle_fields(self):
        self.time_stats = wrap_as_pickle(self.time_stats)

    def unwrap_pickle_fields(self):
        self.time_stats = unwrap_from_pickle(self.time_stats)


class BatchTokenizedGenerateReqInput(BaseBatchReq, kw_only=True):
    # The batch of tokenized requests
    # Routing for request i is batch[i].http_worker_ipc, not http_worker_ipcs[i].
    batch: List[TokenizedGenerateReqInput]

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, i):
        return self.batch[i]

    def __iter__(self):
        return iter(self.batch)


@dataclass
class EmbeddingReqInput:
    # Request ID(s). If omitted, generated during normalization. For batch
    # requests, a string is expanded to per-item IDs using it as a prefix.
    rid: Optional[Union[str, List[str]]] = field(default=None, kw_only=True)
    # The input prompt. It can be a single prompt or a batch of prompts.
    text: Optional[Union[List[List[str]], List[str], str]] = None
    # The token ids for text; one can either specify text or input_ids.
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    # Dummy input embeds for compatibility
    input_embeds: Optional[Union[List[List[List[float]]], List[List[float]]]] = None
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
    # 中译：已解析（带位置）的嵌入覆盖，由 TokenizerManager 或打分逻辑填充。
    # Dummy sampling params for compatibility
    sampling_params: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    # Whether to log metrics for this request (e.g. health_generate calls do not log metrics)
    log_metrics: bool = True  # 中译：是否记录指标
    # The modalities of the image data [image, multi-images, video]
    modalities: Optional[List[str]] = None  # 中译：图像数据的模态标识
    # For cross-encoder requests
    is_cross_encoder_request: bool = False
    # The path to the LoRA adaptors
    lora_path: Optional[Union[List[Optional[str]], str]] = None
    # The uid of LoRA adaptors, should be initialized by tokenizer manager
    lora_id: Optional[Union[List[Optional[str]], str]] = None
    # Resolved embedding overrides with positions (set by tokenizer manager or score mixin).
    # Runtime type: Optional[Union[PositionalEmbeds, List[Optional[PositionalEmbeds]]]]
    positional_embed_overrides: Any = None
    # Routing key for routing-key schedule policy
    routing_key: Optional[str] = None
    # Internal IPC endpoint of the HTTP/tokenizer worker that owns this request.
    # Used to route outputs back in multi-tokenizer mode.
    http_worker_ipc: Optional[str] = field(default=None, kw_only=True)

    # For background responses (OpenAI responses API)
    background: bool = False

    # Priority for the request
    priority: Optional[int] = None

    # The number of dimensions the resulting output embeddings should have. It is applicable for Matryoshka Embeddings.
    # 中译：输出嵌入的目标维度数（适用于 Matryoshka 嵌入——可按需截断到更短维度）。
    dimensions: Optional[int] = None
    # Whether to return pooled hidden states (pre-head transformer output)
    # 中译：是否返回池化后的隐藏状态（pooling head 之前的 transformer 输出）。
    return_pooled_hidden_states: bool = False
    # Whether to return prompt token IDs without computing logprobs
    return_prompt_token_ids: bool = False  # 中译：是否仅返回提示词 token id

    # Propagates trace context via Engine.encode/async_encode
    external_trace_header: Optional[Dict[str, Any]] = None
    received_time: Optional[float] = None

    # Pre-computed delimiter indices for multi-item scoring.
    # Batch-level: List[List[int]] (one per request). After __getitem__: List[int].
    # 中译：多条目打分预计算的分隔符下标（批次级嵌套列表，取单条后变扁平列表）。
    multi_item_delimiter_indices: Optional[Union[List[List[int]], List[int]]] = None

    def regenerate_rid(self):
        """Generate a new request ID and return it."""
        if isinstance(self.rid, list):
            self.rid = [uuid.uuid4().hex for _ in range(len(self.rid))]
        else:
            self.rid = uuid.uuid4().hex
        return self.rid

    def _validate_rid_uniqueness(self):
        """Validate that request IDs within a batch are unique."""
        if isinstance(self.rid, list) and len(set(self.rid)) != len(self.rid):
            counts = Counter(self.rid)
            duplicates = [rid for rid, count in counts.items() if count > 1]
            raise ValueError(
                f"Duplicate request IDs detected within the request: {duplicates}"
            )

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
                rid=self.rid[i],
                text=[self.text[i]] if self.text is not None else None,
                sampling_params=self.sampling_params[i],
                is_cross_encoder_request=True,
                lora_path=self.lora_path[i] if self.lora_path is not None else None,
                lora_id=self.lora_id[i] if self.lora_id is not None else None,
                positional_embed_overrides=self._get_positional_embed_overrides_item(i),
                http_worker_ipc=self.http_worker_ipc,
                priority=self.priority,
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
                rid=self.rid[i],
                text=self.text[i] if self.text is not None else None,
                input_ids=self.input_ids[i] if self.input_ids is not None else None,
                image_data=self.image_data[i] if self.image_data is not None else None,
                video_data=self.video_data[i] if self.video_data is not None else None,
                audio_data=self.audio_data[i] if self.audio_data is not None else None,
                embed_override_token_id=self.embed_override_token_id,
                embed_overrides=(
                    self.embed_overrides[i]
                    if self.embed_overrides is not None
                    else None
                ),
                sampling_params=self.sampling_params[i],
                lora_path=self.lora_path[i] if self.lora_path is not None else None,
                lora_id=self.lora_id[i] if self.lora_id is not None else None,
                positional_embed_overrides=self._get_positional_embed_overrides_item(i),
                http_worker_ipc=self.http_worker_ipc,
                priority=self.priority,
                dimensions=self.dimensions,
                return_pooled_hidden_states=self.return_pooled_hidden_states,
                return_prompt_token_ids=self.return_prompt_token_ids,
                external_trace_header=self.external_trace_header,
                received_time=self.received_time,
                multi_item_delimiter_indices=(
                    self.multi_item_delimiter_indices[i]
                    if self.multi_item_delimiter_indices is not None
                    else None
                ),
            )
        cache[i] = sub
        return sub


class TokenizedEmbeddingReqInput(BaseReq, kw_only=True):
    input_text: Optional[Union[str, List[Union[str, List[str]]]]]
    # The input token ids
    input_ids: Optional[array]  # array[int]
    # The multimodal inputs
    mm_inputs: Optional[MultimodalProcessorOutput]
    # The token type ids
    token_type_ids: Optional[List[int]]
    # Dummy sampling params for compatibility
    sampling_params: SamplingParams
    # LoRA related
    lora_id: Optional[str] = None  # None means just use the base model
    # Embedding overrides to place at specific token positions.
    positional_embed_overrides: Optional[PositionalEmbeds] = None
    # For DP routing
    routed_dp_rank: Optional[int] = None
    # Priority for the request
    priority: Optional[int] = None
    # The number of dimensions the resulting output embeddings should have. It is applicable for Matryoshka Embeddings.
    dimensions: Optional[int] = None
    # Whether to return pooled hidden states (pre-head transformer output)
    return_pooled_hidden_states: bool = False
    # Pre-computed delimiter indices for multi-item scoring
    multi_item_delimiter_indices: Optional[List[int]] = None

    # For observability
    # Pickled Optional[Union[APIServerReqTimeStats, DPControllerReqTimeStats]]
    time_stats: Optional[PickleWrapper] = None

    def wrap_pickle_fields(self):
        self.time_stats = wrap_as_pickle(self.time_stats)

    def unwrap_pickle_fields(self):
        self.time_stats = unwrap_from_pickle(self.time_stats)


class BatchTokenizedEmbeddingReqInput(BaseBatchReq, kw_only=True):
    # The batch of tokenized embedding requests
    # Routing for request i is batch[i].http_worker_ipc, not http_worker_ipcs[i].
    batch: List[TokenizedEmbeddingReqInput]

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, i):
        return self.batch[i]

    def __iter__(self):
        return iter(self.batch)


    # 中译：每个请求的结束原因（如正常停止/达到长度上限/命中 stop 等）；None 表示尚未结束（流式中）。
    # 中译：以下三项供「增量解码」使用（参见 detokenizer_manager.DecodeStatus）。
    # 中译：仅在 --skip-tokenizer-init 时使用：直接输出原始 token id（不做解码）。
TokenLogprobValues = Optional[List[Optional[List[Optional[float]]]]]
TokenLogprobIndices = Optional[List[Optional[List[Optional[int]]]]]
TopLogprobValues = Optional[List[Optional[List[Optional[List[float]]]]]]
TopLogprobIndices = Optional[List[Optional[List[Optional[List[int]]]]]]
TokenIdsLogprobValues = Optional[List[Optional[List[Optional[List[float]]]]]]
TokenIdsLogprobIndices = Optional[List[Optional[List[Optional[List[int]]]]]]
HiddenStateChunk = List[Optional[Union[float, List[float]]]]
OutputHiddenStates = Optional[List[Optional[List[HiddenStateChunk]]]]
CachedTokensDetails = Dict[str, Union[int, str]]
# Serialized form of BaseFinishReason.to_json() — all values are primitives.
FinishReasonDict = Dict[str, Optional[Union[str, int, List[int]]]]


def build_flat_input_top_logprobs_arrays(
    input_top_logprobs_val: List[Optional[List[float]]],
    input_top_logprobs_idx: List[Optional[List[int]]],
    top_logprobs_num: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Convert nested per-position prompt top logprob rows into the flat
    arrays of the `return_flat_raw_top_logprobs` response format.

    Returns (float32 values [rows, k], int32 token ids [rows, k],
    null_prefix). The leading null rows are counted into null_prefix and
    excluded from the arrays. Raises ValueError when the rows are not
    representable by (shape, null_prefix): interior nulls or ragged k,
    e.g. multi-item scoring.
    """
    num_rows = len(input_top_logprobs_val)
    null_prefix = 0
    while null_prefix < num_rows and not input_top_logprobs_val[null_prefix]:
        null_prefix += 1
    val_rows = input_top_logprobs_val[null_prefix:]
    idx_rows = input_top_logprobs_idx[null_prefix:]
    k = len(val_rows[0]) if val_rows else top_logprobs_num
    for offset, row in enumerate(val_rows):
        if row is None or len(row) != k:
            raise ValueError(
                "return_flat_raw_top_logprobs requires rectangular top logprob "
                f"rows with nulls only in the leading prefix; row {null_prefix + offset} "
                f"has {None if row is None else len(row)} entries (expected {k})."
            )
    val_arr = np.asarray(val_rows, dtype=np.float32).reshape(len(val_rows), k)
    idx_arr = np.asarray(idx_rows, dtype=np.int32).reshape(len(idx_rows), k)
    return val_arr, idx_arr, null_prefix


class BatchTokenIDOutput(BaseBatchReq, kw_only=True):
    # The finish reason
    finished_reasons: List[Optional[FinishReasonDict]]
    # For incremental decoding
    decoded_texts: List[str]
    decode_ids: List[array]  # List[array[int]]
    read_offsets: List[int]
    # Only used when `--skip-tokenizer-init` is on
    output_ids: Optional[List[array]]  # Optional[List[array[int]]]
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
    # 中译：按需返回的隐藏状态（如用于推测解码/下游嵌入等）。
    input_token_logprobs_val: TokenLogprobValues
    input_token_logprobs_idx: TokenLogprobIndices
    output_token_logprobs_val: TokenLogprobValues
    output_token_logprobs_idx: TokenLogprobIndices
    input_top_logprobs_val: TopLogprobValues
    input_top_logprobs_idx: TopLogprobIndices
    output_top_logprobs_val: TopLogprobValues
    output_top_logprobs_idx: TopLogprobIndices
    input_token_ids_logprobs_val: TokenIdsLogprobValues
    input_token_ids_logprobs_idx: TokenIdsLogprobIndices
    output_token_ids_logprobs_val: TokenIdsLogprobValues
    output_token_ids_logprobs_idx: TokenIdsLogprobIndices
    output_token_entropy_val: Optional[List[Optional[float]]]
    # Per-request chunks of output-token sampling supports. None when no request
    # in the batch asks for return_sampling_mask.
    output_token_sampling_mask: Optional[List[List]]
    # Per-request chunks of selected-token logprobs renormalized over the
    # corresponding sampling supports. None when sampling masks are not returned.
    output_token_sampling_logprobs: Optional[List[List]]

    # Hidden states
    output_hidden_states: OutputHiddenStates

    # Per-request routed experts (input + output tokens), shape
    # (token, layer, top_k). DetokenizerManager encodes to base64 into
    # BatchStrOutput; on the skip_tokenizer_init path the scheduler sends this
    # straight to TokenizerManager, which encodes on demand.
    # 中译：逐请求的「路由专家」（含输入+输出 token），形状为 (token, layer, top_k)。
    # 中译：逐请求的索引器 top-k（如 DeepSeek 稀疏注意力选中的 token），同样会被编码为 base64。
    routed_experts: Optional[List[Optional[torch.Tensor]]]

    indexer_topk: Optional[List[Optional[torch.Tensor]]]

    # The information of placeholder tokens (e.g., image token)
    # idx is the index of the token in the prompt after expansion.
    # val is the length of padded tokens after expansion.
    # 中译：占位 token（如图像 token）的信息。
    # 中译：每个请求被「回撤（retract）」的次数。显存不足时请求可能被换出重跑。
    # 中译：训练器的 step id（用于 RL 等场景），用来知道本次采样用的是哪一步的权重。
    # 中译：用于 DP（数据并行）负载均衡的负载信息。
    # 中译：自定义附加信息（按 key 对应逐请求的值列表）。
    # 中译：缓存 token 按来源（显存/主机内存/外部存储）的细分统计。
    # 中译：处理每个请求的调度器所在的 DP rank。
    # 中译：可观测性——每个请求在调度器各阶段的耗时统计。
    placeholder_tokens_idx: Optional[List[Optional[List[int]]]]
    placeholder_tokens_val: Optional[List[Optional[List[int]]]]

    # Number of times each request was retracted.
    retraction_counts: Optional[List[int]] = None

    # Per-item beam carrier; None entries are non-beam items in a mixed
    # batch (the whole field is None when the batch has no beam item).
    beam_search_output: Optional[List[Optional[BeamSearchOutput]]] = None

    weight_versions: Optional[List[Optional[WeightVersionSpans]]] = None

    # The trainer step id. Used to know which step's weights are used for sampling.
    token_steps: Optional[List[List[int]]] = None

    # Customized info
    customized_info: Optional[PickleWrapper] = None
    # Detailed breakdown of cached tokens by source (device/host/storage)
    cached_tokens_details: Optional[List[Optional[CachedTokensDetails]]] = None
    # DP rank of the scheduler that processed each request
    dp_ranks: Optional[List[Optional[int]]] = None

    # For observability
    # Pickled Optional[List[SchedulerReqTimeStats]]
    time_stats: Optional[PickleWrapper] = None

    # Multimodal prompt token counts (image/audio/video). None when not applicable.
    image_tokens: Optional[List[int]] = None
    audio_tokens: Optional[List[int]] = None
    video_tokens: Optional[List[int]] = None

    # Verify count: number of verification forward passes
    spec_verify_ct: Optional[List[int]] = None
    # Accepted drafts
    spec_num_correct_drafts: Optional[List[int]] = None
    spec_num_block_accept_tokens: Optional[List[int]] = None
    spec_num_cap_tokens: Optional[List[int]] = None
    # Acceptance histogram
    spec_correct_drafts_histogram: Optional[List[List[int]]] = None
    spec_cap_lens_histogram: Optional[List[List[int]]] = None

    # Scheduler-side flat assembly of prompt top logprobs for requests with
    # return_flat_raw_top_logprobs: float32 / int32 [rows, k] arrays plus the
    # leading-null count (see build_flat_input_top_logprobs_arrays). For such
    # requests the nested input_top_logprobs_val/idx entry is empty. None when
    # no request in the batch uses the flat format.
    input_top_logprobs_val_flat: Optional[List[Optional[np.ndarray]]] = None
    input_top_logprobs_idx_flat: Optional[List[Optional[np.ndarray]]] = None
    input_top_logprobs_flat_null_prefix: Optional[List[Optional[int]]] = None


class BatchStrOutput(BaseBatchReq, kw_only=True):
    # The finish reason
    finished_reasons: List[Optional[FinishReasonDict]]
    # The output decoded strings
    output_strs: List[str]  # 中译：解码后的输出文本（本类核心字段，流式时为本次增量文本）
    # The token ids
    output_ids: Optional[List[array]]

    # Token counts
    # 中译：各类 token 计数（逐请求）。
    prompt_tokens: List[int]  # 提示词 token 数
    completion_tokens: List[int]  # 生成 token 数
    reasoning_tokens: List[int]  # 推理（thinking）阶段 token 数
    cached_tokens: List[int]  # 命中缓存的 token 数

    # Logprobs
    input_token_logprobs_val: TokenLogprobValues
    input_token_logprobs_idx: TokenLogprobIndices
    output_token_logprobs_val: TokenLogprobValues
    output_token_logprobs_idx: TokenLogprobIndices
    input_top_logprobs_val: TopLogprobValues
    input_top_logprobs_idx: TopLogprobIndices
    output_top_logprobs_val: TopLogprobValues
    output_top_logprobs_idx: TopLogprobIndices
    input_token_ids_logprobs_val: TokenIdsLogprobValues
    input_token_ids_logprobs_idx: TokenIdsLogprobIndices
    output_token_ids_logprobs_val: TokenIdsLogprobValues
    output_token_ids_logprobs_idx: TokenIdsLogprobIndices
    output_token_entropy_val: Optional[List[Optional[float]]]
    # Detokenizer pass-through for BatchTokenIDOutput.output_token_sampling_*.
    # None when sampling masks are not returned.
    output_token_sampling_mask: Optional[List[List]]
    output_token_sampling_logprobs: Optional[List[List]]

    # Hidden states
    output_hidden_states: OutputHiddenStates

    # Per-request routed experts, base64-encoded by DetokenizerManager off the
    # tokenizer hot path. Underlying tensor shape is (token, layer, top_k);
    # see BatchTokenIDOutput.routed_experts.
    # 中译：逐请求的路由专家，已由 DetokenizerManager 在热路径外编码为 base64 字符串；
    routed_experts: Optional[List[Optional[str]]]

    indexer_topk: Optional[List[Optional[str]]]

    # The information of placeholder tokens (e.g., image token)
    # idx is the index of the token in the prompt after expansion.
    # val is the length of padded tokens after expansion.
    # 中译：占位 token（如图像 token）信息：idx 为扩展后位置，val 为扩展后填充长度。
    placeholder_tokens_idx: Optional[List[Optional[List[int]]]]
    placeholder_tokens_val: Optional[List[Optional[List[int]]]]

    # Number of times each request was retracted.
    retraction_counts: Optional[List[int]] = None

    # Per-item beam carrier; None entries are non-beam items in a mixed
    # batch (the whole field is None when the batch has no beam item).
    beam_search_output: Optional[List[Optional[BeamSearchOutput]]] = None

    weight_versions: Optional[List[Optional[WeightVersionSpans]]] = None

    # The trainer step id. Used to know which step's weights are used for sampling.
    token_steps: Optional[List[List[int]]] = None

    # Customized info
    customized_info: Optional[PickleWrapper] = None
    # Detailed breakdown of cached tokens by source (device/host/storage)
    cached_tokens_details: Optional[List[Optional[CachedTokensDetails]]] = None
    # DP rank of the scheduler that processed each request
    dp_ranks: Optional[List[Optional[int]]] = None

    # For observability
    # Pickled Optional[List[SchedulerReqTimeStats]]
    time_stats: Optional[PickleWrapper] = None

    # Multimodal prompt token counts (image/audio/video). None when not applicable.
    image_tokens: Optional[List[int]] = None
    audio_tokens: Optional[List[int]] = None
    video_tokens: Optional[List[int]] = None

    # Verify count: number of verification forward passes
    spec_verify_ct: Optional[List[int]] = None
    # Accepted drafts
    spec_num_correct_drafts: Optional[List[int]] = None
    spec_num_block_accept_tokens: Optional[List[int]] = None
    spec_num_cap_tokens: Optional[List[int]] = None
    # Acceptance histogram
    spec_correct_drafts_histogram: Optional[List[List[int]]] = None
    spec_cap_lens_histogram: Optional[List[List[int]]] = None

    # Detokenizer pass-through for the scheduler-side flat prompt top logprob
    # arrays; see BatchTokenIDOutput.input_top_logprobs_val_flat.
    input_top_logprobs_val_flat: Optional[List[Optional[np.ndarray]]] = None
    input_top_logprobs_idx_flat: Optional[List[Optional[np.ndarray]]] = None
    input_top_logprobs_flat_null_prefix: Optional[List[Optional[int]]] = None


class BatchEmbeddingOutput(BaseBatchReq, kw_only=True):
    # The finish reason
    finished_reasons: List[Optional[FinishReasonDict]]
    # The output embedding
    embeddings: List[Union[List[Union[float, List[float]]], Dict[int, float], float]]
    # Token counts
    prompt_tokens: List[int]  # 中译：提示词 token 数
    cached_tokens: List[int]  # 中译：命中缓存的 token 数
    # Placeholder token info
    # 中译：可选的池化隐藏状态（pooling head 之前的输出）。整批堆叠成单个张量发送，减少 pickle 开销。
    # 中译：清空分层缓存（HiCache，多级 KV 缓存）的请求。
    # 中译：清空 HiCache 的应答，success 表示是否成功。
    # 中译：刷新（清空）前缀缓存（prefix cache）的请求；timeout_s 为可选超时。
    # 中译：刷新缓存的应答。
    placeholder_tokens_idx: Optional[List[Optional[List[int]]]]
    placeholder_tokens_val: Optional[List[Optional[List[int]]]]

    # Number of times each request was retracted.
    retraction_counts: Optional[List[int]] = None
    # Detailed breakdown of cached tokens by source (device/host/storage)
    cached_tokens_details: Optional[List[Optional[CachedTokensDetails]]] = None

    # For observability
    # Pickled Optional[List[SchedulerReqTimeStats]]
    time_stats: Optional[PickleWrapper] = None

    # Optional pooled hidden states (pre-head transformer output).
    # Two IPC formats, disambiguated by len vs len(rids):
    #   Stacked:     [stacked_tensor(N, ...)] — len 1, reduces pickle overhead
    #   Non-stacked: [t0, t1, ..., tN]       — len N, when shapes differ or None entries exist
    pooled_hidden_states: Optional[List[Optional[torch.Tensor]]] = None


class ClearHiCacheReqInput(BaseReq, kw_only=True):
    pass


class ClearHiCacheReqOutput(BaseReq, kw_only=True):
    success: bool


class FlushCacheReqInput(BaseReq, kw_only=True):
    timeout_s: Optional[float] = None


class FlushCacheReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str = ""


    # 中译：向引擎添加「外部语料（external corpus）」的请求，用于把外部文档预置进缓存以加速命中。
    # 中译：添加外部语料的应答。
class AddExternalCorpusReqInput(BaseReq, kw_only=True):
    corpus_id: Optional[str] = None
    file_path: Optional[str] = None
    documents: Optional[List[str]] = None
    token_chunks: Optional[List[List[int]]] = None


class AddExternalCorpusReqOutput(BaseReq, kw_only=True):
    success: bool
    corpus_id: str = ""
    message: str = ""
    loaded_token_count: int = 0  # 实际载入的 token 数


    # 中译：移除指定外部语料的请求。
    # 中译：移除外部语料的应答。
class RemoveExternalCorpusReqInput(BaseReq, kw_only=True):
    corpus_id: str


class RemoveExternalCorpusReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str = ""


    # 中译：列出当前所有外部语料的请求。
    # 中译：列出外部语料的应答，corpus_token_counts 为各语料 id 到其 token 数的映射。
class ListExternalCorporaReqInput(BaseReq, kw_only=True):
    pass


class ListExternalCorporaReqOutput(BaseReq, kw_only=True):
    success: bool
    corpus_token_counts: Dict[str, int] = msgspec.field(default_factory=dict)
    message: str = ""


class AttachHiCacheStorageReqInput(BaseReq, kw_only=True):
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


class AttachHiCacheStorageReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str = ""


class DetachHiCacheStorageReqInput(BaseReq, kw_only=True):
    """Dynamically detach (disable) HiCache storage backend at runtime."""

    pass


class DetachHiCacheStorageReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str = ""


class PauseGenerationReqInput(BaseReq, kw_only=True):
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


class ContinueGenerationReqInput(BaseReq, kw_only=True):
    # Call torch.cuda.empty_cache() before un-pausing. Returns blocks
    # cached by the PyTorch allocator (left over from transient allocs
    # during post-weight-update processing) back to the driver before
    # inference resumes, with no race against active streams. Set to
    # False to skip the empty_cache call.
    # 中译：恢复生成（解除暂停）的请求。torch_empty_cache=True 时在解除前先调用
    #       torch.cuda.empty_cache()，把权重更新后处理期残留的临时分配块归还给驱动，
    #       且不与活跃 stream 竞争；置 False 可跳过该调用。
    torch_empty_cache: bool = True


class TokenizerWorkerRegistrationReq(BaseReq, kw_only=True):
    """Sent by each TokenizerWorker on startup to register its IPC name with the router.

    中译：多 tokenizer 模式下，每个 TokenizerWorker 启动时向路由器注册自己 IPC 名的消息。
    """

    worker_ipc_name: str  # 中译：该 worker 的 IPC 名


class PauseContinueBroadcastReq(BaseReq, kw_only=True):
    """Broadcast from router to all workers to set is_pause state.

    中译：路由器向所有 worker 广播暂停/恢复状态的消息。
    """

    is_pause: bool  # 中译：True 暂停 / False 恢复


class UpdateWeightFromDiskReqInput(BaseReq, kw_only=True):
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
    # 中译：从磁盘更新权重的应答。
    flush_cache: bool = True
    # Tensor metadata from the JSON request body, so it is already msgpack-native.
    manifest: Optional[Dict[str, Any]] = None


class UpdateWeightFromDiskReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str
    # Number of paused requests during weight sync.
    num_paused_requests: int = 0


class UpdateWeightsFromDistributedReqInput(BaseReq, kw_only=True):
    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
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


    # 中译：从分布式组更新权重的应答。
class UpdateWeightsFromDistributedReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


class UpdateWeightsFromTensorReqInput(BaseReq, kw_only=True):
    """Internal IPC request for updating model weights from serialized tensors."""

    # Serialized named tensors, normalized to raw MultiprocessingSerializer
    # bytes before scheduler IPC. Python Engine callers construct this field
    # with bytes directly. FastAPI HTTP callers send base64 strings because JSON
    # has no bytes type; the Annotated Base64Bytes marker is used only by the
    # msgspec-to-Pydantic schema for the HTTP protocol to decode those strings.
    serialized_named_tensors: Annotated[List[bytes], Base64Bytes()]
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


    # 中译：从张量更新权重的应答。
class UpdateWeightsFromTensorReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


class InitWeightsSendGroupForRemoteInstanceReqInput(BaseReq, kw_only=True):
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
class UpdateWeightsFromIPCReqInput(BaseReq, kw_only=True):
    # ZMQ socket paths for each device UUID
    zmq_handles: Dict[str, str]  # 中译：各设备 UUID 对应的 ZMQ 句柄路径
    # Whether to flush cache after weight update
    flush_cache: bool = True
    # Optional: Update weight version along with weights
    weight_version: Optional[str] = None
    # Whether to call torch.cuda.empty_cache() during flush
    torch_empty_cache: bool = False


    # 中译：通过 IPC 更新权重的应答。
class UpdateWeightsFromIPCReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


    # 中译：初始化权重发送组的应答。
class InitWeightsSendGroupForRemoteInstanceReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


class SendWeightsToRemoteInstanceReqInput(BaseReq, kw_only=True):
    # The master address
    master_address: str
    # The ports for each rank's communication group
    ports: str
    # The group name
    group_name: str = "weight_send_group"


    # 中译：发送权重到远程实例的应答。
class SendWeightsToRemoteInstanceReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


    # 中译：更新专家（MoE expert）备份的请求。
    # 中译：把权重备份到 DRAM（主机内存）的请求。
class UpdateExpertBackupReq(BaseReq, kw_only=True):
    pass


class ExpertWeightPointer(msgspec.Struct, kw_only=True, array_like=True):
    # One expert weight's pointer + byte length in the DRAM backup buffer.
    # array_like: the map has tens of thousands of entries, so positional
    # encoding drops the repeated field names from the wire.
    weight_ptr: int
    byte_size: int


class BackupDramReq(BaseReq, kw_only=True):
    rank: int
    weight_pointer_map: Dict[str, ExpertWeightPointer]
    session_id: str
    buffer_size: int


class InitWeightsUpdateGroupReqInput(BaseReq, kw_only=True):
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


    # 中译：初始化权重更新组的应答。
class InitWeightsUpdateGroupReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


    # 中译：销毁权重更新通信组的请求。
    # 中译：销毁权重更新组的应答。
class DestroyWeightsUpdateGroupReqInput(BaseReq, kw_only=True):
    group_name: str = "weight_update_group"


class DestroyWeightsUpdateGroupReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


    # 中译：仅更新权重版本号（不动权重本身）的请求。
class UpdateWeightVersionReqInput(BaseReq, kw_only=True):
    # The new weight version
    new_version: str  # 中译：新版本号
    # Whether to abort all running requests before updating
    abort_all_requests: bool = True


    # 中译：按参数名读取权重的请求（调试/校验用）。
    # 中译：按名读取权重的应答，parameter 为截断后的权重值列表。
class UpdateWeightVersionReqOutput(BaseReq, kw_only=True):
    pass


class GetWeightsByNameReqInput(BaseReq, kw_only=True):
    name: str
    truncate_size: int = 100


class GetWeightsByNameReqOutput(BaseReq, kw_only=True):
    # A flat List[float] or a per-row List[List[float]]. The union is on the
    # element: Union[List[float], List[List[float]]] is invalid msgspec.
    parameter: Optional[List[Union[float, List[float]]]]


class ReleaseMemoryOccupationReqInput(BaseReq, kw_only=True):
    # Optional tags to identify the memory region, which is primarily used for RL
    # Currently we only support `weights` and `kv_cache`
    # 中译：释放显存占用的请求（主要用于 RL，把权重/KV cache 显存暂时让出给训练）。
    #       tags 标识要释放的内存区域，目前仅支持 `weights` 和 `kv_cache`。
    tags: Optional[List[str]] = None


    # 中译：释放显存占用的应答。
class ReleaseMemoryOccupationReqOutput(BaseReq, kw_only=True):
    pass


class ResumeMemoryOccupationReqInput(BaseReq, kw_only=True):
    # Optional tags to identify the memory region, which is primarily used for RL
    # Currently we only support `weights` and `kv_cache`
    # 中译：恢复（重新占用）显存的请求，与 ReleaseMemoryOccupation 配对使用。
    tags: Optional[List[str]] = None


    # 中译：恢复显存占用的应答。
    # 中译：校验权重的请求，action 指定校验方式（默认 checksum 校验和）。
class ResumeMemoryOccupationReqOutput(BaseReq, kw_only=True):
    pass


class CheckWeightsReqInput(BaseReq, kw_only=True):
    action: str = "checksum"
    allow_quant_error: bool = False


    # 中译：校验权重的应答，payload 携带校验结果细节。
# Wire versions of the pydantic ParallelismInfo/ChecksumInfo in
# sglang.srt.utils.weight_checker. Not array_like: the payload is read by field
# name and re-serialized to JSON, so it must stay a {field: value} map.
class ParallelismInfo(msgspec.Struct, kw_only=True):
    tp_rank: int
    tp_size: int
    dp_rank: int
    dp_size: int
    pp_rank: int
    pp_size: int
    rank: int
    size: int


class ChecksumInfo(msgspec.Struct, kw_only=True):
    checksums: Dict[str, str]
    per_gpu_checksum: str
    parallelism_info: ParallelismInfo


class CheckWeightsReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str
    # One ChecksumInfo per TP rank. The producer wraps the tp==1 result in a
    # one-element list so the shape is always a list.
    payload: Optional[List[ChecksumInfo]] = None


    # 中译：人为减速的请求（调试/复现时序问题用）。每次前向后睡眠 forward_sleep_time 秒。
    # 中译：减速请求的应答。
class SlowDownReqInput(BaseReq, kw_only=True):
    forward_sleep_time: Optional[float]


class SlowDownReqOutput(BaseReq, kw_only=True):
    pass


class AbortReq(BaseReq, kw_only=True):
    # Whether to abort all requests
    abort_all: bool = False
    # The finished reason data (from BaseFinishReason.to_json())
    finished_reason: Optional[FinishReasonDict] = None
    abort_message: Optional[str] = None
    weight_versions: Optional[WeightVersionSpans] = None

    def __post_init__(self):
        # FIXME: This is a hack to keep the same with the old code
        # 中译：兼容旧代码的 hack——rid 为 None 时置为空串。
        if self.rid is None:
            self.rid = ""


    # 中译：各 rank 是否活跃的状态列表。
    # 中译：查询调度器内部状态的请求。
    # 中译：内部状态查询应答，internal_state 为状态键值对。
    # 中译：在运行时修改调度器内部参数的请求（server_args 为待覆盖的配置项）。
    # 中译：设置内部状态的应答，updated 表示是否生效，并回传当前 server_args。
class ActiveRanksOutput(BaseReq, kw_only=True):
    status: List[bool]


class ElasticScaleUpdateReq(BaseReq, kw_only=True):
    """Report asynchronous Elastic EP scale completion or failure."""

    success: bool
    effective_ep_size: int
    slot_offset: int = 0
    slot_count: int = 0
    error: Optional[str] = None


class ScaleElasticEPReqInput(BaseReq, kw_only=True):
    """Request to scale EP by changing the effective EP size (dp_attention mode)."""

    new_ep_size: int


class ScaleElasticEPReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str
    old_ep_size: int = 0
    new_ep_size: int = 0
    pending_ep_size: Optional[int] = None
    scale_phase: str = "idle"


class GetInternalStateReq(BaseReq, kw_only=True):
    pass


class GetInternalStateReqOutput(BaseReq, kw_only=True):
    # A vars() dump of ServerArgs, left untyped because a struct would drift. The
    # producer sanitizes it with msgspec_to_builtins so every value is
    # msgpack-native.
    internal_state: Dict[str, Any]


class SetInternalStateReq(BaseReq, kw_only=True):
    # Only numeric scheduler knobs are accepted (see Scheduler.set_internal_state).
    server_args: Dict[str, Union[int, float]]


class SetInternalStateReqOutput(BaseReq, kw_only=True):
    updated: bool


class ProfileReqType(Enum):
    START_PROFILE = 1
    STOP_PROFILE = 2


class ProfileReq(BaseReq, kw_only=True):
    req_type: ProfileReqType = ProfileReqType.START_PROFILE
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
    record_shapes: Optional[bool] = None
    profile_id: Optional[str] = None
    # Merge profiles from all ranks into a single trace
    merge_profiles: bool = False  # 中译：是否把各 rank 的 profile 合并成一个 trace
    # The prefix of the profile filenames
    profile_prefix: Optional[str] = None  # 中译：trace 文件名前缀
    # Only profile these stages and ignore others
    # 中译：profile 请求类型——启动 / 停止。
    # 中译：实际下发给调度器的 profile 控制消息（由 ProfileReqInput + 类型组装而成）。
    # 中译：profile 控制的应答。
    profile_stages: Optional[List[str]] = None
    # Add iteration-level annotations (KV / request aggregates) for roofline-style analysis
    detailed_annotations: bool = False


class ProfileReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


    # 中译：冻结垃圾回收（freeze GC）的请求，减少长期对象的反复扫描开销。
    # 中译：打开一个会话（用于持续对话）的请求。
    # 中译：关闭指定会话的请求。
    # 中译：打开会话的应答，返回实际会话 id 与是否成功。
class FreezeGCReq(BaseReq, kw_only=True):
    pass


class ShutdownReq(BaseReq, kw_only=True):
    # Broadcast across TP ranks via the normal recv path, so all ranks break
    # the scheduler loop on the same iteration.
    pass


class ConfigureLoggingReq(BaseReq, kw_only=True):
    log_requests: Optional[bool] = None
    log_requests_level: Optional[int] = None
    log_requests_format: Optional[str] = None
    log_level: Optional[str] = None
    dump_requests_folder: Optional[str] = None
    dump_requests_threshold: Optional[int] = None
    crash_dump_folder: Optional[str] = None
    dump_requests_exclude_meta_keys: Optional[List[str]] = None


class OpenSessionReqInput(BaseReq, kw_only=True):
    capacity_of_str_len: int
    session_id: Optional[str] = None
    streaming: Optional[bool] = None
    timeout: Optional[float] = None


class CloseSessionReqInput(BaseReq, kw_only=True):
    session_id: str


class OpenSessionReqOutput(BaseReq, kw_only=True):
    session_id: Optional[str]
    success: bool


    # 中译：健康检查的输出占位对象。
class HealthCheckOutput(BaseReq, kw_only=True):
    pass


class ExpertDistributionReqType(Enum):
    # 中译：MoE 专家分布统计的控制类型——开始记录 / 停止记录 / 导出记录。
    START_RECORD = 1
    STOP_RECORD = 2
    DUMP_RECORD = 3


    # 中译：控制 MoE 专家分布统计的请求（分析负载是否均衡）。
    # 中译：专家分布统计控制的应答。
    # 中译：函数调用（function/tool calling）中的函数定义。
    # 中译：可供模型调用的工具，封装一个 Function。
class ExpertDistributionReq(BaseReq, kw_only=True):
    action: ExpertDistributionReqType


class ExpertDistributionReqOutput(BaseReq, kw_only=True):
    pass


class Function(msgspec.Struct, kw_only=True, array_like=True):
    description: Optional[str] = None
    name: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return msgspec_struct_pydantic_core_schema(cls, handler)


class Tool(msgspec.Struct, kw_only=True, array_like=True):
    function: Function
    type: str = "function"

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return msgspec_struct_pydantic_core_schema(cls, handler)


    # 中译：解析模型输出中的函数调用的请求（把文本解析成结构化的工具调用）。
class ParseFunctionCallReq(BaseReq, kw_only=True):
    text: str  # The text to parse.
    tools: List[Tool] = msgspec.field(
        default_factory=list
    )  # A list of available function tools (name, parameters, etc.).
    tool_call_parser: Optional[str] = (
        None  # Specify the parser type, e.g. 'llama3', 'qwen25', or 'mistral'. If not specified, tries all.
    )


    # 中译：把模型输出中的「推理（thinking）」部分与最终答案分离的请求。
class SeparateReasoningReqInput(BaseReq, kw_only=True):
    text: str  # The text to parse.
    reasoning_parser: str  # Specify the parser type, e.g., "deepseek-r1".
    # 中译：指定推理解析器类型（如 deepseek-r1）。
    return_blocks: bool = False  # If True, also return segmented reasoning blocks.
    # 中译：为 True 时额外返回分段后的推理块。


    # 中译：兼容 Google Vertex AI 协议的生成请求。
    # 中译：通用 RPC 调用请求——按方法名 method 调用调度器内部方法，parameters 为参数。
class VertexGenerateReqInput(BaseReq, kw_only=True):
    # Both fields come from the JSON request body, so they are already
    # msgpack-native.
    instances: List[Dict[str, Any]]
    parameters: Optional[Dict[str, Any]] = None


class RpcReqInput(BaseReq, kw_only=True):
    method: str
    # collective_rpc kwargs are flat scalars across all in-tree callers.
    parameters: Optional[Dict[str, Union[bool, int, float, str, None]]] = None


    # 中译：RPC 调用的应答。
class RpcReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


class LoadLoRAAdapterReqInput(BaseReq, kw_only=True):
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


class UnloadLoRAAdapterReqInput(BaseReq, kw_only=True):
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


class LoadLoRAAdapterFromTensorsReqInput(BaseReq, kw_only=True):
    lora_name: str
    # The PEFT adapter_config.json, already JSON — a tighter type would only add
    # decode strictness with no benefit.
    config_dict: Dict[str, Any]
    # One serialized copy of the adapter tensors per TP rank; each rank
    # deserializes only its own copy. Same normalization conventions as
    # UpdateWeightsFromTensorReqInput.serialized_named_tensors.
    serialized_named_tensors: Annotated[List[bytes], Base64Bytes()]
    pinned: bool = False
    added_tokens_config: Optional[Dict[str, int]] = None
    lora_id: Optional[str] = None
    load_format: Optional[str] = None
    expected_checksums: Optional[Dict[str, str]] = None

    def to_ref(self) -> LoRARef:
        # 中译：转换为 LoRARef；lora_path 用特殊标记 "__tensor__" 表示来自张量。
        return LoRARef(
            lora_id=self.lora_id,
            lora_name=self.lora_name,
            lora_path="__tensor__",
            pinned=self.pinned,
        )


    # 中译：LoRA 加载/卸载的统一应答。loaded_adapters 为当前已加载的适配器映射。
class LoRAUpdateOutput(BaseReq, kw_only=True):
    success: bool
    error_message: Optional[str] = None
    loaded_adapters: Optional[Dict[str, Union[str, LoRARef]]] = None


# 中译：三种 LoRA 操作（加载/卸载/从张量加载）共用同一种应答类型 LoRAUpdateOutput。
LoadLoRAAdapterReqOutput = UnloadLoRAAdapterReqOutput = (
    LoadLoRAAdapterFromTensorsReqOutput
) = LoRAUpdateOutput


class BlockReqType(Enum):
    # 中译：阻塞控制类型——阻塞 / 解除阻塞。
    BLOCK = 1
    UNBLOCK = 2


    # 中译：阻塞/解除阻塞调度器的请求。
    # 中译：本实例角色（prefill/decode/null）；它本身不是指标，仅用于区分。
    # 中译：合法的查询分区集合（核心/内存/推测解码/LoRA/PD分离/队列/全部）。
    # 中译：正在运行（已进入 running_batch、占用 KV cache 做前向）的请求数。
    # 中译：在等待队列中、尚未开始推理的请求数。
    # 中译：等待队列中「未命中前缀缓存、仍需做 prefill 计算」的输入 token 数。
    # 中译：当前正在使用（已占用 KV cache 槽位）的 token 数。
    # 中译：num_used_tokens 加上「待处理的 prefill token」（等待队列的序列长度，
    # 中译：该 rank 的最大 token 容量（KV cache 可容纳的 token 上限）。
    # 中译：token 池使用率。注意（见上方 FIXME）：它实际是所有池（KV、SWA、mamba）中
    # 中译：生成吞吐（tokens/秒），衡量该 rank 当前的产出速度。
    # 中译：前缀缓存（prefix cache）命中率。
    # 中译：整体利用率（综合衡量该 rank 的繁忙程度）。
    # 中译：最大并发运行请求数（running_batch 的容量上限）。
    # 中译：以下为可选的分项细分指标，仅当对应 include 分区被请求时才填充，否则为 None。
    # 中译：负载更新广播消息——把各 DP rank 的最新负载推送给负载均衡/路由组件。
    # 中译：设置张量转储（dump）元信息的请求（调试/排查数值问题时给 dump 打标签）。
    # 中译：设置转储元信息的应答。
    # 中译：触发「延迟张量转储」的请求。
    # 中译：延迟张量转储的应答。
    # 中译：转储器（dumper）的通用控制请求，按 method+body 调用。
class BlockReqInput(BaseReq, kw_only=True):
    req_type: BlockReqType


class SetInjectDumpMetadataReqOutput(BaseReq, kw_only=True):
    success: bool


class LazyDumpTensorsReqInput(BaseReq, kw_only=True):
    pass


class LazyDumpTensorsReqOutput(BaseReq, kw_only=True):
    success: bool


class DumperControlReqInput(BaseReq, kw_only=True):
    method: str
    # JSON request body (guarded to be a dict at the /dumper endpoint).
    body: Dict[str, Any]


    # 中译：转储器控制的应答。
class DumperControlReqOutput(BaseReq, kw_only=True):
    success: bool
    # JSON-native per-worker response dicts.
    response: List[Dict[str, Any]]
    error: str = ""


# The following request types are either defined in other files,
# or not subclasses of BaseReq/BaseBatchReq, so we skip the check for them.
_IGNORE_REQ_TYPES_CHECK = (
    GenerateReqInput.__name__,
    EmbeddingReqInput.__name__,
)


def _check_all_req_types():
    """A helper function to check all request types are defined in this file."""
    import inspect
    import sys

    all_classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    for class_type in all_classes:
        # check its name
        name = class_type[0]
        if class_type[1].__module__ != __name__:
            continue
        if name in _IGNORE_REQ_TYPES_CHECK:
            continue
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


def wrap_as_pickle(obj: object) -> object:
    if obj is None:
        return None
    if _USE_PICKLE_IPC:
        return obj
    return PickleWrapper(pickle.dumps(obj))


def unwrap_from_pickle(obj: Optional[object]) -> Optional[object]:
    if obj is None:
        return None
    if _USE_PICKLE_IPC:
        return obj
    if not isinstance(obj, PickleWrapper):
        # Already materialized: the embedded Rust server attaches in-process
        # objects (native-MM `mm_inputs`) without a pickle hop.
        return obj
    return pickle.loads(obj.data)


_struct_types = tuple(
    cls
    for cls in BaseReq.__subclasses__()
    + BaseBatchReq.__subclasses__()
    + [PickleWrapper]
)
# Primitive types that msgpack can serialize directly without PickleWrapper.
# Do not include str here: msgspec rejects a Union containing both str and bytes
# as multiple str-like arms. Top-level strings use PickleWrapper; string fields
# inside typed structs are still decoded by their struct schemas.
_primitive_types = (int, float, bool, bytes)
_all_types = _struct_types + _primitive_types

_msgpack_encoder = msgspec.msgpack.Encoder(enc_hook=enc_hook)
_msgpack_decoder = msgspec.msgpack.Decoder(
    Union[_all_types], dec_hook=dec_hook, ext_hook=ext_hook
)
_USE_PICKLE_IPC = envs.SGLANG_USE_PICKLE_IPC.get()


def hook_custom_types(*new_types: Type):
    global _msgpack_decoder, _all_types
    _all_types = tuple(dict.fromkeys(_all_types + new_types))
    _msgpack_decoder = msgspec.msgpack.Decoder(
        Union[_all_types], dec_hook=dec_hook, ext_hook=ext_hook
    )


def _maybe_wrap_pickle(obj: Any) -> Any:
    if isinstance(obj, (msgspec.Struct, *_primitive_types)):
        return obj

    raise TypeError(
        f"Cannot serialize object of type {type(obj)} over msgpack IPC. "
        "Add a precise msgspec-compatible type, or use an explicit PickleWrapper "
        "field via wrap_as_pickle(...) for the opaque payload."
    )


def _maybe_unwrap_pickle(obj: Any) -> Any:
    if isinstance(obj, PickleWrapper):
        obj = pickle.loads(obj.data)
        if envs.SGLANG_LOG_PICKLE_IPC_OBJECTS.get():
            logger.info(f"Object of type {type(obj)} is unwrapped from PickleWrapper.")
        return obj

    return obj


def msgpack_encode(obj: Any) -> bytes:
    return _msgpack_encoder.encode(_maybe_wrap_pickle(obj))


def msgpack_decode(data: bytes) -> Any:
    return _maybe_unwrap_pickle(_msgpack_decoder.decode(data))


def sock_send(socket: zmq.Socket, obj: Any, flags: int = 0) -> None:
    if _USE_PICKLE_IPC:
        socket.send_pyobj(obj, flags=flags, protocol=pickle.HIGHEST_PROTOCOL)
        return

    socket.send(msgpack_encode(obj), flags=flags)


def sock_recv(socket: zmq.Socket, flags: int = 0) -> Any:
    if _USE_PICKLE_IPC:
        return socket.recv_pyobj(flags=flags)

    data = socket.recv(flags=flags)
    return msgpack_decode(data)


async def async_sock_send(socket: zmq.asyncio.Socket, obj: Any, flags: int = 0) -> None:
    if _USE_PICKLE_IPC:
        await socket.send_pyobj(obj, flags=flags, protocol=pickle.HIGHEST_PROTOCOL)
        return

    await socket.send(msgpack_encode(obj), flags=flags)


async def async_sock_recv(socket: zmq.asyncio.Socket, flags: int = 0) -> Any:
    if _USE_PICKLE_IPC:
        return await socket.recv_pyobj(flags=flags)

    data = await socket.recv(flags=flags)
    return msgpack_decode(data)
