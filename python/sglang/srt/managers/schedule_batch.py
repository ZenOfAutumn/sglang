from __future__ import annotations

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import (
    get_disagg,
    get_parallel,
    get_schedule,
    get_serving,
    get_spec,
    mamba_cache_chunk_size,
    mamba_checkpoint_grid,
    mamba_extra_buffer_enabled,
    mamba_extra_buffer_lazy_enabled,
    mamba_track_grid,
)
from sglang.srt.utils.common import (
    Range,
    ceil_align,
    flatten_arrays_to_pinned_cpu,
    is_pin_memory_available,
)
from sglang.srt.utils.weight_versions import (
    WeightVersionEvent,
    truncate_weight_version_events,
)

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
Store information about requests and batches.

The following is the flow of data structures for a batch:

ScheduleBatch -> ForwardBatch

- ScheduleBatch is managed by `scheduler.py::Scheduler`.
  It contains high-level scheduling data. Most of the data is on the CPU.
- ForwardBatch is managed by `model_runner.py::ModelRunner`.
  It contains low-level tensor data. Most of the data consists of GPU tensors.
  It is constructed directly from a ScheduleBatch by `ForwardBatch.init_new`.

中译：本文件定义「请求（Req）」与「批次（ScheduleBatch）」这两个最核心的数据结构，
      以及它们的状态机（什么时候完成、如何 prefill/decode、如何过滤/合并）。

数据流（一个 batch 的生命周期）：ScheduleBatch -> ForwardBatch
- ScheduleBatch 由调度器 `scheduler.py::Scheduler` 管理，承载「高层调度信息」，
  大部分数据在 CPU 上（请求列表、各种长度、CPU 镜像张量等）。
- ForwardBatch 由 `model_runner.py::ModelRunner` 管理，承载「底层张量数据」，
  大部分是 GPU 张量；它由 `ForwardBatch.init_new` 直接从一个 ScheduleBatch 构造而来。
- 注意：历史上 ScheduleBatch 与 ForwardBatch 之间曾有一层 ModelWorkerBatch 中间结构，
  现已移除，ForwardBatch 直接读取 ScheduleBatch 的字段（哪些字段会被读取见各字段分组注释）。
"""

import copy
import dataclasses
import logging
import re
import sys
from array import array
from concurrent.futures import Future
from enum import Enum, auto
from functools import lru_cache
from http import HTTPStatus
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    TypeAlias,
    Union,
)

import msgspec
import numpy as np
import torch

from sglang.srt.beam_search.batch_tail import (
    BeamTail,
    append_beam_tail,
    beam_retraction_order,
    num_beam_member_rows,
    strip_beam_tail,
)
from sglang.srt.beam_search.fork import free_member_rows
from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.disaggregation.base import BaseKVSender
from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    ScheduleBatchDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST, DisaggregationMode
from sglang.srt.dllm.mixin.req import ReqDllmMixin
from sglang.srt.environ import envs
from sglang.srt.managers.embed_types import PositionalEmbeds
from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import (
    NewTokenRatioTracker,
)
from sglang.srt.mem_cache.allocation import (
    alloc_for_decode,
    alloc_for_extend,
)
from sglang.srt.mem_cache.allocation_sizing import get_alloc_reserve_per_decode
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    MatchPrefixParams,
    zero_match_result,
)
from sglang.srt.mem_cache.common import (
    RetractionBackup,
    evict_from_tree_cache,
    free_swa_out_of_window_slots,
    release_kv_cache,
    retraction_backup,
)
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.multimodal.transport.cuda_ipc import (
    DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
    CudaIpcTensorTransportProxy,
)
from sglang.srt.observability.metrics_collector import (
    DPCooperationInfo,
    SchedulerMetricsCollector,
)
from sglang.srt.observability.req_time_stats import (
    APIServerReqTimeStats,
    DPControllerReqTimeStats,
    SchedulerReqTimeStats,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import flatten_nested_list
from sglang.srt.utils.token_sequence_matcher import TokenSequenceMatcher

if TYPE_CHECKING:
    from typing import Any, Dict

    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.managers.scheduler_components.metrics_reporter import PrefillStats
    from sglang.srt.session.session_controller import Session
    from sglang.srt.speculative.spec_info import SpecInput, SpeculativeAlgorithm

# 中译：增量解码（incremental detokenize）的初始「环绕上下文」回看长度（token 数）。
#       首次解码时多向前看 5 个 token 作为上下文，保证跨 token 边界的字符能正确拼接。
INIT_INCREMENTAL_DETOKENIZATION_OFFSET = 5

# Constant used as the base offset for MM (multimodal) pad values.
# This ensures pad_values don't overlap with valid text token IDs.
# 中译：多模态（multimodal）pad_value 的基准偏移常量。多模态占位 token 用一个特殊
#       「伪 token id」表示，从 100 万起算，确保它与真实文本 token id 不冲突。
MM_PAD_SHIFT_VALUE = 1_000_000
_MM_HASH_MASK = (1 << 64) - 1

logger = logging.getLogger(__name__)


ReturnHiddenStatesMode = Union[bool, Literal["last"]]


def get_return_hidden_states_mode(
    return_hidden_states: ReturnHiddenStatesMode,
) -> CaptureHiddenMode:
    if return_hidden_states is True:
        return CaptureHiddenMode.FULL
    if return_hidden_states == "last":
        return CaptureHiddenMode.LAST
    if return_hidden_states is False:
        return CaptureHiddenMode.NULL
    raise ValueError(
        "return_hidden_states must be a boolean or the string literal 'last'."
    )


def get_request_return_hidden_states_mode(
    return_hidden_states: Union[List[ReturnHiddenStatesMode], ReturnHiddenStatesMode],
) -> CaptureHiddenMode:
    if isinstance(return_hidden_states, list):
        return max(
            (get_return_hidden_states_mode(mode) for mode in return_hidden_states),
            default=CaptureHiddenMode.NULL,
        )
    return get_return_hidden_states_mode(return_hidden_states)


def get_batch_return_hidden_states_mode(reqs: List[Req]) -> CaptureHiddenMode:
    mode = CaptureHiddenMode.NULL
    for req in reqs:
        mode = max(mode, req.return_hidden_states_mode)
    return mode


def need_return_hidden_states(
    return_hidden_states: Union[List[ReturnHiddenStatesMode], ReturnHiddenStatesMode],
) -> bool:
    return get_request_return_hidden_states_mode(return_hidden_states).need_capture()


@lru_cache(maxsize=1)
def sanity_check_mm_pad_shift_value(vocab_size: int) -> None:
    # 中译：健全性检查——若模型词表大小超过 MM_PAD_SHIFT_VALUE，多模态伪 token id 会与
    #       真实 token id 重叠，需调大 MM_PAD_SHIFT_VALUE。用 lru_cache 保证只检查一次。
    if vocab_size > MM_PAD_SHIFT_VALUE:
        raise ValueError(
            f"Model vocab_size ({vocab_size}) exceeds MM_PAD_SHIFT_VALUE ({MM_PAD_SHIFT_VALUE}). "
            f"MM pad_values may overlap with valid token IDs. "
            f"Please increase MM_PAD_SHIFT_VALUE in schedule_batch.py."
        )


def split_cached_prefix_by_tier(
    prefix_len: int, host_hit_len: int, storage_hit_len: int
) -> tuple[int, int, int]:
    """Split a request's cached prefix into (device, host, storage) tokens.

    prefix_len is len(prefix_indices) AFTER host load-back, so it contains the
    host-loaded portion; host_hit_len in turn contains the storage-prefetched
    portion (storage is clamped to it to handle edge cases).
    """
    storage = min(host_hit_len, storage_hit_len)
    host = host_hit_len - storage
    device = max(0, prefix_len - host_hit_len)
    return device, host, storage


def _compute_pad_value(hash: int) -> int:
    """Compute pad value from hash.

    中译：由特征 hash 计算多模态占位 token 的 pad_value。基准偏移 + (hash mod 2^30)，
          既保证落在词表之外，又让相同特征得到相同 pad_value（便于 RadixAttention 缓存）。
    """
    return MM_PAD_SHIFT_VALUE + (hash % (1 << 30))


# 中译：FinishReason（完成原因）体系——记录一个请求为什么结束，并能转成 OpenAI 风格 JSON。
class BaseFinishReason:
    """中译：所有「完成原因」的基类，子类需实现 to_json()。"""

    def to_json(self):
        raise NotImplementedError()


# 中译：命中了「停止 token」而结束（matched 为命中的 token id 或 id 列表）。
class FINISH_MATCHED_TOKEN(BaseFinishReason):
    def __init__(self, matched: Union[int, List[int]]):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


# 中译：命中了「停止字符串」而结束（matched 为命中的字符串）。
class FINISH_MATCHED_STR(BaseFinishReason):
    def __init__(self, matched: str):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


# 中译：命中了「停止正则」而结束（matched 为命中的正则字符串）。
class FINISHED_MATCHED_REGEX(BaseFinishReason):
    def __init__(self, matched: str):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


# 中译：因达到最大生成长度（max_new_tokens）而结束。
class FINISH_LENGTH(BaseFinishReason):
    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def to_json(self):
        return {
            "type": "length",  # to match OpenAI API's return value
            "length": self.length,
        }


# 中译：请求被中止（abort）而结束，常见于出错或主动取消，携带错误信息与 HTTP 状态码。
class FINISH_ABORT(BaseFinishReason):
    def __init__(self, message=None, status_code=None, err_type=None):
        super().__init__()
        self.message = message or "Aborted"
        self.status_code = status_code
        self.err_type = err_type

    def to_json(self):
        return {
            "type": "abort",
            "message": self.message,
            "status_code": self.status_code,
            "err_type": self.err_type,
        }


class Modality(Enum):
    """中译：多模态模态类型枚举——图像 / 视频 / 音频。"""

    IMAGE = auto()
    VIDEO = auto()
    AUDIO = auto()

    @staticmethod
    def from_str(modality_str: str):
        # 中译：从字符串（大小写不敏感）解析出 Modality；非法值抛 ValueError。
        try:
            return Modality[modality_str.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid modality string: {modality_str}. Valid modalities are: {[m.name for m in Modality]}"
            )

    @staticmethod
    def all():
        return [Modality.IMAGE, Modality.VIDEO, Modality.AUDIO]


class MultimodalInputFormat(Enum):
    """中译：多模态输入的数据形态。
    - NORMAL：常规原始特征（如 pixel_values）。
    - PROCESSOR_OUTPUT：processor 处理后的输出。
    - PRECOMPUTED_EMBEDDING：已预计算好的编码器 embedding（直接当作最终视觉/音频嵌入）。
    """

    NORMAL = auto()
    PROCESSOR_OUTPUT = auto()
    PRECOMPUTED_EMBEDDING = auto()


# Msgpack-native containers and Ext-decoded tensor/transport leaves. Tuple
# containers intentionally decode as lists, matching msgpack's native model.
MultimodalDataValue: TypeAlias = object


class MultimodalDataItem(msgspec.Struct, kw_only=True, dict=True, array_like=True):
    """
    One MultimodalDataItem represents a single multimodal input (one image, one video, or one audio).
    For example, if there are 3 images and 1 audio, there will be 4 MultimodalDataItems.

    Each item has its own hash and pad_value, enabling per-image RadixAttention caching.

    We put the common fields first and the model-specific fields in model_specific_data.

    中译：一个 MultimodalDataItem 代表「一份」多模态输入（一张图 / 一段视频 / 一段音频）。
          例如 3 图 + 1 音频 = 4 个 item。每个 item 有自己的 hash 和 pad_value，
          从而支持「按图」做 RadixAttention 前缀缓存（相同图复用 KV）。
          通用字段放前面，模型特有字段统一塞进 model_specific_data 字典。
    """

    modality: Modality
    hash: Optional[int] = None
    pad_value: Optional[int] = None
    offsets: Optional[List[Tuple[int, int]]] = None

    format: MultimodalInputFormat = MultimodalInputFormat.NORMAL  # 数据形态（见 MultimodalInputFormat）

    # the raw features returned by processor, e.g. pixel_values or audio_features
    # 中译：processor 返回的原始特征（如 pixel_values / audio_features）。
    # 中译：预计算好的 embedding，作为最终编码器嵌入直接使用。
    # 中译：模型特有数据（不同 VL 模型需要的额外字段）统一放这个字典里。
        # 中译：让 model_specific_data 里的键可以像普通属性一样直接 item.xxx 访问。
    feature: Optional[MultimodalDataValue] = None
    # the precomputed embeddings, passed as final encoder embeddings
    # One and only one of the feature and precomputed_embeddings will be empty
    precomputed_embeddings: Optional[MultimodalDataValue] = None
    # Keep precomputed_embeddings on GPU after use (EPD pool/GPU receive path)
    keep_device_embedding: bool = False

    # Processor-owned tensors/arrays/scalars/transports. msgspec rejects a
    # precise union with multiple custom types, but accepts Ext-decoded values
    # under object.
    model_specific_data: Dict[str, MultimodalDataValue] = msgspec.field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.hash is not None:
            msgspec.Struct.__setattr__(self, "hash", self.hash & _MM_HASH_MASK)

    def __getattr__(self, name: str) -> MultimodalDataValue:
        if name in self.model_specific_data:
            return self.model_specific_data[name]

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: MultimodalDataValue) -> None:
        if name in self.__struct_fields__:
            if name == "hash" and isinstance(value, int):
                value &= _MM_HASH_MASK
            msgspec.Struct.__setattr__(self, name, value)
        else:
            self.model_specific_data[name] = value

    def __setitem__(self, key: str, value: MultimodalDataValue) -> None:
        setattr(self, key, value)

    def set(self, key: str, value: MultimodalDataValue) -> None:
        self.__setitem__(key, value)

    def set_hash(self, hash_value: int) -> None:
        self.hash = hash_value
        self.pad_value = _compute_pad_value(hash_value)

    @staticmethod
    def is_empty_list(l):
        if l is None:
            return True
        return len([item for item in flatten_nested_list(l) if item is not None]) == 0

    def set_pad_value(self):
        if self.pad_value is not None:
            return

        from sglang.srt.multimodal.cache import resolve_multimodal_item_hash

        self.hash = resolve_multimodal_item_hash(
            existing_hash=self.hash,
            feature=self.feature,
            precomputed_embeddings=self.precomputed_embeddings,
        )
        self.pad_value = _compute_pad_value(self.hash)

    def is_modality(self, modality: Modality) -> bool:
        return self.modality == modality

    def is_audio(self):
        return self.modality == Modality.AUDIO

    def is_image(self):
        return self.modality == Modality.IMAGE

    def is_video(self):
        return self.modality == Modality.VIDEO

    def is_valid(self) -> bool:
        # 中译：是否为有效的多模态 item（属于图/视频/音频任一）。
        return self.is_image() or self.is_video() or self.is_audio()

    def validate(self):
        ...
        # TODO

    def is_precomputed_embedding(self):
        # 中译：是否为「预计算 embedding」形态（已是最终编码器嵌入，无需再过编码器）。
        return self.format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING

    @staticmethod
    def from_dict(obj: dict):
        # 中译：从字典构造 MultimodalDataItem（modality 支持字符串），并执行 validate。
        kwargs = dict(obj)
        modality = kwargs.pop("modality")
        if isinstance(modality, str):
            modality = Modality[modality]
        ret = MultimodalDataItem(modality=modality, **kwargs)
        ret.validate()
        return ret

    def has_cuda_ipc_proxy(self):
        return (
            isinstance(self.feature, CudaIpcTensorTransportProxy)
            or isinstance(self.precomputed_embeddings, CudaIpcTensorTransportProxy)
            or any(
                isinstance(value, CudaIpcTensorTransportProxy)
                for value in self.model_specific_data.values()
            )
        )

    def reconstruct(self, target_device: int, ipc_consumer_count: int = 1):
        """materialize cuda ipc proxy tensors in-place on target_device"""
        if isinstance(self.feature, CudaIpcTensorTransportProxy):
            consumer_count = self._resolve_transport_consumer_count(
                self.feature, ipc_consumer_count
            )
            if consumer_count == 1:
                self.feature = self.feature.reconstruct_on_target_device(target_device)
            else:
                self.feature = self.feature.reconstruct_on_target_device(
                    target_device, consumer_count=consumer_count
                )
        if isinstance(self.precomputed_embeddings, CudaIpcTensorTransportProxy):
            self.precomputed_embeddings = (
                self.precomputed_embeddings.reconstruct_on_target_device(target_device)
            )
        for extra_key in self.model_specific_data:
            if isinstance(
                self.model_specific_data[extra_key], CudaIpcTensorTransportProxy
            ):
                extra_data = self.model_specific_data[
                    extra_key
                ].reconstruct_on_target_device(target_device)
                self.model_specific_data[extra_key] = extra_data

    def can_defer_cuda_ipc_feature_reconstruction(self) -> bool:
        """Whether a DP-aware model will materialize this feature lazily.

        Hashing and pad-value generation must already have completed on the
        tokenizer worker.  Any additional IPC proxy would still need eager
        reconstruction, so keep the narrow fast path feature-only.
        """
        return (
            self.model_specific_data.get(
                DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY, False
            )
            and self.hash is not None
            and self.pad_value is not None
            and isinstance(self.feature, CudaIpcTensorTransportProxy)
            and not isinstance(self.precomputed_embeddings, CudaIpcTensorTransportProxy)
            and not any(
                isinstance(value, CudaIpcTensorTransportProxy)
                for value in self.model_specific_data.values()
            )
        )

    def acknowledge_deferred_cuda_ipc_feature(self, consumer_count: int = 1):
        """Release a lazy IPC feature when an embedding-cache hit skips ViT."""
        if isinstance(self.feature, CudaIpcTensorTransportProxy):
            consumer_count = self._resolve_transport_consumer_count(
                self.feature, consumer_count
            )
            self.feature.acknowledge_consumption(consumer_count)

    @staticmethod
    def _resolve_transport_consumer_count(proxy, requested_count: int) -> int:
        """Clamp a group acknowledgement to the proxy's actual consumer set."""
        proxy_count = getattr(
            proxy,
            "total_consumer_count",
            getattr(proxy, "consumer_count", requested_count),
        )
        return min(requested_count, proxy_count)


class MultimodalProcessorOutput(
    msgspec.Struct, kw_only=True, dict=True, array_like=True, weakref=True
):
    """Raw output from multimodal processors before scheduler-side preparation (pad, hash).

    This is the typed replacement for the dict previously returned by
    ``BaseMultimodalProcessor.process_mm_data_async``.  Preprocessed inputs may
    already carry ``pad_value`` and ``hash`` to avoid hashing the same tensor once
    per scheduler TP rank.

    中译：多模态 processor 的「原始输出」（在调度器侧做 pad/hash 之前）。它是一个带类型的
          dataclass，替代了过去 process_mm_data_async 返回的裸 dict。预处理过的输入可能已带
          pad_value/hash，避免在每个 TP rank 上重复 hash 同一张量。
    """

    mm_items: List[MultimodalDataItem]
    input_ids: Optional[List[int]] = None
    padded_input_ids: Optional[List[int]] = None

    # image
    im_token_id: Optional[int] = None
    im_start_id: Optional[int] = None
    im_end_id: Optional[int] = None
    slice_start_id: Optional[int] = None
    slice_end_id: Optional[int] = None

    # video
    video_token_id: Optional[int] = None

    # audio
    audio_token_id: Optional[int] = None
    audio_start_id: Optional[int] = None
    audio_end_id: Optional[int] = None

    # QWen2-VL related
    mrope_positions: Optional[torch.Tensor] = None
    mrope_position_delta: Optional[torch.Tensor] = None

    # Moss-VL related
    vision_position_ids: Optional[torch.Tensor] = None
    media_nums_per_sample: Optional[List[int]] = None
    visible_frame_counts: Optional[torch.Tensor] = None

    # for transformers-compatibility
    token_type_ids: Optional[torch.Tensor] = None

    @staticmethod
    def from_dict(d: dict) -> MultimodalProcessorOutput:
        return MultimodalProcessorOutput(
            mm_items=d["mm_items"],
            input_ids=d.get("input_ids"),
            padded_input_ids=d.get("padded_input_ids"),
            im_token_id=d.get("im_token_id"),
            im_start_id=d.get("im_start_id"),
            im_end_id=d.get("im_end_id"),
            slice_start_id=d.get("slice_start_id"),
            slice_end_id=d.get("slice_end_id"),
            video_token_id=d.get("video_token_id"),
            audio_token_id=d.get("audio_token_id"),
            audio_start_id=d.get("audio_start_id"),
            audio_end_id=d.get("audio_end_id"),
            mrope_positions=d.get("mrope_positions"),
            mrope_position_delta=d.get("mrope_position_delta"),
            vision_position_ids=d.get("vision_position_ids"),
            media_nums_per_sample=d.get("media_nums_per_sample"),
            visible_frame_counts=d.get("visible_frame_counts"),
        )

    @staticmethod
    def build_padded_input_ids(input_ids, mm_items: List[MultimodalDataItem]):
        """pad the input_ids with mm_items if it's not already padded

        中译：用各 mm_item 的 pad_value 把 input_ids 里对应区间 [start, end] 替换为占位 id，
              得到「填充后的 input_ids」。若缺少 pad_value/offsets 信息则返回 None。
        """
        if input_ids is None or not mm_items:
            return None

        for item in mm_items:
            if item.pad_value is None or item.offsets is None:
                return None

        if isinstance(input_ids, torch.Tensor):
            padded_input_ids = input_ids.flatten().tolist()
        else:
            padded_input_ids = list(input_ids)

        for item in mm_items:
            for start, end in item.offsets:
                padded_input_ids[start : end + 1] = [item.pad_value] * (end - start + 1)
        return padded_input_ids


@dataclasses.dataclass
class MultimodalInputs:
    """The multimodal data related inputs.

    中译：挂在 Req 上的「多模态输入」聚合体，汇集该请求的所有 mm_items 以及各种特殊 token id
          （图像/视频/音频的起止标记）和位置编码相关信息（mrope 等）。请求合并时通过 merge() 拼接。
    """

    # items of data
    mm_items: List[MultimodalDataItem]  # 本请求的所有多模态 item 列表
    padded_input_ids: Optional[List[int]] = None  # 已用 pad_value 填充占位后的 input_ids
    image_pad_len: Optional[list] = None  # 每张图占位长度
    num_image_tokens: Optional[int] = None  # 图像 token 总数（encoder-decoder 用）

    # image
    im_token_id: Optional[int] = None
    im_start_id: Optional[int] = None
    im_end_id: Optional[int] = None
    slice_start_id: Optional[int] = None
    slice_end_id: Optional[int] = None

    # video
    video_token_id: Optional[int] = None

    # audio
    audio_token_id: Optional[int] = None
    audio_start_id: Optional[int] = None
    audio_end_id: Optional[int] = None

    # QWen2-VL related
    mrope_positions: Optional[torch.Tensor] = None
    mrope_position_delta: Optional[torch.Tensor] = None
    mrope_position_delta_repeated_cache: Optional[torch.Tensor] = None

    # Moss-VL related
    vision_position_ids: Optional[torch.Tensor] = None
    media_nums_per_sample: Optional[List[int]] = None
    visible_frame_counts: Optional[torch.Tensor] = None

    def release_features(self):
        """Release feature tensors to free GPU memory.

        中译：编码完成后释放原始特征张量，回收（GPU）内存。
        """
        for item in self.mm_items:
            item.feature = None

    @staticmethod
    def from_processor_output(obj: MultimodalProcessorOutput):
        # 中译：从 processor 的原始输出构造 MultimodalInputs。过程包含：
        #       过滤无效 item -> 还原 CUDA IPC 代理张量 -> （可选）把特征临时搬到 GPU 加速 hash
        #       -> 计算每个 item 的 pad_value -> 把特征搬回 CPU -> 拷贝各可选字段。
        mm_items = obj.mm_items
        assert isinstance(mm_items, list)
        mm_items = [item for item in mm_items if item.is_valid()]

        # try reconstructing from cuda-ipc
        reconstruct_device = None
        for mm_item in mm_items:
            if (
                mm_item.has_cuda_ipc_proxy()
                and not mm_item.can_defer_cuda_ipc_feature_reconstruction()
            ):
                if reconstruct_device is None:
                    reconstruct_device = torch.cuda.current_device()
                mm_item.reconstruct(reconstruct_device)

        if envs.SGLANG_MM_BUFFER_SIZE_MB.get() > 0:
            # Multi-modal feature hashing optimization:
            # When SGLANG_MM_BUFFER_SIZE_MB > 0, we temporarily move feature tensors to GPU
            # for faster hash computation, while avoiding OOM issues.
            from sglang.srt.managers.mm_utils import (
                init_feature_buffer,
                is_feature_buffer_initialized,
                reset_buffer_offset,
                try_add_to_buffer,
            )

            device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            if not is_feature_buffer_initialized():
                init_feature_buffer(device)
            reset_buffer_offset()
            for item in mm_items:
                if item.feature is not None:
                    if isinstance(item.feature, torch.Tensor):
                        item.feature = try_add_to_buffer(item.feature)

        for item in mm_items:
            item.set_pad_value()

        if envs.SGLANG_MM_BUFFER_SIZE_MB.get() > 0:
            for item in mm_items:
                if item.feature is not None:
                    item.feature = item.feature.to("cpu", non_blocking=True)

        mm_inputs = MultimodalInputs(
            mm_items=mm_items,
            padded_input_ids=obj.padded_input_ids,
        )
        optional_args = [
            "mrope_positions",
            "mrope_position_delta",
            "im_token_id",
            "im_start_id",
            "im_end_id",
            "video_token_id",
            "slice_start_id",
            "slice_end_id",
            "audio_start_id",
            "audio_end_id",
            "audio_token_id",
            "vision_position_ids",
            "media_nums_per_sample",
            "visible_frame_counts",
        ]
        for arg in optional_args:
            val = getattr(obj, arg, None)
            if val is not None:
                setattr(mm_inputs, arg, val)

        return mm_inputs

    def contains_image_inputs(self) -> bool:
        # 中译：是否包含图像输入。
        return any(item.is_image() for item in self.mm_items)

    def contains_video_inputs(self) -> bool:
        # 中译：是否包含视频输入。
        return any(item.is_video() for item in self.mm_items)

    def contains_audio_inputs(self) -> bool:
        # 中译：是否包含音频输入。
        return any(item.is_audio() for item in self.mm_items)

    def contains_mm_input(self) -> bool:
        # 中译：是否包含任意有效多模态输入。
        return any(True for item in self.mm_items if item.is_valid())

    def compute_mm_token_counts(self) -> Tuple[int, int, int]:
        """Count prompt tokens consumed by each modality (image, audio, video).

        A modality's token count is the total span covered by its items'
        offsets. Returns a (image_tokens, audio_tokens, video_tokens) tuple.
        """
        image_tokens = audio_tokens = video_tokens = 0
        for item in self.mm_items:
            if not item.offsets:
                continue
            num_tokens = sum(end - start + 1 for start, end in item.offsets)
            if item.is_image():
                image_tokens += num_tokens
            elif item.is_audio():
                audio_tokens += num_tokens
            elif item.is_video():
                video_tokens += num_tokens
        return image_tokens, audio_tokens, video_tokens

    def merge(self, other: MultimodalInputs):

        # args needed to be merged
        optional_args = [
            "mm_items",
            "image_pad_len",
        ]
        for arg in optional_args:
            self_arg = getattr(self, arg, None)
            if self_arg is not None:
                setattr(self, arg, self_arg + getattr(other, arg))

        mrope_positions = self.mrope_positions
        if mrope_positions is not None:
            if other.mrope_positions is None:
                self.mrope_positions = mrope_positions
            else:
                self.mrope_positions = torch.cat(
                    [self.mrope_positions, other.mrope_positions], dim=1
                )

        mrope_position_delta = self.mrope_position_delta
        if mrope_position_delta is not None:
            if other.mrope_position_delta is None:
                self.mrope_position_delta = mrope_position_delta
            else:
                self.mrope_position_delta = torch.cat(
                    [self.mrope_position_delta, other.mrope_position_delta], dim=0
                )

        for key, val in other.__dict__.items():
            if "_id" in key:
                # set token_ids
                if getattr(self, key, None) is None:
                    setattr(self, key, getattr(other, key, None))
        # other args would be kept intact


@dataclasses.dataclass(slots=True, kw_only=True)
class ReqLogprob:
    """中译：单个请求的 logprob（对数概率）相关数据容器。
    既保存「输入侧」的 logprob（input_*），也保存「输出侧」逐步生成的 logprob（output_*）。
    其中 top_logprobs 表示每步 top-k 候选的对数概率，token_ids_logprobs 表示用户指定关注的某些 token 的对数概率。
    """

    top_logprobs_num: int  # 需要返回的 top-k 候选数
    token_ids_logprob: Optional[List[int]]  # 用户指定要追踪 logprob 的 token id 列表
    input_token_logprobs_val: Optional[List[float]] = None
    input_token_logprobs_idx: Optional[List[int]] = None
    input_top_logprobs_val: Optional[List[List[float]]] = None
    input_top_logprobs_idx: Optional[List[List[int]]] = None
    # Flat replacements for the rows above (see
    # build_flat_input_top_logprobs_arrays); when set, the nested rows are
    # emptied and the arrays ship instead.
    input_top_logprobs_val_flat: Optional[np.ndarray] = None
    input_top_logprobs_idx_flat: Optional[np.ndarray] = None
    input_top_logprobs_flat_null_prefix: Optional[int] = None
    input_token_ids_logprobs_val: Optional[List[List[float]]] = None
    input_token_ids_logprobs_idx: Optional[List[List[int]]] = None
    output_token_logprobs_val: Optional[list] = None
    output_token_logprobs_idx: Optional[list] = None
    output_top_logprobs_val: Optional[list] = None
    output_top_logprobs_idx: Optional[list] = None
    # Can contain either lists or GPU tensors (delayed copy optimization for prefill-only scoring)
    output_token_ids_logprobs_val: Optional[List[Union[List[float], torch.Tensor]]] = (
        None
    )
    output_token_ids_logprobs_idx: Optional[list] = None


@dataclasses.dataclass(slots=True, kw_only=True)
class ReqKvInfo:
    # Device KV a request holds outside the prefix cache. Always present on the Req;
    # whether any KV is held is `is_held` (a row is registered).
    req_pool_idx: Optional[int] = None  # req_to_token row, the register for the slots

    # The request's own KV is [cache_protected_len, kv_allocated_len).
    cache_protected_len: int = 0  # tree cache owns [0, here) (matched or inserted)
    kv_committed_len: int = 0  # KV content committed up to here, <= kv_allocated_len
    kv_allocated_len: int = 0

    # SWA slots in [swa_dead_lo(page_size), swa_evicted_seqlen) are already freed.
    swa_evict_floor: int = 0  # [0, here) never window-evicted (prefill-aware SWA)
    swa_evicted_seqlen: int = 0  # SWA eviction cursor

    def swa_dead_lo(self, page_size: int) -> int:
        # Lowest SWA position this request may free itself: above the tree-owned
        # prefix and above the eviction shield, page-aligned upward.
        lo = max(self.cache_protected_len, self.swa_evict_floor)
        if page_size > 1 and lo > self.cache_protected_len:
            lo = ceil_align(lo, page_size)
        return lo

    @property
    def is_held(self) -> bool:
        return self.req_pool_idx is not None

    @property
    def is_released(self) -> bool:
        return self.kv_allocated_len == 0 and self.swa_evicted_seqlen == 0

    def mark_released(self) -> None:
        self.kv_allocated_len = 0
        self.swa_evicted_seqlen = 0


class Req(ReqDllmMixin):
    """The input and output status of a request.

    中译：Req 是单个推理请求的「全状态对象」，贯穿其整个生命周期。它聚合了：
      - 输入/输出 token（origin_input_ids、output_ids、full_untruncated_fill_ids 等）；
      - 采样参数、logprob 设置、约束解码（grammar）状态；
      - 前缀缓存/KV 缓存相关的各种 offset 与长度（prefix_indices、kv_committed_len 等）；
      - 完成判定状态机（finished_reason / to_finish / finished_len）；
      - 增量解码 offset（surr_offset / read_offset）；
      - 多模态输入、投机解码统计、PD 分离（disaggregation）传输状态等。
    调度器据此决定何时 prefill、何时 decode、何时回收内存、何时判定请求结束。
    """

    def __init__(
        self,
        rid: str,
        origin_input_text: str,
        origin_input_ids: array[int],
        sampling_params: SamplingParams,
        return_logprob: bool = False,
        top_logprobs_num: int = 0,
        dllm_config: Optional[DllmConfig] = None,
        token_ids_logprob: List[int] = None,
        return_sampling_mask: bool = False,
        return_flat_raw_top_logprobs: bool = False,
        stream: bool = False,
        origin_input_ids_unpadded: Optional[array[int]] = None,
        lora_id: Optional[str] = None,
        input_embeds: Optional[List[List[float]]] = None,
        positional_embed_overrides: Optional[PositionalEmbeds] = None,
        token_type_ids: List[int] = None,
        session: Optional[Session] = None,
        custom_logit_processor: Optional[str] = None,
        require_reasoning: bool = False,
        return_hidden_states: ReturnHiddenStatesMode = False,
        return_routed_experts: bool = False,
        routed_experts_start_len: int = 0,
        return_indexer_topk: bool = False,
        eos_token_ids: Optional[Set[int]] = None,
        bootstrap_host: Optional[str] = None,
        bootstrap_port: Optional[int] = None,
        bootstrap_room: Optional[int] = None,
        disagg_mode: Optional[DisaggregationMode] = None,
        routed_dp_rank: Optional[int] = None,
        disagg_prefill_dp_rank: Optional[int] = None,
        vocab_size: Optional[int] = None,
        priority: Optional[int] = None,
        metrics_collector: Optional[SchedulerMetricsCollector] = None,
        extra_key: Optional[str] = None,
        routing_key: Optional[str] = None,
        dimensions: Optional[int] = None,
        http_worker_ipc: Optional[str] = None,
        time_stats: Optional[
            Union[APIServerReqTimeStats, DPControllerReqTimeStats]
        ] = None,
        return_pooled_hidden_states: bool = False,
        multi_item_delimiter_indices: Optional[List[int]] = None,
        session_id: Optional[str] = None,
        cache_salt: Optional[str] = None,
    ):
        # Input and output info
        # 中译：输入与输出信息。
        self.rid = rid  # 请求唯一 id（request id）
        self.origin_input_ids = origin_input_ids  # 原始输入 token（可能含图像 padding 占位）
        self.origin_input_ids_unpadded = (
            origin_input_ids_unpadded
            if origin_input_ids_unpadded
            else self.origin_input_ids
        )  # Before image padding
        # 中译：图像 padding 之前的原始输入 token（用于增量解码等需要「真实文本 token」的场景）。
        # Each decode stage's output ids. Append-only by contract:
        # _refresh_fill_ids infers how many output tokens are already in
        # full_untruncated_fill_ids from lengths alone, so in-place rewrites
        # that preserve length would silently corrupt fill_ids.
        # 中译：每个 decode 阶段产出的输出 token。约定为「只追加」：_refresh_fill_ids 仅靠长度
        #       推断已有多少输出 token 进入 full_untruncated_fill_ids，所以保持长度不变的原地改写
        #       会悄无声息地破坏 fill_ids（务必只 append，勿原地改）。
        self.output_ids = array("q")
        # Full untruncated sequence: origin + output (+ DLLM mask block).
        # 中译：完整未截断序列 = 原始输入 + 输出（+ DLLM 掩码块）。由 _refresh_fill_ids 保持同步；
        # 中译：本次实际要参与 forward 的 fill_ids 长度（chunked prefill 时按块推进，
        # 中译：请求级 KV 缓存内存管理。
        # 中译：cross-encoder 模型用的 token 类型 id（区分句子 A/B 等）。
        # 中译：SWA（滑动窗口注意力）缓存中已被移除（窗口外）的 KV 长度，详见上面英文对两种缓存的说明。
        # Kept in sync by _refresh_fill_ids; admission only updates
        # extend_range, never mutates this array's length.
        self.full_untruncated_fill_ids = array("q")
        self.extend_range: Optional[Range] = None
        self.dllm_initialized: bool = False

        self.session = session
        self.session_id = session_id
        # Used by the session radix cache to reject registration after a close/reopen.
        self.session_generation: Optional[int] = None
        self.input_embeds = input_embeds
        self.positional_embed_overrides = positional_embed_overrides
        self.multi_item_delimiter_indices = multi_item_delimiter_indices

        # For req-level memory management
        self.kv = ReqKvInfo()
        self.retraction_backup: Optional[RetractionBackup] = None

        # for cross-encoder model
        self.token_type_ids = token_type_ids

        # The index of the extend / decode batch
        # 中译：该请求历经的 extend / decode 批次计数（每参与一次相应阶段就 +1，用于各种调度判断）。
        self.extend_batch_idx = 0
        self.decode_batch_idx = 0

        # For multi-http worker
        # 中译：多 HTTP worker 模式下，标识结果应回送到哪个 worker 的 IPC 通道。
        self.http_worker_ipc = http_worker_ipc

        # Require reasoning for the request
        # 中译：该请求是否需要「推理（reasoning/思考）」阶段（如带 <think> 的模型）。
        self.require_reasoning = require_reasoning

        # State indicating whether the reasoning phase has finished (only meaningful when require_reasoning is True)
        # 中译：推理阶段是否已结束（仅在 require_reasoning 为 True 时有意义）；reasoning_tokens 统计推理消耗的 token 数。
        self._is_reasoning_over = False
        self.reasoning_tokens = 0
        self._think_end_matcher: Optional[TokenSequenceMatcher] = None
        self._think_end_match_len = 0

        # Sampling info
        if isinstance(sampling_params.custom_params, dict):
            sampling_params = copy.copy(sampling_params)
            sampling_params.custom_params = sampling_params.custom_params | {
                "__req__": self
            }
        self.sampling_params = sampling_params
        self.custom_logit_processor = custom_logit_processor
        self.return_hidden_states = return_hidden_states
        self.return_hidden_states_mode = get_return_hidden_states_mode(
            return_hidden_states
        )

        # Extra key for caller-defined request classification.
        if lora_id is not None:
            extra_key = (
                extra_key or ""
            ) + lora_id  # lora_id is concatenated to the extra key

        # 中译：内存池索引信息。req_pool_idx 是该请求在 req_to_token 池中的行号；
        self.extra_key = extra_key
        self.cache_salt = cache_salt or None
        self.lora_id = lora_id
        self.routing_key = routing_key

        # Memory pool info
        self.mamba_pool_idx: Optional[torch.Tensor] = None  # shape (1)
        self.mamba_ping_pong_track_buffer: Optional[torch.Tensor] = None  # shape (2)
        self.mamba_next_track_idx: Optional[int] = None  # 0 or 1
        self.mamba_last_track_idx: Optional[int] = None  # 0 or 1
        self.mamba_last_track_seqlen: Optional[int] = (
            None  # seq len of the last cached mamba state
        )
        # the branching point seqlen to track mamba state. If set, given by prefix match,
        # it will be the tracked seqlen in the ping pong buffer for the right prefill pass.
        # 中译：用于跟踪 mamba 状态的「分叉点」seqlen。若由前缀匹配给出，它就是本次 prefill
        #       在 ping-pong 双槽缓冲中所跟踪的 seqlen。
        self.mamba_branching_seqlen: Optional[int] = None
        # Deferred COW: source mamba pool index from radix cache node (copy on forward stream)
        # 中译：延迟 COW（写时复制）：来自 radix 缓存节点的源 mamba 池索引（在 forward 流上复制）。
        self.mamba_cow_src_index: Optional[torch.Tensor] = None
        # Deferred clear: newly allocated mamba slot needs zeroing on forward stream
        # 中译：延迟清零：新分配的 mamba 槽需在 forward 流上置零。
        self.mamba_needs_clear: bool = False
        # Lazy extra buffer: skip radix cache insert when prealloc failed at
        # boundary — the forward overwrites the only slot, corrupting the state.
        # 中译：懒式额外缓冲：当在边界处预分配失败时跳过 radix 缓存插入——否则 forward 会覆写唯一的槽、破坏状态。
        self.mamba_lazy_is_insert: bool = True

        # Check finish
        # 中译：完成判定相关状态。
        self.tokenizer = None
        # 中译：完成原因；为 None 表示尚未结束（finished() 即判断它是否非 None）。
        self.finished_reason: Optional[BaseFinishReason] = None
        # finished position (in output_ids), used when checking stop conditions with speculative decoding
        # 中译：结束位置（在 output_ids 中的长度，含停止 token）。投机解码一步接受多个 token 时，
        #       需要精确知道在第几个 token 处触发了停止，以便截断多余的输出。
        self.finished_len = None
        # Whether this request has finished output
        # 中译：该请求是否已完成输出（供输出处理流程标记）。
        self.finished_output = None
        # If we want to abort the request in the middle of the event loop,
        # set to_finish instead of directly setting finished_reason.
        # Note: We should never set finished_reason in the middle, the req will get filtered and never respond
        # 中译：若想在事件循环「中途」中止请求，应设置 to_finish 而非直接设 finished_reason。
        #       因为一旦中途设了 finished_reason，请求会被 filter_batch 过滤掉、再也无法回复客户端；
        #       to_finish 会在 update_finish_state 的安全时机被正式转为 finished_reason。
        self.to_finish: Optional[BaseFinishReason] = None
        self.stream = stream  # 是否流式输出
        self.eos_token_ids = eos_token_ids  # 结束 token id 集合
        self.vocab_size = vocab_size  # 词表大小（用于越界/NaN 检测）
        self.priority = priority  # 请求优先级（调度排序用）

        # For incremental decoding
        # ----- | --------- read_ids -------|
        # ----- |   surr_ids  |
        # xxxxx | xxxxxxxxxxx | xxxxxxxxxxx |
        # ----- ^ ----------- ^ ----------- ^
        # ----- 1 ----------- 2 ----------- 3
        # 1: surr_offset
        # 2: read_offset
        # 3: last token
        # 中译：增量解码用的两个 offset（与 detokenizer 端的同名概念一致）：
        #       surr_offset（位置1）= 环绕上下文起点，多回看几个 token 以保证跨 token 字符拼接正确；
        #       read_offset（位置2）= 已读取/已发送的边界；位置3 = 最后一个 token。
        #       [surr, read) 是上下文，[surr, 末尾) 是本次要解码的全部，两者解码相减得新增文本。
        self.surr_offset = None  # Surrounding offset to defeat the cleanup algorithm
        self.read_offset = None
        self.decoded_text = ""  # 已解码出的文本（用于停止字符串匹配等）

        # For multimodal inputs
        # 中译：该请求的多模态输入聚合体（无多模态时为 None）。
        self.multimodal_inputs: Optional[MultimodalInputs] = None
        # Pre-computed multimodal prompt token counts; populated on the prefill
        # node and transferred to decode via the metadata buffer in disagg (PD) mode.
        self.mm_image_tokens: int = 0
        self.mm_audio_tokens: int = 0
        self.mm_video_tokens: int = 0

        # Prefix info
        # 中译：前缀缓存（prefix cache）相关信息——决定 prefill 时哪些 token 可复用已有 KV。
        # The indices to kv cache for the shared prefix.
        # 中译：命中的共享前缀在 KV 缓存中的索引（device 上）；其长度 = 已缓存可复用的前缀 token 数。
        self.prefix_indices: torch.Tensor = torch.empty((0,), dtype=torch.int64)
        # 中译：本次 prefill 需要真正计算的 token 数（= 输入长度 - 已命中前缀长度）。
        # 中译：在当前 extend 批次内、logprob 计算起点的「相对位置」（详见 set_extend_input_len）。
        # TODO(ispobock): rename to last_device_node
        # 中译：前缀匹配得到的 radix 树节点引用：last_node（最后命中的 device 节点）、
        #       last_host_node（host 侧）、best_match_node（最佳匹配节点），用于锁定/释放缓存。
        self.last_node: Any = None
        self.last_host_node: Any = None
        self.best_match_node: Any = None
        # Per-component host hit lengths split off from host_hit_length:
        # 中译：按缓存层拆分的 host（CPU）命中长度——普通/SWA/Mamba 各自的 host 命中 token 数，
        #       决定是否需要从 L2（host）做 H2D 回载（needs_host_load_back）。
        self.host_hit_length = 0
        self.swa_host_hit_length = 0
        self.mamba_host_hit_length = 0
        # Total cached prefix length (on-device prefix_indices + host_hit_length),
        # capped at the max allowed prefix. Set during prefix matching at schedule
        # time and used to estimate uncached tokens / sort by longest prefix for
        # load reporting.
        # 中译：命中的前缀总长度（device 上的 prefix_indices + host 命中），上限为最大允许前缀。
        #       在调度期前缀匹配时设置，用于估算「未缓存 token 数」、按最长前缀排序做负载上报。
        self.num_matched_prefix_tokens = 0
        # Tokens loaded from storage backend (L3) during prefetch for this request
        # 中译：本请求在预取阶段从 L3 存储后端载入的 token 数（HiCache 分层缓存的最底层）。
        self.storage_hit_length = 0
        # Storage prefetch retry state while queued
        # (see Scheduler._retry_missed_storage_prefetches).
        self.storage_prefetch_retry_pending = False
        self.storage_prefetch_retry_wait_polls = 0
        self.storage_prefetch_retry_attempts = 0
        # The node to lock until for swa radix tree lock ref
        # 中译：SWA radix 树需要锁定到的节点 uuid（防止该前缀在使用期间被淘汰）。
        self.swa_uuid_for_lock: Optional[int] = None
        # Whether the prefill-time SWA tree lock has been released early
        # 中译：prefill 期加的 SWA 树锁是否已提前释放（decode 越过滑动窗口后即可释放）。
        self.swa_prefix_lock_released: bool = False
        # 中译：已插入 tree cache 的前缀长度（受保护、不会被普通淘汰回收的那段）。
        # per-component nodes this req skipped locking (e.g. mamba on the decode
        # hold, already COW'd), so their dec releases only what it took.
        self.skip_lock_node_ids: dict = {}

        # Whether or not if it is chunked. It increments whenever
        # it is chunked, and decrement whenever chunked request is
        # processed.
        # 中译：该请求处于「分块 prefill」中间块的计数。每被切一块 +1、处理完一块 -1，
        #       >0 表示还有未跑完的中间块（不能立即判定整请求结束）。
        self.inflight_middle_chunks = 0

        # For retraction
        # 中译：抢占/回撤（retraction）相关。is_retracted：当前是否处于被回撤状态；
        #       retracted_stain：是否「曾经」被回撤过（一旦为 True 永久保留，影响 cached_tokens 统计）。
        self.is_retracted = False
        # Indicates if the req has ever been retracted.
        # 中译：标识该请求是否「曾经」被回撤过（一旦为 True 便永久保留）。
        self.retracted_stain = False

        self.weight_version_events: List[WeightVersionEvent] = []

        # Incremental streamining
        # 中译：增量流式发送用的各种 offset，记录「已经发送到哪里」，下次只发新增部分。
        self.send_token_offset: int = 0  # 已发送的（解码后）文本 token 偏移
        self.send_decode_id_offset: int = 0  # 已发送的 decode token id 偏移
        # TODO (Byron): send_output_token_logprobs_offset and send_decode_id_offset can be different in disaggregation mode
        # because the decode server does not have the first output token logprobs
        # 中译：已发送的「输出 token logprob」偏移。在 PD 分离模式下它可能与 send_decode_id_offset
        #       不同步，因为 decode 实例没有第一个输出 token 的 logprob。
        self.send_output_token_logprobs_offset: int = 0
        self.send_output_sampling_mask_offset: int = 0

        # Logprobs (arguments)
        # 中译：logprob 入参与返回值容器。return_logprob 决定是否计算并返回对数概率。
        self.return_logprob = return_logprob
        # Start index to compute logprob from.
        # 中译：从序列的哪个绝对位置开始计算 logprob（-1 表示只返回新生成 token 的 logprob）。
        self.logprob_start_len = 0
        self.logprob = ReqLogprob(
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
        )
        self.temp_scaled_logprobs = False
        self.top_p_normalized_logprobs = False
        self.return_sampling_mask = return_sampling_mask
        self.return_flat_raw_top_logprobs = return_flat_raw_top_logprobs

        # Logprobs (return values)
        # 中译：logprob 返回值的中间暂存区。
        # True means the input logprob has been already sent to detokenizer.
        # 中译：为 True 表示「输入侧 logprob」已发送给 detokenizer（只需发一次）。
        self.input_logprob_sent: bool = False
        # Temporary holder to store input_token_logprobs.
        # 中译：临时存放输入 token logprob 的容器（计算后、发送前的暂存）。
        self.input_token_logprobs: Optional[List[Tuple[int]]] = None
        # 中译：以下 temp_* 字段临时存放输入侧 top-k logprob 的值/索引、以及用户指定 token 的 logprob 值/索引。
        self.temp_input_top_logprobs_val: Optional[List[torch.Tensor]] = None
        self.temp_input_top_logprobs_idx: Optional[List[int]] = None
        self.temp_input_token_ids_logprobs_val: Optional[List[float]] = None
        self.temp_input_token_ids_logprobs_idx: Optional[List[int]] = None

        # 中译：return_logprob 为 True 时，预先创建输出侧 logprob 的容器，decode 时逐步追加。
        if return_logprob:
            # shape: (bs, 1)
            # 中译：每步被选中 token 的 logprob 值/索引。
            self.logprob.output_token_logprobs_val = []
            self.logprob.output_token_logprobs_idx = []
            # shape: (bs, k)
            # 中译：每步 top-k 候选的 logprob 值/索引。
            self.logprob.output_top_logprobs_val = []
            self.logprob.output_top_logprobs_idx = []
            # Can contain either lists or GPU tensors (delayed copy optimization for prefill-only scoring)
            # 中译：用户指定关注 token 的 logprob 值/索引。值可能是 list 或 GPU 张量
            #       （prefill-only 打分场景的延迟拷贝优化）。
            self.logprob.output_token_ids_logprobs_val = []
            self.logprob.output_token_ids_logprobs_idx = []
        # 中译：隐藏状态返回值。hidden_states 用 list；PD + MTP 传输时改用 tensor（hidden_states_tensor）。
        if return_sampling_mask:
            self.output_token_sampling_mask = []
            self.output_token_sampling_logprobs = []
        else:
            self.output_token_sampling_mask = None
            self.output_token_sampling_logprobs = None
        self.hidden_states: List[List[float]] = []
        self.hidden_states_tensor = None  # Note: use tensor instead of list to transfer hidden_states when PD + MTP
        # 中译：输出侧 top-k 概率与其索引（部分投机/MTP 场景需要）。
        self.output_topk_p = None
        self.output_topk_index = None
        self.output_dsa_topk_indices = None

        # capture routed experts
        # 中译：MoE 路由专家捕获。return_routed_experts 决定是否记录每个 token 路由到的专家，
        #       routed_experts_start_len 是开始记录的位置，routed_experts 存结果（cpu 张量，形状 (seqlen, topk)）。
        self.return_routed_experts = return_routed_experts
        self.routed_experts_start_len = routed_experts_start_len
        self.routed_experts: Optional[torch.Tensor] = (
            None  # cpu tensor: shape (seqlen, topk)
        )

        # 中译：indexer top-k 捕获（DSA 等带索引器的注意力）。return_indexer_topk 决定是否返回，
        #       indexer_topk 存结果（cpu 张量，形状 (seqlen, num_indexer_layers, index_topk)）。
        self.return_indexer_topk = return_indexer_topk
        self.indexer_topk: Optional[torch.Tensor] = (
            None  # cpu tensor: shape (seqlen, num_indexer_layers, index_topk)
        )
        # Customized info
        # 中译：自定义透传信息（如自定义 logit processor 产出的额外数据）。
        self.customized_info: Optional[Dict[str, List[Any]]] = None

        # Embedding (return values)
        # 中译：嵌入/奖励模型的输出向量（生成模型为 None）。
        self.embedding = None

        # Constrained decoding
        # 中译：约束解码（grammar，如 JSON Schema / 正则）相关。grammar_key 是语法缓存键；
        #       grammar 是编译好的语法对象或其异步 Future；grammar_wait_ct 记录等待语法编译的轮数。
        self.grammar_key: Optional[Tuple[str, str]] = None
        self.grammar: Optional[Union[BaseGrammarObject, Future[BaseGrammarObject]]] = (
            None
        )
        self.grammar_wait_ct = 0

        # The number of cached tokens that were already cached in the KV cache
        # 中译：本请求命中缓存（无需重算）的 token 数；already_computed 记录已计算到的位置，
        #       两者配合在 chunked prefill 多块之间正确累加 cached_tokens（避免重复计数）。
        self.cached_tokens = 0
        self.already_computed = 0

        # Detailed breakdown of cached tokens by source (for HiCache)
        # 中译：HiCache 分层缓存下，按来源细分的命中 token 数（device/host/L3 storage 三层）。
        self.cached_tokens_device = 0  # Tokens from device cache (GPU)
        self.cached_tokens_host = 0  # Tokens from host cache (CPU memory)
        self.cached_tokens_storage = 0  # Tokens from L3 storage backend
        self._cache_breakdown_computed = (
            False  # Track if breakdown was already computed
        )  # 中译：是否已计算过上面的分层细分（只在首块计算一次）。

        # Per-request count of verification forward passes.
        # 中译：投机解码中，本请求经历的「验证」forward 次数。
        self.spec_verify_ct = 0

        # Per-request count of accepted draft tokens (excludes the bonus token).
        # 中译：本请求被接受的草稿（draft）token 总数（不含每步必得的 bonus token）。
        self.spec_num_correct_drafts = 0

        self.spec_num_block_accept_tokens = 0

        self.spec_num_cap_tokens = 0

        # Acceptance histogram for speculative decoding.
        # List index = number of accepted tokens in a step, List value = count of steps with that many accepted tokens.
        # Example: histogram[0] = 5 means 5 steps with 0 accepted tokens, histogram[3] = 10 means 10 steps with 3 accepted tokens.
        # 中译：投机解码的「接受数」直方图。下标 = 某步接受的 token 数，值 = 出现该接受数的步数。
        #       例：histogram[0]=5 表示有 5 步 0 接受；histogram[3]=10 表示有 10 步各接受 3 个。用于统计接受率分布。
        self.spec_correct_drafts_histogram: List[int] = []

        self.spec_cap_lens_histogram: List[int] = []

        # The number of times this request has been retracted / preempted.
        # 中译：本请求被回撤/抢占的累计次数（reset_for_retract 不重置它，用于统计）。
        self.retraction_count = 0
        self.retraction_mb_id = None  # 回撤发生时所属的微批（microbatch）id（PP 场景）

        # For observability
        # 中译：可观测性（监控/打点）。metrics_collector 上报指标；time_stats 记录请求各阶段时间戳
        #       （入队、首次 forward、完成等）；has_log_time_stats 标记该请求的耗时统计是否已打日志。
        self.metrics_collector = metrics_collector
        if time_stats is not None:
            self.time_stats = SchedulerReqTimeStats.new_from_obj(time_stats)
        else:
            self.time_stats = SchedulerReqTimeStats(disagg_mode=disagg_mode)
        self.time_stats.set_metrics_collector(metrics_collector)
        self.time_stats.set_scheduler_recv_time()
        self.has_log_time_stats: bool = False

        # For disaggregation
        # 中译：PD 分离（prefill/decode 分别部署）相关。bootstrap_host/port/room 用于在 prefill
        #       与 decode 实例之间建立 KV 传输的「握手」连接；disagg_kv_sender 负责实际发送 KV。
        self.bootstrap_host: str = bootstrap_host
        self.bootstrap_port: Optional[int] = bootstrap_port
        self.bootstrap_room: Optional[int] = bootstrap_room
        # Decode-local: the already-emitted boundary token to replay when a
        # retracted request is rebootstrapped. Set in pause_generation(retract)
        # and consumed in the decode transfer commit; never plumbed to prefill.
        self.pd_rebootstrap_forced_output_id: Optional[int] = None
        self.skip_radix_cache_insert = bootstrap_host == FAKE_BOOTSTRAP_HOST
        self.disagg_kv_sender: Optional[BaseKVSender] = None

        # 中译：路由 DP rank：routed_dp_rank 为请求被路由到的 DP rank；
        #       disagg_prefill_dp_rank 为 PD 分离下 prefill 实例所用的 DP rank。
        self.routed_dp_rank: Optional[int] = routed_dp_rank
        self.disagg_prefill_dp_rank: Optional[int] = disagg_prefill_dp_rank

        # the start index of the sent kv cache
        # We want to send it chunk by chunk for chunked prefill.
        # After every chunk forward, we do the following:
        # 中译：PD 分离下已发送 KV 的起始下标。为支持 chunked prefill，KV 按块发送：每跑完一块就
        # kv_send(req.input_ids[req.start_send_idx:req.extend_range.end])
        # start_send_idx = req.extend_range.end
        self.start_send_idx: int = 0
        self.disagg_decode_prefix_len: int = 0

        # For overlap schedule, we delay the kv transfer until `process_batch_result_disagg_prefill` rather than `process_prefill_chunk` in non-overlap
        # This is because kv is not ready in `process_prefill_chunk`.
        # We use `tmp_end_idx` to store the end index of the kv cache to send.
        # 中译：重叠调度下，KV 传输被推迟到 process_batch_result_disagg_prefill（而非非重叠的
        #       process_prefill_chunk），因为此刻 KV 尚未就绪。tmp_end_idx 暂存本次要发送 KV 的结束下标。
        self.tmp_end_idx: int = -1
        # Decode-side cached-prefix length; base of the staging chunk grid
        # (start_send_idx starts here but advances with every send).
        self.disagg_decode_prefix_len: int = 0
        # At-rest device-resident prefix end, snapshotted on the request's
        # first prefill batch; the cached-prefix early-send never goes past it.
        self.early_send_prefix_end: Optional[int] = None
        self.metadata_buffer_index: int = -1
        # Used in overlap sequence to signal that an optimistic request should
        # abort chunking. Set in create_sender, consumed in process_batch_result.
        # 中译：重叠流程中用于标记「乐观请求」应中止分块。在 create_sender 中设置、
        #       在 process_batch_result 中消费。
        self.pending_bootstrap = False
        # Number of optimistic prefill forward passes started. preserved across retracts.
        self.prefill_attempt_count = 0

        # For Matryoshka embeddings
        # 中译：Matryoshka（套娃）嵌入的目标维度——可截取嵌入向量的前 dimensions 维。
        self.dimensions = dimensions

        # Beam search overlay: leader and internal members share one BeamGroup.
        self.beam_group = None

        # Whether to return pooled hidden states (pre-head transformer output)
        # 中译：是否返回池化后的隐藏状态（输出头之前的 transformer 输出）；pooled_hidden_state 存结果。
        self.return_pooled_hidden_states = return_pooled_hidden_states
        self.pooled_hidden_state = None

        # For diffusion LLM
        # 中译：扩散式 LLM（diffusion LLM）相关状态初始化（按 dllm_config 配置）。
        self.init_diffusion_llm(dllm_config)

        # For hisparse
        # 中译：HiSparse（分层稀疏注意力）暂存标记——该请求是否处于 staging（等待换入）状态。
        self.hisparse_staging = False

    @property
    def seqlen(self) -> int:
        """Get the current sequence length of the request.

        中译：当前序列总长度 = 原始输入长度 + 已生成输出长度。
        """
        return len(self.origin_input_ids) + len(self.output_ids)

    @property
    def is_beam_leader(self) -> bool:
        """The user-visible row of a beam group; the group's other rows are
        scheduler-internal and never streamed to the user."""
        return self.beam_group is not None and self.beam_group.leader is self

    @property
    def is_prefill_only(self) -> bool:
        """Check if this request is prefill-only (no token generation needed).

        中译：是否为「仅 prefill」请求（不需要生成 token，如 embedding/打分场景，max_new_tokens==0）。
              注意：开启投机解码时，prefill_only 优化被禁用（故要求 spec_alg 为 None）。
        """
        # NOTE: when spec is enabled, prefill_only optimizations are disabled

        spec_alg = get_spec().speculative_algorithm
        return self.sampling_params.max_new_tokens == 0 and spec_alg is None

    @property
    def output_ids_through_stop(self) -> array[int]:
        """Get the output ids through the stop condition. Stop position is included.

        中译：取「到停止位置为止」的输出 token（含停止 token）。投机解码一步可能多接受几个
              token，但越过停止点的部分不应输出，故按 finished_len 截断。
        """
        if self.finished_len is not None:
            return self.output_ids[: self.finished_len]
        return self.output_ids

    def needs_host_load_back(self) -> bool:
        """Whether any cache layer has a host hit that needs L2 H2D load_back.

        中译：是否有任一缓存层在 host（CPU/L2）命中、因而需要 H2D（host->device）回载 KV。
        """
        return (
            self.host_hit_length > 0
            or self.swa_host_hit_length > 0
            or self.mamba_host_hit_length > 0
        )

        # 中译：返回可提交进缓存的 KV 长度。开启 strip_thinking_cache 且有推理 token 时，
        # 中译：只上报 prompt 前缀，让「思考 + 回答」落入超额分配区、由 release_kv_cache 回收（见 #22373）。
        # 中译：本函数在存在 KV 缓存「超额分配」时调用。超额分配即分配的 KV 多于已提交长度，
    def detach_kv(self) -> ReqKvInfo:
        # Hand the KV record to a new holder; the req keeps a fresh empty one.
        kv, self.kv = self.kv, ReqKvInfo()
        return kv

    def effective_kv_committed_len(self) -> int:
        # Report only the prompt prefix so thinking + answer fall into the
        # overallocated range and are reclaimed by release_kv_cache. #22373.
        if get_serving().strip_thinking_cache and self.reasoning_tokens > 0:
            return min(self.kv.kv_committed_len, len(self.origin_input_ids))
        return self.kv.kv_committed_len

    def update_spec_correct_drafts_histogram(self, num_correct_drafts: int):
        """Record one step accepted draft count (excludes bonus token) into the histogram."""
        if len(self.spec_correct_drafts_histogram) <= num_correct_drafts:
            self.spec_correct_drafts_histogram.extend(
                [0] * (num_correct_drafts - len(self.spec_correct_drafts_histogram) + 1)
            )
        self.spec_correct_drafts_histogram[num_correct_drafts] += 1

    def update_spec_cap_lens_histogram(self, cap_len: int):
        cap_len = int(cap_len)
        if len(self.spec_cap_lens_histogram) <= cap_len:
            self.spec_cap_lens_histogram.extend(
                [0] * (cap_len - len(self.spec_cap_lens_histogram) + 1)
            )
        self.spec_cap_lens_histogram[cap_len] += 1

    def extend_image_inputs(self, image_inputs):
        # 中译：向本请求追加多模态输入；首次直接赋值，否则与已有的 multimodal_inputs 合并。
        if self.multimodal_inputs is None:
            self.multimodal_inputs = image_inputs
        else:
            self.multimodal_inputs.merge(image_inputs)

    def finished(self) -> bool:
        # Whether request reached finished condition
        # 中译：请求是否已达到结束条件——即 finished_reason 是否已被设置。
        return self.finished_reason is not None

    def set_extend_range(self, start: int, end: int) -> None:
        self.extend_range = Range(start, end)

    def get_fill_ids(self) -> array:
        # 中译：取本次要参与 forward 的 fill_ids，即 full_untruncated_fill_ids 的前 fill_len 个。
        return self.full_untruncated_fill_ids[: self.extend_range.end]

    def _refresh_fill_ids(self) -> None:
        """Keep full_untruncated_fill_ids == origin_input_ids + output_ids by
        appending only the new output tokens.

        Falls back to a full rebuild when the in-place append is invalid:
        - aliasing: scheduler_pp_mixin assigns full_untruncated_fill_ids =
          origin_input_ids directly, so extending in place would write output
          tokens into the origin;
        - lengths disagree: fresh req (array still empty), retraction
          (output_ids reset to empty), or set_finish_with_abort (origin
          replaced by a 1-token stub).

        中译：保持 full_untruncated_fill_ids == origin_input_ids + output_ids，
              快路径只「追加」新产生的输出 token（O(新增) 而非 O(全长)）。
              以下情况无法原地追加，退化为整体重建：
              - 别名（aliasing）：scheduler_pp_mixin 曾把 full_untruncated_fill_ids 直接指向
                origin_input_ids，此时原地 extend 会把输出写进原始输入；
              - 长度对不上：全新请求（数组仍空）、回撤（output_ids 被清空）、
                set_finish_with_abort（origin 被替换为 1-token 占位）。
        """
        n_have_output = len(self.full_untruncated_fill_ids) - len(self.origin_input_ids)
        if (
            self.full_untruncated_fill_ids is not self.origin_input_ids
            and 0 <= n_have_output <= len(self.output_ids)
        ):
            self.full_untruncated_fill_ids.extend(self.output_ids[n_have_output:])
        else:
            self.full_untruncated_fill_ids = self.origin_input_ids + self.output_ids

    def init_next_round_input(
        self,
        tree_cache: Optional[BasePrefixCache] = None,
        cow_mamba: Optional[bool] = None,
    ):
        # 中译：为「下一轮 forward」准备输入。这是请求每次进入调度前的关键预处理：
        #       1) 刷新 full_untruncated_fill_ids（DLLM 走专门路径）；
        #       2) 用 tree_cache.match_prefix 做前缀匹配，得到可复用的 prefix_indices 及各命中长度，
        #          并填好 last_node/cache_protected_len 等缓存锁定相关字段；
        #       3) 据此计算本轮需要真正 prefill 的 token 数 set_extend_input_len(input_len - 前缀长度)。
        # 中译：DLLM（扩散式 LLM）走专用的 fill_ids 初始化与阶段判定；普通自回归模型只需刷新 fill_ids。
        if self.is_dllm():
            self._init_fill_ids_for_dllm()
            self.determine_dllm_phase()
        else:
            self._refresh_fill_ids()

        # 中译：本请求的总输入长度（= 原始输入 + 已生成输出，含 DLLM 掩码块）。
        input_len = len(self.full_untruncated_fill_ids)

        # Streaming sessions reuse committed KV from the session slot, so
        # custom logprob_start_len is not supported — override to -1.
        # 中译：流式会话复用会话槽中已提交的 KV，故不支持自定义 logprob_start_len——强制覆盖为 -1
        #       （只返回新生成 token 的 logprob）。
        if (
            self.session is not None
            and self.session.streaming
            and self.return_logprob
            and self.logprob_start_len >= 0
        ):
            logger.warning(
                "logprob_start_len=%d is not supported for streaming sessions "
                "and will be ignored (rid=%s). Only new-token logprobs are returned.",
                self.logprob_start_len,
                self.rid,
            )
            self.logprob_start_len = -1

        # Pass the full array with a raw-token cap (limit) instead of slicing,
        # avoiding an O(context) copy per prefill-batch build.
        # 中译：传整个数组 + 一个 limit 上限，而非先切片，避免每次构建 prefill 批次都做 O(上下文) 拷贝。
        token_ids_to_match = self.full_untruncated_fill_ids
        key_limit: Optional[int] = self._compute_max_prefix_len(input_len)

        # SWA lives in a per-request ring that's not content-stable and is never
        # stored in the radix tree, so a reused prefix carries stale SWA. Cap the
        # match by the trailing sliding window so it gets re-prefilled, rewriting
        # this request's SWA ring. No-op for other layouts.
        if tree_cache is not None:
            reprefill_tail = tree_cache.swa_reprefill_tail_tokens()
            if reprefill_tail:
                capped = max(0, input_len - reprefill_tail)
                key_limit = capped if key_limit is None else min(key_limit, capped)

        # Disable prefix caching when embed overrides are present: same token IDs
        # with different override vectors must not share cached KV values.
        # 中译：存在位置嵌入覆盖（positional_embed_overrides）时禁用前缀缓存——相同 token id 但
        #       覆盖向量不同的请求，绝不能共享同一份缓存 KV，故清空待匹配 key。
        if self.positional_embed_overrides is not None:
            token_ids_to_match = array("q")
            key_limit = None

        # 中译：传入了前缀树缓存时，做前缀匹配并回填可复用的缓存信息。
        if tree_cache is not None:
            # 中译：cow_mamba 未指定时，根据缓存是否支持 mamba 自动决定（是否需对 mamba 状态写时复制）。
            if cow_mamba is None:
                cow_mamba = tree_cache.supports_mamba()
            # unified_kv SWA lives in a per-request ring that is not content-stable
            # and never cached in the radix tree, so a reused prefix carries stale
            # SWA. Cap the match by the trailing sliding window so it is re-prefilled
            # into this request's ring. No-op for other layouts (returns 0).
            reprefill_tail = tree_cache.swa_reprefill_tail_tokens()
            if reprefill_tail:
                capped = max(0, input_len - reprefill_tail)
                key_limit = capped if key_limit is None else min(key_limit, capped)
            match_result = tree_cache.match_prefix(
                MatchPrefixParams(
                    key=RadixKey(
                        token_ids=token_ids_to_match,
                        extra_key=self.extra_key,
                        limit=key_limit,
                        cache_salt=self.cache_salt,
                    ),
                    req=self,
                    cow_mamba=cow_mamba,
                )
            )
            # 中译：调试开关 SGLANG_RADIX_FORCE_MISS 为真时，强制把匹配结果置零（模拟缓存全未命中）。
            if envs.SGLANG_RADIX_FORCE_MISS.get():
            # 中译：把匹配结果解包到请求字段：prefix_indices（可复用前缀的 KV 索引）、
                match_result = zero_match_result(
                    tree_cache, match_result, extra_key=self.extra_key
                )
            (
                self.prefix_indices,
                self.last_node,
                self.last_host_node,
                self.best_match_node,
                self.host_hit_length,
                self.swa_host_hit_length,
                self.mamba_host_hit_length,
                self.mamba_branching_seqlen,
            ) = (
                match_result.device_indices,
                match_result.last_device_node,
                match_result.last_host_node,
                match_result.best_match_node,
                match_result.host_hit_length,
                match_result.swa_host_hit_length,
                match_result.mamba_host_hit_length,
                match_result.mamba_branching_seqlen,
            )
            # 中译：记录受保护的缓存前缀长度；缓存未给出时退而取已命中 prefix_indices 的长度。
            if match_result.cache_protected_len is not None:
                self.kv.cache_protected_len = match_result.cache_protected_len
            else:
                self.kv.cache_protected_len = len(self.prefix_indices)

            # 中译：DLLM 还需根据命中前缀更新掩码块的偏移。
            if self.is_dllm():
                self._update_block_offset_for_dllm()

        # 中译：被回撤的多模态请求，重新 prefill 前需按已生成输出长度延展 mrope（多模态旋转位置）编码。
        if (
            self.is_retracted
            and self.multimodal_inputs is not None
            and self.multimodal_inputs.mrope_positions is not None
        ):
            from sglang.srt.managers.mm_utils import (
                extend_mrope_positions_for_retracted_request,
            )

            self.multimodal_inputs.mrope_positions = (
                extend_mrope_positions_for_retracted_request(
                    self.multimodal_inputs.mrope_positions, len(self.output_ids)
                )
            )

        # 中译：本轮真正需要 prefill 的 token 数 = 总输入长度 - 已命中前缀长度。
    def _compute_max_prefix_len(self, input_len: int) -> int:
        # NOTE: the matched length is at most 1 less than the input length to enable logprob computation
        # 中译：计算允许匹配的最大前缀长度。匹配长度至多比输入少 1（以便至少留 1 个 token 算 logprob）；
        #       若指定了 logprob_start_len 还需进一步受其限制。
        max_prefix_len = input_len - 1
        if self.return_logprob and self.logprob_start_len >= 0:
            max_prefix_len = min(max_prefix_len, self.logprob_start_len)
        return max(max_prefix_len, 0)

    # Based on https://github.com/vllm-project/vllm/blob/7a64d24aad69e4d2548aa0bf528d9fe63428ab01/vllm/transformers_utils/detokenizer.py#L194-L313
    # 中译：本方法实现参考自 vLLM 的 detokenizer（见上方链接）。
    def init_incremental_detokenize(self):
        # 中译：准备增量解码所需的 token 序列与读取偏移。首次调用时初始化 surr_offset/read_offset
        #       并把「环绕上下文 + 输出」拼好；后续只追加新输出。返回 (surr_and_decode_ids,
        #       read_offset-surr_offset) 供 detokenizer 解码出本次新增文本。
        first_iter = self.surr_offset is None or self.read_offset is None

        output_ids = self.output_ids_through_stop

        if first_iter:
            self.read_offset = len(self.origin_input_ids_unpadded)
            self.surr_offset = max(
                self.read_offset - INIT_INCREMENTAL_DETOKENIZATION_OFFSET, 0
            )
            self.surr_and_decode_ids = (
                self.origin_input_ids_unpadded[self.surr_offset :] + output_ids
            )
            self.cur_decode_ids_len = len(output_ids)
        else:
            self.surr_and_decode_ids.extend(output_ids[self.cur_decode_ids_len :])
            self.cur_decode_ids_len = len(output_ids)

        return self.surr_and_decode_ids, self.read_offset - self.surr_offset

    def _stop_match_tail_len(self, new_accepted_len: int) -> int:
        # 中译：计算检测停止字符串/正则时需要回看的「尾部 token 数」。基础取停止串/正则的最大长度+1，
        #       再加上本步新接受 token 数-1，确保投机解码一步多接受时不会漏掉中间触发的停止串。
        max_len_tail_str = max(
            self.sampling_params.stop_str_max_len + 1,
            self.sampling_params.stop_regex_max_len + 1,
        )
        # Cover all newly accepted tokens so an early stop string is not missed
        # when speculative decoding accepts multiple tokens per step.
        return min(
            max_len_tail_str + max(new_accepted_len - 1, 0), len(self.output_ids)
        )

    def tail_str(self, new_accepted_len: int = 1) -> str:
        # Check stop strings and stop regex patterns together
        # 中译：把输出尾部若干 token 解码为字符串，供停止串/正则匹配；无停止串/正则时直接返回空串。
        if (
            len(self.sampling_params.stop_strs) == 0
            and len(self.sampling_params.stop_regex_strs) == 0
        ):
            return ""

        tail_len = self._stop_match_tail_len(new_accepted_len)
        return self.tokenizer.decode(self.output_ids[-tail_len:])

    def check_match_stop_str_prefix(self) -> bool:
        if not self.sampling_params.stop_strs:
            return False

        tail_str = self.tail_str()

        # Early return if tail_str is empty
        # 中译：尾部文本为空则提前返回。
        if not tail_str:
            return False

        for stop_str in self.sampling_params.stop_strs:
            if not stop_str:
                continue
            # Check if stop_str is contained in tail_str (fastest check first)
            # 中译：先做最快的检查——停止串是否整体包含在尾部文本中。
            if stop_str in tail_str:
                return True

            # Check if tail_str suffix matches stop_str prefix
            # Only check if stop_str is not empty, it's for stream output
            # 中译：再检查尾部文本的后缀是否与停止串的前缀相匹配（用于流式输出，判断停止串是否正在形成）。
            min_len = min(len(tail_str), len(stop_str))
            for i in range(1, min_len + 1):
                if tail_str[-i:] == stop_str[:i]:
                    return True

        return False

    def _check_token_based_finish(self, new_accepted_tokens: List[int]) -> bool:
        # 中译：基于「停止 token」的结束判定。逐个检查本步新接受的 token 是否命中
        #       stop_token_ids / eos_token_ids / tokenizer 的 eos 等；命中则设 finished_reason
        #       与 finished_len（命中位置+1），返回 True。ignore_eos 时直接跳过。
        if self.sampling_params.ignore_eos:
            return False

        # Check stop token ids
        # 中译：检查停止 token id。
        matched_eos = False

        for i, token_id in enumerate(new_accepted_tokens):
            if self.sampling_params.stop_token_ids:
                matched_eos |= token_id in self.sampling_params.stop_token_ids
            if self.eos_token_ids:
                matched_eos |= token_id in self.eos_token_ids
            if self.tokenizer is not None:
                matched_eos |= token_id == self.tokenizer.eos_token_id
                if self.tokenizer.additional_stop_token_ids:
                    matched_eos |= token_id in self.tokenizer.additional_stop_token_ids
            if matched_eos:
                self.finished_reason = FINISH_MATCHED_TOKEN(matched=token_id)
                matched_pos = len(self.output_ids) - len(new_accepted_tokens) + i
                self.finished_len = matched_pos + 1
                return True

        return False

    def _locate_str_stop_finished_len(
        self,
        new_accepted_len: int,
        *,
        stop_str: Optional[str] = None,
        stop_regex: Optional[str] = None,
    ) -> int:
        """Map a matched stop string/regex to output_ids length (stop included).

        中译：把「命中的停止串/正则」精确映射回 output_ids 的长度（含停止串本身），
              以便正确截断输出。投机解码一步可能接受多个 token，需在尾窗口内逐 token 试探定位。
        """

        def matched(text: str) -> bool:
            if stop_str is not None:
                return stop_str in text
            return re.search(stop_regex, text) is not None

        tail_len = self._stop_match_tail_len(new_accepted_len)
        start = len(self.output_ids) - tail_len
        token_window = self.output_ids[start:]

        # Old prefixes were checked in the previous step.
        # 中译：较早的前缀在上一步已检查过，此处仅需从本步新接受 token 起逐个试探。
        for token_count in range(
            max(1, len(token_window) - new_accepted_len + 1), len(token_window)
        ):
            if matched(self.tokenizer.decode(token_window[:token_count])):
                return start + token_count

        # The full tail window is already known to match by the caller.
        # 中译：调用方已知整个尾部窗口能匹配，故兼底返回全部输出长度。
        return len(self.output_ids)

    def _check_str_based_finish(self, new_accepted_len: int = 1):
        # 中译：基于「停止字符串/停止正则」的结束判定。解码尾部文本后逐一匹配停止串与正则，
        #       命中则设置对应 finished_reason，并用 _locate_str_stop_finished_len 定位 finished_len。
        if (
            len(self.sampling_params.stop_strs) > 0
            or len(self.sampling_params.stop_regex_strs) > 0
        ):
            tail_str = self.tail_str(new_accepted_len)

            # Check stop strings
            # 中译：检查停止字符串。
            if len(self.sampling_params.stop_strs) > 0:
                for stop_str in self.sampling_params.stop_strs:
                    stop_str_in_tail = stop_str in tail_str
                    if stop_str_in_tail or stop_str in self.decoded_text:
                        self.finished_reason = FINISH_MATCHED_STR(matched=stop_str)
                        if stop_str_in_tail:
                            self.finished_len = self._locate_str_stop_finished_len(
                                new_accepted_len, stop_str=stop_str
                            )
                        return True

            # Check stop regex
            # 中译：检查停止正则。
            if len(self.sampling_params.stop_regex_strs) > 0:
                for stop_regex_str in self.sampling_params.stop_regex_strs:
                    # Seatbelt, not validation: patterns are checked at ingress
                    # (Python's `normalize`, or the rust server's stricter
                    # `stop_regex_bound`). This runs per decode step on the hot
                    # path, so an `re.error` escaping here would take the whole
                    # scheduler down over one malformed request. Fail that request
                    # instead.
                    try:
                        matched = re.search(stop_regex_str, tail_str)
                    except (re.error, RecursionError) as e:
                        logger.warning(
                            "req %s: invalid stop_regex %r (%s); aborting the request",
                            self.rid,
                            stop_regex_str,
                            e,
                        )
                        self.finished_reason = FINISH_ABORT(
                            f"invalid stop_regex {stop_regex_str!r}: {e}",
                            HTTPStatus.BAD_REQUEST,
                            "BadRequestError",
                        )
                        break
                    if matched:
                        self.finished_reason = FINISHED_MATCHED_REGEX(
                            matched=stop_regex_str
                        )
                        self.finished_len = self._locate_str_stop_finished_len(
                            new_accepted_len, stop_regex=stop_regex_str
                        )
                        return True

        return False

    def _check_vocab_boundary_finish(self, new_accepted_tokens: List[int] = None):
        # 中译：词表越界/NaN 防护。若采样出的 token id 超出 [0, vocab_size)（通常意味着出现 NaN），
        #       用一个合法停止 token 覆盖它并强制结束请求，避免崩溃或产生非法输出。
        for i, token_id in enumerate(new_accepted_tokens):
            if token_id < 0 or (
                self.vocab_size is not None and token_id >= self.vocab_size
            ):
                offset = len(self.output_ids) - len(new_accepted_tokens) + i
                if self.sampling_params.stop_token_ids:
                    self.output_ids[offset] = next(
                        iter(self.sampling_params.stop_token_ids)
                    )
                if self.eos_token_ids:
                    self.output_ids[offset] = next(iter(self.eos_token_ids))
                self.finished_reason = FINISH_MATCHED_STR(matched="NaN happened")
                self.finished_len = offset + 1
                return True

        return False

    def _cap_finished_len_at_max_new_tokens(self) -> None:
        """Demote a stop matched beyond the length budget to a length finish.

        Speculative decoding can accept a run that both crosses
        ``max_new_tokens`` and contains a stop; a stop located past the cap
        must not extend the emitted output beyond the cap.
        """
        max_new_tokens = self.sampling_params.max_new_tokens
        if self.finished_len is not None and self.finished_len > max_new_tokens:
            self.finished_reason = FINISH_LENGTH(length=max_new_tokens)
            self.finished_len = max_new_tokens

    def update_finish_state(self, new_accepted_len: int = 1):
        # 中译：请求结束状态机的总入口，每生成一步后调用，按优先级依次检查各种结束条件：
        #       已结束 -> 待中止(to_finish) -> 达到最大长度 -> grammar 终止 -> 停止 token
        #       -> 词表越界/NaN -> 停止串/正则。任一命中即设置 finished_reason 并返回。
        if self.finished():
            return

        if self.to_finish:
            # 中译：之前请求被标记为待中止（如 OOM、客户端取消），在此安全时机正式转为结束。
            self.finished_reason = self.to_finish
            self.to_finish = None
            return

        new_accepted_tokens = self.output_ids[-new_accepted_len:]

        # Sanitize out-of-range / NaN token ids before any decode.
        if self._check_vocab_boundary_finish(new_accepted_tokens):
            self._cap_finished_len_at_max_new_tokens()
            return

        # Stop string beats EOS/stop-token matched in the same step (speculative
        # decoding can accept >1 token): token-based would trim only the last
        # token and leak the stop string.
        if self._check_str_based_finish(new_accepted_len):
            self._cap_finished_len_at_max_new_tokens()
            return

        # Stop token/EOS beats the length cap for the same reason: a spec accept
        # run can cross max_new_tokens in the very step the EOS lands, and a
        # length-first finish would keep the over-accepted tokens after the EOS
        # (up to the cap) in the emitted output.
        if self._check_token_based_finish(new_accepted_tokens):
            self._cap_finished_len_at_max_new_tokens()
            return

        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self.finished_reason = FINISH_LENGTH(
                length=self.sampling_params.max_new_tokens
            )
            self.finished_len = self.sampling_params.max_new_tokens
            return

        if self.grammar is not None and self.grammar.is_terminated():
            self.finished_reason = FINISH_MATCHED_TOKEN(matched=self.output_ids[-1])
            return

    def reset_for_retract(self):
        # 中译：请求被回撤（抢占）时，清空其所有「本轮调度态」字段，使其能像新请求一样重新被 prefill。
        #       注意：保留 retraction_count（累计回撤次数）与 retracted_stain（曾被回撤标记）不重置。
        # Increment retraction count before resetting other state. We should not reset this
        # since we are tracking the total number of retractions for each request.
        # 中译：在重置其他状态前先累加回撤次数；该计数不应被重置，因为它记录每个请求的累计回撤次数。
        self.retraction_count += 1

        self.prefix_indices = torch.empty((0,), dtype=torch.int64)
        self.routed_experts = None
        self.indexer_topk = None
        self.last_node = None
        self.kv.cache_protected_len = 0
        self.num_matched_prefix_tokens = 0
        self.swa_uuid_for_lock = None
        self.swa_prefix_lock_released = False
        self.skip_lock_node_ids = {}
        self.extend_range = None
        self.dllm_initialized = False
        self.is_retracted = True
        self.retracted_stain = True
        self.input_token_logprobs = None
        self.temp_input_top_logprobs_val = None
        self.temp_input_top_logprobs_idx = None
        self.temp_input_token_ids_logprobs_val = None
        self.temp_input_token_ids_logprobs_idx = None
        self.inflight_middle_chunks = 0
        self.mamba_pool_idx = None
        self.mamba_ping_pong_track_buffer = None
        self.mamba_next_track_idx = None
        self.mamba_last_track_idx = None
        self.mamba_last_track_seqlen = None
        self.mamba_branching_seqlen = None
        self.mamba_cow_src_index = None
        self.mamba_needs_clear = False
        self.already_computed = 0
        assert not self.kv.is_held, "expect it is already released"
        self.kv.kv_committed_len = 0
        self.extend_batch_idx = 0
        self.decode_batch_idx = 0

        # When using input_embeds, we cannot easily mix the original input embeddings
        # with the newly generated output token IDs during re-prefill of retracted request.
        # output_ids will have no use, but will lead to wrong size cache indexes.
        # Therefore, we discard the generated output_ids and restart prefill and generation
        # to ensure shape consistency in KV cache.
        # 中译：使用 input_embeds 时，回撤请求重新 prefill 过程中难以将原始输入嵌入与新生成的
        #       输出 token id 混合。output_ids 不仅无用，还会导致缓存索引尺寸错误，故丢弃已生成的
        #       output_ids，重新开始 prefill 与生成，以保证 KV 缓存的形状一致。
        if self.input_embeds is not None:
            self.output_ids = array("q")
            self.weight_version_events = truncate_weight_version_events(
                self.weight_version_events, num_kept_tokens=self.send_token_offset
            )

    def _mamba_pool_needing_backup(self, req_to_token_pool, allocator):
        if allocator.get_kvcache().cpu_copy_carries_mamba:
            return None
        if not isinstance(req_to_token_pool, HybridReqToTokenPool):
            return None
        return req_to_token_pool.mamba_pool

    def offload_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):
        # 中译：把本请求的 KV 缓存（及 mamba 状态）从 device 拷贝到 CPU 暂存（PD 分离 decode 端回撤时用）。
        token_indices = req_to_token_pool.req_to_token[
            self.kv.req_pool_idx, : self.seqlen - 1
        ]
        # Copies over both the kv cache and mamba state if available
        # 中译：若存在，同时拷贝 KV 缓存与 mamba 状态。
        # 中译：与 offload_kv_cache 相反，把 CPU 暂存的 KV（及 mamba 状态）回载到 device，并释放暂存。
        mamba_pool = self._mamba_pool_needing_backup(
            req_to_token_pool, token_to_kv_pool_allocator
        )
        self.retraction_backup = RetractionBackup(
            cpu_tensors=token_to_kv_pool_allocator.get_cpu_copy(
                token_indices, mamba_indices=self.mamba_pool_idx
            ),
            mamba_cpu=(
                mamba_pool.get_cpu_copy(self.mamba_pool_idx.unsqueeze(0))
                if mamba_pool is not None and self.mamba_pool_idx is not None
                else None
            ),
        )

    def load_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):
        assert self.retraction_backup is not None
        token_indices = req_to_token_pool.req_to_token[
            self.kv.req_pool_idx, : self.seqlen - 1
        ]
        # Loads both the kv cache and mamba state if exists
        # 中译：若存在，同时回载 KV 缓存与 mamba 状态。
        mamba_cpu = self.retraction_backup.mamba_cpu
        if mamba_cpu is not None and self.mamba_pool_idx is not None:
            req_to_token_pool.mamba_pool.load_cpu_copy(
                mamba_cpu, self.mamba_pool_idx.unsqueeze(0)
            )
        token_to_kv_pool_allocator.load_cpu_copy(
            self.retraction_backup.cpu_tensors,
            token_indices,
            mamba_indices=self.mamba_pool_idx,
        )
        self.retraction_backup = None

    def build_rebootstrap_payload(self) -> dict:
        """Build the prefill ``/generate`` payload that asks the original prefill
        worker to recompute this request's prefix KV under the current weights
        (PD true-retraction rebootstrap).

        ``input_ids`` are coerced to plain ``int`` so the payload is always
        JSON-serializable even when ``origin_input_ids``/``output_ids`` hold
        numpy scalars. The sampling-param allow-list forces ``max_new_tokens=1``
        and drops stop/grammar/min_new_tokens so the recompute only re-derives
        the prefix KV and samples a single handoff token. The already-emitted
        boundary token is replayed on the *decode* side (the transfer commit
        overrides the sampled handoff with it), so it is intentionally not sent
        to the prefill here.
        """
        # TODO: multi-modal requests are not supported here. The payload only
        # carries token ``input_ids`` and drops any image/audio/video inputs, so
        # the rebootstrap recompute would not reproduce the original prefix KV
        # for multi-modal requests. Add multi-modal support before enabling it.
        sp = self.sampling_params
        return {
            "input_ids": [int(x) for x in self.origin_input_ids]
            + [int(x) for x in self.output_ids],
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": sp.temperature,
                "top_p": sp.top_p,
                "top_k": sp.top_k,
                "min_p": sp.min_p,
                "frequency_penalty": sp.frequency_penalty,
                "presence_penalty": sp.presence_penalty,
                "repetition_penalty": sp.repetition_penalty,
                "ignore_eos": sp.ignore_eos,
                "skip_special_tokens": sp.skip_special_tokens,
                "spaces_between_special_tokens": sp.spaces_between_special_tokens,
                "no_stop_trim": sp.no_stop_trim,
            },
            "return_logprob": False,
            "stream": False,
            "rid": self.rid,
            "bootstrap_host": self.bootstrap_host,
            "bootstrap_port": self.bootstrap_port,
            "bootstrap_room": self.bootstrap_room,
            "priority": self.priority,
            "extra_key": self.extra_key,
            "cache_salt": self.cache_salt,
            "routing_key": self.routing_key,
            "disagg_prefill_dp_rank": self.disagg_prefill_dp_rank,
        }

    def log_time_stats(self):
        # 中译：打印本请求的耗时统计（输入/缓存/输出长度及各阶段时长）。用 has_log_time_stats
        #       做幂等保护——overlap 调度会提前调度一个 decode 批，导致此方法被调用两次。
        # If overlap schedule, we schedule one decode batch ahead so this gets called twice.
        # 中译：overlap 调度下会提前调度一个 decode 批次，导致本方法被调用两次，故需幂等保护。
        if self.has_log_time_stats:
            return

        bootstrap_info = (
            f", bootstrap_room={self.bootstrap_room}"
            if self.bootstrap_room is not None
            else ""
        )
        prefix = (
            f"ReqTimeStats("
            f"rid={self.rid}{bootstrap_info}, "
            f"input_len={len(self.origin_input_ids)}, "
            f"cached_input_len={self.cached_tokens}, "
            f"output_len={len(self.output_ids)}, "
            f"attempts={self.prefill_attempt_count}, "
            f"type={self.time_stats.disagg_mode_str()})"
        )
        logger.info(f"{prefix}: {self.time_stats.convert_to_duration()}")
        self.has_log_time_stats = True

        # 中译：设置本轮 extend（prefill）要处理的 token 数，并据此换算 logprob 在本批次内的相对起点。
        # 中译：关键变量含义：
        # 中译：把请求标记为「以中止收场」。清空多模态/grammar，并把 origin_input_ids 替换为单个占位
    def set_finish_with_abort(self, error_msg: str):
        if get_parallel().tp_rank == 0:
            logger.error(f"{error_msg}, {self.rid=}")
        self.multimodal_inputs = None
        self.grammar = None
        self.origin_input_ids = array(
            "q", [0]
        )  # set it to one token to skip the long prefill  # 中译：设为单个 token 以跳过漫长的 prefill
        self.return_logprob = False
        self.logprob_start_len = -1
        self.to_finish = FINISH_ABORT(
            error_msg, HTTPStatus.BAD_REQUEST, "BadRequestError"
        )

        # 中译：累计「推理（思考）阶段」消耗的 token 数。一旦遇到 think_end_id（思考结束标记），
    def update_reasoning_tokens(self, token_id, think_end_ids):
        if self._is_reasoning_over:
            return

        if not isinstance(token_id, list):
            token_id = [token_id]

        if self._think_end_matcher is None:
            self._think_end_matcher = TokenSequenceMatcher(think_end_ids)

        matched = self._think_end_match_len
        for position, token in enumerate(token_id):
            matched = self._think_end_matcher.advance(matched, token)
            if matched == len(self._think_end_matcher):
                self.reasoning_tokens += position + 1
                self._is_reasoning_over = True
                return

        self._think_end_match_len = matched
        self.reasoning_tokens += len(token_id)

    def __repr__(self):
        return (
            f"Req(rid={self.rid}, "
            f"input_ids={self.origin_input_ids}, output_ids={self.output_ids}, "
            f"{self.grammar=}, "
            f"{self.sampling_params=})"
        )


class _MambaRadixCacheV2TrackEntry(NamedTuple):
    """中译：Mamba radix cache v2 的「状态追踪」条目——是否追踪、追踪槽索引、追踪到的 seqlen。"""

    track_mask: bool
    track_index: int
    track_seqlen: int


def mamba_lazy_spec_in_window(
    req, mamba_track_interval: int, max_draft_tokens: int
) -> bool:
    """Whether a track-interval crossing is reachable by an in-flight verify.

    kv_committed_len lags device seq_lens by up to one verify under overlap;
    the 2x window absorbs it.
    """
    seq_len = req.kv.kv_committed_len
    window = 2 * max_draft_tokens
    return seq_len // mamba_track_interval != (seq_len + window) // mamba_track_interval


def set_mamba_track_indices_from_reqs(
    batch, track_positions: Optional[List[int]] = None
):
    """Build mamba_track_indices from req objects (authoritative source).

    track_positions: optional per-req ping-pong position override (the lazy
    spec track plan, see mamba_lazy_spec_prepare).
    """
    req_to_token_pool = batch.req_to_token_pool
    all_buffers = req_to_token_pool.req_index_to_mamba_ping_pong_track_buffer_mapping[
        batch.req_pool_indices
    ]  # (bs, ping_pong_size), int64, on device
    if track_positions is None:
        # Guard: mamba_next_track_idx may be None for requests that haven't
        # gone through _alloc_ping_pong_buffer yet (e.g., spec v2 verify path).
        # Default to 0 (first ping-pong slot) to avoid TypeError.
        track_positions = [
            req.mamba_next_track_idx if req.mamba_next_track_idx is not None else 0
            for req in batch.reqs
        ]
    batch.mamba_track_buffer_indices = list(track_positions)
    idx = (
        torch.tensor(
            track_positions,
            dtype=torch.int64,
            pin_memory=True,
        )
        .unsqueeze(1)
        .to(device=all_buffers.device, non_blocking=True)
    )
    batch.mamba_track_indices = (
        torch.gather(all_buffers, 1, idx).squeeze(1).to(torch.int64)
    )


def release_req(
    *,
    req: Req,
    remaing_req_count: int,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    tree_cache: BasePrefixCache,
    hisparse_coordinator: Optional[HiSparseCoordinator],
    # 中译：释放单个请求占用的资源（回撤/抢占时调用）：PD-decode 端先 offload KV，
    offload_kv: bool = True,
) -> bool:
    """Returns False when the KV backup failed and the request cannot be resumed."""
    if hisparse_coordinator is not None and not req.finished():
        hisparse_coordinator.retract_req(req)

    # In decode disaggregation the retracted KV is offloaded to host so it can be
    # restored later without recompute (see resume_retracted_reqs/load_kv_cache).
    # Callers that will recompute the KV instead (PD true-retraction rebootstrap)
    # pass offload_kv=False to skip the wasteful device->host copy.
    backup_saved = True
    if get_disagg().disaggregation_mode == "decode" and offload_kv:
        backup_saved = retraction_backup(
            req,
            tree_cache,
            req_to_token_pool,
            token_to_kv_pool_allocator,
            get_disagg().disaggregation_decode_retraction_backup,
        )
    # TODO (csy): for preempted requests, we may want to insert into the tree
    # 中译：TODO（csy）：对于被抢占的请求，未来可能希望将其插入 tree。
    release_kv_cache(req, tree_cache, is_insert=False)
    # NOTE(lsyin): we should use the newly evictable memory instantly.
    # 中译：应立即复用新释放出的可淘汰内存。
    num_tokens = remaing_req_count * envs.SGLANG_RETRACT_DECODE_STEPS.get()
    evict_from_tree_cache(tree_cache, num_tokens)

    req.reset_for_retract()
    return backup_saved


def retract_all(
    *,
    reqs: List[Req],
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    tree_cache: BasePrefixCache,
    hisparse_coordinator: Optional[HiSparseCoordinator],
    # 中译：批量回撤所有请求，逐个调用 release_req（remaing_req_count 递减，用于估算需腾出的内存）。
    offload_kv: bool = True,
) -> None:
    for idx in range(len(reqs)):
        release_req(
            req=reqs[idx],
            remaing_req_count=len(reqs) - idx,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            hisparse_coordinator=hisparse_coordinator,
            offload_kv=offload_kv,
        )


def compute_extend_logprob_start_len(
    *,
    logprob_start_len: int,
    prefix_len: int,
    extend_len: int,
    full_untruncated_fill_len: int,
) -> int:
    # Key variables:
    # - logprob_start_len: Absolute position in full sequence where logprob computation begins
    # - extend_logprob_start_len: Relative position within current extend batch where logprob computation begins
    # - extend_input_len: Number of tokens that need to be processed in this extend batch
    if logprob_start_len == -1:
        resolved_start = full_untruncated_fill_len
    else:
        # logprob_start_len should be at least the length of the prefix indices
        resolved_start = max(logprob_start_len, prefix_len)
    return min(resolved_start - prefix_len, extend_len)


def _compute_chunked_req_next_prompt_token(
    chunked_req: Optional[Req],
    vocab_size: int,
) -> Optional[int]:
    # 中译：对正在分块 prefill 的请求，取它「下一块」第一个尚未处理的 prompt token；
    """Return the next real prompt token after the fill boundary, skipping
    multimodal placeholder (hash) tokens that lie outside the model vocab."""
    if chunked_req is None:
        return None
    fill_len = chunked_req.extend_range.end
    origin_ids = chunked_req.origin_input_ids
    if fill_len >= len(origin_ids):
        return None
    if origin_ids[fill_len] < vocab_size:
        return int(origin_ids[fill_len])
    return None


@dataclasses.dataclass
class ScheduleBatch(ScheduleBatchDisaggregationDecodeMixin):
    """Store all information of a batch on the scheduler.

    中译：调度器侧「一个批次」的全部信息。它把多个 Req 聚合成可一次性 forward 的批，
          并持有跨向 ForwardBatch 的各种张量与元数据。字段按用途分组（见各分组注释）：
          - Core：请求列表 reqs（ForwardBatch 从中派生 lora_ids/rids/grammars/positions）；
          - 全局共享资源：内存池/缓存/模型配置等（引擎生命周期内不变，各批次相同）；
          - 批次可变调度态：仅调度器使用、不被 ForwardBatch 读取的状态；
          - 跨向 ForwardBatch 的 GPU 张量 / by-value 配置 / host 元数据 / 复合对象。
          构造请用类方法 init_new；prepare_for_extend / prepare_for_decode 负责填充张量；
          filter_batch / merge_batch 负责在调度过程中增删请求。
    """

    # === Core: request list (ForwardBatch derives lora_ids / rids / grammars / positions from it) ===
    # 中译：核心字段——本批次的请求列表（ForwardBatch 由它派生出 lora_ids/rids/grammars/positions）。
    reqs: List[Req]

    # === Global config and shared resources (engine-lifetime; identical across batches) ===
    # 中译：全局配置与共享资源（引擎级、各批次一致）。
    # Memory pool and cache
    # 中译：内存池与缓存——req_to_token 池、token->KV 分配器、前缀树缓存。
    req_to_token_pool: ReqToTokenPool = None
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator = None
    tree_cache: BasePrefixCache = None

    # Batch configs
    # 中译：批次配置。model_config 模型配置；enable_overlap 是否开启重叠调度。
    model_config: ModelConfig = None
    enable_overlap: bool = False

    # Device
    # 中译：设备（默认 cuda）。
    device: str = "cuda"

    # HiSparse (engine-level coordinator ref, same across batches)
    # 中译：HiSparse（分层稀疏注意力）协调器引用（引擎级，各批次一致）。
    hisparse_coordinator: Optional[HiSparseCoordinator] = None

    # === Batch-variant scheduler state (per-batch; not read by ForwardBatch) ===
    # 中译：批次可变的调度态（每批不同，且不会被 ForwardBatch 读取）。
    # Tell whether the current running batch is full so that we can skip
    # the check of whether to prefill new requests.
    # This is an optimization to reduce the overhead of the prefill check.
    # 中译：当前运行批是否已满。满了就跳过「是否要 prefill 新请求」的检查，纯属性能优化。
    batch_is_full: bool = False

    # For chunked prefill in PP
    # 中译：流水线并行（PP）下的分块 prefill 状态——当前正被分块的请求、它下一个 prompt token、
    #       以及本批是否包含该请求的最后一块。
    chunked_req: Optional[Req] = None
    chunked_req_next_prompt_token: Optional[int] = None
    contains_last_prefill_chunk: bool = True

    # For DP attention
    # 中译：DP attention 用：inner_idle_batch 是嵌套的空转批次（某些 rank 无请求时参与同步）。
    inner_idle_batch: Optional[ScheduleBatch] = None
    # Decode requests carried alongside a chunked-prefill batch
    # 中译：随分块 prefill 批次一起携带的 decode 请求。
    decoding_reqs: List[Req] = None

    # For split prefill
    # 中译：分段 prefill 状态：split_index 当前分段下标；split_prefill_finished 是否完成；
    #       split_forward_count 总分段次数；split_forward_batch 对应的分段前向批。
    split_index: int = 0
    split_prefill_finished: bool = False
    split_forward_count: int = 1
    split_forward_batch: ForwardBatch = None

    # CPU mirror of req_pool_indices; schedule-path only (used in overlap_utils,
    # not read by ForwardBatch), stale in spec draft window
    # 中译：req_pool_indices 的 CPU 镜像；仅调度路径使用（overlap_utils 中用到，不被 ForwardBatch 读），
    #       在投机草稿窗口期间可能已过期。
    req_pool_indices_cpu: torch.Tensor = None  # shape: [b], int64

    # Forward-pass metrics
    # 中译：前向耗时指标——本批前向开始时间。
    fpm_start_time: float = 0.0

    # hicache pointer for synchronizing data loading from CPU to GPU
    # 中译：HiCache 消费者指针，用于同步从 CPU 到 GPU 的数据加载。
    hicache_consumer_index: int = -1

    # Metrics
    # 中译：指标：DP 协作信息、prefill 统计、前向迭代计数。
    dp_cooperation_info: Optional[DPCooperationInfo] = None
    prefill_stats: Optional[PrefillStats] = None
    forward_iter: Optional[int] = None
    launch_ts: Optional[float] = None
    after_idle_gap: bool = False

    # === GPU tensors crossing to ForwardBatch (clone targets for stream isolation) ===
    # 中译：会传给 ForwardBatch 的 GPU 张量组（overlap 调度下作为 clone 目标以隔离 CUDA stream）。
    # Batched arguments to model runner
    # 中译：传给 model runner 的批量参数。input_ids 为本批所有 token 拼接后的 1D 张量。
    input_ids: torch.Tensor = None  # shape: [b], int64
    # Staging consumed by resolve_forward_inputs (prefill H2D / mixed gather).
    # 中译：供 resolve_forward_inputs 消费的暂存（prefill 的 H2D 拷贝 / 混合批 gather）；
    #       mix_running_indices 记录混入的 running 请求槽位；input_embeds 为直接以嵌入作输入时的张量。
    prefill_input_ids_cpu: Optional[torch.Tensor] = None
    mix_running_indices: Optional[torch.Tensor] = None
    input_embeds: torch.Tensor = None  # shape: [b, hidden_size], float32

    # Token replacement embeddings and absolute positions (optional).
    # 中译：（可选）token 替换嵌入及其绝对位置（如多模态占位 token 被真实特征替换）。
    replace_embeds: Optional[torch.Tensor] = None
    replace_positions: Optional[torch.Tensor] = None

    # Read by ForwardBatch ngram embedding init
    # 中译：供 ForwardBatch 的 ngram 嵌入初始化读取的 token 表。
    ne_token_table: torch.Tensor = None
    # Mask marking chunked (not-yet-finished) prefill requests whose sampled
    # pseudo next-token must NOT be written into the ngram token table.
    ne_skip_token_table_update: torch.Tensor = None

    req_pool_indices: torch.Tensor = None  # shape: [b], int64  # 各请求在 req_to_token 池中的行号
    seq_lens: torch.Tensor = None  # shape: [b], int64  # 各请求当前序列长度
    # 中译：注意 seq_lens（含填充）与 orig_seq_lens（原始）在长上下文（Qwen-1M）等场景可能不同。

    # The original sequence lengths, Qwen-1M related
    # 中译：原始（未填充）序列长度，Qwen-1M 等长上下文场景相关。
    orig_seq_lens: torch.Tensor = None  # shape: [b], int32

    # The output locations of the KV cache
    # 中译：本批每个 token 的 KV 写入位置（在 KV 池中的槽位索引）。
    out_cache_loc: torch.Tensor = None  # shape: [b], int64
    # DSV4-NPU: KV-only per-pool slot bundle from
    # DSV4NPUTokenToKVPoolAllocator (None elsewhere).
    out_cache_loc_dsv4: Optional[Any] = None

    # For hybrid GDN prefix cache
    # 中译：混合 GDN 前缀缓存用的 mamba 跟踪张量：track_indices 跟踪槽索引、
    #       track_mask 是否需跟踪、track_seqlens 跟踪的 seqlen。
    mamba_track_indices: torch.Tensor = None  # shape: [b], int64
    # Per-batch snapshot of the logical ping-pong positions selected for this
    # forward (normally req.mamba_next_track_idx; spec may override it). Result
    # processing uses it to update req.mamba_last_track_idx, since both req-level
    # indices may advance under overlap.
    mamba_track_buffer_indices: Optional[List[int]] = None  # shape: [b], 0 or 1
    mamba_track_mask: torch.Tensor = None  # shape: [b], bool
    mamba_track_seqlens: torch.Tensor = None  # shape: [b], int64
    mamba_track_mask_cpu: Optional[List[bool]] = None  # shape: [b]
    mamba_track_mask_next_cpu: Optional[List[bool]] = None  # shape: [b]
    mamba_decode_batch_idx_cpu: Optional[List[int]] = None  # shape: [b]
    # Lazy + spec: this iteration's per-req scatter positions
    # (see mamba_lazy_spec_prepare).
    mamba_lazy_spec_track_positions_cpu: Optional[List[int]] = None  # shape: [b]
    # Deferred mamba init ops: COW pairs and clear indices (performed on forward stream)
    # 中译：延迟的 mamba 初始化操作：COW（写时复制）的源/目标索引对与需清零的索引（在前向 stream 上执行）。
    mamba_cow_src_indices: torch.Tensor = None
    mamba_cow_dst_indices: torch.Tensor = None
    mamba_clear_indices: torch.Tensor = None

    # Encoder-decoder device tensors (host fields in the host metadata group)
    # 中译：编码器-解码器架构的 device 张量：encoder_lens 各请求编码器长度、
    #       encoder_out_cache_loc 编码器 KV 写入位置（host 侧字段在 host 元数据组）。
    encoder_lens: Optional[torch.Tensor] = None
    encoder_out_cache_loc: Optional[torch.Tensor] = None

    # It comes empty list if logprob is not required.
    # 中译：输入侧 logprob 计算所需的 token id；不需 logprob 时为空列表。
    extend_input_logprob_token_ids: Optional[torch.Tensor] = None

    # === Config / flags crossing to ForwardBatch (by-value) ===
    # 中译：按值传给 ForwardBatch 的配置/标志组。
    # 中译：forward_mode 是本批的前向模式（EXTEND/DECODE/MIXED/IDLE/...），决定 model runner 走哪条路径。
    forward_mode: ForwardMode = None
    # 中译：global_forward_mode——DP attention 下跨 rank 协商出的全局前向模式。
    global_forward_mode: Optional[ForwardMode] = None

    # For DP attention
    # 中译：DP attention 标志：is_extend_in_batch 本 rank 是否含 extend；all_extend_in_batch 是否全为 extend；
    #       can_run_dp_cuda_graph / can_run_dp_breakable_cuda_graph 是否可走（可打断的）DP CUDA Graph；
    #       tbo_split_seq_index 为 TBO（two-batch overlap）的序列切分位置。
    is_extend_in_batch: bool = False
    can_run_decode_cuda_graph: bool = False
    can_run_dp_prefill_cuda_graph: bool = False
    tbo_split_seq_index: Optional[int] = None
    # Rank-consistent forward mode for the recv skipper, derived from the MLP
    # sync all-gather (the TBO-only `global_forward_mode` is None without TBO).
    recv_skipper_forward_mode: Optional[ForwardMode] = None
    spec_verify_tier_num_tokens: int = -1

    # For processing logprobs
    # 中译：本批是否需返回 logprob。
    return_logprob: bool = False

    # Whether this batch is prefill-only (no token generation needed)
    # 中译：本批是否为「仅 prefill」（无需生成 token，如 embedding/打分）。
    is_prefill_only: bool = False

    # Speculative decoding
    # 中译：本批使用的投机解码算法（None 表示未启用）。
    spec_algorithm: SpeculativeAlgorithm = None

    # Whether to return hidden states
    # 中译：本批是否需返回隐藏状态。
    return_hidden_states: bool = False
    return_hidden_states_mode: CaptureHiddenMode = CaptureHiddenMode.NULL

    # Has grammar
    # 中译：本批是否含约束解码（grammar）请求。
    has_grammar: bool = False

    # The sum of all sequence lengths
    # 中译：所有序列长度之和；extend_num_tokens 为本批 extend 阶段实际要处理的 token 总数。
    seq_lens_sum: int = None
    extend_num_tokens: Optional[int] = None

    # Diffusion LLM
    dllm_config: Optional[DllmConfig] = None

    # === Host metadata crossing to ForwardBatch (CPU lists / mirrors) ===
    # 中译：传给 ForwardBatch 的 host（CPU）侧元数据组（CPU 列表或 GPU 张量的 CPU 镜像）。
    seq_lens_cpu: torch.Tensor = None  # shape: [b], int64  # seq_lens 的 CPU 镜像

    # For multimodal inputs
    # 中译：本批各请求的多模态输入列表（与 reqs 一一对应）。
    multimodal_inputs: Optional[List] = None

    # For processing logprobs
    # 中译：logprob 处理：top_logprobs_nums 各请求的 top-k 个数；token_ids_logprobs 各请求指定要算 logprob 的 token id。
    top_logprobs_nums: Optional[List[int]] = None
    token_ids_logprobs: Optional[List[List[int]]] = None

    # For encoder-decoder architectures
    # 中译：编码器-解码器架构的 host 侧字段：encoder_cached 各请求编码器输出是否已缓存；encoder_lens_cpu 编码器长度的 CPU 镜像。
    encoder_cached: Optional[List[bool]] = None
    encoder_lens_cpu: Optional[List[int]] = None

    # For extend and mixed chunekd prefill
    # 中译：extend / 混合分块 prefill 用的逐请求长度列表：
    #       prefix_lens 各请求已命中前缀长度；extend_lens 各请求本批要 prefill 的 token 数；
    #       extend_logprob_start_lens 各请求 logprob 在本批内的相对起点。
    prefix_lens: List[int] = None
    extend_lens: List[int] = None
    extend_logprob_start_lens: List[int] = None

    # For DP attention
    # 中译：DP attention 下跨各 DP rank 的 token 总数（用于同步/填充）；
    #       global_num_tokens_for_logprob 为需计算 logprob 的跨 rank token 总数。
    global_num_tokens: Optional[List[int]] = None
    global_num_tokens_for_logprob: Optional[List[int]] = None
    global_spec_verify_tier_num_tokens: Optional[List[int]] = None

    # Member rows riding one forward; None whenever reqs and rows are 1:1.
    beam_tail: Optional[BeamTail] = None

    # === Compound crossing to ForwardBatch (carry their own device tensors) ===
    # 中译：传给 ForwardBatch 的复合对象组（它们自带各自的 device 张量）。
    # Sampling info
    # 中译：批量采样信息（温度/top-p/惩罚器等），由 SamplingBatchInfo.from_schedule_batch 构建。
    sampling_info: SamplingBatchInfo = None

    # Speculative decoding
    # spec_info: Optional[SpecInput] = None
    # 中译：投机解码信息（如 EagleDraftInput）。非 None 时，decode 阶段的内存分配/准备交由它接管。
    spec_info: Optional[SpecInput] = None

    # 中译：一次性的「单次 forward 覆盖项」，由 ForwardBatch.init_new 消费后重置。
    @classmethod
    def init_new(
        # 中译：从一组 Req 构造新的 ScheduleBatch，并据各请求汇总出 return_logprob/has_grammar/
        #       is_prefill_only 等批级标志。此时尚未填充张量（由 prepare_for_extend/decode 完成）。
        cls,
        reqs: List[Req],
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tree_cache: BasePrefixCache,
        model_config: ModelConfig,
        enable_overlap: bool,
        spec_algorithm: SpeculativeAlgorithm,
        chunked_req: Optional[Req] = None,
        dllm_config: Optional[DllmConfig] = None,
    ):
        return_logprob = any(req.return_logprob for req in reqs)

        return_hidden_states_mode = get_batch_return_hidden_states_mode(reqs)

        batch = cls(
            reqs=reqs,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=model_config,
            enable_overlap=enable_overlap,
            return_logprob=return_logprob,
            has_grammar=any(req.grammar for req in reqs),
            device=req_to_token_pool.device,
            spec_algorithm=spec_algorithm,
            return_hidden_states=return_hidden_states_mode.need_capture(),
            return_hidden_states_mode=return_hidden_states_mode,
            is_prefill_only=all(req.is_prefill_only for req in reqs),
            chunked_req=chunked_req,
            chunked_req_next_prompt_token=_compute_chunked_req_next_prompt_token(
                chunked_req,
                model_config.vocab_size,
            ),
            dllm_config=dllm_config,
        )
        return batch

    def batch_size(self):
        # 中译：批大小 = 请求数。
        return len(self.reqs)

    def is_empty(self):
        # 中译：批是否为空（无请求）。
        return len(self.reqs) == 0

    def is_dllm(self):
        # 中译：是否为扩散式 LLM（Diffusion LLM）批次。
        return self.dllm_config is not None

    def grammar_needs_sync(self) -> bool:
        """Whether grammar forces this batch onto the synchronous path, i.e. the
        previous batch's result is resolved before this forward."""
        return self.has_grammar and not self.spec_algorithm.supports_grammar_overlap()

    def prepare_encoder_info_extend(
        self, input_ids: List[array[int]], seq_lens: List[int]
    ):
        # 中译：encoder-decoder 模型 extend 阶段的专门处理——把每个请求里的 encoder 部分
        #       从 decoder 输入中剥离，分别记录 encoder/decoder 的 out_cache_loc，并相应调整
        #       seq_lens/extend_lens/prefix_lens 与 logprob 相关长度。
        _pin = is_pin_memory_available(self.device)
        encoder_lens_cpu = []
        encoder_cached = []

        for req in self.reqs:
            im = req.multimodal_inputs
            if im is None or im.num_image_tokens is None:
                # No image input
                encoder_lens_cpu.append(0)
                encoder_cached.append(True)
            else:
                encoder_lens_cpu.append(im.num_image_tokens)
                encoder_cached.append(
                    self.forward_mode.is_decode()
                    or len(req.prefix_indices) >= im.num_image_tokens
                )
        self.encoder_lens_cpu = encoder_lens_cpu
        self.encoder_cached = encoder_cached

        self.encoder_lens = torch.tensor(
            self.encoder_lens_cpu, dtype=torch.int64, pin_memory=_pin
        ).to(self.device, non_blocking=True)

        # Strip encoder infos
        pt = 0
        decoder_out_cache_loc = []
        encoder_out_cache_loc = []
        extend_lens = self.extend_lens[:]
        prefix_lens = self.prefix_lens[:]
        for i, req in enumerate(self.reqs):
            encoder_len = self.encoder_lens_cpu[i]
            seq_lens[i] -= encoder_len

            if len(req.prefix_indices) < encoder_len:
                # NOTE: the encoder part should be considered as a whole
                assert len(req.prefix_indices) == 0
                input_ids[i] = input_ids[i][encoder_len:]
                encoder_out_cache_loc.append(self.out_cache_loc[pt : pt + encoder_len])
                decoder_out_cache_loc.append(
                    self.out_cache_loc[pt + encoder_len : pt + req.extend_range.length]
                )
                extend_lens[i] -= encoder_len
                self.extend_num_tokens = self.extend_num_tokens - encoder_len
            else:
                decoder_out_cache_loc.append(
                    self.out_cache_loc[pt : pt + req.extend_range.length]
                )
                prefix_lens[i] -= encoder_len

            pt += req.extend_range.length
        self.extend_lens = extend_lens
        self.prefix_lens = prefix_lens

        # Reassign: ED stripping rebuilds prefill_input_ids_cpu (CPU pinned);
        # resolve_forward_inputs will H2D this on forward stream. self.input_ids
        # stays None.
        self.prefill_input_ids_cpu = flatten_arrays_to_pinned_cpu(input_ids, _pin)
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, pin_memory=_pin).to(
            self.device, non_blocking=True
        )
        self.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)

        if not decoder_out_cache_loc:
            self.out_cache_loc = torch.zeros(0, dtype=torch.int64).to(
                self.device, non_blocking=True
            )
        else:
            self.out_cache_loc = torch.cat(decoder_out_cache_loc)

        if not encoder_out_cache_loc:
            self.encoder_out_cache_loc = torch.zeros(0, dtype=torch.int64).to(
                self.device, non_blocking=True
            )
        else:
            self.encoder_out_cache_loc = torch.cat(encoder_out_cache_loc)

        assert (
            len(self.out_cache_loc) == self.extend_num_tokens
        ), f"Expected {len(self.out_cache_loc)}, got {self.extend_num_tokens}"

        if self.extend_input_logprob_token_ids is not None:
            new_token_ids_parts = []
            offset = 0
            extend_logprob_start_lens = self.extend_logprob_start_lens[:]
            for i, req in enumerate(self.reqs):
                encoder_len = self.encoder_lens_cpu[i]
                old_start_len = extend_logprob_start_lens[i]
                old_contribution = req.extend_range.length - old_start_len

                if len(req.prefix_indices) < encoder_len:
                    tokens_to_strip = max(0, encoder_len - old_start_len)
                    new_token_ids_parts.append(
                        self.extend_input_logprob_token_ids[
                            offset + tokens_to_strip : offset + old_contribution
                        ]
                    )
                    extend_logprob_start_lens[i] = max(0, old_start_len - encoder_len)
                else:
                    new_token_ids_parts.append(
                        self.extend_input_logprob_token_ids[
                            offset : offset + old_contribution
                        ]
                    )

                offset += old_contribution
            self.extend_logprob_start_lens = extend_logprob_start_lens

            if new_token_ids_parts:
                self.extend_input_logprob_token_ids = torch.cat(new_token_ids_parts)
            else:
                self.extend_input_logprob_token_ids = None

        for i, req in enumerate(self.reqs):
            encoder_len = self.encoder_lens_cpu[i]
            if encoder_len == 0:
                continue
            if len(req.prefix_indices) < encoder_len:
                assert len(req.prefix_indices) == 0
                req.extend_range = req.extend_range._replace(
                    start=req.extend_range.start + encoder_len
                )
            req.logprob_start_len = max(req.logprob_start_len, encoder_len)

    def prepare_for_extend(self):
        # 中译：为 EXTEND（prefill）阶段准备整批的张量与元数据。核心步骤：
        #       1) 每个请求取「去掉已命中前缀后」要新算的 input token；
        #       2) 计算各种长度（seq_lens / prefix_lens / extend_lens 等）并搬上 device；
        #       3) alloc_for_extend 分配 KV 槽位与 req_pool 行号；
        #       4) 逐请求更新内存管理字段、统计 cached_tokens 分层细分、收集多模态/mamba/logprob 信息；
        #       5) 构建批量采样信息 sampling_info。完成后本批即可交给 ForwardBatch.init_new。
        self.forward_mode = ForwardMode.EXTEND

        if self.is_dllm():
            # For DLLM, we use a separate forward mode
            # 中译：扩散式 LLM 使用单独的前向模式 DLLM_EXTEND。
            self.forward_mode = ForwardMode.DLLM_EXTEND

        # Init tensors
        reqs = self.reqs
        # 中译：每个请求只取「前缀之后」的部分作为本次真正要 prefill 的 token。
        input_ids = [r.get_fill_ids()[len(r.prefix_indices) :] for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = [r.extend_range.end for r in reqs]
        orig_seq_lens = [max(r.extend_range.end, len(r.origin_input_ids)) for r in reqs]
        prefix_lens = [len(r.prefix_indices) for r in reqs]
        extend_lens = [r.extend_range.length for r in reqs]
        extend_logprob_start_lens = [
            compute_extend_logprob_start_len(
                logprob_start_len=r.logprob_start_len,
                prefix_len=prefix_lens[i],
                extend_len=extend_lens[i],
                full_untruncated_fill_len=len(r.full_untruncated_fill_ids),
            )
            for i, r in enumerate(reqs)
        ]

        _pin = is_pin_memory_available(self.device)
        # Stay on pinned CPU; H2D is deferred to forward stream via
        # resolve_forward_inputs.
        # 中译：input_ids 先停在锁页（pinned）CPU 上，H2D（host->device）拷贝推迟到
        #       前向 stream 上由 resolve_forward_inputs 执行，避免阻塞调度。
        pinned_input_ids = flatten_arrays_to_pinned_cpu(input_ids, _pin)
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int64, pin_memory=_pin).to(
            self.device, non_blocking=True
        )
        seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        orig_seq_lens_tensor = torch.tensor(
            orig_seq_lens, dtype=torch.int32, pin_memory=_pin
        ).to(self.device, non_blocking=True)

        # Set batch fields needed by alloc_for_extend
        self.prefix_lens = prefix_lens
        self.extend_lens = extend_lens
        self.seq_lens = seq_lens_tensor
        self.seq_lens_cpu = seq_lens_cpu
        self.extend_num_tokens = extend_num_tokens

        # Allocate memory
        # 中译：为本批 extend 分配 KV 缓存写入位置与 req_pool 行号（device 张量 + CPU 镜像）。
        out_cache_loc, req_pool_indices_tensor, req_pool_indices_cpu = alloc_for_extend(
            self
        )

        # Set fields
        input_embeds = []
        all_replace_embeds: List[torch.Tensor] = []
        all_replace_positions: List[int] = []
        has_replace_embeds = False
        input_id_pointer = 0
        input_id_lens = [len(input_id) for input_id in input_ids]
        extend_input_logprob_token_ids = []
        multimodal_inputs = []
        mamba_track_mask_cpu = []
        mamba_track_indices_cpu = []
        mamba_track_seqlens_cpu = []

        # 中译：逐请求填充各项信息（input_embeds、位置嵌入覆盖、多模态、cached_tokens 统计、
        #       mamba 追踪、input logprob token 等）。
        for i, (req, seq_len, pre_len) in enumerate(zip(reqs, seq_lens, prefix_lens)):
            assert seq_len - pre_len == req.extend_range.length

            req.extend_batch_idx += 1

            # 中译：本请求 prefill 后，已提交/已分配 KV 长度都等于当前序列长度。
            # If input_embeds are available, store them
            # 中译：若请求携带预计算的 input_embeds（直接给嵌入而非 token id），则按
            #       extend_input_len 切片后存下。因为分块溢出时 PrefillAdder 会截断
            #       fill_len/extend_input_len，但不会截断 input_embeds，故需手动对齐。
            if req.input_embeds is not None:
                # Slice to match extend_input_len — PrefillAdder truncates
                # fill_len/extend_input_len on chunk overflow but not input_embeds.
                input_embeds.extend(
                    req.input_embeds[pre_len : pre_len + req.extend_range.length]
                )

            if req.positional_embed_overrides is not None:
                # Override positions are absolute in the full sequence.
                # Convert to extend-tensor coordinates by subtracting pre_len,
                # then skip any that fall within the cached prefix.
                # 中译：位置嵌入覆盖项的 position 是“完整序列”中的绝对位置；减去 pre_len 转换成
                #       本次 extend 张量内的坐标，并跳过落在已缓存前缀内的那些位置。
                embeds_to_add = []
                for embed_idx, pos in enumerate(
                    req.positional_embed_overrides.positions
                ):
                    extend_pos = pos - pre_len
                    if extend_pos < 0 or extend_pos >= req.extend_range.length:
                        continue  # Outside current extend chunk, skip
                    embeds_to_add.append((embed_idx, input_id_pointer + extend_pos))
                if embeds_to_add:
                    has_replace_embeds = True
                    indices, positions = zip(*embeds_to_add)
                    all_replace_embeds.append(
                        req.positional_embed_overrides.embeds[list(indices)]
                    )
                    all_replace_positions.extend(positions)
            input_id_pointer += input_id_lens[i]

            multimodal_inputs.append(req.multimodal_inputs)

            # Only calculate cached_tokens once. Once retracted, the 'retracted_stain'
            # flag will always True
            # 中译：cached_tokens 只统计一次。一旦请求被回撤过，retracted_stain 永远为 True，
            #       以后不再重复计入缓存命中（避免重复统计）。
            if not req.retracted_stain:
                new_cached = pre_len - req.already_computed
                req.cached_tokens += new_cached

                # Calculate detailed breakdown of cached tokens by source (for HiCache)
                # Only compute once on FIRST chunk - subsequent chunks in chunked prefill
                # would incorrectly count previously computed tokens as cache hits.
                # 中译：为 HiCache 计算缓存命中的分层细分（device/host/L3）。只在“第一块”计算
                #       一次；分块 prefill 的后续块会错误地把之前已计算 token 当成缓存命中。
                if not req._cache_breakdown_computed:
                    # storage_hit_length is set by scheduler.pop_prefetch_loaded_tokens()
                    # after prefetch completes.
                    (
                        req.cached_tokens_device,
                        req.cached_tokens_host,
                        req.cached_tokens_storage,
                    ) = split_cached_prefix_by_tier(
                        prefix_len=len(req.prefix_indices),
                        host_hit_len=req.host_hit_length,
                        storage_hit_len=req.storage_hit_length,
                    )
                    req._cache_breakdown_computed = True

                req.already_computed = seq_len
            req.is_retracted = False

            if mamba_extra_buffer_enabled():
                track_entry = self._mamba_radix_cache_v2_req_prepare_for_extend(req)
                mamba_track_mask_cpu.append(track_entry.track_mask)
                mamba_track_indices_cpu.append(track_entry.track_index)
                mamba_track_seqlens_cpu.append(track_entry.track_seqlen)

            if self.return_logprob:
                # Find input logprob token ids.
                # First, find a global index within origin_input_ids and slide it by 1
                # to compute input logprobs. It is because you need the next token
                # to compute input logprobs. E.g., (chunk size 2)
                # 中译：找出用于计算 input logprob 的 token id。原理：要算某位置的 input logprob，
                #       需要其“下一个 token”作为目标，所以在 origin_input_ids 里取一个全局起点
                #       后整体向后滑动 1 位；越界部分用 0 补齐（示例见上方英文）。
                #
                # input_logprobs = [1, 2, 3, 4]
                # get_fill_ids() = [1, 2]
                # extend_input_logprob_token_id = [2, 3]
                #
                # Note that it can also overflow. In this case, we pad it with 0.
                # input_logprobs = [1, 2, 3, 4]
                # get_fill_ids() = [3, 4]
                # extend_input_logprob_token_id = [4, 0]
                global_start_idx, global_end_idx = (
                    len(req.prefix_indices),
                    req.extend_range.end,
                )
                if req.logprob_start_len == -1:
                    logprob_start_len = len(req.origin_input_ids)
                else:
                    logprob_start_len = req.logprob_start_len
                # Apply logprob_start_len
                if global_start_idx < logprob_start_len:
                    global_start_idx = logprob_start_len

                logprob_token_ids = req.origin_input_ids[
                    global_start_idx + 1 : global_end_idx + 1
                ]
                extend_input_logprob_token_ids.extend(logprob_token_ids)

                # We will need req.extend_range.length - extend_logprob_start_lens[i] number of
                # tokens, and logprob_token_ids is for input logprob, so pad the rest of them by 0.
                extend_input_logprob_token_ids.extend(
                    [0]
                    * (
                        req.extend_range.length
                        - extend_logprob_start_lens[i]
                        - len(logprob_token_ids)
                    )
                )

        if self.return_logprob:
            extend_input_logprob_token_ids = torch.tensor(
                extend_input_logprob_token_ids
            )
            # Clamp placeholder or out-of-range token IDs (e.g., multimodal hashes)
            # so they stay within the vocab boundary before being sent to GPU.
            extend_input_logprob_token_ids.clamp_(0, self.model_config.vocab_size - 1)
        else:
            extend_input_logprob_token_ids = None

        if has_replace_embeds:
            replace_embeds_tensor = torch.cat(all_replace_embeds, dim=0).to(
                self.device, non_blocking=True
            )
            replace_positions_tensor = torch.tensor(
                all_replace_positions, dtype=torch.long, device=self.device
            )
        else:
            replace_embeds_tensor = None
            replace_positions_tensor = None

        self.input_ids = None
        self.prefill_input_ids_cpu = pinned_input_ids
        self.req_pool_indices = req_pool_indices_tensor
        self.req_pool_indices_cpu = req_pool_indices_cpu
        self.orig_seq_lens = orig_seq_lens_tensor
        self.out_cache_loc = out_cache_loc
        self.input_embeds = (
            torch.tensor(input_embeds, pin_memory=_pin).to(
                self.device, non_blocking=True
            )
            if input_embeds
            else None
        )
        self.replace_embeds = replace_embeds_tensor
        self.replace_positions = replace_positions_tensor
        for mm_input in multimodal_inputs:
            if mm_input is None:
                continue
            if isinstance(mm_input.vision_position_ids, torch.Tensor):
                mm_input.vision_position_ids = mm_input.vision_position_ids.to(
                    self.device, non_blocking=True
                )
            if isinstance(mm_input.visible_frame_counts, torch.Tensor):
                mm_input.visible_frame_counts = mm_input.visible_frame_counts.to(
                    self.device, non_blocking=True
                )
        self.multimodal_inputs = multimodal_inputs
        self.seq_lens_sum = sum(seq_lens)

        if self.return_logprob:
            self.top_logprobs_nums = [r.logprob.top_logprobs_num for r in reqs]
            self.token_ids_logprobs = [r.logprob.token_ids_logprob for r in reqs]

        self.extend_logprob_start_lens = extend_logprob_start_lens
        self.extend_input_logprob_token_ids = extend_input_logprob_token_ids

        if mamba_extra_buffer_enabled():
            self.mamba_track_indices = torch.tensor(
                mamba_track_indices_cpu,
                dtype=torch.int64,
                device=self.device,
            )
            self.mamba_track_mask = torch.tensor(
                mamba_track_mask_cpu,
                dtype=torch.bool,
                device=self.device,
            )
            self.mamba_track_seqlens = torch.tensor(
                mamba_track_seqlens_cpu,
                dtype=torch.int64,
                device=self.device,
            )

        # Collect mamba init info for deferred ops on forward stream
        # 中译：收集 mamba 初始化信息，供后续在前向 stream 上执行延迟的 COW/clear 操作。
        if any(req.mamba_pool_idx is not None for req in reqs):
            self._collect_deferred_mamba_cow_and_clear(reqs)

        if self.model_config.is_encoder_decoder:
            self.prepare_encoder_info_extend(input_ids, seq_lens)

        # Build sampling info
        # 中译：构建批量采样信息（温度、top-p、各类惩罚器等），供模型 forward 后采样使用。
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    def _mamba_radix_cache_v2_req_prepare_for_extend(
        # 中译：为 Mamba radix cache v2 计算单个请求在本次 extend 中的「状态追踪」条目，
        #       决定是否追踪 mamba 状态、追踪到哪个槽、追踪的 seqlen（处理对齐/分支点等边界）。
        self,
        req: Req,
    ) -> _MambaRadixCacheV2TrackEntry:
        cache_chunk_size = mamba_cache_chunk_size()
        state_chunk_size = getattr(
            self.model_config.hf_text_config, "mamba_chunk_size", 64
        )
        # Donated depth must land on the actual DCP-widened radix page, while
        # kernel snapshots stay on the cache chunk grid.
        checkpoint_grid = mamba_checkpoint_grid(self.tree_cache.page_size)

        def _force_track_h(i: int) -> int:
            # h is indexed relative to the extend start, so check that offset.
            assert (i - len(req.prefix_indices)) % cache_chunk_size == 0, (
                f"The force track calculation only handles last-position or "
                f"unaligned seqlens, so it needs a chunk-aligned offset to "
                f"start from. But i={i} prefix_len={len(req.prefix_indices)} "
                f"chunk_size={cache_chunk_size} checkpoint_grid={checkpoint_grid}"
            )
            # There are 3 cases for mamba_track_seqlen passed to mamba_track_seqlens_cpu:
            # 1) aligned with cache_chunk_size-> retrieve from last_recurrent_state
            #    a) is the last position -> retrieve from last_recurrent_state
            #    b) is NOT the last position -> retrieve from h
            # 2) unaligned with cache_chunk_size -> retrieve from h
            # Currently, the math calculation only supports case 1a and 2. So for 1b, we need to add 1
            # to force the math calculation to retrieve the correct mamba state from h.
            # 中译：传给 mamba_track_seqlens_cpu 的 mamba_track_seqlen 有三种情况：
            #       1) 与 mamba_cache_chunk_size 对齐——a) 是最后位置：从 last_recurrent_state 取；
            #          b) 不是最后位置：从 h 取；2) 未对齐：从 h 取。
            #       当前数学计算只支持 1a 与 2，所以针对 1b 要加 1，强制从 h 取出正确的 mamba 状态。
            return i + 1

        mask = req.extend_range.length >= checkpoint_grid
        track_index = req.mamba_ping_pong_track_buffer[req.mamba_next_track_idx].item()
        mamba_track_seqlen = -1
        if mask:
            # mamba_track_seqlen is used to calculate the indices to track in
            # hybrid_linear_attn_backend's _init_track_ssm_indices. Due to the
            # fact that the ssm state between aligned and non-aligned are retrieved differently,
            # if 1) last pos and 2) is aligned, then retrieved from the last_recurrent_state,
            # otherwise retrieved from h (i.e. unaligned).
            # We need to pass the non-aligned seqlen to the calculation. Even though
            # we pass in mamba_track_seqlen, the actual tracked seqlen is mamba_last_track_seqlen.
            mamba_track_seqlen = len(req.prefix_indices) + req.extend_range.length

            # mamba_track_seqlen_aligned/mamba_last_track_seqlen is actual tracked seqlen. Used to pass to
            # mamba radix cache to track which seqlen this mamba state should store at.
            mamba_track_seqlen_aligned = (
                len(req.prefix_indices)
                + (req.extend_range.length // checkpoint_grid) * checkpoint_grid
            )

            # A coarser checkpoint grid may not be a model-state boundary, so
            # force retrieval from the intermediate h state in that case.
            mamba_track_fla_chunk_aligned = (
                len(req.prefix_indices)
                + (req.extend_range.length // state_chunk_size) * state_chunk_size
            )
            if mamba_track_fla_chunk_aligned != mamba_track_seqlen_aligned:
                # We want to track mamba_track_seqlen_aligned, and it's not the last position,
                # so we need to add 1 to the seqlen to retrieve the correct mamba state from h.
                mamba_track_seqlen = _force_track_h(mamba_track_seqlen_aligned)

            # In lazy mode, skip the swap — the second ping-pong slot is not
            # allocated yet; it will be allocated on demand at the track boundary
            # in mamba_lazy_prealloc_at_boundary during prepare_for_decode.
            req.mamba_last_track_idx = req.mamba_next_track_idx
            if not mamba_extra_buffer_lazy_enabled():
                req.mamba_next_track_idx = (
                    self.req_to_token_pool.get_mamba_ping_pong_other_idx(
                        req.mamba_next_track_idx
                    )
                )
            if req.mamba_branching_seqlen is not None:
                # track branching point in this forward if the branching point
                # is within the current extend batch.
                branching_seqlen_aligned_mask = (
                    req.mamba_branching_seqlen - len(req.prefix_indices)
                ) % cache_chunk_size == 0
                if (
                    req.mamba_branching_seqlen > len(req.prefix_indices)
                    and req.mamba_branching_seqlen < mamba_track_seqlen
                    and branching_seqlen_aligned_mask
                ):
                    # We want to track mamba_track_seqlen_aligned, and it's not the last position,
                    # so we need to add 1 to the seqlen to retrieve the correct mamba state from h.
                    # See _force_track_h() for more details.
                    mamba_track_seqlen = _force_track_h(req.mamba_branching_seqlen)
                    mamba_track_seqlen_aligned = req.mamba_branching_seqlen
            req.mamba_last_track_seqlen = mamba_track_seqlen_aligned

        return _MambaRadixCacheV2TrackEntry(
            track_mask=mask,
            track_index=track_index,
            track_seqlen=mamba_track_seqlen,
        )

    def _collect_deferred_mamba_cow_and_clear(self, reqs):
        """Collect deferred COW/clear info from requests.

        中译：从各请求收集「延后到 forward stream 执行」的 mamba 初始化操作：
              COW（copy-on-write，从源槽复制状态对）与 clear（新分配槽需清零），
              汇总成批级张量 mamba_cow_src/dst_indices、mamba_clear_indices。
        """
        cow_src_tensors = []
        cow_dst_tensors = []
        clear_tensors = []
        for req in reqs:
            if req.mamba_cow_src_index is not None:
                cow_src_tensors.append(req.mamba_cow_src_index)
                cow_dst_tensors.append(req.mamba_pool_idx.unsqueeze(0))
                req.mamba_cow_src_index = None
                req.mamba_needs_clear = False
            elif req.mamba_needs_clear:
                clear_tensors.append(req.mamba_pool_idx.unsqueeze(0))
                req.mamba_needs_clear = False
        self.mamba_cow_src_indices = (
            torch.cat(cow_src_tensors) if cow_src_tensors else None
        )
        self.mamba_cow_dst_indices = (
            torch.cat(cow_dst_tensors) if cow_dst_tensors else None
        )
        self.mamba_clear_indices = torch.cat(clear_tensors) if clear_tensors else None

    def prepare_for_split_prefill(self):
        # 中译：分段 prefill：先按常规 extend 准备，再把前向模式改为 SPLIT_PREFILL。
        self.prepare_for_extend()
        # For split prefill, we need to set the forward mode to SPLIT_PREFILL
        # 中译：分段 prefill 需把前向模式设为 SPLIT_PREFILL。
        self.forward_mode = ForwardMode.SPLIT_PREFILL

    def mix_with_running(self, running_batch: ScheduleBatch):
        # 中译：把一批正在 decode 的请求「混入」当前 prefill 批，组成 MIXED 批一起 forward。
        #       running 部分每个请求 extend_input_len 视为 1（即一步 decode），并 merge 进本批，
        #       同时把它们的 prefix_lens/extend_lens 追加进来。
        self.forward_mode = ForwardMode.MIXED
        running_bs = running_batch.batch_size()

        for req in running_batch.reqs:
            req._refresh_fill_ids()
            full_len = len(req.full_untruncated_fill_ids)
            req.set_extend_range(full_len - 1, full_len)

        # Decode tokens of the running portion live in future_map.output_tokens_buf.
        # 中译：running 部分的 decode token 住在 future_map.output_tokens_buf 里（重叠中继），
        #       故这里 input_ids 置 None，改用 mix_running_indices 记下这些请求的槽位。
        self.input_ids = None
        self.mix_running_indices = running_batch.req_pool_indices
        out_cache_loc = torch.cat([self.out_cache_loc, running_batch.out_cache_loc])

        self.merge_batch(running_batch)
        self.out_cache_loc = out_cache_loc

        # For overlap scheduler, the output_ids has one step delay
        # 中译：重叠调度下 output_ids 有一步延迟，故前缀长度计算需相应偏移 delta。
        delta = 0 if self.enable_overlap else -1

        # NOTE: prefix_indices is what has been cached, but we don't cache each decode step
        # 中译：prefix_indices 是已缓存的部分，但 decode 每步并不逐步入缓存，故这里手动
        self.prefix_lens = self.prefix_lens + [
            len(r.origin_input_ids) + len(r.output_ids) + delta
            for r in running_batch.reqs
        ]
        self.extend_lens = self.extend_lens + [1] * running_bs
        self.extend_num_tokens = self.extend_num_tokens + running_bs
        # TODO (lianmin): Revisit this. It should be seq_len - 1
        self.extend_logprob_start_lens = (
            self.extend_logprob_start_lens + [0] * running_bs
        )
        self.is_prefill_only = False

    def convert_decode_to_extend(self):
        """View every decode request as a 1-token extend with its context as
        prefix, making this a plain extend (prefill) batch. Called after the
        DP mlp-sync when a peer rank runs extend, so this rank replays the
        extend graphs; requires prepare_for_decode(); token count unchanged.
        """
        bs = self.batch_size()
        self.forward_mode = ForwardMode.EXTEND
        # Stale residue from this object's life as a prefill batch; the
        # extend-merge path would exclude (silently drop) that req otherwise.
        self.chunked_req = None
        # Also stale residue; None keeps the prefill result path from
        # re-reporting old prefill stats for what is decode work.
        self.prefill_stats = None
        for req in self.reqs:
            req._refresh_fill_ids()
            full_len = len(req.full_untruncated_fill_ids)
            req.set_extend_range(full_len - 1, full_len)

        # Same one-step output_ids delay handling as mix_with_running.
        delta = 0 if self.enable_overlap else -1
        self.prefix_lens = [
            len(r.origin_input_ids) + len(r.output_ids) + delta for r in self.reqs
        ]
        self.extend_lens = [1] * bs
        self.extend_num_tokens = bs
        self.extend_logprob_start_lens = [0] * bs
        self.decoding_reqs = self.reqs
        self.is_prefill_only = False

    def new_tokens_required_next_decode(
        self, selected_indices: Optional[List[int]] = None
    ):
        # 中译：估算「下一步 decode」需要新分配的 KV token 数（按 page 对齐）。无投机解码时，
        #       只有正好落在 page 边界的请求才需新开一页；投机解码走更精确的 _spec_v2 估算。
        page_size = self.token_to_kv_pool_allocator.page_size
        requests = (
            self.reqs
            if selected_indices is None
            else [self.reqs[i] for i in selected_indices]
        )

        if self.spec_algorithm.is_none():
            new_pages = sum(
                1 for r in requests if r.kv.kv_committed_len % page_size == 0
            )
            return new_pages * page_size + num_beam_member_rows(requests)

        return self._new_tokens_required_next_decode_spec_v2(requests, page_size)

    def _new_tokens_required_next_decode_spec_v2(self, requests, page_size):
        """Tight estimate matching eagle_utils.eagle_prepare_for_decode allocation."""
        reserve = get_alloc_reserve_per_decode()
        total = 0
        for r in requests:
            x = max(0, r.kv.kv_committed_len + reserve - r.kv.kv_allocated_len)
            cur = r.kv.kv_allocated_len
            nxt = cur + x
            total += ceil_align(nxt, page_size) - ceil_align(cur, page_size)
        return total

    def check_decode_mem(self, selected_indices: Optional[List[int]] = None):
        # 中译：检查下一步 decode 的 KV 内存是否够用：先按需触发 tree cache 淘汰腾空间，
        """Reclaim evictable tree-cache entries (shortfall only), then report
        whether the next decode step fits in the KV pool."""
        num_tokens = self.new_tokens_required_next_decode(selected_indices)
        evict_from_tree_cache(self.tree_cache, num_tokens)
        return self.token_to_kv_pool_allocator.available_size() >= num_tokens

        # 中译：回撤本批全部请求（释放各自资源），并把 reqs 清空，返回被回撤的请求列表。
    def retract_decode(self) -> Tuple[List[Req], float, List[Req]]:
        """Retract the decoding requests when there is not enough memory."""
        sorted_indices = self._get_decode_retraction_order(self.reqs)
        sorted_indices = beam_retraction_order(sorted_indices, self.reqs)

        retracted_reqs = []
        reqs_to_abort: List[Req] = []
        first_iter = True
        while first_iter or (
            not self.check_decode_mem(selected_indices=sorted_indices)
        ):
            if len(sorted_indices) == 1:
                # Always keep at least one request
                break

            first_iter = False
            idx = sorted_indices.pop()
            req = self.reqs[idx]
            if req.beam_group is not None:
                # Free member rows before release_req: they need the leader's kv
                # info, which still carries the lockstep allocated length.
                req.to_finish = FINISH_ABORT(
                    "Beam search group aborted: KV cache pool is full. Beam "
                    "groups cannot be retracted, so they are aborted instead "
                    "of being requeued.",
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                reqs_to_abort.append(req)
                free_member_rows(
                    req.beam_group,
                    self.req_to_token_pool,
                    self.token_to_kv_pool_allocator,
                )
                # Aborting, so a host backup to resume from would be wasted.
                self.release_req(idx, len(sorted_indices), offload_kv=False)
                continue
            # release memory and don't insert into the tree because we need the space instantly
            if self.release_req(idx, len(sorted_indices)):
                retracted_reqs.append(req)
            else:
                # The retraction host pool could not hold the backup and the
                # device KV is already freed, so the request cannot resume.
                req.to_finish = FINISH_ABORT(
                    "Retraction host KV pool exhausted. Aborting the request.",
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                reqs_to_abort.append(req)
                logger.warning(
                    "retract_decode: aborted request %s, retraction host pool "
                    "exhausted",
                    req.rid,
                )

        if len(sorted_indices) <= 1 and not self.check_decode_mem(
            selected_indices=sorted_indices
        ):
            # Even the last remaining request cannot fit in memory.
            # Instead of crashing the scheduler, gracefully abort it.
            last_idx = sorted_indices.pop()
            last_req = self.reqs[last_idx]
            last_req.to_finish = FINISH_ABORT(
                "Out of memory even after retracting all other requests "
                "in the decode batch. Aborting the last request.",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            reqs_to_abort.append(last_req)
            if last_req.beam_group is not None:
                free_member_rows(
                    last_req.beam_group,
                    self.req_to_token_pool,
                    self.token_to_kv_pool_allocator,
                )
            self.release_req(last_idx, 0, offload_kv=False)
            logger.warning(
                "retract_decode: aborted last request %s due to OOM", last_req.rid
            )

        self.filter_batch(keep_indices=sorted_indices)

        # Reqs in batch are filtered
        new_estimate_ratio = (
            NewTokenRatioTracker.estimate_new_token_ratio_after_retract(self.reqs)
        )

        return retracted_reqs, new_estimate_ratio, reqs_to_abort

    @staticmethod
    def _get_decode_retraction_order(reqs: List[Req]) -> List[int]:
        """Return indices ordered from most-preferred to least-preferred to keep.

        The retraction loop pops from the end of this list, so the least-preferred
        request is retracted first.
        """
        sorted_indices = list(range(len(reqs)))

        # TODO(lsyin): improve retraction policy for radix cache

        def length_key(req: Req) -> Tuple[int, int]:
            return (len(req.output_ids), -len(req.origin_input_ids))

        if get_schedule().retraction_policy == "priority":
            priority_sign = (
                1 if get_schedule().schedule_low_priority_values_first else -1
            )

            def retraction_key(req: Req) -> Tuple[int, int, int]:
                priority = req.priority
                if priority is None:
                    priority = (
                        sys.maxsize
                        if get_schedule().schedule_low_priority_values_first
                        else -sys.maxsize - 1
                    )
                return (priority * (-priority_sign), *length_key(req))

            sorted_indices.sort(
                key=lambda i: retraction_key(reqs[i]),
                reverse=True,
            )
            return sorted_indices

        sorted_indices.sort(
            key=lambda i: length_key(reqs[i]),
            reverse=True,
        )
        return sorted_indices

    def release_req(
        self,
        idx: int,
        remaing_req_count: int,
        offload_kv: bool = True,
    ) -> bool:
        return release_req(
            req=self.reqs[idx],
            remaing_req_count=remaing_req_count,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            hisparse_coordinator=self.hisparse_coordinator,
            offload_kv=offload_kv,
        )

    def prepare_encoder_info_decode(self):
        # Reset the encoder cached status
        # 中译：decode 阶段 encoder 输出已全部缓存，故把每个请求的 encoder_cached 重置为 True。
        self.encoder_cached = [True] * len(self.reqs)

    def prepare_for_idle(self):
        # 中译：准备一个「空转（IDLE）」批次——无任何请求，所有张量置空。常用于 DP attention
        #       下某些 rank 没有真实请求、但仍需参与一次集体 forward 以保持同步。
        self.forward_mode = ForwardMode.IDLE
        self.input_ids = torch.empty(0, dtype=torch.int64, device=self.device)
        self.seq_lens = torch.empty(0, dtype=torch.int64, device=self.device)
        self.seq_lens_cpu = torch.empty(0, dtype=torch.int64)
        self.orig_seq_lens = torch.empty(0, dtype=torch.int32, device=self.device)
        self.out_cache_loc = torch.empty(0, dtype=torch.int64, device=self.device)
        self.req_pool_indices = torch.empty(0, dtype=torch.int64, device=self.device)
        self.req_pool_indices_cpu = torch.empty(0, dtype=torch.int64)
        self.seq_lens_sum = 0
        self.extend_num_tokens = 0
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    def mamba_lazy_prealloc_at_boundary(self, mamba_track_interval: int):
        """Allocate a temporary second ping-pong slot for reqs at a track boundary.

        In lazy mode each request normally holds only 1 ping-pong slot.
        When seq_len hits a track interval boundary, we allocate the
        second slot so the forward pass can write the new tracked state
        there. The old slot is freed after the forward in
        mamba_lazy_post_decode_at_boundary.
        """
        pool = self.req_to_token_pool
        for i, req in enumerate(self.reqs):
            buf = req.mamba_ping_pong_track_buffer
            assert buf is not None
            # Skip reqs not at a track boundary
            if self.seq_lens_cpu[i].item() % mamba_track_interval != 0:
                continue
            other_idx = 1 - req.mamba_next_track_idx
            if buf[other_idx].item() != -1:
                # With overlap the previous forward's post-processing
                # (which frees this slot) hasn't run yet. Skip.
                continue
            if envs.SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL.get():
                new_slot = None
            else:
                # No evict-retry: a transient slot is not worth evicting a
                # cached checkpoint for; on failure tracking degrades in place.
                new_slot = pool.mamba_allocator.alloc(1)
            if new_slot is not None:
                pool.set_mamba_ping_pong_slot(req, other_idx, new_slot[0])
                req.mamba_next_track_idx = other_idx

    def mamba_lazy_spec_prepare(self, mamba_track_interval: int, max_draft_tokens: int):
        """Lazy-mode spec counterpart of mamba_lazy_prealloc_at_boundary.

        A crossing is only *possible* at prepare time (accept length is
        unknown), so ensure the pending slot exists for reqs whose next
        boundary is reachable, WITHOUT swapping mamba_next_track_idx; the
        mask-gated commit writes it only on a real crossing, and
        _mamba_lazy_spec_confirm_crossing promotes it afterwards. The per-req
        scatter position is recorded on the batch and rides the result-queue
        copy (forward isolation restores batch fields).
        """
        pool = self.req_to_token_pool
        track_positions: List[int] = []
        for req in self.reqs:
            buf = req.mamba_ping_pong_track_buffer
            assert buf is not None
            if not mamba_lazy_spec_in_window(
                req, mamba_track_interval, max_draft_tokens
            ):
                # No crossing reachable: the scatter mask stays -1, the
                # position is never written.
                track_positions.append(req.mamba_next_track_idx)
                continue
            other_idx = 1 - req.mamba_next_track_idx
            has_pending = buf[other_idx].item() != -1
            if not has_pending:
                if envs.SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL.get():
                    new_slot = None
                else:
                    # No evict-retry: a transient slot is not worth
                    # evicting a cached checkpoint for.
                    new_slot = pool.mamba_allocator.alloc(1)
                if new_slot is not None:
                    pool.set_mamba_ping_pong_slot(req, other_idx, new_slot[0])
                    has_pending = True
            # On failure the verify scatters in place into the keep slot.
            track_positions.append(
                other_idx if has_pending else req.mamba_next_track_idx
            )
        self.mamba_lazy_spec_track_positions_cpu = track_positions

    def cumulate_penalty_output_tokens(self):
        # Under overlap batch.input_ids is just a placeholder here -- the
        # real token is relayed via future_map and resolved at forward
        # entry. So take the last output token from Req directly
        # (origin_input_ids[-1] on the first decode, before any output).
        last_tokens = [
            req.output_ids[-1] if len(req.output_ids) else req.origin_input_ids[-1]
            for req in self.reqs
        ]
        # Non-blocking H2D so this per-step copy doesn't sync behind the forward.
        # pin_memory (matching the prefill-path tensors) keeps the copy async;
        # is_pin_memory_available falls back to pageable on unsupported devices.
        latest_output_ids = torch.tensor(
            last_tokens,
            dtype=torch.int64,
            pin_memory=is_pin_memory_available(self.device),
        ).to(self.device, non_blocking=True)
        self.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
            latest_output_ids
        )

    def prepare_for_decode(self):
        # 中译：为 DECODE（逐 token 生成）阶段准备本批。要点：
        #       - 投机解码时把准备工作整体交给 draft_input 接管后直接返回；
        #       - 否则：（按需）累计惩罚器输出 token -> alloc_for_decode 分配 1 个 KV 槽/请求
        #         -> 各请求 kv_committed/allocated_len +1、seq_lens +1（overlap 模式用新张量避免竞争）。
        self.forward_mode = ForwardMode.DECODE
        # Decode embeds the last output token via embed_tokens; clear the stale
        # prefill-time tensor so it doesn't leak into ForwardBatch.
        # 中译：decode 阶段通过 embed_tokens 嵌入「上一个输出 token」，清掉 prefill 期残留的
        #       input_embeds，防止它泄漏到 ForwardBatch。
        self.input_embeds = None

        # Clear context parallel metadata - CP is only for prefill, not decode
        # 中译：清除上下文并行（CP）元数据——CP 仅用于 prefill，不用于 decode。
        if hasattr(self, "attn_cp_metadata") and self.attn_cp_metadata is not None:
            self.attn_cp_metadata = None

        if not self.spec_algorithm.is_none():
            # 中译：投机解码下，decode 准备（内存分配、预占、seq_lens 记账）全部交由草稿输入接管。
            # Spec decoding owns decode preparation (allocation, seq-lens bookkeeping).
            from sglang.srt.speculative.spec_utils import spec_prepare_for_decode

            self.mamba_track_mask_cpu = None
            self.mamba_track_mask_next_cpu = None
            self.mamba_decode_batch_idx_cpu = None
            spec_prepare_for_decode(self)
            return

        # Beam member rows ride this decode batch: append them to the row
        # tensors before allocation so alloc_for_decode covers them too.
        strip_beam_tail(self)
        append_beam_tail(self)

        if self.sampling_info.penalizer_orchestrator.is_required:
            # 中译：重叠模式下 batch.input_ids 此刻只是占位符，真正的 token 由 future_map
            self.cumulate_penalty_output_tokens()

        # input_ids is set at end of previous run_batch (placeholder for
        # overlap; next_token_ids cast for non-overlap).
        # 中译：input_ids 在上一次 run_batch 末尾设置（重叠模式下为占位符；非重叠模式下由 next_token_ids 转换而来）。

        if self.model_config.is_encoder_decoder:
            self.prepare_encoder_info_decode()

        # 中译：为每个请求分配 1 个新的 KV 写入位置（decode 每步只生成 1 个 token）。
        # 中译：每个请求 decode 计数 +1，已提交/已分配 KV 长度各 +1。
        # Allocate memory (DSV4-NPU c{4,128}_state alloc lens are computed inside
        # the allocator, triggered from mem_cache/common.py.)
        self.out_cache_loc = alloc_for_decode(self, token_per_req=1)

        for req in self.reqs:
            req.decode_batch_idx += 1

            # 中译：重叠模式下用“新张量”而非原地 +1，避免与已排队等待重叠前向的
        # New-tensor avoids racing model_worker_batch refs queued for
        # overlap forward.
        self.seq_lens = self.seq_lens + 1
        self.seq_lens_cpu = self.seq_lens_cpu + 1
        self.orig_seq_lens = self.orig_seq_lens + 1
        # Sum is recomputed lazily by ForwardBatch.init_new.
        # 中译：seq_lens_sum 置 None，由 ForwardBatch.init_new 懒重算。
        self.seq_lens_sum = None

        if self.hisparse_coordinator is not None:
            self.hisparse_coordinator.map_last_loc_to_buffer(
                self.seq_lens,
                self.out_cache_loc,
                self.req_pool_indices,
                self.seq_lens_cpu,
                self.req_pool_indices_cpu,
            )

        if mamba_extra_buffer_enabled():
            mamba_track_interval = mamba_track_grid(self.tree_cache.page_size)

            if len(self.reqs) == 0:
                self.mamba_track_indices = torch.empty(
                    (0,), dtype=torch.int64, device=self.device
                )
                self.mamba_track_buffer_indices = []
            else:
                if mamba_extra_buffer_lazy_enabled():
                    self.mamba_lazy_prealloc_at_boundary(mamba_track_interval)
                set_mamba_track_indices_from_reqs(self)

            track_remainders_cpu = self.seq_lens_cpu % mamba_track_interval
            track_mask_cpu = track_remainders_cpu == 0
            self.mamba_track_mask_cpu = track_mask_cpu.tolist()
            self.mamba_track_mask_next_cpu = (
                (track_remainders_cpu == mamba_track_interval - 1).tolist()
                if self.enable_overlap
                else None
            )
            # ScheduleBatch.copy() snapshots the list of requests, but the Req
            # objects remain shared. The next overlapped decode can therefore
            # advance their counters before this batch's result is processed.
            self.mamba_decode_batch_idx_cpu = [
                req.decode_batch_idx for req in self.reqs
            ]
            # async H2D
            self.mamba_track_mask = track_mask_cpu.pin_memory().to(
                device=self.device, non_blocking=True
            )

    def filter_batch(
        self,
        chunked_req_to_exclude: Optional[Union[Req, List[Req]]] = None,
        keep_indices: Optional[List[int]] = None,
    ):
        # 中译：从批中「过滤掉」部分请求（默认移除已结束的与指定排除的 chunked 请求），
        strip_beam_tail(self)
        if keep_indices is None:
            if isinstance(chunked_req_to_exclude, Req):
                chunked_req_to_exclude = [chunked_req_to_exclude]
            elif chunked_req_to_exclude is None:
                chunked_req_to_exclude = []
            keep_indices = [
                i
                for i in range(len(self.reqs))
                if not self.reqs[i].finished()
                and self.reqs[i] not in chunked_req_to_exclude
            ]

        if keep_indices is None or len(keep_indices) == 0:
            # Filter out all requests. Stale tensors are left as-is: is_empty()
            # keys off reqs, so callers drop the batch before a forward reads them.
            self.reqs = []
            self.return_hidden_states = False
            self.return_hidden_states_mode = CaptureHiddenMode.NULL
            return

        if len(keep_indices) == len(self.reqs):
            # No need to filter
            return

        # 中译：把保留索引搬上 device（锁页拷贝），用于对各 GPU 张量做 index_select 裁剪。
        keep_indices_device = torch.tensor(
            keep_indices,
            dtype=torch.int64,
            pin_memory=is_pin_memory_available(self.device),
        ).to(self.device, non_blocking=True)

        if self.model_config.is_encoder_decoder:
            self.encoder_lens = self.encoder_lens[keep_indices_device]
            self.encoder_lens_cpu = [self.encoder_lens_cpu[i] for i in keep_indices]

        self.reqs = [self.reqs[i] for i in keep_indices]
        if self.multimodal_inputs is not None:
            self.multimodal_inputs = [self.multimodal_inputs[i] for i in keep_indices]
        self.req_pool_indices = self.req_pool_indices[keep_indices_device]
        self.req_pool_indices_cpu = self.req_pool_indices_cpu[keep_indices]
        self.seq_lens = self.seq_lens[keep_indices_device]
        self.orig_seq_lens = self.orig_seq_lens[keep_indices_device]
        self.out_cache_loc = None
        # Sum is recomputed lazily by ForwardBatch.init_new.
        # 中译：seq_lens_sum 置 None，由 ForwardBatch.init_new 懒重算。
        self.seq_lens_sum = None

        if self.input_ids is not None:
            self.input_ids = self.input_ids[keep_indices_device]
        # Optional under no-verify-sync; resolve_seq_lens repopulates before forward.
        # 中译：seq_lens_cpu 在 no-verify-sync 下可能为空；前向前 resolve_seq_lens 会重新填充。
        if self.seq_lens_cpu is not None:
            self.seq_lens_cpu = self.seq_lens_cpu[keep_indices]

        self.mamba_track_indices = None
        self.mamba_track_buffer_indices = None
        self.mamba_track_mask = None
        self.mamba_track_seqlens = None
        self.mamba_track_mask_cpu = None
        self.mamba_track_mask_next_cpu = None
        self.mamba_decode_batch_idx_cpu = None
        self.mamba_lazy_spec_track_positions_cpu = None
        self.mamba_cow_src_indices = None
        self.mamba_cow_dst_indices = None
        self.mamba_clear_indices = None
        self.return_logprob = any(req.return_logprob for req in self.reqs)
        if self.return_logprob:
            self.top_logprobs_nums = [self.top_logprobs_nums[i] for i in keep_indices]
            self.token_ids_logprobs = [self.token_ids_logprobs[i] for i in keep_indices]
        else:
            self.top_logprobs_nums = None
            self.token_ids_logprobs = None

        self.has_grammar = any(req.grammar for req in self.reqs)
        self.return_hidden_states_mode = get_batch_return_hidden_states_mode(self.reqs)
        self.return_hidden_states = self.return_hidden_states_mode.need_capture()

        self.sampling_info.filter_batch(keep_indices, keep_indices_device)
        if self.spec_info:
            self.spec_info.filter_batch(
                new_indices=keep_indices_device,
                new_indices_cpu=keep_indices,
            )

    def merge_batch(self, other: ScheduleBatch):
        # 中译：把另一个批次 other 合并进本批（常见于 prefill 完成后并入正在 running 的 decode 批）。
        strip_beam_tail(self)
        strip_beam_tail(other)
        # Penalizer orchestrator must be merged before Batch.reqs is merged. This is because
        # orchestrator.merge() depends on Batch.reqs during preparation of each penalizers, so it
        # needs to be called with pre-merged Batch.reqs.
        # 中译：必须「先」合并采样信息（惩罚器编排器），「再」合并 reqs——因为 orchestrator.merge()
        #       在准备各惩罚器时依赖「合并前」的 self.reqs。
        self.sampling_info.merge_batch(other.sampling_info)

        # Encoder-decoder infos
        if self.model_config.is_encoder_decoder:
            self.encoder_lens = torch.cat([self.encoder_lens, other.encoder_lens])
            self.encoder_lens_cpu = self.encoder_lens_cpu + other.encoder_lens_cpu
        self.req_pool_indices = torch.cat(
            [self.req_pool_indices, other.req_pool_indices]
        )
        self.req_pool_indices_cpu = torch.cat(
            [self.req_pool_indices_cpu, other.req_pool_indices_cpu]
        )
        self.seq_lens = torch.cat([self.seq_lens, other.seq_lens])
        self.orig_seq_lens = torch.cat([self.orig_seq_lens, other.orig_seq_lens])
        self.out_cache_loc = None
        # Sum is recomputed lazily by ForwardBatch.init_new.
        # 中译：seq_lens_sum 置 None，由 ForwardBatch.init_new 懒重算。
        self.seq_lens_sum = None
        # Cat only when both sides hold a real token tensor; otherwise drop to
        # None and let resolve_forward_inputs rebuild from the merged
        # req_pool_indices. Mismatch arises e.g. with spec_v1, which keeps its
        # tensor while a relay-staged side is None -- there the worker rebuilds.
        # 中译：只有两边都持有真实 token 张量时才拼接；否则置 None，让
        #       resolve_forward_inputs 从合并后的 req_pool_indices 重建。不一致例如 spec_v1
        #       会保留自己的张量而中继暂存一侧为 None——那种情况由 worker 重建。
        if self.input_ids is not None and other.input_ids is not None:
            self.input_ids = torch.cat([self.input_ids, other.input_ids])
        else:
            self.input_ids = None
        # Optional under no-verify-sync; drop the mirror if either side absent.
        # 中译：seq_lens_cpu 在 no-verify-sync 下可选；只要有一侧缺失就丢弃这份 CPU 镜像。
        if self.seq_lens_cpu is None or other.seq_lens_cpu is None:
            self.seq_lens_cpu = None
        else:
            self.seq_lens_cpu = torch.cat([self.seq_lens_cpu, other.seq_lens_cpu])
        self.mamba_track_indices = None
        self.mamba_track_buffer_indices = None
        self.mamba_track_mask = None
        self.mamba_track_seqlens = None
        self.mamba_track_mask_cpu = None
        self.mamba_track_mask_next_cpu = None
        self.mamba_decode_batch_idx_cpu = None
        self.mamba_lazy_spec_track_positions_cpu = None
        if self.return_logprob and other.return_logprob:
            self.top_logprobs_nums = self.top_logprobs_nums + other.top_logprobs_nums
            self.token_ids_logprobs = self.token_ids_logprobs + other.token_ids_logprobs
        elif self.return_logprob:
            self.top_logprobs_nums = self.top_logprobs_nums + [0] * len(other.reqs)
            self.token_ids_logprobs = self.token_ids_logprobs + [None] * len(other.reqs)
        elif other.return_logprob:
            self.top_logprobs_nums = [0] * len(self.reqs) + other.top_logprobs_nums
            self.token_ids_logprobs = [None] * len(self.reqs) + other.token_ids_logprobs
        self.reqs = self.reqs + other.reqs
        if self.multimodal_inputs is not None:
            self.multimodal_inputs = self.multimodal_inputs + other.multimodal_inputs

        self.return_logprob = self.return_logprob or other.return_logprob
        self.has_grammar = self.has_grammar or other.has_grammar
        self.return_hidden_states_mode = max(
            self.return_hidden_states_mode, other.return_hidden_states_mode
        )
        self.return_hidden_states = self.return_hidden_states_mode.need_capture()
        self.is_prefill_only = self.is_prefill_only and other.is_prefill_only

        if self.spec_info:
            self.spec_info.merge_batch(other.spec_info)

    def copy(self):
        # 中译：制作一个「轻量快照」，只复制 process_batch_result 会用到的字段。reqs 用浅拷贝列表，
        #       使原批后续的 filter_batch/merge_batch 等原地修改不会破坏这份快照（overlap 调度需要）。
        # Only contain fields that will be used by process_batch_result.
        # 中译：只复制 process_batch_result 会用到的字段；reqs 用浅拷贝列表，使原批后续的
        # Shallow-copy the reqs list as a defensive snapshot. filter_batch and
        # merge_batch historically mutated the list in place; they now rebind
        # new lists, but the slice stays so this snapshot never aliases the
        # original.
        return ScheduleBatch(
            reqs=self.reqs[:],
            extend_lens=self.extend_lens,
            prefix_lens=self.prefix_lens,
            req_to_token_pool=self.req_to_token_pool,
            req_pool_indices=self.req_pool_indices,
            model_config=self.model_config,
            forward_mode=self.forward_mode,
            out_cache_loc=self.out_cache_loc,
            return_logprob=self.return_logprob,
            has_grammar=self.has_grammar,
            return_hidden_states=self.return_hidden_states,
            return_hidden_states_mode=self.return_hidden_states_mode,
            decoding_reqs=self.decoding_reqs,
            spec_algorithm=self.spec_algorithm,
            spec_info=self.spec_info,
            global_num_tokens=self.global_num_tokens,
            global_num_tokens_for_logprob=self.global_num_tokens_for_logprob,
            can_run_decode_cuda_graph=self.can_run_decode_cuda_graph,
            can_run_dp_prefill_cuda_graph=self.can_run_dp_prefill_cuda_graph,
            is_extend_in_batch=self.is_extend_in_batch,
            is_prefill_only=self.is_prefill_only,
            seq_lens_cpu=self.seq_lens_cpu,
            enable_overlap=self.enable_overlap,
            mamba_track_indices=self.mamba_track_indices,
            mamba_track_buffer_indices=self.mamba_track_buffer_indices,
            mamba_track_mask=self.mamba_track_mask,
            mamba_track_seqlens=self.mamba_track_seqlens,
            mamba_track_mask_cpu=self.mamba_track_mask_cpu,
            mamba_track_mask_next_cpu=self.mamba_track_mask_next_cpu,
            mamba_decode_batch_idx_cpu=self.mamba_decode_batch_idx_cpu,
            mamba_lazy_spec_track_positions_cpu=self.mamba_lazy_spec_track_positions_cpu,
            dp_cooperation_info=self.dp_cooperation_info,
            prefill_stats=self.prefill_stats,
            fpm_start_time=self.fpm_start_time,
            forward_iter=self.forward_iter,
            launch_ts=self.launch_ts,
            after_idle_gap=self.after_idle_gap,
            extend_num_tokens=self.extend_num_tokens,
        )

    def maybe_evict_swa(self):
        # 中译：滑动窗口注意力（SWA）下，按一定间隔淘汰「滑出窗口」的 KV 槽位以省内存。
        #       decode 阶段每隔 eviction_interval 步淘汰一次；窗口越过后还会把 prefill 期的 SWA 树锁
        #       由「受保护」转为「可淘汰」。chunk cache 的 extend 阶段也会相应淘汰。
        if self.tree_cache.supports_swa():
            sliding_window_size = self.tree_cache.sliding_window_size

            release_leaf_lock = (
                envs.SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW.get()
                and hasattr(self.tree_cache, "dec_swa_lock_only")
            )

            # 中译：eviction_interval：在“SWA token 浪费”与“淘汰开销”之间权衡的间隔。
            eviction_interval = max(1, envs.SGLANG_SWA_EVICTION_INTERVAL.get())
            self.token_to_kv_pool_allocator.free_group_begin()
            for idx, req in enumerate(self.reqs):
                if self.forward_mode.is_decode():
                    # We set evict_swa condition here with two reasons:
                    # 1. In overlap scheduler, we cannot evict swa when req.decode_batch_idx == 0 since the prev extend batch is still running.
                    # 中译：这里的淘汰条件出于两个原因：
                    # 2. Evict only once >= eviction_interval tokens have slid
                    # out of the window, amortizing eviction work while keeping
                    # each request's overshoot within the interval the pool
                    # budget reserves. Gating on accumulated tokens (rather
                    # than an iteration-counter phase) cannot starve because
                    # seqlen progress is monotonic per KV handle.
                    if (
                        req.decode_batch_idx >= 1
                        and req.kv.is_held
                        and req.seqlen - 1 - sliding_window_size
                        >= req.kv.swa_evicted_seqlen + eviction_interval
                    ):
                        self._evict_swa(req, req.seqlen - 1)

                    # Once the decode position has moved past the sliding window,
                    # the SWA portion of the prefill-time tree lock is no longer
                    # needed by this request. Convert it from protected to
                    # evictable so SWA LRU can reclaim it under pressure.
                    # 中译：一旦 decode 位置越过滑动窗口，本请求就不再需要 prefill 期加的 SWA 部分树锁；
                    #       把它从“受保护”转为“可淘汰”，以便内存吃紧时 SWA LRU 能回收。
                    if (
                        release_leaf_lock
                        and not req.swa_prefix_lock_released
                        and req.swa_uuid_for_lock is not None
                        and req.last_node is not None
                        and req.decode_batch_idx >= sliding_window_size
                    ):
                        self.tree_cache.dec_swa_lock_only(
                            req.last_node,
                            req.swa_uuid_for_lock,
                            skip_lock_node_ids=req.skip_lock_node_ids,
                        )
                        req.swa_prefix_lock_released = True
                elif self.forward_mode.is_extend() and self.tree_cache.is_chunk_cache():
                    pre_len = self.prefix_lens[idx]
                    if self.enable_overlap:
                        # In chunked prefill case, when the second extend batch is scheduling, the first extend batch is still running, so we cannot evict swa tokens
                        # 中译：分块 prefill 时，第二个 extend 批正在调度而第一个 extend 批还在跑，
                        #       此时不能淘汰 SWA token。
                        if req.extend_batch_idx < 2:
                            continue
                        else:
                            pre_len = (
                                pre_len - get_schedule().chunked_prefill_size
                                if get_schedule().chunked_prefill_size > 0
                                else pre_len
                            )
                            self._evict_swa(req, pre_len)
                    else:
                        self._evict_swa(req, pre_len)
            self.token_to_kv_pool_allocator.free_group_end()

    def _evict_swa(self, req: Req, pre_len: int):
        assert self.tree_cache.supports_swa(), "prefix cache must support swa"
        free_swa_out_of_window_slots(
            req,
            pre_len,
            sliding_window_size=self.tree_cache.sliding_window_size,
            page_size=self.tree_cache.page_size,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            is_chunk_cache=self.tree_cache.is_chunk_cache(),
            retain_floor=self.tree_cache.swa_retain_floor(req),
        )

    def __str__(self):
        return (
            f"ScheduleBatch(forward_mode={self.forward_mode.name if self.forward_mode else 'None'}, "
            f"#req={(len(self.reqs))})"
        )


class NextBatchPlan(msgspec.Struct):
    batch_to_run: Optional[ScheduleBatch]
    running_batch: ScheduleBatch
