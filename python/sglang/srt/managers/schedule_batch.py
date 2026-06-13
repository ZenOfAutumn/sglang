from __future__ import annotations

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.utils.common import ceil_align, is_pin_memory_available

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
存储请求（request）与批次（batch）的信息。

一个批次的数据结构流转如下：

ScheduleBatch -> ModelWorkerBatch -> ForwardBatch

- ScheduleBatch 由 `scheduler.py::Scheduler` 管理。
  它包含高层调度数据，大部分数据位于 CPU 上。
- ModelWorkerBatch 由 `tp_worker.py::TpModelWorker` 管理。
  它是 `ScheduleBatch` 的子集，仅包含与 GPU 上模型前向相关的数据，
  会从 CPU 调度器转换传递给 GPU 模型运行器。
- ForwardBatch 由 `model_runner.py::ModelRunner` 管理。
  它包含底层张量数据，大部分是 GPU 张量。

TODO(lmzheng)：ModelWorkerBatch 看起来有些冗余，未来考虑移除。
"""

import copy
import dataclasses
import logging
import re
from concurrent.futures import Future
from enum import Enum, auto
from functools import lru_cache
from http import HTTPStatus
from itertools import chain
from typing import TYPE_CHECKING, List, Optional, Set, Tuple, Union

import numpy as np
import torch

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.disaggregation.base import BaseKVSender
from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    ScheduleBatchDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_rank
from sglang.srt.dllm.mixin.req import ReqDllmMixin
from sglang.srt.environ import envs
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchPrefixParams
from sglang.srt.mem_cache.common import (
    alloc_for_decode,
    alloc_for_extend,
    evict_from_tree_cache,
    release_kv_cache,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
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
from sglang.srt.server_args import ServerArgs, get_global_server_args
from sglang.srt.utils import flatten_nested_list
from sglang.srt.utils.cuda_ipc_transport_utils import CudaIpcTensorTransportProxy

if TYPE_CHECKING:
    from typing import Any, Dict

    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.managers.session_controller import Session
    from sglang.srt.observability.scheduler_metrics_mixin import PrefillStats
    from sglang.srt.speculative.eagle_info import EagleDraftInput
    from sglang.srt.speculative.spec_info import SpecInput, SpeculativeAlgorithm

# 增量反分词（detokenization）的初始回看偏移量：生成时多回看几个 token，以正确拼接多字节字符。
INIT_INCREMENTAL_DETOKENIZATION_OFFSET = 5

# 作为多模态（MM）pad 值的基准偏移常量。
# 确保 pad_values 不会与有效的文本 token ID 重叠。
MM_PAD_SHIFT_VALUE = 1_000_000

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def sanity_check_mm_pad_shift_value(vocab_size: int) -> None:
    """校验模型词表大小不超过 MM_PAD_SHIFT_VALUE，防止多模态 pad 值与有效 token ID 重叠。"""
    if vocab_size > MM_PAD_SHIFT_VALUE:
        raise ValueError(
            f"Model vocab_size ({vocab_size}) exceeds MM_PAD_SHIFT_VALUE ({MM_PAD_SHIFT_VALUE}). "
            f"MM pad_values may overlap with valid token IDs. "
            f"Please increase MM_PAD_SHIFT_VALUE in schedule_batch.py."
        )


def _compute_pad_value(hash: int) -> int:
    """根据哈希值计算多模态 pad 值（叠加基准偏移，限制在 30 位范围内）。"""
    return MM_PAD_SHIFT_VALUE + (hash % (1 << 30))


class BaseFinishReason:
    """请求结束原因的基类；is_error 标记是否为错误类结束。子类需实现 to_json。"""

    def __init__(self, is_error: bool = False):
        self.is_error = is_error

    def to_json(self):
        raise NotImplementedError()


class FINISH_MATCHED_TOKEN(BaseFinishReason):
    """因命中停止 token（EOS 或自定义 stop token）而正常结束。"""

    def __init__(self, matched: Union[int, List[int]]):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # 与 OpenAI API 的返回值保持一致
            "matched": self.matched,
        }


class FINISH_MATCHED_STR(BaseFinishReason):
    """因命中停止字符串（stop string）而正常结束。"""

    def __init__(self, matched: str):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # 与 OpenAI API 的返回值保持一致
            "matched": self.matched,
        }


class FINISHED_MATCHED_REGEX(BaseFinishReason):
    """因命中停止正则（stop regex）而正常结束。"""

    def __init__(self, matched: str):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # 与 OpenAI API 的返回值保持一致
            "matched": self.matched,
        }


class FINISH_LENGTH(BaseFinishReason):
    """因达到最大生成长度而结束。"""

    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def to_json(self):
        return {
            "type": "length",  # 与 OpenAI API 的返回值保持一致
            "length": self.length,
        }


class FINISH_ABORT(BaseFinishReason):
    """因被中止（超时、超长、无效请求等）而结束，属于错误类结束。"""

    def __init__(self, message=None, status_code=None, err_type=None):
        super().__init__(is_error=True)
        self.message = message or "Aborted"  # 错误消息
        self.status_code = status_code  # 对应的 HTTP 状态码
        self.err_type = err_type  # 错误类型

    def to_json(self):
        return {
            "type": "abort",
            "message": self.message,
            "status_code": self.status_code,
            "err_type": self.err_type,
        }


class Modality(Enum):
    """模态类型枚举：图像 / 视频 / 音频。"""

    IMAGE = auto()
    VIDEO = auto()
    AUDIO = auto()

    @staticmethod
    def from_str(modality_str: str):
        """从字符串（大小写不敏感）解析模态枚举；无效时报错。"""
        try:
            return Modality[modality_str.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid modality string: {modality_str}. Valid modalities are: {[m.name for m in Modality]}"
            )

    @staticmethod
    def all():
        """返回所有模态类型的列表。"""
        return [Modality.IMAGE, Modality.VIDEO, Modality.AUDIO]


class MultimodalInputFormat(Enum):
    """多模态输入的数据形式：

    - NORMAL：原始输入
    - PROCESSOR_OUTPUT：经处理器处理后的输出（如 pixel_values）
    - PRECOMPUTED_EMBEDDING：预计算好的编码器 embedding
    """

    NORMAL = auto()
    PROCESSOR_OUTPUT = auto()
    PRECOMPUTED_EMBEDDING = auto()


@dataclasses.dataclass
class MultimodalDataItem:
    """
    一个 MultimodalDataItem 代表单个多模态输入（一张图、一段视频或一段音频）。
    例如有 3 张图和 1 段音频时，会有 4 个 MultimodalDataItem。

    每个 item 有自己的 hash 与 pad_value，从而支持按图粒度的 RadixAttention 缓存。

    将通用字段放在前面，模型特定字段放在 model_specific_data 中。
    """

    modality: Modality  # 模态类型
    hash: int = None  # 特征的哈希值（用于缓存命中判断）
    pad_value: int = None  # 占位 token 的 pad 值（由 hash 计算得出）
    offsets: Optional[list] = None  # 该多模态输入在 token 序列中的位置偏移

    format: MultimodalInputFormat = MultimodalInputFormat.NORMAL  # 数据形式

    # 处理器返回的原始特征，如 pixel_values 或 audio_features。
    feature: Union[torch.Tensor, np.ndarray] = None
    # 预计算好的 embedding，作为最终的编码器 embedding 传入。
    # feature 与 precomputed_embeddings 有且仅有一个为空。
    precomputed_embeddings: Optional[Union[torch.Tensor, np.ndarray]] = None

    # 模型特定的数据，以字典形式存储。
    model_specific_data: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __getattr__(self, name: str):
        # 未定义的属性访问回退到 model_specific_data 字典中查找。
        if (
            "model_specific_data" in self.__dict__
            and name in self.__dict__["model_specific_data"]
        ):
            return self.__dict__["model_specific_data"][name]
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def __setitem__(self, key: str, value: Any):
        # 已有字段直接赋值，否则存入 model_specific_data。
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self.model_specific_data[key] = value

    def set(self, key: str, value: Any):
        """设置字段值（__setitem__ 的别名）。"""
        self.__setitem__(key, value)

    @staticmethod
    def is_empty_list(l):
        """判断一个（可嵌套的）列表是否为空（展平后没有非 None 元素）。"""
        if l is None:
            return True
        return len([item for item in flatten_nested_list(l) if item is not None]) == 0

    def set_pad_value(self):
        """在首次对数据哈希后设置 pad 值。同一多模态输入哈希相同，便于缓存复用。"""
        if self.pad_value is not None:
            return

        from sglang.srt.managers.mm_utils import hash_feature

        if envs.SGLANG_MM_SKIP_COMPUTE_HASH.get():
            import uuid

            self.hash = uuid.uuid4().int
            self.pad_value = _compute_pad_value(self.hash)
            return
        if self.hash is None:
            if self.feature is not None:
                hashed_feature = self.feature
            else:
                hashed_feature = self.precomputed_embeddings
            self.hash = hash_feature(hashed_feature)
        assert self.hash is not None
        self.pad_value = _compute_pad_value(self.hash)

    def is_modality(self, modality: Modality) -> bool:
        """判断是否为指定模态。"""
        return self.modality == modality

    def is_audio(self):
        """是否为音频。"""
        return self.modality == Modality.AUDIO

    def is_image(self):
        """是否为图像。"""
        return self.modality == Modality.IMAGE

    def is_video(self):
        """是否为视频。"""
        return self.modality == Modality.VIDEO

    def is_valid(self) -> bool:
        """是否为有效模态（图/视频/音频之一）。"""
        return self.is_image() or self.is_video() or self.is_audio()

    def validate(self):
        ...
        # TODO

    def is_precomputed_embedding(self):
        """是否为预计算 embedding 形式。"""
        return self.format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING

    @staticmethod
    def from_dict(obj: dict):
        """从字典构造 MultimodalDataItem。"""
        kwargs = dict(obj)
        modality = kwargs.pop("modality")
        if isinstance(modality, str):
            modality = Modality[modality]
        ret = MultimodalDataItem(modality=modality, **kwargs)
        ret.validate()
        return ret

    def reconstruct(self):
        """跨进程传输后重建张量：将 CUDA IPC 代理张量在当前设备上还原为真实张量。"""
        if not isinstance(self.feature, CudaIpcTensorTransportProxy):
            return

        reconstruct_device = torch.cuda.current_device()
        if isinstance(self.feature, CudaIpcTensorTransportProxy):
            self.feature = self.feature.reconstruct_on_target_device(reconstruct_device)
        if isinstance(self.precomputed_embeddings, CudaIpcTensorTransportProxy):
            self.precomputed_embeddings = (
                self.precomputed_embeddings.reconstruct_on_target_device(
                    reconstruct_device
                )
            )
        for extra_key in self.model_specific_data:
            if isinstance(
                self.model_specific_data[extra_key], CudaIpcTensorTransportProxy
            ):
                extra_data = self.model_specific_data[
                    extra_key
                ].reconstruct_on_target_device(reconstruct_device)
                self.model_specific_data[extra_key] = extra_data


@dataclasses.dataclass
class MultimodalInputs:
    """一条请求的全部多模态相关输入（聚合多个 MultimodalDataItem 及各类特殊 token id）。"""

    # 多模态数据项列表。
    mm_items: List[MultimodalDataItem]
    image_pad_len: Optional[list] = None  # 每张图的占位长度
    num_image_tokens: Optional[int] = None  # 图像 token 总数

    # 图像相关的特殊 token id。
    im_token_id: Optional[int] = None
    im_start_id: Optional[int] = None
    im_end_id: Optional[int] = None
    slice_start_id: Optional[int] = None
    slice_end_id: Optional[int] = None

    # 视频相关的特殊 token id。
    video_token_id: Optional[int] = None

    # 音频相关的特殊 token id。
    audio_token_id: Optional[int] = None
    audio_start_id: Optional[int] = None
    audio_end_id: Optional[int] = None

    # Qwen2-VL 相关：M-RoPE 位置编码与位置增量。
    mrope_positions: Optional[torch.Tensor] = None
    mrope_position_delta: Optional[torch.Tensor] = None
    mrope_position_delta_repeated_cache: Optional[torch.Tensor] = None

    def release_features(self):
        """释放特征张量以释放 GPU 显存。"""
        for item in self.mm_items:
            item.feature = None

    @staticmethod
    def from_dict(obj: dict):
        """从字典构造 MultimodalInputs：重建张量、过滤无效项、计算 pad 值并填充可选字段。"""
        mm_items = obj["mm_items"]
        for mm_item in mm_items:
            mm_item.reconstruct()

        ret = MultimodalInputs(
            mm_items=mm_items,
        )

        assert isinstance(ret.mm_items, list)
        ret.mm_items = [item for item in ret.mm_items if item.is_valid()]

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
            for item in ret.mm_items:
                if item.feature is not None:
                    if isinstance(item.feature, torch.Tensor):
                        item.feature = try_add_to_buffer(item.feature)

        for item in ret.mm_items:
            item.set_pad_value()

        if envs.SGLANG_MM_BUFFER_SIZE_MB.get() > 0:
            for item in ret.mm_items:
                if item.feature is not None:
                    item.feature = item.feature.to("cpu", non_blocking=True)

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
        ]
        for arg in optional_args:
            if arg in obj:
                setattr(ret, arg, obj[arg])

        return ret

    def contains_image_inputs(self) -> bool:
        """是否含图像输入。"""
        return any(item.is_image() for item in self.mm_items)

    def contains_video_inputs(self) -> bool:
        """是否含视频输入。"""
        return any(item.is_video() for item in self.mm_items)

    def contains_audio_inputs(self) -> bool:
        """是否含音频输入。"""
        return any(item.is_audio() for item in self.mm_items)

    def contains_mm_input(self) -> bool:
        """是否含任意有效多模态输入。"""
        return any(True for item in self.mm_items if item.is_valid())

    def merge(self, other: MultimodalInputs):
        """当请求被合并时，合并其多模态输入（mm_items、占位长度与 M-RoPE 位置）。"""

        # 需要合并（拼接）的字段。
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


class Req(ReqDllmMixin):
    """一条请求的输入与输出状态。

    Req 是调度器中最核心的单位，记录一个请求从输入、prefill、decode 到结束的全部状态：
    输入/输出 token、采样参数、KV 缓存与显存池索引、前缀匹配信息、logprob、多模态输入、
    结束判定、流式输出偏移、PD 分离的 bootstrap 信息等。
    """

    def __init__(
        self,
        rid: str,
        origin_input_text: str,
        origin_input_ids: List[int],
        sampling_params: SamplingParams,
        return_logprob: bool = False,
        top_logprobs_num: int = 0,
        dllm_config: Optional[DllmConfig] = None,
        token_ids_logprob: List[int] = None,
        stream: bool = False,
        origin_input_ids_unpadded: Optional[Tuple[int]] = None,
        lora_id: Optional[str] = None,
        input_embeds: Optional[List[List[float]]] = None,
        token_type_ids: List[int] = None,
        session: Optional[Session] = None,
        custom_logit_processor: Optional[str] = None,
        require_reasoning: bool = False,
        return_hidden_states: bool = False,
        return_routed_experts: bool = False,
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
    ):
        # 输入与输出信息
        self.rid = rid  # 请求唯一标识（request id）
        self.origin_input_text = origin_input_text  # 原始输入文本
        self.origin_input_ids_unpadded = (
            origin_input_ids_unpadded
            if origin_input_ids_unpadded
            else origin_input_ids  # 图像 padding 之前的输入 id
        )
        self.origin_input_ids = origin_input_ids  # 原始输入 token id
        # 每个 decode 阶段的输出 id。
        self.output_ids = []
        # fill_ids = origin_input_ids + output_ids，分块时会更新。
        self.fill_ids = []
        self.session = session  # 所属会话（多轮对话场景）
        self.input_embeds = input_embeds  # 直接传入的输入 embedding（如有）

        # 请求级别的显存管理：已提交/已分配的 KV 长度与释放标记。
        self.kv_committed_len = 0
        self.kv_allocated_len = 0
        self.kv_committed_freed = False
        self.kv_overallocated_freed = False

        # 用于 cross-encoder 模型。
        self.token_type_ids = token_type_ids

        # SWA 缓存中已被移除的 KV 长度。
        # SWA KV 缓存的驱逐行为因缓存类型而异：
        # - Radix 缓存：[cache_protected_len, swa_evicted_seqlen) 范围的 KV 在
        #   `ScheduleBatch.maybe_evict_swa` 中手动释放；[0, cache_protected_len) 范围的 KV 在 radix 缓存驱逐时释放。
        # - Chunk 缓存：[0, swa_evicted_seqlen) 范围的 KV 在 `ScheduleBatch.maybe_evict_swa` 中手动释放。
        self.swa_evicted_seqlen = 0

        # 该请求在 extend / decode 批次中的索引。
        self.extend_batch_idx = 0
        self.decode_batch_idx = 0

        # 用于多 HTTP worker 场景的 IPC 标识。
        self.http_worker_ipc = http_worker_ipc

        # 是否要求推理（仅适用于混合推理模型）。
        self.require_reasoning = require_reasoning

        # 采样信息。若 custom_params 为字典，复制一份并注入本请求引用（供自定义 logit 处理器使用）。
        if isinstance(sampling_params.custom_params, dict):
            sampling_params = copy.copy(sampling_params)
            sampling_params.custom_params = sampling_params.custom_params | {
                "__req__": self
            }
        self.sampling_params = sampling_params
        self.custom_logit_processor = custom_logit_processor
        self.return_hidden_states = return_hidden_states

        # 用于对请求分类的额外 key（如 cache_salt）。
        if lora_id is not None:
            extra_key = (
                extra_key or ""
            ) + lora_id  # 把 lora_id 拼接到 extra key 上

        self.extra_key = extra_key
        self.lora_id = lora_id
        self.routing_key = routing_key

        # 显存池信息。
        self.req_pool_idx: Optional[int] = None  # 请求在 req_to_token_pool 中的索引
        self.mamba_pool_idx: Optional[torch.Tensor] = None  # mamba 状态池索引，形状 (1)
        self.mamba_ping_pong_track_buffer: Optional[torch.Tensor] = None  # ping-pong 跟踪缓冲，形状 (2)
        self.mamba_next_track_idx: Optional[int] = None  # 下一个跟踪位（0 或 1）
        self.mamba_last_track_seqlen: Optional[int] = (
            None  # 最后一次缓存的 mamba 状态序列长度
        )
        # 用于跟踪 mamba 状态的分支点序列长度。若由前缀匹配给出，
        # 则为本次 prefill 在 ping-pong 缓冲中被跟踪的序列长度。
        self.mamba_branching_seqlen: Optional[int] = None

        # 结束判定相关。
        self.tokenizer = None
        self.finished_reason: Optional[BaseFinishReason] = None  # 结束原因
        # 结束位置（在 output_ids 中），在投机解码下检查停止条件时使用。
        self.finished_len = None
        # 该请求是否已完成输出。
        self.finished_output = None
        # 若需在事件循环中途中止请求，应设置 to_finish 而不是直接设置 finished_reason。
        # 注意：绝不能在中途直接设置 finished_reason，否则请求会被过滤掉且永远不会响应。
        self.to_finish: Optional[BaseFinishReason] = None
        self.stream = stream  # 是否流式输出
        self.eos_token_ids = eos_token_ids  # 结束 token id 集合
        self.vocab_size = vocab_size  # 词表大小
        self.priority = priority  # 请求优先级（优先级调度时使用）

        # 用于增量解码（下图展示 surr_offset / read_offset / 最后一个 token 的关系）
        # ----- | --------- read_ids -------|
        # ----- |   surr_ids  |
        # xxxxx | xxxxxxxxxxx | xxxxxxxxxxx |
        # ----- ^ ----------- ^ ----------- ^
        # ----- 1 ----------- 2 ----------- 3
        # 1: surr_offset
        # 2: read_offset
        # 3: last token
        self.surr_offset = None  # 环绕偏移，用于对抗增量反分词的清理算法
        self.read_offset = None
        self.decoded_text = ""  # 已解码的文本

        # 多模态输入。
        self.multimodal_inputs: Optional[MultimodalInputs] = None

        # 前缀信息。
        # 共享前缀对应的 KV 缓存索引。
        self.prefix_indices: torch.Tensor = torch.empty((0,), dtype=torch.int64)
        # 本次 prefill 需要运行的 token 数。
        self.extend_input_len = 0
        # 在 extend 批次中的相对 logprob_start_len。
        self.extend_logprob_start_len = 0
        self.last_node: Any = None  # radix 树中匹配到的最后一个节点
        self.last_host_node: Any = None  # 主机（CPU）端的最后一个节点
        self.host_hit_length = 0  # 主机端命中的前缀长度
        # 预取时从存储后端（L3）为该请求加载的 token 数。
        self.storage_hit_length = 0
        # SWA radix 树锁引用需锁定到的节点。
        self.swa_uuid_for_lock: Optional[int] = None
        # 已插入 tree cache 的前缀长度。
        self.cache_protected_len: int = 0

        # 是否被分块。每被分块一次递增，每处理一个分块请求递减。
        self.is_chunked = 0

        # 用于回退（retraction）。
        self.is_retracted = False
        # 标记该请求是否曾经被回退过。
        self.retracted_stain = False

        # 增量流式输出的偏移计数。
        self.send_token_offset: int = 0
        self.send_decode_id_offset: int = 0
        # TODO (Byron): 在 PD 分离模式下 send_output_token_logprobs_offset 与 send_decode_id_offset 可能不同，
        # 因为 decode 服务器没有第一个输出 token 的 logprobs。
        self.send_output_token_logprobs_offset: int = 0

        # Logprobs（入参）。
        self.return_logprob = return_logprob  # 是否返回 logprob
        # 计算 logprob 的起始索引。
        self.logprob_start_len = 0
        self.top_logprobs_num = top_logprobs_num
        self.token_ids_logprob = token_ids_logprob
        self.temp_scaled_logprobs = False
        self.top_p_normalized_logprobs = False

        # Logprobs（返回值）。
        # True 表示输入 logprob 已发送给 detokenizer。
        self.input_logprob_sent: bool = False
        self.input_token_logprobs_val: Optional[List[float]] = None
        self.input_token_logprobs_idx: Optional[List[int]] = None
        self.input_top_logprobs_val: Optional[List[float]] = None
        self.input_top_logprobs_idx: Optional[List[int]] = None
        self.input_token_ids_logprobs_val: Optional[List[float]] = None
        self.input_token_ids_logprobs_idx: Optional[List[int]] = None
        # 临时存放 input_token_logprobs 的容器。
        self.input_token_logprobs: Optional[List[Tuple[int]]] = None
        self.temp_input_top_logprobs_val: Optional[List[torch.Tensor]] = None
        self.temp_input_top_logprobs_idx: Optional[List[int]] = None
        self.temp_input_token_ids_logprobs_val: Optional[List[float]] = None
        self.temp_input_token_ids_logprobs_idx: Optional[List[int]] = None

        if return_logprob:
            # 形状：(bs, 1)
            self.output_token_logprobs_val = []
            self.output_token_logprobs_idx = []
            # 形状：(bs, k)
            self.output_top_logprobs_val = []
            self.output_top_logprobs_idx = []
            # 可能是列表或 GPU 张量（prefill-only 打分的延迟拷贝优化）。
            self.output_token_ids_logprobs_val: List[
                Union[List[float], torch.Tensor]
            ] = []
            self.output_token_ids_logprobs_idx = []
        else:
            self.output_token_logprobs_val = self.output_token_logprobs_idx = (
                self.output_top_logprobs_val
            ) = self.output_top_logprobs_idx = self.output_token_ids_logprobs_val = (
                self.output_token_ids_logprobs_idx
            ) = None
        self.hidden_states: List[List[float]] = []  # 隐藏状态（如需返回）
        self.hidden_states_tensor = None  # 注：PD + MTP 下用 tensor 而非 list 传输 hidden_states
        self.output_topk_p = None
        self.output_topk_index = None

        # 捕获被路由的专家（MoE）。
        self.return_routed_experts = return_routed_experts
        self.routed_experts: Optional[torch.Tensor] = (
            None  # CPU 张量，形状 (seqlen, topk)
        )
        # 自定义信息。
        self.customized_info: Optional[Dict[str, List[Any]]] = None

        # Embedding（返回值）。
        self.embedding = None

        # 约束解码（结构化输出）。
        self.grammar_key: Optional[Tuple[str, str]] = None
        self.grammar: Optional[Union[BaseGrammarObject, Future[BaseGrammarObject]]] = (
            None
        )
        self.grammar_wait_ct = 0

        # 已缓存在 KV 缓存中的 token 数。
        self.cached_tokens = 0
        self.already_computed = 0

        # 按来源细分的已缓存 token（用于 HiCache）。
        self.cached_tokens_device = 0  # 来自设备缓存（GPU）的 token
        self.cached_tokens_host = 0  # 来自主机缓存（CPU 内存）的 token
        self.cached_tokens_storage = 0  # 来自 L3 存储后端的 token
        self._cache_breakdown_computed = (
            False  # 标记细分是否已计算
        )

        # 投机解码中的验证前向次数，用于计算每请求的平均接受长度。
        self.spec_verify_ct = 0

        # 该请求在投机解码中被接受的 token 数，用于计算接受率与平均接受长度。
        self.spec_accepted_tokens = 0

        # 投机解码的接受直方图。
        # 下标 = 某步接受的 token 数，值 = 接受该数量的步数。
        # 例：histogram[0]=5 表示有 5 步接受 0 个 token；histogram[3]=10 表示有 10 步接受 3 个 token。
        self.spec_acceptance_histogram: List[int] = []

        # 该请求被回退 / 抢占的次数。
        self.retraction_count = 0
        self.retraction_mb_id = None

        # 用于可观测性（指标/时间统计）。
        self.metrics_collector = metrics_collector
        if time_stats is not None:
            self.time_stats = SchedulerReqTimeStats.new_from_obj(time_stats)
        else:
            self.time_stats = SchedulerReqTimeStats(disagg_mode=disagg_mode)
        self.time_stats.set_metrics_collector(metrics_collector)
        self.time_stats.set_scheduler_recv_time()
        self.has_log_time_stats: bool = False

        # 用于 PD 分离：bootstrap 主机/端口/房间号与 KV 发送器。
        self.bootstrap_host: str = bootstrap_host
        self.bootstrap_port: Optional[int] = bootstrap_port
        self.bootstrap_room: Optional[int] = bootstrap_room
        self.disagg_kv_sender: Optional[BaseKVSender] = None

        self.routed_dp_rank: Optional[int] = routed_dp_rank
        self.disagg_prefill_dp_rank: Optional[int] = disagg_prefill_dp_rank

        # 已发送 KV 缓存的起始索引。
        # 分块 prefill 时需逐块发送，每次分块前向后执行：
        # kv_send(req.input_ids[req.start_send_idx:len(req.fill_ids)])
        # start_send_idx = len(req.fill_ids)
        self.start_send_idx: int = 0

        # 在 overlap 调度下，把 KV 传输推迟到 `process_batch_result_disagg_prefill`（而非非 overlap 的 `process_prefill_chunk`），
        # 因为在 `process_prefill_chunk` 时 KV 尚未就绪。用 `tmp_end_idx` 保存待发送 KV 缓存的结束索引。
        self.tmp_end_idx: int = -1
        self.metadata_buffer_index: int = -1

        # 用于 Matryoshka embedding（可变维度 embedding）。
        self.dimensions = dimensions

        # 用于扩散式 LLM（diffusion LLM）。
        self.init_diffusion_llm(dllm_config)

        # 用于 hisparse。
        self.hisparse_staging = False

    @property
    def seqlen(self) -> int:
        """获取请求当前的序列长度（输入 + 已生成输出）。"""
        return len(self.origin_input_ids) + len(self.output_ids)

    @property
    def is_prefill_only(self) -> bool:
        """是否为仅 prefill 请求（无需生成 token）。注：启用投机解码时该优化被禁用。"""
        spec_alg = get_global_server_args().speculative_algorithm
        return self.sampling_params.max_new_tokens == 0 and spec_alg is None

    @property
    def output_ids_through_stop(self) -> List[int]:
        """获取直到停止条件（含停止位置）的输出 id。"""
        if self.finished_len is not None:
            return self.output_ids[: self.finished_len]
        return self.output_ids

    def pop_committed_kv_cache(self) -> int:
        """返回已提交 KV 缓存的长度并标记为已释放。"""
        assert (
            not self.kv_committed_freed
        ), f"Committed KV cache already freed ({self.kv_committed_len=})"
        self.kv_committed_freed = True
        return self.kv_committed_len

    def pop_overallocated_kv_cache(self) -> Tuple[int, int]:
        """返回超额分配的 KV 缓存区间并标记为已释放。"""

        # 注：当存在 KV 缓存超额分配时调用。
        # 超额分配：分配的 KV 缓存多于已提交长度，
        # 例如投机解码可能分配比实际使用更多的 KV 缓存。
        assert (
            not self.kv_overallocated_freed
        ), f"Overallocated KV cache already freed, {self.kv_committed_len=}, {self.kv_allocated_len=}"
        self.kv_overallocated_freed = True
        return self.kv_committed_len, self.kv_allocated_len

    def update_spec_acceptance_histogram(self, accepted_draft_tokens: int):
        """Update the speculative decoding acceptance histogram.

        Args:
            accepted_draft_tokens: Number of draft tokens accepted in this step.
        """
        if len(self.spec_acceptance_histogram) <= accepted_draft_tokens:
            self.spec_acceptance_histogram.extend(
                [0] * (accepted_draft_tokens - len(self.spec_acceptance_histogram) + 1)
            )
        self.spec_acceptance_histogram[accepted_draft_tokens] += 1

    def extend_image_inputs(self, image_inputs):
        """追加/合并多模态（图像）输入。"""
        if self.multimodal_inputs is None:
            self.multimodal_inputs = image_inputs
        else:
            self.multimodal_inputs.merge(image_inputs)

    def finished(self) -> bool:
        """请求是否已达到结束条件。"""
        return self.finished_reason is not None

    def init_next_round_input(
        self,
        tree_cache: Optional[BasePrefixCache] = None,
        cow_mamba: Optional[bool] = None,
    ):
        """为下一轮调度准备输入：更新 fill_ids、做前缀匹配，并计算本次需 prefill 的长度。"""
        if self.is_dllm():
            self._init_fill_ids_for_dllm()
            self.determine_dllm_phase()
        else:
            self.fill_ids = self.origin_input_ids + self.output_ids

        input_len = len(self.fill_ids)

        # Streaming sessions reuse committed KV from the session slot, so
        # custom logprob_start_len is not supported — override to -1.
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

        # NOTE: the matched length is at most 1 less than the input length to enable logprob computation
        max_prefix_len = input_len - 1
        if self.return_logprob and self.logprob_start_len >= 0:
            max_prefix_len = min(max_prefix_len, self.logprob_start_len)
        max_prefix_len = max(max_prefix_len, 0)
        token_ids = self.fill_ids[:max_prefix_len]

        if tree_cache is not None:
            if cow_mamba is None:
                cow_mamba = tree_cache.supports_mamba()
            match_result = tree_cache.match_prefix(
                MatchPrefixParams(
                    key=RadixKey(token_ids=token_ids, extra_key=self.extra_key),
                    req=self,
                    cow_mamba=cow_mamba,
                )
            )
            (
                self.prefix_indices,
                self.last_node,
                self.last_host_node,
                self.host_hit_length,
                self.mamba_branching_seqlen,
            ) = (
                match_result.device_indices,
                match_result.last_device_node,
                match_result.last_host_node,
                match_result.host_hit_length,
                match_result.mamba_branching_seqlen,
            )
            if match_result.cache_protected_len is not None:
                self.cache_protected_len = match_result.cache_protected_len
            else:
                self.cache_protected_len = len(self.prefix_indices)

            if self.is_dllm():
                self._update_block_offset_for_dllm()

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

        self.set_extend_input_len(len(self.fill_ids) - len(self.prefix_indices))

    # 参考 https://github.com/vllm-project/vllm/blob/7a64d24aad69e4d2548aa0bf528d9fe63428ab01/vllm/transformers_utils/detokenizer.py#L194-L313
    def init_incremental_detokenize(self):
        """初始化/推进增量反分词：返回环绕+待解码 id 序列，以及读取偏移。"""
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

    def tail_str(self) -> str:
        """返回输出末尾足够长的解码字符串，用于检查停止字符串/停止正则。"""
        # 一起检查停止字符串与停止正则模式
        if (
            len(self.sampling_params.stop_strs) == 0
            and len(self.sampling_params.stop_regex_strs) == 0
        ):
            return ""

        max_len_tail_str = max(
            self.sampling_params.stop_str_max_len + 1,
            self.sampling_params.stop_regex_max_len + 1,
        )

        tail_len = min(max_len_tail_str, len(self.output_ids))
        return self.tokenizer.decode(self.output_ids[-tail_len:])

    def check_match_stop_str_prefix(self) -> bool:
        """检查末尾字符串的后缀是否与任一停止字符串的前缀重叠（用于流式输出防切断）。"""
        if not self.sampling_params.stop_strs:
            return False

        tail_str = self.tail_str()

        # Early return if tail_str is empty
        if not tail_str:
            return False

        for stop_str in self.sampling_params.stop_strs:
            if not stop_str:
                continue
            # 检查 stop_str 是否包含在 tail_str 中（最快的检查优先）
            if stop_str in tail_str:
                return True

            # 检查 tail_str 的后缀是否与 stop_str 的前缀匹配
            # 仅在 stop_str 非空时检查，用于流式输出
            min_len = min(len(tail_str), len(stop_str))
            for i in range(1, min_len + 1):
                if tail_str[-i:] == stop_str[:i]:
                    return True

        return False

    def _check_token_based_finish(self, new_accepted_tokens: List[int]) -> bool:
        """基于 token 的结束检查：命中 EOS 或停止 token id 则标记结束。"""
        if self.sampling_params.ignore_eos:
            return False

        # 检查停止 token id
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

    def _check_str_based_finish(self):
        """基于字符串/正则的结束检查：命中停止字符串或停止正则则标记结束。"""
        if (
            len(self.sampling_params.stop_strs) > 0
            or len(self.sampling_params.stop_regex_strs) > 0
        ):
            tail_str = self.tail_str()

            # 检查停止字符串
            if len(self.sampling_params.stop_strs) > 0:
                for stop_str in self.sampling_params.stop_strs:
                    if stop_str in tail_str or stop_str in self.decoded_text:
                        self.finished_reason = FINISH_MATCHED_STR(matched=stop_str)
                        return True

            # 检查停止正则
            if len(self.sampling_params.stop_regex_strs) > 0:
                for stop_regex_str in self.sampling_params.stop_regex_strs:
                    if re.search(stop_regex_str, tail_str):
                        self.finished_reason = FINISHED_MATCHED_REGEX(
                            matched=stop_regex_str
                        )
                        return True

        return False

    def _check_vocab_boundary_finish(self, new_accepted_tokens: List[int] = None):
        """词表越界检查：出现越界 token id（如 NaN 导致）时用停止 token 替换并结束。"""
        for i, token_id in enumerate(new_accepted_tokens):
            if token_id > self.vocab_size or token_id < 0:
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

    def check_finished(self, new_accepted_len: int = 1):
        """综合检查请求是否结束：依次检查 to_finish、最大长度、grammar 终止、token/越界/字符串结束。"""
        if self.finished():
            return

        if self.to_finish:
            self.finished_reason = self.to_finish
            self.to_finish = None
            return

        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self.finished_reason = FINISH_LENGTH(
                length=self.sampling_params.max_new_tokens
            )
            self.finished_len = self.sampling_params.max_new_tokens
            return

        if self.grammar is not None:
            if self.grammar.is_terminated():
                self.finished_reason = FINISH_MATCHED_TOKEN(matched=self.output_ids[-1])
                return

        new_accepted_tokens = self.output_ids[-new_accepted_len:]

        if self._check_token_based_finish(new_accepted_tokens):
            return

        if self._check_vocab_boundary_finish(new_accepted_tokens):
            return

        if self._check_str_based_finish():
            return

    def reset_for_retract(self):
        """回退（retract）请求：重置前缀/KV/mamba/logprob 等状态，使请求可重新排队 prefill。"""
        # 在重置其他状态前先递增回退计数。不能重置它，因为要统计每请求的总回退次数。
        self.retraction_count += 1

        self.prefix_indices = torch.empty((0,), dtype=torch.int64)
        self.routed_experts = None
        self.last_node = None
        self.swa_uuid_for_lock = None
        self.extend_input_len = 0
        self.is_retracted = True
        self.retracted_stain = True
        self.input_token_logprobs = None
        self.temp_input_top_logprobs_val = None
        self.temp_input_top_logprobs_idx = None
        self.extend_logprob_start_len = 0
        self.is_chunked = 0
        self.mamba_pool_idx = None
        self.mamba_ping_pong_track_buffer = None
        self.mamba_next_track_idx = None
        self.mamba_last_track_seqlen = None
        self.mamba_branching_seqlen = None
        self.already_computed = 0
        self.kv_allocated_len = 0
        self.kv_committed_len = 0
        self.kv_committed_freed = False
        self.kv_overallocated_freed = False
        self.swa_evicted_seqlen = 0
        self.extend_batch_idx = 0
        self.decode_batch_idx = 0

        # When using input_embeds, we cannot easily mix the original input embeddings
        # with the newly generated output token IDs during re-prefill of retracted request.
        # output_ids will have no use, but will lead to wrong size cache indexes.
        # Therefore, we discard the generated output_ids and restart prefill and generation
        # to ensure shape consistency in KV cache.
        if self.input_embeds is not None:
            self.output_ids = []

    def offload_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):
        """将该请求的 KV 缓存卸载（拷贝）到 CPU。"""
        token_indices = req_to_token_pool.req_to_token[
            self.req_pool_idx, : self.seqlen - 1
        ]
        self.kv_cache_cpu = token_to_kv_pool_allocator.get_cpu_copy(token_indices)

    def load_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):
        """将之前卸载到 CPU 的 KV 缓存重新加载回设备。"""
        token_indices = req_to_token_pool.req_to_token[
            self.req_pool_idx, : self.seqlen - 1
        ]
        token_to_kv_pool_allocator.load_cpu_copy(self.kv_cache_cpu, token_indices)
        del self.kv_cache_cpu

    def log_time_stats(self):
        """记录该请求的时间统计日志（去重，overlap 调度下可能被调用两次）。"""
        # 若为 overlap 调度，会提前调度一个 decode 批次，所以此函数会被调用两次。
        if self.has_log_time_stats:
            return

        bootstrap_info = (
            f", bootstrap_room={self.bootstrap_room}"
            if self.bootstrap_room is not None
            else ""
        )
        prefix = f"Req Time Stats(rid={self.rid}{bootstrap_info}, input len={len(self.origin_input_ids)}, output len={len(self.output_ids)}, type={self.time_stats.disagg_mode_str()})"
        logger.info(f"{prefix}: {self.time_stats.convert_to_duration()}")
        self.has_log_time_stats = True

    def set_extend_input_len(self, extend_input_len: int):
        """设置 extend_input_len，并计算 extend 批次内的相对 logprob_start_len。

        关键变量：
        - logprob_start_len：在完整序列中开始计算 logprob 的绝对位置。
        - extend_logprob_start_len：在当前 extend 批次内开始计算 logprob 的相对位置。
        - extend_input_len：本次 extend 批次需要处理的 token 数。
        """
        self.extend_input_len = extend_input_len
        if self.logprob_start_len == -1:
            logprob_start_len = len(self.fill_ids)
        else:
            # logprob_start_len 至少应不小于前缀索引的长度
            logprob_start_len = max(self.logprob_start_len, len(self.prefix_indices))
        self.extend_logprob_start_len = min(
            logprob_start_len - len(self.prefix_indices),
            self.extend_input_len,
        )

    def set_finish_with_abort(self, error_msg: str):
        """以中止（abort）方式结束请求：清理多模态/grammar，设置错误原因为 BadRequest。"""
        if get_tensor_model_parallel_rank() == 0:
            logger.error(f"{error_msg}, {self.rid=}")
        self.multimodal_inputs = None
        self.grammar = None
        self.origin_input_ids = [0]  # 设为单个 token 以跳过过长的 prefill
        self.return_logprob = False
        self.logprob_start_len = -1
        self.to_finish = FINISH_ABORT(
            error_msg, HTTPStatus.BAD_REQUEST, "BadRequestError"
        )

    def __repr__(self):
        return (
            f"Req(rid={self.rid}, "
            f"input_ids={self.origin_input_ids}, output_ids={self.output_ids}, "
            f"{self.grammar=}, "
            f"{self.sampling_params=})"
        )


@dataclasses.dataclass
class ScheduleBatch(ScheduleBatchDisaggregationDecodeMixin):
    """在调度器上存储一个批次的全部信息（高层调度数据，大部分位于 CPU）。"""

    # 请求、显存池与缓存。
    reqs: List[Req]  # 本批次包含的请求列表
    req_to_token_pool: ReqToTokenPool = None  # 请求 -> token 的映射池
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator = None  # token -> KV 缓存的分配器
    tree_cache: BasePrefixCache = None  # 前缀（radix）缓存
    is_hybrid_swa: bool = False  # 是否为混合 SWA（滑动窗口注意力）

    # 批次配置。
    model_config: ModelConfig = None  # 模型配置
    forward_mode: ForwardMode = None  # 前向模式（EXTEND/DECODE/MIXED 等）
    enable_overlap: bool = False  # 是否启用 overlap 调度
    # 标记当前运行批次是否已满，以便跳过“是否 prefill 新请求”的检查。
    # 这是为减少 prefill 检查开销的优化。
    batch_is_full: bool = False

    # 用于 PP（流水线并行）下的分块 prefill。
    chunked_req: Optional[Req] = None

    # 采样信息。
    sampling_info: SamplingBatchInfo = None

    # 传给模型运行器的批量化参数。
    input_ids: torch.Tensor = None  # 形状：[b]，int64
    input_embeds: torch.Tensor = None  # 形状：[b, hidden_size]，float32
    ne_token_table: torch.Tensor = None
    token_type_ids: torch.Tensor = None  # 形状：[b]，int64
    req_pool_indices: torch.Tensor = None  # 形状：[b]，int64
    seq_lens: torch.Tensor = None  # 形状：[b]，int64
    seq_lens_cpu: torch.Tensor = None  # 形状：[b]，int64
    # KV 缓存的输出位置。
    out_cache_loc: torch.Tensor = None  # 形状：[b]，int64
    output_ids: torch.Tensor = None  # 形状：[b]，int64

    # 用于混合 GDN 前缀缓存（mamba 状态跟踪）。
    mamba_track_indices: torch.Tensor = None  # 形状：[b]，int64
    mamba_track_mask: torch.Tensor = None  # 形状：[b]，bool
    mamba_track_seqlens: torch.Tensor = None  # 形状：[b]，int64

    # 多模态输入。
    multimodal_inputs: Optional[List] = None

    # 所有序列长度之和。
    seq_lens_sum: int = None
    # 原始序列长度（Qwen-1M 相关）。
    orig_seq_lens: torch.Tensor = None  # 形状：[b]，int32

    # 用于 DP（数据并行）注意力。
    inner_idle_batch: Optional[ScheduleBatch] = None
    global_num_tokens: Optional[List[int]] = None
    global_num_tokens_for_logprob: Optional[List[int]] = None
    is_extend_in_batch: bool = False
    all_extend_in_batch: bool = False
    can_run_dp_cuda_graph: bool = False
    tbo_split_seq_index: Optional[int] = None
    global_forward_mode: Optional[ForwardMode] = None

    # 用于处理 logprobs。
    return_logprob: bool = False
    top_logprobs_nums: Optional[List[int]] = None
    token_ids_logprobs: Optional[List[List[int]]] = None

    # 用于 logits 与 logprob 的后处理。
    temp_scaled_logprobs: bool = False
    top_p_normalized_logprobs: bool = False

    # 用于 extend 与混合分块 prefill。
    prefix_lens: List[int] = None  # 各请求命中的前缀长度
    extend_lens: List[int] = None  # 各请求本次 extend 的 token 数
    extend_num_tokens: Optional[int] = None  # 本批次 extend 的 token 总数
    decoding_reqs: List[Req] = None  # 混合批次中处于 decode 阶段的请求
    extend_logprob_start_lens: List[int] = None
    # 若不需 logprob，则为空列表。
    extend_input_logprob_token_ids: Optional[torch.Tensor] = None

    # 用于编码器-解码器（encoder-decoder）架构。
    encoder_cached: Optional[List[bool]] = None
    encoder_lens: Optional[torch.Tensor] = None
    encoder_lens_cpu: Optional[List[int]] = None
    encoder_out_cache_loc: Optional[torch.Tensor] = None

    # 用于 Matryoshka embedding。
    dimensions: Optional[list[int]] = None

    # 用于拆分 prefill。
    split_index: int = 0
    split_prefill_finished: bool = False
    split_forward_count: int = 1
    split_forward_batch: ForwardBatch = None
    seq_lens_cpu_cache: torch.Tensor = None

    # 是否含流式请求。
    has_stream: bool = False

    # 是否含 grammar（结构化输出）请求。
    has_grammar: bool = False

    # 设备。
    device: str = "cuda"

    # 投机解码。
    spec_algorithm: SpeculativeAlgorithm = None
    spec_info: Optional[SpecInput] = None

    # 是否返回隐藏状态。
    return_hidden_states: bool = False

    # 是否返回被捕获的专家（MoE）。
    return_routed_experts: bool = False

    # 该批次是否仅 prefill（无需生成 token）。
    is_prefill_only: bool = False

    # HiCache 指针，用于同步从 CPU 到 GPU 的数据加载。
    hicache_consumer_index: int = -1

    # 扩散式 LLM。
    dllm_config: Optional[DllmConfig] = None

    # 指标。
    dp_cooperation_info: Optional[DPCooperationInfo] = None
    prefill_stats: Optional[PrefillStats] = None

    # HiSparse。
    hisparse_coordinator: Optional[HiSparseCoordinator] = None

    @classmethod
    def init_new(
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
        """从请求列表构造一个新的 ScheduleBatch，并聚合各请求的开关（logprob/流式/grammar 等）。"""
        return_logprob = any(req.return_logprob for req in reqs)

        is_hybrid_swa = False
        if isinstance(token_to_kv_pool_allocator, SWATokenToKVPoolAllocator):
            is_hybrid_swa = True

        return cls(
            reqs=reqs,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            is_hybrid_swa=is_hybrid_swa,
            model_config=model_config,
            enable_overlap=enable_overlap,
            return_logprob=return_logprob,
            has_stream=any(req.stream for req in reqs),
            has_grammar=any(req.grammar for req in reqs),
            device=req_to_token_pool.device,
            spec_algorithm=spec_algorithm,
            return_hidden_states=any(req.return_hidden_states for req in reqs),
            return_routed_experts=any(req.return_routed_experts for req in reqs),
            is_prefill_only=all(req.is_prefill_only for req in reqs),
            chunked_req=chunked_req,
            dllm_config=dllm_config,
        )

    def batch_size(self):
        """返回批次中的请求数。"""
        return len(self.reqs)

    def is_empty(self):
        """批次是否为空。"""
        return len(self.reqs) == 0

    def is_dllm(self):
        """是否为扩散式 LLM 批次。"""
        return self.dllm_config is not None

    def prepare_encoder_info_extend(self, input_ids: List[int], seq_lens: List[int]):
        """为 encoder-decoder 模型的 extend 准备编码器信息：计算编码器长度、拆分编/解码输出位置。"""
        _pin = is_pin_memory_available(self.device)
        self.encoder_lens_cpu = []
        self.encoder_cached = []

        for req in self.reqs:
            im = req.multimodal_inputs
            if im is None or im.num_image_tokens is None:
                # No image input
                self.encoder_lens_cpu.append(0)
                self.encoder_cached.append(True)
            else:
                self.encoder_lens_cpu.append(im.num_image_tokens)
                self.encoder_cached.append(
                    self.forward_mode.is_decode()
                    or len(req.prefix_indices) >= im.num_image_tokens
                )

        self.encoder_lens = torch.tensor(
            self.encoder_lens_cpu, dtype=torch.int64, pin_memory=_pin
        ).to(self.device, non_blocking=True)

        # Strip encoder infos
        pt = 0
        decoder_out_cache_loc = []
        encoder_out_cache_loc = []
        for i, req in enumerate(self.reqs):
            encoder_len = self.encoder_lens_cpu[i]
            seq_lens[i] -= encoder_len

            if len(req.prefix_indices) < encoder_len:
                # NOTE: the encoder part should be considered as a whole
                assert len(req.prefix_indices) == 0
                input_ids[i] = input_ids[i][encoder_len:]
                encoder_out_cache_loc.append(self.out_cache_loc[pt : pt + encoder_len])
                decoder_out_cache_loc.append(
                    self.out_cache_loc[pt + encoder_len : pt + req.extend_input_len]
                )
                self.extend_lens[i] -= encoder_len
                self.extend_num_tokens -= encoder_len
            else:
                decoder_out_cache_loc.append(
                    self.out_cache_loc[pt : pt + req.extend_input_len]
                )
                self.prefix_lens[i] -= encoder_len

            pt += req.extend_input_len

        # Reassign
        self.input_ids = torch.tensor(
            sum(input_ids, []), dtype=torch.int64, pin_memory=_pin
        ).to(self.device, non_blocking=True)
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

    def prepare_for_extend(self):
        """为 EXTEND（prefill）阶段准备批次张量：拼接输入 id、计算序列与前缀长度、分配 KV 等。"""
        self.forward_mode = ForwardMode.EXTEND

        if self.is_dllm():
            # 对 DLLM 使用独立的前向模式
            self.forward_mode = ForwardMode.DLLM_EXTEND

        # Init tensors
        reqs = self.reqs
        input_ids = [r.fill_ids[len(r.prefix_indices) :] for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = [len(r.fill_ids) for r in reqs]
        orig_seq_lens = [max(len(r.fill_ids), len(r.origin_input_ids)) for r in reqs]
        prefix_lens = [len(r.prefix_indices) for r in reqs]
        extend_lens = [r.extend_input_len for r in reqs]

        # 用于 Matryoshka embedding
        if self.model_config.is_matryoshka and any(
            r.dimensions is not None for r in reqs
        ):
            self.dimensions = [
                r.dimensions if r.dimensions else self.model_config.hidden_size
                for r in reqs
            ]

        token_type_ids = [
            r.token_type_ids for r in reqs if r.token_type_ids is not None
        ]

        _pin = is_pin_memory_available(self.device)
        input_ids_tensor = torch.tensor(
            list(chain.from_iterable(input_ids)), dtype=torch.int64, pin_memory=_pin
        ).to(self.device, non_blocking=True)
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int64, pin_memory=_pin).to(
            self.device, non_blocking=True
        )
        seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        orig_seq_lens_tensor = torch.tensor(
            orig_seq_lens, dtype=torch.int32, pin_memory=_pin
        ).to(self.device, non_blocking=True)

        token_type_ids_tensor = None
        if len(token_type_ids) > 0:
            token_type_ids_tensor = torch.tensor(
                sum(token_type_ids, []), dtype=torch.int64, pin_memory=_pin
            ).to(self.device, non_blocking=True)

        # 设置 alloc_for_extend 所需的批次字段
        self.prefix_lens = prefix_lens
        self.extend_lens = extend_lens
        self.seq_lens = seq_lens_tensor
        self.seq_lens_cpu = seq_lens_cpu
        self.extend_num_tokens = extend_num_tokens

        # 分配显存
        out_cache_loc, req_pool_indices_tensor, req_pool_indices = alloc_for_extend(
            self
        )

        # 设置字段
        input_embeds = []
        extend_input_logprob_token_ids = []
        multimodal_inputs = []
        mamba_track_mask_cpu = []
        mamba_track_indices_cpu = []
        mamba_track_seqlens_cpu = []

        for i, (req, seq_len, pre_len) in enumerate(zip(reqs, seq_lens, prefix_lens)):
            req.req_pool_idx = req_pool_indices[i]
            assert seq_len - pre_len == req.extend_input_len

            req.extend_batch_idx += 1

            # 更新请求级别的显存管理字段
            req.kv_committed_len = seq_len
            req.kv_allocated_len = seq_len

            # 若有 input_embeds，则存储之
            if req.input_embeds is not None:
                # 切片以匹配 extend_input_len——分块溢出时 PrefillAdder 会截断
                # fill_ids/extend_input_len，但不会截断 input_embeds。
                input_embeds.extend(
                    req.input_embeds[pre_len : pre_len + req.extend_input_len]
                )

            multimodal_inputs.append(req.multimodal_inputs)

            # cached_tokens 只计算一次。一旦被回退，'retracted_stain' 标记将始终为 True。
            if not req.retracted_stain:
                new_cached = pre_len - req.already_computed
                req.cached_tokens += new_cached

                # 按来源计算已缓存 token 的详细细分（用于 HiCache）。
                # 只在第一个分块计算一次——分块 prefill 的后续分块会错误地把之前
                # 已计算的 token 当作缓存命中。
                if not req._cache_breakdown_computed:
                    # 此时 prefix_indices 已由 schedule_policy 中的 init_load_back 加上主机数据，所以：
                    # - len(prefix_indices) = 设备原有 + 主机加载
                    # - host_hit_length = 来自主机缓存的 token 总数（含存储预取）
                    # - storage_hit_length = 从存储后端加载的 token（L3 命中）
                    # - device_portion = len(prefix_indices) - host_hit_length
                    #
                    # 存储命中现在在预取完成后由调度器跟踪，
                    # storage_hit_length 由 scheduler.pop_prefetch_loaded_tokens() 设置。
                    host_total = req.host_hit_length
                    # 将 storage 限制在 host_total 以处理边界情况
                    storage_portion = min(host_total, req.storage_hit_length)
                    host_portion = host_total - storage_portion
                    device_portion = max(0, len(req.prefix_indices) - host_total)

                    req.cached_tokens_device = device_portion
                    req.cached_tokens_host = host_portion
                    req.cached_tokens_storage = storage_portion
                    req._cache_breakdown_computed = True

                req.already_computed = seq_len
            req.is_retracted = False

            if get_global_server_args().enable_mamba_extra_buffer():
                self._mamba_radix_cache_v2_req_prepare_for_extend(
                    req,
                    mamba_track_mask_cpu,
                    mamba_track_indices_cpu,
                    mamba_track_seqlens_cpu,
                )

            if self.return_logprob:
                # 查找输入 logprob 的 token id。
                # 首先在 origin_input_ids 中找到全局索引并向后滑动 1 以计算输入 logprob，
                # 因为计算输入 logprob 需要下一个 token。例（分块大小 2）：
                #
                # input_logprobs = [1, 2, 3, 4]
                # fill_ids = [1, 2]
                # extend_input_logprob_token_id = [2, 3]
                #
                # 注意也可能溢出，此时用 0 填充：
                # input_logprobs = [1, 2, 3, 4]
                # fill_ids = [3, 4]
                # extend_input_logprob_token_id = [4, 0]
                global_start_idx, global_end_idx = (
                    len(req.prefix_indices),
                    len(req.fill_ids),
                )
                if req.logprob_start_len == -1:
                    logprob_start_len = len(req.origin_input_ids)
                else:
                    logprob_start_len = req.logprob_start_len
                # 应用 logprob_start_len
                if global_start_idx < logprob_start_len:
                    global_start_idx = logprob_start_len

                logprob_token_ids = req.origin_input_ids[
                    global_start_idx + 1 : global_end_idx + 1
                ]
                extend_input_logprob_token_ids.extend(logprob_token_ids)

                # 我们需要 req.extend_input_len - req.extend_logprob_start_len 个 token，
                # 而 logprob_token_ids 是用于输入 logprob 的，所以其余部分用 0 填充。
                extend_input_logprob_token_ids.extend(
                    [0]
                    * (
                        req.extend_input_len
                        - req.extend_logprob_start_len
                        - len(logprob_token_ids)
                    )
                )

        if self.return_logprob:
            extend_input_logprob_token_ids = torch.tensor(
                extend_input_logprob_token_ids
            )
            # 将占位符或越界 token id（如多模态哈希）限制在词表边界内，再发送到 GPU。
            extend_input_logprob_token_ids.clamp_(0, self.model_config.vocab_size - 1)
        else:
            extend_input_logprob_token_ids = None

        self.input_ids = input_ids_tensor
        self.req_pool_indices = req_pool_indices_tensor
        self.orig_seq_lens = orig_seq_lens_tensor
        self.out_cache_loc = out_cache_loc
        self.input_embeds = (
            torch.tensor(input_embeds, pin_memory=_pin).to(
                self.device, non_blocking=True
            )
            if input_embeds
            else None
        )
        for mm_input in multimodal_inputs:
            if mm_input is None:
                continue
            for mm_item in mm_input.mm_items:
                pixel_values = getattr(mm_item, "feature", None)
                if isinstance(pixel_values, torch.Tensor):
                    mm_item.feature = pixel_values.to(self.device, non_blocking=True)
                if get_global_server_args().language_only:
                    precomputed_embeddings = getattr(
                        mm_item, "precomputed_embeddings", None
                    )
                    if isinstance(precomputed_embeddings, torch.Tensor):
                        mm_item.precomputed_embeddings = precomputed_embeddings.to(
                            self.device, non_blocking=True
                        )
        self.multimodal_inputs = multimodal_inputs
        self.token_type_ids = token_type_ids_tensor
        self.seq_lens_sum = sum(seq_lens)

        if self.return_logprob:
            self.top_logprobs_nums = [r.top_logprobs_num for r in reqs]
            self.token_ids_logprobs = [r.token_ids_logprob for r in reqs]

        self.extend_logprob_start_lens = [r.extend_logprob_start_len for r in reqs]
        self.extend_input_logprob_token_ids = extend_input_logprob_token_ids

        if get_global_server_args().enable_mamba_extra_buffer():
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

        if self.model_config.is_encoder_decoder:
            self.prepare_encoder_info_extend(input_ids, seq_lens)

        # 构建采样信息
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    def _mamba_radix_cache_v2_req_prepare_for_extend(
        self,
        req: Req,
        mamba_track_mask_cpu: List[bool],
        mamba_track_indices_cpu: List[int],
        mamba_track_seqlens_cpu: List[int],
    ):
        """为启用 mamba radix 缓存 v2 的请求准备 extend：计算 mamba 状态的跟踪掩码/索引/序列长度。"""

        def _force_track_h(i: int) -> int:
            assert i % FLA_CHUNK_SIZE == 0
            # 传给 mamba_track_seqlens_cpu 的 mamba_track_seqlen 有 3 种情况：
            # 1) 与 FLA_CHUNK_SIZE 对齐 -> 从 last_recurrent_state 获取
            #    a) 是最后位置 -> 从 last_recurrent_state 获取
            #    b) 不是最后位置 -> 从 h 获取
            # 2) 与 FLA_CHUNK_SIZE 不对齐 -> 从 h 获取
            # 目前计算仅支持情况 1a 和 2。所以对于 1b，需要加 1 以强制计算从 h 获取正确的 mamba 状态。
            return i + 1

        mamba_cache_chunk_size = get_global_server_args().mamba_cache_chunk_size
        mask = req.extend_input_len >= mamba_cache_chunk_size
        mamba_track_mask_cpu.append(mask)
        mamba_track_indices_cpu.append(
            req.mamba_ping_pong_track_buffer[req.mamba_next_track_idx].item()
        )
        mamba_track_seqlen = -1
        if mask:
            # mamba_track_seqlen 用于在 hybrid_linear_attn_backend 的 _init_track_ssm_indices 中
            # 计算要跟踪的索引。由于对齐与非对齐时 ssm 状态的获取方式不同：
            # 若 1) 为最后位置 且 2) 对齐，则从 last_recurrent_state 获取；
            # 否则从 h 获取（即非对齐）。
            # 需要将非对齐的 seqlen 传入计算。即使传入的是 mamba_track_seqlen，
            # 实际被跟踪的 seqlen 是 mamba_last_track_seqlen。
            mamba_track_seqlen = len(req.prefix_indices) + req.extend_input_len

            # mamba_track_seqlen_aligned/mamba_last_track_seqlen 是实际被跟踪的 seqlen，
            # 用于传给 mamba radix 缓存，以跟踪该 mamba 状态应存储在哪个 seqlen。
            mamba_track_seqlen_aligned = (
                len(req.prefix_indices)
                + (req.extend_input_len // mamba_cache_chunk_size)
                * mamba_cache_chunk_size
            )

            # mamba_track_fla_chunk_aligned 是基于 FLA_CHUNK_SIZE 对齐的 seqlen。
            # 若 mamba_track_fla_chunk_aligned != mamba_track_seqlen_aligned（当 page_size > FLA_CHUNK_SIZE 时可能成立），
            # 需要通过 _force_track_h() 强制计算从 h 获取正确的 mamba 状态。
            mamba_track_fla_chunk_aligned = (
                len(req.prefix_indices)
                + (req.extend_input_len // FLA_CHUNK_SIZE) * FLA_CHUNK_SIZE
            )
            if mamba_track_fla_chunk_aligned != mamba_track_seqlen_aligned:
                # We want to track mamba_track_seqlen_aligned, and it's not the last position,
                # so we need to add 1 to the seqlen to retrieve the correct mamba state from h.
                mamba_track_seqlen = _force_track_h(mamba_track_seqlen_aligned)

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
                ) % mamba_cache_chunk_size == 0
                if (
                    req.mamba_branching_seqlen > len(req.prefix_indices)
                    and req.mamba_branching_seqlen < mamba_track_seqlen
                    and branching_seqlen_aligned_mask
                ):
                    # 我们要跟踪 mamba_track_seqlen_aligned，且它不是最后位置，
                    # 所以需要对 seqlen 加 1 以从 h 获取正确的 mamba 状态。
                    # 详见 _force_track_h()。
                    mamba_track_seqlen = _force_track_h(req.mamba_branching_seqlen)
                    mamba_track_seqlen_aligned = req.mamba_branching_seqlen
            req.mamba_last_track_seqlen = mamba_track_seqlen_aligned
        mamba_track_seqlens_cpu.append(mamba_track_seqlen)

    def prepare_for_split_prefill(self):
        """为拆分 prefill 准备批次，并将前向模式设为 SPLIT_PREFILL。"""
        self.prepare_for_extend()
        # 对拆分 prefill，需要将前向模式设为 SPLIT_PREFILL
        self.forward_mode = ForwardMode.SPLIT_PREFILL

    def mix_with_running(self, running_batch: "ScheduleBatch"):
        """将当前 prefill 批次与正在运行的 decode 批次混合（混合分块），在同一次前向中同时跳 prefill 与 decode。"""
        self.forward_mode = ForwardMode.MIXED
        running_bs = running_batch.batch_size()

        for req in running_batch.reqs:
            # 正在运行的（decode）请求每步只需 1 个 token
            req.fill_ids = req.origin_input_ids + req.output_ids
            req.set_extend_input_len(1)

        input_ids = torch.cat([self.input_ids, running_batch.input_ids])
        out_cache_loc = torch.cat([self.out_cache_loc, running_batch.out_cache_loc])

        self.merge_batch(running_batch)
        self.input_ids = input_ids
        self.out_cache_loc = out_cache_loc

        # 对 overlap 调度器，output_ids 有一步延迟
        delta = 0 if self.enable_overlap else -1

        # 注：prefix_indices 是已缓存的部分，但我们不会缓存每一个 decode 步
        self.prefix_lens.extend(
            [
                len(r.origin_input_ids) + len(r.output_ids) + delta
                for r in running_batch.reqs
            ]
        )
        self.extend_lens.extend([1] * running_bs)
        self.extend_num_tokens += running_bs
        # TODO (lianmin): 需重新审视。应为 seq_len - 1
        self.extend_logprob_start_lens.extend([0] * running_bs)
        self.is_prefill_only = False

    def new_tokens_required_next_decode(
        self, selected_indices: Optional[List[int]] = None
    ):
        """估算下一步 decode 所需新增的 token（按页对齐，考虑投机解码的超额分配）。"""
        page_size = self.token_to_kv_pool_allocator.page_size
        requests = (
            self.reqs
            if selected_indices is None
            else [self.reqs[i] for i in selected_indices]
        )

        if self.spec_algorithm.is_none():
            new_pages = sum(1 for r in requests if r.kv_committed_len % page_size == 0)
            return new_pages * page_size

        server_args = get_global_server_args()
        len_per_topk = server_args.speculative_num_steps or 1
        spec_topk = server_args.speculative_eagle_topk or 1
        spec_tokens = server_args.speculative_num_draft_tokens

        if page_size > 1 and spec_topk > 1:
            # 最后一个部分页与向上取整对齐
            len_per_topk = ceil_align(len_per_topk + page_size, page_size)
            spec_tokens = ceil_align(spec_tokens, page_size)
        elif page_size > 1:
            # 仅页对齐
            len_per_topk = ceil_align(len_per_topk, page_size)
            spec_tokens = ceil_align(spec_tokens, page_size)

        num_tokens = max(len_per_topk * spec_topk, spec_tokens) * len(requests)

        # v2 eagle 存在超额分配
        return num_tokens * (1 + self.is_spec_v2)

    def check_decode_mem(self, selected_indices: Optional[List[int]] = None):
        """检查显存是否足够下一步 decode；不足时先从 tree cache 驱逐。"""
        num_tokens = self.new_tokens_required_next_decode(selected_indices)
        evict_from_tree_cache(self.tree_cache, num_tokens)
        return self.token_to_kv_pool_allocator.available_size() >= num_tokens

    def retract_all(self, server_args: ServerArgs):
        """回退批次中的所有请求（释放显存并从批次中过滤）。"""
        retracted_reqs = self.reqs
        for idx in range(len(self.reqs)):
            self.release_req(idx, len(self.reqs) - idx, server_args)

        self.filter_batch(retracted_reqs)
        return retracted_reqs

    def retract_decode(
        self, server_args: ServerArgs
    ) -> Tuple[List[Req], float, List[Req]]:
        """显存不足时回退（retract）部分 decode 请求，以腾出显存给其余请求。"""
        sorted_indices = list(range(len(self.reqs)))

        # TODO(lsyin): 改进 radix 缓存的回退策略。
        # 对投机解码，filter_batch 接口只能从后面过滤请求，所以只能从后面回退。
        # TODO(sang): 清理结束路径并支持更好的回退策略。
        if not server_args.speculative_algorithm:
            sorted_indices.sort(
                key=lambda i: (
                    len(self.reqs[i].output_ids),
                    -len(self.reqs[i].origin_input_ids),
                ),
                reverse=True,
            )

        retracted_reqs = []
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
            retracted_reqs.append(req)
            # 释放显存且不插入 tree，因为我们需要立即腾出空间
            self.release_req(idx, len(sorted_indices), server_args)

        reqs_to_abort: List[Req] = []
        if len(sorted_indices) <= 1 and not self.check_decode_mem(
            selected_indices=sorted_indices
        ):
            # Even the last remaining request cannot fit in memory.
            # Instead of crashing the scheduler, gracefully abort it.
            last_idx = sorted_indices.pop()
            last_req = self.reqs[last_idx]
            # 即使回退了所有其他请求，最后一个请求仍装不下，为避免调度器崩溃优雅地中止它。
            last_req.to_finish = FINISH_ABORT(
                "Out of memory even after retracting all other requests "
                "in the decode batch. Aborting the last request.",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            reqs_to_abort.append(last_req)
            self.release_req(last_idx, 0, server_args)
            logger.warning(
                "retract_decode: aborted last request %s due to OOM", last_req.rid
            )

        self.filter_batch(keep_indices=sorted_indices)

        # Reqs in batch are filtered
        total_decoded_tokens = sum(len(r.output_ids) for r in self.reqs)
        total_max_new_tokens = sum(r.sampling_params.max_new_tokens for r in self.reqs)

        new_estimate_ratio = (
            total_decoded_tokens
            + envs.SGLANG_RETRACT_DECODE_STEPS.get() * len(self.reqs)
        ) / (
            total_max_new_tokens + 1
        )  # 避免除零
        new_estimate_ratio = min(1.0, new_estimate_ratio)

        return retracted_reqs, new_estimate_ratio, reqs_to_abort

    def release_req(self, idx: int, remaing_req_count: int, server_args: ServerArgs):
        """释放单个请求的显存/KV 缓存并重置其状态（用于回退/抢占）。"""
        req = self.reqs[idx]

        if server_args.disaggregation_mode == "decode":
            req.offload_kv_cache(
                self.req_to_token_pool, self.token_to_kv_pool_allocator
            )
        # TODO (csy): 对被抢占的请求，可能希望插入 tree
        release_kv_cache(req, self.tree_cache, is_insert=False)
        # 注(lsyin): 应立即使用新可驱逐的显存。
        num_tokens = remaing_req_count * envs.SGLANG_RETRACT_DECODE_STEPS.get()
        evict_from_tree_cache(self.tree_cache, num_tokens)

        req.reset_for_retract()

    def prepare_encoder_info_decode(self):
        """decode 阶段重置编码器缓存状态（编码器输出已在 prefill 阶段缓存）。"""
        # 重置编码器缓存状态
        self.encoder_cached = [True] * len(self.reqs)

    def prepare_for_idle(self):
        """为 IDLE（空转）模式准备批次：DP 注意力下某些 rank 无请求时需要参与集体通信。"""
        self.forward_mode = ForwardMode.IDLE
        self.input_ids = torch.empty(0, dtype=torch.int64, device=self.device)
        self.seq_lens = torch.empty(0, dtype=torch.int64, device=self.device)
        self.seq_lens_cpu = torch.empty(0, dtype=torch.int64)
        self.orig_seq_lens = torch.empty(0, dtype=torch.int32, device=self.device)
        self.out_cache_loc = torch.empty(0, dtype=torch.int64, device=self.device)
        self.req_pool_indices = torch.empty(0, dtype=torch.int64, device=self.device)
        self.seq_lens_sum = 0
        self.extend_num_tokens = 0
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    @property
    def is_spec_v2(self):
        """是否为投机解码 v2 路径（overlap 调度 + 启用投机解码）。"""
        # FIXME: 最终废弃 is_spec_v2
        ret = self.enable_overlap and not self.spec_algorithm.is_none()
        assert not ret or self.spec_algorithm.supports_spec_v2()
        return ret

    def prepare_for_decode(self):
        """为 DECODE 阶段准备批次：将上一步输出作为输入、分配新 KV、递增序列长度等。"""
        self.forward_mode = ForwardMode.DECODE
        bs = len(self.reqs)
        # decode 通过 embed_tokens 嵌入上一个输出 token；清除陈旧的 prefill 阶段张量，
        # 以免泄漏到 ForwardBatch。
        self.input_embeds = None

        # 清除上下文并行（CP）元数据——CP 仅用于 prefill，不用于 decode
        if hasattr(self, "attn_cp_metadata") and self.attn_cp_metadata is not None:
            self.attn_cp_metadata = None
        if hasattr(self, "nsa_cp_metadata") and self.nsa_cp_metadata is not None:
            self.nsa_cp_metadata = None

        if self.is_spec_v2:
            # TODO(spec-v2): 所有 spec v2 都应走这条路径
            draft_input: EagleDraftInput = self.spec_info
            draft_input.prepare_for_decode(self)

        if not self.spec_algorithm.is_none():
            # 若使用投机解码，decode 批次在运行草稿模型后于
            # `forward_batch_speculative_generation` 内部准备。
            return

        if self.sampling_info.penalizer_orchestrator.is_required:
            if self.enable_overlap:
                # TODO: 这可能较慢，需优化。
                delayed_output_ids = torch.tensor(
                    [
                        (
                            req.output_ids[-1]
                            if len(req.output_ids)
                            else req.origin_input_ids[-1]
                        )
                        for req in self.reqs
                    ],
                    dtype=torch.int64,
                    device=self.device,
                )
                self.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                    delayed_output_ids
                )
            else:
                self.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                    self.output_ids.to(torch.int64)
                )

        # 更新字段
        self.input_ids = self.output_ids
        self.output_ids = None

        if self.model_config.is_encoder_decoder:
            self.prepare_encoder_info_decode()

        # 分配显存
        self.out_cache_loc = alloc_for_decode(self, token_per_req=1)

        # 更新请求级别的显存管理字段
        for req in self.reqs:
            req.decode_batch_idx += 1
            req.kv_committed_len += 1
            req.kv_allocated_len += 1

        # 分配后更新 seq_lens
        if self.enable_overlap:
            # overlap 模式下不使用原地操作
            self.seq_lens = self.seq_lens + 1
            self.seq_lens_cpu = self.seq_lens_cpu + 1
            self.orig_seq_lens = self.orig_seq_lens + 1
        else:
            # 更快的原地版本
            self.seq_lens.add_(1)
            self.seq_lens_cpu.add_(1)
            self.orig_seq_lens.add_(1)
        self.seq_lens_sum += bs

        if self.hisparse_coordinator is not None:
            self.hisparse_coordinator.map_last_loc_to_buffer(
                self.seq_lens,
                self.out_cache_loc,
                self.req_pool_indices,
                self.seq_lens_cpu,
            )

        if get_global_server_args().enable_mamba_extra_buffer():
            if len(self.reqs) == 0:
                self.mamba_track_indices = torch.empty(
                    (0,), dtype=torch.int64, device=self.device
                )
            else:
                # already on device
                all_buffers = torch.stack(
                    [req.mamba_ping_pong_track_buffer for req in self.reqs]
                )
                idx = (
                    torch.tensor(
                        [req.mamba_next_track_idx for req in self.reqs],
                        dtype=torch.int64,
                        pin_memory=True,
                    )
                    .unsqueeze(1)
                    .to(device=all_buffers.device, non_blocking=True)
                )
                self.mamba_track_indices = (
                    torch.gather(all_buffers, 1, idx).squeeze(1).to(torch.int64)
                )

            # 异步 H2D（主机到设备）
            self.mamba_track_mask = (
                (self.seq_lens_cpu % get_global_server_args().mamba_track_interval == 0)
                .pin_memory()
                .to(device=self.device, non_blocking=True)
            )

    def maybe_wait_verify_done(self):
        """投机解码 v2 下等待验证完成，以获取正确的 seq_lens。"""
        if self.is_spec_v2:
            draft_input: EagleDraftInput = self.spec_info
            if draft_input.verify_done is not None:
                draft_input.verify_done.synchronize()

    def filter_batch(
        self,
        chunked_req_to_exclude: Optional[Union[Req, List[Req]]] = None,
        keep_indices: Optional[List[int]] = None,
        # FIXME(lsyin): spec v1 废弃后移除此参数
        v1_spec_info_filtered: Optional[bool] = False,
    ):
        """从批次中过滤出要保留的请求（移除已结束/指定排除的请求），并同步各张量字段。"""
        # FIXME(lsyin): 这里用于获取正确的 seq_lens。
        # 批次已启动，但需要它被验证以获取正确的下一批次信息。
        self.maybe_wait_verify_done()

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
            # 过滤掉所有请求
            self.reqs = []
            return

        if len(keep_indices) == len(self.reqs):
            # 无需过滤
            return

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
        self.seq_lens = self.seq_lens[keep_indices_device]
        self.seq_lens_cpu = self.seq_lens_cpu[keep_indices]
        self.orig_seq_lens = self.orig_seq_lens[keep_indices_device]
        self.out_cache_loc = None
        self.seq_lens_sum = self.seq_lens.sum().item()

        if self.output_ids is not None:
            self.output_ids = self.output_ids[keep_indices_device]

        self.mamba_track_indices = None
        self.mamba_track_mask = None
        self.mamba_track_seqlens = None
        self.return_logprob = any(req.return_logprob for req in self.reqs)
        if self.return_logprob:
            self.top_logprobs_nums = [self.top_logprobs_nums[i] for i in keep_indices]
            self.token_ids_logprobs = [self.token_ids_logprobs[i] for i in keep_indices]
        else:
            self.top_logprobs_nums = None
            self.token_ids_logprobs = None

        self.has_stream = any(req.stream for req in self.reqs)
        self.has_grammar = any(req.grammar for req in self.reqs)

        self.sampling_info.filter_batch(keep_indices, keep_indices_device)
        # 注：spec_info 在批次过滤之前被过滤仅发生于：
        # - spec v1 的验证阶段
        # - 仅针对 decode 批次（running_batch）
        has_been_filtered = v1_spec_info_filtered and not self.is_spec_v2

        if self.spec_info:
            self.spec_info.filter_batch(
                new_indices=keep_indices_device,
                has_been_filtered=has_been_filtered,
            )

    def merge_batch(self, other: "ScheduleBatch"):
        """将另一个批次（通常是正在运行的 decode 批次）合并进本批次（通常是 prefill）。"""
        # 常规调度路径下：
        # 1) self 总是 prefill，其 seq_lens 不是 future；
        # 2) other 总是 decode，上一步已完成，所以 verify_done 已同步，此调用为空操作。
        # 在 PD 分离 decode + overlap 下，merge_batch 可能在 filter_batch 之前被调用，
        # running_batch.seq_lens 可能仍是 forward_stream future，故在此同步以避免跨流数据竞争。
        self.maybe_wait_verify_done()

        # 惩罚器编排器必须在合并 Batch.reqs 之前合并。因为 orchestrator.merge()
        # 在准备各惩罚器时依赖 Batch.reqs，需用合并前的 Batch.reqs 调用。
        self.sampling_info.merge_batch(other.sampling_info)

        # 编码器-解码器信息
        if self.model_config.is_encoder_decoder:
            self.encoder_lens = torch.cat([self.encoder_lens, other.encoder_lens])
            self.encoder_lens_cpu.extend(other.encoder_lens_cpu)
        self.req_pool_indices = torch.cat(
            [self.req_pool_indices, other.req_pool_indices]
        )
        self.seq_lens = torch.cat([self.seq_lens, other.seq_lens])
        self.seq_lens_cpu = torch.cat([self.seq_lens_cpu, other.seq_lens_cpu])
        self.orig_seq_lens = torch.cat([self.orig_seq_lens, other.orig_seq_lens])
        self.out_cache_loc = None
        self.seq_lens_sum += other.seq_lens_sum
        if self.output_ids is not None:
            self.output_ids = torch.cat([self.output_ids, other.output_ids])
        self.mamba_track_indices = None
        self.mamba_track_mask = None
        self.mamba_track_seqlens = None
        if self.return_logprob and other.return_logprob:
            self.top_logprobs_nums.extend(other.top_logprobs_nums)
            self.token_ids_logprobs.extend(other.token_ids_logprobs)
        elif self.return_logprob:
            self.top_logprobs_nums.extend([0] * len(other.reqs))
            self.token_ids_logprobs.extend([None] * len(other.reqs))
        elif other.return_logprob:
            self.top_logprobs_nums = [0] * len(self.reqs) + other.top_logprobs_nums
            self.token_ids_logprobs = [None] * len(self.reqs) + other.token_ids_logprobs
        self.reqs.extend(other.reqs)
        if self.multimodal_inputs is not None:
            self.multimodal_inputs.extend(other.multimodal_inputs)

        self.return_logprob |= other.return_logprob
        self.has_stream |= other.has_stream
        self.has_grammar |= other.has_grammar
        self.return_hidden_states |= other.return_hidden_states
        self.is_prefill_only = self.is_prefill_only and other.is_prefill_only

        if self.spec_info:
            self.spec_info.merge_batch(other.spec_info)

    def get_model_worker_batch(
        self, seq_lens_cpu_cache: Optional[torch.Tensor] = None
    ) -> ModelWorkerBatch:
        """将 ScheduleBatch 转换为 ModelWorkerBatch（仅保留 GPU 前向所需的子集字段）。"""
        if self.forward_mode.is_decode_or_idle():
            extend_seq_lens = extend_prefix_lens = extend_logprob_start_lens = None
        else:
            extend_seq_lens = self.extend_lens
            extend_prefix_lens = self.prefix_lens
            extend_logprob_start_lens = self.extend_logprob_start_lens

        if self.sampling_info:
            if self.has_grammar:
                self.sampling_info.grammars = [req.grammar for req in self.reqs]
            else:
                self.sampling_info.grammars = None

        seq_lens_cpu = (
            seq_lens_cpu_cache if seq_lens_cpu_cache is not None else self.seq_lens_cpu
        )

        return ModelWorkerBatch(
            forward_mode=self.forward_mode,
            input_ids=self.input_ids,
            req_pool_indices=self.req_pool_indices,
            seq_lens=self.seq_lens,
            orig_seq_lens=self.orig_seq_lens,
            out_cache_loc=self.out_cache_loc,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=self.seq_lens_sum,
            return_logprob=self.return_logprob,
            top_logprobs_nums=self.top_logprobs_nums,
            token_ids_logprobs=self.token_ids_logprobs,
            global_num_tokens=self.global_num_tokens,
            global_num_tokens_for_logprob=self.global_num_tokens_for_logprob,
            is_extend_in_batch=self.is_extend_in_batch,
            all_extend_in_batch=self.all_extend_in_batch,
            can_run_dp_cuda_graph=self.can_run_dp_cuda_graph,
            tbo_split_seq_index=self.tbo_split_seq_index,
            global_forward_mode=self.global_forward_mode,
            extend_num_tokens=self.extend_num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_logprob_start_lens=extend_logprob_start_lens,
            multimodal_inputs=self.multimodal_inputs,
            encoder_cached=self.encoder_cached,
            encoder_lens=self.encoder_lens,
            encoder_lens_cpu=self.encoder_lens_cpu,
            encoder_out_cache_loc=self.encoder_out_cache_loc,
            lora_ids=[req.lora_id for req in self.reqs],
            sampling_info=self.sampling_info,
            input_embeds=self.input_embeds,
            ne_token_table=self.ne_token_table,
            token_type_ids=self.token_type_ids,
            spec_algorithm=self.spec_algorithm,
            spec_info=self.spec_info,
            hicache_consumer_index=self.hicache_consumer_index,
            capture_hidden_mode=(
                CaptureHiddenMode.FULL
                if self.return_hidden_states
                else (
                    getattr(
                        self.spec_info, "capture_hidden_mode", CaptureHiddenMode.NULL
                    )
                    if self.spec_info
                    else CaptureHiddenMode.NULL
                )
            ),
            extend_input_logprob_token_ids=self.extend_input_logprob_token_ids,
            is_prefill_only=self.is_prefill_only,
            dimensions=self.dimensions,
            dllm_block_offsets=[req.dllm_block_offset for req in self.reqs],
            dllm_config=self.dllm_config,
            reqs=self.reqs,
            has_grammar=self.has_grammar,
            mamba_track_indices=self.mamba_track_indices,
            mamba_track_mask=self.mamba_track_mask,
            mamba_track_seqlens=self.mamba_track_seqlens,
        )

    def copy(self):
        """生成一个仅含 process_batch_result 所需字段的快照（浅拷贝 reqs 列表）。"""
        # 仅包含 process_batch_result 会用到的字段。对 reqs 列表做浅拷贝，
        # 以免对原批次的原地修改（filter_batch、merge_batch）破坏该快照。
        return ScheduleBatch(
            reqs=self.reqs[:],
            req_to_token_pool=self.req_to_token_pool,
            req_pool_indices=self.req_pool_indices,
            model_config=self.model_config,
            forward_mode=self.forward_mode,
            out_cache_loc=self.out_cache_loc,
            return_logprob=self.return_logprob,
            decoding_reqs=self.decoding_reqs,
            spec_algorithm=self.spec_algorithm,
            global_num_tokens=self.global_num_tokens,
            global_num_tokens_for_logprob=self.global_num_tokens_for_logprob,
            can_run_dp_cuda_graph=self.can_run_dp_cuda_graph,
            all_extend_in_batch=self.all_extend_in_batch,
            is_extend_in_batch=self.is_extend_in_batch,
            is_prefill_only=self.is_prefill_only,
            seq_lens_cpu=self.seq_lens_cpu,
            enable_overlap=self.enable_overlap,
            mamba_track_indices=self.mamba_track_indices,
            mamba_track_mask=self.mamba_track_mask,
            mamba_track_seqlens=self.mamba_track_seqlens,
            dp_cooperation_info=self.dp_cooperation_info,
            prefill_stats=self.prefill_stats,
        )

    def maybe_evict_swa(self):
        """对支持 SWA（滑动窗口注意力）的缓存，适时驱逐滑动窗口之外不再需要的 KV。"""
        if self.tree_cache.supports_swa():
            sliding_window_size = self.tree_cache.sliding_window_size
            server_args = get_global_server_args()

            for idx, req in enumerate(self.reqs):
                if self.forward_mode.is_decode():
                    # 这里设置 evict_swa 条件有两个原因：
                    # 1. overlap 调度下，当 req.decode_batch_idx == 0 时不能驱逐 swa，因为上一个 extend 批次仍在运行。
                    # 2. 每 window_size 个 token 驱逐一次 swa 以减少开销。
                    if req.decode_batch_idx % sliding_window_size == 1:
                        self._evict_swa(req, req.seqlen - 1)
                elif self.forward_mode.is_extend() and self.tree_cache.is_chunk_cache():
                    pre_len = self.prefix_lens[idx]
                    if self.enable_overlap:
                        # 分块 prefill 情况下，当第二个 extend 批次在调度时，第一个 extend 批次仍在运行，所以不能驱逐 swa token
                        if req.extend_batch_idx < 2:
                            continue
                        else:
                            pre_len = (
                                pre_len - server_args.chunked_prefill_size
                                if server_args.chunked_prefill_size > 0
                                else pre_len
                            )
                            self._evict_swa(req, pre_len)
                    else:
                        self._evict_swa(req, pre_len)

    def _evict_swa(self, req: Req, pre_len: int):
        """驱逐该请求中既不在 tree cache、也不在滑动窗口内的 SWA KV，并释放其显存。"""
        assert self.tree_cache.supports_swa(), "prefix cache must support swa"
        sliding_window_size = self.tree_cache.sliding_window_size

        # 对 swa radix 缓存，需要驱逐那些既不在 tree cache、也不在滑动窗口内的 token
        assert (
            req.cache_protected_len % self.tree_cache.page_size == 0
        ), "cache_protected_len must be page aligned"
        req.swa_evicted_seqlen = max(req.swa_evicted_seqlen, req.cache_protected_len)

        new_swa_evicted_seqlen = max(
            req.swa_evicted_seqlen, pre_len - sliding_window_size
        )

        if self.tree_cache.page_size > 1:
            new_swa_evicted_seqlen = (
                new_swa_evicted_seqlen // self.tree_cache.page_size
            ) * self.tree_cache.page_size

        if new_swa_evicted_seqlen > req.swa_evicted_seqlen:
            free_slots = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, req.swa_evicted_seqlen : new_swa_evicted_seqlen
            ]
            self.token_to_kv_pool_allocator.free_swa(free_slots)
            req.swa_evicted_seqlen = new_swa_evicted_seqlen

    def __str__(self):
        return (
            f"ScheduleBatch(forward_mode={self.forward_mode.name if self.forward_mode else 'None'}, "
            f"#req={(len(self.reqs))})"
        )


@dataclasses.dataclass
class ModelWorkerBatch:
    """ScheduleBatch 的子集，仅包含 GPU 上模型前向所需的数据，由调度器传递给模型运行器。"""

    # 前向模式
    forward_mode: ForwardMode
    # 输入 token id
    input_ids: torch.Tensor
    # 请求在 req_to_token_pool 中的索引
    req_pool_indices: torch.Tensor
    # 序列长度
    seq_lens: torch.Tensor
    # 输出 token 在 token_to_kv_pool_allocator 中的索引
    out_cache_loc: torch.Tensor
    # CPU 上的序列长度张量
    seq_lens_cpu: Optional[torch.Tensor]
    seq_lens_sum: int

    # 用于 logprob
    return_logprob: bool
    top_logprobs_nums: Optional[List[int]]
    token_ids_logprobs: Optional[List[List[int]]]

    # 用于 DP（数据并行）注意力
    global_num_tokens: Optional[List[int]]
    global_num_tokens_for_logprob: Optional[List[int]]
    is_extend_in_batch: bool
    all_extend_in_batch: bool
    can_run_dp_cuda_graph: bool
    tbo_split_seq_index: Optional[int]
    global_forward_mode: Optional[ForwardMode]

    # 用于 extend（prefill）
    extend_num_tokens: Optional[int]
    extend_seq_lens: Optional[List[int]]
    extend_prefix_lens: Optional[List[int]]
    extend_logprob_start_lens: Optional[List[int]]
    extend_input_logprob_token_ids: Optional[torch.Tensor]

    # 用于多模态
    multimodal_inputs: Optional[List[MultimodalInputs]]

    # 用于编码器-解码器
    encoder_cached: Optional[List[bool]]
    encoder_lens: Optional[torch.Tensor]
    encoder_lens_cpu: Optional[List[int]]
    encoder_out_cache_loc: Optional[torch.Tensor]

    # 用于 LoRA
    lora_ids: Optional[List[str]]

    # 采样信息
    sampling_info: SamplingBatchInfo

    # 原始序列长度（Qwen-1M 相关）
    orig_seq_lens: Optional[torch.Tensor] = None

    # 输入 embedding
    input_embeds: Optional[torch.Tensor] = None

    # 用于 ngram embedding 的 token 表
    ne_token_table: Optional[torch.Tensor] = None

    # 用于 cross-encoder 模型
    token_type_ids: Optional[torch.Tensor] = None

    # 投机解码
    spec_algorithm: SpeculativeAlgorithm = None

    spec_info: Optional[SpecInput] = None

    # 若设置，批次输出将包含本次运行的隐藏状态。
    capture_hidden_mode: CaptureHiddenMode = None
    hicache_consumer_index: int = -1

    # 用于 Matryoshka embedding
    dimensions: Optional[list[int]] = None

    # 该批次是否仅 prefill（无需生成 token）
    is_prefill_only: bool = False

    # 扩散式 LLM
    dllm_block_offsets: Optional[List[int]] = None
    dllm_config: Optional[DllmConfig] = None

    # 用于约束解码
    # FIXME(lsyin): 完全 overlap grammar 后移除此字段
    reqs: Optional[List[Req]] = None
    has_grammar: bool = False

    # 用于归一化之前的隐藏状态
    return_hidden_states_before_norm: bool = False

    # 用于 mamba 状态跟踪
    mamba_track_indices: Optional[torch.Tensor] = None  # 形状：[b]，int64
    mamba_track_mask: Optional[torch.Tensor] = None  # 形状：[b]，bool
    mamba_track_seqlens: Optional[torch.Tensor] = None  # 形状：[b]，int64
