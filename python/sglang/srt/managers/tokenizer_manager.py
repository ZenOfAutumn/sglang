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
"""TokenizerManager is a process that tokenizes the text.

中译：TokenizerManager 是前端请求的「入口/出口」管理器，运行在 HTTP/Engine 进程内。
      它的职责包括：
      1) 入口：接收上层（HTTP server / Engine）的 GenerateReqInput / EmbeddingReqInput，
         做参数归一化、分词（tokenize）、多模态预处理、长度校验，封装为 Tokenized* 请求，
         经 ZMQ 发送给调度器（Scheduler）。
      2) 出口：通过 handle_loop 异步事件循环，从 DetokenizerManager 收回按 rid 标记的输出
         （BatchStrOutput / BatchEmbeddingOutput / BatchTokenIDOutput），按请求 id（rid）
         分发回各自等待中的协程，支持流式（streaming）增量输出。
      3) 全局控制：权重热更新（读写锁串行化）、暂停/继续生成、LoRA 适配器管理、
         指标采集、崩溃转储、优雅退出等。
      核心数据结构是 rid_to_state（rid -> ReqState），把「ZMQ 单向收发」桥接成
      「每个请求一个 async 生成器」的并发模型。
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import logging
import os
import pickle
import signal
import socket
import sys
import threading
import time
from array import array
from collections import deque
from contextlib import nullcontext
from datetime import datetime
from enum import Enum
from functools import lru_cache
from http import HTTPStatus
from typing import Any, Awaitable, Dict, Iterable, List, Optional, Tuple, Union

import fastapi
import numpy as np
import pybase64
import torch
import uvloop
import zmq
import zmq.asyncio
from fastapi import BackgroundTasks

from sglang.srt.beam_search.output import (
    build_beam_search_out,
    try_build_beam_search_out_dict,
)
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.encoder.receiver import create_mm_receiver
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
from sglang.srt.managers.async_dynamic_batch_tokenizer import AsyncDynamicbatchTokenizer
from sglang.srt.managers.disagg_service import start_disagg_service
from sglang.srt.managers.embed_types import PositionalEmbeds
from sglang.srt.managers.io_struct import (
    AbortReq,
    ActiveRanksOutput,
    BaseBatchReq,
    BaseReq,
    BatchEmbeddingOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    ConfigureLoggingReq,
    ContinueGenerationReqInput,
    ElasticScaleUpdateReq,
    EmbeddingReqInput,
    FreezeGCReq,
    GenerateReqInput,
    HealthCheckOutput,
    LoadLoRAAdapterReqInput,
    OpenSessionReqOutput,
    PauseGenerationReqInput,
    ScaleElasticEPReqInput,
    ScaleElasticEPReqOutput,
    SessionParams,
    ShutdownReq,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    async_sock_recv,
    async_sock_send,
    build_flat_input_top_logprobs_arrays,
    sock_send,
    unwrap_from_pickle,
)
from sglang.srt.managers.load_snapshot import create_load_snapshot_reader
from sglang.srt.managers.mm_utils import wrap_shm_features
from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    get_request_return_hidden_states_mode,
)
from sglang.srt.managers.scheduler_input_blocker import input_blocker_guard_region
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.managers.tokenizer_manager_score_mixin import TokenizerManagerScoreMixin
from sglang.srt.managers.utils import (
    compute_num_reserved_tokens,
    is_health_check_generate_req,
)
from sglang.srt.model_executor.forward_batch_info import (
    get_server_return_hidden_states_mode,
)
from sglang.srt.multimodal.transport import determine_tensor_transport_mode
from sglang.srt.observability.cpu_monitor import start_cpu_monitor_thread
from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_TOKENIZER,
    TokenizerMetricsCollector,
    resolve_collector_class,
)
from sglang.srt.observability.req_time_stats import (
    APIServerReqTimeStats,
    convert_time_to_realtime,
    real_time,
    set_time_batch,
)
from sglang.srt.observability.request_metrics_exporter import (
    RequestMetricsExporterManager,
)
from sglang.srt.observability.trace import SpanAttributes, extract_trace_headers
from sglang.srt.runtime_context import (
    assert_published,
    get_context,
    get_device,
    get_disagg,
    get_exec,
    get_lora,
    get_memory,
    get_mm,
    get_model,
    get_observability,
    get_parallel,
    get_schedule,
    get_serving,
    get_spec,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import (
    PortArgs,
    ServerArgs,
)
from sglang.srt.utils import (
    configure_gc_warning,
    freeze_gc,
    get_bool_env_var,
    get_or_create_event_loop,
    kill_process_tree,
)
from sglang.srt.utils.aio_rwlock import RWLock
from sglang.srt.utils.cuda_vmm_transport_utils import CudaVmmFeatureTransport
from sglang.srt.utils.cudacore_pyspy_dump_utils import (
    collect_scheduler_processes,
    pyspy_dump_schedulers,
    trigger_cuda_user_coredump,
)
from sglang.srt.utils.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
    resolve_image_processor_backend,
)
from sglang.srt.utils.network import get_zmq_socket
from sglang.srt.utils.request_logger import RequestLogger
from sglang.srt.utils.watchdog import Watchdog
from sglang.srt.utils.weight_versions import add_weight_versions_to_meta_info
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# 中译：单个请求等待响应时，每次 await 的超时时间（秒）。超时后会检查客户端是否断连，
#       未断连则继续等待（见 _wait_one_response）。
_REQUEST_STATE_WAIT_TIMEOUT = envs.SGLANG_REQUEST_STATE_WAIT_TIMEOUT.get()

logger = logging.getLogger(__name__)

# 中译：增量流式输出时，meta_info 里这几类 logprob 字段也是「增量」的（只含本次新增部分），

def _reject_missing_dispatched_encoder_embedding(request_obj, mm_inputs):
    """Do not silently turn a failed EPD request into local vision work."""
    disagg = get_disagg()
    if (
        mm_inputs is None
        and disagg.language_only
        and disagg.encoder_transfer_backend == "zmq_to_tokenizer"
        and request_obj.need_wait_for_mm_inputs
    ):
        raise fastapi.HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail=(
                "The encoder did not return multimodal embeddings. "
                "The request was not run locally in language-only mode."
            ),
        )
@lru_cache(maxsize=1)
def _ragged_verify_cap_accept() -> bool:
    # The mode env is fixed at server launch; cache to keep it off the
    # per-request metrics path.
    from sglang.srt.speculative.ragged_verify import (
        RaggedVerifyMode,
        read_ragged_verify_mode,
    return read_ragged_verify_mode() is RaggedVerifyMode.CAP_ACCEPT
_INCREMENTAL_STREAMING_META_INFO_KEYS = (
    "output_token_logprobs",
    "output_top_logprobs",
    "output_token_ids_logprobs",
    "output_token_sampling_mask",
    "output_token_sampling_logprobs",
)


@dataclasses.dataclass
class ReqState:
    """Store the state a request.

    中译：保存单个请求在 TokenizerManager 侧的完整运行时状态，是 rid_to_state 字典的值。
    关键字段：
    - out_list：尚未被 _wait_one_response 取走的输出片段队列（生产者 handle_loop 写入，消费者读取）。
    - finished：该请求是否已结束（收到 finish_reason）。
    - event：asyncio.Event，handle_loop 收到新输出后 set，唤醒等待中的 _wait_one_response。
    - obj：原始请求对象（GenerateReqInput / EmbeddingReqInput）。
    - text / text_chunks：惰性累积的输出文本——新片段先入 chunks，需要完整文本时再 join，
      避免每步都重建整串导致 O(n^2) 开销（与 DecodeStatus 的设计同理）。
    - output_ids 及各类 logprob 列表：增量累积的 token id 与对数概率。
    """

    out_list: List[Dict[Any, Any]]
    finished: bool
    event: asyncio.Event
    obj: Union[GenerateReqInput, EmbeddingReqInput]

    # For performance metrics
    time_stats: APIServerReqTimeStats
    last_completion_tokens: int = 1
    ttft_observed: bool = False

    # For streaming output
    last_output_offset: int = 0

    # Accumulate text lazily so incremental streaming can emit the incoming
    # delta directly without rebuilding the full output prefix.
    text: str = ""
    text_chunks: List[str] = dataclasses.field(default_factory=list)

    def append_text(self, chunk: str):
        # 中译：追加一段新文本到 chunks 暂存列表，不立即拼接（延迟合并以省开销）。
        if chunk:
            self.text_chunks.append(chunk)

    def get_text(self) -> str:
        # 中译：取完整输出文本——此刻才把暂存的 chunks 一次性 join 进 text 并清空。
        if self.text_chunks:
            self.text += "".join(self.text_chunks)
            self.text_chunks.clear()
        return self.text

    def get_crash_dump_output(self) -> Dict[Any, Any]:
        # 中译：崩溃转储时构造该请求当前已产出的部分结果（文本 + output_ids）。
        out = {}
        if self.text or self.text_chunks:
            out["text"] = self.get_text()
        if self.output_ids:
            out["output_ids"] = self.output_ids.copy()
        return out

    # For incremental state update.
    # TODO(lianmin): do not initialize some lists if not needed.
    output_ids: List[int] = dataclasses.field(default_factory=list)
    input_token_logprobs_val: List[float] = dataclasses.field(default_factory=list)
    input_token_logprobs_idx: List[int] = dataclasses.field(default_factory=list)
    output_token_logprobs_val: List[float] = dataclasses.field(default_factory=list)
    output_token_logprobs_idx: List[int] = dataclasses.field(default_factory=list)
    input_top_logprobs_val: List[List[float]] = dataclasses.field(default_factory=list)
    input_top_logprobs_idx: List[List[int]] = dataclasses.field(default_factory=list)
    output_top_logprobs_val: List[List[float]] = dataclasses.field(default_factory=list)
    output_top_logprobs_idx: List[List[int]] = dataclasses.field(default_factory=list)
    input_token_ids_logprobs_val: List = dataclasses.field(default_factory=list)
    input_token_ids_logprobs_idx: List = dataclasses.field(default_factory=list)
    output_token_ids_logprobs_val: List = dataclasses.field(default_factory=list)
    output_token_ids_logprobs_idx: List = dataclasses.field(default_factory=list)
    output_token_sampling_mask: List = dataclasses.field(default_factory=list)
    output_token_sampling_logprobs: List = dataclasses.field(default_factory=list)

    # Cached flat-format prompt top logprob fields; rebuilt only when more
    # prefill chunks arrive, so streaming decode chunks reuse the payload.
    input_top_logprobs_flat_fields: Optional[Dict[str, Any]] = None
    input_top_logprobs_flat_num_rows: int = -1
    # Scheduler-assembled flat arrays (val float32 [rows, k], idx int32
    # [rows, k], null_prefix), sent once at prefill completion. When present,
    # the nested input_top_logprobs_val/idx above stay empty.
    input_top_logprobs_scheduler_flat: Optional[Tuple[np.ndarray, np.ndarray, int]] = (
        None
    )

    # For detokenized logprobs
    input_token_logprobs: List[Any] = dataclasses.field(default_factory=list)
    output_token_logprobs: List[Any] = dataclasses.field(default_factory=list)
    input_top_logprobs: List[Any] = dataclasses.field(default_factory=list)
    output_top_logprobs: List[Any] = dataclasses.field(default_factory=list)
    input_token_ids_logprobs: List[Any] = dataclasses.field(default_factory=list)
    output_token_ids_logprobs: List[Any] = dataclasses.field(default_factory=list)
    customized_info_accumulated: Dict[str, List[Any]] = dataclasses.field(
        default_factory=dict
    )

    # For return_prompt_token_ids: stores prompt token IDs captured after tokenization
    prompt_token_ids: Optional[List[int]] = None


def _slice_streaming_output_meta_info(
    meta_info: Dict[Any, Any],
    last_output_offset: int,
    customized_info_keys: Optional[Iterable[str]] = None,
) -> None:
    """Align output-side metadata with the current incremental streaming chunk.

    中译：增量流式输出时，把 meta_info 中的增量字段裁剪为「本次新增」的部分——
          即从 last_output_offset（上一批已发送到的偏移）切到末尾，避免重复下发历史 logprob。
    """
    streaming_meta_info_keys = set(_INCREMENTAL_STREAMING_META_INFO_KEYS)
    if customized_info_keys is not None:
        streaming_meta_info_keys.update(customized_info_keys)
    for key in meta_info.keys() & streaming_meta_info_keys:
        meta_info[key] = meta_info[key][last_output_offset:]


def _build_flat_input_top_logprobs_fields(
    input_top_logprobs_val: List[Optional[List[float]]],
    input_top_logprobs_idx: List[Optional[List[int]]],
    top_logprobs_num: int,
    return_b64: bool = False,
) -> Dict[str, Any]:
    """Build the flat raw prompt top logprob response fields.

    `input_top_logprobs_shape` is the literal [rows, k] shape of the flat
    arrays. The leading null positions (counted by
    `input_top_logprobs_null_prefix`) precede the arrays, so covered position
    i, entry j lives at flat[(i - null_prefix) * k + j] and the covered range
    spans null_prefix + rows positions. With ``return_b64``, the arrays are
    base64 contiguous little-endian binary; the dtype marker fields let the
    widths change later without a wire break.
    """
    val_arr, idx_arr, null_prefix = build_flat_input_top_logprobs_arrays(
        input_top_logprobs_val, input_top_logprobs_idx, top_logprobs_num
    )
    if return_b64:
        return _build_flat_input_top_logprobs_fields_from_arrays(
            val_arr, idx_arr, null_prefix, return_b64=True
        )
    # Flatten the original python rows so the JSON numbers keep their full
    # (float64) precision, matching the pre-scheduler-flat output.
    return {
        "input_top_logprobs_val_flat": [
            v for row in input_top_logprobs_val[null_prefix:] for v in row
        ],
        "input_top_logprobs_idx_flat": [
            i for row in input_top_logprobs_idx[null_prefix:] for i in row
        ],
        "input_top_logprobs_shape": [val_arr.shape[0], val_arr.shape[1]],
        "input_top_logprobs_null_prefix": null_prefix,
    }


def _build_flat_input_top_logprobs_fields_from_arrays(
    val_arr: np.ndarray,
    idx_arr: np.ndarray,
    null_prefix: int,
    return_b64: bool = False,
) -> Dict[str, Any]:
    """Build the flat response fields from scheduler-assembled [rows, k]
    arrays (see `_build_flat_input_top_logprobs_fields` for the field
    semantics)."""
    fields: Dict[str, Any] = {}
    if return_b64:
        fields["input_top_logprobs_val_flat_b64"] = pybase64.b64encode(
            val_arr.tobytes()
        ).decode("utf-8")
        fields["input_top_logprobs_idx_flat_b64"] = pybase64.b64encode(
            idx_arr.tobytes()
        ).decode("utf-8")
        fields["input_top_logprobs_val_flat_b64_dtype"] = "float32"
        fields["input_top_logprobs_idx_flat_b64_dtype"] = "int32"
    else:
        fields["input_top_logprobs_val_flat"] = val_arr.reshape(-1).tolist()
        fields["input_top_logprobs_idx_flat"] = idx_arr.reshape(-1).tolist()
    fields["input_top_logprobs_shape"] = [val_arr.shape[0], val_arr.shape[1]]
    fields["input_top_logprobs_null_prefix"] = null_prefix
    return fields


class InputFormat(Enum):
    """Input format types for tokenization handling.

    中译：分词处理时的输入格式枚举，用于区分三种文本输入形态、走不同的分词与结果提取路径。
    """

    SINGLE_STRING = 1  # Regular single text like "Hello world"
    # 中译：单条文本，如 "Hello world"。
    BATCH_STRINGS = 2  # Regular batch like ["Hello", "World"]
    # 中译：常规批量文本，如 ["Hello", "World"]。
    CROSS_ENCODER_PAIRS = 3  # Cross-encoder pairs like [["query", "document"]]
    # 中译：cross-encoder 句对（用于相似度/排序），如 [["query", "document"]]。


_MANAGER_OWNED_FIELDS = ("model_path", "served_model_name")


class TokenizerManager(TokenizerControlMixin, TokenizerManagerScoreMixin):
    """TokenizerManager is a process that tokenizes the text.

    中译：分词管理器主类。混入 TokenizerControlMixin（控制类接口，如 abort/会话等）与
          TokenizerManagerScoreMixin（打分接口）。__init__ 把各初始化逻辑拆成多个 init_* 方法，
          便于子类按需覆盖（参见 large-class-init-style 约定）。
    """

    @property
    def serving_chat_class(self):
        """Return the serving chat class for OpenAI API.

        Override in subclass to provide custom serving behavior.

        中译：返回 OpenAI Chat API 的服务类；子类可覆盖以提供自定义行为。
              用 property + 延迟 import 避免模块加载期的循环依赖。
        """
        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

        return OpenAIServingChat

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        *,
        start_pd_bootstrap_service: bool = True,
    ):
        # 中译：构造分为多个 init_* 子步骤，顺序有依赖（如 model_config 须先于 tokenizer/metrics）。
        # Parse args
        # 中译：解析并缓存常用 server_args 字段。
        self.server_args = server_args
        assert_published(server_args, role="tokenizer")
        self.startup_time: Optional[Dict[str, Any]] = None
        self.elastic_worker_count = get_parallel().dp_size
        self.elastic_pending_ep_size = None
        self.elastic_scale_phase = "idle"
        self.elastic_last_error = None
        self.enable_metrics = get_observability().enable_metrics
        self.incremental_streaming_output = get_serving().incremental_streaming_output
        self.enable_lora = get_lora().enable_lora
        self.enable_trace = get_observability().enable_trace
        self.allow_auto_truncate = get_serving().allow_auto_truncate
        self.skip_tokenizer_init = get_serving().skip_tokenizer_init
        self.preferred_sampling_params = get_serving().preferred_sampling_params
        self.crash_dump_folder = get_observability().crash_dump_folder

        # Init model config
        # 中译：读取模型配置（上下文长度、词表大小、是否生成模型、推测解码预留 token 等）。
        self.init_model_config()
        self._validate_cuda_vmm_feature_transport_support()

        # Initialize tokenizer and multimodalprocessor
        # 中译：初始化分词器与多模态处理器（含动态批量分词器）。
        self.init_tokenizer_and_processor()

        # Init inter-process communication
        # 中译：初始化 ZMQ 进程间通信通道（收 detokenizer 输出、发请求给 scheduler）。
        self.init_ipc_channels(port_args)

        # Init running status
        # 中译：初始化运行时状态（rid_to_state、事件循环句柄、暂停标志等）。
        self.init_running_status()

        # Init logging and dumping
        # 中译：初始化请求日志与转储（dump）相关设施。
        self.init_request_logging_and_dumping()

        # Init weight update
        # 中译：初始化权重热更新相关状态（读写锁、结果 future、暂停条件变量）。
        self.init_weight_update()

        # Init LoRA status
        # 中译：初始化 LoRA 适配器注册表与缓存。
        self.init_lora()

        # Init PD disaggregation and encoder disaggregation
        # 中译：初始化 PD（prefill/decode）分离与编码器（encoder）分离相关组件。
        self.init_disaggregation(start_pd_bootstrap_service=start_pd_bootstrap_service)

        # Init metric collector and watchdog
        # 中译：初始化指标采集器与（软）看门狗。
        self.init_metric_collector_watchdog()

        # Init request dispatcher
        # 中译：初始化结果分发器（按消息类型路由收到的输出/控制消息）。
        self.init_request_dispatcher()

        # Construct this last so later initialization failures cannot orphan
        # the transport's recycler thread.
        self.cuda_vmm_feature_transport = CudaVmmFeatureTransport(
            self.server_args, self.mm_processor
        )

    def init_model_config(self):
        # 中译：读取模型配置。其中推测解码（speculative，eagle 系列）需在输出槽位预留
        #       draft token 空间，故计算 num_reserved_tokens 用于后续长度校验。
        server_args = self.server_args
        model_config_class = getattr(self, "model_config_class", ModelConfig)

        # Read model args
        self.model_path = get_model().model_path
        self.served_model_name = get_serving().served_model_name
        self.model_config = model_config_class.from_server_args(server_args)
        self.is_generation = self.model_config.is_generation
        self.context_len = self.model_config.context_len
        self.image_token_id = self.model_config.image_token_id
        self.max_req_input_len = None  # Will be set later in engine.py
            # 中译：当前 eagle 实现把 draft token 存在输出 token 槽位里，所以要为 draft token 预留空间。
        self.enable_priority_scheduling = get_schedule().enable_priority_scheduling
        self.default_priority_value = get_schedule().default_priority_value
        self.num_reserved_tokens = compute_num_reserved_tokens()
        self.validate_total_tokens = True

    def init_tokenizer_and_processor(self):
        # 中译：初始化分词器与（多模态）处理器。
        #       多模态模型需额外加载 mm_processor 处理图像/音频/视频；
        #       skip_tokenizer_init 时只置 None（调用方自带 token id）。
        server_args = self.server_args

        # Initialize tokenizer and processor
        if self.model_config.is_multimodal and not get_disagg().language_model_only:
            import_processors("sglang.srt.multimodal.processors")
            if mm_process_pkg := envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE.get():
                import_processors(mm_process_pkg, overwrite=True)
            _processor = get_processor_wrapper()
            transport_mode = determine_tensor_transport_mode()

            # We want to parallelize the image pre-processing so we create an executor for it
            # We create mm_processor for any skip_tokenizer_init to make sure we still encode
            # images even with skip_tokenizer_init=False.
            self.mm_processor = get_mm_processor(
                self.model_config.hf_config,
                server_args,
                _processor,
                transport_mode,
                model_config=self.model_config,
            )

            if get_serving().skip_tokenizer_init:
                self.tokenizer = self.processor = None
            else:
                self.processor = _processor
                self.tokenizer = get_tokenizer_from_processor(self.processor)
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
        else:
            self.mm_processor = self.processor = None

            if get_serving().skip_tokenizer_init:
                self.tokenizer = None
            else:
                self.tokenizer = get_tokenizer(
                    get_serving().tokenizer_path,
                    tokenizer_mode=get_serving().tokenizer_mode,
                    trust_remote_code=get_model().trust_remote_code,
                    revision=get_model().revision,
                    tokenizer_backend=get_serving().tokenizer_backend,
                )

        # Initialize async dynamic batch tokenizer if enabled (common for both multimodal and non-multimodal)
        # 中译：若启用动态批量分词器，则把单条文本的分词请求在后台聚合成 batch 提交，提升吞吐
        #       （仅对单条文本路径生效，见 _tokenize_texts）。
        if (
            get_serving().enable_dynamic_batch_tokenizer
            and not get_serving().skip_tokenizer_init
        ):
            self.async_dynamic_batch_tokenizer = AsyncDynamicbatchTokenizer(
                self.tokenizer,
                max_batch_size=get_serving().dynamic_batch_tokenizer_batch_size,
                batch_wait_timeout_s=get_serving().dynamic_batch_tokenizer_batch_timeout,
            )
        else:
            self.async_dynamic_batch_tokenizer = None

    def _validate_cuda_vmm_feature_transport_support(self) -> None:
        if get_mm().mm_feature_transport != "cuda_vmm":
            return

        from sglang.srt.model_loader.utils import get_model_architecture

        model_class, _ = get_model_architecture(self.model_config)
        if not getattr(model_class, "supports_cuda_vmm_feature_transport", False):
            raise ValueError(
                "--mm-feature-transport=cuda_vmm is not supported by model class "
                f"{model_class.__name__}"
            )

    def init_ipc_channels(self, port_args: PortArgs):
        # 中译：建立 ZMQ 异步通信。recv_from_detokenizer（PULL）收回反向解码后的输出；
        #       send_to_scheduler（PUSH）把分词后的请求发给调度器。
        #       多 tokenizer worker 模式下改用 SenderWrapper，给每个请求带上 tokenizer 的
        #       回程 ipc 名，便于结果按 worker 路由。
        context = zmq.asyncio.Context(2)
        self.recv_from_detokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.tokenizer_ipc_name, True
        )
        if get_serving().tokenizer_worker_num == 1:
            self.send_to_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.scheduler_input_ipc_name, True
            )
            self.tokenizer_ipc_name = None
        else:
            # Use tokenizer_worker_ipc_name in multi-tokenizer mode
            # 中译：多 tokenizer 模式下使用 tokenizer_worker_ipc_name。
            self.send_to_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_worker_ipc_name, False
            )
            self.tokenizer_ipc_name = port_args.tokenizer_ipc_name

        self.load_snapshot_reader = create_load_snapshot_reader(
            port_args,
            caller="TokenizerManager",
        )

    def _dispatch_to_scheduler(self, obj: Any) -> None:
        if self.tokenizer_ipc_name is not None:
            stamp_http_worker_ipc(obj, self.tokenizer_ipc_name)
        sock_send(self.send_to_scheduler, obj)

    async def _async_dispatch_to_scheduler(self, obj: Any) -> None:
        if self.tokenizer_ipc_name is not None:
            stamp_http_worker_ipc(obj, self.tokenizer_ipc_name)
        await async_sock_send(self.send_to_scheduler, obj)

    def init_running_status(self):
        # Request states
        # 中译：rid_to_state 是核心映射：请求 id -> ReqState。入口写入、handle_loop 写输出、
        #       _wait_one_response 读取并在结束时删除。是「ZMQ 单向流」转「按请求并发」的枢纽。
        self.rid_to_state: Dict[str, ReqState] = {}
        # 中译：event_loop / asyncio_tasks——handle_loop 等后台协程的事件循环句柄与任务集合。
        self.event_loop = None
        self.asyncio_tasks = set()

        # Health check
        # 中译：服务健康状态与优雅退出标志；last_receive_tstamp 记录最近一次收到输出的时间。
        self.server_status = ServerStatus.Starting
        self.gracefully_exit = False
        self.last_receive_tstamp = real_time()

        # Session
        self.session_futures = {}  # session_id -> asyncio event
        # 中译：会话 id -> future，用于异步等待开/关会话的响应。

        # Subprocess liveness watchdog — set by Engine or http_server after construction
        # 中译：子进程存活看门狗，由 Engine 或 http_server 在构造后注入（这里先置 None）。
        self._subprocess_watchdog = None

    def init_request_logging_and_dumping(self):
        # TODO: Refactor and organize the log export code.
        # Request logging
        self.request_logger = RequestLogger(
            log_requests=get_observability().log_requests,
            log_requests_level=get_observability().log_requests_level,
            log_requests_format=get_observability().log_requests_format,
            log_requests_target=get_observability().log_requests_target,
        )

        # Dumping
        self.dump_requests_folder = ""  # By default do not dump
        self.dump_requests_threshold = 1000
        self.dump_requests_exclude_meta_keys: List[str] = [
            "routed_experts",
            "hidden_states",
        ]
        self.dump_request_list: List[Tuple] = []
        self.crash_dump_request_list: deque[Tuple] = deque()
        self.crash_dump_performed = False  # Flag to ensure dump is only called once

        # Initialize performance metrics loggers with proper skip names
        _, obj_skip_names, out_skip_names = self.request_logger.metadata
        self.request_metrics_exporter_manager = RequestMetricsExporterManager(
            self.server_args, obj_skip_names, out_skip_names
        )

    def init_weight_update(self):
        # Initial weights status
        # 中译：初始权重是否已就绪。开启 checkpoint_engine 等待权重时，初始置未就绪，
        #       待权重同步完成再放行请求。
        self.initial_weights_loaded = True
        if get_model().checkpoint_engine_wait_weights_before_ready:
            self.initial_weights_loaded = False

        # Weight updates
        # The event to notify the weight sync is finished.
        # 中译：权重热更新的并发控制核心——读写锁 model_update_lock：
        #       普通推理请求持「读锁」(可并发)，权重更新持「写锁」(独占)，
        #       从而保证更新权重时没有 in-flight 请求在用旧权重。
        #       model_update_result 是等待更新结果的 future。
        self.model_update_lock = RWLock()
        self.model_update_result: Optional[Awaitable[UpdateWeightFromDiskReqOutput]] = (
            None
        )
        # 中译：is_pause + is_pause_cond——暂停/继续生成的标志与条件变量；
        self.model_update_expected_workers = self.elastic_worker_count
        self.model_update_tmp: List[UpdateWeightFromDiskReqOutput] = []
        self.is_pause = False
        self.is_pause_cond = asyncio.Condition()

    def init_lora(self):
        # 中译：初始化 LoRA。注意 lora_update_lock 不阻塞推理（与 model_update_lock 不同），
        #       允许 LoRA 加载/卸载与推理重叠；lora_ref_cache 缓存曾加载过的适配器引用，
        #       以便运行期被淘汰后按需重新动态加载。
        # LoRA
        # Initialize the `LoRARegistry` with initial LoRA adapter paths provided in `server_args`.
        # The registry dynamically updates as adapters are loaded / unloaded during runtime. It
        # serves as the source of truth for available adapters and maps user-friendly LoRA names
        # to internally used unique LoRA IDs.
        self.lora_registry = LoRARegistry(get_lora().lora_paths)
        # Lock to serialize LoRA update operations.
        # Please note that, unlike `model_update_lock`, this does not block inference, allowing
        # LoRA updates and inference to overlap.
        self.lora_update_lock = asyncio.Lock()
        # A cache for mapping the lora_name for LoRA adapters that have been loaded at any
        # point to their latest LoRARef objects, so that they can be
        # dynamically loaded if needed for inference
        self.lora_ref_cache: Dict[str, LoRARef] = {}
        if get_lora().lora_paths is not None:
            for lora_ref in get_lora().lora_paths:
                self.lora_ref_cache[lora_ref.lora_name] = lora_ref

        # 中译：初始化分离式部署。PD 分离把 prefill 与 decode 拆到不同实例；
    def init_disaggregation(self, *, start_pd_bootstrap_service: bool = True):
        # PD Disaggregation
        self.disaggregation_mode = DisaggregationMode(get_disagg().disaggregation_mode)
        # Keep a reference so the bootstrap server is not garbage-collected.
        self.bootstrap_server = (
            start_disagg_service() if start_pd_bootstrap_service else None
        )
        # Single-source counter for auto-assigning fake bootstrap_room.
        self.fake_bootstrap_room_counter = 0

        # Encoder Disaggregation
        self.encoder_bootstrap_server = None
        if get_disagg().language_only:
            from sglang.srt.disaggregation.encoder.receiver import (
                EncoderBootstrapServer,
            )

            # Shared mutable URL list: the bootstrap server appends / removes
            # entries as encoders register, the receiver reads from the same
            # list.  Pre-populated with static --encoder-urls so the legacy
            # CLI flag still works (alongside dynamic registrations).
            self.encoder_urls: List[str] = list(get_disagg().encoder_urls)
            self.encoder_bootstrap_server = EncoderBootstrapServer(
                host=get_serving().host,
                port=get_disagg().encoder_bootstrap_port,
                urls=self.encoder_urls,
            )
            self.mm_receiver = create_mm_receiver(
                self.server_args,
                dtype=self.model_config.dtype,
                hf_config=self.model_config.hf_config,
                encode_urls=self.encoder_urls,
            )

    def init_metric_collector_watchdog(self):
        # 中译：初始化指标采集器（Prometheus 风格 labels）与软看门狗（soft watchdog，
        #       检测进程卡死，soft 模式下只告警）。仅在 enable_metrics 时构建采集器。
        # Metrics
        if self.enable_metrics:
            engine_type = DisaggregationMode.to_engine_type(
                get_disagg().disaggregation_mode
            )

            labels = {
                "model_name": get_serving().served_model_name,
                "engine_type": engine_type,
            }
            if self.enable_priority_scheduling:
                labels["priority"] = ""
            if get_observability().tokenizer_metrics_allowed_custom_labels:
                for (
                    label
                ) in get_observability().tokenizer_metrics_allowed_custom_labels:
                    labels[label] = ""
            if get_observability().extra_metric_labels:
                labels.update(get_observability().extra_metric_labels)
            tokenizer_collector_cls = resolve_collector_class(
                STAT_LOGGER_ROLE_TOKENIZER,
                TokenizerMetricsCollector,
            )
            self.metrics_collector = tokenizer_collector_cls(
                server_args=self.server_args,
                labels=labels,
                bucket_time_to_first_token=get_observability().bucket_time_to_first_token,
                bucket_e2e_request_latency=get_observability().bucket_e2e_request_latency,
                bucket_inter_token_latency=get_observability().bucket_inter_token_latency,
            )

            start_cpu_monitor_thread("tokenizer")

        if get_observability().gc_warning_threshold_secs > 0.0:
            configure_gc_warning(get_observability().gc_warning_threshold_secs)
        self.soft_watchdog = Watchdog.create(
            debug_name="TokenizerManager",
            watchdog_timeout=get_device().soft_watchdog_timeout,
            soft=True,
            test_stuck_time=envs.SGLANG_TEST_STUCK_TOKENIZER.get(),
        )

    def set_startup_time(self, startup_time: Dict[str, Any]) -> None:
        self.startup_time = startup_time
        if self.enable_metrics:
            self.metrics_collector.emit_startup_time(startup_time)

    def init_request_dispatcher(self):
        # 中译：构造「基于类型的分发器」，把从 detokenizer 收到的「非批量输出」类消息
        #       （中止、开会话结果、权重更新结果、健康检查等）路由到对应 handler。
        #       批量输出（BatchStrOutput 等）则由 handle_loop 直接交给 _handle_batch_output。
        self._result_dispatcher = TypeBasedDispatcher(
            [
                (AbortReq, self._handle_abort_req),
                (OpenSessionReqOutput, self._handle_open_session_req_output),
                (
                    UpdateWeightFromDiskReqOutput,
                    self._handle_update_weights_from_disk_req_output,
                ),
                (FreezeGCReq, lambda x: None),
                # For handling case when scheduler skips detokenizer and forwards back to the tokenizer manager, we ignore it.
                # 中译：调度器跳过 detokenizer 直接把健康检查结果回传时，这里直接忽略。
                (HealthCheckOutput, lambda x: None),
                # Same skip-detokenizer forwarding case as above.
                (ConfigureLoggingReq, lambda x: None),
                (ActiveRanksOutput, self.update_active_ranks),
                (ElasticScaleUpdateReq, self.forward_elastic_scale_update),
            ]
        )
        self.init_communicators()

        self.sampling_params_class = SamplingParams
        self.signal_handler_class = SignalHandler

    async def generate_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        # 中译：核心入口——异步生成器。每个 HTTP/Engine 请求都从这里进入，逐个 yield 输出片段。
        #       流程：确保后台 handle_loop 已启动 -> 归一化参数 -> 校验 -> 暂停门控 ->
        #       持「读锁」(与权重更新互斥) -> 分词并发给调度器 -> 等待并 yield 各响应。
        # 中译：首次调用时惰性创建后台事件循环任务（handle_loop / sigterm_watchdog）。
        self.auto_create_handle_loop()

        # Normalize the request
        # 中译：归一化批量与各项参数（把单条/批量统一成可索引的形态）。
        obj.normalize_batch_and_arguments()
        self._set_default_priority(obj)
        if (
            isinstance(obj, GenerateReqInput)
            and obj.max_thinking_tokens is not None
            and not get_serving().enable_strict_thinking
        ):
            raise ValueError(
                "max_thinking_tokens requires the server to be launched with "
                "--enable-strict-thinking"
            )

        if isinstance(obj, GenerateReqInput) and obj.routed_dp_rank is not None:
            dp_size = self.elastic_worker_count
            if dp_size <= 1 and obj.routed_dp_rank == 0:
                logger.debug(
                    f"routed_dp_rank={obj.routed_dp_rank} is ignored because dp_size={dp_size}"
                )
            elif obj.routed_dp_rank < 0 or obj.routed_dp_rank >= dp_size:
                raise ValueError(
                    f"routed_dp_rank={obj.routed_dp_rank} out of range [0, {dp_size})"
                )

        # 中译：为请求在 rid_to_state 中登记 ReqState（含 event、时间统计等）。
        self._init_req_state(obj, request)
        try:
            if get_disagg().language_only:
                self._handle_epd_disaggregation_encode_request(obj)

            # Log the request
            self.request_logger.log_received_request(obj, self.tokenizer, request)

        # 中译：暂停门控——若处于暂停态，挂起在条件变量上直到 continue_generation 唤醒。
        # 中译：持「读锁」整个生成过程。权重更新持写锁，故此处保证不会在权重更新中途用到半新半旧权重。
            # 中译：单条请求：分词 -> 发给调度器 -> 逐个 yield 响应；批量请求转交 _handle_batch_request。
            async with self.is_pause_cond:
                await self.is_pause_cond.wait_for(lambda: not self.is_pause)

            async with self.model_update_lock.reader_lock:
                await self._validate_and_resolve_lora(obj)
                # Tokenize the request and send it to the scheduler
                if obj.is_single:
                    tokenized_obj = await self._tokenize_one_request(obj)
                    state = self.rid_to_state[obj.rid]
                    if obj.return_prompt_token_ids:
                        state.prompt_token_ids = list(tokenized_obj.input_ids)
                    self._send_one_request(tokenized_obj)
                    async for response in self._wait_one_response(obj, request):
                        yield response
                else:
                    async for response in self._handle_batch_request(obj, request):
        except BaseException:
            # _init_req_state created a rid_to_state entry per (sub-)request up
            # front. The normal remover is the scheduler-response path
            # (_handle_batch_output), so a failure *before* a request reaches the
            # scheduler -- e.g. input-length validation rejecting an over-context
            # request -- would otherwise leak those entries forever. Drop any that
            # are still pending; entries already removed on the normal completion
            # path are left untouched (pop is a no-op).
            self._discard_pending_req_states(obj)
            raise

    def _detect_input_format(
        self, texts: Union[str, List[str]], is_cross_encoder: bool
    ) -> InputFormat:
        if isinstance(texts, str):
            return InputFormat.SINGLE_STRING

        if (
            is_cross_encoder
            and len(texts) > 0
            and isinstance(texts[0], list)
            and len(texts[0]) == 2
        ):
            return InputFormat.CROSS_ENCODER_PAIRS

        return InputFormat.BATCH_STRINGS

    def _prepare_tokenizer_input(
        self, texts: Union[str, List[str]], input_format: InputFormat
    ) -> Union[List[str], List[List[str]]]:
        """Prepare input for the tokenizer based on detected format.

        中译：按检测出的格式，把输入整理成分词器期望的统一形态（单条字符串需包成 batch）。
        """
        if input_format == InputFormat.SINGLE_STRING:
            return [texts]  # Wrap single string for batch processing
        elif input_format == InputFormat.CROSS_ENCODER_PAIRS:
            return texts  # Already in correct format: [["query", "doc"]]
        else:  # BATCH_STRINGS
            return texts  # Already in correct format: ["text1", "text2"]

    def _extract_tokenizer_results(
        self,
        input_ids: List[List[int]],
        token_type_ids: Optional[List[List[int]]],
        input_format: InputFormat,
        original_batch_size: int,
    ) -> Union[
        Tuple[List[int], Optional[List[int]]],
        Tuple[List[List[int]], Optional[List[List[int]]]],
    ]:
        """Extract results from tokenizer output based on input format.

        中译：按输入格式从分词器输出中提取结果。单条输入（含单个 cross-encoder 句对）时
              解包出第一个元素返回扁平结果；真正的批量则原样返回。
        """

        # For single inputs (string or single cross-encoder pair), extract first element
        if (
            input_format in [InputFormat.SINGLE_STRING, InputFormat.CROSS_ENCODER_PAIRS]
            and original_batch_size == 1
        ):
            single_input_ids = input_ids[0] if input_ids else []
            single_token_type_ids = token_type_ids[0] if token_type_ids else None
            return single_input_ids, single_token_type_ids

        # For true batches, return as-is
        return input_ids, token_type_ids

    async def _tokenize_texts(
        self, texts: Union[str, List[str]], is_cross_encoder: bool = False
    ) -> Union[
        Tuple[List[int], Optional[List[int]]],
        Tuple[List[List[int]], Optional[List[List[int]]]],
    ]:
        if not texts or self.tokenizer is None:
            raise ValueError("texts cannot be empty and tokenizer must be initialized")

        # Step 1: Detect input format and prepare for tokenization
        input_format = self._detect_input_format(texts, is_cross_encoder)
        tokenizer_input = self._prepare_tokenizer_input(texts, input_format)
        original_batch_size = len(texts) if not isinstance(texts, str) else 1

        # Step 2: Set up tokenizer arguments
        tokenizer_kwargs = (
            {"return_token_type_ids": is_cross_encoder} if is_cross_encoder else {}
        )

        # Step 3: Choose tokenization strategy
        use_async_tokenizer = (
            self.async_dynamic_batch_tokenizer is not None
            and input_format == InputFormat.SINGLE_STRING
        )

        if use_async_tokenizer:
            logger.debug("Using async dynamic batch tokenizer for single text")
            result = await self.async_dynamic_batch_tokenizer.encode(
                tokenizer_input[0], **tokenizer_kwargs
            )
            # Convert to batch format for consistency
            input_ids = [result["input_ids"]]
            token_type_ids = (
                [result["token_type_ids"]]
                if is_cross_encoder and result.get("token_type_ids")
                else None
            )
        else:
            logger.debug(f"Using regular tokenizer for {len(tokenizer_input)} inputs")

            if not is_cross_encoder and (not getattr(self.tokenizer, "is_fast", False)):
                input_ids = [self.tokenizer.encode(t) for t in tokenizer_input]
                token_type_ids = None
            else:
                encoded = self.tokenizer(tokenizer_input, **tokenizer_kwargs)
                input_ids = encoded["input_ids"]
                token_type_ids = (
                    encoded.get("token_type_ids") if is_cross_encoder else None
                )

        # vLLM's OpenAI embeddings endpoint includes special tokens for
        # encoder models. EmbeddingGemma's restored Gemma tokenizer adds BOS
        # but, by its checkpoint default, omits EOS. Add EOS explicitly here
        # rather than mutating tokenizer-global post-processing state.
        if (
            self.model_config.is_embedding_gemma
            and self.tokenizer.eos_token_id is not None
        ):
            input_ids = [
                (
                    ids
                    if ids and ids[-1] == self.tokenizer.eos_token_id
                    else [*ids, self.tokenizer.eos_token_id]
                )
                for ids in input_ids
            ]

        # Step 4: Extract results based on input format
        return self._extract_tokenizer_results(
            input_ids, token_type_ids, input_format, original_batch_size
        )

    async def _tokenize_one_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
    ):
        """Tokenize one request.

        中译：对单个请求做完整的「输入准备」：
              1) 确定 input_ids 来源——直接给的 input_embeds / input_ids，或对 text 分词；
              2) 多模态请求交给 mm_processor / mm_receiver 处理，得到含占位 token 的 input_ids
                 与多模态特征（mm_inputs），并处理调用方自带的 mm_hashes（用于前缀缓存对齐）；
              3) 长度等校验后，封装成 Tokenized* 请求对象返回。
        """
        # Tokenize
        input_embeds = None
        input_text = obj.text
        token_type_ids = None
        is_cross_encoder_request = (
            isinstance(obj, EmbeddingReqInput) and obj.is_cross_encoder_request
        )
        if obj.input_embeds is not None:
            # 中译：直接传 input_embeds 时无法做前缀（radix）缓存命中，必须先关闭 radix cache。
            if not get_memory().disable_radix_cache:
                raise ValueError(
                    "input_embeds is provided while disable_radix_cache is False. "
                    "Please add `--disable-radix-cache` when you launch the server "
                    "if you want to use input_embeds as inputs."
                )
            input_embeds = obj.input_embeds
            input_ids = obj.input_ids
        elif obj.input_ids is not None:
            input_ids = obj.input_ids
        else:
            if self.tokenizer is None:
                raise ValueError(
                    "The engine initialized with skip_tokenizer_init=True cannot "
                    "accept text prompts. Please provide input_ids or re-initialize "
                    "the engine with skip_tokenizer_init=False."
                )

            # For audio-only requests (e.g., Whisper), text may be empty.
            # The multimodal processor will provide input_ids later.
            # 中译：纯音频请求（如 Whisper）可能没有文本，先用空占位，稍后由多模态处理器补全 input_ids。
            if not input_text and self.mm_processor and obj.contains_mm_input():
                # Use empty placeholder - multimodal processor will override
                input_ids = []
            else:
                input_ids, token_type_ids = await self._tokenize_texts(
                    input_text, is_cross_encoder_request
                )

        contains_mm_input = obj.contains_mm_input()
        if contains_mm_input and get_disagg().language_model_only:
            raise ValueError(
                "Multimodal inputs are not supported when --language-model-only "
                "is set; the encoder is not loaded. Restart without the flag."
            )
        is_mossvl = (
            "MossVLForConditionalGeneration"
            in self.model_config.hf_config.architectures
        )
        should_run_mm_processor = self.mm_processor is not None and (
            contains_mm_input or is_mossvl
        )

        if should_run_mm_processor:
            if obj.image_data is not None and not isinstance(obj.image_data, list):
                obj.image_data = [obj.image_data]
            if obj.video_data is not None and not isinstance(obj.video_data, list):
                obj.video_data = [obj.video_data]
            if obj.audio_data is not None and not isinstance(obj.audio_data, list):
                obj.audio_data = [obj.audio_data]
            if contains_mm_input:
                self._validate_mm_limits(obj)
                # mm_content_hashes is a GenerateReqInput field; EmbeddingReqInput
                # has no such attribute.
                if isinstance(obj, GenerateReqInput):
                    self._normalize_mm_content_hashes(obj)

            mm_inputs = None
            mm_processor_input = (
                input_ids
                if self.mm_processor.prefer_tokenized_input and input_ids is not None
                else (input_text or input_ids)
            )

            if (
                not get_disagg().language_only
                or get_disagg().encoder_transfer_backend == "zmq_to_tokenizer"
            ):
                if get_disagg().language_only:
                    mm_inputs = await self.mm_receiver.recv_mm_data(
                        request_obj=obj,
                        mm_processor=self.mm_processor,
                        prompt=mm_processor_input,
                        need_wait_for_mm_inputs=obj.need_wait_for_mm_inputs,
                    )
                    _reject_missing_dispatched_encoder_embedding(obj, mm_inputs)
                if mm_inputs is None:
                    if get_disagg().language_only:
                        logger.warning(
                            "Encoder embedding not available, "
                            "falling back to local mm processing"
                        )
                    mm_inputs = await self.mm_processor.process_mm_data_async(
                        image_data=obj.image_data,
                        audio_data=obj.audio_data,
                        input_text=mm_processor_input,
                        request_obj=obj,
                        max_req_input_len=self.max_req_input_len,
                    )
            elif (
                get_disagg().language_only
                and get_disagg().encoder_transfer_backend
                in ["zmq_to_scheduler", "mooncake"]
                and not obj.need_wait_for_mm_inputs
            ):
                # In language_only mode with zmq_to_scheduler/mooncake, if we didn't dispatch
                # to encoder (e.g., only one image), process locally like non-language_only mode
                mm_inputs = await self.mm_processor.process_mm_data_async(
                    image_data=obj.image_data,
                    audio_data=obj.audio_data,
                    input_text=mm_processor_input,
                    request_obj=obj,
                    max_req_input_len=self.max_req_input_len,
                )

            if mm_inputs and mm_inputs.input_ids is not None:
                input_ids = mm_inputs.input_ids
            if mm_inputs and mm_inputs.token_type_ids is not None:
                token_type_ids = mm_inputs.token_type_ids
                if not isinstance(token_type_ids, list):
                    token_type_ids = token_type_ids.flatten().tolist()
            # Caller-supplied per-image hashes (external KV routers, e.g.
            # routing-aware orchestrators that compute a content-addressed
            # hash before dispatch). Setting MultimodalDataItem.hash here
            # short-circuits the internal hash_feature() recompute inside
            # set_pad_value(), making the derived pad_value deterministic
            # from the caller's hash. That alignment lets the router's
            # routing decision agree with sglang's prefix-cache key for
            # the same image. On any per-item parse error or list-length
            # mismatch we fall back to the internal recompute so a
            # malformed mm_hashes never blocks a request.
            caller_mm_hashes = getattr(obj, "mm_hashes", None)
            if caller_mm_hashes and mm_inputs and mm_inputs.mm_items:
                if len(caller_mm_hashes) != len(mm_inputs.mm_items):
                    logger.warning(
                        "mm_hashes length (%d) != mm_items length (%d); "
                        "ignoring caller hashes for this request.",
                        len(caller_mm_hashes),
                        len(mm_inputs.mm_items),
                    )
                else:
                    for item, hex_hash in zip(mm_inputs.mm_items, caller_mm_hashes):
                        if not isinstance(item, MultimodalDataItem):
                            continue
                        try:
                            item.set_hash(int(hex_hash, 16))
                        except (TypeError, ValueError):
                            logger.warning(
                                "Ignoring malformed mm_hashes entry %r; "
                                "this item will fall back to hash_feature().",
                                hex_hash,
                            )
            if (
                envs.SGLANG_MM_PRECOMPUTE_HASH.get()
                and mm_inputs
                and mm_inputs.mm_items
            ):
                for item in mm_inputs.mm_items:
                    if isinstance(item, MultimodalDataItem):
                        item.set_pad_value()
        else:
            mm_inputs = None

        # 中译：长度等合法性校验，再组装成调度器可消费的 Tokenized* 请求对象。
        self._validate_one_request(obj, input_ids)
        return self._create_tokenized_object(
            obj, input_text, input_ids, input_embeds, mm_inputs, token_type_ids
        )

    @staticmethod
    def _normalize_mm_content_hashes(obj: GenerateReqInput) -> None:
        """Merge Native/OpenAI content identities and validate their alignment."""
        from sglang.srt.multimodal.cache import parse_content_hash
        from sglang.srt.utils import ImageData

        images = obj.image_data or []
        explicit = obj.mm_content_hashes
        inline = [
            image.content_hash if isinstance(image, ImageData) else None
            for image in images
        ]
        if explicit is None and not any(inline):
            return
        if explicit is None:
            explicit = inline
        if len(explicit) != len(images):
            raise ValueError(
                f"mm_content_hashes has {len(explicit)} entries for "
                f"{len(images)} images"
            )

        normalized = []
        for index, (provided, embedded) in enumerate(zip(explicit, inline)):
            provided = parse_content_hash(provided)
            embedded = parse_content_hash(embedded)
            if provided is not None and embedded is not None and provided != embedded:
                raise ValueError(f"Conflicting content hashes for image_data[{index}]")
            normalized.append(provided or embedded)
        obj.mm_content_hashes = normalized

    def _validate_one_request(
        self, obj: Union[GenerateReqInput, EmbeddingReqInput], input_ids: List[int]
    ) -> None:
        """Validates that the input token count and the requested token count doesn't exceed the model's context length.

        中译：校验「输入 token 数」以及「输入 + max_new_tokens」是否超出模型上下文长度；
              超限时按 allow_auto_truncate 决定截断或报错。还会校验 embedding/生成专属字段。
        """
        # FIXME: unify the length validation logic with the one in the scheduler.
        _max_req_len = self.context_len
        input_token_num = len(input_ids) if input_ids is not None else 0
        input_token_num += self.num_reserved_tokens

        # Validate input length
        if input_token_num >= self.context_len:
            if self.allow_auto_truncate:
                logger.warning(
                    f"The input ({input_token_num} tokens) is longer than the "
                    f"model's context length ({self.context_len} tokens). "
                    "Truncating the input."
                )
                del input_ids[_max_req_len:]
                input_token_num = len(input_ids)
            else:
                raise ValueError(
                    f"The input ({input_token_num} tokens) is longer than the "
                    f"model's context length ({self.context_len} tokens)."
                )

        # Validate total tokens (input + max_new_tokens)
        max_new_tokens = obj.sampling_params.get("max_new_tokens")
        if (
            self.validate_total_tokens
            and max_new_tokens is not None
            and (max_new_tokens + input_token_num) > _max_req_len
        ):
            if self.allow_auto_truncate:
                logger.warning(
                    f"Requested token count ({input_token_num} input + {max_new_tokens} new) "
                    f"exceeds the model's context length ({self.context_len} tokens). "
                    "Truncating max_new_tokens."
                )
                obj.sampling_params["max_new_tokens"] = max(
                    0, _max_req_len - input_token_num
                )
            else:
                total_tokens = max_new_tokens + input_token_num
                error_msg = (
                    f"Requested token count exceeds the model's maximum context length "
                    f"of {self.context_len} tokens. You requested a total of {total_tokens} "
                    f"tokens: {input_token_num} tokens from the input messages and "
                    f"{max_new_tokens} tokens for the completion. Please reduce the number "
                    f"of tokens in the input messages or the completion to fit within the limit."
                )
                raise ValueError(error_msg)

        # Validate embedding requests
        if isinstance(obj, EmbeddingReqInput) and self.is_generation:
            raise ValueError(
                "This model does not appear to be an embedding model by default. "
                "Please add `--is-embedding` when launching the server or try another model."
            )

        # Validate Matryoshka embeddings
        if isinstance(obj, EmbeddingReqInput):
            self._validate_for_matryoshka_dim(obj)

        # Validate generation-specific fields
        if isinstance(obj, GenerateReqInput):
            self._validate_token_ids_logprob(obj)
            requested_hidden_mode = get_request_return_hidden_states_mode(
                obj.return_hidden_states
            )
            server_hidden_mode = get_server_return_hidden_states_mode()
            if requested_hidden_mode > server_hidden_mode:
                if server_hidden_mode.need_capture():
                    raise ValueError(
                        "The requested return_hidden_states mode exceeds the "
                        f"server maximum `{get_exec().features.return_hidden_states_mode}`. "
                        "Please launch with `--return-hidden-states-mode full` "
                        "to allow return_hidden_states=True."
                    )
                raise ValueError(
                    "The server is not configured to return hidden states. "
                    "Please set `--return-hidden-states-mode last`, "
                    "`--return-hidden-states-mode full`, or the legacy "
                    "`--enable-return-hidden-states` flag."
                )
            if (
                obj.custom_logit_processor
                and not get_exec().features.enable_custom_logit_processor
            ):
                raise ValueError(
                    "The server is not configured to enable custom logit processor. "
                    "Please set `--enable-custom-logit-processor` to enable this feature."
                )

    def _validate_mm_limits(
        self, obj: Union[GenerateReqInput, EmbeddingReqInput]
    ) -> None:
        if not get_mm().limit_mm_data_per_request:
            return

        for modality, limit in get_mm().limit_mm_data_per_request.items():
            data = getattr(obj, f"{modality}_data", None)
            if data:
                count = len(data) if isinstance(data, list) else 1
                if count > limit:
                    raise ValueError(
                        f"{modality.capitalize()} count {count} exceeds limit {limit} per request."
                    )

    def _validate_for_matryoshka_dim(self, obj: EmbeddingReqInput) -> None:
        """Validate the request for Matryoshka dim if it has the field set.

        中译：当 embedding 请求指定了 dimensions（Matryoshka 可变维度）时，校验模型是否支持、
              维度是否合法（>0、在允许集合内、不超过 hidden_size）。
        """
        if obj.dimensions is None:
            return

        if not self.model_config.is_matryoshka:
            raise ValueError(
                f"Model '{self.model_config.model_path}' does not support matryoshka representation, "
                f"changing output dimensions will lead to poor results."
            )

        if obj.dimensions < 1:
            raise ValueError("Requested dimensions must be greater than 0")

        if (
            self.model_config.matryoshka_dimensions
            and obj.dimensions not in self.model_config.matryoshka_dimensions
        ):
            raise ValueError(
                f"Model '{self.model_config.model_path}' only supports {self.model_config.matryoshka_dimensions} matryoshka dimensions, "
                f"using other output dimensions will lead to poor results."
            )

        if obj.dimensions > self.model_config.hidden_size:
            raise ValueError(
                f"Provided dimensions are greater than max embedding dimension: {self.model_config.hidden_size}"
            )

    def _validate_token_ids_logprob(self, obj: GenerateReqInput) -> None:
        # 中译：校验 token_ids_logprob——必须是扁平的 int 列表，且每个 token id 在词表范围内。
        # Batch requests are split into per-request sub-objects before this
        # runs (normalize_batch_and_arguments + __getitem__), so the only
        # legal shape here is the per-request contract of
        # TokenizedGenerateReqInput.token_ids_logprob: a flat list of ints.
        token_ids_logprob = obj.token_ids_logprob
        if not token_ids_logprob:
            return
        if not isinstance(token_ids_logprob, list):
            raise ValueError("token_ids_logprob must be a flat list of integers.")
        vocab_size = self.model_config.vocab_size
        for token_id in token_ids_logprob:
            if not isinstance(token_id, int):
                raise ValueError("token_ids_logprob must be a flat list of integers.")
            if token_id < 0 or token_id >= vocab_size:
                raise ValueError(
                    f"token_ids_logprob contains out-of-vocabulary token id "
                    f"{token_id}; valid range is [0, {vocab_size})."
                )

    def _validate_input_ids_in_vocab(
        self, input_ids: Union[List[int], List[List[int]]], vocab_size: int
    ) -> None:
        # Handle both single sequence and batch of sequences
        if isinstance(input_ids[0], list):
            # Batch of sequences
            for seq in input_ids:
                if any(id >= vocab_size for id in seq):
                    raise ValueError(
                        f"The input_ids {seq} contains values greater than the vocab size ({vocab_size})."
                    )
        else:
            # Single sequence
            if any(id >= vocab_size for id in input_ids):
                raise ValueError(
                    f"The input_ids {input_ids} contains values greater than the vocab size ({vocab_size})."
                )

    def _create_tokenized_object(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        input_text: str,
        input_ids: Optional[List[int]],
        input_embeds: Optional[List[List[float]]] = None,
        mm_inputs=None,
        token_type_ids: Optional[List[int]] = None,
    ) -> Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput]:
        """Create a tokenized request object from common parameters.

        中译：把分词/预处理结果连同采样参数封装成调度器可消费的 Tokenized* 请求对象。
              其中采样参数会先 normalize + verify；input_ids 转为 array('q') 紧凑存储；
              并记录分词完成的时间统计。
        """
        input_ids_arr: Optional[array[int]] = (
            array("q", input_ids) if input_ids is not None else None
        )
        # Parse sampling parameters
        # Note: if there are preferred sampling params, we use them if they are not
        # explicitly passed in sampling_params
        if self.preferred_sampling_params:
            sampling_kwargs = {**self.preferred_sampling_params, **obj.sampling_params}
        else:
            sampling_kwargs = obj.sampling_params
        if isinstance(obj, GenerateReqInput) and obj.max_thinking_tokens is not None:
            sampling_kwargs = dict(sampling_kwargs)
            custom_params = dict(sampling_kwargs.get("custom_params") or {})
            custom_params["thinking_budget"] = obj.max_thinking_tokens
            sampling_kwargs["custom_params"] = custom_params
        sampling_params = self.sampling_params_class(**sampling_kwargs)
        sampling_params.normalize(self.tokenizer)
        sampling_params.verify(self.model_config.vocab_size)

        # Build return object
        if isinstance(obj, GenerateReqInput):
            session_params = (
                SessionParams(**obj.session_params) if obj.session_params else None
            )

            bootstrap_room = obj.bootstrap_room
            if (
                bootstrap_room is None
                and get_disagg().disaggregation_transfer_backend == "fake"
            ):
                bootstrap_room = self.fake_bootstrap_room_counter
                self.fake_bootstrap_room_counter += 1

            tokenized_obj = TokenizedGenerateReqInput(
                input_text=input_text,
                input_ids=input_ids_arr,
                mm_inputs=mm_inputs,
                sampling_params=sampling_params,
                return_logprob=obj.return_logprob,
                logprob_start_len=obj.logprob_start_len,
                top_logprobs_num=obj.top_logprobs_num,
                token_ids_logprob=obj.token_ids_logprob,
                return_sampling_mask=obj.return_sampling_mask,
                return_flat_raw_top_logprobs=obj.return_flat_raw_top_logprobs,
                stream=obj.stream,
                rid=obj.rid,
                http_worker_ipc=obj.http_worker_ipc,
                bootstrap_host=obj.bootstrap_host,
                bootstrap_port=obj.bootstrap_port,
                bootstrap_room=bootstrap_room,
                lora_id=obj.lora_id,
                input_embeds=input_embeds,
                positional_embed_overrides=obj.positional_embed_overrides,
                session_id=obj.session_id,
                session_params=session_params,
                custom_logit_processor=obj.custom_logit_processor,
                require_reasoning=obj.require_reasoning,
                return_hidden_states=obj.return_hidden_states,
                return_routed_experts=obj.return_routed_experts,
                routed_experts_start_len=obj.routed_experts_start_len,
                return_indexer_topk=obj.return_indexer_topk,
                routed_dp_rank=obj.routed_dp_rank,
                disagg_prefill_dp_rank=obj.disagg_prefill_dp_rank,
                priority=obj.priority,
                extra_key=obj.extra_key,
                cache_salt=obj.cache_salt,
                routing_key=obj.routing_key,
                token_type_ids=token_type_ids,
                need_wait_for_mm_inputs=obj.need_wait_for_mm_inputs,
                num_items_assigned=obj.num_items_assigned,
                multi_item_delimiter_indices=obj.multi_item_delimiter_indices,
                encoder_urls=obj.encoder_urls,
            )
        elif isinstance(obj, EmbeddingReqInput):
            # Resolve unresolved embed overrides now that input_ids are available
            positional_embed_overrides = obj.positional_embed_overrides
            if (
                positional_embed_overrides is None
                and obj.embed_overrides is not None
                and obj.embed_override_token_id is not None
            ):
                positional_embed_overrides = self._resolve_embed_overrides(
                    input_ids_arr, obj.embed_override_token_id, obj.embed_overrides
                )

            tokenized_obj = TokenizedEmbeddingReqInput(
                input_text=input_text,
                input_ids=input_ids_arr,
                mm_inputs=mm_inputs,
                token_type_ids=token_type_ids,
                sampling_params=sampling_params,
                positional_embed_overrides=positional_embed_overrides,
                rid=obj.rid,
                priority=obj.priority,
                dimensions=obj.dimensions,
                lora_id=obj.lora_id,
                http_worker_ipc=obj.http_worker_ipc,
                return_pooled_hidden_states=obj.return_pooled_hidden_states,
                multi_item_delimiter_indices=obj.multi_item_delimiter_indices,
            )

        tokenized_obj.time_stats = self.rid_to_state[obj.rid].time_stats
        self.rid_to_state[obj.rid].time_stats.set_tokenize_finish_time()

        return tokenized_obj

    @staticmethod
    def _resolve_embed_overrides(
        input_ids: array[int],
        token_id: int,
        embeds: List[torch.Tensor],
    ) -> PositionalEmbeds:
        positions = [idx for idx, tok in enumerate(input_ids) if tok == token_id]
        if len(positions) != len(embeds):
            raise ValueError(
                f"input contains {len(positions)} occurrences of "
                f"embed_override_token_id={token_id}, "
                f"but embed_overrides has {len(embeds)} entries."
            )
        return PositionalEmbeds(embeds=embeds, positions=positions)

    async def _batch_tokenize_and_process(
        self, batch_size: int, obj: Union[GenerateReqInput, EmbeddingReqInput]
    ) -> List[Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput]]:
        """Handle batch tokenization for text inputs only.

        中译：对一个批次的纯文本请求做「一次性批量分词」（比逐条更快）。若批次中无文本
              （都是预分词的 input_ids），则退化为逐条 _tokenize_one_request。
        """
        logger.debug(f"Starting batch tokenization for {batch_size} text requests")

        # If batch does not have text nothing to tokenize
        # so lets construct the return object
        if not self._batch_has_text(batch_size, obj):
            # All requests already have input_ids, no need to tokenize
            return [await self._tokenize_one_request(obj[i]) for i in range(batch_size)]

        self._validate_batch_tokenization_constraints(batch_size, obj)

        # Collect requests and texts
        requests = [obj[i] for i in range(batch_size)]
        texts = [req.text for req in requests]

        # Check if any request is a cross-encoder request
        is_cross_encoder_request = any(
            isinstance(req, EmbeddingReqInput) and req.is_cross_encoder_request
            for req in requests
        )

        # Batch tokenize all texts using unified method
        input_ids_list, token_type_ids_list = await self._tokenize_texts(
            texts, is_cross_encoder_request
        )

        # Process all requests
        tokenized_objs = []
        for i, req in enumerate(requests):
            self._validate_one_request(obj[i], input_ids_list[i])
            token_type_ids = (
                token_type_ids_list[i] if token_type_ids_list is not None else None
            )
            tokenized_objs.append(
                self._create_tokenized_object(
                    req, req.text, input_ids_list[i], None, None, token_type_ids
                )
            )
        logger.debug(f"Completed batch processing for {batch_size} requests")
        return tokenized_objs

    def _validate_batch_tokenization_constraints(
        self, batch_size: int, obj: Union[GenerateReqInput, EmbeddingReqInput]
    ) -> None:
        """Validate constraints for batch tokenization processing.

        中译：批量分词的前置约束校验——多模态输入、预分词的 input_ids、input_embeds 都不应
              开启批量分词（enable_tokenizer_batch_encode），命中则报错。
        """
        for i in range(batch_size):
            if self.is_generation and obj[i].contains_mm_input():
                raise ValueError(
                    "For multimodal input processing do not set `enable_tokenizer_batch_encode`."
                )
            if obj[i].input_ids is not None:
                raise ValueError(
                    "Batch tokenization is not needed for pre-tokenized input_ids. Do not set `enable_tokenizer_batch_encode`."
                )
            if obj[i].input_embeds is not None:
                raise ValueError(
                    "Batch tokenization is not needed for input_embeds. Do not set `enable_tokenizer_batch_encode`."
                )

    def _batch_has_text(
        self, batch_size: int, obj: Union[GenerateReqInput, EmbeddingReqInput]
    ) -> bool:
        """Check if any request in the batch contains text input.

        中译：判断批次中是否有任一请求含需要分词的文本（或多模态）输入。
        """
        for i in range(batch_size):
            if obj[i].text:
                return True
            elif self.is_generation and obj[i].contains_mm_input():
                return True

        return False

    def _should_use_batch_tokenization(self, batch_size, requests) -> bool:
        """Return True if we should run the tokenizer in batch mode.

        Current policy:
        - Respect explicit server flag `enable_tokenizer_batch_encode`.
        - Or, if no request has text or multimodal input (all use pre-tokenized input_ids or input_embeds), batch the requests without tokenization.
        - Batch tokenization does not support DP attention yet, and it will make everything goes to the first rank currently

        中译：决定是否走批量分词。策略：显式开启 enable_tokenizer_batch_encode；或在未启用
              DP attention 且批次中没有文本/多模态（全是预分词输入）时批量发送。
              注意批量分词尚不支持 DP attention（会把请求都打到第一个 rank）。
        """
        return batch_size > 0 and (
            get_serving().enable_tokenizer_batch_encode
            or (
                (not get_parallel().enable_dp_attention)
                and (not self._batch_has_text(batch_size, requests))
            )
        )

    def _send_one_request(
        self,
        tokenized_obj: Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput],
    ):
        # 中译：把单个分词后的请求经 ZMQ 发给调度器。wrap_shm_features 会把大块多模态特征
        prepared_mm_items = []
        dispatched = False
        try:
            prepared_mm_items = self.cuda_vmm_feature_transport.prepare_for_dispatch(
                (tokenized_obj.mm_inputs,)
            )
            tokenized_obj.time_stats.set_api_server_dispatch_time()
            tokenized_obj = wrap_shm_features(tokenized_obj)
            time_stats = tokenized_obj.time_stats
            tokenized_obj.wrap_pickle_fields()
            self._dispatch_to_scheduler(tokenized_obj)
            dispatched = True
            tokenized_obj.time_stats = time_stats
            tokenized_obj.time_stats.set_api_server_dispatch_finish_time()
        finally:
            if not dispatched:
                self.cuda_vmm_feature_transport.cancel_for_dispatch(prepared_mm_items)

    def _send_batch_request(
        self,
        tokenized_objs: List[
            Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput]
        ],
    ):
        """Send a batch of tokenized requests as a single batched request to the scheduler."""
        prepared_mm_items = []
        dispatched = False
        try:
            prepared_mm_items = self.cuda_vmm_feature_transport.prepare_for_dispatch(
                tokenized_obj.mm_inputs for tokenized_obj in tokenized_objs
            )

            set_time_batch(tokenized_objs, "set_api_server_dispatch_time")
            time_stats = [tokenized_obj.time_stats for tokenized_obj in tokenized_objs]
            for tokenized_obj in tokenized_objs:
                tokenized_obj.wrap_pickle_fields()

            if isinstance(tokenized_objs[0], TokenizedGenerateReqInput):
                batch_req = BatchTokenizedGenerateReqInput(batch=tokenized_objs)
            else:
                batch_req = BatchTokenizedEmbeddingReqInput(batch=tokenized_objs)

            self._dispatch_to_scheduler(batch_req)
            dispatched = True
            for tokenized_obj, time_stat in zip(tokenized_objs, time_stats):
                tokenized_obj.time_stats = time_stat
            set_time_batch(tokenized_objs, "set_api_server_dispatch_finish_time")
        finally:
            if not dispatched:
                self.cuda_vmm_feature_transport.cancel_for_dispatch(prepared_mm_items)

    def _coalesce_streaming_chunks(
        self,
        out_list: list,
        rid: str,
        customized_info_keys: Optional[Iterable[str]] = None,
    ) -> dict:
        """Coalesce multiple incremental streaming chunks into one.

        Both text and output_ids are incremental deltas, so we concatenate them;
        all other fields (meta_info, etc.) are taken from the last chunk.

        中译：增量流式下，若一次唤醒里积压了多个增量片段，把它们合并成一个，避免丢 token。
              text 与 output_ids 是增量 delta 故拼接；其余字段取最后一个片段；
              meta_info 中的增量 logprob 字段也逐段拼接。
        """
        if len(out_list) >= 20:
            logger.warning(
                "Streaming backlog: rid=%s, coalescing %d queued chunks into one. "
                "This may inflate P99 ITL for affected requests.",
                rid,
                len(out_list),
            )
        out = dict(out_list[-1])
        if "output_ids" in out:
            out["output_ids"] = [id for chunk in out_list for id in chunk["output_ids"]]
        if "text" in out:
            out["text"] = "".join(chunk["text"] for chunk in out_list)
        if "meta_info" in out:
            meta_info_list = [chunk["meta_info"] for chunk in out_list]
            meta_info = dict(meta_info_list[-1])
            incremental_streaming_keys = set(_INCREMENTAL_STREAMING_META_INFO_KEYS)
            if customized_info_keys is not None:
                incremental_streaming_keys.update(customized_info_keys)
            for key in incremental_streaming_keys:
                if any(key in m for m in meta_info_list):
                    meta_info[key] = [
                        item for m in meta_info_list for item in m.get(key, [])
                    ]
            out["meta_info"] = meta_info
        return out

    async def _handle_abort_finish_reason(
        self,
        out: dict,
        state: ReqState,
        is_stream: bool,
    ) -> Optional[dict]:
        """Returns the output dict to yield (stream abort), None for normal flow;
        raises ValueError/HTTPException for non-stream aborts."""
        finish_reason = out["meta_info"]["finish_reason"]

        if (
            finish_reason.get("type") == "abort"
            and finish_reason.get("status_code") == HTTPStatus.BAD_REQUEST
        ):
            if not is_stream:
                raise ValueError(finish_reason["message"])
            return out

        if finish_reason.get("type") == "abort" and finish_reason.get(
            "status_code"
        ) in (
            HTTPStatus.SERVICE_UNAVAILABLE,
            HTTPStatus.INTERNAL_SERVER_ERROR,
        ):
            # Delete the key to prevent resending abort request to the scheduler and
            # to ensure aborted request state is cleaned up.
            if state.obj.rid in self.rid_to_state:
                del self.rid_to_state[state.obj.rid]

            # Mark ongoing LoRA request as finished.
            if self.enable_lora and state.obj.lora_path:
                await self.lora_registry.release(state.obj.lora_id)
            if not is_stream:
                raise fastapi.HTTPException(
                    status_code=finish_reason["status_code"],
                    detail=finish_reason["message"],
                )
            return out

        return None

    def _wait_one_response(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        # Batch dispatch builds every waiter before advancing any.
        # Both removers append the output after the del, so the ReqState stays valid.
        state = self.rid_to_state[obj.rid]
        return self._stream_one_response(obj=obj, state=state, request=request)

    async def _stream_one_response(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        state: ReqState,
        request: Optional[fastapi.Request] = None,
    ):
        # Not all request types have `stream` (e.g., EmbeddingReqInput). Default to non-streaming.
        # 中译：并非所有请求类型都有 stream 字段（如 EmbeddingReqInput），默认非流式。
        is_stream = getattr(obj, "stream", False)
        while True:
            try:
                await asyncio.wait_for(
                    state.event.wait(), timeout=_REQUEST_STATE_WAIT_TIMEOUT
                )
            except asyncio.TimeoutError:
                if (
                    request is not None
                    and not obj.background
                    and await request.is_disconnected()
                ):
                    # Abort the request for disconnected requests (non-streaming, waiting queue)
                    self.abort_request(obj.rid)
                    # Use exception to kill the whole call stack and asyncio task
                    raise ValueError(
                        f"Request is disconnected from the client side (type 1). Abort request {obj.rid=}"
                    )
                continue

            # Drain all pending outputs atomically.
            # 中译：原子地取走所有待发输出（换上新空列表），并清掉 event，等待下一批。
            out_list = state.out_list
            state.out_list = []
            finished = state.finished
            state.event.clear()

            # With incremental streaming, each chunk is a delta — coalesce
            # multiple queued chunks to avoid dropping token ids.
            incremental_stream = is_stream and self.incremental_streaming_output
            if incremental_stream and len(out_list) > 1:
                out = self._coalesce_streaming_chunks(
                    out_list,
                    obj.rid,
                    state.customized_info_accumulated.keys(),
                )
            else:
                out = out_list[-1]

            # Resolve deferred text for non-incremental streaming.
            # _handle_batch_output sets "text": None on intermediate chunks
            # to avoid O(n) string rebuild per step (O(n^2) total).
            if (
                is_stream
                and not incremental_stream
                and "text" in out
                and out["text"] is None
            ):
                out["text"] = state.get_text()

            # Flattened so the downstream finished/logging/metrics logic is shared.
            if out.get("beam_results"):
                if not finished:
                    # Intermediate beam output; skip until finished.
                    continue
                out = build_beam_search_out(out)

            if finished:
                # 中译：请求已结束——记录响应发送时间、写日志与指标、处理 abort，最后 yield 收尾并退出循环。
                # Record response sent time right before we log finished results and metrics.
                if not state.time_stats.response_sent_to_client_time:
                    state.time_stats.set_response_sent_to_client_time()
                    out["meta_info"][
                        "response_sent_to_client_ts"
                    ] = state.time_stats.get_response_sent_to_client_realtime()
                self.request_logger.log_finished_request(
                    obj,
                    out,
                    request=request,
                )

                if self.request_metrics_exporter_manager.exporter_enabled():
                    asyncio.create_task(
                        self.request_metrics_exporter_manager.write_record(obj, out)
                    )

                # Check if this was an abort/error created by scheduler
                if isinstance(out["meta_info"].get("finish_reason"), dict):
                    abort_out = await self._handle_abort_finish_reason(
                        out, state, is_stream
                    )
                    if abort_out is not None:
                        yield abort_out
                        break

                yield out
                break

            if is_stream:
                # 中译：流式且未结束——把本次中间片段 yield 出去，循环回去等下一批。
                # Record response sent time right before we send response.
                if not state.time_stats.response_sent_to_client_time:
                    state.time_stats.set_response_sent_to_client_time()
                    out["meta_info"][
                        "response_sent_to_client_ts"
                    ] = state.time_stats.get_response_sent_to_client_realtime()
                yield out
            else:
                # 中译：非流式且未结束——不 yield 中间结果，仅借机检测客户端是否断连。
                if (
                    request is not None
                    and not obj.background
                    and await request.is_disconnected()
                ):
                    # Abort the request for disconnected requests (non-streaming, running)
                    self.abort_request(obj.rid)
                    # Use exception to kill the whole call stack and asyncio task
                    raise ValueError(
                        f"Request is disconnected from the client side (type 3). Abort request {obj.rid=}"
                    )

    async def _handle_batch_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        # 中译：处理批量请求。两条主路：
        #       (a) parallel_sample_num == 1：每条子请求独立分词发送，建立各自的 _wait_one_response 生成器；
        #       (b) parallel_sample_num > 1：先发一份 max_new_tokens=0 的请求把公共前缀缓存（prefix cache）
        #           预热，再为每条 sample 复制请求、分配新 rid 并发送。
        #       最后：非流式 gather 所有结果一次性 yield；流式则用 asyncio.wait 边到边 yield，
        #       并给每个结果标注其在批次中的 index。
        batch_size = obj.batch_size

        generators = []
        rids = []
        if getattr(obj, "parallel_sample_num", 1) == 1:
            if self._should_use_batch_tokenization(batch_size, obj):
                tokenized_objs = await self._batch_tokenize_and_process(batch_size, obj)
                self._send_batch_request(tokenized_objs)

                # Set up generators for each request in the batch
                for i in range(batch_size):
                    tmp_obj = obj[i]
                    state = self.rid_to_state[tmp_obj.rid]
                    if tmp_obj.return_prompt_token_ids:
                        state.prompt_token_ids = list(tokenized_objs[i].input_ids)
                    generators.append(self._wait_one_response(tmp_obj, request))
                    rids.append(tmp_obj.rid)
            else:
                # Sequential tokenization and processing
                with (
                    input_blocker_guard_region(
                        dispatch_to_scheduler=self._dispatch_to_scheduler,
                    )
                    if get_bool_env_var("SGLANG_ENABLE_COLOCATED_BATCH_GEN")
                    else nullcontext()
                ):
                    for i in range(batch_size):
                        tmp_obj = obj[i]
                        tokenized_obj = await self._tokenize_one_request(tmp_obj)
                        state = self.rid_to_state[tmp_obj.rid]
                        if tmp_obj.return_prompt_token_ids:
                            state.prompt_token_ids = list(tokenized_obj.input_ids)
                        self._send_one_request(tokenized_obj)
                        generators.append(self._wait_one_response(tmp_obj, request))
                        rids.append(tmp_obj.rid)
        else:
            # FIXME: When using batch and parallel_sample_num together, the perf is not optimal.
            if batch_size > 128:
                logger.warning(
                    "Sending a single large batch with parallel sampling (n > 1) has not been well optimized. "
                    "The performance might be better if you just duplicate the requests n times or use "
                    "many threads to send them one by one with parallel sampling (n > 1)."
                )

            # Tokenize all requests
            objs = [obj[i] for i in range(batch_size)]
            tokenized_objs = await asyncio.gather(
                *(self._tokenize_one_request(obj) for obj in objs)
            )

            # Cache the common prefix for parallel sampling
            # 中译：并行采样前，先用 max_new_tokens=0 跑一遍把公共前缀写入前缀缓存，
            #       后续 n 个 sample 即可命中缓存、避免重复 prefill。
            for i in range(batch_size):
                tmp_obj = copy.copy(objs[i])
                tokenized_obj = copy.copy(tokenized_objs[i])
                # Ensure independent mm_items so wrap_shm_features won't mutate the original
                if hasattr(tokenized_obj, "mm_inputs") and tokenized_obj.mm_inputs:
                    tokenized_obj.mm_inputs = copy.copy(tokenized_obj.mm_inputs)
                    tokenized_obj.mm_inputs.mm_items = [
                        copy.copy(item) for item in tokenized_obj.mm_inputs.mm_items
                    ]
                tokenized_obj.rid = tmp_obj.regenerate_rid()
                tokenized_obj.sampling_params = copy.copy(tokenized_obj.sampling_params)
                tokenized_obj.sampling_params.max_new_tokens = 0
                tokenized_obj.stream = False
                self._init_req_state(tmp_obj)
                self._send_one_request(tokenized_obj)
                await self._wait_one_response(tmp_obj, request).__anext__()

            # Expand requests, assign new rids for them, and send them
            for i in range(batch_size):
                for _ in range(obj.parallel_sample_num):
                    tmp_obj = copy.copy(objs[i])
                    tokenized_obj = copy.copy(tokenized_objs[i])
                    # Ensure independent mm_items so wrap_shm_features won't mutate the original
                    if hasattr(tokenized_obj, "mm_inputs") and tokenized_obj.mm_inputs:
                        tokenized_obj.mm_inputs = copy.copy(tokenized_obj.mm_inputs)
                        tokenized_obj.mm_inputs.mm_items = [
                            copy.copy(item) for item in tokenized_obj.mm_inputs.mm_items
                        ]
                    tokenized_obj.rid = tmp_obj.regenerate_rid()
                    self._init_req_state(tmp_obj)
                    state = self.rid_to_state[tmp_obj.rid]
                    tokenized_obj.time_stats = state.time_stats
                    if tmp_obj.return_prompt_token_ids:
                        state.prompt_token_ids = list(tokenized_objs[i].input_ids)
                    self._send_one_request(tokenized_obj)
                    generators.append(self._wait_one_response(tmp_obj, request))
                    rids.append(tmp_obj.rid)

                self.rid_to_state[objs[i].rid].time_stats.set_finished_time()
                del self.rid_to_state[objs[i].rid]

        # Wait for all requests
        # 中译：等待所有子请求。非流式直接 gather 后整批 yield；流式逐个完成逐个 yield。
        is_stream = hasattr(obj, "stream") and obj.stream
        if not is_stream:
            outputs = await self._collect_batch_responses(generators)
            yield outputs
        else:
            async for response in self._stream_batch_responses(generators, rids):
                yield response

    async def _collect_batch_responses(self, generators):
        tasks = [asyncio.create_task(gen.__anext__()) for gen in generators]
        try:
            return await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.gather(
                *(gen.aclose() for gen in generators),
                return_exceptions=True,
            )

    async def _stream_batch_responses(self, generators, rids):
        rid_to_index = {rid: i for i, rid in enumerate(rids)}
        task_map = {asyncio.create_task(gen.__anext__()): gen for gen in generators}
        try:
            while task_map:
                done, _ = await asyncio.wait(
                    task_map.keys(), return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    gen = task_map.pop(task)
                    try:
                        result = task.result()
                        result["index"] = rid_to_index[result["meta_info"]["id"]]
                        yield result
                        new_task = asyncio.create_task(gen.__anext__())
                        task_map[new_task] = gen
                    except StopAsyncIteration:
                        pass
        finally:
            pending_tasks = list(task_map)
            for task in pending_tasks:
                task.cancel()
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            await asyncio.gather(
                *(gen.aclose() for gen in generators),
                return_exceptions=True,
            )

    def abort_request(self, rid: str = "", abort_all: bool = False):
        # 中译：中止请求。给调度器发 AbortReq。abort_all=True 时中止全部，
        #       否则按 rid 中止单个（空 rid 会在调度器侧前缀匹配到所有请求，故此处拦掉）。
        # Empty rid would startswith-match every request on the scheduler.
        if not abort_all and not rid:
            logger.warning("Ignore abort_request with empty rid and abort_all=False")
            return
        if (
            not abort_all
            and get_serving().tokenizer_worker_num == 1
            and rid not in self.rid_to_state
        ):
            return
        req = AbortReq(rid=rid, abort_all=abort_all)
        self._dispatch_to_scheduler(req)
        if self.enable_metrics:
            # TODO: also use custom_labels from the request
            self.metrics_collector.observe_one_aborted_request(
                self.metrics_collector.labels
            )

    async def pause_generation(self, obj: PauseGenerationReqInput):
        # 中译：暂停生成。置 is_pause=True，新进入的请求会在 generate_request 处挂起。
        #       abort 模式下还会循环中止全部在途请求，直到 model_update_lock 不再被读锁占用
        #       （即没有 in-flight 请求）为止。
        async with self.is_pause_cond:
            self.is_pause = True
            if obj.mode != "abort":
                await self._async_dispatch_to_scheduler(obj)
            else:
                # we are using the model_update_lock to check if there is still on-going requests.
                # 中译：借用 model_update_lock 的占用情况判断是否还有进行中的请求。
                while True:
                    # TODO: maybe make it async instead of fire-and-forget
                    self.abort_request(abort_all=True)
                    is_locked = await self.model_update_lock.is_locked()
                    if not is_locked:
                        break
                    await asyncio.sleep(1.0)

    async def continue_generation(self, obj: ContinueGenerationReqInput):
        # 中译：恢复生成。清除暂停标志并 notify_all 唤醒所有在 is_pause_cond 上等待的请求。
        async with self.is_pause_cond:
            self.is_pause = False
            await self._async_dispatch_to_scheduler(obj)
            self.is_pause_cond.notify_all()

    async def update_weights_from_disk(
        self,
        obj: UpdateWeightFromDiskReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        # 中译：从磁盘热更新权重。非暂停态下需持「写锁」(独占)，确保没有请求在用旧权重；
        #       若已处于暂停态则无需再加锁（暂停时已无 in-flight 请求）。
        self.auto_create_handle_loop()

        if obj.load_format is None:
            obj.load_format = self.config_value("load_format")
        logger.info("Start update_weights. Load format=%s", obj.load_format)

        if obj.abort_all_requests:
            self.abort_request(abort_all=True)

        # Immediately update the weights if the engine is in paused state
        async with self.is_pause_cond:
            is_paused = self.is_pause

        lock_context = (
            self.model_update_lock.writer_lock if not is_paused else nullcontext()
        )
        async with lock_context:
            success, message, num_paused_requests = (
                await self._wait_for_model_update_from_disk(obj)
            )

        if success and obj.flush_cache and self.mm_processor is not None:
            self.mm_processor.clear_preprocess_cache()
        if success and obj.weight_version is not None:
            self._update_weight_version_if_provided(obj.weight_version)
            message += f" Weight version updated to {obj.weight_version}."

        return success, message, num_paused_requests

    def record_config_updates(self, source: str, **fields) -> None:
        """Record a control-plane config change: a weight update, a parser
        resolved from the chat template, a HiCache mirror attach.

        These land in the config bags like every other post-publish change, so
        one log carries the provenance for the whole process.
        """
        get_context().override(source, **fields)

    def config_value(self, name: str):
        """The value in effect for one config field."""
        if name in _MANAGER_OWNED_FIELDS:
            return getattr(self, name)
        return get_context().config_leaf(name)

    def _dump_config_snapshot(self) -> Optional[Dict[str, Any]]:
        """The config in effect, or None when it cannot be serialized.

        A dump is worth having even when the config is not: request data is the
        part that cannot be reconstructed afterwards.
        """
        try:
            return self.resolved_config_dict(self.server_args.resolved_dict())
        except Exception as e:
            logger.error(f"Failed to snapshot the resolved config for the dump: {e!r}")
            return None

    def resolved_config_dict(self, base: Dict[str, Any]) -> Dict[str, Any]:
        """``base`` (a serialized ``ServerArgs``) with the control-plane changes on top."""
        resolved = get_context().resolved_server_args_dict(base)
        for name in _MANAGER_OWNED_FIELDS:
            resolved[name] = getattr(self, name)
        return resolved

    def _update_model_path_info(self, model_path: str, load_format: str):
        # These two stay on the manager: the readback reads them from here,
        # and a bag write would not reach the other processes anyway.
        self.served_model_name = model_path
        self.model_path = model_path
        self.record_config_updates("tokenizer.update_weights", load_format=load_format)

    async def _wait_for_model_update_from_disk(
        self, obj: UpdateWeightFromDiskReqInput
    ) -> Tuple[bool, str]:
        # 中译：把更新请求发给调度器，并用 future 等待结果。dp_size>1 时需汇总各 DP rank
        expected_workers = self.elastic_worker_count
        self.model_update_expected_workers = expected_workers
        self.model_update_tmp = []
        self.model_update_result = asyncio.Future()
        self._dispatch_to_scheduler(obj)
        if expected_workers == 1:
            result = await self.model_update_result
            if result.success:
                self._update_model_path_info(obj.model_path, obj.load_format)
            return result.success, result.message, result.num_paused_requests
        else:
            result = await self.model_update_result

            all_success = all([r.success for r in result])
            if all_success is True:
                self._update_model_path_info(obj.model_path, obj.load_format)
            all_message = [r.message for r in result]
            all_message = " | ".join(all_message)
            all_paused_requests = [r.num_paused_requests for r in result]
            return all_success, all_message, all_paused_requests

    def configure_logging(self, obj: ConfigureLoggingReq):
        # 中译：运行时调整日志与转储配置（日志级别/格式、dump 目录与阈值等）；改日志级别时同步通知调度器。
        self.request_logger.configure(
            log_requests=obj.log_requests,
            log_requests_level=obj.log_requests_level,
            log_requests_format=obj.log_requests_format,
        )
        if obj.dump_requests_folder is not None:
            self.dump_requests_folder = obj.dump_requests_folder
        if obj.dump_requests_threshold is not None:
            self.dump_requests_threshold = obj.dump_requests_threshold
        if obj.dump_requests_exclude_meta_keys is not None:
            self.dump_requests_exclude_meta_keys = list(
                obj.dump_requests_exclude_meta_keys
            )
        if obj.crash_dump_folder is not None:
            self.crash_dump_folder = obj.crash_dump_folder
        if obj.log_level is not None:
            # setLevel() may raise exception if obj.log_level is illegal string.
            # Let the exception propagate to the caller.
            # Only legal requests will be sent to scheduler.
            logging.getLogger().setLevel(obj.log_level.upper())
            self._dispatch_to_scheduler(obj)
        logging.info(f"Config logging: {obj=}")

    async def freeze_gc(self):
        """Send a freeze_gc message to the scheduler first, then freeze locally."""
        self._dispatch_to_scheduler(FreezeGCReq())
        freeze_gc("Tokenizer Manager")
        return None

    def create_abort_task(self, obj: GenerateReqInput):
        # 中译：构造一个 FastAPI 后台任务——客户端断连时延迟 2s 后中止该请求（单条或批量各 rid）。
        # Abort the request if the client is disconnected.
        async def abort_request():
            await asyncio.sleep(2)
            if obj.is_single:
                self.abort_request(obj.rid)
            else:
                for rid in obj.rid:
                    self.abort_request(rid)

        background_tasks = BackgroundTasks()
        background_tasks.add_task(abort_request)
        return background_tasks

    def auto_create_handle_loop(self):
        # 中译：惰性启动后台协程。首个请求到来时创建 handle_loop（收输出主循环）与
        #       sigterm_watchdog（优雅退出看门狗），并在主线程注册 SIGTERM/SIGQUIT 处理器。
        #       已创建则直接返回（幂等）。
        if self.event_loop is not None:
            return

        # Create and start the handle_loop task
        loop = get_or_create_event_loop()
        self.asyncio_tasks.add(
            loop.create_task(print_exception_wrapper(self.handle_loop))
        )
        self.event_loop = loop

        # We only add signal handler when the tokenizer manager is in the main thread
        # due to the CPython limitation.
        # 中译：受 CPython 限制，只有在主线程才能注册信号处理器。
        if threading.current_thread() is threading.main_thread():
            signal_handler = self.signal_handler_class(self)
            loop.add_signal_handler(signal.SIGTERM, signal_handler.sigterm_handler)
            # Update the signal handler for the process. It overrides the sigquit handler in the launch phase.
            loop.add_signal_handler(
                signal.SIGQUIT, signal_handler.running_phase_sigquit_handler
            )

        self.asyncio_tasks.add(
            loop.create_task(print_exception_wrapper(self.sigterm_watchdog))
        )

    async def handle_loop(self):
        """The event loop that handles requests

        中译：「出口」主事件循环（生产者侧）。持续从 detokenizer 收消息：
              批量输出交给 _handle_batch_output（按 rid 写回各 ReqState 并唤醒等待协程）；
              其余控制类消息交给类型分发器。等待收消息期间临时关闭软看门狗（等消息不算卡死）。
        """
        while True:
            with self.soft_watchdog.disable():
                recv_obj = await async_sock_recv(self.recv_from_detokenizer)
            if isinstance(
                recv_obj,
                (BatchStrOutput, BatchEmbeddingOutput, BatchTokenIDOutput),
            ):
                await self._handle_batch_output(recv_obj)
            else:
                self._result_dispatcher(recv_obj)
            self.last_receive_tstamp = real_time()
            # 中译：喂狗，表明本轮处理正常推进。
            self.soft_watchdog.feed()

    async def _handle_batch_output(
        self,
        recv_obj: Union[
            BatchStrOutput,
            BatchEmbeddingOutput,
            BatchTokenIDOutput,
        ],
    ):
        # 中译：处理来自 detokenizer 的一批输出（核心出口逻辑）。对批次内每个 rid：
        recv_obj.time_stats = unwrap_from_pickle(recv_obj.time_stats)
        if isinstance(recv_obj, (BatchStrOutput, BatchTokenIDOutput)):
            customized_info = unwrap_from_pickle(recv_obj.customized_info)
        else:
            customized_info = None
        pending_notify: dict[str, ReqState] = {}
        batch_notify_size = get_serving().batch_notify_size
        for i, rid in enumerate(recv_obj.rids):
            state = self.rid_to_state.get(rid, None)
            if state is None:
                # Known race: /health_generate pops its rid as soon as ANY message bumps last_receive_tstamp.
                # 中译：已知竞态——/health_generate 一旦因任意消息刷新 last_receive_tstamp 就会弹出其 rid，
                #       故这里对健康检查 rid 直接跳过，不当作错误。
                if rid.startswith(HEALTH_CHECK_RID_PREFIX):
                    continue
                logger.error(
                    f"Received output for {rid=} but the state was deleted in TokenizerManager."
                )
                continue

            # Build meta_info and return value
            meta_info = {
                "id": rid,
                "finish_reason": recv_obj.finished_reasons[i],
                "prompt_tokens": recv_obj.prompt_tokens[i],
                "weight_version": self.config_value("weight_version"),
                "num_retractions": recv_obj.retraction_counts[i],
            }

            if self.enable_metrics:
                if recv_obj.time_stats is not None:
                    scheduler_time_stats = recv_obj.time_stats[i]
                    meta_info.update(scheduler_time_stats.convert_to_output_meta_info())

            if getattr(state.obj, "return_logprob", False):
                self.convert_logprob_style(
                    meta_info,
                    state,
                    state.obj.top_logprobs_num,
                    state.obj.token_ids_logprob,
                    state.obj.return_text_in_logprobs and not self.skip_tokenizer_init,
                    recv_obj,
                    i,
                )
            if (
                isinstance(state.obj, GenerateReqInput)
                and state.obj.return_sampling_mask
            ):
                output_sampling_mask = recv_obj.output_token_sampling_mask
                if output_sampling_mask is not None:
                    state.output_token_sampling_mask.extend(output_sampling_mask[i])
                    output_sampling_logprobs = recv_obj.output_token_sampling_logprobs
                    if output_sampling_logprobs is not None:
                        state.output_token_sampling_logprobs.extend(
                            output_sampling_logprobs[i]
                        )
                    meta_info["output_token_sampling_mask"] = (
                        state.output_token_sampling_mask
                    )
                    meta_info["output_token_sampling_logprobs"] = (
                        state.output_token_sampling_logprobs
                    )
                    meta_info["output_token_sampling_mask_length"] = len(
                        state.output_token_sampling_mask
                    )

            if not isinstance(recv_obj, BatchEmbeddingOutput):
                meta_info.update(
                    {
                        "reasoning_tokens": recv_obj.reasoning_tokens[i],
                        "completion_tokens": recv_obj.completion_tokens[i],
                        "cached_tokens": recv_obj.cached_tokens[i],
                    }
                )
                if (
                    recv_obj.weight_versions is not None
                    and (spans := recv_obj.weight_versions[i]) is not None
                ):
                    add_weight_versions_to_meta_info(
                        meta_info,
                        spans,
                        num_output_tokens=recv_obj.completion_tokens[i],
                    )
                # Add detailed cache breakdown if available
                if (
                    hasattr(recv_obj, "cached_tokens_details")
                    and recv_obj.cached_tokens_details
                ):
                    meta_info["cached_tokens_details"] = recv_obj.cached_tokens_details[
                        i
                    ]
                if customized_info is not None:
                    self.update_request_meta_info(
                        meta_info,
                        state,
                        customized_info,
                        i,
                        recv_obj.finished_reasons[i],
                    )

                # Add multimodal prompt token counts only for requests that
                # actually consumed them, so plain-text meta_info stays unchanged.
                image_tokens_list = getattr(recv_obj, "image_tokens", None)
                audio_tokens_list = getattr(recv_obj, "audio_tokens", None)
                video_tokens_list = getattr(recv_obj, "video_tokens", None)
                if image_tokens_list and image_tokens_list[i]:
                    meta_info["image_tokens"] = image_tokens_list[i]
                if audio_tokens_list and audio_tokens_list[i]:
                    meta_info["audio_tokens"] = audio_tokens_list[i]
                if video_tokens_list and video_tokens_list[i]:
                    meta_info["video_tokens"] = video_tokens_list[i]

            if getattr(recv_obj, "output_hidden_states", None):
                hidden_states = recv_obj.output_hidden_states[i]
                if hidden_states is not None:
                    meta_info["hidden_states"] = hidden_states
            if getattr(recv_obj, "routed_experts", None):
                val = recv_obj.routed_experts[i]
                if val is not None:
                    # BatchStrOutput is pre-encoded by the detokenizer;
                    # BatchTokenIDOutput (skip_tokenizer_init) bypasses it.
                    if isinstance(val, torch.Tensor):
                        val = pybase64.b64encode(val.numpy().tobytes()).decode("utf-8")
                    meta_info["routed_experts"] = val
            if getattr(recv_obj, "indexer_topk", None):
                val = recv_obj.indexer_topk[i]
                if val is not None:
                    if isinstance(val, torch.Tensor):
                        val = pybase64.b64encode(val.numpy().tobytes()).decode("utf-8")
                    meta_info["indexer_topk"] = val
            if getattr(recv_obj, "dp_ranks", None):
                meta_info["dp_rank"] = recv_obj.dp_ranks[i]

            state.finished = recv_obj.finished_reasons[i] is not None
                # 中译：文本输出分支。delta_* 是本批新增的增量；先累积到 state，再按三种情形产出 out_dict：

            # Must run after meta_info is fully populated.
            beam_out_dict = try_build_beam_search_out_dict(recv_obj, i, meta_info)
            if beam_out_dict is not None:
                out_dict = beam_out_dict
            elif isinstance(recv_obj, BatchStrOutput):
                # Not all request types have `stream` (e.g., EmbeddingReqInput). Default to non-streaming.
                is_stream = getattr(state.obj, "stream", False)
                incremental = is_stream and self.incremental_streaming_output
                delta_text = recv_obj.output_strs[i]
                delta_output_ids = list(recv_obj.output_ids[i])
                output_offset = state.last_output_offset
                state.append_text(delta_text)
                state.output_ids.extend(delta_output_ids)

                if is_stream:
                    if incremental:
                        output_token_ids = delta_output_ids
                        _slice_streaming_output_meta_info(
                            meta_info,
                            output_offset,
                            state.customized_info_accumulated.keys(),
                        )
                        state.last_output_offset = len(state.output_ids)
                        out_dict = {
                            "text": delta_text,
                            "output_ids": output_token_ids,
                            "meta_info": meta_info,
                        }
                    elif state.finished:
                        out_dict = {
                            "text": state.get_text(),
                            "output_ids": state.output_ids.copy(),
                            "meta_info": meta_info,
                        }
                    else:
                        # Non-incremental intermediate: pass reference (no
                        # copy) and defer text to _wait_one_response to avoid
                        # O(n) per-step cost that compounds to O(n^2).
                        # 中译：非增量中间步——直接传引用（不拷贝），text 置 None 延迟到 _wait_one_response
                        #       再 get_text()，避免每步重建整串导致 O(n^2) 总开销。
                        out_dict = {
                            "text": None,
                            "output_ids": state.output_ids,
                            "meta_info": meta_info,
                        }
                elif state.finished:
                    out_dict = {
                        "text": state.get_text(),
                        "output_ids": state.output_ids.copy(),
                        "meta_info": meta_info,
                    }
                else:
                    out_dict = None
                if out_dict is not None and state.prompt_token_ids is not None:
                    out_dict["prompt_token_ids"] = state.prompt_token_ids
            elif isinstance(recv_obj, BatchTokenIDOutput):
                # 中译：token id 输出分支（skip_tokenizer_init 时跳过 detokenizer，直接回传 token id），
                #       逻辑与文本分支同构，只是没有 text 字段。
                is_stream = getattr(state.obj, "stream", False)
                incremental = is_stream and self.incremental_streaming_output
                delta_output_ids = list(recv_obj.output_ids[i])
                output_offset = state.last_output_offset
                state.output_ids.extend(delta_output_ids)

                if is_stream:
                    if incremental:
                        output_token_ids = delta_output_ids
                        _slice_streaming_output_meta_info(
                            meta_info,
                            output_offset,
                            state.customized_info_accumulated.keys(),
                        )
                        state.last_output_offset = len(state.output_ids)
                        out_dict = {
                            "output_ids": output_token_ids,
                            "meta_info": meta_info,
                        }
                    elif state.finished:
                        out_dict = {
                            "output_ids": state.output_ids.copy(),
                            "meta_info": meta_info,
                        }
                    else:
                        out_dict = {
                            "output_ids": state.output_ids,
                            "meta_info": meta_info,
                        }
                elif state.finished:
                    out_dict = {
                        "output_ids": state.output_ids.copy(),
                        "meta_info": meta_info,
                    }
                else:
                    out_dict = None
                if out_dict is not None and state.prompt_token_ids is not None:
                    out_dict["prompt_token_ids"] = state.prompt_token_ids
            else:
                # 中译：embedding 输出分支——直接返回向量（无流式/增量概念）。
                assert isinstance(recv_obj, BatchEmbeddingOutput)
                out_dict = {
                    "embedding": recv_obj.embeddings[i],
                    "meta_info": meta_info,
                }
                # Unpack pooled hidden states (PHS).
                # See paired sender logic in output_streamer.py.
                #   Stacked:     len == 1 and N > 1 → unwrap the tensor
                #   Non-stacked: len == N → index directly
                pooled_hidden_states = recv_obj.pooled_hidden_states
                if pooled_hidden_states is not None:
                    if len(pooled_hidden_states) == 1 and len(recv_obj.rids) > 1:
                        pooled_hidden_states = pooled_hidden_states[0]
                    if pooled_hidden_states[i] is not None:
                        out_dict["pooled_hidden_state"] = pooled_hidden_states[i]

            # Set first_token_time on the first output batch.
            # This is the single write point for first_token_time.
            if state.time_stats.first_token_time == 0.0:
                state.time_stats.set_first_token_time()

            if state.finished:
                # 中译：请求结束——补齐 e2e 延迟/推测解码指标，从 rid_to_state 删除该状态并释放 LoRA。
                if state.time_stats.trace_ctx.tracing_enable:
                    state.time_stats.trace_ctx.trace_set_root_attrs(
                        self.convert_to_span_attrs(state, recv_obj, i)
                    )
                state.time_stats.set_finished_time()
                meta_info["e2e_latency"] = state.time_stats.get_e2e_latency()

                if get_spec().speculative_algorithm:
                    self._calculate_spec_decoding_metrics(meta_info, recv_obj, i)
                if self.enable_metrics:
                    scheduler_time_stats = (
                        recv_obj.time_stats[i]
                        if recv_obj.time_stats is not None
                        else None
                    )
                    completion_tokens = (
                        recv_obj.completion_tokens[i]
                        if not isinstance(recv_obj, BatchEmbeddingOutput)
                        else 0
                    )
                    meta_info.update(
                        state.time_stats.convert_to_output_meta_info(
                            scheduler_time_stats, completion_tokens
                        )
                    )

                del self.rid_to_state[rid]

                # Mark ongoing LoRA request as finished.
                if self.enable_lora and state.obj.lora_path:
                    asyncio.create_task(self.lora_registry.release(state.obj.lora_id))

            if out_dict is not None:
                # 中译：把本次输出排入该请求的 out_list，并把 state 记入待通知集合。
                #       攒够 batch_notify_size 个再统一 set event 唤醒，减少调度抖动；
                #       await sleep(0) 让出事件循环，给消费者协程运行机会。
                state.out_list.append(out_dict)
                pending_notify[rid] = state

                if len(pending_notify) >= batch_notify_size:
                    for s in pending_notify.values():
                        s.event.set()
                    pending_notify = {}
                    await asyncio.sleep(0)

            if self.enable_metrics and state.obj.log_metrics:
                self.collect_metrics(state, recv_obj, i)
            if self.dump_requests_folder and state.finished and state.obj.log_metrics:
                self.dump_requests(state, out_dict)
            if self.crash_dump_folder and state.finished and state.obj.log_metrics:
                self.record_request_for_crash_dump(state, out_dict)

        # handle_loop awaits next recv immediately
        # 中译：处理完整批后，把剩余未达攒批阈值的 state 也统一唤醒（handle_loop 随即去收下一批）。
        for s in pending_notify.values():
            s.event.set()

    @staticmethod
    def _accumulate_request_meta_info(
        meta_info: dict,
        state: ReqState,
        key: str,
        values: list,
    ) -> None:
        accumulated = state.customized_info_accumulated.setdefault(key, [])
        accumulated.extend(values)
        meta_info[key] = accumulated

    def update_request_meta_info(
        self,
        meta_info: dict,
        state: ReqState,
        customized_info: dict,
        index: int,
        finish_reason: Optional[dict],
    ) -> None:
        """Accumulate metadata; subclasses may use finish_reason for terminal data."""
        for key, values in customized_info.items():
            self._accumulate_request_meta_info(
                meta_info,
                state,
                key,
                values[index],
            )

    def add_logprob_to_meta_info(
        self,
        meta_info: dict,
        state: ReqState,
        top_logprobs_num: int,
        token_ids_logprob: List[int],
        return_text_in_logprobs: bool,
    ):
        # 中译：把累积的 logprob（含 top-k、指定 token_ids 的 logprob）反向解码为文本后写入 meta_info。
        #       内部只对「新增部分」做 detokenize（按已处理长度增量追加），避免重复解码。
        # 1. Handle regular logprobs
        if len(state.input_token_logprobs_val) > len(state.input_token_logprobs):
            state.input_token_logprobs.extend(
                self.detokenize_logprob_tokens(
                    state.input_token_logprobs_val[len(state.input_token_logprobs) :],
                    state.input_token_logprobs_idx[len(state.input_token_logprobs) :],
                    return_text_in_logprobs,
                )
            )

        if len(state.output_token_logprobs_val) > len(state.output_token_logprobs):
            state.output_token_logprobs.extend(
                self.detokenize_logprob_tokens(
                    state.output_token_logprobs_val[len(state.output_token_logprobs) :],
                    state.output_token_logprobs_idx[len(state.output_token_logprobs) :],
                    return_text_in_logprobs,
                )
            )

        meta_info["input_token_logprobs"] = state.input_token_logprobs
        meta_info["output_token_logprobs"] = state.output_token_logprobs
        meta_info["output_token_logprobs_length"] = len(state.output_token_logprobs)

        # 2. Handle top logprobs
        if top_logprobs_num > 0:
            # Guarded by the caller's return_logprob check, so obj is a
            # GenerateReqInput here.
            use_flat = state.obj.return_flat_raw_top_logprobs
            if use_flat and state.input_top_logprobs_scheduler_flat is not None:
                # The scheduler already assembled the flat arrays (sent once
                # at prefill completion); encode them directly.
                if state.input_top_logprobs_flat_fields is None:
                    val_arr, idx_arr, null_prefix = (
                        state.input_top_logprobs_scheduler_flat
                    )
                    state.input_top_logprobs_flat_fields = (
                        _build_flat_input_top_logprobs_fields_from_arrays(
                            val_arr,
                            idx_arr,
                            null_prefix,
                            return_b64=state.obj.return_flat_raw_top_logprobs_b64,
                        )
                    )
                meta_info.update(state.input_top_logprobs_flat_fields)
            elif use_flat:
                # Flat replaces nested for the input side only.
                if state.input_top_logprobs_flat_num_rows != len(
                    state.input_top_logprobs_val
                ):
                    try:
                        state.input_top_logprobs_flat_fields = (
                            _build_flat_input_top_logprobs_fields(
                                state.input_top_logprobs_val,
                                state.input_top_logprobs_idx,
                                top_logprobs_num,
                                return_b64=state.obj.return_flat_raw_top_logprobs_b64,
                            )
                        )
                    except ValueError as e:
                        # A raise here would disrupt unrelated requests in the
                        # shared batch-output loop; degrade to nested instead.
                        state.input_top_logprobs_flat_fields = None
                        logger.error(
                            "Falling back to nested input top logprobs for "
                            "rid=%s: %s",
                            meta_info.get("id"),
                            e,
                        )
                    state.input_top_logprobs_flat_num_rows = len(
                        state.input_top_logprobs_val
                    )
                if state.input_top_logprobs_flat_fields is not None:
                    meta_info.update(state.input_top_logprobs_flat_fields)
                else:
                    use_flat = False
            if not use_flat:
                if len(state.input_top_logprobs_val) > len(state.input_top_logprobs):
                    state.input_top_logprobs.extend(
                        self.detokenize_top_logprobs_tokens(
                            state.input_top_logprobs_val[
                                len(state.input_top_logprobs) :
                            ],
                            state.input_top_logprobs_idx[
                                len(state.input_top_logprobs) :
                            ],
                            return_text_in_logprobs,
                        )
                    )
                meta_info["input_top_logprobs"] = state.input_top_logprobs
            if len(state.output_top_logprobs_val) > len(state.output_top_logprobs):
                state.output_top_logprobs.extend(
                    self.detokenize_top_logprobs_tokens(
                        state.output_top_logprobs_val[len(state.output_top_logprobs) :],
                        state.output_top_logprobs_idx[len(state.output_top_logprobs) :],
                        return_text_in_logprobs,
                    )
                )
            meta_info["output_top_logprobs"] = state.output_top_logprobs

        # 3. Handle token_ids_logprob
        if token_ids_logprob is not None:
            if len(state.input_token_ids_logprobs_val) > len(
                state.input_token_ids_logprobs
            ):
                state.input_token_ids_logprobs.extend(
                    self.detokenize_top_logprobs_tokens(
                        state.input_token_ids_logprobs_val[
                            len(state.input_token_ids_logprobs) :
                        ],
                        state.input_token_ids_logprobs_idx[
                            len(state.input_token_ids_logprobs) :
                        ],
                        return_text_in_logprobs,
                    )
                )
            if len(state.output_token_ids_logprobs_val) > len(
                state.output_token_ids_logprobs
            ):
                state.output_token_ids_logprobs.extend(
                    self.detokenize_top_logprobs_tokens(
                        state.output_token_ids_logprobs_val[
                            len(state.output_token_ids_logprobs) :
                        ],
                        state.output_token_ids_logprobs_idx[
                            len(state.output_token_ids_logprobs) :
                        ],
                        return_text_in_logprobs,
                    )
                )

            meta_info["input_token_ids_logprobs"] = state.input_token_ids_logprobs
            meta_info["output_token_ids_logprobs"] = state.output_token_ids_logprobs

    def convert_logprob_style(
        self,
        meta_info: dict,
        state: ReqState,
        top_logprobs_num: int,
        token_ids_logprob: List[int],
        return_text_in_logprobs: bool,
        recv_obj: BatchStrOutput,
        recv_obj_index: int,
    ):
        # 中译：把本批 recv_obj 中第 recv_obj_index 个请求的各类 logprob 增量累积到 state，
        if (
            recv_obj.input_token_logprobs_val is not None
            and len(recv_obj.input_token_logprobs_val) > 0
            and recv_obj.input_token_logprobs_val[recv_obj_index] is not None
        ):
            state.input_token_logprobs_val.extend(
                recv_obj.input_token_logprobs_val[recv_obj_index]
            )
            state.input_token_logprobs_idx.extend(
                recv_obj.input_token_logprobs_idx[recv_obj_index]
            )
        if (
            recv_obj.output_token_logprobs_val is not None
            and recv_obj.output_token_logprobs_val[recv_obj_index] is not None
        ):
            state.output_token_logprobs_val.extend(
                recv_obj.output_token_logprobs_val[recv_obj_index]
            )
            state.output_token_logprobs_idx.extend(
                recv_obj.output_token_logprobs_idx[recv_obj_index]
            )

        if top_logprobs_num > 0:
            if len(recv_obj.input_top_logprobs_val) > 0:
                state.input_top_logprobs_val.extend(
                    recv_obj.input_top_logprobs_val[recv_obj_index]
                )
                state.input_top_logprobs_idx.extend(
                    recv_obj.input_top_logprobs_idx[recv_obj_index]
                )
            if (
                recv_obj.input_top_logprobs_val_flat is not None
                and recv_obj.input_top_logprobs_val_flat[recv_obj_index] is not None
            ):
                state.input_top_logprobs_scheduler_flat = (
                    recv_obj.input_top_logprobs_val_flat[recv_obj_index],
                    recv_obj.input_top_logprobs_idx_flat[recv_obj_index],
                    recv_obj.input_top_logprobs_flat_null_prefix[recv_obj_index],
                )
            state.output_top_logprobs_val.extend(
                recv_obj.output_top_logprobs_val[recv_obj_index]
            )
            state.output_top_logprobs_idx.extend(
                recv_obj.output_top_logprobs_idx[recv_obj_index]
            )

        if token_ids_logprob is not None:
            if len(recv_obj.input_token_ids_logprobs_val) > 0:
                state.input_token_ids_logprobs_val.extend(
                    recv_obj.input_token_ids_logprobs_val[recv_obj_index]
                )
                state.input_token_ids_logprobs_idx.extend(
                    recv_obj.input_token_ids_logprobs_idx[recv_obj_index]
                )
            state.output_token_ids_logprobs_val.extend(
                recv_obj.output_token_ids_logprobs_val[recv_obj_index]
            )
            state.output_token_ids_logprobs_idx.extend(
                recv_obj.output_token_ids_logprobs_idx[recv_obj_index]
            )

        self.add_logprob_to_meta_info(
            meta_info,
            state,
            state.obj.top_logprobs_num,
            state.obj.token_ids_logprob,
            return_text_in_logprobs,
        )

    def detokenize_logprob_tokens(
        self,
        token_logprobs_val: List[float],
        token_logprobs_idx: List[int],
        decode_to_text: bool,
    ):
        # 中译：把 (logprob 值, token id) 配对成三元组 (logprob, token_id, text)。
        #       decode_to_text=False 时 text 置 None；否则逐 token 解码出文本。
        if not decode_to_text:
            return [
                (logprob, token_id, None)
                for logprob, token_id in zip(token_logprobs_val, token_logprobs_idx)
            ]
        else:
            assert self.tokenizer is not None
            # In transformers v5, batch_decode([1, 2, 3]) concatenates all tokens
            # into one string. Wrap each ID in its own list so they decode separately.
            token_texts = self.tokenizer.batch_decode(
                [[idx] for idx in token_logprobs_idx]
            )
            return list(zip(token_logprobs_val, token_logprobs_idx, token_texts))

    def detokenize_top_logprobs_tokens(
        self,
        token_logprobs_val: List[float],
        token_logprobs_idx: List[int],
        decode_to_text: bool,
    ):
        # TODO: The current implementation only batches the detokenization for top-k tokens per single position.
        # We should batch all top-k tokens in all positions.
        ret = []
        for i in range(len(token_logprobs_val)):
            if token_logprobs_val[i]:
                ret.append(
                    self.detokenize_logprob_tokens(
                        token_logprobs_val[i], token_logprobs_idx[i], decode_to_text
                    )
                )
            else:
                ret.append(None)
        return ret

    def _calculate_spec_decoding_metrics(
        self,
        meta_info: Dict[str, Any],
        recv_obj: Union[
            BatchStrOutput,
            BatchEmbeddingOutput,
            BatchTokenIDOutput,
        ],
        i: int,
    ) -> None:
        """Calculate speculative decoding metrics, such as acceptance rate and acceptance length metrics.

        中译：计算推测解码（speculative decoding）指标并写入 meta_info：
              接受率（accept_rate = 被接受 draft 数 / 提议 draft 数）、
              平均接受长度（accept_length = completion_tokens / 验证步数，含 bonus token）、
              以及接受数量的直方图等。
        """
        if (
            hasattr(recv_obj, "spec_verify_ct")
            and recv_obj.spec_verify_ct[i] > 0
            and hasattr(recv_obj, "spec_num_correct_drafts")
            and len(recv_obj.spec_num_correct_drafts) > i
        ):
            # Total number of proposed draft tokens per request.
            num_proposed_drafts = recv_obj.spec_verify_ct[i] * (
                get_spec().speculative_num_draft_tokens - 1
            )
            num_correct_drafts = recv_obj.spec_num_correct_drafts[i]

            # Calculate per-request acceptance rate and average acceptance length.
            if num_proposed_drafts > 0:
                # accept_rate: num_correct_drafts / num_proposed_drafts (strict count, no bonus).
                meta_info["spec_accept_rate"] = num_correct_drafts / num_proposed_drafts
                # accept_length: completion_tokens / verify_ct (includes bonus token).
                meta_info["spec_accept_length"] = (
                    recv_obj.completion_tokens[i] / recv_obj.spec_verify_ct[i]
                )

                meta_info["spec_num_correct_drafts"] = num_correct_drafts
                meta_info["spec_num_proposed_drafts"] = num_proposed_drafts
                meta_info["spec_verify_ct"] = recv_obj.spec_verify_ct[i]

                if (
                    getattr(recv_obj, "spec_num_cap_tokens", None) is not None
                    and len(recv_obj.spec_num_cap_tokens) > i
                    and recv_obj.spec_num_cap_tokens[i] > 0
                ):
                    meta_info["spec_cap_length"] = (
                        recv_obj.spec_num_cap_tokens[i] / recv_obj.spec_verify_ct[i]
                    )
                if (
                    _ragged_verify_cap_accept()
                    and getattr(recv_obj, "spec_num_block_accept_tokens", None)
                    is not None
                    and len(recv_obj.spec_num_block_accept_tokens) > i
                ):
                    meta_info["spec_block_accept_length"] = (
                        recv_obj.spec_num_block_accept_tokens[i]
                        / recv_obj.spec_verify_ct[i]
                    )

                # FIXME: backward-compat aliases, remove in next release.
                meta_info["spec_accepted_drafts"] = num_correct_drafts
                meta_info["spec_proposed_drafts"] = num_proposed_drafts

            # Acceptance histogram: tracks how many decoding steps accepted a certain number of draft tokens.
            if (
                recv_obj.spec_correct_drafts_histogram
                and len(recv_obj.spec_correct_drafts_histogram) > i
                and recv_obj.spec_correct_drafts_histogram[i]
            ):
                meta_info["spec_correct_drafts_histogram"] = (
                    recv_obj.spec_correct_drafts_histogram[i]
                )
                # FIXME: backward-compat alias, remove in next release.
                meta_info["spec_accept_histogram"] = (
                    recv_obj.spec_correct_drafts_histogram[i]
                )
            if (
                getattr(recv_obj, "spec_cap_lens_histogram", None)
                and len(recv_obj.spec_cap_lens_histogram) > i
                and recv_obj.spec_cap_lens_histogram[i]
            ):
                meta_info["spec_cap_lens_histogram"] = recv_obj.spec_cap_lens_histogram[
                    i
                ]

    def _request_has_grammar(self, obj: GenerateReqInput) -> bool:
        return (
            obj.sampling_params.get("json_schema", None)
            or obj.sampling_params.get("regex", None)
            or obj.sampling_params.get("ebnf", None)
            or obj.sampling_params.get("structural_tag", None)
        )

    def collect_metrics(self, state: ReqState, recv_obj: BatchStrOutput, i: int):
        # 中译：采集请求级指标。首个 token 上报 TTFT（首 token 延迟），后续上报 ITL（token 间延迟），
        #       请求结束时上报一次完整请求的 prompt/completion/cached token 数与 e2e 延迟等。
        completion_tokens = (
            recv_obj.completion_tokens[i]
            if getattr(recv_obj, "completion_tokens", None)
            else 0
        )

        custom_labels = getattr(state.obj, "custom_labels", None)
        labels = dict(self.metrics_collector.labels)
        if custom_labels:
            labels.update(custom_labels)
        if self.enable_priority_scheduling:
            priority = getattr(state.obj, "priority", None)
            if priority is not None:
                labels["priority"] = str(priority)
        if (
            not state.ttft_observed
            and self.disaggregation_mode != DisaggregationMode.PREFILL
        ):
            state.ttft_observed = True
            state.last_completion_tokens = completion_tokens
            self.metrics_collector.observe_time_to_first_token(
                labels,
                state.time_stats.get_first_token_latency(),
                stream=getattr(state.obj, "stream", False),
            )
        else:
            num_new_tokens = completion_tokens - state.last_completion_tokens
            if num_new_tokens:
                self.metrics_collector.observe_inter_token_latency(
                    labels,
                    state.time_stats.get_interval(),
                    num_new_tokens,
                )
                state.time_stats.set_last_time()
                state.last_completion_tokens = completion_tokens

        if state.finished:
            # Get detailed cache breakdown if available
            cached_tokens_details = None
            if (
                hasattr(recv_obj, "cached_tokens_details")
                and recv_obj.cached_tokens_details
            ):
                cached_tokens_details = recv_obj.cached_tokens_details[i]

            spec_verify_ct = (
                recv_obj.spec_verify_ct[i]
                if hasattr(recv_obj, "spec_verify_ct")
                and recv_obj.spec_verify_ct
                and len(recv_obj.spec_verify_ct) > i
                else 0
            )

            self.metrics_collector.observe_one_finished_request(
                labels,
                recv_obj.prompt_tokens[i],
                completion_tokens,
                recv_obj.cached_tokens[i],
                state.time_stats.get_e2e_latency(),
                self._request_has_grammar(state.obj),
                cached_tokens_details,
                spec_verify_ct=spec_verify_ct,
                is_streaming=getattr(state.obj, "stream", False),
            )

    def dump_requests(self, state: ReqState, out_dict: dict):
        # 中译：把已完成请求（请求对象 + 输出 + 时间戳）攒入列表，达到阈值后异步 pickle 到磁盘
        #       （用于离线分析/复现）。会按配置剔除体积大的 meta 字段（如 hidden_states）。
        if self.dump_requests_exclude_meta_keys and isinstance(
            out_dict.get("meta_info"), dict
        ):
            exclude = self.dump_requests_exclude_meta_keys
            if any(k in out_dict["meta_info"] for k in exclude):
                filtered_meta = {
                    k: v for k, v in out_dict["meta_info"].items() if k not in exclude
                }
                out_dict = {**out_dict, "meta_info": filtered_meta}

        self.dump_request_list.append(
            (
                state.obj,
                out_dict,
                convert_time_to_realtime(state.time_stats.created_time),
                convert_time_to_realtime(state.time_stats.finished_time),
            )
        )

        if len(self.dump_request_list) >= self.dump_requests_threshold:
            filename = os.path.join(
                self.dump_requests_folder,
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pkl",
            )
            self._dump_data_to_file(
                data_list=self.dump_request_list,
                filename=filename,
                log_message=f"Dump {len(self.dump_request_list)} requests to {filename}",
            )
            self.dump_request_list = []

    def record_request_for_crash_dump(self, state: ReqState, out_dict: dict):
        # 中译：把完成的请求记入崩溃转储滚动缓冲区，并淘汰超过 5 分钟的旧记录；
        #       崩溃时 dump_requests_before_crash 会连同未完成请求一起落盘。
        current_time = real_time()
        self.crash_dump_request_list.append(
            (
                state.obj,
                out_dict,
                convert_time_to_realtime(state.time_stats.created_time),
                current_time,
            )
        )
        # Remove requests older than 5 minutes based on finish time
        while (
            self.crash_dump_request_list
            and current_time - self.crash_dump_request_list[0][3] >= 300
        ):
            self.crash_dump_request_list.popleft()

    def _dump_data_to_file(
        self, data_list: List[Tuple], filename: str, log_message: str
    ):
        # 中译：在后台线程把数据连同 server_args 一起 pickle 落盘；
        #       若 server_args 因 trust_remote_code 等无法序列化，则去掉它重试，保证请求数据仍能落盘。
        logger.info(log_message)
        to_dump_with_server_args = {
            "server_args": self.server_args,
            "config_updates": get_context().overrides_log(),
            "resolved_config": self._dump_config_snapshot(),
            "requests": data_list.copy(),
        }

        def background_task():
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "wb") as f:
                try:
                    pickle.dump(to_dump_with_server_args, f)
                except Exception as e:
                    # When the server is launched with --trust-remote-code,
                    # server_args sometimes fails to pickle. Retry without
                    # server_args so the request data still gets persisted.
                    logger.error(
                        f"Failed to pickle dump with server_args: {e!r}; "
                        "retrying without server_args"
                    )
                    f.seek(0)
                    f.truncate()
                    to_dump_with_server_args["server_args"] = None
                    # The snapshot copies the same object field by field.
                    to_dump_with_server_args["resolved_config"] = None
                    pickle.dump(to_dump_with_server_args, f)

        asyncio.create_task(asyncio.to_thread(background_task))

    def dump_requests_before_crash(
        self, hostname: str = os.getenv("HOSTNAME", socket.gethostname())
    ):
        # 中译：崩溃/退出前的转储。把已完成（滚动缓冲）与未完成（rid_to_state 中）的请求一起落盘，
        #       并按需触发 py-spy 栈转储与 CUDA coredump 以便事后排障。用 crash_dump_performed 保证只执行一次。
        should_dump_pyspy = envs.SGLANG_PYSPY_DUMP_BEFORE_CRASH.get()
        should_dump_cuda_coredump = envs.SGLANG_CUDA_COREDUMP_BEFORE_CRASH.get()
        should_dump_diagnostics = should_dump_pyspy or should_dump_cuda_coredump
        if not self.crash_dump_folder and not should_dump_diagnostics:
            return

        if self.crash_dump_performed:
            logger.info(
                "SIGTERM/SIGQUIT/Exception triggered, but crash dump already performed, skipping."
            )
            return
        else:
            self.crash_dump_performed = True

        # Dump requests info
        if self.crash_dump_folder:
            logger.error(f"Dumping requests before crash. {self.crash_dump_folder=}")

            # Add finished requests from crash_dump_request_list
            data_to_dump = []
            if self.crash_dump_request_list:
                data_to_dump.extend(self.crash_dump_request_list)

            # Add unfinished requests from rid_to_state
            unfinished_requests = []
            for rid, state in self.rid_to_state.items():
                if not state.finished:
                    state.time_stats.set_finished_time()
                    unfinished_requests.append(
                        (
                            state.obj,
                            (
                                state.out_list[-1]
                                if state.out_list
                                else state.get_crash_dump_output()
                            ),
                            convert_time_to_realtime(state.time_stats.created_time),
                            convert_time_to_realtime(state.time_stats.finished_time),
                        )
                    )
            if unfinished_requests:
                data_to_dump.extend(unfinished_requests)

            if data_to_dump:
                # Create a file
                filename = os.path.join(
                    self.crash_dump_folder,
                    hostname,
                    f'crash_dump_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pkl',
                )
                os.makedirs(os.path.dirname(filename), exist_ok=True)

                # Write the data to the file
                data_to_dump_with_server_args = {
                    "server_args": self.server_args,
                    "config_updates": get_context().overrides_log(),
                    "resolved_config": self._dump_config_snapshot(),
                    "requests": data_to_dump,
                    "launch_command": " ".join(sys.argv),
                }
                with open(filename, "wb") as f:
                    try:
                        pickle.dump(data_to_dump_with_server_args, f)
                    except Exception as e:
                        # When the server is launched with --trust-remote-code,
                        # server_args sometimes fails to pickle. Retry without
                        # server_args so the request data still gets persisted.
                        logger.error(
                            f"Failed to pickle dump with server_args: {e!r}; "
                            "retrying without server_args"
                        )
                        f.seek(0)
                        f.truncate()
                        data_to_dump_with_server_args["server_args"] = None
                        # The snapshot copies the same object field by field.
                        data_to_dump_with_server_args["resolved_config"] = None
                        pickle.dump(data_to_dump_with_server_args, f)
                logger.error(
                    f"Dumped {len(self.crash_dump_request_list)} finished and {len(unfinished_requests)} unfinished requests before crash to {filename}"
                )

        # Dump pyspy and cuda coredump
        if should_dump_diagnostics:
            logger.info(
                "Sleeping 5 seconds before crash diagnostics to let GPU activity settle."
            )
            time.sleep(5)

            scheduler_procs = collect_scheduler_processes()
            if scheduler_procs:
                if should_dump_pyspy:
                    pyspy_dump_schedulers(scheduler_only=True)

                if should_dump_cuda_coredump:
                    trigger_cuda_user_coredump(scheduler_only=True)
                    cuda_coredump_wait_secs = (
                        envs.SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS.get()
                    )
                    if cuda_coredump_wait_secs > 0:
                        logger.info(
                            "Waiting %.1f seconds for CUDA coredumps before exiting.",
                            cuda_coredump_wait_secs,
                        )
                        time.sleep(cuda_coredump_wait_secs)
            else:
                logger.error(
                    "No live scheduler processes found; skipping py-spy and CUDA coredump."
                )

    async def sigterm_watchdog(self):
        # 中译：优雅退出看门狗。收到 SIGTERM 后 gracefully_exit 置真，循环等待在途请求排空
        #       （rid_to_state 清空）再杀进程；健康检查失败或设置强制关闭标志时则立即退出。
        while not self.gracefully_exit:
            await asyncio.sleep(5)

        # Drain requests
        while True:
            remain_num_req = len(self.rid_to_state)
            remaining_rids = list(self.rid_to_state.keys())

            if self.server_status == ServerStatus.UnHealthy:
                # if health check failed, we should exit immediately
                logger.error(
                    "Signal SIGTERM received while health check failed. Force exiting."
                )
                self.dump_requests_before_crash()
                self.force_exit_handler()
                break

            elif get_bool_env_var("SGL_FORCE_SHUTDOWN"):
                # if force shutdown flag set, exit immediately
                logger.error(
                    "Signal SIGTERM received while force shutdown flag set. Force exiting."
                )
                self.force_exit_handler()
                break

            logger.info(
                f"Gracefully exiting... Remaining number of requests {remain_num_req}. Remaining requests {remaining_rids=}."
            )
            if remain_num_req > 0:
                await asyncio.sleep(5)
            else:
                break

        # Stop the watchdog: child exits are expected during shutdown, not crashes.
        if self._subprocess_watchdog is not None:
            self._subprocess_watchdog.stop()
        # Ask schedulers to release resources in userspace and exit (see
        # ShutdownReq), then wait for them before hard-killing the rest.
        self._dispatch_to_scheduler(ShutdownReq())
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and collect_scheduler_processes():
            time.sleep(0.1)
        kill_process_tree(os.getpid(), include_parent=False, wait_timeout=60)
        sys.exit(0)

    def force_exit_handler(self):
        """Put some custom force exit logic here."""
        pass

    def _handle_abort_req(self, recv_obj: AbortReq):
        # 中译：处理调度器回传的中止消息——把该请求标记为已结束，构造带 abort finish_reason 的输出，
        #       从 rid_to_state 删除并唤醒等待协程（由 _wait_one_response 决定如何对客户端呈现）。
        if is_health_check_generate_req(recv_obj):
            return
        # Two scheduler messages can race in handle_loop for the same rid: a
        # batch output that finishes it normally (deletes rid_to_state[rid])
        # and this abort echo. If the finish wins, the rid is already gone and
        # there is nothing left to abort. Common under mass client
        # disconnects, amplified by prefix / abort_all fan-out.
        state = self.rid_to_state.get(recv_obj.rid)
        if state is None:
            logger.info(
                "Abort request for rid=%s not found in rid_to_state; "
                "likely already finished/removed.",
                recv_obj.rid,
            )
            return
        state.finished = True
        state.time_stats.set_finished_time()

        abort_message = recv_obj.abort_message or "Abort in waiting queue"
        finish_reason = {
            "type": "abort",
            "message": abort_message,
        }
        if recv_obj.finished_reason:
            finish_reason = recv_obj.finished_reason
        meta_info = {
            "id": recv_obj.rid,
            "finish_reason": finish_reason,
            "weight_version": self.config_value("weight_version"),
            "e2e_latency": state.time_stats.get_e2e_latency(),
        }
        if recv_obj.weight_versions is not None:
            add_weight_versions_to_meta_info(
                meta_info,
                recv_obj.weight_versions,
                num_output_tokens=len(state.output_ids),
            )
        is_stream = getattr(state.obj, "stream", False)
        if getattr(state.obj, "return_logprob", False):
            self.add_logprob_to_meta_info(
                meta_info,
                state,
                state.obj.top_logprobs_num,
                state.obj.token_ids_logprob,
                state.obj.return_text_in_logprobs
                and not get_serving().skip_tokenizer_init,
            )

        output_ids = state.output_ids
        meta_info["completion_tokens"] = len(output_ids)
        if is_stream:
            output_ids = [output_ids[-1]] if len(output_ids) > 0 else []
        out = {
            "text": state.get_text(),
            "output_ids": output_ids,
            "meta_info": meta_info,
        }
        del self.rid_to_state[recv_obj.rid]

        state.out_list.append(out)
        state.event.set()

    def update_active_ranks(self, ranks: ActiveRanksOutput):
        self._dispatch_to_scheduler(ranks)

    def forward_elastic_scale_update(self, msg: ElasticScaleUpdateReq):
        if not msg.success:
            self.elastic_pending_ep_size = None
            self.elastic_scale_phase = "failed"
            self.elastic_last_error = msg.error
            return

        self._dispatch_to_scheduler(msg)
        self.elastic_worker_count = msg.effective_ep_size
        self.elastic_pending_ep_size = None
        self.elastic_scale_phase = "serving_expanded"
        self.elastic_last_error = None
        self.update_control_communicator_fan_out(msg.effective_ep_size)

    def get_elastic_ep_state(self):
        return {
            "is_scaling_elastic_ep": self.elastic_pending_ep_size is not None,
            "effective_ep_size": self.elastic_worker_count,
            "pending_ep_size": self.elastic_pending_ep_size,
            "scale_phase": self.elastic_scale_phase,
            "last_error": self.elastic_last_error,
        }

    async def scale_elastic_ep(
        self, obj: ScaleElasticEPReqInput
    ) -> ScaleElasticEPReqOutput:
        """Send a scale request to every DP scheduler."""
        if self.elastic_pending_ep_size is not None:
            return ScaleElasticEPReqOutput(
                success=False,
                message=(
                    "A previous scale operation has not completed yet. Wait until "
                    "all pending ranks have joined before issuing another scale."
                ),
                old_ep_size=self.elastic_worker_count,
                new_ep_size=obj.new_ep_size,
                pending_ep_size=self.elastic_pending_ep_size,
                scale_phase=self.elastic_scale_phase,
            )
        self.auto_create_handle_loop()
        responses: List[ScaleElasticEPReqOutput] = (
            await self.scale_elastic_ep_communicator(obj)
        )
        for res in responses:
            if not res.success:
                self.elastic_scale_phase = res.scale_phase
                self.elastic_pending_ep_size = res.pending_ep_size
                self.elastic_last_error = res.message
                return res
        self.elastic_pending_ep_size = responses[0].pending_ep_size
        self.elastic_scale_phase = responses[0].scale_phase
        self.elastic_last_error = None
        return responses[0]

    def _handle_open_session_req_output(self, recv_obj):
        # 中译：开会话结果回调——把 session_id（成功）或 None（失败）填入对应 future，唤醒等待方。
        future = self.session_futures.get(recv_obj.session_id)
        if future is None:
            logger.warning(
                "Open session response arrived after waiter cleanup: %s",
                recv_obj.session_id,
            )
            return
        if not future.done():
            future.set_result(recv_obj.session_id if recv_obj.success else None)

    def _handle_update_weights_from_disk_req_output(self, recv_obj):
        # 中译：权重更新结果回调。dp_size==1 直接 set future；dp_size>1 时先攒齐各 DP rank 的结果，
        if self.model_update_expected_workers == 1:
            self.model_update_result.set_result(recv_obj)
        else:
            self.model_update_tmp.append(recv_obj)
            if len(self.model_update_tmp) == self.model_update_expected_workers:
                self.model_update_result.set_result(self.model_update_tmp)

    async def _validate_and_resolve_lora(
        self, obj: Union[GenerateReqInput, EmbeddingReqInput]
    ) -> None:
        # 中译：校验并解析请求所需的 LoRA 适配器。未指定则跳过；未启用 LoRA 却请求则报错；
        #       否则把用户友好的 lora_path 名解析为内部 lora_id（必要时重新加载被淘汰的适配器）。
        if not obj.lora_path:
            return

        if not self.enable_lora:
            first_adapter = (
                obj.lora_path
                if isinstance(obj.lora_path, str)
                else next((a for a in obj.lora_path if a), None)
            )

            raise ValueError(
                f"LoRA adapter '{first_adapter}' was requested, but LoRA is not enabled. "
                "Please launch the server with --enable-lora flag and preload adapters "
                "using --lora-paths or /load_lora_adapter endpoint."
            )

        await self._resolve_lora_path(obj)

    async def _resolve_lora_path(self, obj: Union[GenerateReqInput, EmbeddingReqInput]):
        # 中译：把 lora_path 解析为 lora_id。会校验唯一适配器数不超上限、按需重载被动态卸载的适配器，
        #       最后从注册表 acquire lora_id（开始追踪在途 LoRA 请求），并下发到各子请求。
        if isinstance(obj.lora_path, str):
            unique_lora_paths = set([obj.lora_path])
        else:
            unique_lora_paths = set(obj.lora_path)

        if (
            get_lora().max_loaded_loras is not None
            and len(unique_lora_paths) > get_lora().max_loaded_loras
        ):
            raise ValueError(
                f"Received request with {len(unique_lora_paths)} unique loras requested "
                f"but max loaded loras is {get_lora().max_loaded_loras}"
            )

        # Reload all existing LoRA adapters that have been dynamically unloaded
        unregistered_loras = await self.lora_registry.get_unregistered_loras(
            unique_lora_paths
        )
        for lora_path in unregistered_loras:
            if lora_path is None:
                continue

            if lora_path not in self.lora_ref_cache:
                raise ValueError(
                    f"Got LoRA adapter that has never been loaded: {lora_path}\n"
                    f"All loaded adapters: {self.lora_ref_cache.keys()}."
                )

            logger.info(f"Reloading evicted adapter: {lora_path}")
            new_lora_ref = self.lora_ref_cache[lora_path]
            load_result = await self.load_lora_adapter(
                LoadLoRAAdapterReqInput(
                    lora_name=new_lora_ref.lora_name,
                    lora_path=new_lora_ref.lora_path,
                    pinned=new_lora_ref.pinned,
                )
            )
            if (
                not load_result.success
                and "already loaded" not in load_result.error_message
            ):
                raise ValueError(
                    f"Failed to implicitly load LoRA adapter {lora_path}: {load_result.error_message}"
                )

        # Look up the LoRA ID from the registry and start tracking ongoing LoRA requests.
        obj.lora_id = await self.lora_registry.acquire(obj.lora_path)
        # Propagate lora_id to any sub-objects already cached by __getitem__.
        for i, sub_obj in obj.__dict__.get("_sub_obj_cache", {}).items():
            sub_obj.lora_id = (
                obj.lora_id[i] if isinstance(obj.lora_id, list) else obj.lora_id
            )

    def _init_req_state(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        # 中译：为请求（单条或批量的每个子请求）在 rid_to_state 中登记 ReqState。
        #       会把单/批统一成 (rid, sub_obj, bootstrap_room) 列表逐个建状态；rid 重复则报错；
        #       并初始化时间统计与（启用时）链路追踪上下文。
        created_time = obj.received_time

        external_trace_header = None
        if self.enable_trace:
            if obj.external_trace_header:
                # When the request comes from the rust grpc server or Engine there isn't a
                # real request object but we still need to propagate the trace context from
                # the trace context that is explicitly passed in
                external_trace_header = obj.external_trace_header
            elif request:
                external_trace_header = extract_trace_headers(request.headers)
                obj.external_trace_header = external_trace_header

        # Normalize single/batch into a uniform list of (rid, sub_obj, bootstrap_room)
        # 中译：把单条/批量统一成 (rid, sub_obj, bootstrap_room) 列表，便于下面统一建状态。
        if not hasattr(obj, "is_single") or obj.is_single:
            items = [(obj.rid, obj, getattr(obj, "bootstrap_room", None))]
        else:
            items = [
                (
                    obj.rid[i],
                    obj[i],
                    (
                        obj.bootstrap_room[i]
                        if hasattr(obj, "bootstrap_room") and obj.bootstrap_room
                        else None
                    ),
                )
                for i in range(len(obj.rid))
            ]

        for rid, sub_obj, bootstrap_room in items:
            if rid in self.rid_to_state:
                raise ValueError(f"Duplicate request ID detected: {rid}")
            time_stats = APIServerReqTimeStats(disagg_mode=self.disaggregation_mode)
            state = ReqState([], False, asyncio.Event(), sub_obj, time_stats)
            self.rid_to_state[rid] = state
            if self.enable_trace:
                time_stats.init_trace_ctx(rid, bootstrap_room, external_trace_header)
            time_stats.set_created_time(created_time)

    def _discard_pending_req_states(self, obj):
        """Drop rid_to_state entries created by _init_req_state for *obj*.

        Safe to call after a partial/failed dispatch: only entries still present
        are removed, and the scheduler-response path looks up state with
        ``.get(...)`` so a later output for a discarded rid is ignored, not fatal.
        """
        if not hasattr(obj, "is_single") or obj.is_single:
            rids = [obj.rid]
        else:
            rids = obj.rid
        for rid in rids:
            self.rid_to_state.pop(rid, None)

    def _should_dispatch_to_encoder(
        self, obj: Union[GenerateReqInput, EmbeddingReqInput]
    ) -> bool:
        if obj.batch_size > 1:
            logger.warning(
                "Batch request (batch_size=%d) is not supported in EPD disaggregation mode; skipping encoder dispatch.",
                obj.batch_size,
            )
            return False
        if not isinstance(obj, GenerateReqInput) or not obj.contains_mm_input():
            return False

        # Count image / video / audio items for dispatch threshold
        def _count_mm_items(data):
            return (
                len(data) if isinstance(data, list) else (1 if data is not None else 0)
            )

        total_mm_items = (
            _count_mm_items(getattr(obj, "image_data", None))
            + _count_mm_items(getattr(obj, "video_data", None))
            + _count_mm_items(getattr(obj, "audio_data", None))
        )
        return total_mm_items >= envs.SGLANG_ENCODER_DISPATCH_MIN_ITEMS.get()

    def _handle_epd_disaggregation_encode_request(
        self, obj: Union[GenerateReqInput, EmbeddingReqInput]
    ):
        """Handle EPD-disaggregation mode encoding request.

        中译：EPD（encode-prefill-decode）分离模式下处理编码请求。决定是否派发给 encoder，
              并据此设置 need_wait_for_mm_inputs 标志（供 _tokenize_one_request 选择处理路径）；
              对 zmq_to_scheduler/mooncake 后端会主动向 encoder 发送编码请求。
        """
        if isinstance(obj, GenerateReqInput) and obj.contains_mm_input():
            # dispatch to encoder by default
            should_dispatch = True
            if get_disagg().enable_adaptive_dispatch_to_encoder:
                should_dispatch = self._should_dispatch_to_encoder(obj)

            # Set need_wait_for_mm_inputs flag based on whether we dispatch to encoder
            # This flag will be used in _tokenize_one_request to determine processing path
            if should_dispatch:
                obj.need_wait_for_mm_inputs = True
                if get_disagg().encoder_transfer_backend in [
                    "zmq_to_scheduler",
                    "mooncake",
                ]:
                    time_stats_json = None
                    if self.enable_trace:
                        state = self.rid_to_state.get(obj.rid)
                        if state is not None:
                            time_stats_json = state.time_stats.encode_json()

                    self.mm_receiver.send_encode_request(
                        obj, time_stats_json=time_stats_json
                    )
            else:
                obj.need_wait_for_mm_inputs = False

    def convert_to_span_attrs(
        self,
        state: ReqState,
        recv_obj: Union[
            BatchStrOutput,
            BatchEmbeddingOutput,
            BatchTokenIDOutput,
        ],
        i: int,
    ) -> Dict[str, Any]:
        """Convert attributes to span attributes.

        中译：把请求的 token 用量、采样参数、结束原因、延迟等转换为链路追踪（trace）span 属性。
              未启用 trace 时返回空 dict。
        """
        span_attrs = {}

        if not self.enable_trace:
            return span_attrs

        # Token usage attributes
        if not isinstance(recv_obj, BatchEmbeddingOutput):
            span_attrs[SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS] = (
                recv_obj.completion_tokens[i]
            )
        span_attrs[SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS] = recv_obj.prompt_tokens[
            i
        ]
        span_attrs[SpanAttributes.GEN_AI_USAGE_CACHED_TOKENS] = recv_obj.cached_tokens[
            i
        ]

        # Request identifiers
        span_attrs[SpanAttributes.GEN_AI_REQUEST_ID] = (
            str(state.obj.rid) if state.obj.rid else None
        )

        # Sampling parameters
        sampling_params = state.obj.sampling_params or {}

        if max_new_tokens := sampling_params.get("max_new_tokens"):
            span_attrs[SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS] = max_new_tokens

        if top_p := sampling_params.get("top_p"):
            span_attrs[SpanAttributes.GEN_AI_REQUEST_TOP_P] = top_p

        if temperature := sampling_params.get("temperature"):
            span_attrs[SpanAttributes.GEN_AI_REQUEST_TEMPERATURE] = temperature

        if top_k := sampling_params.get("top_k"):
            span_attrs[SpanAttributes.GEN_AI_REQUEST_TOP_K] = top_k

        if n := sampling_params.get("n"):
            span_attrs[SpanAttributes.GEN_AI_REQUEST_N] = n

        # Response attributes
        span_attrs[SpanAttributes.GEN_AI_RESPONSE_MODEL] = self.served_model_name

        finish_reason = (
            recv_obj.finished_reasons[i].get("type")
            if recv_obj.finished_reasons[i]
            else None
        )
        if finish_reason:
            span_attrs[SpanAttributes.GEN_AI_RESPONSE_FINISH_REASONS] = json.dumps(
                [finish_reason]
            )

        # Latency attributes
        span_attrs.update(state.time_stats.convert_to_gen_ai_span_attrs())

        return span_attrs

    def _set_default_priority(self, obj: Union[GenerateReqInput, EmbeddingReqInput]):
        """Set the default priority value.

        中译：启用优先级调度且请求未显式指定优先级时，填入服务端默认优先级。
        """
        if (
            self.enable_priority_scheduling
            and obj.priority is None
            and self.default_priority_value is not None
        ):
            obj.priority = self.default_priority_value


class ServerStatus(Enum):
    # 中译：服务健康状态枚举——Up 正常、Starting 启动中、UnHealthy 不健康（健康检查失败）。
    Up = "Up"
    Starting = "Starting"
    UnHealthy = "UnHealthy"


async def print_exception_wrapper(func):
    """
    Sometimes an asyncio function does not print exception.
    We do another wrapper to handle the exception.

    中译：asyncio 协程有时不会打印未捕获异常。这层包装捕获异常、打印堆栈、做崩溃转储，
          然后杀掉整个进程树并退出，避免「静默卡死」。
    """
    try:
        await func()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"TokenizerManager hit an exception: {traceback}")
        if hasattr(func, "__self__") and isinstance(func.__self__, TokenizerManager):
            func.__self__.dump_requests_before_crash()
        kill_process_tree(os.getpid(), include_parent=True)
        sys.exit(1)


    # 中译：加载多模态 processor 的封装。若该 processor 没有 slow 版本，则自动回退到 fast 版本重试。
    # 中译：决定多模态特征张量的进程间传输方式：跨节点用默认 CPU 传输，单节点用 cuda_ipc（更快）。
def get_processor_wrapper():
    return get_processor(
        get_serving().tokenizer_path,
        tokenizer_mode=get_serving().tokenizer_mode,
        trust_remote_code=get_model().trust_remote_code,
        revision=get_model().revision,
        image_processor_backend=resolve_image_processor_backend(get_mm()),
        tokenizer_backend=get_serving().tokenizer_backend,
        model_name=get_model().model_path,
    )


class SignalHandler:
    # 中译：信号处理器。SIGTERM 触发优雅退出（置 gracefully_exit，由看门狗排空请求后退出）；
    #       运行期 SIGQUIT 通常意味着某子进程挂了，做崩溃转储后杀进程树。
    def __init__(self, tokenizer_manager: TokenizerManager):
        self.tokenizer_manager = tokenizer_manager

    def sigterm_handler(self, signum=None, frame=None):
        logger.warning(
            f"SIGTERM received. {signum=} {frame=}. Draining requests and shutting down..."
        )
        self.tokenizer_manager.gracefully_exit = True

    def running_phase_sigquit_handler(self, signum=None, frame=None):
        logger.error(
            f"SIGQUIT received. {signum=}, {frame=}. It usually means one child failed."
        )
        # Stop subprocess watchdog before killing processes to prevent false-positive
        # crash detection during normal shutdown
        if self.tokenizer_manager._subprocess_watchdog is not None:
            self.tokenizer_manager._subprocess_watchdog.stop()
        self.tokenizer_manager.dump_requests_before_crash()
        kill_process_tree(os.getpid())


# Note: request abort handling logic
# We should handle all of the following cases correctly.
# 中译：请求中止处理逻辑备忘。下表列出需正确处理的所有情形（入口类型 × 是否流式 × 请求所处阶段），
#       以及各情形下：如何中止引擎、如何取消 asyncio 任务、在何处删除 rid_to_state。
#
# | entrypoint | is_streaming | status          | abort engine    | cancel asyncio task   | rid_to_state                |
# | ---------- | ------------ | --------------- | --------------- | --------------------- | --------------------------- |
# | http       | yes          | validation      | background task | fast api              | del in _handle_abort_req    |
# | http       | yes          | waiting queue   | background task | fast api              | del in _handle_abort_req    |
# | http       | yes          | running         | background task | fast api              | del in _handle_batch_output |
# | http       | no           | validation      | http exception  | http exception        | del in _handle_abort_req    |
# | http       | no           | waiting queue   | type 1          | type 1 exception      | del in _handle_abort_req    |
# | http       | no           | running         | type 3          | type 3 exception      | del in _handle_batch_output |
#


def stamp_http_worker_ipc(obj: Any, ipc_name: str) -> None:
    if isinstance(obj, BaseReq):
        obj.http_worker_ipc = ipc_name
    elif isinstance(
        obj, (BatchTokenizedGenerateReqInput, BatchTokenizedEmbeddingReqInput)
    ):
        for req in obj:
            req.http_worker_ipc = ipc_name
    elif isinstance(obj, BaseBatchReq):
        obj.http_worker_ipcs = [ipc_name] * len(obj.rids)
