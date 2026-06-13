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
"""TokenizerManager is a process that tokenizes the text."""

# ============================================================================
# TokenizerManager 总体职责：
# TokenizerManager 运行在「前端进程」（与 HTTP server / Engine 同进程），是用户请求
# 进入推理系统的第一站，也是结果返回用户的最后一站。它的核心工作流是：
#   1. 接收上层（HTTP server 或 Engine）传入的 GenerateReqInput / EmbeddingReqInput；
#   2. 将文本/多模态输入做 tokenize（分词）与预处理，封装为 Tokenized*ReqInput；
#   3. 通过 ZMQ 把请求发送给 Scheduler 进程（运行在另一进程，负责真正的调度与推理）；
#   4. 在一个常驻的异步事件循环 handle_loop 中，从 DetokenizerManager 接收推理输出，
#      按 rid（请求 id）找到对应的 ReqState，把增量结果通过 asyncio.Event 唤醒等待协程；
#   5. 以异步生成器的形式把（流式/非流式）结果 yield 回上层。
#
# 关键设计要点：
# - 进程间通信全部走 ZMQ（PUSH/PULL），与 Scheduler、Detokenizer 解耦；
# - 每个在途请求对应一个 ReqState，rid_to_state 维护 rid -> 状态 的映射；
# - 通过 RWLock(model_update_lock) 让「权重更新」与「推理」互斥，但 LoRA 更新可与推理并发；
# - 通过 Mixin（TokenizerCommunicatorMixin / TokenizerManagerMultiItemMixin）拆分功能，
#   分别承载与 Scheduler 的各类控制面通信、以及 multi-item 评分等扩展能力。
# ============================================================================

# 标准库导入
import asyncio  # 异步事件循环、协程、Event/Condition 等并发原语
import copy
import dataclasses
import json
import logging
import os
import pickle  # 用于跨进程序列化部分对象
import signal  # 进程信号处理（优雅退出 SIGTERM/SIGQUIT 等）
import socket
import sys
import threading
from collections import deque
from contextlib import nullcontext
from datetime import datetime
from enum import Enum
from http import HTTPStatus
from typing import Any, Awaitable, Dict, List, Optional, Tuple, Union

# 第三方库导入
import fastapi
import pybase64  # 高性能 base64 编解码（多模态张量传输等场景使用）
import uvloop  # 基于 libuv 的高性能事件循环，替换默认 asyncio 循环
import zmq  # ZeroMQ：与 Scheduler / Detokenizer 进程通信的底层消息队列
import zmq.asyncio  # ZMQ 的 asyncio 集成（异步收发）
from fastapi import BackgroundTasks

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.disaggregation.encode_receiver import create_mm_receiver
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
from sglang.srt.managers.async_dynamic_batch_tokenizer import AsyncDynamicbatchTokenizer
from sglang.srt.managers.disagg_service import start_disagg_service
from sglang.srt.managers.io_struct import (
    AbortReq,
    ActiveRanksOutput,
    BatchEmbeddingOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    ConfigureLoggingReq,
    ContinueGenerationReqInput,
    EmbeddingReqInput,
    FreezeGCReq,
    GenerateReqInput,
    HealthCheckOutput,
    LoadLoRAAdapterReqInput,
    OpenSessionReqOutput,
    PauseGenerationReqInput,
    SessionParams,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    WatchLoadUpdateReq,
)
from sglang.srt.managers.mm_utils import TensorTransportMode, wrap_shm_features
from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
from sglang.srt.managers.schedule_batch import MultimodalDataItem
from sglang.srt.managers.scheduler import is_health_check_generate_req
from sglang.srt.managers.scheduler_input_blocker import input_blocker_guard_region
from sglang.srt.managers.tokenizer_communicator_mixin import TokenizerCommunicatorMixin
from sglang.srt.managers.tokenizer_manager_multiitem_mixin import (
    TokenizerManagerMultiItemMixin,
)
from sglang.srt.observability.cpu_monitor import start_cpu_monitor_thread
from sglang.srt.observability.metrics_collector import TokenizerMetricsCollector
from sglang.srt.observability.req_time_stats import (
    APIServerReqTimeStats,
    calibrate_time_diff,
    convert_time_to_realtime,
    real_time,
    set_time_batch,
)
from sglang.srt.observability.request_metrics_exporter import (
    RequestMetricsExporterManager,
)
from sglang.srt.observability.trace import SpanAttributes, extract_trace_headers
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import (
    PortArgs,
    ServerArgs,
    set_global_server_args_for_tokenizer,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    configure_gc_warning,
    freeze_gc,
    get_bool_env_var,
    get_or_create_event_loop,
    kill_process_tree,
)
from sglang.srt.utils.aio_rwlock import RWLock
from sglang.srt.utils.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
)
from sglang.srt.utils.network import get_zmq_socket
from sglang.srt.utils.request_logger import RequestLogger
from sglang.srt.utils.watchdog import Watchdog
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

# 全局把 asyncio 的事件循环策略替换为 uvloop，以获得更高的异步 I/O 性能。
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# 等待某个请求状态时的超时时间（秒），由环境变量配置，用于防止永久阻塞。
_REQUEST_STATE_WAIT_TIMEOUT = envs.SGLANG_REQUEST_STATE_WAIT_TIMEOUT.get()

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ReqState:
    """Store the state a request."""

    # 一个请求在 TokenizerManager 侧的「运行时状态」。每来一个请求就创建一个 ReqState，
    # 存入 rid_to_state；handle_loop 收到该请求的输出后更新此对象，并通过 event 唤醒等待协程。

    out_list: List[Dict[Any, Any]]  # 待返回给上层的输出片段列表（流式时不断追加）
    finished: bool  # 该请求是否已经结束（最后一个输出已到达）
    event: asyncio.Event  # 用于「有新输出 / 已结束」时唤醒 _wait_one_response 协程
    obj: Union[GenerateReqInput, EmbeddingReqInput]  # 原始请求对象，便于回溯参数

    # 性能指标相关
    time_stats: APIServerReqTimeStats  # 该请求各阶段耗时统计（用于 metrics）
    last_completion_tokens: int = 1  # 上一次回调时已生成的 token 数（计算增量用）
    ttft_observed: bool = False  # 是否已记录 TTFT（首 token 时延）

    # 流式输出相关：记录已发送到的偏移，避免重复发送。
    last_output_offset: int = 0  # 已输出 token 的偏移
    last_text_offset: int = 0  # 已输出文本的偏移

    # 增量状态更新相关：随着 token 不断到达，逐步累积文本、output_ids、各类 logprobs。
    # TODO(lianmin): do not initialize some lists if not needed.
    text: str = ""
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

    # 反 token 化（detokenize）后的 logprobs（已转成可读形式）。
    input_token_logprobs: List[Any] = dataclasses.field(default_factory=list)
    output_token_logprobs: List[Any] = dataclasses.field(default_factory=list)
    input_top_logprobs: List[Any] = dataclasses.field(default_factory=list)
    output_top_logprobs: List[Any] = dataclasses.field(default_factory=list)
    input_token_ids_logprobs: List[Any] = dataclasses.field(default_factory=list)
    output_token_ids_logprobs: List[Any] = dataclasses.field(default_factory=list)


class InputFormat(Enum):
    """Input format types for tokenization handling."""

    # 输入文本的三种格式，用于决定 tokenize 的处理方式。
    SINGLE_STRING = 1  # 单条文本，如 "Hello world"
    BATCH_STRINGS = 2  # 批量文本，如 ["Hello", "World"]
    CROSS_ENCODER_PAIRS = 3  # 交叉编码器句对，如 [["query", "document"]]（用于相似度/排序）


class TokenizerManager(TokenizerCommunicatorMixin, TokenizerManagerMultiItemMixin):
    """TokenizerManager is a process that tokenizes the text."""

    # 通过多重继承组合两个 Mixin：
    # - TokenizerCommunicatorMixin：封装与 Scheduler 的各类控制面通信（权重更新、profile 等）；
    # - TokenizerManagerMultiItemMixin：封装 multi-item 评分/打分等扩展能力。

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        # 构造函数把初始化拆分成一系列 init_* 子方法，逻辑清晰、便于子类覆盖。

        # 解析基础参数
        self.server_args = server_args
        self.enable_metrics = server_args.enable_metrics  # 是否启用 Prometheus 指标
        self.preferred_sampling_params = server_args.preferred_sampling_params  # 默认采样参数
        self.crash_dump_folder = server_args.crash_dump_folder  # 崩溃时转储请求的目录
        # 把 server_args 设为全局，供 tokenizer 相关模块读取。
        set_global_server_args_for_tokenizer(server_args)

        # 初始化模型配置（上下文长度、是否生成模型、image token id 等）
        self.init_model_config()

        # 初始化分词器与多模态处理器
        self.init_tokenizer_and_processor()

        # 初始化进程间通信（ZMQ 套接字）
        self.init_ipc_channels(port_args)

        # 初始化运行时状态（请求状态表、健康状态、会话表等）
        self.init_running_status()

        # 初始化请求日志与转储
        self.init_request_logging_and_dumping()

        # 初始化权重更新相关状态（读写锁、暂停条件变量等）
        self.init_weight_update()

        # 初始化 LoRA 适配器注册表与缓存
        self.init_lora()

        # 初始化 PD（Prefill-Decode）分离与编码器分离
        self.init_disaggregation()

        # 子进程存活看门狗 —— 由 Engine 或 http_server 在构造之后注入。
        self._subprocess_watchdog = None

        # 初始化指标采集器与看门狗
        self.init_metric_collector_watchdog()

        # 初始化请求分发器（按返回消息类型路由到不同处理函数）
        self.init_request_dispatcher()

    def init_model_config(self):
        # 读取并构造模型配置对象。
        server_args = self.server_args
        # 允许子类通过 model_config_class 注入自定义的 ModelConfig 实现。
        model_config_class = getattr(self, "model_config_class", ModelConfig)

        # 读取模型相关参数
        self.model_path = server_args.model_path  # 模型权重路径
        self.served_model_name = server_args.served_model_name  # 对外暴露的模型名
        self.model_config = model_config_class.from_server_args(server_args)
        self.is_generation = self.model_config.is_generation  # 是否为生成式模型（否则为 embedding 等）
        self.context_len = self.model_config.context_len  # 模型上下文长度上限
        self.image_token_id = self.model_config.image_token_id  # 图像占位 token 的 id
        self.max_req_input_len = None  # 单请求最大输入长度，稍后在 engine.py 中设置
        self.enable_priority_scheduling = server_args.enable_priority_scheduling  # 是否启用优先级调度
        self.default_priority_value = server_args.default_priority_value  # 默认优先级数值
        # 解析投机解码算法（如 EAGLE）。
        speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        if speculative_algorithm.is_eagle():
            # EAGLE 实现中，draft（草稿）token 复用输出 token 槽位，
            # 因此需要为草稿 token 预留空间。预留量取两种口径的较大值。
            self.num_reserved_tokens = max(
                server_args.speculative_eagle_topk * server_args.speculative_num_steps,
                server_args.speculative_num_draft_tokens,
            )
        else:
            self.num_reserved_tokens = 0  # 非投机解码无需预留
        self.validate_total_tokens = True  # 是否校验 输入+输出 总 token 数不超限

    def init_tokenizer_and_processor(self):
        # 初始化分词器（tokenizer）与多模态处理器（processor）。
        server_args = self.server_args

        # 初始化分词器和处理器
        if self.model_config.is_multimodal:
            # 多模态模型：先动态导入内置的多模态处理器实现。
            import_processors("sglang.srt.multimodal.processors")
            # 若配置了外部多模态处理器包，则覆盖导入。
            if mm_process_pkg := envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE.get():
                import_processors(mm_process_pkg, overwrite=True)
            _processor = _get_processor_wrapper(server_args)
            # 决定多模态张量在进程间的传输方式（共享内存 / ZMQ 等）。
            transport_mode = _determine_tensor_transport_mode(self.server_args)

            # 为了并行化图像预处理，这里创建带 executor 的多模态处理器。
            # 即使 skip_tokenizer_init=True 也要创建 mm_processor，以保证仍能对图像进行编码。
            self.mm_processor = get_mm_processor(
                self.model_config.hf_config,
                server_args,
                _processor,
                transport_mode,
                model_config=self.model_config,
            )

            if server_args.skip_tokenizer_init:
                # 跳过分词器初始化（调用方直接提供 input_ids）。
                self.tokenizer = self.processor = None
            else:
                self.processor = _processor
                # 从多模态 processor 中取出底层文本分词器。
                self.tokenizer = get_tokenizer_from_processor(self.processor)
                # 关闭 HuggingFace tokenizer 的并行，避免与上层并发冲突告警。
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                self._initialize_multi_item_delimiter_text()
        else:
            # 纯文本模型：没有多模态处理器。
            self.mm_processor = self.processor = None

            if server_args.skip_tokenizer_init:
                self.tokenizer = None
            else:
                # 按配置加载分词器。
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
                self._initialize_multi_item_delimiter_text()

        # 若启用「动态批处理分词器」则初始化（多模态与纯文本通用）。
        # 它能把短时间内的多个单条 tokenize 请求合批，提高吞吐。
        if (
            server_args.enable_dynamic_batch_tokenizer
            and not server_args.skip_tokenizer_init
        ):
            self.async_dynamic_batch_tokenizer = AsyncDynamicbatchTokenizer(
                self.tokenizer,
                max_batch_size=server_args.dynamic_batch_tokenizer_batch_size,
                batch_wait_timeout_s=server_args.dynamic_batch_tokenizer_batch_timeout,
            )
        else:
            self.async_dynamic_batch_tokenizer = None

    def init_ipc_channels(self, port_args: PortArgs):
        # 初始化与其他进程通信的 ZMQ 套接字。
        context = zmq.asyncio.Context(2)  # 创建 ZMQ 上下文，2 个 I/O 线程
        # PULL 套接字：从 DetokenizerManager 接收推理输出（反 token 化后的结果）。
        self.recv_from_detokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.tokenizer_ipc_name, True
        )
        if self.server_args.tokenizer_worker_num == 1:
            # 单 tokenizer worker：直接 PUSH 给 Scheduler 输入端口。
            self.send_to_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.scheduler_input_ipc_name, True
            )
        else:
            from sglang.srt.managers.multi_tokenizer_mixin import SenderWrapper

            # 多 tokenizer worker 模式：发往 tokenizer_worker_ipc_name。
            send_to_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_worker_ipc_name, False
            )

            # 用 SenderWrapper 包装，确保每个请求都携带自身的 tokenizer_ipc_name，
            # 以便 Scheduler 把响应正确路由回发起该请求的那个 worker。
            self.send_to_scheduler = SenderWrapper(port_args, send_to_scheduler)

    def init_running_status(self):
        # 请求运行时状态：rid -> ReqState 的映射，是整个收发流程的核心数据结构。
        self.rid_to_state: Dict[str, ReqState] = {}
        self.event_loop = None  # 后续懒加载的事件循环引用
        self.asyncio_tasks = set()  # 持有后台任务引用，防止被 GC 回收

        # 健康检查 / 生命周期状态
        self.server_status = ServerStatus.Starting  # 服务器当前状态
        self.gracefully_exit = False  # 是否处于优雅退出流程中
        self.last_receive_tstamp = real_time()  # 最近一次收到输出的时间戳（看门狗用）

        # 会话（session）相关：session_id -> asyncio event
        self.session_futures = {}

    def init_request_logging_and_dumping(self):
        # TODO: Refactor and organize the log export code.
        # 请求日志记录器：按配置记录收到的请求（级别 / 格式 / 目标）。
        self.request_logger = RequestLogger(
            log_requests=self.server_args.log_requests,
            log_requests_level=self.server_args.log_requests_level,
            log_requests_format=self.server_args.log_requests_format,
            log_requests_target=self.server_args.log_requests_target,
        )

        # 请求转储（dumping）：用于离线分析。
        self.dump_requests_folder = ""  # 默认空字符串表示不转储
        self.dump_requests_threshold = 1000  # 累计多少条后落盘
        self.dump_request_list: List[Tuple] = []  # 正常转储缓冲
        self.crash_dump_request_list: deque[Tuple] = deque()  # 崩溃转储缓冲（环形队列）
        self.crash_dump_performed = False  # 确保崩溃转储只执行一次的标志
        self.straggler_request_list: List[Tuple] = []  # 慢请求（拖尾）列表

        # 用合适的 skip 字段名初始化性能指标导出器（这些字段在导出时被忽略）。
        _, obj_skip_names, out_skip_names = self.request_logger.metadata
        self.request_metrics_exporter_manager = RequestMetricsExporterManager(
            self.server_args, obj_skip_names, out_skip_names
        )

    def init_weight_update(self):
        # 初始权重加载状态
        self.initial_weights_loaded = True
        if self.server_args.checkpoint_engine_wait_weights_before_ready:
            # 若配置为「就绪前等待权重」，则初始标记为未加载，待权重同步后再置 True。
            self.initial_weights_loaded = False

        # 权重更新
        # 用读写锁协调「权重更新」与「推理」：推理走 reader_lock（可并发），
        # 权重更新走 writer_lock（独占），从而保证更新期间没有正在进行的推理。
        self.model_update_lock = RWLock()
        # 权重同步完成后可 await 的结果对象。
        self.model_update_result: Optional[Awaitable[UpdateWeightFromDiskReqOutput]] = (
            None
        )
        self.is_pause = False  # 是否处于暂停接收新请求的状态
        self.is_pause_cond = asyncio.Condition()  # 暂停/恢复的条件变量

    def init_lora(self):
        # LoRA
        # 用 server_args 中初始的 LoRA 适配器路径初始化 LoRARegistry。
        # 该注册表会随运行时加载/卸载适配器而动态更新，是「可用适配器」的唯一真相来源，
        # 负责把用户友好的 LoRA 名称映射为内部使用的唯一 LoRA ID。
        self.lora_registry = LoRARegistry(self.server_args.lora_paths)
        # 串行化 LoRA 更新操作的锁。
        # 注意：与 model_update_lock 不同，此锁不阻塞推理，允许 LoRA 更新与推理重叠进行。
        self.lora_update_lock = asyncio.Lock()
        # 缓存：把曾经加载过的 LoRA 适配器名称映射到其最新的 LoRARef 对象，
        # 以便推理时按需动态加载。
        self.lora_ref_cache: Dict[str, LoRARef] = {}
        if self.server_args.lora_paths is not None:
            for lora_ref in self.server_args.lora_paths:
                self.lora_ref_cache[lora_ref.lora_name] = lora_ref

    def init_disaggregation(self):
        # PD（Prefill-Decode）分离：解析分离模式（null / prefill / decode）。
        self.disaggregation_mode = DisaggregationMode(
            self.server_args.disaggregation_mode
        )
        # 启动用于 PD 分离的 bootstrap 服务（建立 prefill/decode 节点间连接）。
        self.bootstrap_server = start_disagg_service(self.server_args)

        # 编码器分离：language_only 模式下，多模态编码在独立进程完成，
        # 这里创建接收器以拿到编码后的多模态特征。
        if self.server_args.language_only:
            self.mm_receiver = create_mm_receiver(
                self.server_args,
                dtype=self.model_config.dtype,
            )

    def init_metric_collector_watchdog(self):
        # 指标采集
        if self.enable_metrics:
            # 所有指标的公共标签（label），至少带模型名。
            labels = {
                "model_name": self.server_args.served_model_name,
                # TODO: Add lora name/path in the future,
            }
            if self.enable_priority_scheduling:
                labels["priority"] = ""  # 优先级调度时增加 priority 维度
            # 追加用户允许的自定义标签（值留空，运行时填充）。
            if self.server_args.tokenizer_metrics_allowed_custom_labels:
                for label in self.server_args.tokenizer_metrics_allowed_custom_labels:
                    labels[label] = ""
            # 追加固定的额外标签键值。
            if self.server_args.extra_metric_labels:
                labels.update(self.server_args.extra_metric_labels)
            # 构造指标采集器：TTFT、端到端时延、token 间时延等直方图的桶配置。
            self.metrics_collector = TokenizerMetricsCollector(
                server_args=self.server_args,
                labels=labels,
                bucket_time_to_first_token=self.server_args.bucket_time_to_first_token,
                bucket_e2e_request_latency=self.server_args.bucket_e2e_request_latency,
                bucket_inter_token_latency=self.server_args.bucket_inter_token_latency,
                collect_tokens_histogram=self.server_args.collect_tokens_histogram,
            )

            # 启动 CPU 监控线程，采集本进程 CPU 使用情况。
            start_cpu_monitor_thread("tokenizer")

        # 若配置了 GC 警告阈值，则在单次 GC 超时时打印告警。
        if self.server_args.gc_warning_threshold_secs > 0.0:
            configure_gc_warning(self.server_args.gc_warning_threshold_secs)
        # 软看门狗：检测 TokenizerManager 是否「卡住」（长时间无进展），soft=True 表示仅告警不杀进程。
        self.soft_watchdog = Watchdog.create(
            debug_name="TokenizerManager",
            watchdog_timeout=self.server_args.soft_watchdog_timeout,
            soft=True,
            test_stuck_time=envs.SGLANG_TEST_STUCK_TOKENIZER.get(),
        )

    def init_request_dispatcher(self):
        # 构造「按消息类型分发」的分发器：从 Detokenizer 收到的不同类型消息路由到对应处理函数。
        self._result_dispatcher = TypeBasedDispatcher(
            [
                (
                    # 三类批量输出（字符串/embedding/token id）统一交给 _handle_batch_output。
                    (
                        BatchStrOutput,
                        BatchEmbeddingOutput,
                        BatchTokenIDOutput,
                    ),
                    self._handle_batch_output,
                ),
                (AbortReq, self._handle_abort_req),  # 请求中止
                (OpenSessionReqOutput, self._handle_open_session_req_output),  # 打开会话结果
                (
                    UpdateWeightFromDiskReqOutput,  # 从磁盘更新权重的结果
                    self._handle_update_weights_from_disk_req_output,
                ),
                (FreezeGCReq, lambda x: None),  # 冻结 GC 的回执，无需处理
                # 当 scheduler 跳过 detokenizer 直接把健康检查结果转发回来时，忽略它。
                (HealthCheckOutput, lambda x: None),
                (ActiveRanksOutput, self.update_active_ranks),  # 更新活跃 rank 信息
            ]
        )
        # 初始化各类控制面通信器（来自 TokenizerCommunicatorMixin）。
        self.init_communicators(self.server_args)

        # 可被子类覆盖的类引用：采样参数类与信号处理类。
        self.sampling_params_class = SamplingParams
        self.signal_handler_class = SignalHandler

    async def generate_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        # 【入口】对外暴露的核心异步生成器：接收一个请求对象，逐步 yield 推理结果。
        # 上层（HTTP server / Engine）对其 async for 迭代即可拿到流式或最终结果。

        # 确保后台事件循环 handle_loop 已经启动（懒启动）。
        self.auto_create_handle_loop()

        # 归一化请求：统一单条/批量的参数表示，填充默认值、分配 rid 等。
        obj.normalize_batch_and_arguments()
        self._set_default_priority(obj)  # 设置默认优先级
        self._validate_rid_not_in_flight(obj)  # 校验 rid 没有在途（防止重复）

        # 若指定了 routed_dp_rank（强制路由到某个 data-parallel rank），做合法性校验。
        if isinstance(obj, GenerateReqInput) and obj.routed_dp_rank is not None:
            dp_size = self.server_args.dp_size
            if dp_size <= 1 and obj.routed_dp_rank == 0:
                # 没有开 DP（或只有 1 个），routed_dp_rank=0 无意义，仅告警。
                logger.warning(
                    f"routed_dp_rank={obj.routed_dp_rank} is ignored because dp_size={dp_size}"
                )
            elif obj.routed_dp_rank < 0 or obj.routed_dp_rank >= dp_size:
                # 越界则报错。
                raise ValueError(
                    f"routed_dp_rank={obj.routed_dp_rank} out of range [0, {dp_size})"
                )

        self._req_stats_init(obj, request)  # 初始化该请求的耗时统计
        if self.server_args.language_only:
            # 编码器分离场景：先处理多模态编码请求（分发到编码器进程）。
            self._handle_epd_disaggregation_encode_request(obj)
        if self.server_args.tokenizer_worker_num > 1:
            # 多 tokenizer worker：给请求附加本 worker 的路由信息。
            self._attach_multi_http_worker_info(obj)

        # 记录收到的请求（按配置可能脱敏/截断）。
        self.request_logger.log_received_request(obj, self.tokenizer, request)

        # 若服务处于暂停状态，则等待恢复后再继续。
        async with self.is_pause_cond:
            await self.is_pause_cond.wait_for(lambda: not self.is_pause)

        # 获取 model_update_lock 的「读锁」：推理可并发，但与权重更新（写锁）互斥。
        async with self.model_update_lock.reader_lock:
            # 校验并解析请求里引用的 LoRA 适配器（把名称解析为内部 id）。
            await self._validate_and_resolve_lora(obj)

            # 分词并发送给 scheduler
            if obj.is_single:
                # 单条请求：tokenize -> 取 state -> 发送 -> 等待并逐条 yield 响应。
                tokenized_obj = await self._tokenize_one_request(obj)
                state = self.rid_to_state[obj.rid]
                self._send_one_request(tokenized_obj)
                async for response in self._wait_one_response(obj, state, request):
                    yield response
            else:
                # 批量请求：交给专门的批处理逻辑。
                async for response in self._handle_batch_request(obj, request):
                    yield response

    def _detect_input_format(
        self, texts: Union[str, List[str]], is_cross_encoder: bool
    ) -> InputFormat:
        """Detect the format of input texts for proper tokenization handling.

        Returns:
            - InputFormat.SINGLE_STRING: Regular single text like "Hello world"
            - InputFormat.BATCH_STRINGS: Regular batch like ["Hello", "World"]
            - InputFormat.CROSS_ENCODER_PAIRS: Cross-encoder pairs like [["query", "document"]]
        """
        # 检测输入文本属于哪种格式，决定后续如何 tokenize。
        if isinstance(texts, str):
            return InputFormat.SINGLE_STRING  # 单字符串

        if (
            is_cross_encoder
            and len(texts) > 0
            and isinstance(texts[0], list)
            and len(texts[0]) == 2
        ):
            # 交叉编码器：元素是长度为 2 的列表（[query, doc]）。
            return InputFormat.CROSS_ENCODER_PAIRS

        return InputFormat.BATCH_STRINGS  # 其余视为普通批量字符串

    def _prepare_tokenizer_input(
        self, texts: Union[str, List[str]], input_format: InputFormat
    ) -> Union[List[str], List[List[str]]]:
        """Prepare input for the tokenizer based on detected format."""
        # 根据检测到的格式，把输入整理成 tokenizer 期望的统一形态。
        if input_format == InputFormat.SINGLE_STRING:
            return [texts]  # 单字符串包成列表，便于按 batch 处理
        elif input_format == InputFormat.CROSS_ENCODER_PAIRS:
            return texts  # 已是正确格式：[["query", "doc"]]
        else:  # BATCH_STRINGS
            return texts  # 已是正确格式：["text1", "text2"]

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
        """Extract results from tokenizer output based on input format."""
        # 根据输入格式，从 tokenizer 的批量输出中抽取出与输入对应的结果形态。

        # 单输入（单字符串或单条交叉编码器句对）：取第一个元素，去掉外层 batch 维度。
        if (
            input_format in [InputFormat.SINGLE_STRING, InputFormat.CROSS_ENCODER_PAIRS]
            and original_batch_size == 1
        ):
            single_input_ids = input_ids[0] if input_ids else []
            single_token_type_ids = token_type_ids[0] if token_type_ids else None
            return single_input_ids, single_token_type_ids

        # 真正的批量输入：原样返回（保留 batch 维度）。
        return input_ids, token_type_ids

    async def _tokenize_texts(
        self, texts: Union[str, List[str]], is_cross_encoder: bool = False
    ) -> Union[
        Tuple[List[int], Optional[List[int]]],
        Tuple[List[List[int]], Optional[List[List[int]]]],
    ]:
        """
        Tokenize text(s) using the appropriate tokenizer strategy.

        This method handles multiple input formats and chooses between async dynamic
        batch tokenizer (for single texts only) and regular tokenizer.

        Args:
            texts: Text input in various formats:

                   Regular cases:
                   - Single string: "How are you?"
                   - Batch of strings: ["Hello", "World", "How are you?"]

                   Cross-encoder cases (sentence pairs for similarity/ranking):
                   - Single pair: [["query text", "document text"]]
                   - Multiple pairs: [["q1", "d1"], ["q2", "d2"], ["q3", "d3"]]

            is_cross_encoder: Whether to return token_type_ids for cross-encoder models.
                             Enables proper handling of sentence pairs with segment IDs.

        Returns:
            Single input cases:
                Tuple[List[int], Optional[List[int]]]: (input_ids, token_type_ids)
                Example: ([101, 2129, 102], [0, 0, 0]) for single text
                Example: ([101, 2129, 102, 4068, 102], [0, 0, 0, 1, 1]) for cross-encoder pair

            Batch input cases:
                Tuple[List[List[int]], Optional[List[List[int]]]]: (batch_input_ids, batch_token_type_ids)
                Example: ([[101, 2129, 102], [101, 4068, 102]], None) for regular batch

            Note: token_type_ids is None unless is_cross_encoder=True.
        """
        # 前置校验：文本非空且分词器已初始化。
        if not texts or self.tokenizer is None:
            raise ValueError("texts cannot be empty and tokenizer must be initialized")

        # 步骤 1：检测输入格式并整理成 tokenizer 输入。
        input_format = self._detect_input_format(texts, is_cross_encoder)
        tokenizer_input = self._prepare_tokenizer_input(texts, input_format)
        original_batch_size = len(texts) if not isinstance(texts, str) else 1

        # 步骤 2：准备 tokenizer 参数（交叉编码器需要 token_type_ids 区分句子段）。
        tokenizer_kwargs = (
            {"return_token_type_ids": is_cross_encoder} if is_cross_encoder else {}
        )

        # 步骤 3：选择分词策略 —— 仅单字符串且启用了动态批处理分词器时走异步合批路径。
        use_async_tokenizer = (
            self.async_dynamic_batch_tokenizer is not None
            and input_format == InputFormat.SINGLE_STRING
        )

        if use_async_tokenizer:
            logger.debug("Using async dynamic batch tokenizer for single text")
            # 异步合批分词：多个并发的单条请求会在底层被合并成一批，提高吞吐。
            result = await self.async_dynamic_batch_tokenizer.encode(
                tokenizer_input[0], **tokenizer_kwargs
            )
            # 转成 batch 格式以保持后续处理一致。
            input_ids = [result["input_ids"]]
            token_type_ids = (
                [result["token_type_ids"]]
                if is_cross_encoder and result.get("token_type_ids")
                else None
            )
        else:
            logger.debug(f"Using regular tokenizer for {len(tokenizer_input)} inputs")
            # 普通路径：直接调用 HuggingFace tokenizer 批量编码。
            encoded = self.tokenizer(tokenizer_input, **tokenizer_kwargs)
            input_ids = encoded["input_ids"]
            token_type_ids = encoded.get("token_type_ids") if is_cross_encoder else None

        # 步骤 4：按输入格式抽取最终结果（决定是否去掉 batch 维度）。
        return self._extract_tokenizer_results(
            input_ids, token_type_ids, input_format, original_batch_size
        )

    async def _tokenize_one_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
    ):
        """Tokenize one request."""
        # 对单条请求做分词与多模态预处理，产出可发送给 scheduler 的 Tokenized*ReqInput。

        input_embeds = None  # 直接传入的输入嵌入（绕过 embedding 层）
        input_text = obj.text
        token_type_ids = None
        # 是否为交叉编码器 embedding 请求。
        is_cross_encoder_request = (
            isinstance(obj, EmbeddingReqInput) and obj.is_cross_encoder_request
        )
        if obj.input_embeds is not None:
            # 情况 A：调用方直接提供了 input_embeds。
            # 此时必须关闭 radix cache，否则前缀缓存会因无法对 embeds 做匹配而出错。
            if not self.server_args.disable_radix_cache:
                raise ValueError(
                    "input_embeds is provided while disable_radix_cache is False. "
                    "Please add `--disable-radix-cache` when you launch the server "
                    "if you want to use input_embeds as inputs."
                )
            input_embeds = obj.input_embeds
            input_ids = obj.input_ids
        elif obj.input_ids is not None:
            # 情况 B：调用方直接提供了 input_ids，无需分词。
            input_ids = obj.input_ids
        else:
            # 情况 C：需要对文本做分词。
            if self.tokenizer is None:
                # skip_tokenizer_init=True 时没有分词器，无法接受文本输入。
                raise ValueError(
                    "The engine initialized with skip_tokenizer_init=True cannot "
                    "accept text prompts. Please provide input_ids or re-initialize "
                    "the engine with skip_tokenizer_init=False."
                )

            # 纯音频请求（如 Whisper）文本可能为空，input_ids 稍后由多模态处理器提供。
            if not input_text and self.mm_processor and obj.contains_mm_input():
                # 先用空占位，后续多模态处理会覆盖。
                input_ids = []
            else:
                # 正常文本分词。
                input_ids, token_type_ids = await self._tokenize_texts(
                    input_text, is_cross_encoder_request
                )

        if self.mm_processor and obj.contains_mm_input():
            if obj.image_data is not None and not isinstance(obj.image_data, list):
                obj.image_data = [obj.image_data]
            if obj.video_data is not None and not isinstance(obj.video_data, list):
                obj.video_data = [obj.video_data]
            if obj.audio_data is not None and not isinstance(obj.audio_data, list):
                obj.audio_data = [obj.audio_data]
            self._validate_mm_limits(obj)

            mm_inputs = None

            if (
                not self.server_args.language_only
                or self.server_args.encoder_transfer_backend
                in ["zmq_to_tokenizer", "mooncake"]
            ):
                if self.server_args.language_only:
                    mm_inputs = await self.mm_receiver.recv_mm_data(
                        request_obj=obj,
                        mm_processor=self.mm_processor,
                        prompt=(input_text or input_ids),
                        need_wait_for_mm_inputs=obj.need_wait_for_mm_inputs,
                    )
                if mm_inputs is None:
                    mm_inputs: Dict = await self.mm_processor.process_mm_data_async(
                        image_data=obj.image_data,
                        audio_data=obj.audio_data,
                        input_text=(input_text or input_ids),
                        request_obj=obj,
                        max_req_input_len=self.max_req_input_len,
                    )
            elif (
                self.server_args.language_only
                and self.server_args.encoder_transfer_backend == "zmq_to_scheduler"
                and not obj.need_wait_for_mm_inputs
            ):
                # In language_only mode with zmq_to_scheduler, if we didn't dispatch
                # to encoder (e.g., only one image), process locally like non-language_only mode
                mm_inputs: Dict = await self.mm_processor.process_mm_data_async(
                    image_data=obj.image_data,
                    audio_data=obj.audio_data,
                    input_text=(input_text or input_ids),
                    request_obj=obj,
                    max_req_input_len=self.max_req_input_len,
                )

            # 多模态处理器可能返回（含图像占位符的）新 input_ids，覆盖之前的占位。
            if mm_inputs and "input_ids" in mm_inputs:
                input_ids = mm_inputs["input_ids"]
            if mm_inputs and "token_type_ids" in mm_inputs:
                token_type_ids = mm_inputs.pop("token_type_ids")
                if not isinstance(token_type_ids, list):
                    # 张量转为 Python list，便于跨进程序列化。
                    token_type_ids = token_type_ids.flatten().tolist()
            # 若启用「预计算多模态哈希」，则提前为各多模态 item 设置 pad 值（用于缓存键）。
            if (
                envs.SGLANG_MM_PRECOMPUTE_HASH.get()
                and mm_inputs
                and "mm_items" in mm_inputs
            ):
                for item in mm_inputs["mm_items"]:
                    if isinstance(item, MultimodalDataItem):
                        item.set_pad_value()
        else:
            mm_inputs = None  # 无多模态输入

        # 校验单请求长度等约束，然后封装成可发送的 Tokenized*ReqInput 对象。
        self._validate_one_request(obj, input_ids)
        return self._create_tokenized_object(
            obj, input_text, input_ids, input_embeds, mm_inputs, token_type_ids
        )

    def _validate_rid_not_in_flight(
        self, obj: Union[GenerateReqInput, EmbeddingReqInput]
    ) -> None:
        """Validate that request IDs are not already in flight."""
        # 校验请求 id 没有正在处理中（避免 rid 冲突导致状态错乱）。
        if obj.rid is None:
            return
        rids = obj.rid if isinstance(obj.rid, list) else [obj.rid]
        # 取本次 rid 与当前在途 rid 的交集。
        conflicts = set(rids) & self.rid_to_state.keys()
        if conflicts:
            raise ValueError(f"Duplicate request IDs detected: {list(conflicts)}")

    def _validate_one_request(
        self, obj: Union[GenerateReqInput, EmbeddingReqInput], input_ids: List[int]
    ) -> None:
        """Validates that the input token count and the requested token count doesn't exceed the model's context length."""
        # 校验「输入 token 数」与「输入+期望输出 token 数」不超过模型上下文长度。
        # FIXME: unify the length validation logic with the one in the scheduler.
        _max_req_len = self.context_len
        input_token_num = len(input_ids) if input_ids is not None else 0
        input_token_num += self.num_reserved_tokens  # 加上投机解码预留的 token

        # 校验输入长度
        if input_token_num >= self.context_len:
            if self.server_args.allow_auto_truncate:
                logger.warning(
                    f"The input ({input_token_num} tokens) is longer than the "
                    f"model's context length ({self.context_len} tokens). "
                    "Truncating the input."
                )
                # 允许自动截断：直接裁掉超出部分。
                del input_ids[_max_req_len:]
                input_token_num = len(input_ids)
            else:
                raise ValueError(
                    f"The input ({input_token_num} tokens) is longer than the "
                    f"model's context length ({self.context_len} tokens)."
                )

        # 校验总 token 数（输入 + 期望生成的 max_new_tokens）。
        max_new_tokens = obj.sampling_params.get("max_new_tokens")
        if (
            self.validate_total_tokens
            and max_new_tokens is not None
            and (max_new_tokens + input_token_num) >= _max_req_len
        ):
            if self.server_args.allow_auto_truncate:
                logger.warning(
                    f"Requested token count ({input_token_num} input + {max_new_tokens} new) "
                    f"exceeds the model's context length ({self.context_len} tokens). "
                    "Truncating max_new_tokens."
                )
                # 自动截断 max_new_tokens，使总数不超限（下限为 0）。
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

        # 校验 embedding 请求：若模型是生成式（非 embedding），则拒绝。
        if isinstance(obj, EmbeddingReqInput) and self.is_generation:
            raise ValueError(
                "This model does not appear to be an embedding model by default. "
                "Please add `--is-embedding` when launching the server or try another model."
            )

        # 校验 Matryoshka（套娃）embedding 的维度参数。
        if isinstance(obj, EmbeddingReqInput):
            self._validate_for_matryoshka_dim(obj)

        # 校验生成请求里的特殊开关是否被服务端启用。
        if isinstance(obj, GenerateReqInput):
            if (
                obj.return_hidden_states
                and not self.server_args.enable_return_hidden_states
            ):
                # 请求返回隐藏状态，但服务未开启该功能。
                raise ValueError(
                    "The server is not configured to return the hidden states. "
                    "Please set `--enable-return-hidden-states` to enable this feature."
                )
            if (
                obj.custom_logit_processor
                and not self.server_args.enable_custom_logit_processor
            ):
                # 请求自定义 logit 处理器，但服务未开启该功能。
                raise ValueError(
                    "The server is not configured to enable custom logit processor. "
                    "Please set `--enable-custom-logit-processor` to enable this feature."
                )

    def _validate_mm_limits(
        self, obj: Union[GenerateReqInput, EmbeddingReqInput]
    ) -> None:
        # 校验单请求内各模态（image/video/audio）的数据数量不超过配置上限。
        if not self.server_args.limit_mm_data_per_request:
            return

        for modality, limit in self.server_args.limit_mm_data_per_request.items():
            data = getattr(obj, f"{modality}_data", None)
            if data:
                count = len(data) if isinstance(data, list) else 1
                if count > limit:
                    raise ValueError(
                        f"{modality.capitalize()} count {count} exceeds limit {limit} per request."
                    )

    def _validate_for_matryoshka_dim(self, obj: EmbeddingReqInput) -> None:
        """Validate the request for Matryoshka dim if it has the field set."""
        # 校验 Matryoshka embedding 的输出维度合法性（仅当请求设置了 dimensions）。
        if obj.dimensions is None:
            return

        # 模型本身不支持 Matryoshka 表征，改维度会显著劣化效果。
        if not self.model_config.is_matryoshka:
            raise ValueError(
                f"Model '{self.model_config.model_path}' does not support matryoshka representation, "
                f"changing output dimensions will lead to poor results."
            )

        if obj.dimensions < 1:
            raise ValueError("Requested dimensions must be greater than 0")

        # 若模型声明了允许的维度集合，则请求维度必须在其中。
        if (
            self.model_config.matryoshka_dimensions
            and obj.dimensions not in self.model_config.matryoshka_dimensions
        ):
            raise ValueError(
                f"Model '{self.model_config.model_path}' only supports {self.model_config.matryoshka_dimensions} matryoshka dimensions, "
                f"using other output dimensions will lead to poor results."
            )

        # 维度不能超过模型隐藏层大小。
        if obj.dimensions > self.model_config.hidden_size:
            raise ValueError(
                f"Provided dimensions are greater than max embedding dimension: {self.model_config.hidden_size}"
            )

    def _validate_input_ids_in_vocab(
        self, input_ids: Union[List[int], List[List[int]]], vocab_size: int
    ) -> None:
        # 校验 input_ids 中的 token id 都在词表范围内（<vocab_size），防止越界。
        # 同时兼容单序列与批量序列两种输入。
        if isinstance(input_ids[0], list):
            # 批量序列
            for seq in input_ids:
                if any(id >= vocab_size for id in seq):
                    raise ValueError(
                        f"The input_ids {seq} contains values greater than the vocab size ({vocab_size})."
                    )
        else:
            # 单序列
            if any(id >= vocab_size for id in input_ids):
                raise ValueError(
                    f"The input_ids {input_ids} contains values greater than the vocab size ({vocab_size})."
                )

    def _create_tokenized_object(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        input_text: str,
        input_ids: List[int],
        input_embeds: Optional[Union[List[float], None]] = None,
        mm_inputs: Optional[Dict] = None,
        token_type_ids: Optional[List[int]] = None,
    ) -> Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput]:
        """Create a tokenized request object from common parameters."""
        # 从通用参数构造最终发送给 scheduler 的 Tokenized*ReqInput 对象。

        # 解析采样参数。
        # 注意：若存在 preferred_sampling_params（服务端默认），且请求未显式覆盖，则采用之。
        # 字典展开顺序保证 obj.sampling_params 中的同名键优先级更高。
        if self.preferred_sampling_params:
            sampling_kwargs = {**self.preferred_sampling_params, **obj.sampling_params}
        else:
            sampling_kwargs = obj.sampling_params
        sampling_params = self.sampling_params_class(**sampling_kwargs)
        sampling_params.normalize(self.tokenizer)  # 归一化（如把 stop 字符串转为 token）
        sampling_params.verify(self.model_config.vocab_size)  # 校验参数合法性

        # 构造返回对象（区分生成 / embedding 两类请求）。
        if isinstance(obj, GenerateReqInput):
            session_params = (
                SessionParams(**obj.session_params) if obj.session_params else None
            )

            tokenized_obj = TokenizedGenerateReqInput(
                input_text,
                input_ids,
                mm_inputs,
                sampling_params,
                obj.return_logprob,
                obj.logprob_start_len,
                obj.top_logprobs_num,
                obj.token_ids_logprob,
                obj.stream,
                rid=obj.rid,
                http_worker_ipc=obj.http_worker_ipc,
                bootstrap_host=obj.bootstrap_host,
                bootstrap_port=obj.bootstrap_port,
                bootstrap_room=obj.bootstrap_room,
                lora_id=obj.lora_id,
                input_embeds=input_embeds,
                session_params=session_params,
                custom_logit_processor=obj.custom_logit_processor,
                require_reasoning=obj.require_reasoning,
                return_hidden_states=obj.return_hidden_states,
                return_routed_experts=obj.return_routed_experts,
                routed_dp_rank=obj.routed_dp_rank,
                disagg_prefill_dp_rank=obj.disagg_prefill_dp_rank,
                priority=obj.priority,
                extra_key=obj.extra_key,
                routing_key=obj.routing_key,
                token_type_ids=token_type_ids,
                need_wait_for_mm_inputs=obj.need_wait_for_mm_inputs,
                num_items_assigned=obj.num_items_assigned,
            )
        elif isinstance(obj, EmbeddingReqInput):
            tokenized_obj = TokenizedEmbeddingReqInput(
                input_text,
                input_ids,
                mm_inputs,
                token_type_ids,
                sampling_params,
                rid=obj.rid,
                priority=obj.priority,
                dimensions=obj.dimensions,
                lora_id=obj.lora_id,
                http_worker_ipc=obj.http_worker_ipc,
            )

        # 把该请求的耗时统计对象挂到 tokenized_obj 上，并记录「分词完成时间」。
        tokenized_obj.time_stats = self.rid_to_state[obj.rid].time_stats
        self.rid_to_state[obj.rid].time_stats.set_tokenize_finish_time()

        return tokenized_obj

    async def _batch_tokenize_and_process(
        self, batch_size: int, obj: Union[GenerateReqInput, EmbeddingReqInput]
    ) -> List[Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput]]:
        """Handle batch tokenization for text inputs only."""
        # 对一批「纯文本」请求做合批分词（启用 enable_tokenizer_batch_encode 时使用）。
        logger.debug(f"Starting batch tokenization for {batch_size} text requests")

        # 若该批没有文本（都已是 input_ids），则无需分词，逐条构造返回对象即可。
        if not self._batch_has_text(batch_size, obj):
            return [await self._tokenize_one_request(obj[i]) for i in range(batch_size)]

        # 校验合批分词的前置约束（不支持多模态/预分词/embeds 混入）。
        self._validate_batch_tokenization_constraints(batch_size, obj)

        # 收集子请求与对应文本。
        requests = [obj[i] for i in range(batch_size)]
        texts = [req.text for req in requests]

        # 该批中是否存在交叉编码器请求。
        is_cross_encoder_request = any(
            isinstance(req, EmbeddingReqInput) and req.is_cross_encoder_request
            for req in requests
        )

        # 用统一方法一次性批量分词所有文本。
        input_ids_list, token_type_ids_list = await self._tokenize_texts(
            texts, is_cross_encoder_request
        )

        # 逐条校验并构造 tokenized 对象。
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
        """Validate constraints for batch tokenization processing."""
        # 合批分词只支持纯文本生成场景，这里逐条拦截不兼容的情况。
        for i in range(batch_size):
            if self.is_generation and obj[i].contains_mm_input():
                # 多模态输入不支持合批分词。
                raise ValueError(
                    "For multimodal input processing do not set `enable_tokenizer_batch_encode`."
                )
            if obj[i].input_ids is not None:
                # 已预分词的 input_ids 无需再合批分词。
                raise ValueError(
                    "Batch tokenization is not needed for pre-tokenized input_ids. Do not set `enable_tokenizer_batch_encode`."
                )
            if obj[i].input_embeds is not None:
                # 直接传入 input_embeds 时也无需合批分词。
                raise ValueError(
                    "Batch tokenization is not needed for input_embeds. Do not set `enable_tokenizer_batch_encode`."
                )

    def _batch_has_text(
        self, batch_size: int, obj: Union[GenerateReqInput, EmbeddingReqInput]
    ) -> bool:
        """Check if any request in the batch contains text input."""
        # 判断该批中是否存在「需要分词」的请求（含文本或多模态输入）。
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
        """
        # 是否对该批走「批量分词」路径。策略：
        # - 显式开启 enable_tokenizer_batch_encode 时走；
        # - 或者：未开 DP attention 且全批都无需分词（都是预分词/embeds），也合批发送；
        # - 注意：批量分词尚不支持 DP attention，目前会把请求全压到第一个 rank。
        return batch_size > 0 and (
            self.server_args.enable_tokenizer_batch_encode
            or (
                (not self.server_args.enable_dp_attention)
                and (not self._batch_has_text(batch_size, requests))
            )
        )

    def _send_one_request(
        self,
        tokenized_obj: Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput],
    ):
        # 把单个已分词请求通过 ZMQ 发送给 scheduler。
        tokenized_obj.time_stats.set_api_server_dispatch_time()  # 记录派发开始时间
        # 把多模态特征等大对象转放到共享内存，ZMQ 仅传引用，降低拷贝开销。
        tokenized_obj = wrap_shm_features(tokenized_obj)
        self.send_to_scheduler.send_pyobj(tokenized_obj)  # 序列化并发送
        tokenized_obj.time_stats.set_api_server_dispatch_finish_time()  # 记录派发完成时间

    def _send_batch_request(
        self,
        tokenized_objs: List[
            Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput]
        ],
    ):
        """Send a batch of tokenized requests as a single batched request to the scheduler."""
        # 把一批已分词请求打包成单个批量请求一次性发送给 scheduler。
        if isinstance(tokenized_objs[0], TokenizedGenerateReqInput):
            batch_req = BatchTokenizedGenerateReqInput(batch=tokenized_objs)
        else:
            batch_req = BatchTokenizedEmbeddingReqInput(batch=tokenized_objs)

        # 批量打点派发时间（对整批统一设置）。
        set_time_batch(tokenized_objs, "set_api_server_dispatch_time")
        self.send_to_scheduler.send_pyobj(batch_req)
        set_time_batch(tokenized_objs, "set_api_server_dispatch_finish_time")

    async def _wait_one_response(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        state: ReqState,
        request: Optional[fastapi.Request] = None,
    ):
        """Wait for the response of one request."""
        # 单个请求的「结果等待协程」：被 generate_request 当作异步生成器使用。
        # 它阻塞在 state.event 上，每当 handle_loop 收到该请求的新输出就被唤醒，
        # 取出 out_list 中的增量结果 yield 出去，直到 finished 为止。
        # 并非所有请求类型都有 `stream` 字段（如 EmbeddingReqInput），默认按非流式处理。
        is_stream = getattr(obj, "stream", False)
        while True:
            try:
                # 等待「有新输出」事件，带超时以便周期性检查客户端是否已断开。
                await asyncio.wait_for(
                    state.event.wait(), timeout=_REQUEST_STATE_WAIT_TIMEOUT
                )
            except asyncio.TimeoutError:
                # 超时后若发现非后台请求且客户端已断连，则中止该请求并抛异常终止整个调用栈。
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
            # With incremental streaming output, each chunk carries only a
            # delta, so every queued chunk must be yielded to avoid dropping
            # token ids. Without it, outputs are cumulative and only the
            # latest chunk contains the full result, so we can safely skip
            # intermediate ones.
            # 原子地取走所有待返回输出：
            # 增量流式下每个 chunk 只携带 delta，必须逐个 yield 否则会丢 token；
            # 非增量下输出是累积的，只有最后一个 chunk 是完整结果，可安全跳过中间块。
            incremental_stream = (
                is_stream and self.server_args.incremental_streaming_output
            )
            out_list = state.out_list
            state.out_list = []  # 取走后清空，等待下一批
            finished = state.finished
            state.event.clear()  # 重置事件，等待下次唤醒

            if incremental_stream and len(out_list) > 1:
                # 增量流式且积压了多块：把多个 delta 合并成一块，减少 yield 次数。
                if len(out_list) >= 20:
                    logger.warning(
                        "Streaming backlog: rid=%s, coalescing %d queued chunks into one. "
                        "This may inflate P99 ITL for affected requests.",
                        obj.rid,
                        len(out_list),
                    )
                # Coalesce all deltas into a single chunk. Both text and
                # output_ids are incremental, so we concatenate them; all
                # other fields (meta_info, etc.) are taken from the last chunk.
                # text 与 output_ids 都是增量的，需要拼接；其余字段（meta_info 等）取最后一块。
                out = dict(out_list[-1])
                if "output_ids" in out:
                    out["output_ids"] = [
                        id for chunk in out_list for id in chunk["output_ids"]
                    ]
                if "text" in out:
                    out["text"] = "".join(chunk["text"] for chunk in out_list)
            else:
                out = out_list[-1]

            if finished:
                # 请求已结束：记录响应发送时间、写日志与 metrics，并处理 abort/error 情形。
                # For non-streaming cases, response has not been sent yet (`response_sent_to_client_time` has not been set yet).
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
                    # Asynchronously write metrics for this request using the exporter manager.
                    asyncio.create_task(
                        self.request_metrics_exporter_manager.write_record(obj, out)
                    )

                # 检查是否为 scheduler 主动产生的 abort/error。
                if isinstance(out["meta_info"].get("finish_reason"), dict):
                    finish_reason = out["meta_info"]["finish_reason"]
                    # 400 BAD_REQUEST：非流式直接抛 ValueError，流式则 yield 后结束。
                    if (
                        finish_reason.get("type") == "abort"
                        and finish_reason.get("status_code") == HTTPStatus.BAD_REQUEST
                    ):
                        if not is_stream:
                            raise ValueError(finish_reason["message"])
                        else:
                            yield out
                            break

                    if finish_reason.get("type") == "abort" and finish_reason.get(
                        "status_code"
                    ) in (
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                    ):
                        # This is an abort request initiated by scheduler.
                        # Delete the key to prevent resending abort request to the scheduler and
                        # to ensure aborted request state is cleaned up.
                        if state.obj.rid in self.rid_to_state:
                            del self.rid_to_state[state.obj.rid]

                        # Mark ongoing LoRA request as finished.
                        if self.server_args.enable_lora and state.obj.lora_path:
                            await self.lora_registry.release(state.obj.lora_id)
                        if not is_stream:
                            raise fastapi.HTTPException(
                                status_code=finish_reason["status_code"],
                                detail=finish_reason["message"],
                            )
                        else:
                            yield out
                            break
                yield out
                break

            if is_stream:
                # 流式：每收到一块就在发送前记录响应时间并 yield 出去。
                if not state.time_stats.response_sent_to_client_time:
                    state.time_stats.set_response_sent_to_client_time()
                    out["meta_info"][
                        "response_sent_to_client_ts"
                    ] = state.time_stats.get_response_sent_to_client_realtime()
                yield out

            if not is_stream:
                # 非流式：未结束前若客户端已断连，则中止请求并抛异常终止任务。
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
        # 批量/并行采样请求的统一入口：把一个 batch（或一个请求的 n 个并行采样）
        # 拆成多个子请求 tokenize 后发给 scheduler，再用各自的 _wait_one_response
        # 生成器收集结果。非流式一次性 gather 返回，流式则边到边 yield。
        batch_size = obj.batch_size

        generators = []
        rids = []
        # parallel_sample_num == 1：普通批量，无并行采样，每个子请求独立处理。
        if getattr(obj, "parallel_sample_num", 1) == 1:
            # 满足条件时走「批量 tokenize 一次性下发」的快路径，减少逐条往返开销。
            if self._should_use_batch_tokenization(batch_size, obj):
                tokenized_objs = await self._batch_tokenize_and_process(batch_size, obj)
                self._send_batch_request(tokenized_objs)

                # Set up generators for each request in the batch
                # 为 batch 中每个子请求建立结果等待生成器，并登记其 rid。
                for i in range(batch_size):
                    tmp_obj = obj[i]
                    state = self.rid_to_state[tmp_obj.rid]
                    state.obj = tmp_obj
                    generators.append(self._wait_one_response(tmp_obj, state, request))
                    rids.append(tmp_obj.rid)
            else:
                # Sequential tokenization and processing
                # 慢路径：逐条 tokenize 并发送（可选 colocated batch gen 时用 input blocker 包裹，
                # 把整批请求作为一个整体送入，避免被其它请求穿插）。
                with (
                    input_blocker_guard_region(send_to_scheduler=self.send_to_scheduler)
                    if get_bool_env_var("SGLANG_ENABLE_COLOCATED_BATCH_GEN")
                    else nullcontext()
                ):
                    for i in range(batch_size):
                        tmp_obj = obj[i]
                        tokenized_obj = await self._tokenize_one_request(tmp_obj)
                        state = self.rid_to_state[tmp_obj.rid]
                        state.obj = tmp_obj
                        self._send_one_request(tokenized_obj)
                        generators.append(
                            self._wait_one_response(tmp_obj, state, request)
                        )
                        rids.append(tmp_obj.rid)
        else:
            # parallel_sample_num > 1：对每个请求做 n 路并行采样。
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
            # 先发一个 max_new_tokens=0 的「预热」请求，让 scheduler 把公共 prompt 前缀
            # 算进 radix cache，后续 n 路采样可直接命中前缀缓存、不必重复 prefill。
            for i in range(batch_size):
                tmp_obj = copy.copy(objs[i])
                tokenized_obj = copy.copy(tokenized_objs[i])
                tokenized_obj.rid = tmp_obj.regenerate_rid()
                tokenized_obj.sampling_params = copy.copy(tokenized_obj.sampling_params)
                tokenized_obj.sampling_params.max_new_tokens = 0  # 不生成新 token，只为缓存前缀
                tokenized_obj.stream = False
                self._req_stats_init(tmp_obj)
                state = self.rid_to_state[tmp_obj.rid]
                tokenized_obj.time_stats = state.time_stats
                self._send_one_request(tokenized_obj)
                await self._wait_one_response(tmp_obj, state, request).__anext__()  # 等预热请求返回，确保前缀已缓存

            # Expand requests, assign new rids for them, and send them
            # 真正展开：每个请求复制成 parallel_sample_num 份，各分配新 rid 并下发。
            for i in range(batch_size):
                for _ in range(obj.parallel_sample_num):
                    tmp_obj = copy.copy(objs[i])
                    tokenized_obj = copy.copy(tokenized_objs[i])
                    tokenized_obj.rid = tmp_obj.regenerate_rid()
                    self._req_stats_init(tmp_obj)
                    state = self.rid_to_state[tmp_obj.rid]
                    tokenized_obj.time_stats = state.time_stats
                    self._send_one_request(tokenized_obj)
                    generators.append(self._wait_one_response(tmp_obj, state, request))
                    rids.append(tmp_obj.rid)

                # 预热请求用的原始 rid 已无用，结算时间并从 state 表中清除。
                self.rid_to_state[objs[i].rid].time_stats.set_finished_time()
                del self.rid_to_state[objs[i].rid]

        # Wait for all requests
        is_stream = hasattr(obj, "stream") and obj.stream
        if not is_stream:
            # 非流式：等所有子请求各产出一个最终结果，聚合成列表一次性返回。
            outputs = await asyncio.gather(*(gen.__anext__() for gen in generators))
            yield outputs
        else:
            # 流式：用 rid 反查在 batch 中的下标，便于上层按原始顺序对齐结果。
            rid_to_index = {rid: i for i, rid in enumerate(rids)}
            # 每个生成器先取一块，任意一块就绪就 yield，再为该生成器排下一块，直到全部耗尽。
            task_map = {asyncio.create_task(gen.__anext__()): gen for gen in generators}
            while task_map:
                done, _ = await asyncio.wait(
                    task_map.keys(), return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    gen = task_map.pop(task)
                    try:
                        result = task.result()
                        result["index"] = rid_to_index[result["meta_info"]["id"]]  # 标记该 chunk 属于哪个子请求
                        yield result
                        new_task = asyncio.create_task(gen.__anext__())
                        task_map[new_task] = gen
                    except StopAsyncIteration:
                        pass  # 该生成器已结束，不再续排任务

    def abort_request(self, rid: str = "", abort_all: bool = False):
        # 向 scheduler 发送中止请求：指定单个 rid，或 abort_all=True 中止全部。
        # 非 abort_all 且本地不存在该 rid 时直接返回，避免发无效中止。
        if not abort_all and rid not in self.rid_to_state:
            return
        req = AbortReq(rid=rid, abort_all=abort_all)
        self.send_to_scheduler.send_pyobj(req)
        if self.enable_metrics:
            # TODO: also use custom_labels from the request
            self.metrics_collector.observe_one_aborted_request(
                self.metrics_collector.labels
            )

    async def pause_generation(self, obj: PauseGenerationReqInput):
        # 暂停生成：置 is_pause 标志，使新请求在 is_pause_cond 上挂起。
        async with self.is_pause_cond:
            self.is_pause = True
            if obj.mode != "abort":
                await self.send_to_scheduler.send_pyobj(obj)
            else:
                # we are using the model_update_lock to check if there is still on-going requests.
                # abort 模式：反复中止全部请求，直到 model_update_lock 不再被占用，
                # 即确认没有在途请求后才结束（轮询等待）。
                while True:
                    # TODO: maybe make it async instead of fire-and-forget
                    self.abort_request(abort_all=True)
                    is_locked = await self.model_update_lock.is_locked()
                    if not is_locked:
                        break
                    await asyncio.sleep(1.0)

    async def continue_generation(self, obj: ContinueGenerationReqInput):
        # 恢复生成：清除 is_pause 并通知 scheduler，唤醒所有挂起在 is_pause_cond 上的请求。
        async with self.is_pause_cond:
            self.is_pause = False
            await self.send_to_scheduler.send_pyobj(obj)
            self.is_pause_cond.notify_all()

    async def update_weights_from_disk(
        self,
        obj: UpdateWeightFromDiskReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        # 从磁盘热更新权重：保证 handle_loop 已启动，按需中止在途请求，
        # 在 model_update_lock 写锁保护下下发更新，等待 scheduler 完成后返回结果。
        self.auto_create_handle_loop()

        # default the load format to the server_args
        if obj.load_format is None:
            obj.load_format = self.server_args.load_format
        logger.info("Start update_weights. Load format=%s", obj.load_format)

        if obj.abort_all_requests:
            self.abort_request(abort_all=True)

        # Immediately update the weights if the engine is in paused state
        # 已暂停状态下不存在在途请求，可跳过写锁直接更新（用 nullcontext 占位）。
        async with self.is_pause_cond:
            is_paused = self.is_pause

        # 未暂停时取写锁，独占以阻止新请求进入，避免权重更新与推理并发。
        lock_context = (
            self.model_update_lock.writer_lock if not is_paused else nullcontext()
        )
        async with lock_context:
            success, message, num_paused_requests = (
                await self._wait_for_model_update_from_disk(obj)
            )

        # 更新成功且指定了 weight_version 时，同步记录新的权重版本号。
        if success and obj.weight_version is not None:
            self._update_weight_version_if_provided(obj.weight_version)
            message += f" Weight version updated to {obj.weight_version}."

        return success, message, num_paused_requests

    def _update_model_path_info(self, model_path: str, load_format: str):
        # 更新成功后同步本地记录的模型路径/加载格式，使 server_args 与实际权重一致。
        self.served_model_name = model_path
        self.server_args.model_path = model_path
        self.server_args.load_format = load_format
        self.model_path = model_path

    async def _wait_for_model_update_from_disk(
        self, obj: UpdateWeightFromDiskReqInput
    ) -> Tuple[bool, str]:
        # 把更新请求发给 scheduler，并通过 Future 异步等待其完成回执（结果由 handle_loop 填入）。
        self.send_to_scheduler.send_pyobj(obj)
        self.model_update_result = asyncio.Future()
        if self.server_args.dp_size == 1:
            # 单 DP：只有一个 worker，直接取其结果。
            result = await self.model_update_result
            if result.success:
                self._update_model_path_info(obj.model_path, obj.load_format)
            return result.success, result.message, result.num_paused_requests
        else:  # self.server_args.dp_size > 1
            # 多 DP：需收齐所有副本的回执，全部成功才算成功，消息与计数逐一聚合。
            self.model_update_tmp = []
            result = await self.model_update_result

            all_success = all([r.success for r in result])
            if all_success is True:
                self._update_model_path_info(obj.model_path, obj.load_format)
            all_message = [r.message for r in result]
            all_message = " | ".join(all_message)
            all_paused_requests = [r.num_paused_requests for r in result]
            return all_success, all_message, all_paused_requests

    def configure_logging(self, obj: ConfigureLoggingReq):
        # 运行期动态调整请求日志配置：开关、级别、格式及 dump/crash dump 目录与阈值。
        self.request_logger.configure(
            log_requests=obj.log_requests,
            log_requests_level=obj.log_requests_level,
            log_requests_format=obj.log_requests_format,
        )
        if obj.dump_requests_folder is not None:
            self.dump_requests_folder = obj.dump_requests_folder
        if obj.dump_requests_threshold is not None:
            self.dump_requests_threshold = obj.dump_requests_threshold
        if obj.crash_dump_folder is not None:
            self.crash_dump_folder = obj.crash_dump_folder
        logging.info(f"Config logging: {obj=}")

    async def freeze_gc(self):
        """Send a freeze_gc message to the scheduler first, then freeze locally."""
        # 先让 scheduler 冻结 GC，再冻结本进程：把已存活对象移入永久代，
        # 降低后续 GC 扫描开销（常用于权重加载完成后稳定运行阶段）。
        self.send_to_scheduler.send_pyobj(FreezeGCReq())
        freeze_gc("Tokenizer Manager")
        return None

    def create_abort_task(self, obj: GenerateReqInput):
        # Abort the request if the client is disconnected.
        # 返回一个 FastAPI 后台任务：响应结束后延迟 2s 触发中止，
        # 用于客户端断连场景下兜底清理 scheduler 上残留的请求。
        async def abort_request():
            await asyncio.sleep(2)
            if obj.is_single:
                self.abort_request(obj.rid)
            else:
                # 批量请求逐个中止对应的 rid。
                for rid in obj.rid:
                    self.abort_request(rid)

        background_tasks = BackgroundTasks()
        background_tasks.add_task(abort_request)
        return background_tasks

    def auto_create_handle_loop(self):
        # 懒启动 handle_loop：首次需要时在事件循环上拉起接收 scheduler 输出的后台任务，
        # 已启动则直接返回（幂等）。
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
        # 仅主线程能注册信号处理器（CPython 限制）：接管 SIGTERM/SIGQUIT 以便优雅退出。
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
        """The event loop that handles requests"""
        # 常驻异步主循环：是 TokenizerManager 接收推理结果的唯一入口。
        # 不断从 detokenizer 的 ZMQ socket 阻塞收取消息对象，按其类型交给
        # _result_dispatcher 分发到对应处理函数（如 _handle_batch_output）。
        while True:
            # 收 ZMQ 期间关闭 soft_watchdog，避免把正常的「等待新消息」误判为卡死。
            with self.soft_watchdog.disable():
                recv_obj = await self.recv_from_detokenizer.recv_pyobj()
            self._result_dispatcher(recv_obj)
            self.last_receive_tstamp = real_time()  # 记录最近一次收到消息的时间，供存活性检测使用
            self.soft_watchdog.feed()  # 喂狗：表明本轮处理正常完成

    def _handle_batch_output(
        self,
        recv_obj: Union[
            BatchStrOutput,
            BatchEmbeddingOutput,
            BatchTokenIDOutput,
        ],
    ):
        # 处理一批来自 scheduler/detokenizer 的输出（生成文本、token id 或 embedding）。
        # 角色：把 batch 内每个请求的结果按 rid 找回各自的 ReqState，累积增量输出、
        # 拼装 meta_info，并唤醒在 _wait_one_response 中阻塞的等待协程把结果回传上层。
        for i, rid in enumerate(recv_obj.rids):
            # recv_obj 里所有字段都是按 batch 下标 i 对齐的并行数组，先按 rid 取回对应 state。
            state = self.rid_to_state.get(rid, None)
            if state is None:
                # state 已被删除（如请求提前 abort 或已完成清理）：丢弃该条输出，避免误处理。
                logger.error(
                    f"Received output for {rid=} but the state was deleted in TokenizerManager."
                )
                continue

            # Build meta_info and return value
            # 拼装回传给上层的元信息，finish_reason 为 None 表示尚未结束（流式中间结果）。
            meta_info = {
                "id": rid,
                "finish_reason": recv_obj.finished_reasons[i],
                "prompt_tokens": recv_obj.prompt_tokens[i],
                "weight_version": self.server_args.weight_version,
                "total_retractions": recv_obj.retraction_counts[i],
            }

            # 开启 metrics 时，把 scheduler 侧的耗时统计（排队/预填/解码等）并入 meta_info。
            if self.enable_metrics:
                if recv_obj.time_stats is not None:
                    scheduler_time_stats = recv_obj.time_stats[i]
                    meta_info.update(scheduler_time_stats.convert_to_output_meta_info())

            # 若请求要求返回 logprob，则把 recv_obj 中的原始 logprob 转换成对外格式写入 meta_info。
            if getattr(state.obj, "return_logprob", False):
                self.convert_logprob_style(
                    meta_info,
                    state,
                    state.obj.top_logprobs_num,
                    state.obj.token_ids_logprob,
                    state.obj.return_text_in_logprobs
                    and not self.server_args.skip_tokenizer_init,
                    recv_obj,
                    i,
                )

            # embedding 请求没有生成 token，故仅对生成类输出补充 completion/cached token 计数。
            if not isinstance(recv_obj, BatchEmbeddingOutput):
                meta_info.update(
                    {
                        "completion_tokens": recv_obj.completion_tokens[i],
                        "cached_tokens": recv_obj.cached_tokens[i],
                    }
                )
                # Add detailed cache breakdown if available
                # 如有更细的 prefix cache 命中明细则一并带上。
                if (
                    hasattr(recv_obj, "cached_tokens_details")
                    and recv_obj.cached_tokens_details
                ):
                    meta_info["cached_tokens_details"] = recv_obj.cached_tokens_details[
                        i
                    ]

            # 以下为可选的附加信息，仅在对应特性开启、且 recv_obj 携带该字段时才写入 meta_info。
            if getattr(recv_obj, "output_hidden_states", None):
                meta_info["hidden_states"] = recv_obj.output_hidden_states[i]
            if getattr(recv_obj, "routed_experts", None):
                # MoE 路由专家张量需序列化为可 JSON 传输的字符串：先转 bytes 再 base64 编码。
                routed_experts_tensor = recv_obj.routed_experts[i]
                if routed_experts_tensor is not None:
                    meta_info["routed_experts"] = pybase64.b64encode(
                        routed_experts_tensor.numpy().tobytes()
                    ).decode("utf-8")
            if getattr(recv_obj, "customized_info", None):
                for k, v in recv_obj.customized_info.items():
                    meta_info[k] = v[i]
            if getattr(recv_obj, "dp_ranks", None):
                meta_info["dp_rank"] = recv_obj.dp_ranks[i]

            # 文本输出分支：detokenizer 已把 token 还原成字符串，先累积到 state.text。
            if isinstance(recv_obj, BatchStrOutput):
                state.text += recv_obj.output_strs[i]
                # Not all request types have `stream` (e.g., EmbeddingReqInput). Default to non-streaming.
                is_stream = getattr(state.obj, "stream", False)
                if self.server_args.incremental_streaming_output and is_stream:
                    # 增量流式：只回传上次 offset 之后新增的 token/文本，并推进 offset，避免重复传整段。
                    state.output_ids.extend(recv_obj.output_ids[i])
                    output_token_ids = state.output_ids[state.last_output_offset :]
                    state.last_output_offset = len(state.output_ids)
                    output_text = state.text[state.last_text_offset :]
                    state.last_text_offset = len(state.text)
                else:
                    # 累积模式：每次都回传到目前为止的完整 token 序列与文本（copy 防止外部改动内部状态）。
                    state.output_ids.extend(recv_obj.output_ids[i])
                    output_token_ids = state.output_ids.copy()
                    output_text = state.text

                out_dict = {
                    "text": output_text,
                    "output_ids": output_token_ids,
                    "meta_info": meta_info,
                }

            # 纯 token id 输出分支（跳过 detokenize，直接回传 id）：增量/累积逻辑同上，只是没有文本字段。
            elif isinstance(recv_obj, BatchTokenIDOutput):
                is_stream = getattr(state.obj, "stream", False)
                if self.server_args.incremental_streaming_output and is_stream:
                    state.output_ids.extend(recv_obj.output_ids[i])
                    output_token_ids = state.output_ids[state.last_output_offset :]
                    state.last_output_offset = len(state.output_ids)
                else:
                    state.output_ids.extend(recv_obj.output_ids[i])
                    output_token_ids = state.output_ids.copy()

                out_dict = {
                    "output_ids": output_token_ids,
                    "meta_info": meta_info,
                }
            else:
                # embedding 分支：无增量概念，直接回传整段向量。
                assert isinstance(recv_obj, BatchEmbeddingOutput)
                out_dict = {
                    "embedding": recv_obj.embeddings[i],
                    "meta_info": meta_info,
                }

            # finish_reason 非空即代表该请求已生成结束，置位 finished 供后续收尾与唤醒判断。
            state.finished = recv_obj.finished_reasons[i] is not None

            # Set first_token_time on the first output batch.
            # This is the single write point for first_token_time.
            # 仅在收到第一批输出时记录首 token 时间（TTFT），全流程唯一写入点。
            if state.time_stats.first_token_time == 0.0:
                state.time_stats.set_first_token_time()

            # 结束分支：补齐 trace 属性、结束时间与端到端时延，并采集相关 metrics。
            if state.finished:
                state.time_stats.trace_ctx.trace_set_root_attrs(
                    self.convert_to_span_attrs(state, recv_obj, i)
                )
                state.time_stats.set_finished_time()
                meta_info["e2e_latency"] = state.time_stats.get_e2e_latency()

                # 启用投机解码时，额外计算接受率等指标写入 meta_info。
                if self.server_args.speculative_algorithm:
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
                if self.server_args.enable_lora and state.obj.lora_path:
                    asyncio.create_task(self.lora_registry.release(state.obj.lora_id))

            state.out_list.append(out_dict)
            state.event.set()

            # Log metrics and dump
            if self.enable_metrics and state.obj.log_metrics:
                self.collect_metrics(state, recv_obj, i)
            if self.dump_requests_folder and state.finished and state.obj.log_metrics:
                self.dump_requests(state, out_dict)
            if self.crash_dump_folder and state.finished and state.obj.log_metrics:
                self.record_request_for_crash_dump(state, out_dict)

        # When skip_tokenizer_init is enabled, tokensizer_manager receives
        # BatchTokenIDOutput.
        if (
            self.server_args.dp_size > 1
            and isinstance(recv_obj, (BatchStrOutput, BatchTokenIDOutput))
            and recv_obj.load is not None
        ):
            load_update_req = WatchLoadUpdateReq(loads=[recv_obj.load])
            self.send_to_scheduler.send_pyobj(load_update_req)

    def add_logprob_to_meta_info(
        self,
        meta_info: dict,
        state: ReqState,
        top_logprobs_num: int,
        token_ids_logprob: List[int],
        return_text_in_logprobs: bool,
    ):
        # 把累积在 state 上的 logprob（含 token id、可选文本）整理后写入 meta_info，返回给上层。
        # 设计上 *_val 是新到的原始值，*_logprobs 是已 detokenize 过的结果；只对新增部分补做 detokenize，避免重复。
        # 1. Handle regular logprobs
        # 普通 logprob：val 比已处理的多出来的部分才需要 detokenize。
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
        # top-k logprob：仅当请求要求返回 top_logprobs（top_logprobs_num > 0）时处理。
        if top_logprobs_num > 0:
            if len(state.input_top_logprobs_val) > len(state.input_top_logprobs):
                state.input_top_logprobs.extend(
                    self.detokenize_top_logprobs_tokens(
                        state.input_top_logprobs_val[len(state.input_top_logprobs) :],
                        state.input_top_logprobs_idx[len(state.input_top_logprobs) :],
                        return_text_in_logprobs,
                    )
                )
            if len(state.output_top_logprobs_val) > len(state.output_top_logprobs):
                state.output_top_logprobs.extend(
                    self.detokenize_top_logprobs_tokens(
                        state.output_top_logprobs_val[len(state.output_top_logprobs) :],
                        state.output_top_logprobs_idx[len(state.output_top_logprobs) :],
                        return_text_in_logprobs,
                    )
                )

            meta_info["input_top_logprobs"] = state.input_top_logprobs
            meta_info["output_top_logprobs"] = state.output_top_logprobs

        # 3. Handle token_ids_logprob
        # 指定 token id 的 logprob：用户显式列出想看的 token id 时才处理。
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
        # 把本批 recv_obj 中第 recv_obj_index 个请求的原始 logprob 数据追加到 state 上，
        # 再调用 add_logprob_to_meta_info 整理写入 meta_info。state 在多批流式输出间累积。
        # 上游未返回 logprob 时直接跳过。
        if recv_obj.input_token_logprobs_val is None:
            return

        # input 部分仅在首批（prefill）返回，后续 decode 批可能为空，故需判空。
        if (
            len(recv_obj.input_token_logprobs_val) > 0
            and recv_obj.input_token_logprobs_val[recv_obj_index] is not None
        ):
            state.input_token_logprobs_val.extend(
                recv_obj.input_token_logprobs_val[recv_obj_index]
            )
            state.input_token_logprobs_idx.extend(
                recv_obj.input_token_logprobs_idx[recv_obj_index]
            )
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

        # 数据累积完毕后统一整理进 meta_info（含必要的 detokenize）。
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
        # 把 (logprob 值, token id) 列表组装成 (logprob, token_id, text) 三元组。
        # decode_to_text 为 False 时不反 tokenize，text 位填 None，省去解码开销。
        if not decode_to_text:
            return [
                (logprob, token_id, None)
                for logprob, token_id in zip(token_logprobs_val, token_logprobs_idx)
            ]
        else:
            assert self.tokenizer is not None
            # In transformers v5, batch_decode([1, 2, 3]) concatenates all tokens
            # into one string. Wrap each ID in its own list so they decode separately.
            # transformers v5 中 batch_decode([1,2,3]) 会把所有 token 拼成一个字符串，
            # 故把每个 id 单独包成一个列表，保证逐 token 独立解码。
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
        # 处理 top-k logprob：外层每个元素对应一个位置，内层是该位置的 top-k 候选。
        # 逐位置调用 detokenize_logprob_tokens；某位置无数据则填 None 占位。
        # TODO: The current implementation only batches the detokenization for top-k tokens per single position.
        # We should batch all top-k tokens in all positions.
        # 当前实现只在单个位置内对 top-k 做批量解码，理想情况应跨所有位置一次性批量解码。
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
        """Calculate speculative decoding metrics, such as acceptance rate and acceptance length metrics."""
        # 计算投机解码（spec decoding）指标：接受率、平均接受长度等，写入 meta_info。
        # 需上游确实带有相关字段且发生过验证（spec_verify_ct > 0）才计算。
        if (
            hasattr(recv_obj, "spec_verify_ct")
            and recv_obj.spec_verify_ct[i] > 0
            and hasattr(recv_obj, "spec_accepted_tokens")
            and len(recv_obj.spec_accepted_tokens) > i
        ):
            # The draft tokens per speculative step (excluding the target-sampled token).
            # 每个投机步提出的 draft token 数（减 1 是排除 target 模型自身采样的那个 token）。
            num_guess_tokens = self.server_args.speculative_num_draft_tokens - 1
            # 总 draft token 数 = 验证步数 × 每步 draft 数。
            total_draft_tokens = recv_obj.spec_verify_ct[i] * num_guess_tokens
            accepted_tokens = recv_obj.spec_accepted_tokens[i]

            # Calculate per-request acceptance rate and average acceptance length.
            # 按请求计算接受率与平均接受长度（除零保护）。
            if total_draft_tokens > 0:
                # Calculate acceptance rate: accepted / (steps * lookahead)
                # 接受率 = 被接受的 draft token / 总 draft token。
                meta_info["spec_accept_rate"] = accepted_tokens / total_draft_tokens
                # 平均接受长度 = 总产出 token / 验证步数（每步平均落地多少 token）。
                meta_info["spec_accept_length"] = (
                    recv_obj.completion_tokens[i] / recv_obj.spec_verify_ct[i]
                )
                meta_info["spec_accept_token_num"] = accepted_tokens
                meta_info["spec_draft_token_num"] = total_draft_tokens
                meta_info["spec_verify_ct"] = recv_obj.spec_verify_ct[i]

            # Acceptance histogram: tracks how many decoding steps accepted a certain number of draft tokens.
            # 接受直方图：统计有多少解码步分别接受了 0/1/2... 个 draft token，反映接受分布。
            if (
                recv_obj.spec_acceptance_histogram
                and len(recv_obj.spec_acceptance_histogram) > i
                and recv_obj.spec_acceptance_histogram[i]
            ):
                meta_info["spec_accept_histogram"] = recv_obj.spec_acceptance_histogram[
                    i
                ]

    def _request_has_grammar(self, obj: GenerateReqInput) -> bool:
        # 判断请求是否使用了约束解码语法（json_schema/regex/ebnf/structural_tag 任一），用于 metrics 打标。
        return (
            obj.sampling_params.get("json_schema", None)
            or obj.sampling_params.get("regex", None)
            or obj.sampling_params.get("ebnf", None)
            or obj.sampling_params.get("structural_tag", None)
        )

    def collect_metrics(self, state: ReqState, recv_obj: BatchStrOutput, i: int):
        # 在每批输出到来时采集 Prometheus 指标：TTFT、token 间时延、完成请求的端到端统计等。
        completion_tokens = (
            recv_obj.completion_tokens[i]
            if getattr(recv_obj, "completion_tokens", None)
            else 0
        )

        # 组装指标标签：在采集器默认标签基础上叠加请求自定义标签与优先级。
        custom_labels = getattr(state.obj, "custom_labels", None)
        labels = dict(self.metrics_collector.labels)
        if custom_labels:
            labels.update(custom_labels)
        if self.enable_priority_scheduling:
            priority = getattr(state.obj, "priority", None)
            if priority is not None:
                labels["priority"] = str(priority)
        # 首 token 尚未记录且本进程不是纯 PREFILL 角色时，记录 TTFT（首 token 时延）。
        # PD 分离下 PREFILL 节点不产出可见 token，故跳过避免重复/错误统计。
        if (
            not state.ttft_observed
            and self.disaggregation_mode != DisaggregationMode.PREFILL
        ):
            state.ttft_observed = True
            state.last_completion_tokens = completion_tokens
            self.metrics_collector.observe_time_to_first_token(
                labels, state.time_stats.get_first_token_latency()
            )
        else:
            # 非首批：用本批新增 token 数与时间间隔统计 token 间时延（ITL）。
            num_new_tokens = completion_tokens - state.last_completion_tokens
            if num_new_tokens:
                self.metrics_collector.observe_inter_token_latency(
                    labels,
                    state.time_stats.get_interval(),
                    num_new_tokens,
                )
                state.time_stats.set_last_time()
                state.last_completion_tokens = completion_tokens

        # 请求结束时记录一次完整的请求级指标。
        if state.finished:
            # 回退（retraction）次数：因显存压力被换出重算的次数，无该字段时记 0。
            retraction_count = (
                recv_obj.retraction_counts[i]
                if getattr(recv_obj, "retraction_counts", None)
                and i < len(recv_obj.retraction_counts)
                else 0
            )

            # Get detailed cache breakdown if available
            # 若上游提供，取更细粒度的 cache 命中分解信息。
            cached_tokens_details = None
            if (
                hasattr(recv_obj, "cached_tokens_details")
                and recv_obj.cached_tokens_details
            ):
                cached_tokens_details = recv_obj.cached_tokens_details[i]

            self.metrics_collector.observe_one_finished_request(
                labels,
                recv_obj.prompt_tokens[i],
                completion_tokens,
                recv_obj.cached_tokens[i],
                state.time_stats.get_e2e_latency(),
                self._request_has_grammar(state.obj),
                retraction_count,
                cached_tokens_details,
            )

    def dump_requests(self, state: ReqState, out_dict: dict):
        # 把已完成请求（输入对象 + 输出 + 起止时间）累积起来，用于离线调试/分析。
        self.dump_request_list.append(
            (
                state.obj,
                out_dict,
                convert_time_to_realtime(state.time_stats.created_time),
                convert_time_to_realtime(state.time_stats.finished_time),
            )
        )

        # 累积到阈值后按时间戳生成文件名，整批 dump 到磁盘并清空缓冲。
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
        # 维护一个滚动窗口，缓存最近完成的请求，供进程崩溃时转储现场（见 dump_requests_before_crash）。
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
        # 按结束时间淘汰窗口中超过 5 分钟（300 秒）的旧请求，控制内存占用。
        while (
            self.crash_dump_request_list
            and current_time - self.crash_dump_request_list[0][3] >= 300
        ):
            self.crash_dump_request_list.popleft()

    def _dump_data_to_file(
        self, data_list: List[Tuple], filename: str, log_message: str
    ):
        # 把请求数据连同 server_args 一起 pickle 落盘的通用方法，供 dump_requests / 崩溃转储复用。
        logger.info(log_message)
        # 先 copy 一份，避免后续调用方清空原列表时影响异步写盘的数据。
        to_dump_with_server_args = {
            "server_args": self.server_args,
            "requests": data_list.copy(),
        }

        # 实际写盘放到后台任务执行，避免阻塞事件循环。
        def background_task():
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "wb") as f:
                pickle.dump(to_dump_with_server_args, f)

        asyncio.create_task(asyncio.to_thread(background_task))

    def dump_requests_before_crash(
        self, hostname: str = os.getenv("HOSTNAME", socket.gethostname())
    ):
        # 进程在收到 SIGTERM/SIGQUIT 或发生异常即将崩溃前，把「已完成 + 在途」的请求
        # 一次性 pickle 转储到磁盘，便于事后复现与排障。
        if not self.crash_dump_folder:
            return  # 未配置转储目录则跳过

        if self.crash_dump_performed:
            # 同一次崩溃可能触发多个信号/异常，保证只转储一次。
            logger.info(
                "SIGTERM/SIGQUIT/Exception triggered, but crash dump already performed, skipping."
            )
            return
        else:
            self.crash_dump_performed = True

        logger.error(f"Dumping requests before crash. {self.crash_dump_folder=}")

        # 先收集已完成请求（来自滚动保存的 crash_dump_request_list）。
        data_to_dump = []
        if self.crash_dump_request_list:
            data_to_dump.extend(self.crash_dump_request_list)

        # 再收集仍在途（未结束）的请求，补打结束时间后一并转储。
        unfinished_requests = []
        for rid, state in self.rid_to_state.items():
            if not state.finished:
                state.time_stats.set_finished_time()
                unfinished_requests.append(
                    (
                        state.obj,
                        state.out_list[-1] if state.out_list else {},
                        convert_time_to_realtime(state.time_stats.created_time),
                        convert_time_to_realtime(state.time_stats.finished_time),
                    )
                )
        if unfinished_requests:
            data_to_dump.extend(unfinished_requests)

        if not data_to_dump:
            return  # 没有任何请求可转储

        # 按 主机名/时间戳 生成转储文件路径（.pkl）。
        filename = os.path.join(
            self.crash_dump_folder,
            hostname,
            f'crash_dump_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pkl',
        )
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # 连同 server_args、启动命令一起 pickle 落盘，方便完整复现现场。
        data_to_dump_with_server_args = {
            "server_args": self.server_args,  # Include server_args in the dump
            "requests": data_to_dump,
            "launch_command": " ".join(sys.argv),
        }
        with open(filename, "wb") as f:
            pickle.dump(data_to_dump_with_server_args, f)
        logger.error(
            f"Dumped {len(self.crash_dump_request_list)} finished and {len(unfinished_requests)} unfinished requests before crash to {filename}"
        )
        return filename

    async def sigterm_watchdog(self):
        # 优雅退出看门狗：后台常驻协程，收到 SIGTERM 后负责 drain 在途请求再退出
        # 先自旋等待退出标志被 SignalHandler 置位
        while not self.gracefully_exit:
            await asyncio.sleep(5)

        # Drain requests
        # 退出前等待已接收的请求处理完（rid_to_state 清空），避免丢弃在途请求
        while True:
            remain_num_req = len(self.rid_to_state)
            remaining_rids = list(self.rid_to_state.keys())

            if self.server_status == ServerStatus.UnHealthy:
                # if health check failed, we should exit immediately
                # 健康检查已失败，没有 drain 的意义，立即强制退出
                logger.error(
                    "Signal SIGTERM received while health check failed. Force exiting."
                )
                self.dump_requests_before_crash()
                self.force_exit_handler()
                break

            elif get_bool_env_var("SGL_FORCE_SHUTDOWN"):
                # if force shutdown flag set, exit immediately
                # 显式设置了强制关闭环境变量，跳过 drain 立即退出
                logger.error(
                    "Signal SIGTERM received while force shutdown flag set. Force exiting."
                )
                self.force_exit_handler()
                break

            logger.info(
                f"Gracefully exiting... Remaining number of requests {remain_num_req}. Remaining requests {remaining_rids=}."
            )
            if remain_num_req > 0:
                # 还有在途请求，等待 5 秒后再次检查
                await asyncio.sleep(5)
            else:
                # 请求已全部完成，落盘后退出循环
                self.dump_requests_before_crash()
                break

        # 杀掉整个进程树（含父进程），确保所有子进程一并退出
        kill_process_tree(os.getpid(), include_parent=True)
        sys.exit(0)

    def force_exit_handler(self):
        """Put some custom force exit logic here."""
        # 预留的强制退出钩子，供子类/部署场景注入自定义清理逻辑
        pass

    def _handle_abort_req(self, recv_obj: AbortReq):
        # 处理 scheduler 回传的中止消息（请求在校验或等待队列阶段被 abort）
        # 构造一个带 abort finish_reason 的输出，唤醒等待该 rid 的上层协程
        if is_health_check_generate_req(recv_obj):
            # 健康检查请求无上层等待者，直接忽略
            return
        state = self.rid_to_state[recv_obj.rid]
        state.finished = True
        state.time_stats.set_finished_time()

        abort_message = recv_obj.abort_message or "Abort in waiting queue"
        finish_reason = {
            "type": "abort",
            "message": abort_message,
        }
        if recv_obj.finished_reason:
            # scheduler 已给出更具体的结束原因时优先采用
            finish_reason = recv_obj.finished_reason
        meta_info = {
            "id": recv_obj.rid,
            "finish_reason": finish_reason,
            "weight_version": self.server_args.weight_version,
            "e2e_latency": state.time_stats.get_e2e_latency(),
        }
        is_stream = getattr(state.obj, "stream", False)
        if getattr(state.obj, "return_logprob", False):
            self.add_logprob_to_meta_info(
                meta_info,
                state,
                state.obj.top_logprobs_num,
                state.obj.token_ids_logprob,
                state.obj.return_text_in_logprobs
                and not self.server_args.skip_tokenizer_init,
            )

        output_ids = state.output_ids
        meta_info["completion_tokens"] = len(output_ids)
        if is_stream:
            # 流式场景只回传最后一个增量 token，避免重复已 yield 过的内容
            output_ids = [output_ids[-1]] if len(output_ids) > 0 else []
        out = {
            "text": state.text,
            "output_ids": output_ids,
            "meta_info": meta_info,
        }
        state.out_list.append(out)
        state.event.set()  # 唤醒等待该 rid 输出的生成协程

    def update_active_ranks(self, ranks: ActiveRanksOutput):
        # 把活跃 rank 信息转发给 scheduler（用于动态调整参与计算的 rank 集合）
        self.send_to_scheduler.send_pyobj(ranks)

    def _handle_open_session_req_output(self, recv_obj):
        # 处理 open_session 控制面回包：用 session_id 填充对应 future（失败则填 None）
        self.session_futures[recv_obj.session_id].set_result(
            recv_obj.session_id if recv_obj.success else None
        )

    def _handle_update_weights_from_disk_req_output(self, recv_obj):
        # 处理从磁盘更新权重的回包
        if self.server_args.dp_size == 1:
            # 单 DP：直接用回包结果完成 future
            self.model_update_result.set_result(recv_obj)
        else:  # self.server_args.dp_size > 1
            # 多 DP：每个 DP rank 各回一次，先暂存
            self.model_update_tmp.append(recv_obj)
            # set future if the all results are received
            # 集齐所有 DP 的结果后再统一完成 future
            if len(self.model_update_tmp) == self.server_args.dp_size:
                self.model_update_result.set_result(self.model_update_tmp)

    async def _validate_and_resolve_lora(
        self, obj: Union[GenerateReqInput, EmbeddingReqInput]
    ) -> None:
        # 校验请求所带的 LoRA 适配器：未启用 LoRA 时报错，启用则进一步解析路径
        if not obj.lora_path:
            # 未指定 LoRA，直接返回
            return

        if not self.server_args.enable_lora:
            # 请求了 LoRA 但服务端未开启该功能，给出明确报错指引
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
        # 解析 LoRA 路径：必要时重新加载被动态卸载的适配器，并向 registry 申请 lora_id
        if isinstance(obj.lora_path, str):
            unique_lora_paths = set([obj.lora_path])
        else:
            # batch 请求里可能有多个不同适配器，去重统计
            unique_lora_paths = set(obj.lora_path)

        if (
            self.server_args.max_loaded_loras is not None
            and len(unique_lora_paths) > self.server_args.max_loaded_loras
        ):
            # 单请求需要的适配器数超过同时可加载上限，无法满足
            raise ValueError(
                f"Received request with {len(unique_lora_paths)} unique loras requested "
                f"but max loaded loras is {self.server_args.max_loaded_loras}"
            )

        # Reload all existing LoRA adapters that have been dynamically unloaded
        # 找出已被动态卸载、当前未注册的适配器，逐个重新加载回来
        unregistered_loras = await self.lora_registry.get_unregistered_loras(
            unique_lora_paths
        )
        for lora_path in unregistered_loras:
            if lora_path is None:
                continue

            if lora_path not in self.lora_ref_cache:
                # 从未加载过的适配器无法隐式重载，报错
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
                # 并发下可能已被其他请求加载（"already loaded"），仅此情况可容忍
                raise ValueError(
                    f"Failed to implicitly load LoRA adapter {lora_path}: {load_result.error_message}"
                )

        # Look up the LoRA ID from the registry and start tracking ongoing LoRA requests.
        # 向 registry 申请 lora_id，并开始跟踪该 LoRA 的在途请求（引用计数）
        obj.lora_id = await self.lora_registry.acquire(obj.lora_path)

    def _req_stats_init(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        # 为请求初始化计时统计与 ReqState：登记到 rid_to_state，并按需建立 trace 上下文
        calibrate_time_diff()  # 校准进程间时钟偏差，保证跨进程时间统计可比
        created_time = obj.received_time

        external_trace_header = None
        if self.server_args.enable_trace:
            if obj.external_trace_header:
                # When the request comes from the rust grpc server or Engine there isn't a
                # real request object but we still need to propagate the trace context from
                # the trace context that is explicitly passed in
                # 来自 rust grpc server 或 Engine 时没有真实 request 对象，
                # 但仍需沿用显式传入的 trace 上下文
                external_trace_header = obj.external_trace_header
            elif request:
                # 普通 HTTP 入口：从请求头里提取 trace 上下文
                external_trace_header = extract_trace_headers(request.headers)
                obj.external_trace_header = external_trace_header

        if not hasattr(obj, "is_single") or obj.is_single:
            # 单请求：建立一个 ReqState 并登记到 rid_to_state
            time_stats = APIServerReqTimeStats(disagg_mode=self.disaggregation_mode)
            state = ReqState([], False, asyncio.Event(), obj, time_stats)
            self.rid_to_state[obj.rid] = state

            if self.server_args.enable_trace:
                bootstrap_room = (
                    obj.bootstrap_room if hasattr(obj, "bootstrap_room") else None
                )
                time_stats.init_trace_ctx(
                    obj.rid,
                    bootstrap_room,
                    external_trace_header,
                )
            time_stats.set_created_time(created_time)
        else:
            # batch 请求：为每个子请求 rid 分别建立 ReqState 与计时
            for i in range(len(obj.rid)):
                time_stats = APIServerReqTimeStats(disagg_mode=self.disaggregation_mode)
                state = ReqState([], False, asyncio.Event(), obj[i], time_stats)
                self.rid_to_state[obj.rid[i]] = state

                if self.server_args.enable_trace:
                    bootstrap_room = (
                        obj.bootstrap_room[i]
                        if hasattr(obj, "bootstrap_room") and obj.bootstrap_room
                        else None
                    )
                    time_stats.init_trace_ctx(
                        obj.rid[i],
                        bootstrap_room,
                        external_trace_header,
                    )
                time_stats.set_created_time(created_time)

    def _should_dispatch_to_encoder(
        self, obj: Union[GenerateReqInput, EmbeddingReqInput]
    ) -> bool:
        """Check if the request should be dispatched to encoder for processing.

        Returns True if the request should be dispatched to encoder (multiple multimodal items),
        False if it should be processed locally (single multimodal item or no multimodal items).

        Args:
            obj: The request input object

        Returns:
            bool: True if should dispatch to encoder, False otherwise
        """
        # 判断请求是否应分派到独立 encoder（EPD 分离模式下，多模态项较多时才值得分派）
        if obj.batch_size > 1:
            # EPD 分离模式暂不支持 batch 请求，不分派
            logger.warning(
                "Batch request (batch_size=%d) is not supported in EPD disaggregation mode; skipping encoder dispatch.",
                obj.batch_size,
            )
            return False
        if not isinstance(obj, GenerateReqInput) or not obj.contains_mm_input():
            # 非生成请求或不含多模态输入，无需分派到 encoder
            return False

        # Count image / video / audio items for dispatch threshold
        # 统计图像/视频/音频项总数，与分派阈值比较
        def _count_mm_items(data):
            # 列表取长度，单项计 1，None 计 0
            return (
                len(data) if isinstance(data, list) else (1 if data is not None else 0)
            )

        total_mm_items = (
            _count_mm_items(getattr(obj, "image_data", None))
            + _count_mm_items(getattr(obj, "video_data", None))
            + _count_mm_items(getattr(obj, "audio_data", None))
        )
        # 多模态项数达到阈值才分派，避免单项请求的跨进程开销得不偿失
        return total_mm_items >= envs.SGLANG_ENCODER_DISPATCH_MIN_ITEMS.get()

    def _handle_epd_disaggregation_encode_request(
        self, obj: Union[GenerateReqInput, EmbeddingReqInput]
    ):
        """Handle EPD-disaggregation mode encoding request."""
        # EPD 分离模式下的编码请求处理：决定是否把多模态编码下放到独立 encoder
        if isinstance(obj, GenerateReqInput) and obj.contains_mm_input():
            # dispatch to encoder by default
            # 默认分派到 encoder
            should_dispatch = True
            if self.server_args.enable_adaptive_dispatch_to_encoder:
                # 开启自适应分派时，按多模态项数阈值动态决定
                should_dispatch = self._should_dispatch_to_encoder(obj)

            # Set need_wait_for_mm_inputs flag based on whether we dispatch to encoder
            # This flag will be used in _tokenize_one_request to determine processing path
            # 该标志供 _tokenize_one_request 决定走哪条处理路径（是否等待 encoder 产出）
            if should_dispatch:
                obj.need_wait_for_mm_inputs = True
                if self.server_args.encoder_transfer_backend == "zmq_to_scheduler":
                    # 经 ZMQ 把编码请求直接发给 scheduler 侧的 encoder
                    self.mm_receiver.send_encode_request(obj)
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
        """Convert attributes to span attributes."""
        # 把请求/响应的各项指标转换为 OpenTelemetry span 属性，供分布式追踪上报
        span_attrs = {}

        if not self.server_args.enable_trace:
            # 未开启 trace 时返回空属性
            return span_attrs

        # Token usage attributes
        # token 用量属性（embedding 输出没有 completion tokens）
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
        # 请求标识
        span_attrs[SpanAttributes.GEN_AI_REQUEST_ID] = (
            str(state.obj.rid) if state.obj.rid else None
        )

        # Sampling parameters
        # 采样参数（仅记录实际设置了的项）
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
        # 响应属性
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
        # 合并各阶段时延属性
        span_attrs.update(state.time_stats.convert_to_gen_ai_span_attrs())

        return span_attrs

    def _set_default_priority(self, obj: Union[GenerateReqInput, EmbeddingReqInput]):
        """Set the default priority value."""
        # 开启优先级调度且请求未显式指定优先级时，填入配置的默认优先级
        if (
            self.enable_priority_scheduling
            and obj.priority is None
            and self.default_priority_value is not None
        ):
            obj.priority = self.default_priority_value


# 服务器健康状态枚举：启动中 / 正常 / 不健康（健康检查失败）
class ServerStatus(Enum):
    Up = "Up"
    Starting = "Starting"
    UnHealthy = "UnHealthy"


async def print_exception_wrapper(func):
    """
    Sometimes an asyncio function does not print exception.
    We do another wrapper to handle the exception.
    """
    # asyncio 后台协程的异常常被静默吞掉，这里包一层确保打印并触发崩溃落盘+退出
    try:
        await func()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"TokenizerManager hit an exception: {traceback}")
        if hasattr(func, "__self__") and isinstance(func.__self__, TokenizerManager):
            # 绑定方法且属于 TokenizerManager 时，崩溃前把在途请求落盘便于排查
            func.__self__.dump_requests_before_crash()
        kill_process_tree(os.getpid(), include_parent=True)
        sys.exit(1)


def _get_processor_wrapper(server_args):
    # 加载多模态 processor 的包装：处理部分模型无 slow 版本时自动回退到 fast 版
    try:
        processor = get_processor(
            server_args.tokenizer_path,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            use_fast=not server_args.disable_fast_image_processor,
        )
    except ValueError as e:
        error_message = str(e)
        if "does not have a slow version" in error_message:
            # 该 processor 没有 slow 版本，自动改用 fast 版重试
            logger.info(
                f"Processor {server_args.tokenizer_path} does not have a slow version. Automatically use fast version"
            )
            processor = get_processor(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                use_fast=True,
            )
        else:
            raise e
    return processor


def _determine_tensor_transport_mode(server_args: ServerArgs) -> TensorTransportMode:
    # 决定张量在进程间的传输方式：跨节点回退到 CPU 默认传输，单节点用 cuda_ipc 共享显存
    is_cross_node = server_args.dist_init_addr

    if is_cross_node:
        # Fallback to default CPU transport for multi-node
        # 多节点无法用 CUDA IPC，回退默认 CPU 传输
        return "default"
    else:
        return "cuda_ipc"


# 信号处理器：把进程信号转译为 TokenizerManager 的退出动作
class SignalHandler:
    def __init__(self, tokenizer_manager: TokenizerManager):
        self.tokenizer_manager = tokenizer_manager

    def sigterm_handler(self, signum=None, frame=None):
        # 收到 SIGTERM：仅置位优雅退出标志，由 sigterm_watchdog 负责 drain 后退出
        logger.warning(
            f"SIGTERM received. {signum=} {frame=}. Draining requests and shutting down..."
        )
        self.tokenizer_manager.gracefully_exit = True

    def running_phase_sigquit_handler(self, signum=None, frame=None):
        # 收到 SIGQUIT：通常意味着某个子进程已挂掉，需立即崩溃式退出
        logger.error(
            f"SIGQUIT received. {signum=}, {frame=}. It usually means one child failed."
        )
        # Stop subprocess watchdog before killing processes to prevent false-positive
        # crash detection during normal shutdown
        # 杀进程前先停掉子进程看门狗，避免正常关停被误判为崩溃
        if self.tokenizer_manager._subprocess_watchdog is not None:
            self.tokenizer_manager._subprocess_watchdog.stop()
        self.tokenizer_manager.dump_requests_before_crash()
        kill_process_tree(os.getpid())


# Note: request abort handling logic
# We should handle all of the following cases correctly.
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
