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
"""A scheduler that manages a tensor parallel GPU worker."""
# 本文件实现 SGLang 的核心调度器（Scheduler）。
# 调度器运行在独立进程中，负责管理张量并行（TP）的 GPU worker，主要职责包括：
#   1. 通过 ZMQ 从 TokenizerManager 接收请求、向 DetokenizerManager 发送结果；
#   2. 维护等待队列（waiting_queue）与运行批次（running_batch），实现连续批处理（continuous batching）；
#   3. 组织 prefill（预填充）与 decode（解码）批次的调度策略、KV 缓存与显存池管理；
#   4. 驱动前向计算（run_batch）并处理输出（process_batch_result）；
#   5. 支持多种高级特性：投机解码、PD 分离部署、流水线并行（PP）、DP attention、
#      分层缓存（HiCache）、LoRA、结构化生成（grammar）等。
# Scheduler 通过继承多个 Mixin 将上述能力组合到一起，事件循环是其运行的主入口。

import faulthandler
import logging
import os
import signal
import sys
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

from sglang.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()

import psutil
import setproctitle
import torch
import torch.distributed
import zmq
from torch.cuda import Stream as CudaStream
from torch.distributed import barrier

from sglang.jit_kernel.ngram_embedding import update_token_table
from sglang.srt.configs.model_config import ModelConfig, ModelImpl
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.encode_receiver import create_mm_receiver
from sglang.srt.disaggregation.prefill import (
    PrefillBootstrapQueue,
    SchedulerDisaggregationPrefillMixin,
    release_req_to_metadata_buffer,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    prepare_abort,
)
from sglang.srt.distributed import get_pp_group, get_world_group
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.attention.mamba.ops import (
    initialize_mamba_selective_state_update_backend,
)
from sglang.srt.layers.dp_attention import (
    compute_dp_attention_world_info,
    get_attention_cp_group,
    get_attention_tp_group,
)
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.lora.lora_overlap_loader import LoRAOverlapLoader
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.io_struct import (
    AbortReq,
    ActiveRanksOutput,
    AttachHiCacheStorageReqInput,
    AttachHiCacheStorageReqOutput,
    BaseBatchReq,
    BaseReq,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    CheckWeightsReqInput,
    ClearHiCacheReqInput,
    ClearHiCacheReqOutput,
    CloseSessionReqInput,
    ContinueGenerationReqInput,
    DestroyWeightsUpdateGroupReqInput,
    DetachHiCacheStorageReqInput,
    DetachHiCacheStorageReqOutput,
    DumperControlReqInput,
    DumperControlReqOutput,
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    ExpertDistributionReqType,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    FreezeGCReq,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetLoadReqInput,
    GetLoadsReqInput,
    GetWeightsByNameReqInput,
    HealthCheckOutput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsSendGroupForRemoteInstanceReqOutput,
    InitWeightsUpdateGroupReqInput,
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterFromTensorsReqOutput,
    LoadLoRAAdapterReqInput,
    LoadLoRAAdapterReqOutput,
    OpenSessionReqInput,
    PauseGenerationReqInput,
    ProfileReq,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    RpcReqInput,
    RpcReqOutput,
    SendWeightsToRemoteInstanceReqInput,
    SendWeightsToRemoteInstanceReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    SlowDownReqInput,
    SlowDownReqOutput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UnloadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.mm_utils import (
    has_shm_features,
    init_mm_embedding_cache,
    unwrap_shm_features,
)
from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.managers.prefill_delayer import (
    PrefillDelayer,
    PrefillDelayerSinglePassExecutor,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    ModelWorkerBatch,
    MultimodalInputs,
    Req,
    ScheduleBatch,
)
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    SchedulePolicy,
)
from sglang.srt.managers.scheduler_dp_attn_mixin import SchedulerDPAttnMixin
from sglang.srt.managers.scheduler_input_blocker import SchedulerInputBlocker
from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.managers.scheduler_profiler_mixin import SchedulerProfilerMixin
from sglang.srt.managers.scheduler_recv_skipper import SchedulerRecvSkipper
from sglang.srt.managers.scheduler_runtime_checker_mixin import (
    SchedulerRuntimeCheckerMixin,
    create_scheduler_watchdog,
)
from sglang.srt.managers.scheduler_update_weights_mixin import (
    SchedulerUpdateWeightsMixin,
)
from sglang.srt.managers.session_controller import SessionController
from sglang.srt.managers.utils import GenerationBatchResult, validate_input_length
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.session_aware_cache import SessionAwareCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.model_loader.utils import get_resolved_model_impl
from sglang.srt.multiplex.multiplexing_mixin import SchedulerMultiplexMixin
from sglang.srt.observability.req_time_stats import (
    real_time,
    set_schedule_time_batch,
    set_time_batch,
)
from sglang.srt.observability.scheduler_metrics_mixin import (
    RECORD_STEP_TIME,
    PrefillStats,
    SchedulerMetricsMixin,
)
from sglang.srt.observability.trace import process_tracing_init, trace_set_thread_info
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import PortArgs, ServerArgs, get_global_server_args
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    DynamicGradMode,
    broadcast_pyobj,
    configure_gc_logger,
    configure_logger,
    freeze_gc,
    get_available_gpu_memory,
    get_bool_env_var,
    get_int_env_var,
    is_mps,
    kill_itself_when_parent_died,
    point_to_point_pyobj,
    require_mlp_sync,
    set_gpu_proc_affinity,
    set_random_seed,
    suppress_other_loggers,
)
from sglang.srt.utils.common import is_npu
from sglang.srt.utils.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
)
from sglang.srt.utils.network import get_zmq_socket
from sglang.srt.utils.numa_utils import get_numa_node_if_available, numa_bind_to_node
from sglang.srt.utils.tensor_bridge import use_mlx
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

if is_mps():
    CudaStreamContext = nullcontext
else:
    from torch.cuda import StreamContext as CudaStreamContext

logger = logging.getLogger(__name__)

# 用于调试目的的 retract（回退）decode 测试开关
TEST_RETRACT = envs.SGLANG_TEST_RETRACT.get()
TEST_RETRACT_INTERVAL = envs.SGLANG_TEST_RETRACT_INTERVAL.get()
TEST_RETRACT_NO_PREFILL_BS = envs.SGLANG_TEST_RETRACT_NO_PREFILL_BS.get()

_is_npu = is_npu()


# Embedding/Reward 模型一次前向计算的结果封装。
@dataclass
class EmbeddingBatchResult:
    # 模型输出的 embedding 张量（可能在 GPU 上）。
    embeddings: torch.Tensor
    # overlap 调度下用于标记拷贝到 CPU 完成的事件。
    copy_done: Optional[torch.cuda.Event] = None

    def copy_to_cpu(self):
        """Copy embeddings tensor to CPU in overlap scheduling."""
        # 在 overlap（CPU 调度与 GPU 计算重叠）调度下，将 embedding 异步拷贝到 CPU，
        # 并记录一个事件，便于后续等待拷贝完成后再读取结果。

        if isinstance(self.embeddings, torch.Tensor):
            self.copy_done = torch.get_device_module(self.embeddings.device).Event()
            self.embeddings = self.embeddings.to("cpu", non_blocking=True)
        else:
            assert isinstance(self.embeddings, list)
            if len(self.embeddings) == 0:
                return

            self.copy_done = torch.get_device_module(self.embeddings[0].device).Event()
            self.embeddings = [
                emb.to("cpu", non_blocking=True) for emb in self.embeddings
            ]

        self.copy_done.record()


# Scheduler 通过继承一系列 Mixin 组合出完整功能，每个 Mixin 负责一类职责：
class Scheduler(
    SchedulerOutputProcessorMixin,  # 输出处理（生成结果的后处理、流式发送等）
    SchedulerUpdateWeightsMixin,  # 在线权重更新（RLHF 等场景）
    SchedulerProfilerMixin,  # 性能分析（profiler）
    SchedulerMetricsMixin,  # 监控指标统计
    SchedulerDisaggregationDecodeMixin,  # PD 分离部署的 decode 端逻辑
    SchedulerDisaggregationPrefillMixin,  # PD 分离部署的 prefill 端逻辑
    SchedulerMultiplexMixin,  # PD 多路复用（pdmux）
    SchedulerRuntimeCheckerMixin,  # 运行期自检（显存、状态一致性等）
    SchedulerPPMixin,  # 流水线并行（Pipeline Parallel）
    SchedulerDPAttnMixin,  # DP attention 相关同步
    SchedulerDllmMixin,  # 扩散式 LLM（diffusion LLM）支持
):
    """A scheduler that manages a tensor parallel GPU worker."""

    def __init__(
        self,
        # 全局服务配置（模型路径、并行规模、各类功能开关等），几乎所有初始化都依赖它。
        server_args: ServerArgs,
        # 进程间通信端口配置（ZMQ ipc 名称、nccl 端口等）。
        port_args: PortArgs,
        # 本调度器进程绑定的 GPU 编号。
        gpu_id: int,
        # 张量并行（Tensor Parallel）的 rank（取值 0..tp_size-1）。
        tp_rank: int,
        # MoE 专家并行（Expert Parallel）的 rank（取值 0..ep_size-1）。
        moe_ep_rank: int,
        # 流水线并行（Pipeline Parallel）的 rank（取值 0..pp_size-1）。
        pp_rank: int,
        # 注意力上下文并行（Attention Context Parallel）的 rank。
        attn_cp_rank: int,
        # MoE 数据并行（Data Parallel）的 rank。
        moe_dp_rank: int,
        # 数据并行（Data Parallel）的 rank；未启用 DP 时为 None（也可由环境变量 SGLANG_DP_RANK 提供）。
        dp_rank: Optional[int],
    ):
        # 标记调度器正处于初始化阶段（用于软看门狗等组件判断当前状态）。
        self.is_initializing = True
        # 先启动软看门狗，监控初始化是否超时/卡死。
        self.init_soft_watchdog(server_args)

        # 解析并保存各类配置参数与分布式 rank 信息。
        # 约定：*_rank 为本进程在该并行维度中的编号，*_size 为该维度的总规模。
        self.server_args = server_args
        self.tp_rank = tp_rank  # 张量并行 rank
        self.moe_ep_rank = moe_ep_rank  # MoE 专家并行 rank
        self.pp_rank = pp_rank  # 流水线并行 rank
        self.attn_cp_rank = attn_cp_rank  # 注意力上下文并行 rank
        self.attn_cp_size = server_args.attn_cp_size  # 注意力上下文并行规模
        self.moe_dp_rank = moe_dp_rank  # MoE 数据并行 rank
        self.moe_dp_size = server_args.moe_dp_size  # MoE 数据并行规模
        self.dp_rank = dp_rank  # 数据并行 rank（未启用时为 None）
        self.tp_size = server_args.tp_size  # 张量并行规模
        self.moe_ep_size = server_args.ep_size  # MoE 专家并行规模
        self.pp_size = server_args.pp_size  # 流水线并行规模
        self.dp_size = server_args.dp_size  # 数据并行规模
        self.nccl_port = port_args.nccl_port  # NCCL 通信端口
        self.schedule_policy = server_args.schedule_policy  # 调度策略（如 fcfs、lpm 等）
        self.enable_priority_scheduling = server_args.enable_priority_scheduling
        self.abort_on_priority_when_disabled = (
            server_args.abort_on_priority_when_disabled
        )
        self.schedule_low_priority_values_first = (
            server_args.schedule_low_priority_values_first
        )
        self.priority_scheduling_preemption_threshold = (
            server_args.priority_scheduling_preemption_threshold
        )
        self.enable_lora = server_args.enable_lora
        self.enable_lora_overlap_loading = server_args.enable_lora_overlap_loading
        self.max_loras_per_batch = server_args.max_loras_per_batch
        self.enable_overlap = not server_args.disable_overlap_schedule
        self.enable_pdmux = server_args.enable_pdmux
        self.skip_tokenizer_init = server_args.skip_tokenizer_init
        self.stream_interval = server_args.stream_interval
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.gpu_id = gpu_id
        self.page_size = server_args.page_size
        self.enable_hierarchical_cache = server_args.enable_hierarchical_cache
        self.enable_hicache_storage = server_args.hicache_storage_backend is not None
        self.max_recv_per_poll = envs.SGLANG_SCHEDULER_MAX_RECV_PER_POLL.get()
        self.enable_hisparse = server_args.enable_hisparse
        self.hisparse_coordinator: Optional[HiSparseCoordinator] = None

        # 分布式 rank 信息：根据是否启用 DP attention 计算 attention 维度下的 TP rank/size 与 DP rank。
        self.attn_tp_rank, self.attn_tp_size, self.attn_dp_rank = (
            compute_dp_attention_world_info(
                server_args.enable_dp_attention,
                self.tp_rank,
                self.tp_size,
                self.dp_size,
                self.attn_cp_size,
            )
        )

        self.enable_kv_cache_events = bool(
            server_args.kv_events_config and self.attn_tp_rank == 0
        )

        # 初始化模型配置（从 server_args 解析模型结构、精度等信息）
        self.init_model_config()

        # 初始化监控指标统计
        self.init_metrics(tp_rank, pp_rank, dp_rank)

        # 初始化进程间通信（ZMQ 套接字：接收请求、发送结果等）
        self.init_ipc_channels(port_args)

        # 初始化 PD 多路复用（pdmux）上下文
        if self.enable_pdmux:
            self.init_pdmux()

        # 初始化分词器（tokenizer）
        self.init_tokenizer()

        # 初始化 MoE 配置与 GEMM 配置（如 FP8 GEMM 等）
        self.init_moe_gemm_config()

        # 初始化 mamba 后端
        self.init_mamba_backend()

        # 启动模型 worker；若启用投机解码则同时启动 draft 模型 worker
        self.init_model_worker()

        if (t := envs.SGLANG_TEST_STUCK_SCHEDULER_INIT.get()) > 0:
            time.sleep(t)

        # 初始化 KV 缓存（radix tree 等）与显存池
        self.init_cache_with_memory_pool()

        # 初始化运行状态（等待队列、运行批次、当前/上一批次等）
        self.init_running_status()

        # 初始化分块预填充（chunked prefill）
        self.init_chunked_prefill()

        # 初始化扩散式 LLM（diffusion LLM）
        self.init_diffusion_llm()

        # 初始化调度策略与新 token 数量预估
        self.init_schedule_policy()

        # 初始化看门狗、显存节省器、输入阻断器与接收跳过器
        self.init_watch_dog_memory_saver_input_blocker()

        # 初始化性能分析器（profiler）
        self.init_profiler()

        # 初始化 prefill-decode 分离部署（PD disaggregation）
        self.init_disaggregation()

        # 初始化 overlap 调度（CPU 调度与 GPU 计算重叠）
        self.init_overlap()

        # 初始化 N-gram embedding（如启用）
        self.maybe_init_ngram_embedding()

        # 当启用确定性推理时，针对不同 attention 后端初始化 prefill 的 KV 切分大小
        self.init_deterministic_inference_config()

        # 初始化请求分发器（将不同类型请求路由到对应处理函数）
        self.init_request_dispatcher()

        # 初始化 LoRA overlap 加载器（将 LoRA 权重加载与计算重叠）
        if self.enable_lora_overlap_loading:
            self.lora_overlap_loader = LoRAOverlapLoader(
                self.tp_worker.model_runner.lora_manager
            )

        # 初始化用于约束生成（结构化输出）的 grammar 后端
        self.grammar_manager = GrammarManager(self)

        # 初始化完成，清除初始化标记。
        self.is_initializing = False

    def init_model_config(self):
        """初始化模型配置；在 NPU 后端下还会根据 diffusion LLM 的 block_size 修正 page_size。"""
        self.model_config = ModelConfig.from_server_args(self.server_args)
        if _is_npu:
            # 确保在 NPU 后端上 page size 不大于 block_size 和 chunked_prefill_size
            # NPU 后端要求所定义的 page size 不超过 block_size 和 chunked_prefill_size
            from sglang.srt.dllm.config import DllmConfig

            self.dllm_config = (  # 用于 diffusion LLM
                DllmConfig.from_server_args(self.server_args)
                if self.server_args.dllm_algorithm is not None
                else None
            )
            if self.dllm_config:
                if self.dllm_config.block_size < self.page_size:
                    logger.warning(
                        "WARNING: "
                        f"The page size {self.page_size} should not be larger than dllm block size {self.dllm_config.block_size}."
                        f"Page size now falls back to {self.dllm_config.block_size}"
                    )
                    self.page_size = self.dllm_config.block_size

    def init_ipc_channels(self, port_args: PortArgs):
        """初始化进程间通信的 ZMQ 套接字。

        仅在入口 rank（pp/attn_tp/attn_cp 均为 0）上创建接收/发送套接字：
        从 TokenizerManager 拉取请求、接收 RPC，并向 Tokenizer/Detokenizer 推送结果。
        非入口 rank 则不创建真实套接字。
        """
        context = zmq.Context(2)
        self.idle_sleeper = None

        if self.pp_rank == 0 and self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )
            self.recv_from_rpc = get_zmq_socket(
                context, zmq.DEALER, port_args.rpc_ipc_name, False
            )

            send_to_tokenizer = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_ipc_name, False
            )
            if self.server_args.skip_tokenizer_init:
                # 直接发送给 TokenizerManager
                send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.tokenizer_ipc_name, False
                )
            else:
                # 发送给 DetokenizerManager
                send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.detokenizer_ipc_name, False
                )

            self.send_to_tokenizer = SenderWrapper(send_to_tokenizer)
            self.send_to_detokenizer = SenderWrapper(send_to_detokenizer)

            if self.server_args.sleep_on_idle:
                self.idle_sleeper = IdleSleeper(
                    [
                        self.recv_from_tokenizer,
                        self.recv_from_rpc,
                    ]
                )
        else:
            self.recv_from_tokenizer = None
            self.recv_from_rpc = None
            self.send_to_tokenizer = SenderWrapper(None)
            self.send_to_detokenizer = SenderWrapper(None)

        if self.current_scheduler_metrics_enabled:
            self.send_metrics_from_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.metrics_ipc_name, False
            )

    def init_tokenizer(self):
        """初始化分词器/多模态处理器，并按需准备推理解析器（reasoning parser）相关 token。"""
        server_args = self.server_args
        self.is_generation = self.model_config.is_generation

        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                    use_fast=not server_args.disable_fast_image_processor,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )

        # 加载多模态处理器，用于 M-RoPE 的回退计算。
        self._mm_processor = None
        if self.model_config.is_multimodal and self.processor is not None:
            try:
                import_processors("sglang.srt.multimodal.processors")
                self._mm_processor = get_mm_processor(
                    self.model_config.hf_config,
                    server_args,
                    self.processor,
                    "default",
                    skip_mm_pool=True,
                )
            except Exception:
                logger.warning(
                    "Failed to load multimodal processor in scheduler; "
                    "M-RoPE fallback will not be available."
                )

        # 若启用了 --reasoning_parser，则设置 reasoning_parser 与 think_end_id
        if self.server_args.reasoning_parser and self.tokenizer:
            reasoning_parser = ReasoningParser(
                model_type=self.server_args.reasoning_parser, stream_reasoning=False
            )
            self.tokenizer.think_end_id = self.tokenizer.encode(
                reasoning_parser.detector.think_end_token, add_special_tokens=False
            )[0]

    def init_mamba_backend(self) -> None:
        """初始化 mamba 的选择性状态更新（selective state update）后端。"""
        initialize_mamba_selective_state_update_backend(self.server_args)

    def init_moe_gemm_config(self):
        """初始化 MoE 及 FP8/FP4 GEMM 相关配置，并确定是否需要 MLP 同步。"""
        # 对于多模态（MM）模型，从 text_config 中检查 MoE 设置
        config_to_check = getattr(
            self.model_config.hf_config, "text_config", self.model_config.hf_config
        )

        if hasattr(config_to_check, "num_experts_per_tok"):
            initialize_moe_config(self.server_args)

        # 为 FP8 和 FP4 后端初始化 GEMM 相关配置。
        initialize_fp8_gemm_config(self.server_args)
        initialize_fp4_gemm_config(self.server_args)

        # 这一步必须在 initialize_moe_config 之后调用
        self.require_mlp_sync = require_mlp_sync(self.server_args)

    def init_tp_model_worker(self):
        """创建张量并行（TP）模型 worker，根据硬件后端选择 MLX 或通用 TpModelWorker。"""
        worker_kwargs = dict(
            server_args=self.server_args,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            moe_ep_rank=self.moe_ep_rank,
            pp_rank=self.pp_rank,
            attn_cp_rank=self.attn_cp_rank,
            moe_dp_rank=self.moe_dp_rank,
            dp_rank=self.dp_rank,
            nccl_port=self.nccl_port,
        )

        # FIXME: 将 tp worker 的初始化逻辑移到 scheduler 之外。
        if use_mlx():
            from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker

            self.tp_worker = MlxTpModelWorker(**worker_kwargs)
        else:
            from sglang.srt.managers.tp_worker import TpModelWorker

            self.tp_worker = TpModelWorker(**worker_kwargs)

    def maybe_init_draft_worker(self):
        """若启用投机解码，则创建 draft（草稿）模型 worker；否则不创建。"""
        if self.spec_algorithm.is_none():
            self.draft_worker = None
            return

        # 为投机解码（speculative decoding）启动一个 draft worker
        draft_worker_kwargs = dict(
            server_args=self.server_args,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            moe_ep_rank=self.moe_ep_rank,
            nccl_port=self.nccl_port,
            target_worker=self.tp_worker,
            dp_rank=self.dp_rank,
            attn_cp_rank=self.attn_cp_rank,
            moe_dp_rank=self.moe_dp_rank,
        )

        if self.server_args.speculative_draft_load_format is not None:
            self.server_args.load_format = (
                self.server_args.speculative_draft_load_format
            )
            logger.info(
                f"Using draft model load_format: '{self.server_args.speculative_draft_load_format}'"
            )

        DraftWorkerClass = self.spec_algorithm.create_worker(self.server_args)
        self.draft_worker = DraftWorkerClass(**draft_worker_kwargs)

    def init_model_worker(self):
        """初始化模型 worker：创建 TP worker 与（可选）draft worker，
        获取 token/显存上限等信息，并初始化各类分布式通信组与随机种子。"""
        self.init_tp_model_worker()
        self.maybe_init_draft_worker()

        # 选定实际使用的 model worker
        if self.spec_algorithm.is_none():
            self.model_worker = self.tp_worker
        else:
            self.model_worker = self.draft_worker

        # 从 model worker 获取 token 数与显存信息
        (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_queued_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            self.forward_stream,
            _,
            _,
            _,
        ) = self.tp_worker.get_worker_info()
        if get_global_server_args().pp_max_micro_batch_size is None:
            get_global_server_args().pp_max_micro_batch_size = max(
                self.max_running_requests // self.pp_size, 1
            )

        self.tp_group = get_tp_group()
        self.tp_cpu_group = self.tp_group.cpu_group
        self.attn_tp_group = get_attention_tp_group()
        self.attn_tp_cpu_group = self.attn_tp_group.cpu_group
        self.attn_cp_group = get_attention_cp_group()
        self.attn_cp_cpu_group = self.attn_cp_group.cpu_group
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()

        # NOTE: dp_tp_* 是请求/数据平面（data-plane）的协调通信组（不是张量集合通信组）。
        # 启用 DP attention 时，范围限定在 attention-TP group；否则使用基础的 TP group。
        # 入口 rank 即该组内的本地 rank 0。
        # 使用 CPU（gloo）通信组来广播 VLM 的 Python 对象，以避免 CUDA
        # stream/device 的耦合（#11910）。
        self.dp_tp_group = (
            self.attn_tp_group
            if self.server_args.enable_dp_attention
            else self.tp_group
        )
        self.dp_tp_cpu_group = self.dp_tp_group.cpu_group

        self.pad_input_ids_func = self.tp_worker.get_pad_input_ids_func()
        set_random_seed(self.random_seed)

        # 打印调试信息
        if self.tp_rank == 0:
            avail_mem = get_available_gpu_memory(
                self.device, self.gpu_id, empty_cache=False
            )
            logger.info(
                f"max_total_num_tokens={self.max_total_num_tokens}, "
                f"chunked_prefill_size={self.server_args.chunked_prefill_size}, "
                f"max_prefill_tokens={self.max_prefill_tokens}, "
                f"max_running_requests={self.max_running_requests}, "
                f"context_len={self.model_config.context_len}, "
                f"{'available_cpu_mem' if self.device == 'cpu' else 'available_gpu_mem'}={avail_mem:.2f} GB"
            )

        if self.enable_metrics and hasattr(self, "metrics_collector"):
            self.metrics_collector.emit_cache_config_info(
                self.page_size, self.max_total_num_tokens // self.page_size
            )

    def init_cache_with_memory_pool(self):
        """初始化 KV 缓存与显存池。

        根据是否启用分块预填充、分层缓存、混合 SWA/SSM、LMCache 等配置，
        选择并构造合适的 tree_cache 实现（ChunkCache/RadixCache/HiRadixCache 等）。
        """
        server_args = self.server_args
        uses_transformers_backend = (
            get_resolved_model_impl(self.model_config) == ModelImpl.TRANSFORMERS
        )

        # 混合（hybrid）显存池
        self.is_hybrid_swa = self.tp_worker.is_hybrid_swa
        self.is_hybrid_ssm = (
            self.tp_worker.model_runner.hybrid_gdn_config is not None
            or self.tp_worker.model_runner.mamba2_config is not None
        )

        self.sliding_window_size = None
        if self.is_hybrid_swa:
            self.sliding_window_size = self.tp_worker.sliding_window_size
            self.full_tokens_per_layer, self.swa_tokens_per_layer = (
                self.tp_worker.get_tokens_per_layer_info()
            )

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            self.tp_worker.get_memory_pool()
        )

        self.disable_radix_cache = server_args.disable_radix_cache or (
            self.model_config.is_multimodal and uses_transformers_backend
        )
        if self.disable_radix_cache and not server_args.disable_radix_cache:
            logger.warning(
                "Radix cache is disabled for multimodal models with the "
                "Transformers backend to avoid multimodal prefix-cache mismatches."
            )

        effective_chunked_prefill_size = server_args.chunked_prefill_size
        if self.model_config.is_multimodal and uses_transformers_backend:
            effective_chunked_prefill_size = None

        params = CacheInitParams(
            disable=self.disable_radix_cache,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            page_size=self.page_size,
            is_eagle=self.spec_algorithm.is_eagle(),
            tp_cache_group=(
                self.attn_tp_cpu_group
                if self.server_args.enable_dp_attention
                else self.tp_cpu_group
            ),
            eviction_policy=server_args.radix_eviction_policy,
            enable_metrics=self.enable_metrics,
            enable_kv_cache_events=self.enable_kv_cache_events,
            enable_mamba_extra_buffer=server_args.enable_mamba_extra_buffer(),
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            chunked_prefill_size=effective_chunked_prefill_size,
            sliding_window_size=self.sliding_window_size,
        )

        if effective_chunked_prefill_size is not None and self.disable_radix_cache:
            if not self.is_hybrid_swa:
                from sglang.srt.mem_cache.chunk_cache import ChunkCache

                self.tree_cache = ChunkCache(params)
            else:
                from sglang.srt.mem_cache.chunk_cache import SWAChunkCache

                self.tree_cache = SWAChunkCache(params)
        else:

            if envs.SGLANG_EXPERIMENTAL_CPP_RADIX_TREE.get():
                # 延迟导入以避免 JIT 开销
                from sglang.srt.mem_cache.radix_cache_cpp import RadixCacheCpp

                logger.info("Using experimental C++ radix tree implementation.")
                self.tree_cache = RadixCacheCpp(params=params, server_args=server_args)
            elif self.enable_hierarchical_cache:
                if self.is_hybrid_ssm:
                    from sglang.srt.mem_cache.hi_mamba_radix_cache import (
                        HiMambaRadixCache,
                    )

                    self.tree_cache = HiMambaRadixCache(
                        params=params, server_args=server_args
                    )
                else:
                    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

                    self.tree_cache = HiRadixCache(
                        params=params, server_args=server_args
                    )
                self.tp_worker.register_hicache_layer_transfer_counter(
                    self.tree_cache.cache_controller.layer_done_counter
                )
            elif self.is_hybrid_swa:
                from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache

                self.tree_cache = SWARadixCache(params=params)
            elif self.is_hybrid_ssm:
                from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache

                self.tree_cache = MambaRadixCache(params)
            elif server_args.enable_lmcache:
                from sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache import (
                    LMCRadixCache,
                )

                self.tree_cache = LMCRadixCache(
                    params=params,
                    model_config=self.model_config,
                    tp_size=self.tp_size,
                    rank=self.tp_rank,
                    tp_group=self.tp_group,
                )
            else:
                self.tree_cache = RadixCache(params)

        if server_args.enable_streaming_session:
            self.tree_cache = SessionAwareCache(self.tree_cache)

        if self.enable_hisparse:
            # coordinator 已在 ModelRunner.initialize() 中、CUDA graph 捕获之前创建
            self.hisparse_coordinator = self.tp_worker.model_runner.hisparse_coordinator
            self.hisparse_coordinator.set_decode_producer_stream(self.forward_stream)

        if (
            server_args.disaggregation_mode == "decode"
            and server_args.disaggregation_decode_enable_offload_kvcache
        ):
            self.decode_offload_manager = DecodeKVCacheOffloadManager(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                tp_group=params.tp_cache_group,
                tree_cache=self.tree_cache,
                server_args=self.server_args,
            )
        else:
            self.decode_offload_manager = None

        embedding_cache_size = envs.SGLANG_VLM_CACHE_SIZE_MB.get()
        init_mm_embedding_cache(embedding_cache_size * 1024 * 1024)

    def init_running_status(self):
        """初始化运行时状态：等待队列、运行批次、当前/上一批次、计数器与会话控制器等。"""
        self.waiting_queue: List[Req] = []
        # 用于连续批处理（continuous batching）的运行中 decode 批次
        self.running_batch: ScheduleBatch = ScheduleBatch(reqs=[], batch_is_full=False)
        # 当前前向计算的批次
        self.cur_batch: Optional[ScheduleBatch] = None
        # 上一个前向计算的批次
        self.last_batch: Optional[ScheduleBatch] = None
        self.forward_ct = 0
        self.return_health_check_ipcs: Deque[Optional[str]] = deque()
        self._pending_flush: Optional[Tuple[FlushCacheReqInput, float]] = None
        self.num_retracted_reqs: int = 0
        self.num_paused_reqs: int = 0
        self.session_controller = SessionController(self.tree_cache)
        self.forward_sleep_time = None
        self._engine_paused = False

    def init_chunked_prefill(self):
        """初始化分块预填充（chunked prefill）配置，并在 PP>1 时初始化动态分块预测器。"""
        self.chunked_prefill_size = self.server_args.chunked_prefill_size
        uses_transformers_backend = (
            get_resolved_model_impl(self.model_config) == ModelImpl.TRANSFORMERS
        )
        if (
            self.chunked_prefill_size is not None
            and self.chunked_prefill_size > 0
            and self.model_config.is_multimodal
            and uses_transformers_backend
        ):
            logger.warning(
                "Chunked prefill is disabled for multimodal models with the "
                "Transformers backend to avoid partial multimodal chunk mismatches."
            )
            self.chunked_prefill_size = None
        elif self.chunked_prefill_size is not None and self.chunked_prefill_size <= 0:
            self.chunked_prefill_size = None
        self.chunked_req = None
        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None
            and self.server_args.enable_mixed_chunk
        )

        # 为 PP 初始化动态分块（dynamic chunking）预测器
        self.enable_dynamic_chunking = (
            self.server_args.enable_dynamic_chunking and self.pp_size > 1
        )
        if self.enable_dynamic_chunking:
            try:
                self.profile_and_init_predictor()
            except Exception as e:
                logger.warning(
                    f"[PP Dynamic Chunk] Failed to profile prefill latency: {e}. "
                    "Dynamic chunking will be disabled."
                )
                self.enable_dynamic_chunking = False

    def init_schedule_policy(self):
        """初始化调度策略、（可选的）prefill 延迟器与优先级抢占，并设置新 token 比例的估计参数。"""
        # 初始化调度策略与新 token 数量估计
        self.policy = SchedulePolicy(
            self.schedule_policy,
            self.tree_cache,
            self.enable_hierarchical_cache,
            self.enable_priority_scheduling,
            self.schedule_low_priority_values_first,
        )
        self.prefill_delayer: Optional[PrefillDelayer] = None
        self.max_prefill_bs: int = 0
        if self.server_args.enable_prefill_delayer:
            self.prefill_delayer = PrefillDelayer(
                dp_size=self.dp_size,
                attn_tp_size=self.attn_tp_size,
                cpu_group=self.tp_cpu_group,
                server_args=self.server_args,
                metrics_collector=(
                    self.metrics_collector if self.enable_metrics else None
                ),
                max_delay_passes=self.server_args.prefill_delayer_max_delay_passes,
                token_usage_low_watermark=self.server_args.prefill_delayer_token_usage_low_watermark,
                device=(
                    self.tp_group.device
                    if self.server_args.disable_overlap_schedule
                    else "cpu"
                ),
            )

        # NOTE: 在优先级调度（priority scheduling）下，默认启用抢占（preemption）。
        self.enable_priority_preemption = (
            self.enable_priority_scheduling
            and not self.server_args.disable_priority_preemption
        )

        self.init_new_token_ratio = min(
            envs.SGLANG_INIT_NEW_TOKEN_RATIO.get()
            * self.server_args.schedule_conservativeness,
            1.0,
        )
        self.min_new_token_ratio = min(
            self.init_new_token_ratio * envs.SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR.get(),
            1.0,
        )
        self.new_token_ratio_decay = (
            self.init_new_token_ratio - self.min_new_token_ratio
        ) / envs.SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS.get()
        self.new_token_ratio = self.init_new_token_ratio

    def init_soft_watchdog(self, server_args: ServerArgs):
        """若配置了软看门狗超时，则创建软看门狗（超时仅告警、不强制终止）。"""
        if (x := server_args.soft_watchdog_timeout) is not None:
            self.soft_watchdog = create_scheduler_watchdog(
                self, watchdog_timeout=x, soft=True
            )

    def init_watch_dog_memory_saver_input_blocker(self):
        """初始化看门狗线程、显存节省器、接收跳过器与输入阻断器，并配置 GC 日志。"""
        # 启动看门狗（watchdog）线程
        self.watchdog = create_scheduler_watchdog(
            self, watchdog_timeout=self.server_args.watchdog_timeout
        )

        # 初始化显存节省器、profiler 与监控指标统计
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.server_args.enable_memory_saver
        )
        self.offload_tags = set()

        # 初始化接收跳过器（recv skipper）与输入阻断器（input blocker）
        self.recv_skipper = SchedulerRecvSkipper.maybe_create(self.server_args)
        self.input_blocker = (
            SchedulerInputBlocker(noop=self.attn_tp_rank != 0)
            if get_bool_env_var("SGLANG_ENABLE_COLOCATED_BATCH_GEN")
            else None
        )

        # 配置 GC 日志
        if envs.SGLANG_LOG_GC.get():
            configure_gc_logger()

    def init_disaggregation(self):
        """初始化 prefill-decode 分离部署（PD disaggregation）的模式、传输后端与相关队列。"""
        self.disaggregation_mode = DisaggregationMode(
            self.server_args.disaggregation_mode
        )
        self.transfer_backend = TransferBackend(
            self.server_args.disaggregation_transfer_backend
        )

        if self.draft_worker is None or self.spec_algorithm.is_ngram():
            draft_token_to_kv_pool = None
        elif self.spec_algorithm.supports_spec_v2() and self.enable_overlap:
            if self.server_args.enable_multi_layer_eagle:
                draft_runner = self.draft_worker.draft_worker.draft_runner_list[0]
            else:
                draft_runner = self.draft_worker.draft_worker.draft_runner
            draft_token_to_kv_pool = draft_runner.token_to_kv_pool
            model_config = draft_runner.model_config
        else:
            # todo: 启用 mtp 时是否需要修复这里？还是说无所谓——因为我们只在 decode 节点启用 mtp，所以不会在 P 和 D 之间传输 draft 的 KV？
            draft_token_to_kv_pool = self.draft_worker.model_runner.token_to_kv_pool
            model_config = self.draft_worker.model_config

        if (
            self.disaggregation_mode == DisaggregationMode.DECODE
        ):  # *2 留出余量。
            buffer_size = (self.req_to_token_pool.size) * 2
            self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                buffer_size
            )
            self.disagg_metadata_buffers = MetadataBuffers(
                buffer_size,
                hidden_size=(
                    model_config.hidden_size
                    if self.spec_algorithm.is_eagle()
                    else 16  # RDMA 的最小 padding 大小
                ),
                hidden_states_dtype=(
                    model_config.dtype
                    if self.spec_algorithm.is_eagle()
                    else torch.float32
                ),
                custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
            )

            # 正在轮询 KV cache 的 decode 请求队列
            self.disagg_decode_transfer_queue = DecodeTransferQueue(
                gloo_group=self.attn_tp_cpu_group,
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                tp_rank=self.tp_rank,
                metadata_buffers=self.disagg_metadata_buffers,
                scheduler=self,
                tree_cache=self.tree_cache,
            )

            # 等待预分配（pre-allocation）的 decode 请求队列
            self.disagg_decode_prealloc_queue = DecodePreallocQueue(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                draft_token_to_kv_pool=draft_token_to_kv_pool,
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                metadata_buffers=self.disagg_metadata_buffers,
                scheduler=self,
                transfer_queue=self.disagg_decode_transfer_queue,
                tree_cache=self.tree_cache,
                gloo_group=self.attn_tp_cpu_group,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                dp_size=self.server_args.dp_size,
                gpu_id=self.gpu_id,
                bootstrap_port=self.server_args.disaggregation_bootstrap_port,
                max_total_num_tokens=self.max_total_num_tokens,
                pp_rank=self.pp_rank,
                num_reserved_decode_tokens=self.server_args.num_reserved_decode_tokens,
                transfer_backend=self.transfer_backend,
            )

        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            # *2 留出余量。
            buffer_size = self.max_running_requests * 2
            self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                buffer_size
            )
            self.disagg_metadata_buffers = MetadataBuffers(
                buffer_size,
                hidden_size=(
                    model_config.hidden_size
                    if self.spec_algorithm.is_eagle()
                    or self.spec_algorithm.is_standalone()
                    else 16  # RDMA 的最小 padding 大小
                ),
                hidden_states_dtype=(
                    model_config.dtype
                    if self.spec_algorithm.is_eagle()
                    or self.spec_algorithm.is_standalone()
                    else torch.float32
                ),
                custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
            )

            self.disagg_prefill_bootstrap_queue = PrefillBootstrapQueue(
                token_to_kv_pool=self.token_to_kv_pool_allocator.get_kvcache(),
                draft_token_to_kv_pool=draft_token_to_kv_pool,
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                metadata_buffers=self.disagg_metadata_buffers,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                gpu_id=self.gpu_id,
                bootstrap_port=self.server_args.disaggregation_bootstrap_port,
                gloo_group=self.attn_tp_cpu_group,
                max_total_num_tokens=self.max_total_num_tokens,
                scheduler=self,
                pp_rank=self.pp_rank,
                pp_size=self.pp_size,
                transfer_backend=self.transfer_backend,
            )
            # 正在进行 KV 发送过程中的 prefill 请求队列
            self.disagg_prefill_inflight_queue: List[Req] = []

        # 为 EPD 分离部署模式初始化 mm（多模态）接收器
        if (
            self.server_args.language_only
            and self.server_args.encoder_transfer_backend == "zmq_to_scheduler"
        ):
            self.mm_receiver = create_mm_receiver(
                self.server_args,
                hf_config=self.model_config.hf_config,
                pp_rank=self.pp_rank,
                tp_rank=self.tp_rank,
                tp_group=self.tp_group,
                scheduler=self,
            )

    def init_overlap(self):
        """初始化 overlap 调度所需的计算流与拷贝流；若启用 overlap，还会创建 FutureMap 等结构。"""
        self.device_module = torch.get_device_module(self.device)

        self.forward_stream_ctx: CudaStreamContext = self.device_module.stream(
            self.forward_stream
        )
        self.copy_stream: CudaStream = self.device_module.Stream()
        self.copy_stream_ctx: CudaStreamContext = self.device_module.stream(
            self.copy_stream
        )

        if not self.enable_overlap:
            self.future_map = None
            return

        self.future_map = FutureMap(
            self.max_running_requests,
            self.chunked_prefill_size,
            self.model_config.context_len,
            self.device,
            self.spec_algorithm,
        )
        self.batch_record_buf = [None] * 2
        self.batch_record_ct = 0

    def maybe_init_ngram_embedding(self):
        """若模型使用 N-gram embedding，则初始化其 token 表与 n/k 等参数。"""
        self.use_ngram_embedding = self.tp_worker.model_config.use_ngram_embedding
        if self.use_ngram_embedding:
            self.token_table = self.tp_worker.model_runner.token_table
            hf_config = self.tp_worker.model_config.hf_config
            self.ngram_embedding_n = hf_config.ngram_embedding_n
            self.ngram_embedding_k = hf_config.ngram_embedding_k

    def _maybe_prepare_ngram_embedding(
        self, batch: Optional[ScheduleBatch]
    ) -> Optional[ScheduleBatch]:
        """在一次前向计算之前，为 ngram embedding 填充 token 表。"""
        if batch is None or not self.use_ngram_embedding:
            return batch
        batch.ne_token_table = self.token_table
        if batch.forward_mode == ForwardMode.EXTEND:
            all_tokens = []
            column_starts = []
            request_lengths = []
            for req in batch.reqs:
                start = len(req.prefix_indices)
                end = start + req.extend_input_len
                fill_ids = req.origin_input_ids + req.output_ids
                if start == 0:
                    tokens = fill_ids[start:end]
                    column_starts.append(0)
                elif start < self.ngram_embedding_n:
                    tokens = fill_ids[0:end]
                    column_starts.append(0)
                else:
                    # 在 prefix_len 之前补上 n-1 个 token，作为 n-gram 上下文
                    tokens = fill_ids[start - self.ngram_embedding_n + 1 : end]
                    column_starts.append(start - self.ngram_embedding_n + 1)
                all_tokens.extend(tokens)
                request_lengths.append(len(tokens))
            dtype = self.token_table.dtype
            device = self.token_table.device
            update_token_table(
                ne_token_table=self.token_table,
                tokens=torch.tensor(all_tokens, dtype=dtype, device=device),
                row_indices=batch.req_pool_indices,
                column_starts=torch.tensor(
                    column_starts, dtype=torch.int32, device=device
                ),
                req_lens=torch.tensor(
                    request_lengths, dtype=torch.int32, device=device
                ),
                ignore_tokens=None,
            )
        return batch

    def init_deterministic_inference_config(self):
        """为不同的 attention 后端初始化确定性推理（deterministic inference）配置。"""
        if not self.server_args.enable_deterministic_inference:
            self.truncation_align_size = None
            return

        backend_sizes = {
            "flashinfer": ("SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE", 4096),
            "triton": ("SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE", 4096),
        }
        env_var, default_size = backend_sizes.get(
            self.server_args.attention_backend, (None, None)
        )
        self.truncation_align_size = (
            get_int_env_var(env_var, default_size) if env_var else None
        )

    def init_request_dispatcher(self):
        """初始化基于请求类型的分发器，将各类输入请求映射到对应的处理函数。"""
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.handle_generate_request),
                (TokenizedEmbeddingReqInput, self.handle_embedding_request),
                (BatchTokenizedGenerateReqInput, self.handle_batch_generate_request),
                (BatchTokenizedEmbeddingReqInput, self.handle_batch_embedding_request),
                (FlushCacheReqInput, self.flush_cache_wrapped),
                (ClearHiCacheReqInput, self.clear_hicache_storage_wrapped),
                (AttachHiCacheStorageReqInput, self.attach_hicache_storage_wrapped),
                (DetachHiCacheStorageReqInput, self.detach_hicache_storage_wrapped),
                (AbortReq, self.abort_request),
                (OpenSessionReqInput, self.open_session),
                (CloseSessionReqInput, self.close_session),
                (UpdateWeightFromDiskReqInput, self.update_weights_from_disk),
                (InitWeightsUpdateGroupReqInput, self.init_weights_update_group),
                (DestroyWeightsUpdateGroupReqInput, self.destroy_weights_update_group),
                (
                    InitWeightsSendGroupForRemoteInstanceReqInput,
                    self.init_weights_send_group_for_remote_instance,
                ),
                (
                    SendWeightsToRemoteInstanceReqInput,
                    self.send_weights_to_remote_instance,
                ),
                (
                    UpdateWeightsFromDistributedReqInput,
                    self.update_weights_from_distributed,
                ),
                (UpdateWeightsFromTensorReqInput, self.update_weights_from_tensor),
                (UpdateWeightsFromIPCReqInput, self.update_weights_from_ipc),
                (GetWeightsByNameReqInput, self.get_weights_by_name),
                (ReleaseMemoryOccupationReqInput, self.release_memory_occupation),
                (ResumeMemoryOccupationReqInput, self.resume_memory_occupation),
                (CheckWeightsReqInput, self.check_weights),
                (SlowDownReqInput, self.slow_down),
                (ProfileReq, self.profile),
                (FreezeGCReq, self.handle_freeze_gc),
                (GetInternalStateReq, self.get_internal_state),
                (SetInternalStateReq, self.set_internal_state),
                (RpcReqInput, self.handle_rpc_request),
                (ExpertDistributionReq, self.expert_distribution_handle),
                (LoadLoRAAdapterReqInput, self.load_lora_adapter),
                (
                    LoadLoRAAdapterFromTensorsReqInput,
                    self.load_lora_adapter_from_tensors,
                ),
                (UnloadLoRAAdapterReqInput, self.unload_lora_adapter),
                (GetLoadReqInput, self.get_load),
                (GetLoadsReqInput, self.get_loads),
                (PauseGenerationReqInput, self.pause_generation),
                (ContinueGenerationReqInput, self.continue_generation),
                (DumperControlReqInput, self.handle_dumper_control),
            ]
        )

    def _abort_on_running_timeout(self):
        """检查运行中请求是否超时，超时则标记为中止（需在启动批次前调用）。"""
        # NOTE: 这应在启动一个批次之前调用，
        # 因为当前的 spec-v1 仍会在 verify 阶段内部对批次进行过滤。
        timeout_s = envs.SGLANG_REQ_RUNNING_TIMEOUT.get()
        if timeout_s <= 0:
            return
        if self.running_batch.is_empty():
            return

        deadline = time.perf_counter() - timeout_s
        for req in self.running_batch.reqs:
            if not req.finished() and 0 < req.time_stats.forward_entry_time < deadline:
                req.to_finish = FINISH_ABORT(
                    "Request running timeout reached.", HTTPStatus.SERVICE_UNAVAILABLE
                )

    def get_init_info(self) -> Dict[str, Any]:
        """返回用于握手（handshake）的调度器初始化信息。

        该方法提供 tokenizer manager 及其他组件用于确认调度器已就绪所需的初始化信息。
        """
        result_dict = {
            "status": "ready",
            "max_total_num_tokens": self.max_total_num_tokens,
            "max_req_input_len": self.max_req_input_len,
        }

        return result_dict

    def run_event_loop(self) -> None:
        """运行调度器的事件循环。

        创建调度流（schedule stream），并分发到相应的事件循环。
        事件循环会一直阻塞，直到进程关闭。
        """
        # 调度器事件循环的总入口：
        # 创建专用的调度流（schedule stream），并根据部署模式分发到具体的事件循环。
        # 事件循环会一直阻塞运行，直到进程关闭。
        self.schedule_stream = self.device_module.Stream(priority=0)
        if self.device == "cpu":
            self.schedule_stream.synchronize = lambda: None  # CPU 上同步为空操作
        with self.device_module.StreamContext(self.schedule_stream):
            dispatch_event_loop(self)

    @DynamicGradMode()
    def event_loop_normal(self):
        """一个普通的调度器循环。"""
        # 普通（非 overlap）调度循环：CPU 调度与 GPU 计算串行执行。
        while True:
            # 1) 接收请求并处理输入（入队、预处理等）
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                # 引擎被暂停时，取消气泡计时并跳过本轮调度。
                self.cancel_bubble_timer()
                continue

            # 2) 获取下一个要运行的批次（优先 prefill，否则 decode）
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            # 3) 运行当前批次并处理输出
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # 服务空闲时，执行自检并重置部分状态。
                self.self_check_during_idle()

            # 4) 更新 last_batch，供下一轮调度合并 prefill/decode 批次使用
            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                # 启用严格显存检查时，在繁忙阶段也做一次自检。
                self.self_check_during_busy()

    @DynamicGradMode()
    def event_loop_overlap(self):
        """一个将 CPU 处理与 GPU 计算重叠（overlap）的调度器循环。"""
        # overlap 调度循环：将当前批次的 CPU 调度与上一批次的 GPU 计算重叠，以提升吞吐。
        # result_queue 缓存已启动但尚未处理输出的批次及其结果。
        self.result_queue: Deque[
            Tuple[ScheduleBatch, Union[GenerationBatchResult, EmbeddingBatchResult]]
        ] = deque()

        def pop_and_process():
            # 处理上一（最早入队）批次的计算结果。
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        while True:
            # 1) 接收请求并处理输入
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue

            # 2) 获取下一个要运行的批次，并判断是否需要禁用本批次的 overlap
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch
            disable_overlap_for_batch = self.is_disable_overlap_for_batch(batch)

            # 若无需将当前批次与上一批次重叠，则可立即处理上一批次的结果。
            if disable_overlap_for_batch:
                pop_and_process()

            # 3) 启动当前批次的前向计算，将（批次副本, 结果）压入队列等待后续处理
            if batch:
                batch_result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None
                self.cancel_bubble_timer()

            # 4) 在 GPU 计算当前批次的同时，处理上一批次的结果（实现重叠）
            if self.last_batch:
                if not disable_overlap_for_batch:
                    pop_and_process()
            elif batch is None:
                # 服务空闲时执行自检并重置部分状态。
                self.self_check_during_idle()

            # 5) 运行当前批次的采样：
            # 由于采样可能依赖上一批次的结果（如 grammar 状态），故需在上一批次处理完后再执行。
            if self.is_generation:
                self.launch_batch_sample_if_needed(batch_result)

            # 6) 更新 last_batch
            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.self_check_during_busy()

    def is_disable_overlap_for_batch(self, batch: ScheduleBatch) -> bool:
        """判断是否需要对本批次禁用 overlap。

        主要场景：连续两个 prefill 批次时禁用 overlap 以改善首 token 延迟（TTFT）；
        以及尚不支持 overlap + 投机解码 + grammar 的组合。在 DP attention 下使用全局同步的
        is_extend_in_batch 以保证各 DP rank 决策一致（避免死锁）。
        """
        # 对于连续两个 prefill 批次，我们禁用 overlap 以改善第一个批次的 TTFT。
        # 这可能会略微损害吞吐，因此用一个环境变量来控制该行为。
        # 在 DP attention 模式下，使用全局同步的 is_extend_in_batch，
        # 以保证所有 DP rank 做出相同的 overlap 决策（避免死锁）。
        # 在非 DP 模式下，直接使用本地的 forward_mode。
        if self.require_mlp_sync:
            is_extend = lambda b: b and b.is_extend_in_batch
        else:
            is_extend = lambda b: b and b.forward_mode.is_extend()

        batch_is_extend = is_extend(batch)
        last_batch_is_extend = is_extend(self.last_batch)

        disable_overlap_for_batch = (
            envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.get()
            and batch_is_extend
            and last_batch_is_extend
        )

        # 我们尚不支持 overlap + spec + grammar 的组合，
        # 因此需要对本批次关闭 overlap。
        # TODO(lsyin): 支持 overlap + spec + grammar
        need_grammar_sync = (
            batch
            and batch.is_spec_v2
            and batch.has_grammar
            and batch.forward_mode.is_decode()
            and len(self.result_queue) > 0
        )

        return disable_overlap_for_batch or need_grammar_sync

    def recv_limit_reached(self, num_recv_reqs: int) -> bool:
        """判断本次轮询接收的请求数是否达到上限（max_recv_per_poll，<0 表示不限制）。"""
        if self.max_recv_per_poll < 0:
            return False
        return num_recv_reqs >= self.max_recv_per_poll

    def recv_requests(
        self,
    ) -> List[Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput, Any]]:
        """在 tp_rank = 0 处接收结果，并广播给所有其他 TP rank。"""
        # 仅在入口 rank 上从 ZMQ 接收请求与 RPC，然后通过广播分发给其他 TP/PP rank；
        # 同时处理 DP attention 下的工作请求/控制请求拆分、EPD 分离的多模态接收以及共享内存特征的解包。

        if self.recv_skipper is not None:
            last_forward_mode = (
                self.last_batch.forward_mode if self.last_batch is not None else None
            )
            if not self.recv_skipper.handle(last_forward_mode):
                return []

        if self.pp_rank == 0:
            if self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
                recv_reqs = []

                while True:
                    try:
                        if self.recv_limit_reached(len(recv_reqs)):
                            break
                        recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_req)

                while True:
                    try:
                        if self.recv_limit_reached(len(recv_reqs)):
                            break
                        recv_rpc = self.recv_from_rpc.recv_pyobj(zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_rpc)
            else:
                recv_reqs = None
        else:
            if self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
                dp_offset = self.attn_dp_rank * self.attn_tp_size
                recv_reqs = point_to_point_pyobj(
                    [],
                    self.pp_rank * self.tp_size + dp_offset,
                    self.world_group.cpu_group,
                    (self.pp_rank - 1) * self.tp_size + dp_offset,
                    self.pp_rank * self.tp_size + dp_offset,
                )
            else:
                recv_reqs = None

        if self.input_blocker is not None:
            recv_reqs = self.input_blocker.handle(recv_reqs)

        if self.server_args.enable_dp_attention:
            if self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
                work_reqs, control_reqs = self._split_work_and_control_reqs(recv_reqs)
            else:
                work_reqs = None
                control_reqs = None

            if self.attn_tp_size != 1:
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.attn_tp_group.rank,
                    self.attn_tp_cpu_group,
                    src=self.attn_tp_group.ranks[0],
                )

            if self.attn_cp_size != 1:
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.attn_cp_group.rank,
                    self.attn_cp_cpu_group,
                    src=self.attn_cp_group.ranks[0],
                )

            if self.tp_size != 1:
                control_reqs = broadcast_pyobj(
                    control_reqs,
                    self.tp_group.rank,
                    self.tp_cpu_group,
                    src=self.tp_group.ranks[0],
                )
            recv_reqs = work_reqs + control_reqs
        elif self.tp_size != 1:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )

        # 在 EPD 分离部署模式下处理多模态（MM）请求
        if (
            self.pp_rank == 0
            and self.server_args.language_only
            and self.server_args.encoder_transfer_backend == "zmq_to_scheduler"
        ):
            recv_reqs, abort_reqs = self.mm_receiver.process_waiting_requests(recv_reqs)
            for req, error_msg, error_code in abort_reqs:

                status_code = (
                    HTTPStatus.BAD_REQUEST
                    if error_code == 400
                    else HTTPStatus.INTERNAL_SERVER_ERROR
                )
                prepare_abort(req, error_msg, status_code=status_code)
                self.stream_output([req], req.return_logprob)

        # 在所有广播完成之后再解包共享内存特征，
        # 这样在 broadcast_pyobj 期间被序列化的只是 ShmPointerMMData 的元数据，
        # 而非完整的张量数据。
        if recv_reqs:
            # 仅在非 DP-attention 路径上需要 barrier：tp_cpu_group 上只有一次
            # broadcast_pyobj，源 rank 会立即返回原始对象，而其他 rank 仍处于
            # pickle.loads（-> __setstate__ -> shm_open）中。如果没有 barrier，
            # 源 rank 可能在其他 rank 打开该段之前就调用 materialize() / shm_unlink。
            # 此处的 recv_reqs 在所有 rank 上是一致的（同一次广播），因此该保护不会死锁。
            #
            # 在 DP-attention 下则无需 barrier：tp_cpu_group 上的 control_reqs
            # 广播（步骤 3）是一个集合通信，它会强制每个 rank 在从步骤 3 返回之前
            # 先完成此前的 attn_tp / attn_cp work_reqs 反序列化（步骤 1-2，它们会调用
            # shm_open）。POSIX 保证 shm_unlink 只移除名字，已打开的句柄仍然有效。
            if (
                not self.server_args.enable_dp_attention
                and self.tp_size > 1
                and self.model_config.is_multimodal
                and has_shm_features(recv_reqs)
            ):
                barrier(group=self.tp_cpu_group)
            for req in recv_reqs:
                unwrap_shm_features(req)

        return recv_reqs

    def _split_work_and_control_reqs(self, recv_reqs: List):
        """将接收到的请求拆分为工作请求（生成/embedding）与控制请求（其他管理类）两类。

        用于 DP attention 下区分广播范围：工作请求在 attn_tp 组广播，控制请求在全 tp 组广播。
        """
        work_reqs = [
            req
            for req in recv_reqs
            if isinstance(
                req,
                (
                    TokenizedGenerateReqInput,
                    TokenizedEmbeddingReqInput,
                    BatchTokenizedGenerateReqInput,
                    BatchTokenizedEmbeddingReqInput,
                ),
            )
        ]
        control_reqs = [
            req
            for req in recv_reqs
            if not isinstance(
                req,
                (
                    TokenizedGenerateReqInput,
                    TokenizedEmbeddingReqInput,
                    BatchTokenizedGenerateReqInput,
                    BatchTokenizedEmbeddingReqInput,
                ),
            )
        ]
        return work_reqs, control_reqs

    def process_input_requests(self, recv_reqs: List):
        """逐个处理接收到的请求：跳过繁忙时的健康检查，经分发器路由到对应处理函数，并回传输出。"""
        now = time.monotonic()
        self.session_controller.maybe_reap(now)
        for recv_req in recv_reqs:
            # 服务繁忙时跳过健康检查——正在处理的请求本身已携带健康信息。
            if is_health_check_generate_req(recv_req) and not self.is_fully_idle(
                for_health_check=True
            ):
                self.return_health_check_ipcs.append(
                    getattr(recv_req, "http_worker_ipc", None)
                )
                continue

            output = self._request_dispatcher(recv_req)
            if output is not None:
                if not isinstance(output, RpcReqOutput):
                    self.send_to_tokenizer.send_output(output, recv_req)
                else:
                    if self.recv_from_rpc is not None:
                        self.recv_from_rpc.send_pyobj(output)

        self._check_pending_flush()

    def init_req_max_new_tokens(self, req):
        """根据模型最大长度与输入长度，限制/修正请求的 max_new_tokens。"""
        req.sampling_params.max_new_tokens = min(
            (
                req.sampling_params.max_new_tokens
                if req.sampling_params.max_new_tokens is not None
                else 1 << 30
            ),
            self.max_req_len - len(req.origin_input_ids) - 1,
        )

    def _process_and_broadcast_mm_inputs(
        self,
        raw_mm_inputs: Optional[dict],
    ):
        """在入口 rank 上一次性物化 MultimodalInputs，并广播给其他 rank。

        入口 rank：
        - 只调用一次 MultimodalInputs.from_dict(raw_mm_inputs) 构造对象
        - 广播给 self.cpu_group 中的其他 rank（当 world_size > 1 时）

        非入口 rank：
        - 当 world_size > 1 时，通过广播接收该对象
        - 否则（单 rank / 无通信组）回退到本地 from_dict 构造

        返回：
            MultimodalInputs | None
        """
        if raw_mm_inputs is None:
            return None

        group_world_size = 1
        try:
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and self.dp_tp_cpu_group is not None
            ):
                group_world_size = torch.distributed.get_world_size(
                    group=self.dp_tp_cpu_group
                )
        except Exception as e:
            logger.warning(
                f"Failed to get world size in mm_inputs handling with {e}, fallback to 1."
            )

        # 当 tp size > 1 时，所有 Scheduler 的 TP rank 都会在 CPU 上重复执行同样的计算，
        # 占用主线程的 CPU 时间。而这段计算逻辑其实只需要在 TP0 上运行一次，再广播给其他 TP rank。
        # 由于 Scheduler 是单线程的，任何较大的 CPU 开销都会影响其他消息的处理。
        # 例如，CPU 占用打到 99.9% 会显著增加 CUDA kernel 的启动时间。
        if self.dp_tp_group.rank_in_group == 0:
            # 只有入口 rank 从 dict 物化一次。
            image_inputs = MultimodalInputs.from_dict(raw_mm_inputs)
            # 广播给其他 TP rank（在组内使用 src=0）。
            if group_world_size > 1:
                obj_list = [image_inputs]
                torch.distributed.broadcast_object_list(
                    obj_list,
                    src=self.dp_tp_group.first_rank,
                    group=self.dp_tp_cpu_group,
                )
                image_inputs = obj_list[0]
        else:
            # 非入口 rank：组规模 > 1 时通过广播接收；否则在本地物化。
            if group_world_size > 1:
                obj_list = [None]
                torch.distributed.broadcast_object_list(
                    obj_list,
                    src=self.dp_tp_group.first_rank,
                    group=self.dp_tp_cpu_group,
                )
                image_inputs = obj_list[0]
            else:
                image_inputs = MultimodalInputs.from_dict(raw_mm_inputs)

        return image_inputs

    def _get_multimodal_inputs(self, mm_inputs_dict: dict):
        """获取多模态输入：根据配置选择在入口 rank 物化并广播，或直接本地 from_dict 构造。"""
        if self.server_args.enable_broadcast_mm_inputs_process:
            return self._process_and_broadcast_mm_inputs(mm_inputs_dict)
        else:
            return MultimodalInputs.from_dict(mm_inputs_dict)

    def _maybe_compute_mrope_positions(self, req) -> None:
        """当 M-RoPE 位置缺失时（例如 gRPC 预处理路径）计算它们。"""
        if self._mm_processor is None:
            return
        mm = req.multimodal_inputs
        if mm is None or mm.mrope_positions is not None:
            return

        mrope_positions, mrope_position_delta = (
            self._mm_processor.compute_mrope_positions(
                req.origin_input_ids, mm.mm_items
            )
        )
        if mrope_positions is not None:
            mm.mrope_positions = mrope_positions
            mm.mrope_position_delta = mrope_position_delta

    def _maybe_clear_mm_inputs(self, batch: ScheduleBatch) -> None:
        """对已完成且非会话的请求，释放其多模态特征与输入以释放显存。"""
        for req in batch.reqs:
            if not req.finished() or not (mm_inputs := req.multimodal_inputs):
                continue
            # 对于会话（session）请求，保留 mm_inputs 供下一个请求使用
            if req.session:
                continue
            # 对于非会话请求，清除特征与 mm_inputs
            mm_inputs.release_features()
            req.multimodal_inputs = None

    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        """处理生成请求：构造 Req 对象，处理会话/多模态输入/PD 分离校验/长度校验/logprob 起始位置，
        最终将请求加入 grammar 队列或等待队列。"""
        # 路由：普通请求 / 会话请求 / 会话不存在
        session_id = (
            recv_req.session_params.id if recv_req.session_params is not None else None
        )

        if session_id is None:
            # 普通的非会话请求
            if recv_req.input_embeds is not None:
                # 根据 input_embeds 的长度生成占位的 input_ids
                seq_length = len(recv_req.input_embeds)
                fake_input_ids = [1] * seq_length
                recv_req.input_ids = fake_input_ids

            if recv_req.bootstrap_port is None:
                # 使用默认的 bootstrap 端口
                recv_req.bootstrap_port = self.server_args.disaggregation_bootstrap_port

            req = Req(
                recv_req.rid,
                recv_req.input_text,
                recv_req.input_ids,
                recv_req.sampling_params,
                return_logprob=recv_req.return_logprob,
                top_logprobs_num=recv_req.top_logprobs_num,
                token_ids_logprob=recv_req.token_ids_logprob,
                stream=recv_req.stream,
                lora_id=recv_req.lora_id,
                input_embeds=recv_req.input_embeds,
                token_type_ids=recv_req.token_type_ids,
                custom_logit_processor=recv_req.custom_logit_processor,
                require_reasoning=recv_req.require_reasoning,
                return_hidden_states=recv_req.return_hidden_states,
                return_routed_experts=recv_req.return_routed_experts,
                eos_token_ids=self.model_config.hf_eos_token_id,
                bootstrap_host=recv_req.bootstrap_host,
                bootstrap_port=recv_req.bootstrap_port,
                bootstrap_room=recv_req.bootstrap_room,
                disagg_mode=self.disaggregation_mode,
                routed_dp_rank=recv_req.routed_dp_rank,
                disagg_prefill_dp_rank=recv_req.disagg_prefill_dp_rank,
                vocab_size=self.model_config.vocab_size,
                priority=recv_req.priority,
                metrics_collector=(
                    self.metrics_collector if self.enable_metrics else None
                ),
                routing_key=recv_req.routing_key,
                http_worker_ipc=recv_req.http_worker_ipc,
                dllm_config=self.dllm_config,
                time_stats=recv_req.time_stats,
            )
            req.tokenizer = self.tokenizer

            if self.disaggregation_mode != DisaggregationMode.NULL:
                # 分离部署模式下的非法请求
                if (
                    recv_req.bootstrap_room is None
                    and self.transfer_backend != TransferBackend.FAKE
                ):
                    error_msg = (
                        f"Invalid request: Disaggregated request received without "
                        f"bootstrap room id. {req.rid=}"
                    )
                    logger.error(error_msg)
                    recv_req.time_stats.trace_ctx.abort(
                        abort_info={"reason": error_msg}
                    )
                    prepare_abort(req, error_msg, status_code=HTTPStatus.BAD_REQUEST)
                    self.stream_output([req], req.return_logprob)
                    return

        elif session_id in self.session_controller:
            # 会话存在：从会话创建请求
            session = self.session_controller.get(session_id)
            req = session.create_req(
                recv_req,
                self.tokenizer,
                self.model_config.vocab_size,
                eos_token_ids=self.model_config.hf_eos_token_id,
            )
            # TODO: 设置 trace 上下文
            if self.enable_metrics:
                req.time_stats.set_metrics_collector(self.metrics_collector)
            if isinstance(req.finished_reason, FINISH_ABORT):
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

        else:
            # 提供了 session ID，但找不到对应的会话
            req = Req(
                recv_req.rid,
                recv_req.input_text,
                recv_req.input_ids,
                recv_req.sampling_params,
                vocab_size=self.model_config.vocab_size,
            )
            req.tokenizer = self.tokenizer
            req.set_finish_with_abort(
                f"Invalid request: session id {session_id} does not exist"
            )
            self.init_req_max_new_tokens(req)
            self._add_request_to_queue(req)
            return

        # 处理多模态输入
        if recv_req.mm_inputs is not None:
            image_inputs = self._get_multimodal_inputs(recv_req.mm_inputs)

            SessionController.adjust_mm_offsets(recv_req, req, image_inputs)

            # 以下步骤本身已经很快，在每个 rank 上本地执行即可。
            # 将单个 image token 扩展为多个占位 token，用于接收图像 embedding。
            # pad 函数与具体模型相关，对某些后端可能为 None。
            if self.pad_input_ids_func:
                req.origin_input_ids = self.pad_input_ids_func(
                    req.origin_input_ids, image_inputs
                )
            req.extend_image_inputs(image_inputs)
            self._maybe_compute_mrope_positions(req)

            if len(req.origin_input_ids) >= self.max_req_input_len:
                req.set_finish_with_abort(
                    error_msg=(
                        "Multimodal prompt is too long after expanding multimodal tokens. "
                        f"After expanding {len(req.origin_input_ids_unpadded)=} => {len(req.origin_input_ids)} >= {self.max_req_input_len}."
                    )
                )
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

        # 返回前先初始化
        self.init_req_max_new_tokens(req)

        # 校验 prompt 长度
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            self.server_args.allow_auto_truncate,
        )
        if error_msg:
            req.set_finish_with_abort(error_msg)
            self._add_request_to_queue(req)
            return

        if not recv_req.return_logprob and recv_req.logprob_start_len != -1:
            # 当 return_logprob 为 False 时，应忽略 logprob_start_len
            recv_req.logprob_start_len = -1

        if recv_req.logprob_start_len == -1:
            if recv_req.return_logprob and recv_req.token_ids_logprob is None:
                # 如果需要 logprob，但既没有设置 token_ids_logprob 也没有设置 logprob_start_len，
                # 则默认返回输出 token 的 logprob
                req.logprob_start_len = len(req.origin_input_ids)
            elif req.is_prefill_only:
                # 对于 logprob_start_len == -1 的 prefill-only 请求，将 logprob_start_len
                # 设置到输入序列之外，从而完全跳过输入 logprob 的计算
                req.logprob_start_len = len(req.origin_input_ids)
            else:
                # 如果 return_logprob 为 False，则只有最后一个 token 需要计算 logprob
                req.logprob_start_len = -1
        else:
            req.logprob_start_len = recv_req.logprob_start_len

        if req.logprob_start_len > len(req.origin_input_ids):
            error_msg = f"{req.logprob_start_len=} is higher than the number of input tokens {len(req.origin_input_ids)=}. Please use a smaller logprob_start_len."
            req.logprob_start_len = -1
            req.set_finish_with_abort(error_msg)
            self._add_request_to_queue(req)
            return

        added_to_grammar_queue = self.grammar_manager.process_req_with_grammar(req)
        if not added_to_grammar_queue:
            self._add_request_to_queue(req)

    def handle_batch_generate_request(
        self,
        recv_req: BatchTokenizedGenerateReqInput,
    ):
        """处理经过优化的批量生成请求。"""
        logger.debug(f"Processing batch generate request with {len(recv_req)} requests")

        # 逐个处理批次中的每个请求
        for tokenized_req in recv_req:
            self.handle_generate_request(tokenized_req)

    def _prefetch_kvcache(self, req: Req):
        """若启用分层缓存存储，则针对请求的新输入 token 从存储后端预取 KV 缓存。"""
        if self.enable_hicache_storage:
            req.init_next_round_input(self.tree_cache, cow_mamba=False)
            last_host_node = req.last_host_node
            if last_host_node.backuped or last_host_node is self.tree_cache.root_node:
                last_hash = last_host_node.get_last_hash_value()
                matched_len = len(req.prefix_indices) + req.host_hit_length
                new_input_tokens = req.fill_ids[matched_len:]

                prefix_keys = (
                    last_host_node.get_prefix_hash_values(last_host_node.parent)
                    if self.tree_cache.hicache_storage_pass_prefix_keys
                    else None
                )
                self.tree_cache.prefetch_from_storage(
                    req.rid,
                    last_host_node,
                    new_input_tokens,
                    last_hash,
                    prefix_keys,
                )

    def _add_request_to_queue(self, req: Req, is_retracted: bool = False):
        """根据部署模式（普通/PD prefill/PD decode）将请求加入相应的队列，并记录时间统计。"""
        if self.disaggregation_mode == DisaggregationMode.NULL:
            if not self._set_or_validate_priority(req):
                return
            if self._abort_on_queued_limit(req):
                return
            self._prefetch_kvcache(req)
            self.waiting_queue.append(req)
            req.time_stats.set_wait_queue_entry_time()
        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._prefetch_kvcache(req)
            self.disagg_prefill_bootstrap_queue.add(
                req, self.model_config.num_key_value_heads
            )
            req.time_stats.set_prefill_bootstrap_queue_entry_time()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.disagg_decode_prealloc_queue.add(req, is_retracted=is_retracted)
            if not is_retracted:
                req.time_stats.set_decode_prealloc_queue_entry_time()
            else:
                req.time_stats.set_retract_time()
        else:
            raise ValueError(f"Invalid {self.disaggregation_mode=}")

    def _set_or_validate_priority(self, req: Req) -> bool:
        """设置默认优先级值，或根据优先级调度模式中止该请求。"""
        if self.enable_priority_scheduling and req.priority is None:
            if self.schedule_low_priority_values_first:
                req.priority = sys.maxsize
            else:
                req.priority = -sys.maxsize - 1
        elif (
            not self.enable_priority_scheduling
            and req.priority is not None
            and self.abort_on_priority_when_disabled
        ):
            abort_req = AbortReq(
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                    "message": "Using priority is disabled for this server. Please send a new request without a priority.",
                },
                rid=req.rid,
            )
            req.time_stats.trace_ctx.abort(abort_info=abort_req.finished_reason)
            self.send_to_tokenizer.send_output(abort_req, req)
            return False
        return True

    def _abort_on_queued_limit(self, recv_req: Req) -> bool:
        """当等待队列已满时，中止新进入的请求或某个已有请求。若新进入的请求被中止则返回 True。"""
        if (
            self.max_queued_requests is None
            or len(self.waiting_queue) + 1 <= self.max_queued_requests
        ):
            return False

        # 默认拒绝新进入的请求。
        req_to_abort = recv_req
        message = "The request queue is full."
        if self.enable_priority_scheduling:
            # 启用优先级调度时，考虑根据优先级中止某个已有请求。
            # direction = 1  => 数值越小优先级越高；-1 => 数值越大优先级越高。
            # max(...) 以 (direction * priority, queue_time_start) 为键，挑出最不被偏好的请求。
            # 平局时：queue_time_start 较晚（较新）的先被驱逐。仅当严格更优时才抢占。
            direction = 1 if self.schedule_low_priority_values_first else -1
            key_fn = lambda item: (
                direction * item[1].priority,
                item[1].time_stats.wait_queue_entry_time,
            )
            idx, candidate_req = max(enumerate(self.waiting_queue), key=key_fn)
            abort_existing_req = (
                direction * recv_req.priority < direction * candidate_req.priority
            )
            if abort_existing_req:
                if self.enable_hicache_storage:
                    # 释放与该请求关联的预取（prefetch）事件
                    self.tree_cache.release_aborted_request(candidate_req.rid)
                elif self.enable_hierarchical_cache:
                    self.tree_cache.terminate_prefetch(candidate_req.rid)
                self.waiting_queue.pop(idx)
                req_to_abort = candidate_req
                message = "The request is aborted by a higher priority request."

        self.send_to_tokenizer.send_output(
            AbortReq(
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                    "message": message,
                },
                rid=req_to_abort.rid,
            ),
            req_to_abort,
        )
        req_to_abort.time_stats.trace_ctx.abort(abort_info={"reason": message})
        return req_to_abort.rid == recv_req.rid

    def _abort_on_waiting_timeout(self):
        """检查等待队列中超时的请求并中止它们（根据等待超时阈值）。"""
        if (timeout_s := envs.SGLANG_REQ_WAITING_TIMEOUT.get()) <= 0:
            return

        deleted_reqs = set()
        deadline = time.perf_counter() - timeout_s
        for req in self.waiting_queue:
            entry_time = req.time_stats.wait_queue_entry_time
            if 0 < entry_time < deadline:
                if self.enable_hicache_storage:
                    # 释放与该请求关联的预取（prefetch）事件
                    self.tree_cache.release_aborted_request(req.rid)
                self.send_to_tokenizer.send_output(
                    AbortReq(
                        finished_reason={
                            "type": "abort",
                            "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                            "message": "Request waiting timeout reached.",
                        },
                        rid=req.rid,
                    ),
                    req,
                )
                deleted_reqs.add(req)

        if deleted_reqs:
            self.waiting_queue = [
                req for req in self.waiting_queue if req not in deleted_reqs
            ]

    def handle_embedding_request(
        self,
        recv_req: TokenizedEmbeddingReqInput,
    ):
        """处理 embedding 请求：构造 Req、处理多模态输入与长度校验，并加入等待队列。"""
        req = Req(
            recv_req.rid,
            recv_req.input_text,
            recv_req.input_ids,
            recv_req.sampling_params,
            token_type_ids=recv_req.token_type_ids,
            routed_dp_rank=recv_req.routed_dp_rank,
            priority=recv_req.priority,
            dimensions=recv_req.dimensions,
            lora_id=recv_req.lora_id,
            http_worker_ipc=recv_req.http_worker_ipc,
            time_stats=recv_req.time_stats,
        )
        req.tokenizer = self.tokenizer

        # 处理多模态输入
        if recv_req.image_inputs is not None:
            image_inputs = self._get_multimodal_inputs(recv_req.image_inputs)
            # 将单个 image token 扩展为多个占位 token，用于接收图像 embedding
            # `pad_input_ids_func` 与具体模型相关，对 embedding 模型或不需要特殊 padding 的
            # 模型可能为 None。
            # 若为 None，则预期 `req.origin_input_ids` 已被正确填充。
            if self.pad_input_ids_func:
                req.origin_input_ids = self.pad_input_ids_func(
                    req.origin_input_ids, image_inputs
                )

            req.extend_image_inputs(image_inputs)
            self._maybe_compute_mrope_positions(req)

            if len(req.origin_input_ids) >= self.max_req_input_len:
                req.set_finish_with_abort(
                    error_msg=(
                        "Multimodal prompt is too long after expanding multimodal tokens. "
                        f"After expanding {len(req.origin_input_ids_unpadded)=} => {len(req.origin_input_ids)} >= {self.max_req_input_len}."
                    )
                )
                self._add_request_to_queue(req)
                return

        # 校验 prompt 长度
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            self.server_args.allow_auto_truncate,
        )
        if error_msg:
            self._add_request_to_queue(req)
            return

        # 复制更多属性
        req.logprob_start_len = -1
        self._add_request_to_queue(req)

    def handle_batch_embedding_request(
        self,
        recv_req: BatchTokenizedEmbeddingReqInput,
    ):
        """处理经过优化的批量 embedding 请求。"""
        logger.debug(
            f"Processing batch embedding request with {len(recv_req)} requests"
        )

        # 逐个处理批次中的每个请求
        for tokenized_req in recv_req:
            self.handle_embedding_request(tokenized_req)

    def stash_chunked_request(self, req: Req):
        """将分块预填充中未完成的请求暂存到 tree_cache（标记为 chunked）。"""
        self.tree_cache.cache_unfinished_req(req, chunked=True)

    def _build_hisparse_decode_batch(self, reqs):
        """为正从 staging 过渡到 decode 的 hisparse 请求构建一个 ScheduleBatch。"""
        device = self.device

        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=self.enable_overlap,
            spec_algorithm=self.spec_algorithm,
        )

        batch.req_pool_indices = torch.tensor(
            [r.req_pool_idx for r in reqs], dtype=torch.int64, device=device
        )
        seq_lens = [len(r.origin_input_ids) + len(r.output_ids) - 1 for r in reqs]
        batch.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=device)
        batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        batch.seq_lens_sum = sum(seq_lens)
        # output_ids = 最后生成的 token，会被 prepare_for_decode 用作 input_ids
        batch.output_ids = torch.tensor(
            [r.output_ids[-1] for r in reqs], dtype=torch.int64, device=device
        )

        # 若有任何请求需要 logprob，则设置相关字段
        if batch.return_logprob:
            batch.top_logprobs_nums = [r.top_logprobs_num for r in reqs]
            batch.token_ids_logprobs = [list(r.origin_input_ids) for r in reqs]

        # 为这些请求从头构建 sampling 信息
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, self.model_config.vocab_size
        )
        # todo hisparse，新批次可能还需要包含其他信息
        return batch

    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        """获取下一个要运行的批次。

        先处理超时请求，然后尝试合并上一个 prefill 批次到运行批次；
        优先返回新的 prefill 批次，否则返回更新后的 decode 批次。
        同时处理 DP attention 同步、ngram embedding 等。
        """
        self._abort_on_waiting_timeout()
        self._abort_on_running_timeout()
        if self.dllm_config is not None:
            self.dllm_manager.filter_finished_reqs()

        # 将 prefill 批次合并到运行批次中
        chunked_req_to_exclude = set()

        if self.dllm_config is not None and self.dllm_manager.any_staging_reqs():
            chunked_req_to_exclude.update(self.dllm_manager.staging_queue)
            for req in self.dllm_manager.staging_queue:
                self.stash_chunked_request(req)

        if self.chunked_req is not None:
            # 将 chunked 请求移出批次，这样我们就只把已完成的请求合并到 running_batch。
            chunked_req_to_exclude.add(self.chunked_req)
            self.stash_chunked_request(self.chunked_req)

        # HiSparse 有自己的 prefill-to-decode 过渡逻辑；跳过 last_batch 的合并。
        if self.enable_hisparse:
            ready_reqs = self.hisparse_coordinator.collect_ready_reqs()
            if len(ready_reqs) > 0:
                new_batch = self._build_hisparse_decode_batch(ready_reqs)
                if self.running_batch.is_empty():
                    self.running_batch = new_batch
                else:
                    self.running_batch.merge_batch(new_batch)
                self.running_batch.hisparse_coordinator = self.hisparse_coordinator

        if (
            not self.enable_hisparse
            and self.last_batch
            and self.last_batch.forward_mode.is_extend()
        ):
            if self.last_batch.chunked_req is not None:
                # 在上下文流水线并行（context pipeline parallelism）中，最后一个 chunk 之后，
                # 当前 microbatch 仍然引用着过期的 chunked_req，需要将其丢弃。
                chunked_req_to_exclude.add(self.last_batch.chunked_req)

            if self.dllm_config is not None and self.last_batch.reqs:
                chunked_req_to_exclude.update(self.last_batch.reqs)

            # 过滤批次
            last_bs = self.last_batch.batch_size()
            self.last_batch.filter_batch(
                chunked_req_to_exclude=list(chunked_req_to_exclude)
            )
            if self.last_batch.batch_size() < last_bs:
                self.running_batch.batch_is_full = False

            # 将新批次合并到运行批次中。
            if not self.last_batch.is_empty():
                if self.running_batch.is_empty():
                    self.running_batch = self.last_batch
                else:
                    # 将 running_batch 与 prefill 批次合并
                    self.running_batch.merge_batch(self.last_batch)

        # 对于 prefill-only 批次，过滤掉已完成的请求，因为它们不会进入 decode 步骤。
        # 这样可以保持 running_batch 的准确性，便于负载上报（通过 /get_load 上报 num_running_reqs）。
        # 该逻辑放在 last_batch 代码块之外，这样即使没有新批次到来（例如流量停止）时，
        # 过期请求也能被清理。
        if self.running_batch.is_prefill_only:
            self.running_batch.filter_batch()

        if self.dllm_config is not None:
            new_batch = self.get_new_batch_dllm()
        else:
            new_batch = self.get_new_batch_prefill()

        need_mlp_sync = self.require_mlp_sync
        if need_mlp_sync and not self.spec_algorithm.is_none():
            # NOTE: 该分支确保在同时启用 spec 与 dp-attn 时，prefill 与 decode 批次不会被混在一起。
            # 在把新批次合并进运行批次之前：
            # 1. 所有 new batch 都为 none -> need_mlp_sync 保持为 true（decode 批次需要同步）。
            # 2. 所有 new batch 都非空（prefill / idle）-> 不需要再额外做一次 mlp sync 准备。
            new_batch = self.maybe_prepare_mlp_sync_batch(new_batch)
            need_mlp_sync = new_batch is None

        if new_batch is not None:
            # 如果可以，优先运行 prefill
            ret = new_batch
        else:
            # 运行 decode（prefill-only 批次跳过）
            if (
                not self.running_batch.is_empty()
                and not self.running_batch.is_prefill_only
            ):
                self.running_batch = self.update_running_batch(self.running_batch)
                ret = self.running_batch if not self.running_batch.is_empty() else None
            else:
                ret = None

        # 处理 DP attention 并记录统计信息
        ret = self.maybe_prepare_mlp_sync_batch(ret, need_sync=need_mlp_sync)

        # 处理 ngram embedding
        ret = self._maybe_prepare_ngram_embedding(ret)

        if ret:
            set_schedule_time_batch(ret)

        return ret

    def get_num_allocatable_reqs(self, running_bs):
        """计算当前还可新增分配的请求数（受 PP 微批次上限与 token 池可用量限制）。"""
        res = get_global_server_args().pp_max_micro_batch_size - running_bs
        if self.pp_size > 1:
            res = min(res, self.req_to_token_pool.available_size())
        return res

    def get_new_batch_prefill(self) -> Optional[ScheduleBatch]:
        """从等待队列中组建一个新的 prefill（预填充）批次。

        考虑 prefill 延迟器、动态分块、LoRA 限制、优先级抢占、分层缓存预取进度等因素，
        返回可运行的新批次；若无可调度请求则返回 None。
        """
        prefill_delayer_single_pass = None
        if self.prefill_delayer:
            # 从多个池中获取 token 使用量
            token_usage = None
            if self.is_hybrid_swa:
                _, _, full_token_usage, swa_token_usage, *_ = self._get_swa_token_info()
                token_usage = max(full_token_usage, swa_token_usage)
            if self.is_hybrid_ssm:
                _, _, full_token_usage, mamba_token_usage, *_ = (
                    self._get_mamba_token_info()
                )
                token_usage = (
                    max(token_usage, mamba_token_usage)
                    if token_usage is not None
                    else max(full_token_usage, mamba_token_usage)
                )
            if token_usage is None:
                _, token_usage, _, _ = self._get_token_info()

            assert token_usage is not None
            prefill_delayer_single_pass = PrefillDelayerSinglePassExecutor(
                self.prefill_delayer, token_usage=token_usage
            )

        ret = self._get_new_batch_prefill_raw(
            prefill_delayer_single_pass=prefill_delayer_single_pass
        )

        if self.prefill_delayer:
            prefill_delayer_single_pass.finalize(actual_prefill=ret is not None)

        return ret

    def _get_new_batch_prefill_raw(
        self, prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor]
    ) -> Optional[ScheduleBatch]:
        # 检查 grammar 队列中的 grammar 是否已就绪
        if self.grammar_manager.has_waiting_grammars():
            ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
            for req in ready_grammar_requests:
                self._add_request_to_queue(req)

        if self.enable_hierarchical_cache:
            self.tree_cache.check_hicache_events()

        if self.enable_priority_preemption:
            # 重置 batch_is_full，以便用 prefill adder 尝试抢占。
            self.running_batch.batch_is_full = False

        if (
            self.running_batch.batch_is_full or len(self.waiting_queue) == 0
        ) and self.chunked_req is None:
            return None

        running_bs = len(self.running_batch.reqs)

        # 当 self.chunked_req 不为 None 时忽略此检查。
        # 在非 PP 情况下，当 self.chunked_req 不为 None 时，num_allocatable_reqs 应始终大于 0，
        # 因为 chunked 请求所占的空间刚刚被释放。
        # 在 PP 情况下，chunked 请求（或 dllm 请求）可能在一个 microbatch 中开始、在另一个 microbatch 中结束，
        # 因此每个 microbatch 的 max_running_requests 不应是严格限制。
        # 相反，我们应始终允许添加 chunked 请求，否则会导致显存泄漏。
        if (
            self.get_num_allocatable_reqs(running_bs) <= 0
            and self.chunked_req is not None
            and not self.enable_priority_preemption
        ):
            self.running_batch.batch_is_full = True
            return None

        # 获取优先级队列
        self.policy.calc_priority(self.waiting_queue, self.running_batch)

        if TEST_RETRACT and running_bs > TEST_RETRACT_NO_PREFILL_BS:
            # 如果正在测试 retraction，且运行批次大小超过 TEST_RETRACT_NO_PREFILL_BS，
            # 则跳过 prefill，让这些请求继续留在等待队列中。
            return None

        # 确定本批次的 chunked_prefill_size
        chunked_prefill_size = self.chunked_prefill_size
        if self.chunked_req is not None and self.enable_dynamic_chunking:
            history_len = len(self.chunked_req.prefix_indices)
            dynamic_size = self.predict_next_chunk_size(history_len)
            if dynamic_size is not None:
                chunked_prefill_size = dynamic_size

        # Prefill 策略
        adder = PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            self.running_batch,
            self.new_token_ratio,
            self.max_prefill_tokens,
            chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
            self.priority_scheduling_preemption_threshold,
            max_prefill_bs=self.max_prefill_bs,
            max_running_requests=self.max_running_requests,
            prefill_max_requests=self.server_args.prefill_max_requests,
            prefill_delayer_single_pass=prefill_delayer_single_pass,
            dllm_config=self.dllm_config,
        )

        if self.chunked_req is not None:
            self.chunked_req.init_next_round_input()
            self.chunked_req = adder.add_chunked_req(self.chunked_req)

        if self.enable_lora:
            running_loras = {req.lora_id for req in self.running_batch.reqs}

        # 从等待队列中取出请求，组成新的 prefill 批次
        for req in self.waiting_queue:
            if self.enable_lora and req.lora_id not in running_loras:
                if self.enable_lora_overlap_loading:
                    # 为了让 LoRA 权重的加载与计算重叠，我们会逐个加载每个 adapter，
                    # 而不是在一个批次里一次性加载它们
                    res = self.lora_overlap_loader.try_overlap_load_lora(
                        req.lora_id, running_loras
                    )
                    if not res:
                        continue
                else:
                    new_lora_set = {req.lora_id} | running_loras
                    if not self.tp_worker.model_runner.lora_manager.validate_lora_batch(
                        new_lora_set
                    ):
                        continue

            running_bs = len(self.running_batch.reqs)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                self.running_batch.batch_is_full = True
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                # 在 prefill 模式下，prealloc 队列和 transfer 队列也会占用显存，
                # 因此我们需要检查实际可用的空间大小。
                if len(adder.can_run_list) >= self.req_to_token_pool.available_size():
                    self.running_batch.batch_is_full = True

            if self.running_batch.batch_is_full:
                if (
                    not self.enable_priority_preemption
                    or not adder.preempt_to_schedule(req, self.server_args)
                ):
                    break

            if self.enable_hicache_storage:
                prefetch_done = self.tree_cache.check_prefetch_progress(req.rid)
                if not prefetch_done:
                    # 跳过仍在进行预取（prefetch）的 staging 请求
                    continue
                # 取出从存储中加载的 token 数量（L3 命中）
                req.storage_hit_length = self.tree_cache.pop_prefetch_loaded_tokens(
                    req.rid
                )

            req.init_next_round_input(self.tree_cache)
            res = adder.add_one_req(
                req,
                has_chunked_req=(self.chunked_req is not None),
                truncation_align_size=self.truncation_align_size,
            )

            if self.enable_lora:
                running_loras.add(req.lora_id)

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    if self.enable_hierarchical_cache:
                        # 在确认确实存在可服务的请求之后再设置 batch_is_full
                        self.running_batch.batch_is_full = len(
                            adder.can_run_list
                        ) > 0 or (not self.running_batch.is_empty())
                    else:
                        self.running_batch.batch_is_full = True
                # 如果该请求未被加入，则回退已匹配的 mamba idx，以避免显存泄漏
                added = len(adder.can_run_list) > 0 and req is adder.can_run_list[-1]
                if not added and req.mamba_pool_idx is not None:
                    self.tree_cache.req_to_token_pool.mamba_pool.free(
                        req.mamba_pool_idx.unsqueeze(-1)
                    )
                    req.mamba_pool_idx = None
                break

        # 更新等待队列
        can_run_list: List[Req] = adder.can_run_list
        if len(can_run_list) == 0:
            return None

        can_run_set = set(can_run_list)
        self.waiting_queue = [x for x in self.waiting_queue if x not in can_run_set]
        if adder.preempt_list:
            for req in adder.preempt_list:
                self._add_request_to_queue(req)

        if adder.new_chunked_req is not None:
            # 更新分块预填充（chunked prefill）
            assert self.chunked_req is None
            self.chunked_req = adder.new_chunked_req

        if self.chunked_req is not None:
            self.chunked_req.is_chunked += 1

        # 记录下来，便于在 forward 之后打印 prefill 统计日志
        self.adder = adder
        self.can_run_list = can_run_list
        self.running_bs = len(self.running_batch.reqs)

        set_time_batch(can_run_list, "set_forward_entry_time")

        # 创建一个新批次
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            chunked_req=self.chunked_req,
        )
        self.max_prefill_bs = max(self.max_prefill_bs, len(can_run_list))
        if self.enable_hierarchical_cache:
            # todo (zhiqiang): 如果触发了 hicache 加载，则禁用 CUDA graph 执行
            new_batch.hicache_consumer_index = (
                self.tree_cache.ready_to_load_host_cache()
            )

        new_batch.prepare_for_extend()

        # 记录 prefill 统计信息，便于在 forward 之后打印日志
        new_batch.prefill_stats = PrefillStats.from_adder(
            adder, self.running_batch.reqs, self.enable_priority_scheduling
        )

        # 混合式（mixed-style）分块预填充
        if (
            self.is_mixed_chunk
            and not self.running_batch.is_empty()
            and not (new_batch.return_logprob or self.running_batch.return_logprob)
            # mix_with_running 会拼接 input_ids，但不会拼接 input_embeds——否则 shape 会不匹配
            and new_batch.input_embeds is None
        ):
            # TODO (lianmin): 支持 return_logprob + 混合式分块预填充
            self.running_batch.filter_batch(v1_spec_info_filtered=True)
            if not self.running_batch.is_empty():
                self.running_batch.prepare_for_decode()
                new_batch.mix_with_running(self.running_batch)
                new_batch.decoding_reqs = self.running_batch.reqs
            self.running_batch = ScheduleBatch(
                reqs=[], batch_is_full=self.running_batch.batch_is_full
            )
        else:
            new_batch.decoding_reqs = None

        return new_batch

    def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch]:
        """更新当前正在运行的 decode 批次。"""
        initial_bs = batch.batch_size()

        batch.filter_batch(v1_spec_info_filtered=True)
        if batch.is_empty():
            batch.batch_is_full = False
            return batch

        # 及时释放已完成 write-through 节点上的 lock_ref，使它们变为可驱逐，
        # 从而为批次调度腾出更多余量。
        if self.enable_hierarchical_cache:
            self.tree_cache.flush_write_through_acks()

        # 检查 decode 是否会显存不足（OOM）
        if (kv_full_retract_flag := not batch.check_decode_mem()) or (
            TEST_RETRACT and self.forward_ct % TEST_RETRACT_INTERVAL == 0
        ):
            old_available_tokens = self.token_to_kv_pool_allocator.available_size()
            old_ratio = self.new_token_ratio
            retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(
                self.server_args
            )
            new_available_tokens = self.token_to_kv_pool_allocator.available_size()
            new_token_gained = new_available_tokens - old_available_tokens

            self.num_retracted_reqs = len(retracted_reqs)
            if self.enable_metrics and len(retracted_reqs) > 0:
                self.metrics_collector.increment_retracted_reqs(
                    num_retracted_reqs=len(retracted_reqs),
                    num_retracted_input_tokens=sum(
                        len(r.origin_input_ids) for r in retracted_reqs
                    ),
                    num_retracted_output_tokens=sum(
                        len(r.output_ids) for r in retracted_reqs
                    ),
                )
            self.new_token_ratio = new_token_ratio
            for req in reqs_to_abort:
                abort_reason: FINISH_ABORT = req.to_finish
                self.send_to_tokenizer.send_output(
                    AbortReq(
                        finished_reason=abort_reason.to_json(),
                        rid=req.rid,
                    ),
                    req,
                )

            msg_prefix = (
                "KV cache pool is full. Retract requests. "
                if kv_full_retract_flag
                else "Testing retraction. "
            )
            msg_details = f"#retracted_reqs: {len(retracted_reqs)}, #new_tokens_gained: {new_token_gained}"
            if kv_full_retract_flag:
                msg_details += (
                    f", #new_token_ratio: {old_ratio:.4f} -> {new_token_ratio:.4f}"
                )
            logger.warning(msg_prefix + msg_details)

            for req in retracted_reqs:
                self._add_request_to_queue(req, is_retracted=True)
                if self.enable_hisparse:
                    self.hisparse_coordinator.retract_req(req)
        else:
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_decay,
                self.min_new_token_ratio,
            )

        if batch.batch_size() < initial_bs:
            batch.batch_is_full = False

        if batch.is_empty():
            return batch

        # 更新批次张量
        batch.prepare_for_decode()
        return batch

    def record_batch_in_overlap(self, model_worker_batch: ModelWorkerBatch):
        """在 overlap 调度下保留对批次张量的引用，避免 GPU 张量被 torch GC 提前释放。"""
        # FIXME(lsyin): 这是一种 hacky 的做法，靠保留引用来避免 GPU 张量被 torch GC 释放
        # NOTE: 更可靠的做法：把所有张量都记录到 forward stream 上
        # NOTE: - 对于所有 future 张量，我们应始终从 future map 中读取
        #       - 对于所有非 future 张量（仅由 schedule stream 产生的），
        #       我们应在整个前向计算过程中保持其引用不被释放
        self.batch_record_ct = (self.batch_record_ct + 1) % 2
        self.batch_record_buf[self.batch_record_ct] = model_worker_batch

    def run_batch(
        self,
        batch: ScheduleBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[GenerationBatchResult, EmbeddingBatchResult]:
        """运行一个批次。"""
        self.forward_ct += 1

        # 是否运行 profiler
        self._profile_batch_predicate(batch)
        if self.forward_sleep_time is not None:
            logger.info(f"Scheduler.run_batch sleep {self.forward_sleep_time}s")
            time.sleep(self.forward_sleep_time)

        # 在 EXTEND 模式下记录 prefill 开始时间
        if batch.forward_mode == ForwardMode.EXTEND:
            set_time_batch(batch.reqs, "set_prefill_run_batch_start_time")

        # 在 PD 分离的 decode 事件循环中处理占位（placeholder）批次
        if batch.forward_mode.is_prebuilt():
            return self._run_batch_prebuilt(batch)

        # 运行前向计算
        if self.is_generation:
            if self.spec_algorithm.is_none() or self.enable_overlap:
                # 大多数情况下，我们使用 model worker batch 来运行前向计算。
                worker_batch_or_batch = batch.get_model_worker_batch()
            else:
                # 在投机解码 v1（非 overlap）情况下，我们直接使用 batch。
                # TODO(lsyin): 统一抽象之后删除这个分支。
                worker_batch_or_batch = batch

            if self.enable_overlap:
                model_worker_batch = worker_batch_or_batch
                self.record_batch_in_overlap(model_worker_batch)

                # Sampling 信息会在前向计算过程中被修改，因此我们保存一份副本。
                model_worker_batch.sampling_info = (
                    model_worker_batch.sampling_info.copy_for_forward()
                )

                bs = len(model_worker_batch.seq_lens)
                future_indices = self.future_map.alloc_future_indices(bs)

                with self.forward_stream_ctx, self.record_bubble_metrics(batch):
                    self.forward_stream.wait_stream(self.schedule_stream)
                    self.future_map.resolve_future(model_worker_batch)
                    with self.record_forward_metrics(batch):
                        batch_result = self.model_worker.forward_batch_generation(
                            model_worker_batch
                            # 这里 pp 与 overlap 不兼容
                        )
                    # FIXME(lsyin): 也许可以把这一步移到 forward_batch_generation 里
                    batch_result.copy_done = self.device_module.Event()
                    if batch_result.delay_sample_func is None:
                        self.future_map.store_to_map(future_indices, batch_result)
                        batch_result.copy_to_cpu(return_logprob=batch.return_logprob)
                    else:
                        batch_result.future_indices = future_indices

                # FIXME(lsyin): 把这个赋值移到别处
                future_indices_or_next_token_ids = -future_indices.indices

                if batch.is_spec_v2:
                    # FIXME(lsyin): 这是 spec v2 的临时代码
                    # 我们只为下一次 draft 输入保留 future indices

                    batch.spec_info = batch_result.next_draft_input
                    batch.spec_info.future_indices = future_indices

                    # batch.spec_info = EagleDraftInput(
                    #     future_indices=future_indices,
                    #     verify_done=batch_result.next_draft_input.verify_done,
                    # )

                    # future 值，通常用于下一个批次的准备
                    # 当前实现严格同步 seq_lens
                    batch.seq_lens = batch_result.next_draft_input.new_seq_lens
            elif self.enable_pdmux and batch.forward_mode.is_split_prefill():
                batch_result = self.tp_worker.forward_batch_split_prefill(batch)
                future_indices_or_next_token_ids = batch_result.next_token_ids
            else:
                kwargs = (
                    {"pp_proxy_tensors": pp_proxy_tensors}
                    if self.spec_algorithm.is_none()
                    else {}
                )
                with self.record_forward_metrics(batch):
                    batch_result = self.model_worker.forward_batch_generation(
                        worker_batch_or_batch, **kwargs
                    )
                future_indices_or_next_token_ids = batch_result.next_token_ids
                self.update_cache_from_scheduler(batch, batch_result)

            # NOTE: future_indices_or_next_token_ids 用于 ScheduleBatch，
            #       将来可能会用 future_indices 来替代它 [TODO(lsyin)]。
            #       我们仍应在 GenerationBatchOutput 中保留原始输出（例如 next_token_ids），
            #       以便在 copy_done 之后用于处理。
            batch.output_ids = future_indices_or_next_token_ids

            # 这 2 个值在处理输出时需要用到，但它们可能会被 overlap 调度修改。
            # 因此我们必须在此处复制它们，以便在输出处理时使用正确的值。
            if batch.return_logprob:
                batch_result.extend_input_len_per_req = [
                    req.extend_input_len for req in batch.reqs
                ]
                batch_result.extend_logprob_start_len_per_req = [
                    req.extend_logprob_start_len for req in batch.reqs
                ]
            else:
                batch_result.extend_input_len_per_req = None
                batch_result.extend_logprob_start_len_per_req = None

            ret = batch_result
        else:  # embedding 或 reward 模型
            model_worker_batch = batch.get_model_worker_batch()

            if self.enable_overlap:
                self.record_batch_in_overlap(model_worker_batch)
                with self.forward_stream_ctx, self.record_bubble_metrics(batch):
                    self.forward_stream.wait_stream(self.schedule_stream)
                    embeddings = self.tp_worker.forward_batch_embedding(
                        model_worker_batch
                    )
                    ret = EmbeddingBatchResult(embeddings=embeddings)
                    ret.copy_to_cpu()
            else:
                embeddings = self.tp_worker.forward_batch_embedding(model_worker_batch)
                ret = EmbeddingBatchResult(embeddings=embeddings)

        # 在 EXTEND 模式下记录 prefill 结束时间
        if batch.forward_mode == ForwardMode.EXTEND:
            set_time_batch(batch.reqs, "set_prefill_run_batch_end_time")

        if (
            self.server_args.enable_dp_attention
            and self.server_args.elastic_ep_backend is not None
        ):
            # 获取标记各 rank 是否处于活跃状态的张量
            tp_active_ranks = self.tp_group.active_ranks.detach().cpu().numpy()
            tp_active_ranks_cpu = self.tp_group.active_ranks_cpu.detach().numpy()
            tp_active_ranks &= tp_active_ranks_cpu
            dp_active_ranks = tp_active_ranks.reshape(self.dp_size, -1).prod(axis=1)
            self.send_to_tokenizer.send_output(
                ActiveRanksOutput(status=dp_active_ranks.tolist())
            )

        return ret

    def launch_batch_sample_if_needed(
        self, batch_result: GenerationBatchResult
    ) -> Union[GenerationBatchResult]:
        """若批次结果需要延迟采样，则执行采样函数，并及时释放不再需要的闭包与大 GPU 张量以避免显存泄漏。"""
        # TODO(lsyin): 在统一 forward_batch_generation 接口之后（与 spec V2 相关），
        # 将延迟采样（delayed sample）设为默认行为。
        if batch_result is None or batch_result.delay_sample_func is None:
            return

        with self.forward_stream_ctx:
            self.forward_stream.wait_stream(self.schedule_stream)
            _batch_result = batch_result.delay_sample_func()
            assert _batch_result is batch_result
            self.future_map.store_to_map(batch_result.future_indices, batch_result)
            batch_result.copy_to_cpu(return_logprob=self.cur_batch.return_logprob)

        # 释放不再需要的闭包和大型 GPU 张量。
        # delay_sample_func 闭包捕获了 forward_batch（其中持有带 vocab_mask 的 sampling_info）
        # 和 logits_output（其中持有 next_token_logits）。如果不清除它们，它们会通过
        # result_queue 和 batch_record_buf 中的 batch_result 一直存活到下一次迭代，
        # 在结构化输出场景下造成持续的显存（VRAM）泄漏。
        batch_result.delay_sample_func = None
        if batch_result.logits_output is not None:
            batch_result.logits_output.next_token_logits = None

    def process_batch_result(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        """根据批次的前向模式（decode/extend/prebuilt/idle 等）分发到对应的结果处理函数。"""
        if batch.forward_mode.is_decode():
            self.process_batch_result_decode(batch, result)
        elif batch.forward_mode.is_extend():
            if batch.is_dllm():
                self.process_batch_result_dllm(batch, result)
            elif self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.process_batch_result_disagg_prefill(batch, result)
            else:
                self.process_batch_result_prefill(batch, result)
        elif batch.forward_mode.is_prebuilt():
            self.process_batch_result_prebuilt(batch)
        elif batch.forward_mode.is_idle():
            self.process_batch_result_idle(batch, result)

        self.log_batch_result_stats(batch, result)
        self._maybe_clear_mm_inputs(batch)
        self.maybe_send_health_check_signal()

    def maybe_send_health_check_signal(self):
        """若有待响应的健康检查，则发送健康信号（避免被长上下文 prefill 阻塞）。"""
        if self.return_health_check_ipcs:
            # 为健康检查返回某种信号。
            # 这用于防止健康检查信号被长上下文 prefill 阻塞。
            # 不过有一个小问题：这条代码路径不会检查 detokenizer manager 的状态。
            self.send_to_tokenizer.send_output(
                HealthCheckOutput(
                    http_worker_ipc=self.return_health_check_ipcs.popleft()
                )
            )

    def _check_pending_flush(self):
        """检查是否有被延迟的 flush_cache 请求：一旦调度器完全空闲则执行清缓存，超时则返回失败。"""
        if self._pending_flush is None:
            return

        pending_req, deadline = self._pending_flush

        if self.is_fully_idle():
            success = self.flush_cache()
            self._pending_flush = None
            self.send_to_tokenizer.send_output(
                FlushCacheReqOutput(success=success), pending_req
            )
            return

        if time.monotonic() >= deadline:
            logging.warning(
                "Deferred flush_cache timed out while waiting for idle state."
            )
            self._pending_flush = None
            self.send_to_tokenizer.send_output(
                FlushCacheReqOutput(
                    success=False, message="Timed out waiting for idle state."
                ),
                pending_req,
            )

    def flush_cache_wrapped(
        self, recv_req: FlushCacheReqInput
    ) -> Optional[FlushCacheReqOutput]:
        """处理清空缓存请求：若当前空闲则立即清空，否则按超时时间延迟到空闲时再执行。"""
        if self._pending_flush is not None:
            return FlushCacheReqOutput(
                success=False,
                message="Another flush_cache is already in progress.",
            )

        timeout_s = float(recv_req.timeout_s or 0.0)
        if timeout_s <= 0.0:
            return FlushCacheReqOutput(success=self.flush_cache())

        if self.is_fully_idle():
            return FlushCacheReqOutput(success=self.flush_cache())

        self._pending_flush = (recv_req, time.monotonic() + timeout_s)
        return None

    def clear_hicache_storage_wrapped(self, recv_req: ClearHiCacheReqInput):
        """处理清空分层缓存存储后端的请求（仅在启用分层缓存时生效）。"""
        if self.enable_hierarchical_cache:
            self.tree_cache.clear_storage_backend()
            logger.info("Hierarchical cache cleared successfully!")
            if_success = True
        else:
            logging.warning("Hierarchical cache is not enabled.")
            if_success = False
        return ClearHiCacheReqOutput(success=if_success)

    def is_fully_idle(self, for_health_check=False) -> bool:
        """判断调度器是否完全空闲。

        综合检查运行批次、等待队列、分块请求、overlap 结果队列以及（非健康检查时）
        grammar 队列、PD 分离队列与 HiCache 在飞异步操作是否均已清空。
        """
        # 健康检查在 process_output 中搭载于正在运行的请求之上。
        # 只有 running_batch + waiting_queue 才能保证 GPU 正在活跃处理；
        # 分离部署的队列（bootstrap/prealloc/transfer）中可能有条目，但并没有任何
        # 请求真正在 GPU 上运行——例如握手卡住、KV cache 满，或传输停滞——
        # 因此它们不能承载健康信息。
        # 批次运行状态
        idle = (
            self.running_batch.is_empty()
            and self.chunked_req is None
            and not self.dllm_manager.any_staging_reqs()
            and (self.last_batch is None or self.last_batch.is_empty())
            and (self.cur_batch is None or self.cur_batch.is_empty())
            and (not self.enable_overlap or len(self.result_queue) == 0)
            and (self.pp_size == 1 or all(x.is_empty() for x in self.running_mbs))
        )

        # 等待中的各队列：waiting + bootstrapping + preallocation + kv transfer（decode）
        idle &= len(self.waiting_queue) == 0

        if not for_health_check:
            # grammar 队列和 prefill inflight 队列可能不会立即产出批次结果，
            # 但它们仍然表明服务器并不空闲。
            idle &= len(self.grammar_manager.grammar_queue) == 0
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                idle &= len(self.disagg_prefill_inflight_queue) == 0
                idle &= len(self.disagg_prefill_bootstrap_queue.queue) == 0

            if self.disaggregation_mode == DisaggregationMode.DECODE:
                idle &= len(self.disagg_decode_prealloc_queue.queue) == 0
                idle &= len(self.disagg_decode_transfer_queue.queue) == 0

            # HiCache：在执行 attach/detach/flush_cache 等破坏性操作之前，
            # 必须先排空在飞的异步操作（GPU↔Host↔L3）。
            if self.enable_hierarchical_cache:
                tc = self.tree_cache
                idle &= len(tc.ongoing_write_through) == 0
                idle &= len(tc.ongoing_load_back) == 0
                if tc.enable_storage:
                    idle &= len(tc.ongoing_prefetch) == 0
                    idle &= len(tc.ongoing_backup) == 0

        return idle

    def attach_hicache_storage_wrapped(
        self, recv_req: AttachHiCacheStorageReqInput
    ) -> AttachHiCacheStorageReqOutput:
        """动态挂载分层缓存存储后端（要求调度器处于空闲状态）。"""
        if not self.enable_hierarchical_cache:
            return AttachHiCacheStorageReqOutput(
                success=False, message="Hierarchical cache is not enabled."
            )

        if not self.is_fully_idle():
            return AttachHiCacheStorageReqOutput(
                success=False,
                message=(
                    "Reject attach: scheduler is not idle. "
                    f"#queue-req={len(self.waiting_queue)} "
                    f"#running-req={len(self.running_batch.reqs)}"
                ),
            )

        if not hasattr(self.tree_cache, "attach_storage_backend"):
            return AttachHiCacheStorageReqOutput(
                success=False,
                message="Current tree_cache implementation does not support dynamic attach.",
            )

        try:
            ok, msg = self.tree_cache.attach_storage_backend(
                storage_backend=recv_req.hicache_storage_backend,
                storage_backend_extra_config_json=recv_req.hicache_storage_backend_extra_config_json,
                served_model_name=self.server_args.served_model_name,
                hicache_storage_prefetch_policy=recv_req.hicache_storage_prefetch_policy,
                hicache_write_policy=recv_req.hicache_write_policy,
            )
        except Exception as e:
            logger.exception("Attach HiCache storage backend failed with exception.")
            return AttachHiCacheStorageReqOutput(success=False, message=str(e))
        if ok:
            self.enable_hicache_storage = True
            self.server_args.hicache_storage_backend = recv_req.hicache_storage_backend
            if recv_req.hicache_storage_backend_extra_config_json is not None:
                self.server_args.hicache_storage_backend_extra_config = (
                    recv_req.hicache_storage_backend_extra_config_json
                )
            if recv_req.hicache_storage_prefetch_policy is not None:
                self.server_args.hicache_storage_prefetch_policy = (
                    recv_req.hicache_storage_prefetch_policy
                )
            if recv_req.hicache_write_policy is not None:
                self.server_args.hicache_write_policy = recv_req.hicache_write_policy
            logger.info(
                f"Attached HiCache storage backend: {recv_req.hicache_storage_backend}"
            )
        return AttachHiCacheStorageReqOutput(success=ok, message=msg)

    def detach_hicache_storage_wrapped(
        self, recv_req: DetachHiCacheStorageReqInput
    ) -> DetachHiCacheStorageReqOutput:
        """动态卸载分层缓存存储后端（幂等操作，要求调度器空闲）。"""
        if not self.enable_hierarchical_cache:
            return DetachHiCacheStorageReqOutput(
                success=False, message="Hierarchical cache is not enabled."
            )

        if not self.is_fully_idle():
            return DetachHiCacheStorageReqOutput(
                success=False,
                message=(
                    "Reject detach: scheduler is not idle. "
                    f"#queue-req={len(self.waiting_queue)} "
                    f"#running-req={len(self.running_batch.reqs)}"
                ),
            )

        if not hasattr(self.tree_cache, "detach_storage_backend"):
            return DetachHiCacheStorageReqOutput(
                success=False,
                message="Current tree_cache implementation does not support dynamic detach.",
            )

        # 幂等的 detach：即使 scheduler 认为存储已被禁用，我们仍会在 tree_cache 中
        # 尽力做一次清理（它可能残留有状态）。
        try:
            ok, msg = self.tree_cache.detach_storage_backend()
        except Exception as e:
            logger.exception("Detach HiCache storage backend failed with exception.")
            return DetachHiCacheStorageReqOutput(success=False, message=str(e))

        if ok or (not self.enable_hicache_storage):
            # 出于幂等性考虑，将“已禁用 / 无事可做”也视为成功。
            self.enable_hicache_storage = False
            self.server_args.hicache_storage_backend = None
            self.server_args.hicache_storage_backend_extra_config = None
            logger.info("Detached HiCache storage backend.")
            return DetachHiCacheStorageReqOutput(
                success=True, message=msg or "HiCache storage backend is detached."
            )

        return DetachHiCacheStorageReqOutput(success=False, message=msg)

    def flush_cache(self):
        """清空显存池与缓存。"""
        if self.is_fully_idle():
            self.cur_batch = None
            self.last_batch = None
            self.tree_cache.reset()
            self.req_to_token_pool.clear()
            self.token_to_kv_pool_allocator.clear()
            self.grammar_manager.clear()
            self.reset_metrics()

            if self.draft_worker:
                self.draft_worker.clear_cache_pool()

            # TODO: 允许按需选择是否清空 cache
            torch.cuda.empty_cache()
            logger.info("Cache flushed successfully!")
            success = True
        else:
            logging.warning(
                f"Cache not flushed because there are pending requests. "
                f"#queue-req: {len(self.waiting_queue)}, "
                f"#running-req: {len(self.running_batch.reqs)}"
            )
            success = False
        return success

    def get_internal_state(self, recv_req: GetInternalStateReq):
        """获取调度器内部状态：含服务参数、吃吐吞、显存使用、投机接受长度等信息。"""
        ret = vars(get_global_server_args())
        ret["last_gen_throughput"] = self.last_gen_throughput
        ret["memory_usage"] = {
            "weight": round(self.tp_worker.model_runner.weight_load_mem_usage, 2),
            "kvcache": round(
                self.token_to_kv_pool_allocator.get_kvcache().mem_usage, 2
            ),
            "token_capacity": int(self.max_total_num_tokens),
            "graph": round(self.tp_worker.model_runner.graph_mem_usage, 2),
        }
        ret["effective_max_running_requests_per_dp"] = self.max_running_requests

        if not self.spec_algorithm.is_none() and self.spec_total_num_forward_ct > 0:
            ret["avg_spec_accept_length"] = (
                self.spec_total_num_accepted_tokens / self.spec_total_num_forward_ct
            )

        if RECORD_STEP_TIME:
            ret["step_time_dict"] = self.step_time_dict

        # 该字段不可序列化。
        ret.pop("model_config", None)

        return GetInternalStateReqOutput(internal_state=ret)

    def set_internal_state(self, recv_req: SetInternalStateReq):
        """设置调度器内部状态：仅允许更新白名单中的参数（如 PP 微批次大小、投机接受阈值）。"""
        server_args_dict = recv_req.server_args
        args_allow_update = set(
            [
                "pp_max_micro_batch_size",
                "speculative_accept_threshold_single",
                "speculative_accept_threshold_acc",
            ]
        )

        if_success = True
        for k, v in server_args_dict.items():
            if k not in args_allow_update:
                logging.warning(f"Updating {k} is not supported.")
                if_success = False
                break
            elif k == "pp_max_micro_batch_size" and (
                v > self.max_running_requests // self.pp_size or v < 1
            ):
                logging.warning(
                    f"Updating {k} to {v} is rejected because it is out of the valid range [1, {self.max_running_requests // self.pp_size}]."
                )
                if_success = False
                break

        if if_success:
            if not self.spec_algorithm.is_none() and self.spec_total_num_forward_ct > 0:
                avg_spec_accept_length = (
                    self.spec_total_num_accepted_tokens / self.spec_total_num_forward_ct
                )
                logger.info(f"{avg_spec_accept_length=}")
            self.spec_total_num_accepted_tokens = self.spec_total_num_forward_ct = 0
            for k, v in server_args_dict.items():
                setattr(get_global_server_args(), k, v)
            logger.info(f"Global server args updated! {get_global_server_args()=}")
        return SetInternalStateReqOutput(
            updated=True,
            server_args=vars(get_global_server_args()),
        )

    def handle_rpc_request(self, recv_req: RpcReqInput):
        """处理 RPC 请求：根据方法名反射调用对应方法，并在各 rank 间 barrier 同步后返回结果。"""
        # 处理 RPC 请求
        logger.info(
            f"handle_rpc_request: {recv_req.method}, param: {recv_req.parameters}"
        )

        success = True
        exec = None
        try:
            func = getattr(self, recv_req.method)
            if recv_req.parameters is not None:
                func(**recv_req.parameters)
            else:
                func()
        except Exception as e:
            success = False
            exec = e
            logger.error(f"Failed to call rpc {recv_req.method}: {str(e)}")

        barrier()
        return RpcReqOutput(success, "" if not exec else str(exec))

    def abort_request(self, recv_req: AbortReq):
        """中止请求。

        采用三种中止方式：直接从等待队列弹出；对 grammar 队列调用 set_finish_with_abort；
        对运行中请求设置 to_finish。同时处理 PD 分离各队列中请求的资源释放。
        """
        # todo hisparse，在 hisparse coordinator 中为被中止的请求释放资源
        # 删除等待队列中的请求
        to_del = []
        for i, req in enumerate(self.waiting_queue):
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                to_del.append(i)

        # 逆序处理，避免删除时出现索引错位问题
        for i in reversed(to_del):
            # 中止方式 1：直接从队列中弹出
            # 这种方式仅适用于尚未开始任何处理的请求。
            # 我们仍需向 TokenizerManager 回传一些信息以清理其状态。
            req = self.waiting_queue.pop(i)
            if self.enable_hicache_storage:
                # 释放与该请求关联的预取（prefetch）事件
                self.tree_cache.release_aborted_request(req.rid)
            self.send_to_tokenizer.send_output(AbortReq(rid=req.rid), req)
            # 在分离部署的 decode 模式下，等待队列中的请求已经分配了 KV cache。
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                if self.enable_hisparse:
                    self.hisparse_coordinator.request_finished(req)
                release_kv_cache(req, self.tree_cache)
            # 在分离部署的 prefill 模式下，释放 metadata buffer 索引
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                release_req_to_metadata_buffer(
                    req, self.req_to_metadata_buffer_idx_allocator
                )

            # 针对 mamba radix cache
            if (
                req.mamba_pool_idx is not None
                and self.disaggregation_mode != DisaggregationMode.DECODE
            ):
                release_kv_cache(req, self.tree_cache, is_insert=False)
            logger.debug(f"Abort queued request. {req.rid=}")

        # 删除 grammar 队列中的请求
        # 中止方式 2：调用 `set_finish_with_abort`
        # 该请求仍会运行一次 prefill 前向计算。
        # 这种情况下，我们把 input_ids 改成只有一个 token，使这次 prefill 开销很小。
        self.grammar_manager.abort_requests(recv_req)

        # 启用 PD 分离部署时，删除不在等待队列中的请求
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # 中止尚未完成 bootstrap 的请求
            for req in self.disagg_prefill_bootstrap_queue.queue:
                if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort bootstrap queue request. {req.rid=}")
                    if hasattr(req.disagg_kv_sender, "abort"):
                        req.disagg_kv_sender.abort()

            # 中止在飞（in-flight）的请求
            for req in self.disagg_prefill_inflight_queue:
                if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort inflight queue request. {req.rid=}")
                    if hasattr(req.disagg_kv_sender, "abort"):
                        req.disagg_kv_sender.abort()

        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            # 中止尚未完成预分配（preallocation）的请求
            for decode_req in self.disagg_decode_prealloc_queue.queue:
                if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort prealloc queue request. {decode_req.req.rid=}")
                    decode_req.kv_receiver.abort()

            # 中止正在等待 kvcache 释放 tree cache 的请求
            for decode_req in self.disagg_decode_transfer_queue.queue:
                if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort transfer queue request. {decode_req.req.rid=}")
                    decode_req.kv_receiver.abort()

            # 中止已被回退（retract）到 CPU cache 的请求
            if self.disagg_decode_prealloc_queue.retracted_queue:
                remaining_retracted = []
                for decode_req in self.disagg_decode_prealloc_queue.retracted_queue:
                    if recv_req.abort_all or decode_req.rid.startswith(recv_req.rid):
                        assert hasattr(decode_req, "kv_cache_cpu")
                        del decode_req.kv_cache_cpu
                        self.send_to_tokenizer.send_output(
                            AbortReq(rid=decode_req.rid), decode_req
                        )
                    else:
                        remaining_retracted.append(decode_req)
                self.disagg_decode_prealloc_queue.retracted_queue = remaining_retracted

        # 删除运行批次中的请求
        if self.cur_batch is self.running_batch or self.cur_batch is None:
            reqs = self.running_batch.reqs
        else:
            reqs = self.running_batch.reqs + self.cur_batch.reqs

        for req in reqs:
            if not req.finished() and (
                recv_req.abort_all or req.rid.startswith(recv_req.rid)
            ):
                # 中止方式 3：设置 `to_finish`
                # 该请求仍会运行一次 decode 前向计算。
                # 之后我们复用所有现有代码来清理 KV cache 的分配。
                logger.debug(f"Abort running request. {req.rid=}")
                req.to_finish = FINISH_ABORT()

    def _pause_engine(self) -> Tuple[List[Req], int]:
        """暂停引擎的底层实现（由子类/Mixin 覆写）。"""
        raise NotImplementedError()

    def pause_generation(self, recv_req: PauseGenerationReqInput):
        """暂停生成。支持原地暂停、合并 last_batch、或回退（retract）重新入队等多种模式。"""
        self._engine_paused = True

        if recv_req.mode == "in_place":
            # 原地暂停：只设置标志位并立即返回。
            # 所有调度器状态（running_batch、last_batch、chunked_req、result_queue）
            # 都保持不变。恢复时，正常的事件循环（get_next_batch_to_run）会通过标准代码路径
            # 处理 last_batch 合并、chunked_req 清理以及 overlap 结果处理。
            # 这样可以避免重复实现批次操作逻辑，以及随之而来的计数错误（accounting bug）。
            return

        if self.enable_overlap and self.last_batch:
            # 处理上一个批次的结果
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        if self.last_batch and self.last_batch.forward_mode.is_extend():
            chunked_req_to_exclude = set()
            self.last_batch.filter_batch(
                chunked_req_to_exclude=list(chunked_req_to_exclude)
            )
            if not self.last_batch.is_empty():
                if self.running_batch.is_empty():
                    self.running_batch = self.last_batch
                else:
                    self.running_batch.merge_batch(self.last_batch)

        self.last_batch = None
        self.cur_batch = None

        if recv_req.mode == "retract":
            self.running_batch.filter_batch(v1_spec_info_filtered=True)
            if len(self.running_batch.reqs) != 0:
                retracted_reqs = self.running_batch.retract_all(self.server_args)
                for req in retracted_reqs:
                    self._add_request_to_queue(req)

            self.running_batch.batch_is_full = False
            self.chunked_req = None

    def continue_generation(self, recv_req: ContinueGenerationReqInput):
        """恢复生成（解除暂停标记）。"""
        self._engine_paused = False

    def load_lora_adapter(
        self, recv_req: LoadLoRAAdapterReqInput
    ) -> LoadLoRAAdapterReqOutput:
        """从磁盘或 huggingface 原地加载一个新的 lora adapter。"""

        result = self.tp_worker.load_lora_adapter(recv_req)
        return result

    def load_lora_adapter_from_tensors(
        self, recv_req: LoadLoRAAdapterFromTensorsReqInput
    ) -> LoadLoRAAdapterFromTensorsReqOutput:
        """从序列化的张量原地加载一个新的 lora adapter。"""

        result = self.tp_worker.load_lora_adapter_from_tensors(recv_req)
        return result

    def unload_lora_adapter(
        self, recv_req: UnloadLoRAAdapterReqInput
    ) -> UnloadLoRAAdapterReqOutput:
        """卸载 lora adapter。"""

        result = self.tp_worker.unload_lora_adapter(recv_req)
        return result

    def init_weights_send_group_for_remote_instance(
        self, recv_req: InitWeightsSendGroupForRemoteInstanceReqInput
    ):
        """初始化 seed 实例与 client 实例之间的通信组。"""
        success, message = self.tp_worker.init_weights_send_group_for_remote_instance(
            recv_req
        )
        return InitWeightsSendGroupForRemoteInstanceReqOutput(success, message)

    def send_weights_to_remote_instance(
        self, recv_req: SendWeightsToRemoteInstanceReqInput
    ):
        """将 seed 实例的权重发送给目标实例。"""
        success, message = self.tp_worker.send_weights_to_remote_instance(recv_req)
        return SendWeightsToRemoteInstanceReqOutput(success, message)

    def slow_down(self, recv_req: SlowDownReqInput):
        """设置每次前向后的休眠时间以人为减速（用于调试/限流）。"""
        t = recv_req.forward_sleep_time
        if t is not None and t <= 0:
            t = None
        self.forward_sleep_time = t
        return SlowDownReqOutput()

    def expert_distribution_handle(self, recv_req: ExpertDistributionReq):
        """处理专家分布记录请求：开始/停止/导出记录（用于 MoE 专家负载分析）。"""
        action = recv_req.action
        if action == ExpertDistributionReqType.START_RECORD:
            get_global_expert_distribution_recorder().start_record()
        elif action == ExpertDistributionReqType.STOP_RECORD:
            get_global_expert_distribution_recorder().stop_record()
        elif action == ExpertDistributionReqType.DUMP_RECORD:
            get_global_expert_distribution_recorder().dump_record()
        else:
            raise ValueError(f"Unrecognized ExpertDistributionReq value: {recv_req=}")
        return ExpertDistributionReqOutput()

    def open_session(self, recv_req: OpenSessionReqInput):
        """打开一个会话（用于多轮对话的状态复用）。"""
        return self.session_controller.open(recv_req)

    def close_session(self, recv_req: CloseSessionReqInput):
        """关闭指定会话。"""
        self.session_controller.close(recv_req)

    def maybe_sleep_on_idle(self):
        """若启用了空闲休眠器，则在空闲时休眠以降低 CPU 功耗。"""
        if self.idle_sleeper is not None:
            self.idle_sleeper.maybe_sleep()

    def handle_freeze_gc(self, recv_req: FreezeGCReq):
        """Handle freeze_gc request: freeze scheduler's GC and forward to detokenizer."""
        freeze_gc("Scheduler")
        self.send_to_detokenizer.send_output(recv_req, recv_req)
        return None

    def handle_dumper_control(self, recv_req: DumperControlReqInput):
        """处理 dumper 控制请求（调试转储工具）：仅在 rank 0 上执行并返回响应。"""
        from sglang.srt.debug_utils.dumper import dumper

        try:
            response: list = []
            if (
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ):
                response = dumper._http_manager.handle_request(
                    method=recv_req.method, body=recv_req.body
                )
            self.send_to_tokenizer.send_output(
                DumperControlReqOutput(success=True, response=response), recv_req
            )
        except Exception as e:
            print(f"[Scheduler] handle_dumper_control error: {e}", flush=True)
            self.send_to_tokenizer.send_output(
                DumperControlReqOutput(success=False, response=[], error=str(e)),
                recv_req,
            )

    # placeholder for override
    def update_cache_from_scheduler(
        self, schedule_batch: ScheduleBatch, batch_result: GenerationBatchResult
    ):
        """供子类/Mixin 覆写的占位方法：根据调度结果更新缓存。"""
        pass


class IdleSleeper:
    """
    In setups which have long inactivity periods it is desirable to reduce
    system power consumption when sglang does nothing. This would lead not only
    to power savings, but also to more CPU thermal headroom when a request
    eventually comes. This is important in cases when multiple GPUs are connected
    as each GPU would otherwise pin one thread at 100% CPU usage.

    The simplest solution is to use zmq.Poller on all sockets that may receive
    data that needs handling immediately.
    """

    def __init__(self, sockets):
        """注册需要监听的套接字，并记录上次清理缓存的时间。"""
        self.poller = zmq.Poller()
        self.last_empty_time = real_time()
        for s in sockets:
            self.poller.register(s, zmq.POLLIN)

        self.empty_cache_interval = envs.SGLANG_EMPTY_CACHE_INTERVAL.get()

    def maybe_sleep(self):
        """阻塞轮询套接字（最多 1s）以让出 CPU；并按间隔定期清空 CUDA 缓存。"""
        self.poller.poll(1000)
        if (
            self.empty_cache_interval > 0
            and real_time() - self.last_empty_time > self.empty_cache_interval
        ):
            self.last_empty_time = real_time()
            torch.cuda.empty_cache()


def is_health_check_generate_req(recv_req):
    """判断一个生成请求是否为健康检查请求（通过 rid 前缀识别）。"""
    rid = getattr(recv_req, "rid", None)
    return rid is not None and rid.startswith(HEALTH_CHECK_RID_PREFIX)


def is_work_request(recv_req):
    """判断是否为工作请求（生成/embedding 及其批量版本）。"""
    return isinstance(
        recv_req,
        (
            TokenizedGenerateReqInput,
            TokenizedEmbeddingReqInput,
            BatchTokenizedGenerateReqInput,
            BatchTokenizedEmbeddingReqInput,
        ),
    )


# ZMQ 发送套接字的轻量包装：socket 为 None 时发送操作为空，并处理多 HTTP worker 场景的 IPC 透传。
class SenderWrapper:
    def __init__(self, socket: zmq.Socket):
        self.socket = socket

    def send_output(
        self,
        output: Union[BaseReq, BaseBatchReq],
        recv_obj: Optional[Union[BaseReq, BaseBatchReq]] = None,
    ):
        if self.socket is None:
            return

        if (
            isinstance(recv_obj, BaseReq)
            and recv_obj.http_worker_ipc is not None
            and output.http_worker_ipc is None
        ):
            # handle communicator reqs for multi-http worker case
            output.http_worker_ipc = recv_obj.http_worker_ipc

        self.socket.send_pyobj(output)


def dispatch_event_loop(scheduler: Scheduler):
    """根据部署模式（普通/PD prefill/PD decode）与是否启用 pdmux/PP/overlap，分发到对应的事件循环。"""
    # Dispatch to the appropriate event loop based on the disaggregation mode
    server_args = scheduler.server_args
    disaggregation_mode: DisaggregationMode = scheduler.disaggregation_mode
    if disaggregation_mode == DisaggregationMode.NULL:
        if scheduler.enable_pdmux:
            scheduler.event_loop_pdmux()
        elif server_args.pp_size > 1:
            scheduler.event_loop_pp()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap()
        else:
            scheduler.event_loop_normal()
    elif disaggregation_mode == DisaggregationMode.PREFILL:
        if server_args.pp_size > 1:
            scheduler.event_loop_pp_disagg_prefill()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap_disagg_prefill()
        else:
            scheduler.event_loop_normal_disagg_prefill()
    elif disaggregation_mode == DisaggregationMode.DECODE:
        if server_args.pp_size > 1:
            scheduler.event_loop_pp_disagg_decode()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap_disagg_decode()
        else:
            scheduler.event_loop_normal_disagg_decode()


def configure_scheduler(
    server_args: ServerArgs,
    tp_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
) -> Optional[int]:
    """Configure scheduler worker: logging, process title, etc.

    Returns:
        dp_rank
    """
    # Generate the logger prefix
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    prefix = ""
    if dp_rank is not None:
        prefix += f" DP{dp_rank}"
    if server_args.pp_size > 1:
        prefix += f" PP{pp_rank}"
    if server_args.attn_cp_size > 1:
        prefix += f" ATTN_CP{attn_cp_rank}"
    if server_args.moe_dp_size > 1:
        prefix += f" MOE_DP{moe_dp_rank}"
    if server_args.tp_size > 1:
        prefix += f" TP{tp_rank}"
    if server_args.ep_size > 1:
        prefix += f" EP{moe_ep_rank}"

    # Config the process
    setproctitle.setproctitle(f"sglang::scheduler{prefix.replace(' ', '_')}")
    faulthandler.enable()

    # Configure the logger
    configure_logger(server_args, prefix=prefix)
    suppress_other_loggers()

    return dp_rank


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
):
    """调度器进程的入口函数。

    负责配置进程（日志/进程名/CPU 亲和性/NUMA/追踪），创建 Scheduler 实例，
    通过管道向父进程回传初始化信息，并运行事件循环直到关闭；出错时通知父进程。
    """
    dp_rank = configure_scheduler(
        server_args, tp_rank, attn_cp_rank, moe_dp_rank, moe_ep_rank, pp_rank, dp_rank
    )

    # 当父进程死亡时让本进程也随之退出，避免产生僵尸进程。
    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    # Set cpu affinity to this gpu process
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(
            server_args.pp_size, server_args.tp_size, server_args.nnodes, gpu_id
        )
    numa_node = get_numa_node_if_available(server_args, gpu_id)
    if numa_node is not None and not envs.SGLANG_NUMA_BIND_V2.get():
        numa_bind_to_node(numa_node)

    # Set up tracing
    if server_args.enable_trace:
        process_tracing_init(server_args.otlp_traces_endpoint, "sglang")
        thread_label = "Scheduler"
        if server_args.disaggregation_mode == "prefill":
            thread_label = "Prefill Scheduler"
        elif server_args.disaggregation_mode == "decode":
            thread_label = "Decode Scheduler"
        trace_set_thread_info(thread_label, tp_rank, dp_rank)

    # Create a scheduler and run the event loop
    try:
        scheduler = Scheduler(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            attn_cp_rank,
            moe_dp_rank,
            dp_rank,
        )

        # Send initialization info back to the parent process
        pipe_writer.send(scheduler.get_init_info())

        # Run the event loop (blocks until shutdown)
        scheduler.run_event_loop()

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
