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
"""A scheduler that manages a tensor parallel GPU worker.

中译：Scheduler（调度器）是 SGLang 推理引擎的"心脏"，运行在独立的 GPU 工作进程中，
      负责管理一个张量并行（tensor parallel）的模型 worker。它的核心职责包括：
      1. 通过 ZMQ 从 TokenizerManager 接收已分词的请求，组织成等待队列（waiting_queue）；
      2. 用调度策略（SchedulePolicy）决定每一步前向（forward）跑哪些请求，
         在 prefill（预填充/首 token）与 decode（逐 token 解码）之间权衡，实现连续批处理
         （continuous batching）；
      3. 调用 model worker 执行前向，并把结果（采样出的 token、logprob 等）流式发回
         DetokenizerManager；
      4. 管理 KV 缓存与显存（req_to_token_pool / token_to_kv_pool / radix tree cache），
         在显存吃紧时做请求回退（retract）；
      5. 支持多种高级特性：投机解码（speculative decoding）、PD 分离
         （prefill/decode disaggregation）、流水线并行（pipeline parallelism）、
         数据并行 attention（DP attention）、分层缓存（hierarchical cache）等。

      本文件最关键的难点是"重叠调度（overlap schedule）"：通过让 CPU 端的调度准备
      （下一批的 schedule）与 GPU 端的前向计算（上一批的 forward）在不同的 CUDA stream 上
      并行执行，隐藏 CPU 开销、提升吞吐。对应 event_loop_normal（不重叠）与
      event_loop_overlap（重叠）两个事件循环。
"""

import dataclasses
import faulthandler
import logging
import os
import signal
import sys
import time
from array import array
from collections import deque
from contextlib import contextmanager, nullcontext
from functools import partial
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, Set, Tuple, Union

from sglang.srt.runtime_context import (
    attention_backends,
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
    publish,
)

from sglang.srt.utils.common import suppress_noisy_warnings  # isort: skip

suppress_noisy_warnings()

import psutil  # isort: skip
import setproctitle
import torch
import torch.distributed
from torch.distributed import barrier

if TYPE_CHECKING:
    from torch.cuda import Stream as CudaStream

try:
    from sglang.kernels.ops.mamba.triton_ops import (
        initialize_mamba_selective_state_update_backend,
    )
except ImportError:
    initialize_mamba_selective_state_update_backend = None
from sglang.srt.beam_search.coordinator import BeamCoordinator
from sglang.srt.configs.model_config import (
    ModelConfig,
    ModelImpl,
    is_deepseek_v4,
    is_minimax_sparse,
)
from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.srt.debug_utils.pr_fix_toggle import maybe_revert_pr_fix
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.encoder.receiver import create_mm_receiver
from sglang.srt.disaggregation.prefill import (
    PrefillBootstrapQueue,
    SchedulerDisaggregationPrefillMixin,
    maybe_release_metadata_buffer,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    get_dsa_seed_metadata_dim,
    prepare_abort,
    unified_memory_disagg_move_gate,
)
from sglang.srt.distributed import get_pp_group, get_world_group
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin
from sglang.srt.environ import envs, exportable_env_vars
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.hardware_backend.mlx.runtime import use_mlx
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.layers.quantization.unquant import initialize_bf16_gemm_config
from sglang.srt.lora.lora_drainer import LoRADrainer
from sglang.srt.lora.lora_overlap_loader import LoRAOverlapLoader
from sglang.srt.managers.disagg_service import maybe_create_ascend_config_store
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.io_struct import (
    AbortReq,
    ActiveRanksOutput,
    AddExternalCorpusReqInput,
    AddExternalCorpusReqOutput,
    AttachHiCacheStorageReqInput,
    AttachHiCacheStorageReqOutput,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    CheckWeightsReqInput,
    ClearHiCacheReqInput,
    ClearHiCacheReqOutput,
    CloseSessionReqInput,
    ConfigureLoggingReq,
    ContinueGenerationReqInput,
    DestroyWeightsUpdateGroupReqInput,
    DetachHiCacheStorageReqInput,
    DetachHiCacheStorageReqOutput,
    DumperControlReqInput,
    DumperControlReqOutput,
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    ExpertDistributionReqType,
    FinishReasonDict,
    FlushCacheReqInput,
    FreezeGCReq,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetWeightsByNameReqInput,
    HealthCheckOutput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsSendGroupForRemoteInstanceReqOutput,
    InitWeightsUpdateGroupReqInput,
    ListExternalCorporaReqInput,
    ListExternalCorporaReqOutput,
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterFromTensorsReqOutput,
    LoadLoRAAdapterReqInput,
    LoadLoRAAdapterReqOutput,
    OpenSessionReqInput,
    PauseGenerationReqInput,
    ProfileReq,
    ReleaseMemoryOccupationReqInput,
    RemoveExternalCorpusReqInput,
    RemoveExternalCorpusReqOutput,
    ResumeMemoryOccupationReqInput,
    RpcReqInput,
    RpcReqOutput,
    ScaleElasticEPReqInput,
    ScaleElasticEPReqOutput,
    SendWeightsToRemoteInstanceReqInput,
    SendWeightsToRemoteInstanceReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    ShutdownReq,
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
    UpdateWeightVersionReqInput,
    UpdateWeightVersionReqOutput,
    sock_send,
)
from sglang.srt.managers.load_snapshot import create_load_snapshot_writer
from sglang.srt.managers.min_free_slots_delayer import (
    MinFreeSlotsDelayer,
    resolve_min_free_slots,
)
from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
from sglang.srt.managers.overlap_utils import (
    RelayPayload,
    decide_needs_confidence_relay,
    decide_needs_cpu_seq_lens,
    resolve_forward_inputs,
)
from sglang.srt.managers.prefill_delayer import (
    PrefillDelayer,
    PrefillDelayerSinglePassExecutor,
    RecentPrefillBatchSizeTracker,
)
from sglang.srt.managers.rust_server import RustServer
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    MultimodalInputs,
    NextBatchPlan,
    Req,
    ScheduleBatch,
    retract_all,
)
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    SchedulePolicy,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.dp_attn import SchedulerDPAttnAdapter
from sglang.srt.managers.scheduler_components.flush_wrapper import SchedulerFlushWrapper
from sglang.srt.managers.scheduler_components.idle_sleeper import (
    IdleSleeper,
    RustServerIdleSleeper,
)
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
    create_scheduler_watchdog,
)
from sglang.srt.managers.scheduler_components.ipc_channels import SchedulerIpcChannels
from sglang.srt.managers.scheduler_components.kv_events_publisher import (
    SchedulerKvEventsPublisher,
)
from sglang.srt.managers.scheduler_components.load_inquirer import SchedulerLoadInquirer
from sglang.srt.managers.scheduler_components.load_publisher import (
    SchedulerLoadPublisher,
)
from sglang.srt.managers.scheduler_components.logprob_result_processor import (
    SchedulerLogprobResultProcessor,
)
from sglang.srt.managers.scheduler_components.memory_usage import (
    build_memory_usage,
    combine_graph_memory_usage,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    RECORD_STEP_TIME,
    PrefillStats,
    SchedulerMetricsReporter,
)
from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import (
    NewTokenRatioTracker,
)
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    SchedulerPoolStatsObserver,
)
from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)
from sglang.srt.managers.scheduler_components.recv_skipper import (
    SchedulerRecvSkipper,
)
from sglang.srt.managers.scheduler_components.request_receiver import (
    SchedulerRequestReceiver,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.managers.scheduler_input_blocker import SchedulerInputBlocker
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.managers.utils import (
    EmbeddingBatchResult,
    GenerationBatchResult,
    is_health_check_generate_req,
    validate_input_length,
)
from sglang.srt.mem_cache import kv_cache_builder
from sglang.srt.mem_cache.common import (
    maybe_cache_unfinished_req,
    release_kv_cache,
    retraction_discard,
)
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_loader.utils import get_resolved_model_impl
from sglang.srt.multiplex.multiplexing_mixin import SchedulerMultiplexMixin
from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
from sglang.srt.observability.req_time_stats import (
    flush_trace_batch,
    set_schedule_time_batch,
    set_time_batch,
)
from sglang.srt.observability.startup_time import build_scheduler_startup_time
from sglang.srt.observability.trace import process_tracing_init, trace_set_thread_info
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.platforms import current_platform
from sglang.srt.plugins import load_plugins
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.server_args import PortArgs, ServerArgs, compute_world_size
from sglang.srt.session.session_controller import SessionController
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dflash_utils import validate_dflash_request
from sglang.srt.speculative.eagle_utils import (
    get_draft_recurrent_hidden_state_spec_from_config,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    DynamicGradMode,
    configure_gc_logger,
    configure_logger,
    freeze_gc,
    get_available_gpu_memory,
    get_bool_env_var,
    get_int_env_var,
    is_cuda,
    is_hip,
    is_mps,
    kill_itself_when_parent_died,
    rank_consensus_checker,
    require_mlp_sync,
    set_gpu_proc_affinity,
    set_random_seed,
    suppress_other_loggers,
    triton_load_watch,
)
from sglang.srt.utils.common import is_npu
from sglang.srt.utils.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
    resolve_image_processor_backend,
)
from sglang.srt.utils.msgspec_utils import msgspec_to_builtins
from sglang.srt.utils.numa_utils import get_numa_node_if_available, numa_bind_to_node
from sglang.srt.utils.nvtx_utils import scheduler_nvtx_method
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils.weight_versions import (
    compute_weight_version_spans,
    record_weight_version_events,
)
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

if is_mps():
    CudaStreamContext = nullcontext
    from sglang.srt.hardware_backend.mlx.scheduler_mixin import SchedulerMlxOverlapMixin
else:
    from torch.cuda import StreamContext as CudaStreamContext

    class SchedulerMlxOverlapMixin:
        pass


logger = logging.getLogger(__name__)


def _prewarm_hccl_group(device, group, device_module):
    warmup_tensor = torch.zeros(1, dtype=torch.int32, device=device)
    torch.distributed.all_reduce(warmup_tensor, group=group)
    device_module.synchronize()


# Test retract decode for debugging purposes
# 中译：以下三个开关仅用于调试/测试请求回退（retract）逻辑：
#   TEST_RETRACT 强制触发回退；TEST_RETRACT_INTERVAL 控制每隔多少步触发一次；
#   TEST_RETRACT_NO_PREFILL_BS 在 running batch 超过该大小时停止 prefill，
#   把请求留在等待队列里以便构造回退场景。
TEST_RETRACT = envs.SGLANG_TEST_RETRACT.get()
TEST_RETRACT_INTERVAL = envs.SGLANG_TEST_RETRACT_INTERVAL.get()
TEST_RETRACT_NO_PREFILL_BS = envs.SGLANG_TEST_RETRACT_NO_PREFILL_BS.get()


STEP_MAX_US = 2_000_000

# Min wall-clock between load publishes on the stalled no-batch path, which
# spins on_idle without sleeping. Bounds the O(queue) get_loads for both the
# DP-balancing writer and the router-facing socket.
LOAD_STALL_REFRESH_S = 0.05


def _accumulate_decode_moment(
    totals: list[float],
    batch_size: int,
    step_us: int,
    generated: int,
) -> None:
    if batch_size <= 0 or step_us <= 0:
        return
    b = float(batch_size)
    t = float(step_us)
    g = float(generated)
    totals[0] += 1.0
    totals[1] += b
    totals[2] += t
    totals[3] += b * b
    totals[4] += b * t
    totals[5] += g


_is_npu = is_npu()
_is_hip = is_hip()


class Scheduler(
    SchedulerDisaggregationDecodeMixin,
    SchedulerDisaggregationPrefillMixin,
    SchedulerMultiplexMixin,
    SchedulerPPMixin,
    SchedulerDllmMixin,
    SchedulerMlxOverlapMixin,
):
    """A scheduler that manages a tensor parallel GPU worker.

    中译：调度器主类。通过多重继承把各类能力以 Mixin 形式拼装进来：
          - SchedulerDisaggregationDecodeMixin / SchedulerDisaggregationPrefillMixin：
            PD 分离模式下 decode / prefill 节点专用的事件循环与处理逻辑；
          - SchedulerMultiplexMixin：PD 复用（pdmux）相关；
          - SchedulerPPMixin：流水线并行（pipeline parallel）的微批（microbatch）调度；
          - SchedulerDllmMixin：扩散式 LLM（diffusion LLM）支持；
          - SchedulerMlxOverlapMixin：Apple MLX 后端的重叠循环。
          __init__ 仅作为"编排者"，按固定顺序调用一系列 init_* 辅助方法，每个方法负责一个
          可被子类单独覆盖的初始化单元（见 large-class-init-style 约定）。
    """

    # Class-level default so on_idle's stall gate works even if a fork
    # overrides init_load_publisher (which would otherwise not set it).
    _last_stall_publish_ts: float = float("-inf")

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        dp_rank: Optional[int],
    ):
        # 中译：is_initializing 标记仍在构造中，软看门狗（soft watchdog）线程会读它来
        # NOTE: KEEP THE FOLLOWING CODE STYLE for this function:
        # Keep __init__ as an orchestrator: sequence init_* and maybe_init_* calls
        # with minimal glue. Move substantial component-specific logic into
        # dedicated methods instead of adding inline blocks here.
        self.init_startup_timing_begin()
        self.is_initializing = True
        # init_soft_watchdog starts a daemon thread that reads these on its first tick.
        # 中译：软看门狗守护线程一启动就会读取 forward_ct / cur_batch，所以必须在
        #       init_soft_watchdog 之前先把它们置好初值，否则线程首个 tick 会读到未定义属性。
        self.forward_ct: int = 0
        self.cur_batch_for_debug: Optional[ScheduleBatch] = None
        self.init_soft_watchdog()

        # Parse args
        # 中译：把 server_args 里的大量配置项展开成 self.* 字段，便于后续热路径直接读取。
        self.server_args = server_args
        self.nccl_port = port_args.nccl_port
        self.schedule_policy = get_schedule().schedule_policy
        self.enable_priority_scheduling = get_schedule().enable_priority_scheduling
        self.abort_on_priority_when_disabled = (
            get_schedule().abort_on_priority_when_disabled
        )
        self.schedule_low_priority_values_first = (
            get_schedule().schedule_low_priority_values_first
        )
        self.priority_scheduling_preemption_threshold = (
            get_schedule().priority_scheduling_preemption_threshold
        )
        # 中译：enable_overlap 是"重叠调度"总开关——只要没显式禁用且非 MLX 后端就开启。
        # 中译：spec_algorithm 表示当前使用的投机解码算法（EAGLE / NGRAM / DFLASH / 无 等）。
        self.enable_lora = get_lora().enable_lora
        self.enable_lora_overlap_loading = get_lora().enable_lora_overlap_loading
        self.max_loras_per_batch = get_lora().max_loras_per_batch
        self.enable_overlap = (
            not get_schedule().disable_overlap_schedule and not use_mlx()
        )
        self.enable_overlap_mlx = (
            not get_schedule().disable_overlap_schedule and use_mlx()
        )
        self.enable_pdmux = get_disagg().enable_pdmux
        self.skip_tokenizer_init = get_serving().skip_tokenizer_init
        self.stream_interval = get_serving().stream_interval
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )
        # 中译：page_size 是 KV 缓存按页分配的页大小（每页容纳的 token 数），是显存管理的基本单位。
        self.page_size = get_schedule().page_size
        self.enable_hierarchical_cache = get_memory().enable_hierarchical_cache
        self.enable_session_radix_cache = get_memory().enable_session_radix_cache
        self.enable_hicache_storage = get_memory().hicache_storage_backend is not None
        self.enable_decode_hicache = (
            get_disagg().disaggregation_decode_enable_radix_cache
            and self.enable_hierarchical_cache
        )
        self.max_recv_per_poll = envs.SGLANG_SCHEDULER_MAX_RECV_PER_POLL.get()
        self.max_new_tokens_limit = envs.SGLANG_MAX_NEW_TOKENS_LIMIT.get()
        self.enable_hisparse = get_memory().enable_hisparse
        self.enable_dp_attention = get_parallel().enable_dp_attention
        self.enable_unified_memory = get_memory().enable_unified_memory

        # Distributed rank info
        # 中译：计算 DP attention 下本 rank 在 attention-TP / attention-DP 子组里的编号与规模。
        #       DP attention 允许 attention 部分用数据并行、其它部分用张量并行，需要单独的
        #       world 信息；随后封装进 ParallelState（self.ps）统一管理各种并行维度的 rank。
        attn_tp_rank, attn_tp_size, attn_dp_rank, attn_dp_size = (
            compute_dp_attention_world_info(
                get_parallel().enable_dp_attention,
                tp_rank,
                get_parallel().tp_size,
                get_parallel().dp_size,
                get_parallel().attn_cp_size,
            )
        )
        self.ps = ParallelState(
            tp_rank=tp_rank,
            tp_size=get_parallel().tp_size,
            pp_rank=pp_rank,
            pp_size=get_parallel().pp_size,
            dp_rank=dp_rank,
            dp_size=get_parallel().dp_size,
            attn_tp_rank=attn_tp_rank,
            attn_tp_size=attn_tp_size,
            attn_cp_rank=attn_cp_rank,
            attn_cp_size=get_parallel().attn_cp_size,
            attn_dcp_rank=tp_rank % get_parallel().dcp_size,
            attn_dcp_size=get_parallel().dcp_size,
            attn_dp_rank=attn_dp_rank,
            attn_dp_size=attn_dp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=get_parallel().ep_size,
            moe_dp_rank=moe_dp_rank,
            moe_dp_size=get_parallel().moe_dp_size,
            gpu_id=gpu_id,
        )

        # Init model configs
        # 中译：加载模型配置（HF config、上下文长度、是否生成模型/多模态等）。
        self.init_model_config()

        # Init metrics stats
        # 中译：初始化指标采集器（Prometheus 等），用于上报吞吐、显存、队列长度等监控数据。
        self.init_metrics_collector(tp_rank, pp_rank, dp_rank)

        # Init inter-process communication
        # 中译：建立与 TokenizerManager / DetokenizerManager / RPC 的 ZMQ 通信通道。
        self.init_ipc_channels(port_args)
        # 中译：空闲休眠器——服务器空闲时阻塞在 socket 上休眠，省 CPU。
        self.init_idle_sleeper()

        # 中译：以下四个 PD 分离队列在此先占位为 None，稍后由 init_disaggregation() 根据本节点
        # 中译：【Prefill 侧-队列1】bootstrap 队列——存放刚进入 prefill 节点、正在与 decode 节点
        # 中译：【Prefill 侧-队列2】inflight（在途）队列——存放已算完 KV、正在把 KV 通过 RDMA/
        # 中译：【Decode 侧-队列1】prealloc（预分配）队列——decode 请求生命周期的第一个队列：
        # 中译：【Decode 侧-队列2】transfer（传输）队列——存放已预分配好落点、正在轮询等待 prefill
        # Init ZBAL, switch allocator should before any torch alloc action
        self.init_zbal_on_npu()

        # Init PD-multiplexing context
        if self.enable_pdmux:
            self.init_pdmux()

        # Init tokenizer
        self.init_tokenizer()

        # Init moe config and GEMM config (FP8 GEMM, etc.)
        self.init_moe_gemm_config()

        # Init mamba backend
        self.init_mamba_backend()

        # Must precede init_model_worker: revert targets like _init_pools run during it,
        # so patching them afterwards is a no-op.
        maybe_revert_pr_fix()

        # Launch a model worker and draft model worker if using speculative decoding
        # 中译：启动模型 worker（加载权重）；若启用投机解码还会额外启动 draft（草稿）模型 worker。
        #       这是初始化中最重、最耗显存的一步。
        self.init_model_worker()

        # 中译：测试钩子——按需在初始化中段人为 sleep，用于复现/调试启动卡死。
        if (t := envs.SGLANG_TEST_STUCK_SCHEDULER_INIT.get()) > 0:
            time.sleep(t)

        # Init cache and memory pool
        # 中译：在权重加载完成、显存占用明确后，构建 KV 缓存与各类内存池：
        #       req_to_token_pool（请求->token 槽位映射）、token_to_kv_pool_allocator
        #       （KV 张量分配器）、tree_cache（radix 前缀树缓存，用于前缀复用）。
        #       结果里还包含混合注意力（hybrid SWA/SSM）相关的分层 token 容量信息。
        result = kv_cache_builder.build_kv_cache(
            server_args=self.server_args,
            model_config=self.model_config,
            tp_worker=self.tp_worker,
            page_size=self.page_size,
            spec_algorithm=self.spec_algorithm,
            attn_tp_cpu_group=self.attn_tp_cpu_group,
            tp_cpu_group=self.tp_cpu_group,
            attn_cp_cpu_group=self.attn_cp_cpu_group,
            enable_metrics=get_observability().enable_metrics,
            enable_kv_cache_events=bool(
                get_observability().kv_events_config
                and self.ps.pp_rank == 0
                and self.ps.attn_tp_rank == 0
                and self.ps.attn_cp_rank == 0
            ),
            ps=self.ps,
            tp_group=self.tp_group,
            pp_group=self.pp_group,
            enable_hierarchical_cache=self.enable_hierarchical_cache,
            hicache_draft_plan=(
                self.draft_worker.hicache_draft_plan
                if self.draft_worker is not None
                else None
            ),
        )
        self.is_hybrid_swa = result.is_hybrid_swa
        self.is_hybrid_ssm = result.is_hybrid_ssm
        self.sliding_window_size = result.sliding_window_size
        self.full_tokens_per_layer = result.full_tokens_per_layer
        self.swa_tokens_per_layer = result.swa_tokens_per_layer
        self.req_to_token_pool = result.req_to_token_pool
        self.token_to_kv_pool_allocator = result.token_to_kv_pool_allocator
        self.disable_radix_cache = result.disable_radix_cache
        self.tree_cache = result.tree_cache
        if self.enable_hierarchical_cache:
            cache_controller = self.tree_cache.cache_controller
            if cache_controller is not None:
                cache_controller.load_fence_stream = (
                    self.tp_worker.model_runner.forward_stream
                )
        self.emit_metrics_constants()
        self.maybe_init_hccl_dp_prewarm()

        if (c := self.tp_worker.model_runner.canary_manager) is not None:
            c.attach_radix_cache(self.tree_cache)

        self.init_hisparse_coordinator()

        if (
            get_disagg().disaggregation_mode == "decode"
            and get_disagg().disaggregation_decode_enable_offload_kvcache
        ):
            self.decode_offload_manager = DecodeKVCacheOffloadManager(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                tp_group=(
                    self.attn_tp_cpu_group
                    if self.enable_dp_attention
                    else self.tp_cpu_group
                ),
                tree_cache=self.tree_cache,
            )
        else:
            self.decode_offload_manager = None

        # Init running status
        # 中译：初始化运行时状态机字段：等待队列 waiting_queue、运行中批次 running_batch、
        #       当前/上一批 cur_batch/last_batch、forward_ct 计数等（详见 init_running_status）。
        self.init_running_status()

        # Init chunked prefill
        # 中译：初始化分块预填充（chunked prefill）——把超长 prompt 拆成多个 chunk 分步 prefill，
        #       避免单步占满显存、并能与 decode 混合（mixed chunk）。
        self.init_chunked_prefill()

        # Init diffusion LLM
        self.init_diffusion_llm()

        self.init_metrics_reporter(tp_rank, pp_rank, dp_rank)

        # Init schedule policy and new token estimation
        # 中译：初始化调度策略（FCFS/LPM 等）与"新增 token 比例"估计器
        #       （new_token_ratio，用于预估 decode 阶段还要消耗多少 KV，指导是否接纳新 prefill）。
        self.init_schedule_policy()

        # Init watchdog, memory saver, input blocker and recv skipper
        # 中译：初始化硬看门狗（卡死则杀进程）、显存节省器、输入阻断器、接收跳过器等运维组件。
        self.init_watch_dog_memory_saver_input_blocker()

        # Init profiler
        self.init_profiler()

        # Start the embedded Rust frontend (rank 0). Must precede
        # init_disaggregation: on PD prefill the rust api listener also serves
        # the KV bootstrap registry, and the KVManagers built there register to
        # it synchronously. (The listener is bound synchronously inside launch,
        # so the registry is accepting once this returns.) Must also precede
        # the request receiver, which reads self.recv_from_tokenizer to pick
        # its ingress transport.
        self.maybe_init_rust_server()

        # Init prefill-decodedisaggregation
        # 中译：初始化 PD 分离（prefill/decode 拆到不同实例）所需的队列与元数据缓冲区。
        self.init_disaggregation()

        # Init overlap schedule
        # 中译：初始化重叠调度所需的 CUDA stream（forward_stream / copy_stream）与
        #       FutureMap（用于在两步之间中继 next-token 等"未来值"）。
        self.init_overlap()

        # Init Ngram Embedding
        self.maybe_init_ngram_embedding()

        # Init prefill kv split size when deterministic inference is enabled with various attention backends
        self.init_deterministic_inference_config()

        self.init_weight_updater()

        # Init request dispatcher
        # 中译：构建"按消息类型路由"的分发器，把收到的各种请求对象映射到对应 handler。
        self.init_request_dispatcher()

        # Init LoRA drainer for fair scheduling
        self.init_lora_drainer()

        # Init LoRA overlap loader
        self.init_lora_overlap_loader()

        # Init the grammar backend for constrained generation
        self.init_grammar_manager()

        self.maybe_init_scripted_scheduler_hook()

        self.init_request_receiver()

        self.init_dp_attn_adapter()

        self.init_pool_stats_observer()

        self.init_invariant_checker()

        self.init_kv_events_publisher()

        self.init_load_publisher()

        self.init_load_inquirer()

        self.init_output_streamer()

        self.init_batch_result_processor()

        # 中译：所有 init_* 完成，构造结束，清除初始化标记（看门狗自此按运行态判定卡死）。
        self.init_rank_consensus_checker()

        self.is_initializing = False
        self.init_startup_timing_summary()

    def init_startup_timing_begin(self) -> None:
        self.scheduler_startup_begin = time.perf_counter()

    def init_startup_timing_summary(self) -> None:
        self.startup_time = build_scheduler_startup_time(
            target_load_weight=self.tp_worker.weight_load_time,
            draft_load_weight=(
                0.0 if self.draft_worker is None else self.draft_worker.weight_load_time
            ),
            kv_cache_allocation=self.kv_cache_allocation_time,
            scheduler_e2e=time.perf_counter() - self.scheduler_startup_begin,
            target_cuda_graph=self.tp_worker.graph_time_usage,
            draft_cuda_graph=(
                None
                if self.draft_worker is None
                else self.draft_worker.graph_time_usage
            ),
        )

    def maybe_init_hccl_dp_prewarm(self) -> None:
        if not (
            _is_npu
            and is_deepseek_v4(self.tp_worker.model_runner.model_config.hf_config)
        ):
            return

        rank = (
            self.ps.dp_rank
            if self.ps.dp_rank is not None
            else self.tp_group.rank_in_group
        )
        logger.info("HCCL DP prewarm start: rank=%s", rank)
        _prewarm_hccl_group(
            device=self.tp_group.device,
            group=self.tp_group.device_group,
            device_module=self.tp_group.device_module,
        )
        logger.info("HCCL DP prewarm done: rank=%s", rank)

    def init_zbal_on_npu(self):
        if _is_npu:
            from sglang.srt.hardware_backend.npu.utils import init_zbal

            if self.ps.pp_size > 1:
                logger.error("only zbal mix mode support pp_size > 1!")
            init_zbal(
                self.ps.tp_size, self.ps.gpu_id, self.ps.tp_rank
            )  # only switch allocator if is mix mode

    def init_model_config(self):
        self.model_config = ModelConfig.from_server_args(self.server_args)
        if _is_npu:
            # make sure the page size is not larger than block_size and chunked_prefill_size on NPU backend
            # the npu backend request the defined page size to be no larger than block_size and chunked_prefill_size
            from sglang.srt.dllm.config import DllmConfig

            self.dllm_config = (  # For diffusion LLM
                DllmConfig.from_server_args(self.server_args)
                if get_exec().dllm.dllm_algorithm is not None
                else None
            )

    def init_metrics_collector(
        self, tp_rank: int, pp_rank: int, dp_rank: Optional[int]
    ) -> None:
        self.metrics_collector_context = SchedulerMetricsCollector.init_new(
            server_args=self.server_args,
            ps=self.ps,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank,
            enable_priority_scheduling=self.enable_priority_scheduling,
            enable_lora=self.enable_lora,
            enable_hierarchical_cache=self.enable_hierarchical_cache,
        )
        self.metrics_collector = self.metrics_collector_context.collector

    def init_ipc_channels(self, port_args: PortArgs):
        is_rank_zero = (
            self.ps.pp_rank == 0
            and self.ps.attn_tp_rank == 0
            and self.ps.attn_cp_rank == 0
        )
        self.ipc_channels = SchedulerIpcChannels.create(
            port_args=port_args,
            is_rank_zero=is_rank_zero,
            # The snapshot taken at construction, not a second bag read: this
            # scheduler gates its tokenizer init on the same value, and the two
            # must not be able to disagree.
            skip_tokenizer_init=self.skip_tokenizer_init,
            metrics_enabled=get_observability().enable_metrics
            and (
                self.ps.attn_tp_rank == 0
                or get_observability().enable_metrics_for_all_schedulers
            ),
            enable_scripted_runtime=envs.SGLANG_TEST_SCRIPTED_RUNTIME.get(),
        )

        self.load_snapshot_writer = None
        self.recv_from_tokenizer = None

        if not is_rank_zero:
            return

        self.recv_from_tokenizer = self.ipc_channels.recv_from_tokenizer
        dp_rank = self.ps.dp_rank if self.ps.dp_rank is not None else 0
        try:
            self.load_snapshot_writer = create_load_snapshot_writer(
                port_args,
                self.ps.dp_size,
                dp_rank,
                publish_interval=get_observability().load_snapshot_publish_interval,
            )
        except Exception as e:
            logger.warning("load snapshot writer init failed: %s", e)

    def init_idle_sleeper(self) -> None:
        if (
            self.ps.pp_rank == 0
            and self.ps.attn_tp_rank == 0
            and self.ps.attn_cp_rank == 0
            and get_device().sleep_on_idle
        ):
            self.idle_sleeper = IdleSleeper(
                sockets=[
                    self.ipc_channels.recv_from_tokenizer,
                    self.ipc_channels.recv_from_rpc,
                ],
            )
        else:
            self.idle_sleeper = None

    def publish_load_snapshot(self, force: bool = False):
        """Returns the LoadSnapshot it published, or None when disabled,
        throttled, or failed — so co-located sinks (the router-facing load
        publisher) can reuse it instead of walking the queues again."""
        writer = self.load_snapshot_writer
        if writer is None:
            return None
        if not force:
            writer.publish_counter += 1
            if writer.publish_counter < writer.publish_interval:
                return None
        writer.publish_counter = 0
        try:
            load = self.load_inquirer.get_loads()
            writer.write(load)
            return load
        except Exception as e:
            logger.warning("load snapshot publish failed: %s", e)
            return None

    def init_tokenizer(self):
        # 中译：初始化分词器/处理器。Scheduler 侧也保留 tokenizer 主要用于：停止符判定、
        #       reasoning_parser 的 think_end token、多模态 M-RoPE 回退计算等。
        #       skip_tokenizer_init 时置为 None（调用方自行处理 token）。
        server_args = self.server_args
        self.is_generation = self.model_config.is_generation

        if self.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    get_serving().tokenizer_path,
                    tokenizer_mode=get_serving().tokenizer_mode,
                    trust_remote_code=get_model().trust_remote_code,
                    revision=get_model().revision,
                    image_processor_backend=resolve_image_processor_backend(get_mm()),
                    tokenizer_backend=get_serving().tokenizer_backend,
                    model_name=get_model().model_path,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                self.tokenizer = get_tokenizer(
                    get_serving().tokenizer_path,
                    tokenizer_mode=get_serving().tokenizer_mode,
                    trust_remote_code=get_model().trust_remote_code,
                    revision=get_model().revision,
                    tokenizer_backend=get_serving().tokenizer_backend,
                )

        # Load multimodal processor for M-RoPE fallback computation.
        self._mm_processor = None
        if (
            self.model_config.is_multimodal
            and self.processor is not None
            and not get_disagg().language_model_only
        ):
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

        if get_serving().reasoning_parser and self.tokenizer:
            reasoning_parser = ReasoningParser(
                model_type=get_serving().reasoning_parser,
                stream_reasoning=False,
                tokenizer=self.tokenizer,
            )
            think_end_ids = self.tokenizer.encode(
                reasoning_parser.detector.think_end_token, add_special_tokens=False
            )
            if think_end_ids:
                self.model_config.think_end_ids = think_end_ids
            else:
                logger.warning(
                    "Reasoning parser think_end_token %r could not be encoded; "
                    "grammar-gated reasoning is disabled.",
                    reasoning_parser.detector.think_end_token,
                )

    def init_mamba_backend(self) -> None:
        if initialize_mamba_selective_state_update_backend is not None:
            initialize_mamba_selective_state_update_backend(self.server_args)

    def init_moe_gemm_config(self):
        # For the MM models, check the text_config for MoE settings
        config_to_check = getattr(
            self.model_config.hf_config, "text_config", self.model_config.hf_config
        )

        # Different MoE architectures expose the per-token expert count under
        # different attribute names (e.g. Gemma4 uses ``top_k_experts``,
        # LongCat-2.0 uses ``moe_topk``).
        moe_topk_attrs = (
            "num_experts_per_tok",
            "num_experts_per_token",
            "top_k_experts",
            "moe_top_k",
            "moe_topk",
        )
        if any(hasattr(config_to_check, attr) for attr in moe_topk_attrs):
            initialize_moe_config()

        # Initialize GEMM-related configuration for FP8 and FP4 backends.
        initialize_fp8_gemm_config()
        initialize_fp4_gemm_config()
        initialize_bf16_gemm_config(self.server_args)

        # This must be called after initialize_moe_config
        self.require_mlp_sync = require_mlp_sync()

    def init_tp_model_worker(self):
        worker_kwargs = dict(
            server_args=self.server_args,
            gpu_id=self.ps.gpu_id,
            ps=self.ps,
            nccl_port=self.nccl_port,
        )

        # FIXME: move tp worker's init logic outside of the scheduler.
        if use_mlx():
            from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker

            self.tp_worker = MlxTpModelWorker(**worker_kwargs)
        else:
            from sglang.srt.managers.tp_worker import TpModelWorker

            self.tp_worker = TpModelWorker(**worker_kwargs)

    def maybe_init_draft_worker(self):
        if self.spec_algorithm.is_none():
            self.draft_worker = None
            self.external_corpus_manager = None
            return

        # Launch a draft worker for speculative decoding. It builds its draft
        # from this process's own config: what differs for the draft — the
        # target's context length, the draft load format, its attention backend
        # — is resolved per runner, not on a config copy.
        draft_worker_kwargs = dict(
            server_args=self.server_args,
            gpu_id=self.ps.gpu_id,
            ps=self.ps,
            nccl_port=self.nccl_port,
            target_worker=self.tp_worker,
        )

        DraftWorkerClass = self.spec_algorithm.create_worker(self.server_args)
        self.draft_worker = DraftWorkerClass(**draft_worker_kwargs)

        if self.spec_algorithm.is_ngram():
            from sglang.srt.speculative.external_corpus_manager import (
                ExternalCorpusManager,
            )

            self.external_corpus_manager = ExternalCorpusManager(
                self.draft_worker,
                self.ipc_channels.send_to_tokenizer.send_output,
            )
        else:
            self.external_corpus_manager = None

    def init_target_memory_pool(self):
        """Allocate target KV cache pools if they have not been allocated yet."""
        if (
            self.tp_worker.model_runner.memory_pool_config is not None
            and self.tp_worker.model_runner.req_to_token_pool is not None
            and self.tp_worker.model_runner.token_to_kv_pool_allocator is not None
        ):
            return
        preloaded_weights_bytes = self.tp_worker.preloaded_weights_bytes
        if self.draft_worker is not None:
            preloaded_weights_bytes += self.draft_worker.preloaded_weights_bytes
        self.tp_worker.model_runner.account_preloaded_weights(preloaded_weights_bytes)
        self.tp_worker.alloc_memory_pool()

    def init_memory_pools(self):
        """Allocate KV cache pools for target and draft workers."""
        self.init_target_memory_pool()
        # Lands the retraction backend on the disagg bag before the draft
        # worker's HiCache plan reads it.
        kv_cache_builder.resolve_decode_retraction_backup(tp_worker=self.tp_worker)
        if self.draft_worker is not None:
            pool, allocator = self.tp_worker.get_memory_pool()
            self.draft_worker.alloc_memory_pool(
                memory_pool_config=self.tp_worker.model_runner.memory_pool_config,
                req_to_token_pool=pool,
                token_to_kv_pool_allocator=allocator,
            )
            self.draft_worker.init_hicache_draft_plan()

    def init_all_attention_backends(self):
        """Initialize attention backends for all workers."""
        self.tp_worker.init_attention_backends()
        if self.draft_worker is not None:
            self.draft_worker.init_attention_backends()

    def init_all_cuda_graphs(self):
        """Capture cuda graphs for all workers."""
        self.tp_worker.init_cuda_graphs()
        if self.draft_worker is not None:
            self.draft_worker.init_cuda_graphs()

    def init_model_worker(self):
        # 中译：初始化模型 worker 的完整流程：加载 TP worker 权重 -> 按需建 draft worker ->
        #       分配 KV 内存池 -> 初始化 attention 后端并 capture CUDA graph ->
        #       从 worker 取回 max_total_num_tokens、device、forward_stream 等关键运行参数。
        # Load model weights.
        self.init_tp_model_worker()
        if self.server_args.is_startup_weight_load_overlap:
            self.tp_worker.start_startup_weight_load()
        self.maybe_init_draft_worker()

        # Prepare KV cache pools for all workers
        tic = time.perf_counter()
        self.init_memory_pools()
        self.kv_cache_allocation_time = time.perf_counter() - tic

        self.init_all_attention_backends()
        self.init_all_cuda_graphs()

        model_runner = self.tp_worker.model_runner
        if model_runner.token_to_kv_pool.post_capture_active:
            tic = time.perf_counter()
            model_runner.post_capture_resize_kv_pool()
            self.kv_cache_allocation_time += time.perf_counter() - tic

        if self.server_args.is_startup_weight_load_overlap:
            self.tp_worker.finalize_startup_weight_load()

        if (
            get_exec().moe.elastic_ep_backend is not None
            and get_exec().moe.ep_join_mode == "recover"
        ):
            model_runner.post_capture_elastic_ep_recover()

        # Dispatch the model worker
        if self.spec_algorithm.is_none():
            self.model_worker = self.tp_worker
        else:
            self.model_worker = self.draft_worker

        # Get token and memory info from the model worker
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
        # DFlash auto-enables the legacy formula; other workloads opt in via
        # --min-free-slots-delay. Built independently of the prefill delayer.
        self.min_free_slots_delayer: Optional[MinFreeSlotsDelayer] = None
        min_free_slots = resolve_min_free_slots(
            get_schedule().min_free_slots_delay,
            self.max_running_requests,
            is_dflash_family=self.spec_algorithm.is_dflash_family(),
        )
        if min_free_slots is not None:
            self.min_free_slots_delayer = MinFreeSlotsDelayer(
                min_free_slots=min_free_slots
            )
        if not get_parallel().pp_max_micro_batch_size:
            get_context().override(
                "scheduler.pp_max_micro_batch_size_default",
                pp_max_micro_batch_size=max(
                    self.max_running_requests // self.ps.pp_size, 1
                ),
            )

        self.tp_group = get_tp_group()
        self.tp_cpu_group = self.tp_group.cpu_group
        self.attn_tp_group = get_parallel().attn_tp_group
        self.attn_tp_cpu_group = self.attn_tp_group.cpu_group
        self.attn_cp_group = get_parallel().attn_cp_group
        self.attn_cp_cpu_group = self.attn_cp_group.cpu_group
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()

        # NOTE: dp_tp_* are request/data-plane coordination groups (not tensor collectives).
        # When DP attention is enabled, scope to the attention-TP group; otherwise use
        # the base TP group. Entry rank is the local rank 0 in that group.
        # Use the CPU (gloo) group to broadcast VLM Python objects and avoid CUDA
        # stream/device coupling (#11910).
        self.dp_tp_group = (
            self.attn_tp_group if self.enable_dp_attention else self.tp_group
        )
        self.dp_tp_cpu_group = self.dp_tp_group.cpu_group

        # TODO(Jialin): Migrate pad_input_ids implementations to return array.
        self.pad_input_ids_func = self.tp_worker.get_pad_input_ids_func()
        set_random_seed(self.random_seed)

        # Print debug info
        self.startup_available_gpu_memory_gb = get_available_gpu_memory(
            self.device, self.ps.gpu_id, empty_cache=False
        )
        if self.ps.tp_rank == 0:
            logger.info(
                f"max_total_num_tokens={self.max_total_num_tokens}, "
                f"chunked_prefill_size={get_schedule().chunked_prefill_size}, "
                f"max_prefill_tokens={self.max_prefill_tokens}, "
                f"max_running_requests={self.max_running_requests}, "
                f"context_len={self.model_config.context_len}, "
                f"{'available_cpu_mem' if self.device == 'cpu' else 'available_gpu_mem'}="
                f"{self.startup_available_gpu_memory_gb:.2f} GB"
            )

    def emit_metrics_constants(self) -> None:
        if not get_observability().enable_metrics:
            return

        self.metrics_collector.emit_constants(
            max_total_num_tokens=self.max_total_num_tokens,
            max_total_num_tokens_swa=self.swa_tokens_per_layer,
            weight_memory_usage_gb=self.tp_worker.model_runner.weight_load_mem_usage,
            kv_cache_memory_usage_gb=(
                self.token_to_kv_pool_allocator.get_kvcache().mem_usage
            ),
            graph_memory_usage_gb=combine_graph_memory_usage(
                self.tp_worker.graph_memory_usage,
                (
                    None
                    if self.draft_worker is None
                    else self.draft_worker.graph_memory_usage
                ),
            ),
            # TODO: max_running_requests_under_SLO has no setter — dead chain.
            max_running_requests_under_SLO=None,
            page_size=self.page_size,
            num_pages=self.max_total_num_tokens // self.page_size,
            context_len=self.model_config.context_len,
            startup_available_gpu_memory_gb=self.startup_available_gpu_memory_gb,
        )

    def init_hisparse_coordinator(self) -> None:
        self.hisparse_coordinator: Optional[HiSparseCoordinator] = None
        if not self.enable_hisparse:
            return

        # Coordinator was created inside ModelRunner.initialize() before CUDA graph capture.
        self.hisparse_coordinator = self.tp_worker.model_runner.hisparse_coordinator
        self.hisparse_coordinator.set_decode_producer_stream(self.forward_stream)

    def init_running_status(self):
        # 中译：初始化调度器的核心运行时状态。这几个字段构成了连续批处理（continuous batching）
        # 中译：waiting_queue——等待被调度（尚未跑前向）的请求队列。
        # Set by the ShutdownReq handler to break the event loop for graceful shutdown.
        self.gracefully_exit = False
        self.waiting_queue: List[Req] = []
        # The running decoding batch for continuous batching
        # 中译：running_batch——当前正在持续解码的批次，连续批处理的主体。
        self.running_batch: ScheduleBatch = ScheduleBatch(reqs=[], batch_is_full=False)
        # The current forward batch
        # 中译：cur_batch——本次循环迭代实际要跑前向的批次（可能是 prefill 或 decode）。
        self.cur_batch_for_debug: Optional[ScheduleBatch] = None
        # The last forward batch
        # 中译：last_batch——上一轮迭代跑过的批次；重叠调度里它的结果会延后到本轮处理。
        self.last_batch: Optional[ScheduleBatch] = None
        # 中译：forward_ct——累计前向次数，用于看门狗判活、测试钩子、定期任务等。
        self.forward_ct = 0
        self.return_health_check_ipcs: Deque[Optional[str]] = deque()
        self.flush_wrapper = SchedulerFlushWrapper(
            flush_cache=self.flush_cache,
            is_fully_idle=self.is_fully_idle,
            ipc_channels=self.ipc_channels,
        )
        self._last_logged_elastic_radix_namespace: Optional[str] = None
        self.session_controller = SessionController(self.tree_cache)
        self.forward_sleep_time = None
        self._engine_paused = False

    def init_chunked_prefill(self):
        self.chunked_prefill_size = get_schedule().chunked_prefill_size
        self.prefill_decode_interval = get_schedule().prefill_decode_interval
        self._prefill_decode_interval_remaining = 0
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
        self._pending_chunked_abort_req = None
        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None and get_schedule().enable_mixed_chunk
        )

        # Init the dynamic chunking predictor for PP
        self.enable_dynamic_chunking = (
            get_schedule().enable_dynamic_chunking and self.ps.pp_size > 1
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

    def _should_defer_prefill(self) -> bool:
        if self._prefill_decode_interval_remaining == 0:
            return False

        self._prefill_decode_interval_remaining -= 1
        return True

    def _arm_prefill_decode_interval(self, batch: Optional[ScheduleBatch]) -> None:
        if self.prefill_decode_interval == 0 or batch is None:
            return

        # DP attention synchronizes this flag across ranks. This keeps every
        # rank on the same prefill/decode cadence even when only one rank has
        # local prefill work. Non-DP scheduling can use the local mode directly.
        is_extend = (
            batch.is_extend_in_batch
            if self.require_mlp_sync
            else batch.forward_mode.is_extend()
        )
        if is_extend:
            self._prefill_decode_interval_remaining = self.prefill_decode_interval

    def init_metrics_reporter(
        self, tp_rank: int, pp_rank: int, dp_rank: Optional[int]
    ) -> None:
        # Override point for deployments that need a specialized reporter.
        self.metrics_reporter = SchedulerMetricsReporter(
            scheduler=self,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank,
            metrics_collector_context=self.metrics_collector_context,
            metrics_collector=self.metrics_collector,
        )

    def init_schedule_policy(self):
        # Init schedule policy and new token estimation
        self.policy = SchedulePolicy(
            self.schedule_policy,
            self.tree_cache,
            self.enable_hierarchical_cache,
            self.enable_priority_scheduling,
            self.schedule_low_priority_values_first,
        )
        self.prefill_delayer: Optional[PrefillDelayer] = None
        self.prefill_bs_tracker = RecentPrefillBatchSizeTracker(
            window_size=envs.SGLANG_PREFILL_DELAYER_MAX_PREFILL_BS_WINDOW_SIZE.get()
        )
        self.max_prefill_bs: int = 0
        if get_schedule().enable_prefill_delayer:
            if get_disagg().disaggregation_mode == "decode":
                logger.info(
                    "Ignoring --enable-prefill-delayer on decode engine "
                    "(no prefill scheduling path; delayer would be a no-op)."
                )
            else:
                self.prefill_delayer = PrefillDelayer(
                    dp_size=self.ps.dp_size,
                    attn_tp_size=self.ps.attn_tp_size,
                    cpu_group=self.tp_cpu_group,
                    device_group=self.tp_group.device_group,
                    metrics_collector=(
                        self.metrics_collector
                        if self.metrics_reporter.enable_metrics
                        else None
                    ),
                    max_delay_passes=get_schedule().prefill_delayer_max_delay_passes,
                    token_usage_low_watermark=get_schedule().prefill_delayer_token_usage_low_watermark,
                    device=self.tp_group.device,
                    debug_log_enabled=self.ps.attn_tp_rank == 0,
                )

        # NOTE: preemption is enabled by default for priority scheduling.
        self.enable_priority_preemption = (
            self.enable_priority_scheduling
            and not get_schedule().disable_priority_preemption
        )

        self.new_token_ratio_tracker = NewTokenRatioTracker.from_config()

    def init_soft_watchdog(self):
        if (x := get_device().soft_watchdog_timeout) is not None:
            self.soft_watchdog = create_scheduler_watchdog(
                self, watchdog_timeout=x, soft=True
            )

    def init_watch_dog_memory_saver_input_blocker(self):
        # Start watchdog thread
        self.watchdog = create_scheduler_watchdog(
            self, watchdog_timeout=get_device().watchdog_timeout
        )

        # Init memory saver, profiler and metric stats
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=get_exec().features.enable_memory_saver
        )

        # Init recv skipper and input blocker
        self.recv_skipper = SchedulerRecvSkipper.maybe_create()
        self.input_blocker = (
            SchedulerInputBlocker(noop=self.ps.attn_tp_rank != 0)
            if get_bool_env_var("SGLANG_ENABLE_COLOCATED_BATCH_GEN")
            else None
        )

        # Configure GC logger
        if envs.SGLANG_LOG_GC.get():
            configure_gc_logger()

    def init_disaggregation(self):
        # 中译：初始化 PD 分离（prefill/decode 拆到不同实例，用 RDMA/Mooncake 等传输 KV）。
        self.mm_receiver = None
        self.disagg_prefill_bootstrap_queue = None
        self.disagg_prefill_inflight_queue = None
        self.disagg_decode_prealloc_queue = None
        self.disagg_decode_transfer_queue = None

        self.disaggregation_mode = DisaggregationMode(get_disagg().disaggregation_mode)
        self.transfer_backend = TransferBackend(
            get_disagg().disaggregation_transfer_backend
        )

        # In rust-server mode the KV bootstrap registry is already serving on
        # the rust api listener (maybe_init_rust_server runs before this
        # method — the PrefillBootstrapQueue's KVManager below registers to it
        # synchronously, and a failed registration only retries ~60s then logs,
        # leaving every PD request unroutable). Only the ascend config store,
        # which start_disagg_service would otherwise create, is left to do.
        if (
            self.disaggregation_mode == DisaggregationMode.PREFILL
            and self._hosts_rust_server()
        ):
            maybe_create_ascend_config_store(
                transfer_backend=self.transfer_backend,
            )

        draft_token_to_kv_pool = (
            self.draft_worker.primary_draft_kv_pool
            if self.draft_worker is not None
            else None
        )

        if self.spec_algorithm.carries_draft_hidden_states():
            # Derive the rank-uniform PD wire schema from config because only the
            # last prefill PP stage owns a draft runner.
            draft_model_config = ModelConfig.from_server_args(
                self.server_args,
                model_path=get_spec().speculative_draft_model_path,
                model_revision=get_spec().speculative_draft_model_revision,
                is_draft_model=True,
            )
            disagg_hidden_size, disagg_hidden_states_dtype = (
                get_draft_recurrent_hidden_state_spec_from_config(
                    draft_model_config, self.spec_algorithm
                )
            )
        else:
            disagg_hidden_size = 16  # minimal padding size for RDMA
            disagg_hidden_states_dtype = torch.float32

        # The PD metadata wire schema must match on P and D even when only D
        # enables spec decoding; a seedless prefill writes the invalid sentinel.
        output_dsa_topk_indices_dim = get_dsa_seed_metadata_dim(
            self.model_config.hf_config
        )

        if (
            self.disaggregation_mode == DisaggregationMode.DECODE
        ):  # *8 headroom for MiniMax-M3; *2 for other models.
            buffer_multiplier = (
                8 if is_minimax_sparse(self.model_config.hf_config) else 2
            )
            buffer_size = (self.req_to_token_pool.size) * buffer_multiplier
            self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                buffer_size
            )
            self.disagg_metadata_buffers = MetadataBuffers(
                buffer_size,
                hidden_size=disagg_hidden_size,
                hidden_states_dtype=disagg_hidden_states_dtype,
                custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
                output_dsa_topk_indices_dim=output_dsa_topk_indices_dim,
            )

            # The decode requests polling kv cache
            self.disagg_decode_transfer_queue = DecodeTransferQueue(
                gloo_group=self.attn_tp_cpu_group,
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                tp_rank=self.ps.tp_rank,
                metadata_buffers=self.disagg_metadata_buffers,
                scheduler=self,
                tree_cache=self.tree_cache,
            )

            # The decode requests pending for pre-allocation
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
                tp_rank=self.ps.tp_rank,
                tp_size=self.ps.tp_size,
                dp_size=get_parallel().dp_size,
                gpu_id=self.ps.gpu_id,
                bootstrap_port=get_disagg().disaggregation_bootstrap_port,
                max_total_num_tokens=self.max_total_num_tokens,
                pp_rank=self.ps.pp_rank,
                num_reserved_decode_tokens=get_disagg().num_reserved_decode_tokens,
                transfer_backend=self.transfer_backend,
            )

        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            # *2 for the headroom.
            buffer_size = self.max_running_requests * 2
            self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                buffer_size
            )
            self.disagg_metadata_buffers = MetadataBuffers(
                buffer_size,
                hidden_size=disagg_hidden_size,
                hidden_states_dtype=disagg_hidden_states_dtype,
                custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
                output_dsa_topk_indices_dim=output_dsa_topk_indices_dim,
            )

            self.disagg_prefill_bootstrap_queue = PrefillBootstrapQueue(
                token_to_kv_pool=self.token_to_kv_pool_allocator.get_kvcache(),
                draft_token_to_kv_pool=draft_token_to_kv_pool,
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                metadata_buffers=self.disagg_metadata_buffers,
                tp_rank=self.ps.tp_rank,
                tp_size=self.ps.tp_size,
                gpu_id=self.ps.gpu_id,
                bootstrap_port=get_disagg().disaggregation_bootstrap_port,
                gloo_group=self.attn_tp_cpu_group,
                max_total_num_tokens=self.max_total_num_tokens,
                scheduler=self,
                pp_rank=self.ps.pp_rank,
                pp_size=self.ps.pp_size,
                transfer_backend=self.transfer_backend,
            )
            # The prefill requests that are in the middle of kv sending
            self.disagg_prefill_inflight_queue: List[Req] = []
            # Requests with a sent chunk that are not yet on the inflight queue.
            self.disagg_prefill_pending_chunk_rids: Set[str] = set()

            self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()

        if (
            self.enable_unified_memory
            and self.disaggregation_mode != DisaggregationMode.NULL
        ):
            self.token_to_kv_pool_allocator.set_disagg_move_gate(
                unified_memory_disagg_move_gate(self)
            )

        # Init mm receiver for EPD disaggregation mode
        if get_disagg().language_only and get_disagg().encoder_transfer_backend in [
            "zmq_to_scheduler",
            "mooncake",
        ]:
            self.mm_receiver = create_mm_receiver(
                self.server_args,
                dtype=self.model_config.dtype,
                hf_config=self.model_config.hf_config,
                pp_rank=self.ps.pp_rank,
                tp_rank=self.ps.tp_rank,
                tp_group=self.tp_group,
                scheduler=self,
            )

    def init_overlap(self):
        # 中译：初始化重叠调度基础设施。即使没开 enable_overlap，FutureMap 也始终创建，
        #       因为非重叠路径同样依赖它在迭代间中继 decode 的 input_ids。
        self.device_module = torch.get_device_module(self.device)

        # FutureMap is always-on: input_ids relay used in both modes.
        # Workers without the spec_v2_attn_backends override fall back to
        # target-only so the helper still produces a safe decision (no
        # accidental opt-out for unaudited shapes).
        if self.draft_worker is not None:
            attn_backends = getattr(
                self.draft_worker,
                "spec_v2_attn_backends",
                (self.tp_worker.model_runner.attn_backend,),
            )
        else:
            attn_backends = (self.tp_worker.model_runner.attn_backend,)
        needs_cpu_seq_lens = decide_needs_cpu_seq_lens(attn_backends)
        needs_confidence_relay = decide_needs_confidence_relay()
        self.future_map = self.spec_algorithm.create_future_map(
            self.device,
            self.req_to_token_pool,
            needs_cpu_seq_lens=needs_cpu_seq_lens,
            needs_confidence_relay=needs_confidence_relay,
        )

        self._confidence_budget_prepare = None
        if (
            needs_confidence_relay
            and self.enable_overlap
            and self.draft_worker is not None
        ):
            self._confidence_budget_prepare = (
                self.draft_worker.get_confidence_budget_prepare()
            )

        if use_mlx():
            # MLX uses its own overlap loop and does not create CUDA streams,
            # but the normal non-overlap scheduler path still relays decode
            # input IDs through FutureMap.
            self.result_queue: Deque = deque()
            return

        # forward_stream_ctx / copy_stream are also used by PP (non-overlap)
        # via scheduler_pp_mixin; init unconditionally to match main.
        # 中译：forward_stream 用于跑前向，copy_stream 用于异步 D2H 拷贝结果。这两个 stream
        #       让前向计算与结果回传/下批调度能在 GPU 上并发，是重叠调度的硬件基础。
        #       PP（非重叠）也会用到，故无条件创建。
        self.forward_stream_ctx: CudaStreamContext = self.device_module.stream(
            self.forward_stream
        )
        self.copy_stream: CudaStream = self.device_module.Stream()
        self.copy_stream_ctx: CudaStreamContext = self.device_module.stream(
            self.copy_stream
        )

        if not self.enable_overlap:
            return

        # 中译：batch_record_buf 是一个长度为 2 的环形缓冲，用来"钉住"最近两轮批次的 GPU 张量
        #       引用，防止它们在前向 stream 仍在使用时被 PyTorch 缓存分配器提前释放
        #       （跨 stream 的张量生命周期问题）。batch_record_ct 是环形写指针。
        self.batch_record_buf = [None] * 2
        self.batch_record_ct = 0

    def maybe_init_ngram_embedding(self):
        self.ngram_embedding_manager = (
            self.tp_worker.model_runner.ngram_embedding_manager
        )
        self.use_ngram_embedding = self.tp_worker.model_config.use_ngram_embedding
        if self.use_ngram_embedding:
            self.token_table = self.tp_worker.model_runner.ngram_embedding_manager.table
            hf_config = self.tp_worker.model_config.hf_config
            self.ngram_embedding_n = hf_config.ngram_embedding_n
            self.ngram_embedding_k = hf_config.ngram_embedding_k

    def init_deterministic_inference_config(self):
        """Initialize deterministic inference configuration for different attention backends."""
        if not get_exec().deterministic.enable_deterministic_inference:
            self.truncation_align_size = None
            return

        backend_sizes = {
            "flashinfer": ("SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE", 4096),
            "triton": ("SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE", 4096),
        }
        # Both entries are prefill knobs (SPLIT_TILE / PREFILL_TRUNCATION):
        # the prefill half decides.
        prefill_backend, _ = attention_backends()
        env_var, default_size = backend_sizes.get(prefill_backend, (None, None))
        self.truncation_align_size = (
            get_int_env_var(env_var, default_size) if env_var else None
        )

    def init_request_dispatcher(self):
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.handle_generate_request),
                (TokenizedEmbeddingReqInput, self.handle_embedding_request),
                (BatchTokenizedGenerateReqInput, self.handle_batch_generate_request),
                (BatchTokenizedEmbeddingReqInput, self.handle_batch_embedding_request),
                (FlushCacheReqInput, self.flush_wrapper.handle),
                (ClearHiCacheReqInput, self.clear_hicache_storage_wrapped),
                (AttachHiCacheStorageReqInput, self.attach_hicache_storage_wrapped),
                (DetachHiCacheStorageReqInput, self.detach_hicache_storage_wrapped),
                (AbortReq, self.abort_request),
                (OpenSessionReqInput, self.open_session),
                (CloseSessionReqInput, self.close_session),
                (
                    UpdateWeightFromDiskReqInput,
                    self.weight_updater.update_weights_from_disk,
                ),
                (
                    InitWeightsUpdateGroupReqInput,
                    self.weight_updater.init_weights_update_group,
                ),
                (
                    DestroyWeightsUpdateGroupReqInput,
                    self.weight_updater.destroy_weights_update_group,
                ),
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
                    self.weight_updater.update_weights_from_distributed,
                ),
                (
                    UpdateWeightsFromTensorReqInput,
                    self.weight_updater.update_weights_from_tensor,
                ),
                (
                    UpdateWeightsFromIPCReqInput,
                    self.weight_updater.update_weights_from_ipc,
                ),
                (
                    UpdateWeightVersionReqInput,
                    self.handle_update_weight_version,
                ),
                (
                    GetWeightsByNameReqInput,
                    self.weight_updater.get_weights_by_name,
                ),
                (
                    ReleaseMemoryOccupationReqInput,
                    self.weight_updater.release_memory_occupation,
                ),
                (
                    ResumeMemoryOccupationReqInput,
                    self.weight_updater.resume_memory_occupation,
                ),
                (
                    CheckWeightsReqInput,
                    self.weight_updater.check_weights,
                ),
                (SlowDownReqInput, self.slow_down),
                (
                    ProfileReq,
                    lambda req: self.profiler_manager._profile(req),
                ),
                (FreezeGCReq, self.handle_freeze_gc),
                (ShutdownReq, self.handle_shutdown),
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
                (PauseGenerationReqInput, self.pause_generation),
                (ContinueGenerationReqInput, self.continue_generation),
                (ConfigureLoggingReq, self.configure_logging),
                (ScaleElasticEPReqInput, self.handle_scale_elastic_ep),
                (DumperControlReqInput, self.handle_dumper_control),
                (AddExternalCorpusReqInput, self.add_external_corpus),
                (
                    RemoveExternalCorpusReqInput,
                    self.remove_external_corpus,
                ),
                (
                    ListExternalCorporaReqInput,
                    self.list_external_corpora,
                ),
            ]
        )

    def _abort_on_running_timeout(self, running_batch: ScheduleBatch):
        # NOTE: this should be called before a batch is launched.
        # 中译：中止"运行过久"的在途请求——即已进入 running_batch、正在 decode 的请求
        #       从首次前向到现在耗时超过阈值的。必须在一个批次启动前调用（见 get_next_batch_to_run），
        #       这样设置的中止标记能在本轮被及时处理，不会与正在进行的前向产生竞态。
        # 中译：读取运行超时阈值（秒）。<=0 表示该功能关闭，直接返回。
        timeout_s = envs.SGLANG_REQ_RUNNING_TIMEOUT.get()
        if timeout_s <= 0:
            return
        # 中译：running_batch 为空（无在途请求）时无需检查。
        if running_batch.is_empty():
            return

        # 中译：deadline 是"截止时刻"——首次前向时间早于它的请求即视为超时。
        #       用 (now - timeout_s) 与各请求的进入时间比较，等价于"已运行 > timeout_s"。
        deadline = time.perf_counter() - timeout_s
            # 中译：forward_entry_time 是该请求首次进入前向的时刻；>0 表示已真正开始前向
        for req in running_batch.reqs:
            if not req.finished() and 0 < req.time_stats.forward_entry_time < deadline:
                # 中译：不直接通知 tokenizer/移出批次，而是设置延迟中止标记 to_finish。
                #       在途请求已占用 KV、可能正处于前向，直接删除不安全；改由后续正常的
                #       完成/清理路径消费该标记（见 get_next_batch_to_run 中对 reqs_to_abort
                #       的处理），统一发送 AbortReq 并释放 KV，避免资源泄漏与竞态。
                req.to_finish = FINISH_ABORT(
                    "Request running timeout reached.", HTTPStatus.SERVICE_UNAVAILABLE
                )

    def get_init_info(self) -> Dict[str, Any]:
        """Return scheduler initialization info for handshake.

        This method provides the initialization info needed by the tokenizer manager
        and other components to verify the scheduler is ready.
        """
        result_dict = {
            "status": "ready",
            "max_total_num_tokens": self.max_total_num_tokens,
            "max_req_input_len": self.max_req_input_len,
            "startup_time": self.startup_time,
        }

        return result_dict

    def release_host_resources(self) -> None:
        # Release pinned host buffers in userspace on graceful shutdown; see
        # HostKVCache.destroy. Called from run_scheduler_process's finally.
        if self.hisparse_coordinator is not None:
            self.hisparse_coordinator.destroy()
        self.tree_cache.release_host_resources()
        if self.decode_offload_manager is not None:
            self.decode_offload_manager.release_host_resources()

        rank_consensus_checker.shutdown()

    def run_event_loop(self) -> None:
        """Run the scheduler's event loop.

        Sets up the schedule stream and dispatches to the appropriate event loop.
        The event loop blocks until shutdown.

        中译：调度器主入口。先建立"调度 stream（schedule_stream）"——所有 CPU 端调度准备
              产生的 GPU 操作都跑在这条 stream 上，与前向 stream（forward_stream）分离，
              是重叠调度的关键。随后按 PD 分离模式 / PP / 重叠等情况分派到具体的事件循环。
              事件循环会一直阻塞运行直到进程关闭。
        """
        # Engine init (graph capture, warmups) is done; from here on any
        # Triton kernel device-load is a lazy first-use at serving time.
        triton_load_watch.install()
        triton_load_watch.mark_serving_started()

        if use_mlx():
            # MLX overlap uses mx.async_eval for CPU/GPU overlap,
            # not PyTorch MPS streams.
            dispatch_event_loop(self)
            return

        self.schedule_stream = self.device_module.Stream(priority=0)
        if self.device == "cpu":
            self.schedule_stream.synchronize = lambda: None  # No-op for CPU
        # 中译：WAR（Write-After-Read）屏障开关。重叠时本步调度会写入与上步前向共享的 GPU 缓冲，
        elif is_cuda() or _is_hip:
            # CUDA/HIP streams come from a fixed round-robin pool. Redraw if this
            # stream aliases forward_stream, which would eliminate scheduler
            # overlap. Only CUDA/HIP streams expose a ``cuda_stream`` handle;
            # other accelerators (e.g. NPU/XPU) skip the alias check.
            _redraws = 0
            while (
                self.schedule_stream.cuda_stream == self.forward_stream.cuda_stream
                and _redraws < 64
            ):
                self.schedule_stream = self.device_module.Stream(priority=0)
                _redraws += 1
        # The global WAR barrier fences the scheduler's next shared-buffer write
        # on the previous forward's read of the unified memory pool.
        self._war_barrier_enabled = is_cuda() or envs.SGLANG_ENABLE_WAR_BARRIER.get()
        with self.device_module.StreamContext(self.schedule_stream):
            dispatch_event_loop(self)

    def _apply_war_barrier(self):
        # WAR: keep later schedule_stream writes behind this forward's shared reads.
        # Clearing matters: a phase that skips the publish then falls back to coarse.
        if not self._war_barrier_enabled:
            return
        runner = self.model_worker.last_shared_read_runner
        ev = runner.shared_read_done_event
        runner.shared_read_done_event = None
        if ev is not None and not envs.SGLANG_FORCE_COARSE_WAR_BARRIER.get():
            self.schedule_stream.wait_event(ev)
        else:
            self.schedule_stream.wait_stream(self.forward_stream)

    @DynamicGradMode()
    def event_loop_normal(self):
        """A normal scheduler loop.

        中译：非重叠（同步）事件循环。每轮严格串行执行：收请求 -> 选批 -> 跑前向 -> 处理结果。
              run_batch 与 process_batch_result 之间没有并行，CPU 调度开销不会被 GPU 计算
              隐藏，因此吞吐通常不如 event_loop_overlap，但逻辑最简单、最易调试。
        """
        while True:
            if self.gracefully_exit:
                break

            # Receive requests
            # 中译：从 TokenizerManager/RPC 拉取新请求，并入队/分发处理。
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            # 中译：引擎被暂停（如权重热更新期间）时只收请求不跑前向。
            if self._engine_paused:
                continue

            # Get the next batch to run
            # 中译：核心调度决策——决定本轮跑哪个批次（优先 prefill，否则 decode，可能为 None）。
            plan = self.get_next_batch_to_run(
                running_batch=self.running_batch, last_batch=self.last_batch
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            self.cur_batch_for_debug = batch

            # Launch the current batch
            if batch:
                # 中译：同步跑前向并立即处理结果（无重叠）。
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, do self-check and re-init some states.
                # 中译：无可跑批次=服务器空闲，做不变量自检、重置状态、按需休眠。
                self._sched_idled = True
                self.on_idle()

            # Update last_batch
            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.invariant_checker.self_check_during_busy()

    @DynamicGradMode()
    def event_loop_overlap(self):
        """A scheduler loop that overlaps the CPU processing and GPU computation.

        中译：重叠（overlap）事件循环——本循环的精髓。核心思想：把"上一批的结果处理"
              推迟到"本批前向已在 GPU 上启动之后"再做，从而让 CPU 端的调度准备
              （get_next_batch_to_run 等）与 GPU 端的前向计算并行，隐藏 CPU 开销。
              实现手段是一个 result_queue：run_batch 启动前向后只把 (batch, result) 入队，
              等下一轮再 pop 出来处理。这样每一轮的 CPU 工作都"叠"在上一轮 GPU 工作之上。
              注意前向结果（如采样出的 token）此刻可能还没拷回 CPU，靠 FutureMap 中继。
        """
        # 中译：result_queue 缓存"已启动前向但尚未处理结果"的批次，最多积压一批。
        self.result_queue: Deque[
            Tuple[ScheduleBatch, Union[GenerationBatchResult, EmbeddingBatchResult]]
        ] = deque()

        def pop_and_process():
            # Process the results of the last batch
            # 处理上一批的前向结果
            # 中译：取出并处理队首（即上一批）的前向结果。
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        while True:
            if self.gracefully_exit:
                break

            # Receive requests
            # 接收请求
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue

            # 中译：WAR 屏障——让调度 stream 等待前向 stream，确保本轮调度对共享 GPU 缓冲的写
            # 中译：判断本批是否需要"关闭重叠"（例如连续两个 prefill、或 spec+grammar 组合），
            # Get the next batch to run
            plan = self.get_next_batch_to_run(
                running_batch=self.running_batch, last_batch=self.last_batch
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            self.cur_batch_for_debug = batch
            disable_overlap_for_batch = self.is_disable_overlap_for_batch(
                batch, last_batch=self.last_batch
            )

            # If we do not need to overlap the current batch with the last batch,
            # we can process the last batch immediately.
            # 如果本批无需与上一批重叠，可以立即处理上一批的结果。
            if disable_overlap_for_batch:
                pop_and_process()
                # Opportunistic flush at the disable_overlap sync boundary:
                # forward_stream is idle (prev forward drained, next not launched),
                # so `_flush`'s non-urgent guard compacts freely. Sync-free, best-effort.
                if self.enable_unified_memory:
                    try:
                        self.token_to_kv_pool_allocator.flush_opportunistic()
                    except Exception:
                        pass

            # Launch the current batch
            # 启动当前批次
            if batch:
                # 中译：启动本批前向，但不立即处理结果——把它（连同 batch 的副本）压入队列，
                #       下一轮再处理，从而实现 CPU/GPU 重叠。batch.copy() 是为了快照本批状态。
                batch_result = self.run_batch(batch)
                # Fence result processing behind this forward's shared reads.
                self._apply_war_barrier()
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None
                self._sched_idled = True

            # Process the last batch
            # 处理上一批
            if self.last_batch:
                # 中译：常规重叠路径——处理上一轮压入队列的批次结果（此时本批前向已在 GPU 上跑着）。
                if not disable_overlap_for_batch:
                    pop_and_process()
            elif batch is None:
                # When the server is idle, do self-check and re-init some states
                # 服务器空闲时：做自检并重置部分状态
                self.on_idle()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            # 它依赖上一批的结果（如语法约束），故须在上一批处理完后再执行。
            # 中译：对本批执行（可能被推迟的）采样。采样可能依赖上一批的结果（如语法约束的状态），
            #       所以必须放在上一批结果处理完之后再做。
            if self.is_generation:
                self.launch_batch_sample_if_needed(batch_result, batch)

            # Update last_batch
            # 更新 last_batch
            self.last_batch = batch

            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.invariant_checker.self_check_during_busy()

        # 中译：决定本批是否要"关闭重叠"（即必须先把上一批结果处理完再启动本批）。两类情形：
    def is_disable_overlap_for_batch(
        self, batch: ScheduleBatch, last_batch: Optional[ScheduleBatch]
    ) -> bool:
        # For two consecutive prefill batches, we disable overlap to improve the TTFT of the first batch.
        # This might slightly hurt the throughput, so we use an environment variable to control it.
        # In DP attention mode, use the globally synchronized is_extend_in_batch
        # so all DP ranks make the same overlap decision (avoiding deadlock).
        # In non-DP mode, use the local forward_mode directly.
        if self.require_mlp_sync:
            is_extend = lambda b: b and b.is_extend_in_batch
        else:
            is_extend = lambda b: b and b.forward_mode.is_extend()

        batch_is_extend = is_extend(batch)
        last_batch_is_extend = is_extend(last_batch)

        disable_overlap_for_batch = (
            envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.get()
            and batch_is_extend
            and last_batch_is_extend
        )

        # Sync so the FSM advance lands before the next batch's bitmask. Permanent
        # path for host-draft algorithms, not a pending migration.
        need_grammar_sync = (
            batch
            and not batch.spec_algorithm.is_none()
            and batch.grammar_needs_sync()
            and batch.forward_mode.is_decode()
            and len(self.result_queue) > 0
        )

        # Algorithms that support grammar overlap advance the FSM inside verify()
        # via the grammar barrier (overlapping the target forward), which resolves
        # whatever result is still pending in the queue — including the
        # extend->decode boundary — so no grammar-specific overlap disable is needed.
        return disable_overlap_for_batch or need_grammar_sync

    def _advance_pending_grammar(self):
        """Grammar barrier (spec-v2 overlap): advance the FSM over any not-yet
        -processed decode result still in the queue, so a following verify()'s
        bitmask sees the previous batch's committed tokens. Invoked mid-worker
        (before generate_token_bitmask) so the CPU advance overlaps the target
        verify forward. Idempotent; no-op when the queue is empty or has no grammar.
        """
        for prev_batch, prev_result in self.result_queue:
            self.batch_result_processor.advance_grammar_fsm(prev_result, prev_batch)

    @scheduler_nvtx_method("scheduler.process_input_requests")
    def process_input_requests(self, recv_reqs: List):
        # 中译：处理本轮收到的所有请求对象。对每个请求按其类型经分发器（_request_dispatcher）
        #       路由到对应 handler（生成/嵌入/权重更新/中止/会话管理等），handler 的返回值
        #       （若有）再回传给 TokenizerManager 或 RPC 端。
        now = time.monotonic()
        # 中译：清理到期的会话（session）。
        self.session_controller.maybe_reap(now)
        if get_mm().mm_feature_transport == "cuda_vmm":
            for recv_req in recv_reqs:
                self._materialize_cuda_vmm_inputs(recv_req)

        for recv_req in recv_reqs:
            # Skip health check when server is busy — ongoing requests already carry health info.
            # 中译：服务器繁忙时跳过健康检查请求——正在跑的请求本身已能证明服务存活，
            #       只需记下其回传通道，稍后由 maybe_send_health_check_signal 应答。
            if is_health_check_generate_req(recv_req) and not self.is_fully_idle(
                for_health_check=True
            ):
                self.return_health_check_ipcs.append(
                    getattr(recv_req, "http_worker_ipc", None)
                )
                continue

            # 中译：按消息类型路由到对应 handler 处理。
            output = self._request_dispatcher(recv_req)
            if output is not None:
                # 中译：RPC 类输出走 recv_from_rpc 通道回传，其余输出走 send_to_tokenizer。
                if self.rust_server is not None:
                    # Embedded Rust server: every control-request response goes
                    # back through the egress ring (the zmq tokenizer socket is
                    # not consumed); the Rust api_server shapes it per-endpoint.
                    self.rust_server.push_control_output(recv_req, output)
                elif isinstance(output, RpcReqOutput):
                    if self.ipc_channels.recv_from_rpc is not None:
                        sock_send(self.ipc_channels.recv_from_rpc, output)
                else:
                    self.ipc_channels.send_to_tokenizer.send_output(output, recv_req)

        self.flush_wrapper.check_pending()
        if self.external_corpus_manager is not None:
            self.external_corpus_manager.check_pending_load()

    def _materialize_cuda_vmm_inputs(self, recv_req):
        """Release VMM slices before request handling can reject the request."""
        if isinstance(
            recv_req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)
        ):
            tokenized_reqs = (recv_req,)
        elif isinstance(
            recv_req,
            (BatchTokenizedGenerateReqInput, BatchTokenizedEmbeddingReqInput),
        ):
            tokenized_reqs = recv_req
        else:
            return

        for tokenized_req in tokenized_reqs:
            if tokenized_req.mm_inputs is not None and not isinstance(
                tokenized_req.mm_inputs, MultimodalInputs
            ):
                tokenized_req.mm_inputs = MultimodalInputs.from_processor_output(
                    tokenized_req.mm_inputs
                )

    def init_profiler(self) -> None:
        self.profiler_manager = SchedulerProfilerManager(
            ps=self.ps,
            dp_tp_cpu_group=self.dp_tp_cpu_group,
            get_forward_ct=lambda: self.forward_ct,
        )

    def init_weight_updater(self) -> None:
        self.weight_updater = SchedulerWeightUpdaterManager(
            tp_worker=self.tp_worker,
            draft_worker=self.draft_worker,
            tp_cpu_group=self.tp_cpu_group,
            memory_saver_adapter=self.memory_saver_adapter,
            flush_cache=self.flush_cache,
            is_fully_idle=self.is_fully_idle,
            scheduler=self,
            metrics_collector=self.metrics_collector,
        )

    def init_lora_drainer(self) -> None:
        if get_lora().lora_drain_wait_threshold > 0.0:
            self.lora_drainer = LoRADrainer(
                get_lora().max_loras_per_batch,
                get_lora().lora_drain_wait_threshold,
            )
        else:
            self.lora_drainer = None

    def init_lora_overlap_loader(self) -> None:
        if self.enable_lora_overlap_loading:
            self.lora_overlap_loader = LoRAOverlapLoader(
                self.tp_worker.model_runner.lora_manager
            )

    def init_grammar_manager(self) -> None:
        self.grammar_manager = GrammarManager(self)

    def maybe_init_scripted_scheduler_hook(self) -> None:
        if envs.SGLANG_TEST_SCRIPTED_RUNTIME.get():
            from sglang.test.scripted_runtime.scheduler_hook import (
                ScriptedSchedulerHook,
            )

            self.scripted_scheduler_hook = ScriptedSchedulerHook(
                scheduler=self,
                tokenizer_recv_proxy=self.ipc_channels.recv_from_tokenizer,
            )
        else:
            self.scripted_scheduler_hook = None

    def _hosts_rust_server(self) -> bool:
        """Whether this scheduler rank embeds the Rust server (rank 0 only) —
        and with it the server-process duties a Python ``TokenizerManager``
        would otherwise own (e.g. serving the PD KV bootstrap registry)."""
        return envs.SGLANG_RUST_SERVER.get() and (
            self.ps.pp_rank == 0
            and self.ps.attn_tp_rank == 0
            and self.ps.attn_cp_rank == 0
        )

    def maybe_init_rust_server(self) -> None:
        """Start the embedded Rust server (rank 0) if ``SGLANG_RUST_SERVER`` is
        set, and point the ingress receiver at it. All the plumbing lives in
        ``RustServer`` (scheduler_components/rust_scheduler.py)."""

        if not self._hosts_rust_server():
            # Always define the attribute: init_output_streamer and the
            # process_input_requests hook read self.rust_server unconditionally.
            self.rust_server = None
            return

        rust_server = RustServer.launch(self)
        self.rust_server = rust_server
        # The rust server *is* the ingress source: SchedulerRequestReceiver
        # drains its request ring (rust_server_mode) instead of a zmq socket.
        self.recv_from_tokenizer = rust_server
        # Park the idle loop on the request ring within the rank-0 rust-server
        self.idle_sleeper = RustServerIdleSleeper(rust_server)

    def init_request_receiver(self) -> None:
        self.request_receiver = SchedulerRequestReceiver(
            recv_from_tokenizer=self.recv_from_tokenizer,
            recv_from_rpc=self.ipc_channels.recv_from_rpc,
            recv_skipper=self.recv_skipper,
            input_blocker=self.input_blocker,
            mm_receiver=self.mm_receiver,
            ps=self.ps,
            tp_group=self.tp_group,
            tp_cpu_group=self.tp_cpu_group,
            attn_tp_group=self.attn_tp_group,
            attn_tp_cpu_group=self.attn_tp_cpu_group,
            attn_cp_group=self.attn_cp_group,
            attn_cp_cpu_group=self.attn_cp_cpu_group,
            world_group=self.world_group,
            server_args=self.server_args,
            model_config=self.model_config,
            max_recv_per_poll=self.max_recv_per_poll,
            stream_output=lambda *a, **kw: self.output_streamer.stream_output(*a, **kw),
            get_last_batch=lambda: self.last_batch,
            scripted_scheduler_hook=self.scripted_scheduler_hook,
        )

    def init_dp_attn_adapter(self) -> None:
        # Spec workers have no .model_runner of their own; the prefill graph
        # runner that votes belongs to the target model.
        target_worker = (
            self.tp_worker.target_worker
            if isinstance(self.tp_worker, BaseSpecWorker)
            else self.tp_worker
        )
        self.dp_attn_adapter = SchedulerDPAttnAdapter(
            model_runner=target_worker.model_runner,
            tp_group=self.tp_group,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            offload_tags=self.weight_updater.offload_tags,
            ps=self.ps,
            model_config=self.model_config,
            enable_overlap=self.enable_overlap,
            spec_algorithm=self.spec_algorithm,
            get_require_mlp_sync=lambda: self.require_mlp_sync,
        )

    def init_pool_stats_observer(self) -> None:
        self.pool_stats_observer = SchedulerPoolStatsObserver(
            tree_cache=self.tree_cache,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            req_to_token_pool=self.req_to_token_pool,
            session_controller=self.session_controller,
            hisparse_coordinator=self.hisparse_coordinator,
            is_hybrid_swa=self.is_hybrid_swa,
            is_hybrid_ssm=self.is_hybrid_ssm,
            enable_hisparse=self.enable_hisparse,
            full_tokens_per_layer=self.full_tokens_per_layer,
            swa_tokens_per_layer=self.swa_tokens_per_layer,
            max_total_num_tokens=self.max_total_num_tokens
            * get_parallel().attn_dcp_size,
            get_last_batch=lambda: self.last_batch,
            get_running_batch=lambda: self.running_batch,
        )

    def init_invariant_checker(self) -> None:
        self.invariant_checker = SchedulerInvariantChecker(
            is_hybrid_swa=self.is_hybrid_swa,
            is_hybrid_ssm=self.is_hybrid_ssm,
            disaggregation_mode=self.disaggregation_mode,
            page_size=self.page_size,
            full_tokens_per_layer=self.full_tokens_per_layer,
            swa_tokens_per_layer=self.swa_tokens_per_layer,
            max_total_num_tokens=self.max_total_num_tokens,
            tree_cache=self.tree_cache,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            req_to_token_pool=self.req_to_token_pool,
            pool_stats_observer=self.pool_stats_observer,
            get_last_batch=lambda: self.last_batch,
            get_running_batch=lambda: self.running_batch,
        )

    def init_rank_consensus_checker(self) -> None:
        groups = []
        if self.attn_cp_group is not None and self.attn_tp_group is not None:
            groups += [self.attn_cp_group, self.attn_tp_group]
        else:
            groups += [self.tp_group]
        if self.pp_group is not None:
            groups += [self.pp_group]
        rank_consensus_checker.configure(groups)

    def init_kv_events_publisher(self) -> None:
        self.kv_events_publisher = SchedulerKvEventsPublisher(
            kv_events_config=get_observability().kv_events_config,
            ps=self.ps,
            attn_tp_rank=self.ps.attn_tp_rank,
            attn_cp_rank=self.ps.attn_cp_rank,
            attn_dp_rank=self.ps.attn_dp_rank,
            dp_rank=self.ps.dp_rank,
            tree_cache=self.tree_cache,
            send_metrics_from_scheduler=self.ipc_channels.send_metrics_from_scheduler,
            max_running_requests=self.max_running_requests,
            max_total_num_tokens=self.max_total_num_tokens,
            get_stats=lambda: self.metrics_reporter.stats,
        )

    def init_load_publisher(self) -> None:
        # Router-facing load reporting; rank gating and no-op fallback live
        # inside the component. Same interval as the DP-balancing writer so
        # the two fire in phase and the load sink always reuses that snapshot
        # instead of walking the queues itself.
        self.load_publisher = SchedulerLoadPublisher(
            kv_events_config=get_observability().kv_events_config,
            ps=self.ps,
            load_publish_endpoint=get_observability().load_publish_endpoint,
            publish_interval=get_observability().load_snapshot_publish_interval,
        )

    def init_load_inquirer(self) -> None:
        self.total_prefill_uncached_tokens = 0
        self.total_prefill_busy_us = 0
        self.decode_moment_totals: list[float] = [0.0] * 6
        self._prev_step: Optional[Tuple[int, float, bool]] = None
        self._sched_idled = False
        self.load_inquirer = SchedulerLoadInquirer(
            disaggregation_mode=self.disaggregation_mode,
            ps=self.ps,
            server_args=self.server_args,
            max_total_num_tokens=self.max_total_num_tokens,
            max_running_requests=self.max_running_requests,
            pool_stats_observer=self.pool_stats_observer,
            tp_worker=self.tp_worker,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            spec_algorithm=self.spec_algorithm,
            get_running_batch=lambda: self.running_batch,
            get_waiting_queue=lambda: self.waiting_queue,
            waiting_queue_prefix_matched=lambda: self.policy.waiting_queue_prefix_matched(
                self.waiting_queue
            ),
            get_recent_cache_hit_rate=lambda: self.metrics_reporter.recent_cache_hit_rate,
            get_stats=lambda: self.metrics_reporter.stats,
            get_chunked_req=lambda: self.chunked_req,
            get_disagg_prefill_bootstrap_queue=lambda: self.disagg_prefill_bootstrap_queue,
            get_disagg_prefill_inflight_queue=lambda: self.disagg_prefill_inflight_queue,
            get_disagg_decode_prealloc_queue=lambda: self.disagg_decode_prealloc_queue,
            get_disagg_decode_transfer_queue=lambda: self.disagg_decode_transfer_queue,
            get_spec_total_num_accept_tokens=lambda: self.metrics_reporter.spec_total_num_accept_tokens,
            get_spec_total_num_forward_ct=lambda: self.metrics_reporter.spec_total_num_forward_ct,
            get_total_prefill_uncached_tokens=lambda: self.total_prefill_uncached_tokens,
            get_total_prefill_busy_us=lambda: self.total_prefill_busy_us,
            get_decode_moment_totals=lambda: self.decode_moment_totals,
        )

    def init_output_streamer(self) -> None:
        self.output_streamer = self.get_output_streamer_class()(
            send_to_detokenizer=self.ipc_channels.send_to_detokenizer,
            tree_cache=self.tree_cache,
            ps=self.ps,
            server_args=self.server_args,
            is_generation=self.is_generation,
            spec_algorithm=self.spec_algorithm,
            disaggregation_mode=self.disaggregation_mode,
            enable_hicache_storage=lambda: self.enable_hicache_storage,
            rust_server=self.rust_server,
        )

    def get_output_streamer_class(self) -> type[SchedulerOutputStreamer]:
        return SchedulerOutputStreamer

    def init_beam_coordinator(self) -> None:
        self.beam_coordinator = BeamCoordinator(
            model_config=self.model_config,
            spec_algorithm=self.spec_algorithm,
            dllm_enabled=self.dllm_config is not None,
            max_req_len=self.max_req_len,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            future_map=self.future_map,
        )

    def init_batch_result_processor(self) -> None:
        self.init_beam_coordinator()
        self.batch_result_processor = SchedulerBatchResultProcessor(
            is_generation=self.is_generation,
            disaggregation_mode=self.disaggregation_mode,
            enable_overlap=self.enable_overlap,
            enable_overlap_mlx=self.enable_overlap_mlx,
            model_config=self.model_config,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            hisparse_coordinator=self.hisparse_coordinator,
            req_to_token_pool=self.req_to_token_pool,
            decode_offload_manager=self.decode_offload_manager,
            metrics_collector=self.metrics_collector,
            metrics_reporter=self.metrics_reporter,
            draft_worker=self.draft_worker,
            model_worker=self.model_worker,
            logprob_result_processor=SchedulerLogprobResultProcessor(
                model_config=self.model_config
            ),
            output_streamer=self.output_streamer,
            beam_coordinator=self.beam_coordinator,
            abort_request=self.abort_request,
        )

    def init_req_max_new_tokens(self, req):
        # 中译：为请求确定 max_new_tokens 的安全上限。必须保证该请求即便跑满也不会超出
        #       max_req_len 和 max_total_num_tokens 的预算，否则会出现"能入队却永远无法被调度"
        #       的请求，堵死队列、最终拖垮健康检查。这里取用户值与两个硬上限的最小值（且非负）。
        # 中译：prompt（输入）的真实 token 数。后面所有预算都要在它之上再留出生成空间。
        input_len = len(req.origin_input_ids)
        max_new_tokens = (
            req.sampling_params.max_new_tokens
            if req.sampling_params.max_new_tokens is not None
            else 1 << 30
        )
        if self.max_new_tokens_limit is not None and self.max_new_tokens_limit > 0:
            if max_new_tokens > self.max_new_tokens_limit:
                logger.warning(
                    f"Capping max_new_tokens of request {req.rid} to "
                    f"SGLANG_MAX_NEW_TOKENS_LIMIT={self.max_new_tokens_limit} "
                    f"(requested: {req.sampling_params.max_new_tokens})."
                )
            max_new_tokens = min(max_new_tokens, self.max_new_tokens_limit)

        # Keep this bound consistent with PrefillAdder's admission budget:
        # ceil_page(input_len) + max_new_tokens + page_size must be strictly
        # smaller than max_total_num_tokens. Otherwise a request can be accepted
        # into the waiting queue but can never be scheduled, blocking the queue
        # and eventually making health checks fail.
        # 中译：这里的上限必须和 PrefillAdder 的准入预算保持一致——
        #       ceil_page(input_len) + max_new_tokens + page_size 必须严格小于 max_total_num_tokens。
        #       否则会出现"请求能进等待队列、却永远无法被调度"的死锁：它既不释放资源也跑不动，
        #       堵住队列，最终把健康检查也拖垮。
        # 中译：把 input_len 向上取整到 page_size 的整数倍。KV cache 以 page（页）为粒度分配，
        #       一条不满整页的 prompt 也会占用一整页，所以预算要按"取整后的页长"来算。
        #       -(-a // b) 是整数向上取整的惯用写法（等价 ceil(a/b)），再乘 page_size 还原成 token 数。
        paged_input_len = -(-input_len // self.page_size) * self.page_size
        # 中译：最终 max_new_tokens = max(0, min(用户请求值, 单请求长度上限, 总显存预算上限))。
        #       三者取最小保证"跑满也不越界"，外层 max(0, ...) 防止上限算出负数时把值变成负。
        req.sampling_params.max_new_tokens = max(
            0,
            min(
                    # 中译：用户显式指定的 max_new_tokens；没指定（None）时用 1<<30（约 10 亿）
                # 中译：单请求长度上限。input + 生成 不得超过 max_req_len，留 1 个 token 余量。
                # 中译：全局 KV cache 预算上限。扣掉本请求取整后的 prompt 页长、再预留一个 page_size
                max_new_tokens,
                self.max_req_len - input_len - 1,
                self.max_total_num_tokens * get_parallel().attn_dcp_size
                - paged_input_len
                - self.page_size
                - 1,
            ),
        )
        # Clipping above can push max_new_tokens below min_new_tokens, which
        # would suppress EOS for the whole generation. Restore the invariant.
        if req.sampling_params.min_new_tokens > req.sampling_params.max_new_tokens:
            req.sampling_params.min_new_tokens = req.sampling_params.max_new_tokens

    def _process_and_broadcast_mm_inputs(
        self,
        raw_mm_inputs,
    ):
        """Materialize MultimodalInputs once on the entry rank and broadcast to others.

        Entry rank:
        - constructs MultimodalInputs.from_processor_output() once
        - broadcasts to other ranks in self.cpu_group (if world_size > 1)

        Non-entry ranks:
        - receive the object via broadcast (if world_size > 1)
        - otherwise (single-rank / no group) fall back to local from_processor_output

        Returns:
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

        # In case tp size > 1, all the Scheduler TP ranks runs the duplicated computing
        # process in CPU which occupies the main thread CPU cycle. This computing logic
        # merely needs to be run on TP0 and be broadcast to other TP ranks.
        # Since the Scheduler is single-threaded, any large CPU cost will impact
        # handling of other messages. For example, CPU hits 99.9% can significantly
        # increase the CUDA kernel launch time.
        if self.dp_tp_group.rank_in_group == 0:
            # Only the entry rank materializes once from dict.
            image_inputs = MultimodalInputs.from_processor_output(raw_mm_inputs)
            # Broadcast to other TP ranks (use src=0 within the group).
            if group_world_size > 1:
                obj_list = [image_inputs]
                torch.distributed.broadcast_object_list(
                    obj_list,
                    src=self.dp_tp_group.first_rank,
                    group=self.dp_tp_cpu_group,
                )
                image_inputs = obj_list[0]
        else:
            # Non-entry ranks: receive if group size > 1; otherwise materialize locally.
            if group_world_size > 1:
                obj_list = [None]
                torch.distributed.broadcast_object_list(
                    obj_list,
                    src=self.dp_tp_group.first_rank,
                    group=self.dp_tp_cpu_group,
                )
                image_inputs = obj_list[0]
            else:
                image_inputs = MultimodalInputs.from_processor_output(raw_mm_inputs)

        return image_inputs

    def _get_multimodal_inputs(self, mm_inputs):
        if isinstance(mm_inputs, MultimodalInputs):
            return mm_inputs

        if get_mm().enable_broadcast_mm_inputs_process:
            return self._process_and_broadcast_mm_inputs(mm_inputs)
        return MultimodalInputs.from_processor_output(mm_inputs)

    @staticmethod
    def _try_apply_padded_mm_input_ids(recv_req, req, image_inputs) -> bool:
        """setup origin_input_ids with trying to reuse existing MultimodalInputs.padded_input_ids first,
        if absent, call pad_input_ids_func"""
        padded_input_ids = image_inputs.padded_input_ids
        if padded_input_ids is None or recv_req.input_ids is None:
            return False

        recv_input_len = len(recv_req.input_ids)
        if len(padded_input_ids) != recv_input_len:
            return False

        prefix_len = len(req.origin_input_ids) - recv_input_len
        if prefix_len < 0:
            return False

        padded_input_ids = array("q", padded_input_ids)
        if prefix_len == 0:
            req.origin_input_ids = padded_input_ids
        else:
            req.origin_input_ids = req.origin_input_ids[:prefix_len] + padded_input_ids
        return True

    def _maybe_compute_mrope_positions(self, req) -> None:
        """Compute M-RoPE positions when they are missing (e.g. gRPC preprocessed path)."""
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

    def _maybe_namespace_elastic_radix_cache(self, req: Req) -> None:
        if (
            get_exec().moe.elastic_ep_backend is None
            or self.disable_radix_cache
            or not self.tree_cache.is_tree_cache()
        ):
            return

        from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager

        inst = ElasticEPStateManager.instance()
        if inst is None:
            return

        namespace = f"elastic_ep_size={ElasticEPStateManager.get_effective_ep_size()}"
        if req.extra_key:
            req.extra_key = f"{req.extra_key}|{namespace}"
        else:
            req.extra_key = namespace

        if self._last_logged_elastic_radix_namespace != namespace:
            self._last_logged_elastic_radix_namespace = namespace
            logger.debug(
                "[Elastic EP][scale] radix cache namespace is now %s",
                namespace,
            )

    def _maybe_clear_mm_inputs(self, batch: ScheduleBatch) -> None:
        for req in batch.reqs:
            if not req.finished() or not (mm_inputs := req.multimodal_inputs):
                continue
            # For session requests, keep mm_inputs for the next request
            if req.session:
                continue
            # For non-session requests, clear features and mm_inputs
            mm_inputs.release_features()
            req.multimodal_inputs = None

    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        """中译：处理一条"生成类"请求。流程：
        1. 按是否带 session_id 分三路构造 Req 对象（普通请求 / 复用会话 / 会话不存在报错）；
        2. PD 分离模式下校验 bootstrap room；
        3. 处理多模态输入（把单个图像占位 token 扩展成多个 dummy token）；
        4. 计算 max_new_tokens 上限、校验 prompt 长度、设置 logprob 起点；
        5. 若需要语法约束则先进语法队列，否则调用 _add_request_to_queue 入队等待调度。
        任何校验失败都会构造一个"带中止原因"的 Req 入队，以便把错误经正常出口回传给用户。
        """
        # Route: normal request / session request / session-not-found
        session_id = (
            recv_req.session_params.id if recv_req.session_params is not None else None
        )
        # Radix-native sessions use only the top-level session_id.
        radix_native_session = (
            recv_req.session_id is not None and self.enable_session_radix_cache
        )

            # 中译：普通（无会话）请求路径。
        if session_id is None or radix_native_session:
            # Normal non-session request, or a radix-native session request
            if recv_req.input_embeds is not None:
                # Generate fake input_ids based on the length of input_embeds
                # 中译：当直接传入 embedding 时没有真实 token id，按其长度造一串占位 input_ids。
                seq_length = len(recv_req.input_embeds)
                recv_req.input_ids = array("q", [1]) * seq_length

            if recv_req.bootstrap_port is None:
                # Use default bootstrap port
                # 中译：PD 分离（Prefill/Decode 分离）下用 bootstrap 端口在 P、D 实例间建立 KV 传输连接；
            # 中译：把传输层的 TokenizedGenerateReqInput 转成调度器内部使用的 Req 对象，
                recv_req.bootstrap_port = get_disagg().disaggregation_bootstrap_port

            is_beam = BeamCoordinator.request_beam_width(recv_req) > 1
            req = Req(
                recv_req.rid,
                recv_req.input_text,
                recv_req.input_ids,
                recv_req.sampling_params,
                return_logprob=recv_req.return_logprob,
                top_logprobs_num=recv_req.top_logprobs_num,
                token_ids_logprob=recv_req.token_ids_logprob,
                return_sampling_mask=recv_req.return_sampling_mask,
                return_flat_raw_top_logprobs=recv_req.return_flat_raw_top_logprobs,
                stream=recv_req.stream,
                lora_id=recv_req.lora_id,
                session_id=recv_req.session_id,
                input_embeds=recv_req.input_embeds,
                positional_embed_overrides=recv_req.positional_embed_overrides,
                token_type_ids=recv_req.token_type_ids,
                custom_logit_processor=recv_req.custom_logit_processor,
                require_reasoning=recv_req.require_reasoning,
                return_hidden_states=recv_req.return_hidden_states,
                return_routed_experts=recv_req.return_routed_experts,
                routed_experts_start_len=recv_req.routed_experts_start_len,
                return_indexer_topk=recv_req.return_indexer_topk,
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
                    self.metrics_collector
                    if self.metrics_reporter.enable_metrics
                    else None
                ),
                routing_key=recv_req.routing_key,
                extra_key=recv_req.extra_key,
                cache_salt=recv_req.cache_salt,
                http_worker_ipc=recv_req.http_worker_ipc,
                dllm_config=self.dllm_config,
                time_stats=recv_req.time_stats,
                multi_item_delimiter_indices=recv_req.multi_item_delimiter_indices,
            )
            req.tokenizer = self.tokenizer

            if radix_native_session:
                req.session_generation = self.tree_cache.ensure_session_generation(
                    recv_req.session_id
                )

            if is_beam:
                error_msg = self.beam_coordinator.validate_and_init(req, recv_req)
                if error_msg:
                    logger.error(error_msg)
                    prepare_abort(req, error_msg, status_code=HTTPStatus.BAD_REQUEST)
                    self.output_streamer.stream_output([req], req.return_logprob)
                    return

            if self.disaggregation_mode != DisaggregationMode.NULL:
                # Invalid request for disaggregated mode
                # 中译：PD 分离模式下，每条请求必须带 bootstrap_room（P/D 两端据此配对同一请求的 KV）。
                #       缺失即非法（FAKE 传输后端是测试用例外，可放行），构造一个带中止原因的 Req
                #       直接经正常输出通道回传错误后返回。
                if (
                    recv_req.bootstrap_room is None
                    and self.transfer_backend != TransferBackend.FAKE
                ):
                    error_msg = (
                        f"Invalid request: Disaggregated request received without "
                        f"bootstrap room id. {req.rid=}"
                    )
                    logger.error(error_msg)
                    if not envs.SGLANG_RUST_SERVER.get():
                        recv_req.time_stats.trace_ctx.abort(
                            abort_info={"reason": error_msg}
                        )
                    prepare_abort(req, error_msg, status_code=HTTPStatus.BAD_REQUEST)
                    self.output_streamer.stream_output([req], req.return_logprob)
                    return

        elif (
            session_id in self.session_controller
            and not self.session_controller.get(session_id).close_on_finish
        ):
            # Session exists and is not closing: create request from session
            # 中译：会话复用路径。会话存在且未处于"用完即关"状态时，由 session 基于历史上下文
            #       创建新 Req（会把之前轮次的 token 接续进来），实现多轮对话的前缀复用。
            session = self.session_controller.get(session_id)
            req = session.create_req(
                recv_req,
                self.tokenizer,
                self.model_config.vocab_size,
                eos_token_ids=self.model_config.hf_eos_token_id,
            )
            if self.enable_session_radix_cache:
                req.session_generation = self.tree_cache.ensure_session_generation(
                    session_id
                )
            # TODO: set trace context
            if self.metrics_reporter.enable_metrics:
                req.time_stats.set_metrics_collector(self.metrics_collector)
            # 中译：create_req 内部若已判定请求非法（如续接位置越界），会把 finished_reason 设成
            #       FINISH_ABORT；此时直接入队让错误经正常出口回传，不再走后续校验。
            if isinstance(req.finished_reason, FINISH_ABORT):
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

        else:
            # Session not found, or session is closing
            # 中译：会话不存在、或会话正在关闭。两种情况都属于非法请求，
            #       构造带对应错误信息的中止 Req 入队回传。
            if session_id in self.session_controller:
                error_msg = (
                    f"Invalid request: close was requested for session {session_id}"
                )
            else:
                error_msg = f"Invalid request: session id {session_id} does not exist"
            req = Req(
                recv_req.rid,
                recv_req.input_text,
                recv_req.input_ids,
                recv_req.sampling_params,
                vocab_size=self.model_config.vocab_size,
                http_worker_ipc=recv_req.http_worker_ipc,
            )
            req.tokenizer = self.tokenizer
            req.set_finish_with_abort(error_msg)
            self.init_req_max_new_tokens(req)
            self._add_request_to_queue(req)
            return

        # 中译：DFlash 投机解码算法对请求有额外约束（与 overlap 调度的兼容性等），
        self._maybe_namespace_elastic_radix_cache(req)

        if self.spec_algorithm.is_dflash_family():
            error_msg = validate_dflash_request(req, self.enable_overlap)
            if error_msg is not None:
                req.set_finish_with_abort(error_msg)
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

        if (
            req.return_sampling_mask
            and self.disaggregation_mode != DisaggregationMode.NULL
            and not self.disagg_metadata_buffers.enable_sampling_mask
        ):
            error_msg = (
                "return_sampling_mask with disaggregation requires "
                "SGLANG_DISAGGREGATION_SAMPLING_MASK_MAX_TOKENS > 0."
            )
            req.set_finish_with_abort(error_msg)
            self.init_req_max_new_tokens(req)
            self._add_request_to_queue(req)
            return

        if req.return_sampling_mask and req.sampling_params.top_k == TOP_K_ALL:
            error_msg = (
                "return_sampling_mask requires finite top_k; top_p-only sampling "
                "is valid but can return huge masks in the tail, blowing up "
                "metadata, so we need a safety cap."
            )
            req.set_finish_with_abort(error_msg)
            self.init_req_max_new_tokens(req)
            self._add_request_to_queue(req)
            return

        if req.return_sampling_mask and not self.spec_algorithm.is_none():
            # Spec workers do not emit one sampling support per accepted token, so
            # the returned mask would not align 1:1 with generated tokens. Reject
            # the combination instead of silently returning a misaligned mask.
            error_msg = (
                "return_sampling_mask is not supported with speculative decoding."
            )
            req.set_finish_with_abort(error_msg)
            self.init_req_max_new_tokens(req)
            self._add_request_to_queue(req)
            return

        if req.return_sampling_mask and get_exec().kernel.sampling_backend == "ascend":
            # The ascend backend samples from logits directly and never builds the
            # top-k/top-p support, so it cannot produce a sampling mask.
            error_msg = (
                "return_sampling_mask is not supported with the ascend "
                "sampling backend."
            )
            req.set_finish_with_abort(error_msg)
            self.init_req_max_new_tokens(req)
            self._add_request_to_queue(req)
            return

        # Handle multimodal inputs
        # 中译：处理多模态输入。关键动作是把单个图像占位 token 扩展成与图像 embedding 数量
        #       相匹配的多个 dummy token，以便前向时把图像特征填进对应位置。
        if recv_req.mm_inputs is not None:
            image_inputs = self._get_multimodal_inputs(recv_req.mm_inputs)

            SessionController.adjust_mm_offsets(recv_req, req, image_inputs)

            # The following steps are already fast, execute locally on each rank.
            # Expand a single image token into multiple dummy tokens for receiving image embeddings.
            # The pad function is model-specific and can be None for some backends.
            if (
                not self._try_apply_padded_mm_input_ids(recv_req, req, image_inputs)
                and self.pad_input_ids_func
            ):
                req.origin_input_ids = array(
                    "q", self.pad_input_ids_func(req.origin_input_ids, image_inputs)
                )
            req.extend_image_inputs(image_inputs)
            self._maybe_compute_mrope_positions(req)

            # 中译：图像占位 token 扩展后 prompt 可能暴涨，这里校验扩展后的真实长度是否超出
            #       单请求输入上限，超了就中止（错误信息里同时给出扩展前后长度便于排查）。
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

        # initialize before returning
        # 中译：在做后续长度/logprob 校验之前，先把 max_new_tokens 收敛到安全上限
        #       （见 init_req_max_new_tokens），保证即便后面提前 return 入队，该值也已就绪。
        self.init_req_max_new_tokens(req)

        # Validate prompt length
        # 中译：校验 prompt 长度是否超过输入上限。若开启 allow_auto_truncate 会自动截断而非报错；
        #       否则返回错误信息并中止请求。
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            get_serving().allow_auto_truncate,
        )
        if error_msg:
            req.set_finish_with_abort(error_msg)
            self._add_request_to_queue(req)
            return

        # 中译：以下确定 logprob_start_len——从输入序列的哪个位置开始计算并返回 logprob。
        if not recv_req.return_logprob and recv_req.logprob_start_len != -1:
            # When return_logprob is False, logprob_start_len should be ignored
            # 中译：没要 logprob 却传了起点，忽略该起点（置 -1）。
            recv_req.logprob_start_len = -1

        if recv_req.logprob_start_len == -1:
            # 中译：起点为 -1（未指定）时，按场景推断默认起点。
            if recv_req.return_logprob and recv_req.token_ids_logprob is None:
                # If logprob is required but neither token_ids_logprob nor logprob_start_len is
                # set, return the logprobs for output tokens by default
                # 中译：要 logprob 但既没指定 token_ids 也没指定起点，默认只返回"输出 token"的 logprob，
                #       即起点设在输入末尾（跳过对 prompt 部分算 logprob）。
                req.logprob_start_len = len(req.origin_input_ids)
            elif req.is_prefill_only:
                # For prefill-only requests with logprob_start_len == -1, set logprob_start_len
                # beyond input sequence to skip input logprob computation entirely
                # 中译：prefill-only 请求把起点设到输入序列之外，完全跳过对输入的 logprob 计算。
                req.logprob_start_len = len(req.origin_input_ids)
            else:
                # If return_logprob is False, only the last token requires logprob computation
                # 中译：其余情况只有最后一个 token 需要 logprob，用 -1 表示。
                req.logprob_start_len = -1
        else:
            # 中译：用户显式给定了起点，直接采用。
            req.logprob_start_len = recv_req.logprob_start_len

        # 中译：起点不能超过输入 token 数，越界则中止。
        if req.logprob_start_len > len(req.origin_input_ids):
            error_msg = f"{req.logprob_start_len=} is higher than the number of input tokens {len(req.origin_input_ids)=}. Please use a smaller logprob_start_len."
            req.logprob_start_len = -1
            req.set_finish_with_abort(error_msg)
            self._add_request_to_queue(req)
            return

        # 中译：若请求要返回 MoE 路由的 expert 选择（return_routed_experts），
        if (
            get_device().mlx_enable_sampling
            and req.return_logprob
            and 0 <= req.logprob_start_len < len(req.origin_input_ids)
        ):
            # The MLX sampling path computes output logprobs only; the
            # prefill result carries no input_token_logprobs, so letting
            # this through would crash output processing.
            error_msg = (
                "Prompt input logprobs (logprob_start_len) are not supported "
                "on the MLX sampling path; omit logprob_start_len to get "
                "output logprobs."
            )
            req.logprob_start_len = -1
            req.set_finish_with_abort(error_msg)
            self._add_request_to_queue(req)
            return

        if recv_req.return_routed_experts:
            error_msg = None
            if recv_req.routed_experts_start_len < 0:
                error_msg = (
                    f"{recv_req.routed_experts_start_len=} is lower than 0. "
                    "Please use a non-negative routed_experts_start_len."
                )

            if recv_req.routed_experts_start_len > len(req.origin_input_ids):
                error_msg = (
                    f"{recv_req.routed_experts_start_len=} is higher than the "
                    f"number of input tokens {len(req.origin_input_ids)=}. Please "
                    f"use a smaller routed_experts_start_len."
                )

            if error_msg is not None:
                req.routed_experts_start_len = 0
                req.set_finish_with_abort(error_msg)
                self._add_request_to_queue(req)
                return

        # 中译：若请求带结构化输出/语法约束（如 JSON schema、正则），需先编译语法，
        #       编译期间请求放进语法队列；否则直接进等待队列。
        added_to_grammar_queue = self.grammar_manager.process_req_with_grammar(req)
        if not added_to_grammar_queue:
            self._add_request_to_queue(req)

    def handle_batch_generate_request(
        self,
        recv_req: BatchTokenizedGenerateReqInput,
    ):
        """Handle optimized batch generate request."""
        logger.debug(f"Processing batch generate request with {len(recv_req)} requests")

        # Process each request in the batch
        for tokenized_req in recv_req:
            self.handle_generate_request(tokenized_req)

    def _prefetch_kvcache(self, req: Req):
        # 中译：请求入队时触发的 HiCache「L3 存储预取」。目标是在请求真正被调度进 prefill 之前，
        #       就异步地把它能从外部存储（L3，磁盘/远端）命中的 KV 前缀提前搬回 host 内存（L2），
        #       从而在调度到该请求时省掉这部分前缀的重算，缩短首 token 延迟。整个过程是异步、
        #       尽力而为的——发起后请求照常排队，后续在调度循环里通过 check_prefetch_progress 查进度。
        # 中译：仅在启用了 HiCache 存储后端（L3）时才有预取的意义；否则直接跳过。
        if self.enable_hicache_storage:
            # 中译：先做一次前缀匹配，计算本请求在现有 radix tree 中的命中情况。这会填充
            #       req.prefix_indices（device/L1 命中的槽位）、req.host_hit_length（仅 host/L2
            #       命中的长度）与 req.last_host_node（沿父链最近一个仍有 host 备份的节点）。
            #       cow_mamba=False：此处只为算命中、发预取，不需要为 Mamba 状态做写时复制。
            req.init_next_round_input(self.tree_cache, cow_mamba=False)
            # 中译：last_host_node 是「已匹配前缀」在 host 侧的落点，也是本次预取拼接的锚点。
            # 中译：只有当该锚点在 host 上确有备份（backuped），或它就是树根（root，表示没有任何
                # 中译：取锚点对应的最后一个页哈希值，作为存储后端定位「接下来这段前缀」的起始 key。
                # 中译：已命中的总前缀长度 = device/L1 命中(prefix_indices) + 仅 host/L2 命中(host_hit_length)。
                # 中译：从完整（未截断）的输入 token 序列中，切掉已命中的前缀，剩下的就是「本地缺失、
                # 中译：可选地为存储后端准备「前缀哈希链」。某些后端（如 mooncake）在按页 key 查询时
            tree_cache = self.tree_cache
            buffer_mode = get_memory().hicache_host_memory_mode == "buffer_only"
            last_host_node = req.last_host_node
            # Buffer mode host-backups nothing, so match_prefix anchors at
            # root; re-anchor on the deepest device node. The anchor is only
            # read for hash/extra-key context here (never locked), so a device
            # node serves. Cache mode keeps the is_backuped gate below: its
            # write-through prefix is contiguous from root, so an unbacked
            # anchor means a guaranteed storage miss.
            if (
                buffer_mode
                and tree_cache.is_root(last_host_node)
                and not tree_cache.is_root(req.last_node)
            ):
                last_host_node = req.last_node

            if (
                tree_cache.is_backuped(last_host_node)
                or tree_cache.is_root(last_host_node)
                or (
                    buffer_mode
                    and tree_cache.get_last_hash_value(last_host_node) is not None
                )
            ):
                matched_len = len(req.prefix_indices) + req.host_hit_length
                match_end = req._compute_max_prefix_len(
                    len(req.full_untruncated_fill_ids)
                )
                new_input_tokens = req.full_untruncated_fill_ids[matched_len:match_end]
                prefix_keys = (
                    tree_cache.get_prefix_hash_values(last_host_node)
                    if tree_cache.hicache_storage_pass_prefix_keys
                    else None
                )
                # 中译：发起异步预取：在 host 池中为缺失前缀分配落地空间并交给 cache_controller 后台搬运。
                tree_cache.prefetch_from_storage(
                    req.rid,
                    last_host_node,
                    new_input_tokens,
                    tree_cache.get_last_hash_value(last_host_node),
                    prefix_keys,
                    matched_prefix_tokens=req.full_untruncated_fill_ids[:matched_len],
                    extra_key=req.extra_key,
                    cache_salt=req.cache_salt,
                )

    def _retry_missed_storage_prefetches(self):
        """Re-issue the availability check for queued requests whose prefetch
        missed. Pacing counts scheduling passes so TP ranks re-issue on the
        same pass; a sweep (the admission loop stops at the first
        unschedulable request) covers the whole queue.

        中译：对「存储层（L3）预取未命中」的排队请求，按节奏重新发起一次可用性检查与预取。
              背景：HiCache 分三级——L1 设备显存、L2 主机内存、L3 外部存储（如 Mooncake/3FS）。
              请求入队时会调用 _prefetch_kvcache 尝试把 L3 中的前缀 KV 预取到 L2；但预取可能被
              拒绝或放弃（缺失后缀太短、预取限流、staging 空间紧张等），此时 tree_cache 会记下
              该 rid 的「miss」标记。请求在 waiting_queue 中排队期间，存储侧状态可能变化
              （限流或 staging 压力解除；设备侧匹配被驱逐，原本太短的缺口变长、值得预取），
              重试一次预取就可能省下一次完整的 prefill 重算。
              节奏设计：
              - 以「调度轮次」而非墙钟时间计数：各 TP rank 调度轮次一致，保证在同一轮重发，
                避免 rank 间对同一请求的前缀命中长度不一致；
              - 遍历整个 waiting_queue：组批循环遇到第一个无法调度的请求就会停，若只在组批时
                重试，排在后面的请求永远轮不到，因此这里单独扫一遍全队列。
        """
        # 中译：两次重试之间需等待的调度轮数；<=0 表示关闭该功能（默认 0，即关闭）。
        interval = get_memory().hicache_storage_prefetch_retry_poll_interval
        # 中译：功能关闭或没有排队请求时直接返回，不做任何工作。
        if interval <= 0 or not self.waiting_queue:
            return
        # 中译：每个请求最多重试的次数（默认 4），防止对确实不存在的前缀无限重试。
        max_attempts = get_memory().hicache_storage_prefetch_retry_max_attempts
        # 中译：扫描整个等待队列（而不只是本轮能被组批的前缀部分）。
        for req in self.waiting_queue:
            # 中译：弹出该请求的「预取未命中」标记（弹出即清除，避免重复计入）；
            #       UnifiedRadixCache 会记录该标记；HiRadixCache（dense 路径）不跟踪，恒返回 False。
            if self.tree_cache.pop_storage_prefetch_miss(req.rid):
                # 中译：本轮新发生 miss，标记为「待重试」。
                req.storage_prefetch_retry_pending = True
                # 中译：等待计数清零，从本轮起重新计算间隔；即便之前在等，也以最新一次 miss 为准。
                req.storage_prefetch_retry_wait_polls = 0
            # 中译：没有待重试的 miss，或已用尽重试次数，则跳过该请求。
            if (
                not req.storage_prefetch_retry_pending
                or req.storage_prefetch_retry_attempts >= max_attempts
            ):
                continue
            # 中译：已等待的调度轮数 +1。
            req.storage_prefetch_retry_wait_polls += 1
            # 中译：未满间隔则继续等，给存储侧状态变化留出时间，也避免高频重试压垮存储后端。
            if req.storage_prefetch_retry_wait_polls <= interval:
                continue
            # 中译：间隔已满，清除待重试标记；若本次重试再次 miss，tree_cache 会重新打标记。
            req.storage_prefetch_retry_pending = False
            # 中译：累计重试次数，用于与 max_attempts 比较。
            req.storage_prefetch_retry_attempts += 1
            # 中译：调试日志，记录请求 rid 与第几次重试。
            logger.debug(
                "HiCache storage prefetch retry req=%s attempt=%d",
                req.rid,
                req.storage_prefetch_retry_attempts,
            )
            # 中译：重新执行一次前缀匹配与 L3→L2 异步预取；结果在后续组批时经预取进度检查生效。
            self._prefetch_kvcache(req)

    def _add_request_to_queue(self, req: Req, is_retracted: bool = False):
        # 中译：把请求送入相应队列。非分离模式进 waiting_queue；PD 分离的 prefill 节点进
        #       bootstrap 队列、decode 节点进 prealloc 队列。is_retracted 表示这是一个被回退
        #       （retract，因显存不足从 running batch 踢出）后重新入队的请求。
        if not self._set_or_validate_priority(req):
            return
        if self.disaggregation_mode == DisaggregationMode.NULL:
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
        """Set the default priority value, or abort the request based on the priority scheduling mode."""
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
            abort_req = _make_abort_req(
                req,
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                    "message": "Using priority is disabled for this server. Please send a new request without a priority.",
                },
            )
            req.time_stats.trace_ctx.abort(abort_info=abort_req.finished_reason)
            self.ipc_channels.send_to_tokenizer.send_output(abort_req, req)
            return False
        return True

    def _abort_on_queued_limit(self, recv_req: Req) -> bool:
        """Abort an incoming or existing request if the waiting queue is full. Returns True if the incoming request is aborted."""
        if (
            self.max_queued_requests is None
            or len(self.waiting_queue) + 1 <= self.max_queued_requests
        ):
            return False

        # Reject the incoming request by default.
        req_to_abort = recv_req
        message = "The request queue is full."
        if self.enable_priority_scheduling:
            # With priority scheduling, consider aboritng an existing request based on the priority.
            # direction = 1  => smaller number = higher priority; -1 => larger number = higher priority.
            # max(...) + (direction * priority, queue_time_start) picks the least-preferred request.
            # Tie: later queue_time_start (newer) is evicted first. Preempt only if strictly better.
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
                    # Release prefetch events associated with the request
                    self.tree_cache.release_aborted_request(candidate_req.rid)
                self.waiting_queue.pop(idx)
                self.beam_coordinator.retire_group(candidate_req)
                req_to_abort = candidate_req
                message = "The request is aborted by a higher priority request."

        self.ipc_channels.send_to_tokenizer.send_output(
            _make_abort_req(
                req_to_abort,
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                    "message": message,
                },
            ),
            req_to_abort,
        )
        req_to_abort.time_stats.trace_ctx.abort(abort_info={"reason": message})
        return req_to_abort.rid == recv_req.rid

    def _abort_on_waiting_timeout(self):
        # 中译：中止"等待过久"的排队请求——即在 waiting_queue 中排队、尚未被调度进 prefill 的请求
        #       从入队到现在耗时超过阈值的。与 _abort_on_running_timeout 不同：这些请求还没分配 KV、
        #       没开始前向，可以直接中止并从队列移除，无需走延迟标记。
        # 中译：读取等待超时阈值（秒）。<=0 表示功能关闭，直接返回（海象赋值顺带取值）。
        if (timeout_s := envs.SGLANG_REQ_WAITING_TIMEOUT.get()) <= 0:
            return

        deleted_reqs = set()  # 本轮被超时中止、待从队列剔除的请求集合
        # 中译：deadline 同上——入队时间早于它即视为等待超时（已等待 > timeout_s）。
        deadline = time.perf_counter() - timeout_s
        for req in self.waiting_queue:
            # 中译：wait_queue_entry_time 是该请求进入等待队列的时刻；>0 表示已正式入队。
            entry_time = req.time_stats.wait_queue_entry_time
            if 0 < entry_time < deadline:
                if self.enable_hicache_storage:
                    # Release prefetch events associated with the request
                    # 中译：开启分层缓存存储时，排队期间可能已发起 KV 预取（prefetch）；
                    #       中止前需释放与该请求关联的预取事件，避免悬挂资源。
                    self.tree_cache.release_aborted_request(req.rid)
                # 中译：等待中的请求无在途前向，可直接向 tokenizer 发送 AbortReq
                #       （503 SERVICE_UNAVAILABLE），由其向客户端返回中止结果。
                self.ipc_channels.send_to_tokenizer.send_output(
                    _make_abort_req(
                        req,
                        finished_reason={
                            "type": "abort",
                            "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                            "message": "Request waiting timeout reached.",
                        },
                    ),
                    req,
                )
                deleted_reqs.add(req)
                self.beam_coordinator.retire_group(req)

        # 中译：一次性重建等待队列，过滤掉所有被中止的请求（避免在遍历中修改列表）。
        if deleted_reqs:
            self.waiting_queue = [
                req for req in self.waiting_queue if req not in deleted_reqs
            ]

    def handle_embedding_request(
        self,
        recv_req: TokenizedEmbeddingReqInput,
    ):
        req = Req(
            recv_req.rid,
            recv_req.input_text,
            recv_req.input_ids,
            recv_req.sampling_params,
            positional_embed_overrides=recv_req.positional_embed_overrides,
            token_type_ids=recv_req.token_type_ids,
            routed_dp_rank=recv_req.routed_dp_rank,
            priority=recv_req.priority,
            dimensions=recv_req.dimensions,
            lora_id=recv_req.lora_id,
            http_worker_ipc=recv_req.http_worker_ipc,
            time_stats=recv_req.time_stats,
            return_pooled_hidden_states=recv_req.return_pooled_hidden_states,
            multi_item_delimiter_indices=recv_req.multi_item_delimiter_indices,
        )
        req.tokenizer = self.tokenizer
        self._maybe_namespace_elastic_radix_cache(req)

        # Handle multimodal inputs
        if recv_req.mm_inputs is not None:
            image_inputs = self._get_multimodal_inputs(recv_req.mm_inputs)
            # Expand a single image token into multiple dummy tokens for receiving image embeddings
            # The `pad_input_ids_func` is model-specific and may be None for
            # embedding models or models not requiring special padding.
            # If None, `req.origin_input_ids` is expected to be correctly populated already.
            if (
                not self._try_apply_padded_mm_input_ids(recv_req, req, image_inputs)
                and self.pad_input_ids_func
            ):
                # See companion call site above for the array.array wrap rationale.
                req.origin_input_ids = array(
                    "q", self.pad_input_ids_func(req.origin_input_ids, image_inputs)
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

        # Validate prompts length
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            get_serving().allow_auto_truncate,
        )
        if error_msg:
            self._add_request_to_queue(req)
            return

        # Copy more attributes
        req.logprob_start_len = -1
        self._add_request_to_queue(req)

    def handle_batch_embedding_request(
        self,
        recv_req: BatchTokenizedEmbeddingReqInput,
    ):
        """Handle optimized batch embedding request."""
        logger.debug(
            f"Processing batch embedding request with {len(recv_req)} requests"
        )

        # Process each request in the batch
        for tokenized_req in recv_req:
            self.handle_embedding_request(tokenized_req)

    def stash_chunked_request(self, req: Req):
        maybe_cache_unfinished_req(req, self.tree_cache, chunked=True)

    def process_pending_chunked_abort(self) -> None:
        """Abort an in-flight chunked-prefill request once it is safe to do so.

        ``abort_request`` only records the target in ``_pending_chunked_abort_req``
        (tearing it down mid-iteration is unsafe). Clearing ``chunked_req`` here at
        the top of the scheduling step stops the next chunk from launching; the
        chunk already launched is drained when its result is resolved. Under overlap
        the result lands a step later, so the batch-result processors keep
        ``inflight_middle_chunks`` accounting intact and skip the aborted chunk:
        ``process_batch_result_disagg_prefill`` via its ``is_aborted`` drop, and
        ``process_batch_result_prefill`` via its chunked branch (the finished req
        is excluded from streaming and its logprob offset is still accounted).
        Mirrors ``handle_bootstrap_failure``.
        """
        req = self._pending_chunked_abort_req
        if req is None:
            return
        if self.chunked_req is not req:
            # Already past chunked prefill; the running-batch abort path handles
            # it. Drop the marker once the request is actually gone.
            if req.finished() or req.kv.req_pool_idx is None:
                self._pending_chunked_abort_req = None
            return

        prepare_abort(req, "Aborted")
        req.time_stats.trace_ctx.abort(abort_info={"reason": "Aborted"})
        req.to_finish = None
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.clear_pending_chunk_send(req)
            req.disagg_kv_sender.abort()
            maybe_release_metadata_buffer(
                req, self.req_to_metadata_buffer_idx_allocator
            )
            req.pending_bootstrap = False
        if self.enable_hicache_storage:
            self.tree_cache.release_aborted_request(req.rid)
        release_kv_cache(req, self.tree_cache, is_insert=False)

        self.chunked_req = None
        self._pending_chunked_abort_req = None
        self.ipc_channels.send_to_tokenizer.send_output(_make_abort_req(req), req)
        logger.debug(f"Abort chunked prefill request. {req.rid=}")

    def _build_hisparse_decode_batch(self, reqs):
        """Build a ScheduleBatch for hisparse requests transitioning from staging to decode."""
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

        req_pool_indices = [r.kv.req_pool_idx for r in reqs]
        batch.req_pool_indices = torch.tensor(
            req_pool_indices, dtype=torch.int64, device=device
        )
        batch.req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
        seq_lens = [len(r.origin_input_ids) + len(r.output_ids) - 1 for r in reqs]
        batch.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=device)
        batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        batch.seq_lens_sum = sum(seq_lens)
        # Stash last token into relay; resolve_forward_inputs will gather.
        last_tokens = torch.tensor(
            [r.output_ids[-1] for r in reqs], dtype=torch.int64, device=device
        )
        self.future_map.stash(
            batch.req_pool_indices, RelayPayload(bonus_tokens=last_tokens)
        )
        batch.input_ids = None

        if batch.return_logprob:
            batch.top_logprobs_nums = [r.logprob.top_logprobs_num for r in reqs]
            batch.token_ids_logprobs = [list(r.origin_input_ids) for r in reqs]

        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, self.model_config.vocab_size
        )
        # todo hisparse, maybe other info to contain for the new batch
        return batch

    @scheduler_nvtx_method("scheduler.get_next_batch_to_run")
        # 中译：调度器最核心的决策函数——决定本轮迭代到底跑哪个批次。总体策略是"prefill 优先"：
        # 中译：FPM（forward pass metrics）记录本批调度起始时刻。
    def get_next_batch_to_run(
        self, running_batch: ScheduleBatch, last_batch: Optional[ScheduleBatch]
    ) -> NextBatchPlan:
        self.process_pending_chunked_abort()

        if self.enable_fpm:
            self._fpm_batch_t0 = time.monotonic()
        # 中译：先按超时规则中止等待过久 / 运行过久的请求。
        self._abort_on_waiting_timeout()
        self._abort_on_running_timeout(running_batch)
        if self.dllm_config is not None:
            self.dllm_manager.filter_finished_reqs()

        # Merge the prefill batch into the running batch
        # 中译：把上一批的 prefill 结果合并进 running_batch。但"分块请求（chunked_req）"还没
        #       prefill 完，要先排除在外，避免它被当成已完成请求并入 decode。
        chunked_req_to_exclude = set()

        if self.dllm_config is not None and self.dllm_manager.any_staging_reqs():
            chunked_req_to_exclude.update(self.dllm_manager.staging_queue)
            for req in self.dllm_manager.staging_queue:
                if self.dllm_config.first_done_first_out_mode:
                    if not req.dllm_incomplete_ids:
                        self.stash_chunked_request(req)
                        self.req_to_token_pool.free(req)
                    # Otherwise, keep req slot/KV for reuse.
                else:
                    self.stash_chunked_request(req)

        if self.chunked_req is not None:
            # Move the chunked request out of the batch so that we can merge
            # only finished requests to running_batch.
            # 中译：把分块请求移出批次，这样合并进 running_batch 的就只有真正 prefill 完成的请求。
            #       分块请求只 prefill 了一个 chunk、尚未完成，不能进入 decode 阶段。
            chunked_req_to_exclude.add(self.chunked_req)

            # Stash (cache) the previous chunk only when it produced new KV
            # beyond what is already cached. A parked chunk (add_chunked_req
            # 中译：仅当本 chunk 产生了超出已缓存范围的新 KV 时，才把它 stash（缓存）进 radix tree。
            # hybrid-SWA early-return) leaves extend_range.end ==
            # len(prefix_indices), so there is nothing new to cache and
            # stashing would be a no-op.
            if self.chunked_req.extend_range.end > len(self.chunked_req.prefix_indices):
                self.stash_chunked_request(self.chunked_req)

        # HiSparse has its own prefill-to-decode transition; skip last_batch merge.
        # 中译：HiSparse（分层稀疏注意力）有自己的 prefill→decode 转换逻辑，跳过常规的 last_batch 合并。
        #       它通过 hisparse_coordinator 收集已就绪的请求，单独构建 decode 批并入 running_batch。
        if self.enable_hisparse:
            ready_reqs = self.hisparse_coordinator.collect_ready_reqs()
            if len(ready_reqs) > 0:
                new_batch = self._build_hisparse_decode_batch(ready_reqs)
                if running_batch.is_empty():
                    running_batch = new_batch
                else:
                    running_batch.merge_batch(new_batch)
                running_batch.hisparse_coordinator = self.hisparse_coordinator
            # Reset batch_is_full so the scheduler can schedule more prefills.
            # 中译：重置 batch_is_full 标记，让调度器后续还能继续攒入更多 prefill 请求。
            running_batch.batch_is_full = False

        if (
            not self.enable_hisparse
            and last_batch
            and last_batch.forward_mode.is_extend()
        ):
            if last_batch.chunked_req is not None:
                # In the context pipeline parallelism, after the last chunk, the current microbatch still track outdated chunked_req.
                # We need to discard it.
                # 中译：在上下文流水线并行（context PP）下，最后一个 chunk 跑完后，当前微批仍可能持有
                chunked_req_to_exclude.add(last_batch.chunked_req)

            if self.dllm_config is not None and last_batch.reqs:
                chunked_req_to_exclude.update(last_batch.reqs)

            # Filter batch
            # 中译：从上一批中过滤掉需要排除的分块请求，并记录过滤前后的批大小。
            # 中译：若批大小变小（有请求被过滤出去），说明腾出了名额，重置 batch_is_full。
            # 中译：把上一批里已完成 prefill 的请求并入 running_batch，使其进入 decode 阶段。
                    # 中译：running_batch 非空时，把上一批 prefill 完成的请求合并进去。
            last_bs = last_batch.batch_size()
            last_batch.filter_batch(chunked_req_to_exclude=list(chunked_req_to_exclude))
            if last_batch.batch_size() < last_bs:
                running_batch.batch_is_full = False

            # Merge the new batch into the running batch.
            if not last_batch.is_empty():
                if running_batch.is_empty():
                    running_batch = last_batch
                else:
                    # Merge running_batch with prefill batch
                    running_batch.merge_batch(last_batch)

        # For prefill-only batch, filter out finished requests since they
        # won't go through the decode step. This keeps running_batch accurate
        # for load reporting (num_running_reqs via /v1/loads).
        # Runs outside the last_batch block so stale requests are cleaned
        # even when no new batches arrive (e.g. traffic stops).
        # 中译：对于 prefill-only 批次（如纯 embedding 请求），其请求不会经过 decode 步骤，
        if running_batch.is_prefill_only:
            running_batch.filter_batch()
            if running_batch.is_empty():
                running_batch.batch_is_full = False

        # 中译：尝试从等待队列里攒一个新的 prefill 批（扩散 LLM 走专用路径）。
        if self.dllm_config is not None:
            new_batch = self.get_new_batch_dllm(running_batch)
        elif self._should_defer_prefill():
            new_batch = None
        else:
            prefill_plan = self.get_new_batch_prefill(running_batch)
            new_batch = prefill_plan.batch_to_run
            running_batch = prefill_plan.running_batch

        # 中译：DP attention / MoE 等需要各 DP rank 在每步同步 MLP 计算（require_mlp_sync）。
        need_mlp_sync = self.require_mlp_sync
        if (
            need_mlp_sync
            and not self.spec_algorithm.is_none()
            and not get_spec().speculative_skip_dp_mlp_sync
        ):
            # NOTE: This branch makes sure prefill and decode batches will not be mixed when spec and dp-attn is enabled.
            # Before merging the new batch into running batch:
            # 1. All new batches are none -> need_mlp_sync remains true (sync is needed for decode batch).
            # 2. All new batches are some (prefill / idle) -> we do not need prepare mlp sync one more time.
            # 中译：当同时启用投机解码（spec）和 DP attention 时，本分支确保 prefill 批与 decode 批
            #       不会被混在同一次同步里。在把新批并入 running batch 之前：
            #       1. 所有 DP rank 的新批都为 None → need_mlp_sync 保持 True（decode 批仍需同步）；
            #       2. 所有 DP rank 的新批都非 None（prefill 或 idle）→ 已经同步过，无需再次准备 MLP 同步。
            new_batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(new_batch)
            need_mlp_sync = new_batch is None

        if new_batch is not None:
            # Run prefill first if possible
            # 中译：prefill 优先——只要攒出了 prefill 批就先跑它。
            ret = new_batch
        else:
            # Run decode (skip for prefill-only batches)
            # 中译：没有可跑的 prefill，则对 running_batch 跑一步 decode（纯 prefill 请求不参与）。
                # 中译：update_running_batch 会过滤已完成请求、必要时回退请求，再准备 decode 张量。
            if not running_batch.is_empty() and not running_batch.is_prefill_only:
                running_batch = self.update_running_batch(running_batch)
                ret = running_batch if not running_batch.is_empty() else None
            else:
                ret = None

        # Handle DP attention and log stats
        # 中译：处理 DP attention 同步——若本轮仍需 MLP 同步（need_mlp_sync 为 True，通常是 decode 批），
        #       在此为各 DP rank 准备同步批（必要时插入 idle 批，保证所有 rank 步调一致、避免死锁）。
        ret = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(
            ret, need_sync=need_mlp_sync
        )
        # Decode->extend conversion keeps a heterogeneous dp step replayable.
        converted = self.dp_attn_adapter.maybe_convert_decode_to_extend(ret)
        if converted is running_batch and converted.forward_mode.is_extend():
            # The converted batch re-enters via the last_batch extend-merge
            # next iteration; empty running_batch or it merges with itself.
            running_batch = ScheduleBatch(
                reqs=[], batch_is_full=running_batch.batch_is_full
            )
        ret = converted
        self._arm_prefill_decode_interval(ret)

        # Handle ngram embedding
        # 中译：若启用了 ngram embedding，在前向之前填好 token 查找表。
        ret = self.ngram_embedding_manager.prepare_for_forward(
            ret, chunked_req=self.chunked_req
        )

        # 中译：为最终选定的批记录调度时刻（用于时延统计）；启用 FPM 时还记录本批调度起始时间。
        if ret:
            set_schedule_time_batch(ret)
            if self.enable_fpm:
                ret.fpm_start_time = self._fpm_batch_t0

        # 中译：返回本轮要跑的批次。可能是 prefill 批、decode 批、DP 同步用的 idle 批，或 None（空闲）。
        return NextBatchPlan(batch_to_run=ret, running_batch=running_batch)

    def get_num_allocatable_reqs(
        self,
        running_bs: int,
        beam_width: Optional[int] = None,
        running_batch: Optional[ScheduleBatch] = None,
    ) -> int:
        # 中译：计算本轮还能再接纳多少个新请求进入 prefill 批（请求个数名额，不是 token 数）。
        #       结果取「单个 micro-batch 的请求数预算」与「req_to_token_pool 剩余槽位」的较小值；
        #       组批时 len(adder.can_run_list) 达到该值即视为批已满（batch_is_full）。
        #       running_bs：当前 running batch 中的请求数；
        #       beam_width：候选请求为 beam search 时的束宽，一个请求在 decode 时要占 beam_width 行；
        #       running_batch：用于统计 beam 待占行数的批，缺省用 self.running_batch。
        # 中译：请求数预算 = 每个 micro-batch 的最大请求数 - 已在运行的请求数。
        #       pp_max_micro_batch_size 未配置时默认为 max_running_requests // pp_size（非 PP 时即 max_running_requests）。
        pp_budget = get_parallel().pp_max_micro_batch_size - running_bs
        # 中译：req_to_token_pool 的剩余槽位数。每个在途请求占一个槽位（req_pool_idx）；
        #       PD prefill 端 inflight 队列中等待 KV 传输完成的请求也仍占着槽位。
        available = self.req_to_token_pool.available_size()

        # 中译：扣除 beam search 已接纳、但成员行尚未展开的组将要占用的槽位
        #       （每组还需 beam_width - 1 行），避免槽位被超额承诺；结果不低于 0。
        active_batch = running_batch or self.running_batch
        available = max(
            available - self.beam_coordinator.pending_member_rows(active_batch), 0
        )
        # 中译：两类约束取较小值。
        res = min(pp_budget, available)
        if beam_width is not None:
            # A beam candidate owns beam_width rows once decoding.
            # 中译：beam 候选请求进入 decode 后要占 beam_width 行，因此可接纳数还要除以束宽。
            res = min(res, available // beam_width)

        return res

        # 中译：构造新 prefill 批的对外入口。在真正攒批（_get_new_batch_prefill_raw）外面
    def get_new_batch_prefill(self, running_batch: ScheduleBatch) -> NextBatchPlan:
        prefill_delayer_single_pass = None
        if self.prefill_delayer:
            # Get max usage across all pools for prefill delay decision
            max_pool_usage = (
                self.pool_stats_observer.get_pool_stats().get_max_pool_usage()
            )
            prefill_delayer_single_pass = PrefillDelayerSinglePassExecutor(
                self.prefill_delayer, token_usage=max_pool_usage
            )

        ret, running_batch = self._get_new_batch_prefill_raw(
            prefill_delayer_single_pass=prefill_delayer_single_pass,
            running_batch=running_batch,
        )

        if self.prefill_delayer:
            observed_prefill_bs = prefill_delayer_single_pass.finalize(
                actual_prefill_bs=ret.batch_size() if ret is not None else 0
            )
            if observed_prefill_bs > 0:
                self.max_prefill_bs = self.prefill_bs_tracker.observe_attempt(
                    observed_prefill_bs
                )

        return NextBatchPlan(batch_to_run=ret, running_batch=running_batch)

    def _get_new_batch_prefill_raw(
        # 中译：真正的 prefill 攒批逻辑。流程概要：
        self,
        prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor],
        running_batch: ScheduleBatch,
    ) -> Tuple[Optional[ScheduleBatch], ScheduleBatch]:
        # Check if the grammar is ready in the grammar queue
        # 中译：检查语法队列里是否有已就绪（编译完成）的请求，有则放回等待队列参与本次调度。
        if self.grammar_manager.has_waiting_grammars():
            ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
            for req in ready_grammar_requests:
                self._add_request_to_queue(req)

        # 中译：启用分层缓存（hierarchical cache）时，推进一次 HiCache 事件（主机↔设备间的
        if self.enable_hierarchical_cache or get_memory().enable_flexkv:
            self.tree_cache.check_hicache_events()
            if self.enable_hicache_storage:
                self._retry_missed_storage_prefetches()

        if self.enable_priority_preemption or self.is_hybrid_swa:
            # Reset batch_is_full to try preemption with a prefill adder.
            # 中译：启用优先级抢占或 hybrid-SWA 时，先重置 batch_is_full，以便后面用 PrefillAdder
            running_batch.batch_is_full = False

        # 中译：批已满 或 没有等待请求，且没有未完成的分块请求 → 本轮无新 prefill 可攒。
        if (
            running_batch.batch_is_full or len(self.waiting_queue) == 0
        ) and self.chunked_req is None:
            return None, running_batch

        # 中译：当前 running batch 大小。DFLASH 投机解码下，若正在“为攒批而延迟 prefill”（让
        running_bs = len(running_batch.reqs)
        # Skipped during a chunked prefill: that pass must proceed regardless.
        if (
            self.min_free_slots_delayer is not None
            and self.chunked_req is None
            and self.min_free_slots_delayer.should_delay(
                running_bs=running_bs,
                num_allocatable_reqs=self.get_num_allocatable_reqs(
                    running_bs, running_batch=running_batch
                ),
            )
        ):
            return None, running_batch

        # Ignore the check if self.chunked_req is not None.
        # In the non-PP case, when self.chunked_req is not None, num_allocatable_reqs should always be greater than 0,
        # as the space for the chunked requests has just been released.
        # In PP case, chunked requests (or dllm requests) can start in one microbatch and end in another microbatch, so the max_running_requests per microbatch should not be strict.
        # Instead, we should always allow chunked requests to be added, otherwise, there will be a memory leak.
        # 中译：可分配名额检查。若“可再接纳的请求数 <= 0”且无未完成分块请求、未开抢占，
        #       则标记批已满并返回 None。但如果有 chunked_req 则跳过该检查：
        #       - 非 PP 场景：chunked_req 存在时刚释放了其空间，num_allocatable_reqs 应总 > 0；
        #       - PP 场景：分块/dllm 请求可能跨微批跳始跳终，每微批的 max_running_requests 不应严格限制，
        #         必须始终允许分块请求加入，否则会造成显存泄漏。
        if (
            self.get_num_allocatable_reqs(running_bs, running_batch=running_batch) <= 0
            and self.chunked_req is None
            and not self.enable_priority_preemption
        ):
            running_batch.batch_is_full = True
            return None, running_batch

        # Get priority queue
        # 中译：按调度策略（FCFS/LPM/优先级等）对等待队列排序，决定接下来优先考虑哪些请求。
        self.policy.calc_priority(self.waiting_queue, running_batch)

        if TEST_RETRACT and running_bs > TEST_RETRACT_NO_PREFILL_BS:
            # If we are testing retraction and the running batch size exceeds
            # TEST_RETRACT_NO_PREFILL_BS, we skip the prefill to keep the requests
            # in the waiting queue.
            # 中译：回退（retract）测试专用分支——当 running batch 大小超过阈值时跳过 prefill，
            return None, running_batch

        # Determine chunked_prefill_size for this batch
        # 中译：确定本批的分块大小。默认用静态配置 chunked_prefill_size；若启用了 PP 动态分块
        #       且存在未完成分块请求，则根据已 prefill 的历史长度预测下一个 chunk 的最优大小。
        chunked_prefill_size = self.chunked_prefill_size
        if self.chunked_req is not None and self.enable_dynamic_chunking:
            history_len = len(self.chunked_req.prefix_indices)
            dynamic_size = self.predict_next_chunk_size(history_len)
            if dynamic_size is not None:
                chunked_prefill_size = dynamic_size

        # Prefill policy
        # 中译：PrefillAdder 封装了"在当前预算下能否再加一个请求"的全部判定逻辑
        # Get BLOCK_M from the backend for tile-budget admission logic
        attn_backend = self.tp_worker.model_runner.attn_backend
        if hasattr(attn_backend, "extend_attention_block_m"):
            prefill_tile_block_m = attn_backend.extend_attention_block_m
        else:
            prefill_tile_block_m = 64  # Fallback for non-Triton backends

        adder = PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            running_batch,
            self.new_token_ratio_tracker.current,
            self.max_prefill_tokens,
            chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
            self.priority_scheduling_preemption_threshold,
            max_prefill_bs=int(self.max_prefill_bs),
            max_running_requests=self.max_running_requests,
            prefill_max_requests=get_schedule().prefill_max_requests,
            prefill_delayer_single_pass=prefill_delayer_single_pass,
            dllm_config=self.dllm_config,
            waiting_queue_len=len(self.waiting_queue),
            prefill_tile_block_m=prefill_tile_block_m,
        )

        # 中译：若存在未完成的分块请求，优先把它加进本批（继续 prefill 下一个 chunk），
        #       避免分块请求被新请求饥饿。init_next_round_input 重算本轮要填的 token。
        if self.chunked_req is not None:
            self.chunked_req.init_next_round_input()
            self.chunked_req = adder.add_chunked_req(self.chunked_req)

        # 中译：LoRA 场景：统计当前 running batch 中未完成请求所用的 adapter 集合，后续用于限制
        #       同一批内 adapter 数量（max_loras_per_batch）。
        if self.enable_lora:
            running_loras = {
                req.lora_id for req in running_batch.reqs if not req.finished()
            }
            # Account for LoRAs that are already loaded in the adder, such as chunked requests
            # 中译：把 adder 中已加载的 adapter（如分块请求的）也纳入集合。
            running_loras.update(req.lora_id for req in adder.can_run_list)

            # 中译：LoRA drainer 负责公平调度：更新各 adapter 的“排空（draining）”状态，避免某个
            #       adapter 长期占用位置、其他 adapter 请求饥饿。
            if self.lora_drainer:
                self.lora_drainer.update_draining_state(
                    self.waiting_queue,
                    running_batch.reqs,
                )

        # 中译：Mamba/混合线性模型的状态空间分配器（若有）。alloc_group_begin/end 把本轮的
        #       多个分配归为一组，便于失败时整组回滚。
        mamba_allocator = getattr(self.req_to_token_pool, "mamba_allocator", None)
        if mamba_allocator is not None:
            mamba_allocator.alloc_group_begin(len(self.waiting_queue))
        # Get requests from the waiting queue to a new prefill batch
        # 中译：从等待队列中逐个取出请求去组装新的 prefill 批。
        # 中译：遍历等待队列，逐个尝试把请求加入本次 prefill 批（adder.can_run_list）。
        #       一旦预算耗尽（batch_is_full 且无法抢占）就 break。
        for req in self.waiting_queue:
            # 中译：LoRA 约束——若该请求的 adapter 当前不可调度（超出同批 adapter 数上限等）则跳过。
            if self.enable_lora and not self._can_schedule_lora_req(req, running_loras):
                continue

            # 中译：已选请求数达到可分配名额上限，标记批已满。
            running_bs = len(running_batch.reqs)
            candidate_beam_width = (
                req.beam_group.beam_width if req.beam_group is not None else None
            )
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(
                running_bs,
                candidate_beam_width,
                running_batch=running_batch,
            ):
                running_batch.batch_is_full = True
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                # In prefill mode, prealloc queue and transfer queue can also take memory,
                # so we need to check if the available size for the actual available size.
                # 中译：PD 分离的 prefill 模式下，prealloc 队列与 transfer 队列也会占用 req_to_token_pool，
                #       所以还要额外检查该池的可用名额，不能只看名额上限。
                if len(adder.can_run_list) >= self.req_to_token_pool.available_size():
                    running_batch.batch_is_full = True

            # 中译：批已满：若未开抢占、或抢占也无法为该请求腾出空间，则停止遥历等待队列。
            if running_batch.batch_is_full:
                if (
                    not self.enable_priority_preemption
                    or not adder.preempt_to_schedule(req)
                ):
                    break

            # 中译：启用 HiCache 存储（L3）时，检查该请求从存储预取 KV 的进度。
            if self.enable_hicache_storage:
                prefetch_done = self.tree_cache.check_prefetch_progress(req.rid)
                if not prefetch_done:
                    # skip staging requests that are ongoing prefetch
                    # 中译：预取未完成的请求先跳过，等下轮再调度。
                    continue
                # Pop the number of tokens loaded from storage (L3 hits)
                # 中译：取出从存储（L3 命中）加载回来的 token 数，计入该请求的命中长度。
                loaded_tokens = self.tree_cache.pop_prefetch_loaded_tokens(req.rid)
                if loaded_tokens > 0:
                    req.storage_hit_length = loaded_tokens

            # 中译：计算该请求本轮要喂入的 token（结合 radix tree 前缀复用，跳过已缓存前缀）。
            req.init_next_round_input(self.tree_cache)
            # 中译：尝试把该请求加入本批，返回结果指示成功/因 token 不足/因预算满等。
            if (
                self.enable_hicache_storage
                and get_memory().hicache_host_memory_mode == "buffer_only"
            ):
                # Buffer mode: surface a staged prefetch as the request's host
                # hit (consumed through init_load_back) plus its SWA window,
                # which consumption allocates and the request lock pins —
                # uncharged, the batch alloc can OOM. Planned against the same
                # live prefix admission uses, so only the splice-able span
                # tail is charged and unusable holds are freed. Set AFTER
                # init_next_round_input (which recomputes host_hit). Mamba
                # (fenced in init_hicache) will need the same charge via
                # mamba_host_hit_length.
                held_tokens, held_swa_tokens = self.tree_cache.plan_staged_splice(
                    req.rid, len(req.prefix_indices)
                )
                if held_tokens > 0:
                    req.host_hit_length = held_tokens
                    req.swa_host_hit_length = held_swa_tokens
            res = adder.add_one_req(
                req,
                has_chunked_req=(self.chunked_req is not None),
                truncation_align_size=self.truncation_align_size,
            )

            if self.enable_lora:
                running_loras.add(req.lora_id)

            # 中译：返回值非 CONTINUE 表示本请求未能加入（预算耗尽），需收尾并 break。
            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    if self.enable_hierarchical_cache:
                        # Set batch_is_full after making sure there are requests that can be served
                        # 中译：分层缓存下，只有“确实有请求能被服务”时才置 batch_is_full，
                        running_batch.batch_is_full = len(adder.can_run_list) > 0 or (
                            not running_batch.is_empty()
                        )
                    else:
                        running_batch.batch_is_full = True
                # revert matched mamba idx to avoid memory leak, if req is not added.
                # Only free if the slot was freshly allocated in this batch (not
                # pre-existing from a session). Session-held slots have their own
                # lifecycle and freeing them here causes double-free.
                # 中译：若该请求最终未加入本批，需回收它在本轮新分配的 mamba 槽位以防显存泄漏。
                #       仅当该槽位是本轮新分配（而非会话 session 预先持有）时才释放——
                #       session 持有的槽位有自己独立的生命周期，在此释放会造成 double-free。
                added = len(adder.can_run_list) > 0 and req is adder.can_run_list[-1]
                if not added:
                    # init_next_round_input() may stage deferred Mamba COW/clear
                    # metadata before add_one_req() rejects the request.
                    req.mamba_cow_src_index = None
                    req.mamba_needs_clear = False
                    if req.mamba_pool_idx is not None and not getattr(
                        req, "session", None
                    ):
                        self.tree_cache.req_to_token_pool.mamba_allocator.free(
                            req.mamba_pool_idx.unsqueeze(-1)
                        )
                        req.mamba_pool_idx = None
                break

        if mamba_allocator is not None:
            mamba_allocator.alloc_group_end()

        # Update waiting queue
        # 中译：can_run_list 是本轮被选中跑 prefill 的请求；为空说明攒批失败，返回 None。
        can_run_list: List[Req] = adder.can_run_list
        if len(can_run_list) == 0:
            return None, running_batch

        # 中译：把已选中的请求从等待队列移除（被抢占的 preempt_list 再放回队列）。
        can_run_set = set(can_run_list)
        self.waiting_queue = [x for x in self.waiting_queue if x not in can_run_set]
        if adder.preempt_list:
            for req in adder.preempt_list:
                self._add_request_to_queue(req)

        # 中译：本轮攒批过程中产生了新的分块请求（某个 long prompt 被拆成多 chunk），
        #       记录为当前 chunked_req。断言保证同时只存在一个未完成的分块请求。
        if adder.new_chunked_req is not None:
            # Update chunked prefill
            assert self.chunked_req is None
            self.chunked_req = adder.new_chunked_req

        # 中译：累加“在途中间 chunk”计数（用于 PP 等场景跟踪分块请求的进度）。
        if self.chunked_req is not None:
            self.chunked_req.inflight_middle_chunks += 1

        set_time_batch(can_run_list, "set_forward_entry_time")

        # Create a new batch
        # 中译：用选中的请求组装新的 prefill 批，并 prepare_for_extend 准备前向所需张量。
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

        # 中译：标记本批是否包含“最后一个 prefill chunk”。若无分块请求，或本批不是单独的分块
        #       请求（请求数≠ 1），则视为含最后一块；该标记决定 prefill 后是否生成首 token。
        new_batch.contains_last_prefill_chunk = (
            self.chunked_req is None or len(can_run_list) != 1
        )

        # 中译：记录历史最大 prefill 批大小（用于后续名额估算与调优）。
        if self.enable_hierarchical_cache:
            # todo (zhiqiang): disable cuda graph execution if hicache loading triggered
            # 中译：分层缓存下，记录本批需从主机缓存加载 KV 的消费者索引（供后续加载使用）。
            new_batch.hicache_consumer_index = (
                self.tree_cache.ready_to_load_host_cache()
            )

        # 中译：prepare_for_extend 为 prefill 前向准备所需张量（input_ids、位置、KV 槽位分配等）。
        new_batch.prepare_for_extend()

        if self.tp_worker.model_runner.prefill_aware_swa:
            for req in can_run_list:
                req.kv.swa_evict_floor = req.extend_range.end

        # Record prefill stats for logging after forward.
        # 中译：记录本批 prefill 的统计信息（请求数、token 数、抢占与待处理 token 等），
        #       供前向完成后打日志用。
        new_batch.prefill_stats = PrefillStats.from_adder(
            adder,
            running_batch.reqs,
            self.enable_priority_scheduling,
            num_pending_tokens=self.load_inquirer._get_num_pending_tokens(
                chunk_deduct=(
                    self.chunked_req.extend_range.length
                    if self.chunked_req is not None
                    else 0
                ),
            ),
        )

        # Mixed-style chunked prefill
        # 中译：混合分块——在同一次前向里把新 prefill 与正在 decode 的请求拼在一起跑，
        #       减少单独 decode 步的开销。受 logprob / input_embeds 等条件限制。
        if (
            self.is_mixed_chunk
            and not running_batch.is_empty()
            and not (new_batch.return_logprob or running_batch.return_logprob)
            # mix_with_running cats input_ids but not input_embeds — shapes would mismatch
            # 中译：mix_with_running 会拼接 input_ids 但不拼接 input_embeds，若有 input_embeds 则形状会不匹配。
            and new_batch.input_embeds is None
            # Beam member rows are not supported inside a mixed extend batch.
            and all(r.beam_group is None for r in running_batch.reqs)
        ):
            # TODO (lianmin): support return_logprob + mixed chunked prefill
            # 中译：先过滤掉 running_batch 中已完成的请求，准备好 decode 张量后，再把它们拼进本
            # 中译：running 请求已被并入 new_batch，清空 running_batch（仅保留 batch_is_full 标记）。
            running_batch.filter_batch()
            if not running_batch.is_empty():
                running_batch.prepare_for_decode()
                new_batch.mix_with_running(running_batch)
                new_batch.decoding_reqs = running_batch.reqs
            running_batch = ScheduleBatch(
                reqs=[], batch_is_full=running_batch.batch_is_full
            )
        else:
            # 中译：未启用混合分块，本批不携带 decode 请求。
            new_batch.decoding_reqs = None

        # 中译：返回组装好的新 prefill 批（可能已混入部分 decode 请求）。
        return new_batch, running_batch

    def _can_schedule_lora_req(
        self, req: Req, running_loras: set[Optional[str]]
    ) -> bool:
        """
        Check if a LoRA request can be scheduled.

        This method checks two conditions:
        1. The drainer allows scheduling (based on draining state)
        2. The LoRA adapter can be loaded (either already running or can be added)
        """
        if self.lora_drainer and not self.lora_drainer.can_schedule(req):
            return False

        if req.lora_id in running_loras:
            return True

        if self.enable_lora_overlap_loading:
            # For overlapping loading of LoRA weights with computation, we will load each
            # adapter one at a time, as opposed to loading them in one batch
            return self.lora_overlap_loader.try_overlap_load_lora(
                req.lora_id, running_loras
            )
        else:
            new_lora_set = {req.lora_id} | running_loras
            return self.tp_worker.model_runner.lora_manager.validate_lora_batch(
                new_lora_set
            )

    def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch]:
        """Update the current running decoding batch.

        中译：为 decode 步准备 running_batch。关键步骤：
              1. filter_batch：移除已完成的请求；
              2. check_decode_mem：检查 KV 缓存是否还够再 decode 一步，若不够则触发
                 retract_decode（回退部分请求到等待队列，腾出显存），并调整 new_token_ratio；
              3. prepare_for_decode：构建 decode 前向所需的张量（位置、序列长度等）。
        """
        initial_bs = batch.batch_size()

        # 中译：先剔除已完成的请求。
        batch.filter_batch()
        if batch.is_empty():
            batch.batch_is_full = False
            return batch

        # Check if decode out of memory
        # 中译：检查显存是否够下一步 decode。check_decode_mem() 返回 False（显存不足）
        #       或命中测试回退条件时，进入回退分支：把部分请求踢回等待队列以释放 KV。
        if (kv_full_retract_flag := not batch.check_decode_mem()) or (
            TEST_RETRACT and self.forward_ct % TEST_RETRACT_INTERVAL == 0
        ):
            old_available_tokens = self.token_to_kv_pool_allocator.available_size()
            old_ratio = self.new_token_ratio_tracker.current
            mamba_allocator = getattr(
                self.tree_cache.req_to_token_pool, "mamba_allocator", None
            )
            old_mamba_available = (
                mamba_allocator.available_size()
                if mamba_allocator is not None
                else None
            )
            # 中译：retract_decode 选出一批请求回退（其 KV 释放、状态回退到等待重调度），
            retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode()
            new_available_tokens = self.token_to_kv_pool_allocator.available_size()
            new_token_gained = new_available_tokens - old_available_tokens
            mamba_num_gained = (
                mamba_allocator.available_size() - old_mamba_available
                if mamba_allocator is not None
                else None
            )

            self.metrics_reporter.num_retracted_reqs = len(retracted_reqs)
            if self.metrics_reporter.enable_metrics and len(retracted_reqs) > 0:
                self.metrics_reporter.metrics_collector.increment_retracted_reqs(
                    num_retracted_reqs=len(retracted_reqs),
                    num_retracted_input_tokens=sum(
                        len(r.origin_input_ids) for r in retracted_reqs
                    ),
                    num_retracted_output_tokens=sum(
                        len(r.output_ids) for r in retracted_reqs
                    ),
                )
            self.new_token_ratio_tracker.current = new_token_ratio
            for req in reqs_to_abort:
                abort_reason: FINISH_ABORT = req.to_finish
                self.ipc_channels.send_to_tokenizer.send_output(
                    _make_abort_req(req, finished_reason=abort_reason.to_json()),
                    req,
                )
            for req in reqs_to_abort:
                # Member rows were freed inside retract_decode; the group only
                # has to leave the live set.
                self.beam_coordinator.retire_group(req)

            msg_prefix = (
                "KV cache pool is full. Retract requests. "
                if kv_full_retract_flag
                else "Testing retraction. "
            )
            msg_details = f"#retracted_reqs: {len(retracted_reqs)}, #new_tokens_gained: {new_token_gained}"
            if mamba_num_gained is not None:
                msg_details += f", #mamba_num_gained: {mamba_num_gained}"
            if kv_full_retract_flag:
                msg_details += (
                    f", #new_token_ratio: {old_ratio:.4f} -> {new_token_ratio:.4f}"
                )
            logger.warning(msg_prefix + msg_details)

            # 中译：被回退的请求重新入队（标记 is_retracted），稍后再次参与调度。
            for req in retracted_reqs:
                self._add_request_to_queue(req, is_retracted=True)
        else:
            # 中译：未发生回退则让 new_token_ratio 逐步衰减（趋于乐观，敢接纳更多请求）。
            self.new_token_ratio_tracker.decay_step()

        if batch.batch_size() < initial_bs:
            batch.batch_is_full = False

        if batch.is_empty():
            return batch

        # Update batch tensors
        # 中译：构建本步 decode 前向所需的张量。
        batch.prepare_for_decode()
        return batch

    def record_batch_in_overlap(self, batch: ScheduleBatch):
        # 中译：重叠模式下，把本批的所有字段（含 GPU 张量）快照进长度为 2 的环形缓冲，
        #       钉住引用 2 个迭代周期，防止它们在前向 stream 还在使用时被 torch 缓存分配器
        #       回收（跨 stream 张量生命周期问题）。
        # FIXME(lsyin): hacky way to keep a reference to avoid GPU tensors being freed by torch GC
        # NOTE: More Reliable: record all tensors into the forward stream
        # NOTE: - for all future tensors, we shall always read from future map
        #       - for all non-future tensors (produced only by schedule stream),
        #       we shall keep its reference not being release during all the forwarding pass
        # Snapshot all fields: spec V2 rebinds seq_lens / spec_info mid-forward.
        attr_snapshot = [
            getattr(batch, f.name, None) for f in dataclasses.fields(batch)
        ]
        self.batch_record_ct = (self.batch_record_ct + 1) % 2
        # List (not tuple) so that workers can register additional refs via
        # GenerationBatchResult.extra_keep_alive_refs after forward returns.
        self.batch_record_buf[self.batch_record_ct] = [batch, attr_snapshot]

    @contextmanager
    def _forward_isolation(self, batch: ScheduleBatch, *, overlap: bool):
        """Make SB transactional across one forward (overlap and non-overlap).

        中译：把一次前向期间对 ScheduleBatch（SB）的修改变成"事务性"的——前向可能在内部
              改写 batch 的字段（投机解码 V2 尤其会改 forward_mode/input_ids/seq_lens/
              spec_info），本上下文管理器在进入时快照、退出时还原，避免污染下一轮调度；
              同时把 sampling_info 换成"仅供本次前向使用的副本"，防止重复累加惩罚项；
              重叠模式下还会把快照钉进 batch_record_buf 维持张量跨 stream 生命周期。

        1. Snapshot SB fields so V2's mid-forward mutations (forward_mode /
           input_ids / seq_lens / spec_info / ...) can be undone. V1 / non-spec
           only need sampling_info restored - V1 carries spec_info forward as
           next-iter draft input.
        2. Substitute sampling_info with a forward-only copy (orchestrator=None,
           shares the pre-accumulated penalty buffer) so V2's multiple init_new
           calls don't double-accumulate penalties.
        3. (overlap=True only) Pin (batch, snapshot) into batch_record_buf
           for 2 iters so GPU tensors in the snapshot survive the caching
           allocator past the forward stream. Must run AFTER the sampling_info
           swap so the forward-only copy gets pinned. The non-overlap (sync) path
           runs on a single stream and doesn't allocate batch_record_buf, so it
           passes overlap=False.
        """
        # 1. snapshot
        snapshot_v2_full = not batch.spec_algorithm.is_none()
        sched_snapshot = (
            {f.name: getattr(batch, f.name) for f in dataclasses.fields(batch)}
            if snapshot_v2_full
            else None
        )
        sched_sampling_info = batch.sampling_info

        # 2. sampling_info substitute
        if sched_sampling_info is not None:
            batch.sampling_info = sched_sampling_info.copy_for_forward()

        # 3. pin for 2-iter tensor lifetime (overlap path only)
        if overlap:
            self.record_batch_in_overlap(batch)

        try:
            yield
        finally:
            if snapshot_v2_full:
                for name, value in sched_snapshot.items():
                    setattr(batch, name, value)
            else:
                batch.sampling_info = sched_sampling_info

    @scheduler_nvtx_method("scheduler.run_batch")
    def run_batch(
        self,
        batch: ScheduleBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[GenerationBatchResult, EmbeddingBatchResult]:
        """Run a batch.

        中译：在 model worker 上执行一个批次的前向。本方法是 prefill/decode 共用的前向入口，
              内部按以下维度分多条路径：是否生成模型、是否重叠、是否投机解码、是否 PD 复用。
              重叠路径（enable_overlap）的要点：在 forward_stream 上跑前向，前向产出的
              next-token 等"未来值"通过 future_map.publish/stash 在迭代间中继，因此前向后
              立即把 batch.input_ids 置 None（下一轮从 future_map 重建）；并用 copy_stream
              异步把结果拷回 CPU，做到不阻塞调度。
        """
        # 中译：递增全局前向计数，并记录到本批（供看门狗、profiler、回退测试等使用）。
        self.forward_ct += 1
        batch.forward_iter = self.forward_ct
        batch.launch_ts = time.monotonic()
        batch.after_idle_gap = self._sched_idled
        self._sched_idled = False

        # 中译：脚本化调度器钩子（测试/复现场景用），在每次 run_batch 前回调。
        if self.scripted_scheduler_hook is not None:
            self.scripted_scheduler_hook.on_run_batch(batch)

        # Whether to run the profiler
        # 中译：判定本批是否需要启动/停止 profiler 采集。
        self.profiler_manager._profile_batch_predicate(batch)
        # 中译：调试钩子——按需在前向前人为 sleep，用于复现调度时序问题。
        if self.forward_sleep_time is not None:
            logger.info(f"Scheduler.run_batch sleep {self.forward_sleep_time}s")
            time.sleep(self.forward_sleep_time)

        # Place holder handling for pd-disagg decode event loop
        # 中译：PD 分离 decode 事件循环中的"占位（prebuilt）"批处理：这类批的 KV 已由
        #       prefill 节点传来，无需再跑正常前向，走专用的 _run_batch_prebuilt 路径。
        if batch.forward_mode.is_prebuilt():
            return self._run_batch_prebuilt(batch)

        # PD prefill: early-send cached prefix KV, overlapping the suffix forward.
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            for req in batch.reqs:
                self.maybe_send_cached_prefix_chunk(req)

        # Run forward
        # 中译：按“是否生成模型”分两大类路径：生成模型需采样出 next token，
        #       嵌入/奖励模型只需前向得到向量。
        if self.is_generation:
            if self.enable_overlap:
                # 中译：重叠路径（生成模型）。
                # Self-gates on batch.spec_info.future_indices; non-spec_v2
                # no-ops (ForwardBatch.init_new lazily computes the sum).
                # 中译：预先解析本批的 seq_lens_cpu。仅投机 V2 需要（依赖 future_indices），
                #       非投机 V2 为空操作（ForwardBatch.init_new 会懒计算该和）。
                self.future_map.resolve_seq_lens_cpu(batch)
                if self._confidence_budget_prepare is not None:
                    self._confidence_budget_prepare(batch, self.future_map)

                with self.forward_stream_ctx:
                    # 中译：让前向 stream 等待调度 stream——保证本批前向所依赖的调度准备已完成。
                    self.forward_stream.wait_stream(self.schedule_stream)
                    # resolve consumes SB staging (prefill_input_ids_cpu /
                    # mix_running_indices). Run OUTSIDE isolation so the
                    # snapshot captures the post-consume state — restoring
                    # post-forward must not un-consume staging.
                    # 中译：resolve_forward_inputs 会“消费” ScheduleBatch 的暂存字段
                    #       （prefill_input_ids_cpu / mix_running_indices），从 future_map 重建
                    #       真正的 input_ids。故意放在事务隔离（_forward_isolation）之外，
                    #       让快照捕获“消费后”状态；否则前向后还原会错误地“反消费”暂存。
                    resolve_forward_inputs(batch, self.future_map)

                    with self._forward_isolation(batch, overlap=True):
                        # 中译：future_indices 是本批请求在 req_to_token_pool 中的槽位索引，
                        #       同时也用作 future_map 中“未来值”的发布/暂存键。
                        future_indices = batch.req_pool_indices

                        # Spec_v2 fires on_publish mid-worker (between verify and
                        # draft_extend) so schedule prep can overlap with draft_extend.
                        # Non-spec has no later work — scheduler publishes after return.
                        # 中译：投机 V2 会在 worker 内部（verify 与 draft_extend 之间）触发
                        fwd_kwargs = {}
                        if not batch.spec_algorithm.is_none():
                            fwd_kwargs["on_publish"] = partial(
                                self.future_map.publish, future_indices
                            )
                            # Grammar-overlap-capable workers advance the grammar FSM
                            # inside verify() before building the bitmask; hand them the
                            # barrier that resolves the previous batch's committed
                            # tokens (overlapping the target forward).
                            if batch.spec_algorithm.supports_grammar_overlap():
                                fwd_kwargs["grammar_barrier"] = (
                                    self._advance_pending_grammar
                                )

                        # FIXME: pp is not compatible with overlap
                        # 中译：调用 worker 执行前向生成（采样出下一 token）。
                        batch_result = self.model_worker.forward_batch_generation(
                            batch, **fwd_kwargs
                        )
                        # 中译：非投机解码：前向返回后，把"下一步序列长度（+1）"发布到 future_map，
                        #       供下一轮调度读取（重叠的关键中继）。
                        if batch.spec_algorithm.is_none():
                            self.future_map.publish(future_indices, batch.seq_lens + 1)
                        # Park any refs the worker wants kept alive 2 iters
                        # (cross-stream tensor lifetime; pinned in the same
                        # ring slot as the SB attr snapshot).
                        # 中译：worker 可能要求某些张量多存活 2 个迭代（跨 stream 生命周期问题），
                        #       把它们钉进与 SB 快照同一个环形缓冲槽位，避免被缓存分配器提前回收。
                        if batch_result.extra_keep_alive_refs:
                            self.batch_record_buf[self.batch_record_ct].extend(
                                batch_result.extra_keep_alive_refs
                            )
                        if self.enable_unified_memory:
                            # Record a `forward_done` event after the forward (before
                            # copy_to_cpu); lazy-compaction `_flush` gates src reuse on
                            # it. Only the unified pool's allocator exposes these hooks.
                            allocator = self.token_to_kv_pool_allocator
                            forward_done = self.device_module.Event()
                            forward_done.record(stream=self.forward_stream)
                            allocator.set_latest_forward_done_event(forward_done)
                            # Write-set classification: hand the allocator this
                            # forward's virtual out_cache_loc as a tensor ref (no GPU work).
                            allocator.set_inflight_forward(
                                forward_done,
                                batch.out_cache_loc,
                            )
                        # FIXME(lsyin): maybe move this to forward_batch_generation
                        # 中译：创建一个 CUDA Event 标记 D2H 拷贝完成点，供后续结果处理同步等待。
                        batch_result.copy_done = self.device_module.Event()
                        # 中译：delay_sample_func 为 None 表示本批采样已在 worker 内完成，可立即
                        #       把“未来值”（投机为 draft 输入，非投机为 next_token_ids）暂存到
                        #       future_map，并启动异步 D2H 拷贝；否则采样被推迟（如结构化输出
                        #       需等上一批语法状态），仅记录 future_indices，到采样时再处理。
                        if batch_result.delay_sample_func is None:
                            self._relay_forward_payload(
                                batch, future_indices, batch_result
                            )
                            if _is_hip:
                                # Cross-stream sync costs more than the tiny D2H it
                                # overlaps.
                                batch_result.copy_to_cpu(
                                    return_logprob=batch.return_logprob,
                                    return_hidden_states=batch.return_hidden_states,
                                )
                            else:
                                # Result D2H on copy_stream overlaps the next forward
                                # instead of serializing on forward_stream; it's a leaf
                                # gated by copy_done, so nothing on forward_stream waits.
                                self.copy_stream.wait_stream(self.forward_stream)
                                with self.copy_stream_ctx:
                                    batch_result.copy_to_cpu(
                                        return_logprob=batch.return_logprob,
                                        return_hidden_states=batch.return_hidden_states,
                                    )
                        else:
                            batch_result.future_indices = future_indices

                # Next-iter input_ids relayed via future_map.
                # 中译：下一轮的 input_ids 改由 future_map 中继，这里清空避免读到本轮旧值。
                batch.input_ids = None

                # 中译：投机解码下，把本次产出的 draft 输入作为下一轮 spec_info 带入，并记下其
                #       future_indices（下一轮从 future_map 取回真实值）。
                if not batch.spec_algorithm.is_none():
                    batch.spec_info = batch_result.next_draft_input
                    batch.spec_info.future_dsa_topk_indices_available = (
                        batch.spec_info.dsa_topk_indices is not None
                    )
                    batch.spec_info.future_indices = future_indices
            elif self.enable_pdmux and batch.forward_mode.is_split_prefill():
                # 中译：PD 复用（pdmux）的“拆分 prefill”路径：先解析输入、跑拆分 prefill，
                #       若产出了 next token 则照样用 future_map 中继，并清空 input_ids。
                resolve_forward_inputs(batch, self.future_map)
                batch_result = self.tp_worker.forward_batch_split_prefill(batch)
                self._relay_forward_payload(batch, batch.req_pool_indices, batch_result)
                batch.input_ids = None
                self._copy_auxiliary_output_to_cpu(batch, batch_result)
            elif not batch.spec_algorithm.is_none():
                # Non-overlap: drive the V2 worker synchronously (no
                # future_map relay / on_publish).
                # 中译：非重叠的投机路径：同步驱动 V2 worker（不经 future_map 中继、无 on_publish）。
                resolve_forward_inputs(batch, self.future_map)
                with self._forward_isolation(batch, overlap=False):
                    batch_result = self.model_worker.forward_batch_generation(
                        batch, pp_proxy_tensors=pp_proxy_tensors
                    )
                # The isolation restore reverted the worker's in-forward SB edits;
                # re-apply what must carry to the next iter.
                # 中译：事务隔离退出时已回滚 worker 在前向中对 SB 的修改，这里手动重新应用
                #       那些必须带到下一轮的字段（spec_info、新的 seq_lens 等）。
                batch.spec_info = batch_result.next_draft_input
                if batch_result.new_seq_lens is not None:
                    batch.seq_lens = batch_result.new_seq_lens
                    if batch.seq_lens_cpu is not None:
                        batch.seq_lens_cpu = batch_result.new_seq_lens.to("cpu")
                        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
                batch.input_ids = None  # rebuilt next iter from draft_token
                self.update_cache_from_scheduler(batch, batch_result)
                # 中译：同步一次 D2H 拷贝，使结果处理器能读到 CPU 上的张量。
                # Only the last PP rank owns real results requiring D2H; other ranks
                # consume device tensors rebuilt from the output ring.
                batch_result.copy_done = self.device_module.Event()
                if batch_result.has_sampled_token_ids and self.ps.pp_size == 1:
                    batch_result.copy_to_cpu(
                        return_logprob=batch.return_logprob,
                        return_hidden_states=batch.return_hidden_states,
                    )
            else:
                # 中译：非重叠、非投机的普通生成路径（也兼容 PP 的 proxy 张量传递）。
                kwargs = (
                    {"pp_proxy_tensors": pp_proxy_tensors}
                    if self.spec_algorithm.is_none()
                    else {}
                )
                resolve_forward_inputs(batch, self.future_map)
                batch_result = self.model_worker.forward_batch_generation(
                    batch, **kwargs
                )
                if batch_result.has_sampled_token_ids:
                    # Non-spec: relay via future_map, gathered next iter.
                    # 中译：非投机：把 next_token_ids 暂存到 future_map 中继，下一轮再汇集取回。
                    self._relay_forward_payload(
                        batch, batch.req_pool_indices, batch_result
                    )
                    batch.input_ids = None
                self.update_cache_from_scheduler(batch, batch_result)
                self._copy_auxiliary_output_to_cpu(batch, batch_result)

            # These 2 values are needed for processing the output, but the values can be
            # modified by overlap schedule. So we have to copy them here so that
            # we can use the correct values in output processing.
            # 中译：以下两个值（每请求的 extend 输入长度、logprob 起点）处理输出时需要，
            if batch.return_logprob or batch.return_hidden_states:
                batch_result.extend_input_len_per_req = [
                    req.extend_range.length if req.extend_range is not None else 0
                    for req in batch.reqs
                ]
            else:
                batch_result.extend_input_len_per_req = None

            if batch.return_logprob:
                batch_result.extend_logprob_start_len_per_req = (
                    batch.extend_logprob_start_lens
                )
            else:
                batch_result.extend_logprob_start_len_per_req = None

            ret = batch_result
        else:  # embedding or reward model
            # 中译：非生成模型（embedding / reward）——只做前向得到向量，无需采样。
            if self.enable_overlap:
                # 中译：重叠模式下同样先钉住本批张量生命周期，再在 forward_stream 上跑前向。
                self.record_batch_in_overlap(batch)
                with self.forward_stream_ctx:
                    self.forward_stream.wait_stream(self.schedule_stream)
                    resolve_forward_inputs(batch, self.future_map)
                    pooler_output, can_run_cuda_graph = (
                        self.tp_worker.forward_batch_embedding(batch)
                    )
                    ret = EmbeddingBatchResult(
                        embeddings=pooler_output.embeddings,
                        pooled_hidden_states=pooler_output.pooled_hidden_states,
                        can_run_cuda_graph=can_run_cuda_graph,
                    )
                    ret.copy_to_cpu()
            else:
                resolve_forward_inputs(batch, self.future_map)
                pooler_output, can_run_cuda_graph = (
                    self.tp_worker.forward_batch_embedding(batch)
                )
                ret = EmbeddingBatchResult(
                    embeddings=pooler_output.embeddings,
                    pooled_hidden_states=pooler_output.pooled_hidden_states,
                    can_run_cuda_graph=can_run_cuda_graph,
                )

        self._maybe_report_active_ranks()

        return ret

    def _maybe_report_active_ranks(self) -> None:
        if not (
            self.enable_dp_attention and get_exec().moe.elastic_ep_backend is not None
        ):
            return
        from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager

        inst = ElasticEPStateManager.instance()
        if inst is not None and inst.active_ranks_cpu is not None:
            self.ipc_channels.send_to_tokenizer.send_output(
                ActiveRanksOutput(
                    status=[bool(x) for x in inst.active_ranks_cpu.tolist()]
                )
            )
        else:
            logger.debug("[Elastic EP] active rank state is unavailable")
            return

        model_runner = self.tp_worker.model_runner
        pending = model_runner._pending_elastic_scale_update
        if pending is not None:
            self.ipc_channels.send_to_tokenizer.send_output(pending)
            model_runner._pending_elastic_scale_update = None

    def _relay_forward_payload(
        self,
        batch: ScheduleBatch,
        future_indices: torch.Tensor,
        batch_result: GenerationBatchResult,
    ) -> None:
        """Stash this iter's relay payload for next iter's resolve_forward_inputs."""
        if self.spec_algorithm.is_ngram():
            if batch_result.next_draft_input is not None:
                payload = RelayPayload.from_ngram(batch_result.next_draft_input)
                self.future_map.stash(future_indices, payload)
            return
        if batch_result.next_draft_input is not None:
            payload = RelayPayload.from_draft_input(batch_result.next_draft_input)
        elif batch_result.has_sampled_token_ids:
            payload = RelayPayload(bonus_tokens=batch_result.next_token_ids)
        else:
            return
        if batch.beam_tail is not None:
            # The worker sliced the tail off before sampling, so sampled tokens
            # cover only the reqs-aligned rows; the coordinator relays the rest.
            future_indices = future_indices[: batch.beam_tail.num_base_rows]
        self.future_map.stash(future_indices, payload)
        self.beam_coordinator.maybe_select_and_relay(
            batch, batch_result, chunked_req=self.chunked_req
        )

    def _copy_auxiliary_output_to_cpu(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        logits_output = result.logits_output
        if (
            logits_output is None
            or logits_output.auxiliary_device_output is None
            or result.auxiliary_host_output is not None
        ):
            return
        # PP transports the device output to the first rank before copying it.
        if self.ps.pp_size > 1:
            return
        if result.copy_done is not None:
            raise RuntimeError(
                "generation result has an uncopied auxiliary output after its "
                "device-to-host copy was scheduled"
            )
        result.copy_done = self.device_module.Event()
        result.copy_to_cpu(
            return_logprob=batch.return_logprob,
            return_hidden_states=batch.return_hidden_states,
        )

    def launch_batch_sample_if_needed(
        self, batch_result: GenerationBatchResult, cur_batch: ScheduleBatch
    ) -> Union[GenerationBatchResult]:
        # TODO(lsyin): make the delayed sample a default behavior after
        # unifying the forward_batch_generation interface (related to spec V2).
        if batch_result is None or batch_result.delay_sample_func is None:
            return

        with self.forward_stream_ctx:
            self.forward_stream.wait_stream(self.schedule_stream)
            _batch_result = batch_result.delay_sample_func()
            assert _batch_result is batch_result
            # Delay-sample is non-spec only; relays the sampled bonus tokens.
            self._relay_forward_payload(
                cur_batch, batch_result.future_indices, batch_result
            )

        # Run device-to-host copy on a separate stream to avoid blocking the
        # forward stream. The copy waits for the sampled result and can overlap
        # with subsequent forward computation.
        self.copy_stream.wait_stream(self.forward_stream)
        with self.copy_stream_ctx:
            batch_result.copy_to_cpu(
                return_logprob=cur_batch.return_logprob,
                return_hidden_states=cur_batch.return_hidden_states,
            )

        # Release the closure and large GPU tensors that are no longer needed.
        # The delay_sample_func closure captures forward_batch (which holds
        # sampling_info with vocab_mask) and logits_output (which holds
        # next_token_logits). Without clearing these, they stay alive via
        # batch_result in result_queue and batch_record_buf until the next
        # iteration, causing a steady VRAM leak with structured output.
        batch_result.delay_sample_func = None
        if batch_result.logits_output is not None:
            batch_result.logits_output.next_token_logits = None

    @scheduler_nvtx_method("scheduler.process_batch_result")
    def process_batch_result(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        # 中译：处理一个批次前向后的结果。按 forward_mode（decode / extend(prefill) / prebuilt /
        # Flush async trace ops here: in overlap mode this CPU work runs while
        # the next batch's GPU forward is in flight, giving free overlap.
        flush_trace_batch(batch.reqs)
        snapshot = self.publish_load_snapshot(force=batch.forward_mode.is_extend())
        # Router-facing gauge on the dedicated PUB socket, reusing the
        # snapshot above rather than walking the queues again.
        self.load_publisher.publish_load_stat(
            self.load_inquirer.get_loads,
            force=batch.forward_mode.is_extend(),
            snapshot=snapshot,
        )

        if batch.forward_mode.is_decode():
            self.batch_result_processor.process_batch_result_decode(batch, result)
        elif batch.forward_mode.is_extend():
            if batch.is_dllm():
                self.process_batch_result_dllm(batch, result)
            elif self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.process_batch_result_disagg_prefill(batch, result)
            else:
                self.batch_result_processor.process_batch_result_prefill(batch, result)
        elif batch.forward_mode.is_prebuilt():
            self.batch_result_processor.process_batch_result_prebuilt(batch)
        elif batch.forward_mode.is_idle():
            self.batch_result_processor.process_batch_result_idle(batch, result)

        self._record_step_counters(batch, result)

        self.metrics_reporter.log_batch_result_stats(batch, result)

        # Emit forward pass metrics (every iteration when enabled)
        if self.enable_fpm:
            self.metrics_reporter._emit_forward_pass_metrics(batch, result)

        self._maybe_clear_mm_inputs(batch)
        self.maybe_send_health_check_signal()
        self.metrics_reporter.update_device_timer()

    def _record_step_counters(
        self, batch: ScheduleBatch, result: GenerationBatchResult
    ) -> None:
        mode = batch.forward_mode
        is_prefill = mode.is_extend_without_speculative()
        if not (is_prefill or mode.is_decode() or mode.is_target_verify()):
            return
        if all(is_health_check_generate_req(req) for req in batch.reqs):
            return
        prev = self._prev_step
        self._prev_step = (batch.forward_iter, batch.launch_ts, is_prefill)
        # An idle pass keeps forward_iter contiguous (forward_ct advances in run_batch).
        if prev is None or batch.after_idle_gap:
            return
        prev_iter, prev_ts, prev_is_prefill = prev
        if prev_iter + 1 != batch.forward_iter or prev_is_prefill != is_prefill:
            return
        step_us = int((batch.launch_ts - prev_ts) * 1e6)
        if not 0 < step_us < STEP_MAX_US:
            return
        if is_prefill:
            self.total_prefill_busy_us += step_us
            self.total_prefill_uncached_tokens += batch.extend_num_tokens
        else:
            batch_size = len(batch.reqs)
            _accumulate_decode_moment(
                self.decode_moment_totals,
                batch_size,
                step_us,
                batch_size + result.num_correct_drafts,
            )

    def maybe_send_health_check_signal(self):
        if self.return_health_check_ipcs:
            # Return some signal for the health check.
            # This is used to prevent the health check signal being blocked by long context prefill.
            # However, one minor issue is that this code path does not check the status of detokenizer manager.
            self.ipc_channels.send_to_tokenizer.send_output(
                HealthCheckOutput(
                    http_worker_ipc=self.return_health_check_ipcs.popleft()
                )
            )

    def add_external_corpus(
        self, recv_req: AddExternalCorpusReqInput
    ) -> Optional[AddExternalCorpusReqOutput]:
        if self.external_corpus_manager is None:
            return AddExternalCorpusReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        return self.external_corpus_manager.add(recv_req)

    def remove_external_corpus(
        self, recv_req: RemoveExternalCorpusReqInput
    ) -> RemoveExternalCorpusReqOutput:
        if self.external_corpus_manager is None:
            return RemoveExternalCorpusReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        return self.external_corpus_manager.remove(recv_req)

    def list_external_corpora(
        self, recv_req: ListExternalCorporaReqInput
    ) -> ListExternalCorporaReqOutput:
        if self.external_corpus_manager is None:
            return ListExternalCorporaReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        return self.external_corpus_manager.list(recv_req)

    def clear_hicache_storage_wrapped(self, recv_req: ClearHiCacheReqInput):
        if self.enable_hierarchical_cache:
            self.tree_cache.clear_storage_backend()
            logger.info("Hierarchical cache cleared successfully!")
            if_success = True
        else:
            logging.warning("Hierarchical cache is not enabled.")
            if_success = False
        return ClearHiCacheReqOutput(success=if_success)

    def on_idle(self):
        # 中译：仅当确实"完全空闲"（无任何在跑/待跑/在途请求）时才执行，否则直接返回。
        """Idle housekeeping: guard, check, metrics, reset, sleep."""
        # Flush any health-check signal deferred while the engine was busy.
        self.maybe_send_health_check_signal()

        # Publish before the fully-idle gate: a no-batch-but-not-idle stall
        # (queues parked under KV pressure / disagg transfer) has no
        # process_batch_result to publish the growing gauge, and gating here
        # froze /get_loads, DP balancing, and the LoadStat for the stall. This
        # path spins without sleeping, so a wall-clock floor bounds the
        # O(queue) get_loads for both sinks; the fully-idle publish runs
        # post-flush below.
        if not self.is_fully_idle():
            now = time.monotonic()
            if now - self._last_stall_publish_ts >= LOAD_STALL_REFRESH_S:
                self._last_stall_publish_ts = now
                snapshot = self.publish_load_snapshot(force=True)
                self.load_publisher.publish_load_stat(
                    self.load_inquirer.get_loads, force=True, snapshot=snapshot
                )
            return

        if self.enable_unified_memory:
            try:
                self.token_to_kv_pool_allocator.flush_opportunistic()
            except Exception:
                pass

        # memory leak check (skipped for hisparse — pool counters intentionally
        # diverge during host-backup, see _get_swa_token_info clamp).
        # Also skipped while deferred KV releases are pending: they hold pages out
        # of the allocator by design, so the pool is transiently below `total` and
        # would trip the idle leak invariant. Resumes once the holds resolve.
        deferred_pending = (
            self.disaggregation_mode == DisaggregationMode.DECODE
            and self.disagg_decode_transfer_queue.has_pending_deferred_releases()
        )
        if not self.enable_hisparse and not deferred_pending:
            has_leak, messages = self.invariant_checker._check_all_pools(
                self.pool_stats_observer.get_pool_stats(),
            )
            if has_leak:
                self.invariant_checker._report_leak("pool", "\n".join(messages))
            self.invariant_checker._check_req_pool()

        # tree cache sanity check
        self.invariant_checker._check_tree_cache()

        # metrics every 30s
        self.metrics_reporter._maybe_log_idle_metrics()

        # kv event publishing
        self.kv_events_publisher.publish_kv_events()

        # reset token ratio
        self.new_token_ratio_tracker.reset()

        # Fully-idle publish, post-flush so the gauge reflects compacted KV.
        # Forced (immediate) so the busy->idle transition is never delayed.
        snapshot = self.publish_load_snapshot(force=True)
        self.load_publisher.publish_load_stat(
            self.load_inquirer.get_loads, force=True, snapshot=snapshot
        )

        # sleep until next event
        self.maybe_sleep_on_idle()

    def is_fully_idle(self, for_health_check=False) -> bool:
        # 中译：判断调度器是否"完全空闲"——综合检查 running_batch、等待队列、分块请求、
        #       重叠结果队列、PP 微批、PD 分离各队列、HiSparse/HiCache 在途异步操作等全部为空。
        #       for_health_check=True 时只看会真正占用 GPU 的部分（运行批 + 等待队列），
        #       因为只有这些能证明服务在处理请求、可承载健康信息。
        # Health check piggybacks on running requests in process_output.
        # Only running_batch + waiting_queue guarantee active GPU processing;
        # disagg queues (bootstrap/prealloc/transfer) may have items without
        # any request actually running on GPU — e.g. stuck handshake, full
        # KV cache, or stalled transfer — so they can't carry health info.
        # Batch running status
        idle = (
            self.running_batch.is_empty()
            and self.chunked_req is None
            and not self.dllm_manager.any_staging_reqs()
            and (self.last_batch is None or self.last_batch.is_empty())
            and (not self.enable_overlap or len(self.result_queue) == 0)
            and self._pp_microbatches_drained()
        )

        # Waiting queues: waiting + bootstrapping + preallocation + kv transfer (decode)
        idle &= len(self.waiting_queue) == 0

        if (
            for_health_check
            and not self._engine_paused
            and self.disaggregation_mode == DisaggregationMode.DECODE
            and self.disagg_decode_prealloc_queue is not None
        ):
            idle &= len(self.disagg_decode_prealloc_queue.retracted_queue) == 0

        if not for_health_check:
            # Grammar queue and prefill inflight queue may not produce batch
            # results instantly, but they still indicate the server is not idle.
            idle &= len(self.grammar_manager.grammar_queue) == 0
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                idle &= len(self.disagg_prefill_inflight_queue) == 0
                idle &= len(self.disagg_prefill_bootstrap_queue.queue) == 0

            if self.disaggregation_mode == DisaggregationMode.DECODE:
                idle &= len(self.disagg_decode_prealloc_queue.queue) == 0
                idle &= len(self.disagg_decode_prealloc_queue.retracted_queue) == 0
                idle &= len(self.disagg_decode_transfer_queue.queue) == 0
                if self.decode_offload_manager is not None:
                    idle &= len(self.decode_offload_manager.ongoing_offload) == 0

            # HiSparse: staging requests transitioning prefill -> decode
            if self.enable_hisparse:
                idle &= not self.hisparse_coordinator.has_ongoing_staging()

            # HiCache: in-flight async ops (GPU↔Host↔L3) must drain before
            # destructive operations like attach/detach/flush_cache.
            if self.enable_hierarchical_cache:
                tc = self.tree_cache
                idle &= len(tc.ongoing_write_through) == 0
                idle &= len(tc.ongoing_load_back) == 0
                if tc.enable_storage:
                    idle &= len(tc.ongoing_prefetch) == 0
                    idle &= len(tc.ongoing_backup) == 0
                    if get_memory().hicache_host_memory_mode == "buffer_only":
                        # Queued writes, staged prefetches, and in-flight
                        # storage writes still hold host staging
                        # (buffer-mode unified tree only).
                        idle &= tc.buffer_pipeline.is_idle()

        return idle

    def _pp_microbatches_drained(self) -> bool:
        if self.ps.pp_size == 1:
            return True
        return all(x.is_empty() for x in self.running_mbs) and all(
            mb is None or mb.is_empty() for mb in self.mbs
        )

    def attach_hicache_storage_wrapped(
        self, recv_req: AttachHiCacheStorageReqInput
    ) -> AttachHiCacheStorageReqOutput:
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
                served_model_name=get_serving().served_model_name,
                hicache_storage_prefetch_policy=recv_req.hicache_storage_prefetch_policy,
                hicache_write_policy=recv_req.hicache_write_policy,
            )
        except Exception as e:
            logger.exception("Attach HiCache storage backend failed with exception.")
            return AttachHiCacheStorageReqOutput(success=False, message=str(e))
        if ok:
            self.enable_hicache_storage = True
            hicache_fields = {
                "hicache_storage_backend": recv_req.hicache_storage_backend
            }
            if recv_req.hicache_storage_backend_extra_config_json is not None:
                hicache_fields["hicache_storage_backend_extra_config"] = (
                    recv_req.hicache_storage_backend_extra_config_json
                )
            if recv_req.hicache_storage_prefetch_policy is not None:
                hicache_fields["hicache_storage_prefetch_policy"] = (
                    recv_req.hicache_storage_prefetch_policy
                )
            if recv_req.hicache_write_policy is not None:
                hicache_fields["hicache_write_policy"] = recv_req.hicache_write_policy
            get_context().override("scheduler.attach_hicache", **hicache_fields)
            logger.info(
                f"Attached HiCache storage backend: {recv_req.hicache_storage_backend}"
            )
        return AttachHiCacheStorageReqOutput(success=ok, message=msg)

    def detach_hicache_storage_wrapped(
        self, recv_req: DetachHiCacheStorageReqInput
    ) -> DetachHiCacheStorageReqOutput:
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

        # Idempotent detach: even if scheduler thinks storage is disabled, we still
        # attempt best-effort cleanup in tree_cache (it may have leftover state).
        try:
            ok, msg = self.tree_cache.detach_storage_backend()
        except Exception as e:
            logger.exception("Detach HiCache storage backend failed with exception.")
            return DetachHiCacheStorageReqOutput(success=False, message=str(e))

        if ok or (not self.enable_hicache_storage):
            # Treat "already disabled / nothing to do" as success for idempotence.
            self.enable_hicache_storage = False
            get_context().override(
                "scheduler.detach_hicache",
                hicache_storage_backend=None,
                hicache_storage_backend_extra_config=None,
            )
            logger.info("Detached HiCache storage backend.")
            return DetachHiCacheStorageReqOutput(
                success=True, message=msg or "HiCache storage backend is detached."
            )

        return DetachHiCacheStorageReqOutput(success=False, message=msg)

    def flush_cache(self, empty_cache: bool = True):
        """Flush memory pools (e.g., KV cache, Mamba cache) and optionally empty device allocator cache.

        中译：清空所有内存池（KV 缓存、radix tree cache、Mamba 缓存等）并重置指标。
              只有在"完全空闲"时才允许执行（否则会破坏在跑请求的状态），有 pending 请求则拒绝。
        """
        if self.is_fully_idle():
            self.cur_batch_for_debug = None
            self.last_batch = None
            self.tree_cache.reset()
            self.req_to_token_pool.clear()
            self.token_to_kv_pool_allocator.clear()
            self.req_to_token_pool.reset_aux_cache_allocator()
            self.grammar_manager.clear()
            self.metrics_reporter.reset_metrics()

            if self.draft_worker:
                self.draft_worker.clear_cache_pool()

            if empty_cache:
                current_platform.empty_cache()
            # Per-DP-group leader logs once: ranks within a DP group are
            # state-synchronous, but DP groups may diverge.
            if self.metrics_reporter.is_stats_logging_rank:
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
        # Resolved config (pristine server_args + post-publish overrides) so a
        # readback reflects values changed via /set_internal_state, not startup.
        ret = get_context().resolved_server_args_dict()
        ret["world_size"] = compute_world_size(
            enable_dp_attention=get_parallel().enable_dp_attention,
            dp_size=get_parallel().dp_size,
            tp_size=get_parallel().tp_size,
            pp_size=get_parallel().pp_size,
        )
        ret["last_gen_throughput"] = self.metrics_reporter.last_gen_throughput
        draft_graph_memory_usage = (
            None if self.draft_worker is None else self.draft_worker.graph_memory_usage
        )
        ret["memory_usage"] = build_memory_usage(
            weight_gb=self.tp_worker.model_runner.weight_load_mem_usage,
            kv_cache_gb=self.token_to_kv_pool_allocator.get_kvcache().mem_usage,
            startup_available_gb=self.startup_available_gpu_memory_gb,
            token_capacity=self.max_total_num_tokens,
            token_capacity_swa=self.swa_tokens_per_layer,
            target_graph_memory_usage=self.tp_worker.graph_memory_usage,
            draft_graph_memory_usage=draft_graph_memory_usage,
        )
        ret["startup_time"] = self.startup_time
        ret["effective_max_running_requests_per_dp"] = self.max_running_requests

        if get_exec().moe.elastic_ep_backend is not None:
            from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager

            ret["is_scaling_elastic_ep"] = ElasticEPStateManager.is_scaling()
            ret["effective_ep_size"] = ElasticEPStateManager.get_effective_ep_size()
            ret["pending_ep_size"] = ElasticEPStateManager.get_pending_ep_size()
            ret["scale_phase"] = ElasticEPStateManager.get_scale_phase()
            ret["elastic_ep_last_error"] = ElasticEPStateManager.get_last_error()

        if (
            not self.spec_algorithm.is_none()
            and self.metrics_reporter.spec_total_num_forward_ct > 0
        ):
            ret["avg_spec_accept_length"] = (
                self.metrics_reporter.spec_total_num_accept_tokens
                / self.metrics_reporter.spec_total_num_forward_ct
            )

        if RECORD_STEP_TIME:
            ret["step_time_dict"] = self.metrics_reporter.step_time_dict

        if self.spec_algorithm.is_dspark() and self.draft_worker is not None:
            info_record = self.draft_worker.dump_info_records()
            if info_record is not None:
                ret["dspark_info_record"] = info_record

        if envs.SGLANG_EXPOSE_OWN_ENV_VARS.get():
            ret["env_vars"] = exportable_env_vars()

        # A bound signal handler is not msgpack-serializable, and no reader
        # consumes it.
        ret.pop("custom_sigquit_handler", None)

        return GetInternalStateReqOutput(internal_state=msgspec_to_builtins(ret))

    def set_internal_state(self, recv_req: SetInternalStateReq):
        server_args_dict = recv_req.server_args
        args_allow_update = set(
            [
                "pp_max_micro_batch_size",
                "speculative_accept_threshold_single",
                "speculative_accept_threshold_acc",
                "dspark_force_budget_frac",
                "dspark_clear_info_records",
            ]
        )

        if_success = True
        for k, v in server_args_dict.items():
            if k not in args_allow_update:
                logging.warning(f"Updating {k} is not supported.")
                if_success = False
                break
            elif k == "pp_max_micro_batch_size" and (
                v > self.max_running_requests // self.ps.pp_size or v < 1
            ):
                logging.warning(
                    f"Updating {k} to {v} is rejected because it is out of the valid range [1, {self.max_running_requests // self.ps.pp_size}]."
                )
                if_success = False
                break
            elif k == "dspark_force_budget_frac":
                if not self.spec_algorithm.is_dspark() or not hasattr(
                    self.draft_worker, "set_dspark_forced_budget_frac"
                ):
                    logging.warning(
                        "dspark_force_budget_frac requires a DSpark draft worker."
                    )
                    if_success = False
                    break
                if v is not None and not (0.0 < float(v) <= 1.0):
                    logging.warning(
                        f"dspark_force_budget_frac must be in (0, 1] or null, got {v}."
                    )
                    if_success = False
                    break
            elif k == "dspark_clear_info_records":
                if not self.spec_algorithm.is_dspark() or not hasattr(
                    self.draft_worker, "clear_info_records"
                ):
                    logging.warning(
                        "dspark_clear_info_records requires a DSpark draft worker."
                    )
                    if_success = False
                    break

        if if_success:
            if (
                not self.spec_algorithm.is_none()
                and self.metrics_reporter.spec_total_num_forward_ct > 0
            ):
                avg_spec_accept_length = (
                    self.metrics_reporter.spec_total_num_accept_tokens
                    / self.metrics_reporter.spec_total_num_forward_ct
                )
                logger.info(f"{avg_spec_accept_length=}")
            self.metrics_reporter.spec_total_num_accept_tokens = (
                self.metrics_reporter.spec_total_num_forward_ct
            ) = 0
            # DSpark control keys are worker commands, not server args; route
            # them to the draft worker and keep them out of the override.
            remaining = dict(server_args_dict)
            frac = remaining.pop("dspark_force_budget_frac", None)
            if "dspark_force_budget_frac" in server_args_dict:
                self.draft_worker.set_dspark_forced_budget_frac(
                    None if frac is None else float(frac)
                )
            if remaining.pop("dspark_clear_info_records", None):
                self.draft_worker.clear_info_records()
            if remaining:
                get_context().override(source="update_server_args", **remaining)
            logger.info(f"Config updated via context override: {remaining}")

        return SetInternalStateReqOutput(updated=if_success)

    def save_remote_model(self, **kwargs):
        self.weight_updater.save_remote_model(kwargs)

    def save_sharded_model(self, **kwargs):
        self.weight_updater.save_sharded_model(kwargs)

    def handle_rpc_request(self, recv_req: RpcReqInput):
        # Handle RPC requests
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

        barrier(group=self.tp_group.cpu_group)
        return RpcReqOutput(success=success, message="" if not exec else str(exec))

    def handle_update_weight_version(
        self, recv_req: UpdateWeightVersionReqInput
    ) -> UpdateWeightVersionReqOutput:
        self.record_weight_version_change(new_version=recv_req.new_version)
        return UpdateWeightVersionReqOutput()

    def record_weight_version_change(self, new_version: Optional[str]) -> None:
        if new_version is None or new_version == get_serving().weight_version:
            return

        old_version = get_serving().weight_version
        get_context().override("scheduler.weight_version", weight_version=new_version)

        live_reqs = {
            *self.collect_inflight_reqs(),
            *self.waiting_queue,
            *([self.chunked_req] if self.chunked_req is not None else []),
        }
        if self.hisparse_coordinator is not None:
            live_reqs.update(
                act.req for act in self.hisparse_coordinator.ack_staging_queue
            )
        num_recorded = record_weight_version_events(live_reqs, old_version=old_version)
        logger.info(
            f"Weight version changed. {old_version=} {new_version=} {num_recorded=}"
        )

    def collect_inflight_reqs(self) -> Set[Req]:
        if self.ps.pp_size == 1:
            inflight_batches = [self.running_batch, self.last_batch]
        else:
            inflight_batches = [*self.running_mbs, *self.mbs]
        return {
            req for batch in inflight_batches if batch is not None for req in batch.reqs
        }

    def abort_request(self, recv_req: AbortReq):
        if (chunked_req := self.chunked_req) is not None:
            if recv_req.abort_all or chunked_req.rid.startswith(recv_req.rid):
                self._pending_chunked_abort_req = chunked_req

        # todo hisparse, release resources for abort requests in hisparse coordinator
        # Abort requests still waiting for encoder embeddings (EPD language-only)
        if self.mm_receiver is not None:
            self.mm_receiver.abort_waiting_requests(recv_req)

        # Delete requests in the waiting queue
        to_del = []
        for i, req in enumerate(self.waiting_queue):
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                to_del.append(i)

        # Sort in reverse order to avoid index issues when deleting
        for i in reversed(to_del):
            # Abort method 1: directly pop from the queue
            # This only works for requests that have not started anything.
            # We still need to send something back to TokenizerManager to clean up the state.
            req = self.waiting_queue.pop(i)
            self.beam_coordinator.retire_group(req)
            if self.enable_hicache_storage:
                # to release prefetch events associated with the request
                self.tree_cache.release_aborted_request(req.rid)
            self.ipc_channels.send_to_tokenizer.send_output(_make_abort_req(req), req)
            # For disaggregation decode mode, the request in the waiting queue has KV cache allocated.
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                release_kv_cache(req, self.tree_cache)
            # For disaggregation prefill mode, free the metadata buffer index
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                bootstrap_pending = req.pending_bootstrap
                maybe_release_metadata_buffer(
                    req, self.req_to_metadata_buffer_idx_allocator
                )
                if (
                    bootstrap_pending
                    and hasattr(req, "disagg_kv_sender")
                    and req.disagg_kv_sender is not None
                ):
                    if hasattr(req.disagg_kv_sender, "abort"):
                        req.disagg_kv_sender.abort()

            # For mamba radix cache
            if (
                req.mamba_pool_idx is not None
                and self.disaggregation_mode != DisaggregationMode.DECODE
            ):
                release_kv_cache(req, self.tree_cache, is_insert=False)
            logger.debug(f"Abort queued request. {req.rid=}")

        if self.dllm_config is not None:
            for req in self.dllm_manager.pop_aborted_reqs(
                recv_req.abort_all, recv_req.rid
            ):
                if self.enable_hicache_storage:
                    self.tree_cache.release_aborted_request(req.rid)
                self.ipc_channels.send_to_tokenizer.send_output(
                    _make_abort_req(req), req
                )
                if (
                    req.kv.req_pool_idx is not None
                    or getattr(req, "mamba_pool_idx", None) is not None
                ):
                    release_kv_cache(req, self.tree_cache, is_insert=False)
                logger.debug(f"Abort dLLM queued request. {req.rid=}")

        # Delete the requests in the grammar queue
        # Abort method 2: call `set_finish_with_abort`
        # The request will still run one prefill forward pass.
        # In this case, we change the input_ids to be only one token to make this prefill cheap.
        self.grammar_manager.abort_requests(recv_req)

        # Delete requests not in the waiting queue when PD disaggregation is enabled
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # Abort requests that have not yet been bootstrapped
            for req in self.disagg_prefill_bootstrap_queue.queue:
                if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort bootstrap queue request. {req.rid=}")
                    if self.enable_hicache_storage:
                        self.tree_cache.release_aborted_request(req.rid)

                    if hasattr(req.disagg_kv_sender, "abort"):
                        req.disagg_kv_sender.abort()
                    if self.ps.pp_size > 1:
                        prepare_abort(req, "Aborted by AbortReq.")

            # Abort in-flight requests
            for req in self.disagg_prefill_inflight_queue:
                if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort inflight queue request. {req.rid=}")
                    if hasattr(req.disagg_kv_sender, "abort"):
                        req.disagg_kv_sender.abort()

        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            # Abort requests that have not yet finished preallocation
            for decode_req in self.disagg_decode_prealloc_queue.queue:
                if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort prealloc queue request. {decode_req.req.rid=}")
                    decode_req.kv_receiver.abort()
                    if self.ps.pp_size > 1:
                        prepare_abort(decode_req.req, "Aborted by AbortReq.")

            # Abort requests waiting for kvcache to release tree cache
            for decode_req in self.disagg_decode_transfer_queue.queue:
                if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort transfer queue request. {decode_req.req.rid=}")
                    receiver = decode_req.kv_receiver
                    receiver.abort()
                    # Arm drain-ack accounting once the ABORT is sent, so acks
                    # arriving before this req is deferred (e.g. during the next
                    # forward step) are captured. A fresh set also drops stale acks
                    # from a prior request that reused this bootstrap_room. A
                    # redundant abort only re-wipes -- holds longer, never releases
                    # early -- so no transition guard is needed.
                    if (
                        receiver.kv_mgr.enable_deferred_decode_kv_release
                        and receiver.abort_notified
                    ):
                        receiver.kv_mgr.register_deferred_abort_room(
                            decode_req.req.bootstrap_room
                        )

            # Abort requests whose KV is already backed up for retraction.
            if self.disagg_decode_prealloc_queue.retracted_queue:
                remaining_retracted = []
                for decode_req in self.disagg_decode_prealloc_queue.retracted_queue:
                    if recv_req.abort_all or decode_req.rid.startswith(recv_req.rid):
                        retraction_discard(
                            decode_req,
                            self.tree_cache,
                            get_disagg().disaggregation_decode_retraction_backup,
                        )
                        self.ipc_channels.send_to_tokenizer.send_output(
                            _make_abort_req(decode_req), decode_req
                        )
                    else:
                        remaining_retracted.append(decode_req)
                self.disagg_decode_prealloc_queue.retracted_queue = remaining_retracted

        # Delete requests in the running batch
        for req in self.collect_inflight_reqs():
            if not req.finished() and (
                recv_req.abort_all or req.rid.startswith(recv_req.rid)
            ):
                # Abort method 3: set `to_finish`
                # The request will still run one decode forward pass.
                # Then we reuse all existing code to clean up the KV cache allocation.
                logger.debug(f"Abort running request. {req.rid=}")
                req.to_finish = FINISH_ABORT()

    def _pause_engine(self) -> Tuple[List[Req], int]:
        raise NotImplementedError()

    def pause_generation(self, recv_req: PauseGenerationReqInput):
        # 中译：暂停生成（常用于权重热更新等场景）。置 _engine_paused 后事件循环只收请求不跑前向。
        assert recv_req.mode in ("in_place", "retract")
        self._engine_paused = True

        if recv_req.mode == "in_place":
            # In-place pause: just set the flag and return immediately.
            # All scheduler state (running_batch, last_batch, chunked_req,
            # result_queue) is left untouched. On resume, the normal event
            # loop (get_next_batch_to_run) handles last_batch merge,
            # chunked_req cleanup, and overlap result processing through
            # the standard code paths. This avoids duplicating batch
            # manipulation logic and the accounting bugs that come with it.
            return

        if self.enable_overlap and self.last_batch:
            # Process the results of the last batch
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        retract_reqs = [r for r in self.running_batch.reqs if not r.finished()]
        if (
            self.last_batch is not None
            and self.last_batch.forward_mode.is_extend()
            # Skip merge for disagg prefill: completed prefill requests are
            # already in disagg_prefill_inflight_queue. Merging them into
            # running_batch leaks them, since the prefill event loop never
            # calls update_running_batch to clean them up.
            and self.disaggregation_mode != DisaggregationMode.PREFILL
        ):
            retract_reqs += [r for r in self.last_batch.reqs if not r.finished()]

        if (
            self.chunked_req is not None
            and not self.chunked_req.finished()
            and self.chunked_req not in retract_reqs
            and self.disaggregation_mode != DisaggregationMode.PREFILL
        ):
            retract_reqs.append(self.chunked_req)

        self.last_batch = None
        self.cur_batch_for_debug = None

        if retract_reqs:
            # Decode-side retract always rebootstraps (recomputes the KV from
            # the prefill), so skip the device->host KV offload that release_req
            # would otherwise do; the offloaded copy would be immediately
            # discarded. Non-decode modes ignore offload_kv (they never offload).
            retract_all(
                reqs=retract_reqs,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                tree_cache=self.tree_cache,
                hisparse_coordinator=self.hisparse_coordinator,
                offload_kv=False,
            )
        self.running_batch.reqs = []
        for req in retract_reqs:
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                if req.output_ids:
                    req.pd_rebootstrap_forced_output_id = req.output_ids.pop()
                req.pd_rebootstrap_in_progress = True
                req.time_stats.set_retract_time()
                self.disagg_decode_prealloc_queue.hold_rebootstrap(req)
            else:
                self._add_request_to_queue(req)
        self.running_batch.batch_is_full = False
        # In disagg-PREFILL, keep a live mid-chunk chunked_req rather than retract it:
        # freeing its KV under a live disagg KV-sender crashes pop_bootstrapped or
        # sends freed/reused KV to decode. Kept, it resumes prefill after the pause.
        # TODO(disagg-prefill-retract): tear the sender down (abort + release metadata
        # buffer + reset pending_bootstrap) before freeing KV, then retract for real.
        # Until then a weight-update pause leaves stale-weight prefix KV (off-policy).
        if self.disaggregation_mode != DisaggregationMode.PREFILL:
            self.chunked_req = None

        # Surface the paused state to dashboards immediately. The scheduler
        # event loop short-circuits before reaching ``on_idle`` while paused,
        # so without this hop ``gen_throughput`` retains its last non-zero
        # value and KV events are not flushed for the entire pause window
        # (e.g. across a weight update). Zero the gauge, force a one-shot
        # idle log by resetting the rate-limit timestamp, and flush pending
        # KV events.
        self.metrics_reporter.last_gen_throughput = 0.0
        if self.metrics_reporter.current_scheduler_metrics_enabled:
            self.metrics_reporter.metrics_collector.last_log_time = 0.0
            self.metrics_reporter._maybe_log_idle_metrics()
        self.kv_events_publisher.publish_kv_events()

    def continue_generation(self, recv_req: ContinueGenerationReqInput):
        if recv_req.torch_empty_cache:
            before_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            torch.cuda.empty_cache()
            after_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            logger.info(
                f"[continue_generation] torch.cuda.empty_cache() called: "
                f"reserved {before_mb:.1f} MB -> {after_mb:.1f} MB "
                f"(freed {before_mb - after_mb:.1f} MB)"
            )
        # Enqueue any rebootstrap requests that were staged during a
        # retract-mode pause. Deferring until resume keeps the preallocation
        # queue empty during the pause window (so an intervening weight update
        # can flush the cache) and recomputes the prefix KV under the updated
        # weights.
        if (
            self.disaggregation_mode == DisaggregationMode.DECODE
            and self.disagg_decode_prealloc_queue is not None
        ):
            self.disagg_decode_prealloc_queue.enqueue_held_rebootstrap()
        self._engine_paused = False

    def handle_scale_elastic_ep(
        self, recv_req: ScaleElasticEPReqInput
    ) -> ScaleElasticEPReqOutput:
        """Begin a pending elastic EP scale-up request."""
        from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager

        old_ep_size = ElasticEPStateManager.get_effective_ep_size()
        new_ep_size = recv_req.new_ep_size
        max_ep_size = get_parallel().max_ep_size or old_ep_size

        logger.debug(
            "[Elastic EP][scale] request received: new_ep_size=%d "
            "old_ep_size=%d max_ep_size=%d",
            new_ep_size,
            old_ep_size,
            max_ep_size,
        )

        if new_ep_size <= old_ep_size:
            return ScaleElasticEPReqOutput(
                success=False,
                message=(
                    f"new_ep_size ({new_ep_size}) must be greater than current "
                    f"effective_ep_size ({old_ep_size})."
                ),
                old_ep_size=old_ep_size,
                new_ep_size=new_ep_size,
            )
        if new_ep_size > max_ep_size:
            return ScaleElasticEPReqOutput(
                success=False,
                message=(
                    f"new_ep_size ({new_ep_size}) exceeds --max-ep-size "
                    f"({max_ep_size}). Restart with a larger --max-ep-size."
                ),
                old_ep_size=old_ep_size,
                new_ep_size=new_ep_size,
            )
        if ElasticEPStateManager.is_scaling():
            return ScaleElasticEPReqOutput(
                success=False,
                message=(
                    "A previous scale operation has not completed yet. Wait until "
                    "all pending ranks have joined before issuing another scale."
                ),
                old_ep_size=old_ep_size,
                new_ep_size=new_ep_size,
                pending_ep_size=ElasticEPStateManager.get_pending_ep_size(),
                scale_phase=ElasticEPStateManager.get_scale_phase(),
            )

        if not ElasticEPStateManager.request_scale(new_ep_size):
            return ScaleElasticEPReqOutput(
                success=False,
                message=(
                    "Failed to queue elastic EP scale: no elastic state or "
                    "scale already pending."
                ),
                old_ep_size=old_ep_size,
                new_ep_size=new_ep_size,
                pending_ep_size=ElasticEPStateManager.get_pending_ep_size(),
                scale_phase=ElasticEPStateManager.get_scale_phase(),
            )
        if (eplb_manager := self.tp_worker.model_runner.eplb_manager) is not None:
            eplb_manager.disable_rebalance("elastic EP scale-up is pending")
        logger.debug(
            "[Elastic EP][scale] scale requested: target_ep_size=%d; "
            "waiting for a joining cohort",
            new_ep_size,
        )

        return ScaleElasticEPReqOutput(
            success=True,
            message=f"Scaling initiated from {old_ep_size} to {new_ep_size}",
            old_ep_size=old_ep_size,
            new_ep_size=new_ep_size,
            pending_ep_size=ElasticEPStateManager.get_pending_ep_size(),
            scale_phase=ElasticEPStateManager.get_scale_phase(),
        )

    def load_lora_adapter(
        self, recv_req: LoadLoRAAdapterReqInput
    ) -> LoadLoRAAdapterReqOutput:
        """In-place loading a new lora adapter from disk or huggingface."""

        result = self.tp_worker.load_lora_adapter(recv_req)
        return result

    def load_lora_adapter_from_tensors(
        self, recv_req: LoadLoRAAdapterFromTensorsReqInput
    ) -> LoadLoRAAdapterFromTensorsReqOutput:
        """In-place loading a new lora adapter from serialized tensors."""

        result = self.tp_worker.load_lora_adapter_from_tensors(recv_req)
        return result

    def unload_lora_adapter(
        self, recv_req: UnloadLoRAAdapterReqInput
    ) -> UnloadLoRAAdapterReqOutput:
        """Unload the lora adapter."""

        result = self.tp_worker.unload_lora_adapter(recv_req)
        return result

    def init_weights_send_group_for_remote_instance(
        self, recv_req: InitWeightsSendGroupForRemoteInstanceReqInput
    ):
        """Init the seed and client instance communication group."""
        success, message = self.tp_worker.init_weights_send_group_for_remote_instance(
            recv_req
        )
        return InitWeightsSendGroupForRemoteInstanceReqOutput(
            success=success, message=message
        )

    def send_weights_to_remote_instance(
        self, recv_req: SendWeightsToRemoteInstanceReqInput
    ):
        """Send the seed instance weights to the destination instance."""
        success, message = self.tp_worker.send_weights_to_remote_instance(recv_req)
        return SendWeightsToRemoteInstanceReqOutput(success=success, message=message)

    def slow_down(self, recv_req: SlowDownReqInput):
        t = recv_req.forward_sleep_time
        if t is not None and t <= 0:
            t = None
        self.forward_sleep_time = t
        return SlowDownReqOutput()

    def expert_distribution_handle(self, recv_req: ExpertDistributionReq):
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
        output = self.session_controller.open(recv_req)
        if output.success and self.enable_session_radix_cache:
            self.tree_cache.open_radix_session(recv_req.session_id)
        if self.ps.pp_rank == 0 and self.ps.tp_rank == 0 and self.ps.attn_cp_rank == 0:
            return output
        return None

    def close_session(self, recv_req: CloseSessionReqInput):
        if self.enable_session_radix_cache:
            self.tree_cache.release_radix_session(recv_req.session_id)
        if (
            recv_req.session_id in self.session_controller
            or not self.enable_session_radix_cache
        ):
            self.session_controller.close(recv_req)

    def maybe_sleep_on_idle(self):
        if self.idle_sleeper is not None:
            self.idle_sleeper.maybe_sleep()

    def handle_freeze_gc(self, recv_req: FreezeGCReq):
        """Handle freeze_gc request: freeze scheduler's GC and forward to detokenizer."""
        freeze_gc("Scheduler")
        self.ipc_channels.send_to_detokenizer.send_output(recv_req, recv_req)
        return None

    def handle_shutdown(self, recv_req: ShutdownReq):
        # Break the event loop; the finally in run_scheduler_process releases resources.
        self.gracefully_exit = True
        return None

    def configure_logging(self, recv_req: ConfigureLoggingReq):
        if recv_req.log_level is not None:
            logging.getLogger().setLevel(recv_req.log_level.upper())
        self.ipc_channels.send_to_detokenizer.send_output(recv_req, recv_req)

    def handle_dumper_control(self, recv_req: DumperControlReqInput):
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
            self.ipc_channels.send_to_tokenizer.send_output(
                DumperControlReqOutput(success=True, response=response), recv_req
            )
        except Exception as e:
            print(f"[Scheduler] handle_dumper_control error: {e}", flush=True)
            self.ipc_channels.send_to_tokenizer.send_output(
                DumperControlReqOutput(success=False, response=[], error=str(e)),
                recv_req,
            )

    # placeholder for override
    def update_cache_from_scheduler(
        self, schedule_batch: ScheduleBatch, batch_result: GenerationBatchResult
    ):
        pass


def dispatch_event_loop(scheduler: Scheduler):
    # 中译：按"PD 分离模式 + 是否 PP + 是否 pdmux/MLX + 是否重叠"组合，选择并进入对应的事件循环。
    # The live PP property asserts before torch.distributed init (MLX stub).
    disaggregation_mode: DisaggregationMode = scheduler.disaggregation_mode
    if disaggregation_mode == DisaggregationMode.NULL:
        if scheduler.enable_pdmux:
            scheduler.event_loop_pdmux()
        elif get_parallel().pp_size > 1:
            scheduler.event_loop_pp()
        elif scheduler.enable_overlap_mlx:
            scheduler.event_loop_overlap_mlx()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap()
        else:
            scheduler.event_loop_normal()
    elif disaggregation_mode == DisaggregationMode.PREFILL:
        if get_parallel().pp_size > 1:
            scheduler.event_loop_pp_disagg_prefill()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap_disagg_prefill()
        else:
            scheduler.event_loop_normal_disagg_prefill()
    elif disaggregation_mode == DisaggregationMode.DECODE:
        if get_parallel().pp_size > 1:
            scheduler.event_loop_pp_disagg_decode()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap_disagg_decode()
        else:
            scheduler.event_loop_normal_disagg_decode()


def configure_scheduler_process(
    server_args: ServerArgs,
    gpu_id: int,
    tp_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    display_tp_rank: Optional[int] = None,
    display_dp_rank: Optional[int] = None,
    display_moe_ep_rank: Optional[int] = None,
) -> Optional[int]:
    """Configure scheduler worker logging and process title.

    display_* ranks are cosmetic; runtime ranks stay local.
    """
    kill_itself_when_parent_died()

    # Generate the logger prefix
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    shown_dp = display_dp_rank if display_dp_rank is not None else dp_rank
    shown_tp = display_tp_rank if display_tp_rank is not None else tp_rank
    shown_moe_ep = (
        display_moe_ep_rank if display_moe_ep_rank is not None else moe_ep_rank
    )

    prefix = ""
    if shown_dp is not None:
        prefix += f" DP{shown_dp}"
    if get_parallel().pp_size > 1:
        prefix += f" PP{pp_rank}"
    if get_parallel().attn_cp_size > 1:
        prefix += f" ATTN_CP{attn_cp_rank}"
    if get_parallel().moe_dp_size > 1:
        prefix += f" MOE_DP{moe_dp_rank}"
    if get_parallel().tp_size > 1:
        prefix += f" TP{shown_tp}"
    if get_parallel().ep_size > 1:
        prefix += f" EP{shown_moe_ep}"

    # Config the process
    setproctitle.setproctitle(f"sglang::scheduler{prefix.replace(' ', '_')}")
    faulthandler.enable()

    # Configure the logger
    configure_logger(server_args, prefix=prefix)
    suppress_other_loggers()

    # Set cpu affinity to this gpu process
    if envs.SGLANG_SET_CPU_AFFINITY.get():
        set_gpu_proc_affinity(
            get_parallel().pp_size,
            get_parallel().tp_size,
            get_parallel().nnodes,
            gpu_id,
        )
    if not envs.SGLANG_NUMA_BIND_V2.get():
        numa_node = get_numa_node_if_available(server_args, gpu_id)
        if numa_node is not None:
            numa_bind_to_node(numa_node)

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
    display_tp_rank: Optional[int] = None,
    display_dp_rank: Optional[int] = None,
    display_moe_ep_rank: Optional[int] = None,
):
    # 中译：调度器子进程的入口函数。由父进程为每个 GPU/rank 启动。流程：
    #       加载插件 -> 配置进程（日志/进程名/CPU 亲和性）-> 可选初始化 tracing ->
    #       创建 Scheduler 实例 -> 通过 pipe 把初始化信息回传父进程 -> 进入事件循环。
    #       出异常时记录、给父进程发 SIGQUIT，并按需 SIGKILL 整个进程组以避免噪声回溯。
    # Load plugins so hooks can override Scheduler and its dependencies.
    load_plugins()
    # Publish before anything in this process reads configuration.
    publish(server_args, role="scheduler")
    dp_rank = configure_scheduler_process(
        server_args,
        gpu_id,
        tp_rank,
        attn_cp_rank,
        moe_dp_rank,
        moe_ep_rank,
        pp_rank,
        dp_rank,
        display_tp_rank=display_tp_rank,
        display_dp_rank=display_dp_rank,
        display_moe_ep_rank=display_moe_ep_rank,
    )
    parent_process = psutil.Process().parent()

    # Set up tracing
    if get_observability().enable_trace:
        process_tracing_init(
            get_observability().otlp_traces_endpoint,
            "sglang",
            trace_modules=get_observability().trace_modules,
        )
        thread_label = "Scheduler"
        if get_disagg().disaggregation_mode == "prefill":
            thread_label = "Prefill Scheduler"
        elif get_disagg().disaggregation_mode == "decode":
            thread_label = "Decode Scheduler"
        trace_set_thread_info(thread_label, tp_rank, dp_rank, pp_rank)

    # Create a scheduler and run the event loop
    # 中译：创建调度器并运行事件循环（注意参数顺序与 Scheduler.__init__ 略有差异）。
    scheduler = None
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
        # 中译：把初始化信息（max_total_num_tokens 等握手数据）通过管道回传父进程，告知"已就绪"。
        pipe_writer.send(scheduler.get_init_info())

        # 中译：进入事件循环，阻塞运行直到进程关闭。
        # Run the event loop (blocks until a ShutdownReq sets gracefully_exit)
        scheduler.run_event_loop()

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
        # Opt-in: SIGKILL the pgroup so sibling ranks don't spew thousands
        # of NCCL/TCPStore tracebacks before they finally die.
        if envs.SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION.get():
            try:
                os.killpg(os.getpgrp(), signal.SIGKILL)
            except Exception:
                pass
    finally:
        if scheduler is not None:
            # FPM has a background ZMQ publisher thread that needs explicit
            # teardown to flush queued metrics and close the socket cleanly.
            scheduler.metrics_reporter._shutdown_fpm()
            # Graceful path only: on the exception path the GPU may be wedged
            # and the synchronize() in destroy() could itself hang.
            if scheduler.gracefully_exit:
                scheduler.release_host_resources()


def _make_abort_req(
    req: Req, finished_reason: Optional[FinishReasonDict] = None
) -> AbortReq:
    return AbortReq(
        rid=req.rid,
        finished_reason=finished_reason,
        weight_versions=compute_weight_version_spans(
            req.weight_version_events,
            current_version=get_serving().weight_version,
            num_output_tokens=len(req.output_ids),
        ),
    )
