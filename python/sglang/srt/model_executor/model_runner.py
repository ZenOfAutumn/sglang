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
"""ModelRunner runs the forward passes of the models.

中文说明：ModelRunner 是模型推理的核心执行器，负责驱动模型的前向（forward）计算。
它贯穿了从权重加载、分布式环境初始化、KV 缓存内存池分配、
注意力后端（attention backend）初始化、CUDA Graph 捕获，到最终执行
预填（prefill）/解码（decode）前向与采样（sampling）的整个生命周期。
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import time
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.distributed as dist

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import (
    AttentionArch,
    ModelConfig,
    ModelImpl,
)
from sglang.srt.configs.update_config import adjust_config_with_unaligned_cpu_tp
from sglang.srt.debug_utils.dumper import dumper
from sglang.srt.distributed import bootstrap
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    maybe_init_shared_mooncake_transfer_engine,
)
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.elastic_ep.elastic_ep import (
    ElasticEPStateManager,
    get_healthy_expert_location_src_rank,
    get_scale_cohort_target,
    join_process_groups,
    join_scale_process_group,
    maybe_rebalance_after_rank_fault,
    maybe_recover_ep_ranks,
    register_scale_cohort,
    try_admit_scale_ranks,
)
from sglang.srt.elastic_ep.expert_backup_client import ExpertBackupClient
from sglang.srt.environ import envs
from sglang.srt.eplb.eplb_manager import EPLBManager
from sglang.srt.eplb.expert_distribution import (
    ExpertDistributionMetrics,
    ExpertDistributionRecorder,
    get_global_expert_distribution_recorder,
    set_global_expert_distribution_recorder,
)
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    append_trivial_expert_slots,
    broadcast_global_expert_location_metadata,
    compute_initial_expert_location_metadata,
    format_expert_location_layout,
    get_global_expert_location_metadata,
    set_global_expert_location_metadata,
)
from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
from sglang.srt.kv_canary.api import install_canary
from sglang.srt.kv_canary.runner.canary_manager import context_tuple
from sglang.srt.kv_canary.token_oracle.install import install_token_oracle_from_env
from sglang.srt.layers import deep_gemm_wrapper, model_parallel
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
from sglang.srt.layers.cp.utils import (
    get_cp_strategy,
    is_cp_v2_active,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import create_sampler
from sglang.srt.layers.utils.cp_utils import is_mla_prefill_cp_enabled
from sglang.srt.lora.lora_manager import LoRAManager, init_lora_cuda_graph_moe_buffers
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.schedule_batch import sanity_check_mm_pad_shift_value
from sglang.srt.mem_cache import kv_cache_dtype
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.kv_cache_configurator import (
    KVCacheConfigurator,
)
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.model_executor.cuda_graph_config import (
    cuda_graph_fully_disabled,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    PPProxyTensors,
)
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    has_forward_context,
)
from sglang.srt.model_executor.graph_memory_usage import (
    replace_graph_memory_usage,
    replace_graph_time_usage,
)
from sglang.srt.model_executor.model_runner_components import misc_utils
from sglang.srt.model_executor.model_runner_components.attention_backend_setup import (
    build_attention_backends,
    configure_aux_hidden_state_capture,
    get_attention_backend,
    resolve_attention_backend_strs,
)
from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
    capture_cuda_graphs,
    capture_decode_graph,
    capture_prefill_graph,
)
from sglang.srt.model_executor.model_runner_components.kv_pool_runtime import (
    compute_post_capture_kv_resize,
    is_post_capture_kv_active,
)
from sglang.srt.model_executor.model_runner_components.layer_setup import (
    ModelLayerInfo,
    adjust_hybrid_swa_layer_ids,
    resolve_layer_indices,
)
from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    build_load_config,
    dist_barrier_after_load,
    load_kv_cache_scales,
    load_model_with_memory_saver,
    maybe_downgrade_dtype_for_legacy_gpu,
    maybe_enable_ipc_weight_cache,
    maybe_precompile_model_kernels_after_loading,
    maybe_register_debug_tensor_dump_hook,
    maybe_trigger_remote_instance_nccl_send_group,
    report_online_quantization,
    resolve_sliding_window_size,
)
from sglang.srt.model_executor.model_runner_components.moe_ep_setup import (
    check_quantized_moe_compatibility,
    init_lplb_solvers,
    prepare_moe_topk,
)
from sglang.srt.model_executor.model_runner_components.ngram_embedding_manager import (
    NgramEmbeddingManager,
)
from sglang.srt.model_executor.model_runner_components.remote_instance_weight_transporter import (
    RemoteInstanceWeightTransporter,
)
from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
    SpecAuxHiddenStateConfig,
    resolve_spec_aux_hidden_state_config,
)
from sglang.srt.model_executor.model_runner_components.weight_exporter import (
    WeightExporter,
)
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.model_executor.runner import (
    EagerRunner,
    get_batch_sizes_to_capture,
)
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import (
    assert_published,
    get_context,
    get_device,
    get_exec,
    get_global_dwdp_manager,
    get_lora,
    get_memory,
    get_model,
    get_observability,
    get_parallel,
    get_schedule,
    get_spec,
    is_ep_joiner,
    is_ep_scale_joiner,
    max_speculative_num_draft_tokens,
    remote_instance_transfer_engine_enabled,
    set_global_dwdp_manager,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_observer import SamplingObserver
from sglang.srt.server_args import (  # noqa: F401  (re-export)
    CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS,
    ServerArgs,
    add_chunked_prefix_cache_attention_backend,
    get_global_server_args,
)
from sglang.srt.speculative.adaptive_spec_params import (
    resolve_candidate_steps_from_config,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import resolve_num_tokens_per_req
from sglang.srt.state_capturer.base import TopkCaptureOutput
from sglang.srt.state_capturer.indexer_topk import (
    create_indexer_capturer,
    get_global_indexer_capturer,
    set_global_indexer_capturer,
)
from sglang.srt.state_capturer.routed_experts import (
    RoutedExpertsCapturer,
    disable_routed_experts_capture_for_draft,
    get_global_experts_capturer,
    set_global_experts_capturer,
)
from sglang.srt.utils import (
    cpu_has_amx_support,
    enable_show_time_cost,
    get_available_gpu_memory,
    is_host_cpu_arm64,
    is_npu,
    numa_utils,
    require_gathered_buffer,
    reserve_rope_cache_for_long_sequences,
    set_cuda_arch,
    slow_rank_detector,
)
from sglang.srt.utils.device_timer import device_timer_ctx
from sglang.srt.utils.nvtx_pytorch_hooks import PytHooks
from sglang.srt.utils.nvtx_utils import profile_range
from sglang.srt.utils.offloader import (
    create_offloader,
    get_offloader,
    set_offloader,
)
from sglang.srt.utils.profile_utils import build_step_span_name
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils.weight_checker import WeightChecker

_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu_arm64 = is_host_cpu_arm64()

if _is_npu:
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend

    init_npu_backend()
elif current_platform.is_out_of_tree():
    current_platform.init_backend()

# 中译：用于检测模型加载过程中“拖后腿”（加载过慢）的 rank。
# Detect stragger ranks in model loading


logger = logging.getLogger(__name__)


    # 中译：从（可能是多模态或组合型的）模型中解析出其底层的语言模型子模块。
def _prefill_cuda_graph_allows_context_parallel(
    prefill_runner, forward_batch: ForwardBatch
) -> bool:
    """Allow CP only through a runner that captured the validated CP-v2 body."""
    return get_cp_strategy() is None or (
        bool(getattr(prefill_runner, "enable_cp_v2_bcg_capture", False))
        and is_cp_v2_active(forward_batch)
    )


@dataclass
class ModelRunnerOutput:
    # 中译：ModelRunner 一次前向的输出容器。
    # logits_output：logits 处理器输出，或流水线并行（PP）场景下的代理张量。
    logits_output: Union[LogitsProcessorOutput, PPProxyTensors]
    # can_run_graph：本次前向是否命中了 CUDA Graph（即是否以 graph replay 方式执行）。
    can_run_graph: bool
    # expert_distribution_metrics / routed_experts_output / indexer_topk_output：
    #   MoE 专家分布指标与状态捕获（state capturer）输出，仅在开启相应功能时非空。
    expert_distribution_metrics: Optional[ExpertDistributionMetrics] = None
    routed_experts_output: Optional[TopkCaptureOutput] = None
    indexer_topk_output: Optional[TopkCaptureOutput] = None


    # 中译：懒加载（首次用到时才构建）的 eager 模式输入缓冲区注册表，以及它被创建时所依据的容量
def resolve_draft_attention_backend(
    *,
    draft_attention_backend: Optional[str],
    is_draft_worker: bool,
) -> Optional[str]:
    """The attention backend a runner uses because it is a draft runner.

    ``None`` for a target runner. For a draft: the backend the algorithm that
    built it resolved (the supported-backend fallback in
    ``build_draft_tp_worker``), else ``--speculative-draft-attention-backend``.
    It belongs to the runner, not the process: target and draft coexist.
    """
    if not is_draft_worker:
        return None
    return draft_attention_backend or get_spec().speculative_draft_attention_backend
class ModelRunner:
    """ModelRunner runs the forward passes of the models."""

    @property
    def sampling_observer(self) -> Optional[SamplingObserver]:
        return self._sampling_observer

    @sampling_observer.setter
    def sampling_observer(self, observer: Optional[SamplingObserver]) -> None:
        if observer is not None and not self.supports_sampling_observer():
            raise ValueError(
                "sampling observers are not supported by the configured "
                "sampling path"
            )
        self._sampling_observer = observer

    def supports_sampling_observer(self) -> bool:
        """Whether this runner's sampling path publishes observer output."""
        return self.server_args.dllm_algorithm is None and self.spec_algorithm.is_none()

    def __init__(
        self,
        # 中译：模型配置对象，包含 HF 配置、是否生成模型、注意力架构（MHA/MLA）、是否多模态等元信息。
        model_config: ModelConfig,
        # 中译：静态显存占比，即权重 + KV 缓存等固定占用相对于 GPU 总显存的上限比例（0~1）。
        mem_fraction_static: float,
        # 中译：本 worker 绑定的物理 GPU 编号。
        gpu_id: int,
        # 中译：张量并行（Tensor Parallel）中当前进程的 rank（从 0 开始）。
        # 中译：张量并行的总规模（参与 TP 的进程数）。
        # 中译：MoE 专家并行（Expert Parallel）中当前进程的 rank。
        # 中译：MoE 专家并行的总规模。
        # 中译：流水线并行（Pipeline Parallel）中当前进程所处的 stage rank。
        # 中译：流水线并行的总规模（stage 数）。
        # 中译：用于初始化 NCCL 分布式通信组的端口号。
        ps: ParallelState,
        nccl_port: int,
        # 中译：全局服务端参数（ServerArgs），承载几乎所有用户可配置的运行时选项。
        server_args: ServerArgs,
        # 中译：数据并行（Data Parallel）中当前进程的 rank；非 DP 场景为 None。
        # 中译：注意力上下文并行（Context Parallel）中当前进程的 rank；未启用时为 None。
        # 中译：MoE 数据并行中当前进程的 rank；未启用时为 None。
        # 中译：是否为投机解码（speculative decoding）中的草稿（draft）worker；True 表示运行的是小草稿模型。
        is_draft_worker: bool = False,
        # 中译：请求到 token 的映射内存池；为 None 时由 ModelRunner 自行分配，传入则复用（如 draft 复用 target 的池）。
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        # 中译：token 到 KV 缓存的内存池分配器；为 None 时自行分配，传入则复用。
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
        # 中译：内存池尺寸配置；target 上由 `_resolve_memory_pool_config` 解析，draft worker 直接传入以复用。
        memory_pool_config: Optional[MemoryPoolConfig] = None,
        # 中译：草稿模型索引；在多草稿模型（如 EAGLE 多步）场景下用于区分不同的 draft 模型。
        draft_model_idx: Optional[int] = None,
        draft_attention_backend: Optional[str] = None,
    ):
        # Parse args
        # 中译：解析构造参数，将各种并行/设备/模型配置保存为实例属性。
        # mem_fraction_static：静态显存占比（权重 + KV 缓存等固定占用的显存上限比例）。
        self.mem_fraction_static = mem_fraction_static
        # Set on target by `_resolve_memory_pool_config`; passed in for draft
        # workers so they reuse target's resolved sizes (replaces legacy
        # `server_args._draft_pool_config` mutation hack).
        # 中译：在 target（主模型）上由 `_resolve_memory_pool_config` 设置；对于 draft（草稿）
        #       worker 则直接传入，以便复用 target 已解析好的内存池尺寸（取代了旧的
        #       `server_args._draft_pool_config` 直接改写的不优雅做法）。
        self.memory_pool_config = memory_pool_config
        self.gpu_id = gpu_id
        self.ps = ps
        self.model_config = model_config
        self.dist_port = nccl_port
        self.server_args = server_args
        self.is_draft_worker = is_draft_worker
        # The process entry published; a draft runner is not one (it must not
        # clobber the target's config), so only the target checks.
        if not is_draft_worker:
            assert_published(server_args, role="scheduler")
        # Set by maybe_init_lora_manager; stays None when LoRA is off and on
        # draft runners, which serve adapters' target model unadapted.
        self.lora_manager: Optional[LoRAManager] = None
        self.device = get_device().device
        self.draft_attention_backend = resolve_draft_attention_backend(
            draft_attention_backend=draft_attention_backend,
            is_draft_worker=is_draft_worker,
        )
        # This runner's own load format, resolved before anything keys off it:
        # the remote-instance transfer engine is initialized at the top of
        # initialize(), long before the weights are loaded.
        self.draft_load_format = self._resolve_draft_load_format()
        self.is_generation = model_config.is_generation
        self.device_timer = None
        self.is_multimodal = model_config.is_multimodal
        self.is_multimodal_chunked_prefill_supported = (
            model_config.is_multimodal_chunked_prefill_supported
        )
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )
        self.capture_tail_hooks = []
        self.page_size = get_schedule().page_size
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.mtp_draft_device_pools = ()
        self.is_hybrid_swa = model_config.is_hybrid_swa
        self.is_hybrid_swa_compress = model_config.is_hybrid_swa_compress
        self.use_mla_backend = self.model_config.attention_arch == AttentionArch.MLA
        self.attention_chunk_size = model_config.attention_chunk_size
        self.enable_elastic_ep = get_exec().moe.elastic_ep_backend is not None
        self.forward_pass_id = 0
        self._pending_elastic_scale_update = None
        self.init_new_workspace = False
        self.draft_model_idx = draft_model_idx
        self.enable_hisparse = get_memory().enable_hisparse
        self._sampling_observer: Optional[SamplingObserver] = None

        self.init_startup_observability()

        self.init_remote_instance_weight_transporter()

        self.init_msprobe()

        # auxiliary hidden capture mode. TODO: expose this to server args?
        # 中译：辅助隐藏状态（aux hidden state）捕获模式的相关开关。用于投机采样
            # 中译：加载草稿模型的配置，以获取其层数，用于为 KV 缓存定尺寸。
                    # 中译：若配置中没有辅助层信息，则置为 None。
            # 中译：选择需要捕获的主模型目标层，用于构建 DFlash 的上下文特征。
        # 中译：若开启了耗时统计展示，则启用相应的计时输出。
        # 中译：针对特定模型的参数调整（例如某些模型需要特殊的注意力架构/配置修正）。
        # 中译：在调度器进程中设置全局 server_args，供后续各模块读取。
        # 中译：（不优雅的临时做法）将 use_mla_backend 写入全局 server_args。
        self.init_spec_aux_hidden_state()

        # Apply the rank zero filter to logger
        if get_observability().show_time_cost:
            enable_show_time_cost()
        misc_utils.maybe_disable_chunked_prefix_cache(
            use_mla_backend=self.use_mla_backend,
            is_draft_worker=self.is_draft_worker,
        )

        # Init OpenMP threads binding for CPU
        # 中译：在 CPU 设备上初始化 OpenMP 线程绑定（绑核），以提升 NUMA 亲和性与性能。
        if self.device == "cpu":
            self.init_threads_binding()

        # Set float32 matmul precision
        if get_exec().features.enable_tf32_matmul:
            torch.set_float32_matmul_precision("high")

        # Set device early so that TransferEngine init (e.g. Ascend NPU)
        # can access the device context.
        try:
            torch.get_device_module(self.device).set_device(ps.gpu_id)
        except Exception:
            import os

            logger.warning(
                f"Context: {self.device=} {ps.gpu_id=} {os.environ.get('CUDA_VISIBLE_DEVICES')=} {ps.tp_rank=} {ps.tp_size=}"
            )
            raise

        # Initialize MooncakeTransferEngine BEFORE init_torch_distributed so
        # that the shared TE can be passed to the Mooncake PG backend (avoids
        # creating duplicate TransferEngines).
        self.init_shared_mooncake_transfer_engine()

        # Get available memory before model loading.
        # Stored for later use by alloc_memory_pool().
        # 中译：在加载模型之前记录可用显存（同时完成 torch 分布式初始化）。
        # 中译：初始化共享的 Mooncake 传输引擎（用于 PD 分离/远程权重传输等场景）。
        self.init_torch_distributed()

        # Init forward stream for overlap schedule
        # 中译：为 overlap 调度（计算与调度重叠）创建专用的前向 CUDA stream。
        self.forward_stream = torch.get_device_module(self.device).Stream()

        # 中译：根据 server_args 创建并设置 CPU offload（将部分权重/激活卸载到 CPU）的加载器。
        # Read-done mailbox: the scheduler's WAR barrier reads it from the runner
        # its worker names, and treats None as the coarse whole-forward fence.
        self.shared_read_done_event: Optional[torch.cuda.Event] = None

        # CPU offload
        set_offloader(create_offloader(dp_rank=self.ps.dp_rank))

        self._weight_checker = WeightChecker(get_model=lambda: self.model, ps=self.ps)

        if envs.SGLANG_DETECT_SLOW_RANK.get():
            slow_rank_detector.execute()

        # Init mindspore running environment when model impl is "mindspore"
        self.init_mindspore_runner()

        # Update deep gemm configure
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            deep_gemm_wrapper.update_deep_gemm_config(gpu_id)

        # For hisparse (must be set before initialize() so CUDA graph capture can see it)
        # 中译：用于 hisparse（分层稀疏注意力）。必须在 initialize() 之前设置，
        #       以便 CUDA Graph 捕获时能看到它。
        self.hisparse_coordinator = None

        # The native overlap path replaces this during load_model(). Keep the
        # no-pending-work invariant for lightweight backends that override the
        # base initialization and weight-loading flow.
        self.startup_weight_load = None

        # Load model weights and configure
        # 中译：加载模型权重并完成配置（这是 __init__ 中最重的一步）。
        self.initialize()
        # 中译：检查量化 MoE 的兼容性（例如某些量化方案与 MoE 后端是否可同时使用）。
        self.check_quantized_moe_compatibility()

        self._initialize_elastic_ep_joiner()

        if self.is_multimodal:
            # 中译：多模态模型需检查多模态 pad/shift 值与词表大小是否一致（避免占位符越界）。
            sanity_check_mm_pad_shift_value(self.model_config.vocab_size)

        # Temporary cached values
        # 中译：一些临时缓存值（例如模型 forward 是否支持流水线并行代理张量参数）。
        self.support_pp = (
            "pp_proxy_tensors" in inspect.signature(self.model.forward).parameters
        )

            # 中译：开启流水线并行（PP）时，模型必须支持 PP，否则报错。
        if self.ps.pp_size > 1:
            assert (
                self.support_pp
            ), "Pipeline Parallel is not compatible with this model."

        # For weight updates
        # 中译：用于在线权重更新（如 RLHF rollout）的进程组句柄缓存。
        self.init_weight_updater()
        self.init_weight_exporter()

    def init_startup_observability(self) -> None:
        self.weight_load_time = 0.0
        self.graph_memory_usage: dict[str, float] = {}
        self.graph_time_usage: dict[str, float] = {}

    def _initialize_elastic_ep_joiner(self) -> None:
        if not (get_exec().moe.elastic_ep_backend is not None and is_ep_scale_joiner()):
            return

        join_effective_ep_size = get_parallel().ep_join_rank_offset + self.ps.tp_size
        dist.barrier(group=self.tp_group.cpu_group)
        if self.ps.tp_rank == 0:
            register_scale_cohort(
                get_parallel().ep_join_rank_offset,
                join_effective_ep_size,
            )
        join_scale_process_group()
        get_context().override("elastic_ep.scale_join", ep_size=join_effective_ep_size)

        global_ep_rank = self.ps.tp_rank + get_parallel().ep_join_rank_offset
        broadcast_global_expert_location_metadata(
            model_config=self.model_config,
            moe_ep_rank=global_ep_rank,
            src_rank=0,
        )
        set_global_expert_distribution_recorder(
            ExpertDistributionRecorder.init_new(
                get_global_expert_location_metadata(),
                rank=global_ep_rank,
            )
        )

        # 中译：初始化 msprobe（昂腾提供的精度调试/张量 dump 工具）。
        from sglang.srt.layers.dp_attention import (
            enable_joiner_all_gather,
            update_dp_attention_post_scale,
        )

        enable_joiner_all_gather()
        update_dp_attention_post_scale(
            new_dp_size=join_effective_ep_size,
            new_dp_rank=global_ep_rank,
        get_context().override("elastic_ep.scale_join", dp_size=join_effective_ep_size)
        if self.eplb_manager is not None:
            self.eplb_manager.disable_rebalance(
                "EPLB rebalance is disabled while elastic EP scale-up "
                "is being finalized"
            )

        state = ElasticEPStateManager.instance()
        if state is not None:
            state.active_ranks.zero_()
            state.active_ranks[:join_effective_ep_size] = 1
            state.snapshot_active_to_last()
            state.sync_active_to_cpu()
            state.scale_phase = "syncing_new_world"
        self._elastic_scale_ready_barrier(
            target_size=join_effective_ep_size,
            log_tag="JOINER",
        )
        if state is not None:
            state.scale_phase = "serving_expanded"
        self._rearm_eplb_after_elastic_scale()

    def init_msprobe(self):
        self.msprobe_debugger = misc_utils.create_msprobe_debugger()

    def init_weight_updater(self):
        self.weight_updater = WeightUpdater(
            tp_rank=self.ps.tp_rank,
            device=self.device,
            gpu_id=self.gpu_id,
            model_config=self.model_config,
            custom_weight_loaders=get_model().custom_weight_loader,
            get_model=lambda: self.model,
            update_model_fields=self.update_model_fields,
            recapture_cuda_graph=self.init_decode_cuda_graph,
            get_model_runner=lambda: self,
        )

    def init_spec_aux_hidden_state(self):
        self.spec_aux_config: SpecAuxHiddenStateConfig = (
            resolve_spec_aux_hidden_state_config(
                server_args=self.server_args,
                model_config=self.model_config,
                spec_algorithm=self.spec_algorithm,
                is_draft_worker=self.is_draft_worker,
            )
        )

    def init_weight_exporter(self):
        self.weight_exporter = WeightExporter(
            tp_rank=self.ps.tp_rank,
            tp_size=self.ps.tp_size,
            gpu_id=self.gpu_id,
            get_model_path=lambda: self.model_config.model_path,
            get_model=lambda: self.model,
        )

    def init_remote_instance_weight_transporter(self):
        self.remote_instance_weight_transporter = RemoteInstanceWeightTransporter(
            get_model=lambda: self.model,
            tp_rank=self.ps.tp_rank,
            gpu_id=self.gpu_id,
        )

    def init_ngram_embedding_manager(self):
        self.ngram_embedding_manager = NgramEmbeddingManager.from_model(
            model=self.model,
            model_config=self.model_config,
            req_to_token_pool=self.req_to_token_pool,
            max_running_requests=self.max_running_requests,
            device=self.device,
        )

    def init_kv_cache_configurator(self):
        self.kv_cache_configurator = KVCacheConfigurator(
            device=self.device,
            gpu_id=self.gpu_id,
            ps=self.ps,
            pp_group=self.pp_group,
            model=self.model,
            model_config=self.model_config,
            server_args=self.server_args,
            kv_cache_dtype=self.kv_cache_dtype,
            kv_cache_dtype_str=self.kv_cache_dtype_str,
            model_dtype=self.dtype,
            page_size=self.page_size,
            sliding_window_size=self.sliding_window_size,
            spec_algorithm=self.spec_algorithm,
            is_draft_worker=self.is_draft_worker,
            post_capture_kv_active=is_post_capture_kv_active(
                server_args=self.server_args, is_draft_worker=self.is_draft_worker
            ),
            spec_aux_config=self.spec_aux_config,
            is_hybrid_swa=self.is_hybrid_swa,
            is_hybrid_swa_compress=self.is_hybrid_swa_compress,
            use_mla_backend=self.use_mla_backend,
            layer_info=self.layer_info,
            forward_stream=self.forward_stream,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            memory_pool_config=self.memory_pool_config,
            draft_model_idx=self.draft_model_idx,
        )

    def init_mindspore_runner(self):
        # Init the mindspore runner
        # for now, there is only some communication initialization work
        # 中译：初始化 MindSpore 运行器。目前仅做一些通信初始化工作，
        if get_model().model_impl.lower() == ModelImpl.MINDSPORE and _is_npu:
            from sglang.srt.model_executor.mindspore_runner import init_ms_distributed

            init_ms_distributed(
                world_size=self.ps.tp_size * self.ps.pp_size,
                rank=self.ps.tp_size * self.ps.pp_rank + self.ps.tp_rank,
                local_rank=self.gpu_id,
                server_args=self.server_args,
                port=self.dist_port,
            )

    def initialize(self):
        # 中译：核心初始化流程。依次完成：内存节省器创建、专家位置/分布记录初始化、
        # 中译：创建 torch 显存节省器适配器（可在空闲时释放/重建显存以节省占用）。
        # 中译：专家并行（EP）。若开启 EPLB（专家负载均衡）且非草稿 worker，则创建 EPLBManager。
        # 中译：创建采样器、加载模型权重、准备 MoE 的 top-k 路由。
        self.init_memory_saver_adapter()
        self.maybe_init_remote_instance_transfer_engine()
        self.maybe_init_expert_location_metadata()
        self.maybe_init_lplb_solvers()
        self.maybe_init_eplb_manager()
        self.expert_location_updater = ExpertLocationUpdater()
        self.maybe_init_elastic_ep()
        self.init_token_oracle()
        self.sampler = create_sampler()
        self.load_model()
        prepare_moe_topk(
            model=self.model,
            model_config=self.model_config,
            moe_ep_size=self.ps.moe_ep_size,
            moe_ep_rank=self.ps.moe_ep_rank,
        )

        # 中译：加载专家权重备份客户端（仅在弹性 EP + 开启专家备份时创建）。
        self.maybe_init_dwdp()

        # Must run before backend/graph init so no draft graph records a
        # routed-experts capture-write kernel.
        if self.is_draft_worker:
            disable_routed_experts_capture_for_draft(self.model)
        self.maybe_init_expert_backup_client()
        self.remote_instance_weight_transporter.maybe_register_and_publish_weight_info()
        self.layer_info: ModelLayerInfo = resolve_layer_indices(
            model=self.model,
            model_config=self.model_config,
            is_draft_worker=self.is_draft_worker,
            spec_algorithm=self.spec_algorithm,
        )
        adjust_hybrid_swa_layer_ids(
            start_layer=self.layer_info.start_layer,
            end_layer=self.layer_info.end_layer,
            is_hybrid_swa=self.is_hybrid_swa,
        self.maybe_apply_post_load_model_transforms()
        self.maybe_init_lora_manager()
        self.maybe_enable_batch_invariant_mode()
        self.configure_kv_cache_dtype()
    def init_memory_saver_adapter(self):
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=get_exec().features.enable_memory_saver
    def maybe_init_remote_instance_transfer_engine(self):
        if remote_instance_transfer_engine_enabled(load_format=self.draft_load_format):
            self.remote_instance_weight_transporter.init_engine()
    def maybe_init_expert_location_metadata(self):
            return
        expert_rank = self.ps.moe_ep_rank + (
            get_parallel().ep_join_rank_offset if is_ep_scale_joiner() else 0
        set_global_expert_location_metadata(
            compute_initial_expert_location_metadata(
                moe_ep_rank=expert_rank,
        if self.ps.tp_rank == 0 and envs.SGLANG_LOG_EXPERT_LOCATION_METADATA.get():
            logger.info(
                "Initial expert_location_metadata:\n%s",
                format_expert_location_layout(get_global_expert_location_metadata()),
        set_global_expert_distribution_recorder(
            ExpertDistributionRecorder.init_new(
                get_global_expert_location_metadata(),
                rank=expert_rank,
    def maybe_init_lplb_solvers(self):
        if get_exec().moe.ep_dispatch_algorithm == "lp" and not self.is_draft_worker:
            init_lplb_solvers(model_config=self.model_config)
    def maybe_init_eplb_manager(self):
        self.eplb_manager = (
            EPLBManager(
                ps=self.ps,
                get_model=lambda: self.model,
                get_expert_location_updater=lambda: self.expert_location_updater,
                get_expert_backup_client=lambda: self.expert_backup_client,
                get_weight_updater=lambda: self.weight_updater,
            if get_exec().moe.enable_eplb and (not self.is_draft_worker)
            else None
    def maybe_init_elastic_ep(self):
        if get_exec().moe.elastic_ep_backend:
            ElasticEPStateManager.init(self.server_args)
    def init_token_oracle(self):
        # The oracle sampler is process-wide, so a draft would overwrite the
        # target's with its own vocab -- which a DFlash draft does not have.
            self._token_oracle_manager = None
        self._token_oracle_manager = install_token_oracle_from_env(
            vocab_size=self.model_config.vocab_size,
    def maybe_init_expert_backup_client(self):
        self.expert_backup_client = (
            ExpertBackupClient(
                model_config=self.model_config,
                moe_ep_size=self.ps.moe_ep_size,
                moe_ep_rank=self.ps.moe_ep_rank,
                get_model=lambda: self.model,
            )
            if (
                get_exec().moe.enable_elastic_expert_backup
                and get_exec().moe.elastic_ep_backend is not None
            )
            else None
        )

        # 中译：对于 DeepSeek-V3 / GLM-4.5 等带 MTP（Multi-Token Prediction）的模型，MTP 层会被单
        # 中译：start_layer/end_layer 是本 rank 负责的层区间（PP 场景下每个 rank 只拥有部分层），
        # 中译：针对混合 SWA（滑动窗口注意力）模型在 PP 下调整层划分。
        # 中译：对于 LoopCoder 类模型，每轮循环都有独立的 layer_id，故需乘以 loop_num。
        # 中译：应用 torchao 量化配置。
        # 中译：在分层加载模式下 torchao 可能已被应用过，避免重复应用。
        # 中译：若模型支持 torch 原生张量并行（TP）且 tp_size>1，则应用之。
    def maybe_apply_post_load_model_transforms(self):
        supports_torch_tp = getattr(self.model, "supports_torch_tp", False)
        if self.ps.tp_size > 1 and supports_torch_tp:
            self.apply_torch_tp()

        # 中译：初始化 LoRA 管理器（若启用 LoRA）。
    def maybe_init_lora_manager(self):
        # Adapters apply to the target model only; the draft runs unadapted.
        if get_lora().enable_lora and not self.is_draft_worker:
            self.init_lora_manager()

        # 中译：若开启确定性推理，则启用 batch invariant 模式（使结果与 batch 组合无关，
    def maybe_enable_batch_invariant_mode(self):
        if get_exec().deterministic.enable_deterministic_inference:
            from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

            enable_batch_invariant_mode()

        # 中译：推导（确定）KV 缓存的数据类型。
        # 中译：在权重加载阶段结束时快照记录空闲显存。KV 池 profiling 使用该值，
    def get_pp_proxy_topk_size(self) -> Optional[int]:
        return misc_utils.resolve_pp_proxy_topk_size(
            model_config=self.model_config,
            pp_size=self.ps.pp_size,
            pp_rank=self.ps.pp_rank,
            start_layer=self.layer_info.start_layer,
        )

    def get_pp_proxy_residual_num_blocks(self) -> Optional[int]:
        return misc_utils.resolve_pp_proxy_residual_num_blocks(
            model_config=self.model_config,
            pp_size=self.ps.pp_size,
            pp_rank=self.ps.pp_rank,
            start_layer=self.layer_info.start_layer,
        )

    def decode_num_tokens_per_req(
        self, *, num_draft_tokens: Optional[int] = None
    ) -> int:
        """Logits rows per decode batch slot."""
        if self.spec_algorithm.is_speculative():
            if self.spec_algorithm.is_dspark() and self.is_draft_worker:
                from sglang.srt.speculative.dspark_components.dspark_config import (
                    get_dspark_sample_from_anchor,
                )

                if not get_dspark_sample_from_anchor(self.model_config.hf_config):
                    if num_draft_tokens is None:
                        num_draft_tokens = get_spec().speculative_num_draft_tokens
                    return int(num_draft_tokens)
            return resolve_num_tokens_per_req(
                phase="target_verify",
                spec_algorithm=self.spec_algorithm,
                is_draft_worker=self.is_draft_worker,
                num_draft_tokens=num_draft_tokens,
            )
        dllm_config = DllmConfig.from_server_args(self.server_args)
        return dllm_config.block_size if dllm_config is not None else 1

    def max_decode_logits_rows(self) -> int:
        """Rows the shared logits buffer needs."""
        # Resolution can turn speculative_adaptive off, so the effective value
        # lives in the bags while the startup record keeps the CLI input.
        spec = get_spec()
        draft_token_counts = [max_speculative_num_draft_tokens()]
        if spec.speculative_adaptive:
            draft_token_counts.extend(
                steps + 1
                for steps in resolve_candidate_steps_from_config(
                    spec.speculative_adaptive_config
                )
            )

        max_rows = 0
        for draft_tokens in draft_token_counts:
            num_tokens_per_req = self.decode_num_tokens_per_req(
                num_draft_tokens=draft_tokens
            )
            capture_bs, _ = get_batch_sizes_to_capture(self, num_tokens_per_req)
            max_rows = max(max_rows, max(capture_bs) * num_tokens_per_req)
        return max_rows

    @property
    def preloaded_weights_bytes(self) -> int:
        value = self.loader.preloaded_weights_bytes
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "ModelLoader.preloaded_weights_bytes must be a non-negative int, "
                f"got {value!r}"
            )
        return value

    def account_preloaded_weights(self, preloaded_weights_bytes: int) -> None:
        # Dist-init sampled B after the daemon already held weights, so slack
        # (B * (1 - mem_fraction_static)) is too small. Add those bytes back
        # onto the existing MIN'd baseline. Skip when nothing was preloaded.
        if preloaded_weights_bytes == 0:
            return
        self.pre_model_load_memory += preloaded_weights_bytes / (1 << 30)

    def alloc_memory_pool(self, memory_pool_config: Optional[MemoryPoolConfig] = None):
        """Allocate KV cache memory pools only (no backends or cuda graphs).

        中译：仅分配 KV 缓存内存池（不包含注意力后端与 CUDA Graph 的初始化）。
        """
        if memory_pool_config is not None:
            self.memory_pool_config = memory_pool_config

        self.init_kv_cache_configurator()
        result = self.kv_cache_configurator.configure(
            pre_model_load_memory=self.pre_model_load_memory
        )
        self.max_total_num_tokens = result.max_total_num_tokens
        self.max_running_requests = result.max_running_requests
        self.req_to_token_pool = result.req_to_token_pool
        self.token_to_kv_pool = result.token_to_kv_pool
        self.token_to_kv_pool_allocator = result.token_to_kv_pool_allocator
        self.memory_pool_config = result.memory_pool_config
        if self.is_hybrid_swa:
            self.full_max_total_num_tokens = result.full_max_total_num_tokens
            self.swa_max_total_num_tokens = result.swa_max_total_num_tokens
        # Keep a reference so the shared byte buffer is not GC'd.
        self._unified_memory_pool = result.unified_memory_pool

        self._init_post_memory_pool_components()

    def _init_post_memory_pool_components(self):
        """Post-pool component wiring, split out of alloc_memory_pool so forks
        that build bespoke memory pools can reuse it after allocating them."""
        # Must be called AFTER init_memory_pool so the pool object exists for
        # canary to monkey-patch, and BEFORE init_decode_cuda_graph so warmup
        # forwards captured into the graph see the patched pool methods.
        # 中译：必须在 init_memory_pool 之后调用（这样内存池对象已存在，供 canary 做 monkey-patch），
        #       并且必须在 init_decode_cuda_graph 之前（这样被捕获进 graph 的 warmup 前向能看到被
        #       patch 过的内存池方法）。
        self.canary_manager = install_canary(
            server_args=self.server_args,
            model_runner=self,
            token_oracle_manager=self._token_oracle_manager,
        )

        # Init ngram embedding token table
        # 中译：初始化 n-gram embedding 的 token 表（若模型使用了 ngram embedding）。
            # 中译：初始化 hisparse（分层稀疏注意力）协调器，管理 host/device 上的稀疏 KV 缓存。
        self.init_ngram_embedding_manager()

        self.maybe_init_hisparse_coordinator()

        # 中译：初始化路由专家捕获器与 indexer top-k 捕获器（用于调试/状态导出）。
        self.init_routed_experts_capturer()
        self.init_indexer_capturer()

        # 中译：注意力后端与 CUDA Graph runner 的占位字段，稍后在 init_backends() 中真正初始化。
        self.graph_shared_output = None

    def maybe_init_hisparse_coordinator(self):
        if not self.enable_hisparse:
            return
        from sglang.srt.managers.hisparse_coordinator import (
            HiSparseCoordinator,
            resolve_shared_index_layers,
        )
        from sglang.srt.mem_cache.sparsity import parse_hisparse_config

        hisparse_cfg = parse_hisparse_config()
        hisparse_top_k = getattr(
            self.model_config.hf_text_config, "index_topk", hisparse_cfg.top_k
        )
        self.hisparse_coordinator = HiSparseCoordinator(
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            top_k=hisparse_top_k,
            device_buffer_size=hisparse_cfg.device_buffer_size,
            device=self.device,
            tp_group=(
                self.attention_tp_group.cpu_group
                if get_parallel().enable_dp_attention
                else self.tp_group.cpu_group
            ),
            host_to_device_ratio=hisparse_cfg.host_to_device_ratio,
            swap_in_block_size=hisparse_cfg.swap_in_block_size,
            shared_index_layers=resolve_shared_index_layers(
                hf_text_config=self.model_config.hf_text_config,
                pp_size=self.ps.pp_size,
                is_speculative=self.spec_algorithm.is_speculative(),
            ),
        )

    def post_capture_resize_kv_pool(self):
        resize = compute_post_capture_kv_resize(self)
        self.max_total_num_tokens = resize.max_total_num_tokens
        if self.is_hybrid_swa:
            self.full_max_total_num_tokens = resize.full_max_total_num_tokens
            self.swa_max_total_num_tokens = resize.swa_max_total_num_tokens
        if self.memory_pool_config is not None:
            self.memory_pool_config.max_total_num_tokens = resize.max_total_num_tokens
            self.memory_pool_config.full_max_total_num_tokens = (
                resize.full_max_total_num_tokens
            )
            self.memory_pool_config.swa_max_total_num_tokens = (
                resize.swa_max_total_num_tokens
            )
        if resize.capped_max_running_requests is not None:
            self.max_running_requests = resize.capped_max_running_requests
            if self.memory_pool_config is not None:
                self.memory_pool_config.max_running_requests = (
                    resize.capped_max_running_requests
                )

    def post_capture_elastic_ep_recover(self):
        join_process_groups()

        global_ep_rank = self.ps.tp_rank + get_parallel().ep_join_rank_offset
        broadcast_global_expert_location_metadata(
            model_config=self.model_config,
            moe_ep_rank=global_ep_rank,
            src_rank=get_healthy_expert_location_src_rank(
                invoked_in_elastic_ep_rejoin_path=True
            ),
        )
        set_global_expert_distribution_recorder(
            ExpertDistributionRecorder.init_new(
                get_global_expert_location_metadata(),
                rank=global_ep_rank,
            )
        )

        ElasticEPStateManager.instance().reset()

    def init_attention_backends(self):
        """Initialize attention backends only (no cuda graph capture)."""
        # Must be called BEFORE init_decode_cuda_graph() so CUDA graph capture
        # runs with aux hidden state capture enabled.
        # 中译：TODO：未来把设备相关的初始化分支重构进 platform 接口（另开 PR）。
            # 中译：NPU 上对 zbal 混合模式的懒初始化（在开启 cuda graph 时需在 graph 捕获之前完成）。
        configure_aux_hidden_state_capture(
            model=self.model,
            eagle_use_aux_hidden_state=self.spec_aux_config.eagle_use_aux_hidden_state,
            eagle_aux_hidden_state_layer_ids=self.spec_aux_config.eagle_aux_hidden_state_layer_ids,
            dflash_use_aux_hidden_state=self.spec_aux_config.dflash_use_aux_hidden_state,
            dflash_target_layer_ids=self.spec_aux_config.dflash_target_layer_ids,
            is_dspark=self.spec_algorithm.is_dspark(),
        )
        # Resolve before building: backends read the pair off the runner while
        # they construct (the FlashInfer KV-access check).
        resolved = resolve_attention_backend_strs(model_runner=self)
        self.prefill_attention_backend_str = resolved.prefill
        self.decode_attention_backend_str = resolved.decode
        backends = build_attention_backends(model_runner=self)
        self.attn_backend = backends.attn_backend
        self.decode_attn_backend = backends.decode_attn_backend
        self.decode_attn_backend_group = backends.decode_attn_backend_group

        if get_parallel().dcp_enabled and get_parallel().dcp_replicate_q_proj:
            self._prepare_replicated_q_proj()

    def _prepare_replicated_q_proj(self) -> None:
        # --dcp-replicate-q-proj: gather each rank's attn_tp head-shard of
        # q_b_proj / w_kc into full-head buffers once here (pre-capture) so the
        # MLA decode path can skip the per-layer Q all-gather. bf16/fp16 only.
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

        dcp_group = get_parallel().dcp_group
        if dcp_group.world_size <= 1:
            return
        n_prepared = 0
        for m in self.model.modules():
            if not isinstance(m, DeepseekV2AttentionMLA):
                continue
            if m.w_kc is None:
                continue
            qp = m.q_b_proj if m.has_q_b_proj else m.q_proj
            # q-replicate only supports the unquantized bf16/fp16 absorb path;
            # quantized q-proj (packed weights) and non-16-bit w_kc keep the
            # per-layer Q all-gather.
            if (
                m.w_kc.dtype not in (torch.bfloat16, torch.float16)
                or not isinstance(qp.quant_method, UnquantizedLinearMethod)
                or qp.weight.dtype not in (torch.bfloat16, torch.float16)
            ):
                logger.warning(
                    "dcp_replicate_q_proj: skipping quantized q-proj/w_kc "
                    "(bf16/fp16 only); this layer keeps the Q all-gather."
                )
                continue
            m.w_kc_qrep = dcp_group.all_gather(m.w_kc.contiguous(), dim=0)
            m.q_b_proj_qrep_weight = dcp_group.all_gather(
                qp.weight.data.contiguous(), dim=0
            )
            n_prepared += 1
        logger.info(
            "dcp_replicate_q_proj: prepared full-head Q weights for %d MLA layers",
            n_prepared,
        )

            # 中译：根据配置为模型注册前向 hook（用于调试/观测）。
        # 中译：初始化预填阶段的（分段）CUDA Graph runner。
        # 中译：预分配对称内存池（用于 symmetric memory all-reduce 等场景）。
            # 中译：标记 canary 初始化完成。
        # 中译：在流水线并行（PP）下，重新计算本 rank 实际拥有的「全量注意力层」与
        # 中译：初始化路由专家捕获器（用于在返回结果中携带 MoE 路由选中的专家信息）。
    def init_cuda_graphs(self, capture_decode_cuda_graph: bool = True):
        capture = capture_cuda_graphs(
            model_runner=self, capture_decode_cuda_graph=capture_decode_cuda_graph
        )
        self.eager_runner = capture.eager_runner
        self.prefill_cuda_graph_runner = capture.prefill.runner
        self.decode_cuda_graph_runner = capture.decode.runner
        self.graph_memory_usage = capture.memory_usage
        self.graph_time_usage = capture.time_usage

    def init_routed_experts_capturer(self):
        if self.is_draft_worker:
            # Capture is target-only. The draft worker runs in the same process
            # as its target and inits after it, so installing a capturer here
            # would overwrite the target's process-global one.
            return

        set_global_experts_capturer(
            RoutedExpertsCapturer.create(
                model=self.model,
                model_config=self.model_config,
                num_tokens=self.max_token_pool_size + self.page_size,
                max_running_requests=self.max_running_requests,
                device=self.device,
            )
        )

    def init_indexer_capturer(self):
        # 中译：初始化 indexer top-k 捕获器（用于在返回结果中携带 indexer 的 top-k 信息）。
        # 中译：生产端的接线仅支持 CUDA（Indexer.forward_cuda + MLA skip_topk 路径）；
        set_global_indexer_capturer(
            create_indexer_capturer(
                model_config=self.model_config,
                num_tokens=self.max_token_pool_size + self.page_size,
                max_running_requests=self.max_running_requests,
                device=self.device,
            )
        )

        # 中译：为“远程实例权重加载”初始化 Mooncake 传输引擎（需安装 mooncake）。
        # 中译：针对具体模型特性调整 server_args（例如：不支持分块预填的多模态模型需关闭
        # 中译：检查量化 MoE 的参数可整除性约束：tp_size 必须能被 ep_size 整除，
        # 中译：初始化 torch 分布式环境。依次完成：绑定设备、选择通信后端、
            # 中译：若未开启 P2P 检查，则 monkey-patch 掉 P2P 访问检查（避免额外开销）。
        # 中译：允许外部编排系统（如 trainpi）覆盖分布式初始化方法。当设为 "env://" 时，
                    # 中译：将 OpenMP 线程绑定到 CPU 核。
                    # 中译：设置 LOCAL_SIZE 以提示 SGLang 使用基于共享内存的 AllReduce。
            # 中译：仅在 target（主模型）worker 上初始化分布式环境（草稿 worker 复用主模型的）。
            # 中译：预热 NCCL/RCCL，以消除首个请求的冷启动延迟。由 --pre-warm-nccl 控制
                # 中译：通过一次 all_reduce 初始化（预热）NCCL/RCCL 通信器。
        # 中译：检查张量并行下各卡显存是否均衡。若某卡可用显存明显偏低（被其他进程占用），
    def check_quantized_moe_compatibility(self):
        check_quantized_moe_compatibility(
            model_config=self.model_config,
            tp_size=self.ps.tp_size,
            moe_ep_size=self.ps.moe_ep_size,
            moe_dp_size=self.ps.moe_dp_size,
        )

    def init_torch_distributed(self):
        result = bootstrap.init_torch_distributed(
            server_args=self.server_args,
            device=self.device,
            ps=self.ps,
            dist_port=self.dist_port,
            is_draft_worker=self.is_draft_worker,
            local_omp_cpuid=self.local_omp_cpuid if self.device == "cpu" else None,
        self.tp_group = result.tp_group
        self.pp_group = result.pp_group
        self.attention_tp_group = result.attention_tp_group
        self.pre_model_load_memory = result.pre_model_load_memory
    def init_shared_mooncake_transfer_engine(self):
        maybe_init_shared_mooncake_transfer_engine(gpu_id=self.gpu_id)

    def load_model(self):
        # 中译：加载模型权重。包含：设备能力检查与 dtype 回退、准备模型配置、
        #       调用对应 loader 加载权重等。
        tic_total = time.perf_counter()
        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Load weight begin. avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        # This can reduce thread conflicts and speed up weight loading.
        # 中译：将线程数设为 1 可减少线程冲突、加速权重加载。
        if self.device != "cpu":
            torch.set_num_threads(1)
        if self.device == "cuda":
                # 中译：计算能力低于 sm80，由于缺乏 bfloat16 支持，改用 float16。
        # 中译：准备模型配置（包括 ModelOpt 量化配置与加载配置 LoadConfig）。
            maybe_downgrade_dtype_for_legacy_gpu(model_config=self.model_config)

        set_cuda_arch()
        draft_load_format = self.draft_load_format
        self.load_config = build_load_config(
            server_args=self.server_args,
            load_format=draft_load_format,
            tp_rank=self.ps.tp_rank,
            remote_instance_weight_transporter_engine=self.remote_instance_weight_transporter.engine,
            remote_instance_weight_transporter_session_id=self.remote_instance_weight_transporter.session_id,
            draft_model_idx=self.draft_model_idx,
            weight_cache_mode=get_model().weight_cache_mode,
            weight_cache_socket=get_model().weight_cache_socket,
        )

        # If the weight cache is enabled, override the load format to IPC_CACHE
        # and derive the per-rank daemon socket. Idempotent across reloads.
        maybe_enable_ipc_weight_cache(
            load_config=self.load_config,
            tp_size=self.ps.tp_size,
            pp_rank=self.ps.pp_rank,
            tp_rank=self.ps.tp_rank,
        )
        if self.device == "cpu":
            self.model_config = adjust_config_with_unaligned_cpu_tp(
                self.model_config, self.load_config, self.ps.tp_size
            )

        # 中译：加载模型。这里临时 monkey-patch vllm 的并行状态（待 linear.py 的量化不再
        maybe_trigger_remote_instance_nccl_send_group(
            tp_rank=self.ps.tp_rank,
            load_format=draft_load_format,
        )

        with self._load_format_scope(draft_load_format):
            loaded = load_model_with_memory_saver(
                server_args=self.server_args,
                model_config=self.model_config,
                load_config=self.load_config,
                device=self.device,
                gpu_id=self.gpu_id,
                memory_saver_adapter=self.memory_saver_adapter,
                is_draft_worker=self.is_draft_worker,
            )
        self.loader = loaded.loader
        self.model = loaded.model
        self.startup_weight_load = loaded.startup_weight_load
        if loaded.remote_instance_weight_info is not None:
            self.remote_instance_weight_transporter.weight_info = (
                loaded.remote_instance_weight_info
            )
        # 中译：加载权重后需清理缓存。为避免与 memory_saver_adapter.region 冲突，

        if not self.is_draft_worker:
            # 中译：非草稿 worker 在加载完后执行 offloader 的后置初始化。
            get_offloader().post_init()

        self.maybe_precompile_model_kernels_after_loading()

        # Register model for layerwise NVTX profiling if enabled
        # 中译：若启用逐层 NVTX profiling，为模型注册相应 hook。
        if get_exec().comm.enable_layerwise_nvtx_marker:
            pyt_hooks = PytHooks()
            pyt_hooks.register_hooks(self.model, module_prefix="model")

        # Same leaf `configure_kv_cache_dtype` reads: the bag, not the startup
        # record, so the FP8 gate and the pool cannot disagree after an
        # override. (The runner's own stamp is not set yet -- load_model runs
        # before configure_kv_cache_dtype.)
        load_kv_cache_scales(
            model=self.model,
            kv_cache_dtype=get_model().kv_cache_dtype,
        )

            # 中译：模型配置中的滑动窗口字段对不同类型模型含义可能不同（如 dllm），
        self.sliding_window_size = resolve_sliding_window_size(
            self.model, self.model_config
        )

        self.prefill_aware_swa = (
            hasattr(self.model, "is_prefill_aware_swa")
            and self.model.is_prefill_aware_swa()

        self.dtype = self.model_config.dtype

        after_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        self.weight_load_mem_usage = before_avail_memory - after_avail_memory
        self.weight_load_time = time.perf_counter() - tic_total
        # Get quantization config from ModelConfig
        # This handles both config.json (standard) and hf_quant_config.json (ModelOpt)
        # 中译：从 ModelConfig 获取量化配置（同时兼容标准的 config.json 与 ModelOpt 的
        #       hf_quant_config.json），仅用于日志展示。
        quant_str = self.model_config.get_quantization_config_log_str()

        # 中译：TODO：确保所有模型都有 `quant_config` 属性，且所有在线量化方法都注册它们
        if self.startup_weight_load is None:
            logger.info(
                f"Load weight end. "
                f"elapsed={self.weight_load_time:.2f} s, "
                f"type={type(self.model).__name__}, "
                f"{quant_str + ', ' if quant_str else ''}"
                f"avail mem={after_avail_memory:.2f} GB, "
                f"mem usage={self.weight_load_mem_usage:.2f} GB."
            )

        report_online_quantization(
            model=self.model,
        )

        maybe_register_debug_tensor_dump_hook(
            model=self.model,
            spec_algorithm=self.spec_algorithm,
            is_draft_worker=self.is_draft_worker,
            tp_size=self.ps.tp_size,
            tp_rank=self.ps.tp_rank,
            pp_rank=self.ps.pp_rank,
        )

        if dumper.may_enable:
            dumper.apply_source_patches()
            dumper.register_non_intrusive_dumper(self.model)

        # Pre-expand RoPE cache before CUDA Graph capture
        # 中译：在 CUDA Graph 捕获之前预先扩展 RoPE 缓存（以支持长序列）。
        reserve_rope_cache_for_long_sequences(
            self.model,
            self.model_config,
            logger,
        )

            # 中译：Mooncake 不支持 `monitored_barrier`，改用普通 barrier。
            # 中译：处理部分 rank 未完成加载的情况（用带超时的 monitored_barrier 捕捉掉队者）。
        # 中译：为启用了 DeepEP waterfill 的 MoE TopK 模块准备负载均衡器（balancer）。
            # 中译：静态 EPLB 会在 Waterfill 之前将 TopK id 重映射为物理专家 id，因此凗余
        # 中译：更新（重布）专家位置。若某些逻辑专家无法通过 P2P 获得（p2p_missing），
        if self.startup_weight_load is None:
            dist_barrier_after_load(
                elastic_ep_backend=get_exec().moe.elastic_ep_backend,
                tp_rank=self.ps.tp_rank,
                is_ep_joiner=self.server_args.is_ep_joiner,
            )

    def start_startup_weight_load(self) -> None:
        assert self.startup_weight_load is not None
        self.startup_weight_load.start_prefetch()
    def finalize_startup_weight_load(self) -> None:
        """Commit the real weights, then run the post-load barrier.
        The barrier moves here because ``load_model`` returns with sentinel
        values under overlap, so this is the first point at which "weights are
        loaded" is true for this rank. It follows the commit and its validation
        deliberately: a rank that fails to commit must not report readiness. A
        commit failure is terminal for the process, so peer ranks observe it as
        a barrier timeout rather than a clean collective abort, which matches
        the existing startup contract for load failures.
        """
        self.startup_weight_load.finalize()
            is_ep_joiner=is_ep_joiner(),
        )
        self.startup_weight_load = None

            # 中译：从磁盘加载缺失的专家权重。
                # 中译：仅过滤并加载缺失的专家权重。
                # 中译：模型未实现 generate_weight_name_filter，退而从磁盘/DRAM 做一次全量重载。
        # 中译：（弹性 EP）尝试恢复失效的 rank。检查是否有不活跃的 rank，若有则尝试恢复，
        # 中译：TODO(性能)：在 CUDA 张量上调用 `active_ranks.all()` 会触发 host-device 同步，
    def maybe_precompile_model_kernels_after_loading(self) -> None:
        maybe_precompile_model_kernels_after_loading(self.model, self.device)

    def maybe_init_dwdp(self):
        if self.is_draft_worker:
            return
        if get_parallel().dwdp_size <= 1:
            return
        from sglang.srt.layers.moe.dwdp import DwdpManager

        # 中译：`ranks_to_recover` 使用的是 tp_group 内的索引。当前 Mooncake 弹性 EP 实现
        # 中译：try_recover_ranks 通过 Mooncake EP 后端轮询对端状态。Mooncake 的内部语义保证
        # 中译：选出一个“健康”的 rank 作为广播专家位置元数据的源 rank。
        # 中译：不要以 `self.server_args.elastic_ep_rejoin` 作为依据。一个以 rejoin 身份启动的
        # 中译：目前仅支持 DefaultModelLoader。
            # 中译：若需要（且设备支持），重新捕获解码阶段的 CUDA Graph（因为权重已变）。
        # 中译：为“向远程实例发送权重”初始化一个自定义进程组（每个 tp_rank 一个端口）。
        # 中译：通过之前初始化的发送组，将本模型所有参数 broadcast 给远程实例；
            # 中译：发送组不存在，提示需先调用 init_weights_send_group_for_remote_instance。
        # 中译：发送权重完成后销毁该进程组。
        # 中译：销毁指定的权重更新进程组。
        # 中译：以“扁平化分桶（flattened bucket）”格式从分布式组更新权重：先把多个张量拼成
        # 中译：直接从（可能跨进程序列化的）张量更新权重。根据 load_format 选择不同加载
            # 中译：处理扁平化分桶格式。
        # 中译：必须在 patch 之后获取设备，否则设备会不正确。
        # 中译：TODO：（chenyang）增加对 Qwen 系列模型的支持。
        manager = DwdpManager(self.server_args)
        set_global_dwdp_manager(manager)
        manager.setup(self.model)

    def init_lora_manager(self):
        # 中译：初始化 LoRA 管理器（管理多个 LoRA 适配器的加载/卸载与批内路由）。
        self.lora_manager = LoRAManager(
            base_model=self.model,
            base_hf_config=self.model_config.hf_config,
            max_loras_per_batch=get_lora().max_loras_per_batch,
            load_config=self.load_config,
            dtype=self.dtype,
            server_args=self.server_args,
            lora_backend=get_lora().lora_backend,
            tp_size=self.ps.tp_size,
            tp_rank=self.ps.tp_rank,
            max_lora_rank=get_lora().max_lora_rank,
            target_modules=get_lora().lora_target_modules,
            lora_paths=get_lora().lora_paths,
        )
        if not cuda_graph_fully_disabled():
            init_lora_cuda_graph_moe_buffers(
                model=self.model,
                lora_manager=self.lora_manager,
                dtype=self.dtype,
            )

    def load_lora_adapter(self, lora_ref: LoRARef):
        """Load a new lora adapter from disk or huggingface."""
        return self.lora_manager.load_lora_adapter(lora_ref)

    def load_lora_adapter_from_tensors(
        self, lora_ref: LoRARef, tensors, config_dict, added_tokens_config=None
    ):
        return self.lora_manager.load_lora_adapter_from_tensors(
            lora_ref, tensors, config_dict, added_tokens_config
        )

    def unload_lora_adapter(self, lora_ref: LoRARef):
            # 中译：NemotronH 的 MTP 草稿模型没有 Mamba 层（pattern 形如 "*E"），故不应使用
        """Unload a lora adapter that was previously loaded during initialization or dynamic loading."""
        return self.lora_manager.unload_lora_adapter(lora_ref)

    @property
    def effective_max_total_num_tokens(self):
        """Return the max token pool size considering hybrid swa settings."""
        if self.is_hybrid_swa:
            capacity = self.full_max_total_num_tokens or self.swa_max_total_num_tokens
        else:
            capacity = self.max_total_num_tokens
        if (req_to_token_pool := getattr(self, "req_to_token_pool", None)) is not None:
            return req_to_token_pool.schedulable_token_capacity(capacity)
        return capacity
    def max_token_pool_size(self):
        """Return the max token pool size considering hybrid swa and hisparse settings."""
        if self.enable_hisparse:
            # HiSparse uses the host-backed full pool capacity.
            size_full = getattr(self.token_to_kv_pool_allocator, "size_full", None)
            if size_full is not None:
                return size_full
        return self.effective_max_total_num_tokens

    def _load_format_scope(self, load_format: Optional[str]):
        """Make this runner's load format the published one while it loads.

        Model code reads it off the bag during construction (Inkling replaces
        per-element noise in its shared-expert scales under dummy loading), so a
        draft loading a different way than the target needs its own value live
        for the load, and the target's back afterwards.
        """
        if load_format is None:
            return contextlib.nullcontext()
        return get_model().override(load_format=load_format)

    def _resolve_draft_load_format(self) -> Optional[str]:
        """``--speculative-draft-load-format``, for a draft runner only.

        The draft loads its own checkpoint, so its load format is this runner's
        own resolved value; the target keeps ``--load-format``.
        """
        if not self.is_draft_worker:
            return None
        load_format = get_spec().speculative_draft_load_format
        if load_format is not None:
            logger.info(f"Using draft model load_format: '{load_format}'")
        return load_format

    def configure_kv_cache_dtype(self):
        # 中译：根据 --kv-cache-dtype 与模型量化配置，确定 KV 缓存的实际数据类型
        spec_algorithm = getattr(self, "spec_algorithm", None)
        resolved_kv_cache_dtype, self.kv_cache_dtype = (
            kv_cache_dtype.configure_kv_cache_dtype(
                server_args_kv_cache_dtype=get_model().kv_cache_dtype,
                model=getattr(self, "model", None),
                model_dtype=getattr(self, "dtype", torch.bfloat16),
                is_draft_worker=getattr(self, "is_draft_worker", False),
                is_dflash=(
                    spec_algorithm.is_dflash_family()
                    if spec_algorithm is not None
                    else False
                ),
                speculative_draft_attention_backend=self.draft_attention_backend,
                speculative_draft_kv_cache_dtype=get_spec().speculative_draft_kv_cache_dtype,
            )
        # This runner's OWN resolved dtype string (target or draft). Attention
        # backends read it directly instead of the process-global get_model()
        # bag: a draft runner does not publish its args, so the bag would carry
        # the target's dtype and mis-drive the draft's FP8 cast/descale paths.
        self.kv_cache_dtype_str = (
            resolved_kv_cache_dtype
            if resolved_kv_cache_dtype is not None
            else get_model().kv_cache_dtype

    def _get_attention_backend(self, init_new_workspace: bool = False):
        return get_attention_backend(
            model_runner=self, init_new_workspace=init_new_workspace

    def _decode_cuda_graph_runner_cls(self):
        """Decode CUDA-graph runner class to construct.

            # 中译：prefill 与 decode 后端不同，使用 HybridAttnBackend 将两者组合。
        # 中译：根据后端名称字符串，从注册表 ATTENTION_BACKENDS 中构造对应的注意力后端。
        Subclasses can override this to install specialized decode graph runners.
        """
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        # 中译：CuteDSL v1（cutedsl runner + deepep a2a）绕过了 MoeRunner，不能被 autotune——
        # 中译：TODO(smor)：支持 flashinfer autotune 的其他场景，例如 mamba 后端。
        # 中译：在非默认 stream 上运行预热，以避免使用 --enable-symm-mem 时 NCCL 2.29+ 的
        # 中译：根据模型/量化/并行等关键信息生成 autotune 缓存文件路径（按 sm 架构、flashinfer
        # 中译：使预热与调度器的 MLP-sync 填充对齐（否则预热形状会与实际运行不一致）。
        # 中译：辅助隐藏状态捕获（eagle3/dflash）已由 initialize() 中的
        # 中译：EXTEND（预填/拓展）模式下需构造 extend 相关的长度/起始位置张量。
            # 中译：PP 第一段（PP0）在发送前已对 hidden_states 做了 cp 切分。
            # 中译：MTP 模型（如 deepseek_nextn）在前向时会读取 spec_info.hidden_states，
        # 中译：若模型使用 n-gram embedding，则初始化其 token 表与各模块的缓冲区。
            # 中译：尺寸与 req_to_token 一致（按 req_pool_idx 索引）。
            # 中译：TODO：目前 CUDA Graph 仅捕获解码步骤，而解码步骤只存在于生成类模型。
        return DecodeCudaGraphRunner

    def init_decode_cuda_graph(self):
        self.decode_cuda_graph_runner = None
        capture = capture_decode_graph(model_runner=self)
        self.decode_cuda_graph_runner = capture.runner
        self.graph_memory_usage = replace_graph_memory_usage(
            self.graph_memory_usage,
            capture.memory_usage,
            phases=("decode", "target_verify", "draft_decode"),
        )
        self.graph_time_usage = replace_graph_time_usage(
            self.graph_time_usage,
            capture.time_usage,
            phases=("decode", "target_verify", "draft_decode"),
        )

    def init_prefill_cuda_graph(self, force_for_draft_worker: bool = False):
        # 中译：草稿模型在 __init__ 阶段跳过这里；eagle worker 会在 init_lm_head 之后显式
        # 中译：从模型中收集注意力层与 MoE 层（供分段图捕获使用）。
        # 中译：解析出拥有 layers 的模型：兼容 CausalLM 包装（.model.layers）与直接的
            # 中译：TODO(yuwei)：支持非标准 GQA。若有部分层不是标准 GQA，则禁用分段 CUDA Graph。
        self.prefill_cuda_graph_runner = None
        capture = capture_prefill_graph(
            model_runner=self,
            eager_runner=self.eager_runner,
            force_for_draft_worker=force_for_draft_worker,
        )
        self.prefill_cuda_graph_runner = capture.runner
        self.graph_memory_usage = replace_graph_memory_usage(
            self.graph_memory_usage,
            capture.memory_usage,
            phases=("prefill", "draft_prefill"),
        )
        self.graph_time_usage = replace_graph_time_usage(
            self.graph_time_usage,
            capture.time_usage,
            phases=("prefill", "draft_prefill"),
        )

    def init_threads_binding(self):
        # 中译：初始化 CPU 的 OpenMP 线程绑定（绑核）。根据 SGLANG_CPU_OMP_THREADS_BIND 与
        # 中译：应用 torch 原生张量并行（通过 device_mesh 将模型切分到多个设备上）。
        # With --enable-dp-attention, dp partitions the existing TP group
        # rather than spawning additional processes, so dp_size must not be
        # multiplied into the process count here (unlike regular DP, where
        # dp_size * tp_size * pp_size is the true worker count).
        dp_size = 1 if get_parallel().enable_dp_attention else self.ps.dp_size
        self.local_omp_cpuid = numa_utils.init_threads_binding(
            numa_index=self.gpu_id,
            world_size=dp_size * self.ps.tp_size * self.ps.pp_size,
        )

    def apply_torch_tp(self):
        model_parallel.apply_torch_tp(
            model=self.model, device=self.device, tp_size=self.ps.tp_size

    def update_decode_attn_backend(self, stream_idx: int):
        # 中译：（PDMux）按 stream 索引切换当前使用的解码注意力后端。
        self.decode_attn_backend = self.decode_attn_backend_group[stream_idx]

        # 中译：首次使用时构建；当某个 batch 超过当前容量时，按“下一个 2 的幂”扩容。
    def prepare_dummy_forward_batch(self, forward_batch: ForwardBatch) -> ForwardBatch:
        """Customize a runner-created dummy batch before attention metadata initialization."""
        return forward_batch

    def _prepare_eager_forward_batch(self, forward_batch: ForwardBatch) -> None:
        """Pad / normalize a batch for the eager (non-cuda-graph) forward.
        Runs the DP/MLP-sync padding, the attn-tp num_token_non_padded
        normalization, and the hisparse-coordinator refresh that the eager
        forward path needs — the cuda-graph path does the equivalent inside the
        runner's capture/replay, so this is skipped there.
        """
        # For MLP sync
        if forward_batch.global_num_tokens_cpu is not None:
            forward_batch.prepare_mlp_sync_batch(self)
        else:
            forward_batch.prepare_attn_tp_scatter_input(self)
        # Normalize num_token_non_padded to be local to this attention TP rank if needed.
        # The skip is scoped to DSACPLayerCommunicator-style CP (DSA, MLA): those
        # flavors already feed a zigzag-split rank-local layout whose token count
        # should not be further divided by attn_tp_size. MHA-arch prefill CP
        # (Qwen3/Qwen2 MoE) keeps the attn_tp-replicated layout and wants the
        # adjustment to run — see docs/design/prefill-cp-mla.md §Phase 5.
        if (
            forward_batch.num_token_non_padded is not None
            and forward_batch.global_num_tokens_gpu is not None
            and require_gathered_buffer()
            and not is_dsa_enable_prefill_cp()
            and not is_mla_prefill_cp_enabled()
        ):
            forward_batch.adjust_num_token_non_padded_for_attn_tp()

        # Hisparse coordinator — backends now read it from self.model_runner.
        if self.hisparse_coordinator is not None:
            self.hisparse_coordinator.num_real_reqs.fill_(forward_batch.batch_size)

    def _pp_kwargs(self, pp_proxy_tensors) -> dict:
        """Build the pp_proxy_tensors forward kwarg, in one place.

        # 中译：eager（非 CUDA Graph）执行路径下，将 forward_batch 的输入填入可复用的 eager 缓冲区
        # 中译：执行一次解码（decode）前向（eager 路径，未命中 CUDA Graph 时）。
        # 中译：设置额外参数，并初始化注意力元数据。
                # 中译：PDMux 选择一个按 stream 区分的后端；通过当前 ForwardContext 发布给模型层的
        # 中译：FIXME：为所有模型都加上 pp_proxy_tensors 参数。
        # 中译：启动前向（可选包裹设备计时器）。
        # 中译：执行一次 EXTEND（预填/拓展）前向。返回 (输出, 是否命中分段 CUDA Graph)。
        # 中译：准备额外参数（PP 代理张量、input_embeds、embedding 覆盖等）。
        Pipeline-parallel proxy tensors are threaded into model.forward only
        when the model accepts them (``support_pp``).
        """
        return {"pp_proxy_tensors": pp_proxy_tensors} if self.support_pp else {}

    def _extend_forward_kwargs(
        self, forward_batch: ForwardBatch, pp_proxy_tensors
    ) -> dict:
        """Build the extend/prefill model.forward kwargs (pp_proxy_tensors +
        input_embeds / replace_embeds overrides + get_embedding), shared by the
        prefill cuda-graph path and the EagerRunner's eager extend path."""
        kwargs = self._pp_kwargs(pp_proxy_tensors)
        if forward_batch.input_embeds is not None:
            kwargs["input_embeds"] = forward_batch.input_embeds.bfloat16()
        if (
            forward_batch.replace_embeds is not None
            and forward_batch.replace_positions is not None
        ):
            # Token embedding overrides: get base embeddings, scatter replacements
            # 中译：token embedding 覆盖：先取基础 embedding，再将指定位置的 embedding 替换为给定值。
            if "input_embeds" not in kwargs:
                embed_layer = self.model.get_input_embeddings()
                kwargs["input_embeds"] = embed_layer(forward_batch.input_ids)
            kwargs["input_embeds"][forward_batch.replace_positions] = (
                forward_batch.replace_embeds.to(kwargs["input_embeds"].dtype)
            )
        if not self.is_generation:
            kwargs["get_embedding"] = True
        # 中译：检查是否可以走分段（piecewise）CUDA Graph（需 runner 存在且本 batch 可由其执行）。
            # 中译：TODO：这里 device_timer.wrap 范围过宽——它还包含了 replay_prepare 的时间。
                # 中译：AMD/HIP：当开启 PCG（分段 CUDA Graph）但 batch 超过最大捕获尺寸时，在
        return kwargs

    def forward_split_prefill(
        self,
        forward_batch: ForwardBatch,
        reinit_attn_backend: bool = False,
        forward_count: int = 1,
    ) -> LogitsProcessorOutput:
        # 中译：按层分段（split）执行预填前向：每次只跑 forward_count 层，通过 split_index 记录
        #       进度，用于分段预填调度。
        if forward_batch.split_index == 0 or reinit_attn_backend:
            self.attn_backend.init_forward_metadata(forward_batch)
        next_split_index = min(
            forward_batch.split_index + forward_count,
            self.model_config.num_hidden_layers,
        )
        with device_timer_ctx(self.device_timer, "split_prefill"):
            ret = self.model.forward_split_prefill(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                (forward_batch.split_index, next_split_index),
            )
        forward_batch.split_index = next_split_index
        return ret

    def forward(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: Optional[bool] = None,  # deprecated
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> ModelRunnerOutput:
        # 中译：ModelRunner 的前向总入口。负责：递增前向计数、开启调试器/profiling/canary/
        #       专家分布记录等上下文，调用 _forward_raw 执行实际前向，并在结束后处理弹性 EP 重平衡、
        #       状态捕获输出、EPLB 回调与调试器收尾。
        # Deprecated kwarg: pre-planners mark the batch themselves now.
        # 中译：已废弃参数：现在由预规划器（pre-planner）自行标记 batch。
        forward_batch.apply_deprecated_skip_attn_backend_init(skip_attn_backend_init)

        self.forward_pass_id += 1

        # 中译：尝试启动 msprobe 精度调试器。

        # Try msprob debugger
        if self.msprobe_debugger is not None:
            rank_id = (
                self.gpu_id
                if self.ps.attn_dp_size is not None and self.ps.attn_dp_size > 1
                else None
            )
            self.msprobe_debugger.start(model=self.model, rank_id=rank_id)

        # Step span
        # 中译：为本步创建 profiling 的跨度（span）。
        step_span_ctx = profile_range(build_step_span_name(forward_batch))

        canary_ctx = (
            context_tuple(
                c.with_ops_outside_graph(
                    single_forward_indices=[0],
                    maybe_inaccurate_forward_batch=forward_batch,
                ),
                c.with_active_single_forward_manager(0),
            )
            if not self.is_draft_worker and ((c := self.canary_manager) is not None)
            else contextlib.nullcontext()
        )

        with (
            canary_ctx,
            step_span_ctx,
            get_global_expert_distribution_recorder().with_forward_pass(
                self.forward_pass_id,
                forward_batch,
            ) as recorder_outputs,
        ):
            output = self._forward_raw(
                forward_batch,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
            if self.enable_elastic_ep:
                output = self._maybe_rebalance_after_rank_fault(
                    output,
                    forward_batch,
                    pp_proxy_tensors,
                    reinit_attn_backend,
                    split_forward_count,
                )
        output.expert_distribution_metrics = recorder_outputs.get("metrics")

        no_copy_to_cpu = not get_schedule().disable_overlap_schedule
        if (
            not self.is_draft_worker
            and (experts_capturer := get_global_experts_capturer()) is not None
        ):
            output.routed_experts_output = experts_capturer.on_forward_end(
                forward_batch=forward_batch,
                can_run_graph=output.can_run_graph,
                cuda_graph_batch=getattr(self.decode_cuda_graph_runner, "bs", None),
                no_copy_to_cpu=no_copy_to_cpu,
            )

        if (indexer_capturer := get_global_indexer_capturer()) is not None:
            output.indexer_topk_output = indexer_capturer.on_forward_end(
                forward_batch=forward_batch,
                can_run_graph=output.can_run_graph,
                cuda_graph_batch=getattr(self.decode_cuda_graph_runner, "bs", None),
                no_copy_to_cpu=no_copy_to_cpu,
            )

        if self.eplb_manager is not None:
            self.eplb_manager.on_forward_pass_end()

        if dumper.may_enable:
            dumper.step()

        if self.msprobe_debugger is not None:
            self.msprobe_debugger.stop()
            self.msprobe_debugger.step()

        if get_exec().moe.elastic_ep_backend is not None:
            self.maybe_join_ep_ranks()

        return output

    def _maybe_execute_deferred_mamba_cow_and_clear(
        self, forward_batch: ForwardBatch
    ) -> None:
        """Run deferred clear/COW on the forward stream, before the mamba layers
        read the pool, so the copies don't race the scheduler copy stream.

        No-op unless this is an extend forward on a mamba model's target worker;
        COW/clear only happen at prefix match on extend.
        """
        pool = self.req_to_token_pool
        if (
            not isinstance(pool, HybridReqToTokenPool)
            or self.is_draft_worker
            or not forward_batch.forward_mode.is_extend()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            return
        if (
            forward_batch.mamba_clear_indices is not None
            and len(forward_batch.mamba_clear_indices) > 0
        ):
            # mamba_pool is a pure PHYSICAL store; translate before zeroing or
            # clear_slots zeroes the wrong physical slots.
            pool.mamba_pool.clear_slots(
                pool.translate_mamba_indices(forward_batch.mamba_clear_indices)
            )
        if (
            forward_batch.mamba_cow_src_indices is not None
            and len(forward_batch.mamba_cow_src_indices) > 0
        ):
            if pool.mamba_ckpt_pool is not None:
                # int8 checkpoints: dequantize src int8 ckpt slot into the active bf16 dst.
                pool.mamba_ckpt_pool.load_to_active(
                    pool.mamba_pool,
                    forward_batch.mamba_cow_src_indices,
                    forward_batch.mamba_cow_dst_indices,
                )
            else:
                # mamba_pool is a pure PHYSICAL store; translate both COW slot ids.
                pool.mamba_pool.copy_from(
                    pool.translate_mamba_indices(forward_batch.mamba_cow_src_indices),
                    pool.translate_mamba_indices(forward_batch.mamba_cow_dst_indices),
                )
        forward_batch.mamba_clear_indices = None
        forward_batch.mamba_cow_src_indices = None
        forward_batch.mamba_cow_dst_indices = None

    def _forward_raw(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors],
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> ModelRunnerOutput:
        # 中译：实际的前向分发逻辑。优先尝试命中 CUDA Graph；否则根据 forward_mode
        #       （decode/split_prefill/extend/idle）分别调用对应的 forward_* 方法。
        if has_forward_context():
            ctx_mgr = contextlib.nullcontext()
        else:
            ctx_mgr = forward_context(ForwardContext(attn_backend=self.attn_backend))
        with ctx_mgr:
            mode_check = (
                forward_batch.forward_mode.is_cpu_graph
                if self.device == "cpu"
                else forward_batch.forward_mode.is_cuda_graph
            )
            can_run_graph = bool(
                mode_check()
                and self.decode_cuda_graph_runner
                and self.decode_cuda_graph_runner.can_run_graph(forward_batch)
            )

            if (
                forward_batch.forward_mode.is_decode()
                and self.hisparse_coordinator is not None
            ):
                forward_batch.hisparse_coordinator = self.hisparse_coordinator
                self.hisparse_coordinator.wait_for_pending_backup()
                self.hisparse_coordinator.num_real_reqs.fill_(forward_batch.batch_size)

            # Replay cuda graph if applicable
            # 中译：若可行，直接 replay 已捕获的解码 CUDA Graph（最快路径）。
            if can_run_graph:
                ret = self.decode_cuda_graph_runner.execute(
                    forward_batch,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
                return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)

            # 中译：用于 MLP 同步（DP attention 场景下各 DP rank 的 token 数需对齐）。
            # DP / MLP-sync padding + attn-tp normalization. Only the decode
            # cuda-graph path above pre-pads its static buffers and returns
            # early; split prefill, the prefill cuda graph, and the eager
            # forward all run the live batch and need this first — it sets
            # global_dp_buffer_len / padded token counts that graph eligibility
            # and the collectives depend on.
            self._prepare_eager_forward_batch(forward_batch)

            # Deferred mamba COW/clear on the forward stream, before the extend
            # dispatch below reads the pool.
            self._maybe_execute_deferred_mamba_cow_and_clear(forward_batch)

            dwdp_mgr = get_global_dwdp_manager()
            if dwdp_mgr is not None:
                dwdp_mgr.prefetch_first_layers()

            # 中译：未命中 CUDA Graph 时，按 forward_mode 分发到各 eager 前向方法。
            if forward_batch.forward_mode.is_split_prefill():
                # Layer-split mode; stays on ModelRunner, not the eager runner.
                ret = self.forward_split_prefill(
                    forward_batch,
                    reinit_attn_backend=reinit_attn_backend,
                    forward_count=split_forward_count,
                )
            elif (
                forward_batch.forward_mode.is_extend(include_draft_extend_v2=True)
                and not isinstance(self.prefill_cuda_graph_runner, EagerRunner)
                and self.prefill_cuda_graph_runner is not None
                and self.prefill_cuda_graph_runner.can_run_graph(forward_batch)
                and _prefill_cuda_graph_allows_context_parallel(
                    self.prefill_cuda_graph_runner, forward_batch
                )
            ):
                # Prefill cuda graph (piecewise).
                kwargs = self._extend_forward_kwargs(forward_batch, pp_proxy_tensors)
                category = (
                    "target_verify"
                    if forward_batch.forward_mode.is_target_verify()
                    else "extend"
                )
                # TODO: the timing here is too broad -- it also includes
                # load_batch time. Move it into the prefill cuda graph runner
                # to capture only the model.forward part.
                with device_timer_ctx(self.device_timer, category):
                    ret = self.prefill_cuda_graph_runner.execute(
                        forward_batch, **kwargs
                    )
                can_run_graph = True
            else:
                # Eager: decode / extend / idle dispatched inside the runner.
                ret = self.eager_runner.execute(
                    forward_batch, pp_proxy_tensors=pp_proxy_tensors
                )

            if (
                forward_batch.global_num_tokens_cpu is not None
                and self.pp_group.is_last_rank
            ):
                forward_batch.post_forward_mlp_sync_batch(ret)

            return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)

    def _preprocess_logits(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        observer: Optional[SamplingObserver] = None,
    ):
        # 中译：采样前的 logits 预处理：更新正则/词表 mask、应用 logits bias，并及时释放
        #       vocab_mask 显存。
        # NOTE: In overlap mode, the function update_regex_vocab_mask (in sample)
        #       was executed after we processed last batch's results.
        # 中译：注意：overlap 模式下，update_regex_vocab_mask（在 sample 中）是在处理上一批
        #       结果之后执行的。

        # Calculate logits bias and apply it to next_token_logits.
        # 中译：计算 logits bias 并应用到 next_token_logits 上。
        sampling_info.update_regex_vocab_mask()
        observer_state = None
        if observer is not None:
            observer_state = sampling_info.apply_logits_bias_with_observer(
                logits_output.next_token_logits,
                observer=observer,
            )
        else:
            sampling_info.apply_logits_bias(logits_output.next_token_logits)

        # Release the vocab_mask GPU tensor immediately after it has been applied
        # to the logits. In overlap scheduling, the sampling_info (and its
        # grammar_mask) can be kept alive by the delay_sample_func closure and
        # batch_record_buf until the next iteration, causing a steady VRAM leak
        # when structured output (grammar) is used.
        # 中译：在 vocab_mask 应用到 logits 后立即释放其 GPU 张量。overlap 调度下，sampling_info
        sampling_info.grammar_mask = None
        return observer_state

    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Sample and compute logprobs and update logits_output.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output

        Returns:
            A list of next_token_ids

        中译：基于模型前向产出的 logits 采样下一个 token，并按需计算 logprob、就地更新
              logits_output。这是 TpModelWorker 在前向之后调用的采样入口，真正的采样实现
              委托给 self.sampler（应用温度/top-p/top-k、惩罚项等后从分布中采样）。

        参数：
            logits_output：模型前向输出的 logits（含 next_token_logits 等）。
            forward_batch：产生该 logits 的前向批次，携带采样所需信息（sampling_info、
                           是否返回 logprob、positions/seq_lens 等）。

        返回：
            next_token_ids：本批次每个请求采样得到的下一个 token id。
        """
        # 中译：采样前对 logits 做预处理——更新正则/词表 mask、应用 logits bias，
        # LogitsProcessorOutput is normally invocation-scoped, but CUDA graph
        # runners may reuse backing objects. Never leak an auxiliary result from
        # a previous replay into a request with no observer state.
        logits_output.auxiliary_device_output = None
        observer = self.sampling_observer
        # Preserve two-argument overrides when observation is inactive.
        if observer is not None and observer.is_active(forward_batch.sampling_info):
            observer_state = self._preprocess_logits(
                logits_output,
                forward_batch.sampling_info,
                observer=observer,
            )
        else:
                logits_output, forward_batch.sampling_info

        # Sample the next tokens
        # 中译：调用采样器采样下一个 token。最后一个参数是「取 logits 的位置」：
        next_token_ids = self.sampler(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.return_logprob,  # 是否需要返回 logprob
            forward_batch.top_logprobs_nums,  # 每个位置返回的 top-k logprob 数量
            forward_batch.token_ids_logprobs,  # 额外指定要返回 logprob 的 token id 集合
            # For prefill, we only use the position of the last token.
            # 中译：decode 模式逐 token 推进，直接用 positions；
            #       prefill（extend）模式只需每条序列最后一个 token 的位置（seq_lens - 1），
            #       因为只有最后一个位置才产出「下一个 token」的 logits。
            (
                forward_batch.positions
                if forward_batch.forward_mode.is_decode()
                else forward_batch.seq_lens - 1
            ),
        )
        # 中译：若模型使用 n-gram embedding，则把刚采样出的 token 写回其 token 表（供后续步使用）。
        if observer_state is not None:
            logits_output.auxiliary_device_output = observer.after_sample(
                observer_state,
                next_token_ids,
            )
        self.ngram_embedding_manager.update_after_decode(
            next_token_ids=next_token_ids,
            forward_batch=forward_batch,
        return next_token_ids

    def compute_logprobs_only(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> None:
        """
        Compute token_ids_logprobs without performing sampling.

        Optimized path for prefill-only requests that need token_ids_logprobs but don't
        require next token generation. Skips expensive sampling operations
        while still providing requested probability information.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output
        """
        logits_output.auxiliary_device_output = None
        if not forward_batch.token_ids_logprobs:
            return

        self._preprocess_logits(logits_output, forward_batch.sampling_info)

        # Delegate to sampler for logprob-only computation
        # This populates logits_output with requested token probabilities
        self.sampler.compute_logprobs_only(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
        )

        # 中译：将模型保存到远程存储（由 url 指定）。
        # 中译：将模型以分片（sharded）形式保存到本地路径。
    def check_weights(self, action: str, allow_quant_error: bool = False):
        return self._weight_checker.handle(
            action=action, allow_quant_error=allow_quant_error
        )

        # 中译：检查/校验权重（由 WeightChecker 处理，用于调试/验证场景）。
        # 中译：预分配对称内存池。
        # 中译：PyTorch 的内存池在 OOM 场景下不会去碎片化，所以需要预先分配一大块内存以限制碎片。
    def _expand_eplb_metadata_for_scale(
        self,
        from_ep_size: int,
        effective_size: int,
    ) -> None:
        metadata = get_global_expert_location_metadata()
        if metadata is None:
            return
        old_num_physical = metadata.num_physical_experts
        num_local = old_num_physical // from_ep_size
        added = num_local * effective_size - old_num_physical
        if added <= 0:
            return

        initial_ep_size = get_parallel().elastic_ep_initial_size
        assert initial_ep_size is not None
        get_context().override("elastic_ep.scale", ep_size=effective_size)

        expanded_p2l = append_trivial_expert_slots(
            metadata.physical_to_logical_map,
            added,
            metadata.num_logical_experts,
            start=old_num_physical - num_local * initial_ep_size,
        )
        new_metadata = ExpertLocationMetadata.init_by_mapping(
            self.model_config,
            physical_to_logical_map=expanded_p2l,
            moe_ep_rank=self._elastic_global_rank(),
        )
        set_global_expert_location_metadata(new_metadata, allow_overwrite=True)

    def _elastic_global_rank(self) -> int:
        return self.ps.tp_rank + get_parallel().ep_join_rank_offset

    def _rearm_eplb_after_elastic_scale(self) -> None:
        if self.eplb_manager is None:
            return
        recorder = get_global_expert_distribution_recorder()
        if not recorder.recording:
            recorder.start_record()
        self.eplb_manager.enable_rebalance()

    def _reset_eplb_after_elastic_scale_failure(self) -> None:
        if self.eplb_manager is None:
            return
        set_global_expert_distribution_recorder(
            ExpertDistributionRecorder.init_new(
                get_global_expert_location_metadata(),
                rank=self._elastic_global_rank(),
            )
        )
        self._rearm_eplb_after_elastic_scale()

    def _report_elastic_scale_failure(self, error: str, effective_size: int) -> None:
        if self.ps.tp_rank != 0 or is_ep_scale_joiner():
            return
        from sglang.srt.managers.io_struct import ElasticScaleUpdateReq

        self._pending_elastic_scale_update = ElasticScaleUpdateReq(
            success=False,
            effective_ep_size=effective_size,
            error=error,
        )

    def _elastic_scale_ready_barrier(self, target_size: int, log_tag: str) -> None:
        if self.ps.tp_rank == 0:
            logger.debug(
                "[Elastic EP][scale] %s entering post-scale WORLD barrier "
                "(target_ep_size=%d)",
                log_tag,
                target_size,
            )
        dist.barrier(group=dist.group.WORLD)
        if self.ps.tp_rank == 0:
            logger.debug(
                "[Elastic EP][scale] %s passed post-scale WORLD barrier "
                "(target_ep_size=%d)",
                log_tag,
                target_size,
            )

    def _finalize_scale_up(
        self,
        ranks_to_join: list[int],
        target_size: int,
        effective_size: int,
    ) -> None:
        self.forward_pass_id = 0
        ElasticEPStateManager.mark_configuring_data_plane()

        state = ElasticEPStateManager.instance()
        for rank in ranks_to_join:
            state.active_ranks[rank] = 1
        state.snapshot_active_to_last()
        state.sync_active_to_cpu()
        if self.eplb_manager is not None:
            self.eplb_manager.reset_generator()

        self._expand_eplb_metadata_for_scale(
            from_ep_size=effective_size,
            effective_size=target_size,
        )
        broadcast_global_expert_location_metadata(
            model_config=self.model_config,
            moe_ep_rank=self._elastic_global_rank(),
            src_rank=0,
        )

        ElasticEPStateManager.on_scale(effective_size, target_size)
        set_global_expert_distribution_recorder(
            ExpertDistributionRecorder.init_new(
                get_global_expert_location_metadata(),
                rank=self._elastic_global_rank(),
            )
        )

        if self.eplb_manager is not None:
            self.eplb_manager.disable_rebalance(
                "EPLB rebalance is disabled while elastic EP scale-up "
                "is being finalized"
            )

        from sglang.srt.layers.dp_attention import update_dp_attention_post_scale

        update_dp_attention_post_scale(
            new_dp_size=target_size,
            new_dp_rank=self._elastic_global_rank(),
        )
        get_context().override("elastic_ep.scale", dp_size=target_size)

        ElasticEPStateManager.mark_syncing_new_world()
        self._elastic_scale_ready_barrier(
            target_size=target_size,
            log_tag="JOINER" if is_ep_scale_joiner() else "PRIMARY",
        )
        ElasticEPStateManager.commit_scale()
        self._rearm_eplb_after_elastic_scale()

        if self.ps.tp_rank == 0 and not is_ep_scale_joiner():
            from sglang.srt.managers.io_struct import ElasticScaleUpdateReq

            self._pending_elastic_scale_update = ElasticScaleUpdateReq(
                success=True,
                effective_ep_size=target_size,
                slot_offset=effective_size,
                slot_count=target_size - effective_size,
            )
            logger.info(
                "[Elastic EP] Scale completed: old_ep_size=%d "
                "new_ep_size=%d joined_ranks=%s",
                effective_size,
                target_size,
                ranks_to_join,
            )

    def maybe_join_ep_ranks(self) -> None:
        if not ElasticEPStateManager.is_scaling():
            return

        state = ElasticEPStateManager.instance()
        effective_size = ElasticEPStateManager.get_effective_ep_size()
        pending_size = ElasticEPStateManager.get_pending_ep_size()

        if pending_size is None:
            if state is not None and state.has_scaled:
                error = (
                    "Elastic EP rank recovery is unsupported after runtime scale-up. "
                    "Restart the expanded deployment."
                )
                ElasticEPStateManager.fail_recovery(error)
                self._report_elastic_scale_failure(error, effective_size)
                if self.ps.tp_rank == 0 and not is_ep_scale_joiner():
                    logger.error("[Elastic EP] %s", error)
                return

            recovered = maybe_recover_ep_ranks(
                tp_group=self.tp_group,
                eplb_manager=self.eplb_manager,
                model_config=self.model_config,
                moe_ep_rank=self._elastic_global_rank(),
            )
            if recovered:
                self.forward_pass_id = 0
            return

        local_timeout = (
            state.pending_since is not None
            and time.monotonic() - state.pending_since
            > get_exec().moe.elastic_ep_scale_timeout
        )
        timeout = state.active_ranks.new_tensor(int(local_timeout))
        dist.all_reduce(timeout, op=dist.ReduceOp.MAX, group=dist.group.WORLD)
        if timeout.item():
            error = f"Timed out waiting for ranks to join target EP size {pending_size}"
            ElasticEPStateManager.fail_scale(error)
            self._reset_eplb_after_elastic_scale_failure()
            self._report_elastic_scale_failure(error, effective_size)
            if self.ps.tp_rank == 0 and not is_ep_scale_joiner():
                logger.error("[Elastic EP] %s", error)
            return

        if state.scale_phase == "waiting_for_cohort":
            cohort_target = get_scale_cohort_target(effective_size)
            if cohort_target is None:
                return
            if cohort_target != pending_size:
                error = (
                    f"Requested target EP size {pending_size} does not match "
                    f"joining cohort target {cohort_target}"
                )
                ElasticEPStateManager.fail_scale(error)
                self._reset_eplb_after_elastic_scale_failure()
                self._report_elastic_scale_failure(error, effective_size)
                if self.ps.tp_rank == 0 and not is_ep_scale_joiner():
                    logger.error("[Elastic EP] %s", error)
                return
            if not ElasticEPStateManager.begin_scale():
                return

        ranks_to_join = list(range(effective_size, pending_size))
        if not ranks_to_join:
            return

        current_platform.synchronize()
        ElasticEPStateManager.mark_joining()
        if try_admit_scale_ranks(ranks_to_join):
            self._finalize_scale_up(
                ranks_to_join=ranks_to_join,
                target_size=pending_size,
                effective_size=effective_size,
            )

    def _maybe_rebalance_after_rank_fault(
        self,
        output: ModelRunnerOutput,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors],
        reinit_attn_backend: bool,
        split_forward_count: int,
    ) -> ModelRunnerOutput:
        # 中译：（弹性 EP）当检测到 rank 故障导致活跃集变化时，触发一次 EPLB 重平衡，
        if maybe_rebalance_after_rank_fault(eplb_manager=self.eplb_manager):
            output = self._forward_raw(
                forward_batch,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
        return output

    # 中译：以 direct 方式直接将命名张量加载到模型参数（使用默认权重加载器）。
    # 中译：解包张量：若是跨进程序列化的 LocalSerializedTensor，则取出对应 tp_rank 的部分，
    def update_model_fields(
        self,
        new_model: torch.nn.Module,
        *,
        model_path: str,
        load_format: str,
        load_config: LoadConfig,
    ) -> None:
        self.model = new_model
        # The record says what model this PROCESS serves; a draft's weight
        # update is not that (its own state is on the runner).
        if not self.is_draft_worker:
            get_context().override(
                "model_runner.update_model_fields",
                model_path=model_path,
                load_format=load_format,
            )
        self.load_config = load_config
