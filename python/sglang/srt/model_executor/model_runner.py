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
import datetime
import gc
import hashlib
import inspect
import logging
import os
import socket
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from sglang.jit_kernel.ngram_embedding import update_token_table_decode
from sglang.srt.compilation.torch_compile_decoration import set_torch_compile_config
from sglang.srt.configs import (
    BailingHybridConfig,
    FalconH1Config,
    GraniteMoeHybridConfig,
    InternS2PreviewConfig,
    JetNemotronConfig,
    JetVLMConfig,
    KimiLinearConfig,
    Lfm2Config,
    Lfm2MoeConfig,
    Lfm2VlConfig,
    NemotronH_Nano_VL_V2_Config,
    NemotronHConfig,
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3NextConfig,
    ZayaConfig,
)
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.linear_attn_model_registry import get_linear_attn_config
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.configs.model_config import (
    AttentionArch,
    ModelConfig,
    ModelImpl,
    get_num_indexer_layers,
)
from sglang.srt.configs.update_config import adjust_config_with_unaligned_cpu_tp
from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.debug_utils.dumper import dumper
from sglang.srt.debug_utils.tensor_dump_forward_hook import (
    register_forward_hook_for_model,
)
from sglang.srt.distributed import (
    get_default_distributed_backend,
    get_pp_group,
    get_tp_group,
    get_world_group,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
    set_mscclpp_all_reduce,
    set_torch_symm_mem_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.distributed.parallel_state import monkey_patch_vllm_parallel_state
from sglang.srt.elastic_ep.elastic_ep import (
    ElasticEPStateManager,
    join_process_groups,
    try_recover_ranks,
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
    broadcast_global_expert_location_metadata,
    compute_initial_expert_location_metadata,
    get_global_expert_location_metadata,
    set_global_expert_location_metadata,
)
from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.kv_canary.api import install_canary
from sglang.srt.kv_canary.runner.canary_manager import context_tuple
from sglang.srt.kv_canary.token_oracle.install import install_token_oracle_from_env
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.attention.attention_registry import (
    ATTENTION_BACKENDS,
    attn_backend_wrapper,
)
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    get_attention_tp_group,
    get_attention_tp_size,
    initialize_dp_attention,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.hash_topk import HashTopK
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
from sglang.srt.layers.sampler import create_sampler
from sglang.srt.layers.torchao_utils import apply_torchao_config_to_model
from sglang.srt.layers.utils.cp_utils import is_mla_prefill_cp_enabled
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.schedule_batch import sanity_check_mm_pad_shift_value
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    CudaGraphBufferRegistry,
    build_decode_registry,
    build_prefill_registry,
)
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
    cuda_graph_fully_disabled,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    has_forward_context,
)
from sglang.srt.model_executor.hook_manager import register_forward_hooks
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.model_executor.runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    _allocate_decode_buffers,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    enable_tc_piecewise_cuda_graph,
    set_tc_piecewise_forward_context,
)
from sglang.srt.model_loader.loader import DefaultModelLoader, get_model_loader
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    RemoteInstanceWeightLoaderBackend,
    register_memory_region,
    trigger_init_weights_send_group_for_remote_instance_request,
)
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.platforms import current_platform
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.spec_info import (
    SpeculativeAlgorithm,
    create_dummy_verify_input,
)
from sglang.srt.state_capturer.base import TopkCaptureOutput
from sglang.srt.state_capturer.indexer_topk import (
    create_indexer_capturer,
    get_global_indexer_capturer,
    set_global_indexer_capturer,
)
from sglang.srt.state_capturer.routed_experts import (
    RoutedExpertsCapturer,
    get_global_experts_capturer,
    set_global_experts_capturer,
)
from sglang.srt.utils import (
    MultiprocessingSerializer,
    broadcast_pyobj,
    cpu_has_amx_support,
    dynamic_import,
    empty_context,
    enable_show_time_cost,
    get_available_gpu_memory,
    get_bool_env_var,
    get_cpu_ids_by_node,
    init_custom_process_group,
    is_hip,
    is_host_cpu_arm64,
    is_npu,
    log_info_on_rank0,
    monkey_patch_p2p_access_check,
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_tp_gather,
    reserve_rope_cache_for_long_sequences,
    set_cuda_arch,
    slow_rank_detector,
)
from sglang.srt.utils.common import ceil_align, next_power_of_2, require_mlp_sync
from sglang.srt.utils.network import NetworkAddress, get_local_ip_auto
from sglang.srt.utils.nvtx_pytorch_hooks import PytHooks
from sglang.srt.utils.nvtx_utils import profile_range
from sglang.srt.utils.offloader import (
    create_offloader_from_server_args,
    get_offloader,
    set_offloader,
)
from sglang.srt.utils.patch_torch import (
    monkey_patch_torch_reductions,
    register_sgl_tp_rank,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils.weight_checker import WeightChecker
from sglang.srt.weight_sync.tensor_bucket import (
    FlattenedTensorBucket,
    FlattenedTensorMetadata,
)

_is_hip = is_hip()
_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu_arm64 = is_host_cpu_arm64()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _is_npu:
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend

    init_npu_backend()
elif current_platform.is_out_of_tree():
    current_platform.init_backend()

MLA_ATTENTION_BACKENDS = [
    "aiter",
    "flashinfer",
    "fa3",
    "fa4",
    "triton",
    "flashmla",
    "cutedsl_mla",
    "cutlass_mla",
    "trtllm_mla",
    "tokenspeed_mla",
    "ascend",
    "dsa",
    "nsa",  # Deprecated alias for "dsa"
    "intel_xpu",
]

CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS = [
    "flashinfer",
    "fa3",
    "fa4",
    "flashmla",
    "cutedsl_mla",
    "cutlass_mla",
    "trtllm_mla",
    "tokenspeed_mla",
]

TORCH_DTYPE_TO_KV_CACHE_STR = {
    torch.float8_e4m3fn: "fp8_e4m3",
    torch.float8_e4m3fnuz: "fp8_e4m3",
    torch.float8_e5m2: "fp8_e5m2",
    torch.bfloat16: "bf16",
}


def add_mla_attention_backend(backend_name):
    # 将一个注意力后端名称注册到 MLA（Multi-head Latent Attention）支持列表中。
    if backend_name not in MLA_ATTENTION_BACKENDS:
        MLA_ATTENTION_BACKENDS.append(backend_name)
        logger.info(f"Added {backend_name} to MLA_ATTENTION_BACKENDS.")


def add_chunked_prefix_cache_attention_backend(backend_name):
    # 将一个注意力后端名称注册为“支持分块前缀缓存（chunked prefix cache）”的后端。
    if backend_name not in CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS:
        CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS.append(backend_name)
        logger.info(
            f"Added {backend_name} to CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS."
        )


# Detect stragger ranks in model loading
# 中译：用于检测模型加载过程中“拖后腿”（加载过慢）的 rank。
UNBALANCED_MODEL_LOADING_TIMEOUT_S = 480  # leave more time for post data processing  # 中译：留出更多时间用于后续数据处理


logger = logging.getLogger(__name__)

_UNSET: Any = object()


def resolve_language_model(model: nn.Module) -> nn.Module:
    # 中译：从（可能是多模态或组合型的）模型中解析出其底层的语言模型子模块。
    # 不同模型的属性命名不同（thinker.model / model / language_model），这里逐一适配。
    model_cls_name = model.__class__.__name__
    if model_cls_name == "Qwen3OmniMoeForConditionalGeneration":
        return model.thinker.model
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "language_model"):
        return model.language_model
    return model.model


class RankZeroFilter(logging.Filter):
    """Filter that only allows INFO level logs from rank 0, but allows all other levels from any rank.

    中译：一个日志过滤器。它只允许 rank 0 输出 INFO 级别的日志（避免多卡/多进程
    场景下 INFO 日志刷屏），但允许任意 rank 输出其他级别（如 WARNING/ERROR）的日志。
    """

    def __init__(self, is_rank_zero):
        super().__init__()
        self.is_rank_zero = is_rank_zero

    def filter(self, record):
        if record.levelno == logging.INFO:
            return self.is_rank_zero
        return True


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


@dataclass
class _EagerBufferRegistry:
    # Lazily-built eager input-buffer registry plus the capacity it was sized to.
    # 中译：懒加载（首次用到时才构建）的 eager 模式输入缓冲区注册表，以及它被创建时所依据的容量
    #       （max_bs / max_num_tokens）。当请求超过该容量时需重建。
    registry: Optional[CudaGraphBufferRegistry] = None
    max_bs: int = 0
    max_num_tokens: int = 0


class ModelRunner(ModelRunnerKVCacheMixin):
    """ModelRunner runs the forward passes of the models.

    中译：ModelRunner 负责执行模型的前向计算。它是调度器（Scheduler）与底层模型、
    各种后端（注意力/量化/通信）之间的桥梁，同时通过混入 ModelRunnerKVCacheMixin
    获得 KV 缓存相关的能力。
    """

    def __init__(
        self,
        # 中译：模型配置对象，包含 HF 配置、是否生成模型、注意力架构（MHA/MLA）、是否多模态等元信息。
        model_config: ModelConfig,
        # 中译：静态显存占比，即权重 + KV 缓存等固定占用相对于 GPU 总显存的上限比例（0~1）。
        mem_fraction_static: float,
        # 中译：本 worker 绑定的物理 GPU 编号。
        gpu_id: int,
        # 中译：张量并行（Tensor Parallel）中当前进程的 rank（从 0 开始）。
        tp_rank: int,
        # 中译：张量并行的总规模（参与 TP 的进程数）。
        tp_size: int,
        # 中译：MoE 专家并行（Expert Parallel）中当前进程的 rank。
        moe_ep_rank: int,
        # 中译：MoE 专家并行的总规模。
        moe_ep_size: int,
        # 中译：流水线并行（Pipeline Parallel）中当前进程所处的 stage rank。
        pp_rank: int,
        # 中译：流水线并行的总规模（stage 数）。
        pp_size: int,
        # 中译：用于初始化 NCCL 分布式通信组的端口号。
        nccl_port: int,
        # 中译：全局服务端参数（ServerArgs），承载几乎所有用户可配置的运行时选项。
        server_args: ServerArgs,
        # 中译：数据并行（Data Parallel）中当前进程的 rank；非 DP 场景为 None。
        dp_rank: Optional[int] = None,
        # 中译：注意力上下文并行（Context Parallel）中当前进程的 rank；未启用时为 None。
        attn_cp_rank: Optional[int] = None,
        # 中译：MoE 数据并行中当前进程的 rank；未启用时为 None。
        moe_dp_rank: Optional[int] = None,
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
        self.device = server_args.device
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.moe_ep_rank = moe_ep_rank
        self.moe_ep_size = moe_ep_size
        self.dp_rank = dp_rank
        self.dp_size = server_args.dp_size if server_args.enable_dp_attention else 1
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.attn_cp_rank = attn_cp_rank
        self.attn_cp_size = server_args.attn_cp_size
        self.moe_dp_rank = moe_dp_rank
        self.moe_dp_size = server_args.moe_dp_size
        self.model_config = model_config
        self.dist_port = nccl_port
        self.server_args = server_args
        self.is_draft_worker = is_draft_worker
        self.is_generation = model_config.is_generation
        self.device_timer = None
        self.is_multimodal = model_config.is_multimodal
        self.is_multimodal_chunked_prefill_supported = (
            model_config.is_multimodal_chunked_prefill_supported
        )
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.page_size = server_args.page_size
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.is_hybrid_swa = model_config.is_hybrid_swa
        self.is_hybrid_swa_compress = getattr(
            model_config, "is_hybrid_swa_compress", False
        )
        self.use_mla_backend = self.model_config.attention_arch == AttentionArch.MLA
        self.attention_chunk_size = model_config.attention_chunk_size
        rope_scaling = getattr(
            model_config.hf_text_config, "rope_parameters", None
        ) or getattr(model_config.hf_text_config, "rope_scaling", {})
        self.model_is_mrope = (
            rope_scaling is not None and "mrope_section" in rope_scaling
        )
        self.enable_elastic_ep = server_args.elastic_ep_backend is not None
        self.forward_pass_id = 0
        self.init_new_workspace = False
        self._eager_decode_registry = _EagerBufferRegistry()
        self._eager_prefill_registry = _EagerBufferRegistry()
        self.draft_model_idx = draft_model_idx
        self.enable_hisparse = server_args.enable_hisparse

        self.remote_instance_transfer_engine = None
        self.remote_instance_transfer_engine_session_id = ""
        self.remote_instance_transfer_engine_weight_info = None

        self.msprobe_debugger = None
        if server_args.msprobe_dump_config is not None:
            self.init_msprobe()

        # auxiliary hidden capture mode. TODO: expose this to server args?
        # 中译：辅助隐藏状态（aux hidden state）捕获模式的相关开关。用于投机采样
        #       （EAGLE / DFlash）场景，需要捕获主模型中间层的隐藏状态供草稿模型使用。
        #       TODO：未来是否把这些开关暴露到 server args？
        self.eagle_use_aux_hidden_state = False
        self.eagle_draft_num_layers = None
        self.dflash_use_aux_hidden_state = False
        self.dflash_target_layer_ids = None
        self.dflash_draft_num_layers = None
        if (
            (self.spec_algorithm.is_eagle() or self.spec_algorithm.is_standalone())
            and not self.is_draft_worker
            and server_args.speculative_draft_model_path
        ):
            # Load draft config to get layer count for KV cache sizing
            # 中译：加载草稿模型的配置，以获取其层数，用于为 KV 缓存定尺寸。
            draft_model_config = self._build_model_config(
                server_args,
                model_path=server_args.speculative_draft_model_path,
                model_revision=server_args.speculative_draft_model_revision,
                is_draft_model=True,
            )
            num_nextn_predict_layers = draft_model_config.num_nextn_predict_layers
            if num_nextn_predict_layers is not None:
                self.eagle_draft_num_layers = int(num_nextn_predict_layers)
            else:
                self.eagle_draft_num_layers = int(
                    max(
                        draft_model_config.num_hidden_layers,
                        draft_model_config.num_attention_layers,
                    )
                )

            if self.spec_algorithm.is_eagle3():
                self.eagle_use_aux_hidden_state = True
                try:
                    eagle_config = getattr(
                        draft_model_config.hf_config, "eagle_config", None
                    )
                    self.eagle_use_aux_hidden_state = eagle_config.get(
                        "use_aux_hidden_state", True
                    )
                    self.eagle_aux_hidden_state_layer_ids = eagle_config[
                        "eagle_aux_hidden_state_layer_ids"
                    ]
                except:
                    # if there is no aux layer, set to None
                    # 中译：若配置中没有辅助层信息，则置为 None。
                    self.eagle_aux_hidden_state_layer_ids = None

        if self.spec_algorithm.is_dflash() and not self.is_draft_worker:
            from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config

            # Select target layers to capture for building DFlash context features.
            # 中译：选择需要捕获的主模型目标层，用于构建 DFlash 的上下文特征。
            draft_model_config = self._build_model_config(
                server_args,
                model_path=(server_args.speculative_draft_model_path),
                model_revision=server_args.speculative_draft_model_revision,
                is_draft_model=True,
            )
            dflash_draft_config = parse_dflash_draft_config(
                draft_hf_config=draft_model_config.hf_config
            )
            draft_num_layers = dflash_draft_config.require_num_layers()
            trained_target_layers = dflash_draft_config.num_target_layers

            target_num_layers = getattr(
                self.model_config.hf_text_config, "num_hidden_layers", None
            )
            if target_num_layers is None:
                raise ValueError(
                    "DFLASH requires target num_hidden_layers in config. "
                    f"Got target={target_num_layers}."
                )
            target_num_layers = int(target_num_layers)

            if (
                trained_target_layers is not None
                and trained_target_layers != target_num_layers
            ):
                logger.warning(
                    "DFLASH draft config num_target_layers=%s differs from runtime target num_hidden_layers=%s; "
                    "selecting capture layers based on the runtime target model.",
                    trained_target_layers,
                    target_num_layers,
                )

            self.dflash_use_aux_hidden_state = True
            self.dflash_draft_num_layers = int(draft_num_layers)
            self.dflash_target_layer_ids = dflash_draft_config.resolve_target_layer_ids(
                target_num_layers=int(target_num_layers),
                draft_num_layers=int(draft_num_layers),
            )

        # Apply the rank zero filter to logger
        # 中译：若开启了耗时统计展示，则启用相应的计时输出。
        if server_args.show_time_cost:
            enable_show_time_cost()

        # Model-specific adjustment
        # 中译：针对特定模型的参数调整（例如某些模型需要特殊的注意力架构/配置修正）。
        self.model_specific_adjustment()

        # Set the global server_args in the scheduler process
        # 中译：在调度器进程中设置全局 server_args，供后续各模块读取。
        set_global_server_args_for_scheduler(server_args)
        global_server_args = get_global_server_args()

        # FIXME: hacky set `use_mla_backend`
        # 中译：（不优雅的临时做法）将 use_mla_backend 写入全局 server_args。
        global_server_args.use_mla_backend = self.use_mla_backend

        # Init OpenMP threads binding for CPU
        # 中译：在 CPU 设备上初始化 OpenMP 线程绑定（绑核），以提升 NUMA 亲和性与性能。
        if self.device == "cpu":
            self.init_threads_binding()

        # Get available memory before model loading.
        # Stored for later use by alloc_memory_pool().
        # 中译：在加载模型之前记录可用显存（同时完成 torch 分布式初始化）。
        #       该值保存起来供后续 alloc_memory_pool() 计算 KV 缓存大小时使用。
        self.pre_model_load_memory = self.init_torch_distributed()

        # Initialize MooncakeTransferEngine
        # 中译：初始化共享的 Mooncake 传输引擎（用于 PD 分离/远程权重传输等场景）。
        self.init_shared_mooncake_transfer_engine()

        # Init forward stream for overlap schedule
        # 中译：为 overlap 调度（计算与调度重叠）创建专用的前向 CUDA stream。
        self.forward_stream = torch.get_device_module(self.device).Stream()

        # CPU offload
        # 中译：根据 server_args 创建并设置 CPU offload（将部分权重/激活卸载到 CPU）的加载器。
        set_offloader(create_offloader_from_server_args(server_args, dp_rank=dp_rank))

        self._weight_checker = WeightChecker(model_runner=self)

        if envs.SGLANG_DETECT_SLOW_RANK.get():
            slow_rank_detector.execute()

        # Init mindspore running environment when model impl is "mindspore"
        self.init_mindspore_runner()

        # Update deep gemm configure
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            deep_gemm_wrapper.update_deep_gemm_config(gpu_id, server_args)

        # For hisparse (must be set before initialize() so CUDA graph capture can see it)
        # 中译：用于 hisparse（分层稀疏注意力）。必须在 initialize() 之前设置，
        #       以便 CUDA Graph 捕获时能看到它。
        self.hisparse_coordinator = None

        self._linear_attn_registry_cache: Any = _UNSET

        # Load model weights and configure
        # 中译：加载模型权重并完成配置（这是 __init__ 中最重的一步）。
        self.initialize()
        # 中译：检查量化 MoE 的兼容性（例如某些量化方案与 MoE 后端是否可同时使用）。
        self.check_quantized_moe_compatibility()

        if (
            self.server_args.elastic_ep_backend is not None
            and self.server_args.elastic_ep_rejoin
        ):
            join_process_groups()
            broadcast_global_expert_location_metadata(
                src_rank=self._get_healthy_expert_location_src_rank(
                    invoked_in_elastic_ep_rejoin_path=True
                )
            )
            ElasticEPStateManager.instance().reset()

        if self.is_multimodal:
            # 中译：多模态模型需检查多模态 pad/shift 值与词表大小是否一致（避免占位符越界）。
            sanity_check_mm_pad_shift_value(self.model_config.vocab_size)

        # Temporary cached values
        # 中译：一些临时缓存值（例如模型 forward 是否支持流水线并行代理张量参数）。
        self.support_pp = (
            "pp_proxy_tensors" in inspect.signature(self.model.forward).parameters
        )

        if self.pp_size > 1:
            # 中译：开启流水线并行（PP）时，模型必须支持 PP，否则报错。
            assert (
                self.support_pp
            ), "Pipeline Parallel is not compatible with this model."

        # For weight updates
        # 中译：用于在线权重更新（如 RLHF rollout）的进程组句柄缓存。
        self._model_update_group = {}
        self._weights_send_group = {}

    def _build_model_config(
        self, server_args, model_path=None, model_revision=None, is_draft_model=False
    ):
        return ModelConfig.from_server_args(
            server_args,
            model_path=model_path,
            model_revision=model_revision,
            is_draft_model=is_draft_model,
        )

    def init_msprobe(self):
        # Init the msprobe
        # 中译：初始化 msprobe（昂腾提供的精度调试/张量 dump 工具）。
        try:
            from msprobe.pytorch import PrecisionDebugger, seed_all
        except ImportError:
            logger.warning(
                "Please install msprobe for tensor data dump: pip install mindstudio-probe --pre, "
                "see https://gitcode.com/Ascend/msprobe for details."
            )
            return
        seed_all(mode=True)
        self.msprobe_debugger = PrecisionDebugger(
            config_path=self.server_args.msprobe_dump_config
        )

    def init_mindspore_runner(self):
        # Init the mindspore runner
        # for now, there is only some communication initialization work
        # 中译：初始化 MindSpore 运行器。目前仅做一些通信初始化工作，
        #       仅在 model_impl 为 mindspore 且设备为 NPU 时生效。
        if self.server_args.model_impl.lower() == ModelImpl.MINDSPORE and _is_npu:
            from sglang.srt.model_executor.mindspore_runner import init_ms_distributed

            init_ms_distributed(
                world_size=self.tp_size * self.pp_size,
                rank=self.tp_size * self.pp_rank + self.tp_rank,
                local_rank=self.gpu_id,
                server_args=self.server_args,
                port=self.dist_port,
            )

    def initialize(self):
        # 中译：核心初始化流程。依次完成：内存节省器创建、专家位置/分布记录初始化、
        #       加载模型、计算有效层范围、应用量化/张量并行/LoRA，以及推导 KV 缓存 dtype 等。
        server_args = self.server_args

        # 中译：创建 torch 显存节省器适配器（可在空闲时释放/重建显存以节省占用）。
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.server_args.enable_memory_saver
        )

        if self.server_args.remote_instance_weight_loader_use_transfer_engine():
            self.remote_instance_init_transfer_engine()

        if not self.is_draft_worker:
            set_global_expert_location_metadata(
                compute_initial_expert_location_metadata(
                    server_args=server_args,
                    model_config=self.model_config,
                    moe_ep_rank=self.moe_ep_rank,
                )
            )
            if self.tp_rank == 0 and envs.SGLANG_LOG_EXPERT_LOCATION_METADATA.get():
                logger.info(
                    f"Initial expert_location_metadata: {get_global_expert_location_metadata()}"
                )

            set_global_expert_distribution_recorder(
                ExpertDistributionRecorder.init_new(
                    server_args,
                    get_global_expert_location_metadata(),
                    rank=self.tp_rank,
                )
            )

        # Expert parallelism
        # 中译：专家并行（EP）。若开启 EPLB（专家负载均衡）且非草稿 worker，则创建 EPLBManager。
        self.eplb_manager = (
            EPLBManager(self)
            if self.server_args.enable_eplb and (not self.is_draft_worker)
            else None
        )
        self.expert_location_updater = ExpertLocationUpdater()

        if self.server_args.elastic_ep_backend:
            ElasticEPStateManager.init(self.server_args)
        self._token_oracle_manager = install_token_oracle_from_env(
            server_args=server_args,
            vocab_size=self.model_config.vocab_size,
        )
        # Load the model
        # 中译：创建采样器、加载模型权重、准备 MoE 的 top-k 路由。
        self.sampler = create_sampler()
        self.load_model()
        self._prepare_moe_topk()

        # Load the expert backup client
        # 中译：加载专家权重备份客户端（仅在弹性 EP + 开启专家备份时创建）。
        self.expert_backup_client = (
            ExpertBackupClient(self.server_args, self)
            if (
                self.server_args.enable_elastic_expert_backup
                and self.server_args.elastic_ep_backend is not None
            )
            else None
        )

        if (
            self.server_args.remote_instance_weight_loader_use_transfer_engine()
            # ModelExpress owns TransferEngine memory registration and metadata
            # publishing for backend=modelexpress. Re-registering here would
            # overlap the same weight buffers.
            and self.server_args.remote_instance_weight_loader_backend
            != RemoteInstanceWeightLoaderBackend.MODELEXPRESS
            and self.remote_instance_transfer_engine is not None
            and self.remote_instance_transfer_engine_weight_info is None
        ):
            # Register memory and upstream the transfer engine info to the bootstrap server
            self.remote_instance_transfer_engine_weight_info = register_memory_region(
                self.model, self.remote_instance_transfer_engine
            )
            self._register_to_engine_info_bootstrap()

        # 中译：对于 DeepSeek-V3 / GLM-4.5 等带 MTP（Multi-Token Prediction）的模型，MTP 层会被单
        #       独作为投机解码的草稿模型使用，此时用 `num_nextn_predict_layers` 决定层数。
        #       某些 EAGLE3 草稿（如 nvidia/Kimi-K2.5-Thinking-Eagle3）携带完整的 DeepSeek-V3 配置
        #       schema 并显式设置 `num_nextn_predict_layers: 0`，需把它等同于该字段缺失——否则
        #       草稿 worker 会走下面的 MTP 分支且 model_num_layers=0，导致草稿 KV 池尺寸为零，
        #       并在首次前向时报 IndexError。
        # For MTP models like DeepSeek-V3 or GLM-4.5, the MTP layer(s) are used separately as draft
        # models for speculative decoding. In those cases, `num_nextn_predict_layers` is used to
        # determine the number of layers.
        # Some EAGLE3 drafts (e.g. nvidia/Kimi-K2.5-Thinking-Eagle3) carry the full DeepSeek-V3
        # config schema and explicitly set `num_nextn_predict_layers: 0`. Treat that the same as
        # the field being absent — otherwise the draft worker takes the MTP branch below with
        # model_num_layers=0, sizing the draft KV pool to zero and producing an IndexError on
        # the first forward (`set_mla_kv_buffer` -> `self.kv_buffer[layer_id - self.start_layer]`).
        _nnpl = self.model_config.num_nextn_predict_layers
        model_has_mtp_layers = _nnpl is not None and _nnpl > 0
        model_num_layers = (
            self.model_config.num_nextn_predict_layers
            if self.is_draft_worker and model_has_mtp_layers
            else max(
                self.model_config.num_hidden_layers,
                self.model_config.num_attention_layers,
            )
        )
        if self.model_config.hf_config.architectures[0] == "MiMoV2MTP":
            model_num_layers = 1
        elif self.model_config.hf_config.architectures[0] == "Step3p5MTP":
            model_num_layers = 1
        # 中译：start_layer/end_layer 是本 rank 负责的层区间（PP 场景下每个 rank 只拥有部分层），
        #       num_effective_layers 为本 rank 实际拥有的层数。
        self.start_layer = getattr(self.model, "start_layer", 0)
        self.end_layer = getattr(self.model, "end_layer", model_num_layers)
        self.num_effective_layers = self.end_layer - self.start_layer

        # 中译：针对混合 SWA（滑动窗口注意力）模型在 PP 下调整层划分。
        self.adjust_hybrid_swa_layers_for_pp()

        # For LoopCoder models, each loop has its own layer_id, so we need to multiply by loop_num
        # 中译：对于 LoopCoder 类模型，每轮循环都有独立的 layer_id，故需乘以 loop_num。
        loop_num = getattr(self.model_config.hf_config, "loop_num", 1)
        if loop_num > 1:
            self.num_effective_layers = self.num_effective_layers * loop_num

        assert (
            (not model_has_mtp_layers)
            or (self.spec_algorithm.is_none())
            or (
                (not self.spec_algorithm.is_none())
                and (self.num_effective_layers == model_num_layers)
            )
        ), "PP is not compatible with MTP models."

        # Apply torchao quantization
        # 中译：应用 torchao 量化配置。
        torchao_applied = getattr(self.model, "torchao_applied", False)
        # In layered loading, torchao may have been applied
        # 中译：在分层加载模式下 torchao 可能已被应用过，避免重复应用。
        if not torchao_applied:
            apply_torchao_config_to_model(
                self.model, get_global_server_args().torchao_config
            )

        # Apply torch TP if the model supports it
        # 中译：若模型支持 torch 原生张量并行（TP）且 tp_size>1，则应用之。
        supports_torch_tp = getattr(self.model, "supports_torch_tp", False)
        if self.tp_size > 1 and supports_torch_tp:
            self.apply_torch_tp()

        # Init lora
        # 中译：初始化 LoRA 管理器（若启用 LoRA）。
        if server_args.enable_lora:
            self.init_lora_manager()
            if not cuda_graph_fully_disabled():
                # Phase 1 of LoRA CUDA graph init: pre-allocate large MoE
                # intermediate buffers before init_memory_pool() so memory
                # profiling accounts for them. The buffers are reused by
                # any captured graph (decode today; widen here so any
                # future prefill capture path also picks them up).
                self._init_lora_cuda_graph_moe_buffers()

        # Enable batch invariant mode
        # 中译：若开启确定性推理，则启用 batch invariant 模式（使结果与 batch 组合无关，
        #       从而可复现）。
        if server_args.enable_deterministic_inference:
            from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

            enable_batch_invariant_mode()

        # Deduce KV cache dtype
        # 中译：推导（确定）KV 缓存的数据类型。
        self.configure_kv_cache_dtype()

        # Snapshot free memory at the end of the weight-load phase. KV-pool
        # profiling uses this instead of measuring at alloc_memory_pool()
        # time: draft-model weights load between the two phases and must stay
        # outside the --mem-fraction-static budget (deployments tune the
        # fraction assuming draft weights live in the non-static slack).
        # 中译：在权重加载阶段结束时快照记录空闲显存。KV 池 profiling 使用该值，
        #       而不是在 alloc_memory_pool() 时才测量：因为草稿模型权重是在这两个阶段之间
        #       加载的，它必须落在 --mem-fraction-static 预算之外（部署时按“草稿权重住在
        #       非静态富余量里”来调整该比例）。
        self.post_model_load_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=get_world_group().world_size > 1,
            cpu_group=get_world_group().cpu_group,
        )

    def alloc_memory_pool(self, memory_pool_config: Optional[MemoryPoolConfig] = None):
        """Allocate KV cache memory pools only (no backends or cuda graphs).

        中译：仅分配 KV 缓存内存池（不包含注意力后端与 CUDA Graph 的初始化）。
        """
        if memory_pool_config is not None:
            self.memory_pool_config = memory_pool_config

        self.init_memory_pool(self.pre_model_load_memory)

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
        self.maybe_init_ngram_embedding()

        if self.enable_hisparse:
            # 中译：初始化 hisparse（分层稀疏注意力）协调器，管理 host/device 上的稀疏 KV 缓存。
            from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
            from sglang.srt.mem_cache.sparsity import parse_hisparse_config

            hisparse_cfg = parse_hisparse_config(self.server_args)
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
                    if self.server_args.enable_dp_attention
                    else self.tp_group.cpu_group
                ),
                host_to_device_ratio=hisparse_cfg.host_to_device_ratio,
            )

        # 中译：初始化路由专家捕获器与 indexer top-k 捕获器（用于调试/状态导出）。
        self.init_routed_experts_capturer()
        self.init_indexer_capturer()

        # 中译：注意力后端与 CUDA Graph runner 的占位字段，稍后在 init_backends() 中真正初始化。
        self.attn_backend = None
        self.decode_attn_backend = None
        self.decode_attn_backend_group = []
        self.decode_cuda_graph_runner = None
        self.graph_mem_usage = 0
        self.prefill_cuda_graph_runner = None

    def init_backends(self, disable_cuda_graph: bool = False):
        """Initialize attention backends and capture cuda graphs.

        中译：初始化注意力后端并捕获 CUDA Graph。会根据不同设备（cuda/cpu/npu/其他）走
        不同的初始化分支。
        """
        server_args = self.server_args

        # TODO: Refactor device-specific init branches into platform interface (separate PR).
        # Must be called BEFORE init_decode_cuda_graph() so CUDA graph capture
        # runs with aux hidden state capture enabled.
        # 中译：TODO：未来把设备相关的初始化分支重构进 platform 接口（另开 PR）。
        #       必须在 init_decode_cuda_graph() 之前调用，以便 CUDA Graph 捕获时已启用
        #       辅助隐藏状态捕获。
        self.init_aux_hidden_state_capture()

        if self.device == "cuda" or self.device == "musa":
            self.init_cublas()
            self.init_attention_backend()
            self.kernel_warmup()
            self._pre_initialize_flashinfer_allreduce_workspace()
            if not disable_cuda_graph:
                self.init_decode_cuda_graph()
        elif self.device == "cpu":
            self.init_attention_backend()
            if not disable_cuda_graph:
                self.init_decode_cuda_graph()
        elif self.device == "npu":
            self.init_attention_backend()
            # lazy init for zbal with mix mode (before graph capture when enable_cuda_graph)
            # 中译：NPU 上对 zbal 混合模式的懒初始化（在开启 cuda graph 时需在 graph 捕获之前完成）。
            if envs.SGLANG_ZBAL_LOCAL_MEM_SIZE.get() > 0 and not self.is_draft_worker:
                from sglang.srt.hardware_backend.npu.utils import lazy_init_zbal_gva_mem

                lazy_init_zbal_gva_mem(
                    self.device,
                    self.gpu_id,
                    get_world_group().rank_in_group,
                    get_world_group().world_size,
                    get_world_group().cpu_group,
                )
            if not disable_cuda_graph:
                self.init_decode_cuda_graph()
        elif current_platform.is_out_of_tree():
            self.init_attention_backend()
            if current_platform.support_cuda_graph() and not disable_cuda_graph:
                self.init_decode_cuda_graph()
            else:
                self.decode_cuda_graph_runner = None
                self.graph_mem_usage = 0
        else:
            self.decode_cuda_graph_runner = None
            self.graph_mem_usage = 0
            self.init_attention_backend()

        if disable_cuda_graph:
            self.decode_cuda_graph_runner = None
            self.graph_mem_usage = 0

        if server_args.forward_hooks:
            # 中译：根据配置为模型注册前向 hook（用于调试/观测）。
            register_forward_hooks(self.model, server_args.forward_hooks)

        # 中译：初始化预填阶段的（分段）CUDA Graph runner。
        self.init_prefill_cuda_graph()

        # 中译：预分配对称内存池（用于 symmetric memory all-reduce 等场景）。
        self.prealloc_symmetric_memory_pool()

        if self.canary_manager is not None and not self.is_draft_worker:
            # 中译：标记 canary 初始化完成。
            self.canary_manager.mark_init_finished()

    def adjust_hybrid_swa_layers_for_pp(self):
        # 中译：在流水线并行（PP）下，重新计算本 rank 实际拥有的「全量注意力层」与
        #       「滑动窗口注意力（SWA）层」的 id 集合（只保留落在本 rank 层区间内的）。
        if not self.is_hybrid_swa:
            return

        if self.model_config.is_deepseek_v4_arch:
            return

        full_attention_layer_ids = [
            layer_idx
            for layer_idx in range(self.start_layer, self.end_layer + 1)
            if hasattr(self.model_config, "full_attention_layer_ids")
            and layer_idx in self.model_config.full_attention_layer_ids
        ]
        swa_attention_layer_ids = [
            layer_idx
            for layer_idx in range(self.start_layer, self.end_layer + 1)
            if hasattr(self.model_config, "swa_attention_layer_ids")
            and layer_idx in self.model_config.swa_attention_layer_ids
        ]
        self.model_config.swa_attention_layer_ids = swa_attention_layer_ids
        self.model_config.full_attention_layer_ids = full_attention_layer_ids

    def init_routed_experts_capturer(self):
        # 中译：初始化路由专家捕获器（用于在返回结果中携带 MoE 路由选中的专家信息）。
        if not self.server_args.disable_shared_experts_fusion and hasattr(
            self.model, "num_fused_shared_experts"
        ):
            num_fused_shared_experts = self.model.num_fused_shared_experts
        else:
            num_fused_shared_experts = 0

        set_global_experts_capturer(
            RoutedExpertsCapturer.create(
                enable=get_global_server_args().enable_return_routed_experts,
                model_config=self.model_config,
                num_fused_shared_experts=num_fused_shared_experts,
                num_tokens=self.max_total_num_tokens + self.page_size,
                max_running_requests=self.max_running_requests,
                device=self.device,
            )
        )

    def init_indexer_capturer(self):
        # 中译：初始化 indexer top-k 捕获器（用于在返回结果中携带 indexer 的 top-k 信息）。
        enable = get_global_server_args().enable_return_indexer_topk
        # Producer wiring is CUDA-only (Indexer.forward_cuda + MLA skip_topk
        # path); other backends would create a capturer but never feed it.
        # 中译：生产端的接线仅支持 CUDA（Indexer.forward_cuda + MLA skip_topk 路径）；
        #       其他后端即使创建了捕获器也不会向其填数据，故非 CUDA 时禁用。
        if enable and self.device != "cuda":
            logger.warning(
                "indexer-topk capture is CUDA-only; %s backend not yet wired. "
                "Disabling capturer.",
                self.device,
            )
            set_global_indexer_capturer(None)
            return

        hf_text_config = self.model_config.hf_text_config
        num_indexer_layers = get_num_indexer_layers(hf_text_config)
        index_topk = getattr(hf_text_config, "index_topk", 0)
        set_global_indexer_capturer(
            create_indexer_capturer(
                enable=enable,
                num_indexer_layers=num_indexer_layers,
                index_topk=index_topk,
                num_tokens=self.max_total_num_tokens + self.page_size,
                max_running_requests=self.max_running_requests,
                device=self.device,
            )
        )

    def init_aux_hidden_state_capture(self):
        """Configure auxiliary hidden state capture for speculative decoding.

        Must be called before CUDA graph capture so the captured graphs
        include aux hidden state output paths.

        中译：为投机解码配置辅助隐藏状态（aux hidden state）的捕获。必须在 CUDA Graph
        捕获之前调用，以保证被捕获的 graph 包含辅助隐藏状态的输出路径。
        """
        if self.eagle_use_aux_hidden_state:
            self.model.set_eagle3_layers_to_capture(
                self.eagle_aux_hidden_state_layer_ids
            )
        if self.dflash_use_aux_hidden_state:
            if not hasattr(self.model, "set_dflash_layers_to_capture"):
                raise ValueError(
                    f"Model {self.model.__class__.__name__} does not implement "
                    "set_dflash_layers_to_capture, which is required for DFLASH."
                )
            self.model.set_dflash_layers_to_capture(self.dflash_target_layer_ids)

    def remote_instance_init_transfer_engine(self):
        # 中译：为“远程实例权重加载”初始化 Mooncake 传输引擎（需安装 mooncake）。
        try:
            from mooncake.engine import TransferEngine
        except ImportError as e:
            logger.warning(
                "Please install mooncake for using remote instance transfer engine: pip install mooncake"
            )
            return
        self.remote_instance_transfer_engine = TransferEngine()
        local_ip = get_local_ip_auto()
        self.remote_instance_transfer_engine.initialize(
            local_ip,
            "P2PHANDSHAKE",
            envs.MOONCAKE_PROTOCOL.get(),
            envs.MOONCAKE_DEVICE.get(),
        )
        self.remote_instance_transfer_engine_session_id = NetworkAddress(
            local_ip, self.remote_instance_transfer_engine.get_rpc_port()
        ).to_host_port_str()

    def _register_to_engine_info_bootstrap(self):
        """Register transfer engine info with the EngineInfoBootstrapServer via HTTP PUT.

        The bootstrap server runs on node_rank==0. For multi-node setups, the
        host is derived from dist_init_addr. For single-node, use 127.0.0.1.

        中译：通过 HTTP PUT 将传输引擎信息注册到 EngineInfoBootstrapServer。
        该 bootstrap 服务运行在 node_rank==0 上；多机场景下 host 从 dist_init_addr 推导，
        单机场景则使用 127.0.0.1。
        """
        import requests as http_requests

        if self.server_args.dist_init_addr:
            # Multi-node: bootstrap server is on the head node (node_rank==0).
            # Derive host from dist_init_addr (shared across all nodes).
            bootstrap_host = (
                NetworkAddress.parse(self.server_args.dist_init_addr).resolved().host
            )
        else:
            bootstrap_host = "127.0.0.1"

        bootstrap_port = self.server_args.engine_info_bootstrap_port
        bootstrap_na = NetworkAddress(bootstrap_host, bootstrap_port)
        url = f"{bootstrap_na.to_url()}/register_transfer_engine_info"

        payload = {
            "tp_rank": self.tp_rank,
            "transfer_engine_info": {
                "session_id": self.remote_instance_transfer_engine_session_id,
                "weights_info_dict": self.remote_instance_transfer_engine_weight_info,
            },
        }

        try:
            resp = http_requests.put(url, json=payload, timeout=5)
            if resp.status_code == 200:
                logger.info(
                    f"Registered transfer engine info for tp_rank={self.tp_rank} "
                    f"with bootstrap server at {bootstrap_na}"
                )
            else:
                logger.error(
                    f"Failed to register transfer engine info for tp_rank={self.tp_rank}: "
                    f"{resp.status_code}, {resp.text}"
                )
        except Exception as e:
            logger.error(
                f"Failed to register transfer engine info for tp_rank={self.tp_rank}: {e}"
            )

    def model_specific_adjustment(self):
        # 中译：针对具体模型特性调整 server_args（例如：不支持分块预填的多模态模型需关闭
        #       chunked prefill；非 MLA 或后端不支持时禁用分块前缀缓存）。
        server_args = self.server_args

        if self.is_multimodal:
            if not self.is_multimodal_chunked_prefill_supported:
                server_args.chunked_prefill_size = -1
                logger.info(
                    f"Automatically turn off --chunked-prefill-size as it is not supported for "
                    f"{self.model_config.hf_config.model_type}"
                )

        if (
            not self.use_mla_backend
            or server_args.attention_backend
            not in CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS
        ):
            server_args.disable_chunked_prefix_cache = True

        if not server_args.disable_chunked_prefix_cache:
            log_info_on_rank0(logger, "Chunked prefix cache is turned on.")

    def check_quantized_moe_compatibility(self):
        # 中译：检查量化 MoE 的参数可整除性约束：tp_size 必须能被 ep_size 整除，
        #       且 moe_intermediate_size 必须能被 moe_tp_size 整除，否则报错。
        if (
            quantization_config := getattr(
                self.model_config.hf_config, "quantization_config", None
            )
        ) is not None and (
            weight_block_size := quantization_config.get("weight_block_size", None)
        ) is not None:
            weight_block_size_n = weight_block_size[0]

            if self.tp_size % self.moe_ep_size != 0:
                raise ValueError(
                    f"tp_size {self.tp_size} must be divisible by ep_size {self.moe_ep_size}"
                )
            moe_tp_size = self.tp_size // self.moe_ep_size // self.moe_dp_size

            moe_intermediate_size = getattr(
                self.model_config.hf_text_config, "moe_intermediate_size", None
            )
            if moe_intermediate_size is None:
                return

            if moe_intermediate_size % moe_tp_size != 0:
                raise ValueError(
                    f"moe_intermediate_size {moe_intermediate_size} must be divisible by moe_tp_size ({moe_tp_size}) which is tp_size ({self.tp_size}) divided by moe_ep_size ({self.moe_ep_size})."
                )

            if (
                not envs.SGLANG_SHARED_EXPERT_TP1.get()
                and (moe_intermediate_size // moe_tp_size) % weight_block_size_n != 0
                and not _use_aiter
            ):
                raise ValueError(
                    f"For quantized MoE models, please make sure ({moe_intermediate_size=} / {moe_tp_size=}) % {weight_block_size_n=} == 0 "
                    f"where moe_tp_size is equal to tp_size ({self.tp_size}) divided by ep_size ({self.moe_ep_size}). "
                    f"You can fix this by setting arguments `--tp` and `--ep` correctly."
                )

    def init_torch_distributed(self):
        # 中译：初始化 torch 分布式环境。依次完成：绑定设备、选择通信后端、
        #       设置 all-reduce 策略、初始化分布式环境与各种并行组（TP/PP/EP/DP）、
        #       预热 NCCL/RCCL，并返回模型加载前的可用显存。
        tic = time.perf_counter()
        logger.info("Init torch distributed begin.")

        try:
            torch.get_device_module(self.device).set_device(self.gpu_id)
        except Exception:
            logger.warning(
                f"Context: {self.device=} {self.gpu_id=} {os.environ.get('CUDA_VISIBLE_DEVICES')=} {self.tp_rank=} {self.tp_size=}"
            )
            raise

        backend = get_default_distributed_backend(self.device)
        if self.device == "cuda" and self.server_args.elastic_ep_backend == "mooncake":
            backend = "mooncake"
            if self.server_args.mooncake_ib_device:
                from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
                    get_ib_devices_for_gpu,
                )

                ib_device_for_gpu = get_ib_devices_for_gpu(
                    self.server_args.mooncake_ib_device, self.gpu_id
                )
                mooncake_ib_device = (
                    ib_device_for_gpu.split(",") if ib_device_for_gpu else []
                )
                try:
                    from mooncake import ep as mooncake_ep

                    mooncake_ep.set_device_filter(mooncake_ib_device)
                except:
                    pass  # A warning will be raised in `init_distributed_environment`

        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        if not self.server_args.enable_p2p_check:
            # 中译：若未开启 P2P 检查，则 monkey-patch 掉 P2P 访问检查（避免额外开销）。
            monkey_patch_p2p_access_check()

        # Allow external orchestrators (e.g. trainpi) to override the distributed
        # init method.  When set to "env://", torch uses MASTER_ADDR/MASTER_PORT
        # env-vars and an externally-created TCPStore, completely avoiding port
        # conflicts with intra-host collocation.
        # 中译：允许外部编排系统（如 trainpi）覆盖分布式初始化方法。当设为 "env://" 时，
        #       torch 使用 MASTER_ADDR/MASTER_PORT 环境变量与外部创建的 TCPStore，从而完全
        #       避免与同主机其他进程的端口冲突。
        dist_init_method_override = envs.SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE.get()
        if dist_init_method_override:
            dist_init_method = dist_init_method_override
        elif self.server_args.dist_init_addr:
            na = NetworkAddress.parse(self.server_args.dist_init_addr)
            dist_init_method = na.to_tcp()
        else:
            dist_init_method = NetworkAddress(
                self.server_args.host or "127.0.0.1", self.dist_port
            ).to_tcp()
        set_custom_all_reduce(not self.server_args.disable_custom_all_reduce)
        set_mscclpp_all_reduce(self.server_args.enable_mscclpp)
        set_torch_symm_mem_all_reduce(self.server_args.enable_torch_symm_mem)

        if not self.is_draft_worker:
            if self.device == "cpu":
                if _is_cpu_amx_available or _is_cpu_arm64:
                    # Bind OpenMP threads to CPU cores
                    # 中译：将 OpenMP 线程绑定到 CPU 核。
                    torch.ops.sgl_kernel.init_cpu_threads_env(self.local_omp_cpuid)

                    # Set local size to hint SGLang to use shared memory based AllReduce
                    # 中译：设置 LOCAL_SIZE 以提示 SGLang 使用基于共享内存的 AllReduce。
                    os.environ["LOCAL_SIZE"] = str(self.tp_size)
                    torch.ops.sgl_kernel.initialize(self.tp_size, self.tp_rank)

                else:
                    logger.warning(
                        "init_cpu_threads_env and shared memory based AllReduce is disabled, only intel amx backend and arm64 are supported"
                    )

            # Only initialize the distributed environment on the target model worker.
            # 中译：仅在 target（主模型）worker 上初始化分布式环境（草稿 worker 复用主模型的）。
            init_distributed_environment(
                backend=backend,
                world_size=self.tp_size * self.pp_size,
                rank=self.tp_size * self.pp_rank + self.tp_rank,
                local_rank=self.gpu_id,
                distributed_init_method=dist_init_method,
                timeout=self.server_args.dist_timeout,
                moe_a2a_backend=self.server_args.moe_a2a_backend,
                recovered_rank=self.server_args.elastic_ep_rejoin,
            )
            initialize_model_parallel(
                tensor_model_parallel_size=self.tp_size,
                attention_data_parallel_size=self.dp_size,
                pipeline_model_parallel_size=self.pp_size,
                expert_model_parallel_size=self.moe_ep_size,
                attention_context_model_parallel_size=self.attn_cp_size,
                moe_data_model_parallel_size=self.moe_dp_size,
                duplicate_tp_group=self.server_args.enable_pdmux,
                enable_symm_mem=self.server_args.enable_symm_mem,
                recovered_rank=self.server_args.elastic_ep_rejoin,
            )
            initialize_dp_attention(
                server_args=self.server_args,
                model_config=self.model_config,
            )
            if is_npu():
                register_sgl_tp_rank(self.gpu_id)

            # Pre-warm NCCL/RCCL to eliminate cold-start latency in first request
            # Controlled by --pre-warm-nccl flag (default: enabled on AMD GPUs)
            # 中译：预热 NCCL/RCCL，以消除首个请求的冷启动延迟。由 --pre-warm-nccl 控制
            #       （默认在 AMD GPU 上开启）。仅多卡（TP/PP/EP>1）时需要。
            if self.server_args.pre_warm_nccl and (
                self.tp_size > 1 or self.pp_size > 1 or self.moe_ep_size > 1
            ):
                warmup_start = time.perf_counter()
                tp_group_handle = get_tp_group().device_group

                # Single warmup all_reduce to initialize NCCL/RCCL communicator
                # 中译：通过一次 all_reduce 初始化（预热）NCCL/RCCL 通信器。
                warmup_tensor = torch.zeros(1, device=torch.cuda.current_device())
                dist.all_reduce(warmup_tensor, group=tp_group_handle)
                current_platform.synchronize()

                warmup_elapsed = time.perf_counter() - warmup_start
                logger.info(
                    f"NCCL/RCCL warmup completed in {warmup_elapsed:.3f}s "
                    f"(tp_size={self.tp_size}, pp_size={self.pp_size}, ep_size={self.moe_ep_size})"
                )

        pre_model_load_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=get_world_group().world_size > 1,
            cpu_group=get_world_group().cpu_group,
        )
        self.tp_group = get_tp_group()
        self.pp_group = get_pp_group()
        self.attention_tp_group = get_attention_tp_group()

        # Check memory for tensor parallelism
        # 中译：检查张量并行下各卡显存是否均衡。若某卡可用显存明显偏低（被其他进程占用），
        #       根据开关决定报错或警告。
        local_gpu_memory = get_available_gpu_memory(self.device, self.gpu_id)
        if self.tp_size > 1 and not self.is_draft_worker:
            if pre_model_load_memory < local_gpu_memory * 0.9:
                msg = "The memory capacity is unbalanced. Some GPUs may be occupied by other processes. "
                msg += f"{pre_model_load_memory=}, {local_gpu_memory=}, {local_gpu_memory * 0.9=}"
                if envs.SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK.get():
                    raise RuntimeError(msg)
                else:
                    logger.warning(msg)

        logger.info(
            f"Init torch distributed ends. elapsed={time.perf_counter() - tic:.2f} s, "
            f"mem usage={(before_avail_memory - local_gpu_memory):.2f} GB"
        )
        return pre_model_load_memory

    def init_shared_mooncake_transfer_engine(self):
        """
        Need MooncakeTransferEngine when:
        1) PD disaggregation uses mooncake for KV transfer (prefill/decode)
        2) HiCache uses mooncake storage backend
        3) Encoder disaggregation uses mooncake

        中译：以下场景需要 MooncakeTransferEngine：
        1）PD 分离使用 mooncake 传输 KV（prefill/decode）；
        2）HiCache 使用 mooncake 存储后端；
        3）编码器分离使用 mooncake。
        """
        use_mooncake_te = (
            (
                self.server_args.disaggregation_mode != "null"
                and self.server_args.disaggregation_transfer_backend == "mooncake"
            )
            or (
                self.server_args.enable_hierarchical_cache
                and self.server_args.hicache_storage_backend == "mooncake"
                and envs.SGLANG_HICACHE_MOONCAKE_REUSE_TE.get()
            )
            or (
                self.server_args.encoder_only
                and self.server_args.encoder_transfer_backend == "mooncake"
            )
            or (
                self.server_args.language_only
                and self.server_args.encoder_transfer_backend == "mooncake"
            )
            or (
                self.server_args.enable_elastic_expert_backup
                and self.server_args.elastic_ep_backend is not None
            )
        )

        if use_mooncake_te:
            from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
                init_mooncake_transfer_engine,
            )

            init_mooncake_transfer_engine(
                hostname=get_local_ip_auto(),
                gpu_id=self.gpu_id,
                ib_device=(
                    self.server_args.disaggregation_ib_device
                    or self.server_args.mooncake_ib_device
                ),
            )

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
            if torch.cuda.get_device_capability()[0] < 8:
                # 中译：计算能力低于 sm80，由于缺乏 bfloat16 支持，改用 float16。
                logger.info(
                    "Compute capability below sm80. Use float16 due to lack of bfloat16 support."
                )
                self.server_args.dtype = "float16"
                self.model_config.dtype = torch.float16
                if torch.cuda.get_device_capability()[1] < 5:
                    raise RuntimeError("SGLang only supports sm75 and above.")

        set_cuda_arch()

        # Prepare the model config
        # 中译：准备模型配置（包括 ModelOpt 量化配置与加载配置 LoadConfig）。
        from sglang.srt.configs.modelopt_config import ModelOptConfig

        modelopt_config = ModelOptConfig(
            quant=self.server_args.modelopt_quant,
            checkpoint_restore_path=self.server_args.modelopt_checkpoint_restore_path,
            checkpoint_save_path=self.server_args.modelopt_checkpoint_save_path,
            export_path=self.server_args.modelopt_export_path,
            quantize_and_serve=self.server_args.quantize_and_serve,
        )

        self.load_config = LoadConfig(
            load_format=self.server_args.load_format,
            download_dir=self.server_args.download_dir,
            model_loader_extra_config=self.server_args.model_loader_extra_config,
            tp_rank=self.tp_rank,
            remote_instance_weight_loader_seed_instance_ip=self.server_args.remote_instance_weight_loader_seed_instance_ip,
            remote_instance_weight_loader_seed_instance_service_port=self.server_args.remote_instance_weight_loader_seed_instance_service_port,
            remote_instance_weight_loader_send_weights_group_ports=self.server_args.remote_instance_weight_loader_send_weights_group_ports,
            remote_instance_weight_loader_backend=self.server_args.remote_instance_weight_loader_backend,
            remote_instance_weight_loader_transfer_engine=self.remote_instance_transfer_engine,
            remote_instance_weight_loader_transfer_engine_session_id=self.remote_instance_transfer_engine_session_id,
            modelexpress_url=self.server_args.modelexpress_url,
            modelexpress_transport=self.server_args.modelexpress_transport,
            modelopt_config=modelopt_config,
            rl_quant_profile=self.server_args.rl_quant_profile,
            draft_model_idx=self.draft_model_idx,
        )
        if self.device == "cpu":
            self.model_config = adjust_config_with_unaligned_cpu_tp(
                self.model_config, self.load_config, self.tp_size
            )

        if (
            self.server_args.load_format == LoadFormat.REMOTE_INSTANCE
            and self.server_args.remote_instance_weight_loader_backend
            == RemoteInstanceWeightLoaderBackend.NCCL
        ):
            if self.tp_rank == 0:
                instance_ip = NetworkAddress.resolve_host(socket.gethostname())
                t = threading.Thread(
                    target=trigger_init_weights_send_group_for_remote_instance_request,
                    args=(
                        self.server_args.remote_instance_weight_loader_seed_instance_ip,
                        self.server_args.remote_instance_weight_loader_seed_instance_service_port,
                        self.server_args.remote_instance_weight_loader_send_weights_group_ports,
                        instance_ip,
                    ),
                )
                t.start()

        # Load the model
        # Remove monkey_patch when linear.py quant remove dependencies with vllm
        # 中译：加载模型。这里临时 monkey-patch vllm 的并行状态（待 linear.py 的量化不再
        #       依赖 vllm 后可移除）。
        monkey_patch_vllm_parallel_state()

        enable_cpu_backup = self.server_args.enable_weights_cpu_backup or (
            self.is_draft_worker and self.server_args.enable_draft_weights_cpu_backup
        )
        with self.memory_saver_adapter.region(
            GPU_MEMORY_TYPE_WEIGHTS,
            enable_cpu_backup=enable_cpu_backup,
        ):
            self.loader = get_model_loader(
                load_config=self.load_config,
                model_config=self.model_config,
            )
            self.model = self.loader.load_model(
                model_config=self.model_config,
                device_config=DeviceConfig(self.device, self.gpu_id),
            )
            if hasattr(self.loader, "remote_instance_transfer_engine_weight_info"):
                self.remote_instance_transfer_engine_weight_info = (
                    self.loader.remote_instance_transfer_engine_weight_info
                )
        # Cache needs to be cleared after loading model weights (in the self.loader.load_model function).
        # To avoid conflict with memory_saver_adapter.region, empty_cache operation is now moved here.
        # 中译：加载权重后需清理缓存。为避免与 memory_saver_adapter.region 冲突，
        #       empty_cache 操作被移到这里。
        if _is_npu:
            torch.npu.empty_cache()
        monkey_patch_vllm_parallel_state(reverse=True)

        if not self.is_draft_worker:
            # 中译：非草稿 worker 在加载完后执行 offloader 的后置初始化。
            get_offloader().post_init()

        # Register model for layerwise NVTX profiling if enabled
        # 中译：若启用逐层 NVTX profiling，为模型注册相应 hook。
        if self.server_args.enable_layerwise_nvtx_marker:
            pyt_hooks = PytHooks()
            pyt_hooks.register_hooks(self.model, module_prefix="model")

        if self.server_args.kv_cache_dtype == "fp8_e4m3":
            if self.server_args.quantization_param_path is not None:
                if callable(getattr(self.model, "load_kv_cache_scales", None)):
                    self.model.load_kv_cache_scales(
                        self.server_args.quantization_param_path
                    )
                    logger.info(
                        "Loaded KV cache scaling factors from %s",
                        self.server_args.quantization_param_path,
                    )
                else:
                    raise RuntimeError(
                        "Using FP8 KV cache and scaling factors provided but "
                        "model %s does not support loading scaling factors.",
                        self.model.__class__,
                    )
            else:
                logger.warning(
                    "Using FP8 KV cache but no scaling factors "
                    "provided. Defaulting to scaling factors of 1.0. "
                    "This may lead to less accurate results!"
                )

        # Parse other args
        self.sliding_window_size = None
        if hasattr(self.model, "get_attention_sliding_window_size"):
            self.sliding_window_size = self.model.get_attention_sliding_window_size()
        elif (
            self.model_config.is_hybrid_swa
            and self.model_config.sliding_window_size is not None
        ):
            # sliding window field in model config may have different meaning for different kinds of models (e.g., dllm), here we only consider the sliding window in SWA model
            # 中译：模型配置中的滑动窗口字段对不同类型模型含义可能不同（如 dllm），
            #       这里仅考虑 SWA 模型中的滑动窗口。
            self.sliding_window_size = self.model_config.sliding_window_size
        elif self.model_config.attention_chunk_size is not None:
            self.sliding_window_size = self.model_config.attention_chunk_size
            logger.info(
                f"Setting sliding_window_size to be attention_chunk_size: {self.sliding_window_size}"
            )

        self.dtype = self.model_config.dtype

        after_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        self.weight_load_mem_usage = before_avail_memory - after_avail_memory
        # Get quantization config from ModelConfig
        # This handles both config.json (standard) and hf_quant_config.json (ModelOpt)
        # 中译：从 ModelConfig 获取量化配置（同时兼容标准的 config.json 与 ModelOpt 的
        #       hf_quant_config.json），仅用于日志展示。
        quant_str = self.model_config.get_quantization_config_log_str()

        logger.info(
            f"Load weight end. "
            f"elapsed={time.perf_counter() - tic_total:.2f} s, "
            f"type={type(self.model).__name__}, "
            f"{quant_str + ', ' if quant_str else ''}"
            f"avail mem={after_avail_memory:.2f} GB, "
            f"mem usage={self.weight_load_mem_usage:.2f} GB."
        )

        # TODO: Make sure all models have `quant_config` attribute, and all online quantization methods register which layers they actually quantize.
        # TODO: Move this online-quantization reporting out of ModelRunner.
        # 中译：TODO：确保所有模型都有 `quant_config` 属性，且所有在线量化方法都注册它们
        #       实际量化了哪些层。TODO：把这段在线量化的统计上报从 ModelRunner 中移出。
        quantized_layers = getattr(
            getattr(self.model, "quant_config", None), "quantized_layers", None
        )
        if (
            hasattr(self.model, "quant_config")
            and hasattr(self.model.quant_config, "quantized_layers")
            and self.server_args.quantization is not None
        ):
            type_counts, quantized_layers_count = (
                self.model.quant_config.quantized_layers
            )
            type_summary = ", ".join(f"{t}: {c}" for t, c in type_counts.items())
            logger.info(
                f"Online {self.server_args.quantization} quantization: quantized {quantized_layers_count} layers in total ({type_summary})."
            )

        if self.server_args.debug_tensor_dump_output_folder is not None:
            dump_folder = self.server_args.debug_tensor_dump_output_folder
            if self.spec_algorithm.is_eagle():
                role = "draft" if self.is_draft_worker else "target"
                dump_folder = os.path.join(dump_folder, role)
            register_forward_hook_for_model(
                self.model,
                dump_folder,
                self.server_args.debug_tensor_dump_layers,
                self.tp_size,
                self.tp_rank,
                self.pp_rank,
            )

        if dumper.may_enable:
            dumper.apply_source_patches()
            dumper.register_non_intrusive_dumper(self.model)

        # Pre-expand RoPE cache before CUDA Graph capture
        # 中译：在 CUDA Graph 捕获之前预先扩展 RoPE 缓存（以支持长序列）。
        reserve_rope_cache_for_long_sequences(
            self.model,
            self.server_args,
            self.model_config,
            logger,
        )

        if self.server_args.elastic_ep_backend == "mooncake":
            # Mooncake does not support `monitored_barrier`
            # 中译：Mooncake 不支持 `monitored_barrier`，改用普通 barrier。
            dist.barrier(group=get_tp_group().cpu_group)
        else:
            # Handle the case where some ranks do not finish loading.
            # 中译：处理部分 rank 未完成加载的情况（用带超时的 monitored_barrier 捕捉掉队者）。
            try:
                dist.monitored_barrier(
                    group=get_tp_group().cpu_group,
                    timeout=datetime.timedelta(
                        seconds=UNBALANCED_MODEL_LOADING_TIMEOUT_S
                    ),
                    wait_all_ranks=True,
                )
            except RuntimeError:
                raise ValueError(
                    f"TP rank {self.tp_rank} could finish the model loading, but there are other ranks that didn't finish loading. It is likely due to unexpected failures (e.g., OOM) or a slow node."
                ) from None

    def _prepare_moe_topk(self):
        # 中译：为启用了 DeepEP waterfill 的 MoE TopK 模块准备负载均衡器（balancer）。
        balancer_cls = None
        num_prepared = 0
        num_routed_experts = None
        for module in self.model.modules():
            if not isinstance(module, (TopK, HashTopK)):
                continue
            if (
                not module.enable_deepep_waterfill
                or module.deepep_waterfill_balancer is not None
            ):
                continue
            if num_routed_experts is None:
                num_routed_experts = getattr(
                    self.model_config.hf_config, "n_routed_experts", None
                )
                if num_routed_experts is None:
                    raise ValueError(
                        "DeepEP waterfill requires model config n_routed_experts."
                    )
            if balancer_cls is None:
                from sglang.srt.layers.moe.deepep_waterfill import (
                    DeepEPWaterfillBalancer,
                )

                balancer_cls = DeepEPWaterfillBalancer
            # Static EPLB remaps TopK ids to physical expert ids before Waterfill.
            # Redundant experts therefore need to be included in the per-rank
            # expert count used for Waterfill's shared-expert slot remapping.
            # 中译：静态 EPLB 会在 Waterfill 之前将 TopK id 重映射为物理专家 id，因此凗余
            #       专家也需计入 Waterfill 共享专家槽位重映射所用的每 rank 专家数。
            num_physical_routed_experts = (
                num_routed_experts + self.server_args.ep_num_redundant_experts
            )
            if isinstance(module, TopK):
                routed_scaling_factor = module.topk_config.routed_scaling_factor
            else:
                routed_scaling_factor = module.routed_scaling_factor
            module.deepep_waterfill_balancer = balancer_cls(
                num_routed_experts=num_physical_routed_experts,
                world_size=self.moe_ep_size,
                rank=self.moe_ep_rank,
                layer_id=module.layer_id,
                routed_scaling_factor=(
                    routed_scaling_factor if routed_scaling_factor is not None else 1.0
                ),
            )
            num_prepared += 1
        if num_prepared:
            log_info_on_rank0(
                logger, f"Prepared {num_prepared} DeepEP waterfill TopK modules."
            )

    def update_expert_location(
        self,
        new_expert_location_metadata: ExpertLocationMetadata,
        update_layer_ids: List[int],
    ):
        # 中译：更新（重布）专家位置。若某些逻辑专家无法通过 P2P 获得（p2p_missing），
        #       则从磁盘/DRAM 备份重新加载这些缺失的专家权重。
        p2p_missing_logical_experts = self.expert_location_updater.update(
            self.model.routed_experts_weights_of_layer,
            new_expert_location_metadata,
            update_layer_ids=update_layer_ids,
            nnodes=self.server_args.nnodes,
            rank=self.tp_rank,
        )

        if len(p2p_missing_logical_experts) > 0:
            # Load the missing expert weights from disk
            # 中译：从磁盘加载缺失的专家权重。
            if callable(getattr(self.model, "generate_weight_name_filter", None)):
                # Filter and load only missing expert weights
                # 中译：仅过滤并加载缺失的专家权重。
                weight_name_filter = self.model.generate_weight_name_filter(
                    p2p_missing_logical_experts
                )
            else:
                # Do a full reload from disk/DRAM
                # 中译：模型未实现 generate_weight_name_filter，退而从磁盘/DRAM 做一次全量重载。
                logger.info(
                    "[Elastic EP] Model does not implement generate_weight_name_filter. "
                    "Performing full weight reload."
                )
                weight_name_filter = None

            if (
                self.expert_backup_client is not None
                and self.expert_backup_client.use_backup
            ):
                # Load the missing weights from the DRAM backup
                self.expert_backup_client.update_weights(weight_name_filter)
            else:
                # Load the missing weights from disk
                self.update_weights_from_disk(
                    get_global_server_args().model_path,
                    get_global_server_args().load_format,
                    weight_name_filter=weight_name_filter,
                )

    def maybe_recover_ep_ranks(self):
        # 中译：（弹性 EP）尝试恢复失效的 rank。检查是否有不活跃的 rank，若有则尝试恢复，
        #       并重置前向计数器、重广播专家位置元数据与随机种子。
        # TODO(perf): `active_ranks.all()` on a CUDA tensor triggers host-device
        # synchronization, and this function is on the forward-path.
        # This check only runs when `--elastic-ep-backend` is enabled, so the
        # synchronization overhead does not propagate to other configs.
        # Leave for future optimization of the elastic EP path.
        # 中译：TODO(性能)：在 CUDA 张量上调用 `active_ranks.all()` 会触发 host-device 同步，
        #       而本函数位于前向路径上。该检查仅在启用 --elastic-ep-backend 时运行，故同步
        #       开销不会波及其他配置。留作弹性 EP 路径未来的优化。
        if self.tp_group.active_ranks.all() and self.tp_group.active_ranks_cpu.all():
            return

        tp_active_ranks = self.tp_group.active_ranks.detach().cpu().numpy()
        tp_active_ranks_cpu = self.tp_group.active_ranks_cpu.detach().numpy()
        tp_active_ranks &= tp_active_ranks_cpu
        # NOTE: `ranks_to_recover` uses indices in `tp_group`. For the current
        # Mooncake elastic EP implementation we assume `--pp-size=1`, so the
        # tp-group index is the same as the global rank index.
        # 中译：`ranks_to_recover` 使用的是 tp_group 内的索引。当前 Mooncake 弹性 EP 实现
        #       假设 --pp-size=1，故 tp-group 索引与全局 rank 索引一致。
        ranks_to_recover = [
            i for i in range(len(tp_active_ranks)) if not tp_active_ranks[i]
        ]

        # try_recover_ranks polls peer state via Mooncake EP backend.
        # Mooncake's internal semantics guarantee that all ranks observe
        # consistent peer readiness state, so collective operations below
        # are safe even though polling appears local.
        # 中译：try_recover_ranks 通过 Mooncake EP 后端轮询对端状态。Mooncake 的内部语义保证
        #       所有 rank 观察到一致的对端就绪状态，因此即使轮询看似本地，下面的集体通信仍安全。
        if ranks_to_recover and try_recover_ranks(ranks_to_recover):
            self.forward_pass_id = 0
            self.eplb_manager.reset_generator()
            broadcast_global_expert_location_metadata(
                src_rank=self._get_healthy_expert_location_src_rank(
                    invoked_in_elastic_ep_rejoin_path=False
                )
            )
            ElasticEPStateManager.instance().reset()

            broadcast_pyobj(
                [self.server_args.random_seed],
                get_world_group().rank,
                get_world_group().cpu_group,
                src=get_world_group().ranks[0],
            )
            logger.info(f"recover ranks {ranks_to_recover} done")

    def _get_healthy_expert_location_src_rank(
        self, invoked_in_elastic_ep_rejoin_path: bool
    ) -> int:
        # 中译：选出一个“健康”的 rank 作为广播专家位置元数据的源 rank。
        world_group = get_world_group()
        # NOTE: do not key off `self.server_args.elastic_ep_rejoin` here.
        # A rank that was started as a rejoin rank may later act as a healthy
        # rank in a subsequent recovery cycle.
        # 中译：不要以 `self.server_args.elastic_ep_rejoin` 作为依据。一个以 rejoin 身份启动的
        #       rank，在后续的恢复周期中可能会扮演健康 rank 的角色。
        local_rejoin_flag = bool(invoked_in_elastic_ep_rejoin_path)
        gathered_rejoin_flags = world_group.all_gather_object(local_rejoin_flag)

        for rank_in_group, is_rejoin_rank in enumerate(gathered_rejoin_flags):
            if not is_rejoin_rank:
                return world_group.ranks[rank_in_group]

        raise RuntimeError(
            "No healthy rank found for broadcasting expert location metadata. "
            "All ranks are marked as elastic_ep_rejoin."
        )

    def update_weights_from_disk(
        self,
        model_path: str,
        load_format: str,
        weight_name_filter: Optional[Callable[[str], bool]] = None,
        recapture_cuda_graph: bool = False,
    ) -> tuple[bool, str]:
        """Update engine weights in-place from the disk.

        中译：从磁盘原地（in-place）在线更新引擎权重。若加载失败会回滚到原权重。
        """
        logger.info(
            f"Update engine weights online from disk begin. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id, empty_cache=False):.2f} GB"
        )

        target_device = torch.device(self.device)
        self.model_config.model_path = model_path
        load_config = LoadConfig(load_format=load_format)

        # Only support DefaultModelLoader for now
        # 中译：目前仅支持 DefaultModelLoader。
        loader = get_model_loader(load_config, self.model_config)
        if not isinstance(loader, DefaultModelLoader):
            message = f"Failed to get model loader: {loader}."
            return False, message

        def get_weight_iter(config):
            iter = loader._get_weights_iterator(
                DefaultModelLoader.Source.init_new(config, self.model)
            )
            if weight_name_filter is not None:
                iter = (
                    (name, weight) for name, weight in iter if weight_name_filter(name)
                )

            return iter

        def model_load_weights(model, iter):
            loader.load_weights_and_postprocess(model, iter, target_device)
            return model

        with set_default_torch_dtype(self.model_config.dtype):
            try:
                iter = get_weight_iter(self.model_config)
            except Exception as e:
                message = f"Failed to get weights iterator: {e}."
                return False, message
            try:
                model = model_load_weights(self.model, iter)
            except Exception as e:
                message = (
                    f"Failed to update weights: {e}.\nRolling back to original weights."
                )
                del iter
                gc.collect()
                iter = get_weight_iter(self.model_config)
                self.model = model_load_weights(self.model, iter)
                return False, message

        self.model = model
        self.server_args.model_path = model_path
        self.server_args.load_format = load_format
        self.load_config = load_config

        if recapture_cuda_graph and (
            self.device == "cuda"
            or self.device == "musa"
            or (
                current_platform.is_out_of_tree()
                and current_platform.support_cuda_graph()
            )
        ):
            # 中译：若需要（且设备支持），重新捕获解码阶段的 CUDA Graph（因为权重已变）。
            self.init_decode_cuda_graph()

        logger.info("Update weights end.")
        return True, "Succeeded to update model weights."

    def init_weights_send_group_for_remote_instance(
        self,
        master_address,
        ports,
        group_rank,
        world_size,
        group_name,
        backend="nccl",
    ):
        # 中译：为“向远程实例发送权重”初始化一个自定义进程组（每个 tp_rank 一个端口）。
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        ports_list = ports.split(",")
        assert (
            len(ports_list) == self.tp_size
        ), f"Expected {self.tp_size} ports, but got {len(ports_list)} ports."
        group_port = ports_list[self.tp_rank]
        group_name = f"{group_name}_{group_port}_{self.tp_rank}"

        logger.info(
            f"init custom process group: tp_rank={self.tp_rank}, gpu_id={self.gpu_id}, master_address={master_address}, master_port={group_port}, "
            f"group_rank={group_rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        current_platform.empty_cache()
        success = False
        message = ""
        try:
            na = NetworkAddress(master_address, group_port)
            self._weights_send_group[group_name] = init_custom_process_group(
                backend=backend,
                init_method=na.to_tcp(),
                world_size=world_size,
                rank=group_rank,
                group_name=group_name,
                device_id=torch.device("cuda", self.gpu_id),
            )
            dist.barrier(group=self._weights_send_group[group_name])
            success = True
            message = f"Succeeded to init group through {na.to_host_port_str()} group."
        except Exception as e:
            message = f"Failed to init group: {e}."
            logger.error(message)

        current_platform.empty_cache()
        return success, message

    def send_weights_to_remote_instance(
        self,
        master_address,
        ports,
        group_name,
    ):
        # 中译：通过之前初始化的发送组，将本模型所有参数 broadcast 给远程实例；
        #       发送完成后销毁该进程组。
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        ports_list = ports.split(",")
        assert (
            len(ports_list) == self.tp_size
        ), f"Expected {self.tp_size} ports, but got {len(ports_list)} ports."
        group_port = ports_list[self.tp_rank]
        group_name = f"{group_name}_{group_port}_{self.tp_rank}"

        if self._weights_send_group[group_name] is not None:
            send_group = self._weights_send_group[group_name]
        else:
            # 中译：发送组不存在，提示需先调用 init_weights_send_group_for_remote_instance。
            message = f"Group {group_name} not in _weights_send_group list. Please call `init_weights_send_group_for_remote_instance` first."
            logger.error(message)
            return False, message

        current_platform.empty_cache()
        success = False
        na = NetworkAddress(master_address, group_port)
        message = ""
        try:
            for _, weights in self.model.named_parameters():
                torch.distributed.broadcast(
                    weights,
                    src=0,
                    group=send_group,
                )
            success = True
            message = f"Succeeded to send weights through {na.to_host_port_str()} {group_name}."
        except Exception as e:
            message = f"Failed to send weights: {e}."
            logger.error(message)

        # destroy the process group after sending weights
        # 中译：发送权重完成后销毁该进程组。
        del self._weights_send_group[group_name]
        torch.distributed.distributed_c10d.destroy_process_group(send_group)
        current_platform.empty_cache()
        return success, message

    def init_weights_update_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
    ):
        """Initialize the Torch process group for model parameter updates.

        `_model_update_group` is used in the RLHF workflow, where rank
        0 is the actor model in the training engine, and the other ranks are
        the inference engine, which is used for rollout.

        In the RLHF workflow, the training engine updates the model
        weights/parameters online, and broadcasts them to the inference
        engine through the `_model_update_group` process group.

        中译：初始化用于模型参数更新的 Torch 进程组。
        `_model_update_group` 用于 RLHF 流程：rank 0 是训练引擎中的 actor 模型，
        其余 rank 是用于 rollout 的推理引擎。训练引擎在线更新权重后，通过该进程组
        将权重广播给推理引擎。
        """
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        rank = rank_offset + self.tp_rank

        logger.info(
            f"init custom process group: master_address={master_address}, master_port={master_port}, "
            f"rank_offset={rank_offset}, rank={rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        try:
            na = NetworkAddress(master_address, master_port)
            self._model_update_group[group_name] = init_custom_process_group(
                backend=backend,
                init_method=na.to_tcp(),
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )
            return True, "Succeeded to initialize custom process group."
        except Exception as e:
            message = f"Failed to initialize custom process group: {e}."
            logger.error(message)
            return False, message

    def destroy_weights_update_group(self, group_name):
        # 中译：销毁指定的权重更新进程组。
        try:
            if group_name in self._model_update_group:
                pg = self._model_update_group.pop(group_name)
                torch.distributed.destroy_process_group(pg)
                return True, "Succeeded to destroy custom process group."
            else:
                return False, "The group to be destroyed does not exist."
        except Exception as e:
            message = f"Failed to destroy custom process group: {e}."
            logger.error(message)
            return False, message

    def update_weights_from_distributed(
        self,
        names,
        dtypes,
        shapes,
        group_name,
        load_format: Optional[str] = None,
    ):
        """
        Update specific parameter in the model weights online
        through `_model_update_group` process group.

        Args:
            name: the name of the parameter to be updated.
            dtype: the data type of the parameter to be updated.
            shape: the shape of the parameter to be updated.

        中译：通过 `_model_update_group` 进程组在线更新模型权重中的指定参数。
        参数：name/dtype/shape 分别为待更新参数的名称/数据类型/形状（均为列表）。
        实现上通过 broadcast（src=0）从训练端拉取权重。
        """

        assert group_name in self._model_update_group, (
            f"Group {group_name} not in {list(self._model_update_group.keys())}. "
            "Please call `init_weights_update_group` first."
        )

        if load_format == "flattened_bucket":
            return self._update_bucketed_weights_from_distributed(
                names, dtypes, shapes, group_name
            )
        try:
            weights = []
            handles = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                weight = torch.empty(shape, dtype=target_dtype, device=self.device)
                handles.append(
                    torch.distributed.broadcast(
                        weight,
                        src=0,
                        group=self._model_update_group[group_name],
                        async_op=True,
                    )
                )
                weights.append((name, weight))
            for handle in handles:
                handle.wait()

            self.model.load_weights(weights)
            return True, "Succeeded to update parameter online."

        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    def _update_bucketed_weights_from_distributed(
        self, names, dtypes, shapes, group_name
    ):
        # 中译：以“扁平化分桶（flattened bucket）”格式从分布式组更新权重：先把多个张量拼成
        #       一个扁平化大张量一次 broadcast（减少通信次数），再重建回各张量并加载。
        try:
            named_tensors = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                named_tensors.append(
                    (name, torch.empty(shape, dtype=target_dtype, device=self.device))
                )
            bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            flattened_tensor = bucket.get_flattened_tensor()
            torch.distributed.broadcast(
                flattened_tensor,
                src=0,
                group=self._model_update_group[group_name],
            )
            reconstructed_tensors = bucket.reconstruct_tensors()
            self.model.load_weights(reconstructed_tensors)
            return True, f"Succeeded to update parameter online."
        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, Union[torch.Tensor, LocalSerializedTensor]]],
        load_format: Optional[str] = None,
    ):
        # 中译：直接从（可能跨进程序列化的）张量更新权重。根据 load_format 选择不同加载
        #       路径（flattened_bucket / direct / 自定义 loader / 默认）。
        monkey_patch_torch_reductions()
        if load_format == "flattened_bucket":
            # Handle flattened bucket format
            # 中译：处理扁平化分桶格式。
            return self._update_weights_from_flattened_bucket(
                flattened_tensor_bucket_dict=named_tensors
            )

        # We need to get device after patch otherwise the device would be wrong
        # 中译：必须在 patch 之后获取设备，否则设备会不正确。
        device_module = torch.get_device_module(self.device)
        infered_device = device_module.current_device()

        named_tensors = [
            (name, _unwrap_tensor(tensor, tp_rank=self.tp_rank, device=infered_device))
            for name, tensor in named_tensors
        ]
        if load_format == "direct":
            _model_load_weights_direct(self.model, named_tensors)
        elif load_format in self.server_args.custom_weight_loader:
            custom_loader = dynamic_import(load_format)
            custom_loader(self.model, named_tensors)
        elif load_format is None:
            self.model.load_weights(named_tensors)
        else:
            raise NotImplementedError(f"Unknown load_format={load_format}")
        return True, "Success"

    def _update_weights_from_flattened_bucket(
        self,
        flattened_tensor_bucket_dict,
    ):
        """Handle flattened bucket format for weight updates

        中译：处理扁平化分桶格式的权重更新（从扁平化大张量 + 元数据重建出各张量并加载）。
        """
        flattened_tensor = flattened_tensor_bucket_dict["flattened_tensor"]
        metadata = flattened_tensor_bucket_dict["metadata"]

        # Convert metadata dict to our format
        converted_metadata = []
        for meta in metadata:
            converted_meta = FlattenedTensorMetadata(
                name=meta.name,
                shape=meta.shape,
                dtype=meta.dtype,
                start_idx=meta.start_idx,
                end_idx=meta.end_idx,
                numel=meta.numel,
            )
            converted_metadata.append(converted_meta)

        # Create bucket and reconstruct tensors
        bucket = FlattenedTensorBucket(
            flattened_tensor=flattened_tensor, metadata=converted_metadata
        )
        reconstructed_tensors = bucket.reconstruct_tensors()

        # Load the reconstructed tensors using the standard method
        self.model.load_weights(reconstructed_tensors)

        return True, "Success"

    def get_weights_by_name(
        self, name: str, truncate_size: int = 100
    ) -> Optional[torch.Tensor]:
        """Get the weights of the parameter by its name. Similar to `get_parameter` in Hugging Face.

        Only used for unit test with an unoptimized performance.
        For optimized performance, please use torch.save and torch.load.

        中译：按参数名获取权重（类似 HuggingFace 的 `get_parameter`）。性能未优化，仅用于
        单测；追求性能请使用 torch.save / torch.load。
        """
        # TODO: (chenyang) Add support for Qwen models.
        # 中译：TODO：（chenyang）增加对 Qwen 系列模型的支持。
        try:
            return self.model.get_weights_by_name(
                name, truncate_size, tp_size=self.tp_size
            )
        except Exception as e:
            logger.error(f"Error when getting parameter {name}: {e}")
            return None

    def init_lora_manager(self):
        # 中译：初始化 LoRA 管理器（管理多个 LoRA 适配器的加载/卸载与批内路由）。
        self.lora_manager = LoRAManager(
            base_model=self.model,
            base_hf_config=self.model_config.hf_config,
            max_loras_per_batch=self.server_args.max_loras_per_batch,
            load_config=self.load_config,
            dtype=self.dtype,
            server_args=self.server_args,
            lora_backend=self.server_args.lora_backend,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            max_lora_rank=self.server_args.max_lora_rank,
            target_modules=self.server_args.lora_target_modules,
            lora_paths=self.server_args.lora_paths,
        )

    def _init_lora_cuda_graph_moe_buffers(self):
        """Phase 1 of LoRA CUDA graph init: pre-allocate MoE intermediate buffers.

        Must be called before init_memory_pool() so that memory profiling
        sees the reduced available memory and sizes KV cache correctly.
        All MoE LoRA layers share one set of buffers (managed by the
        lora_backend) since they execute sequentially during forward.

        Phase 2 (dense LoRA batch metadata) is handled later in
        CudaGraphRunner.__init__() via lora_manager.init_cuda_graph_batch_info(),
        because it needs capture-time parameters (max_bs, num_tokens_per_bs)
        that are only available at that stage.

        中译：LoRA CUDA Graph 初始化的第一阶段：预分配 MoE 中间缓冲区。必须在
        init_memory_pool() 之前调用，以便内存 profiling 看到减少后的可用内存并正确为
        KV 缓存定尺寸。所有 MoE LoRA 层共享一套缓冲区（由 lora_backend 管理），因为
        它们在前向时是串行执行的。第二阶段（稠密 LoRA 的 batch 元数据）稍后在
        CudaGraphRunner.__init__() 中处理，因为它需要捕获时的参数（max_bs、num_tokens_per_bs）。
        """
        from sglang.srt.lora.layers import FusedMoEWithLoRA

        max_bs = self.server_args.cuda_graph_config.decode.max_bs
        max_loras = self.server_args.max_loras_per_batch
        for module in self.model.modules():
            if isinstance(module, FusedMoEWithLoRA):
                self.lora_manager.init_cuda_graph_moe_buffers(
                    max_bs, max_loras, self.dtype, module
                )
                logger.info(
                    f"Pre-allocated shared MoE LoRA CUDA graph buffers "
                    f"(max_bs={max_bs}, max_loras={max_loras})"
                )
                break

    def load_lora_adapter(self, lora_ref: LoRARef):
        """Load a new lora adapter from disk or huggingface.

        中译：从磁盘或 HuggingFace 加载一个新的 LoRA 适配器。
        """

        logger.info(
            f"LoRA adapter loading starts: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        result = self.lora_manager.load_lora_adapter(lora_ref)

        logger.info(
            f"LoRA adapter loading completes: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        return result

    def load_lora_adapter_from_tensors(
        self, lora_ref: LoRARef, tensors, config_dict, added_tokens_config=None
    ):
        logger.info(f"LoRA adapter loading from tensors starts: {lora_ref}.")
        result = self.lora_manager.load_lora_adapter_from_tensors(
            lora_ref, tensors, config_dict, added_tokens_config
        )
        logger.info(f"LoRA adapter loading from tensors completes: {lora_ref}.")
        return result

    def unload_lora_adapter(self, lora_ref: LoRARef):
        """Unload a lora adapter that was previously loaded during initialization or dynamic loading.

        中译：卸载一个之前（初始化或动态加载时）加载的 LoRA 适配器。
        """

        logger.info(
            f"LoRA adapter unloading starts: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        result = self.lora_manager.unload_lora_adapter(lora_ref)

        logger.info(
            f"LoRA adapter unloading completes: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        return result

    @property
    def qwen3_next_config(self):
        config = self.model_config.hf_config
        if isinstance(config, Qwen3NextConfig):
            return config
        return None

    @property
    def hybrid_lightning_config(self):
        config = self.model_config.hf_config
        if isinstance(config, BailingHybridConfig):
            return config
        return None

    @property
    def hybrid_gdn_config(self):
        config = self.model_config.hf_config.get_text_config()
        if isinstance(
            config,
            Qwen3NextConfig
            | Qwen3_5Config
            | Qwen3_5MoeConfig
            | InternS2PreviewConfig
            | JetNemotronConfig
            | JetVLMConfig,
        ):
            return config
        return None

    @property
    def mamba2_config(self):
        config = self.model_config.hf_config
        if isinstance(config, NemotronHConfig) and self.is_draft_worker:
            # NemotronH MTP draft models have no Mamba layers (pattern like "*E")
            # so they shouldn't use HybridLinearAttnBackend
            # 中译：NemotronH 的 MTP 草稿模型没有 Mamba 层（pattern 形如 "*E"），故不应使用
            #       HybridLinearAttnBackend。
            pattern = getattr(config, "mtp_hybrid_override_pattern", None)
            if pattern is not None and "M" not in pattern:
                return None
        if isinstance(
            config,
            FalconH1Config
            | NemotronHConfig
            | Lfm2Config
            | Lfm2MoeConfig
            | Lfm2VlConfig
            | ZayaConfig,
        ):
            return config
        if isinstance(config, NemotronH_Nano_VL_V2_Config):
            return config.llm_config

        if isinstance(config, GraniteMoeHybridConfig):
            has_mamba = any(
                layer_type == "mamba"
                for layer_type in getattr(config, "layer_types", [])
            )
            if not has_mamba:
                return None
            else:
                return config

        return None

    @property
    def max_token_pool_size(self):
        """Return the max token pool size considering hybrid swa settings.

        中译：返回考虑了混合 SWA 设置后的最大 token 池大小。
        """
        if self.is_hybrid_swa:
            return self.full_max_total_num_tokens
        else:
            return self.max_total_num_tokens

    @property
    def kimi_linear_config(self):
        config = self.model_config.hf_config
        if isinstance(config, KimiLinearConfig):
            return config
        return None

    def _get_linear_attn_registry_result(self):
        if self._linear_attn_registry_cache is _UNSET:
            self._linear_attn_registry_cache = get_linear_attn_config(
                self.model_config.hf_config
            )
        return self._linear_attn_registry_cache

    @property
    def linear_attn_model_spec(self):
        result = self._get_linear_attn_registry_result()
        return result[0] if result else None

    @property
    def mambaish_config(self):
        existing = (
            self.mamba2_config
            or self.hybrid_gdn_config
            or self.kimi_linear_config
            or self.hybrid_lightning_config
        )
        if existing:
            return existing
        result = self._get_linear_attn_registry_result()
        return result[1] if result else None

    def configure_kv_cache_dtype(self):
        # 中译：根据 --kv-cache-dtype 与模型量化配置，确定 KV 缓存的实际数据类型
        #       （auto/fp8_e5m2/fp8_e4m3/bf16/fp4_e2m1 等），HIP 与非 HIP 平台取用不同的 fp8 表示。
        if self.server_args.kv_cache_dtype == "auto":
            quant_config = getattr(self.model, "quant_config", None)
            kv_cache_quant_algo = getattr(quant_config, "kv_cache_quant_algo", None)
            if (
                isinstance(kv_cache_quant_algo, str)
                and kv_cache_quant_algo.upper() == "FP8"
            ):
                if _is_hip:
                    self.kv_cache_dtype = fp8_dtype
                    self.server_args.kv_cache_dtype = TORCH_DTYPE_TO_KV_CACHE_STR[
                        self.kv_cache_dtype
                    ]
                else:
                    self.kv_cache_dtype = torch.float8_e4m3fn
                    self.server_args.kv_cache_dtype = TORCH_DTYPE_TO_KV_CACHE_STR[
                        self.kv_cache_dtype
                    ]
            else:
                self.kv_cache_dtype = self.dtype
        elif self.server_args.kv_cache_dtype == "fp8_e5m2":
            if _is_hip:  # Using natively supported format
                self.kv_cache_dtype = fp8_dtype
            else:
                self.kv_cache_dtype = torch.float8_e5m2
        elif self.server_args.kv_cache_dtype == "fp8_e4m3":
            if _is_hip:  # Using natively supported format
                self.kv_cache_dtype = fp8_dtype
            else:
                self.kv_cache_dtype = torch.float8_e4m3fn
        elif self.server_args.kv_cache_dtype in ("bf16", "bfloat16"):
            self.kv_cache_dtype = torch.bfloat16
        elif self.server_args.kv_cache_dtype == "fp4_e2m1":
            if hasattr(torch, "float4_e2m1fn_x2"):
                self.kv_cache_dtype = torch.float4_e2m1fn_x2
                logger.warning(f"FP4 (E2M1) KV Cache might lead to a accuracy drop!")
            else:
                logger.warning(
                    f"--kv-cache-dtype falls back to 'auto' because this torch version does not support torch.float4_e2m1fn_x2"
                )
                self.kv_cache_dtype = self.dtype
        else:
            raise ValueError(
                f"Unsupported kv_cache_dtype: {self.server_args.kv_cache_dtype}."
            )

    def init_cublas(self):
        """We need to run a small matmul to init cublas. Otherwise, it will raise some errors later.

        中译：跑一个小型 matmul 以初始化 cuBLAS，否则后续可能报错。
        """
        dtype = torch.float16
        device = "cuda"
        a = torch.ones((16, 16), dtype=dtype, device=device)
        b = torch.ones((16, 16), dtype=dtype, device=device)
        c = a @ b
        return c

    def init_attention_backend(self):
        """Init attention kernel backend.

        中译：初始化注意力核后端。根据是否启用 PDMux / 双 batch overlap 走不同分支。
        """
        if self.server_args.enable_pdmux:
            self.attn_backend = self._get_attention_backend(init_new_workspace=True)
            self.decode_attn_backend_group = []
            for _ in range(self.server_args.sm_group_num):
                self.decode_attn_backend_group.append(self._get_attention_backend())
            self.decode_attn_backend = self.decode_attn_backend_group[0]
        elif self.server_args.enable_two_batch_overlap and not self.is_draft_worker:
            self.attn_backend = TboAttnBackend.init_new(self._get_attention_backend)
        else:
            self.attn_backend = self._get_attention_backend()

    def _get_attention_backend(self, init_new_workspace: bool = False):
        """Init attention kernel backend.

        中译：构造并返回一个注意力后端。若 prefill 与 decode 指定了不同后端，则使用
        HybridAttnBackend 分别包装（该特性为实验性）。
        """
        draft_attn_backend = self.server_args.speculative_draft_attention_backend
        if self.is_draft_worker and draft_attn_backend:
            logger.warning(
                f"Overriding draft attention backend to {draft_attn_backend}."
            )
            return self._get_attention_backend_from_str(
                draft_attn_backend,
                init_new_workspace=init_new_workspace,
            )

        (
            self.prefill_attention_backend_str,
            self.decode_attention_backend_str,
        ) = self.server_args.get_attention_backends()

        if self.decode_attention_backend_str != self.prefill_attention_backend_str:
            # 中译：prefill 与 decode 后端不同，使用 HybridAttnBackend 将两者组合。
            from sglang.srt.layers.attention.hybrid_attn_backend import (
                HybridAttnBackend,
            )

            attn_backend = HybridAttnBackend(
                self,
                decode_backend=self._get_attention_backend_from_str(
                    self.decode_attention_backend_str,
                    init_new_workspace=init_new_workspace,
                ),
                prefill_backend=self._get_attention_backend_from_str(
                    self.prefill_attention_backend_str,
                    init_new_workspace=init_new_workspace,
                ),
            )
            logger.info(
                f"Using hybrid attention backend for decode and prefill: "
                f"decode_backend={self.decode_attention_backend_str}, "
                f"prefill_backend={self.prefill_attention_backend_str}."
            )
            logger.warning(
                "Warning: Attention backend specified by --attention-backend or default backend might be overridden."
                "The feature of hybrid attention backend is experimental and unstable. Please raise an issue if you encounter any problem."
            )
        else:
            attn_backend = self._get_attention_backend_from_str(
                self.server_args.attention_backend,
                init_new_workspace=init_new_workspace,
            )

        (
            get_global_server_args().prefill_attention_backend,
            get_global_server_args().decode_attention_backend,
        ) = (self.prefill_attention_backend_str, self.decode_attention_backend_str)
        return attn_backend

    def _get_attention_backend_from_str(
        self, backend_str: str, init_new_workspace: bool = False
    ):
        # 中译：根据后端名称字符串，从注册表 ATTENTION_BACKENDS 中构造对应的注意力后端。
        if backend_str not in ATTENTION_BACKENDS:
            raise ValueError(f"Invalid attention backend: {backend_str}")
        self.init_new_workspace = init_new_workspace
        full_attention_backend = ATTENTION_BACKENDS[backend_str](self)
        return attn_backend_wrapper(self, full_attention_backend)

    def kernel_warmup(self):
        """Warmup and tune kernels before cuda graph capture.

        中译：在 CUDA Graph 捕获之前预热并调优内核（如 flashinfer autotune、PP 并行 DeepGEMM 预热）。
        """
        if self.device != "cuda":
            return

        if self._should_run_flashinfer_autotune():
            self._flashinfer_autotune()

        if (
            envs.SGLANG_PP_PARALLEL_DEEPGEMM_WARMUP.get()
            and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and self.pp_size > 1
            and not self.spec_algorithm.is_speculative()
        ):
            from sglang.srt.layers.deep_gemm_wrapper.compile_utils import (
                pp_parallel_deep_gemm_warmup,
            )

            pp_parallel_deep_gemm_warmup(self)

    def _pre_initialize_flashinfer_allreduce_workspace(self):
        """Pre-initialize flashinfer allreduce fusion workspaces.

        Must run before CUDA graph capture to avoid collective operations
        (broadcasts, barriers) inside the graph capture context, which can
        deadlock with custom_all_reduce.register_graph_buffers.

        中译：预初始化 flashinfer 的 all-reduce 融合工作区。必须在 CUDA Graph 捕获之前运行，
        以避免在 graph 捕获上下文内出现集体通信（广播、barrier），这可能与
        custom_all_reduce.register_graph_buffers 发生死锁。
        """
        if not self.server_args.enable_flashinfer_allreduce_fusion:
            return

        from sglang.srt.layers.communicator import FUSE_ALLREDUCE_MAX_BATCH_SIZE
        from sglang.srt.layers.flashinfer_comm_fusion import pre_initialize_workspaces

        pre_initialize_workspaces(
            max_token_num=FUSE_ALLREDUCE_MAX_BATCH_SIZE,
            hidden_dim=self.model_config.hidden_size,
            dtype=self.dtype,
        )

    def _should_run_flashinfer_autotune(self) -> bool:
        """Check if flashinfer autotune should be run.

        中译：判断是否应运行 flashinfer autotune。会检查是否被禁用、MoE/FP4 后端是否需要
        调优、设备计算能力、以及投机场景等条件。
        """
        if self.server_args.disable_flashinfer_autotune:
            return False

        # CuteDSL v1 (cutedsl runner + deepep a2a) bypasses MoeRunner and must not
        # be autotuned -- its _dummy_run would dispatch more tokens per rank than
        # SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK, tripping a DeepEP assert.
        # Read server_args directly to avoid depending on initialize_moe_config()
        # having already populated the MoE backend globals.
        # 中译：CuteDSL v1（cutedsl runner + deepep a2a）绕过了 MoeRunner，不能被 autotune——
        #       其 _dummy_run 会每 rank 派发超过 SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK 的
        #       token，触发 DeepEP 断言。这里直接读 server_args，避免依赖 initialize_moe_config()
        #       已填充好 MoE 后端全局变量。
        if (
            self.server_args.moe_runner_backend == "flashinfer_cutedsl"
            and self.server_args.moe_a2a_backend == "deepep"
        ):
            return False

        backend_str = self.server_args.moe_runner_backend

        # TODO smor- support other cases for flashinfer autotune, such as, mamba backend
        # 中译：TODO(smor)：支持 flashinfer autotune 的其他场景，例如 mamba 后端。

        moe_needs_autotune = backend_str in [
            "flashinfer_trtllm",
            "flashinfer_trtllm_routed",
            "flashinfer_mxfp4",
            "flashinfer_cutedsl",
            "flashinfer_cutlass",
        ]

        from sglang.srt.layers.quantization.fp4_utils import (
            get_fp4_gemm_runner_backend,
        )

        model_uses_fp4 = self.model_config.quantization in (
            "modelopt_fp4",
            "modelopt_mixed",
        )
        fp4_gemm_needs_autotune = model_uses_fp4 and (
            get_fp4_gemm_runner_backend().is_flashinfer_cutlass()
            or get_fp4_gemm_runner_backend().is_flashinfer_cutedsl()
        )

        if not (moe_needs_autotune or fp4_gemm_needs_autotune):
            return False

        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            return False

        if self.spec_algorithm.is_speculative():
            return not self.is_draft_worker

        return True

    def _flashinfer_autotune(self):
        """Run flashinfer autotune.

        中译：运行 flashinfer autotune。根据缓存开关决定是否复用缓存，并在非默认 stream 上
        跑一次 dummy 前向以完成调优。
        """
        from flashinfer.autotuner import autotune

        from sglang.srt.layers.logits_processor import autotune_dummy_run_mode

        cache_path = self._flashinfer_autotune_cache_path()
        if envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.get():
            autotune_cache = cache_path
            logger.info("Running FlashInfer autotune with cache: %s", autotune_cache)
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            runs_dir = cache_path.parent / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            autotune_cache = (
                runs_dir / f"{cache_path.stem}.{timestamp}{cache_path.suffix}"
            )
            logger.info(
                "Running FlashInfer autotune (cache reuse DISABLED via "
                "SGLANG_FLASHINFER_AUTOTUNE_CACHE=0); writing fresh result to: %s",
                autotune_cache,
            )

        # Run warmup on the non-default stream to avoid NCCL 2.29+ cudaMemcpyBatchAsync
        # calls on default stream (unsupported by CUDA) when --enable-symm-mem is used.
        # 中译：在非默认 stream 上运行预热，以避免使用 --enable-symm-mem 时 NCCL 2.29+ 的
        #       cudaMemcpyBatchAsync 落在默认 stream 上（CUDA 不支持）。
        self.forward_stream.wait_stream(torch.cuda.current_stream())
        with torch.get_device_module(self.device).stream(self.forward_stream):
            with (
                torch.inference_mode(),
                autotune(True, cache=str(autotune_cache)),
                autotune_dummy_run_mode(),
            ):
                self._dummy_run(batch_size=self.req_to_token_pool.size)
        torch.cuda.current_stream().wait_stream(self.forward_stream)
        logger.info("FlashInfer autotune completed.")

    def _flashinfer_autotune_cache_path(self) -> Path:
        # 中译：根据模型/量化/并行等关键信息生成 autotune 缓存文件路径（按 sm 架构、flashinfer
        #       版本、哈希 key、各 rank 区分）。
        import flashinfer

        major, minor = torch.cuda.get_device_capability(self.device)
        arch = f"sm{major}{minor}"
        flashinfer_version = getattr(flashinfer, "__version__", "unknown")

        server_args = self.server_args
        model_key = "|".join(
            [
                str(server_args.model_path),
                str(self.dtype),
                str(server_args.quantization),
                str(server_args.moe_runner_backend),
                str(self.tp_size),
                str(self.pp_size),
                str(self.dp_size),
                str(self.moe_ep_size),
                str(self.model_config.hf_config.__class__.__name__),
            ]
        )
        cache_key = hashlib.sha256(model_key.encode()).hexdigest()[:16]
        cache_dir = (
            Path(envs.SGLANG_CACHE_DIR.get())
            / "flashinfer"
            / "autotune"
            / flashinfer_version
            / arch
            / cache_key
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        return (
            cache_dir
            / f"rank_tp{self.tp_rank}_pp{self.pp_rank}_dp{self.dp_rank or 0}.json"
        )

    def _dummy_run(
        self,
        batch_size: int,
        run_ctx=None,
        forward_mode_override: Optional[ForwardMode] = None,
    ):
        """Run a dummy forward pass for warmup/profiling.

        forward_mode_override forces EXTEND/DECODE regardless of
        is_generation (used by the PP-parallel DeepGEMM warmup).

        中译：运行一次虚拟（dummy）前向，用于预热/profiling。forward_mode_override 可强制使用
        EXTEND/DECODE 模式（不管 is_generation 如何），供 PP 并行 DeepGEMM 预热使用。
        """
        if forward_mode_override is not None:
            capture_forward_mode = forward_mode_override
        elif self.is_generation:
            capture_forward_mode = ForwardMode.DECODE
        else:
            capture_forward_mode = ForwardMode.EXTEND
        capture_hidden_mode = CaptureHiddenMode.NULL
        num_tokens_per_bs = 1
        if self.spec_algorithm.is_speculative():
            if self.is_draft_worker:
                if not self.spec_algorithm.supports_target_verify_for_draft():
                    raise RuntimeError("This should not happen")
            capture_forward_mode = ForwardMode.TARGET_VERIFY
            num_tokens_per_bs = (
                self.spec_algorithm.get_num_tokens_per_bs_for_target_verify(
                    self.server_args.speculative_num_draft_tokens, self.is_draft_worker
                )
            )

        if self.server_args.enable_return_hidden_states:
            capture_hidden_mode = CaptureHiddenMode.FULL

        num_tokens = batch_size * num_tokens_per_bs

        # Keep warmup aligned with scheduler MLP-sync padding.
        # 中译：使预热与调度器的 MLP-sync 填充对齐（否则预热形状会与实际运行不一致）。
        if require_mlp_sync(self.server_args):
            attn_tp_size = get_attention_tp_size()
            if attn_tp_size > 1 and num_tokens % attn_tp_size != 0:
                num_tokens = ceil_align(num_tokens, attn_tp_size)
                batch_size = num_tokens // num_tokens_per_bs

        seq_len_fill_value = self.attn_backend.get_cuda_graph_seq_len_fill_value()

        if self.server_args.enable_torch_compile:
            set_torch_compile_config()
            should_disable_torch_compile = not getattr(
                self.model, "_can_torch_compile", True
            )
            if should_disable_torch_compile:
                log_info_on_rank0(
                    logger,
                    "Transformers backend model reports it is not torch.compile "
                    "compatible (e.g. dynamic rope scaling). Disabling torch.compile.",
                )
                self.server_args.enable_torch_compile = False

        # NOTE: aux hidden state capture (eagle3/dflash) is already
        # configured by init_aux_hidden_state_capture() in initialize().
        # 中译：辅助隐藏状态捕获（eagle3/dflash）已由 initialize() 中的
        #       init_aux_hidden_state_capture() 配置完毕。

        require_mlp_tp_gather_ = require_mlp_tp_gather(self.server_args)
        if require_gathered_buffer(self.server_args):
            assert require_mlp_tp_gather_ or require_attn_tp_gather(self.server_args)

        buffers = _allocate_decode_buffers(
            device=self.device,
            max_bs=batch_size,
            max_num_token=num_tokens,
            hidden_size=self.model_config.hidden_size,
            vocab_size=self.model_config.vocab_size,
            dtype=self.model_config.dtype,
            dp_size=self.server_args.dp_size,
            pp_size=self.server_args.pp_size,
            is_encoder_decoder=self.model_config.is_encoder_decoder,
            require_mlp_tp_gather=require_mlp_tp_gather_,
            seq_len_fill_value=seq_len_fill_value,
            encoder_len_fill_value=(
                getattr(self.model_config.hf_config, "max_source_positions", 0)
                if self.model_config.is_encoder_decoder
                else 0
            ),
            num_tokens_per_bs=num_tokens_per_bs,
            cache_loc_dtype=torch.int64,
            enable_mamba_track=False,
            hc_hidden_size=getattr(self.model_config, "hc_hidden_size", None),
        )
        buffers.num_token_non_padded[...] = num_tokens

        # For extend mode
        # 中译：EXTEND（预填/拓展）模式下需构造 extend 相关的长度/起始位置张量。
        if capture_forward_mode == ForwardMode.EXTEND:
            extend_prefix_lens_cpu = [0] * batch_size
            extend_seq_lens_cpu = [seq_len_fill_value] * batch_size
            extend_num_tokens = num_tokens
            extend_seq_lens = torch.full(
                (batch_size,), seq_len_fill_value, dtype=torch.int32, device=self.device
            )
            extend_prefix_lens = torch.zeros(
                (batch_size,), dtype=torch.int32, device=self.device
            )
            extend_start_loc = torch.arange(
                0, num_tokens, num_tokens_per_bs, dtype=torch.int32, device=self.device
            )
        else:
            extend_prefix_lens_cpu = None
            extend_seq_lens_cpu = None
            extend_num_tokens = None
            extend_seq_lens = None
            extend_prefix_lens = None
            extend_start_loc = None

        if self.server_args.pp_size > 1:
            # PP0 already cp-split hidden_states before send.
            # 中译：PP 第一段（PP0）在发送前已对 hidden_states 做了 cp 切分。
            pp_hidden_tokens = num_tokens
            if (
                capture_forward_mode == ForwardMode.EXTEND
                and self.pp_rank != 0
                and self.attn_cp_size > 1
            ):
                pp_hidden_tokens = num_tokens // self.attn_cp_size
            pp_proxy_tensors = PPProxyTensors(
                {k: v[:pp_hidden_tokens] for k, v in buffers.pp_proxy_tensors.items()}
            )

        if require_mlp_tp_gather_:
            global_num_tokens_cpu = [num_tokens] * self.server_args.dp_size
        elif require_attn_tp_gather(self.server_args):
            global_num_tokens_cpu = [num_tokens]
        else:
            global_num_tokens_cpu = None

        if global_num_tokens_cpu is not None:
            global_dp_buffer_len = sum(global_num_tokens_cpu)
            num_tokens_tensor = torch.tensor(
                global_num_tokens_cpu, dtype=torch.int32, device=self.device
            )
            buffers.global_num_tokens_gpu.copy_(num_tokens_tensor)
            buffers.global_num_tokens_for_logprob_gpu.copy_(num_tokens_tensor)
        else:
            global_dp_buffer_len = None
            global_num_tokens_cpu = None

        spec_info = create_dummy_verify_input(
            self.spec_algorithm,
            self.server_args,
            buffers.custom_mask,
            num_tokens_per_bs,
            self.is_draft_worker,
        )
        if spec_info is not None and (
            self.spec_algorithm.is_eagle() or self.spec_algorithm.is_standalone()
        ):
            # MTP models (e.g. deepseek_nextn) read spec_info.hidden_states
            # during forward; provide a dummy so warmup doesn't crash.
            # 中译：MTP 模型（如 deepseek_nextn）在前向时会读取 spec_info.hidden_states，
            #       这里提供一个虚拟值以免预热崩溃。
            spec_info.hidden_states = torch.zeros(
                (num_tokens, self.model_config.hidden_size),
                dtype=self.dtype,
                device=self.device,
            )
        if capture_hidden_mode != CaptureHiddenMode.FULL:
            capture_hidden_mode = (
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            )

        if self.server_args.enable_lora:
            lora_ids = [None] * batch_size
        else:
            lora_ids = None

        forward_batch = ForwardBatch(
            forward_mode=capture_forward_mode,
            batch_size=batch_size,
            input_ids=buffers.input_ids,
            req_pool_indices=buffers.req_pool_indices,
            seq_lens=buffers.seq_lens,
            seq_lens_cpu=buffers.seq_lens_cpu,
            next_token_logits_buffer=buffers.next_token_logits_buffer,
            orig_seq_lens=buffers.seq_lens,
            out_cache_loc=buffers.out_cache_loc,
            seq_lens_sum=buffers.seq_lens.sum().item(),
            encoder_lens=buffers.encoder_lens,
            return_logprob=False,
            positions=buffers.positions,
            extend_num_tokens=extend_num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=extend_start_loc,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            global_num_tokens_gpu=buffers.global_num_tokens_gpu,
            global_num_tokens_cpu=global_num_tokens_cpu,
            global_num_tokens_for_logprob_gpu=buffers.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            mrope_positions=buffers.mrope_positions,
            spec_algorithm=self.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=capture_hidden_mode,
            num_token_non_padded=buffers.num_token_non_padded,
            global_forward_mode=capture_forward_mode,
            lora_ids=lora_ids,
        )

        if lora_ids is not None:
            self.lora_manager.prepare_lora_batch(forward_batch)

        self.attn_backend.init_forward_metadata(forward_batch)

        def run_once():
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
                global_num_tokens_cpu,
            )
            set_is_extend_in_batch(False)

            kwargs = {}
            if (
                self.server_args.pp_size > 1
                and "pp_proxy_tensors"
                in inspect.signature(self.model.forward).parameters
            ):
                kwargs["pp_proxy_tensors"] = PPProxyTensors(
                    {k: v.clone() for k, v in pp_proxy_tensors.tensors.items()}
                )
            if not self.is_generation:
                kwargs["get_embedding"] = True

            logits_output_or_pp_proxy_tensors = self.model.forward(
                buffers.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )
            return logits_output_or_pp_proxy_tensors

        torch.get_device_module(self.device).synchronize()
        self.tp_group.barrier()
        with forward_context(ForwardContext(attn_backend=self.attn_backend)):
            with torch.inference_mode(), run_ctx or empty_context():
                run_once()

    def maybe_init_ngram_embedding(self):
        # 中译：若模型使用 n-gram embedding，则初始化其 token 表与各模块的缓冲区。
        self.use_ngram_embedding = self.model_config.use_ngram_embedding
        if self.use_ngram_embedding:
            from sglang.srt.layers.n_gram_embedding import NgramEmbedding

            # Sized to mirror req_to_token (indexed by req_pool_idx).
            # 中译：尺寸与 req_to_token 一致（按 req_pool_idx 索引）。
            self.token_table = torch.empty(
                self.req_to_token_pool.req_to_token.shape[0],
                self.model_config.context_len,
                dtype=torch.int32,
                device=self.device,
            )
            chunked_prefill_size = self.server_args.chunked_prefill_size
            assert (
                chunked_prefill_size is not None and chunked_prefill_size > 0
            ), "Ngram embedding requires chunked prefill to be enabled (chunked_prefill_size > 0)"
            for module in self.model.modules():
                if isinstance(module, NgramEmbedding):
                    module.init_buffers(
                        self.max_running_requests, chunked_prefill_size, self.device
                    )

    def maybe_update_ngram_token_table(
        self,
        next_token_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        """Update the ngram embedding token table after sampling.

        中译：在采样之后更新 n-gram embedding 的 token 表（把新生成的 token 写入对应位置）。
        """
        ngram_embedding_info = forward_batch.ngram_embedding_info
        if ngram_embedding_info is None:
            return
        ngram_embedding_info.out_column_starts[: forward_batch.batch_size] = (
            forward_batch.seq_lens
        )
        ngram_embedding_info.out_req_lens[: forward_batch.batch_size] = 1
        update_token_table_decode(
            ne_token_table=ngram_embedding_info.token_table,
            tokens=next_token_ids.to(torch.int32),
            row_indices=forward_batch.req_pool_indices,
            column_starts=ngram_embedding_info.out_column_starts,
        )

    def init_decode_cuda_graph(self):
        """Capture device graphs.

        中译：捕获解码阶段的设备图（CUDA/CPU/NPU graph）。仅对生成类模型生效。
        """
        self.decode_cuda_graph_runner = None
        self.graph_mem_usage = 0

        if not self.is_generation:
            # TODO: Currently, cuda graph only captures decode steps, which only exists for generation models
            # 中译：TODO：目前 CUDA Graph 仅捕获解码步骤，而解码步骤只存在于生成类模型。
            return

        if self.server_args.model_impl.lower() == ModelImpl.MINDSPORE:
            return

        if self.device != "cpu" and check_cuda_graph_backend(
            Phase.DECODE, Backend.DISABLED
        ):
            return

        if self.device == "cpu" and not self.server_args.enable_torch_compile:
            return

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        graph_backend = defaultdict(
            lambda: f"{current_platform.device_name} graph",
            {
                "cuda": "cuda graph",
                "musa": "cuda graph",
                "cpu": "cpu graph",
                "npu": "npu graph",
            },
        )
        logger.info(
            f"Capture {graph_backend[self.device]} begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
        )
        if current_platform.is_out_of_tree():
            GraphRunnerCls = current_platform.get_graph_runner_cls()
            self.decode_cuda_graph_runner = GraphRunnerCls(self)
        else:
            from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
                DecodeCudaGraphRunner,
            )

            graph_runners = defaultdict(
                lambda: DecodeCudaGraphRunner,
                {
                    "cpu": CPUGraphRunner,
                    "npu": NPUGraphRunner,
                },
            )
            self.decode_cuda_graph_runner = graph_runners[self.device](self)

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        self.graph_mem_usage = before_mem - after_mem
        logger.info(
            f"Capture {graph_backend[self.device]} end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
            f"mem usage={self.graph_mem_usage:.2f} GB. avail mem={after_mem:.2f} GB."
        )

    def init_prefill_cuda_graph(self, force_for_draft_worker: bool = False):
        """Initialize piecewise CUDA graph runner.

        中译：初始化预填阶段的“分段（piecewise）CUDA Graph”runner。会从模型中收集注意力层/
        MoE 层/indexer 等，并在满足条件时捕获分段图。多种不支持场景下会提前返回。
        """
        self.prefill_cuda_graph_runner = None

        if check_cuda_graph_backend(Phase.PREFILL, Backend.DISABLED):
            logger.info(
                "Disable prefill CUDA graph because cuda_graph_config "
                "resolved prefill.backend='disabled' (e.g. via "
                "--cuda-graph-backend-prefill=disabled or auto-disable rules)."
            )
            return

        # Draft models skip here during __init__; the eagle worker calls
        # this method explicitly (force_for_draft_worker=True) after
        # init_lm_head so graphs capture the final embedding weights.
        # 中译：草稿模型在 __init__ 阶段跳过这里；eagle worker 会在 init_lm_head 之后显式
        #       调用本方法（force_for_draft_worker=True），以便 graph 捕获到最终的 embedding 权重。
        if self.is_draft_worker and not force_for_draft_worker:
            return

        # Disable piecewise CUDA graph for non-language models
        if not hasattr(self.model, "model"):
            logger.warning(
                "Disable piecewise CUDA graph because the model is not a language model"
            )
            return

        # Disable piecewise CUDA graph for non capture size
        if not self.server_args.cuda_graph_config.prefill.bs:
            logger.warning(
                "Disable piecewise CUDA graph because the capture size is not set"
            )
            return

        # Collect attention layers and moe layers from the model
        # 中译：从模型中收集注意力层与 MoE 层（供分段图捕获使用）。
        self.model.model = resolve_language_model(self.model)
        language_model = getattr(self.model, "language_model", self.model)

        # Resolve model with layers: handle CausalLM wrapper (.model.layers) and direct TextModel (.layers)
        # 中译：解析出拥有 layers 的模型：兼容 CausalLM 包装（.model.layers）与直接的
        #       TextModel（.layers）。
        if hasattr(language_model, "model") and hasattr(language_model.model, "layers"):
            layer_model = language_model.model
        elif hasattr(language_model, "layers"):
            layer_model = language_model
        else:
            logger.warning(
                "Disable piecewise CUDA graph because the model does not have a 'layers' attribute"
            )
            return

        self.attention_layers = []
        self.moe_layers = []
        self.moe_fusions = []
        self.dsa_indexers = []
        for layer in layer_model.layers:
            attn_layer = None
            if hasattr(layer, "self_attn"):
                if hasattr(layer.self_attn, "attn"):
                    attn_layer = layer.self_attn.attn
                elif hasattr(layer.self_attn, "attn_mqa"):
                    # For DeepSeek model
                    attn_layer = layer.self_attn.attn_mqa
                    if _is_hip and hasattr(layer.self_attn, "attn_mha"):
                        attn_layer._pcg_mha_companion = layer.self_attn.attn_mha
            # For hybrid model
            elif hasattr(layer, "attn"):
                attn_layer = layer.attn
            elif hasattr(layer, "linear_attn"):
                if hasattr(layer.linear_attn, "attn"):
                    attn_layer = layer.linear_attn.attn
                else:
                    attn_layer = layer.linear_attn
            # For InternVL model
            elif hasattr(layer, "attention"):
                if hasattr(layer.attention, "attn"):
                    attn_layer = layer.attention.attn
            # For NemotronH and similar hybrid models using 'mixer' attribute
            elif hasattr(layer, "mixer"):
                if hasattr(layer.mixer, "attn"):
                    attn_layer = layer.mixer.attn
                elif hasattr(layer, "_forward_mamba"):
                    # Mamba layer with split op support - store the layer itself
                    attn_layer = layer

            if attn_layer is not None:
                self.attention_layers.append(attn_layer)
            elif hasattr(layer, "mixer"):
                self.attention_layers.append(None)

            moe_block = None
            moe_fusion = None
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                moe_block = layer.mlp.experts
                moe_fusion = layer.mlp
            if hasattr(layer, "block_sparse_moe") and hasattr(
                layer.block_sparse_moe, "experts"
            ):
                moe_block = layer.block_sparse_moe.experts
                moe_fusion = layer.block_sparse_moe
            if hasattr(layer, "moe") and hasattr(layer.moe, "experts"):
                moe_block = layer.moe.experts
                moe_fusion = layer.moe
            # For NemotronH MoE layers using 'mixer' attribute
            if hasattr(layer, "mixer") and hasattr(layer.mixer, "experts"):
                moe_block = layer.mixer.experts
                moe_fusion = layer.mixer
            self.moe_layers.append(moe_block)
            self.moe_fusions.append(moe_fusion)
            # NSA indexers (None for layers without NSA)
            dsa_indexer = None
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "indexer"):
                dsa_indexer = layer.self_attn.indexer
            self.dsa_indexers.append(dsa_indexer)

        if len(self.attention_layers) < self.model_config.num_hidden_layers:
            # TODO(yuwei): support Non-Standard GQA
            # 中译：TODO(yuwei)：支持非标准 GQA。若有部分层不是标准 GQA，则禁用分段 CUDA Graph。
            log_info_on_rank0(
                logger,
                "Disable piecewise CUDA graph because some layers do not apply Standard GQA",
            )
            return

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture piecewise CUDA graph begin. avail mem={before_mem:.2f} GB"
        )

        self.prefill_cuda_graph_runner = PrefillCudaGraphRunner(self)

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        mem_usage = before_mem - after_mem
        logger.info(
            f"Capture piecewise CUDA graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
            f"mem usage={mem_usage:.2f} GB. avail mem={after_mem:.2f} GB."
        )

    def init_threads_binding(self):
        # 中译：初始化 CPU 的 OpenMP 线程绑定（绑核）。根据 SGLANG_CPU_OMP_THREADS_BIND 与
        #       NUMA 拓扑，为每个 tp_rank 分配应绑定的 CPU 核。
        omp_cpuids = os.environ.get("SGLANG_CPU_OMP_THREADS_BIND", "all")
        cpu_ids_by_node = get_cpu_ids_by_node()
        n_numa_node = len(cpu_ids_by_node)
        if omp_cpuids == "all":
            assert self.tp_size <= n_numa_node, (
                f"SGLANG_CPU_OMP_THREADS_BIND is not set, in this case, "
                f"tp_size {self.tp_size} should be smaller than or equal to number of numa node on the machine {n_numa_node}. "
                f"If you need tp_size to be larger than number of numa node, please set the CPU cores for each tp rank via SGLANG_CPU_OMP_THREADS_BIND explicitly. "
                f"For example, on a machine with 2 numa nodes, where core 0-31 are on numa node 0 and core 32-63 are on numa node 1, "
                f"it is suggested to use -tp 2 and bind tp rank 0 to core 0-31 and tp rank 1 to core 32-63. "
                f"This is the default behavior if SGLANG_CPU_OMP_THREADS_BIND is not set and it is the same as setting SGLANG_CPU_OMP_THREADS_BIND=0-31|32-63. "
                f"If you do need tp_size to be larger than the number of numa nodes, you could set SGLANG_CPU_OMP_THREADS_BIND explicitly for example SGLANG_CPU_OMP_THREADS_BIND=0-15|16-31|32-47|48-63 and run with -tp 4. "
                f"If you don't want each tp rank to use all the cores on one numa node, you could set for example SGLANG_CPU_OMP_THREADS_BIND=0-15|32-47 and run with -tp 2."
            )
            if self.tp_size < n_numa_node:
                logger.warning(
                    f"Detected the current machine has {n_numa_node} numa nodes available, but tp_size is set to {self.tp_size}, so only {self.tp_size} numa nodes are used."
                )
            self.local_omp_cpuid = cpu_ids_by_node[self.tp_rank]
        else:
            threads_bind_list = omp_cpuids.split("|")
            assert self.tp_size == len(threads_bind_list), (
                f"SGLANG_CPU_OMP_THREADS_BIND setting must be aligned with TP size parameter ({self.tp_size}). "
                f"Please double check your settings."
            )
            self.local_omp_cpuid = threads_bind_list[self.tp_rank]
            if self.tp_size > n_numa_node:
                logger.warning(
                    f"TP size ({self.tp_size})is larger than numa node number ({n_numa_node}), "
                    f"in this case the available memory amount of each rank cannot be determined in prior. "
                    f"Please set proper `--max-total-tokens` to avoid the out-of-memory error."
                )

    def apply_torch_tp(self):
        # 中译：应用 torch 原生张量并行（通过 device_mesh 将模型切分到多个设备上）。
        logger.info(f"Enabling torch tensor parallelism on {self.tp_size} devices.")
        from sglang.srt.layers.model_parallel import tensor_parallel

        device_mesh = torch.distributed.init_device_mesh(self.device, (self.tp_size,))
        tensor_parallel(self.model, device_mesh)

    def update_decode_attn_backend(self, stream_idx: int):
        # 中译：（PDMux）按 stream 索引切换当前使用的解码注意力后端。
        self.decode_attn_backend = self.decode_attn_backend_group[stream_idx]

    def _ensure_eager_registry(
        self,
        cache: _EagerBufferRegistry,
        raw_bs: int,
        raw_num_tokens: int,
        build: Callable[[int, int], CudaGraphBufferRegistry],
    ) -> CudaGraphBufferRegistry:
        # Built on first use and grown (next power of two) when a batch exceeds
        # the current capacity.
        # 中译：首次使用时构建；当某个 batch 超过当前容量时，按“下一个 2 的幂”扩容。
        if (
            cache.registry is not None
            and raw_bs <= cache.max_bs
            and raw_num_tokens <= cache.max_num_tokens
        ):
            return cache.registry
        cache.max_bs = next_power_of_2(max(raw_bs, cache.max_bs))
        cache.max_num_tokens = next_power_of_2(
            max(raw_num_tokens, cache.max_num_tokens)
        )
        cache.registry = build(cache.max_bs, cache.max_num_tokens)
        return cache.registry

    def _ensure_eager_decode_registry(
        self, raw_bs: int, raw_num_tokens: int
    ) -> CudaGraphBufferRegistry:
        is_encoder_decoder = self.model_config.is_encoder_decoder
        return self._ensure_eager_registry(
            self._eager_decode_registry,
            raw_bs,
            raw_num_tokens,
            lambda bs, num_tokens: build_decode_registry(
                device=self.device,
                max_bs=bs,
                max_num_token=num_tokens,
                # Eager has no padding so this sentinel is never read; 0 avoids the
                # cuda-graph-only fill-value method that some backends lack.
                seq_len_fill_value=0,
                cache_loc_dtype=torch.int64,
                enable_mamba_track=(
                    self.server_args.enable_mamba_extra_buffer()
                    and self.spec_algorithm.is_none()
                ),
                is_encoder_decoder=is_encoder_decoder,
                encoder_len_fill_value=(
                    getattr(self.model_config.hf_config, "max_source_positions", 0)
                    if is_encoder_decoder
                    else 0
                ),
                enable_num_token_non_padded=False,
                register_global_num_tokens=False,
                require_gathered_buffer=False,
                require_mlp_tp_gather=False,
                dp_size=self.server_args.dp_size,
                share_pool=False,
                source=None,
            ),
        )

    def _ensure_eager_prefill_registry(
        self, raw_bs: int, raw_num_tokens: int
    ) -> CudaGraphBufferRegistry:
        return self._ensure_eager_registry(
            self._eager_prefill_registry,
            raw_bs,
            raw_num_tokens,
            lambda bs, num_tokens: build_prefill_registry(
                device=self.device,
                max_bs=bs,
                max_num_token=num_tokens,
                cache_loc_dtype=torch.int64,
                is_multimodal=self.is_multimodal,
                enable_mamba_track=False,
                register_input_embeds=False,
                share_pool=False,
                source=None,
            ),
        )

    def _eager_fb_view(
        self, forward_batch: ForwardBatch, pp_proxy_tensors=None
    ) -> ForwardBatch:
        # 中译：eager（非 CUDA Graph）执行路径下，将 forward_batch 的输入填入可复用的 eager 缓冲区
        #       注册表并返回一个视图（view），以减少反复分配。若开启了“不拷贝”环境变量则直接
        #       返回原 batch 的副本。
        if envs.SGLANG_EAGER_INPUT_NO_COPY.get():
            return replace(forward_batch)
        raw_bs = forward_batch.batch_size
        raw_num_tokens = forward_batch.input_ids.shape[0]
        ensure = (
            self._ensure_eager_prefill_registry
            if forward_batch.forward_mode.is_extend(include_draft_extend_v2=True)
            else self._ensure_eager_decode_registry
        )
        registry = ensure(raw_bs, raw_num_tokens)
        registry.fill_from(
            forward_batch,
            raw_bs=raw_bs,
            padded_bs=raw_bs,
            raw_num_tokens=raw_num_tokens,
            padded_num_tokens=raw_num_tokens,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        return registry.extract_buffer(
            padded_bs=raw_bs,
            padded_num_tokens=raw_num_tokens,
            forward_batch_template=forward_batch,
        )

    def forward_decode(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors=None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        # 中译：执行一次解码（decode）前向（eager 路径，未命中 CUDA Graph 时）。
        if not self.server_args.enable_pdmux:
            forward_batch = self._eager_fb_view(forward_batch, pp_proxy_tensors)
        # Set extra arguments
        # 中译：设置额外参数，并初始化注意力元数据。
        pdmux_override = False
        if forward_batch.needs_forward_metadata_init():
            if hasattr(self.model, "prepare_forward_batch"):
                # Prepare model-specific attention metadata before planning,
                # e.g. Moss-VL's prefill cross-attention custom mask.
                self.model.prepare_forward_batch(forward_batch)
            if self.server_args.enable_pdmux:
                self.decode_attn_backend.init_forward_metadata(forward_batch)
                # PDmux selects a per-stream backend; publish it to model-layer
                # readers via the active ForwardContext so RadixAttention etc.
                # dispatch against the right backend for this forward.
                # 中译：PDMux 选择一个按 stream 区分的后端；通过当前 ForwardContext 发布给模型层的
                #       读取者，使 RadixAttention 等在本次前向中调度到正确的后端。
                pdmux_override = True
            else:
                self.attn_backend.init_forward_metadata(forward_batch)
        # FIXME: add pp_proxy_tensors arg to all models
        # 中译：FIXME：为所有模型都加上 pp_proxy_tensors 参数。
        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors

        # Launch forward
        # 中译：启动前向（可选包裹设备计时器）。
        ctx = (
            self.device_timer.wrap(metadata={"category": "decode"})
            if self.device_timer
            else contextlib.nullcontext()
        )

        def _do_forward():
            return self.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )

        with ctx:
            if pdmux_override:
                with forward_context(
                    ForwardContext(attn_backend=self.decode_attn_backend)
                ):
                    return _do_forward()
            return _do_forward()

    def forward_extend(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors=None,
    ) -> Tuple[
        Union[LogitsProcessorOutput, PPProxyTensors, EmbeddingPoolerOutput], bool
    ]:
        # 中译：执行一次 EXTEND（预填/拓展）前向。返回 (输出, 是否命中分段 CUDA Graph)。
        # Setup extra arguments
        # 中译：准备额外参数（PP 代理张量、input_embeds、embedding 覆盖等）。
        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors
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

        # Check piecewies cuda graph
        # 中译：检查是否可以走分段（piecewise）CUDA Graph（需 runner 存在且本 batch 可由其执行）。
        can_run_graph = (
            self.prefill_cuda_graph_runner is not None
            and self.prefill_cuda_graph_runner.can_run(forward_batch)
        )
        if can_run_graph:
            # TODO: device_timer.wrap is too broad here — it also includes
            # replay_prepare time. Move timing into the prefill cuda graph
            # runner to capture only the model.forward part.
            # 中译：TODO：这里 device_timer.wrap 范围过宽——它还包含了 replay_prepare 的时间。
            #       应把计时移到分段 CUDA Graph runner 里，只统计 model.forward 部分。
            ctx = (
                self.device_timer.wrap(metadata={"category": "extend"})
                if self.device_timer
                else contextlib.nullcontext()
            )
            with ctx:
                ret = self.prefill_cuda_graph_runner.replay(forward_batch, **kwargs)
            return (ret, can_run_graph)

        if not self.server_args.enable_pdmux:
            forward_batch = self._eager_fb_view(forward_batch, pp_proxy_tensors)

        # Launch model forward
        if forward_batch.needs_forward_metadata_init():
            if hasattr(self.model, "prepare_forward_batch"):
                # Prepare model-specific attention metadata before planning,
                # e.g. Moss-VL's prefill cross-attention custom mask.
                self.model.prepare_forward_batch(forward_batch)
            self.attn_backend.init_forward_metadata(forward_batch)

        ctx = (
            self.device_timer.wrap(metadata={"category": "extend"})
            if self.device_timer
            else contextlib.nullcontext()
        )
        with ctx:
            if _is_hip and self.prefill_cuda_graph_runner is not None:
                # AMD/HIP: when PCG is enabled but the batch exceeds max captured
                # size, run eagerly under enable_tc_piecewise_cuda_graph() and
                # set_tc_piecewise_forward_context() so that (a) Dynamo guards on
                # _in_tc_piecewise_cuda_graph stay consistent with the PCG-traced
                # graph (preventing runtime recompilation) and (b) PCG-specific
                # code paths (MoE, attention) can access their layer objects.
                # 中译：AMD/HIP：当开启 PCG（分段 CUDA Graph）但 batch 超过最大捕获尺寸时，在
                #       enable_tc_piecewise_cuda_graph() 与 set_tc_piecewise_forward_context() 下以
                #       eager 方式运行，以保证：(a) Dynamo 对 _in_tc_piecewise_cuda_graph 的 guard 与
                #       PCG 跟踪的 graph 保持一致（避免运行时重编译）；(b) PCG 专用代码路径（MoE、
                #       注意力）能访问到其层对象。
                with (
                    enable_tc_piecewise_cuda_graph(),
                    set_tc_piecewise_forward_context(
                        forward_batch,
                        self.attention_layers,
                        getattr(self.model, "quant_config", None),
                        self.moe_layers,
                        self.moe_fusions,
                        dsa_indexers=self.dsa_indexers,
                    ),
                ):
                    ret = self.model.forward(
                        forward_batch.input_ids,
                        forward_batch.positions,
                        forward_batch,
                        **kwargs,
                    )
            else:
                ret = self.model.forward(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                    **kwargs,
                )
        return (ret, can_run_graph)

    def forward_idle(
        self, forward_batch: ForwardBatch, pp_proxy_tensors=None
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        # In DP Attention, IDLE batches may be padded (batch_size > 0) for MLP
        # sync. Reinit metadata for the padded case so attention kernels see
        # the right batch_size (e.g. DSA Indexer). For the unpadded case
        # (batch_size == 0) explicitly drop any stale forward_metadata left
        # over from the previous forward — without this, attention layers
        # called from the idle path can re-read a prior batch's req_pool
        # indices and trigger SWA mapping use-after-free.
        if forward_batch.batch_size > 0:
            if not self.server_args.enable_pdmux:
                forward_batch = self._eager_fb_view(forward_batch, pp_proxy_tensors)
            self.attn_backend.init_forward_metadata(forward_batch)
        else:
            self.attn_backend.forward_metadata = None

        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors
        ctx = (
            self.device_timer.wrap(metadata={"category": "idle"})
            if self.device_timer
            else contextlib.nullcontext()
        )
        with ctx:
            return self.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )

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
        ctx = (
            self.device_timer.wrap(metadata={"category": "split_prefill"})
            if self.device_timer
            else contextlib.nullcontext()
        )
        with ctx:
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
                self.gpu_id if self.dp_size is not None and self.dp_size > 1 else None
            )
            self.msprobe_debugger.start(model=self.model, rank_id=rank_id)

        # Step span
        # 中译：为本步创建 profiling 的跨度（span）。
        step_span_ctx = profile_range(_build_step_span_name(forward_batch))

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

        no_copy_to_cpu = not self.server_args.disable_overlap_schedule
        if (experts_capturer := get_global_experts_capturer()) is not None:
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

        if self.server_args.elastic_ep_backend is not None:
            self.maybe_recover_ep_ranks()

        return output

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
                and self.decode_cuda_graph_runner.can_run(forward_batch)
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
                ret = self.decode_cuda_graph_runner.replay(
                    forward_batch,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
                return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)

            # For MLP sync
            # 中译：用于 MLP 同步（DP attention 场景下各 DP rank 的 token 数需对齐）。
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
                and require_gathered_buffer(self.server_args)
                and not is_dsa_enable_prefill_cp()
                and not is_mla_prefill_cp_enabled()
            ):
                forward_batch.adjust_num_token_non_padded_for_attn_tp(
                    server_args=self.server_args,
                )

            # Hisparse coordinator — backends now read it from self.model_runner.
            if self.hisparse_coordinator is not None:
                self.hisparse_coordinator.num_real_reqs.fill_(forward_batch.batch_size)

            # Forward without cuda graph
            # 中译：未命中 CUDA Graph 时，按 forward_mode 分发到各 eager 前向方法。
            if forward_batch.forward_mode.is_decode():
                ret = self.forward_decode(
                    forward_batch,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
            elif forward_batch.forward_mode.is_split_prefill():
                ret = self.forward_split_prefill(
                    forward_batch,
                    reinit_attn_backend=reinit_attn_backend,
                    forward_count=split_forward_count,
                )
            elif forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
                ret, can_run_graph = self.forward_extend(
                    forward_batch,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
            elif forward_batch.forward_mode.is_idle():
                ret = self.forward_idle(
                    forward_batch, pp_proxy_tensors=pp_proxy_tensors
                )
            else:
                raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode}")

            if (
                forward_batch.global_num_tokens_cpu is not None
                and self.pp_group.is_last_rank
            ):
                forward_batch.post_forward_mlp_sync_batch(ret)

            return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)

    def _preprocess_logits(
        self, logits_output: LogitsProcessorOutput, sampling_info: SamplingBatchInfo
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
        sampling_info.apply_logits_bias(logits_output.next_token_logits)

        # Release the vocab_mask GPU tensor immediately after it has been applied
        # to the logits. In overlap scheduling, the sampling_info (and its
        # vocab_mask) can be kept alive by the delay_sample_func closure and
        # batch_record_buf until the next iteration, causing a steady VRAM leak
        # when structured output (grammar) is used.
        # 中译：在 vocab_mask 应用到 logits 后立即释放其 GPU 张量。overlap 调度下，sampling_info
        #       （及其 vocab_mask）可能被 delay_sample_func 闭包与 batch_record_buf 保活到下一迭代，
        #       在使用结构化输出（grammar）时造成持续的显存泄漏。
        sampling_info.vocab_mask = None

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
        #       并在 mask 用完后及时释放其显存（详见 _preprocess_logits）。
        self._preprocess_logits(logits_output, forward_batch.sampling_info)

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
        self.maybe_update_ngram_token_table(next_token_ids, forward_batch)
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
        if not forward_batch.token_ids_logprobs:
            return

        # Preprocess logits (same as in sample method)
        self._preprocess_logits(logits_output, forward_batch.sampling_info)

        # Delegate to sampler for logprob-only computation
        # This populates logits_output with requested token probabilities
        self.sampler.compute_logprobs_only(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
        )

    def save_remote_model(self, url: str):
        # 中译：将模型保存到远程存储（由 url 指定）。
        from sglang.srt.model_loader.loader import RemoteModelLoader

        logger.info(f"Saving model to {url}")
        RemoteModelLoader.save_model(self.model, self.model_config.model_path, url)

    def save_sharded_model(
        self, path: str, pattern: Optional[str] = None, max_size: Optional[int] = None
    ):
        # 中译：将模型以分片（sharded）形式保存到本地路径。
        from sglang.srt.model_loader.loader import ShardedStateLoader

        logger.info(
            f"Save sharded model to {path} with pattern {pattern} and max_size {max_size}"
        )
        ShardedStateLoader.save_model(self.model, path, pattern, max_size)

    def check_weights(self, action: str):
        # 中译：检查/校验权重（由 WeightChecker 处理，用于调试/验证场景）。
        return self._weight_checker.handle(action=action)

    def update_weights_from_ipc(self, recv_req):
        """Update weights from IPC for checkpoint-engine integration.

        中译：通过 IPC 更新权重（用于与 checkpoint-engine 集成）。
        """
        try:
            from sglang.srt.checkpoint_engine.checkpoint_engine_worker import (
                SGLangCheckpointEngineWorkerExtensionImpl,
            )

            # Create a worker extension that integrates with SGLang's model
            worker = SGLangCheckpointEngineWorkerExtensionImpl(self)
            worker.update_weights_from_ipc(recv_req.zmq_handles)
            return True, "IPC weight update completed successfully"
        except ImportError as e:
            return False, f"IPC weight update failed: ImportError {e}"
        except Exception as e:
            logger.error(f"IPC weight update failed: {e}")
            return False, str(e)

    def prealloc_symmetric_memory_pool(self):
        # 中译：预分配对称内存池。
        # PyTorch mempools never de-fragment memory in OOM scenarios, so we need to pre-allocate a large chunk of memory to limit fragmentation.
        # 中译：PyTorch 的内存池在 OOM 场景下不会去碎片化，所以需要预先分配一大块内存以限制碎片。
        if (
            self.is_draft_worker
            or not self.server_args.enable_symm_mem
            or envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get() <= 0
        ):
            return

        # Memory allocation is tied to a cuda stream, use the forward stream
        with torch.get_device_module(self.device).stream(self.forward_stream):
            logger.info(
                f"Pre-allocating symmetric memory pool with {envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get()} GiB"
            )
            with use_symmetric_memory(get_tp_group()):
                torch.empty(
                    (envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get() * 1024 * 1024 * 1024,),
                    dtype=torch.uint8,
                    device=self.device,
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
        #       并重新执行一次前向以获取正确输出。
        elastic_ep_state = ElasticEPStateManager.instance()
        if elastic_ep_state is not None and not elastic_ep_state.is_active_equal_last():
            elastic_ep_state.snapshot_active_to_last()
            elastic_ep_state.sync_active_to_cpu()
            logging.info("EPLB due to rank faults")
            gen = self.eplb_manager.rebalance()
            while True:
                try:
                    next(gen)
                except StopIteration:
                    break
            output = self._forward_raw(
                forward_batch,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
        return output


def _model_load_weights_direct(model, named_tensors: List[Tuple[str, torch.Tensor]]):
    # 中译：以 direct 方式直接将命名张量加载到模型参数（使用默认权重加载器）。
    params_dict = dict(model.named_parameters())
    for name, tensor in named_tensors:
        default_weight_loader(params_dict[name], tensor)


def _unwrap_tensor(tensor, tp_rank, device):
    # 中译：解包张量：若是跨进程序列化的 LocalSerializedTensor，则取出对应 tp_rank 的部分，
    #       再搬到目标设备。
    if isinstance(tensor, LocalSerializedTensor):
        tensor = tensor.get(tp_rank)
    return tensor.to(device)


def _build_step_span_name(forward_batch: ForwardBatch) -> str:
    """Build a profile-trace span name for one forward step.

    中译：为一次前向步骤构建 profiling 跟踪的跨度（span）名称。
    """
    mode = forward_batch.forward_mode
    bs = forward_batch.batch_size
    if mode == ForwardMode.EXTEND:
        ext_toks = forward_batch.extend_num_tokens or 0
        return f"step[EXTEND bs={bs} toks={ext_toks}]"
    return f"step[{mode.name} bs={bs}]"


@dataclass
class LocalSerializedTensor:
    """torch.Tensor that gets serialized by MultiprocessingSerializer (which only serializes a pointer and not the data).
    The i-th element in the list corresponds to i-th rank's GPU.

    中译：一个由 MultiprocessingSerializer 序列化的 torch.Tensor（它只序列化指针而非数据本身）。
    列表中第 i 个元素对应第 i 个 rank 的 GPU。
    """

    values: List[bytes]

    def get(self, rank: int):
        return MultiprocessingSerializer.deserialize(self.values[rank])
