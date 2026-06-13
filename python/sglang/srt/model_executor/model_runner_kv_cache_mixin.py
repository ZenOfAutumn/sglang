from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.configs.model_config import get_nsa_index_head_dim, is_deepseek_nsa
from sglang.srt.distributed.parallel_state import get_world_group
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.hisparse_memory_pool import (
    HiSparseNSATokenToKVPool,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.memory_pool import (
    DoubleSparseTokenToKVPool,
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
    MHATokenToKVPoolFP4,
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
    NSATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool, SWATokenToKVPoolAllocator
from sglang.srt.utils.common import (
    get_available_gpu_memory,
    is_float4_e2m1fn_x2,
    is_npu,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


@dataclass
class MemoryPoolConfig:
    """解析后的显存池配置，在目标 worker 与草稿 worker 之间共享。"""

    max_total_num_tokens: int  # KV 缓存可容纳的最大 token 总数
    max_running_requests: int  # 最大同时运行请求数
    full_max_total_num_tokens: Optional[int] = None  # 全局注意力层的最大 token 数（混合 SWA 时）
    swa_max_total_num_tokens: Optional[int] = None  # 滑动窗口层的最大 token 数（混合 SWA 时）

    mem_fraction_static: Optional[float] = None  # 静态显存占比（仅用于报错提示）

    def __post_init__(self):
        # token 容量为非正值说明显存不足，提示调大 mem-fraction-static
        if self.max_total_num_tokens <= 0:
            msg = "Not enough memory. Please try to increase --mem-fraction-static."
            if self.mem_fraction_static is not None:
                msg += f" Current value: mem_fraction_static={self.mem_fraction_static}"
            raise RuntimeError(msg)


# mamba 缓存池大小与 max_running_requests 的比值
MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO = 3
MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP = 2  # v2 额外缓冲区在 overlap 调度下的额外比值
MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_OVERLAP = 1  # v2 额外缓冲区在非 overlap 调度下的额外比值

logger = logging.getLogger(__name__)

_is_npu = is_npu()


class ModelRunnerKVCacheMixin:
    """为 ModelRunner 提供 KV 缓存显存池相关能力的 Mixin。

    负责：计算每 token 显存占用、探测可用显存对应的最大 token 数、
    拆分混合 SWA 的全局/滑动窗口 token 配额、创建各类显存池与分配器等。
    """

    def get_cell_size_per_token(self: ModelRunner, num_layers: int) -> int:
        """计算每个 token 在 KV 缓存中占用的字节数（综合所有层、按模型类型区分）。"""
        kv_size = torch._utils._element_size(self.kv_cache_dtype)
        if self.use_mla_backend:
            # MLA 后端：每 token 只需存压缩后的 latent + rope 部分
            cell_size = (
                (self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim)
                * num_layers
                * kv_size
            )
            if is_float4_e2m1fn_x2(self.kv_cache_dtype):
                # FP4 量化需额外的 kv_scale_buffer（缩放因子缓冲区）
                scale_block_size = 16
                cell_size = (cell_size // 2) + (
                    (
                        (
                            self.model_config.kv_lora_rank
                            + self.model_config.qk_rope_head_dim
                        )
                        // scale_block_size
                    )
                    * num_layers
                    * kv_size
                )

            # NSA 模型（DeepSeek V3.2）需加上 indexer 的 KV 缓存开销
            if is_deepseek_nsa(self.model_config.hf_config):
                index_head_dim = get_nsa_index_head_dim(self.model_config.hf_config)
                indexer_size_per_token = (
                    index_head_dim
                    + index_head_dim // NSATokenToKVPool.quant_block_size * 4
                )
                element_size = torch._utils._element_size(
                    NSATokenToKVPool.index_k_with_scale_buffer_dtype
                )
                cell_size += indexer_size_per_token * num_layers * element_size
        else:
            # 非 MLA（如 MHA）后端
            if self.model_config.is_hybrid_swa:
                # 混合 SWA：全局层与滑动窗口层分别计算每 token 占用后求和
                full_layers_num = len(self.model_config.full_attention_layer_ids)
                swa_layers_num = len(self.model_config.swa_attention_layer_ids)

                full_per_token = self.model_config.get_num_kv_heads(
                    get_attention_tp_size()
                ) * (self.model_config.head_dim + self.model_config.v_head_dim)

                swa_per_token = self.model_config.get_swa_num_kv_heads(
                    get_attention_tp_size()
                ) * (self.model_config.swa_head_dim + self.model_config.swa_v_head_dim)

                cell_size = (
                    full_per_token * full_layers_num + swa_per_token * swa_layers_num
                ) * kv_size
            else:
                # 普通（非混合）注意力：KV 头数 * (k维 + v维) * 层数 * 元素字节数
                cell_size = (
                    self.model_config.get_num_kv_heads(get_attention_tp_size())
                    * (self.model_config.head_dim + self.model_config.v_head_dim)
                    * num_layers
                    * kv_size
                )

            if is_float4_e2m1fn_x2(self.kv_cache_dtype):
                # FP4 量化的 kv_scale_buffer（缩放因子缓冲区）
                scale_block_size = 16

                n = self.model_config.get_num_kv_heads(get_attention_tp_size())
                k = self.model_config.head_dim
                cell_size = (cell_size // 2) + (
                    (n * k * num_layers * 2 * kv_size) // scale_block_size
                )
        return cell_size

    def profile_max_num_token(self: ModelRunner, pre_model_load_memory: int):
        """探测加载模型后的可用显存，换算出 KV 缓存可容纳的最大 token 数。"""
        post_model_load_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=get_world_group().world_size > 1,
            cpu_group=get_world_group().cpu_group,
        )

        # 获取用于 KV 缓存计算的层数
        if self.is_draft_worker:
            num_layers = getattr(
                self.model_config.hf_config,
                "num_nextn_predict_layers",
                self.num_effective_layers,
            )
        elif mambaish := self.mambaish_config:
            effective_layer_ids = [
                i
                for i in mambaish.full_attention_layer_ids
                if self.start_layer <= i < self.end_layer
            ]
            num_layers = len(effective_layer_ids)
        else:
            num_layers = self.num_effective_layers

        cell_size = self.get_cell_size_per_token(num_layers)

        # 可用于 KV 缓存的剩余显存 = 加载后可用显存 - 预留的非静态部分
        rest_memory = post_model_load_memory - pre_model_load_memory * (
            1 - self.mem_fraction_static
        )
        if self.mambaish_config is not None:
            # mamba 模型需先扣除 mamba 状态所需显存
            rest_memory = self.handle_max_mamba_cache(rest_memory)

        # 剩余显存（GB 转字节）除以每 token 占用 = 最大 token 数
        return int(rest_memory * (1 << 30)) // cell_size

    def handle_max_mamba_cache(self: ModelRunner, total_rest_memory):
        """为 mamba 状态缓存划分显存，返回扣除 mamba 状态后的剩余显存。

        会根据是否明确指定 max_mamba_cache_size、是否禁用 radix 缓存、
        或按 mamba 与全 KV 显存比值自动求解，来决定 mamba 缓存大小。
        """
        config = self.mambaish_config
        server_args = self.server_args
        assert config is not None

        # 为投机解码使用的 mamba 中间状态预留显存
        if not self.spec_algorithm.is_none():
            assert server_args.speculative_num_draft_tokens is not None
            assert server_args.max_running_requests is not None

            max_running_requests = server_args.max_running_requests // (
                self.dp_size if server_args.enable_dp_attention else 1
            )
            mamba_state_intermediate_size = (
                config.mamba2_cache_params.mamba_cache_per_req
                * max_running_requests
                * server_args.speculative_num_draft_tokens
            )
            total_rest_memory = total_rest_memory - (
                mamba_state_intermediate_size / (1 << 30)
            )

        if server_args.max_mamba_cache_size is not None:
            # 使用显式设置的 max_mamba_cache_size（按 dp 平均）
            server_args.max_mamba_cache_size = server_args.max_mamba_cache_size // (
                server_args.dp_size if server_args.enable_dp_attention else 1
            )
        elif (
            server_args.disable_radix_cache
            and server_args.max_running_requests is not None
        ):
            # 禁用 radix 缓存时，使用显式设置的 max_running_requests
            server_args.max_mamba_cache_size = server_args.max_running_requests // (
                server_args.dp_size if server_args.enable_dp_attention else 1
            )
        else:
            # 使用基于比值的计算，自动适配可用显存
            assert config.mamba2_cache_params.mamba_cache_per_req > 0

            # 根据 mamba 状态显存与全 KV 缓存显存的比值分配，求解方程组：
            # 1. mamba_state_memory + full_kv_cache_memory == total_rest_memory
            # 2. mamba_state_memory / full_kv_cache_memory == server_args.mamba_full_memory_ratio
            mamba_state_memory_raw = (
                total_rest_memory
                * server_args.mamba_full_memory_ratio
                / (1 + server_args.mamba_full_memory_ratio)
            )
            # 根据总 mamba 显存反推出 max_mamba_cache_size
            server_args.max_mamba_cache_size = int(
                (mamba_state_memory_raw * (1 << 30))
                // config.mamba2_cache_params.mamba_cache_per_req
            )

        mamba_state_memory = (
            server_args.max_mamba_cache_size
            * config.mamba2_cache_params.mamba_cache_per_req
            / (1 << 30)
        )
        return total_rest_memory - mamba_state_memory

    def calculate_mla_kv_cache_dim(self: ModelRunner) -> int:
        """计算 MLA 后端的 KV 缓存维度（NSA + FP8 存储时需考虑缩放与 rope 额外存储）。"""
        is_nsa_model = is_deepseek_nsa(self.model_config.hf_config)
        kv_cache_dtype = self.kv_cache_dtype
        kv_lora_rank = self.model_config.kv_lora_rank
        qk_rope_head_dim = self.model_config.qk_rope_head_dim
        kv_cache_dim = kv_lora_rank + qk_rope_head_dim  # 默认的 MLA KV 缓存维度

        # 非 NSA 模型的 MLA KV 缓存维度就是 kv_lora_rank + qk_rope_head_dim
        if not is_nsa_model:
            return kv_cache_dim

        # TRTLLM 后端不会覆盖 MLA KV 缓存的 kv_cache_dim。
        # 假设使用 trtllm MLA 后端时 nsa prefill 与 decode 后端一致，
        # 因为 trtllm 与其他 mla 注意力后端的 KV 缓存布局不同、不兼容。
        if (
            self.server_args.nsa_prefill_backend == "trtllm"
            or self.server_args.nsa_decode_backend == "trtllm"
        ):
            return kv_cache_dim

        quant_block_size = NSATokenToKVPool.quant_block_size
        rope_storage_dtype = NSATokenToKVPool.rope_storage_dtype
        # 为非 trtllm 注意力后端计算 FP8 存储下的 override_kv_cache_dim：
        # kv_lora_rank + 缩放存储（kv_lora_rank // quant_block_size * 4 字节）+ rope 维度存储
        # 注：rope 维度以原始类型（bf16）存储，不量化为 fp8
        if kv_cache_dtype == torch.float8_e4m3fn:
            assert (
                kv_lora_rank % quant_block_size == 0
            ), f"kv_lora_rank {kv_lora_rank} must be multiple of quant_block_size {quant_block_size}"

            return (
                kv_lora_rank
                + kv_lora_rank // quant_block_size * 4
                + qk_rope_head_dim * rope_storage_dtype.itemsize
            )

        return kv_cache_dim

    def _resolve_hybrid_swa_tokens(
        self: ModelRunner, token_capacity: int
    ) -> Tuple[int, int, int]:
        """Split token_capacity into full/swa pools.

        Returns (effective_capacity, full_max_total_num_tokens, swa_max_total_num_tokens).
        """
        page_size = self.server_args.page_size

        assert self.sliding_window_size is not None and self.sliding_window_size > 0
        full_layers_num = len(self.model_config.full_attention_layer_ids)
        swa_layers_num = len(self.model_config.swa_attention_layer_ids)

        assert swa_layers_num > 0, "Hybrid SWA model must have at least one SWA layer"

        def align_page_size(x: int) -> int:
            """向下取整到 page_size 的整数倍。"""
            return (x // page_size) * page_size

        if full_layers_num == 0:
            # 所有层都是 SWA 层
            swa_tokens = align_page_size(token_capacity)
            logger.info(
                f"Use sliding window memory pool (all SWA). swa_layer_tokens={swa_tokens}"
            )
            return swa_tokens, 0, swa_tokens

        swa_full_tokens_ratio = self.server_args.swa_full_tokens_ratio

        # Use unified memory-based allocation for all hybrid SWA models.
        #
        # Let:
        #   F = Full layer per-token memory
        #   S = SWA layer per-token memory (may differ from F)
        #   r = swa_full_tokens_ratio = swa_tokens / full_tokens
        #
        # The profile phase computed:
        #   cell_size = F * n_full + S * n_swa
        #   token_capacity = rest_memory / cell_size
        #   => total_memory = token_capacity * (F * n_full + S * n_swa)
        #
        # We need to solve:
        #   full_tokens * F * n_full + swa_tokens * S * n_swa = total_memory
        #   swa_tokens = full_tokens * r
        #
        # Solution:
        #   full_tokens = total_memory / (F * n_full + r * S * n_swa)
        #               = token_capacity * (F * n_full + S * n_swa) / (F * n_full + r * S * n_swa)

        kv_size = torch._utils._element_size(self.kv_cache_dtype)

        # Full layer per-token memory
        full_per_token = (
            self.model_config.get_num_kv_heads(get_attention_tp_size())
            * (self.model_config.head_dim + self.model_config.v_head_dim)
            * kv_size
        )

        # SWA layer per-token memory
        swa_per_token = (
            self.model_config.get_swa_num_kv_heads(get_attention_tp_size())
            * (self.model_config.swa_head_dim + self.model_config.swa_v_head_dim)
            * kv_size
        )

        # Total memory available from profile
        total_memory = token_capacity * (
            full_per_token * full_layers_num + swa_per_token * swa_layers_num
        )

        # Solve the equations
        denominator = (
            full_per_token * full_layers_num
            + swa_full_tokens_ratio * swa_per_token * swa_layers_num
        )
        assert (
            denominator > 0
        ), f"Invalid denominator={denominator} for memory-based allocation. full_per_token={full_per_token}, full_layers_num={full_layers_num}, swa_per_token={swa_per_token}, swa_layers_num={swa_layers_num}, swa_full_tokens_ratio={swa_full_tokens_ratio}"

        full_tokens = align_page_size(int(total_memory / denominator))
        swa_tokens = align_page_size(int(full_tokens * swa_full_tokens_ratio))

        logger.info(
            f"Use sliding window memory pool. full_layer_tokens={full_tokens}, swa_layer_tokens={swa_tokens}"
        )
        return full_tokens, full_tokens, swa_tokens

    def _calculate_mamba_ratio(self: ModelRunner) -> int:
        """计算 mamba 缓存池大小与 max_running_requests 的比值（含 ping-pong 缓冲额外比值）。"""
        if self.server_args.disable_radix_cache:
            return 1

        additional_ratio = 0
        if self.server_args.enable_mamba_extra_buffer():
            # ping-pong 缓冲区：overlap 调度开启时为 2，否则为 1
            if not self.server_args.disable_overlap_schedule:
                additional_ratio = MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP
            else:
                additional_ratio = MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_OVERLAP

        return MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO + additional_ratio

    def _init_pools(self: ModelRunner):
        """初始化显存池：请求->token 映射池、token->KV 缓存池、以及对应的分配器。"""
        max_num_reqs = self.max_running_requests

        # 初始化 req_to_token_pool（请求->token 映射池）
        if self.req_to_token_pool is None:
            # FIXME(lsyin): 这是使用投机解码时上下文长度问题的临时修复
            extra_max_context_len = 4
            if self.server_args.speculative_num_draft_tokens is not None:
                extra_max_context_len += self.server_args.speculative_num_draft_tokens

            if self.server_args.disaggregation_mode == "decode":
                # PD 分离的 decode 端：使用专用的请求池（支持预分配）
                from sglang.srt.disaggregation.decode import (
                    DecodeReqToTokenPool,
                    HybridMambaDecodeReqToTokenPool,
                )

                # 为预分配请求预订显存：若 max_num_reqs <= 32，则预分配 2 倍请求
                pre_alloc_size = envs.SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS.get()
                pre_alloc_size = (
                    max_num_reqs * 2 if max_num_reqs <= 32 else pre_alloc_size
                )
                if config := self.mambaish_config:
                    self.req_to_token_pool = HybridMambaDecodeReqToTokenPool(
                        size=max_num_reqs,
                        max_context_len=self.model_config.context_len
                        + extra_max_context_len,
                        device=self.device,
                        enable_memory_saver=self.server_args.enable_memory_saver,
                        cache_params=config.mamba2_cache_params,
                        mamba_layer_ids=(
                            [
                                i
                                for i in config.mamba2_cache_params.layers
                                if self.start_layer <= i < self.end_layer
                            ]
                        ),
                        speculative_num_draft_tokens=self.server_args.speculative_num_draft_tokens,
                        enable_mamba_extra_buffer=self.server_args.enable_mamba_extra_buffer(),
                        pre_alloc_size=pre_alloc_size,
                        enable_overlap_schedule=not self.server_args.disable_overlap_schedule,
                        mamba_size=self.server_args.max_mamba_cache_size,
                        start_layer=self.start_layer,
                    )
                else:
                    self.req_to_token_pool = DecodeReqToTokenPool(
                        size=max_num_reqs,
                        max_context_len=self.model_config.context_len
                        + extra_max_context_len,
                        device=self.device,
                        enable_memory_saver=self.server_args.enable_memory_saver,
                        pre_alloc_size=pre_alloc_size,
                    )
            elif config := self.mambaish_config:
                self.req_to_token_pool = HybridReqToTokenPool(
                    size=max_num_reqs,
                    mamba_size=self.server_args.max_mamba_cache_size,
                    mamba_spec_state_size=max_num_reqs,
                    max_context_len=self.model_config.context_len
                    + extra_max_context_len,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    cache_params=config.mamba2_cache_params,
                    mamba_layer_ids=(
                        [
                            i
                            for i in config.mamba2_cache_params.layers
                            if self.start_layer <= i < self.end_layer
                        ]
                    ),
                    enable_mamba_extra_buffer=self.server_args.enable_mamba_extra_buffer(),
                    speculative_num_draft_tokens=self.server_args.speculative_num_draft_tokens,
                    enable_overlap_schedule=not self.server_args.disable_overlap_schedule,
                    start_layer=self.start_layer,
                )
            else:
                self.req_to_token_pool = ReqToTokenPool(
                    size=max_num_reqs,
                    max_context_len=self.model_config.context_len
                    + extra_max_context_len,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                )
        else:
            # 草稿 worker 与目标 worker 共享 req_to_token_pool
            assert self.is_draft_worker

        # 初始化 token_to_kv_pool（token->KV 缓存池）
        is_nsa_model = is_deepseek_nsa(self.model_config.hf_config)
        # 以下按后端类型（昇腾 NPU / MLA / NSA / 双稀疏 / 混合 SWA / mamba / 普通 MHA）选择不同的 KV 池实现
        if self.server_args.attention_backend == "ascend" and not self.mambaish_config:
            # 昇腾 NPU 后端
            if self.is_hybrid_swa:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMHATokenToKVPool,
                )

                kwargs = {}
                if self.is_hybrid_swa_compress:
                    kwargs = {
                        "swa_head_num": max(
                            1,
                            self.model_config.hf_text_config.swa_num_key_value_heads
                            // get_attention_tp_size(),
                        ),
                        "swa_head_dim": self.model_config.hf_text_config.swa_head_dim,
                        "swa_v_head_dim": self.model_config.hf_text_config.swa_v_head_dim,
                        "v_head_dim": self.model_config.hf_text_config.v_head_dim,
                    }
                self.token_to_kv_pool = SWAKVPool(
                    size=self.full_max_total_num_tokens,
                    size_swa=self.swa_max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    swa_attention_layer_ids=self.model_config.swa_attention_layer_ids,
                    full_attention_layer_ids=self.model_config.full_attention_layer_ids,
                    enable_kvcache_transpose=False,
                    device=self.device,
                    token_to_kv_pool_class=NPUMHATokenToKVPool,
                    **kwargs,
                )
            elif self.use_mla_backend:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMLATokenToKVPool,
                )

                self.token_to_kv_pool = NPUMLATokenToKVPool(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    index_head_dim=(
                        self.model_config.index_head_dim if is_nsa_model else None
                    ),
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
            else:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMHATokenToKVPool,
                )

                self.token_to_kv_pool = NPUMHATokenToKVPool(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
        elif self.use_mla_backend and is_nsa_model:
            # MLA + NSA（DeepSeek V3.2）后端
            nsa_pool_kwargs = dict(
                size=self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                kv_lora_rank=self.model_config.kv_lora_rank,
                qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                layer_num=self.num_effective_layers,
                device=self.device,
                kv_cache_dim=self.calculate_mla_kv_cache_dim(),
                enable_memory_saver=self.server_args.enable_memory_saver,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
                index_head_dim=get_nsa_index_head_dim(self.model_config.hf_config),
            )
            if self.enable_hisparse:
                # HiSparse 稀疏注意力：使用支持主机-设备分层的 NSA 池
                from sglang.srt.mem_cache.sparsity import parse_hisparse_config

                hisparse_cfg = parse_hisparse_config(self.server_args)
                nsa_pool_kwargs["host_to_device_ratio"] = (
                    hisparse_cfg.host_to_device_ratio
                )
                self.token_to_kv_pool = HiSparseNSATokenToKVPool(**nsa_pool_kwargs)
            else:
                self.token_to_kv_pool = NSATokenToKVPool(**nsa_pool_kwargs)
        elif self.use_mla_backend and not self.mambaish_config:
            # 普通 MLA 后端（非 NSA），区分 FP4 与普通两种 KV 池
            assert not is_nsa_model
            if is_float4_e2m1fn_x2(self.kv_cache_dtype):
                self.token_to_kv_pool = MLATokenToKVPoolFP4(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
            else:
                self.token_to_kv_pool = MLATokenToKVPool(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
        elif self.server_args.enable_double_sparsity:
            # 双稀疏（Double Sparsity）KV 池
            self.token_to_kv_pool = DoubleSparseTokenToKVPool(
                self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_num_kv_heads(get_attention_tp_size()),
                head_dim=self.model_config.head_dim,
                layer_num=self.num_effective_layers,
                device=self.device,
                heavy_channel_num=self.server_args.ds_heavy_channel_num,
                enable_memory_saver=self.server_args.enable_memory_saver,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
            )
        else:
            # 默认分支：混合 SWA / mamba 混合线性 / 普通 MHA
            if self.is_hybrid_swa:
                # 混合 SWA：使用 SWAKVPool（全局层与滑动窗口层分别管理）
                kwargs = {}
                if self.is_hybrid_swa_compress:
                    kwargs = {
                        "swa_head_num": max(
                            1,
                            self.model_config.hf_text_config.swa_num_key_value_heads
                            // get_attention_tp_size(),
                        ),
                        "swa_head_dim": self.model_config.hf_text_config.swa_head_dim,
                        "swa_v_head_dim": self.model_config.hf_text_config.swa_v_head_dim,
                        "v_head_dim": self.model_config.hf_text_config.v_head_dim,
                    }
                self.token_to_kv_pool = SWAKVPool(
                    size=self.full_max_total_num_tokens,
                    size_swa=self.swa_max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    swa_attention_layer_ids=self.model_config.swa_attention_layer_ids,
                    full_attention_layer_ids=self.model_config.full_attention_layer_ids,
                    enable_kvcache_transpose=False,
                    device=self.device,
                    **kwargs,
                )
            elif config := self.mambaish_config:
                # mamba 混合线性模型：使用 HybridLinearKVPool
                extra_args = {}
                if self.use_mla_backend:
                    extra_args = {
                        "kv_lora_rank": self.model_config.kv_lora_rank,
                        "qk_rope_head_dim": self.model_config.qk_rope_head_dim,
                    }
                self.token_to_kv_pool = HybridLinearKVPool(
                    page_size=self.page_size,
                    size=self.max_total_num_tokens,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    # if draft worker, we only need 1 attention layer's kv pool
                    full_attention_layer_ids=(
                        [0]
                        if self.is_draft_worker
                        else [
                            i
                            for i in config.full_attention_layer_ids
                            if self.start_layer <= i < self.end_layer
                        ]
                    ),
                    enable_kvcache_transpose=False,
                    device=self.device,
                    mamba_pool=self.req_to_token_pool.mamba_pool,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    use_mla=self.use_mla_backend,
                    start_layer=self.start_layer,
                    **extra_args,
                )
            else:
                # 普通 MHA：区分 FP4 与普通两种 KV 池
                if is_float4_e2m1fn_x2(self.kv_cache_dtype):
                    self.token_to_kv_pool = MHATokenToKVPoolFP4(
                        self.max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        head_num=self.model_config.get_num_kv_heads(
                            get_attention_tp_size()
                        ),
                        head_dim=self.model_config.head_dim,
                        layer_num=self.num_effective_layers,
                        device=self.device,
                        enable_memory_saver=self.server_args.enable_memory_saver,
                        start_layer=self.start_layer,
                        end_layer=self.end_layer,
                        enable_alt_stream=not self.server_args.enable_pdmux,
                        enable_kv_cache_copy=(
                            self.server_args.speculative_algorithm is not None
                        ),
                    )
                else:
                    self.token_to_kv_pool = MHATokenToKVPool(
                        self.max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        head_num=self.model_config.get_num_kv_heads(
                            get_attention_tp_size()
                        ),
                        head_dim=self.model_config.head_dim,
                        layer_num=self.num_effective_layers,
                        device=self.device,
                        enable_memory_saver=self.server_args.enable_memory_saver,
                        start_layer=self.start_layer,
                        end_layer=self.end_layer,
                        enable_alt_stream=not self.server_args.enable_pdmux,
                        enable_kv_cache_copy=(
                            self.server_args.speculative_algorithm is not None
                        ),
                    )

        # 初始化 token_to_kv_pool_allocator（KV 缓存分配器）
        # PD 分离模式下需要排序（need_sort）以保证索引连续
        need_sort = self.server_args.disaggregation_mode in ("decode", "prefill")
        if self.token_to_kv_pool_allocator is None:
            if _is_npu and (
                self.server_args.attention_backend == "ascend"
                or self.hybrid_gdn_config is not None
            ):
                if self.is_hybrid_swa:
                    self.token_to_kv_pool_allocator = SWATokenToKVPoolAllocator(
                        self.full_max_total_num_tokens,
                        self.swa_max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                    )
                else:
                    from sglang.srt.hardware_backend.npu.allocator_npu import (
                        NPUPagedTokenToKVPoolAllocator,
                    )

                    self.token_to_kv_pool_allocator = NPUPagedTokenToKVPoolAllocator(
                        self.max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                    )
            else:
                if self.is_hybrid_swa:
                    self.token_to_kv_pool_allocator = SWATokenToKVPoolAllocator(
                        self.full_max_total_num_tokens,
                        self.swa_max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                    )
                else:
                    if self.enable_hisparse:
                        from sglang.srt.mem_cache.sparsity import (
                            parse_hisparse_config,
                        )

                        hisparse_cfg = parse_hisparse_config(self.server_args)
                        self.token_to_kv_pool_allocator = (
                            HiSparseTokenToKVPoolAllocator(
                                self.max_total_num_tokens,
                                page_size=self.page_size,
                                dtype=self.kv_cache_dtype,
                                device=self.device,
                                kvcache=self.token_to_kv_pool,
                                need_sort=need_sort,
                                host_to_device_ratio=hisparse_cfg.host_to_device_ratio,
                            )
                        )
                    elif self.page_size == 1:
                        # page_size==1：逐 token 分配器
                        self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                            self.max_total_num_tokens,
                            dtype=self.kv_cache_dtype,
                            device=self.device,
                            kvcache=self.token_to_kv_pool,
                            need_sort=need_sort,
                        )
                    else:
                        # page_size>1：分页分配器
                        self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                            self.max_total_num_tokens,
                            page_size=self.page_size,
                            dtype=self.kv_cache_dtype,
                            device=self.device,
                            kvcache=self.token_to_kv_pool,
                            need_sort=need_sort,
                        )

        else:
            # 草稿 worker 复用目标 worker 的分配器
            assert self.is_draft_worker
            if self.is_hybrid_swa:
                # 同步全局->SWA 的索引映射
                assert (
                    self.token_to_kv_pool_allocator.__class__
                    == SWATokenToKVPoolAllocator
                )
                self.token_to_kv_pool.full_to_swa_index_mapping = (
                    self.token_to_kv_pool_allocator.full_to_swa_index_mapping
                )

    def _resolve_token_capacity(self: ModelRunner, profiled_tokens: int) -> int:
        """从探测值计算最终 token 池容量：应用用户上限、页对齐与 PP 同步。"""
        user_limit = self.server_args.max_total_tokens

        # 应用用户指定的上限
        if user_limit is not None:
            if user_limit > profiled_tokens:
                logging.warning(
                    f"max_total_tokens={user_limit} is larger than the profiled value "
                    f"{profiled_tokens}. Use the profiled value instead."
                )
            capacity = min(profiled_tokens, user_limit)
        else:
            capacity = profiled_tokens

        # 对齐到页边界
        page_size = self.server_args.page_size
        capacity = capacity // page_size * page_size

        # 在各 PP rank 间同步（各 rank 可能层数不同），取最小值
        if self.pp_size > 1:
            tensor = torch.tensor(capacity, dtype=torch.int64)
            torch.distributed.all_reduce(
                tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=get_world_group().cpu_group,
            )
            capacity = tensor.item()

        return capacity

    def _resolve_max_num_reqs(self: ModelRunner, token_capacity: int) -> int:
        """根据最终 token 容量计算最大并发请求数（每个 dp worker）。"""
        # 估算池大小（当用户指定 max_running_requests 时作为上限）
        estimated = int(token_capacity / self.model_config.context_len * 512)
        estimated = max(min(estimated, 4096), 2048)

        max_num_reqs = self.server_args.max_running_requests
        if max_num_reqs is not None:
            max_num_reqs = min(max_num_reqs // self.dp_size, estimated)
        else:
            max_num_reqs = min(estimated, token_capacity // 2)

        if self.mambaish_config is not None:
            # mamba 模型另受 mamba 缓存容量限制
            ratio = self._calculate_mamba_ratio()
            max_num_reqs = min(
                max_num_reqs, self.server_args.max_mamba_cache_size // ratio
            )

        return max_num_reqs

    def _apply_memory_pool_config(self: ModelRunner, config: MemoryPoolConfig):
        """应用解析好的 MemoryPoolConfig 并初始化显存池。"""
        self.max_total_num_tokens = config.max_total_num_tokens
        self.max_running_requests = config.max_running_requests
        if self.is_hybrid_swa:
            self.full_max_total_num_tokens = config.full_max_total_num_tokens
            self.swa_max_total_num_tokens = config.swa_max_total_num_tokens

        self._init_pools()

    def _resolve_memory_pool_config(
        self: ModelRunner, pre_model_load_memory: int
    ) -> MemoryPoolConfig:
        """探测 GPU 显存并将所有池参数解析为一个配置对象。"""
        profiled_tokens = self.profile_max_num_token(pre_model_load_memory)
        token_capacity = self._resolve_token_capacity(profiled_tokens)

        full_tokens = None
        swa_tokens = None
        if self.is_hybrid_swa:
            # 混合 SWA 需将总容量拆分为全局层与滑动窗口层两部分
            token_capacity, full_tokens, swa_tokens = self._resolve_hybrid_swa_tokens(
                token_capacity
            )

        return MemoryPoolConfig(
            max_total_num_tokens=token_capacity,
            max_running_requests=self._resolve_max_num_reqs(token_capacity),
            full_max_total_num_tokens=full_tokens,
            swa_max_total_num_tokens=swa_tokens,
            mem_fraction_static=self.server_args.mem_fraction_static,
        )

    def init_memory_pool(self: ModelRunner, pre_model_load_memory: int):
        """初始化显存池的入口：解析配置（草稿 worker 复用目标配置）并创建各池。"""
        if not self.spec_algorithm.is_none() and self.is_draft_worker:
            # 投机解码的草稿 worker 复用目标 worker 传入的显存池配置
            assert (
                self.memory_pool_config is not None
            ), "Draft worker requires memory_pool_config"
        else:
            self.memory_pool_config = self._resolve_memory_pool_config(
                pre_model_load_memory
            )

        self._apply_memory_pool_config(self.memory_pool_config)

        logger.info(
            f"Memory pool end. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )
