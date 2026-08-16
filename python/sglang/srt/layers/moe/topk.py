# Copyright 2024 SGLang Team
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

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import (
    TYPE_CHECKING,
    Callable,
    NamedTuple,
    Optional,
    Protocol,
    TypeGuard,
    runtime_checkable,
)

import torch
import torch.nn.functional as F

try:
    from triton_kernels.matmul_ogs import GatherIndx, RoutingData, ScatterIndx
    from triton_kernels.tensor import make_ragged_tensor_metadata
    from triton_kernels.topk import topk as triton_kernels_topk

    def routing(
        logits,
        n_expts_act,
        sm_first=False,
        expt_indx=None,
        simulated_ep=1,
        n_rows=None,
    ):
        """调用 Triton Kernels 完成 Top-K 路由并构造 dispatch/combine 索引。

        输入输出示例：``logits.shape=[4, 8]``、``n_expts_act=2`` 时，为 4 个 token
        各选 2 个专家，返回 ``RoutingData`` 以及供专家分发和结果回收使用的 gather、
        scatter 索引；总 token-expert assignment 数为 8。
        """
        if simulated_ep != 1:
            raise NotImplementedError(
                "simulated_ep routing is not supported with triton_kernels 3.6.0"
            )

        if sm_first:
            logits = torch.softmax(logits, dim=-1)

        sparse_logits = triton_kernels_topk(
            logits,
            n_expts_act,
            apply_softmax=not sm_first,
            y_indx=expt_indx,
            n_rows=n_rows,
        )
        dispatch_indx = sparse_logits.mask_metadata.row_sorted_indx
        combine_indx = sparse_logits.mask_metadata.col_sorted_indx
        ragged_metadata = make_ragged_tensor_metadata(
            sparse_logits.mask_metadata.col_sum, dispatch_indx.shape[0]
        )
        gate_scal = sparse_logits.vals.flatten()[combine_indx]
        routing_data = RoutingData(
            gate_scal,
            ragged_metadata.slice_sizes,
            logits.shape[-1],
            n_expts_act,
            ragged_metadata,
        )
        gather_indx = GatherIndx(combine_indx, dispatch_indx)
        scatter_indx = ScatterIndx(dispatch_indx, combine_indx)
        return routing_data, gather_indx, scatter_indx

except ImportError:
    pass

from sglang.jit_kernel.dsv4 import mask_topk_ids
from sglang.srt.distributed import (
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
    get_tp_group,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.eplb import expert_location_dispatch
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import (
    ExpertLocationDispatchInfo,
    topk_ids_logical_to_physical,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe import get_moe_runner_backend
from sglang.srt.layers.moe.utils import is_deepep_class_backend
from sglang.srt.layers.utils import MultiPlatformOp
from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    get_compiler_backend,
    is_cpu,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    is_xpu,
)
from sglang.srt.utils.patch_torch import register_fake_if_exists

_SGLANG_EXPERIMENTAL_LORA_OPTI = envs.SGLANG_EXPERIMENTAL_LORA_OPTI.get()

if TYPE_CHECKING:
    from sglang.srt.layers.quantization import QuantizationConfig


logger = logging.getLogger(__name__)
_is_cuda = is_cuda()
_is_hip = is_hip()
_is_cpu = is_cpu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_xpu = is_xpu()
_is_npu = is_npu()
_is_xpu = is_xpu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_musa = is_musa()

if _is_cuda:
    from sgl_kernel import moe_fused_gate

    try:
        from flashinfer.fused_moe import fused_topk_deepseek as _fused_topk_deepseek

        from sglang.srt.utils.custom_op import register_custom_op

        @register_custom_op(
            op_name="fused_topk_deepseek",
            mutates_args=["topk_weights", "topk_ids"],
        )
        def fused_topk_deepseek(
            gating_output: torch.Tensor,
            correction_bias: torch.Tensor,
            num_expert_group: int,
            topk_group: int,
            topk: int,
            scaling_factor: float,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            renormalize: bool,
        ) -> None:
            _fused_topk_deepseek(
                gating_output,
                correction_bias,
                num_expert_group,
                topk_group,
                topk,
                scaling_factor,
                topk_weights,
                topk_ids,
                renormalize,
            )

    except ImportError:
        fused_topk_deepseek = None

    try:
        from sgl_kernel import kimi_k2_moe_fused_gate
    except ImportError as e:
        pass

if _is_cuda or _is_hip or _is_xpu:
    from sgl_kernel import topk_softmax

    try:
        from sgl_kernel import topk_sigmoid
    except ImportError:
        pass
if _use_aiter:
    try:
        from aiter import biased_grouped_topk as aiter_biased_grouped_topk
        from aiter.fused_moe import fused_topk as aiter_fused_topk
    except ImportError:
        raise ImportError("aiter is required when SGLANG_USE_AITER is set to True")
if _is_musa:
    try:
        from mate import moe_fused_gate
    except ImportError as e:
        raise ImportError("mate is required for the biased grouped topk.")

    from sglang.srt.hardware_backend.musa.kernels.topk import topk_sigmoid, topk_softmax

# -------------------------------- TopKConfig ---------------------------------------


@dataclass
class TopKConfig:
    """MoE 路由 Top-K 的统一配置。

    该配置只描述“如何从 router logits 选择专家以及如何表达输出”，不保存模型权重。
    ``select_experts`` 会读取这些字段，在普通、分组、带偏置、融合或自定义路由路径间分派。

    参数：
        top_k:
            每个 token 最终激活的专家总数。若启用融合共享专家，该数值同时包含路由专家
            和共享专家，例如 ``top_k=8``、``num_fused_shared_experts=1`` 时，实际从
            router logits 中选择 7 个路由专家，再追加 1 个共享专家。必须大于等于
            ``num_fused_shared_experts``，且路由部分不能超过可路由专家总数。
        use_grouped_topk:
            是否先选择专家组，再从入选组中选择专家。DeepSeek 系列常启用，用于限制
            token 的路由范围及改善跨节点通信局部性。启用时 ``topk_group`` 和
            ``num_expert_group`` 必须同时设置。
        topk_group:
            每个 token 允许进入的专家组数量，仅在 grouped Top-K 中有效。应满足
            ``1 <= topk_group <= num_expert_group``。
        num_expert_group:
            将全部路由专家划分成多少组。一般要求专家总数可被该值整除；每组专家数为
            ``num_experts // num_expert_group``。
        renormalize:
            是否在选出 Top-K 后重新归一化路由权重。为 ``True`` 时，普通路由专家权重
            通常满足每个 token 行和为 1；若融合共享专家，则通常只归一化路由专家部分。
        num_fused_shared_experts:
            附加到 Top-K 输出中的融合共享专家数量。共享专家不参与普通 router 竞争，
            而是在路由专家之后追加；其 ID、权重以及 DeepEP 映射由后处理阶段生成。
        custom_routing_function:
            可选的自定义专家选择函数。设置后跳过内置 grouped/fused Top-K 路径，由该函数
            接收 ``hidden_states``、``gating_output``、``topk``、``renormalize`` 并返回
            ``(topk_weights, topk_ids)``。
        correction_bias:
            形状通常为 ``[num_experts]`` 的专家校正偏置，用于负载均衡。它只加到“选择
            专家所用的分数”上，最终组合权重仍从未加偏置的 score 中取得，避免改变模型
            原始门控概率语义。
        torch_native:
            是否强制使用 PyTorch 原生 Top-K 实现。通常由平台入口动态设置，用于 CPU、
            XPU 回退、调试或当前融合 kernel 不支持的形状。
        routed_scaling_factor:
            路由专家输出或权重的缩放因子。是否在 Top-K 阶段直接乘入权重，由
            ``apply_routed_scaling_factor_on_output`` 决定；否则通常在后续 MoE 输出阶段应用。
        apply_routed_scaling_factor_on_output:
            是否在 Top-K 结果上直接应用 ``routed_scaling_factor``。必须结合具体后端使用，
            部分 CPU/原生路径尚不支持；为 ``False`` 时仅保存缩放因子供后续阶段处理。
        fused_shared_experts_scaling_factor:
            融合共享专家的独立缩放因子，主要用于 AMD/AITER 路径；为 ``None`` 时按 1.0
            处理。它与路由专家的 ``routed_scaling_factor`` 相互独立。
        output_format:
            可选的强制输出格式。``STANDARD`` 立即物化权重和 ID；``TRITON_KERNEL`` 返回
            Triton 路由元数据；``BYPASSED`` 延迟到后续融合 MoE kernel 内执行 Top-K。
            为 ``None`` 时根据当前 MoE runner 后端自动选择。
        scoring_func:
            router logits 的打分函数。常见值为 ``"softmax"``、``"sigmoid"`` 和
            ``"sqrtsoftplus"``；并非所有硬件路径都支持全部取值。

    输入输出示例：
        ``TopKConfig(top_k=2, renormalize=True)`` 配合 ``router_logits.shape=[4,8]``，
        标准路径输出 ``topk_ids/topk_weights.shape=[4,2]``，每个 token 选择 2 个专家。

        ``TopKConfig(top_k=8, use_grouped_topk=True, num_expert_group=8,
        topk_group=4, num_fused_shared_experts=1)`` 表示先从 8 个专家组中选 4 组，
        再选择 7 个路由专家并追加 1 个共享专家，最终输出形状为 ``[num_tokens,8]``。
    """

    top_k: int
    use_grouped_topk: bool = False
    topk_group: Optional[int] = None
    num_expert_group: Optional[int] = None
    renormalize: bool = True
    num_fused_shared_experts: int = 0
    custom_routing_function: Optional[Callable] = None
    correction_bias: Optional[torch.Tensor] = None
    torch_native: bool = False
    routed_scaling_factor: Optional[float] = None
    apply_routed_scaling_factor_on_output: bool = False
    fused_shared_experts_scaling_factor: Optional[float] = None
    output_format: Optional[TopKOutputFormat] = None
    scoring_func: str = "softmax"


# -------------------------------- TopKOutput ---------------------------------------


class TopKOutputChecker:

    @staticmethod
    def format_is_standard(topk_output: TopKOutput) -> TypeGuard[StandardTopKOutput]:
        # ===== 待重构区域 =====
        # 实验性的 Top-K 与 pack 融合输出类型仅在总开关开启时存在。
        if _SGLANG_EXPERIMENTAL_LORA_OPTI:
            return isinstance(
                topk_output, (StandardTopKOutput, StandardTopKOutputPacked)
            )
        # ===== 待重构区域结束 =====
        return isinstance(topk_output, StandardTopKOutput)

    @staticmethod
    def format_is_triton_kernels(
        topk_output: TopKOutput,
    ) -> TypeGuard[TritonKernelTopKOutput]:
        return isinstance(topk_output, TritonKernelTopKOutput)

    @staticmethod
    def format_is_bypassed(topk_output: TopKOutput) -> TypeGuard[BypassedTopKOutput]:
        return isinstance(topk_output, BypassedTopKOutput)


class TopKOutputFormat(IntEnum):
    STANDARD = auto()
    TRITON_KERNEL = auto()
    BYPASSED = auto()


@runtime_checkable
class TopKOutput(Protocol):
    """不同 Top-K 输出格式共同遵循的协议。"""

    @property
    def format(self) -> TopKOutputFormat:
        """返回当前输出对象的格式枚举值。"""
        ...


class StandardTopKOutput(NamedTuple):
    """标准 Top-K 输出：权重、专家 ID 和原始路由 logits。

    输入输出示例：2 个 token、Top-2 路由时，``topk_weights`` 和 ``topk_ids``
    的形状均为 ``[2, 2]``，``router_logits`` 的形状为 ``[2, num_experts]``。
    """

    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor

    @property
    def format(self) -> TopKOutputFormat:
        return TopKOutputFormat.STANDARD


# ===== 待重构区域 =====
# 实验性的 Top-K 与 pack 融合输出载体（SGLANG_OPT_LORA_FUSED_TOPK_PACK）：
# gating kernel 会同时生成 FlashInfer routed-MoE 所需的 packed Top-K。这里使用独立类型，
# 而不是给 StandardTopKOutput 添加第 4 个字段，以保持现有 ``a, b, _ = topk_output``
# 三元组解包兼容；只有受开关控制的实验性 MoE dispatch 会读取 packed_topk_ids。
class StandardTopKOutputPacked(NamedTuple):
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor
    packed_topk_ids: torch.Tensor

    @property
    def format(self) -> TopKOutputFormat:
        return TopKOutputFormat.STANDARD


# ===== END TO BE REFACTORED ====


class TritonKernelTopKOutput(NamedTuple):
    """Triton Kernels 专用 Top-K 输出，保存路由元数据及 gather/scatter 索引。"""

    routing_data: RoutingData
    gather_indx: GatherIndx
    scatter_indx: ScatterIndx

    @property
    def format(self) -> TopKOutputFormat:
        return TopKOutputFormat.TRITON_KERNEL


class BypassedTopKOutput(NamedTuple):
    """延迟路由输出：暂不物化 Top-K 张量，由后续 MoE kernel 内部完成路由。"""

    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    topk_config: TopKConfig
    num_token_non_padded: Optional[torch.Tensor] = None
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None

    @property
    def format(self) -> TopKOutputFormat:
        return TopKOutputFormat.BYPASSED

    def to_standard(self, layer_id: Optional[int] = None) -> StandardTopKOutput:
        """把延迟路由输出物化为标准的 Top-K 张量。

        供需要显式 ``topk_ids`` / ``topk_weights``、无法在内部执行路由的 MoE kernel
        使用。输入输出示例：若 ``router_logits.shape=[3, 8]`` 且 ``top_k=2``，返回的
        ``topk_ids`` 与 ``topk_weights`` 形状均为 ``[3, 2]``。
        """
        return select_experts(
            hidden_states=self.hidden_states,
            router_logits=self.router_logits,
            topk_config=self.topk_config,
            layer_id=layer_id,
            num_token_non_padded=self.num_token_non_padded,
            expert_location_dispatch_info=self.expert_location_dispatch_info,
        )


def _make_round_robin_expert_ids(
    num_tokens: int,
    topk: int,
    num_experts: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    layer_id: Optional[int] = None,
) -> torch.Tensor:
    """为基准测试生成确定性的轮询专家 ID。

    输入输出示例：``num_tokens=3, topk=2, num_experts=4`` 时，步长为 2，返回
    ``[[0,2], [1,3], [2,0]]``；``layer_id`` 会整体平移起始专家。
    """
    if topk == 0:
        return torch.empty((num_tokens, 0), device=device, dtype=dtype)

    step = max(num_experts // topk, 1)
    layer_offset = 0 if layer_id is None else layer_id
    offsets = torch.arange(num_tokens, device=device, dtype=dtype).unsqueeze(1)
    steps = torch.arange(topk, device=device, dtype=dtype).unsqueeze(0) * step
    return (offsets + layer_offset + steps) % num_experts


# -------------------------------- TopK ---------------------------------------


class TopK(MultiPlatformOp):
    """跨平台 MoE Top-K 路由算子。

    参数：
        top_k: 每个 token 最终选择的专家总数，包含融合的共享专家。
        num_fused_shared_experts: 融合共享专家数量，可在 TP 或 EP 模式下启用。
        routed_scaling_factor: 路由专家权重的缩放因子。
        fused_shared_experts_scaling_factor: AMD 平台上融合共享专家的缩放因子。

    输入输出示例：``TopK(top_k=2)`` 接收 ``hidden_states.shape=[4,d]`` 和
    ``router_logits.shape=[4,8]``，标准路径输出形状均为 ``[4,2]`` 的专家权重和 ID，
    表示每个 token 从 8 个专家中选择 2 个。
    """

    def __init__(
        self,
        top_k: int,
        *,
        layer_id: Optional[int] = None,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        renormalize: bool = True,
        num_fused_shared_experts: int = 0,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        correction_bias: Optional[torch.Tensor] = None,
        quant_config: Optional[QuantizationConfig] = None,
        routed_scaling_factor: Optional[float] = None,
        apply_routed_scaling_factor_on_output: Optional[bool] = False,
        output_format: Optional[TopKOutputFormat] = None,
        fused_shared_experts_scaling_factor: Optional[float] = None,
        is_fp4_experts: bool = False,
    ):
        # 注意：scoring_func 当前暂未在此处直接使用，但为后续扩展保留。
        # 详情参见 https://github.com/sgl-project/sglang/pull/4505。
        super().__init__()

        if use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None

        self.layer_id = layer_id
        from sglang.srt.server_args import get_global_server_args

        self.enable_deepep_waterfill = (
            num_fused_shared_experts > 0
            and get_global_server_args().enable_deepep_waterfill
        )

        self.deepep_waterfill_balancer = None
        if self.enable_deepep_waterfill:
            # TODO(ch-wan)：重构共享专家融合与路由 Top-K 融合逻辑。
            top_k -= num_fused_shared_experts
            num_fused_shared_experts = 0
            output_format = TopKOutputFormat.STANDARD

        # 仅用于 flashinfer_mxfp4 后端：True 表示输出 STANDARD，交给
        # Mxfp4FlashinferTrtllmMoEMethod 消费；False 表示 BYPASSED，由 FlashInfer 自身的
        # MXFP4 kernel 在内部完成路由。其他后端忽略该字段。
        self.is_fp4_experts = is_fp4_experts
        self.topk_config = TopKConfig(
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            num_fused_shared_experts=num_fused_shared_experts,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
            routed_scaling_factor=routed_scaling_factor,
            apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            fused_shared_experts_scaling_factor=fused_shared_experts_scaling_factor,
            output_format=output_format,
            scoring_func=scoring_func,
        )

    def _apply_deepep_waterfill(
        self, topk_output: TopKOutput, num_tokens: int
    ) -> TopKOutput:
        """按 DeepEP waterfill 方案扩展 Top-K 结果以改善专家负载均衡。

        输入输出示例：未启用或未配置 balancer 时原样返回；启用后将标准 Top-K 输出
        交给 ``expand_topk``，其具体列数由 waterfill 策略决定。
        """
        if self.enable_deepep_waterfill and self.deepep_waterfill_balancer is None:
            raise RuntimeError(
                "DeepEP waterfill TopK must be prepared by ModelRunner before forward."
            )
        if self.deepep_waterfill_balancer is None:
            return topk_output
        assert TopKOutputChecker.format_is_standard(topk_output)
        return self.deepep_waterfill_balancer.expand_topk(topk_output, num_tokens)

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        num_token_non_padded: Optional[torch.Tensor] = None,
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    ) -> TopKOutput:
        """使用纯 PyTorch 路径执行 Top-K，主要用于通用回退和可移植执行。

        输入输出示例：``hidden_states=[M,d]``、``router_logits=[M,E]``、Top-K 为 K 时，
        返回标准输出，其中 ``topk_ids/topk_weights`` 均为 ``[M,K]``。
        """
        self.topk_config.torch_native = True
        topk_output = select_experts(
            hidden_states=hidden_states,
            layer_id=self.layer_id,
            router_logits=router_logits,
            topk_config=self.topk_config,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )
        return self._apply_deepep_waterfill(topk_output, hidden_states.shape[0])

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        num_token_non_padded: Optional[torch.Tensor] = None,
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    ) -> TopKOutput:
        """执行 CUDA/ROCm Top-K，并根据 MoE runner 选择输出格式。

        标准后端立即生成 ``[M,K]`` 的权重和 ID；Triton Kernels 返回路由索引结构；
        BYPASSED 后端仅封装输入，让后续融合 kernel 完成路由。
        输入输出示例：``router_logits=[4,8]``、Top-2 时，标准输出的权重和 ID 为 ``[4,2]``。
        """
        if self.topk_config.output_format is not None:
            output_format = self.topk_config.output_format
        elif get_moe_runner_backend().is_triton_kernels():
            output_format = TopKOutputFormat.TRITON_KERNEL
        # ===== TO BE REFACTORED ====
        elif get_moe_runner_backend().is_experimental_sgl_trtllm():
            try:
                from sglang.srt.server_args import get_global_server_args

                use_standard_for_lora = bool(get_global_server_args().enable_lora)
            except ValueError:
                use_standard_for_lora = False
            output_format = (
                TopKOutputFormat.STANDARD
                if use_standard_for_lora
                else TopKOutputFormat.BYPASSED
            )
        # ===== 待重构区域结束 =====
        elif get_moe_runner_backend().is_flashinfer_trtllm() or (
            get_moe_runner_backend().is_flashinfer_mxfp4() and not self.is_fp4_experts
        ):
            output_format = TopKOutputFormat.BYPASSED
        else:
            output_format = TopKOutputFormat.STANDARD

        if output_format == TopKOutputFormat.TRITON_KERNEL:
            # renormalize=True 等价于 sm_first=False：先选 Top-K，再在选中项内归一化。
            routing_data, gather_idx, scatter_idx = routing(
                router_logits,
                self.topk_config.top_k,
                sm_first=not self.topk_config.renormalize,
            )
            return TritonKernelTopKOutput(routing_data, gather_idx, scatter_idx)
        elif output_format == TopKOutputFormat.BYPASSED:
            return BypassedTopKOutput(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_config=self.topk_config,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
            )
        else:
            self.topk_config.torch_native = False
            with use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            ):
                topk_output = select_experts(
                    hidden_states=hidden_states,
                    layer_id=self.layer_id,
                    router_logits=router_logits,
                    topk_config=self.topk_config,
                    num_token_non_padded=num_token_non_padded,
                    expert_location_dispatch_info=expert_location_dispatch_info,
                )
        return self._apply_deepep_waterfill(topk_output, hidden_states.shape[0])

    def forward_cpu(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        num_token_non_padded: Optional[torch.Tensor] = None,
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    ) -> TopKOutput:
        """执行 CPU Top-K 路由。

        输入输出示例：``hidden_states=[M,d]``、``router_logits=[M,E]``、Top-K 为 K 时，
        返回标准输出，其中专家权重和 ID 的形状均为 ``[M,K]``。
        """
        topk_output = select_experts(
            hidden_states=hidden_states,
            layer_id=self.layer_id,
            router_logits=router_logits,
            topk_config=self.topk_config,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )
        return self._apply_deepep_waterfill(topk_output, hidden_states.shape[0])

    def forward_npu(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        num_token_non_padded: Optional[torch.Tensor] = None,
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    ) -> TopKOutput:
        """把 NPU Top-K 路由委托给 NPU 专用融合实现。

        输入 ``router_logits=[M,E]`` 时，返回格式由 ``fused_topk_npu`` 决定；标准配置下
        权重和专家 ID 均为 ``[M,K]``。
        """
        from sglang.srt.hardware_backend.npu.moe.topk import fused_topk_npu

        return fused_topk_npu(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_config=self.topk_config,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
            layer_id=self.layer_id,
        )

    def empty_topk_output(self, device: torch.device) -> TopKOutput:
        """创建零 token 批次对应的空 Top-K 输出。

        输入输出示例：Top-2 且无共享专家时，返回权重和 ID 形状 ``[0,2]``；ID dtype
        为 int32，权重为 float32。DeepEP 共享专家路径会追加相应的空列。
        """
        topk = self.topk_config.top_k - self.topk_config.num_fused_shared_experts
        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            topk_weights = torch.empty((0, topk), dtype=torch.float32, device=device)
            topk_ids = torch.full((0, topk), -1, dtype=torch.int32, device=device)
        # FIXME：router_logits 理想形状应为 (0, num_experts)，这里暂以 topk 代替。
        router_logits = torch.empty((0, topk), dtype=torch.float32, device=device)
        topk_output = StandardTopKOutput(topk_weights, topk_ids, router_logits)
        if self.topk_config.num_fused_shared_experts > 0 and is_deepep_class_backend():
            n = self.topk_config.num_fused_shared_experts
            topk_output = topk_output._replace(
                topk_ids=topk_output.topk_ids.new_empty(
                    (0, topk_output.topk_ids.shape[-1] + n)
                ),
                topk_weights=topk_output.topk_weights.new_empty(
                    (0, topk_output.topk_weights.shape[-1] + n)
                ),
            )
        return self._apply_deepep_waterfill(topk_output, 0)

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        num_token_non_padded: Optional[torch.Tensor] = None,
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    ) -> TopKOutput:
        """执行 XPU Top-K，并按 K 和专家数判断使用专用 kernel 还是原生回退。

        Top-K 不超过 8 且专家数不超过 256 时使用 XPU kernel；否则使用 PyTorch 路径。
        输出权重和专家 ID 的形状均为 ``[M,K]``。
        """
        self.topk_config.torch_native = True
        # XPU Top-K kernel 的支持范围：
        #   - 支持 topk_softmax 和 topk_sigmoid；
        #   - 最多支持 Top-8 和 256 个专家。
        self.topk_config.torch_native = not (
            self.topk_config.top_k <= 8 and router_logits.shape[1] <= 256
        )

        return select_experts(
            hidden_states=hidden_states,
            layer_id=self.layer_id,
            router_logits=router_logits,
            topk_config=self.topk_config,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )


# ------------------------------- TopK implementation -------------------------------------


def fused_topk_torch_native(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: torch.Tensor = None,
    scoring_func: str = "softmax",
):
    """用纯 PyTorch 从路由 logits 中选择每个 token 的 Top-K 专家。

    ``correction_bias`` 只影响专家选择，不直接改变最终权重；最终权重从未加偏置的
    score 中 gather。输入输出示例：``gating_output.shape=[3,8]``、``topk=2`` 时，
    返回 ``topk_weights/topk_ids``，形状均为 ``[3,2]``；归一化时每行权重和为 1。
    """

    def scoring_func_impl(gating_output: torch.Tensor) -> torch.Tensor:
        if scoring_func == "softmax":
            return gating_output.softmax(dim=-1)
        elif scoring_func == "sigmoid":
            return gating_output.sigmoid()
        else:
            raise ValueError(f"Invalid scoring function: {scoring_func}")

    if correction_bias is not None:
        n_routed_experts = gating_output.shape[-1]
        scores = scoring_func_impl(gating_output)
        scores_for_choice = scores.view(
            -1, n_routed_experts
        ) + correction_bias.unsqueeze(0)
        topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_ids)
    else:
        assert (
            hidden_states.shape[0] == gating_output.shape[0]
        ), f"Number of tokens mismatch, {hidden_states.shape=} vs {gating_output.shape=}"
        M, _ = hidden_states.shape
        topk_weights = torch.empty(
            M, topk, dtype=torch.float32, device=hidden_states.device
        )
        topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
        topk_weights = scoring_func_impl(gating_output.float())
        topk_weights, topk_ids = torch.topk(topk_weights, topk, dim=-1)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


def fused_topk_softmax_torch_raw_logits(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    """直接在原始 logits 上选 Top-K，再按需对选中 logits 做 softmax。

    与先对全部专家做 softmax 再选择相比，该路径避免计算完整 softmax。输入输出示例：
    输入 ``[M,E]``、``topk=K``，返回形状均为 ``[M,K]`` 的 float32 权重和 int32 ID。
    """
    assert (
        hidden_states.shape[0] == gating_output.shape[0]
    ), f"Number of tokens mismatch, {hidden_states.shape=} vs {gating_output.shape=}"

    _, topk_ids = torch.topk(gating_output, k=topk, dim=-1, sorted=False)
    logits = gating_output.float()
    topk_weights = logits.gather(1, topk_ids)
    if renormalize:
        topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def fused_topk_cpu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: torch.Tensor = None,
    scoring_func: str = "softmax",
):
    # TODO：为 CPU 增加覆盖完整功能的 C++ kernel。
    # 当前 topk_softmax_cpu 只支持不带 correction bias 的普通 softmax 打分；
    # 其他情况回退到 PyTorch 原生实现，例如 MiniMax 会同时设置 correction_bias
    # 和特殊 scoring_func。
    if correction_bias is not None or scoring_func != "softmax":
        return fused_topk_torch_native(
            hidden_states,
            gating_output,
            topk,
            renormalize,
            correction_bias=correction_bias,
            scoring_func=scoring_func,
        )

    topk_weights, topk_ids = torch.ops.sgl_kernel.topk_softmax_cpu(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
    )
    return topk_weights, topk_ids


def apply_topk_weights_cpu(need_apply, topk_weights, inputs):
    """在 CPU 上提前把路由权重乘入专家输入，并清空后续权重语义。

    输入输出示例：``inputs=[2.,4.]``、``topk_weights=[0.5,0.25]`` 且启用时，
    返回加权输入 ``[1.,1.]`` 和全 1 权重；禁用时原样返回。
    """
    if not need_apply:
        return inputs, topk_weights

    # TODO：把下面的加权与权重清理融合进 fused_experts_cpu kernel。
    inputs = inputs * topk_weights.to(inputs.dtype)
    topk_weights = torch.ones_like(
        topk_weights, dtype=torch.float32
    )  # 权重已经乘入 inputs，后续用全 1 避免重复加权。

    return inputs, topk_weights


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: Optional[torch.Tensor] = None,
    scoring_func: str = "softmax",
    packed_out: Optional[torch.Tensor] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
):
    """按平台能力调用融合 kernel，完成 softmax/sigmoid 打分与 Top-K 选择。

    输入输出示例：``hidden_states.shape=[M,d]``、``gating_output.shape=[M,E]``、
    ``topk=K`` 时，返回 ``topk_weights:[M,K]`` 和 ``topk_ids:[M,K]``。
    correction bias 可改变入选专家，但权重仍来自原始打分函数。
    """
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    M, _ = hidden_states.shape

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)

    if scoring_func == "softmax":
        if _use_aiter:

            # 使用 AITER fused_topk 自动分派到适合当前形状和硬件的 kernel。
            topk_weights, topk_ids = aiter_fused_topk(
                hidden_states,
                gating_output,
                topk,
                renormalize,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
            )
        # ===== TO BE REFACTORED ====
        elif packed_out is not None:
            # 融合 gating 与 routed pack：一个 JIT kernel 在一次 launch 中同时写出
            # topk_weights、topk_ids 和 FlashInfer 使用的 packed Top-K。
            from sglang.jit_kernel.trtllm_lora_temp.topk_softmax_pack import (
                topk_softmax_pack,
            )

            topk_softmax_pack(
                topk_weights,
                topk_ids,
                packed_out,
                gating_output,
                renormalize,
                num_token_non_padded=num_token_non_padded,
            )
        # ===== 待重构区域结束 =====
        else:
            topk_softmax(
                topk_weights,
                topk_ids,
                gating_output,
                renormalize,
            )
    elif scoring_func == "sigmoid":
        if _use_aiter and correction_bias is not None:
            aiter_biased_grouped_topk(
                gating_output,
                correction_bias.to(dtype=gating_output.dtype),
                topk_weights,
                topk_ids,
                num_expert_group=1,
                topk_group=1,
                need_renorm=renormalize,
            )
        else:
            topk_sigmoid(
                topk_weights,
                topk_ids,
                gating_output,
                renormalize,
                correction_bias,
            )
    else:
        raise ValueError(f"Invalid scoring function: {scoring_func}")

    return topk_weights, topk_ids


# DeepSeek V2/V3/R1 系列模型使用该分组 Top-K 实现。
@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def grouped_topk_gpu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    """先选择专家组，再仅从入选组内选择 Top-K 专家。

    每组包含 ``num_experts/num_expert_group`` 个专家，先以组内最大 score 作为组分数，
    选出 ``topk_group`` 个组，再屏蔽其他组。输入输出示例：8 个专家分 4 组、选 2 组、
    最终 Top-2 时，输出形状为 ``[num_tokens,2]``，ID 只来自两个入选组。
    """
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = torch.softmax(gating_output, dim=-1)
    num_token = scores.shape[0]
    num_experts = scores.shape[1]
    group_scores = (
        scores.view(num_token, num_expert_group, -1).max(dim=-1).values
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    topk_weights, topk_ids = torch.topk(
        tmp_scores,
        k=topk,
        dim=-1,
        sorted=(True if num_fused_shared_experts > 0 else False),
    )
    if num_fused_shared_experts:
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        if routed_scaling_factor is not None:
            topk_weights[:, -1] = (
                topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor
            )

    if renormalize:
        topk_weights_sum = (
            topk_weights.sum(dim=-1, keepdim=True)
            if num_fused_shared_experts == 0
            else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        )
        topk_weights = topk_weights / topk_weights_sum
        if apply_routed_scaling_factor_on_output:
            topk_weights *= routed_scaling_factor

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)

    return topk_weights, topk_ids


def grouped_topk_cpu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    """调用 CPU 融合 kernel 执行分组 Top-K。

    输入 ``gating_output=[M,E]``、Top-K 为 K 时，返回 ``[M,K]`` 的权重和专家 ID；
    当前 CPU kernel 不支持在输出端额外应用 routed scaling factor。
    """
    assert not apply_routed_scaling_factor_on_output
    return torch.ops.sgl_kernel.grouped_topk_cpu(
        hidden_states,
        gating_output,
        topk,
        renormalize,
        num_expert_group,
        topk_group,
        num_fused_shared_experts,
        routed_scaling_factor,
        # CPU kernel 尚不支持 num_token_non_padded，因此必须传 None。
        num_token_non_padded=None,
    )


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def kimi_k2_biased_topk_impl(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: Optional[float] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    """针对 ``num_expert_group=1`` 的优化版本，例如 384 专家的 Kimi K2。

    只有一个组时无需构造组 mask，可直接在加 correction bias 后的分数上选 Top-K。
    本函数假设 ``num_fused_shared_experts=0``。

    输入输出示例：``gating_output.shape=[M,384]``、``topk=8`` 时，返回形状为
    ``[M,8]`` 的权重和专家 ID；bias 参与选择，原始 sigmoid score 用作权重。
    """
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = gating_output.sigmoid()
    num_token = scores.shape[0]

    # 只有一个专家组时无需构造组 mask，直接加 correction bias 得到选择分数。
    tmp_scores = scores.view(num_token, -1) + correction_bias.unsqueeze(0)

    # 直接选择 Top-K；没有融合共享专家时不依赖排序结果。
    _, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    topk_weights = scores.gather(1, topk_ids)

    if renormalize:
        topk_weights_sum = topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights / topk_weights_sum
        if apply_routed_scaling_factor_on_output:
            topk_weights *= routed_scaling_factor

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)
    return topk_weights, topk_ids


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def biased_topk_impl(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    scoring_func: str = "sigmoid",
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    """执行带专家 correction bias 的非分组 Top-K。

    bias 仅用于平衡专家选择，最终权重从未加 bias 的 score 中取得。输入输出示例：
    ``gating_output=[M,E]``、``correction_bias=[E]``、``topk=K`` 时，返回
    ``topk_weights/topk_ids=[M,K]``。
    """
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    if scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    elif scoring_func == "sqrtsoftplus":
        scores = torch.nn.functional.softplus(gating_output).sqrt()

    num_token = scores.shape[0]
    num_experts = scores.shape[1]

    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
    _, topk_ids = torch.topk(
        scores_for_choice,
        k=topk,
        dim=-1,
        sorted=(True if num_fused_shared_experts > 0 else False),
    )
    topk_weights = scores.gather(1, topk_ids)

    if num_fused_shared_experts:
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        if routed_scaling_factor is not None:
            topk_weights[:, -1] = (
                topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor
            )

    if renormalize:
        topk_weights_sum = (
            topk_weights.sum(dim=-1, keepdim=True)
            if num_fused_shared_experts == 0
            else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        )
        topk_weights = topk_weights / topk_weights_sum
        if apply_routed_scaling_factor_on_output:
            topk_weights *= routed_scaling_factor

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)
    return topk_weights, topk_ids


def biased_topk_jit_kernel_impl(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    scoring_func: str = "sigmoid",
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    if _use_aiter and scoring_func == "sqrtsoftplus" and num_fused_shared_experts == 0:
        from aiter import topk_gating

        num_tokens = gating_output.shape[0]
        topk_weights = torch.empty(
            (num_tokens, topk), dtype=torch.float32, device=gating_output.device
        )
        topk_ids = torch.empty(
            (num_tokens, topk), dtype=torch.int32, device=gating_output.device
        )

        topk_gating(
            topk_weights,
            topk_ids,
            gating_output,
            correction_bias,
            renormalize,
            routed_scaling_factor,
            score_func="sqrtsoftplus",
        )

        return topk_weights, topk_ids

    else:
        from sglang.jit_kernel.moe_fused_gate import moe_fused_gate

        topk_weights, topk_ids = moe_fused_gate(
            gating_output,
            correction_bias,
            topk=topk,
            scoring_func=scoring_func,
            num_fused_shared_experts=num_fused_shared_experts,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
        )
        topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(
            torch.int32
        )
        return topk_weights, topk_ids


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def biased_grouped_topk_impl(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    """执行带 correction bias 的分组 Top-K 原生实现。

    每个组取分数最高的两个专家之和作为组分数，先选组再选专家；bias 只影响选择，
    最终权重来自未加 bias 的 sigmoid score。输入 ``[M,E]``、Top-K=K 时返回 ``[M,K]``。
    """
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = gating_output.sigmoid()
    num_token = scores.shape[0]
    num_experts = scores.shape[1]
    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores_for_choice.masked_fill(
        ~score_mask.bool(), float("-inf")
    )  # [n, e]
    _, topk_ids = torch.topk(
        tmp_scores,
        k=topk,
        dim=-1,
        sorted=(True if num_fused_shared_experts > 0 else False),
    )
    topk_weights = scores.gather(1, topk_ids)

    if num_fused_shared_experts:
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        if routed_scaling_factor is not None:
            topk_weights[:, -1] = (
                topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor
            )

    if renormalize:
        topk_weights_sum = (
            topk_weights.sum(dim=-1, keepdim=True)
            if num_fused_shared_experts == 0
            else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        )
        topk_weights = topk_weights / topk_weights_sum
        if apply_routed_scaling_factor_on_output:
            topk_weights *= routed_scaling_factor

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)

    return topk_weights, topk_ids


def is_power_of_two(n):
    """判断正整数是否为 2 的幂，例如 1、2、8 返回 True，6 和 0 返回 False。"""
    return n > 0 and math.log2(n).is_integer()


def _mask_topk_ids_padded_region(
    topk_ids: torch.Tensor,
    num_token_non_padded: Optional[torch.Tensor] = None,
) -> None:
    """把 padding token 对应的专家 ID 原地设为 -1。

    输入输出示例：``topk_ids.shape=[4,2]``、``num_token_non_padded=3`` 时，前三行
    保持不变，最后一行变为 ``[-1,-1]``。
    """
    if num_token_non_padded is None:
        return
    # TODO：让 CUDA kernel 支持 int32 以外的 dtype。
    if _is_cuda and topk_ids.dtype == torch.int32:
        mask_topk_ids(topk_ids, num_token_non_padded)
    else:
        indices = torch.arange(0, topk_ids.shape[0], device=topk_ids.device)
        topk_ids[indices >= num_token_non_padded, :] = -1


def _zero_topk_weights_padded_region(
    topk_weights: torch.Tensor,
    num_token_non_padded: Optional[torch.Tensor] = None,
):
    """把 padding token 对应的 Top-K 权重原地清零。

    输入输出示例：``topk_weights.shape=[4,2]``、有效 token 数为 3 时，最后一行置 0。
    """
    if num_token_non_padded is None:
        return
    indices = torch.arange(0, topk_weights.shape[0], device=topk_weights.device)
    topk_weights[indices >= num_token_non_padded, :] = 0.0


@torch.compile(dynamic=True, backend=get_compiler_backend())
def _biased_grouped_topk_postprocess(
    topk_ids, expert_location_dispatch_info, num_token_non_padded
):
    """把逻辑专家 ID 映射为物理 ID，并屏蔽 padding token。

    输入输出示例：逻辑 ID ``[[0,2]]`` 经 EPLB 映射表可能变为 ``[[1,3]]``；若该行
    属于 padding 区域，则最终变为 ``[[-1,-1]]``。
    """
    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_ids


def biased_grouped_topk_gpu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    """按硬件与形状选择带 bias 的分组 Top-K GPU 实现。

    优先级包含 FlashInfer、SGLang fused gate、AITER、MUSA、Kimi K2/JIT 和 PyTorch
    回退。输入 ``gating_output=[M,E]``、Top-K=K，统一返回 ``[M,K]`` 的权重和 ID。
    """
    num_tokens = gating_output.shape[0]
    num_experts = gating_output.shape[1]
    experts_per_group = (
        num_experts // num_expert_group if num_expert_group else num_experts
    )

    # 这里只为路由专家计算 Top-K；共享专家会在后面单独追加。
    topk_routed = topk - num_fused_shared_experts
    if (
        _is_cuda
        and fused_topk_deepseek is not None
        and is_power_of_two(num_experts)
        # FlashInfer 的约束只应用于路由专家，不包含后续追加的共享专家。
        and topk_routed <= 8
        and topk_group <= num_expert_group
        and topk_group * num_expert_group >= topk_routed
        and (
            (experts_per_group <= 32 and experts_per_group * topk_group <= 128)
            if num_expert_group > 1
            else num_experts <= 384
        )
    ):
        # FlashInfer 会原地写结果，因此提前分配输出张量。
        topk_weights = torch.empty(
            (num_tokens, topk_routed), dtype=torch.float32, device=gating_output.device
        )
        topk_ids = torch.empty(
            (num_tokens, topk_routed), dtype=torch.int32, device=gating_output.device
        )

        # FlashInfer 总会在 kernel 内部应用 scaling_factor。
        scaling_factor = 1.0
        if routed_scaling_factor is not None and apply_routed_scaling_factor_on_output:
            scaling_factor = routed_scaling_factor

        # 调用 FlashInfer 的 DeepSeek 融合 Top-K kernel。
        fused_topk_deepseek(
            gating_output.to(dtype=torch.float32),
            correction_bias,
            num_expert_group,
            topk_group,
            topk_routed,
            scaling_factor,
            topk_weights,
            topk_ids,
            True,
        )

        if num_fused_shared_experts > 0:
            # 追加共享专家列：第一个共享专家 ID 为 num_experts；权重等于路由专家
            # 权重和除以 scaling_factor，与 biased_grouped_topk_impl 保持一致。
            # DeepEP 融合路径稍后会在 remap 阶段覆盖共享专家 ID 和权重。
            topk_ids = F.pad(topk_ids, (0, num_fused_shared_experts), value=num_experts)
            topk_weights = F.pad(topk_weights, (0, num_fused_shared_experts))
            if routed_scaling_factor is not None:
                topk_weights[:, topk_routed:] = (
                    topk_weights[:, :topk_routed].sum(dim=-1, keepdim=True)
                    / routed_scaling_factor
                )

        return topk_weights, topk_ids

    elif (
        _is_cuda
        # 当前 moe_fused_gate kernel 要求每组专家数不超过 MAX_VPT=32；
        # 后续 kernel 支持 MAX_VPT > 32 后可移除此限制。
        and experts_per_group <= 32
        and is_power_of_two(num_experts)
    ):
        topk_weights, topk_ids = moe_fused_gate(
            gating_output.to(dtype=torch.float32),
            correction_bias,
            num_expert_group,
            topk_group,
            topk,
            num_fused_shared_experts,
            routed_scaling_factor if routed_scaling_factor is not None else 1.0,
            apply_routed_scaling_factor_on_output,
        )

        return topk_weights, topk_ids

    elif _use_aiter:
        assert not apply_routed_scaling_factor_on_output, "Not implemented"
        token = gating_output.shape[0]
        device = gating_output.device
        assert (
            hidden_states.shape[0] == gating_output.shape[0]
        ), f"Number of tokens mismatch: hidden_states.shape[0] = {hidden_states.shape[0]}, gating_output.shape[0] = {gating_output.shape[0]}"
        topk_weights = torch.empty((token, topk), dtype=torch.float32, device=device)
        topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)
        aiter_biased_grouped_topk(
            gating_output,
            correction_bias.to(dtype=gating_output.dtype),
            topk_weights,
            topk_ids,
            num_expert_group,
            topk_group,
            renormalize,
            routed_scaling_factor if routed_scaling_factor is not None else 1.0,
        )
        return topk_weights, topk_ids
    elif _is_musa and (
        gating_output.shape[1] // num_expert_group <= 32
        or (num_expert_group == 1 and gating_output.shape[1] in {160, 256, 384})
    ):
        topk_weights, topk_ids = moe_fused_gate(
            gating_output.to(dtype=torch.float32),
            correction_bias,
            num_expert_group,
            topk_group,
            topk,
            num_fused_shared_experts,
            routed_scaling_factor if routed_scaling_factor is not None else 1.0,
            True,
            apply_routed_scaling_factor_on_output,
        )
    else:
        # Kimi K2（384 专家且只有一个专家组）使用专用优化路径。
        num_experts = gating_output.shape[1]
        if _is_cuda and num_experts == 384 and num_expert_group == 1:
            # ===== TO BE REFACTORED ====
            _use_jit_bf16_gate = False
            if _SGLANG_EXPERIMENTAL_LORA_OPTI:
                from sglang.srt.lora.trtllm_lora_temp.environ import lora_envs

                _use_jit_bf16_gate = (
                    lora_envs.SGLANG_OPT_USE_JIT_KERNEL_KIMI_GATE.get()
                    and lora_envs.SGLANG_OPT_KIMI_GATE_BF16_INPUT.get()
                )
            if _use_jit_bf16_gate:
                from sglang.jit_kernel.trtllm_lora_temp.kimi_k2_moe_fused_gate import (
                    kimi_k2_moe_fused_gate as _kimi_k2_moe_fused_gate,
                )

                # BF16 直通：避免启动两个由主机侧触发的 FP32 类型提升 kernel。
                return _kimi_k2_moe_fused_gate(
                    gating_output,
                    correction_bias,
                    topk=topk,
                    renormalize=renormalize,
                    routed_scaling_factor=routed_scaling_factor,
                    apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
                )
            # ===== 待重构区域结束 =====
            return kimi_k2_moe_fused_gate(
                gating_output.to(dtype=torch.float32),
                correction_bias,
                topk=topk,
                renormalize=renormalize,
                routed_scaling_factor=routed_scaling_factor,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            )
        elif (
            _is_cuda
            and num_expert_group == 1
            and topk_group == 1
            and num_fused_shared_experts == 0
            and num_experts <= 512
            and topk <= 8
        ):
            from sglang.jit_kernel.grouped_topk import grouped_topk as jit_grouped_topk

            scaling = (
                routed_scaling_factor if routed_scaling_factor is not None else 1.0
            )
            if not apply_routed_scaling_factor_on_output:
                scaling = 1.0
            return jit_grouped_topk(
                gating_output.to(dtype=torch.float32),
                correction_bias.to(dtype=torch.float32),
                num_expert_group,
                topk_group,
                topk,
                renormalize,
                scaling,
            )
        elif (
            _is_xpu
            and num_expert_group == 1
            and topk_group == 1
            and num_fused_shared_experts == 0
            and num_experts <= 256
            and topk <= 8
        ):
            if not apply_routed_scaling_factor_on_output:
                scaling = 1.0

            num_tokens = gating_output.shape[0]

            topk_values = torch.empty(
                (num_tokens, topk), dtype=torch.float32, device=gating_output.device
            )
            topk_indices = torch.empty(
                (num_tokens, topk), dtype=torch.int32, device=gating_output.device
            )

            if num_tokens == 0:
                return topk_values, topk_indices

            topk_sigmoid(
                topk_values,
                topk_indices,
                gating_output,
                renormalize,
                correction_bias,
            )
            return topk_values * scaling, topk_indices

        else:
            return biased_grouped_topk_impl(
                hidden_states,
                gating_output,
                correction_bias,
                topk,
                renormalize,
                num_expert_group,
                topk_group,
                num_fused_shared_experts=num_fused_shared_experts,
                routed_scaling_factor=routed_scaling_factor,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            )


def biased_grouped_topk_cpu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    compiled: bool = True,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    return torch.ops.sgl_kernel.biased_grouped_topk_cpu(
        hidden_states,
        gating_output,
        correction_bias,
        topk,
        renormalize,
        num_expert_group,
        topk_group,
        num_fused_shared_experts,
        routed_scaling_factor if apply_routed_scaling_factor_on_output else None,
        # CPU kernel 尚不支持 num_token_non_padded，因此必须传 None。
        num_token_non_padded=None,
    )


if _is_cpu and _is_cpu_amx_available:
    biased_grouped_topk = biased_grouped_topk_cpu
    grouped_topk = grouped_topk_cpu
    fused_topk_native = fused_topk_cpu
    fused_topk = fused_topk_cpu
else:
    biased_grouped_topk = biased_grouped_topk_gpu
    grouped_topk = grouped_topk_gpu
    fused_topk_native = fused_topk_torch_native


def _remap_topk_for_deepep(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_fused_shared_experts: int,
    num_physical_routed_experts: int,
    topk_config: TopKConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把 Top-K 输出重映射到 DeepEP 的交错专家布局。

    DeepEP dispatch 要求每个 rank 的共享专家拥有唯一 ID，才能把 token 路由到正确 rank。
    布局会在各 rank 的路由专家段后插入共享专家槽位：
    ``[routed_0..L-1, shared, routed_L..2L-1, shared, ...]``。

    映射规则：
        - 路由专家：``e -> e + e // num_local_routed``；
        - 共享专家：``ep_rank * num_local_experts + num_local_routed``；
        - 共享权重：``1 / routed_scaling_factor``，抵消 MoE 后续整体缩放。

    输入输出示例：EP=2、每 rank 2 个路由专家和 1 个共享专家时，原路由 ID
    ``[0,1,2,3]`` 映射为 ``[0,1,3,4]``，共享槽位 ID 分别为 rank 0 的 2 和 rank 1 的 5。
    """
    if topk_ids.shape[0] == 0:
        return topk_ids, topk_weights

    ep_size = get_moe_expert_parallel_world_size()
    ep_rank = get_moe_expert_parallel_rank()
    # 静态 EPLB 可能增加冗余物理专家。此处路由 topk_ids 已从逻辑 ID 映射为物理 ID，
    # 因此 DeepEP 交错布局必须使用物理路由专家数量。
    num_local_routed = num_physical_routed_experts // ep_size
    num_local_experts = num_local_routed + num_fused_shared_experts

    # 重映射路由专家 ID：为每个 rank 的共享专家槽位插入空隙。
    routed = topk_ids[:, :-num_fused_shared_experts]
    topk_ids[:, :-num_fused_shared_experts] = routed + routed // num_local_routed

    # 向量化设置共享专家 ID，使其路由回当前 EP rank。
    topk_ids[:, -num_fused_shared_experts:] = (
        ep_rank * num_local_experts
        + num_local_routed
        + torch.arange(num_fused_shared_experts, device=topk_ids.device)
    )

    # 覆盖共享专家权重为 1/routed_scaling_factor，使缩放后的净贡献为 1。
    routed_scaling_factor = topk_config.routed_scaling_factor
    if routed_scaling_factor is not None and routed_scaling_factor != 0:
        topk_weights[:, -num_fused_shared_experts:] = 1.0 / routed_scaling_factor

    return topk_ids, topk_weights


def _post_process_topk_ids(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_config: TopKConfig,
    router_logits: torch.Tensor,
    layer_id: int,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """统一处理 Top-K ID 的 EPLB、padding、共享专家和 DeepEP 映射。

    返回 ``(最终专家ID, 最终权重, 统计用专家ID)``。输入输出示例：输入均为 ``[M,K]``；
    padding 行会被屏蔽，EPLB 可把逻辑 ID 映射为物理 ID，DeepEP 可进一步插入共享槽位。
    """
    num_fused_shared_experts = topk_config.num_fused_shared_experts
    fused_shared_experts_scaling_factor = (
        topk_config.fused_shared_experts_scaling_factor
    )
    if (cap := get_global_experts_capturer()) is not None:
        cap.capture(
            layer_id=layer_id,
            topk_indices=topk_ids,
        )
    recorder_topk_ids = None
    if _is_cuda:
        # 融合共享专家时，它们作为 topk_ids 的额外列追加。EPLB 只能重映射路由专家列；
        # 共享专家列的值等于 n_routed_experts，若用于逻辑到物理映射表会越界。
        if num_fused_shared_experts > 0 and is_deepep_class_backend():
            shared_cols = topk_ids[:, -num_fused_shared_experts:]
            routed_cols = topk_ids[:, :-num_fused_shared_experts]
            routed_cols = _biased_grouped_topk_postprocess(
                routed_cols, expert_location_dispatch_info, num_token_non_padded
            )
            topk_ids = torch.cat([routed_cols, shared_cols], dim=-1)
            # ExpertDistributionRecorder 统计 EPLB 的物理路由专家。DeepEP 随后会向
            # topk_ids 插入每 rank 的共享专家槽位，因此单独保留路由物理 ID 供统计使用。
            recorder_topk_ids = routed_cols
        else:
            topk_ids = _biased_grouped_topk_postprocess(
                topk_ids, expert_location_dispatch_info, num_token_non_padded
            )
    elif _is_hip:
        topk_ids = _biased_grouped_topk_postprocess(
            topk_ids, expert_location_dispatch_info, num_token_non_padded
        )
        # AMD HIP 的 AITER MoE kernel 不能安全处理 topk_ids=-1，负索引会导致非法访存。
        # 因此改为把 padding token 的路由权重清零，使其专家输出在加权求和后贡献为 0。
        _zero_topk_weights_padded_region(topk_weights, num_token_non_padded)

    if recorder_topk_ids is None:
        recorder_topk_ids = topk_ids

    if num_fused_shared_experts > 0 and _use_aiter:
        M, N = router_logits.shape
        scale_factor = (
            1.0
            if fused_shared_experts_scaling_factor is None
            else fused_shared_experts_scaling_factor
        )

        # 延迟导入，避免模块间循环依赖。
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
            fused_append_shared_experts,
        )

        topk_ids, topk_weights = fused_append_shared_experts(
            topk_ids,
            topk_weights,
            num_fused_shared_experts,
            scale_factor,
            N,  # 共享专家 ID 的起始值。
        )

    # DeepEP：重映射到交错专家布局，使每个 rank 的共享专家具有唯一 dispatch ID。
    if num_fused_shared_experts > 0 and is_deepep_class_backend():
        num_physical_routed_experts = (
            expert_location_dispatch_info.num_physical_experts
            if expert_location_dispatch_info is not None
            else router_logits.shape[1]
        )
        topk_ids, topk_weights = _remap_topk_for_deepep(
            topk_ids,
            topk_weights,
            num_fused_shared_experts,
            num_physical_routed_experts,
            topk_config,
        )

    if _is_hip:
        # 追加/重映射共享专家可能在前面的 HIP padding mask 之后重新引入非零权重；
        # 因此函数返回前再次确保 padding token 的所有专家权重均为 0。
        _zero_topk_weights_padded_region(topk_weights, num_token_non_padded)

    return topk_ids, topk_weights, recorder_topk_ids


def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk_config: TopKConfig,
    *,
    layer_id: Optional[int] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
) -> StandardTopKOutput:
    """统一的专家选择入口，根据配置和平台选择具体 Top-K 实现并完成后处理。

    主要阶段：输入逻辑/物理专家变换 → grouped/biased/fused/custom Top-K → 可选基准测试
    路由覆盖 → padding/EPLB/DeepEP 后处理 → 记录专家分布。

    输入输出示例：``hidden_states.shape=[4,d]``、``router_logits.shape=[4,8]``、
    ``top_k=2`` 时，通常返回 ``StandardTopKOutput``：权重和 ID 均为 ``[4,2]``，
    router_logits 保持 ``[4,8]``。实验性融合路径可能返回带 packed ID 的兼容子类型。
    """
    top_k = topk_config.top_k
    use_grouped_topk = topk_config.use_grouped_topk
    topk_group = topk_config.topk_group
    num_expert_group = topk_config.num_expert_group
    renormalize = topk_config.renormalize
    num_fused_shared_experts = topk_config.num_fused_shared_experts
    custom_routing_function = topk_config.custom_routing_function
    correction_bias = topk_config.correction_bias
    torch_native = topk_config.torch_native
    routed_scaling_factor = topk_config.routed_scaling_factor
    apply_routed_scaling_factor_on_output = (
        topk_config.apply_routed_scaling_factor_on_output
    )

    scoring_func = topk_config.scoring_func

    # 仅在后面的 gating+pack 融合分支中设置，其他路径保持 None。
    packed_topk = None

    (
        router_logits,
        correction_bias,
    ) = expert_location_dispatch.transform_select_experts_inputs(
        router_logits=router_logits,
        correction_bias=correction_bias,
        info=expert_location_dispatch_info,
    )

    # DeepSeek V2/V3/R1 系列模型使用 grouped Top-K。
    # 路由 Top-K 数量需要扣除稍后单独追加的融合共享专家。
    num_routed_topk = top_k - num_fused_shared_experts
    if use_grouped_topk:
        assert topk_group is not None
        assert num_expert_group is not None
        if correction_bias is None:
            topk_weights, topk_ids = grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=num_routed_topk if _use_aiter else top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                num_fused_shared_experts=num_fused_shared_experts,
                routed_scaling_factor=routed_scaling_factor,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            )
        else:
            topk_weights, topk_ids = biased_grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                correction_bias=correction_bias,
                topk=num_routed_topk if _use_aiter else top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                num_fused_shared_experts=num_fused_shared_experts,
                routed_scaling_factor=routed_scaling_factor,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            )
    elif torch_native and custom_routing_function is None:
        assert (
            num_token_non_padded is None
        ), "num_token_non_padded is not yet supported in fused_topk_native"
        assert expert_location_dispatch_info is None
        assert not apply_routed_scaling_factor_on_output, "Not implemented"
        topk_weights, topk_ids = fused_topk_native(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=num_routed_topk if _use_aiter else top_k,
            renormalize=renormalize,
            correction_bias=correction_bias,
            scoring_func=scoring_func,
        )
    elif custom_routing_function is None:
        if scoring_func != "sqrtsoftplus":
            assert not apply_routed_scaling_factor_on_output, "Not implemented"

        if scoring_func == "sqrtsoftplus":
            _biased_topk = (
                biased_topk_jit_kernel_impl
                if envs.SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK.get()
                else biased_topk_impl
            )

            topk_weights, topk_ids = _biased_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                correction_bias=correction_bias,
                topk=num_routed_topk if _use_aiter else top_k,
                renormalize=renormalize,
                scoring_func=scoring_func,
                num_fused_shared_experts=num_fused_shared_experts,
                routed_scaling_factor=routed_scaling_factor,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            )
        elif (
            get_moe_runner_backend().is_flashinfer_trtllm_routed()
            and scoring_func == "softmax"
            and correction_bias is None
        ):
            # flashinfer_trtllm_routed 直接在原始 logits 上执行 Top-K。
            topk_weights, topk_ids = fused_topk_softmax_torch_raw_logits(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=num_routed_topk if _use_aiter else top_k,
                renormalize=renormalize,
            )
        else:
            # gating+routed pack 融合仅用于普通 CUDA softmax 路径，并要求没有 EPLB
            # 重映射、共享专家、correction bias 或自定义路由覆盖。
            _fused_topk_pack = False
            if _SGLANG_EXPERIMENTAL_LORA_OPTI:
                from sglang.srt.lora.trtllm_lora_temp.environ import lora_envs

                _fused_topk_pack = lora_envs.SGLANG_OPT_LORA_FUSED_TOPK_PACK.get()
            if (
                _fused_topk_pack
                and _is_cuda
                and not _use_aiter
                and scoring_func == "softmax"
                and correction_bias is None
                and expert_location_dispatch_info is None
                and num_fused_shared_experts == 0
                and not envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.get()
                and not envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.get()
            ):
                num_experts = router_logits.shape[-1]
                if num_experts & (num_experts - 1) == 0 and num_experts <= 512:
                    packed_topk = torch.empty(
                        (hidden_states.shape[0], top_k),
                        dtype=torch.int32,
                        device=hidden_states.device,
                    )

            # Qwen3-MoE 使用 fused_topk。
            _fused_topk_kwargs = {}
            # ===== TO BE REFACTORED ====
            # 只有实验性的 Top-K+pack 融合路径传 packed_out/num_token_non_padded；
            # 默认调用保持上游函数签名，因为 fused_topk_cpu 没有这些参数。
            if packed_topk is not None:
                _fused_topk_kwargs = dict(
                    packed_out=packed_topk,
                    num_token_non_padded=num_token_non_padded,
                )
            # ===== 待重构区域结束 =====
            topk_weights, topk_ids = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=num_routed_topk if _use_aiter else top_k,
                renormalize=renormalize,
                correction_bias=correction_bias,
                scoring_func=scoring_func,
                **_fused_topk_kwargs,
            )
    else:
        assert (
            num_token_non_padded is None
        ), "num_token_non_padded is not yet supported in custom_routing_function"
        assert expert_location_dispatch_info is None
        assert not apply_routed_scaling_factor_on_output, "Not implemented"
        topk_weights, topk_ids = custom_routing_function(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=num_routed_topk if _use_aiter else top_k,
            renormalize=renormalize,
        )

    simulate_uniform_experts = envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.get()
    simulate_round_robin_experts = envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.get()
    if simulate_uniform_experts and simulate_round_robin_experts:
        raise ValueError(
            "SGLANG_SIMULATE_UNIFORM_EXPERTS and "
            "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS are mutually exclusive"
        )

    if simulate_uniform_experts:
        # 仅用于基准测试：用带随机起点的均匀专家分配覆盖真实路由，避免虚假/随机权重
        # 导致专家负载倾斜。禁止用于生产环境。
        num_tokens, k = topk_ids.shape
        num_experts = router_logits.shape[1]
        if k > 0:
            offsets = torch.randint(
                0, num_experts, (num_tokens, 1), device=topk_ids.device
            )
            steps = torch.arange(k, device=topk_ids.device).unsqueeze(0)
            step = max(num_experts // k, 1)
            topk_ids = ((offsets + steps * step) % num_experts).to(topk_ids.dtype)
            topk_weights = torch.ones_like(topk_weights) / k
    elif simulate_round_robin_experts:
        # 仅用于基准测试：用确定性的轮询专家分配覆盖真实路由，避免虚假/随机权重
        # 带来的路由噪声。禁止用于生产环境。
        num_tokens, k = topk_ids.shape
        num_experts = router_logits.shape[1]
        topk_ids = _make_round_robin_expert_ids(
            num_tokens,
            k,
            num_experts,
            device=topk_ids.device,
            dtype=topk_ids.dtype,
            layer_id=layer_id,
        )
        if k > 0:
            topk_weights = torch.full_like(topk_weights, 1.0 / k)

    topk_ids, topk_weights, recorder_topk_ids = _post_process_topk_ids(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        topk_config=topk_config,
        router_logits=router_logits,
        num_token_non_padded=num_token_non_padded,
        layer_id=layer_id,
        expert_location_dispatch_info=expert_location_dispatch_info,
    )

    get_global_expert_distribution_recorder().on_select_experts(
        topk_ids=recorder_topk_ids
    )

    # ===== TO BE REFACTORED ====
    if packed_topk is not None:
        return StandardTopKOutputPacked(
            topk_weights, topk_ids, router_logits, packed_topk
        )
    # ===== END TO BE REFACTORED ====
    return StandardTopKOutput(topk_weights, topk_ids, router_logits)


# 注册 fake 实现，使 torch.compile 能在不执行真实 kernel 的情况下推导输出形状。
if _is_cuda:

    @torch.library.register_fake("sgl_kernel::moe_fused_gate")
    def _moe_fused_gate(
        input_tensor,
        bias,
        num_expert_group,
        topk_group,
        topk,
        num_fused_shared_experts=0,
        routed_scaling_factor=0,
        apply_routed_scaling_factor_on_output=False,
    ):
        num_rows = input_tensor.shape[0]
        topk_weights = torch.empty(
            (num_rows, topk), dtype=torch.float32, device=input_tensor.device
        )
        topk_ids = torch.empty(
            (num_rows, topk), dtype=torch.int32, device=input_tensor.device
        )
        return topk_weights, topk_ids

    @register_fake_if_exists("sgl_kernel::kimi_k2_moe_fused_gate")
    def _kimi_k2_moe_fused_gate(
        input_tensor,
        bias,
        topk,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    ):
        num_rows = input_tensor.shape[0]
        topk_weights = input_tensor.new_empty(
            num_rows,
            topk,
            dtype=torch.float32,
        )
        topk_ids = input_tensor.new_empty(
            num_rows,
            topk,
            dtype=torch.int32,
        )
        return topk_weights, topk_ids
