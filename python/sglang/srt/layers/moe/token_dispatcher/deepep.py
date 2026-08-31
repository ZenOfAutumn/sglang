# ============================================================================
# DeepEP Token Dispatcher（DeepEP 令牌分发器）
# ----------------------------------------------------------------------------
# 本模块封装 DeepEP（DeepSeek Expert Parallelism 通信库）在 MoE 专家并行中的
# 「dispatch（分发）」与「combine（聚合）」两个核心 all-to-all 通信步骤：
#   - dispatch：按 top-k 路由结果，把每个 token 的 hidden_states 发送到其所选
#     专家所在的 rank（GPU/NPU）上；
#   - combine：专家计算完成后，把结果按原路收回并按 top-k 权重加权聚合，还原到
#     每个 token 原本所在的 rank。
#
# 提供两种运行模式：
#   - Normal（普通模式）：面向 prefill（长序列、大 batch），吞吐优先；
#   - Low-Latency（低时延模式，简称 LL）：面向 decode（逐 token 生成），时延优先，
#     使用 RDMA 直达并支持计算/通信重叠（overlap）。
#
# 每个通信步骤都拆成 a/b 两个子阶段（如 dispatch_a / dispatch_b），以便把
# 「发起通信」与「等待通信完成」分离，从而与其它计算重叠，隐藏通信开销。
# ============================================================================
from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, NamedTuple, Optional, Tuple, Union

from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.dp_attention import get_is_extend_in_batch
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    BaseDispatcherConfig,
    CombineInput,
    CombineInputFormat,
    DispatcherBaseHooks,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import (
    DeepEPMode,
    DispatcherOutputDtype,
    get_deepep_config,
    get_deepep_output_dtype,
    is_tbo_enabled,
)
from sglang.srt.utils import (
    get_bool_env_var,
    get_cuda_version,
    is_blackwell,
    is_flashinfer_available,
    is_hip,
    is_npu,
    load_json_config,
)

_is_npu = is_npu()
_use_zbal = _is_npu and envs.SGLANG_ZBAL_LOCAL_MEM_SIZE.get() > 0

if TYPE_CHECKING:
    from sglang.srt.batch_overlap.single_batch_overlap import CombineOverlapArgs

try:
    if _use_zbal:
        from zbal.zbal.deepep_adaptor import Config
        from zbal.zbal_buffer import Buffer
    else:
        from deep_ep import Buffer, Config

    if not _is_npu:
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

    use_deepep = True
except ImportError:
    use_deepep = False

from enum import Enum, IntEnum, auto

import torch
import torch.distributed as dist

from sglang.srt.runtime_context import get_resources

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()

logger = logging.getLogger(__name__)

_NVSHMEM_QP_DEPTH_DEFAULT = 1024


def _set_nvshmem_qp_depth(num_max_dispatch_tokens_per_rank: int) -> None:
    min_qp_depth = 2 * (num_max_dispatch_tokens_per_rank + 1)
    current_qp_depth = int(
        os.environ.get("NVSHMEM_QP_DEPTH", _NVSHMEM_QP_DEPTH_DEFAULT)
    )
    os.environ["NVSHMEM_QP_DEPTH"] = str(
        max(current_qp_depth, _NVSHMEM_QP_DEPTH_DEFAULT, min_qp_depth)
    )


def _is_mnnvl_fabric_supported() -> bool:
    if not is_flashinfer_available():
        return False

    from flashinfer.comm.mnnvl import is_mnnvl_fabric_supported

    return is_mnnvl_fabric_supported(torch.cuda.current_device())


def _deepep_precompile_tp_barrier() -> None:
    # DeepEP 的 all-to-all 操作相比 torch.distributed 的超时时间要短得多，
    # 因此如果不同 rank 的编译速度不一致，很容易触发超时。
    # 为避免这一问题，我们在编译阶段改用 torch.distributed 的 barrier 做同步。
    # 该 barrier 只在编译阶段生效，以免在运行时引入额外的 all-reduce 开销。
    if envs.SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE.get():
        get_tp_group().barrier()


class DeepEPPDispatchHooks(DispatcherBaseHooks):
    """DeepEP 分发 hook 容器。

    在 dispatch_a 与 dispatch_b 之间被调用，用于依次执行所有已注册的 hook 函数，
    常用于插入观测、调试或与其它模块的协同逻辑。
    """

    def __call__(self, dispatcher: BaseDispatcher):
        """依次调用所有已注册的 hook，参数为当前 dispatcher 实例。"""
        for hook_fun in self.hook_dict.values():
            hook_fun(dispatcher)


class DeepEPNormalDispatchOutput(NamedTuple):
    """DeepEP 普通模式（Normal）分发输出。

    字段说明：
        hidden_states: 本 rank 收到的、发往本地专家的所有 token 特征，
            形状约为 [num_recv_tokens, hidden_size]。
        hidden_states_scale: 量化缩放因子（仅 fp8/nvfp4 等量化路径存在，否则为 None）。
        topk_ids: 收到的 token 对应的 top-k 专家 id，形状 [num_recv_tokens, topk]。
        topk_weights: 收到的 token 对应的 top-k 路由权重，形状 [num_recv_tokens, topk]。
        num_recv_tokens_per_expert: 每个本地专家收到的 token 数量列表，
            如 [12, 8, 0, 30] 表示 4 个本地专家各自收到的 token 数。
    """

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    num_recv_tokens_per_expert: List[int]

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_NORMAL


class DeepEPLLDispatchOutput(NamedTuple):
    """DeepEP 低时延模式（Low-Latency）分发输出。

    低时延模式为每个专家预留固定容量的缓冲区，token 按专家分块摆放，
    并用 masked_m 记录每个专家实际有效的 token 数。

    字段说明：
        hidden_states: 收到的 token 特征，按专家分块，
            形状约为 [num_local_experts, max_tokens_per_expert, hidden_size]。
        hidden_states_scale: 量化缩放因子（无量化时为 None）。
        topk_ids: top-k 专家 id，形状 [num_tokens, topk]。
        topk_weights: top-k 路由权重，形状 [num_tokens, topk]。
        masked_m: 每个本地专家实际有效 token 数量的掩码计数，形状 [num_local_experts]。
        expected_m: 每个专家「期望」处理的平均 token 数（用于 GEMM 分组的对齐估计）。
    """

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    masked_m: torch.Tensor
    expected_m: int

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_LL


assert isinstance(DeepEPNormalDispatchOutput, DispatchOutput)
assert isinstance(DeepEPLLDispatchOutput, DispatchOutput)


class DeepEPNormalCombineInput(NamedTuple):
    """DeepEP 普通模式（Normal）聚合输入。

    字段说明：
        hidden_states: 专家计算后的 token 特征（待收回并加权聚合）。
        topk_ids: 每个 token 的 top-k 专家 id。
        topk_weights: 每个 token 的 top-k 路由权重（用于聚合时加权求和）。
    """

    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.DEEPEP_NORMAL


class DeepEPLLCombineInput(NamedTuple):
    """DeepEP 低时延模式（Low-Latency）聚合输入。

    字段说明：
        hidden_states: 专家计算后的 token 特征（按专家分块）。
        topk_ids: 每个 token 的 top-k 专家 id。
        topk_weights: 每个 token 的 top-k 路由权重。
    """

    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.DEEPEP_LL


assert isinstance(DeepEPNormalCombineInput, CombineInput)
assert isinstance(DeepEPLLCombineInput, CombineInput)


class DeepEPDispatchMode(IntEnum):
    """DeepEP 分发模式枚举。

    NORMAL: 普通模式，吞吐优先，用于 prefill。
    LOW_LATENCY: 低时延模式，时延优先，用于 decode。
    """

    NORMAL = auto()
    LOW_LATENCY = auto()


class DeepEPBuffer:
    """Managing facade for the process-wide DeepEP comm buffer; the state
    itself lives on ``ctx.resources`` (one entry per process)."""

    @classmethod
    def _state(cls):
        from types import SimpleNamespace

        buffers = get_resources().buffers
        state = buffers.get("deepep_ep_state")
        if state is None:
            state = SimpleNamespace(
                buffer=None,
                dispatch_mode=None,
                hidden_size=None,
                num_max_dispatch_tokens_per_rank=None,
                num_experts=None,
            )
            buffers["deepep_ep_state"] = state
        return state

    @classmethod
    def get_deepep_buffer(
        cls,
        group: dist.ProcessGroup,
        hidden_size: int,
        param_bytes: int,
        deepep_mode: DeepEPMode,
        num_max_dispatch_tokens_per_rank: int = -1,
        num_experts: int = -1,
    ):
        state = cls._state()
        if state.buffer is not None:
            return state.buffer

        state.hidden_size = hidden_size
        state.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        state.num_experts = num_experts

        num_nvl_bytes, num_rdma_bytes = 0, 0
        if deepep_mode.enable_normal():
            hidden_bytes = hidden_size * param_bytes
            for config in (
                DeepEPConfig.get_instance().normal_dispatch_config
                or Buffer.get_dispatch_config(group.size()),
                DeepEPConfig.get_instance().normal_combine_config
                or Buffer.get_combine_config(group.size()),
            ):
                num_nvl_bytes = max(
                    config.get_nvl_buffer_size_hint(hidden_bytes, group.size()),
                    num_nvl_bytes,
                )
                num_rdma_bytes = max(
                    config.get_rdma_buffer_size_hint(hidden_bytes, group.size()),
                    num_rdma_bytes,
                )
        if deepep_mode.enable_low_latency():
            assert num_max_dispatch_tokens_per_rank != -1
            assert num_experts != -1 and num_experts % group.size() == 0
            if not _is_npu:
                _set_nvshmem_qp_depth(num_max_dispatch_tokens_per_rank)
            num_rdma_bytes = max(
                Buffer.get_low_latency_rdma_size_hint(
                    num_max_dispatch_tokens_per_rank,
                    hidden_size,
                    group.size(),
                    num_experts,
                ),
                num_rdma_bytes,
            )

        # 我们应与 DeepEP 测试脚本的逻辑保持一致来计算 num_qps_per_rank（每 rank 的 QP 数）：
        if deepep_mode == DeepEPMode.NORMAL:
            # 参考：https://github.com/deepseek-ai/DeepEP/blob/main/tests/test_internode.py#L235
            num_qps_per_rank = DeepEPConfig.get_instance().num_sms
        elif deepep_mode == DeepEPMode.LOW_LATENCY:
            # 参考：https://github.com/deepseek-ai/DeepEP/blob/main/tests/test_low_latency.py#L176
            num_qps_per_rank = num_experts // group.size()
        elif deepep_mode == DeepEPMode.AUTO:
            # AUTO 模式下低时延与普通模式都需运行，因此取两者所需 QP 数的最大值
            # 参考：https://github.com/deepseek-ai/DeepEP/blob/main/tests/test_internode.py#L235
            num_qps_per_rank = max(
                DeepEPConfig.get_instance().num_sms, num_experts // group.size()
            )
        else:
            raise NotImplementedError

        if not _is_npu:
            total_num_sms = torch.cuda.get_device_properties(
                device="cuda"
            ).multi_processor_count
            if (
                (deepep_mode != DeepEPMode.LOW_LATENCY)
                and not is_tbo_enabled()
                and (DeepEPConfig.get_instance().num_sms < total_num_sms // 2)
            ):
                logger.warning(
                    f"Only use {DeepEPConfig.get_instance().num_sms} SMs for DeepEP communication. "
                    f"This may result in highly suboptimal performance. "
                    f"Consider using --deepep-config to change the behavior."
                )

        use_mnnvl_fabric = _is_mnnvl_fabric_supported()
        buffer_kwargs = dict(
            low_latency_mode=deepep_mode.enable_low_latency(),
            num_qps_per_rank=num_qps_per_rank,
            allow_mnnvl=use_mnnvl_fabric,
        )
        # 在声明支持 MNNVL fabric 的硬件上使用 CU_MEM_HANDLE_TYPE_FABRIC，
        # 使跨 pod 的 GB200/GB300 EP 组使用 cuMemImportFromShareableHandle，
        # 而不是仅限机内的 cudaIpcOpenMemHandle。
        # 我们发布的 DeepEP 构建按 CUDA 主版本区分：
        #   cu13x -> hybrid-ep，它用 use_fabric 参数控制 fabric，因此当设备声明支持
        #            fabric 时我们传入该参数。
        #   cu12x -> fzyzcjy/DeepEP，无 use_fabric 参数，但在 C++ 层已在支持时自动
        #            启用 fabric，因此我们跳过：
        #            https://github.com/fzyzcjy/DeepEP/blob/814e508537c6ffc775d59f6f1b9ba43f3a65968c/csrc/deep_ep.cpp#L52
        is_cu12 = get_cuda_version()[0] == 12
        if not is_cu12 and use_mnnvl_fabric:
            buffer_kwargs["use_fabric"] = True

        state.buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes, **buffer_kwargs)
        return state.buffer

    @classmethod
    def clean_buffer(cls):
        state = cls._state()
        if not state.buffer.low_latency_mode:
            return
        state.buffer.clean_low_latency_buffer(
            state.num_max_dispatch_tokens_per_rank,
            state.hidden_size,
            state.num_experts,
        )

    @classmethod
    def set_dispatch_mode_as_normal(cls):
        cls._state().dispatch_mode = DeepEPDispatchMode.NORMAL

    @classmethod
    def set_dispatch_mode_as_low_latency(cls):
        state = cls._state()
        if state.dispatch_mode == DeepEPDispatchMode.NORMAL:
            cls.clean_buffer()
        state.dispatch_mode = DeepEPDispatchMode.LOW_LATENCY

    @classmethod
    def set_dispatch_mode(cls, mode: DeepEPMode):
        """根据传入的 DeepEPMode 设置当前分发模式。

        参数：
            mode: DeepEPMode，需为低时延或普通模式，否则抛出异常。
        """
        if mode.is_low_latency():
            cls.set_dispatch_mode_as_low_latency()
        elif mode.is_normal():
            cls.set_dispatch_mode_as_normal()
        else:
            raise Exception("unsupported mode")


class DeepEPConfig(BaseDispatcherConfig):
    """DeepEP 分发器配置（单例）。

    从环境/命令行给出的 JSON 配置中解析普通模式下 dispatch / combine 的
    通信配置（如 SM 数量）。若未提供配置，则使用 DeepEP 默认值。

    属性：
        normal_dispatch_config: 普通模式 dispatch 配置（None 表示用默认）。
        normal_combine_config: 普通模式 combine 配置（None 表示用默认）。
        num_sms: 用于 DeepEP 通信的 SM（流多处理器）数量。
    """

    _instance = None

    def __init__(self):
        config_str = get_deepep_config()
        if config_str:
            config_parsed = load_json_config(config_str)
            if torch.distributed.get_rank() == 0:
                logger.info(f"Use DeepEP Config: {config_parsed}")
            config_dispatch = config_parsed["normal_dispatch"]
            config_combine = config_parsed["normal_combine"]

            self.normal_dispatch_config = Config(**config_dispatch)
            self.normal_combine_config = Config(**config_combine)

            assert config_dispatch["num_sms"] == config_combine["num_sms"]
            self.num_sms = config_dispatch["num_sms"]
        else:
            self.normal_dispatch_config = None
            self.normal_combine_config = None
            self.num_sms = Buffer.num_sms

    @classmethod
    def get_instance(cls):
        """获取全局单例配置（首次调用时创建）。"""
        if cls._instance is None:
            cls._instance = DeepEPConfig()
        return cls._instance


class _DeepEPDispatcherImplBase:
    """DeepEP 分发器实现的抽象基类。

    定义了 dispatch/combine 的 a/b 子阶段接口，以及量化 dtype、重叠参数
    等公共状态。普通模式与低时延模式分别由子类实现。

    主要属性：
        group: 专家并行进程组。
        router_topk: 每个 token 选择的专家个数（top-k）。
        num_experts / num_local_experts: 专家总数 / 本 rank 上的专家数。
        hidden_size: 隐藏层维度。
        deepep_mode: 运行模式。
        num_max_dispatch_tokens_per_rank: 每 rank 最大分发 token 数（低时延相关）。
        handle: DeepEP 内部句柄，在 dispatch 中生成并在 combine 中回传（用于路由还原）。
    """

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
        deepep_mode: DeepEPMode,
    ):
        if not use_deepep:
            raise ImportError(
                "DeepEP is not installed. Please install DeepEP package from "
                "https://github.com/deepseek-ai/deepep."
            )

        self.group = group
        self.router_topk = router_topk
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.deepep_mode = deepep_mode

        self.params_bytes = 2
        # 过大的取值会导致内存占用激增，因此用户应根据实际情况相应调整
        self.num_max_dispatch_tokens_per_rank = (
            envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        )
        # DeepEP 的 internode_ll dispatch 使用 FINISHED_SUM_TAG=1024，
        # 其逻辑要求「从一个 rank 发往另一个 rank 的 token 数」小于该值
        assert self.num_max_dispatch_tokens_per_rank <= 1024

        self.handle = None

        self.quant_config: Optional[dict] = None

        self.overlap_args: Optional[CombineOverlapArgs] = None
        self.meta_overlap_args: Optional[dict] = None

        self.set_deepep_dispatcher_dtype()

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        """dispatch 第一阶段：预处理并发起分发通信（子类实现）。"""
        raise NotImplementedError

    def dispatch_b(self, *args, **kwargs):
        """dispatch 第二阶段：等待通信完成并整理输出（子类实现）。"""
        raise NotImplementedError

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        """combine 第一阶段：发起聚合通信（子类实现）。"""
        raise NotImplementedError

    def combine_b(self, *args, **kwargs):
        """combine 第二阶段：等待聚合通信完成并返回结果（子类实现）。"""
        raise NotImplementedError

    def _get_buffer(self):
        """获取当前模式对应的 DeepEP 缓冲区（子类实现）。"""
        raise NotImplementedError

    def set_quant_config(self, quant_config: dict) -> None:
        """设置量化配置，并据此重新推导分发输出的 dtype。"""
        self.quant_config = quant_config
        self.set_deepep_dispatcher_dtype()

    def set_deepep_dispatcher_dtype(self) -> None:
        """根据量化配置与硬件确定分发输出 dtype，并设置 use_fp8/use_nvfp4 标志。"""
        self.deepep_output_dtype = get_deepep_output_dtype(self)

        # 每种 dtype 对应的配置映射
        config_map = {
            DispatcherOutputDtype.BF16: {
                "use_fp8": False,
                "use_nvfp4": False,
            },
            DispatcherOutputDtype.FP8: {
                "use_fp8": True,
                "use_nvfp4": False,
            },
            # Needed for Ascend A2/A3 NPU case,
            # despite the use_fp8 flag,
            # quantization will be performed in int8
            DispatcherOutputDtype.INT8: {
                "use_fp8": True,
                "use_nvfp4": False,
            },
            DispatcherOutputDtype.NVFP4: {
                "use_fp8": False,
                "use_nvfp4": True,
            },
        }

        # Validate and apply hardware-specific adjustments
        self._validate_and_adjust_dtype()

        # Apply configuration
        config = config_map[self.deepep_output_dtype]
        self.use_fp8 = config["use_fp8"]
        self.use_nvfp4 = config["use_nvfp4"]

        # Handle environment variables
        if _is_npu:
            self._update_int8_quant_env()

    def _validate_and_adjust_dtype(self) -> None:
        """校验分发输出 dtype 与硬件是否兼容，必要时自动调整。

        - 昇腾 NPU：fp8 回退为 int8；nvfp4 不支持（报错）。
        - GPU：int8 回退为 fp8；nvfp4 受支持无需调整。
        """
        if _is_npu:
            if self.deepep_output_dtype == DispatcherOutputDtype.FP8:
                logger.warning_once(
                    "Ascend A2/A3 NPU does not support fp8 "
                    "deepep_dispatcher_output_dtype, switching to int8..."
                )
                self.deepep_output_dtype = DispatcherOutputDtype.INT8
            elif self.deepep_output_dtype == DispatcherOutputDtype.NVFP4:
                raise RuntimeError(
                    "Ascend A2/A3 NPU does not support nvfp4 deepep_dispatcher_output_dtype."
                )
        else:
            if self.deepep_output_dtype == DispatcherOutputDtype.INT8:
                logger.warning_once(
                    "GPU does not support int8 "
                    "deepep_dispatcher_output_dtype, switching to fp8..."
                )
                self.deepep_output_dtype = DispatcherOutputDtype.FP8
            # NVFP4 is supported on GPU, no adjustment needed

    def _update_int8_quant_env(self) -> None:
        """TODO 在 NPU 上为 base 模型与 draft 模型适配不同的量化方案"""
        pass

    def set_overlap_args(
        self, combine_overlap_args: CombineOverlapArgs, meta_overlap_args: dict
    ) -> None:
        """设置计算/通信重叠（overlap）参数，供低时延 combine 使用。"""
        self.overlap_args = combine_overlap_args
        self.meta_overlap_args = meta_overlap_args

    def clear_overlap_args(self) -> None:
        """清除重叠参数。"""
        self.overlap_args = None
        self.meta_overlap_args = None


class _DeepEPDispatcherImplNormal(_DeepEPDispatcherImplBase):
    dispatch_mode = DeepEPMode.NORMAL

    def __init__(self, async_finish: bool, **kwargs):
        super().__init__(**kwargs)

        self.async_finish = async_finish
        self.src2dst = None
        self.quant_config = {}

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        """dispatch 第一阶段：（可选）fp8 量化并捕获 CUDA event，为分发做准备。

        参数：
            hidden_states: 本 rank 的 token 特征，形状 [num_tokens, hidden_size]。
            topk_output: 路由输出，包含 topk_ids 与 topk_weights。

        返回：
            (hidden_states, topk_ids, topk_weights, previous_event) 元组，
            作为 dispatch_b 的输入。previous_event 在异步模式下为捕获的 CUDA event，
            否则为 None。

        示例：
            >>> # hidden_states: [4, 7168]，topk=8
            >>> hs, ids, w, ev = impl.dispatch_a(hidden_states, topk_output)
            >>> ids.dtype
            torch.int64
        """
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
        topk_ids = topk_ids.to(torch.int64)
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and self.use_fp8:
            # TODO 硬编码 128 分块量化，使用 fp8 通信
            hidden_states = sglang_per_token_group_quant_fp8(
                hidden_states,
                128,
                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            )
        previous_event = Buffer.capture() if self.async_finish else None
        return hidden_states, topk_ids, topk_weights, previous_event

    def dispatch_b(self, hidden_states, topk_ids, topk_weights, previous_event):
        """dispatch 第二阶段：执行实际 all-to-all 分发，等待完成并打包输出。

        参数：
            hidden_states, topk_ids, topk_weights, previous_event: 均来自 dispatch_a 的输出。

        返回：
            DeepEPNormalDispatchOutput，包含本 rank 收到的 token 特征、（可选）量化缩放因子、
            top-k id/权重，以及每个本地专家收到的 token 数量。

        示例：
            >>> out = impl.dispatch_b(hs, ids, w, ev)
            >>> out.format
            <DispatchOutputFormat.DEEPEP_NORMAL: ...>
            >>> out.num_recv_tokens_per_expert  # 如 4 个本地专家各自收到的 token 数
            [12, 8, 0, 30]
        """
        (
            hidden_states,
            topk_ids,
            topk_weights,
            num_recv_tokens_per_expert,
            event,
        ) = self._dispatch_core(hidden_states, topk_ids, topk_weights, previous_event)
        event.current_stream_wait() if self.async_finish else ()

        if isinstance(hidden_states, tuple):
            hidden_states, hidden_states_scale = hidden_states
        else:
            hidden_states_scale = None

        return DeepEPNormalDispatchOutput(
            hidden_states,
            hidden_states_scale,
            topk_ids,
            topk_weights,
            num_recv_tokens_per_expert,
        )

    def _dispatch_core(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        previous_event,
    ):
        """普通模式 dispatch 的核心实现：计算布局 -> 执行 all-to-all -> 记录专家分布。

        参数：
            x: token 特征（或 (特征, 量化缩放因子) 元组）。
            topk_ids / topk_weights: top-k 专家 id 与路由权重。
            previous_event: 前置 CUDA event（异步模式下用于依赖排序）。

        返回：
            (recv_x, recv_topk_ids, recv_topk_weights, num_recv_tokens_per_expert, event)。
        """
        buffer = self._get_buffer()
        # 先计算分发布局：统计发往每个 rank / RDMA rank / 专家的 token 数，
        # 以及每个 token 是否属于某 rank，供后续 all-to-all 使用
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = buffer.get_dispatch_layout(
            topk_ids,
            self.num_experts,
            previous_event=previous_event,
            async_finish=self.async_finish,
            allocate_on_comm_stream=previous_event is not None,
        )
        # FIXME：`handle` 本应随 token 一起从 dispatch 传递到 combine。
        # 但这样做会引发一个未知的同步错误；把 `handle` 作为成员变量保存则可正常工作。

        _deepep_precompile_tp_barrier()
        (
            recv_x,
            recv_topk_ids,
            recv_topk_weights,
            num_recv_tokens_per_expert,
            self.handle,
            event,
        ) = buffer.dispatch(
            x,
            topk_idx=topk_ids,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=self.async_finish,
            allocate_on_comm_stream=(previous_event is not None) and self.async_finish,
            expert_alignment=128 if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM else 1,
            config=DeepEPConfig.get_instance().normal_dispatch_config,
        )
        get_global_expert_distribution_recorder().on_deepep_dispatch_normal(
            num_recv_tokens_per_expert,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            num_tokens_per_expert=num_tokens_per_expert,
        )

        return (
            recv_x,
            recv_topk_ids,
            recv_topk_weights,
            num_recv_tokens_per_expert,
            event,
        )

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        """combine 第一阶段：准备待聚合的专家输出并捕获 CUDA event。

        参数：
            hidden_states: 专家计算后的 token 特征。
            topk_ids / topk_weights: top-k 专家 id 与路由权重（普通模式下此处未直接使用，
                权重加权在 DeepEP 内部依据 handle 完成）。

        返回：
            (output, previous_event)，作为 combine_b 的输入。
        """
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM or _use_aiter or _is_npu:
            output = hidden_states
        else:
            raise NotImplementedError()  # 曾支持 triton runner，但目前暂时禁用

        previous_event = Buffer.capture() if self.async_finish else None
        return output, previous_event

    def combine_b(self, output, previous_event):
        """combine 第二阶段：执行 all-to-all 聚合，等待完成并清理句柄。

        参数：
            output, previous_event: 来自 combine_a 的输出。

        返回：
            combined hidden_states，形状还原为 [num_tokens, hidden_size]（每个 token 回到
            其原本所在 rank，并完成 top-k 加权聚合）。
        """
        hidden_states, event = self._combine_core(output, previous_event)
        event.current_stream_wait() if self.async_finish else ()
        self.handle = None
        self.src2dst = None
        return hidden_states

    def _combine_core(self, x: torch.Tensor, previous_event):
        """普通模式 combine 的核心实现：基于 dispatch 时保存的 handle 执行 all-to-all 回传。

        参数：
            x: 待聚合的专家输出。
            previous_event: 前置 CUDA event。

        返回：
            (combined_x, event)，combined_x 为聚合后的 token 特征。
        """
        buffer = self._get_buffer()
        _deepep_precompile_tp_barrier()
        combined_x, _, event = buffer.combine(
            x,
            self.handle,
            async_finish=self.async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=previous_event is not None,
            config=DeepEPConfig.get_instance().normal_combine_config,
        )
        return combined_x, event

    def _get_buffer(self):
        """将缓冲区置为普通模式并返回。"""
        DeepEPBuffer.set_dispatch_mode_as_normal()

        return DeepEPBuffer.get_deepep_buffer(
            self.group,
            self.hidden_size,
            self.params_bytes,
            self.deepep_mode,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
        )


class _DeepEPDispatcherImplLowLatency(_DeepEPDispatcherImplBase):
    dispatch_mode = DeepEPMode.LOW_LATENCY

    def __init__(self, return_recv_hook: bool, **kwargs):
        super().__init__(**kwargs)

        """
        num_max_dispatch_tokens_per_rank：解码引擎中的实际 batch size 应小于 256
        https://github.com/deepseek-ai/DeepEP?tab=readme-ov-file#example-use-in-inference-decoding
        """
        self.return_recv_hook = return_recv_hook
        self.device_module = torch.get_device_module()
        self.quant_config = {}

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        """dispatch 第一阶段（低时延）：发起 low_latency_dispatch 并估算每专家期望 token 数。

        参数：
            hidden_states: 本 rank 的 token 特征，形状 [num_tokens, hidden_size]。
            topk_output: 路由输出，包含 topk_ids 与 topk_weights。

        返回：
            (hidden_states, topk_ids, topk_weights, masked_m, expected_m, event, hook) 元组，
            作为 dispatch_b 的输入。其中 expected_m 为每专家期望处理的平均 token 数，
            计算公式为 (num_tokens * group_size * topk + num_experts) // num_experts。

        示例：
            >>> # num_tokens=1（decode），hidden_size=7168，topk=8
            >>> res = impl.dispatch_a(hidden_states, topk_output)
            >>> hs, ids, w, masked_m, expected_m, ev, hook = res
        """
        buffer = self._get_buffer()
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
        topk_ids = topk_ids.to(torch.int64)
        expected_m = (
            hidden_states.shape[0] * buffer.group_size * topk_ids.shape[1]
            + self.num_experts
        ) // self.num_experts
        hidden_states, masked_m, event, hook = self._dispatch_core(
            hidden_states,
            topk_ids,
            topk_weights,
        )
        return (
            hidden_states,
            topk_ids,
            topk_weights,
            masked_m,
            expected_m,
            event,
            hook,
        )

    def dispatch_b(
        self,
        hidden_states,
        topk_ids,
        topk_weights,
        masked_m,
        expected_m,
        event,
        hook,
    ):
        """dispatch 第二阶段（低时延）：等待接收完成并打包为 DeepEPLLDispatchOutput。

        若启用 return_recv_hook 则调用 hook() 等待接收；否则在当前流等待 event。

        参数：
            均来自 dispatch_a 的输出。

        返回：
            DeepEPLLDispatchOutput，包含按专家分块的 token 特征、（可选）量化缩放因子、
            top-k id/权重、masked_m 与 expected_m。

        示例：
            >>> out = impl.dispatch_b(hs, ids, w, masked_m, expected_m, ev, hook)
            >>> out.format
            <DispatchOutputFormat.DEEPEP_LL: ...>
        """
        hook() if self.return_recv_hook else event.current_stream_wait()

        get_global_expert_distribution_recorder().on_deepep_dispatch_low_latency(
            masked_m
        )

        if isinstance(hidden_states, tuple):
            hidden_states, hidden_states_scale = hidden_states
        else:
            hidden_states_scale = None

        deepep_output = DeepEPLLDispatchOutput(
            hidden_states,
            hidden_states_scale,
            topk_ids,
            topk_weights,
            masked_m,
            expected_m,
        )
        return deepep_output

    def _dispatch_core(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        """低时延 dispatch 的核心实现：调用 buffer.low_latency_dispatch 发起 RDMA 直达分发。

        参数：
            hidden_states: token 特征。
            topk_ids: top-k 专家 id。

        返回：
            (packed_recv_hidden, packed_recv_count, event, hook)：
            分别为按专家打包的接收特征、每专家接收计数、CUDA event 与接收 hook。
        """
        input_global_scale = self.quant_config.get("input_global_scale", None)

        # round_scale / use_ue8m0 是 FP8-DeepGEMM 专用选项；它们会使 DeepEP 返回
        # int32 打包的 UE8M0 缩放因子，而这种格式无法馈送给 flashinfer 的 cutedsl kernel。
        fp8_deepgemm_scale_opts = (
            dict(
                round_scale=deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
                and deep_gemm_wrapper.DEEPGEMM_BLACKWELL,
                use_ue8m0=deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
                and deep_gemm_wrapper.DEEPGEMM_BLACKWELL,
            )
            if self.use_fp8
            else dict()
        )

        buffer = self._get_buffer()
        _deepep_precompile_tp_barrier()
        packed_recv_hidden, self.packed_recv_count, self.handle, event, hook = (
            buffer.low_latency_dispatch(
                hidden_states,
                topk_ids,
                self.num_max_dispatch_tokens_per_rank,
                self.num_experts,
                use_fp8=self.use_fp8,
                **(
                    dict(topk_weights=topk_weights)
                    if _is_npu and not _use_zbal
                    else dict()
                ),
                **(dict(use_nvfp4=True) if self.use_nvfp4 else dict()),
                **(
                    dict(x_global_scale=input_global_scale)
                    if input_global_scale is not None
                    else dict()
                ),
                async_finish=not self.return_recv_hook,
                return_recv_hook=self.return_recv_hook,
                **fp8_deepgemm_scale_opts,
            )
        )
        return packed_recv_hidden, self.packed_recv_count, event, hook

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        """combine 第一阶段（低时延）：发起 low_latency_combine。

        参数：
            hidden_states: 专家计算后的 token 特征（按专家分块）。
            topk_ids / topk_weights: top-k 专家 id 与路由权重（用于加权聚合）。

        返回：
            (hidden_states, event, hook)，作为 combine_b 的输入。
        """
        hidden_states, event, hook = self._combine_core(
            hidden_states,
            topk_ids,
            topk_weights,
        )
        return hidden_states, event, hook

    def combine_b(self, hidden_states, event, hook):
        """combine 第二阶段（低时延）：等待聚合完成，处理可选的重叠流同步。

        若设置了 overlap，会在重叠流与当前流之间做双向等待，确保顺序正确。

        参数：
            hidden_states, event, hook: 来自 combine_a 的输出。

        返回：
            聚合后的 hidden_states。
        """
        overlap_args = self.overlap_args
        if overlap_args is not None:
            overlap_args.stream.wait_stream(self.device_module.current_stream())

        hook() if self.return_recv_hook else event.current_stream_wait()

        if overlap_args is not None:
            self.device_module.current_stream().wait_stream(overlap_args.stream)

        return hidden_states

    def _combine_core(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        """低时延 combine 的核心实现：在（可选的）重叠流上调用 low_latency_combine。

        会根据是否 Blackwell 架构组装不同的 overlap 参数（信号量、阈值、block_m 等）。

        参数：
            hidden_states: 专家输出特征。
            topk_ids / topk_weights: top-k 专家 id 与路由权重。

        返回：
            (combined_hidden_states, event, hook)。
        """
        buffer = self._get_buffer()
        overlap_args = self.overlap_args
        meta_overlap_args = self.meta_overlap_args

        ctx = nullcontext()
        if overlap_args is not None:
            overlap_args.stream.wait_event(overlap_args.wait_event)
            ctx = torch.cuda.stream(overlap_args.stream)

            if is_blackwell():
                overlap_args_dict = dict(
                    overlap=overlap_args.overlap,
                    src_signals=overlap_args.signal,
                    src_signal_expect_value=overlap_args.threshold,
                )
            else:
                overlap_args_dict = dict(
                    overlap=overlap_args.overlap,
                    packed_recv_count=self.packed_recv_count,
                    comp_signal=overlap_args.signal,
                    block_m=meta_overlap_args["block_m"],
                    threshold=meta_overlap_args["threshold"],
                    num_sms=overlap_args.num_sms,
                )
        else:
            overlap_args_dict = {}

        with ctx:
            _deepep_precompile_tp_barrier()
            combined_hidden_states, event, hook = buffer.low_latency_combine(
                x=hidden_states,
                topk_idx=topk_ids,
                topk_weights=topk_weights,
                handle=self.handle,
                async_finish=not self.return_recv_hook,
                return_recv_hook=self.return_recv_hook,
                **overlap_args_dict,
            )

        self.packed_recv_count = self.handle = None
        return combined_hidden_states, event, hook

    def _get_buffer(self):
        """将缓冲区置为低时延模式并返回。"""
        DeepEPBuffer.set_dispatch_mode_as_low_latency()
        return DeepEPBuffer.get_deepep_buffer(
            self.group,
            self.hidden_size,
            self.params_bytes,
            self.deepep_mode,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
        )


@dataclass
class _Stage(Enum):
    INITIAL = auto()
    AFTER_DISPATCH_A = auto()
    AFTER_DISPATCH_B = auto()
    AFTER_COMBINE_A = auto()


class DeepEPDispatcher(BaseDispatcher):
    """DeepEP 分发器对外统一入口。

    根据运行模式（普通/低时延/AUTO）在内部持有对应的实现（_DeepEPDispatcherImplNormal
    / _DeepEPDispatcherImplLowLatency），并对外暴露 dispatch / combine（及其 a/b 子阶段）。
    AUTO 模式会在运行时按当前是 prefill 还是 decode 自动选择具体实现。

    典型用法：
        >>> dispatcher = DeepEPDispatcher(group, router_topk=8, num_experts=256,
        ...     num_local_experts=8, hidden_size=7168, params_dtype=torch.bfloat16)
        >>> dispatch_out = dispatcher.dispatch(hidden_states, topk_output)
        >>> # ... 专家计算得到 expert_out ...
        >>> combine_in = DeepEPNormalCombineInput(expert_out, topk_ids, topk_weights)
        >>> result = dispatcher.combine(combine_in)  # 形状还原为 [num_tokens, hidden_size]

    参数：
        group: 专家并行进程组。
        router_topk: 每个 token 选择的专家个数（top-k）。
        permute_fusion: 是否启用置换融合优化。
        num_experts / num_local_experts: 专家总数 / 本 rank 上的专家数。
        hidden_size: 隐藏层维度。
        params_dtype: 参数 dtype（如 torch.bfloat16）。
        deepep_mode: 运行模式，默认 AUTO（运行时自动选择普通/低时延）。
        async_finish: 普通模式是否异步完成（用于通信/计算重叠）。
        return_recv_hook: 低时延模式是否返回接收 hook（用于通信/计算重叠）。
    """

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.AUTO,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        super().__init__()

        self.deepep_mode = deepep_mode

        common_kwargs = dict(
            group=group,
            router_topk=router_topk,
            permute_fusion=permute_fusion,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            params_dtype=params_dtype,
            deepep_mode=deepep_mode,
        )

        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher = _DeepEPDispatcherImplLowLatency(
                return_recv_hook=return_recv_hook,
                **common_kwargs,
            )
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher = _DeepEPDispatcherImplNormal(
                async_finish=async_finish,
                **common_kwargs,
            )

        self._stage = _Stage.INITIAL
        self._deepep_dispatch_hooks = DeepEPPDispatchHooks()

        # DeepEP/Mooncake/Nixl 会用 -1 标记无效的 topk 槽位；AITER 的 pre_permute
        # 会把它们重定向到索引 num_local_experts 处的「汇点」槽位，并在这里被掩码屏蔽。
        self.expert_mask_gpu = None
        if _use_aiter and num_local_experts is not None:
            expert_mask = torch.zeros(
                num_local_experts + 1,
                device=torch.cuda.current_device(),
                dtype=torch.int,
            )
            expert_mask[:-1] = 1
            self.expert_mask_gpu = expert_mask

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> DispatchOutput:
        """同步执行完整分发（dispatch_a -> 触发 hook -> dispatch_b）。

        参数：
            hidden_states: 本 rank 的 token 特征，形状 [num_tokens, hidden_size]。
            topk_output: 路由输出（包含 topk_ids 与 topk_weights）。

        返回：
            DispatchOutput（普通模式为 DeepEPNormalDispatchOutput，低时延为
            DeepEPLLDispatchOutput）。
        """
        self.dispatch_a(hidden_states, topk_output)
        if self._deepep_dispatch_hooks is not None:
            self._deepep_dispatch_hooks(self)
        ret = self.dispatch_b()
        return ret

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        """分发第一阶段：推进状态机并调用具体实现的 dispatch_a，保存中间状态。"""
        self._update_stage(_Stage.INITIAL, _Stage.AFTER_DISPATCH_A)
        inner_state = self._get_impl().dispatch_a(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )
        self._dispatch_intermediate_state = inner_state

    def dispatch_b(self):
        """分发第二阶段：取回中间状态并调用具体实现的 dispatch_b，返回 DispatchOutput。"""
        self._update_stage(_Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)
        inner_state = self._dispatch_intermediate_state
        del self._dispatch_intermediate_state
        return self._get_impl().dispatch_b(*inner_state)

    def combine(
        self,
        combine_input: CombineInput,
    ) -> torch.Tensor:
        """同步执行完整聚合（combine_a -> combine_b）。

        参数：
            combine_input: CombineInput（普通模式为 DeepEPNormalCombineInput，低时延为
                DeepEPLLCombineInput），包含专家输出特征、topk_ids 与 topk_weights。

        返回：
            聚合后的 hidden_states，形状 [num_tokens, hidden_size]。
        """
        self.combine_a(combine_input)
        ret = self.combine_b()
        return ret

    def combine_a(
        self,
        combine_input: CombineInput,
    ):
        """聚合第一阶段：拆包输入、推进状态机并调用具体实现的 combine_a。"""
        hidden_states, topk_ids, topk_weights = combine_input
        self._update_stage(_Stage.AFTER_DISPATCH_B, _Stage.AFTER_COMBINE_A)
        inner_state = self._get_impl().combine_a(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        self._combine_intermediate_state = inner_state

    def combine_b(self):
        """聚合第二阶段：取回中间状态并调用具体实现的 combine_b，返回聚合结果。"""
        self._update_stage(_Stage.AFTER_COMBINE_A, _Stage.INITIAL)
        inner_state = self._combine_intermediate_state
        del self._combine_intermediate_state
        return self._get_impl().combine_b(*inner_state)

    def _get_impl(self) -> _DeepEPDispatcherImplBase:
        """解析当前应使用的具体实现（普通/低时延）。

        AUTO 模式下根据当前是否为 prefill（is_extend_in_batch）来自动选择：
        prefill 选普通模式，decode 选低时延模式。
        """
        is_extend_in_batch = get_is_extend_in_batch()
        resolved_deepep_mode = self.deepep_mode.resolve(is_extend_in_batch)
        if resolved_deepep_mode == DeepEPMode.NORMAL:
            return self._normal_dispatcher
        elif resolved_deepep_mode == DeepEPMode.LOW_LATENCY:
            return self._low_latency_dispatcher
        else:
            raise ValueError(f"Invalid deepep_mode: {self.deepep_mode}")

    def _update_stage(self, old_stage, new_stage):
        """推进阶段状态机：断言当前处于 old_stage，然后切换到 new_stage。

        用于防止 a/b 子阶段被错误顺序或重复调用。
        """
        assert self._stage == old_stage
        self._stage = new_stage

    def set_quant_config(self, quant_config: dict):
        """为已启用的（低时延/普通）子分发器同步设置量化配置。"""
        super().set_quant_config(quant_config)
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher.set_quant_config(quant_config)
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher.set_quant_config(quant_config)

    def set_overlap_args(
        self, combine_overlap_args: CombineOverlapArgs, meta_overlap_args: dict
    ):
        """为已启用的子分发器同步设置计算/通信重叠参数。"""
        super().set_overlap_args(combine_overlap_args, meta_overlap_args)
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher.set_overlap_args(
                combine_overlap_args, meta_overlap_args
            )
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher.set_overlap_args(
                combine_overlap_args, meta_overlap_args
            )

    def clear_overlap_args(self):
        """清除已启用子分发器的重叠参数。"""
        super().clear_overlap_args()
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher.clear_overlap_args()
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher.clear_overlap_args()

    def register_deepep_dispatch_hook(self, hook):
        """注册分发 hook（在 dispatch_a 与 dispatch_b 之间被触发）。"""
        return self._deepep_dispatch_hooks.register_hook(hook)
