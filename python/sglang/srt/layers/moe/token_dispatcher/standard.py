from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional, Tuple

import torch

from sglang.srt.distributed import (
    get_tp_group,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import (
    get_dp_global_num_tokens,
    get_local_dp_buffer,
    is_allocation_symmetric,
    is_dp_max_padding,
    mask_dp_pad_moe_topk_ids,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutputChecker
from sglang.srt.layers.moe.utils import (
    get_moe_a2a_backend,
    get_moe_runner_backend,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.common import (
    get_bool_env_var,
    get_device,
    is_hip,
)

# AITER 是 AMD/HIP 平台上的 MoE 计算后端。它接收全局专家 ID，并通过
# expert_mask_gpu 判断当前 rank 上哪些专家有效，因此不在此处改写 topk_ids。
_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

from sglang.srt.environ import envs as _envs

_MASK_DP_PAD_MOE = _envs.SGLANG_OPT_MASK_DP_PAD_MOE.get()

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKOutput


# FlashInfer 并非所有部署环境的必选依赖，因此延迟到模块导入阶段做能力探测。
# 缺少依赖时保持占位值为 None，仅在真正选择 FP4 all-gather 路径时抛出明确错误。
try:
    from flashinfer import (
        nvfp4_block_scale_interleave as nvfp4_block_scale_interleave_flashinfer,
    )

    from sglang.srt.layers.quantization.modelopt_quant import (
        fp4_quantize as fp4_quantize_flashinfer,
    )
except ImportError:
    fp4_quantize_flashinfer = None
    nvfp4_block_scale_interleave_flashinfer = None


class StandardDispatchOutput(NamedTuple):
    """标准 dispatcher 传给 MoE runner 的结构化输出。

    标准路径通常不执行专用 All-to-All，而是保留输入 token 布局，并在需要时把
    ``topk_ids`` 从全局专家 ID 映射成本 rank 的本地专家 ID。特殊的 FlashInfer
    CUTLASS FP4 路径会先量化并 all-gather，因此输出首维可能是多个 DP rank 的
    token 数之和。

    字段：
        hidden_states：普通路径为激活值 ``[T_local, H]``；FP4 路径为打包后的
            ``uint8`` 张量，典型 shape 为 ``[T_global, H / 2]``。
        hidden_states_scale：普通路径为 ``None``；FP4 路径为量化 scale，典型
            shape 为 ``[T_global, H / 16]``，随后转换成 CUTLASS 所需布局。
        topk_output：与 hidden 对齐的 Top-K 专家 ID、权重和可选 router logits。

    输入输出示例：
        普通路径输入 hidden ``[8, 4096]``、Top-K ``[8, 2]``，输出仍为
        ``hidden_states=[8, 4096]``、``hidden_states_scale=None``。
        若 EP rank 1 只持有全局专家 4～7，则全局 ID ``[4, 7]`` 可映射为
        本地 ID ``[0, 3]``，不属于本 rank 的专家映射为 ``-1``。
    """

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_output: TopKOutput
    # SGLANG_OPT_MOE_QUANT_ONCE: optional pre-quantized (q, scale) pair for
    # ``hidden_states`` (per-token-group-128 fp8, q rows possibly padded to a
    # multiple of 4). Consumed by the standard->triton fused runner so it can
    # skip its own activation quant; ``hidden_states`` itself stays bf16.
    hidden_states_pre_quant: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    @property
    def format(self) -> DispatchOutputFormat:
        """返回标准 dispatch 数据格式标识。

        输入：当前 ``StandardDispatchOutput`` 实例。
        输出示例：``DispatchOutputFormat.STANDARD``。
        """
        return DispatchOutputFormat.STANDARD


# 运行时确认 NamedTuple 满足公共协议：它必须包含 hidden_states 和 format。
assert isinstance(StandardDispatchOutput, DispatchOutput)


class StandardCombineInput(NamedTuple):
    """标准 dispatcher 的 combine 输入。

    字段：
        hidden_states：本地 MoE runner 已完成专家计算和 Top-K 聚合后的输出。

    输入输出示例：
        普通路径输入 ``hidden_states.shape == [8, 4096]``，combine 直接返回同一
        张量；FP4 all-gather 路径可输入全局 ``[T_global, 4096]``，combine 通过
        reduce-scatter 恢复当前 DP rank 的 ``[T_local, 4096]``。
    """

    hidden_states: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        """返回标准 combine 数据格式标识。

        输入：当前 ``StandardCombineInput`` 实例。
        输出示例：``CombineInputFormat.STANDARD``。
        """
        return CombineInputFormat.STANDARD


# 运行时确认 NamedTuple 至少提供公共 CombineInput 协议要求的 format 属性。
assert isinstance(StandardCombineInput, CombineInput)


class StandardDispatcher(BaseDispatcher):
    def __init__(self, moe_runner_config: MoeRunnerConfig):
        """初始化后端能力、专家布局和延迟创建的映射缓存。

        参数：
            moe_runner_config：包含全局/本地专家数、共享专家数、hidden size 等
                MoE runner 配置。本类主要使用专家数量字段。

        输入输出示例：
            输入 ``num_experts=8``、``num_local_experts=4``、
            ``num_fused_shared_experts=0``，且当前 ``ep_rank=1``；初始化后
            ``num_local_routed_experts=4``。首次 dispatch 时会构造映射
            ``[-1, -1, -1, -1, 0, 1, 2, 3]``。
        """
        super().__init__()
        self.moe_ep_size = get_parallel().moe_ep_size
        backend = get_moe_runner_backend()
        self.enable_flashinfer_cutlass_moe = backend.is_flashinfer_cutlass()
        self.enable_flashinfer_mxfp4_moe = backend.is_flashinfer_mxfp4()
        self.enable_flashinfer_trtllm_routed_moe = backend.is_flashinfer_trtllm_routed()
        # AITER fast paths can be on while the MoE runner stays Triton; only the
        # AITER runner keeps global expert IDs, so Triton must remap to local range.
        self.use_aiter_moe_runner = backend.is_aiter() or (
            backend.is_auto() and _use_aiter and get_moe_a2a_backend().supports_aiter()
        )
        # Skip local expert mapping when the backend handles EP with global expert IDs:
        # - cutlass / cutedsl / trtllm_routed handle EP internally
        # - mxfp4 dispatcher mapping is already global
        # - hpc_ops consumes global ids together with rank_ep / num_expert_total
        self.skip_local_expert_mapping = (
            backend.is_flashinfer_cutlass()
            or backend.is_flashinfer_cutedsl()
            or backend.is_flashinfer_trtllm()
            or backend.is_experimental_sgl_trtllm()
            or backend.is_flashinfer_trtllm_routed()
            or backend.is_hpc_ops()
            or self.enable_flashinfer_mxfp4_moe
        )

        self.num_experts = moe_runner_config.num_experts
        self.num_local_experts = moe_runner_config.num_local_experts
        self.num_local_shared_experts = moe_runner_config.num_fused_shared_experts
        self.num_local_routed_experts = (
            self.num_local_experts - self.num_local_shared_experts
        )
        self.moe_ep_rank = get_parallel().moe_ep_rank
        self.local_expert_mapping = None

        # AITER 使用单独的有效专家 mask，而不是直接把 topk_ids 映射成本地 ID。
        self.expert_mask_gpu = None

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> StandardDispatchOutput:
        """准备标准 MoE runner 所需的 hidden states 和 Top-K 路由信息。

        参数：
            hidden_states：本 rank 当前 token 激活，典型 shape 为 ``[T_local, H]``。
            topk_output：Top-K 路由结果，``topk_ids``、``topk_weights`` 的典型
                shape 均为 ``[T_local, K]``。

        返回：
            ``StandardDispatchOutput``。常规路径保留 hidden shape，只按需要改写
            专家 ID；FlashInfer FP4 路径返回 all-gather 后的量化 hidden、scale 和
            全局 Top-K 数据。

        输入输出示例：
            假设 ``ep_size=2``、8 个专家、每 rank 4 个专家，当前 rank 为 1。
            输入 ``topk_ids=[[4, 1], [7, 5]]``，常规映射后得到
            ``[[0, -1], [3, 1]]``；``-1`` 表示该专家不在当前 rank。

            FP4 + DP Attention 路径中，两个 rank 的 token 数分别为 3 和 5，当前
            hidden ``[3, 4096]`` 量化后参与 all-gatherv，输出 token 首维为 8；
            combine 阶段再按 ``[3, 5]`` reduce-scatter 回各来源 rank。
        """
        if should_use_flashinfer_cutlass_moe_fp4_allgather():
            # FlashInfer CUTLASS FP4 路径需要聚合所有 DP rank 的量化激活和 Top-K。
            # 通信前量化可降低 payload；通信后再转换 scale 布局，避免发送 swizzle
            # 后可能带来的额外布局约束。
            if (
                fp4_quantize_flashinfer is None
                or nvfp4_block_scale_interleave_flashinfer is None
            ):
                raise RuntimeError(
                    "FlashInfer fp4_quantize and nvfp4_block_scale_interleave "
                    "are required for the flashinfer_cutlass FP4 all-gather "
                    "path."
                )

            # 上面的依赖检查既用于给用户提供明确错误，也用于帮助静态类型检查器
            # 确认后续两个可选函数在该分支中一定可调用。
            # IDE 对可选依赖的跨模块控制流收窄能力有限；运行时检查已在上方完成，
            # 此处只通过 cast 告知类型检查器这两个对象在当前分支中必定可调用。
            fp4_quantize = cast(Callable, cast(object, fp4_quantize_flashinfer))
            block_scale_interleave = cast(
                Callable, cast(object, nvfp4_block_scale_interleave_flashinfer)
            )

            # FP4 all-gather 需要标准 Top-K 的 ids/weights/router_logits 三个字段。
            assert TopKOutputChecker.format_is_standard(topk_output)

            # input_global_scale 由量化方法通过 BaseDispatcher.set_quant_config 注入。
            global_scale = self.quant_config.get("input_global_scale", None)
            assert global_scale is not None, "input_global_scale is not set"
            topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids

            # 在通信前完成 NVFP4 量化，通信后再对 scale 做 CUTLASS 布局交错。
            # 对称内存允许通信后端在条件满足时使用更高效的对称内存 collective。
            with use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            ):
                if hidden_states.shape[0] > 0:
                    x, x_sf = fp4_quantize(
                        hidden_states, global_scale, is_sf_swizzled_layout=False
                    )
                else:
                    # 某个 DP rank 可以没有 token，但 collective 仍要求所有 rank 参与。
                    # 构造首维为 0、其余维度合法的量化张量，避免调用量化 kernel。
                    x_col = hidden_states.shape[1]
                    x = torch.zeros(
                        0, x_col // 2, dtype=torch.uint8, device=hidden_states.device
                    )
                    x_sf = torch.zeros(
                        0, x_col // 16, dtype=torch.uint8, device=hidden_states.device
                    )

            # all_gatherv 支持各 DP rank 不同的 token 数；sizes 记录每个 rank 的
            # T_local，使四个张量按照相同 token 边界拼接。
            topk_weights, topk_ids, x, x_sf = get_tp_group().all_gatherv(
                [topk_weights, topk_ids, x, x_sf], sizes=get_dp_global_num_tokens()
            )

            # TODO：将 scale interleave 融合进 CUTLASS MoE，减少一次独立布局转换。
            x_sf = block_scale_interleave(x_sf)

            hidden_states = x
            hidden_states_scale = x_sf
            topk_output = StandardTopKOutput(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                # 该特殊路径的 runner 当前不消费 router_logits，原值仅用于保持协议。
                router_logits=topk_output.router_logits,
            )
        else:
            # 常规标准路径不搬运、不量化 hidden；显式赋值用于统一后续返回结构。
            hidden_states = hidden_states
            hidden_states_scale = None

        # 仅当 EP 实际切分了专家、runner 不原生处理全局 ID、且 Top-K 使用标准格式
        # 时，才需要构造 global-to-local expert 映射。
        if (
            self.moe_ep_size > 1
            and not self.skip_local_expert_mapping
            and TopKOutputChecker.format_is_standard(topk_output)
        ):
            if self.local_expert_mapping is None:
                device = get_device()

                # 默认填 -1：所有不属于当前 rank 的全局专家都视为无效 assignment。
                self.local_expert_mapping = torch.full(
                    (self.num_experts,), -1, dtype=torch.int32, device=device
                )

                # routed experts 按 EP rank 连续切分。例：每 rank 4 个专家、rank=1，
                # 则全局 [4:8] 分别映射到本地 [0:4]。
                self.local_expert_mapping[
                    self.moe_ep_rank
                    * self.num_local_routed_experts : (self.moe_ep_rank + 1)
                    * self.num_local_routed_experts
                ] = torch.arange(
                    0, self.num_local_routed_experts, dtype=torch.int32, device=device
                )

                if self.num_local_shared_experts > 0:
                    # fused shared experts 位于全局专家表尾部，并在每个 rank 本地存在；
                    # 它们排在本地 routed experts 之后。
                    self.local_expert_mapping[-self.num_local_shared_experts :] = (
                        torch.arange(
                            self.num_local_routed_experts,
                            self.num_local_routed_experts
                            + self.num_local_shared_experts,
                            dtype=torch.int32,
                            device="cpu",
                        )
                    )

        if self.local_expert_mapping is not None and not self.skip_local_expert_mapping:
            if self.use_aiter_moe_runner and self.expert_mask_gpu is None:
                self.expert_mask_gpu = (
                    (
                        (self.local_expert_mapping >= 0)
                        & (self.local_expert_mapping < self.num_local_experts)
                    )
                    .to(torch.int32)
                    .to(device="cuda")
                )
            elif not self.use_aiter_moe_runner:
                if TopKOutputChecker.format_is_standard(topk_output):
                    topk_ids_local = self.local_expert_mapping[topk_output.topk_ids]
                    # Drop dp-attention MAX_LEN pad rows from the dispatch:
                    # pad rows carry stale hidden through the router and
                    # their expert outputs are discarded downstream — pure
                    # wasted compute (and a masked-grouped-GEMM workspace
                    # blow-up when they collide on the same top-k).  Must
                    # run POST-translation (a pre-translation -1 aliases to
                    # the mapping table's last entry); -1 is the drop
                    # sentinel both the triton and deep_gemm runners honor.
                    if _MASK_DP_PAD_MOE and is_dp_max_padding():
                        mask_dp_pad_moe_topk_ids(topk_ids_local)
                    topk_output = topk_output._replace(topk_ids=topk_ids_local)
                elif TopKOutputChecker.format_is_triton_kernels(topk_output):
                    # Triton-kernels 的非标准 Top-K 映射尚未实现；避免静默产生错路由。
                    raise NotImplementedError()

        return StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            topk_output=topk_output,
        )

    def combine(self, combine_input: StandardCombineInput) -> torch.Tensor:
        """返回标准 MoE 结果，必要时执行与 FP4 all-gather 配对的 reduce-scatter。

        参数：
            combine_input：MoE runner 计算完成后的 ``StandardCombineInput``。
                常规路径中 hidden 为 ``[T_local, H]``；FP4 all-gather 路径中通常为
                ``[T_global, H]``。

        返回：
            当前来源 rank 的 hidden states。常规路径直接返回输入；FP4 路径把全局
            token 结果按各 DP rank 的 token 数切分并求和，返回 ``[T_local, H]``。

        输入输出示例：
            常规路径输入 ``[8, 4096]``，输出仍为 ``[8, 4096]``，且不发生通信。
            若两个 DP rank 的 token 数为 ``[3, 5]``，每个 rank 的 runner 产生
            ``[8, 4096]`` 全局结果，则 reduce-scatter 后 rank 0 输出 ``[3, 4096]``，
            rank 1 输出 ``[5, 4096]``。
        """
        (hidden_states,) = combine_input
        if should_use_flashinfer_cutlass_moe_fp4_allgather():
            # get_local_dp_buffer 返回按当前 rank 的 T_local 预分配的目标 buffer。
            # global_hidden_states 是 runner 对 all-gather 后全局 token 的计算结果。
            hidden_states, global_hidden_states = (
                get_local_dp_buffer(get_tp_group()),
                hidden_states,
            )

            # reduce_scatterv 与 dispatch 中的 all_gatherv 配对：先对各 rank 的专家
            # 贡献求和，再根据不等长 sizes 把对应 token 区间送回来源 DP rank。
            get_tp_group().reduce_scatterv(
                global_hidden_states,
                output=hidden_states,
                sizes=get_dp_global_num_tokens(),
            )
        return hidden_states
