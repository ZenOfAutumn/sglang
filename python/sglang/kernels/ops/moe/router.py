from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.srt.utils import is_hip

_is_hip = is_hip()


@triton.jit
def fused_moe_router_cudacore_kernel(
    input_ptr,  # 输入 hidden states，形状为 (bs, hidden_dim)。
    moe_router_weight_ptr,  # 输入 router 权重，形状为 (num_experts, hidden_dim)。
    topk_weights_ptr,  # 输出路由权重，形状为 (bs, topk)。
    topk_ids_ptr,  # 输出专家 ID，形状为 (bs, topk)。
    correction_bias_ptr,
    is_correction_bias: tl.constexpr,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    moe_softcapping: tl.constexpr,
    moe_renormalize: tl.constexpr,  # 暂不支持对入选 Top-K 权重二次归一化。
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """CUDA Core 版融合 router kernel：一个 Triton program 处理一个 token。

    对每个 token 直接完成 ``x @ router_weight.T``、softcap、可选 correction bias、
    softmax 概率计算和 Top-K 选择，不物化完整 ``[bs, num_experts]`` router logits。

    输入输出示例：``input=[4,4096]``、``weight=[8,4096]``、``topk=2`` 时启动
    4 个 program，最终写出 ``topk_weights/topk_ids=[4,2]``。
    """
    # 第 pid 个 program 负责 batch 中第 pid 个 token。
    pid = tl.program_id(axis=0)

    # hidden_dim 补齐到 2 的幂，mask 防止访问实际隐藏维之外的元素。
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    # router 权重以隐藏维 K 为连续维度：每一行对应一个专家。
    expert_offsets = tl.arange(0, num_experts)[:, None]
    router_mask = mask[None, :]
    w_router = tl.load(
        moe_router_weight_ptr + expert_offsets * hidden_dim + offsets[None, :],
        mask=router_mask,
        other=0.0,
    )

    x = tl.load(input_ptr + pid * hidden_dim + offsets, mask=mask, other=0.0)

    # TODO：评估是否可用 tl.dot；当前逐元素乘加走 CUDA Core，更适合小 batch/专家数。
    logits = tl.sum((w_router.to(tl.float32) * x[None, :].to(tl.float32)), axis=-1)

    # 对 router logits 做 softcap：c * tanh(logit / c)，抑制极端值。
    if moe_softcapping == 0:
        logits_softcapped = logits
    else:
        logits_scaled = logits / moe_softcapping
        exped = tl.exp(2 * logits_scaled)
        top = exped - 1
        bottom = exped + 1
        logits_softcapped = top / bottom * moe_softcapping

    # softcap 后再加专家校正偏置；该偏置会同时影响专家选择和这里计算的权重。
    if is_correction_bias:
        bias = tl.load(correction_bias_ptr + tl.arange(0, num_experts))
        logits_softcapped = logits_softcapped + bias

    # 选择 Top-K，调用方应保证 1 <= topk <= num_experts。
    # 以下耗时注释来自该 kernel 开发时的微基准，仅用于观察各步骤增量。

    # 约 5.38 us：完成 logits 与 softmax 分母计算前的准备。

    top1 = tl.argmax(logits_softcapped, axis=0)
    tl.store(topk_ids_ptr + pid * topk + 0, top1)  # 5.63 us

    top1_v = tl.max(logits_softcapped, axis=0)
    invsumexp = 1.0 / tl.sum(tl.exp(logits_softcapped - top1_v), axis=0)

    tl.store(
        topk_weights_ptr + pid * topk + 0,
        invsumexp,
    )  # 5.73 us

    if topk >= 2:
        top2 = tl.argmax(
            tl.where(
                tl.arange(0, num_experts) != top1, logits_softcapped, float("-inf")
            ),
            axis=0,
        )
        tl.store(topk_ids_ptr + pid * topk + 1, top2)
        top2_v = tl.sum(logits_softcapped * (tl.arange(0, num_experts) == top2), axis=0)
        tl.store(
            topk_weights_ptr + pid * topk + 1,
            tl.exp(top2_v - top1_v) * invsumexp,
        )  # 5.95us

    # Top-K>2 时需要循环排除已选专家，速度通常较慢；该路径更适合较小 K。
    if topk > 2:
        topk_mask = tl.full(logits_softcapped.shape, 1.0, dtype=logits_softcapped.dtype)
        topk_mask = tl.where(
            tl.arange(0, num_experts) != top1, topk_mask, float("-inf")
        )
        topk_mask = tl.where(
            tl.arange(0, num_experts) != top2, topk_mask, float("-inf")
        )
        for i in range(2, topk):
            topi = tl.argmax(logits_softcapped + topk_mask, axis=0)
            topk_mask = tl.where(
                tl.arange(0, num_experts) != topi, topk_mask, float("-inf")
            )
            tl.store(topk_ids_ptr + pid * topk + i, topi)
            topi_v = tl.sum(
                logits_softcapped * (tl.arange(0, num_experts) == topi), axis=0
            )
            tl.store(
                topk_weights_ptr + pid * topk + i,
                tl.exp(topi_v - top1_v) * invsumexp,
            )
    # 当前输出是完整专家 softmax 中入选项的概率，尚未实现 Top-K 内再次归一化。


def fused_moe_router_cudacore(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    topk: int,
    moe_softcapping: float,
    correction_bias: Optional[torch.Tensor] = None,
):
    """启动 CUDA Core 版融合 MoE router。

    参数：
        x: token hidden states，形状 ``[bs, hidden_dim]``。
        router_weight: router 线性层权重，形状 ``[num_experts, hidden_dim]``。
        topk: 每个 token 选择的专家数。
        moe_softcapping: logits softcap 上限；0 表示禁用。
        correction_bias: 可选专家偏置，形状 ``[num_experts]``。

    输入输出示例：``x=[4,4096]``、``router_weight=[8,4096]``、``topk=2``，返回
    float32 权重 ``[4,2]`` 和 int32 专家 ID ``[4,2]``。
    """
    assert len(x.shape) == 2 and x.shape[1] == router_weight.shape[1]
    bs, hidden_dim = x.shape
    num_experts = router_weight.shape[0]

    # 不分配完整 router_logits；kernel 计算后直接保留 Top-K，节省中间显存和 kernel launch。
    topk_weights = torch.empty((bs, topk), dtype=torch.float32, device=x.device)
    topk_ids = torch.empty((bs, topk), dtype=torch.int32, device=x.device)
    is_correction_bias = correction_bias is not None

    max_warps = 16 if _is_hip else 32
    config = {
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), max_warps), 4
        ),
    }

    fused_moe_router_cudacore_kernel[(bs,)](
        x,
        router_weight,
        topk_weights,
        topk_ids,
        correction_bias,
        is_correction_bias=is_correction_bias,
        num_experts=num_experts,
        topk=topk,
        moe_softcapping=moe_softcapping,
        moe_renormalize=False,
        hidden_dim=hidden_dim,
        **config,
    )

    return topk_weights, topk_ids


@triton.jit
def fused_moe_router_tensorcore_kernel(
    a_ptr,  # 输入 hidden states，形状为 (bs, hidden_dim)。
    b_ptr,  # 输入 router 权重，形状为 (num_experts, hidden_dim)。
    topk_weights_ptr,  # 输出路由权重，形状为 (bs, topk)。
    topk_ids_ptr,  # 输出专家 ID，形状为 (bs, topk)。
    bs,
    num_experts: tl.constexpr,
    topk: tl.constexpr,  # Tensor Core 路径目前只支持 topk <= 2。
    moe_softcapping: tl.constexpr,
    moe_renormalize: tl.constexpr,  # 暂不支持对入选 Top-K 权重二次归一化。
    correction_bias_ptr,
    is_correction_bias: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_bn: tl.constexpr,
    dp_attn_workaround_flag: tl.constexpr,
):
    # 1. get block id
    pid = tl.program_id(axis=0)

    # 2. 创建 A、B 首个分块的指针。
    # 2.1 按 token 维 M 和隐藏维 K 计算 A 的指针。
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
    bs_mask = offs_m < bs
    offs_k = tl.arange(0, BLOCK_SIZE_K)[None, :]
    a_ptrs = a_ptr + (offs_m * stride_am + offs_k)

    # 2.2 按隐藏维 K 和专家维 N 计算 B 的指针。
    # 注意：B 的物理布局以 K 为连续维度，逻辑计算时转置为 [K, N]。
    offs_k = tl.arange(0, BLOCK_SIZE_K)[None, :]
    offs_n = tl.arange(0, BLOCK_SIZE_N)[:, None]
    expert_mask = offs_n < num_experts
    b_ptrs = b_ptr + (offs_n * stride_bn + offs_k)

    # 3. 创建 [BLOCK_SIZE_M, BLOCK_SIZE_N] 的 FP32 累加器；沿 K 维遍历分块，
    #    并转置 B tile 后执行矩阵乘。
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K // BLOCK_SIZE_K):  # hidden_dim % BLOCK_SIZE_K == 0
        a = tl.load(
            a_ptrs,
            mask=bs_mask,
            other=0.0,
        ).to(tl.float32)
        b = tl.load(b_ptrs, mask=expert_mask, other=0.0).to(tl.float32).T
        acc += tl.dot(a, b)

        # 把 A、B 指针推进到下一个 K 分块。
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K

    # 4. 对 logits 应用 c * tanh(logit/c) softcap。
    if moe_softcapping == 0:
        logits_softcapped = acc
    else:
        logits_scaled = acc / moe_softcapping
        exped = tl.exp(2 * logits_scaled)
        logits_softcapped = (exped - 1) / (exped + 1) * moe_softcapping

    # softcap 后再加专家校正偏置；该偏置会同时影响专家选择和这里计算的权重。
    if is_correction_bias:
        bias = tl.load(
            correction_bias_ptr + tl.arange(0, BLOCK_SIZE_N)[None, :],
            mask=expert_mask.T,
            other=0.0,
        )
        logits_softcapped = logits_softcapped + bias

    if dp_attn_workaround_flag:
        logits_softcapped = tl.where(
            logits_softcapped != logits_softcapped, -1e9, logits_softcapped
        )

    # 5. 屏蔽补齐的专家列，选择 Top-1 并计算其完整 softmax 概率。
    arange_block_size_n = tl.arange(0, BLOCK_SIZE_N)[None, :]
    cond_top1 = arange_block_size_n < num_experts
    top1 = tl.argmax(tl.where(cond_top1, logits_softcapped, float("-inf")), axis=1)
    top1_v = tl.max(
        tl.where(cond_top1, logits_softcapped, float("-inf")), axis=1, keep_dims=True
    )
    top1_invsumexp = 1.0 / tl.sum(
        tl.where(cond_top1, tl.exp(logits_softcapped - top1_v), 0.0), axis=1
    )

    # 6. 将 Top-1 专家 ID 和权重写入输出，并屏蔽超出真实 batch 的 token。
    offs_top1 = pid * topk * BLOCK_SIZE_M + topk * tl.arange(0, BLOCK_SIZE_M)
    top1_mask = offs_top1 < bs * topk
    tl.store(topk_ids_ptr + offs_top1, top1, mask=top1_mask)
    tl.store(
        topk_weights_ptr + offs_top1,
        top1_invsumexp,
        mask=top1_mask,
    )

    # 7. Top-2 场景：排除 Top-1 后选择次大专家。
    if topk == 2:
        cond_top2 = (arange_block_size_n < num_experts) & (
            arange_block_size_n != top1[:, None]
        )
        top2 = tl.argmax(
            tl.where(cond_top2, logits_softcapped, float("-inf")),
            axis=1,
            keep_dims=True,
        )
        top2_v = tl.sum(
            logits_softcapped * (arange_block_size_n == top2), axis=1, keep_dims=True
        )
        top2_invsumexp = tl.exp(top2_v - top1_v) * top1_invsumexp[:, None]

        # 将 Top-2 专家 ID 和对应的完整 softmax 概率写入输出。
        offs_top2 = (
            pid * topk * BLOCK_SIZE_M + topk * tl.arange(0, BLOCK_SIZE_M)[:, None] + 1
        )
        top2_mask = offs_top2 < bs * topk
        tl.store(topk_ids_ptr + offs_top2, top2, mask=top2_mask)
        tl.store(
            topk_weights_ptr + offs_top2,
            top2_invsumexp,
            mask=top2_mask,
        )


def fused_moe_router_tensorcore(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    topk: int,
    moe_softcapping: float,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_K: int,
    correction_bias: Optional[torch.Tensor] = None,
):
    """启动 Tensor Core 版融合 MoE router。

    ``BLOCK_SIZE_M/N/K`` 分别控制 token、专家、隐藏维的 GEMM tile。要求专家数不超过
    ``BLOCK_SIZE_N``、``hidden_dim`` 可被 ``BLOCK_SIZE_K`` 整除且 ``topk <= 2``。

    输入输出示例：``x=[1024,4096]``、``router_weight=[64,4096]``、tile 为
    ``[32,64,256]`` 且 ``topk=2`` 时，返回权重和专家 ID，形状均为 ``[1024,2]``。
    """
    assert len(x.shape) == 2 and x.shape[1] == router_weight.shape[1]
    bs, hidden_dim = x.shape
    num_experts = router_weight.shape[0]

    assert num_experts <= BLOCK_SIZE_N
    assert hidden_dim % BLOCK_SIZE_K == 0
    assert topk <= 2

    topk_weights = torch.empty((bs, topk), dtype=torch.float32, device=x.device)
    topk_ids = torch.empty((bs, topk), dtype=torch.int32, device=x.device)
    is_correction_bias = correction_bias is not None

    grid = (triton.cdiv(bs, BLOCK_SIZE_M) * triton.cdiv(num_experts, BLOCK_SIZE_N),)

    # TODO(ch-wan)：DP Attention 的临时兼容逻辑；后续应让 router 支持 mask，
    # 从而直接跳过用于对齐不同 DP rank token 数量的 padding token。
    from sglang.srt.layers.dp_attention import is_dp_attention_enabled

    dp_attn_workaround_flag = is_dp_attention_enabled()

    fused_moe_router_tensorcore_kernel[grid](
        a_ptr=x,
        b_ptr=router_weight,
        topk_weights_ptr=topk_weights,
        topk_ids_ptr=topk_ids,
        bs=bs,
        num_experts=num_experts,
        topk=topk,
        moe_softcapping=moe_softcapping,
        moe_renormalize=False,
        K=hidden_dim,
        correction_bias_ptr=correction_bias,
        is_correction_bias=is_correction_bias,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        stride_am=hidden_dim,
        stride_bn=hidden_dim,
        dp_attn_workaround_flag=dp_attn_workaround_flag,
    )

    return topk_weights, topk_ids


def fused_moe_router_shim(
    moe_softcapping,
    hidden_states,
    gating_output,
    topk,
    renormalize,
    correction_bias: Optional[torch.Tensor] = None,
    enable_deterministic_inference: bool = False,
):
    """根据输入规模和确定性要求选择融合 router 实现。

    参数：
        moe_softcapping: router logits 的 softcap 上限，0 表示禁用。
        hidden_states: token hidden states，形状 ``[bs, hidden_dim]``。
        gating_output: router 权重，形状 ``[num_experts, hidden_dim]``；虽然沿用
            ``gating_output`` 名称，这里传入的并不是已计算出的 logits。
        topk: 每个 token 选择的专家数。
        renormalize: 是否在入选 Top-K 内重新归一化；当前必须为 ``False``。
        correction_bias: 可选的逐专家校正偏置，形状 ``[num_experts]``。
        enable_deterministic_inference: 为 ``True`` 时固定使用单 token CUDA Core
            kernel，避免不同 GEMM 执行路径带来的非确定性。

    输入输出示例：``hidden_states=[128,4096]``、``gating_output=[8,4096]``、
    ``topk=2`` 时，返回 ``(topk_weights, topk_ids)``，两者形状均为 ``[128,2]``。
    """
    assert not renormalize
    assert (
        len(hidden_states.shape) == 2
        and hidden_states.shape[1] == gating_output.shape[1]
    )
    bs, hidden_dim = hidden_states.shape
    num_experts = gating_output.shape[0]

    BLOCK_SIZE_M = 32

    BLOCK_SIZE_N = max(num_experts, 16)
    # 专家数较多时缩小 K tile，避免单个 program 的共享内存占用过大而 OOM。
    BLOCK_SIZE_K = 256 if num_experts < 256 else 64

    if (
        (bs >= 512 or num_experts > 8)
        and hidden_dim % BLOCK_SIZE_K == 0
        # 确定性推理固定使用一个 token 一个 program 的 CUDA Core 路径。
        and not enable_deterministic_inference
    ):
        # batch 或专家数较大时，使用 Tensor Core 加速分块矩阵乘。
        return fused_moe_router_tensorcore(
            x=hidden_states,
            router_weight=gating_output,
            topk=topk,
            moe_softcapping=moe_softcapping,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            correction_bias=correction_bias,
        )
    else:
        # 小规模输入或确定性推理使用不依赖 Tensor Core GEMM 的 CUDA Core kernel。
        return fused_moe_router_cudacore(
            x=hidden_states,
            router_weight=gating_output,
            topk=topk,
            moe_softcapping=moe_softcapping,
            correction_bias=correction_bias,
        )


@triton.jit
def router_gate_matvec_kernel(
    x_ptr,  # (M, K) bf16/fp16/fp32, row-major
    w_ptr,  # (E, K) fp32/bf16/fp16, k-major
    out_ptr,  # (M, E) fp32
    K,
    E,
    stride_xm,
    stride_we,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):
    """Router-gate logits as a single matvec launch, fp32 accumulation for
    any float weight dtype. BLOCK_K covers the whole K in one masked load
    (single iteration for K <= BLOCK_K): a cold gate weight then costs one
    HBM round trip per CTA instead of a serial dependent-load chain, which
    is what dominates in the real model where ~94MB/layer of expert traffic
    flushes L2 between gate calls.
    """
    pid_m = tl.program_id(0)
    pid_e = tl.program_id(1)
    e_offs = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = e_offs < E

    # First K tile, weight load ahead of the PDL wait (see docstring).
    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < K
    w = tl.load(
        w_ptr + e_offs[:, None] * stride_we + k_offs[None, :],
        mask=e_mask[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    x = tl.load(x_ptr + pid_m * stride_xm + k_offs, mask=k_mask, other=0.0).to(
        tl.float32
    )
    acc = tl.sum(w * x[None, :], axis=1)

    for k0 in range(BLOCK_K, K, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        x = tl.load(x_ptr + pid_m * stride_xm + k_offs, mask=k_mask, other=0.0).to(
            tl.float32
        )
        w = tl.load(
            w_ptr + e_offs[:, None] * stride_we + k_offs[None, :],
            mask=e_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)

    tl.store(
        out_ptr + pid_m * E + e_offs, acc.to(out_ptr.dtype.element_ty), mask=e_mask
    )


# Cold-cache tuned on H20-3e (41 rotating gate weights so each call misses L2,
# like the real model); expected to carry to B200 (more SMs favor the wide
# grid even more) — re-tune with benchmark/kernels/bench_router_gate_matvec.py.
ROUTER_GATE_MATVEC_BLOCK_E = 4
ROUTER_GATE_MATVEC_NUM_WARPS = 8
# Beyond this M the per-M re-reads of the gate weight outgrow the library
# GEMM (cold H20-3e: bf16 wins to M=12, fp32 to M=8; cap at the lower).
ROUTER_GATE_MATVEC_MAX_M = 8


def router_gate_matvec(
    hidden_states: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Small-M router-gate logits: one triton launch replacing the library
    path — for fp32 gate weights the eager upcast + fp32 GEMM + splitKreduce
    triple, for bf16 the F.linear GEMV. Returns fp32 (M, E) logits with fp32
    accumulation (deterministic; for fp32 weights 0 top-8 routing flips over
    30104 random draws vs the fp32 reference; for bf16 weights this is
    slightly MORE precise than the library GEMV, so near-tie logits can
    round-trip differently — same order as the bf16-vs-fp32 gate change).

    Cold-cache (41 rotating weights, in-graph, H20-3e, E=513, K=2560), us/call:

        M      lib bf16   matvec bf16   lib fp32 chain   matvec fp32
        1        4.2         4.3            6.4              6.2
        2       13.8         4.5           14.6              6.4
        4       14.0         5.4           20.7             10.4
        8       14.6        10.4           20.1             18.1
        16      14.6        18.9  (lib)    20.9             30.9  (lib)

    Callers must gate on M <= ROUTER_GATE_MATVEC_MAX_M; prefill-sized M
    keeps the library GEMM."""
    assert (
        weight.dtype
        in (
            torch.float32,
            torch.bfloat16,
            torch.float16,
        )
        and weight.is_contiguous()
    )
    M, K = hidden_states.shape
    E = weight.shape[0]
    out = torch.empty((M, E), dtype=torch.float32, device=hidden_states.device)
    block_e = ROUTER_GATE_MATVEC_BLOCK_E
    # Single k-iteration whenever K fits one block: no serial dependent-load
    # chain on a cold weight.
    block_k = min(4096, triton.next_power_of_2(K))
    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}
    router_gate_matvec_kernel[(M, triton.cdiv(E, block_e))](
        hidden_states,
        weight,
        out,
        K,
        E,
        hidden_states.stride(0),
        weight.stride(0),
        BLOCK_E=block_e,
        BLOCK_K=block_k,
        num_warps=ROUTER_GATE_MATVEC_NUM_WARPS,
        **pdl_kwargs,
    )
    return out
