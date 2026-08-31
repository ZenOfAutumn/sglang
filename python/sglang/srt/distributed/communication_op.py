# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 改编自 https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/communication_op.py

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributed

from .parallel_state import (
    get_attn_tp_group,
    get_moe_ep_group,
    get_moe_tp_group,
    get_tp_group,
)


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """在张量模型并行（TP）组内对输入张量执行 All-Reduce。

    每个 TP rank 提供一个形状相同的张量；通信后，各 rank 都得到所有输入
    按元素归约后的结果。具体归约操作及通信后端由 TP 组协调器负责。
    """
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_quant_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """在张量模型并行（TP）组内执行量化 All-Reduce。

    与普通 All-Reduce 语义一致，但允许底层通信实现通过量化降低通信数据量；
    具体量化格式、精度处理和通信后端由 TP 组协调器决定。
    """
    return get_tp_group().quant_all_reduce(input_)


def tensor_model_parallel_fused_allreduce_rmsnorm(
    input_: torch.Tensor,
    residual_inp_: torch.Tensor,
    weight_: torch.Tensor,
    eps: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """融合执行 TP All-Reduce 与 RMSNorm。

    ``GroupCoordinator`` 负责选择执行策略和后端：它可能调用通信器原生的
    融合接口或自定义融合内核；如果当前环境不支持融合实现，则返回 ``None``，
    由调用方执行通用的回退路径。

    Args:
        input_: 需要先在 TP 组内执行 All-Reduce 的输入张量。
        residual_inp_: 与归约结果相加的残差张量。
        weight_: RMSNorm 的缩放权重。
        eps: RMSNorm 为避免除零而使用的数值稳定项。

    Returns:
        融合后端可用时，返回规范化结果和更新后的残差组成的二元组；否则返回
        ``None``。
    """
    return get_tp_group().fused_allreduce_rmsnorm(input_, residual_inp_, weight_, eps)


def tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_group(
    input_: torch.Tensor,
    residual_inp_: torch.Tensor,
    weight_: torch.Tensor,
    eps: float,
    group_size: int = 128,
    emit_bf16: bool = False,
    transpose_scale: bool = False,
) -> Optional[Tuple[torch.Tensor, ...]]:
    """Fused TP all-reduce + RMSNorm + per-group FP8 quant (ROCm/aiter).

    Returns ``(fp8_output, residual_out, per_group_scale)`` by default, or
    ``(fp8_output, residual_out, per_group_scale, bf16_output)`` when
    ``emit_bf16=True`` (kernel writes both fp8 and the pre-quantization bf16
    normed output — no extra kernel). ``None`` when the backend cannot
    service the request (non-AMD, custom AR disabled, shape unsupported).
    Callers MUST handle ``None`` by falling back to the separate
    fused-AR-RMSNorm + per-group-quant path.

    ``transpose_scale=True`` asks the kernel to emit the per-group scale in the
    column-major layout the gfx95 bpreshuffle GEMM consumes, so callers can skip
    the post-kernel ``materialize_bpreshuffle_fp8_scale`` transpose.
    """
    return get_tp_group().fused_allreduce_rmsnorm_quant_per_group(
        input_,
        residual_inp_,
        weight_,
        eps,
        group_size,
        emit_bf16=emit_bf16,
        transpose_scale=transpose_scale,
    )


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """在张量模型并行（TP）组内对输入张量执行 All-Gather。

    收集每个 TP rank 的输入，并沿 ``dim`` 维拼接；完整结果会返回到组内的
    每个 rank。``dim`` 默认为最后一个维度。
    """
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> Optional[torch.Tensor]:
    """在张量模型并行（TP）组内对输入张量执行 Gather。

    收集每个 TP rank 的输入并沿 ``dim`` 维拼接，但只在目标 rank ``dst``
    上返回完整张量；非目标 rank 通常返回 ``None``。
    """
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None, src: int = 0
):
    """从源 rank 向张量模型并行（TP）组广播张量字典。

    ``src`` rank 提供待广播的字典，组内其他 rank 接收相同的数据。若
    ``torch.distributed`` 尚未初始化，则不进行通信并原样返回输入，以兼容
    单进程执行场景。
    """
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)


def attention_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """在注意力张量并行组内对输入张量执行 All-Reduce。

    各注意力并行 rank 最终都会得到所有输入按元素归约后的结果。
    """
    return get_attn_tp_group().all_reduce(input_)


def attention_tensor_model_parallel_quant_all_reduce(
    input_: torch.Tensor,
) -> torch.Tensor:
    """在注意力张量并行组内执行量化 All-Reduce。

    底层实现可通过量化减少注意力并行 rank 之间的通信数据量。
    """
    return get_attn_tp_group().quant_all_reduce(input_)


def moe_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """在 MoE 张量并行组内对输入张量执行 All-Reduce。

    该通信组覆盖同一 MoE 张量并行分片集合，各 rank 最终获得按元素归约后的
    完整结果。
    """
    return get_moe_tp_group().all_reduce(input_)


def moe_expert_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """在 MoE 专家并行（EP）组内对输入张量执行 All-Reduce。

    各 EP rank 的输入会按元素归约，并将归约结果返回给组内所有 rank。
    """
    return get_moe_ep_group().all_reduce(input_)
