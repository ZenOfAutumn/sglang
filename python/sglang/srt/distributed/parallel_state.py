# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/parallel_state.py

# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
"""分布式状态管理。
本模块从 PyTorch 手中接管分布式环境的控制权。典型使用流程如下：

- 调用 `init_distributed_environment` 初始化分布式环境；
- 调用 `initialize_model_parallel` 或 `ensure_model_parallel_initialized`
  初始化各类模型并行进程组；

- 中间可以是任何涉及分布式通信的代码；

- 调用 `destroy_model_parallel` 销毁模型并行进程组；
- 调用 `destroy_distributed_environment` 销毁分布式环境。

如果只需要使用分布式环境而不需要模型/流水线并行，可以跳过模型并行的
初始化与销毁步骤。
"""

import contextlib
import gc
import logging
import os
import pickle
import weakref
from collections import namedtuple
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing import shared_memory
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import patch

import torch
import torch.distributed
from torch.distributed import Backend, ProcessGroup

from sglang.srt import platforms
from sglang.srt.compilation.compilation_config import register_split_op
from sglang.srt.distributed.utils import set_global_tcp_store
from sglang.srt.environ import envs
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.platforms.device_mixin import _DEVICE_TO_DISTRIBUTED_BACKEND
from sglang.srt.utils import (
    get_current_device_stream_fast,
    get_int_env_var,
    is_cpu,
    is_cuda_alike,
    is_hip,
    is_musa,
    is_npu,
    is_shm_available,
    is_xpu,
)
from sglang.srt.utils.custom_op import register_custom_op
from sglang.srt.utils.network import get_local_ip_auto

_is_npu = is_npu()
_is_cpu = is_cpu()
_is_xpu = is_xpu()
_is_musa = is_musa()

TensorMetadata = namedtuple("TensorMetadata", ["device", "dtype", "size"])

# 使用 int 值而非 ReduceOp.SUM，以便兼容 torch.compile。
REDUCE_OP_SUM = int(torch.distributed.ReduceOp.SUM)

# 复用用户提供的分布式超时来创建模型并行子组，避免运行时集合通信静默退回后端默认超时。
_MODEL_PARALLEL_GROUP_TIMEOUT: Optional[timedelta] = None


def get_torch_distributed_pg_options(group_name=None):
    """构造 NPU/HCCL 进程组选项，其他设备返回 ``None``。

    输入输出示例：
        - CUDA 环境：``get_torch_distributed_pg_options("tp") -> None``；
        - NPU 环境且组名为 ``"moe_ep"``：返回配置了 HCCL 缓冲区大小的 Options；
        - NPU 环境但组名为 ``"tp"``：返回 ``None``，沿用 PyTorch 默认选项。
    """
    if not _is_npu:
        return None

    # 仅为默认组或 MoE 相关组创建 HCCL 选项。
    if group_name is not None and "moe" not in group_name:
        return None

    import torch_npu

    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    hccl_buffer_size = int(
        os.environ.get("DEEPEP_HCCL_BUFFSIZE") or os.environ.get("HCCL_BUFFSIZE") or 200
    )
    options.hccl_config = {"hccl_buffer_size": hccl_buffer_size}
    return options


@dataclass
class GraphCaptureContext:
    stream: torch.get_device_module().Stream


@dataclass
class P2PWork:
    work: Optional[torch.distributed.Work]
    payload: Optional[torch.Tensor]


def _split_tensor_dict(
    tensor_dict: Dict[str, Union[torch.Tensor, Any]],
) -> Tuple[List[Tuple[str, Any]], List[torch.Tensor]]:
    """把张量字典拆分成两部分：
    1. 一个 (key, value) 列表：若 value 是张量，则替换为它的元数据（TensorMetadata）；
    2. 一个张量列表。

    这样做便于在通信时先广播/发送元数据（走 CPU 组），再单独传输张量本体（走 device 组）。

    输入示例：
        {"hidden": torch.randn(2, 4, device="cuda:0"), "step": 3}
    输出示例：
        metadata_list = [("hidden", TensorMetadata("cuda", torch.float32, (2, 4))),
                         ("step", 3)]
        tensor_list   = [<那个 (2,4) 的 cuda 张量>]
    """
    metadata_list: List[Tuple[str, Any]] = []
    tensor_list: List[torch.Tensor] = []
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            # 注意：这里不能直接用 `value.device`，因为它不仅含设备类型还含设备索引
            # （例如 "cuda:0"）。我们只需要设备类型，接收端会自行设置设备索引。
            device = value.device.type
            metadata_list.append(
                (key, TensorMetadata(device, value.dtype, value.size()))
            )
            tensor_list.append(value)
        else:
            metadata_list.append((key, value))
    return metadata_list, tensor_list


_group_name_counter: Dict[str, int] = {}


def _get_unique_name(name: str) -> str:
    """为进程组生成一个唯一名字（同名调用会自增后缀）。

    输入/输出示例：
        _get_unique_name("tp") -> "tp:0"
        _get_unique_name("tp") -> "tp:1"
    """
    if name not in _group_name_counter:
        _group_name_counter[name] = 0
    newname = f"{name}:{_group_name_counter[name]}"
    _group_name_counter[name] += 1
    return newname


_groups: Dict[str, Callable[[], Optional["GroupCoordinator"]]] = {}


def _register_group(group: "GroupCoordinator") -> None:
    """用弱引用注册通信组，供只接受字符串参数的自定义算子反查。

    输入输出示例：
        输入 ``group.unique_name == "tp:0"`` 后，``_groups["tp:0"]()`` 返回该组；
        本函数无返回值，组销毁后弱引用会返回 ``None``。
    """
    _groups[group.unique_name] = weakref.ref(group)


@register_custom_op(mutates_args=["tensor"])
@register_split_op()
def inplace_all_reduce(tensor: torch.Tensor, group_name: str) -> None:
    """按组名执行原地 all-reduce，供 ``torch.compile`` 可追踪的自定义算子调用。

    输入输出示例（两 rank、SUM）：rank 0 输入 ``[1, 2]``，rank 1 输入 ``[3, 4]``；
    调用后两个 rank 的原张量都变为 ``[4, 6]``，函数返回 ``None``。
    """
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    group._all_reduce_in_place(tensor)


@register_custom_op(out_shape="tensor")
def outplace_all_reduce(
    tensor: torch.Tensor, group_name: str, outplace_all_reduce_method: str
) -> torch.Tensor:
    """按组名执行非原地 all-reduce，返回新张量并保留输入张量不变。

    输入输出示例（两 rank、SUM）：rank 0 输入 ``[1]``，rank 1 输入 ``[2]``；
    两边均返回新张量 ``[3]``，原输入仍分别为 ``[1]`` 和 ``[2]``。
    """
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    return group._all_reduce_out_place(tensor, outplace_all_reduce_method)


@register_custom_op(mutates_args=["output"])
def reg_all_gather_into_tensor(
    output: torch.Tensor, input: torch.Tensor, group_name: str
) -> None:
    """把各 rank 的输入沿第 0 维收集到预分配的 ``output``。

    输入输出示例（两 rank）：输入分别为 ``[[1, 2]]``、``[[3, 4]]``，
    每个 rank 的 ``output`` 最终都为 ``[[1, 2], [3, 4]]``；函数无返回值。
    """
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    group._all_gather_into_tensor(output, input)


@register_custom_op(mutates_args=["output"])
def reg_reduce_scatter_tensor(
    output: torch.Tensor, input: torch.Tensor, group_name: str
) -> None:
    """对输入先逐元素归约，再把第 0 维分片写入各 rank 的 ``output``。

    输入输出示例（两 rank、SUM）：rank 0 输入 ``[1, 2, 3, 4]``，rank 1 输入
    ``[10, 20, 30, 40]``；rank 0 输出 ``[11, 22]``，rank 1 输出 ``[33, 44]``。
    """
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    group._reduce_scatter_tensor(output, input)


@register_custom_op(mutates_args=["output"])
def reg_all_to_all_single(
    output: torch.Tensor, input: torch.Tensor, group_name: str
) -> None:
    """执行等长 all-to-all，把每个 rank 的输入均分后交换给所有 rank。

    输入输出示例（两 rank）：rank 0 输入 ``[0, 1, 2, 3]``，rank 1 输入
    ``[10, 11, 12, 13]``；rank 0 输出 ``[0, 1, 10, 11]``，rank 1 输出
    ``[2, 3, 12, 13]``。
    """
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    group._all_to_all_single(output, input)


class GroupCoordinator:
    """
    针对一组进程的 PyTorch ProcessGroup 封装。

    PyTorch 的 ProcessGroup 只能绑定到一种具体的通信后端
        （例如 NCCL、Gloo、MPI 等）。
    GroupCoordinator 负责该组内进程之间的所有通信操作，它可以把一次通信
        路由到某个具体实现上（例如根据张量大小以及是否处于 CUDA Graph 模式
        来切换 allreduce 的实现）。

    设计要点：
    1. 每个并行维度（TP / PP / MoE-EP / MoE-TP / ATTN-TP / ATTN-CP 等）都会
       各自创建一个 GroupCoordinator 实例；
    2. 每个实例同时持有两个 ProcessGroup：`device_group`（NCCL/HCCL 等，走
       device 通信）和 `cpu_group`（gloo，走 CPU，用于对象/元数据同步）；
    3. 实例会按需创建多种加速通信器（pynccl / custom all-reduce / quick
       all-reduce / mscclpp / torch symm-mem / 共享内存广播队列），并在运行时
       根据条件择优选择。
    """

    # 可用属性说明：
    rank: int  # 当前进程的全局 rank
    ranks: List[int]  # 本组内所有进程的全局 rank 列表
    world_size: int  # 本组的进程数
    # `local_rank` 与 `rank_in_group` 的区别：
    # 假设有一个跨两个节点、大小为 4 的通信组：
    # 进程     | 节点 | 全局 Rank | 节点内 Local Rank | 组内 Rank
    #   0     |   0  |  0        |     0            |       0
    #   1     |   0  |  1        |     1            |       1
    #   2     |   1  |  2        |     0            |       2
    #   3     |   1  |  3        |     1            |       3
    local_rank: int  # 节点内的本地 rank，用于绑定具体设备（GPU 编号）
    rank_in_group: int  # 在本通信组内部的 rank
    cpu_group: ProcessGroup  # 用于 CPU 侧通信的进程组（gloo 后端）
    device_group: ProcessGroup  # 用于设备侧通信的进程组（NCCL/HCCL 等后端）
    use_pynccl: bool  # 是否倾向于使用 PyNccl 的提示位
    use_pymscclpp: bool  # 是否倾向于使用 PyMscclpp 的提示位
    use_custom_allreduce: bool  # 是否倾向于使用 CustomAllreduce 的提示位
    use_torch_symm_mem_all_reduce: (
        bool  # 是否倾向于使用 TorchSymmMemAllReduce 的提示位
    )
    use_message_queue_broadcaster: (
        bool  # 是否倾向于使用消息队列广播器（共享内存）的提示位
    )
    # 只有当 world size > 1 时才会创建下面这些通信器
    pynccl_comm: Optional[Any]  # PyNccl 通信器
    ca_comm: Optional[Any]  # Custom allreduce（自定义 all-reduce）通信器
    torch_symm_mem_comm: Optional[Any]  # Torch 对称内存通信器
    mq_broadcaster: Optional[Any]  # 基于共享内存的广播器

    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
        use_pynccl: bool,
        use_pymscclpp: bool,
        use_custom_allreduce: bool,
        use_torch_symm_mem_all_reduce: bool,
        use_hpu_communicator: bool,
        use_xpu_communicator: bool,
        use_npu_communicator: bool,
        use_message_queue_broadcaster: bool = False,
        group_name: Optional[str] = None,
        gloo_timeout: timedelta = timedelta(seconds=120 * 60),
        recovered_rank: bool = False,
    ):
        """
        Args:
            group_ranks: 全局范围内所有同类通信组的 rank 划分，例如 TP=2、world=4 时
                为 [[0, 1], [2, 3]]。每个 rank 都需要遍历并创建全部子组（集体操作
                要求所有进程同步调用 new_group），但只保留自己所在的那一个。
            local_rank: 本节点内的 rank，用于选择设备。
            torch_distributed_backend: torch.distributed 后端名（nccl/hccl/mooncake 等）。
            use_pynccl / use_pymscclpp / use_custom_allreduce /
            use_torch_symm_mem_all_reduce: 各种加速通信实现的开关。
            use_hpu_communicator / use_xpu_communicator / use_npu_communicator:
                非 CUDA 硬件后端的专用通信器开关。
            use_message_queue_broadcaster: 是否启用共享内存消息队列广播（用于
                高频小对象广播，比 gloo broadcast_object 快很多）。
            group_name: 组名（tp/pp/moe_ep ...），会被加上递增后缀变成全局唯一名。
            gloo_timeout: CPU（gloo）组的超时时长。
            recovered_rank: 弹性 EP 场景下表示本 rank 是“恢复恢入”的，需要走特殊路径。
        """
        # 设置组信息：生成全局唯一名并注册到全局表 _groups，
        # 以便 torch 自定义算子（只能传字符串）能通过名字反查到本对象。
        group_name = group_name or "anonymous"
        self.unique_name = _get_unique_name(group_name)
        _register_group(self)

        # 设置 rank 信息
        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.device_group = None
        self.cpu_group = None
        # 单机内的进程数，用于判断 CPU 共享内存集体通信（shm）是否可用
        self.local_size = get_int_env_var("LOCAL_SIZE", 0)

        # 根据硬件平台确定本 rank 绑定的设备
        if is_cuda_alike():
            device_id = (
                0 if envs.SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS.get() else local_rank
            )
            self.device = torch.device(f"cuda:{device_id}")
        elif _is_npu:
            self.device = torch.device(f"npu:{local_rank}")
        elif _is_xpu:
            self.device = torch.device(f"xpu:{local_rank}")
        elif _is_musa:
            self.device = torch.device(f"musa:{local_rank}")
        else:
            self.device = torch.device("cpu")
        self.device_module = torch.get_device_module(self.device)

        # 遍历所有子组划分：torch.distributed.new_group 是集体调用，必须由所有进程
        # 以相同顺序调用；每个进程只保留自己所在的那个子组句柄。
        for ranks in group_ranks:
            # active_ranks 用于弹性 EP（mooncake 后端）标记哪些 rank 当前存活
            active_ranks = torch.ones(len(ranks), dtype=torch.int32, device=self.device)
            active_ranks_cpu = torch.ones(len(ranks), dtype=torch.int32)
            # 复用用户传入的全局超时，避免子组静默地退回后端默认值
            subgroup_timeout = _MODEL_PARALLEL_GROUP_TIMEOUT
            if "mooncake" in torch_distributed_backend:
                from mooncake.ep import MooncakeBackendOptions

                device_group = torch.distributed.new_group(
                    ranks,
                    backend="mooncake",
                    pg_options=MooncakeBackendOptions(active_ranks, recovered_rank),
                    timeout=subgroup_timeout,
                )
                cpu_group = torch.distributed.new_group(
                    ranks,
                    backend="mooncake-cpu",
                    pg_options=MooncakeBackendOptions(active_ranks_cpu, recovered_rank),
                    timeout=subgroup_timeout,
                )
            else:
                pg_options = get_torch_distributed_pg_options(group_name)
                device_group = torch.distributed.new_group(
                    ranks,
                    backend=torch_distributed_backend,
                    pg_options=pg_options,
                    timeout=subgroup_timeout,
                )
                # 额外再建一个 `gloo` 后端的组，使得进程之间可以直接通过 CPU 协同
                # （例如传递 Python 对象、barrier），避免占用 device 流与显存。
                cpu_group = torch.distributed.new_group(
                    ranks, backend="gloo", timeout=gloo_timeout
                )
            # 只有当前 rank 属于这个子组时，才把它记录为自己的通信组
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self.device_group = device_group
                self.cpu_group = cpu_group
                self.active_ranks = active_ranks
                self.active_ranks_cpu = active_ranks_cpu

        assert self.cpu_group is not None
        assert self.device_group is not None

        # 保存各类通信实现的开关
        self.use_pynccl = use_pynccl
        self.use_pymscclpp = use_pymscclpp
        self.use_custom_allreduce = use_custom_allreduce
        self.use_torch_symm_mem_all_reduce = use_torch_symm_mem_all_reduce
        self.use_hpu_communicator = use_hpu_communicator
        self.use_xpu_communicator = use_xpu_communicator
        self.use_npu_communicator = use_npu_communicator
        self.use_message_queue_broadcaster = use_message_queue_broadcaster

        # 延迟导入，避免文档构建（无 GPU 环境）时导入失败
        from sglang.srt.distributed.device_communicators.custom_all_reduce import (
            dispatch_custom_allreduce,
        )
        from sglang.srt.distributed.device_communicators.pymscclpp import (
            PyMscclppCommunicator,
        )
        from sglang.srt.distributed.device_communicators.pynccl import (
            PyNcclCommunicator,
        )
        from sglang.srt.distributed.device_communicators.pynccl_allocator import (
            debug_check_symmetric_mempool,
            is_symmetric_memory_enabled,
            use_symmetric_memory,
        )
        from sglang.srt.distributed.device_communicators.torch_symm_mem import (
            TorchSymmMemCommunicator,
        )
        from sglang.srt.layers.dp_attention import is_allocation_symmetric

        self.is_symmetric_memory_enabled = is_symmetric_memory_enabled
        self.use_symmetric_memory = use_symmetric_memory
        self.is_allocation_symmetric = is_allocation_symmetric
        self.debug_check_symmetric_mempool = debug_check_symmetric_mempool
        # ROCm 上额外提供 QuickAllReduce 实现
        if is_hip():
            from sglang.srt.distributed.device_communicators.quick_all_reduce import (
                QuickAllReduce,
                qr_rocm_arch_available,
            )

        self.pynccl_comm: Optional[PyNcclCommunicator] = None
        if use_pynccl and self.world_size > 1:
            self.pynccl_comm = PyNcclCommunicator(
                group=self.cpu_group,
                device=self.device,
            )

        self.pymscclpp_comm: Optional[PyMscclppCommunicator] = None
        if use_pymscclpp and self.world_size > 1:
            self.pymscclpp_comm = PyMscclppCommunicator(
                group=self.cpu_group,
                device=self.device,
            )

        self.ca_comm: Optional[Any] = None
        self.qr_comm: Optional[QuickAllReduce] = None
        if use_custom_allreduce and self.world_size > 1:
            # 初始化自定义的快速 all-reduce 实现（小张量下比 NCCL 延迟更低）。
            try:
                CAClass = dispatch_custom_allreduce(
                    group=self.cpu_group,
                    device=self.device,
                )
                self.ca_comm = CAClass(
                    group=self.cpu_group,
                    device=self.device,
                )
            except Exception as e:
                logger.warning(
                    f"Setup Custom allreduce failed with {e}. To silence this "
                    "warning, specify --disable-custom-all-reduce explicitly."
                )

            if is_hip():
                try:
                    # 在 AMD（rocm >= gfx942）上初始化 quick all-reduce 实现。
                    # quick reduce 是对 custom allreduce 的补充（适用于不同尺寸区间）。
                    # 基于 quickreduce (https://github.com/mk1-project/quickreduce)。
                    if qr_rocm_arch_available():
                        self.qr_comm = QuickAllReduce(
                            group=self.cpu_group, device=self.device
                        )
                except Exception as e:
                    logger.warning(f"Failed to initialize QuickAllReduce: {e}")
        elif self.world_size > 1 and is_hip():
            logger.info("[AR] All-reduce call path: NCCL (custom AR disabled)")

        self.torch_symm_mem_comm: Optional[TorchSymmMemCommunicator] = None
        if self.use_torch_symm_mem_all_reduce and self.world_size > 1:
            self.torch_symm_mem_comm = TorchSymmMemCommunicator(
                group=self.cpu_group,
                device=self.device,
            )

        # 为其他硬件后端（HPU / XPU / NPU）创建专用通信器
        from sglang.srt.distributed.device_communicators.hpu_communicator import (
            HpuCommunicator,
        )
        from sglang.srt.distributed.device_communicators.npu_communicator import (
            NpuCommunicator,
        )
        from sglang.srt.distributed.device_communicators.xpu_communicator import (
            XpuCommunicator,
        )

        self.hpu_communicator: Optional[HpuCommunicator] = None
        if use_hpu_communicator and self.world_size > 1:
            self.hpu_communicator = HpuCommunicator(group=self.device_group)

        self.xpu_communicator: Optional[XpuCommunicator] = None
        if use_xpu_communicator and self.world_size > 1:
            self.xpu_communicator = XpuCommunicator(group=self.device_group)

        self.npu_communicator: Optional[NpuCommunicator] = None
        if use_npu_communicator and self.world_size > 1:
            self.npu_communicator = NpuCommunicator(group=self.device_group)

        # 创建基于共享内存的消息队列（用于 rank0 向其他 rank 高频广播 Python 对象）
        from sglang.srt.distributed.device_communicators.shm_broadcast import (
            MessageQueue,
        )

        self.mq_broadcaster: Optional[MessageQueue] = None
        if use_message_queue_broadcaster and self.world_size > 1 and not recovered_rank:
            # 恢复恢入的 rank 在 elastic_ep.py 里自行创建 mq_broadcaster
            # 1 << 22 为单个缓冲区字节数（4MB），6 为缓冲区个数
            self.mq_broadcaster = MessageQueue.create_from_process_group(
                self.cpu_group, 1 << 22, 6
            )

    def __repr__(self):
        return (
            f"ranks={self.ranks} rank={self.rank} local_rank={self.local_rank} use_pynccl={self.use_pynccl} "
            f"device_group={self.device_group} cpu_group={self.cpu_group} unique_name={self.unique_name} "
            f"world_size={self.world_size} rank_in_group={self.rank_in_group}"
        )

    @property
    def first_rank(self):
        """返回本组内第一个进程的全局 rank"""
        return self.ranks[0]

    @property
    def last_rank(self):
        """返回本组内最后一个进程的全局 rank"""
        return self.ranks[-1]

    @property
    def is_first_rank(self):
        """返回调用方是否为本组内的第一个进程"""
        return self.rank == self.first_rank

    @property
    def is_last_rank(self):
        """返回调用方是否为本组内的最后一个进程"""
        return self.rank == self.last_rank

    @property
    def next_rank(self):
        """返回调用方在环上的后继进程的全局 rank（循环）"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(rank_in_group + 1) % world_size]

    @property
    def prev_rank(self):
        """返回调用方在环上的前驱进程的全局 rank（循环）"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(rank_in_group - 1) % world_size]

    @contextmanager
    def graph_capture(
        self,
        graph_capture_context: Optional[GraphCaptureContext] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ):
        """
        CUDA Graph 捕获期间使用的上下文管理器。

        作用：把捕获切换到专用 stream 上，并将各个通信器切到“可被图捕获”的状态。
        """
        if graph_capture_context is None:
            if stream is None:
                stream = self.device_module.Stream()
            graph_capture_context = GraphCaptureContext(stream)
        else:
            stream = graph_capture_context.stream
        # custom quick allreduce 不需要额外的上下文，因为 IPC 句柄已经在 init() 中
        # 收集完毕，可以直接被图捕获。
        ca_comm = self.ca_comm
        maybe_ca_context = nullcontext() if ca_comm is None else ca_comm.capture()

        # 在另一个 stream 上捕获图之前，确保所有初始化操作已经完成
        curr_stream = get_current_device_stream_fast()
        if curr_stream != stream:
            stream.wait_stream(curr_stream)

        with self.device_module.stream(stream), maybe_ca_context:
            # 在 graph 模式下，对集体通信操作必须非常小心。当前的支持情况如下：
            #     allreduce \ 模式  | Eager（即时）|  Graph  |
            # --------------------------------------------
            # quick allreduce        |   启用     |  启用  |
            # custom allreduce       |   启用     |  启用  |
            # PyNccl                 |   禁用     |  启用  |
            # PyMscclpp              |   禁用     |  启用  |
            # TorchSymmMem           |   禁用     |  启用  |
            # torch.distributed      |   启用     |  禁用  |
            #
            # 注：开启 custom quick allreduce 时会做一次运行时检查，如果张量太小，
            #  会自动回退到下一个可用方案。
            # 注：custom allreduce 也有运行时检查，如果张量太大，会回退到下一个
            #  可用方案。
            # 注：PyMscclpp 需要提前注册张量，在 eager 模式下会引入很大开销，
            #  因此只在 graph 模式下支持。
            # 总结：我们根据上表中的算法优先级及各自的使用条件，为每种模式选择
            #  合适的 allreduce 实现。
            pynccl_comm = self.pynccl_comm
            maybe_pynccl_context: Any
            if not pynccl_comm:
                maybe_pynccl_context = nullcontext()
            else:
                maybe_pynccl_context = pynccl_comm.change_state(enable=True)

            pymscclpp_comm = self.pymscclpp_comm
            maybe_pymscclpp_context: Any
            if not pymscclpp_comm:
                maybe_pymscclpp_context = nullcontext()
            else:
                maybe_pymscclpp_context = pymscclpp_comm.change_state(enable=True)
            with maybe_pynccl_context, maybe_pymscclpp_context:
                yield graph_capture_context

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """
        面向用户的 all-reduce 入口，在真正执行 all-reduce 之前做一层分发。

        语义：组内所有 rank 各自提供一个相同形状的 `input_`，输出是它们的逐元素
        求和（SUM），且每个 rank 都得到完整结果。在 SGLang 中的典型使用场景：
        张量并行（TP）中 RowParallelLinear 的输出聚合、Attention 输出投影之后。

        之所以需要这层，是因为 Dynamo 不支持向自定义算子传入任意对象（这里就是
         `self`）。我们只能把组名以字符串传入，然后在算子内部根据组名反查到
         GroupCoordinator，再把 all-reduce 分发给它。

        另外，PyTorch 自定义算子不允许在同一个算子里既原地修改又返回新张量，
        所以必须提前判断本次操作是原地（in-place）还是非原地（out-of-place）：
          - 原地：调用 `inplace_all_reduce`，结果写回 `input_` 本身；
          - 非原地：调用 `outplace_all_reduce`，返回一个新张量（自定义 kernel 往往
            会写到它自己的临时/注册缓冲区）。

        整体分发优先级（从上到下）：
          CPU shm → HPU/XPU/NPU 专用 → pynccl+symm-mem →
          ca(custom) → qr(quick, ROCm) → pymscclpp → torch symm-mem → pynccl → 原地回退

        输入输出示例（两 rank、SUM）：rank 0 输入 ``tensor([1., 2.])``，rank 1 输入
        ``tensor([3., 4.])``；两个 rank 都得到 ``tensor([4., 6.])``。
        """
        # 只有 1 张 GPU 时直接返回，无需通信
        if self.world_size == 1:
            return input_

        # CPU 张量：单机多进程下优先走共享内存 kernel（避开 gloo 的 socket 开销），
        # 否则回退到 torch.distributed。注意两者都是原地语义。
        if input_.is_cpu:
            if is_shm_available(input_.dtype, self.world_size, self.local_size):
                torch.ops.sgl_kernel.shm_allreduce(input_, REDUCE_OP_SUM)
            else:
                torch.distributed.all_reduce(input_, group=self.device_group)
            return input_

        # 非 CUDA 硬件后端：交给各自的专用通信器处理
        if self.hpu_communicator is not None and not self.hpu_communicator.disabled:
            return self.hpu_communicator.all_reduce(input_)

        if self.xpu_communicator is not None and not self.xpu_communicator.disabled:
            return self.xpu_communicator.all_reduce(input_)

        if self.npu_communicator is not None and not self.npu_communicator.disabled:
            return self.npu_communicator.all_reduce(input_)

        # mscclpp 有自己的尺寸/条件阈值，先算好结果，后面多处复用
        should_use_pymscclpp_allreduce = (
            self.pymscclpp_comm is not None
            and self.pymscclpp_comm.should_mscclpp_allreduce(input_)
        )
        # 开启对称内存（symmetric memory）时：各 rank 的缓冲区地址已提前互相注册，
        # NCCL 可以走零拷贝的快路径，因此直接原地 all-reduce 并提前返回。
        if (
            self.pynccl_comm is not None
            and self.is_symmetric_memory_enabled()
            and not should_use_pymscclpp_allreduce
        ):
            self.debug_check_symmetric_mempool(self, {"input": input_}, "all_reduce")
            with self.pynccl_comm.change_state(enable=True):
                self.pynccl_comm.all_reduce(input_)
                return input_

        # 按优先级选择一种“非原地” all-reduce 实现；全部不满足则回退到原地实现。
        # 注意：这里只是“选方法名”，真正的调用在 _all_reduce_out_place 里完成。
        outplace_all_reduce_method = None
        if (
            self.ca_comm is not None
            and not self.ca_comm.disabled
            and not should_use_pymscclpp_allreduce
            # should_custom_ar 会检查张量尺寸上限与对齐；超过阈值则不适用
            and self.ca_comm.should_custom_ar(input_)
        ):
            outplace_all_reduce_method = "ca"
        elif (
            self.qr_comm is not None
            and not self.qr_comm.disabled
            and self.qr_comm.should_quick_allreduce(input_)
        ):
            outplace_all_reduce_method = "qr"
        elif self.pymscclpp_comm is not None and should_use_pymscclpp_allreduce:
            outplace_all_reduce_method = "pymscclpp"
        elif (
            self.torch_symm_mem_comm is not None
            and not self.torch_symm_mem_comm.disabled
            and self.torch_symm_mem_comm.should_torch_symm_mem_allreduce(input_)
        ):
            outplace_all_reduce_method = "torch_symm_mem"
        elif is_in_tc_piecewise_cuda_graph() and self.pynccl_comm is not None:
            # piecewise cuda graph 下不能原地改写图外的输入缓冲区，
            # 因此使用 pynccl 的非原地 allreduce。
            outplace_all_reduce_method = "pynccl"
        if outplace_all_reduce_method is not None:
            # 通过自定义算子调用（传组名字符串而非 self），以便被 torch.compile 追踪
            return outplace_all_reduce(
                input_,
                group_name=self.unique_name,
                outplace_all_reduce_method=outplace_all_reduce_method,
            )
        else:
            # 原地版：算子声明 mutates_args=["tensor"]，结果直接写回 input_
            inplace_all_reduce(input_, group_name=self.unique_name)
            return input_

    def quant_all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """
        面向用户的量化 all-reduce，用法与 all_reduce 类似（仅 NPU 支持）。
        先将数据量化再通信，以降低通信量。

        输入输出示例（两 rank）：rank 0 输入 ``[1., 2.]``、rank 1 输入 ``[3., 4.]``，
        两边均得到近似 ``[4., 6.]``；具体误差取决于 NPU 通信器的量化格式。
        """
        # 只有 1 张卡时直接返回
        if self.world_size == 1:
            return input_

        if self.npu_communicator is not None and not self.npu_communicator.disabled:
            return self.npu_communicator.quant_all_reduce(input_)
        else:
            inplace_all_reduce(input_, group_name=self.unique_name)
            return input_

    def fused_allreduce_rmsnorm(
        self,
        input_: torch.Tensor,
        residual_inp_: torch.Tensor,
        weight_: torch.Tensor,
        eps: float,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """尝试通过 custom all-reduce 通信器执行融合的 all-reduce + RMSNorm。仅 ROCm/HIP 支持。

        返回 None 表示不支持融合路径，调用方应回退到“all-reduce + 单独 RMSNorm”。

        输入输出示例：输入形状均为 ``[num_tokens, hidden_size]`` 时，支持路径返回
        ``(归一化输出, 更新后的 residual)``；当前设备或通信器不支持时返回 ``None``。
        """
        ca_comm = self.ca_comm
        if ca_comm is None or getattr(ca_comm, "disabled", True):
            return None

        # 如果通信器自己提供了融合 API，优先使用它。
        if hasattr(ca_comm, "fused_allreduce_rmsnorm"):
            try:
                return ca_comm.fused_allreduce_rmsnorm(
                    input_, residual_inp_, weight_, eps
                )
            except Exception:
                # 失败则回退到下面的 custom_fused_ar_rms 路径。
                pass

        if not hasattr(ca_comm, "custom_fused_ar_rms"):
            return None

        # 融合 AR+RMSNorm 的 1-stage / 2-stage 选择：
        # 1-stage kernel 每个 token 启动一个 block，上限为 80 个 token（kMaxBlocks）。
        # 这里用字节数阈值做保护，使得大批 prefill 落到 2-stage kernel，而不是直接
        # 报运行时错误。AITER 的 C++ dispatch 已经会判断哪些 hidden_dim 支持 1-stage。
        if envs.SGLANG_USE_1STAGE_ALLREDUCE.is_set():
            use_1stage_ar = envs.SGLANG_USE_1STAGE_ALLREDUCE.get()
        else:
            total_bytes = input_.numel() * input_.element_size()
            use_1stage_ar = total_bytes <= 128 * 1024

        if (
            getattr(ca_comm, "_IS_CAPTURING", False)
            and not torch.cuda.is_current_stream_capturing()
            and is_in_tc_piecewise_cuda_graph()
        ):
            if not hasattr(ca_comm, "fused_ar_rms"):
                return None
            return ca_comm.fused_ar_rms(
                input_,
                residual_inp_,
                w=weight_,
                eps=eps,
                registered=False,
                use_1stage=use_1stage_ar,
            )
        fused_outputs = ca_comm.custom_fused_ar_rms(
            input_,
            residual_inp_,
            weight_,
            eps,
            use_1stage_ar,
        )
        return fused_outputs

    def _all_reduce_out_place(
        self, input_: torch.Tensor, outplace_all_reduce_method: str
    ) -> torch.Tensor:
        """非原地 all-reduce 的实际执行体：根据上层选定的方法名调用对应通信器。

        由 `outplace_all_reduce` 自定义算子回调进来（算子只能拿到组名字符串，
        因此先反查到本对象再调用本方法）。返回的是新张量，不修改 `input_`。
        """
        ca_comm = self.ca_comm
        qr_comm = self.qr_comm
        pymscclpp_comm = self.pymscclpp_comm
        torch_symm_mem_comm = self.torch_symm_mem_comm
        pynccl_comm = self.pynccl_comm
        # 能走到这里说明上层已经选中了某个通信器，至少有一个存在
        assert any([qr_comm, ca_comm, pymscclpp_comm, torch_symm_mem_comm, pynccl_comm])
        if outplace_all_reduce_method == "ca":
            assert not ca_comm.disabled
            out = ca_comm.custom_all_reduce(input_)
        elif outplace_all_reduce_method == "qr":
            assert not qr_comm.disabled
            out = qr_comm.quick_all_reduce(input_)
        elif outplace_all_reduce_method == "torch_symm_mem":
            assert not torch_symm_mem_comm.disabled
            out = torch_symm_mem_comm.all_reduce(input_)
        elif outplace_all_reduce_method == "pymscclpp":
            assert not pymscclpp_comm.disabled
            out = pymscclpp_comm.all_reduce(input_)
        elif outplace_all_reduce_method == "pynccl":
            # change_state(enable=True) 临时打开 pynccl（它在 eager 模式下默认 disabled）
            with pynccl_comm.change_state(enable=True):
                out = pynccl_comm.outplace_all_reduce(input_)
        assert out is not None
        return out

    def _all_reduce_in_place(self, input_: torch.Tensor) -> None:
        """原地 all-reduce：结果直接覆盖写回 `input_`，无返回值。

        优先级：pynccl（CUDA Graph 内可用）→ torch symm-mem → torch.distributed。
        由 `inplace_all_reduce` 自定义算子回调进来。
        """
        pynccl_comm = self.pynccl_comm
        torch_symm_mem_comm = self.torch_symm_mem_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.all_reduce(input_)
        elif torch_symm_mem_comm is not None and not torch_symm_mem_comm.disabled:
            torch_symm_mem_comm.all_reduce(input_)
        else:
            torch.distributed.all_reduce(input_, group=self.device_group)

    def _reduce_scatter_tensor(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
    ) -> torch.Tensor:
        """reduce-scatter 的实际执行体。

        语义：先对所有 rank 的 `input` 逐元素求和，再沿第 0 维均分为 world_size 段，
        第 i 段写入 rank i 的 `output`。因此：
            input.shape[0] == world_size * output.shape[0]
        可以理解为 `all_reduce` + `取本 rank 分片`，但通信量只有 all-reduce 的一半。
        典型用途：TP 下的序列并行（SP）——用 reduce-scatter 替代 all-reduce，
        让后续的 LayerNorm 等逐 token 算子只在分片上计算。

        注：`output` 由调用方预先分配，本方法原地填充。

        输入输出示例（两 rank、SUM）：两个 rank 分别输入 ``[1,2,3,4]`` 和
        ``[10,20,30,40]``；rank 0 的 ``output`` 为 ``[11,22]``，rank 1 为
        ``[33,44]``。
        """
        pynccl_comm = self.pynccl_comm
        # 开启对称内存时，即使 pynccl 处于 disabled 也要走 pynccl（才能用到零拷贝快路径）
        if pynccl_comm is not None and (
            not pynccl_comm.disabled or self.is_symmetric_memory_enabled()
        ):
            # 调试校验：确认传入的张量确实来自对称内存池
            self.debug_check_symmetric_mempool(
                self, {"output": output, "input": input}, "reduce_scatter_tensor"
            )
            with pynccl_comm.change_state(enable=True):
                pynccl_comm.reduce_scatter(output, input)
        else:
            torch.distributed.reduce_scatter_tensor(
                output, input, group=self.device_group
            )
        return output

    def reduce_scatter_tensor(self, output: torch.Tensor, input: torch.Tensor):
        """reduce-scatter 对外入口（等长版，结果写入预分配的 `output`）。

        NPU 直接调用实现；其他平台走自定义算子 `reg_reduce_scatter_tensor`，
        以便 torch.compile 能正确识别“会修改 output”这一副作用。

        输入输出示例（world_size=2）：``input.shape == [8, d]`` 时，每个 rank 需传入
        ``output.shape == [4, d]``；调用后不同 rank 分别持有归约结果的前、后半段。
        """
        if _is_npu:
            self._reduce_scatter_tensor(output, input)
        else:
            reg_reduce_scatter_tensor(output, input, group_name=self.unique_name)

    def _all_to_all_single(self, output: torch.Tensor, input: torch.Tensor) -> None:
        """all-to-all 的实际执行体：每个 rank 把自己的数据均分后分发给全部 rank。

        输入输出示例见 ``all_to_all_single``；本方法只负责调用底层进程组。
        """
        torch.distributed.all_to_all_single(output, input, group=self.device_group)

    def all_to_all_single(self, output: torch.Tensor, input: torch.Tensor):
        """all-to-all 对外入口；单卡时退化为一次拷贝。

        输入输出示例（两 rank）：rank 0 输入 ``[0,1,2,3]``，rank 1 输入
        ``[10,11,12,13]``；rank 0 输出 ``[0,1,10,11]``，rank 1 输出
        ``[2,3,12,13]``。
        """
        if self.world_size == 1:
            output.copy_(input)
            return
        reg_all_to_all_single(output, input, group_name=self.unique_name)

    def reduce_scatter(
        self,
        output: torch.Tensor,
        input_list: List[torch.Tensor],
    ) -> None:
        """列表形式的 reduce-scatter。

        与 `reduce_scatter_tensor` 的区别：输入是长度为 world_size 的张量列表（每个
        元素对应一个目标 rank 的分片），而不是单个已拼接好的大张量；
        结果（对应本 rank 分片的求和）写入 `output`。

        输入输出示例（两 rank）：rank 0 输入列表 ``[[1,2], [3,4]]``，rank 1 输入
        ``[[10,20], [30,40]]``；rank 0 输出 ``[11,22]``，rank 1 输出 ``[33,44]``。
        """
        # TODO(ch-wan): 待支持其他后端（目前只有 torch.distributed 路径）
        torch.distributed.reduce_scatter(output, input_list, group=self.device_group)
        return output

    def reduce_scatterv(
        self,
        input_: torch.Tensor,
        output: Optional[torch.Tensor] = None,
        sizes: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """变长版 reduce-scatter（v = variable）：允许每个 rank 分到不同长度的分片。

        典型场景：DP attention 下各 DP rank 的 token 数不相等，无法均分。

        Args:
            input_: 完整的待归约张量，第 0 维长度应等于 sum(sizes)。
            output: 可选的预分配输出；为 None 时内部自行分配。
            sizes: 长度为 world_size 的列表，描述每个 rank 应得到的行数；
                为 None 时要求可以均分。
        Returns:
            本 rank 对应的那一分片的归约结果。

        注：变长语义仅 pynccl 支持，torch.distributed 无对应接口，因此无回退路径。

        输入输出示例（两 rank，``sizes=[1, 2]``）：归约后的完整第 0 维长度为 3，
        rank 0 得到 1 行，rank 1 得到 2 行。
        """
        world_size = self.world_size
        pynccl_comm = self.pynccl_comm

        with pynccl_comm.change_state(enable=True):
            assert (
                pynccl_comm is not None and not pynccl_comm.disabled
            ), "pynccl is required for reduce_scatterv"

            # 推导本 rank 输出分片的行数
            if sizes is not None:
                assert len(sizes) == world_size
                assert input_.shape[0] == sum(sizes)
                chunk_size = sizes[self.rank_in_group]
            else:
                assert input_.shape[0] % world_size == 0
                chunk_size = input_.shape[0] // world_size
            # 除第 0 维外的其余维度保持不变
            output_shape = (chunk_size,) + input_.shape[1:]

            if output is None:
                output = torch.empty(
                    output_shape, dtype=input_.dtype, device=input_.device
                )
            else:
                assert output.shape == output_shape

            pynccl_comm.reduce_scatter(output, input_, sizes=sizes)
            return output

    def _all_gather_into_tensor(self, output: torch.Tensor, input: torch.Tensor):
        """all-gather 的实际执行体。

        语义：把各 rank 的 `input` 按 rank 顺序沿第 0 维拼接，写入预分配的 `output`：
            output.shape[0] == world_size * input.shape[0]
        它是 reduce-scatter 的“对偶”操作，两者合起来等价于一次 all-reduce。
        """
        # Aiter 自定义 all-gather（ROCm）。设置 SGLANG_USE_AITER_AG=0 可关闭。
        # 形状/布局的校验仍然由 Aiter 的 should_custom_ag 负责：
        # 16B 对齐、弱连续、拓扑受支持，以及单 rank 尺寸 <= max_size/(world*2)。
        # 命中时直接写入调用方预分配的 `output`：CUDA Graph 捕获中用 all_gather_reg，
        # 否则用 all_gather_unreg。
        ca_comm = self.ca_comm
        if (
            is_hip()
            and envs.SGLANG_USE_AITER_AG.get()
            and self._has_aiter_custom_all_gather()
            and input.is_contiguous()
            and output.is_contiguous()
            and input.dtype in (torch.float32, torch.float16, torch.bfloat16)
            and ca_comm.should_custom_ag(input)
        ):
            # _IS_CAPTURING 表示当前处于 CUDA Graph 的捕获/预热阶段
            if getattr(ca_comm, "_IS_CAPTURING", False):
                if torch.cuda.is_current_stream_capturing():
                    # 真正在捕获：用已注册缓冲区版本，保证地址在回放时仍然有效
                    ca_comm.all_gather_reg(input, out=output, dim=0)
                elif is_in_tc_piecewise_cuda_graph():
                    ca_comm.all_gather_unreg(input, out=output, dim=0)
                else:
                    # 真正的 CUDA graph 预热阶段：避免发起不同的主机侧集体通信。
                    output.zero_()
                return
            else:
                ca_comm.all_gather_unreg(input, out=output, dim=0)
                return

        # 通用路径：优先 pynccl（可被 CUDA Graph 捕获），否则回退 torch.distributed
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and (
            not pynccl_comm.disabled or self.is_symmetric_memory_enabled()
        ):
            self.debug_check_symmetric_mempool(
                self, {"output": output}, "all_gather_into_tensor"
            )
            with pynccl_comm.change_state(enable=True):
                pynccl_comm.all_gather(output, input)
        else:
            torch.distributed.all_gather_into_tensor(
                output, input, group=self.device_group
            )

    def _has_aiter_custom_all_gather(self) -> bool:
        """判断当前 ca_comm 是否具备 Aiter 自定义 all-gather 的全部接口。"""
        if self._deterministic_collectives_enabled():
            return False
        ca_comm = self.ca_comm
        return (
            ca_comm is not None
            and not getattr(ca_comm, "disabled", True)
            and hasattr(ca_comm, "should_custom_ag")
            and hasattr(ca_comm, "all_gather_reg")
            and hasattr(ca_comm, "all_gather_unreg")
        )

    @staticmethod
    def _deterministic_collectives_enabled() -> bool:
        """是否要求集体通信具备确定性（确定性推理下需要禁用部分优化路径）。"""
        if envs.SGLANG_USE_1STAGE_ALLREDUCE.is_set():
            return envs.SGLANG_USE_1STAGE_ALLREDUCE.get()
        return envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()

    def all_gather_into_tensor(self, output: torch.Tensor, input: torch.Tensor):
        """all-gather 对外入口（结果写入预分配的 `output`，沿第 0 维拼接）。

        NPU/XPU 直接调用实现；其他平台走自定义算子 `reg_all_gather_into_tensor`，
        以便 torch.compile 能正确识别“会修改 output”这一副作用。

        输入输出示例（两 rank）：各 rank 输入形状 ``[1, d]``，预分配输出形状
        ``[2, d]``；调用后每个 rank 的输出都按 rank 顺序包含两份输入。
        """
        if _is_npu or _is_xpu:
            self._all_gather_into_tensor(output, input)
        else:
            reg_all_gather_into_tensor(output, input, group_name=self.unique_name)

    def cp_all_gather_into_tensor_async(
        self, output: torch.Tensor, input: torch.Tensor, stream: torch.cuda.Stream
    ):
        """
        在指定 stream 上实现异步的 `allgather` 操作。
        （默认的 `torch.distributed.all_gather_into_tensor` 会触发 event 同步），
        从而消除由同步导致的 CPU 侧 launch-kernel 阻塞问题。
        具体实现上使用 pynccl 提供的接口，去掉 event 的同步逻辑。
        主要用于 context parallel（CP）下的通信-计算重叠。

        输入输出示例（两 rank）：各 rank 输入 ``[1, d]``、输出缓冲区 ``[2, d]``；
        调用立即把通信排入指定 stream，待该 stream 完成后输出包含两份输入。
        """
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is None or pynccl_comm.disabled:
            self.all_gather_into_tensor(output, input)
        else:
            pynccl_comm.cp_all_gather_into_tensor(output, input, stream=stream)

    def all_gather(
        self,
        input_: torch.Tensor,
        dim: int = -1,
        output_tensor_list: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """在指定维度 `dim` 上做 all-gather，返回拼接后的新张量。

        语义：组内每个 rank 提供一个相同形状的 `input_`，输出是按 rank 顺序在
        `dim` 维上拼接的结果，且每个 rank 都拿到完整拼接结果：
            output.shape[dim] == world_size * input_.shape[dim]
        典型用途：ColumnParallelLinear 的输出汇总、SP 中从分片恢复完整序列、
        DP attention 中汇集各 DP rank 的 hidden states。

        Args:
            input_: 本 rank 的输入分片。
            dim: 在哪个维度上拼接（支持负数，默认最后一维）。
            output_tensor_list: 可选；传入时改为列表形式的原地 all-gather，
                结果逐个写入列表元素（不做拼接，返回值不是拼接张量）。

        输入输出示例（两 rank，``dim=0``）：rank 0 输入 ``[[1,2]]``，rank 1 输入
        ``[[3,4]]``；每个 rank 都返回 ``[[1,2], [3,4]]``。
        """
        world_size = self.world_size
        # 只有 1 张 GPU 时跳过通信
        if world_size == 1:
            if output_tensor_list is not None:
                logger.warning(
                    "Performing in-place all-gather with a group size of 1. "
                    "This may be unnecessary; consider bypassing it for better efficiency."
                )
                output_tensor_list[0].copy_(input_)
                return None
            else:
                return input_

        if output_tensor_list is not None:
            # TODO(ch-wan): 待支持其他后端
            return torch.distributed.all_gather(
                output_tensor_list, input_, group=self.device_group
            )

        assert (
            -input_.dim() <= dim < input_.dim()
        ), f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"

        # HPU 使用 HPU 专用通信器。
        hpu_comm = self.hpu_communicator
        if hpu_comm is not None and not hpu_comm.disabled:
            return hpu_comm.all_gather(input_, dim)

        # NPU 使用 NPU 专用通信器。
        npu_comm = self.npu_communicator
        if npu_comm is not None and not npu_comm.disabled:
            return npu_comm.all_gather(input_, dim)

        if dim < 0:
            # 把负数维度转换为正数。
            dim += input_.dim()
        input_size = input_.size()
        # 注意：这里必须用 concat 风格的 all-gather，
        # stack 风格的 all-gather 与 torch.compile 存在兼容性问题，
        # 参见 https://github.com/pytorch/pytorch/issues/138795
        output_size = (input_size[0] * world_size,) + input_size[1:]
        # 分配输出张量（条件允许时从对称内存池分配，以便使用 symm-mem 优化）。
        with self.use_symmetric_memory(
            self, disabled=not self.is_allocation_symmetric()
        ):
            output_tensor = torch.empty(
                output_size, dtype=input_.dtype, device=input_.device
            )

        # 执行 all-gather。
        if input_.is_cpu:
            if is_shm_available(input_.dtype, self.world_size, self.local_size):
                return torch.ops.sgl_kernel.shm_allgather(input_, dim)
            else:
                torch.distributed.all_gather_into_tensor(
                    output_tensor, input_, group=self.device_group
                )
        else:
            self.all_gather_into_tensor(output_tensor, input_)

        # 底层 all-gather 总是沿第 0 维拼接，这里把它重排为“沿指定维度 dim 拼接”：
        # 1) 先把 rank 维度显式化：[W*d0, ...] -> [W, d0, ...]
        output_tensor = output_tensor.reshape((world_size,) + input_size)
        # 2) 把 rank 维度搬到 dim 的前面：[W, ..., d_dim, ...] -> [..., W, d_dim, ...]
        output_tensor = output_tensor.movedim(0, dim)
        # 3) 将 (W, d_dim) 合并为一个维度，得到最终形状
        output_tensor = output_tensor.reshape(
            input_size[:dim] + (world_size * input_size[dim],) + input_size[dim + 1 :]
        )
        return output_tensor

    def all_gatherv(
        self,
        input_: Union[torch.Tensor, List[torch.Tensor]],
        sizes: Optional[List[int]] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        变长版 all-gather（v = variable）：支持每个 rank 长度不同，
        也支持一次传入多个输入张量（合并为一次 NCCL 批量提交）。

        与 `all_gather` 的区别：只能沿第 0 维拼接，但允许各 rank 行数不等。
        典型场景：DP attention 下各 DP rank 的 token 数不相等。

        Args:
            input_: 单个张量或张量列表（列表内各张量共用同一份 `sizes`）。
            sizes: 长度为 world_size 的列表，表示每个 rank 要 gather 的行数；
                为 None 时退化为等长 all-gather。
        Returns:
            与输入一一对应的输出张量列表（即使传入的是单个张量）。

        注：变长语义仅 pynccl 支持，无 torch.distributed 回退路径。

        输入输出示例（两 rank，``sizes=[1, 2]``）：rank 0 输入形状 ``[1,d]``，
        rank 1 输入形状 ``[2,d]``；两边输出列表中的张量形状均为 ``[3,d]``。
        """
        world_size = self.world_size
        pynccl_comm = self.pynccl_comm

        with pynccl_comm.change_state(enable=True):
            assert (
                pynccl_comm is not None and not pynccl_comm.disabled
            ), "pynccl is required for all_gatherv"

            def _all_gather_allocate_output(
                input_: torch.Tensor, sizes: Optional[List[int]] = None
            ):
                input_size = input_.size()
                if sizes is not None:
                    assert len(sizes) == world_size
                    assert input_.shape[0] == sizes[self.rank_in_group]
                    output_size = (sum(sizes),) + input_size[1:]
                    # 如果组内所有输入形状一致，就不需要 'sizes'（退化为等长 all-gather）
                    if all(s == sizes[0] for s in sizes):
                        sizes = None
                else:
                    output_size = (input_size[0] * world_size,) + input_size[1:]
                # 分配输出张量（变长场景下不能使用对称内存）。
                with self.use_symmetric_memory(self, disabled=sizes is not None):
                    output_tensor = torch.empty(
                        output_size, dtype=input_.dtype, device=input_.device
                    )
                return output_tensor, sizes

            # 统一成列表形式处理，后续逻辑不再区分单/多输入
            if isinstance(input_, torch.Tensor):
                input_ = [input_]

            # 先为所有输入分配好输出，再一次性提交通信
            output_list = []
            size_list = []
            for inp in input_:
                output_tensor, s = _all_gather_allocate_output(inp, sizes=sizes)
                output_list.append(output_tensor)
                size_list.append(s)

            # 用 group_start/group_end 把多个 all-gather 合并为一次 NCCL 批量提交，减少开销
            pynccl_comm.group_start()
            for i, inp in enumerate(input_):
                pynccl_comm.all_gather(output_list[i], inp, sizes=size_list[i])
            pynccl_comm.group_end()

            return output_list

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> Optional[torch.Tensor]:
        """把各 rank 的张量汇聚到目标 rank（只有 dst 拿到完整结果）。

        注：假设输入张量在所有 rank 上位于相同类型的设备上。
        注：`dst` 是目标进程的“组内 rank”，不是全局 rank。

        输入输出示例（两 rank，``dst=0, dim=0``）：输入分别为 ``[[1]]``、``[[2]]``；
        rank 0 返回 ``[[1],[2]]``，rank 1 返回 ``None``。
        """
        world_size = self.world_size
        # 只有 1 张 GPU 时跳过通信
        if world_size == 1:
            return input_
        assert (
            -input_.dim() <= dim < input_.dim()
        ), f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        if dim < 0:
            # 把负数维度转换为正数。
            dim += input_.dim()
        if self.xpu_communicator is not None and not self.xpu_communicator.disabled:
            return self.xpu_communicator.gather(input_, self.rank_in_group, dst, dim)
        # 只有目标 rank 需要分配接收缓冲区。
        if self.rank_in_group == dst:
            gather_list = [torch.empty_like(input_) for _ in range(world_size)]
        else:
            gather_list = None
        # 执行 gather。
        torch.distributed.gather(
            input_, gather_list, dst=self.ranks[dst], group=self.device_group
        )
        if self.rank_in_group == dst:
            output_tensor = torch.cat(gather_list, dim=dim)
        else:
            output_tensor = None
        return output_tensor

    def broadcast(self, input_: torch.Tensor, src: int = 0):
        """广播输入张量（原地写入，非 src 端的内容会被覆盖）。
        注：`src` 是源进程的“组内 rank”。

        输入输出示例（两 rank，``src=0``）：rank 0 输入 ``[1,2]``，rank 1 输入任意
        占位张量；调用后两个 rank 的张量都为 ``[1,2]``。
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # 只有 1 张 GPU 时跳过通信
        if self.world_size == 1:
            return input_
        # 执行广播。
        torch.distributed.broadcast(
            input_, src=self.ranks[src], group=self.device_group
        )
        return input_

    def broadcast_object(self, obj: Optional[Any] = None, src: int = 0):
        """广播任意 Python 对象（通过 pickle 序列化，走 CPU 组）。
        注：`src` 是源进程的“组内 rank”。

        输入输出示例：源 rank 传入 ``{"step": 3}``，其余 rank 传入 ``None``；
        所有 rank 最终都返回 ``{"step": 3}``。
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # 只有 1 张 GPU 时跳过通信
        if self.world_size == 1:
            return obj
        # 如果启用了共享内存消息队列，优先走它（延迟远低于 gloo）
        if self.mq_broadcaster is not None:
            assert src == 0, "Message queue broadcaster only supports src=0"
            return self.mq_broadcaster.broadcast_object(obj)
        if self.rank_in_group == src:
            torch.distributed.broadcast_object_list(
                [obj], src=self.ranks[src], group=self.cpu_group
            )
            return obj
        else:
            recv = [None]
            torch.distributed.broadcast_object_list(
                recv, src=self.ranks[src], group=self.cpu_group
            )
            return recv[0]

    def broadcast_object_list(
        self, obj_list: List[Any], src: int = 0, group: Optional[ProcessGroup] = None
    ):
        """广播一个对象列表（原地写入 obj_list）。
        注：`src` 是源进程的“组内 rank”。

        输入输出示例：源 rank 传入 ``["prefill", 8]``，其余 rank 传入
        ``[None, None]``；调用后每个列表都变为 ``["prefill", 8]``。
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # 只有 1 张 GPU 时跳过通信
        if self.world_size == 1:
            return obj_list
        # 执行广播。
        torch.distributed.broadcast_object_list(
            obj_list, src=self.ranks[src], group=self.device_group
        )
        return obj_list

    def all_gather_object(self, obj: Any) -> List[Any]:
        """收集组内每个 rank 的对象，返回按组内 rank 排序的列表（走 CPU 组）。

        输入输出示例：两个 rank 分别传入 ``{"rank": 0}``、``{"rank": 1}``，
        两边均返回 ``[{"rank": 0}, {"rank": 1}]``。
        """
        objs = [None] * self.world_size
        torch.distributed.all_gather_object(objs, obj, group=self.cpu_group)
        return objs

    def send_object(
        self,
        obj: Any,
        dst: int,
        async_send: bool = False,
    ) -> List[P2PWork]:
        """
        向目标 rank 发送一个 Python 对象。本函数的所有通信都走 CPU 组。

        协议：先发送 8 字节的长度，再发送序列化后的字节流；接收端按同样顺序读取。

        TODO: 如果需要 GPU 通信，请新增一个参数（如 data_group、group），
        或使用其他函数（如 send），或实现一个新函数（如 send_object_device）。

        注：`dst` 是目标进程的“组内 rank”。
        返回：async_send=True 时返回未完成的 P2PWork 列表，需调用方自行等待。

        输入输出示例：rank 0 调用 ``send_object({"step": 3}, dst=1)``，rank 1 调用
        ``recv_object(src=0)`` 后返回 ``{"step": 3}``；同步发送时返回空列表，异步发送
        时返回两个分别对应“长度”和“数据本体”的 ``P2PWork``。
        """

        assert dst < self.world_size, f"Invalid dst rank ({dst})"
        assert dst != self.rank_in_group, (
            "Invalid destination rank. Destination rank is the same "
            "as the current rank."
        )
        send_func = torch.distributed.isend if async_send else torch.distributed.send

        # 将对象序列化为字节张量，并计算其字节数
        object_tensor = torch.frombuffer(pickle.dumps(obj), dtype=torch.uint8)
        size_tensor = torch.tensor(
            [object_tensor.numel()], dtype=torch.long, device="cpu"
        )

        # 先发送对象大小，使接收端能预先分配缓冲区
        p2p_work = []
        size_work = send_func(
            size_tensor,
            self.ranks[dst],
            group=self.cpu_group,
        )
        if async_send:
            p2p_work.append(P2PWork(size_work, size_tensor))

        # 再发送对象本体
        object_work = send_func(
            object_tensor,
            self.ranks[dst],
            group=self.cpu_group,
        )
        if async_send:
            p2p_work.append(P2PWork(object_work, object_tensor))

        return p2p_work

    def recv_object(
        self,
        src: int,
    ) -> Any:
        """从源 rank 接收一个 Python 对象（与 ``send_object`` 配对，走 CPU 组）。

        ``src`` 是源进程的组内 rank。输入输出示例：发送端传入 ``["a", 1]`` 后，
        ``recv_object(src=0)`` 返回反序列化后的 ``["a", 1]``。
        """

        assert src < self.world_size, f"Invalid src rank ({src})"
        assert (
            src != self.rank_in_group
        ), "Invalid source rank. Source rank is the same as the current rank."

        size_tensor = torch.empty(1, dtype=torch.long, device="cpu")

        # 先接收对象大小。
        # 这里必须用 irecv，才能同时兼容发送端使用 isend 和 send 两种情况。
        work = torch.distributed.irecv(
            size_tensor, src=self.ranks[src], group=self.cpu_group
        )
        work.wait()

        # 用于接收序列化对象字节流的张量。
        object_tensor: Any = torch.empty(  # type: ignore[call-overload]
            size_tensor.item(),  # type: ignore[arg-type]
            dtype=torch.uint8,
            device="cpu",
        )

        work = torch.distributed.irecv(
            object_tensor, src=self.ranks[src], group=self.cpu_group
        )
        work.wait()

        obj = pickle.loads(object_tensor.numpy())
        return obj

    def broadcast_tensor_dict(
        self,
        tensor_dict: Optional[Dict[str, Union[torch.Tensor, Any]]] = None,
        src: int = 0,
        group: Optional[ProcessGroup] = None,
        metadata_group: Optional[ProcessGroup] = None,
    ) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:
        """广播一个张量字典（元数据走 CPU 组，张量本体走 device 组）。

        思路：先把字典拆成“元数据列表 + 张量列表”，广播元数据让接收端知道每个
        张量的 shape/dtype/device 并预分配，再异步广播各个张量。
        注：`src` 是源进程的“组内 rank”。

        输入输出示例：源 rank 传入 ``{"hidden": tensor([[1,2]]), "step": 3}``，
        其余 rank 传入 ``None``；所有 rank 最终返回键、非张量值及张量内容都相同的字典。
        """
        # 只有 1 张 GPU 时跳过通信
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return tensor_dict

        group = self.device_group
        metadata_group = self.cpu_group
        assert src < self.world_size, f"Invalid src rank ({src})"

        rank_in_group = self.rank_in_group
        if rank_in_group == src:
            metadata_list: List[Tuple[Any, Any]] = []
            assert isinstance(
                tensor_dict, dict
            ), f"Expecting a dictionary, got {type(tensor_dict)}"
            metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
            # `metadata_list` 位于 CPU 内存。
            # `broadcast_object_list` 的序列化与反序列化全部在 CPU 上发生，
            # 因此可以使用 CPU 组。
            self.broadcast_object(metadata_list, src=src)
            async_handles = []
            for tensor in tensor_list:
                if tensor.numel() == 0:
                    # 空张量无需广播，直接跳过。
                    continue
                if tensor.is_cpu:
                    # CPU 张量走 metadata_group（gloo）
                    handle = torch.distributed.broadcast(
                        tensor, src=self.ranks[src], group=metadata_group, async_op=True
                    )
                else:
                    # GPU 张量走 device 组
                    handle = torch.distributed.broadcast(
                        tensor, src=self.ranks[src], group=group, async_op=True
                    )
                async_handles.append(handle)
            # 先全部发起异步广播，再统一等待，以获得更好的重叠
            for async_handle in async_handles:
                async_handle.wait()

        else:
            # 非源 rank：先拿到元数据，再按元数据预分配张量并接收
            metadata_list = self.broadcast_object(None, src=src)
            tensor_dict = {}
            async_handles = []
            for key, value in metadata_list:
                if isinstance(value, TensorMetadata):
                    tensor = torch.empty(
                        value.size, dtype=value.dtype, device=value.device
                    )
                    if tensor.numel() == 0:
                        # 空张量无需广播，直接跳过。
                        tensor_dict[key] = tensor
                        continue
                    if tensor.is_cpu:
                        # CPU 张量走 metadata_group（gloo）
                        handle = torch.distributed.broadcast(
                            tensor,
                            src=self.ranks[src],
                            group=metadata_group,
                            async_op=True,
                        )
                    else:
                        # GPU 张量走 device 组
                        handle = torch.distributed.broadcast(
                            tensor, src=self.ranks[src], group=group, async_op=True
                        )
                    async_handles.append(handle)
                    tensor_dict[key] = tensor
                else:
                    tensor_dict[key] = value
            for async_handle in async_handles:
                async_handle.wait()
        return tensor_dict

    def send_tensor_dict(
        self,
        tensor_dict: Dict[str, Union[torch.Tensor, Any]],
        dst: Optional[int] = None,
        all_gather_group: Optional["GroupCoordinator"] = None,
        async_send: bool = False,
    ) -> Optional[List[P2PWork]]:
        """发送一个张量字典（主要用于流水线并行 PP 的阶段间传输）。

        `all_gather_group` 不为空时启用 send-allgather 优化：发送方只发自己负责的
        那一切片，接收方再在该组内 all-gather 拼回完整张量，以减少跨节点流量。
        注：`dst` 是目标进程的“组内 rank”；默认为环上的下一个 rank。

        输入输出示例：PP 前一阶段调用
        ``send_tensor_dict({"hidden": tensor([[1,2]])}, dst=1)``，后一阶段调用
        ``recv_tensor_dict(src=0)`` 后得到相同 shape、dtype 和内容的 ``hidden`` 张量。
        """
        # 只有 1 张 GPU 时跳过通信
        if self.world_size == 1:
            return tensor_dict

        all_gather_size = 1 if all_gather_group is None else all_gather_group.world_size
        all_gather_rank = (
            0 if all_gather_group is None else all_gather_group.rank_in_group
        )

        group = self.device_group
        metadata_group = self.cpu_group

        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        assert dst < self.world_size, f"Invalid dst rank ({dst})"

        assert isinstance(
            tensor_dict, dict
        ), f"Expecting a dictionary, got {type(tensor_dict)}"
        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        # 注：虽然改用设备到设备（D2D）传输会为序列化引入额外的设备到主机（D2H）
        # 拷贝开销，但我们的基准测试表明 D2D 的整体传输性能更好，原因在于：
        # 1. D2D 传输带宽更高
        # 2. 可以让 send 与 recv 操作重叠
        # 因此净收益足以支撑这一方案。

        send_func = torch.distributed.isend if async_send else torch.distributed.send
        # 先把元数据（key 与张量 shape/dtype/device）发过去
        p2p_works = self.send_object(metadata_list, dst=dst, async_send=async_send)

        for tensor in tensor_list:
            if tensor.numel() == 0:
                # 空张量不需要发送。
                continue

            # send-allgather 优化：只发送一个分片，由接收方 all-gather 拼回。
            if all_gather_group is not None and tensor.numel() % all_gather_size == 0:
                tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]

            comm_group = metadata_group if tensor.is_cpu else group
            work = send_func(tensor, self.ranks[dst], group=comm_group)
            if async_send:
                p2p_works.append(P2PWork(work, tensor))
        return p2p_works

    def recv_tensor_dict(
        self,
        src: Optional[int] = None,
        all_gather_group: Optional["GroupCoordinator"] = None,
    ) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:
        """接收一个张量字典（与 ``send_tensor_dict`` 配对使用）。

        ``src`` 是源进程的组内 rank；默认为环上的上一个 rank。输入输出示例：发送端
        发送 ``{"hidden": tensor([[1,2]]), "step": 3}`` 后，本函数返回同结构字典；
        张量会按元数据在当前设备重新分配，非张量值直接从元数据恢复。
        """
        # 只有 1 张 GPU 时跳过通信
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return None

        all_gather_size = 1 if all_gather_group is None else all_gather_group.world_size
        all_gather_rank = (
            0 if all_gather_group is None else all_gather_group.rank_in_group
        )

        group = self.device_group
        metadata_group = self.cpu_group

        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        assert src < self.world_size, f"Invalid src rank ({src})"

        # 先拿到元数据，再按元数据逐个预分配并接收张量
        recv_metadata_list = self.recv_object(src=src)
        tensor_dict: Dict[str, Any] = {}
        for key, value in recv_metadata_list:
            if isinstance(value, TensorMetadata):
                tensor = torch.empty(value.size, dtype=value.dtype, device=value.device)
                if tensor.numel() == 0:
                    # 空张量无需传输，直接跳过。
                    tensor_dict[key] = tensor
                    continue

                # send-allgather 优化：对端只发了一个分片，这里需要 all-gather 拼回。
                use_all_gather = (
                    all_gather_group is not None
                    and tensor.numel() % all_gather_size == 0
                )

                if use_all_gather:
                    orig_shape = tensor.shape
                    tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]

                # 这里必须用 irecv，才能同时兼容发送端使用 isend 和 send 两种情况。
                comm_group = metadata_group if tensor.is_cpu else group
                work = torch.distributed.irecv(
                    tensor, src=self.ranks[src], group=comm_group
                )
                work.wait()

                if use_all_gather:
                    tensor = all_gather_group.all_gather(tensor, dim=0)
                    tensor = tensor.reshape(orig_shape)

                tensor_dict[key] = tensor
            else:
                tensor_dict[key] = value
        return tensor_dict

    def barrier(self):
        """阻塞当前 rank，直到组内所有 rank 都到达该屏障。

        不使用 ``device_group``：NCCL barrier 内部会广播隐式创建的 GPU 张量，容易
        扰乱当前设备，因此这里固定使用 CPU 组。输入输出示例：四个 rank 到达时间不同，
        最先到达的 rank 会等待最后一个 rank 到达，随后四者继续执行；无返回值。
        """
        torch.distributed.barrier(group=self.cpu_group)

    def send(self, tensor: torch.Tensor, dst: Optional[int] = None) -> None:
        """向目标 rank 发送一个张量。

        ``dst`` 是目标进程的组内 rank；省略时发送给环上的下一个 rank。
        输入输出示例：组 ``[0,1]`` 的 rank 0 调用 ``send(tensor([1,2]), dst=1)``；
        rank 1 可用 ``recv(torch.Size([2]), torch.int64, src=0)`` 得到 ``[1,2]``。
        本函数无返回值；底层 PyNccl 路径可异步排队，torch.distributed 路径为阻塞 send。
        """
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size

        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.send(tensor, dst)
        else:
            torch.distributed.send(tensor, self.ranks[dst], self.device_group)

    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: Optional[int] = None
    ) -> torch.Tensor:
        """从源 rank 接收一个张量，调用方必须提前知道形状与类型。

        ``src`` 是源进程的组内 rank；省略时从环上的上一个 rank 接收。
        输入输出示例：rank 0 发送 ``tensor([1.,2.])`` 后，rank 1 调用
        ``recv(torch.Size([2]), torch.float32, src=0)``，返回 ``tensor([1.,2.])``。
        """
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size

        tensor = torch.empty(size, dtype=dtype, device=self.device)
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.recv(tensor, src)
        else:
            torch.distributed.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    def destroy(self):
        """销毁本组持有的所有进程组与通信器，释放相应资源。"""
        if self.device_group is not None:
            torch.distributed.destroy_process_group(self.device_group)
            self.device_group = None
        if self.cpu_group is not None:
            torch.distributed.destroy_process_group(self.cpu_group)
            self.cpu_group = None
        if self.pynccl_comm is not None:
            self.pynccl_comm = None
        if self.pymscclpp_comm is not None:
            self.pymscclpp_comm.destroy()
        if self.ca_comm is not None:
            self.ca_comm = None
        if self.mq_broadcaster is not None:
            self.mq_broadcaster = None


_WORLD: Optional[GroupCoordinator] = None


def get_world_group() -> GroupCoordinator:
    assert _WORLD is not None, "world group is not initialized"
    return _WORLD


def init_world_group(
    ranks: List[int], local_rank: int, backend: str, recovered_rank: bool = False
) -> GroupCoordinator:
    """创建覆盖全部进程的 WORLD 组协调器。

    输入输出示例：输入 ``ranks=[0,1,2,3], local_rank=1, backend="nccl"``，
    当前全局 rank 1 返回 ``world_size=4``、``rank_in_group=1`` 的协调器。
    """
    return GroupCoordinator(
        group_ranks=[ranks],
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_pynccl=False,
        use_pymscclpp=False,
        use_custom_allreduce=False,
        use_torch_symm_mem_all_reduce=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        use_npu_communicator=False,
        group_name="world",
        recovered_rank=recovered_rank,
    )


def init_model_parallel_group(
    group_ranks: List[List[int]],
    local_rank: int,
    backend: str,
    use_pynccl: Optional[bool] = None,
    use_custom_allreduce: Optional[bool] = None,
    use_message_queue_broadcaster: bool = False,
    group_name: Optional[str] = None,
    use_mscclpp_allreduce: Optional[bool] = None,
    use_torch_symm_mem_allreduce: Optional[bool] = None,
    recovered_rank: bool = False,
) -> GroupCoordinator:
    """创建某一并行维度的组协调器，并按平台与全局开关选择通信优化。

    输入输出示例：``group_ranks=[[0,1],[2,3]]`` 且当前 rank 为 2 时，返回仅持有
    ``ranks=[2,3]``、``rank_in_group=0`` 的协调器；所有进程仍须以相同顺序创建全部子组。
    """
    if use_custom_allreduce is None:
        use_custom_allreduce = _ENABLE_CUSTOM_ALL_REDUCE
    if use_mscclpp_allreduce is None:
        use_mscclpp_allreduce = _ENABLE_MSCCLPP_ALL_REDUCE
    if use_torch_symm_mem_allreduce is None:
        use_torch_symm_mem_allreduce = _ENABLE_TORCH_SYMM_MEM_ALL_REDUCE
    return GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_pynccl=(
            not (_is_npu or _is_xpu or backend == "mooncake")
            if use_pynccl is None
            else use_pynccl
        ),
        use_pymscclpp=use_mscclpp_allreduce,
        use_custom_allreduce=use_custom_allreduce,
        use_torch_symm_mem_all_reduce=use_torch_symm_mem_allreduce,
        use_hpu_communicator=True,
        use_xpu_communicator=True,
        use_npu_communicator=True,
        use_message_queue_broadcaster=use_message_queue_broadcaster,
        group_name=group_name,
        recovered_rank=recovered_rank,
    )


_TP: Optional[GroupCoordinator] = None
_ATTN_TP: Optional[GroupCoordinator] = None
_ATTN_CP: Optional[GroupCoordinator] = None

# 为 PD-Multiplexing 的 prefill 阶段复制一份独立的 GroupCoordinator。
_PDMUX_PREFILL_TP_GROUP: Optional[GroupCoordinator] = None

_ENABLE_PDMUX_P_TP: bool = False


def set_pdmux_status(enable_prefill_multiplexing: bool):
    """切换是否让 ``get_tp_group`` 返回 PD-Multiplexing 的 prefill TP 组。

    输入输出示例：传入 ``True`` 后，``get_tp_group()`` 返回
    ``_PDMUX_PREFILL_TP_GROUP``；传入 ``False`` 后恢复返回普通 ``_TP``。无返回值。
    """
    global _ENABLE_PDMUX_P_TP
    _ENABLE_PDMUX_P_TP = enable_prefill_multiplexing


def get_tp_group() -> GroupCoordinator:
    """返回当前生效的张量并行（TP）组。

    输出示例：普通模式返回 ``_TP``；PD-Multiplexing prefill 模式返回其专用 TP 组。
    未初始化时抛出断言错误。
    """
    if _ENABLE_PDMUX_P_TP:
        assert (
            _PDMUX_PREFILL_TP_GROUP is not None
        ), "tensor model parallel group for PD-Multiplexing Prefill is not initialized"
        return _PDMUX_PREFILL_TP_GROUP
    assert _TP is not None, "tensor model parallel group is not initialized"
    return _TP


def get_attn_tp_group() -> GroupCoordinator:
    """返回注意力张量并行（Attention TP）组；未初始化时抛出断言错误。"""
    assert (
        _ATTN_TP is not None
    ), "attention tensor model parallel group is not initialized"
    return _ATTN_TP


def get_attn_cp_group() -> GroupCoordinator:
    """返回注意力上下文并行（Attention CP）组；未初始化时抛出断言错误。"""
    assert (
        _ATTN_CP is not None
    ), "attention context model parallel group is not initialized"
    return _ATTN_CP


_MOE_DP: Optional[GroupCoordinator] = None
_MOE_EP: Optional[GroupCoordinator] = None
_MOE_TP: Optional[GroupCoordinator] = None


def get_moe_dp_group() -> GroupCoordinator:
    """返回 MoE 数据并行组；例如 MoE DP=2 时其 ``world_size`` 为 2。"""
    assert _MOE_DP is not None, "moe data parallel group is not initialized"
    return _MOE_DP


def get_moe_ep_group() -> GroupCoordinator:
    """返回 MoE 专家并行组；例如 MoE EP=8 时其 ``world_size`` 为 8。"""
    assert _MOE_EP is not None, "expert model parallel group is not initialized"
    return _MOE_EP


def get_moe_tp_group() -> GroupCoordinator:
    """返回 MoE 张量并行组；例如 MoE TP=2 时其 ``world_size`` 为 2。"""
    assert _MOE_TP is not None, "expert model parallel group is not initialized"
    return _MOE_TP


# 为保持向后兼容而保留旧函数名。
get_tensor_model_parallel_group = get_tp_group

_PP: Optional[GroupCoordinator] = None


def get_pp_group() -> GroupCoordinator:
    """返回流水线并行（PP）组；例如 PP=4 时其 ``world_size`` 为 4。"""
    assert _PP is not None, "pipeline model parallel group is not initialized"
    return _PP


# 为保持向后兼容而保留旧函数名。
get_pipeline_model_parallel_group = get_pp_group


def get_mooncake_transfer_engine():
    """返回设备通信器中已初始化的共享 ``MooncakeTransferEngine``。

    若尚未初始化则返回 ``None``。该实例由解耦式 Mooncake 后端和
    ``mem_cache/mooncake_store`` 共同使用。

    输入输出示例：初始化 Mooncake 通信器后返回共享引擎对象；未启用 Mooncake 时
    返回 ``None``。
    """
    from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
        get_mooncake_transfer_engine as _get_engine,
    )

    return _get_engine()


@contextmanager
def graph_capture(stream: Optional[torch.cuda.Stream] = None):
    """为 CUDA Graph 捕获统一切换各模型并行通信组的上下文管理器。

    进入上下文时，TP、PP、MoE EP/TP 通信器会切换到支持图捕获的状态，并在独立
    stream 上执行待捕获 kernel，避免混入默认 stream 后台启动的其他 kernel。退出后
    恢复通信器状态。返回的 ``GraphCaptureContext`` 当前只保存捕获所用 stream。

    输入输出示例：
        ``with graph_capture() as ctx: model(x)``；代码块内 ``ctx.stream`` 是捕获流，
        退出代码块后通信器恢复普通执行状态。
    """
    with (
        get_tp_group().graph_capture(stream=stream) as context,
        get_pp_group().graph_capture(context),
    ):
        with contextlib.ExitStack() as stack:
            seen = {id(_TP)}
            for group in (_MOE_EP, _MOE_TP):
                if group is not None and id(group) not in seen:
                    seen.add(id(group))
                    stack.enter_context(group.graph_capture(context))
            yield context


logger = logging.getLogger(__name__)

_ENABLE_CUSTOM_ALL_REDUCE = True
_ENABLE_MSCCLPP_ALL_REDUCE = False
_ENABLE_TORCH_SYMM_MEM_ALL_REDUCE = False


def set_custom_all_reduce(enable: bool):
    global _ENABLE_CUSTOM_ALL_REDUCE
    _ENABLE_CUSTOM_ALL_REDUCE = enable


def set_mscclpp_all_reduce(enable: bool):
    global _ENABLE_MSCCLPP_ALL_REDUCE
    _ENABLE_MSCCLPP_ALL_REDUCE = enable


def set_torch_symm_mem_all_reduce(enable: bool):
    global _ENABLE_TORCH_SYMM_MEM_ALL_REDUCE
    _ENABLE_TORCH_SYMM_MEM_ALL_REDUCE = enable


# TODO：重构仓库内的平台实现，最终移除这个包装函数。
def get_default_distributed_backend(device: str) -> str:
    """返回设备类型对应的默认 PyTorch 分布式后端。

    输入输出示例：``"cuda" -> "nccl"``、``"cpu" -> "gloo"``；实际当前平台的
    返回值由平台实现决定。
    """
    # 这里刻意通过 ``platforms.current_platform`` 动态访问，而不是直接导入对象。
    # 这样每次调用都会经过 platforms 包的惰性 ``__getattr__``，从而感知运行时对
    # ``_current_platform`` 的覆盖。
    if device == platforms.current_platform.device_type:
        return platforms.current_platform.get_torch_distributed_backend_str()
    return _DEVICE_TO_DISTRIBUTED_BACKEND.get(device, "gloo")


def _create_global_tcp_store(rank: int, world_size: int) -> None:
    """创建供所有 rank 协调使用的全局 ``TCPStore``。

    该 Store 用于 NIXL 缓冲区初始化等不适合走 device collective 的控制面协作。
    rank 0 作为服务端，其余 rank 连接同一 ``MASTER_ADDR:port``。

    输入输出示例：输入 ``rank=0, world_size=8`` 时创建服务端并注册到全局状态；
    输入 ``rank=1, world_size=8`` 时连接该服务端。本函数无显式返回值。
    """
    from torch.distributed import TCPStore

    master_ip = os.environ.get("MASTER_ADDR")

    if not master_ip:
        logger.warning(
            "Could not determine master IP for global TCPStore. "
            "Broadcasting from rank 0 to all ranks."
        )

    base_store_port = envs.SGLANG_TCP_STORE_PORT.get()

    # 环境变量未提供主节点地址时，由 rank 0 获取本机 IP 并广播给所有 rank。
    # ``broadcast_object_list`` 可适配任意后端，并自动处理 CPU/GPU 对象通信。
    if not master_ip:
        if rank == 0:
            master_ip = get_local_ip_auto()
            ip_list = [master_ip]
        else:
            ip_list = [None]

        torch.distributed.broadcast_object_list(ip_list, src=0)
        master_ip = ip_list[0]

    try:
        tcp_store = TCPStore(
            host_name=master_ip,
            port=base_store_port,
            world_size=world_size,
            is_master=(rank == 0),
        )
        set_global_tcp_store(tcp_store)
        logger.info(
            "Created global TCPStore at %s:%d (rank=%d, world_size=%d)",
            master_ip,
            base_store_port,
            rank,
            world_size,
        )
    except Exception as e:
        logger.warning(
            "Failed to create global TCPStore at %s:%d: %s. "
            "Components requiring TCPStore (like NIXL) may not work.",
            master_ip,
            base_store_port,
            e,
        )


def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "nccl",
    timeout: Optional[int] = None,
    moe_a2a_backend: Optional[str] = None,
    recovered_rank: bool = False,
):
    """初始化 PyTorch 分布式环境以及 SGLang 的 WORLD 组封装。

    参数：
        world_size: 全局进程数；每个进程必须传入相同值。
        rank: 当前进程的全局 rank，范围为 ``[0, world_size)``。
        distributed_init_method: rendezvous 地址，通常为 ``env://`` 或 ``tcp://...``。
        local_rank: 当前进程在节点内的设备编号；``-1`` 时从环境或全局 rank 推导。
        backend: device 进程组后端，如 ``nccl``、``gloo``、``mooncake``。
        timeout: 进程组操作超时秒数。
        moe_a2a_backend: MoE all-to-all 后端；为 ``nixl`` 时额外创建 TCPStore。
        recovered_rank: 当前 rank 是否由弹性恢复流程重新加入。

    输入输出示例：四卡单机的每个进程分别调用
    ``init_distributed_environment(4, rank, "env://", rank, "nccl")``；调用结束后
    ``get_world_group().ranks == [0,1,2,3]``，函数无显式返回值。
    """
    logger.debug(
        "world_size=%d rank=%d local_rank=%d " "distributed_init_method=%s backend=%s",
        world_size,
        rank,
        local_rank,
        distributed_init_method,
        backend,
    )
    if "mooncake" in backend:
        try:
            from mooncake import ep as mooncake_ep
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run SGLang with Mooncake Backend."
            ) from e
        mooncake_ep.set_host_ip(get_local_ip_auto())

    if not torch.distributed.is_initialized():
        global _MODEL_PARALLEL_GROUP_TIMEOUT
        assert distributed_init_method is not None, (
            "distributed_init_method must be provided when initializing "
            "distributed environment"
        )
        if timeout is not None:
            assert isinstance(timeout, (int)), "timeout must be a number"
            assert timeout > 0, "timeout must be positive"
            timeout = timedelta(seconds=timeout)

        _MODEL_PARALLEL_GROUP_TIMEOUT = timeout

        if backend == "mooncake":
            from mooncake.ep import MooncakeBackendOptions

            # 此分支仅在 Mooncake 后端下执行，因此这里固定使用 ``cuda`` 是安全的。
            active_ranks = torch.ones(world_size, dtype=torch.int32, device="cuda")
            pg_options = MooncakeBackendOptions(active_ranks, recovered_rank)
        else:
            pg_options = get_torch_distributed_pg_options()

        # 这里的 backend 用于创建包含全部 rank 的 WORLD 进程组。
        torch.distributed.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
            timeout=timeout,
            pg_options=pg_options,
        )

        # NIXL 需要额外的全局 TCPStore 完成各 rank 间的控制面协调。
        if moe_a2a_backend == "nixl":
            _create_global_tcp_store(rank, world_size)

    # 设置节点内 local rank。PyTorch ProcessGroup 本身不提供 local_rank，
    # 参见 https://github.com/pytorch/pytorch/issues/122816。
    if local_rank == -1:
        # 未显式设置 local rank 通常意味着单节点部署，此时可用全局 rank 作为 local rank。
        if distributed_init_method == "env://":
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        else:
            local_rank = rank
    global _WORLD
    if _WORLD is None:
        ranks = list(range(torch.distributed.get_world_size()))
        _WORLD = init_world_group(
            ranks, local_rank, backend, recovered_rank=recovered_rank
        )
    else:
        assert (
            _WORLD.world_size == torch.distributed.get_world_size()
        ), "world group already initialized with a different world size"


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    attention_data_parallel_size: int = 1,
    attention_context_model_parallel_size: int = 1,
    moe_data_model_parallel_size: int = 1,
    backend: Optional[str] = None,
    duplicate_tp_group: bool = False,
    enable_symm_mem: bool = False,
    recovered_rank: bool = False,
) -> None:
    """
    初始化模型并行相关的各类进程组。

    本函数在 `world` 组（全部 rank）之上，按照传入的各并行度切分出
    张量并行（TP）、注意力上下文并行（Attention CP）、注意力张量并行（Attention TP）、
    MoE 数据并行（MoE DP）、MoE 专家并行（MoE EP）、MoE 张量并行（MoE TP）
    以及流水线并行（PP）等进程组，并保存到对应的全局变量中。

    参数：
        tensor_model_parallel_size: 张量并行使用的 GPU 数（即外层 TP 规模，也是
            一个模型实例内除 PP 外的 rank 域大小）。
        expert_model_parallel_size: MoE 专家并行（EP）使用的 GPU 数。
        pipeline_model_parallel_size: 流水线并行（PP）使用的 GPU 数。
        attention_data_parallel_size: 注意力数据并行（DP Attention）使用的 GPU 数。
        attention_context_model_parallel_size: 注意力上下文并行（CP）使用的 GPU 数。
        moe_data_model_parallel_size: MoE 数据并行使用的 GPU 数。
        backend: 进程组通信后端；为 None 时沿用 world 组的后端。
        duplicate_tp_group: 是否为 PD 分离多路复用（PD-Multiplexing）的 prefill
            额外复制一份 TP 组。
        enable_symm_mem: 是否启用对称内存（影响 Attention TP 是否走 pynccl）。
        recovered_rank: 弹性场景下标记该 rank 是否为恢复出来的 rank。

    分解关系（均需整除）：
        attn_tp_size = tensor_model_parallel_size // attn_cp_size // attn_dp_size
        moe_tp_size  = tensor_model_parallel_size // moe_ep_size  // moe_dp_size

    ------------------------------------------------------------------
    示例 1（TP + PP）：
        输入：world_size=8，tensor_model_parallel_size=2，
              pipeline_model_parallel_size=4，其余为 1。
        输出分组：
            4 个张量并行组（TP）：
                [g0, g1], [g2, g3], [g4, g5], [g6, g7]
            2 个流水线并行组（PP）：
                [g0, g2, g4, g6], [g1, g3, g5, g7]

    示例 2（Attention CP/DP + MoE DP/EP）：
        输入：world_size=8，tensor_model_parallel_size=4，
              attn_cp_size=2、attn_tp_size=4（attn_dp_size=1），
              moe_dp_size=2、moe_ep_size=4（moe_tp_size=1）。
        输出分组：
            2 个张量并行组（TP）：
                [g0, g1, g2, g3], [g4, g5, g6, g7]
            4 个注意力上下文并行组（Attention CP）：
                [g0, g4], [g1, g5], [g2, g6], [g3, g7]
            2 个 MoE 专家并行组（MoE EP）：
                [g0, g1, g2, g3], [g4, g5, g6, g7]
            4 个 MoE 数据并行组（MoE DP）：
                [g0, g4], [g1, g5], [g2, g6], [g3, g7]

    注意：为了通信效率，调用方应保证相邻 rank 位于同一台 DGX 机器上。例如使用
    2 台各 8 卡、共 16 卡的机器时，rank 0~7 属于第一台，rank 8~15 属于第二台。

    返回：无（结果写入 `_TP`、`_ATTN_CP`、`_ATTN_TP`、`_MOE_DP`、`_MOE_EP`、
    `_MOE_TP`、`_PP` 等全局进程组变量）。
    """
    # 获取 world size，并做基本一致性校验。
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    # 一个模型实例的 rank 域大小 = TP × PP，必须与总卡数一致。
    if world_size != tensor_model_parallel_size * pipeline_model_parallel_size:
        raise RuntimeError(
            f"world_size ({world_size}) is not equal to "
            f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
            f"pipeline_model_parallel_size ({pipeline_model_parallel_size})"
        )

    # 构建张量并行（TP）组：把连续的 tensor_model_parallel_size 个 rank 归为一组。
    # 例如 world_size=8、TP=2 时得到 [0,1],[2,3],[4,5],[6,7]。
    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    global _TP
    assert _TP is None, "tensor model parallel group is already initialized"
    group_ranks = []
    for tp_group_idx in range(num_tensor_model_parallel_groups):
        ranks = list(
            range(
                tp_group_idx * tensor_model_parallel_size,
                (tp_group_idx + 1) * tensor_model_parallel_size,
            )
        )
        group_ranks.append(ranks)

    # 消息队列广播器（message queue broadcaster）只在张量并行组内使用。
    _TP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=envs.SGLANG_USE_MESSAGE_QUEUE_BROADCASTER.get(),
        group_name="tp",
        recovered_rank=recovered_rank,
    )

    # PD 分离多路复用：为 prefill 复制一份与 TP 相同 rank 划分的独立进程组，
    # 使 prefill 与 decode 的 TP 通信互不干扰。
    if duplicate_tp_group:
        global _PDMUX_PREFILL_TP_GROUP
        assert (
            _PDMUX_PREFILL_TP_GROUP is None
        ), "tensor model parallel group for PD-Multiplexing Prefill is already initialized"
        _PDMUX_PREFILL_TP_GROUP = init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            use_message_queue_broadcaster=envs.SGLANG_USE_MESSAGE_QUEUE_BROADCASTER.get(),
            group_name="pdmux_prefill_tp",
            recovered_rank=recovered_rank,
        )
        if _TP.pynccl_comm:
            _TP.pynccl_comm.disabled = False
            _PDMUX_PREFILL_TP_GROUP.pynccl_comm.disabled = False

    # 注意力侧的三个并行度：DP（按 token/请求切）、CP（按序列切）、TP（按 head/张量切）。
    # 三者相乘应等于外层 TP 规模，故 attn_tp_size 由整除推出。
    attn_dp_size = attention_data_parallel_size
    attn_cp_size = attention_context_model_parallel_size
    attn_tp_size = tensor_model_parallel_size // attn_cp_size // attn_dp_size

    # 构建注意力上下文并行（Attention CP）组。
    # 若 CP 规模等于外层 TP，则整组共享，直接复用 _TP，避免新建冗余组。
    global _ATTN_CP
    assert (
        _ATTN_CP is None
    ), "attention context model parallel group is already initialized"
    if attn_cp_size == tensor_model_parallel_size:
        _ATTN_CP = _TP
    else:
        # 在每个 TP 组内，固定 (dp_idx, attn_tp_idx)，以 attn_tp_size 为步长跨越
        # CP 维度收集 rank，得到一个 CP 组；因此同一 CP 组内的 rank 拥有相同的
        # DP/TP 坐标，只在序列切分（CP）维度上不同。
        group_ranks = []
        for tp_group_idx in range(num_tensor_model_parallel_groups):
            for dp_idx in range(attn_dp_size):
                for attn_tp_idx in range(attn_tp_size):
                    st = (
                        tp_group_idx * tensor_model_parallel_size
                        + dp_idx * attn_tp_size * attn_cp_size
                        + attn_tp_idx
                    )
                    en = (
                        tp_group_idx * tensor_model_parallel_size
                        + (dp_idx + 1) * attn_tp_size * attn_cp_size
                        + attn_tp_idx
                    )
                    ranks = list(range(st, en, attn_tp_size))
                    group_ranks.append(ranks)
        # 备注：上方 group_ranks 在 else 分支起始处已重置为空列表。
        _ATTN_CP = init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            use_message_queue_broadcaster=envs.SGLANG_USE_MESSAGE_QUEUE_BROADCASTER.get(),
            group_name="attn_cp",
            recovered_rank=recovered_rank,
        )

    from sglang.srt.layers.sampler import SYNC_TOKEN_IDS_ACROSS_TP

    # 构建注意力张量并行（Attention TP）组。
    # 若 attn_tp_size 等于外层 TP，则整组即一个 Attention TP 组，直接复用 _TP。
    global _ATTN_TP
    assert (
        _ATTN_TP is None
    ), "attention tensor model parallel group is already initialized"
    if attn_tp_size == tensor_model_parallel_size:
        _ATTN_TP = _TP
    else:
        # 每个 TP 组内，把连续的 attn_tp_size 个 rank 归为一个 Attention TP 组；
        # (cp, dp) 的每种组合各占一段连续区间。
        group_ranks = []
        for tp_group_idx in range(num_tensor_model_parallel_groups):
            for cp_dp_combined_idx in range(attn_cp_size * attn_dp_size):
                st = (
                    tp_group_idx * tensor_model_parallel_size
                    + cp_dp_combined_idx * attn_tp_size
                )
                en = (
                    tp_group_idx * tensor_model_parallel_size
                    + (cp_dp_combined_idx + 1) * attn_tp_size
                )
                ranks = list(range(st, en))
                group_ranks.append(ranks)

        _ATTN_TP = init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            use_pynccl=SYNC_TOKEN_IDS_ACROSS_TP or enable_symm_mem,
            use_mscclpp_allreduce=False,
            use_custom_allreduce=False,
            use_torch_symm_mem_allreduce=False,
            use_message_queue_broadcaster=envs.SGLANG_USE_MESSAGE_QUEUE_BROADCASTER.get(),
            group_name="attention_tp",
            recovered_rank=recovered_rank,
        )

    # MoE 侧的三个并行度：DP（按 token 切）、EP（按专家切）、TP（切单个专家的张量维）。
    # 三者相乘应等于外层 TP 规模，故 moe_tp_size 由整除推出。
    moe_ep_size = expert_model_parallel_size
    moe_dp_size = moe_data_model_parallel_size
    moe_tp_size = tensor_model_parallel_size // moe_ep_size // moe_dp_size

    # 构建 MoE 数据并行（MoE DP）组。
    global _MOE_DP
    assert _MOE_DP is None, "moe data parallel group is already initialized"
    if attn_cp_size > moe_dp_size:
        # 当 moe_dp_size < attn_cp_size 时，CP 的各 rank 在进入 MoE 前必须先共享 token。
        # 此时让 MoE DP 组直接复用 CP 组（_ATTN_CP），从而由已有的 DP allgather/scatter
        # 承担这部分 token 共享，无需再新建组。
        _MOE_DP = _ATTN_CP
    elif moe_dp_size == tensor_model_parallel_size:
        # MoE DP 覆盖整个 TP 组时，直接复用 _TP。
        _MOE_DP = _TP
    else:
        # 在每个 TP 组内，固定 (moe_tp, moe_ep) 的组合坐标 tp_ep_combined_idx，
        # 以 moe_tp_size*moe_ep_size 为步长跨 DP 维度收集 rank，得到一个 MoE DP 组。
        group_ranks = []
        for tp_group_idx in range(num_tensor_model_parallel_groups):
            for tp_ep_combined_idx in range(moe_tp_size * moe_ep_size):
                st = tp_group_idx * tensor_model_parallel_size + tp_ep_combined_idx
                en = (
                    tp_group_idx + 1
                ) * tensor_model_parallel_size + tp_ep_combined_idx
                ranks = list(range(st, en, moe_tp_size * moe_ep_size))
                group_ranks.append(ranks)
        _MOE_DP = init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            group_name="moe_dp",
            recovered_rank=recovered_rank,
        )

    # 构建 MoE 专家并行（MoE EP）组。
    # 若 moe_ep_size 等于外层 TP，则整组即一个 EP 组，直接复用 _TP。
    global _MOE_EP
    assert _MOE_EP is None, "expert model parallel group is already initialized"
    if moe_ep_size == tensor_model_parallel_size:
        _MOE_EP = _TP
    else:
        # 在每个 TP 组内，固定 (moe_dp_idx, moe_tp_idx)，以 moe_tp_size 为步长跨
        # EP 维度收集 rank，得到一个 EP 组：组内 rank 拥有相同的 DP/TP 坐标，
        # 只在专家切分（EP）维度上不同。
        group_ranks = []
        for tp_group_idx in range(num_tensor_model_parallel_groups):
            for moe_dp_idx in range(moe_dp_size):
                for moe_tp_idx in range(moe_tp_size):
                    st = (
                        tp_group_idx * tensor_model_parallel_size
                        + moe_dp_idx * moe_ep_size * moe_tp_size
                        + moe_tp_idx
                    )
                    en = st + moe_ep_size * moe_tp_size
                    ranks = list(range(st, en, moe_tp_size))
                    group_ranks.append(ranks)
        _MOE_EP = init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            use_pynccl=False,
            use_custom_allreduce=False,
            group_name="moe_ep",
            recovered_rank=recovered_rank,
        )

    # 构建 MoE 张量并行（MoE TP）组：用于切分单个专家的张量维。
    # 若 moe_tp_size 等于外层 TP，则整组即一个 MoE TP 组，直接复用 _TP。
    global _MOE_TP
    assert _MOE_TP is None, "expert model parallel group is already initialized"
    if moe_tp_size == tensor_model_parallel_size:
        _MOE_TP = _TP
    else:
        # 每个 TP 组内，把连续的 moe_tp_size 个 rank 归为一个 MoE TP 组；
        # (ep, dp) 的每种组合各占一段连续区间。
        group_ranks = []
        for tp_group_idx in range(num_tensor_model_parallel_groups):
            for ep_dp_combined_idx in range(moe_ep_size * moe_dp_size):
                st = (
                    tp_group_idx * tensor_model_parallel_size
                    + ep_dp_combined_idx * moe_tp_size
                )
                en = (
                    tp_group_idx * tensor_model_parallel_size
                    + (ep_dp_combined_idx + 1) * moe_tp_size
                )
                ranks = list(range(st, en))
                group_ranks.append(ranks)
        _MOE_TP = init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            use_pynccl=False,
            use_custom_allreduce=False,
            group_name="moe_tp",
            recovered_rank=recovered_rank,
        )

    # 构建流水线并行（PP）组：以 num_pipeline_model_parallel_groups 为步长跨越整个
    # world 采样 rank，使同一 PP 组内的 rank 分属不同流水线阶段。
    # 例如 world_size=8、PP=4 时得到 [0,2,4,6],[1,3,5,7]。
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    global _PP
    assert _PP is None, "pipeline model parallel group is already initialized"
    group_ranks = []
    for pp_group_idx in range(num_pipeline_model_parallel_groups):
        ranks = list(
            range(pp_group_idx, world_size, num_pipeline_model_parallel_groups)
        )
        group_ranks.append(ranks)
    # 流水线并行不需要自定义 all-reduce。
    _PP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_custom_allreduce=False,
        group_name="pp",
        recovered_rank=recovered_rank,
    )


def create_custom_parallel_group(
    group_ranks: List[int], backend: str = "gloo"
) -> Optional[torch.distributed.ProcessGroup]:
    """
    根据提供的 rank 列表创建一个自定义并行组。

    各 rank 会先通过 all_gather_object 汇总彼此想加入的组，去重后按统一顺序
    逐个创建（torch.distributed.new_group 要求所有 rank 以相同顺序集体调用），
    最后只返回当前 rank 所属的那个组。

    参数：
        group_ranks: 当前进程想加入的 rank 列表
                     （例如 rank 0 传入 [0...7]，rank 8 传入 [8...15]）。
        backend: 通信后端（默认 "gloo"）。

    返回：
        若当前 rank 在 group_ranks 中，则返回对应 ProcessGroup；否则返回 None。

    示例（world_size=16）：
        rank 0~7   传入 [0..7]  -> 得到含 [0..7]  的组；
        rank 8~15  传入 [8..15] -> 得到含 [8..15] 的组。
    """
    assert torch.distributed.is_initialized()

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    local_config = sorted(list(set(group_ranks)))
    gathered_configs = [None for _ in range(world_size)]

    torch.distributed.all_gather_object(gathered_configs, local_config)

    unique_groups = []
    seen_signatures = set()

    for config in gathered_configs:
        config_tuple = tuple(config)
        if config_tuple not in seen_signatures:
            seen_signatures.add(config_tuple)
            unique_groups.append(list(config_tuple))

    unique_groups.sort(key=lambda x: x[0])

    my_new_group = None

    for g_ranks in unique_groups:
        group = torch.distributed.new_group(ranks=g_ranks, backend=backend)

        if set(g_ranks) == set(local_config):
            my_new_group = group
            logger.debug(
                f"Rank {rank} successfully created/joined custom group: {g_ranks}"
            )

    # 仅返回当前 rank 所在的那个组（其余组已在各自的 rank 上创建）。
    return my_new_group


def ensure_model_parallel_initialized(
    tensor_model_parallel_size: int,
    expert_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    backend: Optional[str] = None,
) -> None:
    """确保模型并行组已按期望规模初始化。

    尚未初始化时调用 ``initialize_model_parallel``；已初始化时校验 TP 与 PP 规模，
    不匹配则抛出断言错误。

    输入输出示例：在 ``world_size=8`` 下传入 ``TP=2, EP=1, PP=4``，首次调用创建
    对应进程组，后续相同参数调用仅校验状态；函数无返回值。
    """
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)
    if not model_parallel_is_initialized():
        initialize_model_parallel(
            tensor_model_parallel_size,
            expert_model_parallel_size,
            pipeline_model_parallel_size,
            backend,
        )
        return

    assert get_tensor_model_parallel_world_size() == tensor_model_parallel_size, (
        "tensor parallel group already initialized, but of unexpected size: "
        f"{get_tensor_model_parallel_world_size()=} vs. "
        f"{tensor_model_parallel_size=}"
    )
    pp_world_size = get_pp_group().world_size
    assert pp_world_size == pipeline_model_parallel_size, (
        "pipeline parallel group already initialized, but of unexpected size: "
        f"{pp_world_size=} vs. "
        f"{pipeline_model_parallel_size=}"
    )


def model_parallel_is_initialized():
    """检查张量并行组与流水线并行组是否已初始化。"""
    return _TP is not None and _PP is not None


_TP_STATE_PATCHED = False


@contextmanager
def patch_tensor_parallel_group(tp_group: GroupCoordinator):
    """临时替换（patch）当前 TP 组，直到本上下文管理器退出时恢复。

    用于推投解码（speculative decoding）的 draft worker：让 draft 模型以与
    target 模型不同的 TP 度运行。

    参数：
        tp_group (GroupCoordinator): 用于临时生效的 TP 组协调器。
    """
    global _TP_STATE_PATCHED
    assert not _TP_STATE_PATCHED, "Should not call when it's already patched"

    _TP_STATE_PATCHED = True
    old_tp_group = get_tp_group()
    global _TP
    _TP = tp_group
    try:
        yield
    finally:
        # 恢复为原来的状态。
        _TP_STATE_PATCHED = False
        _TP = old_tp_group


def get_world_size():
    """返回 world 组的进程数（world size）。"""
    return get_world_group().world_size


def get_world_rank():
    """返回当前进程在 world 组内的 rank。"""
    return get_world_group().rank_in_group


def get_tensor_model_parallel_world_size():
    """返回张量并行组（TP）的进程数。"""
    return get_tp_group().world_size


def get_tensor_model_parallel_rank():
    """返回当前进程在张量并行组（TP）内的 rank。"""
    return get_tp_group().rank_in_group


# 注意力张量并行（Attention TP）查询接口。
def get_attn_tensor_model_parallel_world_size():
    """返回注意力张量并行组（Attention TP）的进程数。"""
    return get_attn_tp_group().world_size


def get_attn_tensor_model_parallel_rank():
    """返回当前进程在注意力张量并行组（Attention TP）内的 rank。"""
    return get_attn_tp_group().rank_in_group


# 注意力上下文并行（Attention CP）查询接口。
def get_attn_context_model_parallel_world_size():
    """返回注意力上下文并行组（Attention CP）的进程数。"""
    return get_attn_cp_group().world_size


def get_attn_context_model_parallel_rank():
    """返回当前进程在注意力上下文并行组（Attention CP）内的 rank。"""
    return get_attn_cp_group().rank_in_group


def get_pipeline_model_parallel_world_size():
    """返回流水线并行组（PP）的进程数。"""
    return get_pp_group().world_size


def get_pipeline_model_parallel_rank():
    """返回当前进程在流水线并行组（PP）内的 rank。"""
    return get_pp_group().rank_in_group


# MoE 数据并行（MoE DP）查询接口。
def get_moe_data_parallel_world_size():
    """返回 MoE 数据并行组（MoE DP）的进程数。"""
    return get_moe_dp_group().world_size


def get_moe_data_parallel_rank():
    """返回当前进程在 MoE 数据并行组（MoE DP）内的 rank。"""
    return get_moe_dp_group().rank_in_group


# MoE 专家并行（MoE EP）查询接口。
def get_moe_expert_parallel_world_size():
    """返回 MoE 专家并行组（MoE EP）的进程数。"""
    return get_moe_ep_group().world_size


def get_moe_expert_parallel_rank():
    """返回当前进程在 MoE 专家并行组（MoE EP）内的 rank。"""
    return get_moe_ep_group().rank_in_group


# MoE 张量并行（MoE TP）查询接口。
def get_moe_tensor_parallel_world_size():
    """返回 MoE 张量并行组（MoE TP）的进程数。"""
    return get_moe_tp_group().world_size


def get_moe_tensor_parallel_rank():
    """返回当前进程在 MoE 张量并行组（MoE TP）内的 rank。"""
    return get_moe_tp_group().rank_in_group


def destroy_model_parallel():
    """销毁所有模型并行组并清空全局引用。

    输入输出示例：初始化 TP/PP/EP 后调用本函数，随后 ``model_parallel_is_initialized()``
    返回 ``False``；函数无返回值。共享/别名进程组会避免重复销毁。
    """
    global _TP
    if _TP:
        _TP.destroy()
    _TP = None

    global _PP
    if _PP:
        _PP.destroy()
    _PP = None

    global _MOE_EP
    if _MOE_EP:
        _MOE_EP.destroy()
    _MOE_EP = None

    global _MOE_TP
    if _MOE_TP:
        _MOE_TP.destroy()
    _MOE_TP = None

    global _ATTN_CP
    global _MOE_DP
    # 先销毁 _MOE_DP 再销毁 _ATTN_CP，因为 _MOE_DP 可能是 _ATTN_CP 的别名（同一对象）。
    # 只有当 _MOE_DP 不是其他组的别名时，才真正销毁它，避免重复销毁。
    if _MOE_DP and _MOE_DP is not _ATTN_CP and _MOE_DP is not _TP:
        _MOE_DP.destroy()
    _MOE_DP = None
    if _ATTN_CP:
        _ATTN_CP.destroy()
    _ATTN_CP = None

    global _ATTN_TP
    if _ATTN_TP:
        _ATTN_TP.destroy()
    _ATTN_TP = None

    global _PDMUX_PREFILL_TP_GROUP
    if _PDMUX_PREFILL_TP_GROUP:  # type: ignore[union-attr]
        _PDMUX_PREFILL_TP_GROUP.destroy()
    _PDMUX_PREFILL_TP_GROUP = None


def destroy_distributed_environment():
    """销毁 WORLD 组和 PyTorch 默认进程组，并重置模型并行超时配置。

    输入输出示例：调用后 ``torch.distributed.is_initialized()`` 为 ``False``，
    ``_WORLD`` 为 ``None``；重复调用也是安全的。
    """
    global _WORLD, _MODEL_PARALLEL_GROUP_TIMEOUT
    if _WORLD:
        _WORLD.destroy()
    _WORLD = None
    _MODEL_PARALLEL_GROUP_TIMEOUT = None
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def cleanup_dist_env_and_memory(shutdown_ray: bool = False):
    """清理分布式环境、可选关闭 Ray，并回收主机与设备缓存。

    输入输出示例：``cleanup_dist_env_and_memory(shutdown_ray=True)`` 会依次销毁模型并行
    与 WORLD 组、关闭 Ray、执行 Python GC，并清理当前硬件后端的设备缓存。无返回值。
    """
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    if shutdown_ray:
        import ray  # 延迟导入 Ray

        ray.shutdown()
    gc.collect()
    if not _is_cpu:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch._C, "_host_emptyCache"):
                torch._C._host_emptyCache()
            else:
                logger.warning(
                    "torch._C._host_emptyCache() only available in Pytorch >=2.5"
                )
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
        elif hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.empty_cache()
        elif hasattr(torch, "musa") and torch.musa.is_available():
            torch.musa.empty_cache()


def in_the_same_node_as(pg: ProcessGroup, source_rank: int = 0) -> List[bool]:
    """
    一个集合操作：返回每个 rank 是否与 source_rank 位于同一节点。它通过“能否
    共享同一块共享内存”来判断进程是否附着在同一个内存系统上。

    返回：长度为 world_size 的 List[bool]，第 i 位为 True 表示组内 rank i 与
    source_rank 在同一节点。

    输入输出示例：四个 rank 中 0、1 位于节点 A，2、3 位于节点 B，调用
    ``in_the_same_node_as(pg, source_rank=0)`` 后所有 rank 均返回
    ``[True, True, False, False]``。
    """
    assert (
        torch.distributed.get_backend(pg) != torch.distributed.Backend.NCCL
    ), "in_the_same_node_as should be tested with a non-NCCL group."
    # 组内的本地 rank。
    rank = torch.distributed.get_rank(group=pg)
    world_size = torch.distributed.get_world_size(group=pg)

    # 每个进程各自保存结果的本地张量。
    is_in_the_same_node = torch.tensor([0] * world_size, dtype=torch.int32)

    # 组内各进程的全局 rank。
    ranks = torch.distributed.get_process_group_ranks(pg)

    magic_message = b"magic_message"
    shm = None

    try:
        with contextlib.suppress(OSError):
            if rank == source_rank:
                # 创建一块共享内存段。
                shm = shared_memory.SharedMemory(create=True, size=128)
                shm.buf[: len(magic_message)] = magic_message
                torch.distributed.broadcast_object_list(
                    [shm.name], src=ranks[source_rank], group=pg
                )
                is_in_the_same_node[rank] = 1
            else:
                # 尝试打开那块共享内存段。
                recv = [None]
                torch.distributed.broadcast_object_list(
                    recv, src=ranks[source_rank], group=pg
                )
                name = recv[0]
                # 修复 https://stackoverflow.com/q/62748654/9191338：
                # 即使共享内存不是本进程创建的，Python 也会错误地去追踪它。
                # 下面的 patch 是一个规避方法。
                with patch(
                    "multiprocessing.resource_tracker.register",
                    lambda *args, **kwargs: None,
                ):
                    shm = shared_memory.SharedMemory(name=name)
                if shm.buf[: len(magic_message)] == magic_message:
                    is_in_the_same_node[rank] = 1
    except Exception as e:
        logger.error("Error ignored in is_in_the_same_node: %s", e)
    finally:
        if shm:
            shm.close()

    torch.distributed.barrier(group=pg)

    # 所有 rank 完成探测后，由源 rank 删除共享内存段。
    with contextlib.suppress(OSError):
        if rank == source_rank and shm:
            shm.unlink()
    torch.distributed.all_reduce(is_in_the_same_node, group=pg)

    return [x == 1 for x in is_in_the_same_node.tolist()]


vllm_get_pp_group = None
vllm_get_tp_group = None
vllm_get_world_group = None


def monkey_patch_vllm_parallel_state(reverse: bool = False):
    """让 vLLM 的并行组查询函数临时复用 SGLang 的实现，或恢复原实现。

    输入输出示例：传入 ``False`` 后，``vllm.get_tp_group`` 指向本模块的
    ``get_tp_group``；传入 ``True`` 后恢复首次 patch 前保存的 vLLM 函数。若未安装
    vLLM 则直接返回。本函数无显式返回值。
    """
    try:
        import vllm.distributed.parallel_state as vllm_parallel_state
    except ImportError:
        return

    global vllm_get_pp_group, vllm_get_tp_group, vllm_get_world_group
    if vllm_get_pp_group is None:
        vllm_get_pp_group = vllm_parallel_state.get_pp_group
        vllm_get_tp_group = vllm_parallel_state.get_tp_group
        vllm_get_world_group = vllm_parallel_state.get_world_group
    if reverse:
        setattr(vllm_parallel_state, "get_pp_group", vllm_get_pp_group)
        setattr(vllm_parallel_state, "get_tp_group", vllm_get_tp_group)
        setattr(vllm_parallel_state, "get_world_group", vllm_get_world_group)
    else:
        setattr(vllm_parallel_state, "get_pp_group", get_pp_group)
        setattr(vllm_parallel_state, "get_tp_group", get_tp_group)
        setattr(vllm_parallel_state, "get_world_group", get_world_group)
