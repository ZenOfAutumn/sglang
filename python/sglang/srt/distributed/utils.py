# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/utils.py

# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/utils.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import dataclasses
import logging
import os
import pickle
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Sequence, Tuple

import torch
from torch.distributed import TCPStore

from sglang.srt.runtime_context import get_resources

logger = logging.getLogger(__name__)


def set_global_tcp_store(store: TCPStore) -> None:
    """Install the shared TCPStore created during distributed initialization;
    the handle lives on ``ctx.resources``."""

    get_resources().tcp_store = store
    logger.info("Global TCPStore has been set")


def get_global_tcp_store() -> Optional[TCPStore]:
    """获取已存在的全局 TCPStore。

    提供对「分布式初始化时创建的那个共享 TCPStore」的访问入口。
    所有组件（例如 NIXL buffer）都应使用这同一个 store 来做协调，
    以保证大家看到的是同一份元数据。

    返回:
        全局 TCPStore 实例；若尚未初始化则返回 None（并打印一条 warning）。

    示例:
        >>> store = get_global_tcp_store()
        >>> if store is not None:
        ...     store.set("ready", b"1")     # 写入一个键
        ...     val = store.get("ready")      # 其他进程可读到 b"1"
    """

    store = get_resources().tcp_store
    if store is None:
        logger.warning(
            "Global TCPStore not found. Make sure init_distributed_environment "
            "was called with a tcp:// init method."
        )
    return store


def ensure_divisibility(numerator, denominator):
    """断言 numerator 能被 denominator 整除，否则抛 AssertionError。

    张量并行里大量维度切分（如 hidden_size / tp_size）都要求整除，
    这里做前置校验，把「除不尽」这种配置错误尽早暴露出来。

    参数:
        numerator: 被除数，例如 hidden_size = 4096。
        denominator: 除数，例如 tp_size = 8。

    示例:
        >>> ensure_divisibility(4096, 8)   # 通过，无返回
        >>> ensure_divisibility(4096, 7)   # AssertionError: 4096 is not divisible by 7
    """
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def divide(numerator, denominator):
    """先校验整除，再返回整数商 numerator // denominator。

    相当于「带整除校验的整数除法」，避免出现除不尽被静默截断的隐患。

    参数:
        numerator: 被除数，例如 4096。
        denominator: 除数，例如 8。

    返回:
        整数商。

    示例:
        >>> divide(4096, 8)   # 返回 512（每个 TP rank 分到的维度大小）
        512
    """
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> Sequence[torch.Tensor]:
    """沿张量的「最后一维」等分成 num_partitions 份。

    张量并行中常用来把一个大矩阵/激活按列切给多个 rank
    （例如把 QKV 投影的输出按 head 维切分）。

    参数:
        tensor: 输入张量，其最后一维大小必须能被 num_partitions 整除。
        num_partitions: 切分份数（通常等于 tp_size）。
        contiguous_split_chunks: 是否让每个分片在内存中连续。
            torch.split 默认返回的是「视图」（共享底层存储、非连续），
            若下游需要连续内存（如某些 kernel、跨进程发送）应置为 True。

    返回:
        分片张量组成的序列，长度为 num_partitions。

    示例:
        >>> x = torch.arange(2 * 6).reshape(2, 6)   # 形状 (2, 6)，最后一维 = 6
        >>> parts = split_tensor_along_last_dim(x, num_partitions=3)
        >>> len(parts), parts[0].shape             # 切成 3 份，每份最后一维 = 6/3 = 2
        (3, torch.Size([2, 2]))
    """
    # 取最后一维的下标，以及每份在该维上的大小（此处会校验整除）。
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # 执行切分。
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # 注意：torch.split 默认不会生成连续张量（返回的是视图）。
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def get_pp_indices(
    num_hidden_layers: int, pp_rank: int, pp_size: int
) -> Tuple[int, int]:
    """计算「当前 PP rank 负责哪一段连续的 decoder layer」。

    流水线并行（PP）按 layer 切模型：把 num_hidden_layers 层切成 pp_size 段，
    本函数返回本 rank 负责的左闭右开区间 [start_layer, end_layer)。
    默认尽量均分；若层数不能被 pp_size 整除，则把多出来的 N 层（N = 余数）
    分给「最后 N 个」stage（末尾 stage 各多 1 层）。

    也可用环境变量 SGLANG_PP_LAYER_PARTITION="a,b,c,..." 手动指定每段层数
    （逗号分隔，元素个数须等于 pp_size，之和须等于 num_hidden_layers），
    用于各卡算力/显存不均衡时做非均匀切分。

    参数:
        num_hidden_layers: 模型总层数 L，例如 32。
        pp_rank: 当前 stage 的编号，取值 [0, pp_size)。
        pp_size: PP stage 总数 p。

    返回:
        (start_layer, end_layer) 左闭右开区间；本 rank 负责的层数 = end - start。

    示例:
        # 例 1：均分。32 层、4 个 stage，每段 8 层。
        >>> get_pp_indices(32, 0, 4)   # (0, 8)
        >>> get_pp_indices(32, 3, 4)   # (24, 32)

        # 例 2：不整除。34 层、4 个 stage，余数 = 2，最后 2 个 stage 各多 1 层
        #        → 各段层数为 [8, 8, 9, 9]。
        >>> get_pp_indices(34, 0, 4)   # (0, 8)    普通段
        >>> get_pp_indices(34, 1, 4)   # (8, 16)   普通段
        >>> get_pp_indices(34, 2, 4)   # (16, 25)  多 1 层
        >>> get_pp_indices(34, 3, 4)   # (25, 34)  多 1 层

        # 例 3：手动分区。设 SGLANG_PP_LAYER_PARTITION="6,10,10,6"（和须 = 32）
        >>> get_pp_indices(32, 1, 4)   # (6, 16)   本段 10 层
    """
    # 环境变量未设置时为 None，走下面的自动均分逻辑。
    partition_list_str = os.getenv("SGLANG_PP_LAYER_PARTITION", None)
    if pp_size == 1:
        # A singleton draft PP group owns every layer and must ignore the target's
        # process-global pipeline partition list.
        partition_list_str = None
    if partition_list_str is not None:
        # —— 分支 A：手动指定各段层数 ——
        try:
            partitions = [int(layer) for layer in partition_list_str.split(",")]
        except ValueError as err:
            raise ValueError(
                "Invalid partition string: {}".format(partition_list_str)
            ) from err
        # 段数必须与 pp_size 一致，否则某些 rank 无对应配置。
        if len(partitions) != pp_size:
            raise ValueError(f"{len(partitions)=} does not match {pp_size=}.")
        # 各段之和必须等于总层数，否则会漏层或越界。
        if sum(partitions) != num_hidden_layers:
            raise ValueError(f"{sum(partitions)=} does not match {num_hidden_layers=}.")
        # 起点 = 前面所有段的层数之和；终点 = 起点 + 本段层数。
        start_layer = sum(partitions[:pp_rank])
        end_layer = start_layer + partitions[pp_rank]
    else:
        # —— 分支 B：自动均分（默认）——
        base_layers = num_hidden_layers // pp_size  # 每段至少有的层数
        remainder = num_hidden_layers % pp_size  # 多出来、需要摊派的层数
        # 把多出来的 remainder 层分给「最后 remainder 个」stage。
        if pp_rank >= pp_size - remainder:
            # 本 stage 属于「多 1 层」的末尾组。
            # 前面有 (pp_size - remainder) 个普通段（各 base_layers 层），
            # 减去它是为了扣掉「本段之前那些普通段少算的那 1 层偏移」。
            partitions_without_extra_layer = pp_size - remainder
            start_layer = pp_rank * (base_layers + 1) - partitions_without_extra_layer
            end_layer = start_layer + (base_layers + 1)
        else:
            # 本 stage 属于普通段，只拿 base_layers 层。
            start_layer = pp_rank * base_layers
            end_layer = start_layer + base_layers

    return (start_layer, end_layer)


@dataclasses.dataclass
class StatelessProcessGroup:
    """「无状态」进程组：持有一个元数据 store 以及本组的 rank、world_size。

    仅用于在进程间交换「元数据」（少量、可 pickle 的 Python 对象），
    底层走 store（TCPStore）的 KV 读写，而非 NCCL。真正的数据面通信
    （传大张量）请另建 NCCL 相关对象。

    「无状态」的含义见 create() 的说明：它不像 torch.distributed.init_process_group
    那样污染全局状态，因此同一批进程可以自由地组成多个不同的组。

    通信语义靠「每对 (src, dst) 各自维护一个自增计数器」实现：发送方把对象
    写到形如 send_to/{dst}/{seq} 的键，接收方按同样的序号去读，从而保证
    同一对进程间的多次收发严格按顺序一一配对（类似带序号的邮箱）。
    """

    rank: int  # 本进程在组内的编号
    world_size: int  # 组内进程总数
    store: torch._C._distributed_c10d.Store  # 底层 KV 存储（通常是 TCPStore）
    data_expiration_seconds: int = 3600  # 数据过期时间，默认 1 小时（防止 store 无限膨胀）

    # 点对点发送计数器：目标 rank -> 已向它发送的次数（下一个序号）
    send_dst_counter: Dict[int, int] = dataclasses.field(default_factory=dict)
    # 点对点接收计数器：源 rank -> 已从它接收的次数（下一个序号）
    recv_src_counter: Dict[int, int] = dataclasses.field(default_factory=dict)
    # 广播发送计数器：本 rank 作为 src 已广播的次数
    broadcast_send_counter: int = 0
    # 广播接收计数器：源 rank -> 已从它接收的广播次数
    broadcast_recv_src_counter: Dict[int, int] = dataclasses.field(default_factory=dict)

    # 记录已写入 store 的条目 (key, 时间戳)，用于按时间过期清理（见 expire_data）。
    entries: Deque[Tuple[str, float]] = dataclasses.field(default_factory=deque)

    def __post_init__(self):
        # dataclass 初始化后：校验 rank 合法，并把各计数器按 world_size 清零。
        assert self.rank < self.world_size
        self.send_dst_counter = {i: 0 for i in range(self.world_size)}
        self.recv_src_counter = {i: 0 for i in range(self.world_size)}
        self.broadcast_recv_src_counter = {i: 0 for i in range(self.world_size)}

    def send_obj(self, obj: Any, dst: int):
        """向目标 rank 发送一个对象（点对点）。

        实现：把 obj 用 pickle 序列化后写入键 send_to/{dst}/{seq}，
        seq 是「本 rank 发往 dst」的自增序号；接收方 recv_obj 按同序号读取，
        从而保证一一配对、不错位。写入前先清理过期数据。

        参数:
            obj: 任意可被 pickle 序列化的对象，例如 {"lr": 1e-4, "step": 3}。
            dst: 目标进程的 rank，例如 1。

        示例:
            # rank 0 上执行：
            >>> pg.send_obj({"hello": 1}, dst=1)
            # rank 1 上对应执行 pg.recv_obj(src=0) 即可拿到 {"hello": 1}
        """
        self.expire_data()
        key = f"send_to/{dst}/{self.send_dst_counter[dst]}"
        self.store.set(key, pickle.dumps(obj))
        self.send_dst_counter[dst] += 1
        self.entries.append((key, time.perf_counter()))

    def expire_data(self):
        """清理写入 store 超过 data_expiration_seconds 秒的旧数据。

        entries 是按写入时间排好序的队列，从最旧一端开始检查：
        一旦遇到「还没过期」的条目就停止（后面的更新，必然也没过期）。
        目的是防止 store 里积累的历史键无限增长。
        """
        while self.entries:
            # 检查最旧的一条。
            key, timestamp = self.entries[0]
            if time.perf_counter() - timestamp > self.data_expiration_seconds:
                self.store.delete_key(key)
                self.entries.popleft()
            else:
                break

    def recv_obj(self, src: int) -> Any:
        """从源 rank 接收一个对象（点对点），与 send_obj 配对。

        实现：按「本 rank 从 src 接收」的自增序号，去读键
        send_to/{self.rank}/{seq}，再 pickle 反序列化。若对应键还没被
        发送方写入，store.get 会阻塞等待，天然起到同步作用。

        参数:
            src: 源进程的 rank，例如 0。

        返回:
            发送方通过 send_obj 传来的对象。

        示例:
            # rank 1 上执行（对应 rank 0 的 pg.send_obj({"hello": 1}, dst=1)）：
            >>> pg.recv_obj(src=0)   # {"hello": 1}
        """
        obj = pickle.loads(
            self.store.get(f"send_to/{self.rank}/{self.recv_src_counter[src]}")
        )
        self.recv_src_counter[src] += 1
        return obj

    def broadcast_obj(self, obj: Optional[Any], src: int) -> Any:
        """把一个对象从源 rank 广播给组内所有其他 rank。

        注意：所有 rank 收到后不会做清理（不 delete 键），因此仅适合
        「有限次数」的场景（如初始化握手），不要在热路径里高频调用。

        角色约定：
            - 当 self.rank == src（发送方）：把 obj 写入 broadcast_from/{src}/{seq} 并原样返回 obj，
              此时 obj 必须是有效对象。
            - 其他 rank（接收方）：obj 传 None 即可，按序号读取并返回收到的对象。

        参数:
            obj: 发送方传要广播的对象；接收方传 None。
            src: 广播源的 rank。

        返回:
            发送方返回原 obj；接收方返回收到的对象。

        示例:
            # 所有 rank 都调用同一行，约定 src=0：
            >>> cfg = pg.broadcast_obj(config if pg.rank == 0 else None, src=0)
            # 结束后每个 rank 的 cfg 都等于 rank 0 传入的 config
        """
        if self.rank == src:
            self.expire_data()
            key = f"broadcast_from/{src}/" f"{self.broadcast_send_counter}"
            self.store.set(key, pickle.dumps(obj))
            self.broadcast_send_counter += 1
            self.entries.append((key, time.perf_counter()))
            return obj
        else:
            key = f"broadcast_from/{src}/" f"{self.broadcast_recv_src_counter[src]}"
            recv_obj = pickle.loads(self.store.get(key))
            self.broadcast_recv_src_counter[src] += 1
            return recv_obj

    def all_gather_obj(self, obj: Any) -> list[Any]:
        """all-gather：每个 rank 贡献一个对象，最终所有 rank 都拿到「全体对象的列表」。

        实现方式是依次以每个 rank 为 src 做一轮 broadcast_obj：轮到自己时广播出去，
        否则接收。返回列表按 rank 顺序排列（第 i 个元素来自 rank i）。

        参数:
            obj: 本 rank 要贡献的对象。

        返回:
            长度为 world_size 的列表，gathered[i] 是 rank i 贡献的对象。

        示例:
            # 假设 world_size=3，各 rank 分别传入自己的 rank 值：
            >>> pg.all_gather_obj(pg.rank)   # 每个 rank 都得到 [0, 1, 2]
        """
        gathered_objs = []
        for i in range(self.world_size):
            if i == self.rank:
                gathered_objs.append(obj)
                self.broadcast_obj(obj, src=self.rank)
            else:
                recv_obj = self.broadcast_obj(None, src=i)
                gathered_objs.append(recv_obj)
        return gathered_objs

    def barrier(self):
        """同步栅栏：阻塞直到组内所有 rank 都到达此处。

        复用 broadcast_obj 逐个 rank 走一遍（广播的内容是 None，只借用其
        「写-读」配对来达成同步），全部完成即代表大家都到齐了。

        示例:
            >>> pg.barrier()   # 所有 rank 在此对齐后再继续
        """
        for i in range(self.world_size):
            if i == self.rank:
                self.broadcast_obj(None, src=self.rank)
            else:
                self.broadcast_obj(None, src=i)

    @staticmethod
    def create(
        host: str,
        port: int,
        rank: int,
        world_size: int,
        data_expiration_seconds: int = 3600,
    ) -> "StatelessProcessGroup":
        """torch.distributed.init_process_group 的「无状态」替代品，不污染全局状态。

        动机：若进程 A、B 已通过 torch.distributed.init_process_group 组成一个组，
        之后又想让 A、B、C、D 组成另一个组，在 PyTorch 里是做不到的——因为 A、B
        已被绑进旧组，C、D 无法再加入。本函数用来绕开这个限制。

        init_process_group 是「全局」调用，而本函数是「无状态」调用：它只返回一个
        用于交换元数据的 StatelessProcessGroup 对象，不改动任何全局状态。于是
        A、B 可以先 create 一个组，A、B、C、D 又能再 create 出另一个组，互不干扰。

        参数:
            host: master 节点主机名/IP，例如 "127.0.0.1"。
            port: master 节点端口，例如 29500。
            rank: 本进程在组内的编号；rank == 0 者作为 TCPStore 的 master。
            world_size: 组内进程总数。
            data_expiration_seconds: store 中数据的过期秒数，默认 3600（1 小时）。

        返回:
            初始化好的 StatelessProcessGroup 实例。

        示例:
            # 每个进程用相同的 host/port/world_size、各自的 rank 调用：
            >>> pg = StatelessProcessGroup.create(
            ...     host="127.0.0.1", port=29500, rank=rank, world_size=4)
            >>> pg.barrier()   # 之后即可用它交换元数据
        """  # noqa
        store = TCPStore(
            host_name=host,
            port=port,
            world_size=world_size,
            is_master=(rank == 0),
        )

        return StatelessProcessGroup(
            rank=rank,
            world_size=world_size,
            store=store,
            data_expiration_seconds=data_expiration_seconds,
        )
