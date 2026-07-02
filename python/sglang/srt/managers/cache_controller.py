from __future__ import annotations

"""
Copyright 2023-2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

中译：HiCache 分层 KV 缓存控制器（HiCacheController）。
      SGLang 把 KV 缓存组织成多级分层存储：
        - L1：GPU 设备显存（device memory），最快、最稀缺；
        - L2：主机内存（host memory / CPU 内存），容量更大、稍慢；
        - L3：外部存储后端（storage backend，如 hf3fs / mooncake / eic / nixl 等），
              容量最大、最慢，可跨进程/跨节点持久化复用。
      本控制器负责在这几级之间搬运 KV 缓存：
        - write / load：在 L1(GPU) 与 L2(host) 之间双向拷贝（写回 / 加载）；
        - prefetch / backup（write_storage）：在 L2(host) 与 L3(storage) 之间双向拷贝
          （预取 / 备份）。
      为了与模型前向计算重叠（overlap）以隐藏拷贝延迟，控制器使用独立的 CUDA stream、
      逐层完成事件（LayerLoadingEvent / LayerDoneCounter）以及后台线程
      （prefetch_thread / backup_thread）来异步执行这些 IO。
"""

import logging
import threading
import time
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, List, NamedTuple, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import (
    STORAGE_BATCH_SIZE,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool_host import HostKVCache

from sglang.srt.distributed import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.utils import get_device_module

logger = logging.getLogger(__name__)

device_module = get_device_module()


class LayerLoadingEvent:
    """逐层加载事件集合。

    中译：为「逐层加载（layer-by-layer load）」维护一组 CUDA 事件，让消费者（模型前向）
          可以按层等待 KV 加载完成，从而把加载与计算重叠（每加载完一层就能马上算那一层，
          不必等整批全部加载完）。
          - load_events[i]：第 i 层加载完成事件；
          - start_event：本批加载在控制器 stream 上的起始事件。
    """

    def __init__(self, num_layers: int):
        self._num_layers = num_layers
        self.load_events = [device_module.Event() for _ in range(num_layers)]
        self.start_event = device_module.Event()  # start event on controller stream
        # 中译：start_event 在控制器 stream 上标记本批加载的起点。

    def complete(self, layer_index: int):
        # 中译：在当前 stream 上记录「第 layer_index 层加载完成」事件。
        assert 0 <= layer_index < self._num_layers
        self.load_events[layer_index].record()

    def wait(self, layer_index: int):
        # 中译：让当前 stream 等待「第 layer_index 层加载完成」事件（消费者按层等待）。
        device_module.current_stream().wait_event(self.load_events[layer_index])

    @property
    def finish_event(self):
        # 中译：最后一层的完成事件，即整批加载的「结束事件」。
        return self.load_events[-1]


class LayerDoneCounter:
    """逐层加载完成计数器（生产者/消费者）。

    中译：管理多套 LayerLoadingEvent，用于「重叠模式（overlap mode）」下的生产者-消费者协调。
          生产者（加载线程 start_loading）每发起一批加载就轮转占用一套事件；
          消费者（模型前向）通过 set_consumer 选定要等待哪一套，再用 wait_until 按层等待。
          维护多套（num_counters=3）是为了让相邻批次的加载/消费能并发进行而不互相覆盖。
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        # extra producer and consumer counters for overlap mode
        # 中译：为重叠模式额外准备多套计数器（生产者/消费者各自推进，互不阻塞）。
        self.num_counters = 3
        self.events = [LayerLoadingEvent(num_layers) for _ in range(self.num_counters)]
        self.producer_index = -1
        self.consumer_index = -1

    def update_producer(self):
        # 中译：生产者轮转到下一套事件并返回其索引。
        #       复用前断言该套的结束事件已 ready，避免覆盖尚未消费完的事件。
        self.producer_index = (self.producer_index + 1) % self.num_counters
        assert self.events[
            self.producer_index
        ].finish_event.query(), (
            "Producer finish event should be ready before being reused."
        )
        return self.producer_index

    def set_consumer(self, index: int):
        # 中译：指定消费者当前要等待的事件套索引（通常为某次 start_loading 返回的 producer_id）。
        self.consumer_index = index

    def wait_until(self, threshold: int):
        # 中译：等待消费套的第 threshold 层加载完成；未设置消费者（<0）时直接返回。
        if self.consumer_index < 0:
            return
        self.events[self.consumer_index].wait(threshold)

    def reset(self):
        # 中译：重置生产者/消费者索引（控制器 reset 时调用）。
        self.producer_index = -1
        self.consumer_index = -1


class CacheOperation:
    """一次 L1<->L2（GPU<->host）缓存搬运操作的描述。

    中译：记录一次缓存拷贝所涉及的主机侧索引（host_indices）与设备侧索引（device_indices）、
          关联的缓存树节点 id（node_ids，用于完成后回执）以及调度优先级（priority）。
          类变量 counter 用于给每个操作分配全局自增 id；多个操作可被 merge_ops 合并成一个，
          以便一次性发起更大批量的拷贝。
    """

    counter = 0  # 全局自增计数器，为每个操作分配唯一 id。

    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        priority: Optional[int] = None,
    ):
        self.host_indices = host_indices
        self.device_indices = device_indices
        self.node_ids = [node_id]
        self.data = None

        self.id = CacheOperation.counter
        CacheOperation.counter += 1
        # default priority is the order of creation
        # 中译：默认优先级即创建顺序（id），越早创建优先级数值越小、越优先。
        self.priority = priority if priority is not None else self.id

    @staticmethod
    def merge_ops(ops: List[CacheOperation]) -> CacheOperation:
        # 中译：把多个操作合并为一个：拼接各自的 host/device 索引，合并 node_ids，
        #       取最小 priority 作为合并后的优先级。单个时直接返回，避免无谓拷贝。
        assert len(ops) > 0
        if len(ops) == 1:
            return ops[0]

        host_indices = torch.cat([op.host_indices for op in ops])
        device_indices = torch.cat([op.device_indices for op in ops])
        node_ids = []
        priority = min(op.priority for op in ops)
        for op in ops:
            node_ids.extend(op.node_ids)
        merged_op = CacheOperation(host_indices, device_indices, -1, priority)
        merged_op.node_ids = node_ids
        return merged_op

    def __lt__(self, other: CacheOperation):
        # 中译：按 priority 比较，使 CacheOperation 可用于优先级队列/排序。
        return self.priority < other.priority


class HiCacheAck(NamedTuple):
    """一次 write/load 操作的完成回执。

    中译：携带该批拷贝的起始事件、结束事件以及涉及的缓存节点 id 列表。
          调度器据此查询事件是否完成，并在完成后标记对应节点的缓存状态。
    """

    start_event: device_module.Event
    finish_event: device_module.Event
    node_ids: List[int]


class TransferBuffer:
    """
    Overlapping buffer preparation and transfer operations to improve throughput.

    中译：传输缓冲区。用一个有界队列把「缓冲准备」与「实际传输」解耦并重叠起来，
          以提升吞吐。put/get 受 stop_event 控制，便于停止时及时退出而不死等。
    """

    def __init__(self, stop_event, buffer_count: int = 3) -> None:
        self.stop_event = stop_event
        self.buffers = Queue(maxsize=buffer_count)

    def full(self) -> bool:
        # 中译：缓冲队列是否已满。
        return self.buffers.full()

    def empty(self) -> bool:
        # 中译：缓冲队列是否为空。
        return self.buffers.empty()

    def put(self, item, block=True, timeout=1) -> None:
        # 中译：放入一个待传输项。队列满时循环重试（带超时），但只要 stop_event 置位就退出；
        #       非阻塞模式下满则直接放弃。
        while not self.stop_event.is_set():
            try:
                self.buffers.put(item, block=block, timeout=timeout)
                break
            except Full:
                if not block:
                    break
                continue
            except Exception as e:
                logger.error(e)

    def get(self, block=True, timeout=1) -> Optional[CacheOperation]:
        # 中译：取出一个待传输项；超时为空则返回 None（供调用方轮询）。
        try:
            return self.buffers.get(block=block, timeout=timeout)
        except Empty:
            return None
        except Exception as e:
            logger.error(e)

    def clear(self):
        # 中译：清空缓冲队列（reset 时使用）。
        self.buffers.queue.clear()


class StorageOperation:
    """一次 L2<->L3（host<->storage）存储操作的描述。

    中译：描述一次与外部存储后端（L3）的交互（备份 backup 或预取 prefetch）。
          - host_indices：涉及的主机内存页索引；
          - token_ids：这些页对应的 token 序列（用于计算哈希键）；
          - last_hash：前缀链的最后一个哈希，作为本批哈希计算的起点（前缀缓存复用）；
          - hash_value：每页的哈希键列表（即存储后端的 key）；
          - prefix_keys：前缀键，部分后端用于组织/校验前缀；
          - completed_tokens：已完成传输的 token 数（进度）。
    """

    counter = 0  # 全局自增计数器，为每个存储操作分配唯一 id。

    def __init__(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        self.host_indices = host_indices
        self.token_ids = token_ids
        self.last_hash = last_hash
        self.completed_tokens = 0
        self.hash_value = hash_value if hash_value is not None else []
        self.prefix_keys = prefix_keys

        self.id = StorageOperation.counter
        StorageOperation.counter += 1

    def __lt__(self, other: StorageOperation):
        # 中译：按创建 id 排序（先创建的先处理，近似 FIFO）。
        return self.id < other.id


class PrefetchOperation(StorageOperation):
    """一次预取（L3->L2）操作，在 StorageOperation 之上增加可中断能力。

    中译：预取是异步、可被取消的：主调度线程可在收益不足或资源紧张时调用 mark_terminate
          提前终止，后台 IO 线程通过 increment 的返回值感知终止。用锁保护 completed_tokens
          与终止标志，确保跨线程读写安全。还记录 request_id 与 start_time 便于追踪与超时管理。
    """

    def __init__(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        self.request_id = request_id

        self._lock = threading.Lock()
        self._terminated_flag = False
        self.start_time = time.monotonic()

        super().__init__(host_indices, token_ids, last_hash, prefix_keys=prefix_keys)

    def increment(self, num_tokens: int):
        # 中译：原子地推进已完成 token 数。若操作已被终止则返回 False，调用方据此停止后续传输。
        with self._lock:
            if self._terminated_flag:
                return False
            self.completed_tokens += num_tokens
            return True

    def mark_terminate(self):
        # 中译：标记该预取操作终止（线程安全），后台线程在下次 increment 时感知并停止。
        with self._lock:
            self._terminated_flag = True

    def is_terminated(self) -> bool:
        # 中译：查询该预取操作是否已被终止。
        return self._terminated_flag


class HiCacheController:
    """分层 KV 缓存（HiCache）控制器，统筹 L1/L2/L3 三级之间的 KV 缓存搬运。

    中译：系统中的核心角色，被 HiRadixCache / Scheduler 持有，负责：
          - L1<->L2（GPU<->host）：write 写回、load 加载，使用独立 CUDA stream 与逐层事件
            实现与前向计算的重叠；
          - L2<->L3（host<->storage）：prefetch 预取、write_storage 备份，由后台线程异步执行。
          关键协作对象：
          - token_to_kv_pool_allocator / mem_pool_device：L1 设备侧 KV 池及其分配器；
          - mem_pool_host：L2 主机侧 KV 池；
          - storage_backend：L3 外部存储后端（可在运行时 attach/detach）；
          - 各 ProcessGroup（tp/attn_cp/attn_tp/pp）：分布式下做预取命中数的跨 rank 同步。
          还支持「草稿（draft）KV 池」（投机解码）随目标 KV 一起搭车搬运。
    """

    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        mem_pool_host: HostKVCache,
        page_size: int,
        tp_group: torch.distributed.ProcessGroup,
        load_cache_event: threading.Event,
        attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
        attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
        pp_group: Optional[torch.distributed.ProcessGroup] = None,
        write_policy: str = "write_through_selective",
        io_backend: str = "",
        storage_backend: Optional[str] = None,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        enable_storage_metrics: bool = False,
    ):
        # 中译：保存各分布式进程组（张量并行、注意力 cp/tp、流水线并行），供后续做集合通信。
        self.tp_group = tp_group  # 张量并行进程组
        self.attn_cp_group = attn_cp_group  # 注意力上下文并行（context parallel）进程组
        self.attn_tp_group = attn_tp_group  # 注意力张量并行进程组
        self.pp_group = pp_group  # 流水线并行进程组
        # 预取命中数跨 rank 同步用的进程组列表（gloo），由 _create_prefetch_sync_groups 建立。
        self.prefetch_sync_groups: List[torch.distributed.ProcessGroup] = []
        # L1 设备侧 KV 池的分配器（负责 device 页的分配/释放）。
        self.mem_pool_device_allocator = token_to_kv_pool_allocator
        mem_pool_device = token_to_kv_pool_allocator.get_kvcache()
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        # 中译：混合线性 KV 池只对其中的 full KV 子池做分层缓存搬运。
        if isinstance(mem_pool_device, HybridLinearKVPool):
            mem_pool_device = mem_pool_device.full_kv_pool
        self.mem_pool_device = mem_pool_device  # L1 设备侧（GPU）KV 池
        self.mem_pool_host = mem_pool_host  # L2 主机侧（CPU 内存）KV 池
        self.write_policy = write_policy  # 写回策略：write_through / write_through_selective / write_back
        self.page_size = page_size  # 每页 token 数（分配与搬运的最小粒度）
        self.io_backend = io_backend  # device<->host 拷贝所用的 IO 后端标识
        self.enable_storage = False  # 是否已启用 L3 存储后端（attach 成功后置 True）
        self.storage_backend = None  # L3 存储后端实例（未 attach 时为 None）
        self.storage_backend_type = None  # L3 存储后端类型名（如 file / mooncake 等）
        self.enable_storage_metrics = enable_storage_metrics  # 是否采集存储命中/耗时等指标

        # Draft KV pool support (best-effort piggyback on target L2/L3 ops).
        # 中译：草稿（draft）KV 池支持（投机解码用）——尽力而为地搭车在目标 KV 的 L2/L3 操作上一起搬运。
        self.has_draft = False  # 是否存在草稿 KV 池（投机解码启用时为 True）
        self.mem_pool_device_draft = None  # 草稿 KV 的 L1 设备侧池
        self.mem_pool_host_draft = None  # 草稿 KV 的 L2 主机侧池
        self.draft_page_get_func = None  # 草稿 KV 的 storage 读页函数（attach 时设置）
        self.draft_page_set_func = None  # 草稿 KV 的 storage 写页函数（attach 时设置）

        # Default storage page IO functions (may be overridden by attach).
        # 中译：默认的存储页读/写函数（attach 后端时可能被替换为零拷贝版本）。
        self.page_get_func = self._generic_page_get  # storage 读页函数（attach 后可替换为零拷贝版）
        self.page_set_func = self._generic_page_set  # storage 写页函数（attach 后可替换为零拷贝版）

        # Dedicated stop event for storage background threads (prefetch/backup).
        # NOTE: Do NOT reuse `self.stop_event` here since it also guards core HiCache
        # transfer buffers (CPU<->GPU). We want to allow runtime attach/detach of
        # storage without stopping the whole controller.
        # 中译：存储后台线程（预取/备份）专用的停止事件。
        #       注意：不要复用 self.stop_event——后者还守护核心的 CPU<->GPU 传输缓冲；
        #       单独一个事件才能在运行时 attach/detach 存储后端而不必停掉整个控制器。
        self.storage_stop_event = threading.Event()  # 存储后台线程（预取/备份）专用停止信号

        self.device = self.mem_pool_device.device  # 执行设备（如 cuda:0）
        self.layer_num = self.mem_pool_device.layer_num  # 模型层数（逐层搬运时使用）
        # 逐层完成计数器：搬运每层完成时打点，供计算与拷贝流水线重叠。
        self.layer_done_counter = LayerDoneCounter(self.layer_num)
        # 中译：把逐层完成计数器注册给设备 KV 池，使其在逐层搬运时回调标记每层完成事件。
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)

        if write_policy not in [
            "write_through",
            "write_through_selective",
            "write_back",
        ]:
            raise ValueError(f"Invalid write policy: {write_policy}")

        # self.write_queue = PriorityQueue[CacheOperation]()
        # 中译：L1<->L2 的待处理队列与完成回执队列。load/write 各一对：
        #       *_queue 暂存待发起的操作，ack_*_queue 暂存已发起、等待事件完成的回执。
        self.load_queue: List[CacheOperation] = []  # 待发起的 L2->L1 加载操作队列
        self.write_queue: List[CacheOperation] = []  # 待发起的 L1->L2 写回操作队列
        self.ack_load_queue: List[HiCacheAck] = []  # 已发起加载、等待事件完成的回执队列
        self.ack_write_queue: List[HiCacheAck] = []  # 已发起写回、等待事件完成的回执队列

        self.stop_event = threading.Event()  # 控制器核心（CPU<->GPU 传输缓冲）的停止信号
        self.write_buffer = TransferBuffer(self.stop_event)  # 写回用的中转缓冲
        self.load_buffer = TransferBuffer(self.stop_event, buffer_count=10)  # 加载用的中转缓冲

        # 中译：写回与加载各用独立 CUDA stream，与主计算 stream 并发以隐藏拷贝延迟。
        self.write_stream = device_module.Stream()  # 专用写回流（L1->L2）
        self.load_stream = device_module.Stream()  # 专用加载流（L2->L1）

        # If a storage backend is provided at startup, treat it as an implicit attach,
        # so init/runtime share the same lifecycle semantics and code paths.
        # 中译：若启动时就给定了存储后端，则视作一次隐式 attach，让「初始化时启用」与
        #       「运行时启用」走完全相同的生命周期语义与代码路径。
        if storage_backend is not None:
            try:
                self.attach_storage_backend(
                    storage_backend=storage_backend,
                    prefetch_threshold=prefetch_threshold,
                    model_name=model_name,
                    storage_backend_extra_config=storage_backend_extra_config,
                )
            except ValueError as e:
                # Preserve the historical error shape on init for unknown backends.
                # 中译：保持初始化阶段对未知后端的历史报错形态（向后兼容）。
                raise ValueError(f"Failed to create storage backend: {e}") from e

    def get_attn_cp_rank_and_size(self) -> tuple[int, int]:
        """Derive CP rank/size from the attn_cp process group.

        中译：从注意力上下文并行（attn_cp）进程组推导本 rank 的 cp 序号与组大小；
              未启用时返回 (0, 1)。
        """
        if self.attn_cp_group is not None:
            return (
                torch.distributed.get_rank(group=self.attn_cp_group),
                torch.distributed.get_world_size(group=self.attn_cp_group),
            )
        return 0, 1

    def _create_prefetch_sync_groups(self) -> None:
        # 中译：为预取命中数同步创建专用的 gloo 进程组。
        #       存储是按 rank 分片/复制的，需要在同一组 rank 间对「命中页数」取 MIN 对齐，
        #       以保证各 rank 预取的页数一致。这里按 attn_cp/attn_tp（或退化到 tp）去重后建组。
        from sglang.srt.distributed.parallel_state import create_custom_parallel_group

        self.prefetch_sync_groups = []
        seen_rank_sets = set()  # 已建组的 rank 集合，用于去重避免重复建组。

        if self.attn_cp_group is not None or self.attn_tp_group is not None:
            base_groups = [self.attn_cp_group, self.attn_tp_group]
        else:
            base_groups = [self.tp_group]

        for group in base_groups:
            if group is None or torch.distributed.get_world_size(group=group) == 1:
                continue
            group_ranks = tuple(torch.distributed.get_process_group_ranks(group))
            if group_ranks in seen_rank_sets:
                continue
            seen_rank_sets.add(group_ranks)
            self.prefetch_sync_groups.append(
                create_custom_parallel_group(
                    group_ranks=list(group_ranks), backend="gloo"
                )
            )

    def _destroy_prefetch_sync_groups(self) -> None:
        # 中译：销毁预取同步进程组（detach 存储后端时调用），逐个尽力销毁、忽略异常。
        for group in self.prefetch_sync_groups:
            try:
                torch.distributed.destroy_process_group(group)
            except Exception:
                pass
        self.prefetch_sync_groups = []

    def _all_reduce_prefetch_groups(self, tensor: torch.Tensor, op) -> None:
        # 中译：在所有预取同步组上对 tensor 做 all_reduce（如对命中页数取 MIN 以跨 rank 对齐）。
        for group in self.prefetch_sync_groups:
            torch.distributed.all_reduce(tensor, op=op, group=group)

    def _start_storage_threads(self):
        """Start storage prefetch/backup threads and their queues.

        This is used by runtime attach, and also by reset when storage is enabled.

        中译：启动存储后台线程（预取线程、备份线程）及其配套队列。
              运行时 attach 存储后端时调用，启用存储后的 reset 也会用到。
              同时初始化预取撤销队列、备份回执队列、主机内存释放队列等通信通道。
        """
        assert self.enable_storage
        assert not self.storage_stop_event.is_set()

        # 预取线程：从 L3 storage 异步拉取到 host（L2）。
        self.prefetch_thread = threading.Thread(
            target=self.prefetch_thread_func, daemon=True
        )
        # 备份线程：把 host（L2）异步写回 L3 storage。
        self.backup_thread = threading.Thread(
            target=self.backup_thread_func, daemon=True
        )
        self.prefetch_queue = Queue()  # 待处理的预取操作队列
        self.backup_queue = Queue()  # 待处理的备份操作队列

        self.prefetch_revoke_queue: Queue[str] = Queue()  # 被撤销的预取请求 id 队列
        self.ack_backup_queue: Queue[StorageOperation] = Queue()  # 备份完成回执队列
        self.host_mem_release_queue: Queue[torch.Tensor] = Queue()  # 待释放的 host 页索引队列

        self.prefetch_thread.start()
        self.backup_thread.start()

    def _stop_storage_threads(self):
        """Stop storage prefetch/backup threads and drain internal queues.

        Caller should ensure no in-flight requests.

        中译：停止存储预取/备份线程并清空内部队列。调用方需保证此时没有进行中的请求。
              做法：先置位 storage_stop_event，再向各队列塞 None「唤醒」可能阻塞的线程，
              最后 join 等待退出；若仍有线程存活则报错（避免悬挂线程触碰已释放状态）。
        """
        # Always request stop. This is safe even when storage is already disabled,
        # and makes detach truly idempotent (previous partial detach may have left
        # threads alive).
        # NOTE: do NOT clear stop_event unless threads have fully stopped; otherwise
        # a still-alive thread may resume and touch released state.
        # 中译：务必在线程完全停止前不要清除 stop_event，否则尚存活的线程可能恢复运行、
        #       触碰已释放的状态。
        self.storage_stop_event.set()

        # Best-effort wakeups so threads exit promptly even if blocked on queues.
        # 中译：向各队列塞 None 做「尽力而为」的唤醒，让阻塞在队列上的线程能及时退出。
        try:
            if hasattr(self, "prefetch_queue"):
                self.prefetch_queue.put_nowait(None)
            if hasattr(self, "backup_queue"):
                self.backup_queue.put_nowait(None)
            if hasattr(self, "prefetch_buffer"):
                self.prefetch_buffer.put_nowait(None)
        except Exception:
            pass

        # Best-effort joins (threads are daemon, but join keeps state clean).
        threads = []
        if hasattr(self, "prefetch_thread"):
            threads.append(self.prefetch_thread)
        if hasattr(self, "backup_thread"):
            threads.append(self.backup_thread)
        if hasattr(self, "prefetch_io_aux_thread"):
            threads.append(self.prefetch_io_aux_thread)

        for t in threads:
            try:
                t.join(timeout=10)
            except Exception:
                pass

        alive = [t for t in threads if getattr(t, "is_alive", lambda: False)()]
        if alive:
            logger.error(
                "Failed to stop HiCache storage threads cleanly: %s",
                [getattr(t, "name", repr(t)) for t in alive],
            )
            raise RuntimeError("Failed to stop HiCache storage threads cleanly.")

    def attach_storage_backend(
        self,
        storage_backend: str,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
    ):
        """Attach (enable) storage backend at runtime.

        Requirement: no in-flight requests. This call is expected to run on the scheduler
        thread (control path), not concurrently with prefetch/backup.

        中译：在运行时挂载（启用）L3 存储后端。
              要求：此时无进行中的请求；应在调度器线程（控制路径）上调用，不与预取/备份并发。
              流程：先确保旧线程已停 -> 生成存储配置 -> 经工厂创建后端 -> 选择普通/零拷贝
              读写函数 -> 建预取同步组 -> 登记草稿池 -> 启动后台线程。
              采用「失败可回滚」设计：任一步出错都会清理并把控制器状态还原，便于后续重试 attach。
        """
        if self.enable_storage:
            raise RuntimeError("Storage backend already attached.")

        # Defensive: a previous partial detach may have flipped `enable_storage` but
        # left background threads alive. Attaching on top of them is unsafe.
        try:
            self._stop_storage_threads()
        except Exception as e:
            raise RuntimeError(
                "Cannot attach storage backend: previous detach did not stop storage threads cleanly."
            ) from e

        # Rollback-safe init: if creation fails, keep controller state consistent
        # for future attach attempts.
        self.storage_backend_type = storage_backend
        from sglang.srt.mem_cache.utils import get_hash_str

        self.get_hash_str = get_hash_str
        self.storage_config = self._generate_storage_config(
            model_name, storage_backend_extra_config
        )
        # for MLA models, only one rank needs to backup the KV cache
        # 中译：MLA 类模型的 KV 是按 rank 复制的，只需一个 rank（rank 0）负责备份，其余跳过。
        self.backup_skip = (
            self.storage_config.is_mla_model
            # todo: load balancing
            and self.storage_config.tp_rank != 0
        )

        # Use storage backend factory for dynamic backend creation
        from sglang.srt.mem_cache.storage import StorageBackendFactory

        try:
            self.storage_backend = StorageBackendFactory.create_backend(
                storage_backend, self.storage_config, self.mem_pool_host
            )
            self.storage_backend.register_mem_pool_host(self.mem_pool_host)

            self.enable_storage = True
            # todo: threshold policy for prefetching
            # 中译：预取阈值——命中 token 数低于此值就不值得预取（至少为一页）。
            self.prefetch_threshold = max(prefetch_threshold, self.page_size)
            # 中译：预取占用上限——约为「host 池比 device 池多出的容量」的 80%，留有余量防打满。
            self.prefetch_capacity_limit = max(
                0, int(0.8 * (self.mem_pool_host.size - self.mem_pool_device.size))
            )
            # tracking the number of tokens locked in prefetching, updated by the main scheduler thread
            # 中译：当前被预取锁定的 token 数（由主调度线程更新），用于限流判断。
            self.prefetch_tokens_occupied = 0

            # Use dedicated gloo groups so storage prefetch sync is isolated
            # from other collectives and consistent across CPxTP participants.
            # 中译：用专用 gloo 组做预取同步，与其它集合通信隔离，并在 CP×TP 参与者间保持一致。
            self._create_prefetch_sync_groups()

            # Select the get and set functions
            # 中译：选择页读写函数。部分后端支持零拷贝（zero-copy），直接在 host 池与后端间传输，
            #       性能更好；否则退化到通用的、经中转缓冲的 generic 路径。
            self.page_get_func = self._generic_page_get
            self.page_set_func = self._generic_page_set

            if (
                self.storage_backend_type
                in ["hf3fs", "mooncake", "eic", "nixl", "simm"]
            ) or (
                self.storage_backend_type == "dynamic"
                and bool(self.storage_config.extra_config.get("interface_v1", 0))
            ):
                self.page_get_func = self._page_get_zero_copy
                self.page_set_func = self._page_set_zero_copy

            self._maybe_register_draft_with_storage()

            # Ensure stop_event is clear before starting threads.
            # 中译：启动线程前确保停止事件已清除（否则刚启动的线程会立刻退出）。
            self.storage_stop_event.clear()
            self._start_storage_threads()
        except Exception:
            # Best-effort cleanup for partial init.
            # 中译：初始化中途失败——尽力回滚清理（停线程、销毁进程组、关后端、复位标志/函数）。
            try:
                self._stop_storage_threads()
            except Exception:
                pass
            self._destroy_prefetch_sync_groups()
            try:
                if (
                    hasattr(self, "storage_backend")
                    and self.storage_backend is not None
                ):
                    if hasattr(self.storage_backend, "close"):
                        self.storage_backend.close()
            except Exception:
                pass
            self.storage_backend = None
            self.storage_backend_type = None
            self.enable_storage = False
            self.page_get_func = self._generic_page_get
            self.page_set_func = self._generic_page_set
            self.draft_page_get_func = None
            self.draft_page_set_func = None
            raise

    def detach_storage_backend(self):
        """Detach (disable) storage backend at runtime.

        Requirement: no in-flight requests. This will stop storage threads and release
        the backend instance (best-effort close).

        中译：在运行时卸载（禁用）L3 存储后端。要求无进行中的请求。
              会停掉存储线程、销毁进程组、关闭后端实例（尽力 close），并复位相关状态。
              设计为幂等：即使 enable_storage 已为 False，也会尽量清理上次未完成 detach 的残留。
              若线程未能干净停止则抛错（不静默成功），避免在线程仍存活时翻转 enable_storage。
        """
        # Idempotent cleanup: even if `enable_storage` is already False,
        # we may still have leftover resources (threads/backend/process group) from a
        # previous partial detach. We attempt cleanup whenever possible.
        try:
            self._stop_storage_threads()
        except Exception as e:
            # Do not proceed tearing down backend/process group if threads are not
            # fully stopped; otherwise still-alive threads may touch released state.
            # Caller can retry detach.
            logger.exception("Stop storage threads failed: %s", e)
            # IMPORTANT: Do not silently succeed. Upper layers rely on exceptions here
            # to avoid flipping `enable_storage` flags while threads are still alive.
            raise RuntimeError("Stop storage threads failed; detach aborted.") from e

        # Best-effort destroy process groups created for storage ops.
        self._destroy_prefetch_sync_groups()

        # Best-effort close (some backends rely on GC/destructor).
        try:
            if (
                hasattr(self, "storage_backend")
                and self.storage_backend is not None
                and hasattr(self.storage_backend, "close")
            ):
                self.storage_backend.close()
        except Exception:
            logger.exception("Failed to close storage backend cleanly.")

        self.storage_backend = None
        self.storage_backend_type = None
        self.enable_storage = False
        self.page_get_func = self._generic_page_get
        self.page_set_func = self._generic_page_set
        self.draft_page_get_func = None
        self.draft_page_set_func = None
        # Now it's safe to clear the stop event for future re-attach.
        self.storage_stop_event.clear()

    def _generate_storage_config(
        self,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
    ):
        # 中译：依据当前并行配置（tp/dp/pp/attn_cp 的 rank 与 size）和 KV 池类型，
        #       生成传给存储后端的配置 HiCacheStorageConfig（含布局、是否 rank 复制、是否切分 head 等）。
        if storage_backend_extra_config is None:
            storage_backend_extra_config = {}

        # 中译：启用 DP 注意力时，tp/dp rank 取自注意力并行视角；否则取自全局张量并行。
        if is_dp_attention_enabled():
            self.tp_rank = get_attention_tp_rank()
            self.tp_size = get_attention_tp_size()
            self.dp_rank = get_attention_dp_rank()
        else:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
            self.dp_rank = 0

        self.pp_rank = get_pipeline_model_parallel_rank()
        self.pp_size = get_pipeline_model_parallel_world_size()

        # Currently, NPUMLATokenToKVPool is the subclass of MLATokenToKVPool.
        # DeepSeekV4TokenToKVPool has compressed MLA-style rank-replicated cache
        # data. storage only needs rank 0 to write it back.
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

        is_mla_model = isinstance(self.mem_pool_device, MLATokenToKVPool)
        is_compressed_mla_model = isinstance(
            self.mem_pool_device, DeepSeekV4TokenToKVPool
        )
        is_rank_replicated = is_mla_model or is_compressed_mla_model
        # Least Common Multiple among heterogeneous tp size
        # 中译：异构 tp 规模间的最小公倍数；用于不同 tp 配置间复用同一份存储时对齐分片粒度。
        tp_lcm_size = storage_backend_extra_config.pop("tp_lcm_size", None)
        should_split_heads = False

        if tp_lcm_size:
            assert (
                tp_lcm_size % self.tp_size == 0
            ), "tp_lcm_size must be divisible by tp_size."
            should_split_heads = (
                not is_rank_replicated
                and self.mem_pool_host.layout == "page_head"
                and tp_lcm_size > self.tp_size
            )

        attn_cp_rank, attn_cp_size = self.get_attn_cp_rank_and_size()

        return HiCacheStorageConfig(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            attn_cp_rank=attn_cp_rank,
            attn_cp_size=attn_cp_size,
            # TODO(hzh): Rename is_mla_model to is_rank_replicated.
            is_mla_model=is_rank_replicated,
            enable_storage_metrics=self.enable_storage_metrics,
            is_page_first_layout=self.mem_pool_host.layout == "page_first",
            model_name=model_name,
            tp_lcm_size=tp_lcm_size,
            should_split_heads=should_split_heads,
            extra_config=storage_backend_extra_config,
        )

    def reset(self):
        # 中译：重置控制器。先置位两个停止事件让后台线程退出，清空所有队列与缓冲；
        #       若启用存储则 join 线程、清空存储相关队列；最后清除停止事件并重启存储线程。
        self.stop_event.set()
        self.storage_stop_event.set()

        self.write_queue.clear()
        self.load_queue.clear()
        self.write_buffer.clear()
        self.load_buffer.clear()
        self.ack_write_queue.clear()
        self.ack_load_queue.clear()
        if self.enable_storage:
            self.prefetch_thread.join()
            self.backup_thread.join()
            self.prefetch_queue.queue.clear()
            self.backup_queue.queue.clear()
            self.prefetch_revoke_queue.queue.clear()
            self.ack_backup_queue.queue.clear()
            self.host_mem_release_queue.queue.clear()
            self.prefetch_tokens_occupied = 0

        self.stop_event.clear()
        self.storage_stop_event.clear()

        if self.enable_storage:
            self.prefetch_thread = threading.Thread(
                target=self.prefetch_thread_func, daemon=True
            )
            self.backup_thread = threading.Thread(
                target=self.backup_thread_func, daemon=True
            )
            self.prefetch_thread.start()
            self.backup_thread.start()

    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Back up KV caches from device memory to host memory.

        中译：把 KV 缓存从设备显存（L1）写回主机内存（L2）。
              先在 host 池分配落点；分配失败（host 满）返回 None；否则入队并触发写出，
              返回分配到的 host_indices 供上层登记。
        """
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        self.write_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        self.start_writing()
        return host_indices

    def start_writing(self) -> None:
        # 中译：把写队列中的操作合并为一个，在专用 write_stream 上发起 GPU->host 的逐层拷贝，
        #       并登记 (start_event, finish_event, node_ids) 回执供后续查询完成。
        if len(self.write_queue) == 0:
            return

        op = CacheOperation.merge_ops(self.write_queue)
        host_indices, device_indices = self.move_indices(
            op.host_indices, op.device_indices
        )
        self.write_queue.clear()

        start_event = device_module.Event()
        finish_event = device_module.Event()

        start_event.record()
        with device_module.stream(self.write_stream):
            start_event.wait(self.write_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device, host_indices, device_indices, self.io_backend
            )
            if self.has_draft:
                self.mem_pool_host_draft.backup_from_device_all_layer(
                    self.mem_pool_device_draft,
                    host_indices,
                    device_indices,
                    self.io_backend,
                )
            finish_event.record()
            # NOTE: We must save the host indices and device indices here,
            # this is because we need to guarantee that these tensors are
            # still alive when the write stream is executing.
            # 中译：必须在此通过 record_stream 标记这两个索引张量被 write_stream 使用，
            #       以保证异步拷贝执行期间它们不会被提前释放（跨 stream 的生命周期保护）。
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_stream)

        self.ack_write_queue.append(HiCacheAck(start_event, finish_event, op.node_ids))

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Load KV caches from host memory to device memory.

        中译：把 KV 缓存从主机内存（L2）加载到设备显存（L1）。
              先在设备池分配落点；分配失败（显存满）返回 None；否则入队并返回 device_indices。
              注意：此处只入队，实际加载由 start_loading 在加载 stream 上逐层发起。
        """
        device_indices = self.mem_pool_device_allocator.alloc(len(host_indices))
        if device_indices is None:
            return None
        self.load_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        return device_indices

    def move_indices(self, host_indices: torch.Tensor, device_indices: torch.Tensor):
        # move indices to GPU if using kernels, to host if using direct indexing
        # 中译：按 IO 后端调整索引张量所在设备并对齐布局：
        #       - kernel：用自定义 kernel 拷贝，host_indices 需在 GPU 上；
        #       - direct：直接按索引拷贝，按 host 池布局（layer_first/page_first_direct）决定
        #         是否排序及把 device_indices 移到 CPU；
        #       - kernel_ascend：昇腾 kernel 路径，device_indices 移到 CPU。
        if self.io_backend == "kernel":
            if not host_indices.is_cuda:
                host_indices = host_indices.to(self.device, non_blocking=True)
            return host_indices, device_indices
        elif self.io_backend == "direct":
            if self.mem_pool_host.layout == "layer_first":
                device_indices = device_indices.cpu()
                host_indices, idx = host_indices.sort()
                return host_indices, device_indices.index_select(0, idx)
            elif self.mem_pool_host.layout == "page_first_direct":
                return host_indices, device_indices.cpu()
            else:
                raise ValueError(
                    f"Unsupported layout {self.mem_pool_host.layout!r} for io backend 'direct'"
                )
        elif self.io_backend == "kernel_ascend":
            return host_indices, device_indices.cpu()
        else:
            raise ValueError(f"Unsupported io backend")

    def start_loading(self) -> int:
        # 中译：发起一批 host->device 加载。合并加载队列，轮转占用一套逐层事件（producer_id），
        #       在 load_stream 上逐层拷贝并在每层完成后 record 事件，使消费者（前向）可按层等待。
        #       登记完成回执并返回 producer_id，供消费者 set_consumer。无待加载时返回 -1。
        if len(self.load_queue) == 0:
            return -1

        producer_id = self.layer_done_counter.update_producer()
        op = CacheOperation.merge_ops(self.load_queue)
        host_indices, device_indices = self.move_indices(
            op.host_indices, op.device_indices
        )
        self.load_queue.clear()
        producer_event = self.layer_done_counter.events[producer_id]
        producer_event.start_event.record()

        with device_module.stream(self.load_stream):
            producer_event.start_event.wait(self.load_stream)
            for i in range(self.layer_num):
                self.mem_pool_host.load_to_device_per_layer(
                    self.mem_pool_device,
                    host_indices,
                    device_indices,
                    i,
                    self.io_backend,
                )
                if self.has_draft and i < self.mem_pool_host_draft.layer_num:
                    self.mem_pool_host_draft.load_to_device_per_layer(
                        self.mem_pool_device_draft,
                        host_indices,
                        device_indices,
                        i,
                        self.io_backend,
                    )
                producer_event.complete(i)
            # NOTE: We must save the host indices and device indices here,
            # this is because we need to guarantee that these tensors are
            # still alive when the load stream is executing.
            # 中译：同 start_writing——record_stream 保证异步加载执行期间索引张量不被提前释放。
            if host_indices.is_cuda:
                host_indices.record_stream(self.load_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.load_stream)

        self.ack_load_queue.append(
            HiCacheAck(
                start_event=producer_event.start_event,
                finish_event=producer_event.finish_event,
                node_ids=op.node_ids,
            )
        )
        return producer_id

    def evict_device(self, device_indices: torch.Tensor) -> int:
        # 中译：从 L1 设备池释放（驱逐）这些槽位，返回释放的数量。
        self.mem_pool_device_allocator.free(device_indices)
        return len(device_indices)

    def evict_host(self, host_indices: torch.Tensor, backup_only: bool = True) -> int:
        # 中译：从 L2 主机池释放（驱逐）这些槽位，返回释放的数量。
        #       目前仅支持 backup_only 策略（其它驱逐策略尚未实现）。
        if not backup_only:
            raise ValueError("Other eviction policies are not supported yet.")

        self.mem_pool_host.free(host_indices)
        return len(host_indices)

    def set_draft_kv_pool(self, draft_device_pool, draft_host_pool) -> None:
        """Register draft KV pools so L2/L3 ops piggyback draft transfers.

        中译：登记草稿（draft）KV 池（投机解码用），使其 L2/L3 搬运搭车在目标 KV 操作上一起完成。
              若存储后端已挂载则立即接好草稿 L3 的 IO 路径，否则延迟到 attach 时再接。
        """
        self.has_draft = True
        self.mem_pool_device_draft = draft_device_pool
        self.mem_pool_host_draft = draft_host_pool
        logger.info(
            "HiCache draft KV registered: %s (host %d slots)",
            type(draft_device_pool).__name__,
            draft_host_pool.size,
        )

        # If storage is already attached, wire up the draft I/O path now.
        # Otherwise this will be deferred until attach_storage_backend().
        self._maybe_register_draft_with_storage()

    def _maybe_register_draft_with_storage(self) -> None:
        """Pick the draft L3 IO implementation.

        中译：根据存储后端类型选择草稿 KV 的 L3 读写实现。
              - mooncake：用多池零拷贝 v2 接口（需先注册草稿 host 池；不支持 split_heads 时禁用）；
              - hf3fs/eic/nixl/simm：暂不支持草稿池注册，禁用草稿 L3；
              - 其它通用后端：用 generic 实现（以 `{hash}.draft` 作为键，与目标页区分）。
              未启用草稿或未启用存储时，将草稿读写函数置空（不参与）。
        """
        self.draft_page_get_func = None
        self.draft_page_set_func = None
        if not self.has_draft or not self.enable_storage:
            return

        backend = self.storage_backend_type

        # Multi-pool zero-copy backends.
        if backend == "mooncake":
            if self.storage_config.should_split_heads:
                logger.warning(
                    "HiCache draft L3 disabled: should_split_heads not yet "
                    "supported on the mooncake v2 path."
                )
                return
            self.storage_backend.register_mem_host_pool_v2(
                self.mem_pool_host_draft, PoolName.DRAFT
            )
            self.draft_page_get_func = self._draft_page_get_v2
            self.draft_page_set_func = self._draft_page_set_v2
            return

        # TODO: support "hf3fs", "eic", "nixl", "simm"
        if backend in {"hf3fs", "eic", "nixl", "simm"}:
            logger.warning(
                "HiCache draft L3 disabled: backend %s does not yet support "
                "draft pool registration.",
                backend,
            )
            return

        # Generic backends.
        self.draft_page_get_func = self._draft_page_get_generic
        self.draft_page_set_func = self._draft_page_set_generic

    def prefetch(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> PrefetchOperation:
        """
        Prefetch KV caches from storage backend to host memory.

        中译：从 L3 存储后端预取 KV 缓存到 L2 主机内存。仅创建预取操作并投入预取队列，
              由后台 prefetch_thread 异步执行；返回该操作句柄供上层跟踪/终止。
        """
        operation = PrefetchOperation(
            request_id, host_indices, new_input_tokens, last_hash, prefix_keys
        )
        self.prefetch_queue.put(operation)
        return operation

    def terminate_prefetch(self, operation):
        # 中译：终止一个预取操作，返回其已完成的 token 数与已命中的哈希键列表（供上层登记/回收）。
        operation.mark_terminate()
        return operation.completed_tokens, operation.hash_value

    def append_host_mem_release(self, host_indices: torch.Tensor):
        # 中译：把待释放的 host 槽位按页拆分后投入释放队列，由主调度线程统一回收 host 内存。
        if host_indices.numel() == 0:
            return
        pages = host_indices.split(self.mem_pool_host.page_size)
        for page in pages:
            self.host_mem_release_queue.put(page)

    def _page_get_zero_copy(
        self, operation, hash_values, host_indices, extra_info=None
    ):
        # 中译：零拷贝预取——后端直接把数据写入 host 池对应槽位（batch_get_v1）。
        #       从头连续累计成功页数，一旦遇到失败页即停止（前缀必须连续命中）。
        results = self.storage_backend.batch_get_v1(
            hash_values, host_indices, extra_info
        )
        inc = 0
        for i in range(len(hash_values)):
            if not results[i]:
                logger.warning(
                    f"Prefetch operation {operation.request_id} failed to retrieve page {hash_values[i]}."
                )
                break
            inc += self.page_size
        operation.increment(inc)

    # todo: deprecate
    # 中译：通用预取（非零拷贝，待废弃）——先取出页数据，再逐页写入 host 池。
    #       关键顺序：必须先 set 数据、再 increment 完成数，否则该页可能在写入前被读取。
    def _generic_page_get(self, operation, hash_values, host_indices, extra_info=None):
        dummy_page_dst = [
            self.mem_pool_host.get_dummy_flat_data_page() for _ in hash_values
        ]
        page_data = self.storage_backend.batch_get(hash_values, dummy_page_dst)
        if page_data is None:
            return
        for i in range(len(hash_values)):
            if page_data[i] is None:
                logger.warning(
                    f"Prefetch operation {operation.request_id} failed to retrieve page {hash_values[i]}."
                )
                break
            # Must set the data before increasing the completed tokens.
            # Otherwise this page may be read before being set.
            self.mem_pool_host.set_from_flat_data_page(
                host_indices[i * self.page_size],
                page_data[i],
            )
            if not operation.increment(self.page_size):
                break  # Operation terminated by controller

    def _page_transfer(self, operation):
        # 中译：把一次预取操作命中的所有页从 L3 拉到 L2，分批（STORAGE_BATCH_SIZE）传输。
        #       每批：先尽力读草稿 L3（须在发布目标完成前，避免竞态把目标 KV 先加载回来），
        #       再读目标页；若本批未达预期完成数则终止操作。
        # Transfer batch by batch
        prefix_keys = operation.prefix_keys
        for i in range(0, len(operation.hash_value), STORAGE_BATCH_SIZE):
            batch_hashes = operation.hash_value[i : i + STORAGE_BATCH_SIZE]
            batch_host_indices = operation.host_indices[
                i * self.page_size : (i + len(batch_hashes)) * self.page_size
            ]

            # Best-effort draft L3 read before publishing target completion.
            # Otherwise wait_complete can race and load back target KV before
            # draft KV reaches host memory.
            if self.has_draft:
                self._draft_page_get(batch_hashes, batch_host_indices)

            prev_completed_tokens = operation.completed_tokens
            # Get one batch token, and update the completed_tokens if succeed
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            self.page_get_func(operation, batch_hashes, batch_host_indices, extra_info)
            # Check termination
            if (
                operation.completed_tokens
                != prev_completed_tokens + len(batch_hashes) * self.page_size
            ):
                operation.mark_terminate()
                break  # Some operations fail or operation terminated by controller

            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes

    def prefetch_io_aux_func(self):
        """
        Auxiliary function conducting IO operations for prefetching.

        中译：预取的 IO 辅助线程主体。不断从 prefetch_buffer 取出操作并执行实际页传输；
              传输结束后把「未完成部分」的 host 槽位释放掉（无论是失败还是被控制器终止）。
        """
        while not self.storage_stop_event.is_set():
            try:
                operation = self.prefetch_buffer.get(block=True, timeout=1)
                if operation is None:
                    continue
                self._page_transfer(operation)
                # operation terminated by controller, release pre-allocated memory
                self.append_host_mem_release(
                    operation.host_indices[operation.completed_tokens :]
                )
            except Empty:
                continue

    def prefetch_rate_limited(self) -> bool:
        """
        Rate limit the prefetching operations to avoid overwhelming the storage backend.

        中译：预取限流——避免压垮存储后端或耗尽 host 内存。
              当前策略：被预取锁定的 token 数达到容量上限即拒绝（返回 True 表示应限流）。
        """
        # cancel prefetch if too much memory is occupied
        # 中译：占用过多内存时取消预取。
        if self.prefetch_tokens_occupied >= self.prefetch_capacity_limit:
            return True
        # todo: more sophisticated rate limiting based on storage backend performance
        return False

    def _storage_hit_query(self, operation) -> tuple[list[str], int]:
        # 中译：查询某次预取在 L3 中的「前缀命中」情况。逐页计算哈希键（每页基于上一页哈希链式生成），
        #       分批调用 batch_exists 探测存在性；命中必须连续——一旦某批未全部命中即停止。
        #       返回命中的哈希键列表与命中的 token 总数。
        last_hash = operation.last_hash
        tokens_to_fetch = operation.token_ids
        prefix_keys = operation.prefix_keys.copy() if operation.prefix_keys else None

        storage_query_count = 0
        hash_value = []

        for start in range(
            0, len(tokens_to_fetch), self.page_size * STORAGE_BATCH_SIZE
        ):
            end = min(start + self.page_size * STORAGE_BATCH_SIZE, len(tokens_to_fetch))
            batch_tokens = tokens_to_fetch[start:end]
            batch_hashes = []
            for i in range(0, len(batch_tokens), self.page_size):
                last_hash = self.get_hash_str(
                    batch_tokens[i : i + self.page_size], last_hash
                )
                batch_hashes.append(last_hash)
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            hit_page_num = self.storage_backend.batch_exists(batch_hashes, extra_info)
            hash_value.extend(batch_hashes[:hit_page_num])
            storage_query_count += hit_page_num * self.page_size
            if hit_page_num < len(batch_hashes):
                break
            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes

        return hash_value, storage_query_count

    def prefetch_thread_func(self):
        """
        Manage prefetching operations from storage backend to host memory.

        中译：预取主线程。对每个预取操作：先查 L3 命中页数，再跨 rank all_reduce 取 MIN 对齐
              （保证各 rank 预取页数一致）；命中不足阈值则撤销预取并释放预分配的 host 内存，
              否则裁剪 host_indices/hash_value 到命中范围并投入 prefetch_buffer 交由 IO 辅助线程传输。
        """
        self.prefetch_buffer = Queue()
        self.prefetch_io_aux_thread = threading.Thread(
            target=self.prefetch_io_aux_func, daemon=True
        )
        self.prefetch_io_aux_thread.start()
        while (not self.storage_stop_event.is_set()) or not self.prefetch_queue.empty():
            try:
                operation = self.prefetch_queue.get(block=True, timeout=1)
                if operation is None:
                    continue
                hash_value, storage_hit_count = self._storage_hit_query(operation)
                storage_hit_count_tensor = torch.tensor(
                    storage_hit_count, dtype=torch.int
                )
                self._all_reduce_prefetch_groups(
                    storage_hit_count_tensor, torch.distributed.ReduceOp.MIN
                )
                storage_hit_count = storage_hit_count_tensor.item()

                if storage_hit_count < self.prefetch_threshold:
                    # not to prefetch if not enough benefits
                    # 中译：命中收益不足阈值——撤销该请求的预取并归还预分配的 host 内存。
                    self.prefetch_revoke_queue.put(operation.request_id)
                    self.append_host_mem_release(operation.host_indices)
                    logger.debug(
                        f"Revoking prefetch for request {operation.request_id} due to insufficient hits ({storage_hit_count})."
                    )
                else:
                    operation.hash_value = hash_value[
                        : (storage_hit_count // self.page_size)
                    ]
                    # free the pre-allocated memory for pages that are not hit
                    # 中译：未命中的那部分页对应的预分配 host 内存先行释放，只保留命中范围。
                    self.append_host_mem_release(
                        operation.host_indices[storage_hit_count:]
                    )
                    operation.host_indices = operation.host_indices[:storage_hit_count]
                    logger.debug(
                        f"Prefetching {len(operation.hash_value)} pages for request {operation.request_id}."
                    )
                    self.prefetch_buffer.put(operation)

            except Empty:
                continue

    def write_storage(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> int:
        """
        Write KV caches from host memory to storage backend.

        中译：把 KV 缓存从 L2 主机内存备份到 L3 存储后端。仅创建备份操作投入 backup_queue，
              由后台 backup_thread 异步执行；返回操作 id 供上层跟踪。
        """
        operation = StorageOperation(
            host_indices, token_ids, hash_value=hash_value, prefix_keys=prefix_keys
        )
        self.backup_queue.put(operation)
        return operation.id

    # todo: deprecate
    def _generic_page_set(self, hash_values, host_indices, extra_info=None) -> bool:
        # 中译：通用页写出（非零拷贝，待废弃）——从 host 池取出各页数据后 batch_set 到后端。
        data = [
            self.mem_pool_host.get_data_page(host_indices[i * self.page_size])
            for i in range(len(hash_values))
        ]
        return self.storage_backend.batch_set(hash_values, data)

    def _page_set_zero_copy(self, hash_values, host_indices, extra_info=None) -> bool:
        # 中译：零拷贝页写出——后端直接从 host 池槽位读取并写入（batch_set_v1），全部成功才算成功。
        return all(
            self.storage_backend.batch_set_v1(hash_values, host_indices, extra_info)
        )

    def _draft_page_set(self, hash_values, host_indices) -> None:
        """Best-effort write draft KV pages to L3 alongside the target backup.

        中译：尽力而为地把草稿 KV 页随目标备份一起写入 L3；未配置或出错则静默跳过（不影响主流程）。
        """
        if self.draft_page_set_func is None:
            return
        try:
            self.draft_page_set_func(hash_values, host_indices)
        except Exception:
            logger.debug(
                "Draft L3 write failed (best-effort), skipping.", exc_info=True
            )

    def _draft_page_get(self, hash_values, host_indices) -> None:
        """Best-effort read draft KV pages from L3 (mirrors `_draft_page_set`).

        中译：尽力而为地从 L3 读回草稿 KV 页（与 _draft_page_set 对称）；出错静默跳过。
        """
        if self.draft_page_get_func is None:
            return
        try:
            self.draft_page_get_func(hash_values, host_indices)
        except Exception:
            logger.debug("Draft L3 read failed (best-effort), skipping.", exc_info=True)

    def _draft_page_set_v2(self, hash_values, host_indices) -> None:
        # 中译：多池零拷贝 v2 写出——以 DRAFT 池名打包 PoolTransfer 交给后端 batch_set_v2。
        self.storage_backend.batch_set_v2(
            [
                PoolTransfer(
                    name=PoolName.DRAFT,
                    host_indices=host_indices,
                    keys=list(hash_values),
                )
            ]
        )

    def _draft_page_get_v2(self, hash_values, host_indices) -> None:
        # 中译：多池零拷贝 v2 读回——以 DRAFT 池名打包 PoolTransfer 交给后端 batch_get_v2。
        self.storage_backend.batch_get_v2(
            [
                PoolTransfer(
                    name=PoolName.DRAFT,
                    host_indices=host_indices,
                    keys=list(hash_values),
                )
            ]
        )

    def _draft_page_set_generic(self, hash_values, host_indices) -> None:
        # `{hash}.draft` mirrors HiCacheStorage._get_component_key's
        # `{key}.{pool_name}` convention so target/draft pages never collide.
        # 中译：草稿键用 `{hash}.draft`，沿用 HiCacheStorage._get_component_key 的
        #       `{key}.{pool_name}` 约定，确保目标页与草稿页的键永不冲突。
        draft_keys = [f"{h}.{PoolName.DRAFT}" for h in hash_values]
        draft_data = [
            self.mem_pool_host_draft.get_data_page(host_indices[i * self.page_size])
            for i in range(len(draft_keys))
        ]
        self.storage_backend.batch_set(draft_keys, draft_data)

    def _draft_page_get_generic(self, hash_values, host_indices) -> None:
        # 中译：通用草稿读回——按 `{hash}.draft` 键 batch_get 取回草稿页，逐页写入草稿 host 池。
        draft_keys = [f"{h}.{PoolName.DRAFT}" for h in hash_values]
        draft_dummy = [
            self.mem_pool_host_draft.get_dummy_flat_data_page() for _ in draft_keys
        ]
        draft_pages = self.storage_backend.batch_get(draft_keys, draft_dummy)
        if draft_pages is None:
            return
        for i, p in enumerate(draft_pages):
            if p is not None:
                self.mem_pool_host_draft.set_from_flat_data_page(
                    host_indices[i * self.page_size], p
                )

    # Backup batch by batch
    def _page_backup(self, operation):
        # Backup batch by batch
        # 中译：把一次备份操作的所有页分批写入 L3。每批：写目标页 -> 失败则告警并中止；
        #       成功后尽力把对应草稿页也写入 L3；最后累加已完成 token 数。
        prefix_keys = operation.prefix_keys
        for i in range(0, len(operation.hash_value), STORAGE_BATCH_SIZE):
            batch_hashes = operation.hash_value[i : i + STORAGE_BATCH_SIZE]
            batch_host_indices = operation.host_indices[
                i * self.page_size : (i + len(batch_hashes)) * self.page_size
            ]
            # Set one batch token, and record if success.
            # todo: allow partial success
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            success = self.page_set_func(batch_hashes, batch_host_indices, extra_info)
            if not success:
                logger.warning(
                    f"Write page to storage: {len(batch_hashes)} pages failed."
                )
                break

            # Best-effort draft L3 write alongside target.
            if self.has_draft:
                self._draft_page_set(batch_hashes, batch_host_indices)

            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes
            operation.completed_tokens += self.page_size * len(batch_hashes)

    def backup_thread_func(self):
        """
        Manage backup operations from host memory to storage backend.

        中译：备份主线程。不断从 backup_queue 取出操作并执行页写出（MLA 模型下非 rank0 会跳过实际写出），
              完成后把操作放入 ack_backup_queue 供主调度线程回执处理。
        """
        while not self.storage_stop_event.is_set():
            try:
                operation = self.backup_queue.get(block=True, timeout=1)
                if operation is None:
                    continue

                if not self.backup_skip:
                    self._page_backup(operation)
                self.ack_backup_queue.put(operation)

            except Empty:
                continue
