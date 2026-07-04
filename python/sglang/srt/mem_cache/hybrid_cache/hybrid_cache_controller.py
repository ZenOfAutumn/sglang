from __future__ import annotations

import json
import logging
import os
import threading
import time
from queue import Queue
from typing import TYPE_CHECKING, Any, Callable, List, Optional

import torch

from sglang.srt.managers.cache_controller import CacheOperation as BaseCacheOperation
from sglang.srt.managers.cache_controller import (
    HiCacheAck,
)
from sglang.srt.managers.cache_controller import (
    HiCacheController as BaseHiCacheController,
)
from sglang.srt.managers.cache_controller import (
    LayerDoneCounter,
)
from sglang.srt.managers.cache_controller import (
    StorageOperation as BaseStorageOperation,
)
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.memory_pool_host import PoolEntry
from sglang.srt.utils import get_device_module

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

logger = logging.getLogger(__name__)
device_module = get_device_module()

# 本模块是「混合模型」（如线性注意力 / Mamba：KV pool + 额外状态 pool）场景下的
# HiCache 控制器实现。相比通用的 HiCacheController，它在 device/host/storage 三级
# 搬运的基础上，额外支持「多 pool 协同搬运」：KV 主池之外还挂接若干 extra pool
# （如 Mamba 状态、SWA 索引等），并通过 PoolTransfer 统一描述与调度这些附加搬运。


class CacheOperation(BaseCacheOperation):
    """一次 device<->host 搬运操作（在通用 CacheOperation 基础上扩展 pool_transfers）。

    pool_transfers：除 KV 主池外，本次操作还需要一并搬运的「附加池」列表。混合模型里
    一个逻辑节点的完整状态可能分散在多个池中，需要一起搬运才能保持一致。
    """

    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        priority: Optional[int] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        super().__init__(host_indices, device_indices, node_id, priority)
        self.pool_transfers = pool_transfers

    @staticmethod
    def merge_pool_transfers(
        ops: List[CacheOperation],
    ) -> Optional[list[PoolTransfer]]:
        # 把多个操作里「同名同来源」的 PoolTransfer 合并成一个（各自的索引拼接起来），
        # 以便批量执行一次搬运，减少调度开销。
        # 分组键为 (池名, 索引来源池)：只有这两者都相同的 transfer 才能安全拼接。
        grouped: dict[tuple[PoolName, Optional[PoolName]], list[PoolTransfer]] = {}
        for op in ops:
            for t in op.pool_transfers or []:
                grouped.setdefault((t.name, t.indices_from_pool), []).append(t)
        if not grouped:
            return None

        def cat_or_none(tensors):
            # 拼接一组张量；若全为 None 则返回 None（该字段本次无内容）。
            parts = [x for x in tensors if x is not None]
            return torch.cat(parts) if parts else None

        return [
            PoolTransfer(
                name=ts[0].name,
                host_indices=cat_or_none(t.host_indices for t in ts),
                device_indices=cat_or_none(t.device_indices for t in ts),
                keys=[k for t in ts if t.keys for k in t.keys] or None,
                hit_policy=ts[0].hit_policy,
                indices_from_pool=ts[0].indices_from_pool,
            )
            for ts in grouped.values()
        ]

    @staticmethod
    def merge_ops(ops: List[CacheOperation]) -> CacheOperation:
        # 把队列里的多个搬运操作合并成一个大操作，一次性提交给 device 流执行。
        if len(ops) == 1:
            return ops[0]
        # KV 主池的 host/device 索引直接首尾拼接。
        host_indices = torch.cat([op.host_indices for op in ops])
        device_indices = torch.cat([op.device_indices for op in ops])
        node_ids = []
        # 合并后的优先级取各操作中的最小值（数值越小优先级越高）。
        priority = min(op.priority for op in ops)
        for op in ops:
            node_ids.extend(op.node_ids)
        merged = CacheOperation(
            host_indices,
            device_indices,
            -1,
            priority,
            pool_transfers=CacheOperation.merge_pool_transfers(ops),
        )
        merged.node_ids = node_ids
        return merged


class StorageOperation(BaseStorageOperation):
    """一次 host<->storage（L3）搬运操作，扩展了多池支持。

    pool_transfers：本次 storage 读写涉及的附加池搬运描述；
    pool_storage_result：记录各池实际命中/写入的页数，供上层统计与截断对齐。
    """

    def __init__(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        super().__init__(host_indices, token_ids, last_hash, hash_value, prefix_keys)
        self.pool_transfers = pool_transfers
        self.pool_storage_result = PoolTransferResult.empty()


class PrefetchOperation(StorageOperation):
    """从 L3 storage 预取到 host 的操作，带「可增量、可提前终止」的线程安全簿记。

    预取由 storage 线程异步推进，主线程可随时请求终止（尽力而为策略）。通过一把锁
    协调 completed_tokens 的累加与终止标志，避免竞态。
    """

    def __init__(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        self.request_id = request_id
        self._lock = threading.Lock()
        self._terminated_flag = False
        self.start_time = time.monotonic()
        super().__init__(
            host_indices,
            token_ids,
            last_hash,
            prefix_keys=prefix_keys,
            pool_transfers=pool_transfers,
        )
        # 若没有附加池搬运，则附加池部分一开始就视为已完成。
        self.pool_transfers_done = not bool(pool_transfers)

    def increment(self, num_tokens: int):
        # 累加已完成的 token 数；若已被标记终止则拒绝累加并返回 False。
        with self._lock:
            if self._terminated_flag:
                return False
            self.completed_tokens += num_tokens
            return True

    def mark_terminate(self):
        # 标记该预取为「终止」，使其在下一次进度检查时尽快收尾。
        with self._lock:
            self._terminated_flag = True

    def is_terminated(self) -> bool:
        # 查询该预取是否已被标记终止。
        return self._terminated_flag


class HybridCacheController(BaseHiCacheController):
    """混合模型的分级缓存控制器。

    在通用 HiCacheController 基础上，额外管理「KV 主池 + 若干 extra pool」的多池协同
    搬运（device<->host<->storage）。典型适用于 Mamba / 线性注意力、SWA 等需要同时维护
    KV 与额外状态的模型。
    """

    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        mem_pool_host: Any,
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
        transfer_layer_num: Optional[int] = None,
        enable_storage_metrics: bool = False,
    ):
        # 先缓存一份启动时的 storage_backend，稍后手动 attach（下方父类先传 None，避免在
        # 额外池尚未就绪时就启动 storage 线程）。
        startup_storage_backend = storage_backend
        # 各额外池各自的 host 内存释放队列（KV 主池的释放队列由父类维护）。
        self.extra_host_mem_release_queues: dict[PoolName, Queue[torch.Tensor]] = {}
        super().__init__(
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            mem_pool_host=mem_pool_host,
            page_size=page_size,
            tp_group=tp_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            pp_group=pp_group,
            write_policy=write_policy,
            io_backend=io_backend,
            # 故意传 None：先完成基类初始化，等额外池就绪后再由下方 attach_storage_backend 启动。
            storage_backend=None,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
        )
        # 覆盖 layer_num：混合模型需要搬运「所有层」（例如线性模型的 KV + Mamba），
        # 而不只是 full_kv_pool 报告的全注意力层。因此重建逐层完成计数器。
        if transfer_layer_num is not None and transfer_layer_num != self.layer_num:
            self.layer_num = transfer_layer_num
            self.layer_done_counter = LayerDoneCounter(self.layer_num)

        # 若启动时已指定后端，现在（额外池已就绪）才真正 attach，并把各 host 池注册给后端。
        if startup_storage_backend is not None:
            self.attach_storage_backend(
                storage_backend=startup_storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=model_name,
                storage_backend_extra_config=storage_backend_extra_config,
                host_pools=getattr(mem_pool_host, "entries", None),
            )

    def _start_storage_threads(self):
        # 启动基类的 storage 线程后，再为各额外池初始化各自的 host 内存释放队列。
        super()._start_storage_threads()
        self._init_extra_host_mem_release_queues()

    def attach_storage_backend(
        self,
        storage_backend: str,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        host_pools: Optional[list[PoolEntry]] = None,
    ):
        # 先走基类的 attach（启动 storage 线程、启用 prefetch/backup 路径）。
        super().attach_storage_backend(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
        )

        # 再把每个 host 池（含 KV 主池与各额外池）注册到后端，使后端能直接向这些
        # host 内存做零拷贝/DMA 搬运。
        for entry in host_pools or []:
            self.storage_backend.register_mem_host_pool_v2(entry.host_pool, entry.name)

    @staticmethod
    def parse_storage_backend_extra_config(
        storage_backend_extra_config: Optional[str],
    ) -> tuple[dict, int, float, float, bool]:
        # 解析 storage 后端的 extra config，抽取预取相关参数，剩余键继续透传给后端。
        # 输入可以是 JSON 字符串，也可以是以 "@" 为前缀的 json/toml/yaml 文件路径。
        extra_config = {}
        if storage_backend_extra_config:
            if storage_backend_extra_config.startswith("@"):
                # 从 json/toml/yaml 文件读取配置（根据扩展名选择解析器）。
                path = storage_backend_extra_config[1:]
                ext = os.path.splitext(path)[1].lower()
                with open(path, "rb" if ext == ".toml" else "r") as f:
                    if ext == ".json":
                        extra_config = json.load(f)
                    elif ext == ".toml":
                        import tomllib

                        extra_config = tomllib.load(f)
                    elif ext in (".yaml", ".yml"):
                        import yaml

                        extra_config = yaml.safe_load(f)
                    else:
                        raise ValueError(
                            f"Unsupported config file {path} (config format: {ext})"
                        )
            else:
                # 直接从 JSON 字符串解析配置。
                extra_config = json.loads(storage_backend_extra_config)

        # 从 extra_config 中弹出预取相关参数（弹出后剩余键继续透传给后端）。
        prefetch_threshold = extra_config.pop("prefetch_threshold", 256)
        prefetch_timeout_base = extra_config.pop("prefetch_timeout_base", 1)
        prefetch_timeout_per_ki_token = extra_config.pop(
            "prefetch_timeout_per_ki_token", 0.25
        )
        hicache_storage_pass_prefix_keys = extra_config.pop(
            "hicache_storage_pass_prefix_keys", False
        )

        if not isinstance(prefetch_threshold, int):
            raise ValueError(
                f"prefetch_threshold must be int, got {type(prefetch_threshold).__name__}"
            )
        if not isinstance(prefetch_timeout_base, (int, float)):
            raise ValueError(
                f"prefetch_timeout_base must be number, got {type(prefetch_timeout_base).__name__}"
            )
        if not isinstance(prefetch_timeout_per_ki_token, (int, float)):
            raise ValueError(
                "prefetch_timeout_per_ki_token must be number, got "
                f"{type(prefetch_timeout_per_ki_token).__name__}"
            )
        if not isinstance(hicache_storage_pass_prefix_keys, bool):
            raise ValueError(
                "hicache_storage_pass_prefix_keys must be bool, got "
                f"{type(hicache_storage_pass_prefix_keys).__name__}"
            )

        return (
            extra_config,
            prefetch_threshold,
            float(prefetch_timeout_base),
            float(prefetch_timeout_per_ki_token),
            hicache_storage_pass_prefix_keys,
        )

    def clear_storage_backend(self) -> bool:
        # 清空 L3 storage 后端的全部内容（仅部分后端支持 clear 操作）。
        if not self.enable_storage:
            logger.warning("Hierarchical cache storage backend is not enabled.")
            return False
        if not hasattr(self.storage_backend, "clear"):
            logger.warning(
                "Storage backend %s does not support clear operation.",
                type(self.storage_backend).__name__,
            )
            return False
        self.storage_backend.clear()
        return True

    def _init_extra_host_mem_release_queues(self) -> None:
        # 为每个额外池建立一个独立的 host 内存释放队列。
        # 跳过「主索引锚点」池（anchor）：它的索引与 KV 主池共用，由主池的释放路径统一管理。
        self.extra_host_mem_release_queues = {}
        entries = getattr(self.mem_pool_host, "entries", None) or []
        anchor_entry = getattr(self.mem_pool_host, "anchor_entry", None)
        for entry in entries:
            if entry is anchor_entry or entry.is_primary_index_anchor:
                continue
            self.extra_host_mem_release_queues[entry.name] = Queue()

    def _append_host_mem_release_pages(
        self, release_queue: Queue, host_indices: torch.Tensor, page_size: int
    ) -> None:
        # 把待释放的 host 索引按页拆分后逐页入队（页是释放的最小粒度）。
        if host_indices.numel() == 0:
            return
        for page in host_indices.split(page_size):
            release_queue.put(page)

    def append_host_mem_release(
        self,
        host_indices: Optional[torch.Tensor] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ):
        # 登记待释放的 host 内存（延迟到控制队列排空时统一 free）。
        # KV 主池部分：入主释放队列。
        if host_indices is not None:
            self._append_host_mem_release_pages(
                self.host_mem_release_queue,
                host_indices,
                self.mem_pool_host.page_size,
            )
        # 额外池部分：分别入各自的释放队列。
        for transfer in extra_pools or []:
            if transfer.host_indices is None or transfer.host_indices.numel() == 0:
                continue
            entry = self.mem_pool_host.entry_map.get(transfer.name)
            # 跳过不需单独释放的情形：池不存在、主索引锚点（随 KV 主池释放）、
            # 或索引派生自其他池（不拥有自己的 host 内存）。
            if (
                entry is None
                or entry.is_primary_index_anchor
                or transfer.indices_from_pool is not None
            ):
                continue
            release_queue = self.extra_host_mem_release_queues.get(transfer.name)
            if release_queue is None:
                continue
            self._append_host_mem_release_pages(
                release_queue, transfer.host_indices, entry.host_pool.page_size
            )

    def reset(self):
        # 重置控制器：先走父类重置，再清空 KV 主池与各额外池的释放队列及预取占用计数。
        super().reset()
        if self.enable_storage:
            self.host_mem_release_queue.queue.clear()
            for release_queue in self.extra_host_mem_release_queues.values():
                release_queue.queue.clear()
            self.prefetch_tokens_occupied = 0

    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> Optional[torch.Tensor]:
        # 发起一次 device→host 写穿透：先在 host 主池分配落地空间，再为各额外池分配 host 索引。
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        # 为 extra pool 自动分配 host 索引（已有的保留）；失败时回滚、释放已分配的 host 空间。
        pool_transfers = self._resolve_pool_transfers_allocation(
            extra_pools,
            alloc_host=True,
            kv_device_indices=device_indices,
            kv_host_indices=host_indices,
        )
        if pool_transfers is None and extra_pools:
            self.mem_pool_host.free(host_indices)
            return None

        # 入写队列，等待 start_writing 批量合并后提交到写流。
        self.write_queue.append(
            CacheOperation(
                host_indices,
                device_indices,
                node_id,
                priority,
                pool_transfers=pool_transfers or None,
            )
        )
        self.start_writing()
        return host_indices

    def start_writing(self) -> None:
        # 把写队列里的操作合并为一，在专用写流上一次性把所有层的 KV（及额外池）从 device 备份到 host。
        if not self.write_queue:
            return
        op = CacheOperation.merge_ops(self.write_queue)
        # 把索引搬到执行设备上，并得到归一化后的额外池搬运描述。
        host_indices, device_indices, resolved_pool_transfers = (
            self.move_hybrid_indices(op)
        )
        self.write_queue.clear()
        # start/finish 事件用于跟踪本次异步搬运的开始与完成。
        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_stream):
            start_event.wait(self.write_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_indices,
                device_indices,
                self.io_backend,
                pool_transfers=resolved_pool_transfers,
            )
            # 若启用了投机解码的 draft 模型，同样备份其 draft KV。
            if self.has_draft and host_indices.numel() > 0:
                self.mem_pool_host_draft.backup_from_device_all_layer(
                    self.mem_pool_device_draft,
                    host_indices,
                    device_indices,
                    self.io_backend,
                )
            finish_event.record()
            # 在写流上登记所有参与搬运的张量，避免它们在异步搬运完成前被提前回收。
            self._record_transfer_indices_on_stream(
                self.write_stream,
                host_indices,
                device_indices,
                resolved_pool_transfers,
            )
        # 登记写完成 ack，由上层 writing_check 收割。
        self.ack_write_queue.append(HiCacheAck(start_event, finish_event, op.node_ids))

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> Optional[torch.Tensor]:
        # 发起一次 host→device 回载：在 device 侧为 KV 分配目标空间，再为各额外池分配 device 索引。
        need_load_kv = host_indices.numel() > 0

        # 优先用「全注意力分配器」（若存在），否则回退到默认 device 分配器。
        full_allocator = getattr(
            self.mem_pool_device_allocator,
            "full_attn_allocator",
            self.mem_pool_device_allocator,
        )
        if not need_load_kv:
            # 无 KV 需回载（仅额外池）：用空张量占位。
            device_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        else:
            device_indices = full_allocator.alloc(len(host_indices))
            if device_indices is None:
                return None

        # 为 extra pool 自动分配 device 索引（已有的保留）；失败时回滚并释放已分配的 KV device 空间。
        pool_transfers = self._resolve_pool_transfers_allocation(
            extra_pools,
            alloc_host=False,
            kv_device_indices=device_indices,
            kv_host_indices=host_indices,
        )
        if pool_transfers is None and extra_pools:
            if need_load_kv:
                full_allocator.free(device_indices)
            return None

        # 入回载队列，等待 start_loading 批量合并后提交。
        self.load_queue.append(
            CacheOperation(
                host_indices,
                device_indices,
                node_id,
                priority,
                pool_transfers=pool_transfers or None,
            )
        )
        return device_indices

    def start_loading(self) -> int:
        # 把回载队列合并为一，在专用回载流上「逐层」把 KV（及额外池）从 host 回载到 device。
        # 逐层完成时通过 layer_done_counter 通知消费方，使计算可与回载流水线重叠。
        if not self.load_queue:
            return -1
        # 登记一个 producer，拿到它的逐层事件组，供下游按 consumer index 追踪完成情况。
        producer_id = self.layer_done_counter.update_producer()
        op = CacheOperation.merge_ops(self.load_queue)
        host_indices, device_indices, resolved_pool_transfers = (
            self.move_hybrid_indices(op)
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
                    pool_transfers=resolved_pool_transfers,
                )
                # draft 模型（如果有）同样逐层回载，但层数可能少于主模型，需越界保护。
                if (
                    self.has_draft
                    and host_indices.numel() > 0
                    and i < self.mem_pool_host_draft.layer_num
                ):
                    self.mem_pool_host_draft.load_to_device_per_layer(
                        self.mem_pool_device_draft,
                        host_indices,
                        device_indices,
                        i,
                        self.io_backend,
                    )
                # 标记第 i 层回载完成，唤醒等待该层的消费方。
                producer_event.complete(i)
            self._record_transfer_indices_on_stream(
                self.load_stream,
                host_indices,
                device_indices,
                resolved_pool_transfers,
            )
        self.ack_load_queue.append(
            HiCacheAck(
                producer_event.start_event,
                producer_event.finish_event,
                op.node_ids,
            )
        )
        return producer_id

    def _record_transfer_indices_on_stream(
        self,
        stream: torch.Stream,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ) -> None:
        # 对参与搬运的 CUDA 张量调用 record_stream，告诉分配器「该张量仍被本流使用」，
        # 避免在异步搬运尚未完成时内存被提前复用（造成数据损坏）。
        if host_indices.is_cuda:
            host_indices.record_stream(stream)
        if device_indices.is_cuda:
            device_indices.record_stream(stream)
        for transfer in pool_transfers or []:
            if transfer.host_indices is not None and transfer.host_indices.is_cuda:
                transfer.host_indices.record_stream(stream)
            if transfer.device_indices is not None and transfer.device_indices.is_cuda:
                transfer.device_indices.record_stream(stream)

    def prefetch(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> PrefetchOperation:
        # 创建一个预取操作并入预取队列，交由 storage 线程异步从 L3 拉取到 host。
        operation = PrefetchOperation(
            request_id,
            host_indices,
            new_input_tokens,
            last_hash,
            prefix_keys=prefix_keys,
            pool_transfers=extra_pools,
        )
        self.prefetch_queue.put(operation)
        return operation

    def write_storage(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> int:
        # 创建一个 host→storage 的备份操作并入备份队列，返回操作 id 供上层追踪 ack。
        operation = StorageOperation(
            host_indices,
            token_ids,
            hash_value=hash_value,
            prefix_keys=prefix_keys,
            pool_transfers=extra_pools,
        )
        self.backup_queue.put(operation)
        return operation.id

    def _storage_hit_query(self, operation) -> tuple[list[str], int]:
        # 向 L3 storage 查询：本次 token 序列有多长的前缀已存在于后端（用于决定预取长度）。
        # 逐页链式计算哈希：每页的哈希依赖上一页的哈希（last_hash），形成前缀链。
        last_hash = operation.last_hash
        hash_value = []
        for start in range(0, len(operation.token_ids), self.page_size):
            last_hash = self.get_hash_str(
                operation.token_ids[start : start + self.page_size], last_hash
            )
            hash_value.append(last_hash)

        extra_info = HiCacheStorageExtraInfo(
            prefix_keys=operation.prefix_keys.copy() if operation.prefix_keys else None
        )
        if operation.pool_transfers:
            # 多池场景：用 v2 接口，同时查询 KV 与各额外池的命中情况。
            hit_result = self.storage_backend.batch_exists_v2(
                hash_value, operation.pool_transfers, extra_info
            )
        else:
            # 纯 KV 场景：只查 KV 命中页数。
            kv_hit_count = self.storage_backend.batch_exists(hash_value, extra_info)
            hit_result = PoolTransferResult(
                kv_hit_pages=kv_hit_count, extra_pool_hit_pages={}
            )

        kv_hit_pages = hit_result.kv_hit_pages
        operation.pool_storage_result.update_kv_hit_pages(kv_hit_pages)

        # 返回：命中那几页的哈希列表，以及换算成 token 数的命中长度。
        return (
            hash_value[:kv_hit_pages],
            kv_hit_pages * self.page_size,
        )

    def move_hybrid_indices(
        self, operation: CacheOperation
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[list[PoolTransfer]]]:
        # 把 KV 主池与各额外池的 host/device 索引都搬到执行设备上，并返回归一化后的搬运描述。
        host_indices, device_indices = self.move_indices(
            operation.host_indices, operation.device_indices
        )
        resolved_pool_transfers = None
        if operation.pool_transfers:
            resolved_pool_transfers = []
            for transfer in operation.pool_transfers:
                transfer_host_indices, transfer_device_indices = self.move_indices(
                    transfer.host_indices, transfer.device_indices
                )
                # 保持原始 PoolTransfer 不变：因为归属于 radix 树的搬运可能仍引用着
                # 树上的 host 状态。控制器只需要一份「执行时的归一化副本」。
                resolved_pool_transfers.append(
                    PoolTransfer(
                        name=transfer.name,
                        host_indices=transfer_host_indices,
                        device_indices=transfer_device_indices,
                        keys=transfer.keys,
                        hit_policy=transfer.hit_policy,
                        indices_from_pool=transfer.indices_from_pool,
                    )
                )
        return host_indices, device_indices, resolved_pool_transfers

    def _page_transfer(self, operation):
        # 先搬 KV 主池 —— 它决定了实际完成的页数。
        super()._page_transfer(operation)

        # 额外池只在 KV 完全完成后才搬。若 KV 提前终止（IO 失败、超时、TP 不一致），
        # 则完全跳过额外 IO，以避免数据错位。
        kv_completed_pages = operation.completed_tokens // self.page_size
        if operation.pool_transfers and kv_completed_pages == len(operation.hash_value):
            self._sync_trailing_keys(
                operation.pool_transfers, operation.hash_value, kv_completed_pages
            )
            self._resolve_sidecar_derived_pool_transfers(operation)
            results = self.storage_backend.batch_get_v2(operation.pool_transfers)
            operation.pool_storage_result.update_extra_pool_hit_pages(results)
        operation.pool_transfers_done = True

    def _page_backup(self, operation):
        # 先备份额外池（解析派生搬运后批量写入）。
        if operation.pool_transfers:
            self._resolve_sidecar_derived_pool_transfers(operation)
            results = self.storage_backend.batch_set_v2(operation.pool_transfers)
            operation.pool_storage_result.update_extra_pool_hit_pages(results)

        # 再备份 KV 主池。
        super()._page_backup(operation)

    def _resolve_sidecar_derived_pool_transfers(self, operation):
        # 解析「派生型」额外池：它们不拥有自己的索引/键，而是从另一个源池（KV 或另一个
        # 额外池）借用 host_indices 与 keys。在真正读写 storage 前，把这些字段从源池填好。
        for transfer in operation.pool_transfers:
            if transfer.indices_from_pool is None:
                continue
            if transfer.indices_from_pool != PoolName.KV:
                # 源是另一个额外池：在本次搬运列表里找到那个「拥有自己索引」的源 transfer。
                source = next(
                    (
                        t
                        for t in operation.pool_transfers
                        if t.indices_from_pool is None
                        and t.name == transfer.indices_from_pool
                    ),
                    None,
                )
                if source is None:
                    raise AssertionError(
                        "Storage sidecar derived pool source missing: "
                        f"{transfer.name} from {transfer.indices_from_pool}."
                    )
                transfer.host_indices = source.host_indices
                if transfer.keys is None:
                    transfer.keys = source.keys
            else:
                # 源是 KV 主池：直接用本次操作的 KV host 索引与页哈希作为索引/键。
                transfer.host_indices = operation.host_indices
                if transfer.keys is None:
                    transfer.keys = operation.hash_value

    def _sync_trailing_keys(
        self,
        pool_transfers: list[PoolTransfer],
        all_hashes: list[str],
        kv_hit_pages: int,
    ) -> None:
        """在 KV 命中被截断后，重新对齐「尾页型」附加池（sidecar）的 keys。

        当 storage 实际命中长度短于原先的目标前缀时，每个池搬运的 keys 必须更新为
        「实际命中范围」的最后 N 个哈希，而不是原目标范围的最后 N 个哈希。
        对 mamba（N=1）就是最后一个命中页的哈希；对 SWA（N>1）则是最后 N 个命中页的滑动窗口。
        """
        for transfer in pool_transfers:
            # 只处理「尾页」命中策略的池；其他策略不受 KV 截断影响。
            if transfer.hit_policy != PoolHitPolicy.TRAILING_PAGES:
                continue
            trailing_n = len(transfer.keys) if transfer.keys else 1
            transfer.keys = all_hashes[max(0, kv_hit_pages - trailing_n) : kv_hit_pages]

    def _resolve_pool_transfers_allocation(
        self,
        extra_pools: Optional[list[PoolTransfer]],
        alloc_host: bool,
        kv_device_indices: Optional[torch.Tensor] = None,
        kv_host_indices: Optional[torch.Tensor] = None,
    ) -> Optional[list[PoolTransfer]]:
        """为那些索引为 None 的 PoolTransfer 自动分配 host 或 device 索引。

        采用「全部成功或全部回滚」的原子语义：只要任一额外池分配失败，就把本次已成功
        分配的全部释放并返回 None，避免部分分配造成泄漏与不一致。
        """
        if not extra_pools:
            return None
        # 记录 (池, 释放函数, 已分配索引)，供失败时原子回滚。
        newly_allocated: list[tuple[PoolTransfer, Callable, torch.Tensor]] = []
        # 派生型搬运（索引来自其他池）延后处理，等源池分配好再借用。
        derived_transfers: list[PoolTransfer] = []

        def rollback_allocated() -> None:
            # 回滚：把已成功分配的都释放，并清空对应字段。
            for prev_pool, prev_free_fn, prev_indices in newly_allocated:
                prev_free_fn(prev_indices)
                if alloc_host:
                    prev_pool.host_indices = None
                else:
                    prev_pool.device_indices = None

        for pool in extra_pools:
            if pool.indices_from_pool is not None:
                # 派生型：先收集，稍后统一从源池借用索引。
                derived_transfers.append(pool)
                continue
            entry = self.mem_pool_host.entry_map.get(pool.name)
            if entry is None:
                continue
            if alloc_host:
                # 分配 host 索引：仅当尚未分配 host 且已有 device 索引（以其长度为准）时才做。
                if pool.host_indices is not None or pool.device_indices is None:
                    continue
                alloc_fn = entry.host_pool.alloc
                free_fn = entry.host_pool.free
                evict_fn = entry.host_evict_fn
                size = len(pool.device_indices)
            else:
                # 分配 device 索引：仅当尚未分配 device 且已有 host 索引时才做。
                if pool.device_indices is not None or pool.host_indices is None:
                    continue
                # 对于 device_pool 是「原始 KV 池（layout）」而非分配器的池（如 SWA），
                # 用 device_alloc_fn / device_free_fn 覆盖 entry.device_pool 的默认方法。
                alloc_fn = entry.device_alloc_fn or entry.device_pool.alloc
                free_fn = entry.device_free_fn or entry.device_pool.free
                evict_fn = entry.device_evict_fn
                size = len(pool.host_indices)
            indices = alloc_fn(size)
            if indices is None and evict_fn:
                # 分配失败且有淘汰函数：先淘汰出空间再重试一次。
                evict_fn(size)
                indices = alloc_fn(size)
            if indices is None:
                # 原子回滚：释放本次已成功分配的一切。
                rollback_allocated()
                return None
            if alloc_host:
                pool.host_indices = indices
            else:
                pool.device_indices = indices
            newly_allocated.append((pool, free_fn, indices))

        # 为延后的派生型池从其源池赋予索引。
        for pool in derived_transfers:
            if pool.indices_from_pool == PoolName.KV:
                # 源为 KV 主池：直接复用传入的 KV host/device 索引。
                pool.host_indices = kv_host_indices
                pool.device_indices = kv_device_indices
                continue

            # 源为另一个额外池：找到那个「拥有自己索引」的源 transfer 并借用其 host/device 索引。
            source = next(
                (
                    transfer
                    for transfer in extra_pools
                    if transfer.indices_from_pool is None
                    and transfer.name == pool.indices_from_pool
                ),
                None,
            )
            if source is None:
                rollback_allocated()
                return None
            pool.host_indices = source.host_indices
            pool.device_indices = source.device_indices
        return extra_pools
