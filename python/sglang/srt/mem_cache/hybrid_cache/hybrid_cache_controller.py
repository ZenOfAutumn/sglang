from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import replace
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, Callable, List, Optional

import torch

from sglang.srt.managers.cache_controller import (
    CacheOperation,
)
from sglang.srt.managers.cache_controller import (
    HiCacheController as BaseHiCacheController,
)
from sglang.srt.managers.cache_controller import (
    LayerDoneCounter,
    PrefetchAck,
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
    count_pool_hits,
)
from sglang.srt.mem_cache.l2_transfer import L2Transfer
from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

logger = logging.getLogger(__name__)


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
        token_ids: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        self.request_id = request_id
        self._lock = threading.Lock()
        self._terminated_flag = False
        self.storage_hit_count = 0
        self.start_time = time.monotonic()
        super().__init__(
            None,
            token_ids,
            last_hash,
            prefix_keys=prefix_keys,
            pool_transfers=pool_transfers,
        )
        # 若没有附加池搬运，则附加池部分一开始就视为已完成。
        self.pool_transfers_done = not bool(pool_transfers)

    def mark_terminate(self):
        # 标记该预取为「终止」，使其在下一次进度检查时尽快收尾。
        with self._lock:
            self._terminated_flag = True

    def is_terminated(self) -> bool:
        with self._lock:
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
        host_memory_mode: str = "cache",
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
            host_memory_mode=host_memory_mode,
        )
        # 覆盖 layer_num：混合模型需要搬运「所有层」（例如线性模型的 KV + Mamba），
        # 而不只是 full_kv_pool 报告的全注意力层。因此重建逐层完成计数器。
        if transfer_layer_num is not None and transfer_layer_num != self.layer_num:
            self.layer_num = transfer_layer_num
            self.layer_done_counter = LayerDoneCounter(self.layer_num)

        self.storage_host_pool = mem_pool_host.anchor_entry.host_pool
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

    def register_host_pool_entry(self, entry: PoolEntry) -> None:
        if not isinstance(self.mem_pool_host, HostPoolGroup):
            raise TypeError("Dynamic HiCache sidecars require HostPoolGroup.")
        self.mem_pool_host.add_entry(entry)
        if not entry.is_primary_index_anchor:
            self.extra_host_mem_release_queues.setdefault(entry.name, Queue())
        if self.enable_storage and self.storage_backend is not None:
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
        pool_transfers = self.mem_pool_host.resolve_host_transfers(
            extra_pools,
            primary_device_indices=device_indices,
            primary_host_indices=host_indices,
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

    def _move_op_indices(
        self, op: CacheOperation
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[list[PoolTransfer]]]:
        return self.move_hybrid_indices(op)

    def _move_write_operation(
        self, op: CacheOperation
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[list[PoolTransfer]]]:
        host_group = self.mem_pool_host
        if self.io_backend != "kernel" or host_group.layout != "page_first":
            return self.move_hybrid_indices(op)
        if not getattr(host_group, "supports_per_pool_backup_indices", False):
            if not getattr(host_group, "can_use_write_back_jit", False):
                return self.move_hybrid_indices(op)
            return op.host_indices, op.device_indices, op.pool_transfers

        def move_for_pool(host_pool, host_indices, device_indices):
            if getattr(host_pool, "can_use_write_back_jit", False):
                if host_indices.is_cuda:
                    host_indices = host_indices.cpu()
                return host_indices, device_indices
            return self.move_indices(host_indices, device_indices)

        host_indices, device_indices = move_for_pool(
            host_group.anchor_entry.host_pool,
            op.host_indices,
            op.device_indices,
        )
        pool_transfers = []
        for transfer in op.pool_transfers or []:
            entry = host_group.entry_map[transfer.name]
            transfer_host_indices, transfer_device_indices = move_for_pool(
                entry.host_pool,
                transfer.host_indices,
                transfer.device_indices,
            )
            pool_transfers.append(
                replace(
                    transfer,
                    host_indices=transfer_host_indices,
                    device_indices=transfer_device_indices,
                )
            )
        return host_indices, device_indices, pool_transfers or None

    def _l2_transfers(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ) -> list[L2Transfer]:
        anchor = self.mem_pool_host.anchor_entry
        transfers = []
        if host_indices.numel() > 0:
            transfers.append(
                L2Transfer(
                    host_pool=anchor.host_pool,
                    device_pool=anchor.device_pool,
                    host_indices=host_indices,
                    device_indices=device_indices,
                    layer_mapper=anchor.layer_mapper,
                )
            )
        for pool_transfer in pool_transfers or []:
            if (
                pool_transfer.host_indices is None
                or pool_transfer.device_indices is None
            ):
                raise ValueError(f"Unresolved L2 transfer for {pool_transfer.name}.")
            entry = self.mem_pool_host.entry_map[pool_transfer.name]
            transfers.append(
                L2Transfer(
                    host_pool=entry.host_pool,
                    device_pool=entry.device_pool,
                    host_indices=pool_transfer.host_indices,
                    device_indices=pool_transfer.device_indices,
                    layer_mapper=entry.layer_mapper,
                )
            )
        return transfers

    def _l2_load_transfers(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ) -> list[L2Transfer]:
        transfers = self._l2_transfers(host_indices, device_indices, pool_transfers)
        transfers_by_entry = {
            (id(t.host_pool), id(t.device_pool)): t for t in transfers
        }
        for entry in self.mem_pool_host.entry_map.values():
            target_transfer = transfers_by_entry.get(
                (id(entry.host_pool), id(entry.device_pool))
            )
            if target_transfer is None or target_transfer.layer_mapper is None:
                continue
            for depth, draft_device_pool in enumerate(entry.packed_draft_device_pools):
                draft_host_layer = target_transfer.layer_mapper(self.layer_num + depth)
                if draft_host_layer is None:
                    continue

                def draft_layer_mapper(
                    layer_id: int,
                    *,
                    expected_layer_id: int = depth,
                    host_layer_id: int = draft_host_layer,
                ) -> Optional[int]:
                    if layer_id == expected_layer_id:
                        return host_layer_id
                    return None

                transfers.append(
                    L2Transfer(
                        host_pool=target_transfer.host_pool,
                        device_pool=draft_device_pool,
                        host_indices=target_transfer.host_indices,
                        device_indices=target_transfer.device_indices,
                        layer_mapper=draft_layer_mapper,
                        is_draft=True,
                    )
                )
        return transfers

    def _num_tokens_by_pool(self, op: CacheOperation) -> dict[str, int]:
        """Per-pool token counts for a merged transfer op (anchor + extra
        pools), shared by D->H write and H->D load acks; sidecar transfers
        reusing another pool's indices are excluded."""
        counts = {self.mem_pool_host.anchor_entry.name.value: len(op.device_indices)}
        for transfer in op.pool_transfers or []:
            if transfer.indices_from_pool is not None or transfer.host_indices is None:
                continue
            name = transfer.name.value
            counts[name] = counts.get(name, 0) + len(transfer.host_indices)
        return counts

    def _transfer_num_bytes(self, op: CacheOperation) -> int:
        """Total bytes moved by a merged transfer op across all pools.

        Sidecar transfers riding another pool's indices are included here but
        excluded from the per-pool token counts.
        """
        kv_tokens = len(op.device_indices)
        num_bytes = kv_tokens * self.mem_pool_host.anchor_entry.host_pool.size_per_token
        # Slot counts of the pools sidecars can ride on.
        source_len = {self.mem_pool_host.anchor_entry.name: kv_tokens}
        for t in op.pool_transfers or []:
            if t.indices_from_pool is None and t.host_indices is not None:
                source_len[t.name] = len(t.host_indices)
        for t in op.pool_transfers or []:
            entry = self.mem_pool_host.entry_map.get(t.name)
            if entry is None:
                continue
            if t.indices_from_pool is not None:
                num_slots = source_len.get(t.indices_from_pool, 0)
            else:
                num_slots = len(t.host_indices) if t.host_indices is not None else 0
            num_bytes += num_slots * entry.host_pool.size_per_token
        return num_bytes

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

        pool_transfers = self._resolve_device_transfers(
            extra_pools,
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

    def prefetch(
        self,
        request_id: str,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> PrefetchOperation:
        # 创建一个预取操作并入预取队列，交由 storage 线程异步从 L3 拉取到 host。
        operation = PrefetchOperation(
            request_id,
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
        hash_value = self.get_hash_str(
            operation.token_ids, operation.last_hash, page_size=self.page_size
        )
        operation.all_hash_values = hash_value

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

    def _page_transfer(self, operation: PrefetchOperation) -> bool:
        # KV pools and KV-derived pools first — determines actual completed page count
        kv_completed_pages = super()._page_transfer(operation)

        # Read non-KV derived sidecar pool, e.g. SWA, Mamba.
        self._page_transfer_sidecar(operation, kv_completed_pages)

    def _page_transfer_sidecar(
        self, operation: PrefetchOperation, kv_completed_pages: int
    ) -> None:
        if operation.pool_transfers is None:
            return

        # Extra pools only after KV fully completes. If KV terminated early
        # (IO failure, timeout, TP mismatch), skip extra IO entirely to avoid
        # data misalignment.
        pool_hits: dict[str, int] = {}
        if not operation.is_terminated() and kv_completed_pages == len(
            operation.hash_value
        ):
            # KV-derived sidecar pools are handled in CacheController._page_transfer_kv_batch.
            # Only handle non-KV-derived sidecar pools here.
            transfers_nonkv = [
                transfer
                for transfer in operation.pool_transfers
                if transfer.indices_from_pool != PoolName.KV
            ]
            self._sync_trailing_keys(
                transfers_nonkv, operation.hash_value, kv_completed_pages
            )
            self._resolve_sidecar_nonkv_derived_pool_transfers(operation)
            results = self.storage_backend.batch_get_v2(transfers_nonkv)
            pool_hits = count_pool_hits(results)
        # Emit PrefetchAck to prefetch_sync_queue, even the operation has been canceled by the
        # scheduler thread.  The prefetch sync thread expects the same number of PrefetchAck objects
        # to perform all_reduce.
        self.prefetch_sync_queue.put(
            PrefetchAck(
                rid=operation.request_id,
                operation=operation,
                pool_hits=pool_hits,
            )
        )
        return

    def _page_backup(self, operation):
        # MLA KV is replicated across TP ranks and should still be written only
        # by TP0. Rank-sharded sidecars still need every TP rank.
        backup_transfers = [
            transfer
            for transfer in operation.pool_transfers or []
            if self.should_backup(transfer)
        ]

        if backup_transfers:
            self._resolve_sidecar_kv_derived_pool_transfers(operation)
            self._resolve_sidecar_nonkv_derived_pool_transfers(operation)
            results = self.storage_backend.batch_set_v2(backup_transfers)
            pool_hits = count_pool_hits(results)
            operation.pool_storage_result.update_extra_pool_hit_pages(pool_hits)

        if not self.backup_skip:
            super()._page_backup(operation)
        else:
            sidecar_ok = bool(backup_transfers)
            if sidecar_ok:
                for transfer in backup_transfers:
                    result = results.get(transfer.name)
                    if result is None:
                        result = results.get(transfer.name.value)
                    expected = len(transfer.keys or [])
                    if expected == 0 and transfer.host_indices is not None:
                        expected = int(transfer.host_indices.numel())
                    if (
                        not isinstance(result, (list, tuple))
                        or len(result) != expected
                        or not all(bool(ok) for ok in result)
                    ):
                        sidecar_ok = False
                        break
            operation.completed_tokens = (
                len(operation.hash_value) * self.page_size if sidecar_ok else 0
            )

    def should_backup(self, transfer: PoolTransfer) -> bool:
        if not self.backup_skip:
            return True

        # Kimi-K3 Mamba/KDA state is TP-sharded even when the primary MLA KV
        # pool is replicated.
        if transfer.name == PoolName.MAMBA:
            return True

        # Mooncake gives MHA draft and draft-SWA objects rank-specific keys.
        # MLA/DeepSeek-V4 draft pools remain TP0-only.
        if self.storage_backend_type == "mooncake" and transfer.name in (
            PoolName.DRAFT,
            PoolName.DRAFT_SWA,
        ):
            entry = self.mem_pool_host.entry_map.get(transfer.name)
            return entry is not None and isinstance(
                entry.host_pool, MHATokenToKVPoolHost
            )

        return False

    def backup_thread_func(self):
        """Back up rank-sharded sidecars on every TP rank.

        The base implementation skips the entire operation on non-zero MLA TP
        ranks. That optimization is valid for replicated MLA KV, but not for
        hybrid rank-sharded pools such as Kimi-K3 Mamba state.
        """
        while not self.storage_stop_event.is_set():
            try:
                operation = self.backup_queue.get(block=True, timeout=1)
                if operation is None:
                    continue
                self._page_backup(operation)
                self.ack_backup_queue.put(operation)
            except Empty:
                continue

    def _resolve_sidecar_kv_derived_pool_transfers(self, operation):
        for transfer in operation.pool_transfers:
            if transfer.indices_from_pool == PoolName.KV:
                transfer.host_indices = operation.host_indices
                if transfer.keys is None:
                    transfer.keys = operation.hash_value

    def _resolve_sidecar_nonkv_derived_pool_transfers(self, operation):
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
                pass

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
            if transfer.host_indices is None:
                continue
            entry = self.mem_pool_host.entry_map.get(transfer.name)
            pool_page_size = (
                entry.host_pool.page_size if entry is not None else self.page_size
            )
            needed = len(transfer.keys) * pool_page_size
            if transfer.host_indices.numel() > needed:
                # The hit undershot the pre-allocated window buffer. Backends
                # fetch keys zipped against the buffer head, so shrink the
                # transfer to match and release the tail now — otherwise the
                # length mismatch makes batch_get_v2 fetch nothing and the
                # whole window is silently lost downstream.
                self.append_host_mem_release(
                    extra_pools=[
                        PoolTransfer(
                            name=transfer.name,
                            host_indices=transfer.host_indices[needed:],
                        )
                    ]
                )
                transfer.host_indices = transfer.host_indices[:needed]

    def _resolve_device_transfers(
        self,
        extra_pools: Optional[list[PoolTransfer]],
        kv_device_indices: Optional[torch.Tensor] = None,
        kv_host_indices: Optional[torch.Tensor] = None,
    ) -> Optional[list[PoolTransfer]]:
        """Allocate unresolved side-pool device indices atomically."""
        if not extra_pools:
            return None
        newly_allocated: list[tuple[PoolTransfer, Callable, torch.Tensor]] = []
        # 派生型搬运（索引来自其他池）延后处理，等源池分配好再借用。
        derived_transfers: list[PoolTransfer] = []

        def rollback_allocated() -> None:
            # 回滚：把已成功分配的都释放，并清空对应字段。
            for prev_pool, prev_free_fn, prev_indices in newly_allocated:
                prev_free_fn(prev_indices)
                prev_pool.device_indices = None

        for pool in extra_pools:
            if pool.indices_from_pool is not None:
                # 派生型：先收集，稍后统一从源池借用索引。
                derived_transfers.append(pool)
                continue
            entry = self.mem_pool_host.entry_map.get(pool.name)
            if entry is None:
                continue
            if pool.device_indices is not None or pool.host_indices is None:
                continue
            # device_alloc_fn / device_free_fn override entry.device_pool's
            # methods for pools whose device_pool is a raw KV pool (layout)
            # rather than an allocator (e.g. SWA).
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

    def _reduce_prefetch_ack(self, ack: PrefetchAck) -> None:
        # Handle KV-derived pool.
        super()._reduce_prefetch_ack(ack)

        # Handle other sidecar pools, e.g. SWA, Mamba.
        if ack.pool_hits is not None:
            # "for ... in PoolName" ensures the same order across all ranks.
            # On prefetch failure, pool_hits may be empty dict.
            packed = torch.tensor(
                [ack.pool_hits.get(pool.value, 0) for pool in PoolName],
                dtype=torch.int,
            )
            self._all_reduce(
                packed,
                torch.distributed.ReduceOp.MIN,
                self.prefetch_completion_sync_groups,
            )
            for i, pool in enumerate(PoolName):
                ack.pool_hits[pool.value] = packed[i].item()
