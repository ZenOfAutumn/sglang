import atexit
import concurrent.futures
import json
import logging
import os
import signal
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Any, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache
from sglang.srt.mem_cache.storage.hf3fs.hf3fs_client import Hf3fsClient
from sglang.srt.observability.metrics_collector import StorageMetrics

logger = logging.getLogger(__name__)


# ============================================================================
# 本文件实现 HiCache 的 HF3FS（幻方 3FS 高性能分布式文件系统）L3 存储后端。
#
# 职责：把 KV 缓存的「页」（page）持久化到 HF3FS 文件中，并支持按页读写、
# 存在性查询与淘汰。它把「数据」与「元数据」分离：
#   - 数据面：页字节直接读写到一个大文件的固定偏移处（page_index * bytes_per_page）；
#   - 元数据面：由 Hf3fsMetadataInterface 负责「key -> page_index」的映射、页分配/回收，
#     可以是本地实现（单机）或全局元数据服务（多机共享，MLA 模型必需）。
# 读写通过多个 Hf3fsClient + 线程池并发执行，以打满 3FS 的高带宽。
# ============================================================================


class Hf3fsMetadataInterface(ABC):
    """HF3FS 元数据操作接口。

    负责维护「缓存 key -> 文件页索引（page_index）」的映射，以及页的分配、确认、
    回收与查询。数据字节本身不经过这里（由 Hf3fsClient 直接读写文件），这里只管元数据。
    namespace 用于区分不同缓存池（KV / MAMBA / INDEXER 等），各池的键空间与页空间相互隔离。
    """

    @abstractmethod
    def initialize(
        self, rank: int, num_pages: int, namespace: PoolName = PoolName.KV
    ) -> None:
        """用指定的页数初始化元数据服务（为某个 rank / namespace 建立页空间）。"""
        pass

    @abstractmethod
    def reserve_and_allocate_page_indices(
        self,
        rank: int,
        keys: List[Tuple[str, str]],
        namespace: PoolName = PoolName.KV,
    ) -> List[Tuple[bool, int]]:
        """为指定的 keys 预留并分配文件页索引（写入前调用）。

        Args:
            rank: 进程的 rank。
            keys: 待分配页索引的 key 列表；每个元组是 (key, 其前缀块的 key)。
            namespace: 元数据所属的命名空间（缓存池类型）。
        Returns:
            List[Tuple[bool, int]]: 与输入等长的列表，每个元组为
                (该 key 是否已存在, 分配到的页索引)。已存在则无需重复写入；
                页索引为 -1 表示分配失败（空间不足）。
        """
        pass

    @abstractmethod
    def confirm_write(
        self,
        rank: int,
        written_keys_to_confirm: List[Tuple[str, int]],
        pages_to_release: List[int],
        namespace: PoolName = PoolName.KV,
    ) -> None:
        """确认某些键值对已成功写入存储（写入完成后调用，使映射正式生效）。

        Args:
            rank: 进程的 rank。
            written_keys_to_confirm: (key, 对应页索引) 列表——这些写入已成功，登记映射。
            pages_to_release: 需要释放的页索引列表（写入失败或多余的预留页，归还页池）。
            namespace: 元数据所属的命名空间（缓存池类型）。
        """
        pass

    @abstractmethod
    def get_page_indices(
        self, rank: int, keys: List[str], namespace: PoolName = PoolName.KV
    ) -> List[Optional[int]]:
        """查询指定 keys 对应的页索引（读取前调用，用于定位文件偏移）。

        Args:
            rank: 进程的 rank。
            keys: key 列表。
            namespace: 元数据所属的命名空间（缓存池类型）。
        Returns:
            List[Optional[int]]: 与 keys 等长的页索引列表；未命中的 key 对应 None。
        """
        pass

    @abstractmethod
    def delete_keys(
        self, rank: int, keys: List[str], namespace: PoolName = PoolName.KV
    ) -> None:
        """删除指定 keys 及其关联的页（回收页空间）。"""
        pass

    @abstractmethod
    def exists(
        self, rank: int, keys: List[str], namespace: PoolName = PoolName.KV
    ) -> List[bool]:
        """检查指定 keys 是否存在（用于命中查询）。"""
        pass

    @abstractmethod
    def clear(self, rank: int, namespace: PoolName = PoolName.KV) -> None:
        """清空该 rank 下所有键值对与页分配。"""
        pass


class AtomicCounter:
    """线程安全的循环计数器：next() 返回 0..n-1 并回绕。

    用于在多个 Hf3fsClient 之间轮询（round-robin）派发 IO 任务，均衡负载。
    """

    def __init__(self, n: int):
        assert n > 0
        self.n = n
        self._value = 0
        self._lock = threading.Lock()

    def next(self) -> int:
        # 取当前值并把内部计数 +1（模 n 回绕），全程持锁保证原子性。
        with self._lock:
            current = self._value
            self._value = (current + 1) % self.n
            return current


def synchronized():
    # 方法级同步装饰器：进入被装饰方法前先获取 self.lock，退出后自动释放。
    def _decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with self.lock:
                return func(self, *args, **kwargs)

        return wrapper

    return _decorator


def create_hf3fs_client(
    path: str,
    size: int,
    bytes_per_page: int,
    entries: int,
    client_timeout: int,
    use_mock: bool = False,
) -> Hf3fsClient:
    """工厂函数：创建合适的 HF3FS 客户端。

    Args:
        path: 存储文件路径。
        size: 存储文件总大小。
        bytes_per_page: 每页字节数。
        entries: 批量操作的条目数（单次 batch_read/batch_write 的最大页数）。
        use_mock: 是否使用 mock 客户端（测试用）而非真实的 usrbio 客户端。
    Returns:
        Hf3fsClient 实例（mock 或基于 usrbio 的真实客户端）。
    """
    if use_mock:
        from sglang.srt.mem_cache.storage.hf3fs.hf3fs_client import Hf3fsMockClient

        logger.info(f"[Rank Using Hf3fsMockClient for testing")
        return Hf3fsMockClient(path, size, bytes_per_page, entries)
    else:
        from sglang.srt.mem_cache.storage.hf3fs.hf3fs_usrbio_client import (
            Hf3fsUsrBioClient,
        )

        return Hf3fsUsrBioClient(path, size, bytes_per_page, entries, client_timeout)


@dataclass
class _PoolStorageCtx:
    """混合 KV 缓存下「每个额外池」的存储上下文。

    KV 主池的存储参数直接放在 HiCacheHF3FS 实例上；而 Mamba / SWA / INDEXER 等
    额外池各有不同的页大小与独立文件，用本结构分别保存它们的客户端与页配置。
    """

    pool_name: str  # 池名（如 PoolName.MAMBA / PoolName.INDEXER）
    bytes_per_page: int  # 该池每页字节数（与 KV 主池可能不同）
    num_pages: int  # 该池在其文件中可容纳的页总数 = file_size // bytes_per_page
    namespace: PoolName  # 元数据命名空间，与其他池隔离
    clients: List[Hf3fsClient]  # 该池专用的 Hf3fsClient 列表（指向该池的独立文件）
    gb_per_page: float  # 每页 GB 数，用于带宽统计


class HiCacheHF3FS(HiCacheStorage):
    """把 KV 缓存页存储到 HF3FS 文件中的 HiCache 后端。"""

    # 环境变量名：指向 HF3FS 的 JSON 配置文件路径（未设置时回退到单机默认配置）。
    default_env_var: str = "SGLANG_HICACHE_HF3FS_CONFIG_PATH"

    def __init__(
        self,
        rank: int,
        file_path: str,
        file_size: int,
        numjobs: int,
        bytes_per_page: int,
        entries: int,
        client_timeout: int,
        dtype: torch.dtype,
        metadata_client: Hf3fsMetadataInterface,
        is_mla_model: bool = False,
        is_page_first_layout: bool = False,
        use_mock_client: bool = False,
        enable_storage_metrics: bool = False,
    ):
        self.rank = rank  # 当前进程 rank（MLA 模型下会被强制置 0，见下）
        self.file_path = file_path  # 存储大文件路径（KV 主池的数据写到这里）
        self.file_size = file_size  # 文件总字节数
        self.numjobs = numjobs  # 并发 IO 客户端/线程数量
        self.bytes_per_page = bytes_per_page  # 每页字节数（KV 主池）
        self.gb_per_page = bytes_per_page / (1 << 30)  # 每页 GB 数，用于带宽统计
        self.entries = entries  # 单个 batch 读/写的最大页数
        self.client_timeout = client_timeout  # 客户端超时（秒）
        self.dtype = dtype  # KV 张量的数据类型
        self.metadata_client = metadata_client  # 元数据客户端（本地或全局）
        self.is_mla_model = is_mla_model  # 是否 MLA 模型（MLA 下各 rank 共享同一份 KV）
        self.is_page_first_layout = is_page_first_layout  # 宿主内存是否为 page-first 布局
        self.enable_storage_metrics = enable_storage_metrics  # 是否采集带宽/页数指标
        self.use_mock_client = use_mock_client  # 是否使用 mock 客户端（测试）
        self.numel = self.bytes_per_page // self.dtype.itemsize  # 每页元素个数
        self.num_pages = self.file_size // self.bytes_per_page  # 文件可容纳总页数
        self.skip_backup = False  # 是否跳过写入（备份）；MLA 非 0 号 rank 无需重复写
        # MLA 模型：所有 rank 的 KV 完全相同，只需 rank 0 写入，其余 rank 只读。
        # 因此把非 0 号 rank 重定向到 rank 0 的键空间，并跳过它们的写入。
        if self.is_mla_model and self.rank != 0:
            self.skip_backup = True
            self.rank = 0

        self.is_zero_copy = False  # 是否零拷贝（取决于宿主内存布局，在 register 时确定）

        logger.info(
            f"[Rank {self.rank}] HiCacheHF3FS Client Initializing: "
            f"file_path={self.file_path}, "
            f"file_size={self.file_size / (2 ** 30):.2f} GB, "
            f"num_pages={self.num_pages}, "
            f"is_mla_model={self.is_mla_model}"
        )

        # 轮询计数器：在 numjobs 个客户端之间均衡派发 IO。
        self.ac = AtomicCounter(self.numjobs)
        # 创建 numjobs 个 HF3FS 客户端，配合下方线程池实现并发读写。
        self.clients = [
            create_hf3fs_client(
                self.file_path,
                self.file_size,
                self.bytes_per_page,
                self.entries,
                self.client_timeout,
                use_mock_client,
            )
            for _ in range(numjobs)
        ]
        # 线程池：把 batch_read/batch_write 拆成多个子任务并发提交，打满 3FS 带宽。
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.numjobs, thread_name_prefix=f"HiCacheHF3FS-Rank{self.rank}"
        )

        # 初始化 KV 主池的元数据（建立 num_pages 大小的页空间）。
        self.metadata_client.initialize(self.rank, self.num_pages)
        self.lock = threading.RLock()  # 配合 synchronized() 装饰器使用的可重入锁
        self._pool_storage_ctx: dict = {}  # 额外池名 -> _PoolStorageCtx（混合缓存用）

        # 进程退出/收到信号时自动关闭客户端与线程池，避免文件句柄泄露。
        atexit.register(self.close)

        signal.signal(signal.SIGINT, lambda sig, frame: self.close())
        signal.signal(signal.SIGTERM, lambda sig, frame: self.close())
        signal.signal(signal.SIGQUIT, lambda sig, frame: self.close())

        # 以下四个列表用于累积指标，由 get_stats() 取走并清空：
        self.prefetch_pgs = []  # 每次预取（读）的页数
        self.backup_pgs = []  # 每次备份（写）的页数
        self.prefetch_bandwidth = []  # 每次预取的带宽（GB/s）
        self.backup_bandwidth = []  # 每次备份的带宽（GB/s）

    @staticmethod
    def from_env_config(
        bytes_per_page: int,
        dtype: torch.dtype,
        storage_config: HiCacheStorageConfig = None,
    ) -> "HiCacheHF3FS":
        """从环境配置创建一个 HiCacheHF3FS 实例。

        环境：
            - 使用 `HiCacheHF3FS.default_env_var` 指向的环境变量定位 JSON 配置文件；
            - 若未设置该环境变量，则回退到本地单机默认配置。

        Raises:
            ValueError: 当 MLA 模型缺少全局元数据服务器，或配置缺少必需字段时抛出。
        """
        from sglang.srt.mem_cache.storage.hf3fs.mini_3fs_metadata_server import (
            Hf3fsGlobalMetadataClient,
            Hf3fsLocalMetadataClient,
        )

        use_mock_client = False
        # 从 storage_config 提取 rank / 是否 MLA / 布局 / 是否用 mock 客户端；无配置则用默认值。
        if storage_config is not None:
            rank, is_mla_model, is_page_first_layout = (
                storage_config.tp_rank,
                storage_config.is_mla_model,
                storage_config.is_page_first_layout,
            )

            if storage_config.extra_config is not None:
                use_mock_client = storage_config.extra_config.get(
                    "use_mock_hf3fs_client", False
                )
        else:
            rank, is_mla_model, is_page_first_layout = (
                0,
                False,
                False,
            )

        # MLA 模型共享同一份 KV，必须依赖全局元数据服务器才能多机共享；否则报错。
        mla_unsupported_msg = f"MLA model is not supported without global metadata server, please refer to https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/storage/hf3fs/docs/deploy_sglang_3fs_multinode.md"

        config_path = os.getenv(HiCacheHF3FS.default_env_var)
        # 情形 A：未提供配置文件——回退到本地单机默认配置（本地元数据客户端）。
        if not config_path:
            if is_mla_model:
                raise ValueError(mla_unsupported_msg)

            return HiCacheHF3FS(
                rank=rank,
                file_path=f"/data/hicache.{rank}.bin",
                file_size=1 << 40,
                numjobs=16,
                bytes_per_page=bytes_per_page,
                entries=8,
                client_timeout=5,
                dtype=dtype,
                metadata_client=Hf3fsLocalMetadataClient(),
                is_page_first_layout=is_page_first_layout,
                use_mock_client=use_mock_client,
            )

        # 情形 B：提供了配置文件——读取并解析 JSON。
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {config_path}: {str(e)}")

        # 校验必需字段（metadata_server_url 现在是可选的）。
        required_keys = {
            "file_path_prefix",
            "file_size",
            "numjobs",
            "entries",
        }
        missing_keys = required_keys - set(config.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys in config: {missing_keys}")

        # 根据配置选择元数据客户端。
        if config.get("metadata_server_url"):
            # 配置了服务地址：使用全局元数据客户端连接元数据服务器（支持多机共享）。
            metadata_server_url = config["metadata_server_url"]
            metadata_client = Hf3fsGlobalMetadataClient(metadata_server_url)

            logger.info(
                f"Using global metadata client with server url: {metadata_server_url}"
            )
        else:
            # 只有使用全局元数据客户端时才能启用 MLA 优化，否则报错。
            if is_mla_model:
                raise ValueError(mla_unsupported_msg)

            # 单机部署：使用本地元数据客户端。
            metadata_client = Hf3fsLocalMetadataClient()

        # MLA 模型下所有 rank 共用同一个文件（rank_for_path=0），否则每个 rank 一个文件。
        rank_for_path = 0 if is_mla_model else rank
        return HiCacheHF3FS(
            rank=rank,
            # MLA 模型让所有 rank 使用同一个文件路径
            file_path=f"{config['file_path_prefix']}.{rank_for_path}.bin",
            file_size=int(config["file_size"]),
            numjobs=int(config["numjobs"]),
            bytes_per_page=bytes_per_page,
            entries=int(config["entries"]),
            client_timeout=config.get("client_timeout", 5),
            dtype=dtype,
            metadata_client=metadata_client,
            is_mla_model=is_mla_model,
            is_page_first_layout=is_page_first_layout,
            use_mock_client=use_mock_client,
            enable_storage_metrics=storage_config.enable_storage_metrics,
        )

    def _batch_get(
        self,
        keys: List[str],
        values: List[torch.Tensor],
    ) -> List[bool]:
        """批量读取：把 keys 对应的页从 HF3FS 文件读入 values 张量。返回每个 key 是否成功。"""
        # 1) 先查元数据拿到每个 key 的页索引（未命中为 None）。
        page_indices = self.metadata_client.get_page_indices(self.rank, keys)
        if len(page_indices) != len(keys):
            logger.error(
                f"[Rank {self.rank}] HiCacheHF3FS get: page_indices length {len(page_indices)} mismatch keys length {len(keys)}."
            )
            return [False] * len(keys)
        # 2) 只处理命中的 key：记下它们在原列表中的下标，并换算成文件字节偏移。
        batch_indices, file_offsets = [], []
        for i, page_index in enumerate(page_indices):
            if page_index is not None:
                batch_indices.append(i)
                file_offsets.append(page_index * self.bytes_per_page)

        # 3) 目标张量必须连续，才能直接作为 IO 缓冲区。
        for target_location in values:
            assert target_location.is_contiguous()
        file_results = values

        start_time = time.perf_counter()

        # 4) 按 entries 分批，轮询客户端并发提交 batch_read；然后按顺序汇总结果。
        futures = [
            self.executor.submit(
                self.clients[self.ac.next()].batch_read,
                file_offsets[i : i + self.entries],
                file_results[i : i + self.entries],
            )
            for i in range(0, len(batch_indices), self.entries)
        ]
        read_results = [result for future in futures for result in future.result()]

        end_time = time.perf_counter()
        ionum = len(batch_indices)

        # 5) 可选：记录这次预取的页数与带宽。
        if self.enable_storage_metrics:
            self.prefetch_pgs.append(ionum)
            self.prefetch_bandwidth.append(
                ionum / (end_time - start_time) * self.gb_per_page
            )

        # 6) 只有实际读到的字节数等于一整页才算成功，写回对应下标。
        results = [False] * len(keys)
        for batch_index, read_result in zip(batch_indices, read_results):
            if read_result == self.bytes_per_page:
                results[batch_index] = True
            else:
                logger.error(
                    f"[Rank {self.rank}] HiCacheHF3FS get {keys[batch_index]} failed"
                )

        return results

    def _batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
    ) -> List[bool]:
        """批量写入：把 values 写入 keys 对应的页。返回每个 key 是否已写入（含已存在）。"""
        # MLA 后端：只需一个 rank 备份 KV，其余 rank 直接返回成功。
        if self.skip_backup:
            return True

        # 1) 为每个 key 预留/分配页索引（前缀块 key 暂未使用，留空）。
        # Todo: Add prefix block's hash key
        key_with_prefix = [(key, "") for key in keys]
        indices = self.metadata_client.reserve_and_allocate_page_indices(
            self.rank, key_with_prefix
        )
        if len(indices) != len(keys):
            logger.error(
                f"[Rank {self.rank}] HiCacheHF3FS batch_get: mismatched lengths {len(indices)} != {len(keys)}"
            )
            # 长度不匹配属于异常情况：把已分配的页全部释放，避免泄露。
            if indices:
                self.metadata_client.confirm_write(
                    self.rank, [], [index[1] for index in indices]
                )
            return [False] * len(keys)
        batch_indices, file_offsets, file_values = [], [], []
        pages_to_release = []

        # 2) 跳过已存在（is_written）与分配失败（page_index == -1）的项，其余准备写入。
        for i, (value, (is_written, page_index)) in enumerate(zip(values, indices)):
            if is_written or page_index == -1:
                continue

            batch_indices.append(i)
            file_offsets.append(page_index * self.bytes_per_page)
            assert value.is_contiguous()
            file_values.append(value)

        start_time = time.perf_counter()

        # 3) 按 entries 分批并发写入。
        futures = [
            self.executor.submit(
                self.clients[self.ac.next()].batch_write,
                file_offsets[i : i + self.entries],
                file_values[i : i + self.entries],
            )
            for i in range(0, len(batch_indices), self.entries)
        ]
        write_results = [
            result == self.bytes_per_page
            for future in futures
            for result in future.result()
        ]

        end_time = time.perf_counter()
        ionum = len(batch_indices)

        # 4) 可选：记录这次备份的页数与带宽。
        if self.enable_storage_metrics:
            self.backup_pgs.append(ionum)
            self.backup_bandwidth.append(
                ionum / (end_time - start_time) * self.gb_per_page
            )

        # 5) 根据写入结果区分：成功的登记确认，失败的释放页。
        # results 初始为“是否已存在”，再用实际写入结果覆盖实际写过的项。
        written_keys_to_confirm = []
        results = [index[0] for index in indices]
        for batch_index, write_result in zip(batch_indices, write_results):
            key = keys[batch_index]
            page_index = indices[batch_index][1]
            if write_result:
                written_keys_to_confirm.append((key, page_index))
            else:
                logger.error(f"[Rank {self.rank}] HiCacheHF3FS set {key} failed")
                pages_to_release.append(page_index)
            results[batch_index] = write_result

        # 6) 一次性向元数据确认写入结果（登记映射 + 释放失败页）。
        if len(written_keys_to_confirm) > 0 or len(pages_to_release) > 0:
            self.metadata_client.confirm_write(
                self.rank, written_keys_to_confirm, pages_to_release
            )

        return results

    def delete(self, key: str) -> None:
        """删除单个 key（回收其页）。"""
        self.metadata_client.delete_keys(self.rank, [key])

    def exists(self, key: str) -> bool:
        """查询单个 key 是否存在。"""
        result = self.metadata_client.exists(self.rank, [key])
        return result[0] if result else False

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        """返回 keys 从头开始连续命中的前缀长度（页数）。

        由于 KV 缓存是前缀共享的，只有从头连续命中的部分才能被复用，
        因此遇到第一个未命中就停下。
        MHA 非 MLA 的零拷贝模式下，每页拆成 k/v 两个 key，故用 factor=2 换算回页数。
        """
        factor = 1
        if self.is_zero_copy and not self.is_mla_model:
            keys = self._get_mha_zero_copy_keys(keys)
            factor = 2

        results = self.metadata_client.exists(self.rank, keys)

        # 从头扫描，累计连续命中的个数（遇到首个未命中即停）。
        i = 0
        while i < len(keys) and results[i]:
            i += 1

        return i // factor

    def clear(self) -> None:
        """清空 KV 主池以及所有额外池的元数据（不报错，只记日志）。"""
        try:
            self.metadata_client.clear(self.rank)
            # 逐个清空混合缓存的额外池（每个池一个 namespace）。
            for ctx in getattr(self, "_pool_storage_ctx", {}).values():
                self.metadata_client.clear(self.rank, namespace=ctx.namespace)
            logger.info(f"Cleared HiCacheHF3FS for rank {self.rank}")
        except Exception as e:
            logger.error(f"Failed to clear HiCacheHF3FS: {e}")

    def close(self) -> None:
        """关闭所有客户端（含额外池）并优雅关闭线程池，释放文件句柄。"""
        try:
            for c in self.clients:
                c.close()
            for ctx in getattr(self, "_pool_storage_ctx", {}).values():
                for c in ctx.clients:
                    c.close()
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"close HiCacheHF3FS: {e}")
        logger.info("close HiCacheHF3FS")

    def get_stats(self):
        """取走并清空累积的带宽/页数指标（读后即清，便于周期性上报）。"""
        storage_metrics = StorageMetrics()
        storage_metrics.prefetch_pgs.extend(self.prefetch_pgs)
        storage_metrics.backup_pgs.extend(self.backup_pgs)
        storage_metrics.prefetch_bandwidth.extend(self.prefetch_bandwidth)
        storage_metrics.backup_bandwidth.extend(self.backup_bandwidth)
        self.prefetch_pgs.clear()
        self.backup_pgs.clear()
        self.prefetch_bandwidth.clear()
        self.backup_bandwidth.clear()
        return storage_metrics

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        """注册 KV 主池的宿主内存。根据其内存布局决定是否启用零拷贝。

        当宿主内存为 page_first / page_first_direct 布局时，单页字节在内存中连续，
        可直接作为 IO 缓冲区，无需额外拷贝（零拷贝）。
        """
        super().register_mem_pool_host(mem_pool_host)
        self.is_zero_copy = self.mem_pool_host.layout in [
            "page_first",
            "page_first_direct",
        ]

        logger.info(f"{self.is_zero_copy=}, layout={self.mem_pool_host.layout}")

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name):
        """注册一个额外池（Mamba / SWA / INDEXER 等）的宿主内存，并为其建立独立的
        存储文件、客户端与元数据命名空间。KV 主池已由 register_mem_pool_host 处理，此处跳过。
        """
        if host_pool_name == PoolName.KV:
            return
        super().register_mem_host_pool_v2(host_pool, host_pool_name)

        # 计算该池的页大小与可容纳页数，并为其分配一个独立文件（文件名后缀为池名）。
        pool_page_size = getattr(host_pool, "page_size", 1) or 1
        pool_bytes_per_page = host_pool.get_ksize_per_token() * pool_page_size
        pool_num_pages = self.file_size // pool_bytes_per_page
        pool_file_path = f"{self.file_path}.{host_pool_name}"
        namespace = host_pool_name  # e.g. PoolName.MAMBA, PoolName.INDEXER

        # 为该池创建专用客户端（数量与主池一致）。
        pool_clients = [
            create_hf3fs_client(
                pool_file_path,
                self.file_size,
                pool_bytes_per_page,
                self.entries,
                self.client_timeout,
                self.use_mock_client,
            )
            for _ in range(self.numjobs)
        ]

        # 为该池在对应 namespace 下初始化元数据页空间。
        self.metadata_client.initialize(self.rank, pool_num_pages, namespace=namespace)

        # 登记该池的存储上下文，供后续 _pool_batch_get/_pool_batch_set 使用。
        self._pool_storage_ctx[host_pool_name] = _PoolStorageCtx(
            pool_name=host_pool_name,
            bytes_per_page=pool_bytes_per_page,
            num_pages=pool_num_pages,
            namespace=namespace,
            clients=pool_clients,
            gb_per_page=pool_bytes_per_page / (1 << 30),
        )
        logger.info(
            f"[Rank {self.rank}] Registered hybrid pool '{host_pool_name}': "
            f"bytes_per_page={pool_bytes_per_page}, num_pages={pool_num_pages}, "
            f"namespace={namespace}, file={pool_file_path}"
        )

    def _get_mha_zero_copy_keys(self, keys: List[str]) -> List[str]:
        """MHA 零拷贝：把每个页 key 拆成 k/v 两个子 key（k 和 v 分开存储）。"""
        _keys = []
        for k in keys:
            _keys.append(f"{k}-k")
            _keys.append(f"{k}-v")
        return _keys

    def _get_mha_zero_copy_values(
        self, values: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """MHA 零拷贝：把每页的 [k, v] 张量拆成两个独立张量，与拆分后的 key 一一对应。"""
        _values = []
        for value in values:
            _values.append(value[0])
            _values.append(value[1])
        return _values

    def _batch_get_preprocess(self, keys, host_indices):
        """读取前预处理：根据 host_indices 构造接收数据的目标缓冲区 values。

        - 零拷贝：直接拿宿主内存中的页作为目标（读入后无需再拷）；
        - 非零拷贝：先用临时的 flat 缓冲区接收，后续再写回宿主内存。
        """
        page_num = len(host_indices) // self.mem_pool_host.page_size
        # host_indices to kv_buffer
        flat = not self.is_zero_copy
        values = (
            [
                self.mem_pool_host.get_data_page(
                    host_indices[i * self.mem_pool_host.page_size], flat=flat
                )
                for i in range(page_num)
            ]
            if self.is_zero_copy
            else [
                self.mem_pool_host.get_dummy_flat_data_page() for _ in range(page_num)
            ]
        )

        if self.is_zero_copy and not self.is_mla_model:
            keys = self._get_mha_zero_copy_keys(keys)
            values = self._get_mha_zero_copy_values(values)

        return keys, values

    def _batch_get_postprocess(self, host_indices, values, results):
        """读取后后处理：把读到的数据落地到宿主内存，并把结果换算回页粒度。

        - 零拷贝：数据已直接读到宿主内存，无需拷贝；MHA 还需把 k/v 两个结果合并为一页；
        - 非零拷贝：逐页把临时缓冲区写回宿主内存；遇到首个失败页即停（前缀连续性）。
        """
        page_num = len(host_indices) // self.mem_pool_host.page_size

        if self.is_zero_copy:
            if not self.is_mla_model:
                # MHA 下每页拆为 k/v 两项，两者都成功才算该页成功。
                results = [
                    (results[2 * i] and results[2 * i + 1]) for i in range(page_num)
                ]
                results = results[:page_num]
            return results

        # 非零拷贝：把读到的 flat 数据页写回对应的宿主内存位置。
        for i in range(page_num):
            if not results[i]:
                break
            self.mem_pool_host.set_from_flat_data_page(
                host_indices[i * self.mem_pool_host.page_size], values[i]
            )

        return results

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        """混合缓存下的命中查询：先算 KV 主池命中页数，再逐个额外池取交集。

        最终可复用的页数受限于所有池中最小的命中长度（因为每页需在所有池都命中才能用）。
        返回 PoolTransferResult：（最终可用页数, 各池命中页数字典）。
        """
        # 1) KV 主池命中的前缀页数（上限）。
        kv_pages = self.batch_exists(keys, extra_info)

        hit_count: dict = {PoolName.KV: kv_pages} if kv_pages else {}
        final_pages = kv_pages

        # 2) 逐个额外池求交集；无数据可用（final_pages==0）或池未注册时提前结束。
        for transfer in pool_transfers or []:
            if final_pages == 0:
                break

            pool_name = transfer.name
            ctx = self._pool_storage_ctx.get(pool_name)
            if ctx is None:
                final_pages = 0
                break

            # 池内的 key 带上池名后缀，在该池 namespace 下查存在性。
            component_keys = [f"{key}_{pool_name}" for key in keys[:kv_pages]]
            exists_results = self.metadata_client.exists(
                self.rank, component_keys, namespace=ctx.namespace
            )

            # 3) 根据池的命中策略计算边界 boundary：
            boundary = 0
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                # ALL_PAGES：从头连续命中，遇到首个未命中就截断（全命中则取 kv_pages）。
                try:
                    boundary = exists_results.index(False)
                except ValueError:
                    boundary = kv_pages
            elif transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                # TRAILING_PAGES：只要最后 trailing 页都命中即可（从长到短找最大可用前缀）。
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                for prefix_len in range(kv_pages, 0, -1):
                    if all(
                        exists_results[i]
                        for i in range(max(0, prefix_len - trailing), prefix_len)
                    ):
                        boundary = prefix_len
                        break

            if boundary:
                hit_count[pool_name] = boundary
            # 4) 最终可用页数取各池边界的最小值。
            final_pages = min(final_pages, boundary)

        return PoolTransferResult(final_pages, hit_count)

    def _pool_batch_get(self, transfer: PoolTransfer) -> List[bool]:
        """额外池的批量读取：与 _batch_get 类似，但使用该池专用的客户端/namespace/页大小。"""
        pool_name = transfer.name
        ctx = self._pool_storage_ctx[pool_name]
        host_pool = self.registered_pools[pool_name]
        keys = transfer.keys
        host_indices = transfer.host_indices
        page_size = getattr(host_pool, "page_size", 1) or 1
        page_num = len(keys)

        # 1) 池内 key 带上池名后缀，在该池 namespace 下查页索引。
        component_keys = [f"{key}_{pool_name}" for key in keys]
        page_indices = self.metadata_client.get_page_indices(
            self.rank, component_keys, namespace=ctx.namespace
        )

        # 2) 对命中页准备 flat 接收缓冲区并计算文件偏移。
        batch_indices, file_offsets, values = [], [], []
        for i, page_index in enumerate(page_indices):
            if page_index is not None:
                batch_indices.append(i)
                file_offsets.append(page_index * ctx.bytes_per_page)
                values.append(host_pool.get_dummy_flat_data_page())

        if not batch_indices:
            return [False] * page_num

        # 3) 按 entries 分批并发读取（使用该池专用客户端）。
        start_time = time.perf_counter()
        futures = [
            self.executor.submit(
                ctx.clients[self.ac.next()].batch_read,
                file_offsets[j : j + self.entries],
                values[j : j + self.entries],
            )
            for j in range(0, len(batch_indices), self.entries)
        ]
        read_results = [r for f in futures for r in f.result()]
        end_time = time.perf_counter()
        ionum = len(batch_indices)

        if self.enable_storage_metrics:
            self.prefetch_pgs.append(ionum)
            self.prefetch_bandwidth.append(
                ionum / (end_time - start_time) * ctx.gb_per_page
            )

        # 4) 读成功的页写回该池对应的宿主内存位置。
        results = [False] * page_num
        for idx, (batch_idx, read_result) in enumerate(
            zip(batch_indices, read_results)
        ):
            if read_result == ctx.bytes_per_page:
                host_idx = host_indices[batch_idx * page_size].item()
                host_pool.set_from_flat_data_page(host_idx, values[idx])
                results[batch_idx] = True
            else:
                logger.error(
                    f"[Rank {self.rank}][Pool {pool_name.upper()}] HiCacheHF3FS get {keys[batch_idx]} failed"
                )

        return results

    def _pool_batch_set(self, transfer: PoolTransfer) -> List[bool]:
        """额外池的批量写入：与 _batch_set 类似，但使用该池专用的客户端/namespace/页大小。"""
        pool_name = transfer.name
        ctx = self._pool_storage_ctx[pool_name]
        host_pool = self.registered_pools[pool_name]
        keys = transfer.keys
        host_indices = transfer.host_indices
        page_size = getattr(host_pool, "page_size", 1) or 1
        page_num = len(keys)

        # 1) 池内 key 带上池名后缀，在该池 namespace 下预留/分配页。
        component_keys = [f"{key}_{pool_name}" for key in keys]
        key_with_prefix = [(k, "") for k in component_keys]
        indices = self.metadata_client.reserve_and_allocate_page_indices(
            self.rank, key_with_prefix, namespace=ctx.namespace
        )

        if len(indices) != page_num:
            logger.error(
                f"[Rank {self.rank}] Pool {pool_name}: mismatched indices length"
            )
            if indices:
                self.metadata_client.confirm_write(
                    self.rank, [], [idx[1] for idx in indices], namespace=ctx.namespace
                )
            return [False] * page_num

        # 2) 跳过已存在/分配失败的项，其余从宿主内存取出 flat 数据页准备写入。
        batch_indices, file_offsets, file_values = [], [], []
        for i, (is_written, page_index) in enumerate(indices):
            if is_written or page_index == -1:
                continue
            batch_indices.append(i)
            file_offsets.append(page_index * ctx.bytes_per_page)
            host_idx = host_indices[i * page_size].item()
            data = host_pool.get_data_page(host_idx, flat=True)
            assert data.is_contiguous()
            file_values.append(data)

        # 3) 按 entries 分批并发写入。
        start_time = time.perf_counter()
        futures = [
            self.executor.submit(
                ctx.clients[self.ac.next()].batch_write,
                file_offsets[j : j + self.entries],
                file_values[j : j + self.entries],
            )
            for j in range(0, len(batch_indices), self.entries)
        ]
        write_results = [r == ctx.bytes_per_page for f in futures for r in f.result()]
        end_time = time.perf_counter()
        ionum = len(batch_indices)

        if self.enable_storage_metrics:
            self.backup_pgs.append(ionum)
            self.backup_bandwidth.append(
                ionum / (end_time - start_time) * ctx.gb_per_page
            )

        # 4) 成功的登记确认，失败的释放页；results 初始为“是否已存在”。
        written_keys_to_confirm = []
        pages_to_release = []
        results = [idx[0] for idx in indices]
        for batch_idx, write_ok in zip(batch_indices, write_results):
            key = component_keys[batch_idx]
            page_index = indices[batch_idx][1]
            if write_ok:
                written_keys_to_confirm.append((key, page_index))
            else:
                logger.error(
                    f"[Rank {self.rank}][Pool {pool_name.upper()}] HiCacheHF3FS set {keys[batch_idx]} failed"
                )
                pages_to_release.append(page_index)
            results[batch_idx] = write_ok

        if written_keys_to_confirm or pages_to_release:
            self.metadata_client.confirm_write(
                self.rank,
                written_keys_to_confirm,
                pages_to_release,
                namespace=ctx.namespace,
            )

        return results

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict:
        """v2 接口（混合缓存）：逐个额外池批量读取，返回 {池名: 每页是否成功}。"""
        results = {}
        for transfer in transfers:
            results[transfer.name] = self._pool_batch_get(transfer)
        return results

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict:
        """v2 接口（混合缓存）：逐个额外池批量写入，返回 {池名: 每页是否成功}。"""
        results = {}
        for transfer in transfers:
            results[transfer.name] = self._pool_batch_set(transfer)
        return results

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """v1 接口（KV 主池）：预处理 -> 批量读 -> 后处理（落地到宿主内存）。"""
        keys, values = self._batch_get_preprocess(keys, host_indices)
        results = self._batch_get(keys, values)
        return self._batch_get_postprocess(host_indices, values, results)

    def _batch_set_preprocess(self, keys, host_indices):
        """写入前预处理：从 host_indices 取出待写入的数据页 values（零拷贝时直接引用宿主内存）。"""
        page_num = len(host_indices) // self.mem_pool_host.page_size
        # host_indices to kv_buffer
        flat = not self.is_zero_copy
        values = [
            self.mem_pool_host.get_data_page(
                host_indices[i * self.mem_pool_host.page_size], flat=flat
            )
            for i in range(page_num)
        ]

        if self.is_zero_copy and not self.is_mla_model:
            keys = self._get_mha_zero_copy_keys(keys)
            values = self._get_mha_zero_copy_values(values)

        return keys, values

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """v1 接口（KV 主池）：预处理 -> 批量写。返回每个项是否已写入（含已存在）。"""
        len_keys = len(keys)
        keys, values = self._batch_set_preprocess(keys, host_indices)
        results = self._batch_set(keys, values)
        return results

    # 以下为已废弃的旧接口（单条/无 host_indices 的读写），已被 v1/v2 接口取代，仅保留签名。
    # Deprecated
    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        pass

    # Deprecated
    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None] | int:
        pass

    # Deprecated
    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        pass

    # Deprecated
    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        pass
