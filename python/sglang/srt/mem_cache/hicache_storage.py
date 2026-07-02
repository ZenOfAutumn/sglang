from __future__ import annotations

import logging
import os
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional, Set

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool_host import HostKVCache

logger = logging.getLogger(__name__)

# 单次批量存储 IO 调用中可处理的最大页数（page）。
STORAGE_BATCH_SIZE = 128


@dataclass
class HiCacheStorageConfig:
    # HiCache 存储后端的配置：主要用于在分布式（TP/PP/CP）场景下，
    # 为每个 rank 生成互不冲突的存储键前缀，并描述模型布局特征。
    tp_rank: int  # 张量并行（TP）当前 rank
    tp_size: int  # 张量并行总数
    pp_rank: int  # 流水线并行（PP）当前 rank
    pp_size: int  # 流水线并行总数
    attn_cp_rank: int  # 注意力上下文并行（CP）当前 rank
    attn_cp_size: int  # 注意力上下文并行总数
    is_mla_model: bool  # 是否为 MLA 模型（MLA 的 KV 在 TP 间共享，故键不含 tp 信息）
    enable_storage_metrics: bool  # 是否开启存储侧指标统计
    is_page_first_layout: bool  # KV 内存布局是否为 page-first（页优先）
    model_name: Optional[str]  # 模型名，用于隔离不同模型的缓存
    tp_lcm_size: Optional[int] = None  # TP 尺寸的最小公倍数（跨配置共享时用）
    should_split_heads: bool = False  # 是否需要按注意力头切分
    extra_config: Optional[dict] = None  # 后端自定义的额外配置


@dataclass
class HiCacheStorageExtraInfo:
    # 传递给存储后端的附加信息（如前缀链式键、后端私有参数）。
    prefix_keys: Optional[List[str]] = None  # 当前页之前的前缀键列表（用于前缀链式定位）
    extra_info: Optional[dict] = None  # 其它后端自定义信息


@dataclass(frozen=True)
class PrefetchTimeoutConfig:
    """HiCache 所用「线性预取超时」策略的可调参数。

    超时时间随预取 token 数线性增长：timeout = min(max, base + per_ki_token * tokens/1024)。
    """

    base: float = 2.0  # 秒，与 token 数无关的固定开销
    per_ki_token: float = 0.1  # 秒，每 1024 个 token 增加的时间
    max: float = 30.0  # 秒，线性超时的上限


class PoolName(str, Enum):
    """约定俗成的缓存池名称，用作 PoolTransfer / PoolEntry 的标识符。

    不同模型/特性会用到不同的缓存池（KV、Mamba 状态、SWA 窗口、索引器等），
    每个池用一个唯一名称区分，存储键也据此加后缀以避免互相覆盖。
    """

    KV = "kv"
    MAMBA = "mamba"
    SWA = "swa"
    INDEXER = "indexer"
    # TODO(hzh0425): 当前 DeepSeek V4 的池命名较冗长；下个 PR 会统一规整为
    # 'COMPRESSED_KV / COMPRESSED_INDEXER / COMPRESSED_STATE'。
    DEEPSEEK_V4_C4 = "deepseek_v4_c4"
    DEEPSEEK_V4_C4_INDEXER = "deepseek_v4_c4_indexer"
    DEEPSEEK_V4_C128 = "deepseek_v4_c128"
    DEEPSEEK_V4_C4_STATE = "deepseek_v4_c4_state"
    DEEPSEEK_V4_C4_INDEXER_STATE = "deepseek_v4_c4_indexer_state"
    DEEPSEEK_V4_C128_STATE = "deepseek_v4_c128_state"

    # 投机解码的草稿（draft）KV 池
    DRAFT = "draft"

    def __str__(self) -> str:
        return self.value


class PoolHitPolicy(str, Enum):
    """batch_exists_v2 中各缓存池前缀匹配所用的「命中策略」。

    ALL_PAGES      : 前缀区间 [0, kv_hit) 内的每一页都必须存在（如 DSA 池）。
    TRAILING_PAGES : 只要求前缀「末尾」的最后 N 页存在（如 Mamba/SWA 状态池）。
    """

    ALL_PAGES = "all_pages"
    TRAILING_PAGES = "trailing_pages"


@dataclass
class PoolTransfer:
    """batch v2 接口统一使用的「单个缓存池传输描述符」。

    device <-> host 路径：使用 host_indices + device_indices（显存与主机内存间拷贝）
    host <-> storage 路径：使用 host_indices + keys（主机内存与后端存储间读写）
    nodes_to_load   ：本次传输涉及的、已被淘汰（需重新加载）的节点

    数值示例（Mamba 混合模型，page_size=4，备份 12 个 token = 3 页）：
      KV 主池（每页都搬，ALL_PAGES）：
        device_indices=[0,1,2], host_indices=[100,101,102], keys=["h0","h1","h2"]
      Mamba 状态池（只保留末页，TRAILING_PAGES）：
        device_indices=[7], host_indices=[50], keys=["h2"]
      其中 keys 为逐页链式哈希（h1 依赖 h0，h2 依赖 h1）。
      若 storage 实际只命中 2 页（kv_hit_pages=2），_sync_trailing_keys 会把
      Mamba 的 keys 从 ["h2"] 重对齐为实际命中范围的末页 ["h1"]（SWA 取 N 页则为 ["h0","h1"]）。
    """

    name: PoolName  # 本传输针对的缓存池名
    host_indices: Optional[torch.Tensor] = None  # 主机内存侧的页索引
    device_indices: Optional[torch.Tensor] = None  # 设备（显存）侧的页索引
    keys: Optional[List[str]] = None  # 后端存储侧的键列表（按页）
    hit_policy: PoolHitPolicy = PoolHitPolicy.ALL_PAGES  # 本池的命中策略
    nodes_to_load: Optional[List[Any]] = None  # 本次要加载的被淘汰节点
    indices_from_pool: Optional[PoolName] = None  # 索引复用自哪个源池（见 SidecarPoolSpec）


@dataclass(frozen=True)
class SidecarPoolSpec:
    """「附属池」规格：其传输索引直接复用某个真实源池的索引，无需单独计算。

    用于那些与某个主池页对齐、但数据独立存储的辅助池，避免重复维护索引。
    """

    pool_name: PoolName  # 附属池自身名称
    indices_from_pool: PoolName  # 从哪个源池复用传输索引
    hit_policy: PoolHitPolicy = PoolHitPolicy.ALL_PAGES  # 命中策略


@dataclass
class PoolTransferResult:
    """记录每个缓存池实际成功处理了多少页。"""

    kv_hit_pages: int  # KV 池命中/成功处理的页数（可用 KV 前缀长度）
    extra_pool_hit_pages: dict[str, int]  # 各附加池名 -> 成功页数

    @classmethod
    def empty(cls) -> PoolTransferResult:
        # 构造一个“零命中”的空结果。
        return cls(0, {})

    def update_kv_hit_pages(self, kv_hit_pages: int) -> None:
        """跨多个批次累计 kv_hit_pages（取最大值 = 最后一个成功批次的结果）。"""
        self.kv_hit_pages = max(self.kv_hit_pages, kv_hit_pages)

    def update_extra_pool_hit_pages(self, results: dict[str, List[bool]]) -> None:
        """记录每个附加池实际加载/写入成功的页数（布尔列表中 True 的个数）。"""
        self.extra_pool_hit_pages.update(
            {name: sum(rs) for name, rs in results.items()}
        )


class HiCacheStorage(ABC):
    """
    HiCacheStorage 提供了一个通用的键-值接口，用于存储和读取 KV 缓存。
    它抽象了底层存储机制，使得可以接入不同的存储后端实现（本地文件、分布式对象存储等）。
    """

    # todo：存储后端的页大小不一定要与主机内存池的页大小相同
    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        # 注册主机侧 KV 缓存池（v1 接口，单一池）。
        self.mem_pool_host = mem_pool_host

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name):
        # 注册主机侧缓存池（v2 接口，按名称区分多个池）。
        if not hasattr(self, "registered_pools"):
            self.registered_pools = {}
        self.registered_pools[host_pool_name] = host_pool

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        """检查哪些缓存页已存在于后端存储中，并遵循各缓存池各自的命中策略。

        采用「最长前缀」语义：返回从头开始连续命中的页数。

        附加池命中策略（``PoolTransfer.hit_policy``）
        ------------------------------------------------------
        ``pool_transfers`` 中的每个 ``PoolTransfer`` 描述一个辅助缓存池
        （例如 Mamba SSM 状态），它必须与 KV 页同时存在。最终的 ``final_pages``
        取所有池的最小值，因此任何一个辅助页缺失都会缩短可用前缀。

        - ``"all_pages"``（默认）：本池在 [0, kv_hit) 范围内的每一页都必须存在。
          适用于“前缀中每个 token 都需要”的池（例如 DeepSeek DSA 池）。

        - ``"trailing_pages"``：只需 KV 前缀的「最后」 ``len(transfer.keys)`` 页存在。
          适用于“数据只覆盖前缀末尾”的池（例如 Mamba/SWA 池）。

        返回
        -------
        PoolTransferResult
            ``kv_hit_pages`` = 可用 KV 前缀的页长度。
            ``extra_pool_hit_pages`` 将每个池名映射到实际找到的页数。
        """
        raise NotImplementedError()

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        """为每个 PoolTransfer 从后端存储读取数据到主机内存。

        返回一个字典：池名 -> 逐页成功与否的布尔列表。
        """
        raise NotImplementedError()

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        """为每个 PoolTransfer 将主机内存中的数据写入后端存储。

        返回一个字典：池名 -> 逐页成功与否的布尔列表。
        """
        raise NotImplementedError()

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        批量读取多个键对应的值。
        返回一个布尔列表，表示每个键是否读取成功。
        """
        pass

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        批量存储多个键值对。
        返回一个布尔列表，表示每个键是否写入成功。
        """
        pass

    @abstractmethod
    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        """
        读取给定键所关联的值。
        若键不存在则返回 None。
        """
        pass

    # TODO: 待废弃
    @abstractmethod
    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None] | int:
        """
        批量读取多个键对应的值。
        返回一个列表，每个元素为对应的张量或 None。
        """
        pass

    @abstractmethod
    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        存储给定键所关联的值。
        操作成功返回 True，否则返回 False。
        """
        pass

    # TODO: 待废弃
    @abstractmethod
    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        批量存储多个键值对。
        全部成功返回 True，否则返回 False。
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        检查该键是否存在于存储中。
        存在返回 True，否则返回 False。
        """
        pass

    # TODO: 使用更细粒度的返回类型（例如 List[bool]）
    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        """
        检查这些键是否存在于存储中。
        返回从开头起连续存在的键的个数（即最长连续前缀长度）。
        子类可覆写以提供更高效的实现。
        """
        # 逐个检查，遇到第一个不存在的键就返回当前下标（即连续命中的个数）。
        for i in range(len(keys)):
            if not self.exists(keys[i]):
                return i
        return len(keys)

    def clear(self) -> None:
        pass

    def get_stats(self):
        return None


class HiCacheFile(HiCacheStorage):
    """基于本地文件系统的 HiCache 存储后端：每个（键, 组件）对应一个 .bin 文件，
    存储的是原始字节。所有 LRU / 容量计账与磁盘淘汰逻辑都下放到 evictor 中，
    使本后端保持为一个轻量的“原始字节存储”。
    """

    def __init__(
        self, storage_config: HiCacheStorageConfig, file_path: str = "/tmp/hicache"
    ):
        self.file_path = envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.get() or file_path

        tp_rank, tp_size, pp_rank, pp_size, model_name, is_mla_model = (
            storage_config.tp_rank,
            storage_config.tp_size,
            storage_config.pp_rank,
            storage_config.pp_size,
            storage_config.model_name,
            storage_config.is_mla_model,
        )
        attn_cp_rank = storage_config.attn_cp_rank
        attn_cp_size = storage_config.attn_cp_size
        # 把模型名中的 "/" 换成 "-"，避免被当成路径分隔符。
        model_name = "-".join(model_name.split("/")) if model_name else ""
        enable_pp = pp_size > 1
        # 根据模型名 + 并行配置拼出存储键后缀，使不同模型/不同并行切分的缓存互不混淆。
        self.config_suffix = f"_{model_name}"
        # 非 MLA 模型：每个 TP rank 持有不同的 KV 分片，故键需区分 tp_rank/tp_size；
        # MLA 模型：KV 在 TP 间共享，不加 tp 信息，以便跨 rank 复用同一份缓存。
        if not is_mla_model:
            self.config_suffix += f"_{tp_rank}_{tp_size}"
        if enable_pp:
            self.config_suffix += f"_{pp_size}_{pp_rank}"
        # 在 NSA 上下文并行（CP）下，每个 CP rank 只持有每一页中互不重叠的一部分，
        # 所以要给每个 rank 单独的文件键，避免跨 rank 的写入竞争。
        if attn_cp_size > 1:
            self.config_suffix += f"_cp{attn_cp_rank}_{attn_cp_size}"

        # 只由 tp_rank==0 且 attn_cp_rank==0 的进程创建目录，避免多 rank 重复创建。
        if not os.path.exists(self.file_path) and tp_rank == 0 and attn_cp_rank == 0:
            os.makedirs(self.file_path)
            logger.info(f"Created HiCacheFile storage directory at {self.file_path}")

        # 所有 LRU / 容量计账与磁盘淘汰都交给 evictor，使本后端保持为轻量的原始字节存储。
        # 采用延迟导入：storage 包的 __init__ 会拉入后端工厂，后端工厂又会导入本模块，
        # 若在顶层导入会造成循环导入。
        from sglang.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor

        self._evictor = LRUFileEvictor(
            self.file_path,
            self.config_suffix,
            tp_rank=tp_rank,
            is_mla_model=is_mla_model,
            extra_config=storage_config.extra_config,
        )

    def _get_suffixed_key(self, key: str) -> str:
        # 给原始键拼上配置后缀（含模型名、TP/PP/CP rank 等），以隔离不同配置的缓存。
        return key + self.config_suffix

    def _get_component_key(self, key: str, component_name: Optional[str] = None) -> str:
        # 生成“组件级”存储键：KV 主组件不加组件后缀，其它池（如 mamba/swa）追加 ".组件名"。
        if component_name is None or component_name in ("__default__", PoolName.KV):
            return self._get_suffixed_key(key)
        return self._get_suffixed_key(f"{key}.{component_name}")

    def _get_component_path(
        self, key: str, component_name: Optional[str] = None
    ) -> str:
        # 由组件键拼出完整的 .bin 文件路径。
        return os.path.join(
            self.file_path, f"{self._get_component_key(key, component_name)}.bin"
        )

    def get(
        self,
        key: str,
        target_location: torch.Tensor,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        # 从磁盘读取单个键的原始字节到 target_location；命中返回该张量，未命中返回 None。
        suffixed = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{suffixed}.bin")
        try:
            # 期望读取的字节数 = 目标张量的元素个数 * 每元素字节数。
            expected = target_location.numel() * target_location.element_size()
            # 直接读入 target_location 的底层缓冲区（零拷贝），避免额外内存分配。
            with open(tensor_path, "rb", buffering=0) as f:
                buf = memoryview(target_location.view(torch.uint8).contiguous().numpy())
                # 读到的字节数不足说明文件损坏/裁断，报错。
                if f.readinto(buf) != expected:
                    raise IOError(f"Short read for {suffixed}")
            # 读取成功后刷新 LRU 访问时间（避免刚用过的条目被淘汰）。
            self._evictor.touch(suffixed, tensor_path)
            return target_location
        except FileNotFoundError:
            # 文件不存在即未命中，返回 None。
            logger.warning(f"Failed to fetch {key} from HiCacheFile storage.")
            return None

    def batch_get(
        self,
        keys: List[str],
        target_locations: List[torch.Tensor],
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        # 批量读取：逐个调用 get，返回每个键对应的张量或 None。
        return [
            self.get(key, target_location)
            for key, target_location in zip(
                keys, target_locations or [None] * len(keys)
            )
        ]

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        # 将单个键的张量以原始字节写入磁盘（经“临时文件 + 原子重命名”保证原子性）。
        suffixed = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{suffixed}.bin")

        # 快速路径：相同键已在磁盘上。只刷新访问时间、跳过重写。
        if os.path.exists(tensor_path):
            logger.debug(f"Key {key} already exists. Skipped.")
            self._evictor.touch(suffixed, tensor_path)
            return True

        tmp_path = None
        reserved = False
        try:
            value_bytes = value.numel() * value.element_size()
            # 请 evictor 准入并预留磁盘空间（必要时会先淘汰旧条目）。预留失败则放弃写入。
            if not self._evictor.reserve(suffixed, value_bytes, key=key):
                return False
            reserved = True

            # 先写到唯一临时文件，再原子重命名为正式文件：
            # 保证“要么看到完整文件、要么看不到”，避免并发读到写一半的文件。
            # 临时名含 pid + 线程 id + uuid，避免多进程/多线程同时写同一键时冲突。
            tmp_path = (
                f"{tensor_path}.tmp."
                f"{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
            )
            value.contiguous().view(dtype=torch.uint8).numpy().tofile(tmp_path)
            os.replace(tmp_path, tensor_path)
            # 正式文件落盘后提交预留（让 evictor 正式记账该条目的占用）。
            self._evictor.commit(suffixed)
            return True
        except Exception as e:
            logger.error(f"Failed to save tensor {key}: {e}")
            # 出错时回滚预留，并清理可能已写一半的临时文件。
            if reserved:
                self._evictor.abort(suffixed)
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            return False

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        # 批量写入：逐个调用 set，任一失败即返回 False（不保证原子性）。
        for key, value in zip(keys, values):
            if not self.set(key, value):
                return False
        return True

    def exists(self, key: str) -> bool:
        # 检查带后缀的键对应的 .bin 文件是否存在。
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        return os.path.exists(tensor_path)

    def _collect_existing_component_keys(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
    ) -> Set[str]:
        # 一次性收集本次关心的所有组件文件名（KV 主组件 + 各附加池组件），
        # 然后用一次 scandir 扫目录取交集，避免逐个 os.path.exists 的高频 syscall。
        target_files = {f"{self._get_component_key(key)}.bin" for key in keys}
        for transfer in pool_transfers or []:
            for key in keys:
                target_files.add(f"{self._get_component_key(key, transfer.name)}.bin")

        existing_files = set()
        with os.scandir(self.file_path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name in target_files:
                    existing_files.add(entry.name)
        return existing_files

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        existing_files = self._collect_existing_component_keys(keys, pool_transfers)

        def has_component(page_idx: int, name: str) -> bool:
            # 判断第 page_idx 页、名为 name 的组件文件是否存在。
            return (
                f"{self._get_component_key(keys[page_idx], name)}.bin" in existing_files
            )

        # 存储中存在的、最长连续 KV 前缀：从头扫描，遇到第一个缺失的 KV 页即停。
        kv_pages = next(
            (
                i
                for i in range(len(keys))
                if f"{self._get_component_key(keys[i])}.bin" not in existing_files
            ),
            len(keys),
        )

        hit_count: dict[str, int] = {PoolName.KV: kv_pages} if kv_pages else {}
        final_pages = kv_pages

        # 依次用各附加池的命中策略去“收紧”可用前缀：final_pages 取所有池的最小值。
        for transfer in pool_transfers or []:
            if final_pages == 0:
                break
            name = transfer.name
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                # ALL_PAGES：[0, kv_pages) 内逐页要求都存在，遇到第一个缺失页即为边界。
                boundary = next(
                    (i for i in range(kv_pages) if not has_component(i, name)), kv_pages
                )
            else:  # trailing_pages
                # TRAILING_PAGES：只要求末尾连续 trailing 页存在。从长到短试探，
                # 找到最大的 prefix_len，使其末尾 trailing 页都存在。
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                boundary = 0
                for prefix_len in range(kv_pages, 0, -1):
                    if all(
                        has_component(i, name)
                        for i in range(max(0, prefix_len - trailing), prefix_len)
                    ):
                        boundary = prefix_len
                        break
            if boundary:
                hit_count[name] = boundary
            final_pages = min(final_pages, boundary)

        return PoolTransferResult(final_pages, hit_count)

    def _log_key(self, pool_name: str, key: str) -> str:
        # 根据池名构造实际使用的存储键：KV 池用原键，其它池追加 ".池名"。
        return key if pool_name == PoolName.KV else f"{key}.{pool_name}"

    def _read_page(self, pool_name: str, key: str, host_pool, page_offset: int) -> bool:
        """从存储读取一页，写入 host_pool 的 page_offset 位置。成功返回 True。"""
        storage_key = self._log_key(pool_name, key)
        data_page = self.get(storage_key, host_pool.get_dummy_flat_data_page())
        if data_page is None:
            return False
        host_pool.set_from_flat_data_page(page_offset, data_page)
        return True

    def _write_page(
        self, pool_name: str, key: str, host_pool, page_offset: int
    ) -> bool:
        """将 host_pool 中 page_offset 位置的一页以原始字节写入存储。成功返回 True。"""
        storage_key = self._log_key(pool_name, key)
        data_page = host_pool.get_data_page(page_offset, flat=True)
        return self.set(storage_key, data_page)

    def _batch_io_v2(self, transfers: List[PoolTransfer], op_fn):
        # batch_get_v2 / batch_set_v2 的公共骨架：逐个池、逐页调用 op_fn（读页或写页）。
        results: dict[str, List[bool]] = {}
        for transfer in transfers:
            host_pool = self.registered_pools[transfer.name]
            keys = transfer.keys or []
            page_size = getattr(host_pool, "page_size", 1) or 1
            # 期望的主机索引个数 = 页数 * 每页 token 数。
            expected = len(keys) * page_size
            host_indices = transfer.host_indices

            # 索引长度与期望不符则本池全部记为失败，避免越界访问。
            if host_indices is None or host_indices.numel() != expected:
                logger.error(
                    "%s indices length mismatch for %s: expected %s, got %s",
                    op_fn.__name__,
                    transfer.name,
                    expected,
                    host_indices.numel() if host_indices is not None else 0,
                )
                results[transfer.name] = [False] * len(keys)
                continue

            results[transfer.name] = [
                op_fn(transfer.name, key, host_pool, host_indices[i * page_size].item())
                for i, key in enumerate(keys)
            ]
        return results

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        return self._batch_io_v2(transfers, self._read_page)

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        return self._batch_io_v2(transfers, self._write_page)

    def clear(self) -> bool:
        # 清空整个存储目录：删除所有文件并重置 evictor 的计账。
        try:
            for filename in os.listdir(self.file_path):
                file_path = os.path.join(self.file_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            self._evictor.clear()
            logger.info("Cleared all entries in HiCacheFile storage.")
            return True
        except Exception as e:
            logger.error(f"Failed to clear HiCacheFile storage: {e}")
            return False
