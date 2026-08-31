"""前缀缓存（Prefix Cache）的抽象基类与公共数据结构定义。

本模块是 SGLang 各类前缀缓存实现的“契约层”，主要包含三部分内容：

1. ``PrefixCacheTrait``：用 ``Protocol`` 描述前缀缓存必须暴露的几个公共属性
   （请求-token 映射池、KV 池分配器、page 大小、是否禁用），供结构化类型检查使用。
2. 一组 ``@dataclass`` 形式的参数/结果对象（``MatchPrefixParams`` / ``InsertParams``
   / ``EvictParams`` / ``IncLockRefResult`` 等）。这些对象把“匹配前缀、插入、淘汰、
   加减引用计数”等操作的入参与返回值统一封装起来，从而让不同缓存实现
   （RadixCache、ChunkCache、HiCache、SWA、Mamba 等）能够共享同一套方法签名。
3. ``BasePrefixCache``：所有前缀缓存的抽象基类，声明了必须实现的抽象方法
   （reset / match_prefix / cache_finished_req / evict / inc_lock_ref ...），
   并为一批可选能力（SWA、Mamba、HiCache 分级写回、流式会话等）提供了
   “默认无操作 / 默认不支持”的实现，使派生类只需按需覆写。

设计意图：通过把入参/返回值收敛成统一的 dataclass，再配合基类里的默认实现，
不同缓存策略可以在调度器（Scheduler）侧以完全一致的接口被调用，从而把
“缓存策略的差异”都隐藏在各自的子类实现里。
"""

from __future__ import annotations

import dataclasses
import time
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.events import KVCacheEventRecorder
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_RADIX_CACHE,
    RadixCacheMetricsCollector,
    resolve_collector_class,
)
from sglang.srt.runtime_context import get_observability

# 这些导入只在静态类型检查（TYPE_CHECKING）时生效，运行时不会真正 import，
# 用于避免循环依赖（schedule_batch / radix_cache 反过来又会引用本模块）。
if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import HiCacheController
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.cache_action import (
        CacheAction,
        ComponentAction,
    )


# ``runtime_checkable`` 让这个 Protocol 支持 ``isinstance(obj, PrefixCacheTrait)``
# 形式的运行时鸭子类型检查（只检查是否具备下列属性，不检查具体类型）。
@runtime_checkable
class PrefixCacheTrait(Protocol):
    # 任意前缀缓存都必须暴露的几个公共属性：
    req_to_token_pool: ReqToTokenPool  # 请求 -> 各 token 在 KV 池中槽位的映射表
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator  # token -> KV 缓存槽位的分配器
    page_size: int  # 分页粒度（每页包含多少个 token），page 对齐是命中/淘汰的基本单位
    disable: bool  # 是否整体禁用前缀复用（禁用后每次都视作未命中）


@dataclasses.dataclass
class MatchPrefixParams:
    """match_prefix（前缀匹配）操作的统一入参，跨不同缓存类型通用。"""

    key: RadixKey  # 待匹配的前缀键（通常由 token 序列等信息构成）

    # 仅 Mamba 缓存使用的字段：
    cow_mamba: bool = False  # 是否对 Mamba 状态执行写时复制（copy-on-write）
    req: Optional[Req] = None  # 关联的请求对象，部分缓存需要它做上下文判断


@dataclasses.dataclass
class InsertParams:
    """insert（插入）操作的统一入参，跨不同缓存类型通用。"""

    key: Optional[RadixKey] = None  # 要插入的前缀键
    value: Optional[torch.Tensor] = None  # 与该键对应的 KV 缓存槽位索引张量

    # 仅 Mamba 缓存使用的字段：
    mamba_value: Optional[torch.Tensor] = None  # Mamba 状态对应的槽位

    # DSV4 NPU C128 sidecar pages, one page id per physical C128 page group.
    c128_value: Optional[torch.Tensor] = None

    # SWA specific
    prev_prefix_len: int = 0
    swa_evicted_seqlen: int = 0

    # General
    chunked: bool = False
    priority: int = 0
    track_adopted_ranges: bool = False


@dataclasses.dataclass
class InsertResult:
    """insert（插入）操作的返回结果。"""

    prefix_len: int
    total_len: int = 0
    last_device_node: Any = None
    mamba_exist: bool = False
    inserted_host_node: Any = None
    host_insert_dropped: bool = False
    adopted_ranges: Optional[dict[ComponentType, list[tuple[int, int]]]] = None
    # Controller-applied actions from the non-stepped channels (e.g. insert_host); the stepped insert emits via InsertStepResult.actions.
    cache_actions: list[CacheAction | ComponentAction] = dataclasses.field(
        default_factory=list
    )

    def record_adopted_range(
        self, component_type: ComponentType, start: int, end: int
    ) -> None:
        if self.adopted_ranges is None or start >= end:
            return
        ranges = self.adopted_ranges.setdefault(component_type, [])
        if ranges and start <= ranges[-1][1]:
            prev_start, prev_end = ranges[-1]
            ranges[-1] = (min(prev_start, start), max(prev_end, end))
        else:
            ranges.append((start, end))


@dataclasses.dataclass
class EvictParams:
    """evict（淘汰）操作的统一入参，跨不同缓存类型通用。"""

    num_tokens: int = 0  # 期望淘汰的 Full-KV token 数量
    swa_num_tokens: int = 0  # 期望淘汰的 SWA token 数量
    mamba_num: int = 0  # 期望淘汰的 Mamba 状态数量


@dataclasses.dataclass
class EvictResult:
    """evict（淘汰）操作的返回结果，记录实际淘汰量。"""

    num_tokens_evicted: int = 0  # 实际淘汰的 Full-KV token 数量
    swa_num_tokens_evicted: int = 0  # 实际淘汰的 SWA token 数量
    mamba_num_evicted: int = 0  # 实际淘汰的 Mamba 状态数量


# 锁引用计数（lock_ref）机制说明：
#   前缀缓存以引用计数“锁住”正在被使用的树节点，被锁住的节点不会被淘汰。
#   inc_lock_ref 在请求开始使用某段前缀时加锁，dec_lock_ref 在使用结束时解锁。
@dataclasses.dataclass
class IncLockRefResult:
    """inc_lock_ref（增加锁引用计数）操作的返回结果。"""

    delta: Optional[int] = None  # 本次加锁导致的引用计数增量
    swa_uuid_for_lock: Optional[int] = None  # SWA 设备侧锁对应的句柄/唯一标识
    swa_uuid_for_host_lock: Optional[int] = None  # SWA host（CPU）侧锁对应的句柄/唯一标识
    # 加锁时刻处于“墓碑（tombstone）”状态的组件节点集合。
    # 墓碑表示该节点的设备值已失效但记录仍在。释放锁时重放（replay）这个集合，
    # 可以防止一个短命的锁在墓碑随后变回有效设备值之后，
    # 误吃掉后来的一次 load-back（回载）或请求锁。
    skip_lock_node_ids: dict[ComponentType, set[int]] = dataclasses.field(
        default_factory=dict
    )

    def to_dec_params(self) -> DecLockRefParams:
        """转换为对应的 DecLockRefParams，供后续 dec_lock_ref 解锁时使用。

        这样可以保证“加锁时记录的上下文”被原样带到解锁阶段，
        其中 ``skip_lock_node_ids`` 会做一次浅拷贝（重建每个 set），
        避免解锁参数与原结果共享同一可变集合。
        """
        return DecLockRefParams(
            swa_uuid_for_lock=self.swa_uuid_for_lock,
            swa_uuid_for_host_lock=self.swa_uuid_for_host_lock,
            skip_lock_node_ids={
                component_type: set(node_ids)
                for component_type, node_ids in self.skip_lock_node_ids.items()
            },
        )


@dataclasses.dataclass
class DecLockRefParams:
    """dec_lock_ref（减少锁引用计数）操作的入参。

    字段含义与 ``IncLockRefResult`` 对应，用于在解锁时还原加锁时的上下文。
    """

    swa_uuid_for_lock: Optional[int] = None  # SWA 设备侧锁句柄
    swa_uuid_for_host_lock: Optional[int] = None  # SWA host 侧锁句柄
    skip_lock_node_ids: dict[ComponentType, set[int]] = dataclasses.field(
        default_factory=dict
    )  # 加锁时为墓碑、解锁时需跳过的节点集合（见 IncLockRefResult 说明）


@dataclasses.dataclass
class DecLockRefResult:
    """dec_lock_ref（减少锁引用计数）操作的返回结果。"""

    delta: Optional[int] = None  # 本次解锁导致的引用计数增量（通常为负）


@dataclasses.dataclass
class InitLoadBackParams:
    """init_load_back（准备把 KV 从 host 回载到 device）操作的统一入参。"""

    best_match_node: Any  # 回载的锚点节点，即匹配阶段被所有组件校验器接受的最深节点
    host_hit_length: int  # 在 host（CPU）侧命中、需要回载到 device 的 token 数量
    mem_quota: Optional[int] = None  # 本次回载可用的显存配额（限制一次回载多少）
    req: Optional[Req] = None  # 关联的请求对象


class MatchResult(NamedTuple):
    """前缀匹配（match_prefix）操作的返回结果。

    Attributes:
        device_indices  :   Indices of the KV cache on the device matched by common prefix.
        last_device_node:   The last TreeNode on the device that was matched.
        last_host_node  :   The last TreeNode on the host that was matched.
                            Note that if HiCache is not enabled,
                            this **must** be the same as `last_device_node`.
                            Reserved for L3 storage prefetch anchoring; L2 load_back
                            uses `best_match_node` instead.
        best_match_node :   Deepest node accepted by all component validators
                            during match_prefix. Anchor for every L2 host->device
                            load_back walk (FULL / SWA / ...). For legacy caches
                            that don't run multi-component validation, set this
                            equal to `last_host_node`.
        host_hit_length :   Number of Full-KV tokens that hit on host (CPU) and need to be
                            loaded back to device. Pure-KV cache semantics;
        swa_host_hit_length  :   Number of SWA tokens that hit on host (within the sliding
                            window) and will be load-back into the SWA device pool.
        mamba_host_hit_length:   Number of Mamba slots that hit on host and will be load-back
                            into the Mamba device pool. Typically 0 or 1.
        mamba_branching_seqlen: The mamba radix cache branching point, which is the longest
                                page-aligned position that could've been cache hit if there
                                exists a mamba state.
        full_kv_hit_length: Longest Full-KV prefix available on either device or
                            host, independent of other components.
    """

    device_indices: torch.Tensor
    last_device_node: Any
    last_host_node: Any
    best_match_node: Any
    host_hit_length: int = 0
    swa_host_hit_length: int = 0
    mamba_host_hit_length: int = 0
    mamba_branching_seqlen: Optional[int] = None
    cache_protected_len: Optional[int] = None
    full_kv_hit_length: int = 0
    # Actions the Controller applies: CacheActions itself, ComponentActions routed to the owning component.
    cache_actions: Sequence[CacheAction | ComponentAction] = ()


def zero_match_result(
    tree_cache, match_result: MatchResult, extra_key: Optional[str] = None
) -> MatchResult:
    if tree_cache.is_chunk_cache():
        # Chunk（分块）缓存的 match_prefix 本就直接返回未命中，没有可回溯的 root_node，
        # 因此原样返回即可。
        return match_result
    root = tree_cache.root_node_handle(extra_key=extra_key)
    return match_result._replace(
        # [:0] 取空切片：保留原张量的 dtype 与 device（例如 CUDA int64），
        # 又不必新分配一个空张量。
        device_indices=match_result.device_indices[:0],
        last_device_node=root,
        last_host_node=root,
        best_match_node=root,
        host_hit_length=0,
        swa_host_hit_length=0,
        mamba_host_hit_length=0,
        full_kv_hit_length=0,
    )


class BasePrefixCache(ABC, PrefixCacheTrait):
    """所有前缀缓存的抽象基类，可按 rid（请求 id）或 key（前缀键）建立索引。

    本类定义了前缀缓存对外的统一接口：
      * 用 ``@abstractmethod`` 声明子类**必须**实现的核心操作
        （reset / match_prefix / cache_finished_req / cache_unfinished_req
        / evict / inc_lock_ref / dec_lock_ref）。
      * 为一批可选能力提供“默认实现”，使得不需要这些能力的子类无需覆写：
        - 各种 size 查询默认返回 0；
        - SWA / Mamba / 流式会话等能力默认“不支持”；
        - HiCache 相关的分级写回/回载方法默认抛 NotImplementedError 或为空操作。
    这样调度器可以用同一套接口调用任意缓存实现，差异都收敛在子类内部。
    """

    metrics_collector: Optional[RadixCacheMetricsCollector] = (
        None  # 该缓存的指标采集器（observability metrics），默认未初始化
    )
    cache_controller: Optional[HiCacheController] = None
    # Set by caches that publish KV placement events; None means they don't.
    kv_events: Optional[KVCacheEventRecorder] = None

    def init_metrics_collector(self):
        labels = {"cache_type": self.__class__.__name__}
        if get_observability().extra_metric_labels:
            labels.update(get_observability().extra_metric_labels)
        radix_cache_cls = resolve_collector_class(
            STAT_LOGGER_ROLE_RADIX_CACHE,
            RadixCacheMetricsCollector,
        )
        self.metrics_collector = radix_cache_cls(labels=labels)

    def update_eviction_metrics(self, num_evicted: int, start_time: float):
        """记录一次淘汰（eviction）的耗时与淘汰 token 数到指标系统。

        仅当采集器已初始化且本次确有淘汰（num_evicted > 0）时才上报，
        避免空操作污染监控数据。``start_time`` 应为淘汰开始前用
        ``time.perf_counter()`` 取得的时间戳。
        """
        if self.metrics_collector is not None and num_evicted > 0:
            self.metrics_collector.observe_eviction_duration(
                time.perf_counter() - start_time
            )
            self.metrics_collector.increment_eviction_num_tokens(num_evicted)

    def release_host_resources(self) -> None:
        """Release pinned host buffers in userspace on graceful shutdown.

        Kernel-side unpinning during process reclaim can stall teardown for
        tens of seconds (see HostKVCache.destroy). Idempotent.
        """

    @abstractmethod
    def reset(self):
        """重置缓存到初始空状态（清空所有缓存的前缀）。子类必须实现。"""
        pass

    @abstractmethod
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """匹配给定 key 的最长公共前缀，返回命中信息。子类必须实现。"""
        pass

    def supports_fast_match_prefix(self) -> bool:
        """是否支持“快速前缀匹配”路径，默认不支持。

        支持快速匹配的子类可覆写为 True，调度器据此走更轻量的匹配逻辑。
        """
        return False

    def resolve_node_handle(self, node_handle: Any) -> Any:
        """Map a node handle to its node -- e.g. UnifiedRadixCache looks up the
        node object from its NodeId. Temporary API for the Unified Radix Cache
        split migration.

        TODO(Jialin): Remove after the Unified Radix Cache split.
        """
        return node_handle

    def root_node_handle(self, extra_key: Optional[str] = None) -> Any:
        """The root handle as match results carry it -- the raw node by default,
        the root's NodeId for UnifiedRadixCache. extra_key scopes the root for
        implementations that shard trees per cache namespace."""
        return self.root_node

    def is_backuped(self, node: Any) -> bool:
        """Whether the node's Full KV is present on host."""
        return node.backuped

    def is_root(self, node: Any) -> bool:
        """Whether the node is a tree root."""
        return node is self.root_node

    def get_last_hash_value(self, node: Any) -> Optional[str]:
        """The node's last page hash, or None when it was never hashed."""
        return node.get_last_hash_value()

    def get_prefix_hash_values(self, node: Any) -> list[str]:
        """The hash chain of the node's ancestors, in root-to-parent order."""
        return node.get_prefix_hash_values(node.parent)

    @abstractmethod
    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs):
        """请求结束时，把其 KV 缓存写入缓存结构（默认插入）。子类必须实现。"""
        pass

    @abstractmethod
    def cache_unfinished_req(self, req: Req, **kwargs):
        """请求尚未结束（如分块/中间态）时，缓存其已生成的部分。子类必须实现。"""
        pass

    @abstractmethod
    def evict(self, params: EvictParams) -> EvictResult:
        """按需淘汰缓存以腾出空间，返回实际淘汰量。子类必须实现。"""
        pass

    def evict_for_alloc(self, params: EvictParams) -> EvictResult:
        """Evict cache entries to cover allocator shortfalls.

        The default implementation preserves the component-count semantics of
        :meth:`evict`. Multi-component caches backed by shared memory can
        override this entry point to stop once collateral frees make the
        requested allocation feasible.
        """
        return self.evict(params)

    @abstractmethod
    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        """对某节点增加锁引用计数，防止其在使用期间被淘汰。子类必须实现。"""
        pass

    @abstractmethod
    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        """对某节点减少锁引用计数，使用结束后允许其被淘汰。子类必须实现。"""
        pass

    # ---- size 查询：默认全部返回 0，由具体缓存子类按需覆写 ----
    # “evictable（可淘汰）”指当前未被锁、可被淘汰以腾出空间的 token 量；
    # “protected（受保护）”指被锁引用、暂不可淘汰的 token 量。
    # full / swa 前缀分别对应 Full-KV 池与 SWA 池。
    def evictable_size(self):
        """当前可淘汰的 token 总量。"""
        return 0

    def full_evictable_size(self):
        """Full-KV 池中可淘汰的 token 量。"""
        return 0

    def swa_evictable_size(self):
        """SWA 池中可淘汰的 token 量。"""
        return 0

    def protected_size(self):
        """当前受保护（被锁、不可淘汰）的 token 总量。"""
        return 0

    def full_protected_size(self):
        """Full-KV 池中受保护的 token 量。"""
        return 0

    def swa_protected_size(self):
        """SWA 池中受保护的 token 量。"""
        return 0

    def total_size(self):
        """缓存可容纳的 token 总量；基类不提供，需子类实现。"""
        raise NotImplementedError()

    def pretty_print(self):
        """以可读形式打印缓存内部结构（调试用）；需子类实现。"""
        raise NotImplementedError()

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ) -> Tuple[torch.Tensor, Any]:
        """准备把 KV 缓存从 host（CPU）回载到 device（显存）。

        仅 HiCache 等分级缓存需要此能力，基类默认不支持。
        """
        raise NotImplementedError()

    def ready_to_load_host_cache(self) -> Any:
        """通知缓存控制器开始执行 KV 缓存的回载（host->device）。

        仅 HiCache 等分级缓存需要此能力，基类默认不支持。
        """
        raise NotImplementedError()

    def check_hicache_events(self) -> Any:
        """检查 HiCache 相关活动，必要时更新 radix 树并在各 TP（张量并行）worker 间同步。

        仅 HiCache 场景需要，基类默认不支持。
        """
        raise NotImplementedError()

    def take_events(self):
        return [] if self.kv_events is None else self.kv_events.take()

    # ---- 能力声明：默认均为“不支持”，具备相应能力的子类覆写为 True ----
    def supports_swa(self) -> bool:
        """是否支持 SWA（滑动窗口注意力）缓存。"""
        return False

    def swa_retain_floor(self, req) -> int | None:
        # A match lands on a state checkpoint rather than on the tail, so a cache
        # that pairs SWA with mamba/conv checkpoints has to keep the window behind
        # the last checkpoint. Those caches override this. Everyone else has
        # nothing deeper than the tail to protect.
        return None

    def swa_reprefill_tail_tokens(self) -> int:
        # Only the unified_kv compress-only HiCache layout needs to hold back a
        # trailing sliding window for re-prefill; every other cache keeps SWA
        # content-stable and overrides this where relevant.
        return 0

    def supports_mamba(self) -> bool:
        """是否支持 Mamba 状态缓存。"""
        return False

    def supports_streaming_session(self) -> bool:
        """是否支持流式会话（streaming session）。"""
        return False

    def release_session(self, session_id: str) -> None:
        """释放指定流式会话占用的资源，默认无操作。"""
        pass

    def release_radix_session(self, session_id: str) -> None:
        pass

    def session_held_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        """会话当前持有的 token 总量。"""
        return 0

    def session_held_full_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        """会话当前在 Full-KV 池中持有的 token 量。"""
        return 0

    def session_held_swa_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        """会话当前在 SWA 池中持有的 token 量。"""
        return 0

    def session_held_req_count(self, active_pool_idxs: Optional[set] = None) -> int:
        """会话当前持有的请求数量。"""
        return 0

    def session_held_mamba_slots(self, active_pool_idxs: Optional[set] = None) -> int:
        """会话当前持有的 Mamba 槽位数量。"""
        return 0

    def is_chunk_cache(self) -> bool:
        """是否为 Chunk（分块）缓存。默认 False，即默认是树形缓存。"""
        return False

    def is_tree_cache(self) -> bool:
        """是否为树形（radix tree）缓存，定义为“非 Chunk 缓存”。"""
        return not self.is_chunk_cache()

    def available_and_evictable_str(self) -> str:
        """返回一行可读字符串，汇报“可用 + 可淘汰”的 token 数量（用于日志/调试）。

        可用 token 来自 KV 池分配器的空闲量，加上当前可淘汰量，即为短期内
        实际可供新请求使用的 token 上限。
        """
        available_size = self.token_to_kv_pool_allocator.available_size()
        evictable_size = self.evictable_size()
        return f"Available tokens: {available_size + evictable_size} ({available_size=} + {evictable_size=})\n"
