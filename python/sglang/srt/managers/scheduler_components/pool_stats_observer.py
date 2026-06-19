"""Pool statistics observation for the scheduler.

中译：调度器（Scheduler）的「内存/KV 池统计观测」组件。
      本模块负责汇总各类 KV 缓存池的使用情况，向上层提供统一的统计快照：
      - PoolStats：一个数据快照，描述某一时刻各池（full / SWA / mamba / HiSparse）
        的占用量、使用率、可用与可淘汰（evictable）大小，并能格式化成日志文本，
        以及把这些字段回填到 SchedulerStats（用于上报指标）。
      - SchedulerPoolStatsObserver：观测器本体，持有各内存池/缓存的引用，
        按当前模型类型（普通 / hybrid-SWA 滑窗 / hybrid-SSM mamba / HiSparse 分层）
        采集对应的 token 统计，生成 PoolStats。
      术语：full 池指完整注意力的 KV 池；SWA 指滑动窗口注意力（sliding window attention）；
      SSM/mamba 指状态空间模型（Mamba）的状态池；HiSparse 指设备/主机分层稀疏缓存。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
    Tuple,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


# 中译：占位空类。仅用于类型注解（避免与真正的 SchedulerStats 形成循环 import），
#       运行时被同名真实类型遮蔽（# type: ignore[no-redef]）。
class SchedulerStats: ...  # type: ignore[no-redef]


@dataclasses.dataclass
class PoolStats:
    # 中译：某一时刻所有内存池统计的「快照」数据类。按模型类型只填充相应字段，
    #       通过 is_hybrid_swa / is_hybrid_ssm / is_hisparse 标志区分哪些字段有效。
    # For full pools (required)
    # 中译：full（完整注意力）KV 池——必填字段。
    full_num_used: int  # 已使用 token 数
    full_token_usage: float  # 使用率（已用 / 总量）
    full_available_size: int  # 当前可分配（空闲）大小
    full_evictable_size: int  # 可被淘汰（缓存中、必要时可释放）的大小

    # 中译：标志位——指示本快照属于哪种池形态，决定下方哪些可选字段被填充。
    is_hybrid_swa: bool = False
    is_hybrid_ssm: bool = False
    is_hisparse: bool = False

    # For hybrid-swa pools
    # 中译：hybrid-SWA（滑动窗口注意力）池专用字段。
    swa_num_used: Optional[int] = None
    swa_token_usage: Optional[float] = None
    swa_available_size: Optional[int] = None
    swa_evictable_size: Optional[int] = None

    # For mamba pools
    # 中译：mamba（SSM 状态空间模型）状态池专用字段。
    mamba_num_used: Optional[int] = None
    mamba_usage: Optional[float] = None
    mamba_available_size: Optional[int] = None
    mamba_evictable_size: Optional[int] = None

    # HiSparse device/host breakdown for decode logs (plain KV pool only)
    # 中译：HiSparse 的设备（GPU）/主机（CPU）分层明细，仅用于普通 KV 池的 decode 日志。
    hisparse_device_tokens: Optional[int] = None
    hisparse_device_token_usage: Optional[float] = None
    hisparse_host_tokens: Optional[int] = None
    hisparse_host_token_usage: Optional[float] = None

    def get_kv_token_stats(self) -> Tuple[int, float]:
        # 中译：返回 (已用 token 数, 使用率) 二元组，作为对外的「KV 占用」概览。
        #       SWA 模式下取 full 与 swa 两者的较大值（瓶颈池决定整体占用）。
        # NOTE: mamba pool is not included in the "token usage" calculation.
        # 中译：注意——mamba 池不计入「token usage」统计（它衡量的是 KV token，与 mamba 状态不同口径）。
        if self.is_hybrid_swa:
            num_used = max(self.full_num_used, self.swa_num_used)
            token_usage = max(self.full_token_usage, self.swa_token_usage)
        else:
            num_used = self.full_num_used
            token_usage = self.full_token_usage

        return num_used, token_usage

    def get_max_pool_usage(self) -> float:
        # 中译：返回所有相关池中「最高的使用率」，即整体内存压力的瓶颈值
        #       （SWA / mamba 池存在时一并参与取 max）。用于触发限流/淘汰等决策。
        usage = self.full_token_usage
        if self.is_hybrid_swa:
            usage = max(usage, self.swa_token_usage)
        if self.is_hybrid_ssm:
            usage = max(usage, self.mamba_usage)
        assert usage is not None and usage >= 0, f"{usage=} is not valid"
        return usage

    def get_prefill_usage_msg_parts(self) -> List[str]:
        # 中译：组装 prefill 阶段日志的「使用率」文本片段列表（按池类型选择展示项）。
        parts = []
        if self.is_hybrid_swa:
            parts += [
                f"full token usage: {self.full_token_usage:.2f}",
                f"swa token usage: {self.swa_token_usage:.2f}",
            ]
        if self.is_hybrid_ssm:
            if not self.is_hybrid_swa:
                parts.append(f"full token usage: {self.full_token_usage:.2f}")
            parts.append(f"mamba usage: {self.mamba_usage:.2f}")
        if not parts:
            parts.append(f"token usage: {self.full_token_usage:.2f}")
        return parts

    def get_decode_usage_msg_parts(self) -> List[str]:
        # 中译：组装 decode 阶段日志的文本片段列表，比 prefill 版更详细
        #       （额外含已用 token 数、HiSparse 的 GPU/CPU 分层明细等）。
        parts = []
        if self.is_hybrid_swa:
            parts += [
                f"#full token: {self.full_num_used}",
                f"full token usage: {self.full_token_usage:.2f}",
                f"#swa token: {self.swa_num_used}",
                f"swa token usage: {self.swa_token_usage:.2f}",
            ]
        if self.is_hybrid_ssm:
            if not self.is_hybrid_swa:
                parts += [
                    f"#full token: {self.full_num_used}",
                    f"full token usage: {self.full_token_usage:.2f}",
                ]
            parts += [
                f"mamba num: {self.mamba_num_used}",
                f"mamba usage: {self.mamba_usage:.2f}",
            ]
        if self.is_hisparse:
            parts += [
                f"#gpu token: {self.hisparse_device_tokens}",
                f"gpu token usage: {self.hisparse_device_token_usage:.2f}",
                f"#cpu token: {self.hisparse_host_tokens}",
                f"cpu token usage: {self.hisparse_host_token_usage:.2f}",
            ]
        if not parts:
            parts.append(
                f"#token: {self.full_num_used}, token usage: {self.full_token_usage:.2f}"
            )
        return parts

    def update_scheduler_stats(self, stats: SchedulerStats) -> None:
        """Update pool-related fields on SchedulerStats.

        中译：把本快照里的各池数据回填到 SchedulerStats 对象（供 metrics 上报使用），
              按池类型有选择地写入 SWA / mamba 字段。
        """
        num_used, _ = self.get_kv_token_stats()
        stats.num_used_tokens = num_used
        stats.token_usage = round(self.get_max_pool_usage(), 2)
        stats.full_token_usage = self.full_token_usage
        if self.is_hybrid_swa:
            stats.swa_token_usage = self.swa_token_usage
            stats.swa_available_tokens = self.swa_available_size
            stats.swa_evictable_tokens = self.swa_evictable_size
            stats.swa_used_tokens = self.swa_num_used
        if self.is_hybrid_ssm:
            stats.mamba_usage = self.mamba_usage
            stats.mamba_available_tokens = self.mamba_available_size
            stats.mamba_evictable_tokens = self.mamba_evictable_size
            stats.mamba_used_tokens = self.mamba_num_used
        stats.kv_available_tokens = self.full_available_size
        stats.kv_evictable_tokens = self.full_evictable_size
        stats.kv_used_tokens = self.full_num_used


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerPoolStatsObserver:
    # 中译：内存池统计观测器。冻结（frozen）数据类，持有各内存池/缓存以及一些回调，
    #       负责按当前模型形态采集 token 统计并产出 PoolStats 快照。
    tree_cache: BasePrefixCache  # 前缀缓存（radix tree），提供 evictable/protected 等统计
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator  # KV 池分配器（可用大小来源）
    req_to_token_pool: ReqToTokenPool  # 请求槽池（含 mamba 子池）
    session_controller: Any  # 会话控制器，用于统计「会话持有」的 token/槽位
    hisparse_coordinator: Any  # HiSparse 分层缓存协调器
    is_hybrid_swa: bool  # 是否 hybrid-SWA（滑窗注意力）模型
    is_hybrid_ssm: bool  # 是否 hybrid-SSM（mamba）模型
    enable_hisparse: bool  # 是否启用 HiSparse 分层
    full_tokens_per_layer: Any  # full 池每层总 token 数（SWA 口径）
    swa_tokens_per_layer: Any  # swa 池每层总 token 数
    max_total_num_tokens: int  # 全局最大 token 数
    get_last_batch: Callable  # 取上一批次（回调）
    get_running_batch: Callable  # 取运行中批次（回调）

    def streaming_session_count(self) -> int:
        # 中译：统计当前处于「流式（streaming）」状态的会话数量。
        return sum(
            1
            for session in self.session_controller.sessions.values()
            if session.streaming
        )

    def active_pool_idxs(self) -> set:
        """Pool idxs currently owned by reqs in last_batch / running_batch.

        Used to decide which session slots' KV is owned by batch reqs
        (and thus counted via uncached_size, not session_held).

        中译：返回当前被 last_batch / running_batch 中请求占用的池下标（req_pool_idx）集合。
              用于判定哪些会话槽位的 KV 实际由「活跃批次请求」持有——这部分应计入
              uncached_size（未缓存）而非 session_held（会话持有），避免重复计数。
        """
        idxs = set()
        for batch in [self.get_last_batch(), self.get_running_batch()]:
            if batch is None or batch.is_empty():
                continue
            for req in batch.reqs:
                if req.req_pool_idx is not None:
                    idxs.add(req.req_pool_idx)
        return idxs

    def session_held_tokens(self) -> int:
        # 中译：被会话（session）持有、且不属于活跃批次的 token 数（避免与批次内 token 重复计数）。
        return self.tree_cache.session_held_tokens(self.active_pool_idxs())

    def session_held_full_tokens(self) -> int:
        # 中译：同上，限 full 池口径。
        return self.tree_cache.session_held_full_tokens(self.active_pool_idxs())

    def session_held_swa_tokens(self) -> int:
        # 中译：同上，限 swa 池口径。
        return self.tree_cache.session_held_swa_tokens(self.active_pool_idxs())

    def session_held_req_count(self) -> int:
        # 中译：被会话持有的请求槽位数量（用于 req_to_token_pool 的不变量检查）。
        return self.tree_cache.session_held_req_count()

    def session_held_mamba_slots(self) -> int:
        # 中译：被会话持有的 mamba 状态槽位数量。
        return self.tree_cache.session_held_mamba_slots(self.active_pool_idxs())

    def get_pool_stats(self) -> PoolStats:
        # 中译：对外主入口——按模型形态分派到对应的采集方法，产出统一的 PoolStats 快照。
        #       SWA 与 SSM 可共存：先按 SWA/普通采集，再把 mamba 字段叠加上去。
        if self.is_hybrid_swa:
            pool_stats = self._get_swa_token_info()
        elif self.is_hybrid_ssm:
            pool_stats = self._get_mamba_token_info()
        else:
            pool_stats = self._get_token_info()

        if self.enable_hisparse:
            # 中译：启用 HiSparse 时，叠加设备/主机分层的 token 明细。
            pool_stats = self._get_hisparse_token_info(pool_stats)

        # swa + ssm can coexist: overlay mamba fields onto swa stats
        # 中译：SWA 与 SSM 可同时存在——此处把 mamba 字段叠加到已采集的（SWA）快照上。
        if self.is_hybrid_ssm:
            mamba_stats = self._get_mamba_token_info()
            pool_stats.is_hybrid_ssm = True
            pool_stats.mamba_num_used = mamba_stats.mamba_num_used
            pool_stats.mamba_usage = mamba_stats.mamba_usage
            pool_stats.mamba_available_size = mamba_stats.mamba_available_size
            pool_stats.mamba_evictable_size = mamba_stats.mamba_evictable_size

        return pool_stats

    def _get_token_info(self) -> PoolStats:
        # 中译：普通（非 SWA / 非 SSM）模型的 token 统计。
        #       已用 = 总量 - (可用 + 可淘汰)；其中可淘汰指缓存中、必要时能释放的部分。
        available_size = self.token_to_kv_pool_allocator.available_size()
        evictable_size = self.tree_cache.evictable_size()
        num_used = self.max_total_num_tokens - (available_size + evictable_size)
        token_usage = num_used / self.max_total_num_tokens
        return PoolStats(
            full_num_used=num_used,
            full_token_usage=token_usage,
            full_available_size=available_size,
            full_evictable_size=evictable_size,
        )

    def _get_hisparse_token_info(self, pool_stats: PoolStats) -> PoolStats:
        # 中译：向已有快照叠加 HiSparse 的设备（GPU）/主机（CPU）分层 token 明细。
        #       用 dataclasses.replace 生成新快照（PoolStats 由 @dataclass 生成、可 replace）。
        if self.enable_hisparse and self.hisparse_coordinator is not None:
            h = self.hisparse_coordinator.get_token_stats()
            return dataclasses.replace(
                pool_stats,
                is_hisparse=True,
                hisparse_device_tokens=h.device_tokens,
                hisparse_device_token_usage=h.device_token_usage,
                hisparse_host_tokens=h.host_tokens,
                hisparse_host_token_usage=h.host_token_usage,
            )
        return pool_stats

    def _get_mamba_token_info(self):
        # 中译：mamba（SSM）模型的 token 统计：同时计算 full KV 池与 mamba 状态池两套数据。
        #       仅当前缀缓存支持 mamba 且是树形缓存时，evictable 才有意义，否则取 0。
        is_mamba_radix_cache = (
            self.tree_cache.supports_mamba() and self.tree_cache.is_tree_cache()
        )
        full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable_size = (
            self.tree_cache.full_evictable_size() if is_mamba_radix_cache else 0
        )
        mamba_available_size = self.req_to_token_pool.mamba_allocator.available_size()
        mamba_evictable_size = (
            self.tree_cache.mamba_evictable_size() if is_mamba_radix_cache else 0
        )
        full_num_used = self.token_to_kv_pool_allocator.size - (
            full_available_size + full_evictable_size
        )
        mamba_num_used = self.req_to_token_pool.mamba_pool.size - (
            mamba_available_size + mamba_evictable_size
        )
        full_token_usage = full_num_used / self.token_to_kv_pool_allocator.size
        mamba_usage = mamba_num_used / self.req_to_token_pool.mamba_pool.size

        return PoolStats(
            is_hybrid_ssm=True,
            full_num_used=full_num_used,
            full_token_usage=full_token_usage,
            full_available_size=full_available_size,
            full_evictable_size=full_evictable_size,
            mamba_num_used=mamba_num_used,
            mamba_usage=mamba_usage,
            mamba_available_size=mamba_available_size,
            mamba_evictable_size=mamba_evictable_size,
        )

    def _get_swa_token_info(self) -> PoolStats:
        # 中译：hybrid-SWA（滑窗注意力）模型的 token 统计：分别计算 full 与 swa 两个池。
        #       每池：已用 = 每层总量 - (可用 + 可淘汰)。
        full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        full_evictable_size = self.tree_cache.full_evictable_size()
        swa_available_size = self.token_to_kv_pool_allocator.swa_available_size()
        swa_evictable_size = self.tree_cache.swa_evictable_size()
        full_num_used = self.full_tokens_per_layer - (
            full_available_size + full_evictable_size
        )
        swa_num_used = self.swa_tokens_per_layer - (
            swa_available_size + swa_evictable_size
        )
        # FIXME(hisparse): host-backup transiently over-releases the device pool
        # counter, producing negative full_num_used / swa_num_used. We clamp to 0
        # to keep token_usage / leak checks sane, but the underlying accounting
        # bug should be fixed so the clamp can go away.
        # 中译：HiSparse 的主机备份过程会瞬时「过度释放」设备池计数，导致已用数算出负值；
        #       这里 clamp 到 0 以保证使用率/泄漏检查不出错。这是临时规避，根本的计数 bug 待修。
        if self.enable_hisparse:
            full_num_used = max(0, full_num_used)
            swa_num_used = max(0, swa_num_used)
        full_token_usage = full_num_used / self.full_tokens_per_layer
        swa_token_usage = swa_num_used / self.swa_tokens_per_layer

        return PoolStats(
            is_hybrid_swa=True,
            full_num_used=full_num_used,
            full_token_usage=full_token_usage,
            full_available_size=full_available_size,
            full_evictable_size=full_evictable_size,
            swa_num_used=swa_num_used,
            swa_token_usage=swa_token_usage,
            swa_available_size=swa_available_size,
            swa_evictable_size=swa_evictable_size,
        )
