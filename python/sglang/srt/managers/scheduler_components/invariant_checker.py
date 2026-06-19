"""Runtime memory-pool invariant / consistency checks for the scheduler.

中译：调度器的「运行时不变量（invariant）/一致性检查」组件。
      核心思想：对每个内存池，维护一个会计恒等式——
        available + evictable + protected + session_held + uncached == total
      （可用 + 可淘汰 + 受保护 + 会话持有 + 未缓存 == 总量）。
      若该等式不成立，说明发生了内存「泄漏」（某些 token/页未被正确归还），
      据此告警或直接报错（取决于严格检查的开关级别）。
      - SchedulerInvariantChecker：对 full / SWA / mamba / req 各池做检查，
        分「忙碌中（busy）」与「空闲（idle）」两种时机；mamba 泄漏时还会做页级诊断。
      - create_scheduler_watchdog：构造调度器看门狗，卡死时 dump 池状态辅助排查。
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Callable,
    Deque,
    List,
    Optional,
    Tuple,
)

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    PoolStats,
    SchedulerPoolStatsObserver,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import (
    ceil_align,
    raise_error_or_warn,
)
from sglang.srt.utils.watchdog import WatchdogRaw

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


logger = logging.getLogger(__name__)

# Number of recent busy-check messages buffered for the level-1 dump-on-leak path.
# 中译：level-1（安静模式）下缓存的最近忙碌检查日志条数；仅在检测到泄漏时才把这些回放打印出来。
BUSY_MEM_CHECK_LOG_RING_SIZE = 1000


@dataclass(kw_only=True, slots=True)
class SchedulerInvariantChecker:
    # 中译：不变量检查器。持有各内存池/缓存引用及池统计观测器，提供多种一致性检查方法。
    #       count_*_warnings 用于累计告警次数（非严格模式下用于限流/统计）。
    is_hybrid_swa: bool
    is_hybrid_ssm: bool
    disaggregation_mode: DisaggregationMode
    page_size: int
    full_tokens_per_layer: Optional[int]
    swa_tokens_per_layer: Optional[int]
    max_total_num_tokens: int
    server_args: ServerArgs
    tree_cache: BasePrefixCache
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    req_to_token_pool: ReqToTokenPool
    pool_stats_observer: SchedulerPoolStatsObserver
    get_last_batch: Callable
    get_running_batch: Callable
    count_req_pool_leak_warnings: int = 0
    count_memory_leak_warnings: int = 0
    recent_busy_msgs: Deque[str] = field(
        default_factory=lambda: deque(maxlen=BUSY_MEM_CHECK_LOG_RING_SIZE)
    )

    @staticmethod
    def _check_pool_invariant(
        pool_name: str,
        available: int,
        evictable: int,
        protected: int,
        session_held: int,
        total: int,
        uncached: int = 0,
    ) -> Tuple[bool, str]:
        """Check: available + evictable + protected + session_held + uncached == total.

        中译：核心不变量检查——把五部分加总与 total 比较，不等即判定为泄漏（leak）。
              返回 (是否泄漏, 诊断文本)。
        """
        total_accounted = available + evictable + protected + session_held + uncached
        leak = total_accounted != total
        msg = (
            f"[{pool_name}] {total=}, {available=}, {evictable=}, "
            f"{protected=}, {session_held=}, {uncached=}"
        )
        return leak, msg

    def _check_full_pool(self, ps: PoolStats, uncached: int = 0) -> Tuple[bool, str]:
        # 中译：检查 full 池的不变量。protected/session_held/total 的取法随模型形态而不同
        #       （SWA 用每层总量、mamba 用分配器 size、普通用 max_total_num_tokens）。
        if self.is_hybrid_swa:
            protected = self.tree_cache.full_protected_size()
            session_held = self.pool_stats_observer.session_held_full_tokens()
            total = self.full_tokens_per_layer
        elif self.is_hybrid_ssm and self.tree_cache.supports_mamba():
            protected = self.tree_cache.full_protected_size()
            session_held = self.pool_stats_observer.session_held_tokens()
            total = self.token_to_kv_pool_allocator.size
        else:
            protected = self.tree_cache.protected_size()
            session_held = self.pool_stats_observer.session_held_tokens()
            total = self.max_total_num_tokens
        return self._check_pool_invariant(
            "full",
            ps.full_available_size,
            ps.full_evictable_size,
            protected,
            session_held,
            total,
            uncached,
        )

    def _check_swa_pool(self, ps: PoolStats, uncached: int = 0) -> Tuple[bool, str]:
        # 中译：检查 swa（滑窗）池的不变量。
        return self._check_pool_invariant(
            "swa",
            ps.swa_available_size,
            ps.swa_evictable_size,
            self.tree_cache.swa_protected_size(),
            self.pool_stats_observer.session_held_swa_tokens(),
            self.swa_tokens_per_layer,
            uncached,
        )

    def _check_mamba_pool(self, ps: PoolStats) -> Tuple[bool, str]:
        # 中译：检查 mamba 状态池的不变量。一旦发现泄漏，进一步做「页级诊断」：
        #       通过 期望页集合 - 空闲页 - 已缓存页 计算出「泄漏的页」，附在诊断信息里，
        #       full 与 mamba 两套页都各算一遍，便于定位是哪部分没归还。
        leak, msg = self._check_pool_invariant(
            "mamba",
            ps.mamba_available_size,
            ps.mamba_evictable_size,
            self.tree_cache.mamba_protected_size(),
            self.pool_stats_observer.session_held_mamba_slots(),
            self.req_to_token_pool.mamba_pool.size,
        )
        if leak:
            # Page-level leak diagnosis for mamba
            # 中译：mamba 的页级泄漏诊断——分别算出 full 和 mamba 各自「应存在但既不空闲也不在缓存」的页。
            free_full_pages = set(
                self.token_to_kv_pool_allocator.free_pages.tolist()
                + self.token_to_kv_pool_allocator.release_pages.tolist()
            )
            cached_full_pages = set(self.tree_cache.all_values_flatten().tolist())
            expected_full_pages = set(
                range(1, self.token_to_kv_pool_allocator.size + 1)
            )
            leaked_full_pages = (
                expected_full_pages - free_full_pages - cached_full_pages
            )
            mamba_allocator = self.req_to_token_pool.mamba_allocator
            free_mamba_pages = set(mamba_allocator.free_slots.tolist())
            cached_mamba_pages = set(
                self.tree_cache.all_mamba_values_flatten().tolist()
            )
            expected_mamba_pages = set(range(1, mamba_allocator.size + 1))
            leaked_mamba_pages = (
                expected_mamba_pages - free_mamba_pages - cached_mamba_pages
            )
            msg += (
                f", leaked_full_pages={leaked_full_pages or None}"
                f", leaked_mamba_pages={leaked_mamba_pages or None}"
            )
        return leak, msg

    def _get_total_uncached_sizes(
        self,
    ) -> Tuple[int, int]:
        """Sum uncached tokens for full and SWA pools across all active batches.

        Returns (full_uncached, swa_uncached). For non-SWA models, swa_uncached is 0.

        For full pool: uncached = allocated - cache_protected_len
        For SWA pool:  uncached = allocated - max(cache_protected_len, swa_evicted_seqlen)

        中译：统计所有活跃批次中「已分配但尚未进入缓存」的 token 数（按 full / swa 池分别求和）。
              full 池：uncached = 已分配长度 - 已被缓存保护的长度；
              swa 池：uncached = 已分配长度 - max(缓存保护长度, swa 已淘汰序列长度)。
        """
        # After decode: running_batch IS last_batch (same object), count once.
        # After prefill: they differ, both hold uncached tokens.
        # Use identity (is / is not), not membership or ==: ScheduleBatch's
        # dataclass __eq__ compares tensor fields and raises on ambiguous bools.
        # 中译：decode 之后 running_batch 与 last_batch 是同一对象，只数一次；
        #       prefill 之后两者不同，都持有 uncached token，需各数一次。
        #       务必用「is / is not」做同一性判断而非 == 或 in——ScheduleBatch 的 dataclass
        #       __eq__ 会比较张量字段，在布尔语义模糊时会抛异常。
        last_batch = self.get_last_batch()
        running_batch = self.get_running_batch()
        batches = [last_batch]
        if (
            running_batch is not None
            and running_batch is not last_batch
            and not running_batch.is_empty()
        ):
            batches.append(running_batch)

        full_uncached = 0
        swa_uncached = 0
        for batch in batches:
            for req in batch.reqs:
                # 中译：已释放（committed/overallocated 都已 free）或尚未分配槽位的请求，跳过不计。
                assert req.kv_committed_freed == req.kv_overallocated_freed
                if req.kv_committed_freed or req.req_pool_idx is None:
                    continue

                allocated_len = req.kv_allocated_len
                # 中译：分页大于 1 时，已分配长度需向上对齐到页边界（缓存保护长度本就是页对齐的）。
                if self.page_size > 1:
                    allocated_len = ceil_align(allocated_len, self.page_size)
                    assert req.cache_protected_len % self.page_size == 0

                full_uncached += allocated_len - req.cache_protected_len
                if self.is_hybrid_swa:
                    swa_uncached += allocated_len - max(
                        req.cache_protected_len, req.swa_evicted_seqlen
                    )

        return full_uncached, swa_uncached

    def self_check_during_busy(self):
        # 中译：「忙碌中」的自检（每轮前向后调用）。计入 uncached 后检查 full/SWA 池不变量。
        #       日志行为受环境变量 SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY 的级别控制：
        #       >1 每轮都打印；==1 安静缓存、仅泄漏时回放最近若干条。最后用 assert 强制保证无泄漏。
        if self.get_last_batch() is None:
            return

        ps = self.pool_stats_observer.get_pool_stats()
        full_uncached, swa_uncached = self._get_total_uncached_sizes()

        full_leak, full_msg = self._check_full_pool(ps, uncached=full_uncached)

        swa_leak, swa_msg = False, ""
        if self.is_hybrid_swa:
            swa_leak, swa_msg = self._check_swa_pool(ps, uncached=swa_uncached)

        level = envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get()
        full_line = f"[Mem Check (BUSY)] {full_msg}"
        swa_line = f"[Mem Check (BUSY)] {swa_msg}" if swa_msg else None

        if level > 1:
            # Verbose: log every iteration.
            # 中译：详尽模式——每轮都打印。
            logger.info(full_line)
            if swa_line:
                logger.info(swa_line)
        elif level == 1:
            # Quiet: buffer and stay silent; flush the recent ones only on a leak.
            # 中译：安静模式——平时只缓存不打印；仅在检测到泄漏时把最近缓存的若干条一并刷出。
            self.recent_busy_msgs.append(full_line)
            if swa_line:
                self.recent_busy_msgs.append(swa_line)
            if full_leak or swa_leak:
                for msg in self.recent_busy_msgs:
                    logger.info(msg)

        assert not full_leak, f"Full Pool Mem Leak Detected! {full_msg}"
        assert not swa_leak, f"SWA Pool Mem Leak Detected! {swa_msg}"

    def _check_req_pool(self):
        # 中译：检查请求槽池（req_to_token_pool）的不变量：空闲槽 + 会话持有槽 == 总槽数。
        #       DECODE 分离模式下总量需加上预分配（pre_alloc）大小。不满足则按严格级别告警/报错。
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            req_total_size = (
                self.req_to_token_pool.size + self.req_to_token_pool.pre_alloc_size
            )
        else:
            req_total_size = self.req_to_token_pool.size

        session_req_count = self.pool_stats_observer.session_held_req_count()
        if len(self.req_to_token_pool.free_slots) + session_req_count != req_total_size:
            msg = (
                "req_to_token_pool memory leak detected!"
                f"available_size={len(self.req_to_token_pool.free_slots)}, "
                f"session_held={session_req_count}, "
                f"total_size={self.req_to_token_pool.size}\n"
            )
            raise_error_or_warn(
                self,
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
                "count_req_pool_leak_warnings",
                msg,
            )

    def _report_leak(self, pool_name: str, token_msg: str):
        # 中译：统一的泄漏上报入口。按 idle 严格检查开关决定是抛错还是仅告警，并累加告警计数。
        msg = f"{pool_name} memory leak detected! {token_msg}"
        raise_error_or_warn(
            self,
            envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
            "count_memory_leak_warnings",
            msg,
        )

    def _check_all_pools(
        self, ps: PoolStats, uncached: int = 0
    ) -> Tuple[bool, List[str]]:
        """Check memory invariant across all pools. Returns (has_leak, messages).

        中译：对所有相关池（full，以及按需的 swa / mamba）逐一检查不变量，
              汇总「是否存在泄漏」与各池诊断文本列表。
        """
        has_leak = False
        messages = []

        full_leak, full_msg = self._check_full_pool(ps, uncached=uncached)
        has_leak |= full_leak
        messages.append(full_msg)

        if self.is_hybrid_swa:
            swa_leak, swa_msg = self._check_swa_pool(ps)
            has_leak |= swa_leak
            messages.append(swa_msg)

        if self.is_hybrid_ssm and self.tree_cache.supports_mamba():
            mamba_leak, mamba_msg = self._check_mamba_pool(ps)
            has_leak |= mamba_leak
            messages.append(mamba_msg)

        return has_leak, messages

    def _check_tree_cache(self):
        # 中译：对前缀缓存（树形缓存）做自洽性检查，仅在 SWA / mamba 等特殊缓存场景下触发。
        if (
            self.tree_cache.is_tree_cache()
            and (self.is_hybrid_swa and self.tree_cache.supports_swa())
            or (self.is_hybrid_ssm and self.tree_cache.supports_mamba())
        ):
            self.tree_cache.sanity_check()


def create_scheduler_watchdog(
    scheduler: Scheduler, watchdog_timeout: float, soft: bool = False
) -> WatchdogRaw:
    # 中译：为调度器创建看门狗（watchdog）。通过 forward_ct 计数判断是否在推进；
    #       一旦判定卡死，调用 dump_info 打印当前批次与各池状态，辅助定位死锁/泄漏。
    def dump_info() -> str:
        # 中译：卡死时的现场转储。初始化阶段直接返回空串（此时尚无有效批次/池状态）。
        if scheduler.is_initializing:
            return ""
        _, messages = scheduler.invariant_checker._check_all_pools(
            scheduler.pool_stats_observer.get_pool_stats(),
        )
        return (
            f"{scheduler.cur_batch.batch_size()=}\n"
            f"{scheduler.cur_batch.reqs=}\n" + "\n".join(messages)
        )

    return WatchdogRaw(
        debug_name="Scheduler",
        get_counter=lambda: scheduler.forward_ct,
        is_active=lambda: scheduler.is_initializing or scheduler.cur_batch is not None,
        watchdog_timeout=watchdog_timeout,
        soft=soft,
        dump_info=dump_info,
    )
