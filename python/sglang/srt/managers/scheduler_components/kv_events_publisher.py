"""中译：调度器侧的「KV cache 事件 / 指标」发布模块。

对外（如 PD 分离架构中的路由/调度组件、外部监控）发布两类信息：
1) KV cache 事件（block 的新增/淘汰/命中等），由前缀缓存树 tree_cache 收集，
   经 EventPublisher 批量发布，供全局做前缀感知的路由调度；
2) KvMetrics 运行指标（活跃/总槽位、活跃/总 KV block、等待请求数、缓存使用率与命中率等），
   通过 ZMQ 推送给上层做负载感知。

为避免重复上报，仅在「注意力的 attn_tp_rank 与 attn_cp_rank 均为 0」的进程上启用发布。
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
)

import zmq

from sglang.srt.disaggregation.kv_events import (
    EventPublisherFactory,
    KVEventBatch,
)

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache


# 中译：此处声明一个空的 SchedulerStats 占位类型（仅供类型标注/避免重定义），
#       真正的统计数据在运行时由 get_stats() 回调提供。
class SchedulerStats: ...  # type: ignore[no-redef]


@dataclasses.dataclass
class KvMetrics:
    """中译：一次上报的 KV cache 运行指标快照。各字段含义见下方逐行注释。"""

    request_active_slots: int = 0  # 当前正在运行的请求数（占用的槽位）
    request_total_slots: int = 0  # 可容纳的最大并发请求数（总槽位）
    kv_active_blocks: int = 0  # 当前已使用的 KV block 数
    kv_total_blocks: int = 0  # KV cache 的总 block 数
    num_requests_waiting: int = 0  # 等待队列中的请求数
    gpu_cache_usage_perc: float = 0.0  # KV cache（token）使用率
    gpu_prefix_cache_hit_rate: float = 0.0  # 前缀缓存命中率
    data_parallel_rank: int = 0  # 数据并行 rank（标识来源分片）


@dataclass(kw_only=True, slots=True)
class SchedulerKvEventsPublisher:
    """中译：KV cache 事件 / 指标发布器，被 Scheduler 组合使用。

    持有发布所需的并行 rank 信息、前缀缓存树、ZMQ 指标 socket、容量上限与取统计的回调。
    """

    kv_events_config: Optional[str]  # 事件发布的配置（为空则整体不启用）
    ps: ParallelState  # 并行状态（含各类 rank）
    attn_tp_rank: int
    attn_cp_rank: int
    attn_dp_rank: int
    dp_rank: Optional[int]
    tree_cache: BasePrefixCache  # 前缀缓存树，事件的来源
    send_metrics_from_scheduler: Optional[zmq.Socket]  # 推送 KvMetrics 的 ZMQ socket
    max_running_requests: int  # 最大并发请求数（总槽位）
    max_total_num_tokens: int  # KV cache 总 token 容量
    get_stats: Callable  # 取当前调度统计的回调
    enable_kv_cache_events: bool = False  # 是否在本进程启用发布
    kv_event_publisher: Any = None  # 事件发布器实例

    def __post_init__(self) -> None:
        self.init_kv_events(self.kv_events_config)

    def init_kv_events(self, kv_events_config: Optional[str]):
        # 中译：仅在配置非空、且 attn_tp_rank 与 attn_cp_rank 均为 0 时启用，避免多 rank 重复上报。
        self.enable_kv_cache_events = bool(
            kv_events_config and self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0
        )

        if self.enable_kv_cache_events:
            # 中译：按配置与 attn_dp_rank 创建对应的事件发布器（如 ZMQ/其他后端）。
            self.kv_event_publisher = EventPublisherFactory.create(
                kv_events_config, self.ps.attn_dp_rank
            )

    def emit_kv_metrics(self):
        """中译：采集一份当前 KvMetrics 快照并通过 ZMQ 推送出去。"""
        if not self.enable_kv_cache_events:
            return

        # 中译：从 get_stats() 取实时统计，填充各指标字段。
        kv_metrics = KvMetrics()
        kv_metrics.request_active_slots = self.get_stats().num_running_reqs.total
        kv_metrics.request_total_slots = self.max_running_requests
        # 中译：活跃 block 数 = 使用率 × 总容量（由比例还原成绝对数量）。
        kv_metrics.kv_active_blocks = int(
            self.get_stats().token_usage * self.max_total_num_tokens
        )
        kv_metrics.kv_total_blocks = self.max_total_num_tokens
        kv_metrics.num_requests_waiting = self.get_stats().num_queue_reqs.total
        kv_metrics.gpu_cache_usage_perc = self.get_stats().token_usage
        kv_metrics.gpu_prefix_cache_hit_rate = self.get_stats().cache_hit_rate
        kv_metrics.data_parallel_rank = (
            self.ps.dp_rank if self.ps.dp_rank is not None else 0
        )

        # 中译：socket 未关闭时才推送，避免在关停过程中向已关闭的 socket 发送。
        if not self.send_metrics_from_scheduler.closed:
            self.send_metrics_from_scheduler.send_pyobj(kv_metrics)

    def publish_kv_events(self):
        """中译：把前缀缓存树累积的 KV cache 事件取出并批量发布。"""
        if not self.enable_kv_cache_events:
            return

        # 中译：take_events 取走自上次以来累积的事件（取后清空）；有则打上时间戳成批发布。
        events = self.tree_cache.take_events()
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)
