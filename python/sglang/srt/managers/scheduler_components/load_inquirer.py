"""Load metrics inquiry for the scheduler.

中译：调度器的「负载查询」组件，为 /v1/loads 端点提供综合负载指标。
      汇总当前运行/等待请求数、待 prefill 的 token 数、KV 占用与使用率，
      以及（按需）内存、投机解码（speculative）、LoRA、PD 分离（disaggregation）、
      各队列长度等细分指标，供负载均衡/路由器（router）做调度决策。
      这些数据来自调度器内部的各种队列、批次与池统计观测器（通过构造时传入的回调读取）。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    DisaggregationMetrics,
    GetLoadsReqInput,
    GetLoadsReqOutput,
    LoRAMetrics,
    MemoryMetrics,
    QueueMetrics,
    SpeculativeMetrics,
)

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.managers.scheduler_components.pool_stats_observer import (
        SchedulerPoolStatsObserver,
    )
    from sglang.srt.managers.tp_worker import BaseTpWorker
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerLoadInquirer:
    # 中译：负载查询器。冻结数据类，通过一组 get_* 回调读取调度器内部状态（队列、批次、统计等），
    #       计算并打包成对外的负载指标。这样设计可避免与 Scheduler 形成强耦合/循环依赖。
    disaggregation_mode: DisaggregationMode
    ps: ParallelState
    server_args: ServerArgs
    max_total_num_tokens: int
    max_running_requests: int
    pool_stats_observer: SchedulerPoolStatsObserver
    tp_worker: BaseTpWorker
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    spec_algorithm: SpeculativeAlgorithm
    get_running_batch: Callable
    get_waiting_queue: Callable
    get_stats: Callable
    get_chunked_req: Callable
    get_disagg_prefill_bootstrap_queue: Callable
    get_disagg_prefill_inflight_queue: Callable
    get_disagg_decode_prealloc_queue: Callable
    get_disagg_decode_transfer_queue: Callable
    get_spec_total_num_accept_tokens: Callable
    get_spec_total_num_forward_ct: Callable

    def _get_num_pending_tokens(self, chunk_deduct: int = 0) -> int:
        """Get the total number of tokens pending prefill.

        This includes tokens from waiting queue requests plus remaining tokens
        from the currently chunked request.

        Args:
            chunk_deduct: extra tokens to subtract from the chunked request's
                remaining count. At batch-scheduling time the current chunk
                has been planned but ``prefix_indices`` does not yet include it,
                so callers pass ``extend_input_len`` here. At load-reporting
                time ``prefix_indices`` is already up-to-date, so the default
                0 is correct.

        中译：统计待 prefill 的 token 总数 = 等待队列中各请求的序列长度之和
              + 当前分块（chunked）请求剩余未处理的 token 数。
              chunk_deduct：从分块请求剩余数中额外扣除的 token。批次调度时当前 chunk 已规划但
              prefix_indices 尚未包含它，调用方传入 extend_input_len；负载上报时 prefix_indices
              已更新，故默认 0 即可。
        """
        num_pending_tokens = sum(req.seqlen for req in self.get_waiting_queue())
        if self.get_chunked_req() is not None:
            req = self.get_chunked_req()
            num_pending_tokens += req.seqlen - len(req.prefix_indices) - chunk_deduct
        return num_pending_tokens

    def get_num_waiting_uncached_tokens(self) -> int:
        """Get uncached input tokens waiting for prefill compute.

        中译：统计等待 prefill 计算的「未命中缓存」输入 token 数（即真正需要计算的部分）。
              对每个等待请求 = 序列长度 - 已匹配前缀缓存的长度，再对当前分块请求做同样处理。
              纯 DECODE 模式无 prefill 计算，直接返回 0。
        """
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            return 0
        num_tokens = 0
        for req in self.get_waiting_queue():
            # if match-in-waiting-queue disabled, this metric returns seq_lens
            # 中译：若禁用「等待队列内前缀匹配」，num_matched_prefix_tokens 为 0，此指标即退化为序列长度。
            num_tokens += max(0, req.seqlen - req.num_matched_prefix_tokens)
        cr = self.get_chunked_req()
        if cr is not None:
            num_tokens += max(0, cr.seqlen - len(cr.prefix_indices))
        return num_tokens

    def get_loads(self, req: GetLoadsReqInput = None) -> GetLoadsReqOutput:
        """
        Get comprehensive load metrics for /v1/loads endpoint.

        Args:
            req: Request containing include list and optional dp_rank filter

        Returns:
            GetLoadsReqOutput with core metrics and optional detailed sections

        中译：/v1/loads 端点的主入口，汇总综合负载指标。
              include 控制返回哪些细分段（默认仅 core；"all" 表示全部）；
              可选段包括 memory / spec / lora / disagg / queues。
        """
        if req is None:
            req = GetLoadsReqInput()

        # 中译：解析 include 集合；为空时默认只返回核心（core）指标。
        include = set(req.include) if req.include else {"core"}
        include_all = "all" in include

        num_running_reqs = len(self.get_running_batch().reqs)

        # 中译：汇总所有等待类队列。PD 分离模式下还需纳入各自的引导/预分配/传输/被退回队列。
        waiting_queues = [self.get_waiting_queue()]
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            waiting_queues.append(self.get_disagg_prefill_bootstrap_queue().queue)
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            waiting_queues.append(self.get_disagg_decode_prealloc_queue().queue)
            waiting_queues.append(self.get_disagg_decode_transfer_queue().queue)
            waiting_queues.append(
                self.get_disagg_decode_prealloc_queue().retracted_queue
            )

        num_waiting_reqs = sum(len(queue) for queue in waiting_queues)
        num_waiting_uncached_tokens = self.get_num_waiting_uncached_tokens()
        # 中译：从池统计观测器取「已用 token 数」与「KV 使用率」。
        num_used_tokens, kv_token_usage = (
            self.pool_stats_observer.get_pool_stats().get_kv_token_stats()
        )
        # 中译：总 token 数 ≈ 已用（运行中） + 所有等待请求的序列长度之和。
        num_total_tokens = num_used_tokens + sum(
            req.seqlen for queue in waiting_queues for req in queue
        )

        # 中译：内存细分段——权重/KV 缓存/CUDA Graph 显存占用与 token 容量。属性缺失时降级跳过。
        memory = None
        if include_all or "memory" in include:
            try:
                memory = MemoryMetrics(
                    weight_gb=round(
                        self.tp_worker.model_runner.weight_load_mem_usage, 3
                    ),
                    kv_cache_gb=round(
                        self.token_to_kv_pool_allocator.get_kvcache().mem_usage, 3
                    ),
                    graph_gb=round(self.tp_worker.model_runner.graph_mem_usage, 3),
                    token_capacity=int(self.max_total_num_tokens),
                )
            except AttributeError as e:
                logger.debug(f"Memory metrics not available: {e}")

        # 中译：投机解码（speculative）细分段——平均接受长度（接受 token 数 / 前向次数）与接受率。
        #       仅在启用了投机算法且已有前向计数时才填充。
        speculative = None
        if include_all or "spec" in include:
            if (
                not self.spec_algorithm.is_none()
                and self.get_spec_total_num_forward_ct() > 0
            ):
                speculative = SpeculativeMetrics(
                    accept_length=(
                        self.get_spec_total_num_accept_tokens()
                        / self.get_spec_total_num_forward_ct()
                    ),
                    accept_rate=self.get_stats().spec_accept_rate,
                )

        # 中译：LoRA 细分段——适配器槽位的已用/总数/利用率（仅启用 LoRA 时）。
        lora = None
        if include_all or "lora" in include:
            if self.server_args.enable_lora:
                lora = LoRAMetrics(
                    slots_used=self.get_stats().lora_pool_slots_used,
                    slots_total=self.get_stats().lora_pool_slots_total,
                    utilization=self.get_stats().lora_pool_utilization,
                )

        # 中译：PD 分离（disaggregation）细分段——按 prefill / decode 角色统计各专用队列长度，
        #       以及 KV 传输速度/时延等指标。
        disaggregation = None
        if include_all or "disagg" in include:
            mode_str = "null"
            prefill_bootstrap = 0
            prefill_inflight = 0
            decode_prealloc = 0
            decode_transfer = 0
            decode_retracted = 0

            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                mode_str = "prefill"
                prefill_bootstrap = len(self.get_disagg_prefill_bootstrap_queue().queue)
                prefill_inflight = len(self.get_disagg_prefill_inflight_queue())
            elif self.disaggregation_mode == DisaggregationMode.DECODE:
                mode_str = "decode"
                decode_prealloc = len(self.get_disagg_decode_prealloc_queue().queue)
                decode_transfer = len(self.get_disagg_decode_transfer_queue().queue)
                decode_retracted = len(
                    self.get_disagg_decode_prealloc_queue().retracted_queue
                )

            disaggregation = DisaggregationMetrics(
                mode=mode_str,
                prefill_bootstrap_queue_reqs=prefill_bootstrap,
                prefill_inflight_queue_reqs=prefill_inflight,
                decode_prealloc_queue_reqs=decode_prealloc,
                decode_transfer_queue_reqs=decode_transfer,
                decode_retracted_queue_reqs=decode_retracted,
                kv_transfer_speed_gb_s=self.get_stats().kv_transfer_speed_gb_s,
                kv_transfer_latency_ms=self.get_stats().kv_transfer_latency_ms,
            )

        # 中译：队列细分段——等待 / 语法约束(grammar) / 暂停 / 被退回(retracted) 各队列的请求数。
        queues = None
        if include_all or "queues" in include:
            queues = QueueMetrics(
                waiting=len(self.get_waiting_queue()),
                grammar=self.get_stats().num_grammar_queue_reqs,
                paused=self.get_stats().num_paused_reqs,
                retracted=self.get_stats().num_retracted_reqs,
            )

        # 中译：把核心指标与上述各可选细分段打包返回（未请求的段保持为 None）。
        return GetLoadsReqOutput(
            dp_rank=self.ps.dp_rank,
            timestamp=time.time(),
            num_running_reqs=num_running_reqs,
            num_waiting_reqs=num_waiting_reqs,
            num_waiting_uncached_tokens=num_waiting_uncached_tokens,
            num_used_tokens=num_used_tokens,
            num_total_tokens=num_total_tokens,
            max_total_num_tokens=self.max_total_num_tokens,
            token_usage=round(kv_token_usage, 4),
            gen_throughput=round(self.get_stats().gen_throughput, 2),
            cache_hit_rate=round(self.get_stats().cache_hit_rate, 4),
            utilization=round(self.get_stats().utilization, 4),
            max_running_requests=self.max_running_requests,
            memory=memory,
            speculative=speculative,
            lora=lora,
            disaggregation=disaggregation,
            queues=queues,
        )
