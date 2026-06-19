"""Metrics reporting helpers for the Scheduler.

中译：Scheduler（调度器）的指标/监控上报组件。
      核心类 SchedulerMetricsReporter 负责在 prefill / decode 的每一步采集并上报
      各类运行指标：吞吐（input/gen throughput）、延迟（gap latency）、KV/显存利用率
      （token_usage、utilization）、队列状态（等待队列、PD 分离的各级队列）、投机解码
      接受率（spec accept length/rate）、MFU（估算的 TFLOPS 与显存带宽）等。
      这些指标一方面以人类可读的日志行（logger.info）打印，另一方面通过
      SchedulerMetricsCollector 推送为 Prometheus 风格的时序指标。
      此外还包含 FPM（Forward Pass Metrics，逐次前向的细粒度指标，经 ZMQ PUB 发布）。
"""

from __future__ import annotations

import dataclasses
import logging
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    List,
    Optional,
    Tuple,
    Union,
)

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.observability.metrics_collector import (
    DPCooperationInfo,
    QueueCount,
    SchedulerMetricsCollector,
    SchedulerMetricsCollectorContext,
    SchedulerStats,
    compute_routing_key_stats,
)
from sglang.srt.utils.device_timer import DeviceTimer
from sglang.srt.utils.scheduler_status_logger import SchedulerStatusLogger

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.schedule_policy import PrefillAdder
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.utils import EmbeddingBatchResult


logger = logging.getLogger(__name__)


# 中译：以下三个常量在模块加载时一次性从环境变量读取（避免热路径反复取值）：
# RECORD_STEP_TIME：是否记录每个 batch size 对应的单步耗时（用于性能分析）。
RECORD_STEP_TIME = envs.SGLANG_RECORD_STEP_TIME.get()
# LOG_FORWARD_ITERS：日志行中是否带上前向迭代序号 [forward_iter]，便于对齐前向次数。
LOG_FORWARD_ITERS = envs.SGLANG_LOG_FORWARD_ITERS.get()
# ENABLE_METRICS_DEVICE_TIMER：是否启用设备计时器（DeviceTimer）测量 GPU 前向占用率（fwd occupancy）。
ENABLE_METRICS_DEVICE_TIMER = envs.SGLANG_ENABLE_METRICS_DEVICE_TIMER.get()


def _decode_total_seq_lens(batch: ScheduleBatch) -> int:
    """Sync-free sum of seq_lens for decode metrics.

    中译：为 decode 指标统计「批次内所有序列长度之和」（即 KV 上下文总 token 数）。
          「Sync-free」指优先用已在 CPU 上的 seq_lens_cpu 求和，避免触发 GPU→CPU 同步拖慢调度。
    """
    if batch.seq_lens_cpu is not None:
        return int(batch.seq_lens_cpu.sum().item())
    # 中译：没有 CPU 缓存时退化为逐请求累加 seqlen（可能需要同步，较慢）。
    return sum(req.seqlen for req in batch.reqs)


@dataclasses.dataclass
class PrefillStats:
    """Stats for logging prefill batch metrics.

    中译：用于记录/上报一次 prefill 批次指标的数据载体。各字段含义：
    - log_input_tokens：本次 prefill 真正需要计算的新增 token 数（不含命中缓存的）。
    - log_hit_tokens：命中前缀缓存（prefix cache）而无需重算的 token 数。
    - new_token_ratio：新 token 比例，调度器据此预估显存压力的动态参数。
    - num_running_reqs：当前正在运行的请求数（QueueCount，可区分优先级桶）。
    - num_new_seqs：本批次新加入运行的序列数（= len(can_run_list)）。
    - reprocessed_log_input_tokens / reprocessed_log_hit_tokens：因被回退（retract）
      而重新处理的 token 数，计算真实缓存命中率时需从总量中扣除。
    - num_pending_tokens：仍在排队、尚未处理的 token 数。
    """

    log_input_tokens: int
    log_hit_tokens: int
    new_token_ratio: float
    num_running_reqs: QueueCount
    num_new_seqs: int  # len(can_run_list)
    reprocessed_log_input_tokens: int = 0
    reprocessed_log_hit_tokens: int = 0
    num_pending_tokens: int = 0

    @classmethod
    def from_adder(
        cls,
        adder: PrefillAdder,
        running_reqs: List[Req],
        enable_priority_scheduling: bool = False,
        num_pending_tokens: int = 0,
    ):
        # 中译：工厂方法——从 PrefillAdder（负责把等待队列里的请求装入本次 prefill 批次的组件）
        #       和当前运行请求列表中提取各项统计，构造 PrefillStats。
        return cls(
            log_input_tokens=adder.log_input_tokens,
            log_hit_tokens=adder.log_hit_tokens,
            reprocessed_log_input_tokens=adder.reprocessed_log_input_tokens,
            reprocessed_log_hit_tokens=adder.reprocessed_log_hit_tokens,
            new_token_ratio=adder.new_token_ratio,
            num_running_reqs=QueueCount.from_reqs(
                running_reqs, enable_priority_scheduling
            ),
            num_new_seqs=len(adder.can_run_list),
            num_pending_tokens=num_pending_tokens,
        )


@dataclass(kw_only=True)
class SchedulerMetricsReporter:
    """中译：调度器指标上报器。挂在 Scheduler 上，负责把 prefill/decode 过程中的运行状态
    汇集成日志与 Prometheus 指标。各字段：
    - scheduler：回指宿主调度器，用于读取其运行时状态（队列、内存池、worker 等）。
    - tp_rank / pp_rank / dp_rank：当前进程在张量并行 / 流水并行 / 数据并行中的 rank。
    - metrics_collector_context：指标采集的总开关与「是否本 rank 负责打日志」等上下文。
    - metrics_collector：真正把指标推给 Prometheus 的采集器（未启用指标时为 None）。
    - num_retracted_reqs / num_paused_reqs：自上次上报以来被回退 / 暂停的请求数累加器，
      上报后清零（见 report_*_stats）。
    """

    scheduler: Scheduler
    tp_rank: int
    pp_rank: int
    dp_rank: Optional[int]
    metrics_collector_context: SchedulerMetricsCollectorContext
    metrics_collector: Optional[SchedulerMetricsCollector]
    num_retracted_reqs: int = 0
    num_paused_reqs: int = 0

    def __post_init__(self) -> None:
        # 中译：dataclass 构造后回调。从采集上下文展开几个常用开关，再初始化指标状态与设备计时器。
        # enable_metrics：总开关，是否启用指标采集。
        # is_stats_logging_rank：本 rank 是否负责打印人类可读的统计日志。
        # current_scheduler_metrics_enabled：当前是否应推送 Prometheus 指标（随阶段动态变化）。
        # enable_kv_cache_events：是否发布 KV cache 相关事件。
        self.enable_metrics = self.metrics_collector_context.enable_metrics
        self.is_stats_logging_rank = (
            self.metrics_collector_context.is_stats_logging_rank
        )
        self.current_scheduler_metrics_enabled = (
            self.metrics_collector_context.current_scheduler_metrics_enabled
        )
        self.enable_kv_cache_events = (
            self.metrics_collector_context.enable_kv_cache_events
        )
        self._init_metrics(self.tp_rank, self.pp_rank, self.dp_rank)
        self._install_device_timer_on_runners()

    def _init_metrics(
        self,
        tp_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
    ):
        # Basic stats
        # 中译：基础统计量初始化。
        # forward_ct_decode：decode 前向计数器，用于按 decode_log_interval 周期性触发重指标。
        self.forward_ct_decode = 0
        # num_generated_tokens：自上次 decode 上报以来生成的 token 数（算吞吐用，上报后清零）。
        self.num_generated_tokens = 0
        # last_*_stats_tic：上次 decode / prefill 上报的时间戳，用于算两次上报之间的 gap latency。
        self.last_decode_stats_tic = time.perf_counter()
        self.last_prefill_stats_tic = time.perf_counter()
        # last_gen_throughput / last_input_throughput：最近一次算出的生成 / 输入吞吐（token/s）。
        self.last_gen_throughput: float = 0.0
        self.last_input_throughput: float = 0.0
        self.step_time_dict = defaultdict(list)  # Dict[batch size -> step time]
        # 中译：stats 是聚合所有上报字段的容器，每次上报前填充、再交给 collector 推送。
        self.stats = SchedulerStats()
        # 中译：根据设备类型选择日志里 graph backend 的显示标签（默认 cuda graph）。
        self._graph_backend_label = {
            "cpu": "cpu graph",
            "npu": "npu graph",
            "musa": "musa graph",
        }.get(getattr(self.scheduler, "device", ""), "cuda graph")

        # Cumulative spec-decoding counters (reset every decode_log_interval).
        # Each update adds (num_correct_drafts + bs, bs).
        # `*_accept_tokens` = drafts + bonus; `*_correct_drafts` = drafts-only.
        # 中译：投机解码（speculative decoding）累计计数器，每个 decode_log_interval 周期清零。
        #       每次更新累加 (num_correct_drafts + bs, bs)：
        #       *_accept_tokens = 被接受的草稿 token + bonus（必采的额外 token）；
        #       *_correct_drafts = 仅指被验证通过的草稿 token。
        #       带 total_ 前缀的是「全生命周期」累计（不随周期清零）。
        self.spec_num_accept_tokens = 0  # per-log-interval
        self.spec_num_forward_ct = 0
        self.spec_total_num_accept_tokens = 0  # lifetime
        self.spec_total_num_forward_ct = 0

        # For PD disaggregation
        # 中译：PD（prefill/decode）分离部署时的 KV 传输指标——
        #       传输速度（GB/s）与传输延迟（ms），由 prefill 端把 KV 搬到 decode 端时统计。
        self.kv_transfer_speed_gb_s: float = 0.0
        self.kv_transfer_latency_ms: float = 0.0

        # 中译：MFU（Model FLOPs Utilization）相关指标默认关闭，启用指标且开关打开时才初始化常量。
        self.enable_mfu_metrics = False

        if self.enable_metrics:
            self.enable_mfu_metrics = self.scheduler.server_args.enable_mfu_metrics
            if self.enable_mfu_metrics:
                self._init_estimated_perf_constants()
                # 中译：MFU 在一个 decode_log_interval 内累加的 FLOPs / 读字节 / 写字节，上报后清零。
                self._mfu_log_flops = 0.0
                self._mfu_log_read_bytes = 0.0
                self._mfu_log_write_bytes = 0.0

        # 中译：前向占用率（GPU 实际计算时间 / 墙钟时间），未测到时为 NaN。
        self.fwd_occupancy = float("nan")

        self.forward_pass_device_timer: Optional[DeviceTimer] = None

        if ENABLE_METRICS_DEVICE_TIMER:
            # 中译：维护一个滑动窗口来统计 GPU 占用率：累计窗口内 batch 数、GPU 时间与窗口起点。
            self._device_timer_window_batch_count = 0
            self._device_timer_window_gpu_time = 0.0
            self._device_timer_window_start = None

            def _wrap_execution_reporter(**kwargs):
                # 中译：DeviceTimer 每测得一段 GPU 执行时间 t 就回调此函数：
                #       既累加到本窗口 GPU 时间，又（启用指标时）累计上报「前向执行秒数」指标。
                self._device_timer_window_gpu_time += kwargs["t"]
                if self.enable_metrics:
                    self.metrics_collector.increment_forward_execution_seconds(**kwargs)

            self.forward_pass_device_timer = DeviceTimer(
                reporter=_wrap_execution_reporter,
            )

        # 中译：初始化 FPM（逐次前向指标）发布器（满足条件时）。
        self._init_fpm()

        # 中译：调度器状态日志器（可选），周期性 dump 批次与等待队列的快照便于排障。

        self.scheduler_status_logger = SchedulerStatusLogger.maybe_create(
            enable_metrics=self.enable_metrics
        )

    def _install_device_timer_on_runners(self):
        # 中译：把同一个 DeviceTimer 实例安装到所有 model runner 上，使其前向计时统一汇集。
        #       未启用设备计时器时直接返回。
        if self.forward_pass_device_timer is None:
            return
        timer = self.forward_pass_device_timer
        # 中译：主模型 runner。
        self.scheduler.tp_worker.model_runner.device_timer = timer
        # 中译：若启用投机解码，还要把计时器装到草稿模型（draft）的 runner 上（可能有多个）。
        if self.scheduler.draft_worker is not None:
            dw = getattr(self.scheduler.draft_worker, "draft_worker", None)
            if dw is not None:
                if hasattr(dw, "draft_runner"):
                    dw.draft_runner.device_timer = timer
                for r in getattr(dw, "draft_runner_list", []):
                    r.device_timer = timer

    def _init_fpm(self):
        """Initialize Forward Pass Metrics (FPM) publisher if configured.

        中译：按需初始化 FPM（Forward Pass Metrics，逐次前向的细粒度指标）发布器。
              仅在「启用了 FPM、且本进程是 attention TP 的 rank0、且处于流水并行最后一级」时
              才发布——这样每个 DP 副本只由一个进程发布、避免重复。通过 ZMQ PUB 端点对外推送。
        """
        self.scheduler.enable_fpm = False
        if (
            self.scheduler.server_args.enable_forward_pass_metrics
            and self.scheduler.ps.attn_tp_rank == 0
            and self.scheduler.ps.pp_rank == self.scheduler.ps.pp_size - 1
        ):
            from sglang.srt.observability.forward_pass_metrics import (
                _FpmPublisherThread,
            )

            self.scheduler._fpm_dp_rank = (
                self.scheduler.ps.dp_rank
                if self.scheduler.ps.dp_rank is not None
                else 0
            )
            self.scheduler._fpm_worker_id = (
                self.scheduler.server_args.forward_pass_metrics_worker_id
            )
            base_endpoint = self.scheduler.server_args.forward_pass_metrics_ipc_name
            # 中译：未显式指定 IPC 端点时，用一个临时文件名生成 ipc:// 端点并回写到 server_args。
            if base_endpoint is None:
                ipc_path = tempfile.NamedTemporaryFile(delete=False).name
                base_endpoint = f"ipc://{ipc_path}"
                self.scheduler.server_args.forward_pass_metrics_ipc_name = base_endpoint
            # 中译：每个 DP rank 用独立后缀的端点，互不干扰。
            endpoint = f"{base_endpoint}.{self.scheduler._fpm_dp_rank}"
            self.scheduler._fpm_publisher = _FpmPublisherThread(
                endpoint,
                worker_id=self.scheduler._fpm_worker_id,
                dp_rank=self.scheduler._fpm_dp_rank,
            )
            # 中译：累计 GPU 时间，用作每次前向的精确 wall_time（见 _emit_forward_pass_metrics）。
            self.scheduler._fpm_gpu_time_acc = 0.0

            def _fpm_device_timer_reporter(t, **_kwargs):
                # 中译：DeviceTimer 回调，把每段 GPU 时间累加到 FPM 的累计器。
                self.scheduler._fpm_gpu_time_acc += t

            # 中译：若已有 DeviceTimer（占用率统计用）则复用并追加 reporter，否则新建一个。
            if self.forward_pass_device_timer is not None:
                self.forward_pass_device_timer.add_reporter(_fpm_device_timer_reporter)
            else:
                self.forward_pass_device_timer = DeviceTimer(
                    reporter=_fpm_device_timer_reporter,
                )
            # 中译：标记 FPM 使用的是精确 GPU 计时（否则回退到 monotonic 墙钟）。
            self.scheduler._fpm_uses_device_timer = True
            self.scheduler.enable_fpm = True
            logger.info(
                "FPM: ZMQ PUB bound on %s (dp_rank=%d, device_timer=%s)",
                endpoint,
                self.scheduler._fpm_dp_rank,
                self.scheduler._fpm_uses_device_timer,
            )

    def _build_scheduled_request_metrics(self, batch: ScheduleBatch):
        # 中译：为 FPM 构造「本次被调度执行的请求」的统计（区分 prefill / decode 两类）。
        #       WelfordAccumulator 是在线方差累加器，可一遍扫描得到均值/方差，避免存全部样本。
        from sglang.srt.observability.forward_pass_metrics import (
            ScheduledRequestMetrics,
            WelfordAccumulator,
        )

        num_prefill_requests = 0
        sum_prefill_tokens = 0
        sum_prefill_kv_tokens = 0
        prefill_lengths = WelfordAccumulator()

        # 中译：mixed 模式下一个批次同时含 prefill 与 decode，需用 decoding_reqs 把 decode 请求剔除，
        #       剩下的才是 prefill 请求；纯 extend 模式则整批都是 prefill；纯 decode 则没有 prefill。
        if batch.forward_mode.is_mixed():
            decode_req_ids = {id(req) for req in batch.decoding_reqs or []}
            prefill_reqs = [req for req in batch.reqs if id(req) not in decode_req_ids]
        elif batch.forward_mode.is_extend():
            prefill_reqs = batch.reqs
        else:
            prefill_reqs = []

        if prefill_reqs:
            stats = batch.prefill_stats
            for req in prefill_reqs:
                prefill_lengths.add(len(req.origin_input_ids))
            num_prefill_requests = stats.num_new_seqs if stats else len(prefill_reqs)
            sum_prefill_tokens = stats.log_input_tokens if stats else 0
            # 中译：prefill 的 KV token 数 = 命中前缀缓存的部分（prefix_indices 的长度之和）。
            sum_prefill_kv_tokens = sum(len(req.prefix_indices) for req in prefill_reqs)

        # 中译：统计 decode 请求的 KV 上下文长度分布（数量/总和/方差）。
        decode_kv = WelfordAccumulator()
        if batch.forward_mode.is_mixed():
            for req in batch.decoding_reqs or []:
                decode_kv.add(req.seqlen)
        elif batch.forward_mode.is_decode():
            for sl in batch.seq_lens_cpu:
                decode_kv.add(int(sl))

        return ScheduledRequestMetrics(
            num_prefill_requests=num_prefill_requests,
            sum_prefill_tokens=sum_prefill_tokens,
            var_prefill_length=prefill_lengths.variance(),
            sum_prefill_kv_tokens=sum_prefill_kv_tokens,
            num_decode_requests=decode_kv.count,
            sum_decode_kv_tokens=decode_kv.total,
            var_decode_kv_tokens=decode_kv.variance(),
        )

    def _build_queued_request_metrics(self):
        # 中译：为 FPM 构造「仍在排队等待」的请求统计。按部署模式从不同队列取数：
        #       PREFILL 节点看 bootstrap 队列；DECODE 节点看 prealloc/transfer 队列；
        #       非分离（normal）模式则看统一的 waiting_queue（已产出过 token 的算 decode，否则算 prefill）。
        from sglang.srt.observability.forward_pass_metrics import (
            QueuedRequestMetrics,
            WelfordAccumulator,
        )

        prefill_q = WelfordAccumulator()
        decode_q = WelfordAccumulator()
        if self.scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
            for req in self.scheduler.disagg_prefill_bootstrap_queue.queue:
                prefill_q.add(len(req.origin_input_ids))
        elif self.scheduler.disaggregation_mode == DisaggregationMode.DECODE:
            for req in self.scheduler.disagg_decode_prealloc_queue.queue:
                decode_q.add(req.seqlen)
            for req in self.scheduler.disagg_decode_transfer_queue.queue:
                decode_q.add(req.seqlen)
        else:
            for req in self.scheduler.waiting_queue:
                if len(req.output_ids) > 0:
                    decode_q.add(req.seqlen)
                else:
                    prefill_q.add(len(req.origin_input_ids))

        return QueuedRequestMetrics(
            num_prefill_requests=prefill_q.count,
            sum_prefill_tokens=prefill_q.total,
            var_prefill_length=prefill_q.variance(),
            num_decode_requests=decode_q.count,
            sum_decode_kv_tokens=decode_q.total,
            var_decode_kv_tokens=decode_q.variance(),
        )

    def _active_spec_config_snapshot(self) -> dict[str, int]:
        """Read the currently active speculative decoding configuration.

        中译：读取当前生效的投机解码配置：每轮草稿步数（num_steps）与草稿 token 数
              （num_draft_tokens）。优先取 draft_worker 上的实时值，缺失时回退到 server_args。
        """
        draft_worker = self.scheduler.draft_worker
        if draft_worker is None:
            # 中译：未启用投机解码，配置全为 0。
            return {
                "num_steps": 0,
                "num_draft_tokens": 0,
            }

        # Fallback to server_args if draft_worker does not have the attributes.
        server_args = self.scheduler.server_args
        num_steps = getattr(
            draft_worker, "speculative_num_steps", server_args.speculative_num_steps
        )
        num_draft_tokens = getattr(
            draft_worker,
            "speculative_num_draft_tokens",
            server_args.speculative_num_draft_tokens,
        )

        return {
            "num_steps": num_steps or 0,
            "num_draft_tokens": num_draft_tokens or 0,
        }

    def update_spec_metrics(self, bs: int, num_correct_drafts: int):
        # 中译：每次投机验证后更新计数。接受 token = 正确草稿数 + bs（bs 即 bonus，每序列必采的额外 token）；
        #       forward_ct 累加 bs（一次前向处理 bs 条序列）。
        self.spec_num_accept_tokens += num_correct_drafts + bs
        self.spec_num_forward_ct += bs

        # Bonus tokens updated elsewhere
        # 中译：bonus token 的计数在别处更新；这里只把「正确草稿数」计入已生成 token。
        self.num_generated_tokens += num_correct_drafts

    def _init_estimated_perf_constants(self) -> None:
        # 中译：预计算 MFU 估算所需的「每 token 常量」（FLOPs、读/写字节数等），
        #       这些只依赖模型结构与 dtype，构造时算一次即可，后续 prefill/decode 直接乘以 token 数。
        model_config = self.scheduler.model_config
        hf_text_config = model_config.hf_text_config

        hidden_size = float(model_config.hidden_size)
        num_layers = float(getattr(model_config, "num_attention_layers", 0))
        head_dim = float(getattr(model_config, "head_dim", 0))
        num_attn_heads = float(
            model_config.get_num_attention_heads(self.scheduler.ps.tp_size)
        )
        num_kv_heads = float(model_config.get_num_kv_heads(self.scheduler.ps.tp_size))
        intermediate_size = getattr(hf_text_config, "intermediate_size", None)
        if intermediate_size is None:
            intermediate_size = getattr(hf_text_config, "ffn_hidden_size", 0)
        intermediate_size = float(intermediate_size)

        # 中译：单个元素的字节数（如 fp16/bf16 为 2），取不到时默认按 2 字节估。
        dtype_num_bytes = getattr(model_config.dtype, "itemsize", None)
        if dtype_num_bytes is None:
            dtype_num_bytes = 2
        # Keep this estimator lightweight and consistent with current server dtype.
        # KV cache quantization-aware bytes can be added in a follow-up.
        # 中译：保持估算器轻量，统一用当前 dtype 字节数近似激活/权重/KV cache 的元素大小；
        #       KV cache 量化感知的字节数留待后续完善。
        act_bytes = float(dtype_num_bytes)
        w_bytes = float(dtype_num_bytes)
        cache_bytes = float(dtype_num_bytes)

        # Linear-layer FLOPs per token on one GPU.
        # 中译：单 GPU 上每 token 的线性层 FLOPs：注意力的 QKVO 投影 + MLP（约 6*h*ffn）乘以层数。
        attn_linear_flops = (
            2.0 * hidden_size * head_dim * (num_attn_heads + 2.0 * num_kv_heads)
            + 2.0 * hidden_size * head_dim * num_attn_heads
        )
        mlp_flops = (
            6.0 * hidden_size * intermediate_size if intermediate_size > 0 else 0.0
        )
        self._linear_flops_per_token = max(
            0.0, (attn_linear_flops + mlp_flops) * num_layers
        )

        # Attention dot-product FLOPs coefficient to multiply token-context product.
        # attn_qk + attn_av = 4 * q * TC * d * L
        # 中译：注意力点积（QK^T 与 AV）的 FLOPs 系数，乘以「token×上下文长度」乘积得到注意力 FLOPs。
        #       系数 4 = QK 与 AV 各 2 次乘加，q 为 head 数、d 为 head_dim、L 为层数。
        self._attn_dot_flops_coeff = 4.0 * num_attn_heads * head_dim * num_layers

        # KV cache bytes (write one K and one V vector per generated token).
        # 中译：每生成 1 个 token 写入 KV cache 的字节数（每层各写 1 个 K、1 个 V 向量，故系数为 2）。
        self._kv_cache_bytes_per_token = (
            2.0 * num_layers * num_kv_heads * head_dim * cache_bytes
        )

        # Weight read bytes per token.
        # 中译：每 token 需从显存读取的权重字节数（注意力投影权重 + MLP 权重，乘以层数）。
        self._weight_read_bytes_per_token = (
            hidden_size
            * head_dim
            * (num_attn_heads + 2.0 * num_kv_heads)
            * w_bytes
            * num_layers
            + hidden_size * head_dim * num_attn_heads * w_bytes * num_layers
            + (
                3.0 * hidden_size * intermediate_size * w_bytes * num_layers
                if intermediate_size > 0
                else 0.0
            )
        )

        # Activation movement bytes per token (coarse approximation).
        # 中译：每 token 激活值搬运的字节数（粗略近似，含 QKV 输入/输出激活等）。
        self._qkv_act_bytes_per_token = (
            hidden_size * act_bytes * num_layers
            + (num_attn_heads + 2.0 * num_kv_heads) * head_dim * act_bytes * num_layers
            + head_dim * num_attn_heads * act_bytes * num_layers
            + hidden_size * act_bytes * num_layers
        )
        self._ffn_act_bytes_per_token = (
            3.0 * intermediate_size * act_bytes * num_layers
            if intermediate_size > 0
            else 0.0
        )

        # Prefill reads Q/K/V activations from on-device memory.
        # 中译：prefill 阶段从显存读取 Q/K/V 激活的每 token 字节数。
        self._prefill_attn_act_read_per_token = (
            (num_attn_heads + 2.0 * num_kv_heads) * head_dim * act_bytes * num_layers
        )

        # Decode reads Q from activation memory; K/V reads are from KV cache.
        # 中译：decode 阶段只从激活读 Q（K/V 来自 KV cache，单独按上下文长度计入读字节）。
        self._decode_q_read_bytes_per_token = (
            num_attn_heads * head_dim * act_bytes * num_layers
        )

    def _estimate_prefill_perf(self, num_tokens: int) -> Tuple[float, float, float]:
        # 中译：估算一次 prefill 的 (FLOPs, 读字节, 写字节)。
        tokens = max(0, int(num_tokens))
        if tokens == 0:
            return 0.0, 0.0, 0.0

        # Causal prefill token-context product.
        # 中译：因果注意力下每个位置只能看前文，token×上下文乘积为等差求和 = n(n+1)/2。
        context_product = tokens * (tokens + 1) / 2.0
        flops = (
            tokens * self._linear_flops_per_token
            + self._attn_dot_flops_coeff * context_product
        )

        read_bytes = (
            tokens * self._weight_read_bytes_per_token
            + tokens * self._qkv_act_bytes_per_token
            + tokens * self._prefill_attn_act_read_per_token
        )
        write_bytes = (
            tokens * self._kv_cache_bytes_per_token
            + tokens * self._qkv_act_bytes_per_token
            + tokens * self._ffn_act_bytes_per_token
        )
        return flops, read_bytes, write_bytes

    def _estimate_decode_perf(
        self, batch: ScheduleBatch, num_tokens: int
    ) -> Tuple[float, float, float]:
        # 中译：估算一次 decode 的 (FLOPs, 读字节, 写字节)。
        tokens = max(0, int(num_tokens))
        if tokens == 0:
            return 0.0, 0.0, 0.0

        # 中译：decode 每个 token 都要对全部历史上下文做注意力，故按所有序列长度之和计。
        total_context = float(_decode_total_seq_lens(batch))
        flops = (
            tokens * self._linear_flops_per_token
            + self._attn_dot_flops_coeff * total_context
        )
        read_bytes = (
            tokens * self._weight_read_bytes_per_token
            + tokens * self._qkv_act_bytes_per_token
            + tokens * self._decode_q_read_bytes_per_token
            + total_context * self._kv_cache_bytes_per_token
        )
        write_bytes = (
            tokens * self._kv_cache_bytes_per_token
            + tokens * self._qkv_act_bytes_per_token
            + tokens * self._ffn_act_bytes_per_token
        )
        return flops, read_bytes, write_bytes

    def reset_metrics(self):
        # 中译：重置全部计数器（含生命周期累计的投机计数），通常在测试或显式重置场景调用。
        self.forward_ct_decode = 0
        self.num_generated_tokens = 0
        self.spec_num_accept_tokens = 0
        self.spec_num_forward_ct = 0
        self.spec_total_num_accept_tokens = 0
        self.spec_total_num_forward_ct = 0

    def report_prefill_stats(
        self,
        batch: Optional[ScheduleBatch],
        prefill_stats: PrefillStats,
        can_run_cuda_graph: bool,
        dp_cooperation_info: Optional[DPCooperationInfo] = None,
    ):
        # 中译：上报一次 prefill 批次的统计。既负责打印日志，也负责填充并推送 Prometheus 指标。
        # 中译：若本 rank 既不打日志、也不推指标，则无事可做，直接返回。
        if (
            not self.is_stats_logging_rank
            and not self.current_scheduler_metrics_enabled
        ):
            return

        # 中译：算与上次 prefill 上报的时间间隔（gap latency），据此推算输入吞吐（token/s）。
        now = time.perf_counter()
        gap_latency = now - self.last_prefill_stats_tic
        self.last_prefill_stats_tic = now
        self.last_input_throughput = (
            prefill_stats.log_input_tokens / gap_latency if gap_latency > 0 else 0.0
        )

        # 中译：取内存池快照，拼出 token 使用率的日志片段（KV/显存占用等）。
        pool_stats = self.scheduler.pool_stats_observer.get_pool_stats()
        token_usage_msg = ", ".join(pool_stats.get_prefill_usage_msg_parts()) + ", "

        self.stats.new_token_ratio = prefill_stats.new_token_ratio
        # 中译：日志里可选带上前向迭代号，便于和前向次数对齐排查。
        batch_iter = (
            batch.forward_iter
            if batch is not None and batch.forward_iter is not None
            else self.scheduler.forward_ct
        )
        iter_msg = f" [{batch_iter}]" if LOG_FORWARD_ITERS else ""

        # 中译：拼装人类可读的 prefill 日志行。各字段：新序列数、新算 token、命中缓存 token、
        #       内存使用、运行中请求数、等待队列长度、待处理 token 数。
        msg = (
            f"Prefill batch{iter_msg}, "
            f"#new-seq: {prefill_stats.num_new_seqs}, "
            f"#new-token: {prefill_stats.log_input_tokens}, "
            f"#cached-token: {prefill_stats.log_hit_tokens}, "
            f"{token_usage_msg}"
            f"#running-req: {prefill_stats.num_running_reqs.total}, "
            f"#queue-req: {len(self.scheduler.waiting_queue)}, "
            f"#pending-token: {prefill_stats.num_pending_tokens}, "
        )

        # 中译：PD 分离的 prefill 节点额外打印 bootstrap（握手中）与 inflight（KV 传输中）队列长度。
        if self.scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
            msg += f"#bootstrap-req: {len(self.scheduler.disagg_prefill_bootstrap_queue.queue)}, "
            msg += (
                f"#inflight-req: {len(self.scheduler.disagg_prefill_inflight_queue)}, "
            )

        if (
            self.scheduler.server_args.language_only
            and self.scheduler.server_args.encoder_transfer_backend
            == "zmq_to_scheduler"
        ):
            msg += (
                f"waiting-image-req: {len(self.scheduler.mm_receiver.waiting_list)}, "
            )

        # 中译：是否能跑 CUDA/设备图、以及输入吞吐。
        msg += f"{self._graph_backend_label}: {can_run_cuda_graph}, "
        msg += f"input throughput (token/s): {self.last_input_throughput:.2f}"

        # 中译：启用 MFU 时附加估算的 prefill 算力（TFLOPS/s，单 GPU）。
        if self.enable_mfu_metrics and gap_latency > 0:
            flops, _, _ = self._estimate_prefill_perf(prefill_stats.log_input_tokens)
            tflops_per_s = flops / gap_latency / 1e12
            msg += f", est. prefill TFLOPS/s (per GPU): {tflops_per_s:.2f}"

        # 中译：启用设备计时器时附加前向 GPU 占用率。
        if ENABLE_METRICS_DEVICE_TIMER:
            msg += f", fwd occupancy: {self.fwd_occupancy:.2f}%"

        # 中译：负责打日志的 rank 打印上面拼好的 msg。
        if self.is_stats_logging_rank:
            logger.info(msg)
        # 中译：需要推 Prometheus 指标时，逐项填充 self.stats 并上报。
        if self.current_scheduler_metrics_enabled:
            self.metrics_collector.increment_prefill_cuda_graph_pass(
                value=can_run_cuda_graph
            )
            self.metrics_collector.increment_realtime_tokens(
                prefill_compute_tokens=prefill_stats.log_input_tokens,
                prefill_cache_tokens=prefill_stats.log_hit_tokens,
                dp_cooperation_info=dp_cooperation_info,
            )
            if self.enable_mfu_metrics:
                flops, read_bytes, write_bytes = self._estimate_prefill_perf(
                    prefill_stats.log_input_tokens
                )
                self.metrics_collector.increment_estimated_perf(
                    num_flops_per_gpu=flops,
                    num_read_bytes_per_gpu=read_bytes,
                    num_write_bytes_per_gpu=write_bytes,
                )

            priority_enabled = self.scheduler.enable_priority_scheduling
            # 中译：计算真实缓存命中率时，需把「被回退后重处理」的 token 从分子分母里扣除，
            #       否则重算会虚增 token 数、扭曲命中率。
            effective_input_tokens = (
                prefill_stats.log_input_tokens
                - prefill_stats.reprocessed_log_input_tokens
            )
            effective_hit_tokens = (
                prefill_stats.log_hit_tokens - prefill_stats.reprocessed_log_hit_tokens
            )
            total_tokens = effective_input_tokens + effective_hit_tokens
            cache_hit_rate = (
                effective_hit_tokens / total_tokens if total_tokens > 0 else 0.0
            )

            # Basics
            # 中译：基础指标——运行中请求数、等待队列长度、语法约束队列长度、缓存命中率。
            self.stats.num_running_reqs = prefill_stats.num_running_reqs
            self.stats.num_queue_reqs = QueueCount.from_reqs(
                self.scheduler.waiting_queue, priority_enabled
            )
            self.stats.num_grammar_queue_reqs = len(self.scheduler.grammar_manager)
            self.stats.cache_hit_rate = cache_hit_rate

            # Memory pool usage ratios / Absolute token counts
            # 中译：把内存池使用率与绝对 token 数填入 stats（token_usage 等）。
            pool_stats.update_scheduler_stats(self.stats)

            # Retract
            # 中译：上报自上次以来被回退/暂停的请求数，随后清零累加器。
            self.stats.num_retracted_reqs = self.num_retracted_reqs
            self.stats.num_paused_reqs = self.num_paused_reqs
            self.num_retracted_reqs = self.num_paused_reqs = 0

            # PD disaggregation
            # 中译：PD 分离时记录各级队列长度；prefill 端还记录 KV 传输速度/延迟。
            if self.scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_bootstrap_queue_reqs = QueueCount.from_reqs(
                    self.scheduler.disagg_prefill_bootstrap_queue.queue,
                    priority_enabled,
                )
                self.stats.num_prefill_inflight_queue_reqs = QueueCount.from_reqs(
                    self.scheduler.disagg_prefill_inflight_queue, priority_enabled
                )
                self.stats.kv_transfer_speed_gb_s = self.kv_transfer_speed_gb_s
                self.stats.kv_transfer_latency_ms = self.kv_transfer_latency_ms
            elif self.scheduler.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = QueueCount.from_reqs(
                    self.scheduler.disagg_decode_prealloc_queue.queue, priority_enabled
                )
                self.stats.num_decode_transfer_queue_reqs = QueueCount.from_reqs(
                    self.scheduler.disagg_decode_transfer_queue.queue, priority_enabled
                )

            # Utilization / LoRA / HiCache
            # 中译：补充利用率、前向占用率、LoRA 池占用、分层缓存（HiCache）主机层统计，最后统一上报。
            self._calculate_utilization()
            self.stats.fwd_occupancy = self.fwd_occupancy
            self._update_lora_metrics()
            self._log_hicache_stats()
            self.metrics_collector.log_stats(self.stats)
            # 中译：发布 KV cache 相关指标（聚合值）。
            self.scheduler.kv_events_publisher.emit_kv_metrics()
        # 中译：发布 KV cache 事件（无论是否推指标，每次都会调用）。
        self.scheduler.kv_events_publisher.publish_kv_events()

    def report_decode_stats(
        self,
        can_run_cuda_graph: bool,
        running_batch: ScheduleBatch = None,
        num_correct_drafts: int = 0,
    ):
        # 中译：上报 decode 统计。分两部分：每次迭代都做的轻量工作 + 每隔 decode_log_interval 做的重工作。
        batch = running_batch or self.scheduler.running_batch

        # Every-iteration work: realtime token counting + status logger
        # 中译：每次迭代的轻量工作——实时 token 计数（推 realtime_tokens 指标）+ 状态日志 dump。
        if self.current_scheduler_metrics_enabled:
            # 中译：本次 decode 产出的 token = batch 大小 + 投机正确草稿数。
            decode_tokens = batch.batch_size() + num_correct_drafts
            self.metrics_collector.increment_realtime_tokens(
                # TODO unify this w/ the bumping logic in `Scheduler.num_generated_tokens` accumulator
                decode_tokens=decode_tokens,
                dp_cooperation_info=batch.dp_cooperation_info,
            )
            if self.enable_mfu_metrics:
                flops, read_bytes, write_bytes = self._estimate_decode_perf(
                    batch, decode_tokens
                )
                self.metrics_collector.increment_estimated_perf(
                    num_flops_per_gpu=flops,
                    num_read_bytes_per_gpu=read_bytes,
                    num_write_bytes_per_gpu=write_bytes,
                )
                self._mfu_log_flops += flops
                self._mfu_log_read_bytes += read_bytes
                self._mfu_log_write_bytes += write_bytes

            if x := self.scheduler_status_logger:
                x.maybe_dump(batch, self.scheduler.waiting_queue)

        # Periodic work: log + heavy metrics at decode_log_interval
        # 中译：以下为周期性重工作——仅当 decode 计数对 decode_log_interval 取模为 0 时才执行，
        #       避免每步都做昂贵的统计/日志。
        if self.forward_ct_decode % self.scheduler.server_args.decode_log_interval != 0:
            return
        if (
            not self.is_stats_logging_rank
            and not self.current_scheduler_metrics_enabled
        ):
            return

        # 中译：算两次周期上报之间的间隔与生成吞吐（= 周期内生成 token / 间隔），然后清零计数。
        gap_latency = time.perf_counter() - self.last_decode_stats_tic
        self.last_decode_stats_tic = time.perf_counter()
        self.last_gen_throughput = self.num_generated_tokens / gap_latency

        self.num_generated_tokens = 0
        num_running_reqs = len(batch.reqs)

        pool_stats = self.scheduler.pool_stats_observer.get_pool_stats()
        token_usage_msg = ", ".join(pool_stats.get_decode_usage_msg_parts()) + ", "

        # 中译：可选记录「每个 batch size 对应的平均单步耗时」，供性能分析（除以 interval 得单步均值）。
        if RECORD_STEP_TIME:
            self.step_time_dict[num_running_reqs].append(
                gap_latency / self.scheduler.server_args.decode_log_interval
            )

        batch_iter = (
            batch.forward_iter
            if batch is not None and batch.forward_iter is not None
            else self.scheduler.forward_ct
        )
        iter_msg = f" [{batch_iter}]" if LOG_FORWARD_ITERS else ""
        msg = f"Decode batch{iter_msg}, #running-req: {num_running_reqs}, {token_usage_msg}"

        # 中译：投机解码指标。未启用时接受长度/接受率均为 0。
        spec_num_steps = 0
        spec_num_draft_tokens = 0
        if self.scheduler.spec_algorithm.is_none():
            spec_accept_length = 0
            spec_accept_rate = 0
        else:
            # 中译：平均接受长度 = 周期内接受 token 总数 / 前向次数（每次前向平均「白拿」多少 token）。
            spec_accept_length = self.spec_num_accept_tokens / self.spec_num_forward_ct
            # 中译：反推被接受的纯草稿数 = 接受总数 - bonus（forward_ct 累计的就是 bonus 总数）。
            num_correct_drafts = self.spec_num_accept_tokens - self.spec_num_forward_ct
            if self.scheduler.server_args.speculative_num_draft_tokens:
                draft_per_round = (
                    self.scheduler.server_args.speculative_num_draft_tokens - 1
                )
            else:
                draft_per_round = self.scheduler.server_args.speculative_num_steps or 0
            # 中译：接受率 = 被接受的草稿数 / 提出的草稿总数（每轮提出 draft_per_round 个 × 前向次数）。
            total_draft_tokens = self.spec_num_forward_ct * draft_per_round
            spec_accept_rate = (
                num_correct_drafts / total_draft_tokens if total_draft_tokens > 0 else 0
            )
            # 中译：把本周期计数累加到生命周期累计，然后清零本周期计数。
            self.spec_total_num_accept_tokens += self.spec_num_accept_tokens
            self.spec_total_num_forward_ct += self.spec_num_forward_ct
            self.spec_num_accept_tokens = self.spec_num_forward_ct = 0
            msg += f"accept len: {spec_accept_length:.2f}, accept rate: {spec_accept_rate:.2f}, "

            if self.current_scheduler_metrics_enabled:
                spec_snapshot = self._active_spec_config_snapshot()
                spec_num_steps = spec_snapshot["num_steps"]
                spec_num_draft_tokens = spec_snapshot["num_draft_tokens"]

        # 中译：decode 阶段无前缀缓存命中概念，命中率固定为 0。
        cache_hit_rate = 0.0

        # 中译：PD 分离的 decode 节点额外打印预分配占用率、预分配/传输/被回退队列长度。
        if self.scheduler.disaggregation_mode == DisaggregationMode.DECODE:
            msg += f"pre-allocated usage: {self.scheduler.disagg_decode_prealloc_queue.num_tokens_pre_allocated / self.scheduler.max_total_num_tokens:.2f}, "
            msg += f"#prealloc-req: {len(self.scheduler.disagg_decode_prealloc_queue.queue)}, "
            msg += f"#transfer-req: {len(self.scheduler.disagg_decode_transfer_queue.queue)}, "
            msg += f"#retracted-req: {len(self.scheduler.disagg_decode_prealloc_queue.retracted_queue)}, "

        if (
            self.scheduler.server_args.language_only
            and self.scheduler.server_args.encoder_transfer_backend
            == "zmq_to_scheduler"
        ):
            msg += (
                f"waiting-image-req: {len(self.scheduler.mm_receiver.waiting_list)}, "
            )

        msg += (
            f"{self._graph_backend_label}: {can_run_cuda_graph}, "
            f"gen throughput (token/s): {self.last_gen_throughput:.2f}, "
            f"#queue-req: {len(self.scheduler.waiting_queue)}"
        )

        # 中译：启用 MFU 时，用周期内累计的 FLOPs/读/写字节除以间隔，得到算力与显存带宽估算，随后清零。
        if self.enable_mfu_metrics and gap_latency > 0:
            flops_per_s = self._mfu_log_flops / gap_latency
            read_bytes_per_s = self._mfu_log_read_bytes / gap_latency
            write_bytes_per_s = self._mfu_log_write_bytes / gap_latency
            tflops_per_s = flops_per_s / 1e12
            read_gb_per_s = read_bytes_per_s / 1e9
            write_gb_per_s = write_bytes_per_s / 1e9
            msg += (
                f", est. decode TFLOPS/s (per GPU): {tflops_per_s:.2f}, "
                f"est. read BW (GB/s per GPU): {read_gb_per_s:.2f}, "
                f"est. write BW (GB/s per GPU): {write_gb_per_s:.2f}"
            )
            self._mfu_log_flops = 0.0
            self._mfu_log_read_bytes = 0.0
            self._mfu_log_write_bytes = 0.0

        if ENABLE_METRICS_DEVICE_TIMER:
            msg += f", fwd occupancy: {self.fwd_occupancy:.2f}%"

        if self.is_stats_logging_rank:
            logger.info(msg)
        if self.current_scheduler_metrics_enabled:
            priority_enabled = self.scheduler.enable_priority_scheduling

            # Basics
            # 中译：基础指标——运行/等待/语法队列请求数、生成吞吐、命中率、decode 上下文长度之和。
            self.stats.num_running_reqs = QueueCount.from_reqs(
                batch.reqs, priority_enabled
            )
            self.stats.num_queue_reqs = QueueCount.from_reqs(
                self.scheduler.waiting_queue, priority_enabled
            )
            self.stats.num_grammar_queue_reqs = len(self.scheduler.grammar_manager)
            self.stats.gen_throughput = self.last_gen_throughput
            self.stats.cache_hit_rate = cache_hit_rate
            self.stats.decode_sum_seq_lens = _decode_total_seq_lens(batch)

            # Memory pool usage ratios / Absolute token counts
            # 中译：填入内存池使用率与绝对 token 数。
            pool_stats.update_scheduler_stats(self.stats)

            # Speculative decoding
            # 中译：投机解码指标——接受长度、接受率、每轮步数、草稿 token 数。
            self.stats.spec_accept_length = spec_accept_length
            self.stats.spec_accept_rate = spec_accept_rate
            self.stats.spec_num_steps = spec_num_steps
            self.stats.spec_num_draft_tokens = spec_num_draft_tokens

            # Retract
            self.stats.num_retracted_reqs = self.num_retracted_reqs
            self.stats.num_paused_reqs = self.num_paused_reqs
            self.num_retracted_reqs = self.num_paused_reqs = 0

            # PD disaggregation
            if self.scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_bootstrap_queue_reqs = QueueCount.from_reqs(
                    self.scheduler.disagg_prefill_bootstrap_queue.queue,
                    priority_enabled,
                )
                self.stats.num_prefill_inflight_queue_reqs = QueueCount.from_reqs(
                    self.scheduler.disagg_prefill_inflight_queue, priority_enabled
                )
            elif self.scheduler.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = QueueCount.from_reqs(
                    self.scheduler.disagg_decode_prealloc_queue.queue, priority_enabled
                )
                self.stats.num_decode_transfer_queue_reqs = QueueCount.from_reqs(
                    self.scheduler.disagg_decode_transfer_queue.queue, priority_enabled
                )

            # Streaming session metrics
            # 中译：流式会话指标——当前活跃会话数、以及这些会话占住（held）的 token 数。
            self.stats.num_streaming_sessions = (
                self.scheduler.pool_stats_observer.streaming_session_count()
            )
            self.stats.streaming_session_held_tokens = (
                self.scheduler.pool_stats_observer.session_held_tokens()
            )

            # Routing key metrics
            # (to reduce the overhead, we only compute this when all requests have routing_key)
            # 中译：路由键（routing key）指标。为降开销，仅当批内所有请求都带 routing_key 时才计算：
            #       统计运行中请求的唯一路由键数及各键计数，再合并等待队列得到全量计数。
            if all(r.routing_key is not None for r in batch.reqs):
                running_routing_keys = [r.routing_key for r in batch.reqs]
                waiting_routing_keys = [
                    r.routing_key for r in self.scheduler.waiting_queue
                ]
                (
                    self.stats.num_unique_running_routing_keys,
                    self.stats.routing_key_running_req_counts,
                ) = compute_routing_key_stats(running_routing_keys)
                _, self.stats.routing_key_all_req_counts = compute_routing_key_stats(
                    running_routing_keys + waiting_routing_keys
                )

            # Utilization / LoRA / HiCache
            # 中译：补充利用率、前向占用率、LoRA 池占用、HiCache 主机层统计，最后统一上报。
            self._calculate_utilization()
            self.stats.fwd_occupancy = self.fwd_occupancy
            self._update_lora_metrics()
            self._log_hicache_stats()
            self.metrics_collector.log_stats(self.stats)
            self.scheduler.kv_events_publisher.emit_kv_metrics()
        self.scheduler.kv_events_publisher.publish_kv_events()

    def log_batch_result_stats(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        # 中译：从一次前向结果中提取额外指标。目前是 EPLB（专家并行负载均衡）的均衡度，
        #       仅生成类结果（GenerationBatchResult）且开启指标时才上报。
        if not self.enable_metrics:
            return
        if not isinstance(result, GenerationBatchResult):
            return

        # 中译：balancedness 衡量 MoE 各专家负载是否均匀（越接近 1 越均衡），按前向模式打标签上报。
        if (m := result.expert_distribution_metrics) is not None:
            self.metrics_collector.increment_eplb_balancedness(
                forward_mode=batch.forward_mode.name.lower(),
                balancedness=m.eplb_balancedness.item(),
            )

    def _emit_forward_pass_metrics(
        self,
        batch: ScheduleBatch,
        result=None,
    ):
        """Emit per-iteration ForwardPassMetrics over ZMQ PUB.

        Prefers GPU-accurate timing from DeviceTimer (which wraps
        model_runner.forward / cuda_graph.replay via PR #24197).
        Falls back to monotonic clock when DeviceTimer is not enabled.

        中译：通过 ZMQ PUB 发布逐次前向的 FPM 指标。优先用 DeviceTimer 的精确 GPU 计时
              （它包裹了 model_runner.forward / cuda_graph.replay），未启用时回退到 monotonic 墙钟。
        """
        if not self.scheduler.enable_fpm:
            return

        from sglang.srt.observability.forward_pass_metrics import (
            ForwardPassMetrics,
        )

        if self.scheduler._fpm_uses_device_timer:
            # 中译：先触发上报回调把本次 GPU 时间累计进来，取出后清零；为 0 说明本次无前向，跳过。
            self.forward_pass_device_timer._report()
            wall_time = self.scheduler._fpm_gpu_time_acc
            self.scheduler._fpm_gpu_time_acc = 0.0
            if wall_time == 0.0:
                return
        else:
            # 中译：无设备计时器时，用 monotonic 时钟从前向起点到现在的差作为耗时。
            wall_time = max(0.0, time.monotonic() - batch.fpm_start_time)

        fpm = ForwardPassMetrics(
            worker_id=self.scheduler._fpm_worker_id,
            dp_rank=self.scheduler._fpm_dp_rank,
            wall_time=wall_time,
            scheduled_requests=self._build_scheduled_request_metrics(batch),
            queued_requests=self._build_queued_request_metrics(),
        )
        self.scheduler._fpm_publisher.publish(fpm)

    def _shutdown_fpm(self):
        """Shut down the FPM publisher thread.

        中译：关闭 FPM 发布线程（进程退出/清理时调用）。
        """
        if self.scheduler.enable_fpm:
            self.scheduler._fpm_publisher.shutdown()

    def _log_hicache_stats(self):
        """Populate HiCache host-tier stats on self.stats.

        These are pushed to Prometheus by SchedulerMetricsCollector.log_stats().

        中译：填充分层缓存（HiCache）主机层（host tier）统计到 self.stats：已用/总 token 数。
              这些值随后由 SchedulerMetricsCollector.log_stats() 推到 Prometheus。
        """
        if not self.scheduler.enable_hierarchical_cache:
            return

        # 中译：兼容两种命名的主机侧 KV 池属性，取到后用「总量 - 可用量」算已用 token 数。
        host_pool = getattr(
            self.scheduler.tree_cache, "token_to_kv_pool_host", None
        ) or getattr(self.scheduler.tree_cache, "full_kv_pool_host", None)
        assert host_pool is not None, "Host pool not found"
        self.stats.hicache_host_used_tokens = (
            host_pool.size - host_pool.available_size()
        )
        self.stats.hicache_host_total_tokens = host_pool.size

    def _update_lora_metrics(self):
        """Update LoRA pool metrics for monitoring and autoscaling.

        中译：更新 LoRA 内存池指标（用于监控与自动扩缩容）：占用槽位/总槽位/利用率。
              通过遍历当前运行批次中各请求的 lora_id 统计「活跃适配器数」，更贴近真实负载。
        """
        if not self.scheduler.enable_lora:
            return

        try:
            # Get LoRA memory pool stats
            lora_manager = self.scheduler.tp_worker.model_runner.lora_manager
            if lora_manager is None or lora_manager.memory_pool is None:
                return

            mem_pool = lora_manager.memory_pool
            slots_total = mem_pool.max_loras_per_batch

            # Calculate active adapters from running batch
            # This gives a true measure of current load for autoscaling purposes
            active_lora_ids = set()

            # For PP mode, check all running micro batches
            # 中译：流水并行（PP）下有多个 micro batch，需逐个遍历；否则只看单个 running_batch。
            if self.scheduler.server_args.pp_size > 1:
                for batch in self.scheduler.running_mbs:
                    if batch and hasattr(batch, "reqs"):
                        for req in batch.reqs:
                            if hasattr(req, "lora_id") and req.lora_id is not None:
                                active_lora_ids.add(req.lora_id)
            # For normal mode, check running_batch
            elif self.scheduler.running_batch:
                if hasattr(self.scheduler.running_batch, "reqs"):
                    for req in self.scheduler.running_batch.reqs:
                        if hasattr(req, "lora_id") and req.lora_id is not None:
                            active_lora_ids.add(req.lora_id)

            # Count active adapters (excluding None for base model)
            slots_used = len(active_lora_ids)
            utilization = slots_used / slots_total if slots_total > 0 else 0.0

            # Update stats
            self.stats.lora_pool_slots_used = slots_used
            self.stats.lora_pool_slots_total = slots_total
            self.stats.lora_pool_utilization = utilization

        except Exception as e:
            logger.warning(f"Failed to update LoRA metrics: {e}")

    def _calculate_utilization(self):
        # 中译：计算整体利用率指标。prefill 节点不计算（置 -1 表示 N/A）。
        if self.scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
            self.stats.utilization = -1
        else:
            # TODO: max_running_requests_under_SLO has no setter — sglang:utilization stuck at 0 (regressed #22713).
            # 中译：利用率取两者较大值——「运行请求数 / SLO 下最大并发」与「token 使用率 / 0.9」。
            #       （注：max_running_requests_under_SLO 目前无 setter，导致该指标可能恒为 0，见 #22713）
            max_under_slo = getattr(
                self.scheduler, "max_running_requests_under_SLO", None
            )
            if max_under_slo is not None and max_under_slo > 0:
                self.stats.utilization = max(
                    self.stats.num_running_reqs.total / max_under_slo,
                    self.stats.token_usage / 0.9,
                )

    def update_device_timer(self):
        # 中译：更新设备计时器窗口，计算前向 GPU 占用率（fwd_occupancy = 窗口内 GPU 时间 / 墙钟时间）。
        #       每过 decode_log_interval 个 batch 重开一个窗口。
        if not ENABLE_METRICS_DEVICE_TIMER:
            return
        self.forward_pass_device_timer._report()
        now = time.perf_counter()
        # 中译：窗口第一个 batch——记录窗口起点、清零累计 GPU 时间（此时不更新占用率，保留上次值）。
        if self._device_timer_window_batch_count == 0:
            # Window start: keep the last published value instead of NaN-ing
            # the gauge. Readers sample it asynchronously, and the window
            # boundary can phase-lock with the decode-log cadence, turning a
            # one-tick NaN into NaN on every log line. NaN is published only
            # when truly stale (reset_device_timer_window after idle).
            self._device_timer_window_start = now
            self._device_timer_window_gpu_time = 0.0
        else:
            # 中译：非首个 batch——用墙钟经过时间作分母，算出占用率百分比（封顶 100%）。
            cpu_time = now - self._device_timer_window_start
            if cpu_time > 0:
                self.fwd_occupancy = min(
                    self._device_timer_window_gpu_time / cpu_time * 100, 100
                )
        self._device_timer_window_batch_count += 1
        # 中译：窗口计满一个 decode_log_interval 后归零，下次重开新窗口。
        if (
            self._device_timer_window_batch_count
            >= self.scheduler.server_args.decode_log_interval
        ):
            self._device_timer_window_batch_count = 0

    def reset_device_timer_window(self):
        # 中译：重置占用率窗口（如长时间空闲后），把 fwd_occupancy 置为 NaN 表示数据已陈旧。
        if ENABLE_METRICS_DEVICE_TIMER:
            self._device_timer_window_batch_count = 0
            self.fwd_occupancy = float("nan")

    def _maybe_log_idle_metrics(self):
        """Collect and log metrics every 30 seconds during idle.

        中译：空闲（无请求）期间，每 30 秒仍上报一次指标，确保 Prometheus 不因长时间无数据而断点。
              此时吞吐固定为 0，但仍刷新内存池、队列、会话等状态量。
        """
        if (
            not self.current_scheduler_metrics_enabled
            or time.perf_counter() <= self.metrics_collector.last_log_time + 30
        ):
            return

        self.scheduler.pool_stats_observer.get_pool_stats().update_scheduler_stats(
            self.stats
        )
        self.stats.num_streaming_sessions = (
            self.scheduler.pool_stats_observer.streaming_session_count()
        )
        self.stats.streaming_session_held_tokens = (
            self.scheduler.pool_stats_observer.session_held_tokens()
        )

        priority_enabled = self.scheduler.enable_priority_scheduling
        self.stats.num_running_reqs = QueueCount.from_reqs(
            self.scheduler.running_batch.reqs, priority_enabled
        )
        self.stats.gen_throughput = 0
        self.stats.num_queue_reqs = QueueCount.from_reqs(
            self.scheduler.waiting_queue, priority_enabled
        )
        self.stats.num_grammar_queue_reqs = len(self.scheduler.grammar_manager)
        if self.scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
            self.stats.num_prefill_bootstrap_queue_reqs = QueueCount.from_reqs(
                self.scheduler.disagg_prefill_bootstrap_queue.queue, priority_enabled
            )
            self.stats.num_prefill_inflight_queue_reqs = QueueCount.from_reqs(
                self.scheduler.disagg_prefill_inflight_queue, priority_enabled
            )
        if self.scheduler.disaggregation_mode == DisaggregationMode.DECODE:
            self.stats.num_decode_prealloc_queue_reqs = QueueCount.from_reqs(
                self.scheduler.disagg_decode_prealloc_queue.queue, priority_enabled
            )
            self.stats.num_decode_transfer_queue_reqs = QueueCount.from_reqs(
                self.scheduler.disagg_decode_transfer_queue.queue, priority_enabled
            )
        self.metrics_collector.log_stats(self.stats)
