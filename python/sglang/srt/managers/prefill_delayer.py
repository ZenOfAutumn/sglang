"""Cross-rank prefill delaying to balance load across DP ranks.

中译：跨 rank 的 prefill 延迟器。
      在「PD 不分离 + overlap 调度」下，prefill 批次会和正在跑的 decode 批次合批。
      若过早插入 prefill，会让 decode 达不到最大 batch size，拖低吞吐；而不同 DP（数据并行）
      rank 的负载又各不相同，需要全局协调。本模块通过每轮在所有 DP/TP rank 间用 all-gather
      汇总各自的「是否可 prefill / token 使用率 / 运行批大小 / 等待队列长度」等信息，统一决定
      本轮是否推迟 prefill，在「攒大 batch 提升吞吐」与「别等太久（最大延迟轮数 / token 水位线
      兜底 / wall-clock 超时）」之间做权衡。
"""

import dataclasses
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector

# 中译：调试日志开关（环境变量），开启后会打印每次延迟决策的原因与计数。
_DEBUG_LOG = get_bool_env_var("SGLANG_PREFILL_DELAYER_DEBUG_LOG")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _State:
    """PrefillDelayer 的内部不可变状态：记录某次延迟决策过程中的累计信息。

    当调度器决定推迟一次 prefill 时，会创建/更新该状态，用于跟踪已经连续推迟了多少轮，
    以及本次等待从何时开始（便于统计等待耗时与判断是否超过最大延迟轮数）。
    """

    # 已经连续推迟 prefill 的轮数（forward pass 次数）。
    delayed_count: int = 0
    # 本次等待的起始时间戳（用于统计等待了多久）。
    start_time: float = field(default_factory=time.perf_counter)

    def bump_delayed_count(self) -> "_State":
        """返回一个 delayed_count 加 1 的新状态（frozen 不可变，故用 replace 复制）。"""
        return dataclasses.replace(self, delayed_count=self.delayed_count + 1)


class _NegotiateOutput(NamedTuple):
    """一次“是否允许 prefill”协商的输出结果。

    协商会跨所有 DP/TP rank 汇总各自的可 prefill 状态，统一给出本轮决策。
    """

    # 协商后应保存的新状态；None 表示无需继续等待、清空状态。
    next_state: Optional[_State]
    # 对全局可 prefill 情况的估计："all" / "none" / "mixed"。
    input_estimation: str
    # 本轮是否允许执行 prefill。
    output_allow: bool
    # 决策原因（no_wait / wait_success / delay / wait_timeout / token_watermark 等），用于打点。
    output_reason: str
    # 全局可 prefill 的 rank 数量。
    num_prefillable: int
    # 因 token 使用率低于水位线而强制放行的 rank 数量。
    num_token_watermark_force_allow: int
    # Accumulated wait of the prefill being released on this pass. Carried
    # explicitly because `next_state` is None on every release path and thus
    # cannot convey it to the metrics observation.
    wait_forward_passes: int = 0
    wait_seconds: float = 0.0


class PrefillDelayer:
    """Prefill 延迟器：在 PD 不分离 + overlap 调度下，决定是否推迟本轮 prefill。

    背景：prefill 与正在运行的 decode 合批时，若过早插入 prefill，会让 decode 批次
    达不到最大 batch size，从而降低整体吞吐。该组件在每轮调度跨所有 rank 协商，
    在“稍等几轮以攒出更大 batch”与“不能等太久（最大延迟轮数 / token 水位线兜底）”
    之间做权衡。

    约束：仅在 disaggregation_mode == "null"（PD 不分离）且启用 overlap 调度时可用。
    """

    def __init__(
        self,
        dp_size: int,
        attn_tp_size: int,
        cpu_group,
        server_args,
        max_delay_passes: int,
        token_usage_low_watermark: Optional[float],
        metrics_collector: Optional["SchedulerMetricsCollector"] = None,
        device: Optional["torch.device"] = "cpu",
        device_group=None,
    ):
        # 中译：_max_delay_passes —— 最多连续推迟多少轮 forward（兜底，避免无限等待）。
        self._max_delay_passes = max_delay_passes
        # 中译：_token_usage_low_watermark —— token（KV）使用率低水位线，低于它说明 GPU 没吃饱，强制放行。
        self._token_usage_low_watermark = token_usage_low_watermark
        # Queue-based trigger is opt-in: activates only when queue_min_ratio
        # is explicitly set. Additive with the slot-based trigger.
        # 中译：基于「等待队列长度」的触发器是可选项，仅当显式设置 queue_min_ratio 时启用；
        #       与基于「slot（空位）」的触发器叠加生效（满足任一即推迟）。
        self._queue_min_ratio = server_args.prefill_delayer_queue_min_ratio
        # Fall back to 5000ms if unset; this is a local safety cap, not a
        # semantic default, so we don't surface it via ServerArgs.
        # 中译：未设置时回退到 5000ms。这是本地安全上限（wall-clock 超时兜底），不是语义默认值，
        #       故不通过 ServerArgs 暴露给用户。
        self._max_delay_ms = server_args.prefill_delayer_max_delay_ms
        if self._max_delay_ms is None:
            self._max_delay_ms = 5000.0
        self._queue_trigger_enabled = self._queue_min_ratio is not None
        logger.info(
            f"PrefillDelayer initialized with "
            f"max_delay_passes={self._max_delay_passes} "
            f"token_usage_low_watermark={self._token_usage_low_watermark} "
            f"queue_min_ratio={self._queue_min_ratio} "
            f"max_delay_ms={self._max_delay_ms} "
            f"queue_trigger_enabled={self._queue_trigger_enabled}"
        )
        self.dp_size = dp_size
        self.enable_dp_attention = server_args.enable_dp_attention
        # 中译：未启用 DP attention 时，所有 DP rank 等价，all-gather 的 DP 维度退化为 1。
        dp_size_dim = dp_size if self.enable_dp_attention else 1

        # Mirror scheduler_dp_attn_mixin's NCCL all-gather path: when the
        # env flag is on (or overlap scheduling is disabled), ride the NCCL
        # device group on `device` instead of gloo on CPU.
        # 中译：与 scheduler_dp_attn_mixin 的 NCCL all-gather 路径保持一致：当环境开关打开
        #       （或 overlap 调度被关闭）时，用设备上的 NCCL group 做 all-gather，而不是 CPU 上的 gloo。
        use_nccl = (
            server_args.disable_overlap_schedule
            or envs.SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH.get()
        )
        if use_nccl:
            assert (
                device_group is not None
            ), "device_group is required when using NCCL for PrefillDelayer all-gather"
            self._gather_group = device_group
            self._gather_device = device
        else:
            self._gather_group = cpu_group
            self._gather_device = "cpu"

        # Fields packed per rank into the all-gather tensor: prefillable,
        # token_watermark_force_allow, running_batch, max_prefill_bs,
        # waiting_queue_len.
        # 中译：每个 rank 打包进 all-gather 张量的 5 个字段（int64）：
        #       是否可 prefill、是否因低水位强制放行、运行批大小、最大 prefill 批大小、等待队列长度。
        self._global_info_buffer = torch.empty(
            (dp_size_dim, attn_tp_size, 5),
            dtype=torch.int64,
            device=self._gather_device,
        )

        self._metrics_collector = metrics_collector

        self._curr_state: Optional[_State] = None
        self.skip_first_delayer = True

        assert (
            not server_args.disable_overlap_schedule
        ), "To use PrefillDelayer, disable_overlap_schedule must be False."

    def _negotiate_should_allow_prefill(
        self,
        local_prefillable: bool,
        token_usage: float,
        running_batch: int = 0,
        max_prefill_bs: int = 0,
        max_running_requests: int = 0,
        waiting_queue_len: int = 0,
    ) -> _NegotiateOutput:
        # 中译：有状态版本——调用「近似纯函数」做协商，并把返回的新状态写回 self._curr_state。
        out = self._negotiate_should_allow_prefill_pure(
            prev_state=self._curr_state,
            local_prefillable=local_prefillable,
            token_usage=token_usage,
            running_batch=running_batch,
            max_prefill_bs=max_prefill_bs,
            max_running_requests=max_running_requests,
            waiting_queue_len=waiting_queue_len,
        )
        self._curr_state = out.next_state
        return out

    # (Almost) pure function, do not modify self state
    # 中译：（几乎）纯函数：除了内部的 all-gather 通信外不读写 self 状态，决策只依赖入参，便于测试推理。
    def _negotiate_should_allow_prefill_pure(
        self,
        prev_state: Optional[_State],
        local_prefillable: bool,
        token_usage: float,
        running_batch: int = 0,
        max_prefill_bs: int = 0,
        max_running_requests: int = 0,
        waiting_queue_len: int = 0,
    ) -> _NegotiateOutput:
        # Compute local states
        # 中译：本地态——本 rank 是否「因 token 使用率低于水位线而要求强制放行」。
        #       (x := ...) 是海象赋值，仅当水位线已配置且当前使用率低于它时为 True。
        local_token_watermark_force_allow = (
            local_prefillable
            and ((x := self._token_usage_low_watermark) is not None)
            and (token_usage < x)
        )

        # Gather global states
        # 中译：跨所有 rank all-gather，汇总成全局态（只取每个 DP 组 TP0 的信息）。
        tp0_info = self._gather_info(
            local_prefillable=local_prefillable,
            local_token_watermark_force_allow=local_token_watermark_force_allow,
            running_batch=running_batch,
            max_prefill_bs=max_prefill_bs,
            waiting_queue_len=waiting_queue_len,
        )
        global_prefillable = tp0_info[:, 0]
        global_token_watermark_force_allow = tp0_info[:, 1]
        global_running_batch = tp0_info[:, 2]
        global_max_prefill_bs = tp0_info[:, 3]
        global_waiting_queue_len = tp0_info[:, 4]

        # Compute derived global states
        # 中译：派生全局态——根据各 rank 的「可 prefill」标志判断全局形态：
        #       全部可 prefill → "all"；全部不可 → "none"；部分可部分不可 → "mixed"。
        if global_prefillable.min().item() > 0:
            prefillable_status = "all"
        elif global_prefillable.max().item() == 0:
            prefillable_status = "none"
        else:
            prefillable_status = "mixed"
        global_exists_token_watermark_force_allow = (
            global_token_watermark_force_allow.max().item() > 0
        )
        debug_info = dict(
            input_estimation=prefillable_status,
            num_prefillable=global_prefillable.sum().item(),
            num_token_watermark_force_allow=global_token_watermark_force_allow.sum().item(),
        )

        # Wait accumulated so far, taken from prev_state. Release paths attach
        # this so the wait histograms observe the real value; delay paths leave
        # the defaults (0) since the wait isn't finished and isn't observed.
        wait_info = dict(
            wait_forward_passes=prev_state.delayed_count if prev_state else 0,
            wait_seconds=(
                (time.perf_counter() - prev_state.start_time) if prev_state else 0.0
            ),
        )

        # Compute outputs
        # 中译：根据全局形态分三种情况给出决策（"all" / "none" / "mixed"）。
        if prefillable_status == "all":
            # 中译：情况一——所有 rank 都可 prefill。此时才考虑「为攒大 batch 而延迟」。
            # Safety valve: low KV usage means GPU is underutilized, skip
            # delay. Mirrors the check in the "mixed" branch.
            # 中译：安全阀——只要有任一 rank 触发了低水位强制放行，说明 GPU 没吃饱，跳过延迟直接放行。
            #       （与下方 "mixed" 分支里的同名检查对应）
            if global_exists_token_watermark_force_allow:
                return _NegotiateOutput(
                    next_state=None,
                    output_allow=True,
                    output_reason="token_watermark",
                    **debug_info,
                    **wait_info,
                )

            if not self.enable_dp_attention:
                # 中译：未启用 DP attention 时 max_running_requests 是全局总量，
                #       这里向上取整地均摊到每个 DP rank，便于与各 rank 的运行批大小比较。
                max_running_requests = (
                    max_running_requests + self.dp_size - 1
                ) // self.dp_size

            global_running_batch_max = int(global_running_batch.max().item())
            global_max_prefill_bs_max = int(global_max_prefill_bs.max().item())
            global_waiting_queue_max = int(global_waiting_queue_len.max().item())

            # Queue-based trigger: delay prefill until the waiting queue
            # reaches queue_min = min(running_req * ratio, max_prefill_bs),
            # capped by a wall-clock timeout to bound worst-case TTFT.
            # Targets workloads where decode requests finish one-at-a-time
            # and fragment prefill into many tiny batches.
            # 中译：基于队列的触发条件——等待队列长度未达到 queue_min =
            #       min(运行请求数 * ratio, max_prefill_bs) 时就推迟 prefill；
            #       并用 wall-clock 超时（_max_delay_ms）封顶以约束最坏情况的 TTFT（首 token 时延）。
            #       针对的是「decode 请求逐个完成、把 prefill 切碎成许多小批」的负载。
            queue_condition = False
            if self._queue_trigger_enabled and global_running_batch_max > 0:
                queue_min_effective = min(
                    int(global_running_batch_max * self._queue_min_ratio),
                    global_max_prefill_bs_max,
                )
                queue_condition = (
                    queue_min_effective > 0
                    and global_waiting_queue_max < queue_min_effective
                )
                if queue_condition and prev_state is not None:
                    # 中译：已等待时长超过上限则取消队列触发，强制不再继续等。
                    elapsed_ms = (time.perf_counter() - prev_state.start_time) * 1000.0
                    if elapsed_ms >= self._max_delay_ms:
                        queue_condition = False

            # 中译：基于 slot（空位）的触发条件——可用并发空位不足以容纳一个最大 prefill 批时，
            #       说明此刻插入 prefill 会挤占 decode 空间，倾向于推迟。
            slot_condition = (
                max_running_requests - global_running_batch_max
                < global_max_prefill_bs_max
            )

            if slot_condition or queue_condition:
                # When the "max_decode_bs - running_bs < max_prefill_bs" condition is met,
                # the first merge_batch causes the decoding to fail to reach the maximum batch size.
                # 中译：满足上述条件时，第一次 merge_batch 会使 decode 达不到最大 batch size。
                #       但首次出现该条件时先放行一次（skip_first_delayer），避免冷启动时一直空等。
                if self.skip_first_delayer:
                    self.skip_first_delayer = False
                    pass
                else:
                    # 中译：非首次——确实推迟本轮 prefill，累加 delayed_count 并返回 output_allow=False。
                    next_state = prev_state or _State()
                    next_state = next_state.bump_delayed_count()
                    return _NegotiateOutput(
                        next_state=next_state,
                        output_allow=False,
                        output_reason="delay",
                        **debug_info,
                    )
            # 中译：放行 prefill。若之前等待过则原因记为 wait_success，否则为 no_wait（本轮无需等待）。
            exist_previous_wait = prev_state is not None
            return _NegotiateOutput(
                next_state=None,
                output_allow=True,
                output_reason="wait_success" if exist_previous_wait else "no_wait",
                **debug_info,
                **wait_info,
            )
        elif prefillable_status == "none":
            # 中译：情况二——没有任何 rank 可 prefill。允许与否都无所谓，为简单起见直接放行。
            return _NegotiateOutput(
                next_state=None,
                # It does not matter whether we allow or not, thus we allow for simplicity
                output_allow=True,
                output_reason="",
                **debug_info,
                **wait_info,
            )
        elif prefillable_status == "mixed":
            # 中译：情况三——部分 rank 可 prefill、部分不可。为对齐各 rank，倾向于等慢的 rank 跟上。
            if global_exists_token_watermark_force_allow:
                return _NegotiateOutput(
                    next_state=None,
                    output_allow=True,
                    output_reason="token_watermark",
                    **debug_info,
                    **wait_info,
                )

            # 中译：未达到最大延迟轮数则继续推迟（delay）；否则超时放行（wait_timeout），避免无限等待。
            prev_delayed_count = prev_state.delayed_count if prev_state else 0
            if prev_delayed_count < self._max_delay_passes - 1:
                next_state = prev_state or _State()
                next_state = next_state.bump_delayed_count()
                return _NegotiateOutput(
                    next_state=next_state,
                    output_allow=False,
                    output_reason="delay",
                    **debug_info,
                )
            else:
                return _NegotiateOutput(
                    next_state=None,
                    output_allow=True,
                    output_reason="wait_timeout",
                    **debug_info,
                    **wait_info,
                )
        else:
            raise NotImplementedError

    def _gather_info(
        self,
        local_prefillable: bool,
        local_token_watermark_force_allow: bool,
        running_batch: int = 0,
        max_prefill_bs: int = 0,
        waiting_queue_len: int = 0,
    ):
        # 中译：把本 rank 的 5 个字段打包成张量，all-gather 到全局缓冲，再返回各 DP 组 TP0 的信息。
        local_info = torch.tensor(
            [
                int(local_prefillable),
                int(local_token_watermark_force_allow),
                running_batch,
                max_prefill_bs,
                waiting_queue_len,
            ],
            device=self._gather_device,
            dtype=torch.int64,
        )
        torch.distributed.all_gather_into_tensor(
            self._global_info_buffer.flatten(),
            local_info,
            group=self._gather_group,
        )
        # 中译：只取每个 DP 组内 TP rank 0 的信息（同组各 TP rank 的协商输入一致）。
        tp0_info = self._global_info_buffer[:, 0, :]
        return tp0_info


class PrefillDelayerSinglePassExecutor:
    """单轮（single pass）执行器：封装某一轮调度内对 PrefillDelayer 的一次性使用。

    保证每轮最多真正协商一次（结果缓存到 _result），并在 finalize 时上报本轮指标；
    若本轮从未调用协商，finalize 会以 local_prefillable=False 补一次默认协商。
    """

    def __init__(self, prefill_delayer: PrefillDelayer, token_usage: float):
        self._prefill_delayer = prefill_delayer
        self._token_usage = token_usage
        self._result: Optional[_NegotiateOutput] = None

    @property
    def _called(self) -> bool:
        # 中译：本轮是否已经协商过（结果已缓存到 _result）。
        return self._result is not None

    def finalize(self, *, actual_prefill: bool):
        # 中译：本轮收尾——若全程未协商过，则以 local_prefillable=False 补一次默认协商，
        #       然后上报本轮指标（决策结果 vs 实际是否真的执行了 prefill）。
        if not self._called:
            self.negotiate_should_allow_prefill(local_prefillable=False)

        _record_single_pass_result(
            actual_execution=actual_prefill,
            output=self._result,
            metrics_collector=self._prefill_delayer._metrics_collector,
        )

    def negotiate_should_allow_prefill(
        self,
        local_prefillable: bool,
        running_batch: int = 0,
        max_prefill_bs: int = 0,
        max_running_requests: int = 0,
        waiting_queue_len: int = 0,
    ) -> bool:
        # 中译：本轮首次调用才真正协商并缓存结果；后续调用直接复用缓存的 output_allow（保证每轮只协商一次）。
        if not self._called:
            self._result = self._prefill_delayer._negotiate_should_allow_prefill(
                local_prefillable=local_prefillable,
                token_usage=self._token_usage,
                running_batch=running_batch,
                max_prefill_bs=max_prefill_bs,
                max_running_requests=max_running_requests,
                waiting_queue_len=waiting_queue_len,
            )
        return self._result.output_allow


def _record_single_pass_result(
    actual_execution: bool,
    output: _NegotiateOutput,
    metrics_collector: Optional["SchedulerMetricsCollector"],
) -> None:
    # 中译：把本轮单次协商结果记录到日志（调试时）并上报到指标收集器（若有）。
    #       actual_execution 表示本轮最终是否真的执行了 prefill（协商放行不等于一定执行）。
    if _DEBUG_LOG:
        if output.output_allow and (output.output_reason == "wait_timeout"):
            logger.info(
                f"PrefillDelayer timeout thus not forbid prefill "
                f"(num_prefillable={output.num_prefillable}, "
                f"actual_execution={actual_execution})"
            )
        elif output.output_allow and (output.output_reason == "token_watermark"):
            logger.info(
                f"PrefillDelayer force allow prefill due to low watermark. "
                f"(num_prefillable={output.num_prefillable}, "
                f"num_token_watermark_force_allow={output.num_token_watermark_force_allow}, "
                f"actual_execution={actual_execution})"
            )
        else:
            assert output.output_reason in {
                "",
                "wait_success",
                "no_wait",
                "delay",
            }

    if metrics_collector is not None:
        metrics_collector.observe_prefill_delayer_outcome(
            forward_passes=output.wait_forward_passes,
            wait_seconds=output.wait_seconds,
            input_estimation=output.input_estimation,
            output_allow=output.output_allow,
            output_reason=output.output_reason,
            actual_execution=actual_execution,
        )
