import logging
import time
from typing import Callable, Optional, Tuple

from sglang.srt.managers.io_struct import FlushCacheReqInput, FlushCacheReqOutput
from sglang.srt.managers.scheduler_components.ipc_channels import (
    SchedulerIpcChannels,
)


class SchedulerFlushWrapper:
    """中译：flush_cache（清空缓存）请求的包装器。

    flush_cache 用于清空前缀缓存（prefix cache）等运行时缓存。直接清空要求系统处于空闲
    状态（否则会破坏正在进行的请求）。本类支持两种语义：
    - timeout_s <= 0：立即执行清空（不等待空闲）。
    - timeout_s > 0：若当前已空闲则立即清空；否则把请求挂起（pending），由调度器主循环
      周期性调用 check_pending()，等到系统空闲时再清空，或超时后返回失败。
    同一时刻只允许一个挂起的 flush 请求。
    """

    def __init__(
        self,
        *,
        flush_cache: Callable[[], bool],
        is_fully_idle: Callable[[], bool],
        ipc_channels: SchedulerIpcChannels,
    ) -> None:
        # 中译：注入的实际清缓存动作（返回是否成功）。
        self._flush_cache = flush_cache
        # 中译：判断系统是否完全空闲（无在跑/排队请求）的回调。
        self._is_fully_idle = is_fully_idle
        # 中译：IPC 通道，用于把延迟执行的结果回传给 tokenizer。
        self._ipc_channels = ipc_channels
        # 中译：挂起中的请求及其截止时间 (req, deadline)；None 表示无挂起。
        self._pending: Optional[Tuple[FlushCacheReqInput, float]] = None

    def handle(self, recv_req: FlushCacheReqInput) -> Optional[FlushCacheReqOutput]:
        # 中译：已有一个 flush 在等待中——拒绝新的请求（同时只允许一个挂起）。
        if self._pending is not None:
            return FlushCacheReqOutput(
                success=False,
                message="Another flush_cache is already in progress.",
            )

        timeout_s = float(recv_req.timeout_s or 0.0)
        # 中译：未设置超时（<=0）——立即强制清空并同步返回结果。
        if timeout_s <= 0.0:
            return FlushCacheReqOutput(success=self._flush_cache())

        # 中译：设置了超时但当前已空闲——可立即安全清空并同步返回。
        if self._is_fully_idle():
            return FlushCacheReqOutput(success=self._flush_cache())

        # 中译：当前繁忙——挂起请求并记录绝对截止时间，返回 None（结果稍后异步回传）。
        self._pending = (recv_req, time.monotonic() + timeout_s)
        return None

    def check_pending(self) -> None:
        # 中译：由调度器主循环周期性调用，推进挂起中的 flush 请求。无挂起则直接返回。
        if self._pending is None:
            return

        pending_req, deadline = self._pending

        # 中译：系统已变空闲——执行清空，清除挂起状态，并把结果异步回传给原请求方。
        if self._is_fully_idle():
            success = self._flush_cache()
            self._pending = None
            self._ipc_channels.send_to_tokenizer.send_output(
                FlushCacheReqOutput(success=success), pending_req
            )
            return

        # 中译：仍未空闲且已到截止时间——放弃等待，回传超时失败。
        if time.monotonic() >= deadline:
            logging.warning(
                "Deferred flush_cache timed out while waiting for idle state."
            )
            self._pending = None
            self._ipc_channels.send_to_tokenizer.send_output(
                FlushCacheReqOutput(
                    success=False, message="Timed out waiting for idle state."
                ),
                pending_req,
            )
