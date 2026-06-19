from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from contextlib import contextmanager
from multiprocessing import Process
from typing import Callable, List, Optional

import psutil

from sglang.srt.utils.cudacore_pyspy_dump_utils import pyspy_dump_schedulers

logger = logging.getLogger(__name__)


class Watchdog:
    """看门狗（Watchdog）抽象基类，对外暴露统一接口。

    用途：监控某个工作循环（如 Scheduler 的主循环）是否「卡死」。调用方在每轮循环里
    调用 feed() 喂狗；若在 watchdog_timeout 时间内一直没有被喂狗，则判定为卡死，
    打印诊断信息并（在非 soft 模式下）向父进程发送 SIGQUIT 触发清理。

    本基类自身的 feed()/disable() 为空实现，真正逻辑在子类 _WatchdogReal 中；
    通过 create() 工厂方法按是否配置超时来返回真实实现或空实现（_WatchdogNoop）。
    """

    @staticmethod
    def create(
        debug_name: str,
        watchdog_timeout: Optional[float],
        soft: bool = False,
        test_stuck_time: float = 0,
    ) -> Watchdog:
        """工厂方法：根据是否配置超时返回对应的看门狗实现。

        Args:
            debug_name: 看门狗名称，用于日志中区分是哪个组件的看门狗。
            watchdog_timeout: 超时时间（秒）。为 None 表示禁用看门狗，返回空实现。
            soft: 软模式。为 True 时超时只记录日志、不杀进程；为 False 时会发 SIGQUIT。
            test_stuck_time: 测试用参数，让看门狗在首次喂狗时故意 sleep 这么久，
                             以验证超时检测逻辑是否生效。

        Returns:
            未配置超时返回 _WatchdogNoop（空操作），否则返回 _WatchdogReal（真实实现）。
        """
        if watchdog_timeout is None:
            # 未启用看门狗时，卡死测试也必须关闭（卡死测试依赖 soft 看门狗存在）。
            assert (
                test_stuck_time == 0
            ), f"stuck tester can be enabled only if soft watchdog is enabled."
            return _WatchdogNoop()
        return _WatchdogReal(
            debug_name=debug_name,
            watchdog_timeout=watchdog_timeout,
            soft=soft,
            test_stuck_time=test_stuck_time,
        )

    def feed(self):
        """喂狗：表示「我还活着」。基类为空实现，由子类重写。"""
        pass

    @contextmanager
    def disable(self):
        """上下文管理器：在 with 代码块内临时关闭看门狗检测。

        用于已知会长时间阻塞、但属于正常情况的代码段（如加载大模型权重），
        避免被误判为卡死。基类为空实现。
        """
        yield


class _WatchdogReal(Watchdog):
    """看门狗的真实实现：通过一个单调递增的计数器表达「活跃度」。

    工作原理：
    - 每次 feed() 让 self._counter 自增；
    - 后台监控线程（WatchdogRaw）周期性地读取该计数器，
      若在 watchdog_timeout 内计数器没有变化，则判定卡死。
    """

    def __init__(
        self,
        debug_name: str,
        watchdog_timeout: float,
        soft: bool = False,
        test_stuck_time: float = 0,
    ):
        self._counter = 0  # 喂狗计数器，每次 feed() 自增，供监控线程读取判断是否卡死。
        self._active = True  # 是否处于激活状态；disable() 期间为 False，监控线程会跳过检测。
        self._test_stuck_time = test_stuck_time  # 测试用：首次喂狗时故意阻塞的秒数。
        self._test_stuck_triggered = False  # 标记故意卡死是否已触发过（只触发一次）。
        # 创建底层监控对象，通过回调读取计数器与激活状态，实现关注点分离。
        self._raw = WatchdogRaw(
            debug_name=debug_name,
            get_counter=lambda: self._counter,
            is_active=lambda: self._active,
            watchdog_timeout=watchdog_timeout,
            soft=soft,
        )
        logger.info(f"Watchdog {self._raw.debug_name} initialized.")
        if self._test_stuck_time > 0:
            logger.info(
                f"Watchdog {self._raw.debug_name} is configured to use {test_stuck_time=}."
            )

    def feed(self):
        """喂狗：自增计数器表示主循环仍在正常推进。"""
        # 只触发一次故意卡死，避免阻塞服务启动时的健康检查，
        # 同时又能验证看门狗的超时检测逻辑是否生效。
        if self._test_stuck_time > 0 and not self._test_stuck_triggered:
            self._test_stuck_triggered = True
            logger.info(
                f"Watchdog {self._raw.debug_name} start deliberately stuck for {self._test_stuck_time}s"
            )
            time.sleep(self._test_stuck_time)
            logger.info(
                f"Watchdog {self._raw.debug_name} end deliberately stuck for {self._test_stuck_time}s"
            )

        self._counter += 1

    @contextmanager
    def disable(self):
        """临时关闭检测：将 _active 置为 False，with 块结束后再恢复为 True。"""
        assert self._active  # 进入前必须是激活态，防止嵌套调用导致状态错乱。
        self._active = False
        try:
            yield
        finally:
            assert not self._active  # 确认期间没有被其他地方意外改回激活态。
            self._active = True


class _WatchdogNoop(Watchdog):
    """看门狗的空实现：未配置超时时使用，feed()/disable() 全部为无操作。"""

    pass


class WatchdogRaw:
    """看门狗的底层实现：在独立守护线程中轮询计数器，超时则触发诊断与（可选）杀进程。

    采用回调（get_counter / is_active）而非直接持有上层对象，做到与喂狗逻辑解耦，
    便于复用与测试。
    """

    def __init__(
        self,
        debug_name: str,
        get_counter: Callable[[], int],
        is_active: Callable[[], bool],
        watchdog_timeout: float,
        soft: bool = False,
        dump_info: Optional[Callable[[], str]] = None,
    ):
        self.debug_name = debug_name  # 看门狗名称，用于日志标识。
        self.get_counter = get_counter  # 回调：读取当前喂狗计数器值。
        self.is_active = is_active  # 回调：判断当前是否需要检测（disable 期间为 False）。
        self.watchdog_timeout = watchdog_timeout  # 超时阈值（秒）。
        self.soft = soft  # 软模式：超时仅记录日志，不向父进程发信号。
        self.dump_info = dump_info  # 可选回调：超时时输出额外调试信息。

        # 记录父进程句柄，超时后向其发送 SIGQUIT。
        self.parent_process = psutil.Process().parent()
        # 启动守护线程进行后台轮询，主进程退出时该线程自动结束。
        t = threading.Thread(target=self._watchdog_thread, daemon=True)
        t.start()

    def _watchdog_thread(self):
        """守护线程入口：循环执行检测，捕获异常以防线程静默崩溃。"""
        try:
            while True:
                self._watchdog_once()
        except Exception as e:
            logger.error(
                f"{self.debug_name} watchdog thread crashed: {e}", exc_info=True
            )

    def _watchdog_once(self):
        """执行一轮完整的「监测 -> 检测到卡死 -> 处理」流程。

        内层 while 循环持续轮询：只要计数器在 watchdog_timeout 内有变化，就刷新基准；
        一旦超过阈值仍无变化，则跳出循环进入卡死处理流程。
        """
        watchdog_last_counter = 0  # 上次记录的计数器值。
        watchdog_last_time = time.perf_counter()  # 上次计数器变化的时间点。

        while True:
            current = time.perf_counter()
            if self.is_active():  # 仅在激活状态下检测（disable 期间跳过）。
                current_counter = self.get_counter()
                if watchdog_last_counter == current_counter:
                    # 计数器无变化：判断是否已超过超时阈值。
                    if current > watchdog_last_time + self.watchdog_timeout:
                        break  # 卡死，跳出循环进入处理。
                else:
                    # 计数器有推进，刷新基准值与时间。
                    watchdog_last_counter = current_counter
                    watchdog_last_time = current
            # 以超时时间的一半为间隔轮询，兼顾及时性与开销。
            time.sleep(self.watchdog_timeout / 2)

        # —— 以下为检测到卡死后的处理流程 ——
        # 若提供了额外调试信息回调，则打印出来。
        if self.dump_info is not None and (info_msg := self.dump_info()):
            logger.error(f"{self.debug_name} debug info:\n{info_msg}")

        # 对所有 scheduler 进程做 py-spy 堆栈转储，便于定位卡死位置。
        pyspy_dump_schedulers()
        logger.error(
            f"{self.debug_name} watchdog timeout "
            f"({self.watchdog_timeout=}, {self.soft=})"
        )
        # 刷新标准错误/标准输出缓冲，确保日志即时落地。
        print(file=sys.stderr, flush=True)
        print(file=sys.stdout, flush=True)

        if not self.soft:
            # 非软模式：稍等片刻让父进程把错误信息打印完，再发 SIGQUIT 触发清理。
            time.sleep(5)
            self.parent_process.send_signal(signal.SIGQUIT)


class SubprocessWatchdog:
    """Monitors subprocess liveness and triggers SIGQUIT when a crash is detected.

    When a subprocess crashes (e.g., NCCL timeout causing C++ std::terminate()),
    Python exception handlers never run, leaving the main process as a zombie
    service. This watchdog polls subprocess liveness in a daemon thread and
    sends SIGQUIT to trigger proper cleanup.

    See: https://github.com/sgl-project/sglang/issues/18421
    """

    def __init__(
        self,
        processes: List[Process],
        process_names: Optional[List[str]] = None,
        interval: float = 1.0,
    ):
        self._processes = processes
        self._names = process_names or [f"process_{i}" for i in range(len(processes))]
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None or not self._processes:
            return
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="subprocess-watchdog"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 2)
            self._thread = None

    def _monitor_loop(self) -> None:
        try:
            while not self._stop_event.wait(self._interval):
                if self._check_processes():
                    return
        except Exception as e:
            logger.error(f"SubprocessWatchdog thread crashed: {e}", exc_info=True)

    def _check_processes(self) -> bool:
        for proc, name in zip(self._processes, self._names):
            if proc.is_alive() or proc.exitcode == 0:
                continue

            logger.error(
                f"Subprocess {name} (pid={proc.pid}) crashed "
                f"with exit code {proc.exitcode}. "
                f"Triggering SIGQUIT for cleanup..."
            )
            os.kill(os.getpid(), signal.SIGQUIT)
            return True
        return False
