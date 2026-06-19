from __future__ import annotations

import asyncio
import copy
from collections import deque
from typing import Deque, Generic, List, Optional, TypeVar

import zmq

T = TypeVar("T")


class FanOutCommunicator(Generic[T]):
    """Fan-out request + collect response primitive over zmq.

    One send is fanned out to `fan_out` recipients; the caller awaits until
    all `fan_out` responses are collected. Supports two modes:
    - "queueing": requests are serialized; concurrent callers wait in a FIFO queue.
    - "watching": concurrent callers share a single in-flight request and all
      receive the same result when it completes.

    Only one request is in-flight at any time in either mode.

    中译：基于 zmq 的「扇出请求 + 汇集响应」原语。
          一次 send 会被扇出（fan-out）到 `fan_out` 个接收方（例如多个 DP/TP rank），
          调用方会一直 await，直到把全部 `fan_out` 个响应都收齐。支持两种模式：
          - "queueing"（排队）：请求被串行化，并发调用者在一个 FIFO 队列中依次等待，
            各自独立发送一次自己的请求。
          - "watching"（共享观察）：并发调用者共享同一个在途请求（只发送一次），
            请求完成时所有等待者都拿到同一份结果。
          两种模式下，任意时刻都只允许有一个请求在途（in-flight）。
          典型用途：TokenizerManager 等需要向所有底层 worker 广播控制类请求
          （如更新权重、释放显存等）并等待全部确认时使用。
    """

    def __init__(self, sender: zmq.Socket, fan_out: int, mode="queueing"):
        """初始化扇出通信器。

        参数：
            sender：用于发送请求的 zmq 套接字（PUSH/PUB 等）。
            fan_out：扇出的接收方数量，即每次请求预期要收齐的响应条数。
            mode："queueing"（排队，默认）或 "watching"（共享观察）。
        副作用：仅记录配置与初始化内部状态，不发送任何消息。
        """
        self._sender = sender
        self._fan_out = fan_out
        self._mode = mode
        # 中译：当前在途请求的「完成事件」；收齐 fan_out 个响应后被 set。None 表示当前无在途请求。
        self._result_event: Optional[asyncio.Event] = None
        # 中译：当前在途请求已收集到的响应列表（边收边 append）。
        self._result_values: Optional[List[T]] = None
        # 中译：排队模式下等待轮到自己的调用者队列（每个元素是一个「就绪事件」，FIFO 唤醒）。
        self._ready_queue: Deque[asyncio.Event] = deque()

        assert mode in ["queueing", "watching"]

    async def queueing_call(self, obj: T):
        """排队模式下发起一次扇出请求并等待收齐全部响应。

        语义：并发调用者串行执行。若已有请求在途或队列非空，则把自己挂到 FIFO 队列
        中等待被唤醒；轮到自己后再发送请求，收齐 fan_out 个响应后返回，并唤醒下一个。

        参数 obj：要扇出的请求对象；为 None 时跳过发送（仅用于等待/占位）。
        返回：长度为 fan_out 的响应列表。
        """
        ready_event = asyncio.Event()
        # 中译：若已有请求在途，或前面还有人在排队，则进入队列等待轮到自己。
        if self._result_event is not None or len(self._ready_queue) > 0:
            self._ready_queue.append(ready_event)
            await ready_event.wait()
            # 中译：被唤醒时，上一个请求必定已收尾、共享状态已清空。
            assert self._result_event is None
            assert self._result_values is None

        if obj is not None:
            self._sender.send_pyobj(obj)

        # 中译：建立本次请求的共享状态并等待 handle_recv 收齐响应后 set 事件。
        self._result_event = asyncio.Event()
        self._result_values = []
        await self._result_event.wait()
        result_values = self._result_values
        # 中译：本次结果已取出，清空共享状态，为下一个排队者腾位。
        self._result_event = self._result_values = None

        # 中译：唤醒队列中的下一个等待者（FIFO）。
        if len(self._ready_queue) > 0:
            self._ready_queue.popleft().set()

        return result_values

    async def watching_call(self, obj):
        """共享观察模式下发起/搭车一次扇出请求并等待结果。

        语义：并发调用者共享同一个在途请求——第一个进入者负责真正发送并建立共享状态，
        其余进入者「搭车」等待同一个事件；请求完成时所有等待者各自拿到结果的深拷贝。

        参数 obj：要扇出的请求对象；为 None 时跳过发送。
        返回：响应列表的深拷贝（避免多个等待者共享同一可变对象）。
        """
        # 中译：若当前无在途请求，则由「第一个进入者」创建共享状态并发送请求。
        if self._result_event is None:
            assert self._result_values is None
            self._result_values = []
            self._result_event = asyncio.Event()

            if obj is not None:
                self._sender.send_pyobj(obj)

        # Capture local refs before await -- after event fires, the first
        # awakened coroutine clears shared state; later awaiters use local refs.
        # 中译：在 await 前先抓住本地引用——事件触发后，第一个被唤醒的协程会清空共享状态，
        #       后续被唤醒者只能依赖这里抓到的本地引用来读取结果。
        values = self._result_values
        event = self._result_event
        await event.wait()

        result_values = copy.deepcopy(values)
        # 中译：仅由「第一个唤醒者」（其抓到的 event 仍是当前 event）负责清空共享状态，避免重复清理。
        if self._result_event is event:
            self._result_event = self._result_values = None
        return result_values

    async def __call__(self, obj):
        """可调用入口：按构造时指定的模式路由到 queueing_call 或 watching_call。"""
        if self._mode == "queueing":
            return await self.queueing_call(obj)
        else:
            return await self.watching_call(obj)

    def handle_recv(self, recv_obj: T):
        """接收回调：每收到一个响应就 append，收齐 fan_out 个后触发完成事件。

        通常由外层事件循环在 zmq 收到响应消息时调用。副作用：可能 set 完成事件，
        从而唤醒正在 await 的 queueing_call / watching_call。
        """
        self._result_values.append(recv_obj)
        if len(self._result_values) == self._fan_out:
            self._result_event.set()

    @staticmethod
    def merge_results(results):
        """把多个 rank 的响应合并为单一的 (是否全部成功, 拼接后的消息) 二元组。

        参数 results：各响应对象的列表，每个对象需有 success/message 字段。
        返回：(all_success, all_message)——全部成功才为 True；消息用 " | " 连接。
        """
        all_success = all([r.success for r in results])
        all_message = [r.message for r in results]
        all_message = " | ".join(all_message)
        return all_success, all_message
