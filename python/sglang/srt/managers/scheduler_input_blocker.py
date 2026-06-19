# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Temporarily block/stash incoming requests behind a global barrier.

中译：在「全局屏障（global barrier）」下临时阻塞并暂存进入的请求。
      某些操作（如权重热更新）要求所有 rank 进入「全局静止」状态后再统一恢复，否则会出现
      不一致。SchedulerInputBlocker 收到 BLOCK 请求后进入阻塞态，把后续到来的普通请求先暂存
      （pending）而非立即处理；收到 UNBLOCK 后进入「全局解锁屏障」态，等所有 rank 都到达该屏障
      （poll_global_arrived 轮询确认），再统一放行先前暂存的请求，回到正常态。
"""

import logging
from contextlib import contextmanager
from enum import Enum, auto
from typing import Any, List, Optional

from sglang.srt.managers.io_struct import BlockReqInput, BlockReqType
from sglang.srt.utils.poll_based_barrier import PollBasedBarrier

logger = logging.getLogger(__name__)


class SchedulerInputBlocker:
    """调度器输入阻塞器：在全局屏障期间暂存请求，屏障到达后统一放行。

    中译：noop（no-operation）模式用于「本 rank 不实际参与阻塞」的场景（如非 TP0），
          此时 handle 收到的 recv_reqs 为 None，仅参与全局屏障的协同。
    """

    def __init__(self, noop: bool):
        # 中译：初始为未阻塞态；_pending_reqs 暂存阻塞期间到来的请求；
        #       _global_unblock_barrier 用于跨 rank 协同「何时可统一解锁」。
        self._state = _State.UNBLOCKED
        self._pending_reqs = []
        self._noop = noop
        self._global_unblock_barrier = PollBasedBarrier(noop=noop)

    def handle(self, recv_reqs: Optional[List[Any]]):
        # 中译：noop 模式下 recv_reqs 必须为 None，反之必须非 None（断言两者一致）。
        assert (recv_reqs is None) == self._noop

        if not self._noop:
            # 中译：非 noop——逐个处理收到的请求（普通请求可能被放行或暂存，BLOCK/UNBLOCK 切换状态）。
            output_reqs = []
            for recv_req in recv_reqs:
                output_reqs += self._handle_recv_req(recv_req)

        # 中译：轮询「是否所有 rank 都已到达解锁屏障」。
        global_arrived_unblock_barrier = (
            self._global_unblock_barrier.poll_global_arrived()
        )
        # 中译：当本 rank 处于解锁屏障态且全局都已到达时，放行先前暂存的请求并回到正常态。
        if (
            self._state == _State.GLOBAL_UNBLOCK_BARRIER
            and global_arrived_unblock_barrier
        ):
            output_reqs += self._handle_arrive_unblock_barrier()

        if not self._noop:
            return output_reqs

    def _handle_recv_req(self, recv_req):
        # 中译：处理单个请求。BlockReqInput 是控制信号（BLOCK/UNBLOCK），其余为普通业务请求。
        if isinstance(recv_req, BlockReqInput):
            if recv_req.type == BlockReqType.BLOCK:
                self._execute_block_req()
                return []
            elif recv_req.type == BlockReqType.UNBLOCK:
                self._execute_unblock_req()
                return []
            else:
                raise NotImplementedError(f"{recv_req=}")
        else:
            # 中译：普通请求——未阻塞时直接放行；阻塞期间则暂存到 pending，待解锁后统一放行。
            if self._state == _State.UNBLOCKED:
                return [recv_req]
            else:
                self._pending_reqs.append(recv_req)
                return []

    def _execute_block_req(self):
        # 中译：处理 BLOCK 信号——从「未阻塞」切到「阻塞」态，此后普通请求都会被暂存。
        logger.info("Handle block req")
        self._change_state(original=_State.UNBLOCKED, target=_State.BLOCKED)

    def _execute_unblock_req(self):
        # 中译：处理 UNBLOCK 信号——从「阻塞」切到「全局解锁屏障」态，并在屏障上标记本 rank 已到达。
        #       注意此时还不能立即放行，需等所有 rank 都到达屏障（见 _handle_arrive_unblock_barrier）。
        logger.info("Handle unblock req")
        self._change_state(
            original=_State.BLOCKED, target=_State.GLOBAL_UNBLOCK_BARRIER
        )
        self._global_unblock_barrier.local_arrive()

    def _handle_arrive_unblock_barrier(self):
        # 中译：全局屏障到达——回到未阻塞态，并把暂存的请求全部取出返回、清空 pending。
        logger.info(f"Arrived at unblock barrier ({len(self._pending_reqs)=})")
        self._change_state(
            original=_State.GLOBAL_UNBLOCK_BARRIER, target=_State.UNBLOCKED
        )
        output_reqs = [*self._pending_reqs]
        self._pending_reqs.clear()
        return output_reqs

    def _change_state(self, original: "_State", target: "_State"):
        # 中译：状态切换并校验——只允许从预期的 original 态切到 target 态，否则断言失败（防状态错乱）。
        assert self._state == original, f"{self._state=} {original=} {target=}"
        self._state = target


class _State(Enum):
    # 中译：阻塞器的三种状态。
    UNBLOCKED = auto()  # 未阻塞：普通请求直接放行。
    BLOCKED = auto()  # 已阻塞：普通请求被暂存。
    GLOBAL_UNBLOCK_BARRIER = auto()  # 等待全局解锁屏障：本 rank 已收到 UNBLOCK，等其他 rank 到齐。


@contextmanager
def input_blocker_guard_region(send_to_scheduler):
    # 中译：上下文管理器——进入时向调度器发送 BLOCK 请求，退出时（无论是否异常）发送 UNBLOCK，
    #       用于把一段「需要全局静止」的临界区（如权重更新）包裹起来。
    send_to_scheduler.send_pyobj(BlockReqInput(BlockReqType.BLOCK))
    try:
        yield
    finally:
        send_to_scheduler.send_pyobj(BlockReqInput(BlockReqType.UNBLOCK))
