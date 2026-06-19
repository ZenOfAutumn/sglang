from __future__ import annotations

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

"""
Mixin classes and utils for multi-http-worker mode
This file uses multiple processes to handle requests and tokenization, reducing the overhead of python and http server.

中译：「多 HTTP worker 模式」相关的 Mixin 类与工具集合。
      在该模式下，SGLang 用「多个进程」来分担请求处理与分词（tokenization）工作，
      从而降低单个 Python 进程 + HTTP 服务的 GIL/吞吐瓶颈。整体数据流如下：

        若干 TokenizerWorker（各自的 HTTP worker 进程，负责分词）
              │  前向：worker → router → scheduler
              ▼
        MultiTokenizerRouter（路由器，前后向中转 + 广播 pause/continue）
              │  反向：detokenizer → router → 对应的 worker
              ▼
        Scheduler ──► DetokenizerManager（可有 N 个，由 MultiDetokenizerRouter 分流）

      本文件提供：
      - SocketMapping：按 IPC 名称缓存 ZMQ PUSH 套接字的小工具。
      - MultiHttpWorkerDetokenizerMixin：给 DetokenizerManager 注入多 worker 事件循环。
      - MultiTokenizerRouter / MultiDetokenizerRouter：前后向路由器。
      - TokenizerWorker：多 worker 模式下的 TokenizerManager 子类。
      - 共享内存（shared_memory）读写工具，用于在进程间传递启动参数。
"""

import asyncio
import logging
import multiprocessing as multiprocessing
import os
import pickle
import signal
import sys
import threading
import zlib
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import psutil
import setproctitle
import zmq
import zmq.asyncio

from sglang.srt.disaggregation.utils import DisaggregationMode, TransferBackend
from sglang.srt.managers.disagg_service import start_disagg_service
from sglang.srt.managers.io_struct import (
    BaseBatchReq,
    BaseReq,
    BatchEmbeddingOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
    ContinueGenerationReqInput,
    FreezeGCReq,
    PauseContinueBroadcast,
    PauseGenerationReqInput,
    TokenizerWorkerRegistration,
)
from sglang.srt.managers.load_snapshot import (
    create_load_snapshot_reader,
    zmq_reader_owner,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    configure_logger,
    kill_itself_when_parent_died,
    kill_process_tree,
)
from sglang.srt.utils.network import get_zmq_socket
from sglang.utils import get_exception_traceback

if TYPE_CHECKING:
    from sglang.srt.managers.detokenizer_manager import DetokenizerManager

logger = logging.getLogger(__name__)


class SocketMapping:
    """按 IPC 名称（ipc_name）缓存并复用 ZMQ PUSH 套接字的小工具。

    中译：在多 worker 模式下，路由器/detokenizer 需要把结果分发给「目标 IPC 地址各不
          相同」的多个下游进程。本类维护 {ipc_name -> PUSH socket} 的映射，按需懒创建
          套接字并复用，避免每次发送都重新连接。所有套接字共享同一个 zmq.Context。
    """

    def __init__(self):
        # 中译：每个 SocketMapping 自带一个 ZMQ 上下文，以及 ipc_name -> socket 的缓存表。
        self._zmq_context = zmq.Context()
        self._mapping: Dict[str, zmq.Socket] = {}

    def clear_all_sockets(self):
        # 中译：关闭并清空所有已缓存的套接字（进程退出/异常清理时调用）。
        for socket in self._mapping.values():
            socket.close()
        self._mapping.clear()

    def _register_ipc_mapping(self, ipc_name: str, is_tokenizer: bool):
        # 中译：为某个 ipc_name 懒创建一个 PUSH 套接字并登记到映射表。
        #       is_tokenizer 仅用于日志区分目标是 tokenizer 还是 detokenizer。
        type_str = "tokenizer" if is_tokenizer else "detokenizer"
        if ipc_name in self._mapping:
            # 中译：已注册过则跳过（重复注册仅告警，不重建套接字）。
            logger.warning(f"{type_str} already registered {ipc_name=}, skipping...")
            return
        logger.info(f"Registering {type_str} {ipc_name=} in SocketMapping...")
        socket = get_zmq_socket(self._zmq_context, zmq.PUSH, ipc_name, False)
        self._mapping[ipc_name] = socket

    def send_output(self, ipc_name: str, output: Any, is_tokenizer: bool = False):
        # 中译：把 output 通过对应 ipc_name 的 PUSH 套接字发送出去；首次使用时自动注册。
        if ipc_name is None:
            # Some unhandled cases
            # 中译：某些未处理场景下 ipc_name 可能为空，此时只告警并丢弃（避免崩溃）。
            logger.warning(f"IPC name is None, output type={type(output)}, skipping...")
            return

        if ipc_name not in self._mapping:
            self._register_ipc_mapping(ipc_name, is_tokenizer=is_tokenizer)
        self._mapping[ipc_name].send_pyobj(output)


def _extract_field_by_index(
    output: Any, field_name: str, index: int, check_length: bool = True
) -> Any:
    """Extract a field value from output by index, handling None and length checks.

    Args:
        output: The output object containing the field
        field_name: The name of the field to extract
        index: The index to access in the field list
        check_length: If True, check both field existence and length. If False, only check field existence.

    Returns:
        A list containing the field value at index, or None if not available.

    中译：从批量 output 对象里按下标 index 抽取某个字段的「单条」值，用于把一个批次的
          输出拆成「逐请求」的小对象（见 _handle_output_by_index）。会处理 None 与长度越界。
    参数：
        output：包含该字段的批量输出对象。
        field_name：要抽取的字段名。
        index：在该字段（列表/字典）中的下标。
        check_length：True 时同时检查字段存在性与长度；False 时只检查存在性
                      （用于本就可能为空的可选字段，如各种 logprobs）。
    返回：
        把 index 处的值包成单元素列表返回；不可用时返回 None。
        若字段是 dict，则对每个 value 取 index 处的元素，重组成同结构的新 dict。
    """
    field = getattr(output, field_name, None)
    if field is None:
        return None

    # 中译：字段是 dict 时（如按 key 分组的统计），对每个 value 列表分别取第 index 项。
    if isinstance(field, dict):
        new_field = {}
        for k, v in field.items():
            new_field[k] = v[index] if len(v) > index else None
        return new_field

    # 中译：需要检查长度时，越界则返回 None，避免 IndexError。
    if check_length:
        if len(field) <= index:
            return None

    return [field[index]]


def _handle_output_by_index(output, i):
    """NOTE: A maintainable method is better here.

    中译：把一个「批量输出对象」拆出第 i 条，构造成只含单条数据的同类型新对象，
          以便分发给对应那一条请求的 TokenizerWorker。支持三种批量类型：
          BatchTokenIDOutput / BatchEmbeddingOutput / BatchStrOutput；其余类型原样返回。
          NOTE 原注释：这里逐字段手写并不优雅，理想做法是更易维护的通用拆分方法。
    """
    if isinstance(output, BatchTokenIDOutput):
        new_output = BatchTokenIDOutput(
            rids=[output.rids[i]],
            spec_verify_ct=_extract_field_by_index(output, "spec_verify_ct", i),
            spec_num_correct_drafts=_extract_field_by_index(
                output, "spec_num_correct_drafts", i
            ),
            spec_correct_drafts_histogram=_extract_field_by_index(
                output, "spec_correct_drafts_histogram", i
            ),
            time_stats=_extract_field_by_index(output, "time_stats", i),
            finished_reasons=_extract_field_by_index(output, "finished_reasons", i),
            decoded_texts=_extract_field_by_index(output, "decoded_texts", i),
            decode_ids=_extract_field_by_index(output, "decode_ids", i),
            read_offsets=_extract_field_by_index(output, "read_offsets", i),
            output_ids=_extract_field_by_index(output, "output_ids", i),
            skip_special_tokens=_extract_field_by_index(
                output, "skip_special_tokens", i
            ),
            spaces_between_special_tokens=_extract_field_by_index(
                output, "spaces_between_special_tokens", i
            ),
            no_stop_trim=_extract_field_by_index(output, "no_stop_trim", i),
            prompt_tokens=_extract_field_by_index(output, "prompt_tokens", i),
            completion_tokens=_extract_field_by_index(output, "completion_tokens", i),
            reasoning_tokens=_extract_field_by_index(output, "reasoning_tokens", i),
            cached_tokens=_extract_field_by_index(output, "cached_tokens", i),
            cached_tokens_details=_extract_field_by_index(
                output, "cached_tokens_details", i
            ),
            input_token_logprobs_val=_extract_field_by_index(
                output, "input_token_logprobs_val", i, check_length=False
            ),
            input_token_logprobs_idx=_extract_field_by_index(
                output, "input_token_logprobs_idx", i, check_length=False
            ),
            output_token_logprobs_val=_extract_field_by_index(
                output, "output_token_logprobs_val", i, check_length=False
            ),
            output_token_logprobs_idx=_extract_field_by_index(
                output, "output_token_logprobs_idx", i, check_length=False
            ),
            input_top_logprobs_val=_extract_field_by_index(
                output, "input_top_logprobs_val", i, check_length=False
            ),
            input_top_logprobs_idx=_extract_field_by_index(
                output, "input_top_logprobs_idx", i, check_length=False
            ),
            output_top_logprobs_val=_extract_field_by_index(
                output, "output_top_logprobs_val", i, check_length=False
            ),
            output_top_logprobs_idx=_extract_field_by_index(
                output, "output_top_logprobs_idx", i, check_length=False
            ),
            input_token_ids_logprobs_val=_extract_field_by_index(
                output, "input_token_ids_logprobs_val", i, check_length=False
            ),
            input_token_ids_logprobs_idx=_extract_field_by_index(
                output, "input_token_ids_logprobs_idx", i, check_length=False
            ),
            output_token_ids_logprobs_val=_extract_field_by_index(
                output, "output_token_ids_logprobs_val", i, check_length=False
            ),
            output_token_ids_logprobs_idx=_extract_field_by_index(
                output, "output_token_ids_logprobs_idx", i, check_length=False
            ),
            output_token_entropy_val=_extract_field_by_index(
                output, "output_token_entropy_val", i, check_length=False
            ),
            output_hidden_states=_extract_field_by_index(
                output, "output_hidden_states", i, check_length=False
            ),
            routed_experts=_extract_field_by_index(
                output, "routed_experts", i, check_length=False
            ),
            indexer_topk=_extract_field_by_index(
                output, "indexer_topk", i, check_length=False
            ),
            retraction_counts=_extract_field_by_index(output, "retraction_counts", i),
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
            token_steps=_extract_field_by_index(
                output, "token_steps", i, check_length=False
            ),
            customized_info=_extract_field_by_index(
                output, "customized_info", i, check_length=False
            ),
            dp_ranks=_extract_field_by_index(output, "dp_ranks", i, check_length=False),
        )
    elif isinstance(output, BatchEmbeddingOutput):
        new_output = BatchEmbeddingOutput(
            rids=[output.rids[i]],
            finished_reasons=_extract_field_by_index(output, "finished_reasons", i),
            embeddings=_extract_field_by_index(output, "embeddings", i),
            prompt_tokens=_extract_field_by_index(output, "prompt_tokens", i),
            cached_tokens=_extract_field_by_index(output, "cached_tokens", i),
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
        )
    elif isinstance(output, BatchStrOutput):
        new_output = BatchStrOutput(
            rids=[output.rids[i]],
            spec_verify_ct=_extract_field_by_index(output, "spec_verify_ct", i),
            spec_num_correct_drafts=_extract_field_by_index(
                output, "spec_num_correct_drafts", i
            ),
            spec_correct_drafts_histogram=_extract_field_by_index(
                output, "spec_correct_drafts_histogram", i
            ),
            time_stats=_extract_field_by_index(output, "time_stats", i),
            finished_reasons=_extract_field_by_index(output, "finished_reasons", i),
            output_strs=_extract_field_by_index(output, "output_strs", i),
            output_ids=_extract_field_by_index(output, "output_ids", i),
            prompt_tokens=_extract_field_by_index(output, "prompt_tokens", i),
            completion_tokens=_extract_field_by_index(output, "completion_tokens", i),
            reasoning_tokens=_extract_field_by_index(output, "reasoning_tokens", i),
            cached_tokens=_extract_field_by_index(output, "cached_tokens", i),
            cached_tokens_details=_extract_field_by_index(
                output, "cached_tokens_details", i
            ),
            input_token_logprobs_val=_extract_field_by_index(
                output, "input_token_logprobs_val", i, check_length=False
            ),
            input_token_logprobs_idx=_extract_field_by_index(
                output, "input_token_logprobs_idx", i, check_length=False
            ),
            output_token_logprobs_val=_extract_field_by_index(
                output, "output_token_logprobs_val", i, check_length=False
            ),
            output_token_logprobs_idx=_extract_field_by_index(
                output, "output_token_logprobs_idx", i, check_length=False
            ),
            input_top_logprobs_val=_extract_field_by_index(
                output, "input_top_logprobs_val", i, check_length=False
            ),
            input_top_logprobs_idx=_extract_field_by_index(
                output, "input_top_logprobs_idx", i, check_length=False
            ),
            output_top_logprobs_val=_extract_field_by_index(
                output, "output_top_logprobs_val", i, check_length=False
            ),
            output_top_logprobs_idx=_extract_field_by_index(
                output, "output_top_logprobs_idx", i, check_length=False
            ),
            input_token_ids_logprobs_val=_extract_field_by_index(
                output, "input_token_ids_logprobs_val", i, check_length=False
            ),
            input_token_ids_logprobs_idx=_extract_field_by_index(
                output, "input_token_ids_logprobs_idx", i, check_length=False
            ),
            output_token_ids_logprobs_val=_extract_field_by_index(
                output, "output_token_ids_logprobs_val", i, check_length=False
            ),
            output_token_ids_logprobs_idx=_extract_field_by_index(
                output, "output_token_ids_logprobs_idx", i, check_length=False
            ),
            output_token_entropy_val=_extract_field_by_index(
                output, "output_token_entropy_val", i, check_length=False
            ),
            output_hidden_states=_extract_field_by_index(
                output, "output_hidden_states", i, check_length=False
            ),
            routed_experts=_extract_field_by_index(
                output, "routed_experts", i, check_length=False
            ),
            indexer_topk=_extract_field_by_index(
                output, "indexer_topk", i, check_length=False
            ),
            customized_info=_extract_field_by_index(
                output, "customized_info", i, check_length=False
            ),
            dp_ranks=_extract_field_by_index(output, "dp_ranks", i, check_length=False),
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
            retraction_counts=_extract_field_by_index(output, "retraction_counts", i),
            token_steps=_extract_field_by_index(
                output, "token_steps", i, check_length=False
            ),
        )
    else:
        new_output = output
    return new_output


class MultiHttpWorkerDetokenizerMixin:
    """Mixin class for DetokenizerManager

    中译：DetokenizerManager 的 Mixin（混入类），为其补充「多 HTTP worker 模式」专用能力。
          DetokenizerManager 继承本类后，即可用 multi_http_worker_event_loop 取代普通事件循环：
          解码完成后不再走单一的 send_to_tokenizer 套接字，而是按每条请求携带的
          http_worker_ipc(s) 把结果「扇出（fan out）」回各自的发起方 TokenizerWorker。
    """

    def maybe_clear_socket_mapping(self: DetokenizerManager):
        # 中译：若已建立 socket_mapping（仅多 worker 模式下存在），清理其所有套接字。
        #       供异常退出时安全清理使用。
        if hasattr(self, "socket_mapping"):
            self.socket_mapping.clear_all_sockets()

    def multi_http_worker_event_loop(self: DetokenizerManager):
        """The event loop that handles requests, for multi multi-http-worker mode

        中译：多 HTTP worker 模式下的主事件循环。不断从调度器收消息 → 分发解码 →
              把结果按来源回送给对应的 TokenizerWorker。
        """
        self.socket_mapping = SocketMapping()
        while True:
            recv_obj = self.recv_from_scheduler.recv_pyobj()
            output = self._request_dispatcher(recv_obj)
            if output is None:
                # 中译：handler 返回 None（如 FreezeGC/配置日志类请求）时无需回送，继续下一轮。
                continue

            # Fan out the output back to the originating tokenizer worker(s).
            # In multi-detokenizer mode the upstream MultiDetokenizerRouter may
            # forward either batched or single requests, so handle both shapes.
            # 中译：把输出扇出回发起方 TokenizerWorker。多 detokenizer 模式下，上游
            #       MultiDetokenizerRouter 可能转发「批量」或「单条」请求，两种形态都要处理。
            if isinstance(recv_obj, BaseBatchReq):
                # 中译：批量请求——按每条的 http_worker_ipc 拆出单条结果，分别回送。
                for i, ipc_name in enumerate(recv_obj.http_worker_ipcs):
                    new_output = _handle_output_by_index(output, i)
                    self.socket_mapping.send_output(
                        ipc_name, new_output, is_tokenizer=True
                    )
            elif isinstance(recv_obj, BaseReq):
                # 中译：单条请求——直接回送到它自带的 http_worker_ipc。
                self.socket_mapping.send_output(
                    recv_obj.http_worker_ipc, output, is_tokenizer=True
                )
            else:
                raise ValueError(
                    f"multi_http_worker_event_loop got unexpected req type {type(recv_obj)}"
                )


class MultiTokenizerRouter:
    """A router between tokenizer managers and the scheduler/detokenizer manager.

    Forward: tokenizer managers → router → scheduler.
    Backward: detokenizer manager → router → tokenizer managers.
    Also broadcasts pause/continue to all tokenizer managers for consistent is_pause state.

    中译：位于「多个 TokenizerWorker」与「scheduler / detokenizer」之间的路由器。
          前向：各 TokenizerWorker → 路由器 → 调度器（汇聚）。
          反向：detokenizer → 路由器 → 对应的 TokenizerWorker（按 IPC 分发）。
          此外还把 pause/continue（暂停/继续生成）广播给所有 worker，
          以保证每个 worker 的 is_pause 状态保持一致。
          内部用一个独立的 asyncio 事件循环线程跑两个协程：前向 router_worker_obj 与
          反向 handle_loop。
    """

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        # 中译：创建 3 个 ZMQ asyncio 套接字：从 detokenizer 收结果、向 scheduler 发、从 worker 收。
        self.server_args = server_args
        context = zmq.asyncio.Context(3)
        self.recv_from_detokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.tokenizer_ipc_name, True
        )
        self.send_to_scheduler = get_zmq_socket(
            context, zmq.PUSH, port_args.scheduler_input_ipc_name, True
        )
        self.receive_from_worker = get_zmq_socket(
            context, zmq.PULL, port_args.tokenizer_worker_ipc_name, True
        )
        # 中译：起一个后台线程跑独立的 asyncio 事件循环，并在其上调度前向/反向两个协程。
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        # 中译：前向协程——把 worker 的请求转发到 scheduler。
        self._task = asyncio.run_coroutine_threadsafe(
            self.router_worker_obj(), self._loop
        )
        # 中译：反向协程——把 detokenizer 的结果路由回 worker（包一层异常打印）。
        self._handle_task = asyncio.run_coroutine_threadsafe(
            print_exception_wrapper(self.handle_loop), self._loop
        )

        # In multi-tokenizer mode the N TokenizerWorker processes cannot each
        # bind the zmq PULL socket used for load snapshots, so the single
        # MultiTokenizerRouter process owns it (zmq -> SHM) and the workers
        # read SHM only. Drain it event-driven via the socket's fd instead of
        # polling on a timer.
        # 中译：多 tokenizer 模式下，N 个 TokenizerWorker 无法各自 bind 同一个用于「加载快照
        #       （load snapshot）」的 PULL 套接字，故由本路由器进程独占它（zmq → 共享内存 SHM），
        #       worker 只读 SHM。这里用套接字 fd 的可读事件来驱动排空，而非定时轮询。
        self.load_snapshot_reader = None
        if zmq_reader_owner(server_args, "MultiTokenizerRouter"):
            self.load_snapshot_reader = create_load_snapshot_reader(
                server_args, port_args, caller="MultiTokenizerRouter"
            )
            self._loop.call_soon_threadsafe(self._register_load_snapshot_reader)

        # 中译：PD 分离（disaggregation）场景下启动 bootstrap 服务。
        self.disaggregation_bootstrap_server = start_disagg_service(self.server_args)

        # Worker IPC names for pause/continue broadcasting
        # 中译：记录所有已注册 worker 的 IPC 名，用于 pause/continue 广播。
        self.all_worker_ipcs: set[str] = set()
        # Shared socket mapping (both coroutines run on self._loop, so safe)
        # 中译：共享的套接字映射（前后向两个协程都跑在 self._loop 上，单线程访问故安全）。
        self.socket_mapping = SocketMapping()

    def _run_loop(self):
        # 中译：后台线程入口——永久运行该事件循环。
        self._loop.run_forever()

    def _register_load_snapshot_reader(self):
        """Drain zmq load snapshots into SHM whenever the PULL socket is readable.

        zmq exposes an edge-triggered fd; ``poll()`` drains it until empty, which
        also re-arms the fd, so TokenizerWorkers reading SHM stay up to date
        without any timer.

        中译：每当 PULL 套接字可读时，把收到的「加载快照」排空写入共享内存（SHM）。
              zmq 暴露的是「边沿触发（edge-triggered）」的 fd；poll() 会把它读空，同时
              重新装备（re-arm）该 fd，因此读 SHM 的 TokenizerWorker 无需任何定时器即可保持最新。
        """
        assert self.load_snapshot_reader is not None
        # 中译：把 reader 的 fd 注册到事件循环，可读时回调 poll() 排空。
        self._loop.add_reader(
            self.load_snapshot_reader.fileno(), self.load_snapshot_reader.poll
        )
        # Drain anything already queued before the fd was registered.
        # 中译：先排空在注册之前就已经排队的数据。
        self.load_snapshot_reader.poll()

    async def router_worker_obj(self):
        """Forward path: workers → scheduler, with pause/continue broadcast.

        中译：前向路径——从各 worker 收请求并转发给 scheduler；其中对 worker 注册、
              pause/continue 三类特殊消息做单独处理。
        """
        while True:
            recv_obj = await self.receive_from_worker.recv_pyobj()

            # 中译：worker 注册消息——把该 worker 的 IPC 名记入集合（用于后续广播），不转发。
            if isinstance(recv_obj, TokenizerWorkerRegistration):
                if recv_obj.worker_ipc_name not in self.all_worker_ipcs:
                    self.all_worker_ipcs.add(recv_obj.worker_ipc_name)
                    logger.info(
                        f"Router registered worker IPC: {recv_obj.worker_ipc_name} "
                        f"(total: {len(self.all_worker_ipcs)})"
                    )
                continue

            if isinstance(
                recv_obj, (PauseGenerationReqInput, ContinueGenerationReqInput)
            ):
                # Broadcast to ALL workers so every worker's is_pause is set
                # 中译：暂停/继续请求——广播给所有 worker，使每个 worker 的 is_pause 同步更新。
                is_pause = isinstance(recv_obj, PauseGenerationReqInput)
                broadcast = PauseContinueBroadcast(is_pause=is_pause)
                for ipc_name in self.all_worker_ipcs:
                    self.socket_mapping.send_output(ipc_name, broadcast)
                # Forward to scheduler rank 0 (it broadcasts to all TP/PP/DP
                # ranks internally). Skip for abort mode which drains via polling.
                # 中译：转发给 scheduler 的 rank 0（它再内部广播给所有 TP/PP/DP rank）。
                #       abort 模式例外——它靠轮询排空，不需要在此转发。
                if not (
                    isinstance(recv_obj, PauseGenerationReqInput)
                    and recv_obj.mode == "abort"
                ):
                    await self.send_to_scheduler.send_pyobj(recv_obj)
                continue

            # 中译：普通请求——直接转发给 scheduler。
            await self.send_to_scheduler.send_pyobj(recv_obj)

    async def handle_loop(self):
        """Backward path: detokenizer → route results to correct worker.

        中译：反向路径——从 detokenizer 收结果，并路由到正确的 worker。
        """
        while True:
            recv_obj = await self.recv_from_detokenizer.recv_pyobj()
            await self._distribute_result_to_workers(recv_obj)

    async def _distribute_result_to_workers(self, recv_obj):
        # 中译：把一条（或一批）结果按携带的 http_worker_ipc(s) 分发回各 worker。
        #       单条用 http_worker_ipc；批量用 http_worker_ipcs 列表，逐条拆分后分发。
        if isinstance(recv_obj, BaseReq):
            ipc_names = [recv_obj.http_worker_ipc]
        elif isinstance(recv_obj, BaseBatchReq):
            ipc_names = recv_obj.http_worker_ipcs
        else:
            raise ValueError(f"Unknown recv_obj type: {type(recv_obj)}")

        for i, ipc_name in enumerate(ipc_names):
            new_recv_obj = _handle_output_by_index(recv_obj, i)
            self.socket_mapping.send_output(ipc_name, new_recv_obj)


class MultiDetokenizerRouter:
    """Route scheduler outputs to one of N DetokenizerManager workers.

    Each request is pinned to a worker by hashing its ``http_worker_ipc`` with
    ``zlib.crc32`` (deterministic across runs), so all outputs of the same rid
    always land on the same detokenizer and ``decode_status`` stays consistent.

    中译：把调度器的输出分流到 N 个 DetokenizerManager worker 之一。
          每条请求按其 http_worker_ipc 用 zlib.crc32 取哈希、对 worker 数取模来「钉死」到
          固定 worker（crc32 跨运行确定性一致）。这样同一个 rid 的所有输出始终落到同一个
          detokenizer，使其增量解码状态 decode_status 保持连贯一致（不会因换 worker 而丢状态）。
    """

    def __init__(self, ipc_name_list: List[str], port_args: PortArgs):
        # 中译：ipc_name_list 是各 detokenizer worker 的 IPC 地址；num_workers 为分流取模的基数。
        self.ipc_name_list = ipc_name_list
        self.num_workers = len(ipc_name_list)
        self.socket_mapping = SocketMapping()
        context = zmq.Context(2)
        self.recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, port_args.detokenizer_ipc_name, True
        )

    def _pick(self, key: str) -> str:
        # 中译：按 key（即 http_worker_ipc）做 crc32 哈希取模，选出目标 detokenizer 的 IPC 名。
        return self.ipc_name_list[zlib.crc32(key.encode()) % self.num_workers]

    def _send(self, ipc_name: str, obj: Any) -> None:
        # 中译：向指定 detokenizer worker 发送对象（目标是 detokenizer，故 is_tokenizer=False）。
        self.socket_mapping.send_output(ipc_name, obj, is_tokenizer=False)

    def event_loop(self):
        # 中译：主事件循环——从调度器收输出，按类型分流到对应的 detokenizer worker。
        while True:
            recv_obj = self.recv_from_scheduler.recv_pyobj()

            # FreezeGCReq must freeze every detokenizer process.
            # 中译：冻结 GC 请求必须作用于每个 detokenizer 进程，故广播给全部 worker。
            if isinstance(recv_obj, FreezeGCReq):
                for ipc in self.ipc_name_list:
                    self._send(ipc, recv_obj)
                continue

            # Single request: route by its own http_worker_ipc.
            # 中译：单条请求——按它自己的 http_worker_ipc 选 worker 后发送。
            if isinstance(recv_obj, BaseReq):
                assert (
                    recv_obj.http_worker_ipc is not None
                ), f"Single req {recv_obj.rid=} missing http_worker_ipc"
                self._send(self._pick(recv_obj.http_worker_ipc), recv_obj)
                continue

            # Batch request.
            # 中译：批量请求。
            if isinstance(recv_obj, BaseBatchReq):
                # Idle/no-op batch (rids=[]): broadcast to all detokenizers
                # 中译：空转（idle）批次（rids 为空）——广播给所有 detokenizer。
                if not recv_obj.rids:
                    for ipc in self.ipc_name_list:
                        self._send(ipc, recv_obj)
                    continue

                ipcs = recv_obj.http_worker_ipcs
                assert (
                    ipcs is not None
                    and len(ipcs) == len(recv_obj.rids)
                    and all(x is not None for x in ipcs)
                ), f"Batch req {recv_obj.rids=} has invalid http_worker_ipcs"

                # Split per-item and route each by its own ipc.
                # 中译：把批次按条拆开，每条按各自的 ipc 单独路由。
                for i, ipc_key in enumerate(ipcs):
                    one = _handle_output_by_index(recv_obj, i)
                    # 中译：若拆分后仍是原对象，说明该类型不支持拆分，属于不应发生的情况。
                    if one is recv_obj:
                        raise TypeError(f"Cannot split {type(recv_obj)}")
                    one.http_worker_ipcs = [ipc_key]
                    self._send(self._pick(ipc_key), one)
                continue

            raise ValueError(
                f"MultiDetokenizerRouter got unsupported type {type(recv_obj)}"
            )


def run_multi_detokenizer_router_process(
    ipc_name_list: List[str],
    server_args: ServerArgs,
    port_args: PortArgs,
):
    # 中译：MultiDetokenizerRouter 进程的入口函数。设置进程名/日志，创建路由器进入事件循环；
    #       出异常时清理套接字并给父进程发 SIGQUIT 触发整体退出。
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang::detokenizer_router")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    router = None
    try:
        router = MultiDetokenizerRouter(ipc_name_list, port_args)
        router.event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"MultiDetokenizerRouter hit an exception: {traceback}")
        if router is not None:
            router.socket_mapping.clear_all_sockets()
        parent_process.send_signal(signal.SIGQUIT)


class TokenizerWorker(TokenizerManager):
    """Tokenizer Worker in multi-http-worker mode

    中译：多 HTTP worker 模式下的 TokenizerManager 子类。每个 HTTP worker 进程跑一个
          TokenizerWorker，负责本进程的分词与请求生命周期管理，并通过路由器与调度器/
          detokenizer 通信。相比基类，它额外：向路由器注册自身 IPC、参与 pause/continue
          广播协议、给发出的请求打上 http_worker_ipc 以便结果能回送到本进程。
    """

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        setproctitle.setproctitle(f"sglang::tokenizer_worker:{os.getpid()}")
        # prevent init prefill bootstrapserver again
        # 中译：先临时把分离模式置为 "null"，避免基类 __init__ 里重复启动 prefill bootstrap 服务；
        #       构造完成后再恢复原始的 disaggregation_mode（见下方）。
        disaggregation_mode = server_args.disaggregation_mode
        server_args.disaggregation_mode = "null"
        super().__init__(server_args, port_args)

        # 中译：以进程 PID 作为 worker_id；记录本 worker 的 tokenizer IPC 名（结果回送地址）。
        self.worker_id = os.getpid()
        self.tokenizer_ipc_name = port_args.tokenizer_ipc_name

        # For PD disaggregtion
        # 中译：恢复并重建 PD 分离相关配置（模式与传输后端）。
        self.server_args.disaggregation_mode = disaggregation_mode
        self.disaggregation_mode = DisaggregationMode(
            self.server_args.disaggregation_mode
        )
        self.disaggregation_transfer_backend = TransferBackend(
            self.server_args.disaggregation_transfer_backend
        )

        # Register this worker with the router for pause/continue broadcasting
        # 中译：向路由器注册本 worker 的 IPC 名，使其能在 pause/continue 时被广播到。
        reg = TokenizerWorkerRegistration(worker_ipc_name=self.tokenizer_ipc_name)
        self.send_to_scheduler.send_pyobj(reg)

        # Future for awaiting pause/continue broadcast confirmation
        # 中译：用于等待 pause/continue 广播确认的 Future（由发起方持有，收到广播回执后置位）。
        self._pause_continue_future: Optional[asyncio.Future] = None

        # Register PauseContinueBroadcast in the result dispatcher so
        # handle_loop routes it to _handle_pause_continue_broadcast
        # 中译：把 PauseContinueBroadcast 注册进结果分发器，使反向 handle_loop 能把广播
        #       路由到 _handle_pause_continue_broadcast 处理。
        from sglang.utils import TypeBasedDispatcher

        self._result_dispatcher += TypeBasedDispatcher(
            [(PauseContinueBroadcast, self._handle_pause_continue_broadcast)]
        )

    async def pause_generation(self, obj: PauseGenerationReqInput):
        # 中译：发起「暂停生成」。创建 Future 后把请求发给路由器（由其广播给所有 worker，
        #       非 abort 模式还会转发给调度器），再等待广播确认。
        loop = asyncio.get_event_loop()
        self._pause_continue_future = loop.create_future()
        # Send to router which will broadcast to all workers
        # (router also handles forwarding to scheduler for non-abort modes)
        self.send_to_scheduler.send_pyobj(obj)
        await self._pause_continue_future

        if obj.mode == "abort":
            # Abort polling: only the originator checks its own lock state
            # 中译：abort 模式——只有发起方轮询自己的锁状态：反复 abort 全部请求，直到
            #       模型更新锁释放为止。
            while True:
                self.abort_request(abort_all=True)
                is_locked = await self.model_update_lock.is_locked()
                if not is_locked:
                    break
                await asyncio.sleep(1.0)

    async def continue_generation(self, obj: ContinueGenerationReqInput):
        # 中译：发起「继续生成」。流程与 pause 对称：创建 Future、发请求、等待广播确认。
        loop = asyncio.get_event_loop()
        self._pause_continue_future = loop.create_future()
        self.send_to_scheduler.send_pyobj(obj)
        await self._pause_continue_future

    def _handle_pause_continue_broadcast(self, obj: PauseContinueBroadcast):
        """Called from handle_loop when a broadcast arrives from the router.

        中译：当路由器发来的 pause/continue 广播到达时，由反向 handle_loop 调用此方法；
              它把实际状态应用工作丢到事件循环里异步执行。
        """
        loop = asyncio.get_event_loop()
        loop.create_task(self._apply_pause_continue_broadcast(obj))

    async def _apply_pause_continue_broadcast(self, obj: PauseContinueBroadcast):
        """Apply pause/continue state under the condition lock.

        中译：在条件锁（is_pause_cond）保护下应用 pause/continue 状态。继续时唤醒所有
              在该条件变量上等待的协程；若本 worker 是发起方，则顺带兑现其等待中的 Future。
        """
        async with self.is_pause_cond:
            if obj.is_pause:
                self.is_pause = True
            else:
                self.is_pause = False
                self.is_pause_cond.notify_all()

        # Resolve the pending future if this worker initiated the pause/continue
        # 中译：若本 worker 是该 pause/continue 的发起方，兑现其挂起的 Future 以解除等待。
        if self._pause_continue_future and not self._pause_continue_future.done():
            self._pause_continue_future.set_result(True)
            self._pause_continue_future = None

    def _attach_multi_http_worker_info(self, req: Union[BaseReq, BaseBatchReq]):
        # 中译：给即将发出的请求打上本 worker 的 IPC 名，使其结果日后能被路由器/detokenizer
        #       正确回送到本进程。单条写 http_worker_ipc，批量则为每条写同一 IPC 名。
        if isinstance(req, BaseReq):
            req.http_worker_ipc = self.tokenizer_ipc_name
        elif isinstance(req, BaseBatchReq):
            req.http_worker_ipcs = [self.tokenizer_ipc_name] * len(req.rids)
        else:
            raise ValueError(f"Unknown req type: {type(req)}")


async def print_exception_wrapper(func):
    """
    Sometimes an asyncio function does not print exception.
    We do another wrapper to handle the exception.

    中译：asyncio 协程出错时有时不会打印异常，这里再包一层来捕获并处理：记录堆栈、
          （若是路由器）转储崩溃前的请求，然后杀掉整个进程树并退出，避免静默失败。
    """
    try:
        await func()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"MultiTokenizerRouter hit an exception: {traceback}")
        # 中译：若该协程属于 MultiTokenizerRouter，崩溃前先把在途请求转储下来便于排查。
        if hasattr(func, "__self__") and isinstance(
            func.__self__, MultiTokenizerRouter
        ):
            func.__self__.dump_requests_before_crash()
        # 中译：杀掉整个进程树（含父进程）并退出，触发整体重启而非僵在半死状态。
        kill_process_tree(os.getpid(), include_parent=True)
        sys.exit(1)


def get_main_process_id() -> int:
    """Get the main process ID.

    Supports override via SGLANG_GRANIAN_PARENT_PID for workers whose
    multiprocessing parent PID differs from the shared-memory owner.

    中译：获取主进程 ID。支持用环境变量 SGLANG_GRANIAN_PARENT_PID 覆盖——适用于那些
          multiprocessing 父 PID 与「共享内存拥有者」不一致的 worker（如 Granian 部署场景）。
    """
    from sglang.srt.environ import envs

    override = envs.SGLANG_GRANIAN_PARENT_PID.get()
    if override is not None:
        return override
    return multiprocessing.current_process()._parent_pid


def write_to_shared_memory(obj, name: str) -> shared_memory.SharedMemory:
    """Write data to shared memory

    中译：把任意对象 pickle 序列化后写入命名共享内存（SHM）。若同名 SHM 已存在但容量不足，
          先 unlink 再按新大小重建；不存在则直接创建。返回该 SHM 句柄（由调用方负责关闭）。
    """
    serialized = pickle.dumps(obj)
    size = len(serialized)
    try:
        # Try to open existing shared memory
        # 中译：尝试打开已存在的同名共享内存。
        shm = shared_memory.SharedMemory(name=name)
        # If size is insufficient, close and recreate
        # 中译：若现有容量不足以放下新数据，则关闭、解除链接后按新大小重建。
        if shm.size < size:
            shm.close()
            shm.unlink()
            shm = shared_memory.SharedMemory(create=True, size=size, name=name)
    except FileNotFoundError:
        # If not present, create new shared memory
        # 中译：不存在同名 SHM 时，新建一块。
        shm = shared_memory.SharedMemory(create=True, size=size, name=name)

    shm.buf[:size] = serialized
    return shm


def read_from_shared_memory(name: str) -> Any:
    """Read data from shared memory

    中译：从命名共享内存读取并 unpickle 还原对象。未找到对应 SHM 时抛 FileNotFoundError。
    """
    try:
        shm = shared_memory.SharedMemory(name=name)
        data = pickle.loads(bytes(shm.buf))
        shm.close()
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Shared memory {name} not found")


def write_data_for_multi_tokenizer(
    port_args: PortArgs, server_args: ServerArgs, scheduler_info: Dict
):
    """Write args information to share memory for multi-tokenizer

    中译：把多 tokenizer 模式启动所需的参数（port_args、server_args、scheduler_info）写入
          以当前进程 PID 命名的共享内存，供各 TokenizerWorker 子进程读取并据此初始化。
    """
    # get main process ID
    # 中译：取主进程与当前进程 PID 并记录日志（仅用于诊断）。
    main_pid = get_main_process_id()
    current_pid = os.getpid()
    logger.info(f"main process ID: {main_pid}, current process ID: {current_pid}")
    args = (port_args, server_args, scheduler_info)
    args_shm = write_to_shared_memory(args, f"multi_tokenizer_args_{current_pid}")
    args_shm.close()

    return args_shm


class SenderWrapper:
    """对「发往调度器的套接字」的轻量包装。

    中译：在 send_pyobj 时自动给 BaseReq 类型的请求补上 http_worker_ipc（本 worker 的
          tokenizer IPC 名），使其结果日后能被正确回送，调用方无需每处手动设置。
    """

    def __init__(self, port_args: PortArgs, send_to_scheduler: zmq.Socket):
        self.port_args = port_args
        self.send_to_scheduler = send_to_scheduler

    def send_pyobj(self, obj):
        # 中译：发送前，若是 BaseReq 则自动打上本 worker 的 IPC 名，再转发给调度器套接字。
        if isinstance(obj, BaseReq):
            obj.http_worker_ipc = self.port_args.tokenizer_ipc_name
        self.send_to_scheduler.send_pyobj(obj)
