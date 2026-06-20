# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import logging
import time
import uuid
from array import array
from typing import TYPE_CHECKING, Dict, Optional

from sglang.srt.managers.io_struct import (
    CloseSessionReqInput,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.utils.common import log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

logger = logging.getLogger(__name__)


# ============================================================================
# 本文件实现 SGLang 的「会话（Session）」管理，用于支撑多轮对话（multi-turn）：
# 让后一轮请求能在前一轮请求的上下文（prompt + 已生成输出）之上继续追加，
# 从而复用 KV cache、避免每轮都重新 prefill 整段历史。
#
# 三个核心类的职责：
#   - SessionReqNode：把同一会话内的多个请求(Req)组织成一棵「请求树」的节点，
#                     父子关系表示「某请求是在另一请求基础上追加/分支而来」。
#   - Session       ：单个会话的状态机，负责把「上一轮请求」与「本轮新输入」拼接成
#                     一个新的 Req，并管理流式(streaming)会话的在途状态与提交点。
#   - SessionController：所有会话的集合管理者，负责开启/关闭/超时回收会话。
# ============================================================================


class SessionReqNode:
    """会话请求树(request tree)中的一个节点，封装单个请求 `Req`。

    同一会话内的多轮请求通过 parent/children 形成一棵树：
    - 普通(非流式)会话下，后一轮请求挂在它所追加的那一轮请求(last_req_node)之下，
      形成链/树，便于 replace 时连同其后续分支(children)一起清理。
    - 节点持有对 `Req` 的引用，并提供级联中止(abort)与清理(clear)能力。
    """

    def __init__(
        self,
        req: Req,
        parent: Optional[SessionReqNode] = None,
        children=None,
    ):
        # 中译：本节点对应的请求对象。
        self.req = req
        # 中译：父节点；若为 None 则是树根（会话的第一轮请求）。
        self.parent = parent
        # 中译：把自己登记到父节点的 children 列表，建立树形父子关系。
        if parent is not None:
            parent.children.append(self)
        # 中译：子节点列表（在本请求基础上继续追加的后续请求）。
        self.children = [] if not children else children

    def clear_children(self, req_dict):
        """递归清理本节点的所有子孙节点，并清空 children（保留本节点自身）。

        用于 replace 场景：替换某一轮请求时，其之上派生的所有后续轮次都失效。
        req_dict 是会话的 `req_nodes`（rid -> SessionReqNode），需同步删除。
        """
        for req_node in self.children:
            req_node.clear(req_dict)
        self.children = []

    def clear(self, req_dict):
        """递归清理本节点及其所有子孙：中止未完成的请求并从 req_dict 中移除。"""
        # 中译：先深度优先清理所有子孙节点。
        for req_node in self.children:
            req_node.clear(req_dict)

        # 中译：若该请求尚未结束，则标记为「中止(abort)」，让调度器据此终止它。
        if self.req.finished_reason is None:
            self.req.to_finish = FINISH_ABORT()
        # 中译：从会话的 rid->node 映射中删除本请求。
        del req_dict[self.req.rid]

    def abort(self):
        """仅中止本节点对应的请求（不触及子孙），若它还未结束。"""
        if self.req.finished_reason is None:
            self.req.to_finish = FINISH_ABORT()

    def __str__(self):
        """以缩进文本形式打印整棵请求树，便于调试观察会话的多轮结构。"""
        return self._str_helper(self.req.rid)

    def _str_helper(self, prefix=""):
        # 中译：递归构造树形文本：叶子节点直接换行；否则首个孩子接在同一行用
        #       " -- " 连接，其余孩子另起一行用 " \- " 表示分支。
        if len(self.children) == 0:
            return prefix + "\n"
        else:
            origin_prefix = prefix
            prefix += " -- " + self.children[0].req.rid
            ret = self.children[0]._str_helper(prefix)
            for child in self.children[1:]:
                prefix = " " * len(origin_prefix) + " \\- " + child.req.rid
                ret += child._str_helper(prefix)
            return ret


class Session:
    """单个会话的状态机：把多轮请求按上下文拼接，并管理其生命周期。

    两种模式：
    - 普通模式：用 `req_nodes` 维护一棵请求树，支持 replace / offset /
      drop_previous_output 等「重写历史」的操作（通过拷贝方式拼接 token）。
    - 流式模式(streaming)：同一时刻只允许一个在途(inflight)请求，且只支持
      简单追加(append)。为复用显存，会「原地」复用上一轮请求的 token 数组
      （见 _share_token_arrays），并用 committed_* 长度作为可回滚的提交点。
    """

    def __init__(
        self,
        capacity_of_str_len: int,
        session_id: Optional[str] = None,
        streaming: bool = False,
        timeout: Optional[float] = None,
    ):
        # 中译：会话 id；未指定则随机生成。
        self.session_id = session_id if session_id is not None else uuid.uuid4().hex
        # 中译：会话允许累积的字符串长度上限。
        self.capacity_of_str_len = capacity_of_str_len
        # 中译：是否为流式会话（仅允许单个在途请求、仅支持简单追加）。
        self.streaming = streaming
        # 中译：空闲超时时间（秒）；None 表示永不超时。
        self.timeout = timeout
        # 中译：最近一次活跃时间（单调时钟），用于超时判定。
        self.last_active_time: float = time.monotonic()
        # 中译：会话内请求树节点表：rid -> SessionReqNode。
        self.req_nodes: Dict[str, SessionReqNode] = {}
        # 中译：标记「关闭请求到达时仍有在途请求」，需延迟到请求完成后再释放。
        self.close_on_finish: bool = False
        # 中译：流式会话当前是否有一个在途(未完成)请求。
        self._inflight: bool = False
        # Token-array lengths of last_req as of its finish_req. The share path
        # appends speculatively beyond these; only finish_req confirms them, so
        # _share_token_arrays trims back first (heals aborted turns).
        # 中译：以下三个 committed_* 记录上一轮请求在 finish_req 时刻各 token 数组的
        #       「已确认长度」。流式原地复用路径会在其后追加未确认 token，只有
        #       finish_req 才确认；因此下一轮 _share_token_arrays 会先裁回到这些
        #       长度，从而修复中途 abort 的那一轮残留的 token。
        # Token-array lengths of last_req as of its finish_req. The share path
        # appends speculatively beyond these; only finish_req confirms them, so
        # _share_token_arrays trims back first (heals aborted turns).
        self.committed_origin_len: Optional[int] = None
        self.committed_unpadded_len: Optional[int] = None
        self.committed_fill_len: Optional[int] = None

    def is_timed_out(self) -> bool:
        """判断会话是否已空闲超时（超过 timeout 未活跃）。"""
        if self.timeout is None:
            return False
        return time.monotonic() - self.last_active_time > self.timeout

    @staticmethod
    def _strip_bos_token(req: TokenizedGenerateReqInput, tokenizer) -> None:
        """Trim a leading BOS on an appended turn; shift mm offsets to match."""
        # 中译：追加轮次的输入不应再带句首 BOS（因为它接在上一轮之后），
        #       若首 token 是 BOS 则去掉；同时多模态 offsets 需随之前移 1。
        if not (
            tokenizer is not None
            and req.input_ids
            and req.input_ids[0] == tokenizer.bos_token_id
        ):
            return
        # 中译：去掉首位 BOS token。
        req.input_ids = req.input_ids[1:]
        if req.mm_inputs:
            for item in req.mm_inputs.mm_items:
                if item.offsets:
                    if any(s == 0 for s, _ in item.offsets):
                        logging.warning(
                            "mm_item offset starts at 0 (BOS position), "
                            "clamping to 0 after BOS strip"
                        )
                    item.offsets = [
                        (max(0, s - 1), max(0, e - 1)) for s, e in item.offsets
                    ]

    def _share_token_arrays(self, last_req: Req, new_input_ids):
        """Plain streaming append: reuse last_req's token arrays in place.

        Trims each array back to its committed length first — an earlier turn
        may have appended its tokens and then aborted before finish_req, and
        req_nodes still points at last_req, so anything beyond the committed
        lengths is unconfirmed. Then extends with last turn's output and the
        new input. Returns (input_ids, input_ids_unpadded, carry_fill);
        carry_fill (== the new origin) spares the first fill_ids rebuild.

        中译：流式简单追加的「原地复用」路径（避免拷贝整段历史，省显存与拷贝开销）：
          1) 先把每个 token 数组裁回到 committed_* 长度——因为上一轮可能追加了 token 后
             又在 finish_req 前 abort 了，超出已确认长度的部分都是未确认的，需丢弃。
          2) 再依次拼上「上一轮的输出」与「本轮新输入」。
          3) 返回 (input_ids, input_ids_unpadded, carry_fill)；carry_fill 可省去首次 fill_ids 重建。
        """
        # 中译：取上一轮输出（受 max_new_tokens 限制）作为要拼接的「上轮生成结果」。
        out_tail = last_req.output_ids[: last_req.sampling_params.max_new_tokens]

        # 中译：原地复用 origin_input_ids，先裁剪掉未确认的尾部。
        input_ids = last_req.origin_input_ids
        del input_ids[self.committed_origin_len :]
        # 中译：若 unpadded 数组与 origin 是同一个对象，直接复用；否则同样裁剪其尾部。
        if last_req.origin_input_ids_unpadded is input_ids:
            input_ids_unpadded = input_ids
        else:
            input_ids_unpadded = last_req.origin_input_ids_unpadded
            del input_ids_unpadded[self.committed_unpadded_len :]

        # 中译：full_untruncated_fill_ids 是预构建好的 fill_ids；尝试增量复用以省重建。
        carry_fill = last_req.full_untruncated_fill_ids
        if (
            not isinstance(carry_fill, array)
            or carry_fill is input_ids
            or carry_fill is input_ids_unpadded
        ):
            # Unexpected type or aliased with an origin array (extending it
            # below would double-append): let _refresh_fill_ids rebuild.
            # 中译：类型不符预期，或与 origin 数组是同一对象（下面再 extend 会重复追加），
            #       则放弃复用，置 None 让后续 _refresh_fill_ids 重建。
            carry_fill = None
        else:
            # 中译：裁剪 fill_ids 到已确认长度。
            del carry_fill[self.committed_fill_len :]
            # 中译：baked = fill_ids 中已烘焙(超出 origin 的)部分长度，即已含多少上轮输出。
            baked = len(carry_fill) - len(input_ids)
            if 0 <= baked <= len(out_tail):
                # 中译：补齐剩余的上轮输出与本轮新输入，完成增量复用。
                carry_fill.extend(out_tail[baked:])
                carry_fill.extend(new_input_ids)
            else:
                # 中译：烘焙长度不在合理区间，放弃复用，交由后续重建。
                carry_fill = None

        # 中译：把「上轮输出 + 本轮输入」追加到 origin 数组，得到新一轮的完整输入。
        input_ids.extend(out_tail)
        input_ids.extend(new_input_ids)
        # 中译：若 unpadded 是独立数组，同步追加。
        if input_ids_unpadded is not input_ids:
            input_ids_unpadded.extend(out_tail)
            input_ids_unpadded.extend(new_input_ids)
        return input_ids, input_ids_unpadded, carry_fill

    @staticmethod
    def _concat_token_arrays(
        last_req: Req, req: TokenizedGenerateReqInput, session_params
    ):
        """Copy-based assembly for replace/offset/drop_previous_output turns.

        中译：面向 replace / offset / drop_previous_output 等「重写历史」轮次的「拷贝拼接」。
        与原地复用不同，这里必须生成新数组，不能修改 last_req 的原数组。
        三种调整：
          - 默认：origin + 上轮输出 + 本轮输入；
          - drop_previous_output：丢弃上轮输出，仅保留 origin；
          - offset：从指定 offset 处截断并接上本轮输入（实现「从某位置重写」）。
        """
        # 中译：上轮输出（受 max_new_tokens 限制）。
        out_tail = last_req.output_ids[: last_req.sampling_params.max_new_tokens]

        # 中译：默认＝上轮输入 + 上轮输出（拷贝生成新数组）。
        input_ids = last_req.origin_input_ids + out_tail
        # 中译：drop_previous_output：不要上轮输出，只保留上轮输入。
        if session_params.drop_previous_output:
            input_ids = last_req.origin_input_ids[:]
        # 中译：offset：从 offset 处截断后拼接本轮输入；否则直接追加本轮输入。
        if session_params.offset and session_params.offset != 0:
            input_ids = input_ids[: session_params.offset] + req.input_ids
        else:
            input_ids += req.input_ids

        input_ids_unpadded = last_req.origin_input_ids_unpadded + out_tail
        if session_params.drop_previous_output:
            input_ids_unpadded = last_req.origin_input_ids_unpadded[:]
        if session_params.offset and session_params.offset != 0:
            input_ids_unpadded = (
                input_ids_unpadded[: session_params.offset] + req.input_ids
            )
        else:
            input_ids_unpadded += req.input_ids
        return input_ids, input_ids_unpadded

    def create_req(
        self,
        req: TokenizedGenerateReqInput,
        tokenizer,
        vocab_size: int,
        eos_token_ids=None,
    ):
        """会话的核心方法：把「本轮新请求」与「上一轮上下文」拼接成一个新的 Req。

        总体流程：
          1) 根据模式(streaming / replace / 普通追加)定位上一轮请求 last_req，
             并校验是否合法（非法则置 abort 标记，稍后构造一个直接中止的 Req）。
          2) 拼接 token 数组：流式简单追加走「原地复用」(_share_token_arrays)，
             其余走「拷贝拼接」(_concat_token_arrays)。
          3) 用拼好的 input_ids 构造新的 Req，并继承多模态输入等字段。
          4) 按模式更新会话状态（流式置 _inflight；普通模式挂入请求树）。
        """
        assert req.session_params is not None
        # 中译：刷新活跃时间，避免被超时回收。
        self.last_active_time = time.monotonic()
        session_params = req.session_params

        # 中译：last_req_node/last_req 为「要在其上追加」的上一轮请求；
        #       abort/abort_message 记录是否需要直接中止本请求及原因。
        last_req_node = None
        last_req = None
        abort = False
        abort_message = ""
        if self.streaming:
            # Streaming sessions: only simple appends allowed; reject otherwise.
            # 中译：流式会话只允许简单追加；replace/offset/drop_previous_output 一律拒绝，
            #       且同一时刻不能有第二个在途请求。
            if self._inflight:
                abort = True
                abort_message = "Streaming session already has an active request."
            elif session_params.replace:
                abort = True
                abort_message = "Streaming sessions do not support replace."
            elif session_params.drop_previous_output:
                abort = True
                abort_message = (
                    "Streaming sessions do not support drop_previous_output."
                )
            elif session_params.offset and session_params.offset != 0:
                abort = True
                abort_message = "Streaming sessions do not support offset."
            elif self.req_nodes:
                assert len(self.req_nodes) == 1
                # Peek (don't pop) the single req_node. req_nodes is updated
                # only in finish_req after the request completes successfully.
                # 中译：流式会话至多保留一个已完成的请求节点；这里只「读取」不弹出，
                #       因为 req_nodes 仅在 finish_req(请求成功完成)时才更新。
                [last_req_node] = self.req_nodes.values()
                last_req = last_req_node.req
        elif session_params.replace:
            # 中译：replace 模式——用本轮请求替换历史中的某一轮(或全部)。
            if session_params.rid is None:
                # 中译：未指定 rid——清空整棵请求树（从头替换）。
                for _, req_node in self.req_nodes.items():
                    req_node.clear(self.req_nodes)
            else:
                if session_params.rid not in self.req_nodes:
                    abort = True
                    abort_message = "Invalid request session id"
                else:
                    # 中译：替换指定 rid 的那一轮：中止该请求并清掉其后所有派生分支，
                    #       但保留它本身作为本轮追加的基础。
                    last_req_node = self.req_nodes[session_params.rid]
                    last_req_node.abort()
                    last_req = last_req_node.req
                    last_req_node.clear_children(self.req_nodes)
        else:
            # 中译：普通追加模式——在指定 rid 的请求之后继续。
            if session_params.rid is not None:
                if session_params.rid not in self.req_nodes:
                    abort = True
                    abort_message = "Invalid request session id"
                else:
                    last_req_node = self.req_nodes[session_params.rid]
                    last_req = last_req_node.req
                    # 中译：只能在「已完成」的请求之上追加，否则上下文不完整，拒绝。
                    if not last_req.finished():
                        abort = True
                        abort_message = "Session request is appending to a request that hasn't finished."
                        logging.warning(abort_message)

        # 中译：carry_fill 用于把上一轮预构建好的 fill_ids 直接传递给新请求，
        #       省去一次 fill_ids 重建（仅原地复用路径可能产生）。
        carry_fill = None
        if last_req is not None:
            self._strip_bos_token(req, tokenizer)
            # In-place sharing is only safe for the plain streaming append:
            # streaming sessions allow a single inflight request, last_req has
            # finished, and the committed_* lengths recorded by finish_req let
            # _share_token_arrays trim away tokens appended by an aborted turn.
            # offset / drop_previous_output rewrite history and must copy.
            can_share_token_arrays = (
                self.streaming
                and self.committed_origin_len is not None
                and not session_params.drop_previous_output
                and not (session_params.offset and session_params.offset != 0)
            )
            if can_share_token_arrays:
                input_ids, input_ids_unpadded, carry_fill = self._share_token_arrays(
                    last_req, req.input_ids
                )
            else:
                input_ids, input_ids_unpadded = self._concat_token_arrays(
                    last_req, req, session_params
                )
        else:
            input_ids = req.input_ids
            input_ids_unpadded = req.input_ids

        new_req = Req(
            rid=req.rid,
            origin_input_text=None,
            origin_input_ids=input_ids,
            origin_input_ids_unpadded=input_ids_unpadded,
            sampling_params=req.sampling_params,
            lora_id=req.lora_id,
            session=self,
            custom_logit_processor=req.custom_logit_processor,
            stream=req.stream,
            return_logprob=req.return_logprob,
            top_logprobs_num=req.top_logprobs_num,
            token_ids_logprob=req.token_ids_logprob,
            vocab_size=vocab_size,
            eos_token_ids=eos_token_ids,
            require_reasoning=req.require_reasoning,
            return_hidden_states=req.return_hidden_states,
            return_routed_experts=req.return_routed_experts,
            routed_experts_start_len=req.routed_experts_start_len,
            priority=req.priority,
            routing_key=req.routing_key,
            extra_key=req.extra_key,
            http_worker_ipc=req.http_worker_ipc,
            time_stats=req.time_stats,
        )
        if last_req is not None:
            new_req.multimodal_inputs = last_req.multimodal_inputs
        new_req.tokenizer = tokenizer
        if carry_fill is not None:
            new_req.full_untruncated_fill_ids = carry_fill

        if abort:
            # 中译：前面校验失败——构造一个直接进入中止状态的请求返回。
            new_req.set_finish_with_abort(abort_message)
        elif self.streaming:
            # req_nodes is NOT updated here — finish_req() handles it.
            # 中译：流式模式此处只置在途标记；req_nodes 留给 finish_req 在成功后更新。
            self._inflight = True
        else:
            # 中译：普通模式——把新请求作为 last_req_node 的子节点挂入请求树。
            new_req_node = SessionReqNode(new_req, last_req_node)
            self.req_nodes[req.rid] = new_req_node

        return new_req

    def finish_req(self, req):
        """Update req_nodes after a streaming request finishes successfully."""
        # 中译：流式请求成功完成后调用——清在途标记，并用本请求替换掉旧的节点。
        self._inflight = False
        if self.req_nodes:
            # 中译：解除上一轮请求与会话的关联并清空，使其 KV/资源可被正常回收。
            [prev_node] = self.req_nodes.values()
            prev_node.req.session = None
            self.req_nodes.clear()
        self.req_nodes[req.rid] = SessionReqNode(req)
        # Confirm this req's token arrays as the session's rollback point.
        # 中译：把本请求各 token 数组的当前长度「确认」为会话的回滚点，
        #       供下一轮 _share_token_arrays 裁剪未确认 token 时使用。
        self.committed_origin_len = len(req.origin_input_ids)
        self.committed_unpadded_len = len(req.origin_input_ids_unpadded)
        self.committed_fill_len = len(req.full_untruncated_fill_ids)

    def abort_req(self):
        """Clear inflight flag on abort (req_nodes stays unchanged)."""
        # 中译：请求中止时仅清除在途标记，req_nodes 维持不变（提交点不前移）。
        self._inflight = False


class SessionController:
    """所有会话的集合管理者：负责开启/关闭/超时回收会话，并与前缀缓存联动。

    持有所有活跃 Session（`sessions`）与前缀缓存 `tree_cache`；会话关闭时
    需通过 tree_cache.release_session 释放其占用的 KV/缓存资源。对于仍有在途
    请求的会话，采用「延迟关闭」，避免误释放正在解码的 KV 内存。
    """

    def __init__(self, tree_cache: BasePrefixCache):
        # 中译：活跃会话表：session_id -> Session。
        self.sessions: Dict[str, Session] = {}
        # 中译：上次执行超时回收(reap)的时间，用于限频。
        self._last_reap_time: float = 0.0
        # 中译：前缀缓存，会话关闭时由它释放该会话占用的 KV 资源。
        self.tree_cache = tree_cache

    def __contains__(self, session_id: str) -> bool:
        """支持 `session_id in controller` 语法。"""
        return session_id in self.sessions

    def get(self, session_id: str) -> Optional[Session]:
        """按 id 取会话，不存在返回 None。"""
        return self.sessions.get(session_id)

    def open(self, recv_req: OpenSessionReqInput) -> OpenSessionReqOutput:
        """开启一个新会话；id 已存在或为空则失败。返回是否成功及会话 id。"""
        session_id = recv_req.session_id
        if session_id in self.sessions:
            logger.warning(f"session id {session_id} already exist, cannot open.")
            return OpenSessionReqOutput(session_id, False)
        elif session_id is None:
            logger.warning("session id is None, cannot open.")
            return OpenSessionReqOutput(session_id, False)
        else:
            self.sessions[session_id] = Session(
                recv_req.capacity_of_str_len,
                session_id,
                streaming=bool(recv_req.streaming),
                timeout=recv_req.timeout,
            )
            log_info_on_rank0(
                logger, f"Session opened: {session_id} (active={len(self.sessions)})"
            )
            return OpenSessionReqOutput(session_id, True)

    def close(self, recv_req: CloseSessionReqInput):
        """关闭会话的对外入口：会话不存在则告警，否则走 _close 释放逻辑。"""
        session_id = recv_req.session_id
        if session_id not in self.sessions:
            logger.warning(f"session id {session_id} does not exist, cannot delete.")
        else:
            self._close(session_id)

    def _close(self, session_id: str):
        """真正执行会话释放：若仍有在途请求则延迟关闭，否则释放多模态特征与 KV 资源。"""
        session = self.sessions[session_id]
        req = None
        has_unfinished_request = False
        if session.streaming and session._inflight:
            has_unfinished_request = True
        elif session.streaming and session.req_nodes:
            assert len(session.req_nodes) == 1
            [last_node] = session.req_nodes.values()
            req = last_node.req
            if not req.finished():
                has_unfinished_request = True

        if has_unfinished_request:
            # An in-flight request is still decoding on this session's KV
            # memory. Freeing now would corrupt the scheduler. Mark the
            # session for deferred cleanup: the request keeps its session
            # reference so cache_finished_req takes the streaming path,
            # and we schedule release_session for after it completes.
            # 中译：仍有请求在该会话的 KV 内存上解码，此时释放会破坏调度器。
            #       标记为「完成后再关闭」：请求保留 session 引用以走流式清理路径，
            #       等它完成后再由 maybe_reap 触发真正的 release_session。
            session.close_on_finish = True
            logger.info(
                "Deferring session close for %s (unfinished request)",
                session_id,
            )
            return

        # No owning request -- safe to release immediately.
        if session.streaming and session.req_nodes:
            req = next(iter(session.req_nodes.values())).req
            req.session = None

        # Release multimodal features held by session requests.
        # Session reqs skip the normal mm cleanup path (scheduler and
        # output_processor) so features stay alive until the session closes.
        # 中译：释放会话请求持有的多模态特征。会话请求会跳过正常的 mm 清理
        #       路径（调度器与 output_processor），因此特征会一直存活到会话关闭。
        #       用 seen_mm 去重，避免多个请求共享同一 mm 对象时重复释放。
        seen_mm = set()
        for node in session.req_nodes.values():
            mm = node.req.multimodal_inputs
            if mm is not None and id(mm) not in seen_mm:
                seen_mm.add(id(mm))
                mm.release_features()
            node.req.multimodal_inputs = None

        # 中译：通知前缀缓存释放该会话占用的 KV 资源，并从活跃会话表中删除。
        self.tree_cache.release_session(session_id)
        del self.sessions[session_id]
        log_info_on_rank0(
            logger, f"Session closed: {session_id} (active={len(self.sessions)})"
        )

    def maybe_reap(self, now: float, interval: float = 1.0):
        """周期性回收（默认每秒一次）：处理延迟关闭的会话与超时会话。"""
        # reap sessions every second
        # 中译：限频——距上次回收超过 interval 秒才执行。
        if now - self._last_reap_time > interval:
            self._last_reap_time = now

            # Finish deferred closes for sessions whose requests completed.
            # 中译：找出「已标记延迟关闭且其请求已全部完成」的会话。
            pending = [
                sid
                for sid, session in self.sessions.items()
                if session.close_on_finish and self._all_requests_finished(session)
            ]
            for sid in pending:
                log_info_on_rank0(
                    logger, f"Deferred close ready for session {sid}, releasing."
                )
                # Reset close_on_finish so _close proceeds with the release.
                # 中译：复位 close_on_finish，使 _close 这次走真正的释放分支而非再次延迟。
                self.sessions[sid].close_on_finish = False
                self._close(sid)

            # 中译：再关闭所有已空闲超时的会话。
            timed_out = [
                sid for sid, session in self.sessions.items() if session.is_timed_out()
            ]
            for sid in timed_out:
                log_info_on_rank0(logger, f"Session {sid} timed out, closing.")
                self._close(sid)

    @staticmethod
    def _all_requests_finished(session: Session) -> bool:
        """判断会话内所有请求是否都已结束（无请求也视为已完成）。"""
        if not session.req_nodes:
            return True
        return all(node.req.finished() for node in session.req_nodes.values())

    @staticmethod
    def adjust_mm_offsets(recv_req: TokenizedGenerateReqInput, req: Req, image_inputs):
        """会话请求下，按前缀长度修正多模态输入的 offsets。

        中译：Session.create_req 会把之前的上下文 prepend 到 origin_input_ids 前面，
        因此来自本轮新 prompt 的多模态 offsets 需整体后移一个前缀长度。
        """
        # For session requests, adjust mm_inputs offsets by the prefix length.
        # Session.create_req prepends previous context to origin_input_ids,
        # so offsets from the new prompt need to be shifted.
        # 中译：若本轮输入长度 >= 拼接后的 origin，说明没有前缀，无需调整。
        if len(recv_req.input_ids) >= len(req.origin_input_ids):
            return
        # 中译：前缀长度 = 拼接后总长 - 本轮新输入长。
        prefix_len = len(req.origin_input_ids) - len(recv_req.input_ids)
        for mm_item in image_inputs.mm_items:
            if mm_item.offsets:
                mm_item.offsets = [
                    (start + prefix_len, end + prefix_len)
                    for start, end in mm_item.offsets
                ]
