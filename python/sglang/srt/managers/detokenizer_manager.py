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
"""DetokenizerManager is a process that detokenizes the token ids.

中译：DetokenizerManager 是一个独立进程，负责把调度器（Scheduler）产出的 token id
      反向解码（detokenize）为可读文本，再转发给 TokenizerManager / HTTP 工作进程。
      核心难点是「增量解码（incremental decoding）」：流式输出时需要只发送本次新增的
      文本，同时正确处理多字节 UTF-8 字符被拆在不同 token 间的边界情况。
"""

import dataclasses
import logging
import os
import signal
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple, Union

import psutil
import pybase64
import setproctitle
import torch
import zmq

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    BatchEmbeddingOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
    ConfigureLoggingReq,
    FreezeGCReq,
)
from sglang.srt.managers.multi_tokenizer_mixin import MultiHttpWorkerDetokenizerMixin
from sglang.srt.observability.cpu_monitor import start_cpu_monitor_thread
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import configure_logger, freeze_gc, kill_itself_when_parent_died
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.srt.utils.network import get_zmq_socket
from sglang.srt.utils.patch_tokenizer import decode_without_hf_kwargs
from sglang.srt.utils.watchdog import Watchdog
from sglang.utils import (
    TypeBasedDispatcher,
    find_printable_text,
    get_exception_traceback,
)

logger = logging.getLogger(__name__)

# Maximum number of request states that detokenizer can hold. When exceeded,
# oldest request states will be evicted. Default: 65536 (1<<16).
# For more details, see: https://github.com/sgl-project/sglang/issues/2812
# Use power of 2 values for better memory allocation.
# 中译：detokenizer 可保持的请求状态（DecodeStatus）最大数量，超过后淘汰最旧的状态。
#       默认 65536（1<<16）；取 2 的幂便于内存分配。若请求被提前淘汰会报错（见下方增量解码），
#       可通过环境变量 SGLANG_DETOKENIZER_MAX_STATES 调大。
DETOKENIZER_MAX_STATES = int(os.environ.get("SGLANG_DETOKENIZER_MAX_STATES", 1 << 16))


@dataclasses.dataclass
class DecodeStatus:
    """Store the status of incremental decoding.

    中译：保存单个请求「增量解码」的进度状态。
    几个 offset 是理解增量解码的关键（量词皆为 token 下标，除 sent_offset 为字符下标）：
    - decode_ids：到目前为止累积的全部 token id。
    - surr_offset：「环绕上下文（surrounding）」起点。解码时多带一段前文（surr..read）作为
      上下文，以保证跨 token 边界的字符能正确拼接。
    - read_offset：「已解码读取」起点。[surr_offset, read_offset) 是上下文，
      [surr_offset, len(decode_ids)) 是本次要解码的全部。
    - sent_offset：已发送给下游的「字符」偏移，用于流式只发送新增部分。

    为减少反复字符串拼接开销，已提交文本用 decoded_text + decoded_text_chunks 两部分维护：
    新文本先追加到 chunks 列表，需要完整文本时再一次性 join 合并。
    """

    decoded_text: str  # 已提交（已合并）的解码文本
    decode_ids: List[int]  # 累积的全部 token id
    surr_offset: int  # 环绕上下文起点（token 下标）
    read_offset: int  # 已解码读取起点（token 下标）
    # Offset that's sent to tokenizer for incremental update.
    # 中译：已发送给下游的字符偏移（注意是「字符」下标，不是 token 下标）。
    sent_offset: int = 0
    decoded_text_len: int = dataclasses.field(init=False)  # decoded_text + chunks 的总字符长度
    decoded_text_chunks: List[str] = dataclasses.field(default_factory=list)  # 未合并的新文本片段

    def __post_init__(self):
        # 中译：初始化时记录初始文本长度，后续 append 时增量维护，避免反复 len() 整串。
        self.decoded_text_len = len(self.decoded_text)

    def append_decoded_text(self, text: str):
        # 中译：追加一段新文本（暂存到 chunks，不立即拼接），并同步更新总长度。
        if text:
            self.decoded_text_chunks.append(text)
            self.decoded_text_len += len(text)

    def get_decoded_text(self) -> str:
        # 中译：取完整的已解码文本——此时才把暂存的 chunks 一次性 join 到 decoded_text 并清空。
        if self.decoded_text_chunks:
            self.decoded_text += "".join(self.decoded_text_chunks)
            self.decoded_text_chunks.clear()
        return self.decoded_text


class DetokenizerManager(MultiHttpWorkerDetokenizerMixin):
    """DetokenizerManager is a process that detokenizes the token ids.

    中译：去 token 化管理器。从调度器收取 token id 批次，解码为文本后转发给下游。
          通过 ZMQ 与上下游进程通信，用基于类型的分发器（TypeBasedDispatcher）路由不同消息。
    """

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        # Init inter-process communication
        # 中译：初始化进程间通信（ZMQ 套接字）。
        self.init_ipc_channels(port_args, server_args)

        # Init tokenizer
        # 中译：初始化分词器（用于反向解码）。
        self.init_tokenizer(server_args)

        # Init running status
        # 中译：初始化运行时状态（解码状态字典、看门狗、监控等）。
        self.init_running_status(server_args)

        # Init dispatcher
        # 中译：初始化请求分发器（按消息类型路由到各 handler）。
        self.init_request_dispatcher()

    def init_ipc_channels(self, port_args: PortArgs, server_args: ServerArgs):
        # 中译：初始化进程间通信通道（ZMQ）。
        #       recv_from_scheduler（PULL）从调度器拉取待解码的批次；
        #       send_to_tokenizer（PUSH）把解码结果推回 TokenizerManager。
        context = zmq.Context(2)
        self.recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, port_args.detokenizer_ipc_name, True
        )
        # In multi-tokenizer mode, results are pushed back to each TokenizerWorker
        # directly via SocketMapping inside multi_http_worker_event_loop, so the
        # single send_to_tokenizer socket is unused.
        if server_args.tokenizer_worker_num == 1:
            self.send_to_tokenizer = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_ipc_name, False
            )

    def init_tokenizer(self, server_args: ServerArgs):
        # 中译：初始化分词器（用于把 token id 反向解码为文本）。
        #       若启用了 skip_tokenizer_init（调用方自行处理 token）则不加载，置为 None。
        if server_args.skip_tokenizer_init:
            self.tokenizer = None
        else:
            self.tokenizer = get_tokenizer(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                tokenizer_backend=server_args.tokenizer_backend,
            )

    def init_running_status(self, server_args: ServerArgs):
        # 中译：初始化运行时状态。
        # decode_status：限容量字典，保存各请求的增量解码进度（超容淘汰最旧）。
        self.decode_status = LimitedCapacityDict(capacity=DETOKENIZER_MAX_STATES)
        # disable_tokenizer_batch_decode：是否禁用批量解码（避免某些边界问题）。
        self.disable_tokenizer_batch_decode = server_args.disable_tokenizer_batch_decode
        # is_tool_call_parser_gpt_oss：是否使用 gpt-oss 工具调用解析器（影响停止符裁剪）。
        self.is_tool_call_parser_gpt_oss = server_args.tool_call_parser == "gpt-oss"

        # 中译：软看门狗（soft watchdog），用于检测该进程是否卡死（soft 模式下只告警不直接杀进程）。
        self.soft_watchdog = Watchdog.create(
            debug_name="DetokenizerManager",
            watchdog_timeout=server_args.soft_watchdog_timeout,
            soft=True,
            test_stuck_time=envs.SGLANG_TEST_STUCK_DETOKENIZER.get(),
        )

        # 中译：启用指标采集时，启动一个 CPU 监控线程上报 detokenizer 进程的 CPU 使用。
        if server_args.enable_metrics:
            start_cpu_monitor_thread("detokenizer")

    def init_request_dispatcher(self):
        # 中译：初始化「基于类型的分发器」——按收到的消息类型路由到对应的 handler：
        #       嵌入输出、token id 批次输出、冻结 GC 请求、配置日志请求。
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (BatchEmbeddingOutput, self.handle_batch_embedding_out),
                (BatchTokenIDOutput, self.handle_batch_token_id_out),
                (FreezeGCReq, self.handle_freeze_gc_req),
                (ConfigureLoggingReq, self.handle_configure_logging_req),
            ]
        )

    def event_loop(self):
        """The event loop that handles requests

        中译：主事件循环（单 tokenizer 模式）。不断从调度器收消息 → 分发处理 → 把结果回传。
        """
        while True:
            # 中译：阻塞接收期间暂时关闭软看门狗（等消息不算卡死）。
            with self.soft_watchdog.disable():
                recv_obj = self.recv_from_scheduler.recv_pyobj()
            # 中译：按消息类型路由到对应 handler。
            output = self._request_dispatcher(recv_obj)
            if output is not None:
                self.send_to_tokenizer.send_pyobj(output)
            # 中译：喂狗（表明本轮处理正常推进）。
            self.soft_watchdog.feed()

    def trim_matched_stop(
        self, output: Union[str, List[int]], finished_reason: Dict, no_stop_trim: bool
    ):
        if not finished_reason:
            return output

        matched = finished_reason.get("matched", None)
        if not matched:
            return output

        # TODO(lmzheng): handle the case where multiple stop strs are hit

        # Trim stop str.
        # 中译：情形一——停止符是字符串：找到位置，按 no_stop_trim 决定保留还是去掉。
        if isinstance(matched, str) and isinstance(output, str):
            pos = output.find(matched)
            if pos == -1:
                return output
            end = pos + len(matched)
            return output[:end] if no_stop_trim else output[:pos]

        # Trim stop token.
        # 中译：情形二——停止符是 token（int）且 output 是 token 列表。
        if isinstance(matched, int) and isinstance(output, list):
            if no_stop_trim:
                return output
            # 200012 <|call|> is the tool call token and one of eos tokens for gpt-oss model
            # 中译：200012 是 gpt-oss 模型的工具调用 token <|call|>，也是其 eos 之一，需保留不裁。
            if output[-1] == 200012 and self.is_tool_call_parser_gpt_oss:
                return output
            assert len(output) > 0
            # NOTE: We can always assume the last token is the matched stop token
            # 中译：可以总是假设最后一个 token 就是命中的停止 token，去掉它。
            return output[:-1]
        return output

    def handle_batch_embedding_out(self, recv_obj: BatchEmbeddingOutput):
        # If it is embedding model, no detokenization is needed.
        # 中译：embedding 模型的输出是向量、无需反向解码，直接原样返回。
        return recv_obj

    def _grouped_batch_decode(
        self,
        ids_list: List[List[int]],
        skip_list: List[bool],
        space_list: List[bool],
    ) -> List[str]:
        """Batch decode with grouping by (skip_special_tokens, spaces_between_special_tokens).

        中译：批量解码，并按 (skip_special_tokens, spaces_between_special_tokens) 两个标志分组。
              同一次 batch_decode 要求所有行的这两个标志一致，所以标志不同时需按组分别解码。
        """
        n = len(ids_list)
        if n == 0:
            return []

        # Empty token spans decode to "" but tokenizer.batch_decode (and the
        # slow per-row decode_without_hf_kwargs path) still pays per-row
        # overhead; under high-concurrency streaming this adds up. Filter
        # empties out, decode the rest, then scatter back.
        # 中译：空 token 段解码结果为 ""，但 batch_decode（及逐行慢路径）仍会为每行付出开销；
        #       高并发流式下这会累加。故先过滤掉空行，只解码非空的，再按原位置散回。
        keep_idx: Optional[List[int]] = None
        if not all(ids_list):
            keep_idx = [i for i, ids in enumerate(ids_list) if ids]
            if not keep_idx:
                return [""] * n
            ids_list = [ids_list[i] for i in keep_idx]
            skip_list = [skip_list[i] for i in keep_idx]
            space_list = [space_list[i] for i in keep_idx]

        # 中译：非 fast tokenizer（纯 Python 实现）没有高效的 batch_decode，只能逐行解码。
        if not getattr(self.tokenizer, "is_fast", False):
            decoded = [
                decode_without_hf_kwargs(self.tokenizer, ids, skip)
                for ids, skip in zip(ids_list, skip_list)
            ]
        else:
            # fast path: all rows share the same (skip, space) flags.
            # 中译：快路径——若所有行的 (skip, space) 标志都相同，直接一次 batch_decode。
            first_skip, first_space = skip_list[0], space_list[0]
            if all(
                s == first_skip and sp == first_space
                for s, sp in zip(skip_list, space_list)
            ):
                decoded = self.tokenizer.batch_decode(
                    ids_list,
                    skip_special_tokens=first_skip,
                    spaces_between_special_tokens=first_space,
                )
            else:
                # Group indices by (skip, space) tuple and decode each group.
                # 中译：标志不一致时——按 (skip, space) 分组，每组各自 batch_decode，再按原下标填回。
                groups: Dict[Tuple[bool, bool], List[int]] = defaultdict(list)
                for idx, (skip, space) in enumerate(zip(skip_list, space_list)):
                    groups[(skip, space)].append(idx)

                decoded = [""] * len(ids_list)
                for (skip, space), indices in groups.items():
                    group_decoded = self.tokenizer.batch_decode(
                        [ids_list[idx] for idx in indices],
                        skip_special_tokens=skip,
                        spaces_between_special_tokens=space,
                    )
                    for idx, text in zip(indices, group_decoded):
                        decoded[idx] = text

        if keep_idx is None:
            return decoded
        results = [""] * n
        for i, text in zip(keep_idx, decoded):
            results[i] = text
        return results

    def _decode_batch_token_id_output(self, recv_obj: BatchTokenIDOutput):
        # 中译：批量增量解码的核心方法。对批次内每个请求：维护/更新其 DecodeStatus，
        #       解码出「上下文文本 surr」与「读取文本 read」，两者之差即本次新增文本。
        bs = len(recv_obj.rids)

        # Initialize decode status
        # 中译：准备每个请求本次要解码的 token。首次出现的请求新建 DecodeStatus，否则追加新 token。
        read_ids, surr_ids = [], []
        for i in range(bs):
            rid = recv_obj.rids[i]
            if rid not in self.decode_status:
                s = DecodeStatus(
                    decoded_text=recv_obj.decoded_texts[i],
                    decode_ids=list(recv_obj.decode_ids[i]),
                    surr_offset=0,
                    read_offset=recv_obj.read_offsets[i],
                )
                self.decode_status[rid] = s
            else:
                s = self.decode_status[rid]
                s.decode_ids.extend(recv_obj.decode_ids[i])

            # 中译：read_ids = 从 surr_offset 到末尾（含本次新 token，并裁掉命中的停止符）；
            #       surr_ids = 仅 [surr_offset, read_offset) 的上下文部分。两者解码后相减得新文本。
            read_ids.append(
                self.trim_matched_stop(
                    s.decode_ids[s.surr_offset :],
                    recv_obj.finished_reasons[i],
                    recv_obj.no_stop_trim[i],
                )
            )
            surr_ids.append(s.decode_ids[s.surr_offset : s.read_offset])

        # Decode token ids to strings
        # 中译：把 token id 解码为字符串。默认走批量解码（更快）。
        if not self.disable_tokenizer_batch_decode:
            surr_texts = self._grouped_batch_decode(
                surr_ids,
                recv_obj.skip_special_tokens,
                recv_obj.spaces_between_special_tokens,
            )
            read_texts = self._grouped_batch_decode(
                read_ids,
                recv_obj.skip_special_tokens,
                recv_obj.spaces_between_special_tokens,
            )
        else:
            # Do not use batch decode to prevent some detokenization edge cases (e.g., gpt-oss).
            # 中译：禁用批量解码时逐行解码，以避免某些反向解码的边界问题（如 gpt-oss）。
            surr_texts = [
                self.tokenizer.decode(
                    surr, skip_special_tokens=skip, spaces_between_special_tokens=space
                )
                for surr, skip, space in zip(
                    surr_ids,
                    recv_obj.skip_special_tokens,
                    recv_obj.spaces_between_special_tokens,
                )
            ]
            read_texts = [
                self.tokenizer.decode(
                    read, skip_special_tokens=skip, spaces_between_special_tokens=space
                )
                for read, skip, space in zip(
                    read_ids,
                    recv_obj.skip_special_tokens,
                    recv_obj.spaces_between_special_tokens,
                )
            ]

        # Incremental decoding
        # 中译：增量解码主逻辑——逐请求计算本次应该发送的新增文本。
        output_strs = []
        for i in range(bs):
            rid = recv_obj.rids[i]
            try:
                s = self.decode_status[rid]
            except KeyError:
                # 中译：状态丢失（通常因状态数超限被淘汰），提示调大 SGLANG_DETOKENIZER_MAX_STATES。
                raise RuntimeError(
                    f"Decode status not found for request {rid}. "
                    "It may be due to the request being evicted from the decode status due to memory pressure. "
                    "Please increase the maximum number of requests by setting "
                    "the SGLANG_DETOKENIZER_MAX_STATES environment variable to a bigger value than the default value. "
                    f"The current value is {DETOKENIZER_MAX_STATES}. "
                    "For more details, see: https://github.com/sgl-project/sglang/issues/2812"
                )
            # 中译：关键一步——read 文本去掉「上下文 surr 文本」的前缀，得到本次真正新增的文本。
            #       （多带 surr 上下文是为了跨 token 拼接正确，发送时再减掉这段前缀）
            new_text = read_texts[i][len(surr_texts[i]) :]
            if recv_obj.finished_reasons[i] is None:
                # Streaming. Invariant: sent_offset >= decoded_text_len. The
                # gap (`pending`) is "printable but uncommitted" text emitted
                # in a prior "�" recovery step; we skip it from this step's
                # emission so we don't double-send.
                pending = s.sent_offset - s.decoded_text_len
                if new_text and not new_text.endswith("�"):
                    # Clean text: commit to decoded_text and advance offsets.
                    s.append_decoded_text(new_text)
                    s.surr_offset = s.read_offset
                    s.read_offset = len(s.decode_ids)
                    s.sent_offset = s.decoded_text_len
                    output_strs.append(new_text[pending:] if pending else new_text)
                else:
                    # Incomplete UTF-8: emit the printable prefix only; do not
                    # commit (token offsets stay so the next iteration retries
                    # with more tokens).
                    printable = find_printable_text(new_text)
                    s.sent_offset = s.decoded_text_len + len(printable)
                    output_strs.append(printable[pending:] if pending else printable)
                continue

            if rid in self.decode_status:
                del self.decode_status[rid]

            # Finished: materialize once, trim the matched stop, emit the tail.
            output_str = self.trim_matched_stop(
                s.get_decoded_text() + new_text,
                recv_obj.finished_reasons[i],
                recv_obj.no_stop_trim[i],
            )
            incremental_output = output_str[s.sent_offset :]
            s.sent_offset = len(output_str)
            output_strs.append(incremental_output)

        return output_strs

    @staticmethod
    def _b64_encode_per_request(
        data_list: Optional[List[Optional[torch.Tensor]]],
    ) -> Optional[List[Optional[str]]]:
        # 中译：把「每请求一个张量」的列表编码为 base64 字符串（如路由专家、索引器 topk 等），
        #       放在反向解码热路径之外处理。输入为 None 则返回 None；逐项的 None 保持 None。
        """Encode a per-request list of tensors as base64 strings, off the
        tokenizer hot path. Returns None when the input is None; per-item None
        stays None.
        """
        if data_list is None:
            return None
        return [
            (
                pybase64.b64encode(item.numpy().tobytes()).decode("utf-8")
                if item is not None
                else None
            )
            for item in data_list
        ]

    def handle_batch_token_id_out(self, recv_obj: BatchTokenIDOutput):
        # 中译：处理 token id 批次输出的 handler：先增量解码出文本，再组装成 BatchStrOutput 转发。
        # If handling idle batch, set output_strs to [].
        # 中译：若为空转（idle）批次（无请求），输出文本置为空列表。
        output_strs = (
            self._decode_batch_token_id_output(recv_obj)
            if len(recv_obj.rids) > 0
            else []
        )
        routed_experts = self._b64_encode_per_request(recv_obj.routed_experts)
        indexer_topk = self._b64_encode_per_request(recv_obj.indexer_topk)
        return BatchStrOutput(
            rids=recv_obj.rids,
            http_worker_ipcs=recv_obj.http_worker_ipcs,
            finished_reasons=recv_obj.finished_reasons,
            output_strs=output_strs,
            output_ids=recv_obj.output_ids,
            prompt_tokens=recv_obj.prompt_tokens,
            reasoning_tokens=recv_obj.reasoning_tokens,
            completion_tokens=recv_obj.completion_tokens,
            cached_tokens=recv_obj.cached_tokens,
            cached_tokens_details=recv_obj.cached_tokens_details,
            spec_verify_ct=recv_obj.spec_verify_ct,
            spec_num_correct_drafts=recv_obj.spec_num_correct_drafts,
            spec_correct_drafts_histogram=recv_obj.spec_correct_drafts_histogram,
            input_token_logprobs_val=recv_obj.input_token_logprobs_val,
            input_token_logprobs_idx=recv_obj.input_token_logprobs_idx,
            output_token_logprobs_val=recv_obj.output_token_logprobs_val,
            output_token_logprobs_idx=recv_obj.output_token_logprobs_idx,
            input_top_logprobs_val=recv_obj.input_top_logprobs_val,
            input_top_logprobs_idx=recv_obj.input_top_logprobs_idx,
            output_top_logprobs_val=recv_obj.output_top_logprobs_val,
            output_top_logprobs_idx=recv_obj.output_top_logprobs_idx,
            input_token_ids_logprobs_val=recv_obj.input_token_ids_logprobs_val,
            input_token_ids_logprobs_idx=recv_obj.input_token_ids_logprobs_idx,
            output_token_ids_logprobs_val=recv_obj.output_token_ids_logprobs_val,
            output_token_ids_logprobs_idx=recv_obj.output_token_ids_logprobs_idx,
            output_token_entropy_val=recv_obj.output_token_entropy_val,
            output_hidden_states=recv_obj.output_hidden_states,
            routed_experts=routed_experts,
            indexer_topk=indexer_topk,
            customized_info=recv_obj.customized_info,
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
            retraction_counts=recv_obj.retraction_counts,
            token_steps=recv_obj.token_steps,
            dp_ranks=recv_obj.dp_ranks,
            time_stats=recv_obj.time_stats,
        )

    def handle_freeze_gc_req(self, recv_req: FreezeGCReq):
        # 中译：冻结垃圾回收（把当前存活对象移到永久代、不再扫描），减少 GC 开销。
        freeze_gc("Detokenizer Manager")
        return None

    def handle_configure_logging_req(self, recv_req: ConfigureLoggingReq):
        # 中译：运行时调整日志级别。
        if recv_req.log_level is not None:
            logging.getLogger().setLevel(recv_req.log_level.upper())


def is_health_check_request(rid: Optional[str]) -> bool:
    # 中译：判断一个请求 id 是否为健康检查请求（按特定前缀识别）。
    return isinstance(rid, str) and rid.startswith(HEALTH_CHECK_RID_PREFIX)


class LimitedCapacityDict(OrderedDict):
    """限容量有序字典：键数量达到上限后，插入新键时自动淘汰「最旧插入」的键。

    设计动机：
        DetokenizerManager 用 decode_status 字典按请求 id（rid）缓存每个请求的增量
        解码进度（已解码到的偏移、上次输出的文本等）。请求会源源不断地到来，若用普通
        dict 保存，已结束的请求状态不会被清理，长期运行会导致内存无限增长。
        本类通过「有界 + 自动淘汰」来约束 decode_status 的内存占用上限。

    淘汰策略：
        基于 OrderedDict 的插入顺序，淘汰最早插入的项（last=False，即 FIFO）。
        由于活跃请求会持续更新、不断被读取，正常情况下被淘汰的都是早已完成、不再需要
        的请求状态，因此实践中接近 LRU 的效果，一般不会误删仍在使用的状态。

    容量大小由构造参数 capacity 指定（见 DETOKENIZER_MAX_STATES）。
    """

    def __init__(self, capacity: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capacity = capacity  # 字典可容纳的最大键数量。

    def __setitem__(self, key, value):
        # 注意：这里用 len(self) >= capacity 判断，是在「插入前」检查。
        # 若插入的是已存在的 key（仅更新值），也会先淘汰一个最旧项，属可接受的近似策略。
        if len(self) >= self.capacity:
            # Remove the oldest element (first item in the dict)
            # 中译：容量已满，移除最旧插入的元素（有序字典的首项，last=False）。
            self.popitem(last=False)
        # Set the new item
        # 中译：写入新键值对；新键会被追加到有序字典末尾，成为「最新」项。
        super().__setitem__(key, value)


def run_detokenizer_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    detokenizer_manager_class=DetokenizerManager,
):
    # 中译：detokenizer 进程的入口函数。负责设置进程名/日志，创建管理器并进入事件循环；
    #       出异常时记录错误并给父进程发 SIGQUIT。
    # 中译：注册「父进程死亡时本进程自杀」，避免父进程退出后留下孤儿进程。
    kill_itself_when_parent_died()
    # 中译：设置进程标题，便于在 ps/top 中识别该 detokenizer 进程。
    setproctitle.setproctitle("sglang::detokenizer")
    # 中译：按 server_args 配置日志（级别、格式等）。
    configure_logger(server_args)
    # 中译：获取父进程句柄，用于异常时向其发信号。
    parent_process = psutil.Process().parent()

    # 中译：先置空，确保异常处理里能安全判断 manager 是否已创建。
    manager = None
    try:
        # 中译：创建去 token 化管理器（默认 DetokenizerManager，可由参数注入子类）。
        manager = detokenizer_manager_class(server_args, port_args)
        # 中译：单 tokenizer 模式走普通事件循环；多 tokenizer 模式走多 HTTP worker 事件循环。
        if server_args.tokenizer_worker_num == 1:
            manager.event_loop()
        else:
            manager.multi_http_worker_event_loop()
    except Exception:
        # 中译：捕获所有异常，记录完整堆栈，便于排查。
        traceback = get_exception_traceback()
        logger.error(f"DetokenizerManager hit an exception: {traceback}")
        # 中译：若管理器已创建，清理可能残留的 socket 映射（多 tokenizer 模式下的连接）。
        if manager is not None:
            manager.maybe_clear_socket_mapping()
        # 中译：向父进程发送 SIGQUIT，通知整体退出（避免单进程崩溃后系统僵在半死状态）。
        parent_process.send_signal(signal.SIGQUIT)
