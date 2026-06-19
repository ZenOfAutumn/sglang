"""中译：调度器侧的「输出流式发送器」。

本模块负责把调度器（Scheduler）每一步推理产出的结果，按需打包成批量输出对象
（生成模型用 BatchTokenIDOutput，embedding/reward 模型用 BatchEmbeddingOutput）并通过
ZMQ 发往 DetokenizerManager。核心难点：
- 流式（stream）语义：未结束的请求只在满足 stream_interval 等条件时才发送增量；
  各种 offset（send_token_offset / send_decode_id_offset 等）保证「只发新增、不重发」。
- 大量可选字段：logprobs、hidden_states、routed_experts、indexer_topk、投机解码统计、
  缓存命中明细等，按开关决定是否收集，避免无谓开销。

实现上用 _GenerationStreamAccumulator 累加器逐请求 accept()，最后一次性 to_payload()
组装为可序列化对象，再交给 send_to_detokenizer 发送。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    List,
    Optional,
)

import torch
import zmq

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    BatchEmbeddingOutput,
    BatchTokenIDOutput,
    GetLoadsReqInput,
)
from sglang.srt.managers.schedule_batch import (
    BaseFinishReason,
    Req,
)
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


# 中译：即使是非流式（非 stream）请求，也每隔这么多个 token 强制发送一次输出，
#       既能更新进度/指标，也避免长请求长时间静默。可由环境变量配置。
DEFAULT_FORCE_STREAM_INTERVAL = envs.SGLANG_FORCE_STREAM_INTERVAL.get()


@dataclass(kw_only=True, slots=True)
class SchedulerOutputStreamer:
    """中译：输出流式发送器。聚合一批请求的输出并发送给 detokenizer。

    各依赖字段（均由调度器注入）：
    - send_to_detokenizer：发往 detokenizer 的发送封装（带 send_output 方法）。
    - tree_cache：前缀缓存树，用于查询存储后端类型等。
    - ps：并行状态（ParallelState），含 dp_rank、attn_tp_rank 等。
    - is_generation：是否生成模型（否则走 embedding/reward 分支）。
    - spec_algorithm：投机解码算法（none 时不收集投机统计）。
    - disaggregation_mode：PD 分离模式（影响 input logprobs 是否发送等）。
    - enable_hicache_storage / load_inquirer_get_loads：分别为「是否启用分层缓存存储」
      与「查询负载」的回调。
    - _test_stream_output_count：仅测试用的计数器（配合故障注入）。
    """

    send_to_detokenizer: zmq.Socket
    tree_cache: BasePrefixCache
    ps: ParallelState
    server_args: ServerArgs
    is_generation: bool
    spec_algorithm: SpeculativeAlgorithm
    disaggregation_mode: DisaggregationMode
    enable_hicache_storage: Callable[[], bool]
    load_inquirer_get_loads: Callable[..., Any]
    _test_stream_output_count: int = 0

    def _get_storage_backend_type(self) -> str:
        """Get storage backend type from tree_cache.

        中译：从前缀缓存的 cache_controller 上取出存储后端的类型名（如分层缓存的
              磁盘/远端后端类名）；取不到时返回 "none"。
        """
        storage_backend_type = "none"
        cache_controller = getattr(self.tree_cache, "cache_controller", None)
        if cache_controller and hasattr(cache_controller, "storage_backend"):
            storage_backend = cache_controller.storage_backend
            if storage_backend is not None:
                storage_backend_type = type(storage_backend).__name__
        return storage_backend_type

    def get_cached_tokens_details(self, req: Req) -> Optional[dict]:
        """Get detailed cache breakdown for a request, if available.

        Returns:
            - None if no cached tokens at all
            - {"device": X, "host": Y} without storage breakdown
            - {"device": X, "host": Y, "storage": Z} with storage breakdown

        中译：返回某请求缓存命中的明细（按来源拆分）。
            - 无任何缓存命中：返回 None；
            - 有分层命中：返回 {"device": 设备(GPU), "host": 主机(CPU)}，
              视情况再带上 "storage"（L3/外部存储）与 "storage_backend"（后端类型名）。
        """
        if (
            req.cached_tokens_device > 0
            or req.cached_tokens_host > 0
            or req.cached_tokens_storage > 0
        ):
            details = {
                "device": req.cached_tokens_device,
                "host": req.cached_tokens_host,
            }
            # In PD mode the L3 hit is produced on prefill and reported on
            # decode via metadata, while decode may not have a local storage backend.
            # 中译：PD 分离模式下，L3 命中在 prefill 端产生、通过元数据在 decode 端上报，
            #       而 decode 端本身可能没有本地存储后端，故 storage>0 或启用了分层存储时都补上该字段。
            if req.cached_tokens_storage > 0 or self.enable_hicache_storage():
                details["storage"] = req.cached_tokens_storage
            if self.enable_hicache_storage():
                details["storage_backend"] = self._get_storage_backend_type()
            return details

        # 中译：未细分但有总的缓存命中数时，退化为只报设备命中。
        if req.cached_tokens > 0:
            return {
                "device": req.cached_tokens,
                "host": 0,
            }

        return None

    def stream_output(
        self,
        reqs: List[Req],
        return_logprob: bool,
        skip_req: Optional[Req] = None,
    ):
        """Stream the output to detokenizer.

        中译：把一批请求的输出流式发往 detokenizer 的统一入口。按模型类型分流到
              生成模型 / embedding（或 reward）两条路径。skip_req 用于跳过某个请求
              （例如分块预填充中尚不应输出的请求）。
        """
        if self.is_generation:
            self._stream_output_generation(reqs, return_logprob, skip_req)
        else:  # embedding or reward model
            self._stream_output_embedding(reqs)

        # 中译：仅测试用——若设置了「发送 N 次后崩溃」的环境变量，则触发故障注入，
        #       用于验证异常恢复路径。生产环境该值为 0，不会触发。
        if envs.SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS.get() > 0:
            self._trigger_crash_for_tests(
                envs.SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS.get()
            )

    def _trigger_crash_for_tests(self, crash_threshold: int):
        # Crash trigger: crash after stream_output is called N times
        # This is used for testing purposes.
        # 中译：测试用的崩溃触发器——stream_output 被调用累计达到阈值次数后抛异常。
        self._test_stream_output_count += 1
        if self._test_stream_output_count >= crash_threshold:
            raise RuntimeError(
                f"Test crash after stream_output called {self._test_stream_output_count} times"
            )

    def _stream_output_generation(
        self,
        reqs: List[Req],
        return_logprob: bool,
        skip_req: Optional[Req] = None,
        is_idle_batch: bool = False,
    ):
        # 中译：生成模型的流式发送。先扫一遍本批，确定是否需要收集 hidden_states /
        #       routed_experts / indexer_topk（只要批内有任一请求需要就收集），
        #       以避免对不需要的字段做无谓的搬运。
        return_hidden_states = any(
            req.return_hidden_states for req in reqs if req is not skip_req
        )
        return_routed_experts = any(
            req.return_routed_experts for req in reqs if req is not skip_req
        )
        return_indexer_topk = any(
            req.return_indexer_topk for req in reqs if req is not skip_req
        )

        acc = _GenerationStreamAccumulator(
            return_logprob=return_logprob,
            return_hidden_states=return_hidden_states,
            return_routed_experts=return_routed_experts,
            return_indexer_topk=return_indexer_topk,
            spec_algorithm=self.spec_algorithm,
            disaggregation_mode=self.disaggregation_mode,
            default_stream_interval=self.server_args.stream_interval,
            default_force_stream_interval=DEFAULT_FORCE_STREAM_INTERVAL,
            get_cached_tokens_details=self.get_cached_tokens_details,
        )
        # 中译：查询当前核心负载（用于负载均衡/上报），随本批一起发送。
        load = self.load_inquirer_get_loads(GetLoadsReqInput(include=["core"]))

        for req in reqs:
            if req is skip_req:
                continue
            if req.finished() and req.finished_output:
                # With the overlap schedule, a request will try to output twice and hit this line twice
                # because of the one additional delayed token. This "continue" prevented the dummy output.
                # 中译：在 overlap（重叠）调度下，因多出一个延迟 token，已完成请求会尝试输出两次、
                #       两次走到这里；用 continue 跳过第二次的「空输出」，避免重复发送。
                continue

            # 中译：把该请求纳入累加器（内部决定是否真的产出本次增量），并按需记录时间统计。
            acc.accept(req=req)
            self._maybe_log_time_stats(req=req)

        # Send to detokenizer
        # 中译：把累加结果组装为可发送的 payload；为 None（无内容且非空转批次）时不发送。
        payload = acc.to_payload(
            load=load,
            dp_rank=self.ps.dp_rank,
            is_idle_batch=is_idle_batch,
            has_reqs=bool(reqs),
        )
        if payload is not None:
            self.send_to_detokenizer.send_output(payload)

    def _maybe_log_time_stats(self, *, req: Req) -> None:
        # 中译：仅在请求已结束、且当前是注意力张量并行的 0 号 rank、且开启了时间统计日志时，
        #       记录该请求的耗时统计（避免多 rank 重复打印）。
        if (
            req.finished()
            and self.ps.attn_tp_rank == 0
            and self.server_args.enable_request_time_stats_logging
        ):
            req.log_time_stats()

    def _stream_output_embedding(self, reqs: List[Req]):
        # 中译：embedding / reward 模型的输出发送。只处理已结束的请求，收集其向量与统计，
        #       组装成 BatchEmbeddingOutput 发往下游。
        rids = []
        http_worker_ipcs = []
        finished_reasons: List[BaseFinishReason] = []

        embeddings = []
        prompt_tokens = []
        cached_tokens = []
        cached_tokens_details = []  # Detailed breakdown by cache source
        time_stats = []
        retraction_counts = []
        phs_list = []
        has_phs = False
        for req in reqs:
            if req.finished():
                rids.append(req.rid)
                http_worker_ipcs.append(req.http_worker_ipc)
                finished_reasons.append(req.finished_reason.to_json())
                embeddings.append(req.embedding)
                prompt_tokens.append(len(req.origin_input_ids))
                cached_tokens.append(req.cached_tokens)

                # Collect detailed cache breakdown if available
                cached_tokens_details.append(self.get_cached_tokens_details(req))
                time_stats.append(req.time_stats)
                retraction_counts.append(req.retraction_count)

                phs = req.pooled_hidden_state
                phs_list.append(phs)
                if phs is not None:
                    has_phs = True

        # Optimize PHS for pickle: torch.stack reduces N __reduce_ex__
        # calls to 1 across the ZMQ IPC boundary.  We can only stack when
        # *every* entry is non-None (homogeneous batch); mixed batches
        # (some requests want PHS, others don't) keep the raw list so
        # positional indexing on the receiver side stays correct.
        # 中译：优化 pooled_hidden_state（PHS）的序列化：把 N 个张量 stack 成一个，
        #       可将跨 ZMQ 边界的 N 次 pickle 调用降为 1 次。仅当批内每项都非 None
        #       且形状一致（同质批）才能 stack；混合批保留原列表，以保证接收端按位置索引仍正确。
        stacked_phs = None
        if has_phs:
            all_have_phs = all(t is not None for t in phs_list)
            if all_have_phs:
                if all(t.shape == phs_list[0].shape for t in phs_list):
                    stacked_phs = torch.stack(phs_list)
                else:
                    stacked_phs = phs_list
            else:
                stacked_phs = phs_list

        self.send_to_detokenizer.send_output(
            BatchEmbeddingOutput(
                rids=rids,
                http_worker_ipcs=http_worker_ipcs,
                time_stats=time_stats,
                finished_reasons=finished_reasons,
                embeddings=embeddings,
                prompt_tokens=prompt_tokens,
                cached_tokens=cached_tokens,
                cached_tokens_details=cached_tokens_details,
                placeholder_tokens_idx=None,
                placeholder_tokens_val=None,
                retraction_counts=retraction_counts,
                pooled_hidden_states=stacked_phs,
            )
        )


@dataclass(slots=True, kw_only=True)
class _GenerationStreamAccumulator:
    """中译：生成模型输出累加器。

    逐请求调用 accept() 把字段追加到各个并行的 list（rids、output_ids、logprobs 等
    一一对应、下标一致），最后 to_payload() 一次性打包为 BatchTokenIDOutput。
    顶部的布尔开关（return_logprob / return_hidden_states 等）决定哪些可选字段在
    __post_init__ 中初始化为 list 并参与收集，未开启的保持 None 以省开销。
    """

    return_logprob: bool
    return_hidden_states: bool
    return_routed_experts: bool
    return_indexer_topk: bool
    spec_algorithm: Any
    disaggregation_mode: DisaggregationMode
    default_stream_interval: int
    default_force_stream_interval: int
    get_cached_tokens_details: Callable[[Req], Optional[dict]]

    rids: list = field(default_factory=list)
    http_worker_ipcs: list = field(default_factory=list)
    finished_reasons: list = field(default_factory=list)
    decoded_texts: list = field(default_factory=list)
    decode_ids_list: list = field(default_factory=list)
    read_offsets: list = field(default_factory=list)
    output_ids: list = field(default_factory=list)
    skip_special_tokens: list = field(default_factory=list)
    spaces_between_special_tokens: list = field(default_factory=list)
    no_stop_trim: list = field(default_factory=list)
    prompt_tokens: list = field(default_factory=list)
    reasoning_tokens: list = field(default_factory=list)
    completion_tokens: list = field(default_factory=list)
    cached_tokens: list = field(default_factory=list)
    cached_tokens_details: list = field(
        default_factory=list
    )  # Detailed breakdown by cache source
    spec_verify_ct: list = field(default_factory=list)
    spec_num_correct_drafts: list = field(default_factory=list)
    spec_correct_drafts_histogram: list = field(default_factory=list)
    retraction_counts: list = field(default_factory=list)
    output_hidden_states: Optional[list] = None
    routed_experts: Optional[list] = None
    indexer_topk: Optional[list] = None
    customized_info: dict = field(default_factory=dict)
    time_stats: list = field(default_factory=list)
    input_token_logprobs_val: Optional[list] = None
    input_token_logprobs_idx: Optional[list] = None
    output_token_logprobs_val: Optional[list] = None
    output_token_logprobs_idx: Optional[list] = None
    input_top_logprobs_val: Optional[list] = None
    input_top_logprobs_idx: Optional[list] = None
    output_top_logprobs_val: Optional[list] = None
    output_top_logprobs_idx: Optional[list] = None
    input_token_ids_logprobs_val: Optional[list] = None
    input_token_ids_logprobs_idx: Optional[list] = None
    output_token_ids_logprobs_val: Optional[list] = None
    output_token_ids_logprobs_idx: Optional[list] = None

    def __post_init__(self) -> None:
        # 中译：根据开关延迟初始化各可选字段为空 list（否则保持 None，表示本批不收集）。
        if self.return_hidden_states:
            self.output_hidden_states = []
        if self.return_routed_experts:
            self.routed_experts = []
        if self.return_indexer_topk:
            self.indexer_topk = []

        if self.return_logprob:
            self.input_token_logprobs_val = []
            self.input_token_logprobs_idx = []
            self.output_token_logprobs_val = []
            self.output_token_logprobs_idx = []
            self.input_top_logprobs_val = []
            self.input_top_logprobs_idx = []
            self.output_top_logprobs_val = []
            self.output_top_logprobs_idx = []
            self.input_token_ids_logprobs_val = []
            self.input_token_ids_logprobs_idx = []
            self.output_token_ids_logprobs_val = []
            self.output_token_ids_logprobs_idx = []

    def accept(self, *, req: Req) -> None:
        # 中译：判定本请求本步是否需要输出，并在需要时把其各字段追加进累加器。
        if req.finished():
            # 中译：请求已结束——必定输出最终结果。用 finished_output 标记避免重复发送，
            #       并记录结束时的输出长度 finished_len。
            assert not req.finished_output
            req.finished_output = True
            if req.finished_len is None:
                req.finished_len = len(req.output_ids)
            should_output = True
        else:
            if req.stream:
                # 中译：流式请求——按 stream_interval 决定本步是否吐出增量。
                stream_interval = (
                    req.sampling_params.stream_interval or self.default_stream_interval
                )

                # origin stream_interval logic
                # 中译：原始的间隔判断逻辑。interval>1 时用「余 1」错相位，避免与首 token 重合；
                #       interval==1（每步都发）时用「余 0」。
                should_output = (
                    len(req.output_ids) % stream_interval == 1
                    if stream_interval > 1
                    else len(req.output_ids) % stream_interval == 0
                )

                if should_output:
                    # check_match_stop_str_prefix if  tail_str's suffix match stop_str prefix
                    # 中译：若当前尾部文本的后缀正好是某个 stop 字符串的前缀，则暂不输出——
                    #       等更多 token 到来以确认是否命中停止符，避免把可能被裁掉的文本提前发出。
                    should_output &= not req.check_match_stop_str_prefix()
            else:
                # 中译：非流式请求——平时不发，仅每隔 force_stream_interval 个 token 强制发一次。
                should_output = (
                    len(req.output_ids) % self.default_force_stream_interval == 0
                )

        if not should_output:
            return

        # 中译：本请求各种「已发送偏移」的起点，配合下方切片实现「只发新增」。
        send_token_offset = req.send_token_offset
        send_output_token_logprobs_offset = req.send_output_token_logprobs_offset
        self.rids.append(req.rid)
        self.http_worker_ipcs.append(req.http_worker_ipc)
        self.finished_reasons.append(
            req.finished_reason.to_json() if req.finished_reason else None
        )
        self.decoded_texts.append(req.decoded_text)
        # 中译：初始化增量反向解码所需的数据：累计 token 列表与读取起点（read_offset）。
        decode_ids, read_offset = req.init_incremental_detokenize()

        # 中译：只发送上次之后新增的 token（从 send_decode_id_offset 起切片），减少 IPC 体积。
        self.decode_ids_list.append(decode_ids[req.send_decode_id_offset :])

        # Exclude the tokens after stop condition
        # 中译：output_ids_through_stop 已排除命中停止条件之后的 token（只保留有效输出）。
        output_ids_ = req.output_ids_through_stop

        # 中译：推进 send_decode_id_offset，记录本次已发送到的位置，供下次切片续接。
        req.send_decode_id_offset = len(decode_ids)
        self.read_offsets.append(read_offset)
        # 中译：output_ids 同样只取新增部分（从 send_token_offset 起），并推进偏移。
        self.output_ids.append(output_ids_[send_token_offset:])
        req.send_token_offset = len(output_ids_)
        self.skip_special_tokens.append(req.sampling_params.skip_special_tokens)
        self.spaces_between_special_tokens.append(
            req.sampling_params.spaces_between_special_tokens
        )
        self.no_stop_trim.append(req.sampling_params.no_stop_trim)
        self.prompt_tokens.append(len(req.origin_input_ids))
        self.reasoning_tokens.append(req.reasoning_tokens)
        self.completion_tokens.append(len(output_ids_))
        self.cached_tokens.append(req.cached_tokens)

        # Collect detailed cache breakdown if available
        self.cached_tokens_details.append(self.get_cached_tokens_details(req))

        self.retraction_counts.append(req.retraction_count)

        self.time_stats.append(req.time_stats)

        # 中译：启用投机解码时，收集其统计（验证次数、被接受的草稿数、正确草稿直方图）。
        if not self.spec_algorithm.is_none():
            self.spec_verify_ct.append(req.spec_verify_ct)
            self.spec_num_correct_drafts.append(req.spec_num_correct_drafts)
            self.spec_correct_drafts_histogram.append(req.spec_correct_drafts_histogram)

        if self.return_logprob:
            # 中译：input logprobs 只需发送一次（input_logprob_sent 标记）。
            #       且 PD 分离的 decode 端不发 input logprobs，并要求已在 prefill 阶段算出。
            if (
                req.return_logprob
                and not req.input_logprob_sent
                # Decode server does not send input logprobs
                and self.disaggregation_mode != DisaggregationMode.DECODE
                # Only send when input logprobs have been computed (after prefill)
                and req.logprob.input_token_logprobs_val is not None
            ):
                self.input_token_logprobs_val.append(
                    req.logprob.input_token_logprobs_val
                )
                self.input_token_logprobs_idx.append(
                    req.logprob.input_token_logprobs_idx
                )
                self.input_top_logprobs_val.append(req.logprob.input_top_logprobs_val)
                self.input_top_logprobs_idx.append(req.logprob.input_top_logprobs_idx)
                self.input_token_ids_logprobs_val.append(
                    req.logprob.input_token_ids_logprobs_val
                )
                self.input_token_ids_logprobs_idx.append(
                    req.logprob.input_token_ids_logprobs_idx
                )
                req.input_logprob_sent = True
            else:
                # 中译：不满足发送条件——填充空列表占位，保持各 list 下标与请求一一对应。
                self.input_token_logprobs_val.append([])
                self.input_token_logprobs_idx.append([])
                self.input_top_logprobs_val.append([])
                self.input_top_logprobs_idx.append([])
                self.input_token_ids_logprobs_val.append([])
                self.input_token_ids_logprobs_idx.append([])

            if req.return_logprob:
                # 中译：output logprobs 取 [已发送偏移, logprob_end) 的增量；至少取 1 个，
                #       从而保证 prefill 阶段最后一个位置的 logprob 也会被发出。
                logprob_end = max(len(output_ids_), 1)
                self.output_token_logprobs_val.append(
                    req.logprob.output_token_logprobs_val[
                        send_output_token_logprobs_offset:logprob_end
                    ]
                )
                self.output_token_logprobs_idx.append(
                    req.logprob.output_token_logprobs_idx[
                        send_output_token_logprobs_offset:logprob_end
                    ]
                )
                self.output_top_logprobs_val.append(
                    req.logprob.output_top_logprobs_val[
                        send_output_token_logprobs_offset:logprob_end
                    ]
                )
                self.output_top_logprobs_idx.append(
                    req.logprob.output_top_logprobs_idx[
                        send_output_token_logprobs_offset:logprob_end
                    ]
                )
                self.output_token_ids_logprobs_val.append(
                    req.logprob.output_token_ids_logprobs_val[
                        send_output_token_logprobs_offset:logprob_end
                    ]
                )
                self.output_token_ids_logprobs_idx.append(
                    req.logprob.output_token_ids_logprobs_idx[
                        send_output_token_logprobs_offset:logprob_end
                    ]
                )
                # 中译：推进 output logprobs 的已发送偏移，下次从这里续发。
                req.send_output_token_logprobs_offset = logprob_end
            else:
                self.output_token_logprobs_val.append([])
                self.output_token_logprobs_idx.append([])
                self.output_top_logprobs_val.append([])
                self.output_top_logprobs_idx.append([])
                self.output_token_ids_logprobs_val.append([])
                self.output_token_ids_logprobs_idx.append([])

        if self.return_hidden_states:
            self.output_hidden_states.append(
                req.hidden_states if req.return_hidden_states else None
            )
        if self.return_routed_experts:
            self.routed_experts.append(
                req.routed_experts if req.return_routed_experts else None
            )
        if self.return_indexer_topk:
            self.indexer_topk.append(
                req.indexer_topk if req.return_indexer_topk else None
            )

        # 中译：自定义信息按 key 分别收集，同样只取本次新增的 [send_token_offset, len) 区间。
        if req.customized_info is not None:
            for k, v in req.customized_info.items():
                if k not in self.customized_info:
                    self.customized_info[k] = []
                self.customized_info[k].append(v[send_token_offset : len(output_ids_)])

    def to_payload(
        self, *, load, dp_rank: int, is_idle_batch: bool, has_reqs: bool
    ) -> Optional[BatchTokenIDOutput]:
        # 中译：把累加结果打包为可发送对象。既无请求又非空转批次时返回 None（无需发送）。
        #       空转（idle）批次虽无请求，仍需发送以驱动下游/保持节奏。
        if not (has_reqs or is_idle_batch):
            return None
        # 中译：为每个请求标注其所属的 data-parallel rank（无请求时置 None）。
        dp_ranks = [dp_rank] * len(self.rids) if self.rids else None
        return BatchTokenIDOutput(
            rids=self.rids,
            http_worker_ipcs=self.http_worker_ipcs,
            spec_verify_ct=self.spec_verify_ct,
            spec_num_correct_drafts=self.spec_num_correct_drafts,
            spec_correct_drafts_histogram=self.spec_correct_drafts_histogram,
            time_stats=self.time_stats,
            finished_reasons=self.finished_reasons,
            decoded_texts=self.decoded_texts,
            decode_ids=self.decode_ids_list,
            read_offsets=self.read_offsets,
            output_ids=self.output_ids,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
            no_stop_trim=self.no_stop_trim,
            prompt_tokens=self.prompt_tokens,
            reasoning_tokens=self.reasoning_tokens,
            completion_tokens=self.completion_tokens,
            cached_tokens=self.cached_tokens,
            cached_tokens_details=self.cached_tokens_details,
            input_token_logprobs_val=self.input_token_logprobs_val,
            input_token_logprobs_idx=self.input_token_logprobs_idx,
            output_token_logprobs_val=self.output_token_logprobs_val,
            output_token_logprobs_idx=self.output_token_logprobs_idx,
            input_top_logprobs_val=self.input_top_logprobs_val,
            input_top_logprobs_idx=self.input_top_logprobs_idx,
            output_top_logprobs_val=self.output_top_logprobs_val,
            output_top_logprobs_idx=self.output_top_logprobs_idx,
            input_token_ids_logprobs_val=self.input_token_ids_logprobs_val,
            input_token_ids_logprobs_idx=self.input_token_ids_logprobs_idx,
            output_token_ids_logprobs_val=self.output_token_ids_logprobs_val,
            output_token_ids_logprobs_idx=self.output_token_ids_logprobs_idx,
            output_token_entropy_val=None,
            output_hidden_states=self.output_hidden_states,
            routed_experts=self.routed_experts,
            indexer_topk=self.indexer_topk,
            customized_info=self.customized_info,
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
            retraction_counts=self.retraction_counts,
            load=load,
            dp_ranks=dp_ranks,
        )
