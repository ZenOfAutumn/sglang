# 中译：本模块定义 SchedulerBatchResultProcessor——调度器（Scheduler）的「批次结果处理器」。
#       职责：把模型 worker 前向（forward）产出的原始结果（logits、采样出的 next token、
#       logprob、hidden states、投机解码验证结果等）整理为对每个请求（Req）的可输出状态，
#       并完成完成态判定、KV cache 释放/缓存、流式回包组织。
#       三条主路径对应三种 forward 模式：
#         - prefill（extend，含分块 chunked prefill）：process_batch_result_prefill
#         - decode（逐 token 自回归，含投机解码）：process_batch_result_decode
#         - idle（空转批次，无实际请求）：process_batch_result_idle
#       另有 disaggregation（PD 分离）DECODE 端的预构建路径 process_batch_result_prebuilt。
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Callable,
    List,
    Optional,
    Tuple,
    Union,
)

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_MATCHED_TOKEN,
    Req,
    ScheduleBatch,
    mamba_lazy_spec_in_window,
)
from sglang.srt.mem_cache.common import (
    maybe_cache_unfinished_req,
    release_kv_cache,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    get_required_capture_hidden_mode,
    get_server_return_hidden_states_mode,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_memory,
    get_observability,
    mamba_extra_buffer_lazy_enabled,
    mamba_track_grid,
    max_speculative_num_draft_tokens,
)
from sglang.srt.sampling.sampling_observer import CommittedTokens
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.state_capturer.indexer_topk import get_global_indexer_capturer
from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer

if TYPE_CHECKING:
    from sglang.srt.beam_search.coordinator import BeamCoordinator
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
        DecodeKVCacheOffloadManager,
    )
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.managers.scheduler_components.logprob_result_processor import (
        SchedulerLogprobResultProcessor,
    )
    from sglang.srt.managers.scheduler_components.metrics_reporter import (
        SchedulerMetricsReporter,
    )
    from sglang.srt.managers.scheduler_components.output_streamer import (
        SchedulerOutputStreamer,
    )
    from sglang.srt.managers.tp_worker import BaseTpWorker
    from sglang.srt.managers.utils import (
        EmbeddingBatchResult,
        GenerationBatchResult,
    )
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
    from sglang.srt.sampling.sampling_observer import HostAuxiliaryOutput

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerBatchResultProcessor:
    is_generation: bool
    disaggregation_mode: DisaggregationMode
    enable_overlap: bool
    enable_overlap_mlx: bool
    model_config: ModelConfig
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator  # token→KV 槽位分配器
    tree_cache: BasePrefixCache  # 前缀（radix）缓存，用于复用 KV
    hisparse_coordinator: Optional[HiSparseCoordinator]
    req_to_token_pool: ReqToTokenPool  # 请求→token 映射池
    decode_offload_manager: Optional[DecodeKVCacheOffloadManager]  # decode 端 KV 卸载管理器
    metrics_collector: SchedulerMetricsCollector
    metrics_reporter: SchedulerMetricsReporter
    draft_worker: BaseTpWorker
    model_worker: BaseTpWorker
    logprob_result_processor: SchedulerLogprobResultProcessor
    output_streamer: SchedulerOutputStreamer
    beam_coordinator: BeamCoordinator
    abort_request: Callable

    def process_batch_result_prebuilt(self, batch: ScheduleBatch):
        # 中译：处理 PD 分离架构下 DECODE 端的「预构建（prebuilt）」批次结果。
        #       此时 prefill 已在 prefill 引擎完成、KV 也已传过来，DECODE 端只需更新
        #       完成态、必要时释放 KV，并把结果流式输出（logprob 由 prefill 引擎负责）。
        assert self.disaggregation_mode == DisaggregationMode.DECODE
        # 中译：开启 radix cache 时，用 free_group 包裹批量释放，减少分配器加锁/碎片开销。
        use_free_group = get_disagg().disaggregation_decode_enable_radix_cache
        if use_free_group:
            self.token_to_kv_pool_allocator.free_group_begin()
        for req in batch.reqs:
            req.time_stats.set_decode_prebuilt_finish_time()
            req.update_finish_state()  # 根据已生成 token 重新判断是否触发停止条件
            if req.finished():
                req.time_stats.set_quick_finish_time()
                if get_memory().enable_hisparse:
                    self.hisparse_coordinator.request_finished(req)
                release_kv_cache(req, self.tree_cache)  # 已完成：释放该请求占用的 KV

        # Note: Logprobs should be handled on the prefill engine.
        # 中译：注意——logprob 应由 prefill 引擎处理，此处不再计算。
        self.output_streamer.stream_output(batch.reqs, batch.return_logprob)
        if use_free_group:
            self.token_to_kv_pool_allocator.free_group_end()  # 提交本组释放

    def _maybe_collect_routed_experts(self, req: Req):
        """Collect routed experts for a finished request.

        Returns immediately if `return_routed_experts` was not set on the
        request, so non-opted-in reqs don't pay the host-gather cost.

        Honors the caller's absolute start so the response covers
        `[start_len, seqlen - 1)`. The default start_len is 0, which returns
        the full sequence.

        Logs a soft warning if the resulting tensor's row count differs from
        the expected `seqlen - 1 - start_len`, to catch silent regressions.

        中译：为已完成的请求收集「路由专家（routed experts，MoE 选中的专家）」信息。
              未在请求上开启 return_routed_experts 时立即返回，避免未订阅的请求白白付出
              主机端 gather 开销。覆盖区间 [start_len, seqlen - 1)，默认 start_len=0 即全序列。
              若结果张量行数与预期 (seqlen - 1 - start_len) 不符，会打软告警以捕捉静默回归。
        """
        if not req.return_routed_experts:
            return
        capturer = get_global_experts_capturer()
        if capturer is None:
            return
        start_len = req.routed_experts_start_len
        seqlen = len(req.origin_input_ids) + len(req.output_ids_through_stop)
        req.routed_experts = capturer.get_topk(
            req_pool_idx=req.kv.req_pool_idx,
            seqlen=seqlen,
            req_to_token_pool=self.req_to_token_pool,
            start_len=start_len,
        )

        expected_rows = max(0, seqlen - 1 - start_len)
        if (
            req.routed_experts is not None
            and req.routed_experts.shape[0] != expected_rows
        ):
            logger.warning(
                "routed_experts row-count mismatch for req %s: got %d, expected %d "
                "(seqlen=%d, raw_seqlen=%d, cached_tokens=%d, start_len=%s). "
                "This indicates a silent bug.",
                req.rid,
                req.routed_experts.shape[0],
                expected_rows,
                seqlen,
                req.seqlen,
                req.cached_tokens,
                req.routed_experts_start_len,
            )

    def _maybe_collect_indexer_topk(self, req: Req):
        # 中译：为完成的请求收集 indexer 的 topk 索引（稀疏注意力等场景的调试/状态信息）。
        #       仅当全局 indexer capturer 存在时才采集，否则直接返回。
        capturer = get_global_indexer_capturer()
        if capturer is None:
            return
        seqlen = len(req.origin_input_ids) + len(req.output_ids_through_stop)
        req.indexer_topk = capturer.get_topk(
            req_pool_idx=req.kv.req_pool_idx,
            seqlen=seqlen,
            req_to_token_pool=self.req_to_token_pool,
        )

    def _maybe_collect_customized_info(
        self,
        i: int,
        req: Req,
        logits_output: LogitsProcessorOutput,
    ):
        # 中译：从批次级的 customized_info 中切出第 i 个请求对应的元素，累积到该请求上。
        if logits_output is not None and logits_output.customized_info is not None:
            if req.customized_info is None:
                req.customized_info = {}
            for k, v in logits_output.customized_info.items():
                if k not in req.customized_info:
                    req.customized_info[k] = []
                # Copy the element so it doesn't retain the entire batch
                # tensor/array via a view reference.
                # 中译：必须拷贝该元素——否则切片是「视图（view）」，会让整个批次张量/数组
                #       无法被回收，造成内存长期占用。
                elem = v[i]
                if isinstance(elem, torch.Tensor):
                    elem = elem.clone()
                elif hasattr(elem, "copy") and callable(elem.copy):
                    elem = elem.copy()
                req.customized_info[k].append(elem)

    @staticmethod
    def _visible_output_len(req: Req) -> int:
        return req.finished_len if req.finished_len is not None else len(req.output_ids)

    @classmethod
    def snapshot_auxiliary_output_starts(
        cls,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> Optional[List[int]]:
        if result.auxiliary_host_output is None:
            return None
        return [cls._visible_output_len(req) for req in batch.reqs]

    @classmethod
    def _build_auxiliary_commits(
        cls,
        batch: ScheduleBatch,
        output_starts: List[int],
    ) -> List[Optional[CommittedTokens]]:
        commits: List[Optional[CommittedTokens]] = []
        for req, output_start in zip(batch.reqs, output_starts, strict=True):
            output_end = cls._visible_output_len(req)
            if output_end < output_start:
                raise RuntimeError("committed output length moved backwards")
            if output_end == output_start:
                commits.append(None)
                continue
            commits.append(
                CommittedTokens(
                    output_index=output_start,
                    token_ids=tuple(req.output_ids[output_start:output_end]),
                )
            )
        return commits

    @classmethod
    def consume_auxiliary_output(
        cls,
        batch: ScheduleBatch,
        output: HostAuxiliaryOutput,
        output_starts: List[int],
    ) -> None:
        output.consume(batch, cls._build_auxiliary_commits(batch, output_starts))

    def process_batch_result_prefill(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        # 中译：处理 prefill（extend）批次的前向结果，是 prefill 阶段「前向算完之后」的收尾入口。
        skip_stream_req = None
        self.token_to_kv_pool_allocator.free_group_begin()

        if self.is_generation:
            # 中译：copy_done 是「结果 GPU→CPU 拷贝完成」事件，先同步确保数据已就绪。
            if result.copy_done is not None:
                result.copy_done.synchronize()
            # 中译：MoE 模型若开启了路由专家（routed experts）观测，前向产物是异步句柄，
            auxiliary_output_starts = self.snapshot_auxiliary_output_starts(
                batch, result
            )
            auxiliary_output = result.auxiliary_host_output
            if result.routed_experts_output is not None:
                result.routed_experts_output.finalize()
                result.routed_experts_output = None
            # 中译：稀疏注意力 indexer 的 top-k 结果同理：落地后清空引用。
            if result.indexer_topk_output is not None:
                result.indexer_topk_output.finalize()
                result.indexer_topk_output = None

            # 中译：从 result 解包本轮需要的四个字段：
            #   logits_output                     —— 模型前向输出（含 logits、各类 logprob、hidden states）；
            #   next_token_ids                    —— 每个请求采样得到的首个 next token（GPU 张量）；
            #   extend_input_len_per_req          —— 各请求本轮 extend（prefill）的输入长度；
            #   extend_logprob_start_len_per_req  —— 各请求 input logprob 的起始位置（从第几个 token 开始算）。
            (
                logits_output,
                next_token_ids,
                extend_input_len_per_req,
                extend_logprob_start_len_per_req,
            ) = (
                result.logits_output,
                result.next_token_ids,
                result.extend_input_len_per_req,
                result.extend_logprob_start_len_per_req,
            )

            # Move next_token_ids and logprobs to cpu
            # 中译：把采样出的 next token id 与 logprob 从 GPU 张量搬到 CPU（Python list），
            #       便于后续逐请求的纯 CPU 处理。
            next_token_ids = next_token_ids.tolist()
            self.move_logprobs_to_cpu(batch=batch, logits_output=logits_output)

            # 中译：流水线并行（PP）下，纯分块批次可跳过输出通信这一优化；此处校验其不变量，
            #       防止占位的全零输出被误当作真实 token 消费（详见该方法 docstring）。
            self._validate_pp_skip_output_comm(batch, result)

            hidden_state_offset = 0
            prefill_hidden_capture_mode = self._get_prefill_hidden_capture_mode(
                batch,
            )

            # Check finish conditions
            logprob_pt = 0  # input logprob 在扁平数组中的读取游标（pointer）

            for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
                    # 中译：混合批次里的 decode 请求、或已被回退（retract）的请求，跳过 prefill 处理。
                if (
                    batch.return_hidden_states
                    and logits_output.hidden_states is not None
                ):
                    assert extend_input_len_per_req is not None
                    hidden_state_offset = self._append_prefill_hidden_states(
                        req=req,
                        logits_output=logits_output,
                        hidden_state_offset=hidden_state_offset,
                        capture_hidden_mode=prefill_hidden_capture_mode,
                        extend_input_len=extend_input_len_per_req[i],
                        store=(
                            not req.finished()
                            and not req.is_retracted
                            and req.inflight_middle_chunks <= 0
                        ),
                    )

                if (
                    req.finished() and req.inflight_middle_chunks <= 0
                ) or req.is_retracted:
                    # Decode req in a mixed batch, or a retracted req. Keep an
                    # aborted middle chunk in the chunked branch long enough to
                    # drain its accounting without streaming it.
                    continue

                # 中译：inflight_middle_chunks<=0 表示这是该请求的最后一个 prefill chunk，
                #       prefill 至此完成，可以采纳首个生成 token；否则是中间分块（见 else）。
                if req.inflight_middle_chunks <= 0:
                    req.time_stats.set_prefill_finished_time()

                    # 中译：prefill 完成后产生的首个 token，追加进该请求的输出序列。
                    # 中译：若开启 reasoning，据该 token 更新「思考段/回答段」的边界统计。
                    # 中译：根据最新输出更新完成态（命中 stop token / 达到 max_new_tokens 等）。
                    if req.beam_group is not None:
                        # The relay point already replaced the sampled-token
                        # append; the group owns all finish semantics.
                        self.beam_coordinator.commit_prefill(
                            req, up_to_tick=batch.forward_iter
                        )
                    else:
                        # req output_ids are set here
                        req.output_ids.append(next_token_id)

                        self._maybe_update_reasoning_tokens(req, next_token_id)

                        req.update_finish_state()
                    if req.finished():
                        # 中译：刚生成首 token 就触发停止：收集专家/indexer 信息并释放 KV。
                        self._maybe_collect_routed_experts(req)
                        self._maybe_collect_indexer_topk(req)
                        release_kv_cache(req, self.tree_cache)
                        req.time_stats.set_completion_time()
                    elif not batch.decoding_reqs or req not in batch.decoding_reqs:
                        # 中译：未完成且不会立即进入本批 decode 的请求：把其 KV 前缀写入缓存以便复用。
                        maybe_cache_unfinished_req(req, self.tree_cache)
                        if get_memory().enable_hisparse:
                            self.hisparse_coordinator.admit_request_into_staging(req)

                    # 中译：收集模型自定义的逐请求附加信息（如有），按需累积到 req 上。
                    self._maybe_collect_customized_info(i, req, logits_output)

                    # 中译：请求要求返回 logprob 时，切出本请求对应的 input/output logprob；
                    #       logprob_pt 是扁平数组的读取游标，处理完后返回更新值供下个请求接续。
                    if batch.return_logprob:
                        logprob_pt = self._apply_prefill_logprobs(
                            req=req,
                            i=i,
                            logits_output=logits_output,
                            extend_input_len_per_req=extend_input_len_per_req,
                            extend_logprob_start_len_per_req=extend_logprob_start_len_per_req,
                            next_token_ids=next_token_ids,
                            logprob_pt=logprob_pt,
                        )

                    # 中译：请求要求返回 hidden states 时，从拼接后的 hidden_states 中按偏移切出
                    if req.return_sampling_mask:
                        self.add_sampling_mask_return_values(i, req, logits_output)

                    # 中译：约束解码（grammar，如 JSON/正则）下，用刚生成的 token 推进语法状态机，
                    #       以便下一步据语法约束屏蔽非法 token。
                    if req.grammar is not None:
                        self._apply_prefill_grammar(
                            req=req,
                            next_token_id=next_token_id,
                            already_advanced=result.grammar_advanced,
                        )

                else:
                    # being chunked reqs' prefill is not finished
                    # 中译：仍在分块中的请求，prefill 尚未结束——计数减一，等待后续 chunk。
                    req.inflight_middle_chunks -= 1
                    # There is only at most one request being currently chunked.
                    # Because this request does not finish prefill,
                    # we don't want to stream the request currently being chunked.
                    # 中译：同一时刻至多只有一个请求处于分块中；它 prefill 未完成，
                    #       因此本轮不对它做流式输出（用 skip_stream_req 标记）。
                    skip_stream_req = req

                    # Incrementally update input logprobs.
                    # 中译：分块 prefill 下增量更新 input logprob（每个 chunk 累计一段）。
                    if batch.return_logprob:
                        logprob_pt = self._apply_chunked_prefill_logprobs(
                            req=req,
                            i=i,
                            logits_output=logits_output,
                            extend_input_len_per_req=extend_input_len_per_req,
                            extend_logprob_start_len_per_req=extend_logprob_start_len_per_req,
                            logprob_pt=logprob_pt,
                        )

                    req.time_stats.set_last_chunked_prefill_finish_time()

            if auxiliary_output is not None:
                self.consume_auxiliary_output(
                    batch,
                    auxiliary_output,
                    auxiliary_output_starts,
                )

        else:  # embedding or reward model
            # 中译：embedding / reward 模型分支——产出的是向量而非 token，无需采样与解码，
            #       只需把 embedding 落到各请求上，并填一个占位 dummy token 走通完成流程。
            if result.copy_done is not None:
                result.copy_done.synchronize()

            embeddings = self._convert_embeddings(result=result)
            phs = result.pooled_hidden_states

            if phs is not None:
                if isinstance(phs, list):
                    phs = [t.cpu().detach() for t in phs]
                else:
                    phs = phs.cpu().detach()

            # Check finish conditions
            for i, req in enumerate(batch.reqs):
                if req.is_retracted:
                    continue

                req.embedding = embeddings[i]
                if req.return_pooled_hidden_states and phs is not None:
                    req.pooled_hidden_state = phs[i]
                if req.inflight_middle_chunks <= 0:
                    req.time_stats.set_prefill_finished_time()
                    # Dummy output token for embedding models
                    # 中译：embedding 模型没有真正的输出 token，填 0 作占位以复用统一的完成态逻辑。
                    req.output_ids.append(0)
                    req.update_finish_state()

                    if req.finished():
                        release_kv_cache(req, self.tree_cache)
                        req.time_stats.set_completion_time()
                    else:
                        maybe_cache_unfinished_req(req, self.tree_cache)
                else:
                    # being chunked reqs' prefill is not finished
                    req.inflight_middle_chunks -= 1
                    req.time_stats.set_last_chunked_prefill_finish_time()

        # 中译：把本批结果流式输出（跳过仍在分块中的 skip_stream_req）。
        self.token_to_kv_pool_allocator.free_group_end()
        self.output_streamer.stream_output(
            batch.reqs, batch.return_logprob, skip_stream_req
        )

        can_run_cuda_graph = result.can_run_cuda_graph
        # 中译：上报 prefill 阶段的统计指标（吞吐、是否走 CUDA graph 等）。
        # None on decode->extend converted batches; they are decode work and
        # have no prefill stats to report.
        if batch.prefill_stats is not None:
            self.metrics_reporter.report_prefill_stats(
                batch=batch,
                prefill_stats=batch.prefill_stats,
                can_run_cuda_graph=can_run_cuda_graph,
                dp_cooperation_info=batch.dp_cooperation_info,
            )

    def _convert_embeddings(self, *, result: EmbeddingBatchResult) -> list:
        # 中译：把前向产出的 embedding 张量转换为可序列化的 Python 结构。
        #       稀疏模式（sparse head）下转为「{token_id: value}」的稀疏字典列表；
        #       稠密模式下直接 tolist() 转为浮点列表。
        is_sparse = envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.is_set()

        embeddings = result.embeddings

        if is_sparse:
            batch_ids, token_ids = embeddings.indices()
            values = embeddings.values()

            embeddings = [{} for _ in range(embeddings.size(0))]
            for i in range(batch_ids.shape[0]):
                embeddings[batch_ids[i].item()][token_ids[i].item()] = values[i].item()
        else:
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.tolist()
            else:
                embeddings = [tensor.tolist() for tensor in embeddings]
        return embeddings

    def move_logprobs_to_cpu(
        self,
        *,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
    ) -> None:
        # 中译：仅当请求需要 logprob 时，把 logits_output 上各类 logprob 张量批量搬到 CPU
        #       （转为 list），供后续逐请求处理。
        if batch.return_logprob:
            if logits_output.next_token_logprobs is not None:
                logits_output.next_token_logprobs = (
                    logits_output.next_token_logprobs.tolist()
                )
            if logits_output.input_token_logprobs is not None:
                logits_output.input_token_logprobs = tuple(
                    logits_output.input_token_logprobs.tolist()
                )
            if logits_output.next_token_top_logprobs_val:
                logits_output.next_token_top_logprobs_val = [
                    v.tolist() for v in logits_output.next_token_top_logprobs_val
                ]
                logits_output.next_token_top_logprobs_idx = [
                    x.tolist() for x in logits_output.next_token_top_logprobs_idx
                ]
            if logits_output.next_token_token_ids_logprobs_val:
                logits_output.next_token_token_ids_logprobs_val = [
                    v.tolist() for v in logits_output.next_token_token_ids_logprobs_val
                ]

    def _apply_prefill_logprobs(
        self,
        *,
        req: Req,
        i: int,
        logits_output: LogitsProcessorOutput,
        extend_input_len_per_req: Optional[List[int]],
        extend_logprob_start_len_per_req: Optional[List[int]],
        next_token_ids: List[int],
        logprob_pt: int,
    ) -> int:
        # 中译：处理一次完成 prefill 的请求的 logprob：算出该请求的 input logprob 数量，
        #       追加其 input/output logprob 返回值，并把扁平游标 logprob_pt 前移后返回。
        assert extend_logprob_start_len_per_req is not None
        assert extend_input_len_per_req is not None
        extend_logprob_start_len = extend_logprob_start_len_per_req[i]
        extend_input_len = extend_input_len_per_req[i]

        num_input_logprobs = self.logprob_result_processor.calculate_num_input_logprobs(
            req,
            extend_input_len,
            extend_logprob_start_len,
        )

        if req.return_logprob:
            self.logprob_result_processor.add_logprob_return_values(
                i,
                req,
                logprob_pt,
                next_token_ids,
                num_input_logprobs,
                logits_output,
            )
        logprob_pt += num_input_logprobs
        return logprob_pt

    @staticmethod
    def _validate_pp_skip_output_comm(
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        """Validate PP skip output comm correctness.

        - When skip=True: all reqs must be middle chunks (inflight_middle_chunks > 0)
          so placeholder zeros are never consumed via req.output_ids.append().
        - When skip=False: at least one req should consume next_token_ids
          (inflight_middle_chunks <= 0), otherwise warn.

        中译：校验「流水线并行（PP）跳过纯分块批次输出通信」这一优化的不变量。
              - skip=True 时：批内所有请求都必须是中间分块（inflight_middle_chunks > 0），
                这样占位的全零输出就绝不会被 req.output_ids.append() 误消费；否则断言失败。
              - skip=False 时：至少应有一个请求消费了 next_token_ids（inflight_middle_chunks<=0），
                否则打告警（说明本可跳过通信却没跳，疑似回归）。
        """
        if not envs.SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM.get():
            return

        if not getattr(result, "skipped_output_comm", False):
            if batch.forward_mode.is_extend() and not batch.forward_mode.is_prebuilt():
                has_consumed_output = any(
                    req.inflight_middle_chunks <= 0
                    for req in batch.reqs
                    if not req.finished() and not req.is_retracted
                )
                if not has_consumed_output and len(batch.reqs) > 0:
                    chunks = list([r.inflight_middle_chunks for r in batch.reqs])
                    logger.warning(
                        f"PP non-skip output comm: no req consumed next_token_ids. "
                        f"contains_last_prefill_chunk={batch.contains_last_prefill_chunk}, "
                        f"num_reqs={len(batch.reqs)}, all inflight_middle_chunks={chunks}"
                    )
            return

        for req in batch.reqs:
            if not req.finished() and not req.is_retracted:
                assert req.inflight_middle_chunks > 0, (
                    f"PP skip output comm invariant violated: req {req.rid} "
                    f"has inflight_middle_chunks={req.inflight_middle_chunks} "
                    f"but output was skipped (contains_last_prefill_chunk="
                    f"{batch.contains_last_prefill_chunk}). "
                    f"Placeholder zeros would be appended to output_ids."
                )

    def _append_prefill_hidden_states(
        self,
        *,
        req: Req,
        logits_output: LogitsProcessorOutput,
        hidden_state_offset: int,
        capture_hidden_mode: CaptureHiddenMode,
        extend_input_len: int,
        store: bool = True,
    ) -> int:
        # 中译：从批次拼接的 hidden_states 中切出本请求对应的那一段（长度为其输入 token 数），
        # 中译：把 prefill 后生成的首 token 喂给该请求的 grammar（约束解码状态机），
        # 中译：FIXME——此 try/except 用于兜底 xgrammar 偶发的意外异常。
        if capture_hidden_mode.is_full():
            start = hidden_state_offset
            hidden_state_offset += extend_input_len
            if not store or not req.return_hidden_states:
                return hidden_state_offset

            req_hidden_states = logits_output.hidden_states[start:hidden_state_offset]
            if req.return_hidden_states is True:
                req.hidden_states.append(req_hidden_states.cpu().clone().tolist())
            elif req.return_hidden_states == "last":
                req.hidden_states.append(req_hidden_states[-1].cpu().tolist())
        elif capture_hidden_mode.is_last():
            index = hidden_state_offset
            hidden_state_offset += 1
            if store and req.return_hidden_states:
                req.hidden_states.append(
                    logits_output.hidden_states[index].cpu().tolist()
                )
        else:
            raise ValueError(
                f"Unexpected hidden states capture mode: {capture_hidden_mode}"
            )
        return hidden_state_offset

    @staticmethod
    def _append_decode_hidden_states(
        *,
        req: Req,
        hidden_states: torch.Tensor,
        start: int,
        accept_len: int,
    ) -> None:
        if accept_len <= 0:
            return

        if req.return_hidden_states == "last":
            valid_accept_len = accept_len
            if req.finished_len is not None:
                step_start = len(req.output_ids) - accept_len
                valid_accept_len = max(
                    0,
                    min(accept_len, req.finished_len - step_start),
                )
            if valid_accept_len > 0:
                req.hidden_states[:] = [
                    hidden_states[start + valid_accept_len - 1].cpu().tolist()
                ]
        else:
            req.hidden_states.extend(
                hidden_states[start : start + accept_len].cpu().tolist()
            )

    @staticmethod
    def _get_prefill_hidden_capture_mode(batch: ScheduleBatch) -> CaptureHiddenMode:
        return get_required_capture_hidden_mode(
            max(
                batch.return_hidden_states_mode,
                get_server_return_hidden_states_mode(),
            ),
            batch.spec_info,
        )

    def _apply_prefill_grammar(
        self, *, req: Req, next_token_id: int, already_advanced: bool = False
    ) -> None:
        # The grammar barrier may have already advanced the FSM over this prefilled
        # token (spec overlap path); only advance if not, but always sync
        # grammar.finished.
        if not already_advanced:
            # FIXME: this try-except block is for handling unexpected xgrammar issue.
            try:
                req.grammar.accept_token(next_token_id)
            except ValueError as e:
                # Grammar accept_token can raise ValueError if the token is not in the
                # grammar. This can happen if the grammar is not set correctly or the
                # token is invalid.
                logger.error(
                    f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                )
                req.to_finish = FINISH_ABORT()
        req.grammar.finished = req.finished()

    def _apply_chunked_prefill_logprobs(
        self,
        *,
        req: Req,
        i: int,
        logits_output: LogitsProcessorOutput,
        extend_input_len_per_req: Optional[List[int]],
        extend_logprob_start_len_per_req: Optional[List[int]],
        logprob_pt: int,
    ) -> int:
        # 中译：分块 prefill 场景下，仅对「本 chunk 新覆盖到的输入 token 区间」增量补充
        #       input logprob（last_prefill_chunk=False，表示还不是最后一块）。
        extend_logprob_start_len = extend_logprob_start_len_per_req[i]
        extend_input_len = extend_input_len_per_req[i]
        if extend_logprob_start_len < extend_input_len:
            # Update input logprobs.
            num_input_logprobs = (
                self.logprob_result_processor.calculate_num_input_logprobs(
                    req,
                    extend_input_len,
                    extend_logprob_start_len,
                )
            )
            if req.return_logprob:
                self.logprob_result_processor.add_input_logprob_return_values(
                    i,
                    req,
                    logits_output,
                    logprob_pt,
                    num_input_logprobs,
                    last_prefill_chunk=False,
                )
            logprob_pt += num_input_logprobs
        return logprob_pt

    def _resolve_spec_v2_tokens(
        self,
        result: GenerationBatchResult,
        batch: ScheduleBatch,
    ) -> List[List[int]]:
        """Resolve the padded next token ids for spec-v2 (overlap and non-overlap).

        中译：解析 spec-v2（投机解码 v2，含 overlap 与非 overlap）下被 padding 过的 next token。
              每个请求在张量里占固定 stride（= speculative_num_draft_tokens）长度，
              真正被接受（accept）的只有前 accept_lens[i] 个，需按此切出每请求的接受 token。
              accept_lens 含 bonus token，故每请求「correct drafts（不含 bonus）」= accept_lens-1，
              整批 num_correct_drafts = sum(accept_lens) - 请求数（每个请求各减去 1 个 bonus）。
        """
        assert result.next_token_ids.is_cpu
        assert result.accept_lens.is_cpu

        next_token_ids = result.next_token_ids.tolist()
        accept_lens = result.accept_lens.tolist()  # 每请求接受的 token 数（含 bonus）
        result.num_correct_drafts = sum(accept_lens) - len(batch.reqs)  # 整批正确草稿数（去 bonus）
        result.num_correct_drafts_per_req_cpu = [x - 1 for x in accept_lens]  # 每请求正确草稿数

        block_accept_lens = (
            result.block_accept_lens.tolist()
            if result.block_accept_lens is not None
            else None
        )
        result.num_block_accept_tokens = (
            sum(block_accept_lens) if block_accept_lens else 0
        )
        cap_lens = result.cap_lens.tolist() if result.cap_lens is not None else None
        result.num_cap_tokens = sum(cap_lens) if cap_lens else 0

        # Feed the adaptive controller now that accept_lens is on CPU,
        # instead of doing a synchronous GPU→CPU copy in the worker hot path.
        # BaseSpecWorker provides a no-op default for non-adaptive workers.
        # 中译：accept_lens 已在 CPU 上，趁此把每请求正确草稿数喂给自适应控制器，
        #       避免在 worker 热路径里做同步的 GPU→CPU 拷贝。非自适应 worker 是空实现。
        self.model_worker.on_verify_complete_cpu(
            result.num_correct_drafts_per_req_cpu, batch_size=len(batch.reqs)
        )

        # Advance the grammar FSM over this batch's committed tokens (idempotent):
        # the EAGLE overlap path already did this inside verify() via the grammar
        # barrier; otherwise advance now. advance_grammar_fsm self-gates on per-req
        # grammar (the queued batch.copy() does not carry has_grammar) and consumes
        # result.grammar_retained_tokens below instead of re-advancing.
        self.advance_grammar_fsm(result, batch)

        predict_tokens = []
        # In adaptive spec-v2, the worker state may already have switched when this
        # delayed result is processed. Use the draft token count recorded on result.
        # 中译：自适应 spec-v2 下，处理这条延迟结果时 worker 状态可能已切换，
        #       因此用 result 上记录的草稿 token 数（stride），而非当前 worker 的值。
        stride = result.speculative_num_draft_tokens
        assert stride is not None, "spec-v2 result missing speculative_num_draft_tokens"

        for i, req in enumerate(batch.reqs):
            # 中译：从扁平数组中第 i 个请求的 stride 段里，取前 accept_lens[i] 个为接受 token。
                # 中译：被回退的请求其已提交/已分配 KV 已被 reset_for_retract() 清零，跳过。
                    # 中译：EAGLE 在 prepare_for_decode 时预占了 bonus 槽位，完成时回退 1。
            # 中译：更新该请求「已提交 KV 长度」。不同投机算法对 bonus 槽的记账方式不同：
                # 中译：DFLASH 已物化「接受的草稿 token + bonus token」，直接加 accept_lens。
                # 中译：EAGLE 已预占 bonus 槽位，故只加 accept_lens-1 避免重复计。
            # 中译：累计该请求的正确草稿数，并更新其「正确草稿长度」直方图（用于自适应/统计）。
            accept_tokens = next_token_ids[i * stride : i * stride + accept_lens[i]]

            if req.is_retracted or req.finished():
                # Nothing to settle: no worker pre-claims the bonus, so
                # kv_committed_len already holds the committed prefix.
                pass
            else:
                if req.grammar is not None:
                    # FSM already advanced + truncated by advance_grammar_fsm; reuse
                    # the retained (grammar-legal) run instead of advancing again.
                    accept_tokens = result.grammar_retained_tokens[i]

                # Commit the full accepted run (drafts + bonus).
                num_accept_tokens = len(accept_tokens)
                req.kv.kv_committed_len += num_accept_tokens
                req.spec_verify_ct += 1

                num_correct_drafts = result.num_correct_drafts_per_req_cpu[i]
                req.spec_num_correct_drafts += num_correct_drafts
                req.update_spec_correct_drafts_histogram(num_correct_drafts)

                if block_accept_lens is not None:
                    req.spec_num_block_accept_tokens += block_accept_lens[i]
                if cap_lens is not None:
                    req.spec_num_cap_tokens += cap_lens[i]
                    req.update_spec_cap_lens_histogram(cap_lens[i])

            predict_tokens.append(accept_tokens)

        return predict_tokens

    def _accept_grammar_tokens(
        self, req: Req, tokens: Union[int, List[int]]
    ) -> List[int]:
        """Advance the grammar over the accepted token(s), stopping at the token
        that terminates it.

        ``tokens`` is a single sampled token (normal decode) or the whole
        verified run (spec decode). Returns the retained prefix; for spec the
        suffix past grammar completion is dropped so it is never committed to KV
        nor emitted. Advances the grammar FSM only -- ``grammar.finished`` is
        synced by the caller once the finish state is updated.
        """
        if isinstance(tokens, int):
            tokens = [tokens]
        retained = []
        try:
            for token_id in tokens:
                req.grammar.accept_token(token_id)
                retained.append(token_id)
                if req.grammar.is_terminated():
                    break
        except ValueError as e:
            # accept_token raises ValueError if the token is not in the grammar
            # (misconfigured grammar or invalid token); abort the request.
            logger.error(
                f"Grammar accept_token failed for req {req.rid} with token "
                f"{tokens}: {e}"
            )
            req.to_finish = FINISH_ABORT()
        return retained

    def advance_grammar_fsm(
        self, result: GenerationBatchResult, batch: ScheduleBatch
    ) -> None:
        """Advance each req's grammar FSM over the tokens THIS batch committed, and
        (for decode) memoize the grammar-truncated run on ``result``.

        This is the single place the spec-v2 FSM advances. It is idempotent
        (``result.grammar_advanced``) so it runs either eagerly — inside ``verify(N)``
        via the scheduler's grammar barrier, so the advance overlaps the target-verify
        forward — or lazily from the result processors on the non-overlap / non-EAGLE
        paths. It handles both decode (the accepted spec run) and extend (the single
        prefilled token) results, so the barrier can resolve whatever the previous
        batch was — e.g. the extend->decode boundary.
        """
        if result.grammar_advanced or not batch.has_grammar:
            return
        is_decode = batch.forward_mode.is_decode()
        if not (is_decode or batch.forward_mode.is_extend()):
            return
        if result.copy_done is not None:
            result.copy_done.synchronize()
        next_token_ids = result.next_token_ids.tolist()

        if not is_decode:
            # Extend: advance over the single token each completed-prefill req emitted
            # (mirrors process_batch_result_prefill's per-req token indexing).
            for i, req in enumerate(batch.reqs):
                if (
                    req.grammar is None
                    or req.is_retracted
                    or req.finished()
                    or req.inflight_middle_chunks > 0
                ):
                    continue
                self._accept_grammar_tokens(req, next_token_ids[i])
            result.grammar_advanced = True
            return

        # Decode: only the spec-v2 path reaches here (the grammar barrier for
        # spec-overlap workers and _resolve_spec_v2_tokens). Non-spec grammar decode
        # advances its FSM in process_batch_result_decode and has no accept_lens, so
        # bail out defensively.
        if result.accept_lens is None:
            return
        accept_lens = result.accept_lens.tolist()
        stride = result.speculative_num_draft_tokens
        assert stride is not None, "spec-v2 result missing speculative_num_draft_tokens"
        retained = [None] * len(batch.reqs)
        for i, req in enumerate(batch.reqs):
            if req.grammar is None or req.is_retracted or req.finished():
                continue
            accept_tokens = next_token_ids[i * stride : i * stride + accept_lens[i]]
            # Stop accepting once the grammar terminates so the over-drafted suffix
            # is never committed to KV nor emitted; this advances the FSM.
            retained[i] = self._accept_grammar_tokens(req, accept_tokens)
        result.grammar_retained_tokens = retained
        result.grammar_advanced = True

    def process_batch_result_idle(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        # 中译：处理空转（idle）批次——批内无实际请求（如为维持调度节奏跑的空批），
        #       只需同步拷贝事件、走一遍空闲流式输出即可。
        if result.copy_done is not None:
            result.copy_done.synchronize()

        self.output_streamer._stream_output_generation(
            batch.reqs, batch.return_logprob, is_idle_batch=True
        )

    def process_batch_result_decode(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        # 中译：处理 decode（逐 token 自回归）批次的前向结果。逐请求把新 token 追加进输出
        #       （非投机：1 个；投机：多个接受 token），更新完成态、按需做 logprob/hidden
        #       states/grammar 处理与 KV 释放，最后流式输出并上报 decode 指标。
        if result.copy_done is not None:
            result.copy_done.synchronize()
        auxiliary_output_starts = self.snapshot_auxiliary_output_starts(batch, result)
        auxiliary_output = result.auxiliary_host_output
        if result.routed_experts_output is not None:
            result.routed_experts_output.finalize()
            result.routed_experts_output = None
        if result.indexer_topk_output is not None:
            result.indexer_topk_output.finalize()
            result.indexer_topk_output = None

        logits_output, next_token_ids, can_run_cuda_graph = (
            result.logits_output,
            result.next_token_ids,
            result.can_run_cuda_graph,
        )

        # 中译：把 next_token_ids 归一化为 Python list（投机解码下为「每请求一个接受 token 列表」），
        #       并把 logprob 一并搬到 CPU。
        next_token_ids, next_token_logprobs = self._normalize_decode_outputs(
            batch=batch,
            result=result,
            logits_output=logits_output,
            next_token_ids=next_token_ids,
        )

        self.metrics_reporter.num_generated_tokens += len(batch.reqs)
        if not batch.spec_algorithm.is_none():
            self.metrics_reporter.update_spec_metrics(
                batch.batch_size(),
                result.num_correct_drafts,
                num_block_accept_tokens=result.num_block_accept_tokens,
                num_cap_tokens=result.num_cap_tokens,
            )
        if get_observability().enable_metrics:
            self.metrics_collector.increment_decode_cuda_graph_pass(
                value=can_run_cuda_graph
            )

        self.token_to_kv_pool_allocator.free_group_begin()  # 批量释放分组开始

        # Folds the relay point's selection into the DAG and sets the finish
        # states the loop below observes. Beam + spec is rejected at admission.
        newly_finished_beam_groups = set()
        if batch.spec_algorithm.is_none() and logits_output is not None:
            newly_finished_beam_groups = self.beam_coordinator.commit_decode(batch)

        for i, req in enumerate(batch.reqs):
            req: Req

            if req.beam_group is not None:
                # Under overlap a finished row reappears for one overshoot tick;
                # gate on the committing tick so this runs exactly once.
                if req.finished() and (
                    id(req.beam_group) not in newly_finished_beam_groups
                ):
                    continue
                req.time_stats.set_last_decode_finish_time()
                self._handle_finish_state_updated_req(
                    req, batch, result, i, logits_output
                )
                continue

            if (self.enable_overlap or self.enable_overlap_mlx) and (
                req.finished() or req.is_retracted
            ):
                # NOTE: This (req.finished() or req.is_retracted) should only happen when overlap scheduling is enabled.
                # And all the over-allocated tokens will be freed in `release_kv_cache`.
                # 中译：仅在 overlap 调度下，迭代中才会遇到已完成/已回退的请求（结果是上一步延迟而来）；
                #       此类请求多分配的 token 都会在 release_kv_cache 里释放，这里直接跳过。
                continue

            # 中译：非投机解码每步只接受 1 个 token；投机（v2）解码一步可能接受多个 token。
            # next_token_id is a per-req list: 1 token for non-spec, the verified
            # run for spec (already grammar-truncated in _resolve_spec_v2_tokens).
            next_token_id = next_token_ids[i]
            is_spec = not batch.spec_algorithm.is_none()

            req.output_ids.extend(next_token_id)
            new_accept_len = len(next_token_id)

            self._maybe_update_reasoning_tokens(req, next_token_id)
            req.time_stats.set_last_decode_finish_time()
            req.update_finish_state(new_accept_len)

            self._handle_finish_state_updated_req(req, batch, result, i, logits_output)

            if req.return_logprob:
                self._apply_decode_logprobs(
                    req=req,
                    i=i,
                    batch=batch,
                    next_token_id=next_token_id,
                    next_token_logprobs=next_token_logprobs,
                    logits_output=logits_output,
                )

            if req.return_sampling_mask:
                # return_sampling_mask + speculative decoding is rejected at
                # request entry, so this remains one support mask per token.
                self.add_sampling_mask_return_values(i, req, logits_output)

            if req.return_hidden_states and logits_output.hidden_states is not None:
                # hidden_states is [bs * stride, hidden_dim], one row per emitted
                # token; stride = speculative_num_draft_tokens for spec, 1 for non-spec.
                stride = result.speculative_num_draft_tokens or 1
                accept_len = len(next_token_id)
                start = i * stride
                self._append_decode_hidden_states(
                    req=req,
                    hidden_states=logits_output.hidden_states,
                    start=start,
                    accept_len=accept_len,
                )

            if req.grammar is not None:
                if not is_spec:
                    # Normal decode advances the grammar for its single token
                    # here; spec already advanced it in _resolve_spec_v2_tokens.
                    self._accept_grammar_tokens(req, next_token_id)
                req.grammar.finished = req.finished()

        if auxiliary_output is not None:
            self.consume_auxiliary_output(
                batch,
                auxiliary_output,
                auxiliary_output_starts,
            )

        self.output_streamer.stream_output(batch.reqs, batch.return_logprob)
        self.token_to_kv_pool_allocator.free_group_end()  # 批量释放分组结束、统一提交

        # 中译：decode 前向计数 +1，对 2^30 取模防止长期运行后整数无界增长。
        self.metrics_reporter.forward_ct_decode = (
            self.metrics_reporter.forward_ct_decode + 1
        ) % (1 << 30)
        self.metrics_reporter.report_decode_stats(
            can_run_cuda_graph,
            running_batch=batch,
            num_correct_drafts=result.num_correct_drafts,
        )

    def _normalize_decode_outputs(
        self,
        *,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        logits_output: LogitsProcessorOutput,
        next_token_ids: Union[torch.Tensor, List[int]],
    ) -> Tuple[Union[List[int], List[List[int]]], Optional[List[float]]]:
        # 中译：把 decode 输出归一化为 CPU 上的 Python 结构：
        #       投机解码 → 调 _resolve_spec_v2_tokens 得「每请求一个接受 token 列表」；
        #       MLX 路径已是 list[int]，跳过张量转换；否则 tolist()。需要时同步搬好 logprob。
        next_token_logprobs = None
        # Normalize to a uniform per-req list of accepted tokens (List[List[int]]):
        # spec unpacks the padded verify output; non-spec wraps its single token.
        if not batch.spec_algorithm.is_none():
            next_token_ids = self._resolve_spec_v2_tokens(result, batch)
        else:
            # CUDA workers return a device tensor, MLX a host list[int]; both -> list.
            ids = (
                next_token_ids.tolist()
                if torch.is_tensor(next_token_ids)
                else next_token_ids
            )
            next_token_ids = [[t] for t in ids]

        if batch.return_logprob:
            next_token_logprobs = logits_output.next_token_logprobs.tolist()
            if logits_output.next_token_top_logprobs_val:
                logits_output.next_token_top_logprobs_val = [
                    v.tolist() for v in logits_output.next_token_top_logprobs_val
                ]
                logits_output.next_token_top_logprobs_idx = [
                    x.tolist() for x in logits_output.next_token_top_logprobs_idx
                ]

            if logits_output.next_token_token_ids_logprobs_val:
                logits_output.next_token_token_ids_logprobs_val = [
                    v.tolist() for v in logits_output.next_token_token_ids_logprobs_val
                ]
        return next_token_ids, next_token_logprobs

    def _apply_decode_logprobs(
        self,
        *,
        req: Req,
        i: int,
        batch: ScheduleBatch,
        next_token_id: Union[int, List[int]],
        next_token_logprobs: list,
        logits_output: LogitsProcessorOutput,
    ) -> None:
        # 中译：归一化——非投机解码每步 1 个 token，投机解码每步可能有多个接受 token。
        # accepted_ids is already a per-req list; non-spec logprobs are flat, so
        # the scalar logprob still needs wrapping.
        if not batch.spec_algorithm.is_none():
            accepted_logprobs = next_token_logprobs[i]
            accepted_ids = next_token_id
            max_accept = len(accepted_logprobs)  # 该请求接受的 token 数（用于定位扁平 top-logprob）
        else:
            accepted_logprobs = [next_token_logprobs[i]]
            accepted_ids = next_token_id
            max_accept = 1

        for j, tok_id in enumerate(accepted_ids):
            req.logprob.output_token_logprobs_val.append(accepted_logprobs[j])
            req.logprob.output_token_logprobs_idx.append(tok_id)
            if req.logprob.top_logprobs_num > 0:
                # 中译：top-logprob 在批次里是按 (请求 i, 接受位 j) 扁平排布的，换算其下标。
                flat_idx = i * max_accept + j
                req.logprob.output_top_logprobs_val.append(
                    logits_output.next_token_top_logprobs_val[flat_idx]
                )
                req.logprob.output_top_logprobs_idx.append(
                    logits_output.next_token_top_logprobs_idx[flat_idx]
                )
            if req.logprob.token_ids_logprob is not None:
                flat_idx = i * max_accept + j
                req.logprob.output_token_ids_logprobs_val.append(
                    logits_output.next_token_token_ids_logprobs_val[flat_idx]
                )
                req.logprob.output_token_ids_logprobs_idx.append(
                    logits_output.next_token_token_ids_logprobs_idx[flat_idx]
                )

    def add_sampling_mask_return_values(
        self,
        i: int,
        req: Req,
        output: LogitsProcessorOutput,
    ) -> None:
        # 中译：把本步生成/接受的 token 喂给该请求的 grammar 状态机以推进约束解码；
        # 中译：FIXME——此 try/except 用于兜底 xgrammar 偶发的意外异常。
                # 中译：投机解码下 next_token_id 是「接受 token 列表」，逐个喂给文法。
        """Attach sparse sampling support metadata to the return values."""
        mask = output.next_token_sampling_mask_idx
        logprobs = output.next_token_sampling_logprobs
        req.output_token_sampling_mask.append(None if mask is None else mask[i])
        req.output_token_sampling_logprobs.append(
            None if logprobs is None else logprobs[i]
        )

    def _handle_finish_state_updated_req(
        self,
        req: Req,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        i: int,
        logits_output: LogitsProcessorOutput,
    ):
        # 中译：decode 单请求在 update_finish_state 之后的统一收尾：mamba 状态维护、
        # 中译：必须在 update_finish_state 之后调用，确保内部用到的 req.finished() 已是最新值。
        lazy = mamba_extra_buffer_lazy_enabled()
        known_mamba_boundary = None
        completed_mamba_boundary = None
        lookahead = 0
        if batch.mamba_track_mask_cpu is not None:
            completed_mamba_boundary = bool(batch.mamba_track_mask_cpu[i])
            lookahead = req.decode_batch_idx - batch.mamba_decode_batch_idx_cpu[i]
            assert lookahead in (0, 1), (
                f"mamba result lookahead={lookahead} for req {req.rid}; "
                "overlap advanced more than one decode batch"
            )
            if lookahead == 0:
                known_mamba_boundary = bool(batch.mamba_track_mask_cpu[i])
            else:
                known_mamba_boundary = bool(batch.mamba_track_mask_next_cpu[i])

            if completed_mamba_boundary and not lazy:
                req.mamba_last_track_idx = batch.mamba_track_buffer_indices[i]
                req.mamba_last_track_seqlen = req.kv.kv_committed_len - lookahead
            elif (
                req.finished()
                and lazy
                and lookahead == 1
                and known_mamba_boundary
                and req.mamba_next_track_idx == req.mamba_last_track_idx
            ):
                req.mamba_lazy_is_insert = False

        # Called here (after update_finish_state) so req.finished() is valid
        # for mamba_lazy_post_decode_at_boundary inside.
        should_update = completed_mamba_boundary if lazy else known_mamba_boundary
        if should_update is None or should_update:
            self._mamba_prefix_cache_update(
                req,
                batch,
                result,
                i,
                known_boundary=not lazy and known_mamba_boundary is True,
            )

        if (
            get_disagg().disaggregation_decode_enable_offload_kvcache
            and not req.finished()
        ):
            # 中译：开启 decode 端 KV 卸载且请求未完成：把其 KV 异步卸载到主机内存以省显存。
            self.decode_offload_manager.offload_kv_cache(req)

        if req.finished():
            # isinstance narrowing: create_worker may also return plain
            # TpModelWorker-based drafts, which carry no spec-worker hooks.
            if isinstance(self.draft_worker, BaseSpecWorker):
                self.draft_worker.note_request_finished(
                    rid=req.rid,
                    natural_stop=isinstance(req.finished_reason, FINISH_MATCHED_TOKEN),
                )

            # delete feature to save memory
            # 中译：请求已完成——释放多模态特征以省内存（无会话复用需求时）。
            if req.multimodal_inputs is not None and req.session is None:
                req.multimodal_inputs.release_features()
            self._maybe_collect_routed_experts(req)
            self._maybe_collect_indexer_topk(req)

            if get_disagg().disaggregation_decode_enable_offload_kvcache:
                # Asynchronously offload KV cache; release_kv_cache will be called after Device->Host transfer completes
                # 中译：异步卸载 KV；待 Device→Host 传输完成后再真正 release_kv_cache。
                #       若卸载未启动（返回 False），则立即走完成时的兜底释放。
                if not self.decode_offload_manager.offload_kv_cache(req):
                    self.decode_offload_manager.finalize_release_on_finish(req)
            else:
                if get_memory().enable_hisparse:
                    self.hisparse_coordinator.request_finished(req)
                # 中译：若 worker 提供了 KV 释放前的准备钩子（可选），先调用它。
                prepare_release = getattr(
                    self.model_worker, "prepare_for_kv_cache_release", None
                )
                if callable(prepare_release):
                    prepare_release(req)
                is_insert = (
                    req.mamba_lazy_is_insert
                    if mamba_extra_buffer_lazy_enabled()
                    else True
                )
                release_kv_cache(req, self.tree_cache, is_insert=is_insert)

            req.time_stats.set_completion_time()

        self._maybe_collect_customized_info(i, req, logits_output)

    def _maybe_update_reasoning_tokens(
        self,
        req: Req,
        next_token_id: Union[int, List[int]],
    ):
        # 中译：若请求开启了推理（reasoning）且模型定义了「思考结束」token，则据新 token
        think_end_ids = self.model_config.think_end_ids
        if req.require_reasoning and think_end_ids:
            req.update_reasoning_tokens(next_token_id, think_end_ids)

    def _mamba_prefix_cache_update(
        self,
        req: Req,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        i: int,
        known_boundary: bool = False,
    ) -> None:
        """Update mamba track state at ping-pong boundaries.

        Non-lazy: swap the ping-pong index so the next forward writes to
        the alternate slot.
        Lazy: keep the same index (prealloc handles the swap) and run
        post-decode cleanup to free the temporary second slot.

        中译：在 mamba「乒乓（ping-pong）」边界处更新追踪状态（mamba 用双槽轮换保存 SSM 状态）。
              非 lazy 模式：切换乒乓下标，使下一次前向写入另一个槽。
              lazy 模式：保持下标不变（由预分配负责切换），并在 decode 后做清理以释放临时的第二槽。
              仅在跨越追踪间隔（track interval）边界时才动作，否则直接返回。
        """
        if req.mamba_ping_pong_track_buffer is None:
            return

        lazy = mamba_extra_buffer_lazy_enabled()
        if known_boundary:
            self._mamba_assert_committed_len_lookahead(req)
            track_seqlen = req.kv.kv_committed_len
            assert track_seqlen % mamba_track_grid(self.tree_cache.page_size) == 0
            at_boundary = True
        else:
            at_boundary, track_seqlen = self._mamba_check_track_boundary(
                req, batch, result, i
            )

        if lazy and not batch.spec_algorithm.is_none():
            # For spec, at_boundary means this step actually crossed an interval.
            self._mamba_lazy_spec_update(req, batch, i, at_boundary, track_seqlen)
            return

        if not at_boundary:
            return

        track_idx = req.mamba_next_track_idx
        if not known_boundary and batch.mamba_track_buffer_indices is not None:
            track_idx = batch.mamba_track_buffer_indices[i]
        if not known_boundary:
            req.mamba_last_track_seqlen = track_seqlen
        if lazy:
            self.mamba_lazy_post_decode_at_boundary(req, batch, track_idx)
        else:
            if not known_boundary:
                req.mamba_last_track_idx = track_idx
            req.mamba_next_track_idx = (
                batch.req_to_token_pool.get_mamba_ping_pong_other_idx(track_idx)
            )

    def _mamba_lazy_spec_update(
        self, req: Req, batch: ScheduleBatch, i: int, crossed: bool, track_seqlen: int
    ) -> None:
        """Lazy + spec post-processing.

        Running req with a confirmed crossing: the commit scattered the
        crossing state into the planned position; promote it to keep unless
        the plan already pointed at the keep slot (in-place fallback, or
        promoted by an earlier overlapped confirmation).
        Finishing req: donate the keep slot only if no scatter wrote it
        (this step) or may still write it (the in-flight next step).
        """
        positions = batch.mamba_lazy_spec_track_positions_cpu
        planned_pos = (
            positions[i]
            if positions is not None and i < len(positions)
            else None  # filtered/merged snapshot without a plan: be conservative
        )

        if req.finished():
            # Skip the donation if a scatter wrote or may still write the keep slot.
            keep_written_by_this_step = (
                crossed and planned_pos == req.mamba_next_track_idx
            )
            other_idx = 1 - req.mamba_next_track_idx
            # Recompute the in-flight verify's plan (kv_committed_len is
            # frozen since its prepare, so the recompute is exact).
            keep_may_be_written_in_flight = req.mamba_ping_pong_track_buffer[
                other_idx
            ].item() == -1 and mamba_lazy_spec_in_window(
                req,
                mamba_track_grid(self.tree_cache.page_size),
                max_speculative_num_draft_tokens(),
            )
            if (
                planned_pos is None
                or keep_written_by_this_step
                or keep_may_be_written_in_flight
            ):
                req.mamba_lazy_is_insert = False
            return

        if not crossed or planned_pos is None:
            return
        if planned_pos != req.mamba_next_track_idx:
            # Promote pending -> keep: free the old checkpoint, repoint.
            pool = batch.req_to_token_pool
            keep_idx = req.mamba_next_track_idx
            keep_val = req.mamba_ping_pong_track_buffer[keep_idx]
            pool.mamba_allocator.free(keep_val.unsqueeze(0))
            pool.set_mamba_ping_pong_slot(req, keep_idx, -1)
            req.mamba_next_track_idx = planned_pos
        # else: in-place fallback, or promoted by an earlier confirmation —
        # keep holds the track_seqlen state either way.
        req.mamba_last_track_idx = planned_pos
        req.mamba_last_track_seqlen = track_seqlen

    @staticmethod
    def _mamba_assert_committed_len_lookahead(req: Req) -> None:
        """Alarm if overlap advances beyond the scheduler's modeled window."""
        assert req.output_ids, (
            "mamba track boundary reached before a decode token was appended "
            f"(req {req.rid}); output_ids is empty"
        )
        token_seq_len = len(req.origin_input_ids) + len(req.output_ids) - 1
        assert (req.kv.kv_committed_len - token_seq_len) in (0, 1), (
            f"mamba track boundary: kv_committed_len={req.kv.kv_committed_len} "
            f"leads seq_len={token_seq_len} by more than one (req {req.rid}); "
            "overlap lookahead wider than assumed"
        )

    def _mamba_check_track_boundary(self, req, batch, result, i):
        """Check if this decode step crosses a mamba track interval boundary.

        Returns (at_boundary, track_seqlen).  The boundary condition
        matches what the forward's tracking mask used:
        ``prepare_for_decode`` increments both ``seq_lens_cpu`` and
        ``kv_committed_len`` by 1, then checks
        ``seq_lens_cpu % interval == 0``.  Subtracting the overlap
        lookahead from ``kv_committed_len`` reproduces that check.

        For spec decode, the boundary is detected by comparing the
        accepted seq_len range against interval boundaries.

        中译：判断本次 decode 是否跨越了 mamba 追踪间隔的边界，返回 (是否在边界, 该边界的 seqlen)。
              边界判定要与前向里 tracking mask 用的一致：prepare_for_decode 会把 seq_lens_cpu 与
              kv_committed_len 都 +1，再判断 seq_lens_cpu % interval == 0；这里用 kv_committed_len
              复现该判断（其值必为 interval 的整数倍，故页对齐）。
              投机解码下则通过比较「接受的 seqlen 区间」是否跨过 interval 边界来判定。
        """
        interval = mamba_track_grid(self.tree_cache.page_size)

        if batch.spec_algorithm.is_none():
            lookahead = req.decode_batch_idx - batch.mamba_decode_batch_idx_cpu[i]
            committed_len = req.kv.kv_committed_len - lookahead
            if committed_len % interval == 0:
                return True, committed_len
        elif result.num_correct_drafts_per_req_cpu is not None:
            cur = req.seqlen - 1
            prev = cur - result.num_correct_drafts_per_req_cpu[i] - 1
            if cur // interval != prev // interval:
                return True, cur // interval * interval

        return False, 0

    def mamba_lazy_post_decode_at_boundary(
        self, req: Req, batch: ScheduleBatch, track_idx: int
    ):
        """Commit a completed lazy-mode boundary and free its old slot."""
        req.mamba_last_track_idx = track_idx
        req.mamba_next_track_idx = track_idx
        other_idx = 1 - track_idx
        other_val = req.mamba_ping_pong_track_buffer[other_idx].item()
        if other_val != -1:
            pool = batch.req_to_token_pool
            pool.mamba_allocator.free(
                req.mamba_ping_pong_track_buffer[other_idx].unsqueeze(0)
            )
            pool.set_mamba_ping_pong_slot(req, other_idx, -1)
