from __future__ import annotations

import logging
from array import array
from http import HTTPStatus
from typing import TYPE_CHECKING, List

import torch

from sglang.srt.mem_cache.common import maybe_cache_unfinished_req
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.server_args import ServerArgs


class ScheduleBatchDisaggregationDecodeMixin:
    """PD 分离（prefill/decode disaggregation）中、decode 实例专用的 ScheduleBatch 混入类。

    在 PD 分离架构下，prefill 实例完成首次前向且把 KV 传输给 decode 实例后，decode 侧
    拿到的是一批「已经预构建好」（prebuilt）的请求：它们的 KV 已存在于本地 KV 池，
    无需再真正跑一次 extend/prefill。本混入类提供两个阶段的方法：
    - ``prepare_for_prebuilt``：为这批请求填好一次 PREBUILT 前向所需的各项元数据；
    - ``process_prebuilt``：处理首个待解码 token（语法接受、投机草稿输入、future 映射）。
    """

    def prepare_for_prebuilt(self: ScheduleBatch):
        """通过填充元数据来准备一次「预构建的 extend」。

        改编自 ``prepare_for_extend()``：区别在于这里的请求 KV 已由 prefill 实例传入并落到
        本地 KV 池，因此无需重新分配/计算 KV，只需把前向所需的张量与字段组装好。
        """

        # 标记为 PREBUILT 前向模式（区别于普通的 EXTEND / DECODE）。
        self.forward_mode = ForwardMode.PREBUILT
        reqs = self.reqs
        # 每个请求的待填充 token：去掉已命中前缀（prefix_indices）后的那部分输入。
        input_ids = [r.get_fill_ids()[len(r.prefix_indices) :] for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = []
        pre_lens = []
        req_pool_indices = []

        # 预先算出总长度（所有请求 extend 部分的 token 总数），一次性分配输出缓存位置张量。
        total_size = sum(req.extend_input_len for req in reqs)
        out_cache_loc = torch.empty(total_size, dtype=torch.int64, device=self.device)

        # 单趟遍历填充该张量（避免多次拼接/拷贝）。
        offset = 0
        for i, req in enumerate(reqs):
            req_pool_indices.append(req.req_pool_idx)
            pre_len = len(req.prefix_indices)

            # 从 req_to_token 映射中取出该请求 extend 部分对应的 KV 槽位索引，
            # 拼到全局的 out_cache_loc 里（告诉前向：输出写到这些 KV 槽位）。
            chunk = self.req_to_token_pool.req_to_token[req.req_pool_idx][
                pre_len : pre_len + req.extend_input_len
            ]
            assert (
                offset + req.extend_input_len <= total_size
            ), f"Exceeds total size: offset={offset}, req.extend_input_len={req.extend_input_len}, total_size={total_size}"
            out_cache_loc[offset : offset + req.extend_input_len] = chunk
            offset += req.extend_input_len

            # 序列长度 = 原始输入长度 + 已生成输出（减 1，因最后一个输出 token 本次才待解码）。
            seq_len = len(req.origin_input_ids) + max(0, len(req.output_ids) - 1)
            seq_lens.append(seq_len)
            if len(req.output_ids) == 0:
                # 尚未产出任何输出时，序列只包含前缀 + 本次 extend，两者应严格对应。
                assert (
                    seq_len - pre_len == req.extend_input_len
                ), f"seq_len={seq_len}, pre_len={pre_len}, req.extend_input_len={req.extend_input_len}"

            if not req.retracted_stain:
                # 限幅（clamp）以避免重复计数：already_computed 在 _commit_transfer_to_req 里
                # 已用 prefill 上报的 cached_tokens 初始化，因此当 decode 侧的前缀短于 prefill
                # 上报值时，不应从 cached_tokens 中减去。
                delta = max(0, pre_len - req.already_computed)
                req.cached_tokens += delta
                req.cached_tokens_device += delta
                req.already_computed = seq_len
            req.is_retracted = False
            pre_lens.append(pre_len)
            # PREBUILT 阶段不重算 prefill 的 logprob，起始位置置 0。
            req.extend_logprob_start_len = 0

        extend_input_logprob_token_ids = None

        # 把逐请求收集到的元数据写回 batch 字段（同时准备 device 与 cpu 两份，供不同路径使用）。
        self.input_ids = torch.tensor(
            sum(input_ids, array("q")), dtype=torch.int32, device=self.device
        )
        self.req_pool_indices = torch.tensor(
            req_pool_indices, dtype=torch.int64, device=self.device
        )
        self.req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=self.device)
        self.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        self.orig_seq_lens = torch.tensor(
            seq_lens, dtype=torch.int32, device=self.device
        )
        self.out_cache_loc = out_cache_loc
        self.seq_lens_sum = sum(seq_lens)

        if self.return_logprob:
            # 需返回 logprob 时，额外收集每个请求的 top-k 数量与指定 token 集合。
            self.top_logprobs_nums = [r.logprob.top_logprobs_num for r in reqs]
            self.token_ids_logprobs = [r.logprob.token_ids_logprob for r in reqs]

        self.extend_num_tokens = extend_num_tokens
        self.prefix_lens = [len(r.prefix_indices) for r in reqs]
        self.extend_lens = [r.extend_input_len for r in reqs]
        self.extend_logprob_start_lens = [r.extend_logprob_start_len for r in reqs]
        self.extend_input_logprob_token_ids = extend_input_logprob_token_ids
        self.multimodal_inputs = [r.multimodal_inputs for r in reqs]

        # 构建采样信息（温度、top-p/top-k、惩罚项等），供后续采样使用。
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    def process_prebuilt(
        self: ScheduleBatch,
        server_args: ServerArgs,
        future_map: FutureMap,
    ):
        """把缓存的「最后一个输入 token」指派给调度 batch。

        prefill 实例已生成首个输出 token 并随 KV 一起传给 decode。这里把这个 token 作为
        decode 首次迭代的输入：若启用投机解码则构建草稿输入，否则通过 future_map 暂存。
        同时处理约束解码（grammar）对首 token 的接受。
        """
        last_tokens: List[int] = []
        for req in self.reqs:
            # prefill 已产出的首个输出 token（作为本次 decode 的输入）。
            last_tokens.append(req.output_ids[-1])
            # 未完成请求按需回填前缀缓存（把已有序列插入 radix 树）。
            maybe_cache_unfinished_req(req, self.tree_cache)
            if req.grammar is not None:
                # FIXME: 这个 try-except 是为了处理 xgrammar 的意外异常。
                try:
                    # 若 current_token 不为 None，说明该 grammar 来自一个被回退（retracted）的请求，
                    # 该 token 已被接受过，不应重复接受。
                    if req.grammar.current_token is None:
                        req.grammar.accept_token(req.output_ids[-1])
                except ValueError as e:
                    from sglang.srt.managers.schedule_batch import FINISH_ABORT

                    # 若 token 不在该 grammar 允许的集合内，accept_token 会抛 ValueError。
                    # 这可能发生在 grammar 未正确设置或 token 非法时。
                    # 用 to_finish（而非 finished_reason），以便 process_batch_result_prebuilt
                    # 统一经由 update_finish_state -> release_kv_cache 在一处释放资源。
                    error_message = f"Grammar accept_token failed for req {req.rid} with token {req.output_ids[-1]}: {e}"
                    req.to_finish = FINISH_ABORT(
                        error_message, HTTPStatus.INTERNAL_SERVER_ERROR
                    )
                req.grammar.finished = req.finished()
        last_tokens_tensor = torch.tensor(
            last_tokens, dtype=torch.int64, device=self.device
        )

        # 交由投机解码算法基于首 token 构建分离场景的草稿输入（若启用）。
        spec_info = self.spec_algorithm.build_disagg_draft_input(
            self,
            server_args,
            last_tokens_tensor,
            future_map,
        )
        if spec_info is not None:
            self.spec_info = spec_info
        else:
            # 非投机场景：把首 token 暂存到 relay，使首次 DECODE 的 resolve_forward_inputs
            # 能像处理其他解码迭代那样收集到它。
            future_map.stash(self.req_pool_indices, last_tokens_tensor)
            self.input_ids = None
