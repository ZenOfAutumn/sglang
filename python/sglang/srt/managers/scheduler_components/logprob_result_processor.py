# 中译：本模块负责「logprob（对数概率）结果处理」，是 Scheduler 侧从模型前向输出
#       (LogitsProcessorOutput) 中提取并组织各类 logprob 的工具类所在。
#       主要处理两大类 logprob，并各自区分「input（输入/prompt 部分）」与「output（生成部分）」：
#         - token logprob：被选中 token 的对数概率（val）及其 token id（idx）。
#         - top logprob：每个位置上概率最高的若干 token 的 logprob 及 id。
#         - token_ids logprob：用户指定的一组特定 token id 的 logprob。
#       难点一：增量/分块（chunked）prefill 下，input logprob 要跨多个 chunk 累积，
#               只有在最后一个 chunk（last_prefill_chunk）才一次性整理成最终结果。
#       难点二：multi-item scoring（MIS，多条目打分）模式下，只有「分隔符 token」所在位置
#               才有 logprob，对齐规则与普通请求不同（普通请求会在头部补 None、并丢弃末尾
#               的采样 token），故大量逻辑要按 is_multi_item_scoring 分流。
from __future__ import annotations

from dataclasses import dataclass
from typing import (
    List,
    Tuple,
)

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.server_args import (
    MIS_DELIMITER_TOKEN_ID,
    ServerArgs,
)


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerLogprobResultProcessor:
    # 中译：Scheduler 侧的 logprob 结果处理器。frozen=True 表示实例不可变（只读配置），
    #       它本身不持有可变状态，所有「写入」都作用在传入的 Req 对象上。
    server_args: ServerArgs
    model_config: ModelConfig

    def _process_input_token_logprobs(
        self, req: Req, input_token_logprobs: List
    ) -> None:
        """Process input token logprobs values and indices.

        中译：整理 input（prompt 部分）token logprob 的「值（val）」与「索引（idx，即 token id）」。
        """
        is_multi_item_scoring = self._is_multi_item_scoring(req)

        # Process logprob values - handle multi-item scoring vs regular requests
        # 中译：处理 logprob 值——区分 MIS 与普通请求。
        if is_multi_item_scoring:
            # Multi-item scoring: use all logprobs as-is
            # 中译：MIS 模式：所有 logprob 原样使用（每个分隔符位置一项，无需对齐偏移）。
            req.logprob.input_token_logprobs_val = input_token_logprobs
        else:
            # Regular request: add None at start, remove last (sampling token)
            # 中译：普通请求：首位补 None（第一个 token 无前驱、无 logprob），
            #       并去掉末尾那个用于采样的 token（它属于 output 而非 input）。
            req.logprob.input_token_logprobs_val = [None] + input_token_logprobs[:-1]

        # Process logprob indices based on scoring type
        # 中译：按打分类型处理 logprob 索引（token id 序列）。
        if is_multi_item_scoring:
            # MIS scores come from input_token_ids_logprobs, not input_token_logprobs.
            # But the shared pipeline requires input_token_logprobs_idx to be the same
            # length as input_token_logprobs_val (validated at line 816). We fill with
            # MIS_DELIMITER_TOKEN_ID as a dummy — score_request() ignores this field.
            # 中译：MIS 的分数实际来自 input_token_ids_logprobs，而非这里的 input_token_logprobs；
            #       但共享流水线要求 idx 与 val 等长（下游有长度校验），故用 MIS_DELIMITER_TOKEN_ID
            #       占位填充。score_request() 会忽略该字段，所以填占位值是安全的。
            delimiter_count = len(req.multi_item_delimiter_indices)
            input_token_logprobs_idx = [MIS_DELIMITER_TOKEN_ID] * delimiter_count
        else:
            # Regular request: include all tokens from logprob_start_len onwards
            # 中译：普通请求：取 logprob_start_len 之后的全部 token id 作为索引。
            input_token_logprobs_idx = req.origin_input_ids[req.logprob_start_len :]

        # Clip padded hash values from image tokens to prevent detokenization errors
        # 中译：图像 token 在 input_ids 中是「填充的哈希值」，可能超出词表范围；
        #       这里把 >= vocab_size-1 的值截断为 0，避免后续反向解码出错。
        req.logprob.input_token_logprobs_idx = [
            x if x < self.model_config.vocab_size - 1 else 0
            for x in input_token_logprobs_idx
        ]

    def _process_input_top_logprobs(self, req: Req) -> None:
        """Process input top logprobs.

        中译：整理 input 部分的 top logprob（每个位置概率最高的若干 token）。
              这些值在各 prefill chunk 中暂存于 req.temp_*，此处汇总成最终结果。
        """
        # 中译：top_logprobs_num <= 0 表示用户未请求 top logprob，直接跳过。
        if req.logprob.top_logprobs_num <= 0:
            return

        is_multi_item_scoring = self._is_multi_item_scoring(req)

        # Initialize arrays - multi-item scoring starts empty, others start with None
        # 中译：初始化数组——MIS 从空开始；普通请求首位补 None（与 token logprob 对齐规则一致）。
        req.logprob.input_top_logprobs_val = [] if is_multi_item_scoring else [None]
        req.logprob.input_top_logprobs_idx = [] if is_multi_item_scoring else [None]

        # Extend arrays with temp values
        # 中译：把各 chunk 暂存的 temp 值依次摊平（extend）到最终数组中。strict=True 确保 val/idx 等长。
        for val, idx in zip(
            req.temp_input_top_logprobs_val,
            req.temp_input_top_logprobs_idx,
            strict=True,
        ):
            req.logprob.input_top_logprobs_val.extend(val)
            req.logprob.input_top_logprobs_idx.extend(idx)

        # Remove last token (sampling token) for non multi-item scoring requests
        # 中译：普通请求去掉末尾的采样 token（它属于 output），MIS 不做此处理。
        if not is_multi_item_scoring:
            req.logprob.input_top_logprobs_val.pop()
            req.logprob.input_top_logprobs_idx.pop()

        # Clean up temp storage
        # 中译：清空临时存储，释放内存（结果已转入最终字段）。
        req.temp_input_top_logprobs_idx = None
        req.temp_input_top_logprobs_val = None

    def _process_input_token_ids_logprobs(self, req: Req) -> None:
        """Process input token IDs logprobs.

        中译：整理 input 部分「指定 token id 列表」的 logprob（用户通过 token_ids_logprob 指定）。
        """
        # 中译：未指定要查询的 token id 列表则跳过。
        if req.logprob.token_ids_logprob is None:
            return

        is_multi_item_scoring = self._is_multi_item_scoring(req)

        # Initialize arrays - multi-item scoring starts empty, others start with None
        # 中译：初始化数组——MIS 从空开始；普通请求首位补 None。
        req.logprob.input_token_ids_logprobs_val = (
            [] if is_multi_item_scoring else [None]
        )
        req.logprob.input_token_ids_logprobs_idx = (
            [] if is_multi_item_scoring else [None]
        )

        # Process temp values - convert tensors to lists and extend arrays
        # 中译：处理暂存值——必要时把张量转成 list，再摊平到最终数组。
        for val, idx in zip(
            req.temp_input_token_ids_logprobs_val,
            req.temp_input_token_ids_logprobs_idx,
            strict=True,
        ):
            # 中译：单个 token 的 logprob 可能是标量（非 list），统一包装成 list 再 extend。
            val_list = val.tolist() if isinstance(val, torch.Tensor) else val
            req.logprob.input_token_ids_logprobs_val.extend(
                val_list if isinstance(val_list, list) else [val_list]
            )
            req.logprob.input_token_ids_logprobs_idx.extend(idx)

        # Remove last token (sampling token) for non multi-item scoring requests
        # 中译：普通请求去掉末尾采样 token；MIS 不处理。
        if not is_multi_item_scoring:
            req.logprob.input_token_ids_logprobs_val.pop()
            req.logprob.input_token_ids_logprobs_idx.pop()

        # Clean up temp storage
        # 中译：清空临时存储。
        req.temp_input_token_ids_logprobs_idx = None
        req.temp_input_token_ids_logprobs_val = None

    def _calculate_relevant_tokens_len(self, req: Req) -> int:
        """Calculate the expected length of logprob arrays based on whether multi-item scoring is enabled.

        For multi-item scoring, only delimiter positions have logprobs.
        For regular requests, all positions from logprob_start_len onwards have logprobs.

        中译：计算 logprob 数组的「期望长度」，用于后续断言校验是否对齐。
              - MIS：只有分隔符位置有 logprob，长度 = 分隔符个数。
              - 普通请求：logprob_start_len 之后所有位置都有 logprob。
        """
        is_multi_item_scoring = self._is_multi_item_scoring(req)

        if is_multi_item_scoring:
            return len(req.multi_item_delimiter_indices)
        else:
            return len(req.origin_input_ids[req.logprob_start_len :])

    def calculate_num_input_logprobs(
        self,
        req: Req,
        extend_input_len: int,
        extend_logprob_start_len: int,
    ) -> int:
        """Calculate the number of input logprobs based on whether multi-item scoring is enabled.

        For multi-item scoring, only delimiter positions have logprobs.
        For regular requests, all positions in the range have logprobs.

        中译：计算「本次 extend（prefill）范围内」应有的 input logprob 数量。
              该数量用于从前向输出中切取本 chunk 对应的 logprob 片段。
        """
        is_multi_item_scoring = self._is_multi_item_scoring(req)

        if is_multi_item_scoring:
            # Count pre-computed delimiter indices within the extend range
            # 中译：MIS——只统计落在本 extend 区间 [start, input_len) 内的分隔符个数。
            return sum(
                1
                for idx in req.multi_item_delimiter_indices
                if extend_logprob_start_len <= idx < extend_input_len
            )
        else:
            # Regular request: all tokens in the range
            # 中译：普通请求——区间内每个 token 都有 logprob，数量即区间长度。
            return extend_input_len - extend_logprob_start_len

    def _is_multi_item_scoring(self, req: Req) -> bool:
        """Check if request uses multi-item scoring.

        Multi-item scoring applies to prefill-only requests when a delimiter
        token is configured. In this mode, only positions containing the
        delimiter token receive logprobs.

        中译：判断该请求是否走 multi-item scoring（多条目打分）模式。
              三个条件需同时满足：服务端开启 enable_mis、请求是 prefill-only、且配置了分隔符索引。
        """
        return (
            self.server_args.enable_mis
            and req.is_prefill_only
            and req.multi_item_delimiter_indices is not None
        )

    def add_input_logprob_return_values(
        self,
        i: int,
        req: Req,
        output: LogitsProcessorOutput,
        logprob_pt: int,
        num_input_logprobs: int,
        last_prefill_chunk: bool,  # If True, it means prefill is finished.
    ):
        """Incrementally add input logprobs to `req`.

        Args:
            i: The request index in a batch.
            req: The request. Input logprobs inside req are modified as a
                consequence of the API
            logprob_pt: Pointer into the prefill ids processed.
            output: Logit processor output that's used to compute input logprobs
            last_prefill_chunk: True if it is the last prefill (when chunked).
                Some of input logprob operation should only happen at the last
                prefill (e.g., computing input token logprobs).

        中译：增量地把 input logprob 累加到 req 上（支持分块 prefill）。
              每个 chunk 调用一次：本 chunk 的 token logprob 追加到 req.input_token_logprobs，
              top / token_ids logprob 暂存到 req.temp_*。
              只有当 last_prefill_chunk=True（最后一块，prefill 结束）时，才调用上面的
              _process_* 系列把暂存数据整理成最终结果并做长度校验。
        参数：
              i —— 该请求在 batch 中的下标；
              logprob_pt —— 在已处理 prefill ids 中的偏移指针；
              num_input_logprobs —— 本 chunk 内应取的 input logprob 数量。
        """
        assert output.input_token_logprobs is not None
        # 中译：以下若干 if 是「惰性初始化」——首次进入时把累积/暂存容器建为空 list。
        if req.input_token_logprobs is None:
            req.input_token_logprobs = []
        if req.temp_input_top_logprobs_val is None:
            req.temp_input_top_logprobs_val = []
        if req.temp_input_top_logprobs_idx is None:
            req.temp_input_top_logprobs_idx = []
        if req.temp_input_token_ids_logprobs_val is None:
            req.temp_input_token_ids_logprobs_val = []
        if req.temp_input_token_ids_logprobs_idx is None:
            req.temp_input_token_ids_logprobs_idx = []

        if req.logprob.input_token_logprobs_val is not None:
            # The input logprob has been already computed. It only happens
            # upon retract.
            # 中译：input logprob 已算过则直接返回。这只会发生在「请求被回撤（retract）后重跑」时。
            if req.logprob.top_logprobs_num > 0:
                assert req.logprob.input_token_logprobs_val is not None
            return

        # Important for the performance.
        # 中译：input_token_logprobs 是 tuple（不可变、访问快），这里断言类型对性能很关键。
        assert isinstance(output.input_token_logprobs, tuple)
        input_token_logprobs: Tuple[int] = output.input_token_logprobs
        # 中译：按 [logprob_pt, logprob_pt+num) 切出本 chunk 对应的片段，再追加累积。
        input_token_logprobs = input_token_logprobs[
            logprob_pt : logprob_pt + num_input_logprobs
        ]
        req.input_token_logprobs.extend(input_token_logprobs)

        if req.logprob.top_logprobs_num > 0:
            req.temp_input_top_logprobs_val.append(output.input_top_logprobs_val[i])
            req.temp_input_top_logprobs_idx.append(output.input_top_logprobs_idx[i])

        if req.logprob.token_ids_logprob is not None:
            req.temp_input_token_ids_logprobs_val.append(
                output.input_token_ids_logprobs_val[i]
            )
            req.temp_input_token_ids_logprobs_idx.append(
                output.input_token_ids_logprobs_idx[i]
            )

        if last_prefill_chunk:
            # 中译：到达最后一块——把累积的 input_token_logprobs 取出，清空累积容器，
            #       然后用 helper 把三类 input logprob 整理为最终结果。
            input_token_logprobs = req.input_token_logprobs
            req.input_token_logprobs = None
            assert req.logprob.input_token_logprobs_val is None
            assert req.logprob.input_token_logprobs_idx is None
            assert req.logprob.input_top_logprobs_val is None
            assert req.logprob.input_top_logprobs_idx is None

            # Process all input logprob types using helper functions
            # 中译：用上面的 helper 依次整理 token / top / token_ids 三类 input logprob。
            self._process_input_token_logprobs(req, input_token_logprobs)
            self._process_input_top_logprobs(req)

            self._process_input_token_ids_logprobs(req)

            # 中译：若请求要求返回 logprob，则校验各数组长度都与期望长度一致（防止对齐错误）。
            if req.return_logprob:
                relevant_tokens_len = self._calculate_relevant_tokens_len(req)
                assert len(req.logprob.input_token_logprobs_val) == relevant_tokens_len
                assert len(req.logprob.input_token_logprobs_idx) == relevant_tokens_len
                if req.logprob.top_logprobs_num > 0:
                    assert (
                        len(req.logprob.input_top_logprobs_val) == relevant_tokens_len
                    )
                    assert (
                        len(req.logprob.input_top_logprobs_idx) == relevant_tokens_len
                    )
                if req.logprob.token_ids_logprob is not None:
                    assert (
                        len(req.logprob.input_token_ids_logprobs_val)
                        == relevant_tokens_len
                    )
                    assert (
                        len(req.logprob.input_token_ids_logprobs_idx)
                        == relevant_tokens_len
                    )

    def add_logprob_return_values(
        self,
        i: int,
        req: Req,
        pt: int,
        next_token_ids: List[int],
        num_input_logprobs: int,
        output: LogitsProcessorOutput,
    ):
        """Attach logprobs to the return values.

        中译：把本步生成（output 部分）的 logprob 挂到 req 上，并在需要时一并处理 input logprob。
              每个解码步对一个请求调用一次：追加被采样 token 的 logprob、（可选）top logprob、
              （可选）指定 token_ids 的 logprob。
        """
        # 中译：追加本步采样 token 的 logprob 值与其 token id。
        if output.next_token_logprobs is not None:
            req.logprob.output_token_logprobs_val.append(output.next_token_logprobs[i])
            req.logprob.output_token_logprobs_idx.append(next_token_ids[i])

        # Only add input logprobs if there are input tokens to process
        # Note: For prefill-only requests with default logprob_start_len, this will be 0,
        # meaning we only compute output logprobs (which is the intended behavior)
        # 中译：仅当有 input token 需处理时才补 input logprob。
        #       注：prefill-only 且 logprob_start_len 为默认值时该值为 0，即只算 output logprob（符合预期）。
        if num_input_logprobs > 0:
            self.add_input_logprob_return_values(
                i,
                req,
                output,
                pt,
                num_input_logprobs,
                last_prefill_chunk=True,
            )
        else:
            # 中译：无 input logprob 可处理时，仍要把相关字段初始化为空 list（下游期望它们是 list）。
            self._initialize_empty_logprob_containers(req)

        # 中译：追加本步 output 的 top logprob（若用户请求了 top_logprobs_num）。
        if req.logprob.top_logprobs_num > 0:
            req.logprob.output_top_logprobs_val.append(
                output.next_token_top_logprobs_val[i]
            )
            req.logprob.output_top_logprobs_idx.append(
                output.next_token_top_logprobs_idx[i]
            )

        # 中译：追加本步对「指定 token id 列表」的 logprob（若用户指定了 token_ids_logprob）。
        if (
            req.logprob.token_ids_logprob is not None
            and output.next_token_token_ids_logprobs_val is not None
        ):
            # Convert GPU tensor to list if needed
            # 中译：值可能是 GPU 张量，需要时转成 list 再存。
            logprobs_val = output.next_token_token_ids_logprobs_val[i]
            if isinstance(logprobs_val, torch.Tensor):
                logprobs_val = logprobs_val.tolist()
            req.logprob.output_token_ids_logprobs_val.append(logprobs_val)
            req.logprob.output_token_ids_logprobs_idx.append(
                output.next_token_token_ids_logprobs_idx[i]
            )

        return num_input_logprobs

    def _initialize_empty_logprob_containers(self, req: Req) -> None:
        """
        Initialize logprob fields to empty lists if unset.

        This is needed for prefill-only requests where the normal initialization
        flow might be bypassed, but downstream code expects these fields to be lists.

        中译：把尚未设置的 input logprob 字段初始化为空 list。
              prefill-only 请求可能跳过常规初始化流程，但下游代码假定这些字段是 list，
              故在此兜底初始化以免出现 None 引发的错误。
        """
        if req.logprob.input_token_logprobs_val is None:
            req.logprob.input_token_logprobs_val = []
        if req.logprob.input_token_logprobs_idx is None:
            req.logprob.input_token_logprobs_idx = []
        if req.logprob.input_top_logprobs_val is None:
            req.logprob.input_top_logprobs_val = []
        if req.logprob.input_top_logprobs_idx is None:
            req.logprob.input_top_logprobs_idx = []
        if req.logprob.input_token_ids_logprobs_val is None:
            req.logprob.input_token_ids_logprobs_val = []
        if req.logprob.input_token_ids_logprobs_idx is None:
            req.logprob.input_token_ids_logprobs_idx = []
