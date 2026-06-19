"""Scoring mixin for TokenizerManager.

中译：本模块为 TokenizerManager 提供「打分（score）」相关能力（以 Mixin 形式混入）。
      所谓打分，是在给定 (query + item) 拼接序列后，计算指定标签 token（label_token_ids）
      出现的概率／分数。典型用途包括 reranker（重排序）、奖励模型（reward model）打分、
      分类模型推理等。

      本模块支持两类模型与两种打分模式：
      - 模型类型：生成式模型（CausalLM，依赖 logprob）与序列分类模型
        （SequenceClassification，直接读取分类头输出的 pooled logits）。
      - 打分模式：单条打分（single-item，每个 query+item 各自独立成一个请求）与
        多条打分（multi-item scoring，简称 MIS；用分隔符把 query 与多个 item 拼成
        单条序列，一次前向取多个分隔符位置的 logprob，提升吞吐，需开启 --enable-mis）。

      实际的前向推理仍委托给底层的 generate_request；本 Mixin 主要负责输入构造
      （分词、拼接 input_ids、解析 embedding 覆盖位置）与输出解析（提取并归一化分数）。
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from sglang.srt.configs.model_config import is_cross_encoding_pooler_model
from sglang.srt.managers.embed_types import PositionalEmbeds
from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.server_args import MIS_DELIMITER_TOKEN_ID

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """中译：一次打分请求的结果容器（不可变 frozen dataclass）。

    字段含义：
    - scores：每个 item 一个分数列表，分数顺序与 label_token_ids 对应。
    - prompt_tokens：本次处理消耗的 prompt token 数（用于计费/统计）。
    - pooled_hidden_states：可选的「池化后隐藏状态」（任务头之前的 transformer 输出），
      仅在 return_pooled_hidden_states=True 且模型支持时填充，否则为 None。
    """

    scores: List[List[float]]
    prompt_tokens: int = 0
    # Per-item pooled hidden states (pre-head transformer output).
    # CPU tensors when return_pooled_hidden_states=True; kept as tensors so
    # in-process consumers (gRPC, engine API) avoid a .tolist() round-trip.
    # The HTTP path converts to lists in serving_score.py before JSON serialization.
    # Same layout as scores: one tensor per item (not a single packed 2D tensor).
    # 中译：每个 item 对应的 pooled hidden states（任务头之前的 transformer 输出）。
    #       当 return_pooled_hidden_states=True 时为 CPU 张量；保持张量形态是为了让进程内
    #       消费者（gRPC、engine API）省去一次 .tolist() 的来回转换开销。HTTP 路径会在
    #       serving_score.py 里在 JSON 序列化前再转为列表。布局与 scores 一致：每个 item
    #       一个张量，而非打包成单个 2D 张量。
    pooled_hidden_states: Optional[List[Optional[torch.Tensor]]] = None


class TokenizerManagerScoreMixin:
    """中译：TokenizerManager 的「打分」能力 Mixin。

    通过混入（mixin）方式给 TokenizerManager 提供 score_prompts / score_request 等接口，
    本身不持有独立状态，而是复用宿主类的属性与方法（如 self.tokenizer、self.is_generation、
    self.server_args、self.model_config、self.generate_request）。

    职责分工：
    - 对外入口：score_prompts（整段 prompt 打分）与 score_request（query+item 打分）。
    - 输入构造：分词、拼接 input_ids、构建多条打分序列、解析 embedding 覆盖位置。
    - 输出解析：从调度器返回的 logprob 或 embedding 中提取并归一化分数。
    """

    async def score_prompts(
        self,
        prompts: Union[str, List[str], List[List[int]]],
        label_token_ids: List[int],
        apply_softmax: bool = False,
        request: Optional[Any] = None,
    ) -> ScoreResult:
        """
        Score probabilities of specified token IDs after each *full prompt*.

        This is a thin wrapper over `score_request` that treats `prompts` as
        already-composed inputs (i.e., no query/item concatenation needed).

        Args:
            prompts: A single prompt string, a list of prompt strings, or a list of
                pre-tokenized prompt token ID sequences.
            label_token_ids: Token IDs to compute probabilities for.
            apply_softmax: Whether to normalize probabilities using softmax.
            request: Optional FastAPI request object.

        Returns:
            ScoreResult with:
                scores: List of score lists, one for each prompt, each in the order of label_token_ids.
                prompt_tokens: The number of prompt tokens processed.

        中译：对每个「完整 prompt」之后，计算指定 token id（label_token_ids）的出现概率。

        本方法是 score_request 的轻量包装：把 prompts 当作已经组装好的完整输入
        （即不需要再做 query/item 拼接），因此固定传入空 query。

        参数：
            prompts：单个 prompt 字符串、prompt 字符串列表，或已分词的 token id 序列列表。
            label_token_ids：需要计算概率的目标 token id 列表。
            apply_softmax：是否对结果做 softmax 归一化。
            request：可选的 FastAPI 请求对象。

        返回：
            ScoreResult，其 scores 为每个 prompt 一个分数列表（顺序与 label_token_ids 一致），
            prompt_tokens 为处理的 prompt token 数。
        """
        # Text prompts
        # 中译：文本类 prompt（字符串、字符串列表、或空列表）。以空 query 走单条打分。
        if isinstance(prompts, str) or (
            isinstance(prompts, list) and (not prompts or isinstance(prompts[0], str))
        ):
            return await self.score_request(
                query="",
                items=prompts,  # type: ignore[arg-type]
                label_token_ids=label_token_ids,
                apply_softmax=apply_softmax,
                item_first=False,
                request=request,
            )

        # Tokenized prompts
        # 中译：已分词的 token id 序列列表。以空 token 列表作为 query。
        if isinstance(prompts, list) and (not prompts or isinstance(prompts[0], list)):
            return await self.score_request(
                query=[],
                items=prompts,
                label_token_ids=label_token_ids,
                apply_softmax=apply_softmax,
                item_first=False,
                request=request,
            )

        raise ValueError("Invalid prompts type for score_prompts.")

    def _build_multi_item_token_sequence(
        self, query: List[int], items: List[List[int]], delimiter_token_id: int
    ) -> Tuple[List[int], List[int]]:
        """
        Build a single token sequence for multi-item scoring.
        Format: query<delimiter>item1<delimiter>item2<delimiter>item3<delimiter>

        Args:
            query: Query token IDs
            items: List of item token ID sequences
            delimiter_token_id: Token ID to use as delimiter

        Returns:
            Tuple of (combined token sequence, delimiter indices)

        中译：为「多条打分（multi-item scoring）」构建单条 token 序列。
        格式：query<分隔符>item1<分隔符>item2<分隔符>item3<分隔符>

        参数：
            query：query 的 token id 序列。
            items：各 item 的 token id 序列列表。
            delimiter_token_id：用作分隔符的 token id。

        返回：
            (combined token sequence, delimiter indices) —— 拼接后的整条序列，
            以及每个分隔符在序列中的下标（后续在这些位置提取 logprob）。
        """
        combined_sequence = query[:]  # Start with query  # 中译：以 query 的拷贝起头
        delimiter_indices = []

        for item in items:
            # 中译：先记录分隔符位置，再追加分隔符与该 item 的 token。
            delimiter_indices.append(len(combined_sequence))
            combined_sequence.append(delimiter_token_id)  # Add delimiter
            combined_sequence.extend(item)  # Add item tokens

        # Add final delimiter after the last item for logprob extraction
        # 中译：在最后一个 item 之后再补一个分隔符，作为最后一个 item 的 logprob 提取点。
        delimiter_indices.append(len(combined_sequence))
        combined_sequence.append(delimiter_token_id)

        return combined_sequence, delimiter_indices

    def _batch_tokenize_query_and_items(
        self,
        query: Optional[Union[str, List[int]]],
        items: Optional[Union[str, List[str], List[List[int]]]],
    ) -> Tuple[List[int], List[List[int]]]:
        """
        Tokenize query and items into token IDs.

        Args:
            query: The query text (str) or pre-tokenized token IDs (List[int]).
            items: Item texts or pre-tokenized token IDs.

        Returns:
            (query_ids, items_ids): query token IDs and list of per-item token IDs.

        中译：把 query 与 items 分词成 token id。

        参数：
            query：query 文本（str）或已分词的 token id（List[int]）。
            items：item 文本或已分词的 token id。

        返回：
            (query_ids, items_ids)：query 的 token id，以及每个 item 的 token id 列表。
            已经是 token id 的输入直接拷贝，文本输入则调用 self.tokenizer.encode 分词。
        """
        if isinstance(query, str):
            query_ids = self.tokenizer.encode(query)
        else:
            query_ids = list(query)

        items_list = [items] if isinstance(items, str) else items

        items_ids = []
        for item in items_list:
            if isinstance(item, str):
                items_ids.append(self.tokenizer.encode(item))
            else:
                items_ids.append(list(item))

        return query_ids, items_ids

    def _process_multi_item_scoring_results(
        self,
        results: Any,
        items: List,
        label_token_ids: Optional[List[int]],
        apply_softmax: bool,
        batch_request=None,
        return_pooled_hidden_states: bool = False,
    ) -> ScoreResult:
        """
        Process results from multi-item scoring request.

        Extracts per-delimiter scores from whichever field the scheduler
        populated (input_token_ids_logprobs for generation models,
        embedding for classification models), then uniformly validates,
        skips the query-boundary delimiter, and normalizes.

        Args:
            results: Results from generate_request
            items: List of items being scored
            label_token_ids: Token IDs to extract scores for
            apply_softmax: Whether to apply softmax normalization
            batch_request: The original batch request containing input sequence
            return_pooled_hidden_states: Whether to extract pooled hidden states
                from the result and include them in the ScoreResult.

        Returns:
            ScoreResult with per-item scores, prompt token count, and optional
            pooled_hidden_states (when return_pooled_hidden_states=True and the
            model populated the field).

        中译：解析「多条打分（multi-item scoring）」请求的返回结果。

        从调度器实际填充的字段里提取每个分隔符位置的分数（生成式模型用
        input_token_ids_logprobs，分类模型用 embedding），然后统一做校验、跳过
        query 边界的那个分隔符、并按需归一化。

        参数：
            results：generate_request 的返回结果。
            items：被打分的 item 列表。
            label_token_ids：需要提取分数的 token id。
            apply_softmax：是否做 softmax 归一化。
            batch_request：原始批请求（含输入序列）。
            return_pooled_hidden_states：是否一并提取 pooled hidden states。

        返回：
            ScoreResult，包含每个 item 的分数、prompt token 数，以及可选的
            pooled_hidden_states（仅当开启且模型填充了该字段时）。
        """
        # 中译：多条打分时输入只有一条拼接序列，故取第一个（或唯一）结果即可。
        single_result = results[0] if isinstance(results, list) else results
        meta_info = single_result.get("meta_info", {})
        num_items = len(items) if isinstance(items, list) else 1
        # 中译：期望的分隔符数量 = item 数 + 1（query 与首个 item 之间还有一个边界分隔符）。
        expected_count = num_items + 1
        request_id = meta_info.get("id", "<unknown>")
        prompt_tokens = meta_info.get("prompt_tokens", 0)

        # Extract per-delimiter scores from whichever field has them
        # 中译：根据模型类型，从对应字段提取各分隔符位置的分数。
        input_logprobs = meta_info.get("input_token_ids_logprobs", [])
        embedding = single_result.get("embedding")

        if input_logprobs:
            # Generation model: extract label-token logprobs at each delimiter
            # 中译：生成式模型——在每个分隔符位置提取目标 token 的 logprob 并转为分数。
            per_delimiter_scores = []
            for logprobs_data in input_logprobs:
                logprobs = self._extract_logprobs_for_tokens(
                    logprobs_data, label_token_ids
                )
                score_list = self._convert_logprobs_to_scores(
                    logprobs, label_token_ids, apply_softmax
                )
                per_delimiter_scores.append(score_list)
        elif embedding is not None:
            # Classification model: scores are directly in 2D embedding.
            # 中译：分类模型——分数直接以 2D embedding 形式给出，每行对应一个分隔符位置。
            if apply_softmax:
                scores_tensor = (
                    torch.tensor(embedding)
                    if isinstance(embedding, list)
                    else embedding
                )
                scores_tensor = torch.nn.functional.softmax(scores_tensor, dim=-1)
                per_delimiter_scores = scores_tensor.tolist()
            else:
                per_delimiter_scores = (
                    embedding if isinstance(embedding, list) else embedding.tolist()
                )
        else:
            raise RuntimeError(
                f"No scoring data found for multi-item scoring request {request_id}. "
                "Expected either input_token_ids_logprobs or embedding."
            )

        # Validate delimiter count
        # 中译：校验分隔符数量是否与预期一致，不一致说明拼接或解析出错。
        if len(per_delimiter_scores) != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} delimiter entries for multi-item scoring "
                f"with {num_items} items, but got {len(per_delimiter_scores)}. "
                f"Request ID: {request_id}"
            )

        # Skip the first delimiter (query-item boundary)
        # 中译：丢弃第一个分隔符（query 与首个 item 的边界），其余才是各 item 的分数。
        scores = per_delimiter_scores[1:]

        phs_list = None
        if return_pooled_hidden_states:
            raw_phs = single_result.get("pooled_hidden_state")
            if raw_phs is not None and len(raw_phs) == expected_count:
                phs_list = raw_phs[1:]

        return ScoreResult(
            scores=scores,
            prompt_tokens=prompt_tokens,
            pooled_hidden_states=phs_list,
        )

    def _process_single_item_scoring_results(
        self,
        results: Any,
        label_token_ids: Optional[List[int]],
        apply_softmax: bool,
        return_pooled_hidden_states: bool = False,
    ) -> ScoreResult:
        """
        Process results from single-item scoring request.

        For generation (CausalLM) models: reads output_token_ids_logprobs.
        For non-generation (SequenceClassification) models: reads the embedding field
        which contains pooled class logits from the classification head.

        Args:
            results: Results from generate_request
            label_token_ids: Token IDs to extract scores for (generation models only)
            apply_softmax: Whether to apply softmax normalization
            return_pooled_hidden_states: Whether to extract pooled hidden states

        Returns:
            ScoreResult with per-item scores, prompt token count, and optional pooled_hidden_states.

        中译：解析「单条打分（single-item scoring）」请求的返回结果。

        - 生成式模型（CausalLM）：读取 output_token_ids_logprobs。
        - 非生成式模型（SequenceClassification）：读取 embedding 字段，其中是分类头输出的
          pooled class logits。

        参数：
            results：generate_request 的返回结果（每个 item 一条）。
            label_token_ids：需要提取分数的 token id（仅生成式模型用到）。
            apply_softmax：是否做 softmax 归一化。
            return_pooled_hidden_states：是否提取 pooled hidden states。

        返回：
            ScoreResult，包含每个 item 的分数、prompt token 数，以及可选的 pooled_hidden_states。
        """
        scores = []
        phs_list = []
        has_phs = False
        prompt_tokens = 0

        is_generation = self.is_generation
        if is_generation:
            for result in results:
                # For single-item scoring, logprobs are in output_token_ids_logprobs
                # 中译：单条打分时，目标 token 的 logprob 位于 output_token_ids_logprobs。
                output_logprobs = result["meta_info"].get(
                    "output_token_ids_logprobs", []
                )
                prompt_tokens += result["meta_info"].get("prompt_tokens", 0)

                if not output_logprobs or len(output_logprobs) == 0:
                    raise RuntimeError(
                        f"output_logprobs is empty for request "
                        f"{result['meta_info'].get('id', '<unknown>')}."
                    )

                # Extract logprobs for the first (and only) position
                # 中译：单条打分只关心第一个（也是唯一一个）位置的 logprob。
                logprobs = self._extract_logprobs_for_tokens(
                    output_logprobs[0], label_token_ids
                )
                score_list = self._convert_logprobs_to_scores(
                    logprobs, label_token_ids, apply_softmax
                )
                scores.append(score_list)
        else:
            for result in results:
                embedding = result.get("embedding", None)
                if embedding is None:
                    raise ValueError("Embedding not found in the result.")

                prompt_tokens += result.get("meta_info", {}).get("prompt_tokens", 0)

                if apply_softmax:
                    embedding = torch.softmax(
                        torch.as_tensor(embedding), dim=-1
                    ).tolist()

                # The classification head produces per-token logits, which the pooler reduces
                # into a single vector per input. That vector is returned in the `.embeddings`
                # field — not as semantic embeddings, but as pooled classification logits.
                # The field name is reused for compatibility with the existing
                # EmbeddingPoolerOutput API.
                # 中译：分类头产生逐 token 的 logits，pooler 把它们规约成每个输入一个向量。
                #       该向量复用 `.embeddings` 字段返回——它并非语义 embedding，而是池化后的
                #       分类 logits；之所以借用此字段名，是为了兼容既有的 EmbeddingPoolerOutput API。
                scores.append(embedding)

                if return_pooled_hidden_states:
                    phs = result.get("pooled_hidden_state")
                    phs_list.append(phs)
                    if phs is not None:
                        has_phs = True

        return ScoreResult(
            scores=scores,
            prompt_tokens=prompt_tokens,
            pooled_hidden_states=phs_list if has_phs else None,
        )

    # ------------------------------------------------------------------
    # Embed override position resolution
    # ------------------------------------------------------------------

    def _resolve_overrides_for_sequence(
        self,
        token_ids: List[int],
        embeds: Optional[List[torch.Tensor]],
        embed_override_token_id: int,
        position_offset: int = 0,
        label: str = "input",
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """Scan token_ids for placeholder occurrences and pair with embeddings.

        Args:
            token_ids: The token sequence to scan.
            embeds: Embedding tensors to place at placeholder positions (None = skip).
            embed_override_token_id: The placeholder token ID.
            position_offset: Added to each found position (for absolute coordinates).
            label: Label for error messages (e.g. "query", "items[2]").

        Returns:
            (embeds, positions) lists. Empty lists if embeds is None.

        中译：扫描 token_ids 中占位 token（embed_override_token_id）出现的位置，
        并与传入的覆盖 embedding 一一配对。

        所谓 embedding 覆盖（embed override）：把序列中某些占位 token 的嵌入向量替换为
        外部直接提供的向量（如多模态特征），占位 token 仅用于标记位置。

        参数：
            token_ids：待扫描的 token 序列。
            embeds：放到占位位置的 embedding 张量（None 表示跳过）。
            embed_override_token_id：占位 token id。
            position_offset：加到每个找到的位置上（用于换算成绝对坐标）。
            label：报错信息中的标签（如 "query"、"items[2]"）。

        返回：
            (embeds, positions) 两个列表；若 embeds 为 None 则返回空列表。
            占位 token 数量与提供的 embedding 数量必须一致，否则报错。
        """
        if embeds is None:
            return [], []
        positions = [
            idx + position_offset
            for idx, tok in enumerate(token_ids)
            if tok == embed_override_token_id
        ]
        if len(positions) != len(embeds):
            raise ValueError(
                f"{label} contains {len(positions)} occurrences of "
                f"embed_override_token_id={embed_override_token_id}, "
                f"but {len(embeds)} override embeddings were provided."
            )
        return embeds, positions

    def _resolve_embed_overrides_for_request(
        self,
        query: List[int],
        item: List[int],
        embed_override_token_id: int,
        query_embed_overrides: Optional[List[torch.Tensor]],
        item_embeds: Optional[List[torch.Tensor]],
        item_position_offset: int,
        item_label: str,
    ) -> Optional[PositionalEmbeds]:
        """Resolve embed overrides for a single query+item pair.

        Returns PositionalEmbeds if any overrides exist, None otherwise.

        中译：为单个 query+item 对解析 embedding 覆盖。

        分别在 query 与 item 上解析占位位置（item 需加 item_position_offset 偏移到拼接后的
        绝对坐标），合并后若存在覆盖则返回 PositionalEmbeds，否则返回 None。
        """
        q_embeds, q_positions = self._resolve_overrides_for_sequence(
            query,
            query_embed_overrides,
            embed_override_token_id,
            position_offset=0,
            label="query",
        )
        i_embeds, i_positions = self._resolve_overrides_for_sequence(
            item,
            item_embeds,
            embed_override_token_id,
            position_offset=item_position_offset,
            label=item_label,
        )
        all_embeds = q_embeds + i_embeds
        all_positions = q_positions + i_positions
        if not all_embeds:
            return None
        return PositionalEmbeds(embeds=all_embeds, positions=all_positions)

    # ------------------------------------------------------------------
    # Input preparation (tokenization + input_ids construction)
    # ------------------------------------------------------------------

    def _build_token_id_inputs(
        self,
        query: List[int],
        items: List[List[int]],
        item_first: bool,
        use_multi_item_scoring: bool,
        embed_override_token_id: Optional[int],
        query_embed_overrides: Optional[List[torch.Tensor]],
        item_embed_overrides: Optional[List[Optional[List[torch.Tensor]]]],
    ) -> Tuple[None, List[List[int]], Optional[list], Optional[List[int]]]:
        """Build input_ids and resolve embed overrides for token-ID inputs.

        Works identically for multi-item-scoring and single-item modes — the only difference is
        how input_ids are assembled and what position offset each item gets.

        Returns:
            (text_prompts, input_ids, positional_embed_overrides, delimiter_indices)

        中译：为「token id 输入」构建 input_ids，并解析 embedding 覆盖。

        多条打分模式与单条模式逻辑相通——唯一区别在于 input_ids 如何拼装、以及每个 item
        获得的位置偏移不同。返回的第一个元素恒为 None（token id 输入没有文本 prompt）。

        返回：
            (text_prompts, input_ids, positional_embed_overrides, delimiter_indices)
        """
        # Both query and items are token IDs
        has_embeds = (
            query_embed_overrides is not None or item_embed_overrides is not None
        )

        # Query placeholder positions are invariant across items — resolve once.
        # (No-op returning ([], []) if has_embeds is False or query_embed_overrides is None.)
        # 中译：query 中的占位位置不随 item 变化，故只解析一次复用。
        #       （若无 embedding 覆盖则返回空列表，相当于空操作。）
        q_embeds, q_positions = self._resolve_overrides_for_sequence(
            query,
            query_embed_overrides,
            embed_override_token_id,
            position_offset=0,
            label="query",
        )

        if use_multi_item_scoring:
            # Multi-item scoring: concatenate with placeholder delimiter token.
            # Positions are derived from item lengths (delimiter_indices), not
            # by scanning for this token — it exists only for FlashInfer compat.
            # 中译：多条打分——用占位分隔符 token 把 query 与各 item 拼成单条序列。
            #       分隔符位置由各 item 长度推算（delimiter_indices），并非靠扫描该 token 得到；
            #       该 token 仅为兼容 FlashInfer 而存在。
            delimiter_token_id = MIS_DELIMITER_TOKEN_ID
            combined_input_ids, delimiter_indices = (
                self._build_multi_item_token_sequence(query, items, delimiter_token_id)
            )
            input_ids = [combined_input_ids]

            if not has_embeds:
                return None, input_ids, None, delimiter_indices

            # Resolve embed overrides across the combined multi-item-scoring sequence.
            # 中译：在拼接后的整条序列上解析 embedding 覆盖，逐 item 累加绝对位置偏移。
            all_embeds: List[torch.Tensor] = list(q_embeds)
            all_positions: List[int] = list(q_positions)
            current_offset = len(query) + 1  # +1 for first delimiter  # 中译：+1 为首个分隔符
            for i, item in enumerate(items):
                item_embs = item_embed_overrides[i] if item_embed_overrides else None
                i_embeds, i_positions = self._resolve_overrides_for_sequence(
                    item,
                    item_embs,
                    embed_override_token_id,
                    position_offset=current_offset,
                    label=f"items[{i}]",
                )
                all_embeds.extend(i_embeds)
                all_positions.extend(i_positions)
                current_offset += len(item) + 1  # +1 for delimiter  # 中译：+1 为该 item 后的分隔符

            if all_embeds:
                # PositionalEmbeds.__post_init__ does the single torch.cat stack.
                # 中译：PositionalEmbeds 的 __post_init__ 会把这些张量一次性 torch.cat 堆叠。
                positional_embed_overrides = [
                    PositionalEmbeds(embeds=all_embeds, positions=all_positions)
                ]
            else:
                positional_embed_overrides = None
            return None, input_ids, positional_embed_overrides, delimiter_indices

        else:
            # Single-item scoring: process each item separately
            # 中译：单条打分——每个 item 单独成一条 input_ids。item_first 决定 item 在前还是 query 在前。
            if item_first:
                input_ids = [item + query for item in items]
            else:
                input_ids = [query + item for item in items]

            if not has_embeds:
                return None, input_ids, None, None

            positional_embed_overrides = []
            any_overrides = False
            for i, item in enumerate(items):
                item_embs = item_embed_overrides[i] if item_embed_overrides else None
                i_embeds, i_positions = self._resolve_overrides_for_sequence(
                    item,
                    item_embs,
                    embed_override_token_id,
                    position_offset=len(query),
                    label=f"items[{i}]",
                )
                combined_embeds = q_embeds + i_embeds
                if combined_embeds:
                    positional_embed_overrides.append(
                        PositionalEmbeds(
                            embeds=combined_embeds,
                            positions=q_positions + i_positions,
                        )
                    )
                    any_overrides = True
                else:
                    positional_embed_overrides.append(None)

            return (
                None,
                input_ids,
                positional_embed_overrides if any_overrides else None,
                None,
            )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def score_request(
        self,
        query: Optional[Union[str, List[int]]] = None,
        items: Optional[Union[str, List[str], List[List[int]]]] = None,
        label_token_ids: Optional[List[int]] = None,
        apply_softmax: bool = False,
        item_first: bool = False,
        embed_override_token_id: Optional[int] = None,
        query_embed_overrides: Optional[List[torch.Tensor]] = None,
        item_embed_overrides: Optional[List[Optional[List[torch.Tensor]]]] = None,
        request: Optional[Any] = None,
        return_pooled_hidden_states: bool = False,
    ) -> ScoreResult:
        """
        Score the probability of specified token IDs appearing after the given (query + item) pair.

        This method supports two scoring approaches:
        1. Single-Item scoring (default): Process each query+item pair independently
        2. Multi-Item scoring: When --enable-mis is set, combine query and
           multiple items into a single sequence using delimiter for efficient processing.
           Note: item_first parameter is ignored in multi-item scoring mode since it uses
           a fixed format: query<delimiter>item1<delimiter>item2<delimiter>item3<delimiter>

           Multi-item scoring works with both text and pre-tokenized inputs:
           - Text: query<delimiter_text>item1<delimiter_text>item2<delimiter_text>item3<delimiter_text>
           - Tokens: query<delimiter_token_id>item1<delimiter_token_id>item2<delimiter_token_id>item3<delimiter_token_id>

        Supports two model types:
        - Generation (CausalLM): Requires label_token_ids; returns logprob-based scores.
        - SequenceClassification: label_token_ids is optional; returns pooled class logits.

        Args:
            query: The query text or pre-tokenized query token IDs
            items: The item text(s) or pre-tokenized item token IDs
            label_token_ids: List of token IDs to compute probabilities for
            apply_softmax: Whether to normalize probabilities using softmax
            item_first: If True, prepend items to query. Ignored for multi-item scoring.
            embed_override_token_id: Placeholder token ID for embedding override positions.
            query_embed_overrides: Embedding vectors replacing placeholder tokens in query.
            item_embed_overrides: Per-item embedding vectors replacing placeholder tokens in items.
            request: Optional FastAPI request object
            return_pooled_hidden_states: Whether to include the raw pooled transformer
                hidden states (before the task-specific head) in the result. Only
                supported for non-generation models (SequenceClassification,
                RewardModel). Raises ValueError for CausalLM models.

        Returns:
            ScoreResult with:
                scores: List of score lists, one per item.
                prompt_tokens: The number of prompt tokens processed.
                pooled_hidden_states: Per-item CPU tensors when
                    return_pooled_hidden_states=True and the model supports it;
                    None otherwise.

        中译：打分主入口——计算指定 token id 出现在 (query + item) 拼接序列之后的概率/分数。

        支持两种打分方式：
        1. 单条打分（默认）：每个 query+item 对各自独立处理。
        2. 多条打分（设置 --enable-mis 时）：用分隔符把 query 与多个 item 合成单条序列，
           一次前向高效处理。注意此模式下忽略 item_first，固定采用
           query<分隔符>item1<分隔符>item2<分隔符>item3<分隔符> 的格式；文本与已分词输入皆可。

        支持两类模型：
        - 生成式（CausalLM）：必须提供 label_token_ids，返回基于 logprob 的分数。
        - 序列分类（SequenceClassification）：label_token_ids 可选，返回池化后的分类 logits。

        参数：见上方英文；其中 embed_override_token_id / query_embed_overrides /
        item_embed_overrides 用于 embedding 覆盖；return_pooled_hidden_states 仅非生成式模型
        支持（对 CausalLM 会抛 ValueError）。

        副作用：最终通过 self.generate_request 提交一次实际推理请求。
        """
        is_generation = self.is_generation

        # 中译：以下为一系列输入合法性校验（模型类型、items 是否为空、embedding 覆盖参数搭配、
        #       label_token_ids 是否越界词表等），任一不满足即抛错或提前返回空结果。
        if is_generation and label_token_ids is None:
            raise ValueError(
                "label_token_ids is required for generation (CausalLM) models."
            )
        if items is None:
            raise ValueError("items must be provided")
        if not items:
            return ScoreResult(scores=[], prompt_tokens=0)

        has_embeds = (
            query_embed_overrides is not None or item_embed_overrides is not None
        )
        if has_embeds and embed_override_token_id is None:
            raise ValueError(
                "embed_override_token_id is required when query_embed_overrides "
                "or item_embed_overrides are supplied."
            )
        if item_first and has_embeds:
            raise ValueError("item_first is not supported when embeddings are supplied")
        if item_embed_overrides is not None and len(item_embed_overrides) != len(items):
            raise ValueError(
                f"item_embed_overrides length ({len(item_embed_overrides)}) "
                f"must match items length ({len(items)})."
            )
        if self.tokenizer is not None and label_token_ids is not None:
            vocab_size = self.tokenizer.vocab_size
            for token_id in label_token_ids:
                if token_id >= vocab_size:
                    raise ValueError(
                        f"Token ID {token_id} is out of vocabulary (vocab size: {vocab_size})"
                    )

        # Check if multi-item scoring is enabled
        # 中译：是否启用多条打分（由服务端 --enable-mis 决定）。
        use_multi_item_scoring = self.server_args.enable_mis

        input_ids = None
        text_prompts = None
        positional_embed_overrides = None
        delimiter_indices = None

        # 中译：纯文本输入（query 为 str 且无 embedding 覆盖）走文本路径，可直接交给分词器；
        #       其余情况需先分词成 token id 以便定位拼接位置/覆盖位置。
        use_text_prompts = isinstance(query, str) and not has_embeds

        if use_text_prompts:
            # Both query and items are text
            items_list = [items] if isinstance(items, str) else items
            if use_multi_item_scoring:
                # Tokenize separately, then combine at token level with placeholder
                # delimiter. Positions come from item lengths (delimiter_indices),
                # not from scanning for this token — it's for FlashInfer compat only.
                delimiter_token_id = MIS_DELIMITER_TOKEN_ID
                query_ids, items_ids = self._batch_tokenize_query_and_items(
                    query, items_list
                )
                combined_input_ids, delimiter_indices = (
                    self._build_multi_item_token_sequence(
                        query_ids, items_ids, delimiter_token_id
                    )
                )
                input_ids = [combined_input_ids]
            else:
                # Single-item scoring: create separate prompts for each item
                if item_first:
                    text_prompts = [f"{item}{query}" for item in items_list]
                else:
                    text_prompts = [f"{query}{item}" for item in items_list]

        elif (
            isinstance(query, list)
            and isinstance(items, list)
            and items
            and isinstance(items[0], list)
        ):
            # Both query and items are token IDs — tokenize text inputs if needed for embed overrides
            query_ids, items_ids = query, items
            _, input_ids, positional_embed_overrides, delimiter_indices = (
                self._build_token_id_inputs(
                    query_ids,
                    items_ids,
                    item_first,
                    use_multi_item_scoring,
                    embed_override_token_id,
                    query_embed_overrides,
                    item_embed_overrides,
                )
            )
        elif has_embeds:
            # Text inputs with embed overrides — need to tokenize first to resolve positions
            # 中译：带 embedding 覆盖的文本输入——必须先分词，才能定位占位 token 的位置。
            query_ids, items_ids = self._batch_tokenize_query_and_items(query, items)
            _, input_ids, positional_embed_overrides, delimiter_indices = (
                self._build_token_id_inputs(
                    query_ids,
                    items_ids,
                    item_first,
                    use_multi_item_scoring,
                    embed_override_token_id,
                    query_embed_overrides,
                    item_embed_overrides,
                )
            )
        else:
            raise ValueError(
                "Invalid combination of query/items types for score_request."
            )

        if return_pooled_hidden_states:
            if is_generation:
                raise ValueError(
                    "return_pooled_hidden_states is not supported for CausalLM models. "
                    "It requires a model with a task-specific head "
                    "(e.g. SequenceClassification or RewardModel)."
                )
            model_config = self.model_config
            if model_config is not None:
                archs = getattr(model_config.hf_config, "architectures", []) or []
                if is_cross_encoding_pooler_model(archs):
                    raise ValueError(
                        f"return_pooled_hidden_states is not supported for "
                        f"{archs[0]}. This model uses CrossEncodingPooler which "
                        f"does not expose pre-head hidden states."
                    )

        # Create the appropriate request type
        # 中译：根据模型类型构造对应的请求：生成式用 GenerateReqInput（要 logprob），
        #       分类式用 EmbeddingReqInput；多条打分时附带分隔符下标。
        mis_delimiter_indices = [delimiter_indices] if use_multi_item_scoring else None
        if is_generation:
            batch_request = GenerateReqInput(
                text=text_prompts,
                input_ids=input_ids,
                token_ids_logprob=label_token_ids,
                return_logprob=True,
                # Set logprob_start_len=0 for multi-item scoring since we want logprobs at all delimiter positions
                logprob_start_len=0 if use_multi_item_scoring else -1,
                stream=False,
                sampling_params={"max_new_tokens": 0},
                positional_embed_overrides=positional_embed_overrides,
                multi_item_delimiter_indices=mis_delimiter_indices,
            )
        else:
            batch_request = EmbeddingReqInput(
                text=text_prompts,
                input_ids=input_ids,
                positional_embed_overrides=positional_embed_overrides,
                return_pooled_hidden_states=return_pooled_hidden_states,
                multi_item_delimiter_indices=mis_delimiter_indices,
            )

        # 中译：提交推理。generate_request 为异步生成器，打分非流式，取首个产出即为完整结果。
        results = await self.generate_request(batch_request, request).__anext__()

        if use_multi_item_scoring:
            # Multi-item scoring: extract scores from input_token_ids_logprobs or embedding
            return self._process_multi_item_scoring_results(
                results,
                items,
                label_token_ids,
                apply_softmax,
                batch_request,
                return_pooled_hidden_states,
            )
        else:
            # Single-item scoring: process each result separately
            return self._process_single_item_scoring_results(
                results, label_token_ids, apply_softmax, return_pooled_hidden_states
            )

    def _convert_logprobs_to_scores(
        self,
        logprobs: Dict[int, float],
        label_token_ids: List[int],
        apply_softmax: bool,
    ) -> List[float]:
        """
        Convert logprobs dictionary to ordered score list.

        Args:
            logprobs: Dictionary mapping token_id to logprob
            label_token_ids: Token IDs in desired order
            apply_softmax: Whether to apply softmax normalization

        Returns:
            List of scores in the same order as label_token_ids

        中译：把 logprob 字典转换成按 label_token_ids 顺序排列的分数列表。

        参数：
            logprobs：token_id -> logprob 的映射。
            label_token_ids：期望的输出顺序。
            apply_softmax：是否做 softmax 归一化。

        返回：
            与 label_token_ids 顺序一致的分数列表。缺失的 token 取 -inf。
        """
        # 中译：按顺序取分数，缺失的 token 用 -inf 占位。
        score_list = [
            logprobs.get(token_id, float("-inf")) for token_id in label_token_ids
        ]

        if apply_softmax:
            score_list = torch.softmax(torch.tensor(score_list), dim=0).tolist()
        else:
            # Convert logprobs to probabilities if not using softmax
            # 中译：不做 softmax 时，直接对 logprob 取指数还原为概率（-inf 归零）。
            score_list = [
                math.exp(x) if x != float("-inf") else 0.0 for x in score_list
            ]

        return score_list

    def _extract_logprobs_for_tokens(
        self, logprobs_data: List, label_token_ids: List[int]
    ) -> Dict[int, float]:
        """
        Extract logprobs for specified token IDs from logprobs data.

        Args:
            logprobs_data: List of (logprob, token_id, text) tuples
            label_token_ids: Token IDs to extract logprobs for

        Returns:
            Dictionary mapping token_id to logprob

        中译：从 logprob 数据中提取指定 token id 的 logprob。

        参数：
            logprobs_data：(logprob, token_id, text) 三元组列表。
            label_token_ids：需要提取的目标 token id。

        返回：
            token_id -> logprob 的字典（仅包含命中 label_token_ids 的项）。
        """
        logprobs = {}
        if logprobs_data:
            for logprob, token_id, _ in logprobs_data:
                if token_id in label_token_ids:
                    logprobs[token_id] = logprob
        return logprobs
