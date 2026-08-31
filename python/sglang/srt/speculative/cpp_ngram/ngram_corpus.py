# -*- coding: utf-8 -*-
"""N-gram 语料库（后缀 trie）的 Python 封装。

这是 NGRAM 投机解码的「草稿来源」。它做两件事：
  1. **put**：把已经生成/输入过的 token 序列插入 trie，累积统计信息；
  2. **get**：拿当前请求的末尾若干 token 去 trie 里做后缀匹配，
     BFS 展开出一棵候选树，作为下一轮的草稿 token。

真正的 trie 是 C++ 实现（经 `sglang.jit_kernel.ngram_corpus` JIT 编译加载），
本类只负责参数透传、req_id ↔ 内部 state_id 的映射，以及外部语料的记账。

为什么需要 state_id：C++ 侧为每个请求维护「上一次匹配到 trie 哪个节点」的
增量状态，避免每轮都从头重新匹配整条历史。Python 侧用字符串 req_id，
C++ 侧用紧凑的整数 state_id，本类负责两者的双向维护。
"""

import logging
from collections.abc import Iterable, Sequence
from typing import Dict, List, Tuple

import numpy as np

from sglang.jit_kernel.ngram_corpus import get_ngram_corpus_cls

logger = logging.getLogger(__name__)


class NgramCorpus:
    """N-gram 后缀 trie 的封装。

    构造参数（默认值对应 server_args 中的 speculative_ngram_* 系列）：
        max_trie_depth: trie 存储/匹配的最大后缀长度，默认 18。
            含义：一条历史 [a,b,c,...] 插入时，最多记录长度 18 的后缀。
            调大 → 匹配更精确但内存涨；调小 → 更容易命中但候选更泛。
        min_bfs_breadth: BFS 每层最小展开宽度，默认 1。
        max_bfs_breadth: BFS 每层最大展开宽度，默认 8（CLI 默认 10）。
            含义：某个 trie 节点有 20 个不同后继时，最多只取频次最高的 8 个。
        draft_token_num: 一棵草稿树的节点预算，默认 8（CLI 默认 12/16）。
            BFS 展开到攒够这么多节点就停。
        match_type: "BFS" 按新近度展开 / "PROB" 按频次概率展开，默认 "BFS"。
        capacity: trie 最多容纳的条目数，默认 100 万（CLI 默认 1000 万）。
            满了之后按 LRU 淘汰。粗估内存：每条约几十字节，1000 万条约数百 MB CPU 内存。
        external_sam_budget: 给「外部语料后缀自动机」预留的草稿节点数，默认 0（不启用）。
            例如设为 4 且 draft_token_num=16，则 16 个节点里留 4 个给外部语料的匹配结果，
            其余 12 个给运行时累积的语料。
        external_corpus_max_tokens: 外部语料 token 数上限，默认 1000 万，
            超出则启动失败——用来防止外部语料撑爆 CPU 内存。
    """

    def __init__(
        self,
        max_trie_depth=18,
        min_bfs_breadth=1,
        max_bfs_breadth=8,
        draft_token_num=8,
        match_type="BFS",
        capacity=1000000,
        external_sam_budget=0,
        external_corpus_max_tokens=10000000,
    ) -> None:
        # 取到 JIT 编译好的 C++ trie 类（首次调用会触发编译，之后走缓存）
        cls = get_ngram_corpus_cls()
        self._obj = cls(
            capacity=capacity,
            max_trie_depth=max_trie_depth,
            min_bfs_breadth=min_bfs_breadth,
            max_bfs_breadth=max_bfs_breadth,
            draft_token_num=draft_token_num,
            match_type=match_type,
            external_sam_budget=external_sam_budget,
            external_corpus_max_tokens=external_corpus_max_tokens,
        )
        self.draft_token_num = draft_token_num
        self.external_corpus_max_tokens = external_corpus_max_tokens
        # req_id（字符串）→ C++ 侧增量匹配状态槽位（整数），单调分配、不复用
        self._req_id_to_state_id: Dict[str, int] = {}
        self._next_state_id: int = 0
        # 外部语料记账：corpus_id → 该语料贡献的 token 数
        self._corpus_token_counts: Dict[str, int] = {}
        self._total_loaded_tokens: int = 0

    def _get_state_id(self, req_id: str) -> int:
        """把 req_id 映射到 C++ 侧的增量匹配状态槽位，不存在则新分配。

        示例：
            _get_state_id("chatcmpl-abc") → 0   （首次，分配新槽位）
            _get_state_id("chatcmpl-xyz") → 1
            _get_state_id("chatcmpl-abc") → 0   （再次，复用已有槽位）
        """
        sid = self._req_id_to_state_id.get(req_id)
        if sid is None:
            sid = self._next_state_id
            self._next_state_id += 1
            self._req_id_to_state_id[req_id] = sid
        return sid

    def batch_put(self, batch_tokens: List[List[int]]):
        """把一批 token 序列插入 trie，供后续匹配使用（异步，需配合 synchronize）。

        参数：
            batch_tokens: 每个请求的 token 序列，长度不必相同。
                例如 [[1, 2, 3, 4, 5], [1, 2, 3, 44, 55]]
                插入后 trie 里就有了 "1→2→3" 这个共享前缀，
                以及 3 之后可能跟 4 或 44 两个分支。
        """
        self._obj.insert(batch_tokens)

    def synchronize(self):
        """等待 C++ 侧异步插入完成。

        batch_put 是异步提交的（插入在后台线程做，以免阻塞 GPU 主循环），
        因此在 batch_get 读取之前必须先 synchronize，否则可能匹配不到刚插入的内容。
        """
        self._obj.synchronize()  # type: ignore

    @property
    def remaining_token_budget(self) -> int:
        """外部语料还能再加载多少 token。

        示例：上限 1000 万、已加载 300 万 → 返回 700 万。
        """
        return self.external_corpus_max_tokens - self._total_loaded_tokens

    def load_external_corpus_named(
        self, corpus_id: str, chunks: Iterable[Sequence[int]]
    ) -> int:
        """把一份具名外部语料装进后缀自动机（SAM）。

        外部语料用于「预热」：比如你的业务是固定模板的 JSON 输出，
        提前把模板语料灌进来，第一个请求就能有高命中率，
        而不用等运行时慢慢累积。

        参数：
            corpus_id: 语料标识，同一 id 不允许重复加载。例："product-faq-v1"
            chunks: 已 tokenize 的分块序列，例 [[1,2,3], [4,5,6,7], ...]

        返回：
            实际加载的 token 数，例如 350000。可能小于语料总量——
            当剩余预算不足时 C++ 侧会截断。

        异常：
            ValueError: corpus_id 已存在时抛出，需先 remove_external_corpus。
        """
        if corpus_id in self._corpus_token_counts:
            raise ValueError(
                f"External corpus '{corpus_id}' already exists. Remove it before "
                f"adding a new corpus with the same id."
            )
        # 中译（Note by kpham-sgl）：这里读到的 remaining_token_budget 可能是过期的
        # （例如加载过程中有其他语料被移除），这会让预算判断偏保守。
        # 这是可以接受的——否则 load_external_corpus_named 就得在每个 chunk 之后
        # 重新检查一次预算，效率太低。
        _, loaded_token_count = self._obj.load_external_corpus_named(
            corpus_id, chunks, self.remaining_token_budget
        )
        return loaded_token_count

    # 中译：加载成功后再提交语料的记账信息。只能在后台线程 join 之后调用
    #（或在同步版 load_external_corpus_named 返回之后）。
    # 之所以把「加载」和「记账」拆成两步：加载可能在后台线程里跑很久，
    # 中途失败时不应污染 _total_loaded_tokens。
    def commit_external_corpus_load(
        self, corpus_id: str, loaded_token_count: int
    ) -> None:
        """提交一次外部语料加载的记账。

        示例：commit_external_corpus_load("product-faq-v1", 350000)
        之后 _total_loaded_tokens 增加 350000，remaining_token_budget 相应减少。
        """
        self._corpus_token_counts[corpus_id] = loaded_token_count
        self._total_loaded_tokens += loaded_token_count

    def remove_external_corpus(self, corpus_id: str) -> None:
        """卸载一份外部语料，并归还其占用的 token 预算。

        示例：某语料加载了 35 万 token，移除后 remaining_token_budget 增加 35 万。
        """
        self._obj.remove_corpus(corpus_id)
        old_count = self._corpus_token_counts.pop(corpus_id, 0)
        self._total_loaded_tokens -= old_count

    def list_external_corpora(self) -> Dict[str, int]:
        """列出当前已加载的外部语料。

        返回：corpus_id → token 数的字典
            例如 {"product-faq-v1": 350000, "code-snippets": 120000}
        """
        return self._obj.list_corpora()

    def reset(self):
        """清空整个 trie 与所有请求的匹配状态。

        注意 state_id 计数器也归零——重置后新请求会从 0 开始重新分配槽位。
        """
        self._obj.reset()  # type: ignore
        self._req_id_to_state_id.clear()
        self._next_state_id = 0

    def batch_get(
        self,
        req_ids: List[str],
        batch_tokens: List[List[int]],
        total_lens: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """核心方法：按各请求的历史后缀在 trie 中匹配，BFS 展开出草稿树。

        参数（示例 bs=3, draft_token_num=8）：
            req_ids: 各请求 id，用于取增量匹配状态
                ["query-0", "query-1", "query-2"]
            batch_tokens: 各请求本轮**新增**的 token（增量匹配，不必传整条历史）
                [[1, 2, 3], [3, 44], [3, 6, 999]]
            total_lens: 各请求的历史总长度，C++ 侧据此判断状态是否连续
                [3, 2, 3]

        返回 (decoding_ids, decoding_masks)：
            decoding_ids: shape [bs * draft_token_num] = [24]，
                每 8 个一组是一个请求的草稿树节点（不足则补填充值）。
                承接类 docstring 的例子，若 trie 里存过
                [1,2,3,4,5,...] 和 [1,2,3,44,55,...]，
                则 query [1,2,3] 匹配到 3 的后继有 {4, 44} 两个分支，
                该请求那 8 个位置形如 [3, 4, 44, 5, 55, 6, 66, ...]
            decoding_masks: shape [bs * draft_token_num * draft_token_num] = [192]，
                每个请求一个 8x8 的下三角形状 0/1 矩阵，
                mask[i][j]=1 表示「节点 i 的祖先链上包含节点 j」。
                这个矩阵直接决定了树形注意力的可见性。

        注意调用前必须先 synchronize()，否则可能读不到最新插入的内容。
        """
        state_ids = [self._get_state_id(rid) for rid in req_ids]
        return self._obj.match_stateful(state_ids, batch_tokens, total_lens)

    def erase_match_state(self, req_ids: List[str]):
        """请求离开 batch 时清理其增量匹配状态，释放 C++ 侧槽位。

        参数：
            req_ids: 已结束/被抢占的请求 id 列表，例 ["query-0", "query-2"]

        不存在的 id 会被静默跳过（pop 带默认值），所以重复调用是安全的。
        必须清理：否则长时间运行后 state_id 会无限增长，C++ 侧状态表持续膨胀。
        """
        state_ids = []
        for rid in req_ids:
            sid = self._req_id_to_state_id.pop(rid, None)
            if sid is not None:
                state_ids.append(sid)
        if state_ids:
            self._obj.erase_states(state_ids)

    def leaf_paths_from_mask(
        self, tokens: List[int], tree_mask: List[List[int]]
    ) -> List[List[int]]:
        """
        中译：根据 0/1 的 tree_mask 找出所有叶子路径
        （即那些不是任何其他路径之前缀的路径）。

        Args（中译）：
            tree_mask : List[List[int]]   # n×n 的 0/1 矩阵
            tokens    : List[int]         # 与列一一对应的 token 列表

        Returns（中译）：
            List[List[int]]  # 仅包含叶子路径的 token 列表，保持其出现顺序

        --------------------------------------------------------------
        这个方法只用于调试展示（把树还原成人能读的几条路径）。
        原理：mask 的第 i 行标出了「节点 i 的完整祖先链」，
        所以第 i 行的 1 的集合就是「从根到 i 的路径」。
        若 A 行的集合 ⊆ B 行的集合，说明 A 是 B 的前缀，A 就不是叶子。

        数值示例（承接前面的树，n=4）：
            tokens    = [464("The"), 4062("quick"), 3105("slow"), 7586("brown")]
            tree_mask = [[1, 0, 0, 0],   # 行0: 路径 {0}       = The
                         [1, 1, 0, 0],   # 行1: 路径 {0,1}     = The quick
                         [1, 0, 1, 0],   # 行2: 路径 {0,2}     = The slow
                         [1, 1, 0, 1]]   # 行3: 路径 {0,1,3}   = The quick brown

        倒序扫描：
            行3 {0,1,3} → 尚无 leaf 包含它 → 收为叶子
            行2 {0,2}   → 不被 {0,1,3} 包含 → 收为叶子
            行1 {0,1}   → ⊆ {0,1,3}，是前缀 → 跳过
            行0 {0}     → ⊆ {0,1,3}，是前缀 → 跳过

        最终 leaf_rows 复原顺序为 [2, 3]，返回：
            [[464, 3105],        # "The slow"
             [464, 4062, 7586]]  # "The quick brown"
        """

        # 把每一行的 1 所在列号收集成集合，便于用 ⊆ 判断前缀关系
        row_sets = [
            (i, {idx for idx, v in enumerate(row) if v == 1})
            for i, row in enumerate(tree_mask)
        ]
        leaf_sets = []
        leaf_rows = []

        # 倒序扫描：越靠后的行路径越长，先被收作叶子；
        # 之后遇到的短路径若是已收叶子的子集（前缀），就跳过。
        for i, cur_set in reversed(row_sets):
            if any(cur_set <= kept for kept in leaf_sets):
                continue
            leaf_sets.append(cur_set)
            leaf_rows.append(i)

        # 倒序收集的，翻回原始出现顺序
        leaf_rows.reverse()
        result = []
        # 把每条叶子路径的列号还原成 token 序列
        for r in leaf_rows:
            path = [tokens[col] for col in range(len(tokens)) if tree_mask[r][col] == 1]
            result.append(path)

        return result

    def debug_result(
        self, decoding_ids: np.ndarray, decoding_masks: np.ndarray, tokenizer=None
    ):
        """把 batch_get 的扁平输出还原成可读的草稿路径并打日志。

        参数（示例 bs=3, draft_token_num=8）：
            decoding_ids: batch_get 的第一个返回值，扁平 shape [24]
            decoding_masks: batch_get 的第二个返回值，扁平 shape [192]
            tokenizer: 传入则额外 decode 成文本，便于肉眼判断草稿质量。

        典型输出：
            draft path 0: [3, 4, 5, 6] -> " is a test"
            draft path 0: [3, 44, 55] -> " was another"

        排查命中率低时很有用：如果打出来的路径明显和实际输出不沾边，
        说明该场景不适合 NGRAM（见 max_bfs_breadth / match_type 调参）。
        """
        # 扁平 [24] → 按请求分组 [3, 8]
        decoding_ids = decoding_ids.reshape(-1, self.draft_token_num)
        # 扁平 [192] → 每请求一个方阵 [3, 8, 8]
        decoding_masks = decoding_masks.reshape(
            -1, self.draft_token_num, self.draft_token_num
        )
        logger.info(f"\n{decoding_ids=}\n{decoding_masks=}")
        for i in range(decoding_ids.shape[0]):
            leaf_paths = self.leaf_paths_from_mask(
                decoding_ids[i].tolist(), decoding_masks[i].tolist()
            )
            if tokenizer is None:
                logger.info(f"draft path {i}: {leaf_paths}")
            else:
                logger.info(f"result {i}:")
                for leaf_path in leaf_paths:
                    logger.info(
                        f"draft path {i}: {leaf_path} -> {tokenizer.decode(leaf_path, ensure_ascii=False)}"
                    )


# 中译：主函数——可直接 `python ngram_corpus.py` 运行的最小自测样例，
# 演示 put → synchronize → get → debug 的完整链路。
if __name__ == "__main__":
    format = f"%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(
        level=logging.DEBUG,
        format=format,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    # 两条历史序列，共享前缀 [1,2,3]，之后分叉为 4... 和 44...
    token_ids = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 44, 55, 66, 77, 88, 99, 100],
    ]
    corpus = NgramCorpus(max_trie_depth=12, draft_token_num=8)
    corpus.batch_put(token_ids)

    # 插入是异步的，读之前必须同步
    corpus.synchronize()
    # 三种典型 query：
    #   [1, 2, 3]   → 命中共享前缀，3 之后有 {4, 44} 两个分支 → 树会分叉
    #   [3, 44]     → 命中第二条历史，后续是单链 55→66→77...
    #   [3, 6, 999] → 999 从未出现过 → 匹配失败，草稿树基本为空
    queries = [[1, 2, 3], [3, 44], [3, 6, 999]]
    decoding_ids, decoding_masks = corpus.batch_get(
        req_ids=[f"query-{i}" for i in range(len(queries))],
        batch_tokens=queries,
        total_lens=[len(q) for q in queries],
    )

    corpus.debug_result(decoding_ids, decoding_masks)
