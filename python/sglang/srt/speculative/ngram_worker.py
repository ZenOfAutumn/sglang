"""NGRAM 投机解码的主 Worker。

与 EAGLE 的根本区别：**没有草稿模型**。草稿 token 来自 CPU 上的 N-gram 后缀
trie（`cpp_ngram/`），靠「历史上出现过的 token 序列会再次出现」这个假设。

一轮完整流程（`forward_batch_generation`）：

    ┌─ 1. Corpus Match ──────────────────────────────────────┐
    │   拿每个请求的末尾 max_trie_depth 个 token 去 trie 匹配 │
    │   BFS 展开出 draft_token_num 个节点的草稿树            │
    │   （_prepare_draft_tokens，纯 CPU）                     │
    └────────────────────────────────────────────────────────┘
                            ↓
    ┌─ 2. Target Verify ─────────────────────────────────────┐
    │   目标模型一次前向验证整棵树（forward_mode 改写为       │
    │   TARGET_VERIFY，input_ids 换成 draft_tokens）          │
    └────────────────────────────────────────────────────────┘
                            ↓
    ┌─ 3. Sample & Accept ───────────────────────────────────┐
    │   eagle_sample() 沿树路径贪心接受，复用 EAGLE 的实现    │
    └────────────────────────────────────────────────────────┘
                            ↓
    ┌─ 4. Corpus Update ─────────────────────────────────────┐
    │   把本轮实际生成的 token 写回 trie，供后续请求匹配      │
    │   （_update_ngram_corpus）                              │
    └────────────────────────────────────────────────────────┘

性能特征：GSM8K 上 CI 的接受长度门槛是 1.8（见
`python/sglang/test/server_fixtures/ngram_fixture.py`），
明显低于 EAGLE3/MTP 的 2.6~3.2。适合输出高度重复的场景
（代码补全、RAG 摘录、固定模板 JSON），不适合开放式生成。

已知约束（见 docs/advanced_features/speculative_decoding.md）：
仅 CUDA、不支持 --enable-dp-attention、会禁用 overlap scheduler 与
mixed chunked prefill。但它是目前唯一能与 LoRA 共存的投机算法。
"""

import logging
from typing import List, Optional

import numpy as np
import torch
from sgl_kernel.speculative import reconstruct_indices_from_tree_mask

from sglang.srt.layers.utils.logprob import compute_spec_v2_logprobs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, EagleDraftWorkerBase
from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus
from sglang.srt.speculative.eagle_utils import eagle_sample
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.spec_utils import (
    commit_mamba_states_after_verify,
    generate_token_bitmask,
    move_accept_tokens_to_target_kvcache,
    prepare_mamba_track_for_verify,
    record_stream_for_v2_verify,
)
from sglang.srt.speculative.triton_ops.cache_locs import (
    assign_extend_cache_locs_func as assign_extend_cache_locs_func,
)
from sglang.srt.utils.async_probe import maybe_detect_inf, maybe_detect_nan

logger = logging.getLogger(__name__)


# 树形注意力 mask 的两种形态：
#   FULL_MASK: [draft_token_num, seq_len + draft_token_num] 的完整矩阵
#   QLEN_MASK: 只存树内 [draft_token_num, draft_token_num] 部分，历史全可见由后端隐式处理
# 见下方 _prepare_for_speculative_decoding 中的性能说明（QLEN 约快 8%）。
USE_FULL_MASK = True


class NGRAMWorker(BaseSpecWorker):
    """NGRAM 投机解码 Worker。

    它「包着」目标 worker：所有真正的模型前向都委托给 `_target_worker`，
    本类只负责草稿的产生（trie 匹配）与消费（验证后写回）。
    因此 `draft_worker` 属性返回 None——根本不存在草稿模型。
    """

    def alloc_memory_pool(self, **kwargs):
        """接管目标 worker 的内存池句柄，并预分配草稿树用的常驻张量。

        中译：__init__ 执行时目标内存池尚不存在，所以这一步必须延后到这里。
        调用时序：Scheduler 先建好 target worker 的内存池，再回调本方法。
        """
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            self._target_worker.get_memory_pool()
        )
        self.max_batch_size = self.model_runner.max_running_requests
        self._init_preallocated_tensors()

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.enable_overlap = not server_args.disable_overlap_schedule
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size
        self.draft_token_num: int = server_args.speculative_num_draft_tokens
        self.max_trie_depth: int = server_args.speculative_ngram_max_trie_depth
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        # req_to_token_pool / token_to_kv_pool_allocator are set in
        # alloc_memory_pool(), after the target pools are allocated.
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"

        self.adaptive_controller = None
        # 中译：上一轮 decode batch 的 rid 集合；用于清理已离开 batch 的请求在
        # 语料库中的匹配状态（见 forward_batch_generation）。
        # 为什么要自己记：req.finished() 在 overlap 模式下滞后一轮，不可靠。
        self._prev_decode_rids: set = set()

        self.ngram_corpus = NgramCorpus(
            min_bfs_breadth=server_args.speculative_ngram_min_bfs_breadth,
            max_bfs_breadth=server_args.speculative_ngram_max_bfs_breadth,
            match_type=server_args.speculative_ngram_match_type,
            capacity=server_args.speculative_ngram_capacity,
            max_trie_depth=server_args.speculative_ngram_max_trie_depth,
            draft_token_num=server_args.speculative_num_draft_tokens,
            external_sam_budget=server_args.speculative_ngram_external_sam_budget,
            external_corpus_max_tokens=server_args.speculative_ngram_external_corpus_max_tokens,
        )
        # 启动期预热：把 --speculative-ngram-external-corpus-path 指定的语料
        # 同步灌进 SAM。这里用同步加载（不像运行时 HTTP 加载那样开后台线程），
        # 因为服务还没开始接请求，阻塞在这里是安全的；
        # 而且语料没装好就服务，头几个请求的命中率会很难看。
        if server_args.speculative_ngram_external_corpus_path is not None:
            from sglang.srt.speculative.cpp_ngram.external_corpus import (
                iter_external_corpus_chunks,
            )

            # 用语料文件路径本身作为 corpus_id，天然唯一
            corpus_path = server_args.speculative_ngram_external_corpus_path
            chunks = list(
                iter_external_corpus_chunks(
                    corpus_path,
                    target_worker.tokenizer,
                    server_args.speculative_ngram_external_corpus_max_tokens,
                )
            )
            loaded = self.add_external_corpus(corpus_path, chunks)
            self.commit_corpus_load(corpus_path, loaded)
            logger.info(
                "Loaded external ngram corpus '%s' (%d tokens).",
                corpus_path,
                loaded,
            )

    @property
    def target_worker(self) -> TpModelWorker:
        return self._target_worker

    @property
    def draft_worker(self) -> Optional[EagleDraftWorkerBase]:
        """中译：NGRAM 没有草稿模型；草稿来自 CPU 侧的语料库。

        返回 None 是有语义的——调度器会用 `self.draft_worker or self.tp_worker`
        这种写法来决定把请求派发给谁（见下方 update_weights_from_tensor）。
        """
        return None

    def clear_cache_pool(self):
        """清空语料库与所有匹配状态。

        触发时机：/flush_cache 接口、权重更新后等需要彻底重置的场合。
        注意这会丢掉所有累积的 n-gram，命中率要重新爬升；
        外部预热语料也一并清空。
        """
        self.ngram_corpus.reset()
        self._prev_decode_rids = set()

    def update_weights_from_tensor(self, recv_req):
        """把权重更新请求透传给目标 worker。

        中译：NGRAM 没有自己的草稿权重——n-gram 语料库只是一个由请求 token 流
        构建出来的 CPU 查找结构——而它的 `model_runner` 与目标 worker 共享。
        调度器的 mixin 是通过 `self.draft_worker or self.tp_worker` 来派发的，
        所以如果没有这个方法，在 `--speculative-algorithm NGRAM` 下任何调用
        `update_weights_from_tensor` 的地方都会抛 AttributeError。
        """
        return self.target_worker.update_weights_from_tensor(recv_req)

    def add_external_corpus(self, corpus_id: str, token_chunks: list[list[int]]) -> int:
        """加载外部语料，返回实际加载的 token 数（如 350000）。

        注意这里只「加载」不「记账」，记账要另外调 commit_corpus_load——
        因为本方法可能在后台线程里跑很久，中途失败不应污染预算统计。
        """
        return self.ngram_corpus.load_external_corpus_named(corpus_id, token_chunks)

    def commit_corpus_load(self, corpus_id: str, loaded_token_count: int) -> None:
        """提交一次加载的记账，例 commit_corpus_load("faq-v1", 350000)。"""
        self.ngram_corpus.commit_external_corpus_load(corpus_id, loaded_token_count)

    def remove_external_corpus(self, corpus_id: str) -> None:
        """卸载指定外部语料并归还其 token 预算。"""
        self.ngram_corpus.remove_external_corpus(corpus_id)

    def list_external_corpora(self) -> dict[str, int]:
        """返回 corpus_id → token 数，例 {"faq-v1": 350000}。"""
        return self.ngram_corpus.list_external_corpora()

    def _efficient_concat_last_n(self, seq1: List[int], seq2: List[int], n: int):
        """取「seq1 ⊕ seq2」的最后 n 个元素，但避免真的拼接整个 seq1。

        用途：请求的 origin_input_ids（prompt，可能上万 token）与 output_ids
        逻辑上是一条序列，但匹配只需要末尾 max_trie_depth（默认 18）个。
        朴素写法 `(seq1 + seq2)[-n:]` 会复制整个 prompt，这里避免了。

        数值示例（n=5）：
            seq1=[1,2,3,4,5,6,7,8,9,10]，seq2=[91,92,93,94,95,96,97]
                → seq2 已有 7 ≥ 5 个 → 返回 seq2[-5:] = [93,94,95,96,97]
                  （完全不碰 seq1）

            seq1=[1,2,3,4,5,6,7,8,9,10]，seq2=[91,92]
                → seq2 只有 2 个，还差 3 个 → 返回
                  seq1[-3:] + seq2 = [8,9,10,91,92]

            seq1=[1,2]，seq2=[]
                → 还差 5 个但 seq1 只有 2 个 → 返回 [1,2]（切片不会越界）
        """
        seq2_len = len(seq2)
        if seq2_len >= n:
            return seq2[-n:]

        need_from_seq1 = n - seq2_len
        return seq1[-need_from_seq1:] + seq2

    def _init_preallocated_tensors(self):
        """按最大 batch 预分配所有草稿张量，运行时只做 copy_ 不再分配显存。

        这样做有两个原因：
          1. 避免每轮 decode 都 malloc/free 显存（CUDA 分配器抖动）；
          2. CUDA graph 捕获要求张量地址稳定。

        数值示例（max_batch_size=8, draft_token_num=16）：
            max_total_drafts    = 8 * 16      = 128
            max_total_mask_size = 8 * 16 * 16 = 2048

            draft_tokens          shape [128]      int64  → 1 KB
            positions             shape [128]      int64  → 1 KB
            retrieve_indexes      shape [8, 16]    int64
            retrieve_next_token   shape [8, 16]    int64
            retrieve_next_sibling shape [8, 16]    int64
            tree_mask             shape [2048]     bool   → 2 KB

        总量极小（几 KB），这也是 NGRAM 相比 EAGLE 的一大优势——
        没有草稿模型权重，几乎不占额外显存。
        """
        max_total_drafts = self.max_batch_size * self.draft_token_num
        max_total_mask_size = (
            self.max_batch_size * self.draft_token_num * self.draft_token_num
        )

        self.draft_tokens = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        self.retrieve_indexes = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.retrieve_next_token = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.retrieve_next_sibling = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.positions = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        self.tree_mask = torch.empty(
            (max_total_mask_size,), dtype=torch.bool, device=self.device
        )

        # 下面这些 *_batch 列表是「按 batch size 索引的视图缓存」：
        # 第 bs 个元素就是对应 batch size 的切片视图（不复制数据）。
        # 运行时 `self.draft_tokens_batch[bs]` 即可 O(1) 拿到正确形状的张量，
        # 省掉每轮重复切片的 Python 开销。
        self.draft_tokens_batch = []
        self.tree_mask_batch = []
        self.retrieve_indexes_batch = []
        self.retrieve_next_token_batch = []
        self.retrieve_next_sibling_batch = []
        self.positions_batch = []

        # 从 0 开始建到 max_batch_size（含），所以可以直接用 bs 当下标。
        # 例：max_batch_size=8, draft_token_num=16 时，
        #   positions_batch[3]   = positions[:48]      （3*16）
        #   tree_mask_batch[3]   = tree_mask[:768]     （3*16*16）
        #   retrieve_indexes_batch[3] = retrieve_indexes[:3, :]
        for bs in range(0, self.max_batch_size + 1):
            self.retrieve_indexes_batch.append(self.retrieve_indexes[:bs, :])
            self.retrieve_next_token_batch.append(self.retrieve_next_token[:bs, :])
            self.retrieve_next_sibling_batch.append(self.retrieve_next_sibling[:bs, :])
            self.positions_batch.append(self.positions[: bs * self.draft_token_num])
            self.draft_tokens_batch.append(
                self.draft_tokens[: bs * self.draft_token_num]
            )
            self.tree_mask_batch.append(
                self.tree_mask[: bs * self.draft_token_num * self.draft_token_num]
            )

    def on_verify_complete_cpu(
        self, num_correct_drafts_per_req: list[int], batch_size: int = 0
    ) -> None:
        """验证完成后的 CPU 侧回调，把接受情况喂给自适应控制器。

        参数：
            num_correct_drafts_per_req: 各请求本轮**猜对的草稿数**（不含 bonus token）
                例 [2, 0, 5] —— 第 1 个请求猜对 2 个，第 2 个全猜错，第 3 个猜对 5 个
            batch_size: 由结果处理器以关键字参数传入，本实现未使用。

        中译：签名必须与 BaseSpecWorker.on_verify_complete_cpu 保持一致；
        结果处理器是用关键字参数 batch_size 来调用它的。

        自适应控制器据此动态调整投机参数（如草稿数），命中率低时少猜以减少浪费。
        未启用 --speculative-adaptive 时 adaptive_controller 为 None，本方法空转。
        """
        if self.adaptive_controller is not None:
            self.adaptive_controller.on_verify_complete(num_correct_drafts_per_req)

    def _prepare_draft_tokens(
        self, batch: ScheduleBatch
    ) -> tuple[np.ndarray, np.ndarray]:
        """步骤 1：为 batch 内每个请求从语料库匹配出草稿树。

        参数：
            batch: 当前 decode batch，需已带上一轮的 spec_info。

        返回 (req_drafts, mask)，均为 numpy 数组：
            req_drafts: shape [bs * draft_token_num]，草稿 token id 拍平
                bs=2, draft_token_num=16 → shape (32,)
            mask: shape [bs * draft_token_num * draft_token_num]，0/1 树形 mask 拍平
                bs=2, draft_token_num=16 → shape (512,)

        断言：匹配结果必须恰好是 bs*draft_token_num 个——C++ 侧即使匹配不足
        也会用填充值补齐，所以这里是个不变量而非运行时校验。
        """
        bs = len(batch.reqs)
        stride = self.draft_token_num

        # 上一轮验证的接受结果。overlap 模式下这些还没写进 req.output_ids，
        # 需要在下面手动"拼接"回历史，否则匹配用的上下文会缺尾巴。
        prev_token_ids, prev_accept_lens = (
            batch.spec_info.accept_tokens,
            batch.spec_info.accept_lens,
        )
        # trie 匹配在 CPU 上做，先把张量搬下来（一次 D2H，几十字节，开销可忽略）
        if not prev_token_ids.is_cpu:
            prev_token_ids = prev_token_ids.cpu()
            prev_accept_lens = prev_accept_lens.cpu()
        # 中译：worker 级别的暂存区——在草稿准备阶段写入，
        # 在同一次 forward 调用内、verify 之后由 _update_ngram_corpus 消费。
        # 存成 Python list 而非张量，因为后续都是纯 CPU 的切片操作。
        self.prev_token_ids = prev_token_ids.tolist()
        self.prev_accept_lens = prev_accept_lens.tolist()

        # 等待上一轮 batch_put 的异步插入落盘，否则可能匹配不到刚生成的 token
        self.ngram_corpus.synchronize()
        req_ids = []
        batch_tokens = []
        total_lens = []
        assert len(batch.reqs) == len(self.prev_accept_lens)
        # 中译：overlap 模式的结果处理滞后一轮，所以上一轮被接受的 token 还没进入
        # req.output_ids，必须从 spec_info 里拼接回来。而同步模式和带 grammar 的
        # batch 会在下一次草稿准备之前就处理完结果，此时 output_ids 已经是完整的，
        # 再拼接就会把尾部重复一遍。
        use_prev_tokens = self.enable_overlap and not batch.has_grammar
        i = 0
        for req in batch.reqs:
            # 从拍平的 accept_tokens 里切出第 i 个请求真正被接受的那几个。
            # 例：stride=16、prev_accept_lens=[3, 1] 时，
            #     i=0 取 prev_token_ids[0:3]，i=1 取 prev_token_ids[16:17]
            #     —— 注意起点是 i*stride 而非累加，因为存储是按 stride 定长分段的。
            prev_tokens = (
                self.prev_token_ids[i * stride : i * stride + self.prev_accept_lens[i]]
                if use_prev_tokens
                else []
            )
            # 拼出用于匹配的「历史后缀」，长度 ≤ max_trie_depth（默认 18）。
            # 逻辑序列 = prompt ⊕ 已输出 ⊕ 上一轮新接受的，取最后 18 个。
            # 先对 output_ids 做 [-18:] 是为了避免把长输出整个复制一遍。
            check_token = self._efficient_concat_last_n(
                list(req.origin_input_ids),
                list(req.output_ids[-self.max_trie_depth :]) + prev_tokens,
                self.max_trie_depth,
            )
            req_ids.append(req.rid)
            batch_tokens.append(check_token)
            i += 1
            # total_lens 传的是**逻辑总长度**（不是 check_token 的长度）——
            # C++ 侧靠它判断增量匹配状态是否连续：若本次总长 = 上次总长 + 增量，
            # 就能接着上次的 trie 节点往下走，否则从头重新匹配。
            total_lens.append(
                len(req.origin_input_ids) + len(req.output_ids) + len(prev_tokens)
            )
        # 进入 C++ trie 做批量匹配 + BFS 展开，这是整个草稿阶段唯一的重活
        req_drafts, mask = self.ngram_corpus.batch_get(
            req_ids, batch_tokens, total_lens
        )
        total_draft_token_num = len(req_drafts)

        # 中译：检查是否需要做投机解码；这里我们始终强制执行。
        # 即匹配失败的请求也会占满 draft_token_num 个位置（填充值），
        # 保持张量形状恒定以便 CUDA graph 复用——代价是白验证一些无效候选。
        assert (
            total_draft_token_num == bs * self.draft_token_num
        ), f"{total_draft_token_num=}, {bs=}, {self.draft_token_num=}"
        return req_drafts, mask

    def _prepare_for_speculative_decoding(self, batch: ScheduleBatch):
        """把一个普通的 decode batch **原地改写**成树形验证 batch。

        改写内容：
            forward_mode:  DECODE      → TARGET_VERIFY
            input_ids:     [bs] 个     → [bs*draft_token_num] 个草稿 token
            out_cache_loc: 每请求 1 槽 → 每请求 draft_token_num 槽
            spec_info:     上轮的      → 新构造的 NgramVerifyInput

        非 decode 的 batch（extend / idle）直接返回，不做任何改写。
        """
        # 中译：仅处理 decode——extend 走的是普通的目标模型前向；而 IDLE batch
        # 必须保留它自己的 forward_mode，不能被下面改写成 TARGET_VERIFY
        #（等 DP attention 支持落地后这一点才会真正生效）。
        if not batch.forward_mode.is_decode():
            return

        bs = len(batch.reqs)

        # 从预分配池里按当前 bs 取视图，零拷贝、零分配
        retrieve_index = self.retrieve_indexes_batch[bs]
        retrieve_next_token = self.retrieve_next_token_batch[bs]
        retrieve_next_sibling = self.retrieve_next_sibling_batch[bs]
        positions = self.positions_batch[bs]
        tree_mask = self.tree_mask_batch[bs]
        draft_tokens = self.draft_tokens_batch[bs]

        # CPU 侧匹配出草稿树，再异步搬上 GPU（non_blocking + 预分配 = 可与后续重叠）
        req_drafts, mask = self._prepare_draft_tokens(batch)
        tree_mask.copy_(torch.from_numpy(mask), non_blocking=True)
        draft_tokens.copy_(torch.from_numpy(req_drafts), non_blocking=True)

        # 中译：利用 tree_mask 生成 positions 和若干索引。
        # 这个 kernel 把 0/1 矩阵反解成「左孩子右兄弟」表示（见 ngram_info.py 头注释）：
        #   positions            ← 每个节点的绝对位置 = seq_len + 该节点在树中的深度
        #   retrieve_index       ← 节点的全局下标
        #   retrieve_next_token  ← 第一个子节点下标，无子为 -1
        #   retrieve_next_sibling← 下一个兄弟下标，无兄弟为 -1
        # 带 # mutable 标注的四个参数是**输出**参数，被 kernel 原地写入。
        reconstruct_indices_from_tree_mask(
            tree_mask,
            batch.seq_lens,
            positions,  # mutable
            retrieve_index,  # mutable
            retrieve_next_token,  # mutable
            retrieve_next_sibling,  # mutable
            bs,
            self.draft_token_num,
        )

        # 中译（NOTE）：QLEN_MASK 比 FULL_MASK 更快，但需要在 flashinfer 侧做相应改动。
        # 实测约有 8% 的性能提升（效果大致与 batch size 成正比）。
        #
        # FULL_MASK 的做法：为每个请求拼出 [draft_token_num, seq_len + draft_token_num]
        # 的完整可见性矩阵——左半段全 1（历史 KV 对所有草稿 token 可见），
        # 右半段是树内 mask（只能看见自己的祖先链）。
        #
        # 示例（draft_token_num=4, seq_len=100）：
        #   req_mask 左半 = ones(4, 100)
        #   req_mask 右半 = mask[i]，即 4x4 的树内可见性
        #   拼接后 shape (4, 104)，flatten 成 416 个 bool
        #   bs=2、另一请求 seq_len=50 → 216 个，总计 632
        #   （与 ngram_info.generate_attn_arg_prefill 的 mask_numel 推导一致）
        if USE_FULL_MASK:
            tree_mask = []
            mask = mask.reshape(bs, self.draft_token_num, self.draft_token_num)
            # 中译（TODO by siyuan）：这里的 for 循环在大 batch size 下会带来显著开销，
            # 可以改写成一个 kernel。
            for i in range(bs):
                seq_len = batch.seq_lens_cpu[i]
                # 历史部分全可见：每个草稿 token 都能 attend 到全部已有 KV
                req_mask = torch.ones(
                    (self.draft_token_num, seq_len), device=self.device
                )
                req_mask = torch.cat(
                    (
                        req_mask,
                        torch.from_numpy(mask[i]).to(
                            device=self.device, non_blocking=True
                        ),
                    ),
                    dim=1,
                ).to(torch.bool)
                tree_mask.append(req_mask.flatten())
            tree_mask = torch.cat(tree_mask, dim=0)

        # ↓↓↓ 以下是对 batch 的原地改写：普通 decode → 树形验证 ↓↓↓
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        # 输入从「每请求 1 个 token」变成「每请求 draft_token_num 个草稿 token」
        batch.input_ids = draft_tokens
        # 相应地，KV cache 也要为每个请求预留 draft_token_num 个连续槽位，
        # 区间为 [seq_lens, seq_lens + draft_token_num)。
        # 例：seq_lens=[100, 50], draft_token_num=16
        #     → req0 占 [100,116)，req1 占 [50,66)
        # 未被接受的槽位会在验证后由 move_accept_tokens_to_target_kvcache 回收。
        batch.out_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=batch.req_to_token_pool.req_to_token,
            start_offset=batch.seq_lens,
            end_offset=batch.seq_lens + self.draft_token_num,
            batch_size=bs,
            draft_token_num=self.draft_token_num,
            device=self.device,
        )

        # Mamba/线性注意力类模型的状态需要在验证前打快照，
        # 因为它们的状态是就地演进的，猜错时必须能回滚（纯 Transformer 模型此处空转）
        prepare_mamba_track_for_verify(batch)

        # 打包成验证输入，交给注意力后端与采样阶段
        batch.spec_info = NgramVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=positions,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            draft_token_num=self.draft_token_num,
        )

    def _update_ngram_corpus(self, batch: ScheduleBatch):
        """步骤 4：把本轮各请求的最新历史后缀写回语料库，供后续匹配。

        参数：
            batch: 刚完成验证的 batch。

        无返回值。插入是异步的——下一轮 _prepare_draft_tokens 里的
        synchronize() 负责等它落盘。

        插入内容与 _prepare_draft_tokens 里用来查询的 check_token 构造方式完全一致
        （同样是末尾 max_trie_depth 个 token），这保证了「这一轮查什么、
        下一轮就能匹配到什么」的自洽性。

        数值示例（max_trie_depth=18）：
            某请求 prompt 长 500、已输出 30、本轮又接受 3 个
            → put_ids 是这 533 个 token 的最后 18 个
            → trie 中因此新增了这条长度 18 的后缀及其所有前缀
        """
        batch_tokens = []
        i, stride = 0, self.draft_token_num
        # 中译：与 _prepare_draft_tokens 中相同的拼接条件——只有 overlap 模式下
        # 才存在「已接受但尚未进入 req.output_ids」的 token。
        use_prev_tokens = self.enable_overlap and not batch.has_grammar
        for req in batch.reqs:
            # 中译（FIXME）：要不要把 'extend'（即 prompt 部分）也插进缓存，
            # 经过测试差别不大，所以目前不插入。
            # 换句话说：只有 decode 阶段产生的 token 会进 trie，
            # prompt 本身不会——这也是为什么外部预热语料有价值。
            # if batch.forward_mode.is_extend():
            #     put_ids = req.origin_input_ids + req.output_ids
            # else:
            prev_tokens = (
                self.prev_token_ids[i * stride : i * stride + self.prev_accept_lens[i]]
                if use_prev_tokens
                else []
            )
            put_ids = self._efficient_concat_last_n(
                list(req.origin_input_ids),
                list(req.output_ids[-self.max_trie_depth :]) + prev_tokens,
                self.max_trie_depth,
            )
            batch_tokens.append(put_ids)
            i += 1
        self.ngram_corpus.batch_put(batch_tokens)

    def forward_batch_generation(
        self, batch: ScheduleBatch, on_publish=None
    ) -> GenerationBatchResult:
        """一轮完整的 NGRAM 投机解码：匹配 → 验证 → 接受 → 写回。

        参数：
            batch: 待处理的 batch（decode 或 extend）。
            on_publish: 可选回调，overlap 模式下用它把 new_seq_lens 提前
                发布给调度器，让下一轮的准备工作能与本轮 GPU 计算重叠。

        返回 GenerationBatchResult，关键字段（示例 bs=2, draft_token_num=16）：
            next_token_ids: 本轮确定输出的 token（拍平），长度 = accept_lens 之和
            accept_lens: 各请求接受了几个 token（**含 bonus**），如 tensor([3, 1])
                最小值恒为 1——即使草稿全猜错，目标模型自己那个 token 总是对的。
                最大值为 draft_token_num。
            new_seq_lens: 接受后的新序列长度，如 tensor([103, 51])
            next_draft_input: 下一轮用的 NgramVerifyInput（只带 accept_* 字段）

        两条分支：
            decode  → 走完整的树形验证（if 分支）
            extend  → 退化为普通前向，不投机（else 分支）；
                      prefill 阶段本来就是计算密集的，投机没有收益。
        """
        # 记录 CUDA stream，供 V2 验证路径做跨流同步
        fwd_stream = torch.get_device_module(self.device).current_stream()
        record_stream_for_v2_verify(batch, None, fwd_stream)
        bs = len(batch.reqs)

        # 打点：草稿阶段耗时（trace_only=True 表示只在开启 trace 时才真正记录）
        set_time_batch(batch.reqs, "set_spec_draft_start_time", trace_only=True)
        # 步骤 1+2 的准备：匹配草稿树并把 batch 改写成 TARGET_VERIFY
        self._prepare_for_speculative_decoding(batch)
        set_time_batch(batch.reqs, "set_spec_draft_end_time", trace_only=True)

        verify_input: NgramVerifyInput = batch.spec_info
        # 默认每个请求至少接受 1 个 token（目标模型自己吐的那个），
        # 走验证分支时会被 eagle_sample 的真实结果覆盖。
        accept_lens = torch.ones(bs, dtype=torch.int32, device=self.device)

        if batch.forward_mode.is_target_verify():
            # 中译：如有需要，先在 CPU 上准备 grammar 数据。
            # 提前搬到 CPU 是为了让下面的 bitmask 生成能与 GPU 前向并行进行。
            if batch.has_grammar:
                retrieve_next_token_cpu = verify_input.retrieve_next_token.cpu()
                retrieve_next_sibling_cpu = verify_input.retrieve_next_sibling.cpu()
                draft_tokens_cpu = verify_input.draft_token.view(
                    verify_input.retrieve_next_token.shape
                ).cpu()

            # 步骤 2：目标模型一次前向验证整棵草稿树。
            # 这是唯一一次模型调用——NGRAM 的全部收益就来自「一次前向验证 N 个候选」。
            batch_result = self.target_worker.forward_batch_generation(
                batch, is_verify=True
            )

            logits_output, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.can_run_cuda_graph,
            )

            verify_input: NgramVerifyInput = batch.spec_info
            vocab_mask = None
            if batch.has_grammar:
                # 中译：为结构化输出生成 logit mask。
                # 让 bitmask 生成这些 CPU 操作与前向计算重叠进行。
                # 树形验证下每个候选节点的语法状态都不同，所以要逐节点算 mask。
                vocab_mask = generate_token_bitmask(
                    batch.reqs,
                    verify_input,
                    retrieve_next_token_cpu,
                    retrieve_next_sibling_cpu,
                    draft_tokens_cpu,
                    batch.sampling_info.vocab_size,
                )

                if vocab_mask is not None:
                    assert verify_input.grammar is not None
                    vocab_mask = vocab_mask.to(verify_input.retrieve_next_token.device)
                    # 中译（NOTE by sk）：否则这个 vocab mask 会是上一个 extend 阶段
                    # 遗留下来的那个，被套用后会产出错误结果。
                    # 所以这里必须显式清空 sampling_info 上的旧 mask。
                    batch.sampling_info.vocab_mask = None

            # 中译：采样。
            # 先做 NaN/Inf 探针——投机解码下数值异常很难排查，
            # 因为错误会经由「接受」传播到后续轮次。
            maybe_detect_nan(
                logits_output.next_token_logits, "verify: target model logits"
            )
            maybe_detect_inf(
                logits_output.next_token_logits, "verify: target model logits"
            )
            (
                predict,
                accept_lens,
                accept_index,
            # 步骤 3：沿树路径贪心接受。直接复用 EAGLE 的实现——
            # 草稿来源不同，但「验证一棵树」这件事是完全一样的。
            # 返回：
            #   predict      各节点的目标模型预测，shape [bs, draft_token_num]
            #   accept_lens  各请求接受长度（含 bonus），如 tensor([3, 1])
            #   accept_index 被接受节点的下标，shape [bs, draft_token_num]
            ) = eagle_sample(verify_input, batch, logits_output, vocab_mask)
            # 例：seq_lens=[100, 50] ⊕ accept_lens=[3, 1] → [103, 51]
            new_seq_lens = batch.seq_lens + accept_lens
            # 按实际接受长度提交 Mamba 状态，丢弃猜错分支上的状态演进
            commit_mamba_states_after_verify(
                self.target_worker,
                batch,
                accept_lens,
                accept_index,
                self.draft_token_num,
            )
            # 按接受下标取出实际输出的 token 并拍平
            accept_tokens = predict[accept_index].flatten()
            next_token_ids = accept_tokens

            # 中译：KV 搬运器要的是「纯草稿」的计数。NGRAM 的 accept_lens
            # 包含了 bonus token（与调度器的输出口径一致），所以要减 1。
            # 例：accept_lens=[3, 1] → 猜对的草稿数 = [2, 0]
            #     （第 2 个请求草稿全错，只拿到目标模型自己的 bonus token）
            num_correct_drafts_per_req = accept_lens - 1
            # 把被接受路径上的 KV 搬到正确位置，并释放猜错分支占用的槽位
            move_accept_tokens_to_target_kvcache(
                batch,
                accept_index,
                num_correct_drafts_per_req,
                self.token_to_kv_pool_allocator,
            )
            if batch.return_logprob:
                # 中译：最后一个参数是 accept_index 的行宽减 1。NGRAM 的
                # accept_index 形状是 (bs, draft_token_num)——它的树深度不像
                # EAGLE 的 (bs, spec_steps + 1) 那样受 spec_steps 约束。
                # 例：draft_token_num=16 → 传 15
                compute_spec_v2_logprobs(
                    batch,
                    logits_output,
                    predict,
                    accept_index,
                    self.draft_token_num - 1,
                )

            if on_publish is not None:
                on_publish(new_seq_lens)

            # 步骤 4：把本轮生成的 token 写回语料库
            self._update_ngram_corpus(batch)
            # 中译：清除已离开 decode batch 的请求的匹配状态。
            # 这里不能用 req.finished()：在 overlap 模式下它是在结果处理阶段才翻转的，
            # 比请求真正离开 batch 晚一轮。
            # 最后一批的条目会在空闲期间一直保留（数量有界且很小）。
            #
            # 做法是差集：上一轮有、这一轮没有的 rid 就是已离开的。
            # 例：_prev_decode_rids={"a","b","c"}，cur_rids={"a","c"}
            #     → departed_rids={"b"} → 清理 b 的 C++ 侧状态槽位
            cur_rids = {req.rid for req in batch.reqs}
            departed_rids = self._prev_decode_rids - cur_rids
            if departed_rids:
                self.ngram_corpus.erase_match_state(list(departed_rids))
            self._prev_decode_rids = cur_rids
            # 改回 DECODE：batch 是原地改写的，必须还原，否则调度器后续逻辑会错乱
            batch.forward_mode = ForwardMode.DECODE

        else:
            # 非 decode（即 extend/prefill）：不投机，走普通前向。
            # prefill 本身已经是计算密集的，投机拿不到收益。
            batch_result = self.target_worker.forward_batch_generation(batch)
            logits_output, predict, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.next_token_ids,
                batch_result.can_run_cuda_graph,
            )
            # extend 阶段序列长度由调度器自己推进，这里只 clone 一份不做修改
            new_seq_lens = batch.seq_lens.clone()

            # 把 extend 的输出伪装成「接受了 1 个 token」的验证结果，
            # 使下一轮 _prepare_draft_tokens 能用统一的方式读取，无需分支判断。
            # 例：bs=2, draft_token_num=16, predict=[5, 9]
            #     → accept_tokens 形如 [5,0,0,...,0, 9,0,0,...,0]（32 个元素）
            #       配合上面 accept_lens 的默认值 [1, 1]，只有首位有效。
            accept_tokens = torch.zeros(
                bs, self.draft_token_num, dtype=torch.int32, device=self.device
            )
            accept_tokens[:, 0] = predict
            accept_tokens = accept_tokens.flatten()
            next_token_ids = predict

            if on_publish is not None:
                on_publish(new_seq_lens)

        # 中译：构造下一轮的草稿输入。
        # 只填 accept_* 与 new_seq_lens 这几个字段——树结构字段要等下一轮
        # _prepare_for_speculative_decoding 里重新匹配后才有。
        next_draft_input = NgramVerifyInput(
            draft_token_num=self.draft_token_num,
            new_seq_lens=new_seq_lens,
            accept_tokens=accept_tokens,
            accept_lens=accept_lens,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            can_run_cuda_graph=can_run_cuda_graph,
            accept_lens=accept_lens,
            # 中译：由非 overlap 的 V2 调度器分支消费，用于在隔离状态恢复之后
            # 推进 batch.seq_lens；overlap 模式则改为通过 on_publish 传递。
            new_seq_lens=new_seq_lens,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=self.speculative_num_draft_tokens,
        )
