"""NGRAM 投机解码的「验证输入」数据结构。

NGRAM 与 EAGLE 的区别在于草稿来源：EAGLE 用草稿模型前向产生候选 token，
NGRAM 则直接从 N-gram 语料库（后缀 trie）里按历史 token 匹配出候选，
不需要任何额外模型。但两者的**验证阶段完全一致**——都是把候选组织成一棵树，
让目标模型一次前向验证整棵树，再沿树路径做采样/接受。

因此本文件的核心就是描述「这棵草稿树」，供注意力后端构造 mask、
供 `eagle_sample()` 沿树遍历接受 token。

--------------------------------------------------------------------------
草稿树的表示方法（贯穿全文件的示例）
--------------------------------------------------------------------------
设 draft_token_num = 4，语料库匹配出这样一棵树（root 是上一轮已确定的 token）：

        idx0: "The"  (root，上一轮的 bonus token)
         ├── idx1: "quick"
         │    └── idx3: "brown"
         └── idx2: "slow"

则各字段取值为：

    draft_token          = [ "The", "quick", "slow", "brown" ]  # 实际是 token id
                           tensor([  464,  4062,  3105,  7586], dtype=int64)
    positions            = [ 100, 101, 101, 102 ]   # 各节点在序列中的绝对位置
                                                     # idx1/idx2 是兄弟，位置相同
    retrieve_next_token  = [   1,   3,  -1,  -1 ]   # 各节点的「第一个子节点」下标
                           # root→quick, quick→brown, slow/brown 无子 → -1
    retrieve_next_sibling= [  -1,   2,  -1,  -1 ]   # 各节点的「下一个兄弟」下标
                           # quick 的兄弟是 slow；其余无兄弟 → -1

`retrieve_next_token` + `retrieve_next_sibling` 是「左孩子右兄弟」表示法，
用两个扁平数组就能完整描述任意形状的树，且能在 CUDA kernel 里 O(1) 跳转。

注意 NGRAM 的树是**不规则**的：每层分支数由语料匹配结果决定，
不像 EAGLE 那样每层固定 topk 个分支（见下方 `tree_topk` 返回 -1）。
"""

from __future__ import annotations

from typing import List, Optional

import torch

from sglang.kernels.ops.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType


class NgramVerifyInput(SpecInput):
    def __init__(
        self,
        draft_token: torch.Tensor = None,
        custom_mask: torch.Tensor = None,
        positions: torch.Tensor = None,
        retrieve_index: torch.Tensor = None,
        retrieve_next_token: torch.Tensor = None,
        retrieve_next_sibling: torch.Tensor = None,
        draft_token_num: int = None,
        future_indices: Optional[torch.Tensor] = None,
        new_seq_lens: Optional[torch.Tensor] = None,
        accept_tokens: Optional[torch.Tensor] = None,
        accept_lens: Optional[torch.Tensor] = None,
    ):
        super().__init__(SpecInputType.NGRAM_VERIFY)
        self.draft_token = draft_token
        self.custom_mask = custom_mask
        self.positions = positions
        self.retrieve_index = retrieve_index
        self.retrieve_next_token = retrieve_next_token
        self.retrieve_next_sibling = retrieve_next_sibling
        self.draft_token_num = draft_token_num
        self.num_tokens_per_req = draft_token_num
        self.num_tokens_for_logprob_per_req = draft_token_num

        # V2 overlap worker 专用输入
        self.future_indices = future_indices
        self.new_seq_lens = new_seq_lens
        self.accept_tokens = accept_tokens
        self.accept_lens = accept_lens

        # 设备从任一已知 tensor 推断：非 overlap 模式下 custom_mask 一定存在，
        # overlap 模式下 custom_mask 可能为 None，此时退回用 new_seq_lens。
        self.device = (
            custom_mask.device if custom_mask is not None else new_seq_lens.device
        )

    @property
    def max_tree_depth(self) -> int:
        """草稿树的最大深度。

        中译：NGRAM 的树按「节点预算」而非「深度」来限制——语料库 BFS 只在
        节点数用完时停止，所以一次长匹配完全可能把 draft_token_num 个节点
        串成一条链（此时深度 = draft_token_num）。因此对这棵树来说
        `spec_steps`（EAGLE 的草稿步数）是没有意义的。

        示例：draft_token_num=16 时返回 16，即最坏情况下树退化为 16 层单链。
        """
        return self.draft_token_num

    @property
    def tree_topk(self) -> int:
        """每层的固定分支数；NGRAM 返回 -1 表示「不固定」。

        中译：不规则树——每层分支数完全由语料匹配结果决定。
        对比 EAGLE：EAGLE 每层严格展开 speculative_eagle_topk 个分支，
        所以那里返回的是一个正整数。
        """
        return -1

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        """为「树形验证」这一次目标模型前向构造注意力参数。

        验证阶段本质是一次 prefill：一次性喂入 bs*draft_token_num 个 query
        token（整棵草稿树），每个 query 都要 attend 到「历史 KV + 树内祖先」。

        参数（示例 bs=2, draft_token_num=4）：
            req_pool_indices: 各请求在 req_to_token_pool 中的槽位号，shape [bs]
                tensor([3, 7])
            paged_kernel_lens: 各请求**验证前**已有的 KV 长度，shape [bs]
                tensor([100, 50])
            paged_kernel_lens_sum: 上者之和，int
                150
            req_to_token: 请求槽位 → KV cache 页索引的映射表，
                shape [max_num_reqs, max_context_len]

        返回 (kv_indices, cum_kv_seq_len, qo_indptr, custom_mask)：
            kv_indices: 拍平的 KV 页索引，长度 150 + 4*2 = 158
            cum_kv_seq_len: KV 前缀和指针，tensor([0, 104, 158])
            qo_indptr: query 前缀和指针，tensor([0, 4, 8])
            custom_mask: 树形 mask，长度 632（见下方 mask_numel 推导）
        """
        bs = len(req_pool_indices)

        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=self.device)

        # 每个请求的 KV 长度都要加上本轮草稿树的节点数——因为这 4 个草稿 token
        # 自己也会写入 KV cache 并被后续节点 attend 到。
        # [100, 50] → [104, 54]
        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        # 前缀和形式的 KV 指针：[0, 104, 158]
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        # query 侧指针：每个请求恰好贡献 draft_token_num 个 query。
        # arange([0,1,2]) * 4 → [0, 4, 8]
        self.qo_indptr = (
            torch.arange(0, bs + 1, dtype=torch.int32, device=self.device)
            * self.draft_token_num
        )

        # 总 KV 页数 = 历史 150 + 本轮新增 4*2 = 158
        kv_indices = torch.empty(
            paged_kernel_lens_sum + self.draft_token_num * bs,
            dtype=torch.int32,
            device=self.device,
        )

        # 由 Triton kernel 按 req_to_token 表把各请求的页索引拍平写进 kv_indices。
        # 结果形如 [req3 的 104 个页号..., req7 的 54 个页号...]
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )

        # 中译：当 CUDA graph 把 batch size 补齐到超过实际请求数时，需要给
        # custom_mask 补上尾部填充。
        #
        # mask 的元素总数 = Σ_req (该请求的 KV 长度 × draft_token_num)
        #                = Σ_req ((历史长度 + draft_token_num) × draft_token_num)
        #                = 历史总长 × draft_token_num + draft_token_num² × bs
        # 代入示例：150*4 + 4²*2 = 600 + 32 = 632
        # 校验：req0 贡献 104*4=416，req1 贡献 54*4=216，合计 632 ✓
        mask_numel = (
            paged_kernel_lens_sum * self.draft_token_num
            + (self.draft_token_num**2) * bs
        )
        custom_mask = self.custom_mask
        # 补齐部分填 True（全可见）：这些位置对应 CUDA graph 的 padding 请求，
        # 计算结果会被丢弃，填什么都不影响正确性。
        if custom_mask.numel() < mask_numel:
            custom_mask = torch.cat(
                [
                    custom_mask,
                    torch.full(
                        (mask_numel - custom_mask.numel(),),
                        True,
                        dtype=torch.bool,
                        device=self.device,
                    ),
                ],
                dim=0,
            )

        return kv_indices, cum_kv_seq_len, self.qo_indptr, custom_mask

    def filter_batch(
        self,
        new_indices: torch.Tensor,
        new_indices_cpu: Optional[List[int]] = None,
    ):
        if self.future_indices is not None:
            self.future_indices = self.future_indices[new_indices]
            return

        if self.new_seq_lens is not None:
            self.new_seq_lens = self.new_seq_lens[new_indices]
        self.accept_tokens = self.accept_tokens.reshape(-1, self.draft_token_num)[
            new_indices, :
        ]
        self.accept_tokens = self.accept_tokens.flatten()
        self.accept_lens = self.accept_lens[new_indices]

    def merge_batch(self, spec_info: NgramVerifyInput):
        """新请求加入 batch 时，把另一个 NgramVerifyInput 的各 tensor 拼接到自己尾部。

        参数：
            spec_info: 待并入的另一批请求的验证输入。

        效果（self 有 2 个请求、spec_info 有 1 个，draft_token_num=4）：
            accept_lens:   [3, 1] ⊕ [2]        → [3, 1, 2]
            accept_tokens: 8 个元素 ⊕ 4 个元素 → 12 个元素
            new_seq_lens:  [102, 52] ⊕ [77]    → [102, 52, 77]

        注意 filter_batch 需要 reshape 而这里不用：拼接是在拍平的一维上做的，
        天然保持「按请求分段」的布局。
        """
        if self.future_indices is not None:
            assert spec_info.future_indices is not None
            self.future_indices = torch.cat(
                (self.future_indices, spec_info.future_indices), dim=0
            )
            return

        if self.new_seq_lens is not None:
            assert spec_info.new_seq_lens is not None
            self.new_seq_lens = torch.cat(
                (self.new_seq_lens, spec_info.new_seq_lens), dim=0
            )
        self.accept_tokens = torch.cat(
            (self.accept_tokens, spec_info.accept_tokens), dim=0
        )
        self.accept_lens = torch.cat((self.accept_lens, spec_info.accept_lens), dim=0)
