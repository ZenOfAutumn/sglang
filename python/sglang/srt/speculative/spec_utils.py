"""投机解码通用工具函数模块。

本模块提供 EAGLE / NGRAM / DFLASH 等投机解码算法共用的工具函数，包括：

    - KV cache 索引管理：draft_kv_indices_buffer_width / draft_kv_indices_used_len
    - stream 管理：record_stream_each / record_stream_for_v2_verify
    - topk 选择：select_top_k_tokens / _select_top_k_tokens_first / _select_top_k_tokens_later
    - 语法约束掩码：traverse_tree / generate_token_bitmask
    - token map 加载：load_token_map
    - KV cache 搬移：move_accept_tokens_to_target_kvcache
    - Mamba 状态管理：prepare_mamba_track_for_verify / commit_mamba_states_after_verify
    - 仿真工具：_sample_simulated_acc_len / generate_simulated_accept_index
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, List, Literal, Optional, Tuple

import torch
from huggingface_hub import snapshot_download

from sglang.kernels.ops.speculative.cache_locs import (
    align_evict_mask_to_page_size as align_evict_mask_to_page_size,
)
from sglang.kernels.ops.speculative.cache_locs import (
    assign_extend_cache_locs as assign_extend_cache_locs,
)
from sglang.kernels.ops.speculative.cache_locs import (
    filter_finished_cache_loc_kernel as filter_finished_cache_loc_kernel,
)
from sglang.kernels.ops.speculative.cache_locs import (
    generate_draft_decode_kv_indices as generate_draft_decode_kv_indices,
)
from sglang.kernels.ops.speculative.cache_locs import (
    get_src_tgt_cache_loc as get_src_tgt_cache_loc,
)
from sglang.kernels.ops.speculative.cache_locs import (
    get_target_cache_loc as get_target_cache_loc,
)
from sglang.kernels.ops.speculative.eagle import (
    fill_accept_out_cache_loc_func as fill_accept_out_cache_loc_func,
)
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.constrained.base_grammar_backend import GrammarMask
from sglang.srt.distributed.parallel_state import (
    GroupCoordinator,
    get_self_pp_group,
    patch_pipeline_parallel_group,
    patch_tensor_parallel_group,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import set_mamba_track_indices_from_reqs
from sglang.srt.managers.utils import _async_d2h
from sglang.srt.mem_cache.allocation import (
    assign_req_to_token_pool as assign_req_to_token_pool,
)
from sglang.srt.mem_cache.allocation import (
    assign_req_to_token_pool_func as assign_req_to_token_pool_func,
)
from sglang.srt.runtime_context import (
    get_spec,
    mamba_extra_buffer_enabled,
    mamba_extra_buffer_lazy_enabled,
    mamba_track_grid,
    max_speculative_num_draft_tokens,
)
from sglang.srt.utils import (
    is_cpu,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    is_xpu,
    next_power_of_2,
)
from sglang.srt.utils.async_probe import maybe_detect_oob
from sglang.srt.utils.nvtx_utils import profile_range

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_musa = is_musa()
_is_xpu = is_xpu()
_is_cpu = is_cpu()

if TYPE_CHECKING:
    from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


if _is_cuda:
    from sgl_kernel import fast_topk
elif _is_hip:
    from sgl_kernel import fast_topk
else:
    from sglang.srt.utils.common import fast_topk

if _is_cpu:
    from sgl_kernel import assign_extend_cache_locs_cpu


logger = logging.getLogger(__name__)


def resolve_num_tokens_per_req(
    *,
    phase: Literal["draft_decode", "draft_extend", "target_verify"],
    spec_algorithm=None,
    is_draft_worker: bool = False,
    num_draft_tokens: Optional[int] = None,
) -> int:
    """Single static derivation point for a spec phase's per-request token
    width (sizes capture shapes / buffers); the per-forward dynamic width
    lives on ``SpecInput.num_tokens_per_req``. Draft phases are
    EAGLE-family-only; "target_verify" is algorithm-generic via the hook.

    The widths come from the bags: adaptive spec captures each candidate step
    config with that config's leaves overridden, so the buffers being sized
    must follow the override rather than the startup values.
    """
    spec = get_spec()
    if phase == "draft_decode":
        return spec.speculative_eagle_topk
    if phase == "draft_extend":
        return spec.speculative_num_draft_tokens
    if phase == "target_verify":
        if num_draft_tokens is None:
            num_draft_tokens = spec.speculative_num_draft_tokens
        return spec_algorithm.get_num_tokens_per_req_for_target_verify(
            num_draft_tokens, is_draft_worker
        )
    raise ValueError(f"Unknown speculative phase: {phase}")


def fast_sample(probs: torch.Tensor, num_samples: int = 1):
    """Draw from `probs` via the Gumbel-max trick: argmax(probs / Exp(1)).

    Distributionally equivalent to torch.multinomial, but avoids multinomial's
    device-side distribution-validity assert, which the draft CUDA graph would
    otherwise capture and replay every step. q is clamped off zero so a zero
    draw can't yield inf/NaN scores that argmax would wrongly select; fp32
    avoids bf16 argmax ties biasing the draw. Set SGLANG_OPT_USE_GUMBEL_SAMPLE=0
    to fall back to torch.multinomial.
    """
    if not envs.SGLANG_OPT_USE_GUMBEL_SAMPLE.get():
        sample_index = torch.multinomial(probs, num_samples=num_samples)
        return probs.gather(1, sample_index), sample_index
    q = torch.empty_like(probs, dtype=torch.float32).exponential_(1.0)
    q.clamp_min_(torch.finfo(torch.float32).tiny)
    scores = probs.float() / q
    if num_samples == 1:
        sample_index = scores.argmax(dim=-1, keepdim=True)
    else:
        sample_index = scores.topk(num_samples, dim=-1).indices
    sample_p = probs.gather(1, sample_index)
    return sample_p, sample_index


def renorm_draft_probs(
    next_token_logits: torch.Tensor,
    sampling_info,
    use_rejection_sampling: bool,
) -> torch.Tensor:
    """Draft-side next-token distribution.

    Plain softmax, except under rejection sampling where logits are
    temperature-scaled so the draft proposal q tracks the target sampling
    temperature (higher acceptance; correctness holds for any q).
    """
    if not use_rejection_sampling or not next_token_logits.size(0):
        return torch.softmax(next_token_logits, dim=-1)
    return torch.softmax(next_token_logits / sampling_info.temperatures, dim=-1)


def sample_draft_proposal(next_token_logits: torch.Tensor, temperatures: torch.Tensor):
    """Leviathan draft proposal: q = softmax(logits / T), X ~ q.

    Returns (q, q(X), X). The verify's accept test coin*q(X) < p(X) is unbiased
    only if q is exactly the distribution X was drawn from, so callers must hand
    the returned q (not a recomputed one) to the verify.
    """
    probs = torch.softmax(next_token_logits / temperatures, dim=-1)
    topk_p, topk_index = fast_sample(probs, num_samples=1)
    return probs, topk_p, topk_index


# Simulate acceptance length for benchmarking purposes
SIMULATE_ACC_LEN = envs.SGLANG_SIMULATE_ACC_LEN.get()  # turn off if < 0
SIMULATE_ACC_METHOD = envs.SGLANG_SIMULATE_ACC_METHOD.get()
SIMULATE_ACC_TOKEN_MODE = envs.SGLANG_SIMULATE_ACC_TOKEN_MODE.get()

TREE_TRAVERSE_TIME_THRESHOLD = 1  # TODO: 合理设置此阈值
TREE_SPEC_KERNEL_AVAILABLE = (
    _is_cuda or _is_musa
)  # 此 kernel 目前仅支持 CUDA 和 MUSA


def draft_kv_indices_buffer_width(
    num_seqs: int, topk: int, max_context_len: int
) -> int:
    """计算 EAGLE draft-decode kv_indices 缓冲区的每步行宽。

    每个序列的 topk 个分支各自最多关注 max_context_len 个 KV 槽位；
    topk 因子是必须的——省略会导致分配不足和行溢出 (#27338, #27460)。

    参数：
        num_seqs:       序列数
        topk:           每步每请求的候选数
        max_context_len: 最大上下文长度

    返回：缓冲区行宽 = num_seqs * topk * max_context_len

    示例：
        >>> draft_kv_indices_buffer_width(num_seqs=4, topk=8, max_context_len=4096)
        131072
    """
    assert (
        num_seqs * topk * max_context_len < 2**31
    ), "kv_indices 平坦偏移会溢出 int32；请减小 batch/topk/context"
    return num_seqs * topk * max_context_len


def draft_kv_indices_used_len(
    seq_lens_sum: int, topk: int, bs: int, num_steps: int
) -> int:
    """计算 num_steps 步 draft-decode 所使用的 kv_indices 长度。

    bs = topk * num_seqs 个分支，每步每分支追加一个索引。
    调用时 num_steps = i + 1（每步切片）或 speculative_num_steps（容量断言）。

    参数：
        seq_lens_sum: 所有序列长度之和
        topk:         每步每请求的候选数
        bs:           分支数 = topk * num_seqs
        num_steps:    草稿步数

    返回：kv_indices 使用长度 = seq_lens_sum * topk + bs * num_steps

    示例：
        >>> draft_kv_indices_used_len(seq_lens_sum=128, topk=8, bs=32, num_steps=3)
        1216
    """
    return seq_lens_sum * topk + bs * num_steps


def record_stream_each(tensors, stream):
    """对 tensors 中每个 CUDA tensor 调用 record_stream(stream)。

    跳过非 tensor 或非 CUDA 的条目。通知缓存分配器这些 tensor 也在
    `stream` 上使用，避免在 Python 引用释放后、stream 上的排队工作
    仍在执行时回收内存。

    参数：
        tensors: tensor 列表（可含非 tensor）
        stream:  CUDA stream

    示例：
        >>> record_stream_each([tensor_a, tensor_b, None], cuda_stream)
        # tensor_a 和 tensor_b 被标记为在 cuda_stream 上使用
    """
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.is_cuda:
            t.record_stream(stream)


def record_stream_for_v2_verify(batch, verify_input, fwd_stream):
    """将 prepare 前的 ScheduleBatch / verify_input GPU tensor 标记为在 fwd_stream 上使用。

    Spec V2 在前向过程中修改 ScheduleBatch（`prepare_for_verify` 重新绑定
    `batch.input_ids` / `out_cache_loc`；`_draft_extend_for_decode` 之后再次
    替换 `batch.input_ids`）。每次重绑定都丢弃了 ScheduleBatch 对旧 tensor
    的唯一 Python 引用，而验证前向 kernel 可能仍在 `fwd_stream` 上读取其内存；
    `record_stream` 通知缓存分配器在回收内存块前等待 `fwd_stream`。

    参数：
        batch:        当前调度 batch
        verify_input: EagleVerifyInput（可为 None）
        fwd_stream:   前向计算 stream

    注意：仅覆盖 prepare 前的 tensor；调用方还需对 prepare 后的重绑定
    （新的 `batch.input_ids` / `out_cache_loc`）调用 `record_stream_each`。
    """
    candidates = [
        batch.seq_lens,
        batch.req_pool_indices,
        batch.input_ids,
        batch.out_cache_loc,
    ]
    if verify_input is not None:
        candidates.extend(
            [
                getattr(verify_input, attr, None)
                for attr in (
                    "draft_token",
                    "custom_mask",
                    "positions",
                    "retrieve_index",
                    "retrieve_next_token",
                    "retrieve_next_sibling",
                )
            ]
        )
    record_stream_each(candidates, fwd_stream)


def spec_need_hidden_states() -> bool:
    # STANDALONE drafts don't consume `spec_info.hidden_states` (vanilla LLM).
    # multi_layer_eagle, DFLASH, and DSPARK don't relay hidden_states through FutureMap.
    # TODO(lsyin): also skip when step == 1.
    spec = get_spec()
    if spec.speculative_algorithm in ("STANDALONE", "DFLASH", "DSPARK"):
        return False
    return not spec.enable_multi_layer_eagle


@torch.compile(dynamic=True, disable=_is_npu or _is_xpu)
def create_num_accept_tokens_filter(
    num_correct_drafts: torch.Tensor,
    unfinished_index_device: torch.Tensor,
    seq_lens: torch.Tensor,
):
    """创建接受 token 数的过滤器并更新序列长度。

    参数：
        num_correct_drafts:       每请求正确草稿数 [bs]
        unfinished_index_device:  未完成请求的索引
        seq_lens:                 序列长度（会被原地修改）

    返回：num_accept_tokens_filter，未完成请求的接受数 = correct + 1

    示例：
        >>> num_correct = torch.tensor([2, 0, 3])
        >>> unfinished = torch.tensor([0, 2])
        >>> seq_lens = torch.tensor([10, 20, 30])
        >>> filter = create_num_accept_tokens_filter(num_correct, unfinished, seq_lens)
        >>> # filter = tensor([3, 0, 4])
        >>> # seq_lens = tensor([13, 21, 34])
    """
    num_accept_tokens_filter = torch.zeros_like(num_correct_drafts)
    num_accept_tokens_filter[unfinished_index_device] = (
        num_correct_drafts[unfinished_index_device] + 1
    )
    seq_lens.add_(num_correct_drafts + 1)
    return num_accept_tokens_filter


def _select_top_k_tokens_first(
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: Optional[torch.Tensor],
    topk: int,
):
    """草稿第 0 步的 topk 选择：展开首层 topk 个候选。

    参数：
        topk_p:       概率 [bs, topk]
        topk_index:   token id [bs, topk]
        hidden_states: 隐藏状态 [bs, hidden_dim]（可选）
        topk:         候选数

    返回 (input_ids, hidden_states, scores, tree_info)：
        input_ids:     展平的 token id [bs * topk]
        hidden_states: repeat_interleave 后的隐藏状态 [bs * topk, hidden_dim]
        scores:        原始 topk_p
        tree_info:     (scores, token_index, parent_index) 元组
    """
    input_ids = topk_index.flatten()
    if hidden_states is not None:
        hidden_states = hidden_states.repeat_interleave(topk, dim=0)

    tree_info = (
        topk_p.unsqueeze(1),  # (b, 1, topk)
        topk_index,  # (b, topk)
        torch.arange(-1, topk, dtype=torch.long, device=input_ids.device).expand(
            topk_p.shape[0], -1
        ),  # (b, topk + 1) — expand 避免 repeat 的内存分配
    )
    return input_ids, hidden_states, topk_p, tree_info


@torch.compile(dynamic=True, disable=_is_npu or _is_xpu)
def _select_top_k_tokens_later(
    i: int,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
):
    """草稿第 i 步（i > 0）的 topk 选择：从上一步的 topk² 个候选中选 topk 个。

    参数：
        i:            当前步数
        topk_p:       本步概率 [bs * topk, topk]
        topk_index:   本步 token id [bs * topk, topk]
        hidden_states: 上一步隐藏状态 [bs * topk, hidden_dim]
        scores:       上一步的累计概率 [bs * topk]
        topk:         候选数

    返回 (input_ids, hidden_states, scores, tree_info)：
        input_ids:     选中的 token id [bs * topk]
        hidden_states: 选中的隐藏状态 [bs * topk, hidden_dim]
        scores:        新的累计概率
        tree_info:     (expand_scores, topk_index, parent_index)
    """
    topk_sq = topk * topk

    expand_scores = scores.unsqueeze(2) * topk_p.view(-1, topk, topk)
    # (b, topk, 1) * (b, topk, topk) -> (b, topk, topk)

    topk_cs_p, topk_cs_index = fast_topk(
        expand_scores.flatten(start_dim=1), topk, dim=-1
    )  # (b, topk)

    topk_index = topk_index.view(-1, topk_sq)
    input_ids = torch.gather(topk_index, 1, topk_cs_index).flatten()

    if hidden_states is not None and hidden_states.shape[0] > 0:
        flat_cs = topk_cs_index.flatten()
        batch_offsets = torch.arange(
            0, hidden_states.shape[0], step=topk, device=flat_cs.device
        )
        selected_input_index = flat_cs // topk + batch_offsets.repeat_interleave(topk)
        hidden_states = hidden_states[selected_input_index]

    tree_info = (
        expand_scores,  # (b, topk, topk)
        topk_index,  # (b, topk * topk)
        topk_cs_index + (topk_sq * (i - 1) + topk),  # (b, topk)
    )
    return input_ids, hidden_states, topk_cs_p, tree_info


def select_top_k_tokens(
    i: int,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
):
    """根据步数 i 选择 topk token。

    i == 0 时调用 _select_top_k_tokens_first（首层展开），
    i > 0 时调用 _select_top_k_tokens_later（后续步的 topk² → topk 筛选）。

    参数：
        i:            当前步数（从 0 开始）
        topk_p:       概率
        topk_index:   token id
        hidden_states: 隐藏状态
        scores:       累计概率（i=0 时未使用）
        topk:         候选数

    返回：同 _select_top_k_tokens_first / _select_top_k_tokens_later
    """
    if i == 0:
        return _select_top_k_tokens_first(topk_p, topk_index, hidden_states, topk)
    return _select_top_k_tokens_later(
        i, topk_p, topk_index, hidden_states, scores, topk
    )


def sample_simulated_acc_len(
    simulate_acc_len: float,
    simulate_acc_method: str,
    max_len: int,
) -> int:
    """在 [1, max_len] 范围内采样仿真接受长度。

    参数：
        simulate_acc_len:    期望接受长度
        simulate_acc_method: 采样方法 ("multinomial" 或 "match-expected")
        max_len:             最大接受长度

    返回：采样后的接受长度（整数）
    """
    if simulate_acc_method == "multinomial":
        simulated_values = torch.normal(
            mean=simulate_acc_len,
            std=1.0,
            size=(1,),
            device="cpu",
        )
        # 将仿真值 clamp 到 [1, max_len]
        simulated_values = torch.clamp(simulated_values, min=1.0, max=max_len)
        simulate_acc_len = int(simulated_values.round().item())
    elif simulate_acc_method == "match-expected":
        # 多项式采样不匹配期望长度
        # 保留它为了兼容现有测试
        # 但需要匹配期望长度的场景最好用 "match-expected"
        # 一个注意点是这只会采样期望长度的向下或向上取整
        simulate_acc_len = max(1.0, min(max_len, simulate_acc_len))
        lower = int(simulate_acc_len // 1)
        upper = lower + 1 if lower < max_len else lower
        if lower == upper:
            simulate_acc_len = lower
        else:
            weight_upper = simulate_acc_len - lower
            weight_lower = 1.0 - weight_upper
            probs = torch.tensor([weight_lower, weight_upper], device="cpu")
            sampled_index = torch.multinomial(probs, num_samples=1)
            simulate_acc_len = lower if sampled_index == 0 else upper
    else:
        raise ValueError(f"无效的 simulate_acc_method: {simulate_acc_method}")
    return int(simulate_acc_len)


def generate_simulated_accept_index(
    accept_index,
    predict,
    num_correct_drafts,
    candidates,
    target_predict,
    bs,
    spec_steps,
    simulate_acc_len: float = SIMULATE_ACC_LEN,
    simulate_acc_method: str = SIMULATE_ACC_METHOD,
    simulate_acc_token_mode: str = SIMULATE_ACC_TOKEN_MODE,
):
    use_real_draft_tokens = simulate_acc_token_mode == "real-draft-token"

    assert simulate_acc_len > 0.0
    simulate_acc_len = sample_simulated_acc_len(
        simulate_acc_len, simulate_acc_method, spec_steps + 1
    )

    accept_indx_first_col = accept_index[:, 0].view(-1, 1)
    sim_accept_index = torch.full(
        (bs, spec_steps + 1), -1, dtype=torch.int32, device=accept_index.device
    )
    sim_accept_index[:, :simulate_acc_len] = accept_indx_first_col + torch.arange(
        simulate_acc_len, device=accept_index.device
    )
    num_correct_drafts.fill_(simulate_acc_len - 1)

    if not use_real_draft_tokens:
        predict.fill_(100)  # some legit token id
        return sim_accept_index

    # Use the topk=1 draft chain for forced acceptance, then a target-derived bonus.
    if simulate_acc_len > 1:
        draft_node_indices = sim_accept_index[:, : simulate_acc_len - 1].long()
        predict[draft_node_indices] = candidates[:, 1:simulate_acc_len].to(
            dtype=predict.dtype
        )
    bonus_node_indices = sim_accept_index[:, simulate_acc_len - 1].long()
    predict[bonus_node_indices] = target_predict[:, simulate_acc_len - 1].to(
        dtype=predict.dtype
    )
    return sim_accept_index


def traverse_tree(
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    draft_tokens: torch.Tensor,
    grammar: BaseGrammarObject,
    allocate_token_bitmask: torch.Tensor,
    vocab_size: Optional[int] = None,
):
    """遍历草稿模型构建的树，生成 logit mask。

    对树做 DFS：对于每个节点，检查其 token 是否被语法接受。
    如果接受，生成对应的 vocab mask，然后递归访问子节点和兄弟节点。

    参数：
        retrieve_next_token:    子节点索引数组
        retrieve_next_sibling:  兄弟节点索引数组
        draft_tokens:           草稿 token id 数组
        grammar:                语法约束对象
        allocate_token_bitmask: vocab mask 输出缓冲区
        vocab_size:             词表大小（可选）

    示例：
        >>> # 假设树有 5 个节点，节点 0 是根
        >>> traverse_tree(retrieve_next_token, retrieve_next_sibling,
        ...               draft_tokens, grammar, bitmask, vocab_size=32000)
        # 遍历后 bitmask 中包含了每个被接受节点的 vocab mask
    """
    assert (
        retrieve_next_token.shape == retrieve_next_sibling.shape == draft_tokens.shape
    )

    def dfs(
        curr: int,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        parent_pos: int,
    ):
        if curr == 0:
            # 第一个 token 由目标模型生成，因此总是从上一轮接受
            is_accepted = True
        else:
            parent_bitmask = allocate_token_bitmask[parent_pos]
            current_token = draft_tokens[curr]
            if vocab_size and current_token >= vocab_size:
                is_accepted = False
            else:
                # 32 个布尔 mask 值被打包到 32 位整数中
                is_accepted = (
                    parent_bitmask[current_token // 32] & (1 << (current_token % 32))
                ) != 0

        if is_accepted:
            if curr != 0:
                # 接受当前 token
                grammar.accept_token(int(draft_tokens[curr]))
            if not grammar.is_terminated():
                # 为当前 token 生成 bitmask
                grammar.fill_vocab_mask(allocate_token_bitmask, curr)
                if retrieve_next_token[curr] != -1:
                    # 访问子节点
                    dfs(
                        int(retrieve_next_token[curr]),
                        retrieve_next_token,
                        retrieve_next_sibling,
                        curr,
                    )

            if curr != 0:
                # 回滚当前 token
                grammar.rollback(1)

        if retrieve_next_sibling[curr] != -1:
            # 访问兄弟节点
            dfs(
                int(retrieve_next_sibling[curr]),
                retrieve_next_token,
                retrieve_next_sibling,
                parent_pos,
            )

    dfs(0, retrieve_next_token, retrieve_next_sibling, -1)


def generate_token_bitmask(
    reqs: List[Req],
    retrieve_next_token_cpu: torch.Tensor,
    retrieve_next_sibling_cpu: torch.Tensor,
    draft_tokens_cpu: torch.Tensor,
    vocab_size: int,
) -> Tuple[Optional[torch.Tensor], Optional[BaseGrammarObject]]:
    """
    Generate the logit mask for structured output.
    Draft model's token can be either valid or invalid with respect to the grammar.
    We need to perform DFS to
    1. figure out which tokens are accepted by the grammar.
    2. if so, what is the corresponding logit mask.
    """

    num_draft_tokens = draft_tokens_cpu.shape[-1]

    allocate_token_bitmask = None
    assert len(reqs) == retrieve_next_token_cpu.shape[0]
    grammar = None
    for i, req in enumerate(reqs):
        if req.grammar is not None:
            if allocate_token_bitmask is None:
                allocate_token_bitmask = req.grammar.allocate_vocab_mask(
                    vocab_size=vocab_size,
                    batch_size=draft_tokens_cpu.numel(),
                    device="cpu",
                )
            grammar = req.grammar
            s = time.perf_counter()
            traverse_tree(
                retrieve_next_token_cpu[i],
                retrieve_next_sibling_cpu[i],
                draft_tokens_cpu[i],
                req.grammar,
                allocate_token_bitmask[
                    i * num_draft_tokens : (i + 1) * num_draft_tokens
                ],
                vocab_size=vocab_size,
            )
            tree_traverse_time = time.perf_counter() - s
            if tree_traverse_time > TREE_TRAVERSE_TIME_THRESHOLD:
                logger.warning(
                    f"Bit mask 生成耗时 {tree_traverse_time} 秒，"
                    f"语法: {req.grammar}"
                )

    return allocate_token_bitmask, grammar


class GrammarTree:
    """The verify tree the grammar bitmask is built over, on the host.

    ``from_device`` starts an async copy, so build it before the target verify
    launch; ``from_host`` is for algorithms that build the tree there (NGRAM).
    """

    def __init__(self, host: Tuple[torch.Tensor, ...], done_event):
        self._host = host
        self._done = done_event

    @classmethod
    def from_device(
        cls,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        draft_token: torch.Tensor,
    ) -> GrammarTree:
        tensors = (retrieve_next_token, retrieve_next_sibling, draft_token)
        host = tuple(_async_d2h(t) for t in tensors)
        # Sources may be mixed -- an algorithm can synthesize part of the tree on
        # the host -- so the event has to key off whichever one is on device.
        device = next((t.device for t in tensors if t.device.type != "cpu"), None)
        if device is None:
            return cls(host, None)
        done = torch.get_device_module(device).Event()
        done.record()
        return cls(host, done)

    @classmethod
    def from_host(
        cls,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        draft_token: torch.Tensor,
    ) -> GrammarTree:
        return cls((retrieve_next_token, retrieve_next_sibling, draft_token), None)

    @classmethod
    def from_linear_chain(cls, verify_ids_2d: torch.Tensor) -> GrammarTree:
        """Degenerate tree for chain-verify algorithms: node i's only child is i + 1.

        ``verify_ids_2d`` is (bs, chain_len) with column 0 the already-committed
        token, so mask rows line up with the target's logits rows one-for-one.
        Only the ids need a copy; the links are fixed by the shape.
        """
        bs, chain_len = verify_ids_2d.shape
        next_token = torch.full((bs, chain_len), -1, dtype=torch.int64)
        next_token[:, :-1] = torch.arange(1, chain_len, dtype=torch.int64)
        next_sibling = torch.full((bs, chain_len), -1, dtype=torch.int64)
        return cls.from_device(next_token, next_sibling, verify_ids_2d)

    def resolve(self) -> Tuple[torch.Tensor, ...]:
        if self._done is not None:
            self._done.synchronize()
        return self._host


def build_grammar_vocab_mask(
    *,
    reqs: List[Req],
    tree: GrammarTree,
    sampling_info: SamplingBatchInfo,
    device,
    barrier: Optional[Callable[[], None]],
) -> Optional[GrammarMask]:
    """Build the constrained-decoding bitmask over a verify tree and stage it on device.

    Call it after the target verify launch -- every step here is host work, so it all
    overlaps that forward. ``barrier`` advances the previous batch's FSM over its
    committed tokens, which the traversal then reads, so it has to run first.
    """
    if barrier is not None:
        barrier()
    vocab_mask, grammar = generate_token_bitmask(
        reqs,
        *tree.resolve(),
        sampling_info.vocab_size,
    )
    if vocab_mask is None:
        return None

    # non_blocking is safe: the bitmask is pinned (see xgrammar_backend), and stream
    # order keeps the copy ahead of the sampler's apply_vocab_mask.
    vocab_mask = vocab_mask.to(device, non_blocking=True)
    # Otherwise the extend stage's leftover mask is applied instead.
    sampling_info.grammar_mask = None
    return GrammarMask(grammar, vocab_mask)


def load_token_map(token_map_path: str) -> List[int]:
    """加载热门 token 映射表（用于缩减草稿模型词表）。

    参数：
        token_map_path: token map 文件路径，可以是本地路径或 HuggingFace repo 路径

    返回：热门 token id 列表

    示例：
        >>> hot_token_id = load_token_map("model/hot_token_map.pt")
        >>> # hot_token_id = tensor([123, 456, 789, ...])
    """
    if not os.path.exists(token_map_path):
        repo_id = os.path.dirname(token_map_path)
        file_name = os.path.basename(token_map_path)

        cache_dir = None
        if envs.SGLANG_USE_MODELSCOPE.get():
            from modelscope.utils.file_utils import get_model_cache_root

            cached_repo_path = os.path.join(get_model_cache_root(), repo_id)
            if os.path.exists(cached_repo_path):
                cache_dir = cached_repo_path

        if cache_dir is None:
            if envs.SGLANG_USE_MODELSCOPE.get():
                from modelscope.hub.snapshot_download import (
                    snapshot_download as download_func,
                )
            else:
                download_func = snapshot_download
            cache_dir = download_func(
                repo_id,
                ignore_patterns=["*.bin", "*.safetensors"],
            )

        token_map_path = os.path.join(cache_dir, file_name)
    hot_token_id = torch.load(token_map_path, weights_only=True)
    return torch.tensor(hot_token_id, dtype=torch.int64)


@contextmanager
def draft_pp_context():
    # The draft model is one layer and never spans pipeline stages; give it a
    # single-member pp group so it initializes as if pp were off.
    with patch_pipeline_parallel_group(get_self_pp_group()):
        yield


@contextmanager
def draft_tp_context(tp_group: GroupCoordinator):
    """草稿模型的 TP 上下文管理器。

    草稿模型不使用 DP，有独立的 TP 组。
    当前禁用 mscclpp，因为它不支持两个通信组。

    参数：
        tp_group: 草稿模型的 TP 组

    示例：
        >>> with draft_tp_context(draft_tp_group):
        ...     # 在此作用域内，tensor parallel 操作使用草稿模型的 TP 组
        ...     draft_model.forward()
    """
    with patch_tensor_parallel_group(tp_group):
        yield


def spec_stage_span(name: str):
    """Profiler span：标记投机解码的粗粒度阶段。

    参数：
        name: 阶段名称（"draft" / "draft_extend" / "verify"）

    返回：profile_range 上下文管理器

    示例：
        >>> with spec_stage_span("draft"):
        ...     draft_worker.draft(batch)
    """
    return profile_range(name)


def move_accept_tokens_to_target_kvcache(
    batch: ScheduleBatch,
    accept_index: torch.Tensor,
    num_correct_drafts: torch.Tensor,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
):
    """将接受 token（草稿 + bonus）的 KV cache 移到目标模型的 KV cache。

    参数：
        batch: 当前调度 batch
        accept_index: 接受 token 的索引（含 bonus）[bs, spec_steps + 1]
        num_correct_drafts: 每请求正确草稿数（不含 bonus）；
            seq_lens 前进 num_correct_drafts + 1 以覆盖 bonus 槽位
        token_to_kv_pool_allocator: KV cache 分配器

    示例：
        >>> # accept_index = [[5, 6, 7, -1], [10, 11, -1, -1]]
        >>> # num_correct_drafts = [2, 1]
        >>> # 将 accept_index 对应的 KV 从草稿 cache 移到目标 cache
        >>> move_accept_tokens_to_target_kvcache(batch, accept_index,
        ...     num_correct_drafts, allocator)
    """
    bs = len(batch.seq_lens)
    device = batch.seq_lens.device
    # accept_index 元素数，不是 bs * num_draft_tokens：对 topk > 1，
    # 树超过接受链，会过度读取 accept_index（非法内存访问）。
    size = bs * accept_index.shape[1]

    # fill_accept_out_cache_loc 读取 out_cache_loc[accept_index]；-1 哨兵值可行。
    maybe_detect_oob(
        accept_index,
        -1,
        batch.out_cache_loc.size(0),
        "spec v2 move_accept_tokens accept_index",
    )

    tgt_cache_loc = torch.zeros(
        size,
        dtype=torch.int64,
        device=device,
    )
    accept_out_cache_loc = torch.zeros(size, dtype=torch.int64, device=device)
    if _is_cpu:
        assign_extend_cache_locs_cpu(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + num_correct_drafts + 1,
            tgt_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
        )
    else:
        assign_extend_cache_locs[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + num_correct_drafts + 1,
            tgt_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            next_power_of_2(bs),
        )
    fill_accept_out_cache_loc_func(
        accept_index,
        batch.out_cache_loc,
        accept_out_cache_loc,
        size,
    )
    token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
        tgt_cache_loc, accept_out_cache_loc
    )


def prepare_mamba_track_for_verify(batch: ScheduleBatch) -> None:
    """在 TARGET_VERIFY 前从请求重建 mamba track 索引。

    Spec batches skip the refresh in prepare_for_decode, and filter/merge
    null these fields, so they must be rebuilt right before verify. Clearing
    the mask also keeps a stale extend-time mask from triggering in-forward
    tracking during TARGET_VERIFY; tracking is done in
    commit_mamba_states_after_verify instead.

    Lazy: gather the positions planned by mamba_lazy_spec_prepare. Runs
    inside forward isolation, so it must not mutate req/pool state.
    """
    if not mamba_extra_buffer_enabled():
        return
    track_positions = None
    if mamba_extra_buffer_lazy_enabled():
        track_positions = batch.mamba_lazy_spec_track_positions_cpu
        assert track_positions is not None and len(track_positions) == len(
            batch.reqs
        ), (
            "lazy spec verify without a track plan: mamba_lazy_spec_prepare "
            "must run in prepare_for_decode for every spec decode iteration"
        )
    set_mamba_track_indices_from_reqs(batch, track_positions)
    batch.mamba_track_mask = None
    batch.mamba_track_seqlens = None


def _verify_commit_step_indices(
    *,
    batch: ScheduleBatch,
    accept_index: torch.Tensor,
    accept_lens: torch.Tensor,
    draft_token_num: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Step indices for a post-verify state commit: per req, the tree step of
    the last accepted node (reduces to accept_lens - 1 for topk == 1), and the
    mamba-track interval-crossing step (-1 = no crossing; None when tracking
    is off)."""
    bs = accept_lens.shape[0]
    accept_indices_offset = torch.arange(
        0,
        bs * draft_token_num,
        step=draft_token_num,
        dtype=accept_lens.dtype,
        device=accept_lens.device,
    )
    req_idx = torch.arange(bs, dtype=torch.int64, device=accept_lens.device)
    last_correct_step_indices = (
        accept_index[req_idx, (accept_lens - 1).to(torch.int64)] - accept_indices_offset
    )
    if batch.mamba_track_indices is None:
        return last_correct_step_indices, None
    seq_lens_pre_verify = batch.seq_lens
    seq_lens_post_verify = batch.seq_lens + accept_lens
    mamba_track_interval = mamba_track_grid(batch.tree_cache.page_size)
    to_track_mask = (
        seq_lens_pre_verify // mamba_track_interval
        != seq_lens_post_verify // mamba_track_interval
    )
    tracking_point = seq_lens_post_verify // mamba_track_interval * mamba_track_interval
    to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0).to(
        torch.int64
    )
    candidate_track_steps = accept_index[req_idx, to_track_ith] - accept_indices_offset
    mamba_steps_to_track = torch.where(
        to_track_mask,
        candidate_track_steps,
        torch.full_like(candidate_track_steps, -1),
    )
    return last_correct_step_indices, mamba_steps_to_track


def commit_mamba_states_after_verify(
    target_worker: TpModelWorker,
    batch: ScheduleBatch,
    accept_lens: torch.Tensor,
    accept_index: torch.Tensor,
    draft_token_num: int,
) -> None:
    """将接受的 per-step mamba 状态提交到持久化缓存。

    在 TARGET_VERIFY 期间，混合线性注意力后端将 per-step 状态保存在
    中间缓存中，而不是推进持久化 conv/ssm 缓存。接受后，每个请求最后
    一个接受步的状态被提交回去，加上用于前缀缓存跟踪的区间交叉状态
    （mamba extra_buffer 模式）。

    对没有 mamba 式状态的后端或没有 commit 钩子的后端是空操作。

    参数：
        target_worker:     目标模型 worker
        batch:             当前调度 batch
        accept_lens:       每请求接受长度（含 bonus）
        accept_index:      接受路径索引
        draft_token_num:   草稿 token 总数
    """
    model_runner = target_worker.model_runner
    if mambaish_config(model_runner.model_config) is None:
        return

    # ReplaySSM spec-verify path (Part B of #28511): the accepted drafts already
    # live in the per-slot circular ring (written during verify). Instead of
    # scattering an intermediate full SSM state into `temporal`, advance the
    # block-keyed cursors by the accepted count (the ring owns the SSM state; the
    # verify/flush kernel folds it into `temporal` periodically). The CONV state
    # still needs its usual accept-rollback, so we keep the conv-window scatter and
    # skip only the SSM scatter. GDN-only + linear-chain (topk<=1) -- the runtime
    # ring is allocated only then; KDA never allocates the cursors.
    req_pool = model_runner.req_to_token_pool
    mamba_pool = getattr(req_pool, "mamba_pool", None)

    # Fold-every-commit: replay the accepted prefix from the ring into
    # `temporal`; the same fold stores the interval-crossing state to the
    # track slot, so no SSM scatter or force-flush is needed here.
    if (
        mamba_pool is not None
        and getattr(mamba_pool, "replayssm_spec_fold", False)
        and not getattr(mamba_pool, "replayssm_is_kda", False)
    ):
        if batch.forward_mode.is_idle() or accept_index.numel() == 0:
            return
        from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold import (
            commit_gdn_replayssm_fold_after_verify,
        )

        spec_state = req_pool.get_speculative_mamba2_params_all_layers()
        state_batch_indices = req_pool.get_mamba_indices(batch.req_pool_indices)
        last_correct_step_indices, mamba_steps_to_track = _verify_commit_step_indices(
            batch=batch,
            accept_index=accept_index,
            accept_lens=accept_lens,
            draft_token_num=draft_token_num,
        )
        commit_gdn_replayssm_fold_after_verify(
            spec_state=spec_state,
            state_batch_indices=state_batch_indices,
            accept_lens=accept_lens,
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            null_block_id=-1,
        )
        return

    if (
        mamba_pool is not None
        and getattr(mamba_pool, "replayssm_cache_base", None) is not None
        and not getattr(mamba_pool, "replayssm_is_kda", False)
    ):
        if batch.forward_mode.is_idle() or accept_index.numel() == 0:
            return
        from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
            commit_gdn_replayssm_spec,
        )
        from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
            fused_conv_window_scatter_with_mask,
        )

        spec_state = req_pool.get_speculative_mamba2_params_all_layers()
        bs = accept_lens.shape[0]
        state_batch_indices = req_pool.get_mamba_indices(batch.req_pool_indices)
        # Advance the per-slot circular cursors by the accepted count (incl. the
        # bonus token). max_cache_len = ring length L = replayssm_d.shape[-2].
        commit_gdn_replayssm_spec(
            write_pos=mamba_pool.replayssm_write_pos,
            cache_base=mamba_pool.replayssm_cache_base,
            is_flush=mamba_pool.replayssm_is_flush,
            num_accepted=accept_lens,  # [bs], includes the bonus token
            state_batch_indices=state_batch_indices,
            max_cache_len=spec_state.replayssm_d.shape[-2],
            max_spec_len=draft_token_num,
            null_block_id=-1,  # SGLang: valid slots >= 0, padding == -1
        )
        # Roll back / commit the conv state to the last accepted draft step
        # (same logic as the recurrent commit, but conv-only).
        last_correct_step_indices, _ = _verify_commit_step_indices(
            batch=batch,
            accept_index=accept_index,
            accept_lens=accept_lens,
            draft_token_num=draft_token_num,
        )
        fused_conv_window_scatter_with_mask(
            spec_state.conv[0],
            spec_state.intermediate_conv_window[0],
            state_batch_indices,
            last_correct_step_indices,
        )
        # NOTE: radix mamba prefix-caching (mamba_track / extra_buffer) would need
        # a device-side force-flush so `temporal` reflects the ring before a
        # snapshot; not wired for Part B (server_args forbids extra_buffer with
        # --enable-linear-replayssm-spec), so the per-track scatters are intentionally
        # skipped here.
        return

    # KDA ReplaySSM (fold-every-commit): KDA keeps its own recurrent verify kernel
    # for the OUTPUT, so we replay the accepted window into the fp32 checkpoint
    # (`temporal`) here on commit -- `temporal` is always the current committed
    # state. The draft window's raw inputs were written to the ring during verify
    # by the KDA backend. Gate on the fold flag + is_kda (the cursor tensors are
    # never allocated under fold, so they cannot serve as the signal).
    if (
        mamba_pool is not None
        and getattr(mamba_pool, "replayssm_spec_fold", False)
        and getattr(mamba_pool, "replayssm_is_kda", False)
    ):
        if batch.forward_mode.is_idle() or accept_index.numel() == 0:
            return
        from sglang.kernels.ops.attention.fla.kda_replayssm_spec_decode import (
            commit_kda_replayssm_after_verify,
        )

        spec_state = req_pool.get_speculative_mamba2_params_all_layers()
        bs = accept_lens.shape[0]
        state_batch_indices = req_pool.get_mamba_indices(batch.req_pool_indices)
        accept_indices_offset = torch.arange(
            0,
            bs * draft_token_num,
            step=draft_token_num,
            dtype=accept_lens.dtype,
            device=accept_lens.device,
        )
        req_idx = torch.arange(bs, dtype=torch.int64, device=accept_lens.device)
        last_correct_step_indices = (
            accept_index[req_idx, (accept_lens - 1).to(torch.int64)]
            - accept_indices_offset
        )
        # extra_buffer: the interval-crossing step whose state must snapshot into
        # the track ping-pong slot (mirrors the regular commit's
        # mamba_steps_to_track); commit_kda_replayssm_spec folds it in one pass, so
        # `temporal` stays current and no device-side force-flush is needed.
        mamba_track_indices = batch.mamba_track_indices
        mamba_steps_to_track = None
        if mamba_track_indices is not None:
            ti = mamba_track_grid(batch.tree_cache.page_size)
            seq_pre = batch.seq_lens
            seq_post = batch.seq_lens + accept_lens
            to_track_mask = seq_pre // ti != seq_post // ti
            tracking_point = seq_post // ti * ti
            to_track_ith = torch.clamp(tracking_point - seq_pre - 1, min=0).to(
                torch.int64
            )
            candidate = accept_index[req_idx, to_track_ith] - accept_indices_offset
            mamba_steps_to_track = torch.where(
                to_track_mask, candidate, torch.full_like(candidate, -1)
            )
        commit_kda_replayssm_after_verify(
            spec_state=spec_state,
            state_batch_indices=state_batch_indices,
            accept_lens=accept_lens,  # incl. bonus token
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            null_block_id=-1,  # SGLang: valid slots >= 0, padding == -1
        )
        return

    attn_backend = model_runner.attn_backend

    bs = accept_lens.shape[0]
    # `accept_lens` already includes the bonus token (drafts + 1 per req).
    if not batch.forward_mode.is_idle() and accept_index.numel() > 0:
        last_correct_step_indices, mamba_steps_to_track = _verify_commit_step_indices(
            batch=batch,
            accept_index=accept_index,
            accept_lens=accept_lens,
            draft_token_num=draft_token_num,
        )

        if hasattr(attn_backend, "update_mamba_state_after_mtp_verify"):
            attn_backend.update_mamba_state_after_mtp_verify(
                last_correct_step_indices=last_correct_step_indices,
                mamba_track_indices=batch.mamba_track_indices,
                mamba_steps_to_track=mamba_steps_to_track,
                model=model_runner.model,
                req_pool_indices=batch.req_pool_indices[:bs],
            )


def spec_prepare_for_decode(batch: ScheduleBatch) -> None:
    """eagle/ngram share a stateless free function; dflash keeps stateful
    prep on its draft input -- the dispatcher routes.
    """
    if mamba_extra_buffer_lazy_enabled():
        # Scheduler phase (outside forward isolation).
        batch.mamba_lazy_spec_prepare(
            mamba_track_grid(batch.tree_cache.page_size),
            max_speculative_num_draft_tokens(),
        )
    if batch.spec_algorithm.is_dflash_family():
        batch.spec_info.prepare_for_decode(batch)
    else:
        from sglang.srt.speculative.eagle_utils import eagle_prepare_for_decode

        eagle_prepare_for_decode(batch)


def get_plan_stream(
    device: str,
) -> Tuple[Any, contextlib.AbstractContextManager]:
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        plan_stream = torch.get_device_module(device).Stream()
        plan_stream_ctx = torch.get_device_module(device).stream(plan_stream)
        return plan_stream, plan_stream_ctx
    else:
        return None, contextlib.nullcontext()
