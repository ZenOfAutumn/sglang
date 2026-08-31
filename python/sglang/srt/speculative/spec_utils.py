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

import logging
import os
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Optional

import torch
from huggingface_hub import snapshot_download

from sglang.srt.distributed.parallel_state import (
    GroupCoordinator,
    patch_tensor_parallel_group,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import set_mamba_track_indices_from_reqs
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.triton_ops.cache_locs import (
    assign_extend_cache_locs as assign_extend_cache_locs,
)
from sglang.srt.speculative.triton_ops.eagle import (
    fill_accept_out_cache_loc as fill_accept_out_cache_loc,
)
from sglang.srt.utils import is_cuda, is_hip, is_musa, is_npu, next_power_of_2
from sglang.srt.utils.async_probe import maybe_detect_oob
from sglang.srt.utils.nvtx_utils import profile_range

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_musa = is_musa()

if TYPE_CHECKING:
    from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.eagle_info import EagleVerifyInput


if _is_cuda:
    from sgl_kernel import fast_topk
elif _is_hip:
    from sgl_kernel import fast_topk
else:
    from sglang.srt.utils.common import fast_topk


logger = logging.getLogger(__name__)


# 仿真接受长度（用于基准测试；<0 时关闭）
SIMULATE_ACC_LEN = envs.SGLANG_SIMULATE_ACC_LEN.get()
SIMULATE_ACC_METHOD = envs.SGLANG_SIMULATE_ACC_METHOD.get()

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


def spec_need_hidden_states(server_args: Optional[ServerArgs] = None) -> bool:
    """判断当前投机解码配置是否需要传递隐藏状态。

    STANDALONE 草稿不消费 spec_info.hidden_states（普通 LLM）。
    multi_layer_eagle 和 DFLASH 不通过 FutureMap 传递 hidden_states。

    参数：
        server_args: 服务器参数（为 None 时使用全局参数）

    返回：True 表示需要传递隐藏状态
    """
    if server_args is None:
        server_args = get_global_server_args()

    # STANDALONE 草稿不消费 spec_info.hidden_states（普通 LLM）。
    # multi_layer_eagle 和 DFLASH 不通过 FutureMap 传递 hidden_states。
    # TODO(lsyin): step == 1 时也应跳过。
    if server_args.speculative_algorithm in ("STANDALONE", "DFLASH"):
        return False
    return not server_args.enable_multi_layer_eagle


@torch.compile(dynamic=True, disable=_is_npu)
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


@torch.compile(dynamic=True, disable=_is_npu)
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


def _sample_simulated_acc_len(
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
    bs,
    spec_steps,
    simulate_acc_len: float = SIMULATE_ACC_LEN,
    simulate_acc_method: str = SIMULATE_ACC_METHOD,
):
    """生成仿真的接受索引（用于基准测试）。

    参数：
        accept_index:        接受索引 [bs, spec_steps + 1]
        predict:             预测 token（会被填充为 100）
        num_correct_drafts:  正确草稿数（会被覆盖）
        bs:                  batch size
        spec_steps:          草稿步数
        simulate_acc_len:    仿真接受长度
        simulate_acc_method: 采样方法

    返回：sim_accept_index [bs, spec_steps + 1]
    """
    assert simulate_acc_len > 0.0
    simulate_acc_len = _sample_simulated_acc_len(
        simulate_acc_len, simulate_acc_method, spec_steps + 1
    )

    accept_indx_first_col = accept_index[:, 0].view(-1, 1)
    sim_accept_index = torch.full(
        (bs, spec_steps + 1), -1, dtype=torch.int32, device="cuda"
    )
    sim_accept_index[:, :simulate_acc_len] = accept_indx_first_col + torch.arange(
        simulate_acc_len, device=accept_index.device
    )
    num_correct_drafts.fill_(simulate_acc_len - 1)
    predict.fill_(100)  # 某个合法 token id
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
    verify_input: EagleVerifyInput,
    retrieve_next_token_cpu: torch.Tensor,
    retrieve_next_sibling_cpu: torch.Tensor,
    draft_tokens_cpu: torch.Tensor,
    vocab_size: int,
):
    """为结构化输出生成 logit mask。

    草稿模型的 token 可能对语法有效或无效。需要执行 DFS 来：
    1. 确定哪些 token 被语法接受。
    2. 如果接受，对应的 logit mask 是什么。

    参数：
        reqs:                       请求列表
        verify_input:               验证输入
        retrieve_next_token_cpu:    子节点索引（CPU）
        retrieve_next_sibling_cpu:  兄弟节点索引（CPU）
        draft_tokens_cpu:           草稿 token id（CPU）
        vocab_size:                 词表大小

    返回：allocate_token_bitmask 或 None（无语法约束时）
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

    verify_input.grammar = grammar
    return allocate_token_bitmask


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
    assign_extend_cache_locs[(bs,)](
        batch.req_pool_indices,
        batch.req_to_token_pool.req_to_token,
        batch.seq_lens,
        batch.seq_lens + num_correct_drafts + 1,
        tgt_cache_loc,
        batch.req_to_token_pool.req_to_token.shape[1],
        next_power_of_2(bs),
    )
    fill_accept_out_cache_loc[(size,)](
        accept_index,
        batch.out_cache_loc,
        accept_out_cache_loc,
        next_power_of_2(size),
    )
    token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
        tgt_cache_loc, accept_out_cache_loc
    )


def prepare_mamba_track_for_verify(batch: ScheduleBatch) -> None:
    """在 TARGET_VERIFY 前从请求重建 mamba track 索引。

    投机 batch 跳过 prepare_for_decode 中的刷新，且 filter/merge 会清空这些字段，
    所以必须在验证前重建。清除 mask 也防止过期的 extend-time mask
    在 TARGET_VERIFY 期间触发前向跟踪；跟踪在 commit_mamba_states_after_verify
    中完成。

    参数：
        batch: 当前调度 batch
    """
    if not get_global_server_args().enable_mamba_extra_buffer():
        return
    set_mamba_track_indices_from_reqs(batch)
    batch.mamba_track_mask = None
    batch.mamba_track_seqlens = None


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
    if model_runner.mambaish_config is None:
        return
    attn_backend = model_runner.attn_backend
    if not hasattr(attn_backend, "update_mamba_state_after_mtp_verify"):
        return

    bs = accept_lens.shape[0]
    # accept_lens 已包含 bonus token（草稿 + 1 每请求）
    if not batch.forward_mode.is_idle() and accept_index.numel() > 0:
        accept_indices_offset = torch.arange(
            0,
            bs * draft_token_num,
            step=draft_token_num,
            dtype=accept_lens.dtype,
            device=accept_lens.device,
        )
        req_idx = torch.arange(bs, dtype=torch.int64, device=accept_lens.device)
        # 每请求最后一个接受节点的树步数，即要提交 mamba 状态的步；
        # 对 topk == 1 退化为 accept_lens - 1
        last_correct_step_indices = (
            accept_index[req_idx, (accept_lens - 1).to(torch.int64)]
            - accept_indices_offset
        )

        if batch.mamba_track_indices is not None:
            # 如果验证后请求的 seq_lens 跨越了 mamba track 区间，
            # 需要在交叉点更新该请求的 mamba 状态
            seq_lens_pre_verify = batch.seq_lens
            seq_lens_post_verify = batch.seq_lens + accept_lens
            mamba_track_interval = get_global_server_args().mamba_track_interval
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != seq_lens_post_verify // mamba_track_interval
            )
            tracking_point = (
                seq_lens_post_verify // mamba_track_interval * mamba_track_interval
            )
            to_track_ith = torch.clamp(
                tracking_point - seq_lens_pre_verify - 1, min=0
            ).to(torch.int64)
            candidate_track_steps = (
                accept_index[req_idx, to_track_ith] - accept_indices_offset
            )
            mamba_steps_to_track = torch.where(
                to_track_mask,
                candidate_track_steps,
                torch.full_like(candidate_track_steps, -1),
            )
        else:
            mamba_steps_to_track = None

        attn_backend.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=model_runner.model,
        )
