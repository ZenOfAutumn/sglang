"""EAGLE 投机解码的底层工具函数集合。

本模块不包含 Worker 类，只提供被各 EAGLE Worker（EagleWorker / EagleWorkerV2 /
NGRAMWorker / StandaloneWorkerV2 等）共享的纯函数：

    1. 草稿树构建   — build_tree_kernel_efficient()
    2. 草稿结果整理 — organize_draft_results()
    3. 验证前准备   — eagle_prepare_for_verify()
    4. 贪心/拒绝采样 — eagle_sample() / verify_tree_greedy_func()
    5. 辅助工具     — per_step_draft_out_cache_loc() / get_draft_hidden_dim() 等

设计原则：这些函数不持有状态，所有可变参数以「mutable」标注，由调用方
预分配并传入，避免每轮 decode 都 malloc 显存（也兼容 CUDA Graph）。
"""

from __future__ import annotations

import logging
import math
from enum import IntEnum
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.kernels.ops.speculative.spec_tree import (
    sgl_build_tree_kernel_efficient_triton,
    verify_tree_greedy_kernel_triton,
)
from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
    maybe_build_dsv4_verify_bundle,
)
from sglang.srt.mem_cache.allocation import alloc_for_spec_decode
from sglang.srt.mem_cache.allocation_sizing import (
    get_alloc_reserve_per_decode,
    page_aligned_decode_alloc_lens,
)
from sglang.srt.model_executor.runner_utils.pool import borrow_graph_pool
from sglang.srt.runtime_context import get_parallel, get_spec
from sglang.srt.utils import (
    is_cpu,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    is_xpu,
)
from sglang.srt.utils.async_probe import maybe_detect_oob

if TYPE_CHECKING:
    from sglang.srt.constrained.base_grammar_backend import GrammarMask
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
    from sglang.srt.speculative.eagle_info import EagleVerifyInput

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_musa = is_musa()
_is_xpu = is_xpu()
_is_cpu = is_cpu()

logger = logging.getLogger(__name__)

if _is_cuda or _is_hip or _is_musa:
    from sgl_kernel import (
        build_tree_kernel_efficient as sgl_build_tree_kernel_efficient,
    )
elif _is_cpu:
    from sgl_kernel import (
        build_tree_kernel_efficient_cpu as sgl_build_tree_kernel_efficient_cpu,
    )
    from sgl_kernel import verify_tree_greedy_cpu as sgl_verify_tree_greedy_cpu


def per_step_draft_out_cache_loc(
    out_cache_loc: torch.Tensor,
    batch_size: int,
    topk: int,
    num_steps: int,
) -> torch.Tensor:
    """多步 EAGLE 草稿的 out_cache_loc 按步切片。

    将原始布局 [bs, topk, num_steps] 重排为 [num_steps, bs * topk]，
    使得第 i 步的所有候选可以连续写入。

    这是 EagleWorkerV2.draft_forward（按步写入目标）与
    DeepseekV4AttnBackend（按步压缩写入元数据）共享的布局唯一真相源。

    参数：
        out_cache_loc: KV cache 位置张量，shape [bs * topk * num_steps]
        batch_size:    batch 内请求数
        topk:          每步每请求的候选数
        num_steps:     草稿推理步数

    返回：shape [num_steps, bs * topk] 的视图（不复制数据）

    示例：
        >>> loc = torch.arange(2 * 3 * 4)  # shape [24]
        >>> per_step_draft_out_cache_loc(loc, bs=2, topk=3, num_steps=4)
        # 内部: view(2,3,4) → permute(2,0,1) → reshape(4, 6)
        # 结果 shape [4, 6]，第 0 行 = step0 的 6 个候选对应的 cache loc
    """
    expected = batch_size * topk * num_steps
    assert out_cache_loc.shape[0] == expected, (
        f"out_cache_loc.shape[0]={out_cache_loc.shape[0]} != "
        f"batch_size * topk * num_steps = {batch_size}*{topk}*{num_steps}={expected}"
    )
    return (
        out_cache_loc.view(batch_size, topk, num_steps)
        .permute(2, 0, 1)
        .reshape(num_steps, -1)
    )


def _eagle_prefill_tail_tokens(
    batch: ScheduleBatch, next_token_ids: torch.Tensor
) -> torch.Tensor:
    """EAGLE prefill 旋转所需的每请求尾 token。

    对于分块 prefill（chunked prefill）的非最后一块，需要用「下一块的第一个
    prompt token」替代目标模型自产的 next_token_ids，保证链式一致性
    （见 PR #26329）。

    参数：
        batch:           当前调度 batch
        next_token_ids:  目标模型产出的各请求首 token，shape [bs]

    返回：修正后的尾 token 张量，shape [bs]

    示例：
        >>> # batch 中 req0 正在做分块 prefill 的第 1 块（非最后）
        >>> # next_token_ids = [42, 99], next_prompt_token = 7
        >>> # 结果: [7, 99]  ← req0 的尾 token 被替换为下一块的 prompt token
    """
    tail_tokens = next_token_ids.to(batch.input_ids.dtype)
    next_prompt_token = batch.chunked_req_next_prompt_token
    if next_prompt_token is not None:
        for i, r in enumerate(batch.reqs):
            if r is batch.chunked_req:
                tail_tokens = tail_tokens.clone()
                # Keep the scalar as a kernel argument. Assigning a Python scalar
                # through scalar indexing issues a pageable H2D copy and
                # synchronizes the current CUDA stream before draft extend.
                tail_tokens[i : i + 1].fill_(next_prompt_token)
                break
    return tail_tokens


def organize_draft_results(
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    num_draft_token: int,
):
    # b, n, topk; n = 1 + (num_steps-1) * topk
    score_list = torch.cat(score_list, dim=1).flatten(1)
    # b, (topk + (num_steps-1) * topk)
    ss_token_list = torch.cat(token_list, dim=1)
    # 取分数最高的 num_draft_token - 1 个（减 1 是因为 bonus token 不算草稿）
    top_scores = torch.topk(score_list, num_draft_token - 1, dim=-1)
    top_scores_index = top_scores.indices
    # 排序以保证树的结构一致性（父节点必须在子节点之前）
    top_scores_index = torch.sort(top_scores_index).values
    maybe_detect_oob(
        top_scores_index,
        0,
        ss_token_list.shape[1],
        "organize_draft_results: top_scores_index OOB for gather on ss_token_list",
    )
    # 按下标从拼接后的候选列表中提取实际 token id
    draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

    # 拼接父节点下标：最后一步的 parent 不需要（没有后续步骤引用它）
    if len(parents_list) > 1:
        parent_list = torch.cat(parents_list[:-1], dim=1)
    else:
        batch_size = parents_list[0].shape[0]
        parent_list = torch.empty(
            batch_size, 0, dtype=torch.long, device=parents_list[0].device
        )

    return parent_list, top_scores_index, draft_tokens


class TreeMaskMode(IntEnum):
    """树形注意力 mask 的三种模式。

    FULL_MASK:           完整矩阵 [num_verify_tokens, seq_len + num_verify_tokens]，
                         左半段全 True（历史 KV 全可见），右半段为树内可见性。
    QLEN_ONLY:           只存树内 [num_verify_tokens, num_verify_tokens] 部分，
                         历史可见性由注意力后端隐式处理（约快 8%）。
    QLEN_ONLY_BITPACKING: 同 QLEN_ONLY 但用位打包（uint8/16/32）压缩存储，
                         进一步减少显存和带宽。
    """
    FULL_MASK = 0
    QLEN_ONLY = 1
    QLEN_ONLY_BITPACKING = 2


def default_tree_mask_mode() -> TreeMaskMode:
    # The CPU verify attention kernel (intel_amx) consumes the qlen x qlen
    # QLEN_ONLY tree mask directly; FULL_MASK is for the GPU kernels.
    return TreeMaskMode.QLEN_ONLY if _is_cpu else TreeMaskMode.FULL_MASK


def build_tree_kernel_efficient(
    bonus_tokens: torch.Tensor,
    parent_list: List[torch.Tensor],
    top_scores_index: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_sum: int,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK,
    tree_mask_buf: Optional[torch.Tensor] = None,
    fill_prefix_mask: bool = True,
):
    """高效构建 EAGLE 草稿树：拼接 bonus + 草稿，生成树形 mask、位置和检索索引。

    参数：
        bonus_tokens:      目标模型上一轮的输出 token，shape [bs]
        parent_list:       各草稿 token 的父节点下标列表
        top_scores_index:  被选中的候选在拼接后的 token_list 中的列下标
        draft_tokens:      草稿 token id，shape [bs, num_draft_token - 1]
        seq_lens:          各请求的序列长度（不含草稿），shape [bs]
        seq_lens_sum:      sum(seq_lens)，用于计算 FULL_MASK 的总大小
        topk:              每步每请求的候选数
        spec_steps:        草稿推理步数
        num_verify_tokens: 验证时的总 token 数 = spec_steps * topk + 1（bonus）
        tree_mask_mode:    树形 mask 模式，见 TreeMaskMode
        tree_mask_buf:     预分配的 mask 缓冲区（CUDA Graph 友好），None 则内部分配
        position_buf:      预分配的位置缓冲区，None 则内部分配

    返回 (tree_mask, positions, retrieve_index, retrieve_next_token,
           retrieve_next_sibling, draft_tokens)：
        tree_mask:               树形注意力 mask（拍平）
        positions:               每个 token 的绝对位置，shape [bs * num_verify_tokens]
        retrieve_index:          节点全局下标，shape [bs, num_verify_tokens]
        retrieve_next_token:     第一个子节点下标，无子为 -1
        retrieve_next_sibling:   下一个兄弟下标，无兄弟为 -1
        draft_tokens:            拼接了 bonus 的完整草稿 token（拍平）

    示例（bs=1, topk=2, spec_steps=2, num_verify_tokens=5, seq_len=7）：
        >>> # bonus=[10], draft_tokens=[[20,30,40,50]]
        >>> # 拼接后 draft_tokens = [10, 20, 30, 40, 50]
        >>> # 若树深度为 [0, 1, 1, 2, 2]，则 positions = [7, 8, 8, 9, 9]
        >>> # tree_mask 描述每个 token 能 attend 到哪些其他 token
    """
    # 拼接 bonus token（根节点）和草稿 token，拍平成 1D
    draft_tokens = torch.cat((bonus_tokens.unsqueeze(1), draft_tokens), dim=1).flatten()

    # seq_lens_sum == sum(seq_lens); seq_lens: 不含草稿的序列长度
    bs = seq_lens.numel()
    device = seq_lens.device
    # 例如 bs=1 时，tree_mask: [num_verify_tokens, seq_lens_sum + num_verify_tokens]（拍平）
    # 每行表示一个草稿 token 的可见性模式
    # 如果使用位打包（QLEN_ONLY_BITPACKING），则 tree_mask 被压缩存储
    if tree_mask_buf is not None:
        # 使用预分配缓冲区（CUDA Graph 场景），只填充初始值
        tree_mask = tree_mask_buf
        if tree_mask_mode == TreeMaskMode.QLEN_ONLY:
            tree_mask.fill_(True)
        elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
            tree_mask.fill_(0)
        elif tree_mask_mode == TreeMaskMode.FULL_MASK:
            # Only the [0, seq_len) prefix columns depend on this fill; the
            # kernel below writes every tree cell itself. Skip the (up to
            # 100s of MB) per-step memset when nothing reads the mask.
            if fill_prefix_mask:
                tree_mask.fill_(True)
        else:
            raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY:
        # QLEN_ONLY: 只存树内部分 [num_verify_tokens, num_verify_tokens]（每请求）
        tree_mask = torch.full(
            (num_verify_tokens * bs * num_verify_tokens,),
            True,
            dtype=torch.bool,
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
        # 位打包：根据 num_verify_tokens 选择最小够用的整数类型
        packed_dtypes = [torch.uint8, torch.uint16, torch.uint32]
        packed_dtype_idx = int(math.ceil(math.log2((num_verify_tokens + 7) // 8)))
        tree_mask = torch.zeros(
            (num_verify_tokens * bs,),
            dtype=packed_dtypes[packed_dtype_idx],
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.FULL_MASK:
        mask_shape = (
            seq_lens_sum * num_verify_tokens
            + num_verify_tokens * num_verify_tokens * bs,
        )
        # Same reasoning as the preallocated branch above.
        tree_mask = (
            torch.full(mask_shape, True, dtype=torch.bool, device=device)
            if fill_prefix_mask
            else torch.empty(mask_shape, dtype=torch.bool, device=device)
        )
    else:
        raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")

    # TODO: 改成 torch.empty 并融合进 sgl_build_tree_kernel，减少一次 kernel launch
    retrieve_buf = torch.full(
        (3, bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    retrieve_index, retrieve_next_token, retrieve_next_sibling = retrieve_buf
    # position: where each token belongs to
    # e.g. if depth of each draft token is [0, 1, 1, 2] and the prompt length is 7
    # then, positions = [7, 8, 8, 9]
    positions = torch.empty((bs * num_verify_tokens,), device=device, dtype=torch.long)

    # 根据硬件后端调用对应的 tree build kernel
    if _is_npu:
        torch.ops.npu.build_tree_kernel_efficient(
            parent_list.to(dtype=torch.int64),
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    elif _is_xpu:
        sgl_build_tree_kernel_triton(
            parent_list,
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    elif _is_cpu:
        sgl_build_tree_kernel_efficient_cpu(
            parent_list,
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    else:
        sgl_build_tree_kernel_efficient(
            parent_list,
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    return (
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        draft_tokens,
    )


def sgl_build_tree_kernel_triton(
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    verified_seq_len: torch.Tensor,
    tree_mask: torch.Tensor,
    positions: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    topk: int,
    depth: int,
    draft_token_num: int,
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK,
):
    """Triton-based implementation."""
    # TODO: Add support for QLEN_ONLY_BITPACKING mode
    if tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
        raise NotImplementedError(
            "QLEN_ONLY_BITPACKING is not supported in Triton implementation"
        )

    batch_size = verified_seq_len.shape[0]
    seq_len_prefix_sum = torch.cumsum(verified_seq_len, dim=0) - verified_seq_len

    # Launch kernel with one program per batch item
    grid = (batch_size,)

    sgl_build_tree_kernel_efficient_triton[grid](
        parent_list,
        selected_index,
        verified_seq_len,
        seq_len_prefix_sum,
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        topk=topk,
        depth=depth,
        draft_token_num=draft_token_num,
        tree_mask_mode=int(tree_mask_mode),
        batch_size=batch_size,
        parent_list_stride=(
            parent_list.stride(0) if parent_list.dim() > 1 else parent_list.shape[0]
        ),
        selected_index_stride=selected_index.stride(0),
    )


def verify_tree_greedy_triton(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
):
    """Triton-based implementation."""
    batch_size = candidates.shape[0]
    num_speculative_tokens = accept_index.shape[1]
    num_draft_tokens = candidates.shape[1]

    # Launch kernel with one program per batch item
    grid = (batch_size,)

    verify_tree_greedy_kernel_triton[grid](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        target_predict,
        batch_size=batch_size,
        num_speculative_tokens=num_speculative_tokens,
        num_draft_tokens=num_draft_tokens,
    )


def verify_tree_greedy_func(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
    topk: int = -1,
):
    """贪心验证草稿树：沿树路径比较草稿 token 与目标模型预测，接受匹配的。

    参数（predicts / accept_index / accept_token_num 为 mutable 输出）：
        predicts:           输出，各节点的最终预测 token，shape [bs * num_verify_tokens]
        accept_index:       输出，被接受节点的下标，shape [bs, max_tree_depth]
        accept_token_num:   输出，各请求接受的草稿数（不含 bonus），shape [bs]
        candidates:         草稿候选 token，shape [bs, num_verify_tokens]
        retrieve_index:     树节点全局下标
        retrieve_next_token:   第一个子节点下标
        retrieve_next_sibling: 下一个兄弟下标
        target_predict:    目标模型 argmax 预测，shape [bs, num_verify_tokens]
        topk:              保留参数，默认 -1

    返回 (predicts, accept_index, accept_token_num)，与输入的 mutable 参数相同。

    示例（bs=1, num_verify_tokens=5, 树深度=3）：
        >>> # candidates = [[10, 20, 30, 40, 50]]
        >>> # target_predict = [[10, 20, 99, 40, 60]]
        >>> #                节点0匹配→节点1匹配→节点2不匹配→回溯到兄弟节点3匹配
        >>> # 结果: accept_index = [[0, 1, 3, -1]], accept_token_num = [3]
        >>> #       predicts = [10, 20, 40, ...]  (沿接受路径的预测 token)
    """
    if _is_cuda or _is_hip or _is_musa:
        from sgl_kernel import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,  # mutable: 被 kernel 原地写入
            accept_index=accept_index,  # mutable
            accept_token_num=accept_token_num,  # mutable
            candidates=candidates,
            # kwarg 左侧保留 `retrive_*` 拼写以匹配 sgl_kernel 的 op schema
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )

    elif _is_cpu:
        sgl_verify_tree_greedy_cpu(
            predicts=predicts,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=accept_token_num,  # mutable
            candidates=candidates,
            # kwarg LHS retained as `retrive_*` to match the CUDA op schema, so
            # the CPU/CUDA call sites stay grep-symmetric.
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )

    elif _is_npu:
        from sgl_kernel_npu.sample.verify_tree_greedy import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            # kwarg 左侧保留 `retrive_*` 拼写以匹配 sgl_kernel 的 op schema
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )
    elif _is_xpu:
        verify_tree_greedy_triton(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )
    return predicts, accept_index, accept_token_num


def get_draft_input_from_target_hidden_dim(model_runner: ModelRunner) -> int:
    """Width of the target hidden states fed into the draft model.

    This is the single source of truth and is derived entirely from config: for
    EAGLE3 aux mode the draft consumes `num_aux` concatenated target layers
    (each `target_hidden_size` wide); every other arch consumes the per-layer
    `spec_hidden_size`.

    Do NOT read this off a draft projection's `in_features` (e.g. an `fc`
    layer): that width is arch-specific.

    Note: read entirely from the *draft* `model_runner`'s config. The non-aux
    branch assumes the draft's `spec_hidden_size` equals the target hidden width
    fed to the draft (true for standard EAGLE, where the draft mirrors the
    target hidden size); aux mode reads the explicit `target_hidden_size`.
    """
    model_config = model_runner.model_config
    hf_config = model_config.hf_config
    eagle_config = getattr(hf_config, "eagle_config", None) or {}
    get_eagle_config = (
        eagle_config.get
        if isinstance(eagle_config, dict)
        else lambda key, default=None: getattr(eagle_config, key, default)
    )
    use_aux = get_eagle_config("use_aux_hidden_state", True)
    spec_algorithm = model_runner.spec_algorithm

    if not (spec_algorithm is not None and spec_algorithm.is_eagle3() and use_aux):
        return model_config.spec_hidden_size

    target_hidden = getattr(hf_config, "target_hidden_size", None)
    if target_hidden is None:
        target_hidden = model_config.hidden_size
    num_aux = getattr(hf_config, "num_aux_hidden_states", None)
    if num_aux is None:
        layer_ids = get_eagle_config("eagle_aux_hidden_state_layer_ids", None)
        if layer_ids is None:
            layer_ids = getattr(hf_config, "eagle_aux_hidden_state_layer_ids", None)
        num_aux = len(layer_ids) if layer_ids else 3
    return target_hidden * num_aux


def get_draft_recurrent_hidden_state_spec_from_config(
    model_config, spec_algorithm
) -> tuple[Optional[int], Optional[torch.dtype]]:
    """Return hidden_states width/dtype carried between draft decode steps.

    Config-only so callers without a draft runner can reach it: prefill-side PP
    builds the draft on the last stage alone, but the PD metadata wire schema it
    feeds has to come out identical on every rank.
    """
    if spec_algorithm.is_standalone():
        return None, None
    return model_config.spec_hidden_size, model_config.dtype


def get_draft_recurrent_hidden_state_spec(
    model_runner: ModelRunner,
) -> tuple[Optional[int], Optional[torch.dtype]]:
    """Return hidden_states width/dtype carried between draft decode steps."""
    return get_draft_recurrent_hidden_state_spec_from_config(
        model_runner.model_config, model_runner.spec_algorithm
    )


def eagle_prepare_for_verify(
    verify_input: EagleVerifyInput,
    req_to_token_pool: ReqToTokenPool,
    batch: ScheduleBatch,
    target_worker: TpModelWorker,
):
    from sglang.kernels.ops.speculative.cache_locs import (
        assign_extend_cache_locs_uniform_func,
    )
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardBatch,
        ForwardMode,
    )
    from sglang.srt.speculative.spec_utils import prepare_mamba_track_for_verify

    if not batch.forward_mode.is_idle():
        # 分配 KV cache 位置
        bs = len(batch.req_pool_indices)
        batch.input_ids = verify_input.draft_token
        maybe_detect_oob(
            batch.input_ids,
            0,
            batch.model_config.vocab_size,
            "v2 prepare_for_verify input_ids",
        )
        device = batch.device
        # Uniform variant: end offsets (= start + draft_token_num) are computed
        # inside the kernel, keeping the eager `seq_lens + N` add off the host
        # critical path (bs=1 MTP inter-phase seam).
        batch.out_cache_loc = assign_extend_cache_locs_uniform_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=req_to_token_pool.req_to_token,
            start_offset=batch.seq_lens,
            batch_size=bs,
            draft_token_num=verify_input.draft_token_num,
            device=device,
        )

        batch.out_cache_loc_dsv4 = maybe_build_dsv4_verify_bundle(
            batch, verify_input.draft_token_num
        )

        prepare_mamba_track_for_verify(batch)

        # TBO 的 split_spec_info 会读取这两个字段；no-verify-sync 模式下它们保持 None
        verify_input.seq_lens_cpu = batch.seq_lens_cpu
        verify_input.seq_lens_sum = (
            int(batch.seq_lens_cpu.sum()) if batch.seq_lens_cpu is not None else None
        )

    # 设置 forward_mode：IDLE 保持 IDLE，否则改为 TARGET_VERIFY
    batch.forward_mode = (
        ForwardMode.IDLE if batch.forward_mode.is_idle() else ForwardMode.TARGET_VERIFY
    )
    # Standalone 模式不需要捕获目标模型的隐藏状态（草稿模型独立运行）
    capture_mode = (
        CaptureHiddenMode.NULL
        if target_worker.model_runner.spec_algorithm.is_standalone()
        else CaptureHiddenMode.FULL
    )
    verify_forward_batch = ForwardBatch.init_new(
        batch,
        target_worker.model_runner,
        capture_hidden_mode=capture_mode,
        return_hidden_states_before_norm=False,
    )

    # 运行注意力后端 plan 和 CUDA Graph 准备
    can_run_cuda_graph = bool(
        target_worker.model_runner.decode_cuda_graph_runner
        and target_worker.model_runner.decode_cuda_graph_runner.can_run_graph(
            verify_forward_batch
        )
    )
    if can_run_cuda_graph:
        target_worker.model_runner.decode_cuda_graph_runner.load_batch(
            verify_forward_batch
        )
        verify_forward_batch.mark_forward_metadata_ready()
    # 非 CUDA Graph 路径：延迟初始化到 forward_extend，因为它在
    # `_forward_raw -> prepare_mlp_sync_batch` 对 batch 做 padding 之后才运行。
    # 在这里初始化会使用 padding 前的形状，导致 DSv4 indexer 形状不匹配。

    return verify_forward_batch, can_run_cuda_graph


def _seeded_verify_coins(
    *,
    sampling_seed: torch.Tensor,
    seq_lens: torch.Tensor,
    draft_token_num: int,
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Derive deterministic verify-side coins from per-request sampling seeds.

    Mirrors the main seeded-sampling path: murmur_hash32(seed, seq_lens,
    column) mapped to [0, 1). Columns [0, draft_token_num) drive the
    per-draft rejection coins; column draft_token_num drives the final
    fallback-sampling coin.

    Scope: this seeds only the verify-side RNG. With rejection sampling the
    draft workers still pick candidates via unseeded multinomial
    (fast_sample in eagle_worker_v2), so that mode stays non-deterministic
    until the draft RNG is seeded in a follow-up; top-k/greedy draft
    selection is already deterministic.
    """
    from sglang.kernels.ops.sampling.murmur_hash import murmur_hash32

    cols = torch.arange(draft_token_num + 1, device=device, dtype=torch.int64)
    hashed = murmur_hash32(
        sampling_seed.to(torch.uint64), seq_lens.to(torch.uint64), cols
    )
    uniforms = hashed.to(torch.float64) / torch.iinfo(torch.uint32).max
    # The float32 cast rounds the top 129 uint32 hashes to exactly 1.0, but
    # the sampling kernels expect half-open [0, 1) coins: a 1.0 coin walks
    # past the last CDF bucket and can return a zero-probability token.
    # Clamp to the largest float32 below one; every other coin value is
    # untouched, so previously verified bitwise baselines stay intact.
    max_coin = 1.0 - 2**-24
    coins = (
        uniforms[:, :draft_token_num].to(torch.float32).clamp_(max=max_coin)
    ).contiguous()
    coins_for_final_sampling = (
        uniforms[:, draft_token_num].to(torch.float32).clamp_(max=max_coin)
    ).contiguous()
    return coins, coins_for_final_sampling


def _verify_coins(
    *,
    sampling_info: SamplingBatchInfo,
    seq_lens: torch.Tensor,
    draft_token_num: int,
    candidates: torch.Tensor,
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rejection and final-sampling coins for verify: deterministic seeded
    coins when sampling_seed is set (see _seeded_verify_coins), torch.rand
    otherwise.
    """
    if sampling_info.sampling_seed is not None:
        return _seeded_verify_coins(
            sampling_seed=sampling_info.sampling_seed,
            seq_lens=seq_lens,
            draft_token_num=draft_token_num,
            device=device,
        )
    # coins for rejection sampling
    coins = torch.rand_like(candidates, dtype=torch.float32, device=device)
    # coins for final sampling
    coins_for_final_sampling = torch.rand(
        (candidates.shape[0],), dtype=torch.float32, device=device
    )
    return coins, coins_for_final_sampling


def eagle_sample(
    verify_input: EagleVerifyInput,
    batch: ScheduleBatch,
    logits_output: LogitsProcessorOutput,
    grammar_mask: Optional[GrammarMask] = None,
):
    """基于 logits 输出和 batch 中的投机解码信息，验证并找出被接受的 token。

    这是 EAGLE 验证阶段的核心采样函数，支持贪心采样和树形拒绝采样两种模式。
    NGRAM 等不使用草稿模型的算法也复用此函数（因为「验证一棵树」的逻辑是通用的）。

    参数：
        verify_input:  EagleVerifyInput，包含草稿 token、树结构、检索索引等
        batch:         当前调度 batch，包含 sampling_info 等
        logits_output: 目标模型前向的 logits 输出
        vocab_mask:    可选的语法 mask（结构化输出场景），None 则不应用

    返回 (predict, accept_lens, accept_index)：
        predict:      各节点的目标模型预测 token，shape [bs * num_verify_tokens]
        accept_lens:  各请求接受长度（含 bonus），如 tensor([3, 1])
                      最小值恒为 1（bonus token 总是接受），最大值为 draft_token_num
        accept_index: 被接受节点的下标，shape [bs, max_tree_depth]，未填满的位置为 -1

    示例（bs=2, draft_token_num=16, max_tree_depth=5）：
        >>> # 贪心模式：目标模型预测与草稿逐节点比对
        >>> # req0: 草稿=[10,20,30,...]，目标=[10,20,99,...] → 接受 2 个草稿 + 1 bonus = 3
        >>> # req1: 草稿=[50,60,...]，目标=[50,99,...] → 接受 0 个草稿 + 1 bonus = 1
        >>> # 结果: accept_lens = tensor([3, 1])
        >>> #       accept_index = [[0,1,-1,-1,-1], [0,-1,-1,-1,-1]]
    """
    import torch.nn.functional as F

    from sglang.srt.distributed import get_tp_group
    from sglang.srt.layers.dp_attention import (
        is_dp_attention_enabled,
    )
    from sglang.srt.sampling.penaltylib.repetition_penalty import (
        apply_scaling_penalties,
    )
    from sglang.srt.speculative.spec_utils import (
        SIMULATE_ACC_LEN,
        SIMULATE_ACC_TOKEN_MODE,
        generate_simulated_accept_index,
    )
    from sglang.srt.utils.async_probe import maybe_detect_nan, sanitize_nan_logits

    device = batch.device
    # IDLE 模式：空转，返回空张量
    if batch.forward_mode.is_idle():
        predict = torch.empty(0, dtype=torch.int32, device=device)
        num_correct_drafts = torch.empty(0, dtype=torch.int32, device=device)
        accept_index = torch.empty(0, dtype=torch.int32, device=device)
        return predict, num_correct_drafts, accept_index

    bs = len(batch.seq_lens)
    sampling_info = batch.sampling_info
    # verify 阶段 next_token_logits 的布局：[bs * draft_token_num, vocab_size]
    #
    # 与普通 decode（每个请求只占一行）不同，这里草稿树的**每个节点**都占一行——
    # 因为要知道「若走到该节点，目标模型接下来会预测什么」，才能做逐层接受判定。
    # 行的排列是「按请求分块、块内按草稿树的展平顺序」，与 verify_input.draft_token 严格一一对应。
    #
    # 例：bs=2、draft_token_num=4（每请求一棵 4 节点的树），vocab_size=V
    #     draft_token       = [d00, d01, d02, d03,  d10, d11, d12, d13]   # 长度 8
    #     next_token_logits.shape == (8, V)
    #
    #     行号 | 归属  | 语义
    #     -----+-------+--------------------------------------------------
    #      0   | req0  | 前缀 + d00 之后，目标模型对下一个 token 的分布
    #      1   | req0  | 前缀 + (d00 的祖先链) + d01 之后的分布
    #      2   | req0  | 同上，对应 d02
    #      3   | req0  | 同上，对应 d03
    #      4   | req1  | 换请求，重新从该请求的树根开始
    #      5-7 | req1  | d11 / d12 / d13
    #
    # 注意第 i 行是「d_i **之后**」的分布，不是「用来生成 d_i」的分布——所以接受判定时
    # 是拿第 i 行去比对 d_i 在树中的**子节点**，而不是比对 d_i 自己。链尾那一行没有子节点
    # 可比，它采出来的就是 bonus token（即使整条草稿全被拒绝，也至少能白拿这一个）。
    #
    # 树形（topk > 1）下同一行内的父子关系由 retrieve_next_token / retrieve_next_sibling 描述，
    # 布局本身仍是上面这种平铺形式，只是「哪些行是同一个父节点的兄弟」需要查这两张表。
    #
    # 该张量后续会被原地修改（惩罚 add_、grammar mask masked_fill_），且可能来自复用缓冲区
    # forward_batch.next_token_logits_buffer，不要假设它在本函数之后仍保持原值。
    next_token_logits = logits_output.next_token_logits

    # 清理 NaN logits（投机解码下数值异常会经由「接受」传播到后续轮次）
    sanitize_nan_logits(next_token_logits, "verify: target model logits")

    # 应用惩罚（relaxed 版本，适用于投机解码）
    # 这里的惩罚被广播到每个草稿 token 的 logits 上
    #
    # 为什么叫 "relaxed"：普通解码每步只出一个 token，惩罚可以在每步之前按「截至此刻的
    # 真实序列」精确重算；而 verify 一次要判定整棵草稿树（每个请求 draft_token_num 个位置），
    # 树内各节点对应的前缀各不相同，无法为每个节点各算一份惩罚。这里的做法是取
    # 「本轮验证开始时」的那一份惩罚快照，用 repeat_interleave 沿 batch 维复制
    # draft_token_num 份（[bs, V] → [bs * draft_token_num, V]），让同一请求的所有草稿位置
    # 共用同一份惩罚。即：树内新产生的 token 不会立刻反过来惩罚它自己的后继，
    # 惩罚要等下一轮验证才更新——这是投机解码为了批量判定而接受的近似。
    #
    # acc_* 前缀表示 accumulated：由 penalizer_orchestrator 把用户设置的各项惩罚
    # 预先累加成一张 [bs, vocab_size] 的表（见 sampling_batch_info.update_penalties），
    # 此处只做加法/缩放，不再逐项遍历 penalizer。

    # 加性惩罚：presence_penalty、frequency_penalty 等，直接加到 logits 上（负值即压低）
    if sampling_info.acc_additive_penalties is not None:
        next_token_logits.add_(
            torch.repeat_interleave(
                sampling_info.acc_additive_penalties,
                verify_input.draft_token_num,
                dim=0,
            )
        )
    # 乘性惩罚：repetition_penalty，按符号缩放（正 logits 除以系数、负 logits 乘以系数），
    # 因此不能用简单加法，需走 apply_scaling_penalties
    if sampling_info.acc_scaling_penalties is not None:
        apply_scaling_penalties(
            next_token_logits,
            torch.repeat_interleave(
                sampling_info.acc_scaling_penalties, verify_input.draft_token_num, dim=0
            ),
        )
    # 用户显式指定的 per-token 偏置（logit_bias），与生成历史无关，
    # 因此对树内每个位置都是精确的，不存在上面说的近似问题
    if sampling_info.logit_bias is not None:
        next_token_logits.add_(
            torch.repeat_interleave(
                sampling_info.logit_bias, verify_input.draft_token_num, dim=0
            )
        )

    # Apply grammar mask if provided
    if grammar_mask is not None:
        grammar_mask.apply(next_token_logits)

    # ---- 准备四个工作张量 ----------------------------------------------------
    # 以下注释统一用这个例子：bs=2、topk=1（链式草稿）、steps=3，
    # 于是 draft_token_num D=4（3 个草稿 + 1 个 bonus 位）、max_tree_depth=4、词表 V。
    # 树形（topk>1）时 D 会远大于 max_tree_depth，两者的区别在 accept_index 处说明。

    # [bs*D] → [bs, D]：把平铺的草稿 token 按请求分行，便于按请求索引。
    #   draft_token = [d00,d01,d02,d03, d10,d11,d12,d13]   # shape (8,)
    #   candidates  = [[d00,d01,d02,d03],
    #                  [d10,d11,d12,d13]]                  # shape (2, 4)
    candidates = verify_input.draft_token.reshape(bs, verify_input.draft_token_num)

    # 砍掉 logits 的词表维，得到「每个树节点一个输出 token」的形状。
    #   next_token_logits.shape = (bs*D, V) = (8, V) → predict_shape = [8]
    # 这里不用 bs*D 硬算，是为了跟着 logits 的实际形状走（padding 场景下更稳）。
    predict_shape = list(next_token_logits.shape)[:-1]
    # predict: shape (bs*D,) = (8,)，与 draft_token 同布局、逐位置对应。
    # 由下面的 kernel **原地写入**：写的是「该节点被接受后，实际采纳的下一个 token」。
    # 未被接受的位置保持 0（无意义，调用方只按 accept_index 取值）。
    # flatten() 在这里是保险——predict_shape 已是一维，防止上游给出非平铺形状。
    predict = torch.zeros(predict_shape, dtype=torch.int32, device=device).flatten()

    # accept_index: shape (bs, max_tree_depth) = (2, 4)，同样由 kernel 原地写入。
    # 存的是「被接受的那条路径依次经过哪些节点」，用的是 candidates 展平后的**全局下标**
    # （即 0..bs*D-1），未填满的位置留 -1。
    #   例：req0 接受 d00→d01 后失配，req1 首个就失配
    #   accept_index = [[0, 1, -1, -1],
    #                   [4, -1, -1, -1]]
    # 宽度取 max_tree_depth 而不是 D：一条从根到叶的路径最长就是树深，
    # 树形（如 topk=2、steps=3 → D=15、max_tree_depth=4）时二者差距很大，
    # 用 D 会浪费十几倍显存。
    accept_index = torch.full(
        (bs, verify_input.max_tree_depth), -1, dtype=torch.int32, device=device
    )

    # num_correct_drafts: shape (bs,) = (2,)，kernel 原地写入。
    # 函数内部**只计草稿命中数、不含 bonus**（上例为 [2, 0]），
    # 返回时才 +1 变成含 bonus 的接受长度（见函数末尾）。
    num_correct_drafts = torch.empty((bs,), dtype=torch.int32, device=device)

    # Sample tokens
    target_predict = None
    if sampling_info.is_all_greedy or _is_cpu or _is_npu or _is_hip or _is_xpu:
        target_predict = torch.argmax(next_token_logits, dim=-1)
        # [bs*D] → [bs, D]：与 candidates 对齐，kernel 要按请求逐行走树。
        #   target_predict = [[t00,t01,t02,t03],
        #                     [t10,t11,t12,t13]]                  # shape (2, 4)
        # 比对时错开一位：拿 target_predict[i][j]（"d_ij 之后"的预测）去比
        # d_ij 的子节点，而不是比 d_ij 自己——见函数开头对 logits 布局的说明。
        target_predict = target_predict.reshape(bs, verify_input.draft_token_num)
        predict, accept_index, num_correct_drafts = verify_tree_greedy_func(
            predicts=predict,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=num_correct_drafts,  # mutable
            candidates=candidates,
            retrieve_index=verify_input.retrieve_index,
            retrieve_next_token=verify_input.retrieve_next_token,
            retrieve_next_sibling=verify_input.retrieve_next_sibling,
            target_predict=target_predict,
            topk=verify_input.tree_topk,
        )

        if _is_hip:
            # On ROCm, the per-rank draft tokens can differ, so ranks accept a
            # different number of drafts, desynchronize the committed seq_lens, and
            # deadlock the next TP collective. Broadcast from rank 0 to ensure
            # consistency, the same way the sampling branch below does.
            tp_group = (
                get_parallel().attn_tp_group
                if is_dp_attention_enabled()
                else get_tp_group()
            )
            if tp_group.world_size > 1:
                tp_group.broadcast(predict, src=0)
                tp_group.broadcast(accept_index, src=0)
                tp_group.broadcast(num_correct_drafts, src=0)
    else:
        # 拒绝采样模式：用温度/top-k/top-p 归一化后做树形拒绝采样
        from sgl_kernel import (
            top_k_renorm_prob,
            top_p_renorm_prob,
            tree_speculative_sampling_target_only,
        )

        from sglang.kernels.ops.speculative.reject_sampling import (
            chain_speculative_sampling_triton,
        )

        use_rejection_sampling = get_spec().speculative_use_rejection_sampling

        sampling_fn = (
            chain_speculative_sampling_triton
            if use_rejection_sampling
            else tree_speculative_sampling_target_only
        )

        # These full-vocabulary matrices are consumed by the sampling kernel
        # within this step. Returned tensors were allocated before the scope,
        # so the next CUDA graph replay may safely reclaim these borrowed bytes.
        with borrow_graph_pool(user="EAGLE probability borrow"):
            expanded_temperature = torch.repeat_interleave(
                sampling_info.temperatures, verify_input.draft_token_num, dim=0
            )  # (bs * num_draft_tokens, 1)

            target_probs = F.softmax(
                next_token_logits / expanded_temperature, dim=-1
            )  # (bs * num_draft_tokens, vocab_size)
            maybe_detect_nan(target_probs, "v2 verify: target_probs after softmax")
            if sampling_info.need_top_k_sampling:
                target_probs = top_k_renorm_prob(
                    target_probs,
                    torch.repeat_interleave(
                        sampling_info.top_ks, verify_input.draft_token_num, dim=0
                    ),
                )  # (bs * num_draft_tokens, vocab_size)
                maybe_detect_nan(
                    target_probs, "v2 verify: target_probs after top_k_renorm"
                )
            if sampling_info.need_top_p_sampling:
                target_probs = top_p_renorm_prob(
                    target_probs,
                    torch.repeat_interleave(
                        sampling_info.top_ps, verify_input.draft_token_num, dim=0
                    ),
                )
                maybe_detect_nan(
                    target_probs, "v2 verify: target_probs after top_p_renorm"
                )
            target_probs = target_probs.reshape(bs, verify_input.draft_token_num, -1)
            draft_probs = (
                verify_input.draft_probs
                if use_rejection_sampling
                else torch.zeros_like(target_probs)
            )
            # Defense-in-depth behind the spec_hook startup allowlist: validate
            # the actual kernel inputs before the Triton kernel.
            if use_rejection_sampling and (
                draft_probs is None or draft_probs.shape[-1] != target_probs.shape[-1]
            ):
                raise ValueError(
                    "Rejection sampling requires a target-vocab draft proposal "
                    "distribution; the current speculative algorithm/draft worker "
                    "does not produce one (draft_probs missing or vocab-mismatched)."
                )

            coins, coins_for_final_sampling = _verify_coins(
                sampling_info=sampling_info,
                seq_lens=batch.seq_lens,
                draft_token_num=verify_input.draft_token_num,
                candidates=candidates,
                device=device,
            )
            sampling_fn(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=num_correct_drafts,  # mutable
                candidates=candidates,
                # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
                retrive_index=verify_input.retrieve_index,
                retrive_next_token=verify_input.retrieve_next_token,
                retrive_next_sibling=verify_input.retrieve_next_sibling,
                uniform_samples=coins,
                uniform_samples_for_final_sampling=coins_for_final_sampling,
                target_probs=target_probs,
                draft_probs=draft_probs,
                threshold_single=get_spec().speculative_accept_threshold_single,
                threshold_acc=get_spec().speculative_accept_threshold_acc,
                deterministic=True,
            )
            del (
                expanded_temperature,
                target_probs,
                draft_probs,
                coins,
                coins_for_final_sampling,
            )

        # Sync sampling results across TP ranks: different GPUs may
        # produce slightly different target_probs due to floating-point
        # non-determinism in softmax/top_k/top_p, causing different
        # sampled tokens. Broadcast from rank 0 to ensure consistency.
        tp_group = (
            get_parallel().attn_tp_group
            if is_dp_attention_enabled()
            else get_tp_group()
        )
        if tp_group.world_size > 1:
            tp_group.broadcast(predict, src=0)
            tp_group.broadcast(accept_index, src=0)
            tp_group.broadcast(num_correct_drafts, src=0)

    if SIMULATE_ACC_LEN > 0:
        # Do simulation. The helper builds (and returns) a replacement
        # accept_index of width spec_steps + 1, so pass max_tree_depth - 1
        # to keep the simulated width identical to the real one.
        if SIMULATE_ACC_TOKEN_MODE not in ("fixed", "real-draft-token"):
            raise ValueError(
                "Invalid SGLANG_SIMULATE_ACC_TOKEN_MODE "
                f"{SIMULATE_ACC_TOKEN_MODE!r}; expected 'fixed' or "
                "'real-draft-token'."
            )

        if SIMULATE_ACC_TOKEN_MODE == "real-draft-token":
            if verify_input.tree_topk != 1:
                raise ValueError(
                    "SGLANG_SIMULATE_ACC_LEN with real draft tokens currently "
                    "requires speculative_eagle_topk=1."
                )

            # Use target argmax as the synthetic bonus for non-greedy requests.
            if target_predict is None:
                target_predict = torch.argmax(next_token_logits, dim=-1).reshape(
                    bs, verify_input.draft_token_num
                )
        accept_index = generate_simulated_accept_index(
            accept_index=accept_index,
            predict=predict,  # mutable
            num_correct_drafts=num_correct_drafts,  # mutable
            candidates=candidates,
            target_predict=target_predict,
            simulate_acc_len=SIMULATE_ACC_LEN,
            simulate_acc_token_mode=SIMULATE_ACC_TOKEN_MODE,
            bs=bs,
            spec_steps=verify_input.max_tree_depth - 1,
        )

    # `num_correct_drafts` 在函数内部始终只计草稿数（不含 bonus）；
    # 返回时通过 out-of-place +1 加上 bonus token，使返回值语义为「接受长度（含 bonus）」
    # 这样命名语义不会在函数中途翻转（命名规范 C2）
    return predict, num_correct_drafts + 1, accept_index


def eagle_prepare_for_decode(batch: ScheduleBatch):
    batch.maybe_evict_swa()

    bs = batch.batch_size()

    # Accumulate penalty
    # This is a relaxed version of penalties for speculative decoding.
    if batch.sampling_info.penalizer_orchestrator.is_required:
        batch.cumulate_penalty_output_tokens()

    page_size = batch.token_to_kv_pool_allocator.page_size
    double_alloc = get_alloc_reserve_per_decode()

    cur_kv_lens, nxt_kv_lens, num_needed_tokens = page_aligned_decode_alloc_lens(
        batch.reqs,
        reserve=double_alloc,
        page_size=page_size,
    )
    for r in batch.reqs:
        r.decode_batch_idx += 1

    cur_kv_lens_cpu = torch.tensor(cur_kv_lens, dtype=torch.int32, device="cpu")
    nxt_kv_lens_cpu = torch.tensor(nxt_kv_lens, dtype=torch.int32, device="cpu")

    # Fail fast if the page>1 + topk>1 draft over-allocation
    # (get_alloc_reserve_per_decode) outgrows the req_to_token row: the write below
    # would OOB and free would leak KV. The row is widened to hold it in _init_pools
    # (PR #26972); fail here with a clear error, not on a later cryptic CUDA assert.

    if page_size > 1 and (get_spec().speculative_eagle_topk or 1) > 1:
        max_alloc_len = int(nxt_kv_lens_cpu.max())
        row_width = batch.req_to_token_pool.req_to_token.shape[1]
        assert max_alloc_len <= row_width, (
            f"spec v2 page>1 topk>1 draft over-allocation ({max_alloc_len}) exceeds "
            f"req_to_token row width ({row_width}); page_size={page_size}. Widen the "
            f"row to hold committed + get_alloc_reserve_per_decode (PR #26972)."
        )

    # non_blocking H2D: a blocking .to() syncs the schedule stream, which the WAR
    # barrier has chained to the prev forward -> host stalls a full forward.
    cur_kv_lens_device = cur_kv_lens_cpu.to(device=batch.device, non_blocking=True)
    nxt_kv_lens_device = nxt_kv_lens_cpu.to(device=batch.device, non_blocking=True)
    tree_cache = batch.tree_cache
    req_to_token_pool = batch.req_to_token_pool
    req_pool_indices = batch.req_pool_indices
    reqs = batch.reqs
    cur_kv_lens = cur_kv_lens_device
    nxt_kv_lens = nxt_kv_lens_device
    alloc_for_spec_decode(
        tree_cache,
        req_to_token_pool,
        reqs=reqs,
        req_pool_indices=req_pool_indices,
        cur_kv_lens=cur_kv_lens,
        cur_kv_lens_cpu=cur_kv_lens_cpu,
        nxt_kv_lens=nxt_kv_lens,
        nxt_kv_lens_cpu=nxt_kv_lens_cpu,
        num_needed_tokens=num_needed_tokens,
        batch=batch,
    )
