from dataclasses import dataclass
from itertools import accumulate
from typing import Callable, List

import torch
import torch.nn.functional as F
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import (
    attn_cp_all_gather_into_tensor,
    get_attention_cp_group,
    get_attention_cp_rank,
    get_attention_cp_size,
    is_allocation_symmetric,
)
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.server_args import get_global_server_args


@dataclass
class ContextParallelMetadata:
    """描述一次 zigzag CP 切分、通信与还原所需的全部元数据。

    设批大小为 ``bs``、CP 大小为 ``cp_size``，每条序列会先被切成
    ``2 * cp_size`` 个逻辑块。该对象同时记录：原始序列如何切块、当前
    rank 选中哪些首尾块、all-gather 后如何恢复原顺序，以及变长
    FlashAttention 所需的长度张量。

    示例：``bs=1, cp_size=2, seq_len=8`` 时，序列切成 4 个长度为 2 的块。
    CP rank 0 选择块 0 和块 3，rank 1 选择块 1 和块 2；通信完成后再通过
    ``cp_reverse_index`` 恢复为块 0、1、2、3。
    """

    # 布局列表长度均与 bs * cp_segment_num（即 bs * 2 * cp_size）相关。
    split_list: List[int] = None
    zigzag_index: List[int] = None
    cp_reverse_index: List[int] = None
    reverse_split_len: List[int] = None

    # 按 rank 聚合的列表长度为 cp_size。
    # max_rank_len 保存 cp_size 份 max(per_rank_actual_token)，保留列表形式是
    # 为了能直接作为 torch.split() 的分桶长度。
    per_rank_actual_token: List[int] = None
    max_rank_len: List[int] = None

    # 每条序列对应的 FlashAttention 张量，形状为 [bs] 或 [bs+1]。
    kv_len_prev_tensor: torch.Tensor = None  # [bs] int32 CUDA
    kv_len_next_tensor: torch.Tensor = None  # [bs] int32 CUDA
    actual_seq_q_prev_tensor: torch.Tensor = None  # [bs] int32 CUDA
    actual_seq_q_next_tensor: torch.Tensor = None  # [bs] int32 CUDA
    cu_seqlens_q_prev_tensor: torch.Tensor = None  # [bs+1] int32 CUDA
    cu_seqlens_q_next_tensor: torch.Tensor = None  # [bs+1] int32 CUDA

    # 从上述逐序列列表推导出的标量。
    total_q_prev_tokens: int = 0
    total_q_next_tokens: int = 0
    max_seqlen_q_prev: int = 0
    max_seqlen_q_next: int = 0

    # 逐序列 CPU 列表，供 NSA indexer 和诊断逻辑使用。
    kv_len_prev_list: List[int] = None
    kv_len_next_list: List[int] = None
    actual_seq_q_prev_list: List[int] = None
    actual_seq_q_next_list: List[int] = None

    # 当前批次中所有 extend_seq_lens 的总和。
    total_seq_lens: int = 0
    bs: int = 1


def is_prefill_context_parallel_enabled():
    """返回服务是否启用了 prefill 上下文并行。

    示例：启动参数包含 ``--enable-prefill-cp`` 时返回 ``True``，否则返回
    ``False``。
    """
    return get_global_server_args().enable_prefill_context_parallel


def is_prefill_cp_in_seq_split():
    """判断当前是否采用序列内 zigzag（in-seq-split）CP。

    示例：CP 已启用且 ``prefill_cp_mode='in-seq-split'`` 时返回 ``True``；
    采用 round-robin 或未启用 CP 时返回 ``False``。
    """
    return (
        is_prefill_context_parallel_enabled()
        and get_global_server_args().prefill_cp_mode == "in-seq-split"
    )


def get_cp_padding_align_size() -> int:
    """返回 CP 对全局 token 数进行 padding 时所需的对齐粒度。

    zigzag（in-seq-split）需要切出 ``2 * cp_size`` 个逻辑块，因此对齐到
    ``2 * cp_size``；其他 CP 策略对齐到 ``cp_size``。未开启 CP 时
    ``cp_size=1``，padding 等价于空操作。不要额外 padding，否则会破坏
    EAGLE/MTP draft prefill，参见 #23269。该辅助函数用于保证
    ``prepare_mlp_sync_batch`` 与 ``cal_padded_tokens`` 的规则一致。

    示例：zigzag 且 ``cp_size=4`` 时返回 8；round-robin 且
    ``cp_size=4`` 时返回 4；未启用 CP 时返回 1。
    """
    from sglang.srt.layers.attention.dsa.utils import is_dsa_prefill_cp_in_seq_split

    attn_cp_size = get_attention_cp_size()
    if is_prefill_cp_in_seq_split() or is_dsa_prefill_cp_in_seq_split():
        return attn_cp_size * 2
    return attn_cp_size


def is_mla_prefill_cp_enabled() -> bool:
    """判断 MLA 后端是否启用了 prefill CP。

    仅当全局 prefill CP 与 MLA backend 同时开启时返回 ``True``。
    """
    sa = get_global_server_args()
    return sa.enable_prefill_context_parallel and sa.use_mla_backend


def mla_use_prefill_cp(forward_batch, mla_enable_prefill_cp=None):
    """判断当前 MLA 前向批次是否真正走 prefill CP 路径。

    除服务级开关外，还要求批次已有 ``attn_cp_metadata``，且前向模式为
    context-parallel extend。

    示例：metadata 非空、模式为 CP_EXTEND、MLA CP 开启时返回 ``True``；
    decode 批次即使服务开启 CP 也返回 ``False``。
    """
    if mla_enable_prefill_cp is None:
        mla_enable_prefill_cp = is_mla_prefill_cp_enabled()
    return (
        forward_batch.attn_cp_metadata is not None
        and mla_enable_prefill_cp
        and forward_batch.forward_mode.is_context_parallel_extend()
    )


def can_cp_split(seq_len: int, cp_size: int, forward_batch):
    """判断当前批次能否执行 zigzag CP 切分。

    要求 CP 已开启、``cp_size > 1``、当前是纯 CP extend/prefill，并且每条
    序列至少能切出 ``2 * cp_size`` 个非空逻辑块。若 radix cache 命中后
    剩余 extend token 太少，则安全回退到普通 prefill。

    示例：``cp_size=2``、纯 CP extend、每条 extend 长度均为 8 时返回
    ``True``；某条序列只剩 3 个 extend token 时返回 ``False``。
    """
    # 基础条件：必须启用 CP、规模大于 1，并处于 CP extend（prefill）步骤。
    # seq_len // (cp_size * 2) 用于保证负载均衡所需的 2 * cp_size 个块非空。
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    cur_cp_seq_len = seq_len // (cp_size * 2)
    if not (
        cur_cp_seq_len != 0
        and cp_size > 1
        # prepare_context_parallel_metadata 固定了 bs_per_cp_group = 1；这里显式
        # 防护，避免 continuous batching 下发生静默的错误分区。
        and forward_batch.forward_mode.is_context_parallel_extend()
        # is_context_parallel_extend() 对 MIXED（同一步同时 prefill + decode）也
        # 返回 True，但 zigzag 切分只适用于纯 extend。
        and forward_batch.forward_mode != ForwardMode.MIXED
        and is_prefill_context_parallel_enabled()
    ):
        return False

    # bs > 1 时逐序列检查：每条序列都必须足够长，才能切成 2 * cp_size 份。
    # 支持逐序列 radix-cache 前缀；prepare_context_parallel_metadata 会通过
    # prefix_offsets[s] 将前缀计入 kv_len_prev/next。
    extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if extend_lens is None:
        return True

    cp_min = cp_size * 2
    for L in extend_lens:
        if L < cp_min:
            # 长度不足的请求无法 zigzag 切成 2 * cp_size 个块；本批次回退到
            # 普通（非 CP）prefill，而不是报错。典型情况是 radix-cache 命中
            # 大部分前缀后，只剩少量独有的 extend token。
            return False

    return True


def cp_split_and_rebuild_data(forward_batch, input_: torch.Tensor):
    """按当前 CP 策略切分并重排 token 数据。

    zigzag 模式先按 ``split_list`` 将首维切成 ``2 * cp_size`` 个块，再按
    当前 rank 的 ``zigzag_index`` 取一个前段块和一个后段块。round-robin
    模式则按 CP rank 交错抽取。

    示例：输入 token 为 ``[0,1,2,3,4,5,6,7]``、``cp_size=2`` 时，
    zigzag rank 0 输出 ``[0,1,6,7]``，rank 1 输出 ``[2,3,4,5]``；输出
    hidden 维保持不变。
    """
    from sglang.srt.layers.attention.dsa.utils import (
        dsa_cp_round_robin_split_data,
        is_dsa_prefill_cp_round_robin_split,
    )

    if is_dsa_prefill_cp_round_robin_split():
        cp_size = get_attention_cp_size()
        assert (
            input_.shape[0] % cp_size == 0
        ), f"输入张量第 0 维必须能被 CP 大小整除，实际形状为 {input_.shape}，CP 大小为 {cp_size}"
        return dsa_cp_round_robin_split_data(input_)

    input_list = list(
        torch.split(input_, forward_batch.attn_cp_metadata.split_list, dim=0)
    )
    result = torch.cat(
        [input_list[i] for i in forward_batch.attn_cp_metadata.zigzag_index], dim=0
    ).view(-1, input_.shape[-1])
    return result


def cp_split_and_rebuild_position(forward_batch, positions: torch.Tensor):
    """使用与 token 数据相同的 CP 规则切分并重排位置编号。

    position 必须与 hidden/token 保持完全相同的排列，否则 RoPE 和 causal
    mask 会对应错误的 token。

    示例：positions 为 ``[0,1,2,3,4,5,6,7]``、``cp_size=2`` 时，
    zigzag rank 0 输出 ``[0,1,6,7]``，rank 1 输出 ``[2,3,4,5]``。
    """
    from sglang.srt.layers.attention.dsa.utils import (
        dsa_cp_round_robin_split_data,
        is_dsa_prefill_cp_round_robin_split,
    )

    if is_dsa_prefill_cp_round_robin_split():
        cp_size = get_attention_cp_size()
        assert positions.shape[0] % cp_size == 0, (
            f"位置张量第 0 维必须能被 CP 大小整除，实际形状为 {positions.shape}，"
            f"CP 大小为 {cp_size}"
        )
        return dsa_cp_round_robin_split_data(positions)

    position_id_list = list(
        torch.split(positions, forward_batch.attn_cp_metadata.split_list, dim=-1)
    )
    positions = torch.cat(
        [position_id_list[i] for i in forward_batch.attn_cp_metadata.zigzag_index],
        dim=-1,
    )
    return positions


def cp_round_robin_input_ids(input_ids):
    """按 CP rank 对 input_ids 做 round-robin 重排或切分。

    输入示例（``cp_size=8``）：所有 rank 初始均看到
    ``[0,1,2,3,4,5,...]``。

    输出示例：
      - 未启用 MoE A2A 时，每个 rank 保留全部 token，但重排为
        ``[0,8,16,...,1,9,17,...,2,10,18,...]``；
      - 启用 MoE A2A 时，每个 rank 仅保留自己的交错分片：rank 0 为
        ``[0,8,16,...]``，rank 1 为 ``[1,9,17,...]``，依此类推。
    """
    cp_size = get_attention_cp_size()
    cp_rank = get_attention_cp_rank()
    if get_moe_a2a_backend().is_none():
        input_ids = input_ids.reshape(-1, cp_size).T.flatten()
    else:
        input_ids = input_ids[cp_rank::cp_size].contiguous()
    return input_ids


def cp_all_gather_reorganized_into_tensor(input_tensor, cp_size, forward_batch, stream):
    """对二维 CP 张量执行异步 all-gather，并去除 padding 后拼接。

    可用于 KV cache、index_k、hidden_states 等形如 ``[token, hidden]`` 的
    张量。流程为：
      1. 将各 rank 的 token 数补齐到共同的 ``max_len``；
      2. 异步 all-gather；
      3. 按 ``per_rank_actual_token`` 去除各 rank padding 并重新拼接。

    示例：两个 rank 的有效形状分别为 ``[3,128]``、``[5,128]``，通信前
    都补齐为 ``[5,128]``；all-gather 得到 ``[10,128]``，去 padding 后
    返回 ``[8,128]``。
    """
    max_len = forward_batch.attn_cp_metadata.max_rank_len[0]
    pad_size = max_len - input_tensor.shape[0]
    if pad_size > 0:
        input_tensor = F.pad(
            input_tensor, (0, 0, 0, pad_size), mode="constant", value=0
        )
    with use_symmetric_memory(
        get_attention_cp_group(), disabled=not is_allocation_symmetric()
    ):
        input_tensor_full = torch.empty(
            max_len * cp_size,
            input_tensor.shape[1],
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )

    get_attention_cp_group().cp_all_gather_into_tensor_async(
        input_tensor_full, input_tensor, stream
    )

    outputs_list_max = list(
        torch.split(
            input_tensor_full, forward_batch.attn_cp_metadata.max_rank_len, dim=0
        )
    )
    outputs = torch.cat(
        [
            outputs_list_max[index][:per_rank_len]
            for index, per_rank_len in enumerate(
                forward_batch.attn_cp_metadata.per_rank_actual_token
            )
        ],
        dim=0,
    )

    return outputs


def cp_all_gather_reorganized_into_tensor_kv_cache(
    input_tensor, cp_size, forward_batch, stream
):
    """对多维 CP KV cache 执行 all-gather，并去除首维 padding。

    与 ``cp_all_gather_reorganized_into_tensor`` 相同，但保留 token 维之后的
    任意维度，例如 ``[seq_len, num_heads, head_dim]``。

    示例：两个 rank 的有效形状分别为 ``[3,8,128]``、``[5,8,128]``，
    补齐并 gather 后去除 padding，返回形状 ``[8,8,128]``。
    """
    max_len = forward_batch.attn_cp_metadata.max_rank_len[0]
    pad_size = max_len - input_tensor.shape[0]
    if pad_size > 0:
        # 只填充首维（seq_len）。F.pad 的参数按维度逆序排列；n 维张量需要
        # 2*n 个值，因此仅填充首维时使用 [0, 0] * (ndim - 1) + [0, pad_size]。
        padding = [0, 0] * (input_tensor.ndim - 1) + [0, pad_size]
        input_tensor = F.pad(input_tensor, padding, mode="constant", value=0)

    # 创建保留其余所有维度的 all-gather 输出张量。
    with use_symmetric_memory(
        get_attention_cp_group(), disabled=not is_allocation_symmetric()
    ):
        input_tensor_full = torch.empty(
            max_len * cp_size,
            *input_tensor.shape[1:],
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )

    get_attention_cp_group().cp_all_gather_into_tensor_async(
        input_tensor_full, input_tensor, stream
    )

    outputs_list_max = list(
        torch.split(
            input_tensor_full, forward_batch.attn_cp_metadata.max_rank_len, dim=0
        )
    )
    outputs = torch.cat(
        [
            outputs_list_max[index][:per_rank_len]
            for index, per_rank_len in enumerate(
                forward_batch.attn_cp_metadata.per_rank_actual_token
            )
        ],
        dim=0,
    )

    return outputs


def cp_all_gather_rerange_output(input_tensor, cp_size, forward_batch, stream):
    """all-gather 各 CP rank 的二维输出，并恢复原始 token 顺序。

    输入/输出示例（in-seq-split，CP=4）：
    |   +-----------all-gather 前--------------+|
    |   | dp_atten_tp0: block0, block7 |
    |   | dp_atten_tp1: block1, block6 |
    |   | dp_atten_tp2: block2, block5 |
    |   | dp_atten_tp3: block3, block4 |
    |
    |   +----------恢复顺序前-------------------+|
    | block0 | block7 | block1 | block6 | block2 | block5 | block3 | block4 |
    |
    |   +--------------输出结果-----------------+
    | block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7 |
    |   +-------------------------+

    输入/输出示例（round-robin-split）：
    |   +-----------all-gather 前--------------+|
    | dp_atten_tp0: token0, token4, token8, token12, token16, ... |
    | dp_atten_tp1: token1, token5, token9, token13, token17, ... |
    | dp_atten_tp2: token2, token6, token10, token14, token18, ... |
    | dp_atten_tp3: token3, token7, token11, token15, token19, ... |
    |
    |   +--------------输出结果-----------------+
    | token0, token1, token2, token3, token4, token5, token6, token7, ...
    |   +-------------------------+
    """
    from sglang.srt.layers.attention.dsa.utils import (
        is_dsa_prefill_cp_round_robin_split,
    )

    if is_dsa_prefill_cp_round_robin_split():
        with use_symmetric_memory(
            get_attention_cp_group(), disabled=not is_allocation_symmetric()
        ):
            output_tensor = input_tensor.new_empty(
                (input_tensor.shape[0] * cp_size, *input_tensor.shape[1:]),
            )
        attn_cp_all_gather_into_tensor(
            output_tensor,
            input_tensor,
        )
        out_shape = output_tensor.shape
        output_tensor = (
            output_tensor.view(cp_size, -1, *out_shape[1:])
            .transpose(0, 1)
            .reshape(out_shape)
        )
        return output_tensor

    # TODO：这里是否需要去除 padding？
    bs_seq_len, hidden_size = input_tensor.shape
    output_tensor = cp_all_gather_reorganized_into_tensor(
        input_tensor,
        cp_size,
        forward_batch,
        stream,
    )
    outputs_list = list(
        torch.split(
            output_tensor, forward_batch.attn_cp_metadata.reverse_split_len, dim=0
        )
    )
    output_tensor = torch.cat(
        [outputs_list[i] for i in forward_batch.attn_cp_metadata.cp_reverse_index],
        dim=0,
    )
    output_tensor = output_tensor.view(-1, hidden_size)
    return output_tensor


def cp_all_gather_rerange_kv_cache(input_tensor, cp_size, forward_batch, stream):
    """all-gather CP 组全部 rank 的 KV cache，并恢复原始 token 顺序。

    支持 ``[seq_len, ...]`` 多维 KV cache，重排只作用于首个 token 维。

    输入/输出示例（in-seq-split，CP=4）：
    |   +-----------all-gather 前--------------+|
    |   | dp_atten_tp0: block0, block7 |
    |   | dp_atten_tp1: block1, block6 |
    |   | dp_atten_tp2: block2, block5 |
    |   | dp_atten_tp3: block3, block4 |
    |
    |   +----------恢复顺序前-------------------+|
    | block0 | block7 | block1 | block6 | block2 | block5 | block3 | block4 |
    |
    |   +--------------输出结果-----------------+
    | block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7 |
    |   +-------------------------+
    """
    output_tensor = cp_all_gather_reorganized_into_tensor_kv_cache(
        input_tensor,
        cp_size,
        forward_batch,
        stream,
    )
    outputs_list = list(
        torch.split(
            output_tensor, forward_batch.attn_cp_metadata.reverse_split_len, dim=0
        )
    )
    output_tensor = torch.cat(
        [outputs_list[i] for i in forward_batch.attn_cp_metadata.cp_reverse_index],
        dim=0,
    )
    # 无需 reshape：output_tensor 已经具有正确的 [seq_len, ...] 形状。
    return output_tensor


def cp_allgather_and_save_kv_cache(forward_batch, layer, k, v, cp_size, swa_loc=None):
    """汇聚所有 CP rank 的 KV cache，并写入每个 rank 的本地缓存池。

    K/V 会分别 all-gather、去 padding、恢复原 token 顺序，然后通过
    ``set_kv_buffer`` 写入当前层。``swa_loc`` 是 hybrid SWA pool 中预先
    完成 full→SWA 地址转换后的写入位置。

    示例：CP=2 时两个 rank 各持 4 个 token 的 K/V；执行后每个 rank 的
    本地缓存池都写入按原序排列的 8 个 token 的完整 K/V。
    """
    cache_loc = (
        forward_batch.out_cache_loc
        if not layer.is_cross_attention
        else forward_batch.encoder_out_cache_loc
    )

    k = k.contiguous()
    v = v.contiguous()

    key_cache_full = cp_all_gather_rerange_kv_cache(
        k, cp_size, forward_batch, torch.cuda.current_stream()
    )
    value_cache_full = cp_all_gather_rerange_kv_cache(
        v, cp_size, forward_batch, torch.cuda.current_stream()
    )

    get_token_to_kv_pool().set_kv_buffer(
        layer,
        KVWriteLoc(cache_loc, swa_loc),
        key_cache_full,
        value_cache_full,
        layer.k_scale,
        layer.v_scale,
    )


def cp_attn_forward_extend(
    forward_batch,
    q: torch.Tensor,
    device: torch.device,
    attn_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor],
) -> torch.Tensor:
    """分别计算 zigzag 前半块和后半块的 extend attention。

    根据 CP metadata 将 ``q`` 切成 prev/next 两部分，使用各自的变长序列
    元数据调用后端 ``attn_fn`` 两次，最后按原布局拼接结果。

    当 ``bs > 1`` 时，q 布局为
    ``[所有序列的 prev token, 所有序列的 next token]``，切分点是
    ``total_q_prev_tokens``。``cu_seqlens_q_prev/next`` 形状为 ``[bs+1]``，
    通过 FlashAttention 变长接口表达逐序列边界。

    ``attn_fn`` 签名：
    ``attn_fn(q, cu_seqlens_q, cache_seqlens, max_seqlen_q) -> result``。
    只有这四个参数随 CP 前/后半块变化，其他后端参数应由闭包捕获。

    示例：q 的 prev/next token 数分别为 3 和 5 时，函数依次以 ``q[:3]``、
    ``q[3:]`` 调用后端；若输出形状分别为 ``[3,H]``、``[5,H]``，最终返回
    形状 ``[8,H]``。
    """
    cp_meta = forward_batch.attn_cp_metadata

    q_prev = q[: cp_meta.total_q_prev_tokens]
    q_next = q[cp_meta.total_q_prev_tokens :]

    result_prev = attn_fn(
        q_prev,
        cp_meta.cu_seqlens_q_prev_tensor,
        cp_meta.kv_len_prev_tensor,
        cp_meta.max_seqlen_q_prev,
    )
    result_next = attn_fn(
        q_next,
        cp_meta.cu_seqlens_q_next_tensor,
        cp_meta.kv_len_next_tensor,
        cp_meta.max_seqlen_q_next,
    )

    return torch.concat([result_prev, result_next], dim=0)


def prepare_context_parallel_metadata(
    kv_len,
    cp_rank,
    cp_size,
    seqs_len,
    extend_seqs_len=None,
    device="cuda",
):
    from sglang.srt.layers.attention.dsa.utils import (
        is_dsa_prefill_cp_round_robin_split,
    )

    if is_dsa_prefill_cp_round_robin_split():
        return ContextParallelMetadata()

    """为 DSA zigzag CP 构造切分、通信和 attention 长度元数据。

    输入包括总 KV 长度、当前 CP rank、CP 大小、各序列总长度与 extend
    长度；输出 ``ContextParallelMetadata``，供 token 重排、KV all-gather、
    变长 FlashAttention 以及最终逆重排共同使用。

    单序列示例（``cp_size=4``）：
        block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7
        rank 0: block0, block7
        rank 1: block1, block6
        rank 2: block2, block5
        rank 3: block3, block4

    多序列时，每条序列独立切成 ``2 * cp_size`` 个块。rank r 的布局为：
        [s0.block_r, s1.block_r, ..., s_{bs-1}.block_r,
         s0.block_{2*cp_size-1-r}, ..., s_{bs-1}.block_{2*cp_size-1-r}]
    即先放所有序列的 prev 块，再放所有 next 块，因此可在
    ``total_q_prev_tokens`` 处一次切开。

    具体输入/输出示例：``kv_len=8, cp_rank=0, cp_size=2,
    seqs_len=[8], extend_seqs_len=[8]`` 时，输出包含
    ``split_list=[2,2,2,2]``、``zigzag_index=[0,3]``、
    ``per_rank_actual_token=[4,4]``；当前 rank 选中 block0 和 block3。
    """
    assert extend_seqs_len is not None
    extend_seqs_len = [int(x) for x in extend_seqs_len]

    # 将 padding token 计入最后一条序列，使 extend_seqs_len 与 kv_len 对齐。
    pad_len = int(kv_len) - sum(extend_seqs_len)
    if pad_len > 0:
        extend_seqs_len[-1] += pad_len
        if seqs_len is not None and len(seqs_len) == len(extend_seqs_len):
            seqs_len = list(seqs_len)
            seqs_len[-1] += pad_len

    bs = len(extend_seqs_len)
    cp_segment_num = cp_size * 2

    # 每条序列的前缀偏移（radix cache 命中长度）。非 NSA（FlashAttention）
    # 会把前缀通过 prefix_offsets[s] 计入 kv_len_prev/next，使
    # cache_seqlens 覆盖缓存前缀；NSA 保留不含前缀的累计值，由 indexer
    # 自行重新加上偏移。
    if seqs_len is not None and len(seqs_len) == bs:
        prefix_offsets = [
            max(int(seqs_len[s]) - extend_seqs_len[s], 0) for s in range(bs)
        ]
    else:
        prefix_offsets = [0] * bs

    # 逐序列计算块大小：前 L % cp_segment_num 个块各多分配 1 个 token。
    per_seq_block_sizes: List[List[int]] = []
    split_list: List[int] = []
    for s in range(bs):
        L = extend_seqs_len[s]
        base = L // cp_segment_num
        rem = L % cp_segment_num
        blk = [base + 1 if i < rem else base for i in range(cp_segment_num)]
        per_seq_block_sizes.append(blk)
        split_list.extend(blk)

    # 按 rank 聚合：rank r 拥有每条序列的块 r 与块
    # (2 * cp_size - 1 - r)，即一前一后两个 zigzag 块。
    per_rank_actual_token = [0] * cp_size
    for r in range(cp_size):
        total = 0
        for s in range(bs):
            total += (
                per_seq_block_sizes[s][r]
                + per_seq_block_sizes[s][cp_segment_num - 1 - r]
            )
        per_rank_actual_token[r] = total
    max_single_rank = max(per_rank_actual_token) if per_rank_actual_token else 0
    # 保存 cp_size 份相同值，使下游可直接调用 torch.split(x, max_rank_len)。
    max_rank_len = [max_single_rank] * cp_size

    # zigzag_index 从 split_list 的 bs * cp_segment_num 个块中选出当前 rank
    # 拥有的部分，顺序固定为 [所有 prev 块, 所有 next 块]。
    zigzag_index = list(
        range(cp_rank, cp_rank + bs * cp_segment_num, cp_segment_num)
    ) + list(
        range(
            cp_segment_num - cp_rank - 1,
            bs * cp_segment_num,
            cp_segment_num,
        )
    )

    # 逆重排索引：all-gather 后的数据顺序为
    # [rank0 的所有 prev, rank0 的所有 next, rank1 的所有 prev, ...]；
    # cp_reverse_index 将其恢复为 [s0_b0..s0_bN, s1_b0..s1_bN, ...]。
    cp_reverse_index: List[int] = []
    for batch_id in range(bs):
        cp_reverse_index.extend(
            list(range(batch_id, cp_segment_num * bs, 2 * bs))
            + list(
                range(
                    (cp_segment_num - 1) * bs + batch_id,
                    0,
                    -2 * bs,
                )
            )
        )

    # 与上述 all-gather 后拼接顺序对应的各块长度。
    reverse_split_len: List[int] = []
    for r in range(cp_size):
        for s in range(bs):
            reverse_split_len.append(per_seq_block_sizes[s][r])
        for s in range(bs):
            reverse_split_len.append(per_seq_block_sizes[s][cp_segment_num - 1 - r])

    # 供 FlashAttention cache_seqlens 使用的逐序列累计长度：
    # kv_len_prev[s] = 序列 s 的块 [0..cp_rank] 之和（含端点）；
    # kv_len_next[s] = 块 [0..cp_segment_num-cp_rank-1] 之和（含端点）。
    from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp

    nsa_mode = is_dsa_enable_prefill_cp()
    kv_len_prev_list: List[int] = []
    kv_len_next_list: List[int] = []
    actual_seq_q_prev_list: List[int] = []
    actual_seq_q_next_list: List[int] = []
    for s in range(bs):
        blk = per_seq_block_sizes[s]
        cum_prev = sum(blk[: cp_rank + 1])
        cum_next = sum(blk[: cp_segment_num - cp_rank])
        # NSA indexer 会自行加回前缀偏移，因此保留裸累计值；非 NSA
        #（FlashAttention）则直接把前缀计入 cache_seqlens。
        if nsa_mode:
            kv_len_prev_list.append(cum_prev)
            kv_len_next_list.append(cum_next)
        else:
            kv_len_prev_list.append(prefix_offsets[s] + cum_prev)
            kv_len_next_list.append(prefix_offsets[s] + cum_next)
        actual_seq_q_prev_list.append(blk[cp_rank])
        actual_seq_q_next_list.append(blk[cp_segment_num - cp_rank - 1])

    # 构造 FlashAttention 使用的 CUDA 长度张量；device 参数也便于其他设备调用。
    kv_len_prev_tensor = torch.tensor(
        kv_len_prev_list, device=device, dtype=torch.int32
    )
    kv_len_next_tensor = torch.tensor(
        kv_len_next_list, device=device, dtype=torch.int32
    )
    actual_seq_q_prev_tensor = torch.tensor(
        actual_seq_q_prev_list, device=device, dtype=torch.int32
    )
    actual_seq_q_next_tensor = torch.tensor(
        actual_seq_q_next_list, device=device, dtype=torch.int32
    )
    cu_prev = [0] + list(accumulate(actual_seq_q_prev_list))
    cu_next = [0] + list(accumulate(actual_seq_q_next_list))
    cu_seqlens_q_prev_tensor = torch.tensor(cu_prev, device=device, dtype=torch.int32)
    cu_seqlens_q_next_tensor = torch.tensor(cu_next, device=device, dtype=torch.int32)

    total_q_prev_tokens = cu_prev[-1]
    total_q_next_tokens = cu_next[-1]
    max_seqlen_q_prev = max(actual_seq_q_prev_list) if actual_seq_q_prev_list else 0
    max_seqlen_q_next = max(actual_seq_q_next_list) if actual_seq_q_next_list else 0
    total_seq_lens = sum(extend_seqs_len)

    # 基本不变量：metadata 必须构成合法的切分与排列描述。
    # - split_list 有 bs * cp_segment_num 项（所有序列的全部块）；
    # - zigzag_index 有 2 * bs 项（每条序列在当前 rank 的 prev + next）；
    # - cp_reverse_index 有 bs * cp_segment_num 项，可把完整 all-gather 流
    #   恢复为逐序列原始顺序。
    assert len(split_list) == bs * cp_segment_num
    assert sum(split_list) == total_seq_lens
    assert len(zigzag_index) == 2 * bs
    assert len(cp_reverse_index) == bs * cp_segment_num
    assert sorted(cp_reverse_index) == list(range(bs * cp_segment_num))
    assert sum(per_rank_actual_token) == total_seq_lens

    return ContextParallelMetadata(
        split_list=split_list,
        zigzag_index=zigzag_index,
        cp_reverse_index=cp_reverse_index,
        reverse_split_len=reverse_split_len,
        per_rank_actual_token=per_rank_actual_token,
        max_rank_len=max_rank_len,
        kv_len_prev_tensor=kv_len_prev_tensor,
        kv_len_next_tensor=kv_len_next_tensor,
        actual_seq_q_prev_tensor=actual_seq_q_prev_tensor,
        actual_seq_q_next_tensor=actual_seq_q_next_tensor,
        cu_seqlens_q_prev_tensor=cu_seqlens_q_prev_tensor,
        cu_seqlens_q_next_tensor=cu_seqlens_q_next_tensor,
        total_q_prev_tokens=total_q_prev_tokens,
        total_q_next_tokens=total_q_next_tokens,
        max_seqlen_q_prev=max_seqlen_q_prev,
        max_seqlen_q_next=max_seqlen_q_next,
        kv_len_prev_list=kv_len_prev_list,
        kv_len_next_list=kv_len_next_list,
        actual_seq_q_prev_list=actual_seq_q_prev_list,
        actual_seq_q_next_list=actual_seq_q_next_list,
        total_seq_lens=total_seq_lens,
        bs=bs,
    )
