from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_alloc_reserve_per_decode,
    get_last_loc,
)

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_info import (
        EagleDraftInput,
    )


@dataclass
class EagleDraftInputV2Mixin:
    """EAGLE V2 草稿输入的混入（Mixin）类。

    为 EagleDraftInput / NgramVerifyInput 提供 V2 架构下的
    ``prepare_for_decode`` 方法，负责在每轮草稿前为所有请求分配
    KV cache 槽位（包括 bonus token 的预占）。

    示例::

        >>> # EagleDraftInput 继承此 Mixin 后：
        >>> draft_input.prepare_for_decode(batch)
        >>> # batch.tree_cache 中已为每个请求预分配了 double_alloc 个 KV 槽位
        >>> # batch.req_to_token_pool.req_to_token 已更新到新的 KV 位置
    """

    def prepare_for_decode(self: EagleDraftInput, batch: ScheduleBatch):
        """为草稿解码阶段分配 KV cache 槽位。

        为 batch 中每个请求预分配 ``get_alloc_reserve_per_decode()`` 个 KV 槽位
        （通常为 2 * alloc_len_per_decode），并预占 1 个 bonus slot。
        支持页大小为 1（无分页）和大于 1（分页）两种模式。

        输入:
            batch: 调度批次，包含 reqs、req_pool_indices、tree_cache 等

        输出:
            None（原地修改 batch 中的 tree_cache 和 req_to_token_pool）

        示例::

            >>> # 假设 bs=2，每个请求已分配 100 个 KV 槽位
            >>> # double_alloc = 16（每轮预留 16 个槽位）
            >>> # prepare_for_decode 后：
            >>> # - req0 的 kv_allocated_len: 100 → 116
            >>> # - req1 的 kv_allocated_len: 100 → 116
            >>> # - batch.tree_cache 中分配了 32 个新槽位
            >>> # - batch.req_to_token_pool.req_to_token 已填充这些槽位号
        """
        batch.maybe_evict_swa()

        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        bs = batch.batch_size()

        # 累加惩罚项
        # 这是投机解码的宽松版惩罚机制。
        if batch.sampling_info.penalizer_orchestrator.is_required:
            output_ids = torch.tensor(
                [
                    (
                        req.output_ids[-1]
                        if len(req.output_ids)
                        else req.origin_input_ids[-1]
                    )
                    for req in batch.reqs
                ],
                dtype=torch.int64,
                device=batch.device,
            )
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                output_ids
            )

        page_size = batch.token_to_kv_pool_allocator.page_size
        double_alloc = get_alloc_reserve_per_decode()

        cur_kv_lens = [0] * bs
        nxt_kv_lens = [0] * bs
        num_needed_tokens = 0
        for i, r in enumerate(batch.reqs):
            cur = r.kv_allocated_len
            # max(cur, ...) 做钳位，使自适应降级（更小的 alloc_len_per_decode）
            # 不会导致 nxt < cur 从而破坏分配器状态。
            # kv_committed_len 在 overlap 模式下落后 batch.seq_lens 约 1 个 verify，
            # 因此我们比基于 seq_lens 的基线晚一个 batch 响应自适应切换；
            # 2*alloc 的超额分配缓冲区吸收了这个延迟。
            nxt = max(cur, r.kv_committed_len + double_alloc)
            cur_kv_lens[i] = cur
            nxt_kv_lens[i] = nxt
            num_needed_tokens += nxt - cur
            r.kv_allocated_len = nxt
            r.decode_batch_idx += 1
            # 在此预占 bonus 槽位（与普通 decode 相同）；resolve 阶段减 1。
            r.kv_committed_len += 1

        cur_kv_lens_cpu = torch.tensor(cur_kv_lens, dtype=torch.int32, device="cpu")
        nxt_kv_lens_cpu = torch.tensor(nxt_kv_lens, dtype=torch.int32, device="cpu")

        # 快速失败检查：page>1 + topk>1 的草稿超额分配
        # （get_alloc_reserve_per_decode）若超出 req_to_token 行宽：
        # 下面的写入会 OOB 且释放会泄漏 KV。行宽已在 _init_pools 中
        # 扩展以容纳它（PR #26972）；在此处用清晰的错误信息失败，
        # 而非后续的晦涩 CUDA assert。
        from sglang.srt.server_args import get_global_server_args

        if page_size > 1 and (get_global_server_args().speculative_eagle_topk or 1) > 1:
            max_alloc_len = int(nxt_kv_lens_cpu.max())
            row_width = batch.req_to_token_pool.req_to_token.shape[1]
            assert max_alloc_len <= row_width, (
                f"spec v2 page>1 topk>1 草稿超额分配 ({max_alloc_len}) 超过 "
                f"req_to_token 行宽 ({row_width})；page_size={page_size}。"
                f"请扩宽行以容纳 committed + get_alloc_reserve_per_decode (PR #26972)。"
            )

        # non_blocking H2D：阻塞式 .to() 会同步调度流，而 WAR 屏障
        # 已将其链到前一次 forward → 主机停滞一个完整 forward。
        cur_kv_lens_device = cur_kv_lens_cpu.to(device=batch.device, non_blocking=True)
        nxt_kv_lens_device = nxt_kv_lens_cpu.to(device=batch.device, non_blocking=True)
        if page_size == 1:
            # 无分页：简单连续分配
            out_cache_loc = alloc_token_slots(batch.tree_cache, num_needed_tokens)
        else:
            # 分页模式：按页扩展分配
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                cur_kv_lens_device,
            )
            out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                cur_kv_lens_device,
                cur_kv_lens_cpu,
                nxt_kv_lens_device,
                nxt_kv_lens_cpu,
                last_loc,
                num_needed_tokens,
            )

        # 将分配的缓存位置写入 req_to_token 映射表
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            cur_kv_lens_device,
            nxt_kv_lens_device,
            out_cache_loc,
            bs,
        )
