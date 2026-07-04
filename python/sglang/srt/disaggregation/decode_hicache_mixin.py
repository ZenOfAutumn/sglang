"""PD 分离（prefill/decode disaggregation）中 decode 侧的 HiCache 集成 mixin。

在 PD 分离架构下，decode 实例收到远端 prefill 传来的 KV 之前，往往已在本地
缓存了部分前缀。HiCache 分三层：L1=device(GPU 显存)、L2=host(CPU 内存)、
L3=storage(外部存储后端)。本模块负责在 decode 侧「就地恢复」(local restore)：
把命中于 L2/L3 的前缀 KV 先加载回 device(load_back)，从而只需向 prefill
请求缺失部分，减少跨节点 KV 传输量。

核心流程：
  1. 前缀匹配：统计 L1/L2/L3 各命中多少 token(DecodePrefixMatch)。
  2. 预取：若 L3 有命中，先发起 L3->L2 预取。
  3. 恢复状态机：PENDING -> READY / FAILED，驱动 L2->L1 的 load_back DMA。
  4. 提交：把恢复出的 KV 索引写回 req，衔接后续 decode。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.managers.schedule_policy import match_prefix_for_req
from sglang.srt.mem_cache.base_prefix_cache import InitLoadBackParams

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode import DecodeRequest
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


@dataclass
class DecodePrefixMatch:
    # 一次前缀匹配的结果：记录该请求前缀分别命中于 L1/L2/L3 的情况。
    prefix_indices: torch.Tensor  # 已在 device(L1)上的前缀 KV 索引
    l2_host_hit_length: int  # 命中于 L2(host)的 token 数
    l3_storage_hit_length: int  # 命中于 L3(storage)的 token 数
    last_device_node: Any  # 前缀在 radix tree 中最后一个 device 节点
    last_host_node: Any = None  # 最后一个 host 节点(仅当 L3 有命中时才保留)
    prefetch_registered: bool = False  # 是否已成功登记 L3 预取

    @property
    def l1_prefix_len(self) -> int:
        # L1(device)命中长度，即已在显存上的前缀 token 数。
        return len(self.prefix_indices)

    @property
    def decode_prefix_len(self) -> int:
        # 三层合计的可复用前缀总长度(L1+L2+L3)。
        return self.l1_prefix_len + self.l2_host_hit_length + self.l3_storage_hit_length

    @property
    def needs_local_restore(self) -> bool:
        # 只要总前缀超过已在 device 上的部分，就需要做 L2/L3->L1 的本地恢复。
        return self.decode_prefix_len > self.l1_prefix_len

    @property
    def restore_token_count(self) -> int:
        """需要从 L2/L3 load_back 到 device 的 token 数。"""
        return self.decode_prefix_len - self.l1_prefix_len


class HiCacheRestoreResult(Enum):
    """HiCache 本地恢复状态机每次 tick 的结果。"""

    PENDING = "pending"  # 恢复进行中(等待预取/DMA 完成)
    READY = "ready"  # 恢复完成，可继续 decode
    FAILED = "failed"  # 恢复失败(如 device 分配失败)，需降级处理


class DecodeHiCachePreallocMixin:
    """挂到 ``DecodePreallocQueue`` 上的 HiCache 钩子：发起 L3 预取并为待恢复 token 预留额度。

    职责：在 decode 侧请求准入（admission）阶段，把前缀匹配结果转成 DecodePrefixMatch，
    并在需要时向 L3 存储发起预取；同时统计所有 pending 恢复请求占用的 device token 数，
    供准入时做容量核算。
    """

    def _build_decode_prxefix_match(self, req: Req, result: Any) -> DecodePrefixMatch:
        """把一次 ``match_prefix_for_req`` 的结果转换为 ``DecodePrefixMatch``。

        当 decode 侧 HiCache 已启用、且最后一个 host 节点已完成 backup 时，会额外查询
        L3 存储的命中长度（l3_storage_hit_length）。
        """
        prefix_indices = result.device_indices
        l1_prefix_len = len(prefix_indices)
        l2_host_hit_length = result.host_hit_length

        l3_storage_hit_length = 0
        last_host_node = None
        if self.scheduler.enable_decode_hicache:
            last_host_node = result.last_host_node
            # 只有当最后一个 host 节点已 backup（或就是根节点）时，其后缀才可能已写入 L3。
            if last_host_node.backuped or last_host_node is self.tree_cache.root_node:
                # matched_len：L1(device) + L2(host) 已覆盖的前缀长度；其后的 token 才需查 L3。
                matched_len = l1_prefix_len + l2_host_hit_length
                suffix_tokens = req.origin_input_ids[matched_len:]
                last_hash = last_host_node.get_last_hash_value()
                # 部分后端要求把前缀 hash 链一并传入以定位 key；由开关控制是否附带。
                prefix_keys = (
                    last_host_node.get_prefix_hash_values(last_host_node.parent)
                    if self.tree_cache.hicache_storage_pass_prefix_keys
                    else None
                )
                # 向 L3 查询后缀命中长度（此处只查询长度，不发起真正的数据预取）。
                l3_storage_hit_length = self.tree_cache.query_storage_hit_length(
                    last_host_node,
                    suffix_tokens,
                    last_hash,
                    prefix_keys,
                )

        return DecodePrefixMatch(
            prefix_indices=prefix_indices,
            l2_host_hit_length=l2_host_hit_length,
            l3_storage_hit_length=l3_storage_hit_length,
            last_device_node=result.last_device_node,
            # 只有 L3 确有命中时才记录 last_host_node，供后续 _start_hicache_prefetch 使用。
            last_host_node=last_host_node if l3_storage_hit_length > 0 else None,
        )

    def _start_hicache_prefetch(
        self, req: Req, prefix_match: Optional[DecodePrefixMatch]
    ) -> None:
        """在请求准入成功后，向 L3 存储发起真正的数据预取（L3->L2）。

        若预取发起失败，则清空 l3 相关字段，降级为「仅 L2->L1 恢复」。
        """
        # 没有 L3 命中（或未记录 host 节点）就无需预取，直接返回。
        if (
            prefix_match is None
            or prefix_match.l3_storage_hit_length <= 0
            or prefix_match.last_host_node is None
        ):
            return
        try:
            node = prefix_match.last_host_node
            # 取出 L3 命中区间对应的 token 后缀：从 L1+L2 已覆盖处开始，长度为 L3 命中长度。
            matched_len = prefix_match.l1_prefix_len + prefix_match.l2_host_hit_length
            suffix = req.origin_input_ids[
                matched_len : matched_len + prefix_match.l3_storage_hit_length
            ]
            last_hash = node.get_last_hash_value()
            prefix_keys = (
                node.get_prefix_hash_values(node.parent)
                if self.tree_cache.hicache_storage_pass_prefix_keys
                else None
            )
            # 发起异步预取，把 L3 数据搬到 L2(host)；实际完成情况后续通过 check_prefetch_progress 轮询。
            self.tree_cache.prefetch_from_storage(
                req.rid, node, suffix, last_hash, prefix_keys
            )
            # 记录预取是否真正登记成功（在 ongoing_prefetch 中即代表已登记）。
            prefix_match.prefetch_registered = (
                req.rid in self.tree_cache.ongoing_prefetch
            )
        except Exception as e:
            # 预取失败不致命：清空 L3 命中长度，退化为只做 L2->L1 的本地恢复。
            logger.warning(
                "HiCache L3 prefetch failed for rid=%s: %s; falling back to L2-only LoadingBack",
                req.rid,
                e,
            )
            prefix_match.l3_storage_hit_length = 0
            prefix_match.prefetch_registered = False

    def _hicache_pending_restore_tokens(self) -> int:
        """统计所有 pending 恢复请求为 L2/L3 load_back 预留的 device token 总数。

        准入新请求时需扣除这部分「已被预定但尚未落盘」的额度，避免超分配。
        """
        if not self.scheduler.enable_decode_hicache:
            return 0
        # 仅统计：仍处于 PENDING、且尚未拿到已恢复节点（restored_node is None）的请求。
        return sum(
            dr.prefix_match.restore_token_count
            for dr in self.transfer_queue.queue
            if dr.prefix_match is not None
            and dr.hicache_restore_status == HiCacheRestoreResult.PENDING
            and dr.hicache_restored_node is None
        )


class HiCacheRestoreGatedKVReceiver:
    """包装底层 kv_receiver：把 KVPoll.Success 的返回门控在「HiCache 恢复已 READY」之后。

    在 PD 分离场景下，KV 从 prefill 侧传输完成（kv_receiver 返回 Success）并不代表
    本地 L2/L3->L1 恢复也已完成。这里在恢复仍处于 PENDING 时，把 Success 改写成
    Transferring，从而阻止请求过早进入 decode。
    """

    def __init__(self, decode_req: DecodeRequest):
        self.decode_req = decode_req

    def poll(self) -> KVPoll:
        poll = self.decode_req.kv_receiver.poll()
        # KV 传输虽已成功，但本地恢复还没完成 -> 对外仍报「传输中」，延后放行。
        if (
            poll == KVPoll.Success
            and self.decode_req.hicache_restore_status == HiCacheRestoreResult.PENDING
        ):
            return KVPoll.Transferring
        return poll


class DecodeHiCacheTransferMixin:
    """挂到 ``DecodeTransferQueue`` 上的 HiCache 钩子：驱动本地恢复状态机。

    负责推进「L3->L2 预取完成 -> L2->L1 load_back DMA -> 恢复 READY」的整套流程，
    并在请求提交或中止时释放锁引用、清理预取资源。
    """

    def _clean_hicache_prefetch_resources(self, decode_req: DecodeRequest) -> None:
        """清理某请求占用的预取/恢复资源（请求中止或结束时调用）。"""
        # 若曾登记过 L3 预取，通知 tree_cache 释放该 rid 对应的预取资源。
        if (
            decode_req.prefix_match is not None
            and decode_req.prefix_match.prefetch_registered
        ):
            self.tree_cache.release_aborted_request(decode_req.req.rid)
        # 若已持有恢复节点的锁引用，递减引用计数并清空句柄。
        if decode_req.hicache_restored_node is not None:
            self.tree_cache.dec_lock_ref(decode_req.hicache_restored_node)
            decode_req.hicache_restored_node = None

    def _try_hicache_queue_load_back(self, dr: DecodeRequest) -> bool:
        """为 ``dr`` 排入一次 L2->L1 的 load_back 操作；当且仅当真正排入 DMA 时返回 True。

        成功排入时会填充 ``dr.hicache_restored_node`` 与 ``hicache_restored_kv_indices``，
        并持有一次 inc_lock_ref，直到 commit 或 abort 才释放。
        平凡情况（前缀已全在 device / 无需额外覆盖）会直接翻转为 READY 并返回 False；
        失败回退路径会翻转为 FAILED 并返回 False。
        """
        pm = dr.prefix_match

        # 若有 L3 命中，需先等待 L3->L2 预取排空后再继续（无 L3 命中则跳过）。
        if pm.l3_storage_hit_length > 0:
            if not self.tree_cache.check_prefetch_progress(dr.req.rid):
                return False
            self.tree_cache.pop_prefetch_loaded_tokens(dr.req.rid)

        # 重新匹配：此时 req.last_node / prefix_indices 已更新到当前 device 的最新状态。
        rematch = match_prefix_for_req(
            self.tree_cache,
            dr.req,
            dr.req.origin_input_ids,
            cow_mamba=False,
            include_req=True,
        )
        # 发起本地 load_back 准备：分配 device 槽位并返回待搬入的新索引及对应恢复节点。
        new_indices, restored_node = self.tree_cache.init_load_back(
            InitLoadBackParams(
                best_match_node=rematch.best_match_node,
                host_hit_length=rematch.host_hit_length,
                req=dr.req,
            )
        )
        # 失败回退：总覆盖长度 < 所需前缀，通常意味着 device 侧分配失败，标记 FAILED。
        if len(rematch.device_indices) + len(new_indices) < pm.decode_prefix_len:
            logger.warning(
                "HiCache load_back failed for rid=%s: device_indices=%d, "
                "new_indices=%d, expected decode_prefix_len=%d (l1=%d, l2=%d, l3=%d)",
                dr.req.rid,
                len(rematch.device_indices),
                len(new_indices),
                pm.decode_prefix_len,
                pm.l1_prefix_len,
                pm.l2_host_hit_length,
                pm.l3_storage_hit_length,
            )
            dr.hicache_restore_status = HiCacheRestoreResult.FAILED
            return False

        # 记录本次恢复需要写入的 device KV 索引：L1 之后重新匹配到的部分 + 新分配的槽位。
        dr.hicache_restored_kv_indices = torch.cat(
            [rematch.device_indices[pm.l1_prefix_len :], new_indices]
        )
        dr.hicache_restored_node = restored_node
        # 持锁，防止恢复节点在 DMA 完成前被回收；commit/abort 时再 dec_lock_ref 释放。
        self.tree_cache.inc_lock_ref(restored_node)

        if len(new_indices) == 0:
            # 整段前缀已在 device 上，无需 DMA，直接 READY。
            dr.hicache_restore_status = HiCacheRestoreResult.READY
            return False
        return True

    def _process_hicache_local_restores(self, decode_reqs: List[DecodeRequest]) -> None:
        """驱动本地恢复状态机：分三阶段推进 DMA（每个调度 tick 调用一次）。"""
        # 若 tree_cache 不支持 load_back 事件查询，说明未启用该能力，直接返回。
        if not hasattr(self.tree_cache, "is_load_back_event_done"):
            return

        # 先过滤一遍：只保留仍 PENDING 且确需恢复的请求；
        # 平凡完成的请求（无 prefix_match / 无需恢复）直接翻转为 READY。
        active: List[DecodeRequest] = []
        for dr in decode_reqs:
            if dr.hicache_restore_status != HiCacheRestoreResult.PENDING:
                continue
            pm = dr.prefix_match
            if pm is None or not pm.needs_local_restore:
                dr.hicache_restore_status = HiCacheRestoreResult.READY
                continue
            active.append(dr)

        # 阶段 A：把已在途（in-flight）且 DMA 已完成的请求推进为 READY。
        for dr in active:
            if (
                dr.hicache_restored_node is not None
                and self.tree_cache.is_load_back_event_done(
                    dr.hicache_load_consumer_index
                )
            ):
                dr.hicache_restore_status = HiCacheRestoreResult.READY

        # 阶段 B：若下一个槽位空闲，则为尚未排队的请求排入新的 load_back 操作。
        # (producer_index + 1) 的检查确保绝不覆盖仍在途的槽位：若上一个请求占着该槽且未完成，
        # 它的事件不会被 signal，从而在此被拦下、避免数据竞争。
        counter = self.tree_cache.cache_controller.layer_done_counter
        if not self.tree_cache.is_load_back_event_done(
            (counter.producer_index + 1) % counter.num_counters
        ):
            return
        queued = [
            dr
            for dr in active
            if dr.hicache_restored_node is None
            and self._try_hicache_queue_load_back(dr)
        ]
        if not queued:
            return

        # 阶段 C：触发合并后的 DMA，并绑定 consumer_index 供下一个 tick 的阶段 A 轮询。
        consumer_index = self.tree_cache.ready_to_load_host_cache()
        if consumer_index < 0:
            # < 0 表示无实际数据要搬（例如全部命中已在 device），直接置 READY。
            for dr in queued:
                dr.hicache_restore_status = HiCacheRestoreResult.READY
            return
        for dr in queued:
            dr.hicache_load_consumer_index = consumer_index

    def _commit_hicache_local_restore_to_req(self, decode_req: DecodeRequest) -> None:
        """恢复完成并提交给请求时，释放准入阶段持有的 device 节点锁引用。"""
        prefix_match = decode_req.prefix_match
        if prefix_match is None or not prefix_match.needs_local_restore:
            return

        # 释放在前缀匹配阶段对 last_device_node 持有的锁引用。
        self.tree_cache.dec_lock_ref(prefix_match.last_device_node)

        # 把恢复到 device 的 KV 索引写回 req_to_token 映射：覆盖 [L1, decode_prefix] 这段区间。
        self.tree_cache.req_to_token_pool.write(
            (
                decode_req.req.req_pool_idx,
                slice(prefix_match.l1_prefix_len, prefix_match.decode_prefix_len),
            ),
            decode_req.hicache_restored_kv_indices,
        )
        # 更新请求的前缀索引 = 原 L1 前缀 + 本次恢复补齐的部分，并把 last_node 指向恢复节点。
        decode_req.req.prefix_indices = torch.cat(
            [prefix_match.prefix_indices, decode_req.hicache_restored_kv_indices]
        )
        decode_req.req.last_node = decode_req.hicache_restored_node
