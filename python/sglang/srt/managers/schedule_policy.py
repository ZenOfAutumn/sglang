from __future__ import annotations

import logging
from array import array

from sglang.srt.environ import envs
from sglang.srt.managers.prefill_delayer import PrefillDelayerSinglePassExecutor
from sglang.srt.utils import get_bool_env_var

_ROUTING_KEY_POLICY_DEBUG_LOG = get_bool_env_var("SGLANG_ROUTING_KEY_POLICY_DEBUG_LOG")
logger = logging.getLogger(__name__)

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Request scheduler policy

中译：请求调度策略。本模块决定「等待队列（waiting_queue）里的 prefill 请求按什么顺序排队、
      哪些请求能进入本轮 prefill 批次」。包含两大块：
      1) SchedulePolicy：对等待队列排序。支持「感知前缀缓存（cache-aware）」策略
         （LPM 最长前缀匹配、DFS-WEIGHT 深度优先权重）和「不感知缓存（cache-agnostic）」
         策略（FCFS 先到先服务、LOF 最长输出优先、RANDOM、ROUTING-KEY），并可叠加优先级调度。
      2) PrefillAdder：在 token 预算 / 显存（KV cache）约束下，逐个把请求加入本轮可运行列表
         （can_run_list），必要时做分块 prefill（chunked prefill）或抢占（preempt）低优先级请求。
"""

import os
import random
from collections import Counter, defaultdict
from contextlib import contextmanager
from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union

import torch

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.attention.dsa.utils import is_dsa_prefill_cp_in_seq_split
from sglang.srt.layers.utils.cp_utils import is_prefill_context_parallel_enabled
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
    zero_match_result,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.server_args import ServerArgs, get_global_server_args

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

# Clip the estimation of max_new_tokens for the request whose max_new_tokens is very large.
# This can prevent the server from being too conservative.
# Note that this only clips the estimation in the scheduler but does not change the stop
# condition. The request can still generate tokens until it hits the unclipped max_new_tokens.
# 中译：对 max_new_tokens 极大的请求，调度时把「预留显存的估计值」裁剪到此上限，避免服务过于保守
#       （为一个声称要生成几十万 token 的请求预留过多显存）。注意这只裁「估计值」、不改真正的停止
#       条件——请求仍可一直生成到它原始未裁剪的 max_new_tokens 为止。
CLIP_MAX_NEW_TOKENS = int(
    os.environ.get("SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION", "4096")
)

# Threshold for in-batch prefix cache.
# If a request has a matched prefix length (against existing cache) less than this value,
# the scheduler runs the in-batch prefix caching check for this request.
# If we set it to -1, it means we disable in-batch prefix caching.
# 中译：「批内前缀缓存（in-batch prefix caching）」的检查阈值。若一个请求对「已有缓存」的命中前缀
#       长度小于此值，调度器才对它做批内前缀检查（看等待队列里是否有别的请求和它共享前缀）。
#       设为 -1 表示禁用批内前缀缓存。
IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD = int(
    os.environ.get("IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD", "32")
)

# Threshold for in-batch prefix cache.
# If a request has a matched prefix length (within the waiting queue) larger than this value,
# the scheduler deprioritizes this request
# 中译：批内前缀缓存的「降优先级」阈值。若一个请求在「等待队列内部」匹配到的共享前缀长度超过此值，
#       说明已有别的请求会先把该前缀写进缓存，于是把本请求暂时降优先级，让先行者跑完以提升缓存命中率。
IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD = int(
    os.environ.get("IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD", "32")
)


# 中译：ignore_eos 请求做显存估算时，每个请求额外保留的 token 数（用于角落情况的安全余量）。
IGNORE_EOS_RESERVE_TOKENS = 1


def match_prefix_for_req(
    tree_cache: BasePrefixCache,
    req: Req,
    token_ids: Optional[array[int]] = None,
    *,
    cow_mamba: bool = False,
    include_req: bool = False,
):
    """中译：在前缀缓存（RadixCache 等）里为单个请求做前缀匹配，并把匹配结果回填到 req 上。

    匹配结果包括：命中的 device 侧索引（prefix_indices，可复用的 KV cache，无需重算）、命中所在的
    缓存树节点（last_node/best_match_node 等）、host 侧（CPU 内存）命中长度（需 load back 回显存）。
    最终 req.num_matched_prefix_tokens 记录「总命中前缀长度」，是 LPM 等排序的关键依据。
    """
    # 中译：默认用「原始输入 + 已生成输出」作为匹配键（覆盖 prefill 续跑/重试场景）。
    if token_ids is None:
        token_ids = req.origin_input_ids + req.output_ids

    match_result = tree_cache.match_prefix(
        MatchPrefixParams(
            key=RadixKey(token_ids=token_ids, extra_key=req.extra_key),
            cow_mamba=cow_mamba,
            req=req if include_req else None,
        )
    )
    # 中译：调试开关——强制视为「未命中」（清零匹配结果），用于排查缓存相关问题。
    if envs.SGLANG_RADIX_FORCE_MISS.get():
        match_result = zero_match_result(tree_cache, match_result)
    # 中译：把匹配结果解包回填到 req 的各字段（prefix 索引、命中节点、各类 host 命中长度）。
    (
        req.prefix_indices,
        req.last_node,
        req.last_host_node,
        req.best_match_node,
        req.host_hit_length,
        req.swa_host_hit_length,
        req.mamba_host_hit_length,
    ) = (
        match_result.device_indices,
        match_result.last_device_node,
        match_result.last_host_node,
        match_result.best_match_node,
        match_result.host_hit_length,
        match_result.swa_host_hit_length,
        match_result.mamba_host_hit_length,
    )
    # 中译：命中前缀不能等于整条序列（至少要留一个 token 重新计算，否则无法触发生成），
    #       故用 _compute_max_prefix_len 算出允许的最大前缀长度，再对总命中长度取 min 截断。
    max_len = req._compute_max_prefix_len(len(token_ids))
    req.num_matched_prefix_tokens = min(
        len(req.prefix_indices) + req.host_hit_length, max_len
    )
    if match_result.mamba_branching_seqlen is not None:
        req.mamba_branching_seqlen = match_result.mamba_branching_seqlen
    if match_result.cache_protected_len is not None:
        req.cache_protected_len = match_result.cache_protected_len
    return match_result


class CacheAwarePolicy(Enum):
    """Scheduling policies that are aware of the tree cache.

    中译：「感知前缀缓存树」的调度策略——排序时利用 RadixCache 的前缀命中信息，
          让能复用更多缓存的请求优先，从而提升整体缓存命中率、减少重复计算。
    """

    LPM = "lpm"  # longest prefix match
    # 中译：LPM——最长前缀匹配，命中前缀越长越优先。
    DFS_WEIGHT = "dfs-weight"  # depth-first search weighting
    # 中译：DFS-WEIGHT——按缓存树的深度优先遍历 + 子树请求数加权排序，让共享同一子树的请求聚在一起。


class CacheAgnosticPolicy(Enum):
    """Scheduling policies that are not aware of the tree cache.

    中译：「不感知前缀缓存」的调度策略——排序时不看缓存命中，只按到达顺序/输出长度/随机/路由键等。
    """

    FCFS = "fcfs"  # first come first serve
    # 中译：FCFS——先到先服务（按进入等待队列的时间）。
    LOF = "lof"  # longest output first
    # 中译：LOF——最长输出优先（按 max_new_tokens 降序）。
    RANDOM = "random"
    # 中译：RANDOM——随机打乱顺序。
    ROUTING_KEY = "routing-key"  # prioritize by routing key frequency in running batch
    # 中译：ROUTING-KEY——按「路由键」在运行批次中的出现频率排序（让同路由键的请求聚集，利于专家/缓存复用）。


class SchedulePolicy:
    """中译：等待队列排序器。根据所选策略（及是否启用优先级调度）对 waiting_queue 原地重排，
    决定 prefill 请求的处理顺序。"""

    Policy = Union[CacheAwarePolicy, CacheAgnosticPolicy]

    def __init__(
        self,
        policy: str,
        tree_cache: BasePrefixCache,
        enable_hierarchical_cache: bool,
        enable_priority_scheduling: bool,
        schedule_low_priority_values_first: bool,
    ):
        # 中译：校验策略名并按缓存开关调整（如缓存被禁用时把 cache-aware 策略退化为 FCFS）。
        self.policy = self._validate_and_adjust_policy(policy, tree_cache)
        self.tree_cache = tree_cache
        self.enable_hierarchical_cache = enable_hierarchical_cache
        self.enable_priority_scheduling = enable_priority_scheduling
        self.schedule_low_priority_values_first = schedule_low_priority_values_first
        # 中译：优先级排序的符号。priority * priority_sign 作为排序键：
        #       若「低优先级值优先（low first）」取 +1（升序，值小的排前）；否则取 -1（值大的排前）。
        self.priority_sign = 1 if schedule_low_priority_values_first else -1

        # It is used to find the matching prefix for in-batch prefix caching.
        # 中译：一棵「模拟的」临时 RadixCache，仅用于在「同一批等待队列内部」做前缀匹配，
        #       从而检测队列内请求间是否共享前缀（in-batch prefix caching），不影响真实缓存。
        self.waiting_queue_radix_tree = RadixCache.create_simulated()

    def calc_priority(
        self, waiting_queue: List[Req], running_batch: Optional[ScheduleBatch] = None
    ) -> None:
        # 中译：核心入口——按当前生效的策略对 waiting_queue 原地排序（队首即最优先处理）。
        policy = self._determine_active_policy(waiting_queue)

        # Populate req.num_matched_prefix_tokens at schedule time. Cache-aware policies
        # set it in _compute_prefix_matches; do the same full match for
        # cache-agnostic policies when the radix supports it, so the load
        # snapshot has it. Skip on decode (never prefills).
        # 中译：在调度时填充 req.num_matched_prefix_tokens。cache-aware 策略会在
        #       _compute_prefix_matches 里设置它；对 cache-agnostic 策略，只要 radix 支持快速匹配，
        #       也在这里做一次完整匹配，让「负载快照（load snapshot）」拿到该字段。decode 阶段不 prefill，跳过。
        if (
            not isinstance(policy, CacheAwarePolicy)
            and self.tree_cache.supports_fast_match_prefix()
            and get_global_server_args().disaggregation_mode != "decode"
        ):
            for r in waiting_queue:
                match_prefix_for_req(self.tree_cache, r)

        # 中译：FCFS（含被退化为 FCFS 的情况）。若启用优先级调度，则按 (优先级, 入队时间) 排序；否则保持原序直接返回。
        if self.policy == CacheAgnosticPolicy.FCFS:
            if self.enable_priority_scheduling:
                SchedulePolicy._sort_by_priority_and_fcfs(
                    waiting_queue, self.priority_sign
                )
            return

        if isinstance(policy, CacheAwarePolicy):
            # 中译：cache-aware 分支——先计算各请求前缀匹配（顺带得到被「批内降优先级」的请求集合），再据此排序。
            temporary_deprioritized = self._compute_prefix_matches(
                waiting_queue, policy
            )
            if policy == CacheAwarePolicy.LPM:
                SchedulePolicy._sort_by_longest_prefix(
                    waiting_queue, temporary_deprioritized
                )
            elif policy == CacheAwarePolicy.DFS_WEIGHT:
                SchedulePolicy._sort_by_dfs_weight(waiting_queue, self.tree_cache)
            else:
                raise ValueError(f"Unknown CacheAware Policy: {policy=}")
        else:
            # 中译：cache-agnostic 分支——按具体策略选择排序方式。
            if policy == CacheAgnosticPolicy.FCFS:
                pass
            elif policy == CacheAgnosticPolicy.LOF:
                SchedulePolicy._sort_by_longest_output(
                    waiting_queue,
                    self.enable_priority_scheduling,
                    self.priority_sign,
                )
            elif policy == CacheAgnosticPolicy.RANDOM:
                SchedulePolicy._sort_randomly(waiting_queue)
            elif policy == CacheAgnosticPolicy.ROUTING_KEY:
                if running_batch is not None:
                    SchedulePolicy._sort_by_routing_key(waiting_queue, running_batch)
            else:
                raise ValueError(f"Unknown CacheAgnostic Policy: {policy=}")

    def _determine_active_policy(self, waiting_queue: List[Req]) -> Policy:
        # 中译：决定本轮实际生效的策略。队列很大时（LPM 且超 128），前缀匹配+排序代价过高，临时退化为 FCFS。
        if self.policy == CacheAwarePolicy.LPM and len(waiting_queue) > 128:
            # Turn off the expensive prefix matching and sorting when the #queue is large.
            # 中译：队列过长时关闭昂贵的前缀匹配与排序，改用 FCFS 以控制调度开销。
            return CacheAgnosticPolicy.FCFS
        return self.policy

    def _validate_and_adjust_policy(
        self, policy: str, tree_cache: BasePrefixCache
    ) -> Policy:
        """
        Validates the policy and adjusts it if necessary based on tree cache settings.

        中译：校验策略名字符串并据缓存设置调整。先尝试解析为 cache-aware 策略；若缓存被禁用则
              退化为 FCFS；解析失败再尝试 cache-agnostic 策略；仍失败则报错。
        """
        try:
            policy_enum = CacheAwarePolicy(policy)
            if getattr(tree_cache, "disable", True):
                # If tree_cache is disabled, using CacheAgnosticPolicy policy
                # 中译：缓存树被禁用时，cache-aware 策略无意义，回退为 FCFS。
                return CacheAgnosticPolicy.FCFS
            return policy_enum
        except ValueError:
            # 中译：不是 cache-aware 策略名，转而尝试 cache-agnostic；都不是则抛错。
            try:
                return CacheAgnosticPolicy(policy)
            except ValueError:
                raise ValueError(f"Unknown schedule_policy: {policy=}")

    def _compute_prefix_matches(
        self, waiting_queue: List[Req], policy: CacheAwarePolicy
    ) -> Set[int]:
        """
        Computes and caches the matching prefixes for requests in the waiting queue,
            and handles in-batch prefix caching logic.

        中译：为等待队列中每个请求计算前缀匹配（回填到 req），并处理「批内前缀缓存」逻辑：
              当多个请求对已有缓存命中都很短、但它们彼此共享同一前缀时，先只调度其中一个，
              让它把该前缀写进缓存后，其余请求即可命中，从而提升整体命中率。
        返回：本轮被临时降优先级（推后）的请求 rid 集合。
        """
        temporary_deprioritized: Set[int] = set()
        # 中译：清空上一轮的「批内模拟前缀树」，本轮重新构建。
        self.waiting_queue_radix_tree.reset()

        for r in waiting_queue:
            prefix_ids = r.origin_input_ids + r.output_ids
            extra_key = r.extra_key
            # 中译：先对「真实缓存」做前缀匹配（回填 r.prefix_indices 等）。
            match_result = match_prefix_for_req(self.tree_cache, r, prefix_ids)

            # NOTE(sang): This logic is for in-batch prefix caching;
            # If there are more than 1 request that have small matching prefix from
            # existing cache, but all those requests share the same prefix, we prefer
            # to schedule only one of them so that we can increase the cache hit rate.
            # We prefer to set IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD > 0 because too small
            # threshold means we cannot use in-batch prefix caching for short prefixes.
            # It is kind of common when the engine is long running (e.g., imagine the prefix "the").
            # 中译：仅当对真实缓存命中较短（<= 检查阈值）时，才检查队列内部是否有共享前缀。
            if len(r.prefix_indices) <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:
                # 中译：在「批内模拟前缀树」里匹配——看本请求和此前已遍历的队列请求是否共享前缀。
                match_result = self.waiting_queue_radix_tree.match_prefix(
                    MatchPrefixParams(
                        key=RadixKey(token_ids=prefix_ids, extra_key=extra_key)
                    )
                )
                if envs.SGLANG_RADIX_FORCE_MISS.get():
                    match_result = zero_match_result(
                        self.waiting_queue_radix_tree, match_result
                    )
                in_batch_matching_prefixes = match_result.device_indices
                # 中译：若队列内共享前缀已足够长（>= 降优先级阈值），说明前面已有请求会写入该前缀，
                #       本请求降优先级推后；否则把本请求的前缀插入模拟树，供后续请求匹配。
                if (
                    len(in_batch_matching_prefixes)
                    >= IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD
                ):
                    temporary_deprioritized.add(r.rid)
                else:
                    # Insert with a dummy key
                    # 中译：用占位（dummy）value 插入——只关心前缀结构，不存真实 KV 索引。
                    self.waiting_queue_radix_tree.insert(
                        InsertParams(
                            key=RadixKey(token_ids=prefix_ids, extra_key=extra_key),
                            value=torch.empty(len(prefix_ids), dtype=torch.bool),
                        )
                    )
        return temporary_deprioritized

    @staticmethod
    def _sort_by_longest_prefix(
        waiting_queue: List[Req], temporary_deprioritized: Set[int]
    ) -> None:
        """Sorts the waiting queue based on the longest prefix match.

        中译：按「命中前缀长度」降序排序（命中越多越靠前，越能复用缓存）。被批内降优先级的请求
              排序键置为 +inf（升序排到最后）。
        """
        waiting_queue.sort(
            key=lambda r: (
                -r.num_matched_prefix_tokens
                if r.rid not in temporary_deprioritized
                else float("inf")
            )
        )

    @staticmethod
    def _sort_by_dfs_weight(
        waiting_queue: List[Req], tree_cache: BasePrefixCache
    ) -> None:
        """Sorts the waiting queue based on a depth-first search weighting.

        中译：按缓存树的 DFS 权重排序。思路：把每个请求挂到它命中的末端节点 last_node 上，
              每个节点的权重 = 子树内请求数；再对缓存树做 DFS（优先进入权重大的子树），
              遍历到某节点时把挂在它上面的请求依次入队。效果是共享越多前缀的请求越聚集。
        """
        # 中译：建立 last_node -> 命中该节点的请求列表 的映射。
        last_node_to_reqs = defaultdict(list)
        for req in waiting_queue:
            last_node_to_reqs[req.last_node].append(req)

        # 中译：初始化各末端节点的权重为「挂在其上的请求数」，再自底向上累加到祖先节点。
        node_to_weight = defaultdict(int)
        for node in last_node_to_reqs:
            node_to_weight[node] = len(last_node_to_reqs[node])
        SchedulePolicy._calc_weight(tree_cache.root_node, node_to_weight)

        # 中译：清空原队列，按 DFS 顺序重新填充（原地重排）。
        waiting_queue.clear()
        SchedulePolicy._get_dfs_priority(
            tree_cache.root_node,
            node_to_weight,
            last_node_to_reqs,
            waiting_queue,
        )

    @staticmethod
    def _sort_by_longest_output(
        waiting_queue: List[Req],
        enable_priority_scheduling: bool,
        priority_sign: int,
    ) -> None:
        """Sorts the waiting queue based on the longest output (max_new_tokens). If using priority scheduling, sort by priority first.

        中译：LOF——按最长输出（max_new_tokens 降序）排序。若启用优先级调度，则先按优先级、再按输出长度。
        """
        if enable_priority_scheduling:
            waiting_queue.sort(
                key=lambda x: (
                    x.priority * priority_sign,
                    -x.sampling_params.max_new_tokens,
                )
            )
        else:
            waiting_queue.sort(key=lambda x: -x.sampling_params.max_new_tokens)

    @staticmethod
    def _sort_randomly(waiting_queue: List[Req]) -> None:
        """Shuffles the waiting queue randomly.

        中译：RANDOM——随机打乱等待队列。
        """
        random.shuffle(waiting_queue)

    @staticmethod
    def _sort_by_priority_and_fcfs(
        waiting_queue: List[Req], priority_sign: int
    ) -> None:
        """Sorts the waiting queue based on the request priority then received titmestamp.

        中译：先按请求优先级（乘以 priority_sign 决定升/降序）、再按入队时间戳排序。
              即「同优先级内仍按 FCFS」，是 FCFS 叠加优先级调度时的排序方式。
        """
        waiting_queue.sort(
            key=lambda x: (
                x.priority * priority_sign,
                x.time_stats.wait_queue_entry_time,
            )
        )

    @staticmethod
    def _sort_by_routing_key(
        waiting_queue: List[Req], running_batch: ScheduleBatch
    ) -> None:
        """Sorts waiting queue by routing key frequency in running batch.

        中译：ROUTING-KEY——按「路由键在当前运行批次中出现的频率」排序。把路由键与正在运行的请求
              相同（且更频繁）的请求排前，利于 MoE 专家/缓存等的局部性复用。
        """
        # 中译：统计运行批次中各 routing_key 的出现次数。
        routing_key_counts = Counter(
            r.routing_key for r in running_batch.reqs if r.routing_key
        )

        if _ROUTING_KEY_POLICY_DEBUG_LOG:
            waiting_keys_before = [r.routing_key for r in waiting_queue]
            logger.info(
                f"routing_key_counts={dict(routing_key_counts)}, "
                f"waiting_keys_before={waiting_keys_before}"
            )

        if not routing_key_counts:
            return

        # 中译：排序键——命中运行批次路由键的请求归为第 0 组（按出现次数降序排前），其余归为第 1 组排后。
        def sort_key(req: Req):
            key = req.routing_key
            if key and key in routing_key_counts:
                count = routing_key_counts[key]
                return (0, -count, key)
            else:
                return (1, 0, key or "")

        waiting_queue.sort(key=sort_key)

        if _ROUTING_KEY_POLICY_DEBUG_LOG:
            waiting_keys_after = [r.routing_key for r in waiting_queue]
            logger.info(f"waiting_keys_after={waiting_keys_after}")

    @staticmethod
    def _calc_weight(cur_node: TreeNode, node_to_weight: Dict[TreeNode, int]) -> None:
        # 中译：后序遍历，把每个子节点的权重累加到父节点——使每个节点的权重等于其整棵子树内的请求总数。
        for child in cur_node.children.values():
            SchedulePolicy._calc_weight(child, node_to_weight)
            node_to_weight[cur_node] += node_to_weight[child]

    @staticmethod
    def _get_dfs_priority(
        cur_node: TreeNode,
        node_to_priority: Dict[TreeNode, int],
        last_node_to_reqs: Dict[TreeNode, List[Req]],
        q: List,
    ) -> None:
        # 中译：DFS 收集请求——子节点按权重降序优先递归，回到当前节点时把挂在它上面的请求追加入队 q。
        children = [child for child in cur_node.children.values()]
        children.sort(key=lambda x: -node_to_priority[x])
        for child in children:
            SchedulePolicy._get_dfs_priority(
                child, node_to_priority, last_node_to_reqs, q
            )
        q.extend(last_node_to_reqs[cur_node])


class AddReqResult(Enum):
    """中译：向本轮 prefill 批次添加请求的结果状态，用于告诉调用方是否还能继续添加。"""

    CONTINUE = auto()  # Continue to add requests
    # 中译：CONTINUE——预算充足，可继续添加下一个请求。
    NO_TOKEN = auto()  # No token left
    # 中译：NO_TOKEN——显存/KV token 预算耗尽，停止添加。
    OTHER = auto()  # Other reasons to stop adding requests
    # 中译：OTHER——因其他约束（如 max_prefill_tokens、分块预算、请求数上限等）停止添加。


class PrefillAdder:
    """中译：prefill 批次构造器。在多重预算约束（KV token 总量 / 输入 token / 分块 token / SWA 等）下，
    逐个尝试把等待队列中的请求加入本轮可运行列表（can_run_list），支持分块 prefill 与基于优先级的抢占。
    """

    def __init__(
        self,
        page_size: int,
        tree_cache: BasePrefixCache,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        running_batch: ScheduleBatch,
        new_token_ratio: float,
        rem_input_tokens: int,
        rem_chunk_tokens: Optional[int],
        num_mixed_decode_tokens: int = 0,
        priority_scheduling_preemption_threshold: int = 0,
        max_prefill_bs: int = 0,
        max_running_requests: Optional[int] = None,
        prefill_max_requests: Optional[int] = None,
        prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor] = None,
        dllm_config: Optional[DllmConfig] = None,
        waiting_queue_len: int = 0,
    ):
        self.page_size = page_size
        self.tree_cache = tree_cache
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.running_batch = running_batch
        # 中译：new_token_ratio——对运行中请求「剩余将生成 token 数」的折扣系数，用于估算需预留的显存
        #       （并非每个请求都会真生成到 max_new_tokens，故乘以一个 < 1 的比例避免过度保守）。
        self.new_token_ratio = new_token_ratio
        # 中译：本轮 prefill 还可摄入的「输入 token」预算（对应 max_prefill_tokens），先扣掉混合 decode 占用。
        self.rem_input_tokens = rem_input_tokens - num_mixed_decode_tokens
        # 中译：分块 prefill 的单轮 chunk token 预算；为 None 表示未启用分块 prefill。
        self.rem_chunk_tokens = rem_chunk_tokens
        self.dllm_config = dllm_config

        if self.dllm_config is not None:
            self._init_dllm_meta(dllm_config)

        if self.rem_chunk_tokens is not None:
            self.rem_chunk_tokens -= num_mixed_decode_tokens
        # 中译：两个「已占用偏移量」。预算的真实可用 = 物理可用+可驱逐 - offset。
        #       rem_total_token_offset 估计「总 KV 显存」占用（含为运行请求预留的未来生成空间）；
        #       cur_rem_token_offset 估计「当前这一步」实际要占用的显存。二者起点都先计入混合 decode token。
        self.rem_total_token_offset = num_mixed_decode_tokens
        self.cur_rem_token_offset = num_mixed_decode_tokens

        self.req_states = None  # ignore_eos 估算时用的 (剩余 token, 已占 token) 列表，惰性构建
        self.can_run_list = []  # 本轮被接纳、可运行的请求列表
        self.preempt_list = []  # 本轮被抢占（preempt）出运行批次的请求列表
        self.new_chunked_req = None  # 本轮因分块 prefill 未跑完、需续跑的请求
        self.log_hit_tokens = 0
        self.reprocessed_log_hit_tokens = 0
        # TODO(lsyin): report the real input tokens excluding page alignment
        self.log_input_tokens = 0
        self.reprocessed_log_input_tokens = 0

        if running_batch is not None:
            # Estimate the offset in the remaining token space
            # 中译：为「正在运行的请求」预留它们未来还会生成的 token 显存——累加到 total offset，
            #       使新进 prefill 请求的预算判断不会侵占运行请求的空间。
            self.rem_total_token_offset += sum(
                [
                    self._get_running_request_total_token_offset(r)
                    for r in running_batch.reqs
                ]
            )

        # DeepSeek V4 HiSparse wraps an SWATokenToKVPoolAllocator internally and
        # exposes the full SWA allocator interface.
        self.is_hybrid_swa = isinstance(
            self.token_to_kv_pool_allocator,
            (SWATokenToKVPoolAllocator, DeepSeekV4HiSparseTokenToKVPoolAllocator),
        )
        self.is_hybrid_ssm_cache = self.tree_cache.supports_mamba()

        self.rem_swa_token_offset = 0

        self.priority_scheduling_preemption_threshold = (
            priority_scheduling_preemption_threshold
        )
        self.dsa_prefill_cp_in_seq_split = is_dsa_prefill_cp_in_seq_split()
        self.max_running_requests = max_running_requests
        self.prefill_context_parallel_enabled = is_prefill_context_parallel_enabled()
        self.prefill_max_requests = prefill_max_requests
        self.prefill_delayer_single_pass = prefill_delayer_single_pass
        self.max_prefill_bs = max_prefill_bs
        # Snapshot of scheduler waiting_queue length at the start of this
        # prefill pass. Used by PrefillDelayer's queue-based trigger.
        # 中译：本轮 prefill 开始时等待队列长度的快照，供 PrefillDelayer 的「基于队列长度」的触发判断使用。
        self.waiting_queue_len = waiting_queue_len

    def _init_dllm_meta(self, dllm_config: DllmConfig):
        # 中译：初始化扩散式 LLM（dllm）相关预算：按块大小 * 最大并发请求数算出 dllm 的 token 预算。
        self.dllm_block_size = dllm_config.block_size
        max_running_reqs = dllm_config.max_running_requests

        self.rem_dllm_tokens = max_running_reqs * self.dllm_block_size

    def _get_running_request_total_token_offset(self, req: Req) -> int:
        # 中译：估算一个运行中请求未来还需预留的 token 显存：
        #       (max_new_tokens - 已生成数) 裁到 CLIP_MAX_NEW_TOKENS 上限，再乘以折扣系数 new_token_ratio。
        return (
            min(
                (req.sampling_params.max_new_tokens - len(req.output_ids)),
                CLIP_MAX_NEW_TOKENS,
            )
            * self.new_token_ratio
        )

    @property
    def rem_total_tokens(self):
        # 中译：剩余「总」可用 KV token 预算 = 物理可用 + 可驱逐（缓存中未锁定、可被淘汰复用的）- 已占用偏移。
        #       按是否混合 SWA / Mamba(SSM) 缓存走不同的可用量统计口径。
        if self.is_hybrid_swa:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.full_available_size()
                + self.tree_cache.full_evictable_size()
            )
        elif self.is_hybrid_ssm_cache:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.available_size()
                + self.tree_cache.full_evictable_size()
            )
        else:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.available_size()
                + self.tree_cache.evictable_size()
            )
        return available_and_evictable - self.rem_total_token_offset

    @property
    def rem_swa_tokens(self):
        # 中译：剩余 SWA（滑动窗口注意力）池的 token 预算，仅在 is_hybrid_swa 时有意义。
        return (
            self.token_to_kv_pool_allocator.swa_available_size()
            + self.tree_cache.swa_evictable_size()
            - self.rem_swa_token_offset
        )

    @property
    def cur_rem_tokens(self):
        # 中译：「当前这一步」剩余可用 token——与 rem_total_tokens 口径相同，但扣的是 cur_rem_token_offset
        #       （只算当前步实际占用，不含为运行请求预留的未来生成空间），用于本步实际能否放下的判断。
        if self.is_hybrid_swa:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.full_available_size()
                + self.tree_cache.full_evictable_size()
            )
        elif self.is_hybrid_ssm_cache:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.available_size()
                + self.tree_cache.full_evictable_size()
            )
        else:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.available_size()
                + self.tree_cache.evictable_size()
            )

        return available_and_evictable - self.cur_rem_token_offset

    def _swa_budget_for_req(
        self, extend_input_len: int, swa_host_hit_length: int = 0
    ) -> int:
        """SWA pool budget per request. Only valid when is_hybrid_swa is True.

        With chunked prefill + overlap scheduler, the peak SWA occupancy is:
          chunk N (running, not yet in tree) + sliding window (locked in tree)
          + chunk N+1 (new allocation)
        Since chunk N and locked tokens are already excluded from
        swa_available + swa_evictable, the budget only needs to cover the
        chunk N+1 allocation. We floor at sliding_window_size to reserve
        room for the decode phase.
        """
        if self.rem_chunk_tokens is not None:
            alloc = min(extend_input_len, self.rem_chunk_tokens)
        else:
            alloc = extend_input_len
        # 中译：预算下限取「滑动窗口大小」——保证为 decode 阶段留足窗口空间；再加一页对齐开销。
        budget = max(alloc, self.tree_cache.sliding_window_size) + self.page_size
        if swa_host_hit_length > 0:
            budget += self.ceil_paged_tokens(swa_host_hit_length)
        return budget

    def ceil_paged_tokens(self, tokens: int) -> int:
        # 中译：把 token 数向上取整到 page_size 的整数倍（分页分配器按页分配，需对齐）。
        return -(-tokens // self.page_size) * self.page_size

    def budget_state(self):
        # 中译：综合判断当前预算状态，返回 AddReqResult。优先级：先看总/当前 token（含 SWA）是否耗尽，
        #       再看输入 token 预算，最后看分块/dllm 预算。
        # 中译：总预算或当前步预算 <= 0 即视为无 token。
        no_token = self.rem_total_tokens <= 0 or self.cur_rem_tokens <= 0
        if not no_token and self.is_hybrid_swa:
            # 中译：混合 SWA 时还需检查 SWA 池是否耗尽。
            no_token = self.rem_swa_tokens <= 0
        if no_token:
            return AddReqResult.NO_TOKEN

        # 中译：输入 token 预算（max_prefill_tokens）耗尽——以 OTHER 停止（区别于显存耗尽）。
        if self.rem_input_tokens <= 0:
            return AddReqResult.OTHER

        if self.dllm_config is not None:
            if self.rem_dllm_tokens <= 0:
                return AddReqResult.OTHER
        else:
            # 中译：分块 prefill 的本轮 chunk 预算耗尽——停止添加。
            if self.rem_chunk_tokens is not None and self.rem_chunk_tokens <= 0:
                return AddReqResult.OTHER

        return AddReqResult.CONTINUE

    def _update_prefill_budget(
        self,
        prefix_len: int,
        extend_input_len: int,
        max_new_tokens: int,
        retracted_stain: bool,
    ):
        # TODO(lsyin): check this workaround logic, which only ensures the prefill will not out of memory, and may be too conservative
        # 中译：一个请求被接纳后，扣减各项预算（增大对应 offset / 减少剩余预算）。
        extend_input_len = self.ceil_paged_tokens(extend_input_len)

        # alloc_extend reserves an extra page_size per request to make sure the budget doesn't over-commit
        # 中译：每请求额外预留一页（alloc_extend 可能多用一页），防止预算超额承诺导致 OOM。
        page_overhead = self.page_size
        # 中译：total offset 计入「本次输入 + 未来生成(max_new) + 页开销」；cur offset 只计「本次输入 + 页开销」。
        self.rem_total_token_offset += extend_input_len + max_new_tokens + page_overhead
        self.cur_rem_token_offset += extend_input_len + page_overhead
        self.rem_input_tokens -= extend_input_len

        if self.is_hybrid_swa:
            self.rem_swa_token_offset += self._swa_budget_for_req(extend_input_len)

        if self.dllm_config is not None:
            self.rem_dllm_tokens -= extend_input_len
        elif self.rem_chunk_tokens is not None:
            self.rem_chunk_tokens -= extend_input_len

        # reprocessed_log_* is a subset of log_*; metrics_reporter subtracts it
        # when computing the first-attempt prefix cache hit rate.
        # 中译：log_* 用于统计前缀缓存命中率；reprocessed_log_* 是其子集（被抢占重处理的请求），
        #       计算「首次尝试」命中率时由 metrics_reporter 扣除，避免重试污染统计。
        self.log_hit_tokens += prefix_len
        self.log_input_tokens += extend_input_len
        if retracted_stain:
            self.reprocessed_log_hit_tokens += prefix_len
            self.reprocessed_log_input_tokens += extend_input_len

    def _get_dllm_remain_tokens(self) -> int:
        _rem_tokens = min(
            self.rem_dllm_tokens,
            self.dllm_block_size,
            int(self.rem_total_tokens),
        )
        if _rem_tokens <= 0:
            _rem_tokens = self.rem_dllm_tokens

        return _rem_tokens

    def _add_dllm_req(self, req: Req, prefix_len: int):
        # FIXME: consider the case when rem_dllm_tokens < dllm_block_size,
        # the diffusion unmask process may have some problems
        # Make sure at least one page is available
        trunc_len = (
            min(self.rem_dllm_tokens, self.dllm_block_size)
            // self.page_size
            * self.page_size
        )

        req.extend_input_len = trunc_len
        req.fill_len = prefix_len + trunc_len

        self.can_run_list.append(req)

        self._update_prefill_budget(prefix_len, trunc_len, 0, req.retracted_stain)

    def _req_inc_lock_ref(self, req: Req):
        # 中译：为已接纳的请求对其命中节点增加锁引用（防止其前缀缓存在使用期间被驱逐）；
        #       混合 SWA 时记录用于解锁的 swa_uuid_for_lock。
        result = self.tree_cache.inc_lock_ref(req.last_node)
        if self.is_hybrid_swa:
            req.swa_uuid_for_lock = result.swa_uuid_for_lock

    def add_dllm_staging_req(self, req: Req):
        assert self.dllm_config is not None
        _rem_tokens = self._get_dllm_remain_tokens()

        if _rem_tokens <= 0:
            return AddReqResult.NO_TOKEN

        # Truncate input length to available tokens and update request metadata
        truncated = req.extend_input_len > _rem_tokens
        req.extend_input_len = min(req.extend_input_len, _rem_tokens)
        req.fill_len = len(req.prefix_indices) + req.extend_input_len
        self.can_run_list.append(req)

        # Update budget: reserve max_new_tokens only if not truncated
        max_new_tokens = (
            min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
            if not truncated
            else 0
        )
        self._update_prefill_budget(
            0, req.extend_input_len, max_new_tokens, req.retracted_stain
        )

        # Return based on remaining token availability
        return (
            AddReqResult.NO_TOKEN
            if self._get_dllm_remain_tokens() <= 0
            else AddReqResult.CONTINUE
        )

    def add_chunked_req(self, req: Req):
        # 中译：续跑「上一轮未完成的分块 prefill 请求」。按剩余 chunk/总预算截断本轮要处理的 token 数，
        #       更新预算并加入运行列表；若仍未跑完（truncated）则返回 req 以便下轮继续，否则返回 None。
        if self.dllm_config is not None:
            _rem_tokens = self._get_dllm_remain_tokens()
        else:
            _rem_tokens = min(self.rem_chunk_tokens, int(self.rem_total_tokens))
            if self.is_hybrid_swa:
                # alloc_extend needs extend_num_tokens + page_size per request,
                # so reserve one page here to avoid OOM
                _rem_tokens = min(
                    _rem_tokens, int(self.rem_swa_tokens) - self.page_size
                )
            # The chunked_req must be added to the list; otherwise, it will cause a memory leak.
            # Therefore, in certain cases where _rem_tokens <= 0, it should be replaced with rem_chunk_tokens.
            if _rem_tokens <= 0:
                if self.is_hybrid_swa:
                    return req
                _rem_tokens = self.rem_chunk_tokens

        truncated = req.extend_input_len > _rem_tokens
        req.set_extend_input_len(min(req.extend_input_len, _rem_tokens))
        req.fill_len = len(req.prefix_indices) + req.extend_input_len
        self.can_run_list.append(req)
        self._update_prefill_budget(
            0,
            req.extend_input_len,
            (
                min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
                if not truncated
                else 0
            ),
            req.retracted_stain,
        )

        # Return if chunked prefill not finished
        return req if truncated else None

    @contextmanager
    def _lock_node(self, last_node: TreeNode):
        # 中译：上下文管理器——临时锁住命中的缓存节点 last_node（inc_lock_ref），防止在准入判断期间
        #       该前缀被其他逻辑驱逐；退出时务必对称解锁（dec_lock_ref）。
        dec_lock_params = None
        try:
            result = self.tree_cache.inc_lock_ref(last_node)
            if self.tree_cache.is_tree_cache():
                # init_load_back may revive SWA/Mamba tombstones while this
                # temporary admission lock is held. Release must mirror the
                # exact nodes skipped at acquire time.
                dec_lock_params = result.to_dec_params()
            yield None
        finally:
            if dec_lock_params is not None:
                self.tree_cache.dec_lock_ref(last_node, dec_lock_params)
            else:
                self.tree_cache.dec_lock_ref(last_node)

    def add_one_req_ignore_eos(self, req: Req):
        # 中译：ignore_eos 请求（不会因 eos 提前结束、必然生成到 max_new_tokens）的特殊准入路径。
        #       因其生成量确定，需更严格地估算「所有此类请求同时跑到末尾时」的显存峰值，避免后续 OOM。
        paged_input = self.ceil_paged_tokens(req.extend_input_len)
        if paged_input > min(self.cur_rem_tokens, self.rem_total_tokens):
            return AddReqResult.NO_TOKEN
        if self.is_hybrid_swa:
            if self._swa_budget_for_req(req.extend_input_len) > self.rem_swa_tokens:
                return AddReqResult.NO_TOKEN

        # 中译：把请求 r 的 (剩余将生成 token 数, 当前已占 token 数) 记入 req_states，供下方做峰值估算。
        #       ignore_eos 请求用比例 1.0（必跑满），其余用 new_token_ratio 折扣。insert_sort 时按 tokens_left 有序插入。
        def add_req_state(r, insert_sort=False):
            new_token_ratio = (
                1.0 if r.sampling_params.ignore_eos else self.new_token_ratio
            )
            tokens_left = r.sampling_params.max_new_tokens * new_token_ratio - len(
                r.output_ids
            )
            tokens_occupied = len(r.origin_input_ids) + len(r.output_ids)

            if tokens_left <= 0:
                return

            if not insert_sort:
                self.req_states.append((tokens_left, tokens_occupied))
            else:
                i = 0
                for i in range(len(self.req_states)):
                    if tokens_left <= self.req_states[i][0]:
                        break
                self.req_states.insert(i, (tokens_left, tokens_occupied))

        if self.req_states is None:
            self.req_states = []
            add_req_state(req)
            if self.running_batch is not None:
                for r in self.running_batch.reqs:
                    add_req_state(r)
            for r in self.can_run_list:
                add_req_state(r)
            self.req_states.sort(key=lambda x: x[0])
        else:
            add_req_state(req, insert_sort=True)

        if not self.is_hybrid_swa:
            # Skip this logic for swa. The SWA has different memory management, and
            # this mechanism is underestimating the memory usage.
            # 中译：SWA 内存管理不同、此估算会低估占用，故跳过。
            #       下面按 tokens_left 升序遍历：模拟「最先结束的请求陆续释放显存」的过程，
            #       检查在任一时刻是否会出现显存不足（min_free_tokens <= 预留量），若会则拒绝接纳。
            cur_rem_tokens = self.cur_rem_tokens - self.ceil_paged_tokens(
                req.extend_input_len
            )
            tokens_freed = 0
            for i, (tokens_left, tokens_occupied) in enumerate(self.req_states):
                # tokens_left gives a reservative calculation as the last token is not stored
                # 中译：bs 是「此刻仍未结束」的请求数；它们每个还要再吃 tokens_left 个 token 的显存。
                bs = len(self.req_states) - i
                # 中译：当前空闲 + 已释放 - 仍在跑的请求还需占用 = 这一时刻的最小空闲量。
                min_free_tokens = cur_rem_tokens + tokens_freed - tokens_left * bs
                # reserve tokens for corner cases
                # 中译：为角落情况保留少量 token；不足则无法安全接纳，返回 NO_TOKEN。
                if min_free_tokens <= IGNORE_EOS_RESERVE_TOKENS * bs:
                    return AddReqResult.NO_TOKEN
                # 中译：第 i 个请求结束后释放其占用，累加到 tokens_freed，供后续时刻使用。
                tokens_freed += tokens_occupied

        if (self.prefill_delayer_single_pass is not None) and (
            not self.prefill_delayer_single_pass.negotiate_should_allow_prefill(
                local_prefillable=True
            )
        ):
            return AddReqResult.OTHER

        if self.dllm_config is not None:
            if self.rem_dllm_tokens <= 0:
                return AddReqResult.OTHER

            self._add_dllm_req(req, 0)
        elif (
            self.rem_chunk_tokens is None  # chunked prefill is disabled
            or req.extend_input_len <= self.rem_chunk_tokens  # it is the last chunk
        ):
            # Non-chunked prefill — the whole sequence is committed this iter.
            req.fill_len = len(req.full_untruncated_fill_ids)
            assert (
                req.fill_len == len(req.prefix_indices) + req.extend_input_len
            ), f"{req.fill_len=} {len(req.prefix_indices)=} {req.extend_input_len=}"
            self.can_run_list.append(req)
            self._update_prefill_budget(
                0,
                req.extend_input_len,
                min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS),
                req.retracted_stain,
            )
        else:
            if self.rem_chunk_tokens <= 0:
                return AddReqResult.OTHER

            # Chunked prefill
            trunc_len = self.rem_chunk_tokens

            req.set_extend_input_len(trunc_len)
            assert len(req.prefix_indices) == 0
            req.fill_len = len(req.prefix_indices) + trunc_len
            self.can_run_list.append(req)
            self.new_chunked_req = req
            self._update_prefill_budget(0, trunc_len, 0, req.retracted_stain)

        return self.budget_state()

    def add_one_req(
        self, req: Req, has_chunked_req: bool, truncation_align_size: Optional[int]
    ):
        # 中译：常规（非 ignore_eos）请求的核心准入方法。流程概览：
        #   1) 各种前置约束检查（prefill 延迟器、CP 限制、最大请求数、缓存禁用时转 ignore_eos 路径）；
        #   2) 估算所需 token（输入 + 预留生成 + 页开销），与剩余总预算/SWA 预算比对，不够则 NO_TOKEN；
        #   3) 锁住命中节点后，必要时把 host 命中前缀 load back 回显存；
        #   4) 按是否启用分块 prefill，决定整条接纳或截断为一个 chunk；更新预算并加入 can_run_list；
        #   5) 返回 budget_state()，告诉调用方是否还能继续添加。
        if (self.prefill_delayer_single_pass is not None) and (
            not self.prefill_delayer_single_pass.negotiate_should_allow_prefill(
                local_prefillable=True,
                running_batch=self.running_batch.batch_size(),
                max_prefill_bs=self.max_prefill_bs,
                max_running_requests=self.max_running_requests,
                waiting_queue_len=self.waiting_queue_len,
            )
        ):
            return AddReqResult.OTHER
        # TODO support cp with multiple requests
        # Enabling context parallelism currently presents precision issues;
        # therefore, the prefill-batch setting is temporarily set to 1.
        # 中译：序列切分式 prefill 上下文并行（CP）暂存在精度问题，故每轮 prefill 批限制为 1 个请求。
        if (self.dsa_prefill_cp_in_seq_split) and len(self.can_run_list) >= 1:
            return AddReqResult.OTHER

        # 中译：达到本轮 prefill 请求数上限（prefill_max_requests）则停止添加。
        if (x := self.prefill_max_requests) is not None and len(self.can_run_list) >= x:
            return AddReqResult.OTHER

        # 中译：ignore_eos 且缓存被禁用时，走专门的 ignore_eos 准入估算路径。
        if req.sampling_params.ignore_eos and getattr(self.tree_cache, "disable", True):
            return self.add_one_req_ignore_eos(req)

        # Reserve page_size for page-alignment overhead. The paged allocator
        # may consume up to one extra page per request (see alloc_extend), and
        # _update_prefill_budget already accounts for this in the deduction.
        # Without this, admission is more optimistic than the actual budget
        # deduction, allowing over-admission when the pool is nearly full.
        # 中译：估算本请求总共需要的 token = 本次输入 + 预留生成(裁到上限) + 一页对齐开销。
        max_new = min(
            max(req.sampling_params.max_new_tokens - len(req.output_ids), 0),
            CLIP_MAX_NEW_TOKENS,
        )
        total_tokens = req.extend_input_len + max_new + self.page_size

        # adjusting the input_tokens based on host_hit_length and page_size
        # 中译：真实需新算的输入 token = 输入长度 - host 命中长度（命中部分将 load back，无需重算），再页对齐。
        real_input_tokens = req.extend_input_len - req.host_hit_length
        real_input_tokens = self.ceil_paged_tokens(real_input_tokens)
        prefix_len = len(req.prefix_indices)

        # 中译：总需求超过剩余总预算——显存不足，NO_TOKEN。
        if total_tokens >= self.rem_total_tokens:
            return AddReqResult.NO_TOKEN

        if self.is_hybrid_swa:
            swa_needed = self._swa_budget_for_req(
                req.extend_input_len, swa_host_hit_length=req.swa_host_hit_length
            )
            if swa_needed >= self.rem_swa_tokens:
                return AddReqResult.NO_TOKEN

        if (
            self.rem_chunk_tokens is None
            and len(self.can_run_list) != 0
            and real_input_tokens >= self.rem_input_tokens
        ):
            # If without chunked prefill:
            # - if the can_run_list is not empty, we satisfy the constraint of (max_prefill_tokens)
            # - if the can_run_list is empty, always accept the first prefill request
            # 中译：未启用分块 prefill 时：列表非空且输入超出剩余输入预算，则停止（满足 max_prefill_tokens 约束）；
            #       但列表为空时总是接纳第一个 prefill 请求（哪怕超预算，否则会饿死）。
            return AddReqResult.OTHER

        with self._lock_node(req.last_node):
            # self.rem_total_tokens may decrease after the lock acquisition
            # 中译：加锁后 rem_total_tokens 可能因并发驱逐而下降，故在锁内再次复核预算。
            if total_tokens >= self.rem_total_tokens:
                return AddReqResult.NO_TOKEN

            if self.is_hybrid_swa:
                swa_needed = self._swa_budget_for_req(
                    req.extend_input_len, swa_host_hit_length=req.swa_host_hit_length
                )
                if swa_needed >= self.rem_swa_tokens:
                    return AddReqResult.NO_TOKEN

            # 中译：若命中前缀在 host(CPU) 内存，需先 load back 回显存，并据此重算 prefix/extend 长度。
            if req.needs_host_load_back():
                new_indices, req.last_node = self.tree_cache.init_load_back(
                    InitLoadBackParams(
                        best_match_node=req.best_match_node,
                        host_hit_length=req.host_hit_length,
                        req=req,
                    )
                )
                req.prefix_indices = torch.cat([req.prefix_indices, new_indices])
                req.set_extend_input_len(
                    len(req.full_untruncated_fill_ids) - len(req.prefix_indices)
                )
                prefix_len = len(req.prefix_indices)
                req.cache_protected_len = prefix_len

            input_tokens = self.ceil_paged_tokens(req.extend_input_len)

            if (
                self.rem_chunk_tokens is None
                and len(self.can_run_list) != 0
                and input_tokens >= self.rem_input_tokens
            ):
                # If without chunked prefill:
                # - if the can_run_list is not empty, we satisfy the constraint of (max_prefill_tokens)
                # - if the can_run_list is empty, always accept the first prefill request
                return AddReqResult.OTHER

            if self.dllm_config is not None:
                if self.rem_dllm_tokens <= 0:
                    return AddReqResult.OTHER

                assert (
                    truncation_align_size is None
                ), "truncation_align_size is not supported for dllm prefill"

                self._add_dllm_req(req, prefix_len)
                self._req_inc_lock_ref(req)
            elif self.rem_chunk_tokens is None or input_tokens <= self.rem_chunk_tokens:
                # Non-chunked prefill — the whole sequence is committed this iter.
                # 中译：非分块（或本 chunk 已能容纳全部）——整条序列在本轮一次性处理。
                req.fill_len = len(req.full_untruncated_fill_ids)
                assert (
                    req.fill_len == len(req.prefix_indices) + req.extend_input_len
                ), f"{req.fill_len=} {len(req.prefix_indices)=} {req.extend_input_len=}"
                self.can_run_list.append(req)

                self._req_inc_lock_ref(req)
                self._update_prefill_budget(
                    prefix_len,
                    input_tokens,
                    min(
                        req.sampling_params.max_new_tokens,
                        CLIP_MAX_NEW_TOKENS,
                    ),
                    req.retracted_stain,
                )
            else:
                # Make sure at least one page is available
                # 中译：分块 prefill 分支——把本轮要处理的长度截到剩余 chunk 预算（向下页对齐，至少一页）。
                trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size

                if trunc_len <= 0:
                    return AddReqResult.OTHER

                # When truncation align size is set, we want to assert that the prefill prefix length is multiple of truncation align size
                # A typical use case is when deterministic inference is enabled with flashinfer attention backend,
                # we need the prefill prefix length to be multiple of attention split size
                if truncation_align_size is not None:
                    if trunc_len < truncation_align_size:
                        return AddReqResult.OTHER
                    else:
                        trunc_len = truncation_align_size * (
                            trunc_len // truncation_align_size
                        )

                now_input_len = trunc_len + len(req.prefix_indices)
                now_input_len = now_input_len // self.page_size * self.page_size
                trunc_len = now_input_len - len(req.prefix_indices)

                if trunc_len <= 0:
                    return AddReqResult.OTHER

                # Chunked prefill
                req.set_extend_input_len(trunc_len)
                req.fill_len = len(req.prefix_indices) + trunc_len

                self.can_run_list.append(req)
                self.new_chunked_req = req

                self._req_inc_lock_ref(req)
                self._update_prefill_budget(
                    prefix_len, trunc_len, 0, req.retracted_stain
                )

        return self.budget_state()

    def preempt_to_schedule(self, req: Req, server_args: ServerArgs) -> bool:
        """
        Preempt running requests to serve the new request if the priority threshold is met and token count sum is verified.
        Returns True if preemption was committed, and the new request can be scheduled.

        中译：优先级抢占。当新请求 req 优先级足够高（与运行请求的优先级差超过阈值），且抢占若干低优先级
              运行请求所释放的显存足以容纳 req 时，执行抢占：释放被抢占请求的 KV cache 并移出运行批次。
              成功提交抢占（新请求可被调度）返回 True，否则返回 False（不做任何改动）。
        """
        # Iterate running requests to find preemptible requests
        # 中译：优先级符号，决定「数值大」还是「数值小」代表高优先级（与排序约定一致）。
        priority_sign = 1 if server_args.schedule_low_priority_values_first else -1

        # NOTE: A request finishes in two phases:
        #   1) update_finish_state + release_kv_cache  (in process_batch_result)
        #   2) filter out of batch                (in get_next_batch_to_run / update_running_batch)
        # Preemption runs between these two phases (inside get_new_batch_prefill),
        # so running_batch may still contain requests whose KV cache is already freed.
        # We must skip them here to avoid a double-free on release_req.
        # 中译：请求结束分两阶段：1) 标记完成 + 释放 KV cache；2) 从批次中过滤移除。抢占发生在这两步之间，
        #       故 running_batch 里可能仍有「KV 已释放但还没被过滤」的请求，必须跳过它们以免重复释放（double-free）。
        valid_running_reqs = (
            r
            for r in self.running_batch.reqs
            if r not in self.preempt_list and not r.finished()
        )

        # 中译：把候选运行请求按「优先级从低到高、入队时间从晚到早」排序——优先抢占最该被牺牲的（低优先级、晚到）。
        sorted_valid_running_reqs = sorted(
            valid_running_reqs,
            key=lambda x: (
                x.priority * (-priority_sign),
                -x.time_stats.wait_queue_entry_time,
            ),
        )

        preemptible_reqs = []
        # 中译：为容纳新请求还需腾出的 token 量 = 新请求所需 - 当前剩余总预算。
        min_tokens_to_remove = (
            req.extend_input_len
            + min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
            - self.rem_total_tokens
        )
        for running_req in sorted_valid_running_reqs:
            # Priority difference needs to meet the threshold to be preemptible.
            # 中译：优先级差需超过阈值才允许抢占（避免优先级相近时频繁抢占抖动）。
            priority_diff = (req.priority - running_req.priority) * (-priority_sign)

            if priority_diff > self.priority_scheduling_preemption_threshold:
                preemptible_reqs.append(running_req)
                # 中译：累计该请求释放的 token，逼近所需腾出量；够了就停止挑选。
                min_tokens_to_remove -= self._get_running_request_total_token_offset(
                    running_req
                )
                if min_tokens_to_remove <= 0:
                    break
            else:
                # 中译：已按优先级排序，遇到第一个不满足阈值的即可停止（后面只会更不满足）。
                break

        # Check max token count limit can be met
        # 中译：没有可抢占请求，或抢完仍腾不出足够 token——放弃抢占，返回 False。
        if len(preemptible_reqs) == 0 or min_tokens_to_remove > 0:
            return False

        # Preempt running requests. Release allocated resources for immediate usage.
        # 中译：正式执行抢占——释放被抢占请求的资源以供新请求立即使用。
        preemptible_reqs = set(preemptible_reqs)
        keep_indices = []
        release_counter = 0
        for i, running_req in enumerate(self.running_batch.reqs):
            if running_req in preemptible_reqs:
                # 中译：被抢占——回收之前为它预留的 total offset（腾出预算），并释放其 KV cache。
                self.rem_total_token_offset -= (
                    self._get_running_request_total_token_offset(running_req)
                )
                release_counter += 1
                self.running_batch.release_req(
                    i, len(self.running_batch.reqs) - release_counter, server_args
                )
            else:
                # 中译：保留该请求，记录其下标用于随后重建批次。
                keep_indices.append(i)
        # 中译：按保留下标过滤运行批次，并把被抢占请求记入 preempt_list（后续会被退回等待队列重排）。
        self.running_batch.filter_batch(keep_indices=keep_indices)
        self.preempt_list.extend(preemptible_reqs)
        return True
