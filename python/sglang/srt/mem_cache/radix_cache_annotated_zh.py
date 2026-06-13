from __future__ import annotations

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.utils import convert_to_bigram_key

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
用于管理 KV 缓存的基数树（radix tree）数据结构。

【本文件中文注释版】
本文件是 `radix_cache.py` 的带详细中文注释副本，代码逻辑与原文件保持完全一致，
仅添加注释用于学习理解，请勿在生产中直接引用本副本。

核心思想（RadixAttention 前缀缓存）：
- 用一棵基数树存储「token 序列 -> KV cache 索引」的映射；
- 多个请求若共享相同的前缀 token，则共享同一批 KV cache，节省显存与重复计算；
- 通过 extra_key（如 LoRA id、cache salt）对不同命名空间做隔离，避免错误共享；
- 通过 lock_ref 引用计数保护正在使用的节点不被淘汰；
- 当显存不足时，按可配置的淘汰策略（LRU/LFU/FIFO 等）从叶子节点开始淘汰。
"""

import logging
import sys
import time
from collections import defaultdict
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

from sglang.srt.disaggregation.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.evict_policy import (
    EvictionStrategy,
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
    SLRUStrategy,
)
from sglang.srt.mem_cache.hicache_storage import get_hash_str, hash_str_to_int64

if TYPE_CHECKING:
    # 仅用于类型检查，避免运行时循环导入。
    from sglang.srt.managers.schedule_batch import Req


class RadixKey:
    """基数树的键：本质是一段 token id 序列，外加用于命名空间隔离的 extra_key。"""

    def __init__(
        self,
        token_ids: List[int],
        extra_key: Optional[str] = None,
        is_bigram: bool = False,
    ):
        # token id 序列。
        self.token_ids = token_ids
        # 额外键（例如 lora_id、cache_salt），用于隔离不同命名空间的缓存。
        self.extra_key = extra_key
        # 是否为 bigram（二元组）键（EAGLE 投机解码场景使用）。
        self.is_bigram = is_bigram

    def __len__(self) -> int:
        # 键长度即 token 数量。
        return len(self.token_ids)

    def __iter__(self) -> Iterator[int]:
        # 支持迭代 token id。
        return iter(self.token_ids)

    def __getitem__(self, idx: Union[int, slice]) -> "RadixKey":
        # 支持切片/索引，返回新的 RadixKey（继承 extra_key）。
        if isinstance(idx, slice):
            return RadixKey(self.token_ids[idx], self.extra_key)
        return RadixKey([self.token_ids[idx]], self.extra_key)

    def __repr__(self) -> str:
        # 打印时仅预览前 10 个 token，避免日志过长。
        preview = self.token_ids[:10]
        return f"RadixKey(extra_key={self.extra_key!r}, token_ids={preview}{'...' if len(self.token_ids) > 10 else ''})"


def maybe_bigram_convert(
    is_eagle: bool,
    key: RadixKey,
    value: Optional[torch.Tensor] = None,
) -> Tuple[RadixKey, Optional[torch.Tensor]]:
    # 在 EAGLE 投机解码场景下，把普通 token 键转换为 bigram（二元组）键。
    if is_eagle and not key.is_bigram:
        key.token_ids = convert_to_bigram_key(key.token_ids)
        key.is_bigram = True
        if value is not None:
            # bigram 化后 value 需要截断到键的新长度。
            value = value[: len(key)]
    return key, value


def page_align_keys(key: list, page_size) -> list:
    # 将键按 page_size 对齐（截掉末尾不足一页的部分）。
    if page_size == 1:
        return key
    # 向下取整到 page_size 的整数倍。
    page_aligned_len = len(key) // page_size * page_size
    return key[:page_aligned_len]


class TreeNode:
    """基数树的节点。每个节点代表一段连续的 token 子序列及其对应的 KV cache。"""

    # 全局自增计数器，用于给每个节点分配唯一 id。
    counter = 0

    def __init__(self, id: Optional[int] = None, priority: int = 0):
        # 子节点字典：child_key -> TreeNode（defaultdict 便于自动创建）。
        self.children = defaultdict(TreeNode)
        # 父节点指针。
        self.parent: TreeNode = None
        # 本节点对应的键（token 子序列）。
        self.key: RadixKey = None
        # 本节点对应的 KV cache 设备索引张量；为 None 表示已被淘汰。
        self.value: Optional[torch.Tensor] = None
        # 引用计数：>0 表示该节点正被某请求使用，受保护不可淘汰。
        self.lock_ref = 0
        # 最近访问时间（用于 LRU 等淘汰策略）。
        self.last_access_time = time.monotonic()
        # 节点创建时间（用于 FIFO 等淘汰策略）。
        self.creation_time = time.monotonic()

        # 命中次数（用于 LFU 淘汰策略）。
        self.hit_count = 0
        # 主机侧（host/CPU）引用计数：被存储操作引用时递增，保护 host_value 不被淘汰。
        self.host_ref_counter = 0
        # 存储 KV cache 在主机侧的索引（分层缓存 HiCache 场景）。
        self.host_value: Optional[torch.Tensor] = None
        # 存储每个页（page）的哈希值（用于 KV cache 事件上报与去重）。
        self.hash_value: Optional[List[str]] = None
        # 优先级（用于 priority 感知的淘汰策略）。
        self.priority = priority

        # 分配节点 id：未显式指定则用全局计数器。
        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        # value 为 None 即表示该节点的设备侧 KV cache 已被淘汰。
        return self.value is None

    @property
    def backuped(self):
        # host_value 非空表示已在主机侧备份。
        return self.host_value is not None

    def protect_host(self):
        """保护主机侧 value 不被淘汰（引用计数 +1）。"""
        self.host_ref_counter += 1

    def release_host(self):
        """释放主机侧 value，允许其被淘汰（引用计数 -1）。"""
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            # 计数已为 0 还释放，说明逻辑出错。
            raise RuntimeError("Host reference counter is already zero.")

    def get_last_hash_value(self) -> Optional[str]:
        """返回本节点最后一页的哈希值。"""
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    @lru_cache(maxsize=1)
    def get_prefix_hash_values(self, node: TreeNode) -> List[str]:
        # 递归获取从根到该节点的所有页哈希值（带 lru_cache 缓存最近一次结果）。
        if node is None or node.hash_value is None:
            return []

        return node.get_prefix_hash_values(node.parent) + node.hash_value

    def __lt__(self, other: "TreeNode"):
        # 用于堆排序：按最近访问时间比较（配合淘汰策略）。
        return self.last_access_time < other.last_access_time


def _check_extra_key(key0: RadixKey, key1: RadixKey):
    # 前缀匹配必须在相同 extra_key 命名空间下进行，否则属于逻辑错误。
    if key0.extra_key != key1.extra_key:
        raise ValueError(
            f"_key_match should be run on the same extra key, but got key0.extra_key={key0.extra_key} != key1.extra_key={key1.extra_key}"
        )


def _key_match_page_size1(key0: RadixKey, key1: RadixKey):
    # page_size == 1 时逐 token 比较，返回最长公共前缀长度。
    _check_extra_key(key0, key1)
    i = 0
    for k0, k1 in zip(key0.token_ids, key1.token_ids):
        if k0 != k1:
            break
        i += 1
    return i


def _key_match_paged(key0: RadixKey, key1: RadixKey, page_size: int):
    # page_size > 1 时按页比较，返回匹配到的 token 数量（page_size 的整数倍）。
    _check_extra_key(key0, key1)
    min_len = min(len(key0), len(key1))

    i = 0
    while i < min_len:
        # 整页不一致则停止。
        if key0.token_ids[i : i + page_size] != key1.token_ids[i : i + page_size]:
            break
        i += page_size

    return i


def get_child_key(key: RadixKey, page_size: int = 1):
    # 计算用于在 children 字典中索引子节点的 key。
    if page_size == 1:
        # page_size==1：用首个 token。
        plain_key = key.token_ids[0]
    else:
        # page_size>1：用首页 token 组成的元组。
        plain_key = tuple(key.token_ids[:page_size])
    if key.extra_key is None:
        return plain_key
    else:
        # 带 extra_key 时把它一并纳入 child key，实现命名空间隔离。
        return (key.extra_key, plain_key)


def compute_node_hash_values(node: "TreeNode", page_size: int) -> List[str]:
    """为「位置感知」的标识计算基于 SHA256 的哈希值。

    参数：
        node：要计算哈希的 TreeNode。
        page_size：分页大小（按页切分 token）。

    返回：
        SHA256 十六进制字符串列表，每页一个。
    """
    hash_values = []

    # 若存在父节点，则取父节点最后一页的哈希作为链式哈希的起点。
    parent_hash = None
    if node.parent is not None and node.parent.hash_value is not None:
        # 通过判断 key 是否为空来识别父节点是否为 root。
        if len(node.parent.key) > 0 and len(node.parent.hash_value) > 0:
            parent_hash = node.parent.hash_value[-1]

    # 遍历本节点的各页。
    for start in range(0, len(node.key), page_size):
        page_tokens = node.key.token_ids[start : start + page_size]
        if not page_tokens:
            continue

        # 通过 get_hash_str 做基于 SHA256 的链式哈希（带上一页哈希做前缀）。
        hash_val = get_hash_str(page_tokens, prior_hash=parent_hash)
        hash_values.append(hash_val)
        parent_hash = hash_val

    return hash_values


def split_node_hash_value(
    child_hash_value: Optional[List[str]], split_len: int, page_size: int
) -> tuple[Optional[List[str]], Optional[List[str]]]:
    """节点分裂时，在父子节点之间切分 hash_value。

    参数：
        child_hash_value：被分裂的子节点的 hash_value 列表。
        split_len：切分位置（以 token 计）。
        page_size：分页大小（用于换算页数）。

    返回：
        (新节点的 hash_value, 更新后的子节点 hash_value) 二元组。
    """
    if child_hash_value is None:
        return None, None

    # 将 token 维度的 split_len 换算成页数维度的 split_pages。
    if page_size == 1:
        split_pages = split_len
    else:
        split_pages = split_len // page_size

    new_node_hash = child_hash_value[:split_pages]
    child_hash = child_hash_value[split_pages:]

    return new_node_hash, child_hash


class RadixCache(BasePrefixCache):
    """基于基数树的前缀缓存实现（RadixAttention 的核心数据结构）。"""

    def __init__(self, params: CacheInitParams):
        # 是否禁用缓存（禁用时所有操作退化为直接分配/释放）。
        self.disable = params.disable
        # 请求到 token 槽位的映射池。
        self.req_to_token_pool = params.req_to_token_pool
        # token 到 KV cache 的分配器。
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        # 分页大小。
        self.page_size = params.page_size
        # 是否启用 KV cache 事件上报。
        self.enable_kv_cache_events = params.enable_kv_cache_events
        # 是否为 EAGLE 投机解码模式（影响 bigram 转换）。
        self.is_eagle = params.is_eagle
        # 是否禁止把已完成请求插入缓存（确定性模式下使用）。
        self.disable_finished_insert = params.disable_finished_insert
        # 淘汰策略名（统一转小写）。
        self.eviction_policy = params.eviction_policy.lower()

        # KV cache 事件队列。
        self.kv_event_queue = []

        # 如启用指标，则初始化指标采集器。
        if params.enable_metrics:
            self.init_metrics_collector()

        # 推断设备（有分配器则跟随分配器，否则用 CPU，便于模拟）。
        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        # 根据分页大小选择键匹配函数与子键计算函数。
        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = get_child_key
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=self.page_size)
            self.get_child_key_fn = partial(get_child_key, page_size=self.page_size)

        # 按配置选择淘汰策略实现。
        if self.eviction_policy == "lru":
            self.eviction_strategy: EvictionStrategy = LRUStrategy()
        elif self.eviction_policy == "lfu":
            self.eviction_strategy: EvictionStrategy = LFUStrategy()
        elif self.eviction_policy == "fifo":
            self.eviction_strategy: EvictionStrategy = FIFOStrategy()
        elif self.eviction_policy == "mru":
            self.eviction_strategy: EvictionStrategy = MRUStrategy()
        elif self.eviction_policy == "filo":
            self.eviction_strategy: EvictionStrategy = FILOStrategy()
        elif self.eviction_policy == "priority":
            self.eviction_strategy: EvictionStrategy = PriorityStrategy()
        elif self.eviction_policy == "slru":
            self.eviction_strategy: EvictionStrategy = SLRUStrategy()

        else:
            # 未知策略直接报错。
            raise ValueError(
                f"Unknown eviction policy: {self.eviction_policy}. Supported policies: 'lru', 'lfu', 'fifo', 'mru', 'filo', 'priority', 'slru'."
            )

        # 可淘汰叶子节点集合（仅叶子且未被锁定时才可淘汰）。
        self.evictable_leaves = set()
        # 初始化树。
        self.reset()

    @classmethod
    def create_simulated(
        self,
        disable: bool = False,
        mock_allocator: Optional[Any] = None,
        page_size: int = 1,
        enable_kv_cache_events: bool = False,
    ) -> RadixCache:
        """构造一个不依赖真实内存池的基数缓存，用于模拟/测试。"""
        params = CacheInitParams(
            disable=disable,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=page_size,
            enable_kv_cache_events=enable_kv_cache_events,
        )
        return RadixCache(params)

    ##### 公共 API #####

    def reset(self):
        # 用最小优先级初始化 root，使任何真实优先级都能覆盖它。
        self.root_node = TreeNode(priority=-sys.maxsize)
        self.root_node.key = RadixKey(token_ids=[], extra_key=None)
        self.root_node.value = []
        self.root_node.host_value = []
        # root 始终加锁，永不被淘汰。
        self.root_node.lock_ref = 1
        self.root_node.hash_value = []
        # 可淘汰大小与受保护大小清零。
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.evictable_leaves.clear()
        # 上报「全部清空」事件。
        self._record_all_cleared_event()

    def maybe_bigram_convert(
        self, key: RadixKey, value: Optional[torch.Tensor] = None
    ) -> Tuple[RadixKey, Optional[torch.Tensor]]:
        # 实例方法封装：按是否 EAGLE 决定是否做 bigram 转换。
        return maybe_bigram_convert(self.is_eagle, key, value)

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """在基数树中找到 ``key`` 的最长已缓存前缀。

        前缀匹配的逻辑命名空间由 token id 序列与 ``RadixKey`` 携带的可选
        ``extra_key`` 共同决定。前导 token id 相同但 ``extra_key`` 不同的条目
        会被刻意隔离，永不共享前缀节点。其用途包括：

        * 隔离不同 LoRA / adapter ID 的 KV cache。
        * 通过提供不同的 ``extra_key`` 来分离那些刻意不应共享状态的请求
          （例如不同的采样 salt、缓存版本或检索增强上下文）。

        参数：
            params (MatchPrefixParams)：包含查询键的参数（token id 列表与可选
                ``extra_key`` 命名空间标签）。若 ``page_size > 1``，匹配前会在内部
                把长度截断为 ``page_size`` 的整数倍。传入空键将返回空结果，
                且 last node 为 root。

        返回：
            MatchResult：``device_indices`` 是 1 维 ``torch.int64`` 张量，
            为最长已缓存前缀对应的 KV cache 索引拼接结果（长度可能为 0）。
            ``last_device_node`` 与 ``last_host_node``（目前相同）是表示匹配前缀
            终止节点的树节点对象。若匹配恰好落在某个已存储分段内部，本方法
            可能会通过分裂节点来修改内部结构。

        内部更新：
            * 刷新淘汰策略使用的访问元数据（时间戳）。
            * 若查询落在某存储分段内部，则分裂该节点一次以暴露精确边界；
              这种结构细化能提升后续匹配效率，且不会复制数据。
        """
        key = params.key
        # 可能需要做 bigram 转换。
        key, _ = self.maybe_bigram_convert(key)

        def empty_match_result():
            # 构造空匹配结果（device_indices 为空，last node 为 root）。
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
            )

        # 禁用或空键直接返回空结果。
        if self.disable or len(key) == 0:
            return empty_match_result()

        # page_size != 1 时把键按页对齐。
        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        if len(key) == 0:
            return empty_match_result()

        # 从 root 开始递归匹配。
        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            # 多段命中则拼接成一个张量。
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)
        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
        )

    def insert(self, params: InsertParams) -> InsertResult:
        # 向树中插入一段键值，返回已存在的前缀长度。
        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        priority = params.priority
        chunked = params.chunked

        # 未提供 value 时，用 token id 本身作为占位 value（模拟/测试场景）。
        if value is None:
            value = torch.tensor(key.token_ids, dtype=torch.int64)

        key, value = self.maybe_bigram_convert(key, value)

        # 递归插入，返回命中的已有前缀长度。
        prefix_len = self._insert_helper(self.root_node, key, value, priority, chunked)
        return InsertResult(prefix_len=prefix_len)

    def cache_finished_req(self, req: Req, is_insert: bool = True):
        """请求完成时，缓存其 KV 索引。"""
        # 确定性模式下禁止把已完成请求插入基数缓存。
        if self.disable_finished_insert:
            is_insert = False

        # 取出本请求已提交的 KV cache 长度。
        kv_committed_len = req.pop_committed_kv_cache()
        if self.disable:
            # 禁用缓存：直接释放该请求占用的 KV 索引。
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            return

        # 拼接输入与输出 token，截断到已提交长度。
        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # EAGLE 场景下可能需要转换为 bigram 键。
        keys = convert_to_bigram_key(token_ids) if self.is_eagle else token_ids
        # 按页对齐键。
        keys = page_align_keys(keys, self.page_size)
        # 取对齐后的 KV 索引作为 value（拷贝一份，转 int64）。
        values = kv_indices[: len(keys)].to(dtype=torch.int64, copy=True)
        radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

        # 基数缓存会在内存池中持有一份引用。
        if is_insert:
            priority = getattr(req, "priority", 0) or 0
            result = self.insert(
                InsertParams(key=radix_key, value=values, priority=priority)
            )
            new_prefix_len = result.prefix_len
            # 释放那些已经存在于树中的重复 KV 索引。
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : new_prefix_len]
            )
        else:
            # 不插入时，直接释放受保护长度之后的部分。
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : len(keys)]
            )

        # 释放未对齐的尾部（不足一页的部分）。
        self.token_to_kv_pool_allocator.free(kv_indices[len(keys) :])

        # 移除请求槽位，释放缓存锁（引用计数 -1）。
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False):
        """请求尚未完成时，缓存其当前已填充的 KV 索引。"""
        if self.disable:
            return

        # 取当前已填充的 token。
        token_ids = req.fill_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # EAGLE 场景下可能需要转换为 bigram 键。
        keys = convert_to_bigram_key(token_ids) if self.is_eagle else token_ids
        keys = page_align_keys(keys, self.page_size)
        values = kv_indices[: len(keys)].to(dtype=torch.int64, copy=True)
        radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

        # 基数缓存会在内存池中持有一份引用。
        result = self.insert(
            InsertParams(
                key=radix_key,
                value=values,
                chunked=chunked,
                priority=getattr(req, "priority", 0) or 0,
            )
        )
        new_prefix_len = result.prefix_len

        # 释放受保护长度到新前缀长度之间的重复 KV 索引。
        self.token_to_kv_pool_allocator.free(
            kv_indices[req.cache_protected_len : new_prefix_len]
        )

        # 前缀索引可能已被更新，重新匹配一次以复用。
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )
        assert len(new_indices) == len(keys), f"{len(new_indices)=}, {len(keys)=}"

        # 把新索引写回请求的 token 映射表（仅写受保护长度之后的部分）。
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        # cache_protected_len 不总是等于 len(req.prefix_indices)：
        # 当 page_size > 1 时，末尾不足一页的部分会加入 req.prefix_indices，
        # 但这部分 KV 索引并未加入树中。
        # 它应在下一次 cache_unfinished_req 与最终的 cache_finished_req 中被释放，避免内存泄漏。
        # 因此引入 cache_protected_len 字段以确保这部分能被正确释放。
        req.cache_protected_len = len(new_indices)

        # 旧节点解锁、新节点加锁，维护引用计数。
        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` 之后会在 `PrefillAdder::add_chunked_req` 中使用：
        # - page_size != 1：末尾存在不足一页的部分，需保留完整 kv_indices。
        # - eagle 场景：bigram 键只缓存 len - 1 个 kv 索引。
        if len(new_indices) < len(kv_indices):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices

        req.last_node = new_last_node

    def pretty_print(self):
        # 以可读格式打印整棵基数树，并输出总 token 数。
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        # 统计树中所有未被淘汰节点的 token 总数。
        return self._total_size_helper()

    def evict(self, params: EvictParams) -> EvictResult:
        # 淘汰若干 token 以释放显存，返回实际淘汰的 token 数。
        if self.disable:
            return EvictResult()

        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        leaves = list(self.evictable_leaves)
        # 构造小根堆：按淘汰策略给出的优先级排序（优先级小者先淘汰）。
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        # 循环淘汰，直到达到目标 token 数或堆为空。
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            # 释放该叶子节点的 KV cache 并从树中删除。
            self.token_to_kv_pool_allocator.free(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            # 若父节点因此变成可淘汰叶子（无子节点且未锁定），则也入堆。
            if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

            # 上报「块移除」事件。
            self._record_remove_event(x)

        # 更新淘汰相关指标。
        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        # 从 node 一路向上到 root，对路径上所有节点引用计数 +1（加锁保护不被淘汰）。
        if self.disable:
            return IncLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            # 从 0 变为被引用：该节点从「可淘汰」转为「受保护」。
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)
            node = node.parent
        return IncLockRefResult(delta=delta)

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        # 从 node 一路向上到 root，对路径上所有节点引用计数 -1（解锁）。
        if self.disable:
            return DecLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            # 从 1 变为 0：该节点从「受保护」转为「可淘汰」。
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            if node.parent is None:
                # 防御性检查：只有 root 允许 parent 为 None。
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
        return DecLockRefResult(delta=delta)

    def evictable_size(self):
        # 返回当前可淘汰的 token 总数。
        return self.evictable_size_

    def protected_size(self):
        # 返回当前被锁定（受保护）的缓存大小。
        return self.protected_size_

    def all_values_flatten(self):
        # 深度优先遍历，把所有节点的 value 拼成一个张量返回。
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values)

    ##### 内部辅助函数 #####

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        # 从 node 开始，沿子节点逐段匹配 key，返回命中的 value 段列表与终止节点。
        access_time = time.monotonic()
        node.last_access_time = access_time

        child_key = self.get_child_key_fn(key)

        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            # 刷新访问时间（供淘汰策略使用）。
            child.last_access_time = access_time
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                # 仅部分匹配：在匹配边界处分裂节点，暴露精确前缀。
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            else:
                # 完整匹配该子节点：继续向下匹配剩余 key。
                value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # 在 split_len 处把 child 拆成 new_node -> child 两层。
        # new_node 继承 child 的优先级（代表共享前缀部分）。
        new_node = TreeNode(priority=child.priority)
        new_node.hit_count = child.hit_count
        # new_node 接管 child 作为其唯一子节点。
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len].clone()
        # 调整 child 自身：保留 split_len 之后的部分。
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:].clone()
        # 把 new_node 挂到原父节点下。
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        # 若 hash_value 已计算，则一并切分；否则保持 None。
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        return new_node

    def _inc_hit_count(self, node: TreeNode, chunked: bool = False):
        # 跳过分块（chunked）请求的命中计数，避免分块请求对其前一分块创建的节点
        # 造成自我引用式的命中膨胀。
        if chunked:
            return
        node.hit_count += 1

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value,
        priority: int = 0,
        chunked: bool = False,
    ):
        # 将 None 优先级归一为 0。
        if priority is None:
            priority = 0
        access_time = time.monotonic()
        node.last_access_time = access_time
        # 沿路径更新优先级（取最大值，向上传播更高优先级）。
        node.priority = max(node.priority, priority)
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        # 沿已有路径尽量匹配，累计已命中的前缀长度。
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = access_time
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                # 部分匹配：分裂出共享前缀节点。
                new_node = self._split_node(node.key, node, prefix_len)
                new_node.priority = max(new_node.priority, priority)
                self._inc_hit_count(new_node, chunked)
                node = new_node
            else:
                # 完整匹配：更新该节点优先级与命中计数。
                node.priority = max(node.priority, priority)
                self._inc_hit_count(node, chunked)
            if len(key):
                child_key = self.get_child_key_fn(key)

        # 还有剩余 key：创建新节点挂到当前节点下。
        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            self._inc_hit_count(new_node, chunked)
            node.children[child_key] = new_node
            # 新增的部分计入可淘汰大小。
            self.evictable_size_ += len(key)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)
            # 哈希在事件上报时惰性计算。
            self._record_store_event(new_node)
        return total_prefix_length

    def _print_helper(self, node: TreeNode, indent: int):
        """以人类可读的格式打印基数树。"""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key.token_ids[:10],
                f"r={current_node.lock_ref}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                # 校验 children 字典的 key 与子节点 key 一致。
                assert key == self.get_child_key_fn(
                    child.key
                ), f"{key=}, {self.get_child_key_fn(child.key)=}"

    def _delete_leaf(self, node):
        # 从父节点的 children 中移除该叶子节点。
        key = self.get_child_key_fn(node.key)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        # 更新可淘汰大小，并从可淘汰叶子集合中移除。
        self.evictable_size_ -= len(node.key)
        if node in self.evictable_leaves:
            self.evictable_leaves.remove(node)
        # 删除后父节点可能成为新的叶子，更新其状态。
        self._update_leaf_status(node.parent)

    def _update_leaf_status(self, node: TreeNode):
        # 维护「可淘汰叶子」集合：节点已淘汰或被锁定时，不应在集合中。
        if node.evicted or node.lock_ref > 0:
            if node in self.evictable_leaves:
                self.evictable_leaves.remove(node)
            return

        # 只要存在任一未淘汰的子节点，则它不是叶子，移出集合。
        for child in node.children.values():
            if not child.evicted:
                if node in self.evictable_leaves:
                    self.evictable_leaves.remove(node)
                return

        # 否则它是一个可淘汰叶子，加入集合。
        if node not in self.evictable_leaves:
            self.evictable_leaves.add(node)

    def _total_size_helper(self):
        # 迭代统计所有未淘汰节点的 value 长度之和。
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size

    def _record_store_event(self, node: TreeNode):
        # 每个 page_size 大小的分块上报一个 BlockStored 事件。
        if self.enable_kv_cache_events:
            # 若尚未计算哈希则惰性计算。
            if node.hash_value is None:
                node.hash_value = compute_node_hash_values(node, self.page_size)

            # 取父节点最后一页哈希作为第一页的 parent hash。
            parent_block_hash = None
            if node.parent is not None and node.parent != self.root_node:
                if (
                    node.parent.hash_value is not None
                    and len(node.parent.hash_value) > 0
                ):
                    parent_block_hash = hash_str_to_int64(node.parent.hash_value[-1])

            page_index = 0
            # 逐页上报存储事件。
            for start in range(0, len(node.key), self.page_size):
                page_tokens = node.key.token_ids[start : start + self.page_size]
                if not page_tokens:
                    continue

                block_hash = hash_str_to_int64(node.hash_value[page_index])

                self.kv_event_queue.append(
                    BlockStored(
                        block_hashes=[block_hash],
                        parent_block_hash=parent_block_hash,
                        token_ids=page_tokens,
                        block_size=len(page_tokens),
                        lora_id=None,
                        medium=MEDIUM_GPU,
                    )
                )

                # 链式更新 parent hash。
                parent_block_hash = block_hash
                page_index += 1

    def _record_remove_event(self, node: TreeNode):
        # 每个分块上报一个 BlockRemoved 事件。
        if self.enable_kv_cache_events:
            # 惰性计算哈希（必须与存储时一致）。
            if node.hash_value is None:
                node.hash_value = compute_node_hash_values(node, self.page_size)

            page_index = 0
            for start in range(0, len(node.key), self.page_size):
                page_tokens = node.key.token_ids[start : start + self.page_size]
                if not page_tokens:
                    continue

                block_hash = hash_str_to_int64(node.hash_value[page_index])

                self.kv_event_queue.append(
                    BlockRemoved(block_hashes=[block_hash], medium=MEDIUM_GPU)
                )

                page_index += 1

    def _record_all_cleared_event(self):
        # 上报「全部块已清空」事件。
        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

    def take_events(self):
        """原子地取走所有事件并清空队列。

        返回：
            一组 KV cache 事件列表。
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events


if __name__ == "__main__":
    # 简单自测：构造一个模拟基数缓存并插入若干序列，最后打印并做一次前缀匹配。
    tree = RadixCache.create_simulated()

    # 示例 token id 序列（整数列表）。
    tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 3], extra_key=None)))
    tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 3], extra_key=None)))
    tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 4, 5], extra_key=None)))
    tree.insert(
        InsertParams(key=RadixKey(token_ids=[1, 2, 4, 5, 6, 7], extra_key=None))
    )
    tree.insert(
        InsertParams(key=RadixKey(token_ids=[8, 9, 10, 11, 12], extra_key=None))
    )
    tree.pretty_print()

    # 匹配 [1,2,3,13,14]：应命中前缀 [1,2,3]。
    print(
        tree.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=[1, 2, 3, 13, 14], extra_key=None))
        )
    )

