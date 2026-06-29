from __future__ import annotations

from sglang.srt.mem_cache.cache_init_params import CacheInitParams

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
用于管理 KV 缓存的 radix tree（基数树/前缀树）数据结构。

整体思路：
    把不同请求的 token 序列以“公共前缀共享”的方式组织进一棵树。每个 TreeNode
    持有一段连续 token 的 KV 缓存槽位索引（value），从根到某节点的路径就拼出一段
    完整前缀。新请求到来时，沿树匹配最长公共前缀即可直接复用这部分 KV，无需重算，
    这就是 prefix caching（前缀缓存）的核心收益来源。

本文件主要包含三个类：
    * RadixKey  : 前缀键，封装 token 序列及其 extra_key 命名空间，并提供匹配/切片/
                  哈希等操作；同时支持 bigram（相邻二元组）视图以服务 EAGLE 投机解码。
    * TreeNode  : 树节点，存储某段前缀的 KV 索引、锁引用计数、淘汰所需的访问元数据，
                  以及 HiCache 分级缓存所需的 host 侧备份信息。
    * RadixCache: 前缀缓存的具体实现，提供 match_prefix / insert / evict / 加解锁等
                  对外接口（继承自 BasePrefixCache，签名与调度器约定一致）。
"""

import hashlib
import heapq
import logging
import sys
import time
from array import array
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

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
from sglang.srt.mem_cache.events import KVCacheEventMixin
from sglang.srt.mem_cache.utils import get_eviction_strategy, split_node_hash_value

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class RadixKey:
    """前缀缓存的键，封装一段 token 序列及其命名空间信息。

    什么是 bigram（相邻二元组）：
        “bigram” 即“二元组”，指把序列中**相邻的两个 token**绑成一个逻辑单元。
        对序列 [t0, t1, t2, t3] 取 bigram 视图，得到的逻辑单元序列为：
            (t0, t1), (t1, t2), (t2, t3)
        注意相邻二元组之间**共享一个边界 token**（如 t1 同时属于第 1、2 个二元组），
        因此 N 个 token 恰好产生 N-1 个 bigram。这与普通模式（每个 token 自成
        一个逻辑单元，N 个 token 即 N 个单元）不同。
        举例：token 序列 ["我", "爱", "北京"] 的 bigram 单元是
            ("我","爱"), ("爱","北京") —— 3 个 token、2 个二元组。

    is_bigram=True 时：token_ids 仍保存原始 token（N 个 bigram 需要 N+1 个 token）；
    对其切片时，相邻切片会共享一个边界 token（因为二元组天然重叠）。

    bigram 视图主要服务 EAGLE 投机解码：把序列看作
    (t0,t1), (t1,t2), ... 这样相邻成对的逻辑单元来做前缀匹配。
    """

    # 使用 __slots__ 固定属性集合，省去每个实例的 __dict__，降低海量节点下的内存开销。
    __slots__ = ("token_ids", "extra_key", "is_bigram", "limit")

    def __init__(
        self,
        token_ids: array[int],
        extra_key: Optional[str] = None,
        is_bigram: bool = False,
        limit: Optional[int] = None,
    ):
        # token id 序列（两种模式下都存原始 int）
        self.token_ids = token_ids
        # 额外的命名空间键（例如 lora_id、cache_salt）：extra_key 不同的键互不共享前缀
        self.extra_key = extra_key
        # 是否启用 bigram 视图；启用后逻辑长度 = max(0, len(token_ids) - 1)
        self.is_bigram = is_bigram
        # 对原始 token 数的可选上限：效果等同于 token_ids[:limit]，但不做 O(n) 拷贝。
        # None 表示使用全部 token。
        self.limit = limit

    def _raw_len(self) -> int:
        # 考虑 limit 截断后的“原始 token 个数”（尚未换算成 bigram 逻辑长度）。
        n = len(self.token_ids)
        if self.limit is not None and self.limit < n:
            return self.limit
        return n

    def raw_token_ids(self) -> array:
        """返回考虑 `limit` 后的 token_ids（仅在确实被截断时才发生拷贝）。"""
        n = self._raw_len()
        t = self.token_ids
        return t if n == len(t) else t[:n]

    def __len__(self) -> int:
        # 逻辑长度：bigram 模式下为二元组个数（n-1），普通模式下即 token 个数。
        n = self._raw_len()
        if self.is_bigram:
            return n - 1 if n > 0 else 0
        return n

    # TODO(Jialin): 用 numpy 向量化以避免逐个 PyLong 装箱带来的开销
    def __iter__(self) -> Iterator:
        # 迭代逻辑单元：bigram 模式逐个产出 (t_i, t_{i+1})，普通模式逐个产出 token。
        t = self.token_ids
        n = self._raw_len()
        if self.is_bigram:
            for i in range(n - 1 if n > 0 else 0):
                yield (t[i], t[i + 1])
        elif n == len(t):
            yield from t
        else:
            for i in range(n):
                yield t[i]

    def __getitem__(self, idx: Union[int, slice]) -> RadixKey:
        # 先把单个 int 索引归一化成长度为 1 的 slice，这样后续只需处理 slice 一种形态。
        if isinstance(idx, int):
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError(f"RadixKey index out of range: {idx}")
            idx = slice(idx, idx + 1)
        start, stop, step = idx.indices(len(self))
        if step != 1:
            raise ValueError("RadixKey slice step must be 1")

        if self.is_bigram:
            # bigram 区间 [start, stop) 覆盖原始 token 区间 [start, stop + 1)（多取一个边界 token）；
            # 空切片应得到空的原始 token（而不是残留一个悬空的边界 token）。
            raw = self.token_ids[start : stop + 1] if stop > start else array("q")
            return RadixKey(raw, self.extra_key, is_bigram=True)
        return RadixKey(self.token_ids[start:stop], self.extra_key)

    def __repr__(self) -> str:
        # 仅预览前 10 个 token，序列过长时以省略号收尾，避免日志被刷屏。
        preview = self.token_ids[:10]
        return f"RadixKey(extra_key={self.extra_key!r}, token_ids={preview}{'...' if len(self.token_ids) > 10 else ''}, is_bigram={self.is_bigram})"

    def page_aligned(self, page_size: int) -> RadixKey:
        # 把键截断到 page_size 的整数倍长度。page 是命中/插入/淘汰的最小单位，
        # 不足一页的尾部不参与建树。page_size == 1 时无需对齐，直接返回自身。
        if page_size == 1:
            return self
        aligned_len = len(self) // page_size * page_size
        return self[:aligned_len]

    def maybe_to_bigram_view(
        self,
        is_eagle: bool,
        value: Optional[torch.Tensor] = None,
    ) -> Tuple[RadixKey, Optional[torch.Tensor]]:
        # O(1) 操作：只翻转 is_bigram 标志位，而不真正物化出一个二元组列表。
        # value 与原始 token 一一对应，因此需要截断到 bigram 的逻辑长度。
        if is_eagle and not self.is_bigram:
            self.is_bigram = True
            if value is not None:
                value = value[: len(self)]
        return self, value

    def _check_compatible(self, other: RadixKey) -> None:
        # extra_key 不同的键属于不同命名空间，禁止跨命名空间做匹配/比较等操作。
        if self.extra_key != other.extra_key:
            raise ValueError(
                f"RadixKey operations require matching extra_key, but got "
                f"{self.extra_key=} != {other.extra_key=}"
            )

    def match(self, other: RadixKey, page_size: int = 1) -> int:
        """返回与 ``other`` 共享的前缀长度（以逻辑单元计），并向下取整到 ``page_size`` 的整数倍。"""
        self._check_compatible(other)
        t0, t1 = self.token_ids, other.token_ids
        assert type(t0) is type(t1), (type(t0), type(t1))
        n = min(len(t0), len(t1))

        # 用指数（gallop）搜索定位第一个不同的 token：以倍增的窗口前进
        # （每步只做一次 C 层面的切片比较），再在“包含分歧点”的那个窗口里二分查找。
        # 这样在长公共前缀上无需逐 token 的 Python 循环，比线性比较快得多。
        matched_tokens = n
        lo = 0
        step = 1
        while lo < n:
            hi = lo + step if lo + step < n else n
            if t0[lo:hi] != t1[lo:hi]:
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if t0[lo:mid] == t1[lo:mid]:
                        lo = mid
                    else:
                        hi = mid
                matched_tokens = lo
                break
            lo = hi
            step *= 2

        if self.is_bigram:
            # bigram 数 = 匹配的原始 token 数 - 1，再夹到两边的逻辑长度内，避免越界。
            matched = max(0, min(matched_tokens - 1, len(self), len(other)))
            return (matched // page_size) * page_size if page_size > 1 else matched

        matched_tokens = min(matched_tokens, len(self), len(other))
        if page_size == 1:
            return matched_tokens
        return (matched_tokens // page_size) * page_size

    def child_key(self, page_size: int = 1):
        """生成可哈希的 dict 键：取前 ``page_size`` 个逻辑单元，并按 ``extra_key`` 加命名空间前缀。

        RadixCache 用它作为父节点 children 字典的 key 来快速定位子节点，
        因此必须可哈希且能区分不同命名空间。
        """
        t = self.token_ids
        if self.is_bigram:
            if page_size == 1:
                plain = (t[0], t[1])
            else:
                plain = tuple((t[j], t[j + 1]) for j in range(page_size))
        else:
            plain = t[0] if page_size == 1 else tuple(t[:page_size])
        # extra_key 为 None 时直接用 token；否则把它并入键以隔离命名空间。
        return plain if self.extra_key is None else (self.extra_key, plain)

    def hash_page(self, start: int, end: int, prior_hash: Optional[str] = None) -> str:
        """对逻辑单元区间 [start, end) 计算 SHA256；bigram 模式会喂入重叠的 (t_i, t_{i+1}) 字节对。

        prior_hash 用于把前一页的哈希作为种子串联进来，从而让每页哈希隐含其全部前缀，
        这是 HiCache 等按页定位/去重所需的“前缀链式哈希”。
        """
        hasher = hashlib.sha256()
        if prior_hash:
            # 把前一页哈希作为前缀种子，使本页哈希隐含全部历史前缀。
            hasher.update(bytes.fromhex(prior_hash))
        t = self.token_ids
        # 每个 token 以 4 字节小端、无符号方式编码后喂入哈希器。
        if self.is_bigram:
            for j in range(start, end):
                hasher.update(t[j].to_bytes(4, byteorder="little", signed=False))
                hasher.update(t[j + 1].to_bytes(4, byteorder="little", signed=False))
        else:
            for j in range(start, end):
                hasher.update(t[j].to_bytes(4, byteorder="little", signed=False))
        return hasher.hexdigest()


class TreeNode:
    """radix 树的节点，代表一段连续前缀及其对应的 KV 缓存槽位。"""

    # 全局自增计数器，为每个节点分配唯一 id（用于事件上报、调试等）。
    counter = 0

    def __init__(self, id: Optional[int] = None, priority: int = 0):
        # 子节点字典：键为 child_key()，值为子 TreeNode。
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None  # 本节点持有的那段前缀键（相对父节点的增量片段）
        self.value: Optional[torch.Tensor] = None  # 该段前缀在 device 上的 KV 槽位索引；为 None 表示已被淘汰
        self.lock_ref = 0  # 锁引用计数；> 0 表示正被请求使用，不可淘汰
        self.last_access_time = time.monotonic()  # 最近访问时间，供 LRU 等淘汰策略排序
        self.creation_time = time.monotonic()  # 创建时间，供部分淘汰策略使用

        self.hit_count = 0  # 命中次数，供 LFU 等淘汰策略使用
        # host（CPU）侧的引用计数：> 0 时锁住 host 备份不被淘汰，
        # 当节点被某次存储（storage）操作引用时递增。
        self.host_ref_counter = 0
        # KV 缓存在 host（CPU）侧的槽位索引（HiCache 分级缓存的备份）
        self.host_value: Optional[torch.Tensor] = None
        self.write_through_pending_id: Optional[int] = None  # 正在进行的 write-through（写穿透到 host）的标识
        # 本节点各页的哈希值列表（按页计算，惰性填充）
        self.hash_value: Optional[List[str]] = None
        # 优先级，供“优先级感知”的淘汰策略使用
        self.priority = priority

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        # value 为 None 即代表该节点的 device KV 已被淘汰。
        return self.value is None

    @property
    def backuped(self):
        # 是否在 host 侧有备份（HiCache 场景下用于判断能否回载而非重算）。
        return self.host_value is not None

    def protect_host(self):
        """保护 host 侧的值不被淘汰（增加 host 引用计数）。"""
        self.host_ref_counter += 1

    def release_host(self):
        """释放 host 侧的值，使其可被淘汰（减少 host 引用计数）。"""
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            # 计数已为 0 还释放，说明加解锁不配对，属于逻辑错误，直接抛出。
            raise RuntimeError("Host reference counter is already zero.")

    def get_last_hash_value(self) -> Optional[str]:
        """返回本节点最后一页的哈希值（没有则返回 None）。"""
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    def get_prefix_hash_values(self, node: TreeNode) -> List[str]:
        # 递归向上拼接：父节点的全部前缀哈希 + 本节点的哈希，得到从根到此处的完整哈希序列。
        if node is None or node.hash_value is None:
            return []

        return node.get_prefix_hash_values(node.parent) + node.hash_value

    def __lt__(self, other: TreeNode):
        # 定义节点间的“小于”关系：按最近访问时间排序，使其可直接放入淘汰用的最小堆。
        return self.last_access_time < other.last_access_time


class RadixCache(KVCacheEventMixin, BasePrefixCache):
    """基于 radix 树的前缀缓存实现。

    多继承自 KVCacheEventMixin（提供 KV 缓存事件上报能力）与 BasePrefixCache
    （定义统一的对外接口）。所有外部交互都通过 BasePrefixCache 约定的方法进行。
    """

    def __init__(self, params: CacheInitParams):
        # 各项配置均来自 CacheInitParams，统一打包传入以保持构造签名稳定。
        self.disable = params.disable  # 是否禁用前缀复用
        self.req_to_token_pool = params.req_to_token_pool  # 请求 -> token 槽位映射池
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator  # KV 槽位分配器
        self.page_size = params.page_size  # 分页粒度
        self.enable_kv_cache_events = params.enable_kv_cache_events  # 是否上报 KV 缓存事件
        self.is_eagle = params.is_eagle  # 是否为 EAGLE 投机解码（决定是否启用 bigram 视图）
        self.disable_finished_insert = params.disable_finished_insert  # 确定性模式下禁止已完成请求插入树
        self.eviction_policy = params.eviction_policy.lower()  # 淘汰策略名称（如 lru / lfu）

        self.kv_event_queue = []  # 待上报的 KV 缓存事件队列

        if params.enable_metrics:
            self.init_metrics_collector()

        # 推断缓存所在设备：优先取分配器的 device，缺省回退到 CPU。
        if self.token_to_kv_pool_allocator:
            dev = self.token_to_kv_pool_allocator.device
            if isinstance(dev, (str, torch.device)):
                self.device = torch.device(dev)
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        # 按策略名解析出具体的淘汰策略对象（封装“如何为节点打优先级”的逻辑）。
        self.eviction_strategy = get_eviction_strategy(self.eviction_policy)

        # 可淘汰的叶子节点集合：维护这个集合可避免每次淘汰都重新遍历全树找叶子。
        self.evictable_leaves = set()
        self.reset()

    @classmethod
    def create_simulated(
        self,
        disable: bool = False,
        mock_allocator: Optional[Any] = None,
        page_size: int = 1,
        enable_kv_cache_events: bool = False,
    ) -> RadixCache:
        """构造一个不依赖真实内存池的 radix 缓存，仅用于仿真/测试。"""
        params = CacheInitParams(
            disable=disable,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=page_size,
            enable_kv_cache_events=enable_kv_cache_events,
        )
        return RadixCache(params)

    ##### 对外公开接口（Public API） #####

    def reset(self):
        # 用最小优先级初始化根节点，确保任何真实优先级都能覆盖它（根永不被优先淘汰）。
        self.root_node = TreeNode(priority=-sys.maxsize)
        self.root_node.key = RadixKey(token_ids=array("q"), extra_key=None)
        self.root_node.value = []
        self.root_node.host_value = []
        self.root_node.lock_ref = 1  # 根节点永久加锁，永不被淘汰
        self.root_node.hash_value = []
        self.evictable_size_ = 0  # 当前可淘汰 token 计数
        self.protected_size_ = 0  # 当前受保护（被锁）token 计数
        self.evictable_leaves.clear()
        self._empty_match_result = MatchResult(
            device_indices=torch.empty(
                (0,),
                dtype=torch.int64,
                device=self.device,
            ),
            last_device_node=self.root_node,
            last_host_node=self.root_node,
            best_match_node=self.root_node,
        )
        self._record_all_cleared_event()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """在 radix 树中查找 ``key`` 的最长已缓存前缀。

        前缀匹配的逻辑命名空间由 token id 序列与 ``RadixKey`` 携带的可选 ``extra_key``
        共同决定。即便起始 token id 完全相同，只要 ``extra_key`` *不同*，这些条目
        也会被刻意保持隔离、绝不共享前缀节点。该机制的用途包括：

        * 隔离不同 LoRA / adapter ID 的 KV 缓存行。
        * 通过指定不同的 ``extra_key``，把那些本就不应共享状态的请求分开
          （例如不同的采样 salt、缓存版本，或检索增强 RAG 的上下文）。

        Args:
            params (MatchPrefixParams): 包含查找键的参数；键由一串 token id 与可选的
                ``extra_key`` 命名空间标签组成。当 ``page_size > 1`` 时，匹配前会先把
                长度向下截断到 ``page_size`` 的整数倍。传入空键将返回“以根节点为末节点”
                的空结果。

        Returns:
            MatchResult: ``device_indices`` 是一维 ``torch.int64`` 张量，为最长已缓存
            前缀所对应、拼接好的 KV 缓存索引（长度可能为 0）。
            ``last_device_node`` 与 ``last_host_node``（当前两者相同）是表示所匹配前缀
            末端的树节点对象。若匹配恰好终止于某个已存储片段的内部，本方法可能会
            分裂（split）该节点，从而改动内部结构。

        内部副作用:
            * 刷新访问元数据（时间戳），供所配置的淘汰策略使用。
            * 若查找终止于某存储片段内部，会对该节点做一次分裂以暴露精确边界；
              这种结构细化能提升后续匹配效率，且不会复制数据。
        """
        key = params.key
        key, _ = key.maybe_to_bigram_view(self.is_eagle)

        # 禁用前缀复用或键为空时，直接返回预先构造好的空结果（末节点为根）。
        if self.disable or len(key) == 0:
            return self._empty_match_result

        key = key.page_aligned(self.page_size)

        # page 对齐后若长度归零（不足一页），同样视作未命中。
        if len(key) == 0:
            return self._empty_match_result

        # 沿树自根向下匹配，收集途径节点的 value 片段与末端节点。
        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)  # 把各片段拼成一段连续的 device 索引
        else:
            value = self._empty_match_result.device_indices  # 复用空张量，避免新分配
        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
            best_match_node=last_node,
        )

    def insert(self, params: InsertParams) -> InsertResult:
        # 禁用时不入树，命中前缀长度记为 0。
        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        priority = params.priority
        chunked = params.chunked

        key, value = key.maybe_to_bigram_view(self.is_eagle, value)
        key = key.page_aligned(self.page_size)
        if value is not None:
            value = value[: len(key)]  # value 与 key 对齐，丢弃多余尾部
        else:
            # 调试/测试场景下的兜底：直接用 token id 本身充当 value。
            value = torch.tensor(key.token_ids[: len(key)], dtype=torch.int64)

        prefix_len = self._insert_helper(self.root_node, key, value, priority, chunked)
        return InsertResult(prefix_len=prefix_len)

    def cache_finished_req(self, req: Req, is_insert: bool = True):
        """请求结束时缓存其 KV，并释放占用、解锁所持节点。"""
        # 确定性（deterministic）模式下，禁止把已完成请求插入 radix 缓存，
        # 以保证可复现性。
        if self.disable_finished_insert:
            is_insert = False

        kv_committed_len = req.pop_committed_kv_cache()
        if self.disable:
            # 禁用前缀复用时不入树，直接把该请求占用的 KV 槽位全部归还分配器。
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            return

        # 取已确认（committed）部分的完整 token 序列及其对应的 KV 槽位索引。
        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        key_len = len(radix_key)
        values = kv_indices[:key_len].to(dtype=torch.int64, copy=True)

        # 插入树后，radix 缓存会在内存池中持有这些 KV 的一份引用。
        if is_insert:
            priority = getattr(req, "priority", 0) or 0
            result = self.insert(
                InsertParams(key=radix_key, value=values, priority=priority)
            )
            # 释放“本就已在树中”的那段重复 KV（命中前缀部分），只保留树里已有的那一份。
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : result.prefix_len]
            )
        else:
            # 不入树时，释放从受保护长度到对齐长度之间的 KV。
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : key_len]
            )

        # 释放不足一页、未参与建树的对齐尾部。
        self.token_to_kv_pool_allocator.free(kv_indices[key_len:])

        # 释放请求槽位，解开该请求对末端节点持有的缓存锁。
        if req.last_node is not None:
            self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False):
        """请求尚未结束（如分块 prefill 的中间态）时，缓存其已生成的部分。"""
        if self.disable:
            return

        token_ids = req.get_fill_ids()
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        values = kv_indices[: len(radix_key)].to(dtype=torch.int64, copy=True)

        # 插入树后，radix 缓存会在内存池中持有这些 KV 的一份引用。
        result = self.insert(
            InsertParams(
                key=radix_key,
                value=values,
                chunked=chunked,
                priority=getattr(req, "priority", 0) or 0,
            )
        )
        new_prefix_len = result.prefix_len

        # 释放命中前缀部分的重复 KV（这部分树里已有）。
        self.token_to_kv_pool_allocator.free(
            kv_indices[req.cache_protected_len : new_prefix_len]
        )

        # 插入可能分裂/更新了节点，重新匹配一次以拿到最新的索引与末端节点并复用。
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )
        assert len(new_indices) == len(
            radix_key
        ), f"{len(new_indices)=}, {len(radix_key)=}"

        # 把树中（可能已去重/复用）的最新索引写回该请求的 token 映射，覆盖受保护长度之后的部分。
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        # cache_protected_len 不总等于 len(req.prefix_indices)：当 page_size > 1 时，
        # 末尾不足一页的部分会被加入 req.prefix_indices，但这段 KV 索引并未加入树。
        # 这部分需要在下一次 cache_unfinished_req 以及最终的 cache_finished_req 中被释放，
        # 否则会内存泄漏。因此引入 cache_protected_len 字段来确保这段“零头”能被正确释放。
        req.cache_protected_len = len(new_indices)

        # 锁的转移：先解开旧末端节点，再锁住新末端节点，确保正在使用的前缀不被淘汰。
        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # req.prefix_indices 稍后会在 PrefillAdder::add_chunked_req 中使用：
        # - page_size != 1：末尾有不足一页的零头，需保留完整 kv_indices；
        # - eagle 场景：bigram 键只会缓存 len - 1 个 KV 索引。
        if len(new_indices) < len(kv_indices):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices

        req.last_node = new_last_node

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper()

    def evict(self, params: EvictParams) -> EvictResult:
        # 淘汰策略：从所有可淘汰叶子里，按策略优先级用最小堆依次弹出“最该淘汰”的节点，
        # 直到累计释放够 num_tokens 个 token 或无可淘汰节点为止。
        if self.disable:
            return EvictResult()

        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        leaves = list(self.evictable_leaves)
        # 用 (优先级, 节点) 建最小堆：优先级最低者最先被淘汰。
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            # 归还该叶子占用的 KV 槽位，并把它从树中摘除。
            self.token_to_kv_pool_allocator.free(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            # 摘除后，若父节点变成了“无子且未加锁”的新叶子，则它也成为候选，入堆。
            if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

            self._record_remove_event(x)

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        # 从 node 一路向根加锁：保证从根到该节点的整条前缀路径在使用期间都不被淘汰。
        if self.disable:
            return IncLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            # 节点从“未锁”变为“加锁”的那一刻，其 KV 由可淘汰转为受保护，更新两个计数。
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
        # 从 node 一路向根解锁，是 inc_lock_ref 的逆操作。
        if self.disable:
            return DecLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            # 节点从“锁定 1 次”变回“未锁”的那一刻，其 KV 重新变为可淘汰，更新两个计数。
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            # 防御性断言：除根节点外不应出现 parent 为空；否则说明该请求持有了别的树的节点。
            if node.parent is None:
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
        return DecLockRefResult(delta=delta)

    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        # 受保护大小：指被锁住、当前不可淘汰的那部分缓存的 token 数。
        return self.protected_size_

    def all_values_flatten(self):
        # 深度优先遍历整棵树，把所有节点的 value 拼接成一个一维张量（调试/统计用）。
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values)

    ##### 内部辅助函数（Internal Helper Functions） #####

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        # 自给定节点起，沿 children 逐段匹配 key，返回途径各段的 value 列表与末端节点。
        access_time = time.monotonic()
        node.last_access_time = access_time

        child_key = key.child_key(self.page_size)

        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = access_time
            prefix_len = child.key.match(key, page_size=self.page_size)
            if prefix_len < len(child.key):
                # 只匹配到子节点片段的一部分：在精确边界处分裂出 new_node，取其 value 后停止。
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            else:
                # 完整吃掉该子节点片段：累加其 value，下沉到该子节点，继续匹配剩余 key。
                value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = key.child_key(self.page_size)

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # 在 split_len 处把 child 拆成两段，插入一个中间节点：new_node -> child。
        # 结构变为：parent -> new_node(前 split_len) -> child(剩余部分)。
        # new_node 继承 child 的优先级（它代表二者共享的前缀）。
        new_node = TreeNode(priority=child.priority)
        new_node.hit_count = child.hit_count
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref  # 继承锁计数，保证拆分不破坏“受保护”语义
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len].clone()
        # 把 child 收缩为后半段，并把它的父指针改挂到 new_node 下。
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:].clone()
        new_node.parent.children[key.child_key(self.page_size)] = new_node

        # 若哈希已计算过则一并按 split_len 拆分，否则保持为 None（惰性计算）。
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        return new_node

    def _inc_hit_count(self, node: TreeNode, chunked: bool = False):
        # 跳过分块（chunked）请求的命中计数更新：避免“自我引用式膨胀”——
        # 即一个分块请求对它自己在前几个分块中创建的节点反复累加 hit_count。
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
        # 把 None 优先级归一化为 0。
        if priority is None:
            priority = 0
        access_time = time.monotonic()
        node.last_access_time = access_time
        # 沿路径更新优先级（取 max，使更高优先级向上传播）。
        node.priority = max(node.priority, priority)
        if len(key) == 0:
            return 0

        child_key = key.child_key(self.page_size)

        # 先沿已有路径吃掉公共前缀，total_prefix_length 累计命中的已有前缀长度。
        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = access_time
            prefix_len = node.key.match(key, page_size=self.page_size)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                # 只匹配到该节点片段的一部分：分裂出共享前缀节点，新插入内容将挂到它下面。
                new_node = self._split_node(node.key, node, prefix_len)
                new_node.priority = max(new_node.priority, priority)
                self._inc_hit_count(new_node, chunked)
                node = new_node
            else:
                node.priority = max(node.priority, priority)
                self._inc_hit_count(node, chunked)
            if len(key):
                child_key = key.child_key(self.page_size)

        # key 仍有剩余：把这段“树中尚不存在”的新前缀挂为一个新子节点。
        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            self._inc_hit_count(new_node, chunked)
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)  # 新增内容默认可淘汰，更新计数
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)
            # 哈希在事件上报时再惰性计算，此处先登记一次 store 事件。
            self._record_store_event(new_node)
        return total_prefix_length

    def _print_helper(self, node: TreeNode, indent: int):
        """以人类可读的格式打印 radix 树（调试用）。"""
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

                assert key == child.key.child_key(
                    self.page_size
                ), f"{key=}, {child.key.child_key(self.page_size)=}"

    def _delete_leaf(self, node):
        # 从父节点的 children 中摘除该叶子，并同步维护可淘汰计数与可淘汰叶子集合。
        key = node.key.child_key(self.page_size)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.evictable_size_ -= len(node.key)
        if node in self.evictable_leaves:
            self.evictable_leaves.remove(node)
        # 摘除子节点后，父节点可能新晋为可淘汰叶子，需重新评估其状态。
        self._update_leaf_status(node.parent)

    def _update_leaf_status(self, node: TreeNode):
        # 重新评估 node 是否属于“可淘汰叶子”，并据此增删 evictable_leaves 集合。
        # 只有“未被淘汰、未加锁、且没有任何在册（未淘汰）子节点”的节点才算可淘汰叶子。
        if node.evicted or node.lock_ref > 0:
            if node in self.evictable_leaves:
                self.evictable_leaves.remove(node)
            return

        for child in node.children.values():
            if not child.evicted:
                # 仍有在册子节点 => 它不是叶子，不能被淘汰。
                if node in self.evictable_leaves:
                    self.evictable_leaves.remove(node)
                return

        if node not in self.evictable_leaves:
            self.evictable_leaves.add(node)

    def _total_size_helper(self):
        # 迭代式 DFS 统计整棵树中仍在 device 上（未被淘汰）的 KV token 总数。
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


# 直接运行本文件时的小型自测 demo：插入若干序列、打印树、再做一次前缀匹配。
if __name__ == "__main__":
    tree = RadixCache.create_simulated()

    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [1, 2, 3]))))
    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [1, 2, 3]))))
    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [1, 2, 4, 5]))))
    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [1, 2, 4, 5, 6, 7]))))
    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [8, 9, 10, 11, 12]))))
    tree.pretty_print()

    print(
        tree.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=array("q", [1, 2, 3, 13, 14])))
        )
    )
