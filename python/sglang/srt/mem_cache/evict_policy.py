from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode


class EvictionStrategy(ABC):
    """淘汰策略抽象基类。

    Radix Cache 在显存（或缓存空间）不足时需要淘汰部分缓存节点。
    淘汰时会把所有可淘汰的叶子节点按 ``get_priority`` 返回的优先级值排序，
    **优先级值越小的节点越先被淘汰**。不同子类通过实现 ``get_priority``
    来定义不同的淘汰顺序（LRU / LFU / FIFO 等）。
    """

    @abstractmethod
    def get_priority(self, node: TreeNode) -> Union[float, Tuple]:
        """返回节点的淘汰优先级。值越小越先被淘汰。

        返回值可以是单个浮点数，也可以是元组（用于多关键字排序，
        元组按字典序逐项比较）。
        """
        pass


class LRUStrategy(EvictionStrategy):
    """LRU（最近最少使用）：最久未被访问的节点优先淘汰。"""

    def get_priority(self, node: TreeNode) -> float:
        # 直接用最近一次访问时间作为优先级：时间越早（值越小）越先淘汰。
        return node.last_access_time


class LFUStrategy(EvictionStrategy):
    """LFU（最不经常使用）：命中次数最少的节点优先淘汰。"""

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        # 先比命中次数 hit_count（越小越先淘汰），命中次数相同时再按 LRU
        # 比最近访问时间，从而在冷门节点之间也保持稳定的淘汰顺序。
        return (node.hit_count, node.last_access_time)


class FIFOStrategy(EvictionStrategy):
    """FIFO（先进先出）：最早创建的节点优先淘汰。"""

    def get_priority(self, node: TreeNode) -> float:
        # 用节点创建时间作为优先级：创建越早（值越小）越先淘汰。
        return node.creation_time


class MRUStrategy(EvictionStrategy):
    """MRU（最近最常使用）：最近被访问的节点反而优先淘汰。"""

    def get_priority(self, node: TreeNode) -> float:
        # 对最近访问时间取负号，使「时间越晚」对应「优先级值越小」，
        # 从而让最近刚访问过的节点最先被淘汰（与 LRU 相反）。
        return -node.last_access_time


class FILOStrategy(EvictionStrategy):
    """FILO（后进先出）：最近创建的节点优先淘汰。"""

    def get_priority(self, node: TreeNode) -> float:
        # 对创建时间取负号，使「创建越晚」对应「优先级值越小」，
        # 从而让最新创建的节点最先被淘汰（与 FIFO 相反）。
        return -node.creation_time


class PriorityStrategy(EvictionStrategy):
    """优先级感知淘汰：优先级低的节点先被淘汰，同优先级内再按 LRU。"""

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        # 返回 (priority, last_access_time)：先按节点自身的 priority 排序，
        # priority 越小越先淘汰；priority 相同时再按最近访问时间走 LRU。
        return (node.priority, node.last_access_time)


class SLRUStrategy(EvictionStrategy):
    """SLRU（分段 LRU）：把缓存分为「试用段」和「保护段」两个分段。

    节点首次进入时位于试用段，命中次数达到阈值后晋升到保护段。
    淘汰时总是先淘汰试用段的节点，从而保护「被多次命中」的热点数据
    不被偶发的一次性访问挤出缓存。
    """

    def __init__(self, protected_threshold: int = 2):
        # protected_threshold：晋升到保护段所需的命中次数阈值。
        self.protected_threshold = protected_threshold

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        # 优先级逻辑：
        # 值越小 = 越先被淘汰。
        #
        # 分段 0（试用段 Probationary）：hit_count < 阈值
        # 分段 1（保护段 Protected）：hit_count >= 阈值
        #
        # 元组比较：(segment, last_access_time)
        # 分段 0 的节点总是先于分段 1 的节点被淘汰；
        # 在同一分段内，越久未访问（时间越小）的节点越先被淘汰。

        is_protected = 1 if node.hit_count >= self.protected_threshold else 0
        return (is_protected, node.last_access_time)
