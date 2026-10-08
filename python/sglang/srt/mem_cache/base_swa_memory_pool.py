import abc
from typing import List, Tuple

import torch

from sglang.srt.mem_cache.memory_pool import KVCache


class BaseSWAKVPool(KVCache):
    """SWA（滑动窗口注意力）类 KV 池的抽象基类（ABC）。

    【背景】
    一些模型采用「混合注意力」：一部分层是全局注意力（full attention，需要保留
    整段序列的 KV），另一部分层是滑动窗口注意力（SWA，只需保留最近 window_size
    个 token 的 KV）。两类层的 KV 生命周期与容量需求完全不同，因此用两个物理
    子池分别承载：

    - ``full_kv_pool``：承载 full attention 层的 KV（具体子类提供）；
    - ``swa_kv_pool``：承载 SWA 层的 KV（本基类要求子类暴露该属性）。

    分配器 ``SWATokenToKVPoolAllocator`` 会为同一个逻辑 token 分别从两个子池各
    分配一个槽位，并把对应关系记录在 ``full_to_swa_index_mapping``
    （full 池槽位下标 -> SWA 池槽位下标）中，从而支持两个池之间的下标互译。

    本基类被 ``SWATokenToKVPoolAllocator`` 以及 PD 分离（disaggregation）路径
    使用，用于把 SWA 状态与 full KV 状态分开管理。

    子类需实现：
    - ``register_mapping``：注册 full -> swa 的槽位下标映射表；
    - ``translate_loc_from_full_to_swa``：把 full 池槽位下标翻译为 SWA 池槽位下标；
    - ``get_state_buf_infos``：返回 SWA 状态缓冲的（地址/长度/条目长度）描述。
    """

    # 子类必须暴露的 SWA 子池；其本身也是一个 KVCache 实例。
    swa_kv_pool: KVCache

    @abc.abstractmethod
    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor) -> None:
        """注册「full 池槽位下标 -> SWA 池槽位下标」的映射表。

        由分配器在构造完成后调用一次；之后 ``translate_loc_from_full_to_swa``
        就依赖这张表做 O(1) 的向量化翻译。

        参数：
            full_to_swa_index_mapping: int64 张量，长度通常为
                ``full_size + page_size + 1``；第 i 项表示「full 池第 i 个槽位
                对应 SWA 池的哪个槽位」。末尾额外追加一个哨兵值 ``-1``，使得
                full 池的 ``-1``（无效槽位）也能映射为 SWA 池的 ``-1``。

        返回：
            None（仅记录引用，不做数据拷贝）。

        示例：
            >>> mapping = torch.tensor([0, 3, 5, 4, -1])  # 最后一个 -1 是哨兵
            >>> pool.register_mapping(mapping)           # 仅注册，无返回值
            >>> pool.full_to_swa_index_mapping is mapping
            True
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor) -> torch.Tensor:
        """把一批 full 池槽位下标翻译为对应的 SWA 池槽位下标。

        采用张量索引（gather）实现，等价于 ``mapping[kv_indices]``；其中 ``-1``
        经末尾哨兵项映射后仍为 ``-1``。

        参数：
            kv_indices: 一维（或任意形状）的 int64 张量，元素是 full 池槽位下标。

        返回：
            与 ``kv_indices`` 同形状的 int64 张量，元素是对应的 SWA 池槽位下标。

        示例：
            >>> mapping = torch.tensor([7, 1, 9, -1])  # 末位 -1 为哨兵
            >>> pool.register_mapping(mapping)
            >>> kv_indices = torch.tensor([0, 2, 3])
            >>> pool.translate_loc_from_full_to_swa(kv_indices)
            tensor([ 7,  9, -1])
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """返回 SWA 子池「连续缓冲（contiguous buffer）」的物理描述信息。

        主要用于 PD 分离 / memory saver 等需要按物理地址整体搬运 KV 的场景：
        上层拿到每个 buffer 的起始地址、总字节长度、单个条目（通常是一个 token
        槽位）的字节长度后，即可做整块 D2H/H2D 拷贝或跨进程传输。

        参数：
            无。

        返回：
            三元组 ``(data_ptrs, data_lens, item_lens)``：
            - ``data_ptrs``: 每个 SWA 层缓冲张量的内存地址（int）；
            - ``data_lens``: 每个缓冲张量的总字节数（int）；
            - ``item_lens``: 每个「条目」（一个 token 槽位 / 一页）的字节数（int）。

        示例：
            >>> ptrs, lens, items = pool.get_state_buf_infos()
            >>> len(ptrs) == pool.swa_kv_pool.layer_num  # 每层一个 buffer
            True
            >>> items[0]  # 单槽位字节数
            8192
        """
        raise NotImplementedError()
