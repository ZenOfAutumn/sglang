import logging
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    KVCache,
    MHATokenToKVPool,
    MLATokenToKVPool,
    unwrap_write_loc,
)
from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool

logger = logging.getLogger(__name__)
# 1 GB 对应的字节数，用于把字节换算为 GB 打印日志。
GB = 1024 * 1024 * 1024


class SWAKVPool(BaseSWAKVPool):
    """由「可独立配置」的 full 子池与 SWA 子池组合而成的混合 KV 缓存池。

    默认实现是「两个 MHA 池」（``MHATokenToKVPool``）；通过传入
    ``full_kv_pool_class`` 与 ``swa_kv_pool_class`` 可以切换到其它 KV 缓存族
    （例如 MLA / DSA），而无需在「池选择器」中为不同模型编写特化逻辑。

    设计要点：
    - ``full_kv_pool``：承载所有 full attention（全局注意力）层的 KV；
    - ``swa_kv_pool``：承载所有 SWA（滑动窗口注意力）层的 KV，容量通常更小；
    - ``layers_mapping``：全局 layer_id -> (子池内的层下标, 是否 SWA 层)。
      因为子池只持有自己那部分层的连续下标，所以需要做一层翻译；
    - ``full_to_swa_index_mapping``：full 池槽位 -> SWA 池槽位 的映射表，
      由分配器 ``SWATokenToKVPoolAllocator`` 通过 ``register_mapping`` 注入。

    简单示例：
        >>> pool = SWAKVPool(
        ...     size=1024, size_swa=256, page_size=1, dtype=torch.bfloat16,
        ...     head_num=8, head_dim=128,
        ...     swa_attention_layer_ids=[1, 3], full_attention_layer_ids=[0, 2],
        ...     device="cuda",
        ... )
        >>> pool.get_key_buffer(0).shape   # 第 0 层是 full 层，容量较大
        torch.Size([1024, 8, 128])
        >>> pool.get_key_buffer(1).shape   # 第 1 层是 SWA 层，容量较小
        torch.Size([256, 8, 128])
    """

    def __init__(
        self,
        size: int,
        size_swa: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        swa_attention_layer_ids: List[int],
        full_attention_layer_ids: List[int],
        device: str,
        token_to_kv_pool_class: KVCache = MHATokenToKVPool,
        full_kv_pool_class: Optional[type] = None,
        swa_kv_pool_class: Optional[type] = None,
        full_kv_pool_kwargs: Optional[dict] = None,
        swa_kv_pool_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """构造 full 子池与 SWA 子池，并建立全局层号到子池层号的映射。

        参数：
            size: full 子池的槽位总数（full 层可缓存的 token 数）。
            size_swa: SWA 子池的槽位总数（SWA 层可缓存的 token 数）。
            page_size: 分页大小（每页 token 数，paged KV 场景 > 1）。
            dtype: KV 的计算 dtype（如 bf16/fp16/fp8）。
            head_num: KV 头数（GQA 下为 KV head 数）。
            head_dim: 每个 K 头的维度。
            swa_attention_layer_ids: SWA 层的全局 layer_id 列表。
            full_attention_layer_ids: full attention 层的全局 layer_id 列表。
            device: 设备字符串（如 ``"cuda"``）。
            token_to_kv_pool_class: 默认的子池类型（同时用于 full/SWA 子池）。
            full_kv_pool_class: 可选的 full 子池类型（默认取 ``token_to_kv_pool_class``）。
            swa_kv_pool_class: 可选的 SWA 子池类型（默认取 ``token_to_kv_pool_class``）。
            full_kv_pool_kwargs: 传给 full 子池构造函数的额外参数（默认自动组装）。
            swa_kv_pool_kwargs: 传给 SWA 子池构造函数的额外参数（默认自动组装）。
            **kwargs: 透传给子池构造函数的额外参数（如 MLA 的 ``kv_lora_rank`` 等）。

        返回：
            无（构造函数就地初始化实例）。

        示例：
            >>> pool = SWAKVPool(
            ...     size=4096, size_swa=1024, page_size=1, dtype=torch.bfloat16,
            ...     head_num=8, head_dim=128,
            ...     swa_attention_layer_ids=[1], full_attention_layer_ids=[0],
            ...     device="cuda",
            ... )
            >>> pool.layer_num      # 1 个 full 层 + 1 个 SWA 层
            2
            >>> pool.layers_mapping # {全局层号: (子池内层号, 是否 SWA)}
            {0: (0, False), 1: (0, True)}
        """
        self.size = size
        self.size_swa = size_swa
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.device = device
        # SWA 层与 full 层的数量，以及二者之和（本池负责的总层数）。
        self.swa_layer_nums = len(swa_attention_layer_ids)
        self.full_layer_nums = len(full_attention_layer_ids)
        self.layer_num = self.full_layer_nums + self.swa_layer_nums
        self.start_layer = 0
        self.page_size = page_size
        # 逐层传输完成计数器；由上层通过 register_layer_transfer_counter 注入。
        self.layer_transfer_counter = None

        # PD 分离 + NVLink 场景下的自定义内存池（不支持时返回 None）。
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

        # 未显式指定子池类型时，full/SWA 都退化为 token_to_kv_pool_class（默认 MHA）。
        full_kv_pool_class = full_kv_pool_class or token_to_kv_pool_class
        swa_kv_pool_class = swa_kv_pool_class or token_to_kv_pool_class
        # 两个子池共用的基础构造参数。
        common_kwargs = {
            "page_size": page_size,
            "enable_memory_saver": False,
            "device": device,
        }
        if full_kv_pool_kwargs is None:
            # 未显式提供时，按默认规则组装 full 子池参数。
            full_kv_pool_kwargs = {
                **common_kwargs,
                "head_num": head_num,
                "head_dim": head_dim,
                "allocation_label": "Full",  # 分配标签，便于日志区分两个子池
                **kwargs,
            }
            # full 子池只用 full 的 head 配置，剔除 SWA 专用参数避免误传。
            full_kv_pool_kwargs.pop("swa_head_num", None)
            full_kv_pool_kwargs.pop("swa_head_dim", None)
            full_kv_pool_kwargs.pop("swa_v_head_dim", None)
        if swa_kv_pool_kwargs is None:
            # 未显式提供时，按默认规则组装 SWA 子池参数。
            swa_kv_pool_kwargs = {
                **common_kwargs,
                "head_num": head_num,
                "head_dim": head_dim,
                "allocation_label": "SWA",  # 分配标签，便于日志区分两个子池
                **kwargs,
            }

        # full 子池：容量 size，层数 full_layer_nums。
        self.full_kv_pool = full_kv_pool_class(
            size=size,
            dtype=dtype,
            layer_num=self.full_layer_nums,
            **full_kv_pool_kwargs,
        )
        # SWA 子池：容量 size_swa，层数 swa_layer_nums。
        self.swa_kv_pool = swa_kv_pool_class(
            size=size_swa,
            dtype=dtype,
            layer_num=self.swa_layer_nums,
            **swa_kv_pool_kwargs,
        )
        # 以下三个字段用于对齐 MLA/DSA 池的接口：
        self.dsa_kv_cache_store_fp8 = False
        self.kv_cache_dim = None
        self.index_head_dim = None
        if isinstance(self.full_kv_pool, MLATokenToKVPool):
            # MLA 池可能启用 DSA 的 fp8 存储，透出相关维度信息。
            self.dsa_kv_cache_store_fp8 = self.full_kv_pool.dsa_kv_cache_store_fp8
            self.kv_cache_dim = self.full_kv_pool.kv_cache_dim
        if isinstance(self.full_kv_pool, DSATokenToKVPool):
            self.index_head_dim = self.full_kv_pool.index_head_dim
        # {全局 layer_id: (子池内的层下标, 是否 SWA 层)}
        self.layers_mapping: Dict[int, Tuple[int, bool]] = {}
        for full_attn_layer_id, global_layer_id in enumerate(full_attention_layer_ids):
            self.layers_mapping[global_layer_id] = (full_attn_layer_id, False)
        for swa_layer_id, global_layer_id in enumerate(swa_attention_layer_ids):
            self.layers_mapping[global_layer_id] = (swa_layer_id, True)
        # full -> SWA 槽位下标映射表，由分配器稍后 register_mapping 注入。
        self.full_to_swa_index_mapping: Optional[torch.Tensor] = None

        # 统计两个子池合计占用的显存（K + V，单位 GB）。
        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB
        logger.info(
            f"SWAKVPool {'VA upper bound' if self.post_capture_active else 'mem usage'}: {self.mem_usage:.2f} GB, swa size: {self.size_swa}, full size: {self.size}"
        )

    @property
    def post_capture_active(self) -> bool:
        """是否走「捕获后（post-capture）按需 backing」的虚拟地址（VA）路径。

        两个子池共享该标志，因此只要取 full 子池的标志即可（返回 True 表示当前
        为 VA-backed 模式，此时 ``size``/``size_swa`` 只是上界，实际物理后端在
        ``finalize_backing`` 时才确定）。

        参数：
            无（属性）。
        返回：
            bool。
        示例：
            >>> pool.post_capture_active
            False
        """
        return self.full_kv_pool.post_capture_active

    @property
    def post_capture_backed_bytes(self) -> int:
        """两个子池当前「已物理 backend」的 KV 字节数之和（仅 post-capture 模式有意义）。

        参数：
            无（属性）。
        返回：
            int，单位为字节。
        示例：
            >>> pool.post_capture_backed_bytes
            17179869184
        """
        return (
            self.full_kv_pool.post_capture_backed_bytes
            + self.swa_kv_pool.post_capture_backed_bytes
        )

    def finalize_backing(self, config) -> None:
        """把两个子池分别 back 到 post-capture 的最终尺寸，并回写本池的容量记录。

        参数：
            config: 含 ``full_max_total_num_tokens`` 与 ``swa_max_total_num_tokens``
                的配置对象，分别给出 full/SWA 子池最终可用的 token 数。
        返回：
            None。
        示例：
            >>> pool.finalize_backing(config)
            >>> pool.size, pool.size_swa
            (4096, 1024)
        """
        # 按最终 token 数分别 back 两个子池的物理内存。
        self.full_kv_pool._finalize_backing_tokens(config.full_max_total_num_tokens)
        self.swa_kv_pool._finalize_backing_tokens(config.swa_max_total_num_tokens)
        self.size = int(config.full_max_total_num_tokens)
        self.size_swa = int(config.swa_max_total_num_tokens)

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor):
        """注册 full -> SWA 槽位下标映射表（由分配器注入）。

        参数：
            full_to_swa_index_mapping: int64 张量，含义见 ``BaseSWAKVPool.register_mapping``。
        返回：
            None。
        示例：
            >>> pool.register_mapping(torch.tensor([0, 3, 5, 4, -1]))
            >>> pool.full_to_swa_index_mapping.shape
            torch.Size([5])
        """
        self.full_to_swa_index_mapping = full_to_swa_index_mapping

    def register_layer_transfer_counter(self, layer_transfer_counter):
        """注册逐层 KV 传输完成计数器。

        等待（wait）只在当前这个「外壳（wrapper）」上发生，内层子池不能再各自等待，
        因此这里先把计数器记在本对象上，再把两个子池的计数器显式置为 None。

        参数：
            layer_transfer_counter: 逐层传输完成计数器（``LayerDoneCounter``），
                为 None 时表示关闭逐层同步。
        返回：
            None。
        示例：
            >>> pool.register_layer_transfer_counter(counter)
            >>> pool.full_kv_pool.layer_transfer_counter is None
            True
        """
        # 等待由本 wrapper 统一负责；内层子池不得再次等待，避免重复同步。
        self.layer_transfer_counter = layer_transfer_counter
        self.full_kv_pool.register_layer_transfer_counter(None)
        self.swa_kv_pool.register_layer_transfer_counter(None)

    def _wait_for_layer(self, layer_id: int):
        """若启用了逐层传输计数器，则等待该层 KV 搬运完成后再继续。

        参数：
            layer_id: 全局 layer_id。
        返回：
            None。
        示例：
            >>> pool._wait_for_layer(3)  # 关闭同步时直接返回，无副作用
        """
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    def get_kv_size_bytes(self):
        """返回两个子池合计的 (K 字节数, V 字节数)。

        子池可能只返回单一 KV 大小（非二元组），这里统一归一化为 ``(size, 0)``，
        以便跨池求和。

        参数：
            无。
        返回：
            Tuple[int, int]：``(k_size_bytes, v_size_bytes)``。
        示例：
            >>> pool.get_kv_size_bytes()
            (8589934592, 8589934592)
        """

        def split_size(pool):
            # 兼容「返回 (K, V) 二元组」与「返回单一 KV 大小」两种子池实现。
            size = pool.get_kv_size_bytes()
            return size if isinstance(size, tuple) else (size, 0)

        k_size, v_size = split_size(self.full_kv_pool)
        k_size_swa, v_size_swa = split_size(self.swa_kv_pool)
        return k_size + k_size_swa, v_size + v_size_swa

    def is_mla(self) -> bool:
        """判断两个子池是否都是 MLA 池。

        参数：
            无。
        返回：
            bool：当且仅当 full/SWA 子池均为 ``MLATokenToKVPool`` 时为 True。
        示例：
            >>> pool.is_mla()
            False
        """
        return isinstance(self.full_kv_pool, MLATokenToKVPool) and isinstance(
            self.swa_kv_pool, MLATokenToKVPool
        )

    def get_contiguous_buf_infos(self):
        """返回 full 子池的「连续缓冲」物理描述信息。

        本池对外只暴露 full 子池的 contiguous buf infos（SWA 子池的信息通过
        ``get_state_buf_infos`` 单独获取），用于 PD 分离等按物理地址搬运的场景。

        参数：
            无。
        返回：
            Tuple[List[int], List[int], List[int]]：``(data_ptrs, data_lens, item_lens)``。
        示例：
            >>> ptrs, lens, items = pool.get_contiguous_buf_infos()
            >>> len(ptrs) == pool.full_kv_pool.layer_num
            True
        """
        full_kv_data_ptrs, full_kv_data_lens, full_kv_item_lens = (
            self.full_kv_pool.get_contiguous_buf_infos()
        )
        return (
            full_kv_data_ptrs,
            full_kv_data_lens,
            full_kv_item_lens,
        )

    def get_kv_scale_buf_infos(self):
        """返回 full 子池的 KV 量化 scale 缓冲描述信息（非量化池返回空/None）。

        参数：
            无。
        返回：
            与 full 子池 ``get_kv_scale_buf_infos()`` 一致（通常是 ptrs/lens 等）。
        示例：
            >>> pool.get_kv_scale_buf_infos()
            ([], [], [])
        """
        return self.full_kv_pool.get_kv_scale_buf_infos()

    def get_swa_kv_scale_buf_infos(self):
        """返回 SWA 子池的 KV 量化 scale 缓冲描述信息（非量化池返回空/None）。

        参数：
            无。
        返回：
            与 SWA 子池 ``get_kv_scale_buf_infos()`` 一致。
        示例：
            >>> pool.get_swa_kv_scale_buf_infos()
            ([], [], [])
        """
        return self.swa_kv_pool.get_kv_scale_buf_infos()

    def get_state_buf_infos(self):
        """返回 SWA 子池的「连续缓冲」物理描述信息。

        即 ``BaseSWAKVPool.get_state_buf_infos`` 的具体实现：取 SWA 子池的
        contiguous buf infos，供 PD 分离等按地址搬运 SWA 状态。

        参数：
            无。
        返回：
            Tuple[List[int], List[int], List[int]]：SWA 子池的
            ``(data_ptrs, data_lens, item_lens)``。
        示例：
            >>> ptrs, lens, items = pool.get_state_buf_infos()
            >>> len(ptrs) == pool.swa_kv_pool.layer_num
            True
        """
        swa_kv_data_ptrs, swa_kv_data_lens, swa_kv_item_lens = (
            self.swa_kv_pool.get_contiguous_buf_infos()
        )

        return swa_kv_data_ptrs, swa_kv_data_lens, swa_kv_item_lens

    def get_key_buffer(self, layer_id: int):
        """按全局 layer_id 取该层的 K 缓冲张量。

        先做逐层同步（``_wait_for_layer``），再根据 ``layers_mapping`` 判断该层
        属于 SWA 子池还是 full 子池，并把全局层号翻译为子池内的层号后转发。

        参数：
            layer_id: 全局 layer_id。
        返回：
            torch.Tensor：该层 K 缓冲（形状如 ``[size(+page), head_num, head_dim]``）。
        示例：
            >>> pool.get_key_buffer(1).shape   # 第 1 层是 SWA 层
            torch.Size([256, 8, 128])
        """
        self._wait_for_layer(layer_id)
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        if is_swa_layer:
            return self.swa_kv_pool.get_key_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_key_buffer(layer_id_pool)

    def get_value_buffer(self, layer_id: int):
        """按全局 layer_id 取该层的 V 缓冲张量（转发规则同 ``get_key_buffer``）。

        参数：
            layer_id: 全局 layer_id。
        返回：
            torch.Tensor：该层 V 缓冲张量。
        示例：
            >>> pool.get_value_buffer(0).shape
            torch.Size([1024, 8, 128])
        """
        self._wait_for_layer(layer_id)
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        if is_swa_layer:
            return self.swa_kv_pool.get_value_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_value_buffer(layer_id_pool)

    def get_kv_buffer(self, layer_id: int):
        """按全局 layer_id 一次取回该层的 (K, V) 两个缓冲张量。

        参数：
            layer_id: 全局 layer_id。
        返回：
            Tuple[torch.Tensor, torch.Tensor]：``(k_buffer, v_buffer)``。
        示例：
            >>> k, v = pool.get_kv_buffer(1)
            >>> k.shape, v.shape
            (torch.Size([256, 8, 128]), torch.Size([256, 8, 128]))
        """
        self._wait_for_layer(layer_id)
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        if is_swa_layer:
            return self.swa_kv_pool.get_kv_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_kv_buffer(layer_id_pool)

    def get_kv_scale_buffer(self, layer_id: int):
        """按全局 layer_id 取该层的 KV 量化 scale 缓冲（非量化池可能不支持）。

        参数：
            layer_id: 全局 layer_id。
        返回：
            与对应子池 ``get_kv_scale_buffer`` 一致（通常是 ``(k_scale, v_scale)``）。
        示例：
            >>> pool.get_kv_scale_buffer(1)
            (None, None)
        """
        self._wait_for_layer(layer_id)
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        if is_swa_layer:
            return self.swa_kv_pool.get_kv_scale_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_kv_scale_buffer(layer_id_pool)

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor) -> torch.Tensor:
        """把 full 池槽位下标翻译为 SWA 池槽位下标。

        依赖已注册的 ``full_to_swa_index_mapping``；因为映射表末尾追加了哨兵
        ``-1``，所以 ``kv_indices`` 中的 ``-1`` 会映射回 ``-1``（保持「无效」语义）。

        参数：
            kv_indices: 元素为 full 池槽位下标的张量（可含 -1）。
        返回：
            torch.Tensor：同形状的 SWA 池槽位下标张量。
        示例：
            >>> pool.register_mapping(torch.tensor([7, 1, 9, -1]))
            >>> pool.translate_loc_from_full_to_swa(torch.tensor([0, 2, -1]))
            tensor([ 7,  9, -1])
        """
        assert self.full_to_swa_index_mapping is not None
        # kv_indices 中的 -1 会经由映射表末尾追加的哨兵项映射回 -1。
        return self.full_to_swa_index_mapping[kv_indices]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ):
        """把本步新算出的 K/V 回填到对应子池的指定槽位。

        关键点：对于 SWA 层，写入位置使用「已翻译好的 SWA loc」（``swa_loc``），
        该翻译由注意力 backend 在每次前向时统一完成一次；``set_kv_buffer`` 内部
        绝不再做翻译。``loc_info`` 既可能是裸 loc 张量，也可能是打包了
        （full loc, swa loc, full loc）的 ``KVWriteLoc``。

        参数：
            layer: 当前注意力层（用其 ``layer_id`` 定位子池）。
            loc_info: 写入位置，或封装了 full/swa 两套位置的对象。
            cache_k: 本步新算出的 K。
            cache_v: 本步新算出的 V。
            k_scale: K 的量化 scale（默认 1.0）。
            v_scale: V 的量化 scale（默认 1.0）。
        返回：
            None。
        示例：
            >>> pool.set_kv_buffer(layer, swa_loc_tensor, k, v)  # SWA 层
            >>> pool.set_kv_buffer(layer, full_loc_tensor, k, v)  # full 层
        """
        # loc_info 打包了 full loc 与「预先翻译好的」SWA loc。
        loc, swa_loc, _ = unwrap_write_loc(loc_info)
        layer_id = layer.layer_id
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        pool = self.swa_kv_pool if is_swa_layer else self.full_kv_pool
        if is_swa_layer:
            # swa_loc 是 full->SWA 的翻译结果，由注意力 backend 每步前向计算一次；
            # set_kv_buffer 内部从不自行翻译。
            assert swa_loc is not None
            loc = swa_loc
        if isinstance(pool, MLATokenToKVPool):
            # MLA 池只需 loc（K/V 合并在 kv_buffer 中），不接收 scale。
            pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                layer_id_override=layer_id_pool,
            )
        else:
            pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id_pool,
            )

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        """MLA 专用：把 nope/rope 两段 K 回填到 MLA 子池的指定槽位。

        语义与 ``set_kv_buffer`` 一致：SWA 层使用预翻译好的 ``swa_loc``，并要求
        目标子池确实是 MLA 池，否则抛 ``TypeError``。

        参数：
            layer: 当前注意力层。
            loc_info: 写入位置（含 full/swa 两套 loc 的封装对象或裸 loc）。
            cache_k_nope: K 的非位置编码部分。
            cache_k_rope: K 的位置编码（RoPE）部分。
        返回：
            None。
        示例：
            >>> pool.set_mla_kv_buffer(layer, loc_info, k_nope, k_rope)
        """
        loc, swa_loc, _ = unwrap_write_loc(loc_info)
        layer_id_pool, is_swa_layer = self.layers_mapping[layer.layer_id]
        pool = self.swa_kv_pool if is_swa_layer else self.full_kv_pool
        if is_swa_layer:
            assert swa_loc is not None
            loc = swa_loc
        if not isinstance(pool, MLATokenToKVPool):
            raise TypeError(f"Layer {layer.layer_id} is not backed by an MLA KV pool")
        pool.set_mla_kv_buffer(
            None,
            loc,
            cache_k_nope,
            cache_k_rope,
            layer_id_override=layer_id_pool,
        )

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        """DSA indexer：取带 scale 的 index-K 缓冲（仅 full 层支持）。

        参数：
            layer_id: 全局 layer_id（必须是 full 层）。
        返回：
            torch.Tensor：full 子池中该层对应的 index-K（含 scale）缓冲。
        示例：
            >>> pool.get_index_k_with_scale_buffer(0)
            tensor([...])
        """
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        assert not is_swa_layer
        return self.full_kv_pool.get_index_k_with_scale_buffer(layer_id_pool)

    def get_index_k_continuous(self, layer_id: int, *args, **kwargs):
        """DSA indexer：取连续布局的 index-K 缓冲（仅 full 层支持）。

        参数：
            layer_id: 全局 layer_id（必须是 full 层）。
            *args/**kwargs: 透传给 full 子池的同名方法。
        返回：
            与 full 子池 ``get_index_k_continuous`` 一致。
        示例：
            >>> pool.get_index_k_continuous(0)
            tensor([...])
        """
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        assert not is_swa_layer
        return self.full_kv_pool.get_index_k_continuous(layer_id_pool, *args, **kwargs)

    def get_index_k_scale_continuous(self, layer_id: int, *args, **kwargs):
        """DSA indexer：取连续布局的 index-K scale 缓冲（仅 full 层支持）。

        参数：
            layer_id: 全局 layer_id（必须是 full 层）。
            *args/**kwargs: 透传给 full 子池的同名方法。
        返回：
            与 full 子池 ``get_index_k_scale_continuous`` 一致。
        示例：
            >>> pool.get_index_k_scale_continuous(0)
            tensor([...])
        """
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        assert not is_swa_layer
        return self.full_kv_pool.get_index_k_scale_continuous(
            layer_id_pool, *args, **kwargs
        )

    def get_index_k_scale_buffer(self, layer_id: int, *args, **kwargs):
        """DSA indexer：取 index-K scale 缓冲（仅 full 层支持）。

        参数：
            layer_id: 全局 layer_id（必须是 full 层）。
            *args/**kwargs: 透传给 full 子池的同名方法。
        返回：
            与 full 子池 ``get_index_k_scale_buffer`` 一致。
        示例：
            >>> pool.get_index_k_scale_buffer(0)
            tensor([...])
        """
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        assert not is_swa_layer
        return self.full_kv_pool.get_index_k_scale_buffer(
            layer_id_pool, *args, **kwargs
        )

    def set_index_k_scale_buffer(self, layer_id: int, *args, **kwargs):
        """DSA indexer：写入 index-K scale 缓冲（仅 full 层支持）。

        参数：
            layer_id: 全局 layer_id（必须是 full 层）。
            *args/**kwargs: 透传给 full 子池的同名方法。
        返回：
            与 full 子池 ``set_index_k_scale_buffer`` 一致。
        示例：
            >>> pool.set_index_k_scale_buffer(0, loc, scale)
        """
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        assert not is_swa_layer
        return self.full_kv_pool.set_index_k_scale_buffer(
            layer_id_pool, *args, **kwargs
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        """把 KV 从一个槽位搬到另一个槽位（两个子池都要搬）。

        ``tgt_loc`` / ``src_loc`` 是 full 池下标；对 SWA 子池，先用映射表把它们
        翻译为 SWA 池下标，再在 SWA 子池内执行同样的搬移。

        参数：
            tgt_loc: 目标槽位的 full 池下标。
            src_loc: 源槽位的 full 池下标。
        返回：
            None。
        示例：
            >>> pool.move_kv_cache(torch.tensor([5]), torch.tensor([9]))
            # full 池: 9 -> 5；SWA 池: mapping[9] -> mapping[5]
        """
        self.full_kv_pool.move_kv_cache(tgt_loc, src_loc)
        tgt_loc_swa = self.translate_loc_from_full_to_swa(tgt_loc)
        src_loc_swa = self.translate_loc_from_full_to_swa(src_loc)
        self.swa_kv_pool.move_kv_cache(tgt_loc_swa, src_loc_swa)

    def _filter_swa_cpu_copy(self, swa_kv_cpu, row_mask: torch.Tensor):
        """按 ``row_mask`` 过滤 SWA 的 CPU 副本，仅保留需要回灌的行。

        ``swa_kv_cpu`` 是子池产出的嵌套结构：外层按层，每层是一串 chunk，每个
        chunk 又是由若干张量（K/V，量化池还含 block scale）组成的列表。过滤时先
        把每层各 chunk 沿第 0 维拼接、按行掩码取样，再按子池的 chunk 大小重新切块。
        这样可保持 ``load_cpu_copy`` 期望的「分层 + 分块」结构。

        参数：
            swa_kv_cpu: SWA 的 CPU 副本（嵌套 list），可能为 None。
            row_mask: 布尔张量，True 表示该行需要保留。
        返回：
            过滤后的嵌套结构；当 ``swa_kv_cpu`` 为 None 或掩码全为 True 时原样返回。
        示例：
            >>> filtered = pool._filter_swa_cpu_copy(swa_kv_cpu, torch.tensor([True, False, True]))
        """
        if swa_kv_cpu is None:
            return None
        if row_mask is None or bool(torch.all(row_mask).item()):
            # 无掩码或全选，直接原样返回，省掉一次拷贝。
            return swa_kv_cpu

        # 子池的 offload 分块大小；未定义时退化为按总行数一块。
        chunk_size = getattr(
            self.swa_kv_pool, "cpu_offloading_chunk_size", len(row_mask)
        )
        filtered = []
        for layer_chunks in swa_kv_cpu:
            if len(layer_chunks) == 0:
                filtered.append([])
                continue

            # 一个 chunk 就是子池产出的内容：K/V，量化池还会带 block scale。
            # 因此需要对它携带的每个张量都做同样的过滤。
            num_tensors = len(layer_chunks[0])
            tensors = [
                torch.cat([chunk[t] for chunk in layer_chunks], dim=0)[row_mask]
                for t in range(num_tensors)
            ]

            # 按 chunk_size 重新切块，恢复 load_cpu_copy 需要的结构。
            filtered_layer = []
            for i in range(0, len(tensors[0]), chunk_size):
                filtered_layer.append([t[i : i + chunk_size] for t in tensors])
            filtered.append(filtered_layer)
        return filtered

    def get_cpu_copy(self, indices, mamba_indices=None):
        """把指定槽位的 KV 从设备拷贝到 CPU（HiCache offload 用）。

        对 SWA 混合池，需要同时拷贝 full 子池与 SWA 子池的 KV：传入的 ``indices``
        是 full 池下标，先用映射表得到对应的 SWA 下标；由于「仅尾部 SWA 分配」会
        让窗口外的 full 下标没有 SWA 映射，因此只用 ``swa_indices > 0`` 的掩码挑选
        真正有 SWA 数据的行，并保留其位置信息供 ``load_cpu_copy`` 对齐。

        参数：
            indices: 需要 offload 的 full 池槽位下标。
            mamba_indices: 兼容 Mamba 池的额外下标（本池不使用，默认 None）。
        返回：
            dict：``{"full": full_kv_cpu, "swa": swa_kv_cpu, "swa_mask": swa_mask}``。
            - ``full_kv_cpu``: full 子池的 CPU 副本；
            - ``swa_kv_cpu``: SWA 子池的 CPU 副本（无 SWA 数据时为 None）；
            - ``swa_mask``: 标记 ``indices`` 中哪些位置有 SWA 数据（CPU 布尔张量）。
        示例：
            >>> out = pool.get_cpu_copy(torch.tensor([10, 11, 12]))
            >>> out["swa_mask"]
            tensor([ True,  True, False])
        """
        # SWA 场景需要同时拷贝 full 与 SWA 两个子池的 KV：
        # indices 是 full 池下标，借助映射表得到对应的 SWA 下标。
        full_kv_cpu = self.full_kv_pool.get_cpu_copy(indices)

        swa_mask = None
        if self.full_to_swa_index_mapping is not None:
            swa_indices = self.full_to_swa_index_mapping[indices]
            # 槽位 0 被预留为 dummy 槽。仅尾部 SWA 分配时，窗口外的 full KV 下标
            # 没有 SWA 映射，因此只拷贝「有映射」的 SWA token，并保留其位置供
            # load_cpu_copy() 对齐使用。
            swa_mask = swa_indices > 0
            if torch.any(swa_mask):
                swa_kv_cpu = self.swa_kv_pool.get_cpu_copy(swa_indices[swa_mask])
                swa_mask = swa_mask.cpu()
            else:
                swa_kv_cpu = None
        else:
            swa_kv_cpu = None

        return {"full": full_kv_cpu, "swa": swa_kv_cpu, "swa_mask": swa_mask}

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        """把 ``get_cpu_copy`` 产出的 CPU KV 回灌到设备上的指定槽位。

        注意：此处的 ``indices`` 是「新分配的」full 池下标，与 ``get_cpu_copy`` 时
        的 full 下标不同。full 部分直接按 ``indices`` 回灌；SWA 部分则需要结合
        「新映射」与「旧掩码」两者：先取新 SWA 下标，再用旧掩码筛出真正有数据的行，
        最后对 SWA CPU 副本做同样的行过滤，使行序与新下标对齐后回灌。

        参数：
            kv_cache_cpu: ``get_cpu_copy`` 的返回字典（含 full/swa/swa_mask）。
            indices: 新的 full 池槽位下标。
            mamba_indices: 兼容 Mamba 池的额外下标（本池不使用，默认 None）。
        返回：
            None。
        示例：
            >>> cpu = pool.get_cpu_copy(torch.tensor([10, 11]))
            >>> pool.load_cpu_copy(cpu, torch.tensor([20, 21]))  # 回灌到新槽位
        """
        # 把 KV 从 CPU 回灌到 full 与 SWA 两个子池。
        # 注意：这里的 indices 是「新分配」的下标，与 get_cpu_copy 时的下标不同。
        full_kv_cpu = kv_cache_cpu["full"]
        swa_kv_cpu = kv_cache_cpu["swa"]

        # 把 full KV 回灌到新下标处。
        self.full_kv_pool.load_cpu_copy(full_kv_cpu, indices)

        # 若存在 SWA KV 副本，则回灌 SWA 部分。
        if swa_kv_cpu is not None and self.full_to_swa_index_mapping is not None:
            swa_indices = self.full_to_swa_index_mapping[indices]
            new_swa_mask = swa_indices > 0
            old_swa_mask = kv_cache_cpu.get("swa_mask")
            if old_swa_mask is not None:
                # 同时满足「旧掩码有数据」与「新下标有效」的行才需要回灌。
                old_swa_mask = old_swa_mask.to(indices.device)
                row_mask = new_swa_mask[old_swa_mask].cpu()
                swa_indices = swa_indices[old_swa_mask][row_mask.to(indices.device)]
            else:
                row_mask = new_swa_mask.cpu()
                swa_indices = swa_indices[new_swa_mask]

            if swa_indices.numel() == 0:
                return

            # 对 CPU 副本按行掩码过滤，使其行序与 swa_indices 对齐。
            swa_kv_cpu = self._filter_swa_cpu_copy(swa_kv_cpu, row_mask)
            self.swa_kv_pool.load_cpu_copy(swa_kv_cpu, swa_indices)
