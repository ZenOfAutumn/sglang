from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.unified_cache_components import ComponentType
    from sglang.srt.mem_cache.unified_cache_components.tree_component import (
        TreeComponent,
    )


@dataclasses.dataclass
class CacheInitParams:
    """各类前缀缓存（RadixCache / HiRadixCache / MambaRadixCache / UnifiedRadixCache 等）
    共用的初始化参数集合。

    把散落的构造参数聚合成一个 dataclass，好处是：调用方（如 Scheduler / kv_cache_builder）
    只需组装一次该对象，即可传给不同的缓存实现；各缓存子类再按需从中取用自己关心的字段，
    避免各构造函数签名彼此漂移、参数顺序易错的问题。
    """

    # 是否禁用缓存。为 True 时缓存退化为空实现（不命中、不插入），常用于关闭前缀复用做对照或调试。
    disable: bool
    # 请求级映射池：维护「请求 -> 其 token 在 KV 池中的槽位索引」的映射。
    req_to_token_pool: ReqToTokenPool
    # 底层 device（GPU 显存）KV 池的分配器，缓存通过它申请/归还 KV 槽位，并可取出底层 KV pool。
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    # 页大小：以「页」为粒度管理 KV（一页含 page_size 个 token），是匹配/拆分/搬运的最小单位。
    page_size: int

    # 是否为 EAGLE 投机解码场景。为 True 时前缀 key 会采用 bigram（相邻 token 成对）视图。
    is_eagle: bool = False
    # 张量并行（TP）通信组：跨 TP rank 同步缓存元数据时使用。
    tp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    # 注意力上下文并行（Attention CP）通信组。
    attn_cp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    # 注意力张量并行（Attention TP）子组。
    attn_tp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    # 流水线并行（PP）通信组。
    pp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    # 淘汰策略：如 "lru"（最近最少使用）等，决定显存/内存不足时优先淘汰哪些节点。
    eviction_policy: str = "lru"
    # 是否禁止「请求完成时把其序列插入缓存树」。为 True 时完成的请求不回填缓存，用于特定场景避免污染。
    disable_finished_insert: bool = False

    # 是否开启缓存相关指标采集（命中率、淘汰量、存储搬运量等）。
    enable_metrics: bool = False
    # 是否发布 KV cache 事件（如 BlockStored / BlockRemoved），供外部 router 构建/维护缓存索引。
    enable_kv_cache_events: bool = False

    # 是否为 Mamba/线性注意力等混合模型分配额外的状态缓冲区（存放 SSM/卷积等循环状态）。
    enable_mamba_extra_buffer: bool = False
    # 上一个额外缓冲区的「惰性分配」开关：按需延迟分配，降低启动时的峰值显存占用。
    enable_mamba_extra_buffer_lazy: bool = False

    # 当前进程在 PP 流水线中的 rank（第几级）。
    pp_rank: int = 0
    # PP 流水线的总级数。
    pp_size: int = 1

    # 当前进程在注意力上下文并行（CP）维度中的 rank。
    attn_cp_rank: int = 0
    # 注意力上下文并行（CP）的总大小。
    attn_cp_size: int = 1

    # 分块 prefill 的分块大小；None 表示不限制/不分块。影响长 prompt 的分段插入行为。
    chunked_prefill_size: Optional[int] = None

    # 滑动窗口注意力（SWA）的窗口大小；None 表示非滑窗模型。
    sliding_window_size: Optional[int] = None

    # 缓存条目的存活时间（TTL，单位：秒）。为 None 时禁用 TTL（条目不会因超时而被清理）。
    cache_ttl_seconds: Optional[float] = None

    # 统一缓存（UnifiedRadixCache）启用的组件类型元组，例如同时启用 FULL / SWA / MAMBA 等分量，
    # 用于组合出支持多种注意力/状态的统一 radix 树。
    tree_components: Optional[tuple[ComponentType, ...]] = None
    # 组件注册表覆盖项：把某些 ComponentType 映射到自定义的 TreeComponent 实现，
    # 用于替换默认组件（测试或特殊模型时使用）。
    component_registry_override: Optional[dict[ComponentType, type[TreeComponent]]] = (
        None
    )
