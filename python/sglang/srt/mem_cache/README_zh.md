# srt/mem_cache

## 目录用途
KV cache 与内存/前缀缓存管理的核心模块。包含两级内存池（请求到 token、token 到 KV 索引）、各类前缀缓存（RadixCache、ChunkCache、混合/SWA/Mamba 变体）、淘汰策略、分层（HiCache）存储与多模态缓存。为推理调度提供前缀复用、显存分配与分层卸载能力。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| allocator.py | token 到 KV 索引的分配器基类及 `TokenToKVPoolAllocator`/`PagedTokenToKVPoolAllocator`，含 Triton 分配核函数。 |
| base_prefix_cache.py | 前缀缓存抽象基类 `BasePrefixCache` 及统一参数/结果 dataclass（match/insert/evict/lock 等）。 |
| cache_init_params.py | `CacheInitParams` 数据类，集中传递构造前缀缓存所需的全部参数。 |
| chunk_cache.py | `ChunkCache`：RadixCache 关闭时用于分块预填充的简化缓存实现。 |
| common.py | 前缀缓存通用工具，含 `write_req_to_token_pool_triton` 等 Triton 核与 Mamba 状态常量。 |
| evict_policy.py | 淘汰策略 `EvictionStrategy`：LRU/LFU/FIFO/MRU/FILO/Priority 等优先级计算。 |
| flush_cache.py | 命令行脚本，向运行中的服务发送 `/flush_cache` 请求清空 KV 缓存。 |
| hi_mamba_radix_cache.py | `HiMambaRadixCache`：带分层存储的 Mamba 混合 KV 基数树缓存。 |
| hicache_storage.py | 分层存储抽象 `HiCacheStorage`、配置类与本地文件后端 `HiCacheFile`，含哈希工具。 |
| hiradix_cache.py | `HiRadixCache`：在 `RadixCache` 基础上叠加主机内存/外部存储分层卸载与预取。 |
| hisparse_memory_pool.py | `HiSparseNSATokenToKVPool`：稀疏注意力场景下的 NSA 设备 KV 池与主机分层映射。 |
| mamba_radix_cache.py | 管理混合（全量+Mamba）KV cache 的基数树，含 `TreeNode`、`LRUList`、`MambaRadixCache`。 |
| memory_pool.py | 内存池核心：`ReqToTokenPool`、各类 `KVCache`（MHA/MLA/NSA/Hybrid/DoubleSparse 等）物理 KV 存储。 |
| memory_pool_host.py | 主机端 KV 池 `HostKVCache` 及 MHA/MLA/Mamba/NSA 变体，负责设备与主机间分层搬运。 |
| multimodal_cache.py | 多模态嵌入缓存 `MultimodalCache`/`MultiModalStaticCache`，按哈希缓存图像等嵌入。 |
| radix_cache.py | 前缀缓存核心 `RadixCache`、`TreeNode`、`RadixKey` 等基数树实现。 |
| radix_cache_annotated_zh.py | `radix_cache.py` 的带详细中文注释学习副本，逻辑一致，仅供阅读理解。 |
| radix_cache_cpp.py | `RadixCacheCpp`：基于 C++ 基数树实现的 `BasePrefixCache` 封装。 |
| session_aware_cache.py | `SessionAwareCache`：面向会话/流式请求的缓存包装，管理 `SessionSlot`。 |
| swa_memory_pool.py | 滑动窗口注意力内存池 `SWAKVPool` 及分配器 `SWATokenToKVPoolAllocator`。 |
| swa_radix_cache.py | 管理混合（全量+SWA）KV cache 的基数树缓存。 |
| utils.py | 通用工具：MLA KV buffer 读写 Triton 核、自定义内存池初始化、bigram key 转换等。 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| cpp_radix_tree | C++ 实现的基数树及其 Python 绑定。 |
| hybrid_cache | 混合缓存的分层缓存控制器。 |
| sparsity | 稀疏注意力（Quest/DeepSeek NSA 等）算法、后端适配与协调器。 |
| storage | HiCache 分层存储后端（hf3fs、mooncake、nixl、lmcache、eic、aibrix 等）。 |
