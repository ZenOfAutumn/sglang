# srt/mem_cache/storage/mooncake_store

## 目录用途
基于 Mooncake 分布式存储的 HiCache 后端。提供 KV cache 存储后端、主机张量分配器与配置，并扩展支持多模态嵌入缓存的存取与控制。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| embedding_cache_controller.py | 嵌入缓存控制器与 `ContiguousMemoryAllocator`，异步管理嵌入在 Mooncake 中的存取。 |
| mooncake_embedding_store.py | `MooncakeEmbeddingStore`：基于 Mooncake 的多模态嵌入存储实现。 |
| mooncake_store.py | Mooncake KV 存储核心：`MooncakeStore`/`MooncakeBaseStore`、配置与主机张量分配器。 |
| test_mooncake_store.py | Mooncake 存储的单元测试，含批量 key 生成与存取校验。 |
