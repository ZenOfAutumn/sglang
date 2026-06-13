# srt/mem_cache/storage/aibrix_kvcache

## 目录用途
将 AIBrix KVCache 接入 SGLang HiCache 的存储后端实现。基于 `aibrix_kvcache` 库的 `BaseKVCacheManager` 等接口，把 KV cache 的读写映射到 AIBrix 管理的存储上。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| aibrix_kvcache_storage.py | `AibrixKVCacheStorage`：实现 `HiCacheStorage` 接口的 AIBrix KV cache 后端。 |
| unit_test.py | `AIBrixKVCacheStorageTest` 单元测试，验证不同 page size 下的存取行为。 |
