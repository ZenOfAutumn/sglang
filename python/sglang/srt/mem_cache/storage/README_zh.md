# srt/mem_cache/storage

## 目录用途
HiCache 分层存储后端模块。提供统一的存储后端工厂与多种外部/远程 KV cache 存储实现（hf3fs、mooncake、nixl、lmcache、eic、aibrix），用于将 KV cache 从主机内存进一步卸载到分布式或持久化存储。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| __init__.py | 包入口，导出 `StorageBackendFactory`。 |
| backend_factory.py | 存储后端工厂 `StorageBackendFactory`，按名称动态加载并校验 `HiCacheStorage` 子类。 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| aibrix_kvcache | AIBrix KV cache 存储后端适配。 |
| eic | EIC 远程 KV 存储后端。 |
| hf3fs | 基于 HF3FS（3FS）的存储后端、客户端与元数据服务。 |
| lmcache | LMCache 集成的分层基数缓存后端。 |
| mooncake_store | Mooncake 分布式存储后端及嵌入缓存。 |
| nixl | 基于 NIXL 的 HiCache 存储后端。 |
