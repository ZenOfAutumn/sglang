# srt/mem_cache/hybrid_cache

## 目录用途
为混合（全量 + Mamba）KV cache 提供分层缓存控制器，扩展 `managers.cache_controller` 中的基础 HiCache 控制器，协调设备、主机内存池与外部存储之间的预取/卸载操作。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| hybrid_cache_controller.py | 混合缓存控制器，封装 `HybridCacheController`、`PrefetchOperation` 等，处理多池（PoolEntry/PoolTransfer）分层搬运与预取。 |
