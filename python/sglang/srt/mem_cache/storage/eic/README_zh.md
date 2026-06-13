# srt/mem_cache/storage/eic

## 目录用途
基于 EIC 的远程 KV cache 存储后端。通过 `eic` 客户端及 YAML 配置（可选 GPU Direct RDMA）将 KV cache 卸载到 EIC 远程存储，实现 HiCache 分层缓存。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| eic_storage.py | `EICStorage`：实现 `HiCacheStorage` 接口的 EIC 远程存储后端，支持 RDMA 直传与张量池。 |
| test_unit.py | EIC 存储单元测试脚本，按配置初始化 EIC 客户端并验证存取。 |
