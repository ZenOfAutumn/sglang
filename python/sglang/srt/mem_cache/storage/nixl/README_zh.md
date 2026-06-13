# srt/mem_cache/storage/nixl

## 目录用途
基于 NIXL（NVIDIA Inference Xfer Library）的 HiCache 存储后端。通过 NIXL agent 在主机内存与 POSIX/GDS 等插件后端之间传输 KV cache，含后端选择、内存注册与文件管理工具，需安装 `nixl`。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| hicache_nixl.py | `HiCacheNixl`：实现 `HiCacheStorage` 接口的 NIXL 存储后端，封装 nixl_agent 传输。 |
| nixl_utils.py | NIXL 工具：后端配置 `NixlBackendConfig`、插件选择 `NixlBackendSelection`、注册 `NixlRegistration` 与文件管理 `NixlFileManager`。 |
| test_hicache_nixl_storage.py | NIXL 各组件的 unittest 测试套件。 |

## 说明
本目录另含 nixl.config.toml.sample，为 NIXL 后端配置的示例文件。
