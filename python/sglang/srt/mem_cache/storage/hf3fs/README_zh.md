# srt/mem_cache/storage/hf3fs

## 目录用途
基于 HF3FS（3FS 高性能文件系统）的 HiCache 存储后端。提供 HF3FS 客户端抽象与 USRBIO 实现、批量页式读写、元数据管理与轻量元数据服务，将 KV cache 卸载到 3FS 存储。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| hf3fs_client.py | HF3FS 客户端抽象基类 `Hf3fsClient`，定义批量读写接口。 |
| hf3fs_usrbio_client.py | 基于 hf3fs_fuse USRBIO 的客户端实现，使用 ioring/iovec 进行高性能读写。 |
| mini_3fs_metadata_server.py | 基于 FastAPI 的轻量 3FS 元数据服务，管理各 rank 的页元数据。 |
| storage_hf3fs.py | `HiCacheHF3FS` 存储后端及元数据接口 `Hf3fsMetadataInterface`、`AtomicCounter`。 |
| test_hf3fs_utils.py | hf3fs_utils C++ 工具的读写/共享内存 pytest 测试。 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| docs | HF3FS 后端部署与 USRBIO 客户端配置文档。 |

## 说明
本目录另含 hf3fs_utils.cpp，为读写工具的 C++ 源码，由客户端在运行时即时编译。
