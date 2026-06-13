# srt/mem_cache/sparsity/core

## 目录用途
稀疏注意力的核心协调逻辑。维护每请求的稀疏状态跟踪，串联算法（TopK 检索）与后端适配器，并提供稀疏配置数据结构，在前向过程中驱动整体稀疏注意力流程。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| __init__.py | 导出 `RequestTrackers`、`SparseConfig`、`SparseCoordinator`。 |
| sparse_coordinator.py | 稀疏协调器 `SparseCoordinator`、配置 `SparseConfig` 与请求状态跟踪 `RequestTrackers`。 |
