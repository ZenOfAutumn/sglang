# srt/disaggregation/mooncake

## 目录用途
本目录实现基于 Mooncake Transfer Engine（RDMA）的 KV 传输后端，是 PD 分离中常用的生产级后端。它在 common 层之上实现具体的 KV cache 搬运、握手与传输状态管理。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `conn.py` | Mooncake 后端核心：`MooncakeKVManager`/`MooncakeKVSender`/`MooncakeKVReceiver`/`MooncakeKVBootstrapServer`，以及传输块/传输信息/注册信息等数据结构与 `KVTransferError` 异常、`AuxDataCodec` 编解码。 |
| `utils.py` | Mooncake 专用工具：自定义内存池的初始化 `init_mooncake_custom_mem_pool` 与启用检测 `check_mooncake_custom_mem_pool_enabled`。 |
| `__init__.py` | 导出 MooncakeKVManager、MooncakeKVSender、MooncakeKVReceiver、MooncakeKVBootstrapServer。 |
