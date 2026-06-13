# srt/disaggregation/common

## 目录用途
本目录提供各 KV 传输后端共享的通用实现，介于抽象基类（base）与具体后端之间。包含基于 ZMQ/HTTP 的 bootstrap 握手与连接管理、CommonKV 系列基类，以及异构 TP（prefill 与 decode 的 attn_tp_size 不一致）场景下的 GPU 暂存缓冲机制。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `conn.py` | 通用连接层：`CommonKVManager`/`CommonKVSender`/`CommonKVReceiver`/`CommonKVBootstrapServer`，以及 PrefillServerInfo、PrefillRankInfo 等握手元数据；实现 bootstrap 注册与连接建立。 |
| `staging_buffer.py` | 异构 TP KV 传输的 GPU 暂存缓冲：`StagingBuffer`/`StagingAllocator` 及 Triton/torch 的头切片 gather/scatter 内核，将分散的 head 切片聚成连续内存以减少 RDMA 请求数。 |
| `staging_handler.py` | 暂存散射生命周期管理：Decode/Prefill 暂存上下文与处理器、暂存策略，将 staging 逻辑从 decode.py/conn.py 中隔离。 |
| `utils.py` | 通用工具：线程安全的 `FastQueue` 与 `group_concurrent_contiguous`（合并连续 KV 索引块）。 |
| `__init__.py` | 导出 CommonKVManager、CommonKVReceiver、CommonKVBootstrapServer。 |
