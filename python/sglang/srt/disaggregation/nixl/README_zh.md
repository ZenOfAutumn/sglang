# srt/disaggregation/nixl

## 目录用途
本目录实现基于 NIXL 的 KV 传输后端。它在 common 层之上提供 NIXL 的连接管理、KV cache 传输与传输状态跟踪，是 PD 分离可选的 KV 传输后端之一。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `conn.py` | NIXL 后端核心：`NixlKVManager`/`NixlKVSender`/`NixlKVReceiver`/`NixlKVBootstrapServer`，以及 `TransferInfo`、`KVArgsRegisterInfo`、`TransferStatus` 等数据结构。 |
| `__init__.py` | 导出 NixlKVManager、NixlKVSender、NixlKVReceiver、NixlKVBootstrapServer。 |
