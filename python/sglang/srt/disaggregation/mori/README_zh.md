# srt/disaggregation/mori

## 目录用途
本目录实现基于 MORI IO 引擎（RDMA）的 KV 传输后端。它在 common 层之上封装 MORI 的引擎描述、内存描述与传输状态，提供 PD 分离的 KV cache 搬运能力。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `conn.py` | MORI 后端核心：`MoriKVManager`/`MoriKVSender`/`MoriKVReceiver`/`MoriKVBootstrapServer`，以及 `TransferInfo`、`KVArgsRegisterInfo`、`TPSliceConfig`、`AuxDataCodec` 与内存描述列表的打包/解包工具。 |
| `__init__.py` | 导出 MoriKVManager、MoriKVSender、MoriKVReceiver、MoriKVBootstrapServer。 |
