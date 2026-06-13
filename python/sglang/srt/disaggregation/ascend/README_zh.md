# srt/disaggregation/ascend

## 目录用途
本目录实现昇腾（Ascend / NPU）平台的 KV 传输后端。它通过继承 Mooncake 后端的各个类、仅替换底层传输引擎，使 PD 分离的 KV cache 传输能在昇腾设备上运行。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `conn.py` | 昇腾后端连接层：`AscendKVManager`/`AscendKVSender`/`AscendKVReceiver`/`AscendKVBootstrapServer`，均继承自对应的 Mooncake 类，并在昇腾上初始化传输引擎。 |
| `transfer_engine.py` | `AscendTransferEngine`，继承 `MooncakeTransferEngine`，基于 `memfabric_hybrid` 的 TransferEngine 适配昇腾设备。 |
| `__init__.py` | 导出 AscendKVManager、AscendKVSender、AscendKVReceiver、AscendKVBootstrapServer。 |
