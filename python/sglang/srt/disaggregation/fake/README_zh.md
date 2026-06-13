# srt/disaggregation/fake

## 目录用途
本目录提供一个不执行真实 KV 传输的伪（fake）后端，主要用于 warmup 请求等不需要实际搬运 KV cache 的场景，便于在不依赖具体传输后端的情况下走通 PD 分离的请求生命周期。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `conn.py` | 伪后端实现：`FakeKVManager`/`FakeKVSender`/`FakeKVReceiver`，实现 base 抽象接口但不进行真实传输。 |
| `__init__.py` | 导出 FakeKVManager、FakeKVSender、FakeKVReceiver。 |
