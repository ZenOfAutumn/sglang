# srt/disaggregation/base

## 目录用途
本目录定义 PD 分离中 KV 传输后端的抽象接口与公共数据结构。所有具体后端（mooncake、nixl、mori、ascend、fake）都实现这里声明的抽象基类，从而让 prefill/decode 调度逻辑与具体传输实现解耦。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `conn.py` | 定义 `KVArgs`（传输所需的指针/长度等参数）、`KVPoll`（传输状态枚举）以及四个抽象基类 `BaseKVManager`/`BaseKVSender`/`BaseKVReceiver`/`BaseKVBootstrapServer`。 |
| `__init__.py` | 从 `conn.py` 导出 KVArgs、KVPoll 及四个基类，作为后端实现的统一引用入口。 |
