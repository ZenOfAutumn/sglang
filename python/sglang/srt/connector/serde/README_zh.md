# srt/connector/serde

## 目录用途
本目录为 connector 提供张量的序列化（serialize）与反序列化（deserialize）能力，定义统一接口并给出基于 safetensors 的实现。KV 类连接器（如 Redis）借助这里的序列化器在张量与字节流之间转换，以便存取远程存储。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 提供 `create_serde` 工厂函数，按类型创建 `(Serializer, Deserializer)` 对 |
| `serde.py` | 定义抽象基类 `Serializer` 与 `Deserializer`，规定张量与字节互转接口 |
| `safe_serde.py` | 基于 safetensors 的实现 `SafeSerializer`/`SafeDeserializer` |
