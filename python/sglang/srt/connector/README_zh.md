# srt/connector

## 目录用途
本目录实现 SGLang 的远程存储连接器抽象，用于从外部存储（文件系统类如 S3、KV 存储类如 Redis，以及远程推理实例）拉取模型权重或张量数据。它通过统一的 `BaseConnector` 接口屏蔽底层差异，供模型加载等流程按 URL 创建并使用对应连接器。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 定义连接器类型枚举 `ConnectorType`，提供 `create_remote_connector`、`get_connector_type` 工厂函数，并导出基类 |
| `base_connector.py` | 连接器抽象基类 `BaseConnector` 及文件型 `BaseFileConnector`、KV 型 `BaseKVConnector` 接口 |
| `redis.py` | 基于 Redis 的 KV 连接器 `RedisConnector`，配合序列化器读写张量 |
| `remote_instance.py` | 远程推理实例连接器 `RemoteInstanceConnector`，通过分布式进程组在实例间传输权重 |
| `s3.py` | S3 文件连接器 `S3Connector` 及文件列举/过滤工具函数 |
| `utils.py` | 连接器辅助工具，如从数据库拉取文件 `pull_files_from_db`、解析模型名 `parse_model_name` |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| `serde` | 张量序列化/反序列化实现，供 KV 连接器存取张量字节流 |
