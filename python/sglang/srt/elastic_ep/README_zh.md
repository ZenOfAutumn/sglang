# srt/elastic_ep

## 目录用途
本目录实现弹性专家并行（Elastic Expert Parallelism）相关能力。它跟踪专家并行中各 rank 的活跃状态，并提供专家权重在 DRAM 中的备份与跨进程恢复机制，以便在 rank 动态增减或故障时维持 MoE 模型的可用性。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `elastic_ep.py` | 定义 `ElasticEPState`（活跃 rank 张量及与上一轮的比较）与 `ElasticEPStateManager`，管理弹性 EP 的活跃 rank 状态。 |
| `expert_backup_client.py` | `ExpertBackupClient`，基于 zmq 与世界通信组，向备份管理器请求/接收专家权重备份，并含层与专家 ID 解析工具。 |
| `expert_backup_manager.py` | `ExpertBackupManager` 及其进程入口，加载模型权重、维护专家权重在 DRAM 的备份，处理 `BackupDramReq` 等请求。 |
