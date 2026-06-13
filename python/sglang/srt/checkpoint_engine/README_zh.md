# srt/checkpoint_engine

## 目录用途
本目录提供与 checkpoint-engine 的集成，用于在推理服务运行期间热更新模型权重（常见于 RL 训练-推理回路）。它通过 IPC/广播等方式将新权重注入正在运行的 SGLang 实例，无需重启服务。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 模块说明并导出 `update.main` 作为权重更新入口。 |
| `checkpoint_engine_worker.py` | 定义 `SGLangCheckpointEngineWorkerExtension` 及其实现，基于 zmq 与 checkpoint-engine 的 `update_weights_from_ipc`，在 worker 侧通过 IPC 接收并应用权重更新。 |
| `update.py` | 独立/集成的更新脚本，提供 `check_sglang_ready`、`split_checkpoint_files`、`update_weights`、`run_with_torchrun`、`main` 等，支持 broadcast 等方式经 torchrun 向服务推送 checkpoint。 |
