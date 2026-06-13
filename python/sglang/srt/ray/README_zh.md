# srt/ray

## 目录用途
本目录提供基于 Ray 的分布式部署支持，将 SGLang 的 Scheduler 等组件封装为 Ray actor，并以 Ray placement group 进行多节点资源编排，从而以 Ray 方式启动引擎与 HTTP 服务。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 导出 `RayEngine`。 |
| `engine.py` | `RayEngine`（`Engine` 子类），将调度器作为 Ray actor 启动，并含 `RaySchedulerInitResult` 与 placement group bundle 查找逻辑。 |
| `http_server.py` | Ray 感知的 HTTP 服务启动器 `launch_server`，复用 tokenizer manager、detokenizer、scheduler 进程的初始化入口。 |
| `scheduler_actor.py` | `SchedulerActor`，对 SGLang `Scheduler` 的 Ray actor 包装。 |
