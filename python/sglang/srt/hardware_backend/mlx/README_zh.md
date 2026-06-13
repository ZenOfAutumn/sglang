# srt/hardware_backend/mlx

## 目录用途
本目录为 Apple Silicon 提供端到端的 MLX 推理后端，整个模型在 MLX 框架内运行，完全绕过 PyTorch MPS。通过让调度器侧的 ModelRunner 仅保留最小化的 CPU 簿记结构、将真实 KV 缓存与前向计算委托给原生 MLX 运行器，从而在不消耗 GPU 显存的前提下接入 SGLang 调度流程。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `model_runner.py` | 原生 MLX 模型运行器，在 MLX 内完成模型加载与推理，含按请求 KV 缓存状态与多请求缓存合并（`BatchKVCache`/`BatchRotatingKVCache`）逻辑。 |
| `model_runner_stub.py` | 轻量化 `ModelRunner` 子类 `MlxModelRunnerStub`，重写 `load_model`/`initialize` 跳过 PyTorch 权重加载，使用零显存的 `_DummyKVCache` 仅构建 CPU 侧簿记结构供调度器使用。 |
| `tp_worker.py` | MLX 专用张量并行工作进程 `MlxTpModelWorker`，继承 `TpModelWorker` 接入调度器，替换为 stub 运行器并将前向计算路由到原生 MLX 运行器。 |
