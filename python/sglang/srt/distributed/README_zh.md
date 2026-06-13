# srt/distributed

## 目录用途
该目录是 SGLang 的分布式运行时核心，负责接管 PyTorch 分布式环境并管理张量并行（TP）、流水线并行（PP）、专家并行（EP/MoE）等各类并行进程组。它提供统一的集合通信原语（all-reduce、all-gather、broadcast 等）以及进程组协调器，是上层模型并行执行的通信基础设施。具体的硬件/后端通信实现位于子目录 `device_communicators`。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 导出 `communication_op`、`parallel_state`、`utils` 的公共接口。 |
| `communication_op.py` | 基于各并行组协调器封装的张量级集合通信算子（all-reduce、融合 all-reduce+RMSNorm、all-gather、gather、broadcast 等）。 |
| `parallel_state.py` | 分布式状态核心：初始化/销毁分布式环境与模型并行组，定义 `GroupCoordinator` 进程组协调器及 TP/PP/EP/MoE-TP 等组的获取接口。 |
| `naive_distributed.py` | 基于文件系统 rendezvous 的简易分布式实现 `NaiveDistributed`，提供 scatter/all_gather_object 等回退能力。 |
| `utils.py` | 分布式工具：全局 `TCPStore` 管理、`StatelessProcessGroup`、张量切分等 Megatron/vLLM 改编的辅助函数。 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| `device_communicators` | 各硬件平台与后端的通信器实现（CUDA/HIP/NPU/HPU/XPU 自定义 all-reduce、PyNCCL、共享内存广播、Mooncake 传输引擎等）。 |
