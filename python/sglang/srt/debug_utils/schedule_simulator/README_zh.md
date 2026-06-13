# srt/debug_utils/schedule_simulator

## 目录用途
本目录是请求调度模拟器，用于在多 GPU/多引擎场景下分析不同路由（router）与调度（scheduler）策略对负载均衡的影响。它读取真实请求日志或合成请求，模拟逐 step 调度并记录批大小、注意力计算等均衡性指标。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 包初始化，统一导出模拟器、请求、GPU 状态、路由、调度器与指标记录器等公共 API。 |
| `__main__.py` | 模块入口，解析命令行参数并运行模拟。 |
| `entrypoint.py` | 构建命令行参数解析器并驱动模拟流程（选择数据源、路由、调度器与指标）。 |
| `gpu_state.py` | `GPUState` 与 `StepRecord`：单 GPU 的待运行/运行请求队列及每 step 状态快照。 |
| `metrics.py` | 指标记录器（批大小均衡性、注意力计算均衡性、平均批大小等）。 |
| `request.py` | `SimRequest` 数据类：模拟请求的输入/输出长度、已解码 token、分组与前缀长度。 |
| `simulator.py` | `Simulator` 与 `SimulationResult`：核心模拟循环，按路由+调度逐 step 推进并产出结果。 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| `data_source/` | 请求数据源：从请求日志加载或合成随机/分组请求。 |
| `routers/` | 路由策略（随机、轮询、按组粘滞）的实现。 |
| `schedulers/` | 调度策略（如 FIFO）的实现。 |
