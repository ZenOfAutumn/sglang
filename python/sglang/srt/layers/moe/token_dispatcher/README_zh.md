# srt/layers/moe/token_dispatcher

## 目录用途
专家并行下的 token 分发（dispatch）与合并（combine）实现。定义统一的 dispatcher 基类与 DispatchOutput/CombineInput 协议格式，并为多种 all-to-all 通信后端（标准、DeepEP、Mooncake、NIXL、Mori、FlashInfer、NPU FuseEP）提供具体分发器，负责将 token 按路由结果发送到目标专家所在 rank 并回收结果。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 汇总导出各分发器、配置及 DispatchOutput/CombineInput 协议与格式枚举。 |
| `base.py` | 分发器抽象基类 `BaseDispatcher`/`BaseDispatcherConfig`，dispatch/combine 协议、格式枚举、checker 与 pre/post hooks。 |
| `deepep.py` | DeepEP 后端分发器：`DeepEPDispatcher`、`DeepEPBuffer`、normal/low-latency 两种 dispatch/combine 输入输出与配置。 |
| `flashinfer.py` | FlashInfer 后端分发器 `FlashinferDispatcher` 及其 DispatchOutput/CombineInput。 |
| `flashinfer_utils.py` | FlashInfer 通信工具：`TorchDistributedCommBackend`，在无 flashinfer 时提供占位 `CommBackend`。 |
| `fuseep.py` | NPU FuseEP 分发器 `NpuFuseEPDispatcher` 及其 dispatch/combine 数据结构。 |
| `mooncake.py` | Mooncake 后端分发器 `MooncakeEPDispatcher`、`EPBuffer` 与分发实现。 |
| `moriep.py` | Mori EP 后端分发器 `MoriEPDispatcher`，含 normal/low-latency 实现、配置与 mori op 初始化。 |
| `nixl.py` | NIXL 后端分发器 `NixlEPDispatcher` 与 `NixlEPBuffer` 实现。 |
| `standard.py` | 标准（非专用 a2a）分发器 `StandardDispatcher` 及 `StandardDispatchOutput`/`StandardCombineInput`。 |
