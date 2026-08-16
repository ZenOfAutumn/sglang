# MoE Token Dispatcher 代码导读

## 目录定位

该目录实现 MoE 专家并行中的 **token 通信层**，位于 Router 和本地专家计算之间：

```text
hidden_states + TopKOutput
          │
          ▼
BaseDispatcher.dispatch
          │  按专家所在 rank 分发 token 与路由元数据
          ▼
DispatchOutput
          │
          ▼
MoeRunner：布局转换 + 本地专家计算
          │
          ▼
CombineInput
          │
          ▼
BaseDispatcher.combine
          │  将专家输出送回 token 来源 rank
          ▼
原 token 顺序的 hidden_states
```

本目录负责跨 rank 的 dispatch/combine 和通信格式，不负责 Router 打分，也不直接实现专家 grouped GEMM。二者分别位于 `moe/router.py`、`moe/topk.py` 和 `moe/moe_runner/`。

## 统一接口与数据协议

`base.py` 定义所有后端共同遵守的边界：

- `BaseDispatcher.dispatch(hidden_states, topk_output) -> DispatchOutput`；
- `BaseDispatcher.combine(combine_input) -> torch.Tensor`；
- `DispatchOutput` 只强制提供 `hidden_states` 和 `format`；
- `CombineInput` 通过 `format` 标识输入布局；
- `DispatchOutputChecker`、`CombineInputChecker` 使用 `TypeGuard` 将协议收窄为具体后端类型。

当前格式枚举为：

| 格式 | dispatch 后的主要布局 | 典型后端 |
| --- | --- | --- |
| `STANDARD` | 保留标准 `TopKOutput`，必要时完成 global-to-local expert ID 映射 | `StandardDispatcher` |
| `DEEPEP_NORMAL` | received-token 主序，并携带每个本地专家的接收 token 数 | DeepEP normal、Mori normal |
| `DEEPEP_LL` | 面向低延迟计算布局；DeepEP/Mooncake/NIXL 路径可携带 `masked_m`、`expected_m`，具体字段由后端 NamedTuple 决定 | DeepEP low-latency、Mooncake、NIXL、Mori low-latency |
| `FLASHINFER` | FlashInfer A2A workspace 布局，并可提供 runner 直接写入的输出 buffer | `FlashinferDispatcher` |

注意：格式表示 **dispatcher 与 runner 之间的数据契约**，不与具体库一一对应。Mooncake、NIXL 等后端会复用 `DEEPEP_LL` 协议，以便复用后续 runner 和布局转换逻辑。

## 主调用链

dispatcher 由 `fused_moe_triton/layer.py::create_moe_dispatcher` 根据 `--moe-a2a-backend` 创建：

```text
none / megamoe / ascend_fuseep → StandardDispatcher
                                 （ascend_fuseep 实际绕过该 dispatcher）
deepep / mooncake / mori / nixl → MaybeTboDeepEPDispatcher
                                  → 对应后端 dispatcher
flashinfer                      → FlashinferDispatcher
```

一层 MoE 的常规控制流为：

```text
FusedMoE.forward_impl
  ├─ dispatcher.dispatch
  ├─ quant_method.apply / MoeRunner.run
  └─ dispatcher.combine
```

`MoeRunner` 根据 `DispatchOutput.format` 和 runner backend，从 `PermuteMethodPool` 选择 pre-permute/post-permute；若 `FusedOpPool` 中存在对应组合，则可以直接走融合路径。因此 `--moe-a2a-backend` 和 `--moe-runner-backend` 是两条独立但需要兼容的数据路径。

## 后端实现对比

| 后端 | 模式 | 分阶段接口 | 主要特点 |
| --- | --- | --- | --- |
| Standard | 标准 | 否 | 不调用专用 A2A 库；完成本地专家 ID 映射。特定 FlashInfer FP4 路径会执行 `all_gatherv`/`reduce_scatterv`。 |
| DeepEP | normal、low-latency、auto | `dispatch_a/b`、`combine_a/b` | normal 面向较大 token 批次；low-latency 面向 decode。`auto` 根据当前 batch 是否为 extend/prefill 选择实现。 |
| Mooncake | low-latency | `dispatch_a/b`、`combine_a/b` | 复用 `DEEPEP_LL` 格式；当前 normal mode 明确未实现。 |
| NIXL | low-latency | `dispatch_a/b`、`combine_a/b` | 复用 DeepEP LL 输入输出类型，通过全局 `TCPStore` 协调并连接 ranks；当前 normal mode 未实现。 |
| Mori | normal、low-latency | `dispatch_a/b`、`combine_a/b` | 面向 Mori/AMD 通信算子，区分 intra-node、inter-node、low-latency kernel，并支持 BF16/FP8/MXFP4 等 dispatch dtype。 |
| FlashInfer | 单一路径 | 否 | 使用 `MoeAlltoAll` 和预分配 workspace；处理不等长 DP token、空 rank dummy token、padding sentinel，并可让 fused MoE 直接写 combine workspace。 |

DeepEP、Mooncake、NIXL、Mori 的 A/B 两阶段接口用于把通信拆开，使 TBO 或其他 overlap 逻辑能够在两个阶段之间插入计算。各 dispatcher 内的 `_Stage` 会检查调用顺序，完整状态循环为：

```text
INITIAL
  → AFTER_DISPATCH_A
  → AFTER_DISPATCH_B
  → AFTER_COMBINE_A
  → INITIAL
```

## 文件清单

| 文件 | 当前代码职责 |
| --- | --- |
| `__init__.py` | 对外汇总导出统一协议、checker、配置类以及 Standard、DeepEP、Mooncake、NIXL、Mori、FlashInfer 的 dispatcher 和数据类型。 |
| `base.py` | 定义 `BaseDispatcher`、`BaseDispatcherConfig`、`DispatchOutput`、`CombineInput`、格式枚举与类型 checker；提供 dispatch/combine 前后 hooks、量化配置和 overlap 参数入口。 |
| `standard.py` | 实现 `StandardDispatcher`。普通路径主要传递 hidden 与 Top-K，并将全局 expert ID 映射为本地 ID；部分 FlashInfer FP4 + DP Attention 配置下负责量化通信及 gather/scatter。 |
| `deepep.py` | DeepEP 主实现。包含 normal/low-latency 数据类型、`DeepEPConfig`、共享 `DeepEPBuffer`、两种内部实现、A/B 阶段状态机、异步事件与 overlap hook。 |
| `mooncake.py` | Mooncake EP 的 low-latency 实现。管理 `EPBuffer`，复用 `DEEPEP_LL` 格式，并提供与 DeepEP 相同的 A/B 阶段接口。 |
| `nixl.py` | NIXL EP 的 low-latency 实现。管理 `NixlEPBuffer`、TCPStore/rank 连接、A/B 阶段和 elastic EP buffer 清理。 |
| `moriep.py` | Mori EP 实现。定义 normal/low-latency 数据协议、通信模式和 dtype 配置，初始化/缓存 Mori op，并支持分阶段 dispatch/combine。 |
| `flashinfer.py` | FlashInfer `MoeAlltoAll` 实现。计算 workspace、打包 hidden/scale/Top-K payload，处理 dummy token 和 padding，并完成 combine。 |
| `flashinfer_utils.py` | 用 `torch.distributed.ProcessGroup` 适配 FlashInfer 所需的通信后端接口，提供 rank、size、allgather、broadcast、split 和 barrier。 |

目录中没有 `fuseep.py`。Ascend FuseEP 位于 `python/sglang/srt/hardware_backend/npu/moe/fuseep.py`，并绕过常规 dispatcher 控制流。

## 推荐阅读顺序

1. `base.py`：先理解协议、格式和 hooks；
2. `standard.py`：建立最简单的 dispatch/combine 基线；
3. `fused_moe_triton/layer.py::create_moe_dispatcher`：确认后端如何被选择；
4. `deepep.py`：对比 normal 与 low-latency，并跟踪 A/B 状态机；
5. `moe_runner/base.py`、`moe_runner/runner.py`：理解 dispatcher 输出如何转换为 runner 输入；
6. 根据硬件选择 `moriep.py`、`flashinfer.py`、`mooncake.py` 或 `nixl.py`；
7. 最后阅读 `batch_overlap/two_batch_overlap.py`，理解分阶段接口如何用于通信计算重叠。

阅读实现时，建议持续记录以下数据：

- `hidden_states` 的 token 维在 dispatch 前后如何变化；
- `topk_ids` 是 logical、physical、global 还是 local expert ID；
- 无效 assignment 使用 `-1` 还是 `num_experts` 作为 sentinel；
- Top-K 权重在哪一步应用，避免重复加权；
- normal 路径的 `num_recv_tokens_per_expert` 与 low-latency 路径的 `masked_m`/`expected_m` 如何驱动 grouped GEMM；
- event、hook、workspace 的生命周期是否跨越 dispatch、runner 和 combine。
