# 专家并行(Expert Parallelism)

SGLang 中的专家并行(Expert Parallelism,EP)在 Mixture-of-Experts(MoE)模型中将专家权重分布到多个设备上,以解决内存瓶颈,并为高性能推理实现高效扩展。它对于服务大规模 MoE 模型尤为关键——在这类模型中,token 会被动态路由到分布在各 GPU 上的专用专家。通过利用优化的 all-to-all 通信和分组矩阵乘法(grouped GEMMs),EP 降低了延迟、提升了吞吐量,并最大限度减少了 GPU 空闲时间。SGLang 的 EP 通过其模块化框架提供了强大的可扩展性,能够无缝集成自定义 kernel、后端和优化,而无需重构核心逻辑,从而支持多样化的硬件和量化方案。

## 支持的后端与选择指南

SGLang 的 EP 针对不同的使用场景集成了多种高效后端,允许对性能权衡进行细粒度控制。用户通过命令行参数指定后端:
- `--moe-a2a-backend`:选择 all-to-all 通信的后端。
- `--moe-runner-backend`:选择 MoE 计算的后端。

### All-to-All 通信的后端

| Backend      | 描述                                                                 | 使用场景                          |
|--------------|-----------------------------------------------------------------------------|------------------------------------|
| **`none`(默认)** | 为 EP 禁用 all-to-all。使用 All-Reduce 或 All-Gather 进行 token dispatch。 | 混合 EP 和 TP 部署。           |
| `deepep`     | DeepEP,一个用于 MoE 模型中高效 token shuffle 的通信库。 | 大规模 EP 部署。        |
| `mooncake`   | DeepEP 的扩展,用于弹性推理,利用 RDMA 实现高性能数据传输。 | 弹性 EP 服务。 |
| `nixl`       | [NIXL-EP](https://github.com/ai-dynamo/nixl/tree/main/examples/device/ep),一个构建在 NVIDIA 的 [NIXL](https://github.com/ai-dynamo/nixl) 框架之上的弹性 EP 通信库,原生支持 RDMA 和 NVLink。 | 具有容错和动态扩展能力的弹性 EP 服务。 |
| `mori` | MORI-EP,AMD 原生的 all-to-all 通信实现,针对 ROCm 优化。 | AMD GPU 部署。 |
| `flashinfer` | Flashinfer 实现的 all-to-all。 | 大规模 EP 部署。 |
| `ascend_fuseep` | Ascend NPU 原生的融合 all-to-all 通信。 | Ascend NPU 部署。 |

DeepEP 和 Mooncake 后端支持两种 token dispatch 模式:`normal` 模式(针对高吞吐的 prefill 工作负载优化)和 `low_latency` 模式(针对低延迟的 decode 工作负载优化,并兼容 CUDA Graph)。MORI 后端目前仅支持 `normal` 模式。NIXL-EP 目前以 low-latency 模式运行,并支持 CUDA Graph。建议用户设置 `--deepep-mode auto` 以在运行时启用自动 dispatch 模式切换。设置 `--deepep-mode normal` 或 `--deepep-mode low_latency` 对于调试或开发用途很有用。

目前,DeepEP、Mooncake、NIXL-EP、`ascend_fuseep` 和 MORI 仅支持 `ep_size = tp_size` 的情况。对于混合 EP 和 TP(即 `ep_size < tp_size`),仅支持 `none` 后端(基于 All-Reduce 或 All-Gather 的 dispatch)。

### MoE 计算的后端

| Backend                  | 描述                                                                 | 使用场景                          |
|--------------------------|-----------------------------------------------------------------------------|------------------------------------|
| **`auto`(默认)**     | 根据模型架构、硬件(例如 Ampere、Hopper、Blackwell 等 NVIDIA 架构)、量化方案(例如 FP8、FP4)和运行时条件自动选择最优后端。 | 通用部署;无需用户干预即可确保兼容性和性能。 |
| `triton`                 | 基于 Triton 的分组 GEMM 实现。为获得更高性能,强烈建议创建[调优配置](https://github.com/sgl-project/sglang/blob/main/benchmark/kernels/fused_moe_triton/README.md)。 | 自定义 kernel 开发,或需要高扩展性并支持 Torch 编译的场景。 |
| `deep_gemm`              | 针对 MoE 矩阵乘法优化的 DeepGEMM 后端,支持用于 prefill 的连续布局(contiguous layout)和用于 decode 的掩码布局(masked layout);通常通过 JIT 编译以获得性能。 | 采用 FP8 块级量化(block-wise quantization)的大规模 EP 部署。 |
| `cutlass`                | 基于 CUTLASS 的高效 GEMM 后端。 | 支持 CUTLASS 的 NVIDIA 架构。 |
| `flashinfer_trtllm`      | FlashInfer 与 TensorRT-LLM 集成,用于加速 MoE 计算,支持 FP4 通信算子和高性能 GEMM。 | 搭配 TRT-LLM 的 Blackwell。 |
| `flashinfer_trtllm_routed` | FlashInfer 与 TensorRT-LLM 集成,用于加速路由 MoE 计算,消费 SGLang 计算的 top-k 专家分配和权重。 | 搭配 TRT-LLM 的 Blackwell。 |
| `flashinfer_cutlass`     | FlashInfer 与 CUTLASS 结合,用于 MoE 层中的高性能分组 GEMM,高效处理 FP4/FP8 量化。 | 使用 FP4/FP8 模型的 Blackwell。 |
| `flashinfer_mxfp4`       | 针对 MoE runner 中 MXFP4(混合 FP4)量化优化的 FlashInfer 变体,专注于内存高效的低精度推理。 | 使用 MXFP4 的低精度模型。 |
| `flashinfer_cutedsl`     | 带有自定义 DSL 的 FlashInfer,用于灵活高效地生成 MoE kernel,并与 ModelOpt FP4 量化集成。 | 使用 NVFP4 的低精度模型。 |

### 示例

为 DeepSeek-V3 使用 DeepEP 和 DeepGEMM 启动:

```bash
python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --moe-a2a-backend deepep --moe-runner-backend deep_gemm --tp 8 --ep 8
```

## 可扩展的 EP 框架

SGLang 的 EP 框架提供了模块化抽象,便于轻松集成自定义 kernel、后端和优化。它将 MoE 前向传递解耦为多个阶段(dispatch → pre-permute → core runner → post-permute → combine),从而无需重构核心逻辑即可实现无缝扩展。

### 框架概览

该框架以 `FusedMoE` 为核心,作为一个单一、可扩展结构的统一入口点。关键组件包括:
- **Dispatcher**:为 DeepEP 等后端管理 dispatch/combine(实现 `BaseDispatcher` 子类)。
- **MoeRunner**:通过 `MoeRunnerCore` 实现(例如 `TritonRunnerCore`)编排分组 GEMM 的执行。
- **PermuteMethodPool**:自动注册布局转换(例如,通过 `register_pre_permute` 和 `register_post_permute` 进行动态模式下的 pre/post-permute,或通过 `register_fused_func` 进行静态、torch.compile 兼容的融合操作)。
- **TopK Router**:与后端无关的专家选择。

该设计通过 `--moe-a2a-backend` 和 `--moe-runner-backend` 支持多种后端,并通过标准化的 `apply()` 方法集成量化。计算流确保了模块化:

```
[input_hidden_states]
          |
          v
     TopK.forward -> select_experts / triton_kernels.routing / bypass
          |
          v
     [TopKOutput]
          |
          v
   FusedMoE.forward -> Dispatcher.dispatch -> DeepEP / bypass
          |                     |
          |                     v
          |              [DispatchOutput]
          |                     |
          |                     v
          |             quant_method.apply -> MoeRunner.forward
          |                     |              |
          |                     |              v
          |                     | pre-permute + grouped_gemm + post-permute
          |                     |              |
          |                     |--------------
          |                     v
          |               [CombineInput]
          |                     |
          |                     v
          |            Dispatcher.combine -> DeepEP / bypass
          |                     |
          |---------------------
          v
[final_hidden_states]
```

更多细节请参见 [MoE 重构路线图](https://github.com/sgl-project/sglang/issues/8715)。

### 实现新的后端

要添加一个新后端:
1. 对于新的 all-to-all dispatcher,实现一个带有 `dispatch` 和 `combine` 方法的 `BaseDispatcher` 子类。
2. 对于新的 MoE runner 后端,定义一个用于核心操作(例如分组 GEMM)的 `MoeRunnerCore` 子类。
3. 为 dispatcher 或 model runner 定义新的输入/输出格式(例如 `RunnerInput`、`RunnerOutput`)。
4. 注册 permute/unpermute 方法以确保兼容性:
   - **Fused Mode**(静态,兼容 torch.compile):使用 `register_fused_func` 进行端到端操作。
   - **Permute Mode**(动态):为灵活的布局注册 `register_pre_permute` 和 `register_post_permute`。

完整的改动请参见 [MoE 重构实现 PR](https://github.com/sgl-project/sglang/pull/9269),包括类型提示和配置扩展。

### 示例

有关示例实现,请参见 [moe_runner/triton.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/moe_runner/triton.py),它演示了带有已注册融合函数和置换函数的基于 Triton 的分组 GEMM。

## 计算与通信重叠

SGLang 的 EP 采用先进的重叠技术,将通信延迟隐藏在计算之后,从而在 MoE 层中最大化 GPU 利用率。

### Two-Batch Overlap(TBO)

TBO 将请求拆分为微批次(micro-batch),将注意力计算与 dispatch/combine 操作交错执行。执行图中的让出点(yield point)允许暂停以进行重叠,在不引起峰值内存激增的情况下提升整体吞吐量:

```python
operations = [
    self._forward_attn,
    YieldOperation(),  # Overlap with dispatch of prior micro-batch
    self._forward_dispatch,
    self._forward_mlp,
    YieldOperation(),  # Overlap with combine
    self._forward_combine,
]
```

用户需要指定 `--enable-two-batch-overlap` 以解锁高达 2 倍的吞吐量。更多细节请参见 [大规模 EP 博客](https://lmsys.org/blog/2025-05-05-large-scale-ep/#two-batch-overlap)。

### Single-Batch Overlap(SBO)

SGLang 引入了一套 dispatcher-hook 系统用于 Single-Batch Overlap(SBO),它能在单个批次内重叠各种操作——例如共享专家计算与通信——同时将逻辑去中心化以增强模块化。这些 hook 在 `dispatch` 和 `combine` 操作之前和之后执行,而不修改核心 MoE 模块。该设计简化了接口、降低了耦合,并提升了可扩展性。有关实现细节以及将共享专家与 DeepEP 的 combine 操作重叠的示例,请参阅 [PR #13327](https://github.com/sgl-project/sglang/pull/13327)。用户可以设置 `--enable-single-batch-overlap` 来启用此功能。


## 负载均衡器(Workload Balancer)

SGLang 集成了 DeepSeek 的 [专家并行负载均衡器(EPLB)](https://github.com/deepseek-ai/EPLB),以解决 MoE 模型中的路由不均衡问题。通过分析专家激活统计信息,EPLB 计算出最优的专家排布,有策略地放置或复制专家,从而最小化 GPU 利用率的方差、减少空闲周期并增强可扩展性。

要启用 EPLB,使用 `--enable-eplb` 参数。为获得最佳性能,增大批次大小以稳定激活统计信息,并配置周期性再平衡(例如每 1000 个请求一次)以适应不断变化的工作负载。模拟结果表明负载均衡性(平均计算时间与最大计算时间之比)有显著改善,这与吞吐量增益强相关。

更多细节请参阅 [大规模 EP 博客中的 EPLB 章节](https://lmsys.org/blog/2025-05-05-large-scale-ep/#expert-parallelism-load-balancer) 和 [EPLB 仓库](https://github.com/deepseek-ai/eplb)。


## 结合投机解码的 EP

当在 MoE 架构上使用 MTP 的投机解码时,使用 `--speculative-moe-runner-backend` 和 `--speculative-moe-a2a-backend` 参数来为草稿模型(draft model)自定义 MoE 层行为。虽然它们默认沿用目标模型(target model)的设置,但用户可以将它们区分开,以适应目标模型和草稿模型之间不同的精度。

对于像 `nvidia/DeepSeek-R1-0528-NVFP4-v2` 这样的模型,目标模型使用 NVFP4 精度,而草稿模型使用 BF16。要为目标 MoE 层应用 `flashinfer_trtllm` kernel,同时为草稿 MoE 层回退到 triton fused MoE kernel,用户可以如下设置参数:
```
...
--moe-runner-backend flashinfer_trtllm \
--speculative-moe-runner-backend triton \
...
```


## Ascend NPU 指南


### 在 Ascend NPU 上配置 SGLang 的指南
- `--moe-a2a-backend` 仅支持 `deepep` 和 `ascend_fuseep` 后端,
  - `deepep`:其机制与上文描述一致。
  - `ascend_fuseep`:提供一个大型融合算子,将 dispatch 和 combine 之间的所有操作整合在一起以加速 MoE 计算。仅在 PD 分离模式下用于 decode 阶段。
- 无需配置 `--moe-runner-backend` 参数。
- `--deepep-mode`:
  - 在 PD 混合模式下,请设置 `--deepep-mode auto`。
  - 在 PD 分离模式下,prefill 实例设置 `--deepep-mode normal`,decode 实例设置 `--deepep-mode low_latency`。


### DeepEP Ascend 简介

DeepEP Ascend 是 DeepEP 通信库针对华为 Ascend NPU 的适配版本,专为 Mixture-of-Experts(MoE)模型的专家并行(EP)而设计。
它支持 Ant-moving 功能(将序列长度拆分为多个轮次进行流式分批传输),以优化 prefill 阶段集合通信期间占用的缓冲区大小,尤其适用于长序列。

可以通过以下环境变量在 dispatch 和 combine 两个阶段启用 Ant-moving 功能:
- `DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS`:在 dispatch 阶段启用 ant-moving 功能。表示每个 rank 上每轮传输的 token 数量,默认 8192。
- `DEEPEP_NORMAL_LONG_SEQ_ROUND`:在 dispatch 阶段启用 ant-moving 功能。表示每个 rank 上传输的轮数,默认 1。
- `DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ`:在 combine 阶段启用 ant-moving 功能,默认 0(表示禁用)。

`DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS * DEEPEP_NORMAL_LONG_SEQ_ROUND` 表示输入序列长度。当输入序列长度超过 8192 时,建议在 dispatch 和 combine 两个阶段都启用 ant-moving 功能。

环境变量 `HCCL_BUFFSIZE` 用于配置实际分配的缓冲区大小(MB)。其计算公式如下:
```angular2html
# Enable Ant-moving Function
HCCL_BUFFSIZE >= 2 * (102MB + 4MB + DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS * (hidden_size + hidden_size + hidden_size) * topk) + PADDING_BUFFSIZE

# Disable Ant-moving Function
HCCL_BUFFSIZE >= 2 * (102MB + 4MB + TOTAL_SEQ_LEN * (hidden_size + hidden_size) * topk) + PADDING_BUFFSIZE
```
其中各参数说明如下:
- `hidden_size`:模型配置中的隐藏层大小。
- `topk`:被选中的路由专家数量。
- `TOTAL_SEQ_LEN`:输入序列长度。
- `PADDING_BUFFSIZE`:建议取 20 或更大的值。
