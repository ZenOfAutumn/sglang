# srt/layers/moe

## 目录用途
混合专家（MoE）层的总入口与核心实现，涵盖专家路由（topk/router）、多种后端的 fused experts kernel（CUTLASS、FlashInfer TRTLLM、CuteDSL、ROCm AITER、Torch 原生），以及 MoE 全局配置与后端选择工具。子目录进一步细分专家并行、Triton fused MoE、运行器抽象与 token 分发实现。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 导出 `MoeRunner`、`MoeRunnerConfig`、各类后端枚举（`MoeA2ABackend`/`MoeRunnerBackend`/`DeepEPMode`）及 MoE 配置初始化与查询函数。 |
| `cutlass_moe.py` | 基于 CUTLASS 的 fused MoE kernel，提供 FP8 分块缩放与 NVFP4 分组 GEMM 专家计算入口。 |
| `cutlass_moe_params.py` | 定义 `CutlassMoEType` 枚举与 `CutlassMoEParams` 数据类，封装 CUTLASS MoE 的激活/权重/输出步长等参数。 |
| `cutlass_w4a8_moe.py` | CUTLASS W4A8（4bit 权重、8bit 激活）MoE kernel，含普通与 DeepEP low-latency 两种专家并行路径。 |
| `flashinfer_cutedsl_moe.py` | 基于 FlashInfer CuteDSL 的 masked MoE 计算，支持 NVFP4 分块缩放分组 GEMM。 |
| `flashinfer_trtllm_moe.py` | 注册 FlashInfer TRTLLM 的 FP8 block-scale、routed、per-tensor scale MoE 自定义算子封装。 |
| `fused_moe_native.py` | Torch 原生 FusedMoE 前向实现，主要用于 torch.compile 路径。 |
| `kt_ep_wrapper.py` | KTransformers CPU-GPU 异构专家并行包装器，协调 GPU 专家与 CPU（AMX/AVX）专家并行执行。 |
| `rocm_moe_utils.py` | ROCm/AITER 平台的 MoE 工具，含 asm_moe_tkw1 封装及 fp4/mxfp4 上采样 Triton kernel。 |
| `routed_experts_capturer.py` | 捕获并缓存每个 token 路由到的专家 ID，提供设备/主机缓存与全局捕获器接口。 |
| `router.py` | MoE 路由的 Triton kernel（cudacore/tensorcore 版本）与 `FusedMoeRouter` 封装，计算 topk 权重与专家 ID。 |
| `topk.py` | 专家 topk 选择核心，定义 `TopK` 模块与多种 topk/grouped-topk 实现（GPU/CPU、biased、Kimi-K2 等）。 |
| `utils.py` | MoE 全局配置与后端枚举工具：A2A 后端、Runner 后端、DeepEP 模式、TBO 阈值、路由方法类型等。 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| `ep_moe` | 专家并行（Expert Parallelism）MoE 层与相关 Triton kernel。 |
| `fused_moe_triton` | Triton 实现的 fused MoE kernel、配置与 `FusedMoE` 层，含预调优 JSON 配置。 |
| `moe_runner` | MoE 运行器抽象与各后端核心（Triton、DeepGEMM、Marlin、FlashInfer TRTLLM 等）。 |
| `token_dispatcher` | 专家并行下的 token 分发/合并实现（DeepEP、Mooncake、NIXL、Mori、FlashInfer 等）。 |
