# LLM 最核心的核函数库（Kernel Libraries）一览

> 本文梳理当今 LLM 训练 / 推理栈中**最核心的核函数库（kernel library）**，按功能分类：
> GEMM、注意力、量化、通信、DSL/编译器，以及内含大量核的完整推理栈。
> 每个条目给出定位、典型代表与一句话说明，方便快速建立全局地图。

## 目录

1. [总览表](#1-总览表)
2. [GEMM / 通用算子](#2-gemm--通用算子)
3. [注意力](#3-注意力)
4. [量化 / 低精度](#4-量化--低精度)
5. [通信 / 并行](#5-通信--并行)
6. [DSL / 编译器（写核的"语言"）](#6-dsl--编译器写核的语言)
7. [完整推理栈（内含大量核）](#7-完整推理栈内含大量核)
8. [如果只记三个](#8-如果只记三个)

---

## 1. 总览表

| 分类 | 代表库 | 一句话定位 |
| --- | --- | --- |
| GEMM / 通用算子 | cuBLAS / cuBLASLt、CUTLASS、CUB、cuDNN | 官方 GEMM 与基础原语，性能地基 |
| GEMM（专用） | DeepGEMM、Marlin | FP8 GEMM、量化 GEMM |
| 注意力 | FlashAttention、FlashInfer、FlashMLA、xFormers、PagedAttention | Attention 的高效实现 |
| 量化 / 低精度 | TransformerEngine、bitsandbytes、AWQ、GPTQ | FP8/FP4 与权重量化核 |
| 通信 / 并行 | NCCL、NVSHMEM | 多卡集合通信与 PD 传输 |
| DSL / 编译器 | Triton、CUTLASS CUTE、TileLang、Mojo | 编写/定制核的语言与工具 |
| 完整推理栈 | TensorRT-LLM / FasterTransformer、vLLM、SGLang、LMDeploy | 内含大量核的端到端引擎 |

---

## 2. GEMM / 通用算子

- **cuBLAS / cuBLASLt**：NVIDIA 官方 BLAS 库，GEMM 的事实标准；Lt 版提供更灵活的后端选择与 epilogue 融合。
- **CUTLASS**：NVIDIA 开源的 C++ 模板库，cuBLAS/各类高性能核的底层实现基础，可定制 tile 与 pipeline。
- **CUB**：CUDA 的原语库（前缀和、排序、reduce、scan 等），很多核的构件。
- **cuDNN**：深度学习的卷积/归一化等原语集合。

## 3. 注意力

- **FlashAttention（FA1/2/3）**：IO-aware tiling + online softmax，避免物化 \(N\times N\) 注意力矩阵，训练与推理通用。
- **FlashInfer**：面向推理的核函数**工具箱**，覆盖 prefill/decode、paged KV、MHA/GQA/**MLA**、各种 mask/RoPE、采样等。可理解为「可包含/调用类似 FlashAttention 技术的组件库」。
- **FlashMLA**：DeepSeek 开源的 **MLA decode** 专用核（低秩 latent + 解耦 RoPE 场景）。
- **xFormers**：Meta 的 memory-efficient attention 等算子集合。
- **PagedAttention（vLLM 自研核）**：分页 KV cache 的高效 gather/attention，服务端推理常用。

> 快速区分：**FlashAttention 是一种算法/核**，**FlashInfer 是一个推理核库**（覆盖面更广，尤其服务端变长/paged/decode 场景）。

## 4. 量化 / 低精度

- **TransformerEngine**：NVIDIA 出品，主打 FP8（及 FP4）训练/推理的算子与自动混合精度管理。
- **Marlin**：高性能量化 GEMM 核（支持 GPTQ / AWQ / FP8 等权重的快速反量化矩阵乘）。
- **bitsandbytes / AWQ / GPTQ**：常见的权重量化方案，各自带动对应的反量化/GEMM 核。

## 5. 通信 / 并行

- **NCCL**：多卡 AllReduce / AllGather / ReduceScatter 等集合通信，张量并行/流水并行的基础设施。
- **NVSHMEM**：PGAS 风格的单边通信，用于 PD 分离、低延迟 KV 传输等场景。

## 6. DSL / 编译器（写核的"语言"）

- **Triton**（OpenAI）：目前最主流的 Python 级 GPU 核 DSL，SGLang / vLLM 大量核用它编写。
- **CUTLASS CUTE**：CUTLASS 的布局/张量抽象层，用 C++ 表达复杂 tiling。
- **TileLang**：面向 tiling 的高层 DSL，降低写高性能核的门槛。
- **Mojo**：面向 AI 的系统编程语言/编译器，尝试统一 DSL 与部署。

## 7. 完整推理栈（内含大量核）

- **TensorRT-LLM / FasterTransformer**：NVIDIA 端到端推理栈，内置大量融合核与插件。
- **vLLM**：PagedAttention、各类 attention backend 与自定义核。
- **SGLang**：RadixAttention、多种 attention backend、`python/sglang/kernels` 下的自定义核。
- **LMDeploy / TurboMind**：自带高性能注意力与量化核的推理引擎。

## 8. 如果只记三个

1. **CUTLASS / cuBLAS** —— GEMM（算力利用率的地基）。
2. **FlashAttention + FlashInfer** —— 注意力（吞吐/延迟的关键）。
3. **Triton** —— 编写与定制核的 DSL。

> 说明：以上为功能分类的**概览**，各库之间存在交叉与依赖（例如 FlashInfer 内部可调用类似 FA 的技术，TensorRT-LLM / vLLM / SGLang 又大量复用上述 GEMM、注意力与量化核）。
