# 性能

本节介绍 SGLang Diffusion 的主要性能调节手段：attention 后端、缓存加速和性能分析。

## 概述

| 优化 | 类型 | 描述 |
|--------------|------|-------------|
| **Cache-DiT** | 缓存 | 带有 DBCache、TaylorSeer 和 SCM 的 block 级缓存 |
| **TeaCache** | 缓存 | 基于时间相似性的 timestep 级缓存 |
| **Attention Backends** | Kernel | 优化的 attention 实现（FlashAttention、SageAttention 等） |
| **Profiling** | 诊断 | PyTorch Profiler 和 Nsight Systems 指南 |

## 从这里开始

- 使用 [Attention Backends](attention_backends.md) 为你的模型和硬件选择最佳后端。
- 使用 [Caching Acceleration](cache/index.md) 通过 Cache-DiT 或 TeaCache 降低去噪成本。
- 当你需要诊断瓶颈而非猜测时，使用 [Profiling](profiling.md)。

## 缓存一览

- [Cache-DiT](cache/cache_dit.md) 是面向 diffusers pipeline 的 block 级缓存，更侧重于高加速的调优。
- [TeaCache](cache/teacache.md) 是内置于 SGLang 模型家族中的 timestep 级缓存。

```{toctree}
:maxdepth: 1

attention_backends
cache/index
profiling
```

## 当前基线快照

有关 Ring SP 基准测试的详情，请参见：

- [Ring SP Performance](ring_sp_performance.md)

## 参考资料

- [Cache-DiT Repository](https://github.com/vipshop/cache-dit)
- [TeaCache Paper](https://arxiv.org/abs/2411.14324)
