# 缓存加速

SGLang 为 Diffusion Transformer (DiT) 模型提供两种互补的缓存策略。两者都通过跳过冗余计算来降低去噪成本，但它们在不同层级上运作。

## 概述

SGLang 支持两种互补的缓存方法：

| 策略 | 范围 | 机制 | 最适合 |
|----------|-------|-----------|----------|
| **Cache-DiT** | Block 级 | 动态跳过单个 transformer block | 进阶，更高加速 |
| **TeaCache** | Timestep 级 | 基于 L1 相似性跳过整个去噪步骤 | 简单，内置 |

## Cache-DiT

[Cache-DiT](https://github.com/vipshop/cache-dit) 提供 block 级缓存，带有 DBCache 和 TaylorSeer 等高级策略。它可实现高达 **1.69x 的加速**。

详细配置请参见 [cache_dit.md](cache_dit.md)。

### 快速开始

```bash
SGLANG_CACHE_DIT_ENABLED=true \
sglang generate --model-path Qwen/Qwen-Image \
    --prompt "A beautiful sunset over the mountains"
```

### 主要特性

- **DBCache**：基于残差差异的动态 block 级缓存
- **TaylorSeer**：基于泰勒展开的校准以优化缓存
- **SCM**：步级计算掩码，带来额外加速

## TeaCache

TeaCache（基于时间相似性的缓存）通过检测连续去噪步骤何时足够相似以完全跳过计算，来加速 diffusion 推理。

详细文档请参见 [teacache.md](teacache.md)。

### 快速概览

- 跟踪各 timestep 之间调制输入的 L1 距离
- 当累积距离低于阈值时，复用缓存的残差
- 支持 CFG，使用独立的正/负缓存

### 支持的模型

- Wan (wan2.1, wan2.2)
- Hunyuan (HunyuanVideo)
- Z-Image

对于 Flux 和 Qwen 模型，当启用 CFG 时，TeaCache 会被自动禁用。

```{toctree}
:maxdepth: 1

cache_dit
teacache
```

## 参考资料

- [Cache-DiT Repository](https://github.com/vipshop/cache-dit)
- [TeaCache Paper](https://arxiv.org/abs/2411.14324)
