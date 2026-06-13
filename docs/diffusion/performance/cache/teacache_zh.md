# TeaCache

> **注意**：这是 SGLang 提供的两种缓存策略之一。
> 有关所有缓存选项的概述，请参见 [caching](../index.md)。

TeaCache（基于时间相似性的缓存）通过检测连续去噪步骤何时足够相似以完全跳过计算，来加速 diffusion 推理。

## 概述

TeaCache 的工作方式：
1. 跟踪连续 timestep 之间调制输入（modulated input）的 L1 距离
2. 在各步骤上累积重缩放后的 L1 距离
3. 当累积距离低于阈值时，复用缓存的残差
4. 支持 CFG（Classifier-Free Guidance），使用独立的正/负缓存

## 工作原理

### L1 距离跟踪

在每个去噪步骤，TeaCache 计算当前与前一个调制输入之间的相对 L1 距离：

```
rel_l1 = |current - previous|.mean() / |previous|.mean()
```

然后使用多项式系数对该距离进行重缩放并累积：

```
accumulated += poly(coefficients)(rel_l1)
```

### 缓存决策

- 如果 `accumulated >= threshold`：强制计算，重置累加器
- 如果 `accumulated < threshold`：跳过计算，使用缓存的残差

### CFG 支持

对于支持 CFG 缓存分离的模型（Wan、Hunyuan、Z-Image），TeaCache 为正分支和负分支维护独立的缓存：
- 正分支的 `previous_modulated_input` / `previous_residual`
- 负分支的 `previous_modulated_input_negative` / `previous_residual_negative`

对于不支持 CFG 分离的模型（Flux、Qwen），当启用 CFG 时，TeaCache 会被自动禁用。

## 配置

TeaCache 通过采样参数中的 `TeaCacheParams` 进行配置：

```python
from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams

params = TeaCacheParams(
    teacache_thresh=0.1,           # Threshold for accumulated L1 distance
    coefficients=[1.0, 0.0, 0.0],  # Polynomial coefficients for L1 rescaling
)
```

### 参数

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `teacache_thresh` | float | 累积 L1 距离的阈值。越低 = 缓存越多，越快但质量可能越低 |
| `coefficients` | list[float] | 用于 L1 重缩放的多项式系数。需针对模型进行调优 |

### 特定模型的配置

不同的模型可能有不同的最优配置。系数通常按模型进行调优，以平衡速度和质量。

## 支持的模型

TeaCache 已内置于以下模型家族：

| 模型家族 | CFG 缓存分离 | 说明 |
|--------------|---------------------|-------|
| Wan (wan2.1, wan2.2) | 是 | 完全支持 |
| Hunyuan (HunyuanVideo) | 是 | 待支持 |
| Z-Image | 是 | 待支持 |
| Flux | 否 | 待支持 |
| Qwen | 否 | 待支持 |


## 参考资料

- [TeaCache: Accelerating Diffusion Models with Temporal Similarity](https://arxiv.org/abs/2411.14324)
