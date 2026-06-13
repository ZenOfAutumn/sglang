# Cache-DiT

SGLang 集成了 [Cache-DiT](https://github.com/vipshop/cache-dit)，这是一个面向 Diffusion Transformers (DiT) 的缓存加速引擎，可在质量损失极小的情况下实现高达 **1.69x 的推理加速**。

## 概述

**Cache-DiT** 使用智能缓存策略来跳过去噪循环中的冗余计算：

- **DBCache (Dual Block Cache)**：基于残差差异动态决定何时缓存 transformer block
- **TaylorSeer**：使用泰勒展开进行校准以优化缓存决策
- **SCM (Step Computation Masking)**：步级缓存控制，带来额外加速

## 基本用法

通过导出环境变量并使用 `sglang generate` 或 `sglang serve` 来启用 Cache-DiT：

```bash
SGLANG_CACHE_DIT_ENABLED=true \
sglang generate --model-path Qwen/Qwen-Image \
    --prompt "A beautiful sunset over the mountains"
```

## Diffusers 后端

Cache-DiT 支持从自定义 YAML 文件加载加速配置。对于 diffusers pipeline（`diffusers` 后端），通过 `--cache-dit-config` 传入 YAML/JSON 路径。此流程要求 cache-dit >= 1.2.0（`cache_dit.load_configs`）。

### 单 GPU 推理

定义一个包含以下内容的 `cache.yaml` 文件：

- DBCache + TaylorSeer

```yaml
cache_config:
  max_warmup_steps: 8
  warmup_interval: 2
  max_cached_steps: -1
  max_continuous_cached_steps: 2
  Fn_compute_blocks: 1
  Bn_compute_blocks: 0
  residual_diff_threshold: 0.12
  enable_taylorseer: true
  taylorseer_order: 1
```

然后用以下命令应用该配置：

```bash
sglang generate \
  --backend diffusers \
  --model-path Qwen/Qwen-Image \
  --cache-dit-config cache.yaml \
  --prompt "A beautiful sunset over the mountains"
```

- DBCache + TaylorSeer + SCM (Step Computation Mask)

```yaml
cache_config:
  max_warmup_steps: 8
  warmup_interval: 2
  max_cached_steps: -1
  max_continuous_cached_steps: 2
  Fn_compute_blocks: 1
  Bn_compute_blocks: 0
  residual_diff_threshold: 0.12
  enable_taylorseer: true
  taylorseer_order: 1
  # Must set the num_inference_steps for SCM. The SCM will automatically
  # generate the steps computation mask based on the num_inference_steps.
  # Reference: https://cache-dit.readthedocs.io/en/latest/user_guide/CACHE_API/#scm-steps-computation-masking
  num_inference_steps: 28
  steps_computation_mask: fast
```

- DBCache + TaylorSeer + SCM (Step Computation Mask) + Cache CFG

```yaml
cache_config:
  max_warmup_steps: 8
  warmup_interval: 2
  max_cached_steps: -1
  max_continuous_cached_steps: 2
  Fn_compute_blocks: 1
  Bn_compute_blocks: 0
  residual_diff_threshold: 0.12
  enable_taylorseer: true
  taylorseer_order: 1
  num_inference_steps: 28
  steps_computation_mask: fast
  enable_sperate_cfg: true # e.g, Qwen-Image, Wan, Chroma, Ovis-Image, etc.
```

### 分布式推理

- 1D 并行

定义一个仅包含并行配置的 yaml 文件 `parallel.yaml`，内容如下：

```yaml
parallelism_config:
  ulysses_size: auto
  attention_backend: native
```

然后，从 yaml 应用分布式推理加速配置。`ulysses_size: auto` 表示 cache-dit 会自动检测 `world_size` 作为 ulysses_size。否则，你应该手动将其设置为特定的整数，例如 4。

然后用以下命令应用该分布式配置：（注意：请添加 `--num-gpus N` 来指定用于分布式推理的 gpu 数量）

```bash
sglang generate \
  --backend diffusers \
  --num-gpus 4 \
  --model-path Qwen/Qwen-Image \
  --cache-dit-config parallel.yaml \
  --prompt "A futuristic cityscape at sunset"
```

- 2D 并行

你也可以定义一个 2D 并行配置 yaml 文件 `parallel_2d.yaml`，内容如下：

```yaml
parallelism_config:
  ulysses_size: auto
  tp_size: 2
  attention_backend: native
```
然后，从 yaml 应用 2D 并行配置。这里 `tp_size: 2` 表示使用大小为 2 的 tensor parallelism。`ulysses_size: auto` 表示 cache-dit 会自动检测 `world_size // tp_size` 作为 ulysses_size。

- 3D 并行

你也可以定义一个 3D 并行配置 yaml 文件 `parallel_3d.yaml`，内容如下：

```yaml
parallelism_config:
  ulysses_size: 2
  ring_size: 2
  tp_size: 2
  attention_backend: native
```
然后，从 yaml 应用 3D 并行配置。这里 `ulysses_size: 2`、`ring_size: 2`、`tp_size: 2` 表示使用大小为 2 的 ulysses 并行、大小为 2 的 ring 并行和大小为 2 的 tensor 并行。

- Ulysses Anything Attention

要启用 Ulysses Anything Attention，你可以定义一个并行配置 yaml 文件 `parallel_uaa.yaml`，内容如下：

```yaml
parallelism_config:
  ulysses_size: auto
  attention_backend: native
  ulysses_anything: true
```

- Ulysses FP8 通信

对于不支持 NVLink 的设备，你可以启用 Ulysses FP8 通信以进一步减少通信开销。你可以定义一个并行配置 yaml 文件 `parallel_fp8.yaml`，内容如下：

```yaml
parallelism_config:
  ulysses_size: auto
  attention_backend: native
  ulysses_float8: true
```

- 异步 Ulysses CP

你也可以启用异步 ulysses CP 以重叠通信和计算。定义一个并行配置 yaml 文件 `parallel_async.yaml`，内容如下：

```yaml
parallelism_config:
  ulysses_size: auto
  attention_backend: native
  ulysses_async: true # Now, only support for FLUX.1, Qwen-Image, Ovis-Image and Z-Image.
```
然后，从 yaml 应用该配置。这里 `ulysses_async: true` 表示启用异步 ulysses CP。

- TE-P 和 VAE-P

你也可以在 yaml 配置中指定额外的并行模块。例如，定义一个并行配置 yaml 文件 `parallel_extra.yaml`，内容如下：

```yaml
parallelism_config:
  ulysses_size: auto
  attention_backend: native
  extra_parallel_modules: ["text_encoder", "vae"]
```


### 混合缓存与并行

定义一个混合缓存与并行加速配置 yaml 文件 `hybrid.yaml`，内容如下：

```yaml
cache_config:
  max_warmup_steps: 8
  warmup_interval: 2
  max_cached_steps: -1
  max_continuous_cached_steps: 2
  Fn_compute_blocks: 1
  Bn_compute_blocks: 0
  residual_diff_threshold: 0.12
  enable_taylorseer: true
  taylorseer_order: 1
parallelism_config:
  ulysses_size: auto
  attention_backend: native
  extra_parallel_modules: ["text_encoder", "vae"]
```

然后，从 yaml 应用混合缓存与并行加速配置。

```bash
sglang generate \
  --backend diffusers \
  --num-gpus 4 \
  --model-path Qwen/Qwen-Image \
  --cache-dit-config hybrid.yaml \
  --prompt "A beautiful sunset over the mountains"
```

### Attention 后端

在某些情况下，用户可能只想指定 attention 后端而不进行任何其他优化配置。这种情况下，你可以定义一个 yaml 文件 `attention.yaml`，仅包含：

```yaml
attention_backend: "flash" # '_flash_3' for Hopper
```

### 量化

你也可以在 yaml 文件中指定量化配置，需要 `torchao>=0.16.0`。例如，定义一个 yaml 文件 `quantize.yaml`，内容如下：

```yaml
quantize_config: # quantization configuration for transformer modules
  # float8 (DQ), float8_weight_only, float8_blockwise, int8 (DQ), int8_weight_only, etc.
  quant_type: "float8"
  # layers to exclude from quantization (transformer). layers that contains any of the
  # keywords in the exclude_layers list will be excluded from quantization. This is useful
  # for some sensitive layers that are not robust to quantization, e.g., embedding layers.
  exclude_layers:
    - "embedder"
    - "embed"
  verbose: false # whether to print verbose logs during quantization
```
然后，从 yaml 应用量化配置。如果你使用量化，请同时启用 torch.compile 以获得更好的性能。例如：

```bash
sglang generate \
  --backend diffusers \
  --model-path Qwen/Qwen-Image \
  --warmup \
  --cache-dit-config quantize.yaml \
  --enable-torch-compile \
  --dit-cpu-offload false \
  --text-encoder-cpu-offload false \
  --prompt "A beautiful sunset over the mountains"
```

### 组合配置：缓存 + 并行 + 量化

你也可以将以上所有配置组合到单个 yaml 文件 `combined.yaml` 中，内容如下：

```yaml
cache_config:
  max_warmup_steps: 8
  warmup_interval: 2
  max_cached_steps: -1
  max_continuous_cached_steps: 2
  Fn_compute_blocks: 1
  Bn_compute_blocks: 0
  residual_diff_threshold: 0.12
  enable_taylorseer: true
  taylorseer_order: 1
parallelism_config:
  ulysses_size: auto
  attention_backend: native
  extra_parallel_modules: ["text_encoder", "vae"]
quantize_config:
  quant_type: "float8"
  exclude_layers:
    - "embedder"
    - "embed"
  verbose: false
```
然后，从 yaml 应用组合的缓存、并行和量化配置。如果你使用量化，请同时启用 torch.compile 以获得更好的性能。

## 高级配置

### DBCache 参数

DBCache 控制 block 级缓存行为：

| 参数 | 环境变量              | 默认值 | 描述                              |
|-----------|---------------------------|---------|------------------------------------------|
| Fn        | `SGLANG_CACHE_DIT_FN`     | 1       | 始终计算的前若干个 block 的数量 |
| Bn        | `SGLANG_CACHE_DIT_BN`     | 0       | 始终计算的后若干个 block 的数量  |
| W         | `SGLANG_CACHE_DIT_WARMUP` | 4       | 缓存开始前的 warmup 步数       |
| R         | `SGLANG_CACHE_DIT_RDT`    | 0.24    | 残差差异阈值            |
| MC        | `SGLANG_CACHE_DIT_MC`     | 3       | 最大连续缓存步数          |

### TaylorSeer 配置

TaylorSeer 使用泰勒展开提升缓存精度：

| 参数 | 环境变量                  | 默认值 | 描述                     |
|-----------|-------------------------------|---------|---------------------------------|
| Enable    | `SGLANG_CACHE_DIT_TAYLORSEER` | false   | 启用 TaylorSeer 校准器    |
| Order     | `SGLANG_CACHE_DIT_TS_ORDER`   | 1       | 泰勒展开阶数（1 或 2） |

### 组合配置示例

DBCache 和 TaylorSeer 是互补的策略，可以协同工作，你可以同时配置两组参数：

```bash
SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_FN=2 \
SGLANG_CACHE_DIT_BN=1 \
SGLANG_CACHE_DIT_WARMUP=4 \
SGLANG_CACHE_DIT_RDT=0.4 \
SGLANG_CACHE_DIT_MC=4 \
SGLANG_CACHE_DIT_TAYLORSEER=true \
SGLANG_CACHE_DIT_TS_ORDER=2 \
sglang generate --model-path black-forest-labs/FLUX.1-dev \
    --prompt "A curious raccoon in a forest"
```

### SCM (Step Computation Masking)

SCM 提供步级缓存控制，带来额外加速。它决定哪些去噪步骤完整计算，哪些使用缓存的结果。

**SCM 预设**

SCM 通过预设进行配置：

| 预设   | 计算比例 | 速度    | 质量    |
|----------|---------------|----------|------------|
| `none`   | 100%          | 基线 | 最佳       |
| `slow`   | ~75%          | ~1.3x    | 高       |
| `medium` | ~50%          | ~2x      | 良好       |
| `fast`   | ~35%          | ~3x      | 可接受 |
| `ultra`  | ~25%          | ~4x      | 较低      |

**用法**

```bash
SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_SCM_PRESET=medium \
sglang generate --model-path Qwen/Qwen-Image \
    --prompt "A futuristic cityscape at sunset"
```

**自定义 SCM Bins**

要对哪些步骤计算 vs 缓存进行细粒度控制：

```bash
SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_SCM_COMPUTE_BINS="8,3,3,2,2" \
SGLANG_CACHE_DIT_SCM_CACHE_BINS="1,2,2,2,3" \
sglang generate --model-path Qwen/Qwen-Image \
    --prompt "A futuristic cityscape at sunset"
```

**SCM 策略**

| 策略    | 环境变量                          | 描述                                 |
|-----------|---------------------------------------|---------------------------------------------|
| `dynamic` | `SGLANG_CACHE_DIT_SCM_POLICY=dynamic` | 基于内容的自适应缓存（默认） |
| `static`  | `SGLANG_CACHE_DIT_SCM_POLICY=static`  | 固定缓存模式                       |

## 环境变量

所有 Cache-DiT 参数都可以通过环境变量配置。
完整列表请参见 [Environment Variables](../../environment_variables.md)。

## 支持的模型

SGLang Diffusion x Cache-DiT 几乎支持 SGLang Diffusion 中原生支持的所有模型：

| 模型家族 | 示例模型              |
|--------------|-----------------------------|
| Wan          | Wan2.1, Wan2.2              |
| Flux         | FLUX.1-dev, FLUX.2-dev      |
| Z-Image      | Z-Image-Turbo               |
| Qwen         | Qwen-Image, Qwen-Image-Edit |
| Hunyuan      | HunyuanVideo                |

## 性能建议

1. **从默认值开始**：默认参数对大多数模型都能良好工作
2. **使用 TaylorSeer**：它通常能同时提升速度和质量
3. **调整 R 阈值**：较低的值 = 更好的质量，较高的值 = 更快
4. **用 SCM 获取额外速度**：使用 `medium` 预设以获得良好的速度/质量平衡
5. **warmup 很重要**：更高的 warmup = 更稳定的缓存决策

## 局限性

- **SGLang 原生 pipeline**：分布式支持（TP/SP）尚未验证；当 `world_size > 1` 时，Cache-DiT 将被自动禁用。
- **SCM 最小步数**：SCM 需要 >= 8 个推理步数才能有效
- **模型支持**：仅支持在 Cache-DiT 的 BlockAdapterRegister 中注册的模型

## 故障排查

### 低步数时 SCM 被禁用

对于推理步数 < 8 的模型（例如 DMD 蒸馏模型），SCM 将被自动禁用。DBCache 加速仍然有效。

## 参考资料

- [Cache-DiT](https://github.com/vipshop/cache-dit)
- [SGLang Diffusion](../index.md)
