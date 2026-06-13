# Ring SP 基准测试：Wan2.2-TI2V-5B（u1r2 vs 基线）

本页面报告 `Wan2.2-TI2V-5B-Diffusers` 的 Ring-SP 性能，使用：

- 并行配置：`sp=2, ulysses=1, ring=2`（简称：`u1r2`）
- 基线配置：`sp=1, ulysses=1, ring=1`（简称：`u1r1`）

## 基准测试设置

- 模型：`Wan2.2-TI2V-5B-Diffusers`
- GPU：`48G RTX40 series * 2`

## 在线服务

### Ring SP (`u1r2`)

```bash
sglang serve \
  --model-type diffusion \
  --model-path /model/HuggingFace/Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --num-gpus 2 --sp-degree 2 --ulysses-degree 1 --ring-degree 2 \
  --port 8898
```

### 基线 (`u1r1`)

```bash
sglang serve \
  --model-type diffusion \
  --model-path /model/HuggingFace/Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --num-gpus 1 --sp-degree 1 --ulysses-degree 1 --ring-degree 1 \
  --port 8898
```

## 基准测试

### 基准测试免责声明

这些基准测试是在一种特定的设置和命令配置下提供的，仅供参考。实际性能可能因模型设置、运行时环境和请求模式而异。

### 阶段时间分解

| 阶段 / 指标 | `u1r2` (s) | `u1r1` 基线 (s) | 加速比 |
|---|---:|---:|---:|
| InputValidation | 0.1060 | 0.1029 | 0.97x |
| TextEncoding | 1.3965 | 2.2261 | 1.59x |
| LatentPreparation | 0.0002 | 0.0002 | 1.00x |
| TimestepPreparation | 0.0003 | 0.0004 | 1.33x |
| Denoising | 52.6358 | 71.6785 | 1.36x |
| Decoding | 7.6708 | 13.4314 | 1.75x |
| **Total** | **63.74** | **90.63** | **1.42x** |

### 内存使用

| 内存指标 | `u1r2` (GB) | `u1r1` 基线 (GB) | 差值 |
|---|---:|---:|---:|
| Peak GPU Memory | 20.07 | 27.40 | -7.33 |
| Peak Allocated | 13.35 | 20.40 | -7.05 |
| Memory Overhead | 6.72 | 7.00 | -0.28 |
| Overhead Ratio | 33.5% | 25.6% | +7.9pp |

## 总结

- 端到端延迟从 `90.63s` 提升至 `63.74s`（`1.42x`）。
- 主要收益来自 `Denoising`（`1.36x`）和 `Decoding`（`1.75x`）。
- 在 Ring-SP 上，绝对内存使用量明显下降（`Peak GPU Memory -7.33GB`、`Peak Allocated -7.05GB`）。
- 开销比例上升（`+7.9pp`），因此未来的调优可以聚焦于减少通信/运行时开销，同时保持延迟收益。
