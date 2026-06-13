# 昇腾 NPU Ring-SP 性能（Wan2.1-T2V-1.3B）

本页报告在昇腾（Ascend）NPU 上使用 `torch_npu==2.10.0` 时的 Ring-SP 性能。

- 基线配置：`ulysses=1, ring=1`（简写：`u1r1`）
- Ring-SP 配置：`ulysses=1, ring=2`（简写：`u1r2`）

## 基准测试设置

- 模型：`Wan2.1-T2V-1.3B-Diffusers`
- 提示词（Prompt）：`"a cat is playing piano"`
- 框架命令：`sglang generate`
- 运行时：`torch_npu==2.10.0`

## 生成命令

### 基线（`u1r1`）

```bash
sglang generate --model-path /nas/disk1/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "a cat is playing piano" --num-gpus 1 --ring-degree 1 \
    --save-output
```

### Ring-SP（`u1r2`）

```bash
sglang generate --model-path /nas/disk1/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "a cat is playing piano" --num-gpus 2 --ring-degree 2 \
    --save-output
```

## 基准测试结果

基准测试免责声明

这些数据来自单一固定设置和单一提示词用例。实际性能可能因模型设置、环境和工作负载而有所不同。

### 各阶段耗时拆解

| 阶段 / 指标 | `u1r2` (s) | `u1r1` 基线 (s) | 加速比 |
|---|---:|---:|---:|
| InputValidation | 0.0003 | 0.0002 | 0.67x |
| TextEncoding | 3.5936 | 3.5820 | 1.00x |
| LatentPreparation | 0.0007 | 0.0055 | 7.86x |
| TimestepPreparation | 0.0008 | 0.0007 | 0.88x |
| Denoising | 121.2788 | 239.2580 | 1.97x |
| Decoding | 13.8685 | 16.4969 | 1.19x |
| **Total（生成的像素数据）** | **141.86** | **266.50** | **1.88x** |

## 总结

- 在 `torch_npu==2.10.0` 下，Ring-SP（`u1r2`）在本用例中成功运行于 NPU。
- 端到端生成时间从 `266.50s` 提升至 `141.86s`（`1.88x`）。
- 主要收益来自 `DenoisingStage`（`1.97x`），解码（decoding）也有所提升（`1.19x`）。
