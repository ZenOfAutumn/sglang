# 多模态生成的性能分析

本指南介绍 SGLang 中多模态生成 pipeline 的性能分析（profiling）技术。

## PyTorch Profiler

PyTorch Profiler 提供详细的 kernel 执行时间、调用栈和 GPU 利用率指标。

### 去噪阶段性能分析

对采样的 timestep 进行去噪阶段的性能分析（默认：1 个 warmup 步之后的 5 个步骤）：

```bash
sglang generate \
  --model-path Qwen/Qwen-Image \
  --prompt "A Logo With Bold Large Text: SGL Diffusion" \
  --seed 0 \
  --profile
```

**参数：**
- `--profile`：为去噪阶段启用性能分析
- `--num-profiled-timesteps N`：warmup 之后要分析的 timestep 数量（默认：5）
  - 较小的值会减小 trace 文件大小
  - 示例：`--num-profiled-timesteps 10` 在 1 个 warmup 步之后分析 10 个步骤

### 完整 Pipeline 性能分析

分析所有 pipeline 阶段（文本编码、去噪、VAE 解码等）：

```bash
sglang generate \
  --model-path Qwen/Qwen-Image \
  --prompt "A Logo With Bold Large Text: SGL Diffusion" \
  --seed 0 \
  --profile \
  --profile-all-stages
```

**参数：**
- `--profile-all-stages`：与 `--profile` 一起使用，分析所有 pipeline 阶段而不仅仅是去噪阶段

### 输出位置

默认情况下，trace 文件保存在 ./logs/ 目录中。

确切的输出文件路径将在控制台输出中显示，例如：

```bash
[mm-dd hh:mm:ss] Saved profiler traces to: /sgl-workspace/sglang/logs/mocked_fake_id_for_offline_generate-5_steps-global-rank0.trace.json.gz
```

### 查看 Trace

在以下位置加载并可视化 trace 文件：
- https://ui.perfetto.dev/ （推荐）
- chrome://tracing （仅限 Chrome）

对于较大的 trace 文件，请减小 `--num-profiled-timesteps` 或避免使用 `--profile-all-stages`。


### `--perf-dump-path`（阶段/步骤计时转储）

除了 profiler trace 外，你还可以转储一个轻量级的 JSON 报告，其中包含：
- 完整 pipeline 的阶段级计时分解
- 去噪阶段的步级计时分解（每个 diffusion 步）

这有助于快速识别哪个阶段主导了端到端延迟，以及去噪步骤的运行时间是否一致（如果不一致，是哪个步骤出现了异常峰值）。

转储的 JSON 包含一个 `denoise_steps_ms` 字段，格式为对象数组，每个对象带有一个 `step` 键（步骤索引）和一个 `duration_ms` 键。

示例：

```bash
sglang generate \
  --model-path <MODEL_PATH_OR_ID> \
  --prompt "<PROMPT>" \
  --perf-dump-path perf.json
```

## Nsight Systems

Nsight Systems 提供低层次的 CUDA 性能分析，包括 kernel 细节、寄存器使用情况和内存访问模式。

### 安装

安装说明请参见 [SGLang profiling guide](https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/benchmark_and_profiling.md#profile-with-nsight)。

### 基本性能分析

分析整个 pipeline 的执行：

```bash
nsys profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --force-overwrite=true \
  -o QwenImage \
  sglang generate \
    --model-path Qwen/Qwen-Image \
    --prompt "A Logo With Bold Large Text: SGL Diffusion" \
    --seed 0
```

### 针对特定阶段的性能分析

使用 `--delay` 和 `--duration` 来捕获特定阶段并减小文件大小：

```bash
nsys profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --force-overwrite=true \
  --delay 10 \
  --duration 30 \
  -o QwenImage_denoising \
  sglang generate \
    --model-path Qwen/Qwen-Image \
    --prompt "A Logo With Bold Large Text: SGL Diffusion" \
    --seed 0
```

**参数：**
- `--delay N`：在开始捕获前等待 N 秒（跳过初始化开销）
- `--duration N`：捕获 N 秒（聚焦于特定阶段）
- `--force-overwrite`：覆盖已存在的输出文件

## 说明

- **减小 trace 大小**：将 `--num-profiled-timesteps` 设置为较小的值，或在 Nsight Systems 中使用 `--delay`/`--duration`
- **特定阶段分析**：单独使用 `--profile` 分析去噪阶段，添加 `--profile-all-stages` 分析完整 pipeline
- **多次运行**：使用不同的 prompt 和分辨率进行性能分析，以识别跨工作负载的瓶颈

## 常见问题

- 如果你在使用 Nsight Systems 对 `sglang generate` 进行性能分析时发现生成的 profiler 文件没有捕获到任何 CUDA kernel，你可以通过增加模型的推理步数来延长执行时间，从而解决此问题。
