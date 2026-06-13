# TPU

SGLang 通过 SGLang-JAX 后端支持高性能 TPU 推理，该后端专门针对 Google Cloud TPU 进行了优化。基于 JAX 的实现为 TPU 硬件上的大语言模型（LLM）服务工作负载带来了卓越的吞吐量和低延迟。

如遇 TPU 相关的问题或功能请求，请访问 [sglang-jax GitHub issues page](https://github.com/sgl-project/sglang-jax/issues)。

**注意：** SGLang 的 TPU 支持通过 SGLang-JAX 后端实现，这是一个专用的、基于 JAX 的推理引擎，作为独立仓库维护于 [https://github.com/sgl-project/sglang-jax](https://github.com/sgl-project/sglang-jax)。

## 系统要求

### 支持的 TPU 硬件

| TPU Type | HBM Memory | Availability |
|----------|-----------|--------------|
| TPU v6e | 32 GB | Google Cloud |
| TPU v7 | 96 GB per core | Google Cloud |

### 软件要求

- **Python：** 3.12 或更高版本
- **JAX：** 支持 TPU 的最新版本
- **环境：** Google Cloud TPU VM 或兼容的 TPU 运行时
- **可选：** SkyPilot，用于简化云端部署

## 功能支持矩阵

SGLang-JAX 为生产级 LLM 服务提供了全面的 TPU 优化功能：

| Feature | Support Status | Description |
|---------|---------------|-------------|
| High-Throughput Continuous Batching | ✅ | 动态请求批处理，最大化 TPU 利用率 |
| Radix Tree KV Cache | ✅ | 请求间高效共享前缀的内存优化方案 |
| FlashAttention Backend | ✅ | 针对长序列的 TPU 优化注意力内核 |
| Tensor Parallelism | ✅ | 将模型分布到多个 TPU 核心上 |
| Paged Attention | ✅ | 基于分页的灵活 KV 缓存管理 |
| Speculative Decoding (EAGLE/EAGLE3) | ✅ | 对兼容模型可提升 20-40% 吞吐量 |
| Chunked Prefill | ✅ | prefill 与 decode 混合批处理 |
| OpenAI-Compatible API | ✅ | 可直接替换 OpenAI API |
| Data Parallel Attention | 🚧 | 开发中 —— 采用数据并行的注意力计算 |
| Quantization | 🚧 | 开发中 —— 模型量化以降低内存占用 |
| Multi-LoRA | 🚧 | 开发中 —— 同时服务多个 LoRA 适配器 |

### 注意力后端对比

| Backend | Paged Attention | Spec Decoding | MLA | Sliding Window |
|---------|----------------|---------------|-----|----------------|
| FlashAttention (fa) | ✅ | ✅ | ❌ | ✅ |
| Native | ❌ | ❌ | ❌ | ❌ |

**注意：** 由于在内存效率和性能上更优，推荐在生产工作负载中使用 FlashAttention 后端。

## 已优化的模型列表

以下模型已经过测试并针对 TPU 部署进行了优化：

| Model Family | Performance Status |
|--------------|-------------------|
| [Qwen 3](https://huggingface.co/Qwen) | ⭐ 推荐用于生产 |
| [Qwen 3 MoE](https://huggingface.co/Qwen) | ⭐ 性能最佳 |
| [Qwen 2](https://huggingface.co/Qwen) | 有待改进 |
| [Qwen 2 MoE](https://huggingface.co/Qwen) | 有待改进 |
| [Qwen 1.5](https://huggingface.co/Qwen) | 有待改进 |
| [Llama/LLaMA](https://huggingface.co/meta-llama) | 有待改进 |
| [Grok-2](https://huggingface.co/xai-org) | 有待改进 |
| [Gemma 2](https://huggingface.co/google) | 已在 TPU 上验证 |
| Bailing MoE | 有待改进 |

## 安装

### 方法 1：使用 PyPI（推荐）

```bash
pip install sglang-jax
```

### 方法 2：从源码安装

```bash
git clone https://github.com/sgl-project/sglang-jax
cd sglang-jax
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e "python[all]"
```

### 方法 3：使用 Docker

**注意：** TPU 的 Docker 支持目前正在开发中。请使用 PyPI 或源码安装方法。

### 方法 4：使用 SkyPilot 的 Cloud TPU

[SkyPilot](https://github.com/skypilot-org/skypilot) 提供了在 Google Cloud TPU 上的简化部署：

1. 安装 SkyPilot 并配置 GCP 访问（参见 [SkyPilot documentation](https://skypilot.readthedocs.io/)）

2. 创建一个 SkyPilot 配置文件：

<details>
<summary>SkyPilot YAML: <code>sglang-jax.sky.yaml</code></summary>

```yaml
# sglang-jax.sky.yaml
resources:
   accelerators: tpu-v6e-4
   accelerator_args:
      tpu_vm: True
      runtime_version: v2-alpha-tpuv6e

run: |
  git clone https://github.com/sgl-project/sglang-jax.git
  cd sglang-jax
  uv venv --python 3.12
  source .venv/bin/activate
  uv pip install -e "python[all]"
```

</details>

3. 启动你的 TPU 集群：

```bash
# Standard deployment
sky launch -c sglang-jax sglang-jax.sky.yaml --infra=gcp

# With spot instances for cost savings
sky launch -c sglang-jax sglang-jax.sky.yaml --infra=gcp --use-spot
```

## 启动推理服务引擎

### 基础示例：Qwen-7B

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python3 -u -m sgl_jax.launch_server \
    --model-path Qwen/Qwen-7B-Chat \
    --trust-remote-code \
    --dist-init-addr=0.0.0.0:10011 \
    --nnodes=1 \
    --tp-size=4 \
    --device=tpu \
    --random-seed=3 \
    --node-rank=0 \
    --mem-fraction-static=0.8 \
    --max-prefill-tokens=8192 \
    --download-dir=/tmp \
    --dtype=bfloat16 \
    --skip-server-warmup \
    --host 0.0.0.0 \
    --port 30000
```

**关键参数说明：**

1. `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` - 启用 JIT 编译缓存，以加速后续运行时的服务器启动
2. `--tp-size=4` - 张量并行规模；应与你的 TPU 核心数量匹配（通常为 1、4 或 8）
3. `--device=tpu` - 指定 TPU 设备（这是 sglang-jax 的默认值）
4. `--dtype=bfloat16` - 使用 bfloat16 精度，TPU 针对该精度进行了优化
5. `--mem-fraction-static=0.8` - 为静态内存分配 80% 的 TPU HBM（可在 0.2 到 0.9 之间调整）
6. `--max-prefill-tokens=8192` - prefill 阶段处理的最大 token 数

### 高性能配置：Qwen3-8B

适用于追求最佳吞吐量的生产工作负载：

```bash
python3 -u -m sgl_jax.launch_server \
    --model-path Qwen/Qwen3-8B \
    --trust-remote-code \
    --tp-size=4 \
    --device=tpu \
    --mem-fraction-static=0.8 \
    --chunked-prefill-size=2048 \
    --dtype=bfloat16 \
    --max-running-requests=256 \
    --page-size=128 \
    --attention-backend=fa
```

### 进阶：推测解码（EAGLE3）

推测解码（Speculative decoding）可为兼容模型提升 20-40% 的吞吐量：

```bash
python3 -u -m sgl_jax.launch_server \
    --model-path Qwen/Qwen3-32B \
    --trust-remote-code \
    --device=tpu \
    --tp-size=4 \
    --mem-fraction-static=0.8 \
    --max-prefill-tokens=4096 \
    --attention-backend=fa \
    --dtype=bfloat16 \
    --port=30000 \
    --host=0.0.0.0 \
    --disable-overlap-schedule \
    --speculative-algorithm=EAGLE3 \
    --speculative-draft-model-path=AngelSlim/Qwen3-32B_eagle3 \
    --page-size=64 \
    --speculative-eagle-topk=1 \
    --speculative-num-steps=3 \
    --speculative-num-draft-tokens=4
```

**注意：** 推测解码目前支持 Qwen3 和 LLaMA 模型系列。详细配置指导请参见 [Speculative Decoding documentation](https://github.com/sgl-project/sglang-jax/blob/main/docs/features/speculative_decoding.md)。


### 多节点分布式服务

对于需要多个 TPU VM 的大模型：

```bash
# Node 0 (coordinator)
python3 -m sgl_jax.launch_server \
    --model-path MODEL_PATH \
    --dist-init-addr=NODE0_IP:10011 \
    --nnodes=2 \
    --node-rank=0 \
    --tp-size=8 \
    [other parameters...]

# Node 1 (worker)
python3 -m sgl_jax.launch_server \
    --model-path MODEL_PATH \
    --dist-init-addr=NODE0_IP:10011 \
    --nnodes=2 \
    --node-rank=1 \
    --tp-size=8 \
    [other parameters...]
```

## 使用请求进行基准测试

### 吞吐量测试

基础吞吐量基准测试：

```bash
python3 -m sgl_jax.bench_serving \
    --backend sgl-jax \
    --dataset-name random \
    --num-prompts=100 \
    --random-input=512 \
    --random-output=128 \
    --max-concurrency=8 \
    --random-range-ratio=1 \
    --warmup-requests=0
```

### 延迟测试

测量单批次延迟：

```bash
python3 -m sgl_jax.bench_one_batch_server \
    --base-url http://127.0.0.1:30000 \
    --model-path Qwen/Qwen-7B-Chat \
    --batch-size=32 \
    --input-len=256 \
    --output-len=32
```

### 综合基准测试脚本

用于跨不同配置进行系统性性能评估：

```bash
#!/bin/bash
set -e

backend=${1:-sgl-jax}
num_prompts_per_concurrency=3
input_seq_lens=(1024 4096 8192)
output_seq_lens=(1 1024)
max_concurrencies=(8 16 32 64 128 256)

for input_seq_len in "${input_seq_lens[@]}"; do
    for output_seq_len in "${output_seq_lens[@]}"; do
        echo "======================================="
        echo "Testing ISL/OSL: $input_seq_len/$output_seq_len"
        echo "======================================="
        for max_concurrency in "${max_concurrencies[@]}"; do
            num_prompts=$((num_prompts_per_concurrency * max_concurrency))
            python3 -m sgl_jax.bench_serving \
                --backend ${backend} \
                --dataset-name random \
                --num-prompts ${num_prompts} \
                --random-input ${input_seq_len} \
                --random-output ${output_seq_len} \
                --max-concurrency ${max_concurrency} \
                --random-range-ratio 1 \
                --disable-ignore-eos \
                --warmup-requests 0
        done
    done
done
```

查看所有基准测试参数的详细帮助：

```bash
python3 -m sgl_jax.bench_serving --help
```

有关高级基准测试技术以及使用 JAX Profiler 进行性能分析的内容，请参见 [Benchmark and Profiling Guide](https://github.com/sgl-project/sglang-jax/blob/main/docs/developer_guide/benchmark_and_profiling.md)。

## 性能优化

### 内存优化

**降低内存占用：**
- 降低 `--mem-fraction-static`（从 0.8 → 0.5 → 0.3）
- 减小 `--max-prefill-tokens`（从 16384 → 8192 → 4096）
- 减小 `--max-running-requests`

**处理 OOM 错误：**
- 从保守的内存设置开始（`--mem-fraction-static=0.5`）
- 逐步增加，直到找到最佳平衡点
- 增大 `--page-size` 以获得更好的内存局部性（1 → 16 → 64 → 128）

### 吞吐量优化

为最大化每秒 token 数：

- 使用 FlashAttention 后端：`--attention-backend=fa`
- 为 Qwen3 模型启用推测解码（EAGLE3）（提升 20-40%）
- 将 `--max-running-requests` 增加到 256+
- 将 `--mem-fraction-static` 设为 0.8+（如内存允许）
- 使用更大的 page size（64-128）
- 启用 chunked prefill：`--chunked-prefill-size=2048`

### 延迟优化

为最小化首 token 时间（TTFT）和 token 间延迟：

- 将 `--page-size` 减小到 1-4
- 降低 `--max-running-requests`（16-32）以使用更小的批次
- 减小 `--chunked-prefill-size`
- 使用保守的内存设置以避免 GC 暂停

### TPU 专属优化

1. **JIT 编译缓存：**
   ```bash
   export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache
   ```
   始终设置此环境变量，以缓存已编译的内核并加速服务器启动。

2. **数据类型优化：**
   使用 `--dtype=bfloat16` 进行 TPU 原生优化。TPU 专为 bfloat16 计算而设计。

3. **张量并行：**
   将 `--tp-size` 与你的 TPU 核心配置（1、4 或 8）匹配，以实现最佳的模型分布。

4. **注意力后端：**
   生产工作负载始终使用 `--attention-backend=fa`（FlashAttention）。

## 故障排查

### OOM（内存溢出）错误

如果你遇到内存溢出错误：

1. 将 `--mem-fraction-static` 从 0.8 降至 0.5 或更低
2. 将 `--max-prefill-tokens` 从 8192 减小至 4096 或 2048
3. 降低 `--max-running-requests` 以减小并发批次大小
4. 增大 `--page-size` 以提升内存布局效率

### 编译耗时过长

如果服务器启动耗时过长：

1. 确保已正确设置 `JAX_COMPILATION_CACHE_DIR`
2. 理解首次运行需要 JIT 编译（这是正常的）
3. 借助缓存的编译结果，后续运行将显著加快
4. 可考虑使用 `--skip-server-warmup` 将编译推迟到首次请求时进行

### 吞吐量偏低

如果你未达到预期吞吐量：

1. 确认 `--tp-size` 与你的 TPU 核心配置匹配
2. 检查是否启用了 `--attention-backend=fa`
3. 增大 `--max-running-requests` 以形成更大的批次
4. 考虑为兼容模型启用推测解码
5. 确保内存设置允许形成足够大的批次

### 连接问题

如果客户端无法连接到服务器：

1. 为实现外部访问，确保使用 `--host=0.0.0.0`（而非仅 `127.0.0.1`）
2. 确认防火墙规则允许指定端口（默认：30000）的流量
3. 检查服务器进程是否正在运行：`curl http://localhost:30000/health`

## 高级功能

### 推测解码

SGLang-JAX 为 Qwen3 和 LLaMA 模型系列支持 EAGLE 和 EAGLE3 推测解码算法。推测解码可在不影响输出质量的前提下提升 20-40% 的吞吐量。

详细配置和支持的模型组合请参见 [Speculative Decoding documentation](https://github.com/sgl-project/sglang-jax/blob/main/docs/features/speculative_decoding.md)。

### Chunked Prefill

启用 prefill 与 decode 混合批处理，以获得更好的 TPU 利用率：

```bash
--chunked-prefill-size=2048 --enable-mixed-chunk
```

这允许调度器在同一批次中将 prefill 操作与 decode 操作混合，从而提升整体吞吐量。

### 自定义注意力后端

SGLang-JAX 支持基于插件的注意力后端系统。你可以实现针对特定用例优化的自定义注意力内核。

实现细节请参见 [Attention Backend documentation](https://github.com/sgl-project/sglang-jax/blob/main/docs/features/attention_backend.md)。

### 环境验证

在部署前验证你的 TPU 设置：

```bash
python -c "from sgl_jax import check_env; check_env.check_env()"
```

该命令检查：
- 已安装的软件包版本
- TPU 设备的可用性和规格
- 系统资源和配置
- 设置的兼容性

## 贡献

我们欢迎为改进 SGLang-JAX 的 TPU 支持做出贡献！

### 可贡献的方向

**查看 [Development Roadmap](https://github.com/sgl-project/sglang-jax/issues/190)**，了解计划中的功能并寻找贡献新功能的机会。

当前的贡献方向包括：

- 针对特定 TPU 代次的性能优化
- 对更多模型架构的支持
- 文档改进和示例
- Bug 报告和修复
- 基准测试结果和性能分析

### 如何贡献

1. 访问 [sglang-jax repository](https://github.com/sgl-project/sglang-jax)
2. 阅读 [Contribution Guide](https://github.com/sgl-project/sglang-jax/blob/main/docs/developer_guide/contribution_guide.md)
3. 加入 [SGL-JAX Slack community](https://sgl-fru7574.slack.com/archives/C09EBE5HT5X) 参与讨论
4. 在 [sglang-jax/issues](https://github.com/sgl-project/sglang-jax/issues) 报告问题

### 在 TPU 上测试

对于需要 TPU 访问权限进行测试的贡献者：

- 参考 [TPU Resources Guide](https://github.com/sgl-project/sglang-jax/blob/main/docs/developer_guide/tpu_resources_guide.md) 了解获取 TPU 硬件的信息
- 使用 SkyPilot 的 spot 实例进行经济高效的测试
- 遵循 [Benchmark and Profiling Guide](https://github.com/sgl-project/sglang-jax/blob/main/docs/developer_guide/benchmark_and_profiling.md) 进行性能验证

## 参考资料

### 文档

- [SGLang-JAX Repository](https://github.com/sgl-project/sglang-jax)
- [SGLang-JAX Installation Guide](https://github.com/sgl-project/sglang-jax/blob/main/docs/get_started/install.md)
- [Qwen Models Quick Start](https://github.com/sgl-project/sglang-jax/blob/main/docs/basic_usage/qwen.md)
- [Benchmark and Profiling Guide](https://github.com/sgl-project/sglang-jax/blob/main/docs/developer_guide/benchmark_and_profiling.md)
- [Speculative Decoding](https://github.com/sgl-project/sglang-jax/blob/main/docs/features/speculative_decoding.md)

### 外部资源

- [JAX Documentation](https://jax.readthedocs.io/)
- [Google Cloud TPU Documentation](https://cloud.google.com/tpu/docs)
- [SkyPilot Documentation](https://skypilot.readthedocs.io/)
