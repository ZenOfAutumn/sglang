# AMD GPU

本文档介绍如何在 AMD GPU 上运行 SGLang。如果你遇到问题或有疑问，请 [open an issue](https://github.com/sgl-project/sglang/issues)。

## 系统配置

在使用 AMD GPU（例如 MI300X）时，某些系统级优化有助于确保性能稳定。这里以 MI300X 为例。AMD 提供了关于 MI300X 优化和系统调优的官方文档：

- [AMD MI300X Tuning Guides](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/index.html)
- [LLM inference performance validation on AMD Instinct MI300X](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/vllm-benchmark.html)
- [AMD Instinct MI300X System Optimization](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html)
- [AMD Instinct MI300X Workload Optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html)
- [Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html)

**注意：** 我们强烈建议完整阅读这些文档和指南，以充分发挥你的系统性能。

以下是为 SGLang 需要确认或启用的几项关键设置：

### 更新 GRUB 设置

在 `/etc/default/grub` 中，将以下内容追加到 `GRUB_CMDLINE_LINUX`：

```text
pci=realloc=off iommu=pt
```

之后运行 `sudo update-grub`（或你所用发行版的等效命令）并重启。

### 禁用 NUMA 自动平衡

```bash
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
```

你可以使用[这个实用脚本](https://github.com/ROCm/triton/blob/rocm_env/scripts/amd/env_check.sh)来自动执行或验证此更改。

再次提醒，请通读全部文档，以确认你的系统采用了推荐配置。

## 安装 SGLang

你可以使用下列方法之一安装 SGLang。

### 从源码安装

```bash
# Use the last release branch
git clone -b v0.5.9 https://github.com/sgl-project/sglang.git
cd sglang

# Compile sgl-kernel
pip install --upgrade pip
cd sgl-kernel
python setup_rocm.py install

# Install sglang python package along with diffusion support
cd ..
rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_hip]"
```

### 使用 Docker 安装（推荐）

Docker 镜像可在 Docker Hub 的 [lmsysorg/sglang](https://hub.docker.com/r/lmsysorg/sglang/tags) 获取，它们由 [rocm.Dockerfile](https://github.com/sgl-project/sglang/tree/main/docker) 构建。

以下步骤展示如何构建和使用镜像。

1. 构建 Docker 镜像。
   如果你使用预构建镜像，可以跳过此步骤，并在后续步骤中将 `sglang_image` 替换为预构建镜像名称。

   ```bash
   docker build -t sglang_image -f rocm.Dockerfile .
   ```

2. 创建一个方便使用的别名。

   ```bash
   alias drun='docker run -it --rm --network=host --privileged --device=/dev/kfd --device=/dev/dri \
       --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE \
       --security-opt seccomp=unconfined \
       -v $HOME/dockerx:/dockerx \
       -v /data:/data'
   ```

   如果你使用 RDMA，请注意：
     - RDMA 需要 `--network host` 和 `--privileged`。如果你不需要 RDMA，可以移除它们。
     - 如果你使用 RoCE，可能需要设置 `NCCL_IB_GID_INDEX`，例如：`export NCCL_IB_GID_INDEX=3`。

3. 启动服务器。

   **注意：** 将下方的 `<secret>` 替换为你的 [huggingface hub token](https://huggingface.co/docs/hub/en/security-tokens)。

   ```bash
   drun -p 30000:30000 \
       -v ~/.cache/huggingface:/root/.cache/huggingface \
       --env "HF_TOKEN=<secret>" \
       sglang_image \
       python3 -m sglang.launch_server \
       --model-path NousResearch/Meta-Llama-3.1-8B \
       --host 0.0.0.0 \
       --port 30000
   ```

4. 为验证可用性，你可以在另一个终端运行基准测试，或参考[其他文档](https://docs.sglang.io/basic_usage/openai_api_completions.html)向引擎发送请求。

   ```bash
   drun sglang_image \
       python3 -m sglang.bench_serving \
       --backend sglang \
       --dataset-name random \
       --num-prompts 4000 \
       --random-input 128 \
       --random-output 128
   ```

在正确配置好 AMD 系统并安装 SGLang 后，你现在就可以充分利用 AMD 硬件来驱动 SGLang 的机器学习能力了。

## AMD GPU 上的量化

[Quantization documentation](../advanced_features/quantization.md#platform-compatibility) 提供了完整的兼容性矩阵。简而言之：FP8、AWQ、MXFP4、W8A8、GPTQ、compressed-tensors、Quark 以及 **petit_nvfp4**（通过 [Petit](https://github.com/causalflow-ai/petit-kernel) 在 ROCm 上支持 NVFP4）都可以在 AMD 上工作。而依赖 Marlin 或 NVIDIA 专有内核的方法（`awq_marlin`、`gptq_marlin`、`gguf`、`modelopt_fp8`、`modelopt_fp4`）则不可用。

有几点需要注意：

- FP8 通过 Aiter 或 Triton 工作。像 DeepSeek-V3/R1 这样的预量化 FP8 模型开箱即用。
- AWQ 在 AMD 上使用 Triton 反量化（dequantization）内核。更快的 Marlin 路径不可用。
- MXFP4 需要 CDNA3/CDNA4 以及 `SGLANG_USE_AITER=1`。
- `petit_nvfp4` 通过 [Petit](https://github.com/causalflow-ai/petit-kernel) 在 MI250/MI300X 上启用 NVFP4 模型（例如 [Llama 3.3 70B FP4](https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-FP4)）。使用 `pip install petit-kernel` 安装；加载预量化的 NVFP4 模型时无需 `--quantization` 标志。
- `quark_int4fp8_moe` 是一种仅限 AMD 的在线量化方法，用于 CDNA3/CDNA4 上的 MoE 模型。

其中若干后端由 [Aiter](https://github.com/ROCm/aiter) 加速。通过以下方式启用：

```bash
export SGLANG_USE_AITER=1
```

示例 —— 服务一个 AWQ 模型：

```bash
python3 -m sglang.launch_server \
    --model-path hugging-quants/Mixtral-8x7B-Instruct-v0.1-AWQ-INT4 \
    --trust-remote-code \
    --port 30000 --host 0.0.0.0
```

示例 —— FP8 在线量化：

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --quantization fp8 \
    --port 30000 --host 0.0.0.0
```

## 示例

### 运行 DeepSeek-V3

运行 DeepSeek-V3 的唯一区别在于启动服务器的方式。下面是一个示例命令：

```bash
drun -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    --env "HF_TOKEN=<secret>" \
    sglang_image \
    python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \ # <- here
    --tp 8 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000
```

[Running DeepSeek-R1 on a single NDv5 MI300X VM](https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/running-deepseek-r1-on-a-single-ndv5-mi300x-vm/4372726) 也是一个不错的参考。

### 运行 Llama3.1

运行 Llama3.1 与运行 DeepSeek-V3 几乎相同。唯一的区别在于启动服务器时指定的模型，如下面的示例命令所示：

```bash
drun -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    --env "HF_TOKEN=<secret>" \
    sglang_image \
    python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \ # <- here
    --tp 8 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000
```

### 预热步骤（Warmup Step）

当服务器显示 `The server is fired up and ready to roll!` 时，表示启动成功。
