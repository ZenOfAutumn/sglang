# XPU

本文档介绍如何搭建 [SGLang](https://github.com/sgl-project/sglang) 环境并在 Intel GPU 上运行 LLM 推理，[了解更多关于 PyTorch 生态中 Intel GPU 支持的背景信息](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)。

具体而言，SGLang 针对 [Intel® Arc™ Pro B-Series Graphics](https://www.intel.com/content/www/us/en/ark/products/series/242616/intel-arc-pro-b-series-graphics.html) 和 [
Intel® Arc™ B-Series Graphics](https://www.intel.com/content/www/us/en/ark/products/series/240391/intel-arc-b-series-graphics.html) 进行了优化。

## 已优化的模型列表

一批 LLM 已在 Intel GPU 上完成优化，更多模型正在支持中：

| Model Name | BF16 |
|:---:|:---:|
| Llama-3.2-3B | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| Llama-3.1-8B | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Qwen2.5-1.5B |   [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) |

**注意：** 上表中列出的模型标识符已在 [Intel® Arc™ B580 Graphics](https://www.intel.com/content/www/us/en/products/sku/241598/intel-arc-b580-graphics/specifications.html) 上验证通过。

## 安装

### 从源码安装

目前 SGLang XPU 仅支持从源码安装。请参考 ["Getting Started on Intel GPU"](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html) 安装 XPU 依赖。

```bash
# Create and activate a conda environment
conda create -n sgl-xpu python=3.12 -y
conda activate sgl-xpu

# Set PyTorch XPU as primary pip install channel to avoid installing the larger CUDA-enabled version and prevent potential runtime issues.
pip3 install torch==2.10.0+xpu torchao torchvision torchaudio triton-xpu==3.6.0 --index-url https://download.pytorch.org/whl/xpu
pip3 install xgrammar --no-deps # xgrammar will introduce CUDA-enabled triton which might conflict with XPU

# Clone the SGLang code
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout <YOUR-DESIRED-VERSION>

# Use dedicated toml file
cd python
cp pyproject_xpu.toml pyproject.toml
# Install SGLang dependent libs, and build SGLang main package
pip install --upgrade pip setuptools
pip install -v . --extra-index-url https://download.pytorch.org/whl/xpu
```

### 使用 Docker 安装

XPU 的 Docker 镜像正在积极开发中，敬请期待。

## 启动推理服务引擎

启动 SGLang 服务的示例命令：

```bash
python -m sglang.launch_server       \
    --model <MODEL_ID_OR_PATH>       \
    --trust-remote-code              \
    --disable-overlap-schedule       \
    --device xpu                     \
    --host 0.0.0.0                   \
    --tp 2                           \   # using multi GPUs
    --attention-backend intel_xpu    \   # using intel optimized XPU attention backend
    --page-size                      \   # intel_xpu attention backend supports [32, 64, 128]
```

## 使用请求进行基准测试

你可以通过 `bench_serving` 脚本对性能进行基准测试。在另一个终端中运行该命令。

```bash
python -m sglang.bench_serving   \
    --dataset-name random        \
    --random-input-len 1024      \
    --random-output-len 1024     \
    --num-prompts 1              \
    --request-rate inf           \
    --random-range-ratio 1.0
```

各参数的详细说明可通过以下命令查看：

```bash
python -m sglang.bench_serving -h
```

此外，请求也可以使用 [OpenAI Completions API](https://docs.sglang.io/basic_usage/openai_api_completions.html) 构造，并通过命令行（例如使用 `curl`）或你自己的脚本发送。
