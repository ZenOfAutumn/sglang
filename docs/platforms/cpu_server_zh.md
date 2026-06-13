# CPU 服务器

本文档介绍如何搭建 [SGLang](https://github.com/sgl-project/sglang) 环境并在 CPU 服务器上运行 LLM 推理。
SGLang 已在配备 Intel® AMX® 指令集的 CPU 上启用并优化，
这些 CPU 为第 4 代或更新的 Intel® Xeon® 可扩展处理器。

## 已优化的模型列表

一批热门的 LLM 已经过优化并能在 CPU 上高效运行，
包括最知名的开源模型，如 Llama 系列、Qwen 系列，
以及 DeepSeek 系列（如 DeepSeek-R1 和 DeepSeek-V3.1-Terminus）。

| Model Name | BF16 | W8A8_INT8 | FP8 |
|:---:|:---:|:---:|:---:|
| DeepSeek-R1 |   | [meituan/DeepSeek-R1-Channel-INT8](https://huggingface.co/meituan/DeepSeek-R1-Channel-INT8) | [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) |
| DeepSeek-V3.1-Terminus |   | [IntervitensInc/DeepSeek-V3.1-Terminus-Channel-int8](https://huggingface.co/IntervitensInc/DeepSeek-V3.1-Terminus-Channel-int8) | [deepseek-ai/DeepSeek-V3.1-Terminus](https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Terminus) |
| Llama-3.2-3B | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | [RedHatAI/Llama-3.2-3B-quantized.w8a8](https://huggingface.co/RedHatAI/Llama-3.2-3B-Instruct-quantized.w8a8) |   |
| Llama-3.1-8B | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | [RedHatAI/Meta-Llama-3.1-8B-quantized.w8a8](https://huggingface.co/RedHatAI/Meta-Llama-3.1-8B-quantized.w8a8) |   |
| QwQ-32B |   | [RedHatAI/QwQ-32B-quantized.w8a8](https://huggingface.co/RedHatAI/QwQ-32B-quantized.w8a8) |   |
| DeepSeek-Distilled-Llama |   | [RedHatAI/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8](https://huggingface.co/RedHatAI/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8) |   |
| Qwen3-235B |   |   | [Qwen/Qwen3-235B-A22B-FP8](https://huggingface.co/Qwen/Qwen3-235B-A22B-FP8) |

**注意：** 上表中列出的模型标识符
已在第 6 代 Intel® Xeon® P-core 平台上验证通过。

## 安装

### 使用 Docker 安装

推荐使用 Docker 来搭建 SGLang 环境。
我们提供了一个 [Dockerfile](https://github.com/sgl-project/sglang/blob/main/docker/xeon.Dockerfile) 以便于安装。
将下方的 `<secret>` 替换为你的 [HuggingFace access token](https://huggingface.co/docs/hub/en/security-tokens)。

```bash
# Clone the SGLang repository
git clone https://github.com/sgl-project/sglang.git
cd sglang/docker

# Build the docker image
docker build -t sglang-cpu:latest -f xeon.Dockerfile .

# Initiate a docker container
docker run \
    -it \
    --privileged \
    --ipc=host \
    --network=host \
    -v /dev/shm:/dev/shm \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 30000:30000 \
    -e "HF_TOKEN=<secret>" \
    sglang-cpu:latest /bin/bash
```

### 从源码安装

如果你更倾向于在裸金属（bare metal）环境中安装 SGLang，
安装过程如下：

如果系统中尚未安装所需的软件包和库，请预先安装它们。
你可以参考 [the Dockerfile](https://github.com/sgl-project/sglang/blob/main/docker/xeon.Dockerfile#L11)
中基于 Ubuntu 的安装命令作为指引。

1. 安装 `uv` 包管理器，然后创建并激活一个虚拟环境：

```bash
# Taking '/opt' as the example uv env folder, feel free to change it as needed
cd /opt
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.12
source .venv/bin/activate
```

2. 创建一个配置文件，以指定 `torch` 相关包的安装通道
    （也就是 index-url）：

```bash
vim .venv/uv.toml
```

在 `vim` 中按 'a' 进入插入模式，将以下内容粘贴到创建的文件中

```file
[[index]]
name = "torch"
url = "https://download.pytorch.org/whl/cpu"

[[index]]
name = "torchvision"
url = "https://download.pytorch.org/whl/cpu"

[[index]]
name = "torchaudio"
url = "https://download.pytorch.org/whl/cpu"

[[index]]
name = "triton"
url = "https://download.pytorch.org/whl/cpu"

```

保存文件（在 `vim` 中按 'esc' 退出插入模式，然后输入 ':x+Enter'），
并将其设为默认的 `uv` 配置。

```bash
export UV_CONFIG_FILE=/opt/.venv/uv.toml
```

3. 克隆 `sglang` 源码并构建软件包

```bash
# Clone the SGLang code
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout <YOUR-DESIRED-VERSION>

# Use dedicated toml file
cd python
cp pyproject_cpu.toml pyproject.toml
# Install SGLang dependent libs, and build SGLang main package
uv pip install --upgrade pip setuptools
uv pip install .

# Build the CPU backend kernels
cd ../sgl-kernel
cp pyproject_cpu.toml pyproject.toml
uv pip install .
```

4. 设置所需的环境变量

```bash
export SGLANG_USE_CPU_ENGINE=1

# Set 'LD_LIBRARY_PATH' and 'LD_PRELOAD' to ensure the libs can be loaded by sglang processes
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=${LD_PRELOAD}:/opt/.venv/lib/libiomp5.so:${LD_LIBRARY_PATH}/libtcmalloc.so.4:${LD_LIBRARY_PATH}/libtbbmalloc.so.2
```

注意事项：

- 请注意，启用 SGLang 的 CPU 引擎服务需要环境变量 `SGLANG_USE_CPU_ENGINE=1`。

- 如果在 `sgl-kernel` 构建过程中遇到代码编译问题，
    请检查你的 `gcc` 和 `g++` 版本，如果过旧请升级它们。
    建议使用 `gcc-13` 和 `g++-13`，因为它们已在官方 Docker 容器中验证通过。

- 系统库路径通常位于以下目录之一：
    `~/.local/lib/`、`/usr/local/lib/`、`/usr/local/lib64/`、`/usr/lib/`、`/usr/lib64/`
    和 `/usr/lib/x86_64-linux-gnu/`。在上面的示例命令中使用了 `/usr/lib/x86_64-linux-gnu`。
    请根据你的服务器配置调整该路径。

- 建议将以下内容添加到你的 `~/.bashrc` 文件中，
    以避免每次打开新终端时都要设置这些变量：

    ```bash
    source .venv/bin/activate
    export SGLANG_USE_CPU_ENGINE=1
    export LD_LIBRARY_PATH=<YOUR-SYSTEM-LIBRARY-FOLDER>
    export LD_PRELOAD=<YOUR-LIBS-PATHS>
    ```

## 启动推理服务引擎

启动 SGLang 服务的示例命令：

```bash
python -m sglang.launch_server   \
    --model <MODEL_ID_OR_PATH>   \
    --trust-remote-code          \
    --disable-overlap-schedule   \
    --device cpu                 \
    --host 0.0.0.0               \
    --tp 6
```

注意事项：

1. 运行 W8A8 量化模型时，请添加标志 `--quantization w8a8_int8`。

2. 标志 `--tp 6` 指定使用 6 个 rank 进行张量并行（TP6）。
    指定的 TP 数量即执行期间将使用的 TP rank 数量。
    在 CPU 平台上，一个 TP rank 表示一个 sub-NUMA cluster（SNC）。
    通常我们可以通过操作系统获取 SNC 信息（有多少个可用），例如使用 `lscpu` 命令。

    如果指定的 TP rank 数量与 SNC 总数不同，
    系统将自动使用前 `n` 个 SNC。
    请注意 `n` 不能超过 SNC 总数，否则将导致错误。

    `SGLANG_CPU_OMP_THREADS_BIND` 允许显式控制每个张量并行（TP）rank 所使用的 CPU 核心。

    **示例 1**：在 Xeon® 6980P 服务器上以 TP=6 运行 SGLang 服务，使用每个 SNC 的前 40 个核心，
    该服务器的一个 socket 的 3 个 SNC 上分别有 43-43-42 个核心，我们应设置：

    ```bash
    export SGLANG_CPU_OMP_THREADS_BIND="0-39|43-82|86-125|128-167|171-210|214-253"
    ```
    该配置等价于：
    - rank 0: `numactl -C 0-39 -m 0`
    - rank 1: `numactl -C 43-82 -m 1`
    - rank 2: `numactl -C 86-125 -m 2`
    - rank 3: `numactl -C 128-167 -m 3`
    - rank 4: `numactl -C 171-210 -m 4`
    - rank 5: `numactl -C 214-253 -m 5`


    **示例 2**：在 Xeon® 6972P 服务器上以 TP=2 运行 SGLang 服务，跨 3 个 SNC 使用 96 个核心，
    该服务器一个 socket 中的 3 个 SNC 上分别有 32-32-32 个核心，我们应设置：
    ```bash
    export SGLANG_CPU_OMP_THREADS_BIND="0-95|96-191"
    ```
    该配置等价于：
    - rank 0: `numactl -C 0-95 -m 0-2`
    - rank 1: `numactl -C 96-191 -m 3-5`

    请注意，设置 SGLANG_CPU_OMP_THREADS_BIND 后，
    各 rank 的可用内存量可能无法事先确定。
    你可能需要设置合适的 `--max-total-tokens` 以避免内存溢出（out-of-memory）错误。

3. 若要使用 torch.compile 优化解码，请添加标志 `--enable-torch-compile`。
    若要指定使用 `torch.compile` 时的最大批大小，请设置标志 `--torch-compile-max-bs`。
    例如，`--enable-torch-compile --torch-compile-max-bs 4` 表示使用 `torch.compile`
    并将最大批大小设为 4。

4. 服务启动时会自动触发一个预热（warmup）步骤。
    当你看到日志 `The server is fired up and ready to roll!` 时，服务器即已就绪。

## 使用请求进行基准测试

你可以通过 `bench_serving` 脚本对性能进行基准测试。
在另一个终端中运行该命令。示例命令如下：

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

此外，请求也可以使用
[the OpenAI Completions API](https://docs.sglang.io/basic_usage/openai_api_completions.html)
构造，并通过命令行（例如使用 `curl`）或你自己的脚本发送。

## 示例使用命令

大语言模型的参数量从不到 10 亿到数千亿不等。
大于 20B 的稠密（dense）模型预计需要在配备双 socket、共计 6 个 sub-NUMA cluster 的旗舰级第 6 代 Intel® Xeon® 处理器上运行。
约 100 亿参数及以下的稠密模型，
或激活参数少于 100 亿的 MoE（Mixture of Experts）模型，则可以在更常见的
第 4 代或更新的 Intel® Xeon® 处理器上运行，或使用旗舰级第 6 代 Intel® Xeon® 处理器的单个 socket。

### 示例：运行 DeepSeek-V3.1-Terminus

在 Xeon® 6980P 服务器上启动 W8A8_INT8 版 DeepSeek-V3.1-Terminus 服务的示例命令：

```bash
python -m sglang.launch_server                                 \
    --model IntervitensInc/DeepSeek-V3.1-Terminus-Channel-int8 \
    --trust-remote-code                                        \
    --disable-overlap-schedule                                 \
    --device cpu                                               \
    --quantization w8a8_int8                                   \
    --host 0.0.0.0                                             \
    --enable-torch-compile                                     \
    --torch-compile-max-bs 4                                   \
    --tp 6
```

类似地，启动 FP8 版 DeepSeek-V3.1-Terminus 服务的示例命令为：

```bash
python -m sglang.launch_server                     \
    --model deepseek-ai/DeepSeek-V3.1-Terminus     \
    --trust-remote-code                            \
    --disable-overlap-schedule                     \
    --device cpu                                   \
    --host 0.0.0.0                                 \
    --enable-torch-compile                         \
    --torch-compile-max-bs 4                       \
    --tp 6
```

注意：请将 `--torch-compile-max-bs` 设为你部署所需的最大批大小，
其值最高可达 16。示例中的 `4` 仅作说明用途。

### 示例：运行 Llama-3.2-3B

以 BF16 精度启动 Llama-3.2-3B 服务的示例命令：

```bash
python -m sglang.launch_server                     \
    --model meta-llama/Llama-3.2-3B-Instruct       \
    --trust-remote-code                            \
    --disable-overlap-schedule                     \
    --device cpu                                   \
    --host 0.0.0.0                                 \
    --enable-torch-compile                         \
    --torch-compile-max-bs 16                      \
    --tp 2
```

启动 W8A8_INT8 版 Llama-3.2-3B 服务的示例命令：

```bash
python -m sglang.launch_server                     \
    --model RedHatAI/Llama-3.2-3B-quantized.w8a8   \
    --trust-remote-code                            \
    --disable-overlap-schedule                     \
    --device cpu                                   \
    --quantization w8a8_int8                       \
    --host 0.0.0.0                                 \
    --enable-torch-compile                         \
    --torch-compile-max-bs 16                      \
    --tp 2
```

注意：`--torch-compile-max-bs` 和 `--tp` 的设置仅为示例，应根据你的实际配置进行调整。
例如，在 Intel® Xeon® 6980P 服务器上使用 `--tp 3` 来利用 1 个 socket 的 3 个 sub-NUMA cluster。

服务器启动后，你可以使用 `bench_serving` 命令进行测试，或按照[基准测试示例](#benchmarking-with-requests)创建你自己的命令或脚本。
