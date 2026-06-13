# 安装 SGLang-Diffusion

你可以使用以下方法之一安装 SGLang-Diffusion。标准安装已经包含了 SGLang 的优化 kernel 栈,包括 diffusion 工作负载所使用的 `sgl-kernel` 和 JIT kernel。

## 标准安装(NVIDIA GPU)

### 方法 1:使用 pip 或 uv

推荐使用 uv 以获得更快的安装速度:

```bash
pip install --upgrade pip
pip install uv
uv pip install "sglang[diffusion]" --prerelease=allow
```

### 方法 2:从源码安装

```bash
# Use the latest release branch
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Install the Python packages
pip install --upgrade pip
pip install -e "python[diffusion]"

# With uv
uv pip install -e "python[diffusion]" --prerelease=allow
```

### 方法 3:使用 Docker

Docker 镜像可在 Docker Hub 的 [lmsysorg/sglang](https://hub.docker.com/r/lmsysorg/sglang) 获取,由 [Dockerfile](https://github.com/sgl-project/sglang/blob/main/docker/Dockerfile) 构建。
将下面的 `<secret>` 替换为你的 HuggingFace Hub [token](https://huggingface.co/docs/hub/en/security-tokens)。

```bash
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<secret>" \
    --ipc=host \
    lmsysorg/sglang:dev \
    zsh -c '\
        echo "Installing diffusion dependencies..." && \
        pip install -e "python[diffusion]" && \
        echo "Starting SGLang-Diffusion..." && \
        sglang generate \
            --model-path black-forest-labs/FLUX.1-dev \
            --prompt "A logo With Bold Large text: SGL Diffusion" \
            --save-output \
    '
```

## 平台专用:ROCm(AMD GPU)

对于 AMD Instinct GPU(例如 MI300X),你可以使用启用了 ROCm 的 Docker 镜像:

```bash
docker run --device=/dev/kfd --device=/dev/dri --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env HF_TOKEN=<secret> \
  lmsysorg/sglang:v0.5.5.post2-rocm700-mi30x \
  sglang generate --model-path black-forest-labs/FLUX.1-dev --prompt "A logo With Bold Large text: SGL Diffusion" --save-output
```

有关 ROCm 系统的详细配置以及从源码安装,请参见 [AMD GPUs](../platforms/amd_gpu.md)。

## 平台专用:MUSA(Moore Threads GPU)

对于采用 MUSA 软件栈的 Moore Threads GPU(MTGPU),请按照以下说明从源码安装:

```bash
# Clone the repository
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Install the Python packages
pip install --upgrade pip
rm -f python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_musa]"
```

## 平台专用:Ascend NPU

对于 Ascend NPU,请参照 [NPU installation guide](../platforms/ascend/ascend_npu.md)。

快速测试:

```bash
sglang generate --model-path black-forest-labs/FLUX.1-dev \
    --prompt "A logo With Bold Large text: SGL Diffusion" \
    --save-output
```

## 平台专用:Apple MPS

对于 Apple MPS,请按照以下说明从源码安装:

```bash
# Install ffmpeg
brew install ffmpeg

# Install uv
brew install uv

# Clone the repository
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Create and activate a virtual environment
uv venv -p 3.11 sglang-diffusion
source sglang-diffusion/bin/activate

# Install the Python packages
uv pip install --upgrade pip
rm -f python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
uv pip install -e "python[all_mps]"
```
