# 安装 SGLang

你可以使用下列方法之一来安装 SGLang。
本页面主要适用于常见的 NVIDIA GPU 平台。
对于其他或更新的平台，请参考各自的专属页面：[AMD GPUs](../platforms/amd_gpu.md)、[Intel Xeon CPUs](../platforms/cpu_server.md)、[TPU](../platforms/tpu.md)、[NVIDIA DGX Spark](https://lmsys.org/blog/2025-11-03-gpt-oss-on-nvidia-dgx-spark/)、[NVIDIA Jetson](../platforms/nvidia_jetson.md)、[Ascend NPUs](../platforms/ascend/ascend_npu.md) 和 [Intel XPU](../platforms/xpu.md)。

## 方法一：使用 pip 或 uv

推荐使用 uv 以获得更快的安装速度：

```bash
pip install --upgrade pip
pip install uv
uv pip install sglang
```

### 针对 CUDA 13

推荐使用 Docker（参见方法三中关于 B300/GB300/CUDA 13 的说明）。如果你无法使用 Docker，请按以下步骤操作：

1. 首先安装支持 CUDA 13 的 PyTorch：
```bash
# Replace X.Y.Z with the version by your SGLang install
uv pip install torch==X.Y.Z torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

2. 安装 sglang：
```bash
uv pip install sglang
```

3. 从 [the sgl-project whl releases](https://github.com/sgl-project/whl/blob/gh-pages/cu130/sglang-kernel/index.html) 安装适用于 CUDA 13 的 `sglang-kernel` wheel。将 `X.Y.Z` 替换为你的 SGLang 安装所需的 `sglang-kernel` 版本（可通过运行 `uv pip show sglang-kernel` 查看）。示例：
```bash
# x86_64
uv pip install "https://github.com/sgl-project/whl/releases/download/vX.Y.Z/sglang_kernel-X.Y.Z+cu130-cp310-abi3-manylinux2014_x86_64.whl"

# aarch64
uv pip install "https://github.com/sgl-project/whl/releases/download/vX.Y.Z/sglang_kernel-X.Y.Z+cu130-cp310-abi3-manylinux2014_aarch64.whl"
```

### **常见问题的快速修复**
- 如果遇到 `OSError: CUDA_HOME environment variable is not set`，请用以下任一方案将其设置为你的 CUDA 安装根目录：
  1. 使用 `export CUDA_HOME=/usr/local/cuda-<your-cuda-version>` 来设置 `CUDA_HOME` 环境变量。
  2. 先按照 [FlashInfer installation doc](https://docs.flashinfer.ai/installation.html) 安装 FlashInfer，然后按上文所述安装 SGLang。

## 方法二：从源码安装

```bash
# Use the last release branch
git clone -b v0.5.9 https://github.com/sgl-project/sglang.git
cd sglang

# Install the python packages
pip install --upgrade pip
pip install -e "python"
```

**常见问题的快速修复**

- 如果你想开发 SGLang，可以尝试使用 dev docker 镜像。请参考 [setup docker container](../developer_guide/development_guide_using_docker.md#setup-docker-container)。该 docker 镜像为 `lmsysorg/sglang:dev`。

## 方法三：使用 docker

docker 镜像可在 Docker Hub 上的 [lmsysorg/sglang](https://hub.docker.com/r/lmsysorg/sglang/tags) 获取，它们由 [Dockerfile](https://github.com/sgl-project/sglang/tree/main/docker) 构建。
请将下方的 `<secret>` 替换为你的 huggingface hub [token](https://huggingface.co/docs/hub/en/security-tokens)。

```bash
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<secret>" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 30000
```

对于生产环境部署，请使用 `runtime` 变体，它通过排除构建工具和开发依赖项而显著更小（缩减约 40%）：

```bash
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<secret>" \
    --ipc=host \
    lmsysorg/sglang:latest-runtime \
    python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 30000
```

你也可以在[这里](https://hub.docker.com/r/lmsysorg/sglang/tags?name=nightly)找到 nightly docker 镜像。

注意事项：
- 在 B300/GB300 (SM103) 或 CUDA 13 环境下，我们推荐使用 `lmsysorg/sglang:dev-cu13` 的 nightly 镜像，或 `lmsysorg/sglang:latest-cu130-runtime` 的 stable 镜像。请不要在 docker 镜像内将项目重新以可编辑（editable）方式安装，因为这会覆盖 cu13 docker 镜像所指定的库版本。

## 方法四：使用 Kubernetes

请查看 [OME](https://github.com/sgl-project/ome)，这是一个用于企业级管理和部署大语言模型（LLM）的 Kubernetes operator。

<details>
<summary>更多</summary>

1. 选项一：单节点部署（通常用于模型大小可装入一个节点上的 GPU 的情况）

   执行命令 `kubectl apply -f docker/k8s-sglang-service.yaml`，以创建 k8s deployment 和 service，示例使用 llama-31-8b。

2. 选项二：多节点部署（通常用于大型模型需要多个 GPU 节点的情况，例如 `DeepSeek-R1`）

   根据需要修改 LLM 模型路径和参数，然后执行命令 `kubectl apply -f docker/k8s-sglang-distributed-sts.yaml`，以创建两节点的 k8s statefulset 和部署 service。

</details>

## 方法五：使用 docker compose

<details>
<summary>更多</summary>

> 如果你打算将其作为服务来部署，推荐使用此方法。
> 更好的方式是使用 [k8s-sglang-service.yaml](https://github.com/sgl-project/sglang/blob/main/docker/k8s-sglang-service.yaml)。

1. 将 [compose.yml](https://github.com/sgl-project/sglang/blob/main/docker/compose.yaml) 复制到你的本地机器
2. 在终端中执行命令 `docker compose up -d`。
</details>

## 方法六：使用 SkyPilot 在 Kubernetes 或云上运行

<details>
<summary>更多</summary>

要在 Kubernetes 或 12 多个云平台上部署，你可以使用 [SkyPilot](https://github.com/skypilot-org/skypilot)。

1. 安装 SkyPilot 并设置 Kubernetes 集群或云访问：参见 [SkyPilot's documentation](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html)。
2. 用一条命令在你自己的基础设施上部署，并获取 HTTP API 端点：
<details>
<summary>SkyPilot YAML: <code>sglang.yaml</code></summary>

```yaml
# sglang.yaml
envs:
  HF_TOKEN: null

resources:
  image_id: docker:lmsysorg/sglang:latest
  accelerators: A100
  ports: 30000

run: |
  conda deactivate
  python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 30000
```

</details>

```bash
# Deploy on any cloud or Kubernetes cluster. Use --cloud <cloud> to select a specific cloud provider.
HF_TOKEN=<secret> sky launch -c sglang --env HF_TOKEN sglang.yaml

# Get the HTTP API endpoint
sky status --endpoint 30000 sglang
```

3. 若要通过自动扩缩容和故障恢复进一步扩展你的部署，请查看 [SkyServe + SGLang guide](https://github.com/skypilot-org/skypilot/tree/master/llm/sglang#serving-llama-2-with-sglang-for-more-traffic-using-skyserve)。

</details>

## 方法七：在 AWS SageMaker 上运行

<details>
<summary>更多</summary>

要在 AWS SageMaker 上部署 SGLang，请查看 [AWS SageMaker Inference](https://aws.amazon.com/sagemaker/ai/deploy)

Amazon Web Services 为 SGLang 容器提供支持，并提供例行的安全补丁。可用的 SGLang 容器请查看 [AWS SGLang DLCs](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#sglang-containers)

要使用你自己的容器托管模型，请按以下步骤操作：

1. 使用 [sagemaker.Dockerfile](https://github.com/sgl-project/sglang/blob/main/docker/sagemaker.Dockerfile) 以及 [serve](https://github.com/sgl-project/sglang/blob/main/docker/serve) 脚本构建一个 docker 容器。
2. 将你的容器推送到 AWS ECR。

<details>
<summary>Dockerfile 构建脚本：<code>build-and-push.sh</code></summary>

```bash
#!/bin/bash
AWS_ACCOUNT="<YOUR_AWS_ACCOUNT>"
AWS_REGION="<YOUR_AWS_REGION>"
REPOSITORY_NAME="<YOUR_REPOSITORY_NAME>"
IMAGE_TAG="<YOUR_IMAGE_TAG>"

ECR_REGISTRY="${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${ECR_REGISTRY}/${REPOSITORY_NAME}:${IMAGE_TAG}"

echo "Starting build and push process..."

# Login to ECR
echo "Logging into ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REGISTRY}

# Build the image
echo "Building Docker image..."
docker build -t ${IMAGE_URI} -f sagemaker.Dockerfile .

echo "Pushing ${IMAGE_URI}"
docker push ${IMAGE_URI}

echo "Build and push completed successfully!"
```

</details>

3. 在 AWS Sagemaker 上部署模型进行服务，参考 [deploy_and_serve_endpoint.py](https://github.com/sgl-project/sglang/blob/main/examples/sagemaker/deploy_and_serve_endpoint.py)。更多信息请查看 [sagemaker-python-sdk](https://github.com/aws/sagemaker-python-sdk)。
   1. 默认情况下，SageMaker 上的模型服务器将使用以下命令运行：`python3 -m sglang.launch_server --model-path opt/ml/model --host 0.0.0.0 --port 8080`。这对于使用 SageMaker 托管你自己的模型是最优的。
   2. 要修改你的模型服务参数，[serve](https://github.com/sgl-project/sglang/blob/main/docker/serve) 脚本允许通过指定以 `SM_SGLANG_` 为前缀的环境变量来设置 `python3 -m sglang.launch_server --help` cli 中所有可用的选项。
   3. serve 脚本会自动将所有以 `SM_SGLANG_` 为前缀的环境变量从 `SM_SGLANG_INPUT_ARGUMENT` 转换为 `--input-argument`，以便解析传入 `python3 -m sglang.launch_server` cli。
   4. 例如，要运行带有 reasoning parser 的 [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)，只需添加额外的环境变量 `SM_SGLANG_MODEL_PATH=Qwen/Qwen3-0.6B` 和 `SM_SGLANG_REASONING_PARSER=qwen3`。

</details>

## 通用说明

- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) 是默认的 attention kernel 后端。它仅支持 sm75 及以上。如果你在 sm75+ 设备上（例如 T4、A10、A100、L4、L40S、H100）遇到任何与 FlashInfer 相关的问题，请通过添加 `--attention-backend triton --sampling-backend pytorch` 切换到其他 kernel，并在 GitHub 上提交 issue。
- 要在本地重新安装 flashinfer，请使用以下命令：`pip3 install --upgrade flashinfer-python --force-reinstall --no-deps`，然后用 `rm -rf ~/.cache/flashinfer` 删除缓存。
- 在 B300/GB300 上遇到 `ptxas fatal   : Value 'sm_103a' is not defined for option 'gpu-name'` 时，可通过 `export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` 修复。
