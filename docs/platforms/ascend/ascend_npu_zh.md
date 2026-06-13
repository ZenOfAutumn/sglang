
# 支持 NPU 的 SGLang 安装指南

你可以使用下面任意一种方法安装 SGLang。请务必阅读 `System Settings` 部分,以确保集群以最大性能运行。如果你遇到任何问题或困难,欢迎在 [sglang 这里](https://github.com/sgl-project/sglang/issues) 提交 issue。

## SGLang 的组件版本对应关系
| 组件              | 版本                    | 获取方式                                                                                                                                                                                                                       |
|-------------------|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| HDK               | 25.3.RC1                  | [link](https://www.hiascend.com/hardware/firmware-drivers/commercial?product=7&model=33) |
| CANN              | 8.5.0                     | [获取镜像](#obtain-cann-image)                                                                                                                                                                                          |
| Pytorch Adapter   | 7.3.0                   | [link](https://gitcode.com/Ascend/pytorch/releases)                                                                                                                                                                          |
| MemFabric         | 1.0.5                   | `pip install memfabric-hybrid==1.0.5`                                                                                                                                                                 |
| Triton            | 3.2.0                   | `pip install triton-ascend`|
| SGLang NPU Kernel | NA                      | [link](https://github.com/sgl-project/sgl-kernel-npu/releases)                                                                                                                                                               |

<a id="obtain-cann-image"></a>
### 获取 CANN 镜像
你可以通过镜像获取指定版本 CANN 的依赖。
```shell
# for Atlas 800I A3 and Ubuntu OS
docker pull quay.io/ascend/cann:8.5.0-a3-ubuntu22.04-py3.11
# for Atlas 800I A2 and Ubuntu OS
docker pull quay.io/ascend/cann:8.5.0-910b-ubuntu22.04-py3.11
```

## 准备运行环境

### 方法一:从源码安装(含前置依赖)

#### Python 版本

目前仅支持 `python==3.11`。如果你不想破坏系统预装的 python,可以尝试使用 [conda](https://github.com/conda/conda) 安装。

```shell
conda create --name sglang_npu python=3.11
conda activate sglang_npu
```

#### CANN

在 Ascend 上开始使用 SGLang 之前,你需要安装 CANN Toolkit、Kernels 算子包以及 8.3.RC2 或更高版本的 NNAL,请参阅[安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)

#### MemFabric-Hybrid

如果你想使用 PD 分离(PD disaggregation)模式,需要安装 MemFabric-Hybrid。MemFabric-Hybrid 是 Mooncake Transfer Engine 的直接替代品,可在 Ascend NPU 集群上实现 KV cache 传输。

```shell
pip install memfabric-hybrid==1.0.5
```

#### Ascend 上的 Pytorch 与 Pytorch 框架适配器

```shell
PYTORCH_VERSION=2.8.0
TORCHVISION_VERSION=0.23.0
TORCH_NPU_VERSION=2.8.0
pip install torch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu==$TORCH_NPU_VERSION
```

如果你使用其他版本的 `torch` 并安装 `torch_npu`,请参阅[安装指南](https://github.com/Ascend/pytorch/blob/master/README.md)

#### Ascend 上的 Triton

我们为 Ascend 提供了自己实现的 Triton。

```shell
pip install triton-ascend
```
关于在 Ascend 上安装 Triton 的 nightly 构建版本或从源码安装,请参阅[安装指南](https://gitcode.com/Ascend/triton-ascend/blob/master/docs/sources/getting-started/installation.md)

#### SGLang Kernels NPU
我们为 Ascend NPU 提供了 SGL kernels,请参阅[安装指南](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/README.md)。

#### DeepEP 兼容库
我们提供了一个 DeepEP 兼容库,作为 deepseek-ai 的 DeepEP 库的直接替代品,请参阅[安装指南](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md)。

#### 从源码安装 SGLang

```shell
# Use the last release branch
git clone https://github.com/sgl-project/sglang.git
cd sglang
mv python/pyproject_npu.toml python/pyproject.toml
pip install -e python[all_npu]
```

### 方法二:使用 Docker 镜像
#### 获取镜像
你可以下载 SGLang 镜像,或基于 Dockerfile 构建镜像,以获取 Ascend NPU 镜像。
1. 下载 SGLang 镜像
```angular2html
dockerhub: docker.io/lmsysorg/sglang:$tag
# Main-based tag, change main to specific version like v0.5.6,
# you can get image for specific version
Atlas 800I A3 : {main}-cann8.5.0-a3
Atlas 800I A2: {main}-cann8.5.0-910b
```
2. 基于 Dockerfile 构建镜像
```shell
# Clone the SGLang repository
git clone https://github.com/sgl-project/sglang.git
cd sglang/docker

# Build the docker image
# If there are network errors, please modify the Dockerfile to use offline dependencies or use a proxy
docker build -t <image_name> -f npu.Dockerfile .
```

#### 创建 Docker
__注意:__ `--privileged` 和 `--network=host` 是 RDMA 所必需的,而 RDMA 通常是 Ascend NPU 集群所需要的。

__注意:__ 下面的 docker 命令基于 Atlas 800I A3 机器。如果你使用的是 Atlas 800I A2,请确保只把 `davinci[0-7]` 映射进容器。

```shell

alias drun='docker run -it --rm --privileged --network=host --ipc=host --shm-size=16g \
    --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \
    --device=/dev/davinci8 --device=/dev/davinci9 --device=/dev/davinci10 --device=/dev/davinci11 \
    --device=/dev/davinci12 --device=/dev/davinci13 --device=/dev/davinci14 --device=/dev/davinci15 \
    --device=/dev/davinci_manager --device=/dev/hisi_hdc \
    --volume /usr/local/sbin:/usr/local/sbin --volume /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    --volume /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    --volume /etc/ascend_install.info:/etc/ascend_install.info \
    --volume /var/queue_schedule:/var/queue_schedule --volume ~/.cache/:/root/.cache/'

# Add HF_TOKEN env for download model by SGLang.
drun --env "HF_TOKEN=<secret>" \
    <image_name> \
    python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --attention-backend ascend
```

## 系统设置

### CPU 性能电源方案

Ascend 硬件上的默认电源方案是 `ondemand`,这可能会影响性能,建议将其改为 `performance`。

```shell
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Make sure changes are applied successfully
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor # shows performance
```

### 禁用 NUMA 平衡

```shell
sudo sysctl -w kernel.numa_balancing=0
# Check
cat /proc/sys/kernel/numa_balancing # shows 0
```

### 防止系统内存被换出

```shell
sudo sysctl -w vm.swappiness=10

# Check
cat /proc/sys/vm/swappiness # shows 10
```

## 运行 SGLang 服务
### 运行大语言模型服务
#### PD 混合场景
```shell
# Enabling CPU Affinity
export SGLANG_SET_CPU_AFFINITY=1
python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --attention-backend ascend
```

#### PD 分离场景
1. 启动 Prefill Server
```shell
# Enabling CPU Affinity
export SGLANG_SET_CPU_AFFINITY=1

# PIP: recommended to config first Prefill Server IP
# PORT: one free port
# all sglang servers need to be config the same PIP and PORT,
export ASCEND_MF_STORE_URL="tcp://PIP:PORT"
# if you are Atlas 800I A2 hardware and use rdma for kv cache transfer, add this parameter
export ASCEND_MF_TRANSFER_PROTOCOL="device_rdma"
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend ascend \
    --disaggregation-bootstrap-port 8995 \
    --attention-backend ascend \
    --device npu \
    --base-gpu-id 0 \
    --tp-size 1 \
    --host 127.0.0.1 \
    --port 8000
```

2. 启动 Decode Server
```shell
# PIP: recommended to config first Prefill Server IP
# PORT: one free port
# all sglang servers need to be config the same PIP and PORT,
export ASCEND_MF_STORE_URL="tcp://PIP:PORT"
# if you are Atlas 800I A2 hardware and use rdma for kv cache transfer, add this parameter
export ASCEND_MF_TRANSFER_PROTOCOL="device_rdma"
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend ascend \
    --attention-backend ascend \
    --device npu \
    --base-gpu-id 1 \
    --tp-size 1 \
    --host 127.0.0.1 \
    --port 8001
```

3. 启动 Router
```shell
python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://127.0.0.1:8000 8995 \
    --decode http://127.0.0.1:8001 \
    --host 127.0.0.1 \
    --port 6688
```

### 运行多模态语言模型服务
#### PD 混合场景
```shell
python3 -m sglang.launch_server \
    --model-path Qwen3-VL-30B-A3B-Instruct \
    --host 127.0.0.1 \
    --port 8000 \
    --tp 4 \
    --device npu \
    --attention-backend ascend \
    --mm-attention-backend ascend_attn \
    --disable-radix-cache \
    --trust-remote-code \
    --enable-multimodal \
    --sampling-backend ascend
```
