# MindSpore 模型

## 简介

MindSpore 是一个针对 Ascend NPU 优化的高性能 AI 框架。本文档指导用户在 SGLang 中运行 MindSpore 模型。

## 要求

MindSpore 目前仅支持 Ascend NPU 设备。用户需要首先安装 Ascend CANN 软件包。
CANN 软件包可从 [Ascend 官方网站](https://www.hiascend.com)下载。推荐版本为 8.3.RC2。

## 支持的模型

目前支持以下模型:

- **Qwen3**:Dense 和 MoE 模型
- **DeepSeek V3/R1**
- *更多模型即将推出……*

## 安装

> **注意**:目前 MindSpore 模型由一个独立的包 `sgl-mindspore` 提供。对 MindSpore 的支持构建于 SGLang 当前对 Ascend NPU 平台的支持之上。请先[为 Ascend NPU 安装 SGLang](ascend_npu.md),然后再安装 `sgl-mindspore`:

```shell
git clone https://github.com/mindspore-lab/sgl-mindspore.git
cd sgl-mindspore
pip install -e .
```


## 运行模型

当前 SGLang-MindSpore 支持 Qwen3 和 DeepSeek V3/R1 模型。本文档以 Qwen3-8B 为例。

### 离线推理

使用以下脚本进行离线推理:

```python
import sglang as sgl

# Initialize the engine with MindSpore backend
llm = sgl.Engine(
    model_path="/path/to/your/model",  # Local model path
    device="npu",                      # Use NPU device
    model_impl="mindspore",            # MindSpore implementation
    attention_backend="ascend",        # Attention backend
    tp_size=1,                         # Tensor parallelism size
    dp_size=1                          # Data parallelism size
)

# Generate text
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "The future of AI is"
]

sampling_params = {"temperature": 0, "top_p": 0.9}
outputs = llm.generate(prompts, sampling_params)

for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}")
    print(f"Generated: {output['text']}")
    print("---")
```

### 启动服务

使用 MindSpore 后端启动服务:

```bash
# Basic server startup
python3 -m sglang.launch_server \
    --model-path /path/to/your/model \
    --host 0.0.0.0 \
    --device npu \
    --model-impl mindspore \
    --attention-backend ascend \
    --tp-size 1 \
    --dp-size 1
```

多节点分布式服务:

```bash
# Multi-node distributed server
python3 -m sglang.launch_server \
    --model-path /path/to/your/model \
    --host 0.0.0.0 \
    --device npu \
    --model-impl mindspore \
    --attention-backend ascend \
    --dist-init-addr 127.0.0.1:29500 \
    --nnodes 2 \
    --node-rank 0 \
    --tp-size 4 \
    --dp-size 2
```

## 故障排查

#### 调试模式

通过 log-level 参数启用 sglang 调试日志。

```bash
python3 -m sglang.launch_server \
    --model-path /path/to/your/model \
    --host 0.0.0.0 \
    --device npu \
    --model-impl mindspore \
    --attention-backend ascend \
    --log-level DEBUG
```

通过设置环境变量启用 mindspore 的 info 和 debug 日志。

```bash
export GLOG_v=1  # INFO
export GLOG_v=0  # DEBUG
```

#### 显式选择设备

使用以下环境变量显式选择要使用的设备。

```shell
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7  # to set device
```

#### 一些通信环境问题

在某些具有特殊通信环境的环境中,用户需要设置一些环境变量。

```shell
export MS_ENABLE_LCCL=off # current not support LCCL communication mode in SGLang-MindSpore
```

#### 一些 protobuf 依赖

在某些具有特殊 protobuf 版本的环境中,用户需要设置一些环境变量以避免二进制版本不匹配。

```shell
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python  # to avoid protobuf binary version mismatch
```

## 支持
针对 MindSpore 相关的问题:

- 请参阅 [MindSpore 文档](https://www.mindspore.cn/)
