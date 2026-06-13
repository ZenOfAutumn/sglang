# Checkpoint Engine 集成

SGLang 的 checkpoint engine 集成提供了一种使用分布式 checkpoint 加载系统来高效加载模型权重的方式。该功能通过在多个进程和节点间并行化权重加载过程,显著减少了模型加载时间,尤其是对于大模型和多节点设置。

## 概述

checkpoint engine 集成使 SGLang 能够:
- 使用多个进程并行加载模型权重
- 将权重加载分布到多个节点,以提升有效磁盘带宽
- 将权重加载与其他初始化任务(如 CUDA graph 捕获)重叠
- 支持单节点和多节点部署

## 安装

首先,安装 checkpoint engine 包:

```bash
pip install 'checkpoint-engine[p2p]'
```

## 架构

该系统由两个主要组件组成:

1. **SGLang Server**:使用 `--wait-for-initial-weights` 标志运行,在权重就绪之前等待,然后才进入就绪状态
2. **Checkpoint Engine Workers**:独立的进程(由 torchrun 管理),负责加载和分发模型权重

checkpoint engine 使用 parameter server 架构,支持:
- **Broadcast 模式**:权重从加载进程广播到推理进程
- **P2P 模式**:进程间的直接 peer-to-peer 权重传输
- **All 模式**:broadcast 和 P2P 两种方法的组合

## 使用示例

### 单节点设置

**终端 1 - 启动 SGLang Server:**
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp 8 \
    --load-format dummy \
    --wait-for-initial-weights
```

**终端 2 - 运行 Checkpoint Engine:**

使用 sglang 入口:
```bash
python -m sglang.srt.checkpoint_engine.update \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 8
```

直接使用 torchrun:
```bash
torchrun --nproc-per-node 8 \
    examples/checkpoint_engine/update.py \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 8
```

### 多节点设置(2 个节点)

**Node 0:**

启动 SGLang server:
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp 8 \
    --load-format dummy \
    --wait-for-initial-weights \
    --host [IP]
```

运行 checkpoint engine:

使用 sglang 入口(推荐):
```bash
python -m sglang.srt.checkpoint_engine.update \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 8
```

直接使用 torchrun:
```bash
torchrun --nproc-per-node 8 \
    --nnodes 2 \
    --node-rank 0 \
    --master-addr [IP] \
    --master-port 29500 \
    examples/checkpoint_engine/update.py \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 8
```

**Node 1:**

启动 SGLang server:
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp 8 \
    --load-format dummy \
    --wait-for-initial-weights \
    --host [IP]
```

运行 checkpoint engine:

使用 sglang 入口(推荐):
```bash
python -m sglang.srt.checkpoint_engine.update \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 8
```

直接使用 torchrun:
```bash
torchrun --nproc-per-node 8 \
    --nnodes 2 \
    --node-rank 1 \
    --master-addr [IP] \
    --master-port 29500 \
    examples/checkpoint_engine/update.py \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 8
```

### 带 Tensor Parallelism 的多节点设置(TP=16)

**Node 0:**

启动 SGLang server:
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp 8 \
    --load-format dummy \
    --wait-for-initial-weights \
    --host [IP] \
    --dist-init-addr [IP]:9120 \
    --nnodes 2 \
    --node-rank 0
```

运行 checkpoint engine:

使用 sglang 入口(推荐):
```bash
python -m sglang.srt.checkpoint_engine.update \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 16
```

直接使用 torchrun:
```bash
torchrun --nproc-per-node 8 \
    --nnodes 2 \
    --node-rank 0 \
    --master-addr [IP] \
    --master-port 29500 \
    examples/checkpoint_engine/update.py \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 16
```

**Node 1:**

启动 SGLang server:
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp 8 \
    --load-format dummy \
    --wait-for-initial-weights \
    --host [IP] \
    --dist-init-addr [IP]:9120 \
    --nnodes 2 \
    --node-rank 1
```

运行 checkpoint engine:

使用 sglang 入口(推荐):
```bash
python -m sglang.srt.checkpoint_engine.update \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 16
```

直接使用 torchrun:
```bash
torchrun --nproc-per-node 8 \
    --nnodes 2 \
    --node-rank 1 \
    --master-addr [IP] \
    --master-port 29500 \
    examples/checkpoint_engine/update.py \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 16
```

## 配置选项

### SGLang Server 选项

- `--load-format dummy`:使用 dummy 格式进行初始加载(允许与其他任务重叠)
- `--wait-for-initial-weights`:在 checkpoint engine 提供权重之前等待,然后才进入就绪状态
- `--host`:多节点设置的主机地址
- `--dist-init-addr`:用于 tensor parallelism 的分布式初始化地址

### Checkpoint Engine 选项

- `--update-method`:权重更新方法(`broadcast`、`p2p` 或 `all`)
- `--checkpoint-path`:模型 checkpoint 目录的路径
- `--inference-parallel-size`:推理并行进程的数量
- `--endpoint`:SGLang server 端点(默认:`http://localhost:19730`)
- `--checkpoint-name`:checkpoint 的名称(默认:`my-checkpoint-iter-0`)
- `--save-metas-file`:用于保存 checkpoint 元数据的文件
- `--load-metas-file`:用于加载 checkpoint 元数据的文件
- `--uds`:用于通信的 Unix domain socket 路径
- `--weight-version`:权重的版本标识符

## 性能收益

checkpoint engine 在两个主要方面提供了显著的时间节省:

1. **多节点加载**:每个节点只从磁盘加载一部分权重,从而有效提升磁盘带宽。参与的节点越多,加速效果越大。初步测试显示,在 H20-3e 上使用两个节点加载 DeepSeek-R1 时可加速 20 秒。

2. **单进程优化**:使用 dummy 格式允许将磁盘到 CPU 的传输与 CUDA graph 捕获及其他初始化任务重叠,从而带来额外的时间节省。

## 故障排查

- 确保已安装 checkpoint engine 包:`pip install 'checkpoint-engine[p2p]'`
- 验证多节点设置中节点之间的网络连通性
- 检查 checkpoint 路径中包含有效的模型文件
- 监控 SGLang server 与 checkpoint engine 之间的连接错误日志
- 如有调试需要,使用 `--sleep-time` 参数添加延迟

## 参考

- [Checkpoint Engine 仓库](https://github.com/MoonshotAI/checkpoint-engine)
