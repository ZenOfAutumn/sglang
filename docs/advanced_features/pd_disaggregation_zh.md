# PD 分离（PD Disaggregation）

## 为什么需要以及什么是 PD 分离？

大语言模型（LLM）推理包含两个不同的阶段：**Prefill（预填充）** 和 **Decode（解码）**。Prefill 阶段是计算密集型的，需要处理整个输入序列；而 Decode 阶段是内存密集型的，需要管理用于 token 生成的 Key-Value（KV）缓存。传统上，这两个阶段在统一引擎中处理，其中 prefill 和 decode 批次的混合调度会引入低效问题。为了解决这些挑战，我们在 SGLang 中引入了 **Prefill 与 Decode（PD）分离**。

### 统一调度存在的问题

传统的统一引擎将 prefill 和 decode 批次一起处理，会导致两个显著的问题：

1. **Prefill 中断**：到来的 prefill 批次频繁中断正在进行的 decode 批次，导致 token 生成出现大量延迟。
2. **DP Attention 不均衡**：在数据并行（DP）attention 中，一个 DP worker 可能正在处理 prefill 批次，而另一个 DP worker 同时处理 decode 批次，从而导致 decode 延迟增加。

PD 分离通过将这两个阶段分开来解决上述问题，使得每个阶段都能进行针对性的优化。

有关设计细节，请参阅[此链接](https://docs.google.com/document/d/1rQXJwKd5b9b1aOzLh98mnyMhBMhlxXA5ATZTHoQrwvc/edit?tab=t.0)。

目前，我们支持 Mooncake 和 NIXL 作为传输引擎。

## 在 PD 分离模式下进行性能分析（Profiling）

当你需要在 PD 分离模式下对 prefill 或 decode worker 进行性能分析时，请参阅 Benchmark and Profiling 指南中的 [Profile In PD Disaggregation Mode](https://docs.sglang.io/developer_guide/benchmark_and_profiling.html#profile-in-pd-disaggregation-mode) 章节。由于 torch profiler 的限制，prefill 和 decode worker 必须使用专用的命令行选项分别进行性能分析。

## Router 集成

为了在大规模部署 PD 分离时实现负载均衡和容错，SGLang 提供了一个 router。该 router 可以使用多种路由策略在 prefill 实例和 decode 实例之间分发请求。有关如何为 PD 分离设置路由的详细信息（包括配置选项和部署模式），请参阅 [SGLang Model Gateway（前身为 Router）](../advanced_features/sgl_model_gateway.md#prefill-decode-disaggregation)。


## Mooncake
### 环境要求

```bash
uv pip install mooncake-transfer-engine
```

### 使用方法

### Llama 单节点

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --port 30000 \
  --disaggregation-ib-device mlx5_roce0
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --port 30001 \
  --base-gpu-id 1 \
  --disaggregation-ib-device mlx5_roce0
python -m sglang_router.launch_router --pd-disaggregation --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 --host 0.0.0.0 --port 8000
```

### DeepSeek 多节点

```bash
# prefill 0
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-ib-device ${device_name} \
  --disaggregation-mode prefill \
  --host ${local_ip} \
  --port 30000 \
  --trust-remote-code \
  --dist-init-addr ${prefill_master_ip}:5000 \
  --nnodes 2 \
  --node-rank 0 \
  --tp-size 16 \
  --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8
# prefill 1
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-ib-device ${device_name} \
  --disaggregation-mode prefill \
  --host ${local_ip} \
  --port 30000 \
  --trust-remote-code \
  --dist-init-addr ${prefill_master_ip}:5000 \
  --nnodes 2 \
  --node-rank 1 \
  --tp-size 16 \
  --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8
# decode 0
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-ib-device ${device_name} \
  --disaggregation-mode decode \
  --host ${local_ip} \
  --port 30001 \
  --trust-remote-code \
  --dist-init-addr ${decode_master_ip}:5000 \
  --nnodes 2 \
  --node-rank 0 \
  --tp-size 16 \
  --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8 \
  --max-running-requests 128
# decode 1
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-ib-device ${device_name} \
  --disaggregation-mode decode \
  --host ${local_ip} \
  --port 30001 \
  --trust-remote-code \
  --dist-init-addr ${decode_master_ip}:5000 \
  --nnodes 2 \
  --node-rank 1 \
  --tp-size 16 \
  --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8 \
  --max-running-requests 128
```
### 高级配置

使用 Mooncake 的 PD 分离支持以下环境变量，用于对系统行为进行细粒度控制。

#### NVLink 传输配置
要为 mooncake 后端启用 NVLink 传输以进行 KV 缓存传输（推荐用于 NVL72 部署），请设置以下环境变量。请注意，辅助数据传输仍将使用 TCP 作为临时的变通方案。

```bash
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK
export MC_FORCE_MNNVL=True
```

`SGLANG_MOONCAKE_CUSTOM_MEM_POOL` 环境变量用于启用自定义内存池。支持的取值为 `NVLINK`（或 `True`）、`BAREX` 和 `INTRA_NODE_NVLINK`。

#### Prefill 服务器配置
| 变量 | 描述 | 默认值 |
|:--------:|:-----------:|:--------:
| **`SGLANG_DISAGGREGATION_THREAD_POOL_SIZE`** | 控制每个 TP rank 用于 KVCache 传输操作的 worker 线程总数 | 一个由 `int(0.75 * os.cpu_count()) // 8)` 计算得出的动态值，该值被限制为大于 4 且小于 12，以确保效率并防止线程竞争 |
| **`SGLANG_DISAGGREGATION_QUEUE_SIZE`** | 设置并行传输队列的数量。来自多个 decode 实例的 KVCache 传输请求会被分片到这些队列中，从而可以同时共享线程和传输带宽。如果设置为 `1`，则按照 fcfs（先来先服务）策略逐个传输请求 | `4` |
| **`SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT`** | 请求初始化期间接收目标 KV 索引的超时时间（秒） | `300` |
| **`SGLANG_DISAGGREGATION_BOOTSTRAP_ENTRY_CLEANUP_INTERVAL`** | 清理 bootstrap 条目之间的间隔时间（秒） | `120` |

如果可以接受更大的平均 TTFT，你可以执行 `export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600`（10 分钟）以放宽超时条件。
请注意，此设置会导致当正在运行的 decode 节点失去连接时，prefill 实例需要更长的时间来清理受影响的内存资源。

#### Decode 服务器配置
| 变量 | 描述 | 默认值 |
|:--------:|:-----------:|:--------:
| **`SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL`** | 向 prefill bootstrap 服务器进行健康检查之间的间隔时间（秒） | `5.0` |
| **`SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE`** | 在将 prefill 服务器标记为离线之前允许的连续心跳失败次数 | `2` |
| **`SGLANG_DISAGGREGATION_WAITING_TIMEOUT`** | 请求初始化后接收 KV 缓存的超时时间（秒） | `300` |

如果可以接受更大的平均 TTFT，你可以执行 `export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600`（10 分钟）以放宽超时条件。


## NIXL
### 环境要求

通过 pip 安装。

```bash
pip install nixl
```

或者从源码构建——如果你已经安装了 UCX，可能需要采用这种方式。

```bash
git clone https://github.com/ai-dynamo/nixl.git
cd nixl
pip install . --config-settings=setup-args="-Ducx_path=/path/to/ucx"
```


### 使用方法

### Llama 单节点

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --port 30000 \
  --disaggregation-transfer-backend nixl
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --port 30001 \
  --base-gpu-id 1 \
  --disaggregation-transfer-backend nixl
python -m sglang_router.launch_router --pd-disaggregation --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 --host 0.0.0.0 --port 8000
```

### DeepSeek 多节点

```bash
# prefill 0
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-transfer-backend nixl \
  --disaggregation-mode prefill \
  --host ${local_ip} \
  --port 30000 \
  --trust-remote-code \
  --dist-init-addr ${prefill_master_ip}:5000 \
  --nnodes 2 \
  --node-rank 0 \
  --tp-size 16 \
  --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8
# prefill 1
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-transfer-backend nixl \
  --disaggregation-mode prefill \
  --host ${local_ip} \
  --port 30000 \
  --trust-remote-code \
  --dist-init-addr ${prefill_master_ip}:5000 \
  --nnodes 2 \
  --node-rank 1 \
  --tp-size 16 \
  --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8
# decode 0
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-transfer-backend nixl \
  --disaggregation-mode decode \
  --host ${local_ip} \
  --port 30001 \
  --trust-remote-code \
  --dist-init-addr ${decode_master_ip}:5000 \
  --nnodes 2 \
  --node-rank 0 \
  --tp-size 16 \
  --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8 \
  --max-running-requests 128
# decode 1
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-transfer-backend nixl \
  --disaggregation-mode decode \
  --host ${local_ip} \
  --port 30001 \
  --trust-remote-code \
  --dist-init-addr ${decode_master_ip}:5000 \
  --nnodes 2 \
  --node-rank 1 \
  --tp-size 16 \
  --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8 \
  --max-running-requests 128
```

### 高级配置

#### NIXL 后端选择

默认情况下，NIXL 使用 **UCX** 后端进行 KV 缓存传输。你可以通过环境变量 `SGLANG_DISAGGREGATION_NIXL_BACKEND` 根据你的基础设施选择不同的 NIXL 插件后端。

示例：`export SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC`

**可用后端：** UCX（默认）、LIBFABRIC，或任何已安装的 NIXL 插件。

使用示例：
```bash
export SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl \
  --port 30000
```

## ASCEND（昇腾）

### 使用方法

使用 ascend 后端时，需配合 [memfabric_hybrid](https://gitcode.com/Ascend/memfabric_hybrid) 并设置 ASCEND_MF_STORE_URL

```bash
pip install memfabric-hybrid==1.0.0
export ASCEND_MF_STORE_URL="tcp://xxx.xx.xxx.xxx:xxxx"
```
使用 mooncake 后端，更多细节可在 mooncake 章节中找到。
```bash
export ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE=true
```
需要在容器环境中设置 ASCEND_NPU_PHY_ID
```bash
export ASCEND_NPU_PHY_ID=xxx
```


### Llama 单节点

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --port 30000 \
  --disaggregation-transfer-backend ascend
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --port 30001 \
  --base-gpu-id 1 \
  --disaggregation-transfer-backend ascend
python -m sglang_router.launch_router --pd-disaggregation --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 --host 0.0.0.0 --port 8000
```

### DeepSeek 多节点

```bash
# prefill 0
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-transfer-backend ascend \
  --disaggregation-mode prefill \
  --host ${local_ip} \
  --port 30000 \
  --trust-remote-code \
  --dist-init-addr ${prefill_master_ip}:5000 \
  --nnodes 1 \
  --node-rank 0 \
  --tp-size 16
# decode 0
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-transfer-backend ascend \
  --disaggregation-mode decode \
  --host ${local_ip} \
  --port 30001 \
  --trust-remote-code \
  --dist-init-addr ${decode_master_ip}:5000 \
  --nnodes 1 \
  --node-rank 0 \
  --tp-size 16
```
