# Benchmark 与 Profiling

## Benchmark

SGLang 提供了四种在技术栈不同层级运行的 benchmark 工具。下表总结了它们的关键差异：

| 工具                       | HTTP Server                                   | Scheduler                               | 使用场景                                                                   |
| -------------------------- | --------------------------------------------- | --------------------------------------- | -------------------------------------------------------------------------- |
| `bench_serving`            | 是（向运行中的服务器发送异步 HTTP 客户端请求）   | 是（间接地，通过服务器）            | 带延迟指标（TTFT、TPOT、ITL）的真实在线服务 benchmark |
| `bench_one_batch_server`   | 是（向运行中的服务器发送 HTTP 请求） | 是（间接地，通过服务器）            | 包括 HTTP 和 scheduler 开销的端到端单批次延迟      |
| `bench_offline_throughput` | 否                                            | 是（在进程内直接使用 `Engine`） | 无 HTTP 开销的最大吞吐量测量                       |
| `bench_one_batch`          | 否                                            | 否（直接调用 `ModelRunner`）       | 对单个静态批次进行 kernel 级别的延迟 profiling                    |

默认情况下请使用 `bench_serving`，除非有特定需求。

**`bench_serving`** 是一个异步 HTTP 负载测试客户端，它以可控的速率和可配置的并发向运行中的服务器发送请求。它测量真实的在线服务指标，包括 time-to-first-token（TTFT）、time-per-output-token（TPOT）、inter-token latency（ITL）和吞吐量。使用 `num-prompts >= 5 * max-concurrency` 来测量稳态性能。先用 `sglang.launch_server` 启动一个服务器。

  ```bash
  python3 -m sglang.bench_serving --backend sglang --max-concurrency 16 --num-prompts 80 --random-input-len 256 --random-output-len 32 --dataset-name random
  ```

**`bench_one_batch_server`** 将单个批次作为一个 HTTP 请求发送到运行中的服务器。由于只有单个批次，服务器永远不会处于稳态，指标会有偏差。先用 `sglang.launch_server` 启动一个服务器。

  ```bash
  python3 -m sglang.bench_one_batch_server --base-url http://127.0.0.1:30000 --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --batch-size 32 --input-len 256 --output-len 32
  ```

**`bench_offline_throughput`** 在进程内直接实例化 `Engine` 对象（无 HTTP 服务器），并通过 `engine.generate()` 一次性提交所有请求。引擎的 scheduler 处理批处理和执行。这测量了无任何网络开销下可达到的最大吞吐量。

  ```bash
  python3 -m sglang.bench_offline_throughput --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --num-prompts 10
  ```

**`bench_one_batch`** 是最底层的工具。它直接实例化一个 `ModelRunner`，并在一个固定的静态批次上调用 `extend()` / `decode()`，完全绕过 scheduler。prefill 和 decode 阶段分开运行，使 profiling 更容易，但使得指标不真实。由于没有动态批处理，对于真实服务器能够处理的批大小，它可能会内存不足（真实服务器会将 prefill 分块为更小的批次）。这最适合对单个 kernel 性能进行 profiling。

  ```bash
  python3 -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --batch-size 32 --input-len 256 --output-len 32
  ```

## 使用 PyTorch Profiler 进行 Profiling

[Pytorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) 是一个方便的基础工具，用于检查 kernel 执行时间、调用栈以及 kernel 的重叠与占用率。

### 使用 `sglang.bench_serving` 对服务器进行 profiling

```bash
# set trace path
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log

# start server
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct

# send profiling request from client
python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 10 --sharegpt-output-len 100 --profile
```

`SGLANG_TORCH_PROFILER_DIR` 环境变量必须在服务器端和客户端**都**设置；否则，trace 文件将无法正确生成。一种安全的做法是在你的 shell 资源文件中设置它（例如 bash 的 `~/.bashrc`）。

更多详情，请参阅 [Bench Serving Guide](./bench_serving.md)。

### 在 PD Disaggregation 模式下进行 profiling

在 PD disaggregation 模式下进行 profiling 时，由于 torch profiler 的限制，prefill 和 decode worker **必须分别 profiling**。`bench_serving` 命令为此提供了专门的选项：

#### 对 Prefill Worker 进行 profiling

```bash
# set trace path
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log

# start prefill and decode servers (see PD disaggregation docs for setup)
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --disaggregation-mode prefill
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --disaggregation-mode decode --port 30001 --base-gpu-id 1

# start router
python -m sglang_router.launch_router --pd-disaggregation --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 --host 0.0.0.0 --port 8000

# send profiling request targeting prefill workers
python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 10 --sharegpt-output-len 100 --profile --pd-separated --profile-prefill-url http://127.0.0.1:30000
```

#### 对 Decode Worker 进行 profiling

```bash
# send profiling request targeting decode workers
python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 10 --sharegpt-output-len 100 --profile --pd-separated --profile-decode-url http://127.0.0.1:30001
```

#### 重要说明

- `--profile-prefill-url` 和 `--profile-decode-url` 是**互斥的** - 你不能同时对两者进行 profiling
- 这两个选项都支持多个 worker URL 以用于多实例设置：
  ```bash
  # Profile multiple prefill workers
  python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 10 --profile --pd-separated --profile-prefill-url http://127.0.0.1:30000 http://127.0.0.1:30002

  # Profile multiple decode workers
  python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 10 --profile --pd-separated --profile-decode-url http://127.0.0.1:30001 http://127.0.0.1:30003
  ```
- 在启动服务器之前，确保在所有 worker 节点上都设置了 `SGLANG_TORCH_PROFILER_DIR`
- 关于设置 PD disaggregation 的更多详情，请参阅 [PD Disaggregation Guide](../advanced_features/pd_disaggregation.md)

### 使用 `sglang.bench_offline_throughput` 对服务器进行 profiling
```bash
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log

# profile one batch with bench_one_batch.py
# batch size can be controlled with --batch argument
python3 -m sglang.bench_one_batch --model-path meta-llama/Llama-3.1-8B-Instruct --batch 32 --input-len 1024 --output-len 10 --profile

# profile multiple batches with bench_offline_throughput.py
python -m sglang.bench_offline_throughput --model-path meta-llama/Llama-3.1-8B-Instruct --dataset-name random --num-prompts 10 --profile --mem-frac=0.8
```

### 使用 `sglang.profiler` 对服务器进行 profiling

当服务器正在运行时（例如，处理一个 decoding 请求），你可以通过向服务器发送一个 profile 请求来立即开始实时 profiling。

你可以通过运行 `python3 -m sglang.profiler` 来实现。例如：

```
# Terminal 1: Send a generation request
python3 -m sglang.test.send_one

# Terminal 2: Before the above request finishes, quickly launch the following command in a separate terminal.
# It will generate a profile of the above request for several decoding batches.
python3 -m sglang.profiler
```

你也可以将上述操作合并为单个命令

```
python3 -m sglang.test.send_one --profile
```

### 使用 HTTP API 端点对服务器进行 profiling

SGLang 提供 HTTP API 端点来控制运行中服务器上的 profiling。这允许你以编程方式启动和停止 profiling，这对于捕获特定的工作负载模式很有用。

#### 使用 `/start_profile` 端点

`/start_profile` 端点在服务器上启动 profiling。你可以使用以下参数控制 profiling 何时开始以及运行多长时间：

**基本用法：**

```bash
# Start profiling immediately for 10 steps
curl -X POST http://127.0.0.1:30000/start_profile \
  -H "Content-Type: application/json" \
  -d '{
    "num_steps": 10
  }'
```

**参数：**

- `output_dir`（可选）：保存 profile trace 的目录。如果未指定，则使用 `SGLANG_TORCH_PROFILER_DIR` 环境变量，或默认使用 `/tmp`
- `num_steps`（可选）：要 profiling 的步数。如果未指定，profiling 将持续到用 `/end_profile` 手动停止为止
- `start_step`（可选）：开始 profiling 的步数（含）。对于跳过预热迭代很有用
- `activities`（可选）：要 profiling 的活动列表，例如 `["CPU", "GPU"]`。默认是 `["CPU", "GPU"]`
- `merge_profiles`（可选）：是否合并分布式 trace。默认是 `false`

**关于步数范围的说明：** profiling 从 `start_step`（含）开始，并持续 `num_steps` 次迭代。例如，对于 `start_step=3` 和 `num_steps=10`，profiling 捕获步数 3、4、5、6、7、8、9、10、11 和 12（总共 10 步，从第 3 步开始）。

**使用 `start_step` 的高级用法：**

```bash
# Wait 5 steps (warmup), then profile for 10 steps
curl -X POST http://127.0.0.1:30000/start_profile \
  -H "Content-Type: application/json" \
  -d '{
    "output_dir": "/tmp/profiles",
    "start_step": 5,
    "num_steps": 10,
    "activities": ["CPU", "GPU"]
  }'
```

**连续 profiling（手动停止）：**

```bash
# Start profiling without num_steps - must manually stop with /end_profile
curl -X POST http://127.0.0.1:30000/start_profile
```

#### 使用 `/end_profile` 端点

`/end_profile` 端点停止正在进行的 profiling 会话并保存 trace 文件。

```bash
# Stop profiling and save traces
curl -X POST http://127.0.0.1:30000/end_profile
```

仅当你在不指定 `num_steps` 的情况下启动 profiling 时才需要这样做。如果指定了 `num_steps`，profiling 将在那么多步之后自动停止。

#### 示例工作流

```bash
# Terminal 1: Start the server
export SGLANG_TORCH_PROFILER_DIR=/tmp/profiles
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct

# Terminal 2: Start continuous profiling
curl -X POST http://127.0.0.1:30000/start_profile \
  -H "Content-Type: application/json" \
  -d '{
    "start_step": 3
  }'

# Terminal 3: Send requests to generate load
python -m sglang.bench_serving --backend sglang --num-prompts 100

# Terminal 2: Stop profiling when done
curl -X POST http://127.0.0.1:30000/end_profile
```

### 用于分布式 trace 的 Profiler Trace Merger

SGLang 现在支持从具有多种并行类型（TP、DP、PP、EP）的分布式设置中自动合并 profiling trace。此功能对于分析跨分布式运行的性能特别有用。

#### 多节点 Profiling 和共享存储注意事项

完全支持单节点 profiler 输出合并。当在跨多个节点的分布式环境中进行 profiling 时，输出目录应可被所有节点访问的共享存储（例如 NFS、Lustre），以启用 trace 文件的合并。

如果没有可跨节点访问的共享存储，目前不直接支持在 profiling 期间自动合并 trace 文件。

#### HTTP API 用法

```bash
# Start profiling with automatic trace merging enabled
curl -X POST <BASE_URL>/start_profile \
  -H "Content-Type: application/json" \
  -d '{
    "output_dir": "/tmp/profiles", # where to store profile traces
    "num_steps": 10,
    "activities": ["CPU", "GPU"],
    "merge_profiles": true # optional argument to merge profile traces (default=False)
  }'
```

#### 命令行用法

```bash
# Start profiling with merge enabled
python -m sglang.profiler \
  --num-steps 10 \
  --cpu \
  --gpu \
  --output-dir /tmp/profiles \
  --merge-profiles # optional argument to merge profile traces (default=False)
```

#### 输出文件

profile merger 生成：
- 各个 rank 的 trace 文件：`{profile_id}-TP-{tp}-DP-{dp}-PP-{pp}-EP-{ep}.trace.json.gz`
- 合并后的 trace 文件：`merged-{profile_id}.trace.json.gz`

### 可能的 PyTorch bug
如果在任何情况下你遇到以下错误（例如，使用 qwen 2.5 VL 时）：
```bash
RuntimeError: !stack.empty() INTERNAL ASSERT FAILED at "/pytorch/torch/csrc/autograd/profiler_python.cpp":983, please report a bug to PyTorch. Python replay stack is empty.
```
这很可能是在 [Bug: vLLM Profiler](https://github.com/vllm-project/vllm/issues/18240) 和 [Bug: torch.profiler.profile](https://github.com/pytorch/pytorch/issues/101632) 中报告的 PyTorch Bug。作为一种变通方案，你可以通过如下环境变量禁用 `with_stack`：
```bash
export SGLANG_PROFILE_WITH_STACK=False
python -m sglang.bench_offline_throughput --model-path meta-llama/Llama-3.1-8B-Instruct --dataset-name random --num-prompts 10 --profile --mem-frac=0.8
```

### 查看 trace

Trace 文件可以从以下位置加载和可视化：

1. https://ui.perfetto.dev/ （任何浏览器）
2. chrome://tracing （仅 Chrome 浏览器）

如果浏览器因 trace 文件过大而无法打开，
客户端可以通过控制 prompt 的数量和 prompt 输出的长度来生成一个小的 trace 文件（<100MB）。
例如，在对服务器进行 profiling 时，

```bash
python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 2 --sharegpt-output-len 100 --profile
```

此命令使用 `--num-prompts` 参数将 prompt 数量设置为 2，并使用 `--sharegpt-output-len` 参数将输出序列的长度限制为 100，这可以生成一个小的 trace 文件，让浏览器能够流畅打开。

此外，如果你想通过 Trace 中的 cuda kernel 定位到 SGLang Python 源代码，你需要在启动服务时禁用 CUDA Graph。这可以通过在启动服务的命令中使用 `--disable-cuda-graph` 参数来完成。

## 使用 Nsight 进行 Profiling

[Nsight systems](https://docs.nvidia.com/nsight-systems/) 是一个高级工具，它暴露了更多的 profiling 细节，例如寄存器和共享内存使用情况、带注释的代码区域以及底层的 CUDA API 和事件。

1. 前置条件：

   使用 apt 安装，或在 [NVIDIA Docker container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags) 或 [SGLang Docker container](https://github.com/sgl-project/sglang/tree/main/docker) 内运行。

   ```bash
   # install nsys
   # https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html
   apt update
   apt install -y --no-install-recommends gnupg
   echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
   apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
   apt update
   apt install nsight-systems-cli
   ```

2. 要对单个批次进行 profiling，使用

   ```bash
   nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node python3 -m sglang.bench_one_batch --model meta-llama/Meta-Llama-3-8B --batch-size 64 --input-len 512
   ```

3. 要对服务器进行 profiling，例如

   ```bash
   # launch the server, set the delay and duration times according to needs
   # after the duration time has been used up, server will be killed by nsys

   nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node -o sglang.out --delay 60 --duration 70 python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --disable-radix-cache

   # client
   python3 -m sglang.bench_serving --backend sglang --num-prompts 1000 --dataset-name random --random-input 1024 --random-output 512
   ```

   在实践中，我们建议用户将 `--duration` 参数设置为一个较大的值。每当用户希望服务器停止 profiling 时。首先运行：

   ```bash
   nsys sessions list
   ```

   以获取形如 `profile-XXXXX` 的 session id，然后运行：

   ```bash
   nsys stop --session=profile-XXXXX
   ```

   来手动终止 profiler 并立即生成 `nsys-rep` 文件。

4. 使用 NVTX 注释代码区域，例如查看它们的执行时间。

   ```bash
   # install nvtx
   pip install nvtx
   ```

   ```python
   # code snippets
   import nvtx
   with nvtx.annotate("description", color="color"):
       # some critical code
   ```

### 使用 Nsight Systems 进行逐层（Layer-wise）NVTX Profiling

SGLang 提供了内置的逐层 NVTX 注释，可以与 CUDA Profiler 结合使用，在 Nsight Systems 中进行详细的逐层 profiling。这对于在层级别识别性能瓶颈特别有用。

#### 将 `--enable-layerwise-nvtx-marker` 与 Nsight Systems 和 `/start_profile` 结合使用

`--enable-layerwise-nvtx-marker` 标志会自动为你模型中的每一层添加 NVTX marker。当与 Nsight Systems profiling 结合使用以查看详细的逐层性能时，这特别强大。

**方法 1：将 `/start_profile` 与 CUDA_PROFILER 结合使用（用于编程控制）**

此方法允许你在 Nsight Systems 运行时通过 HTTP API 精确控制 profiling 何时启动/停止。

1. 在 Nsight Systems 下启动启用逐层 NVTX 的服务器：

   ```bash
   # Terminal 1: Start server with nsys and capture-range option
   nsys profile --trace-fork-before-exec=true \
     --cuda-graph-trace=node \
     --capture-range=cudaProfilerApi \
     --capture-range-end=stop \
     -o layerwise_profile \
     python -m sglang.launch_server \
       --model-path meta-llama/Llama-3.1-8B-Instruct \
       --enable-layerwise-nvtx-marker \
       --disable-cuda-graph
   ```

   注意：对于被 CUDA graph 捕获的 kernel 启动，不会发出 NVTX marker。使用 `--disable-cuda-graph` 以确保所有逐层 NVTX marker 都被发出到 trace 中。

2. 在另一个终端中，通过带有 `CUDA_PROFILER` 活动的 `/start_profile` 控制 profiling：

   ```bash
   # Terminal 2: Wait for server to be ready, then start CUDA profiling
   # Wait 3 steps for warmup, then profile for 10 steps
   curl -X POST http://127.0.0.1:30000/start_profile \
     -H "Content-Type: application/json" \
     -d '{
       "start_step": 3,
       "num_steps": 10,
       "activities": ["CUDA_PROFILER"]
     }'
   ```

3. 发送请求以产生负载：

   ```bash
   # Terminal 3: Generate workload
   python -m sglang.bench_serving --backend sglang --num-prompts 100
   ```

4. profiling 将在 10 步之后自动停止（由于 `num_steps: 10`）。如果你没有指定 `num_steps`，则需要手动停止它：

   ```bash
   # Terminal 2: Only needed if num_steps was not specified
   curl -X POST http://127.0.0.1:30000/end_profile
   ```

`--capture-range=cudaProfilerApi` 选项告诉 Nsight Systems 只捕获 `cudaProfilerStart()` 和 `cudaProfilerStop()` 调用之间的数据（由 `/start_profile` 和 `/end_profile` 触发），从而减少开销和文件大小。`start_step` 参数跳过前 3 步以避免捕获预热开销。

**方法 2：不使用 `/start_profile` API 的更简单方法**

对于不需要对 profiling 启动/停止进行细粒度控制的更简单场景，你可以用 Nsight Systems 对整个工作负载进行 profiling：

```bash
# Terminal 1: Start server with layerwise NVTX
# Note: --disable-cuda-graph ensures all NVTX markers are emitted
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --enable-layerwise-nvtx-marker \
  --disable-cuda-graph

# Terminal 2: Profile the benchmarking client
nsys profile --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  -o layerwise_profile \
  python -m sglang.bench_serving --backend sglang --num-prompts 10
```

此方法对整个客户端执行进行 profiling，包括所有的服务器交互。逐层 NVTX marker 将在 Nsight Systems 时间线中可见。

**查看 profiling 结果：**

用 Nsight Systems 打开生成的 `.qdrep` 文件：

```bash
nsys-ui layerwise_profile.qdrep
```

在 Nsight Systems GUI 中，你将看到：
- **NVTX ranges**：每一层在时间线中显示为一个带标签的范围，并在 marker 元数据中包含详细信息
- **CUDA kernels**：所有 GPU kernel 都与层注释一起显示
- **Layer hierarchy**：完整的模块路径（例如 `meta-llama/Meta-Llama-3.1-8B-Instruct.model.layers.0.self_attn.qkv_proj`）有助于识别特定的层。该前缀使用来自 `--model-path` 的完整模型路径。
- **Tensor shapes**：输入/输出维度和参数形状包含在 NVTX marker 数据中

**逐层 NVTX profiling 的好处：**

- **细粒度可见性**：精确查看哪些层花费的时间最多
- **内存跟踪**：识别具有大量内存分配的层
- **瓶颈识别**：快速定位低效的操作
- **通信开销**：在多 GPU 设置中，查看每层的通信成本
- **开发调试**：验证模型架构更改是否具有预期的性能影响

## 其他提示

1. 你可以仅提供 config.json 文件来使用 dummy 权重对模型进行 benchmark。这允许在不训练的情况下快速测试模型变体。要做到这一点，在上述命令中添加 `--load-format dummy`，然后你只需要在 checkpoint 文件夹下有一个正确的 `config.json`。
2. 你可以使用 `--json-model-override-args` 对修改了配置（例如更少的层）的模型进行 benchmark。例如，你可以使用以下命令对一个只有 2 层和 2 个 kv head 的模型进行 benchmark：

   ```bash
   python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --batch 32 --input-len 256 --output-len 32 --load-format dummy --json-model-override-args '{"num_hidden_layers": 1, "num_key_value_heads": 1}'
   ```

3. 你可以使用 `--python-backtrace=cuda` 来查看所有 CUDA kernel 的 python 调用栈，就像在 PyTorch Profiler 中一样。（注意事项：这可能会导致基于 CUDA event 的计时出现不准确的过长 kernel 运行时间）
4. 更多参数请参阅 [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)。
