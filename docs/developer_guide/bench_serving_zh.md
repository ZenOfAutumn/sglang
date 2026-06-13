# Bench Serving 指南

本指南解释如何使用 `python -m sglang.bench_serving` 对在线服务的吞吐量和延迟进行 benchmark。它通过 OpenAI 兼容的和原生的端点支持多种推理后端，并产生控制台指标以及可选的 JSONL 输出。

### 它做什么

- 生成合成的或数据集驱动的 prompt 并将其提交到目标服务端点
- 测量吞吐量、time-to-first-token（TTFT）、inter-token latency（ITL）、每请求端到端延迟等
- 支持流式或非流式模式、速率控制和并发限制

### 支持的后端和端点

- `sglang` / `sglang-native`：`POST /generate`
- `sglang-oai`、`vllm`、`lmdeploy`：`POST /v1/completions`
- `sglang-oai-chat`、`vllm-chat`、`lmdeploy-chat`：`POST /v1/chat/completions`
- `trt`（TensorRT-LLM）：`POST /v2/models/ensemble/generate_stream`
- `gserver`：自定义服务器（此脚本中尚未实现）
- `truss`：`POST /v1/models/model:predict`

如果提供了 `--base-url`，请求将发送到它。否则使用 `--host` 和 `--port`。当未提供 `--model` 时，脚本将尝试查询 `GET /v1/models` 以获取一个可用的 model ID（OpenAI 兼容端点）。

### 前置条件

- Python 3.8+
- 此脚本通常使用的依赖：`aiohttp`、`numpy`、`requests`、`tqdm`、`transformers`，以及对某些数据集需要的 `datasets`、`pillow`、`pybase64`。按需安装。
- 一个正在运行且可通过上述端点访问的推理服务器
- 如果你的服务器需要认证，请设置环境变量 `OPENAI_API_KEY`（用作 `Authorization: Bearer <key>`）

### 快速开始

针对一个暴露 `/generate` 的 sglang 服务器运行一个基础 benchmark：

```bash
python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct
```

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --num-prompts 1000 \
  --model meta-llama/Llama-3.1-8B-Instruct
```

或者，使用 OpenAI 兼容端点（completions）：

```bash
python3 -m sglang.bench_serving \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --num-prompts 1000 \
  --model meta-llama/Llama-3.1-8B-Instruct
```

### 数据集

使用 `--dataset-name` 选择：

- `sharegpt`（默认）：加载 ShareGPT 风格的配对；可选地用 `--sharegpt-context-len` 限制，并用 `--sharegpt-output-len` 覆盖输出
- `random`：随机文本长度；从 ShareGPT token 空间采样
- `random-ids`：随机 token id（可能导致无意义内容）
- `image`：生成图像并将其包装在聊天消息中；支持自定义分辨率、多种格式和不同的内容类型
- `generated-shared-prefix`：带有共享长系统 prompt 和短问题的合成数据集
- `mmmu`：从 MMMU（Math split）采样并包含图像

常见的数据集标志：

- `--num-prompts N`：请求数量
- `--random-input-len`、`--random-output-len`、`--random-range-ratio`：用于 random/random-ids/image
- `--image-count`：每个请求的图像数量（用于 `image` 数据集）。

- `--apply-chat-template`：在构造 prompt 时应用 tokenizer 的 chat template
- `--dataset-path PATH`：ShareGPT json 的文件路径；如果为空且不存在，它将被下载并缓存

Generated Shared Prefix 标志（用于 `generated-shared-prefix`）：

- `--gsp-num-groups`
- `--gsp-prompts-per-group`
- `--gsp-system-prompt-len`
- `--gsp-question-len`
- `--gsp-output-len`

Image 数据集标志（用于 `image`）：

- `--image-count`：每个请求的图像数量
- `--image-resolution`：图像分辨率；支持预设（4k、1080p、720p、360p）或自定义的 'heightxwidth' 格式（例如 1080x1920、512x768）
- `--image-format`：图像格式（jpeg 或 png）
- `--image-content`：图像内容类型（random 或 blank）

### 示例

1. 要对每个请求 3 张图像、500 个 prompt、512 输入长度和 512 输出长度的 image 数据集进行 benchmark，你可以运行：

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-3B-Instruct --disable-radix-cache
```

```bash
python -m sglang.bench_serving \
    --backend sglang-oai-chat \
    --dataset-name image \
    --num-prompts 500 \
    --image-count 3 \
    --image-resolution 720p \
    --random-input-len 512 \
    --random-output-len 512
```

2. 要对 3000 个 prompt、1024 输入长度和 1024 输出长度的 random 数据集进行 benchmark，你可以运行：

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-3B-Instruct
```

```bash
python3 -m sglang.bench_serving \
    --backend sglang \
    --dataset-name random \
    --num-prompts 3000 \
    --random-input 1024 \
    --random-output 1024 \
    --random-range-ratio 0.5
```

### 选择模型和 tokenizer

- `--model` 是必需的，除非后端暴露了 `GET /v1/models`，在这种情况下会自动选择第一个 model ID。
- `--tokenizer` 默认为 `--model`。两者都可以是 HF model ID 或本地路径。
- 对于 ModelScope 工作流，设置 `SGLANG_USE_MODELSCOPE=true` 可启用通过 ModelScope 获取（为提速会跳过权重）。
- 如果你的 tokenizer 缺少 chat template，脚本会发出警告，因为对于无意义的输出，token 计数可能不太稳健。

### 速率、并发和流式

- `--request-rate`：每秒请求数。`inf` 立即发送所有请求（突发）。非无穷速率使用泊松过程来生成到达时间。
- `--max-concurrency`：无论到达速率如何，限制并发的在途请求数。
- `--disable-stream`：在支持时切换到非流式模式；此时对于 chat completions，TTFT 等于总延迟。

### 其他关键选项

- `--output-file FILE.jsonl`：将 JSONL 结果追加到文件；如果未指定则自动命名
- `--output-details`：包含每请求的数组（生成的文本、错误、ttft、itl、输入/输出长度）
- `--extra-request-body '{"top_p":0.9,"temperature":0.6}'`：合并到 payload 中（采样参数等）
- `--disable-ignore-eos`：透传 EOS 行为（因后端而异）
- `--warmup-requests N`：先用短输出运行预热请求（默认 1）
- `--flush-cache`：在主运行之前调用 `/flush_cache`（sglang）
- `--profile`：调用 `/start_profile` 和 `/stop_profile`（需要服务器启用 profiling，例如 `SGLANG_TORCH_PROFILER_DIR`）
- `--lora-name name1 name2 ...`：每个请求随机选取一个并传递给后端（例如，sglang 的 `lora_path`）
- `--tokenize-prompt`：发送整数 ID 而非文本（当前仅支持 `--backend sglang`）

### 认证

如果你的目标端点需要 OpenAI 风格的认证，请设置：

```bash
export OPENAI_API_KEY=sk-...yourkey...
```

脚本将为 OpenAI 兼容路由自动添加 `Authorization: Bearer $OPENAI_API_KEY`。

### 指标说明

每次运行后打印：

- Request throughput（req/s）
- Input token throughput（tok/s）- 包括文本和视觉 token
- Output token throughput（tok/s）
- Total token throughput（tok/s）- 包括文本和视觉 token
- Total input text tokens 和 Total input vision tokens - 按模态分解
- Concurrency：所有请求的总时间除以墙钟时间
- End-to-End Latency（ms）：每请求总延迟的 mean/median/std/p99
- Time to First Token（TTFT，ms）：流式模式下的 mean/median/std/p99
- Inter-Token Latency（ITL，ms）：token 之间的 mean/median/std/p95/p99/max
- TPOT（ms）：第一个 token 之后的 token 处理时间，即 `(latency - ttft)/(tokens-1)`
- Accept length（仅 sglang，如果可用）：投机解码（speculative decoding）的接受长度

脚本还使用配置的 tokenizer 对生成的文本进行重新分词，并报告 "retokenized" 计数。

### JSONL 输出格式

当设置了 `--output-file` 时，每次运行追加一个 JSON 对象。基础字段：

- 参数摘要：backend、dataset、request_rate、max_concurrency 等
- 持续时间和总计：completed、total_input_tokens、total_output_tokens、retokenized 总数
- 控制台中打印的吞吐量和延迟统计
- `accept_length`（如果可用，sglang）

使用 `--output-details` 时，扩展对象还包括数组：

- `input_lens`、`output_lens`
- `ttfts`、`itls`（每请求：ITL 数组）
- `generated_texts`、`errors`

### 端到端示例

1) sglang 原生 `/generate`（流式）：

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 1024 --random-range-ratio 0.5 \
  --num-prompts 2000 \
  --request-rate 100 \
  --max-concurrency 512 \
  --output-file sglang_random.jsonl --output-details
```

2) OpenAI 兼容的 Completions（例如 vLLM）：

```bash
python3 -m sglang.bench_serving \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name sharegpt \
  --num-prompts 1000 \
  --sharegpt-output-len 256
```

3) OpenAI 兼容的 Chat Completions（流式）：

```bash
python3 -m sglang.bench_serving \
  --backend vllm-chat \
  --base-url http://127.0.0.1:8000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --num-prompts 500 \
  --apply-chat-template
```

4) 带 chat template 的图像（VLM）：

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model your-vlm-model \
  --dataset-name image \
  --image-count 2 \
  --image-resolution 720p \
  --random-input-len 128 --random-output-len 256 \
  --num-prompts 200 \
  --apply-chat-template
```

4a) 带自定义分辨率的图像：

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model your-vlm-model \
  --dataset-name image \
  --image-count 1 \
  --image-resolution 512x768 \
  --random-input-len 64 --random-output-len 128 \
  --num-prompts 100 \
  --apply-chat-template
```

4b) 采用 PNG 格式和 blank 内容的 1080p 图像：

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model your-vlm-model \
  --dataset-name image \
  --image-count 1 \
  --image-resolution 1080p \
  --image-format png \
  --image-content blank \
  --random-input-len 64 --random-output-len 128 \
  --num-prompts 100 \
  --apply-chat-template
```

5) Generated shared prefix（长系统 prompt + 短问题）：

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 64 --gsp-prompts-per-group 16 \
  --gsp-system-prompt-len 2048 --gsp-question-len 128 --gsp-output-len 256 \
  --num-prompts 1024
```

6) 用于严格长度控制的分词后 prompt（ids）（仅 sglang）：

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --tokenize-prompt \
  --random-input-len 2048 --random-output-len 256 --random-range-ratio 0.2
```

7) Profiling 和缓存刷新（sglang）：

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --profile \
  --flush-cache
```

8) TensorRT-LLM 流式端点：

```bash
python3 -m sglang.bench_serving \
  --backend trt \
  --base-url http://127.0.0.1:8000 \
  --model your-trt-llm-model \
  --dataset-name random \
  --num-prompts 100 \
  --disable-ignore-eos
```

9) 使用 mooncake trace 评估大规模 KVCache 共享（仅 sglang）：

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model model-name \
  --dataset-name mooncake \
  --mooncake-slowdown-factor 1.0 \
  --mooncake-num-rounds 1000 \
  --mooncake-workload conversation|mooncake|agent|synthetic
  --use-trace-timestamps true \
  --random-output-len 256
```

### 故障排查

- 所有请求都失败：验证 `--backend`、服务器 URL/端口、`--model` 和认证。检查脚本打印的预热错误。
- 吞吐量看起来太低：调整 `--request-rate` 和 `--max-concurrency`；验证服务器的批大小/调度；如果合适，确保启用了流式。
- token 计数看起来异常：优先使用带有正确 chat template 的 chat/instruct 模型；否则对无意义内容的分词可能不一致。
- Image/MMMU 数据集：确保你已安装额外的依赖（`pillow`、`datasets`、`pybase64`）。
- 认证错误（401/403）：设置 `OPENAI_API_KEY` 或在你的服务器上禁用认证。

### 注意事项

- 脚本会提高文件描述符软限制（`RLIMIT_NOFILE`），以帮助处理大量并发连接。
- 对于 sglang，运行后会查询 `/server_info` 以在可用时报告投机解码的接受长度。
