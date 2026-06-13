# Qwen3-VL 使用指南

[Qwen3-VL](https://huggingface.co/collections/Qwen/qwen3-vl)
是阿里巴巴最新的多模态大语言模型，具有强大的文本、视觉和推理能力。
SGLang 支持 Qwen3-VL 系列模型，并支持图像和视频输入。

## 使用 SGLang 的启动命令

以下是针对不同硬件 / 精度模式量身定制的建议启动命令

### FP8（量化）模式
适用于支持 FP8 checkpoint 的高显存效率和低延迟优化部署（例如在 H100、H200 上）：
```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --tp 8 \
  --ep 8 \
  --host 0.0.0.0 \
  --port 30000 \
  --keep-mm-feature-on-device
```

### 非 FP8（BF16 / 全精度）模式
适用于使用 BF16 的部署（或未使用 FP8 快照）的 A100/H100：
```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-235B-A22B-Instruct \
  --tp 8 \
  --ep 8 \
  --host 0.0.0.0 \
  --port 30000 \
```

## 硬件相关说明 / 建议

- 在使用 FP8 的 H100 上：使用 FP8 checkpoint 以获得最佳显存效率。
- 在使用 BF16（非 FP8）的 A100 / H100 上：建议使用 `--mm-max-concurrent-calls` 来控制图像/视频推理期间的并行吞吐量和 GPU 显存占用。
- 在 H200 和 B200 上：模型可以"开箱即用"地运行，支持完整上下文长度以及并发的图像 + 视频处理。

## 发送图像/视频请求

### 图像输入：

```python
import requests

url = f"http://localhost:30000/v1/chat/completions"

data = {
    "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true"
                    },
                },
            ],
        }
    ],
    "max_tokens": 300,
}

response = requests.post(url, json=data)
print(response.text)
```

### 视频输入：

```python
import requests

url = f"http://localhost:30000/v1/chat/completions"

data = {
    "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s happening in this video?"},
                {
                    "type": "video_url",
                    "video_url": {
                        "url": "https://github.com/sgl-project/sgl-test-files/raw/refs/heads/main/videos/jobs_presenting_ipod.mp4"
                    },
                },
            ],
        }
    ],
    "max_tokens": 300,
}

response = requests.post(url, json=data)
print(response.text)
```

## 重要的服务器参数和标志

在为**多模态支持**启动模型服务器时，你可以使用以下命令行参数来微调性能和行为：

- `--mm-attention-backend`：指定多模态注意力后端。例如 `fa3`（Flash Attention 3）
- `--mm-max-concurrent-calls <value>`：指定服务器上允许的**最大并发异步多模态数据处理调用数**。用它来控制图像/视频推理期间的并行吞吐量和 GPU 显存占用。
- `--mm-per-request-timeout <seconds>`：定义每个多模态请求的**超时时长（单位：秒）**。如果一个请求超过此时间限制（例如对于非常大的视频输入），它将被自动终止。
- `--keep-mm-feature-on-device`：指示服务器在处理后将**多模态特征张量保留在 GPU 上**。这避免了设备到主机（D2H）的内存拷贝，并能为重复或高频推理工作负载提升性能。
- `SGLANG_USE_CUDA_IPC_TRANSPORT=1`：基于共享内存池的 CUDA IPC，用于多模态数据传输。可显著改善端到端延迟。

### 结合上述优化的示例用法：
```bash
SGLANG_USE_CUDA_IPC_TRANSPORT=1 \
SGLANG_VLM_CACHE_SIZE_MB=0 \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-235B-A22B-Instruct \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  --tp-size 8 \
  --enable-cache-report \
  --log-level info \
  --max-running-requests 64 \
  --mem-fraction-static 0.65 \
  --chunked-prefill-size 8192 \
  --attention-backend fa3 \
  --mm-attention-backend fa3 \
  --enable-metrics
```
