# Ollama 兼容 API

SGLang 提供 Ollama API 兼容性，允许你以 SGLang 作为推理后端来使用 Ollama CLI 和 Python 库。

## 前置条件

```bash
# 安装 Ollama Python 库（用于 Python 客户端）
pip install ollama
```

> **注意**：你不需要安装 Ollama 服务器——SGLang 充当后端。你只需要 `ollama` CLI 或 Python 库作为客户端。

## 端点（Endpoints）

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET, HEAD | 用于 Ollama CLI 的健康检查 |
| `/api/tags` | GET | 列出可用模型 |
| `/api/chat` | POST | 聊天补全（流式与非流式） |
| `/api/generate` | POST | 文本生成（流式与非流式） |
| `/api/show` | POST | 模型信息 |

## 快速开始

### 1. 启动 SGLang 服务器

```bash
python -m sglang.launch_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --port 30001 \
    --host 0.0.0.0
```

> **注意**：与 `ollama run` 一起使用的模型名称必须与你传给 `--model` 的完全一致。

### 2. 使用 Ollama CLI

```bash
# 列出可用模型
OLLAMA_HOST=http://localhost:30001 ollama list

# 交互式聊天
OLLAMA_HOST=http://localhost:30001 ollama run "Qwen/Qwen2.5-1.5B-Instruct"
```

如果要连接到防火墙后的远程服务器：

```bash
# SSH 隧道
ssh -L 30001:localhost:30001 user@gpu-server -N &

# 然后按上文方式使用 Ollama CLI
OLLAMA_HOST=http://localhost:30001 ollama list
```

### 3. 使用 Ollama Python 库

```python
import ollama

client = ollama.Client(host='http://localhost:30001')

# 非流式
response = client.chat(
    model='Qwen/Qwen2.5-1.5B-Instruct',
    messages=[{'role': 'user', 'content': 'Hello!'}]
)
print(response['message']['content'])

# 流式
stream = client.chat(
    model='Qwen/Qwen2.5-1.5B-Instruct',
    messages=[{'role': 'user', 'content': 'Tell me a story'}],
    stream=True
)
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
```

## 智能路由（Smart Router）

如需使用 LLM 评判器在本地 Ollama（快速）和远程 SGLang（强大）之间进行智能路由，请参阅[智能路由文档](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/ollama/README.md)。

## 小结

| Component | Purpose |
|-----------|---------|
| **Ollama API** | 开发者已经熟悉的 CLI/API |
| **SGLang Backend** | 高性能推理引擎 |
| **Smart Router** | 智能路由——简单任务用快速的本地端，复杂任务用强大的远程端 |
