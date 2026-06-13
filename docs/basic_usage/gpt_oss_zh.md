# GPT OSS 使用指南

请参阅 [https://github.com/sgl-project/sglang/issues/8833](https://github.com/sgl-project/sglang/issues/8833)。

## Responses API 与内置工具

### Responses API

GPT‑OSS 兼容 OpenAI Responses API。使用 `client.responses.create(...)`，配合 `model`、`instructions`、`input` 以及可选的 `tools` 来启用内置工具的使用。你可以通过 `instructions` 设置推理级别，例如 "Reasoning: high"（也支持 "medium" 和 "low"）——各级别：low（快速）、medium（平衡）、high（深度）。

### 内置工具

GPT‑OSS 可以调用内置工具进行网络搜索和 Python 执行。你可以使用演示用的工具服务器，或连接到外部的 MCP 工具服务器。

#### Python 工具

- 执行简短的 Python 片段，用于计算、解析和快速脚本。
- 默认在基于 Docker 的沙箱中运行。若要在宿主机上运行，请设置 `PYTHON_EXECUTION_BACKEND=UV`（这会在本地执行模型生成的代码；请谨慎使用）。
- 如果你不使用 UV 后端，请确保 Docker 可用。建议提前运行 `docker pull python:3.11`。

#### 网络搜索工具

- 使用 Exa 后端进行网络搜索。
- 需要 Exa API key；在你的环境中设置 `EXA_API_KEY`。在 `https://exa.ai` 创建 key。

### 工具与推理解析器

- 我们支持 OpenAI Reasoning 和 Tool Call 解析器，以及我们的 SGLang 原生 API 用于工具调用和推理。更多详情请参阅 [reasoning parser](../advanced_features/separate_reasoning.ipynb) 和 [tool parser](../advanced_features/tool_parser.ipynb)。


## 注意事项

- 演示工具请使用 **Python 3.12**。并安装所需的 `gpt-oss` 包。
- 默认演示集成了网络搜索工具（Exa 后端）和一个通过 Docker 运行的演示 Python 解释器。
- 对于搜索，请设置 `EXA_API_KEY`。对于 Python 执行，要么确保 Docker 可用，要么设置 `PYTHON_EXECUTION_BACKEND=UV`。

示例：
```bash
export EXA_API_KEY=YOUR_EXA_KEY
# 可选：在本地而非 Docker 中运行 Python 工具（请谨慎使用）
export PYTHON_EXECUTION_BACKEND=UV
```

使用演示工具服务器启动服务器：

```bash
python3 -m sglang.launch_server \
  --model-path openai/gpt-oss-120b \
  --tool-server demo \
  --tp 2
```

在生产环境中，sglang 可以作为多个服务的 MCP 客户端。我们提供了一个[示例工具服务器](https://github.com/openai/gpt-oss/tree/main/gpt-oss-mcp-server)。启动这些服务器并将 sglang 指向它们：
```bash
mcp run -t sse browser_server.py:mcp
mcp run -t sse python_server.py:mcp

python -m sglang.launch_server ... --tool-server ip-1:port-1,ip-2:port-2
```
这些 URL 应该是暴露服务器信息和有完善文档的工具的 MCP SSE 服务器。这些工具会被添加到 system prompt 中，以便模型可以使用它们。

## 投机解码

SGLang 支持使用 EAGLE3 算法对 GPT-OSS 模型进行投机解码。这可以显著提升解码速度，尤其是在小批量（batch size）的情况下。

**用法**：
添加 `--speculative-algorithm EAGLE3` 以及草稿模型路径。
```bash
python3 -m sglang.launch_server \
  --model-path openai/gpt-oss-120b \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path lmsys/EAGLE3-gpt-oss-120b-bf16 \
  --tp 2
```

```{tip}
要为 EAGLE3 投机解码启用实验性的 overlap scheduler，请设置环境变量 `SGLANG_ENABLE_SPEC_V2=1`。这可以通过在草稿（draft）和验证（verification）阶段之间启用 overlap scheduling 来提升性能。
```

### 快速演示

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="sk-123456"
)

tools = [
    {"type": "code_interpreter"},
    {"type": "web_search_preview"},
]

# Reasoning level example
response = client.responses.create(
    model="openai/gpt-oss-120b",
    instructions="You are a helpful assistant."
    reasoning_effort="high" # Supports high, medium, or low
    input="In one sentence, explain the transformer architecture.",
)
print("====== reasoning: high ======")
print(response.output_text)

# Test python tool
response = client.responses.create(
    model="openai/gpt-oss-120b",
    instructions="You are a helpful assistant, you could use python tool to execute code.",
    input="Use python tool to calculate the sum of 29138749187 and 29138749187", # 58,277,498,374
    tools=tools
)
print("====== test python tool ======")
print(response.output_text)

# Test browser tool
response = client.responses.create(
    model="openai/gpt-oss-120b",
    instructions="You are a helpful assistant, you could use browser to search the web",
    input="Search the web for the latest news about Nvidia stock price",
    tools=tools
)
print("====== test browser tool ======")
print(response.output_text)
```

示例输出：
```
====== test python tool ======
The sum of 29,138,749,187 and 29,138,749,187 is **58,277,498,374**.
====== test browser tool ======
**Recent headlines on Nvidia (NVDA) stock**

| Date (2025) | Source | Key news points | Stock‑price detail |
|-------------|--------|----------------|--------------------|
| **May 13** | Reuters | The market data page shows Nvidia trading “higher” at **$116.61** with no change from the previous close. | **$116.61** – latest trade (delayed ≈ 15 min)【14†L34-L38】 |
| **Aug 18** | CNBC | Morgan Stanley kept an **overweight** rating and lifted its price target to **$206** (up from $200), implying a 14 % upside from the Friday close. The firm notes Nvidia shares have already **jumped 34 % this year**. | No exact price quoted, but the article signals strong upside expectations【9†L27-L31】 |
| **Aug 20** | The Motley Fool | Nvidia is set to release its Q2 earnings on Aug 27. The article lists the **current price of $175.36**, down 0.16 % on the day (as of 3:58 p.m. ET). | **$175.36** – current price on Aug 20【10†L12-L15】【10†L53-L57】 |

**What the news tells us**

* Nvidia’s share price has risen sharply this year – up roughly a third according to Morgan Stanley – and analysts are still raising targets (now $206).
* The most recent market quote (Reuters, May 13) was **$116.61**, but the stock has surged since then, reaching **$175.36** by mid‑August.
* Upcoming earnings on **Aug 27** are a focal point; both the Motley Fool and Morgan Stanley expect the results could keep the rally going.

**Bottom line:** Nvidia’s stock is on a strong upward trajectory in 2025, with price targets climbing toward $200‑$210 and the market price already near $175 as of late August.

```
