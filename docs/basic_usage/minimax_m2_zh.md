# MiniMax M2.5/M2.1/M2 使用指南

[MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5)、[MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1) 和 [MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2) 是由 [MiniMax](https://www.minimax.io/) 打造的先进大语言模型。

MiniMax-M2 系列重新定义了智能体（agent）的效率。这些紧凑、快速且经济高效的 MoE 模型（总参数 2300 亿、激活参数 100 亿）专为编码和智能体任务中的卓越性能而构建，同时保持强大的通用智能。仅凭 100 亿激活参数，MiniMax-M2 系列即可提供当今领先模型所应有的精巧的端到端工具使用性能，并且以一种更精简的形态使部署和扩展比以往任何时候都更容易。

## 支持的模型

本指南适用于以下模型。部署时你只需更换模型名称即可。以下示例使用 **MiniMax-M2**：

- [MiniMaxAI/MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5)
- [MiniMaxAI/MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1)
- [MiniMaxAI/MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2)

## 系统要求

以下为推荐配置；实际需求应根据你的使用场景进行调整：

- 4x 96GB GPU：支持最高 400K tokens 的上下文长度。
- 8x 144GB GPU：支持最高 3M tokens 的上下文长度。

## 使用 Python 部署

4 GPU 部署命令：

```bash
python -m sglang.launch_server \
    --model-path MiniMaxAI/MiniMax-M2 \
    --tp-size 4 \
    --tool-call-parser minimax-m2 \
    --reasoning-parser minimax-append-think \
    --host 0.0.0.0 \
    --trust-remote-code \
    --port 8000 \
    --mem-fraction-static 0.85
```

8 GPU 部署命令：

```bash
python -m sglang.launch_server \
    --model-path MiniMaxAI/MiniMax-M2 \
    --tp-size 8 \
    --ep-size 8 \
    --tool-call-parser minimax-m2 \
    --reasoning-parser minimax-append-think \
    --host 0.0.0.0 \
    --trust-remote-code \
    --port 8000 \
    --mem-fraction-static 0.85
```

### AMD GPU (MI300X/MI325X/MI355X)

8 GPU 部署命令：

```bash
SGLANG_USE_AITER=1 python -m sglang.launch_server \
    --model-path MiniMaxAI/MiniMax-M2.5 \
    --tp-size 8 \
    --ep-size 8 \
    --attention-backend aiter \
    --tool-call-parser minimax-m2 \
    --reasoning-parser minimax-append-think \
    --host 0.0.0.0 \
    --trust-remote-code \
    --port 8000 \
    --mem-fraction-static 0.85
```

## 测试部署

启动后，你可以使用以下命令测试 SGLang 的 OpenAI 兼容 API：

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniMaxAI/MiniMax-M2",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": "Who won the world series in 2020?"}]}
        ]
    }'
```
