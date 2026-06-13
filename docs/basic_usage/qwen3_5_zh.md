# Qwen 3.5 使用指南

Qwen 3.5 是阿里巴巴最新一代的 LLM，采用混合注意力架构、带共享专家的先进 MoE，以及原生多模态能力。

主要架构特性：
- **混合注意力（Hybrid Attention）**：Gated Delta Networks（线性，O(n) 复杂度）与每隔 4 层一次的全注意力相结合，以实现高关联召回能力
- **带共享专家的 MoE（MoE with Shared Experts）**：在 64 个路由专家中激活 Top-8，外加一个用于通用特征的专用共享专家
- **多模态（Multimodal）**：带 Conv3d 的 DeepStack Vision Transformer，实现原生图像和视频理解

## 使用 SGLang 启动 Qwen 3.5

### Dense 模型

在 8 块 GPU 上部署 `Qwen/Qwen3.5-397B-A17B`：

```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-397B-A17B \
    --tp 8 \
    --trust-remote-code
```

### AMD GPU (MI300X / MI325X / MI35X)

在 AMD Instinct GPU 上，使用 `triton` 注意力后端。全注意力层和 Gated Delta Net（线性注意力）层在 ROCm 上都使用基于 Triton 的 kernel：

```bash
SGLANG_USE_AITER=1 python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-397B-A17B \
    --tp 8 \
    --attention-backend triton \
    --trust-remote-code
```

```{tip}
设置 `SGLANG_USE_AITER=1` 以启用 AMD 针对 MoE 和 GEMM 运算优化的 aiter kernel。
```

### 配置技巧

- `--attention-backend`：在 AMD GPU 上为 Qwen 3.5 使用 `triton`。混合注意力架构（Gated Delta Networks + 全注意力）在 ROCm 上与 Triton 后端配合效果最佳。线性注意力（GDN）层始终通过 `GDNAttnBackend` 在内部使用 Triton kernel。
- `--watchdog-timeout`：对于这种大模型，将其增大至 `1200` 或更高，因为权重加载需要相当长的时间。
- `--model-loader-extra-config '{"enable_multithread_load": true}'`：启用并行权重加载以加快启动速度。

### 推理与工具调用

Qwen 3.5 通过 Qwen3 解析器支持推理（reasoning）和工具调用（tool calling）：

```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-397B-A17B \
    --tp 8 \
    --trust-remote-code \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder
```

## 精度评测

你可以使用 `lm-eval` 来评测模型精度：

```bash
pip install lm-eval[api]

lm_eval --model local-completions \
    --model_args '{"base_url": "http://localhost:8000/v1/completions", "model": "Qwen/Qwen3.5-397B-A17B", "num_concurrent": 256, "max_retries": 10, "max_gen_toks": 2048}' \
    --tasks gsm8k \
    --batch_size auto \
    --num_fewshot 5 \
    --trust_remote_code
```

## 其他资源

- [AMD Day 0 Support for Qwen 3.5 on AMD Instinct GPUs](https://www.amd.com/en/developer/resources/technical-articles/2026/day-0-support-for-qwen-3-5-on-amd-instinct-gpus.html)
- [HuggingFace Model Card](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
