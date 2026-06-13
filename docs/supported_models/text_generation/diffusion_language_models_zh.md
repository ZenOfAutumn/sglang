# 扩散语言模型（Diffusion Language Models）

扩散语言模型在具备并行解码能力的非自回归文本生成方面展现出了潜力。与自回归语言模型不同，不同的扩散语言模型需要不同的解码策略。

## 示例启动命令

SGLang 支持不同的 DLLM 算法，例如 `LowConfidence` 和 `JointThreshold`。

```shell
python3 -m sglang.launch_server \
  --model-path inclusionAI/LLaDA2.0-mini \ # example HF/local path
  --dllm-algorithm LowConfidence \
  --dllm-algorithm-config ./config.yaml \ # Optional. Uses the algorithm's default if not set.
  --host 0.0.0.0 \
  --port 30000
```

## 示例配置文件

根据所选算法的不同，配置参数也会有所不同。

LowConfidence 配置：

```yaml
# Confidence threshold for accepting predicted tokens
# - Higher values: More conservative, better quality but slower
# - Lower values: More aggressive, faster but potentially lower quality
# Range: 0.0 - 1.0
threshold: 0.95

# Default: 32, for LLaDA2MoeModelLM
block_size: 32
```

JointThreshold 配置：

```yaml
# Decoding threshold for Mask-to-Token (M2T) phase
# - Higher values: More conservative, better quality but slower
# - Lower values: More aggressive, faster but potentially lower quality
# Range: 0.0 - 1.0
threshold: 0.5
# Decoding threshold for Token-to-Token (T2T) phase
# Range: 0.0 - 1.0
# Setting to 0.0 allows full editing (recommended for most cases).
edit_threshold: 0.0
# Max extra T2T steps after all masks are removed. Prevents infinite loops.
max_post_edit_steps: 16
# 2-gram repetition penalty (default 0).
# An empirical value of 3 is often sufficient to mitigate most repetitions.
penalty_lambda: 0
```

## 示例客户端代码片段

与其他受支持的模型一样，扩散语言模型可以通过 REST API 或 Python 客户端使用。

向已启动的服务器发起生成请求的 Python 客户端示例：

```python
import sglang as sgl

def main():
    llm = sgl.Engine(model_path="inclusionAI/LLaDA2.0-mini",
                     dllm_algorithm="LowConfidence",
                     max_running_requests=1,
                     trust_remote_code=True)

    prompts = [
        "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role> Write a brief introduction of the great wall <|role_end|><role>ASSISTANT</role>"
    ]

    sampling_params = {
        "temperature": 0,
        "max_new_tokens": 1024,
    }

    outputs = llm.generate(prompts, sampling_params)
    print(outputs)

if __name__ == '__main__':
    main()
```

向已启动的服务器发起生成请求的 Curl 示例：

```bash
curl -X POST "http://127.0.0.1:30000/generate" \
     -H "Content-Type: application/json" \
     -d '{
        "text": [
            "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role> Write the number from 1 to 128 <|role_end|><role>ASSISTANT</role>",
            "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role> Write a brief introduction of the great wall <|role_end|><role>ASSISTANT</role>"
        ],
        "stream": true,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 1024
        }
    }'
```

## 支持的模型

下表汇总了支持的模型。

| Model Family               | Example Model                | Description                                                                                          |
| -------------------------- | ---------------------------- | ---------------------------------------------------------------------------------------------------- |
| **LLaDA2.0 (mini, flash)** | `inclusionAI/LLaDA2.0-flash` | LLaDA2.0-flash 是一个扩散语言模型，采用 100B 的混合专家（MoE）架构。 |
| **SDAR (JetLM)**           | `JetLM/SDAR-8B-Chat`         | SDAR 系列扩散语言模型（Chat），dense 架构。                                 |
| **SDAR (JetLM)**           | `JetLM/SDAR-30B-A3B-Chat`    | SDAR 系列扩散语言模型（Chat），MoE 架构。                                   |
