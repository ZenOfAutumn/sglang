# 确定性推理(Deterministic Inference)

## 为什么确定性推理很重要

确定性推理确保 LLM 在多次运行中产生一致的输出,这对以下方面至关重要:
- **强化学习(Reinforcement Learning)**:确保多次运行中 logprobs 的一致性,减少随机噪声,使 RL 训练更稳定、可复现、可调试。
- **测试与调试**:实现可复现的验证
- **生产环境**:提升可靠性和用户体验

即使设置了 `temperature=0`,由于动态批处理以及 GPU kernel 中不同的归约顺序,标准的 LLM 推理仍可能产生不同的输出。

## 非确定性的根本原因

主要来源是**批次大小的变化**。不同的批次大小导致 GPU kernel 以不同的方式拆分归约(reduction)操作,从而产生不同的加法顺序。由于浮点运算的非结合性(`(a + b) + c ≠ a + (b + c)`),即使是相同的输入也会产生不同的结果。


## SGLang 的解决方案

基于 [Thinking Machines Lab 的批次不变算子(batch-invariant operators)](https://github.com/thinking-machines-lab/batch_invariant_ops),SGLang 实现了完全确定性的推理,同时保持与 chunked prefill、CUDA graph、radix cache 和非贪婪采样的兼容性。确定性推理功能的开发路线图可在此 [issue](https://github.com/sgl-project/sglang/issues/10278) 中找到。

### 支持的后端

确定性推理仅支持以下三种注意力后端:**FlashInfer**、**FlashAttention 3(FA3)** 和 **Triton**。

下表显示了确定性推理在不同注意力后端上的功能兼容性:

| 注意力后端 | CUDA Graph | Chunked Prefill | Radix Cache | 非贪婪采样(Temp > 0) |
|-------------------|------------|-----------------|-------------|---------------------|
| **FlashInfer** | ✅ 是 | ✅ 是 | ❌ 否 | ✅ 是 |
| **FlashAttention 3 (FA3)** | ✅ 是 | ✅ 是 | ✅ 是 | ✅ 是 |
| **Triton** | ✅ 是 | ✅ 是 | ✅ 是 | ✅ 是 |

## 用法

### 基本用法

通过添加 `--enable-deterministic-inference` 标志启用确定性推理:

```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --attention-backend fa3 \
    --enable-deterministic-inference
```

### 服务器参数

| 参数 | 类型/默认值 | 描述 |
|----------|--------------|-------------|
| `--enable-deterministic-inference` | flag;默认:禁用 | 启用带批次不变操作的确定性推理 |
| `--attention-backend` | string;默认:fa3 | 选择注意力后端(flashinfer、fa3 或 triton) |

### 配置示例

#### Qwen3-8B
```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --attention-backend flashinfer \
    --enable-deterministic-inference
```

#### Llama 模型
```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --attention-backend fa3 \
    --enable-deterministic-inference
```

#### Qwen3-30B-A3B(MoE 模型)
```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-30B-A3B \
    --attention-backend fa3 \
    --enable-deterministic-inference
```

### 带非贪婪采样的确定性推理(Temperature > 0)

SGLang 通过使用采样种子(sampling seed),即使在非贪婪采样下也支持确定性推理。这对于像 GRPO(Group Relative Policy Optimization)这样需要多个多样化但可复现的响应的强化学习场景特别有用。

#### 默认行为

默认情况下,SGLang 使用 `42` 作为采样种子以实现可复现的采样:

```python
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Tell me a joke",
        "sampling_params": {
            "temperature": 0.8,  # Non-greedy sampling
            "max_new_tokens": 128,
        },
    },
)
print(response.json())
# This will always produce the same response across runs
```

#### 生成多个可复现的响应

要从相同的 prompt 采样不同的响应同时保持可复现性(例如,用于 GRPO 训练),在你的请求中提供不同的采样种子:

```python
import requests

# Prepare a list of sampling seeds for different responses
sampling_seeds = [42, 43, 44, 45, 46]

responses = []
for seed in sampling_seeds:
    response = requests.post(
        "http://localhost:30000/generate",
        json={
            "text": "Tell me a joke",
            "sampling_params": {
                "temperature": 0.8,
                "max_new_tokens": 128,
                "sampling_seed": seed,  # Specify sampling seed
            },
        },
    )
    responses.append(response.json())

# Each seed will produce a different but reproducible response
# Using the same seed will always produce the same response
```

这种方法确保了:
- 不同的种子产生多样化的响应
- 相同的种子在不同的运行中总是产生相同的响应
- 结果对于调试和评估是可复现的


## 验证

运行确定性测试以验证一致的输出:

```bash
# Single test: same prompt, varying batch sizes
python3 -m sglang.test.test_deterministic --test-mode single --n-trials 50

# Prefix test: prompts with different prefix lengths
python3 -m sglang.test.test_deterministic --test-mode prefix --n-trials 50

# Radix Cache Consistency mode: test radix cache determinism (cached vs uncached prefill)
python3 -m sglang.test.test_deterministic --test-mode radix_cache
```

预期结果:所有测试都应显示 `Unique samples: 1`(完美确定性)。
