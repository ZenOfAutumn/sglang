# 奖励模型（Reward Models）

这些模型输出一个标量奖励分数或分类结果，常用于强化学习或内容审核任务。

```{important}
它们通过 `--is-embedding` 运行，部分模型可能需要 `--trust-remote-code`。
```

## 示例启动命令

```shell
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-Math-RM-72B \  # example HF/local path
  --is-embedding \
  --host 0.0.0.0 \
  --tp-size=4 \                          # set for tensor parallelism
  --port 30000 \
```

## 支持的模型

| Model Family (Reward)                                                     | Example HuggingFace Identifier                              | Description                                                                     |
|---------------------------------------------------------------------------|-----------------------------------------------------|---------------------------------------------------------------------------------|
| **Llama (3.1 Reward / `LlamaForSequenceClassification`)**                   | `Skywork/Skywork-Reward-Llama-3.1-8B-v0.2`            | 基于 Llama 3.1（8B）的奖励模型（偏好分类器），用于为 RLHF 的响应打分和排序。  |
| **Gemma 2 (27B Reward / `Gemma2ForSequenceClassification`)**                | `Skywork/Skywork-Reward-Gemma-2-27B-v0.2`             | 衍生自 Gemma‑2（27B），该模型为 RLHF 和多语言任务提供人类偏好打分。  |
| **InternLM 2 (Reward / `InternLM2ForRewardMode`)**                         | `internlm/internlm2-7b-reward`                       | 基于 InternLM 2（7B）的奖励模型，用于对齐流程中引导输出朝向期望行为。  |
| **Qwen2.5 (Reward - Math / `Qwen2ForRewardModel`)**                         | `Qwen/Qwen2.5-Math-RM-72B`                           | 来自 Qwen2.5 系列的 72B 数学专用 RLHF 奖励模型，针对评估和优化响应进行了调优。  |
| **Qwen2.5 (Reward - Sequence / `Qwen2ForSequenceClassification`)**          | `jason9693/Qwen2.5-1.5B-apeach`                      | 一个较小的 Qwen2.5 变体，用于序列分类，提供另一种 RLHF 打分机制。  |
