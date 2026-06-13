# 使用 SGLang 评估新模型

本文档提供了用于评估模型准确率和性能的命令。在开源新模型之前，我们强烈建议运行这些命令，以验证分数是否与你内部的 benchmark 结果一致。

**为了进行交叉验证，请在开源你的模型时，提交用于安装、启动服务器和运行 benchmark 的命令，以及所有的分数和硬件要求。**

[参考：MiniMax M2](https://github.com/sgl-project/sglang/pull/12129)

## 准确率

### LLMs

SGLang 提供了内置脚本来评估常见的 benchmark。

**MMLU**

```bash
python -m sglang.test.run_eval \
  --eval-name mmlu \
  --port 30000 \
  --num-examples 1000 \
  --max-tokens 8192
```

**GSM8K**

```bash
python -m sglang.test.few_shot_gsm8k \
  --host 127.0.0.1 \
  --port 30000 \
  --num-questions 200 \
  --num-shots 5
```

**HellaSwag**

```bash
python benchmark/hellaswag/bench_sglang.py \
  --host 127.0.0.1 \
  --port 30000 \
  --num-questions 200 \
  --num-shots 20
```

**GPQA**

```bash
python -m sglang.test.run_eval \
  --eval-name gpqa \
  --port 30000 \
  --num-examples 198 \
  --max-tokens 120000 \
  --repeat 8
```

```{tip}
对于推理模型，请添加 `--thinking-mode <mode>`（例如 `qwen3`、`deepseek-v3`）。如果模型已强制启用 thinking，则可以跳过此项。
```

**HumanEval**

```bash
pip install human_eval

python -m sglang.test.run_eval \
  --eval-name humaneval \
  --num-examples 10 \
  --port 30000
```

### VLMs

**MMMU**

```bash
python benchmark/mmmu/bench_sglang.py \
  --port 30000 \
  --concurrency 64
```

```{tip}
你可以通过传递 `--extra-request-body '{"max_tokens": 4096}'` 来设置最大 token 数。
```

对于能够处理视频的模型，我们建议将评估扩展到包括 `VideoMME`、`MVBench` 以及其他相关的 benchmark。

## 性能

性能 benchmark 测量 **延迟**（Time To First Token - TTFT）和 **吞吐量**（tokens/second）。

### LLMs

**延迟敏感型 Benchmark**

这模拟了低并发场景（例如单用户）以测量延迟。

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --host 0.0.0.0 \
  --port 30000 \
  --dataset-name random \
  --num-prompts 10 \
  --max-concurrency 1
```

**吞吐量敏感型 Benchmark**

这模拟了高流量场景以测量系统的最大吞吐量。

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --host 0.0.0.0 \
  --port 30000 \
  --dataset-name random \
  --num-prompts 1000 \
  --max-concurrency 100
```

**单批次性能**

你也可以离线 benchmark 处理单个批次的性能。

```bash
python -m sglang.bench_one_batch_server \
  --model <model-path> \
  --batch-size 8 \
  --input-len 1024 \
  --output-len 1024
```

你可以运行更细粒度的 benchmark：

- **低并发**：`--num-prompts 10 --max-concurrency 1`
- **中并发**：`--num-prompts 80 --max-concurrency 16`
- **高并发**：`--num-prompts 500 --max-concurrency 100`

## 报告结果

对于每次评估，请报告：

1.  **指标分数**：准确率 %（LLMs 和 VLMs）；延迟（ms）和吞吐量（tok/s）（仅 LLMs）。
2.  **环境设置**：GPU 类型/数量、SGLang commit hash。
3.  **启动配置**：模型路径、TP size 以及任何特殊标志。
4.  **评估参数**：shots 数量、examples 数量、最大 token 数。
