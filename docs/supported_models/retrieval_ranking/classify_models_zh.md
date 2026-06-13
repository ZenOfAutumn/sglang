# 分类 API（Classification API）

本文档描述了 SGLang 中 `/v1/classify` API 端点的实现，它与 vLLM 的分类 API 格式兼容。

## 概述

分类 API 允许你使用分类模型对文本输入进行分类。此实现遵循与 vLLM 0.7.0 分类 API 相同的格式。

## API 端点

```
POST /v1/classify
```

## 请求格式

```json
{
  "model": "model_name",
  "input": "text to classify"
}
```

### 参数

- `model`（string，必填）：要使用的分类模型名称
- `input`（string，必填）：要分类的文本
- `user`（string，可选）：用于跟踪的用户标识
- `rid`（string，可选）：用于跟踪的请求 ID
- `priority`（integer，可选）：请求优先级

## 响应格式

```json
{
  "id": "classify-9bf17f2847b046c7b2d5495f4b4f9682",
  "object": "list",
  "created": 1745383213,
  "model": "jason9693/Qwen2.5-1.5B-apeach",
  "data": [
    {
      "index": 0,
      "label": "Default",
      "probs": [0.565970778465271, 0.4340292513370514],
      "num_classes": 2
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10,
    "completion_tokens": 0,
    "prompt_tokens_details": null
  }
}
```

### 响应字段

- `id`：分类请求的唯一标识符
- `object`：始终为 "list"
- `created`：请求创建时的 Unix 时间戳
- `model`：用于分类的模型
- `data`：分类结果数组
  - `index`：结果的索引
  - `label`：预测的类别标签
  - `probs`：每个类别的概率数组
  - `num_classes`：类别总数
- `usage`：token 使用信息
  - `prompt_tokens`：输入 token 数量
  - `total_tokens`：token 总数
  - `completion_tokens`：补全 token 数量（分类任务始终为 0）
  - `prompt_tokens_details`：附加的 token 详情（可选）

## 使用示例

### 使用 curl

```bash
curl -v "http://127.0.0.1:8000/v1/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jason9693/Qwen2.5-1.5B-apeach",
    "input": "Loved the new café—coffee was great."
  }'
```

### 使用 Python

```python
import requests
import json

# Make classification request
response = requests.post(
    "http://127.0.0.1:8000/v1/classify",
    headers={"Content-Type": "application/json"},
    json={
        "model": "jason9693/Qwen2.5-1.5B-apeach",
        "input": "Loved the new café—coffee was great."
    }
)

# Parse response
result = response.json()
print(json.dumps(result, indent=2))
```

## 支持的模型

分类 API 适用于 SGLang 支持的任何分类模型，包括：

### 分类模型（多类别）
- `LlamaForSequenceClassification` - 多类别分类
- `Qwen2ForSequenceClassification` - 多类别分类
- `Qwen3ForSequenceClassification` - 多类别分类
- `BertForSequenceClassification` - 多类别分类
- `Gemma2ForSequenceClassification` - 多类别分类

**标签映射**：该 API 会自动使用模型 `config.json` 文件中的 `id2label` 映射，以提供有意义的标签名称，而不是通用的类别名称。如果 `id2label` 不可用，则会回退为 `LABEL_0`、`LABEL_1` 等，或最终回退为 `Class_0`、`Class_1`。

### 奖励模型（单个分数）
- `InternLM2ForRewardModel` - 单个奖励分数
- `Qwen2ForRewardModel` - 单个奖励分数
- `LlamaForSequenceClassificationWithNormal_Weights` - 特殊的奖励模型

**注意**：SGLang 中的 `/classify` 端点最初是为奖励模型设计的，但现在支持所有非生成式模型。我们的 `/v1/classify` 端点为分类任务提供了一个标准化的、与 vLLM 兼容的接口。

## 错误处理

该 API 会返回相应的 HTTP 状态码和错误消息：

- `400 Bad Request`：请求格式无效或缺少必填字段
- `500 Internal Server Error`：服务器端处理错误

错误响应格式：
```json
{
  "error": "Error message",
  "type": "error_type",
  "code": 400
}
```

## 实现细节

分类 API 通过以下方式实现：

1. **Rust Model Gateway**：在 `sgl-model-gateway/src/protocols/spec.rs` 中处理路由以及请求/响应模型
2. **Python HTTP Server**：在 `python/sglang/srt/entrypoints/http_server.py` 中实现实际端点
3. **Classification Service**：在 `python/sglang/srt/entrypoints/openai/serving_classify.py` 中处理分类逻辑

## 测试

使用提供的测试脚本来验证实现：

```bash
python test_classify_api.py
```

## 兼容性

此实现与 vLLM 的分类 API 格式兼容，使分类任务能够从 vLLM 无缝迁移到 SGLang。
