# SGLang 中的 Transformers 回退机制

`sglang` 可以回退到使用 `transformers` 中提供的模型。这对大多数 decoder 风格的语言模型都有效，对视觉-语言模型的支持也即将推出！

## 示例启动命令

默认情况下，如果 sglang 有可用的实现，我们会使用 sglang 的实现。否则，我们会回退到 transformers 的实现。不过，你可以通过将 `--model-impl` 设置为 `transformers` 来切换实现。

```shell
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --host 0.0.0.0 \
  --port 30000 \
  --model-impl transformers
```

## 支持的特性

### 量化

Transformers 回退机制已支持 SGLang 中大多数可用的量化方式（GGUF 除外）。有关 SGLang 支持的量化方式的更多信息，请参阅[量化页面](../../advanced_features/quantization.md)。

### 远程代码

这种回退机制还意味着：hub 上任何能够在 `transformers` 中使用 `trust_remote_code=True` 且正确实现了 attention 的模型，都可以用于生产环境！

一个模型只需要满足以下两点：

```python
from transformers import PreTrainedModel
from torch import nn

class MyAttention(nn.Module):

  def forward(self, hidden_states, **kwargs): # <- kwargs are required

    ...
    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    attn_output, attn_weights = attention_interface(
      self,
      query_states,
      key_states,
      value_states,
      **kwargs,
    )
    ...

class MyModel(PreTrainedModel):
  _supports_attention_backend = True
```

后台发生的过程如下：

1. 加载配置（config）。
2. 从 `auto_map` 中加载 `MyModel` python 类，并检查该模型是否 `_supports_attention_backend`。
3. 使用 `TransformersModel` 后端。参见 `/srt/models/transformers`，它利用了 `self.config._attn_implementation = "sglang"`，因此需要使用 `ALL_ATTENTION_FUNCTIONS`。

就是这样！
