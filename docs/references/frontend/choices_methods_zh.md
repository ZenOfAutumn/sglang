# SGLang 中的 Choices 方法
本文档描述 SGLang 支持的 choices 方法。

可选的 `choices_method` 参数决定了如何选择提供给 SGLang `choices` 原语的选项。只有 `RuntimeEndpoint` 后端支持 `choices_method` 参数。其他后端,如 `OpenAI`,由于 API 限制而有各自定制的选择实现。

## 方法

### Token 长度归一化(Token Length Normalized)

Token 长度归一化是 SGLang 默认的 choices 方法。它选择在其所有 token 上平均 logprob 最高的选项。

用法示例(或者,直接省略 `choices_method` 参数):
```python
@sgl.function
def example(s):
    s += sgl.user("What is the capital of France?")
    s += sgl.assistant(
        sgl.gen(
            "answer",
            choices=["London", "Paris", "Berlin"],
            choices_method=sgl.token_length_normalized,
        )
    )
```


如果某个选项包含许多 token,而其后面的 token 是基于前面的 token 以高置信度预测出来的,这种方法可能表现不佳。例如,如果指定的选项是 `["Paris", "Antidisestablishmentarianism"]`,即使是强大的模型也会在上面的例子中失败。

### 贪心 Token 选择(Greedy Token Selection)

贪心 token 选择只是简单地选择其首个 token 的 logprob 最高的选项。对于其中一个选项是另一个更长选项子集的重叠选项,较短选项的 logprobs 会用其平均 logprob 进行扩展,以便与较长选项进行比较。

用法示例:
```python
@sgl.function
def example(s):
    s += sgl.user("What is the capital of France?")
    s += sgl.assistant(
        sgl.gen(
            "answer",
            choices=["London", "Paris", "Berlin"],
            choices_method=sgl.greedy_token_selection,
        )
    )
```

如果某个选项凭借一个有吸引力的首个 token 把模型引向错误的路径,这种方法可能表现不佳。例如,贪心选择会导致这个例子产生错误的响应:
```python
@sgl.function
def us_president_example(s):
    s += sgl.user("Name a US president.")
    s += sgl.assistant(
        sgl.gen(
            "answer",
            choices=["Donald Duck", "Millard Fillmore"],
            choices_method=sgl.greedy_token_selection,
        )
    )
```

### 无条件似然归一化(Unconditional Likelihood Normalized)

无条件似然归一化选择在用无条件 token logprobs 归一化后平均 token logprob 最高的选项,如 [这篇 EleutherAI 博文](https://blog.eleuther.ai/multiple-choice-normalization/) 所述。该方法会额外产生一次 LLM 调用以获取无条件似然。

用法示例:
```python
@sgl.function
def example(s):
    s += sgl.user("What is the capital of France?")
    s += sgl.assistant(
        sgl.gen(
            "answer",
            choices=["London", "Paris", "Berlin"],
            choices_method=sgl.unconditional_likelihood_normalized,
        )
    )
```
