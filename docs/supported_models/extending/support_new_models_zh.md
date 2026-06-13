# 如何支持新模型

本文档介绍如何在 SGLang 中添加对新语言模型和多模态大语言模型（MLLM）的支持。文中还涵盖如何测试新模型以及注册外部实现。

## 如何支持新的语言模型

要在 SGLang 中支持一个新模型，你只需在 [SGLang 模型目录](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/models) 下添加一个文件即可。你可以借鉴现有的模型实现，并为你的模型创建一个新文件。对于大多数模型，你应该能够找到一个相似的模型作为起点（例如从 Llama 开始）。也请参阅如何[将模型从 vLLM 移植到 SGLang](#port-a-model-from-vllm-to-sglang)。

## 如何支持新的多模态大语言模型

要在 SGLang 中支持一个新的多模态大语言模型（MLLM），除了标准的 LLM 支持外，还有几个关键组件：

1. **将你的新模型注册为多模态**：
   扩展 [model_config.py](https://github.com/sgl-project/sglang/blob/0ab3f437aba729b348a683ab32b35b214456efc7/python/sglang/srt/configs/model_config.py#L561) 中的 `is_multimodal_model`，使其对你的模型返回 `True`。

2. **注册一个新的 chat-template**：
   仅当你的默认 chat-template 无法接受图像作为输入时：在 [conversation.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/parser/conversation.py) 中注册一个新的 chat template 以及相应的匹配函数。

3. **多模态数据处理器**：
   定义一个继承自 `BaseMultimodalProcessor` 的新 `Processor` 类，并将该处理器注册为你的模型的专用处理器。
   更多详情请参阅 [multimodal_processor.py](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/multimodal/processors)。

4. **处理多模态 token**：
   为你的新模型实现一个 `pad_input_ids` 函数。在该函数中，提示词中的多模态 token 应被展开（如有必要）并用多模态数据哈希（multimodal-data-hashes）进行填充，以便 SGLang 能够通过 `RadixAttention` 识别不同的多模态数据。

5. **处理图像特征提取**：
   为你的新模型实现一个 `get_image_feature` 函数，它从原始图像数据中提取图像特征，并将其转换为语言模型所使用的嵌入。

6. **适配视觉注意力（Vision Attention）**：
   将 ViT 的多头 `Attention` 适配为 SGLang 的 `VisionAttention`。

你可以参考 [Qwen2VL](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen2_vl.py) 或其他 mllm 实现。这些模型演示了如何正确处理多模态和文本输入。

## 测试与调试

请在 PR 描述中记录你所有的测试和基准测试结果。

### 交互式调试

对于交互式调试，请对比 Hugging Face/Transformers 与 SGLang 的输出。以下两条命令应该给出相同的文本输出和非常相似的 prefill logits：

- 获取参考输出：
  ```bash
  python3 scripts/playground/reference_hf.py --model-path [new model] --model-type {text,vlm}
  ```
- 获取 SGLang 输出：
  ```bash
  python3 -m sglang.bench_one_batch --correct --model [new model]
  ```

### 将模型添加到测试套件

为确保新模型得到良好维护，请将其添加到测试套件中：在 [test_generation_models.py](https://github.com/sgl-project/sglang/blob/main/test/registered/models/test_generation_models.py) 文件的 `ALL_OTHER_MODELS` 列表中包含它，在你的本地机器上测试新模型，并在你的 PR 中报告在示范性基准（GSM8K、MMLU、MMMU、MMMU-Pro 等）上的结果。 \\
对于 VLM，还需在 `test_vision_openai_server_{x}.py` 中包含一个测试（例如 [test_vision_openai_server_a.py](https://github.com/sgl-project/sglang/blob/main/test/registered/vlm/test_vision_openai_server_a.py)）。

这是在本地机器上测试新模型的一个示例命令：

```bash
ONLY_RUN=Qwen/Qwen2-1.5B python3 -m unittest test_generation_models.TestGenerationModels.test_others
```

### 基准测试

- **（必需）MMMU**：按照 MMMU 基准的 [README.md](https://github.com/sgl-project/sglang/blob/main/benchmark/mmmu/README.md) 获取 SGLang 与 HF Transformer 的准确率对比。SGLang 运行的准确率分数不应明显低于 HF Transformer 运行的分数。同样地，按照 https://docs.sglang.io/developer_guide/benchmark_and_profiling.html 获取性能对比：TTFT 和吞吐量必须达到或超过基线（例如 HF Transformer）。
- **（可选）其他评测**：如果你运行了其他评测，请在 PR 描述中记录结果。

## 将模型从 vLLM 移植到 SGLang

[vLLM 模型目录](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models) 是一个宝贵的资源，因为 vLLM 覆盖了许多模型。SGLang 复用了 vLLM 的接口和部分层，这使得将模型从 vLLM 移植到 SGLang 更加容易。

要将模型从 vLLM 移植到 SGLang：

- 对比以下两个文件以获得指导：
  - [SGLang Llama 实现](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llama.py)
  - [vLLM Llama 实现](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py)
- 主要差异包括：
  - **将 vLLM 的 `Attention` 替换为 `RadixAttention`**（确保向 `RadixAttention` 传入 `layer_id`）。
  - **将 vLLM 的 `LogitsProcessor` 替换为 SGLang 的 `LogitsProcessor`。**
  - **将 ViT 的多头 `Attention` 替换为 SGLang 的 `VisionAttention`。**
  - **将其他 vLLM 层**（例如 `RMSNorm`、`SiluAndMul`）替换为 SGLang 的层。
  - **移除 `Sample`。**
  - **修改 `forward()` 函数**并添加一个 `forward_batch()` 方法。
  - **在末尾添加 `EntryClass`。**
  - **确保新实现只使用 SGLang 组件**，且不依赖任何 vLLM 组件。

注意：请确保将你的新模型添加到受支持模型文档中的受支持模型列表里。

## 注册外部模型实现

除上述方法外，你还可以在启动服务器之前将你的新模型注册到 `ModelRegistry`。这样你无需修改源代码即可集成你的模型。

例如：

```python
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.entrypoints.http_server import launch_server

# For a single model, add it to the registry:
ModelRegistry.models[model_name] = model_class

# For multiple models, you can imitate the import_model_classes() function:
from functools import lru_cache

@lru_cache()
def import_new_model_classes():
    model_arch_name_to_cls = {}
    # Populate model_arch_name_to_cls with your new model classes.
    ...
    return model_arch_name_to_cls

ModelRegistry.models.update(import_new_model_classes())

# Launch the server with your server arguments:
launch_server(server_args)
```

## 示例：实现并服务一个 Llama Wrapper 模型

下面是一个入门级的分步演练，介绍如何在 SGLang 中端到端地实现一个新模型，然后通过[离线引擎（Offline Engine）](https://github.com/sgl-project/sglang/blob/main/docs/basic_usage/offline_engine_api.ipynb)运行它。

### 实现我们的模型

为简单起见，这个新模型将是对 [Llama 3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) 的一个简单封装，我们的目标只是在每次 `forward` 调用中通过对每个单独的 logit 取平方根来对输出 logits 进行偏置（bias）。

让我们先在一个名为 `llama_wrapper.py` 的文件中定义我们的模型。
第一步是从 SRT（SGLang 的内部后端）导入必要的库。

```python
# In the file `llama_wrapper.py`

import torch
from transformers import LlamaConfig
from typing import Optional
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

from sglang.srt.models.llama import LlamaForCausalLM
```

接下来，我们为我们的模型声明一个新的 `class`，并让它继承自 `LlamaForCausalLM`，这使得我们的模型可以访问 `LlamaForCausalLM` 预定义的模块和层，例如 `LlamaAttention` 和 `LlamaMLP`。
注意，几乎所有模型实现的 `__init__` 方法都接受 `config` 和 `quant_config` 作为参数；`config` 和 `quant_config` 通过 [`model_loader/loader.py`](https://github.com/sgl-project/sglang/blob/bf72b80122fd888bf619d17b96fa3e323ab809fc/python/sglang/srt/model_loader/loader.py#L219) 传入。
由于我们继承自 `LlamaForCausalLM`，我们可以将参数直接传递给它的构造函数，构造函数会为我们设置成员变量。

```python
class LlamaWrapper(LlamaForCausalLM):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
```

现在，我们要定义 `forward` 方法，它会在推理时被调用。
注意，`forward` 的签名对于任何模型来说本质上都是相同的；你可以查看 [`models` 目录](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/) 中定义的其他模型作为参考。
要确切了解 `forward` 在 SGLang 运行时内部的何处被调用，请查看 [`ModelRunner` 类](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py) 中的 [`forward_decode`](https://github.com/sgl-project/sglang/blob/bf72b80122fd888bf619d17b96fa3e323ab809fc/python/sglang/srt/model_executor/model_runner.py#L1705) 和 [`forward_extend`](https://github.com/sgl-project/sglang/blob/bf72b80122fd888bf619d17b96fa3e323ab809fc/python/sglang/srt/model_executor/model_runner.py#L1724)。

```python
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
    ) -> LogitsProcessorOutput:
```

现在我们调用 `self.model` 的 `__call__` 方法（`self.model` 是 `LlamaForCausalLM` 在其 `__init__` 方法中定义的成员变量），它最终会调用 `LlamaForCausalLM` 的 `forward` 方法。
之后，我们将 `hidden_states` 输入到我们模型的 `LogitsProcessor` 中（同样在 `LlamaForCausalLM` 中定义）。

```python
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        res: LogitsProcessorOutput = self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
        )
```

在获得下一个 token 的 logits 后，我们终于可以执行偏置（biasing）步骤了。

```python
        orig_logits = res.next_token_logits
        res.next_token_logits = torch.where(
            orig_logits > 0,
            orig_logits.sqrt(),
            orig_logits
        )

        return res
```

现在，我们的 `LlamaWrapper` 模型已经创建完成，可以提供服务了！

### 通过 SGLang 的离线引擎服务我们的模型

本演练的下一步是离线托管我们的新模型，使其能够在本地服务而无需 HTTP 服务器。

首先，创建一个名为 `run.py` 的新文件。
现在，我们必须确保 SGLang 的 `ModelRegistry` 能够找到我们的模型。
为此，我们首先从 Huggingface 下载模型的配置和权重。

```python
# In the file `run.py`

import asyncio
from functools import lru_cache
from huggingface_hub import snapshot_download
from llama_wrapper import LlamaWrapper # Make sure to import our new model!
import sglang as sgl
from sglang.srt.models.registry import ModelRegistry

# Make sure to request access to this model on Huggingface, then export your
# `HF_TOKEN` to download the model snapshot
llama_dir = snapshot_download(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    local_dir="./llama_ckpt",
)
```

现在我们已经把模型存到了磁盘上，我们希望将其指向 `LlamaWrapper`，方法是把 `./llama_ckpt/config.json` 中的 `architectures` 字段改为 `LlamaWrapper`。
这样，当我们把模型检查点的路径传给 SGLang 时，它就会知道我们想使用 "LlamaWrapper" 而不是 "LlamaForCausalLM" 作为我们的模型。

```python
{
  "architectures": [
   #  "LlamaForCausalLM"
    "LlamaWrapper"
  ],
  ...
}
```

然而，如果我们不把 `LlamaWrapper` 类与 "LlamaWrapper" 注册关键字关联起来，SGLang 将无法找到我们的模型。
因此，要注册我们的 `LlamaWrapper`，我们需要遵循上文标题为 "注册外部模型实现" 章节中的步骤。

```python
@lru_cache()
def import_new_model_classes():
    model_arch_name_to_cls = {"LlamaWrapper": LlamaWrapper}
    return model_arch_name_to_cls

ModelRegistry.models.update(import_new_model_classes())
```

最后，当我们创建 `Engine` 时，只需传入本地模型目录的路径。
然后，我们的 `LlamaWrapper` 就可以提供服务了；在本演练中，我们将使用 SGLang `Engine` 的非流式异步生成端点。

```python
def main():
    llm = sgl.Engine(model_path="./llama_ckpt")
    sampling_params = {"temperature": 0.2, "top_k": 5}
    prompts = [
        "Write a short, neutral self-introduction for a fictional character. Hello, my name is",
        "Provide a concise factual statement about France’s capital city. The capital of France is",
        "Explain possible future trends in artificial intelligence. The future of AI is",
    ]

    asyncio.run(run_llm(llm, sampling_params, prompts))

    llm.shutdown()

async def run_llm(
    llm,
    sampling_params,
    prompts,
) -> None:
    outputs = await llm.async_generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt}")
        print(f"Generated text: {output['text']}")

if __name__ == "__main__":
    main()
```

现在，当我们运行 `python run.py` 时，我们将得到新创建的模型的输出！

## 通过标准 CLI 服务外部模型

前面的章节展示了如何通过 `ModelRegistry` 以编程方式注册模型并通过离线引擎提供服务。与 vLLM 模型插件类似，还有一种替代方案，让你能够继续使用标准的 `python -m sglang.launch_server` CLI 而无需修改任何 SGLang 源代码：你可以使用 `SGLANG_EXTERNAL_MODEL_PACKAGE` 环境变量来注册你的模型。

### `EntryClass` 变量

当 SGLang 扫描一个模型包时，它会在你的 Python 文件的模块级别查找变量 `EntryClass`。[模型注册表（model registry）](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/registry.py) 会导入你的文件，检查 `EntryClass`，并注册分配给它的类。如果你使用的是基于 HuggingFace 的模型，这个类的名称需要与你模型 `config.json` 中的 `"architectures"` 字段匹配。

例如，如果你正在实现一个 Llama wrapper，请在模型文件末尾添加这一行：

```python
# This is what "Add EntryClass at the end" means
EntryClass = LlamaWrapper
```

### 示例：纯文本模型

使用上一节中相同的 Llama wrapper，下面介绍如何打包并通过 CLI 提供服务。

1. 创建你的项目

```
sglang_custom_project/
|----setup.py
|----custom_llm/
     |----__init__.py
     |----llama_wrapper.py
```

编写 `setup.py`：

```python
# sglang_custom_project/setup.py

from setuptools import setup, find_packages
setup(
    name="sglang-custom-plugins",
    version="0.1",
    packages=find_packages(),
)
```

2. 编写你的模型代码

在 `llama_wrapper.py` 中，编写你的模型并包含 `EntryClass`：

```python
# sglang_custom_project/custom_llm/llama_wrapper.py

import torch
from typing import Optional
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.llama import LlamaForCausalLM

class LlamaWrapper(LlamaForCausalLM):
    def __init__(self, config, quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "") -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch,
                pp_proxy_tensors=None, input_embeds=None, get_embedding=False):
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        res: LogitsProcessorOutput = self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch,
        )

        orig = res.next_token_logits
        res.next_token_logits = torch.where(orig > 0, orig.sqrt(), orig)
        return res

# Don't forget to add EntryClass
EntryClass = LlamaWrapper
```

3. 安装你的包

在你的 `sglang_custom_project` 目录中运行以下命令，将你的代码安装到当前活动的 Python 环境中：

```bash
pip install -e .
```

4. 更新你的 `config.json`

更新你的 HuggingFace 模型检查点目录下的 `config.json`，使 `architectures` 字段与你的类名匹配：

```json
{
  "architectures": ["LlamaWrapper"],
  ...
}
```

5. 启动服务器

在运行 CLI 之前设置环境变量：

```bash
export SGLANG_EXTERNAL_MODEL_PACKAGE=custom_llm
python -m sglang.launch_server \
    --model-path /path/to/Llama-3.1-8B-Instruct \
    --port 8000
```

`SGLANG_EXTERNAL_MODEL_PACKAGE` 应为包含你模型相关代码的父文件夹名称。在本例中，它应为 `custom_llm`。

### 示例：多模态模型

如果你处理的是多模态模型，仅设置 `SGLANG_EXTERNAL_MODEL_PACKAGE` 是不够的。SGLang 还需要将你的架构识别为多模态，以启用图像/视频处理流水线，并且它需要一个自定义处理器。

你可以通过设置两个额外的环境变量来处理这一点：

- `SGLANG_EXTERNAL_MM_MODEL_ARCH`：将你的架构名称添加到 SGLang 的内部多模态模型列表中。
- `SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE`：告诉 SGLang 在哪里找到你的自定义处理器类。

例如，让我们构建一个基于 Qwen2-VL-Instruct、对 logits 取平方根的自定义模型。

创建项目：

```
sglang_custom_project_vl/
|----setup.py
|----custom_vlm/
     |----__init__.py
     |----qwenvl_wrapper.py
```

编写 `setup.py`：

```python
# sglang_custom_project_vl/setup.py

from setuptools import setup, find_packages
setup(
    name="sglang-custom-plugins-vl",
    version="0.1",
    packages=find_packages(),
)
```

在 `qwenvl_wrapper.py` 中编写模型：

```python
# sglang_custom_project_vl/custom_vlm/qwenvl_wrapper.py
import torch
from sglang.srt.models.qwen2_vl import Qwen2VLForConditionalGeneration
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

class CustomQwen2VL(Qwen2VLForConditionalGeneration):
    def forward(self, input_ids, positions, forward_batch,
                input_embeds=None, get_embedding=False):
        res = super().forward(
            input_ids, positions, forward_batch,
            input_embeds=input_embeds, get_embedding=get_embedding
        )
        if not get_embedding:
            orig = res.next_token_logits
            res.next_token_logits = torch.where(orig > 0, orig.sqrt(), orig)
        return res

class CustomQwen2VLProcessor(QwenVLImageProcessor):
    models = [CustomQwen2VL]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

EntryClass = CustomQwen2VL
```

**注意：** 只要你将处理器与特定的模型类关联，就不需要为自定义处理器单独设置 `EntryClass`。

安装包、更新 `config.json` 并启动：

```bash
pip install -e .
```

```json
{
  "architectures": ["CustomQwen2VL"],
  ...
}
```

```bash
export SGLANG_EXTERNAL_MODEL_PACKAGE=custom_vlm
export SGLANG_EXTERNAL_MM_MODEL_ARCH=CustomQwen2VL
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=custom_vlm

python -m sglang.launch_server \
    --model-path /path/to/Qwen2-VL-2B-Instruct \
    --port 8000 \
    --enable-multimodal
```

## 文档

将其添加到 [generative_models.md](../text_generation/generative_models.md) 或 [multimodal_language_models.md](../text_generation/multimodal_language_models.md) 中的受支持模型表格里。

---

遵循这些指南，你就可以在 SGLang 中添加对新语言模型和多模态大语言模型的支持，并确保它们经过充分测试并能轻松集成到系统中。
