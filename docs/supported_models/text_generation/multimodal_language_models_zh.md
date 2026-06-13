# 多模态语言模型（Multimodal Language Models）

这些模型接受多模态输入（例如图像和文本）并生成文本输出。它们通过多模态编码器增强了语言模型。

## 示例启动命令

```shell
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-11B-Vision-Instruct \  # example HF/local path
  --host 0.0.0.0 \
  --port 30000 \
```

> 有关如何发送多模态请求，请参阅 [OpenAI APIs 章节](https://docs.sglang.io/basic_usage/openai_api_vision.html)。

## 支持的模型

下表汇总了支持的模型。

如果你不确定某个特定架构是否已实现，可以通过 GitHub 进行搜索。例如，要搜索 `Qwen2_5_VLForConditionalGeneration`，请在 GitHub 搜索栏中使用以下表达式：

```
repo:sgl-project/sglang path:/^python\/sglang\/srt\/models\// Qwen2_5_VLForConditionalGeneration
```


| Model Family (Variants)    | Example HuggingFace Identifier             | Description                                                                                                                                                                                                     | Notes |
|----------------------------|--------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
| **Qwen-VL** | `Qwen/Qwen3-VL-235B-A22B-Instruct`              | 阿里巴巴对 Qwen 的视觉-语言扩展；例如，Qwen2.5-VL（7B 及更大变体）能够分析图像内容并就其进行对话。                                                                     |  |
| **DeepSeek-VL2**           | `deepseek-ai/deepseek-vl2`                 | DeepSeek 的视觉-语言变体（带有专用的图像处理器），可对图像和文本输入进行高级多模态推理。                                                                        |  |
| **DeepSeek-OCR / OCR-2**   | `deepseek-ai/DeepSeek-OCR-2`               | 专注于 OCR 的 DeepSeek 模型，用于文档理解和文本提取。                                                                                                                                    | 使用 `--trust-remote-code`。 |
| **Janus-Pro** (1B, 7B)     | `deepseek-ai/Janus-Pro-7B`                 | DeepSeek 的开源多模态模型，既能进行图像理解又能进行图像生成。Janus-Pro 采用解耦架构以分离视觉编码路径，从而在两类任务中都提升了性能。 |  |
| **MiniCPM-V / MiniCPM-o**  | `openbmb/MiniCPM-V-2_6`                    | MiniCPM-V（2.6，约 8B）支持图像输入，MiniCPM-o 还增加了音频/视频；这些多模态 LLM 针对移动/边缘设备上的端侧部署进行了优化。                                                 |  |
| **Llama 3.2 Vision** (11B) | `meta-llama/Llama-3.2-11B-Vision-Instruct` | Llama 3（11B）的视觉增强变体，接受图像输入以进行视觉问答和其他多模态任务。                                                                                     |  |
| **LLaVA** (v1.5 & v1.6)    | *e.g.* `liuhaotian/llava-v1.5-13b`         | 开放的视觉对话模型，为 LLaMA/Vicuna（例如 LLaMA2 13B）添加了图像编码器，以遵循多模态指令提示。                                                                               |  |
| **LLaVA-NeXT** (8B, 72B)   | `lmms-lab/llava-next-72b`                  | 改进的 LLaVA 模型（包含一个 8B Llama3 版本和一个 72B 版本），在多模态基准上提供了更强的视觉指令遵循能力和准确性。                                                       |  |
| **LLaVA-OneVision**        | `lmms-lab/llava-onevision-qwen2-7b-ov`     | 增强的 LLaVA 变体，集成 Qwen 作为骨干网络；通过兼容 OpenAI Vision API 的格式支持多张图像（甚至视频帧）作为输入。                                                 |  |
| **Gemma 3 (Multimodal)**   | `google/gemma-3-4b-it`                     | Gemma 3 的较大模型（4B、12B、27B）在合并的 128K-token 上下文中接受图像（每张图像编码为 256 个 token）与文本。                                                                        |  |
| **Kimi-VL** (A3B)          | `moonshotai/Kimi-VL-A3B-Instruct`          | Kimi-VL 是一个多模态模型，能够从图像中理解并生成文本。                                                                                                                                |  |
| **Mistral-Small-3.1-24B**  | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | Mistral 3.1 是一个多模态模型，能够从文本或图像输入生成文本。它还支持工具调用和结构化输出。 |  |
| **Phi-4-multimodal-instruct**  | `microsoft/Phi-4-multimodal-instruct` | Phi-4-multimodal-instruct 是 Phi-4-mini 模型的多模态变体，通过 LoRA 增强以提升多模态能力。它在 SGLang 中支持文本、视觉和音频模态。 |  |
| **MiMo-VL** (7B)           | `XiaomiMiMo/MiMo-VL-7B-RL`                 | 小米紧凑而强大的视觉-语言模型，配备用于捕捉细粒度视觉细节的原生分辨率 ViT 编码器、用于跨模态对齐的 MLP 投影器，以及针对复杂推理任务优化的 MiMo-7B 语言模型。 |  |
| **GLM-4.5V** (106B) /  **GLM-4.1V**(9B)           | `zai-org/GLM-4.5V`                   | GLM-4.5V 和 GLM-4.1V-Thinking：迈向具备可扩展强化学习的通用多模态推理                                                                                                                                                                                                      | 使用 `--chat-template glm-4v` |
| **GLM-OCR**          | `zai-org/GLM-OCR`                   | GLM-OCR：一个快速准确的通用 OCR 模型                                                                   |  |
| **DotsVLM** (General/OCR)  | `rednote-hilab/dots.vlm1.inst`             | RedNote 的视觉-语言模型，构建于 1.2B 视觉编码器和 DeepSeek V3 LLM 之上，配备从头训练、支持动态分辨率的 NaViT 视觉编码器，并通过结构化图像数据训练增强了 OCR 能力。 |  |
| **DotsVLM-OCR**            | `rednote-hilab/dots.ocr`                   | DotsVLM 的专用 OCR 变体，针对光学字符识别任务进行了优化，具备增强的文本提取和文档理解能力。 | 不要使用 `--trust-remote-code` |
| **NVILA** (8B, 15B, Lite-2B, Lite-8B, Lite-15B) | `Efficient-Large-Model/NVILA-8B` | `chatml` | NVILA 探索了多模态设计的全栈效率，实现了更低的训练成本、更快的部署和更优的性能。 |
| **NVIDIA Nemotron Nano 2.0 VL** | `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` | NVIDIA Nemotron Nano v2 VL 支持多图像推理和视频理解，并具备强大的文档智能、视觉问答和摘要能力。它构建于混合 Mamba-Transformer LLM Nemotron Nano V2 之上，以在长文档和视频场景中实现更高的推理吞吐量。 | 使用 `--trust-remote-code`。你可能需要调整 `--max-mamba-cache-size`（默认值为 512）以适应内存限制。 |
| **Ernie4.5-VL** | `baidu/ERNIE-4.5-VL-28B-A3B-PT`              | 百度的视觉-语言模型（28B、424B）。支持图像和视频理解，也支持思考（thinking）。                                                                     |  |
| **JetVLM** |  | JetVLM 是一个视觉-语言模型，构建于 Jet-Nemotron 之上，专为高性能多模态理解和生成任务而设计。 | 即将推出 |
| **Step3-VL** (10B) | `stepfun-ai/Step3-VL-10B` | StepFun 轻量级开源的 10B 参数 VLM，用于多模态智能，在视觉感知、复杂推理和人类对齐方面表现出色。 |  |
| **Qwen3-Omni** | `Qwen/Qwen3-Omni-30B-A3B-Instruct` |  阿里巴巴的全模态 MoE 模型。目前支持 **Thinker** 组件（针对文本、图像、音频和视频的多模态理解），而 **Talker** 组件（音频生成）尚不支持。 |  |

## 视频输入支持

SGLang 支持视觉-语言模型（VLM）的视频输入，从而支持时序推理任务，例如视频问答、字幕生成和整体场景理解。视频片段会被解码，关键帧会被采样，所得的张量与文本提示一起批处理，使多模态推理能够整合视觉和语言上下文。

| Model Family | Example Identifier | Video notes |
|--------------|--------------------|-------------|
| **Qwen-VL** (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3-Omni) | `Qwen/Qwen3-VL-235B-A22B-Instruct` | 处理器收集 `video_data`，运行 Qwen 的帧采样器，并在推理前将所得特征与文本 token 合并。 |
| **GLM-4v** (4.5V, 4.1V, MOE) | `zai-org/GLM-4.5V` | 视频片段使用 Decord 读取，转换为张量，并连同元数据一起传递给模型以进行旋转位置（rotary-position）处理。 |
| **NVILA** (Full & Lite) | `Efficient-Large-Model/NVILA-8B` | 当存在 `video_data` 时，运行时每个片段采样八帧并将其附加到多模态请求中。 |
| **LLaVA video variants** (LLaVA-NeXT-Video, LLaVA-OneVision) | `lmms-lab/LLaVA-NeXT-Video-7B` | 处理器将视频提示路由到支持视频的 LlavaVid 架构，所提供的示例展示了如何使用 `sgl.video(...)` 片段进行查询。 |
| **NVIDIA Nemotron Nano 2.0 VL** | `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` | 处理器按模型训练设置以 2 FPS 采样，最多 128 帧。该模型使用 [EVS](../../../python/sglang/srt/multimodal/evs/README.md)，这是一种从视频嵌入中移除冗余 token 的剪枝方法。默认 `video_pruning_rate=0.7`。例如，可通过提供 `--json-model-override-args '{"video_pruning_rate": 0.0}'` 来更改此值以禁用 EVS。 |
| **JetVLM** |  | 当存在 `video_data` 时，运行时每个片段采样八帧并将其附加到多模态请求中。 |

在构建提示时使用 `sgl.video(path, num_frames)`，可从你的 SGLang 程序中附加视频片段。

发送视频片段的兼容 OpenAI 的请求示例：

```python
import requests

url = "http://localhost:30000/v1/chat/completions"

data = {
    "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s happening in this video?"},
                {
                    "type": "video_url",
                    "video_url": {
                        "url": "https://github.com/sgl-project/sgl-test-files/raw/refs/heads/main/videos/jobs_presenting_ipod.mp4"
                    },
                },
            ],
        }
    ],
    "max_tokens": 300,
}

response = requests.post(url, json=data)
print(response.text)
```

## 使用说明

### 性能优化

对于多模态模型，你可以使用 `--keep-mm-feature-on-device` 标志来优化延迟，但代价是增加 GPU 内存使用：

- **默认行为**：多模态特征张量在处理后会被移至 CPU，以节省 GPU 内存
- **使用 `--keep-mm-feature-on-device`**：特征张量保留在 GPU 上，减少设备到主机的拷贝开销并改善延迟，但会消耗更多 GPU 内存

当你拥有充足的 GPU 内存并希望将多模态推理的延迟降到最低时，请使用此标志。

### 多模态输入限制

- **使用 `--mm-process-config '{"image":{"max_pixels":1048576},"video":{"fps":3,"max_pixels":602112,"max_frames":60}}'`**：用于设置 `image`、`video` 和 `audio` 输入限制。

这可以减少 GPU 内存使用、提升推理速度并有助于避免 OOM，但可能影响模型性能，因此请根据你的具体用例设置合适的值。目前，只有 `qwen_vl` 支持此配置。请参阅 [qwen_vl processor](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/multimodal/processors/qwen_vl.py) 以了解各参数的含义。

### 多模态模型服务中的双向注意力
**关于服务 Gemma-3 多模态模型的说明**：

正如 [Welcome Gemma 3: Google's all new multimodal, multilingual, long context open LLM](https://huggingface.co/blog/gemma3#multimodality) 中所述，Gemma-3 在 prefill 阶段对图像 token 之间采用双向注意力。目前，SGLang 仅在使用 Triton Attention Backend 时支持双向注意力。但请注意，SGLang 当前的双向注意力实现与 CUDA Graph 和 Chunked Prefill 都不兼容。

要启用双向注意力，你可以使用 `TritonAttnBackend`，同时禁用 CUDA Graph 和 Chunked Prefill。示例启动命令：
```shell
python -m sglang.launch_server \
  --model-path google/gemma-3-4b-it \
  --host 0.0.0.0 --port 30000 \
  --enable-multimodal \
  --dtype bfloat16 --triton-attention-reduce-in-fp32 \
  --attention-backend triton \ # Use Triton attention backend
  --disable-cuda-graph \ # Disable Cuda Graph
  --chunked-prefill-size -1 # Disable Chunked Prefill
```

如果需要更高的服务性能且可接受一定程度的精度损失，你可以选择使用其他 attention 后端，也可以启用 CUDA Graph 和 Chunked Prefill 等特性以获得更好的性能，但请注意，模型将回退到使用因果注意力（causal attention）而非双向注意力。
