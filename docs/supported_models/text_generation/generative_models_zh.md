# 大语言模型（Large Language Models）

这些模型接受文本输入并产生文本输出（例如聊天补全）。它们主要是大语言模型（LLM），其中一些采用混合专家（MoE）架构以实现扩展。

## 示例启动命令

```shell
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-1B-Instruct \  # example HF/local path
  --host 0.0.0.0 \
  --port 30000 \
```

## 支持的模型

下表汇总了支持的模型。

如果你不确定某个特定架构是否已实现，可以通过 GitHub 进行搜索。例如，要搜索 `Qwen3ForCausalLM`，请在 GitHub 搜索栏中使用以下表达式：

```
repo:sgl-project/sglang path:/^python\/sglang\/srt\/models\// Qwen3ForCausalLM
```

| Model Family (Variants)             | Example HuggingFace Identifier                     | Description                                                                            |
|-------------------------------------|--------------------------------------------------|----------------------------------------------------------------------------------------|
| **DeepSeek** (v1, v2, v3/R1)        | `deepseek-ai/DeepSeek-R1`                        | 一系列经强化学习训练、针对推理优化的先进模型（包括一个 671B MoE）；在复杂推理、数学和代码任务上表现顶尖。[SGLang 提供 Deepseek v3/R1 模型专用优化](../../basic_usage/deepseek_v3.md) 和[推理解析器（Reasoning Parser）](../../advanced_features/separate_reasoning.ipynb)|
| **Kimi K2** (Thinking, Instruct)    | `moonshotai/Kimi-K2-Instruct`                    | 月之暗面（Moonshot AI）的 1 万亿参数 MoE 模型（32B 激活），上下文 128K–256K；具备最先进的智能体智能，可在 200–300 次连续工具调用中稳定保持长程自主性。采用 MLA 注意力和原生 INT4 量化。[参见推理解析器文档](../../advanced_features/separate_reasoning.ipynb)|
| **Kimi Linear** (48B-A3B)           | `moonshotai/Kimi-Linear-48B-A3B-Instruct`        | 月之暗面的混合线性注意力模型（总计 48B，3B 激活），上下文 1M token；采用 Kimi Delta Attention（KDA），相比全注意力可实现最高 6 倍的解码加速和 75% 的 KV 缓存削减。 |
| **GPT-OSS**       | `openai/gpt-oss-20b`, `openai/gpt-oss-120b`       | OpenAI 最新的 GPT-OSS 系列，用于复杂推理、智能体任务和多样化的开发者用例。|
| **Qwen** (3.5, 3, 3MoE, 3Next, 2.5, 2 series)       | `Qwen/Qwen3.5-397B-A17B`, `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-30B-A3B`, `Qwen/Qwen3-Next-80B-A3B-Instruct`      | 阿里巴巴最新的 Qwen3 系列，用于复杂推理、语言理解和生成任务；支持 MoE 变体以及上一代 2.5、2 等。[SGLang 提供 Qwen3 专用的推理解析器](../../advanced_features/separate_reasoning.ipynb)|
| **Llama** (2, 3.x, 4 series)        | `meta-llama/Llama-4-Scout-17B-16E-Instruct`       | Meta 的开放 LLM 系列，参数规模从 7B 到 400B（Llama 2、3 和新的 Llama 4），性能广受认可。[SGLang 提供 Llama-4 模型专用优化](../../basic_usage/llama4.md)  |
| **Mistral** (Mixtral, NeMo, Small3) | `mistralai/Mistral-7B-Instruct-v0.2`             | Mistral AI 推出的开放 7B LLM，性能强劲；并扩展为 MoE（"Mixtral"）和 NeMo Megatron 变体以实现更大规模。 |
| **Gemma** (v1, v2, v3)              | `google/gemma-3-1b-it`                            | Google 的高效多语言模型家族（1B–27B）；Gemma 3 提供 128K 上下文窗口，其较大（4B+）变体支持视觉输入。 |
| **Phi** (Phi-1.5, Phi-2, Phi-3, Phi-4, Phi-MoE series) | `microsoft/Phi-4-multimodal-instruct`, `microsoft/Phi-3.5-MoE-instruct` | 微软的 Phi 系列小模型（1.3B–5.6B）；Phi-4-multimodal（5.6B）处理文本、图像和语音，Phi-4-mini 是一个高准确率的文本模型，Phi-3.5-MoE 是一个混合专家模型。 |
| **MiniCPM** (v3, 4B)               | `openbmb/MiniCPM3-4B`                            | OpenBMB 面向边缘设备的紧凑型 LLM 系列；MiniCPM 3（4B）在文本任务上达到 GPT-3.5 级别的结果。 |
| **OLMo** (2, 3) | `allenai/OLMo-3-1125-32B`, `allenai/OLMo-3-32B-Think`, `allenai/OLMo-2-1124-7B-Instruct` | Allen AI 的开放语言模型系列，旨在推动语言模型科学的发展。 |
| **OLMoE** (Open MoE)               | `allenai/OLMoE-1B-7B-0924`                       | Allen AI 的开放混合专家模型（总计 7B，1B 激活参数），通过稀疏专家激活提供最先进的结果。 |
| **MiniMax-M2** (M2, M2.1, M2.5)               | `MiniMaxAI/MiniMax-M2.5`, `MiniMaxAI/MiniMax-M2.1`, `MiniMaxAI/MiniMax-M2` | MiniMax 面向编码与智能体工作流的 SOTA LLM。 |
| **StableLM** (3B, 7B)               | `stabilityai/stablelm-tuned-alpha-7b`            | StabilityAI 的早期开源 LLM（3B 和 7B），用于通用文本生成；是一个具备基本指令遵循能力的演示模型。 |
| **Command-(R,A)** (Cohere)              | `CohereLabs/c4ai-command-r-v01`, `CohereLabs/c4ai-command-r7b-12-2024`, `CohereLabs/c4ai-command-a-03-2025`                 | Cohere 的开放对话 LLM（Command 系列），针对长上下文、检索增强生成和工具使用进行了优化。 |
| **DBRX** (Databricks)              | `databricks/dbrx-instruct`                       | Databricks 的 132B 参数 MoE 模型（36B 激活），在 12T token 上训练；作为完全开放的基础模型，质量可与 GPT-3.5 媲美。 |
| **Grok** (xAI)                     | `xai-org/grok-1`                                | xAI 的 grok-1 模型，以其庞大的规模（314B 参数）和高质量著称；已集成到 SGLang 中以实现高性能推理。 |
| **ChatGLM** (GLM-130B family)       | `THUDM/chatglm2-6b`                              | 智谱 AI 的双语对话模型（6B），擅长中英文对话；针对对话质量和对齐进行了微调。 |
| **InternLM 2** (7B, 20B)           | `internlm/internlm2-7b`                          | 商汤科技的新一代 InternLM（7B 和 20B），提供强大的推理能力和超长上下文支持（最高 200K token）。 |
| **ExaONE 3** (Korean-English)      | `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct`           | LG AI Research 的韩英模型（7.8B），在 8T token 上训练；提供高质量的双语理解和生成。 |
| **Baichuan 2** (7B, 13B)           | `baichuan-inc/Baichuan2-13B-Chat`                | 百川智能的第二代中英文 LLM（7B/13B），性能更优并提供开放的商业许可。 |
| **XVERSE** (MoE)                   | `xverse/XVERSE-MoE-A36B`                         | 元象的开放 MoE LLM（XVERSE-MoE-A36B：总计 255B，36B 激活），支持约 40 种语言；通过专家路由实现 100B+ dense 级别的性能。 |
| **SmolLM** (135M–1.7B)            | `HuggingFaceTB/SmolLM-1.7B`                      | Hugging Face 的超小型 LLM 系列（135M–1.7B 参数），提供出人意料的强劲结果，使移动/边缘设备上的高级 AI 成为可能。 |
| **GLM-4** (Multilingual 9B)        | `ZhipuAI/glm-4-9b-chat`                          | 智谱的 GLM-4 系列（最高 9B 参数）——开放的多语言模型，支持 1M-token 上下文，甚至包含一个 5.6B 多模态变体（Phi-4V）。 |
| **MiMo** (7B series)               | `XiaomiMiMo/MiMo-7B-RL`                         | 小米针对推理优化的模型系列，利用多 token 预测（Multiple-Token Prediction）实现更快的推理。 |
| **ERNIE-4.5** (4.5, 4.5MoE series) | `baidu/ERNIE-4.5-21B-A3B-PT`                    | 百度的 ERNIE-4.5 系列，包含激活参数为 47B 和 3B 的 MoE 模型，最大的模型总参数达 424B，还包含一个 0.3B 的 dense 模型。 |
| **Arcee AFM-4.5B**               | `arcee-ai/AFM-4.5B-Base`                         | Arcee 面向真实世界可靠性和边缘部署的基础模型系列。 |
| **Persimmon** (8B)               | `adept/persimmon-8b-chat`                         | Adept 的开放 8B 模型，具有 16K 上下文窗口和快速推理；为广泛可用性而训练，并采用 Apache 2.0 许可。 |
| **Solar** (10.7B)               | `upstage/SOLAR-10.7B-Instruct-v1.0`                         | Upstage 的 10.7B 参数模型，针对指令遵循任务进行了优化。该架构采用深度扩展（depth-up scaling）方法，提升了模型性能。 |
| **Tele FLM** (52B-1T)               | `CofeAI/Tele-FLM`                         | 智源（BAAI）与 TeleAI 的多语言模型，提供 520 亿和 1 万亿参数变体。它是一个仅解码器的 transformer，在约 2T token 上训练 |
| **Ling** (16.8B–290B) | `inclusionAI/Ling-lite`, `inclusionAI/Ling-plus` | InclusionAI 的开放 MoE 模型。Ling-Lite 总计 16.8B / 2.75B 激活参数，Ling-Plus 总计 290B / 28.8B 激活参数。它们专为 NLP 和复杂推理任务的高性能而设计。 |
| **Granite 3.0, 3.1** (IBM)               | `ibm-granite/granite-3.1-8b-instruct`                          | IBM 的开放 dense 基础模型，针对推理、代码和商业 AI 用例进行了优化。已与 Red Hat 和 watsonx 系统集成。 |
| **Granite 3.0 MoE** (IBM)               | `ibm-granite/granite-3.0-3b-a800m-instruct`                          | IBM 的混合专家模型，以高性价比提供强劲性能。MoE 专家路由专为企业级大规模部署而设计。 |
| **GPT-J** (6B)                    | `EleutherAI/gpt-j-6b`                             | EleutherAI 的类 GPT-2 因果语言模型（6B），在 [Pile](https://pile.eleuther.ai/) 数据集上训练。 |
| **Orion** (14B)               | `OrionStarAI/Orion-14B-Base`                         | OrionStarAI 推出的一系列开源多语言大语言模型，在包含中文、英文、日文、韩文等的 2.5T token 多语言语料上预训练，在这些语言上表现出色。 |
| **Llama Nemotron Super** (v1, v1.5, NVIDIA) | `nvidia/Llama-3_3-Nemotron-Super-49B-v1`, `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5` | [NVIDIA Nemotron](https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/) 多模态模型家族提供专为企业级 AI 智能体设计的最先进推理模型。 |
| **Llama Nemotron Ultra** (v1, NVIDIA) | `nvidia/Llama-3_1-Nemotron-Ultra-253B-v1` | [NVIDIA Nemotron](https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/) 多模态模型家族提供专为企业级 AI 智能体设计的最先进推理模型。 |
| **NVIDIA Nemotron Nano 2.0** | `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | [NVIDIA Nemotron](https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/) 多模态模型家族提供专为企业级 AI 智能体设计的最先进推理模型。`Nemotron-Nano-9B-v2` 是一个混合 Mamba-Transformer 语言模型，旨在提升推理工作负载的吞吐量，同时相比同等规模的模型达到最先进的准确率。 |
| **NVIDIA Nemotron 3 Super** (NVIDIA) | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | [NVIDIA Nemotron](https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/) 3 Super 是一个 120B 参数的 MoE 模型（12B 激活），为企业 AI 智能体提供高质量的推理和生成。 |
| **NVIDIA Nemotron 3 Nano** (NVIDIA) | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | [NVIDIA Nemotron](https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/) 3 Nano 是一个紧凑型模型，专为高效的边缘和企业部署而设计，具备强大的推理能力。 |
| **StarCoder2** (3B-15B) | `bigcode/starcoder2-7b` | StarCoder2 是一系列专门用于代码生成和理解的开放大语言模型（LLM）。它是 StarCoder 的后继者，由 BigCode 项目（Hugging Face、ServiceNow Research 及其他贡献者的合作项目）共同开发。 |
| **Jet-Nemotron** | `jet-ai/Jet-Nemotron-2B` | Jet-Nemotron 是一个全新的混合架构语言模型家族，在超越最先进开源全注意力语言模型的同时，实现了显著的效率提升。 |
| **Trinity** (Nano, Mini) | `arcee-ai/Trinity-Mini` | Arcee 的基础 MoE Trinity 模型家族，以 Apache 2.0 开放权重发布。 |
| **Falcon-H1** (0.5B–34B) | `tiiuae/Falcon-H1-34B-Instruct` | TII 的混合 Mamba-Transformer 架构，结合注意力和状态空间模型以实现高效的长上下文推理。 |
| **Hunyuan-Large** (389B, MoE) | `tencent/Tencent-Hunyuan-Large` | 腾讯的开源 MoE 模型，总计 389B / 52B 激活参数，采用跨层注意力（Cross-Layer Attention, CLA）以提升效率。 |
| **IBM Granite 4.0 (Hybrid, Dense)** | `ibm-granite/granite-4.0-h-micro`, `ibm-granite/granite-4.0-micro` | IBM Granite 4.0 micro 模型：混合 Mamba–MoE（`h-micro`）和 dense（`micro`）变体。面向企业的推理模型 |
| **Sarvam 2** (30B-A2B, 105B-A10B) | `sarvamai/sarvam-2` | Sarvam 的混合专家模型。105B 变体使用 MLA（多头潜在注意力），30B 变体使用 GQA，二者均配备 128 个路由专家。 |
