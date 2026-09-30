# 运行时示例（Runtime examples）

下面的示例大多需要你先在另一个终端启动一个 server，然后才能运行。详细说明请见各示例代码。

## 原生 API（Native API）

* `lora.py`：如何使用 LoRA 适配器的示例。
* `multimodal_embedding.py`：如何执行[多模态 embedding](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct) 的示例。
* `openai_batch_chat.py`：如何处理 chat completions 的批量请求的示例。
* `openai_batch_complete.py`：如何处理 text completions 的批量请求的示例。
* **`openai_chat_with_response_prefill.py`**：
  一个演示如何通过启用 `continue_final_message` 参数，用 OpenAI API 来[预填充回复（prefill a response）](https://eugeneyan.com/writing/prompting/#prefill-claudes-responses)的示例。
  启用后，最后一条（不完整的）assistant 消息会被移除，其内容将作为 prefill 使用，从而让模型续写该消息，而不是开启新一轮对话。更多背景参见 [Anthropic 的 prefill 示例](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prefill-claudes-response#example-structured-data-extraction-with-prefilling)。
* `reward_model.py`：如何从奖励模型中提取分数的示例。
* `vertex_predict.py`：如何将模型部署到 [Vertex AI](https://cloud.google.com/vertex-ai?hl=en) 的示例。
* `chain_of_verification.py`：[验证链（Chain-of-Verification，CoVe）](https://arxiv.org/abs/2309.11495)的示例，用于减少幻觉。模型先起草一个答案，然后在一个**全新的、隔离的会话**中（不共享 KV cache）对其进行验证，以避免自我确认偏差，必要时再进行修正。

## 引擎（Engine）

`engine` 文件夹中的示例展示了如何使用[离线 Engine API](https://docs.sglang.io/basic_usage/offline_engine_api.html#Offline-Engine-API) 完成常见工作流。

* `custom_server.py`：如何部署自定义 server 的示例。
* `embedding.py`：如何提取 embedding 的示例。
* `launch_engine.py`：如何启动 Engine 的示例。
* `offline_batch_inference_eagle.py`：如何使用 [EAGLE](https://docs.sglang.io/advanced_features/speculative_decoding.html) 执行投机解码的示例。
* `offline_batch_inference_torchrun.py`：如何使用 [torchrun](https://pytorch.org/docs/stable/elastic/run.html) 执行推理的示例。
* `offline_batch_inference_vlm.py`：如何配合 Engine 使用 VLM 的示例。
* `offline_batch_inference.py`：如何使用 Engine 对一批示例执行推理的示例。

## 隐状态（Hidden States）

`hidden_states` 文件夹包含使用 SGLang 提取隐状态的示例。请注意，由于需要重建 CUDA Graph，这可能会降低吞吐。

* `hidden_states_engine.py`：如何使用 Engine API 提取隐状态的示例。
* `hidden_states_server.py`：如何使用 Server API 提取隐状态的示例。

## 多模态（Multimodal）

SGLang 支持多种模型架构的多模态输入。`multimodal` 文件夹包含如何使用 URL、文件或编码数据向多模态模型发起请求的示例。示例包括查询 [Llava-OneVision](multimodal/llava_onevision_server.py) 模型（图像、多图、视频）、基于 Llava 的 [Qwen-Llava](multimodal/qwen_llava_server.py) 与 [Llama3-Llava](multimodal/llama3_llava_server.py) 模型（图像、多图），以及 Mistral AI 的 [Pixtral](multimodal/pixtral_server.py)（图像、多图）。

## Token In, Token Out

`token_in_token_out` 文件夹展示了如何执行「输入 token、输出 token」形式的推理。

* `token_in_token_out_{llm|vlm}_{engine|server}.py`：展示如何针对 llm/vlm，使用 engine 或原生 API 完成 token in、token out 工作流。
