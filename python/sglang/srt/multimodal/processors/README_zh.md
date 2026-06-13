# srt/multimodal/processors

## 目录用途
各多模态模型的输入处理器（Processor）实现集合。每个文件通常对应一个多模态模型，负责将图像/视频/音频原始输入预处理为张量、计算多模态占位 token 与偏移、并与对应模型的前向流程对接。`base_processor.py` 提供所有处理器共享的抽象基类与公共数据结构。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| base_processor.py | 处理器基础设施：`BaseMultimodalProcessor` 抽象基类、`MultimodalSpecialTokens`、`BaseMultiModalProcessorOutput` 等公共数据结构与通用预处理流程。 |
| clip.py | CLIP 模型图像处理器（`ClipImageProcessor`）。 |
| deepseek_ocr.py | DeepSeek-OCR 模型处理器（`DeepseekOCRProcessor`）。 |
| deepseek_vl_v2.py | DeepSeek-VL2 模型图像处理器（`DeepseekVL2ImageProcessor`）。 |
| dots_vlm.py | dots.vlm 模型图像处理器（`DotsVLMImageProcessor`）。 |
| ernie45_vl.py | 文心 ERNIE 4.5 VL 模型图像处理器（`Ernie4_5_VLImageProcessor`）。 |
| gemma3.py | Gemma 3 模型图像处理器（`Gemma3SGLangImageProcessor`）。 |
| gemma3n.py | Gemma 3n 模型处理器（`Gemma3nSGLangProcessor`）。 |
| glm4v.py | GLM-4V 模型图像处理器（`Glm4vImageProcessor`）。 |
| glmasr.py | GLM ASR 语音识别模型处理器（`GlmAsrProcessor`）。 |
| interns1pro.py | InternS1 Pro 模型图像处理器（`InternS1_1ImageProcessor`，继承自 Qwen-VL 处理器）。 |
| internvl.py | InternVL 系列模型处理器（`InternVLProcessor`）。 |
| janus_pro.py | Janus-Pro 模型图像处理器（`JanusProImageProcessor`）。 |
| kimi_k25.py | Kimi-K2.5 VL 模型图像处理器（`KimiK2_5VLImageProcessor`）。 |
| kimi_vl.py | Kimi-VL 模型图像处理器（`KimiVLImageProcessor`）。 |
| lightonocr.py | LightOnOCR 模型处理器（`LightOnOCRProcessor`，继承自 Pixtral 处理器）。 |
| llava.py | LLaVA / LLaVA-NeXT 系列模型处理器（`LlavaImageProcessor`、`LlavaMultimodalProcessor`）。 |
| midashenglm.py | MiDashengLM 音频多模态模型处理器（`MiDashengLMMultimodalProcessor`）。 |
| minicpm.py | MiniCPM-V 系列多模态模型处理器（`MiniCPMMultimodalProcessor`）。 |
| mlama.py | Mllama（Llama 3.2 Vision）模型图像处理器（`MllamaImageProcessor`）。 |
| mllama4.py | Llama 4（Mllama4）模型图像处理器（`Mllama4ImageProcessor`）。 |
| nano_nemotron_vl.py | Nano Nemotron VL 模型图像处理器（`NanoNemotronVLImageProcessor`）。 |
| nvila.py | NVILA 多模态模型处理器（`NVILAMultimodalProcessor`）。 |
| paddleocr_vlm.py | PaddleOCR-VL 模型图像处理器（`PaddleOCRVLImageProcessor`，继承自 Qwen-VL 处理器）。 |
| phi4mm.py | Phi-4 多模态模型处理器（`Phi4MMMultimodalProcessor` 及适配器 `Phi4MMProcessorAdapter`）。 |
| pixtral.py | Pixtral 模型处理器（`PixtralProcessor`）。 |
| points_v15_chat.py | POINTS v1.5 Chat 模型处理器（`POINTSV15ChatProcessor`，继承自 Qwen-VL 处理器）。 |
| qwen_audio.py | Qwen2-Audio 音频多模态模型处理器（`Qwen2AudioMultimodalProcessor`）。 |
| qwen_vl.py | Qwen-VL / Qwen2-VL / Qwen2.5-VL 系列图像处理器（`QwenVLImageProcessor`，多个模型的处理器基类）。 |
| sarashina2_vision.py | Sarashina2-Vision 模型处理器（`Sarashina2VisionProcessor`）。 |
| step3_vl.py | Step3-VL 模型处理器（`Step3VLImageProcessor` 及视觉预处理组件 `Step3VisionProcessor`、`ImagePatcher` 等）。 |
| transformers_auto.py | 基于 HuggingFace AutoProcessor 的通用多模态处理器（`TransformersAutoMultimodalProcessor`），用于未单独实现处理器的模型。 |
| whisper.py | Whisper 语音模型处理器（`WhisperProcessor`）。 |
