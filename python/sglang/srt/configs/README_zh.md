# srt/configs

## 目录用途
本目录集中存放 SGLang 推理引擎使用的各类配置定义。一部分是引擎自身的运行时配置（如模型加载、设备、量化、模型元信息等），另一部分是各类模型（尤其是 HuggingFace 尚未原生支持或需要定制的模型）的 `PretrainedConfig` 子类及配套的处理器/工具，使引擎能够正确解析模型结构并构建对应的执行流程。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 汇总导出各模型配置类，构成 configs 包的公共入口 |
| `model_config.py` | 核心模型元信息类 `ModelConfig`，封装注意力架构 `AttentionArch`、模型实现 `ModelImpl` 等，供引擎读取模型结构与超参 |
| `load_config.py` | 权重加载配置 `LoadConfig` 与加载格式枚举 `LoadFormat` |
| `device_config.py` | 设备配置类 `DeviceConfig`，描述运行设备信息 |
| `modelopt_config.py` | ModelOpt 量化相关配置 `ModelOptConfig` |
| `update_config.py` | 配置调整工具函数（如根据 TP 大小、权重块大小调整头数、中间层尺寸等对齐逻辑） |
| `mamba_utils.py` | Mamba2/KimiLinear 等线性/状态空间层的状态形状与缓存参数工具（`Mamba2StateShape`、`Mamba2CacheParams` 等） |
| `utils.py` | 配置/处理器注册工具，提供 `register_image_processor`、`register_processor` |
| `afmoe.py` | Afmoe 模型配置 `AfmoeConfig` |
| `bailing_hybrid.py` | Bailing 混合架构模型配置 `BailingHybridConfig` 及层类型枚举 |
| `chatglm.py` | ChatGLM 模型配置 `ChatGLMConfig` |
| `dbrx.py` | DBRX 模型配置，含注意力、FFN 与整体配置 `DbrxConfig` |
| `deepseek_ocr.py` | DeepSeek-OCR 视觉/投影/语言模型配置及 `DeepseekOCRProcessor` 处理器 |
| `deepseekvl2.py` | DeepSeek-VL2 视觉语言模型配置及 `DeepseekVLV2Processor` 处理器 |
| `dots_ocr.py` | dots.ocr 模型配置 `DotsOCRConfig` 及视觉处理器 |
| `dots_vlm.py` | dots 视觉语言模型配置 `DotsVLMConfig` 及处理器 |
| `exaone.py` | EXAONE 模型配置 `ExaoneConfig` |
| `falcon_h1.py` | Falcon-H1 模型配置 `FalconH1Config` |
| `granitemoehybrid.py` | Granite MoE 混合模型配置 `GraniteMoeHybridConfig` |
| `internvl.py` | InternVL 系列配置（InternLM2、视觉编码器、Chat 配置及分词器） |
| `janus_pro.py` | Janus-Pro 多模态模型配置、视觉/生成对齐配置及 `VLChatProcessor` |
| `jet_nemotron.py` | Jet-Nemotron 模型配置 `JetNemotronConfig` 及块配置 |
| `jet_vlm.py` | Jet 视觉语言模型配置 `JetVLMConfig` |
| `kimi_k25.py` | Kimi K2.5 模型配置及视觉配置 |
| `kimi_linear.py` | Kimi Linear 模型配置 `KimiLinearConfig` |
| `kimi_vl.py` | Kimi-VL 视觉语言模型配置 `KimiVLConfig` |
| `kimi_vl_moonvit.py` | Kimi-VL 的 MoonViT 视觉编码器配置 `MoonViTConfig` |
| `lfm2.py` | LFM2 模型配置 `Lfm2Config` |
| `lfm2_moe.py` | LFM2-MoE 模型配置 `Lfm2MoeConfig` |
| `longcat_flash.py` | LongCat-Flash 模型配置 `LongcatFlashConfig` |
| `nano_nemotron_vl.py` | Nemotron-H Nano VL V2 视觉语言模型配置 |
| `nemotron_h.py` | Nemotron-H 模型配置 `NemotronHConfig` |
| `olmo3.py` | OLMo3 模型配置 `Olmo3Config` 及层类型枚举 |
| `points_v15_chat.py` | POINTS V1.5 Chat 模型配置 `POINTSV15ChatConfig` |
| `qwen3_5.py` | Qwen3.5 系列配置（视觉、文本及 MoE 变体） |
| `qwen3_next.py` | Qwen3-Next 模型配置 `Qwen3NextConfig` 及层类型枚举 |
| `qwen3_omni.py` | Qwen3-Omni MoE 全模态配置（音频、视觉、文本、Thinker、Talker、Code2Wav 等） |
| `qwen3_vl.py` | Qwen3-VL 系列配置（视觉、文本及 MoE 变体） |
| `radio.py` | RADIO 视觉骨干配置 `RadioConfig` |
| `step3_vl.py` | Step3-VL 视觉语言模型配置（视觉编码器、文本及整体配置） |
| `step3p5.py` | Step3.5 模型配置 `Step3p5Config` |
