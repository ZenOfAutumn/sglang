# srt/models

## 目录用途
本目录是 SGLang 中所有受支持模型架构的实现集合。每个 `xxx.py` 通常对应一个或一族模型，文件内通过模块级变量 `EntryClass`（单个类或类列表）声明该文件向引擎暴露的入口模型类。`registry.py` 在启动时扫描本目录，依据各文件的 `EntryClass` 和 HuggingFace 配置中的 `architectures` 字段把模型架构名注册到模型注册表，从而在加载权重时自动匹配并实例化对应的模型类。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `registry.py` | 模型注册表基础设施：扫描本目录、读取各文件 `EntryClass`，按架构名注册并解析为模型类。 |
| `utils.py` | 模型实现共用的工具函数（权重加载、参数命名映射等辅助逻辑）。 |
| `transformers.py` | 对 HuggingFace `transformers` 模型的通用包装回退实现，用于未单独适配的架构。 |
| `afmoe.py` | AFMoE（`AfmoeForCausalLM`）MoE 模型架构实现。 |
| `apertus.py` | Apertus（`ApertusForCausalLM`）模型架构实现。 |
| `arcee.py` | Arcee（`ArceeForCausalLM`）模型架构实现。 |
| `baichuan.py` | 百川 Baichuan（`BaichuanForCausalLM`）模型架构实现。 |
| `bailing_moe.py` | 百灵 Bailing MoE 系列（`BailingMoEForCausalLM` 等多版本）模型架构实现。 |
| `bailing_moe_linear.py` | 百灵 Bailing MoE 线性注意力变体模型架构实现。 |
| `bailing_moe_nextn.py` | 百灵 Bailing MoE 的 NextN（多token预测/投机解码）模块实现。 |
| `bert.py` | BERT 系列（`BertModel`、`Contriever`、序列分类）编码器模型实现。 |
| `chatglm.py` | ChatGLM（`ChatGLMModel`）模型架构实现。 |
| `clip.py` | CLIP（`CLIPModel`）视觉-文本编码器实现。 |
| `commandr.py` | Cohere Command-R 系列（`CohereForCausalLM`、`Cohere2ForCausalLM`）模型实现。 |
| `dbrx.py` | DBRX（`DbrxForCausalLM`）MoE 模型架构实现。 |
| `deepseek.py` | DeepSeek（V1，`DeepseekForCausalLM`）模型架构实现。 |
| `deepseek_janus_pro.py` | DeepSeek Janus-Pro（`MultiModalityCausalLM`）多模态模型实现。 |
| `deepseek_nextn.py` | DeepSeek V3 的 NextN（`DeepseekV3ForCausalLMNextN`）多token预测模块实现。 |
| `deepseek_ocr.py` | DeepSeek-OCR（`DeepseekOCRForCausalLM`）OCR 多模态模型实现。 |
| `deepseek_v2.py` | DeepSeek V2/V3/V3.2（`DeepseekV2/V3/V32ForCausalLM`）核心 MLA/MoE 模型架构实现。 |
| `deepseek_vl2.py` | DeepSeek-VL2（`DeepseekVL2ForCausalLM`）视觉语言模型实现。 |
| `dots_ocr.py` | dots.ocr（`DotsOCRForCausalLM`）OCR 模型实现。 |
| `dots_vlm.py` | dots 视觉语言模型（`DotsVLMForCausalLM`）实现。 |
| `dots_vlm_vit.py` | dots VLM 的视觉编码器（ViT）实现，供 `dots_vlm.py` 使用。 |
| `ernie4.py` | 百度文心 ERNIE 4.5（`Ernie4_5_Moe/ForCausalLM`）模型架构实现。 |
| `ernie45_moe_vl.py` | 文心 ERNIE 4.5 MoE 视觉语言模型实现。 |
| `ernie45_vl.py` | 文心 ERNIE 4.5 VL MoE（`Ernie4_5_VLMoeForConditionalGeneration`）视觉语言模型实现。 |
| `ernie4_eagle.py` | 文心 ERNIE 4.5 MoE 的 EAGLE/MTP（`Ernie4_5_MoeForCausalLMMTP`）投机解码模块实现。 |
| `exaone.py` | LG EXAONE（`ExaoneForCausalLM`）模型架构实现。 |
| `exaone4.py` | EXAONE 4（`Exaone4ForCausalLM`）模型架构实现。 |
| `exaone_moe.py` | EXAONE MoE（`ExaoneMoEForCausalLM`）模型架构实现。 |
| `exaone_moe_mtp.py` | EXAONE MoE 的 MTP（`ExaoneMoEForCausalLMMTP`）多token预测模块实现。 |
| `falcon_h1.py` | Falcon-H1（`FalconH1ForCausalLM`）混合架构模型实现。 |
| `gemma.py` | Google Gemma（`GemmaForCausalLM`）模型架构实现。 |
| `gemma2.py` | Gemma 2（`Gemma2ForCausalLM`）模型架构实现。 |
| `gemma2_reward.py` | Gemma 2 奖励/序列分类模型（`Gemma2ForSequenceClassification`）实现。 |
| `gemma3_causal.py` | Gemma 3 文本因果语言模型（`Gemma3ForCausalLM`）实现。 |
| `gemma3_mm.py` | Gemma 3 多模态（`Gemma3ForConditionalGeneration`）模型实现。 |
| `gemma3n_audio.py` | Gemma 3n 的音频编码器组件实现。 |
| `gemma3n_causal.py` | Gemma 3n 文本因果语言模型（`Gemma3nForCausalLM`）实现。 |
| `gemma3n_mm.py` | Gemma 3n 多模态（`Gemma3nForConditionalGeneration`）模型实现。 |
| `glm4.py` | 智谱 GLM-4（`Glm4ForCausalLM`）模型架构实现。 |
| `glm4_moe.py` | GLM-4 MoE（`Glm4MoeForCausalLM`、`GlmMoeDsaForCausalLM`）模型架构实现。 |
| `glm4_moe_lite.py` | GLM-4 MoE Lite（`Glm4MoeLiteForCausalLM`）轻量版模型实现。 |
| `glm4_moe_nextn.py` | GLM-4 MoE 的 NextN（`Glm4MoeForCausalLMNextN`）多token预测模块实现。 |
| `glm4v.py` | GLM-4V（`Glm4vForConditionalGeneration`）视觉语言模型实现。 |
| `glm4v_moe.py` | GLM-4V MoE（`Glm4vMoeForConditionalGeneration`）视觉语言 MoE 模型实现。 |
| `glm_ocr.py` | GLM-OCR（`GlmOcrForConditionalGeneration`）OCR 多模态模型实现。 |
| `glm_ocr_nextn.py` | GLM-OCR 的 NextN（`GlmOcrForConditionalGenerationNextN`）多token预测模块实现。 |
| `glmasr.py` | GLM-ASR（`GlmAsrForConditionalGeneration`）语音识别多模态模型实现。 |
| `gpt2.py` | GPT-2（`GPT2LMHeadModel`）模型架构实现。 |
| `gpt_bigcode.py` | GPT-BigCode/StarCoder（`GPTBigCodeForCausalLM`）代码模型实现。 |
| `gpt_j.py` | GPT-J（`GPTJForCausalLM`）模型架构实现。 |
| `gpt_oss.py` | GPT-OSS（`GptOssForCausalLM`）模型架构实现。 |
| `granite.py` | IBM Granite（`GraniteForCausalLM`）模型架构实现。 |
| `granitemoe.py` | Granite MoE（`GraniteMoeForCausalLM`）模型架构实现。 |
| `granitemoehybrid.py` | Granite MoE Hybrid（`GraniteMoeHybridForCausalLM`）混合架构模型实现。 |
| `grok.py` | xAI Grok-1（`Grok1ForCausalLM`、`Grok1ModelForCausalLM`）MoE 模型实现。 |
| `hunyuan.py` | 腾讯混元 HunYuan（`HunYuanMoEV1`/`HunYuanDenseV1ForCausalLM`）模型实现。 |
| `idefics2.py` | Idefics2 视觉语言模型组件实现（多模态视觉编码相关）。 |
| `internlm2.py` | InternLM2（`InternLM2ForCausalLM`）模型架构实现。 |
| `internlm2_reward.py` | InternLM2 奖励模型（`InternLM2ForRewardModel`）实现。 |
| `interns1.py` | InternS1（`InternS1ForConditionalGeneration`）多模态模型实现。 |
| `interns1pro.py` | InternS1-Pro（`InternS1ProForConditionalGeneration`）多模态模型实现。 |
| `internvl.py` | InternVL（`InternVLChatModel`）视觉语言模型实现。 |
| `iquest_loopcoder.py` | iQuest LoopCoder（`IQuestLoopCoderForCausalLM`）代码模型实现。 |
| `jet_nemotron.py` | Jet-Nemotron（`JetNemotronForCausalLM`）模型架构实现。 |
| `jet_vlm.py` | Jet 视觉语言模型（`JetVLMForConditionalGeneration`）实现。 |
| `kimi_k25.py` | Kimi K2.5（`KimiK25ForConditionalGeneration`）多模态模型实现。 |
| `kimi_linear.py` | Kimi 线性注意力模型（`KimiLinearForCausalLM`）实现。 |
| `kimi_vl.py` | Kimi-VL（`KimiVLForConditionalGeneration`）视觉语言模型实现。 |
| `kimi_vl_moonvit.py` | Kimi-VL 的 MoonViT 视觉编码器实现，供 `kimi_vl.py` 使用。 |
| `lfm2.py` | Liquid LFM2（`Lfm2ForCausalLM`）模型架构实现。 |
| `lfm2_moe.py` | Liquid LFM2 MoE（`Lfm2MoeForCausalLM`）模型架构实现。 |
| `lightonocr.py` | LightOnOCR（`LightOnOCRForConditionalGeneration`）OCR 多模态模型实现。 |
| `llada2.py` | LLaDA2 MoE（`LLaDA2MoeModelLM`）扩散语言模型实现。 |
| `llama.py` | Meta LLaMA 系列（`LlamaForCausalLM` 等）核心模型架构实现。 |
| `llama4.py` | LLaMA 4（`Llama4ForCausalLM`）文本模型架构实现。 |
| `llama_classification.py` | LLaMA 分类模型（`LlamaForClassification`）实现。 |
| `llama_eagle.py` | LLaMA 的 EAGLE 投机解码草稿模型（`LlamaForCausalLMEagle`）实现。 |
| `llama_eagle3.py` | LLaMA 的 EAGLE3 投机解码草稿模型（`LlamaForCausalLMEagle3`）实现。 |
| `llama_embedding.py` | LLaMA/Mistral 嵌入模型（`LlamaEmbeddingModel`、`MistralModel`）实现。 |
| `llama_reward.py` | LLaMA 奖励模型实现。 |
| `llava.py` | LLaVA 系列视觉语言模型实现。 |
| `llavavid.py` | LLaVA-Video（`LlavaVidForCausalLM`）视频多模态模型实现。 |
| `longcat_flash.py` | LongCat-Flash（`LongcatFlashForCausalLM`）模型架构实现。 |
| `longcat_flash_nextn.py` | LongCat-Flash 的 NextN（`LongcatFlashForCausalLMNextN`）多token预测模块实现。 |
| `midashenglm.py` | MiDashengLM（`MiDashengLMModel`）音频多模态模型实现。 |
| `mimo.py` | 小米 MiMo（`MiMoForCausalLM`）模型架构实现。 |
| `mimo_mtp.py` | MiMo 的 MTP（`MiMoMTP`）多token预测模块实现。 |
| `mimo_v2_flash.py` | MiMo V2 Flash（`MiMoV2FlashForCausalLM`）模型架构实现。 |
| `mimo_v2_flash_nextn.py` | MiMo V2 Flash 的 MTP（`MiMoV2MTP`）多token预测模块实现。 |
| `mindspore.py` | MindSpore 后端模型（`MindSporeForCausalLM`）适配实现。 |
| `minicpm.py` | MiniCPM（`MiniCPMForCausalLM`）模型架构实现。 |
| `minicpm3.py` | MiniCPM3（`MiniCPM3ForCausalLM`）模型架构实现。 |
| `minicpmo.py` | MiniCPM-O（`MiniCPMO`）全模态模型实现。 |
| `minicpmv.py` | MiniCPM-V（`MiniCPMV`）视觉语言模型实现。 |
| `minimax_m2.py` | MiniMax M2（`MiniMaxM2ForCausalLM`）模型架构实现。 |
| `ministral3.py` | Ministral 3（`Ministral3ForCausalLM`）模型架构实现。 |
| `mistral.py` | Mistral（`MistralForCausalLM`、`Mistral3ForConditionalGeneration`）模型架构实现。 |
| `mistral_large_3.py` | Mistral Large 3（`MistralLarge3ForCausalLM`）模型架构实现。 |
| `mistral_large_3_eagle.py` | Mistral Large 3 的 EAGLE 投机解码草稿模型实现。 |
| `mixtral.py` | Mixtral（`MixtralForCausalLM`）MoE 模型架构实现。 |
| `mixtral_quant.py` | Mixtral 量化版（`QuantMixtralForCausalLM`）模型实现。 |
| `mllama.py` | LLaMA 3.2 多模态 Mllama（`MllamaForConditionalGeneration`）实现。 |
| `mllama4.py` | LLaMA 4 多模态（`Llama4ForConditionalGeneration`）实现。 |
| `nano_nemotron_vl.py` | Nano Nemotron VL（`NemotronH_Nano_VL_V2`）视觉语言模型实现。 |
| `nemotron_h.py` | NVIDIA Nemotron-H（`NemotronHForCausalLM`）混合架构模型实现。 |
| `nemotron_h_mtp.py` | Nemotron-H 的 MTP（`NemotronHForCausalLMMTP`）多token预测模块实现。 |
| `nemotron_nas.py` | Nemotron NAS / DeciLM（`DeciLMForCausalLM`）模型架构实现。 |
| `nvila.py` | NVILA（`NVILAForConditionalGeneration`）视觉语言模型实现。 |
| `nvila_lite.py` | NVILA-Lite（`NVILALiteForConditionalGeneration`）视觉语言模型实现。 |
| `olmo.py` | AllenAI OLMo（`OlmoForCausalLM`）模型架构实现。 |
| `olmo2.py` | OLMo 2（`Olmo2ForCausalLM`）模型架构实现。 |
| `olmoe.py` | OLMoE（`OlmoeForCausalLM`）MoE 模型架构实现。 |
| `opt.py` | Meta OPT（`OPTForCausalLM`）模型架构实现。 |
| `orion.py` | Orion（`OrionForCausalLM`）模型架构实现。 |
| `paddleocr_vl.py` | PaddleOCR-VL（`PaddleOCRVLForConditionalGeneration`）OCR 视觉语言模型实现。 |
| `persimmon.py` | Persimmon（`PersimmonForCausalLM`）模型架构实现。 |
| `phi.py` | 微软 Phi（`PhiForCausalLM`）模型架构实现。 |
| `phi3_small.py` | Phi-3-Small（`Phi3SmallForCausalLM`）模型架构实现。 |
| `phi4mm.py` | Phi-4 多模态（`Phi4MMForCausalLM`）模型实现。 |
| `phi4mm_audio.py` | Phi-4 多模态的音频编码器组件实现。 |
| `phi4mm_utils.py` | Phi-4 多模态实现的共用工具函数。 |
| `phimoe.py` | Phi MoE（`PhiMoEForCausalLM`）模型架构实现。 |
| `pixtral.py` | Pixtral（`PixtralForConditionalGeneration`、`PixtralVisionModel`）视觉语言模型实现。 |
| `points_v15_chat.py` | POINTS V1.5 Chat（`POINTSV15ChatModel`）多模态模型实现。 |
| `qwen.py` | 通义千问 Qwen（`QWenLMHeadModel`）模型架构实现。 |
| `qwen2.py` | Qwen2（`Qwen2ForCausalLM`）模型架构实现。 |
| `qwen2_5_vl.py` | Qwen2.5-VL（`Qwen2_5_VLForConditionalGeneration`）视觉语言模型实现。 |
| `qwen2_audio.py` | Qwen2-Audio（`Qwen2AudioForConditionalGeneration`）音频多模态模型实现。 |
| `qwen2_classification.py` | Qwen2 分类模型实现。 |
| `qwen2_eagle.py` | Qwen2 的 EAGLE 投机解码草稿模型（`Qwen2ForCausalLMEagle`）实现。 |
| `qwen2_moe.py` | Qwen2 MoE（`Qwen2MoeForCausalLM`）模型架构实现。 |
| `qwen2_rm.py` | Qwen2 奖励模型（reward model）实现。 |
| `qwen2_vl.py` | Qwen2-VL（`Qwen2VLForConditionalGeneration`）视觉语言模型实现。 |
| `qwen3.py` | Qwen3（`Qwen3ForCausalLM`）模型架构实现。 |
| `qwen3_5.py` | Qwen3.5（`Qwen3_5Moe/ForConditionalGeneration`）模型架构实现。 |
| `qwen3_5_mtp.py` | Qwen3.5 的 MTP（`Qwen3_5ForCausalLMMTP`）多token预测模块实现。 |
| `qwen3_classification.py` | Qwen3 分类模型实现。 |
| `qwen3_moe.py` | Qwen3 MoE（`Qwen3MoeForCausalLM`）模型架构实现。 |
| `qwen3_next.py` | Qwen3-Next（`Qwen3NextForCausalLM`）混合架构模型实现。 |
| `qwen3_next_mtp.py` | Qwen3-Next 的 MTP（`Qwen3NextForCausalLMMTP`）多token预测模块实现。 |
| `qwen3_omni_moe.py` | Qwen3-Omni MoE（`Qwen3OmniMoeForConditionalGeneration`）全模态模型实现。 |
| `qwen3_rm.py` | Qwen3 奖励模型（reward model）实现。 |
| `qwen3_vl.py` | Qwen3-VL（`Qwen3VLForConditionalGeneration`）视觉语言模型实现。 |
| `qwen3_vl_moe.py` | Qwen3-VL MoE（`Qwen3VLMoeForConditionalGeneration`）视觉语言 MoE 模型实现。 |
| `radio.py` | RADIO 视觉编码器（视觉骨干）实现，供多模态模型使用。 |
| `roberta.py` | XLM-RoBERTa（`XLMRobertaModel`、序列分类）编码器模型实现。 |
| `sarashina2_vision.py` | Sarashina2-Vision（`Sarashina2VisionForCausalLM`）视觉语言模型实现。 |
| `sarvam_moe.py` | Sarvam MoE（`SarvamMLAForCausalLM`、`SarvamMoEForCausalLM`）模型架构实现。 |
| `sdar.py` | SDAR（`SDARForCausalLM`）模型架构实现。 |
| `sdar_moe.py` | SDAR MoE（`SDARMoeForCausalLM`）模型架构实现。 |
| `siglip.py` | SigLIP 视觉编码器实现，供多模态模型使用。 |
| `solar.py` | Solar（`SolarForCausalLM`）模型架构实现。 |
| `stablelm.py` | StableLM（`StableLmForCausalLM`）模型架构实现。 |
| `starcoder2.py` | StarCoder2（`Starcoder2ForCausalLM`）代码模型实现。 |
| `step3_vl.py` | Step-3 VL（`Step3VLForConditionalGeneration`）视觉语言模型实现。 |
| `step3_vl_10b.py` | Step-3 VL 10B（`StepVLForConditionalGeneration`）视觉语言模型实现。 |
| `step3p5.py` | Step-3.5（`Step3p5ForCausalLM`）模型架构实现。 |
| `step3p5_mtp.py` | Step-3.5 的 MTP（`Step3p5MTP`）多token预测模块实现。 |
| `teleflm.py` | TeleFLM（`TeleFLMForCausalLM`）模型架构实现。 |
| `torch_native_llama.py` | 纯 PyTorch 原生实现的 LLaMA/Phi3（`TorchNativeLlama/Phi3ForCausalLM`），用于参考/对照。 |
| `whisper.py` | OpenAI Whisper（`WhisperForConditionalGeneration`）语音识别模型实现。 |
| `xverse.py` | 元象 XVERSE（`XverseForCausalLM`）模型架构实现。 |
| `xverse_moe.py` | 元象 XVERSE MoE（`XverseMoeForCausalLM`）模型架构实现。 |
| `yivl.py` | Yi-VL（`YiVLForCausalLM`）视觉语言模型实现。 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| `deepseek_common` | DeepSeek 系列模型共享的公共组件（注意力后端分发、权重加载、设备/量化能力检测等）。 |
