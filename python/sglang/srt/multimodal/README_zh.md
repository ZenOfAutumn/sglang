# srt/multimodal

## 目录用途
多模态输入的处理与特征提取核心模块，负责将图像/视频/音频等原始输入转换为模型可用的张量与占位 token，并提供视觉编码器（ViT）的 CUDA Graph 加速运行器。其中 `processors/` 子目录按多模态模型逐一实现各自的预处理器，`evs/` 子目录实现高效视频采样（EVS）相关逻辑。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| customized_mm_processor_utils.py | 自定义多模态处理器注册工具，提供装饰器将配置类的 `model_type` 映射到自定义 `ProcessorMixin` 处理器，覆盖 HuggingFace 默认处理器。 |
| internvl_utils.py | InternVL 系列的图像变换工具，含 ImageNet 归一化、动态分块（dynamic preprocess）等基于 torchvision 的图像预处理函数。 |
| internvl_vit_cuda_graph_runner.py | 针对 InternVL ViT 视觉编码器的 CUDA Graph 运行器，捕获并复用计算图以加速视觉特征提取。 |
| mm_utils.py | 多模态通用工具集，源自 LLaVA-NeXT，主要实现 anyres / anyres_max 等图像切分与网格处理逻辑（CLIP、SigLip 适用）。 |
| vit_cuda_graph_runner.py | 通用 ViT 视觉编码器的 CUDA Graph 运行器，按视觉注意力等结构捕获计算图以提升推理性能。 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| evs | 高效视频采样（Efficient Video Sampling），对时间冗余的视频 token 进行剪枝以加速 VLM 推理。 |
| processors | 各多模态模型的输入处理器实现，每个文件对应一个（或一类）模型的预处理逻辑。 |
