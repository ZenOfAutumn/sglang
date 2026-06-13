# srt/multimodal/evs

## 目录用途
实现高效视频采样（Efficient Video Sampling, EVS，论文 arXiv:2510.14624），通过剪枝时间上冗余的视频 token 来减少视觉 token 数量、加速 VLM 推理。包含底层算法、模块封装与处理器三层结构。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| __init__.py | 包入口，导出 `EVS`、`EVSConfig`、`EVSEmbeddingResult`、`EVSProcessor` 等公共接口。 |
| evs_core.py | EVS 核心算法，含保留 token 数计算（首帧全保留）、保留掩码（retention mask）计算、每帧 token 数与偏移替换等底层函数。 |
| evs_module.py | EVS 模块封装，定义 `EVS`、`EVSConfig`、`EVSEmbeddingResult` 及数据项类型，对多模态 embedding 应用剪枝逻辑。 |
| evs_processor.py | EVS 处理器，将图像/视频数据项按模态拆分并接入 EVS 剪枝流程，输出处理后的多模态数据项。 |

`README.md` 为原英文说明文档（非 .py 文件）。
