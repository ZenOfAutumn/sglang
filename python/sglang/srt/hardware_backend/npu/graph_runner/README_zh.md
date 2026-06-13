# srt/hardware_backend/npu/graph_runner

## 目录用途
本目录提供昇腾 NPU 的图执行运行器，通过 NPU graph 与 `torch.compile` 捕获并重放前向计算以降低开销。各运行器均继承自对应的 CUDA graph runner，覆盖常规解码、EAGLE 投机解码（draft 与 draft-extend）及视觉编码器（ViT）等场景。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `npu_graph_runner.py` | NPU 通用图运行器 `NPUGraphRunner`（继承 `CudaGraphRunner`），含模型 NPU 化补丁 `patch_model_npu`，以 NPU graph + torch.compile 运行模型。 |
| `eagle_draft_npu_graph_runner.py` | EAGLE 草稿模型的 NPU 图运行器 `EAGLEDraftNpuGraphRunner`（继承 `EAGLEDraftCudaGraphRunner`）。 |
| `eagle_draft_extend_npu_graph_runner.py` | EAGLE draft-extend 阶段的 NPU 图运行器 `EAGLEDraftExtendNpuGraphRunner`（继承 `EAGLEDraftExtendCudaGraphRunner`）。 |
| `vit_npu_graph_runner.py` | 视觉编码器的 NPU 图运行器 `ViTNpuGraphRunner`（继承 `ViTCudaGraphRunner`），适配 `VisionAttention` 的图捕获。 |
