# srt/hardware_backend

## 目录用途
本目录是 SGLang 非 NVIDIA/通用硬件后端适配的命名空间，按硬件平台分子目录组织对应的模型执行、内存、算子等适配实现。当前包含 Apple Silicon（MLX）与华为昇腾（NPU）两类后端。该层目录自身不含 `.py` 文件。

## 子目录
| 子目录 | 说明 |
| --- | --- |
| `mlx` | Apple Silicon 的端到端 MLX 后端适配，绕过 PyTorch MPS 直接在 MLX 框架内运行模型。 |
| `npu` | 华为昇腾 NPU 后端适配，含内存池、注意力、图执行、MoE、量化等子模块。 |
