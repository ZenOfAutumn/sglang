# srt/layers/quantization/configs

## 目录用途
本目录不含 Python 代码，仅存放块级量化 GEMM 的调优 JSON 配置文件。文件名按 `N`/`K` 矩阵维度、`device_name`（如 NVIDIA H100/H200/B200、AMD MI300X 等）、`dtype`（`fp8_w8a8`/`int8_w8a8`）与 `block_shape`（如 `[128, 128]`）命名，运行时据此加载对应硬件与形状的最优内核参数。
