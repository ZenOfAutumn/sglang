# srt/layers/moe/fused_moe_triton/configs

## 目录用途
存放 fused MoE Triton kernel 的预调优配置。本目录不含任何 .py 文件，仅由若干以 Triton 版本命名的子目录组成，每个子目录中是按 GPU 型号、专家数 E 与中间维度 N（及数据类型/block_shape）调优得到的 JSON 配置文件，供 `fused_moe_triton_config.py` 在运行时按设备与形状匹配加载。

## 子目录
| 子目录 | 说明 |
| --- | --- |
| `triton_3_1_0` | Triton 3.1.0 下预调优的 fused MoE kernel JSON 配置（约 127 个）。 |
| `triton_3_2_0` | Triton 3.2.0 下预调优的 fused MoE kernel JSON 配置（约 35 个）。 |
| `triton_3_3_0` | Triton 3.3.0 下预调优的 fused MoE kernel JSON 配置（约 1 个）。 |
| `triton_3_3_1` | Triton 3.3.1 下预调优的 fused MoE kernel JSON 配置（约 21 个）。 |
| `triton_3_4_0` | Triton 3.4.0 下预调优的 fused MoE kernel JSON 配置（约 33 个）。 |
| `triton_3_5_1` | Triton 3.5.1 下预调优的 fused MoE kernel JSON 配置（约 68 个）。 |
