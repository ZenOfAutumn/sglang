# srt/layers/moe/fused_moe_triton/configs/triton_3_5_1

## 目录用途
存放 Triton 3.5.1 版本下、按 GPU 型号与 MoE 形状（专家数 E、中间维度 N 及数据类型）预调优的 fused MoE kernel JSON 配置（约 68 个），运行时由配置加载逻辑按设备与形状匹配选用。
