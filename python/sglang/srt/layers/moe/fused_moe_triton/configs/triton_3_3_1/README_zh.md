# srt/layers/moe/fused_moe_triton/configs/triton_3_3_1

## 目录用途
存放 Triton 3.3.1 版本下、按 GPU 型号与 MoE 形状（专家数 E、中间维度 N、数据类型及 block_shape）预调优的 fused MoE kernel JSON 配置（约 21 个），运行时由配置加载逻辑按设备与形状匹配选用。
