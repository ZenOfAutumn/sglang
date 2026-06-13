# srt/hardware_backend/npu/moe

## 目录用途
本目录提供昇腾 NPU 上的 MoE 专家路由适配，将专家 top-k 选择映射到 NPU 算子（如 `sgl_kernel_npu` 的 l1_norm），并接入 SGLang 的专家分布记录与 EPLB 物理映射等机制。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `topk.py` | NPU 版专家选择 `fused_topk_npu`，复用 `select_experts`/`StandardTopKOutput` 产出 top-k 结果，并对接专家分布记录器与逻辑到物理专家映射。 |
