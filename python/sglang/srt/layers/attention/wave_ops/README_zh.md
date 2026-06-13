# srt/layers/attention/wave_ops

## 目录用途
本目录提供基于 AMD Wave（wave_lang）框架的注意力算子，供 `wave_backend.py` 在 ROCm 平台调用，涵盖 decode、extend 与 prefill 三个阶段的注意力前向计算。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| decode_attention.py | 解码阶段 Wave 注意力算子（`decode_attention_wave`/`decode_attention_fwd`），含 kernel 获取与中间数组形状计算。 |
| extend_attention.py | extend 阶段 Wave 注意力算子（`extend_attention_wave`）及 kernel 获取。 |
| prefill_attention.py | prefill 阶段 Wave 注意力算子（`prefill_attention_wave`）。 |
