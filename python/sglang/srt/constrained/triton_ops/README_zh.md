# srt/constrained/triton_ops

## 目录用途
本目录存放约束解码所需的 Triton GPU 算子实现，用于在采样前高效地按语法生成的 token bitmask 屏蔽不合法 token，加速结构化输出的强制约束过程。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `bitmask_ops.py` | Triton 实现的 token bitmask 原地应用算子（`apply_token_bitmask_inplace_triton` 及其 kernel），将 logits 中被掩码的 token 置为屏蔽值 |
