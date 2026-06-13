# srt/debug_utils/comparator/aligner/token_aligner

## 目录用途
本目录负责跨步骤的 token 对齐，把两侧（如 SGLang 与 Megatron）在不同 step 中产生的 token 按身份/顺序匹配，截取可比公共序列，供逐 token 数值比对。提供 `concat_steps`（简单拼接）与 `smart`（基于序列身份的智能匹配）两种模式。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 空的包初始化文件。 |
| `entrypoint.py` | token 对齐入口：探测辅助张量、按模式构建序列信息并计算对齐计划，返回含 THD 元数据的结果。 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| `concat_steps/` | 简单模式：按顺序拼接各 step 的 token，再截断到两侧公共长度。 |
| `smart/` | 智能模式：加载辅助张量、构建序列信息、按 token 身份匹配并定位。 |
