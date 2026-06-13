# srt/debug_utils/comparator/aligner

## 目录用途
本目录是比对器的对齐子系统总入口，负责在数值比对前把两侧张量调整到可比形态，包括 unshard（去并行分片）、reorder（zigzag 还原）、token 对齐与轴对齐。各阶段拆分为规划（planner）与执行（executor）两步。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 空的包初始化文件。 |
| `axis_aligner.py` | 轴对齐器：当两侧语义维名一致但排列不同时，生成并执行 einops `rearrange` 计划使其轴顺序对齐。 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| `entrypoint/` | 对齐总流程的规划与执行入口，组合 unshard/reorder/token/axis 各子计划。 |
| `reorderer/` | 序列维 zigzag→natural 的重排序规划与执行（含 THD 变体）。 |
| `token_aligner/` | 跨步/智能 token 对齐，将两侧按 token 身份匹配到可比序列。 |
| `unsharder/` | 按并行轴（TP/CP/EP 等）将分片张量拼接/选取/求和还原为完整张量。 |
