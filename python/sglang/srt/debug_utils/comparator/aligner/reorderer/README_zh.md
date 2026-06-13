# srt/debug_utils/comparator/aligner/reorderer

## 目录用途
本目录负责序列维的重排序，将 CP（context parallel）下的 zigzag 切分顺序还原为自然顺序（natural），支持普通张量与 THD（变长打平）两种布局。仅作用于序列/token 维。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 空的包初始化文件。 |
| `executor.py` | 执行重排序计划：按 zigzag→natural 或其 THD 变体，对指定维做 CP 顺序还原。 |
| `planner.py` | 依据 dim 规格中的 ordering 修饰符与并行信息生成重排序计划，校验仅序列维允许 zigzag。 |
| `types.py` | 重排序计划与参数类型（`ZigzagToNaturalParams`、`ZigzagToNaturalThdParams`、`ReordererPlan`）。 |
