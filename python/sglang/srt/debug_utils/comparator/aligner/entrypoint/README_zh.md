# srt/debug_utils/comparator/aligner/entrypoint

## 目录用途
本目录是对齐子系统的总入口，负责把 unshard、reorder、token 对齐、axis 对齐等子计划组合为一个完整的对齐计划（plan）并执行（execute），同时定义计划相关的数据类型与执行轨迹包装类型。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 空的包初始化文件。 |
| `executor.py` | 执行对齐计划：按步骤依次执行各子计划（unshard/reorder/token/axis），产出对齐结果与形状快照轨迹。 |
| `planner.py` | 规划对齐计划：依据两侧元数据、dims 与并行信息计算逐步骤的子计划。 |
| `traced_types.py` | 执行后将每个子计划与其观测到的 `ShapeSnapshot` 配对的轨迹包装类型，便于下游格式化。 |
| `types.py` | 对齐计划的数据类型定义（`AlignerPlan`、逐步骤计划及其子计划的判别联合类型）。 |
