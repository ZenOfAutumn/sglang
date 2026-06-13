# srt/debug_utils/comparator/aligner/unsharder

## 目录用途
本目录负责 unshard（去并行分片）：依据并行信息（TP/CP/EP 等轴的 rank 与 size）把分布在多个 rank 上的张量分片，按 concat / pick / reduce_sum / cp_thd_concat 等操作还原为完整张量，并对声明为 replicated 的轴做一致性校验。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 空的包初始化文件。 |
| `executor.py` | 执行 unshard 计划：按分组对张量做拼接/选取/求和，返回还原张量与 replicated 校验结果。 |
| `parallel_info.py` | 从转储元数据中提取并规范化统一的并行轴信息（rank/size），处理错误哨兵与伪轴。 |
| `planner.py` | 依据 dim 规格与各张量并行坐标计算分组与 unshard 计划（含 replicated 轴与 DP 过滤处理）。 |
| `types.py` | unshard 参数与计划类型（`AxisInfo`、`ConcatParams`、`CpThdConcatParams`、`PickParams`、`ReduceSumParams`、`UnsharderPlan`）。 |
