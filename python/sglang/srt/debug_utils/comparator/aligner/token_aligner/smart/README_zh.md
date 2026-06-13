# srt/debug_utils/comparator/aligner/token_aligner/smart

## 目录用途
本目录实现 token 对齐的 `smart`（智能匹配）模式：加载并规范化各框架的辅助张量（input_ids、positions、seq_lens、seq_ids），按序列身份在两侧之间匹配序列，计算每个 token 的精确定位（step + 步内索引）以截取公共可比序列。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 空的包初始化文件。 |
| `aux_loader.py` | 加载并规范化辅助张量为框架无关的 step/global 辅助数据，提供插件探测与重导出。 |
| `aux_plugins.py` | 框架辅助插件 ABC 及实现，描述各框架的张量名集合、CP 分片名、序列长度提取等。 |
| `executor.py` | 执行 token 对齐计划：按布局把 BS 折叠为 T，依 locator 抽取并拼接匹配 token。 |
| `planner.py` | 由两侧序列信息匹配序列对，校验 input_ids 一致，生成 token 定位计划。 |
| `seq_info_builder.py` | 从 step 辅助数据累积构建每序列的 `TokenAlignerSeqInfo`/`SeqsInfo`。 |
| `types.py` | 序列身份与对齐相关数据类型（`SeqId`、`TokenAlignerStepAux`、`TokenLocator`、`TokenAlignerPlan` 等）。 |
