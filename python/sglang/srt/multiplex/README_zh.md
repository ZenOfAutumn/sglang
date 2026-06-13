# srt/multiplex

## 目录用途
本目录实现 PD 复用（PD multiplexing，pdmux），通过将 GPU 的 SM（流多处理器）按组划分到不同 CUDA Stream，让 prefill 与 decode 在同一张卡上并发执行，从而提升 SM 利用率与吞吐。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `multiplexing_mixin.py` | `SchedulerMultiplexMixin`，为调度器提供 PD 复用调度逻辑，管理多 stream 分组、SM 分配与 prefill/decode 并发执行。 |
| `pdmux_context.py` | PD 复用全局上下文：`PDMuxConfig` 配置、`load_pdmux_config`、按算力约束的 SM 划分（`divide_sm`）、stream 分组初始化与当前 stream 索引/SM 数等查询函数。 |
