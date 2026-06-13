# srt/layers/attention/mamba/ops

## 目录用途
本目录是 Mamba/Mamba2 状态空间模型的核心计算算子集合，包含选择性扫描状态更新与 SSD（State Space Duality）分块算法的各 Triton kernel，以及门控归一化与算子后端调度。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| __init__.py | 包初始化，导出 `mamba_chunk_scan_combined`、`selective_state_update` 等对外算子。 |
| layernorm_gated.py | 门控 LayerNorm/RMSNorm 前向算子（`rms_norm_gated`）。 |
| mamba_ssm.py | 选择性状态更新算子（`selective_state_update`）及其 Triton kernel（decode 单步）。 |
| ssd_bmm.py | SSD 分块批量矩阵乘（bmm chunk）前向算子。 |
| ssd_chunk_scan.py | SSD 分块扫描（chunk scan）前向算子。 |
| ssd_chunk_state.py | SSD 分块状态（chunk state/cumsum/varlen）前向算子。 |
| ssd_combined.py | SSD 分块扫描组合入口 `mamba_chunk_scan_combined`（融合各分块步骤）。 |
| ssd_state_passing.py | SSD 跨分块状态传递（state passing）前向算子。 |
| ssu_dispatch.py | 选择性状态更新算子后端调度（Triton/FlashInfer），按 ServerArgs 初始化。 |
