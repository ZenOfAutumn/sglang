# srt/layers/attention/mamba

## 目录用途
本目录实现 Mamba/Mamba2 状态空间模型（SSM）相关组件，包括 Mamba2 混合器、因果一维卷积、前向元数据、状态散射与门控 RMSNorm，供混合线性注意力后端在 SSM 路径上调用。具体 SSM 扫描算子位于 ops/ 子目录。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| causal_conv1d.py | 因果一维卷积的对外接口（`causal_conv1d_fn`/`causal_conv1d_update`）。 |
| causal_conv1d_triton.py | 因果一维卷积的 Triton kernel 实现（连续批 prefill 与 decode 更新）。 |
| mamba.py | Mamba2 混合器 `MambaMixer2` 及其分片权重加载逻辑。 |
| mamba2_metadata.py | Mamba2 前向元数据 `Mamba2Metadata` 与基础 `ForwardMetadata`。 |
| mamba_state_scatter_triton.py | 带掩码的 Mamba 状态散射融合 Triton 算子（`fused_mamba_state_scatter_with_mask`）。 |
| mixer2_rms_norm_gated.py | Mamba2 门控 RMSNorm 多平台算子 `Mixer2RMSNormGated`。 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| ops | Mamba SSM 核心算子：选择性扫描/状态更新、SSD 分块（bmm/chunk scan/chunk state/state passing/combined）等。 |
