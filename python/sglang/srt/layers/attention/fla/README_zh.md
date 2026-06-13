# srt/layers/attention/fla

## 目录用途
本目录是从 flash-linear-attention（FLA）项目移植/改编的融合线性注意力 Triton 算子集合，主要服务 Gated Delta Rule、KDA（Kimi Delta Attention）、GLA 等线性注意力机制。提供分块（chunk）前向、融合循环（fused recurrent）解码、门控归一化与各类辅助算子，供 linear/ 与 hybrid_linear_attn_backend 调用。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| chunk.py | Gated Delta Rule 的分块前向入口（`chunk_gated_delta_rule`）及 autograd Function。 |
| chunk_delta_h.py | 分块隐藏状态 h 的前向 Triton 算子（blockdim64 kernel）。 |
| chunk_fwd.py | Gated Delta Rule 分块内（intra）前向算子，含 KKT 求解 kernel。 |
| chunk_intra.py | KDA 分块内前向算子（inter-solve 融合与 sub-chunk kernel）。 |
| chunk_intra_token_parallel.py | KDA 分块内前向的 token 并行版本算子。 |
| chunk_o.py | 分块输出 o 的前向计算 Triton 算子（`chunk_fwd_o`）。 |
| chunk_scaled_dot_kkt.py | 分块缩放点积 K·Kᵀ（scaled dot KKT）前向算子。 |
| cumsum.py | 分块局部累加和（标量/向量版 `chunk_local_cumsum`）Triton 算子。 |
| fused_gdn_gating.py | 融合的 GDN（Gated Delta Net）门控计算 Triton 算子。 |
| fused_norm_gate.py | 融合的 LayerNorm/RMSNorm + 门控前向算子及 `FusedRMSNormGated` 模块。 |
| fused_recurrent.py | Gated Delta Rule 的融合循环前向/打包解码/状态更新算子。 |
| fused_sigmoid_gating_recurrent.py | 融合 sigmoid 门控的 delta rule 循环更新算子（解码用）。 |
| index.py | 序列长度/分块索引/分块偏移等辅助张量构造（带 tensor_cache）。 |
| kda.py | KDA（Kimi Delta Attention）算子集合：融合循环、缩放点积 KKT、w/u 重算、GLA 输出等。 |
| l2norm.py | L2 归一化前向 Triton 算子及 `L2Norm` 模块。 |
| layernorm_gated.py | 门控 LayerNorm/RMSNorm 的完整实现（参考实现+kernel+nn.Module）。 |
| op.py | 基础数学算子封装（exp/log/log2 等，支持 FLA fast ops 开关）。 |
| solve_tril.py | 下三角矩阵求逆 Triton 算子（16x16/32x32/64x64 合并）。 |
| utils.py | FLA 通用工具：环境检查、tensor_cache、input_guard、平台检测、误差断言等。 |
| wy_fast.py | WY 表示下 w/u 的快速重算前向 Triton 算子。 |
