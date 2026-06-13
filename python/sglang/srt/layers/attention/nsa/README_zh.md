# srt/layers/attention/nsa

## 目录用途
本目录实现 Native Sparse Attention（NSA，原生稀疏注意力，DeepSeek 系列使用）的支撑组件，包括 top-k 索引器、KV cache 的 FP8 量化/反量化、稀疏注意力 TileLang/Triton 算子、页表索引变换，以及上下文并行（CP）与 MTP（多 token 预测）相关工具。被上层 `nsa_backend.py` 调用。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| nsa_indexer.py | NSA 索引器 `Indexer`（多平台算子）与索引器元数据基类 `BaseIndexerMetadata`，计算稀疏 top-k 索引。 |
| quant_k_cache.py | K cache 的 FP8 量化（含参考实现、快速 kernel 与分离量化版本）。 |
| dequant_k_cache.py | K cache 的反量化（含参考实现、快速 kernel 与分页版本）。 |
| index_buf_accessor.py | 索引缓冲区的 K/S 读写访问器（GetK/SetS 等类）及 Triton 存取 kernel。 |
| tilelang_kernel.py | 基于 TileLang 的稀疏注意力/稀疏 MLA 解码算子及 FP8 激活量化、索引算子。 |
| triton_kernel.py | NSA 的 Triton 算子：激活量化与有效 KV 索引提取。 |
| transform_index.py | prefill/decode 页表索引变换（fast/ref 版本及 Triton kernel）。 |
| nsa_backend_mtp_precompute.py | NSA 后端 MTP 元数据预计算 Mixin（`PrecomputedMetadata`、cu_seqlens 计算）。 |
| nsa_mtp_verification.py | NSA MTP 融合元数据拷贝的单/多后端校验工具。 |
| utils.py | NSA 工具：序列长度计算、上下文并行（CP）切分/重建、padding 与 `NSAContextParallelMetadata`。 |
