# srt/mem_cache/sparsity/backend

## 目录用途
稀疏注意力的注意力后端适配层。负责在前向过程中保存原始 metadata，并依据稀疏算法选出的索引/有效长度改写注意力 metadata，使不同注意力后端（FlashAttention、NSA）能在稀疏 KV 上正确执行。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| __init__.py | 导出 `BackendAdaptor`、`FlashAttentionAdaptor`、`NSABackendAdaptor`。 |
| backend_adaptor.py | 后端适配器抽象基类 `BackendAdaptor` 及 FlashAttention/NSA 实现，提供 `adapt_for_attn_metadata` 等。 |
