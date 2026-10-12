> 来源: <https://github.com/ai-dynamo/dynamo/blob/main/lib/kv-router/README.md>
> 说明: 原 docs/architecture/kv_cache_routing.md 已随文档迁移,此为仓库现存 kv-router 组件 README
> 抓取时间: 2026-09-01 ｜ 格式: 官方原始 Markdown

# dynamo-kv-router

`dynamo-kv-router` 提供 Dynamo 用于将请求引导至缓存重叠度最高的 worker 的核心 KV 感知路由数据结构与调度原语。

## 本 Crate 提供的内容

- 用于前缀重叠索引的 `RadixTree` 和 `ConcurrentRadixTree`
- 面向更高吞吐索引后端的 `ThreadPoolIndexer` 和 `PositionalIndexer`
- 用于请求路由的 `KvRouterConfig`、`RouterQueuePolicy` 和 `LocalScheduler`
- 协议与哈希辅助工具，如 `RouterEvent`、`WorkerId`、
  `compute_block_hash_for_seq` 和 `compute_seq_hash_for_block`

## Rust 基本用法

```rust
use dynamo_kv_router::{
    KvRouterConfig, RadixTree, compute_block_hash_for_seq, compute_seq_hash_for_block,
};
use dynamo_kv_router::protocols::BlockHashOptions;

let prompt_tokens = vec![1_u32, 2, 3, 4, 5, 6, 7, 8];
let local_hashes = compute_block_hash_for_seq(&prompt_tokens, 4, BlockHashOptions::default());
let seq_hashes = compute_seq_hash_for_block(&local_hashes);

let router_config = KvRouterConfig::default();
let index = RadixTree::new();
let scores = index.find_matches(local_hashes, false);

assert!(router_config.use_kv_events);
assert_eq!(seq_hashes.len(), 2);
assert!(scores.scores.is_empty());
```

要实现端到端路由，请将这些 indexer 与 `LocalScheduler` 以及从 crate 根部重新导出的 worker/配置协议类型配合使用。

## 特性（Features）

- `metrics`：router 内部的 Prometheus 指标
- `runtime-protocols`：与 `dynamo-runtime` 的集成点
- `standalone-indexer`：独立 indexer 服务支持
- `bench`：内部基准测试辅助工具

## 延伸阅读

- Router 指南: <https://docs.nvidia.com/dynamo/components/router>
- Indexer 内部实现:
  <https://github.com/ai-dynamo/dynamo/blob/main/lib/kv-router/src/indexer/README.md>
- 分片 KV indexer 基准测试: [INDEXER_BENCH.md](../bench/kv_router/INDEXER_BENCH.md)
- Dynamo 仓库: <https://github.com/ai-dynamo/dynamo>
