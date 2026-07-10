// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Cache-aware tree-lookup microbench.
//!
//! Mirrors the shape of `sgl-model-gateway/benches/radix_tree_benchmark.rs`
//! (specifically the `TokenTree` / `PositionalIndexer` paths — which serve
//! the same role as sgl-router's `HashTree`). The bench measures:
//!
//!   * `insert` — populate one worker's prefix.
//!   * `match_prefix` — score an incoming request against the tree.
//!
//! Output is `criterion`'s default (target/criterion/...). To run:
//!
//!   cargo bench --bench tree_lookup
//!   cargo bench --bench tree_lookup -- --sample-size 30   # faster
//!
//! See `BENCHMARKS.md` for the SMG↔sgl-router comparison table.

// criterion：Rust 的基准测试框架，提供统计学意义上稳定的性能测量。
// black_box 用于阻止编译器对被测代码做过度优化；Throughput 用于报告吞吐量。
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
// HashTree：缓存感知路由所用的前缀树；KvWorkerId：标识一个持有 KV 缓存的 worker。
use sgl_router::policies::kv_events::tree::{HashTree, KvWorkerId};

/// 构造一棵用于基准测试的前缀树。
///
/// - `num_workers`：worker 数量，每个 worker 会插入一段独立的前缀。
/// - `blocks_per_worker`：每个 worker 持有的 KV block（哈希）数量。
/// - `seed`：随机数种子，保证测试可复现。
fn build_tree(num_workers: usize, blocks_per_worker: usize, seed: u64) -> HashTree {
    let tree = HashTree::new();
    // 使用固定种子的确定性随机数生成器，保证每次运行生成相同数据，测试结果可对比。
    let mut rng = StdRng::seed_from_u64(seed);
    for w in 0..num_workers {
        // 为每个 worker 构造一个形如 http://w0:30000 的唯一标识。
        let worker = KvWorkerId::new(format!("http://w{w}:30000"), 0);
        // 每个 worker 持有一段互不相同的（随机）前缀，使前缀树向外扇出，
        // 这更贴近缓存感知路由的真实场景。
        let hashes: Vec<i64> = (0..blocks_per_worker).map(|_| rng.gen::<i64>()).collect();
        // 将该 worker 的前缀插入树中（None 表示无父前缀）。
        tree.insert(&worker, None, &hashes);
    }
    tree
}

/// 基准测试：向前缀树插入一个 worker 前缀的耗时（`insert`）。
fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("hashtree_insert");
    // 对不同规模的 block 数量分别测量，观察插入耗时随前缀长度的变化。
    for &n_blocks in &[8usize, 32, 128, 512] {
        // 以 block 数量作为吞吐量单位，便于报告“每秒可插入多少个 block”。
        group.throughput(Throughput::Elements(n_blocks as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n_blocks), &n_blocks, |b, &n| {
            // 预先生成随机哈希序列，避免把随机数生成的开销计入插入耗时。
            let mut rng = StdRng::seed_from_u64(0xC0FFEE);
            let hashes: Vec<i64> = (0..n).map(|_| rng.gen::<i64>()).collect();
            // iter_batched：每次迭代前用 HashTree::new 构造一棵全新的空树（setup），
            // 只测量真正的插入逻辑，避免树中已有数据干扰测量结果。
            b.iter_batched(
                HashTree::new,
                |tree| {
                    let worker = KvWorkerId::new("http://w:30000".to_string(), 0);
                    // black_box 阻止编译器把插入优化掉。
                    tree.insert(&worker, None, black_box(&hashes));
                    tree
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// 基准测试：在前缀树中匹配一个请求前缀的耗时（`match_prefix`）。
/// 这是缓存感知路由为进入的请求打分（评估命中哪个 worker）的核心操作。
fn bench_match_prefix(c: &mut Criterion) {
    let mut group = c.benchmark_group("hashtree_match_prefix");
    // (workers, blocks_per_worker, query_len) 三元组覆盖真实运行区间：
    // 小集群+中等前缀、中等集群+较长前缀，以及一个压力测试用例。
    let cases = [
        (4usize, 32usize, 8usize),
        (16, 64, 32),
        (64, 128, 64),
        (128, 256, 128),
    ];
    for (workers, bpw, query_len) in cases {
        // 用例标签，例如 w16_bpw64_q32，便于在报告中区分。
        let label = format!("w{workers}_bpw{bpw}_q{query_len}");
        // 以查询长度作为吞吐量单位。
        group.throughput(Throughput::Elements(query_len as u64));
        // 预先构造好待查询的前缀树（不计入测量）。
        let tree = build_tree(workers, bpw, 0xDEADBEEF);
        // 构造一个随机查询前缀，使查询能与树中前缀产生非平凡的部分匹配，
        // 更接近生产环境的热点路径。
        let mut rng = StdRng::seed_from_u64(0x12345);
        let probe: Vec<i64> = (0..query_len).map(|_| rng.gen::<i64>()).collect();
        group.bench_function(label, |b| {
            b.iter(|| {
                // 执行前缀匹配，并用 black_box 读取命中的 block 数，防止被优化掉。
                let m = tree.match_prefix(None, black_box(&probe));
                black_box(m.matched_blocks)
            });
        });
    }
    group.finish();
}

// 注册基准测试组并生成 main 函数入口。
criterion_group!(benches, bench_insert, bench_match_prefix);
criterion_main!(benches);
