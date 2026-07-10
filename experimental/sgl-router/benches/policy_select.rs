// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Policy-selection throughput microbench.
//!
//! Mirrors `sgl-model-gateway/benches/manual_policy_benchmark.rs` —
//! measures how fast the routing layer returns a worker for a given
//! request context, across the policies sgl-router actually ships
//! (round-robin, random, power-of-two-choices). The cache-aware-zmq
//! policy lives in `tree_lookup.rs`; this file targets the non-tree
//! policies' steady-state hot path.

// criterion：Rust 基准测试框架。black_box 阻止编译器优化，Throughput 报告吞吐量。
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
// worker 发现相关类型：模型 ID、worker ID、worker 模式与规格。
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
// 三种非前缀树路由策略：二选一（power-of-two）、随机、轮询。
use sgl_router::policies::power_of_two::PowerOfTwoChoicesPolicy;
use sgl_router::policies::random::RandomPolicy;
use sgl_router::policies::round_robin::RoundRobinPolicy;
// Policy：路由策略 trait；SelectionContext：一次选择所需的上下文（模型、请求体等）。
use sgl_router::policies::{Policy, SelectionContext};
// Worker：单个 worker 实例；WorkerRegistry：worker 注册表。
use sgl_router::workers::{Worker, WorkerRegistry};
use std::sync::Arc;

/// 构造 `n` 个绑定到指定 `model` 的测试 worker，并返回它们的列表。
fn workers(n: usize, model: &str) -> Vec<Arc<Worker>> {
    let registry = WorkerRegistry::default();
    for i in 0..n {
        // 注册一个普通（Plain）模式的 worker，URL 形如 http://w0:30000。
        registry
            .add(WorkerSpec {
                id: WorkerId(format!("w{i}")),
                url: format!("http://w{i}:30000"),
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId(model.into())],
                bootstrap_port: None,
            })
            .expect("test workers are unmixed");
    }
    // 从注册表中取出服务于该模型的所有 worker。
    registry.workers_for(&ModelId(model.into()))
}

/// 通用的策略基准测试：测量给定策略在不同 worker 数量下完成一次选择的耗时。
///
/// - `name`：策略名称，用于报告分组。
/// - `policy`：待测的路由策略（包装为 trait 对象）。
fn bench_policy(c: &mut Criterion, name: &str, policy: Arc<dyn Policy>) {
    let mut group = c.benchmark_group(format!("policy_select::{name}"));
    // 对不同 worker 规模分别测量，观察策略选择耗时随集群规模的变化。
    for &n in &[4usize, 16, 64, 256] {
        let workers = workers(n, "tiny");
        let model = ModelId("tiny".into());
        // 所有迭代使用相同的请求体 —— 仅测量策略每次调用的成本，
        // 而非请求体解析的开销。
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "tiny",
            "messages": [{"role": "user", "content": "hello world"}],
        }))
        .unwrap();
        // 每次迭代完成一次选择，因此吞吐量单位为 1。
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                // 构造选择上下文（包含模型与请求体）。
                let ctx = SelectionContext::new(&model, Some(&body));
                // 执行策略选择；black_box 防止编译器把调用优化掉。
                let chosen = policy.select(black_box(&workers), &ctx);
                black_box(chosen);
            });
        });
    }
    group.finish();
}

/// 轮询策略基准测试：按顺序依次选择 worker。
fn bench_round_robin(c: &mut Criterion) {
    bench_policy(c, "round_robin", Arc::new(RoundRobinPolicy::new()));
}

/// 随机策略基准测试：随机选择一个 worker。
fn bench_random(c: &mut Criterion) {
    bench_policy(c, "random", Arc::new(RandomPolicy::new()));
}

/// 二选一（power-of-two-choices）策略基准测试：随机抽两个 worker，选负载较低的那个。
fn bench_power_of_two(c: &mut Criterion) {
    bench_policy(c, "power_of_two", Arc::new(PowerOfTwoChoicesPolicy::new()));
}

// 注册基准测试组并生成 main 函数入口。
criterion_group!(benches, bench_round_robin, bench_random, bench_power_of_two);
criterion_main!(benches);
