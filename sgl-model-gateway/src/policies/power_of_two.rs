//! Power-of-two choices（二选一）负载均衡策略
//!
//! 核心思想：不去全局扫描所有 Worker 找最优，而是随机抽取两个 Worker，
//! 只在这两者之间选负载更低的一个。相比「全局最小负载」，它以极小的
//! 协调开销就能显著改善尾部负载（避免所有请求同时涌向同一个「当前最空闲」
//! 节点导致的负载振荡），是经典的 "Power of Two Choices" 结论。

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use async_trait::async_trait;
use rand::Rng;
use tracing::debug;

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::core::Worker;

/// Power-of-two choices（二选一）策略。
///
/// 随机选取两个 Worker，并把请求路由到其中负载较低的那个。
/// 这样能在极低的协调开销下获得良好的负载分布。
#[derive(Debug)]
pub struct PowerOfTwoPolicy {
    /// 来自外部监控（LoadMonitor）的缓存负载信息。
    ///
    /// key 为 Worker 的 URL，value 为该 Worker 的负载（此处为 token 级负载，
    /// 保真度高于本地请求计数）。由 `update_loads` 周期性刷新。
    cached_loads: RwLock<HashMap<String, isize>>,
}

impl PowerOfTwoPolicy {
    /// 创建一个负载缓存为空的策略实例。
    pub fn new() -> Self {
        Self {
            cached_loads: RwLock::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl LoadBalancingPolicy for PowerOfTwoPolicy {
    /// 在候选 Worker 中按「二选一」策略选出一个，返回其在 `workers` 中的下标。
    ///
    /// 步骤：先过滤出健康 Worker；随机抽取两个不同的健康 Worker；
    /// 比较二者负载（优先用 token 级负载，缺失则回退到请求计数），选负载较低者。
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        // 仅在健康 Worker 中选择
        let healthy_indices = get_healthy_worker_indices(workers);

        // 没有健康 Worker：无法选择
        if healthy_indices.is_empty() {
            return None;
        }

        // 只有一个健康 Worker：直接返回它，无需二选一
        if healthy_indices.len() == 1 {
            return Some(healthy_indices[0]);
        }

        // 随机抽取两个 Worker —— 用偏移量在 O(1) 内保证抽到的两个互不相同
        let mut rng = rand::rng();
        let idx1 = rng.random_range(0..healthy_indices.len());
        // 从其余下标中挑 idx2：以 idx1 为基准 +1 再加上 [0, len-1) 的随机偏移并取模，
        // 从而必定落在与 idx1 不同的位置上
        let idx2 =
            (idx1 + 1 + rng.random_range(0..healthy_indices.len() - 1)) % healthy_indices.len();

        // 将「健康列表内下标」映射回「原始 workers 数组下标」
        let worker_idx1 = healthy_indices[idx1];
        let worker_idx2 = healthy_indices[idx2];
        let worker1 = &workers[worker_idx1];
        let worker2 = &workers[worker_idx2];

        // 安全地读取缓存负载（读锁获取失败时降级为 None）
        let loads_guard = self.cached_loads.read().ok();

        // 尝试为「两个」Worker 都取到高保真的 token 级负载
        let load1_tokens = loads_guard
            .as_ref()
            .and_then(|m| m.get(worker1.url()).copied());
        let load2_tokens = loads_guard
            .as_ref()
            .and_then(|m| m.get(worker2.url()).copied());

        // 若任一 Worker 缺失 token 数据（如监控采集失败），
        // 必须把「两个」都降级为请求计数来比较，以保证公平性
        // （避免用「token 负载」与「请求数」这类不可比的指标做比较）。
        let (load1, load2) = match (load1_tokens, load2_tokens) {
            (Some(t1), Some(t2)) => {
                // 两者都有 token 数据：直接按 token 负载比较
                (t1, t2)
            }
            _ => {
                // 其一或两者缺失 token 数据：
                // 两者都回退到本地请求计数进行比较
                (worker1.load() as isize, worker2.load() as isize)
            }
        };

        // 选出负载较低的 Worker（相等时优先选第一个）
        let selected_idx = if load1 <= load2 {
            worker_idx1
        } else {
            worker_idx2
        };

        debug!(
            "Power-of-two selection: {}={} vs {}={} -> selected {}",
            worker1.url(),
            load1,
            worker2.url(),
            load2,
            workers[selected_idx].url()
        );

        // 递增被选中 Worker 的「已处理请求」计数器
        workers[selected_idx].increment_processed();

        Some(selected_idx)
    }

    /// 策略名称（用于注册表查找与指标标签）。
    fn name(&self) -> &'static str {
        "power_of_two"
    }

    /// 由外部监控周期性调用，用最新负载快照整体替换缓存。
    /// 写锁获取失败时静默跳过本次更新（下次刷新会补上）。
    fn update_loads(&self, loads: &HashMap<String, isize>) {
        if let Ok(mut cached) = self.cached_loads.write() {
            *cached = loads.clone();
        }
    }

    /// 支持向下转型（downcast）到具体类型，便于按需访问具体策略实现。
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Default for PowerOfTwoPolicy {
    /// 默认实例等价于 `new()`：负载缓存为空。
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    #[tokio::test]
    async fn test_power_of_two_selection() {
        let policy = PowerOfTwoPolicy::new();
        let worker1 = BasicWorkerBuilder::new("http://w1:8000")
            .worker_type(WorkerType::Regular)
            .build();
        let worker2 = BasicWorkerBuilder::new("http://w2:8000")
            .worker_type(WorkerType::Regular)
            .build();
        let worker3 = BasicWorkerBuilder::new("http://w3:8000")
            .worker_type(WorkerType::Regular)
            .build();

        // Set different loads
        for _ in 0..10 {
            worker1.increment_load();
        }
        for _ in 0..5 {
            worker2.increment_load();
        }
        // worker3 has load 0

        let workers: Vec<Arc<dyn Worker>> =
            vec![Arc::new(worker1), Arc::new(worker2), Arc::new(worker3)];

        // Run multiple selections
        let mut selected_counts = [0; 3];
        let info = SelectWorkerInfo::default();
        for _ in 0..100 {
            if let Some(idx) = policy.select_worker(&workers, &info).await {
                selected_counts[idx] += 1;
            }
        }

        // Worker with lowest load (worker3) should be selected most often
        assert!(selected_counts[2] > selected_counts[1]);
        assert!(selected_counts[1] > selected_counts[0]);
    }

    #[tokio::test]
    async fn test_power_of_two_with_cached_loads() {
        let policy = PowerOfTwoPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

        // Update cached loads
        let mut loads = HashMap::new();
        loads.insert("http://w1:8000".to_string(), 100);
        loads.insert("http://w2:8000".to_string(), 10);
        policy.update_loads(&loads);

        // Should prefer worker2 with lower cached load
        let mut w2_selected = 0;
        let info = SelectWorkerInfo::default();
        for _ in 0..50 {
            if let Some(idx) = policy.select_worker(&workers, &info).await {
                if idx == 1 {
                    w2_selected += 1;
                }
            }
        }

        // Worker2 should be selected significantly more often
        assert!(w2_selected > 35); // Should win most of the time
    }

    #[tokio::test]
    async fn test_power_of_two_single_worker() {
        let policy = PowerOfTwoPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        )];

        // With single worker, should always select it
        assert_eq!(
            policy
                .select_worker(&workers, &SelectWorkerInfo::default())
                .await,
            Some(0)
        );
    }

    #[tokio::test]
    async fn test_reproduce_incompatible_metric_bug() {
        use std::{collections::HashMap, sync::Arc};

        use crate::core::{BasicWorkerBuilder, WorkerType};

        // 1. Setup the policy
        let policy = PowerOfTwoPolicy::new();

        // 2. Create Worker A: Idle (0 reqs), but has high token usage in cache
        let worker_a = BasicWorkerBuilder::new("http://worker_a:8000")
            .worker_type(WorkerType::Regular)
            .build();

        // 3. Create Worker B: Busy (5 reqs), but missing from cache
        let worker_b = BasicWorkerBuilder::new("http://worker_b:8000")
            .worker_type(WorkerType::Regular)
            .build();

        // Manually increment load on Worker B to simulate active requests
        for _ in 0..5 {
            worker_b.increment_load();
        }

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(worker_a), Arc::new(worker_b)];

        // 4. Simulate LoadMonitor update:
        // Only Worker A gets a token report. Worker B is missing (e.g. monitor failure).
        let mut loads = HashMap::new();
        loads.insert("http://worker_a:8000".to_string(), 50_000); // 50k tokens load
        policy.update_loads(&loads);

        // 5. Run selection
        let selected_idx = policy
            .select_worker(&workers, &SelectWorkerInfo::default())
            .await
            .expect("Should select a worker");

        // 6. Verify the Fix
        // Logic:
        // - Worker A has token load (50k) but Worker B has NO token load.
        // - Policy should fallback to request counts for BOTH.
        // - A has 0 requests, B has 5 requests.
        // - 0 <= 5, so A should be selected.

        if selected_idx == 0 {
            println!("Bug Fixed: System correctly fell back to request counts and selected idle Worker A.");
        } else {
            println!(
                "Bug PERSISTS: Selected Worker B (Load: 5 reqs) over Worker A (Load: 50k tokens)"
            );
        }

        // Assert that the CORRECT worker (A, index 0) is selected
        assert_eq!(
            selected_idx, 0,
            "The policy failed to handle incompatible metrics. Should select idle Worker A."
        );
    }
    #[tokio::test]
    async fn test_power_of_two_edge_cases() {
        use std::{collections::HashMap, sync::Arc};

        use crate::core::{BasicWorkerBuilder, WorkerType};

        let policy = PowerOfTwoPolicy::new();

        // Helper to create a worker with specific request load
        let create_worker = |url: &str, reqs: usize| {
            let w = BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .build();
            for _ in 0..reqs {
                w.increment_load();
            }
            Arc::new(w)
        };

        //  Scenario 1: Happy Path (Both have Token Data)
        // Worker A: 10 requests, but only 1,000 tokens (Light usage) -> Should be CHOSEN
        // Worker B:  2 requests, but 100,000 tokens (Heavy usage) -> Should be AVOIDED
        // This proves we use high-fidelity metrics when available, ignoring request counts.
        let w_a = create_worker("http://a:8000", 10);
        let w_b = create_worker("http://b:8000", 2);
        let workers_1: Vec<Arc<dyn Worker>> = vec![w_a.clone(), w_b.clone()];

        let mut loads_1 = HashMap::new();
        loads_1.insert("http://a:8000".to_string(), 1_000);
        loads_1.insert("http://b:8000".to_string(), 100_000);
        policy.update_loads(&loads_1);

        let idx_1 = policy
            .select_worker(&workers_1, &SelectWorkerInfo::default())
            .await
            .unwrap();
        assert_eq!(
            idx_1, 0,
            "Happy Path Failed: Should select Worker A (fewer tokens) despite higher request count"
        );

        // Scenario 2: Partial Failure (Worker A has tokens, Worker B is missing)
        // Worker A: 10 requests, 1,000 tokens (Cached)
        // Worker B:  2 requests, MISSING cache
        // Logic: Fallback to requests -> Compare 10 (A) vs 2 (B) -> Select B
        let w_c = create_worker("http://c:8000", 10);
        let w_d = create_worker("http://d:8000", 2);
        let workers_2: Vec<Arc<dyn Worker>> = vec![w_c.clone(), w_d.clone()];

        let mut loads_2 = HashMap::new();
        loads_2.insert("http://c:8000".to_string(), 1_000);
        // http://d:8000 is MISSING
        policy.update_loads(&loads_2);

        let idx_2 = policy
            .select_worker(&workers_2, &SelectWorkerInfo::default())
            .await
            .unwrap();
        assert_eq!(idx_2, 1, "Partial Fail 1 Failed: Should fallback to requests and select Worker B (fewer requests)");

        // Scenario 3: Partial Failure (Worker A is missing, Worker B has tokens)
        // Worker A:  2 requests, MISSING cache
        // Worker B: 10 requests, 1,000 tokens (Cached)
        // Logic: Fallback to requests -> Compare 2 (A) vs 10 (B) -> Select A
        let w_e = create_worker("http://e:8000", 2);
        let w_f = create_worker("http://f:8000", 10);
        let workers_3: Vec<Arc<dyn Worker>> = vec![w_e.clone(), w_f.clone()];

        let mut loads_3 = HashMap::new();
        // http://e:8000 is MISSING
        loads_3.insert("http://f:8000".to_string(), 1_000);
        policy.update_loads(&loads_3);

        let idx_3 = policy
            .select_worker(&workers_3, &SelectWorkerInfo::default())
            .await
            .unwrap();
        assert_eq!(idx_3, 0, "Partial Fail 2 Failed: Should fallback to requests and select Worker A (fewer requests)");

        // Scenario 4: Total Failure (Both missing)
        // Worker A: 5 requests
        // Worker B: 3 requests
        // Logic: Requests vs Requests -> Select B
        let w_g = create_worker("http://g:8000", 5);
        let w_h = create_worker("http://h:8000", 3);
        let workers_4: Vec<Arc<dyn Worker>> = vec![w_g.clone(), w_h.clone()];

        let loads_4 = HashMap::new();
        policy.update_loads(&loads_4);

        let idx_4 = policy
            .select_worker(&workers_4, &SelectWorkerInfo::default())
            .await
            .unwrap();
        assert_eq!(
            idx_4, 1,
            "Total Fail Failed: Should select Worker B based on request count"
        );

        println!("All edge case tests passed successfully.");
    }
}
