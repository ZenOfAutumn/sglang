//! SGLang router 的负载均衡策略
//!
//! 本模块为路由策略提供统一抽象，同时适用于普通（Regular）路由
//! 与 Prefill-Decode（PD，预填充/解码分离）路由两种模式。
//!
//! 所有具体策略（随机、轮询、二选一、缓存感知、前缀哈希、一致性哈希、
//! 手动、分桶等）都实现同一个 [`LoadBalancingPolicy`] trait，以便可插拔地替换。

use std::{fmt::Debug, sync::Arc};

use async_trait::async_trait;
use smg_mesh::OptionalMeshSyncManager;

use crate::core::{HashRing, Worker};

mod bucket;
mod cache_aware;
mod consistent_hashing;
mod factory;
mod manual;
mod power_of_two;
mod prefix_hash;
mod random;
mod registry;
mod round_robin;
pub mod tree;
pub(crate) mod utils;
pub use bucket::BucketPolicy;
pub use cache_aware::CacheAwarePolicy;
pub use consistent_hashing::ConsistentHashingPolicy;
pub use factory::PolicyFactory;
pub use manual::{ManualConfig, ManualPolicy};
pub use power_of_two::PowerOfTwoPolicy;
pub use prefix_hash::{PrefixHashConfig, PrefixHashPolicy};
pub use random::RandomPolicy;
pub use registry::PolicyRegistry;
pub use round_robin::RoundRobinPolicy;
pub use tree::PrefixMatchResult;

/// 负载均衡策略的核心 trait。
///
/// 它为各种路由算法提供统一接口，既适用于普通模式的单 Worker 选择，
/// 也适用于 PD 模式的双 Worker（Prefill + Decode）选择。
///
/// 大多数方法都提供了默认空实现，无状态策略无需重写；
/// 仅 `select_worker`、`name`、`as_any` 为必须实现的方法。
#[async_trait]
pub trait LoadBalancingPolicy: Send + Sync + Debug {
    /// 从可用 Worker 中选出一个，返回其在 `workers` 中的下标。
    ///
    /// 用于普通路由模式（请求发往单个 Worker）。
    /// 使用 `Arc<dyn Worker>` 以获得更好性能并避免不必要的克隆。
    ///
    /// # 参数
    /// * `workers` - 可供选择的 Worker 列表
    /// * `info` - 路由决策所需的附加信息（请求文本、token、头部、哈希环等）
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize>;

    /// 请求完成后更新策略状态。
    ///
    /// 当一个请求完成（无论成功与否）时被调用，供策略更新其内部状态。
    fn on_request_complete(&self, _worker_url: &str, _success: bool) {
        // 默认：无状态策略无需处理
    }

    /// 获取策略名称（用于指标与调试）。
    fn name(&self) -> &'static str;

    /// 该策略是否需要请求文本来做路由决策。
    fn needs_request_text(&self) -> bool {
        false // 默认：大多数策略不需要请求文本
    }

    /// 更新 Worker 的负载信息。
    ///
    /// 针对负载感知类策略，由外部监控周期性传入当前负载信息。
    fn update_loads(&self, _loads: &std::collections::HashMap<String, isize>) {
        // 默认：不使用负载信息的策略无需处理
    }

    /// 设置 mesh 同步管理器（用于多实例间状态同步）。
    fn set_mesh_sync(&mut self, _mesh_sync: OptionalMeshSyncManager) {
        // 默认：不使用 mesh 同步的策略无需处理
    }

    /// 重置策略的内部状态。
    ///
    /// 对维护状态的策略（如轮询的游标）很有用。
    fn reset(&self) {
        // 默认：无状态策略无需处理
    }

    /// 返回 `Any` 以支持向下转型（downcast）到具体策略类型。
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Configuration for cache-aware policy
#[derive(Debug, Clone)]
pub struct CacheAwareConfig {
    pub cache_threshold: f32,
    pub balance_abs_threshold: usize,
    pub balance_rel_threshold: f32,
    pub eviction_interval_secs: u64,
    pub max_tree_size: usize,
}

impl Default for CacheAwareConfig {
    fn default() -> Self {
        Self {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 30,
            max_tree_size: 10000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BucketConfig {
    pub balance_abs_threshold: usize,
    pub balance_rel_threshold: f32,
    pub bucket_adjust_interval_secs: usize,
}

impl Default for BucketConfig {
    fn default() -> Self {
        Self {
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.0001,
            bucket_adjust_interval_secs: 5,
        }
    }
}

/// Helper function to filter healthy workers and return their indices
pub(crate) fn get_healthy_worker_indices(workers: &[Arc<dyn Worker>]) -> Vec<usize> {
    workers
        .iter()
        .enumerate()
        .filter(|(_, w)| w.is_healthy() && w.circuit_breaker().can_execute())
        .map(|(idx, _)| idx)
        .collect()
}

/// Helper function to normalize model_id to a key for policy lookups.
///
/// Returns UNKNOWN_MODEL_ID for empty model_ids to ensure consistent behavior
/// across single-model and multi-model deployments.
#[inline]
pub(crate) fn normalize_model_key(model_id: &str) -> &str {
    if model_id.is_empty() {
        crate::core::UNKNOWN_MODEL_ID
    } else {
        model_id
    }
}

/// Information passed to policy for worker selection
#[derive(Debug, Clone, Default)]
pub struct SelectWorkerInfo<'a> {
    /// Request text for cache-aware routing
    pub request_text: Option<&'a str>,
    /// Tokenized request for prefix-hash routing
    /// Used by PrefixHashPolicy for token-based prefix hashing
    pub tokens: Option<&'a [u32]>,
    /// HTTP headers for header-based routing policies
    /// Policies can extract routing information from headers like:
    /// - X-SMG-Target-Worker: Direct routing to a specific worker by index
    /// - X-SMG-Routing-Key: Consistent hash routing for session affinity
    pub headers: Option<&'a http::HeaderMap>,
    /// Pre-computed hash ring for O(log n) consistent hashing
    /// Built and cached by WorkerRegistry, passed through to avoid per-request rebuilds
    pub hash_ring: Option<Arc<HashRing>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    #[tokio::test]
    async fn test_get_healthy_worker_indices() {
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key2")
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w3:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .build(),
            ),
        ];

        // All healthy initially
        let indices = get_healthy_worker_indices(&workers);
        assert_eq!(indices, vec![0, 1, 2]);

        // Mark one unhealthy
        workers[1].set_healthy(false);
        let indices = get_healthy_worker_indices(&workers);
        assert_eq!(indices, vec![0, 2]);
    }
}
