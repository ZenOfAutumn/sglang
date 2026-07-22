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

/// 缓存感知（cache-aware）路由策略的配置。
///
/// 策略针对每个请求的路由键（通常为 prompt 前缀）计算各 worker 的缓存前缀匹配率：
/// 匹配率高于 [`Self::cache_threshold`] 时优先复用缓存；否则选择缓存树较小的 worker，
/// 为新前缀预留更多缓存空间。发生明显负载失衡时，策略会优先纠正负载而不是保持缓存亲和性。
#[derive(Debug, Clone)]
pub struct CacheAwareConfig {
    /// 走缓存命中路径的最小前缀匹配率，取值应在 `0.0..=1.0`。
    ///
    /// 当最佳 worker 的匹配率严格大于该值时，请求会路由至该 worker 以复用其 KV cache；
    /// 否则路由至缓存树规模最小的健康 worker。较低的值倾向于缓存亲和，较高的值倾向于
    /// 将低相似度请求分散到可用缓存空间更多的 worker。默认值为 `0.5`。
    pub cache_threshold: f32,

    /// 判定 worker 负载失衡所需满足的最小绝对请求数差。
    ///
    /// 仅当最大负载与最小负载之差严格大于此值，且也满足
    /// [`Self::balance_rel_threshold`] 时，策略才将负载视为失衡并优先选择较空闲的 worker。
    /// 默认值为 `32`。
    pub balance_abs_threshold: usize,

    /// 判定 worker 负载失衡所需满足的最小相对负载比。
    ///
    /// 仅当 `max_load > min_load * balance_rel_threshold`，且也满足
    /// [`Self::balance_abs_threshold`] 时，策略才判定失衡。该参数应大于等于 `1.0`；
    /// 值越小，策略越积极地为负载均衡牺牲缓存命中率。默认值为 `1.1`。
    pub balance_rel_threshold: f32,

    /// 缓存前缀树按大小执行 LRU 淘汰的周期，单位为秒。
    ///
    /// 每个周期都会将超过 [`Self::max_tree_size`] 的树裁剪到容量限制以内。设为 `0` 时不启动
    /// 后台淘汰任务，适合测试或由外部机制负责内存控制的场景。默认值为 `30` 秒。
    pub eviction_interval_secs: u64,

    /// 单个 worker 的单棵缓存前缀树允许保留的最大节点数。
    ///
    /// 超过此上限的节点不会立即删除，而是在下一次 `eviction_interval_secs` 触发时按 LRU
    /// 淘汰叶子节点。该值越大，能保留的历史前缀越多，但占用的内存也越高。默认值为 `10_000`。
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

/// 分桶（bucket）路由策略的配置。
///
/// 分桶策略把 worker 划分到若干负载区间（桶）中，并周期性地根据负载分布调整分桶，
/// 从而在保持较低协调开销的同时平滑负载。以下阈值用于判定何时需要重新分桶。
#[derive(Debug, Clone)]
pub struct BucketConfig {
    /// 判定负载失衡所需满足的最小绝对请求数差。
    ///
    /// 仅当最大负载与最小负载之差严格大于此值，且也满足 [`Self::balance_rel_threshold`]
    /// 时，才视为失衡。默认值为 `32`。
    pub balance_abs_threshold: usize,

    /// 判定负载失衡所需满足的最小相对负载比。
    ///
    /// 仅当 `max_load > min_load * balance_rel_threshold`，且也满足
    /// [`Self::balance_abs_threshold`] 时，才判定失衡。默认值为 `1.0001`（对失衡极为敏感）。
    pub balance_rel_threshold: f32,

    /// 后台重新调整分桶的周期，单位为秒。
    ///
    /// 每隔该间隔，策略会依据最新负载分布重新计算分桶边界。默认值为 `5` 秒。
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

/// 辅助函数：筛选出健康的 worker 并返回它们在原始列表中的下标。
///
/// 「健康」需同时满足两个条件：worker 自身被标记为健康（`is_healthy`），
/// 且其熔断器（circuit breaker）当前允许执行（`can_execute`）。各策略在选择前
/// 通常先调用本函数，避免把请求路由到不可用或处于熔断状态的 worker。
pub(crate) fn get_healthy_worker_indices(workers: &[Arc<dyn Worker>]) -> Vec<usize> {
    workers
        .iter()
        .enumerate()
        .filter(|(_, w)| w.is_healthy() && w.circuit_breaker().can_execute())
        .map(|(idx, _)| idx)
        .collect()
}

/// 辅助函数：将 `model_id` 归一化为用于策略查找的键。
///
/// 当 `model_id` 为空时返回 [`crate::core::UNKNOWN_MODEL_ID`]，以保证单模型与多模型
/// 部署下的行为保持一致（空模型 ID 始终映射到同一个已知键）。
#[inline]
pub(crate) fn normalize_model_key(model_id: &str) -> &str {
    if model_id.is_empty() {
        crate::core::UNKNOWN_MODEL_ID
    } else {
        model_id
    }
}

/// 传递给策略用于选择 worker 的上下文信息。
///
/// 不同策略按需读取其中的字段：例如缓存感知策略使用请求文本，前缀哈希策略使用
/// token 序列，基于头部的策略使用 HTTP 头，一致性哈希策略使用预构建的哈希环。
/// 未使用到的字段保持为 `None` 即可。
#[derive(Debug, Clone, Default)]
pub struct SelectWorkerInfo<'a> {
    /// 用于缓存感知路由的请求文本（通常为 prompt），据此计算前缀匹配率。
    pub request_text: Option<&'a str>,
    /// 用于前缀哈希路由的分词结果。
    ///
    /// 由 `PrefixHashPolicy` 使用，基于 token 序列做前缀哈希以实现前缀亲和路由。
    pub tokens: Option<&'a [u32]>,
    /// 用于基于头部的路由策略的 HTTP 头。
    ///
    /// 策略可从头部提取路由信息，例如：
    /// - `X-SMG-Target-Worker`：按下标直接路由到指定 worker；
    /// - `X-SMG-Routing-Key`：按一致性哈希路由以实现会话亲和。
    pub headers: Option<&'a http::HeaderMap>,
    /// 预先构建好的哈希环，用于 O(log n) 复杂度的一致性哈希。
    ///
    /// 由 `WorkerRegistry` 构建并缓存后透传进来，避免每个请求都重复构建哈希环。
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
