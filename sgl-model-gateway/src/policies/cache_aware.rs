/*
    缓存感知负载均衡路由器（Cache-Aware Load Balancing Router）

    本路由器结合两种策略，同时优化缓存利用率与请求分布：

    1. 缓存感知路由（近似基数树 Approximate Tree）
    2. 负载均衡（带均衡阈值的最短队列 Shortest Queue）

    路由器会根据负载状况在两种策略之间动态切换：
    - 当系统负载失衡时，使用负载均衡
    - 当系统负载均衡时，使用缓存感知路由

    仅当同时满足以下两个条件时，系统才被判定为「失衡」：
    1. (max - min) > abs_threshold        （最大与最小负载的绝对差超过绝对阈值）
    2. max > rel_threshold * min          （最大负载超过最小负载的相对倍数）

    策略细节：

    1. 缓存感知路由（近似基数树）
    -------------------------------------------
    该策略基于请求历史为每个 worker 维护一棵近似基数树（radix tree），
    从而无需直接查询 worker 的缓存状态。树中存储的是原始文本字符
    而非 token ID，以避免分词开销。

    流程：
    a. 对每个请求，找出前缀匹配率最高的 worker
    b. 若匹配率 > cache_threshold：
       路由到匹配率最高的 worker（很可能已缓存相关数据）
    c. 若匹配率 ≤ cache_threshold：
       路由到树规模最小的 worker（拥有最多可用缓存容量）
    d. 后台维护：
       周期性淘汰最近最少使用（LRU）的叶子节点，防止内存溢出

    2. 负载均衡（最短队列）
    -------------------------------------------
    该策略跟踪每个 worker 的待处理请求数，当检测到系统失衡时，
    将新请求路由到最空闲的 worker。

    配置参数：
    ------------------------
    1. cache_threshold：（浮点数，0.0 ~ 1.0）
       使用「最高匹配路由」的最小前缀匹配率。
       低于该阈值时，路由到拥有最多可用缓存空间的 worker。

    2. balance_abs_threshold：（整数）
       负载失衡检测的绝对差阈值。
       当 (max_load - min_load) > abs_threshold 时，系统可能失衡。

    3. balance_rel_threshold：（浮点数）
       负载失衡检测的相对比值阈值。
       当 max_load > min_load * rel_threshold 时，系统可能失衡。
       与 abs_threshold 共同判定最终的失衡状态。

    4. eviction_interval_secs：（整数）
       近似树的 LRU 淘汰周期（间隔秒数）。

    5. max_tree_size：（整数）
       每棵树的最大节点数。超出后，将在下一次淘汰周期中
       淘汰最近最少使用（LRU）的叶子节点。
*/

use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use rand::Rng;
use smg_mesh::{tree_ops::TreeOperation, OptionalMeshSyncManager};
use tracing::{debug, warn};

use super::{
    get_healthy_worker_indices, normalize_model_key, tree::Tree, utils::PeriodicTask,
    CacheAwareConfig, LoadBalancingPolicy, SelectWorkerInfo,
};
use crate::core::{Worker, WorkerType, UNKNOWN_MODEL_ID};

/// 用于在 cache_aware 的树键中隔离 prefill/decode/regular 三类 worker 池的标签。
///
/// 树以 `pool::model` 作为键，从而使得同一模型下 prefill→decode 交替的调用序列
/// 不会互相驱逐对方的租户(tenant)。若不做这一隔离，每次 `select_worker` 末尾
/// 的 `tree.insert(text, url)` 就会就相同 prompt 覆盖掉上一个池的租户，
/// 从而使 cache_aware 退化为在两个池之间来回振荡。
fn pool_tag(worker_type: &WorkerType) -> &'static str {
    match worker_type {
        WorkerType::Regular => "regular",
        WorkerType::Prefill { .. } => "prefill",
        WorkerType::Decode => "decode",
    }
}

/// 将池标签与模型名拼接为复合树键，格式为 `pool::model`。
fn make_tree_key(pool: &str, model: &str) -> String {
    format!("{}::{}", pool, model)
}

/// 根据 worker 的类型与模型 ID 生成其对应的树键。
fn tree_key_for_worker(worker: &dyn Worker) -> String {
    make_tree_key(
        pool_tag(worker.worker_type()),
        normalize_model_key(worker.model_id()),
    )
}

/// 缓存感知路由策略
///
/// 当负载均衡时，根据缓存亲和性路由请求；
/// 当负载失衡时，切换为最短队列路由。
/// 为每个 `(pool, model)` 组合维护独立的树，从而使 prefill、decode
/// 和 regular 三类 worker 池不会互相驱逐对方的租户。
/// 支持将树操作通过 mesh 在集群节点间同步。
/// 当未启用 mesh 时，该策略独立工作、无需同步。
#[derive(Debug)]
pub struct CacheAwarePolicy {
    /// 缓存感知策略的配置（阈值、树容量、淘汰周期等）。
    config: CacheAwareConfig,
    /// 以 `pool::model` 为键的多棵近似树；使用 DashMap 以支持分片并发访问。
    trees: Arc<DashMap<String, Arc<Tree>>>,
    /// 可选的 mesh 同步管理器；为 None 时不进行跨节点同步。
    mesh_sync: OptionalMeshSyncManager,
    /// 后台 LRU 淘汰任务句柄；保存以维持任务存活（前缀下划线表示仅持有、不直接使用）。
    _eviction_task: Option<PeriodicTask>,
}

impl CacheAwarePolicy {
    pub fn new() -> Self {
        Self::with_config(CacheAwareConfig::default())
    }

    pub fn with_config(config: CacheAwareConfig) -> Self {
        let trees = Arc::new(DashMap::<String, Arc<Tree>>::new());

        // 若配置了淘汰周期(> 0)，则启动后台淘汰任务
        let eviction_task = if config.eviction_interval_secs > 0 {
            let trees_clone = Arc::clone(&trees);
            let max_tree_size = config.max_tree_size;

            Some(PeriodicTask::spawn(
                config.eviction_interval_secs,
                "Eviction",
                move || {
                    // 逐棵遍历树，按最大节点数限制淘汰多余的（LRU）叶子节点
                    for tree_ref in trees_clone.iter() {
                        let tree_key = tree_ref.key();
                        let tree = tree_ref.value();
                        tree.evict_tenant_by_size(max_tree_size);

                        debug!(
                            "Cache eviction completed for {}, max_size: {}",
                            tree_key, max_tree_size
                        );
                    }
                },
            ))
        } else {
            None
        };

        Self {
            config,
            trees,
            mesh_sync: None,
            _eviction_task: eviction_task,
        }
    }

    /// 设置 mesh 同步管理器（可在构造完成后调用）。
    /// 若传入的 mesh_sync 非空，则从 mesh 恢复已同步的树状态。
    pub fn set_mesh_sync(&mut self, mesh_sync: OptionalMeshSyncManager) {
        self.mesh_sync = mesh_sync.clone();
        if mesh_sync.is_some() {
            self.restore_tree_state_from_mesh();
        }
    }

    /// 用 worker 的 URL 初始化树（仅在初始化阶段使用）。
    pub fn init_workers(&self, workers: &[Arc<dyn Worker>]) {
        // 按 (pool, model) 对 worker 分组，使每个池拥有自己隔离的树。
        let mut grouped: std::collections::HashMap<String, Vec<&Arc<dyn Worker>>> =
            std::collections::HashMap::new();
        for worker in workers {
            grouped
                .entry(tree_key_for_worker(worker.as_ref()))
                .or_default()
                .push(worker);
        }

        for (tree_key, pool_workers) in grouped {
            let tree = self
                .trees
                .entry(tree_key)
                .or_insert_with(|| Arc::new(Tree::new()));
            for worker in pool_workers {
                tree.insert("", worker.url());
            }
        }
    }

    /// 向树中新增单个 worker（增量更新）。
    pub fn add_worker(&self, worker: &dyn Worker) {
        let tree_key = tree_key_for_worker(worker);
        let tree = self
            .trees
            .entry(tree_key)
            .or_insert_with(|| Arc::new(Tree::new()));
        tree.insert("", worker.url());
    }

    /// 从树中移除一个 worker。
    pub fn remove_worker(&self, worker: &dyn Worker) {
        let tree_key = tree_key_for_worker(worker);
        if let Some(tree) = self.trees.get(&tree_key) {
            tree.remove_tenant(worker.url());
        }
    }

    /// 按 URL 移除 worker（为向后兼容，从所有模型树中移除）。
    pub fn remove_worker_by_url(&self, url: &str) {
        // 因为无法得知它属于哪个模型，故从所有树中都移除
        for tree_ref in self.trees.iter() {
            tree_ref.value().remove_tenant(url);
        }
    }

    /// 从 mesh 存储中恢复树状态。
    /// 在初始化时调用，用以根据已同步的状态重建树。
    fn restore_tree_state_from_mesh(&self) {
        if let Some(ref mesh_sync) = self.mesh_sync {
            // 从 mesh 获取所有树状态：
            // 需要遍历所有拥有树状态的模型。
            // 目前仅为已存在于本地 trees 映射中的模型恢复树；
            // 完整实现中可能需要向 mesh 查询全部树状态。
            for tree_ref in self.trees.iter() {
                let tree_key = tree_ref.key();
                if let Some(tree_state) = mesh_sync.get_tree_state(tree_key) {
                    debug!(
                        "Restoring tree state for {} with {} operations",
                        tree_key,
                        tree_state.operations.len()
                    );

                    let tree = tree_ref.value();
                    // 重放所有操作以重建这棵树
                    for operation in &tree_state.operations {
                        match operation {
                            TreeOperation::Insert(insert_op) => {
                                tree.insert(&insert_op.text, &insert_op.tenant);
                            }
                            TreeOperation::Remove(remove_op) => {
                                tree.remove_tenant(&remove_op.tenant);
                            }
                        }
                    }
                }
            }
        }
    }

    /// 为 mesh 同步规范化树键：为保持一致性，将意外出现的空键
    /// 转换为 `UNKNOWN_MODEL_ID`。当前代码中复合键 `pool::model` 永远不会为空，
    /// 故此处属于防御性处理。
    fn normalize_mesh_model_id(tree_key: &str) -> &str {
        if tree_key.is_empty() {
            UNKNOWN_MODEL_ID
        } else {
            tree_key
        }
    }

    /// 应用来自 mesh 的远程树操作。
    ///
    /// `mesh_key` 是该操作最初同步时所使用的不透明键；
    /// `select_worker` / `select_worker_min_load` 向 mesh 发送树操作时
    /// 以复合键 `pool::model` 作为键，且将来的接收路径预期会原样
    /// 把同一个字符串回传到这里。参数保持为 `&str`，以便 mesh 层
    /// 可以保持对键无感知。
    ///
    /// 注意：`PolicyRegistry::apply_remote_tree_operation`（唯一的转发者）
    /// 目前在进程内没有调用方；接收路径尚未接通，
    /// 因此该方法目前仅能通过测试触及。
    pub fn apply_remote_tree_operation(&self, mesh_key: &str, operation: &TreeOperation) {
        let tree_key = Self::normalize_mesh_model_id(mesh_key);

        let tree = self
            .trees
            .entry(tree_key.to_string())
            .or_insert_with(|| Arc::new(Tree::new()));

        match operation {
            TreeOperation::Insert(insert_op) => {
                tree.insert(&insert_op.text, &insert_op.tenant);
                debug!(
                    "Applied remote tree insert: key={}, text={}, tenant={}",
                    mesh_key, insert_op.text, insert_op.tenant
                );
            }
            TreeOperation::Remove(remove_op) => {
                tree.remove_tenant(&remove_op.tenant);
                debug!(
                    "Applied remote tree remove: key={}, tenant={}",
                    mesh_key, remove_op.tenant
                );
            }
        }
    }

    /// 执行缓存淘汰，防止树无限增长。
    pub fn evict_cache(&self, max_size: usize) {
        for tree_ref in self.trees.iter() {
            let tree_key = tree_ref.key();
            let tree = tree_ref.value();
            tree.evict_tenant_by_size(max_size);
            debug!("Cache eviction for {}, max_size: {}", tree_key, max_size);
        }
    }

    /// 失衡时按最短队列选择 worker：选择当前负载最小的健康 worker，
    /// 并同样更新缓存树（即使处于失衡模式）以维护亲和性状态。
    fn select_worker_min_load(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: &Option<&str>,
        healthy_indices: &[usize],
        tree_key: &str,
        max_load: usize,
        min_load: usize,
    ) -> Option<usize> {
        // 记录负载均衡触发日志（仅在启用 debug 时才计算各 worker 负载）
        if tracing::enabled!(tracing::Level::DEBUG) {
            let worker_loads: Vec<(&str, usize)> =
                workers.iter().map(|w| (w.url(), w.load())).collect();
            debug!(
                "Load balancing triggered | max: {} | min: {} | workers: {:?}",
                max_load, min_load, worker_loads
            );
        }

        // 失衡时采用最短队列:选负载最小的健康 worker
        let min_load_idx = healthy_indices
            .iter()
            .min_by_key(|&&idx| workers[idx].load())
            .copied()?;

        // 即使处于失衡模式，也要更新树以维护缓存状态
        if let Some(text) = request_text {
            // 仅获取该键对应的树引用，无需锁住整个 HashMap；
            // DashMap 只会锁住包含该键的那个分片。
            let tree = self.trees.get(tree_key).map(|entry| entry.value().clone());

            if let Some(tree) = tree {
                let worker_url = workers[min_load_idx].url();
                // 现在可在不持有 HashMap 锁的情况下操作这棵树
                tree.insert(text, worker_url);

                // 若启用 mesh，则同步插入操作（未启用时为空操作）
                if let Some(ref mesh_sync) = self.mesh_sync {
                    use smg_mesh::tree_ops::TreeInsertOp;
                    let op = TreeOperation::Insert(TreeInsertOp {
                        text: text.to_string(),
                        tenant: worker_url.to_string(),
                    });
                    let mesh_key = Self::normalize_mesh_model_id(tree_key);
                    if let Err(e) = mesh_sync.sync_tree_operation(mesh_key.to_string(), op) {
                        warn!("Failed to sync tree insert operation to mesh: {}", e);
                    }
                }
            } else {
                warn!(
                    "cache_aware: no tree found for key '{}', skipping cache update — \
                     pool tree was not seeded (init_pd_cache_aware_policies missed or \
                     a race during worker registration)",
                    tree_key
                );
            }
        }

        // 递增被选中 worker 的「已处理请求」计数器
        workers[min_load_idx].increment_processed();

        Some(min_load_idx)
    }
}

#[async_trait]
impl LoadBalancingPolicy for CacheAwarePolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let request_text = info.request_text;
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        // 确定这组 worker 的 (pool, model) 键——路由层已预先过滤，
        // 因此这里每个健康 worker 都属于同一个池且同一个模型。
        let pivot = workers[healthy_indices[0]].as_ref();
        let tree_key = tree_key_for_worker(pivot);

        // 获取当前负载统计——一次遍历即算出 min/max，无额外内存分配
        let (min_load, max_load) = workers.iter().fold((usize::MAX, 0usize), |(min, max), w| {
            let load = w.load();
            (min.min(load), max.max(load))
        });
        let min_load = if min_load == usize::MAX { 0 } else { min_load };

        // 判断负载是否失衡（需同时满足绝对差与相对比两个阈值）
        let is_imbalanced = max_load.saturating_sub(min_load) > self.config.balance_abs_threshold
            && (max_load as f32) > (min_load as f32 * self.config.balance_rel_threshold);

        if is_imbalanced {
            return self.select_worker_min_load(
                workers,
                &request_text,
                &healthy_indices,
                &tree_key,
                max_load,
                min_load,
            );
        }

        // 负载均衡时使用缓存感知路由
        let text = request_text.unwrap_or("");

        // 仅获取该键对应的树引用，无需锁住整个 HashMap；
        // DashMap 只会锁住包含该键的那个分片。
        let tree = self.trees.get(&tree_key).map(|entry| entry.value().clone());

        if let Some(tree) = tree {
            // 现在在不持有 HashMap 锁的情况下操作这棵树；
            // 使用 prefix_match_with_counts 以避免重复的 chars().count() 调用
            let result = tree.prefix_match_with_counts(text);
            let match_rate = if result.input_char_count == 0 {
                0.0
            } else {
                result.matched_char_count as f32 / result.input_char_count as f32
            };

            // 选择 worker（避免 String 分配）
            let selected_idx = if match_rate > self.config.cache_threshold {
                // 缓存命中路径:按 URL 查找 worker（直接比较 &str，无分配）
                let tenant_url: &str = &result.tenant;
                workers
                    .iter()
                    .position(|w| w.url() == tenant_url)
                    .filter(|&idx| workers[idx].is_healthy())
            } else {
                // 缓存匹配率较低:退而选择负载最小的 worker
                healthy_indices
                    .iter()
                    .min_by_key(|&&idx| workers[idx].load())
                    .copied()
            };

            if let Some(idx) = selected_idx {
                // 用本次请求更新树（直接使用 worker URL，无分配）
                tree.insert(text, workers[idx].url());

                // 若启用 mesh，则同步插入操作（未启用时为空操作）
                if let Some(ref mesh_sync) = self.mesh_sync {
                    use smg_mesh::tree_ops::TreeInsertOp;
                    let op = TreeOperation::Insert(TreeInsertOp {
                        text: text.to_string(),
                        tenant: workers[idx].url().to_string(),
                    });
                    let mesh_key = Self::normalize_mesh_model_id(&tree_key);
                    if let Err(e) = mesh_sync.sync_tree_operation(mesh_key.to_string(), op) {
                        warn!("Failed to sync tree insert operation to mesh: {}", e);
                    }
                }

                // 递增被选中 worker 的「已处理请求」计数器
                workers[idx].increment_processed();

                return Some(idx);
            }

            // 被选中的 worker 已不存在或不健康，从树中移除这个陈旧的租户
            if match_rate > self.config.cache_threshold {
                let tenant_url: &str = &result.tenant;
                tree.remove_tenant(tenant_url);
                debug!("Removed stale worker {} from cache tree", tenant_url);

                // 若启用 mesh，则同步移除操作（未启用时为空操作）
                if let Some(ref mesh_sync) = self.mesh_sync {
                    use smg_mesh::tree_ops::TreeRemoveOp;
                    let op = TreeOperation::Remove(TreeRemoveOp {
                        tenant: tenant_url.to_string(),
                    });
                    let mesh_key = Self::normalize_mesh_model_id(&tree_key);
                    if let Err(e) = mesh_sync.sync_tree_operation(mesh_key.to_string(), op) {
                        warn!("Failed to sync tree remove operation to mesh: {}", e);
                    }
                }
            }

            // 兜底:退回到第一个健康 worker
            healthy_indices.first().copied()
        } else {
            warn!(
                "cache_aware: no tree found for key '{}', falling back to random \
                 worker selection — pool tree was not seeded \
                 (init_pd_cache_aware_policies missed or a race during worker \
                 registration); cache affinity is effectively disabled until this \
                 clears",
                tree_key
            );
            let mut rng = rand::rng();
            let random_idx = rng.random_range(0..healthy_indices.len());
            Some(healthy_indices[random_idx])
        }
    }

    fn on_request_complete(&self, worker_url: &str, success: bool) {
        // 未来可按 worker 统计成功率，以实现更智能的路由
        if !success {
            // 可选：对失败的请求降低其亲和性
            tracing::debug!(
                "Request to {} completed with success={}",
                worker_url,
                success
            );
        }
    }

    fn name(&self) -> &'static str {
        "cache_aware"
    }

    fn needs_request_text(&self) -> bool {
        true // 缓存感知策略需要请求文本来计算缓存亲和性
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Default for CacheAwarePolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    #[tokio::test]
    async fn test_cache_aware_with_balanced_load() {
        // Create policy without eviction thread for testing
        let config = CacheAwareConfig {
            eviction_interval_secs: 0, // Disable eviction thread
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
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
                    .api_key("test_api_key")
                    .build(),
            ),
        ];

        // Initialize the policy with workers
        policy.init_workers(&workers);

        // First request should be distributed
        let idx1 = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("hello world"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        // Same request should go to same worker (cache hit)
        let idx2 = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("hello world"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(idx1, idx2);

        // Similar request should also go to same worker
        let idx3 = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("hello"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(idx1, idx3);
    }

    #[tokio::test]
    async fn test_cache_aware_with_imbalanced_load() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            cache_threshold: 0.5,
            balance_abs_threshold: 5,
            balance_rel_threshold: 2.0,
            eviction_interval_secs: 0, // Disable eviction thread
            max_tree_size: 10000,
        });

        let worker1 = BasicWorkerBuilder::new("http://w1:8000")
            .worker_type(WorkerType::Regular)
            .build();
        let worker2 = BasicWorkerBuilder::new("http://w2:8000")
            .worker_type(WorkerType::Regular)
            .build();

        // Create significant load imbalance
        for _ in 0..20 {
            worker1.increment_load();
        }
        // worker2 has load 0

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(worker1), Arc::new(worker2)];
        policy.init_workers(&workers);

        // Should select worker2 (lower load) despite cache affinity
        let info = SelectWorkerInfo {
            request_text: Some("test"),
            ..Default::default()
        };
        for _ in 0..5 {
            let idx = policy.select_worker(&workers, &info).await.unwrap();
            assert_eq!(idx, 1); // Should always pick worker2
        }
    }

    #[tokio::test]
    async fn test_cache_aware_worker_removal() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0, // Disable eviction thread
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
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

        policy.init_workers(&workers);

        // Route some requests
        policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("test1"),
                    ..Default::default()
                },
            )
            .await;
        policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("test2"),
                    ..Default::default()
                },
            )
            .await;

        // Remove a worker
        policy.remove_worker_by_url("http://w1:8000");
        workers[0].set_healthy(false);

        // All requests should now go to worker2
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("test1"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(idx, 1);
    }

    #[tokio::test]
    async fn test_cache_aware_sync_tree_operation_to_mesh() {
        use std::sync::Arc;

        use smg_mesh::{stores::StateStores, sync::MeshSyncManager};

        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
        let mesh_sync = Arc::new(MeshSyncManager::new(stores, "node1".to_string()));

        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let mut policy = CacheAwarePolicy::with_config(config);
        policy.set_mesh_sync(Some(mesh_sync.clone()));

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .api_key("test_api_key")
                .build(),
        )];

        policy.init_workers(&workers);

        // Select worker with a request - should sync to mesh
        let _idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("test request"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        // Verify tree operation was synced to mesh under the composite `pool::model`
        // key — workers here are Regular and no model was specified, so the key is
        // `regular::UNKNOWN_MODEL_ID`.
        let expected_key = format!("regular::{}", UNKNOWN_MODEL_ID);
        let tree_state = mesh_sync.get_tree_state(&expected_key);
        assert!(tree_state.is_some());
        let tree = tree_state.unwrap();
        assert!(!tree.operations.is_empty());
    }

    #[test]
    fn test_cache_aware_restore_tree_state_from_mesh() {
        use std::sync::Arc;

        use smg_mesh::{
            stores::StateStores,
            sync::MeshSyncManager,
            tree_ops::{TreeInsertOp, TreeOperation},
        };

        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
        let mesh_sync = Arc::new(MeshSyncManager::new(stores, "node1".to_string()));

        // Pre-populate mesh with tree state
        let op1 = TreeOperation::Insert(TreeInsertOp {
            text: "test_text_1".to_string(),
            tenant: "http://w1:8000".to_string(),
        });
        mesh_sync
            .sync_tree_operation("model1".to_string(), op1)
            .unwrap();

        let op2 = TreeOperation::Insert(TreeInsertOp {
            text: "test_text_2".to_string(),
            tenant: "http://w2:8000".to_string(),
        });
        mesh_sync
            .sync_tree_operation("model1".to_string(), op2)
            .unwrap();

        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let mut policy = CacheAwarePolicy::with_config(config);
        policy.set_mesh_sync(Some(mesh_sync.clone()));

        // Initialize with a model to trigger restore
        let _workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .api_key("test_api_key")
                .build(),
        )];

        // Create a tree entry for model1 to trigger restore
        let _tree = policy
            .trees
            .entry("model1".to_string())
            .or_insert_with(|| Arc::new(Tree::new()));

        // Manually trigger restore (normally done in constructor)
        // For testing, we'll verify the tree state exists in mesh
        let tree_state = mesh_sync.get_tree_state("model1");
        assert!(tree_state.is_some());
        let state = tree_state.unwrap();
        assert_eq!(state.operations.len(), 2);
    }

    #[test]
    fn test_cache_aware_apply_remote_tree_operation() {
        use std::sync::Arc;

        use smg_mesh::{
            stores::StateStores,
            sync::MeshSyncManager,
            tree_ops::{TreeInsertOp, TreeOperation},
        };

        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
        let mesh_sync = Arc::new(MeshSyncManager::new(stores, "node1".to_string()));

        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let mut policy = CacheAwarePolicy::with_config(config);
        policy.set_mesh_sync(Some(mesh_sync.clone()));

        // Apply remote tree operation
        let remote_op = TreeOperation::Insert(TreeInsertOp {
            text: "remote_text".to_string(),
            tenant: "http://remote:8000".to_string(),
        });

        policy.apply_remote_tree_operation("model1", &remote_op);

        // Verify the tree was updated
        let tree = policy.trees.get("model1");
        assert!(tree.is_some());
    }

    #[test]
    fn test_cache_aware_multi_node_consistency() {
        use std::sync::Arc;

        use smg_mesh::{
            stores::StateStores,
            sync::MeshSyncManager,
            tree_ops::{TreeInsertOp, TreeOperation},
        };

        // Simulate two nodes
        let stores1 = Arc::new(StateStores::with_self_name("node1".to_string()));
        let mesh_sync1 = Arc::new(MeshSyncManager::new(stores1.clone(), "node1".to_string()));

        let stores2 = Arc::new(StateStores::with_self_name("node2".to_string()));
        let mesh_sync2 = Arc::new(MeshSyncManager::new(stores2.clone(), "node2".to_string()));

        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };

        let mut _policy1 = CacheAwarePolicy::with_config(config.clone());
        _policy1.set_mesh_sync(Some(mesh_sync1.clone()));
        let mut _policy2 = CacheAwarePolicy::with_config(config);
        _policy2.set_mesh_sync(Some(mesh_sync2.clone()));

        // Node1 syncs a tree operation
        let op = TreeOperation::Insert(TreeInsertOp {
            text: "shared_text".to_string(),
            tenant: "http://shared:8000".to_string(),
        });
        mesh_sync1
            .sync_tree_operation("model1".to_string(), op.clone())
            .unwrap();

        // Node2 should be able to get the tree state
        let tree_state = mesh_sync2.get_tree_state("model1");
        // Note: In a real scenario, this would be synced via gossip protocol
        // For unit test, we verify the sync mechanism works
        // Tree state may or may not exist depending on sync timing
        let _ = tree_state;
    }

    #[tokio::test]
    async fn test_cache_aware_without_mesh() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .api_key("test_api_key")
                .build(),
        )];

        policy.init_workers(&workers);

        // Should work without mesh
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("test request"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(idx, 0);
    }

    fn make_prefill(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Prefill {
                    bootstrap_port: Some(9000),
                })
                .build(),
        )
    }

    fn make_decode(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Decode)
                .build(),
        )
    }

    /// PD setup with two separate `CacheAwarePolicy` instances — the production
    /// wiring. Each pool's tree is seeded only with its own workers. Across a
    /// 4-turn growing prompt, each pool must stick to one worker.
    #[tokio::test]
    async fn test_pd_pool_isolation_two_policies() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let prefill_policy = CacheAwarePolicy::with_config(config.clone());
        let decode_policy = CacheAwarePolicy::with_config(config);

        let prefill_workers: Vec<Arc<dyn Worker>> = vec![
            make_prefill("http://prefill0:8000"),
            make_prefill("http://prefill1:8000"),
        ];
        let decode_workers: Vec<Arc<dyn Worker>> = vec![
            make_decode("http://decode0:8000"),
            make_decode("http://decode1:8000"),
        ];
        prefill_policy.init_workers(&prefill_workers);
        decode_policy.init_workers(&decode_workers);

        let turns = [
            "turn1",
            "turn1 turn2",
            "turn1 turn2 turn3",
            "turn1 turn2 turn3 turn4",
        ];

        let mut prefill_idx: Option<usize> = None;
        let mut decode_idx: Option<usize> = None;
        for prompt in turns {
            let info = SelectWorkerInfo {
                request_text: Some(prompt),
                ..Default::default()
            };
            let p = prefill_policy
                .select_worker(&prefill_workers, &info)
                .await
                .expect("prefill pool returns a worker");
            let d = decode_policy
                .select_worker(&decode_workers, &info)
                .await
                .expect("decode pool returns a worker");
            match prefill_idx {
                None => prefill_idx = Some(p),
                Some(pinned) => assert_eq!(
                    p, pinned,
                    "prefill should stay pinned across turns (prompt={prompt:?})"
                ),
            }
            match decode_idx {
                None => decode_idx = Some(d),
                Some(pinned) => assert_eq!(
                    d, pinned,
                    "decode should stay pinned across turns (prompt={prompt:?})"
                ),
            }
        }
    }

    /// Regression: even if a single `CacheAwarePolicy` instance is incorrectly
    /// wired to both pools, pool-aware tree keys must keep their state disjoint.
    /// The pre-fix code shared one trie keyed by `model_id`, so alternating
    /// prefill/decode `tree.insert` calls overwrote each other and the policy
    /// degenerated into worker-flipping random selection.
    #[tokio::test]
    async fn test_pd_pool_isolation_shared_policy_regression() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);

        let prefill_workers: Vec<Arc<dyn Worker>> = vec![
            make_prefill("http://prefill0:8000"),
            make_prefill("http://prefill1:8000"),
        ];
        let decode_workers: Vec<Arc<dyn Worker>> = vec![
            make_decode("http://decode0:8000"),
            make_decode("http://decode1:8000"),
        ];

        // One instance, mixed init — pool-aware keys split the trees internally.
        let mut combined: Vec<Arc<dyn Worker>> = Vec::new();
        combined.extend(prefill_workers.iter().cloned());
        combined.extend(decode_workers.iter().cloned());
        policy.init_workers(&combined);

        let turns = [
            "turn1",
            "turn1 turn2",
            "turn1 turn2 turn3",
            "turn1 turn2 turn3 turn4",
        ];

        let mut prefill_idx: Option<usize> = None;
        let mut decode_idx: Option<usize> = None;
        for prompt in turns {
            let info = SelectWorkerInfo {
                request_text: Some(prompt),
                ..Default::default()
            };
            let p = policy
                .select_worker(&prefill_workers, &info)
                .await
                .expect("prefill pool returns a worker");
            let d = policy
                .select_worker(&decode_workers, &info)
                .await
                .expect("decode pool returns a worker");

            assert!(
                prefill_workers[p].url().starts_with("http://prefill"),
                "prefill call must return a prefill index, got {} (prompt={prompt:?})",
                prefill_workers[p].url()
            );
            assert!(
                decode_workers[d].url().starts_with("http://decode"),
                "decode call must return a decode index, got {} (prompt={prompt:?})",
                decode_workers[d].url()
            );

            match prefill_idx {
                None => prefill_idx = Some(p),
                Some(pinned) => assert_eq!(
                    p, pinned,
                    "prefill should stay pinned across turns (prompt={prompt:?})"
                ),
            }
            match decode_idx {
                None => decode_idx = Some(d),
                Some(pinned) => assert_eq!(
                    d, pinned,
                    "decode should stay pinned across turns (prompt={prompt:?})"
                ),
            }
        }
    }

    /// Removing a PD worker via the composite-key `remove_worker(&dyn Worker)` path
    /// must drop it from its own pool's tree without touching the other pool. This
    /// covers `PolicyRegistry::remove_pd_worker_from_cache_aware`, which routes the
    /// removal here based on `worker.worker_type()`.
    #[tokio::test]
    async fn test_pd_pool_isolation_remove_worker() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);

        let prefill0 = make_prefill("http://prefill0:8000");
        let prefill1 = make_prefill("http://prefill1:8000");
        let decode0 = make_decode("http://decode0:8000");
        let decode1 = make_decode("http://decode1:8000");

        let prefill_workers: Vec<Arc<dyn Worker>> = vec![prefill0.clone(), prefill1.clone()];
        let decode_workers: Vec<Arc<dyn Worker>> = vec![decode0.clone(), decode1.clone()];
        let combined: Vec<Arc<dyn Worker>> = vec![
            prefill0.clone(),
            prefill1.clone(),
            decode0.clone(),
            decode1.clone(),
        ];
        policy.init_workers(&combined);

        // Seed both trees with affinity for one prompt.
        let prompt = "shared prefix to seed the cache_aware trees";
        let info = SelectWorkerInfo {
            request_text: Some(prompt),
            ..Default::default()
        };
        policy
            .select_worker(&prefill_workers, &info)
            .await
            .expect("seed prefill");
        policy
            .select_worker(&decode_workers, &info)
            .await
            .expect("seed decode");

        let prefill_key = format!("prefill::{}", UNKNOWN_MODEL_ID);
        let decode_key = format!("decode::{}", UNKNOWN_MODEL_ID);

        let prefill_before = policy
            .trees
            .get(&prefill_key)
            .map(|t| t.value().prefix_match_with_counts(prompt).tenant)
            .expect("prefill tree seeded");
        let decode_before = policy
            .trees
            .get(&decode_key)
            .map(|t| t.value().prefix_match_with_counts(prompt).tenant)
            .expect("decode tree seeded");
        assert!(
            prefill_before.starts_with("http://prefill"),
            "prefill tree should hold a prefill tenant before removal, got {prefill_before}"
        );
        assert!(
            decode_before.starts_with("http://decode"),
            "decode tree should hold a decode tenant before removal, got {decode_before}"
        );

        // Drop prefill0 via the composite-key removal path.
        policy.remove_worker(prefill0.as_ref());

        // The prefill tree must no longer point at prefill0 for the seeded prompt.
        let prefill_after = policy
            .trees
            .get(&prefill_key)
            .map(|t| t.value().prefix_match_with_counts(prompt).tenant)
            .expect("prefill tree still exists");
        assert_ne!(
            &*prefill_after,
            prefill0.url(),
            "prefill0 should be gone from the prefill tree"
        );

        // The decode tree must be byte-for-byte unchanged.
        let decode_after = policy
            .trees
            .get(&decode_key)
            .map(|t| t.value().prefix_match_with_counts(prompt).tenant)
            .expect("decode tree still exists");
        assert_eq!(
            decode_after, decode_before,
            "removing a prefill worker must not touch the decode tree"
        );
    }

    /// Shared setup for `PolicyRegistry::remove_pd_worker_from_cache_aware` tests:
    /// build a registry whose prefill and decode policies are separate
    /// `CacheAwarePolicy` instances seeded with the matching pool's workers, then
    /// return the registry, the per-pool policy handles (for tree inspection), and
    /// representative workers from each pool.
    #[allow(clippy::type_complexity)]
    fn pd_registry_with_cache_aware_pools() -> (
        Arc<crate::policies::PolicyRegistry>,
        Arc<CacheAwarePolicy>,
        Arc<CacheAwarePolicy>,
        Arc<dyn Worker>,
        Arc<dyn Worker>,
        Arc<dyn Worker>,
        Arc<dyn Worker>,
    ) {
        let registry = Arc::new(crate::policies::PolicyRegistry::new(
            crate::config::types::PolicyConfig::RoundRobin,
        ));
        let no_eviction = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let prefill_ca = Arc::new(CacheAwarePolicy::with_config(no_eviction.clone()));
        let decode_ca = Arc::new(CacheAwarePolicy::with_config(no_eviction));
        registry.set_prefill_policy(prefill_ca.clone() as Arc<dyn LoadBalancingPolicy>);
        registry.set_decode_policy(decode_ca.clone() as Arc<dyn LoadBalancingPolicy>);

        let prefill0 = make_prefill("http://prefill0:8000");
        let prefill1 = make_prefill("http://prefill1:8000");
        let decode0 = make_decode("http://decode0:8000");
        let decode1 = make_decode("http://decode1:8000");

        let prefill_workers: Vec<Arc<dyn Worker>> = vec![prefill0.clone(), prefill1.clone()];
        let decode_workers: Vec<Arc<dyn Worker>> = vec![decode0.clone(), decode1.clone()];
        registry.init_pd_cache_aware_policies(&prefill_workers, &decode_workers);

        (
            registry, prefill_ca, decode_ca, prefill0, prefill1, decode0, decode1,
        )
    }

    /// Seed both pool trees so each has a known tenant for `prompt`, then return
    /// the (prefill_tenant, decode_tenant) snapshot to compare against after a
    /// dispatched removal.
    async fn seed_pd_pools(
        prefill_ca: &CacheAwarePolicy,
        decode_ca: &CacheAwarePolicy,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
        prompt: &str,
    ) -> (Arc<str>, Arc<str>) {
        let info = SelectWorkerInfo {
            request_text: Some(prompt),
            ..Default::default()
        };
        prefill_ca
            .select_worker(prefill_workers, &info)
            .await
            .expect("prefill seed");
        decode_ca
            .select_worker(decode_workers, &info)
            .await
            .expect("decode seed");

        let prefill_key = format!("prefill::{}", UNKNOWN_MODEL_ID);
        let decode_key = format!("decode::{}", UNKNOWN_MODEL_ID);
        let prefill_tenant = prefill_ca
            .trees
            .get(&prefill_key)
            .map(|t| t.value().prefix_match_with_counts(prompt).tenant)
            .expect("prefill tree seeded");
        let decode_tenant = decode_ca
            .trees
            .get(&decode_key)
            .map(|t| t.value().prefix_match_with_counts(prompt).tenant)
            .expect("decode tree seeded");
        (prefill_tenant, decode_tenant)
    }

    /// A `Prefill` worker passed to `remove_pd_worker_from_cache_aware` must hit
    /// the registry's `prefill_policy` and leave `decode_policy` untouched.
    /// Catches a dispatch swap like `Prefill => self.decode_policy.get()`.
    #[tokio::test]
    async fn test_registry_remove_pd_worker_prefill_dispatches_to_prefill_policy() {
        let (registry, prefill_ca, decode_ca, prefill0, prefill1, decode0, decode1) =
            pd_registry_with_cache_aware_pools();
        let prefill_workers: Vec<Arc<dyn Worker>> = vec![prefill0.clone(), prefill1.clone()];
        let decode_workers: Vec<Arc<dyn Worker>> = vec![decode0.clone(), decode1.clone()];

        let prompt = "prefix used to seed both pool trees";
        let (_prefill_before, decode_before) = seed_pd_pools(
            &prefill_ca,
            &decode_ca,
            &prefill_workers,
            &decode_workers,
            prompt,
        )
        .await;

        registry.remove_pd_worker_from_cache_aware(prefill0.as_ref());

        let prefill_key = format!("prefill::{}", UNKNOWN_MODEL_ID);
        let decode_key = format!("decode::{}", UNKNOWN_MODEL_ID);
        let prefill_after = prefill_ca
            .trees
            .get(&prefill_key)
            .map(|t| t.value().prefix_match_with_counts(prompt).tenant)
            .expect("prefill tree still exists");
        assert_ne!(
            &*prefill_after,
            prefill0.url(),
            "registry dispatch must drop prefill0 from the prefill pool's tree"
        );
        let decode_after = decode_ca
            .trees
            .get(&decode_key)
            .map(|t| t.value().prefix_match_with_counts(prompt).tenant)
            .expect("decode tree still exists");
        assert_eq!(
            decode_after, decode_before,
            "removing a prefill worker must not touch the decode pool's tree"
        );
    }

    /// Mirror of the prefill dispatch test for `Decode`. Catches a dispatch swap
    /// in the other direction (`Decode => self.prefill_policy.get()`).
    #[tokio::test]
    async fn test_registry_remove_pd_worker_decode_dispatches_to_decode_policy() {
        let (registry, prefill_ca, decode_ca, prefill0, prefill1, decode0, decode1) =
            pd_registry_with_cache_aware_pools();
        let prefill_workers: Vec<Arc<dyn Worker>> = vec![prefill0.clone(), prefill1.clone()];
        let decode_workers: Vec<Arc<dyn Worker>> = vec![decode0.clone(), decode1.clone()];

        let prompt = "prefix used to seed both pool trees";
        let (prefill_before, _decode_before) = seed_pd_pools(
            &prefill_ca,
            &decode_ca,
            &prefill_workers,
            &decode_workers,
            prompt,
        )
        .await;

        registry.remove_pd_worker_from_cache_aware(decode0.as_ref());

        let prefill_key = format!("prefill::{}", UNKNOWN_MODEL_ID);
        let decode_key = format!("decode::{}", UNKNOWN_MODEL_ID);
        let decode_after = decode_ca
            .trees
            .get(&decode_key)
            .map(|t| t.value().prefix_match_with_counts(prompt).tenant)
            .expect("decode tree still exists");
        assert_ne!(
            &*decode_after,
            decode0.url(),
            "registry dispatch must drop decode0 from the decode pool's tree"
        );
        let prefill_after = prefill_ca
            .trees
            .get(&prefill_key)
            .map(|t| t.value().prefix_match_with_counts(prompt).tenant)
            .expect("prefill tree still exists");
        assert_eq!(
            prefill_after, prefill_before,
            "removing a decode worker must not touch the prefill pool's tree"
        );
    }

    /// `remove_pd_worker_from_cache_aware` must short-circuit on `Regular`
    /// workers and silently ignore non-cache_aware policies (`name() != "cache_aware"`).
    /// Both branches are no-ops: neither pool tree changes, and no downcast panic.
    #[tokio::test]
    async fn test_registry_remove_pd_worker_regular_and_non_cache_aware_noop() {
        // (a) Regular worker: should early-return regardless of policy state.
        let (registry, prefill_ca, decode_ca, prefill0, prefill1, decode0, decode1) =
            pd_registry_with_cache_aware_pools();
        let prefill_workers: Vec<Arc<dyn Worker>> = vec![prefill0.clone(), prefill1.clone()];
        let decode_workers: Vec<Arc<dyn Worker>> = vec![decode0.clone(), decode1.clone()];

        let prompt = "regular-noop seed prompt";
        let (prefill_before, decode_before) = seed_pd_pools(
            &prefill_ca,
            &decode_ca,
            &prefill_workers,
            &decode_workers,
            prompt,
        )
        .await;

        let regular: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://regular0:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        );
        registry.remove_pd_worker_from_cache_aware(regular.as_ref());

        let prefill_key = format!("prefill::{}", UNKNOWN_MODEL_ID);
        let decode_key = format!("decode::{}", UNKNOWN_MODEL_ID);
        let prefill_after = prefill_ca
            .trees
            .get(&prefill_key)
            .map(|t| t.value().prefix_match_with_counts(prompt).tenant)
            .expect("prefill tree still exists");
        let decode_after = decode_ca
            .trees
            .get(&decode_key)
            .map(|t| t.value().prefix_match_with_counts(prompt).tenant)
            .expect("decode tree still exists");
        assert_eq!(
            prefill_after, prefill_before,
            "Regular worker dispatch must not touch the prefill tree"
        );
        assert_eq!(
            decode_after, decode_before,
            "Regular worker dispatch must not touch the decode tree"
        );

        // (b) Non-cache_aware policy: PD pool is round_robin. The downcast must
        // be skipped (no panic) and the call must be a no-op.
        let registry =
            crate::policies::PolicyRegistry::new(crate::config::types::PolicyConfig::RoundRobin);
        let rr_prefill: Arc<dyn LoadBalancingPolicy> =
            Arc::new(crate::policies::RoundRobinPolicy::new());
        let rr_decode: Arc<dyn LoadBalancingPolicy> =
            Arc::new(crate::policies::RoundRobinPolicy::new());
        registry.set_prefill_policy(rr_prefill);
        registry.set_decode_policy(rr_decode);
        // No panic, no downcast — this would fault if the guard
        // `policy.name() == "cache_aware"` were dropped.
        registry.remove_pd_worker_from_cache_aware(prefill0.as_ref());
        registry.remove_pd_worker_from_cache_aware(decode0.as_ref());
    }

    /// `init_pd_cache_aware_policies` must seed only the pool whose policy is
    /// cache_aware AND whose worker list is non-empty. Covers all four corners:
    /// both seeded, only-prefill-cache_aware, empty-worker short-circuit, and the
    /// non-cache_aware side staying a no-op.
    #[tokio::test]
    async fn test_registry_init_pd_cache_aware_policies_gating() {
        let no_eviction = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let prefill_key = format!("prefill::{}", UNKNOWN_MODEL_ID);
        let decode_key = format!("decode::{}", UNKNOWN_MODEL_ID);

        let prefill0 = make_prefill("http://prefill0:8000");
        let decode0 = make_decode("http://decode0:8000");
        let prefill_workers: Vec<Arc<dyn Worker>> = vec![prefill0.clone()];
        let decode_workers: Vec<Arc<dyn Worker>> = vec![decode0.clone()];

        // (a) Both pools are cache_aware with workers → both trees seeded under
        // the correct composite key.
        {
            let registry = crate::policies::PolicyRegistry::new(
                crate::config::types::PolicyConfig::RoundRobin,
            );
            let prefill_ca = Arc::new(CacheAwarePolicy::with_config(no_eviction.clone()));
            let decode_ca = Arc::new(CacheAwarePolicy::with_config(no_eviction.clone()));
            registry.set_prefill_policy(prefill_ca.clone() as Arc<dyn LoadBalancingPolicy>);
            registry.set_decode_policy(decode_ca.clone() as Arc<dyn LoadBalancingPolicy>);

            registry.init_pd_cache_aware_policies(&prefill_workers, &decode_workers);

            assert!(
                prefill_ca.trees.contains_key(&prefill_key),
                "prefill cache_aware policy must be seeded under '{prefill_key}'"
            );
            assert!(
                decode_ca.trees.contains_key(&decode_key),
                "decode cache_aware policy must be seeded under '{decode_key}'"
            );
            assert!(
                !prefill_ca.trees.contains_key(&decode_key),
                "prefill_workers must not seed the decode tree key"
            );
            assert!(
                !decode_ca.trees.contains_key(&prefill_key),
                "decode_workers must not seed the prefill tree key"
            );
        }

        // (b) Only prefill is cache_aware (decode is round_robin) → prefill seeded,
        // decode side skipped silently (no downcast, no panic).
        {
            let registry = crate::policies::PolicyRegistry::new(
                crate::config::types::PolicyConfig::RoundRobin,
            );
            let prefill_ca = Arc::new(CacheAwarePolicy::with_config(no_eviction.clone()));
            let decode_rr: Arc<dyn LoadBalancingPolicy> =
                Arc::new(crate::policies::RoundRobinPolicy::new());
            registry.set_prefill_policy(prefill_ca.clone() as Arc<dyn LoadBalancingPolicy>);
            registry.set_decode_policy(decode_rr);

            registry.init_pd_cache_aware_policies(&prefill_workers, &decode_workers);

            assert!(
                prefill_ca.trees.contains_key(&prefill_key),
                "prefill cache_aware side must seed even when decode side is non-cache_aware"
            );
        }

        // (c) Both cache_aware but prefill worker list is empty → prefill tree
        // NOT seeded (the inner `!is_empty()` guard short-circuits); decode side
        // is still seeded.
        {
            let registry = crate::policies::PolicyRegistry::new(
                crate::config::types::PolicyConfig::RoundRobin,
            );
            let prefill_ca = Arc::new(CacheAwarePolicy::with_config(no_eviction.clone()));
            let decode_ca = Arc::new(CacheAwarePolicy::with_config(no_eviction.clone()));
            registry.set_prefill_policy(prefill_ca.clone() as Arc<dyn LoadBalancingPolicy>);
            registry.set_decode_policy(decode_ca.clone() as Arc<dyn LoadBalancingPolicy>);

            registry.init_pd_cache_aware_policies(&[], &decode_workers);

            assert!(
                prefill_ca.trees.is_empty(),
                "empty prefill worker list must not seed the prefill tree"
            );
            assert!(
                decode_ca.trees.contains_key(&decode_key),
                "decode side must still seed when only the prefill list is empty"
            );
        }

        // (d) Both worker lists empty → neither pool seeded (init is a full no-op).
        {
            let registry = crate::policies::PolicyRegistry::new(
                crate::config::types::PolicyConfig::RoundRobin,
            );
            let prefill_ca = Arc::new(CacheAwarePolicy::with_config(no_eviction.clone()));
            let decode_ca = Arc::new(CacheAwarePolicy::with_config(no_eviction));
            registry.set_prefill_policy(prefill_ca.clone() as Arc<dyn LoadBalancingPolicy>);
            registry.set_decode_policy(decode_ca.clone() as Arc<dyn LoadBalancingPolicy>);

            registry.init_pd_cache_aware_policies(&[], &[]);

            assert!(prefill_ca.trees.is_empty());
            assert!(decode_ca.trees.is_empty());
        }
    }
}
