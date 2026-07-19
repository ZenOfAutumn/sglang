//! 面向多路由器场景的 Worker 注册表(Worker Registry)。
//!
//! 为所有后端 worker 提供集中式注册管理,并按模型(model_id)建立索引。
//!
//! # 性能优化
//! 模型索引使用不可变的 Arc 快照(而非 RwLock)以实现无锁读取。
//! 这对「大量请求查询同一模型」的高并发场景至关重要。
//!
//! # 一致性哈希环
//! 注册表为每个模型维护一个预计算的哈希环,以 O(log n) 完成一致性哈希路由。
//! 哈希环仅在 worker 增删时重建,而非每次请求都重建。
//! 采用虚拟节点(每个 worker 150 个)保证分布均匀,并用 blake3 做稳定哈希。

use std::sync::{Arc, RwLock};

use dashmap::DashMap;
use smg_mesh::OptionalMeshSyncManager;
use uuid::Uuid;

use crate::{
    core::{
        circuit_breaker::CircuitState,
        worker::{HealthChecker, RuntimeType, WorkerType},
        ConnectionMode, Worker,
    },
    observability::metrics::Metrics,
};

/// 每个物理 worker 对应的虚拟节点数量,用于让 key 分布更均匀。
/// 150 是常见取值,在内存占用与分布均匀度之间取得较好平衡。
const VIRTUAL_NODES_PER_WORKER: usize = 150;

/// 一致性哈希环,用于 O(log n) 选择 worker。
///
/// 每个 worker 按 hash(worker_url + 虚拟节点序号) 被放到环上的多个位置(虚拟节点),带来:
/// - key 在各 worker 间分布均匀;
/// - worker 增删时仅需重分布约 1/N 的 key(N 为 worker 数);
/// - 通过二分查找实现 O(log n) 定位。
///
/// 采用 blake3 做稳定且快速的哈希,结果在不同 Rust 版本间保持一致。
#[derive(Debug, Clone)]
pub struct HashRing {
    /// 按环上位置排序的 (ring_position, worker_url) 列表。
    /// 每个 worker 有多个条目(虚拟节点)以保证分布均匀。
    /// 用 Arc<str> 让 URL 在所有虚拟节点间共享(150 个引用而非 150 份拷贝)。
    entries: Arc<[(u64, Arc<str>)]>,
}

impl HashRing {
    /// 根据一组 worker 构建哈希环。
    /// 为每个 worker 创建 VIRTUAL_NODES_PER_WORKER 个条目以保证分布均匀。
    pub fn new(workers: &[Arc<dyn Worker>]) -> Self {
        let mut entries: Vec<(u64, Arc<str>)> =
            Vec::with_capacity(workers.len() * VIRTUAL_NODES_PER_WORKER);

        for worker in workers {
            // 每个 worker 只创建一次 Arc<str>,在所有虚拟节点间共享
            let url: Arc<str> = Arc::from(worker.url());
            let url_bytes = url.as_bytes();

            // 为每个 worker 创建多个虚拟节点
            for vnode in 0..VIRTUAL_NODES_PER_WORKER {
                let mut hasher = blake3::Hasher::new();
                hasher.update(url_bytes);
                hasher.update(b"#");
                hasher.update(&(vnode as u64).to_le_bytes());
                let hash = hasher.finalize();
                let pos = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
                entries.push((pos, Arc::clone(&url)));
            }
        }

        // 按环上位置排序,便于后续二分查找
        entries.sort_unstable_by_key(|(pos, _)| *pos);

        Self {
            entries: Arc::from(entries.into_boxed_slice()),
        }
    }

    /// 用 blake3 把字符串哈希成环上位置(跨版本稳定)。
    #[inline]
    fn hash_position(s: &str) -> u64 {
        let hash = blake3::hash(s.as_bytes());
        // 取前 8 字节作为 u64
        u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap())
    }

    /// 用一致性哈希为某个 key 查找目标 worker URL。
    /// 从 key 所在位置起沿顺时针方向,返回第一个健康的 worker URL。
    ///
    /// - `key`:待哈希的路由键
    /// - `is_healthy`:判断某个 worker URL 是否健康的回调
    pub fn find_healthy_url<F>(&self, key: &str, is_healthy: F) -> Option<&str>
    where
        F: Fn(&str) -> bool,
    {
        if self.entries.is_empty() {
            return None;
        }

        let key_pos = Self::hash_position(key);

        // 二分查找定位到 key_pos 处或其之后的第一个条目
        let start = self.entries.partition_point(|(pos, _)| *pos < key_pos);

        // 从 start 起顺时针遍历(到末尾则环回)
        // 记录已检查过的 URL,避免因虚拟节点重复检查同一 worker
        let mut checked_urls =
            std::collections::HashSet::with_capacity(self.worker_count().min(16));

        for i in 0..self.entries.len() {
            let (_, url) = &self.entries[(start + i) % self.entries.len()];
            let url_str: &str = url;

            // 若该 worker 已检查过(来自另一个虚拟节点)则跳过
            if !checked_urls.insert(url_str) {
                continue;
            }

            if is_healthy(url_str) {
                return Some(url_str);
            }
        }

        None
    }

    /// 判断环是否为空
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 获取环上条目总数(含虚拟节点)
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 获取环上不重复的 worker 数量
    pub fn worker_count(&self) -> usize {
        self.entries.len() / VIRTUAL_NODES_PER_WORKER.max(1)
    }
}

/// worker 的唯一标识符(内部为 UUID 字符串)。
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct WorkerId(String);

impl WorkerId {
    /// 生成一个新的 worker ID(随机 UUID v4)。
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// 从已有字符串构造 worker ID。
    pub fn from_string(s: String) -> Self {
        Self(s)
    }

    /// 以字符串形式获取该 ID。
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for WorkerId {
    fn default() -> Self {
        Self::new()
    }
}

/// 模型索引类型别名,使用不可变快照实现无锁读取。
/// 每个模型映射到一个 Arc 包裹的 worker 切片,读取时无需加锁。
/// 更新时创建新快照(写时复制 / copy-on-write 语义)。
type ModelIndex = Arc<DashMap<String, Arc<[Arc<dyn Worker>]>>>;

/// 按模型建立索引的 worker 注册表。
///
/// 内部维护多份互相关联的索引(按 ID / 模型 / 类型 / 连接模式 / URL),
/// 增删 worker 时会同步更新所有索引并重建对应模型的哈希环,以此换取读路径上的
/// 无锁、O(1)/O(log n) 快速查询。所有字段都用 `Arc` + `DashMap` 包裹,
/// 使注册表本身可被廉价克隆、跨线程共享。
#[derive(Debug)]
pub struct WorkerRegistry {
    /// 全部 worker,按 WorkerId 索引(权威数据源)。
    workers: Arc<DashMap<WorkerId, Arc<dyn Worker>>>,

    /// 模型索引,用不可变快照实现 O(1) 无锁查询。
    /// 用 Arc<[T]> 而非 Arc<RwLock<Vec<T>>>,读取只需一次原子引用计数。
    model_index: ModelIndex,

    /// 每个模型对应的一致性哈希环,用于 O(log n) 路由。
    /// 在 worker 增删时按写时复制方式重建。
    hash_rings: Arc<DashMap<String, Arc<HashRing>>>,

    /// 按 worker 类型(Regular / Prefill / Decode)索引 worker ID。
    type_workers: Arc<DashMap<WorkerType, Vec<WorkerId>>>,

    /// 按连接模式(Http / Grpc)索引 worker ID。
    connection_workers: Arc<DashMap<ConnectionMode, Vec<WorkerId>>>,

    /// URL 到 WorkerId 的映射,用于按地址快速定位 worker。
    url_to_id: Arc<DashMap<String, WorkerId>>,

    /// 可选的 mesh 状态同步管理器,用于多 Router 间同步 worker 状态。
    /// 为 None 时注册表独立工作,不做 mesh 同步。
    /// 用 RwLock 包裹以支持初始化后再线程安全地设置 mesh_sync。
    mesh_sync: Arc<RwLock<OptionalMeshSyncManager>>,
}

impl WorkerRegistry {
    /// 创建一个新的空 worker 注册表。
    pub fn new() -> Self {
        Self {
            workers: Arc::new(DashMap::new()),
            model_index: Arc::new(DashMap::new()),
            hash_rings: Arc::new(DashMap::new()),
            type_workers: Arc::new(DashMap::new()),
            connection_workers: Arc::new(DashMap::new()),
            url_to_id: Arc::new(DashMap::new()),
            mesh_sync: Arc::new(RwLock::new(None)),
        }
    }

    /// 根据模型索引中当前的 worker 列表,重建该模型的哈希环。
    fn rebuild_hash_ring(&self, model_id: &str) {
        if let Some(workers) = self.model_index.get(model_id) {
            let ring = HashRing::new(&workers);
            self.hash_rings.insert(model_id.to_string(), Arc::new(ring));
        } else {
            // 该模型已无 worker,移除对应哈希环
            self.hash_rings.remove(model_id);
        }
    }

    /// 获取某个模型的哈希环(O(1) 查询)。
    pub fn get_hash_ring(&self, model_id: &str) -> Option<Arc<HashRing>> {
        self.hash_rings.get(model_id).map(|r| Arc::clone(&r))
    }

    /// 设置 mesh 同步管理器(线程安全,可在初始化之后调用)。
    pub fn set_mesh_sync(&self, mesh_sync: OptionalMeshSyncManager) {
        *self.mesh_sync.write().unwrap() = mesh_sync;
    }

    /// 注册一个新 worker(若同 URL 已存在则更新),返回其 WorkerId。
    pub fn register(&self, worker: Arc<dyn Worker>) -> WorkerId {
        let worker_id = if let Some(existing_id) = self.url_to_id.get(worker.url()) {
            // 该 URL 的 worker 已存在,复用其 ID 进行更新
            existing_id.clone()
        } else {
            WorkerId::new()
        };

        // 存入 worker 主表
        self.workers.insert(worker_id.clone(), worker.clone());

        // 更新 URL 映射
        self.url_to_id
            .insert(worker.url().to_string(), worker_id.clone());

        // 用写时复制方式更新模型索引以支持 O(1) 查询:
        // 生成一个包含新增 worker 的全新不可变快照
        let model_id = worker.model_id().to_string();
        self.model_index
            .entry(model_id.clone())
            .and_modify(|existing| {
                // 基于旧快照生成含新 worker 的新快照
                let mut new_workers: Vec<Arc<dyn Worker>> = existing.iter().cloned().collect();
                new_workers.push(worker.clone());
                *existing = Arc::from(new_workers.into_boxed_slice());
            })
            .or_insert_with(|| Arc::from(vec![worker.clone()].into_boxed_slice()));

        // 重建该模型的哈希环
        self.rebuild_hash_ring(&model_id);

        // 更新类型索引(DashMap 的 key 需要所有权,故 clone)
        self.type_workers
            .entry(worker.worker_type().clone())
            .or_default()
            .push(worker_id.clone());

        // 更新连接模式索引(DashMap 的 key 需要所有权,故 clone)
        self.connection_workers
            .entry(worker.connection_mode().clone())
            .or_default()
            .push(worker_id.clone());

        // 若启用了 mesh 则同步状态(未启用时为空操作)
        if let Some(ref mesh_sync) = *self.mesh_sync.read().unwrap() {
            mesh_sync.sync_worker_state(
                worker_id.as_str().to_string(),
                worker.model_id().to_string(),
                worker.url().to_string(),
                worker.is_healthy(),
                0.0, // TODO: 获取真实负载
            );
        }

        worker_id
    }

    /// 为某个 worker URL 预留(或取回)一个稳定的 UUID。
    /// 使用原子性的 entry API,避免「检查再插入」之间的竞态。
    pub fn reserve_id_for_url(&self, url: &str) -> WorkerId {
        self.url_to_id.entry(url.to_string()).or_default().clone()
    }

    /// 尽力而为地根据 worker ID 查找其 URL。
    pub fn get_url_by_id(&self, worker_id: &WorkerId) -> Option<String> {
        if let Some(worker) = self.get(worker_id) {
            return Some(worker.url().to_string());
        }
        self.url_to_id
            .iter()
            .find_map(|entry| (entry.value() == worker_id).then(|| entry.key().clone()))
    }

    /// 根据 ID 移除一个 worker,返回被移除的 worker。
    pub fn remove(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        if let Some((_, worker)) = self.workers.remove(worker_id) {
            // 从 URL 映射中移除
            self.url_to_id.remove(worker.url());

            // 以写时复制方式从模型索引中移除:
            // 生成一个不含该 worker 的新快照
            let worker_url = worker.url();
            let model_id = worker.model_id().to_string();
            if let Some(mut entry) = self.model_index.get_mut(&model_id) {
                let new_workers: Vec<Arc<dyn Worker>> = entry
                    .iter()
                    .filter(|w| w.url() != worker_url)
                    .cloned()
                    .collect();
                *entry = Arc::from(new_workers.into_boxed_slice());
            }

            // 重建该模型的哈希环
            self.rebuild_hash_ring(&model_id);

            // 从类型索引中移除
            if let Some(mut type_workers) = self.type_workers.get_mut(worker.worker_type()) {
                type_workers.retain(|id| id != worker_id);
            }

            // 从连接模式索引中移除
            if let Some(mut conn_workers) =
                self.connection_workers.get_mut(worker.connection_mode())
            {
                conn_workers.retain(|id| id != worker_id);
            }

            worker.set_healthy(false);
            Metrics::remove_worker_metrics(worker.url());

            // 若启用了 mesh 则同步移除(未启用时为空操作)
            if let Some(ref mesh_sync) = *self.mesh_sync.read().unwrap() {
                mesh_sync.remove_worker_state(worker_id.as_str());
            }

            Some(worker)
        } else {
            None
        }
    }

    /// 根据 URL 移除一个 worker。
    pub fn remove_by_url(&self, url: &str) -> Option<Arc<dyn Worker>> {
        if let Some((_, worker_id)) = self.url_to_id.remove(url) {
            self.remove(&worker_id)
        } else {
            None
        }
    }

    /// 根据 ID 获取 worker。
    pub fn get(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        self.workers.get(worker_id).map(|entry| entry.clone())
    }

    /// 根据 URL 获取 worker。
    pub fn get_by_url(&self, url: &str) -> Option<Arc<dyn Worker>> {
        self.url_to_id.get(url).and_then(|id| self.get(&id))
    }

    /// 空 worker 切片常量,用于「未找到 worker」时返回。
    const EMPTY_WORKERS: &'static [Arc<dyn Worker>] = &[];

    /// 获取某个模型的全部 worker(O(1) 优化、无锁)。
    /// 返回指向不可变 worker 切片的 Arc,仅是一次原子引用计数自增。
    /// 这是零竞争、开销最低的读路径。
    pub fn get_by_model(&self, model_id: &str) -> Arc<[Arc<dyn Worker>]> {
        self.model_index
            .get(model_id)
            .map(|workers| Arc::clone(&workers))
            .unwrap_or_else(|| Arc::from(Self::EMPTY_WORKERS))
    }

    /// 按 worker 类型获取全部 worker。
    pub fn get_by_type(&self, worker_type: &WorkerType) -> Vec<Arc<dyn Worker>> {
        self.type_workers
            .get(worker_type)
            .map(|ids| ids.iter().filter_map(|id| self.get(id)).collect())
            .unwrap_or_default()
    }

    /// 更新 worker 健康状态并同步到 mesh。
    pub fn update_worker_health(&self, worker_id: &WorkerId, is_healthy: bool) {
        if let Some(worker) = self.workers.get(worker_id) {
            // 更新 worker 健康状态(若 Worker trait 提供了相应方法)
            // 目前仅同步到 mesh

            // 若启用了 mesh 则同步(未启用时为空操作)
            if let Some(ref mesh_sync) = *self.mesh_sync.read().unwrap() {
                mesh_sync.sync_worker_state(
                    worker_id.as_str().to_string(),
                    worker.model_id().to_string(),
                    worker.url().to_string(),
                    is_healthy,
                    0.0, // TODO: 获取真实负载
                );
            }
        }
    }

    /// 获取全部 prefill worker(不区分 bootstrap_port)。
    pub fn get_prefill_workers(&self) -> Vec<Arc<dyn Worker>> {
        self.workers
            .iter()
            .filter_map(|entry| {
                let worker = entry.value();
                match worker.worker_type() {
                    WorkerType::Prefill { .. } => Some(worker.clone()),
                    _ => None,
                }
            })
            .collect()
    }

    /// 获取全部 decode worker。
    pub fn get_decode_workers(&self) -> Vec<Arc<dyn Worker>> {
        self.get_by_type(&WorkerType::Decode)
    }

    /// 按连接模式获取全部 worker。
    pub fn get_by_connection(&self, connection_mode: &ConnectionMode) -> Vec<Arc<dyn Worker>> {
        self.connection_workers
            .get(connection_mode)
            .map(|ids| ids.iter().filter_map(|id| self.get(id)).collect())
            .unwrap_or_default()
    }

    /// 获取注册表中 worker 的数量。
    pub fn len(&self) -> usize {
        self.workers.len()
    }

    /// 判断注册表是否为空。
    pub fn is_empty(&self) -> bool {
        self.workers.is_empty()
    }

    /// 获取全部 worker。
    pub fn get_all(&self) -> Vec<Arc<dyn Worker>> {
        self.workers
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// 获取全部 worker 及其 ID。
    pub fn get_all_with_ids(&self) -> Vec<(WorkerId, Arc<dyn Worker>)> {
        self.workers
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    /// 获取全部 worker 的 URL。
    pub fn get_all_urls(&self) -> Vec<String> {
        self.workers
            .iter()
            .map(|entry| entry.value().url().to_string())
            .collect()
    }

    pub fn get_all_urls_with_api_key(&self) -> Vec<(String, Option<String>)> {
        self.workers
            .iter()
            .map(|entry| {
                (
                    entry.value().url().to_string(),
                    entry.value().api_key().clone(),
                )
            })
            .collect()
    }

    /// 获取所有拥有 worker 的模型 ID(无锁)。
    pub fn get_models(&self) -> Vec<String> {
        self.model_index
            .iter()
            .filter(|entry| !entry.value().is_empty())
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// 按多个条件过滤获取 worker。
    ///
    /// 支持灵活组合以下过滤条件:
    /// - model_id:按指定模型过滤
    /// - worker_type:按 worker 类型过滤(Regular、Prefill、Decode)
    /// - connection_mode:按连接模式过滤(Http、Grpc)
    /// - runtime_type:按运行时类型过滤(Sglang、Vllm、External)
    /// - healthy_only:仅返回健康的 worker
    pub fn get_workers_filtered(
        &self,
        model_id: Option<&str>,
        worker_type: Option<WorkerType>,
        connection_mode: Option<ConnectionMode>,
        runtime_type: Option<RuntimeType>,
        healthy_only: bool,
    ) -> Vec<Arc<dyn Worker>> {
        // 根据过滤条件选取开销最小的初始集合:
        // 能用模型索引就用(O(1) 查询)
        let workers: Vec<Arc<dyn Worker>> = if let Some(model) = model_id {
            self.get_by_model(model).to_vec()
        } else {
            self.get_all()
        };

        // 应用其余过滤条件
        workers
            .into_iter()
            .filter(|w| {
                // 若指定了 worker_type 则检查
                if let Some(ref wtype) = worker_type {
                    if *w.worker_type() != *wtype {
                        return false;
                    }
                }

                // 若指定了 connection_mode 则检查(用 matches 以兼容 gRPC 的灵活匹配)
                if let Some(ref conn) = connection_mode {
                    if !w.connection_mode().matches(conn) {
                        return false;
                    }
                }

                // 若指定了 runtime_type 则检查
                if let Some(ref rt) = runtime_type {
                    if w.metadata().runtime_type != *rt {
                        return false;
                    }
                }

                // 若要求只要健康的则检查健康状态
                if healthy_only && !w.is_healthy() {
                    return false;
                }

                true
            })
            .collect()
    }

    /// 获取 worker 统计信息(无锁)。
    pub fn stats(&self) -> WorkerRegistryStats {
        let total_workers = self.workers.len();
        // 直接计数,避免通过 get_models() 分配 Vec(无锁)
        let total_models = self
            .model_index
            .iter()
            .filter(|entry| !entry.value().is_empty())
            .count();

        let mut healthy_count = 0;
        let mut total_load = 0;
        let mut regular_count = 0;
        let mut prefill_count = 0;
        let mut decode_count = 0;
        let mut http_count = 0;
        let mut grpc_count = 0;
        let mut cb_open_count = 0;
        let mut cb_half_open_count = 0;

        // 直接遍历 DashMap,避免通过 get_all() 克隆所有 worker
        for entry in self.workers.iter() {
            let worker = entry.value();
            if worker.is_healthy() {
                healthy_count += 1;
            }
            total_load += worker.load();

            match worker.worker_type() {
                WorkerType::Regular => regular_count += 1,
                WorkerType::Prefill { .. } => prefill_count += 1,
                WorkerType::Decode => decode_count += 1,
            }

            match worker.connection_mode() {
                ConnectionMode::Http => http_count += 1,
                ConnectionMode::Grpc { .. } => grpc_count += 1,
            }

            match worker.circuit_breaker().state() {
                CircuitState::Open => cb_open_count += 1,
                CircuitState::HalfOpen => cb_half_open_count += 1,
                CircuitState::Closed => {}
            }
        }

        WorkerRegistryStats {
            total_workers,
            total_models,
            healthy_workers: healthy_count,
            unhealthy_workers: total_workers.saturating_sub(healthy_count),
            total_load,
            regular_workers: regular_count,
            prefill_workers: prefill_count,
            decode_workers: decode_count,
            http_workers: http_count,
            grpc_workers: grpc_count,
            circuit_breaker_open: cb_open_count,
            circuit_breaker_half_open: cb_half_open_count,
        }
    }

    /// 高效地获取 regular 与 PD worker 的数量(O(1))。
    /// 避免 get_all() 分配内存并遍历所有 worker 的开销。
    pub fn get_worker_distribution(&self) -> (usize, usize) {
        // 复用已有的 type_workers 索引做 O(1) 查询
        let regular_count = self
            .type_workers
            .get(&WorkerType::Regular)
            .map(|v| v.len())
            .unwrap_or(0);

        // 从 DashMap 高效获取 worker 总数
        let total_workers = self.workers.len();

        // 非 Regular 的即为 PD worker
        let pd_count = total_workers.saturating_sub(regular_count);

        (regular_count, pd_count)
    }

    /// 为注册表中所有 worker 启动一个健康检查器。
    /// 应在注册表填充完 worker 后调用一次。
    pub(crate) fn start_health_checker(&self, check_interval_secs: u64) -> HealthChecker {
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        };

        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();
        let workers_ref = self.workers.clone();

        let handle = tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_secs(check_interval_secs));

            loop {
                interval.tick().await;

                // 检查关闭信号
                if shutdown_clone.load(Ordering::Acquire) {
                    tracing::debug!("Registry health checker shutting down");
                    break;
                }

                // 从注册表取出全部 worker
                let workers: Vec<Arc<dyn Worker>> = workers_ref
                    .iter()
                    .map(|entry| entry.value().clone())
                    .collect();

                // 并行执行健康检查以提升性能
                // 在 worker 数量很多时尤为重要
                let health_futures: Vec<_> = workers
                    .iter()
                    .filter(|worker| !worker.metadata().health_config.disable_health_check)
                    .map(|worker| {
                        let worker = worker.clone();
                        async move {
                            let _ = worker.check_health_async().await;
                        }
                    })
                    .collect();
                futures::future::join_all(health_futures).await;
            }
        });

        HealthChecker::new(handle, shutdown)
    }
}

impl Default for WorkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// worker 注册表的统计信息。
#[derive(Debug, Clone)]
pub struct WorkerRegistryStats {
    /// 已注册 worker 总数
    pub total_workers: usize,
    /// 服务的不重复模型数量
    pub total_models: usize,
    /// 通过健康检查的 worker 数
    pub healthy_workers: usize,
    /// 未通过健康检查的 worker 数
    pub unhealthy_workers: usize,
    /// 所有 worker 当前负载之和
    pub total_load: usize,
    /// regular(非 PD)worker 数
    pub regular_workers: usize,
    /// prefill worker 数(PD 模式)
    pub prefill_workers: usize,
    /// decode worker 数(PD 模式)
    pub decode_workers: usize,
    /// 使用 HTTP 连接的 worker 数
    pub http_workers: usize,
    /// 使用 gRPC 连接的 worker 数
    pub grpc_workers: usize,
    /// 熔断器处于 Open 状态(拒绝请求)的 worker 数
    pub circuit_breaker_open: usize,
    /// 熔断器处于 HalfOpen 状态(试探恢复)的 worker 数
    pub circuit_breaker_half_open: usize,
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::core::{circuit_breaker::CircuitBreakerConfig, BasicWorkerBuilder};

    #[test]
    fn test_worker_registry() {
        let registry = WorkerRegistry::new();

        // Create a worker with labels
        let mut labels = HashMap::new();
        labels.insert("model_id".to_string(), "llama-3-8b".to_string());
        labels.insert("priority".to_string(), "50".to_string());
        labels.insert("cost".to_string(), "0.8".to_string());

        let worker: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker1:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        // Register worker
        let worker_id = registry.register(Arc::from(worker));

        assert!(registry.get(&worker_id).is_some());
        assert!(registry.get_by_url("http://worker1:8080").is_some());
        assert_eq!(registry.get_by_model("llama-3-8b").len(), 1);
        assert_eq!(registry.get_by_type(&WorkerType::Regular).len(), 1);
        assert_eq!(registry.get_by_connection(&ConnectionMode::Http).len(), 1);

        let stats = registry.stats();
        assert_eq!(stats.total_workers, 1);
        assert_eq!(stats.total_models, 1);

        // Remove worker
        registry.remove(&worker_id);
        assert!(registry.get(&worker_id).is_none());
    }

    #[test]
    fn test_model_index_fast_lookup() {
        let registry = WorkerRegistry::new();

        // Create workers for different models
        let mut labels1 = HashMap::new();
        labels1.insert("model_id".to_string(), "llama-3".to_string());
        let worker1: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker1:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels1)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        let mut labels2 = HashMap::new();
        labels2.insert("model_id".to_string(), "llama-3".to_string());
        let worker2: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker2:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels2)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        let mut labels3 = HashMap::new();
        labels3.insert("model_id".to_string(), "gpt-4".to_string());
        let worker3: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker3:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels3)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        // Register workers
        registry.register(Arc::from(worker1));
        registry.register(Arc::from(worker2));
        registry.register(Arc::from(worker3));

        let llama_workers = registry.get_by_model("llama-3");
        assert_eq!(llama_workers.len(), 2);
        let urls: Vec<String> = llama_workers.iter().map(|w| w.url().to_string()).collect();
        assert!(urls.contains(&"http://worker1:8080".to_string()));
        assert!(urls.contains(&"http://worker2:8080".to_string()));

        let gpt_workers = registry.get_by_model("gpt-4");
        assert_eq!(gpt_workers.len(), 1);
        assert_eq!(gpt_workers[0].url(), "http://worker3:8080");

        let unknown_workers = registry.get_by_model("unknown-model");
        assert_eq!(unknown_workers.len(), 0);

        registry.remove_by_url("http://worker1:8080");
        let llama_workers_after = registry.get_by_model("llama-3");
        assert_eq!(llama_workers_after.len(), 1);
        assert_eq!(llama_workers_after[0].url(), "http://worker2:8080");
    }
}
