use std::{
    fmt,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, LazyLock, RwLock as StdRwLock,
    },
    time::Duration,
};

use async_trait::async_trait;
use axum::body::Body;
use serde::{Deserialize, Serialize};
use tokio::{sync::OnceCell, time};

use super::{
    model_card::{ModelCard, ProviderType},
    model_type::{Endpoint, ModelType},
    CircuitBreaker, WorkerError, WorkerResult, UNKNOWN_MODEL_ID,
};
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    protocols::worker_spec::WorkerInfo,
    routers::grpc::client::GrpcClient,
};

/// worker 默认优先级（0-100 区间的中间值）
pub const DEFAULT_WORKER_PRIORITY: u32 = 50;

/// worker 默认成本因子（基准成本）
pub const DEFAULT_WORKER_COST: f32 = 1.0;

/// worker 请求的默认 HTTP 客户端超时（单位：秒）
pub const DEFAULT_WORKER_HTTP_TIMEOUT_SECS: u64 = 30;

static WORKER_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(DEFAULT_WORKER_HTTP_TIMEOUT_SECS))
        .build()
        .expect("Failed to create worker HTTP client")
});

/// 按「路由键（routing key）」维度统计的 worker 活跃负载。
///
/// 用于会话亲和等场景：以 routing_key 为粒度记录当前有多少个在途请求，
/// 便于观测某个 worker 上活跃的路由键数量。内部用 `DashMap` 支持并发读写。
pub struct WorkerRoutingKeyLoad {
    /// 所属 worker 的 URL（仅用于指标标签与日志）
    url: String,
    /// 各路由键当前的活跃请求计数；计数归零时会移除对应条目
    active_routing_keys: dashmap::DashMap<String, usize>,
}

impl WorkerRoutingKeyLoad {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            active_routing_keys: dashmap::DashMap::new(),
        }
    }

    /// 返回当前处于活跃状态的路由键数量（而非请求总数）。
    pub fn value(&self) -> usize {
        self.active_routing_keys.len()
    }

    /// 递增指定路由键的活跃计数（键不存在时从 0 开始）。
    pub fn increment(&self, routing_key: &str) {
        *self
            .active_routing_keys
            .entry(routing_key.to_string())
            .or_insert(0) += 1;
        self.update_metrics();
    }

    /// 递减指定路由键的活跃计数；归零时移除该键。
    ///
    /// 若计数已为 0 或键不存在（异常情况），仅记录 warn 日志而不做处理。
    pub fn decrement(&self, routing_key: &str) {
        use dashmap::mapref::entry::Entry;

        match self.active_routing_keys.entry(routing_key.to_string()) {
            Entry::Occupied(mut entry) => {
                let counter = entry.get_mut();
                if *counter > 0 {
                    *counter -= 1;
                    if *counter == 0 {
                        entry.remove();
                    }
                } else {
                    tracing::warn!(
                        worker_url = %self.url,
                        routing_key = %routing_key,
                        "Attempted to decrement routing key counter that is already at 0"
                    );
                }
            }
            Entry::Vacant(_) => {
                tracing::warn!(
                    worker_url = %self.url,
                    routing_key = %routing_key,
                    "Attempted to decrement non-existent routing key"
                );
            }
        }
        self.update_metrics();
    }

    fn update_metrics(&self) {
        Metrics::set_worker_routing_keys_active(&self.url, self.value());
    }
}

impl fmt::Debug for WorkerRoutingKeyLoad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkerRoutingKeyLoad")
            .field("url", &self.url)
            .field("active_routing_keys", &self.value())
            .finish()
    }
}

/// 表示一个后端服务的核心 worker 抽象。
///
/// 各种连接方式（HTTP / gRPC）与角色（Regular / Prefill / Decode）的 worker
/// 都实现该 trait，向路由层提供统一的能力：健康检查、负载与熔断、元数据访问等。
#[async_trait]
pub trait Worker: Send + Sync + fmt::Debug {
    /// 获取 worker 的 URL
    fn url(&self) -> &str;
    /// 获取 worker 的 API key
    fn api_key(&self) -> &Option<String>;
    /// 获取 worker 的类型（Regular、Prefill 或 Decode）。
    /// 返回引用以避免每次访问都克隆。
    fn worker_type(&self) -> &WorkerType;

    /// 获取 worker 的连接模式（HTTP 或 gRPC）。
    /// 返回引用以避免每次访问都克隆。
    fn connection_mode(&self) -> &ConnectionMode;

    /// 获取 PD 模式下的 bootstrap 主机名。
    /// 返回构造时从 URL 解析并缓存的主机名。
    fn bootstrap_host(&self) -> &str {
        &self.metadata().bootstrap_host
    }

    /// 获取 PD 模式下的 bootstrap 端口。
    /// 返回从 `WorkerType::Prefill` 缓存的端口。
    fn bootstrap_port(&self) -> Option<u16> {
        self.metadata().bootstrap_port
    }

    /// 检查 worker 当前是否健康
    fn is_healthy(&self) -> bool;

    /// 设置 worker 的健康状态
    fn set_healthy(&self, healthy: bool);

    /// 对 worker 执行异步健康检查
    async fn check_health_async(&self) -> WorkerResult<()>;

    /// 同步健康检查包装器（用于兼容）
    ///
    /// # 废弃提示
    /// 该方法每次调用都会创建一个新的 Tokio 运行时，开销很大。
    /// 请在异步上下文中优先使用 `check_health_async()`。
    ///
    /// # 性能警告
    /// 每次调用都创建运行时开销显著。仅在无法使用异步版本时才使用本方法。
    #[deprecated(
        since = "0.4.6",
        note = "Use check_health_async() instead. This method creates a new Tokio runtime per call."
    )]
    fn check_health(&self) -> WorkerResult<()> {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| WorkerError::HealthCheckFailed {
                url: self.url().to_string(),
                reason: format!("Failed to create runtime: {}", e),
            })?
            .block_on(self.check_health_async())
    }

    /// 获取当前负载（在途请求数）
    fn load(&self) -> usize;

    /// 递增负载计数器
    fn increment_load(&self);

    /// 递减负载计数器
    fn decrement_load(&self);

    /// 将负载计数器重置为 0（用于同步/恢复）
    fn reset_load(&self) {}

    /// 获取按路由键维度的负载跟踪器
    fn worker_routing_key_load(&self) -> &WorkerRoutingKeyLoad;

    /// 获取已处理请求数
    fn processed_requests(&self) -> usize;

    /// 递增已处理请求计数器
    fn increment_processed(&self);

    /// 获取 worker 专属元数据
    fn metadata(&self) -> &WorkerMetadata;

    /// 获取该 worker 的熔断器
    fn circuit_breaker(&self) -> &CircuitBreaker;

    /// 检查 worker 是否可用（健康 + 熔断器处于关闭/半开状态）
    fn is_available(&self) -> bool {
        self.is_healthy() && self.circuit_breaker().can_execute()
    }

    /// 记录一次向该 worker 发起请求的结果（成功/失败）
    fn record_outcome(&self, success: bool) {
        self.circuit_breaker().record_outcome(success);
    }

    /// 该 worker 是否支持数据并行（DP-aware）
    fn is_dp_aware(&self) -> bool {
        false
    }

    /// 获取不带任何 DP rank 后缀的基础 URL
    fn base_url(&self) -> &str {
        self.url()
    }

    /// 若为 DP-aware worker，获取其 DP rank
    fn dp_rank(&self) -> Option<usize> {
        None
    }

    /// 若该 worker 属于某个 DP 组，获取其 DP 大小
    fn dp_size(&self) -> Option<usize> {
        None
    }

    /// 为 DP-aware 路由变换请求
    async fn prepare_request(&self, req: serde_json::Value) -> WorkerResult<serde_json::Value> {
        Ok(req)
    }

    /// 获取请求实际使用的完整端点 URL
    fn endpoint_url(&self, route: &str) -> String {
        format!("{}{}", self.base_url(), route)
    }

    /// 检查该 worker 是否能处理某个具体请求
    fn can_handle(&self, _req: &serde_json::Value) -> bool {
        true
    }

    /// 获取该 worker 服务的模型 ID。
    /// 优先查 ModelCards，其次回退到 labels。
    fn model_id(&self) -> &str {
        // 优先查 ModelCards
        self.metadata()
            .models
            .first()
            .map(|m| m.id.as_str())
            .or_else(|| {
                // 回退到 labels
                self.metadata().labels.get("model_id").map(|s| s.as_str())
            })
            .unwrap_or(UNKNOWN_MODEL_ID)
    }

    /// 获取该 worker 的优先级（值越大优先级越高）
    fn priority(&self) -> u32 {
        self.metadata()
            .labels
            .get("priority")
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_WORKER_PRIORITY)
    }

    /// 获取该 worker 的成本因子（基准 = 1.0）
    fn cost(&self) -> f32 {
        self.metadata()
            .labels
            .get("cost")
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_WORKER_COST)
    }

    /// 获取指定模型的 tokenizer 路径。
    fn tokenizer_path(&self, model_id: &str) -> Option<&str> {
        self.metadata()
            .find_model(model_id)
            .and_then(|m| m.tokenizer_path.as_deref())
    }

    /// 获取指定模型的推理（reasoning）解析器。
    fn reasoning_parser(&self, model_id: &str) -> Option<&str> {
        self.metadata()
            .find_model(model_id)
            .and_then(|m| m.reasoning_parser.as_deref())
    }

    /// 获取指定模型的工具（tool）解析器。
    fn tool_parser(&self, model_id: &str) -> Option<&str> {
        self.metadata()
            .find_model(model_id)
            .and_then(|m| m.tool_parser.as_deref())
    }

    /// 获取指定模型的 chat 模板。
    fn chat_template(&self, model_id: &str) -> Option<&str> {
        self.metadata()
            .find_model(model_id)
            .and_then(|m| m.chat_template.as_deref())
    }

    /// 获取该 worker 的默认 provider 类型。
    /// `None` 表示原生/透传。
    fn default_provider(&self) -> Option<&ProviderType> {
        self.metadata().default_provider.as_ref()
    }

    /// 获取指定模型的 provider。
    /// 优先级：ModelCard.provider > worker.default_provider
    fn provider_for_model(&self, model_id: &str) -> Option<&ProviderType> {
        self.metadata().provider_for_model(model_id)
    }

    /// 检查模型是否为分类器（具有 id2label 映射）。
    fn is_classifier(&self, model_id: &str) -> bool {
        self.metadata()
            .find_model(model_id)
            .map(|m| m.is_classifier())
            .unwrap_or(false)
    }

    /// 获取分类模型的 id2label 映射。
    /// 若模型不是分类器或未找到，返回 None。
    fn id2label(&self, model_id: &str) -> Option<&std::collections::HashMap<u32, String>> {
        self.metadata()
            .find_model(model_id)
            .filter(|m| m.is_classifier())
            .map(|m| &m.id2label)
    }

    /// 获取模型的分类标签数量。
    fn num_labels(&self, model_id: &str) -> u32 {
        self.metadata()
            .find_model(model_id)
            .map(|m| m.num_labels)
            .unwrap_or(0)
    }

    /// 从分类模型中获取某个类别下标对应的标签。
    /// 若模型未找到或下标不在映射中，返回通用标签（LABEL_N）。
    fn get_label(&self, model_id: &str, class_idx: u32) -> String {
        self.metadata()
            .find_model(model_id)
            .map(|m| m.get_label(class_idx))
            .unwrap_or_else(|| format!("LABEL_{}", class_idx))
    }

    /// 检查该 worker 是否支持指定模型。
    /// 若 models 列表为空，则 worker 接受任意模型。
    fn supports_model(&self, model_id: &str) -> bool {
        self.metadata().supports_model(model_id)
    }

    /// 检查该 worker 是否为指定模型支持某个端点。
    /// 若模型未找到，则回退到 default_model_type。
    fn supports_endpoint(&self, model_id: &str, endpoint: Endpoint) -> bool {
        self.metadata().supports_endpoint(model_id, endpoint)
    }

    /// 获取该 worker 能服务的所有模型。
    fn models(&self) -> &[ModelCard] {
        &self.metadata().models
    }

    /// 为该 worker 设置模型列表（用于延迟发现）。
    /// 默认实现不做任何事——只有 BasicWorker 支持此操作。
    fn set_models(&self, _models: Vec<ModelCard>) {
        // 默认：空实现。BasicWorker 会重写该方法。
    }

    /// 检查该 worker 是否已完成模型发现。
    /// 若通过 set_models() 设置过模型，或元数据中已有模型，则返回 true。
    fn has_models_discovered(&self) -> bool {
        !self.metadata().models.is_empty()
    }

    /// 为该 worker 获取或创建一个 gRPC 客户端。
    /// HTTP worker 返回 None，gRPC worker 返回 Some(client)。
    async fn get_grpc_client(&self) -> WorkerResult<Option<Arc<GrpcClient>>>;

    /// 重置 gRPC 客户端连接（用于重连场景）。
    /// HTTP worker 为空操作。
    async fn reset_grpc_client(&self) -> WorkerResult<()> {
        Ok(())
    }
    async fn grpc_health_check(&self) -> WorkerResult<bool>;
    async fn http_health_check(&self) -> WorkerResult<bool>;
}

/// worker 通信的连接模式
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ConnectionMode {
    /// HTTP/REST 连接
    #[default]
    Http,
    /// gRPC 连接
    Grpc {
        /// gRPC 端点的可选端口（若与 URL 中的不同）
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(default)]
        port: Option<u16>,
    },
}

impl ConnectionMode {
    /// 检查本连接模式是否与另一个匹配，并对 gRPC 做特殊处理。
    /// 当把 `Grpc { port: None }` 作为通配符时，可匹配任意端口的 gRPC 连接。
    pub fn matches(&self, filter: &ConnectionMode) -> bool {
        match (self, filter) {
            (ConnectionMode::Http, ConnectionMode::Http) => true,
            (ConnectionMode::Grpc { .. }, ConnectionMode::Grpc { port: None }) => true,
            (ConnectionMode::Grpc { port: p1 }, ConnectionMode::Grpc { port: p2 }) => p1 == p2,
            _ => false,
        }
    }

    /// 获取该连接模式对应的指标标签
    pub fn as_metric_label(&self) -> &'static str {
        match self {
            ConnectionMode::Http => metrics_labels::CONNECTION_HTTP,
            ConnectionMode::Grpc { .. } => metrics_labels::CONNECTION_GRPC,
        }
    }
}

impl fmt::Display for ConnectionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConnectionMode::Http => write!(f, "HTTP"),
            ConnectionMode::Grpc { port } => match port {
                Some(p) => write!(f, "gRPC(port:{})", p),
                None => write!(f, "gRPC"),
            },
        }
    }
}

/// worker 的运行时实现类型
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum RuntimeType {
    /// SGLang 运行时（默认）
    #[default]
    Sglang,
    /// vLLM 运行时
    Vllm,
    /// 外部 OpenAI 兼容 API（非本地推理）。
    /// 用于路由到 OpenAI、Azure OpenAI、xAI 等外部提供商。
    External,
}

impl fmt::Display for RuntimeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeType::Sglang => write!(f, "sglang"),
            RuntimeType::Vllm => write!(f, "vllm"),
            RuntimeType::External => write!(f, "external"),
        }
    }
}

impl std::str::FromStr for RuntimeType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // 使用 eq_ignore_ascii_case 避免 to_lowercase() 的分配
        if s.eq_ignore_ascii_case("sglang") {
            Ok(RuntimeType::Sglang)
        } else if s.eq_ignore_ascii_case("vllm") {
            Ok(RuntimeType::Vllm)
        } else if s.eq_ignore_ascii_case("external") {
            Ok(RuntimeType::External)
        } else {
            Err(format!("Unknown runtime type: {}", s))
        }
    }
}

/// Worker type classification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WorkerType {
    /// 用于标准路由的普通 worker
    Regular,
    /// PD 分离模式下的 Prefill worker
    Prefill {
        /// 与 decode worker 通信所用的 bootstrap 端口
        bootstrap_port: Option<u16>,
    },
    /// PD 分离模式下的 Decode worker
    Decode,
}

impl fmt::Display for WorkerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorkerType::Regular => write!(f, "Regular"),
            WorkerType::Prefill { bootstrap_port } => match bootstrap_port {
                Some(port) => write!(f, "Prefill(bootstrap:{})", port),
                None => write!(f, "Prefill"),
            },
            WorkerType::Decode => write!(f, "Decode"),
        }
    }
}

impl WorkerType {
    /// 获取该 worker 类型对应的指标标签
    pub fn as_metric_label(&self) -> &'static str {
        match self {
            WorkerType::Regular => metrics_labels::WORKER_REGULAR,
            WorkerType::Prefill { .. } => metrics_labels::WORKER_PREFILL,
            WorkerType::Decode => metrics_labels::WORKER_DECODE,
        }
    }
}

/// 健康检查配置
#[derive(Debug, Clone)]
pub struct HealthConfig {
    /// 健康检查超时（单位：秒）
    pub timeout_secs: u64,
    /// 两次健康检查之间的间隔（单位：秒）
    pub check_interval_secs: u64,
    /// 健康检查的端点路径
    pub endpoint: String,
    /// 标记为不健康前需连续失败的次数
    pub failure_threshold: u32,
    /// 标记为健康前需连续成功的次数
    pub success_threshold: u32,
    /// 是否对该 worker 禁用健康检查
    pub disable_health_check: bool,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            timeout_secs: 5,
            check_interval_secs: 30,
            endpoint: "/health".to_string(),
            failure_threshold: 3,
            success_threshold: 2,
            disable_health_check: false,
        }
    }
}

/// 与 worker 关联的元数据
#[derive(Debug, Clone)]
pub struct WorkerMetadata {
    /// worker URL
    pub url: String,
    /// worker 类型
    pub worker_type: WorkerType,
    /// 连接模式
    pub connection_mode: ConnectionMode,
    /// 运行时类型（针对 gRPC worker）
    pub runtime_type: RuntimeType,
    /// 附加的 label / 标签
    pub labels: std::collections::HashMap<String, String>,
    /// 健康检查配置
    pub health_config: HealthConfig,
    /// API key
    pub api_key: Option<String>,
    /// 缓存的 bootstrap 主机名（构造时从 URL 解析）
    pub bootstrap_host: String,
    /// 缓存的 bootstrap 端口（来自 WorkerType::Prefill）
    pub bootstrap_port: Option<u16>,
    /// 该 worker 能服务的模型。
    /// 若为空，则 worker 接受任意模型（向后兼容行为）。
    pub models: Vec<ModelCard>,
    /// 该 worker 的默认 provider（当模型未指定时使用）。
    /// `None` 表示原生/透传。
    pub default_provider: Option<ProviderType>,
    /// 未知模型的默认模型类型（默认为 LLM 能力）。
    pub default_model_type: ModelType,
}

impl WorkerMetadata {
    /// 按 ID 查找模型卡（包含别名）
    pub fn find_model(&self, model_id: &str) -> Option<&ModelCard> {
        self.models.iter().find(|m| m.matches(model_id))
    }

    /// 检查该 worker 是否能服务指定模型。
    /// 若 models 列表为空，则 worker 接受任意模型（向后兼容）。
    pub fn supports_model(&self, model_id: &str) -> bool {
        self.models.is_empty() || self.find_model(model_id).is_some()
    }

    /// 检查该 worker 是否为指定模型支持某个端点。
    /// 若模型未找到，则回退到 default_model_type。
    pub fn supports_endpoint(&self, model_id: &str, endpoint: Endpoint) -> bool {
        if let Some(model) = self.find_model(model_id) {
            model.supports_endpoint(endpoint)
        } else {
            self.default_model_type.supports_endpoint(endpoint)
        }
    }

    /// 获取指定模型的 provider。
    /// 若找到模型则返回其 provider，否则返回 worker 的默认 provider。
    pub fn provider_for_model(&self, model_id: &str) -> Option<&ProviderType> {
        self.find_model(model_id)
            .and_then(|m| m.provider.as_ref())
            .or(self.default_provider.as_ref())
    }

    /// 获取该 worker 能服务的所有模型 ID
    pub fn model_ids(&self) -> impl Iterator<Item = &str> {
        self.models.iter().map(|m| m.id.as_str())
    }
}

/// 基础 worker 实现
#[derive(Clone)]
pub struct BasicWorker {
    pub metadata: WorkerMetadata,
    pub load_counter: Arc<AtomicUsize>,
    pub worker_routing_key_load: Arc<WorkerRoutingKeyLoad>,
    pub processed_counter: Arc<AtomicUsize>,
    pub healthy: Arc<AtomicBool>,
    pub consecutive_failures: Arc<AtomicUsize>,
    pub consecutive_successes: Arc<AtomicUsize>,
    pub circuit_breaker: CircuitBreaker,
    /// gRPC worker 的延迟初始化 gRPC 客户端。
    /// 使用 OnceCell，初始化后可无锁读取。
    pub grpc_client: Arc<OnceCell<Arc<GrpcClient>>>,
    /// 运行时可变的模型覆盖（用于延迟发现）。
    /// 一旦设置，将在路由决策中覆盖 metadata.models。
    /// 使用 std::sync::RwLock 以便在 supports_model() 中同步访问。
    pub models_override: Arc<StdRwLock<Option<Vec<ModelCard>>>>,
}

impl fmt::Debug for BasicWorker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BasicWorker")
            .field("metadata", &self.metadata)
            .field("healthy", &self.healthy.load(Ordering::Relaxed))
            .field("circuit_breaker", &self.circuit_breaker)
            .field("grpc_client", &"<RwLock>")
            .finish()
    }
}

impl BasicWorker {
    pub fn normalised_url(&self) -> WorkerResult<&str> {
        // 直接用 rfind——无需额外的 contains() 检查；
        // 若未找到 '@'，rfind 会直接返回 None。
        // 例如："http://[::1]:8080@0" -> "http://[::1]:8080" 与 "0"
        if let Some(at_pos) = self.url().rfind('@') {
            let base_url = &self.url()[..at_pos];
            let rank_str = &self.url()[at_pos + 1..];

            // 校验 rank 部分确实是一个数字
            if rank_str.parse::<usize>().is_ok() {
                Ok(base_url)
            } else {
                // 这个 '@' 并非 DP rank 分隔符，返回完整 URL
                Ok(self.url())
            }
        } else {
            Ok(self.url())
        }
    }

    fn update_running_requests_metrics(&self) {
        let load = self.load();
        Metrics::set_worker_requests_active(self.url(), load);
    }
}

#[async_trait]
impl Worker for BasicWorker {
    fn url(&self) -> &str {
        &self.metadata.url
    }

    fn api_key(&self) -> &Option<String> {
        &self.metadata.api_key
    }

    fn worker_type(&self) -> &WorkerType {
        &self.metadata.worker_type
    }

    fn connection_mode(&self) -> &ConnectionMode {
        &self.metadata.connection_mode
    }

    fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
    }

    fn set_healthy(&self, healthy: bool) {
        self.healthy.store(healthy, Ordering::Release);
        Metrics::set_worker_health(self.url(), healthy);
    }

    async fn check_health_async(&self) -> WorkerResult<()> {
        if self.metadata.health_config.disable_health_check {
            if !self.is_healthy() {
                self.set_healthy(true);
            }
            return Ok(());
        }

        let health_result = match &self.metadata.connection_mode {
            ConnectionMode::Http => self.http_health_check().await?,
            ConnectionMode::Grpc { .. } => self.grpc_health_check().await?,
        };

        // 获取用于指标的 worker 类型标签
        let worker_type_str = self.metadata.worker_type.as_metric_label();

        if health_result {
            self.consecutive_failures.store(0, Ordering::Release);
            let successes = self.consecutive_successes.fetch_add(1, Ordering::AcqRel) + 1;

            // 记录健康检查成功指标
            Metrics::record_worker_health_check(worker_type_str, metrics_labels::CB_SUCCESS);

            if !self.is_healthy()
                && successes >= self.metadata.health_config.success_threshold as usize
            {
                self.set_healthy(true);
                self.consecutive_successes.store(0, Ordering::Release);
            }
            Ok(())
        } else {
            self.consecutive_successes.store(0, Ordering::Release);
            let failures = self.consecutive_failures.fetch_add(1, Ordering::AcqRel) + 1;

            // 记录健康检查失败指标
            Metrics::record_worker_health_check(worker_type_str, metrics_labels::CB_FAILURE);

            if self.is_healthy()
                && failures >= self.metadata.health_config.failure_threshold as usize
            {
                self.set_healthy(false);
                self.consecutive_failures.store(0, Ordering::Release);
            }

            Err(WorkerError::HealthCheckFailed {
                url: self.metadata.url.clone(),
                reason: format!("Health check failed (consecutive failures: {})", failures),
            })
        }
    }

    fn load(&self) -> usize {
        self.load_counter.load(Ordering::Relaxed)
    }

    fn increment_load(&self) {
        self.load_counter.fetch_add(1, Ordering::Relaxed);
        self.update_running_requests_metrics();
    }

    fn decrement_load(&self) {
        if self
            .load_counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_sub(1)
            })
            .is_err()
        {
            tracing::warn!(
                worker_url = %self.metadata.url,
                "Attempted to decrement load counter that is already at 0"
            );
        }
        self.update_running_requests_metrics();
    }

    fn reset_load(&self) {
        self.load_counter.store(0, Ordering::Relaxed);
        self.update_running_requests_metrics();
    }

    fn worker_routing_key_load(&self) -> &WorkerRoutingKeyLoad {
        &self.worker_routing_key_load
    }

    fn processed_requests(&self) -> usize {
        self.processed_counter.load(Ordering::Relaxed)
    }

    fn increment_processed(&self) {
        self.processed_counter.fetch_add(1, Ordering::Relaxed);
    }

    fn metadata(&self) -> &WorkerMetadata {
        &self.metadata
    }

    fn circuit_breaker(&self) -> &CircuitBreaker {
        &self.circuit_breaker
    }

    fn supports_model(&self, model_id: &str) -> bool {
        // 优先检查 models_override（用于延迟发现）
        if let Ok(guard) = self.models_override.read() {
            if let Some(ref models) = *guard {
                // 已发现模型——检查是否支持该模型
                return models.iter().any(|m| m.matches(model_id));
            }
        }
        // 回退到 metadata.models（为空 = 通配符 = 在发现前不支持任何模型）
        self.metadata.supports_model(model_id)
    }

    fn set_models(&self, models: Vec<ModelCard>) {
        if let Ok(mut guard) = self.models_override.write() {
            tracing::debug!(
                "Setting {} models for worker {} via lazy discovery",
                models.len(),
                self.metadata.url
            );
            *guard = Some(models);
        }
    }

    fn has_models_discovered(&self) -> bool {
        // 检查 models_override 是否已被设置
        if let Ok(guard) = self.models_override.read() {
            if guard.is_some() {
                return true;
            }
        }
        // 回退到检查 metadata.models
        !self.metadata.models.is_empty()
    }

    async fn get_grpc_client(&self) -> WorkerResult<Option<Arc<GrpcClient>>> {
        match self.metadata.connection_mode {
            ConnectionMode::Http => Ok(None),
            ConnectionMode::Grpc { .. } => {
                // OnceCell 在初始化后提供无锁读取。
                // get_or_try_init 仅在首次调用时获取内部锁。
                let client = self
                    .grpc_client
                    .get_or_try_init(|| async {
                        let runtime_str = self.metadata.runtime_type.to_string();
                        tracing::info!(
                            "Lazily initializing gRPC client ({}) for worker: {}",
                            runtime_str,
                            self.metadata.url
                        );
                        match GrpcClient::connect(&self.metadata.url, &runtime_str).await {
                            Ok(client) => {
                                tracing::info!(
                                    "Successfully connected gRPC client ({}) for worker: {}",
                                    runtime_str,
                                    self.metadata.url
                                );
                                Ok(Arc::new(client))
                            }
                            Err(e) => {
                                tracing::error!(
                                    "Failed to connect gRPC client for worker {}: {}",
                                    self.metadata.url,
                                    e
                                );
                                Err(WorkerError::ConnectionFailed {
                                    url: self.metadata.url.clone(),
                                    reason: format!("Failed to connect to gRPC server: {}", e),
                                })
                            }
                        }
                    })
                    .await?;
                Ok(Some(Arc::clone(client)))
            }
        }
    }

    async fn reset_grpc_client(&self) -> WorkerResult<()> {
        // OnceCell 不支持重置。这是为了无锁性能而有意为之的设计。
        // 若连接失败，应将该 worker 移除后重新添加。
        tracing::debug!(
            "reset_grpc_client called for {} (no-op with OnceCell)",
            self.metadata.url
        );
        Ok(())
    }

    async fn grpc_health_check(&self) -> WorkerResult<bool> {
        let timeout = Duration::from_secs(self.metadata.health_config.timeout_secs);
        let maybe = self.get_grpc_client().await?;
        let Some(grpc_client) = maybe else {
            tracing::error!(
                "Worker {} is not a gRPC worker but connection mode is gRPC",
                self.metadata.url
            );
            return Ok(false);
        };

        match time::timeout(timeout, grpc_client.health_check()).await {
            Ok(Ok(resp)) => {
                tracing::debug!(
                    "gRPC health OK for {}: healthy={}",
                    self.metadata.url,
                    resp.healthy
                );
                Ok(resp.healthy)
            }
            Ok(Err(err)) => {
                tracing::warn!("gRPC health RPC error for {}: {err:?}", self.metadata.url);
                Ok(false)
            }
            Err(_) => {
                tracing::warn!("gRPC health timed out for {}", self.metadata.url);
                Ok(false)
            }
        }
    }

    async fn http_health_check(&self) -> WorkerResult<bool> {
        let timeout = Duration::from_secs(self.metadata.health_config.timeout_secs);

        let url = self.normalised_url()?;
        let health_url = format!("{}{}", url, self.metadata.health_config.endpoint);

        let mut req = WORKER_CLIENT.get(&health_url).timeout(timeout);
        if let Some(api_key) = &self.metadata.api_key {
            req = req.bearer_auth(api_key);
        }

        match req.send().await {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    Ok(true)
                } else {
                    tracing::warn!(
                        "HTTP health check returned non-success status for {}: {}",
                        health_url,
                        status
                    );
                    Ok(false)
                }
            }
            Err(err) => {
                tracing::warn!("HTTP health check failed for {}: {err:?}", health_url);
                Ok(false)
            }
        }
    }
}

/// 处理数据并行路由的 DP-aware worker
#[derive(Debug, Clone)]
pub struct DPAwareWorker {
    /// 底层的基础 worker
    base_worker: BasicWorker,
    /// 该 worker 的 DP rank
    dp_rank: usize,
    /// DP 总大小
    dp_size: usize,
    /// 不带 DP 后缀的基础 URL
    base_url: String,
}

impl DPAwareWorker {
    /// 基于一个预配置的基础 worker 创建新的 DP-aware worker。
    /// 主要由建造者（builder）模式使用。
    pub fn with_base_worker(
        base_worker: BasicWorker,
        base_url: String,
        dp_rank: usize,
        dp_size: usize,
    ) -> Self {
        Self {
            base_worker,
            dp_rank,
            dp_size,
            base_url,
        }
    }
}

#[async_trait]
impl Worker for DPAwareWorker {
    fn url(&self) -> &str {
        self.base_worker.url()
    }

    fn api_key(&self) -> &Option<String> {
        self.base_worker.api_key()
    }

    fn worker_type(&self) -> &WorkerType {
        self.base_worker.worker_type()
    }

    fn connection_mode(&self) -> &ConnectionMode {
        self.base_worker.connection_mode()
    }

    fn is_healthy(&self) -> bool {
        self.base_worker.is_healthy()
    }

    fn set_healthy(&self, healthy: bool) {
        self.base_worker.set_healthy(healthy);
    }

    async fn check_health_async(&self) -> WorkerResult<()> {
        self.base_worker.check_health_async().await
    }

    fn load(&self) -> usize {
        self.base_worker.load()
    }

    fn increment_load(&self) {
        self.base_worker.increment_load();
    }

    fn decrement_load(&self) {
        self.base_worker.decrement_load();
    }

    fn reset_load(&self) {
        self.base_worker.reset_load();
    }

    fn worker_routing_key_load(&self) -> &WorkerRoutingKeyLoad {
        self.base_worker.worker_routing_key_load()
    }

    fn processed_requests(&self) -> usize {
        self.base_worker.processed_requests()
    }

    fn increment_processed(&self) {
        self.base_worker.increment_processed();
    }

    fn metadata(&self) -> &WorkerMetadata {
        self.base_worker.metadata()
    }

    fn circuit_breaker(&self) -> &CircuitBreaker {
        self.base_worker.circuit_breaker()
    }

    fn is_dp_aware(&self) -> bool {
        true
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn dp_rank(&self) -> Option<usize> {
        Some(self.dp_rank)
    }

    fn dp_size(&self) -> Option<usize> {
        Some(self.dp_size)
    }

    async fn prepare_request(&self, mut req: serde_json::Value) -> WorkerResult<serde_json::Value> {
        if let Some(map) = req.as_object_mut() {
            map.insert(
                "data_parallel_rank".to_string(),
                serde_json::json!(self.dp_rank),
            );
            Ok(req)
        } else {
            Err(WorkerError::InvalidConfiguration {
                message: "Request must be a JSON object for DP-aware routing".to_string(),
            })
        }
    }

    fn endpoint_url(&self, route: &str) -> String {
        format!("{}{}", self.base_url, route)
    }

    async fn get_grpc_client(&self) -> WorkerResult<Option<Arc<GrpcClient>>> {
        self.base_worker.get_grpc_client().await
    }

    async fn reset_grpc_client(&self) -> WorkerResult<()> {
        self.base_worker.reset_grpc_client().await
    }

    async fn grpc_health_check(&self) -> WorkerResult<bool> {
        self.base_worker.grpc_health_check().await
    }

    async fn http_health_check(&self) -> WorkerResult<bool> {
        self.base_worker.http_health_check().await
    }
}

/// 用于 worker 负载管理的 RAII 守卫。
///
/// 在 drop 时自动递减 worker 负载。可挂载到 axum 的 Response 上，
/// 将守卫的生命周期与响应体绑定——这对流式响应至关重要：
/// 函数会立即返回，但数据流仍在后台持续推送。
pub struct WorkerLoadGuard {
    worker: Arc<dyn Worker>,
    routing_key: Option<String>,
}

impl WorkerLoadGuard {
    pub fn new(worker: Arc<dyn Worker>, headers: Option<&http::HeaderMap>) -> Self {
        use crate::routers::header_utils::extract_routing_key;

        worker.increment_load();

        let routing_key = extract_routing_key(headers).map(String::from);

        if let Some(ref key) = routing_key {
            worker.worker_routing_key_load().increment(key);
        }

        Self {
            worker,
            routing_key,
        }
    }
}

impl Drop for WorkerLoadGuard {
    fn drop(&mut self) {
        self.worker.decrement_load();
        if let Some(ref key) = self.routing_key {
            self.worker.worker_routing_key_load().decrement(key);
        }
    }
}

/// 携带一个附加值的响应体包装器。
///
/// 当该响应体被 drop 时（流结束或客户端断开），附加值会被自动 drop。
/// 这对于需要与响应体生命周期绑定的 RAII 守卫（如 WorkerLoadGuard）很有用。
pub struct AttachedBody<T> {
    inner: Body,
    _attached: T,
}

impl<T> AttachedBody<T> {
    pub fn new(inner: Body, attached: T) -> Self {
        Self {
            inner,
            _attached: attached,
        }
    }
}

impl<T: Send + Unpin + 'static> AttachedBody<T> {
    pub fn wrap_response(
        response: axum::response::Response,
        attached: T,
    ) -> axum::response::Response {
        let (parts, body) = response.into_parts();
        axum::response::Response::from_parts(parts, Body::new(Self::new(body, attached)))
    }
}

impl<T: Send + Unpin + 'static> http_body::Body for AttachedBody<T> {
    type Data = bytes::Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        std::pin::Pin::new(&mut this.inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

/// 带优雅关闭能力的健康检查器句柄
pub(crate) struct HealthChecker {
    #[allow(dead_code)]
    handle: tokio::task::JoinHandle<()>,
    shutdown: Arc<AtomicBool>,
}

impl fmt::Debug for HealthChecker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HealthChecker")
            .field("shutdown", &self.shutdown.load(Ordering::Relaxed))
            .finish()
    }
}

impl HealthChecker {
    /// 创建一个新的 HealthChecker
    pub fn new(handle: tokio::task::JoinHandle<()>, shutdown: Arc<AtomicBool>) -> Self {
        Self { handle, shutdown }
    }

    /// 优雅地关闭健康检查器
    #[allow(dead_code)]
    pub async fn shutdown(self) {
        self.shutdown.store(true, Ordering::Release);
        let _ = self.handle.await;
    }
}

/// 辅助函数：将 Worker trait 对象转换为 WorkerInfo 结构体
pub fn worker_to_info(worker: &Arc<dyn Worker>) -> WorkerInfo {
    // 缓存多次使用的引用，避免重复的方法调用
    let worker_type = worker.worker_type();
    let connection_mode = worker.connection_mode();
    let url = worker.url();
    let model_id = worker.model_id();

    let worker_type_str = match worker_type {
        WorkerType::Regular => "regular",
        WorkerType::Prefill { .. } => "prefill",
        WorkerType::Decode => "decode",
    };

    let bootstrap_port = match worker_type {
        WorkerType::Prefill { bootstrap_port } => *bootstrap_port,
        _ => None,
    };

    let runtime_type = match connection_mode {
        ConnectionMode::Grpc { .. } => Some(worker.metadata().runtime_type.to_string()),
        ConnectionMode::Http => None,
    };

    WorkerInfo {
        id: url.to_string(),
        url: url.to_string(),
        model_id: model_id.to_string(),
        priority: worker.priority(),
        cost: worker.cost(),
        worker_type: worker_type_str.to_string(),
        is_healthy: worker.is_healthy(),
        load: worker.load(),
        connection_mode: connection_mode.to_string(),
        runtime_type,
        tokenizer_path: worker.tokenizer_path(model_id).map(String::from),
        reasoning_parser: worker.reasoning_parser(model_id).map(String::from),
        tool_parser: worker.tool_parser(model_id).map(String::from),
        chat_template: worker.chat_template(model_id).map(String::from),
        bootstrap_port,
        metadata: worker.metadata().labels.clone(),
        disable_health_check: worker.metadata().health_config.disable_health_check,
        job_status: None,
    }
}

#[cfg(test)]
mod tests {
    use std::{thread, time::Duration};

    use super::*;
    use crate::core::{
        circuit_breaker::{CircuitBreakerConfig, CircuitState},
        DPAwareWorkerBuilder,
    };

    #[test]
    fn test_worker_type_display() {
        assert_eq!(WorkerType::Regular.to_string(), "Regular");
        assert_eq!(
            WorkerType::Prefill {
                bootstrap_port: Some(8080)
            }
            .to_string(),
            "Prefill(bootstrap:8080)"
        );
        assert_eq!(
            WorkerType::Prefill {
                bootstrap_port: None
            }
            .to_string(),
            "Prefill"
        );
        assert_eq!(WorkerType::Decode.to_string(), "Decode");
    }

    #[test]
    fn test_worker_type_equality() {
        assert_eq!(WorkerType::Regular, WorkerType::Regular);
        assert_ne!(WorkerType::Regular, WorkerType::Decode);
        assert_eq!(
            WorkerType::Prefill {
                bootstrap_port: Some(8080)
            },
            WorkerType::Prefill {
                bootstrap_port: Some(8080)
            }
        );
        assert_ne!(
            WorkerType::Prefill {
                bootstrap_port: Some(8080)
            },
            WorkerType::Prefill {
                bootstrap_port: Some(8081)
            }
        );
    }

    #[test]
    fn test_worker_type_clone() {
        let original = WorkerType::Prefill {
            bootstrap_port: Some(8080),
        };
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_health_config_default() {
        let config = HealthConfig::default();
        assert_eq!(config.timeout_secs, 5);
        assert_eq!(config.check_interval_secs, 30);
        assert_eq!(config.endpoint, "/health");
        assert_eq!(config.failure_threshold, 3);
        assert_eq!(config.success_threshold, 2);
        assert!(!config.disable_health_check);
    }

    #[test]
    fn test_health_config_custom() {
        let config = HealthConfig {
            timeout_secs: 10,
            check_interval_secs: 60,
            endpoint: "/healthz".to_string(),
            failure_threshold: 5,
            success_threshold: 3,
            disable_health_check: true,
        };
        assert_eq!(config.timeout_secs, 10);
        assert_eq!(config.check_interval_secs, 60);
        assert_eq!(config.endpoint, "/healthz");
        assert_eq!(config.failure_threshold, 5);
        assert_eq!(config.success_threshold, 3);
        assert!(config.disable_health_check);
    }

    #[test]
    fn test_basic_worker_creation() {
        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();
        assert_eq!(worker.url(), "http://test:8080");
        assert_eq!(worker.worker_type(), &WorkerType::Regular);
        assert!(worker.is_healthy());
        assert_eq!(worker.load(), 0);
        assert_eq!(worker.processed_requests(), 0);
    }

    #[test]
    fn test_worker_with_labels() {
        let mut labels = std::collections::HashMap::new();
        labels.insert("env".to_string(), "prod".to_string());
        labels.insert("zone".to_string(), "us-west".to_string());

        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .labels(labels.clone())
            .build();

        assert_eq!(worker.metadata().labels, labels);
    }

    #[test]
    fn test_worker_with_health_config() {
        let custom_config = HealthConfig {
            timeout_secs: 15,
            check_interval_secs: 45,
            endpoint: "/custom-health".to_string(),
            failure_threshold: 4,
            success_threshold: 2,
            disable_health_check: false,
        };

        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .health_config(custom_config.clone())
            .build();

        assert_eq!(worker.metadata().health_config.timeout_secs, 15);
        assert_eq!(worker.metadata().health_config.check_interval_secs, 45);
        assert_eq!(worker.metadata().health_config.endpoint, "/custom-health");
    }

    #[test]
    fn test_worker_url() {
        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://worker1:8080")
            .worker_type(WorkerType::Regular)
            .build();
        assert_eq!(worker.url(), "http://worker1:8080");
    }

    #[test]
    fn test_worker_type_getter() {
        use crate::core::BasicWorkerBuilder;
        let regular = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();
        assert_eq!(regular.worker_type(), &WorkerType::Regular);

        let prefill = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Prefill {
                bootstrap_port: Some(9090),
            })
            .build();
        assert_eq!(
            prefill.worker_type(),
            &WorkerType::Prefill {
                bootstrap_port: Some(9090)
            }
        );

        let decode = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Decode)
            .build();
        assert_eq!(decode.worker_type(), &WorkerType::Decode);
    }

    #[test]
    fn test_health_status() {
        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();

        assert!(worker.is_healthy());

        worker.set_healthy(false);
        assert!(!worker.is_healthy());

        worker.set_healthy(true);
        assert!(worker.is_healthy());
    }

    #[test]
    fn test_load_counter_operations() {
        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();

        assert_eq!(worker.load(), 0);

        worker.increment_load();
        assert_eq!(worker.load(), 1);

        worker.increment_load();
        worker.increment_load();
        assert_eq!(worker.load(), 3);

        worker.decrement_load();
        assert_eq!(worker.load(), 2);

        worker.decrement_load();
        worker.decrement_load();
        assert_eq!(worker.load(), 0);

        worker.decrement_load();
        assert_eq!(worker.load(), 0);
    }

    #[test]
    fn test_processed_counter() {
        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();

        assert_eq!(worker.processed_requests(), 0);

        for i in 1..=100 {
            worker.increment_processed();
            assert_eq!(worker.processed_requests(), i);
        }
    }

    #[tokio::test]
    async fn test_concurrent_load_increments() {
        use crate::core::BasicWorkerBuilder;
        let worker = Arc::new(
            BasicWorkerBuilder::new("http://test:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        let mut handles = vec![];

        for _ in 0..100 {
            let worker_clone = Arc::clone(&worker);
            let handle = tokio::spawn(async move {
                worker_clone.increment_load();
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(worker.load(), 100);
    }

    #[tokio::test]
    async fn test_concurrent_load_decrements() {
        use crate::core::BasicWorkerBuilder;
        let worker = Arc::new(
            BasicWorkerBuilder::new("http://test:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        for _ in 0..100 {
            worker.increment_load();
        }
        assert_eq!(worker.load(), 100);

        let mut handles = vec![];

        for _ in 0..100 {
            let worker_clone = Arc::clone(&worker);
            let handle = tokio::spawn(async move {
                worker_clone.decrement_load();
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(worker.load(), 0);
    }

    #[tokio::test]
    async fn test_concurrent_health_updates() {
        use crate::core::BasicWorkerBuilder;
        let worker = Arc::new(
            BasicWorkerBuilder::new("http://test:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        let mut handles = vec![];

        for i in 0..100 {
            let worker_clone = Arc::clone(&worker);
            let handle = tokio::spawn(async move {
                worker_clone.set_healthy(i % 2 == 0);
                time::sleep(Duration::from_micros(10)).await;
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }
    }

    #[test]
    fn test_create_regular_worker() {
        use crate::core::BasicWorkerBuilder;
        let worker: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://regular:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );
        assert_eq!(worker.url(), "http://regular:8080");
        assert_eq!(worker.worker_type(), &WorkerType::Regular);
    }

    #[test]
    fn test_create_prefill_worker() {
        use crate::core::BasicWorkerBuilder;
        let worker1: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://prefill:8080")
                .worker_type(WorkerType::Prefill {
                    bootstrap_port: Some(9090),
                })
                .build(),
        );
        assert_eq!(worker1.url(), "http://prefill:8080");
        assert_eq!(
            worker1.worker_type(),
            &WorkerType::Prefill {
                bootstrap_port: Some(9090)
            }
        );

        let worker2: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://prefill:8080")
                .worker_type(WorkerType::Prefill {
                    bootstrap_port: None,
                })
                .build(),
        );
        assert_eq!(
            worker2.worker_type(),
            &WorkerType::Prefill {
                bootstrap_port: None
            }
        );
    }

    #[test]
    fn test_create_decode_worker() {
        use crate::core::BasicWorkerBuilder;
        let worker: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://decode:8080")
                .worker_type(WorkerType::Decode)
                .build(),
        );
        assert_eq!(worker.url(), "http://decode:8080");
        assert_eq!(worker.worker_type(), &WorkerType::Decode);
    }

    #[tokio::test]
    async fn test_check_health_async() {
        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();

        // Health check should fail since there's no actual server
        let result = worker.check_health_async().await;
        assert!(result.is_err());
    }

    #[test]
    fn test_load_counter_performance() {
        use std::time::Instant;

        use crate::core::BasicWorkerBuilder;

        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();
        let iterations = 1_000_000;

        let start = Instant::now();
        for _ in 0..iterations {
            worker.increment_load();
        }
        let duration = start.elapsed();

        let ops_per_sec = iterations as f64 / duration.as_secs_f64();
        println!("Load counter operations per second: {:.0}", ops_per_sec);

        assert!(ops_per_sec > 1_000_000.0);
    }

    #[test]
    fn test_dp_aware_worker_creation() {
        let dp_worker = DPAwareWorkerBuilder::new("http://worker1:8080", 2, 4)
            .worker_type(WorkerType::Regular)
            .build();

        assert_eq!(dp_worker.url(), "http://worker1:8080@2");
        assert_eq!(dp_worker.base_url(), "http://worker1:8080");
        assert!(dp_worker.is_dp_aware());
        assert_eq!(dp_worker.dp_rank(), Some(2));
        assert_eq!(dp_worker.dp_size(), Some(4));
        assert_eq!(dp_worker.worker_type(), &WorkerType::Regular);
    }

    #[test]
    fn test_dp_aware_worker_creation_prefill() {
        let dp_worker = DPAwareWorkerBuilder::new("http://worker1:8080", 1, 2)
            .worker_type(WorkerType::Prefill {
                bootstrap_port: Some(9090),
            })
            .build();

        assert_eq!(dp_worker.url(), "http://worker1:8080@1");
        assert!(dp_worker.is_dp_aware());
        assert_eq!(
            dp_worker.worker_type(),
            &WorkerType::Prefill {
                bootstrap_port: Some(9090)
            }
        );
    }

    #[test]
    fn test_dp_aware_worker_creation_decode() {
        let dp_worker = DPAwareWorkerBuilder::new("http://worker1:8080", 0, 4)
            .worker_type(WorkerType::Decode)
            .build();

        assert_eq!(dp_worker.url(), "http://worker1:8080@0");
        assert!(dp_worker.is_dp_aware());
        assert_eq!(dp_worker.worker_type(), &WorkerType::Decode);
    }

    #[tokio::test]
    async fn test_dp_aware_prepare_request() {
        let dp_worker = DPAwareWorkerBuilder::new("http://worker1:8080", 3, 8)
            .worker_type(WorkerType::Regular)
            .build();

        let original_req = serde_json::json!({
            "prompt": "Hello",
            "max_tokens": 100
        });

        let prepared_req = dp_worker.prepare_request(original_req).await.unwrap();

        assert_eq!(prepared_req["prompt"], "Hello");
        assert_eq!(prepared_req["max_tokens"], 100);
        assert_eq!(prepared_req["data_parallel_rank"], 3);
    }

    #[tokio::test]
    async fn test_dp_aware_prepare_request_invalid() {
        let dp_worker = DPAwareWorkerBuilder::new("http://worker1:8080", 0, 4)
            .worker_type(WorkerType::Regular)
            .build();

        // Non-object JSON should fail
        let invalid_req = serde_json::json!("not an object");
        let result = dp_worker.prepare_request(invalid_req).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            WorkerError::InvalidConfiguration { message } => {
                assert!(message.contains("JSON object"));
            }
            _ => panic!("Expected InvalidConfiguration error"),
        }
    }

    #[test]
    fn test_dp_aware_endpoint_url() {
        let dp_worker = DPAwareWorkerBuilder::new("http://worker1:8080", 1, 4)
            .worker_type(WorkerType::Regular)
            .build();

        assert_eq!(
            dp_worker.endpoint_url("/generate"),
            "http://worker1:8080/generate"
        );
        assert_eq!(
            dp_worker.endpoint_url("/health"),
            "http://worker1:8080/health"
        );
    }

    #[test]
    fn test_dp_aware_worker_delegated_methods() {
        let dp_worker = DPAwareWorkerBuilder::new("http://worker1:8080", 0, 2)
            .worker_type(WorkerType::Regular)
            .build();

        assert!(dp_worker.is_healthy());
        dp_worker.set_healthy(false);
        assert!(!dp_worker.is_healthy());

        assert_eq!(dp_worker.load(), 0);
        dp_worker.increment_load();
        assert_eq!(dp_worker.load(), 1);
        dp_worker.decrement_load();
        assert_eq!(dp_worker.load(), 0);

        assert_eq!(dp_worker.processed_requests(), 0);
        dp_worker.increment_processed();
        assert_eq!(dp_worker.processed_requests(), 1);
    }

    #[test]
    fn test_worker_circuit_breaker() {
        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();

        assert!(worker.is_available());
        assert_eq!(worker.circuit_breaker().state(), CircuitState::Closed);

        worker.record_outcome(false);
        worker.record_outcome(false);

        assert!(worker.is_available());

        worker.record_outcome(false);
        worker.record_outcome(false);
        worker.record_outcome(false);

        assert!(!worker.is_available());
        assert!(worker.is_healthy());
        assert!(!worker.circuit_breaker().can_execute());
    }

    #[test]
    fn test_worker_with_circuit_breaker_config() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 1,
            timeout_duration: Duration::from_millis(100),
            window_duration: Duration::from_secs(60),
        };

        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .circuit_breaker_config(config)
            .build();

        worker.record_outcome(false);
        assert!(worker.is_available());
        worker.record_outcome(false);
        assert!(!worker.is_available());

        thread::sleep(Duration::from_millis(150));

        assert!(worker.is_available());
        assert_eq!(worker.circuit_breaker().state(), CircuitState::HalfOpen);

        worker.record_outcome(true);
        assert_eq!(worker.circuit_breaker().state(), CircuitState::Closed);
    }

    #[test]
    fn test_dp_aware_worker_circuit_breaker() {
        let dp_worker = DPAwareWorkerBuilder::new("http://worker:8080", 0, 2)
            .worker_type(WorkerType::Regular)
            .build();

        assert!(dp_worker.is_available());

        for _ in 0..5 {
            dp_worker.record_outcome(false);
        }

        assert!(!dp_worker.is_available());
        assert_eq!(dp_worker.circuit_breaker().state(), CircuitState::Open);
    }

    #[tokio::test]
    async fn test_mixed_worker_types() {
        use crate::core::BasicWorkerBuilder;
        let regular: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://regular:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );
        let prefill: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://prefill:8080")
                .worker_type(WorkerType::Prefill {
                    bootstrap_port: Some(9090),
                })
                .build(),
        );
        let decode: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://decode:8080")
                .worker_type(WorkerType::Decode)
                .build(),
        );
        let dp_aware_regular: Box<dyn Worker> = Box::new(
            DPAwareWorkerBuilder::new("http://dp:8080", 0, 2)
                .worker_type(WorkerType::Regular)
                .api_key("test_api_key")
                .build(),
        );
        let dp_aware_prefill: Box<dyn Worker> = Box::new(
            DPAwareWorkerBuilder::new("http://dp-prefill:8080", 1, 2)
                .worker_type(WorkerType::Prefill {
                    bootstrap_port: None,
                })
                .api_key("test_api_key")
                .build(),
        );
        let dp_aware_decode: Box<dyn Worker> = Box::new(
            DPAwareWorkerBuilder::new("http://dp-decode:8080", 0, 4)
                .worker_type(WorkerType::Decode)
                .api_key("test_api_key")
                .build(),
        );

        let workers: Vec<Box<dyn Worker>> = vec![
            regular,
            prefill,
            decode,
            dp_aware_regular,
            dp_aware_prefill,
            dp_aware_decode,
        ];

        for worker in &workers {
            assert!(worker.is_healthy());
            assert_eq!(worker.load(), 0);
            assert_eq!(worker.processed_requests(), 0);
        }

        assert!(!workers[0].is_dp_aware());
        assert!(!workers[1].is_dp_aware());
        assert!(!workers[2].is_dp_aware());
        assert!(workers[3].is_dp_aware());
        assert!(workers[4].is_dp_aware());
        assert!(workers[5].is_dp_aware());

        assert_eq!(workers[0].worker_type(), &WorkerType::Regular);
        assert_eq!(
            workers[1].worker_type(),
            &WorkerType::Prefill {
                bootstrap_port: Some(9090)
            }
        );
        assert_eq!(workers[2].worker_type(), &WorkerType::Decode);
        assert_eq!(workers[3].worker_type(), &WorkerType::Regular);
        assert_eq!(
            workers[4].worker_type(),
            &WorkerType::Prefill {
                bootstrap_port: None
            }
        );
        assert_eq!(workers[5].worker_type(), &WorkerType::Decode);
    }

    // === Phase 1.3: WorkerMetadata model methods tests ===

    #[test]
    fn test_worker_metadata_empty_models_accepts_all() {
        let metadata = WorkerMetadata {
            url: "http://test:8080".to_string(),
            worker_type: WorkerType::Regular,
            connection_mode: ConnectionMode::Http,
            runtime_type: RuntimeType::default(),
            labels: std::collections::HashMap::new(),
            health_config: HealthConfig::default(),
            api_key: None,
            bootstrap_host: "test".to_string(),
            bootstrap_port: None,
            models: Vec::new(), // Empty = accepts any model
            default_provider: None,
            default_model_type: ModelType::LLM,
        };

        // Empty models list should accept any model
        assert!(metadata.supports_model("any-model"));
        assert!(metadata.supports_model("gpt-4"));
        assert!(metadata.supports_model("llama-3.1"));
    }

    #[test]
    fn test_worker_metadata_find_model() {
        use super::ModelCard;

        let model1 = ModelCard::new("meta-llama/Llama-3.1-8B")
            .with_alias("llama-3.1-8b")
            .with_alias("llama3.1");
        let model2 = ModelCard::new("gpt-4o");

        let metadata = WorkerMetadata {
            url: "http://test:8080".to_string(),
            worker_type: WorkerType::Regular,
            connection_mode: ConnectionMode::Http,
            runtime_type: RuntimeType::default(),
            labels: std::collections::HashMap::new(),
            health_config: HealthConfig::default(),
            api_key: None,
            bootstrap_host: "test".to_string(),
            bootstrap_port: None,
            models: vec![model1, model2],
            default_provider: None,
            default_model_type: ModelType::LLM,
        };

        // Find by primary ID
        assert!(metadata.find_model("meta-llama/Llama-3.1-8B").is_some());
        assert!(metadata.find_model("gpt-4o").is_some());

        // Find by alias
        assert!(metadata.find_model("llama-3.1-8b").is_some());
        assert!(metadata.find_model("llama3.1").is_some());

        // Not found
        assert!(metadata.find_model("unknown-model").is_none());
    }

    #[test]
    fn test_worker_routing_key_load_increment_decrement() {
        let load = WorkerRoutingKeyLoad::new("http://test:8000");
        assert_eq!(load.value(), 0);

        load.increment("key1");
        assert_eq!(load.value(), 1);

        load.increment("key2");
        assert_eq!(load.value(), 2);

        load.increment("key1");
        assert_eq!(load.value(), 2);

        load.decrement("key1");
        assert_eq!(load.value(), 2);

        load.decrement("key1");
        assert_eq!(load.value(), 1);

        load.decrement("key2");
        assert_eq!(load.value(), 0);
    }

    #[test]
    fn test_worker_routing_key_load_cleanup_on_zero() {
        let load = WorkerRoutingKeyLoad::new("http://test:8000");

        load.increment("key1");
        load.increment("key2");
        load.increment("key3");
        assert_eq!(load.active_routing_keys.len(), 3);

        load.decrement("key1");
        assert_eq!(load.active_routing_keys.len(), 2);

        load.decrement("key2");
        assert_eq!(load.active_routing_keys.len(), 1);

        load.decrement("key3");
        assert_eq!(load.active_routing_keys.len(), 0);
    }

    #[test]
    fn test_worker_routing_key_load_multiple_requests_same_key() {
        let load = WorkerRoutingKeyLoad::new("http://test:8000");

        load.increment("key-1");
        load.increment("key-1");
        load.increment("key-1");
        assert_eq!(load.value(), 1);

        load.decrement("key-1");
        assert_eq!(load.value(), 1);

        load.decrement("key-1");
        assert_eq!(load.value(), 1);

        load.decrement("key-1");
        assert_eq!(load.value(), 0);
        assert_eq!(load.active_routing_keys.len(), 0);
    }

    #[test]
    fn test_worker_routing_key_load_decrement_nonexistent() {
        let load = WorkerRoutingKeyLoad::new("http://test:8000");
        load.decrement("nonexistent");
        assert_eq!(load.value(), 0);
    }

    #[test]
    fn test_worker_load_guard_with_routing_key() {
        use crate::core::BasicWorkerBuilder;

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://test:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        assert_eq!(worker.load(), 0);
        assert_eq!(worker.worker_routing_key_load().value(), 0);

        let mut headers = http::HeaderMap::new();
        headers.insert("x-smg-routing-key", "key-123".parse().unwrap());

        {
            let _guard = WorkerLoadGuard::new(worker.clone(), Some(&headers));
            assert_eq!(worker.load(), 1);
            assert_eq!(worker.worker_routing_key_load().value(), 1);
        }

        assert_eq!(worker.load(), 0);
        assert_eq!(worker.worker_routing_key_load().value(), 0);
    }

    #[test]
    fn test_worker_load_guard_without_routing_key() {
        use crate::core::BasicWorkerBuilder;

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://test:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        assert_eq!(worker.load(), 0);
        assert_eq!(worker.worker_routing_key_load().value(), 0);

        {
            let _guard = WorkerLoadGuard::new(worker.clone(), None);
            assert_eq!(worker.load(), 1);
            assert_eq!(worker.worker_routing_key_load().value(), 0);
        }

        assert_eq!(worker.load(), 0);
        assert_eq!(worker.worker_routing_key_load().value(), 0);
    }

    #[test]
    fn test_worker_load_guard_multiple_same_routing_key() {
        use crate::core::BasicWorkerBuilder;

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://test:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        let mut headers = http::HeaderMap::new();
        headers.insert("x-smg-routing-key", "key-123".parse().unwrap());

        let guard1 = WorkerLoadGuard::new(worker.clone(), Some(&headers));
        assert_eq!(worker.load(), 1);
        assert_eq!(worker.worker_routing_key_load().value(), 1);

        let guard2 = WorkerLoadGuard::new(worker.clone(), Some(&headers));
        assert_eq!(worker.load(), 2);
        assert_eq!(worker.worker_routing_key_load().value(), 1);

        drop(guard1);
        assert_eq!(worker.load(), 1);
        assert_eq!(worker.worker_routing_key_load().value(), 1);

        drop(guard2);
        assert_eq!(worker.load(), 0);
        assert_eq!(worker.worker_routing_key_load().value(), 0);
    }
}
