use std::collections::HashMap;

// Re-export storage config types from data_connector
pub use data_connector::{HistoryBackend, OracleConfig, PostgresConfig, RedisConfig};
use serde::{Deserialize, Serialize};

use super::ConfigResult;
use crate::core::ConnectionMode;

pub const DEFAULT_POOL_IDLE_TIMEOUT_SECS: u64 = 50;
pub const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 10;
pub const DEFAULT_POOL_MAX_IDLE_PER_HOST: usize = 500;
pub const DEFAULT_TCP_KEEPALIVE_SECS: u64 = 30;

/// 路由器主配置
///
/// 该结构体聚合了整个 model gateway 运行所需的全部配置项，
/// 包括路由模式、负载均衡策略、网络监听、限流、重试、熔断、
/// 健康检查、分词器、历史存储后端、TLS/mTLS 以及 WASM 等能力开关。
/// 通常由 CLI 参数或配置文件构建而来。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    /// 路由模式：常规单模型、PD（Prefill-Decode）分离或 OpenAI 兼容等
    pub mode: RoutingMode,
    /// 与后端 worker 的连接方式（HTTP 或 gRPC），未指定时使用默认值
    #[serde(default)]
    pub connection_mode: ConnectionMode,
    /// 负载均衡/路由策略（如 random、round_robin、cache_aware 等）
    pub policy: PolicyConfig,
    /// 路由服务器绑定的监听主机地址（如 0.0.0.0）
    pub host: String,
    /// 路由服务器绑定的监听端口
    pub port: u16,
    /// 允许的最大请求体大小（字节）
    pub max_payload_size: usize,
    /// 单个请求的整体超时时间（秒）
    pub request_timeout_secs: u64,
    /// 等待 worker 启动并完成注册的超时时间（秒）
    pub worker_startup_timeout_secs: u64,
    /// worker 启动检查之间的轮询间隔（秒）
    pub worker_startup_check_interval_secs: u64,
    /// 是否启用数据并行（DP）感知调度
    pub dp_aware: bool,
    /// 访问 worker 时使用的 API 密钥（可选）
    pub api_key: Option<String>,
    /// 服务发现配置（如 Kubernetes 服务发现），未启用时为 None
    pub discovery: Option<DiscoveryConfig>,
    /// Prometheus 指标暴露配置，未启用时为 None
    pub metrics: Option<MetricsConfig>,
    /// OpenTelemetry 链路追踪配置，未启用时为 None
    pub trace_config: Option<TraceConfig>,
    /// 日志文件输出目录（None 表示仅输出到标准输出）
    pub log_dir: Option<String>,
    /// 日志级别（如 debug/info/warn/error）
    pub log_level: Option<String>,
    /// 用于提取请求 ID 的自定义 HTTP 头列表
    pub request_id_headers: Option<Vec<String>>,
    /// 上游 HTTP 连接池中空闲连接的存活超时时间（秒）
    #[serde(default = "default_pool_idle_timeout_secs")]
    pub pool_idle_timeout_secs: u64,
    /// 建立新的上游 HTTP 连接的超时时间（秒）
    #[serde(default = "default_connect_timeout_secs")]
    pub connect_timeout_secs: u64,
    /// 每个上游主机在连接池中保留的最大空闲连接数
    #[serde(default = "default_pool_max_idle_per_host")]
    pub pool_max_idle_per_host: usize,
    /// 上游 HTTP 连接的 TCP keepalive 空闲时间（秒）
    #[serde(default = "default_tcp_keepalive_secs")]
    pub tcp_keepalive_secs: u64,
    /// 最大并发请求数；设为 -1 表示禁用限流
    pub max_concurrent_requests: i32,
    /// 达到并发上限时，待处理请求的排队队列大小
    pub queue_size: usize,
    /// 请求在队列中允许等待的最长时间（秒）
    pub queue_timeout_secs: u64,
    /// 令牌桶补充速率（每秒令牌数）；未设置时默认取 max_concurrent_requests
    pub rate_limit_tokens_per_second: Option<i32>,
    /// CORS 允许的来源列表
    pub cors_allowed_origins: Vec<String>,
    /// 请求重试策略配置
    pub retry: RetryConfig,
    /// 熔断器配置
    pub circuit_breaker: CircuitBreakerConfig,
    /// 为 true 时，将 retry.max_retries 强制覆盖为 1（等效于禁用重试）
    #[serde(default)]
    pub disable_retries: bool,
    /// 为 true 时，将 circuit_breaker.failure_threshold 覆盖为 u32::MAX（等效于禁用熔断）
    #[serde(default)]
    pub disable_circuit_breaker: bool,
    /// worker 健康检查配置
    pub health_check: HealthCheckConfig,
    /// 是否启用 IGW（推理网关）模式以支持多模型
    #[serde(default)]
    pub enable_igw: bool,
    /// 模型路径：可以是 HuggingFace 模型 ID 或本地路径
    pub model_path: Option<String>,
    /// 显式分词器路径；若提供则覆盖 model_path 中的分词器
    pub tokenizer_path: Option<String>,
    /// 聊天模板路径
    pub chat_template: Option<String>,
    /// 历史记录存储后端类型（memory/none/oracle/postgres/redis）
    #[serde(default = "default_history_backend")]
    pub history_backend: HistoryBackend,
    /// Oracle 数据库配置；当 history_backend = "oracle" 时必填
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oracle: Option<OracleConfig>,
    /// PostgreSQL 数据库配置；当 history_backend = "postgres" 时必填
    #[serde(skip_serializing_if = "Option::is_none")]
    pub postgres: Option<PostgresConfig>,
    /// Redis 数据库配置；当 history_backend = "redis" 时必填
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redis: Option<RedisConfig>,
    /// 推理模型的思维链解析器（如 deepseek-r1、qwen3）
    pub reasoning_parser: Option<String>,
    /// 工具调用（tool-call）交互的解析器
    pub tool_call_parser: Option<String>,
    /// 分词器多级缓存配置（L0/L1）
    #[serde(default)]
    pub tokenizer_cache: TokenizerCacheConfig,
    /// 服务端 TLS 证书（PEM 格式）；不参与序列化
    #[serde(skip)]
    pub server_cert: Option<Vec<u8>>,
    /// 服务端 TLS 私钥（PEM 格式）；不参与序列化
    #[serde(skip)]
    pub server_key: Option<Vec<u8>>,
    /// 客户端身份凭证：PEM 格式的证书+私钥合并内容，
    /// 在配置创建阶段从 client_cert_path 与 client_key_path 加载；不参与序列化
    #[serde(skip)]
    pub client_identity: Option<Vec<u8>>,
    /// CA 证书列表（PEM 格式），在配置创建阶段从 ca_cert_paths 加载
    #[serde(default)]
    pub ca_certificates: Vec<Vec<u8>>,
    /// MCP 配置，在配置创建阶段从 mcp_config_path 加载；不参与序列化
    #[serde(skip)]
    pub mcp_config: Option<smg_mcp::McpConfig>,
    /// 是否启用 WASM（WebAssembly）扩展支持
    #[serde(default)]
    pub enable_wasm: bool,
}

/// Tokenizer cache configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenizerCacheConfig {
    /// Whole-string exact match cache
    #[serde(default = "default_enable_l0")]
    pub enable_l0: bool,
    #[serde(default = "default_l0_max_entries")]
    pub l0_max_entries: usize,
    /// Prefix matching at fixed boundaries
    #[serde(default = "default_enable_l1")]
    pub enable_l1: bool,
    #[serde(default = "default_l1_max_memory")]
    pub l1_max_memory: usize,
}

fn default_enable_l0() -> bool {
    false
}

fn default_l0_max_entries() -> usize {
    10_000
}

fn default_enable_l1() -> bool {
    false
}

fn default_l1_max_memory() -> usize {
    50 * 1024 * 1024 // 50MB
}

fn default_pool_idle_timeout_secs() -> u64 {
    DEFAULT_POOL_IDLE_TIMEOUT_SECS
}

fn default_connect_timeout_secs() -> u64 {
    DEFAULT_CONNECT_TIMEOUT_SECS
}

fn default_pool_max_idle_per_host() -> usize {
    DEFAULT_POOL_MAX_IDLE_PER_HOST
}

fn default_tcp_keepalive_secs() -> u64 {
    DEFAULT_TCP_KEEPALIVE_SECS
}

impl TokenizerCacheConfig {
    /// Returns Some(self) if any caching is enabled, None otherwise.
    /// Use this when passing cache config to tokenizer registration workflow.
    pub fn to_option(&self) -> Option<Self> {
        if self.enable_l0 || self.enable_l1 {
            Some(self.clone())
        } else {
            None
        }
    }
}

impl Default for TokenizerCacheConfig {
    fn default() -> Self {
        Self {
            enable_l0: default_enable_l0(),
            l0_max_entries: default_l0_max_entries(),
            enable_l1: default_enable_l1(),
            l1_max_memory: default_l1_max_memory(),
        }
    }
}

fn default_history_backend() -> HistoryBackend {
    HistoryBackend::Memory
}

/// 路由模式配置
///
/// 定义网关如何组织与调度后端 worker，共有三种模式：
/// - `Regular`：常规模式，所有 worker 同时承担 prefill 与 decode
/// - `PrefillDecode`：PD 分离模式，prefill 与 decode 由不同节点分别承担
/// - `OpenAI`：OpenAI 兼容模式，转发到 OpenAI 风格的后端
///
/// 序列化时通过内部标签字段 `type` 区分具体模式。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RoutingMode {
    /// 常规模式：每个 worker 同时处理 prefill 和 decode
    #[serde(rename = "regular")]
    Regular {
        /// worker URL 列表
        worker_urls: Vec<String>,
    },
    /// PD（Prefill-Decode）分离模式：prefill 与 decode 分别由不同节点承担
    #[serde(rename = "prefill_decode")]
    PrefillDecode {
        /// prefill 节点 URL 列表，每项可附带可选的 bootstrap 端口
        prefill_urls: Vec<(String, Option<u16>)>,
        /// decode 节点 URL 列表
        decode_urls: Vec<String>,
        /// prefill 节点的专用路由策略；为 None 时回退到主策略
        #[serde(skip_serializing_if = "Option::is_none")]
        prefill_policy: Option<PolicyConfig>,
        /// decode 节点的专用路由策略；为 None 时回退到主策略
        #[serde(skip_serializing_if = "Option::is_none")]
        decode_policy: Option<PolicyConfig>,
    },
    /// OpenAI 兼容模式：转发到 OpenAI 风格的后端
    #[serde(rename = "openai")]
    OpenAI {
        /// worker URL 列表
        worker_urls: Vec<String>,
    },
}

impl RoutingMode {
    pub fn is_pd_mode(&self) -> bool {
        matches!(self, RoutingMode::PrefillDecode { .. })
    }

    pub fn worker_count(&self) -> usize {
        match self {
            RoutingMode::Regular { worker_urls } => worker_urls.len(),
            RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls,
                ..
            } => prefill_urls.len() + decode_urls.len(),
            RoutingMode::OpenAI { .. } => 1,
        }
    }

    /// Get the effective prefill policy for PD mode
    /// Falls back to the main policy if no specific prefill policy is set
    pub fn get_prefill_policy<'a>(&'a self, main_policy: &'a PolicyConfig) -> &'a PolicyConfig {
        match self {
            RoutingMode::PrefillDecode { prefill_policy, .. } => {
                prefill_policy.as_ref().unwrap_or(main_policy)
            }
            _ => main_policy,
        }
    }

    /// Get the effective decode policy for PD mode
    /// Falls back to the main policy if no specific decode policy is set
    pub fn get_decode_policy<'a>(&'a self, main_policy: &'a PolicyConfig) -> &'a PolicyConfig {
        match self {
            RoutingMode::PrefillDecode { decode_policy, .. } => {
                decode_policy.as_ref().unwrap_or(main_policy)
            }
            _ => main_policy,
        }
    }
}

/// Manual(手动/粘性会话)策略在遇到**新路由键**时的 worker 分配方式。
///
/// Manual 策略通过请求头 `X-SMG-Routing-Key` 提供会话粘性(sticky session):
/// 同一路由键会被稳定映射到同一个 worker。当某个路由键**第一次**出现(即
/// `routing_map` 中还没有它的映射)时,需要为它挑选一个初始 worker,本枚举
/// 就决定了这次「首次分配」采用何种挑选策略。一旦分配完成,后续相同路由键的
/// 请求都会复用该映射,只有在原 worker 变得不健康时才会重新分配。
///
/// > 注意:本模式只影响**新键的初始分配**,不影响已建立映射的粘性行为。
///
/// 序列化时使用 snake_case,例如 `"random"`、`"min_load"`、`"min_group"`。
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ManualAssignmentMode {
    /// 随机分配(默认):从当前健康的 worker 中等概率随机挑选一个。
    ///
    /// 开销最小、无需读取 worker 负载信息,适合各 worker 能力相近、
    /// 且路由键数量足够大能自然均摊到各 worker 的场景。
    #[default]
    Random,

    /// 最小负载分配:挑选**当前运行中请求数(load)最少**的 worker。
    ///
    /// 依据 `Worker::load()`(即正在处理的请求数)选择负载最低者;若存在
    /// 多个并列最小值,则在这些候选中随机选一个以打散热点。适合请求处理
    /// 时长差异较大、希望把新会话导向更空闲实例的场景。
    MinLoad,

    /// 最小分组分配:挑选**当前绑定路由键数量最少**的 worker。
    ///
    /// 依据 `Worker::worker_routing_key_load()`(即该 worker 上活跃路由键
    /// 的数量)选择绑定会话最少者;并列最小值同样随机打散。相比 `MinLoad`
    /// 关注的是「会话/键的分布均衡」而非「实时请求负载」,适合希望各 worker
    /// 承载的独立会话数尽量均匀的场景。
    MinGroup,
}

/// 路由负载均衡策略配置。
///
/// 决定 Router 收到请求后如何在多个后端 worker 之间挑选目标。不同策略在
/// 「负载均衡效果」和「KV cache 命中率 / 会话亲和性」之间做不同权衡:
/// - Random / RoundRobin:最简单,只追求负载均摊,不考虑缓存局部性。
/// - CacheAware / PrefixHash:面向 KV cache,让相同前缀的请求尽量落到同一 worker 以提升缓存命中。
/// - PowerOfTwo / Bucket:基于实时负载做更精细的均衡。
/// - Manual / ConsistentHashing:提供会话粘性(sticky session),同一会话稳定路由到同一 worker。
///
/// 序列化时通过 `type` 字段区分具体策略(内部标签枚举),例如 `{"type": "cache_aware", ...}`。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PolicyConfig {
    /// 随机策略:每次从可用 worker 中等概率随机选一个,无状态、开销最小。
    #[serde(rename = "random")]
    Random,

    /// 轮询策略:按顺序依次分配请求,保证请求数在 worker 间均匀分布。
    #[serde(rename = "round_robin")]
    RoundRobin,

    /// 缓存感知策略:基于 radix tree(基数树)记录各 worker 已缓存的 token 前缀,
    /// 优先把相同前缀的请求路由到已有对应 KV cache 的 worker 以提升命中率;
    /// 负载失衡超过阈值时退回负载均衡,兼顾缓存局部性与均衡。
    #[serde(rename = "cache_aware")]
    CacheAware {
        /// 前缀匹配率阈值:请求与某 worker 缓存前缀的匹配比例 ≥ 该值时判定命中并优先路由。
        cache_threshold: f32,
        /// 负载均衡绝对差阈值:worker 间负载(请求数)绝对差超过该值时触发均衡而非追缓存。
        balance_abs_threshold: usize,
        /// 负载均衡相对比阈值:最大 / 最小负载比超过该值时触发均衡。
        balance_rel_threshold: f32,
        /// radix tree 驱逐周期(秒),周期性清理过期 / 冷门的缓存前缀节点。
        eviction_interval_secs: u64,
        /// radix tree 允许的最大节点数,超过后触发驱逐以限制内存占用。
        max_tree_size: usize,
    },

    /// Power-of-Two-Choices 策略:随机抽取两个 worker,选其中负载更低者,
    /// 以极小开销逼近最优负载均衡效果。
    #[serde(rename = "power_of_two")]
    PowerOfTwo {
        /// 负载信息的刷新间隔(秒),周期性拉取各 worker 的实时负载用于比较。
        load_check_interval_secs: u64,
    },

    /// 分桶(bucket)策略:将 worker 按负载划分到不同桶中,在满足负载均衡约束的前提下路由。
    #[serde(rename = "bucket")]
    Bucket {
        /// 负载均衡的绝对差阈值(worker 间负载绝对差超过此值触发均衡)。
        balance_abs_threshold: usize,
        /// 负载均衡的相对比阈值(最大/最小负载比超过此值触发均衡)。
        balance_rel_threshold: f32,
        /// 桶边界调整周期(秒),周期性根据实时负载重新划分桶边界。
        bucket_adjust_interval_secs: usize,
    },

    /// 手动路由策略,基于 DashMap 实现会话粘性(sticky session):
    /// - 通过 X-SMG-Routing-Key 请求头把同一路由键固定路由到缓存的 worker,或为新键分配一个 worker;
    /// - 提供真正的会话粘性——新增 worker 时不会重新分布已有键(零重分布);
    /// - 若请求未携带路由键,则退回随机选择;
    /// - 缓存条目超出上限时按 LRU / TTL 驱逐。
    #[serde(rename = "manual")]
    Manual {
        /// TTL 驱逐周期(秒,默认 60):周期性扫描并驱逐超过空闲时间的映射。
        #[serde(default = "default_manual_eviction_interval_secs")]
        eviction_interval_secs: u64,
        /// 条目被驱逐前允许的最大空闲时间(秒,默认 14400 = 4 小时)。
        #[serde(default = "default_manual_max_idle_secs")]
        max_idle_secs: u64,
        /// 为新路由键分配 worker 的方式(默认 random),见 [`ManualAssignmentMode`]。
        #[serde(default)]
        assignment_mode: ManualAssignmentMode,
    },

    /// 一致性哈希策略,使用哈希环实现会话亲和性:
    /// - 通过 X-SMG-Target-Worker 请求头按 URL 直接路由到指定 worker;
    /// - 通过 X-SMG-Routing-Key 请求头做一致性哈希路由以保持会话亲和;
    /// - 查找复杂度 O(log n),拓扑变化时只需重分布约 1/N 的键(N 为 worker 数)。
    #[serde(rename = "consistent_hashing")]
    ConsistentHashing,

    /// 前缀哈希策略,面向 KV cache 的轻量级负载均衡,是 cache_aware 基数树的简化替代:
    /// - 根据请求前缀 token 的哈希路由,以获得缓存局部性;
    /// - 使用带有界负载均衡(bounded load)的一致性哈希环;
    /// - 当目标 worker 过载(负载 > 平均值 * load_factor)时沿环向后寻找下一个 worker;
    /// - 查找复杂度 O(log n),优于基数树遍历的 O(prefix_len)。
    #[serde(rename = "prefix_hash")]
    PrefixHash {
        /// 参与哈希的前缀 token 数量(默认 256):只对请求前 N 个 token 做哈希以判定缓存归属。
        #[serde(default = "default_prefix_token_count")]
        prefix_token_count: usize,
        /// 负载因子阈值(默认 1.25):当某 worker 负载 > 平均负载 * 该因子时视为过载,沿环换下一个 worker。
        #[serde(default = "default_load_factor")]
        load_factor: f64,
    },
}

fn default_prefix_token_count() -> usize {
    256
}

fn default_load_factor() -> f64 {
    1.25
}

fn default_manual_eviction_interval_secs() -> u64 {
    60
}

fn default_manual_max_idle_secs() -> u64 {
    4 * 3600
}

impl PolicyConfig {
    pub fn name(&self) -> &'static str {
        match self {
            PolicyConfig::Random => "random",
            PolicyConfig::RoundRobin => "round_robin",
            PolicyConfig::CacheAware { .. } => "cache_aware",
            PolicyConfig::PowerOfTwo { .. } => "power_of_two",
            PolicyConfig::Bucket { .. } => "bucket",
            PolicyConfig::Manual { .. } => "manual",
            PolicyConfig::ConsistentHashing => "consistent_hashing",
            PolicyConfig::PrefixHash { .. } => "prefix_hash",
        }
    }
}

/// Service discovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    pub enabled: bool,
    /// None = all namespaces
    pub namespace: Option<String>,
    pub port: u16,
    pub check_interval_secs: u64,
    /// Regular mode
    pub selector: HashMap<String, String>,
    /// PD mode prefill
    pub prefill_selector: HashMap<String, String>,
    /// PD mode decode
    pub decode_selector: HashMap<String, String>,
    pub bootstrap_port_annotation: String,
    /// Router node discovery for HA (Kubernetes label selector)
    #[serde(default)]
    pub router_selector: HashMap<String, String>,
    /// Annotation key to read mesh port from Router Pods
    #[serde(default = "default_router_mesh_port_annotation")]
    pub router_mesh_port_annotation: String,
}

fn default_router_mesh_port_annotation() -> String {
    "sglang.ai/mesh-port".to_string()
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            namespace: None,
            port: 8000,
            check_interval_secs: 120,
            selector: HashMap::new(),
            prefill_selector: HashMap::new(),
            decode_selector: HashMap::new(),
            bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
            router_selector: HashMap::new(),
            router_mesh_port_annotation: default_router_mesh_port_annotation(),
        }
    }
}

/// Retry configuration for request handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_backoff_ms: u64,
    pub max_backoff_ms: u64,
    pub backoff_multiplier: f32,
    /// D' = D * (1 + U[-j, +j]) where j is jitter factor
    #[serde(default = "default_retry_jitter_factor")]
    pub jitter_factor: f32,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 5,
            initial_backoff_ms: 50,
            max_backoff_ms: 30000,
            backoff_multiplier: 1.5,
            jitter_factor: 0.2,
        }
    }
}

fn default_retry_jitter_factor() -> f32 {
    0.2
}

/// Health check configuration for worker monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout_secs: u64,
    pub check_interval_secs: u64,
    pub endpoint: String,
    pub disable_health_check: bool,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 3,
            success_threshold: 2,
            timeout_secs: 5,
            check_interval_secs: 60,
            endpoint: "/health".to_string(),
            disable_health_check: false,
        }
    }
}

/// Circuit breaker configuration for worker reliability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout_duration_secs: u64,
    pub window_duration_secs: u64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 10,
            success_threshold: 3,
            timeout_duration_secs: 60,
            window_duration_secs: 120,
        }
    }
}

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub port: u16,
    pub host: String,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            port: 29000,
            host: "0.0.0.0".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceConfig {
    pub enable_trace: bool,
    pub otlp_traces_endpoint: String,
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            enable_trace: false,
            otlp_traces_endpoint: "localhost:4317".to_string(),
        }
    }
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            mode: RoutingMode::Regular {
                worker_urls: vec![],
            },
            policy: PolicyConfig::Random,
            host: "0.0.0.0".to_string(),
            port: 3001,
            max_payload_size: 536_870_912,     // 512MB
            request_timeout_secs: 1800,        // 30 minutes
            worker_startup_timeout_secs: 1800, // 30 minutes for large model loading
            worker_startup_check_interval_secs: 30,
            dp_aware: false,
            api_key: None,
            discovery: None,
            metrics: None,
            trace_config: None,
            log_dir: None,
            log_level: None,
            request_id_headers: None,
            pool_idle_timeout_secs: default_pool_idle_timeout_secs(),
            connect_timeout_secs: default_connect_timeout_secs(),
            pool_max_idle_per_host: default_pool_max_idle_per_host(),
            tcp_keepalive_secs: default_tcp_keepalive_secs(),
            max_concurrent_requests: -1,
            queue_size: 100,
            queue_timeout_secs: 60,
            rate_limit_tokens_per_second: None,
            cors_allowed_origins: vec![],
            retry: RetryConfig::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
            disable_retries: false,
            disable_circuit_breaker: false,
            health_check: HealthCheckConfig::default(),
            enable_igw: false,
            connection_mode: ConnectionMode::Http,
            model_path: None,
            tokenizer_path: None,
            chat_template: None,
            history_backend: default_history_backend(),
            oracle: None,
            postgres: None,
            redis: None,
            reasoning_parser: None,
            tool_call_parser: None,
            tokenizer_cache: TokenizerCacheConfig::default(),
            client_identity: None,
            ca_certificates: vec![],
            mcp_config: None,
            enable_wasm: false,
            server_cert: None,
            server_key: None,
        }
    }
}

impl RouterConfig {
    /// Create a new configuration with mode and policy
    pub fn new(mode: RoutingMode, policy: PolicyConfig) -> Self {
        Self {
            mode,
            policy,
            ..Default::default()
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> ConfigResult<()> {
        crate::config::validation::ConfigValidator::validate(self)
    }

    /// Get the routing mode type as a string
    pub fn mode_type(&self) -> &'static str {
        match self.mode {
            RoutingMode::Regular { .. } => "regular",
            RoutingMode::PrefillDecode { .. } => "prefill_decode",
            RoutingMode::OpenAI { .. } => "openai",
        }
    }

    /// Check if service discovery is enabled
    pub fn has_service_discovery(&self) -> bool {
        self.discovery.as_ref().is_some_and(|d| d.enabled)
    }

    /// Check if metrics are enabled
    pub fn has_metrics(&self) -> bool {
        self.metrics.is_some()
    }

    /// Check if tracing is enabled
    pub fn has_tracing(&self) -> bool {
        match &self.trace_config {
            Some(trace_config) => trace_config.enable_trace,
            None => false,
        }
    }

    /// Compute the effective retry config considering disable flag
    pub fn effective_retry_config(&self) -> RetryConfig {
        let mut cfg = self.retry.clone();
        if self.disable_retries {
            cfg.max_retries = 1;
        }
        cfg
    }

    /// Compute the effective circuit breaker config considering disable flag
    pub fn effective_circuit_breaker_config(&self) -> CircuitBreakerConfig {
        let mut cfg = self.circuit_breaker.clone();
        if self.disable_circuit_breaker {
            cfg.failure_threshold = u32::MAX;
        }
        cfg
    }

    /// Check if running in IGW (Inference Gateway) mode
    pub fn is_igw_mode(&self) -> bool {
        self.enable_igw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_config_default() {
        let config = RouterConfig::default();

        assert!(
            matches!(config.mode, RoutingMode::Regular { worker_urls } if worker_urls.is_empty())
        );
        assert!(matches!(config.policy, PolicyConfig::Random));
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 3001);
        assert_eq!(config.max_payload_size, 536_870_912);
        assert_eq!(config.request_timeout_secs, 1800);
        assert_eq!(config.worker_startup_timeout_secs, 1800);
        assert_eq!(config.worker_startup_check_interval_secs, 30);
        assert!(config.discovery.is_none());
        assert!(config.metrics.is_none());
        assert!(config.trace_config.is_none());
        assert!(config.log_dir.is_none());
        assert!(config.log_level.is_none());
        assert_eq!(
            config.pool_idle_timeout_secs,
            DEFAULT_POOL_IDLE_TIMEOUT_SECS
        );
        assert_eq!(config.connect_timeout_secs, DEFAULT_CONNECT_TIMEOUT_SECS);
        assert_eq!(
            config.pool_max_idle_per_host,
            DEFAULT_POOL_MAX_IDLE_PER_HOST
        );
        assert_eq!(config.tcp_keepalive_secs, DEFAULT_TCP_KEEPALIVE_SECS);
    }

    #[test]
    fn test_router_config_new() {
        let mode = RoutingMode::Regular {
            worker_urls: vec!["http://worker1".to_string(), "http://worker2".to_string()],
        };
        let policy = PolicyConfig::RoundRobin;

        let config = RouterConfig::new(mode, policy);

        match config.mode {
            RoutingMode::Regular { worker_urls } => {
                assert_eq!(worker_urls.len(), 2);
                assert_eq!(worker_urls[0], "http://worker1");
                assert_eq!(worker_urls[1], "http://worker2");
            }
            _ => panic!("Expected Regular mode"),
        }

        assert!(matches!(config.policy, PolicyConfig::RoundRobin));
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 3001);
    }

    #[test]
    fn test_router_config_serialization() {
        let config = RouterConfig::builder()
            .regular_mode(vec!["http://worker1".to_string()])
            .random_policy()
            .host("0.0.0.0")
            .port(8080)
            .log_dir("/var/log")
            .log_level("debug")
            .build_unchecked();

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: RouterConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.host, deserialized.host);
        assert_eq!(config.port, deserialized.port);
        assert_eq!(config.max_payload_size, deserialized.max_payload_size);
        assert_eq!(config.log_dir, deserialized.log_dir);
        assert_eq!(config.log_level, deserialized.log_level);
        assert!(deserialized.discovery.is_none());
        assert!(deserialized.metrics.is_none());
        assert!(deserialized.trace_config.is_none());
    }

    #[test]
    fn test_router_config_http_client_deserialization_defaults() {
        let config = RouterConfig::default();
        let mut json = serde_json::to_value(&config).unwrap();
        let json_object = json.as_object_mut().unwrap();
        json_object.remove("pool_idle_timeout_secs");
        json_object.remove("connect_timeout_secs");
        json_object.remove("pool_max_idle_per_host");
        json_object.remove("tcp_keepalive_secs");

        let deserialized: RouterConfig = serde_json::from_value(json).unwrap();

        assert_eq!(
            deserialized.pool_idle_timeout_secs,
            DEFAULT_POOL_IDLE_TIMEOUT_SECS
        );
        assert_eq!(
            deserialized.connect_timeout_secs,
            DEFAULT_CONNECT_TIMEOUT_SECS
        );
        assert_eq!(
            deserialized.pool_max_idle_per_host,
            DEFAULT_POOL_MAX_IDLE_PER_HOST
        );
        assert_eq!(deserialized.tcp_keepalive_secs, DEFAULT_TCP_KEEPALIVE_SECS);
    }

    #[test]
    fn test_routing_mode_is_pd_mode() {
        let regular = RoutingMode::Regular {
            worker_urls: vec!["http://worker1".to_string()],
        };
        assert!(!regular.is_pd_mode());

        let pd = RoutingMode::PrefillDecode {
            prefill_urls: vec![("http://prefill1".to_string(), Some(8001))],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: None,
            decode_policy: None,
        };
        assert!(pd.is_pd_mode());
    }

    #[test]
    fn test_routing_mode_worker_count() {
        let regular = RoutingMode::Regular {
            worker_urls: vec![
                "http://worker1".to_string(),
                "http://worker2".to_string(),
                "http://worker3".to_string(),
            ],
        };
        assert_eq!(regular.worker_count(), 3);

        let pd = RoutingMode::PrefillDecode {
            prefill_urls: vec![
                ("http://prefill1".to_string(), Some(8001)),
                ("http://prefill2".to_string(), None),
            ],
            decode_urls: vec![
                "http://decode1".to_string(),
                "http://decode2".to_string(),
                "http://decode3".to_string(),
            ],
            prefill_policy: None,
            decode_policy: None,
        };
        assert_eq!(pd.worker_count(), 5);

        let empty_regular = RoutingMode::Regular {
            worker_urls: vec![],
        };
        assert_eq!(empty_regular.worker_count(), 0);
    }

    #[test]
    fn test_routing_mode_serialization() {
        let regular = RoutingMode::Regular {
            worker_urls: vec!["http://worker1".to_string()],
        };
        let json = serde_json::to_string(&regular).unwrap();
        assert!(json.contains("\"type\":\"regular\""));
        assert!(json.contains("\"worker_urls\""));

        let pd = RoutingMode::PrefillDecode {
            prefill_urls: vec![("http://prefill1".to_string(), Some(8001))],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: None,
            decode_policy: None,
        };
        let json = serde_json::to_string(&pd).unwrap();
        assert!(json.contains("\"type\":\"prefill_decode\""));
        assert!(json.contains("\"prefill_urls\""));
        assert!(json.contains("\"decode_urls\""));
    }

    #[test]
    fn test_policy_config_name() {
        assert_eq!(PolicyConfig::Random.name(), "random");
        assert_eq!(PolicyConfig::RoundRobin.name(), "round_robin");

        let cache_aware = PolicyConfig::CacheAware {
            cache_threshold: 0.8,
            balance_abs_threshold: 10,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 300,
            max_tree_size: 1000,
        };
        assert_eq!(cache_aware.name(), "cache_aware");

        let power_of_two = PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 60,
        };
        assert_eq!(power_of_two.name(), "power_of_two");
    }

    #[test]
    fn test_policy_config_serialization() {
        let random = PolicyConfig::Random;
        let json = serde_json::to_string(&random).unwrap();
        assert_eq!(json, r#"{"type":"random"}"#);

        let cache_aware = PolicyConfig::CacheAware {
            cache_threshold: 0.8,
            balance_abs_threshold: 10,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 300,
            max_tree_size: 1000,
        };
        let json = serde_json::to_string(&cache_aware).unwrap();
        assert!(json.contains("\"type\":\"cache_aware\""));
        assert!(json.contains("\"cache_threshold\":0.8"));
        assert!(json.contains("\"balance_abs_threshold\":10"));

        let power_of_two = PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 60,
        };
        let json = serde_json::to_string(&power_of_two).unwrap();
        assert!(json.contains("\"type\":\"power_of_two\""));
        assert!(json.contains("\"load_check_interval_secs\":60"));
    }

    #[test]
    fn test_cache_aware_parameters() {
        let cache_aware = PolicyConfig::CacheAware {
            cache_threshold: 0.75,
            balance_abs_threshold: 20,
            balance_rel_threshold: 2.0,
            eviction_interval_secs: 600,
            max_tree_size: 5000,
        };

        match cache_aware {
            PolicyConfig::CacheAware {
                cache_threshold,
                balance_abs_threshold,
                balance_rel_threshold,
                eviction_interval_secs,
                max_tree_size,
            } => {
                assert!((cache_threshold - 0.75).abs() < 0.0001);
                assert_eq!(balance_abs_threshold, 20);
                assert!((balance_rel_threshold - 2.0).abs() < 0.0001);
                assert_eq!(eviction_interval_secs, 600);
                assert_eq!(max_tree_size, 5000);
            }
            _ => panic!("Expected CacheAware"),
        }
    }

    #[test]
    fn test_power_of_two_parameters() {
        let power_of_two = PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 120,
        };

        match power_of_two {
            PolicyConfig::PowerOfTwo {
                load_check_interval_secs,
            } => {
                assert_eq!(load_check_interval_secs, 120);
            }
            _ => panic!("Expected PowerOfTwo"),
        }
    }

    #[test]
    fn test_bucket_parameters() {
        let bucket = PolicyConfig::Bucket {
            balance_abs_threshold: 20,
            balance_rel_threshold: 2.0,
            bucket_adjust_interval_secs: 5,
        };

        match bucket {
            PolicyConfig::Bucket {
                balance_abs_threshold,
                balance_rel_threshold,
                bucket_adjust_interval_secs,
            } => {
                assert_eq!(balance_abs_threshold, 20);
                assert!((balance_rel_threshold - 2.0).abs() < 0.0001);
                assert_eq!(bucket_adjust_interval_secs, 5);
            }
            _ => panic!("Expected Bucket"),
        }
    }

    #[test]
    fn test_discovery_config_default() {
        let config = DiscoveryConfig::default();

        assert!(!config.enabled);
        assert!(config.namespace.is_none());
        assert_eq!(config.port, 8000);
        assert_eq!(config.check_interval_secs, 120);
        assert!(config.selector.is_empty());
        assert!(config.prefill_selector.is_empty());
        assert!(config.decode_selector.is_empty());
        assert_eq!(config.bootstrap_port_annotation, "sglang.ai/bootstrap-port");
    }

    #[test]
    fn test_discovery_config_with_selectors() {
        let mut selector = HashMap::new();
        selector.insert("app".to_string(), "sglang".to_string());
        selector.insert("role".to_string(), "worker".to_string());

        let config = DiscoveryConfig {
            enabled: true,
            namespace: Some("default".to_string()),
            port: 9000,
            check_interval_secs: 30,
            selector: selector.clone(),
            prefill_selector: selector.clone(),
            decode_selector: selector.clone(),
            bootstrap_port_annotation: "custom.io/port".to_string(),
            router_selector: HashMap::new(),
            router_mesh_port_annotation: "sglang.ai/mesh-port".to_string(),
        };

        assert!(config.enabled);
        assert_eq!(config.namespace, Some("default".to_string()));
        assert_eq!(config.port, 9000);
        assert_eq!(config.selector.len(), 2);
        assert_eq!(config.selector.get("app"), Some(&"sglang".to_string()));
    }

    #[test]
    fn test_discovery_config_namespace() {
        let config = DiscoveryConfig {
            namespace: None,
            ..Default::default()
        };
        assert!(config.namespace.is_none());

        let config = DiscoveryConfig {
            namespace: Some("production".to_string()),
            ..Default::default()
        };
        assert_eq!(config.namespace, Some("production".to_string()));
    }

    #[test]
    fn test_metrics_config_default() {
        let config = MetricsConfig::default();

        assert_eq!(config.port, 29000);
        assert_eq!(config.host, "0.0.0.0");
    }

    #[test]
    fn test_metrics_config_custom() {
        let config = MetricsConfig {
            port: 9090,
            host: "0.0.0.0".to_string(),
        };

        assert_eq!(config.port, 9090);
        assert_eq!(config.host, "0.0.0.0");
    }

    #[test]
    fn test_trace_config_default() {
        let config = TraceConfig::default();

        assert!(!config.enable_trace);
        assert_eq!(config.otlp_traces_endpoint, "localhost:4317");
    }

    #[test]
    fn test_trace_config_custom() {
        let config = TraceConfig {
            enable_trace: true,
            otlp_traces_endpoint: "otel-collector:4317".to_string(),
        };

        assert!(config.enable_trace);
        assert_eq!(config.otlp_traces_endpoint, "otel-collector:4317");
    }

    #[test]
    fn test_mode_type() {
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .build_unchecked();
        assert_eq!(config.mode_type(), "regular");

        let config = RouterConfig::builder()
            .prefill_decode_mode(vec![], vec![])
            .build_unchecked();
        assert_eq!(config.mode_type(), "prefill_decode");
    }

    #[test]
    fn test_has_service_discovery() {
        let config = RouterConfig::default();
        assert!(!config.has_service_discovery());

        let config = RouterConfig::builder()
            .discovery_config(DiscoveryConfig {
                enabled: false,
                ..Default::default()
            })
            .build_unchecked();
        assert!(!config.has_service_discovery());

        let config = RouterConfig::builder().enable_discovery().build_unchecked();
        assert!(config.has_service_discovery());
    }

    #[test]
    fn test_has_metrics() {
        let config = RouterConfig::default();
        assert!(!config.has_metrics());

        let config = RouterConfig::builder()
            .metrics_config(MetricsConfig::default())
            .build_unchecked();
        assert!(config.has_metrics());
    }

    #[test]
    fn test_has_tracing() {
        let config = RouterConfig::default();
        assert!(!config.has_tracing());

        let config = RouterConfig::builder()
            .enable_trace("localhost:4317")
            .build_unchecked();
        assert!(config.has_tracing());
    }

    #[test]
    fn test_large_worker_lists() {
        let large_urls: Vec<String> = (0..1000).map(|i| format!("http://worker{}", i)).collect();

        let config = RouterConfig::builder()
            .regular_mode(large_urls.clone())
            .build_unchecked();

        assert_eq!(config.mode.worker_count(), 1000);

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: RouterConfig = serde_json::from_str(&json).unwrap();

        match deserialized.mode {
            RoutingMode::Regular { worker_urls } => {
                assert_eq!(worker_urls.len(), 1000);
            }
            _ => panic!("Expected Regular mode"),
        }
    }

    #[test]
    fn test_unicode_in_config() {
        let config = RouterConfig::builder()
            .regular_mode(vec![
                "http://работник1".to_string(),
                "http://工作者2".to_string(),
            ])
            .log_dir("/日志/目录")
            .build_unchecked();

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: RouterConfig = serde_json::from_str(&json).unwrap();

        match deserialized.mode {
            RoutingMode::Regular { worker_urls } => {
                assert_eq!(worker_urls[0], "http://работник1");
                assert_eq!(worker_urls[1], "http://工作者2");
            }
            _ => panic!("Expected Regular mode"),
        }

        assert_eq!(deserialized.log_dir, Some("/日志/目录".to_string()));
    }

    #[test]
    fn test_empty_string_fields() {
        let config = RouterConfig::builder()
            .host("")
            .log_dir("")
            .log_level("")
            .build_unchecked();

        assert_eq!(config.host, "");
        assert_eq!(config.log_dir, Some("".to_string()));
        assert_eq!(config.log_level, Some("".to_string()));
    }

    #[test]
    fn test_full_pd_mode_config() {
        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![
                    ("http://prefill1:8000".to_string(), Some(8001)),
                    ("http://prefill2:8000".to_string(), None),
                ],
                vec![
                    "http://decode1:8000".to_string(),
                    "http://decode2:8000".to_string(),
                ],
            )
            .power_of_two_policy(30)
            .host("0.0.0.0")
            .port(3000)
            .max_payload_size(1048576)
            .request_timeout_secs(120)
            .worker_startup_timeout_secs(60)
            .worker_startup_check_interval_secs(5)
            .discovery_config(DiscoveryConfig {
                enabled: true,
                namespace: Some("sglang".to_string()),
                ..Default::default()
            })
            .enable_metrics("0.0.0.0", 9090)
            .enable_trace("localhost:4317")
            .log_dir("/var/log/sglang")
            .log_level("info")
            .max_concurrent_requests(64)
            .build_unchecked();

        assert!(config.mode.is_pd_mode());
        assert_eq!(config.mode.worker_count(), 4);
        assert_eq!(config.policy.name(), "power_of_two");
        assert!(config.has_service_discovery());
        assert!(config.has_metrics());
        assert!(config.has_tracing());
    }

    #[test]
    fn test_full_regular_mode_config() {
        let mut selector = HashMap::new();
        selector.insert("app".to_string(), "sglang".to_string());

        let config = RouterConfig::builder()
            .regular_mode(vec![
                "http://worker1:8000".to_string(),
                "http://worker2:8000".to_string(),
                "http://worker3:8000".to_string(),
            ])
            .cache_aware_policy(0.9, 5, 1.2, 600, 10000)
            .host("0.0.0.0")
            .port(3001)
            .max_payload_size(536870912)
            .request_timeout_secs(300)
            .worker_startup_timeout_secs(180)
            .worker_startup_check_interval_secs(15)
            .discovery_config(DiscoveryConfig {
                enabled: true,
                namespace: None,
                port: 8080,
                check_interval_secs: 45,
                selector,
                ..Default::default()
            })
            .metrics_config(MetricsConfig::default())
            .enable_trace("localhost:4317")
            .log_level("debug")
            .max_concurrent_requests(64)
            .build_unchecked();

        assert!(!config.mode.is_pd_mode());
        assert_eq!(config.mode.worker_count(), 3);
        assert_eq!(config.policy.name(), "cache_aware");
        assert!(config.has_service_discovery());
        assert!(config.has_metrics());
        assert!(config.has_tracing());
    }

    #[test]
    fn test_config_with_all_options() {
        let mut selectors = HashMap::new();
        selectors.insert("env".to_string(), "prod".to_string());
        selectors.insert("version".to_string(), "v1".to_string());

        let config = RouterConfig::builder()
            .regular_mode(vec!["http://worker1".to_string()])
            .round_robin_policy()
            .host("::1") // IPv6
            .port(8888)
            .max_payload_size(1024 * 1024 * 512) // 512MB
            .request_timeout_secs(900)
            .worker_startup_timeout_secs(600)
            .worker_startup_check_interval_secs(20)
            .discovery_config(DiscoveryConfig {
                enabled: true,
                namespace: Some("production".to_string()),
                port: 8443,
                check_interval_secs: 120,
                selector: selectors.clone(),
                prefill_selector: selectors.clone(),
                decode_selector: selectors,
                bootstrap_port_annotation: "mycompany.io/bootstrap".to_string(),
                router_selector: HashMap::new(),
                router_mesh_port_annotation: "sglang.ai/mesh-port".to_string(),
            })
            .enable_metrics("::", 9999) // IPv6 any
            .enable_trace("localhost:4317")
            .log_dir("/opt/logs/sglang")
            .log_level("trace")
            .max_concurrent_requests(64)
            .build_unchecked();

        assert!(config.has_service_discovery());
        assert!(config.has_metrics());
        assert!(config.has_tracing());
        assert_eq!(config.mode_type(), "regular");

        let json = serde_json::to_string_pretty(&config).unwrap();
        let deserialized: RouterConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.host, "::1");
        assert_eq!(deserialized.port, 8888);
        assert_eq!(
            deserialized.discovery.unwrap().namespace,
            Some("production".to_string())
        );
    }

    #[test]
    fn test_pd_policy_fallback_both_specified() {
        let pd = RoutingMode::PrefillDecode {
            prefill_urls: vec![("http://prefill1".to_string(), None)],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: Some(PolicyConfig::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                eviction_interval_secs: 60,
                max_tree_size: 1000,
            }),
            decode_policy: Some(PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 60,
            }),
        };

        let main_policy = PolicyConfig::Random;

        match pd.get_prefill_policy(&main_policy) {
            PolicyConfig::CacheAware { .. } => {}
            _ => panic!("Expected CacheAware for prefill"),
        }

        match pd.get_decode_policy(&main_policy) {
            PolicyConfig::PowerOfTwo { .. } => {}
            _ => panic!("Expected PowerOfTwo for decode"),
        }
    }

    #[test]
    fn test_pd_policy_fallback_only_prefill() {
        let pd = RoutingMode::PrefillDecode {
            prefill_urls: vec![("http://prefill1".to_string(), None)],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: Some(PolicyConfig::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                eviction_interval_secs: 60,
                max_tree_size: 1000,
            }),
            decode_policy: None,
        };

        let main_policy = PolicyConfig::RoundRobin;

        match pd.get_prefill_policy(&main_policy) {
            PolicyConfig::CacheAware { .. } => {}
            _ => panic!("Expected CacheAware for prefill"),
        }

        match pd.get_decode_policy(&main_policy) {
            PolicyConfig::RoundRobin => {}
            _ => panic!("Expected RoundRobin for decode"),
        }
    }

    #[test]
    fn test_pd_policy_fallback_only_decode() {
        let pd = RoutingMode::PrefillDecode {
            prefill_urls: vec![("http://prefill1".to_string(), None)],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: None,
            decode_policy: Some(PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 60,
            }),
        };

        let main_policy = PolicyConfig::Random;

        match pd.get_prefill_policy(&main_policy) {
            PolicyConfig::Random => {}
            _ => panic!("Expected Random for prefill"),
        }

        match pd.get_decode_policy(&main_policy) {
            PolicyConfig::PowerOfTwo { .. } => {}
            _ => panic!("Expected PowerOfTwo for decode"),
        }
    }

    #[test]
    fn test_pd_policy_fallback_none_specified() {
        let pd = RoutingMode::PrefillDecode {
            prefill_urls: vec![("http://prefill1".to_string(), None)],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: None,
            decode_policy: None,
        };

        let main_policy = PolicyConfig::CacheAware {
            cache_threshold: 0.7,
            balance_abs_threshold: 20,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 300,
            max_tree_size: 2000,
        };

        match pd.get_prefill_policy(&main_policy) {
            PolicyConfig::CacheAware {
                cache_threshold, ..
            } => {
                assert!((cache_threshold - 0.7).abs() < 0.0001);
            }
            _ => panic!("Expected CacheAware for prefill"),
        }

        match pd.get_decode_policy(&main_policy) {
            PolicyConfig::CacheAware {
                cache_threshold, ..
            } => {
                assert!((cache_threshold - 0.7).abs() < 0.0001);
            }
            _ => panic!("Expected CacheAware for decode"),
        }
    }

    #[test]
    fn test_regular_mode_policy_fallback() {
        let regular = RoutingMode::Regular {
            worker_urls: vec!["http://worker1".to_string()],
        };

        let main_policy = PolicyConfig::RoundRobin;

        match regular.get_prefill_policy(&main_policy) {
            PolicyConfig::RoundRobin => {}
            _ => panic!("Expected RoundRobin for regular mode"),
        }

        match regular.get_decode_policy(&main_policy) {
            PolicyConfig::RoundRobin => {}
            _ => panic!("Expected RoundRobin for regular mode"),
        }
    }
}
