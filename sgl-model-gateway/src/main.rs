use std::collections::HashMap;

use clap::{ArgAction, Parser, Subcommand, ValueEnum};
use rand::{distr::Alphanumeric, Rng};
use smg::{
    auth::{ApiKeyEntry, ControlPlaneAuthConfig, JwtConfig, Role},
    config::{
        CircuitBreakerConfig, ConfigError, ConfigResult, DiscoveryConfig, HealthCheckConfig,
        HistoryBackend, ManualAssignmentMode, MetricsConfig, OracleConfig, PolicyConfig,
        PostgresConfig, RedisConfig, RetryConfig, RouterConfig, RoutingMode, TokenizerCacheConfig,
        TraceConfig, DEFAULT_CONNECT_TIMEOUT_SECS, DEFAULT_POOL_IDLE_TIMEOUT_SECS,
        DEFAULT_POOL_MAX_IDLE_PER_HOST, DEFAULT_TCP_KEEPALIVE_SECS,
    },
    core::ConnectionMode,
    observability::{
        metrics::PrometheusConfig,
        otel_trace::{is_otel_enabled, shutdown_otel},
    },
    server::{self, ServerConfig},
    service_discovery::ServiceDiscoveryConfig,
    version,
};
use smg_mesh::service::MeshServerConfig;
fn parse_prefill_args() -> Vec<(String, Option<u16>)> {
    let args: Vec<String> = std::env::args().collect();
    let mut prefill_entries = Vec::new();
    let mut i = 0;

    while i < args.len() {
        if args[i] == "--prefill" && i + 1 < args.len() {
            let url = args[i + 1].clone();
            let bootstrap_port = if i + 2 < args.len() && !args[i + 2].starts_with("--") {
                if let Ok(port) = args[i + 2].parse::<u16>() {
                    i += 1;
                    Some(port)
                } else if args[i + 2].to_lowercase() == "none" {
                    i += 1;
                    None
                } else {
                    None
                }
            } else {
                None
            };
            prefill_entries.push((url, bootstrap_port));
            i += 2;
        } else {
            i += 1;
        }
    }

    prefill_entries
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum Backend {
    #[value(name = "sglang")]
    Sglang,
    #[value(name = "vllm")]
    Vllm,
    #[value(name = "trtllm")]
    Trtllm,
    #[value(name = "openai")]
    Openai,
    #[value(name = "anthropic")]
    Anthropic,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Backend::Sglang => "sglang",
            Backend::Vllm => "vllm",
            Backend::Trtllm => "trtllm",
            Backend::Openai => "openai",
            Backend::Anthropic => "anthropic",
        };
        write!(f, "{}", s)
    }
}

#[derive(Parser, Debug)]
#[command(name = "sglang-router", alias = "smg", alias = "amg")]
#[command(about = "SGLang Model Gateway - High-performance inference gateway")]
#[command(args_conflicts_with_subcommands = true)]
#[command(long_about = r#"
SGLang Model Gateway - Rust-based inference gateway

Usage:
  smg launch [OPTIONS]     Launch router (short command)
  amg launch [OPTIONS]     Launch router (alternative)
  sglang-router [OPTIONS]  Launch router (full name)

Examples:
  # Regular mode
  smg launch --worker-urls http://worker1:8000 http://worker2:8000

  # PD disaggregated mode
  smg launch --pd-disaggregation \
    --prefill http://127.0.0.1:30001 9001 \
    --prefill http://127.0.0.2:30002 9002 \
    --decode http://127.0.0.3:30003 \
    --decode http://127.0.0.4:30004 \
    --policy cache_aware

  # With different policies
  smg launch --pd-disaggregation \
    --prefill http://127.0.0.1:30001 9001 \
    --prefill http://127.0.0.2:30002 \
    --decode http://127.0.0.3:30003 \
    --decode http://127.0.0.4:30004 \
    --prefill-policy cache_aware --decode-policy power_of_two

"#)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    #[command(flatten)]
    router_args: CliArgs,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Launch the router (same as running without subcommand)
    #[command(visible_alias = "start")]
    Launch {
        #[command(flatten)]
        args: CliArgs,
    },
}

#[derive(Parser, Debug)]
struct CliArgs {
    // ==================== Worker Configuration ====================
    /// 路由服务器绑定的主机地址
    #[arg(long, default_value = "0.0.0.0", help_heading = "Worker Configuration")]
    host: String,

    /// 路由服务器绑定的端口号
    #[arg(long, default_value_t = 30000, help_heading = "Worker Configuration")]
    port: u16,

    /// worker URL 列表（支持 IPv4 和 IPv6）
    #[arg(long, num_args = 0.., help_heading = "Worker Configuration")]
    worker_urls: Vec<String>,

    // ==================== Routing Policy ====================
    /// 使用的负载均衡策略
    #[arg(long, default_value = "cache_aware", value_parser = ["random", "round_robin", "cache_aware", "power_of_two", "prefix_hash", "manual"], help_heading = "Routing Policy")]
    policy: String,

    /// 缓存感知路由的缓存阈值（0.0-1.0）
    #[arg(long, default_value_t = 0.3, help_heading = "Routing Policy")]
    cache_threshold: f32,

    /// 触发负载均衡的绝对阈值
    #[arg(long, default_value_t = 64, help_heading = "Routing Policy")]
    balance_abs_threshold: usize,

    /// 触发负载均衡的相对阈值
    #[arg(long, default_value_t = 1.5, help_heading = "Routing Policy")]
    balance_rel_threshold: f32,

    /// 缓存淘汰操作之间的间隔（秒）
    #[arg(long, default_value_t = 120, help_heading = "Routing Policy")]
    eviction_interval: u64,

    /// 缓存感知路由中近似树的最大大小
    #[arg(long, default_value_t = 67108864, help_heading = "Routing Policy")]
    max_tree_size: usize,

    /// 淘汰前的最大空闲时间（秒）（用于 manual 策略）
    #[arg(long, default_value_t = 14400, help_heading = "Routing Policy")]
    max_idle_secs: u64,

    /// manual 策略遇到新路由键时的分配模式
    #[arg(long, default_value = "random", value_parser = ["random", "min_load", "min_group"], help_heading = "Routing Policy")]
    assignment_mode: String,

    /// prefix_hash 策略使用的前缀 token 数量
    #[arg(long, default_value_t = 256, help_heading = "Routing Policy")]
    prefix_token_count: usize,

    /// prefix_hash 策略的负载因子阈值
    #[arg(long, default_value_t = 1.25, help_heading = "Routing Policy")]
    prefix_hash_load_factor: f64,

    /// 启用数据并行感知调度
    #[arg(long, default_value_t = false, help_heading = "Routing Policy")]
    dp_aware: bool,

    /// 启用 IGW（推理网关）模式以支持多模型
    #[arg(long, default_value_t = false, help_heading = "Routing Policy")]
    enable_igw: bool,

    // ==================== PD Disaggregation ====================
    /// 启用 PD（Prefill-Decode）分离模式
    #[arg(long, default_value_t = false, help_heading = "PD Disaggregation")]
    pd_disaggregation: bool,

    /// Decode 服务器 URL（可多次指定）
    #[arg(long, action = ArgAction::Append, help_heading = "PD Disaggregation")]
    decode: Vec<String>,

    /// PD 模式下 prefill 节点的专用策略
    #[arg(long, value_parser = ["random", "round_robin", "cache_aware", "power_of_two", "prefix_hash", "manual"], help_heading = "PD Disaggregation")]
    prefill_policy: Option<String>,

    /// PD 模式下 decode 节点的专用策略
    #[arg(long, value_parser = ["random", "round_robin", "cache_aware", "power_of_two", "prefix_hash", "manual"], help_heading = "PD Disaggregation")]
    decode_policy: Option<String>,

    /// worker 启动与注册的超时时间（秒）
    #[arg(long, default_value_t = 1800, help_heading = "PD Disaggregation")]
    worker_startup_timeout_secs: u64,

    /// worker 启动检查之间的间隔（秒）
    #[arg(long, default_value_t = 30, help_heading = "PD Disaggregation")]
    worker_startup_check_interval: u64,

    // ==================== Service Discovery (Kubernetes) ====================
    /// 启用 Kubernetes 服务发现
    #[arg(
        long,
        default_value_t = false,
        help_heading = "Service Discovery (Kubernetes)"
    )]
    service_discovery: bool,

    /// Kubernetes 服务发现的标签选择器（格式：key=value）
    #[arg(long, num_args = 0.., help_heading = "Service Discovery (Kubernetes)")]
    selector: Vec<String>,

    /// 发现的 worker pod 所使用的端口
    #[arg(
        long,
        default_value_t = 80,
        help_heading = "Service Discovery (Kubernetes)"
    )]
    service_discovery_port: u16,

    /// 监听 pod 的 Kubernetes 命名空间
    #[arg(long, help_heading = "Service Discovery (Kubernetes)")]
    service_discovery_namespace: Option<String>,

    /// PD 模式下 prefill 服务器 pod 的标签选择器
    #[arg(long, num_args = 0.., help_heading = "Service Discovery (Kubernetes)")]
    prefill_selector: Vec<String>,

    /// PD 模式下 decode 服务器 pod 的标签选择器
    #[arg(long, num_args = 0.., help_heading = "Service Discovery (Kubernetes)")]
    decode_selector: Vec<String>,

    // ==================== Logging ====================
    /// 日志文件存储目录
    #[arg(long, help_heading = "Logging")]
    log_dir: Option<String>,

    /// 设置日志级别
    #[arg(long, default_value = "info", value_parser = ["debug", "info", "warn", "error"], help_heading = "Logging")]
    log_level: String,

    /// 启用结构化 JSON 日志输出（替代纯文本）
    #[arg(long, default_value_t = false, help_heading = "Logging")]
    json_log: bool,

    // ==================== Prometheus Metrics ====================
    /// 暴露 Prometheus 指标的端口
    #[arg(long, default_value_t = 29000, help_heading = "Prometheus Metrics")]
    prometheus_port: u16,

    /// Prometheus 指标服务器绑定的主机地址
    #[arg(long, default_value = "0.0.0.0", help_heading = "Prometheus Metrics")]
    prometheus_host: String,

    /// Prometheus 耗时指标的自定义分桶
    #[arg(long, num_args = 0.., help_heading = "Prometheus Metrics")]
    prometheus_duration_buckets: Vec<f64>,

    // ==================== Request Handling ====================
    /// 用于检查请求 ID 的自定义 HTTP 头
    #[arg(long, num_args = 0.., help_heading = "Request Handling")]
    request_id_headers: Vec<String>,

    /// 请求超时时间（秒）
    #[arg(long, default_value_t = 1800, help_heading = "Request Handling")]
    request_timeout_secs: u64,

    /// 关闭期间等待进行中请求完成的宽限期（秒）
    #[arg(long, default_value_t = 180, help_heading = "Request Handling")]
    shutdown_grace_period_secs: u64,

    /// 最大负载大小（字节）
    #[arg(long, default_value_t = 536870912, help_heading = "Request Handling")]
    max_payload_size: usize,

    /// CORS 允许的源
    #[arg(long, num_args = 0.., help_heading = "Request Handling")]
    cors_allowed_origins: Vec<String>,

    // ==================== HTTP Client ====================
    /// 上游 HTTP 连接池中空闲连接的超时时间（秒）
    #[arg(
        long,
        env = "SMG_POOL_IDLE_TIMEOUT_SECS",
        default_value_t = DEFAULT_POOL_IDLE_TIMEOUT_SECS,
        help_heading = "HTTP Client"
    )]
    pool_idle_timeout_secs: u64,

    /// 新建上游 HTTP 连接的超时时间（秒）
    #[arg(
        long,
        env = "SMG_CONNECT_TIMEOUT_SECS",
        default_value_t = DEFAULT_CONNECT_TIMEOUT_SECS,
        help_heading = "HTTP Client"
    )]
    connect_timeout_secs: u64,

    /// 每个主机保留的最大空闲上游 HTTP 连接数
    #[arg(
        long,
        env = "SMG_POOL_MAX_IDLE_PER_HOST",
        default_value_t = DEFAULT_POOL_MAX_IDLE_PER_HOST,
        help_heading = "HTTP Client"
    )]
    pool_max_idle_per_host: usize,

    /// 上游 HTTP 连接的 TCP keepalive 空闲时间（秒）
    #[arg(
        long,
        env = "SMG_TCP_KEEPALIVE_SECS",
        default_value_t = DEFAULT_TCP_KEEPALIVE_SECS,
        help_heading = "HTTP Client"
    )]
    tcp_keepalive_secs: u64,

    // ==================== Rate Limiting ====================
    /// 最大并发请求数（-1 表示禁用）
    #[arg(long, default_value_t = -1, help_heading = "Rate Limiting")]
    max_concurrent_requests: i32,

    /// 达到限制时待处理请求的队列大小
    #[arg(long, default_value_t = 100, help_heading = "Rate Limiting")]
    queue_size: usize,

    /// 请求在队列中的最大等待时间（秒）
    #[arg(long, default_value_t = 60, help_heading = "Rate Limiting")]
    queue_timeout_secs: u64,

    /// 令牌桶补充速率（每秒令牌数）
    #[arg(long, help_heading = "Rate Limiting")]
    rate_limit_tokens_per_second: Option<i32>,

    // ==================== Retry Configuration ====================
    /// 最大重试次数
    #[arg(long, default_value_t = 5, help_heading = "Retry Configuration")]
    retry_max_retries: u32,

    /// 初始退避延迟（毫秒）
    #[arg(long, default_value_t = 50, help_heading = "Retry Configuration")]
    retry_initial_backoff_ms: u64,

    /// 最大退避延迟（毫秒）
    #[arg(long, default_value_t = 30000, help_heading = "Retry Configuration")]
    retry_max_backoff_ms: u64,

    /// 指数退避的乘数
    #[arg(long, default_value_t = 1.5, help_heading = "Retry Configuration")]
    retry_backoff_multiplier: f32,

    /// 重试延迟的抖动因子（0.0-1.0）
    #[arg(long, default_value_t = 0.2, help_heading = "Retry Configuration")]
    retry_jitter_factor: f32,

    /// 禁用自动重试
    #[arg(long, default_value_t = false, help_heading = "Retry Configuration")]
    disable_retries: bool,

    // ==================== Circuit Breaker ====================
    /// 熔断器打开前的失败次数
    #[arg(long, default_value_t = 10, help_heading = "Circuit Breaker")]
    cb_failure_threshold: u32,

    /// 半开状态下关闭熔断器所需的成功次数
    #[arg(long, default_value_t = 3, help_heading = "Circuit Breaker")]
    cb_success_threshold: u32,

    /// 尝试关闭已打开熔断器前的等待秒数
    #[arg(long, default_value_t = 60, help_heading = "Circuit Breaker")]
    cb_timeout_duration_secs: u64,

    /// 跟踪失败的滑动窗口时长
    #[arg(long, default_value_t = 120, help_heading = "Circuit Breaker")]
    cb_window_duration_secs: u64,

    /// 禁用熔断器
    #[arg(long, default_value_t = false, help_heading = "Circuit Breaker")]
    disable_circuit_breaker: bool,

    // ==================== Health Checks ====================
    /// 标记 worker 为不健康前的失败次数
    #[arg(long, default_value_t = 3, help_heading = "Health Checks")]
    health_failure_threshold: u32,

    /// 标记 worker 为健康前的成功次数
    #[arg(long, default_value_t = 2, help_heading = "Health Checks")]
    health_success_threshold: u32,

    /// 健康检查请求的超时时间（秒）
    #[arg(long, default_value_t = 5, help_heading = "Health Checks")]
    health_check_timeout_secs: u64,

    /// 健康检查之间的间隔（秒）
    #[arg(long, default_value_t = 60, help_heading = "Health Checks")]
    health_check_interval_secs: u64,

    /// 健康检查端点路径
    #[arg(long, default_value = "/health", help_heading = "Health Checks")]
    health_check_endpoint: String,

    /// 启动时禁用所有 worker 健康检查
    #[arg(long, default_value_t = false, help_heading = "Health Checks")]
    disable_health_check: bool,

    // ==================== Tokenizer ====================
    /// 加载分词器的模型路径（HuggingFace ID 或本地路径）
    #[arg(long, help_heading = "Tokenizer")]
    model_path: Option<String>,

    /// 显式指定的分词器路径（覆盖 model_path）
    #[arg(long, help_heading = "Tokenizer")]
    tokenizer_path: Option<String>,

    /// 聊天模板路径
    #[arg(long, help_heading = "Tokenizer")]
    chat_template: Option<String>,

    /// 启用 L0（精确匹配）分词器缓存
    #[arg(long, default_value_t = false, help_heading = "Tokenizer")]
    tokenizer_cache_enable_l0: bool,

    /// L0 分词器缓存的最大条目数
    #[arg(long, default_value_t = 10000, help_heading = "Tokenizer")]
    tokenizer_cache_l0_max_entries: usize,

    /// 启用 L1（前缀匹配）分词器缓存
    #[arg(long, default_value_t = false, help_heading = "Tokenizer")]
    tokenizer_cache_enable_l1: bool,

    /// L1 分词器缓存的最大内存（字节）
    #[arg(long, default_value_t = 52428800, help_heading = "Tokenizer")]
    tokenizer_cache_l1_max_memory: usize,

    // ==================== Parsers ====================
    /// 推理模型的解析器（如 deepseek-r1、qwen3）
    #[arg(long, help_heading = "Parsers")]
    reasoning_parser: Option<String>,

    /// 工具调用交互的解析器
    #[arg(long, help_heading = "Parsers")]
    tool_call_parser: Option<String>,

    /// MCP 服务器配置文件路径
    #[arg(long, help_heading = "Parsers")]
    mcp_config_path: Option<String>,

    // ==================== Backend ====================
    /// 使用的后端运行时
    #[arg(long, value_enum, default_value_t = Backend::Sglang, alias = "runtime", help_heading = "Backend")]
    backend: Backend,

    /// 历史记录存储后端
    #[arg(long, default_value = "memory", value_parser = ["memory", "none", "oracle", "postgres", "redis"], help_heading = "Backend")]
    history_backend: String,

    /// 启用 WebAssembly 支持
    #[arg(long, default_value_t = false, help_heading = "Backend")]
    enable_wasm: bool,

    // ==================== Oracle Database ====================
    /// Oracle ATP wallet 目录路径
    #[arg(long, env = "ATP_WALLET_PATH", help_heading = "Oracle Database")]
    oracle_wallet_path: Option<String>,

    /// 来自 tnsnames.ora 的 Oracle TNS 别名
    #[arg(long, env = "ATP_TNS_ALIAS", help_heading = "Oracle Database")]
    oracle_tns_alias: Option<String>,

    /// Oracle 连接描述符/DSN
    #[arg(long, env = "ATP_DSN", help_heading = "Oracle Database")]
    oracle_dsn: Option<String>,

    /// Oracle 数据库用户名
    #[arg(long, env = "ATP_USER", help_heading = "Oracle Database")]
    oracle_user: Option<String>,

    /// Oracle 数据库密码
    #[arg(long, env = "ATP_PASSWORD", help_heading = "Oracle Database")]
    oracle_password: Option<String>,

    /// Oracle 连接池最小大小
    #[arg(long, env = "ATP_POOL_MIN", help_heading = "Oracle Database")]
    oracle_pool_min: Option<usize>,

    /// Oracle 连接池最大大小
    #[arg(long, env = "ATP_POOL_MAX", help_heading = "Oracle Database")]
    oracle_pool_max: Option<usize>,

    /// Oracle 连接池超时时间（秒）
    #[arg(long, env = "ATP_POOL_TIMEOUT_SECS", help_heading = "Oracle Database")]
    oracle_pool_timeout_secs: Option<u64>,

    // ==================== PostgreSQL Database ====================
    /// PostgreSQL 数据库连接 URL
    #[arg(long, help_heading = "PostgreSQL Database")]
    postgres_db_url: Option<String>,

    /// PostgreSQL 连接池最大大小
    #[arg(long, help_heading = "PostgreSQL Database")]
    postgres_pool_max_size: Option<usize>,

    // ==================== Redis Database ====================
    /// Redis 连接 URL
    #[arg(long, help_heading = "Redis Database")]
    redis_url: Option<String>,

    /// Redis 连接池最大大小
    #[arg(long, help_heading = "Redis Database")]
    redis_pool_max_size: Option<usize>,

    /// Redis 数据保留天数（-1 表示永久保留，默认 30）
    #[arg(long, help_heading = "Redis Database")]
    redis_retention_days: Option<i64>,

    // ==================== TLS/mTLS Security ====================
    /// 服务端 TLS 证书路径（PEM 格式）
    #[arg(long, help_heading = "TLS/mTLS Security")]
    tls_cert_path: Option<String>,

    /// 服务端 TLS 私钥路径（PEM 格式）
    #[arg(long, help_heading = "TLS/mTLS Security")]
    tls_key_path: Option<String>,

    // ==================== Tracing (OpenTelemetry) ====================
    /// 启用 OpenTelemetry 链路追踪
    #[arg(
        long,
        default_value_t = false,
        help_heading = "Tracing (OpenTelemetry)"
    )]
    enable_trace: bool,

    /// OTLP 采集器端点（格式：host:port）
    #[arg(
        long,
        default_value = "localhost:4317",
        help_heading = "Tracing (OpenTelemetry)"
    )]
    otlp_traces_endpoint: String,

    // ==================== Control Plane Authentication ====================
    /// worker 授权的 API 密钥
    #[arg(long, help_heading = "Control Plane Authentication")]
    api_key: Option<String>,

    /// OIDC 认证的 JWT 签发方（issuer）URL
    #[arg(
        long,
        env = "JWT_ISSUER",
        help_heading = "Control Plane Authentication"
    )]
    jwt_issuer: Option<String>,

    /// 期望的 JWT 受众（audience）声明
    #[arg(
        long,
        env = "JWT_AUDIENCE",
        help_heading = "Control Plane Authentication"
    )]
    jwt_audience: Option<String>,

    /// 显式指定的 JWKS URI（未设置时从签发方发现）
    #[arg(
        long,
        env = "JWT_JWKS_URI",
        help_heading = "Control Plane Authentication"
    )]
    jwt_jwks_uri: Option<String>,

    /// 包含角色的 JWT 声明名称
    #[arg(
        long,
        default_value = "roles",
        help_heading = "Control Plane Authentication"
    )]
    jwt_role_claim: String,

    /// 从 IDP 到网关角色的映射（格式：idp_role=gateway_role）
    #[arg(long, action = ArgAction::Append, help_heading = "Control Plane Authentication")]
    jwt_role_mapping: Vec<String>,

    /// 控制面访问的 API 密钥（格式：id:name:role:key）
    #[arg(long = "control-plane-api-keys", action = ArgAction::Append, env = "CONTROL_PLANE_API_KEYS", help_heading = "Control Plane Authentication")]
    control_plane_api_keys: Vec<String>,

    /// 禁用控制面操作的审计日志
    #[arg(
        long,
        default_value_t = false,
        help_heading = "Control Plane Authentication"
    )]
    disable_audit_logging: bool,

    // ==================== Mesh Server ====================
    #[arg(long, default_value_t = false)]
    enable_mesh: bool,

    #[arg(long)]
    mesh_server_name: Option<String>,

    #[arg(long, default_value = "0.0.0.0")]
    mesh_host: String,

    #[arg(long, default_value_t = 39527)]
    mesh_port: u16,

    #[arg(long, num_args = 0..)]
    mesh_peer_urls: Vec<String>,
}

enum OracleConnectSource {
    Dsn { descriptor: String },
    Wallet { path: String, alias: String },
}

/// Parse role mapping from CLI format "idp_role=gateway_role"
fn parse_role_mapping(mapping: &str) -> Option<(String, Role)> {
    let parts: Vec<&str> = mapping.splitn(2, '=').collect();
    if parts.len() != 2 {
        eprintln!(
            "WARNING: Invalid role mapping format '{}'. Expected 'idp_role=gateway_role'",
            mapping
        );
        return None;
    }
    let idp_role = parts[0].to_string();
    let gateway_role = match parts[1].to_lowercase().as_str() {
        "admin" => Role::Admin,
        "user" => Role::User,
        other => {
            eprintln!(
                "WARNING: Invalid gateway role '{}' in mapping. Valid roles: admin, user",
                other
            );
            return None;
        }
    };
    Some((idp_role, gateway_role))
}

/// Parse control plane API key from CLI format "id:name:role:key"
fn parse_control_plane_api_key(key_str: &str) -> Option<ApiKeyEntry> {
    let parts: Vec<&str> = key_str.splitn(4, ':').collect();
    if parts.len() != 4 {
        eprintln!(
            "WARNING: Invalid control-plane-api-key format '{}'. Expected 'id:name:role:key'",
            key_str
        );
        return None;
    }
    let id = parts[0];
    let name = parts[1];
    let role_str = parts[2];
    let key = parts[3];

    let role = match role_str.to_lowercase().as_str() {
        "admin" => Role::Admin,
        "user" => Role::User,
        other => {
            eprintln!(
                "WARNING: Invalid role '{}' in control-plane-api-key. Valid roles: admin, user",
                other
            );
            return None;
        }
    };

    Some(ApiKeyEntry::new(id, name, key, role))
}

impl CliArgs {
    /// Build control plane authentication configuration from CLI args.
    fn build_control_plane_auth_config(&self) -> ControlPlaneAuthConfig {
        // Build JWT config if issuer and audience are provided
        let jwt = match (&self.jwt_issuer, &self.jwt_audience) {
            (Some(issuer), Some(audience)) => {
                let role_mapping: HashMap<String, Role> = self
                    .jwt_role_mapping
                    .iter()
                    .filter_map(|m| parse_role_mapping(m))
                    .collect();

                let mut jwt_config = JwtConfig::new(issuer.clone(), audience.clone());
                jwt_config.role_claim = self.jwt_role_claim.clone();
                jwt_config.role_mapping = role_mapping;
                if let Some(jwks_uri) = &self.jwt_jwks_uri {
                    jwt_config.jwks_uri = Some(jwks_uri.clone());
                }
                Some(jwt_config)
            }
            (Some(_), None) => {
                eprintln!("WARNING: --jwt-issuer provided but --jwt-audience is missing. JWT auth disabled.");
                None
            }
            (None, Some(_)) => {
                eprintln!("WARNING: --jwt-audience provided but --jwt-issuer is missing. JWT auth disabled.");
                None
            }
            (None, None) => None,
        };

        // Build API keys from CLI args
        let api_keys: Vec<ApiKeyEntry> = self
            .control_plane_api_keys
            .iter()
            .filter_map(|k| parse_control_plane_api_key(k))
            .collect();

        ControlPlaneAuthConfig {
            jwt,
            api_keys,
            audit_enabled: !self.disable_audit_logging,
        }
    }

    fn determine_connection_mode(worker_urls: &[String]) -> ConnectionMode {
        for url in worker_urls {
            if url.starts_with("grpc://") || url.starts_with("grpcs://") {
                return ConnectionMode::Grpc { port: None };
            }
        }
        ConnectionMode::Http
    }

    fn parse_selector(selector_list: &[String]) -> HashMap<String, String> {
        let mut map = HashMap::new();
        for item in selector_list {
            if let Some(eq_pos) = item.find('=') {
                let key = item[..eq_pos].to_string();
                let value = item[eq_pos + 1..].to_string();
                map.insert(key, value);
            }
        }
        map
    }

    fn parse_policy(&self, policy_str: &str) -> PolicyConfig {
        match policy_str {
            "random" => PolicyConfig::Random,
            "round_robin" => PolicyConfig::RoundRobin,
            "cache_aware" => PolicyConfig::CacheAware {
                cache_threshold: self.cache_threshold,
                balance_abs_threshold: self.balance_abs_threshold,
                balance_rel_threshold: self.balance_rel_threshold,
                eviction_interval_secs: self.eviction_interval,
                max_tree_size: self.max_tree_size,
            },
            "power_of_two" => PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 5,
            },
            "prefix_hash" => PolicyConfig::PrefixHash {
                prefix_token_count: self.prefix_token_count,
                load_factor: self.prefix_hash_load_factor,
            },
            "manual" => PolicyConfig::Manual {
                eviction_interval_secs: self.eviction_interval,
                max_idle_secs: self.max_idle_secs,
                assignment_mode: match self.assignment_mode.as_str() {
                    "random" => ManualAssignmentMode::Random,
                    "min_load" => ManualAssignmentMode::MinLoad,
                    "min_group" => ManualAssignmentMode::MinGroup,
                    other => panic!("Unknown assignment mode: {}", other),
                },
            },
            _ => PolicyConfig::RoundRobin,
        }
    }

    fn resolve_oracle_connect_details(&self) -> ConfigResult<OracleConnectSource> {
        if let Some(dsn) = self.oracle_dsn.clone() {
            return Ok(OracleConnectSource::Dsn { descriptor: dsn });
        }

        let wallet_path = self
            .oracle_wallet_path
            .clone()
            .ok_or(ConfigError::MissingRequired {
                field: "oracle_wallet_path or ATP_WALLET_PATH".to_string(),
            })?;

        let tns_alias = self
            .oracle_tns_alias
            .clone()
            .ok_or(ConfigError::MissingRequired {
                field: "oracle_tns_alias or ATP_TNS_ALIAS".to_string(),
            })?;

        Ok(OracleConnectSource::Wallet {
            path: wallet_path,
            alias: tns_alias,
        })
    }

    fn build_oracle_config(&self) -> ConfigResult<OracleConfig> {
        let (wallet_path, connect_descriptor) = match self.resolve_oracle_connect_details()? {
            OracleConnectSource::Dsn { descriptor } => (None, descriptor),
            OracleConnectSource::Wallet { path, alias } => (Some(path), alias),
        };
        let username = self
            .oracle_user
            .clone()
            .ok_or(ConfigError::MissingRequired {
                field: "oracle_user or ATP_USER".to_string(),
            })?;
        let password = self
            .oracle_password
            .clone()
            .ok_or(ConfigError::MissingRequired {
                field: "oracle_password or ATP_PASSWORD".to_string(),
            })?;

        let pool_min = self
            .oracle_pool_min
            .unwrap_or_else(OracleConfig::default_pool_min);
        let pool_max = self
            .oracle_pool_max
            .unwrap_or_else(OracleConfig::default_pool_max);

        if pool_min == 0 {
            return Err(ConfigError::InvalidValue {
                field: "oracle_pool_min".to_string(),
                value: pool_min.to_string(),
                reason: "pool minimum must be at least 1".to_string(),
            });
        }

        if pool_max < pool_min {
            return Err(ConfigError::InvalidValue {
                field: "oracle_pool_max".to_string(),
                value: pool_max.to_string(),
                reason: "pool maximum must be greater than or equal to minimum".to_string(),
            });
        }

        let pool_timeout_secs = self
            .oracle_pool_timeout_secs
            .unwrap_or_else(OracleConfig::default_pool_timeout_secs);

        Ok(OracleConfig {
            wallet_path,
            connect_descriptor,
            username,
            password,
            pool_min,
            pool_max,
            pool_timeout_secs,
        })
    }

    fn build_postgres_config(&self) -> ConfigResult<PostgresConfig> {
        let db_url = self.postgres_db_url.clone().unwrap_or_default();
        let pool_max = self
            .postgres_pool_max_size
            .unwrap_or_else(PostgresConfig::default_pool_max);
        let pcf = PostgresConfig { db_url, pool_max };
        pcf.validate().map_err(|e| ConfigError::ValidationFailed {
            reason: e.to_string(),
        })?;
        Ok(pcf)
    }

    fn build_redis_config(&self) -> ConfigResult<RedisConfig> {
        let url = self.redis_url.clone().unwrap_or_default();
        let pool_max = self.redis_pool_max_size.unwrap_or(16);

        let retention_days = match self.redis_retention_days {
            Some(d) if d < 0 => None, // Persistent
            Some(d) => Some(d as u64),
            None => Some(30), // Default 30 days
        };

        let rcf = RedisConfig {
            url,
            pool_max,
            retention_days,
        };
        rcf.validate().map_err(|e| ConfigError::ValidationFailed {
            reason: e.to_string(),
        })?;
        Ok(rcf)
    }

    fn to_router_config(
        &self,
        prefill_urls: Vec<(String, Option<u16>)>,
    ) -> ConfigResult<RouterConfig> {
        // Determine routing mode based on backend type and PD disaggregation flag
        // IGW mode doesn't change routing mode, only affects router initialization
        let mode = if matches!(self.backend, Backend::Openai) {
            RoutingMode::OpenAI {
                worker_urls: self.worker_urls.clone(),
            }
        } else if self.pd_disaggregation {
            RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls: self.decode.clone(),
                prefill_policy: self.prefill_policy.as_ref().map(|p| self.parse_policy(p)),
                decode_policy: self.decode_policy.as_ref().map(|p| self.parse_policy(p)),
            }
        } else {
            RoutingMode::Regular {
                worker_urls: self.worker_urls.clone(),
            }
        };

        let policy = self.parse_policy(&self.policy);

        let discovery = if self.service_discovery {
            Some(DiscoveryConfig {
                enabled: true,
                namespace: self.service_discovery_namespace.clone(),
                port: self.service_discovery_port,
                check_interval_secs: 60,
                selector: Self::parse_selector(&self.selector),
                prefill_selector: Self::parse_selector(&self.prefill_selector),
                decode_selector: Self::parse_selector(&self.decode_selector),
                bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
                router_selector: HashMap::new(), // Can be set via config file
                router_mesh_port_annotation: "sglang.ai/ha-port".to_string(),
            })
        } else {
            None
        };

        let metrics = Some(MetricsConfig {
            port: self.prometheus_port,
            host: self.prometheus_host.clone(),
        });

        let trace_config = Some(TraceConfig {
            enable_trace: self.enable_trace,
            otlp_traces_endpoint: self.otlp_traces_endpoint.clone(),
        });

        let mut all_urls = Vec::new();
        match &mode {
            RoutingMode::Regular { worker_urls } => {
                all_urls.extend(worker_urls.clone());
            }
            RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls,
                ..
            } => {
                for (url, _) in prefill_urls {
                    all_urls.push(url.clone());
                }
                all_urls.extend(decode_urls.clone());
            }
            RoutingMode::OpenAI { .. } => {}
        }
        let connection_mode = match &mode {
            RoutingMode::OpenAI { .. } => ConnectionMode::Http,
            _ => Self::determine_connection_mode(&all_urls),
        };

        let history_backend = match self.history_backend.as_str() {
            "none" => HistoryBackend::None,
            "oracle" => HistoryBackend::Oracle,
            "postgres" => HistoryBackend::Postgres,
            "redis" => HistoryBackend::Redis,
            _ => HistoryBackend::Memory,
        };

        let oracle = if history_backend == HistoryBackend::Oracle {
            Some(self.build_oracle_config()?)
        } else {
            None
        };
        let postgres = if history_backend == HistoryBackend::Postgres {
            Some(self.build_postgres_config()?)
        } else {
            None
        };
        let redis = if history_backend == HistoryBackend::Redis {
            Some(self.build_redis_config()?)
        } else {
            None
        };

        let builder = RouterConfig::builder()
            .mode(mode)
            .policy(policy)
            .connection_mode(connection_mode)
            .host(&self.host)
            .port(self.port)
            .max_payload_size(self.max_payload_size)
            .request_timeout_secs(self.request_timeout_secs)
            .worker_startup_timeout_secs(self.worker_startup_timeout_secs)
            .worker_startup_check_interval_secs(self.worker_startup_check_interval)
            .pool_idle_timeout_secs(self.pool_idle_timeout_secs)
            .connect_timeout_secs(self.connect_timeout_secs)
            .pool_max_idle_per_host(self.pool_max_idle_per_host)
            .tcp_keepalive_secs(self.tcp_keepalive_secs)
            .max_concurrent_requests(self.max_concurrent_requests)
            .queue_size(self.queue_size)
            .queue_timeout_secs(self.queue_timeout_secs)
            .cors_allowed_origins(self.cors_allowed_origins.clone())
            .retry_config(RetryConfig {
                max_retries: self.retry_max_retries,
                initial_backoff_ms: self.retry_initial_backoff_ms,
                max_backoff_ms: self.retry_max_backoff_ms,
                backoff_multiplier: self.retry_backoff_multiplier,
                jitter_factor: self.retry_jitter_factor,
            })
            .circuit_breaker_config(CircuitBreakerConfig {
                failure_threshold: self.cb_failure_threshold,
                success_threshold: self.cb_success_threshold,
                timeout_duration_secs: self.cb_timeout_duration_secs,
                window_duration_secs: self.cb_window_duration_secs,
            })
            .health_check_config(HealthCheckConfig {
                failure_threshold: self.health_failure_threshold,
                success_threshold: self.health_success_threshold,
                timeout_secs: self.health_check_timeout_secs,
                check_interval_secs: self.health_check_interval_secs,
                endpoint: self.health_check_endpoint.clone(),
                disable_health_check: self.disable_health_check,
            })
            .tokenizer_cache(TokenizerCacheConfig {
                enable_l0: self.tokenizer_cache_enable_l0,
                l0_max_entries: self.tokenizer_cache_l0_max_entries,
                enable_l1: self.tokenizer_cache_enable_l1,
                l1_max_memory: self.tokenizer_cache_l1_max_memory,
            })
            .history_backend(history_backend)
            .log_level(&self.log_level)
            .maybe_api_key(self.api_key.as_ref())
            .maybe_discovery(discovery)
            .maybe_metrics(metrics)
            .maybe_trace(trace_config)
            .maybe_log_dir(self.log_dir.as_ref())
            .maybe_request_id_headers(
                (!self.request_id_headers.is_empty()).then(|| self.request_id_headers.clone()),
            )
            .maybe_rate_limit_tokens_per_second(self.rate_limit_tokens_per_second)
            .maybe_model_path(self.model_path.as_ref())
            .maybe_tokenizer_path(self.tokenizer_path.as_ref())
            .maybe_chat_template(self.chat_template.as_ref())
            .maybe_oracle(oracle)
            .maybe_postgres(postgres)
            .maybe_redis(redis)
            .maybe_reasoning_parser(self.reasoning_parser.as_ref())
            .maybe_tool_call_parser(self.tool_call_parser.as_ref())
            .maybe_mcp_config_path(self.mcp_config_path.as_ref())
            .dp_aware(self.dp_aware)
            .retries(!self.disable_retries)
            .circuit_breaker(!self.disable_circuit_breaker)
            .enable_wasm(self.enable_wasm)
            .igw(self.enable_igw)
            .maybe_server_cert_and_key(self.tls_cert_path.as_ref(), self.tls_key_path.as_ref());

        builder.build()
    }

    fn to_server_config(&self, router_config: RouterConfig) -> ServerConfig {
        let service_discovery_config = if self.service_discovery {
            // Get router discovery config from router_config.discovery if available
            let (router_selector, router_mesh_port_annotation) = router_config
                .discovery
                .as_ref()
                .map(|d| {
                    (
                        d.router_selector.clone(),
                        d.router_mesh_port_annotation.clone(),
                    )
                })
                .unwrap_or_else(|| (HashMap::new(), "sglang.ai/mesh-port".to_string()));

            let selector = Self::parse_selector(&self.selector);

            let service_discovery_config = ServiceDiscoveryConfig {
                enabled: true,
                selector,
                check_interval: std::time::Duration::from_secs(60),
                port: self.service_discovery_port,
                namespace: self.service_discovery_namespace.clone(),
                pd_mode: self.pd_disaggregation,
                prefill_selector: Self::parse_selector(&self.prefill_selector),
                decode_selector: Self::parse_selector(&self.decode_selector),
                bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
                router_selector,
                router_mesh_port_annotation,
                igw_mode: self.enable_igw,
            };
            service_discovery_config.warn_if_misconfigured();
            Some(service_discovery_config)
        } else {
            None
        };

        let prometheus_config = Some(PrometheusConfig {
            port: self.prometheus_port,
            host: self.prometheus_host.clone(),
            duration_buckets: if self.prometheus_duration_buckets.is_empty() {
                None
            } else {
                Some(self.prometheus_duration_buckets.clone())
            },
        });

        // Build control plane auth config
        let control_plane_auth = {
            let config = self.build_control_plane_auth_config();
            if config.is_enabled() {
                Some(config)
            } else {
                None
            }
        };

        // ==================== Mesh Server ====================
        let mesh_server_config = if self.enable_mesh {
            let self_name = if let Some(name) = &self.mesh_server_name {
                name.to_string()
            } else {
                // If name is not set, use a random name
                let mut rng = rand::rng();
                let random_string: String =
                    (0..4).map(|_| rng.sample(Alphanumeric) as char).collect();
                format!("Mesh_{}", random_string)
            };

            let peer = self
                .mesh_peer_urls
                .first()
                .and_then(|url| url.parse::<std::net::SocketAddr>().ok());
            if let Ok(addr) =
                format!("{}:{}", self.mesh_host, self.mesh_port).parse::<std::net::SocketAddr>()
            {
                Some(MeshServerConfig {
                    self_name,
                    self_addr: addr,
                    init_peer: peer,
                })
            } else {
                tracing::warn!("Invalid mesh server address, so mesh server will not be started");
                None
            }
        } else {
            None
        };

        ServerConfig {
            host: self.host.clone(),
            port: self.port,
            router_config,
            max_payload_size: self.max_payload_size,
            log_dir: self.log_dir.clone(),
            log_level: Some(self.log_level.clone()),
            json_log: self.json_log,
            service_discovery_config,
            prometheus_config,
            request_timeout_secs: self.request_timeout_secs,
            request_id_headers: if self.request_id_headers.is_empty() {
                None
            } else {
                Some(self.request_id_headers.clone())
            },
            shutdown_grace_period_secs: self.shutdown_grace_period_secs,
            control_plane_auth,
            mesh_server_config,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check for version flags before parsing other args to avoid errors
    let args: Vec<String> = std::env::args().collect();
    for arg in &args {
        if arg == "--version" || arg == "-V" {
            println!("{}", version::get_version_string());
            return Ok(());
        }
        if arg == "--version-verbose" {
            println!("{}", version::get_verbose_version_string());
            return Ok(());
        }
    }

    let prefill_urls = parse_prefill_args();

    let mut filtered_args: Vec<String> = Vec::new();
    let raw_args: Vec<String> = std::env::args().collect();
    let mut i = 0;

    while i < raw_args.len() {
        if raw_args[i] == "--prefill" && i + 1 < raw_args.len() {
            i += 2;
            if i < raw_args.len()
                && !raw_args[i].starts_with("--")
                && (raw_args[i].parse::<u16>().is_ok() || raw_args[i].to_lowercase() == "none")
            {
                i += 1;
            }
        } else {
            filtered_args.push(raw_args[i].clone());
            i += 1;
        }
    }

    let cli = Cli::parse_from(filtered_args);

    // Handle subcommands or use direct args
    let mut cli_args = match cli.command {
        Some(Commands::Launch { args }) => args,
        None => cli.router_args,
    };

    // Automatically enable IGW mode when service discovery is turned on
    if cli_args.service_discovery && !cli_args.enable_igw {
        println!("INFO: IGW mode automatically enabled because service discovery is turned on");
        cli_args.enable_igw = true;
    }

    println!("SGLang Router starting...");
    println!("Host: {}:{}", cli_args.host, cli_args.port);
    let mode_str = if cli_args.enable_igw {
        "IGW (Inference Gateway)".to_string()
    } else if matches!(cli_args.backend, Backend::Openai) {
        "OpenAI Backend".to_string()
    } else if cli_args.pd_disaggregation {
        "PD Disaggregated".to_string()
    } else {
        format!("Regular ({})", cli_args.backend)
    };
    println!("Mode: {}", mode_str);

    match cli_args.backend {
        Backend::Vllm | Backend::Trtllm | Backend::Anthropic => {
            println!(
                "WARNING: runtime '{}' not implemented yet; falling back to regular routing. \
Provide --worker-urls or PD flags as usual.",
                cli_args.backend
            );
        }
        Backend::Sglang | Backend::Openai => {}
    }

    if !cli_args.enable_igw {
        println!("Policy: {}", cli_args.policy);

        if cli_args.pd_disaggregation && !prefill_urls.is_empty() {
            println!("Prefill nodes: {:?}", prefill_urls);
            println!("Decode nodes: {:?}", cli_args.decode);
        }
    }

    let router_config = cli_args.to_router_config(prefill_urls)?;
    router_config.validate()?;
    let server_config = cli_args.to_server_config(router_config);
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async move { server::startup(server_config).await })?;
    if is_otel_enabled() {
        shutdown_otel();
    }
    Ok(())
}
