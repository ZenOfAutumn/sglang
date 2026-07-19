use std::{
    sync::{Arc, OnceLock},
    time::Duration,
};

use data_connector::{
    create_storage, ConversationItemStorage, ConversationStorage, ResponseStorage,
    StorageFactoryConfig,
};
use reqwest::Client;
use smg_mcp::McpManager;
use tracing::debug;

use crate::{
    config::RouterConfig,
    core::{steps::WorkflowEngines, JobQueue, LoadMonitor, WorkerRegistry, WorkerService},
    middleware::TokenBucket,
    observability::inflight_tracker::InFlightRequestTracker,
    policies::PolicyRegistry,
    reasoning_parser::ParserFactory as ReasoningParserFactory,
    routers::router_manager::RouterManager,
    tokenizer::registry::TokenizerRegistry,
    tool_parser::ParserFactory as ToolParserFactory,
    wasm::{config::WasmRuntimeConfig, module_manager::WasmModuleManager},
};

/// Error type for AppContext builder
#[derive(Debug)]
pub struct AppContextBuildError(&'static str);

impl std::fmt::Display for AppContextBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Missing required field: {}", self.0)
    }
}

impl std::error::Error for AppContextBuildError {}

/// 应用全局上下文,聚合 Router 运行所需的全部共享组件。
///
/// 相当于整个网关的「依赖注入容器」:HTTP 请求处理器、后台任务、各路由器都从这里
/// 拿到 client、注册表、存储、解析器等依赖。
///
/// - 实现 `Clone`,但字段几乎都用 `Arc` 包裹,克隆只增加引用计数、共享同一底层实例(廉价)。
/// - 构造统一走 [`AppContextBuilder`],由 [`AppContext::from_config`] 从配置一次性初始化所有组件。
#[derive(Clone)]
pub struct AppContext {
    /// 共享 HTTP 客户端,向所有后端 worker 转发请求。
    /// 按配置带连接池、超时与可选 TLS/mTLS(见 [`AppContextBuilder::with_client`])。
    pub client: Client,

    /// Router 完整运行时配置(策略、端口、存储后端、TLS、各类开关等)。
    pub router_config: RouterConfig,

    /// 全局限流器(令牌桶),None 表示不限并发。
    pub rate_limiter: Option<Arc<TokenBucket>>,

    // ---- 注册表与解析器 ----
    /// 分词器注册表,按 model_id / 自定义名称管理已加载的 tokenizer,供 gRPC / IGW 模式使用。
    pub tokenizer_registry: Arc<TokenizerRegistry>,

    /// 推理内容(reasoning / thinking)解析器工厂,按需创建故为 Option。
    pub reasoning_parser_factory: Option<ReasoningParserFactory>,

    /// 工具调用(function / tool call)解析器工厂,按需创建故为 Option。
    pub tool_parser_factory: Option<ToolParserFactory>,

    /// worker 注册表,维护所有后端 worker 的地址、类型、健康状态,是路由决策的数据源。
    pub worker_registry: Arc<WorkerRegistry>,

    /// 策略注册表,持有当前生效的负载均衡策略(见 [`crate::config::PolicyConfig`])及其运行时状态。
    pub policy_registry: Arc<PolicyRegistry>,

    /// 路由器管理器,统筹多个子路由器(如 PD 分离下的 Prefill/Decode 路由)。
    /// 非 IGW / 多路由场景可能为 None。
    pub router_manager: Option<Arc<RouterManager>>,

    // ---- 持久化存储(trait 对象,可为内存 / Redis / Postgres / Oracle 等实现) ----
    /// Responses API 的持久化存储后端。
    pub response_storage: Arc<dyn ResponseStorage>,

    /// 会话(conversation)元数据存储后端。
    pub conversation_storage: Arc<dyn ConversationStorage>,

    /// 会话条目(单条消息 item)存储后端。
    pub conversation_item_storage: Arc<dyn ConversationItemStorage>,

    /// 负载监控器,周期性拉取各 worker 实时负载供负载均衡策略使用。
    pub load_monitor: Option<Arc<LoadMonitor>>,

    /// 配置中指定的 reasoning 解析器名称(未配置则为 None),用于选择具体解析器实现。
    pub configured_reasoning_parser: Option<String>,

    /// 配置中指定的 tool call 解析器名称(未配置则为 None)。
    pub configured_tool_parser: Option<String>,

    // ---- OnceLock 延迟初始化容器(构建后注入一次,规避循环依赖) ----
    /// worker 后台任务队列;JobQueue 需在 AppContext 构建后才能注入,运行期仅设置一次。
    pub worker_job_queue: Arc<OnceLock<Arc<JobQueue>>>,

    /// 工作流引擎集合(如 Responses / Chat 等工作流步骤引擎),构建后注入一次。
    pub workflow_engines: Arc<OnceLock<WorkflowEngines>>,

    /// MCP(Model Context Protocol)管理器,初始为空配置,MCP server 后续通过任务注册。
    pub mcp_manager: Arc<OnceLock<Arc<McpManager>>>,

    /// WASM 模块管理器,仅在配置启用 WASM 插件时创建。
    pub wasm_manager: Option<Arc<WasmModuleManager>>,

    /// worker 服务层,封装基于注册表与任务队列的 worker 增删改查等业务操作。
    pub worker_service: Arc<WorkerService>,

    /// 在途请求追踪器,统计当前正在处理的请求数,用于优雅关闭与可观测性。
    pub inflight_tracker: Arc<InFlightRequestTracker>,
}

impl std::fmt::Debug for AppContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppContext")
            .field("router_config", &self.router_config)
            .finish_non_exhaustive()
    }
}

pub struct AppContextBuilder {
    client: Option<Client>,
    router_config: Option<RouterConfig>,
    rate_limiter: Option<Arc<TokenBucket>>,
    tokenizer_registry: Option<Arc<TokenizerRegistry>>,
    reasoning_parser_factory: Option<ReasoningParserFactory>,
    tool_parser_factory: Option<ToolParserFactory>,
    worker_registry: Option<Arc<WorkerRegistry>>,
    policy_registry: Option<Arc<PolicyRegistry>>,
    router_manager: Option<Arc<RouterManager>>,
    response_storage: Option<Arc<dyn ResponseStorage>>,
    conversation_storage: Option<Arc<dyn ConversationStorage>>,
    conversation_item_storage: Option<Arc<dyn ConversationItemStorage>>,
    load_monitor: Option<Arc<LoadMonitor>>,
    worker_job_queue: Option<Arc<OnceLock<Arc<JobQueue>>>>,
    workflow_engines: Option<Arc<OnceLock<WorkflowEngines>>>,
    mcp_manager: Option<Arc<OnceLock<Arc<McpManager>>>>,
    wasm_manager: Option<Arc<WasmModuleManager>>,
}

impl AppContext {
    pub fn builder() -> AppContextBuilder {
        AppContextBuilder::new()
    }

    /// Create AppContext from config with all components initialized
    /// This is the main entry point that replaces ~194 lines of initialization in server.rs
    pub async fn from_config(
        router_config: RouterConfig,
        request_timeout_secs: u64,
    ) -> Result<Self, String> {
        AppContextBuilder::from_config(router_config, request_timeout_secs)
            .await?
            .build()
            .map_err(|e| e.to_string())
    }
}

impl AppContextBuilder {
    pub fn new() -> Self {
        Self {
            client: None,
            router_config: None,
            rate_limiter: None,
            tokenizer_registry: None,
            reasoning_parser_factory: None,
            tool_parser_factory: None,
            worker_registry: None,
            policy_registry: None,
            router_manager: None,
            response_storage: None,
            conversation_storage: None,
            conversation_item_storage: None,
            load_monitor: None,
            worker_job_queue: None,
            workflow_engines: None,
            mcp_manager: None,
            wasm_manager: None,
        }
    }

    pub fn client(mut self, client: Client) -> Self {
        self.client = Some(client);
        self
    }

    pub fn router_config(mut self, router_config: RouterConfig) -> Self {
        self.router_config = Some(router_config);
        self
    }

    pub fn rate_limiter(mut self, rate_limiter: Option<Arc<TokenBucket>>) -> Self {
        self.rate_limiter = rate_limiter;
        self
    }

    pub fn tokenizer_registry(mut self, tokenizer_registry: Arc<TokenizerRegistry>) -> Self {
        self.tokenizer_registry = Some(tokenizer_registry);
        self
    }

    pub fn reasoning_parser_factory(
        mut self,
        reasoning_parser_factory: Option<ReasoningParserFactory>,
    ) -> Self {
        self.reasoning_parser_factory = reasoning_parser_factory;
        self
    }

    pub fn tool_parser_factory(mut self, tool_parser_factory: Option<ToolParserFactory>) -> Self {
        self.tool_parser_factory = tool_parser_factory;
        self
    }

    pub fn worker_registry(mut self, worker_registry: Arc<WorkerRegistry>) -> Self {
        self.worker_registry = Some(worker_registry);
        self
    }

    pub fn policy_registry(mut self, policy_registry: Arc<PolicyRegistry>) -> Self {
        self.policy_registry = Some(policy_registry);
        self
    }

    pub fn router_manager(mut self, router_manager: Option<Arc<RouterManager>>) -> Self {
        self.router_manager = router_manager;
        self
    }

    pub fn response_storage(mut self, response_storage: Arc<dyn ResponseStorage>) -> Self {
        self.response_storage = Some(response_storage);
        self
    }

    pub fn conversation_storage(
        mut self,
        conversation_storage: Arc<dyn ConversationStorage>,
    ) -> Self {
        self.conversation_storage = Some(conversation_storage);
        self
    }

    pub fn conversation_item_storage(
        mut self,
        conversation_item_storage: Arc<dyn ConversationItemStorage>,
    ) -> Self {
        self.conversation_item_storage = Some(conversation_item_storage);
        self
    }

    pub fn load_monitor(mut self, load_monitor: Option<Arc<LoadMonitor>>) -> Self {
        self.load_monitor = load_monitor;
        self
    }

    pub fn worker_job_queue(mut self, worker_job_queue: Arc<OnceLock<Arc<JobQueue>>>) -> Self {
        self.worker_job_queue = Some(worker_job_queue);
        self
    }

    pub fn workflow_engines(mut self, workflow_engines: Arc<OnceLock<WorkflowEngines>>) -> Self {
        self.workflow_engines = Some(workflow_engines);
        self
    }

    pub fn mcp_manager(mut self, mcp_manager: Arc<OnceLock<Arc<McpManager>>>) -> Self {
        self.mcp_manager = Some(mcp_manager);
        self
    }

    pub fn wasm_manager(mut self, wasm_manager: Option<Arc<WasmModuleManager>>) -> Self {
        self.wasm_manager = wasm_manager;
        self
    }

    pub fn build(self) -> Result<AppContext, AppContextBuildError> {
        let router_config = self
            .router_config
            .ok_or(AppContextBuildError("router_config"))?;
        let configured_reasoning_parser = router_config.reasoning_parser.clone();
        let configured_tool_parser = router_config.tool_call_parser.clone();

        let worker_registry = self
            .worker_registry
            .ok_or(AppContextBuildError("worker_registry"))?;
        let worker_job_queue = self
            .worker_job_queue
            .ok_or(AppContextBuildError("worker_job_queue"))?;

        // Create WorkerService from the already-built components
        let worker_service = Arc::new(WorkerService::new(
            worker_registry.clone(),
            worker_job_queue.clone(),
            router_config.clone(),
        ));

        Ok(AppContext {
            client: self.client.ok_or(AppContextBuildError("client"))?,
            router_config,
            rate_limiter: self.rate_limiter,
            tokenizer_registry: self
                .tokenizer_registry
                .ok_or(AppContextBuildError("tokenizer_registry"))?,
            reasoning_parser_factory: self.reasoning_parser_factory,
            tool_parser_factory: self.tool_parser_factory,
            worker_registry,
            policy_registry: self
                .policy_registry
                .ok_or(AppContextBuildError("policy_registry"))?,
            router_manager: self.router_manager,
            response_storage: self
                .response_storage
                .ok_or(AppContextBuildError("response_storage"))?,
            conversation_storage: self
                .conversation_storage
                .ok_or(AppContextBuildError("conversation_storage"))?,
            conversation_item_storage: self
                .conversation_item_storage
                .ok_or(AppContextBuildError("conversation_item_storage"))?,
            load_monitor: self.load_monitor,
            configured_reasoning_parser,
            configured_tool_parser,
            worker_job_queue,
            workflow_engines: self
                .workflow_engines
                .ok_or(AppContextBuildError("workflow_engines"))?,
            mcp_manager: self
                .mcp_manager
                .ok_or(AppContextBuildError("mcp_manager"))?,
            wasm_manager: self.wasm_manager,
            worker_service,
            inflight_tracker: InFlightRequestTracker::new(),
        })
    }

    /// Initialize AppContext from config - creates ALL components
    /// This replaces ~194 lines of initialization logic from server.rs
    pub async fn from_config(
        router_config: RouterConfig,
        request_timeout_secs: u64,
    ) -> Result<Self, String> {
        Ok(Self::new()
            .with_client(&router_config, request_timeout_secs)?
            .maybe_rate_limiter(&router_config)
            .with_tokenizer_registry(&router_config)?
            .with_reasoning_parser_factory()
            .with_tool_parser_factory()
            .with_worker_registry()
            .with_policy_registry(&router_config)
            .with_storage(&router_config)?
            .with_load_monitor(&router_config)
            .with_worker_job_queue()
            .with_workflow_engines()
            .with_mcp_manager(&router_config)
            .await?
            .with_wasm_manager(&router_config)?
            .router_config(router_config))
    }

    /// Create HTTP client with TLS/mTLS configuration
    fn with_client(mut self, config: &RouterConfig, timeout_secs: u64) -> Result<Self, String> {
        // FIXME: Current implementation creates a single HTTP client for all workers.
        // This works well for single security domain deployments where all workers share
        // the same CA and can accept the same client certificate.
        //
        // For multi-domain deployments (e.g., different model families with different CAs),
        // this architecture needs significant refactoring:
        // 1. Move client creation into worker registration workflow (per-worker clients)
        // 2. Store client per worker in WorkerRegistry
        // 3. Update PDRouter and other routers to fetch client from worker
        // 4. Add per-worker TLS spec in WorkerConfigRequest
        //
        // Current single-domain approach is sufficient for most deployments.
        //
        // Use rustls TLS backend when TLS/mTLS is configured (client cert or CA certs provided).
        // This ensures proper PKCS#8 key format support. For plain HTTP workers, use default
        // backend to avoid unnecessary TLS initialization overhead.
        let has_tls_config = config.client_identity.is_some() || !config.ca_certificates.is_empty();

        let mut client_builder = Client::builder()
            .pool_idle_timeout(Some(Duration::from_secs(config.pool_idle_timeout_secs)))
            .pool_max_idle_per_host(config.pool_max_idle_per_host)
            .timeout(Duration::from_secs(timeout_secs))
            .connect_timeout(Duration::from_secs(config.connect_timeout_secs))
            .tcp_nodelay(true)
            .tcp_keepalive(Some(Duration::from_secs(config.tcp_keepalive_secs)));

        // Force rustls backend when TLS is configured
        if has_tls_config {
            client_builder = client_builder.use_rustls_tls();
            debug!("Using rustls TLS backend for TLS/mTLS connections");
        }

        // Configure mTLS client identity if provided (certificates already loaded during config creation)
        if let Some(identity_pem) = &config.client_identity {
            let identity = reqwest::Identity::from_pem(identity_pem)
                .map_err(|e| format!("Failed to create client identity: {}", e))?;
            client_builder = client_builder.identity(identity);
            debug!("mTLS client authentication enabled");
        }

        // Add CA certificates for verifying worker TLS (certificates already loaded during config creation)
        for ca_cert in &config.ca_certificates {
            let cert = reqwest::Certificate::from_pem(ca_cert)
                .map_err(|e| format!("Failed to add CA certificate: {}", e))?;
            client_builder = client_builder.add_root_certificate(cert);
        }
        if !config.ca_certificates.is_empty() {
            debug!(
                "Added {} CA certificate(s) for worker verification",
                config.ca_certificates.len()
            );
        }

        let client = client_builder
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        self.client = Some(client);
        Ok(self)
    }

    /// Create rate limiter based on config
    fn maybe_rate_limiter(mut self, config: &RouterConfig) -> Self {
        self.rate_limiter = match config.max_concurrent_requests {
            n if n <= 0 => None,
            n => {
                let rate_limit_tokens = config
                    .rate_limit_tokens_per_second
                    .filter(|&t| t > 0)
                    .unwrap_or(n);
                Some(Arc::new(TokenBucket::new(
                    n as usize,
                    rate_limit_tokens as usize,
                )))
            }
        };
        self
    }

    /// Create reasoning parser factory for gRPC mode or IGW mode
    fn with_reasoning_parser_factory(mut self) -> Self {
        // Initialize reasoning parser factory
        self.reasoning_parser_factory = Some(ReasoningParserFactory::new());
        self
    }

    /// Create tool parser factory for gRPC mode or IGW mode
    fn with_tool_parser_factory(mut self) -> Self {
        // Initialize tool parser factory
        self.tool_parser_factory = Some(ToolParserFactory::new());
        self
    }

    /// Create empty tokenizer registry
    ///
    /// Tokenizers are loaded via the tokenizer_registration workflow, which is triggered:
    /// - At startup (if --tokenizer-path or --model-path is provided)
    /// - When workers connect (registers under model_id)
    /// - Via POST /v1/tokenizers API (registers under user-specified name)
    ///
    /// This unified approach ensures consistent behavior (caching, validation) across all paths.
    fn with_tokenizer_registry(mut self, _config: &RouterConfig) -> Result<Self, String> {
        self.tokenizer_registry = Some(Arc::new(TokenizerRegistry::new()));
        Ok(self)
    }

    /// Create worker registry
    fn with_worker_registry(mut self) -> Self {
        self.worker_registry = Some(Arc::new(WorkerRegistry::new()));
        self
    }

    /// Create policy registry
    fn with_policy_registry(mut self, config: &RouterConfig) -> Self {
        self.policy_registry = Some(Arc::new(PolicyRegistry::new(config.policy.clone())));
        self
    }

    /// Create all storage backends using the factory function
    fn with_storage(mut self, config: &RouterConfig) -> Result<Self, String> {
        let storage_config = StorageFactoryConfig {
            backend: &config.history_backend,
            oracle: config.oracle.as_ref(),
            postgres: config.postgres.as_ref(),
            redis: config.redis.as_ref(),
        };
        let (response_storage, conversation_storage, conversation_item_storage) =
            create_storage(storage_config)?;

        self.response_storage = Some(response_storage);
        self.conversation_storage = Some(conversation_storage);
        self.conversation_item_storage = Some(conversation_item_storage);

        Ok(self)
    }

    /// Create load monitor
    fn with_load_monitor(mut self, config: &RouterConfig) -> Self {
        let client = self
            .client
            .as_ref()
            .expect("client must be set before load monitor");
        self.load_monitor = Some(Arc::new(LoadMonitor::new(
            self.worker_registry
                .as_ref()
                .expect("worker_registry must be set")
                .clone(),
            self.policy_registry
                .as_ref()
                .expect("policy_registry must be set")
                .clone(),
            client.clone(),
            config.worker_startup_check_interval_secs,
        )));
        self
    }

    /// Create worker job queue OnceLock container
    fn with_worker_job_queue(mut self) -> Self {
        self.worker_job_queue = Some(Arc::new(OnceLock::new()));
        self
    }

    /// Create workflow engines OnceLock container
    fn with_workflow_engines(mut self) -> Self {
        self.workflow_engines = Some(Arc::new(OnceLock::new()));
        self
    }

    /// Create and initialize MCP manager with empty config
    ///
    /// This initializes the MCP manager with an empty config and default settings.
    /// MCP servers will be registered later via the InitializeMcpServers job.
    async fn with_mcp_manager(mut self, _router_config: &RouterConfig) -> Result<Self, String> {
        // Create OnceLock container
        let mcp_manager_lock = Arc::new(OnceLock::new());

        // Always create with empty config and defaults
        debug!("Initializing MCP manager with empty config and default settings (5 min TTL, 100 max connections)");

        let empty_config = smg_mcp::McpConfig {
            servers: Vec::new(),
            pool: Default::default(),
            proxy: None,
            warmup: Vec::new(),
            inventory: Default::default(),
        };

        let manager = McpManager::with_defaults(empty_config)
            .await
            .map_err(|e| format!("Failed to initialize MCP manager with defaults: {}", e))?;

        // Store the initialized manager in the OnceLock
        mcp_manager_lock
            .set(Arc::new(manager))
            .map_err(|_| "Failed to set MCP manager in OnceLock".to_string())?;

        self.mcp_manager = Some(mcp_manager_lock);
        Ok(self)
    }

    /// Create wasm manager if enabled in config
    fn with_wasm_manager(mut self, config: &RouterConfig) -> Result<Self, String> {
        self.wasm_manager = if config.enable_wasm {
            Some(Arc::new(
                WasmModuleManager::new(WasmRuntimeConfig::default())
                    .map_err(|e| format!("Failed to initialize WASM module manager: {}", e))?,
            ))
        } else {
            None
        };
        Ok(self)
    }
}

impl Default for AppContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}
