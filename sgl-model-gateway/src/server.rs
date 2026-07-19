use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use axum::{
    extract::{Path, Query, Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use rustls::crypto::ring;
use serde::Deserialize;
use serde_json::{json, Value};
use smg_mesh::{
    rate_limit_window::RateLimitWindow, MeshServerConfig, MeshServerHandler, MeshSyncManager,
};
use tokio::{signal, spawn};
use tracing::{debug, error, info, warn, Level};
use wfaas::LoggingSubscriber;

use crate::{
    app_context::AppContext,
    config::{RouterConfig, RoutingMode},
    core::{
        job_queue::{JobQueue, JobQueueConfig},
        steps::{TokenizerConfigRequest, WorkflowEngines},
        worker::WorkerType,
        worker_manager::WorkerManager,
        Job,
    },
    middleware::{self, AuthConfig, QueuedRequest},
    observability::{
        logging::{self, LoggingConfig},
        metrics::{self, PrometheusConfig},
        otel_trace,
    },
    protocols::{
        chat::ChatCompletionRequest,
        classify::ClassifyRequest,
        completion::CompletionRequest,
        embedding::EmbeddingRequest,
        generate::GenerateRequest,
        parser::{ParseFunctionCallRequest, SeparateReasoningRequest},
        rerank::V1RerankReqInput,
        responses::{ResponsesGetParams, ResponsesRequest},
        tokenize::{AddTokenizerRequest, DetokenizeRequest, TokenizeRequest},
        validated::ValidatedJson,
        worker_spec::{WorkerConfigRequest, WorkerUpdateRequest},
    },
    routers::{
        conversations,
        mesh::{
            get_app_config, get_cluster_status, get_global_rate_limit, get_global_rate_limit_stats,
            get_mesh_health, get_policy_state, get_policy_states, get_worker_state,
            get_worker_states, set_global_rate_limit, trigger_graceful_shutdown, update_app_config,
        },
        parse,
        router_manager::RouterManager,
        tokenize, RouterTrait,
    },
    service_discovery::{start_service_discovery, ServiceDiscoveryConfig},
    tokenizer::TokenizerRegistry,
    wasm::route::{add_wasm_module, list_wasm_modules, remove_wasm_module},
};
#[derive(Clone)]
pub struct AppState {
    pub router: Arc<dyn RouterTrait>,
    pub context: Arc<AppContext>,
    pub concurrency_queue_tx: Option<tokio::sync::mpsc::Sender<QueuedRequest>>,
    pub router_manager: Option<Arc<RouterManager>>,
    pub mesh_handler: Option<Arc<MeshServerHandler>>,
    pub mesh_sync_manager: Option<Arc<MeshSyncManager>>,
}

async fn parse_function_call(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ParseFunctionCallRequest>,
) -> Response {
    parse::parse_function_call(&state.context, &req).await
}

async fn parse_reasoning(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SeparateReasoningRequest>,
) -> Response {
    parse::parse_reasoning(&state.context, &req).await
}

async fn sink_handler() -> Response {
    StatusCode::NOT_FOUND.into_response()
}

async fn liveness() -> Response {
    (StatusCode::OK, "OK").into_response()
}

async fn readiness(State(state): State<Arc<AppState>>) -> Response {
    let workers = state.context.worker_registry.get_all();
    let healthy_workers: Vec<_> = workers.iter().filter(|w| w.is_healthy()).collect();

    let is_ready = if state.context.router_config.enable_igw {
        !healthy_workers.is_empty()
    } else {
        match &state.context.router_config.mode {
            RoutingMode::PrefillDecode { .. } => {
                let has_prefill = healthy_workers
                    .iter()
                    .any(|w| matches!(w.worker_type(), WorkerType::Prefill { .. }));
                let has_decode = healthy_workers
                    .iter()
                    .any(|w| matches!(w.worker_type(), WorkerType::Decode));
                has_prefill && has_decode
            }
            RoutingMode::Regular { .. } => !healthy_workers.is_empty(),
            RoutingMode::OpenAI { .. } => !healthy_workers.is_empty(),
        }
    };

    if is_ready {
        (
            StatusCode::OK,
            Json(json!({
                "status": "ready",
                "healthy_workers": healthy_workers.len(),
                "total_workers": workers.len()
            })),
        )
            .into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "not ready",
                "reason": "insufficient healthy workers"
            })),
        )
            .into_response()
    }
}

async fn health(_state: State<Arc<AppState>>) -> Response {
    liveness().await
}

async fn health_generate(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.health_generate(req).await
}

async fn engine_metrics(State(state): State<Arc<AppState>>) -> Response {
    WorkerManager::get_engine_metrics(&state.context.worker_registry, &state.context.client)
        .await
        .into_response()
}

async fn get_server_info(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.get_server_info(req).await
}

async fn v1_models(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.get_models(req).await
}

async fn get_model_info(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.get_model_info(req).await
}

async fn generate(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    Json(body): Json<GenerateRequest>,
) -> Response {
    let model_id = body.model.as_deref();
    state
        .router
        .route_generate(Some(&headers), &body, model_id)
        .await
}

async fn v1_chat_completions(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    ValidatedJson(body): ValidatedJson<ChatCompletionRequest>,
) -> Response {
    state
        .router
        .route_chat(Some(&headers), &body, Some(&body.model))
        .await
}

async fn v1_completions(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    Json(body): Json<CompletionRequest>,
) -> Response {
    state
        .router
        .route_completion(Some(&headers), &body, Some(&body.model))
        .await
}

async fn v1_rerank(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    Json(body): Json<V1RerankReqInput>,
) -> Response {
    let rerank_body = &body.into();
    state
        .router
        .route_rerank(Some(&headers), rerank_body, Some(&rerank_body.model))
        .await
}

async fn v1_responses(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    ValidatedJson(body): ValidatedJson<ResponsesRequest>,
) -> Response {
    state
        .router
        .route_responses(Some(&headers), &body, Some(&body.model))
        .await
}

async fn v1_embeddings(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    Json(body): Json<EmbeddingRequest>,
) -> Response {
    state
        .router
        .route_embeddings(Some(&headers), &body, Some(&body.model))
        .await
}

async fn v1_classify(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    Json(body): Json<ClassifyRequest>,
) -> Response {
    state
        .router
        .route_classify(Some(&headers), &body, Some(&body.model))
        .await
}

async fn v1_responses_get(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
    headers: http::HeaderMap,
    Query(params): Query<ResponsesGetParams>,
) -> Response {
    state
        .router
        .get_response(Some(&headers), &response_id, &params)
        .await
}

async fn v1_responses_cancel(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
    headers: http::HeaderMap,
) -> Response {
    state
        .router
        .cancel_response(Some(&headers), &response_id)
        .await
}

async fn v1_responses_delete(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
    headers: http::HeaderMap,
) -> Response {
    state
        .router
        .delete_response(Some(&headers), &response_id)
        .await
}

async fn v1_responses_list_input_items(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
    headers: http::HeaderMap,
) -> Response {
    state
        .router
        .list_response_input_items(Some(&headers), &response_id)
        .await
}

async fn v1_conversations_create(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Response {
    conversations::create_conversation(&state.context.conversation_storage, body).await
}

async fn v1_conversations_get(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
) -> Response {
    conversations::get_conversation(&state.context.conversation_storage, &conversation_id).await
}

async fn v1_conversations_update(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    conversations::update_conversation(&state.context.conversation_storage, &conversation_id, body)
        .await
}

async fn v1_conversations_delete(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
) -> Response {
    conversations::delete_conversation(&state.context.conversation_storage, &conversation_id).await
}

#[derive(Deserialize, Default)]
struct ListItemsQuery {
    limit: Option<usize>,
    order: Option<String>,
    after: Option<String>,
}

async fn v1_conversations_list_items(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
    Query(ListItemsQuery {
        limit,
        order,
        after,
    }): Query<ListItemsQuery>,
) -> Response {
    conversations::list_conversation_items(
        &state.context.conversation_storage,
        &state.context.conversation_item_storage,
        &conversation_id,
        limit,
        order.as_deref(),
        after.as_deref(),
    )
    .await
}

#[derive(Deserialize, Default)]
struct GetItemQuery {
    /// Additional fields to include in response (not yet implemented)
    include: Option<Vec<String>>,
}

async fn v1_conversations_create_items(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    conversations::create_conversation_items(
        &state.context.conversation_storage,
        &state.context.conversation_item_storage,
        &conversation_id,
        body,
    )
    .await
}

async fn v1_conversations_get_item(
    State(state): State<Arc<AppState>>,
    Path((conversation_id, item_id)): Path<(String, String)>,
    Query(query): Query<GetItemQuery>,
) -> Response {
    conversations::get_conversation_item(
        &state.context.conversation_storage,
        &state.context.conversation_item_storage,
        &conversation_id,
        &item_id,
        query.include,
    )
    .await
}

async fn v1_conversations_delete_item(
    State(state): State<Arc<AppState>>,
    Path((conversation_id, item_id)): Path<(String, String)>,
) -> Response {
    conversations::delete_conversation_item(
        &state.context.conversation_storage,
        &state.context.conversation_item_storage,
        &conversation_id,
        &item_id,
    )
    .await
}

async fn flush_cache(State(state): State<Arc<AppState>>, _req: Request) -> Response {
    WorkerManager::flush_cache_all(&state.context.worker_registry, &state.context.client)
        .await
        .into_response()
}

async fn get_loads(State(state): State<Arc<AppState>>, _req: Request) -> Response {
    WorkerManager::get_all_worker_loads(&state.context.worker_registry, &state.context.client)
        .await
        .into_response()
}

async fn create_worker(
    State(state): State<Arc<AppState>>,
    Json(config): Json<WorkerConfigRequest>,
) -> Response {
    match state.context.worker_service.create_worker(config).await {
        Ok(result) => result.into_response(),
        Err(err) => err.into_response(),
    }
}

async fn list_workers_rest(State(state): State<Arc<AppState>>) -> Response {
    state.context.worker_service.list_workers().into_response()
}

async fn get_worker(
    State(state): State<Arc<AppState>>,
    Path(worker_id_raw): Path<String>,
) -> Response {
    match state.context.worker_service.get_worker(&worker_id_raw) {
        Ok(result) => result.into_response(),
        Err(err) => err.into_response(),
    }
}

async fn delete_worker(
    State(state): State<Arc<AppState>>,
    Path(worker_id_raw): Path<String>,
) -> Response {
    match state
        .context
        .worker_service
        .delete_worker(&worker_id_raw)
        .await
    {
        Ok(result) => result.into_response(),
        Err(err) => err.into_response(),
    }
}

async fn update_worker(
    State(state): State<Arc<AppState>>,
    Path(worker_id_raw): Path<String>,
    Json(update): Json<WorkerUpdateRequest>,
) -> Response {
    match state
        .context
        .worker_service
        .update_worker(&worker_id_raw, update)
        .await
    {
        Ok(result) => result.into_response(),
        Err(err) => err.into_response(),
    }
}

// ============================================================================
// Tokenize / Detokenize Handlers
// ============================================================================

async fn v1_tokenize(
    State(state): State<Arc<AppState>>,
    Json(request): Json<TokenizeRequest>,
) -> Response {
    tokenize::tokenize(&state.context.tokenizer_registry, request).await
}

async fn v1_detokenize(
    State(state): State<Arc<AppState>>,
    Json(request): Json<DetokenizeRequest>,
) -> Response {
    tokenize::detokenize(&state.context.tokenizer_registry, request).await
}

async fn v1_tokenizers_add(
    State(state): State<Arc<AppState>>,
    Json(request): Json<AddTokenizerRequest>,
) -> Response {
    tokenize::add_tokenizer(&state.context, request).await
}

async fn v1_tokenizers_list(State(state): State<Arc<AppState>>) -> Response {
    tokenize::list_tokenizers(&state.context.tokenizer_registry).await
}

async fn v1_tokenizers_get(
    State(state): State<Arc<AppState>>,
    Path(tokenizer_id): Path<String>,
) -> Response {
    tokenize::get_tokenizer_info(&state.context, &tokenizer_id).await
}

async fn v1_tokenizers_status(
    State(state): State<Arc<AppState>>,
    Path(tokenizer_id): Path<String>,
) -> Response {
    tokenize::get_tokenizer_status(&state.context, &tokenizer_id).await
}

async fn v1_tokenizers_remove(
    State(state): State<Arc<AppState>>,
    Path(tokenizer_id): Path<String>,
) -> Response {
    tokenize::remove_tokenizer(&state.context, &tokenizer_id).await
}

/// HTTP 服务器运行配置
///
/// 该结构体承载启动网关 HTTP 服务所需的顶层配置，
/// 是对 `RouterConfig` 的一层封装，额外补充了服务器监听、
/// 日志、服务发现、指标、请求处理以及优雅关闭等运行期参数。
pub struct ServerConfig {
    /// 服务器绑定的监听主机地址（如 0.0.0.0）
    pub host: String,
    /// 服务器绑定的监听端口
    pub port: u16,
    /// 路由器核心配置（路由模式、策略、后端、限流等）
    pub router_config: RouterConfig,
    /// 允许的最大请求体大小（字节）
    pub max_payload_size: usize,
    /// 日志文件输出目录（None 表示仅输出到标准输出）
    pub log_dir: Option<String>,
    /// 日志级别（如 debug/info/warn/error）
    pub log_level: Option<String>,
    /// 是否以结构化 JSON 格式输出日志（否则为纯文本）
    pub json_log: bool,
    /// 服务发现配置（如 Kubernetes 服务发现），未启用时为 None
    pub service_discovery_config: Option<ServiceDiscoveryConfig>,
    /// Prometheus 指标暴露配置，未启用时为 None
    pub prometheus_config: Option<PrometheusConfig>,
    /// 单个请求的整体超时时间（秒）
    pub request_timeout_secs: u64,
    /// 用于提取请求 ID 的自定义 HTTP 头列表
    pub request_id_headers: Option<Vec<String>>,
    /// 优雅关闭期间等待进行中请求完成的宽限期（秒）
    pub shutdown_grace_period_secs: u64,
    /// 控制面认证配置（JWT / API Key / 审计），未启用时为 None
    pub control_plane_auth: Option<crate::auth::ControlPlaneAuthConfig>,
    /// Mesh 集群互联服务配置，未启用时为 None
    pub mesh_server_config: Option<MeshServerConfig>,
}

pub fn build_app(
    app_state: Arc<AppState>,
    auth_config: AuthConfig,
    control_plane_auth_state: Option<crate::auth::ControlPlaneAuthState>,
    max_payload_size: usize,
    request_id_headers: Vec<String>,
    cors_allowed_origins: Vec<String>,
) -> Router {
    let protected_routes = Router::new()
        .route("/generate", post(generate))
        .route("/v1/chat/completions", post(v1_chat_completions))
        .route("/v1/completions", post(v1_completions))
        .route("/v1/rerank", post(v1_rerank))
        .route("/v1/responses", post(v1_responses))
        .route("/v1/embeddings", post(v1_embeddings))
        .route("/v1/classify", post(v1_classify))
        .route("/v1/responses/{response_id}", get(v1_responses_get))
        .route(
            "/v1/responses/{response_id}/cancel",
            post(v1_responses_cancel),
        )
        .route("/v1/responses/{response_id}", delete(v1_responses_delete))
        .route(
            "/v1/responses/{response_id}/input_items",
            get(v1_responses_list_input_items),
        )
        .route("/v1/conversations", post(v1_conversations_create))
        .route(
            "/v1/conversations/{conversation_id}",
            get(v1_conversations_get)
                .post(v1_conversations_update)
                .delete(v1_conversations_delete),
        )
        .route(
            "/v1/conversations/{conversation_id}/items",
            get(v1_conversations_list_items).post(v1_conversations_create_items),
        )
        .route(
            "/v1/conversations/{conversation_id}/items/{item_id}",
            get(v1_conversations_get_item).delete(v1_conversations_delete_item),
        )
        // Tokenize / Detokenize endpoints
        .route("/v1/tokenize", post(v1_tokenize))
        .route("/v1/detokenize", post(v1_detokenize))
        .route_layer(axum::middleware::from_fn_with_state(
            app_state.clone(),
            middleware::concurrency_limit_middleware,
        ))
        .route_layer(axum::middleware::from_fn_with_state(
            auth_config.clone(),
            middleware::auth_middleware,
        ))
        .route_layer(axum::middleware::from_fn_with_state(
            app_state.clone(),
            middleware::wasm_middleware,
        ));

    let public_routes = Router::new()
        .route("/liveness", get(liveness))
        .route("/readiness", get(readiness))
        .route("/health", get(health))
        .route("/health_generate", get(health_generate))
        .route("/engine_metrics", get(engine_metrics))
        .route("/v1/models", get(v1_models))
        .route("/model_info", get(get_model_info))
        // TODO: Remove `/get_model_info` alias after one release-cycle deprecation window.
        .route("/get_model_info", get(get_model_info))
        .route("/server_info", get(get_server_info))
        // TODO: Remove `/get_server_info` alias after one release-cycle deprecation window.
        .route("/get_server_info", get(get_server_info));

    // Build admin routes with control plane auth if configured, otherwise use simple API key auth
    let admin_routes = Router::new()
        .route("/flush_cache", post(flush_cache))
        .route("/v1/loads", get(get_loads))
        // TODO: Remove `/get_loads` alias after one release-cycle deprecation window.
        .route("/get_loads", get(get_loads))
        .route("/parse/function_call", post(parse_function_call))
        .route("/parse/reasoning", post(parse_reasoning))
        .route("/wasm", post(add_wasm_module))
        .route("/wasm/{module_uuid}", delete(remove_wasm_module))
        .route("/wasm", get(list_wasm_modules))
        // Tokenizer management endpoints
        .route(
            "/v1/tokenizers",
            post(v1_tokenizers_add).get(v1_tokenizers_list),
        )
        .route(
            "/v1/tokenizers/{tokenizer_id}",
            get(v1_tokenizers_get).delete(v1_tokenizers_remove),
        )
        .route(
            "/v1/tokenizers/{tokenizer_id}/status",
            get(v1_tokenizers_status),
        );

    // Build worker routes
    let worker_routes = Router::new()
        .route("/workers", post(create_worker).get(list_workers_rest))
        .route(
            "/workers/{worker_id}",
            get(get_worker).put(update_worker).delete(delete_worker),
        );

    // Apply authentication middleware to control plane routes
    let apply_control_plane_auth = |routes: Router<Arc<AppState>>| {
        if let Some(ref cp_state) = control_plane_auth_state {
            routes.route_layer(axum::middleware::from_fn_with_state(
                cp_state.clone(),
                crate::auth::control_plane_auth_middleware,
            ))
        } else {
            routes.route_layer(axum::middleware::from_fn_with_state(
                auth_config.clone(),
                middleware::auth_middleware,
            ))
        }
    };
    let admin_routes = apply_control_plane_auth(admin_routes);
    let worker_routes = apply_control_plane_auth(worker_routes);

    // HA management routes
    let mesh_routes = Router::new()
        .route("/ha/status", get(get_cluster_status))
        .route("/ha/health", get(get_mesh_health))
        .route("/ha/workers", get(get_worker_states))
        .route("/ha/workers/{worker_id}", get(get_worker_state))
        .route("/ha/policies", get(get_policy_states))
        .route("/ha/policies/{model_id}", get(get_policy_state))
        .route("/ha/config/{key}", get(get_app_config))
        .route("/ha/config", post(update_app_config))
        .route("/ha/rate-limit", post(set_global_rate_limit))
        .route("/ha/rate-limit", get(get_global_rate_limit))
        .route("/ha/rate-limit/stats", get(get_global_rate_limit_stats))
        .route("/ha/shutdown", post(trigger_graceful_shutdown))
        .route_layer(axum::middleware::from_fn_with_state(
            auth_config.clone(),
            middleware::auth_middleware,
        ));

    Router::new()
        .merge(protected_routes)
        .merge(public_routes)
        .merge(admin_routes)
        .merge(worker_routes)
        .merge(mesh_routes)
        .layer(axum::extract::DefaultBodyLimit::max(max_payload_size))
        .layer(tower_http::limit::RequestBodyLimitLayer::new(
            max_payload_size,
        ))
        .layer(middleware::create_logging_layer())
        .layer(middleware::HttpMetricsLayer::new(
            app_state.context.inflight_tracker.clone(),
        ))
        .layer(middleware::RequestIdLayer::new(request_id_headers))
        .layer(create_cors_layer(cors_allowed_origins))
        .fallback(sink_handler)
        .with_state(app_state)
}

/// 网关服务启动入口。
///
/// 该函数按顺序完成整个 model gateway 的初始化与启动，主要步骤包括:
/// 1. 初始化 OpenTelemetry 链路追踪与日志系统(全局仅初始化一次);
/// 2. 按需启动 Prometheus 指标服务;
/// 3. 若启用 Mesh，则构建并启动集群互联服务(状态存储、同步管理、分区探测、限流窗口重置);
/// 4. 构建 `AppContext`、任务队列 `JobQueue` 与各类工作流引擎 `WorkflowEngines`;
/// 5. 异步提交分词器加载、worker 初始化、MCP 服务初始化等后台任务(不阻塞启动);
/// 6. 构建路由管理器 `RouterManager`，并启动健康检查、负载监控、并发限流与请求队列;
/// 7. 按需启动 Kubernetes 服务发现;
/// 8. 构建 Axum 应用并绑定监听地址，支持 TLS/非 TLS，注册优雅关闭信号后开始对外服务。
///
/// # 参数
/// - `config`: 服务器运行配置，见 [`ServerConfig`]。
///
/// # 返回
/// 正常关闭返回 `Ok(())`；初始化或监听失败时返回对应错误。
pub async fn startup(config: ServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    // 全局标记:确保日志系统在整个进程生命周期内只初始化一次
    static LOGGING_INITIALIZED: AtomicBool = AtomicBool::new(false);

    if let Some(trace_config) = &config.router_config.trace_config {
        otel_trace::otel_tracing_init(
            trace_config.enable_trace,
            Some(&trace_config.otlp_traces_endpoint),
        )?;
    }

    let _log_guard = if !LOGGING_INITIALIZED.swap(true, Ordering::SeqCst) {
        Some(logging::init_logging(
            LoggingConfig {
                level: config
                    .log_level
                    .as_deref()
                    .and_then(|s| match s.to_uppercase().parse::<Level>() {
                        Ok(l) => Some(l),
                        Err(_) => {
                            warn!("Invalid log level string: '{s}'. Defaulting to INFO.");
                            None
                        }
                    })
                    .unwrap_or(Level::INFO),
                json_format: config.json_log,
                log_dir: config.log_dir.clone(),
                colorize: true,
                log_file_name: "smg".to_string(),
                log_targets: None,
            },
            config.router_config.trace_config.clone(),
        ))
    } else {
        None
    };

    if let Some(prometheus_config) = &config.prometheus_config {
        metrics::start_prometheus(prometheus_config.clone());
    }

    let (mesh_handler, mesh_sync_manager) = if let Some(mesh_server_config) =
        &config.mesh_server_config
    {
        // 先创建带状态存储的高可用(HA)同步管理器
        use smg_mesh::{partition::PartitionDetector, stores::StateStores, sync::MeshSyncManager};
        let stores = Arc::new(StateStores::with_self_name(
            mesh_server_config.self_name.clone(),
        ));
        let sync_manager = Arc::new(MeshSyncManager::new(
            stores.clone(),
            mesh_server_config.self_name.clone(),
        ));

        // 创建网络分区探测器
        let partition_detector = Arc::new(PartitionDetector::default());

        // 使用当前成员列表初始化限流哈希环
        sync_manager.update_rate_limit_membership();

        // 启动限流窗口重置任务
        let window_manager = RateLimitWindow::new(sync_manager.clone(), 1); // 每 1 秒重置一次
        spawn(async move {
            window_manager.start_reset_task().await;
        });

        // 创建 mesh 服务构建器，并带上状态存储进行构建
        use smg_mesh::service::MeshServerBuilder;
        let builder = MeshServerBuilder::new(
            mesh_server_config.self_name.clone(),
            mesh_server_config.self_addr,
            mesh_server_config.init_peer,
        );
        let (mesh_server, handler) = builder.build_with_stores(Some(stores.clone()));

        // 带上状态存储与分区探测器启动 mesh 服务
        let stores_for_server = stores.clone();
        let sync_manager_for_server = sync_manager.clone();
        let partition_detector_for_server = partition_detector.clone();
        spawn(async move {
            if let Err(e) = mesh_server
                .start_serve_with_stores(
                    Some(stores_for_server),
                    Some(sync_manager_for_server),
                    Some(partition_detector_for_server),
                )
                .await
            {
                tracing::error!("Mesh server failed: {}", e);
            }
        });

        (Some(Arc::new(handler)), Some(sync_manager))
    } else {
        (None, None)
    };

    info!(
        "Starting router on {}:{} | mode: {:?} | policy: {:?} | max_payload: {}MB",
        config.host,
        config.port,
        config.router_config.mode,
        config.router_config.policy,
        config.max_payload_size / (1024 * 1024)
    );

    let app_context = Arc::new(
        AppContext::from_config(config.router_config.clone(), config.request_timeout_secs).await?,
    );

    if config.prometheus_config.is_some() {
        app_context.inflight_tracker.start_sampler(20);
    }

    let weak_context = Arc::downgrade(&app_context);
    let worker_job_queue = JobQueue::new(JobQueueConfig::default(), weak_context);
    app_context
        .worker_job_queue
        .set(worker_job_queue)
        .expect("JobQueue should only be initialized once");

    // 初始化各类型的工作流引擎
    let engines = WorkflowEngines::new(&config.router_config);

    // 为所有工作流引擎订阅日志
    engines.subscribe_all(Arc::new(LoggingSubscriber)).await;

    app_context
        .workflow_engines
        .set(engines)
        .expect("WorkflowEngines should only be initialized once");
    debug!(
        "Workflow engines initialized (health check timeout: {}s)",
        config.router_config.health_check.timeout_secs
    );

    // 如果配置了分词器路径，则提交启动阶段的分词器加载任务
    // 该任务在 worker 初始化之前执行，以确保分词器可用
    if let Some(tokenizer_source) = config
        .router_config
        .tokenizer_path
        .as_ref()
        .or(config.router_config.model_path.as_ref())
    {
        info!("Loading startup tokenizer from: {}", tokenizer_source);

        let job_queue = app_context
            .worker_job_queue
            .get()
            .expect("JobQueue should be initialized");

        let tokenizer_config = TokenizerConfigRequest {
            id: TokenizerRegistry::generate_id(),
            name: tokenizer_source.clone(),
            source: tokenizer_source.clone(),
            chat_template_path: config.router_config.chat_template.clone(),
            cache_config: config.router_config.tokenizer_cache.to_option(),
            fail_on_duplicate: false,
        };

        let job = Job::AddTokenizer {
            config: Box::new(tokenizer_config),
        };

        job_queue
            .submit(job)
            .await
            .map_err(|e| format!("Failed to submit startup tokenizer job: {}", e))?;

        info!("Startup tokenizer job submitted (will complete in background)");
    }

    info!(
        "Initializing workers for routing mode: {:?}",
        config.router_config.mode
    );

    // 向任务队列提交 worker 初始化任务
    let job_queue = app_context
        .worker_job_queue
        .get()
        .expect("JobQueue should be initialized");
    let job = Job::InitializeWorkersFromConfig {
        router_config: Box::new(config.router_config.clone()),
    };
    job_queue
        .submit(job)
        .await
        .map_err(|e| format!("Failed to submit worker initialization job: {}", e))?;

    info!("Worker initialization job submitted (will complete in background)");

    if let Some(mcp_config) = &config.router_config.mcp_config {
        info!("Found {} MCP server(s) in config", mcp_config.servers.len());
        let mcp_job = Job::InitializeMcpServers {
            mcp_config: Box::new(mcp_config.clone()),
        };
        job_queue
            .submit(mcp_job)
            .await
            .map_err(|e| format!("Failed to submit MCP initialization job: {}", e))?;
    } else {
        info!("No MCP config provided, skipping MCP server initialization");
    }

    // 为所有 MCP 服务(静态配置 + LRU 缓存中的动态)启动后台刷新
    if let Some(mcp_manager) = app_context.mcp_manager.get() {
        let refresh_interval = Duration::from_secs(600); // 10 分钟
        let _refresh_handle =
            Arc::clone(mcp_manager).spawn_background_refresh_all(refresh_interval);
        debug!("Started background refresh for all MCP servers (every 10 minutes)");
    }

    let worker_stats = app_context.worker_registry.stats();
    info!(
        "Workers initialized: {} total, {} healthy",
        worker_stats.total_workers, worker_stats.healthy_workers
    );

    let router_manager = RouterManager::from_config(&config, &app_context).await?;
    let router: Arc<dyn RouterTrait> = router_manager.clone();

    if !config.router_config.health_check.disable_health_check {
        let _health_checker = app_context
            .worker_registry
            .start_health_checker(config.router_config.health_check.check_interval_secs);
        debug!(
            "Started health checker for workers with {}s interval",
            config.router_config.health_check.check_interval_secs
        );
    } else {
        info!("Global health checks disabled via CLI/config; skipping health checker");
    }

    if let Some(ref load_monitor) = app_context.load_monitor {
        load_monitor.start().await;
        debug!("Started LoadMonitor for PowerOfTwo policies");
    }

    let (limiter, processor) = middleware::ConcurrencyLimiter::new(
        app_context.rate_limiter.clone(),
        config.router_config.queue_size,
        Duration::from_secs(config.router_config.queue_timeout_secs),
    );

    if app_context.rate_limiter.is_none() {
        info!("Rate limiting is disabled (max_concurrent_requests = -1)");
    }

    match processor {
        Some(proc) => {
            spawn(proc.run());
            debug!(
                "Started request queue (size: {}, timeout: {}s)",
                config.router_config.queue_size, config.router_config.queue_timeout_secs
            );
        }
        None => {
            debug!(
                "Rate limiting enabled (max_concurrent_requests = {}, queue disabled)",
                config.router_config.max_concurrent_requests
            );
        }
    }

    // 若启用了 mesh，则将 mesh 同步管理器设置到 worker 注册表和策略注册表
    // 这样启用 mesh 时这些组件可在各 mesh 节点间同步状态；
    // 未启用 mesh 时它们则独立工作、互不影响。
    // 使用线程安全的 set_mesh_sync 方法，可作用于被 Arc 包裹的注册表
    if let Some(ref sync_manager) = mesh_sync_manager {
        app_context
            .worker_registry
            .set_mesh_sync(Some(sync_manager.clone()));
        info!("Mesh sync manager set on worker registry");

        app_context
            .policy_registry
            .set_mesh_sync(Some(sync_manager.clone()));
        info!("Mesh sync manager set on policy registry");
    }

    // 在将 mesh_handler 移入 app_state 之前，先取出 mesh 集群状态与端口
    let mesh_cluster_state = mesh_handler.as_ref().map(|h| h.state.clone());
    let mesh_port = config
        .mesh_server_config
        .as_ref()
        .map(|c| c.self_addr.port());

    let app_state = Arc::new(AppState {
        router,
        context: app_context.clone(),
        concurrency_queue_tx: limiter.queue_tx.clone(),
        router_manager: Some(router_manager),
        mesh_handler,
        mesh_sync_manager,
    });
    if let Some(service_discovery_config) = config.service_discovery_config {
        if service_discovery_config.enabled {
            let app_context_arc = Arc::clone(&app_state.context);

            match start_service_discovery(
                service_discovery_config,
                app_context_arc,
                mesh_cluster_state,
                mesh_port,
            )
            .await
            {
                Ok(handle) => {
                    info!("Service discovery started");
                    spawn(async move {
                        if let Err(e) = handle.await {
                            error!("Service discovery task failed: {:?}", e);
                        }
                    });
                }
                Err(e) => {
                    error!("Failed to start service discovery: {e}");
                    warn!("Continuing without service discovery");
                }
            }
        }
    }

    info!(
        "Router ready | workers: {:?}",
        WorkerManager::get_worker_urls(&app_state.context.worker_registry)
    );

    let request_id_headers = config.request_id_headers.clone().unwrap_or_else(|| {
        vec![
            "x-request-id".to_string(),
            "x-correlation-id".to_string(),
            "x-trace-id".to_string(),
            "request-id".to_string(),
        ]
    });

    let auth_config = AuthConfig {
        api_key: config.router_config.api_key.clone(),
    };

    // 若已配置，则初始化控制面认证
    let control_plane_auth_state =
        crate::auth::ControlPlaneAuthState::try_init(config.control_plane_auth.as_ref()).await;

    let app = build_app(
        app_state,
        auth_config,
        control_plane_auth_state,
        config.max_payload_size,
        request_id_headers,
        config.router_config.cors_allowed_origins.clone(),
    );

    // TcpListener::bind 接受 &str，并通过 ToSocketAddrs 自动处理 IPv4/IPv6
    let bind_addr = format!("{}:{}", config.host, config.port);
    info!("Starting server on {}", bind_addr);

    // 解析监听地址并设置优雅关闭(TLS 与非 TLS 场景通用)
    let addr: std::net::SocketAddr = bind_addr
        .parse()
        .map_err(|e| format!("Invalid address: {}", e))?;

    let handle = axum_server::Handle::new();
    let handle_clone = handle.clone();
    let grace_period = Duration::from_secs(config.shutdown_grace_period_secs);
    spawn(async move {
        shutdown_signal().await;
        handle_clone.graceful_shutdown(Some(grace_period));
    });

    if let (Some(cert), Some(key)) = (
        &config.router_config.server_cert,
        &config.router_config.server_key,
    ) {
        info!("TLS enabled");
        ring::default_provider()
            .install_default()
            .map_err(|e| format!("Failed to install rustls ring provider: {e:?}"))?;

        let tls_config = axum_server::tls_rustls::RustlsConfig::from_pem(cert.clone(), key.clone())
            .await
            .map_err(|e| format!("Failed to create TLS config: {}", e))?;

        axum_server::bind_rustls(addr, tls_config)
            .handle(handle)
            .serve(app.into_make_service())
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
    } else {
        axum_server::bind(addr)
            .handle(handle)
            .serve(app.into_make_service())
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
    }

    // HA handler 的关闭由 mesh_run! 宏中的信号处理
    // 此处无需手动关闭

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C, starting graceful shutdown");
        },
        _ = terminate => {
            info!("Received terminate signal, starting graceful shutdown");
        },
    }
}

fn create_cors_layer(allowed_origins: Vec<String>) -> tower_http::cors::CorsLayer {
    use tower_http::cors::Any;

    let cors = if allowed_origins.is_empty() {
        tower_http::cors::CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
            .expose_headers(Any)
    } else {
        let origins: Vec<http::HeaderValue> = allowed_origins
            .into_iter()
            .filter_map(|origin| origin.parse().ok())
            .collect();

        tower_http::cors::CorsLayer::new()
            .allow_origin(origins)
            .allow_methods([http::Method::GET, http::Method::POST, http::Method::OPTIONS])
            .allow_headers([http::header::CONTENT_TYPE, http::header::AUTHORIZATION])
            .expose_headers([http::header::HeaderName::from_static("x-request-id")])
    };

    cors.max_age(Duration::from_secs(3600))
}
