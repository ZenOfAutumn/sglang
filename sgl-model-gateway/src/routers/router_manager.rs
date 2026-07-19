//! Router Manager：协调多个 Router 与 Worker 的管理器
//!
//! 根据 `enable_igw` 开关提供两种集中式管理模式：
//! - 单 Router 模式（enable_igw=false）：Router 直接持有并管理 Worker。
//! - 多 Router 模式（enable_igw=true）：由 RouterManager 统一协调所有 Router，
//!   按模型对应 Worker 的能力，在多种 Router（HTTP/gRPC、Regular/PD、OpenAI）之间选择。
//!
//! IGW（Inference Gateway，推理网关）模式的完整说明详见
//! `model_gateway_deep_analysis.md` 第 11 阶段「IGW、多模型与 Router 能力匹配」。

use std::sync::Arc;

use arc_swap::ArcSwap;
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use dashmap::DashMap;
use serde_json::Value;
use tracing::{debug, info, warn};

use crate::{
    app_context::AppContext,
    config::RoutingMode,
    core::{ConnectionMode, RuntimeType, WorkerRegistry, WorkerType},
    protocols::{
        chat::ChatCompletionRequest,
        classify::ClassifyRequest,
        completion::CompletionRequest,
        embedding::EmbeddingRequest,
        generate::GenerateRequest,
        rerank::RerankRequest,
        responses::{ResponsesGetParams, ResponsesRequest},
    },
    routers::RouterTrait,
    server::ServerConfig,
};

/// Router 实例的唯一标识。
///
/// 作为 [`RouterManager`] 内部寻址 Router 的 key,用于区分「连接模式 × 路由模式」
/// 组合出的不同 Router(如 http-regular、grpc-pd 等,见 [`router_ids`])。
///
/// 内部刻意用 `&'static str` 而非 `String`:RouterId 会在每次请求的热路径上被
/// 拷贝与比较,`'static` 常量可完全避免堆分配。派生 `Hash`/`Eq` 以便作为 DashMap 的 key。
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct RouterId(&'static str);

impl RouterId {
    /// 由一个 `'static` 字符串常量构造 RouterId(通常在 [`router_ids`] 中定义)。
    pub const fn new(id: &'static str) -> Self {
        Self(id)
    }

    /// 以字符串形式获取该 ID(用于日志与展示)。
    pub fn as_str(&self) -> &str {
        self.0
    }
}

/// 静态 Router ID 常量集合。
///
/// 使用 `&'static str` 常量以避免在请求热路径上产生堆分配。
pub mod router_ids {
    use super::RouterId;

    pub const HTTP_REGULAR: RouterId = RouterId::new("http-regular");
    pub const HTTP_PD: RouterId = RouterId::new("http-pd");
    pub const HTTP_OPENAI: RouterId = RouterId::new("http-openai");
    pub const GRPC_REGULAR: RouterId = RouterId::new("grpc-regular");
    pub const GRPC_PD: RouterId = RouterId::new("grpc-pd");
}

/// 路由管理器：多 Router 场景下的中枢。
///
/// 负责持有所有 Router 实例，并在每个请求到来时选出合适的 Router 转发。
pub struct RouterManager {
    /// Worker 注册表，用于按模型/类型查询 Worker，驱动 Router 选择。
    worker_registry: Arc<WorkerRegistry>,
    /// Router 主表：RouterId -> Router 实例。使用 DashMap 支持并发读写（写为低频注册）。
    routers: Arc<DashMap<RouterId, Arc<dyn RouterTrait>>>,
    /// Router 列表的无锁快照，供请求热路径零分配、无分片锁地遍历（读多写少）。
    routers_snapshot: ArcSwap<Vec<Arc<dyn RouterTrait>>>,
    /// 默认 Router 的 ID：单 Router 模式下恒定使用，多 Router 模式下作为兜底。
    default_router: Arc<std::sync::RwLock<Option<RouterId>>>,
    /// 是否开启 IGW（内部网关）多 Router 模式。
    enable_igw: bool,
}

impl RouterManager {
    /// 创建一个空的 RouterManager（尚未注册任何 Router）。
    /// 注意：`enable_igw` 默认为 false，真正的取值在 `from_config` 中设置。
    pub fn new(worker_registry: Arc<WorkerRegistry>) -> Self {
        Self {
            worker_registry,
            routers: Arc::new(DashMap::new()),
            routers_snapshot: ArcSwap::from_pointee(Vec::new()),
            default_router: Arc::new(std::sync::RwLock::new(None)),
            enable_igw: false, // 稍后在 from_config 中被正确设置
        }
    }

    /// 根据服务配置构建 RouterManager。
    ///
    /// - IGW 模式：预先创建并注册全部类型的 Router
    ///   （HTTP Regular、gRPC Regular、HTTP PD、gRPC PD、OpenAI），
    ///   即使某个 Router 创建失败也仅记录告警、继续初始化其余 Router。
    /// - 单 Router 模式：仅根据路由模式和连接模式创建一个 Router，并设为默认 Router。
    ///
    /// 若最终没有任何 Router 初始化成功，则返回错误。
    pub async fn from_config(
        config: &ServerConfig,
        app_context: &Arc<AppContext>,
    ) -> Result<Arc<Self>, String> {
        use crate::routers::RouterFactory;

        let mut manager = Self::new(app_context.worker_registry.clone());
        manager.enable_igw = config.router_config.enable_igw;
        let manager = Arc::new(manager);

        if config.router_config.enable_igw {
            info!("Initializing RouterManager in multi-router mode (IGW)");

            match RouterFactory::create_regular_router(app_context).await {
                Ok(http_regular) => {
                    info!("Created HTTP Regular router");
                    manager.register_router(router_ids::HTTP_REGULAR, Arc::from(http_regular));
                }
                Err(e) => {
                    warn!("Failed to create HTTP Regular router: {e}");
                }
            }

            // IGW 模式下总是创建 gRPC Regular router
            match RouterFactory::create_grpc_router(app_context).await {
                Ok(grpc_regular) => {
                    info!("Created gRPC Regular router");
                    manager.register_router(router_ids::GRPC_REGULAR, Arc::from(grpc_regular));
                }
                Err(e) => {
                    warn!("Failed to create gRPC Regular router: {e}");
                }
            }

            info!("PD disaggregation auto-enabled for IGW mode, creating PD routers");

            // 创建 HTTP PD(Prefill/Decode 分离)router
            match RouterFactory::create_pd_router(
                None,
                None,
                &config.router_config.policy,
                app_context,
            )
            .await
            {
                Ok(http_pd) => {
                    info!("Created HTTP PD router");
                    manager.register_router(router_ids::HTTP_PD, Arc::from(http_pd));
                }
                Err(e) => {
                    warn!("Failed to create HTTP PD router: {e}");
                }
            }

            // 创建 gRPC PD(Prefill/Decode 分离)router
            match RouterFactory::create_grpc_pd_router(
                None,
                None,
                &config.router_config.policy,
                app_context,
            )
            .await
            {
                Ok(grpc_pd) => {
                    info!("Created gRPC PD router");
                    manager.register_router(router_ids::GRPC_PD, Arc::from(grpc_pd));
                }
                Err(e) => {
                    warn!("Failed to create gRPC PD router: {e}");
                }
            }

            // 为外部 OpenAI 兼容后端创建 OpenAI router
            match RouterFactory::create_openai_router(app_context).await {
                Ok(openai) => {
                    info!("Created OpenAI router");
                    manager.register_router(router_ids::HTTP_OPENAI, Arc::from(openai));
                }
                Err(e) => {
                    warn!("Failed to create OpenAI router: {e}");
                }
            }

            info!(
                "RouterManager initialized with {} routers for multi-router mode",
                manager.router_count(),
            );
        } else {
            info!("Initializing RouterManager in single-router mode");

            let single_router = Arc::from(RouterFactory::create_router(app_context).await?);
            let router_id = Self::determine_router_id(
                &config.router_config.mode,
                &config.router_config.connection_mode,
            );

            info!("Created single router with ID: {}", router_id.as_str());
            manager.register_router(router_id.clone(), single_router);
            manager.set_default_router(router_id);
        }

        if manager.router_count() == 0 {
            return Err("No routers could be initialized".to_string());
        }

        Ok(manager)
    }

    /// 根据「连接模式 + 路由模式」推导单 Router 模式下应使用的 RouterId。
    /// 注意：gRPC + OpenAI 组合当前统一归到 gRPC Regular。
    pub fn determine_router_id(
        routing_mode: &RoutingMode,
        connection_mode: &ConnectionMode,
    ) -> RouterId {
        match (connection_mode, routing_mode) {
            (ConnectionMode::Http, RoutingMode::Regular { .. }) => router_ids::HTTP_REGULAR,
            (ConnectionMode::Http, RoutingMode::PrefillDecode { .. }) => router_ids::HTTP_PD,
            (ConnectionMode::Http, RoutingMode::OpenAI { .. }) => router_ids::HTTP_OPENAI,
            (ConnectionMode::Grpc { .. }, RoutingMode::Regular { .. }) => router_ids::GRPC_REGULAR,
            (ConnectionMode::Grpc { .. }, RoutingMode::PrefillDecode { .. }) => router_ids::GRPC_PD,
            (ConnectionMode::Grpc { .. }, RoutingMode::OpenAI { .. }) => router_ids::GRPC_REGULAR,
        }
    }

    /// 注册一个 Router。
    ///
    /// 除写入主表外，还会同步重建无锁快照供热路径遍历；
    /// 若当前尚无默认 Router，则将首个注册的 Router 设为默认。
    pub fn register_router(&self, id: RouterId, router: Arc<dyn RouterTrait>) {
        self.routers.insert(id.clone(), router);

        // 更新无锁快照，便于每次请求快速遍历（避免 DashMap 分片锁开销）
        let new_snapshot: Vec<_> = self.routers.iter().map(|e| e.value().clone()).collect();
        self.routers_snapshot.store(Arc::new(new_snapshot));

        let mut default_router = self
            .default_router
            .write()
            .unwrap_or_else(|e| e.into_inner());
        if default_router.is_none() {
            *default_router = Some(id.clone());
            info!("Set default router to {}", id.as_str());
        }
    }

    pub fn set_default_router(&self, id: RouterId) {
        let mut default_router = self
            .default_router
            .write()
            .unwrap_or_else(|e| e.into_inner());
        *default_router = Some(id);
    }

    pub fn router_count(&self) -> usize {
        self.routers.len()
    }

    /// 解析请求的 model_id；未指定时尝试从已注册 Worker 中推断。
    ///
    /// IGW 模式下的行为（无法解析时必须快速失败）：
    /// - 已显式提供 model_id：直接使用；
    /// - 未提供且仅存在一个模型：将其作为隐式默认模型；
    /// - 未提供且存在多个模型：返回 400，要求显式指定模型；
    /// - 无任何模型：返回 503（无可用 Worker）。
    fn resolve_model_id(&self, model_id: Option<&str>) -> Result<String, Box<Response>> {
        // 已显式提供 model_id：直接使用
        if let Some(id) = model_id {
            return Ok(id.to_string());
        }

        // 从 Worker 注册表获取所有可用模型
        let available_models = self.worker_registry.get_models();

        match available_models.len() {
            0 => Err(Box::new(
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "No models available - no workers registered",
                )
                    .into_response(),
            )),
            1 => {
                // 仅一个模型：作为隐式默认模型
                debug!(
                    "Model not specified, using implicit default: {}",
                    available_models[0]
                );
                Ok(available_models[0].clone())
            }
            _ => {
                // 存在多个模型：要求显式指定模型
                Err(Box::new(
                    (
                        StatusCode::BAD_REQUEST,
                        format!(
                            "Model must be specified. Available models: {}",
                            available_models.join(", ")
                        ),
                    )
                        .into_response(),
                ))
            }
        }
    }

    /// 为指定模型选出最合适的 Router（第一级选择的核心逻辑）。
    ///
    /// 依据该模型对应的每个 Worker 的能力打分，取分值最高者对应的 Router：
    /// 优先级 external(OpenAI) > grpc-pd > http-pd > grpc-regular > http-regular。
    /// 若匹配到的 Router 不存在，则回退到默认 Router。
    pub fn get_router_for_model(&self, model_id: &str) -> Option<Arc<dyn RouterTrait>> {
        let workers = self.worker_registry.get_by_model(model_id);

        // 依据 Worker 能力找出得分最高的 RouterId
        // 优先级：external(OpenAI) > grpc-pd > http-pd > grpc-regular > http-regular
        let best_router_id = workers
            .iter()
            .map(|w| {
                let is_pd = matches!(
                    w.worker_type(),
                    WorkerType::Prefill { .. } | WorkerType::Decode
                );
                let is_grpc = matches!(w.connection_mode(), ConnectionMode::Grpc { .. });
                let is_external = matches!(w.metadata().runtime_type, RuntimeType::External);

                if is_external {
                    // 外部 Worker 应通过 OpenAI 兼容 Router 转发
                    return (4, &router_ids::HTTP_OPENAI);
                }

                match (is_grpc, is_pd) {
                    (true, true) => (3, &router_ids::GRPC_PD),
                    (false, true) => (2, &router_ids::HTTP_PD),
                    (true, false) => (1, &router_ids::GRPC_REGULAR),
                    (false, false) => (0, &router_ids::HTTP_REGULAR),
                }
            })
            .max_by_key(|(score, _)| *score)
            .map(|(_, id)| id);

        if let Some(router_id) = best_router_id {
            if let Some(router) = self.routers.get(router_id) {
                return Some(router.clone());
            }
        }

        // 未匹配到则回退到默认 Router
        let default_router = self
            .default_router
            .read()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(ref default_id) = *default_router {
            self.routers.get(default_id).map(|r| r.clone())
        } else {
            None
        }
    }

    /// 为一次请求选择 Router（对外的第一级选择入口）。
    ///
    /// - 单 Router 模式：恒定返回默认 Router。
    /// - 多 Router 模式：
    ///   - 指定了模型：走 `get_router_for_model` 精确查找，并校验对应 Worker 类型可用；
    ///   - 未指定模型：遍历 Router 快照按 `x-prefer-pd` 头部偏好打分，选出最高分且有效的 Router。
    pub fn select_router_for_request(
        &self,
        headers: Option<&HeaderMap>,
        model_id: Option<&str>,
    ) -> Option<Arc<dyn RouterTrait>> {
        // 单 Router 模式（enable_igw=false）：恒定使用默认 Router
        if !self.enable_igw {
            let default_router = self
                .default_router
                .read()
                .unwrap_or_else(|e| e.into_inner());
            if let Some(ref default_id) = *default_router {
                debug!(
                    "Single-router mode: using default router {} for model {:?}",
                    default_id.as_str(),
                    model_id
                );
                return self.routers.get(default_id).map(|r| r.clone());
            }
        }

        // 读取 `x-prefer-pd` 头部：是否偏好 PD（Prefill/Decode 分离）Router
        let prefer_pd = headers
            .and_then(|h| {
                h.get("x-prefer-pd")
                    .and_then(|v| v.to_str().ok())
                    .map(|s| s == "true" || s == "1")
            })
            .unwrap_or(false);

        // 获取当前 Regular / PD 两类 Worker 的数量分布，用于判断 Router 是否有效
        let (num_regular_workers, num_pd_workers) = self.worker_registry.get_worker_distribution();
        let mut best_router = None;
        let mut best_score = -1.0;

        // 抽取 Router 有效性判断为闭包，减少重复：PD Router 需有 PD Worker，反之亦然
        let is_router_valid =
            |is_pd: bool| (is_pd && num_pd_workers > 0) || (!is_pd && num_regular_workers > 0);

        if let Some(model) = model_id {
            // 指定了模型：高效的单次精确查找
            if let Some(router) = self.get_router_for_model(model) {
                if is_router_valid(router.is_pd_mode()) {
                    return Some(router);
                }
            }
        } else {
            // 未指定模型：零分配地遍历 Router 快照（热路径优化）
            // 原子 load 避免了每请求的堆分配与 DashMap 分片锁
            let routers_snapshot = self.routers_snapshot.load();
            for router in routers_snapshot.iter() {
                let mut score = 1.0;

                let is_pd = router.is_pd_mode();
                // 根据 prefer_pd 偏好与 Router 是否为 PD 模式加分：
                // 偏好 PD 且命中 PD 加 2 分；不偏好且命中非 PD 加 1 分
                if prefer_pd && is_pd {
                    score += 2.0;
                } else if !prefer_pd && !is_pd {
                    score += 1.0;
                }
                // TODO: Once routers expose worker stats, we can evaluate:
                // - Average worker priority vs priority_threshold
                // - Average worker cost vs max_cost
                // - Current load and health status

                // 保留得分更高且有效（存在对应类型 Worker）的 Router
                if score > best_score && is_router_valid(is_pd) {
                    best_score = score;
                    best_router = Some(Arc::clone(router));
                }
            }
        }

        best_router
    }
}

/// RouterManager 实现 RouterTrait：对外表现为一个「虚拟 Router」。
///
/// 健康/信息类接口基于聚合信息（全局 Worker/Router 统计）作答；
/// 路由类接口则统一遵循「解析 model_id → 选 Router → 委派给具体 Router」的模式。
#[async_trait]
impl RouterTrait for RouterManager {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // IGW 就绪性：只要至少一个 Router 拥有健康 Worker 就返回 200
        let has_healthy_workers = self
            .worker_registry
            .get_all()
            .iter()
            .any(|w| w.is_healthy());

        if has_healthy_workers {
            (StatusCode::OK, "At least one router has healthy workers").into_response()
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "No routers with healthy workers available",
            )
                .into_response()
        }
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        // TODO: Aggregate info from all routers with healthy workers
        (
            StatusCode::OK,
            serde_json::json!({
                "router_manager": true,
                "routers_count": self.routers.len(),
                "workers_count": self.worker_registry.get_all().len()
            })
            .to_string(),
        )
            .into_response()
    }

    async fn get_models(&self, _req: Request<Body>) -> Response {
        let model_names = self.worker_registry.get_models();

        if model_names.is_empty() {
            (StatusCode::SERVICE_UNAVAILABLE, "No models available").into_response()
        } else {
            // 将模型名转换为 OpenAI 兼容的 model 对象
            let models: Vec<Value> = model_names
                .iter()
                .map(|name| {
                    serde_json::json!({
                        "id": name,
                        "object": "model",
                        "owned_by": "local"
                    })
                })
                .collect();

            (
                StatusCode::OK,
                serde_json::json!({
                    "object": "list",
                    "data": models
                })
                .to_string(),
            )
                .into_response()
        }
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        // 路由到默认 router，若无则用第一个可用 router
        let router_id = {
            let default_router = self
                .default_router
                .read()
                .unwrap_or_else(|e| e.into_inner());
            default_router.clone()
        };

        let router = if let Some(id) = router_id {
            self.routers.get(&id).map(|r| r.clone())
        } else {
            // 无默认 router 时，使用第一个可用 router
            self.routers.iter().next().map(|r| r.value().clone())
        };

        if let Some(router) = router {
            router.get_model_info(req).await
        } else {
            (StatusCode::SERVICE_UNAVAILABLE, "No routers available").into_response()
        }
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        // IGW 模式：解析 model_id，无法解析时快速失败
        // 非 IGW 模式：透传给 router（由 router 自行校验）
        let effective_model_id = if self.enable_igw {
            match self.resolve_model_id(model_id) {
                Ok(id) => Some(id),
                Err(err_response) => return *err_response,
            }
        } else {
            None
        };

        let router =
            self.select_router_for_request(headers, effective_model_id.as_deref().or(model_id));

        if let Some(router) = router {
            router
                .route_generate(headers, body, effective_model_id.as_deref().or(model_id))
                .await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available for this request",
            )
                .into_response()
        }
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        // IGW 模式：先解析 model_id，无法解析则快速失败
        // 非 IGW 模式：直接透传给 Router（由 Router 自行校验）
        let effective_model_id = if self.enable_igw {
            // 优先用传入的 model_id，否则回退到 body.model
            let model = model_id.or(Some(&body.model));
            match self.resolve_model_id(model) {
                Ok(id) => Some(id),
                Err(err_response) => return *err_response,
            }
        } else {
            None
        };

        let router =
            self.select_router_for_request(headers, effective_model_id.as_deref().or(model_id));

        if let Some(router) = router {
            router
                .route_chat(headers, body, effective_model_id.as_deref().or(model_id))
                .await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        // IGW 模式：解析 model_id，无法解析时快速失败
        // 非 IGW 模式：透传给 router（由 router 自行校验）
        let effective_model_id = if self.enable_igw {
            // Use provided model_id or fall back to body.model
            let model = model_id.or(Some(&body.model));
            match self.resolve_model_id(model) {
                Ok(id) => Some(id),
                Err(err_response) => return *err_response,
            }
        } else {
            None
        };

        let router =
            self.select_router_for_request(headers, effective_model_id.as_deref().or(model_id));

        if let Some(router) = router {
            router
                .route_completion(headers, body, effective_model_id.as_deref().or(model_id))
                .await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        let selected_model = model_id.or(Some(body.model.as_str()));
        let router = self.select_router_for_request(headers, selected_model);

        if let Some(router) = router {
            router.route_responses(headers, body, selected_model).await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available to handle responses request",
            )
                .into_response()
        }
    }

    async fn get_response(
        &self,
        headers: Option<&HeaderMap>,
        response_id: &str,
        params: &ResponsesGetParams,
    ) -> Response {
        let router = self.select_router_for_request(headers, None);
        if let Some(router) = router {
            router.get_response(headers, response_id, params).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("No router available to get response '{}'", response_id),
            )
                .into_response()
        }
    }

    async fn cancel_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        let router = self.select_router_for_request(headers, None);
        if let Some(router) = router {
            router.cancel_response(headers, response_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("No router available to cancel response '{}'", response_id),
            )
                .into_response()
        }
    }

    async fn delete_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "responses api not yet implemented in inference gateway mode",
        )
            .into_response()
    }

    async fn list_response_input_items(
        &self,
        headers: Option<&HeaderMap>,
        response_id: &str,
    ) -> Response {
        // Delegate to the default router (typically http-regular)
        // Response storage is shared across all routers via AppContext
        let router = self.select_router_for_request(headers, None);
        if let Some(router) = router {
            router.list_response_input_items(headers, response_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available to list response input items",
            )
                .into_response()
        }
    }

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        body: &EmbeddingRequest,
        model_id: Option<&str>,
    ) -> Response {
        let router = self.select_router_for_request(headers, model_id);

        if let Some(router) = router {
            router.route_embeddings(headers, body, model_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        body: &ClassifyRequest,
        model_id: Option<&str>,
    ) -> Response {
        let router = self.select_router_for_request(headers, model_id);

        if let Some(router) = router {
            router.route_classify(headers, body, model_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        body: &RerankRequest,
        model_id: Option<&str>,
    ) -> Response {
        let router = self.select_router_for_request(headers, model_id);

        if let Some(router) = router {
            router.route_rerank(headers, body, model_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available for rerank request",
            )
                .into_response()
        }
    }

    fn router_type(&self) -> &'static str {
        "manager"
    }
}

impl std::fmt::Debug for RouterManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let default_router = self
            .default_router
            .read()
            .unwrap_or_else(|e| e.into_inner());
        f.debug_struct("RouterManager")
            .field("routers_count", &self.routers.len())
            .field("workers_count", &self.worker_registry.get_all().len())
            .field("default_router", &*default_router)
            .finish()
    }
}
