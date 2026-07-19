//! 强类型工作流引擎集合
//!
//! 本模块为不同类型的工作流提供一组强类型的工作流引擎。
//! 每种工作流类型都拥有自己专用的引擎，并具备编译期类型安全保证。

use std::sync::Arc;

use wfaas::{EventSubscriber, InMemoryStore, WorkflowEngine};

use super::{
    create_external_worker_workflow, create_local_worker_workflow,
    create_mcp_registration_workflow, create_tokenizer_registration_workflow,
    create_wasm_module_registration_workflow, create_wasm_module_removal_workflow,
    create_worker_removal_workflow, create_worker_update_workflow, ExternalWorkerWorkflowData,
    LocalWorkerWorkflowData, McpWorkflowData, TokenizerWorkflowData, WasmRegistrationWorkflowData,
    WasmRemovalWorkflowData, WorkerRemovalWorkflowData, WorkerUpdateWorkflowData,
};
use crate::config::RouterConfig;

/// 本地 worker 工作流引擎的类型别名
pub type LocalWorkerEngine =
    WorkflowEngine<LocalWorkerWorkflowData, InMemoryStore<LocalWorkerWorkflowData>>;

/// 外部 worker 工作流引擎的类型别名
pub type ExternalWorkerEngine =
    WorkflowEngine<ExternalWorkerWorkflowData, InMemoryStore<ExternalWorkerWorkflowData>>;

/// worker 移除工作流引擎的类型别名
pub type WorkerRemovalEngine =
    WorkflowEngine<WorkerRemovalWorkflowData, InMemoryStore<WorkerRemovalWorkflowData>>;

/// worker 更新工作流引擎的类型别名
pub type WorkerUpdateEngine =
    WorkflowEngine<WorkerUpdateWorkflowData, InMemoryStore<WorkerUpdateWorkflowData>>;

/// MCP 注册工作流引擎的类型别名
pub type McpEngine = WorkflowEngine<McpWorkflowData, InMemoryStore<McpWorkflowData>>;

/// 分词器注册工作流引擎的类型别名
pub type TokenizerEngine =
    WorkflowEngine<TokenizerWorkflowData, InMemoryStore<TokenizerWorkflowData>>;

/// WASM 注册工作流引擎的类型别名
pub type WasmRegistrationEngine =
    WorkflowEngine<WasmRegistrationWorkflowData, InMemoryStore<WasmRegistrationWorkflowData>>;

/// WASM 移除工作流引擎的类型别名
pub type WasmRemovalEngine =
    WorkflowEngine<WasmRemovalWorkflowData, InMemoryStore<WasmRemovalWorkflowData>>;

/// 强类型工作流引擎集合
///
/// 每种工作流类型都拥有自己专用的引擎，并具备编译期类型安全保证。
/// 它取代了旧的 `WorkflowEngine<AnyWorkflowData, ...>` 方式(去除了运行期类型擦除)。
///
/// 各引擎均以 `Arc` 包裹，可在多个组件间共享并安全并发访问。
#[derive(Clone, Debug)]
pub struct WorkflowEngines {
    /// 本地 worker 注册工作流引擎
    pub local_worker: Arc<LocalWorkerEngine>,
    /// 外部 worker 注册工作流引擎
    pub external_worker: Arc<ExternalWorkerEngine>,
    /// worker 移除工作流引擎
    pub worker_removal: Arc<WorkerRemovalEngine>,
    /// worker 更新工作流引擎
    pub worker_update: Arc<WorkerUpdateEngine>,
    /// MCP 服务注册工作流引擎
    pub mcp: Arc<McpEngine>,
    /// 分词器注册工作流引擎
    pub tokenizer: Arc<TokenizerEngine>,
    /// WASM 模块注册工作流引擎
    pub wasm_registration: Arc<WasmRegistrationEngine>,
    /// WASM 模块移除工作流引擎
    pub wasm_removal: Arc<WasmRemovalEngine>,
}

impl WorkflowEngines {
    /// 创建并初始化所有工作流引擎，并为其注册对应的工作流定义
    pub fn new(router_config: &RouterConfig) -> Self {
        // 创建本地 worker 引擎
        let local_worker = WorkflowEngine::new();
        local_worker
            .register_workflow(create_local_worker_workflow(router_config))
            .expect("local_worker_registration workflow should be valid");

        // 创建外部 worker 引擎
        let external_worker = WorkflowEngine::new();
        external_worker
            .register_workflow(create_external_worker_workflow())
            .expect("external_worker_registration workflow should be valid");

        // 创建 worker 移除引擎
        let worker_removal = WorkflowEngine::new();
        worker_removal
            .register_workflow(create_worker_removal_workflow())
            .expect("worker_removal workflow should be valid");

        // 创建 worker 更新引擎
        let worker_update = WorkflowEngine::new();
        worker_update
            .register_workflow(create_worker_update_workflow())
            .expect("worker_update workflow should be valid");

        // 创建 MCP 引擎
        let mcp = WorkflowEngine::new();
        mcp.register_workflow(create_mcp_registration_workflow())
            .expect("mcp_registration workflow should be valid");

        // 创建分词器引擎
        let tokenizer = WorkflowEngine::new();
        tokenizer
            .register_workflow(create_tokenizer_registration_workflow())
            .expect("tokenizer_registration workflow should be valid");

        // 创建 WASM 注册引擎
        let wasm_registration = WorkflowEngine::new();
        wasm_registration
            .register_workflow(create_wasm_module_registration_workflow())
            .expect("wasm_module_registration workflow should be valid");

        // 创建 WASM 移除引擎
        let wasm_removal = WorkflowEngine::new();
        wasm_removal
            .register_workflow(create_wasm_module_removal_workflow())
            .expect("wasm_module_removal workflow should be valid");

        Self {
            local_worker: Arc::new(local_worker),
            external_worker: Arc::new(external_worker),
            worker_removal: Arc::new(worker_removal),
            worker_update: Arc::new(worker_update),
            mcp: Arc::new(mcp),
            tokenizer: Arc::new(tokenizer),
            wasm_registration: Arc::new(wasm_registration),
            wasm_removal: Arc::new(wasm_removal),
        }
    }

    /// 将一个事件订阅者订阅到所有工作流引擎
    pub async fn subscribe_all<S: EventSubscriber + 'static>(&self, subscriber: Arc<S>) {
        self.local_worker
            .event_bus()
            .subscribe(subscriber.clone())
            .await;
        self.external_worker
            .event_bus()
            .subscribe(subscriber.clone())
            .await;
        self.worker_removal
            .event_bus()
            .subscribe(subscriber.clone())
            .await;
        self.worker_update
            .event_bus()
            .subscribe(subscriber.clone())
            .await;
        self.mcp.event_bus().subscribe(subscriber.clone()).await;
        self.tokenizer
            .event_bus()
            .subscribe(subscriber.clone())
            .await;
        self.wasm_registration
            .event_bus()
            .subscribe(subscriber.clone())
            .await;
        self.wasm_removal.event_bus().subscribe(subscriber).await;
    }
}
