//! Worker 管理模块
//!
//! 提供 Worker 生命周期相关操作，以及「扇出（fan-out）」式并发请求工具。
//!
//! 本模块包含两大部分：
//! - [`WorkerManager`]：无状态的工具集，向多个 Worker 扇出请求（刷缓存、拉负载、拉指标等）。
//! - [`LoadMonitor`]：后台负载监控服务，周期性拉取各 Worker 负载并推送给需要负载信息的策略。

use std::{collections::HashMap, sync::Arc, time::Duration};

use axum::response::{IntoResponse, Response};
use futures::{
    future,
    stream::{self, StreamExt},
};
use http::StatusCode;
use serde_json::Value;
use tokio::{
    sync::{watch, Mutex},
    task::JoinHandle,
};
use tracing::{debug, info, warn};

use crate::{
    core::{metrics_aggregator::MetricPack, ConnectionMode, Worker, WorkerRegistry, WorkerType},
    policies::PolicyRegistry,
    protocols::worker_spec::{FlushCacheResult, WorkerLoadInfo, WorkerLoadsResult},
};

/// 单次扇出请求的超时时长。
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
/// 扇出时的最大并发数（同时在飞的请求上限）。
const MAX_CONCURRENT: usize = 32;

/// 向单个 Worker 扇出请求的结果。
struct WorkerResponse {
    /// 该 Worker 的 URL。
    url: String,
    /// HTTP 请求的结果（成功拿到响应或网络错误）。
    result: Result<reqwest::Response, reqwest::Error>,
}

/// 并行地向多个 Worker 扇出（fan-out）同一请求。
///
/// 使用 `buffer_unordered` 限制最大并发为 [`MAX_CONCURRENT`]，避免瞬时涌入大量连接。
/// 每个请求都带 [`REQUEST_TIMEOUT`] 超时与可选的 Bearer 鉴权。
async fn fan_out(
    workers: &[Arc<dyn Worker>],
    client: &reqwest::Client,
    endpoint: &str,
    method: reqwest::Method,
) -> Vec<WorkerResponse> {
    let futures: Vec<_> = workers
        .iter()
        .map(|worker| {
            let client = client.clone();
            let url = worker.url().to_string();
            let full_url = format!("{}/{}", url, endpoint);
            let api_key = worker.api_key().clone();
            let method = method.clone();

            async move {
                // 构造带超时的请求，若 Worker 配了 api_key 则附上 Bearer 鉴权
                let mut req = client.request(method, &full_url).timeout(REQUEST_TIMEOUT);
                if let Some(key) = api_key {
                    req = req.bearer_auth(key);
                }
                WorkerResponse {
                    url,
                    result: req.send().await,
                }
            }
        })
        .collect();

    // 以乱序（谁先完成谁先返回）但受限并发的方式收集所有响应
    stream::iter(futures)
        .buffer_unordered(MAX_CONCURRENT)
        .collect()
        .await
}

/// 引擎指标聚合结果：成功时携带聚合后的指标文本，失败时携带错误信息。
pub enum EngineMetricsResult {
    /// 聚合成功，携带 Prometheus 格式的指标文本。
    Ok(String),
    /// 聚合失败，携带错误描述。
    Err(String),
}

impl IntoResponse for EngineMetricsResult {
    fn into_response(self) -> Response {
        match self {
            Self::Ok(text) => (StatusCode::OK, text).into_response(),
            Self::Err(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response(),
        }
    }
}

/// Worker 管理器（无状态工具集）。
///
/// 以关联函数的形式提供一组面向多 Worker 的扇出操作，本身不持有任何状态。
pub struct WorkerManager;

impl WorkerManager {
    /// 获取注册表中所有 Worker 的 URL 列表。
    pub fn get_worker_urls(registry: &Arc<WorkerRegistry>) -> Vec<String> {
        registry
            .get_all()
            .iter()
            .map(|w| w.url().to_string())
            .collect()
    }

    /// 向所有 HTTP Worker 扇出「刷缓存」请求。
    ///
    /// 仅针对 HTTP 连接的 Worker（gRPC 等其他连接方式不走此路径）；
    /// 汇总成功/失败列表并返回 [`FlushCacheResult`]。
    pub async fn flush_cache_all(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> FlushCacheResult {
        let workers = worker_registry.get_all();
        let total_workers = workers.len();

        // 只保留 HTTP 连接模式的 Worker
        let http_workers: Vec<_> = workers
            .into_iter()
            .filter(|w| matches!(w.connection_mode(), ConnectionMode::Http))
            .collect();

        if http_workers.is_empty() {
            return FlushCacheResult {
                successful: vec![],
                failed: vec![],
                total_workers,
                http_workers: 0,
                message: "No HTTP workers available for cache flush".to_string(),
            };
        }

        info!(
            "Flushing cache on {} HTTP workers (out of {} total)",
            http_workers.len(),
            total_workers
        );

        // 并发向各 HTTP Worker 发送 POST /flush_cache
        let responses = fan_out(&http_workers, client, "flush_cache", reqwest::Method::POST).await;

        let mut successful = Vec::new();
        let mut failed = Vec::new();

        // 按响应状态分类：2xx 计成功，其余状态码与网络错误计失败
        for resp in responses {
            match resp.result {
                Ok(r) if r.status().is_success() => successful.push(resp.url),
                Ok(r) => failed.push((resp.url, format!("HTTP {}", r.status()))),
                Err(e) => failed.push((resp.url, e.to_string())),
            }
        }

        let message = if failed.is_empty() {
            format!(
                "Successfully flushed cache on all {} HTTP workers",
                successful.len()
            )
        } else {
            format!(
                "Cache flush: {} succeeded, {} failed",
                successful.len(),
                failed.len()
            )
        };

        info!("{}", message);

        FlushCacheResult {
            successful,
            failed,
            total_workers,
            http_workers: http_workers.len(),
            message,
        }
    }

    /// 并发拉取所有 Worker 的当前负载。
    ///
    /// 仅 HTTP Worker 会实际请求负载接口；非 HTTP Worker 负载统一记为 -1（表示不可用）。
    /// 返回结果中同时统计成功（load >= 0）与失败（load < 0）的数量。
    pub async fn get_all_worker_loads(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> WorkerLoadsResult {
        let workers = worker_registry.get_all();
        let total_workers = workers.len();

        let futures: Vec<_> = workers
            .iter()
            .map(|worker| {
                let url = worker.url().to_string();
                let api_key = worker.api_key().clone();
                // 将 Worker 类型映射为负载信息中的类型标签（Regular 无标签）
                let worker_type = match worker.worker_type() {
                    WorkerType::Regular => None,
                    WorkerType::Prefill { .. } => Some("prefill".to_string()),
                    WorkerType::Decode => Some("decode".to_string()),
                };
                let is_http = matches!(worker.connection_mode(), ConnectionMode::Http);
                let client = client.clone();

                async move {
                    // 仅 HTTP Worker 拉取真实负载；其余一律记为 -1
                    let load = if is_http {
                        Self::parse_load_response(&client, &url, api_key.as_deref()).await
                    } else {
                        -1
                    };
                    WorkerLoadInfo {
                        worker: url,
                        worker_type,
                        load,
                    }
                }
            })
            .collect();

        // 等待全部负载拉取完成，并统计成功/失败数
        let loads = future::join_all(futures).await;
        let successful = loads.iter().filter(|l| l.load >= 0).count();
        let failed = loads.iter().filter(|l| l.load < 0).count();

        WorkerLoadsResult {
            loads,
            total_workers,
            successful,
            failed,
        }
    }

    /// 请求单个 HTTP Worker 的负载接口并解析出总 token 数作为负载值。
    ///
    /// 从响应 JSON 的 `aggregate.total_tokens` 字段提取；
    /// 请求失败、非 2xx、JSON 解析失败或字段缺失时均返回 -1。
    async fn parse_load_response(
        client: &reqwest::Client,
        url: &str,
        api_key: Option<&str>,
    ) -> isize {
        let load_url = format!("{}/v1/loads?include=core", url);
        let mut req = client.get(&load_url).timeout(REQUEST_TIMEOUT);
        if let Some(key) = api_key {
            req = req.bearer_auth(key);
        }

        match req.send().await {
            Ok(r) if r.status().is_success() => match r.json::<Value>().await {
                Ok(json) => json
                    .get("aggregate")
                    .and_then(|a| a.get("total_tokens"))
                    .and_then(|v| v.as_i64())
                    .map(|n| n as isize)
                    .unwrap_or(-1),
                _ => -1,
            },
            _ => -1,
        }
    }

    /// 向所有 Worker 扇出「拉指标」请求，并将各自的 Prometheus 文本聚合为一份。
    ///
    /// 每份指标会打上 `worker_addr` 标签以区分来源；
    /// 若没有任何 Worker 或全部请求失败，则返回错误。
    pub async fn get_engine_metrics(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> EngineMetricsResult {
        let workers = worker_registry.get_all();

        if workers.is_empty() {
            return EngineMetricsResult::Err("No available workers".to_string());
        }

        let responses = fan_out(&workers, client, "metrics", reqwest::Method::GET).await;

        // 收集每个成功响应的指标文本，并附上 worker_addr 标签
        let mut metric_packs = Vec::new();
        for resp in responses {
            if let Ok(r) = resp.result {
                if r.status().is_success() {
                    if let Ok(text) = r.text().await {
                        metric_packs.push(MetricPack {
                            labels: vec![("worker_addr".into(), resp.url)],
                            metrics_text: text,
                        });
                    }
                }
            }
        }

        if metric_packs.is_empty() {
            return EngineMetricsResult::Err("All backend requests failed".to_string());
        }

        match crate::core::metrics_aggregator::aggregate_metrics(metric_packs) {
            Ok(text) => EngineMetricsResult::Ok(text),
            Err(e) => EngineMetricsResult::Err(format!("Failed to aggregate metrics: {}", e)),
        }
    }
}

/// 负载监控服务：周期性拉取各 Worker 负载并推送给需要负载信息的策略。
///
/// 主要服务于 Power-of-Two 等负载感知策略：把负载采集从请求热路径中剥离，
/// 改为后台定时拉取，避免每请求都去查询负载。
pub struct LoadMonitor {
    /// Worker 注册表，用于枚举待拉取负载的 Worker。
    worker_registry: Arc<WorkerRegistry>,
    /// 策略注册表，用于找到需要负载信息的策略（如 Power-of-Two）并推送更新。
    policy_registry: Arc<PolicyRegistry>,
    /// 用于向 Worker 发起负载查询的 HTTP 客户端。
    client: reqwest::Client,
    /// 两次负载拉取之间的时间间隔。
    interval: Duration,
    /// watch 通道发送端：把最新负载快照广播出去。
    tx: watch::Sender<HashMap<String, isize>>,
    /// watch 通道接收端：供订阅方获取最新负载快照。
    rx: watch::Receiver<HashMap<String, isize>>,
    /// 后台监控任务的句柄（加锁保护，便于 start/stop 幂等控制）。
    monitor_handle: Arc<Mutex<Option<JoinHandle<()>>>>,
}

impl LoadMonitor {
    /// 创建一个负载监控器（此时后台任务尚未启动，需调用 `start`）。
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        client: reqwest::Client,
        interval_secs: u64,
    ) -> Self {
        let (tx, rx) = watch::channel(HashMap::new());

        Self {
            worker_registry,
            policy_registry,
            client,
            interval: Duration::from_secs(interval_secs),
            tx,
            rx,
            monitor_handle: Arc::new(Mutex::new(None)),
        }
    }

    /// 启动后台负载监控任务。
    ///
    /// 幂等：若已在运行则直接返回，避免重复启动多个监控循环。
    pub async fn start(&self) {
        let mut handle_guard = self.monitor_handle.lock().await;
        if handle_guard.is_some() {
            debug!("Load monitoring already running");
            return;
        }

        info!(
            "Starting load monitoring with interval: {:?}",
            self.interval
        );

        let worker_registry = Arc::clone(&self.worker_registry);
        let policy_registry = Arc::clone(&self.policy_registry);
        let client = self.client.clone();
        let interval = self.interval;
        let tx = self.tx.clone();

        let handle = tokio::spawn(async move {
            Self::monitor_loop(worker_registry, policy_registry, client, interval, tx).await;
        });

        *handle_guard = Some(handle);
    }

    /// 停止后台负载监控任务并等待其结束。
    pub async fn stop(&self) {
        let mut handle_guard = self.monitor_handle.lock().await;
        if let Some(handle) = handle_guard.take() {
            info!("Stopping load monitoring");
            handle.abort();
            let _ = handle.await; // 等待任务真正结束
        }
    }

    /// 订阅负载快照更新，返回一个 watch 接收端。
    pub fn subscribe(&self) -> watch::Receiver<HashMap<String, isize>> {
        self.rx.clone()
    }

    /// 后台监控主循环：按固定间隔拉取负载并推送给负载感知策略。
    ///
    /// 优化点：若当前没有任何 Power-of-Two 策略，则跳过本轮负载拉取，
    /// 避免为无人使用的负载数据做无谓的网络请求。
    async fn monitor_loop(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        client: reqwest::Client,
        interval: Duration,
        tx: watch::Sender<HashMap<String, isize>>,
    ) {
        let mut interval_timer = tokio::time::interval(interval);

        loop {
            interval_timer.tick().await;

            // 仅当存在 Power-of-Two 策略时才需要负载数据
            let power_of_two_policies = policy_registry.get_all_power_of_two_policies();

            if power_of_two_policies.is_empty() {
                debug!("No PowerOfTwo policies found, skipping load fetch");
                continue;
            }

            // 并发拉取全部 Worker 负载，并展平为 URL -> 负载 的映射
            let result = WorkerManager::get_all_worker_loads(&worker_registry, &client).await;

            let mut loads = HashMap::new();
            for load_info in result.loads {
                loads.insert(load_info.worker, load_info.load);
            }

            // 拉到负载才推送：逐个更新策略内部缓存，并通过 watch 广播最新快照
            if !loads.is_empty() {
                debug!(
                    "Fetched loads from {} workers, updating {} PowerOfTwo policies",
                    loads.len(),
                    power_of_two_policies.len()
                );
                for policy in &power_of_two_policies {
                    policy.update_loads(&loads);
                }
                let _ = tx.send(loads);
            } else {
                warn!("No loads fetched from workers");
            }
        }
    }

    /// 返回后台监控任务是否正在运行。
    pub async fn is_running(&self) -> bool {
        let handle_guard = self.monitor_handle.lock().await;
        handle_guard.is_some()
    }
}

impl Drop for LoadMonitor {
    /// 销毁时尽力终止后台监控任务，避免任务泄漏。
    ///
    /// 使用 `try_lock`（非阻塞）：若此刻锁被占用则放弃终止，
    /// 以免在 Drop 中阻塞。
    fn drop(&mut self) {
        if let Ok(mut handle_guard) = self.monitor_handle.try_lock() {
            if let Some(handle) = handle_guard.take() {
                handle.abort();
            }
        }
    }
}
