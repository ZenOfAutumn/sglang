// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use clap::Parser;
use sgl_router::config::{Cli, LogFormat};
use std::sync::Arc;
// Unix 信号相关类型，用于监听 SIGTERM / SIGINT 实现优雅关机。
use tokio::signal::unix::{signal, Signal, SignalKind};

/// 安装全局的 tracing 日志订阅器。
///
/// 幂等（idempotent）：重复调用不会 panic，而是返回 `Ok`。当 `try_init`
/// 返回错误时，说明已经有其他代码安装了订阅器，因此下方的
/// `tracing::debug!` 会通过那个已存在的订阅器输出 —— 不会递归初始化。
///
/// `format` 决定输出形式：`Json` 每行输出一条 JSON 记录（适用于生产环境
/// 或 k8s 日志聚合器），`Text` 则是便于人阅读的默认格式。
/// 环境变量 `RUST_LOG` 的优先级始终高于 `default_level`。
fn init_tracing(default_level: &str, format: LogFormat) -> Result<()> {
    // 优先从 `RUST_LOG` 环境变量读取日志过滤级别，若未设置则回退到 default_level。
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_level));
    // 根据配置的格式选择 JSON 或纯文本输出，并尝试安装为全局订阅器。
    let install_result = match format {
        LogFormat::Json => tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(true)
            .json()
            .try_init(),
        LogFormat::Text => tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(true)
            .try_init(),
    };
    if let Err(e) = install_result {
        // 第二次安装尝试；已存在的订阅器仍可正常工作。
        // 这里把尝试使用的默认级别打印出来，便于运维人员了解我们曾尝试的配置。
        tracing::debug!(
            default_level = %default_level,
            ?format,
            error = %e,
            "tracing subscriber already installed; continuing"
        );
    }
    Ok(())
}

/// 在配置解析之前先安装一个最小的文本格式订阅器，这样配置解析阶段
/// 发生的错误才有地方可以输出。真正的订阅器（由 `Config.observability` 驱动）
/// 在之后安装；由于已有订阅器存在，第二次 `try_init` 会是一个空操作。
/// 该引导（bootstrap）订阅器同样尊重 `RUST_LOG`，因此即使配置解析失败，
/// 运维人员也能用 `RUST_LOG=debug` 调试启动过程。
fn install_bootstrap_subscriber() {
    // 引导订阅器默认用 info 级别，同样允许 RUST_LOG 覆盖。
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    // 忽略安装结果：即使已有订阅器也无妨。
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .try_init();
}

/// 提前安装 SIGTERM 与 SIGINT 信号处理器，使这里的失败在 `axum::serve`
/// 启动之前就能暴露出来。若安装失败（罕见：容器缺乏信号能力、
/// seccomp 策略限制等），我们返回错误并让进程干净退出，
/// 而不是在无法感知 k8s 终止信号的情况下继续运行。
fn install_signal_handlers() -> Result<(Signal, Signal)> {
    // SIGTERM：k8s / 容器编排器优雅终止进程时发送的信号。
    let sigterm = signal(SignalKind::terminate()).context("install SIGTERM handler")?;
    // SIGINT：终端 Ctrl+C 发送的中断信号。
    let sigint = signal(SignalKind::interrupt()).context("install SIGINT handler")?;
    Ok((sigterm, sigint))
}

#[tokio::main]
async fn main() -> Result<()> {
    // 解析命令行参数。
    let cli = Cli::parse();
    // 先安装引导订阅器，使配置解析错误能有结构化的日志输出。
    // 之后根据配置安装的正式订阅器会因 try_init 的幂等性而成为空操作。
    install_bootstrap_subscriber();
    // 从 CLI 参数解析出完整配置；若失败则附上上下文信息并提前退出。
    let cfg = cli
        .into_config()
        .context("resolve configuration from CLI flags")?;

    // 根据配置中的日志级别与格式安装正式的 tracing 订阅器。
    init_tracing(&cfg.observability.log_level, cfg.observability.log_format)?;

    tracing::info!(
        "sgl-router {} starting on {}:{}",
        env!("CARGO_PKG_VERSION"),
        cfg.server.host,
        cfg.server.port
    );

    // 加载分词器注册表（多个模型可能各自拥有不同分词器），用 Arc 包装以便跨任务共享。
    let tokenizers = Arc::new(
        sgl_router::tokenizer::TokenizerRegistry::load_from_config(&cfg)
            .context("load tokenizers")?,
    );

    // 创建 worker 注册表，后续由发现（discovery）与管理器（manager）任务动态维护。
    let registry = Arc::new(sgl_router::workers::WorkerRegistry::default());

    // 提前构建 KV 事件索引，以便 cache-aware-zmq 策略能共享其 `HashTree`
    // 句柄与 `BlockSizeOracle`。即使没有任何模型使用 `cache_aware_zmq`，
    // 该索引仍会被构建（开销极低），只是不会添加任何订阅者。
    let block_size_oracle = sgl_router::policies::kv_events::BlockSizeOracle::new();
    let kv_index = sgl_router::policies::kv_events::KvEventIndex::new_with_http_and_oracle(
        reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(2))
            .build()
            .expect("default http client builds"),
        Arc::clone(&block_size_oracle),
    );
    let policies = Arc::new(
        sgl_router::policies::factory::build_registry(
            &cfg,
            kv_index.tree(),
            Arc::clone(&tokenizers),
            Arc::clone(&block_size_oracle),
        )
        .context("build policy registry")?,
    );

    // 共享的 ActiveLoadRegistry（活跃负载注册表）+ janitor（清理）任务。
    // janitor 会回收那些存活时长超过 `stale_request_timeout` 的请求条目，
    // 因此一个泄露的 guard（例如代理任务 panic）不会永久抬高某个 worker 的负载。
    // 该注册表在 manager 启动之前构建，这样 manager 才能在收到
    // `DiscoveryEvent::Removed` 时调用 `forget_worker`。
    let stale_timeout = std::time::Duration::from_secs(cfg.active_load.stale_request_timeout_secs);
    let active_load = sgl_router::policies::active_load::ActiveLoadRegistry::new(
        // 使用系统时钟作为时间源，用于判断请求条目是否过期。
        Arc::new(sgl_router::policies::active_load::SystemTimeClock),
        stale_timeout,
    );
    // 清理周期为配置超时的 1/10，并被限制在 [1 秒, 60 秒] 区间内。
    // 较短的超时（测试场景）需要频繁清理，才能在测试窗口内触发；
    // 较长的超时（生产场景）则无需亚分钟级别的检查。
    let sweep_interval = std::time::Duration::from_secs(
        (cfg.active_load.stale_request_timeout_secs / 10).clamp(1, 60),
    );
    // 启动后台 janitor 任务，按 sweep_interval 周期性清理过期请求条目。
    let janitor_handle =
        sgl_router::policies::active_load::spawn_janitor(Arc::clone(&active_load), sweep_interval);

    // 启动 discovery（服务发现）与 manager（worker 管理）任务。
    // spawn_discovery 返回一个事件接收端 event_rx 以及可用于取消的句柄。
    let (event_rx, discovery_handle) = sgl_router::discovery::spawn_discovery(&cfg)
        .await
        .context("spawn discovery")?;
    // 将 KV 事件索引以 Option 形式传给 manager，使其在 worker 增删时更新索引。
    let kv_index_opt: Option<Arc<sgl_router::policies::kv_events::KvEventIndex>> =
        Some(Arc::clone(&kv_index));
    let manager_handle = tokio::spawn(sgl_router::workers::manager::run_with_config(
        event_rx,
        registry.clone(),
        Some(Arc::new(cfg.clone())),
        kv_index_opt,
        Some(Arc::clone(&active_load)),
    ));

    // 构建反向代理客户端，并根据配置设置请求超时时长。
    let proxy = Arc::new(
        sgl_router::proxy::Proxy::new(std::time::Duration::from_secs(
            cfg.proxy.request_timeout_secs,
        ))
        .context("build proxy client")?,
    );

    // 组装应用上下文 AppContext，汇集所有共享组件（配置、分词器、代理、
    // worker 注册表、策略与活跃负载），供后续请求处理使用。
    let ctx = Arc::new(
        sgl_router::server::app_context::AppContext::with_active_load(
            cfg.clone(),
            tokenizers,
            proxy,
            registry,
            policies,
            active_load,
        ),
    );
    // 标记服务就绪，使健康检查（readiness）接口开始返回“就绪”。
    ctx.mark_ready();

    // 基于上下文构建 axum 路由（注册所有 HTTP 端点）。
    let app = sgl_router::server::app::build_router(ctx.clone());

    // 拼接监听地址（host:port）并绑定 TCP 监听器。
    let bind = format!("{}:{}", cfg.server.host, cfg.server.port);
    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .with_context(|| format!("bind {bind}"))?;
    tracing::info!("listening on {bind}");

    // 安装信号处理器（失败则在开始服务前就报错）。
    let (sigterm, sigint) = install_signal_handlers()?;

    // 启动 HTTP 服务，并绑定优雅关机逻辑：收到 SIGTERM/SIGINT 后停止接收新连接。
    let serve = axum::serve(listener, app).with_graceful_shutdown(shutdown_signal(sigterm, sigint));
    let server_result = serve.await.context("axum serve");

    // 尽力而为（best-effort）：关机时取消 discovery + manager + janitor 任务。
    // janitor 句柄的 drop 会发出取消信号；此外我们 await `shutdown`，
    // 使该任务在进程退出前干净地 join —— 这对追踪尾部日志很有用。
    discovery_handle.abort();
    manager_handle.abort();
    janitor_handle.shutdown().await;
    server_result
}

/// 优雅关机信号等待器：监听 SIGTERM 与 SIGINT，任意一个到达即触发关机。
async fn shutdown_signal(mut sigterm: Signal, mut sigint: Signal) {
    // tokio::select! 同时等待两个信号，哪个先到达就执行对应分支。
    tokio::select! {
        _ = sigterm.recv() => tracing::info!("got SIGTERM, shutting down"),
        _ = sigint.recv()  => tracing::info!("got SIGINT, shutting down"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn install_signal_handlers_returns_both() {
        // 锁定这一契约：在标准 tokio 运行时上信号处理器能正常安装。
        // 如果在沙箱化的 runner 上失败，真实服务也同样会安装失败 —— 这正是本用例要揭示的。
        assert!(install_signal_handlers().is_ok());
    }

    #[test]
    fn init_tracing_is_idempotent() {
        // 验证 init_tracing 可重复调用（幂等）而不会 panic。
        let _ = init_tracing("info", LogFormat::Text);
        let _ = init_tracing("info", LogFormat::Text);
    }

    #[test]
    fn init_tracing_accepts_json_format() {
        // 无论我们是否赢得与其他订阅器安装的竞争 —— 该函数都必须返回 Ok。
        assert!(init_tracing("info", LogFormat::Json).is_ok());
    }
}
