# 生产环境请求追踪

SGLang 基于 OpenTelemetry Collector 导出请求追踪数据。你可以在启动服务器时添加 `--enable-trace` 来启用追踪,并使用 `--otlp-traces-endpoint` 配置 OpenTelemetry Collector 端点。

你可以在 https://github.com/sgl-project/sglang/issues/8965 找到可视化效果的示例截图。

## 配置指南
本节说明如何配置请求追踪并导出追踪数据。
1. 安装所需的软件包和工具
    * 安装 Docker 和 Docker Compose
    * 安装依赖
    ```bash
    # enter the SGLang root directory
    pip install -e "python[tracing]"

    # or manually install the dependencies using pip
    pip install opentelemetry-sdk opentelemetry-api opentelemetry-exporter-otlp opentelemetry-exporter-otlp-proto-grpc
    ```

2. 启动 OpenTelemetry collector 和 Jaeger
    ```bash
    docker compose -f examples/monitoring/tracing_compose.yaml up -d
    ```

3. 启用追踪来启动你的 SGLang 服务器
    ```bash
    # set env variables
    export SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS=500
    export SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE=64
    # start the prefill and decode server
    python -m sglang.launch_server --enable-trace --otlp-traces-endpoint 0.0.0.0:4317 <other option>
    # start the model-gate-way
    python -m sglang_router.launch_router --enable-trace --otlp-traces-endpoint 0.0.0.0:4317 <other option>
    ```

    将 `0.0.0.0:4317` 替换为 OpenTelemetry collector 的实际端点。如果你使用 tracing_compose.yaml 启动了 OpenTelemetry collector,默认的接收端口是 4317。

    要使用 HTTP/protobuf span 导出器,设置以下环境变量并指向一个 HTTP 端点,例如 `http://0.0.0.0:4318/v1/traces`。
    ```bash
    export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
    ```


4. 发起一些请求
5. 观察追踪数据是否正在被导出
    * 使用 Web 浏览器访问 Jaeger 的 16686 端口以可视化请求追踪。
    * OpenTelemetry Collector 还会以 JSON 格式将追踪数据导出到 /tmp/otel_trace.json。在后续的补丁中,我们将提供一个工具将该数据转换为 Perfetto 兼容的格式,从而能够在 Perfetto UI 中可视化请求。

6. 动态调整追踪级别
    追踪级别接受从 `0` 到 `3` 的可配置值。不同追踪级别值的含义如下:
    ```
    0: disable tracing
    1: Trace important slices
    2: Trace all slices except nested ones
    3: Trace all slices
    ```
    追踪级别可以通过 HTTP API 动态设置,例如:
    ```bash
    curl http://0.0.0.0:30000/set_trace_level?level=2
    ```
    将 `0.0.0.0:30000` 替换为你实际的服务器地址,并将 `level=2` 替换为你想要设置的级别。

    **注意**:你必须设置参数 `--enable-trace`;否则,无论如何动态调整追踪级别,追踪能力都不会被启用。

## 如何为你感兴趣的 slice 添加追踪?(API 介绍)
我们已经在 tokenizer 和 scheduler 主线程中插入了埋点。如果你希望追踪额外的请求执行段或进行更细粒度的追踪,请使用追踪包中的 API,如下所述。

**以下所有实现都在 python/sglang/srt/observability/req_time_stats.py 中完成。如果你想添加另一个 slice,请在这里进行。**

1. 初始化

    在初始化阶段每个参与追踪的进程都应执行:
    ```python
    process_tracing_init(otlp_traces_endpoint, server_name)
    ```
    otlp_traces_endpoint 从参数中获取,你可以自由设置 server_name,但它应在所有进程中保持一致。

    在初始化阶段每个参与追踪的线程都应执行:
    ```python
    trace_set_thread_info("thread label", tp_rank, dp_rank)
    ```
    "thread label" 可以看作是线程的名称,用于在可视化视图中区分不同的线程。

2. 为请求创建追踪上下文
    每个请求都需要调用 `TraceReqContext()` 来初始化一个请求上下文,用于生成 slice span 并记录请求阶段信息。你可以将其存储在请求对象内,或将其作为全局变量维护。

3. 标记请求的开始和结束
    ```
    trace_ctx.trace_req_start().
    trace_ctx.trace_req_finish()
    ```
    trace_req_start() 和 trace_req_finish() 必须在同一个进程中调用,例如在 tokenizer 中。

4. 为 slice 添加追踪

    * 正常添加 slice 追踪:
        ```python
        trace_ctx.trace_slice_start(RequestStage.TOKENIZER.stage_name)
        trace_ctx.trace_slice_end(RequestStage.TOKENIZER.stage_name)

        or
        trace_ctx.trace_slice(slice: TraceSliceContext)
        ```

    - 线程中最后一个 slice 的结束必须用 thread_finish_flag=True 标记,或显式调用 trace_ctx.abort();否则,该线程的 span 将无法正确生成。
        ```python
        trace_ctx.slice_end(RequestStage.D.stage_name, thread_finish_flag = True)
        trace_ctx.abort()
        ```

5. 当请求执行流转移到另一个线程时,需要显式地重建线程上下文。
    - 接收方:在通过 ZMQ 接收到请求后执行以下代码
        ```python
        trace_ctx.rebuild_thread_context()
        ```

## 如何扩展追踪框架以支持复杂的追踪场景

当前提供的追踪包仍有进一步开发的潜力。如果你希望在其基础上构建更高级的功能,你必须首先理解其现有的设计原理。

追踪框架实现的核心在于 span 结构和 trace 上下文的设计。为了聚合分散的 slice 并支持对多个请求的并发跟踪,我们设计了一个三级 trace 上下文结构或 span 结构:`TraceReqContext`、`TraceThreadContext` 和 `TraceSliceContext`。它们的关系如下:
```
TraceReqContext (req_id="req-123")
├── TraceThreadContext(thread_label="scheduler", tp_rank=0)
|     └── TraceSliceContext(slice_name="prefill")
|
└── TraceThreadContext(thread_label="scheduler", tp_rank=1)
      └── TraceSliceContext(slice_name="prefill")
```

每个被追踪的请求维护一个全局的 `TraceReqContext` 并创建一个对应的请求 span。对于每个处理该请求的线程,都会记录一个 `TraceThreadContext` 并创建一个线程 span。`TraceThreadContext` 嵌套在 `TraceReqContext` 内,而每个当前被追踪的代码 slice(可能是嵌套的)都存储在其关联的 `TraceThreadContext` 中。

除了上述层级结构外,每个 slice 还通过 Span.add_link() 记录其前一个 slice,这可用于追踪执行流。
