# srt/entrypoints

## 目录用途
本目录是 SGLang 运行时（SRT）的对外服务入口层，负责把外部请求接入推理引擎。它实现了进程内调用的 Python `Engine`、基于 FastAPI 的 HTTP 服务器、gRPC 服务器封装，以及 OpenAI/Anthropic/Ollama 兼容 API（位于同名子目录）。同时提供引擎抽象基类、会话上下文、工具调用、SSL 热更新、预热等支撑能力。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `EngineBase.py` | 引擎接口抽象基类，统一定义 generate、权重更新、显存控制等 API，供进程内 Engine 与 HTTP 适配器共同实现。 |
| `context.py` | 会话上下文抽象及实现（SimpleContext / HarmonyContext / StreamingHarmonyContext），管理多轮对话、工具调用与流式解析状态。 |
| `engine.py` | 推理引擎入口，实现进程内 Python API 的 `Engine` 类，负责启动 TokenizerManager/Scheduler/DetokenizerManager 并对外提供生成接口。 |
| `engine_annotated_zh.py` | `engine.py` 的带中文注释副本，逻辑与原文件一致，仅供学习理解，勿在生产中引用。 |
| `engine_info_bootstrap_server.py` | 轻量 HTTP 服务，运行于 node_rank==0 的守护线程，收集各 rank ModelRunner 注册的模型信息（如传输引擎显存注册信息）。 |
| `grpc_server.py` | gRPC 服务器的薄封装，委托给 `smg-grpc-servicer` 包启动带调度器的独立 gRPC 服务。 |
| `harmony_utils.py` | Harmony 编码相关工具，处理 system/developer/user 消息构造、Responses 输入输出解析及流式解析（改编自 vLLM）。 |
| `http_server.py` | 基于 FastAPI 的 HTTP 服务器，定义全局状态、生命周期、健康检查与各类路由，把请求转发给引擎。 |
| `http_server_annotated_zh.py` | `http_server.py` 的带中文注释副本，逻辑一致，仅供学习理解。 |
| `http_server_engine.py` | HTTP 引擎适配器 `HttpServerEngineAdapter`，以子进程启动 HTTP server 并通过请求实现 EngineBase 接口。 |
| `ssl_utils.py` | SSL 证书热更新工具 `SSLCertRefresher`，监听证书文件变化并就地刷新 SSLContext。 |
| `tool.py` | 工具调用抽象基类 `Tool` 及内置实现（HarmonyBrowserTool 浏览器、HarmonyPythonTool Python 执行）。 |
| `v1_loads.py` | `/v1/loads` 端点，返回调度器的负载/显存/队列等详细指标，用于负载均衡与容量规划。 |
| `warmup.py` | 预热注册与执行框架，提供 warmup 装饰器与 execute_warmups，内置 voice_chat 等预热任务。 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| `anthropic` | Anthropic Messages API 兼容层（协议模型与服务处理）。 |
| `ollama` | Ollama 兼容 API（协议、服务处理与本地/远端智能路由）。 |
| `openai` | OpenAI 兼容 API（chat/completions/embedding/rerank/responses 等多种端点的协议与服务实现）。 |
