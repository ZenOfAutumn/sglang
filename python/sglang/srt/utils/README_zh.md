# srt/utils

## 目录用途
本目录是 SGLang 运行时（srt）的通用工具函数集合，涵盖通用辅助函数、HuggingFace/Transformers 适配、网络与 IPC、共享内存、性能分析、日志、设备/内存管理、PyTorch 补丁等横切关注点。`__init__.py` 直接重导出 `common.py`，使旧代码可继续从 `sglang.srt.utils` 引用这些工具。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 重导出 `common.py` 全部符号，保持历史导入路径兼容。 |
| `aio_rwlock.py` | 基于 asyncio 的读写锁 `RWLock`，支持多读单写并发控制。 |
| `auth.py` | HTTP 服务鉴权工具（`AuthDecision`、`AuthLevel` 等），刻意保持轻量、不依赖 torch。 |
| `bench_utils.py` | 基准测试辅助，如 `suppress_stdout_stderr` 静默输出上下文等。 |
| `common.py` | 核心通用工具大全：设备探测（is_cuda/is_hip/is_npu）、端口/序列化、张量与内存辅助、`fast_topk` 等大量函数。 |
| `cuda_ipc_transport_utils.py` | 基于共享内存的 CUDA IPC 传输工具，用于多模态特征缓存的跨进程传递。 |
| `custom_op.py` | 自定义算子注册封装 `register_custom_op`，集成 torch.library 与内核 API 日志。 |
| `device_timer.py` | `DeviceTimer`，用于按区间记录设备侧耗时并上报。 |
| `gauge_histogram.py` | 非累积分桶的 Gauge 直方图（`BucketLabels` 等），适配 Grafana 热力图，与 Rust 网关实现保持同步。 |
| `hf_transformers_utils.py` | HuggingFace Transformers 适配工具：加载配置、tokenizer、processor 等。 |
| `host_shared_memory.py` | `HostSharedMemoryManager`，管理主机端共享内存记录，配合朴素分布式使用。 |
| `http_middleware_patch.py` | 修复 Starlette `BaseHTTPMiddleware` 破坏 `is_disconnected()` 的问题，使非流式请求能在客户端断连时中止。 |
| `json_response.py` | 基于 orjson 的 JSON 序列化工具与 `SGLangORJSONResponse`，统一各端点响应序列化行为。 |
| `log_utils.py` | 日志工具：创建日志目标、JSON 日志、滚动文件处理器等。 |
| `mistral_utils.py` | Mistral 模型配置适配（改编自 vLLM），将 Mistral 配置映射为 `PretrainedConfig`。 |
| `model_file_verifier.py` | 模型文件完整性校验工具，用 SHA256 校验和生成/验证模型文件，可作命令行模块运行。 |
| `multi_stream_utils.py` | 多 CUDA stream 辅助（改编自 trtllm），提供线程局部开关与上下文管理。 |
| `network.py` | 网络工具：获取空闲端口、IP/socket、ZMQ 相关辅助等。 |
| `numa_utils.py` | NUMA 亲和性与绑核工具，控制进程在 NUMA 节点上的 CPU/内存绑定。 |
| `nvtx_pytorch_hooks.py` | 为逐层 NVTX 性能分析注册的 PyTorch hook。 |
| `offloader.py` | 权重卸载器（offload），结合主机共享内存与朴素分布式在主机/设备间搬移模型权重。 |
| `patch_tokenizer.py` | 对特定 tokenizer（如 Kimi tiktoken）打补丁，修正特殊 token 缓存行为。 |
| `patch_torch.py` | 对 PyTorch（如 multiprocessing reductions）打补丁。 |
| `poll_based_barrier.py` | `PollBasedBarrier`，基于轮询的分布式屏障同步。 |
| `profile_merger.py` | `ProfileMerger`，将 TP/DP/PP/EP 各 rank 的 Chrome trace 合并为单个 trace。 |
| `profile_utils.py` | 性能分析辅助（torch profiler 调度、profile 请求处理等）。 |
| `request_logger.py` | 请求日志记录工具，按配置输出请求级日志。 |
| `rpd_utils.py` | ROCm rpd 数据库转 Chrome trace 的工具（`rpd_to_chrome_trace`）。 |
| `runai_utils.py` | RunAI/对象存储（s3/gs/az）模型文件加载辅助（改编自 vLLM）。 |
| `scheduler_status_logger.py` | `SchedulerStatusLogger`，周期性转储调度器/批次状态为 JSON 日志。 |
| `slow_rank_detector.py` | 慢 rank 检测，通过基准测试比较各 rank 性能定位异常节点。 |
| `tensor_bridge.py` | MLX 与 PyTorch 间的张量桥接，在 Apple Silicon 统一内存上尽量零拷贝转换。 |
| `torch_memory_saver_adapter.py` | `torch_memory_saver` 的适配封装，提供显存暂存/释放上下文（缺库时降级）。 |
| `video_decoder.py` | 统一视频解码器，优先 torchcodec，回退 decord。 |
| `watchdog.py` | 看门狗工具，监控子进程并在卡死/异常时触发转储或终止。 |
| `weight_checker.py` | `WeightChecker`，对模型权重（含 fp8 量化反量化）进行一致性检查。 |
