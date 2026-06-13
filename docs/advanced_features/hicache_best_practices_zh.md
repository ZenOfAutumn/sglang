# SGLang HiCache 最佳实践

## 为什么 HiCache 很重要

SGLang HiCache 在传统 RadixAttention 的基础上扩展出一套三层分级 KV 缓存系统,能够显著提升长上下文和多轮对话场景下的性能。通过在 GPU 显存、主机内存和外部存储后端之间智能地管理 KV 缓存,HiCache 解决了传统系统中限制缓存命中率的根本性容量瓶颈。

## 配置指南

## 核心 HiCache 参数

```bash
# Essential HiCache flags
--page-size 64                        # Page size for cache management
--enable-hierarchical-cache           # Enable HiCache
--hicache-ratio 2                     # Host memory ratio (2x GPU memory)
--hicache-size 100                    # Host memory size in GBs, will override the above ratio
--hicache-io-backend kernel           # The I/O backend of moving data between CPU and GPU
--hicache-write-policy write_through  # Cache write policy from GPU to CPU
--hicache-storage-backend             # Optional storage backend (e.g., hf3fs, mooncake, etc.)
```

注意:

- 除了在启动时配置 `--hicache-storage-backend` 之外,SGLang 还支持通过 HTTP 管理端点在**运行时挂载/卸载(attach/detach)** HiCache 存储后端(无需重启)。详见 [运行时挂载/卸载 HiCache 存储后端](hicache_storage_runtime_attach_detach.md)。

## 启用存储后端时的关键配置

### 内存布局优化

```bash
# Page-first: Optimized for I/O efficiency with zero-copy (recommended with kernel backend)
--hicache-mem-layout page_first
# Page-first-direct: Optimized for direct I/O operations (Compatible with fa3 and same zero-copy performance as page_first)
--hicache-mem-layout page_first_direct
# Layer-first
--hicache-mem-layout layer_first
```
**布局兼容性:**
- `page_first`:仅兼容 `kernel` I/O 后端,在使用 `direct` 后端时会自动切换为 `layer_first`
- `page_first_direct`:专为 `direct` I/O 后端设计,具有优化的内存组织方式

### 异构 TP 支持(GQA/MHA 模型)

当不同的部署使用不同的 TP 大小(例如 `tp=4` 和 `tp=8`)并共享同一个存储后端命名空间时,HiCache 存储支持跨集群的 KV 复用。

在 `--hicache-storage-backend-extra-config` 中使用 `tp_lcm_size`:

```bash
# Example: heterogeneous TP = {4, 8}, so lcm = 8
--hicache-storage-backend-extra-config '{"tp_lcm_size": 8}'
```

指南:

- 将 `tp_lcm_size` 设置为所有将共享同一 HiCache 存储的 TP 大小的最小公倍数(LCM)。
- 对于使用 Mooncake 和 `page_head` 布局的 MHA 模型,HiCache 会根据 `tp_lcm_size` 拆分 head 分片,使得 key 可以在异构 TP 部署之间复用。
- 如果所有集群都使用相同的 TP 大小,则不需要此选项。

### 预取策略

```bash
# Best-effort: Terminate prefetch when needed
--hicache-storage-prefetch-policy best_effort
# Wait-complete: Ensure complete prefetch, higher cache reuse
--hicache-storage-prefetch-policy wait_complete
# Timeout: Balance between completion and best-effort
--hicache-storage-prefetch-policy timeout
```

### 与 PD 分离(PD Disaggregation)集成

HiCache 可以与 PD 分离无缝协作。你可以在两种配置之间选择:

1. **仅 Prefill 的 HiCache**:仅在 Prefill 节点上启用 HiCache,允许 Prefill 实例之间共享 KV 缓存
2. **带异步卸载的完整 HiCache**:在 Prefill 节点上启用 HiCache,并在 Decode 节点上启用异步 KV 缓存卸载,使得在多轮对话场景中 Prefill 节点能够复用来自 Decode 节点的 KV 缓存

```bash
# Prefill node with HiCache enabled for cross-prefill sharing (ideal for SystemPrompt scenarios)
python3 -m sglang.launch_server \
  --model-path /xxx/DeepSeek-R1/ \
  --tp 8 \
  --host 0.0.0.0 \
  --port 10000 \
  --enable-metrics \
  --enable-cache-report \
  --mem-fraction-static 0.85 \
  --page-size 64 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-size 0 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-write-policy write_through \
  --hicache-storage-backend hf3fs \
  --hicache-storage-prefetch-policy wait_complete \
  --disaggregation-ib-device mlx5_0 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake

# Decode node with async offloading enabled for KV cache reuse by Prefill (ideal for multi-turn conversations)
python3 -m sglang.launch_server \
  --model-path /xxx/DeepSeek-R1/ \
  --tp 8 \
  --host 0.0.0.0 \
  --port 10000 \
  --enable-metrics \
  --enable-cache-report \
  --page-size 64 \
  --hicache-ratio 2 \
  --hicache-size 0 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-write-policy write_through \
  --hicache-storage-backend hf3fs \
  --hicache-storage-prefetch-policy wait_complete \
  --disaggregation-decode-enable-offload-kvcache \  # Enable async KV cache offloading in decode node
  --disaggregation-ib-device mlx5_0 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake
```


### 使用 HF3FS 部署

下面是一个使用 HiCache-HF3FS 部署 DeepSeek-R1 的示例。更多细节请参见 [HF3FS 文档](../../python/sglang/srt/mem_cache/storage/hf3fs/docs/README.md)。

```bash
python3 -m sglang.launch_server \
  --model-path /xxx/DeepSeek-R1/ \
  --log-level info \
  --tp 8 \
  --host 0.0.0.0 \
  --port 10000 \
  --enable-metrics \
  --enable-cache-report \
  --page-size 64 \
  --mem-fraction-static 0.85 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-size 0 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-write-policy write_through \
  --hicache-storage-backend hf3fs \
  --hicache-storage-prefetch-policy wait_complete \
```

### 使用 Mooncake 部署

下面是一个使用 Mooncake 部署 Qwen3-235B-A22B-Instruct-2507 的示例。更多细节请参见 [Mooncake 文档](../../python/sglang/srt/mem_cache/storage/mooncake_store/README.md)。

```bash
# Set Mooncake environment variables
export MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata"
export MOONCAKE_GLOBAL_SEGMENT_SIZE=816043786240
export MOONCAKE_PROTOCOL="rdma"
export MOONCAKE_DEVICE="$DEVICE_LIST"
export MOONCAKE_MASTER=127.0.0.1:50051

# Launch SGLang server with Mooncake backend
python3 -m sglang.launch_server \
  --model-path $MODEL_PATH \
  --tp 8 \
  --page-size 64 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-storage-backend mooncake \
  --hicache-write-policy write_through \
  --hicache-storage-prefetch-policy timeout
```


## 自定义存储后端集成

要集成一个新的存储后端:

1. **实现三个核心方法:**
   - `get(key)`:按 key 获取 value
   - `exists(key)`:检查 key 是否存在
   - `set(key, value)`:存储键值对

2. **注册你的后端:** 将你的存储后端添加到 HiCache [BackendFactory](../../python/sglang/srt/mem_cache/storage/backend_factory.py#L188)

HiCache 控制器会自动处理所有的调度和同步。

### 动态后端加载

或者,你可以使用动态加载,以避免将你的后端硬编码到代码仓库中:

```bash
python3 -m sglang.launch_server \
  --model-path your-model \
  --enable-hierarchical-cache \
  --hicache-storage-backend dynamic \
  --hicache-storage-backend-extra-config '{"backend_name":"custom_backend_name", "module_path": "your_module_path", "class_name": "YourHiCacheClassName"}'
```

**配置参数:**
- `--hicache-storage-backend`:设置为 `dynamic`
- `--hicache-storage-backend-extra-config`:JSON 配置,包含:
  - `backend_name`:自定义后端标识符
  - `module_path`:你的实现的 Python 模块路径
  - `class_name`:你的 HiCache 实现类名
  - `interface_v1`:0(禁用)或 1(启用),用于控制是否使用 batch_get_v1 和 batch_set_v1 方法


## 社区与支持

- **GitHub Issues**:报告 bug 和功能请求
- **Slack Channel**:在 #sgl-kv-cache-store 中加入社区讨论
- **Documentation**:参阅特定存储后端的指南

---

*本文档将根据社区反馈和新功能持续更新。欢迎贡献和建议!*
