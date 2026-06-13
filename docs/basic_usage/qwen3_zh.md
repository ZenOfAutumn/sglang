# Qwen3-Next 使用指南

SGLang 自[这个 PR](https://github.com/sgl-project/sglang/pull/10233) 起已支持 Qwen3-Next-80B-A3B-Instruct 和 Qwen3-Next-80B-A3B-Thinking。

## 使用 SGLang 启动 Qwen3-Next

在 4xH100/H200 GPU 上部署 Qwen3-Next 模型：

```bash
python3 -m sglang.launch_server --model Qwen/Qwen3-Next-80B-A3B-Instruct --tp 4
```

### 配置技巧
- `--max-mamba-cache-size`：调整 `--max-mamba-cache-size` 可以增加 mamba 缓存空间以及最大并发请求处理能力。作为权衡，它会减少 KV cache 空间。你可以根据工作负载进行调整。
- `--mamba-ssm-dtype`：可选 `bfloat16` 或 `float32`，使用 `bfloat16` 以节省 mamba 缓存大小，使用 `float32` 以获得更精确的结果。默认设置为 `float32`。
- `--mamba-full-memory-ratio`：mamba 状态内存与完整 kv cache 内存的比例。默认值为 0.9。

### Mamba Radix Cache
SGLang 为 Qwen3-Next 模型支持名为 `MambaRadixCache` 的前缀缓存，它通过复用计算结果来提升推理速度。`MambaRadixCache` 有两个版本：
- `no_buffer`：默认版本，也是其他混合线性模型的选择。启用后，出于兼容性原因，SGLang 会自动关闭 overlap schedule。
- `extra_buffer`：一个优化版本，兼容诸如 page size > 1、overlap schedule 和投机解码等特性。它还支持在分支位置存储 mamba 状态。然而，它需要为每个请求额外占用两份 mamba 空间用作乒乓缓冲（ping-pong buffer）。要启用它，请在启动服务器时添加参数 `--mamba-scheduler-strategy extra_buffer`。

### EAGLE 投机解码
**说明**：SGLang 已支持 Qwen3-Next 模型使用 [EAGLE 投机解码](https://docs.sglang.io/advanced_features/speculative_decoding.html#EAGLE-Decoding)。

**用法**：
添加参数 `--speculative-algorithm`、`--speculative-num-steps`、`--speculative-eagle-topk` 和 `--speculative-num-draft-tokens` 来启用此特性。例如：

``` bash
python3 -m sglang.launch_server \
  --model Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tp 4 \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-algo NEXTN
```

详情可参见[这个 PR](https://github.com/sgl-project/sglang/pull/10233)。
