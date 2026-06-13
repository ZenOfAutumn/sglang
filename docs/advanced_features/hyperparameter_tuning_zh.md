# 超参数调优(Hyperparameter Tuning)

## 为离线批量推理实现高吞吐量

实现大批次大小是离线批量推理中获得高吞吐量最重要的事情。
当服务器在稳态下满负荷运行时,在日志中查找以下内容:

```Decode batch. #running-req: 233, #token: 370959, token usage: 0.82, cuda graph: True, gen throughput (token/s): 4594.01, #queue-req: 317```

### 调整请求提交速度以控制 `#queue-req`

`#queue-req` 表示队列中的请求数量。
如果你频繁看到 `#queue-req: 0`,这表明你的客户端代码提交请求太慢。
`#queue-req` 的健康范围是 `100 - 2000`。
然而,避免让 `#queue-req` 太大,因为这会增加服务器上的调度开销。

### 实现高 `token usage`

`token usage` 表示服务器的 KV 缓存内存利用率。`token usage > 0.9` 意味着良好的利用率。

如果你频繁看到 `token usage < 0.9` 且 `#queue-req > 0`,这意味着服务器在接纳新请求方面过于保守。你可以将 `--schedule-conservativeness` 降低到诸如 0.3 的值。
当用户发送许多带有较大 `max_new_tokens` 的请求,但这些请求由于 EOS 或停止字符串而很早就停止时,就可能出现服务器过于保守的情况。

另一方面,如果你看到 `token usage` 非常高,并且频繁看到诸如
`KV cache pool is full. Retract requests. #retracted_reqs: 1, #new_token_ratio: 0.9998 -> 1.0000` 的警告,你可以将 `--schedule-conservativeness` 增加到诸如 1.3 的值。
如果你偶尔(但不频繁,约每分钟 1 次)看到 `KV cache pool is full. Retract requests.`,那是可以接受的。

### 调优 `--mem-fraction-static` 以增加 KV 缓存池容量
SGLang 按如下方式分配内存:

总内存使用量 = 模型权重 + KV 缓存池 + CUDA graph 缓冲区 + 激活值(activations)

`--mem-fraction-static` 参数决定了分配给前两个组件的内存量:

mem_fraction_static = (模型权重 + KV 缓存池) / GPU 内存容量

为了支持更高的并发,你应该通过将 `--mem-fraction-static` 设置得尽可能高来最大化 KV 缓存池容量,同时仍为激活值和 CUDA graph 缓冲区保留足够的内存。

SGLang 使用简单的启发式方法来设置 `--mem-fraction-static` 的默认值,但你可以针对你的用例进行优化。
作为经验法则,为激活值保留 5–8 GB 的内存通常就足够了。你可以通过检查服务器就绪前的日志来确认这一点。
查找类似如下的日志条目:

```
[2025-08-11 17:17:03] max_total_num_tokens=665690, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=4096, context_len=65536, available_gpu_mem=13.50 GB
```

检查 `available_gpu_mem` 的值。
- 如果它在 5–8 GB 之间,设置就很好。
- 如果它太高(例如 10 - 20 GB),增加 `--mem-fraction-static` 以为 KV 缓存分配更多内存。
- 如果它太低,你有可能在之后遇到内存不足(OOM)错误,因此降低 `--mem-fraction-static`。

另一种直接的方法是以 0.01 的增量增加 `--mem-fraction-static`,直到你的工作负载遇到 OOM 错误。

### 通过调优 `--chunked-prefill-size`、`--mem-fraction-static` 和 `--max-running-requests` 来避免内存不足错误

如果你遇到内存不足(OOM)错误,你可以调整以下参数:

- 如果 OOM 发生在 prefill 期间,尝试将 `--chunked-prefill-size` 减小到 `4096` 或 `2048`。这会节省内存,但会减慢长 prompt 的 prefill 速度。
- 如果 OOM 发生在 decode 期间,尝试降低 `--max-running-requests`。
- 你也可以将 `--mem-fraction-static` 减小到一个较小的值,例如 0.8 或 0.7。这会减少 KV 缓存内存池的内存使用量,并有助于防止 prefill 和 decode 期间的 OOM 错误。然而,它会限制最大并发并降低峰值吞吐量。

### 调优 `--cuda-graph-max-bs`
默认情况下,CUDA graph 仅对小批次大小(例如小于 160 或 256)启用。
然而,对于某些模型,尤其是在大张量并行规模下,CUDA graph 对于高达 512 或 768 的批次大小可能很有用。
因此,将 `--cuda-graph-max-bs` 增加到一个更大的值可能是有益的。
注意,CUDA graph 会消耗更多内存,因此你可能需要同时减小 `--mem-fraction-static`。

### 调优 `--dp-size` 和 `--tp-size`

数据并行对吞吐量更有利。当有足够的 GPU 内存时,始终优先选择数据并行以获得吞吐量。请参阅 [SGLang Model Gateway(前身为 Router)](../advanced_features/sgl_model_gateway.md) 以获得比使用 `dp_size` 参数更好的数据并行。

### 尝试其他选项

- `torch.compile` 在小批次大小上加速小模型。你可以使用 `--enable-torch-compile` 启用它。
- 尝试其他量化(例如使用 `--quantization fp8` 的 FP8 量化)
- 尝试其他并行策略(例如 [专家并行](https://lmsys.org/blog/2025-05-05-large-scale-ep/))或针对 deepseek 模型的 DP attention(使用 `--enable-dp-attention --dp-size 8`)。
- 如果工作负载有许多共享前缀,尝试 `--schedule-policy lpm`。这里,`lpm` 代表最长前缀匹配(longest prefix match)。它会对请求重新排序以鼓励更多的缓存命中,但会引入更多的调度开销。
