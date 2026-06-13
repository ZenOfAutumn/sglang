# Ascend NPU 上支持的功能

本节介绍 Ascend NPU 支持的基本功能和特性。如果你遇到问题或有任何
疑问,请[提交一个 issue](https://github.com/sgl-project/sglang/issues)。

如果你想了解每个参数的含义和用法,
请点击 [Server Arguments](https://docs.sglang.io/advanced_features/server_arguments.html)。

## 模型与 tokenizer

| 参数                                   | 默认值   | 可选项                                | 支持的 Server    |
|----------------------------------------|----------|---------------------------------------|:----------------:|
| `--model-path`<br/>`--model`           | `None`   | Type: str                             |      A2, A3      |
| `--tokenizer-path`                     | `None`   | Type: str                             |      A2, A3      |
| `--tokenizer-mode`                     | `auto`   | `auto`, `slow`                        |      A2, A3      |
| `--tokenizer-worker-num`               | `1`      | Type: int                             |      A2, A3      |
| `--skip-tokenizer-init`                | `False`  | 布尔标志(设置以启用)             |      A2, A3      |
| `--load-format`                        | `auto`   | `auto`, `safetensors`                 |      A2, A3      |
| `--model-loader-` <br/> `extra-config` | `{}`     | Type: str                             |      A2, A3      |
| `--trust-remote-code`                  | `False`  | 布尔标志(设置以启用)             |      A2, A3      |
| `--context-length`                     | `None`   | Type: int                             |      A2, A3      |
| `--is-embedding`                       | `False`  | 布尔标志(设置以启用)             |      A2, A3      |
| `--enable-multimodal`                  | `None`   | 布尔标志(设置以启用)             |      A2, A3      |
| `--revision`                           | `None`   | Type: str                             |      A2, A3      |
| `--model-impl`                         | `auto`   | `auto`, `sglang`,<br/> `transformers` |      A2, A3      |

## HTTP server

| 参数                   | 默认值      | 可选项                    | 支持的 Server    |
|------------------------|-------------|---------------------------|:----------------:|
| `--host`               | `127.0.0.1` | Type: str                 |      A2, A3      |
| `--port`               | `30000`     | Type: int                 |      A2, A3      |
| `--skip-server-warmup` | `False`     | 布尔标志(设置以启用)   |      A2, A3      |
| `--warmups`            | `None`      | Type: str                 |      A2, A3      |
| `--nccl-port`          | `None`      | Type: int                 |      A2, A3      |
| `--fastapi-root-path`  | `None`      | Type: str                 |      A2, A3      |
| `--grpc-mode`          | `False`     | 布尔标志(设置以启用)   |      A2, A3      |

## 量化与数据类型

| 参数                                        | 默认值   | 可选项                                  | 支持的 Server    |
|---------------------------------------------|----------|-----------------------------------------|:----------------:|
| `--dtype`                                   | `auto`   | `auto`,<br/> `float16`,<br/> `bfloat16` |      A2, A3      |
| `--quantization`                            | `None`   | `modelslim`                             |      A2, A3      |
| `--quantization-param-path`                 | `None`   | Type: str                               | GPU 专用         |
| `--kv-cache-dtype`                          | `auto`   | `auto`                                  |      A2, A3      |
| `--enable-fp32-lm-head`                     | `False`  | 布尔标志 <br/> (设置以启用)            |      A2, A3      |
| `--modelopt-quant`                          | `None`   | Type: str                               | GPU 专用         |
| `--modelopt-checkpoint-`<br/>`restore-path` | `None`   | Type: str                               | GPU 专用         |
| `--modelopt-checkpoint-`<br/>`save-path`    | `None`   | Type: str                               | GPU 专用         |
| `--modelopt-export-path`                    | `None`   | Type: str                               | GPU 专用         |
| `--quantize-and-serve`                      | `False`  | 布尔标志 <br/> (设置以启用)            | GPU 专用         |
| `--rl-quant-profile`                        | `None`   | Type: str                               | GPU 专用         |

## 内存与调度

| 参数                                                | 默认值   | 可选项                         | 支持的 Server    |
|-----------------------------------------------------|----------|--------------------------------|:----------------:|
| `--mem-fraction-static`                             | `None`   | Type: float                    |      A2, A3      |
| `--max-running-requests`                            | `None`   | Type: int                      |      A2, A3      |
| `--prefill-max-requests`                            | `None`   | Type: int                      |      A2, A3      |
| `--max-queued-requests`                             | `None`   | Type: int                      |      A2, A3      |
| `--max-total-tokens`                                | `None`   | Type: int                      |      A2, A3      |
| `--chunked-prefill-size`                            | `None`   | Type: int                      |      A2, A3      |
| `--max-prefill-tokens`                              | `16384`  | Type: int                      |      A2, A3      |
| `--schedule-policy`                                 | `fcfs`   | `lpm`, `fcfs`                  |      A2, A3      |
| `--enable-priority-`<br/>`scheduling`               | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--schedule-low-priority-`<br/>`values-first`       | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--priority-scheduling-`<br/>`preemption-threshold` | `10`     | Type: int                      |      A2, A3      |
| `--schedule-conservativeness`                       | `1.0`    | Type: float                    |      A2, A3      |
| `--page-size`                                       | `128`    | Type: int                      |      A2, A3      |
| `--swa-full-tokens-ratio`                           | `0.8`    | Type: float                    |      A2, A3      |
| `--disable-hybrid-swa-memory`                       | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--radix-eviction-policy`                           | `lru`    | `lru`,<br/>`lfu`               |      A2, A3      |
| `--enable-prefill-delayer`                          | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--prefill-delayer-max-delay-passes`                | `30`     | Type: int                      |      A2, A3      |
| `--prefill-delayer-token-usage-low-watermark`       | `None`   | Type: float                    |      A2, A3      |
| `--prefill-delayer-forward-passes-buckets`          | `None`   | List[float]                    |      A2, A3      |
| `--prefill-delayer-wait-seconds-buckets`            | `None`   | List[float]                    |      A2, A3      |
| `--abort-on-priority-`<br/>`when-disabled`          | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--enable-dynamic-chunking`                         | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |

## 运行时选项

| 参数                                               | 默认值   | 可选项                    | 支持的 Server    |
|----------------------------------------------------|----------|---------------------------|:----------------:|
| `--device`                                         | `None`   | Type: str                 |      A2, A3      |
| `--tensor-parallel-size`<br/>`--tp-size`           | `1`      | Type: int                 |      A2, A3      |
| `--pipeline-parallel-size`<br/>`--pp-size`         | `1`      | Type: int                 |      A2, A3      |
| `--attention-context-parallel-size`<br/>`--attn-cp-size`  | `1` | Type: int               |      A2, A3      |
| `--moe-data-parallel-size`<br/>`--moe-dp-size`     | `1`      | Type: int                 |      A2, A3      |
| `--pp-max-micro-batch-size`                        | `None`   | Type: int                 |      A2, A3      |
| `--pp-async-batch-depth`                           | `None`   | Type: int                 |      A2, A3      |
| `--stream-interval`                                | `1`      | Type: int                 |      A2, A3      |
| `--incremental-streaming-output`                   | `False`  | 布尔标志(设置以启用)   |      A2, A3      |
| `--random-seed`                                    | `None`   | Type: int                 |      A2, A3      |
| `--constrained-json-`<br/>`whitespace-pattern`     | `None`   | Type: str                 |      A2, A3      |
| `--constrained-json-`<br/>`disable-any-whitespace` | `False`  | 布尔标志(设置以启用)   |      A2, A3      |
| `--watchdog-timeout`                               | `300`    | Type: float               |      A2, A3      |
| `--soft-watchdog-timeout`                          | `300`    | Type: float               |      A2, A3      |
| `--dist-timeout`                                   | `None`   | Type: int                 |      A2, A3      |
| `--download-dir`                                   | `None`   | Type: str                 |      A2, A3      |
| `--model-checksum`                                 | `None`   | Type: str                 |      A2, A3      |
| `--base-gpu-id`                                    | `0`      | Type: int                 |      A2, A3      |
| `--gpu-id-step`                                    | `1`      | Type: int                 |      A2, A3      |
| `--sleep-on-idle`                                  | `False`  | 布尔标志(设置以启用)   |      A2, A3      |

## 日志

| 参数                                               | 默认值            | 可选项                         | 支持的 Server    |
|----------------------------------------------------|-------------------|--------------------------------|:----------------:|
| `--log-level`                                      | `info`            | Type: str                      |      A2, A3      |
| `--log-level-http`                                 | `None`            | Type: str                      |      A2, A3      |
| `--log-requests`                                   | `False`           | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--log-requests-level`                             | `2`               | `0`, `1`, `2`, `3`             |      A2, A3      |
| `--log-requests-format`                            | `text`            | `text`, `json`                 |      A2, A3      |
| `--crash-dump-folder`                              | `None`            | Type: str                      |      A2, A3      |
| `--enable-metrics`                                 | `False`           | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--enable-metrics-for-`<br/>`all-schedulers`       | `False`           | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--tokenizer-metrics-`<br/>`custom-labels-header`  | `x-custom-labels` | Type: str                      |      A2, A3      |
| `--tokenizer-metrics-`<br/>`allowed-custom-labels` | `None`            | List[str]                      |      A2, A3      |
| `--bucket-time-to-`<br/>`first-token`              | `None`            | List[float]                    |      A2, A3      |
| `--bucket-inter-token-`<br/>`latency`              | `None`            | List[float]                    |      A2, A3      |
| `--bucket-e2e-request-`<br/>`latency`              | `None`            | List[float]                    |      A2, A3      |
| `--collect-tokens-`<br/>`histogram`                | `False`           | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--prompt-tokens-buckets`                          | `None`            | List[str]                      |      A2, A3      |
| `--generation-tokens-buckets`                      | `None`            | List[str]                      |      A2, A3      |
| `--gc-warning-threshold-secs`                      | `0.0`             | Type: float                    |      A2, A3      |
| `--decode-log-interval`                            | `40`              | Type: int                      |      A2, A3      |
| `--enable-request-time-`<br/>`stats-logging`       | `False`           | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--kv-events-config`                               | `None`            | Type: str                      | GPU 专用         |
| `--enable-trace`                                   | `False`           | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--oltp-traces-endpoint`                           | `localhost:4317`  | Type: str                      |      A2, A3      |
| `--log-requests-target`                            | `None`            | Type: str                      |      A2, A3      |
| `--uvicorn-access-log-exclude-prefixes`            | `[]`              | List[str]                      |      A2, A3      |

## RequestMetricsExporter 配置

| 参数                                  | 默认值   | 可选项                         | 支持的 Server    |
|---------------------------------------|----------|--------------------------------|:----------------:|
| `--export-metrics-to-`<br/>`file`     | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--export-metrics-to-`<br/>`file-dir` | `None`   | Type: str                      |      A2, A3      |

## API 相关

| 参数                    | 默认值    | 可选项                         | 支持的 Server    |
|-------------------------|-----------|--------------------------------|:----------------:|
| `--api-key`             | `None`    | Type: str                      |      A2, A3      |
| `--admin-api-key`       | `None`    | Type: str                      |      A2, A3      |
| `--served-model-name`   | `None`    | Type: str                      |      A2, A3      |
| `--weight-version`      | `default` | Type: str                      |      A2, A3      |
| `--chat-template`       | `None`    | Type: str                      |      A2, A3      |
| `--hf-chat-template-name` | `None`  | Type: str                      |      A2, A3      |
| `--completion-template` | `None`    | Type: str                      |      A2, A3      |
| `--enable-cache-report` | `False`   | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--reasoning-parser`    | `None`    | `deepseek-r1`<br/>`deepseek-v3`<br/>`glm45`<br/>`gpt-oss`<br/>`kimi`<br/>`qwen3`<br/>`qwen3-thinking`<br/>`step3`                  |      A2, A3      |
| `--tool-call-parser`    | `None`    | `deepseekv3`<br/>`deepseekv31`<br/>`glm`<br/>`glm45`<br/>`glm47`<br/>`gpt-oss`<br/>`kimi_k2`<br/>`llama3`<br/>`mistral`<br/>`pythonic`<br/>`qwen`<br/>`qwen25`<br/>`qwen3_coder`<br/>`step3`<br/>`gigachat3`            |      A2, A3      |
| `--sampling-defaults`   | `model`   | `openai`, `model`              |      A2, A3      |

## 数据并行

| 参数                                   | 默认值        | 可选项                                                   | 支持的 Server    |
|----------------------------------------|---------------|-----------------------------------------------------------|:----------------:|
| `--data-parallel-size`<br/>`--dp-size` | `1`           | Type: int                                                 |      A2, A3      |
| `--load-balance-method`                | `auto` | `auto`,<br/> `round_robin`,<br/> `follow_bootstrap_room`,<br/> `total_requests`,<br/> `total_tokens` |      A2, A3      |

## 多节点分布式服务

| 参数                                      | 默认值   | 可选项    | 支持的 Server    |
|-------------------------------------------|----------|-----------|:----------------:|
| `--dist-init-addr`<br/>`--nccl-init-addr` | `None`   | Type: str |      A2, A3      |
| `--nnodes`                                | `1`      | Type: int |      A2, A3      |
| `--node-rank`                             | `0`      | Type: int |      A2, A3      |

## 模型覆盖参数

| 参数                                 | 默认值   | 可选项    | 支持的 Server    |
|--------------------------------------|----------|-----------|:----------------:|
| `--json-model-override-`<br/>`args`  | `{}`     | Type: str |      A2, A3      |
| `--preferred-sampling-`<br/>`params` | `None`   | Type: str |      A2, A3      |

## LoRA

| 参数                     | 默认值   | 可选项                              | 支持的 Server    |
|--------------------------|----------|-------------------------------------|:----------------:|
| `--enable-lora`          | `False`  | 布尔标志 <br/>(设置以启用)         |      A2, A3      |
| `--max-lora-rank`        | `None`   | Type: int                           |      A2, A3      |
| `--lora-target-modules`  | `None`   | `all`                               |      A2, A3      |
| `--lora-paths`           | `None`   | Type: List[str] /<br/> JSON objects |      A2, A3      |
| `--max-loras-per-batch`  | `8`      | Type: int                           |      A2, A3      |
| `--max-loaded-loras`     | `None`   | Type: int                           |      A2, A3      |
| `--lora-eviction-policy` | `lru`    | `lru`,<br/> `fifo`                  |      A2, A3      |
| `--lora-backend`         | `csgmv`  | `triton`,<br/>`csgmv`,<br/>`ascend`,<br/>`torch_native`  |      A2, A3      |
| `--max-lora-chunk-size`  | `16`     | `16`, `32`,<br/> `64`, `128`        | GPU 专用         |

## Kernel 后端(Attention、Sampling、Grammar、GEMM)

| 参数                                   | 默认值            | 可选项                                                                                         | 支持的 Server    |
|----------------------------------------|-------------------|------------------------------------------------------------------------------------------------|:----------------:|
| `--attention-backend`                  | `None`            | `ascend`                                                                                       |      A2, A3      |
| `--prefill-attention-backend`          | `None`            | `ascend`                                                                                       |      A2, A3      |
| `--decode-attention-backend`           | `None`            | `ascend`                                                                                       |      A2, A3      |
| `--sampling-backend`                   | `None`            | `pytorch`,<br/>`ascend`                                                                        |      A2, A3      |
| `--grammar-backend`                    | `None`            | `xgrammar`                                                                                     |      A2, A3      |
| `--mm-attention-backend`               | `None`            | `ascend_attn`                                                                                  |      A2, A3      |
| `--nsa-prefill-backend`                | `flashmla_sparse` | `flashmla_sparse`,<br/> `flashmla_decode`,<br/>`fa3`,<br/> `tilelang`,<br/> `aiter`            | GPU 专用         |
| `--nsa-decode-backend`                 | `fa3`             | `flashmla_prefill`,<br/> `flashmla_kv`,<br/> `fa3`,<br/>`tilelang`,<br/> `aiter`               | GPU 专用         |
| `--fp8-gemm-backend`                   | `auto`            | `auto`,<br/> `deep_gemm`,<br/> `flashinfer_trtllm`,<br/>`flashinfer_cutlass`,<br/>`flashinfer_deepgemm`,<br/>`cutlass`,<br/> `triton`,<br/> `aiter` | GPU 专用         |
| `--disable-flashinfer-`<br/>`autotune` | `False`           | 布尔标志<br/> (设置以启用)                                                                    | GPU 专用         |

## 推测解码(Speculative decoding)

| 参数                                                             | 默认值    | 可选项                   | 支持的 Server    |
|------------------------------------------------------------------|-----------|--------------------------|:----------------:|
| `--speculative-algorithm`                                        | `None`    | `EAGLE3`,<br/> `NEXTN`   |      A2, A3      |
| `--speculative-draft-model-path`<br/>`--speculative-draft-model` | `None`    | Type: str                |      A2, A3      |
| `--speculative-draft-model-`<br/>`revision`                      | `None`    | Type: str                |      A2, A3      |
| `--speculative-draft-load-format`                                | `None`    | `auto`                   |      A2, A3      |
| `--speculative-num-steps`                                        | `None`    | Type: int                |      A2, A3      |
| `--speculative-eagle-topk`                                       | `None`    | Type: int                |      A2, A3      |
| `--speculative-num-draft-tokens`                                 | `None`    | Type: int                |      A2, A3      |
| `--speculative-accept-`<br/>`threshold-single`                   | `1.0`     | Type: float              | GPU 专用         |
| `--speculative-accept-`<br/>`threshold-acc`                      | `1.0`     | Type: float              | GPU 专用         |
| `--speculative-token-map`                                        | `None`    | Type: str                |      A2, A3      |
| `--speculative-attention-`<br/>`mode`                            | `prefill` | `prefill`,<br/> `decode` |      A2, A3      |
| `--speculative-moe-runner-`<br/>`backend`                        | `None`    | `auto`                   |      A2, A3      |
| `--speculative-moe-a2a-`<br/>`backend`                           | `None`    | `ascend_fuseep`          |      A2, A3      |
| `--speculative-draft-attention-backend`                          | `None`    | `ascend`                 |      A2, A3      |
| `--speculative-draft-model-quantization`                         | `None`    | `unquant`                |      A2, A3      |

## Ngram 推测解码

| 参数                                               | 默认值     | 可选项             | 支持的 Server    |
|----------------------------------------------------|------------|--------------------|:----------------:|
| `--speculative-ngram-`<br/>`min-match-window-size` | `1`        | Type: int          |   实验性         |
| `--speculative-ngram-`<br/>`max-match-window-size` | `12`       | Type: int          |   实验性         |
| `--speculative-ngram-`<br/>`min-bfs-breadth`       | `1`        | Type: int          |   实验性         |
| `--speculative-ngram-`<br/>`max-bfs-breadth`       | `10`       | Type: int          |   实验性         |
| `--speculative-ngram-`<br/>`match-type`            | `BFS`      | `BFS`,<br/> `PROB` |   实验性。`BFS` 使用基于近期(recency)的扩展;`PROB` 使用基于频率(frequency)的扩展。 |
| `--speculative-ngram-`<br/>`max-trie-depth`         | `18`       | Type: int          |   实验性         |
| `--speculative-ngram-`<br/>`capacity`              | `10000000` | Type: int          |   实验性         |

## 专家并行(Expert parallelism)

| 参数                                                  | 默认值    | 可选项                                      | 支持的 Server    |
|-------------------------------------------------------|-----------|---------------------------------------------|:----------------:|
| `--expert-parallel-size`<br/>`--ep-size`<br/>`--ep`   | `1`       | Type: int                                   |      A2, A3      |
| `--moe-a2a-backend`                                   | `none`    | `none`,<br/> `deepep`,<br/> `ascend_fuseep` |      A2, A3      |
| `--moe-runner-backend`                                | `auto`    | `auto`, `triton`                            |      A2, A3      |
| `--flashinfer-mxfp4-`<br/>`moe-precision`             | `default` | `default`,<br/> `bf16`                      | GPU 专用         |
| `--enable-flashinfer-`<br/>`allreduce-fusion`         | `False`   | 布尔标志<br/> (设置以启用)                 | GPU 专用         |
| `--deepep-mode`                                       | `auto`    | `normal`, <br/>`low_latency`,<br/> `auto`   |      A2, A3      |
| `--deepep-config`                                     | `None`    | Type: str                                   | GPU 专用         |
| `--ep-num-redundant-experts`                          | `0`       | Type: int                                   |      A2, A3      |
| `--ep-dispatch-algorithm`                             | `None`    | Type: str                                   |      A2, A3      |
| `--init-expert-location`                              | `trivial` | Type: str                                   |      A2, A3      |
| `--enable-eplb`                                       | `False`   | 布尔标志<br/> (设置以启用)                 |      A2, A3      |
| `--eplb-algorithm`                                    | `auto`    | Type: str                                   |      A2, A3      |
| `--eplb-rebalance-layers-`<br/>`per-chunk`            | `None`    | Type: int                                   |      A2, A3      |
| `--eplb-min-rebalancing-`<br/>`utilization-threshold` | `1.0`     | Type: float                                 |      A2, A3      |
| `--expert-distribution-`<br/>`recorder-mode`          | `None`    | Type: str                                   |      A2, A3      |
| `--expert-distribution-`<br/>`recorder-buffer-size`   | `None`    | Type: int                                   |      A2, A3      |
| `--enable-expert-distribution-`<br/>`metrics`         | `False`   | 布尔标志(设置以启用)                     |      A2, A3      |
| `--moe-dense-tp-size`                                 | `None`    | Type: int                                   |      A2, A3      |
| `--elastic-ep-backend`                                | `None`    | `none`, `mooncake`                          | GPU 专用         |
| `--mooncake-ib-device`                                | `None`    | Type: str                                   | GPU 专用         |

## Mamba Cache

| 参数                         | 默认值    | 可选项                                        | 支持的 Server    |
|------------------------------|-----------|-----------------------------------------------|:----------------:|
| `--max-mamba-cache-size`     | `None`    | Type: int                                     |      A2, A3      |
| `--mamba-ssm-dtype`          | `float32` | `float32`,<br/>`bfloat16`,<br/>`float16`      |      A2, A3      |
| `--mamba-full-memory-ratio`  | `0.9`     | Type: float                                   |      A2, A3      |
| `--mamba-scheduler-strategy` | `auto`    | 仅支持 `auto`、`no_buffer`                    |      A2, A3      |
| `--mamba-track-interval`     | `256`     | Type: int                                     |      A2, A3      |

## 分层缓存(Hierarchical cache)

| 参数                                            | 默认值          | 可选项                                                              | 支持的 Server    |
|-------------------------------------------------|-----------------|---------------------------------------------------------------------|:----------------:|
| `--enable-hierarchical-`<br/>`cache`            | `False`         | 布尔标志<br/> (设置以启用)                                         |      A2, A3      |
| `--hicache-ratio`                               | `2.0`           | Type: float                                                         |      A2, A3      |
| `--hicache-size`                                | `0`             | Type: int                                                           |      A2, A3      |
| `--hicache-write-policy`                        | `write_through` | 目前仅支持 `write_back`                                             |      A2, A3      |
| `--hicache-io-backend`                          | `kernel`        | `kernel_ascend`,<br/>                     `direct`                  |      A2, A3      |
| `--hicache-mem-layout`                          | `layer_first`   | `page_first_direct`,<br/>                  `page_first_kv_split`    |      A2, A3      |
| `--hicache-storage-`<br/>`backend`              | `None`          | `file`                                                              |      A2, A3      |
| `--hicache-storage-`<br/>`prefetch-policy`      | `best_effort`   | `best_effort`,<br/> `wait_complete`,<br/>  `timeout`                | GPU 专用         |
| `--hicache-storage-`<br/>`backend-extra-config` | `None`          | Type: str                                                           | GPU 专用         |

## LMCache

| 参数               | 默认值   | 可选项                         | 支持的 Server    |
|--------------------|----------|--------------------------------|:----------------:|
| `--enable-lmcache` | `False`  | 布尔标志<br/> (设置以启用)    | GPU 专用         |

## Offloading

| 参数                      | 默认值   | 可选项    | 支持的 Server    |
|---------------------------|----------|-----------|:----------------:|
| `--cpu-offload-gb`        | `0`      | Type: int |      A2, A3      |
| `--offload-group-size`    | `-1`     | Type: int |      计划中      |
| `--offload-num-in-group`  | `1`      | Type: int |      计划中      |
| `--offload-prefetch-step` | `1`      | Type: int |      计划中      |
| `--offload-mode`          | `cpu`    | Type: str |      计划中      |

## 多条目评分(multi-item scoring)参数

| 参数                             | 默认值   | 可选项    | 支持的 Server    |
|----------------------------------|----------|-----------|:----------------:|
| `--multi-item-scoring-delimiter` | `None`   | Type: int |      A2, A3      |

## 优化/调试选项

| 参数                                                    | 默认值   | 可选项                         | 支持的 Server    |
|---------------------------------------------------------|----------|--------------------------------|:----------------:|
| `--disable-radix-cache`                                 | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--cuda-graph-max-bs`                                   | `None`   | Type: int                      |      A2, A3      |
| `--cuda-graph-bs`                                       | `None`   | List[int]                      |      A2, A3      |
| `--disable-cuda-graph`                                  | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--disable-cuda-graph-`<br/>`padding`                   | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--enable-profile-`<br/>`cuda-graph`                    | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--enable-cudagraph-gc`                                 | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--enable-nccl-nvls`                                    | `False`  | 布尔标志<br/> (设置以启用)    | GPU 专用         |
| `--enable-symm-mem`                                     | `False`  | 布尔标志<br/> (设置以启用)    | GPU 专用         |
| `--disable-flashinfer-`<br/>`cutlass-moe-fp4-allgather` | `False`  | 布尔标志<br/> (设置以启用)    | GPU 专用         |
| `--enable-tokenizer-`<br/>`batch-encode`                | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--disable-tokenizer-`<br/>`batch-decode`               | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--disable-custom-`<br/>`all-reduce`                    | `False`  | 布尔标志<br/> (设置以启用)    | GPU 专用         |
| `--enable-mscclpp`                                      | `False`  | 布尔标志<br/> (设置以启用)    | GPU 专用         |
| `--enable-torch-`<br/>`symm-mem`                        | `False`  | 布尔标志<br/> (设置以启用)    | GPU 专用         |
| `--disable-overlap`<br/>`-schedule`                     | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--enable-mixed-`<br/>`chunk`                           | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--enable-dp-attention`                                 | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--enable-dp-lm-head`                                   | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--enable-two-`<br/>`batch-overlap`                     | `False`  | 布尔标志<br/> (设置以启用)    |     计划中       |
| `--enable-single-`<br/>`batch-overlap`                  | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--tbo-token-`<br/>`distribution-threshold`             | `0.48`   | Type: float                    |     计划中       |
| `--enable-torch-`<br/>`compile`                         | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--enable-torch-`<br/>`compile-debug-mode`              | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--enable-piecewise-`<br/>`cuda-graph`                  | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--piecewise-cuda-`<br/>`graph-tokens`                  | `None`   | Type: JSON<br/> list           |      A2, A3      |
| `--piecewise-cuda-`<br/>`graph-compiler`                | `eager`  | ["eager", "inductor"]          |      A2, A3      |
| `--torch-compile-max-bs`                                | `32`     | Type: int                      |      A2, A3      |
| `--piecewise-cuda-`<br/>`graph-max-tokens`              | `None`   | Type: int                      |      A2, A3      |
| `--torchao-config`                                      | ``       | Type: str                      | GPU 专用         |
| `--enable-nan-detection`                                | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--enable-p2p-check`                                    | `False`  | 布尔标志<br/> (设置以启用)    | GPU 专用         |
| `--triton-attention-`<br/>`reduce-in-fp32`              | `False`  | 布尔标志<br/> (设置以启用)    | GPU 专用         |
| `--triton-attention-`<br/>`num-kv-splits`               | `8`      | Type: int                      | GPU 专用         |
| `--triton-attention-`<br/>`split-tile-size`             | `None`   | Type: int                      | GPU 专用         |
| `--delete-ckpt-`<br/>`after-loading`                    | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--enable-memory-saver`                                 | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--enable-weights-`<br/>`cpu-backup`                    | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--enable-draft-weights-`<br/>`cpu-backup`              | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--allow-auto-truncate`                                 | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--enable-custom-`<br/>`logit-processor`                | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--flashinfer-mla-`<br/>`disable-ragged`                | `False`  | 布尔标志<br/> (设置以启用)    | GPU 专用         |
| `--disable-shared-`<br/>`experts-fusion`                | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--disable-chunked-`<br/>`prefix-cache`                 | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--disable-fast-`<br/>`image-processor`                 | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--keep-mm-feature-`<br/>`on-device`                    | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--enable-return-`<br/>`hidden-states`                  | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--enable-return-`<br/>`routed-experts`                 | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--scheduler-recv-`<br/>`interval`                      | `1`      | Type: int                      |      A2, A3      |
| `--numa-node`                                           | `None`   | List[int]                      |      A2, A3      |
| `--enable-deterministic-`<br/>`inference`               | `False`  | 布尔标志<br/> (设置以启用)    |     计划中       |
| `--rl-on-policy-target`                                 | `None`   | `fsdp`                         |     计划中       |
| `--enable-layerwise-`<br/>`nvtx-marker`                 | `False`  | 布尔标志<br/> (设置以启用)    | GPU 专用         |
| `--enable-attn-tp-`<br/>`input-scattered`               | `False`  | 布尔标志<br/> (设置以启用)    |   实验性         |
| `--enable-nsa-prefill-`<br/>`context-parallel`          | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--enable-fused-qk-`<br/>`norm-rope`                    | `False`  | 布尔标志<br/> (设置以启用)    | GPU 专用         |

## 动态批处理 tokenizer(Dynamic batch tokenizer)

| 参数                                             | 默认值   | 可选项                         | 支持的 Server    |
|--------------------------------------------------|----------|--------------------------------|:----------------:|
| `--enable-dynamic-`<br/>`batch-tokenizer`        | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--dynamic-batch-`<br/>`tokenizer-batch-size`    | `32`     | Type: int                      |      A2, A3      |
| `--dynamic-batch-`<br/>`tokenizer-batch-timeout` | `0.002`  | Type: float                    |      A2, A3      |

## 调试张量转储(Debug tensor dumps)

| 参数                                       | 默认值   | 可选项    | 支持的 Server    |
|--------------------------------------------|----------|-----------|:----------------:|
| `--debug-tensor-dump-`<br/>`output-folder` | `None`   | Type: str |      A2, A3      |
| `--debug-tensor-dump-`<br/>`layers`        | `None`   | List[int] |      A2, A3      |
| `--debug-tensor-dump-`<br/>`input-file`    | `None`   | Type: str |      A2, A3      |

## PD 分离(PD disaggregation)

| 参数                                                    | 默认值     | 可选项                                | 支持的 Server    |
|---------------------------------------------------------|------------|---------------------------------------|:----------------:|
| `--disaggregation-mode`                                 | `null`     | `null`,<br/> `prefill`,<br/> `decode` |      A2, A3      |
| `--disaggregation-transfer-backend`                     | `mooncake` | `ascend`                              |      A2, A3      |
| `--disaggregation-bootstrap-port`                       | `8998`     | Type: int                             |      A2, A3      |
| `--disaggregation-ib-device`                            | `None`     | Type: str                             | GPU 专用         |
| `--disaggregation-decode-`<br/>`enable-offload-kvcache` | `False`    | 布尔标志<br/> (设置以启用)           |      A2, A3      |
| `--num-reserved-decode-tokens`                          | `512`      | Type: int                             |      A2, A3      |
| `--disaggregation-decode-`<br/>`polling-interval`       | `1`        | Type: int                             |      A2, A3      |

## 编码预填充分离(Encode prefill disaggregation)

| 参数                         | 默认值             | 可选项                                                         | 支持的 Server    |
|------------------------------|--------------------|----------------------------------------------------------------|:----------------:|
| `--encoder-only`             | `False`            | 布尔标志<br/> (设置以启用)                                    |      A2, A3      |
| `--language-only`            | `False`            | 布尔标志<br/> (设置以启用)                                    |      A2, A3      |
| `--encoder-transfer-backend` | `zmq_to_scheduler` | `zmq_to_scheduler`, <br/> `zmq_to_tokenizer`,<br/>  `mooncake` |      A2, A3      |
| `--encoder-urls`             | `[]`               | List[str]                                                      |      A2, A3      |

## 自定义权重加载器(Custom weight loader)

| 参数                                                                    | 默认值   | 可选项                          | 支持的 Server    |
|-------------------------------------------------------------------------|----------|---------------------------------|:----------------:|
| `--custom-weight-loader`                                                | `None`   | List[str]                       |      A2, A3      |
| `--weight-loader-disable-`<br/>`mmap`                                   | `False`  | 布尔标志<br/> (设置以启用)     |      A2, A3      |
| `--remote-instance-weight-`<br/>`loader-seed-instance-ip`               | `None`   | Type: str                       |      A2, A3      |
| `--remote-instance-weight-`<br/>`loader-seed-instance-service-port`     | `None`   | Type: int                       |      A2, A3      |
| `--remote-instance-weight-`<br/>`loader-send-weights-group-ports`       | `None`   | Type: JSON<br/> list            |      A2, A3      |
| `--remote-instance-weight-`<br/>`loader-backend`                        | `nccl`   | `transfer_engine`, <br/> `nccl` |      A2, A3      |
| `--remote-instance-weight-`<br/>`loader-start-seed-via-transfer-engine` | `False`  | 布尔标志<br/> (设置以启用)     | GPU 专用         |

## 用于 PD-Multiplexing

| 参数                  | 默认值   | 可选项                         | 支持的 Server    |
|-----------------------|----------|--------------------------------|:----------------:|
| `--enable-pdmux`      | `False`  | 布尔标志<br/> (设置以启用)    | GPU 专用         |
| `--pdmux-config-path` | `None`   | Type: str                      | GPU 专用         |
| `--sm-group-num`      | `8`      | Type: int                      | GPU 专用         |

## 用于多模态(Multi-Modal)

| 参数                                          | 默认值   | 可选项                         | 支持的 Server    |
|-----------------------------------------------|----------|--------------------------------|:----------------:|
| `--mm-max-concurrent-calls`                   | `32`     | Type: int                      |      A2, A3      |
| `--mm-per-request-timeout`                    | `10.0`   | Type: float                    |      A2, A3      |
| `--enable-broadcast-mm-`<br/>`inputs-process` | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--mm-process-config`                         | `None`   | Type: JSON / Dict              |      A2, A3      |
| `--mm-enable-dp-encoder`                      | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |
| `--limit-mm-data-per-request`                 | `None`   | Type: JSON / Dict              |      A2, A3      |

## 用于 checkpoint 解密

| 参数                            | 默认值   | 可选项                         | 支持的 Server    |
|---------------------------------|----------|--------------------------------|:----------------:|
| `--decrypted-config-file`       | `None`   | Type: str                      |      A2, A3      |
| `--decrypted-draft-config-file` | `None`   | Type: str                      |      A2, A3      |
| `--enable-prefix-mm-cache`      | `False`  | 布尔标志<br/> (设置以启用)    |      A2, A3      |

## 前向钩子(Forward hooks)

| 参数              | 默认值   | 可选项          | 支持的 Server    |
|-------------------|----------|-----------------|:----------------:|
| `--forward-hooks` | `None`   | Type: JSON list |      A2, A3      |

## 配置文件支持

| 参数       | 默认值   | 可选项    | 支持的 Server    |
|------------|----------|-----------|:----------------:|
| `--config` | `None`   | Type: str |      A2, A3      |

## 其他参数

以下参数不受支持,因为它们所依赖的第三方组件与 NPU 不兼容,例如
Ktransformer、checkpoint-engine 等。

| 参数                                                              | 默认值    | 可选项                    |
|-------------------------------------------------------------------|-----------|---------------------------|
| `--checkpoint-engine-` <br/> `wait-weights-` <br/> `before-ready` | `False`   | 布尔标志(设置以启用)   |
| `--kt-weight-path`                                                | `None`    | Type: str                 |
| `--kt-method`                                                     | `AMXINT4` | Type: str                 |
| `--kt-cpuinfer`                                                   | `None`    | Type: int                 |
| `--kt-threadpool-count`                                           | `2`       | Type: int                 |
| `--kt-num-gpu-experts`                                            | `None`    | Type: int                 |
| `--kt-max-deferred-`<br/>`experts-per-token`                      | `None`    | Type: int                 |
