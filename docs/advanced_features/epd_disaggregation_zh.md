# EPD 分离(EPD Disaggregation)

## 为什么需要 EPD 分离?它是什么?

在现代视觉-语言模型(Vision-Language Model,VLM)推理中,请求的执行天然地分解为三个不同的阶段:Encoder、Prefill 和 Decode。
Encoder 阶段执行视觉预处理和基于 ViT 的图像编码,这是高度计算密集型的,但仅在请求初始化时才需要。Prefill 阶段处理完整的多模态输入序列,以初始化语言模型的 Key-Value(KV)缓存;而 Decode 阶段则以内存带宽和 KV 缓存访问为主导,用于自回归式的 token 生成。

现有的部署通常将这三个阶段并置(colocate)在一个统一的执行引擎中,或者最多应用 Prefill–Decode(PD)分离。然而,这类设计仍然将视觉编码与语言 prefill 紧密耦合,导致资源利用低效、对图像密集型工作负载的可扩展性受限,以及在负载下的调度不够理想。

为了解决这些挑战,我们在 SGLang 中引入了 Encoder–Prefill–Decode(EPD)分离。EPD 进一步将视觉编码与语言处理分离开来,实现了 encoder 服务器的独立横向扩展、对多模态请求更好的负载均衡,以及与现有 PD 分离的无缝集成,从而形成一个完全解耦的三层推理架构。

### 用法

你可以使用 `--language-only` 启动一个仅语言模型,或使用 `--encoder-only` 启动一个仅编码器模型。
当启动仅语言模型时,你必须通过 `--encoder-urls` 额外指定 encoder 服务的端点。

我们支持多种 encoder 传输后端,包括 zmq_to_scheduler、zmq_to_tokenizer 和 mooncake(默认为 zmq_to_scheduler)。可以使用 `--encoder-transfer-backend` 选择后端。

### 使用 Mooncake 进行 Encoder 传输

`--encoder-transfer-backend mooncake` 控制 encoder 输出**在 encoder 和 language/prefill 服务之间如何传输**。它是一个 encoder 传输选项,可以独立于全局多模态 embedding 缓存使用。

示例:

```bash
# encoder
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-transfer-backend mooncake \
  --port 30000

# language-only server
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --language-only \
  --encoder-urls http://127.0.0.1:30000 \
  --encoder-transfer-backend mooncake \
  --port 30002
```

### 基于 Mooncake 的全局多模态 embedding 缓存

SGLang 还为 EPD 工作负载支持一个由 Mooncake 支撑的**全局多模态 embedding 缓存**。当在 encoder 服务器上启用时,重复的图像输入可以跨实例复用之前计算好的 ViT embedding,而无需再次运行视觉编码器。

此功能在以下情况下很有用:

- 部署服务于重复或重叠的图像输入,
- encoder 计算成为瓶颈,以及
- 集群中已经具备 Mooncake。

从宏观上看,encoder 会检查图像 embedding 是否已存在于 Mooncake 中。缓存命中的会从全局存储中预取,而未命中的则正常编码,并在后台插入缓存。

要启用它:

- 以与其他 SGLang Mooncake 集成相同的方式安装和配置 Mooncake,
- 在 encoder 服务器上添加 `--enable-mm-global-cache`。

`--enable-mm-global-cache` 控制**多模态 embedding 是否在全局 Mooncake 缓存中查找和存储**。它与 `--encoder-transfer-backend` 是分开的,后者只控制 encoder 输出的传输。

有关 Mooncake 的部署和配置细节,请参见 [HiCache 最佳实践](hicache_best_practices.md#deployment-with-mooncake) 和 [Mooncake 后端 README](../../python/sglang/srt/mem_cache/storage/mooncake_store/README.md)。

示例:

```bash
# Shared Mooncake configuration
export MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata"
export MOONCAKE_MASTER="127.0.0.1:50051"
export MOONCAKE_PROTOCOL="rdma"
export MOONCAKE_GLOBAL_SEGMENT_SIZE="4gb"

# encoder with global multimodal cache enabled
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --enable-mm-global-cache \
  --port 30000

# language-only server
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --language-only \
  --encoder-urls http://127.0.0.1:30000 \
  --port 30002
```

注意:

- 此缓存用于**多模态编码器 embedding**,而非语言模型的 KV 缓存。
- 该功能目前使用 Mooncake 作为共享的后端存储。
- 无论你使用哪种 `--encoder-transfer-backend`,都可以启用它。
- 它与 EPD 或 encoder 分离式 VLM 部署最为相关——在这些部署中,相同的图像很可能跨请求或跨实例出现。

#### Qwen VL

- EP 分离

```bash
# encoder 0
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30000
# encoder 1
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30001
# language-only server
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --language-only \
  --encoder-urls http://127.0.0.1:30000 http://127.0.0.1:30001 \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30002
```

- EPD 分离

```bash
# encoder 0
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30000
# encoder 1
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30001
# prefill 0
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --disaggregation-mode prefill \
  --language-only \
  --encoder-urls http://127.0.0.1:30000 http://127.0.0.1:30001 \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30002
# decode 0
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --disaggregation-mode decode \
  --port 30003
# router
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://$PREFILL_HOST:30002 \
  --decode http://$DECODE_HOST:30003 \
  --port 8000

```

#### gRPC Encoder(EPD)

你可以将 encoder 作为 gRPC 服务器运行,同时保持 prefill/decode 为 HTTP。
当使用 gRPC encoder 时,为 prefill 进程设置 `SGLANG_ENCODER_MM_RECEIVER_MODE=grpc`,使其使用 gRPC 接收器。

```bash
# gRPC encoder
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --grpc-mode \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30000

# prefill (HTTP) - tell it to use gRPC receiver
SGLANG_ENCODER_MM_RECEIVER_MODE=grpc \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --disaggregation-mode prefill \
  --language-only \
  --encoder-urls grpc://127.0.0.1:30000 \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30002

# decode (HTTP)
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --disaggregation-mode decode \
  --port 30003

# router
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://$PREFILL_HOST:30002 \
  --decode http://$DECODE_HOST:30003 \
  --port 8000
```
