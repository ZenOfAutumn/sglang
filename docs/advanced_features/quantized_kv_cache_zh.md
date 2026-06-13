# 量化 KV 缓存(Quantized KV Cache)

量化 KV 缓存通过使用较低精度的数据类型(FP8 或 FP4)代替默认的 BF16 模型精度,来减少键值缓存(key-value cache)存储的内存占用。在自回归生成过程中,LLM 会缓存先前计算的键值对,以避免冗余计算。KV 缓存通常会消耗相当大一部分 GPU 内存,尤其是对于长序列。

量化 KV 缓存是一种内存优化技术,它通过允许缓存更多 token 来主要使吞吐量受益,但可能会引入极小的精度下降,具体取决于所使用的量化格式。

```{warning}
**性能警告**:当量化 KV 缓存必须在用于注意力运算之前先解量化(dequantize)时,如果解量化未与注意力 kernel 融合,性能可能会极其缓慢。请始终验证你所选的注意力后端是否支持量化 KV 缓存。不具备融合支持的后端可能会经历显著的吞吐量下降,这可能会抵消内存收益。

**后端支持**:并非所有注意力后端都支持量化 KV 缓存。关于哪些后端支持它,请参阅 [Attention Backend](attention_backend.md)。
```

## 支持的格式

SGLang 支持以下量化 KV 缓存格式:

### FP8 格式

[OCP(Open Compute Project)](https://www.opencompute.org) 规定了两种常见的 8 位浮点格式:

- **E5M2**(5 个指数位,2 个尾数位):动态范围更大(±57344.0),精度更低
- **E4M3**(4 个指数位,3 个尾数位):精度更高,动态范围更小(±240.0)

### FP4 格式

```{warning}
FP4 量化目前是实验性的。
```

[OCP(Open Compute Project)](https://www.opencompute.org) 规定了 MXFP4(Microscaling FP4),一种 4 位浮点格式:

- **E2M1**(1 个符号位,2 个指数位,1 个尾数位):使用基于块的微缩放(microscaling),将张量划分为由连续元素组成的块,每个块共享一个 8 位的指数缩放因子。虽然 OCP 规定块为 32 个元素,但 SGLang 当前的实现对 KV 缓存量化使用 16 个元素的块。

## 用法

### 启用量化 KV 缓存

要启用量化 KV 缓存,在启动服务器时使用 `--kv-cache-dtype` 参数:

```bash
# Enable FP8 E5M2 KV cache
python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-R1-0528 \
    --kv-cache-dtype fp8_e5m2 \

# Enable FP8 E4M3 KV cache
python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-R1-0528 \
    --kv-cache-dtype fp8_e4m3 \

# Enable FP4 E2M1 KV cache
python3 -m sglang.launch_server \
    --model-path nvidia/DeepSeek-R1-0528-NVFP4 \
    --kv-cache-dtype fp4_e2m1 \
```

### 缩放因子(Scaling Factors)

FP8 量化需要缩放因子来正确地量化和解量化 KV 缓存。

```{note}
目前仅支持 per-tensor(标量)缩放因子。
```

缩放因子可以:

- **从检查点加载**:预量化模型(例如 ModelOpt)可能包含 `k_scale` 和 `v_scale` 参数,它们会被自动加载
- **通过 JSON 提供**:通过 `--quantization-param-path` 提供缩放因子。

JSON 文件应遵循以下格式:

```json
{
  "kv_cache": {
    "dtype": "float8_e4m3fn",
    "scaling_factor": {
      "0": {
        "0": 1.0,
        "1": 1.0
      }
    }
  }
}
```

其中 `scaling_factor` 的外层 key 是张量并行的 rank,内层 key 是层索引。

```{warning}
如果未提供缩放因子且在检查点中也未找到,它将默认为 1.0,这可能会导致精度问题。
```

```{tip}
**FP4(MXFP4)**:与 FP8 不同,FP4 量化在量化和解量化期间会即时(on-the-fly)自动处理缩放因子。不需要预量化模型或外部缩放因子文件——基于块的缩放因子会按需动态计算。
```

## 性能考量

### 内存节省

量化 KV 缓存提供显著的内存节省:
- **BF16 → FP4**:支持的 token 数量约为 BF16 的 3.56 倍(已计入缩放因子的开销)

```{note}
FP4 和 FP8 量化需要额外的内存用于基于块的缩放因子,这会降低相对于原始位宽缩减的有效内存节省。块大小为 16 的 FP4 支持的 token 数量约为 FP8 的 1.78 倍,约为 BF16 的 3.56 倍。FP8 和 BF16 之间的相对 token 容量可以由这些比例推导得出。
```

这使得在相同的内存预算内能够实现更长的上下文长度或更多的并发请求。

### 精度影响

#### FP8 精度

FP8 E4M3 量化通常引入极小的精度下降。其影响取决于模型架构、序列长度和量化格式(一般而言,E4M3 比 E5M2 精度更好)。

#### FP4 精度

FP4(MXFP4)量化提供显著的内存节省,精度影响因模型大小和数据集复杂度而异。来自 [PR #10078](https://github.com/sgl-project/sglang/pull/10078)(MLA)和 [PR #12612](https://github.com/sgl-project/sglang/pull/12612)(MHA)的初步精度测试结果显示:

**大模型(例如 Qwen3-235B-A22B、DeepSeek-R1-0528)**

在大规模模型上,FP4 保持了接近 FP8/BF16 的精度,尤其是在较简单的数据集上:

| Model | Dataset | KV16 | KV8 (FP8 E4M3) | KV4 (FP4 E2M1) |
|-------|---------|------|----------------|----------------|
| Qwen3-235B-A22B | gsm8k | 0.9168 | 0.9181 | 0.9186 |
| Qwen3-235B-A22B | aime25 | 0.7733 | 0.7333 | 0.6000 |
| Qwen3-235B-A22B | gpqa_diamond | 0.7010 | 0.6899 | 0.6778 |
| DeepSeek-R1-0528 | gsm8k | 0.9157 | 0.9154 | 0.9124 |
| DeepSeek-R1-0528 | aime25 | 0.5067 | 0.4934 | 0.4000 |
| DeepSeek-R1-0528 | gpqa_diamond | 0.7707 | 0.7697 | 0.7273 |

**较小模型(例如 GPT-OSS-120B)**

在较小的模型上,FP4 表现出更明显的精度下降,尤其是在具有挑战性的数据集上:

| Model | Dataset | KV16 | KV8 (FP8 E4M3) | KV4 (FP4 E2M1) |
|-------|---------|------|----------------|----------------|
| GPT-OSS-120B | gsm8k | 0.9161 | 0.9163 | 0.9152 |
| GPT-OSS-120B | aime25 | 0.7533 | 0.7667 | 0.3533 |
| GPT-OSS-120B | gpqa_diamond | 0.5081 | 0.5434 | 0.3202 |

**关键观察:**

- **简单数据集(例如 gsm8k)**:FP4 在各种模型大小上都保持了接近 FP8/BF16 的精度
- **模型大小很重要**:大模型(200B+ 参数)通常比小模型更能容忍 FP4 量化
- **上下文长度**:在长上下文场景中,精度下降可能更明显,因为量化误差的累积可能变得显著。

```{tip}
请在你的特定模型和工作负载上评估 FP4 精度。大模型在较简单的任务上通常表现出极小的下降,而较小的模型或复杂的推理任务可能需要 FP8 或 BF16 才能达到可接受的精度。
```

## 最佳实践

- **使用预量化模型**:优先选择离线量化且检查点中包含缩放因子的模型。
- **选择合适的格式**:使用 `fp8_e4m3` 以获得更好的精度(推荐),使用 `fp8_e5m2` 以获得更大的动态范围,或使用 `fp4_e2m1` 以获得最大的内存节省(实验性)
- **检查后端兼容性**:验证你所选的注意力后端是否支持量化 KV 缓存

```{seealso}
- [Quantization](quantization.md)
- [Attention Backend](attention_backend.md)
- [Server Arguments](server_arguments.md)
```
