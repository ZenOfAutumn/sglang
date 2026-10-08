# KV Cache 容量计算与评估（以 DeepSeek-V4-Pro 为例）

> 本文系统讲解 SGLang 中 **KV cache 容量如何计算、如何评估、如何调优**，并以 **DeepSeek-V4-Pro**（MLA + DSA 稀疏注意力架构）为完整算例。
>
> 所有参数与存储布局均对应当前仓库代码：
> - 模型配置：`python/sglang/srt/configs/deepseek_v4.py`
> - 物理存储：`python/sglang/srt/mem_cache/memory_pool.py`（`MLATokenToKVPool` / `DSATokenToKVPool`）
> - 容量推导：`python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`

---

## 0. 一句话结论

**KV cache 总容量（可缓存 token 数）= 可用显存 ÷ 每个 token 的 KV 字节数**。

```
可用显存 = 总显存 × mem_fraction_static − 模型权重 − 激活/中间缓冲 − CUDA Graph 等开销
每 token 字节数 = Σ_layer (该层每 token 的 KV 字节)
max_total_tokens = 可用显存 // 每 token 字节数
```

DeepSeek-V4-Pro 因为采用 **MLA（压缩 latent KV）**，每 token 的 KV 字节数远小于标准 MHA 模型，因此**同样显存能缓存的 token 数大得多**——这正是它能跑超长上下文的根本原因。

---

## 1. 背景：KV cache 容量由什么决定

### 1.1 两级内存池回顾

SGLang 用两级映射管理 KV（详见 `mem_cache/README_zh.md`）：

| 层级 | 结构 | 容量含义 |
| ---- | ---- | -------- |
| 第 2 级 `req_to_token` | `[max_running_requests, max_context_len]` 的 int 表 | 决定**并发请求数 × 单请求最大长度**的索引上限 |
| 第 1 级 `KVCache` 物理张量 | 每层 `[size + page_size, ...]` | 真正占显存的主体，`size` = **可缓存 token 总数** |

> 容量评估的核心目标，就是求出第 1 级的 **`size`（= `max_total_tokens`，可缓存 token 总数）**。

### 1.2 容量推导公式（代码视角）

`model_runner_kv_cache_mixin.py` 中 `profile_max_num_token` 的逻辑可概括为：

```
available_kv_bytes = total_gpu_memory * mem_fraction_static
                     - model_weights_bytes
                     - activation / cuda_graph / misc 开销

cell_size = 每个 token 在「所有层」上占的 KV 字节数

max_total_tokens = available_kv_bytes // cell_size
```

其中 **`cell_size` 完全由模型架构决定**，是本文计算的重点。

---

## 2. DeepSeek-V4-Pro 架构参数

取自 `configs/deepseek_v4.py`（`DeepSeekV4Config` 默认值）：

| 参数 | 值 | 说明 |
| ---- | -- | ---- |
| `num_hidden_layers` | **43** | 总层数（含 MLA 注意力层） |
| `kv_lora_rank` | **512** | MLA 压缩后的 latent KV 维度（NoPE 部分） |
| `qk_rope_head_dim` | **64** | MLA 的 RoPE 部分维度 |
| `kv_cache_dim` | **576** | = `kv_lora_rank + qk_rope_head_dim` = 512 + 64 |
| `num_key_value_heads` | **1** | MLA 本质上是「单头 latent」，不再按 KV 头数乘 |
| `index_head_dim` | **128** | DSA indexer 的 key 维度 |
| `index_n_heads` | **64** | DSA indexer 头数（仅用于选择 top-k，不进主 KV） |
| `index_topk` | **512** | DSA 每步保留的稀疏 token 数 |
| `max_position_embeddings` | **65536** | 最大上下文长度 |

> **关键洞察**：MLA 不像 MHA 那样存「每个 KV 头的完整 K、V」，而是把整层 KV **压缩成一个 576 维的 latent 向量**（`num_key_value_heads = 1`）。这是 DeepSeek 系显存效率的来源。

---

## 3. 每 token KV 字节数（`cell_size`）计算

DeepSeek-V4-Pro 的 KV cache 分**两部分**（对应 `DSATokenToKVPool`，它继承 `MLATokenToKVPool`）：

### 3.1 主 KV：MLA latent（`MLATokenToKVPool`）

物理张量布局（`memory_pool.py:2140`）：

```
kv_buffer[layer]  =  zeros([size + page_size, 1, kv_cache_dim])
                                              ↑   ↑
                                    单头 latent  576 维
```

每 token 每层的主 KV 字节：

```
bytes_mla_per_token_per_layer = kv_cache_dim × dtype_bytes
                              = 576 × dtype_bytes
```

| dtype | dtype_bytes | 每 token 每层 | × 43 层 = 每 token 主 KV |
| ----- | ----------- | ------------- | ------------------------ |
| bf16  | 2           | 1152 B        | **49,536 B ≈ 48.4 KB**   |
| fp8   | 1           | 576 B         | **24,768 B ≈ 24.2 KB**   |

> DeepSeek-V4 在 DSA 模式下主 latent 常以 **fp8** 存储（`DSATokenToKVPool` 的 `override_kv_cache_dim` + fp8 路径），RoPE 部分始终 bf16（`rope_storage_dtype = torch.bfloat16`）。下面算例取 **bf16 上界**便于对比，fp8 时约减半。

### 3.2 DSA indexer KV（`DSATokenToKVPool`）

DSA 额外维护一份 indexer key cache（用于稀疏 top-k 选择），布局（`memory_pool.py:2532`）：

```
index_k_with_scale_buffer[layer] = zeros(
    [num_pages, page_size × (index_head_dim + index_head_dim/quant_block_size × 4)]
)
其中 quant_block_size = 128, index_head_dim = 128
每 token 每层 = 128 (fp8 数据, 1B/元素) + 128/128 × 4 (fp32 scale) = 128 + 4 = 132 B
```

每 token 每层 indexer 字节：

```
bytes_index_per_token_per_layer = 132 B（fp8 data 128B + fp32 scale 4B）
× 43 层 = 5,676 B ≈ 5.5 KB
```

### 3.3 合计 `cell_size`

| 组成 | 每 token（43 层） |
| ---- | ----------------- |
| 主 MLA latent（bf16） | 49,536 B |
| DSA indexer（fp8+scale） | 5,676 B |
| **cell_size（bf16 latent）** | **≈ 55,212 B ≈ 53.9 KB / token** |
| **cell_size（fp8 latent）** | **≈ 30,444 B ≈ 29.7 KB / token** |

> 对比：一个标准 70B MHA 模型（80 层、8 KV 头、head_dim 128、bf16）每 token KV ≈ `80 × 2(K/V) × 8 × 128 × 2 = 327,680 B ≈ 320 KB`。**MLA 把每 token KV 压到约 1/6 ~ 1/10**，这就是超长上下文的底气。

---

## 4. 完整算例：单卡 H200（141 GB）部署

假设 **8×H200（141 GB/卡）** TP=8 部署 DeepSeek-V4-Pro，权重 fp8。

### 4.1 单卡显存预算

| 项目 | 估算 | 说明 |
| ---- | ---- | ---- |
| 单卡总显存 | 141 GB | H200 |
| `mem_fraction_static` | 0.9 | 默认 ~0.9，给 KV+权重的静态占比 |
| 静态可用 | ≈ 127 GB | 141 × 0.9 |
| 模型权重（/卡） | ≈ 671B fp8 / 8 ≈ 84 GB → TP 分摊后约 **80–85 GB** | 671B 总参 fp8 ≈ 671 GB / 8 |
| CUDA Graph + 激活 + 通信缓冲 | ≈ 8–12 GB | 经验值 |
| **可用 KV 显存（/卡）** | **≈ 30 GB** | 127 − 85 − 12 |

> 注：DeepSeek-V4-Pro 为 MoE，单卡权重取决于 EP/TP 切分方式，这里取量级估算。实际以启动日志 `KV Cache is allocated ...` 为准。

### 4.2 可缓存 token 数

```
max_total_tokens ≈ 可用KV显存 / cell_size
```

| latent dtype | cell_size | 30 GB 可缓存 token 数（单卡） | TP=8 全局 |
| ------------ | --------- | ----------------------------- | --------- |
| bf16 | 53.9 KB | 30 GB / 53.9 KB ≈ **583K** | KV 在 TP 间复制/切分依实现而定 |
| fp8  | 29.7 KB | 30 GB / 29.7 KB ≈ **1.06M** | —— |

> 即单卡这 30 GB 在 fp8 latent 下可缓存约 **100 万 token**，足以支撑「64K 上下文 × 十几路并发」或「少数超长会话」。

### 4.3 换算成「并发 × 上下文」

```
可承载并发数 ≈ max_total_tokens / 平均每请求 token 数
```

以 fp8、单卡 ~1.06M token 为例：

| 场景 | 平均序列长度 | 可并发请求数 |
| ---- | ------------ | ------------ |
| 短对话 | 2K | ≈ 530 |
| 中等 RAG | 8K | ≈ 132 |
| 长文档 | 32K | ≈ 33 |
| 满上下文 | 64K | ≈ 16 |

> 这里是**理论上界**；实际受 `max_running_requests`、调度水位、`req_to_token` 表的 `max_context_len` 列宽等共同限制，取最小值。

---

## 5. 评估与验证方法

### 5.1 看启动日志（最直接）

服务启动时会打印实际分配结果：

```
KV Cache is allocated. dtype: ..., #tokens: <max_total_tokens>, KV size: <X> GB
```

- `#tokens` 就是算出的 `max_total_tokens`。
- 与本文公式对账即可验证 `cell_size` 估算是否准确。

### 5.2 反推校验

```
cell_size_实测 = KV size(GB) × 1024^3 / #tokens
```

把实测 `cell_size` 与第 3.3 节理论值对比，偏差应在合理范围（含 indexer、padding 页、对齐开销）。

### 5.3 容量是否够用的判断

| 信号 | 含义 | 处理 |
| ---- | ---- | ---- |
| 频繁 `evict` / cache miss 升高 | KV 容量不足，前缀被反复淘汰 | 调大 `mem_fraction_static` 或减小权重占用 |
| 请求排队、`#queue` 高 | token 预算不足以组批 | 增卡 / 降并发 / 缩上下文 |
| OOM | 估算偏乐观或开销低估 | 调小 `mem_fraction_static` |

---

## 6. 影响容量的可调参数

| 参数 | 作用 | 对容量的影响 |
| ---- | ---- | ------------ |
| `--mem-fraction-static` | 静态显存占比 | ↑ → KV 可用显存↑ → token 数↑（过高易 OOM） |
| `--kv-cache-dtype`（fp8） | KV 存储精度 | fp8 latent 使 `cell_size` 约减半 → token 数翻倍 |
| `--context-length` | 单请求最大长度 | 决定 `req_to_token` 列宽与单请求占用，不改总池但限制单请求 |
| `--max-running-requests` | 最大并发 | 限制 `req_to_token` 行数，与 token 池共同设上限 |
| `--page-size` | 分页粒度 | 影响碎片与对齐，间接影响有效容量（见 README_zh） |
| `--enable-hierarchical-cache` | HiCache 分层 | 把冷 KV 卸载到 host/storage，**扩展可缓存总量**（不增显存） |

---

## 7. 通用计算公式（适用于任意模型）

### 7.1 标准 MHA / GQA

```
cell_size = num_layers × 2(K+V) × num_kv_heads × head_dim × dtype_bytes
max_total_tokens = available_kv_bytes // cell_size
```

### 7.2 MLA（DeepSeek 系）

```
cell_size = num_layers × (kv_lora_rank + qk_rope_head_dim) × dtype_bytes
            [ + DSA: num_layers × (index_head_dim + index_head_dim/quant_block × 4) ]
```

### 7.3 一般化

```
cell_size = Σ_layer (该层每 token 的所有 KV 张量字节)
可用KV显存 = 总显存 × mem_fraction_static − 权重 − 激活 − 图/通信开销
max_total_tokens = 可用KV显存 // cell_size
可并发数 ≈ max_total_tokens / 平均序列长度
```

---

## 8. 小结

1. **容量 = 可用显存 ÷ 每 token KV 字节数**，后者完全由模型架构决定。
2. DeepSeek-V4-Pro 用 **MLA 把每层 KV 压成 576 维 latent（单头）**，加上 **DSA indexer（约 132 B/token/层）**，`cell_size` 约 **30–54 KB/token**（fp8/bf16），仅为同规模 MHA 的 1/6~1/10。
3. **fp8 latent** 几乎可让可缓存 token 数翻倍；**HiCache** 可在不增显存的前提下扩展总量。
4. 评估务必以**启动日志的 `#tokens`** 反推校验，并结合 `evict`/排队/OOM 信号判断是否够用。

> 相关阅读：`mem_cache/README_zh.md`（两级内存池与分配器）、`model_runner_kv_cache_mixin.py`（`max_context_len` 与显存 profiling）。

