# `dp_attention.py` 详细说明

> 对应源码：`python/sglang/srt/layers/dp_attention.py`
>
> 相关文档：`docs/theory/distributed/DP_attention.md`（原理）、`docs/theory/transformer/MLA.md`（MLA）、`python/sglang/srt/managers/README_data_parallel_controller_zh.md`（副本级 DP）

---

## 1. 这个文件解决什么问题

### 1.1 动机：KV Cache 冗余

在 MoE 类模型（DeepSeek-V3 / V3.2 等）中，参数量绝大部分集中在 MoE FFN，而 Attention 部分（尤其是 MLA）的权重很小。若对 Attention 也做纯 TP 切分：

- 每个 TP rank 都要保存**同一批请求**的 KV Cache；
- MLA 的 KV latent 本身已经很小（每 token 每层仅 576 维），按 `tp_size` 再切分后每片过小，kernel 效率低；
- 更关键的是，很多实现里 KV 会在各 rank 间**重复存储**，显存被白白浪费 $tp\_size$ 倍。

**DP Attention 的思路**：Attention 阶段不切模型、切数据。

| 阶段 | 并行方式 | 每个 rank 看到的数据 |
|------|----------|----------------------|
| Attention | DP（数据并行） | 只有自己那一批 token / 请求 |
| MoE / Dense FFN | TP / EP | 需要**全局所有** token |

于是在两个阶段之间必须插入一对通信原语：

```
Attention(local tokens)  --dp_gather-->  FFN/MoE(global tokens)  --dp_scatter-->  Attention(local tokens)
```

本文件提供的就是这一整套「全局状态 + 缓冲区管理 + gather/scatter 实现」。

### 1.2 与「副本级 DP」的区别

务必区分两个同名不同义的 DP：

| 维度 | 副本级 DP（`--dp-size` + `DataParallelController`） | DP Attention（`--enable-dp-attention`） |
|------|--------------------------------------------------|----------------------------------------|
| 切分对象 | **整个模型副本**，各副本完全独立 | 只有 Attention 阶段，切 **token** |
| 通信 | 副本之间无通信 | 每层都要 gather / scatter |
| 权重 | 每个副本一份完整权重 | 同一份权重，rank 间共享 TP 组 |
| 目的 | 提升吞吐（多副本负载均衡） | 降低 KV Cache 显存冗余 |

本文件只负责后者。

---

## 2. 并行维度的划分

### 2.1 三维分解

全局 TP 组被分解为三个正交维度：

$$
tp\_size = attn\_dp\_size \times attn\_cp\_size \times attn\_tp\_size
$$

rank 布局采用 **(dp, cp, tp)** 顺序，其中 `tp` 是变化最快的维度：

$$
tp\_rank = (attn\_dp\_rank \times attn\_cp\_size + attn\_cp\_rank) \times attn\_tp\_size + attn\_tp\_rank
$$

反解即 `compute_dp_attention_world_info`：

```python
attn_dp_size = dp_size if enable_dp_attention else 1
attn_tp_size = tp_size // attn_dp_size // attn_cp_size
attn_tp_rank = tp_rank % attn_tp_size
attn_dp_rank = tp_rank // (attn_tp_size * attn_cp_size)
```

**示例**：`tp_size=8, dp_size=4, attn_cp_size=1` → `attn_tp_size=2`

| `tp_rank` | `attn_dp_rank` | `attn_tp_rank` |
|-----------|----------------|----------------|
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 2 | 1 | 0 |
| 3 | 1 | 1 |
| 4 | 2 | 0 |
| 5 | 2 | 1 |
| 6 | 3 | 0 |
| 7 | 3 | 1 |

解读：8 张卡分成 4 个 DP 组，每组 2 张卡内部再做 TP。KV Cache 只在组内复制 2 份，而不是 8 份，显存冗余降低到 $1/4$。

### 2.2 `local` 系列：`moe_dense_tp_size` 场景

MoE 模型里除了专家层还有 **dense 层**（共享专家 / 稠密 FFN）。这些层参数量小，用全局 TP 切分会导致每片过小、通信占比过高，因此可以用 `--moe-dense-tp-size` 指定一个更小的 TP 度。

这样全局 rank 被切成若干个大小为 `moe_dense_tp_size` 的**子组**，每个子组内部又形成一套独立的 DP/TP 划分。`compute_dp_attention_local_info` 计算的就是这套「局部坐标」：

```python
local_tp_size  = moe_dense_tp_size or tp_size          # 子组大小
local_tp_rank  = tp_rank % local_tp_size               # 子组内偏移
local_dp_size  = max(1, dp_size // (tp_size // local_tp_size))  # 子组内 DP 度

local_attn_tp_size = local_tp_size // local_dp_size
local_attn_dp_rank = local_tp_rank // local_attn_tp_size
local_attn_tp_rank = local_tp_rank % local_attn_tp_size
```

对应全局状态 `_LOCAL_ATTN_DP_RANK` / `_LOCAL_ATTN_DP_SIZE`。

> **注意**：`get_dp_local_info` 等 gather/scatter 路径用的是**全局** DP rank（`get_attention_dp_rank`），而 dense 层内部的划分才用 `local` 系列。不要混用。

---

## 3. 全局状态

文件顶部维护了一组进程级单例变量，由 `initialize_dp_attention()` 在分布式进程组建立后赋值一次，之后各处通过 getter 读取，避免层层透传参数。

| 变量 | 含义 | Getter |
|------|------|--------|
| `_ATTN_DP_RANK` | 全局 attention DP rank | `get_attention_dp_rank()` |
| `_ATTN_DP_SIZE` | 全局 attention DP 组大小 | `get_attention_dp_size()` |
| `_LOCAL_ATTN_DP_RANK` | dense 子组内的局部 DP rank | `get_local_attention_dp_rank()` |
| `_LOCAL_ATTN_DP_SIZE` | dense 子组内的局部 DP 组大小 | `get_local_attention_dp_size()` |
| `_ENABLE_DP_ATTENTION_FLAG` | 是否开启 DP attention | `is_dp_attention_enabled()` |
| `_DP_MAX_LEN_WITH_IDLE` | 混合 SSM 模型开关 | 内部使用 |

TP / CP 相关的 getter（`get_attention_tp_group` 等）则是对 `sglang.srt.distributed` 中通信组的薄封装。

### 3.1 `_DP_MAX_LEN_WITH_IDLE`

通过 HF config 中是否存在 `hybrid_override_pattern` 字段，识别混合 SSM（Mamba + Attention）模型：

```python
_DP_MAX_LEN_WITH_IDLE = (
    getattr(model_config.hf_config, "hybrid_override_pattern", None) is not None
)
```

这类模型的 Mamba 状态更新要求**每个 rank 都必须有输入**，即使该 rank 本轮没有分到 token。因此需要走 MAX_LEN 模式，用「伪造行」让空闲 rank 也参与计算。

### 3.2 `disable_dp_size()` 上下文管理器

用于投机采样（speculative decoding）的 draft worker：draft 模型的并行度可能与 target 模型不同，运行期间需要临时屏蔽 DP 划分。

```python
with disable_dp_size():
    # 此作用域内 get_attention_dp_size() 返回 1
    run_draft_model(...)
# 退出后自动还原（finally 保证异常安全）
```

---

## 4. Padding 模式：`DpPaddingMode`

gather 时各 DP rank 的 token 数通常不同，必须先对齐。这里有两种策略，**对应两种不同的通信原语**。

### 4.1 两种模式对比

设各 rank token 数为 $n_0, n_1, \dots, n_{D-1}$，$D = dp\_size$，记

$$
max\_len = \max_i n_i, \qquad sum\_len = \sum_i n_i
$$

| | `MAX_LEN` | `SUM_LEN` |
|---|---|---|
| 全局 buffer 行数 | $max\_len \times D$ | $sum\_len$ |
| 通信原语 | `all_gather_into_tensor` | `all_reduce` |
| 通信量 | $\sim max\_len \times D$ | $\sim sum\_len \times 2$ |
| 各 rank 长度 | 相同 | 不同 |
| symmetric memory | ✅ 可用 | ❌ 不可用 |
| 适合场景 | decode（各 rank token 数接近） | prefill（长度差异大） |

`all_reduce` 的系数 2 来源于 ring all-reduce 的经典结论：它等价于一次 `reduce_scatter` + 一次 `all_gather`，通信量约为 `all_gather` 的两倍。

### 4.2 选择逻辑

```python
@classmethod
def get_dp_padding_mode(cls, is_extend_in_batch, global_num_tokens):
    dp_size = get_attention_dp_size()

    if is_extend_in_batch and dp_size > 1:
        if _DP_MAX_LEN_WITH_IDLE and min(global_num_tokens) == 0:
            return DpPaddingMode.MAX_LEN      # 混合 SSM 需要伪造行
        return DpPaddingMode.SUM_LEN          # prefill 长度差异大

    max_len = max(global_num_tokens)
    sum_len = sum(global_num_tokens)
    if sum_len * 2 >= max_len * dp_size:
        return cls.MAX_LEN                    # 相等时优先 MAX_LEN
    else:
        return cls.SUM_LEN
```

三层判断：

1. **prefill 且 dp_size > 1** → `SUM_LEN`。prefill 各请求长度差异可能极大（比如一个 rank 4096 token、另一个 32 token），按 max 对齐会产生巨大冗余。
2. **混合 SSM 且存在空闲 rank** → 强制 `MAX_LEN`，保证 Mamba 状态更新不缺输入。
3. **其他情况**按通信量择优；两者相等时（例如 $dp\_size = 1$ 时 $max\_len = sum\_len$）优先 `MAX_LEN`，因为可以启用 symmetric memory 优化。

### 4.3 CUDA Graph 下的默认模式

```python
@classmethod
def get_default_mode_in_cuda_graph(cls) -> DpPaddingMode:
    if _USE_ROCM700A_WA:
        return cls.SUM_LEN     # ROCm 7.0.0 alpha RCCL 临时规避
    else:
        return cls.MAX_LEN
```

CUDA Graph 要求**形状固定**，不能按 batch 动态选择模式，因此统一用 `MAX_LEN`（各 rank 等长，形状可静态确定）。

---

## 5. 缓冲区管理：`_DpGatheredBufferWrapper`

这是一个全部由类变量构成的进程级单例。**它只保存元信息，不持有 buffer 本身**——buffer 在每次 getter 调用时按需 `torch.empty` 分配，从而复用 PyTorch caching allocator 或 symmetric memory 池。

### 5.1 元信息分两类

**静态元信息**（`set_metadata`，初始化时调用一次）：

- `_hidden_size` — buffer 第二维
- `_dtype` / `_device`

**动态元信息**（`set_dp_buffer_len`，每个 batch 前调用一次）：

- `_global_dp_buffer_len` — gather 后的总行数
- `_local_dp_buffer_len` — 本 rank 行数（可能已为 CUDA Graph padding）
- `_dp_max_padding` — 是否 MAX_LEN 模式，决定能否用 symmetric memory
- `_global_num_tokens` — 各 rank token 数列表（CPU 侧）

### 5.2 symmetric memory 的约束

```python
with use_symmetric_memory(group, disabled=not cls._dp_max_padding):
    buffer = torch.empty((len, hidden_size), dtype=..., device=...)
```

对称内存要求**所有 rank 分配完全相同的大小**，只有 MAX_LEN 模式满足这一点。这也是 `is_allocation_symmetric()` 的判据：

```python
def is_allocation_symmetric() -> bool:
    return not is_dp_attention_enabled() or is_dp_max_padding()
```

未开启 DP attention 时天然对称（所有 rank 处理同一批 token）。

---

## 6. 本地切片计算

gather/scatter 都需要知道「本 rank 的数据在全局 buffer 中的位置」，即 $(start, length)$。偏移由前缀和给出：

$$
start_d = \sum_{i<d} n_i, \qquad length_d = n_d
$$

文件提供了 GPU / CPU 两个版本。

### 6.1 `get_dp_local_info` — GPU 张量版

```python
cumtokens = torch.cumsum(forward_batch.global_num_tokens_gpu, dim=0)
local_start_pos = torch.zeros_like(cumtokens[0]) if dp_rank == 0 else cumtokens[dp_rank - 1]
local_num_tokens = forward_batch.global_num_tokens_gpu[dp_rank]
```

**为什么返回 GPU 张量而不是 Python int？**

- 若转成 Python int，需要 `.item()`，会强制 **device → host 同步**，打断流水；
- 更重要的是**无法被 CUDA Graph 捕获**——graph 回放时值必须能在 device 端读取。

结果缓存在 `forward_batch.dp_local_start_pos` / `dp_local_num_tokens` 上，避免同一次 forward 中重复计算（一个模型有几十层，每层至少一次 gather + 一次 scatter）。

### 6.2 `get_dp_local_slice_cpu` — Python int 版

直接读 CPU 侧的 `global_num_tokens_cpu`，不产生 D2H 同步。额外处理 CUDA Graph 布局：

```python
if can_run_graph:
    local_start_pos = dp_rank * cuda_graph_batch   # 等长 padding，直接相乘
else:
    local_start_pos = sum(global_num_tokens[:dp_rank])  # 真实前缀和
```

走 CUDA Graph 时每个 rank 在 buffer 中占据固定的 `cuda_graph_batch` 行，偏移退化为简单乘法。

---

## 7. Triton memcpy kernel

### 7.1 为什么不用 `copy_` 或切片赋值

```python
# 不能这样写：
global_tokens[start : start + sz] = local_tokens
```

因为 `start` 和 `sz` 是 **GPU 标量张量**。用它们做切片索引会：

1. 强制 D2H 同步（要把值取回 host 才能算切片边界）；
2. 产生 host 端依赖，无法被 CUDA Graph 捕获。

自定义 kernel 则可以在 device 端直接 `tl.load(offset_ptr)` 读取这两个标量。

### 7.2 kernel 实现

```python
@triton.jit
def memcpy_triton_kernel(dst_ptr, src_ptr, offset_ptr, sz_ptr,
                         offset_src: tl.constexpr, chunk_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0).to(tl.int64)
    offset = tl.load(offset_ptr).to(tl.int64) * chunk_size   # 行 → 元素
    sz = tl.load(sz_ptr).to(tl.int64) * chunk_size

    start_index = pid * BLOCK_SIZE
    offs = tl.arange(0, BLOCK_SIZE)
    mask = start_index + offs < sz                            # 尾块屏蔽越界

    if offset_src:
        # scatter：src[offset : offset+sz] -> dst[0 : sz]
        data = tl.load(src_ptr + offset + start_index + offs, mask=mask)
        tl.store(dst_ptr + start_index + offs, data, mask=mask)
    else:
        # gather：src[0 : sz] -> dst[offset : offset+sz]
        data = tl.load(src_ptr + start_index + offs, mask=mask)
        tl.store(dst_ptr + offset + start_index + offs, data, mask=mask)
```

关键点：

- `offset_src` 是**编译期常量**（`tl.constexpr`），两个分支各自编译成独立 kernel，运行时无分支开销；
- 所有索引转 `int64`，防止大张量（$rows \times hidden\_size$ 可能超过 $2^{31}$）溢出；
- `chunk_size = prod(shape[1:])`，把「行偏移」放大为「元素偏移」。

### 7.3 Python 封装

```python
def memcpy_triton(dst, src, dim, offset, sz, offset_src):
    max_size = min(src.numel(), dst.numel())   # 用上界估算 grid
    assert dim == 0, "dim != 0 unsupported"
    assert src.shape[1:] == dst.shape[1:]      # 保证每行元素数相同
    chunk_size = prod(src.shape[1:])
    BLOCK_SIZE = 8192
    grid = (triton.cdiv(max_size, BLOCK_SIZE),)
    memcpy_triton_kernel[grid](dst, src, offset, sz, offset_src, chunk_size, BLOCK_SIZE)
```

grid 按 `min(src.numel(), dst.numel())` 估算（实际拷贝量的上界），多余的 block 会被 kernel 内的 `mask` 全部屏蔽掉。

---

## 8. gather：两种实现

`_dp_gather` 是统一入口，按 padding 模式分派：

```python
def _dp_gather(global_tokens, local_tokens, forward_batch, is_partial):
    if forward_batch.dp_padding_mode.is_max_len():
        _dp_gather_via_all_gather(...)
    else:
        _dp_gather_via_all_reduce(...)
```

### 8.1 关键参数 `is_partial`

这是理解 gather 正确性的核心。它描述 `local_tokens` 在 **attn TP 组内**的语义：

| `is_partial` | 含义 | 典型场景 | 处理方式 |
|---|---|---|---|
| `True` | 各 TP rank 持有**部分和**，相加才是完整值 | attention 输出尚未 all-reduce | 所有 TP rank 都参与写入/规约 |
| `False` | 各 TP rank 持有**完整副本** | hidden_states 已 all-reduce；`input_ids` | 只让 rank 0 贡献，其余置零 |

对外暴露为两个语义化的 API：

```python
dp_gather_partial(global, local, fb)    # is_partial=True
dp_gather_replicate(global, local, fb)  # is_partial=False
```

**若用错会怎样**：把 replicate 数据当 partial 处理，`all_reduce` 会把 $attn\_tp\_size$ 份相同的副本累加，结果被放大 $attn\_tp\_size$ 倍。

### 8.2 `_dp_gather_via_all_reduce`（SUM_LEN）

核心技巧：**先清零 + 各 rank 写不重叠的段 + all_reduce(sum) ≡ concat**。

```python
global_tokens.fill_(0)                    # 必须先清零，否则残留数据被累加

if local_tokens.shape[0] > 0 and (is_partial or get_attention_tp_rank() == 0):
    assert local_tokens.untyped_storage() is not global_tokens.untyped_storage()
    memcpy_triton(global_tokens, local_tokens, 0, local_start_pos, local_num_tokens, False)

NUM_GPUS_PER_NODE = 8
if not local_tokens.dtype.is_floating_point and get_tensor_model_parallel_world_size() <= NUM_GPUS_PER_NODE:
    inplace_all_reduce(global_tokens, group_name=get_tp_group().unique_name)
else:
    global_tokens[:] = tensor_model_parallel_all_reduce(global_tokens)
```

逐点说明：

1. **`fill_(0)` 不可省**：buffer 是 `torch.empty` 分配的，含脏数据；all_reduce 是求和，脏数据会污染结果。
2. **别名检查**：`local` 和 `global` 若共享存储，memcpy 会读写重叠导致未定义行为。
3. **整数类型的特殊分支**：`input_ids` 是 int32。单机（$\le 8$ 卡）场景下自定义 all-reduce 走的是**原地路径**，因此必须调 `inplace_all_reduce`，否则返回值语义不匹配。

### 8.3 `_dp_gather_via_all_gather`（MAX_LEN）

```python
if get_attention_tp_size() == 1:
    get_tp_group().all_gather_into_tensor(global_tokens, local_tokens)
    return

if not is_partial:
    if get_attention_tp_rank() != 0:
        local_tokens.fill_(0)          # 只保留 rank 0 的副本

scattered = local_tokens.tensor_split(get_attention_tp_size())[get_attention_tp_rank()]
get_attention_tp_group().reduce_scatter_tensor(scattered, local_tokens)
get_tp_group().all_gather_into_tensor(global_tokens, scattered)
```

**为什么 attn_tp > 1 时不能直接 all_gather？**

同一个 DP 组内有 `attn_tp_size` 个 rank，它们持有的是同一批 token 的数据（副本或部分和）。直接在全局 TP 组上 all_gather，这批 token 会被重复贡献 `attn_tp_size` 次，全局 buffer 布局就错了。

**解法**（两步）：

1. 在 attn TP 组内 `reduce_scatter`：把 partial 相加成完整值，同时切成 `attn_tp_size` 份**互不重叠**的小片，每个 rank 持有一片；
2. 在全局 TP 组上 `all_gather`：所有小片拼接成完整全局张量。

此时每个全局 rank 贡献的都是不重叠的数据，布局正确。总通信量与直接 all_gather 相当。

`is_partial=False` 时先把非 0 rank 置零，是为了让第 1 步的 `reduce_scatter`（求和）不会把副本重复累加——相当于 $x + 0 + 0 + \dots = x$。

---

## 9. scatter

```python
def dp_scatter(local_tokens, global_tokens, forward_batch):
    local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)

    local_tokens.fill_(0)      # padding 行置为确定值
    assert local_tokens.is_contiguous() and global_tokens.is_contiguous()
    if local_tokens.shape[0] > 0:
        assert local_tokens.untyped_storage() is not global_tokens.untyped_storage()
        memcpy_triton(local_tokens, global_tokens, 0, local_start_pos, local_num_tokens, True)
```

scatter 是纯本地操作，**无通信**——只是从全局张量里 memcpy 出自己那一段。

注意 `local_num_tokens` 不一定等于 `local_tokens.shape[0]`：后者可能因 CUDA Graph 而被 padding 到更大的固定值。`fill_(0)` 保证这些多余行是确定的 0 而非脏数据。

---

## 10. 其余通信封装

### 10.1 `dp_reduce_scatter_tensor`

在 DP 维度上规约全局张量并切回各 DP rank：

```python
if get_tensor_model_parallel_world_size() == get_attention_dp_size():
    get_tp_group().reduce_scatter_tensor(output, input)      # attn_tp == 1，一步到位
else:
    scattered = input.tensor_split(tp_size)[tp_rank]
    get_tp_group().reduce_scatter_tensor(scattered, input)   # 步骤 1：全局规约 + 细分
    get_attention_tp_group().all_gather_into_tensor(output, scattered)  # 步骤 2：组内拼回
```

当 $tp\_size = dp\_size$（即 $attn\_tp\_size = 1$）时，全局 TP 组恰好就是 DP 组，一次 `reduce_scatter` 即可。否则全局 reduce_scatter 切得太细（每 rank 只拿到 $1/tp\_size$），需要在 attn TP 组内 all_gather 还原成每个 DP rank 完整的 $1/dp\_size$。

### 10.2 组内原语薄封装

| 函数 | 作用域 |
|---|---|
| `attn_tp_all_reduce` / `attn_tp_all_gather` / `attn_tp_all_gather_into_tensor` / `attn_tp_reduce_scatter_tensor` | attention TP 组 |
| `attn_cp_all_gather_into_tensor` / `attn_cp_reduce_scatter_tensor` | attention CP 组 |
| `moe_cp_all_gather_into_tensor` | MOE_DP 组 |

### 10.3 MoE CP 相关

```python
def is_enable_moe_cp_allgather() -> bool:
    sa = get_global_server_args()
    return sa.attn_cp_size > sa.moe_dp_size
```

当 `attn_cp_size > moe_dp_size` 时，attention 阶段序列被 CP 切分到多个 rank，而 MoE 的 DP 度更小。MoE 需要看到完整序列，因此进入 MoE 前必须跨 CP rank 做一次 all-gather 把被切散的 token 汇总回来。

`get_moe_cp_group()` 返回的 MOE_DP 组在这种情况下会包含 CP 伙伴 rank。

---

## 11. 一次 forward 的完整时序

```
┌─ 每个 batch 开始 ────────────────────────────────────────────┐
│ 1. 收集各 rank token 数 → global_num_tokens                  │
│ 2. DpPaddingMode.get_dp_padding_mode(...) → MAX_LEN/SUM_LEN  │
│ 3. set_dp_buffer_len(global_len, local_len, dp_max_padding)  │
│    set_is_extend_in_batch(...)                               │
└──────────────────────────────────────────────────────────────┘
                            │
                ┌───────────▼───────────┐
                │  逐层循环（每一层）    │
                └───────────┬───────────┘
                            │
   ┌────────────────────────▼─────────────────────────┐
   │ Attention（DP）                                   │
   │   输入：local_tokens  (local_len, hidden)         │
   │   每个 DP rank 只算自己那批 token                  │
   │   KV Cache 只存自己那批请求 → 显存不冗余           │
   └────────────────────────┬─────────────────────────┘
                            │
                  get_global_dp_buffer()
                  dp_gather_partial / dp_gather_replicate
                            │
   ┌────────────────────────▼─────────────────────────┐
   │ MoE / Dense FFN（TP / EP）                        │
   │   输入：global_tokens (global_len, hidden)        │
   │   需要看到全局所有 token 才能做专家路由            │
   └────────────────────────┬─────────────────────────┘
                            │
                    get_local_dp_buffer()
                    dp_scatter（纯本地 memcpy）
                            │
   ┌────────────────────────▼─────────────────────────┐
   │ 回到 local_tokens，进入下一层 Attention           │
   └───────────────────────────────────────────────────┘
```

调度这套流程的是 `LayerCommunicator`（见 `docs/theory/distributed/DP_attention.md`），它通过 `ScatterMode` 描述数据当前处于 `SCATTERED` / `TP_ATTN_FULL` / `FULL` 中的哪种分布状态，并在 `prepare_attn` / `prepare_mlp` / `postprocess_layer` 三个阶段插入合适的通信原语。

---

## 12. 代价与权衡

DP Attention 不是免费的：

| 收益 | 代价 |
|------|------|
| KV Cache 显存冗余从 $tp\_size$ 份降到 $attn\_tp\_size$ 份 | 每层增加一次 gather + 一次 scatter 通信 |
| KV 分片不会过小，attention kernel 效率高 | Attention **权重**在每个 DP rank 上都要存一份（不再按 TP 切分） |
| 可支持更大 batch / 更长上下文 | 各 rank token 数不均衡时产生 padding 浪费 |

关于「权重变多」：设 attention 权重大小为 $W_{attn}$。

- 纯 TP：每 rank 存 $W_{attn} / tp\_size$
- DP Attention：每 rank 存 $W_{attn} / attn\_tp\_size$

由于 $attn\_tp\_size = tp\_size / (attn\_dp\_size \cdot attn\_cp\_size) < tp\_size$，每 rank 的权重占用确实增加了。但对 MoE 模型而言 $W_{attn}$ 远小于 KV Cache 与专家权重，这笔交换非常划算——尤其是 MLA，其权重本身就被低秩压缩过。

特别地，当 $attn\_tp\_size = 1$ 时，MLA 的上投影矩阵从「按 TP 切分」退化为「每 rank 一份完整副本」。

---

## 13. 常见坑

1. **`is_partial` 用错** → 结果被放大 $attn\_tp\_size$ 倍或丢失分量。判断依据：数据是否已在 attn TP 组内 all-reduce 过。
2. **忘记 `fill_(0)`** → SUM_LEN 路径的 all_reduce 会把 `torch.empty` 的脏数据累加进来。
3. **local / global buffer 别名** → 代码里有 `untyped_storage()` 断言拦截，但自定义调用时要留意。
4. **在 gather/scatter 中误用 `local` 系列 rank** → `get_dp_local_info` 必须用全局 `get_attention_dp_rank()`；`local` 系列只用于 dense 层子组。
5. **CUDA Graph 下动态选择 padding 模式** → 形状必须固定，只能用 `get_default_mode_in_cuda_graph()`。
6. **把 GPU 标量转成 Python int** → 触发 D2H 同步并破坏 graph 捕获，这正是 Triton memcpy 存在的理由。

