# 滑动窗口注意力（Sliding Window Attention, SWA）原理详解

> 本文系统介绍滑动窗口注意力（SWA）的数学原理、复杂度与显存收益，
> 以及 SGLang 中针对 SWA 的混合 KV cache、分配器与前缀缓存（Radix Cache）实现。

## 目录

1. [为什么需要 SWA](#1-为什么需要-swa)
2. [SWA 的数学定义](#2-swa-的数学定义)
3. [混合注意力架构（Hybrid SWA）](#3-混合注意力架构hybrid-swa)
4. [复杂度与显存收益分析](#4-复杂度与显存收益分析)
5. [SGLang 中的 SWA 实现](#5-sglang-中的-swa-实现)
6. [支持 SWA 的模型](#6-支持-swa-的模型)
7. [局限与权衡](#7-局限与权衡)

---

## 1. 为什么需要 SWA

标准的因果自注意力（causal self-attention）中，序列里第 $i$ 个 token 需要与它之前的
所有 token（位置 $0 \dots i$）做注意力计算。这带来两个随序列长度 $n$ 增长的代价：

- **计算复杂度** $O(n^2 \cdot d)$：注意力分数矩阵是 $n \times n$ 的。
- **KV cache 显存** $O(n)$：自回归推理时，每个历史 token 的 Key/Value 都要常驻显存，
  以避免重复计算。当上下文从 4K 扩展到 128K，KV cache 会成为显存的主要瓶颈，
  直接限制了并发批量大小（batch size）和最大上下文长度。

SWA 的核心洞察是：**对于语言建模，绝大多数有用的局部依赖都集中在最近的若干个 token 内。**
因此可以让每个 token 只关注最近 $w$ 个 token（$w$ 为窗口大小），把注意力的「视野」
限制在一个固定宽度的滑动窗口里，从而将计算和显存代价从「随序列线性/平方增长」
降为「随窗口大小恒定」。

---

## 2. SWA 的数学定义

设窗口大小为 $w$。在标准因果注意力中，query 位置 $i$ 的注意力掩码（mask）允许它看到
所有 key 位置 $j \le i$：

$$
\text{mask}_{\text{causal}}(i, j) =
\begin{cases}
0 & j \le i \\
-\infty & j > i
\end{cases}
$$

SWA 则在因果约束之上，再叠加一个「左侧窗口」约束，要求 $j$ 与 $i$ 的距离不超过 $w$：

$$
\text{mask}_{\text{swa}}(i, j) =
\begin{cases}
0 & i - w < j \le i \\
-\infty & \text{otherwise}
\end{cases}
$$

即 query $i$ 只能看到区间 $(i - w,\ i]$ 内的 key，共 $w$ 个 token。注意力输出为：

$$
\text{Attn}(i) = \sum_{j = \max(0,\, i-w+1)}^{i}
\text{softmax}_j\!\left(\frac{q_i \cdot k_j^\top}{\sqrt{d}}\right) v_j
$$

### 窗口大小的边界约定

不同框架对「窗口大小」的定义差一个 token，需要特别小心。在 FlashAttention 中，
`window_size = (left, right)`，其中 `left` 表示当前 token 之外**额外**能向左看的 token 数。
因此若希望总可见窗口为 `config.sliding_window` 个 token，传入的 `left` 应为
`sliding_window - 1`。SGLang 在模型侧就完成了这个减一：

```python
# python/sglang/srt/models/gemma2.py:49
def get_attention_sliding_window_size(config):
    return config.sliding_window - 1
```

而 attention backend 直接使用该值，不再重复减一：

```python
# python/sglang/srt/layers/attention/flashattention_backend.py:788
# we don't do layer.sliding_window_size - 1 since in
# model.get_attention_sliding_window_size() we already - 1
window_size = (layer.sliding_window_size, 0) if is_swa_layer else (-1, -1)
```

`(-1, -1)` 表示不启用窗口限制（即退化为全注意力），`(w, 0)` 表示向左看 $w$ 个、
向右看 0 个（因果）。

### 跨层的感受野扩展

一个常见疑问：窗口只有 $w$，模型如何捕捉超过 $w$ 的长程依赖？

答案在于**层的堆叠**。与卷积网络的感受野类似，第 $L$ 层位置 $i$ 的 token 通过窗口
间接「触达」了第 $L-1$ 层 $(i-w,\ i]$ 范围内的表示，而这些表示又各自聚合了再上一层
更早的信息。因此理论感受野约为 $L \times w$。例如 Mistral-7B 采用 $w = 4096$、
32 层，理论感受野可达约 13 万 token。

---

## 3. 混合注意力架构（Hybrid SWA）

纯 SWA（所有层都是滑动窗口）虽然显存最省，但会牺牲对全局信息的精确访问。
现代模型（Gemma 2/3、GPT-OSS、Ministral 等）普遍采用**混合架构**：
让一部分层保持全注意力（full attention），其余层使用滑动窗口，二者交替排布。

以 Gemma 2 为例，偶数层用滑动窗口、奇数层用全局注意力：

```python
# python/sglang/srt/models/gemma2.py:170
use_sliding_window = layer_id % 2 == 0 and hasattr(config, "sliding_window")
...
sliding_window_size=(
    get_attention_sliding_window_size(config)
    if use_sliding_window
    else -1  # 全注意力层
)
```

这样设计的好处：

- **全局层**负责精确的长程检索（如「文章开头提到的名字」），数量少，但 KV 必须全量保留。
- **滑动窗口层**负责高效的局部建模，数量多，KV 只需保留窗口内的部分。

由于全局层占比小，整体 KV cache 显存被显著压缩，同时保留了足够的长程能力。

```
层索引:   0      1      2      3      4      5    ...
类型:    SWA   Full   SWA   Full   SWA   Full
KV保留:  窗口w  全序列 窗口w  全序列 窗口w  全序列
```

---

## 4. 复杂度与显存收益分析

### 计算复杂度

| 注意力类型 | 时间复杂度 | 说明 |
|-----------|-----------|------|
| 全注意力   | $O(n^2 d)$ | 每个 token 看全部历史 |
| 纯 SWA    | $O(n w d)$ | 每个 token 只看 $w$ 个，$w \ll n$ 时近似线性 |
| 混合      | 介于两者之间 | 取决于全局层占比 |

### KV cache 显存

设序列长度 $n$、窗口 $w$、总层数 $N$、其中全局层 $N_f$、滑动窗口层 $N_s = N - N_f$。

- **全注意力模型**：KV 显存 $\propto n \cdot N$。
- **混合 SWA 模型**：全局层仍需 $n \cdot N_f$，但滑动窗口层只需缓存约 $w$ 个 token，
  即 $w \cdot N_s$。总量 $\propto n \cdot N_f + w \cdot N_s$。

当 $w \ll n$ 且 $N_s$ 占多数时，显存节省非常可观。这正是 SGLang 为 SWA 单独设计
KV cache 池的动机——让两类层的 token 容量解耦，各按需分配。

---

## 5. SGLang 中的 SWA 实现

SGLang 的 SWA 支持贯穿四个层面：**注意力层标记 → 双 KV 池 → 双分配器 → 混合前缀缓存**。

### 5.1 注意力层的窗口标记

每个注意力层用 `RadixAttention.sliding_window_size` 标记自己是否为 SWA 层：

```python
# python/sglang/srt/layers/radix_attention.py:80
self.sliding_window_size = sliding_window_size or -1
```

`-1` 表示全注意力层，正数表示滑动窗口层。下游的 KV 池和 attention backend 都依据
该字段决定如何处理每一层。

### 5.2 双 KV 池：`SWAKVPool`

`SWAKVPool` 内部维护**两个独立的物理 KV 池**——一个给全注意力层（`full_kv_pool`，
容量 `size`），一个给滑动窗口层（`swa_kv_pool`，容量 `size_swa`），二者容量可以不同：

```python
# python/sglang/srt/mem_cache/swa_memory_pool.py:28
class SWAKVPool(KVCache):
    """KV cache with separate pools for full and SWA attention layers."""
```

它通过一张 `layers_mapping` 表把「全局层索引」映射到「池内的局部索引 + 是否 SWA 层」：

```python
# python/sglang/srt/mem_cache/swa_memory_pool.py:86
# {layer_id: (index, is_swa_layer)}
self.layers_mapping: Dict[int, Tuple[int, bool]] = {}
```

当某一层读写 KV 时，`get_kv_buffer` / `set_kv_buffer` 会先查表，再路由到对应的池。

### 5.3 双分配器与 `full_to_swa_index_mapping`

`SWATokenToKVPoolAllocator` 同时持有两个底层分配器（full / swa），每次分配 token slot
时，**两个池各分配一份**，并用 `full_to_swa_index_mapping` 记录二者的索引对应关系：

```python
# python/sglang/srt/mem_cache/swa_memory_pool.py:357
alloc_full_indices = self.full_attn_allocator.alloc(need_size)
alloc_swa_indices = self.swa_attn_allocator.alloc(need_size)
...
self.full_to_swa_index_mapping[alloc_full_indices] = alloc_swa_indices
```

关键设计点：

- **统一以 full 索引对外**：上层（scheduler、req）只看到 full 池的索引，SWA 池的索引
  通过 `translate_loc_from_full_to_swa` 隐式换算，简化了上层逻辑。

  ```python
  # python/sglang/srt/mem_cache/swa_memory_pool.py:149
  def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
      return self.full_to_swa_index_mapping[kv_indices].to(torch.int32)
  ```

- **`-1` 哨兵**：映射表末尾追加一个 `-1`，使得无效索引 `-1` 映射后仍为 `-1`
  （`alloc_extend` 中 `last_loc` 的首项常为 `-1`）。

  ```python
  # python/sglang/srt/mem_cache/swa_memory_pool.py:287
  # Note: append one more item of value -1 in the end so -1 maps to -1.
  ```

- **可用容量取两池最小值**：因为分配必须两池同时成功。

  ```python
  # python/sglang/srt/mem_cache/swa_memory_pool.py:311
  def available_size(self):
      return min(
          self.full_attn_allocator.available_size(),
          self.swa_attn_allocator.available_size(),
      )
  ```

### 5.4 显存配比：`swa_full_tokens_ratio`

两个池的容量如何切分总显存？由启动参数 `swa_full_tokens_ratio` 控制（默认 `0.8`）：

```python
# python/sglang/srt/server_args.py:360
swa_full_tokens_ratio: float = 0.8
```

切分逻辑见 `_resolve_hybrid_swa_tokens`。给定总 token 容量与两类层的「每 token 显存占用」，
按比例 $r = \text{swa\_tokens} / \text{full\_tokens}$ 联立求解，使 full 池和 swa 池
共同占满总显存预算：

```python
# python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py:297
def _resolve_hybrid_swa_tokens(self, ...):
    """Split token_capacity into full/swa pools."""
```

当模型所有层都是 SWA 层时，full 池容量为 0，退化为单一 SWA 池。

### 5.5 混合前缀缓存：`SWARadixCache`

SGLang 的核心特性之一是 **RadixAttention 前缀缓存**——用基数树（radix tree）共享多个
请求的公共前缀 KV，命中即跳过重计算。但 SWA 给前缀复用带来一个矛盾：

> 滑动窗口层只保留最近 $w$ 个 token 的 KV。当一个前缀节点对应的 token 已经滑出窗口，
> 它的 **SWA 池 KV 可以被释放**，但它的 **full 池 KV 仍可能被其他请求当作前缀复用**。

`SWARadixCache`（`swa_radix_cache.py`）通过几项机制协调这对矛盾：

1. **墓碑标记（tombstone）**：当某节点的 SWA 层 KV 被释放、但 full 层 KV 仍保留时，
   将其标记为 `swa_tombstone`。墓碑节点的 full 前缀仍可匹配，但不再计入 SWA 的 LRU。

   ```python
   # python/sglang/srt/mem_cache/swa_radix_cache.py:74
   # swa_tombstone is used to indicate the kv indices have been freed for swa layers
   self.swa_tombstone = False
   ```

2. **双引用计数（lock_ref）**：每个节点同时维护 `full_lock_ref` 与 `swa_lock_ref`，
   且满足不变式「`full_lock_ref >= swa_lock_ref`」——锁住 SWA 必然锁住 full，
   反之不必然。这确保正在被引用的 KV 不会被错误回收。

   ```python
   # python/sglang/srt/mem_cache/swa_radix_cache.py:76
   # invariant: for any node, if swa_lock_ref is locked, full_lock_ref must be locked;
   # if full_lock_ref is locked, swa_lock_ref doesn't need to be locked. So,
   # full_lock_ref is always >= swa_lock_ref.
   ```

3. **双 LRU 链表**：维护两条独立的 LRU 链——一条管理 full 池的淘汰，一条管理 swa 池。
   两类 KV 可以按各自的访问热度独立淘汰，互不绑架。

这套机制让 SWA 模型也能享受前缀缓存的加速，同时及时回收滑出窗口的 SWA 显存。

### 5.6 attention backend 的窗口元数据

在前向计算时，backend 需要为 SWA 层准备专门的页表（page table）。
FlashAttention backend 用 `use_sliding_window_kv_pool` 判定是否启用 SWA 池，
并通过 `swa_page_table` 把 full 索引翻译为 swa 索引后供 kernel 使用：

```python
# python/sglang/srt/layers/attention/flashattention_backend.py:678
if self.use_sliding_window_kv_pool:
    metadata.swa_page_table = (
        self.token_to_kv_pool.translate_loc_from_full_to_swa(...)
    )
```

随后按层的 `sliding_window_size` 决定传给 kernel 的 `window_size` 元组（见 §2 边界约定）。

### 5.7 数据流总览

```
请求到达
  │
  ▼
SWATokenToKVPoolAllocator.alloc()        # full / swa 双池各分配一份 slot
  │   └─ 记录 full_to_swa_index_mapping
  ▼
SWARadixCache.match_prefix()             # 匹配公共前缀，双 lock_ref 锁定
  │
  ▼
逐层前向：
  ├─ 全注意力层 → full_kv_pool，window=(-1,-1)
  └─ 滑动窗口层 → swa_kv_pool，window=(w,0)，经 swa_page_table 寻址
  │
  ▼
请求结束 / KV 滑出窗口
  ├─ SWARadixCache 双 LRU 淘汰
  └─ swa 池 KV 释放（可能留 swa_tombstone），full 池 KV 视前缀复用情况保留
```

---

## 6. 支持 SWA 的模型

SGLang 中约有 20 个模型实现引用了 `sliding_window`，典型包括：

- **Gemma 2 / Gemma 3 / Gemma 3n**：偶数层滑动窗口、奇数层全局，交替混合。
- **Mistral / Ministral**：经典纯/混合滑动窗口。
- **GPT-OSS**：混合 SWA + attention sink。
- **Phi-MoE、Cohere Command-R、OLMo 2、EXAONE-MoE、Step3.5、MiMo-V2** 等。

各模型通过 `get_attention_sliding_window_size` 与逐层的 `use_sliding_window` 判定，
把窗口配置传入 `RadixAttention`，其余的 KV 池/缓存逻辑由框架统一处理。

---

## 7. 局限与权衡

- **长程精度损失**：纯 SWA 对超出 $L \times w$ 感受野的依赖会衰减；混合架构通过保留
  少量全局层缓解这一问题，是目前主流折中。
- **前缀复用复杂度上升**：如 §5.5 所述，SWA 与 radix 前缀缓存的协同需要墓碑、
  双引用计数、双 LRU 等额外机制，实现复杂度明显高于全注意力的单池缓存。
- **配比调参**：`swa_full_tokens_ratio` 需要结合模型的全局/窗口层比例与典型上下文长度
  调整；配比不当会导致某一类池先耗尽，拖累整体并发。
- **层级窗口差异**：部分模型不同层窗口大小不同，框架以逐层 `sliding_window_size`
  为准（见 `flashattention_backend.py:384` 注释），需注意元数据准备的正确性。

---

## 参考实现位置

| 模块 | 文件路径 |
|------|---------|
| 注意力层窗口标记 | `python/sglang/srt/layers/radix_attention.py` |
| 双 KV 池 | `python/sglang/srt/mem_cache/swa_memory_pool.py` |
| 混合前缀缓存 | `python/sglang/srt/mem_cache/swa_radix_cache.py` |
| 显存配比切分 | `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` |
| FlashAttention 窗口元数据 | `python/sglang/srt/layers/attention/flashattention_backend.py` |
| 模型层配置示例 | `python/sglang/srt/models/gemma2.py` |
| 启动参数 | `python/sglang/srt/server_args.py` |

---

## 附录 A：SWA 数值计算示例

为直观理解 §2 的窗口掩码与 §4 的显存收益，下面用一组**极小维度**的具体数字
走一遍滑动窗口注意力，并与全注意力对比。

### A.0 设定

- 序列长度 $n = 5$（token 记为 $t_0 \dots t_4$）
- 窗口大小 $w = 2$（每个 query 只能看「自己 + 前 1 个」，即区间 $(i-2,\ i]$）
- 头维 $d = 2$，单头，$\sqrt{d} = \sqrt{2} \approx 1.414$

各 token 的 Query / Key / Value 向量（人为设定，便于手算）：

| token | $q_i$ | $k_i$ | $v_i$ |
|-------|-------|-------|-------|
| $t_0$ | $[1,0]$ | $[1,0]$ | $[10,\ 0]$ |
| $t_1$ | $[0,1]$ | $[0,1]$ | $[0,\ 20]$ |
| $t_2$ | $[1,1]$ | $[1,1]$ | $[30,\ 30]$ |
| $t_3$ | $[1,0]$ | $[1,0]$ | $[40,\ 0]$ |
| $t_4$ | $[0,1]$ | $[0,1]$ | $[0,\ 50]$ |

### A.1 掩码对比（§2）

以 query $t_4$（$i=4$）为例。

- **全注意力**：可见 key 为 $j \le 4$，即 $\{t_0,t_1,t_2,t_3,t_4\}$，共 5 个。
- **SWA（$w=2$）**：可见区间 $(i-2,\ i] = (2,\ 4] = \{t_3,\ t_4\}$，共 2 个。

掩码矩阵（行 = query $i$，列 = key $j$，✓ 可见 / · 屏蔽），$w=2$：

```
        k0  k1  k2  k3  k4
 q0     ✓   ·   ·   ·   ·
 q1     ✓   ✓   ·   ·   ·
 q2     ·   ✓   ✓   ·   ·
 q3     ·   ·   ✓   ✓   ·
 q4     ·   ·   ·   ✓   ✓
```

可见每行最多 $w=2$ 个 ✓，且随 $i$ 增大窗口整体右移——这就是「滑动」窗口。

### A.2 SWA 下 $t_4$ 的注意力输出

只对可见的 $\{t_3, t_4\}$ 计算。打分 $s_j = q_4 \cdot k_j / \sqrt{d}$，其中 $q_4 = [0,1]$：

$$
s_3 = \frac{[0,1]\cdot[1,0]}{\sqrt2} = \frac{0}{\sqrt2} = 0,\qquad
s_4 = \frac{[0,1]\cdot[0,1]}{\sqrt2} = \frac{1}{\sqrt2} \approx 0.707
$$

softmax：

$$
\text{权重} = \text{softmax}([0,\ 0.707]) \approx [0.330,\ 0.670]
$$

加权 Value（只含 $v_3,v_4$）：

$$
\text{out}_4 \approx 0.330\cdot[40,0] + 0.670\cdot[0,50] = [13.2,\ 33.5]
$$

### A.3 对比全注意力下的 $t_4$

全注意力要对 $\{t_0 \dots t_4\}$ 全部计算。打分（$q_4=[0,1]$）：

$$
s_0=0,\quad s_1=\tfrac{1}{\sqrt2}\approx0.707,\quad s_2=\tfrac{1}{\sqrt2}\approx0.707,\quad s_3=0,\quad s_4\approx0.707
$$

softmax 后 $t_1,t_2,t_4$ 权重较高，输出会显著混入 $v_1=[0,20]$、$v_2=[30,30]$ 的贡献，
结果与 A.2 不同。**这正体现 SWA 的取舍**：$t_4$ 看不到更早的 $t_1,t_2$，
牺牲了部分长程信息，换取只算 2 个 key 而非 5 个。

### A.4 计算量与显存对比（§4）

本例 $n=5$、$w=2$：

| 指标 | 全注意力 | SWA（$w=2$） |
|------|---------|--------------|
| 注意力打分对数（下三角） | $1+2+3+4+5 = 15$ | $1+2+2+2+2 = 9$ |
| 单层 KV 需保留的 token | 全部 5 个 | 最近约 2 个 |

当 $n$ 远大于 $w$（如 $n=128\text{K}$、$w=4096$）时：

- 打分对数从 $O(n^2)$ 降到 $O(nw)$；
- 每个 SWA 层 KV 只需保留约 $w$ 个 token，而非 $n$ 个。

这与 §4 的复杂度表、以及 §5 中 SGLang 为 SWA 层单独分配 `swa_kv_pool`
（容量按 $w$ 量级而非 $n$）的动机完全一致。
