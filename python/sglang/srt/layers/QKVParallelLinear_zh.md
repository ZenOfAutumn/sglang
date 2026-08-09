# QKVParallelLinear 原理详解

> 代码位置：`python/sglang/srt/layers/linear.py`（类 `QKVParallelLinear`）
> 配套参数加载：`python/sglang/srt/layers/parameter.py`（`_ColumnvLLMParameter.load_qkv_weight`）

`QKVParallelLinear` 是 SGLang 中用于 Attention 的 Q/K/V 投影层。它做了两件事：

1. **融合（Fusion）**：把 `q_proj` / `k_proj` / `v_proj` 三个矩阵沿输出维拼成一个大矩阵，一次 GEMM 算出 QKV。
2. **张量并行（TP）**：按**注意力头**为粒度做列并行切分，并处理 GQA/MQA 下 KV 头不够分时的**头复制**。

### 符号约定

下文统一使用如下记号（括号内为代码中的对应字段）：

| 符号 | 含义 | 代码字段 |
| --- | --- | --- |
| $H$ | hidden size | `hidden_size` |
| $n_q$ | 全局 Q 头数 | `total_num_heads` |
| $n_{kv}$ | 全局 KV 头数 | `total_num_kv_heads` |
| $d_h$ | 每个头的维度 | `head_size` |
| $d_v$ | V 头的维度（默认 $= d_h$） | `v_head_size` |
| $p$ | 张量并行度 | `tp_size` |
| $i$ | 当前 rank 编号，$0 \le i < p$ | `tp_rank` |
| $n_q^{\text{local}}$ | 本 rank 的 Q 头数 | `num_heads` |
| $n_{kv}^{\text{local}}$ | 本 rank 的 KV 头数 | `num_kv_heads` |
| $r$ | KV 头复制倍数 | `num_kv_head_replicas` |

---

## 1. 为什么要融合 QKV

Attention 的三个投影在数学上是独立的：

$$
Q = X W_q, \qquad K = X W_k, \qquad V = X W_v
$$

三者共享同一个输入 $X \in \mathbb{R}^{T \times H}$，其中 $T$ 为 token 数、$H$ 为 hidden size。各投影矩阵的形状为：

$$
W_q \in \mathbb{R}^{H \times n_q d_h}, \qquad
W_k \in \mathbb{R}^{H \times n_{kv} d_h}, \qquad
W_v \in \mathbb{R}^{H \times n_{kv} d_v}
$$

其中 $n_q$ 是 Q 头数、$n_{kv}$ 是 KV 头数、$d_h$ 是 head size、$d_v$ 是 v head size。

因为共享同一个输入，可以把三个权重沿输出维（列方向）拼接：

$$
W_{qkv} = \begin{bmatrix} W_q & W_k & W_v \end{bmatrix}
\in \mathbb{R}^{H \times (n_q d_h + n_{kv} d_h + n_{kv} d_v)}
$$

于是一次矩阵乘即可同时得到三者：

$$
\begin{bmatrix} Q & K & V \end{bmatrix} = X \, W_{qkv}
$$

**收益**：3 次 GEMM → 1 次 GEMM。对 decode 这种 $M$ 很小（batch 个 token）的瘦长矩阵乘，kernel 启动开销和权重读取的显存带宽是瓶颈，合并后 kernel 数减少、权重访问局部性更好，收益明显。

调用侧用一次 `split` 把结果拆开（`models/llama.py`）：

```209:210:python/sglang/srt/models/llama.py
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
```

注意这里的 `q_size` / `kv_size` 都是**本 rank 分片后**的大小，不是全局大小。

---

## 2. 为什么是「列并行」

`QKVParallelLinear` 继承自 `ColumnParallelLinear`。列并行沿输出维切分权重：

把权重按列切成 $p = \text{tp\_size}$ 份：

$$
W = \begin{bmatrix} W^{(0)} & W^{(1)} & \cdots & W^{(p-1)} \end{bmatrix}
$$

则第 $i$ 张卡计算 $Y^{(i)} = X W^{(i)}$，具有两个关键性质：

- 输入 $X$ 在所有 rank 上是**完整复制**的 → 计算前**不需要通信**；
- 输出 $Y^{(i)}$ 是**数值完整、形状切分**的，拼接即得完整结果：$Y = \begin{bmatrix} Y^{(0)} & \cdots & Y^{(p-1)} \end{bmatrix}$。

> 与行并行对比：行并行切输入维，每卡得到的是**部分和** $Y = \sum_{i} X^{(i)} W^{(i)}$，形状完整但数值不完整，因此必须 all-reduce。

这正好契合 Attention：每张卡拿到「若干个完整的注意力头」，就能独立地对这些头做 attention 计算，各头之间本来就没有耦合。算完之后由 `o_proj`（`RowParallelLinear`）消费这份切分输出，在末尾做一次 all-reduce。

所以整个 Attention 块的通信模式是：

```
hidden_states (完整)
   │  无通信
   ▼
QKVParallelLinear (列并行, gather_output=False)
   │  切分的 q/k/v (本 rank 的若干个头)
   ▼
RadixAttention  (逐头独立，无跨卡通信)
   │  切分的 attn_output
   ▼
RowParallelLinear o_proj (行并行, input_is_parallel=True)
   │  all-reduce  ← 整个 Attention 块唯一一次通信
   ▼
输出 (完整)
```

对应代码里 `gather_output=False` 是写死的：

```1141:1154:python/sglang/srt/layers/linear.py
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=False,
```

因为下游是行并行层，中间那次 all-gather 完全没必要。

---

## 3. 切分粒度：头，而不是任意列

这是 `QKVParallelLinear` 与 `MergedColumnParallelLinear`（gate_up_proj）最本质的区别。

`MergedColumnParallelLinear` 只要求每段能被 `tp_size` 整除，切在哪一列无所谓。而 QKV 的输出列具有**头结构**：连续 $d_h$ 列构成一个头，切分必须落在头边界上。即切点位置 $c$ 必须满足：

$$
c \equiv 0 \pmod{d_h}
$$

否则一个头会被劈成两半，而 attention 的 softmax 是在**单个头内**做归一化的：

$$
\mathrm{Attn}_j = \mathrm{softmax}\!\left(\frac{Q_j K_j^\top}{\sqrt{d_h}}\right) V_j
$$

跨卡劈开后就无法独立完成这个归一化，必须引入额外通信。

```1103:1114:python/sglang/srt/layers/linear.py
        # q 头常规切分：每个 rank 拿 total_num_heads / tp_size 个头。
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
```

---

## 4. GQA/MQA 与 KV 头复制

现代模型普遍用 GQA（Grouped Query Attention）/ MQA，KV 头数远少于 Q 头数，例如 Llama-3-8B：`num_heads=32, num_kv_heads=8`。

当 $p > n_{kv}$ 时（比如 8 个 KV 头跑 TP=16），KV 头**不够分**。SGLang 的处理是：每个 rank 至少保留 1 个 KV 头，并让若干 rank **共享同一个 KV 头的副本**。完整的分配规则为：

$$
n_q^{\text{local}} = \frac{n_q}{p}
$$

$$
\left(n_{kv}^{\text{local}},\; r\right) =
\begin{cases}
\left(1,\; \dfrac{p}{n_{kv}}\right), & p \ge n_{kv} \quad \text{(KV 头不够分，需复制)} \\[2ex]
\left(\dfrac{n_{kv}}{p},\; 1\right), & p < n_{kv} \quad \text{(KV 头足够，常规切分)}
\end{cases}
$$

其中 $r = \text{num\_kv\_head\_replicas}$ 是复制倍数。

rank $i$ 持有的 Q 头与 KV 头的索引集合分别为：

$$
\mathcal{Q}_i = \left\{\, i \cdot n_q^{\text{local}} + j \;\middle|\; 0 \le j < n_q^{\text{local}} \,\right\},
\qquad
\mathcal{K}_i = \left\{\, \left\lfloor i / r \right\rfloor \,\right\}
$$

注意 $\mathcal{Q}_i$ 两两不交（真切分），而 $\mathcal{K}_i$ 在 $r$ 个连续 rank 上取值相同（复制）。

### 具体例子

`total_num_heads=8, total_num_kv_heads=2, tp_size=4`，即 $n_q = 8,\; n_{kv} = 2,\; p = 4$：

| | 值 | 说明 |
| --- | --- | --- |
| `num_heads` | $n_q^{\text{local}} = 8/4 = 2$ | 每 rank 2 个 Q 头 |
| `num_kv_heads` | $n_{kv}^{\text{local}} = 1$ | $p=4 \ge n_{kv}=2$，兜底为 1 |
| `num_kv_head_replicas` | $r = 4/2 = 2$ | 每个 KV 头被 2 个 rank 共享 |

各 rank 持有的头：

| rank | Q 头 | KV 头 | KV 来源 |
| --- | --- | --- | --- |
| 0 | q0, q1 | kv0 | 源 KV 分片 0 |
| 1 | q2, q3 | kv0（副本） | 源 KV 分片 0 |
| 2 | q4, q5 | kv1 | 源 KV 分片 1 |
| 3 | q6, q7 | kv1（副本） | 源 KV 分片 1 |

**代价**：KV 权重和 KV Cache 都被冗余存了 $r$ 份。设单卡 KV Cache 为 $C_{\text{local}}$、全局总量为 $C_{\text{total}}$，则：

$$
C_{\text{total}} = p \cdot C_{\text{local}}
= p \cdot \underbrace{\frac{n_{kv}}{p} \cdot r}_{n_{kv}^{\text{local}} \cdot \text{单头开销}}
\;\propto\; r \cdot n_{kv}
$$

即当 $p \le n_{kv}$ 时 $r = 1$，总量与 TP 无关（单卡开销随 $p$ 线性下降）；而当 $p > n_{kv}$ 时：

$$
r = \frac{p}{n_{kv}} \;\Longrightarrow\;
C_{\text{local}} = \text{const}, \qquad C_{\text{total}} \propto p
$$

这是为了让每张卡都能独立完成自己 Q 头的 attention 而付出的显存代价——否则就要引入跨卡的 KV 通信，得不偿失。

> 这就是「TP 越大，KV Cache 并不会等比例变小」的原因：一旦 $p$ 越过 $n_{kv}$，单卡 KV Cache 就触底不再下降。

---

## 5. output_size 的「虚拟放大」

有一个容易困惑的点：

```1125:1136:python/sglang/srt/layers/linear.py
        output_size = (
            self.num_heads * self.head_size
            + self.num_kv_heads * self.head_size
            + self.num_kv_heads * self.v_head_size
        ) * tp_size
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,      # q_proj
            self.num_kv_heads * self.head_size * tp_size,   # k_proj
            self.num_kv_heads * self.v_head_size * tp_size, # v_proj
        ]
```

传给父类的 `output_size` 不是「真实的全局输出维」。真实值应该是：

$$
D_{\text{real}} = n_q d_h + n_{kv} d_h + n_{kv} d_v
$$

而代码里算的是**先取本 rank 的列数再乘 $p$**：

$$
D_{\text{nominal}} = \left( n_q^{\text{local}} d_h + n_{kv}^{\text{local}} d_h + n_{kv}^{\text{local}} d_v \right) \cdot p
= n_q d_h + r \, n_{kv} (d_h + d_v)
$$

两者差一个复制因子：

$$
D_{\text{nominal}} - D_{\text{real}} = (r - 1) \, n_{kv} (d_h + d_v) \;\ge\; 0
$$

当 $r = 1$ 时二者相等；当 $r > 1$ 时名义值**严格大于**真实值，因为它把每个 KV 副本都计入了。

上例中（$n_{kv}=2,\; r=2,\; d_v = d_h$）：
- 真实全局 KV 输出维：$2 d_h$（每路）
- 名义值：$n_{kv}^{\text{local}}(1) \times d_h \times p(4) = 4 d_h$（放大 2 倍）

**为什么要这样**：父类 `ColumnParallelLinear` 会无条件执行 $D / p$ 并断言整除。若传入 $D_{\text{real}}$，在 $r>1$ 时它未必能被 $p$ 整除（例如 $n_{kv} < p$）；而 $D_{\text{nominal}}$ 根据构造方式必然满足：

$$
\frac{D_{\text{nominal}}}{p} = n_q^{\text{local}} d_h + n_{kv}^{\text{local}} d_h + n_{kv}^{\text{local}} d_v \in \mathbb{Z}
$$

恰好就是本 rank 的真实分片大小。真实的头复制关系被封装在 `weight_loader` 里处理，父类完全不需要知道。

这也意味着：`self.output_size` 这个字段在 KV 复制场景下**没有物理意义**，不要拿它去推断 checkpoint 的形状。

---

## 6. 权重加载：两层偏移

Checkpoint 中通常是 `q_proj.weight` / `k_proj.weight` / `v_proj.weight` 三个独立张量，而运行时是一个融合参数。模型通过 `stacked_params_mapping` 建立映射：

```512:514:python/sglang/srt/models/llama.py
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
```

加载单个分量时需要**两层定位**：

### 第一层：目标偏移（写到融合参数的哪里）

在**本 rank 的本地坐标系**下计算，用 $n_q^{\text{local}}$ / $n_{kv}^{\text{local}}$：

$$
\text{offset}_{\text{dst}}(s) =
\begin{cases}
0, & s = q \\
n_q^{\text{local}} d_h, & s = k \\
\left(n_q^{\text{local}} + n_{kv}^{\text{local}}\right) d_h, & s = v
\end{cases}
\qquad
\text{size}(s) =
\begin{cases}
n_q^{\text{local}} d_h, & s = q \\
n_{kv}^{\text{local}} d_h, & s = k \\
n_{kv}^{\text{local}} d_v, & s = v
\end{cases}
$$

```1162:1168:python/sglang/srt/layers/linear.py
        shard_offset_mapping = {
            "q": 0,
            "k": self.num_heads * self.head_size,
            "v": (self.num_heads + self.num_kv_heads) * self.head_size,
            "total": (self.num_heads + self.num_kv_heads) * self.head_size
            + self.num_kv_heads * self.v_head_size,
        }
```

### 第二层：源偏移（从 checkpoint 张量的哪里读）

这是 QKV 层区别于普通合并列并行层的**核心**：

```1523:1527:python/sglang/srt/layers/linear.py
            if loaded_shard_id == "q":
                shard_id = self.tp_rank
            else:
                shard_id = self.tp_rank // self.num_kv_head_replicas
            start_idx = shard_id * shard_size
```

设本 rank 为 $i$，则源张量上的读取起点为：

$$
\text{offset}_{\text{src}}(s) = \text{size}(s) \cdot
\begin{cases}
i, & s = q \quad \text{(真切分)} \\[1ex]
\left\lfloor i / r \right\rfloor, & s \in \{k, v\} \quad \text{(复制)}
\end{cases}
$$

- **Q**：每个 rank 拿不同的头 → 直接用 $i$；
- **K/V**：$r$ 个 rank 共享同一份 → 用 $\lfloor i / r \rfloor$，使这些 rank 映射到源权重的**同一段**，从而自然实现「复制」。

验证：上例中 $r=2$，则 $\lfloor 0/2 \rfloor = \lfloor 1/2 \rfloor = 0$（rank 0/1 读同一段），$\lfloor 2/2 \rfloor = \lfloor 3/2 \rfloor = 1$（rank 2/3 读同一段），与前面的头分配表一致。

v2 加载路径把这个逻辑下沉到参数对象，`num_heads` 参数传的其实是复制倍数：

```1315:1326:python/sglang/srt/layers/linear.py
        # num_heads 传的是 num_kv_head_replicas：
        # 参数对象靠它把 tp_rank 换算成源权重中的 kv 头下标
        param.load_qkv_weight(
            loaded_weight=loaded_weight,
            num_heads=self.num_kv_head_replicas,
            shard_id=loaded_shard_id,
            ...
        )
```

对应 `parameter.py` 中的那一行：

```383:383:python/sglang/srt/layers/parameter.py
        shard_id = tp_rank if shard_id == "q" else tp_rank // num_heads
```

### 加载路径全景

| 场景 | 入口 | 说明 |
| --- | --- | --- |
| checkpoint 分开存 q/k/v | `weight_loader(param, w, "q"/"k"/"v")` | 主路径，直接单段写入 |
| checkpoint 已融合存 qkv | `weight_loader(param, w, None)` | 先按 `total_num_*` 拆成三段，再递归带 shard_id 回调（如 Phi-3-mini） |
| v2 参数体系 | `weight_loader_v2` | 本层只算 offset/size，切片交给 `param.load_qkv_weight` |
| block 量化 scale | `_load_qkv_block_scale` | 偏移与长度先从「通道数」换算为「block 数」 |
| per-tensor scale | `param.load_qkv_weight(shard_id=0)` | 全层一个标量，不分 q/k/v |
| 预切分权重 | `load_presharded_attn=True` | 跳过所有 narrow，直接拷贝 |

> 注意 `_load_fused_module_from_checkpoint` 与 `_load_qkv_block_scale` 中的偏移用的是 `total_num_*`（**全局坐标系**），而 `_get_shard_offset_mapping` 用的是 `num_*`（**本地坐标系**）。混淆这两套坐标系是读这段代码最常见的坑。

---

## 7. 特殊维度：v_head_size

部分模型（如 MLA 类结构）满足 $d_v \neq d_h$，因此本 rank 三段的列数分别为：

$$
\text{q\_proj\_shard\_size} = n_q^{\text{local}} d_h, \qquad
\text{kv\_proj\_shard\_size} = n_{kv}^{\text{local}} d_h, \qquad
\text{v\_proj\_shard\_size} = n_{kv}^{\text{local}} d_v
$$

代码里 V 段统一用 `v_head_size` 而非 `head_size`：

```1116:1118:python/sglang/srt/layers/linear.py
        self.q_proj_shard_size = self.num_heads * self.head_size
        self.kv_proj_shard_size = self.num_kv_heads * self.head_size
        self.v_proj_shard_size = self.num_kv_heads * self.v_head_size
```

默认 `v_head_size = head_size`，退化为常规情形。

---

## 8. 与其他线性层的对比

| 层 | 切分维 | 切分粒度 | 通信 | 典型用途 |
| --- | --- | --- | --- | --- |
| `ReplicatedLinear` | 不切 | — | 无 | 小层、router |
| `ColumnParallelLinear` | 输出（列） | 任意列 | 可选 all-gather | 通用列并行 |
| `MergedColumnParallelLinear` | 输出（列） | 任意列，分段 | 同上 | `gate_up_proj` |
| **`QKVParallelLinear`** | 输出（列） | **注意力头** | 无（`gather_output=False`） | `qkv_proj` |
| `RowParallelLinear` | 输入（行） | 任意行 | all-reduce | `o_proj`、`down_proj` |

关键差异总结：**`QKVParallelLinear` 是唯一一个「切分粒度受语义约束（必须按头）」且「支持分片复制（KV 副本）」的线性层。**

---

## 9. 自检问题

1. 为什么 QKV 用列并行、o_proj 用行并行？中间为什么不需要通信？
2. $p = 16,\; n_{kv} = 8$ 时，$n_{kv}^{\text{local}}$ 和 $r$ 分别是多少？rank 5 读的是源 KV 权重的第几个分片（即 $\lfloor 5/r \rfloor = ?$）？
3. 为什么 $D_{\text{nominal}} \ge D_{\text{real}}$？取等号的条件是什么？
4. 加载 `k_proj` 时，`shard_offset` 用 `num_heads` 还是 `total_num_heads`？`start_idx` 呢？为什么不同？
5. KV 头复制会带来哪些额外开销？为什么仍然选择复制而不是跨卡通信？

