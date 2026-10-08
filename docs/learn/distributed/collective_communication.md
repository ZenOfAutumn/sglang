# 分布式推理集合通信原语详解（Collective Communication）

> 本文系统介绍分布式推理中用到的所有通信原语（communication primitive），
> **按由简到繁的顺序**：point-to-point、broadcast、scatter、gather、reduce、
> all-gather、reduce-scatter、all-reduce、all-to-all，
> 逐个讲清**语义、示意图、数值示例、通信量、Ring 实现、在 SGLang 各并行方式中的用途**，
> 并给出与 TP/DP/EP/PP/CP 各文档的对应关系。
>
> 前置阅读：`TP.md`（all-reduce/all-gather）、`EP.md`（all-to-all）、`PP.md`（P2P）、`CP.md`（all-gather）。
>
> 延伸阅读：**`cost_model.md`**——本文大量使用的 $\alpha$-$\beta$ 代价模型的系统介绍，
> 以及 LogP、$\alpha$-$\beta$-$\gamma$、带宽层次、对分带宽、Roofline、BSP 等其他分析模型。

## 目录

1. [总览：一张表看懂所有原语](#1-总览一张表看懂所有原语)
2. [基础概念：rank / group / world](#2-基础概念-rank--group--world)
3. [Point-to-Point（P2P：send / recv）](#3-point-to-pointp2psend--recv)
4. [Broadcast（广播）](#4-broadcast广播)
5. [Scatter（分发）](#5-scatter分发)
6. [Gather（收集）](#6-gather收集)
7. [Reduce（归约）](#7-reduce归约)
8. [All-Gather（全收集）](#8-all-gather全收集) · [8.1 为什么要用 Ring？](#81-为什么要用-ring直接一轮全发不行吗)
9. [Reduce-Scatter（归约分发）](#9-reduce-scatter归约分发)
10. [All-Reduce（全归约）](#10-all-reduce全归约) · [Ring 通信量](#ring-all-reduce-通信量) · [通信下界与 Ring 最优性](#为什么-2fracp-1pn-是最优的all-reduce-的通信下界) · [$\alpha$-$\beta$ 代价模型](#完整代价模型为什么-ring-不总是最快)
11. [All-to-All（全交换）](#11-all-to-all全交换) · [11.1 EP token 级示例](#111-ep-token-级示例变长-all-to-all)（[EP 里「分片」是什么](#先厘清ep-里分片到底是什么)） · [11.2 通信量](#112-通信量)
12. [组合关系与通信量对比](#12-组合关系与通信量对比)
13. [SGLang 中的实现](#13-sglang-中的实现)

> **章节顺序按难度递进**：先一对一（P2P）→ 一对多（Broadcast / Scatter）→ 多对一（Gather / Reduce）
> → 多对多的两个基本块（All-Gather / Reduce-Scatter）→ 由它俩合成的 All-Reduce → 最重的 All-to-All。
> 每一节只依赖前面已讲过的原语，尤其是 **All-Reduce（§10）放在它的两个组成部分（§8、§9）之后**，
> 这样 Ring 实现才能直接复用前文的结论。若只关心 TP，可直跳 §10。

---

## 1. 总览：一张表看懂所有原语

设通信组内有 $p$ 个 rank，每个 rank 初始持有大小为 $N$ 的数据（除特别说明）。

表中行序即下文章节序，按**通信模式由简到繁**排列。


| § | 原语                | 通信模式         | 一句话语义                      | 输出在谁手上          | 是否做算术归约 | 典型通信量/rank单向                                | SGLang 用途                     | 示意图（3 个 rank）                        |
| -- | ------------------- | ---------------- | ------------------------------- | --------------------- | -------------- | -------------------------------------------------- | ------------------------------- | ------------------------------------------ |
| 3  | **P2P (send/recv)** | 1 → 1           | 一个 rank 发给另一个 rank       | 指定的目标 rank       | 否             | $N$                                                | PP 层间传激活                   | `[X] . .` → `. [X] .`                     |
| 4  | **Broadcast**       | 1 → N           | 一个 rank 把数据发给所有 rank   | 所有 rank             | 否             | $N$                                                | 权重/元数据同步                 | `[X] . .` → `[X] [X] [X]`                 |
| 5  | **Scatter**         | 1 → N           | 一个 rank 把不同分片发给各 rank | 所有 rank（各拿一片） | 否             | 非 root 收$N/p$；**root 发 $\frac{p-1}{p}N$**      | 少用                            | `[A                                        |
| 6  | **Gather**          | N → 1           | 各 rank 的分片汇总到一个 rank   | 单个 root rank        | 否             | 非 root 发$N$；**root 收 $(p-1)N$**                | logits 汇总                     | `[A] [B] [C]` → `[A                       |
| 7  | **Reduce**          | N → 1           | 所有 rank 数据求和到一个 rank   | 单个 root rank        | 是             | 树形约$2N$；扁平 root 收 $(p-1)N$                  | 少用                            | `[a] [b] [c]` → `[a+b+c] . .`             |
| 8  | **All-Gather**      | N → N           | 各 rank 的分片汇总且都拿到全部  | 所有 rank             | 否             | 每 rank 收发$(p-1)N=\frac{p-1}{p}N_{\text{total}}$ | **CP KV 收集**、SP、TP 输出拼接 | `[A] [B] [C]` → `[A                       |
| 9  | **Reduce-Scatter**  | N → N           | 求和后每 rank 只拿结果的一片    | 所有 rank（各拿一片） | 是             | $\frac{p-1}{p}N$                                   | SP、all-reduce 的前半           | `[a0                                       |
| 10 | **All-Reduce**      | N → N（复合）   | 所有 rank 数据求和且都拿到结果  | 所有 rank             | 是             | $2\frac{p-1}{p}N$                                  | **TP 层内求和**、DP 同步        | `[a] [b] [c]` → `[a+b+c] [a+b+c] [a+b+c]` |
| 11 | **All-to-All**      | N → N（全连接） | 每 rank 把不同分片发给不同 rank | 所有 rank（重排后）   | 否             | $\frac{p-1}{p}N$（**对分带宽 $\frac{p}{4}N$**）    | **EP dispatch/combine**         | `[a0                                       |

> 记忆线索：
>
> - 名字带 **All-** → 结果落在**所有** rank；不带 → 落在**单个** root。
> - 带 **Reduce** → 有算术求和；带 **Gather/Scatter/All-to-All** → 只搬运不求和。
> - **All-Reduce = Reduce-Scatter + All-Gather**（见 §10、§12）。
>
> 还可以把表里的原语两两配对着记：**Broadcast ↔ Reduce**（一发全收 / 全发一收）、
> **Scatter ↔ Gather**（拆开 / 合并）、**All-Gather ↔ Reduce-Scatter**（拼齐 / 求和后分掉）。

> **看“通信量”这列时注意两件事**：
>
> 1. **平均值会骗人**。root 系列（Scatter/Gather/Reduce）的流量**全部压在 root 一个口上**且只能串行，
>    随 $p$ 线性增长；All- 系列虽然搬的字节更多，却能摊到所有链路并行传，**耗时未必更长**（见 §6）。
> 2. **字节数相同不代表代价相同**。All-Gather / Reduce-Scatter / All-to-All 都是 $\frac{p-1}{p}N$，
>    但前两者只需相邻通信、能用 ring 流水，All-to-All 却要求**两两相连**的带宽（见 §11.2）。

---

## 2. 基础概念：rank / group / world

- **rank**：通信组内每个进程/GPU 的编号（0 ~ $p-1$）。
- **group（通信组）**：一组参与同一次集合通信的 rank。SGLang 为每种并行建独立 group：TP group、DP group、PP group、EP group、`_ATTN_CP` group 等（`distributed/parallel_state.py`）。
- **world**：所有进程的全集（world_size = 全局 GPU 数）。

同一张 GPU 可能同时属于多个 group（例如既在 TP group 又在 PP group），不同 group 做不同的集合通信。下面所有原语都是**在某个 group 内部**执行的。

### 贯穿全文的数值示例设定

为了让后面每个原语的数字可以横向对比，下文统一采用 **$p=3$ 个 rank**，每个 rank 持有一个长度为 3 的向量：

$$
x_0 = [1,\ 2,\ 3], \qquad x_1 = [10,\ 20,\ 30], \qquad x_2 = [100,\ 200,\ 300]
$$

这三个向量刻意取不同数量级（个位 / 十位 / 百位），这样求和结果 $[111, 222, 333]$ 一眼就能看出每一位是由谁贡献的：百位来自 rank 2、十位来自 rank 1、个位来自 rank 0。

**两种切分视角**（务必区分，后面反复用到）：

- **把向量当整体**（记作 $x_r$）：用于 broadcast / reduce / all-reduce / gather / all-gather，元素之间不拆开。
- **把向量按位置切成 3 片**（记作 $x_r[j]$，$j=0,1,2$）：用于 scatter / reduce-scatter / all-to-all，第 $j$ 片与 rank $j$ 对应。


| rank | 持有向量$x_r$     | 第0片$x_r[0]$ | 第1片$x_r[1]$ | 第2片$x_r[2]$ |
| ---- | ----------------- | ------------- | ------------- | ------------- |
| 0    | $[1, 2, 3]$       | 1             | 2             | 3             |
| 1    | $[10, 20, 30]$    | 10            | 20            | 30            |
| 2    | $[100, 200, 300]$ | 100           | 200           | 300           |

> 记住这张表的**行**（某个 rank 的全部数据）和**列**（所有 rank 的同一片）：
> **Reduce 类原语沿列求和**，**All-to-All 则是把行列转置**。

---

## 3. Point-to-Point（P2P：send / recv）

**语义**：最基础的通信——rank A 调用 `send` 把张量发给 rank B，rank B 调用 `recv` 接收。点对点，不涉及第三方。

```
rank 0            rank 1
 [X] ──send──────► [X]   (recv)
```

- **同步 send/recv**：阻塞直到完成；
- **异步 isend/irecv**：立即返回一个 handle，之后 `wait()`；用于计算与通信重叠。

**数值示例**（rank 0 发给 rank 1，rank 2 不参与）：


| rank | 通信前            | 动作          | 通信后                        |
| ---- | ----------------- | ------------- | ----------------------------- |
| 0    | $[1, 2, 3]$       | `send(dst=1)` | $[1, 2, 3]$（自己的数据不变） |
| 1    | $[10, 20, 30]$    | `recv(src=0)` | $[1, 2, 3]$（**被覆盖**）     |
| 2    | $[100, 200, 300]$ | 不参与        | $[100, 200, 300]$             |

两个要点：一是 P2P **不做任何算术**，接收方原有的 $[10,20,30]$ 是被直接覆盖而非相加；二是这是唯一**不要求组内所有 rank 都调用**的原语，rank 2 完全不参与，而下面 8 个集合原语必须所有 rank 一起调用，否则会挂起。

**通信量**：一次传 $N$，只占用两个 rank 之间的一条链路。

**SGLang 用途**：

- **PP（流水线并行）** 的核心通信——stage $i$ 算完后把激活（hidden states）P2P 发给 stage $i+1$（`PP.md` §5）。PP 通信稀疏、只在相邻 stage 间，所以能跨机。
- PD 分离下 KV Cache 的跨节点传输在更高层（Mooncake/RDMA），但底层也是点对点语义。

对应代码：`parallel_state.py` 的 `send` / `recv`、`send_tensor_dict` / `recv_tensor_dict`、`send_object` / `recv_object`。

---

## 4. Broadcast（广播）

**语义**：一个 root rank 把它的数据**原样复制**给组内所有其他 rank。结束后所有 rank 持有相同数据。

```
        rank 0 (root)
          [X]
     ┌─────┼─────┐
     ▼     ▼     ▼
  rank0  rank1  rank2
  [X]    [X]    [X]
```

**数值示例**（root = rank 0）：


| rank     | 通信前            | 通信后      |
| -------- | ----------------- | ----------- |
| 0 (root) | $[1, 2, 3]$       | $[1, 2, 3]$ |
| 1        | $[10, 20, 30]$    | $[1, 2, 3]$ |
| 2        | $[100, 200, 300]$ | $[1, 2, 3]$ |

注意 rank 1、2 原有的数据被**整体丢弃**：broadcast 是单向复制，非 root 的输入值不参与任何运算。结束后三个 rank 完全一致——这正是 TP 组内同步采样参数所需要的语义。

**通信量**：每个接收 rank 收到 $N$；Ring/tree 实现下 root 出口带宽是瓶颈，总量约 $(p-1)N$。

**SGLang 用途**：

- 采样参数、批次元数据、`tensor_dict` 从 TP rank 0 广播到其余 rank（保证 TP 组内一致）；
- 权重加载、随机种子同步等一次性同步。

对应代码：`broadcast`、`broadcast_object` / `broadcast_object_list`、`broadcast_tensor_dict`；进程间用共享内存的 `shm_broadcast.py`。

---

## 5. Scatter（分发）

**语义**：Broadcast 的"分片版"——一个 root rank 把一个大张量**切成 $p$ 片，每个 rank 拿到不同的一片**（对比 broadcast 是每个 rank 拿到相同的完整数据）。

```
      rank0 (root) [A|B|C]
     ┌──────┼──────┐  各发一片
     ▼      ▼      ▼
  rank0   rank1   rank2
   [A]     [B]     [C]
```

**数值示例**（root = rank 0，它持有一个长度 9 的大向量，切成 3 段）：


| rank     | 通信前                                   | 通信后            |
| -------- | ---------------------------------------- | ----------------- |
| 0 (root) | $[1, 2, 3,\ 10, 20, 30,\ 100, 200, 300]$ | $[1, 2, 3]$       |
| 1        | （无关）                                 | $[10, 20, 30]$    |
| 2        | （无关）                                 | $[100, 200, 300]$ |

与 broadcast 对比：broadcast 后每 rank 拿到完整的 9 个数，scatter 后每 rank 只拿 3 个且互不相同。
结束后三个 rank 的持有状态，恰好就是 §2 约定的初始设定 $x_0, x_1, x_2$——**后面所有多对多原语都从这个状态出发**。

**通信量**：记**输出**分片大小为 $M$（本例 $M=3$），root 手上的大张量共 $pM$（本例 9）。


|             | 发出                                       | 收到 |
| ----------- | ------------------------------------------ | ---- |
| **root**    | $(p-1)M$（给其他每人一片，自己那片不上网） | 0    |
| **非 root** | 0                                          | $M$  |

若换成总览表的记法（$N$ = root 手上的整个张量 $=pM$），root 出口就是 $\frac{p-1}{p}N$，每个非 root 收 $N/p$。

与 broadcast（§4）的关键区别：两者 root 出口都是 $(p-1) \times$（单份大小），但 broadcast 发的是**同一份数据**，因此可用 tree/ring 层层转发把 root 出口降到 $O(M\log p)$；而 scatter 每片**内容不同**，无论怎么组织，这 $(p-1)M$ 字节都必须从 root 真实出发——**root 出口带宽是硬瓶颈**，这也是实际不拿它做大规模通信的原因。

推理中单独使用较少，更多是作为 reduce-scatter / all-to-all 的组成语义出现。

---

## 6. Gather（收集）

**语义**：Scatter 的逆操作——各 rank 持有一个分片，全部**拼接**（不求和）到一个 root rank。

```
 rank0[A] rank1[B] rank2[C]
     └───────┼───────┘
             ▼ 拼接
      rank0 [A|B|C]   （只有 root 拿到）
```

**数值示例**（root = rank 0）：注意输出是长度 9 的**拼接**结果：


| rank     | 通信前            | 通信后                                   |
| -------- | ----------------- | ---------------------------------------- |
| 0 (root) | $[1, 2, 3]$       | $[1, 2, 3,\ 10, 20, 30,\ 100, 200, 300]$ |
| 1        | $[10, 20, 30]$    | $[10, 20, 30]$（**不变**）               |
| 2        | $[100, 200, 300]$ | $[100, 200, 300]$（**不变**）            |

把 §5 和本节连起来看：**scatter 之后紧接一次 gather，数据原封不动地回到 root**，二者互为逆运算。

**通信量**：完全是 scatter 的镜像（箭头反向）。记每 rank 的分片大小为 $M$：


|             | 发出 | 收到     |
| ----------- | ---- | -------- |
| **root**    | 0    | $(p-1)M$ |
| **非 root** | $M$  | 0        |

这里有个容易搞错的点：**总览表里记的 $N$ 是“单个 rank 发出的量”**（即 $M$），但真正决定耗时的是 **root 的入口 $(p-1)M$**——它随 $p$ **线性增长**，而每个非 root 只发了 $M$。所以看 gather 的开销要看 root，不能看平均值。

**与 all-gather（§8）的正确对比**：all-gather $=$ gather $+$ broadcast，因此**对任何一个节点，all-gather 的收发都 $\ge$ gather，不可能更省**：


|                        | 非 root 发 | 非 root 收 | root 发  | root 收  | 总字节    |
| ---------------------- | ---------- | ---------- | -------- | -------- | --------- |
| **Gather**             | $M$        | 0          | 0        | $(p-1)M$ | $(p-1)M$  |
| **All-Gather**（ring） | $(p-1)M$   | $(p-1)M$   | $(p-1)M$ | $(p-1)M$ | $p(p-1)M$ |

真正的差别不在**字节数**而在**串行度**：

> gather 的 $(p-1)M$ 全部经过 root 一个入口，只能串行灌入，耗时 $\approx \frac{(p-1)M}{B}$；
> ring all-gather 虽然每 rank 要收发 $(p-1)M$，但分成 $p-1$ 步、每步只传 $M$ 且 $p$ 条链路**同时**在传，
> 耗时 $\approx (p-1)\cdot\frac{M}{B}$。二者量级相当——**all-gather 多搬了 $p$ 倍的数据，却没多花多少时间**。

所以结论是「**多拿到全量结果几乎是白送的**」，而不是「all-gather 比 gather 快」。这也是工程上倾向 All- 系列的原因：反正时间差不多，不如让每张卡都拿到结果，省掉后面再 broadcast 一次。

**SGLang 用途**：logits 汇总等少数场景；更常用的是下一步要讲的 all-gather。

---

## 7. Reduce（归约）

**语义**：把所有 rank 的数据**按元素求和**（或 max/min/avg），结果只放到**一个 root rank**。这是本文第一个**做算术**的原语。

```
 rank0[a]  rank1[b]  rank2[c]
     └────────┼────────┘
              ▼  求和
        rank0 [a+b+c]   （只有 root 拿到）
```

**数值示例**（root = rank 0，op = SUM）：按元素（列）求和，$1+10+100=111$，$2+20+200=222$，$3+30+300=333$。


| rank     | 通信前            | 通信后                        |
| -------- | ----------------- | ----------------------------- |
| 0 (root) | $[1, 2, 3]$       | $[111, 222, 333]$             |
| 1        | $[10, 20, 30]$    | $[10, 20, 30]$（**不变**）    |
| 2        | $[100, 200, 300]$ | $[100, 200, 300]$（**不变**） |

换成其他 op 时 root 拿到的分别是：`MAX` → $[100, 200, 300]$，`MIN` → $[1, 2, 3]$，`AVG` → $[37, 74, 111]$。

**与 Gather 对比**（同样是 N → 1，差别只在做不做算术）：


|                | 输出长度         | 信息                                   |
| -------------- | ---------------- | -------------------------------------- |
| Gather（§6）  | $3 \times p = 9$ | **无损**，能区分每个数来自哪个 rank    |
| Reduce（本节） | $3$（不变）      | **被压缩**，只剩下和，无法还原各自贡献 |

**通信量**：记张量大小为 $N$（本例 $N=3$）。这里比 gather 多一层：因为**求和可以在中途做**，实现方式直接影响代价。


| 实现                                | root 入口流量 | 总步数（延迟）              |
| ----------------------------------- | ------------- | --------------------------- |
| **扁平**（所有 rank 直接发给 root） | $(p-1)N$      | 1 轮，但 root 拥塞          |
| **树形**（两两合并、逐层上传）      | $\approx 2N$  | $\lceil \log_2 p \rceil$ 轮 |

关键差别就在于**中间节点能把收到的两份先加成一份再往上传**，数据量不随层数膨胀（$N$ 进、$N$ 出）。而 gather（§6）做不到这点：它必须保留每份原始数据，越往上层包越大，总量雷打不动地是 $(p-1)N$。

> 一句话总结：**能先归约的通信比只能搬运的便宜**。同样的道理到 §10 会再出现一次：
> ring all-reduce 的第一阶段正是靠“边传边加”才把每步的传输量压在 $N/p$。

推理中很少单独用 Reduce（通常需要所有 rank 都拿到结果，故用 §10 的 All-Reduce）。

---

## 8. All-Gather（全收集）

**语义**：Gather 的"人人有份"版——各 rank 的分片拼接，且**每个 rank 都拿到完整拼接结果**。

```
 rank0[A] rank1[B] rank2[C]
     └───────┼───────┘ 拼接并分发回所有 rank
     ┌───────┼───────┐
[A|B|C]  [A|B|C]  [A|B|C]
```

**数值示例**：


| rank | 通信前            | 通信后                                   |
| ---- | ----------------- | ---------------------------------------- |
| 0    | $[1, 2, 3]$       | $[1, 2, 3,\ 10, 20, 30,\ 100, 200, 300]$ |
| 1    | $[10, 20, 30]$    | $[1, 2, 3,\ 10, 20, 30,\ 100, 200, 300]$ |
| 2    | $[100, 200, 300]$ | $[1, 2, 3,\ 10, 20, 30,\ 100, 200, 300]$ |

两个关键细节：

1. **拼接顺序按 rank 号升序**，与谁先到达无关——所以 CP 里 all-gather 回来的 KV 天然就是按序列顺序排好的（前提是分段按 rank 号切分）。
2. **输出长度从 3 变成 9**（$\times p$）。等到 §10 会看到，all-reduce 的输出长度仍是 3，这是两者显存开销的根本差别。

**通信量**：All-Gather 每 rank 约 $\frac{p-1}{p} N_\text{total}$（$N_\text{total}$ 为拼接后总大小）。**注意 All-Gather 不做求和，只搬运**，所以没有 all-reduce 的 2× 系数。

### 8.1 为什么要用 Ring？直接一轮全发不行吗

**行，而且小消息时往往更快**——这一点值得单独澄清，否则容易误以为 all-gather 天然必须走环。

关键区别在于：**All-Gather 没有归约，因此不存在 `recv → add → send` 的数据依赖链**（对比 §10 的 Reduce-Scatter）。每个 rank 手上的分片从一开始就是最终值，谁都可以立刻发给任何人。所以"一轮全发"（direct，又称 all-to-all broadcast）是完全合法的算法。

记每 rank 持有分片 $M$，两种实现的账：

| 实现 | 每 rank 发送量 | 轮次 | $\alpha$-$\beta$ 代价 |
| --- | --- | --- | --- |
| **Direct（一轮全发）** | $(p-1)M$ | **1** | $\alpha + (p-1)M\beta$ |
| **Ring** | $(p-1)M$ | $p-1$ | $(p-1)\alpha + (p-1)M\beta$ |

$$
T_{\text{direct}} = \alpha + (p-1)M\beta, \qquad T_{\text{ring}} = (p-1)\alpha + (p-1)M\beta
$$

**注意两者发送总量完全相同**（都是 $(p-1)M$，都在 §12 说的下界上），而 direct 少了 $(p-2)\alpha$。**单看这个模型，direct 严格优于 Ring。** 那 Ring 的意义何在？

#### Ring 的真正价值：拓扑友好，不是省字节

$\alpha$-$\beta$ 模型隐含假设"每 rank 有一条独立链路"，真实硬件里**一张卡的出口带宽 $B$ 是所有对端共享的**：

- **Direct**：rank 0 要在同一时刻往 $p-1$ 个方向各推 $M$，全部挤在它那**一个出口**上 → $\frac{(p-1)M}{B}$；
- **Ring**：任一时刻每卡只对**一个**邻居发 $M$，出口独占且 $p$ 条链路并行 → $(p-1)\cdot\frac{M}{B}$。

带宽项**仍然相同**。真正的差别在这里：


| | Direct | Ring |
| --- | --- | --- |
| 占用的链路 | 每对 rank 之间一条流，共 $p(p-1)$ 条 | 只用 $p$ 条**相邻**链路 |
| 拓扑要求 | **需要全互联**（NVSwitch） | 只需环状相邻，PCIe / 跨节点 IB 也能跑 |
| 单卡并发连接数 | $p-1$ | 1 |
| 接收端开销 | $p-1$ 套缓冲区 + $p-1$ 组同步 flag 轮询 | 1 套 |

在 NVSwitch 全互联的单机 8 卡上，direct 的 $p-1$ 条并发流各分到 $B/(p-1)$，总时间和 ring 打平；但在**非全互联**拓扑（老式 PCIe 树、跨节点、部分卡间无直连）上，direct 的某些流要走中转或共享上行链路，实际带宽远低于理论值，**Ring 则天然只走物理相邻链路，不挑拓扑**。

还有个隐藏成本：direct 下接收端要同时从 $p-1$ 个源收数，需要 $p-1$ 套接收缓冲区、$p-1$ 组同步 flag 轮询、kernel 内 $p-1$ 路并发 memcpy。**$p$ 一大，$\alpha$ 就不再是常数而退化成 $\alpha(p)$**——这也是 SGLang `custom_all_reduce` 只支持 world size ∈ {2,4,6,8} 的原因之一。

#### 选择结论


| 场景 | 更优实现 |
| --- | --- |
| 小消息 + 全互联（NVLink/NVSwitch） | **Direct 一轮**，省掉 $(p-2)\alpha$ |
| 大消息 | 带宽项相同，**Ring** 链路利用规整、易做 chunk 流水 |
| 非全互联 / 跨节点 | **Ring**，只依赖相邻链路 |
| $p$ 很大 | **Ring 或 tree**，direct 的并发连接与同步开销爆炸 |

NCCL 正是这么分流的：小 size 走 direct-like 路径，大 size 走 ring。

> **反过来印证 §10 的论证**：All-Gather 能一轮做完，说明"$p-1$ 轮串行"**不是 all-reduce 的固有属性**；
> Reduce-Scatter 之所以逼出串行，是因为它有**归约的数据依赖**。
> One-shot all-reduce 能压到 1 轮，正是因为两段都用 direct、放弃中途归约压缩——代价是每卡发 $(p-1)N$ 而非下界 $2\frac{p-1}{p}N$。

**SGLang 用途**：

- **CP（上下文并行）**：attention 阶段各卡把本段 KV all-gather 成完整序列 KV，本段 Q 才能 attend 到前面所有 token（`CP.md` §5，`cp_all_gather_reorganized_into_tensor`）；
- **Sequence Parallel / DP-attention**：把分散的 token 激活收集齐（`attn_tp_all_gather_into_tensor`，`SP.md` §8、`DP_attention.md`）；
- **TP**：`gather_output=True` 时把列并行输出 all-gather 拼回完整维；logits 按需 all-gather 拼成完整词表维（`TP.md` §7）。

对应代码：`all_gather`、`all_gather_into_tensor`、`all_gatherv`（变长）、`gather`、`all_gather_object`。

---

## 9. Reduce-Scatter（归约分发）

**语义**：先把所有 rank 的数据**按元素求和**，再把求和结果**切片**，每个 rank 只拿到结果的**一个分片**（相当于 Reduce + Scatter）。它与 §8 All-Gather 互为"对偶"，两者合起来就是下一节的 All-Reduce。

```
 rank0[a0 a1 a2]  rank1[b0 b1 b2]  rank2[c0 c1 c2]
        └──────── 按位求和 ────────┘
        s0=a0+b0+c0  s1=...  s2=...
        └──── 各 rank 只拿一片 ────┘
   rank0[s0]      rank1[s1]      rank2[s2]
```

**数值示例**（op = SUM）：先按列求和得到 $[111, 222, 333]$，再把这 3 个数按 rank 号分掉：


| rank | 通信前            | 通信后  | 说明                      |
| ---- | ----------------- | ------- | ------------------------- |
| 0    | $[1, 2, 3]$       | $[111]$ | 只拿第0位，$1{+}10{+}100$ |
| 1    | $[10, 20, 30]$    | $[222]$ | 只拿第1位，$2{+}20{+}200$ |
| 2    | $[100, 200, 300]$ | $[333]$ | 只拿第2位，$3{+}30{+}300$ |

**注意这里的输出恰好是"求和结果的三个碎片"**：把它们再做一次 §8 的 all-gather 就能拼回 $[111, 222, 333]$。这个观察就是下一节 All-Reduce 的全部秘密。

这也解释了 SP 省显存的原理：若后续计算（如 LayerNorm）只需要自己那一片 token，就没必要让每卡都存完整的 $[111,222,333]$，**激活显存降到 $1/p$**，而通信量只花了 all-reduce 的一半。

**通信量**：每 rank 约 $\frac{p-1}{p} N$，是 All-Reduce 的**前一半**。

**SGLang 用途**：

- **Sequence Parallel**：把 all-reduce 拆成 `reduce_scatter`（各 rank 只保留自己负责的 token 分片）+ 后续 `all_gather`，在序列维分摊 LayerNorm 等计算，通信量不变而激活显存降到 $1/p$（`SP.md` §4）；
- 是 Ring All-Reduce 的组成阶段（§10）。

对应代码：`reduce_scatter`、`reduce_scatter_tensor`、`reduce_scatterv`（变长）。

---

## 10. All-Reduce（全归约）

**语义**：所有 rank 的数据按元素求和，且**每个 rank 都拿到最终求和结果**。这是 TP 最核心的原语，也是前面几节的集大成者——**它等于 Reduce-Scatter（§9）接上 All-Gather（§8）**。

```
 rank0[a]  rank1[b]  rank2[c]
     └────────┼────────┘  求和并分发回所有 rank
     ┌────────┼────────┐
 rank0     rank1     rank2
[a+b+c]   [a+b+c]   [a+b+c]
```

**数值示例**（op = SUM）：求和结果与 §7 Reduce 相同，但**三个 rank 都拿到**：


| rank | 通信前            | 通信后            |
| ---- | ----------------- | ----------------- |
| 0    | $[1, 2, 3]$       | $[111, 222, 333]$ |
| 1    | $[10, 20, 30]$    | $[111, 222, 333]$ |
| 2    | $[100, 200, 300]$ | $[111, 222, 333]$ |

结果 $[111, 222, 333]$ 的每一位都同时含有三个 rank 的贡献，这正是 TP 行并行的物理意义：每张卡只算出**部分和**（partial sum），all-reduce 之后才是数学上正确的完整输出。

### Ring All-Reduce 通信量

高效实现是 **Ring All-Reduce = Reduce-Scatter + All-Gather** 两阶段，每阶段 $p-1$ 步，每步传 $N/p$：

$$
\text{单 rank 通信量} = 2 \cdot \frac{p-1}{p} \cdot N
$$

$N$ 是被 all-reduce 的张量大小（TP 中约 $s \times d_\text{model}$）。当 $p$ 增大，$\frac{p-1}{p} \to 1$，**每卡通信量趋于常数 $2N$，与 $p$ 无关**——这解释了为什么 TP 尚可扩展，但因为要走高带宽 NVLink，通常限制在单机 8 卡内（详见 `TP.md` §8）。

**Ring 分步数值追踪**（$p=3$，每 rank 的向量按位置切成 3 片，恰好每片 1 个数）：

**阶段一 Reduce-Scatter**（$p-1=2$ 步，每步把一片发给右邻居并累加）。约定 rank $r$ 在第 1 步发出第 $r$ 片，环为 $0 \to 1 \to 2 \to 0$：


| 步骤    | 传输内容                                                             | 累加结果                                                                                                 |
| ------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 初始    | —                                                                   | r0`[1, 2, 3]`，r1 `[10, 20, 30]`，r2 `[100, 200, 300]`                                                   |
| 第 1 步 | r0 的第0片`1` → r1；r1 的第1片 `20` → r2；r2 的第2片 `300` → r0   | r0 第2片$3{+}300{=}303$；r1 第0片 $10{+}1{=}11$；r2 第1片 $200{+}20{=}220$                               |
| 第 2 步 | r1 的第0片`11` → r2；r2 的第1片 `220` → r0；r0 的第2片 `303` → r1 | r2 第0片$100{+}11{=}\mathbf{111}$；r0 第1片 $2{+}220{=}\mathbf{222}$；r1 第2片 $30{+}303{=}\mathbf{333}$ |

此时每个 rank 各**只**持有最终结果的一片，且恰好互不重复：rank 0 有第1片 `222`，rank 1 有第2片 `333`，rank 2 有第0片 `111`。这与 §9 reduce-scatter 的输出完全一致（只是分片归属的 rank 编号因环的走向而轮转）。

**阶段二 All-Gather**（再 $p-1=2$ 步，把这三片无算术地转一圈复制齐）：


| 步骤    | 各 rank 已集齐的片                      |
| ------- | --------------------------------------- |
| 起点    | r0`{1:222}`，r1 `{2:333}`，r2 `{0:111}` |
| 第 1 步 | r0`{1,2}`，r1 `{2,0}`，r2 `{0,1}`       |
| 第 2 步 | 三者均为`{0,1,2}` = $[111, 222, 333]$   |

合计 $2(p-1) = 4$ 步、每步传 $N/p$，与上面的公式 $2\frac{p-1}{p}N$ 一致。关键在于**每一步每张卡只发送 $1/p$ 的数据**，而不是把整个张量广播出去。

### 为什么 $2\frac{p-1}{p}N$ 是最优的：All-Reduce 的通信下界

上面只是"Ring 恰好花了这么多"，还需要回答一个更强的问题：**有没有比它更省的算法？** 答案是没有——$2\frac{p-1}{p}N$ 就是**带宽项的理论下界**，Ring 达到了它。下面分两步证明。

先说清楚**证什么**：下界是对「**任意一个 rank 在整个算法期间的收/发字节数**」而言的，不是对集群总量。因为所有 rank 对称、且它们并行工作，单卡的收发量才决定墙钟时间。

#### 下界一：All-Gather 段——每个 rank 至少要收 $\frac{p-1}{p}N$

**从"结束状态"倒推是最容易的切入点**：all-reduce 结束时，每个 rank 手上都必须有完整的 $N$ 个结果值。

把这 $N$ 个结果按任意方式切成 $p$ 片。对 rank $r$ 来说，**至多有一片**是它"可能自己算出来"的（后面会看到，最好情况下正好是一片）；剩下 $\frac{p-1}{p}N$ 的结果字节，它自己**无论如何都算不出来**——因为算出任何一个结果值 $y[j]=\sum_q x_q[j]$ 都需要全部 $p$ 个 rank 的输入，而 rank $r$ 只有 $x_r$。

于是：

> 这 $\frac{p-1}{p}N$ 字节**只能从网络进来**，$\Rightarrow$ **每 rank 接收量 $\ge \frac{p-1}{p}N$**。

关键是**这一段完全不可压缩**：$N$ 个结果值互不相同、彼此无关，不存在"传一份顶两份"的可能。所以这是个**硬下界**。Ring All-Gather 的 $p-1$ 步、每步 $N/p$，合计恰好 $\frac{p-1}{p}N$——**打平**。

> 注意这里只说"**字节数**达到下界"，没说必须用 ring。All-Gather 没有归约依赖，**direct 一轮全发同样只发 $\frac{p-1}{p}N$、却只需 1 轮**（详见 [§8.1](#81-为什么要用-ring直接一轮全发不行吗)）。真正强制串行的是下界二的归约段。

#### 下界二：Reduce-Scatter 段——每个 rank 至少要发 $\frac{p-1}{p}N$

现在看**发送**方向，用对称性论证：

rank $r$ 的输入 $x_r$（$N$ 个值）必须影响到**其他每一个 rank** 的最终输出——否则改动 $x_r$ 时那个 rank 的结果不变，答案就错了。所以 $x_r$ 的信息必须"离开" rank $r$。

那 rank $r$ 最少要发多少？这里**存在压缩空间**：$x_r$ 不必原样发 $p-1$ 遍，中间节点可以先把收到的和自己的加起来（$N$ 进 $N$ 出，见 §7 树形 reduce 的道理），所以远远不需要 $(p-1)N$。

紧的界来自**负载均衡**：全局要产出 $N$ 个结果值，最优方案是让 $p$ 个 rank **各负责 $N/p$ 个位置的求和**（这正是 Reduce-Scatter 的分片语义）。对 rank $r$：

- 它**自己负责的那 $N/p$ 片**：不用发出去，在本地累加；
- **其余 $\frac{p-1}{p}N$ 的位置**：它持有的对应输入值对别人是必需的，必须发出去（可以先与途中数据合并，但字节数压不下去，因为每个位置的贡献都要抵达其负责者）。

$\Rightarrow$ **每 rank 发送量 $\ge \frac{p-1}{p}N$**。Ring Reduce-Scatter 的 $p-1$ 步 × $N/p$，同样**打平**。

> 顺带解释了下界一里"至多一片"的来源：既然最优分配是每 rank 负责 $1/p$，那它能自产的结果就正好是 $N/p$，剩下 $\frac{p-1}{p}N$ 必须收。两个下界共用同一套分片假设，所以能相加。

两段相加：

$$
T_{\text{AllReduce}}^{\text{lower}} \;=\; \underbrace{\frac{p-1}{p}N}_{\text{Reduce-Scatter}} + \underbrace{\frac{p-1}{p}N}_{\text{All-Gather}} \;=\; 2\,\frac{p-1}{p}N
$$

> 这里隐含一个关键前提：**Reduce-Scatter 这个中间态是必要的**。直觉上，all-reduce 的输出人人相同，可以在"某个时刻全局结果第一次成型"处切一刀——在那之前是归约（可压缩），之后是复制（不可压缩）。任何算法都必须付这两段的钱。

#### 与其他实现对比：省不掉，但可以搬到别处

| 算法 | 每 rank 发送量 | 轮次（延迟） | 备注 |
| --- | --- | --- | --- |
| **朴素全交换**（每人把 $N$ 发给所有人，本地求和） | $(p-1)N$ | 1 | 比下界多 $\frac{p}{2}$ 倍，$p$ 大时不可用 |
| **Reduce-to-root + Broadcast**（扁平） | root 收发各 $(p-1)N$ | 2 | 总量看似 $2N$，但**全压在 root 一个口**，随 $p$ 线性恶化 |
| **Recursive Halving-Doubling** | $2\frac{p-1}{p}N$ | $2\log_2 p$ | **同样达到带宽下界**，且延迟更低；但要求 $p$ 是 2 的幂 |
| **Ring** | $2\frac{p-1}{p}N$ | $2(p-1)$ | 达到下界，任意 $p$ 都适用，只需相邻链路 |
| **Tree / One-shot（小消息）** | $>2\frac{p-1}{p}N$ | $O(\log p)$ 或 1 | **故意超出带宽下界，换更少的轮次** |

可以看到：**带宽项 $2\frac{p-1}{p}N$ 无法被任何算法击败**，不同算法真正在权衡的是**轮次（延迟项）**——下一小节说明为什么轮次无法被压缩。

#### 完整代价模型：为什么 Ring 不总是最快

单看带宽是不够的，标准的 $\alpha$-$\beta$ 模型是（模型本身的系统介绍、参数怎么测、以及 LogP / Roofline / 对分带宽等其他模型，见 `cost_model.md`）：

$$
T = \underbrace{2(p-1)\,\alpha}_{\text{延迟项}} \;+\; \underbrace{2\frac{p-1}{p}N\,\beta}_{\text{带宽项（已达下界）}}
$$

其中 $\alpha$ 是**一步**通信的固定延迟（kernel launch、对端就绪的 flag 轮询/信号量同步、链路固定开销，NVLink 上合计约几 μs），$\beta = 1/B$ 是每字节传输代价。

##### 为什么延迟项是 $2(p-1)\alpha$：轮次串行，而非握手串行

这里有个常见误读需要澄清：**$2(p-1)$ 指的是"轮次数"，不是"握手次数"**。Ring 的某一步内部，$p$ 个 rank 是**同时**在收发的：

```
第 k 步：  r0 ──► r1 ──► r2 ──► r0     （p 条链路并行，只付 1 个 α）
```

所以 $\alpha$ 不是 $p$ 次握手的叠加，一步就是一个 $\alpha$。真正无法压缩的是**步与步之间的数据依赖**，每个 rank 上都存在这样一条 happens-before 链：

$$
\text{recv}_k \;\longrightarrow\; \text{add}_k \;\longrightarrow\; \text{send}_{k+1}
$$

回看前面 Ring 分步追踪那张表就一目了然：**rank 1 在第 2 步要发给 r2 的那个 `11`，是它在第 1 步收到 r0 的 `1` 之后才算出来的（$10+1=11$）**。第 1 步的加法没做完，第 2 步要发的数据在物理上**根本不存在**。

所以即便带宽无限、即便把所有通信都换成异步 `isend`，这 $p-1$ 步也**只能依次发生**。异步只能让通信与*无关*的计算重叠，消不掉这条链上的依赖。

换个角度看更本质：**Ring 拓扑的直径是 $p-1$**——rank 0 的数据要"走到"环上最远的 rank，最少需要这么多跳，每跳至少付一次 $\alpha$。延迟项本质上是**拓扑直径决定的下界**，与带宽无关。

> **两个半程的串行原因并不相同**，值得区分清楚：
>
> - **Reduce-Scatter 段**：`recv → add → send` 是**算法内在的数据依赖**，换任何拓扑都消不掉（只能改变"跳数"，如 halving-doubling 降到 $\log_2 p$）；
> - **All-Gather 段**：**没有归约、没有依赖**，分片一开始就是最终值。它走 $p-1$ 轮**纯粹是"选了 ring 这个拓扑"的结果**——改用 direct 一轮全发，同样只发 $\frac{p-1}{p}N$ 却只需 1 轮（见 [§8.1](#81-为什么要用-ring直接一轮全发不行吗)）。
>
> 换句话说，$2(p-1)$ 里**只有前一半是"必须付"的**，后一半是 ring 为了拓扑友好性主动选择的代价。这正是 one-shot all-reduce 能压到 1~2 轮的空间所在。

> **NCCL 的缓解手段与它的极限**：实现上会把大 tensor 切成多个 chunk 做**流水线**——chunk A 的第 $k+1$ 步可以和 chunk B 的第 $k$ 步重叠，从而把 $\alpha$ 摊薄到接近 $\frac{2(p-1)\alpha}{\text{chunk 数}}$。
> 但 chunk 数受 tensor 大小限制：**小消息切不出足够多的 chunk，流水线填不满，$2(p-1)\alpha$ 就实打实地暴露出来**。这才是小消息下 Ring 变慢的真正机理。

各算法的轮次差异，根源也在拓扑直径：

| 算法 | 轮次 | 为什么能更少 |
| --- | --- | --- |
| **Ring** | $2(p-1)$ | 环的直径 $p-1$，信息只能逐跳接力 |
| **Halving-Doubling** | $2\log_2 p$ | 超立方拓扑，每步通信距离**翻倍**，直径仅 $\log_2 p$ |
| **One-shot（NVLink 直写）** | $1$ | 全互联下每卡直接写所有对端，**没有中转就没有依赖链** |

回到消息大小的选择：

- **大消息（prefill、大 batch）**：$N\beta$ 主导，且 chunk 流水线能摊薄 $\alpha$ → **Ring 最优**。
- **小消息（decode，$N$ 只有几十 KB）**：$2(p-1)\alpha$ 反而主导，此时宁可**多传字节、少走轮次**：
  - **Halving-Doubling**：$2\log_2 p = 6$ 轮（$p=8$），带宽仍在下界上，通常是更好的默认；
  - **One-shot / two-shot**：1~2 轮，每卡发 $(p-1)N$ 远超下界，但 $N$ 很小时多发的字节几乎不花时间。

代入 $p=8$、$\alpha \approx 5\,\mu s$ 直观感受一下：Ring 的延迟项 $=14 \times 5 = 70\,\mu s$，one-shot 仅 $5\,\mu s$。decode 阶段每个 Transformer block 有 2 次 all-reduce，几十层叠加后差距达到毫秒量级——对 TPOT 影响显著。

这正是 SGLang 里 `custom_all_reduce` 存在的理由——它按 tensor 字节数设阈值，小 size 走自研的 one-shot/two-shot NVLink 直写路径，大 size 回落到 NCCL（NCCL 内部再按 size 在 ring / tree / CollNet 之间选）。相关判断逻辑见 `distributed/device_communicators/custom_all_reduce.py` 的 `should_custom_ar`（阈值常量 `_MAX_CAR_SIZE`：CUDA 8 MB、ROCm 16 MB、MUSA 128 MB；且仅支持 world size ∈ {2,4,6,8}）。

#### 一句话总结

> **All-Reduce 的带宽下界是 $2\frac{p-1}{p}N$，来自"归约一半（可压缩）+ 广播一半（不可压缩）"这个绕不开的结构；Ring 用 $2(p-1)$ 轮恰好打平这个下界，代价是延迟项随 $p$ 线性增长——而这 $2(p-1)$ 轮的串行性来自环拓扑的直径与 `recv → add → send` 的数据依赖，不是带宽问题。**
>
> **所以优化 all-reduce 的空间不在"少传字节"（已到下界），而在四个方向："少走轮次"（halving-doubling / one-shot，用超额带宽换延迟）、"摊薄轮次开销"（chunk 流水线）、"压缩字节"（FP8 通信）、或"干脆别做完整 all-reduce"（SP 只做 Reduce-Scatter，见 §9）。**

**SGLang 用途**：

- **TP**：每个 Transformer block 2 次 all-reduce（attention 的 `o_proj` 行并行后、MLP 的 `down_proj` 行并行后），`VocabParallelEmbedding` 也用 all-reduce（`TP.md` §7）；
- **DP**：副本间的梯度/统计同步（推理侧主要是 DP-attention 相关的 token 数同步）。

对应代码：`parallel_state.py: all_reduce`；高性能后端 `device_communicators/`：`pynccl.py`（NCCL）、`custom_all_reduce.py` / `quick_all_reduce.py`（自定义低延迟 all-reduce）、`pymscclpp.py`。

---

## 11. All-to-All（全交换）

**语义**：最"全"的通信——**每个 rank 都把自己的数据切成 $p$ 份，第 $j$ 份发给 rank $j$；同时从每个 rank 收一份。相当于一次分布式的"矩阵转置"。**

```
发送前（行=持有者，列=目标）      接收后
 rank0: [→0 →1 →2]              rank0: 收到各 rank 的第0片
 rank1: [→0 →1 →2]      ==>     rank1: 收到各 rank 的第1片
 rank2: [→0 →1 →2]              rank2: 收到各 rank 的第2片
```

**数值示例**：每 rank 把自己的 3 个数拆开，第 $j$ 个发给 rank $j$：


| rank | 通信前            | 发出                            | 通信后         |
| ---- | ----------------- | ------------------------------- | -------------- |
| 0    | $[1, 2, 3]$       | `1`→r0, `2`→r1, `3`→r2       | $[1, 10, 100]$ |
| 1    | $[10, 20, 30]$    | `10`→r0, `20`→r1, `30`→r2    | $[2, 20, 200]$ |
| 2    | $[100, 200, 300]$ | `100`→r0, `200`→r1, `300`→r2 | $[3, 30, 300]$ |

把输入输出写成矩阵，一眼看出这就是**转置**：

$$
\begin{bmatrix} 1 & 2 & 3 \\ 10 & 20 & 30 \\ 100 & 200 & 300 \end{bmatrix}
\xrightarrow{\ \text{all-to-all}\ }
\begin{bmatrix} 1 & 10 & 100 \\ 2 & 20 & 200 \\ 3 & 30 & 300 \end{bmatrix}
$$

对比记忆：**all-gather（§8）是每 rank 拿到整个矩阵（9 个数），all-to-all 是每 rank 只拿一列（3 个数）**。这正是 EP dispatch 的语义：每张卡把本地 token 按选中的 expert 分类，第 $j$ 类发给持有 expert $j$ 的卡；专家算完后再做一次方向相反的 all-to-all（combine）把结果送回——**再转置一次就回到原布局**。

### 11.1 EP token 级示例（变长 all-to-all）

上面等长转置是理想情形。真实 MoE 里每个 token 选 top-k 个 expert、且各 expert 冷热不均，所以**每 rank 发往不同目标的条数不相等**，必须用变长版本。下面这个例子把 dispatch/combine 的缓冲区布局完整走一遍。

> 与 `EP.md` §5 的分工：那边讲**路由决策**（哪个 token 该去哪个 expert，top-1 场景）；
> 这里讲**通信本身**（top-k 复制、缓冲区排序、split 向量怎么算）。

#### 先厘清：EP 里「分片」到底是什么

常见疑问是「EP 的 all-to-all，token 就是分片吗？」——**是，token 是切分的最小单位，但它与标准 all-to-all 的「均匀分片」有本质差别**：


|                | 标准 All-to-All（上文 $3\times3$ 转置） | EP dispatch（本节）                                  |
| -------------- | --------------------------------------- | ---------------------------------------------------- |
| 分片依据       | **位置**：第 $j$ 片固定发给 rank $j$    | **路由**：token → top-k expert → expert 所在 rank    |
| 每块大小       | 静态均匀，恒为 $N/p$                    | **动态不均**，$c_{ij}$ 每步 forward 都变              |
| 一份数据的去向 | 唯一目的地                              | **可复制 $k$ 份**，分别去往不同 rank                  |
| 接收方是否已知 | 已知（$N/p$）                           | **不知道**，必须先交换计数（第 2 步）                 |
| 对应 API       | `all_to_all_single`（等长）             | `all_to_all_single` + split 向量（**all-to-all-v**）  |

形式化地说：rank $i$ 发给 rank $j$ 的条数

$$
c_{ij} = \bigl|\{\,(t, e) : t \in \text{tokens}(i),\ e \in \text{top-}k(t),\ \text{owner}(e) = j \,\}\bigr|
$$

由 router 在**运行时**决定，因此 $c_{ij}$ 构成的矩阵每步都不同。标准 all-to-all 只是 $c_{ij} \equiv N/p$ 的特例。

还有一点容易混淆——**切的是哪个维度**：

$$
\text{EP（本节）}:\ [N_{\text{local}},\, H] \to [N_{\text{recv}},\, H]
\qquad
\text{SP/Ulysses}:\ [S,\, H/p] \to [S/p,\, H]
$$

EP 的 all-to-all **只在 token 维上重排，hidden 维 $H$ 始终完整不切**；切分口径从「按 rank 本地持有」变成「按 expert 归属」。而 `SP.md` 里 Ulysses 那种「seq 维 ↔ head 维互换」是另一回事，别混为一谈。

正因为分片是路由决定的变长块，才多出了下面第 1、2 步（permute 与交换 split）——等长 all-to-all 完全不需要这两步。

**设定**：EP $p=2$ 卡，$E=4$ 个 expert，**top-k=2**，共 4 个 token。

- **rank 0** 持有 E0、E1；**rank 1** 持有 E2、E3（即 expert $e$ 归属 rank $\lfloor e/2 \rfloor$）；
- token 初始分布：rank 0 有 A、B；rank 1 有 C、D；
- router 结果（每个 token 选 2 个 expert）：


| token | 所在卡 | 选中的 expert |
| ----- | ------ | ------------- |
| A     | rank 0 | E0, E2        |
| B     | rank 0 | E0, E1        |
| C     | rank 1 | E1, E2        |
| D     | rank 1 | E2, E3        |

**第 1 步：本地按 expert 排序（permute）**。top-k=2 意味着**一个 token 会被复制成 2 份**，各自跟着自己的 expert 走。每卡把 $s \times k$ 条记录按「目标 rank → expert 号」排序，使发往同一个 rank 的记录在缓冲区里连续——这是 `all_to_all_single` 的硬性要求：


| rank | 排序后的发送缓冲区（token@expert） | → rank 0 的段 | → rank 1 的段 | `input_split_sizes` |
| ---- | ---------------------------------- | -------------- | -------------- | ------------------- |
| 0    | `A@E0, B@E0, B@E1 ‖ A@E2`         | 3 条           | 1 条           | `[3, 1]`            |
| 1    | `C@E1 ‖ C@E2, D@E2, D@E3`         | 1 条           | 3 条           | `[1, 3]`            |

**第 2 步：交换 split（一次小的 all-to-all）**。接收方事先不知道会收到多少条，所以先用一次 int64 的 all-to-all 交换计数，得到 `output_split_sizes`，再据此分配接收缓冲区。这就是 DeepEP 里 `get_dispatch_layout` / `num_tokens_per_rank` 在做的事。

**第 3 步：dispatch（变长 all-to-all）**：


| rank | `input_split_sizes` | `output_split_sizes` | 收到的记录               | 本地要算的 expert      |
| ---- | ------------------- | -------------------- | ------------------------ | ---------------------- |
| 0    | `[3, 1]`            | `[3, 1]`             | `A@E0, B@E0, B@E1, C@E1` | E0: {A, B}；E1: {B, C} |
| 1    | `[1, 3]`            | `[1, 3]`             | `A@E2, C@E2, D@E2, D@E3` | E2: {A, C, D}；E3: {D} |

几个要点：

- **收发 split 互为转置**：rank 0 的 `input_split_sizes[1] = 1` 恰好等于 rank 1 的 `output_split_sizes[0] = 1`。整体仍是"矩阵转置"，只是每块大小不同。
- **负载不均一目了然**：E2 收到 3 个 token，E3 只收到 1 个。这就是 EPLB（`EP.md` §6）要解决的问题——热点 expert 拖慢它所在的卡。
- **token 被复制**：A 同时出现在 rank 0（走 E0）和 rank 1（走 E2），B 的两份都留在 rank 0。总记录数 $= s \times k = 4 \times 2 = 8$，而不是 4。

**第 4 步：专家计算 + combine**。各卡在本地 expert 上算完后，做一次**方向完全相反**的 all-to-all——把上面的 `input/output_split_sizes` 对调即可，每条结果原路返回：

```
combine 的 split 就是 dispatch 的镜像：
  rank 0:  send [3, 1]  recv [3, 1]   （与 dispatch 相同，因为本例对称）
  一般情形： combine.input_splits == dispatch.output_splits
            combine.output_splits == dispatch.input_splits
```

回到原卡后，每个 token 再把自己 k 份专家输出按 `topk_weights` **加权求和**，得到最终输出。至此张量布局与进入 MoE 层之前完全一致，后续层无感知。

把四步的张量形状串起来看，就是 token 维被反复重排、hidden 维始终不动：

```
dispatch:  [N_local, k, H] --permute--> [Σ_j c_ij, H] --a2a-v--> [N_recv, H]
           （top-k 复制）              （按目标 rank 分段连续）  （本地 expert 计算）

combine:   [N_recv, H]     --a2a-v-->   [Σ_j c_ij, H] --unpermute + 按 topk_weights 加权求和-->
           [N_local, H]
```

所以 EP 的一轮 dispatch+combine 可以概括成：**按 router 结果把 token 变长地「转置」到 expert 所在卡，算完再转置回来**；发送总量是 $k \times N \times H$ 量级而非 $N \times H$（top-k 复制的代价，见 §11.2 的修正项）。

> 上述 split 数值与"combine 后恢复原布局"已用 `torch.distributed.all_to_all_single`（gloo，2 进程）实际跑通验证。
>
> 真实实现比这复杂：DeepEP 会把 permute、split 交换、通信、FP8 量化融合进 CUDA kernel，并用 NVSHMEM 做 low-latency 模式；但**语义就是上面这两次变长 all-to-all**。

### 11.2 通信量

记每 rank 持有 $N$ 数据、切成 $p$ 片各 $N/p$。每 rank 把 $p-1$ 片发出去（第 $r$ 片留给自己），同时收进 $p-1$ 片：

$$
\text{单 rank 发出} = \text{单 rank 收到} = \frac{p-1}{p} N
$$

数字上与 all-gather、reduce-scatter 相同，**但它是这三者里最贵的**，原因不在总量而在流量的**分布形态**：


| 原语                        | 每 rank 收发     | 流量形态                                        | 可否降级到 ring（只用相邻链路） |
| --------------------------- | ---------------- | ----------------------------------------------- | -------------------------- |
| All-Gather / Reduce-Scatter | $\frac{p-1}{p}N$ | 每片要送达**所有** rank，可沿途转发复用          | **可以**，$p-1$ 步流水     |
| **All-to-All**              | $\frac{p-1}{p}N$ | **每对 rank 之间都有独立流量**（$p(p-1)$ 条流） | **不行**，每片目的地都不同 |

关键在于 all-gather 传的是**同一份数据的副本、要发给所有人**，因此可以沿环接力转发（$A$ 的数据经 $B$ 中转到 $C$，$B$ 顺带也拿到了）；而 all-to-all 每一片**只有唯一的目的地**，中转对第三方毫无价值，无法靠转发复用，必须真正打满两两之间的连接。

> 这里说的是"**能否降级到只用相邻链路**"，不是"必须用 ring"。§8.1 已说明 all-gather 在全互联小消息下用 direct 一轮更快；
> 而 all-to-all **没有这个选择余地**——它本质上就要求两两直连的带宽，这才是它最贵的原因。

因此衡量它的指标是**对分带宽（bisection bandwidth）**：把 $p$ 张卡任意均分成两半，跨越切面的流量约为

$$
\text{跨切面流量} \approx \frac{p}{2} \times \frac{p/2}{p} N = \frac{p}{4} N
$$

即**随 $p$ 线性增长**。这正是 EP 规模一大就必须上高对分带宽 IB/RoCE 组网（而 TP 靠机内 NVLink 就够）的根本原因——all-reduce 的每卡通信量趋于常数 $2N$，all-to-all 的对分压力却随卡数线性上升。

**EP 场景的两点修正**：

1. **乘 top-k 因子**：每个 token 复制 $k$ 份（§11.1 中 $k=2$），实际载荷是 $k \times$（token 数 × hidden_size）。
2. **dispatch 与 combine 字节数不等**：SGLang 里 dispatch 常用 FP8（省带宽）、combine 用 BF16（保精度），故 combine 通常更重（见 `EP.md` §8）。

**SGLang 用途**：

- **EP（专家并行）** 的核心——**dispatch**（all-to-all #1：把 token 按其选中 expert 发到对应卡）和 **combine**（all-to-all #2：把专家输出送回 token 原卡）。跨节点 EP（如 `ep_size=128`）对 RDMA/IB 带宽要求极高（`EP.md` §4、§8）。

对应代码：`parallel_state.py: all_to_all_single`；EP 专用高性能后端在 `layers/moe/token_dispatcher/`（DeepEP、Mooncake、NIXL、Mori、FlashInfer）。

---

## 12. 组合关系与通信量对比

### 关键恒等式

```
All-Reduce  =  Reduce-Scatter  +  All-Gather      （§10 = §9 + §8）
（求和分片）      （拼回完整）

Reduce      =  Reduce-Scatter  +  Gather(到root)   （§7  = §9 + §6）
Broadcast   ≈  Scatter         +  All-Gather       （§4  ≈ §5 + §8，概念上）
```

这解释了为什么 All-Reduce 通信量恰好是 Reduce-Scatter（$\frac{p-1}{p}N$）+ All-Gather（$\frac{p-1}{p}N$）= $2\frac{p-1}{p}N$。

第一条恒等式成立的**前提是 reduce 算子满足结合律与交换律**（sum / max / min / prod 都满足），因而"对 $N$ 个位置全局求和"可以按位置维度切开、各自独立完成——Reduce-Scatter 之后全局结果**已经完整存在**，只是分散在 $p$ 个 rank 上，All-Gather 只是把它复制齐。

更进一步，$2\frac{p-1}{p}N$ 不只是"Ring 恰好花这么多"，而是**任何 all-reduce 算法的带宽下界**（归约段可压缩、复制段不可压缩，两段都无法绕过），Ring 正好打平——证明与代价模型见 [§10 通信下界](#为什么-2fracp-1pn-是最优的all-reduce-的通信下界)。

也可以反过来看这张表：**右边用到的都是编号更小的原语**，这正是本文章节顺序的由来——先把 §5~§9 这些"积木"讲完，§10、§11 才是它们的组合与推广。

### 各并行方式主导的通信原语


| 并行方式                              | 主导通信原语                                   | 频率            | 拓扑要求               |
| ------------------------------------- | ---------------------------------------------- | --------------- | ---------------------- |
| **TP**（`TP.md`）                     | All-Reduce                                     | 每 block 2 次   | 单机 NVLink 高带宽     |
| **DP**（`DP.md`）                     | 请求分发 + 少量 All-Reduce                     | 低              | 松散                   |
| **SP**（`SP.md`）                     | Reduce-Scatter + All-Gather（替代 All-Reduce） | 每 block 2 组   | 同 TP（复用 TP group） |
| **DP-attention**（`DP_attention.md`） | All-Gather / Reduce-Scatter                    | 层内切换维度时  | 单机                   |
| **EP**（`EP.md`）                     | All-to-All（dispatch+combine）                 | 每 MoE 层 2 次  | 高对分带宽（IB/RoCE）  |
| **PP**（`PP.md`）                     | P2P（send/recv）                               | 每 stage 边界   | 稀疏，可跨机           |
| **CP**（`CP.md`）                     | All-Gather（KV）                               | 每 attention 层 | 单机（跨机有精度问题） |

**通信成本直觉**（从轻到重）：P2P（点对点）< Broadcast/Gather < All-Gather/Reduce-Scatter < All-Reduce < All-to-All（全连接，最重）。这也是为什么 TP/EP 尽量放单机、PP 才用于跨机。

### 通信量速查

把前面各节的推导汇总到一处。**为了可比，统一用 $M$ 表示"每 rank 持有/负责的那一份"**（对 gather 类即分片大小，对 reduce 类即张量大小），$B$ 为单链路带宽：


| 原语                  | 单节点最大收发           | 总字节    | 瓶颈位置                                | 大致耗时                          |
| --------------------- | ------------------------ | --------- | --------------------------------------- | --------------------------------- |
| P2P（§3）            | $M$                      | $M$       | 单条链路                                | $M/B$                             |
| Broadcast（§4）      | root 发$(p-1)M$          | $(p-1)M$  | root 出口                               | tree 可降至$O(\frac{M}{B}\log p)$ |
| Scatter（§5）        | **root 发 $(p-1)M$**     | $(p-1)M$  | root 出口（每片内容不同，无法转发复用） | $\frac{(p-1)M}{B}$                |
| Gather（§6）         | **root 收 $(p-1)M$**     | $(p-1)M$  | root 入口                               | $\frac{(p-1)M}{B}$                |
| Reduce（§7）         | 树形后约$2M$             | $(p-1)M$  | 树形可分摊                              | $O(\frac{M}{B}\log p)$            |
| All-Gather（§8）     | 每 rank$(p-1)M$          | $p(p-1)M$ | 均摊到各链路                            | $(p-1)\frac{M}{B}$                |
| Reduce-Scatter（§9） | 每 rank$\frac{p-1}{p}M$  | $(p-1)M$  | 均摊到各链路                            | $\frac{p-1}{p}\cdot\frac{M}{B}$   |
| All-Reduce（§10）    | 每 rank$2\frac{p-1}{p}M$ | $2(p-1)M$ | 均摊到各链路                            | $2\frac{p-1}{p}\cdot\frac{M}{B}$  |
| All-to-All（§11）    | 每 rank$\frac{p-1}{p}M$  | $(p-1)M$  | **对分带宽 $\frac{p}{4}M$**             | 受对分带宽限制                    |

读这张表要**把「总字节」和「耗时」分开看**：

- **总字节最少的未必最快**。Gather 只搬 $(p-1)M$、All-Gather 搬 $p(p-1)M$（$p$ 倍），但两者耗时都是 $O(p\frac{M}{B})$ 量级——因为 gather 全挤在 root 一个口串行，all-gather 却是 $p$ 条链路并行。**All-Gather 对每个节点的收发都 $\ge$ Gather，只是"多出来的部分几乎不额外花时间"**。
- **随 $p$ 线性恶化的是 Gather/Scatter（卡在 root 单口）和 All-to-All（卡在对分带宽）**；Broadcast/Reduce 靠树形降到对数，All-Reduce/Reduce-Scatter 靠 ring 把每卡通信量压到趋于常数。

这直接解释了部署选型：**TP（all-reduce）能扩到 8 卡，EP（all-to-all）一大就必须堆 IB/RoCE 带宽**。

---

## 13. SGLang 中的实现

### 13.1 通信原语入口：`python/sglang/srt/distributed/parallel_state.py`

`GroupCoordinator` 封装了一个通信组的所有原语：

（行序与正文章节一致）


| 方法                                                           | 原语                  |
| -------------------------------------------------------------- | --------------------- |
| `send` / `recv` / `send_tensor_dict` / `recv_tensor_dict`      | P2P（§3）            |
| `broadcast` / `broadcast_tensor_dict` / `broadcast_object`     | Broadcast（§4）      |
| `gather`                                                       | Gather（§6）         |
| `all_gather` / `all_gather_into_tensor` / `all_gatherv`        | All-Gather（§8）     |
| `reduce_scatter` / `reduce_scatter_tensor` / `reduce_scatterv` | Reduce-Scatter（§9） |
| `all_reduce`                                                   | All-Reduce（§10）    |
| `all_to_all_single`                                            | All-to-All（§11）    |

各并行 group（`_TP` / `_PP` / `_DP` / `_ATTN_CP` / EP group 等）都是 `GroupCoordinator` 实例。

### 13.2 设备通信后端：`python/sglang/srt/distributed/device_communicators/`

- `pynccl.py` / `pynccl_wrapper.py`：NCCL（NVIDIA 集合通信库），CUDA 默认后端；
- `custom_all_reduce.py` / `custom_all_reduce_v2.py` / `quick_all_reduce.py`：自定义低延迟 all-reduce（小消息优于 NCCL）；
- `pymscclpp.py`：MSCCL++ 后端；
- `shm_broadcast.py`：进程间共享内存广播；
- `npu_communicator.py` / `xpu_communicator.py` / `hpu_communicator.py`：昇腾 / XPU / HPU 平台后端；
- `torch_symm_mem.py`：对称内存（symmetric memory）支持。

### 13.3 EP 专用 all-to-all：`python/sglang/srt/layers/moe/token_dispatcher/`

EP 的 dispatch/combine 不走通用 `all_to_all_single`，而是用专门优化的 DeepEP / Mooncake / NIXL / Mori / FlashInfer 后端（低延迟、支持 FP8 dispatch），见 `EP.md` §8。

### 13.4 目录说明

`python/sglang/srt/distributed/README_zh.md`、`device_communicators/README_zh.md`。

---

## 参考与延伸

- 同目录：`TP.md`（All-Reduce/All-Gather）、`SP.md`（Reduce-Scatter + All-Gather）、`DP.md`、`PP.md`（P2P）、`EP.md`（All-to-All）、`CP.md`（All-Gather）、`DP_attention.md`。
- SGLang 代码：`python/sglang/srt/distributed/parallel_state.py`、`python/sglang/srt/distributed/device_communicators/`。
- 相关背景：NCCL 集合通信语义、Ring All-Reduce 算法（Baidu ring-allreduce）、DeepEP（EP all-to-all 通信库）。
- 通信下界与算法（§10）：Chan et al., *Collective communication: theory, practice, and experience*（CCPE 2007）给出 $\alpha$-$\beta$ 模型下各原语的下界与 recursive halving-doubling；Thakur et al., *Optimization of Collective Communication Operations in MPICH*（IJHPCA 2005）是 MPI 按消息大小切换算法的经典依据。
- **代价模型专题**：同目录 `cost_model.md`——系统讲解 $\alpha$-$\beta$（含半带宽点 $n_{1/2}$、算法交叉点推导）、LogP/LogGP（通信-计算 overlap 的理论依据）、$\alpha$-$\beta$-$\gamma$、带宽层次与分层集合通信、对分带宽、Roofline、BSP，以及它们在 SGLang 中的对应决策。
