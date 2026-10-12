# Preble:面向 Prompt 共享的分布式 LLM 调度系统

> 原文:*Preble: Efficient Distributed Prompt Scheduling for LLM Serving*,Vikranth Srivatsa\*、Zijian He\*(共同一作)、Reyna Abhyankar、Dongming Li、Yiying Zhang,UC San Diego,arXiv:2407.00023v2(2024-10),ICLR 2025
> 代码:https://github.com/WukLab/preble

---

## 1. 一句话概括

单机 prefix caching(SGLang / vLLM APC)只在**单卡内**复用 KV,而生产集群是分布式的;把请求 round-robin 打散,共享前缀就被分散到多张卡上重复 prefill。**Preble 是第一个把「KV 复用」和「计算负载均衡」放在同一个目标函数里协同优化的分布式 LLM 服务系统**,核心是 $E^2$(Exploitation + Exploration)调度算法:

$$
\text{当 } \underbrace{\text{cached\_len}}_{\text{省下的重算}} > \underbrace{\text{missed\_len}}_{\text{新增的计算}} \;\Longrightarrow\; \textbf{Exploit};\quad \text{否则} \;\Longrightarrow\; \textbf{Explore}
$$

在 5 个真实工作负载、2 个模型(Mistral-7B / Llama-3-70B)、2 套集群(4×A6000 / 8×H100)上,相比 round-robin + SGLang,**平均延迟改善 1.5×–14.5×,p99 延迟改善 2×–10×**。

---

## 2. 问题:分布式下的「亲和 vs 均衡」两难

论文把矛盾讲得很直白:

| 做法 | 后果 |
|---|---|
| **Cache-blind 均衡**(round-robin / 最少连接) | 共享同一前缀的请求被分到不同 GPU,**每张卡都要重算一遍这个前缀** |
| **朴素亲和**(同前缀永远同卡) | 最先服务热门前缀的那张卡会**滚雪球式堆积负载**,整体 GPU 利用率反而下降 |

单机方案(SGLang、vLLM APC、Prompt Cache)全部落在第二类的反面:它们让**单卡内**的复用变高效,却对「请求该去哪张卡」完全无话可说。而现有分布式层(DP/TP、LB)全都是 **prompt-unaware** 的。

> Preble 的定位因此非常清楚:**不改 attention kernel、不改引擎缓存,只做「请求落到哪张卡」这一层决策**。它是一个可以叠加在 vLLM/SGLang 之上的独立层。

---

## 3. 工作负载研究(全文的立论基础)

作者研究了 5 类真实负载 + 1 个真实生产 trace(Azure LLM Inference Trace)。这一节的数据是后面所有设计的依据。

### 3.1 五个工作负载的画像

| 工作负载 | 数据集 | Prompt 长度 (均值, 标准差) | Output 长度 | 共享前缀占比 | Key Portion 占比 | 共享该 Key Portion 的请求数 |
|---|---|---|---|---|---|---|
| Tool Use | ToolBench(210k query / 16k tool) | (1835, 742) | (43, 16) | (85%, 13%) | (76%, 16%) | (39, 64) |
| Embodied Agent | ALFWorld(7.5k) | (2285, 471) | (16, 13) | (97%, 14%) | (76%, 12%) | (48, 8) |
| Programming | APPS | (3871, 1656) | (190, 343) | (97%, 7.4%) | (78%, 13%) | (126, 2157) |
| Video QA | NExT-QA(8.5k Q / 1k 片段) | (9865, 5976) | (4, 1.5) | (88%, 32%) | (99%, 0.2%) | (8.6, 2) |
| LooGLE(长文档 QA) | LooGLE(776 doc / 6.4k Q) | (23474, 6105) | (16, 9.9) | (91%, 24%) | (94%, 15%) | (18, 8.6) |

三个关键数量级:

- **prompt / output 比:37× – 2494×**(Video QA 最极端)→ **prefill 完全主导**
- **共享前缀占比:85% – 97%**
- **一条公共序列平均被 8.6 – 126 个请求共享**

### 3.2 Key Portion:本文一个被低估的概念

> **Key Portion 定义**:请求在前缀树路径上,**token 数大于其所有祖先之和**的那个最深节点。

直觉上,它是这条 prompt 中「体量最大、且有独立共享度」的那一段——比如 tool use 里具体某个 tool 的说明书,LooGLE 里的那篇长文档本身,而不是所有请求都有的 system prompt。数据显示 key portion 占 prompt 的 **76%–99%**。

**举例说明**(Tool Use 场景):假设前缀树上已有这些请求(括号内为该节点 token 数):

```
[system prompt] 200
    └── [工具列表说明] 300
            ├── [Tool A 文档] 1200
            │       └── [问题1] 50
            │       └── [问题2] 80
            └── [Tool B 文档] 800
                    └── [问题3] 60
```

以请求「问题2」为例,它的完整 prompt 是 `system prompt(200) + 工具列表(300) + Tool A 文档(1200) + 问题2(80)`。沿路径从根往下逐节点判断「本节点 token 数 > 祖先 token 之和」:

| 节点 | 本节点 tokens | 祖先之和 | 判断 |
|---|---:|---:|---|
| system prompt | 200 | 0 | \(200>0\) ✓ |
| 工具列表 | 300 | 200 | \(300>200\) ✓ |
| Tool A 文档 | 1200 | 500 | \(1200>500\) ✓ |
| 问题2(叶子) | 80 | 1700 | \(80<1700\) ✗ |

Key Portion = **最深的满足条件的节点,即 [Tool A 文档](1200 tokens)**——不是每条请求都有的 system prompt,也不是只占 80 tokens 的具体问题。

**Implication 3(作者原话的意思)**:识别出 key portion 并围绕它做放置,就能**大幅降低调度复杂度**——不用管整棵树,盯住那一段就够了。这其实是后来一系列 prefix-aware 路由工作(DualMap、CacheRoute 的 "business key")的思想雏形。

### 3.3 Azure 真实 trace

| 指标 | Conversation | Programming |
|---|---|---|
| 到达速率 | ≈ 5 req/s | ≈ 7 req/s |
| 平均到达间隔 | 118 ms | 63 ms |
| 平均 prompt/decode 比 | 4 | 92 |
| **最长 20% prompt 的 prompt/decode 比** | **175** | — |
| 到达间隔跨度 | **2 μs – 217 s** | — |

两个要点:一是 chat 的**平均比值只有 4**,但**长尾那 20% 达到 175**——说明「长 prompt」是个重尾现象;二是到达间隔跨 8 个数量级,**负载强度剧烈波动**,这是 §3.5 中「后置负载调整」必须存在的理由。

### 3.4 四条 Insight → 四条 Implication

1. Prompt 远长于 output → **优化 prefill 是主战场**,且 P/D 计算失衡必须被显式处理
2. Prompt 共享普遍且比例高 → **跨请求复用应被分布式系统一等公民对待**
3. 多数请求存在一个 key portion → **按 key portion 放置可以降复杂度**
4. 真实负载强度多变、不同用途特征不同 → **必须同时考虑负载与共享的变化**

---

## 4. 系统架构:两级调度

```
                 incoming request
                        │
                   [Tokenizer]  ← 并行分词层
                        │
        ┌───────────────▼────────────────┐
        │  Global Scheduler (请求级)      │
        │  · 全局 radix 树(多棵)         │
        │  · E² 调度算法                  │
        │  · 后置 rebalance / autoscale   │
        │  · Prefill-Decode 混合          │
        └──┬─────────┬─────────┬─────────┘
           │         │         │        ▲ 异步上报 eviction
      ┌────▼───┐ ┌───▼────┐ ┌──▼─────┐  │
      │ Local  │ │ Local  │ │ Local  │──┘
      │ Sched  │ │ Sched  │ │ Sched  │   (迭代级)
      │ GPU1   │ │ GPU2   │ │GPU3+4  │   · 本地 radix 树
      └────────┘ └────────┘ └────────┘   · 优先级等待队列
                              (TP=2)      · LRU 驱逐 + chunked prefill
```

**分层的三个理由(作者自述)**:

1. 所有请求过同一个 global scheduler → 有一个**集中的全局视图**(集群负载 + 缓存分布),这是 $E^2$ 的前提
2. 全局只做**粗粒度的请求级**决策 → 单个 scheduler 就能撑几百张卡,避免多个分布式全局调度器的一致性复杂度
3. 每 GPU 做**细粒度的迭代级**决策 → 能快速响应显存/请求可用性变化

> 一个 local scheduler 管一个 **model instance**(用 TP 时是多卡,不用时是单卡)。要扩到数据中心级,就部署多个 Preble 集群,各自一个 global scheduler。

---

## 5. Global Scheduler:$E^2$ 算法

### 5.1 数据结构:全局前缀树(radix trees,复数)

注意是**多棵树**——每棵树有一个不同的 root,存该树下所有 prompt 的公共前缀。插入逻辑是标准 radix 树:

- 从头做前缀匹配,不匹配的剩余 token 作为新叶子插入
- 完全不匹配 → **新建一棵树**,该 prompt 整体作 root
- 部分匹配某节点 → **分裂节点**为「匹配部分 + 剩余部分」

每个树节点记三样东西:

1. 节点内 token 数
2. **缓存了该节点 KV 的 GPU 集合**
3. **历史窗口 $H$ 内,每 GPU 上共享该节点的请求数**

回收条件:节点既无缓存 GPU、$H$ 内又无任何请求共享 → 从树中删除。

> 论文对数据结构的描述停在这个抽象层次。**开源实现里的细节比这段文字丰富得多,也有几处与论文表述不一致**——包括「多棵树」实际是单棵虚拟 root 树、`cached_gpus` 与 `gpu_allocations` 是两套独立映射、exploit/explore 判据在代码里退化成了一个叶节点属性判断。完整剖析见 **§14 附录**。

### 5.2 核心判据:Exploit 还是 Explore

设匹配到的前缀长度 $\ell_{\text{cached}}$、未匹配剩余长度 $\ell_{\text{missed}}$:

$$
\text{Exploit} \iff \ell_{\text{missed}} < \ell_{\text{cached}}
$$

**这个判据的经济学含义**:$\ell_{\text{cached}}$ 是**粘在这张卡上能省下的计算**,$\ell_{\text{missed}}$ 是**无论去哪张卡都要付、因而可以拿去做负载均衡的计算**。哪边大就听哪边的。

- **Exploit**:发给**缓存了匹配路径上最长节点**的那台 GPU。若有多台,用下面的 load cost 选最轻的一台。
- **Explore**:在**全部 GPU** 中按 load cost 选最优(先过一道 P/D 平衡的短路判断,见 §5.4)。

> Explore 的意义不是「随便找台空闲卡」,而是**给共享前缀一个换地方生根的机会**——这是维持长期集群效率的关键。这一点比论文里写的还重要:纯 exploit 会让树形结构在几台卡上固化。

### 5.3 Prompt-aware 的负载定义(三段式 cost)

这是 $E^2$ 最实用的部分。对 GPU $i$、新请求 $R_k$:

$$
\text{Cost}_i \;=\; \underbrace{L_i}_{\text{历史负载}} \;+\; \underbrace{M_i}_{\text{驱逐代价}} \;+\; \underbrace{P_i}_{\text{新请求代价}}
$$

**(1) 历史窗口负载 $L_i$**($H$ 默认 3 分钟,实测对 $H$ 不敏感)

$$
L_i \;=\; \sum_{r \in W} \bigl( PT_r + DT_r \bigr)
$$

- $PT_r$:用**在 GPU $i$ 上未命中前缀的 token 数**代入回归函数得到的 prefill 时间
- $DT_r$:用 $H$ 内该 GPU 观测到的**平均输出长度**代入另一个回归函数得到的 decode 时间

**$PT_r$ 与 $DT_r$ 的详细形式**:

Prefill 时间 $PT_r$。设请求 $r$ 在 GPU $i$ 上的未命中长度为 $\ell_{\text{missed}}^{(r)}$,窗口 $H$ 内 GPU $i$ 的其他负载相关量为 batch 内总 token 数 $n_{\text{batch}}$、累计 context $L_{\text{ctx}}$、去重后的 KV token 数 $n_{\text{unique\_kv}}$(即被共享前缀去重后真正要算 attention 的量),则:

$$
PT_r \;=\; \underbrace{f_{\text{lin}}(n_{\text{batch}})}_{\text{linear/MLP 项}} \;+\; \underbrace{f_{\text{attn}}(n_{\text{req}},\, L_{\text{ctx}},\, n_{\text{unique\_kv}})}_{\text{attention 项}}
$$

其中论文正文给出一阶近似:

$$
f_{\text{lin}}(n_{\text{batch}}) \approx a_p \cdot n_{\text{batch}} + b_p
$$

而开源实现里 $f_{\text{lin}}$ 实际是**分段函数**(< 192 token 常数下界、192–384 二次过渡、≥ 384 线性),$f_{\text{attn}}$ 在 $L_{\text{ctx}} > 1024$ 时启用线性项,并在 $n_{\text{unique\_kv}}$ 较小(前缀共享度高)时打对折——即**共享前缀的 attention 折扣被显式建模**;超长 prompt 还追加了 $\text{extend\_len} \times \text{seq\_len}$ 的二次交叉项。完整形态见 §14.12。

关键点:$PT_r$ 的自变量是**未命中长度**而不是 prompt 总长——这就是「prompt-aware 负载」的含义:同样一条 4000 token 的请求,在命中了 3500 token 缓存的卡上,$PT_r$ 只按 500 token 计。

Decode 时间 $DT_r$。设窗口 $H$ 内 GPU $i$ 上请求 $r$ 所属负载类的平均输出长度为 $\bar{o}^{(r)}$,则:

$$
DT_r \;=\; DT(\bar{o}^{(r)}) \approx a_d \cdot \bar{o}^{(r)} + b_d
$$

论文正文用「$H$ 内平均输出长度」;开源实现更精细——按 key portion 节点分别统计历史输出长度取**中位数**(对重尾更稳),再乘以该卡**在线实测的 TPOT 中位数**,见 §14.6。

**为什么用历史窗口而不是瞬时负载?** 作者给了两条理由,都很扎实:

1. 从「调度 $R_k$」到「真正执行 $R_k$」之间,GPU 负载会变
2. **前缀的放置是长期决策**——它会吸引后续请求来 exploit,影响远超一次调度

**(2) 驱逐代价 $M_i$**

$$
M_i \;=\; \sum_{j \in E} PT_j \times N_j, \qquad N_j = \frac{\text{$H$ 内共享节点 $j$ 的请求数}}{\text{$H$ 内 GPU $i$ 的总请求数}}
$$

$E$ 是为了跑 $R_k$ 而必须被驱逐的节点集合(调用 §6 的 LRU 驱逐算法预演一遍)。含义:**一个节点被越多请求共享、自身越长,驱逐它就越贵**。

> 注意作者显式说明:$M_i$ **不含 decode 时间**——decode 不受前缀缓存驱逐影响,且已在 $L_i$ 里算过了。这种防重复计数的细节很见功力。
> 另一个前提假设:「GPU 总是满载运行」,所以**驱逐代价必然发生**,不是可选项。

**(3) 新请求代价 $P_i$**

$$
P_i = PT(\ell_{\text{missed}}^{(k)})
$$

只算 miss 部分的 prefill。**不含 decode**,因为 decode 时间在各卡间相同,而目标是**比较**。

### 5.4 关键实现取舍:用 token 数代替真实计时

Preble **不做在线 profiling**,全局调度器只维护 token 计数,再用**离线标定的回归函数**换算成时间。依据是附录 B 的实测:

- **Prefill 时间对 prompt 长度基本线性**(linear layer 主导 forward)
- **Decode 时间对 context 长度也近似线性**

$$
PT(\ell) \approx a_p \ell + b_p, \qquad DT(\ell) \approx a_d \ell + b_d \quad (\text{每种 GPU 类型离线标定})
$$

> 这是 Preble 能做到低开销的根本原因。代价是:**换模型、换卡、换 batching 配置都要重新标定**,而且线性假设在超长 context(attention 项 $O(\ell^2)$ 开始抬头)下会失真。

### 5.5 后置负载调整:rebalance 与 autoscale

基础 $E^2$ 有个固有缺陷:**前缀一旦落地就长在那儿,直到被驱逐**。但共享它的请求量会随时间变化。两级补救:

**第一级 —— Load Shifting**(适用于单卡还扛得住的浪涌)

$$
\frac{L_{\max}}{L_{\min}} > Th_{\text{bal}} \;\Longrightarrow\; \text{把本应 exploit 重卡的后续请求改导向轻卡,直到比值回落}
$$

$Th_{\text{bal}}$ 可配置,从 GPU 型号 + 模型类型 profiling 得出。

**第二级 —— Prefix Autoscaling**(单卡已经扛不住)

触发条件:rebalance 之后**该前缀的平均排队时间在 $H$ 内翻倍**。动作:**复制该前缀到多张 GPU,并按负载切分其子树**。

> 这就是后来 CacheRoute 里 $k_b = \lceil \lambda_b / q_{\text{cap}} \rceil$ 的反应式版本。Preble 是**事后触发**,CacheRoute 是**事前按速率规划**——两种范式的分水岭。

### 5.6 用「共享度」做 Prefill/Decode 平衡(本文最巧的一个 idea)

作者的洞察:

> **全部 prompt 都命中缓存的请求 ≈ 纯 decode 计算单元;prompt 完全未命中 + 输出短的请求 ≈ 纯 prefill 计算单元。部分命中则介于两者之间。**

因为 prompt/decode 比高达 37×–2494×,这个近似相当准确。于是:

- 当某 GPU 上**堆积了大量 decode 类计算单元**(compute 单元空闲、memory bandwidth 吃紧),把当前**待 explore 的请求**(必然要做 prefill,是 prefill 单元)直接导给它
- 这条策略的**优先级高于 load cost 比较**——理由是「重 decode 的 GPU 有几乎白送的算力」
- 只有当所有 GPU 的 P/D 构成都比较均衡时,才退回 load cost 比较

$$
\text{if } \max_i \text{DecodeRatio}(i) > \text{IMBAL}_R \;\Longrightarrow\; \text{直接返回 } \arg\max_i \text{DecodeRatio}(i)
$$

> **和 chunked prefill / P-D 分离的关系**:那些方案在**单机内部**或**跨机器角色划分**上解决 P/D 失衡;Preble 是在**集群路由层**用「请求的共享度」这个已有信号顺手做了一次 P/D 混合。零额外成本,而且**正交**——Preble 的 local scheduler 里仍然跑 chunked prefill。

### 5.7 算法伪码

```text
Algorithm 1  E² Global Scheduling
function ScheduleRequest(R_k):
    在全局 radix 树上匹配 R_k
    cached_len ← 匹配路径总长
    missed_len ← prompt_len − cached_len

    if missed_len < cached_len:                 ▷ Exploit
        K ← 缓存了匹配路径中最长节点的 GPU 集合
        return argmin_{i∈K} LoadCost(i, R_k)
    else:                                       ▷ Explore
        if max_i DecodeRatio(i) > IMBAL_R:      ▷ P/D 平衡短路
            return argmax_i DecodeRatio(i)
        return argmin_{i∈AllGPUs} LoadCost(i, R_k)
```

```text
Algorithm 2  LoadCost(i, R_k)
    L ← 0; M ← 0; P ← 0
    for R_j in 历史窗口 H:                       ▷ 历史负载
        L += PrefillTime(R_j 在 i 上的 miss 长度)
        L += DecodeTime(H 内平均输出长度)
    E ← 为跑 R_k 需在 i 上驱逐的树节点集合         ▷ 驱逐代价
    for j in E:
        N_j ← (H 内共享 j 的请求数) / (H 内 i 的总请求数)
        M += PrefillTime(len(j)) × N_j
    P ← PrefillTime(R_k 在 i 上的 miss 长度)      ▷ 新请求代价
    return L + M + P
```

```text
Algorithm 3  E² Local Scheduling
function ScheduleRequests():
    for 等待队列中每个请求:
        计算 prefix 命中率 → 分入对应优先级组
    按 round-robin + 配额 处理各优先级组
```

---

## 6. Local Scheduler:迭代级调度与公平性

每个 local scheduler 维护:等待队列 + 本地 radix 树 + 每节点的活跃共享请求数。

**每轮迭代的流程**:

1. 新请求到达 → 匹配本地前缀树、更新树、入等待队列
2. 每次模型迭代后,用**优先级算法**从等待队列组 batch
3. **长且不共享的 prompt** → 按 Sarathi 的方式做 **chunked prefill**
4. 显存不够 → 按 **LRU** 选择树节点(或节点的一部分)驱逐
5. **异步**通知 global scheduler 该驱逐事件(后台处理,不阻塞请求路径)

### 公平性:基于命中率的优先级配额

现有做法的两个极端:

| 策略 | 代表 | 问题 |
|---|---|---|
| FCFS | vLLM | 忽略 prompt 共享,重算多 |
| 最长共享优先 | SGLang | **忽略公平性,低命中率请求可能饿死** |

Preble 的折中:

$$
\text{priority}(R) = \left\lfloor P \cdot \frac{\ell_{\text{cached}}}{\ell_{\text{prompt}}} \right\rfloor,\qquad P \text{ 为可配置的组数}
$$

例:100 token 里命中 63 个、$P=10$ → 优先级 6。组 batch 时**按比例从各组抽取**(如要选 55 个请求:优先级 10 组选 10 个、9 组选 9 个……)。

> **高命中率的请求被优先,但低命中率的请求有保底配额**——既保住了复用收益,又消除了饥饿。消融实验确认:这一项**只改善 p99,不改善均值**,完全符合「公平性机制」的预期。

---

## 7. 实现

- **独立层**,可挂在任意后端上;当前支持 **vLLM** 和 **SGLang**(仅需极小改动)
- **并行分词层**在调度前完成 tokenize
- **异步请求 handler** 处理调度
- **全局 radix 树访问基本无锁**:绝大多数操作是只读的;唯二的写操作是「给节点加一个 GPU」和「命中计数 +1」,两者都可表达为**原子指令**
- 每 GPU 的当前负载计数在「分配新请求」和「节点驱逐」时增量更新
- **rebalance、eviction 记账等非调度任务全部丢到后台线程**,保证前台请求路径的性能

---

## 8. 实验结果

### 8.1 配置

| 项 | 内容 |
|---|---|
| 模型 | Mistral-7B、Llama-3-70B |
| 集群 | 2×A6000 / 4×A6000(双机各 2 卡);8×H100 单机(70B 用 TP=4 + DP=2) |
| 到达过程 | Poisson(除 Azure trace 实验外) |
| Baseline | SGLang / vLLM + **round-robin LB**(即「先均衡、再单卡内共享」) |
| 指标 | RPS 容量、**平均端到端延迟**、**p99 延迟**(含调度+排队+prefill+decode) |

#### Baseline 具体说明

Preble 的主实验不是同时对比多种路由算法，而是在相同推理后端上，将 **Preble 的 prefix-aware 调度**与 **round-robin 负载均衡**对比：

| 对比项 | 路由方式 | Prefix KV Cache |
|---|---|---|
| SGLang baseline | Round-robin | 支持单 GPU 内复用，但路由器不感知 Prefix |
| vLLM baseline（附录） | Round-robin | 支持单 GPU 内复用，但路由器不感知 Prefix |
| Preble | $E^2$ prefix-aware 调度 | 跨 GPU 跟踪 Prefix，并权衡缓存复用与计算负载 |

因此，论文报告的 **1.5×–14.5× 平均延迟改善和 2×–10× p99 改善，主要是相对 SGLang + round-robin**，不能解释为相对其他 cache-aware 路由方案的提升。论文**没有把 Random、Least Requests、Consistent Hashing 作为主要 baseline 进行系统对比**；这些只能作为理解「负载优先」与「亲和优先」两类策略的概念参照，不能写成论文实际 baseline。

> **指标选择的说明**:作者刻意**不用 TTFT / TPOT**。理由是这些负载输出极短,TPOT 意义不大,而 TTFT 几乎等于请求总延迟。这个取舍是自洽的,但也意味着**结果不能直接和以 TTFT/TPOT 为主指标的工作横比**。

### 8.2 端到端结果

**总体:平均延迟 1.5×–14.5×,p99 延迟 2×–10×。**

分负载看:

- **改善最大**:ToolBench、Embodied Agent、Video QA、LooGLE(prefill 主导)
- **改善最小**:Programming —— **decode 长度最长**(190 token),decode 时间开始主导总延迟,而 **Preble 完全不优化 decode**。即便如此仍有 **1.56×–1.8× 平均 / 3×–4× p99**

分规模看:

- 2 → 4 张 A6000:**相对改善保持不变**;绝对值上,**RPS 翻倍而延迟基本持平**(扩展性good)
- A6000 + Mistral-7B → H100 + Llama-3-70B:**相对改善反而变大**

> 这个「模型越大、卡越强,收益越大」的趋势值得注意:大模型的 prefill 绝对成本更高,省掉一次重算的价值也就更高。

### 8.3 Azure trace 混合负载(4×A6000)

用 Azure 的真实到达模式(而非 Poisson)混跑 tool use + video QA:

| 指标 | 相对 SGLang 的改善 |
|---|---|
| 平均延迟 | **6.02×** |
| p99 延迟 | **4.61×** |
| 平均 TPOT | **5.60×** |
| 平均 TTFT | **2.42×** |

有意思的是 **TPOT 改善(5.6×)远大于 TTFT 改善(2.42×)**。Preble 不优化 decode 本身,TPOT 的改善来自**排队与 batch 拥塞的缓解**——即 decode 迭代不再被大量重复 prefill 挤占。

### 8.4 消融实验(ToolBench + Zipf-1.1 前缀流行度,4×A6000)

逐项叠加:

| 配置 | 平均延迟 | p99 延迟 |
|---|---|---|
| SGLang(round-robin baseline) | 基准(最差) | 基准(最差) |
| + $E^2$(无后置调整) | ↓ 明显 | ↓ 明显 |
| + rebalance & autoscale | ↓ | **↓↓(尤其显著)** |
| + Prefill/Decode 感知 | ↓ | ↓ |
| + 优先级等待队列(= ALL) | ≈ 不变 | ↓ |

**读法**:

- $E^2$ 本身贡献最大,来自「动态负载切分」
- **后置调整主要救 p99**——符合直觉,它治的就是热点尾巴
- P/D 感知**两个指标都改善**,因为它提升的是 GPU 资源利用效率本身
- 优先级队列**只改善 p99 不改善均值**——作者自己预测了这一点,实验证实,是很干净的因果验证

> 作者也诚实标注:**这组消融用的是 Zipf-1.1 的倾斜分布**,「其他负载和分布会从不同的技术组合中受益」。换句话说,**各组件的贡献占比不可外推**。

### 8.5 全局调度器的吞吐与扩展性

一次性灌入 50,000 个请求打满调度器:

| 工作负载 | 前缀树复杂度 | 调度吞吐 |
|---|---|---|
| ToolBench | **最复杂** | **245 req/s** |
| Video QA | **最简单** | **2931 req/s** |

按 Mistral-7B 在 A100 上 30–150 tok/s 的 decode 速度和各负载的输出长度换算:

$$
\textbf{单个 global scheduler 可支撑 70 – 391 张并发 A100}
$$

网络处理经测量**不是瓶颈**。若计入 prefill 时间或换更大的模型,可支撑的 GPU 数还会更多。

> **这是全文最重要的工程约束**:245 req/s 意味着复杂前缀树场景下,单调度器的天花板并不高。全局 radix 树搜索是主要开销,而**这是个中心化的、请求路径上的同步操作**。要扩到数据中心级只能靠「部署多个独立 Preble 集群」——**跨集群之间就没有缓存感知了**。

### 8.6 后端与版本鲁棒性(附录 C)

- **vLLM 后端**(Video QA / Mistral-7B-v0.2 / 2 卡):Preble 仍胜出,但**相对改善小于 SGLang 后端**。原因作者列了四条:① vLLM 的 prefix caching 尚是 beta 且不够快;② 未用 FlashInfer kernel;③ **不支持 chunked prefill 与 prefix caching 同时开启**;④ 调度延迟显著
- **SGLang v0.3 + FlashInfer 0.1.6**(相对论文主体用的 v0.1.12):即便 baseline 大幅优化了 32K context 共享性能并降低了调度开销,**Preble 的改善依然存在**

> 第二项很关键——它说明 Preble 的收益**不是靠 baseline 版本老旧刷出来的**。这类验证在系统论文里做得远远不够多。

---

## 9. 与相关工作的定位

| 方向 | 代表工作 | 与 Preble 的关系 |
|---|---|---|
| 单机前缀复用 | SGLang、vLLM APC | **Preble 的后端**;它们解决「单卡内怎么复用」,Preble 解决「请求去哪张卡」 |
| 非前缀的模块化复用 | Prompt Cache | 允许任意子序列复用(错位 positional encoding + 不完整 attention)→ **有损**,可能降低生成质量;且同样是单机 |
| 共享前缀的 attention kernel | Hydragen | **完全正交**,Preble 可用任意 attention kernel |
| Chunked prefill | Sarathi、Sarathi-Serve | 解 P/D 失衡但**不考虑共享**;Preble 在其之上叠加基于共享度的 P/D 混合 |
| P-D 分离 | Splitwise、DistServe、DéjàVu、Mooncake、MemServe | 同上,不考虑 prompt 共享 |
| 公平性 | FastServe | 指出了最长共享优先会饿死请求的问题;Preble 用优先级配额兼顾两者 |

作者的核心主张:**今天没有任何并行/分布式服务系统是 prompt-aware 的**;「先均衡打散、再在单卡内做前缀共享」**从根本上放弃了集群级的共享机会**。

---

## 10. 局限与开放问题

论文自身承认的 + 我认为需要留意的:

**1. 中心化全局调度器是硬上限**
245 req/s(ToolBench)已经是这一层的天花板。全局 radix 树位于同步请求路径上,虽然读多写少可无锁,但**树的规模随 active 前缀集合增长**。跨集群完全无缓存感知。

**2. 线性时间模型的适用边界**
$PT(\ell)$、$DT(\ell)$ 的线性回归在 Mistral-7B/A6000 上成立(linear layer 主导),但换到**超长 context**(attention 的 $O(\ell^2)$ 项抬头)、**MoE 模型**、**激进的 continuous batching** 下会失真。每种 GPU 类型都需要重新离线标定。

**3. Decode 完全没被优化**
Programming 负载的改善明显偏低已经暴露了这一点。当下的 reasoning 模型、长 CoT、agent 长输出场景中,**decode 占比正在上升**,Preble 的收益结构会被稀释。

**4. 输出长度用「$H$ 内平均值」估计**
作者的辩护是「同类负载内输出长度小且相近」——这在 2024 年的这五个负载上成立,但对**输出长度重尾**的现代 reasoning 负载,这个假设相当脆弱,会直接污染 $L_i$ 中的 $DT_r$ 项。

**5. Exploit/Explore 判据是纯 token 计数,不含负载**
$\ell_{\text{missed}} < \ell_{\text{cached}}$ 这个判据里**完全没有集群负载信息**。负载只在选定分支**之后**才通过 load cost 介入。这意味着:一个 90% 命中的请求在极端拥塞下**依然会 exploit**,只是在「缓存了它的那几台卡」里挑最轻的一台。若那几台全都爆了,也没有溢出路径。

**6. 消融结论不可外推**
仅在 ToolBench + Zipf-1.1 上做过,作者主动声明其他分布下各组件的贡献不同。

**7. Baseline 是 round-robin**
这是当时(2024)的合理设定,但它**不包含任何 cache-aware 的 baseline**(一致性哈希、CHWBL 等)。所以 1.5×–14.5× 是相对「完全 cache-blind」的改善,不是相对「其他 cache-aware 路由」的改善。

**8. 没有 SLO 容量口径**
用「延迟-RPS 曲线」而非「给定 p99 SLO 下的容量」。前者更完整,但更难做跨论文对比。

---

## 11. 后续影响:Preble 成了新的 baseline

值得单独一提:Preble 定义的这套问题(**分布式 prefix-aware 路由**)在其后被反复攻击,而 Preble 本身变成了**被对比的对象**。以本仓库里的 CacheRoute(Meta,2026)为例:

| 维度 | Preble(2024) | CacheRoute(2026) |
|---|---|---|
| 决策时机 | **每请求反应式** | **周期性离线规划成一张路由表** |
| 负载信号 | 历史窗口 $H$ 内的实时 cost | 离线估计的 per-key 速率 $\lambda_b$ |
| 缓存信号 | 全局 radix 树(精确前缀匹配) | 业务 key(粗粒度) |
| 副本 | **事后触发** autoscale | **事前** $k_b = \lceil \lambda_b/q_{\text{cap}} \rceil$ |
| 放置 | 贪心选 min-cost GPU | **LPT list scheduling** |
| 请求路径开销 | 全局树搜索(245–2931 req/s) | 查表 + 小集合比负载 |

在 CacheRoute 的 60×H100 / Llama-3.3-70B 实验中,「Preble 风格」策略(**注:是同一 harness 下的重实现,非原系统复现**)得到 72.0% KV 命中 / 76 QPS 容量,被 CacheRoute 的 93.2% / 176 QPS 超越。CacheRoute 给出的解释与 Preble 的第 5 条局限吻合:**反应式决策在高负载下会不断把请求推离已缓存的机器,牺牲命中率去换队列均衡**。

> 但反过来说:**Preble 不需要任何离线速率估计,能应对完全未知的、突发的前缀结构**,而 CacheRoute 需要稳定的路由 key + 有效的速率估计 + shadow replay 门禁。**两者是不同工作点,不是简单的迭代替代关系。**

---

## 12. 工程借鉴清单

即便不直接用 Preble,以下几点是可以单独摘出来的:

1. **判据 $\ell_{\text{missed}} < \ell_{\text{cached}}$ 极其廉价且有明确的经济学意义**——任何 prefix-aware 路由器都可以拿它做第一层分流
2. **用历史窗口负载而非瞬时负载**做路由决策,理由(放置的长期效应)是通用的
3. **驱逐代价要计入路由 cost**,且权重应为「被共享的频度 × 长度」
4. **用 token 计数 + 离线回归代替在线 profiling**,是把成本压到请求路径之外的关键手法
5. **「共享度 ≈ P/D 属性」这个映射几乎零成本**,值得在任何做 P/D 混批的系统里试一下
6. **命中率分桶 + 按组配额**是一个比「最长共享优先」健壮得多的等待队列策略,专治尾延迟
7. 上线前先量自己负载的两个数:**prompt/output 比** 和 **共享前缀占比**。前者决定 prefill 是否真的是瓶颈,后者决定有没有东西可复用——**这两个数不够大,整套方案不值得做**

---

## 13. 个人评述

**做得好的:**

- **问题定义干净且真正是新的**:「单机 prefix caching + cache-blind 分布式层 = 集群级复用机会被系统性浪费」,这个 gap 在 2024 年被指出来是很有价值的,后续一整条线的工作都建立在这个框架上
- **工作负载研究扎实**,不是为了凑 motivation 而做:5 个负载 + 1 个真实 trace,Table 1 给了均值和标准差,附录逐个解释每个负载的 prompt 结构和它代表的更广泛的场景类别
- **Key Portion 是个被低估的抽象**——它把「一棵复杂前缀树」压缩成「一个有代表性的段」,这个降维思路在后续工作里以各种形式反复出现
- **成本模型的细节很讲究**:$M_i$ 不含 decode(避免重复计数)、$P_i$ 不含 decode(因为要做比较)、用回归而非在线 profiling(压请求路径开销)——这些都是实现过的系统才写得出来的东西
- **P/D 平衡的做法很聪明**:用一个已经在手里的信号(共享度)顺手解决另一个问题,零额外成本,且与 chunked prefill 正交
- **附录 C 的版本鲁棒性验证**难能可贵——主动去和优化后的新版 baseline 对比,而不是躺在旧版本上

**需要留意的:**

- **Exploit/Explore 判据不含负载信息**,是整个设计里最理论上可疑的一环。在饱和状态下,exploit 分支没有溢出机制。CacheRoute 的实验(72% 命中率、明显低于 sticky 的 87%)从侧面印证了这个张力的存在
- **中心化调度器 245 req/s** 的数字应该被更醒目地放在正文——它比任何延迟数字都更能决定这套方案的适用范围
- **Decode 是盲区**,而 2024 年之后 decode 的占比一路在涨。今天复现这篇工作,Programming 负载上那个「最小改善」很可能会变成大多数负载的常态
- **平均输出长度估计**是个隐藏的脆弱点,在输出长度重尾的场景下会污染负载估计
- **缺 cache-aware baseline**:1.5×–14.5× 是相对 round-robin 的,把它理解为「Preble 比其他 prefix-aware 路由强多少」是误读

**一句话定位**:Preble 是**第一个把问题讲清楚并给出完整系统实现的工作**,它的算法在今天未必是最优的,但它定义的坐标系(exploit/exploration、prompt-aware load、key portion、集群级复用)至今仍是这个方向的通用语言。**如果只读一篇分布式 prefix caching 的论文,应该读这篇;如果要上生产,则需要参考后续针对其反应式决策与中心化瓶颈的改进。**

---

## 14. 附录:Radix Tree 的源码级实现剖析

> 基于 `github.com/WukLab/preble` 主分支。核心文件:
> - `preble/global_lru_cache.py` —— **全局** radix 树 `LPRadixCache`
> - `preble/global_scheduler_with_time.py` —— $E^2$ 调度器 `GlobalSchedulerWithTime`(论文正文对应的实现;另有一个 `_perf.py` 变体用于纯调度吞吐测试,cost 记账方式略有不同)
> - `python/sglang/srt/managers/router/radix_cache.py` —— **本地** radix 树(fork 自 SGLang 并做了改动)
>
> ⚠️ **目录命名陷阱**:本地树的路径里虽然有 `router/`,但**它跟 Preble 的路由层毫无关系**。`managers/router/` 是 SGLang v0.1.x 的历史命名,同目录下放的是 `model_runner.py`、`infer_batch.py`、`model_rpc.py`,实质是**引擎执行 + 本地调度**层;其中的 `RouterManager` 指的是「单个实例内部把请求路由进 batch」的事件循环(每个 `ModelRpcClient` 一个),不是负载均衡器。`RadixCache` 由 `ModelRpcServer.__init__` 构造(`model_rpc.py:142`),与 `model_runner` 的 `req_to_token_pool` / `token_to_kv_pool` 直接绑定,**活在模型执行进程内**。Preble 的全局路由完全在 `preble/` 包里,不在 `python/sglang/` 下。
> - `preble/benchmarks/exp_configs/model_equations.py` —— 离线标定的时间回归函数

### 14.1 两棵树的分工:一棵管 token,一棵管显存

Preble 里有**两层结构相似但语义完全不同的 radix 树**,论文没有明确区分,读代码时极易混淆:

| | **全局树** `LPRadixCache` | **本地树** `RadixCache` |
|---|---|---|
| 位置 | Global Scheduler 进程内(`preble/`),**每集群一棵** | **模型执行进程内**(`ModelRpcServer`),每个 TP rank 一棵、组内镜像 |
| 节点 `value` 存什么 | **就是 token id 本身**(`value = [x for x in key]`) | **KV pool 的物理 slot 索引**(`torch.Tensor`) |
| 引用计数 | `ref_counter: List[int]`,**长度 = GPU 数** | `lock_ref: int`,标量 |
| 是否真的持有显存 | **否**,纯元数据 | **是**,`evictable_size_` 直接对应可回收显存 |
| 「驱逐」的含义 | 改 `cached_gpus` / `evicted_gpus` 集合,**节点不删** | 真的 `dec_refs` 释放 KV slot,叶子从树上摘掉 |
| 用途 | 路由决策 | prefix 复用 + 显存管理 |

**全局树不存任何 KV,只存 token 序列 + 「谁缓存了我」的位图**。这是 Preble 全局调度器能做到轻量的根本原因——它的内存占用只与 active 前缀集合的 token 总量成正比,与 KV cache 大小无关。

### 14.2 `TreeNode` 的字段:论文里那「三样东西」其实是七样

`global_lru_cache.py:22`:

```python
class TreeNode:
    def __init__(self, num_nodes):
        self.id = uuid4()                       # 节点身份(用于 __hash__/__eq__)
        self.children = defaultdict(TreeNode)   # 按 key[0] 索引的子节点
        self.parent = None
        self.value = None                       # == key,token id 列表
        self.key = None

        self.ref_counter = [0] * num_nodes      # ★ per-GPU 活跃请求引用计数
        self.load = 0                           # 插入路径经过本节点的累计次数
        self.last_access_time = time.time()
        self.evicted_gpus = set()               # ★ 曾缓存但已被驱逐的 GPU
        self.cached_gpus = set()                # ★ 当前物理缓存了本节点 KV 的 GPU
        self.is_leaf = False
        self.decode_length = deque()            # ★ 该节点上观测到的历史输出长度
        self.context_length = 0                 # ★ root→本节点(含)的累计 token 数
        self.depth = 0

    def has_cached_gpu(self, gpu):
        return gpu in self.cached_gpus and gpu not in self.evicted_gpus

    @property
    def context_so_far(self):                   # ★ 祖先累计 token 数(= cached_len)
        return self.context_length - self.num_tokens

    @property
    def num_tokens(self):                       # ★ 本节点 token 数(= missed_len)
        return len(self.value)
```

对照论文 §5.1 说的「每个节点记三样东西」:

| 论文表述 | 代码字段 | 备注 |
|---|---|---|
| 节点内 token 数 | `num_tokens` = `len(value)` | — |
| 缓存了该节点 KV 的 GPU 集合 | `cached_gpus` − `evicted_gpus` | **两个集合的差**,而不是一个集合 |
| $H$ 内每 GPU 上共享该节点的请求数 | **不在节点上**,在 `SlidingWindowHistogram.node_to_count` | 见 §14.6 |

多出来的三个字段是理解实现的关键:

- **`context_length` / `context_so_far`**:让「祖先总长」变成 $O(1)$ 查询而不用回溯,是 §14.4 判据能写成一行的前提
- **`ref_counter` 是数组不是标量**:同一个前缀节点可能同时被多张 GPU 上的活跃请求引用,每张卡独立计数
- **`decode_length` 是 deque**:节点级别记录历史输出长度,用于 cost 里的 decode 项(论文只说「用 $H$ 内平均输出长度」,实现是**按节点分别统计的中位数**,比论文描述更精细)

**`cached_gpus` / `evicted_gpus` 双集合的设计动机**:`evicted_gpus` 不是冗余的。`_evict_by_node` 收到本地驱逐上报时把 GPU 从 `cached_gpus` 移到 `evicted_gpus`;下次有请求真的被路由过来时,`update_allocated_size` 再把它移回去。这样「被驱逐过」这个事实被保留,而不是简单地从集合里删掉——但代码里 `evicted_gpus` **只在 `has_cached_gpu` 里被读取**,实际语义等价于 `cached_gpus \ evicted_gpus`,属于可以简化的历史包袱。

### 14.3 插入与分裂:标准 radix,但分裂时要搬运调度元数据

```python
def _insert_helper(self, node, key, value, split_nodes,
                   parent_context_length=0, depth=0):
    node.last_access_time = time.time()
    node.load += 1

    if key and key[0] in node.children:
        child = node.children[key[0]]
        prefix_len = match(child.key, key)          # 逐 token 比较

        if prefix_len == len(child.key):            # 完全吃下这个 child
            if prefix_len == len(key):
                child.load += 1
                return child                        # 请求完全命中已有节点
            return self._insert_helper(child, key[prefix_len:], ..., depth=depth+1)

        new_node = self._split_node(child.key, child, prefix_len)   # ★ 分裂
        split_nodes[child] = new_node               # ★ 记录 child → 新父节点
        return self._insert_helper(new_node, key[prefix_len:], ..., depth=depth+2)

    if len(key):                                    # 挂新叶子
        new_node = TreeNode(num_nodes=self.num_gpus)
        ...
        new_node.context_length = parent_context_length + len(key)
        node.children[key[0]] = new_node
        return new_node
```

**关键点 1:`match()` 是逐 token 的 $O(\ell)$ 比较**,不是压缩 trie 的常数时间跳转。ToolBench 上单调度器只有 245 req/s(论文 §8.5),根源就在这里:每个请求要沿路径做 $O(\text{prompt\_len})$ 次 token 比较,而 ToolBench 的 prompt 均值 1835 token、树形结构又最复杂。

**关键点 2:`_split_node` 必须深拷贝三样调度状态**(`global_lru_cache.py:237`):

```python
new_node.cached_gpus  = copy.deepcopy(child.cached_gpus)
new_node.evicted_gpus = copy.deepcopy(child.evicted_gpus)
new_node.ref_counter  = copy.deepcopy(child.ref_counter)
new_node.context_length = child.parent.context_length + split_len
```

语义上:被分裂出的父节点**继承了原节点的全部缓存分布和引用**,因为它是原节点的真前缀,凡是缓存了原节点的 GPU 必然也缓存了它。这是正确的。

**关键点 3:分裂会打断 histogram 的记账,必须显式修补**。`insert` 通过 `split_nodes` 字典把「哪些节点被分裂了」回传给调度器,调度器随即调用(`global_scheduler_with_time.py:222`):

```python
def handle_split_node_histogram(self, split_nodes):
    for child, parent_node in split_nodes.items():
        if self.is_large_node(parent_node) and not self.is_large_node(child):
            self.histogram.rename_node(child, parent_node)      # ★ key portion 转移
            parent_node.decode_length.extend(child.decode_length)
            child.decode_length = []
            for gpu in self.gpu_allocations.get(child, {}):
                self.overload_detector.rename_node(child, parent_node, gpu)
```

**这段是全局树里最微妙的地方**:分裂之后,原来那个「key portion」(large node)可能不再是 large node 了,而新分裂出的父节点变成了 key portion。此时窗口直方图里所有以 `child` 为 key 的统计量——计数、命中率、prefill cost、TTFT 时序——都必须**整体改名迁移**到 `parent_node`。漏掉任何一项,负载账本就会漂移。`rename_node` 里逐字段搬运了 7 个 dict 加一遍 `timestamps` 全表扫描重写(`global_scheduler_with_time.py:98`),**这也是插入路径上一个隐藏的 $O(|W|)$ 开销**。

### 14.4 查找:`find_node` 故意不做分裂

```python
def _match_prefix_helper(self, node, key):
    node.last_access_time = time.time()
    if len(key) == 0:
        return node
    if key[0] in node.children:
        child = node.children[key[0]]
        prefix_len = match(child.key, key)
        if prefix_len < len(child.key):
            assert False            # ★ 部分匹配 → 直接断言失败
        return self._match_prefix_helper(child, key[prefix_len:])
    return node
```

对比本地树的 `_match_prefix_helper`:后者在部分匹配时**会调用 `_split_node`**。全局树这里放了个 `assert False`,因为 `find_node` **只在已经 `insert` 过的完整 `input_ids` 上调用**(`finish_request`、`_evict_by_node`、`update_overload_detector`),路径必然存在且节点边界对齐。这是个契约式断言,不是缺陷——但也意味着**全局树的查找不能用于任意 prompt 的只读前缀匹配**,查询必须先 insert。

也就是说:**Preble 的调度路径上没有「只读匹配」这一步,插入即匹配**。这解释了论文 §7 里「全局 radix 树访问基本无锁」的说法为何在实现中并不成立——`GlobalSchedulerWithTime.runtime_selector` 的整个函数体是包在 `with self.lock:` 里的**一把大锁**(`global_scheduler_with_time.py:316`),因为插入会改树结构。论文描述的无锁方案与开源实现存在出入。

### 14.5 Key Portion 与 exploit/explore 判据:论文公式在代码里的真实形态

**Key Portion 的代码定义**(`global_scheduler_with_time.py:258`):

```python
def is_large_node(self, node):
    return node.num_tokens > node.context_so_far     # 本节点 token 数 > 所有祖先之和

def get_important_node(self, node):
    if self.is_large_node(node):
        return node
    return self.get_important_node(node.parent)      # 否则向上回溯
```

完全对应论文 §3.2 的定义,且因为有 `context_length` 缓存,判断是 $O(1)$、回溯最坏 $O(\text{depth})$。

**然后是最值得注意的一处**——论文的 $\ell_{\text{missed}} < \ell_{\text{cached}}$ 在代码里长这样(`global_scheduler_with_time.py:323`):

```python
leaf_node = self.cache.insert(tuple(input_ids), split_nodes=split_nodes)
important_node = self.get_important_node(leaf_node)

if leaf_node.num_tokens < leaf_node.context_so_far:          # ★ Exploit
    gpu_selected = self.get_parent_gpu_allocation(leaf_node)
    runtime_idx = (self.calculate_min_load_cost(leaf_node, gpu_selected)
                   if len(gpu_selected) > 1 else list(gpu_selected)[0])
elif runtime_id_with_highest_hit_rate is not None:           # ★ 短路
    runtime_idx = runtime_id_with_highest_hit_rate
else:                                                        # ★ Explore
    runtime_idx = self.calculate_min_load_cost(leaf_node, selected_gpus=range(self.num_gpus))
```

三个观察:

1. **判据就是 `is_small_node(leaf_node)`**。插入之后,叶节点的 `num_tokens` 恰好是这条 prompt 未匹配上任何已有节点的**剩余部分**,而 `context_so_far` 恰好是匹配上的部分。所以论文的 $\ell_{\text{missed}} < \ell_{\text{cached}}$ 与 `is_large_node(leaf)` 是**同一个谓词的两种写法**。这个等价关系论文没有点破,但它意味着 **key portion 的概念和 exploit/explore 判据在数学上是同一件事**——一个请求走 exploit 分支,当且仅当它的 key portion 不是自己的新叶子。

2. **Exploit 分支查的是 `gpu_allocations`,不是 `cached_gpus`**。`get_parent_gpu_allocation` 沿父链向上找第一个有非空 `gpu_allocations` 的节点,找不到就返回**全部 GPU**。见 §14.7——这是两套映射,论文只讲了一套。

3. **中间那个 `elif` 分支论文正文里没有对应算法**。`runtime_id_with_highest_hit_rate` 来自 `model_runtime_manager.py:419`:

```python
hit_rates = [r.hit_ratio for r in self.runtimes]     # 后台轮询各 GPU 的 /windowed_prefix_hit_ratio
highest_idx = int(np.argmax(hit_rates))
if hit_rates[highest_idx] < 0.7:
    highest_idx = None                                # 没有明显 decode-heavy 的卡 → 不短路
```

**这就是论文 §5.6 的 P/D 平衡短路的真实实现**:论文写的是 $\max_i \text{DecodeRatio}(i) > \text{IMBAL}_R$,代码里用的信号是**各 GPU 窗口内的 prefix 命中率,阈值 0.7**。二者靠的正是论文那个洞察——「全部命中 ≈ 纯 decode 单元」——所以高命中率的 GPU 就是 decode-heavy 的 GPU。**把一个待 explore 的请求(必然 prefill 重)扔给它**。这个从「decode ratio」到「hit ratio」的替换非常聪明:后者是路由器本来就在收集的信号,零额外成本;但也说明论文里的 $\text{IMBAL}_R$ 并没有直接可测的实现对应物。

### 14.6 负载账本:`SlidingWindowHistogram`

论文的 $L_i$ 在代码里是一个以 **key portion 节点** 为 key 的滑动窗口结构(`global_scheduler_with_time.py:23`),窗口 `timedelta(minutes=3)`:

```python
def update(self, timestamp, node, leaf_node, runtime_idx, decoding_length):
    self.timestamps.append((timestamp, node, leaf_node))
    self.histogram[node]     += leaf_node.context_length     # 加权计数(按整条 prompt 长度)
    self.node_to_count[node] += 1
    self.hit_tokens[node]    += leaf_node.context_so_far     # 命中 token 累计
    self.prompt_tokens[node] += leaf_node.context_length
    self.prev_mis_rates[node] = 1 - hit_tokens/prompt_tokens # ★ per-key-portion 未命中率
    self._remove_old_entries(timestamp)                      # 逐出窗口外条目
    self.update_prefill_cost_per_node(node, runtime_idx)
    self.current_decode_lengths_per_gpu[runtime_idx] += decoding_length
```

**cost 的实际公式**(`global_scheduler_with_time.py:149`):

$$
\text{PrefillCost}(n) = \underbrace{\text{missRate}(n)}_{\text{窗口内统计}} \times \underbrace{\text{count}(n)}_{\text{窗口内请求数}} \times \frac{PT\bigl(n.\text{num\_tokens},\; n.\text{context\_length}\bigr)}{|\text{gpu\_allocations}[n]|}
$$

$$
\text{NodeCost}(n, g) = \text{PrefillCost}(n) \;+\; \underbrace{n.\text{ref\_counter}[g]}_{\text{该卡活跃请求数}} \times \underbrace{\text{median}(n.\text{decode\_length})}_{\text{该节点历史输出长度}} \times \underbrace{\text{TPOT}_g}_{\text{该卡实测中位 TPOT}}
$$

几个论文没写的实现细节:

- **除以 $|\text{gpu\_allocations}[n]|$**:一个 key portion 被复制到 $k$ 张卡后,它的 prefill 负载被认为**均摊到 $k$ 张卡**。这正是 §5.5「prefix autoscaling 按负载切分子树」在 cost 层面的体现
- **TPOT 是在线测量的**,不是离线回归:`avg_topt_per_gpu` 是每卡一个 `deque(maxlen=200)`,`finish_request` 时把实测 TPOT 塞进去,取中位数使用(`global_scheduler_with_time.py:368`)。**所以「Preble 完全不做在线 profiling」的说法只对 prefill 成立,decode 侧是在线自适应的**
- **两条增量路径**:`current_prefill_cost_per_gpu` 增量维护(每次 update 用「新 cost 减旧 cost」修正),而 `current_allocation_per_gpu()` 是**全量遍历 histogram 重算**。前者用于 `_perf` 变体,后者用于正文实现的 `calculate_min_load_cost`——**后者是 $O(|\text{histogram}|)$ 的,在每个请求的关键路径上**,这是 245 req/s 的另一个来源

### 14.7 `cached_gpus` vs `gpu_allocations`:论文只讲了一半

这是读代码才会发现的一组区分:

| | `TreeNode.cached_gpus`(节点字段) | `scheduler.gpu_allocations[node]`(调度器字典) |
|---|---|---|
| 语义 | **物理事实**:这张卡上真的有这段 KV | **逻辑归属**:调度器认为这个前缀「属于」哪几张卡 |
| 谁写 | `update_allocated_size`(路由完成后)、`_evict_by_node`(收到驱逐上报) | `update_gpu_allocation_for_parent`(路由时)、rebalance/autoscale |
| 谁读 | LRU 驱逐、`has_cached_gpu` | **exploit 分支选卡**、cost 分摊除数 |
| 是否会被 rebalance 修改 | **否** | **是** |

**rebalance 只改逻辑归属,不搬运任何 KV**。`self.gpu_allocations[node] = {smaller_device}` 之后,后续命中这个前缀的请求会被导向新卡,在那儿**重算一次** prefill,自然地把 KV「长」过去;旧卡上的副本则通过 LRU 自然老化。这是一个非常轻量的「迁移」实现——**没有 KV 传输,只有路由表改写**。代价是迁移瞬间会有一次冷启动。

`update_children` 递归把整棵子树的归属一并改写(`global_scheduler_with_time.py:459`),保证前缀树的子树不会被劈到不同卡上——**这就是论文 §5.5 里「按负载切分其子树」的实现**。

### 14.8 Rebalance 与 Autoscale 的实际触发逻辑

`handle_important_node_stealing`(`global_scheduler_with_time.py:377`)在**每次路由后同步调用**(不是论文暗示的后台线程),但有多重闸门:

```python
if sum(self.per_gpu_load.values()) < 50 * self.num_gpus:  # 冷启动保护
    return
...
if leaf_node.depth - important_node.depth < self.REBALANCING_CHAIN_LENGTH:  # = 3
    self.handle_important_node_stealing(runtime_idx)      # 长链不 rebalance
```

递归主体的逻辑:

1. 按 cost 降序排序所有 GPU,取**最重的**和**最轻的**配对
2. 若 $\text{cost}_{\max} < 1.5 \times \text{cost}_{\min}$(`HIGH_LOAD_THRESHOLD = 1.5`)→ 返回。**这就是论文的 $Th_{\text{bal}}$,硬编码为 1.5**
3. 收集重卡上所有满足「是 large node(key portion) **且** 窗口内计数 > 1」的节点,按 cost 建小顶堆
4. **分两种情况**:
   - **堆里只有 1 个节点** → 这就是 **prefix autoscaling**:不迁移,而是把轻卡**加入**该节点的 `gpu_allocations`(变成 2 副本,cost 减半)。但**必须先通过 `overload_detector.is_node_overloaded`**
   - **堆里多个节点** → **load shifting**:从 cost 最小的开始逐个把归属整体改到轻卡,直到「搬完之后反而轻卡更重」或比值降到 1.5 以下
5. 用更新后的 cost 递归处理 `allocation_cost_with_devices[1:]`(次重 vs 次轻,如此类推)

**Autoscale 的触发条件在代码里是 TTFT 而不是排队时间**(`ttft_overload_detector.py`):

```python
def is_node_overloaded(self, node, gpu):
    avg_first_half, avg_second_half = self.calculate_half_window_averages((node, gpu))
    return avg_second_half >= 2 * avg_first_half     # ★ 后半窗均值 ≥ 2× 前半窗
```

论文 §5.5 写的是「该前缀的平均排队时间在 $H$ 内翻倍」,实现用的是 **3 分钟窗口一分为二、后半段 TTFT 均值 ≥ 前半段 2 倍**。TTFT 包含排队 + prefill,是排队时间的一个上界代理。副本创建成功后 `delete_after_allocation` 清空该 (node, gpu) 的历史,避免连续扩容。

> **注意 `steal` 的一个副作用**:多节点分支里执行的是 `self.gpu_allocations[node] = {smaller_device}`——**整体替换**而非 `add`。也就是说 load shifting 会**取消重卡对该前缀的归属**,后续请求全部导向轻卡。这与「autoscale 是 add、shift 是 move」的直觉一致,但意味着一旦发生 shift,重卡上那份 KV 立刻变成只等 LRU 回收的死数据。

### 14.9 LRU 的三个变体

全局树里有三个名字相近但用途完全不同的驱逐函数:

| 函数 | 用途 | 是否修改状态 |
|---|---|---|
| `evict()` | **死代码**。`if x.ref_counter > 0` 在 `ref_counter` 是 list 时会抛 `TypeError`,说明这条路径从未被执行 | — |
| `virtual_lru_eviction(n, gpu)` | **路由时的「假驱逐」**,用于估算论文的 $M_i$:模拟「为了放下这个请求要牺牲谁」,返回受害者集合 | **否**,纯模拟 |
| `evict_with_runtime_id_without_removing(n, cb, gpu)` | **记账驱逐**:全局侧模拟某卡显存超限后的驱逐,把 GPU 从 `cached_gpus` 移到 `evicted_gpus`,**节点保留在树上** | 是(仅集合) |

`virtual_lru_eviction` 是论文 $M_i$ 的实现,但**默认关闭**——`enable_eviction` 默认 `False`,注释写着 `NOTE: eviction handled by iterative feedback`。也就是说**主实验跑的路径里,cost 函数只有 $L_i$ 和(隐含在 $L$ 里的)$P_i$,论文的驱逐代价项 $M_i$ 是可选项**。这一点值得注意:论文正文把 $M_i$ 写成 cost 的三分之一,实现里它是一个 flag 后面的东西。

**堆序**(`global_lru_cache.py:53`):

```python
def __lt__(self, other):
    if self.ref_counter == other.ref_counter:            # list == list,逐元素比较
        return self.last_access_time < other.last_access_time
    return self.ref_counter < other.ref_counter          # list < list,字典序比较
```

**这是 per-GPU 引用计数向量的字典序比较**,代码自己也标了 `NOTE: for real eviction this two-level sorting might not follow pure LRU`。含义:优先驱逐「在 GPU 0 上引用少」的节点,GPU 0 相同再看 GPU 1……只有整个向量相等时才退化成真正的 LRU。**这不是论文描述的 LRU**,而是一个偏向低 GPU 编号的启发式。`virtual_lru_eviction` 里用的是同一个 `__lt__`。

另外 `collect_nodes_on_runtime_by_ref_cnt_and_access_time` **遍历 `self.all_nodes` 全集**建堆,即每次虚拟驱逐都是 $O(|V| \log |V|)$。开着 `enable_eviction` 时这会成为主要瓶颈——这大概也是它默认关闭的原因。

### 14.10 反向同步:本地驱逐如何回传全局树

本地树在 `evict()` 中支持 **partial eviction**(`radix_cache.py:133`),这是相对上游 SGLang 的改动:

```python
holding_slots = len(x.value)
desired_eviction = (min(holding_slots, num_tokens - num_evicted)
                    if self.enable_partial_eviction else holding_slots)
num_evicted += evict_callback(x.value[-desired_eviction:]).item()   # 只驱逐尾部若干 token
self._delete_leaf(x, desired_eviction)
if collect_evicted_node:
    self.add_node_to_evicted_iteration(x, desired_eviction)
```

`add_node_to_evicted_iteration` 从被驱逐的叶子**向根回溯拼出完整 `input_ids`**,并记录尾部被驱逐的 token 段,打包成 `EvictionData(input_ids, evicted_ids)`。

全局侧 `_evict_by_node`(`global_lru_cache.py:446`)反向消费:

```python
node = self.find_node(input_ids)              # 定位到叶子
while node and node != self.root_node:
    for k, v in node.parent.children.items():
        if v == node:
            num_eviction = match_from_leaf(k, evicted_ids)   # ★ 后缀匹配
            if num_eviction and gpu_id in node.cached_gpus:
                node.cached_gpus.remove(gpu_id)
                node.evicted_gpus.add(gpu_id)
                evicted_ids = evicted_ids[:-num_eviction]    # 剥掉已处理的尾部
            break
    if not evicted_ids:
        break
    node = node.parent                                        # 继续向根
```

**`match_from_leaf` 做的是从尾部开始的反向 token 匹配**,因为驱逐总是从序列尾部发生的。逐层向根剥离,直到 `evicted_ids` 消耗完。

**这条同步链在主分支里是「发送端接通、接收端断开」的**:

- **worker 侧在推**(`manager.py:73`,每个 forward step 后):

```python
out_pyobjs = await self.model_client.step(next_step_input)
if self.model_client.model_server.tree_cache.evicted_iteration:
    await self.send_to_sched.send_pyobj(
        (self.gpu_id, self.model_client.model_server.tree_cache.evicted_iteration))
    self.model_client.model_server.tree_cache.flush_evicted()
```

  推向 `tcp://127.0.0.1:10340`。socket 本身只在 `enable_iterative_eviction` 为真时才创建(TP > 1 时要求任一 rank 打开该 flag)。

- **全局侧没人接**:`global_lru_cache.py:87-89` 里 bind 10340 的 PULL socket 被注释掉,`update_loop()` 没有任何调用点,`aggregate_eviction_updates` 整块注释。

- **唯一走通全流程的是模拟器**:`simulator.py:517` 里没有进程边界,直接函数调用 `cache._update_eviction_event(runtime_id, tree_cache.evicted_iteration)`。

论文 §6 说的「异步通知 global scheduler 该驱逐事件」因此**在真机部署里是未闭环的**,只在 simulator 实验中成立。

### 14.10.1 推论:全局树是开环估计器

把三类信号摊开看,可信度差别很大:

| 信号 | 来源 | 是否闭环 | 默认是否启用 |
|---|---|---|---|
| **树结构 + `cached_gpus`** | **router 自己推演**:`update_allocated_size` 在路由后乐观地沿父链标记「这张卡缓存了这条路径」 | **否**,worker 从不确认 | 始终 |
| **驱逐事件** | worker `EvictionData` → zmq 10340 | **接收端断开**(仅 simulator 闭环) | 否(`--enable-iterative-eviction`) |
| **router 自行推演驱逐** | `handle_eviction()`,硬编码 `max_tokens_gpu = 198516` | 否 | 否(`enable_eviction`) |
| **窗口 prefix 命中率** | `GET /windowed_prefix_hit_ratio` 后台轮询,读的是 worker 真实 tree_cache 统计 | **是** | 需 `--report-hit-ratio`,否则恒返回 `0.0` |
| **TTFT / TPOT / output_len** | `finish_request` 的 per-request 回填 | **是** | 始终 |

结论:**默认配置下 `cached_gpus` 只增不减**,全局树的缓存视图是一个持续乐观偏移的开环估计;真正闭环的只有命中率标量和延迟指标。

这解释了两件事:

1. **§14.5 那个 `hit_rates[highest_idx] >= 0.7` 短路分支的分量比看起来重得多**——它是少数几个用真实 worker 数据**覆盖**估计值的决策点
2. **cost 模型里 $L_i$ 用「历史窗口」而非「当前树状态」是有必要的**:树状态本身不可信,而窗口内的请求计数和实测 TPOT 是可信的。论文给的理由是「放置是长期决策」,实现上还有一层「估计值会漂移」的现实考量

### 14.11 本地调度器的优先级组

论文 §6 的 $\lfloor P \cdot \ell_{\text{cached}}/\ell_{\text{prompt}} \rfloor$ 在 `model_rpc.py:944` 里是:

```python
self.multi_priority_queue: List[List[Req]] = [[] for _ in range(10)]     # P = 10,硬编码
...
hit_ratio = len(req.prefix_indices) / len(req.input_ids)
group = 0 if hit_ratio == 0 else math.ceil(hit_ratio * 10) - 1           # ★ ceil 不是 floor
self.multi_priority_queue[group].append(req)

schedule_group_idx = self.get_schedule_group()
k = 10                                     # tune k to trade off between cache hit and fairness
max_schedule_allowed = (schedule_group_idx + 1) * k     # ★ 组号越高,单次可调度越多
```

`get_schedule_group()` 用 `(idx + 9) % 10` **从高到低轮转**扫描非空组,即优先服务高命中率组,但每轮只服务一个组、且低命中组同样会轮到——这就是论文说的「按比例从各组抽取」的实现。配额与组号线性相关(组 9 一次最多 100 个,组 0 最多 10 个),对应论文例子里的「优先级 10 组选 10 个、9 组选 9 个」。

另外 `check_req_hit` 会在**组内调度前重新计算一次前缀命中**,注释解释得很清楚:

```python
# 1. hit ratio will shift over time
# 2. if you set longer prefix indices the result is incorrect
```

请求在队列里等待期间,别的请求可能已经把它的前缀 cache 进来了(命中率上升),也可能把它驱逐了(下降)。**不重算就会用陈旧的 `prefix_indices` 去 extend,直接产生错误结果**——这是个正确性问题,不只是性能问题。

### 14.12 时间回归函数:并不是论文说的「线性」

论文 §5.4 和本笔记 §5.4 都说「prefill 时间对 prompt 长度基本线性」。实际标定的函数是**分段线性 + attention 项 + 长 prompt 的二次修正**(`model_equations.py:30`):

```python
def mistral_7b_A6000_sglang_linear(num_batched_tokens):
    if   num_batched_tokens >= 384: return (0.10842571 * n + 4.2098) / 1e3
    elif num_batched_tokens >= 192: return (-118 + 1.25*n - 2.56e-3*n**2) / 1e3
    else:                           return 22 / 1e3          # 短 batch 常数下界

def mistral_7b_A6000_sglang_attention(num_reqs, total_context, num_unique_kv):
    if total_context <= 1024: t = 0.32
    else:
        t = 1.86e-4 * total_context + 0.159
        if num_unique_kv/num_reqs <= 1024 and num_reqs*num_unique_kv <= 32*256*2048:
            t /= 2                                            # ★ 共享前缀带来的 attention 折扣
    return t / 1e3

def ..._extend_flashinfer(...):
    base = linear(...) + attention(...)
    attn_quad = 0
    for extend_len, seq_len in zip(input_id_lens, seq_lens):
        if extend_len * seq_len > 1024 * 1024:                # ★ 二次项只在大 prompt×context 时启用
            attn_quad += 1.13e-3*extend_len + 1.75e-3*seq_len + 2.19e-6*extend_len*seq_len
    return (base + attn_quad/1e3) / 0.9                       # /0.9 经验修正系数
```

修正三点认知:

1. **不是单条直线**,是 3 段:< 192 token 取常数(小 batch 被固定开销主导)、192–384 用二次拟合(过渡区)、≥ 384 才是线性
2. **attention 项显式建模了共享前缀的折扣**(`num_unique_kv` 小时 `/2`),说明标定时已经把 prefix caching 的影响吃进去了
3. **$O(\ell \cdot L)$ 的交叉项在 `extend_len × seq_len > 1M` 时才启用**——本笔记 §10.2 说的「线性假设在超长 context 下失真」,作者其实已经打了补丁,只是论文正文没提。补丁的边界(1M)意味着约 1024×1024 以上才纠偏

**但 §10.2 的核心批评依然成立**:这些系数是 `mistral_7b_A6000` 专属的,文件里为每个 (模型, GPU, TP 配置, 后端) 组合都硬编码了一套。换配置就要重新离线跑一遍 profiling。

### 14.13 实现与论文的差异汇总

给要复现或借鉴的人一张清单:

| 论文表述 | 开源实现 | 影响 |
|---|---|---|
| 「多棵 radix 树,每棵一个 root」 | **单棵树 + 空 root**,不同 prompt 挂在 root 的不同 children 下 | 语义等价,但「多棵树」的说法容易误导 |
| 「全局树访问基本无锁,写操作可用原子指令」 | `runtime_selector` 整体包在 `threading.Lock` 里 | **这是 245 req/s 上限的直接原因** |
| cost = $L_i + M_i + P_i$ | $M_i$ 在 `enable_eviction=False` 时**不参与**,而这是默认值 | 主实验的 cost 实际是两项 |
| $L_i$ 中 decode 用「$H$ 内平均输出长度」 | 用**该 key portion 节点的历史输出长度中位数** + **在线实测 TPOT 中位数** | 比论文精细;且 decode 侧其实是在线 profiling |
| Prefill/Decode 平衡用 $\text{DecodeRatio} > \text{IMBAL}_R$ | 用**各卡窗口 prefix 命中率 > 0.7** 作为代理 | 信号更廉价,但阈值语义不同 |
| Autoscale 触发:「排队时间在 $H$ 内翻倍」 | **TTFT 后半窗均值 ≥ 2× 前半窗** | TTFT 是排队时间的上界代理 |
| 本地驱逐异步上报全局 | **发送端接通**(`manager.py:73` 每 step 推 zmq 10340),**接收端被注释掉**(`global_lru_cache.py:87`);仅 simulator 闭环 | 真机部署下全局树的 `cached_gpus` **只增不减**,是开环估计 |
| LRU 驱逐 | 实为「per-GPU ref_counter 向量字典序 + last_access_time」两级排序 | 偏向低编号 GPU,非纯 LRU |
| $Th_{\text{bal}}$「从 profiling 得出」 | 硬编码 `HIGH_LOAD_THRESHOLD = 1.5` | — |
| Prefill 时间「线性」 | 3 段拟合 + attention 项 + 大 prompt 二次交叉项 | 论文简化了,实现更稳健 |

### 14.14 可直接借鉴的实现手法

1. **`context_length` 缓存在节点上**,让「祖先总长」$O(1)$ 可得——这是把 exploit/explore 判据、key portion 判断、cost 计算全部压到常数时间的前提
2. **`split_nodes` 回调字典**:radix 树分裂是所有「在节点上挂统计量」的设计的公共难题。把分裂事件显式回传给上层做元数据迁移,比让树自己感知统计语义要干净得多
3. **`gpu_allocations`(逻辑归属)与 `cached_gpus`(物理事实)分离**:前者可以被调度器任意改写来实现零成本迁移,后者只由真实事件驱动。任何 cache-aware 路由器都值得做这个区分
4. **rebalance 通过「改路由表 + 让 KV 自然长过去」实现迁移**,不传输任何 KV。在没有 KV 传输通道的系统里这是唯一可行的迁移方式,代价只是一次冷启动
5. **`check_req_hit` 在出队前重算命中**:任何「入队时算命中率、出队时用」的设计都有这个正确性陷阱
6. **partial eviction**(只驱逐节点尾部若干 token 而非整个节点):相对上游 SGLang 的一个小而实用的改动,让驱逐粒度与需求量对齐,避免超额驱逐
