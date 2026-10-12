# SkyWalker: A Locality-Aware Cross-Region Load Balancer for LLM Inference —— 论文解读

> 论文：*SkyWalker: A Locality-Aware Cross-Region Load Balancer for LLM Inference*
> 作者：Tian Xia、Ziming Mao、Jamison Kerney、Ethan J. Jackson（UC Berkeley）、Zhifei Li（中国人民大学）、Jiarong Xing（Rice）、Scott Shenker（UC Berkeley / ICSI）、Ion Stoica（UC Berkeley）
> 会议：**EuroSys '26**（Edinburgh，2026-04-27~30），DOI 10.1145/3767295.3769353，arXiv:2505.24095v2
> 早期版本曾名 **SkyLB**；构建于 SkyServe（EuroSys '25）之上，约 3000 行 Python

---

## 1. 一句话概括

SkyWalker 把「KV cache 亲和 vs 负载均衡」这个单区域问题，抬到了**跨地理区域**的层面：既然各区域的流量高峰因时区错开，那就不该按**各区峰值**买机器，而应按**全球峰值**买、再跨区共享。为此它用**两层路由**（LB↔LB 协调，而不是 LB↔replica 直连）解决跨区协调开销，用**区域快照前缀树**近似全局 cache 视图，用**基于 pending requests 的选择性推送**解决 LLM 负载不可预测的问题。结果：吞吐 1.12–2.06×、延迟降低 1.74–6.30×、总服务成本降低 **25%**。

---

## 2. 它要解决的问题

这是本系列里**唯一一篇把成本而非延迟当作一号目标**的论文，问题设定与 Preble / DualMap / CacheRoute 完全不同。

### 2.1 为什么必须长期承诺买机器

GPU 稀缺 + 价格差异巨大：AWS `p5.48xlarge`（8×H100）三年预留价 **\$37.56/小时**，同规格按需价 **\$98.32/小时**——**2.6 倍差**。自建机房按全生命周期 ROI 算，还能比预留云实例再省最多 46.3%。所以主流厂商（OpenAI、字节、xAI）都用预留实例或自建集群。

代价是**没有弹性**：区域内容量买定就不能变。

### 2.2 昼夜节律导致的浪费

WildChat trace 显示，美国、俄罗斯、中国、英国、德国、法国各有各的流量高峰，因时区错开。论文用 5 个 AWS region 的子集测算：

- 聚合前，**单区域负载方差 2.88× ~ 32.64×**
- 聚合后，**全球总负载方差仅 1.29×**

于是：按各区独立峰值配置 vs 按全球聚合峰值配置，**成本可降 40.5%**。而"完美按需自动扩缩"（假设预测精准、零启动延迟——现实中不可能）仍然要花聚合预留方案的 **2.2 倍**。

> 这个论证是全文的基石，也是它区别于所有其他调度论文的地方：**不是让同样的机器跑得更快，而是让需要的机器变少。**

### 2.3 为什么跨区路由在 LLM 上才成立

传统 Web 负载（网页渲染、搜索查询）单请求处理时间只有几十毫秒，而跨区网络延迟高达 200ms——跨区转发的开销远大于收益，所以从来不做。

**LLM 请求要几秒到几十秒**。200ms 相对总时长只是零头。这就第一次让"跨区卸载"在经济上成立了。

### 2.4 天真做法为什么不行

把负载均衡器集中部署在某个 zone（图 1b）会引入：
- **两次跨区 RTT**（客户端→LB，LB→replica）
- 中心化性能瓶颈
- 单点故障

---

## 3. 两个核心挑战

### 3.1 KV Cache 感知的跨区协调

已有前缀感知路由（Preble 的精确前缀树、SGLang Router / DLPM 的近似前缀树）**都假设单区域、单负载均衡器**。扩到多区域时：

- **集中式全局前缀树** → 每个请求都要远程访问，延迟惩罚不可接受
- **各区独立前缀树** → 每个请求都要跨区协调，开销同样不可接受

而且如果多个 LB 各自独立决策，它们**可能同时把同一个热前缀选成目标**，制造出集中式方案不会有的热点。协调本身需要 $O(N_{LB} \times N_{replica})$ 条连接或探测。

### 3.2 LLM 推理负载的不可预测性

输出长度方差极大且无法事先预测（自回归特性）。单个请求可能吃掉几 GB 显存，一个 replica 只能并发几十个请求。后果：

- **Round Robin 下两个 replica 的显存占用差可达 2.64×**（图 4b）
- 队列看起来短的 replica 可能正在跑一堆长请求

论文的措辞很到位：**LLM 工作负载中「路由错误的惩罚（penalty of misrouting）」远比传统 CPU 负载严重**。

---

## 4. 核心设计

### 4.1 两层跨区路由

关键决策：**LB 之间协调，而不是 LB 直连所有 replica**。

```text
Client → (DNS 解析到最近的 LB) → 本区 LB
                                    ├─ 本区有可用 replica → 直接路由（本地优先）
                                    └─ 本区全满 → 转发给远端 LB
                                                    └─ 远端 LB 在其区内做最终放置决策
```

三个好处：
1. 客户端永远连最近的 LB → 不吃第一跳跨区延迟
2. 探测开销从 $O(N_{LB} \times N_{replica})$ 降到 $O(N_{LB}^2)$ —— replica 数远多于 LB 数
3. 新 replica 上线只需通知本区 LB，不用广播全局

**本地优先原则**是硬性的：只要本区还有一个非满 replica，就绝不跨区。跨区只在本区饱和时触发。

### 4.2 前缀相似度分析：为什么 user ID 就够用了大半

作者定义前缀相似度：

$$\text{sim}(a,b) = \frac{\text{len}(\text{common\_prefix}(a,b))}{\min(\text{len}(a), \text{len}(b))}$$

（分母取 min，使得 $a$ 是 $b$ 的前缀时相似度为 1。）

在 WildChat 和 ChatBot Arena 上的测量结果：

| 维度 | ChatBot Arena | WildChat (User) | WildChat (Region) |
| --- | --- | --- | --- |
| 同用户内 | 20.5% | 19.0% | — |
| 跨用户 | 8.3% | 2.5% | — |
| 同区域内 | — | — | 10.9% |
| **跨区域** | — | — | **2.5%** |

两个关键结论：

1. **同用户内相似度是跨用户的 2.47×（Arena）~ 7.60×（WildChat）** → 用 user ID 做一致性哈希就能抓住大部分 cache 复用
2. **跨区域前缀相似度只有 2.5%** → 这是 SkyWalker 敢用"区域快照"近似的**实证依据**：本区请求几乎不会和其他区送过来的请求共享前缀，所以本区 LB 只需记住"我自己发到远端的那些请求"就够了

### 4.3 两种路由策略

#### SkyWalker-CH（隐式前缀感知）

对 user ID / session ID 做**一致性哈希**（ring hash）。同一用户的请求自然落到同一 replica，隐式实现了 cache 亲和。两处扩展：

1. **两层都做一致性哈希**：LB→LB 一次，LB→replica 一次
2. **虚拟节点按可用性跳过**：目标 replica 满了就沿环继续往下找

优点是状态极少、几乎不用改现有软件栈。三个已知失效场景（论文用图 6 量化）：

| 失效场景 | 命中率损失 | 原因 |
| --- | --- | --- |
| **跨用户前缀共享** | −16.49% | 不同用户共享的前缀被哈希到不同 replica |
| **突发请求** | −7.07% | 单用户并发方差可达 4×，全砸到同一 replica |
| **异构用户程序** | −8.78% | 同一用户内前缀结构不一致，user ID 做 key 失效，甚至把不相似请求塞进同一 replica |

#### SkyWalker（显式前缀感知 + 区域快照）

每个 LB 维护**两棵前缀树**：

- **本地树**：管辖的本区 replica 的前缀视图
- **区域快照树（regional snapshot）**：本区**历史上发往各远端区域**的请求前缀

树是**增量构建**的：请求转发时把路径插入 trie，并在路径上每个节点记录选中的目标。有配置的最大树尺寸，超限按插入时间最早的先淘汰。

查找时的**关键剪枝**：

> 遍历到某节点若无「可用」的匹配目标，立即终止。因为**任何子节点的目标集合必然是其父节点的子集**——再往下走也找不到可用 replica。

区域快照**不是**远端区域真实前缀集合的镜像（远端还接收着别的区和它自己本地的请求），而是「本区转发过去、因而可能被复用的前缀」的近似。这个近似之所以成立，正是因为 4.2 节测出的跨区相似度只有 2.5%。

### 4.4 选择性推送（Selective Pushing）

这是本文第二个核心贡献，也是最实用的一条。论文对比三种策略：

| 策略 | 机制 | 问题 |
| --- | --- | --- |
| **Blind Pushing (BP)** | 请求一到就立刻推给某 replica（RR / Least-Load / Preble / SGLang Router 都是） | LLM 处理时间不可预测，队列短 ≠ 负载轻。RR 下显存差 2.64× |
| **SP-O**（限制 outstanding 请求数） | 固定阈值，超过就在 LB 排队 | 阈值该设多少取决于总 token 占用，而输出长度不可预测。实测 Llama-3.1-8B on L4 下最大并发在 **20~50** 之间浮动 |
| **SP-P**（检查 pending 请求）✅ | 用**是否存在 pending 请求**判断 replica 是否已满 | —— |

**pending request** 的定义是关键：**尚未被调度进 continuous batch 的请求**。它存在，就意味着当前 batch 已被 GPU 显存约束卡满、无法再接纳新请求。

这个信号的精妙之处在于：**它不是一个需要调参的阈值，而是推理引擎自己给出的、真实反映显存约束的二值信号**。SP-O 需要猜"多少个请求算满"，SP-P 直接问引擎"你满了吗"。

探测周期 **100ms**（依据：一个 continuous batching step 通常只需几十毫秒，100ms 在响应性与探测开销间取平衡）。

### 4.5 完整路由算法

```text
procedure MonitorAvailability:                      # 后台心跳
  loop:
    for r in LocalReplicas:
      n_pending ← ProbeReplica(r)
      if n_pending > 0: remove r from LocalAvail    # 满了
      else:             add    r to LocalAvail
    for lb in RemoteLBs:
      (n_avail_replica, size_q) ← ProbeLB(lb)
      if n_avail_replica == 0 or size_q > τ:        # τ: 新到请求的小缓冲
           remove lb from RemoteAvail
      else: add    lb to RemoteAvail
    sleep(ProbeInterval)                            # 100ms

procedure SelectCandidate(Request, C):
  if UsePrefixTree:
    Trie ← ReplicaTrie if C is replicas else LBSnapshotTrie
    return MaxPrefixMatch(Trie, GetText(Request), C)
  else:                                             # SkyWalker-CH
    HashRing ← ReplicaRing if C is replicas else LBRing
    return Next(HashRing, Hash(SessionID(Request)), C)

procedure HandleRequest(Request):
  C ← LocalAvail if LocalAvail ≠ ∅ else RemoteAvail   # 本地优先
  t ← SelectCandidate(Request, C)
  Route(Request, t)
```

注意结构：**可用性先做硬过滤，前缀亲和只用来在可用集合内部打破平局**。这与 Preble（先算亲和、再用 load cost 权衡）和 Dynamo（亲和与负载加权相减）的哲学不同——SkyWalker 把负载当作**约束**而非**目标**。

### 4.6 系统实现

- 构建于 **SkyServe**（EuroSys '25，同组工作），约 3000 行 Python
- 兼容任何 OpenAI API 接口的推理引擎（vLLM、SGLang）
- **AWS Route53 DNS** 为每个 LB 建记录，共用一个域名（`sky.walker.net`），按客户端源 IP 解析到最近的 LB
- 请求进 LB 后进 **FCFS 队列**，到队首才做路由决策（这正是选择性推送的载体）
- **中心化 controller** 管理 replica 增删、LB 重配置、健康探测；controller 自身可故障恢复
- **LB 故障恢复**：controller 探测到失败后，把该区 replica 临时交给地理最近的 LB 托管（视作本地 replica），后台恢复；恢复后交还。支持多 LB 并发故障
- **可定制路由策略以满足 GDPR**：可限制只在合规区域间卸载；非 GDPR 区域仍可在 GDPR 区域空闲时向其卸载

---

## 5. 实验方法

**硬件**：AWS（GKE Gateway 对比在 GCP），每 replica 1× **L4 GPU**，最多 12 replica
**模型**：`meta-llama/Llama-3.1-8B-Instruct`，引擎 SGLang
**区域**：美国、欧洲、亚洲三区，客户端也分布在三区
**基线**：GKE Gateway、Round Robin、Least Load、Consistent Hashing、SGLang Router、SkyWalker-CH、SkyWalker
（RR/LL/CH/SGL 四个基线都改造 SkyServe 实现，LB 单点部署在美国；SkyWalker 两个变体与 GKE Gateway 每区一个 LB）

**负载**：

| 负载 | 说明 | 客户端配置 |
| --- | --- | --- |
| ChatBot Arena | 真实多轮对话，用真实 user ID 做哈希键 | 每区 80 并发会话 |
| WildChat | 100 万条多轮对话 + 地理元数据，用 hashed IP 做哈希键；各区只请求本区数据 | US 40 / EU 30 / Asia 30 |
| Tree of Thoughts (ToT) | GSM8K 数学推理，2 分支树（15 请求/树），深度 4，题目 ID 做哈希键 | US 40 / EU 20 / Asia 20；12 replica 均分三区 |
| Mixed Tree | US 跑 4 分支树（85 请求/树，2 并发），其余区跑 2 分支树（20 并发） | 异构负载场景 |

---

## 6. 主要结果

### 6.1 端到端

| 指标 | 结果 |
| --- | --- |
| 吞吐（对单 LB 方案） | 1.12–1.2× |
| 吞吐（对 GKE Gateway） | **1.43–2.06×** |
| P50 / mean TTFT | 基线的 15.87%–57.63% |
| **P90 TTFT** | 基线的 **10.08%–23.38%** |
| P50 端到端延迟 | 1.05–2.14× 改善 |
| **成本** | 9 replica 达到 region-local 12 replica 的吞吐 → **省 25%** |

### 6.2 命中率与负载方差

| 系统 | 前缀命中率（对话负载） | outstanding 请求方差 |
| --- | --- | --- |
| Round Robin | 10.78–16.57% | — |
| Least Load | 28.29–31.13% | — |
| GKE Gateway | 18.08–24.30% | — |
| CH | — | 2.54–4.92× |
| SGLang Router | — | 2.33–5.08× |
| **SkyWalker** | **36.96–46.55%** | 低 |

ToT / Mixed Tree（前缀共享极高的场景）：LL/RR 命中率 58.66–59.32%，**SkyWalker 达 89.56–90.01%**。

### 6.3 一个诚实的反例：ToT 上 CH 略胜 SkyWalker

**均匀树的 ToT 负载下，CH 基线比 SkyWalker 吞吐高 2%**。原因很直接：CH 把同一棵树（同一道题）的请求全哈希到同一 replica，前缀复用做到了极致；而 SkyWalker 的前缀树变体在命中率 < 50% 时会**主动去探索其他空闲 replica**，付出了额外的 prefill 时间。

但**这个优势在异构负载（Mixed Tree）下立刻消失**：CH 继续把突发用户的请求死死钉在同一 replica 上，outstanding 请求方差飙到 **3.36×**（SGLang Router 也有 2.22×）。

作者的总结很克制：SkyWalker 相对 SkyWalker-CH 一致领先 **1.34–8.21%**，代价是 ToT 上 TTFT 略高——**用一点点 TTFT 换整体吞吐和鲁棒性**。

### 6.4 选择性推送的独立验证（单区域微基准）

在 SGLang Router 上分别嫁接三种推送策略，4 replica / 30 client / ToT(branch=2)，全部同区共置：

| 策略 | 吞吐 | P90 TTFT | KV 命中率 |
| --- | --- | --- | --- |
| BP（原版盲推） | 1× | 1× | 68.89% |
| SP-O（固定 outstanding 阈值） | — | ≈ SP-P | — |
| **SP-P（pending 检查）** | **1.27× vs BP，1.4× vs SP-O** | **18.47× 优于 BP** | **89.86%** |

> **P90 TTFT 改善 18.47×** 是全文最惊人的单项数字。而且这是在**单区域**测的——说明选择性推送的价值完全独立于跨区设计，**任何单区域路由器都可以直接吸收这一条**。

### 6.5 昼夜节律场景

US 120 client / Asia 40 / EU 40 的倾斜负载，replica 三区均分：

- 同 replica 数下，SkyWalker 比 region-local 吞吐高 **1.07–1.18×**
- SkyWalker 用 **9 replica** 达到 region-local **12 replica** 的吞吐 → **成本降 25%**

---

## 7. 负面结果与运行边界

1. **ToT 均匀负载下输给 CH 2%**（已在 6.3 讨论）。前缀树的探索行为在共享极度规整时是净损失。
2. **区域快照的近似依赖 2.5% 跨区相似度这一实测值**。如果某个部署里全球用户共享同一批系统提示词或工具描述（企业级 agent 场景很常见），跨区相似度会显著高于 2.5%，快照的近似就会失效——本区 LB 会低估远端 cache 的可用性。
3. **只支持同构 replica**。作者在 §7 明确承认异构加速器（TPU、Inferentia、不同型号 GPU）下前缀感知路由与均衡策略"仍是开放问题"；但指出 SP-P 本身是**硬件无关**的（它只问引擎满没满），天然可迁移。
4. **不做请求特征感知**。§7 承认：短 prompt 的 prefill 成本低，其实更应该优先送给负载稍轻的 replica 而非追求前缀复用。当前策略对所有请求一视同仁。
5. **GDPR 限制全球聚合**。欧盟数据不能出境，所以"全球峰值配置"这个核心经济论证在受监管场景下要打折——只能在 EU 内部聚合，或让非 EU 流量单向卸载到 EU。
6. **前缀共享的侧信道风险**（§7）：已有工作证明 KV cache 共享可被用于跨用户 prompt 泄露（NDSS'25）与时序侧信道（ePrint 2025/167）。作者认为这与 SkyWalker 设计正交，适用于所有 cache 共享系统。这个免责在技术上成立，但跨区共享把攻击面从"同一集群内的租户"扩大到了"全球租户"，风险量级并不相同。
7. **规模有限**：最多 12 个 L4，8B 模型。相比 CacheRoute 的 60×H100 / 70B，这是一个**小得多的 testbed**。跨区设计的价值主张（协调开销）恰恰在大规模才充分暴露。
8. **前缀树用 FIFO 淘汰**（最早插入的先删），而不是 LRU 或按热度。简单，但对长期活跃的热前缀不友好。

---

## 8. 与相关工作的定位

| 系统 | 部署范围 | 协调结构 | 前缀感知 | 负载信号 | 一号目标 |
| --- | --- | --- | --- | --- | --- |
| Preble | 单区域 | 中心化全局调度器 | 精确前缀树 | prompt-aware load cost | 延迟/吞吐 |
| SGLang Router | 单区域 | 全局前缀树 | 近似 | 盲推 | 吞吐 |
| DLPM | 单区域 | 中心化 | 精确 | 带公平性保证 | 延迟 + 公平 |
| DualMap | 单集群 | 无状态双哈希 | 双候选 | TTFT 预测 | 亲和 + 均衡 |
| CacheRoute (Meta) | 单集群 | 周期性全局规划 | LPT 放置 | 规划期容量 | 尾延迟容量 |
| Amazon Bedrock | 跨区 | 托管服务 | ❌ | ❌ | 弹性（限同大洲） |
| GKE Gateway | 跨集群 | 统一入口 | ❌ | ❌ | 通用性 |
| **SkyWalker** | **跨区域** | **两层 LB↔LB** | **CH / 区域快照树** | **pending requests** | **全球成本** |

**SkyWalker 对 Preble 的批评**很直接：Preble 依赖中心化全局调度器，跨区部署时既有跨区通信延迟，又是单点故障，"不适合生产级跨区部署"。这个批评在跨区场景下是成立的，但也要注意 Preble 从未声称要解决跨区问题。

**与 Amazon Bedrock 的区别**：Bedrock 确实支持跨区推理来吸收流量尖峰，但**卸载被限制在同一大洲内**——这恰恰错过了聚合昼夜节律的机会（同大洲时区差异小）。而且 Bedrock 是托管服务，SkyWalker 是可自建的。

---

## 9. 工程借鉴清单

1. **选择性推送 + pending requests 信号，是可以立刻嫁接到任何 LB 上的改动**。P90 TTFT 改善 18.47×，且是在单区域测的。核心洞察：**别猜阈值，直接问推理引擎"你的 continuous batch 满了吗"**。
2. **可用性做硬过滤、亲和只用于打破平局**——这个决策顺序比"加权打分"更鲁棒。加权打分总有权重错配的场景；硬过滤保证了永远不会把请求送进已满的队列。
3. **LB 在队首才做路由决策**，而不是请求一到就决策。这一步把"决策时机"从入队推迟到出队，让决策基于最新状态。
4. **两层协调**（对等层协调而非全连接）在任何 $N_{LB} \ll N_{replica}$ 的系统里都值得考虑，探测开销从 $O(N_{LB} \times N_{replica})$ 降到 $O(N_{LB}^2)$。
5. **前缀树剪枝依据"子节点目标集 ⊆ 父节点目标集"**。这个不变量让遍历可以在第一个无可用目标的节点提前终止，是很干净的实现技巧。
6. **DNS + 源 IP 就地解决"就近接入"**，不需要 anycast 或复杂的边缘路由。
7. **先量化再设计**：作者先测出"跨区前缀相似度只有 2.5%"，才敢用区域快照近似。任何近似都应该有这样一个测量作为依据，而不是直接假设。
8. **user ID 做一致性哈希是极高性价比的起点**。同用户内相似度是跨用户的 2.5–7.6×，意味着最简单的 CH 就能抓住大部分收益。先上 CH，等测出它的三个失效场景真的发生了再上前缀树。
9. **探测周期对齐系统的自然时间尺度**：continuous batch step 是几十毫秒 → 探测周期设 100ms。这个"周期 ≈ 3~5 倍被观测事件时长"的经验法则值得复用。

---

## 10. 个人评述

### 亮点

- **换了一个问题**。本系列其他论文都在问"怎么让延迟更低"，SkyWalker 问的是"怎么让机器买得更少"。40.5% 的理论成本节省与 25% 的实测节省，这是 CFO 能听懂的数字，也是最容易在企业内部立项的角度。
- **"LLM 请求足够长，所以跨区终于划算了"这个论证极其干净**。它把一个二十年来因延迟比例不成立而无人问津的方向，用一句话重新打开了。
- **pending requests 作为负载信号是全文最实用的贡献**，而且被诚实地在单区域独立验证了（§5.2）——作者主动把自己最强的一个技术点从系统主线里剥离出来单独证明，这既是学术严谨也是对读者的慷慨。
- **区域快照的近似有实测依据**（2.5% 跨区相似度）而非拍脑袋。
- **诚实报告了 CH 在 ToT 上赢自己 2%**，并解释了原因。这种自曝短板在系统论文里不多见。
- **GDPR 和侧信道都被正面讨论了**，没有回避跨区共享带来的合规与安全新问题。

### 需要留意的

- **testbed 规模偏小**：12 个 L4 GPU / 8B 模型。跨区协调的开销问题在这个规模下几乎不会暴露——两层路由 vs 全连接的差别，在 $N_{replica} = 12$ 时是 $12 N_{LB}$ vs $9$，谈不上量级差。论文的核心设计论证（协调开销）与实验规模不匹配。
- **成本论证与性能实验用的是两套东西**：成本节省来自 WildChat trace 的**离线测算**（图 3b，40.5%）和一个**模拟倾斜负载**的实验（图 10，25%），而性能数字来自另一组实验。两者没有在同一个端到端场景里合起来验证过"跨区卸载既省钱又不掉性能"。
- **2.5% 跨区相似度是数据集特性，不是普遍规律**。WildChat 是野生 ChatGPT 对话，用户之间本来就没什么共享结构。企业 agent 场景（统一的系统提示词 + 工具描述）跨区相似度会高得多，届时区域快照近似会系统性地低估远端 cache。这是设计中最脆弱的假设，而论文只用一个数据集支撑它。
- **前缀树的 FIFO 淘汰**在长时间运行下会不断丢掉热前缀。生产系统大概率要换成 LRU 或带频次的策略。
- **"本地优先"是硬规则**，没有例外机制。如果本区有一个刚好持有该请求全部前缀的 replica 恰好满了，而远端有个空闲 replica 也持有该前缀，当前策略会等而不是转发——除非本区**所有** replica 都满。这在前缀高度倾斜时可能不是最优。
- **异构加速器与请求特征感知都被推给了 future work**，而这两点在真实的多区域部署里几乎必然存在（不同区域可获得的 GPU 型号本来就不同）。

### 一句话定位

**SkyWalker 是这条研究线上第一个把"省钱"而不是"变快"当作一号目标的工作**，它的成本论证（时区错峰 → 按全球峰值而非各区峰值配置 → 省 40.5%）比它的技术方案更有原创性。技术上真正可以立刻拿走的是 **selective pushing by pending requests**——一个不需要调参、硬件无关、单区域就能生效、P90 TTFT 改善 18 倍的信号。跨区那套两层架构的价值则要等到更大规模才能被真正检验。

---

## 附：与本仓库其他解读的交叉引用

- [`Preble-论文解读.md`](Preble-论文解读.md) —— SkyWalker 明确批评其中心化调度器不适合跨区；两者的 workload 假设也不同
- [`DualMap-论文解读.md`](DualMap-论文解读.md) —— 同为解决亲和/均衡冲突，但 DualMap 改映射几何、SkyWalker 改决策顺序（可用性硬过滤 + 亲和打破平局）
- [`CacheRoute-论文解读.md`](CacheRoute-论文解读.md) —— 规模最大的 testbed（60×H100），可与 SkyWalker 的 12×L4 对照阅读
- [`LLM服务负载均衡与调度-文献综述.md`](LLM服务负载均衡与调度-文献综述.md) —— 整体脉络定位
