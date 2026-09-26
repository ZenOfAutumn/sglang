# From Tensor Buffer to Distributed Memory Hierarchy：KV Cache 管理综述解读

> 论文：*From Tensor Buffer to Distributed Memory Hierarchy: A Survey of KV Cache Management for LLM Serving*
> 作者：Jie Li、Tongyang Wang、Yong Chen（Texas Tech University）
> 版本：arXiv:2607.02574v1，2026-06-30
> 领域：分布式系统、LLM Serving、KV Cache、内存层次
> 原文：https://arxiv.org/abs/2607.02574
> 说明：这是一篇**分类与研究议程型综述**，不是提出新系统并进行端到端性能实验的系统论文。

---

## 1. 一句话概括

这篇论文的核心观点是：**KV Cache 已经不再只是某个请求在单张 GPU 上临时保存的张量，而正在演化成一种具有位置、生命周期、所有权和访问介质语义的分布式内存对象。**

作者据此用四个轴统一分类 35 个条目，覆盖 33 个不同的系统或框架模式：

1. **Locality：KV 在哪里被消费者获得；**
2. **Lifetime：KV 要存活多久、允许跨多大范围复用；**
3. **Ownership：谁负责命名、放置、驱逐和并发仲裁；**
4. **Substrate：KV 通过什么硬件或软件路径被访问和搬运。**

四轴分类进一步归纳出五种架构原型：

- Local-paged；
- Disaggregated-pipeline；
- Shared-store；
- Memory-pool；
- Hybrid-tier。

论文最有价值的结论并不是“哪个系统最快”，而是：**当工作负载和硬件条件基本确定以后，分布式 KV 系统之间最关键、也最容易被端到端指标掩盖的差异，往往是所有权模型。**

---

## 2. 论文试图解决什么问题

### 2.1 传统抽象：KV Cache 是请求私有的临时 Tensor

传统 LLM 推理通常把 KV Cache 理解为：

- 请求到达后，在当前 GPU 上分配；
- Prefill 阶段写入 prompt 对应的 Key 和 Value；
- Decode 阶段反复读取，并在每一步追加新 token 的 KV；
- 请求结束后释放。

这个模型默认了三个条件：

1. KV 不离开生成它的 worker；
2. KV 不跨请求存活；
3. 本地执行引擎拥有完整控制权。

在短上下文、低并发、单轮请求场景中，这个抽象足够有效。但进入长上下文、高并发、多轮会话、RAG 和 Agent 工作负载后，这三个假设都开始失效。

### 2.2 为什么单 GPU、本地请求级抽象不够用了

论文指出三类趋势正在迫使 KV 跨越原来的边界。

#### 趋势一：上下文越来越长

KV Cache 容量随上下文长度线性增长。对于 decoder-only 模型，单请求 KV Cache 的逻辑大小为：

$$
M_{\mathrm{KV}} = 2 L H_{\mathrm{kv}} D S b
$$

其中：

- $L$：Transformer 层数；
- $H_{kv}$：KV head 数量；
- $D$：每个 head 的维度；
- $S$：驻留 token 数；
- $b$：每个 KV 元素占用的字节数；
- 系数 2：分别存储 Key 与 Value。

对 BF16/FP16，$b=2$，因此：

$$
M_{\mathrm{KV}} = 4 L H_{\mathrm{kv}} D S
$$

论文以类似 Llama-3.1-70B 的 GQA 模型为例：

- $L=80$；
- $H_{kv}=8$；
- $D=128$；
- 上下文为 128K tokens。

单请求 KV Cache 约为 **40 GiB**；即使上下文只有 32K，单请求仍约为 **10 GiB**。100 个并发请求的聚合 KV 状态接近 **1 TiB**。

这还没有计入：

- 分配器元数据；
- 分页开销；
- 内存碎片；
- CUDA Graph；
- 临时计算缓冲区；
- 模型权重。

因此，问题并不是“显存利用率还能不能再优化几个百分点”，而是**单卡 HBM 在容量上已经无法承载完整工作集**。

#### 趋势二：跨请求和跨轮次复用越来越普遍

以下内容经常被重复计算：

- 系统提示词；
- 工具定义；
- Few-shot 示例；
- RAG 检索文档；
- 多轮对话历史；
- Agent 的公共工作流前缀。

如果请求结束即释放 KV，或者相同前缀被路由到不同实例，系统就会重复执行昂贵的 Prefill。

#### 趋势三：Agent 和分支推理打破线性生命周期

Agent 工作流可能：

- 从中间状态分叉；
- 回退到早期状态；
- 复用已有推理上下文；
- 在多个 LoRA、工具或执行阶段之间迁移。

因此 KV 的生命周期不再总是“创建—线性增长—释放”，而可能变成带有分叉、共享、回滚和恢复的状态图。

### 2.3 容量不是唯一问题：Decode 的累计读取量更惊人

在 Decode 第 $t$ 步，新 Query 需要读取 prompt 以及此前生成 token 的全部 KV。若 prompt 长度为 $S$，输出长度为 $T$，则累计读取的 token 位置数量为：

$$
\sum_{t=0}^{T-1} (S+t)
= TS + \frac{T(T-1)}{2}
$$

所以标准稠密注意力下，Decode 的累计 KV 读取流量为：

$$
\mathcal{O}(TS+T^2)
$$

论文给出的数量级示例是：若某个 70B GQA 模型每 token KV 约 320 KiB，prompt 为 128K tokens，并继续生成 32K tokens，则累计 KV 读取量约为 **1.4–1.6 PB**。

这解释了一个关键事实：

> KV 管理既是容量问题，也是带宽、访问延迟和数据移动问题。

远端内存容量再大，如果每个 Decode step 都同步等待远端 KV，性能仍可能不可接受。

---

## 3. 从 PagedAttention 到“可管理的 KV 对象”

### 3.1 PagedAttention 的意义不仅是减少碎片

PagedAttention 将连续的大 KV Tensor 拆成固定大小的 block，并通过每个序列的 block table 建立逻辑 token 到物理 block 的映射。

直接收益包括：

- 按需增长，避免按最大长度预留；
- 降低外部碎片；
- 请求结束后可回收 block；
- beam search 或并行采样可以共享 prompt block；
- 为 copy-on-write、迁移、驱逐和跨请求复用提供统一粒度。

更深层意义是：**KV 从匿名内存区域变成了能够被寻址、引用计数、共享、驱逐和迁移的对象。**

### 3.2 RadixAttention 又增加了“名字”

SGLang 的 RadixAttention 通过 radix tree 表示 token prefix，使 KV 不再只有地址，还拥有由 token 序列决定的身份。

新请求可以：

1. 在 radix tree 中寻找最长匹配前缀；
2. 直接引用匹配部分对应的物理 KV block；
3. 只计算未匹配后缀。

一旦 KV 可以跨请求复用，系统就必须回答新的问题：

- 两个请求是否真的允许共享？
- model、tokenizer、RoPE、LoRA adapter 不同时还能否命中？
- 谁维护引用计数？
- 谁决定缓存准入和驱逐？
- 一个请求扩展前缀时，其他消费者能观察到什么？

这正是论文把 **Ownership** 单列为核心轴的原因。

### 3.3 Continuous Batching 让生命周期更加动态

连续批处理允许在每个 Decode iteration 动态加入或移除请求，提高 GPU 利用率，但同时意味着：

- 不同长度、优先级和生命周期的请求共享一个 KV block pool；
- 只要仍有请求、分支或共享前缀引用某个 block，它就不能被回收；
- Prefill、Decode、Eviction 与 Prefix Reuse 的时序开始互相影响。

因此 allocator 与 scheduler 不能再各自独立工作。

---

## 4. 六类技术响应

论文先按技术机制归纳近年的六条路线，再用四轴分类把它们统一起来。

| 路线 | 代表系统/方法 | 主要目标 | 新增系统义务 |
|---|---|---|---|
| P1：本地虚拟化与阶段感知调度 | vLLM、vAttention、Orca、Sarathi-Serve | 提高本地显存利用率与批处理效率 | allocator 与 scheduler 一致性 |
| P2：驱逐、稀疏化、量化、压缩 | H2O、StreamingLLM、KIVI、KVQuant、CacheGen | 减少 KV 容量或传输字节数 | 质量保持、kernel 兼容性 |
| P3：Prefill/Decode 分离 | DistServe、Splitwise、KVDirect | 隔离两阶段资源干扰 | KV handoff、完成通知和准入 |
| P4：跨请求 Prefix/Context 复用 | SGLang、LMCache、Preble、CacheBlend、MemServe | 避免重复 Prefill | 身份、命名、隔离和安全复用 |
| P5：KV Offload 与分层 | FlexGen、Mooncake、ShadowKV、HiCache | 用 DRAM/SSD 扩展容量 | 迁移、预取和 deadline 控制 |
| P6：池化或解耦内存 | TraCT、Beluga、CXL-SpecKV | 将机架级内存变成 KV 池 | 并发、所有权、一致性、争用和故障处理 |

这六条路线不是互斥类别。例如：

- Mooncake 同时包含 P/D 分离、共享存储和分层；
- CacheGen 同时涉及压缩和传输；
- CXL-SpecKV 同时涉及池化内存与推测式预取；
- SGLang 从本地 RadixAttention 延伸到 HiCache 后跨入分层/共享存储。

所以论文认为仅按“优化技术”分类不够，需要一套能描述组合系统的语义坐标。

---

## 5. 四轴分类法

一个系统可被表示为：

$$
(A, B, C, D)
$$

其中 $A$、$B$、$C$、$D$ 分别对应 Locality、Lifetime、Ownership 和 Substrate。一个系统可以在同一轴上占据多个位置。

### 5.1 Axis A：Locality——消费者从哪里获得 KV

| 等级 | 名称 | 含义 |
|---|---|---|
| A0 | Local-only | KV 在生成者或当前 owner 的本地 GPU/worker 上被消费 |
| A1 | Direct remote | 从 producer/owner 直接传给 consumer，通常是单一生产者—消费者路径 |
| A2 | Shared store | KV 位于共享命名空间或分布式对象存储中，合格 worker 可查询并拉取 |
| A3 | Shared memory | worker 通过共享地址空间、load/store 或 memory-mapped 方式访问 KV |

### 如何理解 A1、A2、A3 的差别

- **A1 关注一次 handoff**：例如 Prefill worker 生成 KV，然后发给指定 Decode worker。
- **A2 关注可寻址对象**：消费者先查“KV 在哪里”，然后从共享存储取回。
- **A3 关注共享地址空间**：消费者看到的是一段共享内存，而不是显式的远端对象传输。

Locality 最直接影响：

- 数据访问延迟；
- 网络或总线带宽；
- 关键路径上的远端等待；
- 可用总容量。

但 Locality 不是硬件名称。例如 A1 常由 RDMA 实现，却也可以走 TCP；A3 常由 CXL 实现，但定义重点是共享 load/store 语义，而不是必须叫 CXL。

### 5.2 Axis B：Lifetime——KV 希望活多久

| 等级 | 名称 | 含义 |
|---|---|---|
| B0 | Per-request | 只服务当前请求，请求完成即可释放 |
| B1 | Per-session | 跨同一用户会话的多轮请求保留 |
| B2 | Cross-session | 跨请求、会话、用户或工作负载复用 |
| B3 | Opportunistic | 容量允许时保留、压力下可丢弃；它是修饰符 |
| B4 | Durable/recoverable | 有明确恢复契约，可跨相关进程或节点故障恢复 |

两个容易误解的点：

1. **B3 不是与 B0/B1/B2 平行的作用域。**它描述的是保留强度，所以可以写成 B2+B3，即“跨会话复用，但只是机会式保留”。
2. **StreamingLLM 不是 B1。**它在单条 Decode 流中保留 attention sink 和滑动窗口，本质仍是请求级 KV 压缩，即 B0。

论文发现两个显著空白：

- 没有被调研系统把 **B1 会话级 KV**定义为完整契约；
- 没有被调研系统提供 **B4 可恢复 KV**契约。

这很反直觉：生产聊天服务显然会做会话粘滞和上下文保留，但公开论文通常没有清楚规定：

- 每个 session 的保留预算；
- 跨 session 的驱逐优先级；
- session 迁移语义；
- 租户隔离；
- 节点故障后的恢复保证。

### 5.3 Axis C：Ownership——谁对 KV 负责

| 等级 | 名称 | 含义 |
|---|---|---|
| C0 | Per-worker | 每个 worker 独立分配、命名和驱逐，无全局权威 |
| C1 | Coordinator | 中央组件决定放置、准入、驱逐或路由 |
| C2 | Distributed | worker/cache node 通过目录、哈希、lease 或 P2P 协议协同 |
| C3 | Shared-memory discipline | KV 位于共享内存，由软件约定管理分配、写入和所有权 |

这是全文最强调的轴。

- C0 简单、局部开销低，但跨节点优化能力弱；
- C1 容易实现全局策略，却可能成为扩展性和可用性瓶颈；
- C2 消除单一控制点，但引入目录、锁、租约、同步和一致性成本；
- C3 数据路径看起来很直接，但并发分配、回收、失效和故障语义更难验证。

论文的核心判断是：

> 硬件常常决定 Locality 与 Substrate，工作负载常常决定 Lifetime；在这些条件相近时，系统设计者真正拥有较大自由度的地方是 Ownership。

### 5.4 Axis D：Substrate——KV 通过什么路径移动或暴露

| 等级 | 介质 | 典型用途 |
|---|---|---|
| D-local | GPU HBM / 本地设备内存 | Active KV 的基础驻留层 |
| D0 | NVLink / NVSwitch | 节点内 GPU 间传输、P2P 或 collective |
| D1 | RDMA / GPUDirect RDMA | 低延迟跨节点 KV 传输 |
| D2 | NIXL / UCX 等传输抽象层 | 屏蔽 RDMA、TCP、共享内存、GPU copy 等差异 |
| D3 | CXL 类共享内存 | 内存扩展或共享 KV pool |
| D4 | PCIe 后的 Host DRAM | KV offload 和 staging |
| D5 | SSD / NVMe / 存储系统 | 冷 KV 或大容量复用层 |
| D6 | SmartNIC / DPU / FPGA | 传输、索引、压缩、预取或近数据处理 |

论文强调 D2 是“抽象层”，不是独立物理介质。NIXL 的实际性能仍由底层 RDMA、TCP、共享内存或存储路径决定。

### 典型层次的数量级差异

| 层次/路径 | 容量 | 带宽 | 访问特点 |
|---|---:|---:|---|
| GPU HBM | 约 40–192 GB/GPU | 约 1–5 TB/s | Kernel 直接消费 |
| NVLink/NVSwitch peer GPU | 节点级 GPU 内存 | 单 GPU 数百 GB/s，聚合可达 TB/s | 拓扑相关，适合节点内搬运 |
| RDMA 远端 GPU/DRAM | 集群级 | 端点数十到数百 GB/s | 微秒级，受软件路径影响 |
| Host DRAM | 数百 GB 到 TB/节点 | CPU 侧数十到数百 GB/s | GPU 访问通常需要 PCIe/DMA |
| CXL 内存 | 设备级到机架级 | 数十到数百 GB/s | load/store，但存在 NUMA、争用和一致性问题 |
| SSD/NVMe | TB 到 PB | GB/s 级 | 容量大，但需显式 staging |

这些只是代表范围，不能脱离具体 GPU、NUMA、拓扑、操作大小和软件栈直接比较。

### 5.5 为什么 Granularity 和 Scheduling Objective 不是独立轴

论文把“迁移粒度”“命名粒度”“调度目标”等视为四轴推导出的性质，而不是额外轴。

例如：

- A1 + D1 往往导出 RDMA push/pull；
- A2 往往导出“目录查询 + 远端 fetch”；
- B0 + C0 可以只用请求本地 block table；
- B2 + C2 必须拥有跨请求稳定的 prefix/block/segment 名称；
- C1 的 P/D 分离系统倾向优化全局 goodput；
- C0 的本地系统倾向优化 batching、admission 与显存压力。

这个处理让分类维度保持精简，但也带来一个问题：粒度并不总能完全由四轴决定，后文会进一步讨论。

---

## 6. 五种架构原型

| 原型 | 典型四轴签名 | Anchor 系统 |
|---|---|---|
| Local-paged | A0，B0–B2[+B3]，C0，D-local[+D0] | vLLM、Sarathi-Serve |
| Disaggregated-pipeline | A1，B0，C1/C2，D-local+D1 | DistServe、Splitwise |
| Shared-store | A2，B2[+B3]，C1/C2，D-local+[D1/D4/D5] | LMCache、Preble |
| Memory-pool | A3，B0/B2[+B3]，C3，D-local+D3[+D6] | TraCT |
| Hybrid-tier | A0/A1/A2，B0/B2[+B3]，C0/C1，D-local+[D1/D4/D5] | Mooncake、ShadowKV |

这些签名是“架构包络”，并不表示每个成员都严格等于同一个 tuple。

### 6.1 Local-paged：在本地把 KV 管好

### 定义

KV 仍主要由单个 GPU、节点或本地 serving instance 管理，不建立跨节点共享 KV namespace。

### 主要手段

- PagedAttention block 管理；
- Continuous batching；
- Chunked prefill；
- 虚拟内存；
- 本地 prefix reuse；
- KV eviction、sparsification、quantization、compression。

### 代表系统

- vLLM；
- Sarathi-Serve；
- Orca；
- SGLang local mode；
- vAttention；
- H2O、StreamingLLM、SnapKV、PyramidKV、KIVI、KVQuant、ChunkKV。

### 优点

- 数据路径最短；
- 控制面简单；
- 不需要远端一致性；
- 易于复用成熟推理引擎。

### 局限

- 容量受本地 HBM/DRAM 限制；
- 数据并行副本间容易重复计算；
- 跨实例迁移或共享能力弱；
- 热点前缀可能导致负载倾斜。

### 适用场景

- KV 不需要跨 worker；
- 单机或小规模部署；
- 上下文和并发可被本地显存承受；
- 首要问题是碎片、调度或表示压缩。

### 6.2 Disaggregated-pipeline：把 Prefill 与 Decode 拆开

### 定义

KV 从专门的 Prefill worker 直接传给 Decode worker，并通常保持请求级生命周期 B0。

### 代表系统

- DistServe；
- Splitwise；
- KVDirect；
- P/D-Serve；
- LoongServe 的相邻设计；
- SPAD 等专用硬件方向。

### 为什么要拆分

Prefill 与 Decode 的资源偏好不同：

- Prefill 更偏计算密集、追求大 batch 和矩阵乘利用率；
- Decode 更偏内存带宽和低延迟调度。

混部时两者容易互相干扰；拆分后可以独立配置实例比例、并行策略和 admission policy。

### 核心代价：KV handoff

有两种主要方式：

1. **Prefill 完成后整体传输**：消息少、语义简单，但 Decode worker 等待时间长；
2. **按 layer 或 chunk 流水传输**：可以重叠计算与通信，但增加消息数量、元数据和同步点。

论文认为，现有工作通常报告 TTFT/goodput，却很少单独测量 handoff 粒度带来的元数据和完成通知开销。

### 6.3 Shared-store：KV 成为可复用对象

### 定义

KV 进入全局或半全局命名空间，可跨请求复用，典型生命周期是 B2+B3。

### 代表系统

- LMCache；
- Preble；
- SGLang HiCache；
- CacheBlend；
- MemServe；
- Mooncake 的 store 子系统。

### 必须解决的问题

- KV 如何命名；
- prefix/block 在哪里；
- 谁准入；
- 谁驱逐；
- 多副本如何维护；
- 什么条件下允许复用；
- 如何通知 block 已经可读；
- model/tokenizer/adapter/tenant 边界如何进入 key。

### 本质矛盾

长期保留可以提高命中率，但会同步放大：

- 目录规模；
- prefix lookup 成本；
- 元数据内存；
- location tracking；
- 引用计数；
- 驱逐复杂度。

因此“缓存越多越好”并不成立，关键是复用收益是否足以覆盖查询和传输成本。

### 6.4 Memory-pool：KV 进入共享地址空间

### 定义

KV 位于物理共享内存池中，通常由 CXL 暴露，worker 通过 load/store、DMA 或 memory-mapped 操作访问。

### 代表系统

- TraCT；
- Beluga；
- CXL-SpecKV；
- MemServe 的弹性内存池组件。

### 与 Shared-store 的区别

Shared-store 更像“对象系统”：

$$
\text{key lookup} \rightarrow \text{locate object} \rightarrow \text{fetch}
$$

Memory-pool 更像“共享内存系统”：

$$
\text{shared address} \rightarrow \text{load/store or DMA}
$$

但共享地址空间并不会自动解决：

- 分配；
- 所有权；
- 写入纪律；
- 失效；
- 回收；
- 故障处理；
- 带宽争用。

也就是说，CXL 改变了数据路径，却没有消灭控制面。

### 6.5 Hybrid-tier：把多种机制放进同一控制面

### 定义

Hybrid-tier 不是“用了多级内存”就算，而是系统的核心贡献在于：**用一个集成控制面协调两种以上原型的机制。**

### 代表系统

- Mooncake：P/D 分离 + 共享 KV Store + GPU/CPU/SSD 分层；
- ShadowKV：Host offload + sparse retrieval/reconstruction；
- FlexGen：GPU/CPU/disk offload 的历史前驱；
- Trinity：向量检索与 P/D placement 的边界案例。

### 最难的问题

- 跨层驱逐顺序；
- 哪些 KV 应复制、迁移或重算；
- P/D handoff 与 prefix reuse 如何竞争网络；
- 多种生命周期如何共存；
- Coordinator 是否成为瓶颈；
- HBM、DRAM、SSD 的成本模型如何统一。

Hybrid-tier 最贴近复杂生产系统，但也最难公平评估，因为多个优化会同时影响最终指标。

---

## 7. 代表系统的统一视图

| 系统 | Locality | Lifetime | Ownership | Substrate | 关键贡献 |
|---|---|---|---|---|---|
| vLLM | A0 | B0/B2+B3 | C0 | D-local | PagedAttention、连续批处理、block reuse |
| SGLang local | A0 | B2+B3 | C0 | D-local | RadixAttention、本地跨请求 prefix reuse |
| DistServe | A1 | B0 | C1 | D-local+D1 | P/D 分离、goodput 调度 |
| Splitwise | A1 | B0 | C1 | D-local+D1 | Prefill/Decode 集群分离与阶段感知配置 |
| KVDirect | A1 | B0 | C2 | D-local+D1 | P2P 分布式 KV 传输协议 |
| LMCache | A2 | B2+B3 | C1/C2 | D-local+D4+D5 | content-derived key、分布式 KV object store |
| Preble | A2 | B2+B3 | C1 | D-local+D1 | 集中式 prefix-aware placement 与调度 |
| Mooncake | A1+A2 | B0+B2+B3 | C1 | D-local+D1+D4+D5 | P/D 分离、KV Store、分层控制 |
| TraCT | A3 | B0+B2+B3 | C3 | D-local+D3 | 机架级 CXL 共享内存 KV |
| ShadowKV | A0 | B0 | C0 | D-local+D4 | CPU offload、稀疏检索与重建 |
| ForkKV | A0/A2 | B0/B2+B3 | C0/C2 | D-local+D1 | 分支/Agent KV 的 copy-on-write 语义 |

### 7.1 不能把这张表当成性能排行榜

这些工作使用了不同的：

- 模型与模型规模；
- GPU/NPU 代际；
- 网络；
- 上下文长度；
- Prefix 共享分布；
- 请求到达过程；
- SLO；
- baseline；
- 软件版本。

因此，某论文报告“2× goodput”，另一篇报告“5× throughput”，不能据此断定后者架构更优。

这张表用于回答的是：**系统在管理 KV 时做了什么语义选择**，而不是“谁跑分最高”。

---

## 8. 全文最重要的洞察：Ownership 是隐藏主变量

论文通过两组对照说明这一点。

### 8.1 DistServe 与 KVDirect

两者大致保持以下条件相同：

- Locality：A1，Prefill 到 Decode 的直接远端传输；
- Lifetime：B0，请求级；
- Substrate：D-local+D1，本地 HBM 加 RDMA 类网络。

主要差异在 Ownership：

- DistServe：C1，集中式 goodput-optimized scheduler；
- KVDirect：C2，peer-to-peer 分布式传输和控制。

这意味着它们真正不同的不是“是否用 RDMA”，而是**谁决定传输、谁维护状态、如何协调并发**。

### 8.2 Mooncake 与 LMCache

两者都具有：

- A2 共享存储语义；
- B2+B3 跨请求机会式复用。

差异主要是：

- Mooncake：由集中式 Conductor 管理放置与路由，偏 C1；
- LMCache：使用 content-derived chunk keys，可接集中式或分布式 backend，覆盖 C1/C2。

它们付出的扩展性、元数据与容错成本因此不同。

### 8.3 论文提出的可证伪预测

作者没有只停留在“Ownership 很重要”的定性判断，而是提出一个可测试预测：

> 在吞吐与工作负载匹配时，C1 集中式 ownership 系统的尾延迟方差应低于 C2 分布式 ownership 系统，因为 C1 避免了关键路径上的分布式协调停顿，只承担相对固定的调度成本。

对应地：

- C1 的 P99 TTFT 应主要由排队和 Prefill 计算构成；
- C2 还可能出现目录查询、分布式锁竞争、同步或协议开销。

适合验证的实验对包括：

- DistServe vs KVDirect；
- Mooncake vs LMCache。

但当前论文没有足够数据验证该预测，因为现有系统缺少 P99 分解。这正好引出论文的 measurement-gap audit。

需要注意：这只是**待验证假设**，不是已被实验证明的结论。如果匹配条件下 C1/C2 的尾延迟分解无显著差异，就需要降低 Ownership 在分类中的权重，或加入新的混杂变量。

---

## 9. 为什么 TTFT、TPOT 和 Goodput 不够

### 9.1 TTFT 混合了太多因素

TTFT 同时包含：

- 排队；
- Prefix lookup；
- KV 分配；
- Prefill 计算；
- KV transfer；
- 完成通知；
- Decode worker readiness。

因此 TTFT 下降并不能说明是：

- 命中率提高；
- 调度更合理；
- 传输更快；
- 元数据路径更短；
- Prefill kernel 更快。

### 9.2 TPOT 同样不是纯 Decode 指标

TPOT 可能混合：

- HBM 带宽；
- 远端 KV fetch；
- CPU/SSD prefetch miss；
- 调度延迟；
- 同步等待；
- Offload 恢复。

### 9.3 Goodput 仍然无法归因

Goodput 通常表示满足延迟 SLO 的请求吞吐量，例如：

$$
\mathrm{TTFT} < t_1, \qquad \mathrm{TPOT} < t_2
$$

它把吞吐与延迟结合起来，比单看 QPS 更有意义，但仍严重依赖：

- 具体 workload；
- SLO 阈值；
- admission policy；
- batch size；
- prefix 命中率；
- 传输和预取策略。

所以 Goodput 适合判断“系统在这个条件下是否更好”，不适合解释“哪个 KV 机制为什么更好”。

---

## 10. 七个缺失的 KV 专用测量

这是论文最具实践价值的部分之一。

| 编号 | 缺失测量 | 应报告什么 |
|---|---|---|
| MG1 | Remote KV access patterns | 操作大小、每 token 远端 block 数、每 token 字节数、关键路径 stall |
| MG2 | Metadata path costs | prefix/block lookup、owner tracking、refcount、目录、驱逐决策的 CPU/内存/延迟开销 |
| MG3 | Completion/synchronization overhead | block/layer/request 粒度完成通知的延迟与同步成本 |
| MG4 | Lifetime/reuse distance | block 存活时间、两次复用之间的间隔分布 |
| MG5 | Prefetchability slack | 可预测访问出现到实际 deadline 之间的可预取窗口 |
| MG6 | P99 tail attribution | 将 P99 TTFT/TPOT 分解为排队、计算、传输、元数据和同步 |
| MG7 | Public KV-aware traces | 请求、长度、prefix/session、block access、transfer event 的公开 trace |

### 10.1 MG1：远端 KV 访问模式

仅报告“RDMA 带宽”或“平均传输时延”不够。系统真正需要知道：

- 每次操作多大；
- 每 token 访问多少远端 block；
- 是否存在大量小操作；
- 哪些访问落在关键路径；
- 网络拥塞时 stall 如何分布。

同样的平均带宽，可能对应完全不同的尾延迟。

### 10.2 MG2：元数据路径成本

需要测量：

- radix tree/prefix index 查询；
- block location directory；
- 引用计数；
- copy-on-write bookkeeping；
- owner/lease 更新；
- 驱逐候选计算；
- completion notification。

论文给出一个粗略估算：若 radix tree 有 $N$ 个 active prefix，查询需要 $O(\log N)$ 次 pointer chase，每次 LLC miss 约 100 ns；当 $N=10^4$ 时，一次查询约 1.3 µs。在 $10^5$ req/s 下，仅查询就会消耗约每秒 130 ms 的单核时间，尚未计入其余元数据操作。

这个估算并非严格性能模型，但足以说明：**元数据不是可以永远忽略的常数。**

### 10.3 MG3：完成与同步成本

远端 KV 何时对消费者可见？粒度不同，代价不同：

- 请求级：同步点少，但消费者必须等完整请求 KV 传完；
- Layer 级：能流水重叠，但同步点随层数 $L$ 增长；
- Block 级：最灵活，但消息与元数据可能很多。

现有论文很少同时报告这三种粒度的平均成本、方差与正确性约束。

### 10.4 MG4：生命周期与复用距离

Cache hit rate 只是结果，不足以指导策略。

系统真正需要知道：

- 一个 block 多久后会再次被访问；
- 复用距离分布是否重尾；
- 热度是否稳定；
- 跨 session 的间隔多长；
- 冷却后是否值得放到 DRAM/SSD；
- 哪些 block 值得复制。

没有这些数据，LRU、LFU 或成本感知驱逐通常只能靠经验调参。

### 10.5 MG5：可预取余量

Transformer 的 layer 顺序和 token 顺序具有一定可预测性。关键变量不是“能否预测”，而是：

$$
\mathrm{Slack} = t_{\mathrm{deadline}} - t_{\mathrm{predictable}}
$$

如果从 Host DRAM 拉取 KV 需要的时间小于 Slack，预取可以隐藏延迟；否则即便预测完全正确，也仍会阻塞关键路径。

### 10.6 MG6：P99 尾延迟归因

论文希望 P99 被拆成：

$$
T_{\mathrm{P99}} = T_{\mathrm{queue}} + T_{\mathrm{compute}} + T_{\mathrm{transfer}} + T_{\mathrm{metadata}} + T_{\mathrm{sync}}
$$

严格来说，分位数不能简单线性相加，上式更适合作为逐请求关键路径分解的概念表达；实际评估应记录每个请求的各阶段时间，再分别分析其对 P99 样本的贡献。

### 10.7 MG7：公开 KV-aware Trace

理想 trace 应包括：

- 请求到达时间；
- prompt/output 长度；
- prefix sharing identity；
- session 边界；
- KV block access；
- KV transfer event；
- 可能还应包括 model/adapter/tenant 与 SLO。

已有 Azure 与 Mooncake trace 改善了公开数据条件，但仍缺少足够细的 block、prefix、session 和传输事件信息，难以完成机制归因。

---

## 11. 七个开放设计问题

| 编号 | 设计问题 | 依赖测量 |
|---|---|---|
| DG1 | 分布式 KV 容错 | MG3 + MG4 |
| DG2 | 多租户隔离 | MG6 + MG7 |
| DG3 | 跨复制层级的驱逐 | MG4 + MG5 |
| DG4 | Speculative Decoding × KV | MG1 + MG3 |
| DG5 | MoE × KV | MG1 + MG2 |
| DG6 | 共享 Prefix Cache 的所有权/一致性语义 | MG2 |
| DG7 | Session 级 KV 管理 | MG4 + MG7 |

### 11.1 DG1：分布式 KV 如何容错

当远端 KV Store 在 Decode 中途失败时，系统有三类选择：

1. 从原始 prompt 重新 Prefill；
2. 从其他 tier 或副本恢复；
3. 中止请求。

KV 不等同于普通缓存对象，它的有效性依赖：

- 模型版本；
- tokenizer；
- position encoding；
- adapter；
- attention layout；
- 精度与压缩格式。

因此 B4“可恢复 KV”需要完整的版本、校验、可见性和恢复契约，而不仅是多复制一份数据。

### 11.2 DG2：多租户隔离与时间侧信道

即便不允许租户直接读取他人的 KV，prefix cache hit 也可能通过更低 TTFT 暴露“其他租户是否用过相同前缀”。

潜在防护包括：

- namespace 隔离；
- 准入规则；
- 延迟填充；
- 随机噪声；
- 租户内共享、租户间禁用；
- 对敏感 prompt 禁止缓存。

但防护会损失复用收益，需要通过 tail attribution 与多租户 trace 评估。

### 11.3 DG3：跨 HBM、DRAM、远端 Store 的联合驱逐

同一个 block 可能同时存在于：

- HBM；
- Host DRAM；
- 远端 DRAM；
- SSD。

驱逐不再是“删或不删”，而是：

- 保留副本；
- 降级到更慢层；
- 提升到更快层；
- 删除；
- 未来重算。

合理策略应比较：

$$
\text{Expected Reuse Benefit}
\quad \text{vs.} \quad
\text{Capacity Cost}+
\text{Fetch Cost}+
\text{Replication Cost}
$$

而这依赖真实 reuse distance 与 prefetch slack。

### 11.4 DG4：Speculative Decoding 下 KV 不再线性追加

推测解码会先产生 tentative KV，再接受部分 token、拒绝部分 token。分布式系统必须决定：

- 未验证 KV 是否立即传输；
- rollback 如何传播；
- 分支 KV 是否共享；
- 何时回收 tentative block；
- RDMA 传输与 CXL 共享内存下的回滚代价有何差异。

现有线性 append 假设不再成立。

### 11.5 DG5：MoE 流量与 KV 流量如何协同

标准 MoE 中，KV 通常仍随 Attention 层布局，而不是随 Expert 放置。但 Expert all-to-all 与 KV transfer/prefetch 会争用同一网络。

因此问题可能不是“把 KV 跟随专家放置”，而是：

- 是否联合调度 Expert routing 与 KV prefetch；
- 是否需要拥塞感知；
- 如何区分控制面和数据面拥塞；
- 哪类流量应优先保障尾延迟。

### 11.6 DG6：共享 Prefix Cache 的一致性语义

共享 Prefix Cache 常用：

- refcount；
- copy-on-write；
- append-only；
- location directory。

但很多论文没有明确回答：当请求 A 正在引用前缀 $P$，请求 B 扩展、迁移或驱逐 $P$ 时，A 可以观察到什么？

需要定义至少以下语义：

- block 何时可见；
- 驱逐何时生效；
- owner 迁移时旧地址何时失效；
- 读者如何 pin；
- 失败后目录和数据如何恢复一致。

### 11.7 DG7：真正的 Session 级 KV 管理

一个完整 B1 系统至少应规定：

- 每个 session 的 KV 预算；
- 活跃 session 与休眠 session 的优先级；
- turn 间隔较长时放到哪一层；
- session 迁移时如何搬运；
- 多租户 session 的隔离；
- session 终止与回收语义。

论文认为这是公开研究中的明显空白。

---

## 12. 对 LLM Serving 系统设计的实践启示

### 12.1 先判断 KV 是否真的需要跨边界

如果 workload：

- 上下文不长；
- Prefix 复用低；
- 单机容量足够；
- 网络较慢；
- 模型吞吐主要受计算限制；

那么 Local-paged 很可能是更合理的基线。引入远端 KV Store 会增加复杂度，却不一定产生收益。

### 12.2 P/D 分离主要解决阶段干扰，不等于 Prefix Cache

Disaggregated-pipeline 的核心是：

- 隔离 Prefill 和 Decode；
- 独立扩缩容；
- 将 KV 从 P worker handoff 给 D worker。

如果 KV 请求完成后释放，它仍是 B0，并没有实现跨请求复用。

### 12.3 Prefix-aware Routing 不等于 Shared KV Store

路由器可以通过“把相同前缀送到同一 worker”提高本地命中，但 KV 数据仍可能只存在各 worker 本地。

需要区分：

- **逻辑共享/全局可发现**：调度器知道哪些 worker 缓存了哪些 prefix；
- **物理共享存储**：任意 worker 能通过统一命名空间 fetch KV；
- **共享地址空间**：worker 直接访问共同内存池。

这三者对应的故障、元数据和带宽模型完全不同。

### 12.4 Offload 的正确比较对象包括“重算”

当远端层很慢时，不一定值得取回 KV。调度器应比较：

$$
T_{\mathrm{fetch}} + T_{\mathrm{stage}} + T_{\mathrm{sync}}
\quad \text{与} \quad
T_{\mathrm{recompute}}
$$

如果重算未命中 prefix 更快，就不应盲目追求 cache hit。

进一步还要比较重算对 GPU queue 和其他请求的干扰。

### 12.5 命中率不是最终目标

高命中率可能伴随：

- 热点 worker；
- 大量远端小块访问；
- 高目录查询开销；
- 低价值 KV 长期占用容量；
- 跨租户安全风险。

真正目标应是 SLO 下的成本与吞吐，例如：

$$
\max \; \text{Goodput per Dollar}
$$

同时对 TTFT、TPOT、失败率、网络与内存成本施加约束。

### 12.6 Naming、Readiness、Eviction、Completion 是跨架构公共原语

论文观察到，虽然数据路径不同，但只要 KV 跨 locality 或 lifetime 边界，系统都会需要：

1. **Naming**：这块 KV 对应什么内容与执行环境；
2. **Readiness**：它是否已经生成并可读取；
3. **Eviction**：何时、从哪一层、由谁驱逐；
4. **Completion**：生产、传输、消费完成如何通知。

这些原语可能比某个特定 RDMA/CXL 数据路径更具复用价值。

---

## 13. 如何用四轴分析一个新系统

论文给出的操作流程可以整理为以下五步。

### 第一步：找出受管理的 KV 对象

它是：

- request tensor；
- page/block；
- layer slice；
- prefix；
- semantic chunk；
- speculative branch；
- session state？

### 第二步：确认系统主要改变了什么

- 放置；
- 生命周期；
- 所有权；
- 介质；
- 表示；
- 调度依赖。

### 第三步：分配主 tuple

依次回答：

- KV 从哪里获得？
- 保留到什么时候？
- 谁负责管理？
- 通过什么路径访问？

### 第四步：记录复合模式

如果系统同时存在本地与远端模式，或请求级 handoff 与跨请求 store，应记录多个轴值，而不是强行压成单点。

### 第五步：判断是不是边界系统

如果系统只提供一种传输、压缩或分支机制，却没有完整定义 ownership 与 lifetime，它更适合标为 boundary，而非独立 archetype。

### 示例：Infinite-LLM

论文将其分析为：

- KV 对象：分布式 Attention 中每个 worker 的 KV slice；
- Locality：A2，KV 通过集群级逻辑 pool 被寻址；
- Lifetime：B0，只服务原请求；
- Ownership：C2，目录与路由协议分布式管理；
- Substrate：D-local+D1；
- 角色：靠近 Shared-store，但生命周期更像 Disaggregated-pipeline，因此属于边界案例。

---

## 14. 与常见 Prefix-aware Routing 工作的关系

这篇综述的主线是 KV 内存语义，不是路由算法，所以对路由工作的覆盖并不完整。但四轴可以帮助澄清 Preble、DualMap、CacheRoute 等系统处于哪一层。

### 14.1 Preble

Preble 维护全局 prefix 视图，根据缓存亲和与负载做 placement。论文将其归入 Shared-store，标为 A2、B2+B3、C1。

这个分类强调的是“全局 prefix namespace 与可复用放置”，但应注意：Preble 的 KV 数据主要仍驻留在各 GPU 本地，并不完全等价于 LMCache 那种独立 KV object store。因此把 Preble 归入 A2 是一种偏**逻辑共享命名空间**的解释，而不是说它存在物理集中式共享存储。

### 14.2 DualMap

DualMap 通过两个确定性哈希候选，在有限副本化的 prefix affinity 与负载均衡之间折中。用本文四轴看，其主要特征更接近：

- Locality：数据面通常仍是 A0；
- Lifetime：B2+B3，本地 prefix KV 跨请求机会式复用；
- Ownership：全局 scheduler 为 C1，但 prefix 的候选 owner 由哈希确定；
- Substrate：主要是 D-local，具体实现可叠加远端层。

这说明**路由所有权**和**数据存储 locality**可能不完全绑定：中央路由器可以管理分布式本地缓存，而不需要真正的共享 KV Store。

### 14.3 CacheRoute

CacheRoute 类系统通常把缓存亲和、实例负载和副本规划结合起来，更像在 C1 控制面上做 KV placement planning。其实际 A/B/D 取值取决于：

- KV 是否只留在后端本地；
- 是否允许远端拉取；
- 是否跨 session 复用；
- 是否有多级缓存。

因此，不能仅因系统“KV-aware”就自动归类为 Shared-store。

### 14.4 对路由研究的启示

只比较：

- cache hit rate；
- load balance；
- TTFT；

还不够。建议同时报告：

- prefix directory 更新成本；
- 路由决策延迟；
- stale cache view 比例；
- 重复 KV 副本数；
- worker 失效后的 remapping/re-prefill 成本；
- P99 中 queue、recompute、lookup 和 transfer 的贡献。

---

## 15. 论文的主要贡献

### 15.1 将 KV Cache 从“优化对象”提升为“系统对象”

论文不是再列一遍压缩、驱逐、Offload 技术，而是提出：KV 应像分布式存储对象一样被描述，具有：

- 位置；
- 生命周期；
- 所有者；
- 访问介质；
- 名字；
- 可见性；
- 恢复与隔离语义。

这是一个有用的认知升级。

### 15.2 四轴比单纯按技术机制分类更适合复合系统

Mooncake、SGLang HiCache、MemServe 等系统同时包含多种机制。只按“Offload”“P/D 分离”“Prefix Cache”分类会重复或混淆。四轴 tuple 能显式表达组合。

### 15.3 用五种原型压缩了复杂设计空间

作者把大约三十多个系统归纳成五个架构包络，使读者能先理解结构，再学习具体系统。

### 15.4 强调 Ownership，而非只盯硬件和带宽

很多讨论只关注 RDMA、CXL、HBM、SSD，却忽略：

- 谁做目录；
- 谁做 placement；
- 谁做 eviction；
- 谁承担一致性；
- 谁在故障时恢复。

论文把控制面重新放到核心位置。

### 15.5 Measurement-gap audit 很可执行

七个 MG 不只是泛泛地说“需要更多实验”，而是给出了应该采集的具体指标，并与七个设计问题一一对应。

---

## 16. 局限与值得质疑之处

### 16.1 这不是可复现实验型综述

论文没有构建统一 benchmark 对五类系统做横向实验，也没有用同一模型、硬件和 trace 验证分类产生的性能预测。

所以它能回答：

- 系统在语义上有什么不同；
- 现有评测缺什么；

但不能回答：

- 哪个 archetype 在某 workload 下最好；
- C1 是否真的比 C2 尾延迟更稳定；
- CXL memory-pool 是否优于 RDMA shared-store。

### 16.2 五类原型是定性包络，不是聚类结果

作者明确说明，五类原型是观察到的 recurring envelope，并非统计聚类，也不是穷尽证明。

因此未来可能出现：

- 独立的 session-state archetype；
- 面向 speculative branch 的版本化 KV store；
- MoE 与 KV 联合放置系统；
- durable KV service；
- 完全 serverless 的 KV ownership 模型。

### 16.3 Locality 与 Ownership 并非完全正交

作者认为四轴通过了正交性测试，但实际中存在明显相关性：

- D3 CXL 往往诱导 A3 与 C3；
- A0 通常对应 C0；
- A2 shared store 通常需要 C1/C2。

论文承认这种相关，但仍保留独立轴，因为三者表达不同后果。这个选择有解释力，但分类时会产生边界争议。

### 16.4 Preble 的 A2 分类存在讨论空间

Preble 的核心贡献是全局 prefix-aware scheduling，KV 数据通常驻留于被选 GPU 的本地 cache。把它归为 Shared-store/A2，实际上扩大了“共享存储”的定义，将全局可发现的分布式本地缓存也纳入其中。

如果严格按数据访问路径定义，Preble 可能更接近：

- 数据面 A0；
- 控制面拥有全局 prefix directory；
- C1 ownership。

这暴露了一个潜在缺口：**数据 locality 与目录/命名空间 locality 可能需要分别描述。**

### 16.5 Granularity 未必完全是派生属性

Token、block、layer、prefix、semantic chunk 的粒度会直接影响：

- 元数据规模；
- 传输放大；
- cache pollution；
- copy-on-write 成本；
- completion 次数；
- 可预取窗口。

即使两个系统的 $(A, B, C, D)$ 相同，粒度不同也可能导致数量级性能差异。因此 Granularity 也许值得作为第五轴，或者至少作为每个 tuple 的必填属性，而不只是“derived property”。

### 16.6 Consistency 被排除，但未来可能成为核心轴

论文解释：大多数系统没有明确 consistency contract，所以暂不作为轴。然而恰恰因为系统不报告一致性，它可能不是“不重要”，而是“尚未成熟”。

随着以下场景增多：

- 跨请求共享；
- copy-on-write；
- speculative branch；
- 多副本；
- durable recovery；

一致性模型可能会从隐含实现细节变成一等语义。

### 16.7 调研边界会遗漏“路由改变 KV 行为”的工作

论文排除了通用调度，但实际上一些 prefix-aware router 即使不搬运 KV，也会显著改变：

- KV 的物理放置；
- 副本数量；
- 生命周期；
- 命中率；
- 热点与驱逐。

DualMap、CacheRoute 及大量推理网关中的 prefix-aware routing 可以补充这部分视角。

### 16.8 硬件数据只能作为代表值

HBM、RDMA、CXL、SSD 的带宽与延迟高度依赖：

- 硬件代际；
- PCIe/NVLink 拓扑；
- NUMA；
- 操作大小；
- 是否 GPUDirect；
- 软件栈；
- 并发与拥塞。

因此论文中的层次对比适合建立量级直觉，不适合直接代入容量规划。

---

## 17. 推荐的统一评测框架

基于论文的 measurement gaps，可以整理出一套更完整的评测清单。

### 17.1 工作负载维度

- Prompt length 分布；
- Output length 分布；
- 到达过程与 burstiness；
- Prefix sharing ratio；
- Prefix 热度分布；
- Session 间隔与轮数；
- Tenant 数及跨租户共享条件；
- Model/adapter 混部情况；
- SLO 分类。

### 17.2 容量指标

- HBM/DRAM/SSD 分层占用；
- 逻辑 KV bytes 与物理 KV bytes；
- 分页和碎片开销；
- 元数据/KV 数据比；
- 副本放大率；
- 每层驱逐量。

### 17.3 数据路径指标

- 每请求、每 token 远端 bytes；
- 操作大小分布；
- KV fetch/push 次数；
- RDMA/CXL/PCIe 带宽利用率；
- 传输与计算 overlap；
- critical-path stall；
- re-prefill bytes/tokens。

### 17.4 控制路径指标

- Prefix lookup latency；
- Directory lookup/update latency；
- Refcount/COW 成本；
- Placement decision latency；
- Eviction decision latency；
- Completion notification 次数与成本；
- stale metadata 比例；
- Coordinator CPU 和内存使用率。

### 17.5 缓存质量指标

- Token-weighted hit rate；
- Request hit rate；
- Prefix match length 分布；
- Reuse distance；
- KV lifetime；
- 重复副本数；
- Eviction 后短期再访问率；
- Fetch 与 recompute 的胜率。

### 17.6 端到端指标及归因

- TTFT P50/P95/P99；
- TPOT P50/P95/P99；
- Throughput 与 Goodput；
- SLO attainment；
- 每请求成本；
- P99 样本中的 queue/compute/transfer/metadata/sync 分解。

### 17.7 可靠性与安全指标

- Worker/Store failure 后请求成功率；
- 恢复时间；
- Re-prefill 放大；
- 数据失效与错误复用率；
- 多租户隔离开销；
- Prefix hit 时间侧信道可区分度。

---

## 18. 读完后如何选择架构

可以按以下决策树理解。

### 情况一：KV 不需要跨 worker

优先从 Local-paged 开始：

- PagedAttention；
- Continuous batching；
- Chunked prefill；
- 本地 Prefix Cache；
- 必要时做量化/压缩/驱逐。

### 情况二：主要问题是 Prefill 与 Decode 干扰

考虑 Disaggregated-pipeline，并重点验证：

$$
T_{\mathrm{handoff}} < T_{\mathrm{phase\ interference\ saved}}
$$

同时测量 handoff 粒度、网络占用和 completion overhead。

### 情况三：存在大量跨请求重复 Prefix/RAG Context

考虑 Shared-store 或全局 prefix-aware placement，但先确认：

$$
P_{\mathrm{reuse}} \cdot C_{\mathrm{recompute\ saved}}
>
C_{\mathrm{lookup}} + C_{\mathrm{transfer}} + C_{\mathrm{metadata}} + C_{\mathrm{capacity}}
$$

### 情况四：总 KV 工作集远大于 HBM

考虑 Host DRAM、远端内存、SSD 或 Hybrid-tier，并同时比较 fetch 与 recompute。

### 情况五：拥有 CXL 类共享内存基础设施

可以考虑 Memory-pool，但不能只测内存访问延迟，还要评估：

- pool allocator；
- 热点和带宽争用；
- software coherence；
- ownership；
- failure domain。

### 情况六：生产系统同时具备多种需求

最终往往走向 Hybrid-tier，但应避免一步到位构建过度复杂系统。更合理的方法是：

1. 建立 Local-paged 基线；
2. 识别瓶颈属于容量、复用还是阶段干扰；
3. 只跨越必要的 locality/lifetime 边界；
4. 为新增边界补齐 naming、readiness、eviction、completion；
5. 用机制级指标验证收益来源。

---

## 19. 最值得记住的结论

1. **KV Cache 的角色已经变化。**它从请求私有 Tensor 变成了分布式内存对象。
2. **容量和带宽同样重要。**容量随 $S$ 线性增长，累计 Decode 读取量随 $TS+T^2$ 增长。
3. **四个核心问题是：在哪里、活多久、谁负责、怎么访问。**
4. **五种原型不是替代关系。**Local-paged、P/D 分离、Shared-store、Memory-pool 与 Hybrid-tier 会长期共存。
5. **Ownership 是分布式 KV 系统的隐藏主变量。**相同 RDMA/CXL 硬件上，不同 owner 模型会产生完全不同的扩展性、尾延迟和容错代价。
6. **Cache hit rate 不是最终目标。**需要把 lookup、transfer、metadata、queue 和 recompute 一起纳入成本。
7. **现有评测最大问题是无法归因。**TTFT、TPOT、Goodput 只能说明结果，不能说明机制。
8. **最优先补齐的测量是 Completion、Reuse Distance 和公开 KV-aware Trace。**
9. **B1 与 B4 是两个醒目的空白。**公开系统尚未完整定义 session-level retention 和 durable/recoverable KV。
10. **未来重点会从数据搬运走向语义。**包括一致性、隔离、版本、分支、恢复和多租户共享。

---

## 20. 总体评价

这篇论文的价值不在于提出一个新的高性能 KV Cache 实现，而在于给快速膨胀的研究领域提供一套比较稳定的语言。

它最成功的部分有三点：

- 用四轴把本地分页、P/D 分离、Prefix Cache、Offload、CXL Pool 放进同一个坐标系；
- 指出 Ownership 是现有讨论中被硬件指标掩盖的重要差异；
- 把“评测不完整”具体化为七个可采集的 KV 专用指标。

它的不足也很明确：

- 没有统一实验；
- 原型是定性归纳；
- Preble 等边界系统的分类存在争议；
- Granularity、Consistency 与目录拓扑可能值得成为更显式的维度；
- 对 prefix-aware routing 的覆盖不够完整。

因此，最合适的阅读方式不是把它当成“KV Cache 技术排行榜”，而是把它当成：

1. **系统设计检查表**：设计新系统时逐项回答四个轴；
2. **论文阅读坐标系**：判断不同论文究竟改变了哪层语义；
3. **实验设计模板**：除 TTFT/TPOT 外补齐七类机制指标；
4. **研究选题地图**：从 B1、B4、共享缓存一致性、分支 KV、MoE 竞争和跨层驱逐中寻找空白。

从这个角度看，标题“From Tensor Buffer to Distributed Memory Hierarchy”准确概括了领域演进：**真正的变化不是 KV 变大了，而是 KV 开始拥有分布式系统意义上的身份、位置、生命周期、所有者和故障语义。**

