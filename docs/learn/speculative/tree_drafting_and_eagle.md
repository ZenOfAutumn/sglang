# 树形草稿（Tree Drafting）与 EAGLE 原理详解

EAGLE 的全称是 **Extrapolation Algorithm for Greater Language-model Efficiency**（面向更高语言模型效率的外推算法）。

本文详细介绍树形草稿的动机、树的构建 / 摊平 / 树形注意力掩码 / 树形验证的完整机制，以及 EAGLE 系列（EAGLE-1/2/3）的原理与它们在 SGLang 中的实现。投机解码的基础流程、拒绝采样的无损性证明见同目录 [speculative_decoding.md](./speculative_decoding.md)。

## 1. 为什么从链到树

链式草稿一轮只押注**一条**路径：每步只保留 1 个候选，第 $i$ 个草稿被拒则后缀全部作废。设单 token 接受率为 $\alpha$，链长 $k$ 的期望前进长度 $\mathbb{E}[\tau] = \frac{1-\alpha^{k+1}}{1-\alpha}$ 随 $k$ 快速饱和——链加得再深，也救不了「第一个 token 就猜错」的情形。

树形草稿的思路：**同一笔验证预算，不押一条长链，而是押一棵树**——在分歧大的位置多留几个候选（宽度），在草稿模型有把握的位置往深走（深度）。目标模型一次 forward 验证整棵树上的所有节点，最后取树中**最长的可接受路径**。由于目标模型 decode 是 memory-bound，一次 forward 里多算几十个 token 位置的边际成本很低，树形验证近乎免费地把「一次押注」变成「多路押注」。

直观对比（验证预算同为 7 个 token）：

```
链式（k=7）:  x → a → b → c → d → e → f → g     一条路径，a 错全完

树形:              ┌─ b1 ─ c1
          x ─ a1 ──┤
              │    └─ b2
              ├─ a2 ─ b3
              └─ a3                              a1 错还有 a2/a3 兜底
```

## 2. 树的构建：逐层 top-k 扩展 + 全局分数剪枝

树的形状可以是**静态**的（离线固定，如 Medusa、EAGLE-1）或**动态**的（按草稿模型的置信度现场生长，EAGLE-2 起）。SGLang 实现的是动态树，构建过程由三个参数控制：

- `speculative_num_steps`：草稿自回归步数 = 树深；
- `speculative_eagle_topk`：每步每个活跃分支保留的候选数 = 分支因子；
- `speculative_num_draft_tokens`：最终送验的树节点总数（含根）。

### 2.1 逐层扩展

以上一轮的 bonus token 为**根**。第 1 步：草稿模型给出根的下一 token 分布，取 top-k 个候选作为第一层节点；之后每一步，对当前保留的 top-k 个节点各自展开 top-k 个孩子（候选池 $k \times k$ 个），再按**路径累积概率**（从根到该节点沿途草稿概率之积，作为节点分数）选出全局 top-k 个进入下一步。每个节点记录：

- token id；
- 分数 $s(v) = \prod_{u \in \text{path}(root \to v)} q(u)$；
- 父节点指针（`parents_list`）。

用路径累积概率而非单步概率做分数，是因为一个节点只有在**整条祖先链都被接受**时才有用——分数直接近似「这个节点最终被接受的概率」。

### 2.2 全局剪枝

$k$ 步扩展共产生 $O(\text{steps} \times \text{topk}^2)$ 个候选节点，通常多于验证预算。最后按分数对**所有步产生的全部候选**做一次全局 top-$(\text{num_draft_tokens} - 1)$ 筛选（`organize_draft_results`，`eagle_utils.py:71`），加上根节点凑成 `num_draft_tokens` 个送验节点。全局筛选的效果：草稿模型置信度高时树自动长成「深链」，置信度低时自动长成「宽树」——树形随上下文自适应，这正是 EAGLE-2 动态树的核心。

由于孩子的分数 = 父亲的分数 × 一个 $\le 1$ 的概率，孩子分数必不大于父亲，全局 top-N 选出的节点集合**自动构成一棵连通的树**（选了孩子必然已选了父亲），不需要额外的连通性修补。

## 3. 树的摊平与树形注意力掩码

树是逻辑结构，送入目标模型时必须摊平成一维序列。摊平带来两个问题，分别由自定义 mask 和自定义 position 解决（`build_tree_kernel_efficient`，`eagle_utils.py:107`）：

### 3.1 树形注意力掩码（tree mask）

标准因果 mask 意味着「序列里排在我前面的都能看」，但树上的节点摊平后，排在前面的可能是**兄弟分支**——它们不在我的祖先链上，看了就等于让两条互斥的假设互相污染。因此每个树节点的 attention 规则是：

> 能看到：全部已确认的上下文 + 自己在树中的**祖先路径**；看不到：任何兄弟/旁系分支。

```
树:        root ─┬─ a1 ─── b1
                 └─ a2

摊平序列:  [root, a1, a2, b1]

mask 行（1=可见）:     root  a1  a2  b1
        root 看          1    0   0   0
        a1   看          1    1   0   0
        a2   看          1    0   1   0     ← 看不到兄弟 a1
        b1   看          1    1   0   1     ← 看得到父亲 a1，看不到叔叔 a2
```

SGLang 中 mask 有三种物化模式（`TreeMaskMode`）：`FULL_MASK`（完整 [draft_token_num, seq_len + draft_token_num] 布尔阵）、`QLEN_ONLY`（只物化草稿 token 之间的方阵，上下文部分恒可见不物化）、`QLEN_ONLY_BITPACKING`（再按位压缩），供不同注意力后端选用。这要求注意力后端支持**自定义 mask 的 target-verify 模式**，是投机解码对 attention backend 的主要适配点。

### 3.2 位置编码

摊平序列的下标不能直接当 position：树上**同一深度的兄弟节点必须共享同一个 position**（都是「上下文长度 + 深度」），否则 RoPE 会把并列假设编码成先后关系。上例中 a1、a2 的 position 相同，b1 = a1 的 position + 1。

### 3.3 树结构的编码

验证后要在树上找最长接受路径，因此树结构本身也要传给验证 kernel。SGLang 用三个索引张量表达（`EagleVerifyInput`，`eagle_info.py`）：

| 字段 | 含义 |
| --- | --- |
| `retrieve_index` | 每个节点在摊平序列中的位置 |
| `retrieve_next_token` | 每个节点的第一个孩子（无则 -1） |
| `retrieve_next_sibling` | 每个节点的下一个兄弟（无则 -1） |

「第一孩子 + 下一兄弟」是经典的 left-child right-sibling 表示，遍历某节点的全部孩子 = 走一次 sibling 链表。

## 4. 树形验证与路径选择

目标模型对摊平序列做一次 forward 后，每个树节点位置都得到了目标分布 $p(\cdot \mid \text{该节点的祖先路径})$。接下来从根开始逐层向下走：

### 4.1 Greedy 模式（温度 0）

在当前节点，看目标模型在此位置的 argmax token：若它的某个孩子恰好等于这个 argmax → 走进该孩子，继续向下；否则停止。停止位置追加目标模型的 argmax 作为 bonus/修正 token。SGLang 实现为 `verify_tree_greedy_func`（`eagle_utils.py:221`），一个 kernel 内完成全 batch 的树遍历。

### 4.2 采样模式：树上的拒绝采样

链式拒绝采样在树上的推广（`tree_speculative_sampling_target_only` kernel）：在当前节点，对它的孩子们**依次**做拒绝采样判定——

1. 对第一个孩子 $c_1$：以 $\min\!\left(1, \frac{p(c_1)}{q(c_1)}\right)$ 判定，接受则走进 $c_1$；
2. 若拒绝 $c_1$，**把 $c_1$ 从分布中扣除**（$p \leftarrow \mathrm{norm}(\max(0, p - q_{c_1}))$ 的思想），再对下一个兄弟 $c_2$ 判定；
3. 所有孩子都被拒后，从最终残差分布重采样一个修正 token，路径到此为止。

逐兄弟扣除保证了整体输出分布仍严格等于目标分布 $p$（对单孩子情形退化为链式拒绝采样）——树只是给了拒绝采样「多次机会」，无损性不变。

### 4.3 验证后的收尾

- 最长接受路径上的 token 依次落入输出（`accept_tokens`，含末端 bonus/修正 token）；
- **KV cache 回收**：整棵树的 $\text{num_draft_tokens}$ 个位置在验证时都写了目标模型的 KV，被拒绝分支占用的槽位要释放回池子，接受路径的 KV 按最终序列位置落位（`spec_utils.py` 的缓存槽分配 / KV 索引生成）；
- 接受路径末端的 hidden states 被捕获（`CaptureHiddenMode`），作为下一轮 EAGLE 草稿的输入；bonus token 写入 `EagleDraftInput.bonus_tokens`，作为下一轮树的根。

## 5. EAGLE 原理

### 5.1 核心思想：在特征层做自回归，而不是 token 层

独立小草稿模型的困境：要像目标模型，就得大；要便宜，就得小。EAGLE 的破局点是**不从零预测，而是复用目标模型已经算出来的语义特征**。

观察：目标模型最后一层（lm_head 之前）的 hidden state $f_t$ 携带了「下一个 token 是什么」的几乎全部信息，且特征序列比 token 序列**规律得多**（token 是采样出来的离散随机变量，特征是连续、平滑的）。EAGLE 训练一个单层 decoder（草稿头），做**特征级自回归**：

$$
\hat{f}_{t+1} = \text{DraftHead}\big(\,f_t \oplus \mathrm{Embed}(x_{t+1})\,\big)
$$

输入是两样东西的拼接：

1. **目标模型的特征 $f_t$**：包含目标模型对上文的全部理解；
2. **已采样 token $x_{t+1}$ 的 embedding**：消除采样随机性带来的不确定——特征只决定分布，实际走了哪个分支必须显式告诉草稿头（这是 EAGLE 论文标题 "feature uncertainty" 的含义，也是它比同类方法接受率高的关键）。

预测出 $\hat{f}_{t+1}$ 后，直接过**目标模型自己的 lm_head**（共享，不另训）得到下一 token 分布；把 $\hat{f}_{t+1}$ 再喂回草稿头就能继续往深处滚，滚 $k$ 步长出深度为 $k$ 的树。

成本上：草稿头只有一层 transformer decoder（目标模型的 1/N），且 embedding / lm_head 都与目标模型共享，单步草稿成本约为目标模型的几十分之一；效果上：因为站在目标模型特征的肩膀上，接受率远高于同成本的独立小模型——正是 [speculative_decoding.md](./speculative_decoding.md) 第 7 节所说「便宜（$c$ 小）」与「像目标模型（$\alpha$ 高）」的双赢。

### 5.2 EAGLE-1 → 2 → 3 的演进

| 版本 | 关键改进 | 说明 |
| --- | --- | --- |
| **EAGLE-1** | 特征级自回归 + 静态树 | 树形状离线固定（对所有上下文用同一棵树） |
| **EAGLE-2** | **动态树** | 用草稿头的置信度（路径累积概率）现场决定树的形状：置信高长深、置信低长宽，再全局 top-N 剪枝。即本文第 2 节描述的构建方式 |
| **EAGLE-3** | **多层特征融合 + 直接 token 预测** | 输入从「最后一层特征」换成低/中/高多层特征的拼接融合；放弃「回归特征再过 lm_head」，改为草稿头直接预测 token 分布，训练时用 training-time test（训练中模拟多步草稿展开）对齐多步预测，接受率进一步提升，且允许草稿头随训练数据扩展持续变强 |

EAGLE-3 直接预测 token 而非回归特征，解除了「$\hat{f}$ 必须落在目标模型特征空间」的约束，代价是需要目标模型多层 hidden states 的抽取支持（SGLang 中对应 `CaptureHiddenMode` 与模型侧的 EAGLE3 适配）。

### 5.3 MTP：EAGLE 思路的近亲

DeepSeek-V3 等模型原生自带 MTP（multi-token prediction）头——训练时就学了「一步预测未来多个 token」。其结构与 EAGLE 头高度相似（轻量层 + 复用主干特征），因此可以直接当作草稿头做投机解码。SGLang 中 DeepSeek 系列的 MTP 投机即复用 EAGLE 的 worker 路径（`FROZEN_KV_MTP` 是其冻结 KV 的变体）。

## 6. SGLang 实现导览

代码在 `python/sglang/srt/speculative/`，一轮 EAGLE 投机在 `eagle_worker_v2.py` 中的串联：

```
上一轮 accept 结果
  │  bonus_tokens + 接受路径末端 hidden_states
  ▼
[Draft]   EagleDraftInput (eagle_info.py:154)
  │  草稿头自回归 num_steps 步（CUDA Graph: eagle_draft_cuda_graph_runner.py）
  │  每步 topk 扩展 → score_list / token_list / parents_list
  ▼
[组树]    organize_draft_results (eagle_utils.py:71)
  │  全局 top-(num_draft_tokens-1) 剪枝 → parent_list, draft_tokens
  ▼
[摊平]    build_tree_kernel_efficient (eagle_utils.py:107)
  │  bonus token 拼为根 → tree_mask + positions + retrieve_* 树结构索引
  ▼
[Verify]  EagleVerifyInput (eagle_info.py:30) → 目标模型一次 target-verify forward
  │  自定义 mask 交给注意力后端
  ▼
[Accept]  verify_tree_greedy_func / 树形拒绝采样 kernel
  │  最长接受路径 → EagleVerifyOutput
  ▼
KV 回收落位 + 捕获 hidden states → 写回 EagleDraftInput，进入下一轮
```

关键数据结构字段与本文概念的对应：

| 代码字段 | 本文概念 |
| --- | --- |
| `EagleDraftInput.topk_p` / `topk_index` | 每步 top-k 扩展的候选概率 / token（分数与节点） |
| `EagleDraftInput.hidden_states` | EAGLE 草稿头的特征输入 $f_t$ |
| `EagleDraftInput.bonus_tokens` | 下一轮树的根（上一轮的 "+1" token） |
| `EagleVerifyInput.draft_token` | 摊平后的树节点 token 序列 |
| `EagleVerifyInput.custom_mask` / `positions` | 树形注意力掩码 / 树位置编码 |
| `EagleVerifyInput.retrieve_next_token` / `retrieve_next_sibling` | 第一孩子 / 下一兄弟树结构编码 |
| `EagleVerifyInput.spec_steps` / `topk` / `draft_token_num` | 树深 / 分支因子 / 送验节点总数 |

### 参数调优要点

- `num_steps`（深度）、`eagle_topk`（宽度）、`num_draft_tokens`（预算）三者独立可调：预算固定时，接受率高的模型/数据倾向加深，接受率低的倾向加宽；
- `num_draft_tokens` 直接决定 verify forward 的 token 数与树占用的临时 KV 槽位，大 batch 下应保守；
- 观测 `accept_length`（$\tau$）随参数的变化做网格搜索，$\tau$ 不再增长时预算已饱和；
- `--speculative-token-map` 可将草稿头 lm_head 裁剪到高频子词表，进一步降低草稿成本（FR-Spec）。

## 参考文献

- Li et al., *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*, ICML 2024
- Li et al., *EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees*, EMNLP 2024
- Li et al., *EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test*, 2025
- Cai et al., *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*, ICML 2024
- Miao et al., *SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification*, ASPLOS 2024
