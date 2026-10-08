# 树形草稿（Tree Drafting）原理详解

本文详细介绍树形草稿的动机、树的构建 / 摊平 / 树形注意力掩码 / 树形验证的完整机制。这套机制与草稿模型的来源无关——EAGLE 草稿头、MTP 模块、独立小模型都可以往树上长草稿。投机解码的基础流程、拒绝采样的无损性证明见同目录 [speculative_decoding.md](./speculative_decoding.md)；EAGLE 系列草稿头的原理与 SGLang 实现见 [eagle.md](./eagle.md)。

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
- 接受路径末端的 hidden states 被捕获（`CaptureHiddenMode`），作为下一轮草稿的输入；bonus token 写入 `EagleDraftInput.bonus_tokens`，作为下一轮树的根。

## 参考文献

- Li et al., *EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees*, EMNLP 2024
- Cai et al., *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*, ICML 2024
- Miao et al., *SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification*, ASPLOS 2024
