# NGRAM 投机解码：候选生成、树结构、Overlap 与 Attention Mask

本文基于当前 SGLang 实现，解释以下问题：

1. NGRAM 为什么不需要草稿模型？它的“草稿”从哪里来？
2. `NgramCorpus.batch_get()` 如何构建候选树？`batch_put()` 何时被调用？
3. NGRAM 的树结构与 EAGLE 有什么不同？
4. overlap 模式下，`_prepare_draft_tokens()` 为什么要拼接 `spec_info.accept_tokens`？
5. `USE_FULL_MASK` 有什么作用？为什么有时使用全掩码，有时不用？

---

## 1. NGRAM 为什么不需要草稿模型？

传统投机解码使用一个较小的草稿模型预测候选：

```text
草稿模型前向
    ↓
生成候选 token 或候选树
    ↓
目标模型一次性验证
```

NGRAM 不使用神经网络生成候选，而是基于以下假设：

> 历史中出现过的 token 序列，很可能会再次出现。

它用 CPU 上的历史语料索引代替草稿模型：

```text
当前上下文后缀
    ↓
运行时 N-gram Trie / 外部语料 SAM
    ↓ 后缀匹配
查找历史上跟在该后缀之后的 token
    ↓
构造候选树
    ↓
目标模型一次性验证
```

因此 NGRAM 不需要：

- 草稿模型权重；
- 草稿模型前向计算；
- 草稿模型 KV Cache；
- 验证之后的 draft extend / 草稿状态追赶。

NGRAM 仍然需要目标模型执行树形验证，只是省去了 EAGLE 的多步草稿前向。

代码中 `NGRAMWorker.draft_worker` 直接返回 `None`：

```python
@property
def draft_worker(self) -> Optional[EagleDraftWorkerBase]:
    return None
```

### 1.1 草稿的两个来源

#### 运行时 N-gram Trie

每轮 decode 会把最近的 token 历史通过 `batch_put()` 插入 Trie。后续请求或同一请求出现相同后缀时，Trie 可以返回历史 continuation。

#### 外部语料 SAM

也可以预先加载已 tokenize 的外部语料。外部语料由 suffix automaton（SAM，后缀自动机）索引。

这种方式适合：

- 固定 JSON/XML 模板；
- 代码补全；
- RAG 原文复述；
- 固定格式输出；
- 高重复率业务文本。

外部语料可以避免运行时 Trie 的冷启动问题。

---

## 2. `NgramCorpus.batch_get()` 如何构建候选树？

整体流程如下：

```text
等待异步插入完成
    ↓
构造当前请求的历史后缀
    ↓
匹配这个上下文的所有有效后缀
    ↓
从匹配节点向后展开候选
    ↓
合并重复分支并限制总节点数
    ↓
输出扁平 token + 祖先 mask
```

### 2.1 等待上一轮插入完成

`batch_put()` 的底层插入由 C++ 后台线程异步执行。调用 `batch_get()` 前需要：

```python
self.ngram_corpus.synchronize()
```

否则查询可能读不到上一轮刚提交的语料。

### 2.2 构造查询后缀

每个请求只保留逻辑历史的最后 `max_trie_depth` 个 token：

```text
prompt ⊕ output_ids ⊕ overlap 下尚未写入 output_ids 的 accept_tokens
```

随后传给 `batch_get()`：

- `req_ids`：请求 ID，用于关联 C++ 侧的增量匹配状态；
- `batch_tokens`：当前请求的历史尾部；
- `total_lens`：完整逻辑序列长度，用于判断匹配状态是否连续。

```python
req_drafts, mask = self.ngram_corpus.batch_get(
    req_ids, batch_tokens, total_lens
)
```

Python 封装会把 `req_id` 映射成紧凑的整数 `state_id`，最终调用 C++：

```python
state_ids = [self._get_state_id(rid) for rid in req_ids]
return self._obj.match_stateful(state_ids, batch_tokens, total_lens)
```

### 2.3 Trie 中存储了什么？

插入一个 token 序列时，Trie 从每个起点插入最长 `max_trie_depth` 的子串：

```cpp
for (size_t i = 0; i < len; ++i) {
  auto start = tokens + i;
  auto end = start + std::min(len - i, param_.max_trie_depth);
  // 插入 [start, end)
}
```

例如插入：

```text
[A, B, C, D, E]
```

Trie 会保存类似以下路径：

```text
A → B → C → D → E
B → C → D → E
C → D → E
D → E
E
```

所以查询当前上下文 `[A, B, C, D]` 时，可以尝试：

```text
[A, B, C, D]  # 最长、最精确
[B, C, D]
[C, D]
[D]            # 最短、召回更多但更模糊
```

匹配成功的节点称为 anchor。只有存在后继节点的 anchor 才能提供候选。

### 2.4 增量匹配状态

C++ 侧为每个请求维护 `MatchState`，包括：

- 当前不同后缀长度对应的 Trie anchor；
- 上次已处理的逻辑总长度；
- Trie epoch 和节点版本。

如果本轮 `total_len` 与上一轮连续，只需要消费新增 token 来推进状态；如果：

- 请求历史回退；
- Trie reset；
- 节点被 LRU 淘汰；
- 状态版本失效；

则重新扫描当前后缀。

### 2.5 候选展开模式

#### BFS / Recency 模式

`match_type="BFS"` 时：

- 从最长匹配后缀向较短匹配后缀处理；
- 按节点的新近度/LRU 顺序选择后继；
- 每个节点的展开宽度由 `min_bfs_breadth` 和 `max_bfs_breadth` 控制；
- 相同父节点下相同 token 会合并；
- 达到候选节点预算后停止。

这里的宽度会随 anchor 匹配深度变化：匹配越长、越精确，展开通常越窄；匹配越短、越模糊，允许探索更多分支。

#### PROB / Frequency 模式

`match_type="PROB"` 时：

- Trie 节点记录历史出现频次 `freq`；
- 每个节点最多考虑 `max_bfs_breadth` 个高频后继；
- 使用优先队列，按照路径累计概率优先展开；
- 达到总节点预算后停止。

### 2.6 节点预算

配置中的 `draft_token_num=D` 是一次目标验证的总节点数，包含根节点。因此真正从语料库展开的候选最多为：

\[
D-1
\]

第 0 个节点固定为当前最后一个已知 token，作为候选树的根。

### 2.7 转换成扁平 token 和祖先 mask

候选树完成后，`fillResult()` 会：

1. 将当前最后一个 token 放到索引 0，作为根；
2. 对候选树做 BFS，生成扁平 token 数组；
3. 记录每个节点的父节点索引；
4. 候选不足时用 token 0 padding 到固定长度；
5. 复制父节点的 mask 行并标记自身，生成祖先可见性矩阵。

例如：

```text
A(root)
├── B
│   └── D
└── C
```

会被表示为：

```text
tokens = [A, B, C, D]

mask =
A  1 0 0 0
B  1 1 0 0
C  1 0 1 0
D  1 1 0 1
```

其中：

```text
mask[i][j] = 1
```

表示节点 `i` 可以看到节点 `j`，也就是 `j` 位于 `i` 的祖先链上。

GPU 侧再根据这个 mask 重建：

- `positions`；
- `retrieve_index`；
- `retrieve_next_token`，即第一个子节点；
- `retrieve_next_sibling`，即下一个兄弟节点。

后两个数组构成“左孩子—右兄弟”树表示，验证 kernel 可以用它们遍历任意形状的候选树。

---

## 3. `batch_put()` 何时被调用？

在 decode 的目标模型验证完成之后，调用链是：

```text
batch_get()
    ↓
目标模型验证整棵候选树
    ↓
eagle_sample() 计算接受路径
    ↓
整理接受路径的 KV Cache
    ↓
on_publish(new_seq_lens)
    ↓
_update_ngram_corpus()
    ↓
batch_put()
```

对应代码流程：

```python
predict, accept_lens, accept_index = eagle_sample(...)
new_seq_lens = batch.seq_lens + accept_lens
accept_tokens = predict[accept_index].flatten()

move_accept_tokens_to_target_kvcache(...)

if on_publish is not None:
    on_publish(new_seq_lens)

self._update_ngram_corpus(batch)
```

`_update_ngram_corpus()` 最终执行：

```python
self.ngram_corpus.batch_put(batch_tokens)
```

### 3.1 插入内容

每个请求插入的是当前逻辑历史末尾最多 `max_trie_depth` 个 token：

```text
prompt 尾部 ⊕ output_ids 尾部 ⊕ 必要时的 prev_tokens
```

插入操作是异步的，下一轮 `_prepare_draft_tokens()` 调用 `synchronize()` 后才能保证可见。

### 3.2 当前实现的边界

当前实现只在 decode/TARGET_VERIFY 分支调用 `_update_ngram_corpus()`：

- decode：会更新运行时 Trie；
- extend/prefill：不会将完整 prompt 主动插入 Trie。

代码中已有 FIXME，说明是否将 extend 输入写入 Trie 曾被考虑，但测试收益不明显，当前没有启用。因此运行时 Trie 主要从 decode 输出积累，外部语料对于冷启动仍然很重要。

### 3.3 流水线视角

`accept_tokens` 会被放入返回结果的 `next_draft_input`：

```python
next_draft_input = NgramVerifyInput(
    draft_token_num=self.draft_token_num,
    new_seq_lens=new_seq_lens,
    accept_tokens=accept_tokens,
    accept_lens=accept_lens,
)
```

在 overlap 模式下，这批 token 会在下一次草稿准备阶段作为 `prev_tokens` 被拼回逻辑历史。因此 Trie 的查询和更新都遵循同一套“当前 CPU 已提交历史 + 流水线中尚未提交 token”的视图。

---

## 4. NGRAM 的树结构与 EAGLE 有什么不同？

两者的验证阶段高度复用，都包含：

- `custom_mask`；
- `positions`；
- `retrieve_index`；
- `retrieve_next_token`；
- `retrieve_next_sibling`；
- `eagle_sample()`。

但候选来源和树形约束不同。

| 属性 | EAGLE | NGRAM |
|---|---|---|
| 候选来源 | 草稿模型 logits | Trie/SAM 历史匹配 |
| 树深 | 由草稿前向步数固定 | 不预先固定，由语料路径决定 |
| 分支 | 每层按 `topk` 展开 | 每个节点分支数不规则 |
| 总节点数 | 根据 steps/topk 选择固定验证节点 | 受 `draft_token_num` 总预算限制 |
| `max_tree_depth` | `spec_steps + 1` | `draft_token_num` |
| `tree_topk` | 正整数 `topk` | `-1` |
| 草稿模型状态 | 有权重、KV 和 hidden state | 无 |
| 验证后 draft extend | 需要 | 不需要 |

### 4.1 `max_tree_depth` 的区别

EAGLE 的树深由草稿模型迭代步数确定：

```python
@property
def max_tree_depth(self) -> int:
    return self.spec_steps + 1
```

例如：

```text
spec_steps = 5
max_tree_depth = 6  # 五步草稿 + 根节点
```

即使候选树总共有很多节点，一条根到叶路径也不会超过 6 个节点。

NGRAM 不按步数限制树深，只限制总节点数：

```python
@property
def max_tree_depth(self) -> int:
    return self.draft_token_num
```

例如 `draft_token_num=16`，候选树既可能是：

```text
宽树：root + 15 个直接子节点
```

也可能退化成：

```text
深链：root → t1 → t2 → ... → t15
```

所以最坏路径深度必须按 16 分配 `accept_index`。

### 4.2 `tree_topk` 的区别

EAGLE：

```python
@property
def tree_topk(self) -> int:
    return self.topk
```

它的树由草稿模型按 `speculative_eagle_topk` 规律展开，验证逻辑可以知道固定分支因子。

NGRAM：

```python
@property
def tree_topk(self) -> int:
    return -1
```

`-1` 表示不存在统一的固定分支因子。不同节点可能分别有 1、3、7 个孩子，验证 kernel 必须依靠：

```text
retrieve_next_token
retrieve_next_sibling
```

遍历不规则树。

### 4.3 `tree_topk=-1` 不等于完全没有宽度控制

NGRAM 仍有 `max_bfs_breadth`，但它和 EAGLE 的 `tree_topk` 不是同一层面的概念：

- `max_bfs_breadth`：语料查询阶段，一个 Trie 节点最多考虑多少个后继；
- `tree_topk`：验证阶段能否假设整棵树具有统一的固定 top-k 结构。

NGRAM 的搜索可以限制局部宽度，但最终树仍然是不规则树。

---

## 5. Overlap 模式下为什么要拼接 `spec_info.accept_tokens`？

### 5.1 根因：结果处理滞后一轮

Overlap scheduler 会让下一轮 CPU 准备与当前轮 GPU 计算重叠。其流水线大致为：

```text
GPU 完成第 N 轮 verify
        ↓
第 N 轮 accept_tokens 已经产生
        ↓
CPU 结果处理尚未把它们写入 req.output_ids
        ↓
第 N+1 轮 _prepare_draft_tokens() 已经开始
```

所以在第 N+1 轮准备候选时：

```text
req.output_ids
```

仍然比模型已经确定的逻辑序列少一段。这段缺失 token 保存在上一轮返回的：

```text
batch.spec_info.accept_tokens
batch.spec_info.accept_lens
```

因此必须构造：

```text
真实逻辑历史 = prompt + req.output_ids + prev_tokens
```

其中每个请求的有效 `prev_tokens` 是：

```python
prev_tokens = self.prev_token_ids[
    i * draft_token_num : i * draft_token_num + self.prev_accept_lens[i]
]
```

这是因为 `accept_tokens` 按每请求固定 `draft_token_num` 的 stride 存储，而 `accept_lens[i]` 指明前多少个元素有效。

### 5.2 不拼接会出现什么问题？

#### 后缀匹配错误

`batch_get()` 会基于过期历史查询 Trie，候选 continuation 会落后于实际序列。

#### 增量状态错误或退化

传给 C++ 的 `total_lens` 也会偏小，使请求级 `MatchState` 无法按真实新增 token 正确推进，可能触发状态重建，甚至从错误后缀开始匹配。

### 5.3 为什么同步模式不拼？

同步模式会在下一轮准备开始前完成结果处理，所以 `req.output_ids` 已包含上一轮接受的 token。再次拼接会造成重复。

代码使用：

```python
use_prev_tokens = self.enable_overlap and not batch.grammar_needs_sync()
```

也就是说：

- overlap 且 grammar 不要求同步：拼接；
- 非 overlap：不拼接；
- grammar batch：结果同步处理，不能重复拼接。

`prev_tokens` 还被暂存在 worker 中，供后续 `_update_ngram_corpus()` 使用，确保同一轮的查询和写回基于一致的逻辑历史。

---

## 6. `USE_FULL_MASK` 的作用是什么？

树形验证中，每个候选 query 应当看到：

1. 全部已经提交的历史 KV；
2. 自己所在候选路径上的祖先节点；
3. 不能看到兄弟节点和其他分支。

假设：

- 历史长度为 \(L\)；
- 候选节点数为 \(D\)；
- Trie 返回的树内 mask 为 \(M_{tree}\in\{0,1\}^{D\times D}\)。

### 6.1 `USE_FULL_MASK=True`

显式构造完整掩码：

\[
M_{full}=\left[\mathbf{1}_{D\times L}\mid M_{tree}\right]
\]

其 shape 为：

\[
D\times(L+D)
\]

左半部分全为 1，表示每个候选节点都可以看到全部历史 KV；右半部分是 Trie 构造的树内祖先关系。

当前 NGRAM GPU 路径会为每个请求构造：

```python
req_mask = torch.cat(
    (
        ones_mask,       # [D, L]，历史全部可见
        tree_mask,       # [D, D]，仅祖先可见
    ),
    dim=1,
)
```

#### 优点

- mask 语义完整、自包含；
- attention backend 不需要额外推断历史区域；
- 兼容只接受完整 `Q × K` custom mask 的 GPU 后端。

#### 缺点

内存与构造复杂度为：

\[
O(DL+D^2)
\]

长上下文、大 batch 时，全 1 历史区域会很大；同时还有 Python 循环、tensor 拼接和数据搬运开销。

### 6.2 `USE_FULL_MASK=False`：QLEN-only mask

只传树内部的：

\[
M_{tree}\in\{0,1\}^{D\times D}
\]

历史 KV 的全可见关系由 attention backend 隐式处理。

#### 优点

空间复杂度降为：

\[
O(D^2)
\]

不再随历史长度 \(L\) 线性增长。当前代码注释中的测试结果约快 8%，收益通常随 batch 增大。

#### 缺点

attention backend 必须理解 QLEN-only 布局：

- custom mask 只描述最后 \(D\) 个树节点；
- 前面的历史 KV 默认全部可见。

如果后端仍按完整 mask 解读，就会产生 shape 不匹配或错误的注意力可见性。

### 6.3 为什么有时使用全掩码，有时不用？

这是 attention backend 的 mask 协议和能力差异，不是 NGRAM 算法语义发生变化。

SGLang 通用树 mask 模式包括：

```text
FULL_MASK
QLEN_ONLY
QLEN_ONLY_BITPACKING
```

其中：

- `FULL_MASK`：完整 `[D, L+D]` 矩阵；
- `QLEN_ONLY`：只保留 `[D, D]` 树内部分；
- `QLEN_ONLY_BITPACKING`：在 QLEN-only 基础上位打包，继续降低存储和带宽。

默认策略是：

```python
def default_tree_mask_mode() -> TreeMaskMode:
    return TreeMaskMode.QLEN_ONLY if _is_cpu else TreeMaskMode.FULL_MASK
```

因此当前实现中：

- NGRAM GPU 路径使用 `USE_FULL_MASK=True`，显式拼接历史全 1 mask；
- CPU 路径不会进入 `USE_FULL_MASK and not _is_cpu` 分支，CPU attention kernel 直接消费 QLEN-only mask；
- 其他支持 QLEN-only 或 bit-packing 的后端，可以省掉完整历史 mask。

关键结论是：

> FULL_MASK 和 QLEN-only 表达的是同一棵候选树、同一种注意力语义；前者兼容性更强，后者内存更省且更快。是否能关闭 `USE_FULL_MASK` 取决于对应 attention backend 是否正确实现了“历史 KV 隐式全可见”，不能只改常量而不改后端。

---

## 7. 一轮 NGRAM Decode 的完整时序

```text
上一轮 next_draft_input
  ├── accept_tokens
  └── accept_lens
          ↓
_prepare_draft_tokens()
  ├── 修复 overlap 下尚未进入 output_ids 的 token
  ├── synchronize() 等待异步 Trie 插入
  └── batch_get() 匹配并展开候选树
          ↓
_prepare_for_speculative_decoding()
  ├── 重建 positions / child / sibling
  ├── 构造 FULL_MASK 或 QLEN-only mask
  ├── 分配候选 KV 槽位
  └── DECODE → TARGET_VERIFY
          ↓
目标模型一次前向验证全部候选节点
          ↓
eagle_sample()
  ├── 沿候选树匹配目标输出
  ├── 计算 accept_lens / accept_index
  └── 产生 bonus token
          ↓
move_accept_tokens_to_target_kvcache()
  ├── 压缩接受路径 KV
  └── 回收错误分支 KV
          ↓
_update_ngram_corpus()
  └── batch_put() 异步写入历史后缀
          ↓
构造下一轮 next_draft_input
```

## 8. 核心结论

1. **NGRAM 的草稿不是模型预测，而是历史语料匹配结果。**
2. **`batch_get()` 用当前历史后缀匹配 Trie/SAM，再按新近度或频率展开不规则候选树。**
3. **`batch_put()` 在 decode 验证和接受处理后调用，并异步更新运行时 Trie。**
4. **EAGLE 的树深由 `spec_steps` 决定、分支由固定 `topk` 描述；NGRAM 只受总节点预算约束，树深和分支都不规则。**
5. **Overlap 下 `req.output_ids` 落后一轮，必须拼接 `spec_info.accept_tokens` 才能还原真实逻辑历史。**
6. **FULL_MASK 显式编码“历史全可见 + 树内祖先可见”；QLEN-only 只编码树内部分，更高效但依赖后端支持。**
