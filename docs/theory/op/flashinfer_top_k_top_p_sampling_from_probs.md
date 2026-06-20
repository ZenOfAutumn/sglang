# FlashInfer `top_k_top_p_sampling_from_probs` 原理与实现

本文说明 FlashInfer 提供的高性能采样算子 `top_k_top_p_sampling_from_probs` 的原理，以及它在 SGLang 中是如何被调用的。

## 1. 它解决什么问题

LLM 解码每一步都要从词表概率分布里抽下一个 token。常见的截断采样需要同时施加 **top-k**（只保留概率最高的 $k$ 个）和 **top-p / nucleus**（保留累积概率达到 $p$ 的最小候选集）两种约束，再在剩余候选上按概率重抽样。

朴素实现（即 SGLang 的 pytorch 回退路径 `top_k_top_p_min_p_sampling_from_probs_torch`）的做法是：

1. 对整行概率 **降序排序**（$O(V\log V)$，$V$ 为词表大小，常达 10 万+）；
2. 用排序结果做前缀和，施加 top-k / top-p 掩码置零；
3. 重归一化后 `multinomial` 抽样；
4. 用排序索引映射回原始 token id。

瓶颈在于 **全词表排序 + 多次显存读写**，在大词表、大 batch 下开销显著。FlashInfer 的 `top_k_top_p_sampling_from_probs` 用一个融合 CUDA kernel **免排序** 地完成同样的语义。

## 2. 核心原理：免排序的拒绝采样（Rejection Sampling）

FlashInfer 的关键洞察是：**采样并不需要真正把概率排好序，只需要能从「被 top-k/top-p 截断后的分布」里抽到一个合法 token 即可。** 它用 **拒绝采样 + 并行归约** 取代排序：

### 2.1 基本流程

对每一行（每个请求）概率分布，kernel 在一个 thread block 内循环执行：

1. **从完整分布抽一个候选**：按原始概率 $p_i$ 做一次带权采样，得到候选 token $c$（使用 GPU 上的 counter-based RNG，如 Philox）。
2. **判断该候选是否落在 top-k / top-p 截断集合内**：
   - **top-k 检验**：候选 $c$ 是否属于「概率最高的 $k$ 个」。等价于统计 *有多少个 token 的概率严格大于 $p_c$*——若该计数 $< k$，则 $c$ 在 top-k 集合内。这个计数用一次 **并行归约（block-wide reduction）** 完成，无需排序。
   - **top-p 检验**：候选 $c$ 是否落在 nucleus 内。等价于判断「概率 $\ge p_c$ 的那些 token 的概率之和」是否满足 top-p 约束，同样用并行归约一次扫描得到。
3. **接受或拒绝**：
   - 若候选同时通过 top-k 和 top-p 检验 → **接受**，返回 $c$；
   - 否则 → **拒绝**，重新抽样（回到第 1 步），直到接受或达到最大轮数。

### 2.2 为什么这样是正确的

在被截断（且重归一化）的目标分布上做采样，等价于「在原始分布上反复抽样、只接受落在截断集合内的样本」——这正是拒绝采样的定义。因为接受的样本其相对概率比例与「截断后重归一化」的分布完全一致，所以结果在分布上与「排序 + 掩码 + 重归一化 + multinomial」**等价**，但全程 **不需要 $O(V\log V)$ 排序，也不需要物化整张掩码后的分布**。

### 2.3 `filter_apply_order="joint"` 的含义

SGLang 调用时传入 `filter_apply_order="joint"`：表示 top-k 与 top-p **联合（joint）判定**——对每个被抽中的候选，同时检查它是否落在 top-k 与 top-p 两个集合的 **交集** 内，而不是「先排序施加 top-k 截断、再排序施加 top-p 截断」的串行两步。联合判定让两个约束在同一次拒绝循环里完成，进一步省去中间结果的物化。

> 对照：另一个取值是依次（sequential）应用过滤器，需要中间结果，开销更大；`joint` 是免排序拒绝采样路径的高性能选择。

### 2.4 复杂度与性能

- **时间**：每轮拒绝是一次 $O(V)$ 的并行归约（block 内并行，实际墙钟远低于 $V$）；期望轮数通常很小（top-k/top-p 截断集合相对完整分布占比不会太低，且实现会做优化）。整体优于 $O(V\log V)$ 排序。
- **显存**：原地工作，无需额外的排序缓冲与排序索引数组。
- **kernel 融合**：抽样、检验、接受/拒绝都在一个 kernel 内完成，减少 kernel 启动与显存往返。

## 3. 在 SGLang 中的实现与调用

### 3.1 导入（仅 CUDA / MUSA 后端）

```28:44:python/sglang/srt/layers/sampler.py
if is_cuda():
    from flashinfer.sampling import (
        min_p_sampling_from_probs,
        top_k_top_p_sampling_from_probs,
    )
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
    )

if is_musa():
    from sgl_kernel import (
        min_p_sampling_from_probs,
        top_k_renorm_prob,
        top_k_top_p_sampling_from_probs,
        top_p_renorm_prob,
    )
```

### 3.2 调用路径

当 `sampling_backend == "flashinfer"` 且 **不需要 min-p** 时，直接走 `top_k_top_p_sampling_from_probs` 的联合拒绝采样：

```297:319:python/sglang/srt/layers/sampler.py
            if backend == "flashinfer":
                # 中译：flashinfer 后端是高性能 CUDA kernel，但其内部 RNG 不接受
                # 外部传入的逐请求种子，因此无法支持确定性采样，这里断言种子为空。
                assert (
                    sampling_info.sampling_seed is None
                ), "Sampling seed is not supported for flashinfer backend"
                if sampling_info.need_min_p_sampling:
                    # 中译：min-p 采样路径——先按 top-k、再按 top-p 对概率做截断并重归一化，
                    # 最后基于 min-p 阈值采样。
                    probs = top_k_renorm_prob(probs, sampling_info.top_ks)
                    probs = top_p_renorm_prob(probs, sampling_info.top_ps)
                    batch_next_token_ids = min_p_sampling_from_probs(
                        probs, sampling_info.min_ps
                    )
                else:
                    # 中译：top-k + top-p 联合采样（filter_apply_order="joint" 表示
                    # 两个过滤条件联合应用，而非依次串行）。
                    batch_next_token_ids = top_k_top_p_sampling_from_probs(
                        probs.contiguous(),
                        sampling_info.top_ks,
                        sampling_info.top_ps,
                        filter_apply_order="joint",
                    )
```

要点：

- **输入**：`probs` 形状 `(batch, vocab)`，已过 softmax；调用前 `.contiguous()` 保证内存连续以满足 kernel 要求。`top_ks` / `top_ps` 是逐请求的张量，因此一个 batch 内不同请求可用不同的 k/p。
- **输出**：形状 `(batch,)` 的下一个 token id（kernel 直接返回原始词表下标，无需像排序方案那样再做 `gather` 映射）。
- **min-p 走不同路径**：当 `need_min_p_sampling=True` 时，不走联合拒绝采样，而是先用 `top_k_renorm_prob` / `top_p_renorm_prob` 显式截断重归一化，再用 `min_p_sampling_from_probs` 采样。

### 3.3 关键限制：不支持确定性采样

FlashInfer 采样 kernel 内部使用 **GPU 全局 RNG**，不接受外部传入的逐请求 `sampling_seed`。因此一旦请求需要确定性采样（带 `sampling_seed`），就 **不能** 用 flashinfer 后端，代码用断言保证：

```300:302:python/sglang/srt/layers/sampler.py
                assert (
                    sampling_info.sampling_seed is None
                ), "Sampling seed is not supported for flashinfer backend"
```

需要可复现的确定性采样时，应改用 pytorch 后端（`top_k_top_p_min_p_sampling_from_probs_torch` + `multinomial_with_seed` 的 Gumbel-Max 路径，见 `docs/theory/Glossary.md` 的「随机种子」「torch.multinomial」条目）。

## 4. 与 pytorch 回退实现的对比

| 维度 | flashinfer `top_k_top_p_sampling_from_probs` | pytorch `top_k_top_p_min_p_sampling_from_probs_torch` |
| --- | --- | --- |
| 核心算法 | 免排序拒绝采样 + 并行归约 | 全词表降序排序 + 掩码置零 + multinomial |
| top-k/top-p 应用方式 | `joint` 联合判定 | 串行：先 top-k 再 top-p（再可选 min-p） |
| min-p 支持 | 否（min-p 走单独的 renorm + `min_p_sampling_from_probs`） | 是（同一函数内统一处理） |
| 确定性采样（逐请求种子） | 不支持 | 支持（Gumbel-Max + `murmur_hash32`） |
| 复杂度 | 优于 $O(V\log V)$，kernel 融合 | $O(V\log V)$ 排序 + 多次显存读写 |
| 适用场景 | 高吞吐、大词表、无需复现 | 需要确定性 / min-p / 作为通用回退 |

## 5. 小结

`top_k_top_p_sampling_from_probs` 用 **免排序的联合拒绝采样** 在单个融合 CUDA kernel 内完成 top-k + top-p 截断采样：反复从原始分布抽候选，用并行归约判定其是否落在 top-k 与 top-p 的交集内，接受则返回、否则重抽。它在分布上与「排序+掩码+重归一化+multinomial」等价，但避免了昂贵的全词表排序，因此是 SGLang 在 CUDA 上的高性能默认采样路径；代价是 **不支持逐请求确定性采样**，该场景需回退到 pytorch 后端。

