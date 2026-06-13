# 面向长上下文的流水线并行(Pipeline Parallelism for Long Context)

## 为什么需要流水线并行?

随着大语言模型(LLM)向万亿参数架构和"无限"上下文窗口扩展,底层的服务基础设施必须演进到更细粒度的跨节点并行策略。虽然 KV 缓存技术能有效缓解冗余计算,但它们无法规避超长序列在极大初始输入 token 长度(Input Token Length,ITL)下固有的高昂首 token 时间(Time to First Token,TTFT)。尽管张量并行(Tensor Parallelism,TP)仍是节点内扩展的常规方法,但它在多节点部署中经常遇到通信瓶颈。另一方面,流水线并行只需在每个流水线阶段的边界处进行跨节点通信,相比大规模 TP 可以实现更好的计算-通信重叠。因此,它也是一种有前景的提升吞吐量的并行策略。

详细分析可在这篇 [博客](https://lmsys.org/blog/2026-01-15-chunked-pipeline/) 中找到。

## 基于异步通信的实现重构
借助动态分块预填充(Dynamic Chunked Prefill),流水线并行有潜力降低长上下文输入的 TTFT。对于每个请求,其输入 token 可以被划分为多个 chunk,每个 chunk 都不长于 chunked prefill 大小。同一请求的不同 chunk 可以由不同节点同时处理,从而并行化处理并降低 TTFT。SGLang 已经支持流水线并行(#5724)一段时间,并使其与 PD 分离功能兼容(#8846),但该实现并不完美,在性能上有很大的提升空间。

为了消除这一性能隐患,SGLang 实现了一个带非阻塞异步点对点(P2P)通信的微批处理事件循环(Micro-batching Event Loop),以将 GPU 计算与 CPU 元数据处理及 PP 通信重叠。这确保了当一个微批次正在 GPU 上计算时,下一个微批次已经在被准备并有效地移动到位,从而尽可能保持流水线处于饱和状态。这种方法最早在 #7979 中提出,并在 #11852 中被重新设计和纳入。

实现的关键机制包括:

* **事件循环中解耦的同步/异步逻辑:** scheduler 在 `_pp_send_pyobj_to_next_stage` 中使用 `async_send`。它不等待传输完成,而是返回一个 `P2PWork` 句柄。实际的同步(`P2PWork.work.wait()`)被推迟到调用 `_pp_commit_comm_work` 时,从而允许 CPU 在数据传输过程中执行其他工作——比如调度下一个批次或处理元数据。
* **多流执行(Multi-Stream Execution):** 除了作为同步流的主 `default_stream` 之外,SGLang 还利用专用的 `forward_stream` 和 `copy_stream` 分别执行前向传递的 GPU 计算和 Data-to-Host(D2H)内存传输,以获得更好的重叠。当 `_pp_launch_batch` 正在当前阶段的 GPU 上执行当前微批次时,CPU 使用 `_pp_process_batch_result` 处理上一个微批次的结果。

## 关于动态分块(Dynamic Chunking)的指南

### 为什么需要动态分块
固定大小的分块预填充会在流水线中造成气泡(bubble),尤其是当 pp 大小较大时。这一现象背后的主要原因是,即使每个 chunk 大小相同,模型的运行时间也是不均匀的(由 Transformer 结构带来)。前缀序列长度越大,该 chunk 的运行时间就越长。而这些气泡会传播到下一个阶段,并显著降低更大 pp rank 的扩展效率。

为解决这一问题,SGLang 引入了动态分块机制,以预测下一个 chunk 的最优大小,使其满足以下条件:

Runtime(L + Next Chunk Size) - Runtime(L) = Runtime(Initial Chunk Size)

其中 ***L*** 表示前缀序列长度(Prefix Sequence Length)。通过分析一系列具有不同 ITL 的请求,我们将累积运行时间建模为序列长度的二次函数。利用这个模型,我们为任何给定的前缀长度 ***L*** 求解最优的下一个 chunk 大小。由于注意力机制的计算复杂度随 ***L*** 增长,因此随着 ***L*** 的增大,下一个 chunk 的大小会逐渐减小,以在流水线各阶段之间保持对齐的 chunk 执行时间。

基于这种方法,scheduler 可以在运行时预测并动态减小 chunk 大小,以最小化由阶段不对齐造成的气泡。需要注意的是,scheduler 并不使用原始的预测值。为了便于高效的 KVCache 内存管理并确保与硬件执行效率的亲和性,该值会向下对齐到最接近的 max(`--page-size`, 64) 的倍数。


### Chunked Prefill 大小与平滑因子(Smoothing Factor)

当启用 `--enable-dynamic-chunking` 时,序列的每个 chunk 大小是基于二次模型动态确定的——该模型根据初始 chunk 长度的估计运行时间来预测下一个 chunk 大小。在这种情况下,我们使用 `--chunked-prefill-size` 来设置初始 chunk 大小。当切换到动态分块模式时,初始 chunk 大小(`--chunked-prefill-size`)应设置为与原始 chunked prefill 大小相当的较大值,这样就不会产生太多 chunk。

**`SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR`** 是一个环境变量,用于控制动态分块算法的平滑因子,默认为 0.75。它决定了在 prefill 阶段 chunk 大小可以变化多少。较大的值意味着更激进的 chunk 大小变化,这可能带来更好的性能,但也会导致更大的 chunk 大小变化(末尾的 chunk 大小可能变得非常小,这可能导致性能下降)以及更多的总 chunk 数。当设置为 1 时,chunk 大小将严格基于前述预测下一个 chunk 大小的二次模型来调整。较小的值意味着更保守的 chunk 大小变化,这可能导致更小的 chunk 大小变化和更少的总 chunk 数。当设置为 0 时,chunk 大小不会被动态调整,因此它与传统的固定 chunked prefill 大小的方式相同。

由于硬件、模型和目标工作负载的差异,静态配置很少能在所有场景中都达到最优。因此,在切换到动态分块模式时,达到峰值性能需要一定程度的超参数调优。

**动态分块预填充的调优指南**

* **步骤 1 \- 迭代找出目标 PP 大小的最优固定 chunked prefill 大小**:对于目标 ITL,不同的 PP 大小可能有不同的最优 chunked prefill 大小。因此,用户应根据可用于扩展的资源进行迭代以获得基线。
* **步骤 2 \- 为动态分块选择初始 chunk 大小**:将初始大小设置为最优固定 chunked prefill 大小的 2 倍或 3 倍。这会减少 chunk 的总数,并防止"尾部 chunk(tail chunks)"对硬件的利用率不足。为了对极大的输入 token 长度(ITL)保持效率,动态预测器会自动确保后续的 chunk 至少为该初始大小的 1/4。此外,对于这类情况,也建议使用更大的初始 chunk 大小(例如最优固定 chunked prefill 大小的 4 倍)。
* **步骤 3 \- 平滑因子调整**:该因子控制 chunk 大小调整对二次性能拟合模型所给出预测的严格程度。
  * 1.0:严格遵循模型。
  * **0.6 – 0.85(推荐)**:在动态扩展与硬件稳定性之间取得最佳平衡的典型范围。通过实验,我们发现 0.6 到 0.85 之间的范围通常能为动态分块带来最佳性能。
  * 0:禁用动态调整,回退到传统的固定大小分块。
* **另一个小优化技巧:** 当各层不能被各 rank 均匀整除时,将较大的分区放在更高的 PP rank 上。当更高的 PP rank 在等待前一阶段的结果时,这可以提高 GPU 利用率,从而减少更高 PP rank 上的气泡。如果以 DeepSeek-V3.1 为例,`SGLANG_PP_LAYER_PARTITION=15,15,15,16` 通常比 `16,15,15,15` 表现更好。

## 长上下文的最佳实践

### 调优 Chunked Prefill 大小
优化 chunked prefill 大小对于平衡流水线效率和资源利用率至关重要。理想的大小取决于多种因素,包括模型架构、硬件配置和典型的输入长度。我们建议从一个较小的 chunk 大小开始,例如 4K,然后逐渐增加,直到你为你的特定用例找到最优大小(不同的目标 ITL 和 PP 大小可能有不同的最优 chunked prefill 大小。因此,用户应根据可用于扩展的资源进行迭代以获得基线)。或者,你可以分析硬件容量,并基于 roofline 模型确定最优 chunk 大小。

### 为超长 ITL 启用动态分块并调整平滑因子
SGLang 还提供了一种动态分块解决方案,可以进一步提升性能。此功能目前是实验性功能,需要一定量的调优实验,可能并不适合所有工作负载。此外,微调平滑因子有助于针对特定工作负载和模型特性优化性能。

### NVIDIA H20 上的案例研究

在使用从 2K 到 16K 的固定 chunked prefill 大小评估流水线并行时,实验结果表明,4K 的 chunk 大小为 DeepSeek-V3.1 提供了最优的 prefill TTFT 性能,而 6K 的 chunk 大小为 Qwen3-235B-A22B-FP8 提供了最优的 prefill TTFT 性能。

在启用动态分块时,我们首先将最优固定 chunked prefill 大小按 3 倍缩放作为初始 chunk 大小。通过实验,我们发现 2-3 倍的乘数提供了适当的平衡——既避免了过多的初始流水线气泡,又确保后续 chunk 不会随着上下文长度增加而变得太小。在默认的动态分块平滑因子 0.75 下,我们进行了参数调优,并确定对于 DeepSeek-V3.1,在 12K 初始 chunk 大小下取值 0.65 最优;而对于 Qwen3-235B-A22B-FP8,在 18K 初始 chunk 大小下取值 0.8 最优。

#### DeepSeek-V3.1,128K 输入 Token 长度
```bash
# prefill node 0 (fixed chunked prefill size)
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.1 --trust-remote-code \
  --nnodes 4 --node-rank 0 --tp 8 --pp-size 4 \
  --port 30000 --dist-init-addr <MASTER_NODE_IP> \
  --disable-radix-cache --mem-fraction-static 0.8  \
  --attention-backend fa3 --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 4096
```

```bash
# prefill node 0 (with dynamic chunking)
export SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.65
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.1 --trust-remote-code \
  --nnodes 4 --node-rank 0 --tp 8 --pp-size 4 \
  --port 30000 --dist-init-addr <MASTER_NODE_IP> \
  --disable-radix-cache --mem-fraction-static 0.8  \
  --attention-backend fa3 --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 12288 --enable-dynamic-chunking
```

#### Qwen3-235B-A22B-FP8,128K 输入 Token 长度
```bash
# prefill node 0 (fixed chunked prefill size)
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-235B-A22B-FP8 --trust-remote-code \
  --nnodes 4 --node-rank 0 --tp 4 --pp-size 8 \
  --port 30000 --dist-init-addr <MASTER_NODE_IP> \
  --disable-radix-cache --mem-fraction-static 0.8  \
  --attention-backend fa3 --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 6144
```

```bash
# prefill node 0 (with dynamic chunking)
export SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.8
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-235B-A22B-FP8 --trust-remote-code \
  --nnodes 4 --node-rank 0 --tp 4 --pp-size 8 \
  --port 30000 --dist-init-addr <MASTER_NODE_IP> \
  --disable-radix-cache --mem-fraction-static 0.8  \
  --attention-backend fa3 --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 18432 --enable-dynamic-chunking
```

注意:`--disable-radix-cache` 仅出于可复现基准测试的目的而启用。不建议在生产环境中使用它。

## 流水线并行结合 PD 分离的最佳实践
待补充。敬请关注流水线并行结合 PD 分离的最新更新。
