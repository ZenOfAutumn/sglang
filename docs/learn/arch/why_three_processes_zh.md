# 为什么 SGLang 把 Runtime 拆成三类进程而不是一个？

本文回答：SGLang 的前端 / 调度 / 反分词为什么要拆成三类进程，以及这样拆分对**延迟**与**吞吐**分别有什么好处。

先纠正两个常见误区：

1. 拆分的**根因是 Python GIL**，而不是笼统的"资源隔离"；
2. "多个 Scheduler 共享 tokenizer / detokenizer"的本质是**避免重复 + 独立扩展**，而不是"提高硬件资源利用效率"。

## 一、三类进程各自做什么

| 进程 | 数量 | 职责 | 资源性质 |
| --- | --- | --- | --- |
| **TokenizerManager** | 1 个（前端进程） | HTTP / Engine 入口、请求解析与分发、**tokenize**；用 asyncio 按 `rid` 收集输出、拼装流式 / 非流式响应 | 纯 CPU |
| **Scheduler**（含 `TpModelWorker`） | **每个 TP rank 一个** | 连续批处理调度、KV cache / 显存池分配管理、驱动 GPU forward、**采样与后处理、组织回包** | **CPU + GPU**（唯一持有并驱动 GPU 的进程） |
| **DetokenizerManager** | 1 个 | 把 token id **反分词**成文本、增量拼接、裁剪 stop 串，再回送 TokenizerManager | 纯 CPU |

三者之间全部通过 **ZMQ（PUSH/PULL）** 解耦通信，跨进程协议定义在 `io_struct.py` 中：

- 请求：`TokenizedGenerateReqInput`（分词后，Scheduler 只处理已分词形态）
- 输出：`BatchTokenIDOutput` → 反分词 → `BatchStrOutput`

主链路（沿请求生命周期单向流转）：

```text
HTTP 请求
  → TokenizerManager（tokenize，前端进程，异步）
  → [ZMQ] TokenizedGenerateReqInput
  → Scheduler（独立进程，连续批处理 + KV 管理 + 驱动 GPU forward）
  → [ZMQ] BatchTokenIDOutput
  → DetokenizerManager（反分词、增量拼接）
  → [ZMQ] BatchStrOutput
  → TokenizerManager（按 rid 收集，以流式 / 非流式返回）
```

对应源码：`python/sglang/srt/managers/tokenizer_manager.py`、`python/sglang/srt/managers/scheduler.py`、`python/sglang/srt/managers/detokenizer_manager.py`。

## 二、为什么不能合成一个进程：四个层次的原因

### 1）根本原因：绕开 Python GIL

分词 / 反分词是 **CPU 密集的纯 Python 代码**。如果三者挤在一个进程、靠多线程去做，tokenize / detokenize 会和 Scheduler 的 Python 调度、采样后处理**抢同一把 GIL**，把"驱动 GPU 的循环"卡住，导致 GPU 出现空泡。拆成进程后，各自拥有独立的解释器和锁，才能真正并行地跑在多核上，使 CPU 预处理与 GPU 计算互不阻塞。

换句话说：**"为什么要多进程"的准确答案不是逻辑解耦，而是 GIL 逼出来的。**

### 2）性能目标：CPU / GPU 重叠（流水线）

Scheduler 是唯一驱动 GPU 的进程。让它只做"必须它做"的事，并使 CPU 侧的收包 / 分词 / 反分词与 GPU 前向在时间上重叠——`run_batch` 启动前向后不阻塞，结果延后一轮处理，于是第 \(N\) 轮的 CPU 工作叠在第 \(N-1\) 轮的 GPU 前向之上。

这就是 SGLang "zero-overhead 调度" 的延伸：**CPU 开销被 GPU 时间隐藏**。

### 3）结构目标：共享 + 独立扩展

一个 TP 组里只有 **1 个** tokenizer、**1 个** detokenizer，却有 **N 个** Scheduler（每个 rank 一个）。并且只有 `tp_rank=0` 真正与 tokenizer / detokenizer 收发，其余 rank 靠 broadcast 拿到请求、回包也只在 rank0 侧发出：

```python
# python/sglang/srt/managers/scheduler_components/request_receiver.py
def recv_requests(
    self,
) -> List[Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput, Any]]:
    """Receive results at tp_rank = 0 and broadcast it to all other TP ranks.

    中译：对外主入口。在 tp_rank=0 上真正接收请求，再广播给其余所有 TP rank，
          使各 rank 的请求列表一致。
    """
```

拆开之后可以**独立横向扩展**：前端可开多 tokenizer（`multi_tokenizer_mixin`），调度侧可加 DP 控制器、PD 分离，彼此不耦合。

> ⚠️ 注意：这一步的收益是"**避免 N 份重复**（省 CPU / 内存）+ **便于独立扩展**"，属于**降本与可扩展性**，**不是**"提高硬件资源利用效率"。

### 4）工程目标：故障隔离与可维护性

前端崩溃 / 重启不影响持有 GPU 显存与权重的 Scheduler；调度、分词、解码三块可以分别调试、替换（甚至用 Rust 实现 detokenizer / 前端）。

## 三、对延迟的好处

- **不是**让单请求更快——拆分反而**新增了 ZMQ 序列化 + 3 次 IPC 一跳**，空载单请求延迟其实略变差。
- 真正的收益在**高并发 / 大 batch** 下：避免 CPU 环节阻塞导致 GPU 空转，从而**消除首 token 延迟（TTFT）与逐 token 延迟（TPOT）的抖动**，尾延迟尤其明显。

## 四、对吞吐的好处

- GPU 连续运转、无气泡：CPU 的预处理 / 后处理不再占用 GPU 的等待时间；
- 反分词可与下一 batch 的 forward 重叠，形成批级流水线；
- 单轮墙钟时间从"相加"变为"取大"：

\[
\text{单进程/多线程串行：}\;T \approx T_{cpu} + T_{gpu}
\qquad\Rightarrow\qquad
\text{三进程重叠：}\;T \approx \max(T_{cpu},\, T_{gpu})
\]

理想加速比 \(\approx \dfrac{T_{cpu} + T_{gpu}}{\max(T_{cpu},\, T_{gpu})}\)：当 \(T_{cpu} \approx T_{gpu}\) 时接近 \(2\times\)；当 \(T_{cpu} \ll T_{gpu}\) 时，CPU 开销几乎被完全隐藏。

## 五、代价与取舍

拆分并非"纯赚"，需要清醒认识：

- **IPC 开销**：多跳 ZMQ + 序列化 / 反序列化，单请求延迟变差；
- **复杂度**：请求状态要跨三进程维护（`rid_to_state`、`BatchTokenIDOutput` 等），协议改动需三端同步；
- **异常路径更难**：abort / timeout / 权重更新的资源安全收尾更棘手（见 `to_finish` 延迟标记机制）。

## 六、一句话总结

**拆三类进程，本质不是"分工好看"，而是为了：（1）绕开 GIL、（2）让纯 CPU 环节与 GPU 驱动环节真正并行 / 重叠、（3）让 1 个 tokenizer / detokenizer 服务 N 个 Scheduler，从而免于复制、可独立扩展、（4）故障隔离——用跨进程通信开销换 CPU/GPU 重叠与可扩展性。**

对**吞吐**是"隐藏 CPU、GPU 无气泡"的正收益；对**单个请求的延迟**反而略有增加，收益体现在**高并发下不抖动**。
