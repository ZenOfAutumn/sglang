> 来源: <https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/>
> 抓取时间: 2026-09-01 ｜ 格式: 网页转 Markdown

[‹ 返回博客](/blog)

[‹ 返回博客](/blog)

目录

  * 零开销批处理调度器
  * 缓存感知负载均衡器
  * 面向 DeepSeek 模型的数据并行注意力
  * 基于 XGrammar 的快速结构化输出
  * 致谢
  * 路线图

# SGLang v0.4：零开销批处理调度器、缓存感知负载均衡器、更快的结构化输出

SGLang 团队 2024 年 12 月 4 日

我们很高兴发布 [SGLang v0.4](https://github.com/sgl-project/sglang)，本版本带来了显著的性能提升和新特性：

  * 零开销批处理调度器：吞吐量提升 1.1 倍。
  * 缓存感知负载均衡器：吞吐量最高提升 1.9 倍，缓存命中率提高 3.8 倍。
  * 面向 DeepSeek 模型的数据并行注意力：解码吞吐量最高提升 1.9 倍。
  * 基于 xgrammar 的快速结构化输出：最高快 10 倍。

本博客将逐一介绍这些更新。欢迎大家提供反馈和贡献！

## 零开销批处理调度器

虽然 LLM 推理在 GPU 上运行，但仍有大量工作需要由 CPU 完成，例如批处理调度、内存分配和前缀匹配。一个未经优化的推理引擎可能会把[多达一半的时间花在 CPU 开销上](https://mlsys.wuklab.io/posts/scheduling_overhead/)。SGLang 从一开始就以高效的批处理调度器著称。在这个新版本中，我们将其推向极致，实现了近乎零开销的批处理调度器。这个思路很简单，并且已在 [NanoFlow](https://arxiv.org/abs/2408.12757) 中被提出。基本上，我们可以让 CPU 调度与 GPU 计算重叠执行。调度器提前一个批次运行，并为下一个批次准备好所有所需的元数据。这样一来，我们可以让 GPU 始终保持繁忙，并隐藏诸如 radix cache 操作等昂贵的开销。相关代码在[这里](https://github.com/sgl-project/sglang/blob/85e1a6f3aa5a2288ca85fe3fe922c733b6533fa7/python/sglang/srt/managers/scheduler.py#L399)。实现细节包括通过创建 future token 来解决依赖问题，并精心调度 CUDA 事件与同步。下图展示了 CPU 调度器与 GPU worker 的重叠执行示意。

我们使用 Nsight 性能分析系统验证了零开销的说法。在下图中，有 5 个连续的解码批次，可以看到 GPU 上没有任何一处空闲时间。（注意：该性能分析结果是使用 Triton 注意力后端获得的；FlashInfer 后端仍存在微小的间隙，将在下一个 FlashInfer 版本中解决。）

通过这项优化，SGLang v0.4 现在可以榨干 GPU 的最后一点性能，相较于上一版本实现了 1.1 倍的加速，相较于其他最先进的基线实现了 1.3 倍的加速。这一加速在小模型和大张量并行规模上最为显著。

**使用方式**：该功能默认开启，因此你无需做任何更改！

**复现基准测试**：
[code]
    # 零开销批处理调度器 (v0.4)
    python3 -m sglang.launch_server --model meta-llama/Llama-3.2-3B-Instruct
    python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 500 --random-input 4096 --random-output 2048

    # 旧版批处理调度器 (v0.3)
    python3 -m sglang.launch_server --model meta-llama/Llama-3.2-3B-Instruct --disable-overlap
    python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 500 --random-input 4096 --random-output 2048

[/code]

## 缓存感知负载均衡器

SGLang v0.4 为 LLM 推理引擎引入了缓存感知负载均衡器。该负载均衡器会预测各 worker 上的前缀 KV 缓存命中率，并选择匹配率最高的 worker。测试显示**吞吐量最高提升 1.9 倍，命中率提高 3.8 倍**，且收益随 worker 数量增加而扩大。下图展示了在数据并行场景下，缓存感知负载均衡器与朴素的轮询（round-robin）负载均衡器有何不同。缓存感知负载均衡器维护了一棵对 worker 上实际 radix 树的近似 radix 树。该树以惰性方式更新，几乎没有开销。

以下是一些基准测试结果。新的缓存感知路由器显著提升了吞吐量。

| SGLang v0.3| SGLang v0.4
---|---|---
吞吐量 (token/s)| 82665| 158596
缓存命中率| 20%| 75%

> 该基准测试是在一个具有多个长前缀分组的[工作负载](https://github.com/sgl-project/sglang/pull/1990)上进行的，且每个分组之间完全均衡。性能可能会因工作负载的特性而有所不同，但它应能显著提高缓存命中率

该路由器的主要特性包括

  * **多节点支持**：可将 worker 部署在多台机器上，用单个路由器连接分布式 worker，从而在分布式环境中轻松实现水平扩展，同时保持缓存感知能力。
  * **缓存感知路由**：请求会被发送到命中率更高的 worker，同时执行负载均衡以避免失衡。
  * **免通信设计**：无需为缓存状态同步 worker；而是利用传递的信息来模拟一棵“近似树”。
  * **高性能实现**：以纯 Rust 构建，支持高并发，采用低开销设计，相比基于 Python 的方案提速 2 倍。
  * **独立软件包**：以 "sglang-router" 的名称发布，包含 Python 绑定，并提供便于使用的 CLI 接口。

### 使用方式

安装：
[code]
    pip install sglang-router

[/code]

  1. 同时启动 Worker 和路由器

可直接替换现有的 --dp-size 参数：
[code]
    python -m sglang_router.launch_server \
      --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
      --dp-size 8

[/code]

  2. 仅启动路由器 —— 适合多节点分布式处理：

[code]
    python -m sglang_router.launch_router \
      --worker-urls http://worker1:8000 http://worker2:8000

[/code]

### 复现基准测试：
[code]
    # 硬件: 8x A100 80GB GPU
    # 运行基准测试
    python bench_serving.py \
     --host 127.0.0.1 \
     --port 30000 \
     --dataset-name generated-shared-prefix

    # 使用路由器启动
    python -m sglang_router.launch_server \
     --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
     --dp-size 8

    # 不使用路由器启动（基线）
    python -m sglang.launch_server \
     --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
     --dp-size 8

[/code]

可以阅读[深入解析帖](https://x.com/hsu_byron/status/1864449841239347341)了解更多。还有一篇相关论文（采用了不同的设计与实现）[Preble](https://arxiv.org/abs/2407.00023)，同样构建在 SGLang 之上。

## 面向 DeepSeek 模型的数据并行注意力

推理中最常见的并行策略是张量并行。然而，对某些模型来说，它可能并非最高效的策略。例如，DeepSeek 模型使用 MLA 且只有一个 KV head。如果我们在 8 张 GPU 上使用张量并行，将导致 KV 缓存重复和不必要的内存占用。

为了解决这个问题，我们为多头潜在注意力（MLA）机制实现了数据并行（DP），以提升 DeepSeek 模型的吞吐量。通过对注意力组件采用 DP，KV 缓存被显著减少，从而允许更大的批次大小。在我们的 DP 注意力实现中，每个 DP worker 独立处理不同类型的批次（prefill、decode、idle）。经注意力处理后的数据会在进入混合专家（MoE）层之前在所有 worker 之间进行 all-gather，经过 MoE 处理后，数据再被重新分发回各个 worker。下图展示了这一思路。

以下是在 8 x H100 80GB GPU 上的基准测试结果。通过这项优化，SGLang v0.4 相比 SGLang v0.3 实现了 1.9 倍的解码吞吐量。我们正在通过为 MoE 层集成专家并行来进一步提升吞吐量。你可以查看相关的 PR：[数据并行](https://github.com/sgl-project/sglang/pull/1970)和[专家并行](https://github.com/sgl-project/sglang/pull/2203)。

**使用方式：** 添加 `--enable-dp-attention` 选项即可开启此特性。目前仅支持 DeepSeek 模型。

**复现基准测试：**
[code]
    # 硬件: 8x H100 80GB GPU
    # 如果遇到显存不足（out-of-memory），请尝试将 `--mem-fraction-static` 调低到较小的值，例如 0.75。

    # SGLang 启用 DP 注意力 (v0.4)
    python3 -m sglang.launch_server --model-path neuralmagic/DeepSeek-Coder-V2-Instruct-FP8 --disable-radix-cache --trust-remote-code --tp 8 --enable-dp-attention --mem-fraction-static 0.78
    python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 1 --random-output 512 --random-range-ratio 1 --num-prompts 10000

    # SGLang 未启用 DP 注意力 (v0.3)
    python3 -m sglang.launch_server --model-path neuralmagic/DeepSeek-Coder-V2-Instruct-FP8 --disable-radix-cache --trust-remote-code --tp 8 --mem-fraction-static 0.78
    python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 1 --random-output 512 --random-range-ratio 1 --num-prompts 10000

[/code]

## 基于 XGrammar 的快速结构化输出

凭借其[压缩有限状态机](https://lmsys.org/blog/2024-02-05-compressed-fsm/)，SGLang 一直是 JSON 解码速度最快的推理引擎。在这个新版本中，通过集成更快的语法后端 xgrammar，它变得更快了。
根据基准测试结果，**在 JSON 解码任务上，SGLang + xgrammar 可以比其他开源方案快最多 10 倍**。你可以在 xgrammar 的博客文章中了解更多：
<https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar>。

**使用方式**：启动服务器时添加 `--grammar-backend xgrammar`。
[code]
    python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --grammar-backend xgrammar

[/code]

然后你可以通过 OpenAI 兼容的 API 进行查询。示例见 <https://sgl-project.github.io/backend/openai_api_completions.html#JSON>。

## 致谢

本博客文章中的工作主要由 Byron Hsu、Ke Bao、Lianmin Zheng、Yineng Zhang 和 Ziyi Xu 贡献。我们感谢 Zhiqiang Xie、Liangsheng Yin、Shuo Yang 和 Yilong Zhao 就零开销调度器进行的讨论；感谢 Ying Sheng、Yichuan Wang 和 Shiyi Cao 就缓存感知负载均衡器进行的讨论；感谢 Jiashi Li 就数据并行注意力进行的讨论；以及感谢 Yixin Dong 出色的 xgrammar 库。

## 路线图

这是丰收的一年，我们按照[路线图](https://github.com/sgl-project/sglang/issues/1487)交付了众多特性。社区也在健康成长，开发者和采用者不断增加。下一个版本的重点将是 prefill-decode 分离、投机解码、多级 radix cache、序列并行等等！
