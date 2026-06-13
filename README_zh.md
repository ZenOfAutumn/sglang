<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px"></img>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://static.pepy.tech/badge/sglang?period=month)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

</div>

--------------------------------------------------------------------------------

<p align="center">
<a href="https://lmsys.org/blog/"><b>博客</b></a> |
<a href="https://docs.sglang.io/"><b>文档</b></a> |
<a href="https://roadmap.sglang.io/"><b>路线图</b></a> |
<a href="https://slack.sglang.io/"><b>加入 Slack</b></a> |
<a href="https://meet.sglang.io/"><b>每周开发会议</b></a> |
<a href="https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#slides"><b>幻灯片</b></a>
</p>

## 新闻
- [2026/02] 🔥 在 NVIDIA GB300 NVL72 上借助 SGLang 解锁 25 倍推理性能（[博客](https://lmsys.org/blog/2026-02-20-gb300-inferencex/)）。
- [2026/01] 🔥 SGLang Diffusion 加速视频与图像生成（[博客](https://lmsys.org/blog/2026-01-16-sglang-diffusion/)）。
- [2025/12] SGLang 为最新开源模型提供 day-0 支持（[MiMo-V2-Flash](https://lmsys.org/blog/2025-12-16-mimo-v2-flash/)、[Nemotron 3 Nano](https://lmsys.org/blog/2025-12-15-run-nvidia-nemotron-3-nano/)、[Mistral Large 3](https://github.com/sgl-project/sglang/pull/14213)、[LLaDA 2.0 Diffusion LLM](https://lmsys.org/blog/2025-12-19-diffusion-llm/)、[MiniMax M2](https://lmsys.org/blog/2025-11-04-miminmax-m2/)）。
- [2025/10] 🔥 借助 SGLang-Jax 后端，SGLang 现已可原生运行于 TPU（[博客](https://lmsys.org/blog/2025-10-29-sglang-jax/)）。
- [2025/09] 在 GB200 NVL72 上通过 PD 与大规模 EP 部署 DeepSeek（第二部分）：Prefill 提升 3.8 倍，Decode 吞吐提升 4.8 倍（[博客](https://lmsys.org/blog/2025-09-25-gb200-part-2/)）。
- [2025/09] SGLang 为带稀疏注意力的 DeepSeek-V3.2 提供 Day 0 支持（[博客](https://lmsys.org/blog/2025-09-29-deepseek-V32/)）。
- [2025/08] 8/22 SGLang x AMD 旧金山线下聚会：动手 GPU 工作坊、来自 AMD/xAI/SGLang 的技术分享以及社交活动（[路线图](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_sglang_roadmap.pdf)、[大规模 EP](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_sglang_ep.pdf)、[亮点](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_highlights.pdf)、[AITER/MoRI](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_aiter_mori.pdf)、[Wave](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_wave.pdf)）。

<details>
<summary>更多</summary>

- [2025/11] SGLang Diffusion 加速视频与图像生成（[博客](https://lmsys.org/blog/2025-11-07-sglang-diffusion/)）。
- [2025/10] PyTorch Conference 2025 SGLang 演讲（[幻灯片](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/sglang_pytorch_2025.pdf)）。
- [2025/10] 10/2 SGLang x Nvidia 旧金山线下聚会（[回顾](https://x.com/lmsysorg/status/1975339501934510231)）。
- [2025/08] SGLang 为 OpenAI gpt-oss 模型提供 day-0 支持（[说明](https://github.com/sgl-project/sglang/issues/8833)）
- [2025/06] SGLang 作为每日支撑数万亿 token 的高性能服务基础设施，荣获 a16z 第三批开源 AI 资助（[a16z 博客](https://a16z.com/advancing-open-source-ai-through-benchmarks-and-bold-experimentation/)）。
- [2025/05] 在 96 块 H100 GPU 上通过 PD 分离与大规模专家并行部署 DeepSeek（[博客](https://lmsys.org/blog/2025-05-05-large-scale-ep/)）。
- [2025/06] 在 GB200 NVL72 上通过 PD 与大规模 EP 部署 DeepSeek（第一部分）：解码吞吐提升 2.7 倍（[博客](https://lmsys.org/blog/2025-06-16-gb200-part-1/)）。
- [2025/03] 在 AMD Instinct MI300X 上大幅提升 DeepSeek-R1 推理性能（[AMD 博客](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html)）
- [2025/03] SGLang 加入 PyTorch 生态：高效的 LLM 服务引擎（[PyTorch 博客](https://pytorch.org/blog/sglang-joins-pytorch/)）
- [2025/02] 在 AMD Instinct™ MI300X GPU 上解锁 DeepSeek-R1 推理性能（[AMD 博客](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1_Perf/README.html)）
- [2025/01] SGLang 在 NVIDIA 和 AMD GPU 上为 DeepSeek V3/R1 模型提供首日支持，并包含针对 DeepSeek 的专项优化。（[说明](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3)、[AMD 博客](https://www.amd.com/en/developer/resources/technical-articles/amd-instinct-gpus-power-deepseek-v3-revolutionizing-ai-development-with-sglang.html)、[10 余家其他公司](https://x.com/lmsysorg/status/1887262321636221412)）
- [2024/12] v0.4 发布：零开销批调度器、缓存感知负载均衡器、更快的结构化输出（[博客](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)）。
- [2024/10] 首届 SGLang 线上聚会（[幻灯片](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#the-first-sglang-online-meetup)）。
- [2024/09] v0.3 发布：DeepSeek MLA 提速 7 倍、torch.compile 提速 1.5 倍、多图/视频 LLaVA-OneVision（[博客](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)）。
- [2024/07] v0.2 发布：使用 SGLang Runtime 更快地服务 Llama3（对比 TensorRT-LLM、vLLM）（[博客](https://lmsys.org/blog/2024-07-25-sglang-llama3/)）。
- [2024/02] SGLang 借助压缩有限状态机实现 **3 倍更快的 JSON 解码**（[博客](https://lmsys.org/blog/2024-02-05-compressed-fsm/)）。
- [2024/01] SGLang 借助 RadixAttention 实现最高 **5 倍更快的推理**（[博客](https://lmsys.org/blog/2024-01-17-sglang/)）。
- [2024/01] SGLang 为官方 **LLaVA v1.6** 发布演示提供服务支持（[用法](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#demo)）。

</details>

## 关于
SGLang 是面向大语言模型和多模态模型的高性能服务框架。
它旨在从单 GPU 到大规模分布式集群的各种环境中提供低延迟、高吞吐的推理。
其核心特性包括：

- **高速运行时**：提供高效服务，包含用于前缀缓存的 RadixAttention、零开销 CPU 调度器、prefill-decode 分离、投机解码、连续批处理、分页注意力（paged attention）、张量/流水线/专家/数据并行、结构化输出、分块 prefill、量化（FP4/FP8/INT4/AWQ/GPTQ）以及多 LoRA 批处理。
- **广泛的模型支持**：支持众多语言模型（Llama、Qwen、DeepSeek、Kimi、GLM、GPT、Gemma、Mistral 等）、嵌入模型（e5-mistral、gte、mcdse）、奖励模型（Skywork）以及扩散模型（WAN、Qwen-Image），并可轻松扩展以添加新模型。兼容大多数 Hugging Face 模型和 OpenAI API。
- **广泛的硬件支持**：可运行于 NVIDIA GPU（GB200/B300/H100/A100/Spark/5090）、AMD GPU（MI355/MI300）、Intel Xeon CPU、Google TPU、昇腾 NPU 等。
- **活跃的社区**：SGLang 是开源项目，拥有充满活力的社区支持并被业界广泛采用，在全球支撑超过 400,000 块 GPU。
- **强化学习与后训练基石**：SGLang 是经过验证的 rollout 后端，被用于训练众多前沿模型，具备原生 RL 集成，并被诸如 [**AReaL**](https://github.com/inclusionAI/AReaL)、[**Miles**](https://github.com/radixark/miles)、[**slime**](https://github.com/THUDM/slime)、[**Tunix**](https://github.com/google/tunix)、[**verl**](https://github.com/volcengine/verl) 等知名后训练框架所采用。

## 入门
- [安装 SGLang](https://docs.sglang.io/get_started/install.html)
- [快速开始](https://docs.sglang.io/basic_usage/send_request.html)
- [后端教程](https://docs.sglang.io/basic_usage/openai_api_completions.html)
- [前端教程](https://docs.sglang.io/references/frontend/frontend_tutorial.html)
- [贡献指南](https://docs.sglang.io/developer_guide/contribution_guide.html)

## 基准测试与性能
更多内容请参阅发布博客：[v0.2 博客](https://lmsys.org/blog/2024-07-25-sglang-llama3/)、[v0.3 博客](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)、[v0.4 博客](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)、[大规模专家并行](https://lmsys.org/blog/2025-05-05-large-scale-ep/)、[GB200 机柜级并行](https://lmsys.org/blog/2025-09-25-gb200-part-2/)、[GB300 长上下文](https://lmsys.org/blog/2026-02-19-gb300-longctx/)。

## 采用与赞助
SGLang 已大规模部署，每天在生产环境中生成数万亿 token。它受到众多领先企业和机构的信赖与采用，包括 xAI、AMD、NVIDIA、Intel、LinkedIn、Cursor、Oracle Cloud、Google Cloud、Microsoft Azure、AWS、Atlas Cloud、Voltage Park、Nebius、DataCrunch、Novita、InnoMatrix、MIT、UCLA、华盛顿大学、斯坦福大学、加州大学伯克利分校、清华大学、Jam & Tea Studios、Baseten 以及其他主要科技机构。
作为一款开源 LLM 推理引擎，SGLang 已成为事实上的行业标准，在全球超过 400,000 块 GPU 上运行。
SGLang 目前由非营利开源组织 [LMSYS](https://lmsys.org/about/) 托管。

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="logo" width="800" margin="10px"></img>

## 联系我们
如果企业有意大规模采用或部署 SGLang，包括技术咨询、赞助机会或合作咨询，请通过 [sglang@lmsys.org](mailto:sglang@lmsys.org) 联系我们。

长期活跃的 SGLang 贡献者有资格获得编程智能体的赞助，例如 Cursor、Claude Code 或 OpenAI Codex。请将你最重要的提交或 pull request 发送至 [sglang@lmsys.org](mailto:sglang@lmsys.org)。

## 致谢
我们从以下项目中学习了设计理念并复用了代码：[Guidance](https://github.com/guidance-ai/guidance)、[vLLM](https://github.com/vllm-project/vllm)、[LightLLM](https://github.com/ModelTC/lightllm)、[FlashInfer](https://github.com/flashinfer-ai/flashinfer)、[Outlines](https://github.com/outlines-dev/outlines) 以及 [LMQL](https://github.com/eth-sri/lmql)。

