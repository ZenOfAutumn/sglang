# srt/speculative

## 目录用途
本目录实现 SGLang 的投机解码（speculative decoding）能力，用小模型/草稿机制提前生成候选 token、再由目标模型一次性验证，从而提升解码吞吐。支持 EAGLE / EAGLE3、Standalone（独立草稿模型）、N-gram 等多种算法，并提供草稿树构建、验证、CUDA Graph 加速等公共组件。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `base_spec_worker.py` | 定义 `BaseDraftWorker`、`BaseSpecWorker` 抽象基类，规定草稿/投机 worker 的接口（draft、draft_extend、缓存清理等）。 |
| `draft_utils.py` | `DraftBackendFactory`，根据 server_args 为草稿模型的 decode/extend 阶段创建相应的注意力后端。 |
| `eagle_draft_cuda_graph_runner.py` | EAGLE 草稿模型 decode 阶段的 CUDA Graph 捕获/重放运行器及其输入缓冲区。 |
| `eagle_draft_extend_cuda_graph_runner.py` | EAGLE 草稿模型 extend（prefill 后续）阶段的 CUDA Graph 运行器。 |
| `eagle_info.py` | EAGLE 核心数据结构：`EagleVerifyInput`、`EagleDraftInput`、`EagleVerifyOutput`，承载草稿/验证所需的张量与树形信息。 |
| `eagle_info_v2.py` | EAGLE 信息结构的 v2 Mixin（`EagleDraftInputV2Mixin`、`EagleVerifyInputV2Mixin`）及配套 triton 内核，供 v2 worker 使用。 |
| `eagle_utils.py` | EAGLE 草稿树工具：`organize_draft_results`、`TreeMaskMode`、`build_tree_kernel_efficient`、`verify_tree_greedy_func` 等。 |
| `eagle_worker.py` | `EAGLEWorker`，EAGLE/EAGLE3 投机解码主 worker，串联草稿生成、目标模型验证与接受逻辑。 |
| `eagle_worker_v2.py` | EAGLE worker 的 v2 实现（`EAGLEWorkerV2`、`EagleDraftWorker`），基于 overlap/plan stream 等新架构。 |
| `multi_layer_eagle_draft_extend_cuda_graph_runner.py` | 多层 EAGLE 草稿在 extend 阶段多步推理的 CUDA Graph 运行器。 |
| `multi_layer_eagle_utils.py` | 多层 EAGLE 的 triton 内核工具（input_ids 旋转、隐藏状态池写入等）。 |
| `multi_layer_eagle_worker.py` | 多层 EAGLE 投机解码 worker（v1）。 |
| `multi_layer_eagle_worker_v2.py` | 多层 EAGLE worker 的 v2 实现。 |
| `ngram_info.py` | N-gram 投机的验证输入结构 `NgramVerifyInput` 及相关 triton 内核。 |
| `ngram_worker.py` | `NGRAMWorker`，基于 N-gram 语料匹配生成草稿 token 的投机 worker。 |
| `spec_info.py` | 公共枚举与基类：`SpeculativeAlgorithm`、`SpecInputType`、`SpecInput` 抽象基类。 |
| `spec_utils.py` | 投机解码通用工具集：缓存槽分配、KV 索引生成、top-k 选择、token bitmask、模拟接受、token map 加载、TP 上下文等。 |
| `standalone_worker.py` | `StandaloneWorker`（继承 `EAGLEWorker`），使用独立草稿模型（不与目标模型共享 embedding/lm_head）。 |
| `standalone_worker_v2.py` | Standalone worker 的 v2 实现（`StandaloneDraftWorker` 等）。 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| `cpp_ngram` | N-gram 语料库的 Python 封装，对接底层 C++/JIT 实现。 |
