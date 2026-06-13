# srt/lora

## 目录用途
该目录实现 SGLang 的 LoRA(Low-Rank Adaptation)适配器服务能力，借鉴 S-LoRA 与 Punica 思路，支持在单次前向中同时服务成千上万个并发 LoRA 适配器。它负责适配器的加载、显存池管理、淘汰策略、批次元数据构建，以及将基础模型层替换为带 LoRA 计算的层。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `eviction_policy.py` | LoRA 适配器显存淘汰策略：抽象基类 `EvictionPolicy` 及 LRU、FIFO 实现，`get_eviction_policy` 工厂。 |
| `layers.py` | 各类基础并行层的 LoRA 包装层(嵌入、LMHead、Column/Row/QKV/MergedColumn 并行线性、FusedMoE)及 `get_lora_layer` 工厂。 |
| `lora.py` | `LoRAAdapter` / `LoRALayer`：单个 LoRA 适配器的权重容器与按层组织逻辑。 |
| `lora_config.py` | `LoRAConfig`：从路径或字典加载 HuggingFace LoRA 配置与新增 token 配置。 |
| `lora_manager.py` | `LoRAManager`：LoRA 总管，负责初始化、加载/卸载适配器、替换模型层、准备每批次的 `LoRABatchInfo`。 |
| `lora_moe_runners.py` | LoRA 感知的 MoE 运行器(`TritonRunnerCoreWithLoRA` 等)，在 MoE 计算的特定节点注入 LoRA delta。 |
| `lora_overlap_loader.py` | `LoRAOverlapLoader`：用独立 CUDA stream 异步预加载适配器，实现加载与计算重叠。 |
| `lora_registry.py` | `LoRARef` / `LoRARegistry`：适配器引用记录(唯一 lora_id)与并发安全的注册表。 |
| `mem_pool.py` | `LoRAMemoryPool`：LoRA 权重的统一显存池，管理槽位分配、淘汰与按层缓冲。 |
| `utils.py` | `LoRABatchInfo`、`LoRAType` 及目标模块识别、隐藏维度推断、分段长度生成等工具函数。 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| `backend` | 各 LoRA 计算后端(triton、csgmv、torch、ascend、flashinfer)的统一接口与实现。 |
| `torch_ops` | 基于纯 PyTorch 的 LoRA SGEMM 算子实现。 |
| `triton_ops` | 基于 Triton 的 LoRA 计算核(SGMV shrink/expand、embedding、qkv/gate_up、MoE 等)。 |
