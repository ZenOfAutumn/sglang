# srt/hardware_backend/npu

## 目录用途
本目录为华为昇腾（Ascend）NPU 提供后端适配，覆盖 KV 缓存内存池、显存分配器、缓存预取（CMO）、NPU 专用工具及参数初始化等基础设施，并通过子目录扩展注意力、图执行、模块、MoE 与量化等能力。其核心思路是在复用 SGLang 通用抽象的基础上，针对 NPU 算子（`torch_npu`）与 NZ 数据格式做替换与优化。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `allocator_npu.py` | NPU 分页 KV 缓存分配器 `NPUPagedTokenToKVPoolAllocator`，继承通用分页分配器并按页对齐调整 `alloc_extend` 等分配逻辑。 |
| `cmo.py` | 缓存管理操作（CMO）工具，提供独立预取流的创建/获取与 `prepare_weight_cache`，在执行其他 AIV/通信核时预取 matmul 权重以重叠访存时间。 |
| `memory_pool_npu.py` | NPU KV 缓存内存池 `NPUMHATokenToKVPool`/`NPUMLATokenToKVPool`，针对 NPU 的 MHA/MLA 缓存布局与 FIA 等做适配。 |
| `utils.py` | NPU 通用工具，定义 `NPUACLFormat`/`FusedMoEMode` 枚举、`npu_format_cast` 格式转换、后端初始化 `init_npu_backend`、默认 `ServerArgs` 设置与索引器权重流等。 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| `attention` | NPU 注意力后端实现（Ascend 后端、Torch 原生 SDPA 后端、MLA 预处理）。 |
| `graph_runner` | NPU 图执行运行器，含 EAGLE 投机解码与 ViT 的图捕获/重放。 |
| `modules` | NPU 专用模型模块适配（DeepSeek-V2 MLA、Qwen-VL 图像预处理补丁）。 |
| `moe` | NPU 上的 MoE 路由（专家 top-k 选择）实现。 |
| `quantization` | NPU 量化方法（线性层与融合 MoE 的 W8A8/W4A4/W4A8/W4A16 等）。 |
