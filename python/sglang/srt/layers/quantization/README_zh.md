# srt/layers/quantization

## 目录用途
本目录是 SGLang 的量化子系统，汇集了各类权重/激活量化方法的顶层实现。每种方法通常提供一个 `QuantizationConfig`（描述量化参数与层匹配规则）以及对应的 `LinearMethod`/`FusedMoEMethod`/`KVCacheMethod`（负责创建权重、加载后处理与前向计算）。此外还包含 FP8/INT8 的 Triton 算子、Marlin 内核工具、标量类型工具以及多家量化框架（compressed-tensors、quark、modelslim）的子模块。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 量化方法注册表，`get_quantization_config` 按名称返回对应 Config；vllm 缺失时提供占位类 |
| `auto_round.py` | AutoRound 量化方法（`AutoRoundConfig`），基于符号梯度的权重舍入量化 |
| `awq.py` | AWQ 激活感知权重量化（`AWQConfig`/`AWQMarlinConfig` 及 Linear/MoE 方法，含 NPU 变体） |
| `awq_triton.py` | AWQ 的 Triton 反量化/GEMM 内核实现 |
| `base_config.py` | 量化抽象基类：`QuantizeMethodBase`/`LinearMethodBase`/`FusedMoEMethodBase`/`QuantizationConfig` |
| `base_scheme.py` | 量化 scheme 抽象基类 `BaseLinearScheme`/`BaseMoEScheme`（供 compressed-tensors 等子模块复用） |
| `bitsandbytes.py` | bitsandbytes 4bit/8bit 量化（`BitsAndBytesConfig` 及 Linear/MoE 方法） |
| `blockwise_int8.py` | 分块 INT8 量化（`BlockInt8Config`，块级 w8a8 INT8） |
| `fp4_utils.py` | FP4 GEMM 后端选择工具（`Fp4GemmRunnerBackend` 枚举与初始化/查询函数） |
| `fp8.py` | FP8 量化核心实现（`Fp8Config` 及 Linear/MoE/KVCache 方法） |
| `fp8_kernel.py` | FP8 的 Triton 量化与块级 GEMM 内核（per-token/per-group 量化等） |
| `fp8_utils.py` | FP8 工具：GEMM 后端调度、cutlass/flashinfer/deepgemm 封装与硬件支持检查 |
| `fpgemm_fp8.py` | FBGEMM FP8 量化方法（`FBGEMMFp8Config` 及 Linear 方法） |
| `gguf.py` | GGUF 量化格式支持（`GGUFConfig` 及 Linear/MoE/Embedding 方法） |
| `gptq.py` | GPTQ 量化（`GPTQConfig`/`GPTQMarlinConfig` 及 Linear/MoE 方法，含 Marlin 与 NPU 变体） |
| `int8_kernel.py` | INT8 的 Triton 量化与块级 GEMM 内核（per-token/per-group） |
| `int8_utils.py` | INT8 工具函数：块级 w8a8 INT8 线性、反量化等 |
| `kv_cache.py` | KV cache 量化基类 `BaseKVCacheMethod`（加载 k_scale/v_scale） |
| `kvfp4_tensor.py` | KV cache 的 FP4(E2M1) 量化工具 `KVFP4QuantizeUtil` |
| `marlin_utils.py` | Marlin 内核通用工具：支持性检查、工作区、scale/bias permute 等 |
| `marlin_utils_fp4.py` | NVFP4 在非 Blackwell GPU 上经 Marlin 内核回退执行的工具 |
| `marlin_utils_fp8.py` | FP8 Marlin 内核工具：层预处理、权重打包、torch 量化等 |
| `modelopt_quant.py` | NVIDIA ModelOpt 量化（`ModelOptQuantConfig`/`ModelOptFp8Config` 等，FP8/NVFP4） |
| `moe_wna16.py` | MoE 的 WNA16（权重 N bit、激活 16bit）量化方法（`MoeWNA16Config`，复用 AWQ/GPTQ） |
| `mxfp4.py` | MXFP4 量化（`Mxfp4Config` 及 MoE 方法，含 flashinfer swizzle 支持） |
| `mxfp4_tensor.py` | MXFP4 张量量化工具 `MXFP4QuantizeUtil` |
| `petit.py` | Petit NVFP4 量化方法（`PetitNvFp4Config` 及 Linear 方法） |
| `petit_utils.py` | Petit NVFP4 内核封装与支持性检查（依赖 petit-kernel） |
| `qoq.py` | QoQ（W4A8 量化）方法（`QoQConfig` 及 Linear 方法） |
| `quark_int4fp8_moe.py` | Quark INT4-FP8 MoE 量化方法（`QuarkInt4Fp8Config` 及 MoE 方法） |
| `rocm_mxfp4_utils.py` | ROCm/AITER 上 MXFP4 的融合量化算子重导出 |
| `unquant.py` | 非量化（全精度）方法：Embedding/Linear/FusedMoE 的默认实现 |
| `utils.py` | 通用量化工具：标量类型、层跳过判断、反量化、参数替换等 |
| `w4afp8.py` | W4A8-FP8 MoE 量化方法（`W4AFp8Config` 及 MoE 方法） |
| `w8a8_fp8.py` | W8A8-FP8 量化（`W8A8Fp8Config` 及 Linear/MoE 方法） |
| `w8a8_int8.py` | W8A8-INT8 量化（`W8A8Int8Config` 及 Linear/MoE 方法） |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| `compressed_tensors` | compressed-tensors 量化框架适配（Config + 多种 scheme） |
| `configs` | 块级量化 GEMM 的调优 JSON 配置（无 .py） |
| `modelslim` | 华为 ModelSlim（NPU）量化框架适配 |
| `quark` | AMD Quark 量化框架适配 |
