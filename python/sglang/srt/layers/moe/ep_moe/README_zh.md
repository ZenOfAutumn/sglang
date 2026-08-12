# `srt/layers/moe/ep_moe`

## 1. 目录用途

本目录提供 SGLang MoE（Mixture of Experts）的专家并行（Expert Parallelism，EP）适配层和配套 Triton 数据整理 kernel。

EP 会把专家参数切分到不同 rank：每个 rank 只保存并计算一部分本地专家。路由器为每个 token 选出 Top-K 专家后，系统需要先通过 All-to-All 后端把 token 发送到专家所在 rank，再按本地专家重新排列 token，执行专家 MLP，最后按原始 token 顺序和路由权重合并结果。本目录位于这条链路的“通信格式与专家计算格式衔接”位置。

其主要职责是：

1. 根据当前 MoE All-to-All 后端选择 EP MoE 层实现；
2. 连接 token dispatcher 的 `dispatch` / `combine` 与 `FusedMoE` 的专家计算；
3. 在普通 DeepEP 和低延迟 DeepEP 数据格式之间做分支处理；
4. 为 DeepGEMM、CUTLASS W4A8 等 runner 准备按专家分组的数据；
5. 提供 scatter、gather、重排、反重排、激活和量化融合等 Triton kernel。

> 本目录不负责计算路由 logits 或执行 Top-K 选择。它接收已经生成的 `TopKOutput`，重点处理路由之后的 token 搬运与专家 MLP。

## 2. 文件清单

| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 包初始化文件，目前没有导出逻辑。 |
| `layer.py` | EP MoE 层入口。定义 `DeepEPMoE`，组织 `dispatch -> run_moe_core -> combine`，并由 `get_moe_impl_class` 根据 All-to-All 后端选择实现。 |
| `kernels.py` | EP 数据布局转换所需的 Triton kernel，包括专家分组、`src2dst` 映射、scatter/gather、DeepGEMM 预处理、CUTLASS W4A8 前后重排、SiLU-and-Mul 与 FP8 量化等。 |

## 3. 整体执行流程

设本 rank 输入 token 数为 \(T\)，隐藏维度为 \(H\)，Top-K 为 \(K\)。路由结果通常包含：

- `hidden_states`：形状约为 \(T \times H\)；
- `topk_ids`：形状为 \(T \times K\)，记录每个 token 选择的专家；
- `topk_weights`：形状为 \(T \times K\)，记录对应路由权重。

`DeepEPMoE.forward_impl` 的主路径为：

```text
hidden_states + TopKOutput
          |
          v
 dispatcher.dispatch
  跨 rank 发送 token 和路由元数据
          |
          v
     DispatchOutput
          |
          v
    run_moe_core
  数据重排 -> 专家 MLP -> 结果反重排
          |
          v
      CombineInput
          |
          v
 dispatcher.combine
  跨 rank 送回来源 rank，并恢复 token 顺序
          |
          v
      原始 token 顺序的输出
```

对于一个 token \(x_t\)，最终 MoE 输出可抽象为：

\[
y_t = s \sum_{j=1}^{K} w_{t,j} E_{e_{t,j}}(x_t),
\]

其中 \(e_{t,j}\) 是第 \(j\) 个命中专家，\(w_{t,j}\) 是路由权重，\(E\) 是专家 MLP，\(s\) 是可选的 `routed_scaling_factor`。具体在哪一阶段乘权重或缩放因子，取决于 dispatcher、runner 和量化实现。

### 3.1 普通 DeepEP 模式

普通模式的 `DeepEPNormalDispatchOutput` 保存通信后收到的 token、可选量化 scale、Top-K 信息及每个本地专家收到的 token 数。数据仍以“收到的 token”为主序，单个 token 可能命中多个本地专家，因此进入连续或分组 GEMM 前需要进一步展开：

1. `ep_scatter` 根据 `num_recv_tokens_per_expert` 计算各专家起始偏移；
2. 将每个有效的 token-expert 对复制到对应专家的连续区间；
3. 生成 `m_indices`，供分组 GEMM 识别每一行属于哪个专家；
4. 生成 `output_index`，记录原 token-expert 对到展开后行号的映射；
5. 专家 MLP 完成后，`ep_gather` 根据 `output_index` 取回各专家结果，以 FP32 累加 `topk_weights`，恢复通信后的 token 顺序；
6. dispatcher 的 `combine` 再把结果发回 token 的来源 rank。

### 3.2 DeepEP 低延迟模式

低延迟模式使用 `DeepEPLLDispatchOutput`。dispatcher 已经把数据整理成 masked grouped GEMM 所需的专家分组布局，并提供：

- `masked_m`：每个本地专家实际拥有的 token 数；
- `expected_m`：runner 用于选择或约束 GEMM 配置的预期 \(M\)；
- 分组后的 `hidden_states` 及其量化 scale。

因此 runner 可直接执行 masked grouped GEMM，不需要普通模式中的 `ep_scatter`。计算结果封装为 `DeepEPLLCombineInput` 后交给低延迟 combine 路径。

### 3.3 标准 dispatcher 到 DeepGEMM

`kernels.py` 也服务于非 DeepEP 的标准 dispatcher。当输入仍是原始 \(T \times H\) 布局时，`moe_ep_deepgemm_preprocess` 会：

1. 稳定排序展平后的 `topk_ids`，让相同专家的 token-expert 对相邻；
2. 计算专家段指针 `seg_indptr` 和真实行数 `masked_m`；
3. 构造 `src2dst`，把原始 token-expert 位置映射到专家分组缓冲区；
4. 按 DeepGEMM masked grouped GEMM 要求分配形状为 \(E_{\mathrm{local}} \times M_{\max} \times H\) 的缓冲区；
5. 对 FP8 输入执行 per-token-group 量化并同步重排 scale；
6. 专家计算结束后通过 `post_reorder_triton_kernel` 恢复 token 顺序并进行 Top-K 加权求和。

## 4. `layer.py` 详解

### 4.1 `DeepEPMoE`

`DeepEPMoE` 继承 `FusedMoE`。基类负责专家权重、量化方法、runner 配置和 dispatcher 等通用设施；该类主要补充 DeepEP/Mooncake/Mori/NIXL 等 EP 通信后端所需的控制流。

#### 初始化与兼容路径

构造函数先调用 `FusedMoE.__init__`，然后计算 `deprecate_flag`。这个名字表示是否退回当前统一的基类实现，并不等同于禁用 EP。当以下情况由新 runner/硬件路径处理时，会设置该标志并直接复用 `FusedMoE`：

- AMD HIP 上启用 AITER；
- NPU；
- 启用 JIT DeepGEMM 的 FP8 或 BF16 dispatch 路径；
- FlashInfer CuteDSL 的 ModelOpt FP4；
- DeepEP 低延迟、DeepGEMM、BF16 无量化组合。

只有未退回基类的旧兼容路径才在本类中记录：

- `use_fp8_w8a8`：FP8 W8A8；
- `use_block_quant`：FP8 block quant；
- `use_w4afp8`：W4A/FP8 激活量化；
- `deepep_mode`：普通或低延迟 DeepEP 模式。

当前本类自有 `run_moe_core` 路径实际仅保留 W4A/FP8 的 CUTLASS 调用；其他组合应进入 `FusedMoE` 的统一 runner 路径。普通模式下的无量化旧路径会显式抛出 `NotImplementedError`。

#### `forward`

`forward` 先检查是否处于 tensor-core piecewise CUDA Graph：

- 是：要求标准格式的 `TopKOutput`，调用 `moe_forward_piecewise_cuda_graph_impl`；
- 否：进入 `forward_impl`。

#### `forward_impl`

- `deprecate_flag=True`：委托 `FusedMoE.forward_impl`；
- 否则按 `dispatcher.dispatch -> run_moe_core -> dispatcher.combine` 执行。

这种拆分使通信层与专家计算层保持独立：dispatcher 决定跨 rank 的数据交换格式，MoE runner 决定本地 GEMM 实现。

#### `dispatch` 与 `combine`

这两个方法是对 `self.dispatcher` 的薄封装，主要供需要拆分执行、通信计算重叠或外部编排的路径调用。`combine` 还接受可选 `overlap_args`。

#### `run_moe_core`

该方法通过 `DispatchOutputChecker` 区分输入格式：

- `deepep_normal`：W4A/FP8 调用 `forward_cutlass_w4afp8`；
- `deepep_ll`：W4A/FP8 调用 `forward_cutlass_w4afp8_masked`；
- 退回基类时，由统一 MoE runner 完成格式转换和专家计算。

结果会分别包装为 `DeepEPNormalCombineInput` 或 `DeepEPLLCombineInput`。包装对象同时携带 `topk_ids` 和 `topk_weights`，供 combine 阶段恢复来源和完成加权。

#### W4A/FP8 入口

- `forward_cutlass_w4afp8`：普通 DeepEP 数据格式，调用 `W4AFp8MoEMethod.apply_deepep_normal`；
- `forward_cutlass_w4afp8_masked`：低延迟格式，调用 `W4AFp8MoEMethod.apply_deepep_ll`。

两条路径都要求激活函数为 SiLU。

### 4.2 `get_moe_impl_class`

该函数根据全局 MoE All-to-All backend 选择类：

| 后端 | 返回类 | 原因 |
| --- | --- | --- |
| DeepEP、Mooncake、Mori、NIXL | `DeepEPMoE` | 它们通过兼容的 dispatcher 接口进入 EP 控制流。 |
| Ascend FuseEP | `FusedMoE` | FuseEP 在 NPU 专用实现中绕过 `FusedMoE.forward` 内部的 dispatch/combine。 |
| 其他后端 | `FusedMoE` | 使用统一 MoE 实现。 |

参数 `quant_config` 当前未参与选择，函数中的注释说明这是临时方案。

## 5. `kernels.py` 详解

该文件中的下划线开头函数通常是 Triton device kernel，非下划线函数通常负责参数检查、输出分配或 launch 配置。

### 5.1 Launch 配置

- `_get_launch_config_1d`：根据设备 SM 数、每 SM 最大线程数和数据规模，为一维任务选择 grid 与 block 大小；
- `_get_launch_config_2d`：为“token 行 \(\times\) hidden 列”类型任务选择二维 grid。

两者限制并发 wave 数，避免对较小任务启动过多 block。

### 5.2 排序与 `src2dst` 映射

`src2dst` 是本文件最关键的中间结构之一。将 `topk_ids` 展平后，原位置可写为：

\[
p = tK + j,
\]

其中 \(t\) 是 token 下标，\(j\) 是 Top-K 槽位。\(\mathrm{src2dst}[p]\) 记录该 token-expert 对在“按专家分组”缓冲区中的目标行号。它既用于前向复制输入，也用于专家计算后的反向取回。

相关函数：

- `compute_src2dst_triton_kernel`：对排序索引求逆置换；
- `deepep_compute_src2dst_triton_kernel`：在求逆置换时扣除无效专家 `-1` 所占前缀；
- `deepep_run_moe_deep_preprocess`：稳定排序 Top-K 专家 ID，生成 `reorder_topk_ids`、`src2dst` 和专家段指针 `seg_indptr`；
- `compute_seg_indptr_triton_kernel`：通过二分查找计算每个专家在排序数组中的边界；
- `compute_masked_m_triton_kernel`：由相邻段指针之差得到每个专家的实际 token 数；
- `cutlass_w4_run_moe_ep_preproess`：为 CUTLASS W4 路径生成简化的 `src2dst`。

这里使用稳定排序，以保证同一专家内部的 token-expert 对顺序可预测。

### 5.3 CUTLASS W4A8 前后重排

- `pre_reorder_for_cutlass_moe` / `pre_reorder_triton_kernel_for_cutlass_moe`：根据 `src2dst` 将输入复制到专家连续布局，并可应用输入量化 scale；用哨兵专家 ID `num_local_experts` 跳过非本地项；
- `silu_mul_static_tensorwise_quant_for_cutlass_moe`：融合 \(\operatorname{SiLU}(g) \odot u\) 与静态 tensor-wise 量化；
- `post_reorder_for_cutlass_moe` / `post_reorder_triton_kernel_for_cutlass_moe`：按 `src2dst` 取回专家输出，乘 `topk_weights`，以 FP32 累加，并应用 `routed_scaling_factor`；
- `deepep_permute_triton_kernel` 与 `deepep_post_reorder_triton_kernel`：为 DeepEP W4A8 路径提供更直接的输入复制和输出加权合并。

### 5.4 SiLU-and-Mul 与量化

MoE 专家第一层通常产生拼接的 gate/up 张量。激活计算为：

\[
\operatorname{SwiGLU}(g,u)=\operatorname{SiLU}(g)\odot u
=\frac{g}{1+e^{-g}}\odot u.
\]

相关实现包括：

- `silu_and_mul_masked_fwd`：只处理 \(\mathrm{masked\_m}[e]\) 指定的有效行，输出 BF16；
- `silu_and_mul_masked_post_quant_fwd`：融合激活与 FP8 per-token-group 量化，可使用 UE8M0 scale；
- `silu_and_mul_masked_post_per_tensor_quant_fwd`：融合激活与 FP8 per-tensor/per-expert scale 量化；
- `fp8_per_token_to_per_tensor_quant_triton`：把已有的 FP8 per-token-group 表示转换为统一 per-tensor scale 表示。

masked kernel 不处理专家缓冲区中的 padding 行，从而避免无效计算。

### 5.5 普通 DeepEP 的 `ep_scatter` / `ep_gather`

`ep_scatter` 分两步：

1. `_fwd_kernel_ep_scatter_1` 对 `num_recv_tokens_per_expert` 做前缀和，生成 `expert_start_loc`，并填充 `m_indices`；
2. `_fwd_kernel_ep_scatter_2` 遍历收到的 token 及其 Top-K 专家，通过原子加法为每个有效 token-expert 对分配专家区间中的目标行，复制激活和可选 FP8 scale，同时写入 `output_index`。

`ep_gather` 执行逆过程：读取 `input_index`（即 scatter 生成的 `output_index`），取得各专家输出，乘路由权重并以 FP32 累加到 token 输出。

这组 kernel 连接了两种布局：

```text
通信布局：received-token 主序
              |
          ep_scatter
              v
GEMM 布局：本地 token-expert 对主序
           按 expert 连续排列
              |
           专家 MLP
              |
          ep_gather
              v
通信布局：received-token 主序
```

### 5.6 DeepGEMM masked grouped GEMM 预处理

`moe_ep_deepgemm_preprocess` 将标准布局转换为 DeepGEMM 输入，返回：

- `masked_m`：每个专家真实行数；
- `expected_m`：平均每专家 token 数的向上取整估计；
- `src2dst`：原始 token-expert 对到三维专家缓冲区扁平行号的映射；
- `gateup_input`：形状为 \(E_{\mathrm{local}} \times M_{\max} \times H\)；
- `gateup_input_scale`：FP8 路径对应的分组 scale，BF16 路径为 `None`。

设当前 token 行数为 \(M\)，则代码取

\[
M_{\max}=256\left(\left\lfloor\frac{M}{256}\right\rfloor+1\right),
\]

以满足 masked grouped GEMM 对 \(M\) 维 block 的要求。`fill_gateup_input_triton_kernel` 负责同时复制激活和 scale。

`tma_align_input_scale` 将二维 scale 转换为符合 DeepGEMM TMA 访问要求的列主序效果，并保证相关 \(M\) 轴地址按 \(16\) 字节对齐。

### 5.7 W4A8 grouped GEMM 元数据

- `compute_problem_sizes_w4a8`：根据 `masked_m` 为每个专家写入两次 grouped GEMM 的 \((N, M, K)\) 问题尺寸；
- `deepep_ll_get_cutlass_w4a8_moe_mm_data`：低延迟 DeepEP 路径的包装函数，并确保问题尺寸为 `int32`。

第一次 GEMM 产生 gate/up 两部分，因此输出维度为 \(2N\)；第二次 GEMM 将激活后的 \(N\) 维中间结果投影回 \(K\)。

### 5.8 Zero expert 辅助逻辑

`zero_experts_compute_triton` 用于带特殊 zero expert 语义的模型：

- 把特殊专家从正常专家路由中屏蔽；
- 当 `zero_expert_type == "identity"` 时，按对应路由权重累加原始 `hidden_states`；
- 返回 zero expert 分支的输出，供模型代码与正常专家输出合并。

当前该函数由 LongCat 模型实现使用。调用方传入的 `expert_indices` 和 `expert_scales` 会被原地修改，使用时需注意其副作用。

## 6. 关键数据结构

| 名称 | 含义 |
| --- | --- |
| `TopKOutput` | 路由输出，核心字段为 `topk_ids`、`topk_weights` 和可选 `router_logits`。 |
| `DispatchOutput` | dispatcher 跨 rank 分发后的统一类型；具体可为普通 DeepEP、低延迟 DeepEP 或标准格式。 |
| `DeepEPNormalDispatchOutput` | 普通 DeepEP 接收格式，包含收到的 token、量化 scale、路由信息和每专家 token 计数。 |
| `DeepEPLLDispatchOutput` | 低延迟格式，通常已按专家组织，并携带 `masked_m`、`expected_m`。 |
| `DeepEPNormalCombineInput` | 普通模式本地专家计算结果及 combine 所需路由信息。 |
| `DeepEPLLCombineInput` | 低延迟模式本地专家计算结果及 combine 所需路由信息。 |
| `seg_indptr` | 长度为 \(E + 1\) 的专家分段指针，第 \(e\) 个专家范围为 \([\mathrm{seg\_indptr}[e],\, \mathrm{seg\_indptr}[e+1])\)。 |
| `masked_m` | 每个专家的有效 token 行数，用于 masked grouped GEMM 跳过 padding。 |
| `m_indices` | 连续 GEMM 输入中每一行所属的本地专家 ID。 |
| `src2dst` | 原始 token-expert 对到专家分组布局的目标位置映射。 |
| `output_index` | 普通 DeepEP scatter 后的逆向索引，gather 用它恢复每个 token 的专家输出。 |

## 7. 与相邻模块的关系

- `moe/fused_moe_triton/layer.py`：提供 `FusedMoE` 基类和统一 MoE runner 主流程；
- `moe/token_dispatcher/`：负责跨 rank dispatch/combine，并定义各种 `DispatchOutput` / `CombineInput`；
- `moe/moe_runner/deep_gemm.py`：调用本目录的预处理、scatter/gather 和激活 kernel，把 dispatcher 格式接到 DeepGEMM；
- `moe/cutlass_w4a8_moe.py`：调用本目录的重排、激活量化和 problem-size kernel，执行 CUTLASS W4A8 grouped GEMM；
- `quantization/w4afp8.py`：实现 `W4AFp8MoEMethod`，由 `DeepEPMoE` 的两条 W4A/FP8 入口调用；
- `hardware_backend/npu/moe/fuseep.py`：Ascend FuseEP 专用路径，不通过 `DeepEPMoE` 的常规 dispatch/combine 控制流。

## 8. 阅读建议

建议按以下顺序理解代码：

1. 先看 `layer.py` 的 `DeepEPMoE.forward_impl`，掌握三段式主流程；
2. 阅读 `token_dispatcher/deepep.py` 中两种 dispatch 输出格式；
3. 阅读 `moe_runner/deep_gemm.py` 中 `deepep_normal`、`deepep_ll` 与 `standard` 的 pre/post permute 注册函数；
4. 回到 `kernels.py`，重点理解 `src2dst`、`ep_scatter`、`ep_gather` 和 `moe_ep_deepgemm_preprocess`；
5. 如需研究 W4A8，再结合 `cutlass_w4a8_moe.py` 和 `quantization/w4afp8.py` 阅读。

调试数据错位问题时，优先检查：

- `topk_ids` 中无效专家的哨兵值是否符合当前路径约定；
- `src2dst` / `output_index` 是否与 Top-K 展平顺序一致；
- `num_recv_tokens_per_expert` 的和是否等于 scatter 后有效行数；
- `masked_m` 是否与专家缓冲区实际有效区域一致；
- FP8 激活与 scale 是否执行了完全相同的重排；
- combine 前是否已经乘过 `topk_weights` 或 `routed_scaling_factor`，避免重复加权。
