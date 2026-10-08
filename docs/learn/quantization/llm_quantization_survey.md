# LLM 量化综述：数值格式、方法族谱与生产选型

> 本文是 `docs/learn/quantization/` 下的**总览性**文档，回答三个问题：
> **① 量化到底在优化什么**（收益来源与上限）；**② 目前有哪几类方法、各自解决什么问题**；
> **③ 面对一个具体模型和一张具体卡，选哪条路线**。
>
> 各方法的深入推导见同目录的分篇：GPTQ 的完整数学推导见 [gptq.md](./gptq.md)；
> 工程接口（Config / Method / apply 生命周期）见 [量化子系统学习计划](quantization_learning_plan_zh.md)；
> 启动参数与平台支持矩阵见 [Quantization 使用文档](../../advanced_features/quantization_zh.md)
> 与 [KV Cache 量化](../../advanced_features/quantized_kv_cache_zh.md)。

## 目录

1. [收益来源：量化在优化什么](#1-收益来源量化在优化什么)
2. [数值格式：从一个 FP16 权重到 4 bit](#2-数值格式从一个-fp16-权重到-4-bit)
3. [量化的四个基本要素](#3-量化的四个基本要素)
4. [方法族谱：六个正交维度](#4-方法族谱六个正交维度)
5. [W4A16：当前生产的绝对主力](#5-w4a16当前生产的绝对主力)
6. [W8A8：激活离群值之战](#6-w8a8激活离群值之战)
7. [FP4 与 4 bit 激活：Blackwell 时代](#7-fp4-与-4-bit-激活blackwell-时代)
8. [KV Cache 量化：长上下文的第二战场](#8-kv-cache-量化长上下文的第二战场)
9. [低比特极限与 QAT](#9-低比特极限与-qat)
10. [内核与硬件支持矩阵](#10-内核与硬件支持矩阵)
11. [精度评估与常见坑](#11-精度评估与常见坑)
12. [选型决策树](#12-选型决策树)
13. [SGLang 中的落地映射](#13-sglang-中的落地映射)
14. [趋势与开放问题](#14-趋势与开放问题)
15. [附录：术语表与参考文献](#15-附录术语表与参考文献)

---

## 1. 收益来源：量化在优化什么

### 1.1 三段式负载，三种瓶颈

LLM 推理不是一个同质的负载，量化对每个阶段的收益来源完全不同：

| 阶段 | 计算特征 | 瓶颈 | 量化的主要收益 |
| --- | --- | --- | --- |
| Prefill（处理 prompt） | 大 GEMM，算术强度高 | **算力受限**（compute-bound） | 低精度 Tensor Core 峰值更高（FP8 = 2×FP16，FP4 = 4×FP16） |
| Decode（逐 token 生成） | 每 token 一次 skinny GEMM，算术强度 ≈ 1 | **访存受限**（memory-bound） | 权重位宽降低 → 每次读的字节数降低 |
| 长上下文 / 大 batch | KV cache 读写量占主导 | **显存容量 + 带宽** | KV cache 量化（与权重解耦的独立收益） |

这一点决定了两个直觉上反常识的结论：

- **权重 4 bit 化对 prefill 几乎没用**（prefill 的瓶颈不是权重字节数），它的收益几乎全在 decode 与显存容量上；
- **W8A8 / W4A4 对 decode 的收益不如对 prefill**，因为 decode 时激活本身很小，省不了多少带宽，真正的收益是 Tensor Core 吞吐。

### 1.2 收益上限

Decode 阶段每生成一个 token 需要把全部权重从 HBM 读一遍（batch 较小时），因此：

$$
\text{speedup}_{\text{decode}} \;\lesssim\; \frac{16\ \text{bit}}{\text{权重有效位宽}}
$$

代入实际值：

| 权重量化 | 名义位宽 | 含元数据的有效位宽（g=128） | 理论加速 | 实测典型加速 |
| --- | --- | --- | --- | --- |
| FP16/BF16 基线 | 16 | 16 | 1.0× | 1.0× |
| INT8 / FP8 | 8 | ≈ 8.3 | ~1.9× | 1.3–1.7× |
| INT4（W4A16） | 4 | ≈ 4.16 | ~3.8× | 2.0–3.0× |
| NVFP4（W4A4） | 4 | ≈ 4.5 | ~3.5× | 2.5–3.5×（Blackwell） |

达不到理论上限的原因：KV cache 读写、激活与中间张量的访存、反量化开销、kernel launch 与同步、以及 decode 的 batch 增大后权重访存被摊薄（转向 compute-bound）。

**有效位宽必须把元数据算进去**——这就是为什么 group size 不是越小越好，见 §3.4。

---

## 2. 数值格式：从一个 FP16 权重到 4 bit

### 2.1 三种表示哲学

| 类型 | 表示 | 动态范围 | 优点 | 缺点 |
| --- | --- | --- | --- | --- |
| 整数（INT8/INT4） | 均匀网格 + scale（+zero-point） | 窄，靠 scale 拉伸 | 位运算便宜、无精度浪费 | 对离群值极其敏感，需要分组 |
| 浮点（FP8/FP6/FP4） | 符号 + 指数 + 尾数 | 宽，自带指数 | 抗离群值、动态范围大 | 网格非均匀，尾数位少时量化噪声大 |
| 微缩放浮点（MX/NVFPx） | 浮点 + **共享指数/scale** 的块 | 宽且自适应 | 兼具浮点动态范围与块级精度 | 元数据开销、需要专用内核 |

### 2.2 必须记住的几种格式

| 格式 | 位宽 | 结构 | 动态范围 | 典型用途 |
| --- | --- | --- | --- | --- |
| BF16 / FP16 | 16 | 1-8-7 / 1-5-10 | 大 / 中 | 基线 |
| **E4M3** | 8 | 1-4-3 | ±448 | 权重与激活（主流 FP8） |
| **E5M2** | 8 | 1-5-2 | ±57344 | 梯度、KV cache（需大动态范围时） |
| E8M0 | 8 | 纯指数，无尾数 | $2^{-127}\sim2^{127}$ | MX 系列的**块 scale**（只能是 2 的幂） |
| FP6（E3M2 / E2M3） | 6 | — | 中等 | 新兴，介于 FP8 与 FP4 之间 |
| **E2M1（FP4）** | 4 | 1-2-1 | 仅 16 个码值：$\{0,\pm0.5,\pm1,\pm1.5,\pm2,\pm3,\pm4,\pm6\}$ | 权重与激活（Blackwell） |
| INT8 | 8 | 均匀 | 由 scale 决定 | 老卡 W8A8 |
| INT4 | 4 | 均匀 16 级 | 由 scale 决定 | W4A16 权重 |

E2M1 的码值空间非常稀疏——**单独用 FP4 表示权重而不加块级 scale 是不可用的**，这就是 MX / NVFP4 存在的理由。

### 2.3 微缩放（microscaling）：MX 与 NVFP4

两者都是「FP4 数据 + 块级 scale」，但设计哲学不同：

| 方案 | 数据 | 块大小 | 块 scale | 第二级 scale | 提出方 |
| --- | --- | --- | --- | --- | --- |
| **MXFP4 / MXFP8**（OCP MX） | E2M1 / E4M3 | 32 | E8M0（2 的幂） | 无 | OCP 开放标准 |
| **NVFP4** | E2M1 | 16 | E4M3 | 每 tensor 一个 FP32 | NVIDIA |
| **MXFP8** in SGLang | E4M3 | 32 | E8M0 | 无 | OCP |

工程含义：

- MX 的 E8M0 scale 只能取 $2^k$，**没有尾数 ⇒ 存在最多 ~1.5 倍的 scale 舍入误差**，但换来了极简的硬件实现；
- NVFP4 用 E4M3 scale（有尾数）精度更高，块更小（16），代价是元数据占比更大（4 bit 数据 + 8 bit scale 每 16 个元素 ≈ 4.5 bit/元素）；
- 两者的**位宽开销都在 4.5 bit/元素附近**，即"4 bit"实际上不可达。

### 2.4 硬件支持：格式是被芯片定义的

| 架构 | 代表卡 | INT8 | FP8 | FP4/FP6 | INT4 路线的实际做法 |
| --- | --- | --- | --- | --- | --- |
| Ampere SM80/86 | A100 / A10 / 3090 | ✅ | ❌ | ❌ | 无原生 INT4 MMA，靠 kernel 内解包+反量化走 FP16 Tensor Core |
| Ada SM89 | L40S / 4090 | ✅ | ✅ | ❌ | 同上 |
| Hopper SM90 | H100 / H200 | ✅ | ✅（+ TMA） | ❌ | 同上（Marlin 的主场） |
| Blackwell SM100/120 | B200 / GB200 / 5090 | ✅ | ✅ | ✅ | FP4 原生，W4A16 反而变成过渡方案 |
| CDNA3/4 | MI300X / MI350X | ✅ | ✅（Aiter） | ✅（MXFP4） | — |
| Ascend A2/A3 | 910B | ✅ | WIP | ✅（MXFP4） | — |

一个实用推论：**格式选择首先由硬件决定，其次才由精度决定**。H100 上讨论 W4A4 意义不大（没有 FP4 单元），B200 上死守 W4A16 也在浪费 FP4 算力。

---

## 3. 量化的四个基本要素

无论是哪篇论文，量化方案都可以用四个正交的旋钮描述。

### 3.1 对称 vs 非对称

对称量化（只有一个 scale）：

$$
x_q = \mathrm{round}\!\left(\frac{x}{s}\right), \qquad \hat{x} = s \cdot x_q, \qquad s = \frac{\max|x|}{q_{\max}}
$$

非对称量化（多一个 zero-point $z$）：

$$
x_q = \mathrm{round}\!\left(\frac{x}{s}\right) + z, \qquad \hat{x} = s \cdot (x_q - z)
$$

| 维度 | 对称 | 非对称 |
| --- | --- | --- |
| 元数据 | 1 个 scale | scale + zero-point |
| GEMM | $\hat{Y} = s_x s_w (X_qW_q^\top)$，无修正项 | 需补 $-z_x\sum W_q - z_w \sum X_q + n z_x z_w$ |
| 内核友好度 | 高（INT8/FP8 Tensor Core 直接可用） | 低 |
| 适配分布 | 近似关于 0 对称（权重常见） | 偏斜分布（ReLU 后激活、长尾） |
| 代表 | FP8（e4m3 天然有符号）、GPTQ 权重量化 | 传统 INT8 W8A8、部分 GPTQ/AWQ 变体 |

**FP8 天然是对称的、没有 zero-point**——这是 FP8 能在 Hopper 上"零成本"落地的最重要原因：格式自带符号位，反量化就是一次乘法。

数值示例（含逐元素对照表）见 [gptq.md §2.3](./gptq.md)。

### 3.2 粒度（granularity）

| 粒度 | scale 个数 | 精度 | 元数据开销 | 典型场景 |
| --- | --- | --- | --- | --- |
| per-tensor | 1 | 最低 | ~0 | 老式 INT8、FP8 权重（不够用） |
| per-channel / per-row | $d_{\text{out}}$ | 中 | 小 | W8A8-INT8 权重、AWQ |
| per-token（动态） | $N$（每个 token） | 高 | 小 | 激活侧的标准做法 |
| per-group | $d/g$，$g$ 通常 128 | 高 | 中（~4%） | GPTQ / AWQ W4A16 |
| block-wise 128×128 | 每块一个 | 高 | 中 | DeepSeek 风格 FP8 |
| per-head / per-token（KV） | 每头/每 token | — | — | KV cache 量化 |

### 3.3 静态 vs 动态

- **静态**：scale 离线算好写进 checkpoint。加载快、无运行时开销，但对分布变化不鲁棒。
- **动态（online）**：前向时实时算 scale。鲁棒，但每步多一次 reduction，且 kernel 难以完全融合。
- 实践中往往是**权重静态 + 激活动态**（`W8A8-FP8 per-token` 就是这一组合的标准答案）。

### 3.4 分组大小与元数据开销

以 $4096\times4096$ 层、INT4 为例：

| group size | 有效 bit/权重 | 相对 FP16 压缩 |
| --- | --- | --- |
| per-row | 4.005 | 4.00× |
| **128** | **4.156** | **3.85×** |
| 64 | 4.312 | 3.71× |
| 32 | 4.625 | 3.46× |

**group 越小精度越高但压缩率越低**，128 是工业界的甜点：(a) 元数据开销 < 5%；(b) 能有效隔离离群值；(c) 对齐主流 kernel 的 tiling。更激进（g=64/32）只用于 2–3 bit 或对精度极敏感的场景。

---

## 4. 方法族谱：六个正交维度

| 维度 | 取值 | 典型代表 |
| --- | --- | --- |
| ① 量化对象 | 权重 / 激活 / KV cache / 三者任意组合 | W4A16、W8A8、KVFP8 |
| ② 位宽组合 | W16A16 … W4A4、混合 | W8A8-FP8、W4A16、W4A8、W4A4 |
| ③ 阶段 | PTQ（训练后）/ QAT（训练中） | GPTQ vs LLM-QAT |
| ④ 时机 | 静态 / 动态（online） | GPTQ 离线 vs `--quantization fp8` 在线 |
| ⑤ 目标函数 | 最小化权重误差 / 输出误差 / 端到端 loss | 直接 round vs GPTQ / QAT |
| ⑥ 手段 | 舍入优化 / 误差补偿 / 等价变换 / 码本 / 旋转 | GPTQ / AWQ / SmoothQuant / AQLM / QuaRot |

**"方法"通常是这六个维度的组合**。例如 GPTQ = (权重) × (W4A16) × (PTQ) × (静态) × (最小化输出误差) × (逐列误差补偿)。

按 ② 位宽组合排列的族谱：

```text
                        激活位宽
            16bit        8bit         4bit
        ┌───────────┬────────────┬────────────┐
  16bit │  (基线)   │            │            │
        ├───────────┼────────────┼────────────┤
   8bit │           │ SMOOTHQUANT│            │
权      │           │ FP8 W8A8   │            │
重      │           │ INT8 W8A8  │            │
位      ├───────────┼────────────┼────────────┤
宽 4bit │ GPTQ/AWQ  │ W4A8       │ W4A4       │
        │ HQQ/WNA16 │ (MoE 常见) │ NVFP4/MXFP4│
        │ ← 主力 →  │            │ ← 前沿 →   │
        ├───────────┼────────────┼────────────┤
   2bit │ AQLM/QUIP#/PV-Tuning（需码本）        │
        ├───────────┼────────────┼────────────┤
  ~1bit │ BitNet b1.58（需从零训练）            │
        └───────────┴────────────┴────────────┘
```

**当下（生产环境）的"流行度排序"**：

1. **W4A16（GPTQ / AWQ + Marlin）**——通用 GPU 上部署量化模型的事实标准，显存收益最大、生态最全；
2. **W8A8-FP8（含 block-wise）**——Hopper 及以上服务端首选，精度损失几乎可忽略，DeepSeek 系列原生就是这个格式；
3. **FP8/FP4 KV cache**——长上下文场景的标配，与权重方案解耦；
4. **W4A4 / NVFP4**——Blackwell 上正在起量；
5. **W2A16 / QAT**——实验室与端侧，尚未成为服务端主流。

---

## 5. W4A16：当前生产的绝对主力

### 5.1 为什么是 4 bit 权重

- 显存收益直接且巨大（~3.85×），7B 模型 14GB → 3.8GB，单张消费卡即可服务；
- 精度上 4 bit 是"无损区间"的边界：4 bit + group 128 通常 PPL 增加 < 0.1；
- 激活保持 16 bit ⇒ **不需要处理激活离群值**这个最难的问题；
- 代价：需要专用解包/反量化 kernel（Marlin 等），且无法利用低精度 Tensor Core 峰值。

### 5.2 两大方法：GPTQ 与 AWQ

| 维度 | GPTQ | AWQ |
| --- | --- | --- |
| 核心思想 | 逐列量化 + 逆 Hessian 误差补偿 | 保护激活显著的权重通道 + per-channel scale 搜索 |
| 是否需要校准数据 | 是（通常 128 条） | 是（更少，通常 128 条以内） |
| 是否需要反向传播 | 否（闭式解） | 否（网格搜索，可视为搜索式） |
| 目标 | 最小化**输出重建误差** | 最小化**激活感知的权重重要性加权误差** |
| 关键超参 | `group_size`、`desc_act`（act-order） | `group_size`、`zero_point`、搜索网格 |
| 额外机制 | act-order 按激活方差重排列顺序 | salient channel 的 per-channel scale |
| 弱点 | 对离群权重只能重分配误差 | 搜索成本随粒度上升 |
| 精度体感 | 4 bit 下两者接近，AWQ 在部分模型/任务略优 | 同左 |
| 是否可叠加 | 可（AWQ 的 scale 与 GPTQ 的补偿正交） | 可 |

**实践建议**：不要纠结"哪个更好"，两者在 4 bit + g128 下差异通常在噪声范围内。真正影响精度的是 `group_size`、是否量化 `lm_head`/`down_proj`、以及校准集是否覆盖真实分布。

### 5.3 WNA16：把两者统一到 MoE

`moe_wna16.py`（`MoeWNA16Config`）的做法是：MoE 的专家权重沿用 AWQ/GPTQ 的权重格式，通过 `get_scaled_act_names()` 区分 AWQ（需要激活预缩放）与 GPTQ。理解点在于——**MoE 的量化方法几乎都是 Linear 方法 + MoE 调度壳**。

### 5.4 零校准方案：HQQ

HQQ（Half-Quadratic Quantization）用半二次分裂求解，**不需要校准数据**，在 4 bit 下逼近 GPTQ，在无法获取代表性校准数据（内部模型、多模态、稀疏激活）时非常实用。它揭示的一个事实：GPTQ 的收益有相当一部分来自"per-group 尺度 + 误差补偿"，而非校准数据本身。

---

## 6. W8A8：激活离群值之战

### 6.1 为什么激活比权重难量化

LLM 的激活存在**系统性离群值**：少数通道的数值可达其余通道的 100 倍以上（与 LayerNorm、attention sink 等机制相关）。这导致 per-tensor 激活量化的 scale 被离群值撑大，其余通道信息几乎全丢。

一个粗略的对比：权重分布近似高斯（量化损失~均匀），激活分布是**长尾+离群**（量化损失集中在绝大多数正常通道上）。

### 6.2 三条解决路线

| 路线 | 思路 | 代表 | 效果 |
| --- | --- | --- | --- |
| **等价变换** | 把激活的量化难度搬到权重上 | SmoothQuant（$\hat X = X\,\mathrm{diag}(s)^{-1}$，$\hat W = \mathrm{diag}(s)W$，$s_j = \max|x_j|^\alpha / \max|w_j|^{1-\alpha}$） | W8A8 可用，几乎无精度损失 |
| **混合精度分解** | 离群通道单独走 FP16 | LLM.int8()（99.9% INT8 通道 + 0.1% 离群通道 FP16） | 精度保住，但 kernel 复杂、加速有限 |
| **分组/分块** | 把 scale 粒度做细 | per-token 动态、128×128 block FP8 | 当前最主流的工程解 |

### 6.3 FP8 W8A8：目前的"无损"方案

DeepSeek-V3 开启的 **block-wise FP8** 已成为事实标准：

$$
W: \text{FP8 E4M3} + \text{每 } 128\times128 \text{ 块一个 scale}
\qquad\quad
X: \text{FP8 E4M3} + \text{per-token per-128-channel scale}
$$

| 粒度组合 | 精度 | 说明 |
| --- | --- | --- |
| per-tensor W + per-tensor X | 差 | 只在早期 FP8 试验中使用 |
| per-channel W + per-token X | 好 | `--quantization w8a8_fp8` 的标准路径 |
| **block 128×128 W + per-token X** | 最好 | DeepSeek 风格，权重 scale 细到块级 |

关键点：**权重是静态的，可以离线统计分布以切成小 scale；激活是动态的，只能用 per-token 在线量化**。这条不对称性是理解所有 W8A8 实现的钥匙。

### 6.4 INT8 W8A8 为什么还在

- 兼容性：Ampere 之前的卡只有 INT8；
- 存储优势：INT8 + zero-point 在某些场景比 FP8 有更好的分布贴合；
- 成本：Triton 实现（`blockwise_int8.py`）跨平台可用，不依赖特定 Tensor Core。

缺点同样明显：zero-point 带来的 GEMM 修正项、离群值敏感性、以及 Triton GEMM 相对 CUTLASS/DeepGEMM 的性能差距。

---

## 7. FP4 与 4 bit 激活：Blackwell 时代

### 7.1 4 bit 激活为什么突然可行了

W4A4 之所以从"不可行"变成"前沿"，是三个因素叠加：

1. **格式**：FP4（E2M1）自带指数，动态范围远好于 INT4，配合块级 scale 能贴住激活分布；
2. **硬件**：Blackwell 原生 FP4 MMA，吞吐是 FP8 的 2 倍；
3. **算法**：块级 scale 与旋转类方法（Hadamard）把激活离群值压下去。

### 7.2 两条技术路线

| 路线 | 思路 | 代表 |
| --- | --- | --- |
| **格式/缩放** | 用块级 scale 让 FP4 贴合分布 | NVFP4（ModelOpt、Petit）、MXFP4 |
| **旋转重分布** | 用正交变换（Hadamard）把离群能量打散，使分布变"高斯" | QuaRot、SpinQuant |

旋转类方法的数学内核：对权重做 $W' = WQ$、对激活做 $x' = Q^\top x$（$Q$ 为正交阵，$Q^\top Q = I$），GEMM 结果不变，但**激活的离群通道被均匀摊到所有通道上**。它把"量化难题"转化为"在哪里插入 Hadamard 变换且不引入额外开销"，因此能与任意格式叠加。

### 7.3 混合精度：现实中的折中

真正的生产系统很少全模型单一精度，常见组合：

| 组合 | 动机 |
| --- | --- |
| W4A8（`w4afp8.py`、`quark_int4fp8_moe`） | 权重压到 4 bit 省显存，激活留 8 bit 保精度（MoE 场景特别常见） |
| NVFP4 + FP8 混合 | 敏感层（attention、首末层）用 FP8，MLP 用 FP4 |
| W4A16 + KV FP8 | 最稳的组合，几乎所有部署都在用 |

**MoE 是低比特量化的最佳试验场**：参数量大（省显存收益最大）、单个专家被激活的频率低（对量化误差的容忍度更高）、且专家之间可以做不同精度的自适应分配。

---

## 8. KV Cache 量化：长上下文的第二战场

### 8.1 为什么它是独立战场

权重是固定大小，KV cache 随上下文线性增长。在 128K 上下文下，KV cache 经常**超过权重本身**：

```text
L 层 × 2(K,V) × H_kv 头 × d_head × seq_len × batch × 2 字节

Llama-3-8B（32 层, 8 个 KV 头, d_head=128）：
  1 token ≈ 32 × 2 × 8 × 128 × 2 B = 128 KiB
  128K 上下文单请求 ≈ 16 GiB   ← 与权重（16GB）相当
```

所以**权重压到 4 bit 之后，长上下文的显存瓶颈就转移到 KV cache 上**。这是"权重-激活-KV 三段式量化"的由来。

### 8.2 量化粒度：K 和 V 要用不同粒度

| 对象 | 推荐粒度 | 原因 |
| --- | --- | --- |
| **Key** | per-channel（沿 head_dim） | K 的通道方向存在系统性离群值，且沿序列方向统计稳定 |
| **Value** | per-token（沿序列） | V 的离群值分布在少数 token 上（如 attention sink） |

这一"K 按通道、V 按 token"的不对称设计来自 KIVI，是所有 KV 量化方案的共同起点。SGLang 侧由 `kv_cache.py` 的 `BaseKVCacheMethod` 加载 `k_scale`/`v_scale`，实际量化/反量化发生在 attention backend 消费 `k_scale_float` 的位置。

### 8.3 可用格式（SGLang `--kv-cache-dtype`）

| 取值 | 位宽 | 说明 |
| --- | --- | --- |
| `auto` | 跟随模型 | 默认 |
| `bf16` / `bfloat16` | 16 | 显式关闭 KV 量化 |
| `fp8_e4m3` | 8 | **推荐**，精度与收益平衡最好 |
| `fp8_e5m2` | 8 | 动态范围更大，精度略低 |
| `mxfp8` | 8 | 仅 FA4 后端支持，MX 块 scale |
| `nvfp4` | 4 | NVFP4 recipe，需 CUDA 12.8+ / PyTorch 2.8+ |
| `fp4_mx_block16` / `fp4_e2m1` | 4 | MX 风格 block-16 FP4 |

**注意**：KV 量化收益是"用精度换容量"，与权重方案的收益完全独立且可叠加。工程上最常见的组合是 `W4A16 + KV FP8`。

### 8.4 容易踩的坑

- **量化 KV 后 attention backend 必须支持对应的 scale 消费路径**，否则会静默走回高精度或直接报错；
- **per-tensor vs per-head scale**：checkpoint 里存的 scale 粒度与后端期望的粒度不一致时精度会明显下降（`k_scale = -1.0` 表示"未提供、需要现算"）；
- **投机解码的 draft KV** 是独立的一池（`--speculative-draft-kv-cache-dtype`），需要单独评估。

---

## 9. 低比特极限与 QAT

### 9.1 PTQ 的边界

| 位宽 | PTQ 可行性 | 需要的额外手段 |
| --- | --- | --- |
| 8 bit | 轻松 | 无（或仅需 per-token 动态） |
| 4 bit 权重 | 成熟（PPL 几乎无损） | group scale + 误差补偿 |
| 3 bit 权重 | 可用 | act-order、更大校准集、混合位宽 |
| 2 bit 权重 | 困难 | **码本**（AQLM、QuIP#、PV-Tuning）或 QAT |
| ≤1.58 bit | 基本不可行（PTQ） | **必须从零训练/大规模 QAT**（BitNet） |

### 9.2 两条突破 2 bit 的路线

**① 非均匀码本（改变"网格"本身）**

- **AQLM（Additive Quantization）**：把权重表示为多个码本向量的和，一个 2 bit 索引指向一个码本条目——等价于"网格点由数据学出来"；
- **QuIP#**：随机 Hadamard 旋转（incoherence 处理）+ E8 格点码本；
- **PV-Tuning**：在 2 bit 下通过少量参数微调显著提升精度。

代价是反量化变成查表/向量加，**必须专用 kernel**，通用性差。

**② QAT（把量化写进训练目标）**

- **LLM-QAT**：数据无关蒸馏 + 伪量化；
- **EfficientQAT**：分块训练，把 QAT 成本压到可接受范围；
- **BitNet b1.58**：权重三值 $\{-1,0,1\}$，**乘加退化为加减**，但这要求模型**从一开始就按低比特架构训练**。

**结论**：≤2 bit 已经不是"量化"问题，而是"模型架构与训练方案"问题。生产环境目前不适用。

---

## 10. 内核与硬件支持矩阵

### 10.1 量化必须配内核

一个量化方案能不能用，取决于**是否存在高效的解包-计算内核**。这是"论文方法"与"可用方法"之间的分水岭。

| 内核族 | 服务的方案 | 平台 | 思路 |
| --- | --- | --- | --- |
| **Marlin / Marlin-2** | GPTQ、AWQ、WNA16（W4A16） | CUDA（Ampere+） | 权重重排 + 寄存器内解包，走 FP16 Tensor Core，接近 FP16 GEMM 吞吐 |
| **ExLlamaV2** | GPTQ/AWQ | CUDA | 早期主力，与 Marlin 竞争 |
| **AWQ Triton** | AWQ | 跨平台 | 反量化融合进 GEMM，兼容性好 |
| **CUTLASS / sgl-kernel** | FP8、INT8、NVFP4 | CUDA | 官方拼图，block-wise FP8 的主力 |
| **DeepGEMM** | block FP8（含 MoE） | SM90/SM100 | JIT 编译，针对 DeepSeek 风格布局优化 |
| **FlashInfer** | FP8 / NVFP4 | SM90/SM100/120 | 多种后端（TRT-LLM、CUTLASS、cuDNN） |
| **Triton** | 几乎所有方案的兜底 | 全平台 | 兼容性最好、性能次优 |
| **Aiter** | FP8 / MXFP4 | ROCm | AMD 侧对应实现 |

**Triton 的角色**：它保证了新方案在没有任何专用内核时也能跑起来（`w8a8_int8`、`blockwise_int8` 都是这条路），但性能通常比专用内核低 20–40%。SGLang 的 `--fp8-gemm-backend` / `--fp4-gemm-backend` 就是把这层选择显式暴露给用户。

### 10.2 端到端性能的常见误判

- **省了显存不等于省了时间**：W4A16 在 prefill 阶段可能比 FP16 更慢（解包开销 + 走同样的 FP16 Tensor Core）；
- **量化后吞吐上不去**：常见原因是反量化被放在 GEMM 之外（多一次读写），或者 scale 内存访问没有合并；
- **MoE 量化的收益被调度抵消**：专家 kernel 变小后 launch 开销占比升高。

---

## 11. 精度评估与常见坑

### 11.1 评估层次（缺一不可）

| 层次 | 指标 | 说明 |
| --- | --- | --- |
| 数值 | 权重 MSE、输出相对误差 | 只用于方法调试 |
| 语言建模 | PPL（WikiText-2 / C4） | **最容易过关**，也最不敏感 |
| 分布 | 与 FP16 的 KL divergence | 比 PPL 更能反映退化 |
| 任务 | lm-eval-harness（MMLU / GSM8K / HumanEval） | 推理任务最先暴露问题 |
| 长上下文 | RULER、NIAH | KV 量化必须测 |
| 端到端 | TTFT / ITL / throughput / 显存 | 收益是否真实存在 |

**经验规律**：PPL 几乎不变而推理任务大幅掉点，是量化出问题最常见的形态。只测 PPL 是自欺欺人。

### 11.2 高频坑清单

| 坑 | 现象 | 处理 |
| --- | --- | --- |
| 校准集不匹配 | 特定领域掉点严重 | 用与线上分布一致的语料，128 条以上 |
| 量化了 `lm_head` / `embedding` | 输出退化明显 | 跳过首末层，`--quantization` 侧的忽略规则 |
| MoE router 被量化 | 路由漂移、专家塌缩 | 跳过 `mlp.gate`（SGLang 文档明确提到该限制） |
| 混合位宽 | 与 QKV 融合冲突，加载报错 | 避免同一融合层内混用位宽 |
| group size 不对齐 | 加载报错或走慢路径 | 对齐 128（kernel tiling 要求） |
| KV 量化后没换 backend | 静默走回高精度 | 确认 attention backend 支持对应 scale |
| 用 FP8 模型又加 `--quantization` | 双重量化 | 预量化模型不要再指定量化方法 |
| 只看权重压缩比 | 实际显存没降 | 把 KV cache、元数据、激活峰值一起算 |

---

## 12. 选型决策树

```text
Q1  模型已经是预量化 checkpoint？ （看到 FP8/GPTQ/AWQ/NVFP4 字样）
    └─ 是 → 直接加载，不要加 --quantization
    └─ 否 → Q2

Q2  硬件是？
    ├─ Blackwell（SM100/120）→ 考虑 NVFP4 / W4A8，甚至 W4A4
    ├─ Hopper/Ada  （≥SM89） → FP8 可用，首选 W8A8-FP8 或 W4A16+Marlin
    ├─ Ampere     （SM80/86）→ 无 FP8：W4A16+Marlin 或 INT8 W8A8
    └─ 其他（ROCm/NPU/CPU）→ 看平台支持矩阵，通常 AWQ/GPTQ 或厂商框架

Q3  瓶颈是什么？
    ├─ 显存不够（权重）  → W4A16（GPTQ/AWQ，group 128）
    ├─ 显存不够（长上下文）→ KV FP8（+ 权重方案不变）
    ├─ 吞吐低（prefill 为主）→ W8A8-FP8 或 FP4（要吃 Tensor Core 峰值）
    └─ 延迟高（decode 为主）→ W4A16 或无权重量的 KV 优化

Q4  精度能接受多少损失？
    ├─ 完全不能接受 → FP8 W8A8（per-channel/per-token 或 block）
    ├─ 可接受 <0.1 PPL → W4A16 + g128
    └─ 愿意做验证与调参 → W4A8 / W4A4 / block FP8 细粒度

Q5  是 MoE？ → 优先 W4A8 / FP8；注意跳过 gate，专家 kernel 尺寸对齐
```

**场景速查**：

| 场景 | 推荐 | 备注 |
| --- | --- | --- |
| 通用 GPU 上部署 7B–70B | `W4A16`（AWQ 或 GPTQ，g128）+ `--kv-cache-dtype fp8_e4m3` | 生态最全，效果最稳 |
| H100/H200 上跑 DeepSeek 系列 | 直接用官方 FP8 block checkpoint | 无需再加参数 |
| H100 上跑稠密模型追吞吐 | `w8a8_fp8`（per-channel W + per-token X） | prefill 收益最大 |
| 128K 长上下文服务 | 权重 W4A16/FP8 + KV FP8，必要时 FP4 KV | 容量优先 |
| B200/GB200 新部署 | NVFP4（`modelopt_fp4`） | 硬件利用率最高 |
| 内部模型 / 无校准数据 | HQQ 或 FP8 在线量化 | 免校准 |
| 端侧 / 极低显存 | W4A16 + g32、GGUF（Q4_K_M） | 精度优先于速度 |

---

## 13. SGLang 中的落地映射

### 13.1 `--quantization` → 实现

| 方法名 | Config / 实现位置 | 范式 | 一句话说明 |
| --- | --- | --- | --- |
| `fp8`、`mxfp8` | `fp8.py` | W8A8-FP8 | 核心 FP8 实现，支持 per-tensor 与 128×128 block |
| `w8a8_fp8` | `w8a8_fp8.py` | W8A8-FP8 | 走 sgl-kernel CUTLASS FP8 路径 |
| `w8a8_int8` | `w8a8_int8.py` | W8A8-INT8 | per-channel W + per-token X |
| `blockwise_int8` | `blockwise_int8.py` | W8A8-INT8 | Triton 块级 INT8，跨平台 |
| `modelopt` / `modelopt_fp8` | `modelopt_quant.py` | W8A8-FP8 | NVIDIA ModelOpt checkpoint，自动探测 |
| `modelopt_fp4` | `modelopt_quant.py` | NVFP4 | Blackwell 原生 FP4 |
| `modelopt_mixed` | `modelopt_quant.py` | 混合 | 逐层不同精度 |
| `nvfp4_online` | `nvfp4_online.py` | NVFP4 | 在线重量化（FP8 → NVFP4） |
| `petit_nvfp4` | `petit.py` | NVFP4 | ROCm 侧 NVFP4 |
| `gptq` / `gptq_marlin` | `gptq/`（含 `schemes/`） | W4A16 | Marlin / Triton / CPU / NPU 多 scheme |
| `awq` / `awq_marlin` | `awq/`（含 `schemes/`） | W4A16 | 含 NPU/XPU 变体 |
| `moe_wna16` | `moe_wna16.py` | W4A16（MoE） | 复用 AWQ/GPTQ 服务 MoE |
| `w4afp8` | `w4afp8.py` | W4A8（MoE） | DeepSeek 风格 W4A8 |
| `quark_int4fp8_moe` | `quark_int4fp8_moe.py` | W4A8（MoE） | 在线 INT4→FP8，AMD |
| `mxfp4` | `mxfp4.py`、`npu_mxfp4_w4a4.py` | MXFP4 / W4A4 | CUDA/CPU 走 MoE 路径，NPU 走 W4A4 |
| `mxfp_w4a8` | `npu_mxfp4.py` | W4A8 | NPU |
| `gguf` | `gguf.py` | 多种 | llama.cpp 生态格式 |
| `bitsandbytes` | `bitsandbytes.py` | 4/8 bit | 含 LLM.int8 混合精度分解 |
| `compressed-tensors` | `compressed_tensors/` | 多种 | vLLM 生态 checkpoint，从配置动态选 scheme |
| `quark` / `quark_mxfp4` | `quark/` | 多种 | AMD Quark |
| `modelslim` | `modelslim/` | W8A8 / W4A4 | 华为 NPU |
| `auto-round` / `auto-round-int8` | `auto_round.py` / `w8a8_int8.py` | W4A16 / W8A8 | Intel AutoRound |
| `humming` | `humming.py` | — | 第三方内核封装 |
| `mlx_q4` / `mlx_q8` | `mlx.py` | 4/8 bit | Apple Metal |

### 13.2 相关启动参数

| 参数 | 作用 |
| --- | --- |
| `--quantization <name>` | 指定量化方法（预量化模型**不要**指定） |
| `--kv-cache-dtype <dtype>` | KV cache 格式，见 §8.3 |
| `--fp8-gemm-backend` | block-wise FP8 GEMM 后端：`auto / deep_gemm / flashinfer_* / cutlass / triton / aiter` |
| `--fp4-gemm-backend` | NVFP4 GEMM 后端：`auto / cutlass / flashinfer_cutlass / flashinfer_cudnn / flashinfer_trtllm` |
| `--torchao-config` | torchao 在线量化（`int4wo-128`、`int8wo`、`fp8wo` 等） |
| `--speculative-draft-kv-cache-dtype` | 投机解码 draft KV 的独立格式 |

### 13.3 延伸阅读

- 方法级数学推导：[gptq.md](./gptq.md)
- 子系统代码结构与分阶段学习：[quantization_learning_plan_zh.md](quantization_learning_plan_zh.md)
- 使用与平台支持矩阵：[advanced_features/quantization_zh.md](../../advanced_features/quantization_zh.md)
- KV cache 量化：[advanced_features/quantized_kv_cache_zh.md](../../advanced_features/quantized_kv_cache_zh.md)

---

## 14. 趋势与开放问题

1. **主战场从权重转向激活与 KV**。权重 4 bit 已经"解决"，端到端收益的边际来源正转移到 KV cache（长上下文）和激活（prefill 吞吐）上。
2. **格式标准化之争**：OCP MX（块 32、E8M0）vs NVIDIA NVFP4（块 16、E4M3 + 两级 scale）。胜负由硬件与生态共同决定，短期内会共存。
3. **旋转类方法（QuaRot/SpinQuant）的工程化**：数学上很优雅（Hadamard 使分布高斯化），但插入点选择与 kernel 融合仍在演进，决定 W4A4 能否真正普及。
4. **量化与架构协同设计**：BitNet 证明"为低比特而生的架构"比"给已有架构套量化"上限更高；MLA、稀疏 attention 等新结构对量化的敏感度尚缺系统结论。
5. **量化 × 其他优化技术的交互**：投机解码（draft 模型的精度要求）、PD 分离（prefill 与 decode 的最优量化方案不同）、DP/EP 下的量化通信格式（FP8 通信已在路上）。
6. **评估方法滞后于方法本身**：PPL 已不足以区分方案优劣，长上下文、MoE 路由稳定性、多轮对话退化需要更系统的基准。

---

## 15. 附录：术语表与参考文献

### 15.1 术语速查

| 术语 | 含义 |
| --- | --- |
| WxAy | 权重 x bit、激活 y bit |
| PTQ / QAT | 训练后量化 / 训练中量化 |
| PTQ 校准集 | 用于统计 scale/Hessian 的少量真实文本 |
| online quant | 前向时动态计算激活 scale |
| group size | 共享一套 scale 的权重个数（通常 128） |
| block-wise | 二维分块共享 scale（如 128×128） |
| zero-point | 非对称量化的零点偏移 |
| dequant | 反量化：$\hat{x} = s(x_q - z)$ |
| repack | checkpoint 布局 → 内核所需布局（Marlin 的重排） |
| MX / NVFPx | 两种微缩放浮点标准 |
| E4M3 / E5M2 / E8M0 / E2M1 | FP8/FP4 的位域结构（指数位-尾数位） |
| act-order / `desc_act` | 按激活方差降序重排 GPTQ 的列处理顺序 |
| headroom | 优化空间（profiling 语境） |

### 15.2 关键论文与工具

| 主题 | 参考 |
| --- | --- |
| GPTQ | Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, ICLR 2023 |
| OBQ / OBS 前身 | Frantar & Alistarh 2022；LeCun et al., *Optimal Brain Damage / Surgeon*, 1990 |
| AWQ | Lin et al., *AWQ: Activation-aware Weight Quantization*, MLSys 2024 |
| SmoothQuant | Xiao et al., *SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs*, ICML 2023 |
| LLM.int8() | Dettmers et al., 2022 |
| KV cache 量化 | Liu et al., *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*, 2024 |
| 低比特码本 | AQLM（2024）、QuIP#（2024）、PV-Tuning（2025） |
| 旋转类 | QuaRot（2024）、SpinQuant（2024） |
| 1.58 bit | Ma et al., *The Era of 1-bit LLMs: BitNet b1.58*, 2024 |
| 免校准 | HQQ（2023） |
| FP8 规范 | OCP 8-bit Floating Point Specification（E4M3/E5M2） |
| 微缩放规范 | OCP Microscaling Formats (MX) Specification |
| 工程栈 | Marlin / CUTLASS / DeepGEMM / FlashInfer / Aiter / Triton |
