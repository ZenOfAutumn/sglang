# Tensor Core 与 CUDA Core：原理、区别与选型

> 本文面向 CUDA、深度学习与大模型推理开发者，系统说明 NVIDIA GPU 中 CUDA Core 与 Tensor Core 的定位、执行方式、数据类型、性能边界及实际使用方法。
>
> 先给出最重要的结论：**CUDA Core 是面向通用标量/向量算术的执行资源，Tensor Core 是面向小矩阵乘加（MMA）的专用执行资源。二者不是互相替代的两类 GPU，也通常不是由线程直接“选择核心”；CUDA 线程组成 warp，warp 发出不同类型的指令，由 SM 内对应的执行流水线完成。**

## 目录

1. [一句话理解](#1-一句话理解)
2. [先理解 GPU 的执行层次](#2-先理解-gpu-的执行层次)
3. [CUDA Core 是什么](#3-cuda-core-是什么)
4. [Tensor Core 是什么](#4-tensor-core-是什么)
5. [核心区别](#5-核心区别)
6. [为什么 Tensor Core 特别适合深度学习](#6-为什么-tensor-core-特别适合深度学习)
7. [数据类型与混合精度](#7-数据类型与混合精度)
8. [编程模型与调用路径](#8-编程模型与调用路径)
9. [一个 CUDA kernel 如何同时使用两者](#9-一个-cuda-kernel-如何同时使用两者)
10. [理论性能为什么不等于实际性能](#10-理论性能为什么不等于实际性能)
11. [大模型训练与推理中的典型分工](#11-大模型训练与推理中的典型分工)
12. [如何确认程序是否使用了 Tensor Core](#12-如何确认程序是否使用了-tensor-core)
13. [常见误区](#13-常见误区)
14. [选型建议](#14-选型建议)
15. [总结](#15-总结)

---

## 1. 一句话理解

可以把一个流式多处理器（Streaming Multiprocessor，SM）类比成一间计算车间：

- **CUDA Core** 像大量通用工位，擅长逐元素加、减、乘、乘加、整数运算和控制相关计算；
- **Tensor Core** 像专用矩阵流水线，一次接收多个矩阵分块，集中完成大量乘加；
- **warp scheduler** 像调度员，它向不同执行流水线发出指令；
- **寄存器、共享内存和显存**负责提供操作数并接收结果。

如果计算是：

$$
y_i = \max(0, x_i + b_i)
$$

它主要是逐元素运算，通常由普通算术流水线执行。

如果计算是：

$$
D = A B + C
$$

并且矩阵尺寸、布局、数据类型满足要求，就可以拆成许多小矩阵乘加，由 Tensor Core 高吞吐执行。

---

## 2. 先理解 GPU 的执行层次

讨论两种 Core 前，需要区分 CUDA 编程模型与硬件结构中的几个层次。

### 2.1 从软件到硬件

典型映射关系如下：

```text
CUDA Grid
  └── Thread Block
        └── Warp（通常包含 32 个线程）
              └── 指令流
                    ├── 普通 FP / INT 指令 → 标量算术流水线
                    ├── MMA 指令          → Tensor Core 流水线
                    ├── Load / Store      → 访存单元
                    └── 特殊函数           → SFU 等单元
```

CUDA 程序员编写的是线程级代码，但 GPU 通常以 warp 为基本调度单位。同一 warp 中的线程共同执行一条指令，只是各线程操作的数据不同。

### 2.2 “Core”不等于独立 CPU 核

“CUDA Core”是 NVIDIA 用来描述 GPU 算术能力的术语，但它不应被理解为一个完整 CPU 核：

- 它没有独立的指令流、缓存层级和操作系统线程；
- 指令由 warp scheduler 调度；
- 多个线程以 SIMT（Single Instruction, Multiple Threads）方式协同执行；
- 一个 SM 内还包含 Tensor Core、Load/Store Unit、SFU、寄存器文件、共享内存等资源。

因此，“某张 GPU 有多少个 CUDA Core”只能粗略描述其算术资源规模，不能单独决定真实性能。

### 2.3 “使用哪种 Core”通常由指令决定

开发者一般不会把某个 CUDA 线程绑定到某一颗物理 Core。更准确的描述是：

1. kernel 中的代码被编译为 GPU 指令；
2. warp scheduler 发射这些指令；
3. 普通浮点/整数指令进入相应算术流水线；
4. MMA 类指令进入 Tensor Core 流水线；
5. Load/Store 等指令进入其他专用单元。

所以，Tensor Core 与 CUDA Core 的区别首先是**指令语义和执行流水线的区别**。

---

## 3. CUDA Core 是什么

CUDA Core 可以近似理解为 NVIDIA GPU 中执行通用标量算术指令的基本运算资源。

### 3.1 主要适合的运算

常见操作包括：

- 浮点加法、乘法、融合乘加（FMA）；
- 整数加法、乘法、位运算；
- 地址计算与索引计算；
- 比较、谓词与部分控制逻辑；
- 向量逐元素计算；
- 不能自然映射成矩阵乘加的问题。

以融合乘加为例：

$$
d = a \times b + c
$$

FMA 在一次指令语义中完成乘法和加法，且通常只进行一次舍入。大量逐元素 CUDA 代码最终都会包含这类普通算术指令。

### 3.2 CUDA Core 的优势

- **通用性强**：可以表达几乎所有数值算法中的基础算术；
- **粒度细**：适合每个线程处理一个或少量元素；
- **形状约束少**：不要求数据组织为特定矩阵 tile；
- **适合控制与辅助计算**：如索引、边界判断、归约前处理。

### 3.3 CUDA Core 的局限

面对大规模矩阵乘法时，普通标量 FMA 的单条指令完成的工作较少。即使并行启动大量线程，也需要发出大量指令，消耗更多指令带宽和调度资源。

矩阵乘法：

$$
C_{ij} = \sum_{k=0}^{K-1} A_{ik}B_{kj}
$$

如果完全展开为标量运算，每个输出元素都需要执行约 $K$ 次乘加。Tensor Core 的意义就是把一组结构化的标量乘加合并为矩阵级指令。

### 3.4 术语上的精确性

不同 GPU 架构对 FP32、FP64、INT32 等流水线的组织并不完全相同。日常所说的“CUDA Core”经常主要指 FP32 标量算术单元，但官方规格、架构图和性能指标可能按不同口径统计。

因此，分析具体芯片时应以对应架构白皮书和指令吞吐表为准，不要假设：

- 所有 CUDA Core 都支持完全相同的数据类型；
- FP32、FP64 和 INT32 的峰值吞吐比例固定；
- 两代 GPU 的 Core 数量可以直接横向比较性能。

---

## 4. Tensor Core 是什么

Tensor Core 是 NVIDIA 从 Volta 架构开始引入的矩阵乘加专用执行单元，核心目标是加速深度学习、高性能计算和其他以稠密或结构化矩阵运算为主的工作负载。

### 4.1 核心操作：MMA

Tensor Core 的基本抽象是 MMA（Matrix Multiply-Accumulate）：

$$
D = A B + C
$$

其中 $A$、$B$、$C$、$D$ 是小矩阵分块或逻辑矩阵片段。大矩阵乘法会被切分成大量 tile，再由多个 warp 或 warp group 协同完成。

需要强调：Tensor Core 不是“一次直接计算完整的大矩阵”，而是高效执行许多固定或受约束形状的小矩阵乘加。软件栈负责：

1. 将大矩阵分块；
2. 把 tile 搬运到寄存器或共享内存；
3. 发出 MMA 指令；
4. 累加部分结果；
5. 写回并执行后处理。

### 4.2 为什么吞吐高

一条 MMA 指令表达了大量相互关联的乘加运算。相比逐个发出标量 FMA，它可以：

- 用更少的指令描述更多算术工作；
- 利用专门设计的数据通路；
- 复用矩阵 tile 中的输入元素；
- 提高单位周期内的乘加数量；
- 降低部分指令调度与解码开销。

若 $A \in \mathbb{R}^{M\times K}$、$B \in \mathbb{R}^{K\times N}$，计算 $AB$ 通常包含：

$$
M N K
$$

次乘法和近似相同数量的加法。按常见 FLOP 计数方式，一次乘加记作 2 FLOPs，因此矩阵乘法的计算量约为：

$$
2 M N K \text{ FLOPs}
$$

Tensor Core 能在少量矩阵指令中承载大量这样的乘加。

### 4.3 代际差异

不同代 Tensor Core 支持的数据类型、指令形状、稀疏能力和调度方式不同，例如可能涉及：

- FP16、BF16；
- TF32；
- FP8 及不同 FP8 编码；
- INT8、INT4 等低精度整数；
- 部分架构上的 FP64 Tensor 运算；
- 结构化稀疏矩阵乘法；
- warp-level MMA、异步 MMA 或 warp-group MMA。

这些能力不能脱离具体 Compute Capability 与架构讨论。某种数据类型能存储，不代表某代 Tensor Core 一定能以硬件原生方式执行对应 MMA。

---

## 5. 核心区别

| 维度 | CUDA Core | Tensor Core |
|---|---|---|
| 主要定位 | 通用标量算术 | 专用矩阵乘加 |
| 典型操作 | ADD、MUL、FMA、整数与逻辑运算 | MMA：$D=AB+C$ |
| 操作粒度 | 线程级标量或短向量 | warp / warp group 协作的矩阵 tile |
| 通用性 | 高 | 相对低，需要满足矩阵运算模式 |
| 形状约束 | 较少 | 通常有 tile、对齐和布局要求 |
| 数据类型 | FP、INT 等通用类型，依架构而异 | 重点支持低精度输入和较高精度累加，依架构而异 |
| 峰值吞吐 | 通常低于同精度 Tensor Core 矩阵吞吐 | 对受支持的矩阵运算非常高 |
| 典型场景 | 逐元素算子、索引、控制、归约、后处理 | GEMM、卷积、Attention 中的矩阵乘法 |
| 指令示例 | 标量/向量 FMA 等 | `mma`、`wmma`、`wgmma` 等家族 |
| 性能关键 | occupancy、分支、访存、指令吞吐 | tile、数据复用、对齐、流水线、矩阵规模 |

最本质的差别不是“精度高低”，而是：

- CUDA Core 以**通用标量算术**为中心；
- Tensor Core 以**结构化矩阵乘加**为中心。

---

## 6. 为什么 Tensor Core 特别适合深度学习

### 6.1 神经网络的大头是矩阵乘法

线性层可写为：

$$
Y = XW + b
$$

卷积通常也可以转换或映射为矩阵乘加。Transformer 中常见的大计算包括：

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

$$
S = \frac{QK^T}{\sqrt{d_k}}
$$

$$
O = \operatorname{softmax}(S)V
$$

以及 MLP：

$$
H = \phi(XW_1)W_2
$$

这些算子中的线性投影和矩阵乘法都可以映射到 MMA，因此 Tensor Core 能覆盖训练和推理中的主要计算量。

### 6.2 深度学习通常能容忍适度低精度

很多神经网络不需要每一步都使用 FP32 输入。实践中常使用：

- FP16 / BF16 输入，FP32 累加；
- TF32 乘法语义，FP32 累加；
- FP8 输入配合缩放因子与较高精度累加；
- INT8 / INT4 权重量化或权重激活量化。

低精度带来三个直接收益：

1. 单位存储空间容纳更多元素；
2. 显存与片上存储的传输压力降低；
3. Tensor Core 可以提供更高矩阵吞吐。

### 6.3 低精度不等于低质量

最终数值质量取决于完整的数值方案，而不是单独一个数据类型：

- 累加精度；
- 缩放策略；
- 溢出与下溢处理；
- loss scaling；
- 量化粒度；
- 校准数据；
- 敏感层是否保留高精度。

因此，使用 Tensor Core 时常见的是**混合精度**，而不是把所有计算无差别降到最低精度。

---

## 7. 数据类型与混合精度

### 7.1 输入精度与累加精度是两件事

矩阵乘加可以抽象为：

$$
D_{ij} = C_{ij} + \sum_k A_{ik}B_{kj}
$$

这里至少有三类精度需要区分：

- $A$、$B$ 的输入精度；
- 乘法中间结果的有效精度；
- $C$、$D$ 的累加精度。

例如“FP16 Tensor Core”常表示输入使用 FP16，而累加可以使用 FP32。较高精度累加可减小长归约链中的舍入误差。

### 7.2 TF32 容易被误解

TF32 通常不是 CUDA C++ 中供用户直接声明变量的普通存储类型，而主要是一种 Tensor Core 乘法计算模式。它让 FP32 数据参与矩阵乘法时使用较低的乘法有效精度，同时通常保留 FP32 范围和 FP32 累加能力。

因此：

- “输入张量是 FP32”不一定意味着矩阵乘法完全走传统 FP32 标量流水线；
- 是否允许 TF32 取决于硬件、库版本和框架配置；
- 对精度敏感时，需要显式核对数学模式。

### 7.3 低精度的范围和精度权衡

浮点数大致由符号位、指数位和尾数组成：

$$
x = (-1)^s \times m \times 2^e
$$

- 指数位更多，通常表示范围更大；
- 尾数位更多，通常表示有效精度更高；
- 位宽变小后，需要在范围与精度之间权衡。

BF16 通常保留接近 FP32 的指数范围，但尾数精度较低；FP16 的尾数相对更多，但指数范围更窄。二者在不同模型和训练阶段中的稳定性表现可能不同。

### 7.4 不要仅凭类型名称推断吞吐

同一种类型在不同架构、不同指令形状和不同稀疏模式下，峰值吞吐可能显著不同。正确做法是同时确认：

- GPU 架构与 Compute Capability；
- 实际生成的指令；
- dense 还是 sparse；
- accumulator 类型；
- 厂商规格中的计数口径。

---

## 8. 编程模型与调用路径

开发者可以在多个抽象层级使用 Tensor Core。

### 8.1 高层框架

PyTorch、JAX、TensorFlow 等框架通常通过 cuBLAS、cuBLASLt、cuDNN 或专用 kernel 间接使用 Tensor Core。

例如，一个矩阵乘法：

```python
import torch

x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
w = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
y = x @ w
```

是否走 Tensor Core，不仅由 `dtype` 决定，还与以下因素有关：

- GPU 是否支持该类型的 Tensor Core MMA；
- 矩阵尺寸和 stride 是否合适；
- 框架与底层库的算法选择；
- 对齐、转置和布局；
- 确定性、精度及数学模式配置。

### 8.2 CUDA 数学库

常见库包括：

- cuBLAS / cuBLASLt：GEMM 与矩阵布局、epilogue；
- cuDNN：卷积、Attention 等深度学习算子；
- CUTLASS：可组合的高性能 CUDA 模板；
- 上层框架自带的 fused kernel 或第三方 kernel 库。

一般应优先使用成熟库，因为高性能 Tensor Core kernel 还需要解决：

- tile 划分；
- global memory 到 shared memory 的搬运；
- shared memory 到 register 的搬运；
- 双缓冲或多级流水；
- bank conflict；
- 寄存器压力；
- epilogue 融合；
- 不同形状下的算法选择。

### 8.3 WMMA API

CUDA 提供 warp-level matrix operations（WMMA）接口。下面是简化示意，重点是展示 warp 协同处理矩阵 fragment，而不是完整的生产级 GEMM：

```cpp
#include <mma.h>

using namespace nvcuda;

__global__ void wmma_example(
    const half* a,
    const half* b,
    float* c,
    int lda,
    int ldb,
    int ldc) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
                   wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                   wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, a, lda);
    wmma::load_matrix_sync(b_frag, b, ldb);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(c, c_frag, ldc, wmma::mem_row_major);
}
```

这里 `wmma::mma_sync` 表达的是 warp 级矩阵乘加。fragment 的元素由 warp 内线程共同持有，不能简单假设某个矩阵元素固定属于某个线程。

### 8.4 PTX 与架构相关指令

更底层可以使用 PTX 中的矩阵指令族。随着架构演进，可能出现不同范围和协作粒度的指令，例如 warp-level 或 warp-group-level MMA。

直接编写底层指令可以获得更强控制力，但代价是：

- 强架构相关；
- 约束复杂；
- 可维护性下降；
- 容易因布局、同步或寄存器映射错误而产生问题。

除非在开发基础算子库，一般不应优先从内联 PTX 开始。

---

## 9. 一个 CUDA kernel 如何同时使用两者

一个深度学习 kernel 经常同时包含：

1. 数据加载和地址计算；
2. Tensor Core 矩阵乘加；
3. 普通算术完成缩放、激活或边界处理；
4. 数据写回。

例如带 bias 和激活函数的线性层：

$$
Y = \operatorname{GELU}(XW + b)
$$

可能按以下方式执行：

```text
地址与边界计算             → 普通 INT / FP 流水线
XW 的主 GEMM              → Tensor Core
加 bias、类型转换、GELU    → 普通算术或特殊函数流水线
结果写回                  → Load/Store 单元
```

如果使用 fused epilogue，GEMM 的结果可以在寄存器中直接完成 bias 和激活，再写回显存。这说明高性能并不是“只用 Tensor Core”，而是让多种硬件资源合理配合，并减少中间数据搬运。

---

## 10. 理论性能为什么不等于实际性能

Tensor Core 峰值吞吐很高，但只有在计算能持续向它提供数据时才有意义。

### 10.1 Roofline 视角

算术强度定义为：

$$
I = \frac{\text{FLOPs}}{\text{Bytes transferred}}
$$

可达到的性能受到计算峰值与内存带宽共同约束：

$$
P_{\text{attainable}}
= \min\left(P_{\text{peak}},\ I \times BW_{\text{memory}}\right)
$$

其中：

- $P_{\text{peak}}$ 是对应数据类型和指令模式下的峰值算力；
- $BW_{\text{memory}}$ 是有效内存带宽；
- $I$ 是算术强度。

若算术强度低，即使 Tensor Core 理论峰值极高，kernel 仍可能受显存带宽限制。

### 10.2 常见性能损失来源

#### 矩阵太小

矩阵规模不足时，无法创建足够多的并行 tile，也难以摊薄启动、同步和数据搬运开销。

#### 维度不友好

矩阵维度不能良好匹配 MMA tile 时，可能需要 padding、边界分支或残块处理。库也可能选择非 Tensor Core 路径。

#### 数据布局与对齐不理想

非连续访问、错误 stride、转置成本或地址未满足高效访问要求，会降低内存合并和搬运效率。

#### 数据复用不足

如果每个输入只参与很少的乘加，数据搬运成本会盖过 Tensor Core 的计算优势。

#### shared memory 或寄存器成为瓶颈

Tensor Core 消耗数据很快，若片上搬运跟不上，计算流水线就会等待。过高的寄存器占用还会限制活跃 warp 数。

#### kernel launch 和同步开销

大量小 kernel 会让启动、同步与中间张量读写占据较高比例。算子融合往往比单纯提高某个 GEMM 的峰值利用率更重要。

#### 负载不均衡

不规则稀疏、MoE token 分布不均、变长序列和尾部 tile 都可能造成部分 SM 空闲。

### 10.3 Tensor Core 更快不是无条件成立

对于非常小的矩阵、逐元素算子、复杂分支、不规则 gather/scatter 或低算术强度任务，强行改写成矩阵形式可能增加数据重排和 padding，最终反而更慢。

性能判断必须基于端到端测量，而不是仅根据“是否使用 Tensor Core”。

---

## 11. 大模型训练与推理中的典型分工

### 11.1 Prefill

Prompt prefill 往往一次处理多个 token，矩阵的 $M$ 维较大，GEMM 更容易形成足够并行度，因此通常更容易高效利用 Tensor Core。

典型 Tensor Core 工作包括：

- QKV projection；
- attention 中的 $QK^T$ 与 $PV$；
- output projection；
- MLP / MoE expert GEMM。

其他操作如 RoPE、RMSNorm、采样前处理等，通常包含较多普通算术和访存。

### 11.2 Decode

自回归 decode 每条序列每步通常只产生一个 token。若 batch 较小，矩阵乘法的 $M$ 维很小，计算容易退化为 GEMV 或小 GEMM：

$$
y = xW
$$

此时常见问题是：

- 权重读取量大；
- 每个权重的复用较少；
- 更容易受显存带宽限制；
- Tensor Core 利用率可能明显低于 prefill。

增大有效 batch、连续批处理、量化和融合可以改善吞吐，但会受到延迟目标与请求分布约束。

### 11.3 Attention

FlashAttention 类算法并不是只依赖 Tensor Core。其高性能来自整体设计：

- 用 Tensor Core 加速矩阵乘法部分；
- 在片上完成分块 softmax；
- 通过普通算术计算缩放、最大值、指数和归一化；
- 减少中间 attention matrix 对 HBM 的读写。

它体现了一个关键原则：**高性能 kernel 的目标是优化完整数据流，而不只是让 Tensor Core 忙起来。**

### 11.4 MoE

MoE 中每个 token 只路由到少量 expert。其 expert 计算适合 Tensor Core，但系统性能还取决于：

- top-k 路由和索引计算；
- token permutation；
- all-to-all 通信；
- 每个 expert 的 token 数；
- grouped GEMM 的形状与调度；
- expert 负载均衡。

因此 MoE 即使具有很高的理论矩阵算力，也可能被通信、不规则访存或小矩阵拖慢。

### 11.5 量化推理

INT8、FP8、INT4 等量化能减少权重或激活带宽，并利用对应的 Tensor Core 路径。但完整成本还包括：

- quantize / dequantize；
- scale 和 zero point 加载；
- group-wise 元数据；
- outlier 处理；
- kernel 是否真正融合反量化与 MMA。

只有减少的数据搬运与更高的矩阵吞吐超过这些额外开销，量化才会带来端到端加速。

---

## 12. 如何确认程序是否使用了 Tensor Core

不要仅凭输入类型或框架 API 名称判断。

### 12.1 查看框架与库配置

确认：

- 张量 dtype；
- TF32 或混合精度开关；
- cuBLAS / cuDNN 的数学模式；
- 是否启用了量化 kernel；
- 实际选中的算法或 backend。

### 12.2 检查生成指令

可以反汇编 cubin 或检查编译输出，观察是否出现目标架构对应的 MMA 指令。由于指令名称随架构演进，不应只搜索一个固定字符串。

### 12.3 使用性能分析器

Nsight Compute 可以从以下角度判断：

- Tensor Pipe / MMA 相关指令数量；
- Tensor Core 活跃度；
- SM 吞吐；
- DRAM 与 L2 吞吐；
- shared memory 利用率与 bank conflict；
- stall reason；
- occupancy 与寄存器压力。

分析时应同时回答两个问题：

1. 是否发出了 Tensor Core 指令？
2. Tensor Core 是否持续得到数据并有效工作？

第一问为“使用了没有”，第二问才决定“使用得好不好”。

### 12.4 做对照实验

可比较：

- 不同 dtype；
- 开启/关闭 TF32；
- Tensor Core 友好与不友好的矩阵维度；
- 不同 batch size；
- fused 与 unfused kernel。

测量前需要 warmup，并用 CUDA event 或可靠 profiler 计时，避免把异步执行时间误算成 CPU 提交时间。

---

## 13. 常见误区

### 误区 1：Tensor Core 是比 CUDA Core 更强的通用核心

错误。Tensor Core 专注于矩阵乘加，无法替代地址计算、分支、任意逐元素函数和大量不规则算法。

### 误区 2：用了 FP16 / BF16 就一定用了 Tensor Core

错误。还取决于硬件支持、矩阵形状、布局、对齐、编译与库算法选择。

### 误区 3：FP32 一定只使用 CUDA Core

错误。部分架构可以通过 TF32 等模式让 FP32 输入的矩阵运算使用 Tensor Core，但其乘法精度语义与完整 FP32 FMA 不同。

### 误区 4：Tensor Core 只用于 AI

错误。只要问题能映射为受支持的矩阵乘加，高性能计算、科学计算、信号处理等领域也可以使用 Tensor Core。

### 误区 5：Tensor Core 数量越多，程序一定越快

错误。程序可能受内存带宽、通信、数据布局、kernel launch、同步或其他执行单元限制。

### 误区 6：Tensor Core 与 CUDA Core 二选一

错误。现实 kernel 经常同时使用两者，并使用 Load/Store、SFU、共享内存等其他资源。

### 误区 7：芯片规格中的 TOPS / TFLOPS 可以直接横向比较

错误。比较前必须统一：

- 数据类型；
- dense 或 sparse；
- 是否按 FMA = 2 FLOPs 计数；
- boost clock 假设；
- 累加类型；
- 功耗与散热条件。

### 误区 8：Tensor Core 计算必然降低模型精度

错误。Tensor Core 支持多种输入与累加模式。精度变化取决于具体数值格式和算法配置，而不是“是否使用 Tensor Core”这一事实本身。

---

## 14. 选型建议

### 14.1 优先考虑 Tensor Core 的情况

- 核心计算是 GEMM、卷积或 Attention；
- 矩阵尺寸足够大，或可通过 batching / grouped GEMM 聚合；
- 数据类型受目标 GPU 的 Tensor Core 支持；
- 维度、布局和对齐适合 MMA tile；
- 可接受混合精度或已验证数值误差；
- 计算量足以摊薄数据搬运与调度开销。

### 14.2 主要依赖普通算术流水线的情况

- 逐元素激活、归一化、索引和边界处理；
- 复杂控制流；
- 不规则稀疏、gather / scatter；
- 小规模或难以分块的计算；
- 无法表达为受支持的矩阵乘加；
- 对特定高精度标量语义有严格要求。

### 14.3 实际优化顺序

建议按以下顺序工作：

1. **确认瓶颈**：计算、显存带宽、片上搬运、通信还是启动开销；
2. **优先使用成熟库**：让库完成架构相关的 Tensor Core 选型；
3. **选择合适精度**：同时验证性能与数值质量；
4. **改善形状与布局**：减少残块、转置和非合并访问；
5. **增加数据复用**：使用合理 tile 和共享内存流水线；
6. **融合前后处理**：减少中间张量落入显存；
7. **使用 profiler 验证**：确认实际指令与硬件瓶颈；
8. **端到端复测**：避免局部 kernel 加速但整体变慢。

---

## 15. 总结

CUDA Core 与 Tensor Core 的关系可以归纳为：

1. **CUDA Core 面向通用标量算术，Tensor Core 面向矩阵乘加。**
2. **CUDA 线程不会简单地绑定某种 Core；warp 发出不同指令，由不同执行流水线处理。**
3. **Tensor Core 通过 MMA 在一条指令中表达大量结构化乘加，因此矩阵吞吐远高于逐标量执行。**
4. **Tensor Core 有数据类型、tile、布局、对齐和架构约束，不适合所有计算。**
5. **深度学习 kernel 通常同时使用 Tensor Core、普通算术流水线、访存单元和片上存储。**
6. **是否使用 Tensor Core 不等于是否高效；实际性能还取决于内存带宽、数据复用、矩阵规模、融合和调度。**
7. **不同 GPU 代际差异很大，任何具体吞吐、类型和指令结论都应结合目标架构验证。**

最终，不应把优化目标表述为“尽可能不用 CUDA Core、全部换成 Tensor Core”，而应表述为：

> **把适合矩阵化的主要计算交给 Tensor Core，把通用算术、控制与后处理交给普通执行流水线，并通过良好的数据布局、片上复用和算子融合，让整个 SM 的资源协同工作。**

