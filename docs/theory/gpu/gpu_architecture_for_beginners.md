# GPU 架构入门：从零理解一张显卡如何计算

> 本文面向**完全没有 GPU 基础**的初学者，用尽量少的前置知识，讲清楚一张 NVIDIA GPU 内部由哪些部件组成、它们如何协同完成计算，以及为什么 GPU 适合深度学习和大模型。
>
> 阅读完本文后，你应该能看懂后续更深入的文档（例如同目录的 `tensor_core_vs_cuda_core.md`），并建立起一套稳定的 GPU 心智模型。
>
> 先记住一句话：**CPU 追求“把一件事做得很快”，GPU 追求“把同一件事同时做很多份”。**

## 目录

1. [为什么需要 GPU](#1-为什么需要-gpu)
2. [CPU 与 GPU 的根本区别](#2-cpu-与-gpu-的根本区别)
3. [从神经网络到底层算子](#3-从神经网络到底层算子)
4. [一台 GPU 服务器长什么样](#4-一台-gpu-服务器长什么样)
5. [一张 GPU 的整体结构](#5-一张-gpu-的整体结构)
6. [SM：GPU 真正干活的车间](#6-smgpu-真正干活的车间)
7. [执行模型：线程、Warp 与 SIMT](#7-执行模型线程warp-与-simt)
8. [软件层次：Grid、Block、Thread](#8-软件层次gridblockthread)
9. [内存层级：速度与容量的权衡](#9-内存层级速度与容量的权衡)
10. [一个 kernel 的完整生命周期](#10-一个-kernel-的完整生命周期)
11. [CUDA 编程实战：从 API 到多维索引](#11-cuda-编程实战从-api-到多维索引)
12. [性能直觉：算力墙与内存墙](#12-性能直觉算力墙与内存墙)
13. [GPU 与深度学习为什么天生契合](#13-gpu-与深度学习为什么天生契合)
14. [数值精度：FP32、FP16、BF16 与 FP8](#14-数值精度fp32fp16bf16-与-fp8)
15. [主流 GPU 规格对比](#15-主流-gpu-规格对比)
16. [常见术语速查表](#16-常见术语速查表)
17. [常见误区](#17-常见误区)
18. [下一步学习建议](#18-下一步学习建议)
19. [总结](#19-总结)

---

## 1. 为什么需要 GPU

假设要给一张 4000×4000 的图片，每个像素都加上一个固定亮度值。

- **CPU 思路**：用一个（或几个）很强的核心，一个像素一个像素地算，算完 1600 万次。
- **GPU 思路**：用成千上万个较弱的核心，让它们**同时**各算一个像素，一步基本就能算完。

这两种像素之间**互不依赖**，非常适合“同时做很多份”。这类任务被称为**数据并行（data parallel）**任务。

深度学习中的矩阵乘法、卷积、逐元素激活等，本质上都是“对海量数据做相似运算”，因此天然适合 GPU。

一句话直觉：

> GPU 不是“更快的 CPU”，而是“为大规模并行专门设计的另一种处理器”。

---

## 2. CPU 与 GPU 的根本区别

| 维度 | CPU | GPU |
|---|---|---|
| 设计目标 | 低延迟：单个任务尽快完成 | 高吞吐：单位时间处理尽量多的数据 |
| 核心数量 | 少（几个到几十个），单核很强 | 多（成千上万个执行单元），单个较弱 |
| 控制逻辑 | 复杂：分支预测、乱序执行 | 精简：把晶体管更多用于计算 |
| 缓存 | 大而多层，服务少量线程 | 相对小，靠海量线程并行掩盖延迟 |
| 擅长 | 复杂控制流、串行逻辑、操作系统 | 大规模数据并行计算 |
| 类比 | 几位全能专家 | 一支纪律严明的大军 |

核心差异在于晶体管的分配方式：

- CPU 把大量晶体管用于**控制和缓存**，目的是让单个线程“聪明地、快速地”跑完；
- GPU 把大量晶体管用于**算术单元**，目的是让海量线程“一起”跑。

这也解释了为什么 GPU 不适合跑操作系统或复杂分支逻辑，却极其擅长矩阵运算。

---

## 3. 从神经网络到底层算子

在讲硬件之前，先搞清楚一个问题：**深度学习模型最终交给 GPU 的，到底是什么？**

以一个最简单的文本情感分类任务（判断影评是正面/负面/中性）为例，模型可以只用几层：

```text
"It is a good movie"
        ↓
   Embedding      （查表：把每个 token 变成向量）
        ↓
   Avg Pooling    （把一句话的多个向量平均成一个）
        ↓
   Linear         （矩阵乘 + 偏置）
        ↓
   ReLU           （逐元素取 max(0, x)）
        ↓
   Linear
        ↓
   Softmax        （归一化成概率）
        ↓
   [正面 0.6, 负面 0.02, 中性 0.38]
```

**关键洞察**：这些看起来五花八门的“层”，落到硬件上只有三类基本运算：

| 算子类别 | 例子 | 特征 |
|---|---|---|
| 矩阵乘法（GEMM） | Linear、Attention 的 QK^T 和 PV | 计算量大、数据复用高 |
| 逐元素运算（element-wise） | ReLU、add、scale、GELU | 每个元素独立，算术强度极低 |
| 归约运算（reduce） | sum、mean、max、Softmax 的分母 | 需要跨线程通信/规约 |

所以后面所有关于 GPU 的讨论，本质上都是在问：**这三类算子怎样在硬件上跑得最快？**

- 矩阵乘法 → 需要海量乘加单元 + 高数据复用 → Tensor Core、共享内存分块；
- 逐元素运算 → 完全并行但极度访存受限 → 靠算子融合减少显存往返；
- 归约运算 → 需要 warp 内/block 内通信 → 靠 warp shuffle、共享内存。

大语言模型（LLM）只是把这套结构堆得更深更宽：Transformer 的每一层依然是 GEMM + element-wise + reduce 的组合。模型规模从 GPT-1 的 ~0.1B 涨到今天的上千 B 参数，正是这种规模压力催生了整个“LLM 系统优化”领域。

---

## 4. 一台 GPU 服务器长什么样

初学者常常只想着“一张显卡”，但真实训练/推理跑在**整台服务器**上，性能瓶颈经常出现在显卡之外。

一台典型的 8 卡训练服务器配置（以某高校实验室的 4U 服务器为例）：

```text
┌─────────────────────────────────────────────────────┐
│ 4U 机箱                                              │
│                                                     │
│  2× CPU (AMD EPYC 9354，每颗 32 核)                  │
│  16× 64GB DDR5 内存  = 1TB 系统内存                   │
│  4× NVMe SSD (15.36TB, PCIe Gen4)                   │
│                                                     │
│  8× NVIDIA A6000 (每张 48GB 显存)                    │
│  4× 双槽 NVLink 桥接（两两配对互联）                    │
└─────────────────────────────────────────────────────┘
```

### 4.1 三条关键的数据通道

```text
CPU ←──── DDR5 内存总线 ────→ 系统内存      64~512 GB/s
CPU ←──── PCIe Gen4 x16 ────→ GPU              32 GB/s   ← 最窄的瓶颈
GPU ←──── HBM/GDDR 显存总线 ─→ 显存             768 GB/s ~ 数 TB/s
GPU ←──── NVLink ───────────→ 另一张 GPU     112.5 GB/s
```

数量级差异巨大，记住这个排序：

$$
BW_{\text{PCIe}} \ll BW_{\text{NVLink}} < BW_{\text{系统内存}} \ll BW_{\text{显存}}
$$

- **PCIe Gen4 x16 ≈ 32 GB/s**（16 条 lane × 2 GB/s per lane）——CPU 与 GPU 之间搬数据最慢的一环；
- **NVLink ≈ 112.5 GB/s**——GPU 到 GPU 的高速直连，比走 PCIe 快 3 倍以上；
- **显存带宽**动辄几百 GB/s 到几 TB/s——比 PCIe 快一到两个数量级。

### 4.2 为什么这些数字重要

**场景一：数据并行训练中的梯度同步。**
每个 step 结束，8 张卡要把梯度做 all-reduce。如果两张卡之间只能走 PCIe（32 GB/s），而不是 NVLink（112.5 GB/s），同步时间会成倍增加。这就是为什么服务器上要插 NVLink 桥、为什么 NCCL 会去探测 GPU 间的拓扑。

**场景二：GPU 间通信不一定走同一条路。**
在多路 CPU 服务器上，GPU 挂在不同 CPU 的 PCIe root complex 下面。跨 CPU 的 GPU 通信要经过 CPU 间的互联（Infinity Fabric / UPI），延迟和带宽都更差。这就是所谓的 **NUMA 效应**，也是 `nvidia-smi topo -m` 这个命令存在的意义。

**场景三：数据加载。**
训练数据从 NVMe SSD → 系统内存 → PCIe → 显存。任一环节跟不上，昂贵的 GPU 就在空转。

一句话总结：

> 单卡算力只是故事的一半，**数据能不能及时送到计算单元面前**是另一半。

---

## 5. 一张 GPU 的整体结构

从宏观到微观，一张 GPU 大致可以分为几层：

```text
GPU（整张显卡）
 ├── 显存 HBM / GDDR（全局内存，容量大、离计算单元远）
 ├── 显存控制器（Memory Controller，多个 32-bit 通道并联）
 ├── L2 Cache（所有 SM 共享）
 └── 多个 GPC（图形处理簇）
       └── 多个 SM（流式多处理器，真正执行计算）
             ├── CUDA Core（通用算术单元）
             ├── Tensor Core（矩阵乘加单元）
             ├── Warp Scheduler（指令调度员）
             ├── 寄存器文件（Register File）
             ├── 共享内存 / L1 Cache
             ├── Load/Store 单元（访存）
             └── SFU（特殊函数单元，算 sin/exp 等）
```

用两个真实芯片来落地这张图：

| | RTX A6000（GA102，Ampere） | H100（GH100，Hopper） |
|---|---|---|
| SM 数量 | 84 | 132（SXM）/ 114（PCIe） |
| 显存 | 48GB GDDR6 | 80GB HBM3 |
| 显存控制器 | 12 × 32-bit = 384-bit 总线 | HBM3 堆叠，5120-bit 级别 |
| L2 Cache | 6 MB | 50 MB |
| 每 SM 共享内存/L1 | 128 KB | 256 KB |

关键理解：

- 一张 GPU 由**几十到上百个 SM** 组成；
- **SM 是 GPU 的基本计算单位**，可以类比成一间独立的“计算车间”；
- 所有 SM 共享同一块**显存**和 **L2 Cache**；
- 显存带宽 = 总线位宽 × 有效频率，所以**显存控制器的数量/位宽直接决定带宽上限**；
- 一张 GPU 的性能，主要取决于 SM 的数量、SM 内部的算力，以及显存带宽。

“架构代际”（Ampere → Hopper → Blackwell）升级的通常是三件事：SM 数量变多、每个 SM 内部算力变强（新增 Tensor Core 精度类型）、显存和 Cache 变大变快。

---

## 6. SM：GPU 真正干活的车间

SM（Streaming Multiprocessor）是理解 GPU 的核心。把一个 SM 想象成一间车间：

- **CUDA Core**：大量通用工位，做加、减、乘、乘加、整数和逻辑运算；
- **Tensor Core**：专用的矩阵流水线，一次吞入多个小矩阵块，集中完成大量乘加（深度学习的主力）；
- **Warp Scheduler（调度员）**：决定每个时钟周期让哪一组线程执行哪条指令；
- **寄存器文件**：离计算单元最近、最快的存储，存放线程正在使用的数据；
- **共享内存（Shared Memory）**：车间内部的公共工作台，同一个 Block 的线程可以高速共享数据；
- **Load/Store 单元**：负责从显存搬数据进来、把结果搬出去；
- **SFU**：计算 $\sin$、$\cos$、$\exp$、$\sqrt{\ }$、倒数等特殊函数。

一个 SM 会**同时驻留很多线程**。当一部分线程在等待显存数据（这很慢）时，调度员立刻切换到另一批已经准备好的线程执行。**用大量线程的切换来“掩盖”内存延迟**，是 GPU 高吞吐的关键秘诀。

### 6.1 SM 内部再分成 4 个 partition

现代 NVIDIA SM 并不是一个整块，而是被切成 **4 个处理分区（processing block / partition）**，每个分区有自己独立的一套资源：

```text
┌──────────────────── 一个 SM ────────────────────┐
│                                                 │
│  ┌── Partition 0 ──┐  ┌── Partition 1 ──┐       │
│  │ 32 × CUDA Core  │  │ 32 × CUDA Core  │       │
│  │ 1 × Tensor Core │  │ 1 × Tensor Core │       │
│  │ Warp Scheduler  │  │ Warp Scheduler  │       │
│  │ 64KB 寄存器      │  │ 64KB 寄存器      │       │
│  │ LD/ST + SFU     │  │ LD/ST + SFU     │       │
│  └─────────────────┘  └─────────────────┘       │
│                                                 │
│  ┌── Partition 2 ──┐  ┌── Partition 3 ──┐       │
│  │      同上        │  │      同上        │       │
│  └─────────────────┘  └─────────────────┘       │
│                                                 │
│  ┌───── 共享内存 / L1 Cache（4 个分区共用）─────┐  │
│  │  A6000: 128KB      H100: 256KB            │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

把数字串起来：

$$
4\ \text{partition} \times 32\ \text{CUDA Core} = 128\ \text{CUDA Core / SM}
$$

$$
128\ \text{Core} \times 1\ \text{FP32 FMA / cycle} \Rightarrow \text{每个 SM 每周期完成 128 次 FP32 运算}
$$

$$
4\ \text{partition} \times 64\text{KB} = 256\text{KB}\ \text{寄存器文件 / SM}
$$

**为什么 partition 是 32 核？** 因为一个 Warp 正好是 32 个线程。一个 partition 恰好可以在一个周期里把一整个 Warp 的同一条指令发射出去。这就是 warp size = 32 的硬件由来——不是任意选的数字，而是**硬件分区宽度**的直接体现。

### 6.2 用 SM 数量估算整卡算力

有了 SM 结构，可以自己推导出官方标称的 FP32 算力。以 A6000 为例：

$$
84\ \text{SM} \times 128\ \text{Core/SM} = 10752\ \text{CUDA Core}
$$

$$
10752 \times 2\ \text{FLOP/FMA} \times 1.8\ \text{GHz} \approx 38.7\ \text{TFLOPS (FP32)}
$$

（一次乘加 FMA 算 2 个浮点运算，所以要乘 2。）

作为对照，一颗 128 核的 AMD EPYC 9754 CPU 大约是 576 GFLOPS。也就是说：

| | CPU (EPYC 9754) | GPU (A6000) |
|---|---|---|
| 硬件线程数 | 256 | 10,752 |
| 频率 | 2.25 GHz | 1.8 GHz |
| FP32 算力 | 576 GFLOPS | 38.7 TFLOPS |
| 功耗 | 360 W | 300 W |

**GPU 频率更低、功耗更低，算力却高约 67 倍**——差距全部来自并行度。顺便一提，38.7 TFLOPS 已经超过 2001 年世界排名第一的超级计算机。

---

## 7. 执行模型：线程、Warp 与 SIMT

### 7.1 Warp：32 个线程一起行动

GPU 调度的基本单位不是单个线程，而是 **Warp**——通常由 **32 个线程**组成的一组。

同一个 Warp 里的 32 个线程：

- 在同一时刻执行**同一条指令**；
- 但各自处理**不同的数据**。

这种模式叫 **SIMT（Single Instruction, Multiple Threads，单指令多线程）**。可以理解为教官喊一个口令“向前一步走”，32 个士兵同时执行同一个动作，只是每人站在不同位置。

Warp 的准确定位是：**GPU 创建、管理、调度、执行线程的最小单位。** SM 拿到一个 Thread Block 后，第一件事就是把它切成若干个 Warp。

同一个 Warp 内的 32 个线程：

- 从**同一个程序地址**开始执行；
- 各自拥有**独立的程序计数器（PC）和寄存器**；
- 每个周期执行**一条共同的指令**；
- 但**可以分支并独立执行**（代价见下节）。

### 7.2 Warp 的执行上下文与零开销切换

这是 GPU 与 CPU 在调度上最本质的差别。

**CPU 的线程切换很贵**：要把寄存器、栈指针等状态保存到内存，再加载另一个线程的状态，通常需要几百到几千个周期。所以 CPU 上线程数一般不超过核心数的两倍。

**GPU 的 Warp 切换几乎零开销**：

```text
Warp 的执行上下文（程序计数器 + 寄存器 + 共享内存）
在这个 Warp 的整个生命周期里，都常驻在 SM 上，从不换出。
        ↓
所以切换 Warp 不需要保存/恢复任何状态
        ↓
Warp-to-warp 切换是「瞬时」的
```

Warp Scheduler 每个周期做的事很简单：

1. 从驻留在本 partition 上的所有 Warp 中，挑一个**线程处于活跃（就绪）状态**的；
2. 把下一条指令发射给这个 Warp 的 32 个线程。

于是当 Warp A 因为等显存而卡住时，调度器立刻切到 Warp B、C、D……**延迟被并发彻底掩盖**。这就是第 6 节说的“用线程切换掩盖内存延迟”的具体机制。

**代价是什么？** 由于上下文常驻，一个 SM 能同时容纳多少 Warp，取决于资源够不够分：

$$
\text{驻留 Warp 数} \le \min\left(\frac{\text{寄存器总量}}{\text{每 Warp 寄存器需求}},\ \frac{\text{共享内存总量}}{\text{每 Block 共享内存需求}},\ \text{硬件上限}\right)
$$

这个“实际驻留 Warp 数 / 硬件最大 Warp 数”的比值，就是**占用率（Occupancy）**。kernel 里每个线程用的寄存器越多、每个 Block 申请的共享内存越大，能驻留的 Warp 就越少，掩盖延迟的能力就越弱。

> 这解释了一个反直觉的现象：**把 kernel 写得“更省指令”但用掉更多寄存器，反而可能更慢**——因为占用率掉下来了。

### 7.3 Block、Warp、Partition 的对应关系

以一个 128 线程的 Block 落到 A6000 的 SM 上为例：

```text
Thread Block（128 threads）
        ↓  SM 自动切分
   Warp0   Warp1   Warp2   Warp3     （128 / 32 = 4 个 Warp）
     ↓       ↓       ↓       ↓
Partition0 Partition1 Partition2 Partition3
```

因为一个 SM 有 4 个 partition、每个 partition 有自己的 Warp Scheduler，所以**每个 SM 每周期最多可以并行推进 4 个 Warp**（= 128 个线程的同一步）。

顺带解释一个常见的编程建议：**Block 的线程数最好是 32 的倍数**。如果写成 100 个线程，硬件仍会分配 4 个 Warp（128 个线程槽），最后一个 Warp 里有 28 个线程什么都不做，白白浪费。

### 7.4 分支发散（Warp Divergence）

如果 Warp 里的线程遇到 `if / else` 且走了不同分支，会发生什么？

```cpp
if (threadIdx.x < 16) {
    // 分支 A
} else {
    // 分支 B
}
```

由于同一 Warp 必须执行同一条指令，GPU 只能：

1. 先让走分支 A 的线程执行，走 B 的线程**暂停等待**；
2. 再让走分支 B 的线程执行，走 A 的线程等待。

结果是两个分支被**串行**执行，性能下降。这就是**分支发散**。因此，GPU 编程要尽量让同一 Warp 内的线程走**相同路径**。

一句话总结：

> GPU 喜欢“大家做同样的事”，讨厌“各走各的路”。

---

## 8. 软件层次：Grid、Block、Thread

### 8.1 Host 与 Device：一份代码，两个世界

CUDA 程序的基本模型是**一份源码里同时包含 CPU 代码和 GPU 代码**：

| | Host | Device |
|---|---|---|
| 指的是 | CPU | GPU |
| 跑什么 | 普通 C++ 程序 | CUDA kernel |
| 职责 | 控制流程、分配显存、启动 kernel、搬数据 | 大规模并行计算 |
| 内存 | 系统内存 | 显存 |

两者内存是分开的，所以**必须显式在系统内存和显存之间搬数据**（`cudaMemcpy`），这一步走 PCIe，是第 4 节说的最窄通道。

关于 kernel 本身，有一句话最容易被误解，但特别重要：

> **kernel 的代码本身是「串行」的——它描述的是「一个线程」要做什么；并行性来自于同时启动上万个线程去跑同一份代码。**

所以写 CUDA 的思维方式是：先想清楚“单个线程负责哪一份数据”，再用线程索引把它算出来。

### 8.2 三层线程组织

写 CUDA 程序时，我们组织线程的方式是分层的：

```text
Grid（一次 kernel 启动的全部线程）
 └── Block（线程块，会被分配到某个 SM 上执行）
       └── Warp（32 个线程一组，调度单位）
             └── Thread（最小的执行单元）
```

对应关系：

| 软件概念 | 硬件对应 | 说明 |
|---|---|---|
| Thread（线程） | 在 CUDA Core 等单元上执行 | 处理一份数据 |
| Warp | Warp Scheduler 调度 | 32 线程一起，SIMT 执行 |
| Block（线程块） | 整体分配给**一个 SM** | 块内线程可共享 Shared Memory、可同步 |
| Grid | 分布到**整张 GPU** 的多个 SM | 一次 kernel 的全部线程 |

几个重要规则：

- **一个 Block 只会在一个 SM 上运行**，不会跨 SM；一个 SM 可以同时容纳多个 Block；
- **同一个 Block 内的线程**才能用 Shared Memory 共享数据，并通过 `__syncthreads()` 同步；
- **不同 Block 之间**默认不能直接共享数据或同步，这样设计是为了让 Block 能被灵活地调度到任意 SM，从而适配不同规模的 GPU。

举个例子，把 1024×1024 个元素分给线程处理，常见写法是每个 Block 有 256 个线程，一共启动 4096 个 Block：

```cpp
dim3 block(256);
dim3 grid((1024 * 1024 + 255) / 256);
my_kernel<<<grid, block>>>(data);
```

从软件到硬件的完整映射链条：

```text
Kernel 启动
   ↓
Grid                 ──映射到──→  整张 GPU（所有 SM）
   ↓
Thread Block         ──映射到──→  某一个 SM（不跨 SM）
   ↓
Warp（32 threads）    ──映射到──→  SM 的某个 partition
   ↓
Thread               ──映射到──→  partition 里的一个 CUDA Core 通道
```

**这个设计带来的可扩展性**：因为 Block 之间互不依赖、不能同步，GPU 可以随意把 Block 分派到任意空闲的 SM。同一份代码在 20 个 SM 的入门卡和 132 个 SM 的 H100 上都能跑，只是完成时间不同。这就是 CUDA 编程模型能跨代际复用的根本原因。

### 8.3 编译：nvcc 做了什么

CUDA 源码（`.cu`）由 `nvcc` 编译。它会把文件拆成两半：host 代码交给系统的 C++ 编译器（gcc/clang），device 代码编译成 GPU 指令（PTX 中间表示 → SASS 机器码），最后链接成一个可执行文件或动态库。

编译成动态库供 Python 调用是深度学习框架里最常见的用法：

```bash
nvcc -o output.so --shared src.cu -Xcompiler -fPIC
```

- `--shared`：生成动态库（`.so`）；
- `-Xcompiler -fPIC`：把 `-fPIC` 透传给底层的 host 编译器（动态库要求位置无关代码）；
- 之后就可以用 `ctypes` / `pybind11` / PyTorch 的 `load()` 从 Python 里调用。

实际项目里常见的两种做法：

- **AOT（提前编译）**：构建时用 `nvcc`/CMake 编好，发布二进制，启动快；
- **JIT（即时编译）**：运行时根据实际 shape / dtype 现场编译（Triton、`torch.compile`、SGLang 里的部分 kernel 都是这条路），灵活但首次调用有编译开销。

---

## 9. 内存层级：速度与容量的权衡

GPU 里"离计算单元越近的存储越快、但越小"。这是理解性能的第二个关键。

```text
寄存器 Register      → 最快，最小，每个线程私有
   ↑
共享内存 Shared Mem  → 很快，Block 内线程共享（和 L1 常共用物理空间）
   ↑
L2 Cache            → 全部 SM 共享
   ↑
显存 HBM / GDDR      → 容量大（几十 GB），但相对最慢、延迟最高
```

粗略的速度与容量对比（不同架构差异很大，仅供建立直觉）：

| 存储层级 | 相对速度 | 容量量级 | 作用范围 |
|---|---|---|---|
| 寄存器 | 最快 | 每 partition 64KB（每 SM 256KB） | 单个线程私有 |
| 共享内存 / L1 | 很快 | 每 SM 128KB（A6000）~256KB（H100） | 一个 Block 内共享 |
| L2 Cache | 较快 | 6MB（A6000）~50MB（H100） | 全部 SM 共享 |
| 显存（HBM/GDDR） | 相对慢 | 48~192 GB | 整张 GPU |
| 系统内存 | 很慢（要过 PCIe） | 数百 GB ~ TB | 整台机器 |

把它和第 4 节的机器级通道拼在一起，就是完整的数据流图：

```text
                       ┌──────────── GPU ────────────┐
  CPU ── PCIe ──────→  │  显存 (HBM)                  │
      32 GB/s          │      ↕ 768 GB/s ~ 数 TB/s    │
                       │  L2 Cache（全 SM 共享）        │
                       │      ↕                       │
                       │  共享内存 / L1（每 SM）         │
                       │      ↕                       │
  系统内存               │  寄存器（每 partition）        │
  64~512 GB/s          │      ↕                       │
                       │  CUDA Core / Tensor Core     │
                       └──────────────────────────────┘
                                    ↕ NVLink 112.5 GB/s
                                 另一张 GPU
```

**从这张图能直接读出三条优化原则**：

1. 尽量**不要**跨 PCIe 搬数据（32 GB/s 是最窄的地方）——数据尽量常驻显存；
2. 尽量**减少显存访问**——把数据搬进共享内存/寄存器后反复复用；
3. 多卡通信优先走 **NVLink** 而不是 PCIe。

**为什么这很重要？**

从显存读一个数据，可能要等几百个时钟周期；而寄存器几乎立刻可用。因此高性能 kernel 的核心思路是：

1. 从显存**一次性**把一块数据搬进共享内存/寄存器；
2. 在片上**反复复用**这块数据，做尽量多的计算；
3. 最后再把结果写回显存。

这就是所谓的**数据复用（data reuse）**，也是矩阵乘法分块（tiling）优化的基本原理。

---

## 10. 一个 kernel 的完整生命周期

把前面的概念串起来，看一次 GPU 计算从头到尾发生了什么。以“两个向量相加 $c_i = a_i + b_i$”为例：

```text
1. CPU 端（Host）
   - 在显存上分配 a、b、c 的空间
   - 把 a、b 从内存拷贝到显存

2. CPU 启动 kernel
   - 指定 Grid 和 Block 大小
   - GPU 开始把 Block 分配到各个 SM

3. GPU 端（Device）执行
   - 每个线程计算自己负责的下标 i
   - Load/Store 单元从显存读 a[i]、b[i]
   - CUDA Core 执行加法 a[i] + b[i]
   - 结果写回显存 c[i]
   - 当某些 Warp 等待数据时，调度员切换到其他 Warp

4. CPU 端取回结果
   - 把 c 从显存拷回内存
```

对应的 kernel 代码非常简洁：

```cpp
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 计算全局下标
    if (i < n) {                                     // 边界检查
        c[i] = a[i] + b[i];                          // 每个线程算一个元素
    }
}
```

注意两个初学者常忽略的点：

- **CPU 和 GPU 内存是分开的**，数据必须显式在两者之间拷贝（这也常常是性能瓶颈）；
- `blockIdx.x * blockDim.x + threadIdx.x` 是把“第几个 Block 的第几个线程”换算成“全局第几个元素”的标准公式。

---

## 11. CUDA 编程实战：从 API 到多维索引

上一节给了流程骨架，这一节把每一步的**真实 API 和代码**补全。看完这一节，你应该能自己写出完整可编译的向量加法和矩阵加法。

### 11.1 五个必会的 CUDA API

一个最小可用的 CUDA 程序，Host 端只需要五个操作：

```text
1. cudaMalloc    在显存上申请空间
2. cudaMemcpy    把数据从系统内存搬到显存（H2D）
3. kernel<<<>>>  启动 kernel
4. cudaMemcpy    把结果从显存搬回系统内存（D2H）
5. cudaFree      释放显存
```

#### cudaMalloc：申请显存

```cpp
cudaError_t cudaMalloc(void** devPtr, size_t size);
```

- `devPtr`：**指向指针的指针**——函数要把新分配的显存地址写回你的变量里；
- `size`：**字节数**（不是元素个数），所以要乘 `sizeof(T)`。

```cpp
int   *dA;  cudaMalloc(&dA, n * sizeof(int));
float *dB;  cudaMalloc(&dB, n * sizeof(float));
```

为什么是 `void**` 而不是返回值？因为返回值位置留给了 `cudaError_t` 错误码。这是 CUDA API 的统一风格：**返回值永远是错误码，输出通过指针参数带回。**

这里有个初学者一定会踩的坑：

```text
CPU 内存                        GPU 显存
┌──────────────┐               ┌─────────────────────┐
│ dA (变量本身) │               │  地址 306 起的一块空间 │
│  值 = 306    │ ──指向──────→ │  12 13 14 15 16 ...  │
└──────────────┘               └─────────────────────┘
   ↑ 这个指针变量存在 CPU 上，       ↑ 它指向的数据在 GPU 上
     但它的「值」是一个 GPU 地址
```

也就是说，`dA` 这个**指针变量**住在 CPU 内存里，但它保存的**地址**是显存地址。所以：

```cpp
printf("%d\n", dA[0]);   // ✗ 段错误！CPU 不能直接解引用显存地址
cudaMemcpy(...);         // ✓ 必须通过 API 搬运
```

`cudaMalloc` 分配的是**全局内存（global memory）**，特点是：**所有线程（不分 Block）都能访问**，且在同一个进程内**跨多次 kernel 启动持续存在**。

#### cudaMemcpy：搬数据

```cpp
cudaError_t cudaMemcpy(void* dst, const void* src,
                       size_t count, cudaMemcpyKind kind);
```

注意参数顺序是 **目标在前、源在后**（和 `memcpy` 一致，但和直觉相反，容易写错）。`kind` 常用两个：

| kind | 方向 |
|---|---|
| `cudaMemcpyHostToDevice` | 系统内存 → 显存 |
| `cudaMemcpyDeviceToHost` | 显存 → 系统内存 |

```cpp
cudaMemcpy(dA, hA, n * sizeof(int), cudaMemcpyHostToDevice);   // 上传
cudaMemcpy(hC, dC, n * sizeof(int), cudaMemcpyDeviceToHost);   // 下载
```

**这一步走 PCIe（约 32 GB/s）**，是第 4 节讲的最窄通道。默认版本是**同步**的：CPU 会一直阻塞到拷贝完成。所以优化的第一原则永远是：**尽量减少 H2D/D2H 的次数，让数据常驻显存。**

#### cudaFree：释放

```cpp
cudaError_t cudaFree(void* devPtr);
```

显存不会自动回收，忘记释放就会泄漏。长时间运行的服务里，显存泄漏往往表现为“跑一段时间后 OOM”。

### 11.2 三种内存作用域

第 9 节从性能角度讲了内存层级，这里从**编程角度**看一遍：谁能访问谁。

| 作用域 | 如何声明 | 谁可见 | 生命周期 |
|---|---|---|---|
| 寄存器（thread-local） | kernel 里的普通局部变量 | 仅该线程 | 线程结束即失效 |
| 共享内存（block-local） | `__shared__ float tile[32][32];` | 同一个 Block 内所有线程 | Block 结束即失效 |
| 全局内存（global） | Host 端 `cudaMalloc` | **所有** Block 的所有线程 | 跨 kernel 启动持续存在 |

```cpp
__global__ void demo(float* g_data) {
    int   x = threadIdx.x;          // 寄存器：只有我自己看得见
    __shared__ float tile[256];     // 共享内存：同 Block 的兄弟线程都看得见
    tile[x] = g_data[x];            // 全局内存：谁都看得见
    __syncthreads();                // 等所有兄弟线程都写完，再往下走
    // ... 此后可以放心读 tile 里别人写的数据
}
```

`__syncthreads()` 是 Block 内的**屏障（barrier）**：所有线程都到达这里之后才继续。它只在 Block 内有效——**跨 Block 无法同步**（第 8 节讲过原因）。

### 11.3 函数修饰符：`__global__` / `__device__` / `__host__`

CUDA 把 Host 和 Device 代码写在同一个 `.cu` 文件里，靠修饰符区分：

| 修饰符 | 在哪里**调用** | 在哪里**执行** | 用途 |
|---|---|---|---|
| `__global__` | Host（CPU） | Device（GPU） | **kernel 入口**，必须返回 `void` |
| `__device__` | Device（GPU） | Device（GPU） | GPU 上的辅助函数 |
| `__host__` | Host | Host | 普通 CPU 函数（默认，可省略） |

```cpp
__device__ float square(float x) {      // GPU 内部的辅助函数
    return x * x;
}

__global__ void kernel(float* out, const float* in, int n) {  // kernel 入口
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = square(in[i]);   // 在 GPU 上调用 __device__ 函数
}

__host__ void run() {                    // 普通 CPU 函数
    kernel<<<blocks, threads>>>(d_out, d_in, n);
}
```

可以同时写 `__host__ __device__`，让同一份代码在 CPU 和 GPU 上都能编译出一份，常用于共享数学工具函数。

记住三条硬性规则：

- `__global__` 函数**必须返回 `void`**（结果只能通过指针参数写出）；
- `__global__` 只能从 Host 调用（CUDA 动态并行是例外，初学阶段不用管）；
- kernel 启动是**异步**的——`<<<>>>` 立刻返回，CPU 继续往下跑。想确保 kernel 跑完，要 `cudaDeviceSynchronize()`，或者靠后面的 `cudaMemcpy`（同步版本）隐式等待。

### 11.4 启动配置：`<<<Dg, Db>>>` 与 dim3

```cpp
kernelName<<<Dg, Db>>>(args...);
```

- `Dg`（**grid 维度**）：一个 Grid 里有多少 Block → `Dg.x * Dg.y * Dg.z` 个 Block；
- `Db`（**block 维度**）：一个 Block 里有多少 Thread → `Db.x * Db.y * Db.z` 个线程。

两者都可以是 `int`（等价于只用 x 维）或 `dim3`（最多三维）：

```cpp
dim3 Dg(4, 2, 1);       // 4 × 2 × 1 = 8 个 Block
dim3 Db(8, 8, 1);       // 8 × 8 × 1 = 64 个线程/Block
kernel<<<Dg, Db>>>(args);   // 总共 8 × 64 = 512 个线程
```

**一条必须记住的硬限制**：

$$
Db.x \times Db.y \times Db.z \le 1024
$$

一个 Block 最多 1024 个线程（当前所有主流架构）。原因很直观：Block 必须整体放进**一个 SM**，而 SM 的寄存器和共享内存有限。

所以 `dim3(32, 32)` = 1024 恰好是上限，而 `dim3(64, 64)` = 4096 会**启动失败**（而且是静默失败，只能靠检查 `cudaGetLastError()` 发现）。

**为什么要有多维？** 纯粹是为了代码可读性。处理二维矩阵时，用 `(i, j)` 比手工从一维 id 拆出行列要清晰得多。硬件层面并不关心维度——它最终都会拉平成线性的 Warp。

### 11.5 内置索引变量

编译器在每个 kernel 里自动提供四个内置变量，每个都有 `.x` / `.y` / `.z` 三个字段：

| 变量 | 类型 | 含义 |
|---|---|---|
| `gridDim` | `dim3` | Grid 的尺寸（有多少 Block） |
| `blockIdx` | `uint3` | 当前 Block 在 Grid 中的索引 |
| `blockDim` | `dim3` | Block 的尺寸（每 Block 多少线程） |
| `threadIdx` | `uint3` | 当前线程在 Block 中的索引 |

配对关系很好记：**`XxxDim` 是「总共有多少」，`XxxIdx` 是「我是第几个」。**

一维全局索引的标准公式：

$$
i = \underbrace{blockIdx.x}_{\text{我在第几个 Block}} \times \underbrace{blockDim.x}_{\text{每个 Block 多少线程}} + \underbrace{threadIdx.x}_{\text{我在 Block 里第几个}}
$$

```text
blockDim.x = 4，一共 3 个 Block：

Block 0            Block 1            Block 2
[t0 t1 t2 t3]      [t0 t1 t2 t3]      [t0 t1 t2 t3]
 0  1  2  3         4  5  6  7         8  9  10 11   ← 全局索引 i
```

**为什么几乎每个 kernel 都要写 `if (i < n)`？**

因为 Block 数量要向上取整：

```cpp
int num_blocks = (n + threads_per_block - 1) / threads_per_block;
```

比如 `n = 1000`、每 Block 256 线程，则需要 4 个 Block = 1024 个线程，最后 24 个线程的 `i` 会落在 `[1000, 1024)`。没有边界检查就会**越界写显存**——这类 bug 通常不会立刻崩，而是随机破坏其他数据，极难排查。

### 11.6 完整示例一：向量加法

把上面全部拼起来，这是一个完整可编译的向量加法：

```cpp
__global__ void VecAddKernel(int* A, int* B, int* C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void VecAddCUDA(int* Acpu, int* Bcpu, int* Ccpu, int n) {
    int *dA, *dB, *dC;

    // 1. 申请显存
    cudaMalloc(&dA, n * sizeof(int));
    cudaMalloc(&dB, n * sizeof(int));
    cudaMalloc(&dC, n * sizeof(int));

    // 2. 上传输入（H2D，走 PCIe）
    cudaMemcpy(dA, Acpu, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, Bcpu, n * sizeof(int), cudaMemcpyHostToDevice);

    // 3. 配置并启动 kernel
    int threads_per_block = 256;                                    // 32 的倍数
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;  // 向上取整
    VecAddKernel<<<num_blocks, threads_per_block>>>(dA, dB, dC, n);

    // 4. 下载结果（D2H，同步，会隐式等 kernel 跑完）
    cudaMemcpy(Ccpu, dC, n * sizeof(int), cudaMemcpyDeviceToHost);

    // 5. 释放显存
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}
```

以 `n = 1024` 为例，配置是 4 个 Block × 256 线程。回到硬件视角：每个 Block 被切成 8 个 Warp（256/32），落在某个 SM 的 4 个 partition 上轮转执行。

> 顺便算一下算术强度：读 8 字节（两个 int）、写 4 字节，只做 1 次加法 → $I = 1/12$ FLOP/Byte。这是**极度访存受限**的典型例子。所以向量加法这种算子，实测性能只取决于显存带宽，跟 TFLOPS 毫无关系——这正是第 12 节要讲的事。

### 11.7 完整示例二：矩阵加法（二维索引）

处理 $N \times N$ 矩阵时，用二维配置更自然：

```cpp
__global__ void MatAddKernel(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // 行
    int j = blockIdx.y * blockDim.y + threadIdx.y;   // 列
    if (i < N && j < N) {
        C[i * N + j] = A[i * N + j] + B[i * N + j];  // 手工做「行主序」展平
    }
}

void MatAddCUDA(float* Acpu, float* Bcpu, float* Ccpu, int n) {
    float *dA, *dB, *dC;
    size_t bytes = n * n * sizeof(float);

    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);
    cudaMemcpy(dA, Acpu, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, Bcpu, bytes, cudaMemcpyHostToDevice);

    int THREADS = 32;
    int BLOCKS  = (n + THREADS - 1) / THREADS;
    dim3 threads(THREADS, THREADS);   // 32 × 32 = 1024，正好是上限
    dim3 blocks(BLOCKS, BLOCKS);      // 二维网格覆盖整个矩阵
    MatAddKernel<<<blocks, threads>>>(dA, dB, dC, n);

    cudaMemcpy(Ccpu, dC, bytes, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}
```

两个关键点：

**（1）显存里没有「二维数组」。** 矩阵在显存中是**一维连续**的，必须手工展平。行主序（row-major，C/PyTorch 默认）的公式是：

$$
\text{offset} = i \times N + j
$$

**（2）`dim3(32, 32)` 不是随便选的。** $32 \times 32 = 1024$ 正好用满 Block 上限；同时 32 也是 warp size，让 `threadIdx.x` 连续的 32 个线程恰好组成一个 Warp。

这带来一个至关重要的性能细节——**为什么必须让 `x` 维对应「连续内存方向」**：

```text
一个 Warp 里 threadIdx.x = 0..31（threadIdx.y 相同）
    ↓ 它们访问 C[i*N + j]，j = threadIdx.x 连续
    ↓ 32 个线程访问 32 个连续的 float = 128 字节
    ↓ 硬件用「一次」内存事务就能全部取回  ✓ 合并访问（coalesced）
```

如果把 `i`、`j` 的定义反过来（让 `threadIdx.x` 对应行），同一 Warp 的 32 个线程会访问相隔 $N$ 个元素的地址，需要 32 次独立的内存事务，**带宽利用率掉到 1/32**。这就是**合并访问（memory coalescing）**——初学者最容易忽略、但影响最大的优化点之一。

### 11.8 完整示例三：三维索引与全局 tid

如果 Grid 和 Block 都是三维的，怎么算全局唯一的线程 id？

```cpp
int main() {
    dim3 threads_per_block(2, 4, 8);   // 2×4×8  = 64 线程/Block
    dim3 blocks_per_grid(2, 3, 4);     // 2×3×4  = 24 Block
    fullKernel<<<blocks_per_grid, threads_per_block>>>(d_in, d_out);
    // 总计 24 × 64 = 1536 个线程
}
```

```cpp
__global__ void fullKernel(float* din, float* dout) {
    // 第一步：把三维 blockIdx 展平成一维 block_id
    int block_id = blockIdx.x
                 + blockIdx.y * gridDim.x
                 + blockIdx.z * gridDim.x * gridDim.y;

    // 第二步：算出本 Block 在全局线程序列中的起始偏移
    int block_offset = block_id * blockDim.x * blockDim.y * blockDim.z;

    // 第三步：把三维 threadIdx 展平成 Block 内偏移
    int thread_offset = threadIdx.x
                      + threadIdx.y * blockDim.x
                      + threadIdx.z * blockDim.x * blockDim.y;

    int tid = block_offset + thread_offset;
    dout[tid] = func(din[tid]);
}
```

展平的规律是统一的（x 变化最快，z 最慢）：

$$
\text{flat} = idx.x + idx.y \times Dim.x + idx.z \times Dim.x \times Dim.y
$$

**一个值得思考的问题**：这 1536 个线程能在 A6000 上「同时」跑完吗？

答案是**能，而且远远用不完**。A6000 有 84 个 SM，24 个 Block 只占用 24 个 SM，剩下 60 个 SM 完全闲置。1536 个线程相对于 10752 个 CUDA Core 来说也太少了。

这引出一条实用的配置经验：

> **Block 数量应该明显多于 SM 数量**（通常几倍以上），才能填满整张 GPU 并让调度器有足够的 Warp 来掩盖延迟。Block 太少 = 大部分 SM 空转。

### 11.9 异步拷贝与 TMA（进阶）

前面用的 `cudaMemcpy` 是同步的，CPU 会阻塞等待。实际高性能代码普遍使用**异步**版本，把「搬数据」和「算」重叠起来：

```text
同步（串行）：  [搬 tile0] [算 tile0] [搬 tile1] [算 tile1] ...
异步（重叠）：  [搬 tile0] [搬 tile1] [搬 tile2] ...
                          [算 tile0] [算 tile1] ...
                          ↑ 搬运和计算同时进行，延迟被隐藏
```

Hopper（H100）为此新增了硬件单元 **TMA（Tensor Memory Accelerator）**，专门负责「显存 ↔ 共享内存」的大块异步搬运，并配合 barrier 做同步：

```cpp
cuda::memcpy_async(dst, src, size, barrier);   // 发起异步拷贝，立刻返回
// ... 这期间可以做别的计算 ...
barrier.wait();                                 // 需要数据时才等待
```

TMA 的价值在于：**过去搬运地址计算要占用宝贵的 CUDA Core 和寄存器，现在交给专用硬件**，计算单元可以专注算数。这是 FlashAttention-3、以及 Hopper 上各类高性能 GEMM kernel 的关键基础设施。

初学阶段不需要手写 TMA，但要知道这个方向：**现代 GPU 优化的主线之一，就是让数据搬运和计算尽可能重叠。**

### 11.10 一份 CUDA 编程检查清单

写完 kernel 后逐项自检：

| 检查项 | 为什么 |
|---|---|
| `cudaMalloc` 的 size 乘了 `sizeof(T)` 吗？ | 单位是字节，不是元素数 |
| `cudaMemcpy` 的 dst / src 顺序对吗？ | 目标在前，容易写反 |
| kernel 里有 `if (i < n)` 边界检查吗？ | Block 向上取整会多出线程 |
| Block 线程数是 32 的倍数吗？ | 对齐 warp size，避免空转线程 |
| Block 线程总数 ≤ 1024 吗？ | 硬限制，超了静默启动失败 |
| Block 数量明显多于 SM 数吗？ | 否则大量 SM 空转 |
| `threadIdx.x` 对应的是内存连续方向吗？ | 决定能否合并访问，影响可达 32 倍 |
| `cudaFree` 了吗？ | 显存不会自动回收 |
| 检查了 `cudaGetLastError()` 吗？ | CUDA 错误默认静默，不检查等于看不见 bug |
| H2D/D2H 拷贝次数能再减少吗？ | PCIe 32 GB/s 是最窄通道 |

关于最后一条——错误检查值得单独强调，因为 CUDA API **不会抛异常**，出错只返回错误码：

```cpp
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                  \
                    cudaGetErrorString(err), __FILE__, __LINE__);        \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

CUDA_CHECK(cudaMalloc(&dA, n * sizeof(int)));
kernel<<<blocks, threads>>>(dA, n);
CUDA_CHECK(cudaGetLastError());        // kernel 启动本身的错误（如线程数超限）
CUDA_CHECK(cudaDeviceSynchronize());   // kernel 执行期间的错误（如越界访问）
```

注意 kernel 需要**两次**检查：`cudaGetLastError()` 抓启动配置错误，`cudaDeviceSynchronize()` 抓执行期错误（因为启动是异步的，执行错误要等同步才能观察到）。

---

## 12. 性能直觉：算力墙与内存墙

初学者最容易犯的错误是：**以为算力（FLOPS）越高，程序就越快**。实际上很多程序被**内存带宽**卡住，而不是算力。

### 12.1 两种瓶颈

- **计算受限（compute-bound）**：算术运算太多，计算单元忙不过来。此时提高算力才有用。
- **访存受限（memory-bound）**：数据搬运太多，计算单元大部分时间在“等数据”。此时提高算力没用，要减少数据搬运。

### 12.2 算术强度

用**算术强度（Arithmetic Intensity）**衡量一个程序“每搬 1 字节数据，能做多少次计算”：

$$
I = \frac{\text{计算量（FLOPs）}}{\text{数据搬运量（Bytes）}}
$$

- $I$ 很低（比如向量加法：读 8 字节只做 1 次加法）→ 通常**访存受限**；
- $I$ 很高（比如大矩阵乘法：一块数据被反复复用）→ 有机会**计算受限**。

可达到的性能受两者共同限制（Roofline 模型）：

$$
P_{\text{可达}} = \min\left(P_{\text{峰值算力}},\ I \times BW_{\text{内存带宽}}\right)
$$

**直觉结论**：

> 想让 GPU 跑得快，不仅要有足够的计算，更要**减少从显存搬数据的次数**，让搬进来的数据被尽量多地复用。这就是为什么共享内存、分块、算子融合如此重要。

---

## 13. GPU 与深度学习为什么天生契合

深度学习的计算，绝大部分是**矩阵乘法**和**逐元素运算**，正好命中 GPU 的强项。

一个全连接层：

$$
Y = XW + b
$$

其中 $X$、$W$ 是矩阵。矩阵乘法：

$$
C_{ij} = \sum_{k} A_{ik} B_{kj}
$$

有几个特点让它非常适合 GPU：

1. **高度并行**：每个输出元素 $C_{ij}$ 可以独立计算；
2. **数据复用高**：矩阵 $A$、$B$ 的每个元素都会参与很多次乘加，算术强度高；
3. **规整**：没有复杂分支，同一 Warp 内线程走相同路径，几乎没有发散。

这也是为什么 NVIDIA 专门设计了 **Tensor Core**——用一条指令完成一整块小矩阵的乘加，把矩阵吞吐再拉高一个数量级。关于 CUDA Core 与 Tensor Core 的详细区别，见同目录的 `tensor_core_vs_cuda_core.md`。

在大模型推理中：

- **Prefill 阶段**（一次处理很多 token）矩阵较大，容易喂饱 GPU，通常计算受限；
- **Decode 阶段**（每步只产生一个 token）矩阵很“瘦”，容易变成访存受限，Tensor Core 利用率往往偏低。

理解这一点，就能明白为什么大模型推理要用连续批处理、量化、算子融合等手段来提升 GPU 利用率。

---

## 14. 数值精度：FP32、FP16、BF16 与 FP8

上一节反复提到“量化”“低精度”。这一节把它讲透：**同样是一个数，用多少个 bit 存、这些 bit 怎么分配，直接决定了它能表示多大的范围、多高的精度，以及在 GPU 上算得多快。**

### 14.1 一个浮点数由三部分组成

任何浮点数都可以拆成三段：

$$
x = (-1)^{s} \times (1 + m) \times 2^{(e - \text{bias})}
$$

- **符号位 sign（s）**：1 bit，决定正负；
- **指数位 exponent（e）**：决定**能表示的范围（Range）**——指数位越多，能表示的数越大/越小；
- **尾数位 mantissa（m）**：决定**精度（Precision）**——尾数位越多，同一个数能表示得越“细”。

关键权衡：**在总 bit 数固定时，多给指数位（更大范围）就得少给尾数位（更低精度），反之亦然。** 这就是那句话的含义：

> **每多分 1 个 bit，只能在“范围”和“精度”之间二选一。**

### 14.2 位分配图

下图按 `sign | exponent(范围) | mantissa(精度)` 画出常见格式的位布局，每个 `#` 代表 1 个 bit：

```text
              sign |         exponent (Range)        |                mantissa (Precision)
              -----+---------------------------------+-------------------------------------------
FP32  (1+8+23) [s] | ########  (e8)                   | #######################  (m23)
FP16  (1+5+10) [s] | #####     (e5)                   | ##########               (m10)
BF16  (1+8+7)  [s] | ########  (e8)                   | #######                  (m7)
FP8   E5M2     [s] | #####     (e5)                   | ##                       (m2)
FP8   E4M3     [s] | ####      (e4)                   | ###                      (m3)
```

对照着看几个要点：

- **FP32**（32 bit）：8 位指数 + 23 位尾数，范围和精度都很充足，是训练/推理的“黄金标准”基准；
- **FP16**（16 bit）：指数从 8 位砍到 5 位，**范围明显变小**（容易上溢/下溢），但保留了 10 位尾数，精度尚可；
- **BF16**（16 bit）：反过来，**保留 FP32 同样的 8 位指数**（范围和 FP32 一样大），代价是尾数只剩 7 位（精度更低）；
- **FP8 E5M2**（8 bit）：5 位指数、2 位尾数，**偏向范围**，常用于梯度等动态范围大的场景；
- **FP8 E4M3**（8 bit）：4 位指数、3 位尾数，**偏向精度**，常用于前向的权重/激活。

注意 FP16 和 BF16 都是 16 bit，占用显存一样，但“分配策略”完全不同：

```text
FP16:  范围小、精度较高   → 训练时容易溢出，常需 loss scaling
BF16:  范围大、精度较低   → 范围和 FP32 一致，训练更稳，几乎不用调 loss scaling
```

这就是为什么大模型训练**普遍偏爱 BF16**：宁可牺牲一点尾数精度，也要保住和 FP32 一样的数值范围，避免训练发散。

### 14.3 范围 vs 精度：直观理解

- **范围（Range）** 由指数位决定，回答“我能表示多大或多小的数而不溢出”；
- **精度（Precision）** 由尾数位决定，回答“两个相邻的数之间有多密，能分辨多小的差异”。

打个比方：

- 指数位像**尺子的量程**（能量 1 毫米还是 1 公里）；
- 尾数位像**尺子的刻度密度**（最小刻度是 1 毫米还是 1 厘米）。

位宽变小时，要么缩短量程，要么让刻度变粗，鱼与熊掌不可兼得。

### 14.4 各格式对比表

| 格式 | 总位数 | 符号 | 指数(范围) | 尾数(精度) | 特点 | 典型用途 |
|---|---:|---:|---:|---:|---|---|
| FP32 | 32 | 1 | 8 | 23 | 范围精度都足 | 基准、精度敏感层、累加 |
| TF32 | 19* | 1 | 8 | 10 | FP32 范围 + 低尾数 | Tensor Core 上加速 FP32 矩阵乘 |
| FP16 | 16 | 1 | 5 | 10 | 范围小、精度较好 | 混合精度训练/推理（需 loss scaling） |
| BF16 | 16 | 1 | 8 | 7 | 范围同 FP32、精度低 | 大模型训练首选 |
| FP8 E4M3 | 8 | 1 | 4 | 3 | 偏精度 | 前向权重/激活量化 |
| FP8 E5M2 | 8 | 1 | 5 | 2 | 偏范围 | 梯度等大动态范围 |
| INT8 | 8 | — | — | — | 定点整数 + scale | 推理量化 |

> \* TF32 是一种 Tensor Core 的计算模式，内部有效位约为 19 bit（8 位指数 + 10 位尾数 + 1 符号），并非独立的存储类型，存储时仍占 FP32 的 32 bit。

### 14.5 位宽、显存与速度的关系

位宽变小带来的直接收益，正好对应第 7、9 节讲的**内存墙**：

$$
\text{同样一块权重，位宽减半} \Rightarrow \text{显存占用减半，需要搬运的字节数减半}
$$

- 显存能装下更大的模型；
- 每次从显存搬数据的字节数变少，**缓解访存受限**；
- Tensor Core 对低精度有专门的高吞吐路径，矩阵算得更快。

所以在大模型推理里，FP16/BF16/FP8/INT8 量化几乎是标配。但要记住第 12 节的教训：**收益能否兑现，取决于减少的搬运和更高的吞吐，是否盖过量化/反量化的额外开销**，并且要验证精度是否可接受。

### 14.6 输入精度与累加精度是两回事

初学者常忽略：矩阵乘加里“输入用什么精度”和“累加用什么精度”可以不同。

$$
D_{ij} = C_{ij} + \sum_{k} A_{ik} B_{kj}
$$

典型做法是**输入低精度、累加高精度**，例如：

```text
输入 A、B:  FP16 / BF16 / FP8   (省显存、省带宽、算得快)
中间累加:   FP32              (避免长求和链上的误差累积)
输出 D:     FP16 / BF16 / FP32 (按需要)
```

这样既拿到了低精度的速度和显存收益，又用高精度累加保住了数值质量。这也是“混合精度（mixed precision）”这个词的由来——**不是把一切都降到最低精度，而是每一步用“刚好够用”的精度。**

---

## 15. 主流 GPU 规格对比

把前面所有概念放进真实数字里。以下是 NVIDIA 三代数据中心旗舰卡的官方标称值：

| | A100 (Ampere) | H100 (Hopper) | B200 (Blackwell) |
|---|---|---|---|
| FP64 | 9.7 TFLOPS | 34 TFLOPS | 37 TFLOPS |
| FP64 Tensor Core | 19.5 TFLOPS | 67 TFLOPS | 37 TFLOPS |
| FP32 | 19.5 TFLOPS | 67 TFLOPS | 75 TFLOPS |
| **TF32 Tensor Core** | 312 TFLOPS | 989 TFLOPS | 2.2 PFLOPS |
| **FP16/BF16 Tensor Core** | 624 TFLOPS | 1979 TFLOPS | 4.5 PFLOPS |
| **FP8 Tensor Core** | — | 3958 TFLOPS | 9 PFLOPS |
| **FP4 Tensor Core** | — | — | 18 PFLOPS |
| INT8 Tensor Core | 1248 TOPS | 3958 TOPS | 9 POPS |
| 显存 | 80GB HBM2e | 80GB HBM3 | 192GB HBM3e |
| 显存带宽 | 2 TB/s | 3.2 TB/s | 高达 7.7 TB/s |

> 注：Tensor Core 数值通常为开启稀疏（sparsity）的标称峰值，实际稠密算力约为一半；不同 SKU（SXM / PCIe）也有差异。这里只用来看数量级和趋势。

### 15.1 从这张表能读出什么

**（1）通用算力 vs Tensor Core 算力，差了一个数量级以上。**

以 H100 为例：

$$
\frac{1979\ \text{TFLOPS (BF16 Tensor Core)}}{67\ \text{TFLOPS (FP32 CUDA Core)}} \approx 30\times
$$

这就是为什么深度学习必须走 Tensor Core 路线，也是 `tensor_core_vs_cuda_core.md` 想讲清楚的事。

**（2）精度每降一半，算力大致翻一倍。**

在 H100 上：BF16 → FP8 是 1979 → 3958 TFLOPS，正好 2 倍。在 B200 上，FP8 → FP4 是 9 → 18 PFLOPS，同样 2 倍。这印证了第 14 节的结论：**低精度不只省显存，还直接换来算力。**

**（3）新精度类型是代际升级的核心卖点。**

```text
Ampere (A100)   : 引入 TF32、BF16 Tensor Core
Hopper  (H100)  : 新增 FP8 Tensor Core（配合 Transformer Engine）
Blackwell (B200): 新增 FP4 Tensor Core
```

模型越来越大，硬件的应对方式就是**不断往更低精度探**——只要能在可接受的精度损失下换来吞吐和显存。

**（4）算力增长比带宽增长快。**

从 A100 到 B200：

$$
\text{BF16 算力}: 624 \to 4500\ \text{TFLOPS}\ (\approx 7.2\times) \qquad
\text{显存带宽}: 2 \to 7.7\ \text{TB/s}\ (\approx 3.9\times)
$$

**算力涨得比带宽快，意味着「内存墙」一代比一代更严重。** 换算成算术强度门槛（跑满算力所需的 FLOP/Byte）：

$$
I_{\text{临界}} = \frac{P_{\text{峰值}}}{BW} : \quad \text{A100} \approx 312 \quad \to \quad \text{B200} \approx 584
$$

也就是说，在 B200 上一个 kernel 需要**每搬 1 字节做近 600 次浮点运算**才能跑满算力。这个门槛越来越高，所以算子融合、KV cache 优化、FlashAttention 这类「减少数据搬运」的技术只会越来越重要，而不是越来越不重要。

### 15.2 与消费级/工作站卡的对比

| | RTX A6000 | H100 SXM |
|---|---|---|
| 架构 | Ampere | Hopper |
| SM 数 | 84 | 132 |
| CUDA Core | 10,752 | 16,896 |
| FP32 | 38.7 TFLOPS | 67 TFLOPS |
| 显存 | 48GB GDDR6 | 80GB HBM3 |
| 显存带宽 | 768 GB/s | 3.2 TB/s |

注意**显存带宽差了 4 倍以上**，远超算力差距（约 1.7 倍）。这就是为什么大模型推理（访存受限为主）在数据中心卡上的优势，比纸面算力差距体现得更明显。

---

## 16. 常见术语速查表

| 术语 | 英文 | 一句话解释 |
|---|---|---|
| SM | Streaming Multiprocessor | GPU 的基本计算单元，一间“车间” |
| CUDA Core | — | 通用标量算术单元，做加减乘等 |
| Tensor Core | — | 矩阵乘加专用单元，深度学习主力 |
| Warp | — | 32 个线程一组，GPU 调度的基本单位 |
| SIMT | Single Instruction Multiple Threads | 同一指令、多个线程、各处理不同数据 |
| Thread / Block / Grid | — | 线程 / 线程块 / 全部线程的三层组织 |
| 寄存器 | Register | 最快最小的片上存储，线程私有 |
| 共享内存 | Shared Memory | Block 内线程共享的高速存储 |
| 显存 | HBM / GDDR / VRAM | GPU 的主内存，大但相对慢 |
| Kernel | — | 在 GPU 上运行的一个函数 |
| Host / Device | — | 分别指 CPU 端 / GPU 端 |
| 分支发散 | Warp Divergence | 同 Warp 走不同分支导致串行执行 |
| 算术强度 | Arithmetic Intensity | 每搬 1 字节数据能做多少次计算 |
| 计算受限 / 访存受限 | compute / memory bound | 瓶颈在算力还是在数据搬运 |
| 占用率 | Occupancy | SM 上活跃 Warp 数与最大可容纳数之比 |
| 处理分区 | Processing Block / Partition | SM 内部的 1/4，含 32 核 + 独立调度器和寄存器 |
| 寄存器文件 | Register File | 每 partition 64KB，Warp 上下文常驻其中 |
| GEMM | General Matrix Multiply | 通用矩阵乘，深度学习最主要的算子 |
| NVLink | — | GPU 间高速直连，约 112.5 GB/s |
| PCIe | Peripheral Component Interconnect Express | CPU↔GPU 通道，Gen4 x16 约 32 GB/s |
| nvcc | NVIDIA CUDA Compiler | 拆分并编译 host/device 代码的编译器 |
| PTX / SASS | — | CUDA 的中间表示 / 最终 GPU 机器码 |
| all-reduce | — | 多卡把梯度求和并广播回各卡的集合通信 |
| 全局内存 | Global Memory | `cudaMalloc` 分配的显存，所有线程可见、跨 kernel 存活 |
| `__global__` | — | kernel 修饰符：Host 调用、Device 执行，必须返回 void |
| `__device__` | — | GPU 内部辅助函数：Device 调用、Device 执行 |
| `__shared__` | — | 声明共享内存，Block 内所有线程可见 |
| `__syncthreads()` | — | Block 内屏障，等所有线程到齐再继续（跨 Block 无效） |
| `dim3` | — | 三维尺寸类型，用于指定 Grid/Block 维度 |
| 内置索引变量 | gridDim / blockIdx / blockDim / threadIdx | 「总共多少」与「我是第几个」的四个内置量 |
| 合并访问 | Memory Coalescing | 同 Warp 的 32 个线程访问连续地址，合并成一次内存事务 |
| H2D / D2H | Host to Device / Device to Host | 系统内存↔显存的拷贝方向 |
| TMA | Tensor Memory Accelerator | Hopper 新增的异步大块搬运硬件单元 |

---

## 17. 常见误区

### 误区 1：GPU 就是“更快的 CPU”

错误。GPU 单核往往比 CPU 慢，它靠的是**海量并行**。串行的、分支多的任务，CPU 反而更快。

### 误区 2：CUDA Core 越多性能一定越强

错误。性能还取决于显存带宽、Tensor Core、时钟频率、架构代际和实际负载。不同代 GPU 的 Core 数不能直接横向比较。

### 误区 3：算力（TFLOPS）越高，我的程序就越快

错误。很多程序是**访存受限**的，提高算力没用，关键是减少数据搬运、提高数据复用。

### 误区 4：CPU 内存和 GPU 显存是同一块

错误。它们通常是分开的，数据要显式拷贝。频繁的 Host↔Device 拷贝常常是性能杀手。

### 误区 5：线程越多越好

错误。线程太多会挤占寄存器和共享内存，降低占用率，甚至变慢。要在并行度和资源占用之间平衡。

### 误区 6：一个 Block 可以跨多个 SM 运行

错误。一个 Block 只会被分配到**一个** SM 上执行，这也是为什么跨 Block 不能直接同步。

### 误区 7：Warp 切换和 CPU 线程切换一样昂贵

错误。恰恰相反。Warp 的上下文（PC、寄存器、共享内存）在生命周期内**常驻 SM**，切换不需要保存/恢复状态，几乎零开销。这正是 GPU 敢开上万个线程的前提。

### 误区 8：只要卡够多，多卡就一定线性加速

错误。多卡要同步梯度/激活，走 PCIe（32 GB/s）还是 NVLink（112.5 GB/s）差别巨大，跨 CPU socket 更差。通信开销跟不上就会严重偏离线性扩展。

### 误区 9：kernel 代码必须写得“并行”

错误。kernel 代码本身是**串行**的——它只描述一个线程做什么。并行性来自硬件同时启动上万个线程执行同一份代码。你要做的是用线程索引正确地划分数据。

### 误区 10：Block 线程数可以随便设

不建议。**Block 线程数应为 32 的倍数**，因为硬件按 32 线程的 Warp 分配和调度。设成 100 会占满 4 个 Warp（128 个槽位），其中 28 个线程全程空转。另外有个硬上限：**每个 Block 最多 1024 个线程**，超了会静默启动失败。

### 误区 11：`cudaMalloc` 返回的指针可以在 CPU 上解引用

错误。`dA` 这个**指针变量**住在 CPU 内存里，但它保存的是**显存地址**。`dA[0]` 会段错误，必须通过 `cudaMemcpy` 搬运。

### 误区 12：kernel 启动后马上就执行完了

错误。`kernel<<<>>>` 是**异步**的，立刻返回，CPU 继续往下跑。计时或读结果前需要 `cudaDeviceSynchronize()`（同步版 `cudaMemcpy` 会隐式等待）。这也是为什么“不加同步测出来的 kernel 耗时接近 0”。

### 误区 13：CUDA 出错会抛异常或崩溃

错误。CUDA API **只返回错误码**，不检查等于看不见 bug。kernel 需要检查两次：`cudaGetLastError()` 抓启动配置错误，`cudaDeviceSynchronize()` 抓执行期错误（如越界访问）。

### 误区 14：二维/三维 grid 是硬件特性

错误。多维只是**编程便利**，方便映射矩阵、图像的坐标。硬件最终都会把它们展平成线性的 Warp 序列。

### 误区 15：`threadIdx.x` 和 `threadIdx.y` 谁对应行谁对应列无所谓

错误，而且代价极大。必须让 **`threadIdx.x` 对应内存连续的方向**（行主序下是列 `j`），这样同 Warp 的 32 个线程访问 32 个连续地址，硬件能**合并成一次内存事务**。写反了需要 32 次事务，带宽利用率掉到 1/32。

---

## 18. 下一步学习建议

建议按以下顺序继续深入：

1. **本目录 `tensor_core_vs_cuda_core.md`**：搞清楚两种 Core 的分工，这是理解深度学习性能的关键；
2. **动手写第一个 CUDA 程序**：照着第 11 节的向量加法和矩阵加法敲一遍，亲手体会 Grid/Block/Thread 和五个 API；
3. **写一个朴素矩阵乘，再优化它**：这是 CUDA 学习的经典路径——
   - v1：每个线程算一个 $C_{ij}$，直接读全局内存（会发现很慢）；
   - v2：分块（tiling）+ 共享内存，让每块数据被复用 $T$ 次；
   - v3：调整访存模式实现合并访问，消除 bank conflict；
   - v4：与 cuBLAS 对比，看还差多少倍；
4. **学习内存优化**：共享内存、分块（tiling）、合并访问（coalesced access）、bank conflict；
5. **认识性能分析工具**：用 Nsight Compute / Nsight Systems 看真实瓶颈，而不是靠猜；
6. **理解 Roofline 模型**：判断程序是计算受限还是访存受限；
7. **尝试更现代的写法**：Triton（Python 写 kernel，自动处理很多细节）、CUTLASS（模板化 GEMM）；
8. **走向实际应用**：阅读 FlashAttention、GEMM、MoE 等真实高性能 kernel 的设计思想。

推荐阅读：《Programming Massively Parallel Processors》（第 4 版）第 2~4 章，是 CUDA 编程模型讲得最系统的教材。

学习时始终带着两个问题：

1. 这段计算**能不能并行**、并行度够不够？
2. 数据搬运是不是太多、**能不能复用**？

---

## 19. 总结

理解 GPU 架构，可以归纳为几条主线：

1. **神经网络最终只落成三类算子**：矩阵乘（GEMM）、逐元素运算、归约运算。所有 GPU 优化都是为这三类服务的。
2. **性能是整机的事**：显存带宽（数百 GB/s ~ 几 TB/s）≫ NVLink（112.5 GB/s）> PCIe（32 GB/s）。数据能否及时到位，和算力同等重要。
3. **GPU 为吞吐而生**：用海量较弱的执行单元同时处理数据，而不是让单核跑得飞快。同功耗下算力可达 CPU 的数十倍，差距全部来自并行度。
4. **SM 是基本计算单元**，内部再分 4 个 partition，每个 partition 有 32 个核、独立的 Warp Scheduler 和 64KB 寄存器 → 每 SM 128 核。
5. **Warp 是调度单位**：32 个线程以 SIMT 方式执行同一条指令。warp size = 32 正是 partition 宽度的体现，因此要避免分支发散、Block 线程数取 32 的倍数。
6. **Warp 切换零开销**：上下文常驻 SM，所以能用海量并发掩盖内存延迟；代价是资源限制了驻留 Warp 数（占用率）。
7. **软件分三层**：Grid → Block → Thread，Block 落在单个 SM 上、块内可共享内存与同步；Block 间互不依赖，这才让同一份代码能跨不同规模的 GPU 运行。
8. **内存有层级**：寄存器（线程私有）、共享内存（Block 内）、全局内存（全部可见且跨 kernel 存活）——作用域和速度是一致的，越快越私有。
9. **CUDA 编程只有五步**：`cudaMalloc` → `cudaMemcpy`(H2D) → `kernel<<<Dg, Db>>>` → `cudaMemcpy`(D2H) → `cudaFree`。kernel 用 `__global__` 修饰、必须返回 void、启动是异步的。
10. **索引是 CUDA 的核心技能**：`i = blockIdx.x * blockDim.x + threadIdx.x`；多维要手工展平（x 变化最快）；永远写边界检查 `if (i < n)`。
11. **两个硬性约束**：每 Block ≤ 1024 线程、Block 线程数取 32 的倍数；Block 数量要明显多于 SM 数，否则大量 SM 空转。
12. **合并访问是最容易忽略的大优化**：让 `threadIdx.x` 对应内存连续方向，32 个线程的访问才能合并成一次事务，差距可达 32 倍。
13. **性能有两堵墙**：算力墙和内存墙，很多程序其实卡在内存带宽上。而且算力增长快于带宽增长，**内存墙一代比一代更严重**。
14. **低精度是硬件的主要演进方向**：精度每降一半，Tensor Core 算力大致翻倍；BF16、FP8、FP4 依次成为新代际的卖点。混合精度的要点是「输入低精度、累加高精度」。
15. **搬运与计算要重叠**：从异步 `cudaMemcpyAsync` 到 Hopper 的 TMA，现代 GPU 优化的主线之一就是让数据搬运不阻塞计算。
16. **深度学习天生契合 GPU**：矩阵运算高度并行、数据复用高、控制流规整，因此能充分发挥 GPU 尤其是 Tensor Core 的能力。

最后用两句话收尾：

> **学 GPU 架构，先建立"大规模并行 + 内存层级"这两个直觉；具体的 Core、指令和架构细节，都是围绕这两点服务的。**
>
> **学 CUDA 编程，先想清楚"单个线程负责哪一份数据"，再用索引把它算出来；并行性由硬件提供，你只需要正确地划分。**

