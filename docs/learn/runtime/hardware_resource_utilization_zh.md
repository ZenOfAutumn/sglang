# 硬件资源利用率：观测方法、指标与含义

> 本文回答四个问题：**一套推理服务要看哪些硬件**、**每类硬件用哪些工具观测**、**每个指标的口径是什么**、**指标异常时怎么定位**。
>
> 定位：`docs/learn/runtime/` 下的运行期主题之一。与 `cpu_affinity_numa_zh.md`（CPU 亲和性 / NUMA）、`cuda_stream_concept_zh.md`（CUDA Stream）互补——那两篇讲「怎么调」，本文讲「怎么测、怎么看」。
>
> 约定：命令与参数以本仓库当前版本为准（SGLang 源码见 `python/sglang/`）；Prometheus 指标名均来自 `python/sglang/srt/observability/metrics_collector.py`，行号对应当前版本。不同 GPU 架构（Hopper / Blackwell / 国产卡）对部分硬件计数器的支持有差异，遇到「指标读不到」以硬件与驱动文档为准。

---

## 0. 一页速览

| 资源 | 核心问题 | 首选工具 | 关键指标 | 打满的信号 |
| --- | --- | --- | --- | --- |
| GPU 计算 | 算力是否用满 | `nvidia-smi dmon`、DCGM、Nsight | SM 利用率、Tensor Core 活跃度 | SM/Tensor 持续 >90%，且是瓶颈 |
| GPU 显存 | 容量 / 带宽是否够 | `nvidia-smi`、DCGM、torch memory | memory.used、DRAM 活跃度 | 显存接近上限或 DRAM >90% |
| GPU 互连 | 卡间通信是否成瓶颈 | `nvidia-smi nvlink`、NCCL Tools | NVLink / PCIe 吞吐 | 通信时间占比高 |
| CPU | 是否够喂给 GPU | `mpstat`、`pidstat`、`top -H` | `%usr`/`%sys`/`%iowait`/`%steal` | 单核跑满、调度延迟高 |
| 内存 | 容量 / 带宽 / NUMA | `free`、`vmstat`、`numastat` | RSS、page cache、跨 node 比例 | swap 抖动、跨 NUMA 高 |
| 网络 | 带宽 / 延迟 / 丢包 | `sar -n DEV`、`ethtool -S`、`ibstat` | 带宽、重传、drop | 重传率上升、队列 drop |
| 存储 | IOPS / 带宽 / 时延 | `iostat -x`、`iotop` | `%util`、`await`、IOPS | `%util`≈100% 且 `await` 高 |
| 服务级（SGLang） | 吞吐 / 延迟 / KV 用量 | `GET /metrics` | token_usage、TTFT/ITL、吞吐 | 队列堆积、KV 打满 |

三句话总结：

1. **先服务级、后硬件级**——先看 `/metrics` 判断是「算不动」还是「排不上」，再用硬件工具判断「哪个部件顶到天花板」。
2. **利用率高 ≠ 瓶颈**：SM 利用率 100% 只说明「有活可干」，真正决定性能的是**算力利用率（MFU）与带宽利用率（MBU）谁先到顶**。
3. **观测要成对看**：只看 GPU 会漏掉 CPU/网络瓶颈，只看平均会漏掉长尾，**吞吐必须和延迟一起看**。

---

## 1. 观测的三层视角

| 视角 | 观测对象 | 回答的问题 | 典型工具 / 接口 |
| --- | --- | --- | --- |
| **服务级（L7）** | 请求、token、KV 缓存 | 服务能力到顶了吗？延迟达标吗？ | SGLang `GET /metrics`、`bench_serving` |
| **进程级（L5–L6）** | 各类算子、kernel、通信 | 时间花在哪？谁在等谁？ | torch profiler、`/start_profile`、Nsight Systems |
| **硬件级（L0–L4）** | SM、DRAM、PCIe、NVLink、CPU、网卡、盘 | 物理资源顶到天花板了吗？ | `nvidia-smi`、DCGM、`mpstat`、`iostat` |

**推荐排查顺序**：服务级发现问题 → 进程级定位阶段（prefill / decode / 排队 / 通信）→ 硬件级确认到底是哪个物理资源饱和。

---

## 2. 指标体系总览

按「一个请求的一生」拆解，每个阶段都对应一组指标：

```
请求到达 ──► 排队等待 ──► Prefill(算) ──► Decode(访存为主) ──► 返回
   │            │             │                │               │
 请求速率    队列长度      算力利用率        显存带宽利用率    网络/下发开销
 到达分布    TTFT           SM/Tensor         DRAM/显存        PCIe/NVLink
            queue_time     MFU                MBU
```

| 阶段 | 服务级指标 | 硬件级指标 |
| --- | --- | --- |
| 排队 | `num_queue_reqs`、`queue_time_seconds` | CPU 调度、`%steal` |
| Prefill | TTFT、`uncached_prompt_tokens` | 算力（Tensor Core）、MFU |
| Decode | ITL/TPOT、`decode_sum_seq_lens` | 显存带宽、MBU、KV 占用 |
| 返回/通信 | E2E latency | NVLink/PCIe 吞吐、网络带宽 |

---

## 3. GPU 观测

GPU 是 LLM 推理的主战场，分成 **计算 / 访存 / 显存容量 / 互连 / 功耗温度** 五块来看。

### 3.1 观测工具矩阵

| 工具 | 粒度 | 采样开销 | 适用场景 |
| --- | --- | --- | --- |
| `nvidia-smi` | 秒级快照 | 极低 | 快速看一眼利用率、显存、功耗 |
| `nvidia-smi dmon` | 秒级连续 | 极低 | 压测时的实时曲线（推荐首选） |
| `nvidia-smi pmon` | 秒级进程级 | 极低 | 多进程共享卡时定位是哪个进程 |
| DCGM / `dcgm-exporter` | 细粒度 + Prometheus | 低 | 长时间稳定的生产监控 |
| Nsight Systems | kernel 级时间线 | 中 | 定位 kernel 空档、通信、拷贝重叠 |
| Nsight Compute | 单 kernel 计数器 | 高 | 单 kernel 是 compute-bound 还是 memory-bound |
| torch profiler | PyTorch 算子级 | 中 | 与 SGLang `/start_profile` 联动 |

常用的实时观测命令：

```bash
# 默认快照：GPU 利用率、显存、功耗、温度
nvidia-smi

# 连续监控：p=功耗/u=利用率/c=时钟/v=显存/m=显存带宽/e=ECC/t=温度
# -d 采样间隔(秒)，-c 采样次数
nvidia-smi dmon -s pucvmet -d 1

# 进程级：看清是哪个进程在占卡
nvidia-smi pmon -s um -d 1

# 显存明细：每张卡总量/已用/空闲
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
           --format=csv -l 1

# NVLink 吞吐（Hopper/Ampere 多卡）
nvidia-smi nvlink -s      # 状态与速率
nvidia-smi nvlink -gt d   # 每链路吞吐累计值
```

### 3.2 计算利用率

| 指标 | `nvidia-smi` 字段 | DCGM 字段 | 含义 |
| --- | --- | --- | --- |
| GPU 利用率 | `utilization.gpu` | `DCGM_FI_DEV_GPU_UTIL` | **采样窗口内有 kernel 在跑的时间占比** |
| SM 活跃度 | — | `DCGM_FI_PROF_SM_ACTIVE` | 真正有 warp 在执行的 SM 比例 |
| SM 占用率 | — | `DCGM_FI_PROF_SM_OCCUPANCY` | warp 占用相对理论最大值的比例 |
| Tensor Core 活跃度 | — | `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | Tensor Core 流水线活跃比例 |
| FP32/FP16 流水线 | — | `DCGM_FI_PROF_PIPE_FP32_ACTIVE` 等 | 各计算流水线活跃比例 |

**关键口径与误区**：

- `utilization.gpu` 只是「**时间维度**上有没有 kernel」，**不是算力用了百分之多少**。一个只做了很少计算的 kernel 也能让利用率显示 100%。
- 判断算力是否真正吃满要看 **Tensor Core 活跃度**和 **MFU**（见 §3.7）。解码阶段即使 `utilization.gpu` 很高，MFU 也可能只有个位数——因为它本就是访存瓶颈。
- 用 Nsight Compute 看单 kernel 时，关键结论字段是：

```bash
# 判断单 kernel 是算力受限还是访存受限
ncu --set full -k <kernel_name> ... 
# 关注: Compute (SM) Throughput vs Memory Throughput,
#       以及 Roofline 提示 "Compute Bound" / "Memory Bound"
```

### 3.3 访存利用率（显存带宽）

| 指标 | `nvidia-smi` 字段 | DCGM 字段 | 含义 |
| --- | --- | --- | --- |
| 显存控制器利用率 | `utilization.memory` | `DCGM_FI_DEV_MEM_COPY_UTIL` | 采样窗口内显存控制器忙的时间占比 |
| DRAM 活跃度 | — | `DCGM_FI_PROF_DRAM_ACTIVE` | DRAM 读写活跃比例（**带宽饱和度看这个**） |

自建服务时，也可以用 PyTorch 直接读进程内显存：

```python
import torch
torch.cuda.memory_allocated()      # 当前张量实际占用（字节）
torch.cuda.memory_reserved()       # 缓存分配器向驱动申请的总量（含碎片）
torch.cuda.max_memory_allocated()  # 峰值占用，用于估容量
```

- `allocated ≤ reserved ≤ nvidia-smi 的 used`。三者差额是「PyTorch 缓存池预留但未使用」+「非 PyTorch 分配（如 NCCL、CUDA Graph、cuBLAS workspace）」。
- **decode 阶段通常是 memory-bound**：每生成 1 个 token 要把全部权重从显存读一遍，所以 `DRAM_ACTIVE` 更容易先到顶。

### 3.4 显存容量

| 指标 | 来源 | 含义 |
| --- | --- | --- |
| `memory.used` / `memory.total` | `nvidia-smi` | 整卡已用 / 总量 |
| `DCGM_FI_DEV_FB_USED` / `FB_FREE` | DCGM | 帧缓冲（显存）已用 / 空闲 |
| KV cache 占用 | SGLang `token_usage` | 推理缓存占 KV 池的比例（见 §8） |
| 显存碎片 | reserved − allocated | 分配器碎片，影响可分配上限 |

### 3.5 互连（NVLink / PCIe）

| 指标 | 来源 | 含义 |
| --- | --- | --- |
| NVLink 每链路吞吐 | `nvidia-smi nvlink -gt d` | 每方向累计字节数 |
| PCIe 吞吐 | DCGM `DCGM_FI_PROF_PCIE_TX_BYTES` / `RX_BYTES` | 经 PCIe 收发字节速率 |
| NVLink 吞吐 | DCGM `DCGM_FI_PROF_NVLINK_TX_BYTES` / `RX_BYTES` | 经 NVLink 收发字节速率 |
| 集合通信带宽 | `nccl-tests`（`all_reduce_perf` 等） | 实测 bus bandwidth |

**关注点**：张量并行（TP）每层都要 AllReduce，流水线并行（PP）要做点对点传输，专家并行（EP/MoE）做 All-to-All。如果通信与应用没有重叠（overlap），GPU 会出现周期性空档——表现为 `utilization.gpu` 锯齿状、均值偏低。

### 3.6 功耗 / 温度 / 时钟

| 指标 | 字段 | 含义 |
| --- | --- | --- |
| 功耗 | `power.draw` / `DCGM_FI_DEV_POWER_USAGE` | 实时功耗；撞 `power.limit` 会降频 |
| 温度 | `temperature.gpu` | 撞温度墙会降频 |
| SM 时钟 | `clocks.sm` / `clocks.max.sm` | 实际 vs 最大频率 |
| 降频原因 | `clocks_throttle_reasons.*` | 功耗/温度/其他原因的降频标志 |

**降频是隐形的性能杀手**：利用率指标看起来正常，但吞吐莫名下降时，务必检查是否撞了功耗墙或温度墙。

### 3.7 MFU / MBU 与 Roofline

这是判断「GPU 到底发挥了几成」的核心方法。

**峰值口径**（以 bf16 为例）：

- 峰值算力 $F_{peak}$：厂商标称的 bf16 Tensor FLOPS（注意是否含稀疏，稀疏值一般是稠密的 2 倍）。
- 峰值带宽 $B_{peak}$：HBM 标称带宽。

**利用率的两个定义**：

$$
\text{MFU}=\frac{F_{achieved}}{F_{peak}},\qquad
\text{MBU}=\frac{B_{achieved}}{B_{peak}}
$$

其中

$$
F_{achieved}\approx \frac{2\,N_{active}\,T}{t}
$$

- $N_{active}$：参与计算的激活参数量（不含 MoE 未激活专家）；
- $T$：处理的 token 数（prefill + decode 全部）；
- $t$：墙钟时间；
- 系数 2：每个参数每 token 一次乘加（MAC）= 2 FLOPs。注意这忽略了 attention 的二次项，是工程近似。

$$
B_{achieved}=\frac{\text{总访存字节}}{t}
$$

**Roofline 模型**把两者统一起来。定义算术强度（每字节访存能换来多少计算）：

$$
I=\frac{\text{FLOPs}}{\text{Bytes}}
$$

在 Roofline 图上的拐点（ridge point）为：

$$
I_{ridge}=\frac{F_{peak}}{B_{peak}}
$$

- 当 $I > I_{ridge}$：**compute-bound**，性能上限是 $F_{peak}$，优化方向是减少冗余计算、提高 Tensor Core 利用率；
- 当 $I < I_{ridge}$：**memory-bound**，性能上限是 $I \times B_{peak}$，优化方向是提高数据复用、量化减字节。

**LLM 的典型结论**：

| 阶段 | 算术强度 | 瓶颈 | 优化方向 |
| --- | --- | --- | --- |
| Prefill（长 prompt） | 高 | compute-bound | 提升 MFU、算子融合 |
| Decode（batch 小） | 极低（约等于 $2\times$参数量 / 权重字节） | memory-bound | 增 batch、量化、投机解码、减少权重读取 |

这也解释了为什么 **decode 阶段靠「堆并发」提吞吐**——把 memory-bound 的权重读取摊到更多请求上。

SGLang 提供了估算 MFU 的 Prometheus 指标，用 `--enable-mfu-metrics` 打开（见 §8）。

---

## 4. CPU 观测

LLM 推理里的 CPU 负责：tokenize/detokenize、调度、采样、请求收发、KV 传输的发起。**CPU 一旦成为瓶颈，GPU 会出现「等数据」的空档。**

### 4.1 指标与含义

| 指标 | 含义 | 健康范围参考 |
| --- | --- | --- |
| `%usr` | 用户态 CPU 时间占比 | 与负载正相关 |
| `%sys` | 内核态占比 | 过高说明系统调用/上下文切换多 |
| `%iowait` | 等 IO 的空闲占比 | 持续偏高说明 IO 是瓶颈 |
| `%steal` | 被虚拟化/其他租户抢走的占比 | 云上 >5% 需警惕 |
| `%soft` | 软中断占比 | 网络密集时升高 |
| run queue 长度 | 可运行但等待 CPU 的线程数 | 持续 > 核数说明 CPU 不够 |
| context switch | 上下文切换次数 | 过高说明锁竞争/调度抖动 |
| load average | 1/5/15 分钟平均负载 | > 逻辑核数需关注 |

### 4.2 观测命令

```bash
# 总体：观察 %usr/%sys/%iowait/%steal/%soft
vmstat 1

# 每核明细：找热点核
mpstat -P ALL 1

# 进程级 + 线程级：找热点线程（SGLang 有 scheduler / detokenizer 等多线程）
pidstat -u -t -p <pid> 1

# top 的线程视图：Shift+H 展开线程
top -H -p <pid>

# 查看进程 CPU 亲和性（是否钉核，配合 cpu_affinity_numa_zh.md）
taskset -cp <pid>
```

### 4.3 SGLang 进程 CPU 指标

SGLang 暴露 `sglang:process_cpu_seconds_total`（带 `component` 标签），含义是该进程累计消耗的 CPU 秒数（user + system）：

```python
cpu_seconds_total = Counter(
    name="sglang:process_cpu_seconds_total",
    documentation="Total CPU time consumed by this process (user + system)",
    labelnames=["component"],
)
```

它的**导数**（`rate(...)`）就是该组件的 CPU 核数占用。多个 component 相加接近机器总核数时，说明 CPU 已成为瓶颈。

---

## 5. 内存观测

| 指标 | 含义 | 关注点 |
| --- | --- | --- |
| RSS | 进程实际驻留物理内存 | 是否接近机器容量 |
| VSZ | 虚拟地址空间 | 通常不用太关注 |
| page cache | 文件页缓存 | 模型权重从盘加载时有用 |
| available | 内核估算的可用内存 | **比 `free` 更准** |
| swap in/out (`si`/`so`) | 换入换出速率 | 非 0 就是危险信号 |
| HugePages | 大页使用 | 大页可降低 TLB miss |
| NUMA 命中 | 本地/远端访问比例 | 远端访问高会拖慢带宽 |

```bash
free -h                          # 总体
vmstat 1                         # si/so 换页、cache
numastat -p <pid>                # 每 NUMA node 的内存分布
numastat -m                      # 系统级 NUMA 内存
cat /proc/<pid>/numa_maps | head # 逐内存区映射
cat /proc/meminfo | grep -i huge # 大页
```

**与 GPU 的关系**：模型权重常先加载到 host 内存再 H2D 拷贝；CPU 内存带宽和 NUMA 亲和性会影响加载速度（见 `cpu_affinity_numa_zh.md`）。运行期如果开了 host 侧 KV offload，host 内存带宽也会进热路径。

---

## 6. 网络观测

分布式推理（PD 分离、多机 TP/PP/EP）对网络非常敏感。

| 指标 | 含义 | 健康信号 |
| --- | --- | --- |
| 带宽 `rxkB/s`/`txkB/s` | 收发字节速率 | 接近网卡线速即饱和 |
| PPS | 每秒包数 | 小包多时可能先到 PPS 上限 |
| 重传率 | TCP retransmit 占比 | 应接近 0 |
| 丢包/溢出 | `rx_dropped`、`tx_dropped`、`rx_fifo_errors` | 应为 0，非 0 说明队列溢出 |
| RDMA/IB 速率 | InfiniBand 端口速率 | 接近线速即饱和 |

```bash
sar -n DEV 1                    # 各网卡带宽/包速率
sar -n EDEV 1                   # 错误与丢包
ethtool -S <iface>              # 网卡详细计数（drop、error、retrans）
ethtool <iface>                 # 链路速率与双工
ip -s link                      # 快速看收发包与错误
ibstat                          # InfiniBand 端口状态与速率（若有）
# 集合通信实测
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8
```

**与 GPU 的关系**：跨机通信走 RDMA 网卡，卡间走 NVLink/PCIe。网络瓶颈会让 GPU 空闲，表现为 `utilization.gpu` 周期性掉底。

---

## 7. 存储 IO 观测

存储主要影响**模型加载**、**KV offload / prefix cache 落盘**、**日志与 trace 落盘**。

| 指标 | 含义 | 关注点 |
| --- | --- | --- |
| `r/s`、`w/s` | 每秒读/写 IOPS | 是否到设备上限 |
| `rkB/s`、`wkB/s` | 读写带宽 | 是否到设备上限 |
| `await` | 每个 IO 平均耗时（含排队） | 越高越慢 |
| `%util` | 设备忙碌时间占比 | ≈100% 说明饱和 |
| `r_await`/`w_await` | 读/写分别的时延 | 区分读还是写慢 |
| `aqu-sz` | 平均队列深度 | 高 + 高 await = 拥塞 |

```bash
iostat -x 1        # 扩展指标，看 %util / await / aqu-sz
iostat -dxm 1      # 以 MB 为单位
iotop -o           # 只看有 IO 的进程
pidstat -d 1       # 进程级 IO
```

**注意**：NVMe SSD 上 `%util` 经常显示 100% 但仍有富余（因为并行队列），此时应以 **`await` 和 `aqu-sz`** 为准，`%util` 在 NVMe 上参考价值有限。

---

## 8. 服务级观测：SGLang Prometheus 指标

硬件指标告诉你「物理资源用了多少」，服务级指标告诉你「服务产出了多少」。SGLang 用 `--enable-metrics` 打开 Prometheus 端点：

```bash
python -m sglang.launch_server \
  --model-path <model> \
  --enable-metrics \
  --enable-mfu-metrics      # 额外暴露 MFU 相关指标（可选）
```

打开后，通过 `GET http://<host>:<port>/metrics` 抓取（端点由 `add_prometheus_middleware` 挂载，路径 `/metrics`）。

### 8.1 关键指标清单

**请求与队列**（`SchedulerMetricsCollector`）：

| 指标名 | 类型 | 含义 |
| --- | --- | --- |
| `sglang:num_running_reqs` | Gauge | 当前运行中的请求数 |
| `sglang:num_queue_reqs` | Gauge | 当前排队等待的请求数 |
| `sglang:num_grammar_queue_reqs` | Gauge | 受结构化输出约束排队数 |
| `sglang:gen_throughput` | Gauge | 生成吞吐（token/s） |
| `sglang:num_requests_total` | Counter | 累计处理请求数（带 `is_streaming`） |
| `sglang:num_aborted_requests_total` | Counter | 累计中止请求数 |

**KV 缓存**：

| 指标名 | 类型 | 含义 |
| --- | --- | --- |
| `sglang:token_usage` | Gauge | KV 池用量比例（0–1），**推理侧最关键的容量指标** |
| `sglang:full_token_usage` | Gauge | 含 SWA/mamba 的完整用量 |
| `sglang:kv_used_tokens` / `kv_available_tokens` / `kv_evictable_tokens` | Gauge | KV token 已用/可用/可淘汰 |
| `sglang:num_used_tokens` | Gauge | 实际使用的 token 数 |
| `sglang:cache_hit_rate` | Gauge | 前缀缓存命中率 |
| `sglang:cached_tokens_total` | Counter | 按来源（device/host/storage）统计的缓存命中 token |
| `sglang:decode_sum_seq_lens` | Gauge | 解码阶段所有序列长度之和（≈ decode batch 的 KV 规模） |
| `sglang:swa_*` / `sglang:mamba_*` | Gauge | 滑窗注意力 / Mamba 状态缓存用量 |

**延迟与 token 长度**（关键直方图，`_` 后缀为 Prometheus 自动生成的 `_bucket` / `_sum` / `_count`）：

| 指标名 | 含义 |
| --- | --- |
| `sglang:time_to_first_token_seconds` | TTFT，首 token 延迟（prefill + 排队） |
| `sglang:inter_token_latency_seconds` | ITL，相邻 token 间隔（decode 单步时延） |
| `sglang:e2e_request_latency_seconds` | E2E，端到端请求延迟 |
| `sglang:queue_time_seconds` | 排队等待时长 |
| `sglang:prompt_tokens_histogram` | 输入长度分布 |
| `sglang:uncached_prompt_tokens_histogram` | 需真正计算的输入长度（去掉前缀命中） |
| `sglang:generation_tokens_histogram` | 输出长度分布 |

**CPU / 启动**：

| 指标名 | 含义 |
| --- | --- |
| `sglang:process_cpu_seconds_total` | 进程累计 CPU 秒数（`component` 维度） |
| `sglang:startup_time_seconds` | 启动各阶段耗时（`phase`: load_weight / kv_cache_allocation / scheduler_e2e / tokenizer_e2e） |
| `sglang:startup_cuda_graph_time_seconds` | CUDA Graph 捕获耗时 |

**PD 分离 / KV 传输**（如有）：

| 指标名 | 含义 |
| --- | --- |
| `sglang:kv_transfer_speed_gb_s` | KV 传输速率 |
| `sglang:kv_transfer_latency_ms` | KV 传输时延 |
| `sglang:kv_transfer_total_mb` | 传输总量 |

### 8.2 指标口径与计算

从直方图算分位延迟（Prometheus 查询）：

```promql
# TTFT P95（秒）
histogram_quantile(0.95,
  sum(rate(sglang:time_to_first_token_seconds_bucket[1m])) by (le))

# 平均 ITL = 所有 token 间隔之和 / token 间隔次数
rate(sglang:inter_token_latency_seconds_sum[1m])
  / rate(sglang:inter_token_latency_seconds_count[1m])

# 平均 TTFT
rate(sglang:time_to_first_token_seconds_sum[1m])
  / rate(sglang:time_to_first_token_seconds_count[1m])
```

**为什么必须看分位数而不是只看均值**：LLM 请求长度差异大，均值会被长尾或短请求掩盖。**P95/P99 才是 SLA 的真实体感**，也是判断「队列是否开始堆积」的敏感信号。

**吞吐与延迟的关系**（利特尔法则 Little's Law）：

$$
L = \lambda \times W
$$

- $L$：系统内并发请求数（≈ `num_running_reqs` + `num_queue_reqs`）；
- $\lambda$：到达速率（req/s）；
- $W$：单请求平均驻留时间（≈ E2E latency）。

当 $\lambda$ 继续增大而 $L$ 已受并发上限约束时，$W$ 会线性上升——这就是「吞吐到顶、延迟起飞」的拐点。判断饱和应看：**`num_queue_reqs` 持续 >0 且 TTFT 快速上升**。

### 8.3 自定义压测与观测

```bash
# 随机数据集压测：固定输入/输出长度与请求数
python -m sglang.bench_serving \
  --backend sglang \
  --num-prompts 1000 \
  --random-input-len 1024 \
  --random-output-len 256 \
  --request-rate inf        # inf = 一次性全部发送，测最大吞吐
                            # 有限值 = 泊松到达，测延迟-吞吐曲线
```

关键参数口径（`bench_serving`）：

| 参数 | 含义 |
| --- | --- |
| `--num-prompts` | 总请求数 |
| `--random-input-len` / `--random-output-len` | 随机数据集的输入 / 输出长度 |
| `--random-range-ratio` | 长度采样比例范围（0 = 固定长度） |
| `--request-rate` | 到达速率（req/s）；`inf` 表示全部同时发送 |
| `--max-concurrency` | 并发上限（模拟上层限流） |

压测输出的核心指标（与 §8.1 对应）：`Request throughput (req/s)`、`Input/Output/Total token throughput (tok/s)`、`Mean TTFT (ms)`、`Mean TPOT (ms)`、`Mean E2E Latency (ms)`。

**扫描方法**：固定输入/输出长度，从低到高调 `--request-rate`，记录每个点的吞吐与 P95 延迟，画出「吞吐-延迟曲线」，拐点即为服务的最佳工作点。

### 8.4 进程级 profiling

```bash
# 启动 GPU profiler（Nsight Systems 等），可与压测配合
curl -X POST http://<host>:<port>/start_profile
# ... 执行压测 ...
curl -X POST http://<host>:<port>/stop_profile
```

配合硬件工具（`nvidia-smi dmon`、`nsys`、`ncu`）即可定位「时间花在哪个 kernel / 通信 / 拷贝」。

### 8.5 压测后的清理：只 flush，不重启

**规则**：每轮压测结束后，若**没有改动部署参数**（模型、并行度、显存分配、量化、调度策略等），**只需清空 KV 缓存（前缀缓存），无需重启模型服务**。重启模型意味着重新加载权重、重建 CUDA Graph、重新预热，代价高。

```bash
# 清空 radix（前缀）缓存
curl -X POST "http://<host>:<port>/flush_cache"

# 带超时：等待最多 timeout 秒让运行中/等待中的请求结束再清
curl -X POST "http://<host>:<port>/flush_cache?timeout=30"
```

**注意（来自接口实现）**：

- `flush_cache` **仅在无运行中/等待中请求时才会执行**；否则不执行（返回失败）。因此请求全部结束（或给足 `timeout`）后再清；
- 若**改动了部署参数**（`--tp-size`、`--mem-fraction-static`、`--quantization` 等），则必须重启服务，不能只清缓存；
- 清缓存后可用 `sglang:cache_hit_rate` 与 `sglang:token_usage` 确认已归零，再开始下一轮压测，保证各轮基线一致。

---

## 9. 指标口径与公式汇总

| 名称 | 公式 | 说明 |
| --- | --- | --- |
| 算术强度 | $I=\dfrac{\text{FLOPs}}{\text{Bytes}}$ | 计算与访存之比 |
| Roofline 拐点 | $I_{ridge}=\dfrac{F_{peak}}{B_{peak}}$ | 区分 compute/memory bound |
| MFU | $\dfrac{F_{achieved}}{F_{peak}}$ | 算力利用率 |
| MBU | $\dfrac{B_{achieved}}{B_{peak}}$ | 带宽利用率 |
| prefill/decode FLOPs 近似 | $F_{achieved}\approx 2\,N_{active}\,T$ | 忽略 attention 二次项 |
| 平均 ITL | $\dfrac{\sum \Delta t_{token}}{N_{token}-1}$ | 可近似为 $1/\text{单步时延}$ |
| Little's Law | $L=\lambda W$ | 并发 = 到达率 × 驻留时间 |
| 有效吞吐 | $\dfrac{\text{有效 token 数}}{t}$ | 建议扣除缓存命中与截断 |

**单位与量纲提醒**：

- FLOPS（大写）是「每秒浮点运算次数」，FLOPs（小写 s）常指「浮点运算总次数」，混用是常见错误；
- 带宽 GB/s 与 Gib/s、GB 与 GiB 要区分（1 GiB = 1.074 GB）；
- 标称算力要分清 **dense vs sparse**、**bf16 vs fp8**，否则 MFU 会差 2 倍。

---

## 10. 瓶颈判定决策树

```
吞吐上不去 / 延迟超标
│
├─ 看 /metrics：num_queue_reqs 是否持续 > 0？
│   ├─ 是 → 服务确实忙不过来 → 继续往下找哪个资源饱和
│   └─ 否 → 可能是客户端/网络/单请求问题，而非服务算力
│
├─ GPU：utilization.gpu 是否持续 > 90%？
│   ├─ 否 → GPU 在等东西 → 查 CPU(§4)/网络(§6)/IO(§7)，看是否有部件 > 90%
│   └─ 是 → 看是算力还是带宽受限
│        ├─ Tensor Core 活跃度 / MFU 高 → compute-bound
│        │     → 提高 MFU：算子融合、减少冗余计算、优化 kernel
│        └─ DRAM_ACTIVE / MBU 高 → memory-bound（decode 常见）
│              → 增大 batch、量化、投机解码、减少权重读取
│
├─ KV 显存是否打满？token_usage ≈ 1
│   └─ 是 → 增大 mem-fraction、降并发、减少上下文长度、开前缀缓存
│
├─ 通信是否占比高？（TP/PP/EP 场景）
│   └─ 是 → 检查 NVLink/PCIe/RDMA 带宽，检查是否通信-计算重叠
│
└─ 是否降频？（功耗墙/温度墙）
    └─ 是 → 改善散热、限频、增加卡数分摊
```

---

## 11. 标准观测流程（以一次压测为例）

推荐「**先清缓存、再观测、后定位**」的闭环：

1. **清理基线**：`POST /flush_cache?timeout=<sec>`，确认无残留 KV（`token_usage`、`cache_hit_rate` 归零）。若改了部署参数则重启服务。
2. **开启采集**（另开终端，避免污染 CPU）：
   ```bash
   nvidia-smi dmon -s pucvmet -d 1 -c 600 > gpu.log &
   mpstat -P ALL 1 > cpu.log &
   sar -n DEV 1 > net.log &
   iostat -x 1 > io.log &
   while true; do curl -s localhost:<port>/metrics >> metrics.log; sleep 1; done &
   ```
3. **发起压测**：`python -m sglang.bench_serving ...`，记录吞吐与分位延迟。
4. **对齐时间轴**：把服务级指标（TTFT/ITL/队列）、硬件级指标（SM/DRAM/CPU/网络）按时间戳对齐看。
5. **定位瓶颈**：按 §10 决策树逐层排除。
6. **结束采集、汇总**：`kill` 后台采集进程，绘制曲线与结论。
7. **清理**：`POST /flush_cache`，回到干净状态，准备下一轮（参数未变则不重启）。

**观测本身的开销**：`nvidia-smi dmon`、`mpstat`、`sar` 开销极低；Nsight / `ncu` 开销大，**只用于短时定位，不要贯穿全程压测**，否则会污染性能数据。

---

## 12. 常见误区

| 误区 | 事实 |
| --- | --- |
| `utilization.gpu=100%` 就是算力打满 | 它只是「有 kernel 在跑的时间占比」，要看 Tensor Core 活跃度 / MFU |
| 只看 GPU 利用率就下结论 | CPU、网络、IO 同样可能成为瓶颈，必须全栈一起看 |
| 只看平均延迟 | 长尾（P95/P99）才是用户体感与 SLA |
| 吞吐高就好 | 吞吐与延迟要成对看，吞吐到顶后延迟会陡增 |
| 显存 `used` 就是模型大小 | 还含 KV 缓存、CUDA Graph、通信缓冲、碎片 |
| NVMe 上 `%util=100%` 就是饱和 | NVMe 有并行队列，应看 `await` 和 `aqu-sz` |
| MFU 低就是代码差 | decode 天然 memory-bound，MFU 低是正常的，应看 MBU |
| 每轮压测都重启服务 | 参数未变时只需 `flush_cache`，重启代价高且引入冷启动噪声 |
| 标称算力直接当 $F_{peak}$ | 要区分 dense/sparse、bf16/fp8，口径不统一会导致 MFU 差 2 倍 |

---

## 附录 A：命令速查表

```bash
# ---------- GPU ----------
nvidia-smi                                     # 快照
nvidia-smi dmon -s pucvmet -d 1                # 连续监控
nvidia-smi pmon -s um -d 1                     # 进程级
nvidia-smi nvlink -s / -gt d                   # NVLink 状态 / 吞吐
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv -l 1

# ---------- CPU ----------
mpstat -P ALL 1                                # 每核
vmstat 1                                       # 总体 + 内存 + 换页
pidstat -u -t -p <pid> 1                       # 进程/线程
top -H -p <pid>                                # 线程视图
taskset -cp <pid>                              # 亲和性

# ---------- 内存 / NUMA ----------
free -h
numastat -p <pid>
cat /proc/meminfo | grep -i huge

# ---------- 网络 ----------
sar -n DEV 1
sar -n EDEV 1
ethtool -S <iface>
ibstat

# ---------- 存储 ----------
iostat -x 1
iotop -o
pidstat -d 1

# ---------- 服务级 ----------
# 打开指标：launch_server 加 --enable-metrics [--enable-mfu-metrics]
curl -s http://<host>:<port>/metrics
curl -X POST http://<host>:<port>/start_profile
curl -X POST http://<host>:<port>/stop_profile
curl -X POST "http://<host>:<port>/flush_cache?timeout=30"

# ---------- 压测 ----------
python -m sglang.bench_serving \
  --backend sglang --num-prompts 1000 \
  --random-input-len 1024 --random-output-len 256 \
  --request-rate inf
```

---

## 附录 B：代码索引（当前版本）

| 内容 | 位置 |
| --- | --- |
| Prometheus 指标定义 | `python/sglang/srt/observability/metrics_collector.py` |
| 进程 CPU 指标 | `python/sglang/srt/observability/cpu_monitor.py` |
| `/metrics` 端点挂载 | `python/sglang/srt/utils/common.py`（`add_prometheus_middleware`） |
| `/flush_cache` 接口 | `python/sglang/srt/entrypoints/http_server.py` |
| `/start_profile`、`/stop_profile` | `python/sglang/srt/entrypoints/http_server.py` |
| 指标相关启动参数 | `python/sglang/srt/server_args.py`（`--enable-metrics`、`--enable-mfu-metrics` 等） |
| 压测脚本与参数 | `benchmark/`（`bench_serving.py` 等） |
