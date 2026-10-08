# CPU 亲和性与 NUMA

> 本文回答三个问题：**CPU 亲和性是什么**、**NUMA 是什么**、**两者是什么关系**，并给出 SGLang 中的落地实现与配置手段。
>
> 与 `scheduler_subprocess_startup_zh.md` 的关系：那篇讲 `configure_scheduler_process` 的启动链路，其中第 ② 步的「亲和性 / NUMA」只在表格里一笔带过；本文把这两项单独展开。
>
> 代码块路径均相对仓库根目录（`sglang/`），行号对应当前版本。

---

## 0. 一页速览

| | NUMA | CPU 亲和性（CPU affinity） |
| --- | --- | --- |
| 是什么 | **硬件拓扑事实**：内存按 socket 分块，本地快、远端慢 | **内核调度机制**：限制进程/线程只能跑在哪些核上 |
| 谁决定 | 主板/CPU 厂商，客观存在，不可改 | 可配置的约束（策略） |
| 作用对象 | 内存页的落脚 node、PCIe 设备的归属 | 线程的调度范围（CPU 集合） |
| 接口 | `numactl`、`set_mempolicy(2)`、`mbind(2)` | `sched_setaffinity(2)`、`taskset`、`psutil.cpu_affinity` |
| 目的 | 让访存落在本地 node，避免跨 socket 带宽/延迟损失 | 防核间迁移、防多 rank 抢核，稳定 CPU 侧算力 |
| 在 SGLang 中 | `numa_utils` / `--numa-node` / `SGLANG_NUMA_BIND_V2` | `set_gpu_proc_affinity` / `SGLANG_SET_CPU_AFFINITY` |

三句话总结：

1. **CPU 亲和性**回答「用哪些核」，**NUMA** 回答「内存和核应该落在哪个 socket」。
2. 两者**正交**：只做亲和性也是有效的性能调优（与 NUMA 无关）；只做内存策略则线程仍可能被调度到别的 socket。
3. **亲和性是 NUMA 优化的必要不充分条件**——把线程钉到目标 node 的核上只是前提，还要配内存策略才能保证页落在本地。

---

## 1. NUMA 是什么

**NUMA = Non-Uniform Memory Access（非统一内存访问）**。多路（多 socket）服务器上，内存不是「一块共享的」，而是**按 CPU 插槽物理切分**：每个 CPU 有自己的内存控制器与本地 DRAM，访问走哪条路径、代价不同，这就是「非统一」。

```
                 UPI / QPI（socket 间互连总线，跨 node 访问必经此路）
                            │
     ┌───────────────┐      │      ┌───────────────┐
     │   Socket 0    │◄─────┴─────►│   Socket 1    │
     │   CPU + MC    │             │   CPU + MC    │
     │      │        │             │      │        │
     │    ▼ 本地直连 │             │    ▼ 本地直连 │
     │   本地 DRAM   │             │   本地 DRAM   │
     │   GPU0/GPU1   │             │   GPU2/GPU3   │
     └───────────────┘             └───────────────┘
       = NUMA node 0                = NUMA node 1

  图例：
    MC          = Memory Controller，内存控制器（集成在 socket 内，直连本地 DRAM）
    UPI / QPI   = Intel Ultra Path Interconnect / QuickPath Interconnect，socket 间互连总线
    node N      = NUMA 节点编号，由「socket + 本地 DRAM + 挂在该 socket PCIe root 下的设备」构成
    本地直连     = CPU → 自己的 MC → 本地 DRAM，即本地访问（快）
    跨 node 访问 = 穿过 UPI/QPI 读另一 node 的 DRAM，即远端访问（延迟、带宽都差）
```

### 1.1 访存代价不对称

- 一个 NUMA node ≈ 一个 socket + 它直连的 DRAM（以及挂在该 PCIe root 下的设备）
- 访问本 node 内存：走本地内存控制器，快
- 访问别的 node 内存：要过 UPI/QPI 互连，延迟和带宽都差

$$\text{远端访问延迟} \approx 1.3\text{–}2\times \text{本地访问延迟}$$

### 1.2 页的落脚点由「首次触碰」决定

默认策略下，一页内存由**哪个 node 的 CPU 第一次写**，就分配到该 node（first-touch）。因此「线程跑在哪个 node」会隐式决定「内存落在哪个 node」——线程跑错 node 会引发大量远端访问。

### 1.3 设备也有归属

GPU 物理挂在某个 socket 的 PCIe root 下。若 host 侧线程与该 GPU 跨 socket：

- host 内存 ↔ GPU 的拷贝要走远端内存 + 交叉互连，带宽下降
- 这正是 SGLang 要「按 GPU 查 NUMA node 再绑进程」的原因

### 1.4 与 UMA 的对比

单路机器所有 CPU 访存等价（Uniform），不存在这些问题；NUMA 优化只在多路机器上有意义。

### 1.5 观测手段

```bash
numactl --hardware            # 有哪些 node、每个 node 的 CPU 与内存
lscpu | grep -i numa
ls /sys/devices/system/node/  # node0 node1 ...
cat /sys/devices/system/node/node0/meminfo
```

### 1.6 内存策略三档

| 策略 | 语义 | 强度 |
| --- | --- | --- |
| `MPOL_BIND`（`--membind=N`） | 只能从指定 node 分配 | 硬绑，可能因 cpuset/权限失败 |
| `MPOL_PREFERRED`（`--preferred=N`） | 优先本地，失败可退化 | 软偏好 |
| `MPOL_INTERLEAVE` | 各 node 轮转打散 | 提带宽、不降延迟 |

---

## 2. CPU 亲和性是什么

把进程/线程**钉死在一组固定 CPU 核**上，内核调度器不再把它迁移到别的核。

### 2.1 收益

- 避免跨核迁移导致的 L1/L2 缓存反复失效
- 避免多个进程（如各 TP rank 的 scheduler）互相抢核
- 让 CPU 侧工作（请求调度、tokenize、采样、KV 索引管理、显存拷贝下发）得到**稳定算力**，降低 TTFT / ITL 抖动

### 2.2 独立于 NUMA 的价值

即使机器只有一个 node（UMA），亲和性依然有效——此时它的目的纯粹是「隔离与稳定」，与 NUMA 无关。反过来，亲和性**不保证**内存本地化：它只管调度范围，不管页分配到哪。

---

## 3. 两者的关系

### 3.1 正交的两条轴

| | 管什么 | 解决的痛点 |
| --- | --- | --- |
| CPU 亲和性 | 用**哪些核** | 抢核、核间迁移导致的性能抖动 |
| NUMA | 核 + 内存落在**哪个 socket** | 跨 socket 访存的带宽与延迟损失 |

可以只做其一，也可以组合；生产上通常是「按 GPU 所属 node 选定核集合，再把进程绑上去 + 内存偏好本地 node」。

### 3.2 完整 NUMA 绑定 = 亲和性 + 内存策略

$$\text{NUMA 绑定} = \underbrace{\text{CPU 亲和性}}_{\text{限定核集合}} \;+\; \underbrace{\text{内存策略（membind/preferred）}}_{\text{限定内存落脚 node}}$$

- 只绑核 → 内存仍可能大量落在远端 node
- 只设内存策略 → 线程仍可能被调度到别的 socket 上跑

所以：**亲和性是 NUMA 优化的实现手段与前提条件，而不是 NUMA 本身**。

### 3.3 实现层面：亲和性常常是 NUMA 绑定的「前置输入」

典型写法就是「取当前允许的 CPU 集合 ∩ 目标 node 的核集合 → 设为新亲和性」，再叠加内存策略。SGLang 的 `numa_bind_to_node` 正是这个模式：

```169:179:python/sglang/srt/utils/numa_utils.py
    node_cpus = _node_cpus(node)
    if node_cpus:
        allowed_cpus = os.sched_getaffinity(0)
        target_cpus = node_cpus & allowed_cpus
        if not target_cpus:
            _handle_numa_bind_failure(node, allowed_cpus)
            return
        os.sched_setaffinity(0, target_cpus)
    else:
        libnuma.numa_run_on_node(ctypes.c_int(node))
    libnuma.numa_set_preferred(ctypes.c_int(node))
```

注意这里的**顺序敏感性**：如果先做了一次纯 CPU 亲和性绑定（把进程收窄到某段核），再做上面的交集，交集可能变空 → NUMA 绑定被跳过并告警。见 §4.5。

---

## 4. SGLang 中的落地实现

### 4.1 调用位置：`configure_scheduler_process`

Scheduler 子进程入口 `run_scheduler_process` 在做完日志/进程名配置后，集中做亲和性与 NUMA 配置：

```5777:5788:python/sglang/srt/managers/scheduler.py
    # Set cpu affinity to this gpu process
    if envs.SGLANG_SET_CPU_AFFINITY.get():
        set_gpu_proc_affinity(
            get_parallel().pp_size,
            get_parallel().tp_size,
            get_parallel().nnodes,
            gpu_id,
        )
    if not envs.SGLANG_NUMA_BIND_V2.get():
        numa_node = get_numa_node_if_available(server_args, gpu_id)
        if numa_node is not None:
            numa_bind_to_node(numa_node)
```

要点：

- 亲和性由 `SGLANG_SET_CPU_AFFINITY` 控制，**默认关闭**
- NUMA 的「进程内绑定」只在 `SGLANG_NUMA_BIND_V2` **关闭**时执行；V2（默认开启）改在子进程启动前用 `numactl` 包一层，见 §4.4

### 4.2 CPU 亲和性：`set_gpu_proc_affinity`

**代码归属**：`python/sglang/srt/utils/common.py` · 模块级函数 `set_gpu_proc_affinity`（无类）

思路：把本节点物理核按「每节点 TP 副本数」均分，给每个 `gpu_id` 一段**连续**物理核；超线程开启时同时绑定对应的 HT 兄弟核。

```2914:2934:python/sglang/srt/utils/common.py
    # total physical cores
    total_pcores = psutil.cpu_count(logical=False)
    # physical cores per TP (N.B. more Cores than GPUs on node)
    num_cores_bind = total_pcores // tp_size_per_node

    # able to handle multiple DP per node
    start_cpu_id = (gpu_id * num_cores_bind) % total_pcores
    end_cpu_id = start_cpu_id + num_cores_bind

    if psutil.cpu_count() != psutil.cpu_count(logical=False):
        # HT on
        lower_cpu_ids = [id for id in range(start_cpu_id, end_cpu_id)]
        upper_cpu_ids = [id + total_pcores for id in range(start_cpu_id, end_cpu_id)]
        bind_cpu_ids = list(itertools.chain(lower_cpu_ids, upper_cpu_ids))
    else:
        # HT off
        bind_cpu_ids = [id for id in range(start_cpu_id, end_cpu_id)]

    # set cpu_affinity to current process
    p.cpu_affinity(bind_cpu_ids)
```

其中：

$$\text{tp\_size\_per\_node} = \frac{\text{tp\_size}}{\max(\lfloor \text{nnodes}/\text{pp\_size} \rfloor,\,1)}$$

$$\text{num\_cores\_bind} = \left\lfloor \frac{\text{total\_pcores}}{\text{tp\_size\_per\_node}} \right\rfloor, \qquad \text{start} = (gpu\_id \times \text{num\_cores\_bind}) \bmod \text{total\_pcores}$$

注意这段逻辑**只看核数量和 `gpu_id`，完全不感知 NUMA 拓扑**——这正是它和 §4.3/§4.4 的分工差别。

### 4.3 NUMA 节点来源：`get_numa_node_if_available`

**代码归属**：`python/sglang/srt/utils/numa_utils.py` · 模块级函数 `get_numa_node_if_available`

优先级：**命令行 `--numa-node` > 自动探测（NVML 查 GPU 的 memory affinity）**

```130:145:python/sglang/srt/utils/numa_utils.py
    if server_args.numa_node is not None:
        return server_args.numa_node[gpu_id]
    if not envs.SGLANG_AUTO_NUMA_BIND.get():
        return None
    if _is_numa_available():
        queried_numa_node = _query_numa_node_for_gpu(gpu_id)
        if len(queried_numa_node) == 0:
            return None
        if len(queried_numa_node) > 1:
            # get_numa_node_for_gpu could return multiple nodes, we use the first one for now.
            # I don't think there any hardware configs that would have more than one.
            logger.warning(
                f"Multiple NUMA nodes found for GPU {gpu_id}: {queried_numa_node}. Using the first one."
            )
        return queried_numa_node[0]
    return None
```

- `--numa-node` 是列表，**第 i 个元素对应第 i 个子进程**（即按 `gpu_id` 索引）：

```1594:1598:python/sglang/srt/server_args.py
    numa_node: A[
        Optional[List[int]],
        "Sets the numa node for the subprocesses. i-th element corresponds to i-th subprocess. If unset, will be automatically detected on NUMA systems.",
        NS("device"),
    ] = None
```

- 自动探测的可用性前置条件（`_is_numa_available`）：是 CUDA 环境、存在 `/sys/devices/system/node/node1`、有 `numactl`（V2 时必需）、并且**有权限调用 `set_mempolicy`**（容器里通常需要 `--cap-add SYS_NICE`）；否则告警并跳过。

### 4.4 两条 NUMA 绑定路径

**V1（`SGLANG_NUMA_BIND_V2=false`）——进程内绑定**

在 Python 进程内直接调 libnuma / `sched_setaffinity`：`numa_bind_to_node(node)`（见 §3.3 代码），同时设「CPU 绑定」与「内存偏好本地 node」。

**V2（`SGLANG_NUMA_BIND_V2=true`，默认）——启动前用 `numactl` 包一层**

在父进程 `proc.start()` 处，把子进程要 exec 的解释器替换成 `numactl ... python` 脚本，从而在 Python 启动前就完成绑定：

```914:918:python/sglang/srt/entrypoints/engine.py
                        with (
                            memory_saver_adapter.configure_subprocess(),
                            numa_utils.configure_subprocess(server_args, gpu_id),
                        ):
                            proc.start()
```

`configure_subprocess` 内部构造的参数形态：

- node 的核全部落在当前 cpuset 内 → `--cpunodebind=N --membind=N`
- 只有部分核可用 → `--physcpubind=<cpu_list> --membind=N`

**降级链（避免 seccomp/cpuset 拒绝导致子进程静默崩溃）**：先用 `numactl <args> true` 试跑探测，逐级放宽内存策略而保留 CPU 绑定：

$$\texttt{--membind=N} \;\rightarrow\; \texttt{--preferred=N} \;\rightarrow\; \text{丢掉内存段（仅 CPU 绑定）} \;\rightarrow\; \text{整体放弃}$$

全部失败时按 `SGLANG_CRASH_ON_NUMA_BIND_FAILURE` 决定「告警继续」还是「直接抛错」。

### 4.5 顺序与冲突：三个易踩的坑

1. **亲和性先于 NUMA（V1 路径）**：`set_gpu_proc_affinity` 先把进程收窄到某段核，随后 `numa_bind_to_node` 再与该 node 的核求交集，交集可能为空 → 告警跳过 NUMA 绑定。因此默认走 V2（`numactl` 一次性同时指定 CPU 与内存，不存在二次收窄问题），V1 路径下建议二者不要同时开。
2. **容器权限**：无 `SYS_NICE`/属主权限时 `set_mempolicy` 被拒，NUMA 绑定静默跳过（只 warn），容易误判为「配了但没生效」。
3. **`CUDA_VISIBLE_DEVICES` 重排**：`gpu_id` 是 CUDA 逻辑序号，查 NVML 前需转成物理序号（`_get_nvml_device_index`），否则可能绑到「另一个物理 GPU 所属的 node」——日志里会打印 `logical_gpu_id` / `physical_gpu_id` / `CUDA_VISIBLE_DEVICES` 便于核对。

---

## 5. 开关与配置速查

| 配置 | 类型 | 默认 | 作用 |
| --- | --- | --- | --- |
| `SGLANG_SET_CPU_AFFINITY` | env | `False` | 是否按 `gpu_id` 绑定 CPU 核（`set_gpu_proc_affinity`） |
| `SGLANG_AUTO_NUMA_BIND` | env | `True` | 未指定 `--numa-node` 时，是否用 NVML 自动探测 GPU 所属 node |
| `SGLANG_NUMA_BIND_V2` | env | `True` | 用 `numactl` 在启动前绑定（含降级链）；关闭则退回进程内 libnuma 绑定 |
| `SGLANG_CRASH_ON_NUMA_BIND_FAILURE` | env | `False` | 绑定失败时告警继续 / 直接抛错 |
| `--numa-node` | 启动参数 | `None` | 手工指定每个子进程的 node，第 i 个对应第 i 个 `gpu_id` |

环境变量定义位置：

```1339:1342:python/sglang/srt/environ.py
SGLANG_SET_CPU_AFFINITY = EnvBool(False)
SGLANG_NUMA_BIND_V2 = EnvBool(True)
SGLANG_AUTO_NUMA_BIND = EnvBool(True)
SGLANG_CRASH_ON_NUMA_BIND_FAILURE = EnvBool(False)
```

典型用法：

```bash
# 1) 校验机器拓扑与 GPU 归属
numactl --hardware
nvidia-smi topo -m

# 2) 自动 NUMA 绑定（默认已开），容器需放开权限
docker run --cap-add SYS_NICE ...

# 3) 手工指定：4 张卡的 node 归属
python -m sglang.launch_server --model-path <path> --numa-node 0 0 1 1

# 4) 叠加 CPU 亲和性（隔离抢核）
SGLANG_SET_CPU_AFFINITY=1 python -m sglang.launch_server --model-path <path>
```

---

## 6. 常见故障与排查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 日志 `numa not available on this system, skip bind action` | 单路机器 / 无 node1 / libnuma 缺失 | 正常跳过；单路机器无需 NUMA |
| 日志 `User lacks permission to set NUMA affinity` | 容器未给 `SYS_NICE`，`set_mempolicy` 被拒 | 加 `--cap-add SYS_NICE` |
| 日志 `NUMA node N has no CPU cores allowed by the current affinity [...]` | 与前置 CPU 亲和性交集为空 | 关掉 `SGLANG_SET_CPU_AFFINITY`，或改进用 V2 |
| 启动即无回溯崩溃（V2） | `numactl` 拒绝了 `--membind`（seccomp/cpuset） | 已内置降级链；仍失败看 `SGLANG_CRASH_ON_NUMA_BIND_FAILURE` 与 probe stderr |
| 绑定「看着配了」但性能无变化 | 绑错 node（`CUDA_VISIBLE_DEVICES` 重排） | 校对日志中的 `logical_gpu_id` / `physical_gpu_id` |

---

## 7. 小结

- **NUMA 是硬件事实**：本地内存快、跨 socket 慢，页落脚点由 first-touch 决定，设备也有 socket 归属。
- **CPU 亲和性是内核机制**：决定线程能跑在哪些核上；它本身与 NUMA 无关，但它是让进程「待在正确 socket」的载体。
- **完整 NUMA 优化 = 亲和性（限核）+ 内存策略（限内存 node）**；亲和性是必要不充分条件。
- **SGLang 的分工**：`set_gpu_proc_affinity` 纯按核数/`gpu_id` 做亲和性（不感知拓扑）；`numa_utils` 负责查 GPU 所属 node 并绑定（V1 进程内 libnuma，V2 启动前 `numactl`，带三级降级）。
- **实际运维**：容器加 `SYS_NICE`；核对 `logical/physical_gpu_id`；避免 V1 路径下同时开亲和性与 NUMA 绑定。
