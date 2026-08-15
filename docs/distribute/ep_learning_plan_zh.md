# SGLang EP 专家并行专项学习与实践计划

> 本计划是 `parallelism_learning_plan_zh.md` 阶段 P4（第 6–7 天）的展开版，按每天 4 小时、共 8 小时设计。目标不是逐行读完所有 MoE 代码，而是建立一条可验证的主线：
>
> **rank 拓扑 → Router/Top-K → Dispatch → 本地专家布局与计算 → Combine → EPLB → 部署与性能分析**。

## 一、完成后的能力目标

完成本计划后，应能独立回答并验证以下问题：

1. EP、MoE TP、MoE DP、Attention TP 分别切什么，如何复用同一批 rank；
2. `tp_size`、`ep_size`、`moe_dp_size`、`moe_tp_size` 的关系；
3. 一个 token 从 Router 选出 Top-K 专家后，如何跨 rank dispatch、执行专家 MLP，再 combine 回来源 rank；
4. `TopKOutput`、`DispatchOutput`、`CombineInput`、`src2dst`、`seg_indptr`、`masked_m` 各自解决什么问题；
5. DeepEP normal 与 low-latency 模式的数据布局和适用场景；
6. 为什么 EP 会动态负载倾斜，以及 EPLB 如何通过统计、复制和重排专家缓解；
7. 如何设计 EP 实验，区分计算瓶颈、通信瓶颈和专家负载不均；
8. 如何根据模型、硬件和工作负载选择 `--moe-a2a-backend` 与 `--moe-runner-backend`。

最终产出物：

- 一张 EP rank 拓扑图；
- 一份 Top-K 路由与 dispatch/combine 手算表；
- 一张 SGLang EP 代码调用链图；
- 一份 EP 实验记录表；
- 一份 EPLB 指标与专家热点分析记录。

---

## 二、先建立统一心智模型

### 2.1 一层 MoE 的数学语义

设输入 token 数为 $T$，hidden size 为 $H$，逻辑专家数为 $E$，每个 token 选择 $K$ 个专家。Router 输出：

$$
\text{topk\_ids}\in\mathbb{N}^{T\times K},\qquad
\text{topk\_weights}\in\mathbb{R}^{T\times K}.
$$

第 $t$ 个 token 的输出可抽象为：

$$
y_t=s\sum_{j=1}^{K}w_{t,j}E_{e_{t,j}}(x_t),
$$

其中 $e_{t,j}$ 是命中的专家，$w_{t,j}$ 是路由权重，$s$ 是可选的 `routed_scaling_factor`。

### 2.2 EP 的三段式数据流

```text
hidden_states + TopKOutput
          │
          ▼
Dispatcher.dispatch
按 expert 所属 rank 发送 token 和路由元数据
          │
          ▼
DispatchOutput
          │
          ▼
本地布局转换 + grouped GEMM 专家计算 + 反重排
          │
          ▼
CombineInput
          │
          ▼
Dispatcher.combine
把专家输出送回 token 来源 rank，并完成/衔接 Top-K 加权
          │
          ▼
原 token 顺序的 hidden_states
```

### 2.3 rank 分解公式

在一个外层 TP 域内，SGLang 的 MoE rank mesh 满足：

$$
moe\_tp\_size=
\frac{tp\_size}{ep\_size\times moe\_dp\_size}.
$$

因此必须满足：

$$
ep\_size\times moe\_dp\_size\mid tp\_size.
$$

示例：


| `tp_size` | `ep_size` | `moe_dp_size` | `moe_tp_size` | 含义                                               |
| --------: | --------: | ------------: | ------------: | -------------------------------------------------- |
|         8 |         8 |             1 |             1 | 纯 EP：8 个 rank 分专家，不切单个专家              |
|         8 |         4 |             1 |             2 | $EP=4$，每个 local expert 再做 2 路 TP             |
|         8 |         2 |             2 |             2 | $MoE\ DP=2$，每个 DP 副本内 $EP=2$、$expert\ TP=2$ |
|         8 |         1 |             1 |             8 | 不做 EP，所有专家按 8 路 MoE TP 切分               |

专用 A2A 后端（如 DeepEP、Mooncake、NIXL、Mori）通常要求 $ep\_size=tp\_size$，此时 $moe\_tp\_size=1$。混合 EP + TP（$ep\_size<tp\_size$）通常走 `moe_a2a_backend=none` 的标准路径，实际约束以 `ServerArgs` 为准。

---

## 三、两天总安排


| 时间               | 学习主题                   | 代码主入口                                             | 实践产出                |
| ------------------ | -------------------------- | ------------------------------------------------------ | ----------------------- |
| 第 1 天 0:00–0:45 | MoE/EP 原理与路由手算      | `docs/theory/distributed/EP.md`                        | 路由手算表              |
| 第 1 天 0:45–1:30 | EP/TP/DP rank 拓扑         | `distributed/parallel_state.py`                        | 8 卡 rank 图            |
| 第 1 天 1:30–2:15 | Router 与 Top-K            | `moe/router.py`、`moe/topk.py`                         | TopKOutput shape 表     |
| 第 1 天 2:15–3:15 | Dispatcher 抽象与后端      | `token_dispatcher/base.py`、`standard.py`、`deepep.py` | dispatch/combine 协议图 |
| 第 1 天 3:15–4:00 | EP MoE 总调用链            | `fused_moe_triton/layer.py`、`ep_moe/layer.py`         | 一层 MoE 时序图         |
| 第 2 天 0:00–1:00 | 本地专家布局与 kernel      | `ep_moe/kernels.py`                                    | `src2dst` 手算          |
| 第 2 天 1:00–1:45 | MoE runner 与 grouped GEMM | `moe_runner/`                                          | normal/LL 布局对比      |
| 第 2 天 1:45–2:45 | EPLB 统计与重排            | `eplb/`                                                | 热点专家重排方案        |
| 第 2 天 2:45–3:45 | 部署与性能实验             | `server_args.py`、部署文档                             | benchmark 对比表        |
| 第 2 天 3:45–4:00 | 总结与通关检查             | 全链路                                                 | 口述 + 图示验收         |

---

## 四、第 1 天：从路由语义到完整调用链

## 步骤 1：手算 Router、Dispatch、Combine（45 分钟）

### 阅读

1. `docs/theory/distributed/EP.md` §1–§5；
2. 重点理解 `topk_ids`、`topk_weights`、logical expert、physical expert、local expert。

### 手算实践

设：

- 2 个 rank；
- 4 个专家：rank 0 持有 E0/E1，rank 1 持有 E2/E3；
- 每个 rank 初始有 2 个 token：rank 0 仅持有 A、B，rank 1 仅持有 C、D；这里模拟的是 token 已按来源 rank 分片的场景，而不是纯 Attention TP 下各 rank 都复制完整 token 的场景；
- $Top\text{-}K=2$。

自行填写：


| 来源 rank | token | `topk_ids` | `topk_weights` | dispatch 目标 rank |
| --------- | ----- | ---------- | -------------- | ------------------ |
| 0         | A     | [E0, E2]   | [0.7, 0.3]     | 0、1               |
| 0         | B     | [E3, E1]   | [0.6, 0.4]     | 1、0               |
| 1         | C     | [E2, E3]   | [0.8, 0.2]     | 1                  |
| 1         | D     | [E1, E0]   | [0.55, 0.45]   | 0                  |

继续推导：

1. 每个 rank dispatch 后收到哪些 token-expert assignment：
   - rank 0：`(A, E0)`、`(B, E1)`、`(D, E1)`、`(D, E0)`；
   - rank 1：`(A, E2)`、`(B, E3)`、`(C, E2)`、`(C, E3)`。
2. 每个 local expert 的 assignment 数：
   - E0：2（A、D）；E1：2（B、D）；E2：2（A、C）；E3：2（B、C）。
3. 必须跨卡的 assignment 与 hidden payload：
   - 跨卡 assignments：`(A, E2)`、`(B, E3)`、`(D, E1)`、`(D, E0)`，共 4 个；
   - A、B 各向远端 rank 发送一份 hidden；D 命中的 E0、E1 都在 rank 0，同一目标 rank 可复用 D 的 hidden，因此优化后的跨卡 hidden payload 共 3 份：A、B、D；若后端按 assignment 独立发送，则可能是 4 份。
4. 专家输出如何回到来源 rank：
   - rank 1 将 E2(A)、E3(B) 的输出送回 rank 0；
   - rank 0 将 E1(D)、E0(D) 的输出送回 rank 1；
   - C 的两个专家都在其来源 rank 1，无需跨卡回传；后端可在目标 rank 先完成同一 token 的局部加权，再减少 combine payload。
5. 每个 token 按 `topk_weights` 合并：

$$
y_A=0.7E_0(x_A)+0.3E_2(x_A),
\qquad
y_B=0.6E_3(x_B)+0.4E_1(x_B),
$$

$$
y_C=0.8E_2(x_C)+0.2E_3(x_C),
\qquad
y_D=0.55E_1(x_D)+0.45E_0(x_D).
$$

### 验收标准

- 能解释为什么 $Top\text{-}K=2$ 不必然产生两份远端 hidden payload：两个专家可能位于同一目标 rank；
- 能区分“token 数”和“token-expert assignment 数”；
- 能解释 dispatch 与 combine 为什么方向相反。

---

## 步骤 2：画出 SGLang 的 MoE rank mesh（45 分钟）

### 阅读

阅读 `python/sglang/srt/distributed/parallel_state.py`：

- `initialize_model_parallel`；
- `moe_ep_size`、`moe_dp_size`、`moe_tp_size` 的计算；
- `_MOE_EP`、`_MOE_DP`、`_MOE_TP` 的建组循环；
- `get_moe_ep_group()`、`get_moe_tp_group()`。

### 纸笔实践 A：纯 EP

配置：

$$
tp\_size=8,\quad ep\_size=8,\quad moe\_dp\_size=1.
$$

推导：

$$
moe\_tp\_size=8/(8\times1)=1.
$$

画出 `_MOE_EP` 的 rank 集合：

$$
\_MOE\_EP=[0,1,2,3,4,5,6,7],
$$

并说明为什么 `_MOE_EP` 可以直接复用 `_TP`。

### 纸笔实践 B：EP + expert TP

配置：

$$
tp\_size=8,\quad ep\_size=4,\quad moe\_dp\_size=1.
$$

推导 $moe\_tp\_size=2$，再根据建组循环写出：

- MoE EP groups；
- MoE TP groups；
- 每个 rank 的 `(ep_rank, moe_tp_rank)` 坐标。

### 纸笔实践 C：MoE DP + EP + TP

配置：

$$
tp\_size=8,\quad ep\_size=2,\quad moe\_dp\_size=2.
$$

推导 $moe\_tp\_size=2$，画出三维坐标：

```text
(moe_dp_rank, ep_rank, moe_tp_rank)
```

### 验收标准

- 任给合法的三个 size，能在 5 分钟内写出 `moe_tp_size`；
- 能解释为什么 EP 和 TP 可在同一 TP 域内重组，而不一定相乘占用额外 GPU；
- 能指出专用 A2A 后端为何常把 `ep_size` 固定为 `tp_size`。

---

## 步骤 3：跟踪 Router 与 Top-K（45 分钟）

### 阅读顺序

1. `python/sglang/srt/layers/moe/topk.py`
   - `TopKOutput`；
   - `TopK`；
   - grouped top-k、biased top-k 等分支只需了解入口；
2. `python/sglang/srt/layers/moe/router.py`
   - `FusedMoeRouter`；
   - router linear 如何连接 Top-K；
3. `python/sglang/srt/eplb/expert_location_dispatch.py`
   - `topk_ids_logical_to_physical`；
   - static/dynamic 映射。

### shape 追踪实践

假设：

$$
T=6,\qquad H=7168,\qquad E=256,\qquad K=8.
$$

记录：


| 数据            | 典型 shape  | 语义                             |
| --------------- | ----------- | -------------------------------- |
| `hidden_states` | `[6, 7168]` | 本 rank 当前 token               |
| `router_logits` | `[6, 256]`  | 每 token 对逻辑专家的分数        |
| `topk_ids`      | `[6, 8]`    | 每 token 选中的专家              |
| `topk_weights`  | `[6, 8]`    | 对应加权系数                     |
| 展平 assignment | `[48]`      | $6\times8=48$ 个 token-expert 对 |

### EPLB 映射小实验（纸笔）

设逻辑专家 E1 被复制为 physical expert P1/P5：

1. Router 语义仍输出 logical E1；
2. dispatch 映射决定本次发往 P1 还是 P5；
3. 说明为什么不能直接把 logical ID 当成固定 physical ID。

### 验收标准

- 能解释 Router 与 Dispatcher 的职责边界；
- 能解释 $topk\_ids=-1$ 一类哨兵值为何必须在布局转换中正确处理；
- 能说明 EPLB 不改变模型的逻辑专家语义，只改变 physical placement。

---

## 步骤 4：理解 Dispatcher 协议与后端（60 分钟）

### 阅读顺序

1. `python/sglang/srt/layers/moe/token_dispatcher/README_zh.md`；
2. `token_dispatcher/base.py`：
   - `DispatchOutputFormat`；
   - `DispatchOutput`；
   - `CombineInputFormat`；
   - `CombineInput`；
   - `BaseDispatcher.dispatch/combine`；
   - pre/post hooks；
3. `token_dispatcher/standard.py`：标准路径；
4. `token_dispatcher/deepep.py`：只抓 normal 与 low-latency 的输入输出结构；
5. 可选：`mooncake.py`、`nixl.py` 的分阶段 `dispatch_a/b`、`combine_a/b`。

### 对比表实践


| 维度               | Standard               | DeepEP normal         | DeepEP low-latency         |
| ------------------ | ---------------------- | --------------------- | -------------------------- |
| 典型场景           | 无专用 A2A、混合 EP+TP | prefill/大 token 吞吐 | decode/小 batch 低延迟     |
| dispatch 后布局    | 标准 token/Top-K       | received-token 主序   | masked grouped GEMM 布局   |
| 是否需额外 scatter | 取决于 runner          | 通常需要              | 通常不需要普通`ep_scatter` |
| CUDA Graph         | 依后端                 | 通常关注吞吐          | 专门面向低延迟与图兼容     |

### 协议不变量实践

为 dispatch/combine 各写 3 条不变量，例如：

- dispatch 后有效 assignment 总数等于各 local expert token 数之和；
- combine 输出首维应恢复为来源 rank 的原 token 数；
- Top-K 权重只能被应用一次，不能在本地 gather 与 combine 中重复加权。

### 验收标准

- 能解释为什么 SGLang 要用协议类型，而不是让 MoE runner 直接依赖 DeepEP；
- 能说清通信后端和计算后端是两条独立选择轴；
- 能指出 hooks 如何支持 Single-Batch Overlap，而不改核心 MoE 层。

---

## 步骤 5：串起一层 MoE 的代码调用链（45 分钟）

### 阅读顺序

1. `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
   - `FusedMoE.forward`；
   - `FusedMoE.forward_impl`；
   - `FusedMoE.run_moe_core`；
2. `python/sglang/srt/layers/moe/ep_moe/layer.py`
   - `get_moe_impl_class`；
   - `DeepEPMoE.forward_impl`；
   - `dispatch`、`run_moe_core`、`combine`；
3. `python/sglang/srt/layers/moe/README_zh.md` 与 `ep_moe/README_zh.md` 对照复习。

### 必画时序图

```text
模型 MoE block
  │
  ├─ Router / TopK → TopKOutput
  │
  └─ FusedMoE 或 DeepEPMoE
       ├─ dispatcher.dispatch
       ├─ quant_method.apply / MoeRunner / run_moe_core
       └─ dispatcher.combine
```

在图上标出以下分支：

- `moe_a2a_backend` 如何影响实现类和 dispatcher；
- `moe_runner_backend` 如何影响 grouped GEMM；
- normal / low-latency 如何影响 `DispatchOutput` 格式；
- 量化方法何时参与。

### 验收标准

- 能从模型层入口口述到最终输出，不跳过数据格式转换；
- 能解释 `DeepEPMoE` 与 `FusedMoE` 不是“EP/非 EP”的简单二分；
- 能说明 Ascend FuseEP 为什么绕过普通 dispatcher 控制流。

---

## 五、第 2 天：数据布局、EPLB 与真实性能实验

## 步骤 6：手算 `src2dst` 与专家连续布局（60 分钟）

### 阅读

阅读 `python/sglang/srt/layers/moe/ep_moe/kernels.py`：

- `deepep_run_moe_deep_preprocess`；
- `compute_src2dst_triton_kernel`；
- `compute_seg_indptr_triton_kernel`；
- `ep_scatter` / `ep_gather`；
- `moe_ep_deepgemm_preprocess`。

暂时忽略 Triton 指针细节，先理解数据变换。

### 手算输入

设接收端有 3 个 token，$Top\text{-}K=2$，本地专家 ID 为 $0,1,2$：

```text
topk_ids = [[2, 0],
            [1, 2],
            [0, 1]]
```

展平后：

```text
[2, 0, 1, 2, 0, 1]
```

实践要求：

1. 稳定排序专家 ID；
2. 写出 `reorder_ids`；
3. 求逆置换 `src2dst`；
4. 写出 `seg_indptr`；
5. 写出 `masked_m`；
6. 把 3 行 hidden 展开成 6 行、按 expert 连续的 GEMM 输入；
7. 假设每个 assignment 得到一个标量输出，用 `topk_weights` 手算 `ep_gather`。

### 必须理解的两种布局

```text
通信布局：received-token 主序
              │ ep_scatter / pre-permute
              ▼
计算布局：token-expert assignment 主序，按 expert 连续
              │ grouped GEMM
              ▼
通信布局：received-token 主序
              │ combine
              ▼
来源 rank 的原 token 顺序
```

### 验收标准

- 能解释为什么 grouped GEMM 需要同一 expert 的行连续；
- 能解释 `src2dst` 是展平 assignment 的逆置换，而不是 token 索引；
- 能区分 `seg_indptr`（专家边界）和 `masked_m`（每专家有效行数）。

---

## 步骤 7：理解 MoE runner 与 normal/LL 布局（45 分钟）

### 阅读

1. `python/sglang/srt/layers/moe/moe_runner/`：先看公共抽象，再选当前硬件常用后端；
2. NVIDIA Hopper + FP8 可重点看 `deep_gemm.py`；
3. 通用理解可看 Triton runner；
4. W4A8/Blackwell 再扩展到 CUTLASS/FlashInfer 路径。

### 对比实践

分别画出：

**Normal/contiguous：**

```text
received tokens → scatter/pre-permute → 按 expert 连续 → grouped GEMM → gather
```

**Low-latency/masked：**

```text
dispatch 已生成 [local_expert, capacity, hidden]
→ masked_m 指示有效行
→ masked grouped GEMM
→ combine
```

记录 `expected_m`、`masked_m`、`M_max` 各自用于：

- kernel 配置选择；
- 跳过 padding；
- 分配静态容量或 CUDA Graph buffer。

### 验收标准

- 能解释 prefill 偏 normal、decode 偏 low-latency 的原因；
- 能解释 token 很少时 grouped GEMM 利用率为何下降；
- 能区分通信格式和 runner 计算格式。

---

## 步骤 8：EPLB 负载统计与专家重排（60 分钟）

### 阅读顺序

1. `docs/theory/distributed/EP.md` §6；
2. `python/sglang/srt/eplb/expert_distribution.py`：负载记录与 balancedness；
3. `eplb/eplb_manager.py`：`on_forward_pass_end`、`rebalance`；
4. `eplb/expert_location.py`：logical/physical 映射；
5. `eplb/expert_location_dispatch.py`：Top-K ID 映射；
6. `eplb/eplb_algorithms/`：只看算法输入输出，暂不深究所有细节。

### 手算热点场景

设 4 个 logical experts，4 个 rank，最近窗口内负载：

```text
E0=1000, E1=120, E2=100, E3=80
```

有 2 个 redundant physical slots：

1. 计算总 physical experts：$4+2=6$；
2. 判断该数量能否均匀分给 $ep\_size=4$；
3. 若不能，调整冗余数使其满足整除约束；
4. 为 E0 设计复制方案；
5. 写出一组可能的 `physical_to_logical_map`；
6. 解释 Router 仍输出 E0 时，dispatch 如何选择多个 E0 replica。

### 指标实践

重点关注：

- `--enable-expert-distribution-metrics`；
- `sglang:eplb_*` 相关指标；
- balancedness 越接近 1 通常越均衡；
- 最大 rank 负载比平均负载更能决定尾延迟。

建议记录：


| 时段    | 最大 expert 负载 | 平均负载 | balancedness | 最忙 rank | 吞吐 |
| ------- | ---------------: | -------: | -----------: | --------- | ---: |
| EPLB 前 |                  |          |              |           |      |
| EPLB 后 |                  |          |              |           |      |

### 验收标准

- 能解释“专家均匀放置”不等于“运行时负载均匀”；
- 能解释 logical expert 与 physical replica 的关系；
- 能说明 EPLB 的收益与成本：统计、显存冗余、权重搬迁、映射更新。

---

## 步骤 9：部署与性能实验（60 分钟）

> 以下命令是模板，模型路径、端口和硬件参数按本地环境调整。先确认模型支持对应量化和 runner，不要把不同后端的结果直接混为一谈。

### 实验 A：基线与纯 EP

基线（按模型支持情况使用 TP 或默认 MoE 路径）：

```bash
python -m sglang.launch_server \
  --model-path <MOE_MODEL> \
  --tp-size 8 \
  --ep-size 1 \
  --moe-a2a-backend none
```

纯 EP + DeepEP：

```bash
python -m sglang.launch_server \
  --model-path <MOE_MODEL> \
  --tp-size 8 \
  --ep-size 8 \
  --moe-a2a-backend deepep \
  --deepep-mode auto
```

比较：

- 单 rank 专家权重显存；
- prefill throughput；
- decode throughput；
- TPOT/TTFT；
- dispatch/combine 时间；
- 各 rank GPU 利用率。

### 实验 B：normal 与 low-latency

固定模型、EP size、请求集和并发，仅改变：

```text
--deepep-mode normal
--deepep-mode low_latency
--deepep-mode auto
```

建议至少使用两类 workload：

1. 长 prompt、短输出：偏 prefill；
2. 短 prompt、长输出：偏 decode。

预期观察：normal 更关注大 token 吞吐；low-latency 更关注 decode 小消息延迟。结果以真实 benchmark 为准。

### 实验 C：EPLB 前后对比

在 EP 配置上增加：

```bash
--enable-eplb \
--enable-expert-distribution-metrics \
--ep-num-redundant-experts <满足整除约束的数量> \
--eplb-rebalance-num-iterations 1000
```

控制变量：

- 相同请求集；
- 相同随机种子和采样参数；
- 相同并发与请求长度分布；
- 至少覆盖一个足以稳定专家统计的时间窗口。

### benchmark 记录表


| 配置    | A2A backend | Runner | EP | 模式        | EPLB | TTFT | TPOT | 吞吐 | 峰值显存 | balancedness |
| ------- | ----------- | ------ | -: | ----------- | ---- | ---: | ---: | ---: | -------: | -----------: |
| 基线    | none        | auto   |  1 | -           | 关   |      |      |      |          |              |
| 纯 EP   | deepep      | auto   |  8 | normal      | 关   |      |      |      |          |              |
| 纯 EP   | deepep      | auto   |  8 | low_latency | 关   |      |      |      |          |              |
| EP+EPLB | deepep      | auto   |  8 | auto        | 开   |      |      |      |          |              |

### 结果解释框架

若 EP 变慢，依次检查：

1. token 数是否太少，A2A 小消息成本占主导；
2. 是否跨机，网络带宽/尾延迟是否成为瓶颈；
3. 专家负载是否倾斜，最忙 rank 是否拖慢全组；
4. runner 是否适配 dtype、量化和 GPU 架构；
5. normal/low-latency 是否与 workload 匹配；
6. padding/capacity 是否造成大量无效行；
7. 是否启用了适合的通信计算重叠。

### 无 GPU 替代方案

无法运行多卡时，仍完成以下实践：

1. 手算步骤 1、2、6、8；
2. 从 `server_args.py` 整理一张参数约束表；
3. 对 `DeepEPMoE.forward_impl` 做静态调用链追踪；
4. 对 normal/low-latency 的 dataclass 字段做 shape 表；
5. 用纸笔比较 $T=32$ 与 $T=4096$ 时通信启动开销和 grouped GEMM 利用率。

---

## 六、关键源码导航图

```text
配置与建组
server_args.py
  ├─ _handle_moe_kernel_config
  ├─ _handle_a2a_moe
  └─ _handle_eplb_and_dispatch
        │
parallel_state.py::initialize_model_parallel
  ├─ _MOE_EP
  ├─ _MOE_DP
  └─ _MOE_TP

路由
model MoE block
  └─ router.py::FusedMoeRouter
       └─ topk.py::TopK
            └─ TopKOutput(topk_ids, topk_weights, router_logits)

MoE 主流程
fused_moe_triton/layer.py::FusedMoE.forward_impl
或 ep_moe/layer.py::DeepEPMoE.forward_impl
  ├─ dispatcher.dispatch
  ├─ run_moe_core / quant_method.apply / MoeRunner
  └─ dispatcher.combine

通信格式
token_dispatcher/base.py
  ├─ Standard
  ├─ DeepEP normal
  ├─ DeepEP low-latency
  ├─ FlashInfer
  └─ Mooncake/NIXL/Mori

本地专家布局与计算
ep_moe/kernels.py
  ├─ sort / seg_indptr / src2dst
  ├─ ep_scatter
  ├─ grouped GEMM
  └─ ep_gather / post-reorder

负载均衡
eplb/expert_distribution.py
  → eplb_manager.py
  → eplb_algorithms/
  → expert_location.py
  → expert_location_dispatch.py
```

---

## 七、参数约束检查清单

启动前至少检查：

1. $tp\_size$ 能被 $ep\_size\times moe\_dp\_size$ 整除；
2. physical expert 数能被 `ep_size` 整除：

$$
(E_{logical}+E_{redundant})\bmod ep\_size=0;
$$

3. 专用 A2A 后端是否要求 $ep\_size=tp\_size$；
4. `moe_runner_backend` 是否支持模型 dtype、量化格式和 GPU 架构；
5. DeepEP normal/low-latency 是否匹配 prefill/decode；
6. CUDA Graph 与当前 dispatcher 模式是否兼容；
7. EPLB 是否已启用负载记录，统计窗口是否足够；
8. 跨机 EP 的 IB/RoCE 带宽和拓扑是否满足 A2A 要求；
9. speculative/MTP 路径是否需要单独设置 MoE A2A 和 runner backend；
10. shared expert 是否参与融合或 overlap，避免重复计算/重复加权。

---

## 八、常见误区

1. **“EP 才能减少专家权重显存”**：不准确。TP 也能切 expert 权重；EP 的独特之处是沿 expert 维切分并通过 token 路由通信实现独立计算。
2. **“EP size 与 TP size 总是相乘”**：不准确。它们通常在同一个外层 TP rank 域内重组，满足

$$
moe\_tp\_size=\frac{tp\_size}{ep\_size\times moe\_dp\_size}.
$$

3. **“All-to-All 一定发送 $T\times K$ 份 hidden”**：不一定。同一 token 命中的多个专家若在同一目标 rank，后端可复用 hidden payload；本地命中也不走网络。
4. **“Router 输出 physical expert ID”**：启用 EPLB 后通常先表达 logical expert，再映射到 physical replica。
5. **“负载均衡只看每 expert token 数平均值”**：不够。性能由最忙 expert/rank 和尾部同步决定。
6. **“DeepEP 就是 MoE 计算 kernel”**：不准确。DeepEP 主要解决 token dispatch/combine；本地专家 GEMM 由 DeepGEMM、CUTLASS、Triton 等 runner 完成。
7. **“combine 一定负责 Top-K 加权”**：最终语义必须完成加权，但具体发生在本地 post-permute、combine 前后或融合算子中，取决于后端。

---

## 九、最终通关任务

不用看资料，在 20 分钟内完成：

### 任务 1：口述

从 `[T,H]` hidden 开始，完整讲清：

```text
router_logits → topk_ids/weights → logical-to-physical 映射
→ dispatch → local expert layout → grouped GEMM
→ post-reorder/topk weighted sum → combine → [T,H]
```

### 任务 2：拓扑

给定：

$$
tp\_size=16,\quad ep\_size=4,\quad moe\_dp\_size=2.
$$

计算 `moe_tp_size`，并画出一个 TP 域内的 MoE DP、EP、TP groups。

### 任务 3：数据布局

给定一个 `[T,K]` 的 `topk_ids` 小矩阵，手算排序结果、`seg_indptr`、`masked_m` 和 `src2dst`。

### 任务 4：性能诊断

解释以下现象可能的根因和验证方法：

- $EP=8$ 比 $EP=1$ 慢；
- 平均 GPU 利用率不低，但 TPOT 尾延迟很高；
- prefill 快、decode 慢；
- EPLB 开启后显存上升但吞吐没有改善；
- 跨机后 dispatch 时间激增。

### 通关标准

满足以下条件即完成 P4：

- 能画出 EP 一层完整数据流和 rank mesh；
- 能手算 Top-K dispatch/combine 与专家连续布局；
- 能从 `FusedMoE.forward_impl` 追到 dispatcher、runner 和 combine；
- 能解释 DeepEP normal/low-latency 差异；
- 能用 balancedness、最忙 rank、dispatch/combine 时间判断负载与通信瓶颈；
- 能设计一组控制变量明确的 EP/EPLB benchmark。

---

## 十、延伸阅读顺序

完成本计划后建议继续：

1. `docs/advanced_features/expert_parallelism_zh.md`：部署、后端、overlap；
2. `python/sglang/srt/layers/moe/ep_moe/README_zh.md`：kernel 与布局细节；
3. `python/sglang/srt/layers/moe/token_dispatcher/README_zh.md`：后端目录导航；
4. `python/sglang/srt/layers/moe/moe_runner/`：深入当前硬件的 grouped GEMM；
5. `python/sglang/srt/eplb/eplb_simulator/`：离线分析和重排模拟；
6. `docs/theory/distributed/DP_attention.md`：理解 Attention DP 与 EP 的 token gather/scatter 衔接；
7. Two-Batch Overlap / Single-Batch Overlap：研究通信计算重叠与共享专家融合。
