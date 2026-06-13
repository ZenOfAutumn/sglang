# AMD 平台 PD 分离（Disaggregation）测试学习文档

> 目录：`test/registered/amd/disaggregation/`
> 适用读者：希望理解 SGLang 在 **AMD ROCm GPU** 上 **Prefill/Decode 分离（PD Disaggregation）** CI 测试的开发者。

---

## 1. 背景：什么是 PD 分离

大语言模型推理分为两个阶段，二者的计算特征截然不同：

| 阶段 | 含义 | 计算特征 | 瓶颈 |
| --- | --- | --- | --- |
| **Prefill（预填充）** | 一次性处理完整 prompt，计算所有输入 token 的 KV cache | 计算密集（compute-bound），并行度高 | 算力（FLOPs） |
| **Decode（解码）** | 自回归逐 token 生成，每步只算一个新 token | 访存密集（memory-bound），并行度低 | 显存带宽 |

**PD 分离**把这两个阶段拆到**不同的 GPU/实例**上运行：

```
                    ┌─────────────┐
   请求 ──────────▶ │ Load Balancer│  (mini-lb / sglang_router)
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              ▼                          ▼
       ┌────────────┐   KV cache 传输  ┌────────────┐
       │  Prefill   │ ───────────────▶ │   Decode   │
       │  实例      │   (RDMA/Mooncake) │   实例      │
       └────────────┘                  └────────────┘
```

- **Prefill 实例**：算完 prompt 的 KV cache 后，通过高速网络（RDMA）把 KV cache 传给 Decode 实例。
- **Decode 实例**：接收 KV cache，继续自回归生成。
- **KV 传输后端**：CI 中默认用 **Mooncake**；KV 通过 **RDMA / InfiniBand** 设备传输。
- **Bootstrap**：Prefill 与 Decode 通过一个 bootstrap 端口建立配对握手。

> 更完整的设计说明见 `docs/advanced_features/pd_disaggregation.md`。

---

## 2. 目录内容总览

```
test/registered/amd/disaggregation/
├── test_disaggregation_basic.py   # 基础功能 + 容错 + 抢占回收 测试
└── test_disaggregation_pp.py      # 叠加流水线并行(PP) 的精度测试
```

两个文件都依赖同一套基础设施：

| 组件 | 路径 | 作用 |
| --- | --- | --- |
| `PDDisaggregationServerBase` | `python/sglang/test/server_fixtures/disaggregation_fixture.py` | 测试基类：拉起 Prefill/Decode/LB 三个进程、端口分配、健康检查、清理 |
| `register_amd_ci` | `python/sglang/test/ci/ci_register.py` | 把测试登记到 AMD CI 套件，标注预估时长 |
| `run_eval`（few-shot GSM8K） | `python/sglang/test/few_shot_gsm8k.py` | 用 GSM8K 数学题做 few-shot 评测，返回 accuracy |
| `popen_launch_pd_server` | `python/sglang/test/test_utils.py` | 以子进程启动一个 PD server 实例 |

---

## 3. 测试基类：`PDDisaggregationServerBase`

这是理解全部测试的关键。它继承自 `CustomTestCase`（SGLang 自定义的 `unittest.TestCase`）。

### 3.1 端口规划（`setUpClass`）

基于 `DEFAULT_URL_FOR_TEST` 的基础端口推导出 4 套端口，避免冲突：

```python
cls.lb_port        = base_port           # 负载均衡器（对外入口）
cls.prefill_port   = base_port + 100     # Prefill 实例
cls.decode_port    = base_port + 200     # Decode 实例
cls.bootstrap_port = base_port + 500     # Prefill/Decode 握手端口
```

### 3.2 传输后端与 RDMA 设备选择

```python
if is_in_ci():
    transfer_backend = ["--disaggregation-transfer-backend", "mooncake"]
    rdma_devices     = ["--disaggregation-ib-device", get_rdma_devices_args()]
else:
    # 本地：从环境变量 SGLANG_TEST_PD_DISAGG_BACKEND / _DEVICES 读取
```

`get_rdma_devices_args()` 会按以下优先级解析 RDMA 网卡：
1. 环境变量 `SGLANG_CI_RDMA_ALL_DEVICES`
2. 自动探测 `/sys/class/infiniband`（仅保留 ACTIVE、速率 ≥ 100Gbps、非 eth 命名的设备）
3. 兜底 `mlx5_roce0..7`

并依据 `CUDA_VISIBLE_DEVICES` 做 **GPU → RDMA 网卡** 的就近映射（每 4 个 GPU 为一组共享相邻网卡）。

### 3.3 启动负载均衡器（`launch_lb`）

用 `sglang_router` 以 PD 分离 + mini-lb 模式拉起：

```python
python3 -m sglang_router.launch_router \
    --pd-disaggregation --mini-lb \
    --prefill <prefill_url> --decode <decode_url> \
    --host <host> --port <lb_port>
```

### 3.4 生命周期

| 方法 | 作用 |
| --- | --- |
| `setUpClass` | 端口/后端配置（子类会 `super().setUpClass()` 后再拉起 server） |
| `wait_server_ready` | 轮询 `/health` 直到就绪或超时 |
| `tearDownClass` | `kill_process_tree` 杀掉 LB/Decode/Prefill 三个进程，再 sleep 5s |

> **统一启动顺序**（各子类 `setUpClass` 均遵循）：
> 1. `start_prefill()` / `start_decode()` —— **非阻塞**启动两个 server
> 2. `wait_server_ready(...)` —— 阻塞等待两者 `/health` 就绪
> 3. `launch_lb()` —— 启动负载均衡器并等待就绪

---

## 4. `test_disaggregation_basic.py` 详解

注册到 CI 套件：

```python
register_amd_ci(est_time=600, suite="stage-b-test-large-8-gpu-35x-disaggregation-amd")
```

共三个测试类，都使用 `DEFAULT_MODEL_NAME_FOR_TEST`，TP=1，attention backend 为 `aiter`（AMD 优化后端），并设置 `SGLANG_USE_AITER=1`。

### 4.1 `TestDisaggregationAccuracy` —— 基础功能（核心，已启用）

Prefill 用 GPU 0，Decode 用 GPU 1（`--base-gpu-id 1`，`--mem-fraction-static 0.8`）。包含 4 个用例：

| 用例 | 验证点 |
| --- | --- |
| `test_gsm8k` | 跑 200 道 GSM8K few-shot，断言 **accuracy > 0.70**（验证 PD 分离下精度正确） |
| `test_logprob` | 请求 `return_logprob`，断言 `output_logprobs` 长度 == `completion_tokens`，且 `input_logprobs` 非空 |
| `test_structured_output` | 用 `json_schema` 约束输出，断言结果是合法 JSON（验证约束解码在 PD 下可用） |
| `test_first_token_finish` | 首 token 即 EOS/停止词的边界场景：① 提高 EOS 的 logit_bias → 只生成 1 token；② 加 `ignore_eos` → 生成 > 1 token；③ 指定 stop 词 → 生成 1 token |

### 4.2 `TestDisaggregationMooncakeFailure` —— 容错（已注释，未启用）

- 通过 `DISAGGREGATION_TEST_FAILURE_PROB=0.05` 注入 5% 的 KV 传输失败率。
- `test_gsm8k` 预期出现大量失败，但**服务器不能崩溃**：捕获异常后访问 `/health_generate`，断言两端仍返回 200。
- `tearDownClass` 会清理该环境变量。

### 4.3 `TestDisaggregationSimulatedRetract` —— 抢占/回收（已注释，未启用）

- 通过 `SGLANG_TEST_RETRACT=true` 强制触发请求 **retract（抢占回收）** 逻辑（显存紧张时把请求踢回队列重排）。
- `test_gsm8k` 在频繁回收的情况下仍要求 **accuracy > 0.70**，验证回收不影响正确性。

---

## 5. `test_disaggregation_pp.py` 详解

在 PD 分离基础上叠加 **流水线并行（Pipeline Parallelism, PP）** 与 **张量并行（Tensor Parallelism, TP）**。

注册到 CI：

```python
register_amd_ci(est_time=600, suite="stage-b-test-large-8-gpu-35x-disaggregation-amd")
```

与 basic 的区别：
- 模型用 `try_cached_model(...)` 包装（优先用本地缓存权重，避免重复下载）。
- 统一设置 `--disable-overlap-schedule`、attention backend `aiter`。
- 每个 `test_gsm8k` 结尾 `time.sleep(5)`，留时间让**显存检查**触发（验证无泄漏）。

三个测试类的并行拓扑对比：

| 测试类 | Prefill 并行 | Decode 并行 | 额外特性 | 状态 |
| --- | --- | --- | --- | --- |
| `TestDisaggregationPrefillPPAccuracy` | TP=2, **PP=2** | TP=2（base-gpu-id=4） | — | **已启用** |
| `TestDisaggregationPrefillPPDynamicChunkAccuracy` | TP=2, **PP=2** | TP=2（base-gpu-id=4） | `--enable-dynamic-chunking` | 已注释 |
| `TestDisaggregationDecodePPAccuracy` | TP=2, **PP=2** | TP=2, **PP=2**（base-gpu-id=4） | Decode 端也开 PP | 已注释 |

> GPU 布局：Prefill 占用 GPU 0-3，Decode 通过 `--base-gpu-id 4` 占用 GPU 4-7，因此该套件需要 **8 卡**（对应套件名 `large-8-gpu`）。

三个类的 `test_gsm8k` 逻辑一致：200 道题，断言 **accuracy > 0.70**。

---

## 6. 关键命令行参数速查

| 参数 | 含义 |
| --- | --- |
| `--disaggregation-mode prefill/decode` | 实例角色 |
| `--disaggregation-bootstrap-port` | Prefill/Decode 握手端口 |
| `--disaggregation-transfer-backend mooncake` | KV 传输后端 |
| `--disaggregation-ib-device <dev>` | RDMA/IB 网卡 |
| `--tp-size / --tp` | 张量并行度 |
| `--pp-size` | 流水线并行度 |
| `--base-gpu-id` | 起始 GPU 编号（用于错开 Prefill/Decode 占卡） |
| `--attention-backend aiter` | AMD 上的注意力后端 |
| `--mem-fraction-static` | 静态显存占比 |
| `--disable-overlap-schedule` | 关闭调度重叠（PP 场景需要） |
| `--enable-dynamic-chunking` | 动态分块预填充 |

环境变量：

| 变量 | 作用 |
| --- | --- |
| `SGLANG_USE_AITER=1` | 启用 AMD AITER 优化 |
| `SGLANG_TEST_RDMA_DEVICE` | 指定 RDMA 设备 |
| `DISAGGREGATION_TEST_FAILURE_PROB` | 注入 KV 传输失败率（容错测试） |
| `SGLANG_TEST_RETRACT=true` | 强制触发请求抢占回收 |

---

## 7. 如何运行

> 这些用例需要 **AMD ROCm 多 GPU 环境**（basic 至少 2 卡，pp 需 8 卡）与 RDMA 网络，通常在 CI 中运行。

```bash
# 运行 basic 全部用例
python3 -m unittest test.registered.amd.disaggregation.test_disaggregation_basic -v

# 仅运行基础精度类
python3 -m unittest \
  test.registered.amd.disaggregation.test_disaggregation_basic.TestDisaggregationAccuracy -v

# 运行 PP 用例
python3 -m unittest test.registered.amd.disaggregation.test_disaggregation_pp -v
```

本地运行需先指定传输后端与 RDMA 设备（非 CI 分支）：

```bash
export SGLANG_USE_AITER=1
export SGLANG_TEST_RDMA_DEVICE=mlx5_roce0   # 按实际网卡填写
```

---

## 8. 阅读这套测试的建议路径

1. 先读 `disaggregation_fixture.py` 的 `PDDisaggregationServerBase`，掌握「三进程 + 端口规划 + 健康检查」骨架。
2. 再读 `test_disaggregation_basic.py::TestDisaggregationAccuracy`，理解一次完整的「启动 → 等就绪 → 跑 GSM8K → 断言精度」流程。
3. 对照 `test_logprob / test_structured_output / test_first_token_finish` 理解功能正确性如何被验证。
4. 最后看容错（Failure）、回收（Retract）、并行（PP）三类对同一骨架的扩展方式。

---

## 9. 小结

- 这套测试的**核心目标**是：在 AMD ROCm 上验证 **PD 分离 + 多种并行/容错配置** 下的**精度**与**鲁棒性**。
- 验证手段统一为 **GSM8K few-shot 精度（>0.70）** + 功能性断言（logprob、JSON、停止条件）。
- 所有测试共用 `PDDisaggregationServerBase` 这一套「拉三进程」的脚手架，差异仅在于**启动参数**与**注入的环境变量**。

