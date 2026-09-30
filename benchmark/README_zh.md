# benchmark 目录说明

> 本文件说明仓库根 `benchmark/` 目录下各脚本与子目录的用途，方便快速定位「想知道某件事该跑哪个脚本」。

## 定位与边界

`benchmark/` 目录主要负责三类工作：

1. **数据集精度评测**：对启动好的 SGLang server 发请求，统计准确率 / WER 等质量指标（每个数据集一个子目录）。
2. **特性 / 系统专项基准**：针对某个特性（分层缓存、线性注意力、投机解码等）做端到端或半端到端验证。
3. **单算子微基准与自动调参**：`kernels/` 下对单个 kernel 做延迟对比 / autotune，产出配置给运行时使用。

需要注意：

- **端到端吞吐/延迟的官方入口不在本目录**，而是 `python/sglang/bench_serving.py`（`python -m sglang.bench_serving`）。
- **可复用的评测框架/数据加载**在 Python 包 `python/sglang/benchmark/`（`bench_utils.py`、`datasets/`、`utils.py`）中；本目录脚本大量 `import sglang.benchmark...` 复用该框架。
- 约定：精度评测脚本通常叫 `bench_sglang.py`（被测方），另配 `bench_hf.py`（HF 基线）；算子微基准通常 `import sglang.kernels.ops.*` 与 `sglang.benchmark.bench_utils.run_bench`。

## 目录总览

```
benchmark/
├── bench_adaptive_speculative.py   # 顶层独立脚本
├── lean_kernel_sweep.py            # 顶层独立脚本
│
├── asr/  boolq/  gsm8k/  hellaswag/  mmlu/  mmmu/  llava_bench/  ocr/   # 数据集精度评测
├── prefill_only/                   # embeddings / score 接口评测
│
├── hicache/                        # 分层 KV 缓存专项
├── unified_memory/                 # 统一显存池 peer-aware 驱逐
├── bench_attention_sink/           # attention sink kernel
├── bench_linear_attention/         # 线性注意力（GDN/KDA）
├── fla/                            # FLA layernorm_gated
├── bench_rope/                     # mRoPE index builder
├── lora/                           # 多 LoRA 服务
├── scheduler/                      # 调度器 token 存储
├── io/                             # 请求体校验
├── bench_pynccl_allocator/         # PyNccl 显存段追踪
│
├── hf3fs/                          # HF3FS 存储后端
├── blog_v0_2/                      # 复现 v0.2 博客对比
├── deepseek_v3/                    # DeepSeek 部署与基准指引
│
└── kernels/                        # 算子微基准与调参（见下）
```

---

## 一、顶层直接脚本

| 脚本 | 作用 |
| --- | --- |
| `bench_adaptive_speculative.py` | 自适应投机解码 vs 静态基线。对同一负载分别压自适应 server 与一个或多个静态 server，对比吞吐、延迟、接受长度；负载分 `low`（低接受率）、`high`（高接受率）、`transition`（低/高交替，压测运行时切换）。 |
| `lean_kernel_sweep.py` | Lean vs 标准 SplitK 的 decode attention 扫参。覆盖 Qwen2.5-7B（28Q/4KV）与 Llama-3.1-8B（32Q/8KV），batch∈{1,2,4,8,16,32} × context∈{8K…128K}，报告单次 kernel 延迟、加速比、与 SplitK 的 cosine 一致性，并输出 CSV。 |

---

## 二、数据集精度评测

每个目录先启动对应模型的 SGLang server，再跑评测脚本；部分目录附带 HF/OpenAI 基线或数据下载脚本。

| 目录 | 指标 | 主要文件 | 备注 |
| --- | --- | --- | --- |
| `gsm8k/` | 数学推理准确率 | `bench_sglang.py`、`README.md` | 支持 GSM8K Platinum（`--platinum`） |
| `mmlu/` | 知识问答准确率 | `bench_sglang.py`、`bench_hf.py`、`download_data.sh` | 需先下载数据 |
| `hellaswag/` | 常识推理准确率 | `bench_sglang.py` | — |
| `boolq/` | 是非问答准确率 | `bench_sglang.py`、`convert_parquet_to_json.py`、`parquet_to_json.sh` | 需先转换数据 |
| `mmmu/` | 多模态 VLM 准确率 | `bench_sglang.py`、`bench_hf.py`、`data_utils.py`、`eval_utils.py`、`prompt_format.yaml` | 支持 `--lora-path`、`--response-answer-regex` |
| `llava_bench/` | 图像问答 + MME | `bench_sglang.py`、`bench_hf_*.sh`、`download_images.py`、`questions.jsonl` | 含 HF / llama.cpp 对照 |
| `ocr/` | DeepSeek-OCR 在 olmOCR-bench 上的通过率 | `bench_sglang.py`、`eval_utils.py`、`generate_report.py` | 覆盖 7 类文档测试，可出 HTML 报告 |
| `asr/` | 语音识别 WER / 吞吐 | `bench_sglang.py`、`README.md` | 支持 Whisper / Qwen3-ASR，chat 与 transcription 两种 API |
| `prefill_only/` | embeddings / score 接口性能 | `bench_embeddings.py`、`bench_score.py`、`util.py` | 纯 HTTP 压测 `/v1/embeddings`、`/v1/score`，可配 RPS/Poisson |

---

## 三、特性 / 系统专项基准

### `hicache/` —— 分层 KV 缓存（HiCache）

| 文件 | 作用 |
| --- | --- |
| `bench_multiturn.py` | 合成多轮对话基准（增长历史、round barrier、固定 seed），对比 radix / fcfs / hicache 等配置 |
| `bench_serving.py` | 在线 serving 基准（自 `bench_serving` 改造，动态请求） |
| `bench_warm_cache.py` | 精确控制共享前缀比例的 warm-cache 基准（先 flush，再 warm 前缀，再压全量 prompt） |
| `bench_long_context.py` | 长上下文负载（委托 `bench_multiturn.py`） |
| `bench_mix.py` / `bench_mix.sh` | 并发多轮混合负载；`.sh` 负责起 server + 读取配置 JSON 运行 |
| `bench_buffer_mode.py` | HiCache `buffer_only` 模式矩阵：对 (model, host-memory-mode) 组合跑 multiturn / longctx，报告命中率、延迟与流水线计数 |
| `bench_hicache_write_back.py` | write-back / KV 传输 kernel 基准（`sgl_kernel.kvcacheio` + `triton.testing`） |
| `perf.py` | HiCache 控制器微基准（prefetch / storage 操作、allocator、memory pool） |
| `data_processing.py` | 数据集加载与 prompt 准备（sharegpt 等） |
| `nextqa.py` | NExTQA 视频数据集加载器 |
| `download.sh` | 下载 sharegpt / ultragpt / loogle / nextqa 数据 |

### 其他专项

| 目录 | 作用 |
| --- | --- |
| `unified_memory/` | 统一显存池的 peer-aware 驱逐策略验证：用可复用前缀填满共享 FULL/Mamba pool，观察分配器容量增益是否让 radix 驱逐提前停止 |
| `bench_linear_attention/` | 线性注意力（GDN / KDA）kernel 对比，详见下表 |
| `bench_attention_sink/` | attention sink 的 Triton decode kernel 性能报告（Triton perf_report，扫 seq_len × batch） |
| `fla/` | FLA `layernorm_gated`（`_layer_norm_fwd` / `rms_norm_ref`）微基准 |
| `bench_rope/` | `MRotaryEmbedding.get_rope_index_glm4v`（GLM4V mRoPE index builder）基准，合成多模态输入 |
| `lora/` | 多 LoRA 服务：`launch_server.py` 拉起带 `--lora-paths` 的 server，`lora_bench.py` 压测 |
| `scheduler/` | `bench_token_storage.py`：对比 `list[int]` vs `array.array('q')` 作为 `Req.origin_input_ids` / `output_ids` 全生命周期开销 |
| `io/` | `bench_input_ids_validator.py`：FastAPI 请求体绑定中 `input_ids` 校验开销（pydantic 逐元素 vs C 循环校验器） |
| `bench_pynccl_allocator/` | `bench_segment_tracking.py`：比较 `nccl_allocator_register_segments_with_comm()` 与 `torch.cuda.memory.memory_snapshot()` 的 CPU 开销 |

### `bench_linear_attention/` 明细

> 均为 kernel 级「候选 vs 基线」对比 + 数值一致性校验，多数面向 SM100 / Blackwell 或 AMD。

| 文件 | 对比内容 |
| --- | --- |
| `bench_gdn_decode.py` | GDN packed decode（单 kernel）vs baseline（split + view + gating） |
| `bench_gdn_prefill.py` | Triton GDN vs FlashInfer GDN（prefill） |
| `bench_gdn_prefill_cutedsl.py` | Triton GDN vs CuTeDSL GDN（prefill，SM100） |
| `bench_gdn_qkv_split.py` | `fused_qkv_split_gdn_prefill`（Triton 融合投影） |
| `bench_kda_decode.py` | KDA packed decode vs baseline（含 Helion） |
| `bench_kda_prefill_cutedsl.py` | Triton KDA vs CuTeDSL KDA（prefill，SM100，含 Helion） |
| `bench_kda_flashinfer_mtp.py` | FlashInfer KDA（SM100）vs Triton KDA 的 decode 与 MTP verify |
| `bench_cutedsl_kda_decode.py` | CuTe DSL KDA decode vs Triton KDA decode |
| `bench_fused_gate_cumsum.py` | 融合 Gate+Cumsum vs 分步（KDA） |
| `bench_int8_checkpoint_reuse.py` | int8 线性注意力 checkpoint pool 的前缀复用容量与延迟（需运行中的 server） |

---

## 四、存储、复现与部署

| 目录 / 文件 | 作用 |
| --- | --- |
| `hf3fs/` | HF3FS 存储后端基准：`bench_client.py`（客户端读写）、`bench_storage.py`（HiCacheHF3FS）、`bench_zerocopy.py`（zero-copy）、`bench.sh`（一键脚本，含起 server 用 `--hicache-storage-backend hf3fs`） |
| `blog_v0_2/` | 复现 SGLang v0.2 博客中 405B 上 SGLang / vLLM / TensorRT-LLM 的对比：`405b_*.sh`、`README.md`、`config.md`（TRT/vLLM/SGLang 的模型配置样例） |
| `deepseek_v3/` | DeepSeek V3/V3.1/R1 的部署方案与精度/性能基准指引（文档性质，含各类硬件与量化（FP8/FP4/AWQ/INT8）的启动命令与 bench 命令） |

---

## 五、算子微基准与调参（`kernels/`）

统一目标：对单个 kernel 做「候选 vs 基线」延迟对比，和/或 autotune 出最优 launch 配置。

| 子目录 | 内容 |
| --- | --- |
| `kernels/all_reduce/` | 集合通信 all-reduce 对比：`benchmark_all_reduce.py`（SGLang custom vs torch symm-mem）、`benchmark_aiter.py`（vs Aiter）、`benchmark_mscclpp.py`（torch/NCCL vs MSCCL++ eager/graph vs PyNccl）、`benchmark_torch_symm_mem.py`、`benchmark_fused_ar_rms_amd.py`（AMD fused AR+RMSNorm）、`benchmark_fused_ar_rms_quant_amd.py`（AMD fused AR+RMSNorm+per-group FP8 quant） |
| `kernels/all_gather/` | `benchmark_aiter.py`：logical TP all-gather（RCCL）vs Aiter custom all-gather |
| `kernels/flashinfer_allreduce_fusion/` | `benchmark_fused_collective.py`：AR+Residual+RMSNorm（±FP8/FP4 量化）融合 vs 分离实现 |
| `kernels/fused_moe_triton/` | MoE kernel 调参与对比：`tuning_fused_moe_triton.py`（TP/EP、FP8/INT8、MLLM 调参）、`tuning_fused_moe_triton_sep.py`（上下 kernel 分开调、TMA）、`tuning_client.py`（产生 topk_ids）、`benchmark_sglang_fused_moe_triton.py`、`benchmark_vllm_vs_sglang_fused_moe_triton.py`、`benchmark_torch_compile_fused_moe.py` |
| `kernels/quantization/` | `tuning_block_wise_kernel.py`（W8A8 block-wise FP8/INT8 autotune，产出 config JSON）、`bench_int8_quant.py`（per-token int8 quant vs vllm/torch） |
| `kernels/deepseek/` | DeepGEMM 相关：`benchmark_deepgemm_fp8_gemm.py`、`benchmark_deepgemm_fp8_gemm_blackwell.py`、`benchmark_deepgemm_fp8_group_gemm.py`、`benchmark_cute_dsl_fp8_paged_mqa_logits.py`（DS2/3 attention logits）、`benchmark_q8kv8_kv_gather.py`、`benchmark_q8kv8_q_prep.py` |
| `kernels/attention/` | `bench_flash_attention_fp8.py`（FA4 CuTe-DSL FP8，SM100）、`bench_gdn_replayssm_decode.py`（ReplaySSM GDN decode）、`sm90_config_search.py`（SM90 attention fwd/bwd 可行配置搜索）、`fa4_benchmark_utils.py`（辅助函数） |
| `kernels/decoding_attention_triton/` | `triton_flashinfer_cudnn.py`：Triton / FlashInfer / cuDNN 三种 decode attention 对比 |
| `kernels/elementwise/` | `benchmark_concat_mla.py`：`concat_mla_k` Triton vs `sgl_kernel` |
| `kernels/scheduler_batch/` | 调度批相关小 kernel：`benchmark_get_last_loc_triton.py`、`benchmark_write_req_to_token_pool_triton.py` |
| `kernels/sliding_window_attention_triton/` | `bench_triton_swa_kernel.py`：滑窗注意力 Triton kernel |
| `kernels/verify_splitkv_triton/` | `bench_verify_splitkv.py`：EAGLE-verify split-KV kernel vs `extend_attention_fwd`（面向 AMD MI35x/gfx950） |
| `kernels/deepep/` | `tuning_deepep.py`、`deepep_utils.py`：DeepEP 通信调参（产出 `deepep_tuned.json`） |
| `kernels/lora_csgmv/` | `tune_lora_csgmv.py`：LoRA CSGMV（shrink/expand）kernel 调参 |
| `kernels/`（根） | `bench_fused_gate_sigmoid_mul_add.py`、`bench_fused_sigmoid_mul.py`（Qwen3.5 MoE gate）、`bench_fused_rmsnorm_fp8_quant.py`（RMSNorm+静态 FP8 量化，flashinfer vs CuTe-DSL vs unfused）、`bench_paged_mqa_metadata.py`（paged MQA metadata JIT kernel） |

---

## 六、相关但不在本目录的入口

| 入口 | 说明 |
| --- | --- |
| `python/sglang/bench_serving.py` | 官方在线/离线服务吞吐基准（`python -m sglang.bench_serving`），本目录许多脚本基于它改造 |
| `python/sglang/benchmark/` | 可复用 benchmark 框架：`bench_utils.py`（`run_bench` 等）、`datasets/`、`utils.py` |
| `python/sglang/bench_one_batch.py` / `bench_one_batch_server.py` | 单批前向延迟基准 |
| `test/`、`python/sglang/test/` | CI 测试套件与测试工具（`test/run_suite.py`） |

---

## 七、通用使用约定

1. **先跑后读**：精度评测类脚本需要先启动 server（如 `python -m sglang.launch_server --model-path ... --port 30000`），再用脚本连接该端口。
2. **数据先行**：多数数据集评测需要先下载/转换数据（见各子目录的 `README.md` / `download*.sh`）。
3. **算子级脚本**：`kernels/` 下的脚本多为直接调 kernel，需要装好 `sglang`、`sgl-kernel` 及（部分脚本）`deep_gemm` / `flashinfer` / `vllm` / `aiter` / `deep_ep` 等依赖；部分脚本用 `torchrun` 多进程起（集合通信类）。
4. **调参产物**：`kernels/*/tuning_*.py` 生成的 JSON 需放到运行时对应目录（如 MoE 放 `srt/layers/moe/moe_runner/triton_utils/configs/...`，量化放 `srt/layers/quantization/configs/`）才能生效。
