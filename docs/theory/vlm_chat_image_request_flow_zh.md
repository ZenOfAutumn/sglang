# Chat 接口带 `image_url` 的请求处理流程（至 Prefill 执行前）

本文梳理一条 `POST /v1/chat/completions` 请求，在 `messages[].content` 中携带 `image_url` 时，从 HTTP 入口一路走到 **语言模型 Prefill 正式执行前** 的完整链路。示例模型以 Qwen2.5-VL 为主（Jinja chat template 路径，最常见），其他 VLM 只在"模型专属"环节有差异。

示例请求：

```json
{
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
      {"type": "text", "text": "图里有什么？"}
    ]
  }]
}
```

---

## 0. 总览：三个进程、四个阶段

```mermaid
flowchart LR
    subgraph P1["进程 A：HTTP Server + TokenizerManager（主进程）"]
        A1["① OpenAI 协议层<br/>消息拆解 + 渲染 chat template"]
        A2["② 多模态预处理<br/>下载/解码图片 → HF Processor<br/>→ 占位符展开 + 像素特征 + hash"]
    end
    subgraph P2["进程 B：Scheduler（每个 GPU rank 一个）"]
        B1["③ 构造 Req<br/>占位 token 替换为 pad_value<br/>入 waiting_queue"]
        B2["④ 组 Prefill 批<br/>前缀匹配 → PrefillAdder<br/>→ ScheduleBatch / ForwardBatch"]
    end
    subgraph P3["Scheduler 进程内：ModelRunner"]
        C1["⑤ 模型 forward 入口<br/>ViT 编码 + embedding 拼接"]
        C2["⑥ 语言模型 Prefill<br/>（本文终点）"]
    end
    A1 --> A2 -->|ZMQ: TokenizedGenerateReqInput| B1 --> B2 --> C1 --> C2
```

| 阶段 | 输入 | 输出 |
|---|---|---|
| ① 协议层 | OpenAI JSON | `GenerateReqInput(text=渲染后的 prompt, image_data=[ImageData])` |
| ② 多模态预处理 | prompt 文本 + 图片 URL | `input_ids`（占位已展开）、`mm_items`（pixel_values / grid_thw / offsets / hash）、`mrope_positions` |
| ③ Scheduler 收请求 | `TokenizedGenerateReqInput` | `Req`（`origin_input_ids` 中图片 token 已替换为 pad_value） |
| ④ 组批 | `waiting_queue` | `ScheduleBatch` → `ModelWorkerBatch` → `ForwardBatch` |
| ⑤ forward 入口 | `ForwardBatch` | `input_embeds`（文本 embedding 中图片位置被 ViT 输出覆盖） |

---

## 1. 阶段 ①：OpenAI 协议层（`serving_chat.py`）

```mermaid
flowchart TD
    S([POST /v1/chat/completions]) --> V["FastAPI 解析为 ChatCompletionRequest<br/>image_url → ChatCompletionMessageContentImagePart"]
    V --> H["OpenAIServingChat.handle_request<br/>(serving_base)"]
    H --> VAL{"_validate_request<br/>模型是否支持该模态？"}
    VAL -- 否 --> E400([400 BadRequest])
    VAL -- 是 --> EXE["run_in_executor 线程池中执行<br/>_convert_to_internal_request<br/>（避免 Jinja 渲染/分词阻塞事件循环）"]
    EXE --> PM["_process_messages<br/>处理 reasoning / tools / tool_call 约束"]
    PM --> BR{"选择模板路径"}
    BR -- "request.input_ids 已给出" --> RAW["直接使用 input_ids<br/>image_data=None"]
    BR -- "chat_template_name 为空（默认）" --> JJ["_apply_jinja_template"]
    BR -- "指定了 --chat-template（conversation 模板）" --> CV["_apply_conversation_template<br/>generate_chat_conv：<br/>每个 image_url 追加 conv.image_token<br/>并 conv.append_image(url)"]

    JJ --> PCF["对每条 message 调用<br/>process_content_for_template_format"]
    PCF --> FMT{"模板 content 格式<br/>（AST 检测）"}
    FMT -- openai --> OA["image_url 块 → image_data.append(ImageData(url, detail, ...))<br/>content 中替换为 {type: 'image'}"]
    FMT -- string --> ST["只保留文本拼接<br/>（图片被丢弃）"]
    OA --> REN["_render_and_encode_chat_template<br/>tokenizer.apply_chat_template 渲染<br/>{type:image} → '&lt;|vision_start|&gt;&lt;|image_pad|&gt;&lt;|vision_end|&gt;'"]
    REN --> DEC["is_multimodal：prompt = 渲染文本<br/>（单个占位符，尚未按图片大小展开）"]
    CV --> DEC2["prompt = conv.get_prompt()<br/>image_data = conv.image_data"]
    DEC --> MPR["MessageProcessingResult<br/>(prompt, prompt_ids, image_data, modalities, stop)"]
    DEC2 --> MPR
    RAW --> MPR
    MPR --> SP["request.to_sampling_params()"]
    SP --> EP{"_engine_prompt<br/>有图片 → 传 text，<br/>交给 MM processor 重新分词"}
    EP --> GRI["GenerateReqInput(<br/>text=prompt, image_data=[ImageData],<br/>sampling_params, modalities, stream, rid ...)"]
    GRI --> ST2{"stream?"}
    ST2 -- 是 --> GS["_handle_streaming_request<br/>_generate_chat_stream"]
    ST2 -- 否 --> GN["_handle_non_streaming_request"]
    GS --> TM(["tokenizer_manager.generate_request(...)"])
    GN --> TM
```

要点：

- **图片在这一层不下载**，只把 URL（或 base64 / `data:` / `file://`）包装成 `ImageData` 收集起来。
- **占位符只有一个**：chat template 只渲染出 `<|image_pad|>` 一个 token，真实需要的 token 数（取决于分辨率）要等阶段 ② 的 HF Processor 计算后才展开。
- 对多模态模型，协议层**有意传文本而不是 token id**（`_engine_prompt`），因为 MM processor 需要在文本上定位并展开占位符。
- 流式请求会先 `__anext__()` 拉取第一个结果，使校验错误（如超长）能以 HTTP 400 返回，而不是混进 SSE 流。

---

## 2. 阶段 ②：TokenizerManager 与多模态预处理

### 2.1 TokenizerManager 主干

```mermaid
flowchart TD
    G(["generate_request(obj)"]) --> N["obj.normalize_batch_and_arguments()<br/>_set_default_priority"]
    N --> INIT["_init_req_state：rid_to_state 建立 ReqState"]
    INIT --> EPD{"EPD 分离<br/>language_only？"}
    EPD -- 是 --> ENC["_handle_epd_disaggregation_encode_request<br/>把图片派发给独立 Encoder 服务"]
    EPD -- 否 --> PAUSE
    ENC --> PAUSE["等待 is_pause_cond / LoRA 校验"]
    PAUSE --> TOK["_tokenize_one_request(obj)"]

    TOK --> C1{"contains_mm_input<br/>且 processor 会从原始 prompt 生成 ids？"}
    C1 -- 是 --> NOIDS["input_ids = None<br/>（跳过文本分词，由 processor 负责）"]
    C1 -- 否 --> TT["_tokenize_texts(prompt)<br/>先得到一份纯文本 input_ids"]
    NOIDS --> MMQ
    TT --> MMQ{"should_run_mm_processor？"}
    MMQ -- 否 --> VAL
    MMQ -- 是 --> LIM["_validate_mm_limits（图片数量上限）<br/>_normalize_mm_content_hashes"]
    LIM --> WHERE{"EPD 模式下<br/>embedding 已由 Encoder 送达？"}
    WHERE -- "是（zmq_to_tokenizer）" --> RECV["mm_receiver.recv_mm_data<br/>直接拿到预计算 embedding"]
    WHERE -- 否 --> PROC["mm_processor.process_mm_data_async(<br/>image_data, input_text=prompt, request_obj)"]
    RECV --> OUT
    PROC --> OUT["MultimodalProcessorOutput"]
    OUT --> REP["input_ids ← mm_inputs.input_ids<br/>（已展开的完整 token 序列）"]
    REP --> HASH["可选：调用方 mm_hashes 覆盖 item.hash<br/>可选：SGLANG_MM_PRECOMPUTE_HASH 预计算 pad_value"]
    HASH --> VAL["_validate_one_request<br/>上下文长度 / max_new_tokens 校验"]
    VAL --> CTO["_create_tokenized_object<br/>→ TokenizedGenerateReqInput(<br/>input_ids, mm_inputs, sampling_params ...)"]
    CTO --> SEND["_send_one_request<br/>· CUDA VMM / CUDA IPC 特征传输准备<br/>· wrap_shm_features：大张量走共享内存<br/>· _dispatch_to_scheduler（ZMQ PUSH）"]
    SEND --> WAIT(["_wait_one_response：等待 Detokenizer 回传结果"])
```

### 2.2 MM Processor 内部（以 `QwenVLImageProcessor` 为例）

```mermaid
flowchart TD
    P(["process_mm_data_async"]) --> CACHE{"artifact 快路径可用？<br/>无 video/audio、无已预处理输入，<br/>且缓存已启用或模型强制走 artifact"}
    CACHE -- 是 --> ART["prepare_media_artifacts<br/>按 content_digest + 预处理配置复用图片产物<br/>compose_image_artifacts 直接拼装输出<br/>（内部流程见 §2.3）"]
    CACHE -- 否 --> UNC["_process_mm_data_uncached"]
    ART -- 拼装失败/占位符对不上 --> UNC
    ART -- 成功 --> RET

    UNC --> LOAD["load_mm_data(prompt, image_data, mm_tokens)"]
    LOAD --> SPLIT["按多模态 token 正则切分 prompt<br/>统计占位符个数"]
    SPLIT --> CNT{"占位符数 == 图片数？"}
    CNT -- 是 --> FAST["fast_load_mm_data"]
    CNT -- 否 --> LEG["legacy_load_mm_data"]
    FAST --> IO["IO 线程池并发 _load_single_item<br/>load_image：http(s) 下载 / base64 / data: / file://<br/>PIL 解码（或 nvJPEG GPU 解码）→ RGB"]
    LEG --> IO
    IO --> BASE["BaseMultiModalProcessorOutput<br/>(input_text, images=[PIL...])"]
    BASE --> PAC["process_and_combine_mm_data_async<br/>（可在 mm_processor_executor 进程/线程池中运行）"]
    PAC --> HF["process_mm_data → HF AutoProcessor(<br/>text=[prompt], images=[...])<br/>· 图像 resize / 切 patch / 归一化 → pixel_values<br/>· 计算 image_grid_thw<br/>· 把 1 个 &lt;|image_pad|&gt; 展开成 t·h·w/merge² 个"]
    HF --> COL["collect_mm_items_from_processor_output<br/>→ MultimodalDataItem(modality=IMAGE,<br/>feature=pixel_values, image_grid_thw)"]
    COL --> OFF["get_mm_item_offsets：在展开后的 input_ids 中<br/>定位每张图的 [start, end] 区间"]
    OFF --> EXP["get_new_expanded_mm_items<br/>把打包在一起的多图拆成每图一个 item"]
    EXP --> FIN["_finalize_mm_items<br/>· set_pad_value：对特征算 hash → pad_value(s)<br/>· 特征转 CPU / CUDA IPC / 共享内存，准备跨进程传输"]
    FIN --> PAD["build_padded_input_ids<br/>把 offsets 区间内 token 替换为 pad_value 序列"]
    PAD --> ROPE["计算 M-RoPE：<br/>_compute_image_only_mrope_positions_from_offsets<br/>或 MRotaryEmbedding.get_rope_index<br/>→ mrope_positions (3, L), mrope_position_delta"]
    ROPE --> RET(["MultimodalProcessorOutput(<br/>input_ids, padded_input_ids, mm_items,<br/>im_token_id, mrope_positions, ...)"])
```

要点：

- **占位符展开发生在这里**：一张 448×448 的图在 Qwen2.5-VL 中会从 1 个 `<|image_pad|>` 变成 256 个（patch=14，merge=2）。
- **pad_value 是 RadixCache 能复用图片前缀的关键**：每张图按特征内容 hash 派生出一组 ≥ `MM_PAD_SHIFT_VALUE`（1,000,000）的伪 token id，替换图片占位区。相同图片 → 相同 pad 序列 → 前缀树可命中；不同图片即便文本相同也不会误命中。
- 图像解码在 IO 线程池里完成，HF Processor 可在独立 executor 中执行，避免阻塞 TokenizerManager 的事件循环。
- **这里其实有两条路径**：`prepare_media_artifacts`（预处理器缓存打开，命中可省掉「下载 + 解码 + HF processor」）与 `_process_mm_data_uncached`（全量走 `load_mm_data`）。两者的产物最终都收敛成同一个 `MultimodalProcessorOutput`，流程细节见 §2.3。

### 2.3 `prepare_media_artifacts` 的内部执行流程

`process_mm_data_async`（`qwen_vl.py:1175`）先做一个分支判断，只有满足下列全部条件才走 artifact 快路径，否则直接用 `_process_mm_data_uncached` 全量处理：

- `image_data` 非空，且 `video_data` / `audio_data` 都为空（artifact 目前只覆盖图片）；
- `image_data` 里没有已预处理的输入（`_is_preprocessed_input`）；
- `mm_preprocess_cache.enabled`（`--mm-preprocess-cache-size-mb > 0`，按 tokenizer worker 均分预算）**或** 模型声明了 `uses_media_artifacts_without_cache`（`qwen3_vl` / `qwen3_vl_moe`，缓存关掉也走 artifact，只是不保留）。

随后按缓存是否启用在两个入口间二选一：`prepare_media_artifacts`（带 LRU + 单飞）或 `prepare_media_artifacts_without_cache`（每次都重算、不入缓存）。两者共用同一套 snapshot / decode / `prepare_artifact_batch` 契约。

```mermaid
flowchart TD
    P(["process_mm_data_async"]) --> GATE{"artifact 快路径可用？"}
    GATE -- 否 --> UNC["_process_mm_data_uncached"]
    GATE -- 是 --> PICK{"mm_preprocess_cache.enabled？"}
    PICK -- 是 --> PMA["prepare_media_artifacts"]
    PICK -- 否 --> PMW["prepare_media_artifacts_without_cache"]

    PMA --> S1["① 对齐校验：content_hashes / featureless_hit_mask 长度须等于媒体数<br/>parse_content_hash 归一化为 sha256:64位hex"]
    S1 --> S1B["② fast path：trust_mm_content_hashes 且调用方给了 hash<br/>直接用 caller_hash 建 key 查缓存，完全不读媒体"]
    S1B --> S2
    PMW --> S2["③ snapshot_media_source 在 IO 线程池并发执行<br/>http(s) → download_remote_media 真正下载<br/>data:/base64/file:// → 取字节<br/>→ MediaSnapshot content_digest = sha256(bytes)"]
    S2 --> S2B{"caller_hash 与实测 digest 一致？"}
    S2B -- 否 --> ERR(["ValueError: content hash mismatch"])
    S2B -- 是 --> S3["④ _artifact_key = sha256(content_digest + modality<br/>+ processor_fingerprint + preprocess_kwargs)"]
    S3 --> S4["⑤ _get_cached_artifact 查 LRU<br/>key/digest 不一致视为脏条目并驱逐<br/>同请求内重复图片按 key 去重"]
    S4 --> HIT{"拿到可用 artifact？"}
    HIT -- 是 --> S8
    HIT -- 否 --> S5["⑥ lookup_or_claim_many 单飞占位<br/>CacheLookup → 命中<br/>should_compute=True → 本次计算<br/>should_compute=False → 等别人算"]
    S5 --> S6["⑦ _compute_cache_misses：后台 task + asyncio.shield<br/>decode_media_snapshot：bytes → PIL / tensor<br/>prepare_artifact_batch：HF image processor<br/>→ pixel_values + image_grid_thw<br/>→ feature_hash 由 artifact_key 派生<br/>validate_artifact → complete_miss"]
    S6 --> S7["⑧ 等被 join 的 miss，按原始输入顺序回填 artifacts"]
    S7 --> S9
    S8 --> S9["⑨ compose_image_artifacts<br/>占位符计数校验 → build_input_ids 展开<br/>→ MultimodalDataItem + pad_value<br/>→ padded_input_ids / M-RoPE"]
    S9 --> OK{"与 prompt 对齐成功？"}
    OK -- 是 --> OUT(["MultimodalProcessorOutput"])
    OK -- 否 --> UNC
```

逐步说明：

1. **参数对齐**：`content_hashes`、`featureless_hit_mask` 长度必须与媒体数一致；哈希统一走 `parse_content_hash`（只接受 `sha256:<64 hex>`，并转小写）。`featureless_hit_mask` 是给「只保留元数据、不留特征」的 artifact 开的绿灯——默认全 `False`，即本次请求必须拿到带特征的 artifact。
2. **可信 hash 快路径**：`trust_mm_content_hashes` 打开且调用方在 `image_url.content_hash` 里给了哈希时，直接用调用方 hash 算 key 查缓存；**命中就完全不读媒体**（不下载、不解码），这是最快的一条路。未命中则退回下一步重新实测。
3. **snapshot（这里是下载点）**：未命中的下标在 `io_executor` 线程池里并发 `snapshot_media_source` → `snapshot_media` → `_read_media_bytes`：`http(s)` 走 `get_image_bytes` → `download_remote_media`（`REQUEST_TIMEOUT`，默认 3s）；`file://` 读本地文件；`data:`/裸 base64 直接解码。得到 `MediaSnapshot(payload, content_digest, size_bytes)`。
   - **URL 本身不是身份**：路径/URL 每次都会被解析成字节再算 digest，远端内容变了自然 miss，不会复用错图。
   - 若调用方 hash 与实际 digest 不一致，直接报 `content hash mismatch`。
4. **建 artifact key**：`_artifact_key` = `sha256(canonical_json{content_digest, modality, processor_fingerprint, preprocess_kwargs})`。`processor_fingerprint` 覆盖 transformers 版本、processor 类名、`model_type`/`architectures`、模型 revision、`disable_fast_image_processor`、`mm_process_config` 以及 processor 自报的 `preprocess_fingerprint_payload`；`preprocess_kwargs` 把 `detail` 等会改变产物的请求选项纳入 key（未知选项也纳入，宁可 miss 不可误命中）。
5. **查缓存 + 去重**：`_get_cached_artifact` 在锁内比对 `artifact_key` 与 `content_digest`，不一致的条目按脏数据驱逐；`featureless` 条目若本次请求不能用，就留给 miss 路径处理。同一个请求里重复出现的同一张图按 key 去重，只预处理一次。
6. **单飞占位**：`lookup_or_claim_many` 对每个 unique key 返回 `CacheLookup`（命中）或 `CacheMiss`；并发请求打同一个 key 时，只有第一个 `should_compute=True`，其余 `should_compute=False` 去 `wait_for_miss` 等结果（`singleflight_joins` 计数）。
7. **计算 miss**：`create_background_task(_compute_cache_misses)` + `asyncio.shield`，所以本请求被取消也不会打断别的请求正在等的共享计算。内部两步：
   - `decode_media_snapshot`（IO 线程池）：bytes → PIL / numpy / tensor；这里再调 `load_image` 只是**解码已下载的字节**，不会重新下载。
   - `_run_preprocess_and_build_artifact_batch`：有 `mm_processor_executor` 就丢进 executor，否则本地执行。Qwen 侧就是 `prepare_artifact_batch`：`processor.image_processor(images=[...], return_tensors="pt")` 批处理 → `pixel_values` + `image_grid_thw`，再按网格长度把打包特征切成每图一份，`feature_hash = resolve_multimodal_item_hash(namespace=artifact_key)`，顺序、digest、key 都要与入参一致（`validate_artifact`）。
   - `complete_miss(artifact, cache_value=artifact.cache_value())`：等待者拿到完整 artifact，缓存里只留可缓存形态（必要时丢掉 CUDA 特征，只留元数据）。
8. **收尾**：等被 join 的 miss 全部返回，按原始处理器输入顺序回填；任一未解析即 `RuntimeError`（保证顺序与图片一一对应）。
9. **compose**：`compose_image_artifacts(input_text, artifacts)` 把「与 prompt 无关的产物」拼成当前请求的值：
   - 每图一个 grid，拼成 `image_grid_thw`；
   - `_normalize_prompt_for_fast_path` 用正则把占位符统一成原生 image token 并计数，数量与 artifact 数不等 → 返回 `None`；
   - `_cached_image_prompt_template`（`lru_cache(256)`，key = prompt + grid 形状）做 `build_input_ids`，展开占位符得到 `input_ids` / `offsets`，同时算 M-RoPE；
   - 组 `MultimodalDataItem`（`feature=artifact.feature`、`offsets`、`set_hash(feature_hash)`）→ `build_padded_input_ids` → `_prepare_mm_items_for_transport`（CPU / CUDA IPC / 共享内存）→ `MultimodalProcessorOutput`；
   - 任何一步对不上（占位符数不符、grid 非法），返回 `None` 并由调用方回退 `_process_mm_data_uncached`。

三个 hash 的分工，别混：

| 标识 | 覆盖范围 | 用途 |
|---|---|---|
| `content_digest` | 媒体字节的 sha256 | 媒体内容身份；与调用方 `content_hash` 比对 |
| `artifact_key` | digest + modality + processor fingerprint + 预处理选项 | 预处理器 LRU 的 key；变了就必然重算 |
| `feature_hash` | 由 `artifact_key` 派生 | `MultimodalDataItem.hash` → pad_value / ViT embedding cache / RadixCache 的 key |

要点：

- **artifact 与 prompt 解耦**：artifact 只装「模型相关、与文本无关」的产物（尺寸、token 数、encoder grid、可缓存的像素特征），**不含 prompt token 与 offsets**；这些在 compose 阶段才生成，所以同一张图配不同文本也能复用 artifact。
- **命中路径省掉的正是最贵的三件事**：下载、解码、HF image processor；剩下的只有 prompt 展开 + M-RoPE，开销与图片数量线性但很小。
- **失败不致命**：snapshot/decode/预处理异常会通过 `fail_miss` 传播给所有等待者；compose 对不齐时静默回退 uncached 路径，功能不受影响。

---

## 3. 阶段 ③：Scheduler 接收请求并构造 `Req`

```mermaid
flowchart TD
    L(["event_loop_normal / event_loop_overlap"]) --> ING["ingest_requests<br/>RequestReceiver.recv_requests：从 ZMQ 拉取<br/>（TP>1 时在 rank 间广播）"]
    ING --> PIR["process_input_requests"]
    PIR --> VMM["_materialize_cuda_vmm_inputs<br/>还原 CUDA VMM 传来的特征"]
    VMM --> DISP["request_dispatcher：<br/>TokenizedGenerateReqInput → handle_generate_request"]
    DISP --> REQ["_handle_generate_request<br/>new Req(rid, origin_input_ids, sampling_params, ...)"]
    REQ --> GMM["_get_multimodal_inputs<br/>MultimodalInputs.from_processor_output<br/>· CUDA IPC 特征 reconstruct 到本 GPU<br/>· 补齐缺失的 set_pad_value<br/>（多 rank 时可只算一次再广播：_process_and_broadcast_mm_inputs）"]
    GMM --> ERR{"多模态处理出错？"}
    ERR -- 是 --> ABORT["req.set_finish_with_abort(500)<br/>仍入队，由后续流程返回错误"]
    ERR -- 否 --> PAD{"processor 已给出<br/>padded_input_ids 且长度一致？"}
    PAD -- 是 --> RPL["_try_apply_padded_mm_input_ids<br/>origin_input_ids ← padded_input_ids"]
    PAD -- 否 --> PIF["duplicate_pad_mm_input_ids<br/>调用模型的 pad_input_ids（如 pad_mm_tokens）<br/>把图片 token 区间替换为 pad_value"]
    RPL --> EXT["req.extend_image_inputs(image_inputs)<br/>req.multimodal_inputs = MultimodalInputs"]
    PIF --> EXT
    EXT --> MR["_maybe_compute_mrope_positions<br/>（processor 未给出时在此补算）"]
    MR --> LEN{"展开后长度 ≥ max_req_input_len？"}
    LEN -- 是 --> ABORT2["abort：多模态展开后过长"]
    LEN -- 否 --> CHK["init_req_max_new_tokens<br/>validate_input_length<br/>logprob_start_len 等参数校验"]
    CHK --> GRAM{"需要结构化输出 grammar？"}
    GRAM -- 是 --> GQ["grammar_manager 队列<br/>（编译完成后再入 waiting_queue）"]
    GRAM -- 否 --> Q["_add_request_to_queue<br/>· 优先级校验 / 排队上限<br/>· _prefetch_kvcache（HiCache）<br/>· waiting_queue.append(req)"]
```

此时 `req.origin_input_ids` 形如：

```
[<|im_start|>, user, \n, <|vision_start|>, PAD₀, PAD₁, PAD₂, PAD₃, PAD₀, ... (256 个), <|vision_end|>, 图, 里, 有, 什么, ？, <|im_end|>, \n, <|im_start|>, assistant, \n]
                                            └─────── 由该图片 hash 派生的 pad_value 序列 ───────┘
```

---

## 4. 阶段 ④：组 Prefill 批并下发到 ModelRunner

```mermaid
flowchart TD
    NB(["get_next_batch_to_run"]) --> GNP["get_new_batch_prefill → _get_new_batch_prefill_raw"]
    GNP --> RDY["grammar 就绪的请求补入 waiting_queue"]
    RDY --> EMP{"running_batch 已满<br/>或 waiting_queue 为空？"}
    EMP -- 是 --> NONE(["本轮不做 Prefill"])
    EMP -- 否 --> PRI["policy.calc_priority<br/>（LPM / FCFS / DFS-weight 等，<br/>LPM 会基于 radix 前缀匹配长度排序）"]
    PRI --> ADD["创建 PrefillAdder<br/>（token 预算、chunked_prefill_size、KV 余量）"]
    ADD --> LOOP["遍历 waiting_queue"]
    LOOP --> INR["req.init_next_round_input(tree_cache)<br/>用 fill_ids（含 pad_value）在 RadixCache 做 match_prefix<br/>→ prefix_indices / extend_input_len"]
    INR --> ONE["adder.add_one_req(req)<br/>预算不够时可截断为 chunked prefill"]
    ONE --> MORE{"还能加？"}
    MORE -- 是 --> LOOP
    MORE -- 否 --> SB["ScheduleBatch.init_new(can_run_list, ...)"]
    SB --> PFE["prepare_for_extend<br/>· 分配 req_to_token / KV slot<br/>· 拼接 input_ids（只含未命中前缀的部分）<br/>· 收集每个 req 的 multimodal_inputs<br/>· 设置 extend_prefix_lens / extend_seq_lens"]
    PFE --> MIX{"mixed chunk？"}
    MIX -- 是 --> MW["与 running decode 批合并"]
    MIX -- 否 --> RB
    MW --> RB["run_batch(batch)"]
    RB --> MWB["batch.get_model_worker_batch()<br/>→ ModelWorkerBatch(multimodal_inputs, ...)"]
    MWB --> FBG["tp_worker.forward_batch_generation"]
    FBG --> FB["ForwardBatch.init_new<br/>· mm_inputs = batch.multimodal_inputs<br/>· model_is_mrope：_compute_mrope_positions<br/>  按 extend 区间切出 (3, L) 位置"]
    FB --> MRF["model_runner.forward → _forward_raw<br/>forward_mode = EXTEND → forward_extend<br/>初始化 attention backend 元数据"]
    MRF --> MF(["model.forward(input_ids, positions, forward_batch)"])
```

要点：

- RadixCache 的 key 是**替换成 pad_value 后的 token 序列**，因此"同一张图 + 同样前缀"的第二次请求可以直接命中，跳过这部分 Prefill（也跳过 ViT）。
- 若图片被 chunked prefill 切开，`extend_prefix_lens / extend_seq_lens` 会传到下一阶段，用于只取出当前 chunk 覆盖到的那段视觉 embedding。

---

## 5. 阶段 ⑤：模型 forward 入口 —— 视觉编码与 embedding 拼接

这是"Prefill 正式执行"之前的最后一步：先把图片变成 embedding，塞进文本 embedding 序列里，然后才交给语言模型做 Prefill。

```mermaid
flowchart TD
    MF(["Qwen2_5_VLForConditionalGeneration.forward"]) --> POS["positions = forward_batch.mrope_positions"]
    POS --> GMR["general_mm_embed_routine"]
    GMR --> PMI["prepare_mm_inputs<br/>（仅 extend 且 contains_mm_inputs 时处理多模态）"]
    PMI --> EMI["embed_mm_inputs<br/>（EPD 自适应派发时走 _embed_mm_inputs_with_split）"]
    EMI --> MOD["按模态分组；IMAGE → embedder = get_image_feature"]
    MOD --> PH["placeholder_tensor = 本批所有图片的 pad_values（排序）"]
    PH --> GEM["get_embedding_and_mask"]
    GEM --> PRE{"预计算 embedding？<br/>（EPD Encoder 送来）"}
    PRE -- 是 --> USE["直接使用"]
    PRE -- 否 --> CHK["_get_chunked_prefill_embedding<br/>· 按 item.hash 查 MultiModalStaticCache<br/>· 未命中 → get_image_feature(items)<br/>  = ViT(pixel_values, grid_thw)<br/>· 写回 embedding cache<br/>· 按 prefix/extend 区间切出本 chunk 部分"]
    USE --> MSK
    CHK --> MSK["_get_multimodal_mask<br/>input_ids ∈ pad_values 的位置 → mask"]
    MSK --> TE["input_ids.clamp_(0, vocab-1)<br/>input_embeds = embed_tokens(input_ids)"]
    TE --> SC["_scatter_mm_embedding<br/>把视觉 embedding 覆盖到 mask 位置"]
    SC --> OFL["GPU 特征 offload 回 CPU<br/>（chunked prefill 后续 chunk 缓存未命中时兜底）"]
    OFL --> LM(["language_model(input_ids=None,<br/>input_embeds=..., positions=mrope_positions)<br/>━━ 语言模型 Prefill 正式开始 ━━"])
```

要点：

- pad_value 远超词表大小，所以查文本 embedding 前要先 `clamp`，这些位置的结果随后被视觉 embedding 覆盖。
- ViT 输出按图片 hash 缓存在 `MultiModalStaticCache`，chunked prefill 的后续 chunk 或相同图片的新请求都能复用。
- 至此得到 `input_embeds` + `mrope_positions`，进入语言模型各层的 attention / MLP，即真正的 Prefill。

---

## 6. 端到端时序图

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant H as HTTP Server<br/>(OpenAIServingChat)
    participant T as TokenizerManager
    participant M as MM Processor<br/>(IO 线程池 / executor)
    participant S as Scheduler
    participant R as ModelRunner / Model

    C->>H: POST /v1/chat/completions (image_url)
    H->>H: 校验 → 渲染 chat template<br/>收集 ImageData，content→{type:image}
    H->>T: generate_request(GenerateReqInput(text, image_data))
    T->>M: process_mm_data_async(image_data, prompt)
    M->>M: 下载 + 解码图片
    M->>M: HF Processor：pixel_values / grid_thw<br/>展开 <|image_pad|>
    M->>M: offsets + hash → pad_value，M-RoPE 位置
    M-->>T: MultimodalProcessorOutput
    T->>T: 长度校验，构造 TokenizedGenerateReqInput
    T->>S: ZMQ 发送（特征走 SHM / CUDA IPC）
    S->>S: from_processor_output，pad input_ids<br/>new Req → waiting_queue
    S->>S: calc_priority → match_prefix(含 pad_value)<br/>PrefillAdder → ScheduleBatch.prepare_for_extend
    S->>R: forward_batch_generation(ForwardBatch EXTEND)
    R->>R: ViT 编码（查 embedding cache）
    R->>R: 文本 embedding + 视觉 embedding scatter
    Note over R: ↓ 语言模型 Prefill 正式执行（本文终点）
```

---

## 7. 代码索引

| 环节 | 位置 |
|---|---|
| HTTP 路由 | `python/sglang/srt/entrypoints/http_server.py:1849` |
| 通用请求处理骨架 | `python/sglang/srt/entrypoints/openai/serving_base.py:80` `handle_request` |
| 转内部请求 | `python/sglang/srt/entrypoints/openai/serving_chat.py:1240` `_convert_to_internal_request` |
| 消息处理 / 模板分派 | `serving_chat.py:1470` `_process_messages` |
| Jinja 模板路径 | `serving_chat.py:1654` `_apply_jinja_template` |
| Conversation 模板路径 | `serving_chat.py:1975` `_apply_conversation_template`；`parser/conversation.py:608` `generate_chat_conv` |
| image_url 拆解 | `python/sglang/srt/parser/jinja_template_utils.py:183` `process_content_for_template_format` |
| 多模态传 text 的决策 | `serving_chat.py` `_engine_prompt` |
| `ImageData` 结构 | `python/sglang/srt/utils/common.py:2001` |
| TokenizerManager 入口 | `python/sglang/srt/managers/tokenizer_manager.py:895` `generate_request` |
| 分词 + MM 处理 | `tokenizer_manager.py:1096` `_tokenize_one_request` |
| 构造 Tokenized 对象 | `tokenizer_manager.py:1533` `_create_tokenized_object` |
| 发往 Scheduler | `tokenizer_manager.py:1769` `_send_one_request` |
| Qwen-VL processor | `python/sglang/srt/multimodal/processors/qwen_vl.py:1175` `process_mm_data_async`、`:1005` `_process_mm_data_uncached`、`:782` `prepare_artifact_batch`、`:864` `compose_image_artifacts` |
| artifact 缓存编排 | `multimodal/media_artifacts/base.py:259` `prepare_media_artifacts`、`:415` `prepare_media_artifacts_without_cache`、`:475` `_compute_cache_misses`、`:227` `_get_cached_artifact` |
| artifact 快照 / 解码 | `media_artifacts/base.py:165` `snapshot_media_source`、`:142` `decode_media_snapshot`；`multimodal/cache/identity.py:147` `snapshot_media`、`:130` `_read_media_bytes`、`:2182`（`utils/common.py`）`get_image_bytes` |
| 缓存 key 与指纹 | `multimodal/cache/identity.py:311` `build_artifact_key`、`:175` `media_preprocess_kwargs`、`:329` `resolve_multimodal_item_hash`、`:380` `build_processor_fingerprint` |
| LRU + 单飞 | `python/sglang/srt/multimodal/cache/preprocess_cache.py:121` `MultimodalPreprocessCache`、`:172` `get_if_present`、`:339` `lookup_or_claim_many`、`:390` `complete_miss`、`:432` `wait_for_miss` |
| 缓存开关 / 预算 | `arg_groups/fields/mm.py:99` `mm_preprocess_cache_size_mb`、`:111` `trust_mm_content_hashes`；`base_processor.py:286` 预算按 tokenizer worker 均分、`:353`（`qwen_vl.py`）`uses_media_artifacts_without_cache` |
| 媒体加载 | `multimodal/processors/base_processor.py:1239` `load_mm_data`、`:972` `_load_single_item`；`utils/common.py:2140` `load_image` |
| HF Processor 调用 | `base_processor.py:848` `process_mm_data` |
| 组装 mm_items / offsets | `base_processor.py:1805` `process_and_combine_mm_data`、`:1969` `_finalize_mm_items` |
| pad_value 与 hash | `python/sglang/srt/managers/schedule_batch.py:377` `MultimodalDataItem`、`:476` `set_pad_value`、`:684` `build_padded_input_ids` |
| Scheduler 主循环 | `python/sglang/srt/managers/scheduler.py:2021` `event_loop_normal` |
| 收请求 | `scheduler.py:2179` `ingest_requests`、`:2219` `process_input_requests` |
| 构造 Req + 多模态 | `scheduler.py:2938` `handle_generate_request`、`:2831` `_get_multimodal_inputs`、`:2843` `_try_apply_padded_mm_input_ids` |
| 模型侧 pad 函数 | `python/sglang/srt/managers/mm_utils.py:241` `duplicate_pad_mm_input_ids`、`:313` `pad_mm_tokens` |
| 入队 | `scheduler.py:3475` `_add_request_to_queue` |
| 组 Prefill 批 | `scheduler.py:4047` `get_new_batch_prefill`、`:4074` `_get_new_batch_prefill_raw` |
| prepare_for_extend | `schedule_batch.py:2870` |
| ForwardBatch 构造 / M-RoPE | `python/sglang/srt/model_executor/forward_batch_info.py:958` `init_new` |
| ModelRunner forward | `python/sglang/srt/model_executor/model_runner.py:1813` |
| Qwen2.5-VL forward / ViT | `python/sglang/srt/models/qwen2_5_vl.py:845` `forward`、`:765` `get_image_feature` |
| embedding 拼接 | `mm_utils.py:841` `general_mm_embed_routine`、`:694` `prepare_mm_inputs`、`:443` `embed_mm_inputs` |
| embedding 缓存 / 切 chunk / mask | `python/sglang/srt/managers/mm_schedule.py:719` `get_embedding_and_mask`、`:541` `_get_chunked_prefill_embedding`、`:655` `_get_multimodal_mask` |
