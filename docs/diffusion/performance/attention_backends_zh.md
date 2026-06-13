# Attention 后端

本文档介绍 sglang diffusion（`sglang.multimodal_gen`）中可用的 attention 后端以及如何选择它们。

## 概述

Attention 后端由 `AttentionBackendEnum`（`sglang.multimodal_gen.runtime.platforms.interface.AttentionBackendEnum`）定义，并通过 CLI flag `--attention-backend` 选择。

后端选择由共享的 attention 层执行（例如 `sglang.multimodal_gen.runtime.layers.attention.layer` 中的 `LocalAttention` / `USPAttention` / `UlyssesAttention`），因此适用于任何使用这些层的模型组件（例如 diffusion transformer / DiT 和 encoder）。

使用 diffusers 后端时，`--attention-backend` 会被透传给 diffusers 的 `set_attention_backend`（例如 `flash`、`_flash_3_hub`、`sage`、`xformers`、`native`）。

- **CUDA**：在支持时优先选用 FlashAttention（FA3/FA4）；否则回退到 PyTorch SDPA。
- **ROCm**：在可用时使用 FlashAttention；否则回退到 PyTorch SDPA。
- **MPS**：始终使用 PyTorch SDPA。
- **NPU**：对于 ring attention 使用 FA，否则使用 PyTorch SDPA。

## 后端选项

对于 SGLang 原生 pipeline，CLI 接受 `AttentionBackendEnum` 的小写名称。下表列出了内置平台所实现的后端。`fa3`/`fa4` 被接受为 `fa` 的别名。

| CLI 值 | Enum 值 | 说明 |
|---|---|---|
| `fa` / `fa3` / `fa4` | `FA` | FlashAttention。`fa3/fa4` 在参数解析期间被归一化为 `fa`（`ServerArgs.__post_init__`）。 |
| `torch_sdpa` | `TORCH_SDPA` | PyTorch `scaled_dot_product_attention`。 |
| `sliding_tile_attn` | `SLIDING_TILE_ATTN` | Sliding Tile Attention (STA)。需要 `st_attn`。通过 `--attention-backend-config` 配置。 |
| `sage_attn` | `SAGE_ATTN` | 需要 `sageattention`。上游 SageAttention CUDA 扩展针对 SM80/SM86/SM89/SM90/SM120（compute capability 8.0/8.6/8.9/9.0/12.0）；参见上游 `setup.py`：https://github.com/thu-ml/SageAttention/blob/main/setup.py。 |
| `sage_attn_3` | `SAGE_ATTN_3` | 需要按上游说明安装 SageAttention3。 |
| `video_sparse_attn` | `VIDEO_SPARSE_ATTN` | 需要 `vsa`。通过 `--attention-backend-config` 配置 `sparsity`。 |
| `vmoba_attn` | `VMOBA_ATTN` | 需要 `kernel.attn.vmoba_attn.vmoba`。通过 `--attention-backend-config` 配置。 |
| `aiter` | `AITER` | 需要 `aiter`。 |
| `aiter_sage` | `AITER_SAGE` | 需要 `aiter`。 |
| `sparse_video_gen_2_attn` | `SPARSE_VIDEO_GEN_2_ATTN` | 需要 `svg`。安装说明参见 https://github.com/svg-project/Sparse-VideoGen。 |

## 选择优先级

`runtime/layers/attention/selector.py` 中的选择顺序为：

1. `global_force_attn_backend(...)` / `global_force_attn_backend_context_manager(...)`
2. CLI `--attention-backend`（`ServerArgs.attention_backend`）
3. 自动选择（平台能力、dtype 和已安装的软件包）

## 配置

某些后端需要额外的配置。你可以通过 `--attention-backend-config` 传入这些参数。该参数接受：
- 指向 JSON 或 YAML 配置文件的路径。
- 一个 JSON 字符串（例如 `'{"sparsity": 0.5}'`）。
- 键值对（例如 `"sparsity=0.5,enable_x=true"`）。

### 支持的配置参数

**Sliding Tile Attention (`sliding_tile_attn`)**

| 参数 | 类型 | 描述 | 默认值 |
| :--- | :--- | :--- | :--- |
| `mask_strategy_file_path` | `str` | **必填。** mask 策略 JSON 文件的路径。 | - |
| `sta_mode` | `str` | STA 的模式。 | `STA_inference` |
| `skip_time_steps` | `int` | 在切换到 sparse attention 之前使用 full attention 的步数。 | `15` |

**Video Sparse Attention (`video_sparse_attn`)**

| 参数 | 类型 | 描述 | 默认值 |
| :--- | :--- | :--- | :--- |
| `sparsity` | `float` | 验证稀疏度（0.0 - 1.0）。 | `0.0` |

**V-MoBA (`vmoba_attn`)**

| 参数 | 类型 | 描述 | 默认值 |
| :--- | :--- | :--- | :--- |
| `temporal_chunk_size` | `int` | 时间维度的 chunk 大小。 | - |
| `temporal_topk` | `int` | 在时间维度中选择的 Top-K token。 | - |
| `spatial_chunk_size` | `list[int]` | 空间维度（H, W）的 chunk 大小。 | - |
| `spatial_topk` | `int` | 在空间维度中选择的 Top-K token。 | - |
| `st_chunk_size` | `list[int]` | 时空维度（T, H, W）的 chunk 大小。 | - |
| `st_topk` | `int` | 在时空维度中选择的 Top-K token。 | - |
| `moba_select_mode` | `str` | 选择模式（例如 `threshold`）。 | `threshold` |
| `moba_threshold` | `float` | 选择的阈值。 | `0.25` |
| `moba_threshold_type` | `str` | 阈值类型（例如 `query_head`）。 | `query_head` |
| `first_full_step` | `int` | 使用 full attention 的初始步数。 | `12` |
| `first_full_layer` | `int` | 使用 full attention 的初始层数。 | `0` |
| `temporal_layer` | `int` | 时间层的数量。 | `1` |
| `spatial_layer` | `int` | 空间层的数量。 | `1` |
| `st_layer` | `int` | 时空层的数量。 | `1` |

## 平台支持矩阵

| 后端 | CUDA | ROCm | MPS | NPU | 说明 |
|---|---:|---:|---:|---:|---|
| `fa` | ✅ | ✅ | ❌ | ✅ | CUDA 需要 SM80+ 和 fp16/bf16。仅在已安装所需运行时的情况下才使用 FlashAttention；否则回退到 `torch_sdpa`。NPU 无需额外安装 |
| `torch_sdpa` | ✅ | ✅ | ✅ | ✅ | 跨平台兼容性最好的选项。 |
| `sliding_tile_attn` | ✅ | ❌ | ❌ | ❌ | 仅限 CUDA。需要 `st_attn`。通过 `--attention-backend-config` 配置。 |
| `sage_attn` | ✅ | ❌ | ❌ | ❌ | 仅限 CUDA（可选依赖）。 |
| `sage_attn_3` | ✅ | ❌ | ❌ | ❌ | 仅限 CUDA（可选依赖）。 |
| `video_sparse_attn` | ✅ | ❌ | ❌ | ❌ | 仅限 CUDA。需要 `vsa`。通过 `--attention-backend-config` 配置 `sparsity`。 |
| `vmoba_attn` | ✅ | ❌ | ❌ | ❌ | 仅限 CUDA。需要 `kernel.attn.vmoba_attn.vmoba`。通过 `--attention-backend-config` 配置。 |
| `aiter` | ❌ | ✅ | ❌ | ❌ | 需要 `aiter`。 |
| `aiter_sage` | ❌ | ✅ | ❌ | ❌ | 需要 `aiter`。 |
| `sparse_video_gen_2_attn` | ✅ | ❌ | ❌ | ❌ | 仅限 CUDA。需要 `svg`。 |

## 用法

### 通过 CLI 选择后端

```bash
sglang generate \
  --model-path <MODEL_PATH_OR_ID> \
  --prompt "..." \
  --attention-backend fa
```

```bash
sglang generate \
  --model-path <MODEL_PATH_OR_ID> \
  --prompt "..." \
  --attention-backend torch_sdpa
```

### 使用 Sliding Tile Attention (STA)

```bash
# Pass the mask strategy file path via config
sglang generate \
  --model-path <MODEL_PATH_OR_ID> \
  --prompt "..." \
  --attention-backend sliding_tile_attn \
  --attention-backend-config "mask_strategy_file_path=/abs/path/to/mask_strategy.json"
```

### ROCm / MPS 注意事项

- ROCm：根据你环境中可用的内容，使用 `--attention-backend torch_sdpa` 或 `fa`。
- MPS：该平台实现始终使用 `torch_sdpa`。
