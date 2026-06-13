# SGLang Diffusion CLI

使用 CLI 通过 `sglang generate` 进行一次性生成,或通过 `sglang serve` 启动一个持久化的 HTTP 服务端。

### 针对非 diffusers 模型的 Overlay 仓库

如果 `--model-path` 指向一个受支持的非 diffusers 源仓库,SGLang 可以通过一个自托管的 overlay 仓库来解析它。

SGLang 会首先检查一个内置的 overlay 注册表。具体的内置映射可以随着时间推移逐步添加,而无需更改 CLI 接口。

覆盖示例:

```bash
export SGLANG_DIFFUSION_MODEL_OVERLAY_REGISTRY='{
  "Wan-AI/Wan2.2-S2V-14B": {
    "overlay_repo_id": "your-org/Wan2.2-S2V-14B-overlay",
    "overlay_revision": "main"
  }
}'

sglang generate \
  --model-path Wan-AI/Wan2.2-S2V-14B \
  --config configs/wan_s2v.yaml
```

overlay 仓库应该是一个完整的 diffusers 风格/组件化仓库

如果 overlay 仓库本身包含 `_overlay/overlay_manifest.json`,你也可以将其作为 `--model-path` 传入。

注意事项:
1. `SGLANG_DIFFUSION_MODEL_OVERLAY_REGISTRY` 仅是一个用于开发和调试的可选覆盖项。它接受一个 JSON 对象或一个指向 JSON 文件的路径,并且可以在当前进程中扩展或替换内置条目。
2. 首次加载时,SGLang 会:
   - 从 overlay 仓库下载 overlay 元数据
   - 从原始源仓库下载所需的文件
   - 在 `~/.cache/sgl_diffusion/materialized_models/` 下物化一个本地标准组件仓库
3. 后续加载会复用已物化的本地仓库。该物化仓库就是运行时作为普通组件化模型目录加载的内容。


## 快速开始

### Generate

```bash
sglang generate \
  --model-path Qwen/Qwen-Image \
  --prompt "A beautiful sunset over the mountains" \
  --save-output
```

### Serve

```bash
sglang serve \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --num-gpus 4 \
  --ulysses-degree 2 \
  --ring-degree 2 \
  --port 30010
```

有关请求和响应示例,请参见 [OpenAI-Compatible API](openai_api.md)。

```{tip}
使用 `sglang generate --help` 和 `sglang serve --help` 查看完整的参数列表。CLI 的帮助输出是详尽 flag 的权威来源。
```

## 常用选项

### 模型与运行时

- `--model-path {MODEL}`:模型路径或 Hugging Face model ID
- `--lora-path {PATH}` 和 `--lora-nickname {NAME}`:加载一个 LoRA adapter
- `--num-gpus {N}`:要使用的 GPU 数量
- `--tp-size {N}`:张量并行大小,主要用于 encoder
- `--sp-degree {N}`:序列并行大小
- `--ulysses-degree {N}` 和 `--ring-degree {N}`:USP 并行控制
- `--attention-backend {BACKEND}`:原生 SGLang pipeline 的 attention 后端
- `--attention-backend-config {CONFIG}`:attention 后端配置

### 采样与输出

- `--prompt {PROMPT}` 和 `--negative-prompt {PROMPT}`
- `--num-inference-steps {STEPS}` 和 `--seed {SEED}`
- `--height {HEIGHT}`、`--width {WIDTH}`、`--num-frames {N}`、`--fps {FPS}`
- `--output-path {PATH}`、`--output-file-name {NAME}`、`--save-output`、`--return-frames`

有关帧插值和超分辨率,请参见 [Post-Processing](post_processing.md)。

### 量化 transformer

对于量化后的 transformer 检查点,优先使用:

- `--model-path` 用于基础 pipeline
- `--transformer-path` 用于一个量化后的 `transformers` transformer 组件文件夹
- `--transformer-weights-path` 用于一个量化后的 safetensors 文件、目录或 repo

有关支持的量化系列和示例,请参见 [Quantization](../quantization.md)。

## 配置文件

使用 `--config` 加载 JSON 或 YAML 配置。命令行 flag 会覆盖配置文件中的值。

```bash
sglang generate --config config.yaml
```

示例:

```yaml
model_path: FastVideo/FastHunyuan-diffusers
prompt: A beautiful woman in a red dress walking down a street
output_path: outputs/
num_gpus: 2
sp_size: 2
tp_size: 1
num_frames: 45
height: 720
width: 1280
num_inference_steps: 6
seed: 1024
fps: 24
precision: bf16
vae_precision: fp16
vae_tiling: true
vae_sp: true
enable_torch_compile: false
```

## Generate

`sglang generate` 运行单个生成任务,并在任务完成时退出。

```bash
sglang generate \
  --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --text-encoder-cpu-offload \
  --pin-cpu-memory \
  --num-gpus 4 \
  --ulysses-degree 2 \
  --ring-degree 2 \
  --prompt "A curious raccoon" \
  --save-output \
  --output-path outputs \
  --output-file-name "a-curious-raccoon.mp4"
```

```{note}
仅用于 HTTP 服务端的参数会被 `sglang generate` 忽略。
```

对于 diffusers pipeline,可以通过 `SGLANG_CACHE_DIT_ENABLED=true` 或 `--cache-dit-config` 启用 Cache-DiT。参见 [Cache-DiT](../performance/cache/cache_dit.md)。

## Serve

`sglang serve` 启动 HTTP 服务端,并保持模型加载状态以处理重复请求。

```bash
sglang serve \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --text-encoder-cpu-offload \
  --pin-cpu-memory \
  --num-gpus 4 \
  --ulysses-degree 2 \
  --ring-degree 2 \
  --port 30010
```

### 云存储

SGLang Diffusion 可以在生成完成后将生成的图像和视频上传到兼容 S3 的对象存储。

```bash
export SGLANG_CLOUD_STORAGE_TYPE=s3
export SGLANG_S3_BUCKET_NAME=my-bucket
export SGLANG_S3_ACCESS_KEY_ID=your-access-key
export SGLANG_S3_SECRET_ACCESS_KEY=your-secret-key
export SGLANG_S3_ENDPOINT_URL=https://minio.example.com
```

有关完整的存储选项集,请参见 [Environment Variables](../environment_variables.md)。

## 组件路径覆盖

使用 `--<component>-path` 覆盖单个 pipeline 组件,例如 `vae`、`transformer` 或 `text_encoder`。

```bash
sglang serve \
  --model-path black-forest-labs/FLUX.2-dev \
  --vae-path fal/FLUX.2-Tiny-AutoEncoder
```

组件键必须与模型 `model_index.json` 中的键匹配,且该路径必须是 Hugging Face repo ID 或一个完整的组件目录。

## Diffusers 后端

使用 `--backend diffusers` 在不存在原生 SGLang 实现时,或当某个模型需要自定义 pipeline 类时,强制使用原生 diffusers pipeline。

### 关键选项

| Argument | Values | Description |
|----------|--------|-------------|
| `--backend` | `auto`, `sglang`, `diffusers` | 选择原生 SGLang、强制原生,或强制 diffusers |
| `--diffusers-attention-backend` | `flash`, `_flash_3_hub`, `sage`, `xformers`, `native` | diffusers pipeline 的 attention 后端 |
| `--trust-remote-code` | flag | 对于带有自定义 pipeline 类的模型是必需的 |
| `--vae-tiling` and `--vae-slicing` | flag | 降低 VAE 解码的内存占用 |
| `--dit-precision` and `--vae-precision` | `fp16`, `bf16`, `fp32` | 精度控制 |
| `--enable-torch-compile` | flag | 启用 `torch.compile` |
| `--cache-dit-config` | `{PATH}` | diffusers pipeline 的 Cache-DiT 配置 |

### 示例

```bash
sglang generate \
  --model-path AIDC-AI/Ovis-Image-7B \
  --backend diffusers \
  --trust-remote-code \
  --diffusers-attention-backend flash \
  --prompt "A serene Japanese garden with cherry blossoms" \
  --height 1024 \
  --width 1024 \
  --num-inference-steps 30 \
  --save-output \
  --output-path outputs \
  --output-file-name ovis_garden.png
```

对于未在 CLI 中暴露的 pipeline 专用参数,请在配置文件中传入 `diffusers_kwargs`。
