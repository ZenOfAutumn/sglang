# SGLang Diffusion OpenAI API

SGLang diffusion HTTP 服务端实现了一个 OpenAI 兼容的 API,用于图像和视频生成,以及 LoRA adapter 管理。

## 前置条件

- 如果你计划使用 OpenAI Python SDK,需要 Python 3.11+。

## Serve

使用 `sglang serve` 命令启动服务端。

### 启动服务端

```bash
SERVER_ARGS=(
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers
  --text-encoder-cpu-offload
  --pin-cpu-memory
  --num-gpus 4
  --ulysses-degree=2
  --ring-degree=2
  --port 30010
)

sglang serve "${SERVER_ARGS[@]}"
```

- **--model-path**:模型路径或 model ID。
- **--port**:要监听的 HTTP 端口(默认:`30000`)。

**获取模型信息**

**Endpoint:** `GET /models`

返回此服务端所服务模型的信息,包括模型路径、任务类型、pipeline 配置和精度设置。

**Curl 示例:**

```bash
curl -sS -X GET "http://localhost:30010/models"
```

**响应示例:**

```json
{
  "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "task_type": "T2V",
  "pipeline_name": "wan_pipeline",
  "pipeline_class": "WanPipeline",
  "num_gpus": 4,
  "dit_precision": "bf16",
  "vae_precision": "fp16"
}
```

---

## 端点

### 图像生成

服务端在 `/v1/images` 命名空间下实现了一个 OpenAI 兼容的 Images API。

**创建图像**

**Endpoint:** `POST /v1/images/generations`

**Python 示例(b64_json 响应):**

```python
import base64
from openai import OpenAI

client = OpenAI(api_key="sk-proj-1234567890", base_url="http://localhost:30010/v1")

img = client.images.generate(
    prompt="A calico cat playing a piano on stage",
    size="1024x1024",
    n=1,
    response_format="b64_json",
)

image_bytes = base64.b64decode(img.data[0].b64_json)
with open("output.png", "wb") as f:
    f.write(image_bytes)
```

**Curl 示例:**

```bash
curl -sS -X POST "http://localhost:30010/v1/images/generations" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-proj-1234567890" \
  -d '{
        "prompt": "A calico cat playing a piano on stage",
        "size": "1024x1024",
        "n": 1,
        "response_format": "b64_json"
      }'
```

> **注意**
> 如果使用 `response_format=url` 但未配置云存储,API 会返回一个相对 URL,例如 `/v1/images/<IMAGE_ID>/content`。

**编辑图像**

**Endpoint:** `POST /v1/images/edits`

此端点接受一个 multipart form 上传,包含输入图像和一个文本 prompt。服务端可以返回 base64 编码的图像或一个用于下载图像的 URL。

**Curl 示例(b64_json 响应):**

```bash
curl -sS -X POST "http://localhost:30010/v1/images/edits" \
  -H "Authorization: Bearer sk-proj-1234567890" \
  -F "image=@local_input_image.png" \
  -F "url=image_url.jpg" \
  -F "prompt=A calico cat playing a piano on stage" \
  -F "size=1024x1024" \
  -F "response_format=b64_json"
```

**Curl 示例(URL 响应):**

```bash
curl -sS -X POST "http://localhost:30010/v1/images/edits" \
  -H "Authorization: Bearer sk-proj-1234567890" \
  -F "image=@local_input_image.png" \
  -F "url=image_url.jpg" \
  -F "prompt=A calico cat playing a piano on stage" \
  -F "size=1024x1024" \
  -F "response_format=url"
```

**下载图像内容**

当在 `POST /v1/images/generations` 或 `POST /v1/images/edits` 中使用 `response_format=url` 时,API 会返回一个相对 URL,例如 `/v1/images/<IMAGE_ID>/content`。

**Endpoint:** `GET /v1/images/{image_id}/content`

**Curl 示例:**

```bash
curl -sS -L "http://localhost:30010/v1/images/<IMAGE_ID>/content" \
  -H "Authorization: Bearer sk-proj-1234567890" \
  -o output.png
```

### 视频生成

服务端在 `/v1/videos` 命名空间下实现了 OpenAI Videos API 的一个子集。

**创建视频**

**Endpoint:** `POST /v1/videos`

**Python 示例:**

```python
from openai import OpenAI

client = OpenAI(api_key="sk-proj-1234567890", base_url="http://localhost:30010/v1")

video = client.videos.create(
    prompt="A calico cat playing a piano on stage",
    size="1280x720"
)
print(f"Video ID: {video.id}, Status: {video.status}")
```

**Curl 示例:**

```bash
curl -sS -X POST "http://localhost:30010/v1/videos" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-proj-1234567890" \
  -d '{
        "prompt": "A calico cat playing a piano on stage",
        "size": "1280x720"
      }'
```

**列出视频**

**Endpoint:** `GET /v1/videos`

**Python 示例:**

```python
videos = client.videos.list()
for item in videos.data:
    print(item.id, item.status)
```

**Curl 示例:**

```bash
curl -sS -X GET "http://localhost:30010/v1/videos" \
  -H "Authorization: Bearer sk-proj-1234567890"
```

**下载视频内容**

**Endpoint:** `GET /v1/videos/{video_id}/content`

**Python 示例:**

```python
import time

# Poll for completion
while True:
    page = client.videos.list()
    item = next((v for v in page.data if v.id == video_id), None)
    if item and item.status == "completed":
        break
    time.sleep(5)

# Download content
resp = client.videos.download_content(video_id=video_id)
with open("output.mp4", "wb") as f:
    f.write(resp.read())
```

**Curl 示例:**

```bash
curl -sS -L "http://localhost:30010/v1/videos/<VIDEO_ID>/content" \
  -H "Authorization: Bearer sk-proj-1234567890" \
  -o output.mp4
```

---

### LoRA 管理

服务端支持 LoRA adapter 的动态加载、merge 和 unmerge。

**重要提示:**
- 互斥:同一时间只能 *merge*(激活)一个 LoRA
- 切换:要切换 LoRA,你必须先 `unmerge` 当前的,然后再 `set` 新的
- 缓存:服务端会在内存中缓存已加载的 LoRA 权重。切换回先前已加载的 LoRA(相同路径)开销很小

**设置 LoRA Adapter**

加载一个或多个 LoRA adapter 并将其权重 merge 到模型中。同时支持单个 LoRA(向后兼容)和多个 LoRA adapter。

**Endpoint:** `POST /v1/set_lora`

**参数:**
- `lora_nickname`(string 或 string 列表,必需):LoRA adapter 的唯一标识符。可以是单个字符串,或者对于多个 LoRA 是一个字符串列表
- `lora_path`(string 或 string/None 列表,可选):`.safetensors` 文件或 Hugging Face repo ID 的路径。首次加载时必需;如果重新激活一个已缓存的 nickname 则为可选。如果是列表,长度必须与 `lora_nickname` 匹配
- `target`(string 或 string 列表,可选):将 LoRA 应用到哪个/哪些 transformer。如果是列表,长度必须与 `lora_nickname` 匹配。有效值:
  - `"all"`(默认):应用到所有 transformer
  - `"transformer"`:仅应用到主 transformer(Wan2.2 的 high noise)
  - `"transformer_2"`:仅应用到 transformer_2(Wan2.2 的 low noise)
  - `"critic"`:仅应用到 critic 模型
- `strength`(float 或 float 列表,可选):merge 时的 LoRA 强度,默认 1.0。如果是列表,长度必须与 `lora_nickname` 匹配。小于 1.0 的值会减弱效果,大于 1.0 的值会放大效果

**单个 LoRA 示例:**

```bash
curl -X POST http://localhost:30010/v1/set_lora \
  -H "Content-Type: application/json" \
  -d '{
        "lora_nickname": "lora_name",
        "lora_path": "/path/to/lora.safetensors",
        "target": "all",
        "strength": 0.8
      }'
```

**多个 LoRA 示例:**

```bash
curl -X POST http://localhost:30010/v1/set_lora \
  -H "Content-Type: application/json" \
  -d '{
        "lora_nickname": ["lora_1", "lora_2"],
        "lora_path": ["/path/to/lora1.safetensors", "/path/to/lora2.safetensors"],
        "target": ["transformer", "transformer_2"],
        "strength": [0.8, 1.0]
      }'
```

**多个 LoRA 应用到相同 Target:**

```bash
curl -X POST http://localhost:30010/v1/set_lora \
  -H "Content-Type: application/json" \
  -d '{
        "lora_nickname": ["style_lora", "character_lora"],
        "lora_path": ["/path/to/style.safetensors", "/path/to/character.safetensors"],
        "target": "all",
        "strength": [0.7, 0.9]
      }'
```

> [!NOTE]
> 使用多个 LoRA 时:
> - 所有列表参数(`lora_nickname`、`lora_path`、`target`、`strength`)必须具有相同的长度
> - 如果 `target` 或 `strength` 是单个值,它将应用到所有 LoRA
> - 应用到相同 target 的多个 LoRA 将按顺序 merge


**Merge LoRA 权重**

手动将当前设置的 LoRA 权重 merge 到基础模型中。

> [!NOTE]
> `set_lora` 会自动执行一次 merge,因此通常只有在你手动 unmerge 之后,想要重新应用相同的 LoRA 而不再次调用 `set_lora` 时,才需要这一步。*

**Endpoint:** `POST /v1/merge_lora_weights`

**参数:**
- `target`(string,可选):要 merge 哪个/哪些 transformer。"all"(默认)、"transformer"、"transformer_2"、"critic" 之一
- `strength`(float,可选):merge 时的 LoRA 强度,默认 1.0。小于 1.0 的值会减弱效果,大于 1.0 的值会放大效果

**Curl 示例:**

```bash
curl -X POST http://localhost:30010/v1/merge_lora_weights \
  -H "Content-Type: application/json" \
  -d '{"strength": 0.8}'
```


**Unmerge LoRA 权重**

将当前激活的 LoRA 权重从基础模型中 unmerge,将其恢复到原始状态。在设置不同的 LoRA 之前**必须**调用此操作。

**Endpoint:** `POST /v1/unmerge_lora_weights`

**Curl 示例:**

```bash
curl -X POST http://localhost:30010/v1/unmerge_lora_weights \
  -H "Content-Type: application/json"
```

**列出 LoRA Adapter**

返回已加载的 LoRA adapter 以及每个模块当前的应用状态。

**Endpoint:** `GET /v1/list_loras`

**Curl 示例:**

```bash
curl -sS -X GET "http://localhost:30010/v1/list_loras"
```

**响应示例:**

```json
{
  "loaded_adapters": [
    { "nickname": "lora_a", "path": "/weights/lora_a.safetensors" },
    { "nickname": "lora_b", "path": "/weights/lora_b.safetensors" }
  ],
  "active": {
    "transformer": [
      {
        "nickname": "lora2",
        "path": "tarn59/pixel_art_style_lora_z_image_turbo",
        "merged": true,
        "strength": 1.0
      }
    ]
  }
}
```

注意:
- 如果当前 pipeline 未启用 LoRA,服务端将返回一个错误。
- `num_lora_layers_with_weights` 只统计为激活 adapter 应用了 LoRA 权重的层。

### 示例:切换 LoRA

1.  设置 LoRA A:
    ```bash
    curl -X POST http://localhost:30010/v1/set_lora -d '{"lora_nickname": "lora_a", "lora_path": "path/to/A"}'
    ```
2.  使用 LoRA A 进行生成……
3.  Unmerge LoRA A:
    ```bash
    curl -X POST http://localhost:30010/v1/unmerge_lora_weights
    ```
4.  设置 LoRA B:
    ```bash
    curl -X POST http://localhost:30010/v1/set_lora -d '{"lora_nickname": "lora_b", "lora_path": "path/to/B"}'
    ```
5.  使用 LoRA B 进行生成……

### 调整输出质量

服务端支持通过 `output-quality` 和 `output-compression` 参数,为图像和视频生成调整输出质量和压缩级别。

#### 参数

- **`output-quality`**(string,可选):预设质量级别,会自动设置压缩。**默认为 `"default"`**。有效值:
  - `"maximum"`:最高质量(100)
  - `"high"`:高质量(90)
  - `"medium"`:中等质量(55)
  - `"low"`:较低质量(35)
  - `"default"`:根据媒体类型自动调整(视频为 50,图像为 75)

- **`output-compression`**(integer,可选):直接的压缩级别覆盖(0-100)。**默认为 `None`**。当提供时(非 `None`),优先于 `output-quality`。
  - `0`:最低质量,最小文件大小
  - `100`:最高质量,最大文件大小

#### 说明

- **优先级**:当同时提供 `output-quality` 和 `output-compression` 时,`output-compression` 优先
- **格式支持**:质量设置适用于 JPEG 和视频格式。PNG 使用无损压缩,会忽略这些设置
- **文件大小与质量**:较低的压缩值(或 "low" 质量预设)会产生较小的文件,但可能出现可见的伪影
