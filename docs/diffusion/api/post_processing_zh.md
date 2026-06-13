# 后处理

SGLang diffusion 支持可选的后处理步骤,这些步骤在生成之后运行,用于改善时间平滑度(帧插值)或空间分辨率(超分辨率)。这些步骤独立于 diffusion 模型,且可以在单次运行中组合使用。

当两者都启用时,**帧插值先运行**(增加帧数),然后**超分辨率在每一帧上运行**(增加空间分辨率)。

---

## 帧插值(仅视频)

帧插值在每一对连续生成帧之间合成新帧,在不重新运行 diffusion 模型的情况下产生更平滑的运动。

`--frame-interpolation-exp` flag 控制应用多少轮插值:每一轮都在相邻帧之间的每个间隙中插入一个新帧,因此输出帧数遵循以下公式:

> **(N − 1) × 2^exp + 1**
>
> 例如,5 个原始帧在 `exp=1` 时 → 4 个间隙 × 1 个新帧 + 5 个原始帧 = **9** 帧;
> 在 `exp=2` 时 → **17** 帧。

### CLI 参数

| Argument | Description |
|----------|-------------|
| `--enable-frame-interpolation` | 启用帧插值。模型权重在首次使用时自动下载。 |
| `--frame-interpolation-exp {EXP}` | 插值指数 —— `1` = 2× 时间分辨率,`2` = 4×,以此类推(默认:`1`) |
| `--frame-interpolation-scale {SCALE}` | RIFE 推理 scale;对于高分辨率输入使用 `0.5` 以节省内存(默认:`1.0`) |
| `--frame-interpolation-model-path {PATH}` | 包含 RIFE `flownet.pkl` 权重的本地目录或 HuggingFace repo ID(默认:`elfgum/RIFE-4.22.lite`,自动下载) |

### 支持的模型

帧插值使用 [RIFE](https://github.com/hzwer/Practical-RIFE)(Real-Time Intermediate Flow Estimation)架构。仅支持 **RIFE 4.22.lite**(带 4-scale `IFBlock` 骨干的 `IFNet`)。网络拓扑是硬编码的,因此通过 `--frame-interpolation-model-path` 提供的自定义权重必须是与此架构兼容的 `flownet.pkl` 检查点。

其他 RIFE 版本(例如,block 数量不同的较旧 `v4.x` 变体)或完全不同的帧插值方法(FILM、AMT 等)**不受支持**。

| Weight | HuggingFace Repo | Description |
|--------|------------------|-------------|
| RIFE 4.22.lite *(default)* | [`elfgum/RIFE-4.22.lite`](https://huggingface.co/elfgum/RIFE-4.22.lite) | 轻量级模型,首次使用时自动下载 |

### 示例

生成一个 5 帧的视频并插值到 9 帧((5 − 1) × 2¹ + 1 = 9):

```bash
sglang generate \
  --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt "A dog running through a park" \
  --num-frames 5 \
  --enable-frame-interpolation \
  --frame-interpolation-exp 1 \
  --save-output
```

---

## 超分辨率(图像和视频)

超分辨率使用 [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) 提升生成的图像或视频帧的空间分辨率。模型权重在首次使用时自动下载,并为后续运行缓存。

### CLI 参数

| Argument | Description |
|----------|-------------|
| `--enable-upscaling` | 使用 Real-ESRGAN 启用生成后的超分辨率。 |
| `--upscaling-scale {SCALE}` | 期望的超分辨率倍数(默认:`4`)。内部使用 4× 模型;如果请求了不同的 scale,会在网络输出之后应用一次双三次(bicubic)缩放。 |
| `--upscaling-model-path {PATH}` | Real-ESRGAN 权重的本地 `.pth` 文件、HuggingFace repo ID 或 `repo_id:filename`(默认:`ai-forever/Real-ESRGAN` 搭配 `RealESRGAN_x4.pth`,自动下载)。使用 `repo_id:filename` 格式可指定 HuggingFace repo 中的自定义权重文件(例如 `my-org/my-esrgan:weights.pth`)。 |

### 支持的模型

超分辨率支持两种 Real-ESRGAN 网络架构。正确的架构会从检查点的键中**自动检测**,因此你只需将 `--upscaling-model-path` 指向一个有效的 `.pth` 文件即可:

| Architecture | Example Weights | Description |
|--------------|-----------------|-------------|
| **RRDBNet** | `RealESRGAN_x4plus.pth` | 更重的模型,质量更高;最适合照片 |
| **SRVGGNetCompact** | `RealESRGAN_x4.pth` *(default)*, `realesr-animevideov3.pth`, `realesr-general-x4v3.pth` | 轻量级模型;推理更快,适合视频 |

默认权重文件是 [`ai-forever/Real-ESRGAN`](https://huggingface.co/ai-forever/Real-ESRGAN) 搭配 `RealESRGAN_x4.pth`(SRVGGNetCompact,4× 原生 scale)。

其他超分辨率模型(例如 SwinIR、HAT、BSRGAN)**不受支持** —— 仅兼容使用上述两种架构的 Real-ESRGAN 检查点。

### 示例

生成一张 1024×1024 的图像并超分辨率到 4096×4096:

```bash
sglang generate \
  --model-path black-forest-labs/FLUX.2-dev \
  --prompt "A cat sitting on a windowsill" \
  --output-size 1024x1024 \
  --enable-upscaling \
  --save-output
```

生成一个视频并将每一帧超分辨率 4×:

```bash
sglang generate \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --prompt "A curious raccoon" \
  --enable-upscaling \
  --upscaling-scale 4 \
  --save-output
```

---

## 组合帧插值与超分辨率

帧插值和超分辨率可以在单次运行中组合使用。先应用插值(增加帧数),然后对每一帧应用超分辨率(增加空间分辨率)。

示例 —— 生成 5 帧,插值到 9 帧,并将每一帧超分辨率 4×:

```bash
sglang generate \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --prompt "A curious raccoon" \
  --num-frames 5 \
  --enable-frame-interpolation \
  --frame-interpolation-exp 1 \
  --enable-upscaling \
  --upscaling-scale 4 \
  --save-output
```
