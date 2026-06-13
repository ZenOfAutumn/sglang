# 量化

SGLang-Diffusion 支持量化后的 transformer 检查点。在大多数情况下,请将基础模型与量化 transformer override 分开管理。

## 快速参考

使用以下路径:

- `--model-path`:基础模型或原始模型
- `--transformer-path`:一个量化后的 transformers 风格 transformer 组件目录,该目录本身已包含自己的 `config.json`
- `--transformer-weights-path`:以单个 safetensors 文件、分片 safetensors 目录、本地路径或 Hugging Face repo ID 形式提供的量化 transformer 权重

推荐示例:

```bash
sglang generate \
  --model-path black-forest-labs/FLUX.2-dev \
  --transformer-weights-path black-forest-labs/FLUX.2-dev-NVFP4 \
  --prompt "a curious pikachu"
```

对于量化后的 transformers 风格 transformer 组件文件夹:

```bash
sglang generate \
  --model-path /path/to/base-model \
  --transformer-path /path/to/quantized-transformer \
  --prompt "A Logo With Bold Large Text: SGL Diffusion"
```

注意:某些特定于模型的集成也接受直接将量化 repo 或本地目录作为 `--model-path`,但那是一种兼容性路径。如果一个 repo 包含多个候选检查点,请显式传入 `--transformer-weights-path`。

## 量化系列

这里,`quant_family` 指的是一个共享相同 CLI 用法和 loader 行为的检查点及加载系列。它不仅仅是数值精度或某个 kernel 后端。

| quant_family     | checkpoint form                                                                            | canonical CLI                                        | supported models                                             | extra dependency                      | platform / notes                                                                                                      |
|------------------|--------------------------------------------------------------------------------------------|------------------------------------------------------|--------------------------------------------------------------|---------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| `fp8`            | 量化后的 transformer 组件文件夹,或带有 `quantization_config` 元数据的 safetensors | `--transformer-path` or `--transformer-weights-path` | ALL                                                          | None                                  | 同时支持组件文件夹和单文件两种流程                                                             |
| `nvfp4-modelopt` | NVFP4 safetensors 文件、分片目录,或提供 transformer 权重的 repo           | `--transformer-weights-path`                         | FLUX.2                                                       | `comfy-kitchen` optional on Blackwell | Blackwell 在可用时可使用一套最佳性能工具包;否则 SGLang 会回退到通用的 ModelOpt FP4 路径 |
| `nunchaku-svdq`  | 预量化的 Nunchaku transformer 权重,通常命名为 `svdq-{int4\|fp4}_r{rank}-...`   | `--transformer-weights-path`                         | 特定于模型的支持,例如 Qwen-Image、FLUX 和 Z-Image | `nunchaku`                            | SGLang 可以从文件名推断精度和 rank,并同时支持 `int4` 和 `nvfp4`                            |
| `msmodelslim`    | 预量化的 msmodelslim transformer 权重                                              | `--model-path`                                       | Wan2.2 系列                                                | None                                  | 目前仅兼容 Ascend NPU 系列,并同时支持 `w8a8` 和 `w4a4`                              |

## NVFP4

### 用法示例

推荐用法是将基础模型与量化 transformer override 分开:

```bash
sglang generate \
  --model-path black-forest-labs/FLUX.2-dev \
  --transformer-weights-path black-forest-labs/FLUX.2-dev-NVFP4 \
  --prompt "A Logo With Bold Large Text: SGL Diffusion" \
  --save-output
```

SGLang 也支持直接将 NVFP4 repo 或本地目录作为 `--model-path` 传入:

```bash
sglang generate \
  --model-path black-forest-labs/FLUX.2-dev-NVFP4 \
  --prompt "A Logo With Bold Large Text: SGL Diffusion" \
  --save-output
```

### 说明

- 对于 NVFP4 transformer 检查点,`--transformer-weights-path` 仍然是规范的 CLI 用法。
- 直接使用 `--model-path` 加载是针对 FLUX.2 NVFP4 风格 repo 或本地目录的一种兼容性路径。
- 如果显式提供了 `--transformer-weights-path`,它会优先于兼容性的 `--model-path` 流程。
- 对于本地目录,SGLang 会首先查找 `*-mixed.safetensors`,然后回退到从该目录加载。
- 在 Blackwell 上,`comfy-kitchen` 在可用时可提供最佳性能路径;否则 SGLang 会回退到通用的 ModelOpt FP4 路径。

## Nunchaku(SVDQuant)

### 安装

首先安装运行时依赖:

```bash
pip install nunchaku
```

有关平台专用的安装方法和故障排查,请参见 [Nunchaku installation guide](https://nunchaku.tech/docs/nunchaku/installation/installation.html)。

### 文件命名与自动检测

对于 Nunchaku 检查点,`--model-path` 仍应指向原始基础模型,而 `--transformer-weights-path` 指向量化后的 transformer 权重。

如果 `--transformer-weights-path` 的 basename 包含模式 `svdq-(int4|fp4)_r{rank}`,SGLang 会自动:
- 启用 SVDQuant
- 推断 `--quantization-precision`
- 推断 `--quantization-rank`

示例:

| checkpoint name fragment | inferred precision | inferred rank | notes |
|--------------------------|--------------------|---------------|-------|
| `svdq-int4_r32`          | `int4`             | `32`          | 标准 INT4 检查点 |
| `svdq-int4_r128`         | `int4`             | `128`         | 更高质量的 INT4 检查点 |
| `svdq-fp4_r32`           | `nvfp4`            | `32`          | 文件名中的 `fp4` 映射到 CLI 值 `nvfp4` |
| `svdq-fp4_r128`          | `nvfp4`            | `128`         | 更高质量的 NVFP4 检查点 |

常见文件名:

| filename | precision | rank | typical use |
|----------|-----------|------|-------------|
| `svdq-int4_r32-qwen-image.safetensors` | `int4` | `32` | 均衡默认 |
| `svdq-int4_r128-qwen-image.safetensors` | `int4` | `128` | 注重质量 |
| `svdq-fp4_r32-qwen-image.safetensors` | `nvfp4` | `32` | RTX 50 系列 / NVFP4 路径 |
| `svdq-fp4_r128-qwen-image.safetensors` | `nvfp4` | `128` | 注重质量的 NVFP4 |
| `svdq-int4_r32-qwen-image-lightningv1.0-4steps.safetensors` | `int4` | `32` | Lightning 4 步 |
| `svdq-int4_r128-qwen-image-lightningv1.1-8steps.safetensors` | `int4` | `128` | Lightning 8 步 |

如果你的检查点名称不遵循此约定,请显式传入 `--enable-svdquant`、`--quantization-precision` 和 `--quantization-rank`。

### 用法示例

推荐的自动检测流程:

```bash
sglang generate \
  --model-path Qwen/Qwen-Image \
  --transformer-weights-path /path/to/svdq-int4_r32-qwen-image.safetensors \
  --prompt "a beautiful sunset" \
  --save-output
```

当文件名未编码量化设置时手动覆盖:

```bash
sglang generate \
  --model-path Qwen/Qwen-Image \
  --transformer-weights-path /path/to/custom_nunchaku_checkpoint.safetensors \
  --enable-svdquant \
  --quantization-precision int4 \
  --quantization-rank 128 \
  --prompt "a beautiful sunset" \
  --save-output
```

### 说明

- `--transformer-weights-path` 是 Nunchaku 检查点的规范 flag。诸如 `quantized_model_path` 之类的较旧配置名称会被当作兼容性别名处理。
- 仅当检查点 basename 匹配 `svdq-(int4|fp4)_r{rank}` 时才会发生自动检测。
- CLI 值为 `int4` 和 `nvfp4`。在文件名中,NVFP4 变体写作 `fp4`。
- Lightning 检查点通常期望匹配的 `--num-inference-steps`,例如 `4` 或 `8`。
- 当前的运行时校验仅允许在 NVIDIA CUDA Ampere(SM8x)或 SM12x GPU 上使用 Nunchaku。Hopper(SM90)目前会被拒绝。

## [ModelSlim](https://gitcode.com/Ascend/msmodelslim)
MindStudio-ModelSlim(msModelSlim)是由 MindStudio 推出并针对 Ascend 硬件优化的模型离线量化压缩工具。

- **安装**

    ```bash
    # Clone repo and install msmodelslim:
    git clone https://gitcode.com/Ascend/msmodelslim.git
    cd msmodelslim
    bash install.sh
    ```

- **Multimodal_sd 量化**

    下载大模型的原始浮点权重。以 Wan2.2-T2V-A14B 为例,你可以前往 [Wan2.2-T2V-A14B](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B) 获取原始模型权重。然后安装其他依赖(与模型相关,参考 modelscope 模型卡片)。
    > 注意:你可以在 [modelscope/Eco-Tech](https://modelscope.cn/models/Eco-Tech) 上找到经过验证的预量化模型。

  使用一键量化运行量化(推荐):

  ```bash
  msmodelslim quant \
    --model_path /path/to/wan2_2_float_weights \
    --save_path /path/to/wan2_2_quantized_weights \
    --device npu \
    --model_type Wan2_2 \
    --quant_type w8a8 \
    --trust_remote_code True
  ```

  有关模型量化的更详细示例以及关于它们支持情况的信息,请参见 ModelSLim repo 中的 [examples](https://gitcode.com/Ascend/msmodelslim/blob/master/example/multimodal_sd/README.md) 部分。

  > 注意:SGLang 不支持量化 embedding,使用 msmodelslim 量化时请禁用该选项。

- **自动检测与不同格式**

    对于 msmodelslim 检查点,只需指定 ```--model-path``` 即可,量化的检测会通过解析 `quant_model_description.json` 配置自动对每一层进行。

    对于 `Wan2.2`,仅支持 `Diffusers` 权重存储格式,而 modelslim 是以原始的 `Wan2.2` 格式保存量化模型的,转换时请使用 `python/sglang/multimodal_gen/tools/wan_repack.py` 脚本:

    ```bash
    python wan_repack.py \
      --input-path {path_to_quantized_model} \
      --output-path {path_to_converted_model}
    ```

    之后,请从原始 `Diffusers` 检查点复制所有文件(`transformer`/`tranfsormer_2` 文件夹除外)

- **用法示例**

    使用自动检测流程:

    ```bash
    sglang generate \
      --model-path Eco-Tech/Wan2.2-T2V-A14B-Diffusers-w8a8 \
      --prompt "a beautiful sunset" \
      --save-output
    ```

- **可用的量化方法**:
    - [x]  ```W4A4_DYNAMIC``` 线性层,在线量化激活值
    - [x]  ```W8A8``` 线性层,离线量化激活值
    - [x]  ```W8A8_DYNAMIC``` 线性层,在线量化激活值
    - [ ]  ```mxfp8``` 线性层(开发中)
