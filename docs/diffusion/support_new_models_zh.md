# 如何支持新的 Diffusion 模型

本文档说明如何在 SGLang Diffusion 中添加对新 diffusion 模型的支持。

## 架构概览

SGLang Diffusion 在设计上兼顾性能与灵活性,构建于 pipeline 架构之上。这种设计允许开发者为各种 diffusion 模型构建 pipeline,同时使核心生成循环保持标准化以便优化。

其核心架构围绕两个关键概念展开,正如我们的[博客文章](https://lmsys.org/blog/2025-11-07-sglang-diffusion/#architecture)中所强调的:

-   **`ComposedPipeline`**:该类编排一系列 `PipelineStage`,以定义特定模型的完整生成过程。它充当模型的主入口点,并管理 diffusion 过程中各阶段之间的数据流。
-   **`PipelineStage`**:每个 stage 都是一个模块化组件,封装了 diffusion 过程中的一项功能。示例包括 prompt 编码、去噪循环或 VAE 解码。

### 两种 Pipeline 风格

SGLang Diffusion 支持两种 pipeline 组合风格。两者都有效;请选择最适合你模型的那一种。

#### 风格 A:混合式单体 Pipeline(推荐的默认方式)

对大多数新模型而言推荐的默认方式。它使用三阶段结构:

```
BeforeDenoisingStage (model-specific)  →  DenoisingStage (standard)  →  DecodingStage (standard)
```

| Stage | Ownership | Responsibility |
|-------|-----------|----------------|
| `{Model}BeforeDenoisingStage` | 特定于模型 | 所有预处理:输入校验、文本/图像编码、latent 准备、timestep 计算 |
| `DenoisingStage` | 框架标准 | 去噪循环(DiT/UNet 前向传播),在所有模型间共享 |
| `DecodingStage` | 框架标准 | 从 latent 空间到像素空间的 VAE 解码,在所有模型间共享 |

**为什么推荐?** 现代 diffusion 模型往往具有高度异构的预处理需求 —— 不同的文本编码器、不同的 latent 格式、不同的 conditioning 机制。混合式方法将每个模型的预处理隔离开来,避免了带有过多条件逻辑的脆弱共享 stage,并让开发者能够快速移植 Diffusers 参考代码。

#### 风格 B:模块化组合风格

使用框架细粒度的标准 stage(`TextEncodingStage`、`LatentPreparationStage`、`TimestepPreparationStage` 等),通过组合来构建 pipeline。诸如 `add_standard_t2i_stages()` 和 `add_standard_ti2i_stages()` 之类的便捷方法使这种方式非常简洁。

这种风格适用于以下情况:
- **新模型的预处理可以大量复用现有 stage** —— 例如,一个使用标准 CLIP/T5 文本编码 + 标准 latent 准备且只需极少定制的模型。
- **某个特定于模型的优化需要被提取为一个独立 stage** —— 例如,一个专门的编码或 conditioning 步骤,因被拆为独立 stage 而在性能分析、并行控制或跨多个 pipeline 变体复用方面获益。

#### 如何选择

| Situation | Recommended Style |
|-----------|-------------------|
| 模型具有独特/复杂的预处理(VLM captioning、AR token 生成、自定义 latent packing 等) | **Hybrid** —— 合并为一个 BeforeDenoisingStage |
| 模型恰好契合标准的 text-to-image 或 text+image-to-image 模式 | **Modular** —— 使用 `add_standard_t2i_stages()` / `add_standard_ti2i_stages()` |
| 移植一个带有许多自定义步骤的 Diffusers pipeline | **Hybrid** —— 将 `__call__` 逻辑复制到单个 stage 中 |
| 添加一个与现有模型共享大部分逻辑的变体 | **Modular** —— 复用现有 stage,通过 PipelineConfig 回调进行定制 |
| 某个特定预处理步骤需要特殊的并行或性能分析隔离 | **Modular** —— 将该步骤提取为一个专用 stage |

## 实现所需的关键组件

要添加对新 diffusion 模型的支持,你需要定义或配置以下组件:

1.  **`PipelineConfig`**:一个 dataclass,持有你模型 pipeline 的静态配置 —— 精度设置、模型架构参数,以及标准 `DenoisingStage` 和 `DecodingStage` 所使用的回调方法。每个模型都有自己的子类。

2.  **`SamplingParams`**:一个定义运行时生成参数的 dataclass —— `prompt`、`negative_prompt`、`guidance_scale`、`num_inference_steps`、`seed`、`height`、`width` 等。

3.  **预处理 stage**:要么是单个特定于模型的 `{Model}BeforeDenoisingStage`(混合式风格),要么是标准 stage 的组合(模块化风格)。参见上文的[两种 Pipeline 风格](#two-pipeline-styles)。

4.  **`ComposedPipeline`**:一个将你的预处理 stage 与标准 `DenoisingStage` 和 `DecodingStage` 连接起来的类。参见基础定义:
    - [`ComposedPipelineBase`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/composed_pipeline_base.py)
    - [`PipelineStage`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/base.py)
    - [Central registry](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/registry.py)

5.  **Modules(模型组件)**:每个 pipeline 引用从模型仓库(例如 Diffusers 的 `model_index.json`)加载的模块:
    - `text_encoder`:将文本 prompt 编码为 embedding。
    - `tokenizer`:为文本编码器对原始文本输入进行分词。
    - `processor`:预处理图像并提取特征;常用于 image-to-image 任务。
    - `image_encoder`:专门的图像特征提取器。
    - `dit/transformer`:在 latent 空间中运行的核心去噪网络(DiT/UNet 架构)。
    - `scheduler`:控制 timestep 调度和去噪动态。
    - `vae`:变分自编码器,用于在像素空间和 latent 空间之间进行编码/解码。

## Pipeline Stages 参考

### 核心 Stage(所有 pipeline 都会使用)

| Stage Class                      | Description                                                                                             |
| -------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `DenoisingStage`                 | 执行主去噪循环,迭代地将模型(DiT/UNet)应用于 latent 以进行精炼。      |
| `DecodingStage`                  | 使用 VAE 将最终的 latent 张量解码回像素空间。                                    |
| `DmdDenoisingStage`              | 针对 DMD 模型架构的专门去噪 stage。                                              |
| `CausalDMDDenoisingStage`        | 针对特定视频模型的专门因果去噪 stage。                                         |

### 预处理 Stage(用于模块化组合风格)

以下细粒度 stage 可以组合起来,构建 pipeline 的预处理部分。它们最适合那些预处理大体契合标准模式的模型。如果你的模型需要大量定制,请考虑采用混合式风格,使用单个 `BeforeDenoisingStage`。

| Stage Class                      | Description                                                                                             |
| -------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `InputValidationStage`           | 校验用户提供的 `SamplingParams`。                                                               |
| `TextEncodingStage`              | 使用一个或多个文本编码器将文本 prompt 编码为 embedding。                                   |
| `ImageEncodingStage`             | 将输入图像编码为 embedding,常用于 image-to-image 任务。                               |
| `ImageVAEEncodingStage`          | 使用 VAE 将输入图像编码到 latent 空间。                                                 |
| `TimestepPreparationStage`       | 为 diffusion 过程准备 scheduler 的 timestep。                                           |
| `LatentPreparationStage`         | 创建将被去噪的初始带噪 latent 张量。                                          |

## 实现指南

### 步骤 1:获取并研究参考实现

在编写任何代码之前,先获取该模型的原始实现或 Diffusers pipeline 代码:
- 该模型的 Diffusers pipeline 源码(例如 `diffusers` 库或 HuggingFace repo 中的 `pipeline_*.py` 文件)
- 或者该模型的官方参考实现(例如来自模型作者的 GitHub repo)
- 或者 HuggingFace model ID,用于查找 `model_index.json` 和关联的 pipeline 类

获得参考代码后,要彻底地研究它:

1. 找到模型的 `model_index.json` 以确定所需的模块。
2. 阅读 Diffusers pipeline 的 `__call__` 方法,以理解:
   - 文本 prompt 是如何编码的
   - latent 是如何准备的(shape、dtype、scaling)
   - timestep/sigma 是如何计算的
   - DiT 期望什么样的 conditioning kwargs
   - 去噪循环是如何工作的
   - VAE 解码是如何完成的

### 步骤 2:评估对现有 Pipeline 和 Stage 的复用

在创建任何新文件之前,先检查是否有现有的 pipeline 或 stage 可以复用或扩展。只有在现有 pipeline/stage 需要大量结构性改动,或者不存在架构相似的实现时,才创建新的 pipeline/stage。

- **与现有 pipeline 对比**(Flux、Wan、Qwen-Image、GLM-Image、HunyuanVideo、LTX 等)。如果新模型与某个现有模型共享其大部分结构,优先添加一个新的 config 变体或复用现有 stage。
- **检查现有 stage**,位于 `runtime/pipelines_core/stages/` 和 `stages/model_specific_stages/`。
- **检查现有模型组件** —— 许多模型共享 VAE(例如 `AutoencoderKL`)、文本编码器(CLIP、T5)和 scheduler。直接复用它们。

### 步骤 3:实现模型组件

适配模型的核心组件:

- **DiT/Transformer**:在 [`runtime/models/dits/`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/) 中实现
- **Encoders**:在 [`runtime/models/encoders/`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/encoders/) 中实现
- **VAEs**:在 [`runtime/models/vaes/`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/vaes/) 中实现
- **Schedulers**:如有需要,在 [`runtime/models/schedulers/`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/schedulers/) 中实现

尽可能使用 SGLang 的融合 kernel(参见 `LayerNormScaleShift`、`RMSNormScaleShift`、`apply_qk_norm` 等)。

**Tensor Parallel(TP)和 Sequence Parallel(SP)**:对于多 GPU 部署,建议为 DiT 模型添加 TP/SP 支持。这可以在单 GPU 实现得到验证之后逐步进行。参考实现:
- **Wan model**(`runtime/models/dits/wanvideo.py`)—— 完整的 TP + SP:用于 attention 的 `ColumnParallelLinear`/`RowParallelLinear`,通过 `get_sp_world_size()` 进行序列维度分片
- **Qwen-Image model**(`runtime/models/dits/qwen_image.py`)—— 通过 `USPAttention`(Ulysses + Ring Attention)实现 SP

### 步骤 4:创建 Config

- **DiT Config**:`configs/models/dits/{model_name}.py`
- **VAE Config**:`configs/models/vaes/{model_name}.py`
- **SamplingParams**:`configs/sample/{model_name}.py`

### 步骤 5:创建 PipelineConfig

`PipelineConfig` 提供标准 `DenoisingStage` 和 `DecodingStage` 所使用的回调:

```python
# python/sglang/multimodal_gen/configs/pipeline_configs/my_model.py

@dataclass
class MyModelPipelineConfig(ImagePipelineConfig):
    task_type: ModelTaskType = ModelTaskType.T2I
    vae_precision: str = "bf16"
    should_use_guidance: bool = True
    dit_config: DiTConfig = field(default_factory=MyModelDitConfig)
    vae_config: VAEConfig = field(default_factory=MyModelVAEConfig)

    def get_freqs_cis(self, batch, device, rotary_emb, dtype):
        """Prepare rotary position embeddings for the DiT."""
        ...

    def prepare_pos_cond_kwargs(self, batch, latent_model_input, t, **kwargs):
        """Build positive conditioning kwargs for each denoising step."""
        return {
            "hidden_states": latent_model_input,
            "encoder_hidden_states": batch.prompt_embeds[0],
            "timestep": t,
        }

    def prepare_neg_cond_kwargs(self, batch, latent_model_input, t, **kwargs):
        """Build negative conditioning kwargs for CFG."""
        return {
            "hidden_states": latent_model_input,
            "encoder_hidden_states": batch.negative_prompt_embeds[0],
            "timestep": t,
        }

    def get_decode_scale_and_shift(self):
        """Return (scale, shift) for latent denormalization before VAE decode."""
        ...
```

### 步骤 6:实现预处理

根据你模型的需求进行选择(参见[如何选择](#how-to-choose)):

#### 选项 A:BeforeDenoisingStage(混合式风格)

创建一个处理所有预处理的单个 stage。当模型具有自定义/复杂预处理逻辑时为最佳选择。

```python
# python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/my_model.py

class MyModelBeforeDenoisingStage(PipelineStage):
    """Monolithic pre-processing stage for MyModel.

    Consolidates: input validation, text/image encoding, latent
    preparation, and timestep computation.
    """

    def __init__(self, vae, text_encoder, tokenizer, transformer, scheduler):
        super().__init__()
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.scheduler = scheduler

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()

        # 1. Encode prompt (model-specific logic)
        prompt_embeds, negative_prompt_embeds = self._encode_prompt(...)

        # 2. Prepare latents
        latents = self._prepare_latents(...)

        # 3. Prepare timesteps
        timesteps, sigmas = self._prepare_timesteps(...)

        # 4. Populate batch for DenoisingStage
        batch.prompt_embeds = [prompt_embeds]
        batch.negative_prompt_embeds = [negative_prompt_embeds]
        batch.latents = latents
        batch.timesteps = timesteps
        batch.num_inference_steps = len(timesteps)
        batch.sigmas = sigmas.tolist()
        batch.generator = generator
        batch.raw_latent_shape = latents.shape
        return batch
```

#### 选项 B:标准 Stage(模块化风格)

完全跳过创建自定义 stage —— 通过 `PipelineConfig` 回调进行配置并使用框架辅助方法。当模型契合标准模式时为最佳选择。

(此选项没有单独的 stage 文件;步骤 7 中的 pipeline 类直接调用 `add_standard_t2i_stages()`。)

**`DenoisingStage` 期望的关键 batch 字段**(无论你选择哪个选项):

| Field | Type | Description |
|-------|------|-------------|
| `batch.latents` | `torch.Tensor` | 初始带噪 latent 张量 |
| `batch.timesteps` | `torch.Tensor` | timestep 调度 |
| `batch.num_inference_steps` | `int` | 去噪步数 |
| `batch.sigmas` | `list[float]` | sigma 调度(必须是 Python list,而不是 numpy) |
| `batch.prompt_embeds` | `list[torch.Tensor]` | 正向 prompt embedding(包装在 list 中) |
| `batch.negative_prompt_embeds` | `list[torch.Tensor]` | 负向 prompt embedding(包装在 list 中) |
| `batch.generator` | `torch.Generator` | 用于可复现性的 RNG generator |
| `batch.raw_latent_shape` | `tuple` | 任何 packing 之前的原始 latent shape |

### 步骤 7:定义 Pipeline 类

#### 混合式风格

```python
# python/sglang/multimodal_gen/runtime/pipelines/my_model.py

class MyModelPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "MyModelPipeline"  # Must match model_index.json _class_name

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        # 1. Monolithic pre-processing (model-specific)
        self.add_stage(
            MyModelBeforeDenoisingStage(
                vae=self.get_module("vae"),
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        # 2. Standard denoising loop (framework-provided)
        self.add_stage(
            DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        # 3. Standard VAE decoding (framework-provided)
        self.add_standard_decoding_stage()


EntryClass = [MyModelPipeline]
```

#### 模块化风格

```python
# python/sglang/multimodal_gen/runtime/pipelines/my_model.py

class MyModelPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "MyModelPipeline"

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        # All pre-processing + denoising + decoding in one call
        self.add_standard_t2i_stages(
            prepare_extra_timestep_kwargs=[prepare_mu],  # model-specific hooks
        )


EntryClass = [MyModelPipeline]
```

### 步骤 8:注册模型

在 [`registry.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/registry.py) 中注册你的 config:

```python
register_configs(
    model_family="my_model",
    sampling_param_cls=MyModelSamplingParams,
    pipeline_config_cls=MyModelPipelineConfig,
    hf_model_paths=["org/my-model-name"],
)
```

你 pipeline 文件中的 `EntryClass` 会被 registry 自动发现 —— pipeline 类本身无需额外注册。

### 步骤 9:验证输出质量

实现完成后,验证生成的输出不是噪声。带噪或乱码的输出是实现不正确的最常见迹象。常见原因包括:

- 不正确的 latent scale/shift 因子
- 错误的 timestep/sigma 调度(顺序、dtype 或取值范围)
- 不匹配的 conditioning kwargs
- 旋转位置编码风格不匹配(`is_neox_style`)

调试方法是:使用相同的 seed,将中间张量值与 Diffusers 参考 pipeline 进行对比。

## 参考实现

### 混合式风格

| Model | Pipeline | BeforeDenoisingStage | PipelineConfig |
|-------|----------|---------------------|----------------|
| GLM-Image | `runtime/pipelines/glm_image.py` | `stages/model_specific_stages/glm_image.py` | `configs/pipeline_configs/glm_image.py` |
| Qwen-Image-Layered | `runtime/pipelines/qwen_image.py` | `stages/model_specific_stages/qwen_image_layered.py` | `configs/pipeline_configs/qwen_image.py` |

### 模块化风格

| Model | Pipeline | Notes |
|-------|----------|-------|
| Qwen-Image (T2I) | `runtime/pipelines/qwen_image.py` | 使用 `add_standard_t2i_stages()` |
| Qwen-Image-Edit | `runtime/pipelines/qwen_image.py` | 使用 `add_standard_ti2i_stages()` |
| Flux | `runtime/pipelines/flux.py` | 使用带自定义 `prepare_mu` 的 `add_standard_t2i_stages()` |
| Wan | `runtime/pipelines/wan_pipeline.py` | 使用 `add_standard_ti2v_stages()` |

## 检查清单

在提交你的实现之前,请验证:

**通用(两种风格):**
- [ ] 在 `runtime/pipelines/{model_name}.py` 的 **Pipeline 文件**,带有 `EntryClass`
- [ ] 在 `configs/pipeline_configs/{model_name}.py` 的 **PipelineConfig**
- [ ] 在 `configs/sample/{model_name}.py` 的 **SamplingParams**
- [ ] 在 `runtime/models/dits/{model_name}.py` 的 **DiT 模型**
- [ ] 在 `configs/models/dits/` 和 `configs/models/vaes/` 的 **模型 config**(DiT、VAE)
- [ ] 在 `registry.py` 中通过 `register_configs()` 的 **Registry 条目**
- [ ] `pipeline_name` 与 Diffusers `model_index.json` 的 `_class_name` 匹配
- [ ] `_required_config_modules` 列出了 `model_index.json` 中的所有模块
- [ ] `PipelineConfig` 回调(`prepare_pos_cond_kwargs` 等)与 DiT 的 `forward()` 签名匹配
- [ ] 使用框架标准的 `DenoisingStage` 和 `DecodingStage`(而非自定义去噪循环)
- [ ] 为 DiT 模型考虑了 **TP/SP 支持**(推荐;TP+SP 参考 `wanvideo.py`,USPAttention 参考 `qwen_image.py`)
- [ ] **已验证输出质量** —— 生成的图像/视频不是噪声;已与 Diffusers 参考输出对比

**仅混合式风格:**
- [ ] 在 `stages/model_specific_stages/{model_name}.py` 的 **BeforeDenoisingStage**
- [ ] `BeforeDenoisingStage.forward()` 填充了 `DenoisingStage` 所需的所有 batch 字段
