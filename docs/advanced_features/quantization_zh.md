# Quantization

SGLang 支持多种 quantization 方法,包括离线 quantization 和在线动态 quantization。

离线 quantization 在推理时直接加载预先量化好的模型权重。这是 GPTQ、AWQ 等 quantization 方法所必需的,这些方法使用校准数据集从原始权重中收集并预先计算各种统计量。

在线 quantization 在运行时动态计算缩放参数,例如模型权重的最大值/最小值。与 NVIDIA FP8 训练中的 [delayed scaling](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html#Mixed-precision-training-with-FP8) 机制类似,在线 quantization 会即时计算合适的缩放因子,从而将高精度权重转换为低精度格式。

**注意:为了获得更好的性能、可用性和便利性,推荐使用离线 quantization 而非在线 quantization。**

如果你使用的是预量化模型,请不要同时添加 `--quantization` 来启用在线 quantization。
关于常见的预量化模型,请访问 HF 上的 [Unsloth](https://huggingface.co/unsloth)、[NVIDIA ModelOpt](https://huggingface.co/collections/nvidia/inference-optimized-checkpoints-with-model-optimizer)
或 [NeuralMagic](https://huggingface.co/collections/neuralmagic) 合集,获取一些
经过质量验证的常见量化模型。量化模型在量化后必须通过 benchmark 验证,
以防止出现异常的 quantization 损失回退。

## Platform Compatibility

下表总结了各种 quantization 方法在 NVIDIA、AMD GPU 以及 Ascend NPU 上的支持情况。

| Method | NVIDIA GPUs | AMD GPUs (MI300X/MI325X/MI350X) | Ascend NPUs (A2/A3) | Notes |
|--------|:-----------:|:-------------------------------:|:-----------------------:|-------|
| `fp8` | Yes | Yes | WIP | AMD 上使用 Aiter 或 Triton 后端 |
| `mxfp4` | Yes | Yes | WIP | 需要支持 MXFP 的 CDNA3/CDNA4;使用 Aiter |
| `blockwise_int8` | Yes | Yes | No | 基于 Triton,两个平台均可用 |
| `w8a8_int8` | Yes | Yes | No | |
| `w8a8_fp8` | Yes | Yes | No | AMD 上使用 Aiter 或 Triton FP8 |
| `awq` | Yes | Yes | Yes | AMD 上使用 Triton dequantize(NVIDIA 上则使用优化的 CUDA kernel)。Ascend 上使用 CANN kernel|
| `gptq` | Yes | Yes | Yes | AMD 上使用 Triton 或 vLLM kernel。Ascend 上使用 CANN kernel|
| `compressed-tensors` | Yes | Yes | Partial | AMD 上 FP8/MoE 使用 Aiter 路径。Ascend 上使用 CANN kernel,尚不支持 `FP8`|
| `quark` | Yes | Yes | No | AMD Quark quantization;AMD 上使用 Aiter GEMM 路径 |
| `auto-round` | Yes | Yes | Partial | 平台无关(Intel auto-round)。Ascend 上使用 CANN kernel|
| `quark_int4fp8_moe` | No | Yes | No | 仅 AMD;在线 INT4-to-FP8 MoE quantization(CDNA3/CDNA4) |
| `awq_marlin` | Yes | No | No | Marlin kernel 仅支持 CUDA |
| `gptq_marlin` | Yes | No | No | Marlin kernel 仅支持 CUDA |
| `gguf` | Yes | No | WIP | sgl-kernel 中仅支持 CUDA 的 kernel |
| `modelopt` / `modelopt_fp8` | Yes (Hopper/SM90+) | No | No | [NVIDIA ModelOpt](https://github.com/NVIDIA/Model-Optimizer);需要 NVIDIA 硬件 |
| `modelopt_fp4` | Yes (Blackwell/SM100+) | No | No | [NVIDIA ModelOpt](https://github.com/NVIDIA/Model-Optimizer);Blackwell(B200、GB200)上原生支持 FP4 |
| `petit_nvfp4` | No | Yes (MI250/MI300X/MI325X) | No | 通过 [Petit](https://github.com/causalflow-ai/petit-kernel) 在 ROCm 上启用 NVFP4;在 NVIDIA Blackwell 上请使用 `modelopt_fp4`。在 AMD 上加载 NVFP4 模型时会自动选择。参见 [LMSYS blog](https://lmsys.org/blog/2025-09-21-petit-amdgpu/) 和 [AMD ROCm blog](https://rocm.blogs.amd.com/artificial-intelligence/fp4-mixed-precision/README.html)。 |
| `bitsandbytes` | Yes | Experimental | No | 取决于 bitsandbytes 对 ROCm 的支持 |
| `torchao` (`int4wo` 等) | Yes | Partial | No | AMD 上不支持 `int4wo`;其他方法可能可用 |
| `modelslim` | No | No | Yes | Ascend quantization;使用 CANN kernel |

在 AMD 上,其中若干方法使用 [Aiter](https://github.com/ROCm/aiter) 进行加速——在标注处设置 `SGLANG_USE_AITER=1`。安装和配置详情请参见 [AMD GPU setup](../platforms/amd_gpu.md)。

在 Ascend 上,支持各种层的 quantization 配置,详情请参见 [Ascend NPU quantization](../platforms/ascend/ascend_npu_quantization.md)。

## GEMM Backends for FP4/FP8 Quantization

:::{note}
仅 **blockwise FP8** 和 **NVFP4** GEMM 支持后端选择。在运行 FP8 或 FP4 量化模型时,你可以通过 `--fp8-gemm-backend` 和 `--fp4-gemm-backend` 来选择 GEMM 后端。
:::

### `--fp8-gemm-backend` (Blockwise FP8 GEMM)

| Backend | Hardware | Description |
|---------|----------|-------------|
| `auto` | All | 根据硬件自动选择 |
| `deep_gemm` | SM90, SM100 | JIT 编译;安装 DeepGEMM 后启用 |
| `flashinfer_trtllm` | SM100 | FlashInfer TensorRT-LLM 后端;最适合低延迟 |
| `flashinfer_cutlass` | SM100/120 | FlashInfer CUTLASS groupwise FP8 GEMM |
| `flashinfer_deepgemm` | SM90 | 在解码中针对较小的 M 维度使用 swapAB 优化 |
| `cutlass` | SM90, SM100/120 | sgl-kernel CUTLASS |
| `triton` | All | 回退方案;兼容性广 |
| `aiter` | ROCm | AMD AITER 后端 |

**`auto` 的选择顺序:** 1) DeepGEMM(SM90/SM100,已安装);2) FlashInfer TRTLLM(SM100,FlashInfer 可用);3) CUTLASS(SM90/SM100/120);4) AITER(AMD);5) Triton。**例外:** SM120 始终解析为 Triton。

### `--fp4-gemm-backend` (NVFP4 GEMM)

| Backend | Hardware | Description |
|---------|----------|-------------|
| `auto` | SM100/120 | 自动选择:SM120 上为 `flashinfer_cudnn`;SM100 上为 `flashinfer_cutlass` |
| `cutlass` | SM100/120 | SGLang CUTLASS kernel |
| `flashinfer_cutlass` | SM100/120 | FlashInfer CUTLASS 后端 |
| `flashinfer_cudnn` | SM100/120 (CUDA 13+, cuDNN 9.15+) | FlashInfer cuDNN 后端;在 SM120 上用于提升性能 |
| `flashinfer_trtllm` | SM100 | FlashInfer TensorRT-LLM 后端 |

当 FlashInfer 对 NVFP4 不可用时,会自动回退使用 SGLang CUTLASS kernel。

## Offline Quantization

要加载已经量化好的模型,只需加载模型权重和配置即可。**再次强调,如果模型已离线量化,
启动引擎时无需添加 `--quantization` 参数。quantization 方法会从下载的
Hugging Face 或 msModelSlim 配置中解析。例如,DeepSeek V3/R1 模型已经是 FP8,所以不要添加多余的参数。**

```bash
python3 -m sglang.launch_server \
    --model-path hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
    --port 30000 --host 0.0.0.0
```

请注意,如果你的模型是 **per-channel 量化(INT8 或 FP8)且带有 per-token 动态量化激活**,你可以选择加上 `--quantization w8a8_int8` 或 `--quantization w8a8_fp8`,以调用 sgl-kernel 中对应的 CUTLASS int8_kernel 或 fp8_kernel。此操作将忽略 Hugging Face 配置中的 quantization 设置。例如,对于 `neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic`,如果你以 `--quantization w8a8_fp8` 运行,系统将使用 SGLang 的 `W8A8Fp8Config` 来调用 sgl-kernel,而不是为 vLLM kernel 使用 `CompressedTensorsConfig`。

```bash
python3 -m sglang.launch_server \
    --model-path neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic \
    --quantization w8a8_fp8 \
    --port 30000 --host 0.0.0.0
```

### Examples of Offline Model Quantization

#### Using [Unsloth](https://docs.unsloth.ai/basics/inference-and-deployment/sglang-guide)

我们强烈建议使用 Unsloth 来量化和加载模型。请参考 [SGLang Deployment & Inference Guide with Unsloth](https://docs.unsloth.ai/basics/inference-and-deployment/sglang-guide)。

#### Using [auto-round](https://github.com/intel/auto-round)

```bash
# Install
pip install auto-round
```

- LLM quantization

```py
# for LLM
from auto_round import AutoRound
model_id = "meta-llama/Llama-3.2-1B-Instruct"
quant_path = "Llama-3.2-1B-Instruct-autoround-4bit"
# Scheme examples: "W2A16", "W3A16", "W4A16", "W8A16", "NVFP4", "MXFP4" (no real kernels), "GGUF:Q4_K_M", etc.
scheme = "W4A16"
format = "auto_round"
autoround = AutoRound(model_id, scheme=scheme)
autoround.quantize_and_save(quant_path, format=format) # quantize and save

```

- VLM quantization
```py
# for VLMs
from auto_round import AutoRoundMLLM
model_name = "Qwen/Qwen2-VL-2B-Instruct"
quant_path = "Qwen2-VL-2B-Instruct-autoround-4bit"
scheme = "W4A16"
format = "auto_round"
autoround = AutoRoundMLLM(model_name, scheme)
autoround.quantize_and_save(quant_path, format=format) # quantize and save

```

- Command Line Usage (Gaudi/CPU/Intel GPU/CUDA)

```bash
auto-round \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --bits 4 \
    --group_size 128 \
    --format "auto_round" \
    --output_dir ./tmp_autoround
```

- 已知问题

目前有若干限制会影响在 sglang 中加载离线量化模型,这些问题在未来的 sglang 更新中可能会得到解决。如果你遇到任何问题,可以考虑使用 Hugging Face Transformers 作为替代方案。

1. 混合位宽 Quantization 的限制

    混合位宽 quantization 尚未完全支持。由于 vLLM 的层融合(例如 QKV 融合),对同一融合层内的组件应用不同的位宽会导致兼容性问题。


2. 对量化 MoE 模型的支持有限

    量化后的 MoE 模型可能因 kernel 限制(例如不支持 mlp.gate 层的 quantization)而遇到推理问题。请尝试跳过对这些层的量化,以避免此类错误。


3. 对量化 VLM 的支持有限
    <details>
        <summary>VLM failure cases</summary>

    Qwen2.5-VL-7B

    auto_round:auto_gptq format:  Accuracy is close to zero.

    GPTQ format:  Fails with:
    ```
    The output size is not aligned with the quantized weight shape
    ```
    auto_round:auto_awq and AWQ format:  These work as expected.
    </details>

#### Using [GPTQModel](https://github.com/ModelCloud/GPTQModel)

```bash
# install
pip install gptqmodel --no-build-isolation -v
```

```py
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

model_id = "meta-llama/Llama-3.2-1B-Instruct"
quant_path = "Llama-3.2-1B-Instruct-gptqmodel-4bit"

calibration_dataset = load_dataset(
    "allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz",
    split="train"
  ).select(range(1024))["text"]

quant_config = QuantizeConfig(bits=4, group_size=128) # quantization config
model = GPTQModel.load(model_id, quant_config) # load model

model.quantize(calibration_dataset, batch_size=2) # quantize
model.save(quant_path) # save model
```

#### Using [LLM Compressor](https://github.com/vllm-project/llm-compressor/)

```bash
# install
pip install llmcompressor
```

这里,我们以将 `meta-llama/Meta-Llama-3-8B-Instruct` 量化为 `FP8` 为例,详细说明如何进行离线 quantization。

```python
from transformers import AutoTokenizer
from llmcompressor.transformers import SparseAutoModelForCausalLM
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Step 1: Load the original model.
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

model = SparseAutoModelForCausalLM.from_pretrained(
  MODEL_ID, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Step 2: Perform offline quantization.
# Step 2.1: Configure the simple PTQ quantization.
recipe = QuantizationModifier(
  targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])

# Step 2.2: Apply the quantization algorithm.
oneshot(model=model, recipe=recipe)

# Step 3: Save the model.
SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-Dynamic"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
```

然后,你可以使用以下命令直接通过 `SGLang` 使用量化后的模型:

```bash
python3 -m sglang.launch_server \
    --model-path $PWD/Meta-Llama-3-8B-Instruct-FP8-Dynamic \
    --port 30000 --host 0.0.0.0
```

#### Using [NVIDIA ModelOpt](https://github.com/NVIDIA/Model-Optimizer)

NVIDIA Model Optimizer(ModelOpt)提供针对 NVIDIA 硬件优化的先进 quantization 技术。

**离线 vs. 在线 Quantization:**

SGLang 为 ModelOpt 支持两种模式。

* **离线 Quantization(预量化):**
    * **用法:** 从 Hugging Face 下载预量化模型,或运行一次 `hf_ptq.py` 来创建一个新的量化 checkpoint,然后加载这个量化 checkpoint。
    * **优点:** 服务器启动快,可在部署前验证 quantization,资源使用高效。
    * **缺点:** 需要一个额外的准备步骤。

* **在线 Quantization(量化并服务):**
    * **用法:** 加载一个标准的 BF16/FP16 模型并添加一个 flag。引擎会在*启动时*应用 quantization。
    * **优点:** 方便(无需新的 checkpoint)。
    * **缺点:** **启动时间长**,初始化期间会增加 VRAM 使用量(有 OOM 风险)。

以下章节将指导你使用离线路径:加载预量化模型或创建你自己的 checkpoint。

##### Using Pre-Quantized Checkpoints

如果模型已经量化(例如来自 Hugging Face),你可以直接加载它。

* **FP8 Models:**
    使用 `--quantization modelopt_fp8`。
    ```bash
    python3 -m sglang.launch_server \
        --model-path nvidia/Llama-3.1-8B-Instruct-FP8 \
        --quantization modelopt_fp8 \
        --port 30000
    ```

* **FP4 Models:**
    使用 `--quantization modelopt_fp4`。
    ```bash
    python3 -m sglang.launch_server \
        --model-path nvidia/Llama-3.3-70B-Instruct-NVFP4 \
        --quantization modelopt_fp4 \
        --port 30000
    ```

##### Creating Your Own Quantized Checkpoints

如果你的模型没有现成的预量化 checkpoint,你可以使用 NVIDIA Model Optimizer 的 `hf_ptq.py` 脚本创建一个。

**为什么要量化?**
- 减少 VRAM 使用
- 更高的 throughput 和更低的延迟
- 更灵活的部署(可在更小的 GPU 上)

**哪些部分可以量化?**
- 整个模型
- 仅 MLP 层
- KV cache

**`hf_ptq.py` 中的关键选项:**

`--qformat`: Quantization 格式 `fp8`、`nvfp4`、`nvfp4_mlp_only`

`--kv_cache_qformat`: KV cache quantization 格式(默认:`fp8`)

**注意:** 默认的 `kv_cache_qformat` 可能并非对所有用例都是最优的。请考虑显式设置该值。

**硬件要求:** 推荐 Hopper 及更高架构。GPU 内存不足可能导致权重卸载(offloading),从而使量化时间极长。

关于详细用法和支持的模型架构,请参见 [NVIDIA Model Optimizer LLM PTQ](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_ptq)。

SGLang 包含一个用于使用 ModelOpt 量化模型并自动导出以供部署的简化工作流。

##### Installation

首先,安装 ModelOpt:

```bash
pip install nvidia-modelopt
```

##### Quantization and Export Workflow

SGLang 提供了一个示例脚本,演示完整的 ModelOpt quantization 和导出工作流。请从 SGLang 仓库根目录运行(参见 [modelopt_quantize_and_export.py](https://github.com/sgl-project/sglang/blob/main/examples/usage/modelopt_quantize_and_export.py)):

```bash
# Quantize and export a model using ModelOpt FP8 quantization
python examples/usage/modelopt_quantize_and_export.py quantize \
    --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --export-dir ./quantized_tinyllama_fp8 \
    --quantization-method modelopt_fp8

# For FP4 quantization (requires Blackwell GPU)
python examples/usage/modelopt_quantize_and_export.py quantize \
    --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --export-dir ./quantized_tinyllama_fp4 \
    --quantization-method modelopt_fp4
```

##### Available Quantization Methods

- `modelopt_fp8`: FP8 quantization,在 NVIDIA Hopper 和 Blackwell GPU 上具有最佳性能
- `modelopt_fp4`: FP4 quantization,在 Nvidia Blackwell GPU 上具有最佳性能

##### Python API Usage

你也可以通过编程方式使用 ModelOpt quantization:

```python
import sglang as sgl
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_loader.loader import get_model_loader

# Configure model with ModelOpt quantization and export
model_config = ModelConfig(
    model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quantization="modelopt_fp8",  # or "modelopt_fp4"
    trust_remote_code=True,
)

load_config = LoadConfig(
    modelopt_export_path="./exported_model",
    modelopt_checkpoint_save_path="./checkpoint.pth",  # optional, fake quantized checkpoint
)
device_config = DeviceConfig(device="cuda")

# Load and quantize the model (export happens automatically)
model_loader = get_model_loader(load_config, model_config)
quantized_model = model_loader.load_model(
    model_config=model_config,
    device_config=device_config,
)
```

##### Deploying Quantized Models

在量化和导出之后,你可以使用 SGLang 部署该模型:

```bash
# Deploy the exported quantized model
python -m sglang.launch_server \
    --model-path ./quantized_tinyllama_fp8 \
    --quantization modelopt \
    --port 30000 --host 0.0.0.0
```

或者使用 Python API(使用与量化步骤中 `modelopt_export_path` 相同的路径):

```python
import sglang as sgl

def main():
    # Deploy exported ModelOpt quantized model
    # Path must match modelopt_export_path from quantize step (e.g., ./exported_model)
    llm = sgl.Engine(
        model_path="./exported_model",
        quantization="modelopt",
    )

    # Run inference
    prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
    ]
    sampling_params = {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": 100,
    }

    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        print(f"Prompt: {prompts[i]}")
        print(f"Output: {output['text']}")

if __name__ == "__main__":
    main()

```

##### Advanced Features

**Checkpoint Management**: 保存并恢复 fake quantized checkpoint 以供复用:

```bash
# Save the fake quantized checkpoint during quantization
python examples/usage/modelopt_quantize_and_export.py quantize \
    --model-path meta-llama/Llama-3.2-1B-Instruct \
    --export-dir ./quantized_model \
    --quantization-method modelopt_fp8 \
    --checkpoint-save-path ./my_checkpoint.pth

# The checkpoint can be reused for future quantization runs and skip calibration
```

**Export-only Workflow**: 如果你已经有一个 fake quantized ModelOpt checkpoint,你可以直接导出它。完整 API 请参见 [LoadConfig](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/configs/load_config.py):

```python
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_loader.loader import get_model_loader

model_config = ModelConfig(
    model_path="meta-llama/Llama-3.2-1B-Instruct",
    quantization="modelopt_fp8",
    trust_remote_code=True,
)

load_config = LoadConfig(
    modelopt_checkpoint_restore_path="./my_checkpoint.pth",
    modelopt_export_path="./exported_model",
)

# Load and export the model (DeviceConfig defaults to device="cuda")
model_loader = get_model_loader(load_config, model_config)
model_loader.load_model(model_config=model_config, device_config=DeviceConfig())
```

##### Benefits of ModelOpt

- **硬件优化**: 专门针对 NVIDIA GPU 架构进行优化
- **先进 Quantization**: 支持前沿的 FP8 和 FP4 quantization 技术
- **无缝集成**: 自动导出为 HuggingFace 格式以便于部署
- **基于校准**: 使用校准数据集以获得最佳 quantization 质量
- **生产就绪**: 企业级 quantization,有 NVIDIA 支持

#### Using [ModelSlim](https://gitcode.com/Ascend/msmodelslim)
MindStudio-ModelSlim(msModelSlim)是由 MindStudio 推出、针对 Ascend 硬件优化的模型离线 quantization 压缩工具。

- **Installation**

    ```bash
    # Clone repo and install msmodelslim:
    git clone https://gitcode.com/Ascend/msmodelslim.git
    cd msmodelslim
    bash install.sh
    ```

- **LLM quantization**

    下载大模型的原始浮点权重。以 Qwen3-32B 为例,你可以前往 [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) 获取原始模型权重。然后安装其他依赖(与模型相关,请参考 huggingface model card)。
    > 注意:你可以在 [modelscope/Eco-Tech](https://modelscope.cn/models/Eco-Tech) 上找到经过验证的预量化模型。

    _传统的量化方法需要准备校准数据文件(```.jsonl``` 格式)用于量化过程中的校准。_
    ```bash
    Qwen3-32B/      # floating-point model downloaded from official HF (or modelscope) repo
    msmodelslim/    # msmodelslim repo
      |----- lab_calib # calibration date folder (put your dataset here in ```.jsonl``` format or use pre-prepared ones)
          |----- some file (such as laos_calib.jsonl)
      |----- lab_practice # best practice folder with configs for quantization
          |----- model folder (such as qwen3_5_moe folder) # folder with quantization configs
              |----- quant_config (such as qwen3_5_moe_w8a8.yaml) # quantization config
      |----- another folders
    output_folder/   # generated by below command
      |----- quant_model_weights-00001-of-0001.safetensors # quantized weights
      |----- quant_model_description.json # file with description of the quantization methods for each layer (```W4A4_DYNAMIC```, etc.)
      |----- another files (such as config.json, tokenizer.json, etc.)
    ```
    使用一键量化运行 quantization(推荐):
    ```bash
    msmodelslim quant \
    --model_path ${MODEL_PATH} \
    --save_path ${SAVE_PATH} \
    --device npu:0,1 \
    --model_type Qwen3-32B \
    --quant_type w8a8 \
    --trust_remote_code True
    ```

- **Usage Example**
    ```bash
    python3 -m sglang.launch_server \
    --model-path $PWD/Qwen3-32B-w8a8 \
    --port 30000 --host 0.0.0.0
    ```

- **Available Quantization Methods**:
    - [x]  ```W4A4_DYNAMIC``` linear with online quantization of activations
    - [x]  ```W8A8``` linear with offline quantization of activations
    - [x]  ```W8A8_DYNAMIC``` linear with online quantization of activations
    - [x]  ```W4A4_DYNAMIC``` MOE with online quantization of activations
    - [x]  ```W4A8_DYNAMIC``` MOE with online quantization of activations
    - [x]  ```W8A8_DYNAMIC``` MOE with online quantization of activations
    - [ ]  ```W4A8``` linear TBD
    - [ ]  ```W4A16``` linear TBD
    - [ ]  ```W48A16``` linear TBD
    - [ ]  ```W4A16``` MoE in progress
    - [ ]  ```W8A16``` MoE in progress
    - [ ]  ```KV Cache``` in progress
    - [ ]  ```Attention``` in progress


关于更多模型量化的详细示例,以及它们的支持信息,请参见 ModelSLim 仓库中的 [examples](https://gitcode.com/Ascend/msmodelslim/blob/master/example/README.md) 章节。

## Online Quantization

要启用在线 quantization,你只需在命令行中指定 `--quantization`。例如,你可以使用以下命令启动服务器,为模型 `meta-llama/Meta-Llama-3.1-8B-Instruct` 启用 `FP8` quantization:

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --quantization fp8 \
    --port 30000 --host 0.0.0.0
```

我们的团队正在支持更多在线 quantization 方法。SGLang 即将支持包括但不限于 `["awq", "gptq", "marlin", "gptq_marlin", "awq_marlin", "bitsandbytes", "gguf"]` 的方法。

### torchao online quantization method

SGLang 还支持基于 [torchao](https://github.com/pytorch/ao) 的 quantization 方法。你只需在命令行中指定 `--torchao-config` 即可支持此功能。例如,如果你想为模型 `meta-llama/Meta-Llama-3.1-8B-Instruct` 启用 `int4wo-128`,你可以使用以下命令启动服务器:

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --torchao-config int4wo-128 \
    --port 30000 --host 0.0.0.0
```

SGLang 支持以下基于 torchao 的 quantization 方法 `["int8dq", "int8wo", "fp8wo", "fp8dq-per_tensor", "fp8dq-per_row", "int4wo-32", "int4wo-64", "int4wo-128", "int4wo-256"]`。

注意:根据[此 issue](https://github.com/sgl-project/sglang/issues/2219#issuecomment-2561890230),`"int8dq"` 方法目前在与 cuda graph capture 一起使用时存在一些 bug。因此我们建议在使用 `"int8dq"` 方法时禁用 cuda graph capture。即,请使用以下命令:

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --torchao-config int8dq \
    --disable-cuda-graph \
    --port 30000 --host 0.0.0.0
```

### `quark_int4fp8_moe` online quantization method

在 AMD GPU(CDNA3 或 CDNA4 架构)上运行的 SGLang 支持 quantization 方法 `--quantization quark_int4fp8_moe`,它会将原本采用高精度(bfloat16、float16 或 float32)的 [MoE layers](https://github.com/sgl-project/sglang/blob/v0.4.8/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L271) 替换为使用动态量化为 int4 的权重,这些权重在推理时被上转换(upcast)为 float8,以在 float8 精度下进行计算,激活则即时动态量化为 float8。

其他层(例如 attention 层中的投影)的权重则直接在线量化为 float8。

## Reference

- [GPTQModel](https://github.com/ModelCloud/GPTQModel)
- [LLM Compressor](https://github.com/vllm-project/llm-compressor/)
- [NVIDIA Model Optimizer (ModelOpt)](https://github.com/NVIDIA/Model-Optimizer)
- [NVIDIA Model Optimizer LLM PTQ](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_ptq)
- [Petit: NVFP4 on ROCm](https://github.com/causalflow-ai/petit-kernel) — [LMSYS blog](https://lmsys.org/blog/2025-09-21-petit-amdgpu/), [AMD ROCm blog](https://rocm.blogs.amd.com/artificial-intelligence/fp4-mixed-precision/README.html)
- [Torchao: PyTorch Architecture Optimization](https://github.com/pytorch/ao)
- [vLLM Quantization](https://docs.vllm.ai/en/latest/quantization/)
- [auto-round](https://github.com/intel/auto-round)
- [ModelSlim](https://gitcode.com/Ascend/msmodelslim)
