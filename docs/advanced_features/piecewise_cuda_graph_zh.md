# 分段 CUDA Graph(Piecewise CUDA Graph)

## 动机

标准 CUDA graph 将整个模型前向传递捕获为单个图。这对 decode(固定批次大小)效果很好,但对 extend/prefill 则不适用——因为后者每次迭代的 token 数量各不相同。

分段 CUDA Graph(Piecewise CUDA Graph,PCG)通过在"分割点"(split point,例如 MoE dispatch 算子)处将模型的计算图切分成若干片段(大致每层一个)来解决这个问题。每个片段针对一组预定义的 token 长度被捕获为独立的 CUDA graph。在运行时,输入被填充到最接近的已捕获大小,然后逐片段重放。这消除了 prefill/extend 的 kernel 启动开销,同时仍然支持动态形状。

最近我们**默认启用了 PCG**,这意味着旧的 `--enable-piecewise-cuda-graph` 标志已被弃用。使用 `--disable-piecewise-cuda-graph` 来关闭它。

## 用法

对于受支持的配置,PCG 默认启用。无需额外标志:

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct
```

### 禁用 PCG

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --disable-piecewise-cuda-graph
```

### 自定义捕获大小

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --piecewise-cuda-graph-max-tokens 2048
```

### 服务器参数

| 参数 | 默认值 | 描述 |
|---|---|---|
| `--disable-piecewise-cuda-graph` | `False` | 为 extend/prefill 禁用 PCG。 |
| `--enforce-piecewise-cuda-graph` | `False` | 强制启用 PCG,跳过所有自动禁用条件。仅用于测试。 |
| `--piecewise-cuda-graph-max-tokens` | `None`(自动) | 要捕获的最大 token 数量。默认为 `chunked_prefill_size`(非 MLA)或 `2048`(MLA)。 |
| `--piecewise-cuda-graph-tokens` | `None`(自动) | 要捕获的 token 长度的显式列表。如果未设置则自动生成。 |
| `--piecewise-cuda-graph-compiler` | `"eager"` | 用于已捕获子图的编译器后端。可选:`eager`、`inductor`。 |
| ~~`--enable-piecewise-cuda-graph`~~ | — | **已弃用。** PCG 现在默认启用。使用 `--enforce-piecewise-cuda-graph` 来跳过自动禁用条件。 |

## Bug 报告

PCG 默认启用,但仍处于实验阶段。由于 PCG 依赖 `torch.compile` 来追踪模型的前向传递,大多数 bug 都是由 torch compile 追踪失败引起的(例如,不可追踪的算子、动态控制流或图中断)。如果你遇到任何与 PCG 相关的问题,请通过在启动命令中添加 `--disable-piecewise-cuda-graph` 来禁用它,并在 [GitHub Issues](https://github.com/sgl-project/sglang/issues/new/choose) 上报告该 bug。我们非常感谢你帮助改进此功能。

### 对于用户

如果你在服务器启动期间看到如下错误信息,这是一个 PCG bug:

```
Piecewise CUDA Graph is enabled by default as an experimental feature.
To work around this error, add --disable-piecewise-cuda-graph to your launch command.
Please report this issue at https://github.com/sgl-project/sglang/issues/new/choose
```

要绕过它,请在启动命令中添加 `--disable-piecewise-cuda-graph`。在提交 bug 报告时,请包含:
1. 完整的错误堆栈跟踪
2. 模型名称和量化方法
3. 带所有参数的启动命令
4. GPU 类型和驱动版本

### 对于开发者

由于 PCG 依赖 `torch.compile` 来追踪模型的前向传递,新开发的 CUDA kernel(包括 JIT kernel 和 sgl-kernel)通常开箱即用时与 `torch.compile` 不兼容。追踪会在不可追踪的操作上失败,例如 kernel 内部的 JIT 编译、文件 I/O 或动态模块加载。

要使一个 kernel 与 PCG 兼容,你需要使用来自 `sglang.srt.utils.custom_op` 的 `register_custom_op` 将其注册为自定义算子。这会将该 kernel 包装为已编译图中的一个不透明节点,使得 `torch.compile` 不会追踪其内部。

**用法示例(JIT kernel):**

```python
from sglang.srt.utils.custom_op import register_custom_op

# Inplace operator (no return value)
@register_custom_op(mutates_args=["output_q", "output_s"])
def per_token_group_quant_8bit(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
) -> None:
    # kernel implementation ...
```

**用法示例(有输出的算子):**

```python
# out_shape indicates which argument has the same shape as the output
@register_custom_op(mutates_args=["x"], out_shape=0)
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x.add_(y)
```

对于包装外部库函数(例如 FlashInfer kernel),请改用 `register_custom_op_from_extern`。完整的 API 文档请参见 `python/sglang/srt/utils/custom_op.py`。

## 工作原理
### Torch compile 后端

PCG 使用带有自定义后端(`SGLangBackend`)的 `torch.compile` 来分割并编译模型的前向传递。流程如下:

```
model.forward wrapper
→ torch.compile(..., backend=SGLangBackend)
→ FX graph
→ split_graph() at registered split ops
→ split_gm (top-level graph that chains the pieces)
→ replace capturable submodules with CUDAPiecewiseBackend
→ runtime dispatch: eager split ops + per-piece capture/replay
```

- **Install(安装)**:`install_torch_compiled()` 将 `model.forward` 替换为一个包装函数。当 `is_in_piecewise_cuda_graph()` 返回 True 时,包装器分发到已编译的可调用对象;否则回退到原始的 forward。第一次通过此路径调用会触发 Dynamo 追踪和图编译——CUDA graph 重放只在捕获阶段完成之后才会发生。

- **Split(分割)**:当 `torch.compile` 追踪模型时,`SGLangBackend` 接收 FX 图并调用 `split_graph()`。`CompilationConfig.split_ops` 中列出的算子被视为分割点,因此图在每个分割点处被切开。这些 split-op 子模块在运行时以 eager 方式执行,而周围的子模块被编译并由 `CUDAPiecewiseBackend` 包装。其结果是一个顶层"拼接图"(`split_gm`),其子节点如 `submod_0`、`submod_1`……交替排列着可捕获子图和 eager split-op 子模块。

- **Replace(替换)**:`PiecewiseCompileInterpreter` 遍历 `split_gm` 中的每个可捕获子模块,将其针对通用(动态)形状编译,并就地替换为一个 `CUDAPiecewiseBackend` 实例。Split-op 子模块(例如 attention、all-reduce)保持原样,在运行时以 eager 方式执行。

- **Dispatch(分发)**:在运行时,调用 `split_gm` 会执行拼接图,后者依次调用每个子模块。Split-op 子模块以 eager 方式运行。每个 `CUDAPiecewiseBackend` 子模块经历三个阶段:
  - **Compile warmup(编译预热)** —— 运行通用形状的已编译路径。
  - **Capture(捕获)** —— 对每个捕获大小,运行一次预热传递,然后记录一个 CUDA graph。
  - **Steady-state replay(稳态重放)** —— 在每次前向传递时重放已捕获的 CUDA graph。

### 分段 CUDA graph runner

`PiecewiseCudaGraphRunner` 通过三个阶段编排完整的生命周期:

- **Compile(编译)** —— 用一次虚拟(dummy)前向传递预热 JIT kernel,然后用 `torch.compile` 包装模型,触发 Dynamo 追踪以分割 FX 图,并为每个子图片段创建 `CUDAPiecewiseBackend` 实例。

- **Capture(捕获)** —— 按逆序(从大到小)遍历捕获大小。对每个大小,运行两次前向传递(一次预热,一次 CUDA graph 捕获)。

- **Replay(重放)** —— 在运行时,通过二分查找找到大于等于实际 token 数量的最小已捕获大小,将输入零填充复制到静态缓冲区中,重放已捕获的 CUDA graph,然后将输出切片回实际的 token 数量。

### 内存优化

PCG 的内存开销来自两部分:**torch 内存分配器**和**非 torch 内存**。

得益于若干优化,torch 内存分配器的开销微不足道:一个全局共享内存池在所有 CUDA graph runner 和捕获大小之间复用;捕获按逆序(从大到小)进行,因此较小的图会复用较大图分配的内存;最后一个子图的输出张量以弱引用形式存储,以最大化内存复用。

主要的内存开销来自非 torch 内存——CUDA graph 对象本身需要 GPU 内存来存储记录的 kernel 启动参数和内部状态。这一开销随捕获大小的数量而增长,这也是为什么 `piecewise_cuda_graph_max_tokens` 默认被保守地设上限。

### 形状配置
分段 CUDA graph 为一组 token 数量预先捕获图。在运行时,实际的 token 数量会向上取整到最接近的已捕获大小(通过二分查找),然后重放相应的图。如果 token 数量超过最大的已捕获大小,运行时会回退到正常的(非图)前向路径。

默认的捕获计划以逐渐增加的粒度自动生成:

| Token 范围 | 步长 |
|-------------|-----------|
| 4 – 32      | 4         |
| 48 – 256    | 16        |
| 288 – 512   | 32        |
| 576 – 1024  | 64        |
| 1280 – 4096 | 256       |
| 4096+       | 512       |

对于自动生成的计划,大小以 `--piecewise-cuda-graph-max-tokens` 为上限。默认上限对非 MLA 模型是 `chunked_prefill_size`,对 MLA 后端模型是 `2048`。如果设置了 `--max-total-tokens`,上限会进一步被限制为不超过它。此外,作为临时变通方案,Llama-2 模型被自动上限为 4096 个 token。

## 兼容性

PCG 在以下场景中会被自动禁用。我们正在积极扩展兼容性——其中许多场景的支持很快就会到来。

- 被禁用的模型架构(例如 `DeepseekV32ForCausalLM`)
- 投机解码(Speculative decoding)
- DP attention
- 流水线并行(`pp_size > 1`)
- 非 CUDA 硬件(AMD ROCm、Ascend NPU)
- MoE A2A 后端
- LoRA
- 多模态 / VLM 模型
- DLLM(diffusion LLM)
- 确定性推理(Deterministic inference)
- PD 分离
- 专家分布记录器(Expert distribution recorder)/ EPLB

使用 `--enforce-piecewise-cuda-graph` 来跳过所有自动禁用检查(仅用于测试/调试)。

## 代码参考

| 文件 | 描述 |
|---|---|
| `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` | 主 runner:初始化、捕获、重放 |
| `python/sglang/srt/compilation/compile.py` | `install_torch_compiled` trampoline |
| `python/sglang/srt/compilation/backend.py` | `SGLangBackend`、图分割、分段编译 |
| `python/sglang/srt/compilation/cuda_piecewise_backend.py` | 每个子图的 CUDA graph 捕获/重放 |
| `python/sglang/srt/compilation/piecewise_context_manager.py` | 全局上下文标志和 `ForwardContext` |
| `python/sglang/srt/compilation/compilation_config.py` | 捕获大小、split ops、编译器配置 |
| `python/sglang/srt/utils/custom_op.py` | 用于 torch.compile 兼容性的 `register_custom_op` |
| `python/sglang/srt/server_args.py` | 服务器参数和自动禁用逻辑 |
