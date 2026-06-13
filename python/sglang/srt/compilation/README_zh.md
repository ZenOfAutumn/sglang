# srt/compilation

## 目录用途
本目录是 SGLang 的图编译子系统，移植并改造自 vLLM。它在 `torch.compile` 基础上实现按算子切分的"分段"（piecewise）编译与分段 CUDA Graph，结合 Inductor 后端、自定义 FX 后处理 pass 与编译缓存，提升模型前向的执行性能并兼容多硬件后端（CUDA/NPU）。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `backend.py` | 编译主后端：`SGLangBackend`、`CompilerManager`、图切分（`split_graph`/`SplitItem`）与 `PiecewiseCompileInterpreter`，串联编译器选择与分段编译流程。 |
| `compilation_config.py` | `CompilationConfig` 编译配置类，以及 `register_split_op` 切分算子注册装饰器与 `SPLIT_OPS` 列表。 |
| `compilation_counter.py` | `CompilationCounter` 数据类，统计模型/图/分段图/后端编译/CUDA Graph 捕获等计数，用于测试与诊断。 |
| `compile.py` | `install_torch_compiled` 入口，配合 `IntermediateTensors`、动态维度推断等，将 torch.compile 安装到模型 forward。 |
| `compiler_interface.py` | 编译器抽象接口 `CompilerInterface` 及 `InductorAdaptor`/`EagerAdapter` 实现，封装 Inductor 编译与缓存键计算。 |
| `cuda_piecewise_backend.py` | `CUDAPiecewiseBackend` 与 `ConcreteSizeEntry`，按具体形状捕获/重放分段 CUDA Graph。 |
| `fix_functionalization.py` | `FixFunctionalizationPass`，对部分节点去函数化以消除冗余张量拷贝的 FX pass。 |
| `fx_utils.py` | FX 图工具函数（`is_func`、`find_auto_fn`、`find_op_nodes` 等）用于在图中定位/匹配节点。 |
| `inductor_pass.py` | Inductor pass 基类体系：`InductorPass`、`SGLangInductorPass`、`CallableInductorPass`、`PassContext` 及 `pass_context` 上下文。 |
| `npu_piecewise_backend.py` | `NPUPiecewiseBackend`，继承 CUDA 后端以适配昇腾 NPU 的分段图执行。 |
| `pass_manager.py` | `PostGradPassManager`，组织并按序运行 post-grad 自定义 FX pass。 |
| `piecewise_context_manager.py` | 分段 CUDA Graph 上下文与前向上下文管理：捕获流、`ForwardContext`、`set/get_forward_context` 等状态开关。 |
| `weak_ref_tensor.py` | `weak_ref_tensors` 工具，按硬件后端创建张量弱引用（CUDA/HIP/MUSA/NPU），用于 CUDA Graph 内存复用。 |
