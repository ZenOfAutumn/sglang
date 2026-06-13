# JIT Kernel 开发指南

## 环境设置

我们强烈建议在 JIT kernel 开发中使用 `clangd` 作为语言服务器。
对于 Ubuntu/Debian，你可以从 [apt.llvm.org](https://apt.llvm.org/) 下载 clangd。
如果你使用 VS Code，我们建议安装 `clangd` 扩展以获得更好的 IDE 集成。

所有与 JIT 相关的文件都位于 `python/sglang/jit_kernel`。
与提前编译（AOT）CUDA/C++ 二进制文件的 `sgl-kernel` 不同，即时编译（JIT）kernel 在运行时编译。
因此，无法生成静态的 `compile_commands.json`。
为了启用 `clangd` 的代码补全，运行 `python -m sglang.jit_kernel` 以在当前目录中生成 `.clangd` 配置文件。
生成该文件后，重启 clangd 语言服务器。它现在应该能识别所有 JIT kernel 文件了。

## 代码结构

### C++ 实现

C++ 源代码位于 `python/sglang/jit_kernel/csrc`。
可复用的函数应放置在 `python/sglang/jit_kernel/include`。

我们使用 [tvm-ffi](https://github.com/apache/tvm-ffi) 进行高效的外部语言绑定。
关于高级用法（例如导出 C++ 对象），请参考[文档](https://tvm.apache.org/ffi/)。
通常，`tvm::ffi::TensorView` 足以从 Python 传递 PyTorch Tensors。

### Python 接口

Python 接口定义在 `python/sglang/jit_kernel` 中。
`python/sglang/jit_kernel/utils.py` 中的 `load_jit` 工具函数加载并返回编译后的模块。
要导出一个 C++ 函数（例如 `cpp_func`），将 `cuda_wrappers=[("func", "cpp_func")]` 传递给 `load_jit`。
然后该函数可以在 Python 中以 `module.func` 调用。

对于缓存编译后的模块，请优先使用 `sglang.jit_kernel.utils.cache_once` 而非 `functools.lru_cache`。
`functools.lru_cache` 与 `torch.compile` 不兼容。

### C++ 工具

以下 C++ 工具可用：

#### 整数范围

与 PyTorch 类似，我们提供了一个 `irange` 函数来表示一个整数范围。

```C++
#include <sgl_kernel/utils.h>

void test() {
  for (auto i : host::irange(100)) { // [0, 100)
    // do something
  }
  for (auto i : host::irange(0, 100)) { // [0, 100)
    // do something
  }
}

```

#### 运行时检查

`RuntimeCheck` 在运行时验证条件。它接受可选参数用于错误报告。
如果检查失败，这些参数会被输出以帮助调试。
`RuntimeDeviceCheck` 验证上一次 kernel 启动的状态。

```C++
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

void test() {
  host::RuntimeCheck(1 + 1 == 2, 1 + 1, " != ", 2);
  host::RuntimeDeviceCheck();
  // check the provided `cudaError_t`
  host::RuntimeDeviceCheck(cudaGetLastError());
}

```

#### Tensor 检查

`TensorMatcher` 提供了一种可读的方式来验证和提取 tensor 的形状信息。

```cpp
#include <sgl_kernel/tensor.h>

void test(const tvm::ffi::TensorView k_cache, const tvm::ffi::TensorView v_cache) {
  using namespace host;

  auto D = SymbolicSize{"D"};  // cache dimension
  auto N = SymbolicSize{"N"};  // kvcache stride
  auto dtype = SymbolicDType{};
  auto device = SymbolicDevice{};

  TensorMatcher({-1, D})  //
      .with_strides({N, 1})
      .with_dtype<int32_t, int64_t>(dtype)
      .with_device<kDLCUDA, kDLCPU>(device)
      .verify(k_cache)
      .verify(v_cache);
}
```

在验证之前，使用预期的 stride、dtype 和 device 属性配置 `TensorMatcher`。
- 如果省略 `with_strides`，则该 tensor 预期为连续的（contiguous）。
- `with_dtype` 中的模板参数限制了允许的数据类型。
- `with_device` 中的模板参数限制了允许的设备。
- 传递给 `with_xxx` 方法的值会强制进行相等性检查。
- 为 size 或 stride 传递 `-1` 允许匹配任意值。

一个 `Symbolic` 变量在所有验证中必须解析为相同的值。
使用 `.unwrap()` 在验证后检索匹配到的值。

> 注意：`TensorMatcher` 是一个临时表达式，不应存储在变量中。

> 提示：在 `TensorMatcher` 链的末尾添加 `//` 以强制正确的缩进。

#### Kernel 启动

`LaunchKernel::resolve_device` 从 PyTorch 检索当前的 `cudaStream`。
也可以使用 `LaunchKernel` 直接启动 kernel。

```cpp
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>

__global__ void kernel() {}

void test() {
  const auto num_blocks = 1;
  const auto num_threads = 32;
  const auto dynamic_smem = 0;

  DLDevice dev;  // suppose this is initialized properly
  host::LaunchKernel(num_blocks, num_threads, dev)(kernel);

  cudaStream_t stream = host::LaunchKernel::resolve_device(dev);
  host::LaunchKernel(num_blocks, num_threads, stream, dynamic_smem)(kernel);
}

```

## 添加新 kernel

本节将完整地、端到端地演示一个向系统中添加新 JIT kernel 的示例。
我们以一个简单的 add_constant kernel 作为贯穿的示例，它将一个常量整数值加到输入 tensor 的每个元素上。

从概念上讲，Python 接口看起来像这样：

```python
def add_constant(src: torch.Tensor, c: int):
    return src + c
```

### 步骤 1：编写 C++ kernel

在 [jit_kernel/csrc/add_constant.cuh](../../python/sglang/jit_kernel/csrc/add_constant.cuh) 中编写你的 CUDA kernel。出于演示目的，我们将常量值作为模板参数传递。

```cpp
#include <sgl_kernel/tensor.h>   // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.cuh>  // For LaunchKernel
#include <sgl_kernel/utils.h>    // For div_ceil, RuntimeCheck

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

namespace {

template <int32_t kConstant>
__global__ void add_constant_kernel(int32_t* dst, const int32_t* src, size_t length) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    dst[idx] = src[idx] + kConstant;
  }
}

constexpr size_t kBlockSize = 256;

// You can also use struct with static method as an alternative
template <int32_t kConstant>
void add_constant(tvm::ffi::TensorView dst, tvm::ffi::TensorView src) {
  using namespace host;

  // 1. Validate input tensors
  SymbolicSize N = {"num_elements"};
  SymbolicDevice device_;
  TensorMatcher({N})                  // 1D tensor, must be contiguous
      .with_dtype<int32_t>()          // must be int32
      .with_device<kDLCUDA>(device_)  // must be on CUDA device
      .verify(dst)                    // check tensor dst
      .verify(src);                   // check tensor src

  // 2. Extract required parameters, prepare for kernel launch
  const size_t num_elements = N.unwrap();
  const size_t grid_size = div_ceil(num_elements, kBlockSize);
  const DLDevice device = device_.unwrap();
  // some extra runtime checks using host::RuntimeCheck
  RuntimeCheck(num_elements > 0, "We only support non-empty tensors, got num_elements = ", num_elements);

  // 3. Launch the kernel. Error code will be automatically checked.
  LaunchKernel(grid_size, kBlockSize, device /*, dynamic_smem*/)(
      // kernel function
      add_constant_kernel<kConstant>,
      // kernel arguments
      static_cast<int32_t*>(dst.data_ptr()),
      static_cast<int32_t*>(src.data_ptr()),
      num_elements);
}

}  // namespace

```

### 步骤 2：创建 Python 接口

接下来，通过一个 Python 包装器暴露该 kernel。
在 [jit_kernel/add_constant.py](../../python/sglang/jit_kernel/add_constant.py) 创建一个新文件并暴露所需的接口。

```python
from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_add_constant_module(constant: int) -> Module:
    args = make_cpp_args(constant)  # pass all the template argument
    return load_jit(
        "add_constant",
        *args,
        cuda_files=["add_constant.cuh"],
        cuda_wrappers=[("add_constant", f"add_constant<{args}>")],
    )


def add_constant(src: torch.Tensor, constant: int) -> torch.Tensor:
    if not src.is_cuda:
        raise RuntimeError("src must be a CUDA tensor")
    if src.dtype != torch.int32:
        raise RuntimeError(f"Unsupported dtype {src.dtype}. Supported: int32")
    dst = torch.empty_like(src)
    module = _jit_add_constant_module(constant)
    module.add_constant(dst, src)
    return dst

```

保持 Python 包装器轻量，但仍要在分发之前验证基本的不变量，例如 device 和 dtype。在当前的 JIT/FFI 路径中，无效的 tensor 在启动前并不总是能被安全地拒绝。

### 步骤 3：使用你的 kernel

最后，像普通的 Python 函数一样导入并使用该 kernel：

```python
from sglang.jit_kernel.add_constant import add_constant
```

要查看一个完整的、可运行的示例，请参考 [test_add_constant.py](../../python/sglang/jit_kernel/tests/test_add_constant.py)。

## C++ Include 库参考

JIT kernel 框架在 `python/sglang/jit_kernel/include/sgl_kernel/` 中提供了一组可复用的 C++ 头文件。每个头文件都设计得轻量且自包含。下面是每个头文件及其关键 API 的摘要。

### 核心工具

| 头文件 | 命名空间 | 用途 |
|--------|-----------|---------|
| `utils.h` | `host` | 主机端基本工具：`RuntimeCheck`、`Panic`、`div_ceil`、`irange` |
| `utils.cuh` | `device` / `host` | 类型别名（`fp16_t`、`bf16_t`、...）、`SGL_DEVICE` 宏、PDL helpers、`LaunchKernel`、`RuntimeDeviceCheck` |
| `source_location.h` | (global) | 用于错误报告的可移植 `std::source_location` 包装器 |
| `runtime.cuh` | `host::runtime` | CUDA 运行时查询：`get_blocks_per_sm`、`get_sm_count`、`get_cc_major`、`get_runtime_version`、`get_available_dynamic_smem_per_block` |

### Tensor 验证

| 头文件 | 命名空间 | 用途 |
|--------|-----------|---------|
| `tensor.h` | `host` | `TensorMatcher`、`SymbolicSize`、`SymbolicDType`、`SymbolicDevice` |

### 数学与类型系统

| 头文件 | 命名空间 | 用途 |
|--------|-----------|---------|
| `math.cuh` | `device::math` | `max`、`min`、`abs`、`sqrt`、`rsqrt`、`exp`、`sin`、`cos`、常量 |
| `type.cuh` | (global) / `device` | `dtype_trait<T>`、`packed_t<T>`、`device::cast<To>(from)` |

### 内存访问

| 头文件 | 命名空间 | 用途 |
|--------|-----------|---------|
| `vec.cuh` | `device` | `AlignedVector<T, N>` - 向量化加载/存储（最高 128 位；256 位需要 Blackwell GPU） |
| `tile.cuh` | `device::tile` | `Memory<T>` - 协作式分块内存 I/O（thread/warp/CTA） |

### 并行原语

| 头文件 | 命名空间 | 用途 |
|--------|-----------|---------|
| `warp.cuh` | `device::warp` | 通过 `__shfl_xor_sync` 实现的 `reduce_sum`、`reduce_max` |
| `cta.cuh` | `device::cta` | 通过共享内存跨 warp 的 `reduce_max` |
| `atomic.cuh` | `device::atomic` | `max` - 原子 float max（CUDA + ROCm fallback） |

### 可复用的 Kernel 模板

| 头文件 | 命名空间 | 用途 |
|--------|-----------|---------|
| `impl/norm.cuh` | `host::norm` / `device::norm` | RMSNorm 构建块（warp 与 CTA 路径，`StorageType`） |
