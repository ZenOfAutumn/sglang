# srt/mem_cache/cpp_radix_tree

## 目录用途
基数树前缀缓存的 C++ 高性能实现及其 Python 绑定。Python 侧通过 `torch.utils.cpp_extension.load` 即时编译 C++ 源码，供 `radix_cache_cpp.py` 调用，以降低纯 Python 基数树的开销。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| radix_tree.py | 加载并封装 C++ 基数树扩展，导出 `RadixTreeCpp`、`TreeNodeCpp`、`IOHandle` 等供 Python 使用。 |

## 说明
本目录另含 C++/头文件（common.h、tree_v2.cpp/.h、tree_v2_binding.cpp、tree_v2_debug.cpp、tree_v2_impl.h、tree_v2_node.h），为基数树的 C++ 实现、调试与 pybind 绑定源码，由 radix_tree.py 在运行时编译。
