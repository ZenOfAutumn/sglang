# srt/distributed/device_communicators

## 目录用途
该目录为 `parallel_state` 中的进程组协调器提供具体的设备/后端通信器实现。它针对不同硬件平台（NVIDIA CUDA、AMD HIP、华为 NPU、Intel XPU、Habana HPU 等）和不同通信路径（自定义 all-reduce、PyNCCL、MSCCL++、PyTorch 对称内存、共享内存广播、Mooncake 跨节点传输）封装统一接口，使上层无需关心底层硬件即可完成高性能集合通信。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `all_reduce_utils.py` | 通信相关常量配置，如 PyTorch 对称内存 all-reduce 在不同 SM 架构/卡数下的最大消息尺寸表。 |
| `cuda_wrapper.py` | 纯 Python ctypes 封装的 cudart 库，免编译直接调用少量 CUDA Runtime 函数（如 IPC 内存句柄）。 |
| `custom_all_reduce.py` | 自定义 all-reduce 通信器，基于 sgl_kernel 在 NVLink 互联下提供低延迟 all-reduce，支持 CUDA Graph。 |
| `custom_all_reduce_ops.py` | 自定义 all-reduce 的底层算子绑定层，封装 `sgl_kernel.allreduce`，并探测 CUSTOM/QUICK/MSCCLPP AR 可用性。 |
| `custom_all_reduce_utils.py` | 自定义 all-reduce 的辅助工具：P2P/NVLink 可用性探测、弱连续性判断、同节点检测等。 |
| `custom_all_reduce_v2.py` | 新版自定义 all-reduce，基于 jit_kernel 选择 one-shot push/pull 等算法模式（含尺寸阈值配置）。 |
| `hpu_communicator.py` | Habana HPU 平台通信器，封装 all-reduce/all-gather 并处理 HPU lazy collectives 相关问题。 |
| `mooncake_transfer_engine.py` | Mooncake 跨节点传输引擎封装，含 IB 设备解析与全局引擎实例管理，用于 PD 分离等远程数据传输。 |
| `npu_communicator.py` | 华为昇腾 NPU 平台通信器，封装 all-reduce/all-gather/gather 等集合操作。 |
| `pymscclpp.py` | MSCCL++ 通信器，基于 one-shot LL 算法实现单/多节点低延迟 all-reduce。 |
| `pynccl.py` | PyNCCL 通信器 `PyNcclCommunicator`，纯 Python 调用 NCCL 库，可在 CUDA Graph 捕获下安全使用。 |
| `pynccl_allocator.py` | NCCL 显存分配器，注册 NCCL 窗口内存（ncclCommWindowRegister）以支持对称内存/注册缓冲区。 |
| `pynccl_wrapper.py` | 纯 Python ctypes 封装的 NCCL 库，可通过环境变量切换 NCCL 版本，配合 CUDA Graph 使用。 |
| `quick_all_reduce.py` | AMD ROCm 平台的 Quick all-reduce 实现，针对特定 GPU 架构提供快速 all-reduce。 |
| `shm_broadcast.py` | 基于共享内存环形缓冲区 `ShmRingBuffer` + ZMQ 的进程间广播，用于同节点元数据/张量字典广播。 |
| `torch_symm_mem.py` | PyTorch 对称内存集合通信封装 `TorchSymmMemCommunicator`，提供基于 symmetric memory 的 all-reduce。 |
| `xpu_communicator.py` | Intel XPU 平台通信器，封装 all-reduce/gather（gather 用 all_gather 实现以兼容 Ray）。 |
