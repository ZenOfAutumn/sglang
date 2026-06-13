# srt/disaggregation

## 目录用途
本目录实现 SGLang 的 PD 分离（Prefill/Decode Disaggregation）架构：把请求的预填充（prefill）与解码（decode）阶段拆分到不同的服务实例上运行，并通过可插拔的 KV 传输后端在两端之间搬运 KV cache。该目录顶层放置 prefill/decode 两端的请求生命周期调度逻辑、KV 传输的通用工具与抽象，以及面向多模态的 EPD（Encode-Prefill-Decode）编码服务；各 KV 传输后端实现位于子目录中。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `decode.py` | Decode 端请求生命周期管理：预分配队列、传输队列、等待队列与运行批，以及 Decode 端调度器 Mixin、Req-to-token 池等。 |
| `decode_kvcache_offload_manager.py` | `DecodeKVCacheOffloadManager`，管理 Decode 端 KV cache 的卸载（offload）生命周期与操作。 |
| `decode_schedule_batch_mixin.py` | `ScheduleBatchDisaggregationDecodeMixin`，为 ScheduleBatch 提供预构建 extend 批（跳过 prefill 前向、仅填充元数据）的能力。 |
| `prefill.py` | Prefill 端请求生命周期管理：bootstrap 队列、等待队列、在途（inflight）队列，以及 Prefill 端调度器 Mixin。 |
| `utils.py` | PD 分离通用工具：`DisaggregationMode`/`TransferBackend`/`KVClassType` 枚举、元数据缓冲与索引分配器、轮询归约、KV 页索引换算、后端类工厂 `get_kv_class` 等。 |
| `kv_events.py` | KV cache 事件定义与发布：BlockStored/BlockRemoved 等事件、ZMQ 事件发布器及发布器工厂。 |
| `encode_server.py` | 多模态 EPD 编码服务：`MMEncoder` 编码器、编码进程启动与基于 FastAPI 的服务入口。 |
| `encode_receiver.py` | 编码结果接收端：嵌入数据结构、HTTP/gRPC 多模态接收器及其工厂 `create_mm_receiver`。 |
| `encode_grpc_server.py` | 基于 gRPC 的编码服务实现，用于多模态输入的编码（EPD 模式）。 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| `base` | KV 传输的抽象基类（Manager/Sender/Receiver/BootstrapServer 与 KVArgs/KVPoll）。 |
| `common` | 各后端共享的通用实现：CommonKV 系列、bootstrap 服务、异构 TP 暂存缓冲与工具。 |
| `mooncake` | 基于 Mooncake Transfer Engine 的 KV 传输后端。 |
| `nixl` | 基于 NIXL 的 KV 传输后端。 |
| `mori` | 基于 MORI IO 引擎的 KV 传输后端。 |
| `ascend` | 昇腾（Ascend）平台的 KV 传输后端，复用 Mooncake 实现。 |
| `fake` | 仅用于 warmup、不做真实 KV 传输的伪后端。 |
