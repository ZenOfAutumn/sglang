# 环境变量

SGLang 支持各种与 Ascend NPU 相关的环境变量,可用于配置其运行时行为。
本文档提供了一份常用环境变量列表,并力求持续更新。

## 在 SGLang 中直接使用

| Environment Variable                             | 描述                                                                                                                                                          | Default Value |
|--------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| `SGLANG_NPU_USE_MLAPO`                           | 在 MLA 模型的注意力<br/>预处理阶段采用 `MLAPO` 融合算子。                                                                 | `false`       |
| `SGLANG_USE_FIA_NZ`                              | 将 KV Cache 重排为 FIA NZ 格式。<br/> `SGLANG_USE_FIA_NZ` 必须与 `SGLANG_NPU_USE_MLAPO` 一起启用                                                   | `false`       |
| `SGLANG_NPU_USE_MULTI_STREAM`                    | 在 DeepSeek 模型中启用共享专家(shared experts)<br/>与路由专家(routing experts)的双流计算。<br/>在 DeepSeek NSA Indexer 中启用双流计算。 | `false`       |
| `SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT`           | 禁止将模型权重张量转换为特定的 NPU <br/> ACL 格式。                                                                                        | `false`       |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | 每个 rank 上分发(dispatch)的最大 token 数。                                                                                       | `128`         |

## 在 DeepEP Ascend 中使用

| Environment Variable                      | 描述                                                                                                                   | Default Value |
|-------------------------------------------|------------------------------------------------------------------------------------------------------------------------|---------------|
| `DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS` | 在 dispatch 阶段启用 ant-moving 功能。表示<br/>每个 rank 上每轮传输的 token 数。 | `8192`        |
| `DEEPEP_NORMAL_LONG_SEQ_ROUND`            | 在 dispatch 阶段启用 ant-moving 功能。表示<br/>每个 rank 上传输的轮数。           | `1`           |
| `DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ`   | 在 combine 阶段启用 ant-moving 功能。<br/> 值为 `0` 表示禁用。                                       | `0`           |
| `MOE_ENABLE_TOPK_NEG_ONE`                 | 当 DEEPEP 要处理的 expert ID 中包含 -1 时,<br/>需要启用此项。                                    | `0`           |
| `DEEP_NORMAL_MODE_USE_INT8_QUANT`         | 在 dispatch 算子中将 x 量化为 int8 并返回 (tensor, scales)。                                                 | `0`           |

## 其他

| Environment Variable     | 描述                                                                                                                                                                                                                                                                | Default Value |
|--------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| `TASK_QUEUE_ENABLE`      | 用于控制关于 task_queue 算子的<br/>分发队列优化级别。[Detail](https://www.hiascend.com/document/detail/zh/Pytorch/730/comref/Envvariables/docs/zh/environment_variable_reference/TASK_QUEUE_ENABLE.md)                         | `1`           |
| `INF_NAN_MODE_ENABLE`    | 控制芯片使用饱和模式还是 INF_NAN 模式。[Detail](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha001/apiref/envref/envref_07_0056.html)                                                                                   | `1`           |
| `STREAMS_PER_DEVICE`     | 配置流池(stream pool)的最大流数量。[Detail](https://www.hiascend.com/document/detail/zh/Pytorch/720/comref/Envvariables/Envir_041.html)                                                                                                         | `32`          |
| `PYTORCH_NPU_ALLOC_CONF` | 控制缓存分配器(cache allocator)的行为。<br/>该变量会改变内存使用,并可能导致性能波动。[Detail](https://www.hiascend.com/document/detail/zh/Pytorch/700/comref/Envvariables/Envir_012.html)                                         |               |
| `ASCEND_MF_STORE_URL`    | PD 分离时 MemFabric 中 config store 的地址,<br/>通常设置为 P 主节点的 IP 地址,<br/> 并附带任意端口号。                                                                                                     |               |
| `ASCEND_LAUNCH_BLOCKING` | 控制算子执行期间是否启用同步模式。[Detail](https://www.hiascend.com/document/detail/zh/Pytorch/710/comref/Envvariables/Envir_006.html)                                                                                               | `0`           |
| `HCCL_OP_EXPANSION_MODE` | 配置通信算法调度的展开位置。[Detail](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha001/apiref/envref/envref_07_0094.html)                                                                         |               |
| `HCCL_BUFFSIZE`          | 控制两个 NPU 之间共享数据的缓冲区大小。<br/>单位为 MB,值必须大于或等于 1。[Detail](https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/ptmoddevg/trainingmigrguide/performance_tuning_0047.html) | `200`         |
| `HCCL_SOCKET_IFNAME`     | 配置 HCCL 初始化期间 Host 使用的<br/>网卡名称。[Detail](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/apiref/envvar/envref_07_0075.html)                                                                     |               |
| `GLOO_SOCKET_IFNAME`     | 配置 GLOO 通信的网络接口名称。                                                                                                                                                                              |               |
