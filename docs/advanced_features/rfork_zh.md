# R-Fork

R-Fork(Tensor Remote Fork)是一种新颖的权重加载方法,它利用高效的节点间 GPU 到 GPU 数据传输路径,以零拷贝(zero-copy)的方式将张量从一个正在运行的 SGLang 实例加载到一个新实例。它可以通过将模型权重加载时间从几分钟缩短到仅仅几秒钟,从而显著优化 SGLang 实例的启动时间。

要了解关于 R-Fork 的更多细节,请查看 **<a href=https://lmsys.org/blog/2025-12-10-rfork/> R-Fork 博客 </a>**

## 用法

| 参数     | 用法                                      |
|--------------|--------------------------------------------|
| load-format  | 设置为 `remote_instance` 以启用 R-Fork。 |
| remote-instance-weight-loader-backend | `nccl`、`transfer_engine` 或 `modelexpress`。默认为 `nccl`。 |
| remote-instance-weight-loader-seed-instance-ip | 将提供模型权重的种子实例(seed instance)的 IP 地址。由 `nccl` 和 `transfer_engine` 后端使用。 |
| remote-instance-weight-loader-seed-instance-service-port | 种子实例的 HTTP 服务器正在监听的端口。由 `nccl` 和 `transfer_engine` 后端使用。 |
| remote-instance-weight-loader-send-weights-group-ports | 种子实例上可用端口的列表,这些端口将用于在种子实例和客户端实例之间构建 NCCL 通信组。仅 `nccl` 后端需要。 |
| remote-instance-weight-loader-start-seed-via-transfer-engine | 设置后会启动一个支持 TransferEngine 作为后端的种子服务。当使用 `transfer_engine` 作为后端时,种子实例需要它。 |
| modelexpress-config | `modelexpress` 后端的 JSON 配置。键:`"url"`(必需,ModelExpress 服务器的 gRPC host:port)、`"model_name"`(可选,默认为 `--model-path`)、`"source"`(可选 bool,种子模式时为 `true`)。 |

### 使用 NCCL 作为后端

种子实例:
```shell
python -m sglang.launch_server [args]
```

客户端实例:
```shell
python -m sglang.launch_server [args] \
  --load-format remote_instance \
  --remote-instance-weight-loader-seed-instance-ip [seed_instance_ip] \
  --remote-instance-weight-loader-seed-instance-service-port [seed_instance_service_port] \
  --remote-instance-weight-loader-send-weights-group-ports [send_weights_nccl_group_ports_list]  \
  --remote-instance-weight-loader-backend nccl
```

### 使用 TransferEngine 作为后端

种子实例:
```shell
python -m sglang.launch_server [args] \
  --remote-instance-weight-loader-start-seed-via-transfer-engine
```

```shell
python -m sglang.launch_server [args] \
  --load-format remote_instance \
  --remote-instance-weight-loader-seed-instance-ip [seed_instance_ip] \
  --remote-instance-weight-loader-seed-instance-service-port [seed_instance_service_port] \
  --remote-instance-weight-loader-backend transfer_engine
```

### 使用 ModelExpress 作为后端

[ModelExpress](https://github.com/ai-dynamo/modelexpress) 是一个协调服务,用于管理 P2P 权重传输的元数据。它通过提供一个集中式的注册表(种子向其发布,客户端从其发现)来消除直接配置种子 IP/端口的需要。在底层,它使用 TransferEngine(Mooncake)进行实际的 RDMA 数据传输。

需要一个正在运行的 ModelExpress 服务器。有关设置说明,请参见 [ModelExpress 文档](https://github.com/ai-dynamo/modelexpress)。

种子实例:
```shell
python -m sglang.launch_server [args] \
  --modelexpress-config '{"url": "[modelexpress_grpc_host:port]", "model_name": "[model_name]", "source": true}'
```

客户端实例:
```shell
python -m sglang.launch_server [args] \
  --load-format remote_instance \
  --remote-instance-weight-loader-backend modelexpress \
  --modelexpress-config '{"url": "[modelexpress_grpc_host:port]", "model_name": "[model_name]"}'
```

种子将其 TransferEngine 会话 ID 和张量布局发布到 ModelExpress。客户端查询 ModelExpress 以发现种子,然后通过 RDMA 直接拉取权重。这实现了无需硬编码 IP 的动态种子发现,并支持通过单个 ModelExpress 实例服务多个模型。
