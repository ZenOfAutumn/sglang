# 在 Kubernetes 上部署

本文档用于在 Kubernetes (K8S) 集群上部署基于 RoCE 网络的 SGLang 双节点推理服务。

[LeaderWorkerSet (LWS)](https://github.com/kubernetes-sigs/lws) 是一个 Kubernetes API,旨在解决 AI/ML 推理工作负载的常见部署模式。一个主要的使用场景是多主机/多节点分布式推理。

SGLang 也可以通过 LWS 在 Kubernetes 上部署,以实现分布式模型服务。

有关使用 LWS 在 Kubernetes 上部署 SGLang 的更多细节,请参阅本指南。

这里我们以部署 DeepSeek-R1 为例。

## 先决条件

1. 至少需要两个 Kubernetes 节点,每个节点配备两套 H20 系统和八张 GPU。

2. 确保你的 K8S 集群已正确安装 LWS。如果尚未设置,请遵循[安装说明](https://github.com/kubernetes-sigs/lws/blob/main/site/content/en/docs/installation/_index.md)。**注意:** 对于 LWS 版本 ≤0.5.x,你必须使用 Downward API 来获取 `LWS_WORKER_INDEX`,因为对该功能的原生支持是在 v0.6.0 中引入的。

## 基础示例

有关基础示例文档,请参阅 [Deploy Distributed Inference Service with SGLang and LWS on GPUs](https://github.com/kubernetes-sigs/lws/tree/main/docs/examples/sglang)。

然而,该文档仅涵盖了基本的 NCCL socket 模式。

在本节中,我们将做一些简单的修改,以使该设置适配 RDMA 场景。

## RDMA RoCE 案例

* 检查你的环境:

```bash
[root@node1 ~]# ibstatus
Infiniband device 'mlx5_bond_0' port 1 status:
        default gid:     fe80:0000:0000:0000:0225:9dff:fe64:c79a
        base lid:        0x0
        sm lid:          0x0
        state:           4: ACTIVE
        phys state:      5: LinkUp
        rate:            200 Gb/sec (2X NDR)
        link_layer:      Ethernet

Infiniband device 'mlx5_bond_1' port 1 status:
        default gid:     fe80:0000:0000:0000:0225:9dff:fe6e:c3ec
        base lid:        0x0
        sm lid:          0x0
        state:           4: ACTIVE
        phys state:      5: LinkUp
        rate:            200 Gb/sec (2X NDR)
        link_layer:      Ethernet

Infiniband device 'mlx5_bond_2' port 1 status:
        default gid:     fe80:0000:0000:0000:0225:9dff:fe73:0dd7
        base lid:        0x0
        sm lid:          0x0
        state:           4: ACTIVE
        phys state:      5: LinkUp
        rate:            200 Gb/sec (2X NDR)
        link_layer:      Ethernet

Infiniband device 'mlx5_bond_3' port 1 status:
        default gid:     fe80:0000:0000:0000:0225:9dff:fe36:f7ff
        base lid:        0x0
        sm lid:          0x0
        state:           4: ACTIVE
        phys state:      5: LinkUp
        rate:            200 Gb/sec (2X NDR)
        link_layer:      Ethernet
```

* 准备用于在 k8s 上部署的 `lws.yaml` 文件。

```yaml
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: sglang
spec:
  replicas: 1
  leaderWorkerTemplate:
    size: 2
    restartPolicy: RecreateGroupOnPodRestart
    leaderTemplate:
      metadata:
        labels:
          role: leader
      spec:
        dnsPolicy: ClusterFirstWithHostNet
        hostNetwork: true
        hostIPC: true
        containers:
          - name: sglang-leader
            image: sglang:latest
            securityContext:
              privileged: true
            env:
              - name: NCCL_IB_GID_INDEX
                value: "3"
            command:
              - python3
              - -m
              - sglang.launch_server
              - --model-path
              - /work/models
              - --mem-fraction-static
              -  "0.93"
              - --torch-compile-max-bs
              - "8"
              - --max-running-requests
              - "20"
              - --tp
              - "16" # Size of Tensor Parallelism
              - --dist-init-addr
              - $(LWS_LEADER_ADDRESS):20000
              - --nnodes
              - $(LWS_GROUP_SIZE)
              - --node-rank
              - $(LWS_WORKER_INDEX)
              - --trust-remote-code
              - --host
              - "0.0.0.0"
              - --port
              - "40000"
            resources:
              limits:
                nvidia.com/gpu: "8"
            ports:
              - containerPort: 40000
            readinessProbe:
              tcpSocket:
                port: 40000
              initialDelaySeconds: 15
              periodSeconds: 10
            volumeMounts:
              - mountPath: /dev/shm
                name: dshm
              - name: model
                mountPath: /work/models
              - name: ib
                mountPath: /dev/infiniband
        volumes:
          - name: dshm
            emptyDir:
              medium: Memory
          - name: model
            hostPath:
              path: '< your models dir >' # modify it according your models dir
          - name: ib
            hostPath:
              path: /dev/infiniband
    workerTemplate:
      spec:
        dnsPolicy: ClusterFirstWithHostNet
        hostNetwork: true
        hostIPC: true
        containers:
          - name: sglang-worker
            image: sglang:latest
            securityContext:
              privileged: true
            env:
            - name: NCCL_IB_GID_INDEX
              value: "3"
            command:
              - python3
              - -m
              - sglang.launch_server
              - --model-path
              - /work/models
              - --mem-fraction-static
              - "0.93"
              - --torch-compile-max-bs
              - "8"
              - --max-running-requests
              - "20"
              - --tp
              - "16" # Size of Tensor Parallelism
              - --dist-init-addr
              - $(LWS_LEADER_ADDRESS):20000
              - --nnodes
              - $(LWS_GROUP_SIZE)
              - --node-rank
              - $(LWS_WORKER_INDEX)
              - --trust-remote-code
            resources:
              limits:
                nvidia.com/gpu: "8"
            volumeMounts:
              - mountPath: /dev/shm
                name: dshm
              - name: model
                mountPath: /work/models
              - name: ib
                mountPath: /dev/infiniband
        volumes:
          - name: dshm
            emptyDir:
              medium: Memory
          - name: ib
            hostPath:
              path: /dev/infiniband
          - name: model
            hostPath:
              path: /data1/models/deepseek_v3_moe
---
apiVersion: v1
kind: Service
metadata:
  name: sglang-leader
spec:
  selector:
    leaderworkerset.sigs.k8s.io/name: sglang
    role: leader
  ports:
    - protocol: TCP
      port: 40000
      targetPort: 40000

```

* 然后使用 `kubectl apply -f lws.yaml`,你将得到如下输出。

```text
NAME           READY   STATUS    RESTARTS       AGE
sglang-0       0/1     Running   0              9s
sglang-0-1     1/1     Running   0              9s
```

等待 sglang leader(`sglang-0`)状态变为 1/1,这表示它已 `Ready`。

你可以使用命令 `kubectl logs -f sglang-0` 查看 leader 节点的日志。

成功后,你应该会看到类似这样的输出:

```text
[2025-02-17 05:27:24 TP1] Capture cuda graph end. Time elapsed: 84.89 s
[2025-02-17 05:27:24 TP6] max_total_num_tokens=712400, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=50, context_len=163840
[2025-02-17 05:27:24 TP0] max_total_num_tokens=712400, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=50, context_len=163840
[2025-02-17 05:27:24 TP7] max_total_num_tokens=712400, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=50, context_len=163840
[2025-02-17 05:27:24 TP3] max_total_num_tokens=712400, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=50, context_len=163840
[2025-02-17 05:27:24 TP2] max_total_num_tokens=712400, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=50, context_len=163840
[2025-02-17 05:27:24 TP4] max_total_num_tokens=712400, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=50, context_len=163840
[2025-02-17 05:27:24 TP1] max_total_num_tokens=712400, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=50, context_len=163840
[2025-02-17 05:27:24 TP5] max_total_num_tokens=712400, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=50, context_len=163840
[2025-02-17 05:27:24] INFO:     Started server process [1]
[2025-02-17 05:27:24] INFO:     Waiting for application startup.
[2025-02-17 05:27:24] INFO:     Application startup complete.
[2025-02-17 05:27:24] INFO:     Uvicorn running on http://0.0.0.0:40000 (Press CTRL+C to quit)
[2025-02-17 05:27:25] INFO:     127.0.0.1:48908 - "GET /get_model_info HTTP/1.1" 200 OK
[2025-02-17 05:27:25 TP0] Prefill batch. #new-seq: 1, #new-token: 7, #cached-token: 0, cache hit rate: 0.00%, token usage: 0.00, #running-req: 0, #queue-req: 0
[2025-02-17 05:27:32] INFO:     127.0.0.1:48924 - "POST /generate HTTP/1.1" 200 OK
[2025-02-17 05:27:32] The server is fired up and ready to roll!
```

如果它没有成功启动,请按照以下步骤检查是否存在任何遗留问题。谢谢!

### 调试

* 设置 `NCCL_DEBUG=TRACE` 来检查它是否是 NCCL 通信问题。

这应该可以解决大多数与 NCCL 相关的问题。

***注意:如果你发现 NCCL_DEBUG=TRACE 在容器环境中不起作用,但进程卡住或者你遇到难以诊断的问题,请尝试切换到不同的容器镜像。某些镜像可能无法正确处理标准错误输出。***

#### RoCE 场景

* 请确保集群环境中有可用的 RDMA 设备。
* 请确保集群中的节点配备了支持 RoCE 的 Mellanox 网卡。在本例中,我们使用 Mellanox ConnectX 5 型号的网卡,并已安装了正确的 OFED 驱动。如果没有,请参阅文档 [Install OFED Driver](https://docs.nvidia.com/networking/display/mlnxofedv461000/installing+mellanox+ofed) 来安装驱动。
* 检查你的环境:

  ```shell
  $ lspci -nn | grep Eth | grep Mellanox
  0000:7f:00.0 Ethernet controller [0200]: Mellanox Technologies MT43244 BlueField-3 integrated ConnectX-7 network controller [15b3:a2dc] (rev 01)
  0000:7f:00.1 Ethernet controller [0200]: Mellanox Technologies MT43244 BlueField-3 integrated ConnectX-7 network controller [15b3:a2dc] (rev 01)
  0000:c7:00.0 Ethernet controller [0200]: Mellanox Technologies MT43244 BlueField-3 integrated ConnectX-7 network controller [15b3:a2dc] (rev 01)
  0000:c7:00.1 Ethernet controller [0200]: Mellanox Technologies MT43244 BlueField-3 integrated ConnectX-7 network controller [15b3:a2dc] (rev 01)
  0001:08:00.0 Ethernet controller [0200]: Mellanox Technologies MT43244 BlueField-3 integrated ConnectX-7 network controller [15b3:a2dc] (rev 01)
  0001:08:00.1 Ethernet controller [0200]: Mellanox Technologies MT43244 BlueField-3 integrated ConnectX-7 network controller [15b3:a2dc] (rev 01)
  0001:a2:00.0 Ethernet controller [0200]: Mellanox Technologies MT43244 BlueField-3 integrated ConnectX-7 network controller [15b3:a2dc] (rev 01)
  0001:a2:00.1 Ethernet controller [0200]: Mellanox Technologies MT43244 BlueField-3 integrated ConnectX-7 network controller [15b3:a2dc] (rev 01)
  ```

* 检查 OFED 驱动:

  ```shell
  ofed_info -s
  OFED-internal-23.07-0.5.0:
  ```

* 显示 RDMA 链路状态并检查 IB 设备:

  ```shell
  $ rdma link show
  8/1: mlx5_bond_0/1: state ACTIVE physical_state LINK_UP netdev reth0
  9/1: mlx5_bond_1/1: state ACTIVE physical_state LINK_UP netdev reth2
  10/1: mlx5_bond_2/1: state ACTIVE physical_state LINK_UP netdev reth4
  11/1: mlx5_bond_3/1: state ACTIVE physical_state LINK_UP netdev reth6

  $ ibdev2netdev
  8/1: mlx5_bond_0/1: state ACTIVE physical_state LINK_UP netdev reth0
  9/1: mlx5_bond_1/1: state ACTIVE physical_state LINK_UP netdev reth2
  10/1: mlx5_bond_2/1: state ACTIVE physical_state LINK_UP netdev reth4
  11/1: mlx5_bond_3/1: state ACTIVE physical_state LINK_UP netdev reth6
  ```

* 在主机上测试 RoCE 网络速度:

  ```shell
  yum install qperf
  # for server：
  execute qperf
  # for client
  qperf -t 60 -cm1 <server_ip>   rc_rdma_write_bw
  ```

* 检查在你的容器中 RDMA 是否可访问:

  ```shell
  # ibv_devices
  # ibv_devinfo
  ```

## 成功的关键

* 在上面的 YAML 配置中,请注意 NCCL 环境变量。对于较旧版本的 NCCL,你应该检查 NCCL_IB_GID_INDEX 环境设置。
* NCCL_SOCKET_IFNAME 也很关键,但在容器化环境中,这通常不是问题。
* 在某些情况下,需要正确配置 GLOO_SOCKET_IFNAME。
* NCCL_DEBUG 对于故障排查至关重要,但我发现有时它在容器内不显示错误日志。这可能与你使用的 Docker 镜像有关。如有需要,你可能想尝试切换镜像。
* 避免使用基于 Ubuntu 18.04 的 Docker 镜像,因为它们往往存在兼容性问题。

## 遗留问题

* 在 Kubernetes、Docker 或 Containerd 环境中,我们使用 hostNetwork 来防止性能下降。
* 我们使用了特权(privileged)模式,这并不安全。此外,在容器化环境中,无法实现完全的 GPU 隔离。

## TODO

* 与 [k8s-rdma-shared-dev-plugin](https://github.com/Mellanox/k8s-rdma-shared-dev-plugin) 集成。
