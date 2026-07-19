# Mesh 模式部署指南

本文档说明 SGL Model Gateway 在 **Mesh（多实例高可用）模式** 下的部署方式。Mesh 模式让多个 Gateway 实例组成一个集群，通过 Gossip 协议 + CRDT 实现 worker 状态、路由策略、全局限流计数等信息的**最终一致同步**，从而支持多副本水平扩展与高可用（HA）。

---

## 1. Mesh 模式解决什么问题

单实例 Gateway 的路由决策（缓存亲和、负载视图、限流计数）只保存在本进程内存中。当部署多个 Gateway 副本时，会出现以下问题：

- 每个副本各自维护一份 worker 负载视图，路由决策不一致；
- 全局限流（`limit_per_second`）无法跨副本共享计数，实际限流值被放大 N 倍；
- 缓存亲和（cache-aware）路由在副本间无法协同，命中率下降。

Mesh 模式通过在 Gateway 副本之间建立一张互联网络（mesh），使用 CRDT 同步以下状态：

| 同步内容 | 说明 |
| --- | --- |
| Membership（成员） | 集群中各节点的地址、状态、版本 |
| Worker 状态 | 各 worker 的注册信息与负载 |
| Policy 状态 | 路由策略相关的共享状态 |
| App 配置 | 通过 `/ha/config` 写入的键值配置 |
| 全局限流窗口 | 跨副本共享的限流计数器 |

> 底层由 `smg-mesh` crate 提供，使用 `crdts`（无冲突复制数据类型）保证多实例状态最终一致。

---

## 2. 核心概念与端口

一个启用 Mesh 的 Gateway 实例会同时监听两类端口：

| 端口 | 用途 | 相关参数 |
| --- | --- | --- |
| HTTP 服务端口 | 对外提供 OpenAI 兼容 API + `/ha/*` 管理 API | `--port`（默认 30000） |
| Mesh 互联端口 | 节点间 Gossip / 状态同步 | `--mesh-port`（默认 39527） |
| Metrics 端口 | Prometheus 指标 | `--prometheus-port`（默认 29000） |

节点通过 **peer URL**（其它节点的 `host:mesh_port`）互相发现并加入集群。

---

## 3. 启动参数（CLI）

Mesh 相关参数定义在 `src/main.rs`：

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--enable-mesh` | flag | `false` | 启用 Mesh 服务 |
| `--mesh-server-name` | string | 随机 `Mesh_xxxx` | 当前节点在集群中的唯一名字，建议显式设置为 Pod 名 |
| `--mesh-host` | string | `0.0.0.0` | Mesh 端口绑定地址 |
| `--mesh-port` | u16 | `39527` | Mesh 互联端口 |
| `--mesh-peer-urls` | list | 空 | 初始 peer 地址列表（`host:port`），用于加入已有集群 |

> 说明：`--mesh-peer-urls` 目前取列表中的**第一个**地址作为初始 peer（`init_peer`）进行 join。首个启动的种子节点可不填 peer，后续节点填种子节点地址即可。

### 3.1 手动组网示例（两节点）

节点 A（种子节点）：

```bash
sgl-model-gateway \
  --port 30000 \
  --worker-urls http://worker1:8000 http://worker2:8000 \
  --enable-mesh \
  --mesh-server-name node-a \
  --mesh-host 0.0.0.0 \
  --mesh-port 39527
```

节点 B（加入 A）：

```bash
sgl-model-gateway \
  --port 30000 \
  --worker-urls http://worker1:8000 http://worker2:8000 \
  --enable-mesh \
  --mesh-server-name node-b \
  --mesh-host 0.0.0.0 \
  --mesh-port 39527 \
  --mesh-peer-urls <node-a-ip>:39527
```

---

## 4. Kubernetes 部署（推荐）

在 K8s 中，Gateway 副本数不固定、IP 动态变化，因此**不建议手动写死 peer 列表**，而应使用**服务发现自动发现 Router 节点**并加入 mesh。

### 4.1 工作原理

1. Gateway 开启 `--service-discovery` 与 `--enable-mesh`；
2. 服务发现除了发现 worker 外，还会用 `router_selector` 这个 label 选择器匹配**其它 Gateway（Router）Pod**；
3. 对每个匹配到的 Router Pod，从其 annotation（默认键 `sglang.ai/ha-port`）读取 mesh 端口，构造节点地址 `pod_ip:mesh_port`；
4. 将这些节点写入 mesh 的 `ClusterState`，实现副本自动组网；
5. Pod 删除时对应节点被标记为 `Down`，不健康时标记为 `Suspected`。

> 关键代码：`src/service_discovery.rs` 的 `start_router_discovery`，以及 `PodInfo::from_pod` 中对 `is_router` / `mesh_port` 的解析。
>
> ⚠️ 注意：`router_selector` 与 `router_mesh_port_annotation` **只能通过配置文件设置**（CLI 中 `router_selector` 被初始化为空）。若不配置 `router_selector`，router 自动发现不会启用，mesh 需要退回手动 peer 方式组网。

### 4.2 RBAC（服务发现所需权限）

服务发现需要 watch Pod 的权限，参考 `e2e_test/k8s_integration/manifests/rbac.yaml`：

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: smg-gateway
  namespace: smg-test
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: smg-gateway
  namespace: smg-test
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: smg-gateway
  namespace: smg-test
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: smg-gateway
subjects:
  - kind: ServiceAccount
    name: smg-gateway
    namespace: smg-test
```

### 4.3 Deployment（多副本 + Mesh + 自动发现）

以下在 `e2e_test/k8s_integration/manifests/gateway.yaml` 基础上扩展 mesh：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: smg-gateway
  namespace: smg-test
spec:
  replicas: 3                       # 多副本
  selector:
    matchLabels:
      app: smg-gateway
  template:
    metadata:
      labels:
        app: smg-gateway            # 供 router_selector 匹配
      annotations:
        sglang.ai/ha-port: "39527"  # 供发现方读取 mesh 端口
    spec:
      serviceAccountName: smg-gateway
      containers:
        - name: gateway
          image: smg-gateway:latest
          args:
            # ---- Worker 服务发现 ----
            - "--service-discovery"
            - "--selector"
            - "app=fake-worker"
            - "--service-discovery-port"
            - "8000"
            - "--service-discovery-namespace"
            - "smg-test"
            # ---- 基础端口 ----
            - "--port"
            - "30000"
            - "--prometheus-port"
            - "29000"
            # ---- Mesh ----
            - "--enable-mesh"
            - "--mesh-server-name"
            - "$(POD_NAME)"          # 使用 Pod 名做节点名，保证唯一
            - "--mesh-host"
            - "0.0.0.0"
            - "--mesh-port"
            - "39527"
          env:
            - name: POD_NAME
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name
          ports:
            - containerPort: 30000
              name: http
            - containerPort: 29000
              name: metrics
            - containerPort: 39527
              name: mesh          # 暴露 mesh 端口供副本互联
          readinessProbe:
            httpGet:
              path: /liveness
              port: 30000
            initialDelaySeconds: 3
            periodSeconds: 3
          livenessProbe:
            httpGet:
              path: /liveness
              port: 30000
            initialDelaySeconds: 5
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: smg-gateway
  namespace: smg-test
spec:
  type: NodePort
  selector:
    app: smg-gateway
  ports:
    - name: http
      port: 30000
      targetPort: 30000
    - name: metrics
      port: 29000
      targetPort: 29000
```

### 4.4 通过配置文件启用 Router 自动发现

由于 `router_selector` 只能来自配置文件，需要向 Gateway 传入包含如下 `discovery` 段的配置（`DiscoveryConfig`）：

```json
{
  "discovery": {
    "enabled": true,
    "namespace": "smg-test",
    "selector": { "app": "fake-worker" },
    "router_selector": { "app": "smg-gateway" },
    "router_mesh_port_annotation": "sglang.ai/ha-port"
  }
}
```

- `router_selector`：匹配其它 Gateway Pod 的 label（需与 Deployment 的 `template.metadata.labels` 一致）；
- `router_mesh_port_annotation`：从 Pod annotation 读取 mesh 端口的键，默认 `sglang.ai/ha-port`，需与 Pod annotation 一致。

> 只要 `router_selector` 非空且 Mesh 已启用，Gateway 就会启动 router 发现任务，自动把其它副本加入 mesh 集群；否则该任务会被跳过。

---

## 5. Mesh 管理 API

启用 Mesh 后，Gateway 暴露一组 `/ha/*` 管理接口（定义于 `src/routers/mesh/handlers.rs`，路由注册于 `src/server.rs`）。这些接口受控制面认证中间件保护。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/ha/status` | 集群状态（节点列表、地址、状态、版本） |
| GET | `/ha/health` | Mesh 健康检查（节点名、集群大小） |
| GET | `/ha/workers` | 所有 worker 同步状态 |
| GET | `/ha/workers/{worker_id}` | 指定 worker 状态 |
| GET | `/ha/policies` | 所有 policy 同步状态 |
| GET | `/ha/policies/{model_id}` | 指定 model 的 policy 状态 |
| GET | `/ha/config/{key}` | 读取 App 配置（hex 编码返回） |
| POST | `/ha/config` | 写入 App 配置（`{"key","value"}`，value 为 hex 字符串） |
| GET | `/ha/rate-limit` | 读取全局限流配置 |
| POST | `/ha/rate-limit` | 设置全局限流（`{"limit_per_second"}`） |
| GET | `/ha/rate-limit/stats` | 全局限流统计（配置值、当前计数、剩余） |
| POST | `/ha/shutdown` | 触发优雅下线（graceful shutdown） |

### 5.1 常用操作示例

查看集群状态：

```bash
curl http://<gateway>:30000/ha/status
```

设置全局限流（跨副本共享计数）：

```bash
curl -X POST http://<gateway>:30000/ha/rate-limit \
  -H 'Content-Type: application/json' \
  -d '{"limit_per_second": 1000}'
```

优雅下线某个副本：

```bash
curl -X POST http://<gateway>:30000/ha/shutdown
```

---

## 6. 优雅下线与滚动升级

- 通过 `--shutdown-grace-period-secs`（默认 180）控制优雅退出的宽限时长；
- 可先调用 `POST /ha/shutdown` 让节点从集群中平滑摘除，再由编排系统回收 Pod；
- K8s 滚动升级（`replicas > 1`）时，配合上面的 router 自动发现，新副本会自动加入 mesh，被删除副本会被标记为 `Down`，无需人工干预 peer 列表。

---

## 7. 部署清单（Checklist）

- [ ] 每个副本设置**唯一**的 `--mesh-server-name`（推荐用 `$(POD_NAME)`）
- [ ] 所有副本 `--mesh-port` 一致，并在容器 / Service 中暴露
- [ ] K8s 环境配置 RBAC，授予 Pod `get/list/watch` 权限
- [ ] Deployment 打上供 `router_selector` 匹配的 label，并加上 `sglang.ai/ha-port` annotation
- [ ] 通过配置文件设置 `discovery.router_selector`（CLI 不支持）
- [ ] 非 K8s / 无服务发现场景，用 `--mesh-peer-urls` 指向种子节点手动组网
- [ ] 需要跨副本一致的全局限流时，通过 `/ha/rate-limit` 统一设置

---

## 8. 相关代码索引

| 功能 | 文件 |
| --- | --- |
| CLI 参数与 `MeshServerConfig` 构建 | `src/main.rs` |
| Mesh 初始化、路由注册、状态存储 | `src/server.rs` |
| Router 自动发现、Pod 解析 | `src/service_discovery.rs` |
| `/ha/*` 管理 API | `src/routers/mesh/handlers.rs` |
| 发现配置结构体 `DiscoveryConfig` | `src/config/types.rs` |
| K8s 部署 / RBAC 参考 | `e2e_test/k8s_integration/manifests/` |

