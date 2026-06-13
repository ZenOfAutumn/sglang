# srt/debug_utils/schedule_simulator/routers

## 目录用途
本目录定义调度模拟器的路由策略，决定每个到来的请求被分配到哪个 GPU/引擎，用于对比不同路由方式的负载分布效果。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 包初始化，导出 `RouterPolicy`、`RandomRouter`、`RoundRobinRouter`、`StickyRouter`。 |
| `base.py` | 路由策略抽象基类 `RouterPolicy`，定义 `route` 接口。 |
| `random_router.py` | 随机路由：将请求随机分配到某个 GPU。 |
| `round_robin_router.py` | 轮询路由：按计数器在各 GPU 间轮流分配。 |
| `sticky_router.py` | 粘滞路由：同一 group 的请求固定路由到同一 GPU（无 group 则随机）。 |
