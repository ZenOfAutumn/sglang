# srt/debug_utils/schedule_simulator/schedulers

## 目录用途
本目录定义调度模拟器中单 GPU 的调度策略，决定在每个 step 如何从待运行队列中选取请求进入运行、以及在超出 token 上限时如何驱逐请求。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 包初始化，导出 `SchedulerPolicy`、`FIFOScheduler`。 |
| `base.py` | 调度策略抽象基类 `SchedulerPolicy`，定义 `schedule` 接口。 |
| `fifo_scheduler.py` | FIFO 调度：超限时从队尾驱逐，再按顺序把可容纳的待运行请求启动。 |
