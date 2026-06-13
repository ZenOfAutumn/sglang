# srt/debug_utils/schedule_simulator/data_source

## 目录用途
本目录为调度模拟器提供请求数据源，既可从真实请求日志加载已完成请求，也可合成随机请求或带共享系统提示的分组（GSP）请求。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 包初始化，导出 `load_from_request_logger`、`generate_random_requests`、`generate_gsp_requests`。 |
| `data_loader.py` | 从 request_logger JSON（逐行）解析 `request.finished` 事件，构造 `SimRequest` 列表。 |
| `data_synthesis.py` | 合成请求：随机输入/输出长度，或带共享系统提示的分组请求（GSP）。 |
