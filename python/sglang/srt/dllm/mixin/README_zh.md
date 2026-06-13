# srt/dllm/mixin

## 目录用途
本目录提供将扩散式 LLM（dLLM）能力混入现有运行时组件的 Mixin 类，使请求对象和调度器在不大幅改动核心代码的前提下，支持 dLLM 的分块、分阶段（staging/incoming 的 prefill/decode）处理流程。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `req.py` | 定义 `DllmReqPhase` 阶段枚举（staging/incoming 的 prefill/decode）与 `ReqDllmMixin`，为 `Req` 增加 dLLM 阶段、块偏移等初始化与状态。 |
| `scheduler.py` | `SchedulerDllmMixin` 与 `DllmManager`，为调度器注入 dLLM 初始化、批组织、KV 缓存释放与去噪推进等调度逻辑。 |
