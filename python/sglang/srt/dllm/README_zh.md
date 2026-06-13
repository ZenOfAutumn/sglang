# srt/dllm

## 目录用途
本目录提供扩散式语言模型（Diffusion LLM，dLLM）的推理支持。与自回归逐 token 生成不同，dLLM 以块为单位、通过多步去噪/揭示被 mask 的 token 来生成文本。本目录定义其配置、去噪算法及与调度器/请求的集成。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `config.py` | `DllmConfig`，封装算法名、算法参数、块大小、mask id、最大并发请求数等，并支持从 `ServerArgs` 构建。 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| `algorithm` | dLLM 去噪/揭示算法（基类与具体策略），详见其 README_zh.md。 |
| `mixin` | 将 dLLM 能力混入请求与调度器的 Mixin，详见其 README_zh.md。 |
