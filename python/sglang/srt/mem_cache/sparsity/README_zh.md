# srt/mem_cache/sparsity

## 目录用途
稀疏注意力（可检索 KV cache 压缩）框架。提供统一的稀疏算法接口、注意力后端适配器与运行时协调器，支持 Quest、DeepSeek NSA 等按页/按 token 的 TopK 检索方案，并通过工厂按配置创建协调器。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| __init__.py | 包导出入口，汇总算法、后端、核心协调器与工厂函数。 |
| factory.py | 稀疏协调器工厂：算法注册表、`create/get/register_sparse_coordinator`、`parse_hisparse_config` 配置解析。 |

## 子目录
| 子目录 | 说明 |
| --- | --- |
| algorithms | 稀疏注意力算法（基类、Quest、DeepSeek NSA）。 |
| backend | 注意力后端适配器（FlashAttention、NSA）。 |
| core | 稀疏协调器与请求状态跟踪等核心逻辑。 |
