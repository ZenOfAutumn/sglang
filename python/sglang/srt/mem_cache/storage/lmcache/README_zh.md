# srt/mem_cache/storage/lmcache

## 目录用途
集成 LMCache 的分层基数缓存后端。通过 LMCache 的 SGLang 适配器（`LMCacheLayerwiseConnector` 等）实现逐层 KV cache 的存取与卸载，需安装 `lmcache`。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| lmc_radix_cache.py | `LMCRadixCache`：在 `RadixCache` 基础上对接 LMCache 进行逐层 KV 存取的基数缓存。 |
| unit_test.py | LMCache 集成的单元测试，验证 load/store metadata 等流程。 |

## 说明
本目录另含 example_config.yaml，为 LMCache 的示例配置文件，供单元测试与使用参考。
