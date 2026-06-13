# srt/speculative/cpp_ngram

## 目录用途
本目录为 N-gram 投机解码提供 N-gram 语料库（trie）的 Python 封装，对接底层 C++/JIT 内核实现高效的批量插入与匹配，供 `ngram_worker` 检索草稿 token。

## 文件清单
| 文件 | 说明 |
| --- | --- |
| `ngram_corpus.py` | `NgramCorpus` 类，封装底层 C++ ngram trie（经 `jit_kernel.ngram_corpus` 获取实现），提供 `batch_put`/`batch_get`/`synchronize`/`reset`、按 tree_mask 提取叶子路径等方法。 |

## 说明
本目录还包含非 .py 文件：`.clang-format`（底层 C++ 代码格式化配置）。
