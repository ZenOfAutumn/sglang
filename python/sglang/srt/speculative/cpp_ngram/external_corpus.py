"""外部 N-gram 语料文件（JSONL）的读取与分块。

对应 CLI 参数 `--speculative-ngram-external-corpus-path`。
用途是「预热」：把业务上高频复现的文本（FAQ 模板、代码片段、固定报文格式）
提前灌进后缀自动机（SAM），让第一个请求就能享受高命中率，
而不必等运行时慢慢累积 N-gram。

输入文件格式：JSONL，**每行是一个 JSON 字符串**（不是对象）。
    "How do I reset my password?"
    "To reset your password, go to Settings > Security."
    "def fibonacci(n):\\n    if n <= 1:\\n        return n"

本模块把这些文档 tokenize，用分隔符拼接，再切成固定大小的块流式产出，
避免一次性把整个语料读进内存。
"""

import json
from collections.abc import Iterator
from pathlib import Path

# 中译：必须与 suffix_automaton.h 中的 SuffixAutomaton::kSeparatorToken 保持一致。
#
# 取 int32 最小值（-2147483648）作为文档分隔符：正常 token id 都是非负的，
# 用一个不可能与真实 token 冲突的值，保证 SAM 不会把「上一篇文档的结尾」
# 和「下一篇文档的开头」误当成一个连续的 n-gram 来匹配。
SEPARATOR_TOKEN = -(2**31)

# 中译：把 tokenize 后的文档流式送进 SAM 时使用的默认分块大小。
# 4096 是内存占用与调用开销之间的折中：太小则跨语言调用次数过多，
# 太大则单次驻留内存高。
DEFAULT_CHUNK_SIZE = 4096


def iter_external_corpus_chunks(
    path: str, tokenizer, max_tokens: int, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> Iterator[list[int]]:
    """中译：从 JSONL 语料文件中读取文档，切分并逐块产出固定大小的 token 块。

    参数：
        path: JSONL 文件路径，例 "/data/faq_corpus.jsonl"
        tokenizer: HF tokenizer 实例，用于把文本编码成 token id
        max_tokens: token 总数上限，超过即报错。对应 CLI 的
            --speculative-ngram-external-corpus-max-tokens（默认 1000 万）
        chunk_size: 每次产出的块大小，默认 4096

    产出（yield）：
        List[int]，长度 ≤ chunk_size 的 token id 块。

    完整数值示例（假设 chunk_size=4，max_tokens=100）：
        文件内容：
            "hello world"          → tokenize 得 [15496, 995]
            "goodbye world"        → tokenize 得 [11274, 16390, 995]

        处理第 1 篇：has_previous_doc=False → 无分隔符
            total_tokens: 0 → 2
            产出 [15496, 995]                    （长度 2 < chunk_size，一块出完）

        处理第 2 篇：has_previous_doc=True → 前置分隔符
            separator_cost=1，total_tokens: 2 → 2+1+3 = 6
            token_ids 变为 [-2147483648, 11274, 16390, 995]
            产出 [-2147483648, 11274, 16390, 995]  （长度 4 = chunk_size）

        若第 2 篇有 10 个 token，则加分隔符后 11 个，会被切成
        3 块产出：[4个]、[4个]、[3个]

    异常：
        ValueError: 文件不存在 / 未提供 tokenizer / max_tokens ≤ 0 /
            某行不是合法 JSON / 某行不是 JSON 字符串 / 超出 token 上限。
        所有异常都在启动阶段抛出，属于「快速失败」——
        宁可起不来，也不要带着半份语料上线。
    """
    corpus_path = Path(path)
    if not corpus_path.is_file():
        raise ValueError(f"External ngram corpus path does not exist: {path}")
    if tokenizer is None:
        raise ValueError("A tokenizer is required to load an external ngram corpus.")
    if max_tokens <= 0:
        raise ValueError("External ngram corpus max tokens must be positive.")

    total_tokens = 0
    # 标记是否已产出过文档：决定当前文档前面要不要插入分隔符
    has_previous_doc = False
    with corpus_path.open("r", encoding="utf-8") as f:
        # line_no 从 1 开始，报错信息里能直接对应到文件行号
        for line_no, line in enumerate(f, start=1):
            # 跳过空行，允许语料文件里有空白行做分节
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in external ngram corpus at line {line_no}: {e.msg}"
                ) from e

            if not isinstance(record, str):
                raise ValueError(
                    "Invalid external ngram corpus record at line "
                    f"{line_no}: expected a JSON string."
                )

            # add_special_tokens=False：语料是「素材」不是「对话」，
            # 不能掺入 BOS/EOS，否则会污染 n-gram 的统计分布。
            token_ids = list(tokenizer.encode(record, add_special_tokens=False))
            # 空字符串编码后为空，跳过且不消耗分隔符预算
            if not token_ids:
                continue

            # 预算核算要把分隔符本身也算进去（第一篇文档前面不加分隔符）
            separator_cost = 1 if has_previous_doc else 0
            next_total_tokens = total_tokens + separator_cost + len(token_ids)
            # 先检查再累加：超限时 total_tokens 仍是「已成功加载」的数量，
            # 报错信息因此能准确告诉用户在哪一行、加载了多少之后爆的。
            if next_total_tokens > max_tokens:
                raise ValueError(
                    "External ngram corpus exceeds the configured token limit "
                    f"({max_tokens}) at line {line_no} after loading "
                    f"{total_tokens} tokens."
                )
            total_tokens = next_total_tokens

            # 在文档之间插入分隔符，切断跨文档的虚假 n-gram
            if has_previous_doc:
                token_ids = [SEPARATOR_TOKEN] + token_ids
            # 按 chunk_size 切块流式产出，最后一块可能不满
            for i in range(0, len(token_ids), chunk_size):
                yield token_ids[i : i + chunk_size]
            has_previous_doc = True
