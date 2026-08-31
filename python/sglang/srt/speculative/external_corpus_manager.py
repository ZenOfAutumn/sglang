"""中译：管理 NGRAM 投机解码所用的外部 SAM（后缀自动机）语料。

负责 add/remove/list 操作以及异步后台加载。
由 Scheduler 使用——它不是 mixin，而是一个独立的管理器对象。

--------------------------------------------------------------------------
为什么需要异步加载
--------------------------------------------------------------------------
加载一份几十万 token 的语料要花几秒到几十秒。Scheduler 是单线程事件循环，
如果同步加载，这段时间里所有推理请求都会卡住。

所以设计成：
    add()                → 起后台线程立即返回 None（不回响应）
    check_pending_load() → 每轮事件循环轮询一次，线程结束了才发响应

对应的 HTTP 接口是 POST /add_external_corpus，
运行时热加载语料而无需重启服务。

同一时刻只允许一个加载任务在跑（见 add 里的 _pending_load 判断），
避免多份语料并发写 SAM 造成状态竞争。
"""

import logging
import threading
from typing import Callable, Optional, Tuple

from sglang.srt.managers.io_struct import (
    AddExternalCorpusReqInput,
    AddExternalCorpusReqOutput,
    ListExternalCorporaReqInput,
    ListExternalCorporaReqOutput,
    RemoveExternalCorpusReqInput,
    RemoveExternalCorpusReqOutput,
)

logger = logging.getLogger(__name__)


class ExternalCorpusManager:
    """中译：为单个 scheduler 管理外部 SAM 语料的生命周期。

    Args（中译）：
        draft_worker: NGRAMWorker 实例（必须具备 add_external_corpus、
            remove_external_corpus、list_external_corpora 这几个方法）。
        send_response: 形如 callable(output, recv_req) 的回调，
            用于把延迟产生的响应发回 tokenizer manager。
    """

    def __init__(self, draft_worker, send_response: Callable):
        self._worker = draft_worker
        self._send_response = send_response
        # 正在进行的加载任务：(原始请求, 后台线程)；None 表示当前空闲
        self._pending_load: Optional[
            Tuple[AddExternalCorpusReqInput, threading.Thread]
        ] = None
        # 后台线程写、主线程读的结果暂存槽。无需加锁——
        # 靠 thread.join() 建立 happens-before 关系保证可见性（见下）。
        self._load_result: Optional[AddExternalCorpusReqOutput] = None

    def check_pending_load(self):
        """中译：由 scheduler 事件循环轮询调用。加载完成时发出响应。

        调用时机：每轮事件循环一次，因此本方法必须**非阻塞**——
        线程还活着就立刻返回，绝不能在这里等。

        三种返回路径：
            无任务          → 直接 return
            线程仍在跑      → 直接 return（下一轮再看）
            线程已结束      → join 取结果，成功则提交记账，然后回响应
        """
        if self._pending_load is None:
            return
        recv_req, thread = self._pending_load
        if thread.is_alive():
            return
        self._pending_load = None
        # 中译：join 用于建立正式的 happens-before 关系，保证 _load_result 可见。
        # 线程虽已 not alive，但没有 join 的话，主线程读到的 _load_result
        # 在内存模型上不保证是后台线程写入的最新值。
        thread.join()  # formal happens-before for _load_result visibility
        result = self._load_result
        self._load_result = None
        # 只有加载成功才把 token 数计入总预算——失败的加载不应占用配额
        if result.success:
            self._worker.commit_corpus_load(result.corpus_id, result.loaded_token_count)
        self._send_response(result, recv_req)

    def add(
        self, recv_req: AddExternalCorpusReqInput
    ) -> Optional[AddExternalCorpusReqOutput]:
        """异步加载一份外部语料。

        参数：
            recv_req: 含 corpus_id 与 token_chunks 两个字段
                corpus_id   = "product-faq-v1"
                token_chunks = 由 iter_external_corpus_chunks 产出的块迭代器，
                               形如 [[15496, 995], [-2147483648, 11274, ...], ...]

        返回：
            None                        → 已起后台线程，响应稍后由
                                          check_pending_load 发出
            AddExternalCorpusReqOutput  → 立即失败（已有任务在跑），
                                          success=False

        成功时最终发出的响应形如：
            success=True, corpus_id="product-faq-v1",
            loaded_token_count=350000,
            message="Loaded corpus 'product-faq-v1' with 350000 tokens."
        """
        if self._pending_load is not None:
            return AddExternalCorpusReqOutput(
                success=False,
                message="Another corpus load is already in progress.",
            )

        # 后台线程体：把结果（成功或异常）写进 _load_result，绝不抛出到线程外——
        # 线程里抛异常只会打印到 stderr，请求方会永远等不到响应。
        def _build():
            try:
                loaded = self._worker.add_external_corpus(
                    recv_req.corpus_id, recv_req.token_chunks
                )
                self._load_result = AddExternalCorpusReqOutput(
                    success=True,
                    corpus_id=recv_req.corpus_id,
                    message=f"Loaded corpus '{recv_req.corpus_id}' with {loaded} tokens.",
                    loaded_token_count=loaded,
                )
            except Exception as e:
                self._load_result = AddExternalCorpusReqOutput(
                    success=False, message=str(e)
                )

        # daemon=True：进程退出时不等这个线程，避免加载中途关服卡住
        thread = threading.Thread(target=_build, daemon=True)
        self._pending_load = (recv_req, thread)
        thread.start()
        return None  # 中译：响应稍后由 check_pending_load 发出

    # 中译（FIXME by kpham-sgl）：在有加载任务挂起期间移除语料属于未定义行为，
    # 应当被显式禁止。
    # 也就是说：当前若在 add 尚未完成时调用 remove，SAM 状态可能被并发破坏，
    # 调用方需自行避免这种时序。
    def remove(
        self, recv_req: RemoveExternalCorpusReqInput
    ) -> RemoveExternalCorpusReqOutput:
        """同步移除一份外部语料。

        参数：
            recv_req.corpus_id = "product-faq-v1"

        返回（同步，无后台线程——移除只是删几个 SAM 节点，很快）：
            成功 → success=True, message="Removed corpus 'product-faq-v1'."
            失败 → success=False, message=异常文本（如 id 不存在）
        """
        try:
            self._worker.remove_external_corpus(recv_req.corpus_id)
            return RemoveExternalCorpusReqOutput(
                success=True,
                message=f"Removed corpus '{recv_req.corpus_id}'.",
            )
        except Exception as e:
            return RemoveExternalCorpusReqOutput(success=False, message=str(e))

    def list(
        self, recv_req: ListExternalCorporaReqInput
    ) -> ListExternalCorporaReqOutput:
        """列出当前已加载的全部外部语料及其 token 数。

        返回示例：
            success=True,
            corpus_token_counts={"product-faq-v1": 350000, "code-snippets": 120000}

        用途：运维排查——确认预热语料是否真的加载上了、占了多少预算。
        """
        try:
            token_counts = self._worker.list_external_corpora()
            return ListExternalCorporaReqOutput(
                success=True,
                corpus_token_counts=token_counts,
            )
        except Exception as e:
            return ListExternalCorporaReqOutput(success=False, message=str(e))
