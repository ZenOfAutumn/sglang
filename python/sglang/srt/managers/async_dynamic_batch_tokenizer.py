"""
Asynchronous dynamic batch tokenizer for SGLang.

This module provides an async tokenizer with dynamic batching capabilities
to reduce tokenization overhead when multiple requests arrive concurrently.

中译：SGLang 的「异步动态批量分词器」。
      本模块提供一个异步分词器，具备「动态批处理（dynamic batching）」能力：
      当多个请求并发到达时，把它们短暂攒成一批一起做分词，从而摊薄每次分词的固定开销
      （函数调用、Python/HF tokenizer 启动等），提升高并发场景下的吞吐。
      它通常被 TokenizerManager 持有，用于对单条字符串 prompt 做编码。
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AsyncDynamicbatchTokenizer:
    """Asynchronous tokenizer with dynamic batching for single string prompts.

    Dynamically batches pending encode requests from a queue to reduce overhead.
    Only handles single string prompts - regular batch processing of multiple
    strings per request should be handled at a higher level.
    A single-thread ThreadPoolExecutor is used so the event loop stays responsive.

    Note: Uses lazy initialization for asyncio components because this class
    is instantiated in TokenizerManager.__init__() before the event loop starts.

    中译：面向「单条字符串 prompt」的异步动态批量分词器。
          - 从一个队列中取出待处理的 encode 请求，动态地攒批以摊薄分词开销。
          - 只处理单条字符串 prompt；一个请求里含多条字符串的常规批处理应在更上层完成。
          - 使用单线程 ThreadPoolExecutor 来执行阻塞式的分词调用，从而不阻塞 asyncio 事件循环。
          - 注意：asyncio 相关组件（队列、后台任务）采用「惰性初始化」，因为本类是在
            TokenizerManager.__init__() 中、事件循环尚未启动时就被实例化的，此时还不能创建
            依赖运行中事件循环的对象。
    """

    def __init__(
        self,
        tokenizer,
        max_batch_size: int = 32,
        batch_wait_timeout_s: float = 0.002,
    ) -> None:
        """初始化动态批量分词器。

        参数：
            tokenizer：底层分词器（可调用对象，如 HF tokenizer），实际执行编码。
            max_batch_size：单批最多攒多少条请求，达到即立刻处理。
            batch_wait_timeout_s：攒批的最长等待时间（秒），超时即处理已攒到的请求。
        副作用：创建单线程线程池；asyncio 组件留待 _ensure_initialized 惰性创建。
        """
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.batch_wait_timeout_s = batch_wait_timeout_s

        # Single queue for all encode requests - initialized lazily
        # 中译：承载所有 encode 请求的单一队列；惰性初始化（等事件循环就绪后再建）。
        self._queue: Optional[asyncio.Queue] = None
        # 中译：后台「攒批循环」任务句柄；同样惰性创建。
        self._batcher_task: Optional[asyncio.Task] = None

        # Single-thread executor for blocking tokenizer calls
        # 中译：单线程执行器，专门跑阻塞式分词调用，避免阻塞事件循环。
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of event loop dependent components.

        中译：惰性初始化依赖事件循环的组件（队列 + 后台攒批任务）。
              首次真正 encode 时调用，确保此时事件循环已在运行；仅执行一次。
        """
        if not self._initialized:
            self._queue = asyncio.Queue()
            self._batcher_task = asyncio.create_task(self._dynamic_batch_loop())
            self._initialized = True

    async def __call__(self, prompt: str, **kwargs) -> Any:
        """Encode a single prompt.

        中译：可调用入口，等价于 encode()，对单条 prompt 进行编码。
        """
        return await self.encode(prompt, **kwargs)

    async def encode(self, prompt: str, **kwargs) -> Any:
        """Encode a single prompt.

        中译：对单条 prompt 异步编码。做法：把 (prompt, kwargs, future) 放入队列，
              交由后台攒批循环处理，然后 await 该 future 拿到结果。
        参数：prompt 为待编码字符串；kwargs 透传给底层分词器。
        返回：底层分词器对该 prompt 的编码结果。
        """
        self._ensure_initialized()
        # 中译：为本次请求创建一个 future，后台批处理完成后会回填结果/异常。
        result_future: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._queue.put((prompt, kwargs, result_future))
        return await result_future

    async def _dynamic_batch_loop(self):
        """Dynamically batch incoming encode requests for efficiency.

        中译：后台常驻协程——动态地把到来的 encode 请求攒成批以提升效率。
              取到首个请求后，若队列里还有更多请求，则在 batch_wait_timeout_s 内
              继续收集（上限 max_batch_size），随后整批处理。单次异常不会终止循环。
        """
        while True:
            try:
                # Get the first request
                # 中译：阻塞等待队列中的第一个请求（一批的起点）。
                prompt, kwargs, result_future = await self._queue.get()

                # Collect requests into dynamic batch
                # 中译：用三个并行列表收集本批请求（prompt、对应 kwargs、对应 future）。
                prompts = [prompt]
                kwargs_list = [kwargs]
                result_futures = [result_future]

                # Check if there are more items immediately available in the queue
                # If queue is empty, process single item immediately without timeout
                # 中译：若队列已空，说明没有其他请求在等——立即处理这单条，不引入任何等待延迟。
                if self._queue.empty():
                    # No other requests waiting, process immediately
                    pass
                else:
                    # There might be more requests, wait for dynamic batching opportunity
                    # 中译：队列里还有请求，进入「攒批窗口」，尽量多收集一些以形成更大的批。
                    start_time = asyncio.get_running_loop().time()

                    # Collect more requests up to max_batch_size or batch_wait_timeout_s
                    # 中译：在「达到 max_batch_size」或「超过 batch_wait_timeout_s」之前持续收集。
                    while len(prompts) < self.max_batch_size:
                        elapsed = asyncio.get_running_loop().time() - start_time
                        if elapsed >= self.batch_wait_timeout_s:
                            break

                        # 中译：以「剩余等待时间」为超时去取下一个请求；超时即结束攒批、立刻处理。
                        remaining_time = self.batch_wait_timeout_s - elapsed
                        try:
                            prompt, kwargs, result_future = await asyncio.wait_for(
                                self._queue.get(), remaining_time
                            )
                            prompts.append(prompt)
                            kwargs_list.append(kwargs)
                            result_futures.append(result_future)
                        except asyncio.TimeoutError:
                            break

                # Log dynamic batch information
                # 中译：记录本批规模（调试用）。
                logger.debug(
                    f"AsyncDynamicbatchTokenizer: Processing dynamic batch of size {len(prompts)}"
                )

                # Process the dynamic batch
                # 中译：交给批处理方法实际执行编码并回填各 future。
                await self._process_dynamic_batch(prompts, kwargs_list, result_futures)

            except Exception as e:
                # 中译：单轮异常仅记录、不退出循环，保证后续请求仍能被处理。
                logger.error(f"Error in dynamic batch loop: {e}")
                # Continue the loop to handle other requests

    async def _process_dynamic_batch(
        self,
        prompts: List[str],
        kwargs_list: List[Dict],
        result_futures: List[asyncio.Future],
    ) -> None:
        """Process a dynamic batch of encode requests for single string prompts.

        中译：处理一批单字符串 prompt 的编码请求。
              关键点：只有当批内所有请求的 kwargs 完全一致时，才能合并成一次批量分词调用
              （大幅提速）；否则只能逐条分词。无论哪条路径，编码都在线程池里执行以免阻塞
              事件循环，最后把结果（或异常）回填到各自的 future。
        参数：prompts/kwargs_list/result_futures 三者一一对应，长度相同。
        返回：None；通过设置各 future 的结果或异常来交付。
        """
        # Check if all kwargs are identical for efficient batch processing
        # 中译：检查批内所有 kwargs 是否完全相同——这是能否走批量快路径的前提。
        first_kw = kwargs_list[0]
        can_batch = all(kw == first_kw for kw in kwargs_list[1:])
        kwargs = first_kw if can_batch else None

        try:
            # If every request uses identical kwargs we can run a single
            # batch tokenizer call for a big speed-up.
            # 中译：快路径——kwargs 一致且不止一条时，一次性批量分词，提速显著。
            if can_batch and len(prompts) > 1:
                encode_fn = partial(self.tokenizer, prompts, **kwargs)
                results = await asyncio.get_running_loop().run_in_executor(
                    self._executor, encode_fn
                )

                # 中译：批量结果是「字段 -> 列表」结构，按下标 i 把每条结果拆回各自的 future。
                for i, fut in enumerate(result_futures):
                    if not fut.done():
                        data = {k: v[i] for k, v in results.items()}
                        fut.set_result(data)
            else:
                # Process each request individually due to different kwargs
                # 中译：慢路径——kwargs 不一致（或仅单条），只能逐条分词。
                if len(prompts) > 1 and not can_batch:
                    # 中译：多条但 kwargs 不同导致无法批量，告警提示尽量统一分词参数以获得性能收益。
                    logger.warning(
                        f"AsyncDynamicbatchTokenizer: Dynamic batching disabled for batch of {len(prompts)} "
                        f"requests due to differing kwargs. This reduces performance benefits. "
                        f"Consider using consistent tokenization parameters across requests."
                    )

                # 中译：逐条调用分词器；同样放到线程池里执行。
                encode_fn = lambda prompts=prompts, kwargs=kwargs_list: [
                    self.tokenizer(p, **kw) for p, kw in zip(prompts, kwargs_list)
                ]
                results = await asyncio.get_running_loop().run_in_executor(
                    self._executor, encode_fn
                )

                for fut, res in zip(result_futures, results):
                    if not fut.done():
                        fut.set_result(res)
        except Exception as e:
            # 中译：处理失败时，把异常分发给本批所有未完成的 future，让各等待者都能感知到错误。
            logger.error(f"Error in dynamic batch processing: {e}")
            for fut in result_futures:
                if not fut.done():
                    fut.set_exception(e)

    def __del__(self):
        """Clean up background tasks.

        中译：析构时清理后台资源——取消仍在运行的攒批任务，并关闭线程池（不等待其完成）。
        """
        if hasattr(self, "_batcher_task") and self._batcher_task:
            if not self._batcher_task.done():
                self._batcher_task.cancel()
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
