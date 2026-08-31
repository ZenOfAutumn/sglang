from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_info import (
        EagleDraftExtendInput,
        EagleDraftInput,
    )


def duplicate_prefix_tail_to_draft_branches(
    token_to_kv_pool,
    rows: torch.Tensor,
    prefix_base: torch.Tensor,
    last_page: torch.Tensor,
    num_new_pages: torch.Tensor,
    topk: int,
    page_size: int,
) -> None:
    """将前缀部分尾页复制到每个分支首页的空洞中（page>1 且 topk>1 时生效）。

    草稿解码扩展阶段按 block id（cache_loc // page_size）读取每个分支自己的草稿页，
    因此分支 b>=1 的空洞槽位 [0, last_page) 必须持有真实的前缀尾部 KV
    （分支 0 的首页本身就是它）。镜像 V1 #7725 的实现。

    输入:
        token_to_kv_pool: KV cache 池
        rows: req_to_token 中每个请求的 token 行映射，shape=(bs, num_pages)
        prefix_base: 每个请求前缀的页起始位置，shape=(bs,)
        last_page: 每个请求前缀尾页中已用的槽位数，shape=(bs,)
        num_new_pages: 每个分支需要新增的页数，shape=(bs,)
        topk: top-k 分支数
        page_size: 每页的 token 数

    输出:
        None（原地修改 KV cache）

    示例::

        >>> # 假设 bs=2, topk=4, page_size=16
        >>> # prefix_base = [48, 32], last_page = [3, 5]
        >>> # 分支 1~3 的首页 [0, last_page) 槽位会被填充为分支 0 的前缀尾部 KV
        >>> duplicate_prefix_tail_to_draft_branches(
        ...     kv_pool, rows, prefix_base, last_page, num_new_pages, topk=4, page_size=16)
    """
    if topk <= 1:
        return
    bs = rows.shape[0]
    page_off = torch.arange(page_size, device=rows.device, dtype=torch.int64)
    branches = torch.arange(1, topk, device=rows.device, dtype=torch.int64).view(
        1, topk - 1, 1
    )
    # 源：前缀尾页 [prefix_base, prefix_base + page_size)，每个分支一份
    src_pos = (prefix_base.view(bs, 1, 1) + page_off.view(1, 1, page_size)).expand(
        bs, topk - 1, page_size
    )
    # 目标：分支 b 的首页 [prefix_base + b*num_new_pages*page, + page_size)
    tgt_pos = (
        prefix_base.view(bs, 1, 1)
        + branches * (num_new_pages.view(bs, 1, 1) * page_size)
        + page_off.view(1, 1, page_size)
    )
    # 仅 [0, last_page) 持有真实前缀 KV；
    # [last_page, page_size) 是分支自己的草稿槽位，不能被覆盖
    vmask = (page_off.view(1, 1, page_size) < last_page.view(bs, 1, 1)).expand(
        bs, topk - 1, page_size
    )
    src_slots = torch.gather(rows, 1, src_pos.reshape(bs, -1)).reshape(
        bs, topk - 1, page_size
    )[vmask]
    tgt_slots = torch.gather(rows, 1, tgt_pos.reshape(bs, -1)).reshape(
        bs, topk - 1, page_size
    )[vmask]
    if src_slots.numel() > 0:
        token_to_kv_pool.move_kv_cache(tgt_slots, src_slots)


class EagleDraftWorkerBase(ABC):
    """EAGLE 草稿 Worker 抽象基类。

    定义草稿阶段的核心接口：draft（草稿前向）和 draft_extend（KV 补齐扩展）。
    所有具体草稿 worker（如 EAGLE 草稿模型 worker）都应继承此类。
    """

    @abstractmethod
    def draft(self):
        """草稿前向：用草稿模型预测下一批候选 token。

        此方法由子类实现，执行草稿模型的前向推理，生成候选 token 及其
        概率分布（top-k），供目标模型验证。
        """
        pass

    @abstractmethod
    def draft_extend(self):
        """草稿扩展：为草稿 token 补齐 KV cache。

        此方法由子类实现，执行 extend 前向以写入草稿 token 的 KV，
        使后续验证阶段的注意力计算可以正确引用草稿 token 的 KV。
        """
        pass

    def alloc_memory_pool(self, **kwargs):
        """分配草稿模型的内存池。默认空操作，子类按需覆盖。"""
        pass

    def init_backends(self):
        """初始化后端：先标准后端（无 CUDA Graph），再草稿专用后端。

        子类应通过上下文管理器（draft_tp_context、speculative_moe_backend_context 等）
        包装此方法，而非重新实现逻辑。
        """
        self.draft_worker.init_backends(disable_cuda_graph=True)
        self.init_attention_backend()
        self.init_cuda_graphs()

    def prepare_for_draft_extend(
        self,
        draft_extend_input: EagleDraftExtendInput,
        batch: ScheduleBatch,
        predict: torch.Tensor,
        num_draft_tokens: int,
        draft_model_runner: Any,
        cuda_graph_runner: Any,
    ):
        """为草稿扩展阶段准备 ForwardBatch。

        将草稿模型的预测 token 设置为输入，构建 DRAFT_EXTEND_V2 模式的
        ForwardBatch，写入 KV cache 并初始化注意力元数据。

        输入:
            draft_extend_input: 草稿扩展输入数据结构
            batch: 调度批次
            predict: 草稿模型预测的 token id，shape=(bs * num_draft_tokens,)
            num_draft_tokens: 每个请求的草稿 token 数
            draft_model_runner: 草稿模型的 ModelRunner
            cuda_graph_runner: CUDA Graph runner（可能为 None）

        输出:
            ForwardBatch: 构建好的前向批次，seq_lens 已加上 num_draft_tokens

        示例::

            >>> # batch.seq_lens = [10, 20], num_draft_tokens = 8
            >>> fb = worker.prepare_for_draft_extend(
            ...     ext_input, batch, predict_tokens, 8, runner, graph_runner)
            >>> fb.seq_lens  # [18, 28]
            >>> fb.forward_mode
            <ForwardMode.DRAFT_EXTEND_V2: ...>
        """
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )
        from sglang.srt.utils.async_probe import maybe_detect_oob
        from sglang.srt.utils.common import is_npu

        bs = len(batch.seq_lens)
        extend_num_tokens = bs * num_draft_tokens
        # seq_lens_cpu 不存在时走 GPU-only 路径，避免 .tolist()/.cpu()
        gpu_only = batch.seq_lens_cpu is None

        batch.spec_info = draft_extend_input
        batch.input_ids = predict
        maybe_detect_oob(
            batch.input_ids,
            0,
            batch.model_config.vocab_size,
            "v2 prepare_for_draft_extend input_ids",
        )
        # init_new 要求 prefix_lens 和 extend_lens 同时为 list 或同时为 Tensor；
        # gpu_only 生成设备张量以跳过 H2D 传输
        if gpu_only:
            batch.prefix_lens = batch.seq_lens.to(torch.int32)
            batch.extend_lens = torch.full(
                (bs,), num_draft_tokens, dtype=torch.int32, device=batch.seq_lens.device
            )
        else:
            batch.prefix_lens = batch.seq_lens_cpu.tolist()
            batch.extend_lens = [num_draft_tokens] * bs
        batch.extend_num_tokens = extend_num_tokens
        capture_mode = (
            CaptureHiddenMode.NULL
            if draft_model_runner.spec_algorithm.is_standalone()
            else CaptureHiddenMode.FULL
        )
        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.DRAFT_EXTEND_V2
        )
        batch.capture_hidden_mode = capture_mode
        forward_batch = ForwardBatch.init_new(batch, draft_model_runner)
        # 前向看到的是写入后的长度（draft extend 写入 num_draft_tokens 个槽位）；
        # 变更保留在 forward_batch 上以保护 ScheduleBatch.seq_lens
        forward_batch.seq_lens = forward_batch.seq_lens + num_draft_tokens
        if not gpu_only:
            forward_batch.seq_lens_cpu = forward_batch.seq_lens_cpu + num_draft_tokens
            forward_batch.seq_lens_sum = int(forward_batch.seq_lens_cpu.sum())
        else:
            # 提供 CPU 镜像（extend_seq_lens 均为 num_draft_tokens），
            # 使后端 max() 从列表读取，避免每次迭代的 D2H 同步
            forward_batch.extend_seq_lens_cpu = [num_draft_tokens] * bs
        can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run(forward_batch)
        if not batch.forward_mode.is_idle() and not can_cuda_graph:
            draft_model_runner.attn_backend.init_forward_metadata(forward_batch)
            # 计划性 pre-pad；不要选择 post-pad re-plan。
            # DSA 的 indexer 无法在 DP-padded batch 上重建 deep_gemm schedule_meta
            # （`_batch_size == batch_size` 断言，见 #27091）；
            # 已标记的 pre-pad 元数据按原样使用，匹配已验证的
            # skip_attn_backend_init=True 行为。
            # NPU + --disable-cuda-graph 时，prepare_mlp_sync_batch padding 后
            # block_table 形状不匹配；将 re-init 推迟到 forward_extend（post-pad）
            if not is_npu() or can_cuda_graph:
                forward_batch.mark_forward_metadata_ready()
        return forward_batch

    def prepare_for_draft(
        self,
        draft_input: EagleDraftInput,
        req_to_token_pool: ReqToTokenPool,
        batch: ScheduleBatch,
        cuda_graph_runner: EAGLEDraftCudaGraphRunner,
        draft_model_runner: ModelRunner,
        topk: int,
        num_steps: int,
    ):
        """为草稿前向阶段准备 ForwardBatch 和缓存位置。

        分配草稿 token 的 KV cache 位置（out_cache_loc），设置 positions、
        capture_hidden_mode 等参数，构建 ForwardBatch。

        输入:
            draft_input: 草稿输入数据结构
            req_to_token_pool: 请求到 token 映射池
            batch: 调度批次
            cuda_graph_runner: CUDA Graph runner
            draft_model_runner: 草稿模型 ModelRunner
            topk: top-k 分支数
            num_steps: 草稿步数

        输出:
            (forward_batch, can_cuda_graph) 元组:
            - forward_batch: 构建好的前向批次
            - can_cuda_graph: 是否可以使用 CUDA Graph

        示例::

            >>> # bs=2, topk=4, num_steps=5
            >>> fb, can_graph = worker.prepare_for_draft(
            ...     draft_input, pool, batch, graph_runner, runner, topk=4, num_steps=5)
            >>> fb.positions.shape  # (2 * 4,) = (8,)
            >>> can_graph
            True
        """
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
        )
        from sglang.srt.speculative.triton_ops.cache_locs import (
            assign_draft_cache_locs_contiguous,
        )

        if not batch.forward_mode.is_idle():
            bs = len(batch.seq_lens)

            # 分配缓存位置（草稿写入目标）
            page_size = batch.token_to_kv_pool_allocator.page_size
            if page_size == 1 or topk == 1:
                # 页大小为 1 或 topk 为 1：简单连续分配
                batch.out_cache_loc = torch.empty(
                    (bs * topk * num_steps,),
                    dtype=torch.int64,
                    device=batch.device,
                )
                # FIXME(lsyin): 对齐默认代码路径
                assign_draft_cache_locs_contiguous[(bs,)](
                    batch.req_pool_indices,
                    req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    batch.out_cache_loc,
                    req_to_token_pool.req_to_token.shape[1],
                    topk,
                    num_steps,
                )
            else:
                # page_size > 1 且 topk > 1：按分支页对齐分配草稿页。
                # 将 out_cache_loc 从页对齐的树区域缩减到密集草稿槽位
                # （跳过每个分支重复的前缀尾页槽位和尾部 padding），
                # 匹配 generate_draft_decode_kv_indices 的分页读取公式:
                # prefix_base + t*num_new_pages*page + last_page + s
                # base 是 batch.seq_lens（草稿时 == KV-ready 的已提交前缀；
                # bonus 是 verify 写入的树根，不属于 [0:seq_lens]）
                rows = req_to_token_pool.req_to_token[batch.req_pool_indices.long()]
                seq_lens = batch.seq_lens.to(torch.int64)
                last_page = seq_lens % page_size
                prefix_base = seq_lens - last_page
                num_new_pages = (last_page + num_steps + page_size - 1) // page_size
                topk_ids = torch.arange(
                    topk, device=rows.device, dtype=torch.int64
                ).view(1, topk)
                starts = (
                    prefix_base.view(bs, 1)
                    + topk_ids * (num_new_pages.view(bs, 1) * page_size)
                    + last_page.view(bs, 1)
                )
                steps = torch.arange(
                    num_steps, device=rows.device, dtype=torch.int64
                ).view(1, 1, num_steps)
                pos = (starts.view(bs, topk, 1) + steps).reshape(bs, topk * num_steps)
                batch.out_cache_loc = (
                    torch.gather(rows, 1, pos).reshape(-1).contiguous()
                )

                # 每个分支的页对齐区域起始有 `last_page` 个空洞槽位，
                # 与前缀尾页重叠；将真实前缀尾 KV 复制进去，
                # 使整页读取保持一致（见辅助函数文档）
                duplicate_prefix_tail_to_draft_branches(
                    draft_model_runner.token_to_kv_pool,
                    rows,
                    prefix_base,
                    last_page,
                    num_new_pages,
                    topk,
                    page_size,
                )

        # 获取 forward batch
        draft_input.num_tokens_per_req = topk
        draft_input.num_tokens_for_logprob_per_req = topk
        capture_mode = (
            CaptureHiddenMode.NULL
            if draft_model_runner.spec_algorithm.is_standalone()
            else CaptureHiddenMode.LAST
        )
        draft_input.positions = batch.seq_lens.repeat_interleave(topk, dim=0)
        batch.capture_hidden_mode = capture_mode
        forward_batch = ForwardBatch.init_new(batch, draft_model_runner)
        can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run(forward_batch)
        return forward_batch, can_cuda_graph


class BaseSpecWorker(ABC):
    """投机解码 Worker 抽象基类。

    定义目标 worker 和草稿 worker 的访问接口，以及验证完成后的钩子方法。
    所有具体投机 worker（EAGLEWorkerV2、NGRAMWorker、DFlashWorkerV2 等）
    都继承此类。

    示例::

        >>> class MySpecWorker(BaseSpecWorker):
        ...     @property
        ...     def target_worker(self):
        ...         return self._target_worker
        ...     @property
        ...     def draft_worker(self):
        ...         return self._draft_worker
        ...     def clear_cache_pool(self):
        ...         self.target_worker.clear_cache_pool()
    """

    @property
    @abstractmethod
    def target_worker(self) -> TpModelWorker:
        """目标模型 worker（大模型）。"""
        pass

    @property
    @abstractmethod
    def draft_worker(self) -> EagleDraftWorkerBase:
        """草稿模型 worker。"""
        pass

    @property
    def spec_v2_attn_backends(self) -> tuple:
        """spec_v2 前向涉及的注意力后端元组。

        被 decide_needs_cpu_seq_lens 按位或合并。
        默认仅返回目标后端；子类可扩展以加入草稿后端。
        """
        return (self.target_worker.model_runner.attn_backend,)

    @abstractmethod
    def clear_cache_pool(self):
        """清空缓存池。

        TODO: 将此抽象方法移到 BaseTpWorker，通过 self.model_runner 调用。
        """
        pass

    def alloc_memory_pool(self, **kwargs):
        """分配内存池。默认空操作。"""
        pass

    def init_backends(self):
        """初始化后端。默认空操作。"""
        pass

    def on_verify_complete_cpu(
        self, num_correct_drafts_per_req: list[int], batch_size: int = 0
    ) -> None:
        """验证完成后、接受计数已在 CPU 上的钩子方法。

        默认空操作。自适应感知的 worker 覆盖此方法以向控制器馈送数据，
        而无需在 worker 热路径中强制 GPU→CPU 同步。

        输入:
            num_correct_drafts_per_req: 每个请求接受的草稿 token 数列表
            batch_size: 批次大小

        示例::

            >>> # 验证完成后，自适应控制器根据接受率调整 num_steps
            >>> worker.on_verify_complete_cpu([3, 5, 2, 7], batch_size=4)
        """
        pass

    def activate_step_by_batch(self, batch_size: int) -> None:
        """根据当前批次大小激活最优的自适应步数。

        默认空操作。自适应感知的 worker 覆盖此方法以在每轮草稿前
        切换运行时状态。

        输入:
            batch_size: 当前批次大小

        示例::

            >>> # batch_size=1 时激活 8 步草稿，batch_size=32 时激活 2 步
            >>> worker.activate_step_by_batch(batch_size=1)
        """
        pass
