"""EAGLE V2 投机解码 Worker 实现。

本模块包含两个核心类：

    1. EagleDraftWorker  — 草稿模型 Worker，负责多步草稿前向（draft_forward）、
       草稿扩展（draft_extend）、树形 mask 构建等。
    2. EAGLEWorkerV2     — 顶层投机解码 Worker，协调目标模型验证与草稿模型推理，
      实现完整的「草稿→验证→接受→草稿扩展」循环。

一轮完整 decode 流程（forward_batch_generation 的 else 分支）：

    ┌─ 1. Draft ──────────────────────────────────────────┐
    │   草稿模型多步前向（speculative_num_steps 步），       │
    │   每步产生 topk 个候选，构建草稿树                     │
    │   → EagleDraftWorker.draft()                         │
    └──────────────────────────────────────────────────────┘
                            ↓
    ┌─ 2. Target Verify ──────────────────────────────────┐
    │   目标模型一次前向验证整棵草稿树                       │
    │   → EAGLEWorkerV2.verify()                           │
    │   → eagle_sample() 沿树路径贪心/拒绝采样               │
    └──────────────────────────────────────────────────────┘
                            ↓
    ┌─ 3. Draft Extend ───────────────────────────────────┐
    │   用验证接受结果喂给草稿模型做一次 extend，              │
    │   产生下一轮 draft 需要的 hidden states 和 topk       │
    │   → EagleDraftWorker._draft_extend_for_decode()      │
    └──────────────────────────────────────────────────────┘

extend/prefill 流程（forward_batch_generation 的 if 分支）：
    目标模型 prefill → 草稿模型 _draft_extend_for_prefill
    （prefill 阶段不投机，只填充草稿模型 KV cache）
"""

import contextlib
import logging
import time
from typing import List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_extend_npu_graph_runner import (
    EAGLEDraftExtendNpuGraphRunner,
)
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner import (
    EAGLEDraftNpuGraphRunner,
)
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.kv_canary.runner.canary_manager import context_tuple
from sglang.srt.layers.attention.tokenspeed_mla_backend import TokenspeedMLABackend
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
)
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.layers.utils.logprob import compute_spec_v2_logprobs
from sglang.srt.managers.io_struct import (
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner import DecodeCudaGraphRunner
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
)
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, EagleDraftWorkerBase
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import (
    EagleDraftExtendInput,
    EagleDraftInput,
    EagleVerifyInput,
)
from sglang.srt.speculative.eagle_info_v2 import fill_bonus_tokens
from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    _eagle_prefill_tail_tokens,
    build_tree_kernel_efficient,
    eagle_prepare_for_verify,
    eagle_sample,
    organize_draft_results,
    per_step_draft_out_cache_loc,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    commit_mamba_states_after_verify,
    draft_tp_context,
    generate_token_bitmask,
    load_token_map,
    move_accept_tokens_to_target_kvcache,
    record_stream_each,
    record_stream_for_v2_verify,
    select_top_k_tokens,
    spec_stage_span,
)
from sglang.srt.utils.async_probe import (
    maybe_detect_inf,
    maybe_detect_nan,
    maybe_detect_oob,
)
from sglang.srt.utils.common import (
    MultiprocessingSerializer,
    empty_context,
    fast_topk,
    get_available_gpu_memory,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    log_info_on_rank0,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

_is_npu = is_npu()
_is_cuda = is_cuda()
_is_musa = is_musa()
_is_hip = is_hip()

logger = logging.getLogger(__name__)


def _get_plan_stream(
    device: str,
) -> Tuple[any, contextlib.AbstractContextManager]:
    """获取用于 overlap plan 的 CUDA stream 及其上下文管理器。

    当启用 SGLANG_ENABLE_OVERLAP_PLAN_STREAM 环境变量时，创建一个独立的
    CUDA stream 用于计划阶段的计算与主计算流重叠；否则返回 None 和空上下文。

    参数：
        device: 设备字符串，如 "cuda:0"

    返回：(plan_stream, plan_stream_ctx)
        plan_stream:     CUDA stream 对象，或 None
        plan_stream_ctx: 对应的 stream 上下文管理器
    """
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        plan_stream = torch.get_device_module(device).Stream()
        plan_stream_ctx = torch.get_device_module(device).stream(plan_stream)
        return plan_stream, plan_stream_ctx
    else:
        return None, contextlib.nullcontext()


class EagleDraftWorker(EagleDraftWorkerBase):
    """EAGLE 草稿模型 Worker，负责草稿模型的多步前向推理和草稿树构建。

    核心职责：
        1. 加载和管理草稿模型权重
        2. 多步草稿前向（draft_forward）：每步产生 topk 个候选 token
        3. 草稿扩展（draft_extend）：用验证结果更新草稿模型状态
        4. 构建草稿树（通过 build_tree_kernel_efficient）

    属性说明：
        topk:                       每步每请求的候选数
        speculative_num_steps:      草稿推理步数
        speculative_num_draft_tokens: 草稿 token 总数（含 bonus）
        hot_token_id:               热门 token 映射（EAGLE3 中用于缩减词表）
        tree_mask_mode:             树形注意力 mask 模式
        cuda_graph_runner:          草稿 decode 的 CUDA Graph runner
        cuda_graph_runner_for_draft_extend: 草稿 extend 的 CUDA Graph runner
    """
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: int,
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # copy args
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.nccl_port = nccl_port
        self.target_worker = target_worker
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank

        # Args for easy access
        self.device = server_args.device
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Pre-allocated constants for the topk=1 chain fast path in draft_forward.
        self._topk1_parents_prealloc = None
        self._topk1_score_indices_prealloc = None
        self._rebuild_topk1_chain_buffers()

        # Load draft model weights only.
        if server_args.enable_dp_attention and self.speculative_algorithm.is_eagle3():
            ctx = draft_tp_context(get_attention_tp_group())
        else:
            ctx = empty_context()
        with (
            ctx
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.draft_worker = TpModelWorker(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # spec workers don't support pipeline parallelism
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
            )

        # Alias for better readability
        self.draft_runner = self.draft_worker.model_runner
        self.eagle_use_aux_hidden_state = False
        if self.speculative_algorithm.is_eagle3():
            eagle_config = getattr(
                self.draft_runner.model_config.hf_config, "eagle_config", {}
            )
            self.eagle_use_aux_hidden_state = eagle_config.get(
                "use_aux_hidden_state", True
            )
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        self.tree_mask_mode = TreeMaskMode.FULL_MASK

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        """分配草稿模型的 KV cache 池（由调度器调用）。

        参数：
            memory_pool_config:         内存池配置
            req_to_token_pool:          请求→token 位置映射池
            token_to_kv_pool_allocator: token→KV cache 位置分配器
        """
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.draft_worker.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        self.init_token_map()
        self.init_lm_head()

    def init_backends(self):
        """初始化草稿模型的所有后端：注意力后端、CUDA Graph 等。

        流程：
            1. 初始化草稿 worker 的模型后端（禁用 CUDA Graph）
            2. 初始化多步注意力后端
            3. 如启用 breakable prefill CUDA Graph，初始化 prefill graph
            4. 捕获 decode 和 extend 的 CUDA Graph
        """
        with self.draft_tp_context(
            self.draft_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.draft_worker.init_backends(disable_cuda_graph=True)
            self.init_attention_backend()
            if check_cuda_graph_backend(Phase.PREFILL, Backend.BREAKABLE):
                self.draft_runner.init_prefill_cuda_graph(force_for_draft_worker=True)
            self.init_cuda_graphs()

        if (c := self.draft_runner.canary_manager) is not None:
            c.mark_init_finished()

    def _rebuild_topk1_chain_buffers(self) -> None:
        """重建 topk=1 链式快速路径的预分配缓冲区。

        topk=1 时草稿树退化为一条链（每个节点只有一个子节点），
        parent_list 和 top_scores_index 是运行时常量，可以预分配。

        必须在 speculative_num_steps / speculative_num_draft_tokens 变化后调用。

        断言：topk=1 时 speculative_num_draft_tokens == speculative_num_steps + 1
        （因为链的节点数 = 步数 + 1 个 bonus）
        """
        # topk=1 时草稿树退化为链，parent_list 和 top_scores_index 是运行时常量
        if self.topk != 1:
            return
        # _override_worker_state can set both directly, bypassing the hook that
        # pins this relation; the fast path is only valid when it holds.
        assert self.speculative_num_draft_tokens == self.speculative_num_steps + 1, (
            "topk=1 requires speculative_num_draft_tokens == speculative_num_steps + 1, "
            f"got {self.speculative_num_draft_tokens} and {self.speculative_num_steps}"
        )
        num_steps = self.speculative_num_steps
        sa = self.server_args
        decode_max_bs = (
            sa.cuda_graph_config.decode.max_bs
            if sa.cuda_graph_config is not None
            else None
        )
        max_bs = max(
            decode_max_bs or 0,
            sa.max_running_requests or 0,
            1,
        )
        # A single-step chain has no parent entries (slow path drops the last
        # step). repeat (not expand): the kernel reads these as contiguous.
        parent_width = num_steps if num_steps > 1 else 0
        self._topk1_parents_prealloc = torch.arange(
            -1, parent_width - 1, dtype=torch.long, device=self.device
        ).repeat(max_bs, 1)
        self._topk1_score_indices_prealloc = torch.arange(
            num_steps, dtype=torch.long, device=self.device
        ).repeat(max_bs, 1)

    def init_token_map(self):
        """加载热门 token 映射表（用于缩减草稿模型的词表大小）。

        EAGLE3 模型自带 hot token id，忽略外部指定的 token map。
        非 EAGLE3 模型如果指定了 --speculative-token-map，则加载并用于
        将草稿模型的输出从热门子集映射回完整词表。
        """
        # 加载热门 token id
        if self.speculative_algorithm.is_eagle3():
            if self.server_args.speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif self.server_args.speculative_token_map is not None:
            self.hot_token_id = load_token_map(self.server_args.speculative_token_map)
            self.server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

    def init_lm_head(self):
        """初始化草稿模型的 embedding 和 lm_head。

        EAGLE3：大多数情况不共享 lm_head，但某些模型（如 nvidia/gpt-oss-120b-Eagle3）
        会共享。如果草稿模型的 load_lm_head_from_target 为 True，则从目标模型复制。

        非 EAGLE3：共享目标模型的 embedding 和 lm_head。如果有 hot_token_id，
        则只保留热门 token 对应的行。
        """
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        if self.speculative_algorithm.is_eagle3():
            # most cases EAGLE3 models don't share lm_head
            # but some models (e.g. nvidia/gpt-oss-120b-Eagle3) shares
            if (
                hasattr(self.draft_runner.model, "load_lm_head_from_target")
                and self.draft_runner.model.load_lm_head_from_target
            ):
                self.draft_runner.model.set_embed_and_head(embed, head)
            else:
                self.draft_runner.model.set_embed(embed)

            # grab hot token ids
            if self.draft_runner.model.hot_token_id is not None:
                self.hot_token_id = self.draft_runner.model.hot_token_id.to(
                    embed.device
                )

        else:
            if self.hot_token_id is not None:
                head = head.clone()
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]

            # Share the embedding and lm_head
            self.draft_runner.model.set_embed_and_head(embed, head)

    def init_attention_backend(self):
        """初始化多步注意力后端和草稿扩展后端。

        创建两类后端：
            1. draft_attn_backend:         decode 阶段的多步注意力后端
            2. draft_extend_attn_backend:  extend 阶段的注意力后端
        """
        # 创建多步注意力后端和 CUDA Graph runner

        self.draft_extend_attn_backend = None

        draft_backend_factory = DraftBackendFactory(
            self.server_args,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )

        # Initialize decode attention backend
        self.draft_attn_backend = draft_backend_factory.create_decode_backend()

        # Initialize draft extend attention backend (respects speculative_attention_mode setting)
        self.draft_extend_attn_backend = (
            draft_backend_factory.create_draft_extend_backend()
        )

        self.draft_runner.draft_attn_backend = self.draft_attn_backend
        if self.draft_extend_attn_backend is not None:
            self.draft_runner.attn_backend = self.draft_extend_attn_backend
        self.tree_mask_mode = TreeMaskMode.FULL_MASK

    def init_cuda_graphs(self):
        """捕获草稿模型 decode 和 extend 的 CUDA Graph。

        仅当 speculative_num_steps > 1 时捕获 decode graph
        （单步草稿不需要 graph，draft_forward 只做采样）。

        extend graph 仅在特定注意力后端（Triton/TRTLLM/Tokenspeed）下捕获。
        """
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED):
            return

        if self.server_args.model_impl == "mindspore":
            return

        Device2DraftCudaGraphRunner = {
            "npu": EAGLEDraftNpuGraphRunner,
            "cuda": EAGLEDraftCudaGraphRunner,
            "musa": EAGLEDraftCudaGraphRunner,
        }
        # Capture draft
        if self.speculative_num_steps > 1:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB",
            )
            self.cuda_graph_runner = Device2DraftCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB.",
            )

        Device2ExtendCudaGraphRunner = {
            "npu": EAGLEDraftExtendNpuGraphRunner,
            "cuda": EAGLEDraftExtendCudaGraphRunner,
            "musa": EAGLEDraftCudaGraphRunner,
        }
        supports_hip_aiter_draft_extend_graph = False
        if _is_hip:
            # Keep import local so non-HIP environments do not require aiter.
            from sglang.srt.layers.attention.aiter_backend import (
                AiterMultiStepDraftBackend,
            )

            supports_hip_aiter_draft_extend_graph = isinstance(
                self.draft_attn_backend, AiterMultiStepDraftBackend
            )

        supports_cuda_draft_extend_graph = (_is_cuda or _is_musa) and isinstance(
            self.draft_extend_attn_backend,
            (
                TritonAttnBackend,
                TRTLLMMLABackend,
                TRTLLMHAAttnBackend,
                TokenspeedMLABackend,
            ),
        )
        # Capture extend
        # TODO: support draft extend cuda graph for more attention backends
        if self.draft_extend_attn_backend and (
            _is_npu
            or supports_cuda_draft_extend_graph
            or supports_hip_aiter_draft_extend_graph
        ):
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft extend cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB",
            )
            self.cuda_graph_runner_for_draft_extend = Device2ExtendCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft extend cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB.",
            )

    def draft(self, batch: ScheduleBatch):
        """草稿前向：用草稿模型预测下一批候选 token，构建草稿树。

        参数：
            batch: 当前调度 batch，需已带上一轮的 spec_info（EagleDraftInput）

        返回：EagleVerifyInput，包含草稿 token、树形 mask、位置和检索索引

        流程：
            1. prepare_for_draft: 准备 ForwardBatch
            2. CUDA Graph replay 或 draft_forward: 多步草稿推理
            3. build_tree_kernel_efficient: 构建草稿树

        示例（bs=2, topk=4, num_steps=4, num_draft_tokens=17）：
            >>> # 草稿模型 4 步推理，每步每请求 4 个候选
            >>> # 拼接后 16 个草稿 + 1 个 bonus = 17 个验证 token
            >>> # 返回 EagleVerifyInput(draft_token=shape[34], ...)
        """
        draft_input: EagleDraftInput = batch.spec_info
        forward_batch, can_cuda_graph = self.prepare_for_draft(
            draft_input,
            self.req_to_token_pool,
            batch,
            self.cuda_graph_runner,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )

        n_inner = self.speculative_num_steps - 1
        canary_outside_ctx = (
            c.with_ops_outside_graph(
                single_forward_indices=list(range(n_inner)),
                maybe_inaccurate_forward_batch=forward_batch,
            )
            if (c := self.draft_runner.canary_manager) is not None
            else contextlib.nullcontext()
        )

        with canary_outside_ctx:
            # Run draft
            if can_cuda_graph:
                parent_list, top_scores_index, draft_tokens = (
                    self.cuda_graph_runner.replay(forward_batch)
                )
            else:
                if (
                    not forward_batch.forward_mode.is_idle()
                    and self.speculative_num_steps > 1
                ):
                    # Skip attention backend init for 1-step draft,
                    # `draft_forward` only does sample in this case.
                    self.draft_attn_backend.init_forward_metadata(forward_batch)
                    forward_batch.mark_forward_metadata_ready()
                parent_list, top_scores_index, draft_tokens = self.draft_forward(
                    forward_batch
                )

        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        # Build tree mask
        # Directly write to cuda graph buffers for verify attn
        tree_mask_buf, position_buf = (
            self.target_worker.model_runner.attn_backend.get_verify_buffers_to_fill_after_draft()
        )

        # build_tree_kernel uses seq_lens_sum only to size the (non-preallocated)
        # tree mask; over-size is safe. Skip per-iter .sum().item() D2H via UB.
        seq_lens_sum = batch.seq_lens_sum
        if seq_lens_sum is None:
            if tree_mask_buf is None:
                max_context_len = (
                    self.target_worker.model_runner.attn_backend.max_context_len
                )
                seq_lens_sum = batch.seq_lens.shape[0] * max_context_len
            else:
                # tree_mask_buf preallocated -> kernel ignores seq_lens_sum.
                seq_lens_sum = 0

        (
            tree_mask,
            position,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            draft_input.bonus_tokens,
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            self.tree_mask_mode,
            tree_mask_buf,
            position_buf,
        )

        return EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=None,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )

    def draft_forward(self, forward_batch: ForwardBatch):
        """多步草稿前向推理的核心实现。

        参数：
            forward_batch: 包含 spec_info（EagleDraftInput）的 ForwardBatch

        返回 (parent_list, top_scores_index, draft_tokens)：
            parent_list:      各候选的父节点下标
            top_scores_index: 被选中候选的列下标
            draft_tokens:     草稿 token id，shape [bs, num_draft_tokens - 1]

        流程：
            1. 解析 spec_info 中的 topk_p、topk_index、hidden_states
            2. 循环 speculative_num_steps 步：
               a. select_top_k_tokens: 选出当前步的输入 token
               b. 草稿模型前向 → logits
               c. softmax + topk → 下一步的候选
               d. 位置 +1
            3. organize_draft_results: 整理多步结果

        topk=1 快速路径：链式拓扑下 parent 和 index 是常量，直接用预分配缓冲区。
        """
        # 解析参数
        spec_info: EagleDraftInput = forward_batch.spec_info
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")

        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]

        out_cache_loc = per_step_draft_out_cache_loc(
            out_cache_loc,
            forward_batch.batch_size,
            self.topk,
            self.speculative_num_steps,
        )

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # Forward multiple steps
        scores = None
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            # We don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
            if i == self.speculative_num_steps - 1:
                break

            # Set inputs
            forward_batch.input_ids = input_ids
            # Qwen3-MoE MTP uses a fused RoPE + KV-store path whose cache_loc
            # argument must be contiguous.
            if (
                self.draft_runner.model_config.hf_config.architectures[0]
                == "Qwen3MoeForCausalLMMTP"
            ):
                out_cache_loc = out_cache_loc.contiguous()
            forward_batch.out_cache_loc = out_cache_loc[i]
            spec_info.hidden_states = hidden_states

            # Run forward under a per-step ForwardContext so the model layer
            # reads attn_backends[i] for the i-th draft step, plus a canary
            # index context so canary tracks which draft step is active.
            canary_index_ctx = (
                c.with_active_single_forward_manager(i)
                if (c := self.draft_runner.canary_manager) is not None
                else contextlib.nullcontext()
            )
            with (
                forward_context(
                    ForwardContext(
                        attn_backend=self.draft_attn_backend.attn_backends[i]
                    )
                ),
                canary_index_ctx,
            ):
                logits_output = self.draft_runner.forward(forward_batch).logits_output
            maybe_detect_nan(logits_output.next_token_logits, f"draft_forward step {i}")
            maybe_detect_inf(logits_output.next_token_logits, f"draft_forward step {i}")
            if self.topk == 1 and not _is_hip:
                # topk=1 → degenerate single-path tree; `topk_p` is unused
                # downstream, so skip softmax and just argmax over logits.
                # Gated to CUDA: on ROCm the argmax tie-break diverges from
                # the softmax+max path on FP8 logits and corrupts MTP draft
                # selection (DSV3.2 MTP GSM8K, see #26358).
                topk_index = torch.argmax(
                    logits_output.next_token_logits, dim=-1, keepdim=True
                )
                topk_p = torch.ones_like(topk_index, dtype=torch.float32)
            else:
                probs = torch.softmax(logits_output.next_token_logits, dim=-1)
                topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            maybe_detect_oob(
                topk_index,
                0,
                logits_output.next_token_logits.shape[-1],
                f"draft_forward step {i}: topk_index OOB vs vocab_size={logits_output.next_token_logits.shape[-1]}",
            )
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states
            forward_batch.positions.add_(1)

        # Organize the results
        if (
            self.topk == 1
            and token_list[0].shape[0] <= self._topk1_parents_prealloc.shape[0]
        ):
            # Chain topology: draft_tokens = concat of per-step tokens; the
            # full-length topk/sort/gather over score_list collapses to an
            # identity. parent_list and top_scores_index are runtime-invariant
            # constants pre-allocated on the worker. Oversized batches (rare,
            # would silently truncate the slice) fall through to the slow path.
            bs = token_list[0].shape[0]
            draft_tokens = torch.cat(token_list, dim=1)
            top_scores_index = self._topk1_score_indices_prealloc[:bs]
            parent_list = self._topk1_parents_prealloc[:bs]
            return parent_list, top_scores_index, draft_tokens

        return organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )

    def draft_extend(self):
        pass

    def _draft_extend_for_prefill(
        self,
        batch: ScheduleBatch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        mm_input_embeds: Optional[torch.Tensor] = None,
    ):
        """prefill 后的草稿扩展：正确填充草稿模型的 KV cache。

        参数：
            batch:               当前调度 batch
            target_hidden_states: 目标模型前向的隐藏状态
            next_token_ids:      目标模型生成的下一 token id
            mm_input_embeds:     多模态输入嵌入（可选）

        返回：EagleDraftInput，包含 topk_p、topk_index、hidden_states、bonus_tokens

        流程：
            1. 构造输入 ids（分块 prefill 感知的尾 token 旋转）
            2. 设置 spec_info 为 EagleDraftExtendInput
            3. 草稿模型前向（capture_hidden_mode=LAST）
            4. softmax + topk → 下一轮 draft 输入
        """
        # 构造 input_ids
        if not batch.forward_mode.is_idle():
            # Chunked-prefill-aware tail tokens (see PR #26329).
            tail_tokens = _eagle_prefill_tail_tokens(batch, next_token_ids)
            pt = 0
            for i, extend_len in enumerate(batch.extend_lens):
                input_ids = batch.input_ids[pt : pt + extend_len]
                batch.input_ids[pt : pt + extend_len] = torch.cat(
                    (input_ids[1:], tail_tokens[i].reshape(1))
                )
                pt += extend_len

        # Draft-extend spec_info for the extend forward; carries only
        # hidden_states + shape info.
        batch.spec_info = EagleDraftExtendInput(
            hidden_states=target_hidden_states,
            # draft mode is same with decode mode, only 1 token per req
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )

        # Run forward (LAST mode: only the final hidden state per request,
        # to feed the next draft step which expects [bs, hidden_dim]).
        # STANDALONE skips hidden states end-to-end.
        capture_hidden_mode = (
            CaptureHiddenMode.NULL
            if self.speculative_algorithm.is_standalone()
            else CaptureHiddenMode.LAST
        )
        batch.capture_hidden_mode = capture_hidden_mode
        forward_batch = ForwardBatch.init_new(batch, self.draft_runner)
        forward_batch.return_logprob = False
        if mm_input_embeds is not None:
            forward_batch.mm_input_embeds = mm_input_embeds

        canary_ctx = (
            context_tuple(
                c.with_ops_outside_graph(
                    single_forward_indices=[0],
                    maybe_inaccurate_forward_batch=forward_batch,
                ),
                c.with_active_single_forward_manager(0),
            )
            if (c := self.draft_runner.canary_manager) is not None
            else contextlib.nullcontext()
        )
        with canary_ctx:
            logits_output = self.draft_runner.forward(forward_batch).logits_output
        maybe_detect_nan(logits_output.next_token_logits, "draft_extend_for_prefill")
        maybe_detect_inf(logits_output.next_token_logits, "draft_extend_for_prefill")

        # Assemble the next-iter draft spec_info from the extend output.
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
        return EagleDraftInput(
            topk_p=topk_p,
            topk_index=topk_index,
            hidden_states=logits_output.hidden_states,
            bonus_tokens=next_token_ids,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )

    def _draft_extend_for_decode(
        self, batch: ScheduleBatch, batch_result: GenerationBatchResult
    ):
        """decode 后的草稿扩展：用验证接受结果更新草稿模型状态。

        参数：
            batch:        当前调度 batch
            batch_result: 验证阶段的 GenerationBatchResult

        无返回值，直接修改 batch_result.next_draft_input 的字段。

        流程：
            1. 构造 EagleDraftExtendInput（含接受长度信息）
            2. prepare_for_draft_extend: 准备 ForwardBatch
            3. CUDA Graph replay 或草稿模型前向
            4. 按 select_index 提取有效行
            5. softmax + topk → 下一轮 draft 的 topk_p/topk_index/hidden_states
        """
        # Batch 2: 草稿扩展
        draft_extend_input = EagleDraftExtendInput(
            hidden_states=batch_result.logits_output.hidden_states,
            # accept_lens includes the bonus token; correct drafts exclude it.
            num_correct_drafts=batch_result.accept_lens - 1,
            num_accept_tokens=batch_result.accept_lens,
            # Draft-extend fills the whole tree width (num_draft_tokens) per req,
            # not num_steps + 1, so DP MLP-sync padding stays consistent for topk > 1.
            num_tokens_per_req=self.speculative_num_draft_tokens,
            num_tokens_for_logprob_per_req=self.speculative_num_draft_tokens,
        )
        select_index = (
            torch.arange(len(batch.seq_lens), device=self.device)
            * self.speculative_num_draft_tokens
            + batch_result.accept_lens
            - 1
        )

        # Prepare for draft extend in a separate stream
        with self.plan_stream_ctx:
            forward_batch = self.prepare_for_draft_extend(
                draft_extend_input,
                batch,
                batch_result.next_token_ids,
                self.speculative_num_draft_tokens,
                self.draft_runner,
                self.cuda_graph_runner_for_draft_extend,
            )

        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )

        # Run draft extend batch in the main compute stream
        can_cuda_graph = (
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run(forward_batch)
        )

        canary_ctx = (
            context_tuple(
                c.with_ops_outside_graph(
                    single_forward_indices=[0],
                    maybe_inaccurate_forward_batch=forward_batch,
                ),
                c.with_active_single_forward_manager(0),
            )
            if (c := self.draft_runner.canary_manager) is not None
            else contextlib.nullcontext()
        )
        with canary_ctx:
            if can_cuda_graph:
                draft_logits_output = self.cuda_graph_runner_for_draft_extend.replay(
                    forward_batch
                )
            else:
                draft_logits_output = self.draft_runner.forward(
                    forward_batch
                ).logits_output

        maybe_detect_nan(
            draft_logits_output.next_token_logits,
            f"draft_extend_for_decode (cuda_graph={can_cuda_graph})",
        )
        maybe_detect_inf(
            draft_logits_output.next_token_logits,
            f"draft_extend_for_decode (cuda_graph={can_cuda_graph})",
        )

        # Reorganize the spec info for the next batch
        draft_logits_output.next_token_logits = draft_logits_output.next_token_logits[
            select_index
        ]
        if draft_logits_output.hidden_states is not None:
            draft_logits_output.hidden_states = draft_logits_output.hidden_states[
                select_index
            ]
        # The draft-extend graph only anchors full logits; selected-row topk is
        # owned by the worker for both graph and eager paths.
        if self.topk == 1 and not _is_hip:
            # Gated to CUDA: see #26358 — ROCm's argmax tie-break corrupts
            # MTP draft selection on FP8 logits.
            ret_topk_index = torch.argmax(
                draft_logits_output.next_token_logits, dim=-1, keepdim=True
            )
            ret_topk_p = torch.ones_like(ret_topk_index, dtype=torch.float32)
        else:
            probs = torch.softmax(draft_logits_output.next_token_logits, dim=-1)
            ret_topk_p, ret_topk_index = fast_topk(probs, self.topk, dim=-1)
        ret_hidden_states = draft_logits_output.hidden_states

        # Construct the return values
        next_draft_input = batch_result.next_draft_input
        (
            next_draft_input.topk_p,
            next_draft_input.topk_index,
            next_draft_input.hidden_states,
        ) = (
            ret_topk_p,
            ret_topk_index,
            ret_hidden_states,
        )


class EAGLEWorkerV2(BaseSpecWorker):
    """EAGLE V2 顶层投机解码 Worker，协调目标模型与草稿模型。

    继承 BaseSpecWorker，实现完整的「草稿→验证→接受→草稿扩展」循环。

    核心属性：
        _target_worker:       目标模型 worker
        _draft_worker:        草稿模型 worker（EagleDraftWorker）
        adaptive_controller:  自适应投机解码控制器（可选）
        topk:                 每步每请求的候选数
        speculative_num_steps:      草稿推理步数
        speculative_num_draft_tokens: 草稿 token 总数（含 bonus）

    示例：
        >>> # 初始化：server_args 指定 EAGLE3 算法
        >>> worker = EAGLEWorkerV2(server_args, gpu_id=0, tp_rank=0, ...)
        >>> # 每轮 decode：
        >>> result = worker.forward_batch_generation(batch)
        >>> # result.next_token_ids = 接受的 token
        >>> # result.accept_lens = tensor([3, 1, 5])  各请求接受长度
    """
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # 解析参数
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.tp_rank = tp_rank
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # 将草稿模型的上下文长度覆盖为目标模型的上下文长度
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # 创建草稿 worker
        self._draft_worker = EagleDraftWorker(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            attn_cp_rank,
            moe_dp_rank,
            nccl_port,
            target_worker,
        )

        # 自适应投机解码控制器（可选）
        self.adaptive_controller: Optional[AdaptiveController] = None
        if server_args.speculative_adaptive:
            self.adaptive_controller = AdaptiveController(
                self,
                config_path=server_args.speculative_adaptive_config,
            )

        # 预分配的 dummy tensor（用于 CUDA Graph 静态输入）
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        # 获取 overlap plan 用的 CUDA stream
        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    @property
    def spec_v2_attn_backends(self) -> tuple:
        """返回 spec_v2 前向涉及的所有注意力后端。

        被 decide_needs_cpu_seq_lens 用于决定是否需要将 seq_lens 从 D2H 传到 CPU。

        返回：(target_attn_backend, draft_attn_backend, draft_extend_attn_backend)
        """
        return (
            self._target_worker.model_runner.attn_backend,
            self._draft_worker.draft_attn_backend,
            self._draft_worker.draft_extend_attn_backend
            or self._draft_worker.draft_runner.attn_backend,
        )

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        """分配 KV cache 内存池。

        草稿模型与目标模型共享 KV cache 池。
        """
        self._draft_worker.alloc_memory_pool(
            memory_pool_config, req_to_token_pool, token_to_kv_pool_allocator
        )
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

    def init_backends(self):
        """初始化所有后端：草稿模型后端、自适应控制器运行时状态。"""
        self._draft_worker.init_backends()
        # 在目标模型和草稿模型后端都就绪后，构建自适应运行时状态
        if self.adaptive_controller is not None:
            with (
                self._draft_worker.draft_tp_context(
                    self._draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                self.adaptive_controller.register(
                    SpecRuntimeState(
                        speculative_num_steps=self.speculative_num_steps,
                        speculative_num_draft_tokens=self.speculative_num_draft_tokens,
                        draft_attn_backend=self._draft_worker.draft_attn_backend,
                        cuda_graph_runner=self._draft_worker.cuda_graph_runner,
                        target_attn_backend=self._target_worker.model_runner.attn_backend,
                        target_graph_runner=self._target_worker.model_runner.decode_cuda_graph_runner,
                        draft_extend_attn_backend=self._draft_worker.draft_extend_attn_backend,
                        cuda_graph_runner_for_draft_extend=self._draft_worker.cuda_graph_runner_for_draft_extend,
                    )
                )
                self.adaptive_controller.init_states(
                    cuda_graph_bs=(
                        None
                        if self.server_args.disable_cuda_graph
                        else self.server_args.cuda_graph_bs_decode
                    ),
                )

    @property
    def target_worker(self):
        """返回目标模型 worker。"""
        return self._target_worker

    @property
    def draft_worker(self):
        """返回草稿模型 worker。"""
        return self._draft_worker

    def clear_cache_pool(self):
        """清空 KV cache 池（空操作，由调度器统一清空共享池）。"""
        # 分配器和 KV cache 池与目标模型共享，由调度器统一清空
        pass

    def forward_batch_generation(self, batch: ScheduleBatch, on_publish=None):
        """前向生成入口：根据 forward_mode 分为 extend/prefill 和 decode 两条路径。

        参数：
            batch:      当前调度 batch
            on_publish: 回调函数，在目标模型前向完成后发布 new_seq_lens

        返回：GenerationBatchResult

        extend/prefill 路径（if 分支）：
            1. 目标模型 prefill（capture_hidden_mode=FULL）
            2. 草稿模型 _draft_extend_for_prefill（填充草稿 KV cache）

        decode 路径（else 分支）：
            1. 草稿模型 draft（多步草稿推理 + 树构建）
            2. 目标模型 verify（验证草稿树 + 采样接受）
            3. 草稿模型 _draft_extend_for_decode（用接受结果扩展草稿状态）

        示例：
            >>> # decode 路径：
            >>> result = worker.forward_batch_generation(batch)
            >>> # result.next_token_ids shape = [bs, spec_steps + 1]
            >>> # result.accept_lens = tensor([3, 1, 5])  各请求接受长度
            >>> # result.next_draft_input = EagleDraftInput(bonus_tokens=...)
        """
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            # ---- extend/prefill 路径 ----
            # 目标模型 prefill，捕获全部隐藏状态（FULL）
            target_capture_mode = (
                CaptureHiddenMode.NULL
                if self.speculative_algorithm.is_standalone()
                else CaptureHiddenMode.FULL
            )
            batch.capture_hidden_mode = target_capture_mode
            batch_output = self.target_worker.forward_batch_generation(batch)

            # spec_v2 约定：batch.seq_lens = 本次迭代前的长度。
            # Extend 处理了 L 个 prompt token；下一轮 verify 期望相同的 L。
            batch_output.new_seq_lens = batch.seq_lens
            # 在 draft_extend 之前发布，使 fence 位于 target-end
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)

            # 草稿模型 prefill 扩展
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
                spec_stage_span("draft_extend"),
            ):
                batch_output.next_draft_input = (
                    self.draft_worker._draft_extend_for_prefill(
                        batch,
                        batch_output.logits_output.hidden_states,
                        batch_output.next_token_ids,
                        batch_output.logits_output.mm_input_embeds,
                    )
                )
                return batch_output
        else:
            # ---- decode 路径 ----
            # 自适应：根据 batch size 激活合适的步数配置
            self.activate_step_by_batch(batch.seq_lens.shape[0])

            # 首轮 decode 时 spec_info 为 None，创建 idle 输入
            if batch.spec_info is None:
                capture_mode = (
                    CaptureHiddenMode.NULL
                    if self.speculative_algorithm.is_standalone()
                    else CaptureHiddenMode.LAST
                )
                batch.spec_info = EagleDraftInput.create_idle_input(
                    device=self.device,
                    hidden_size=EagleDraftInput.hidden_size_for(self.draft_worker),
                    dtype=EagleDraftInput.dtype_for(self.draft_worker),
                    topk=self.topk,
                    capture_hidden_mode=capture_mode,
                )

            # 步骤 1: 草稿前向（多步推理 + 草稿树构建）
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
                spec_stage_span("draft"),
            ):
                verify_input: EagleVerifyInput = self.draft_worker.draft(batch)
            assert verify_input.is_verify_input()
            batch.spec_info = verify_input

            # 步骤 2: 目标模型验证（一次前向验证整棵草稿树 + 采样接受）
            batch_output = self.verify(batch)

            # 在 draft_extend 之前发布，使 fence 位于 verify-end
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)

            # 步骤 3: 草稿扩展（用接受结果更新草稿模型状态）
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
                spec_stage_span("draft_extend"),
            ):
                self.draft_worker._draft_extend_for_decode(batch, batch_output)

            return batch_output

    def on_verify_complete_cpu(
        self, num_correct_drafts_per_req: list[int], batch_size: int = 0
    ) -> None:
        """验证完成后 CPU 侧回调：通知自适应控制器更新统计。

        参数：
            num_correct_drafts_per_req: 每个请求接受的草稿 token 数列表
            batch_size:                 当前 batch 大小
        """
        if self.adaptive_controller is not None:
            self.adaptive_controller.on_verify_complete(
                num_correct_drafts_per_req, batch_size=batch_size
            )

    def activate_step_by_batch(self, batch_size: int) -> None:
        """根据 batch size 激活自适应投机步数配置。

        参数：
            batch_size: 当前 batch 大小
        """
        if self.adaptive_controller is not None:
            self.adaptive_controller.activate_step_by_batch(batch_size)

    # -- 自适应投机解码协议 --

    def build_adaptive_runtime_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs=None,
    ) -> SpecRuntimeState:
        """为给定步数配置构建 SpecRuntimeState。

        参数：
            speculative_num_steps:       草稿推理步数
            speculative_num_draft_tokens: 草稿 token 总数
            cuda_graph_bs:               CUDA Graph 支持的 batch size 列表

        返回：SpecRuntimeState，包含该步数配置下的所有后端和 graph runner

        流程：
            1. 临时覆盖 worker 的步数参数
            2. 重新初始化草稿注意力后端和 CUDA Graph
            3. 构建目标模型注意力后端和 CUDA Graph runner
            4. 组装 SpecRuntimeState
        """
        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)

        with self._override_worker_state(
            speculative_num_steps,
            speculative_num_draft_tokens,
            cuda_graph_bs=cuda_graph_bs,
        ):
            self._draft_worker.init_attention_backend()
            self._draft_worker.init_cuda_graphs()

            # 构建目标模型的注意力后端和 CUDA Graph runner
            target_model_runner = self._target_worker.model_runner
            backup_init = target_model_runner.init_new_workspace
            try:
                target_attn_backend = target_model_runner._get_attention_backend(
                    init_new_workspace=True
                )
            finally:
                target_model_runner.init_new_workspace = backup_init

            target_graph_runner = None
            if not check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED):
                TargetGraphRunnerCls = (
                    NPUGraphRunner if _is_npu else DecodeCudaGraphRunner
                )
                target_graph_runner = TargetGraphRunnerCls(
                    target_model_runner,
                    attn_backend=target_attn_backend,
                    speculative_num_steps=speculative_num_steps,
                    speculative_num_draft_tokens=speculative_num_draft_tokens,
                )

            state = SpecRuntimeState(
                speculative_num_steps=speculative_num_steps,
                speculative_num_draft_tokens=speculative_num_draft_tokens,
                draft_attn_backend=self._draft_worker.draft_attn_backend,
                cuda_graph_runner=self._draft_worker.cuda_graph_runner,
                target_attn_backend=target_attn_backend,
                target_graph_runner=target_graph_runner,
                draft_extend_attn_backend=self._draft_worker.draft_extend_attn_backend,
                cuda_graph_runner_for_draft_extend=self._draft_worker.cuda_graph_runner_for_draft_extend,
            )

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        log_info_on_rank0(
            logger,
            f"Built adaptive runtime state steps={speculative_num_steps}: "
            f"elapsed={time.perf_counter() - tic:.2f}s, "
            f"mem={(before_mem - after_mem):.2f}GB",
        )

        return state

    def apply_runtime_state(self, state: SpecRuntimeState) -> None:
        """将预构建的运行时状态应用到当前 worker。

        参数：
            state: 要应用的 SpecRuntimeState

        流程：
            1. 如果步数相同则跳过
            2. 更新顶层和草稿侧的步数参数、注意力后端、CUDA Graph runner
            3. 更新目标侧的注意力后端和 CUDA Graph runner
            4. 同步 server_args
        """
        if self.speculative_num_steps == state.speculative_num_steps:
            return

        log_info_on_rank0(
            logger,
            "切换自适应运行时状态: "
            f"steps {self.speculative_num_steps} -> {state.speculative_num_steps}, "
            f"draft_tokens {self.speculative_num_draft_tokens} -> "
            f"{state.speculative_num_draft_tokens}",
        )

        # 顶层参数
        self.speculative_num_steps = state.speculative_num_steps
        self.speculative_num_draft_tokens = state.speculative_num_draft_tokens

        # 草稿侧
        dw = self._draft_worker
        dw.speculative_num_steps = state.speculative_num_steps
        dw.speculative_num_draft_tokens = state.speculative_num_draft_tokens
        dw.draft_attn_backend = state.draft_attn_backend
        dw.draft_runner.draft_attn_backend = state.draft_attn_backend
        dw.cuda_graph_runner = state.cuda_graph_runner
        dw.draft_extend_attn_backend = state.draft_extend_attn_backend
        # 保持 runner 的 attn_backend 与活跃的 draft-extend 后端同步
        # （draft-extend 前向读取 draft_runner.attn_backend）；
        # 与 init_attention_backend 保持一致。当为 None 时，runner 保持
        # 已初始化的后端（在各步数配置间保持一致）。
        if state.draft_extend_attn_backend is not None:
            dw.draft_runner.attn_backend = state.draft_extend_attn_backend
        dw.cuda_graph_runner_for_draft_extend = state.cuda_graph_runner_for_draft_extend
        dw._rebuild_topk1_chain_buffers()

        # 目标侧
        self._target_worker.model_runner.attn_backend = state.target_attn_backend
        self._target_worker.model_runner.decode_cuda_graph_runner = (
            state.target_graph_runner
        )

        # 同步 server_args
        self.server_args.speculative_num_steps = state.speculative_num_steps
        self.server_args.speculative_num_draft_tokens = (
            state.speculative_num_draft_tokens
        )

    @contextlib.contextmanager
    def _override_worker_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: list[int] | None = None,
    ):
        """临时覆盖 server_args 和 worker 属性，用于 CUDA Graph 捕获。

        参数：
            speculative_num_steps:       临时步数
            speculative_num_draft_tokens: 临时草稿 token 数
            cuda_graph_bs:               CUDA Graph batch size 列表

        用法：
            >>> with worker._override_worker_state(steps=4, draft_tokens=17):
            ...     # 在此作用域内，worker 的步数参数被临时修改
            ...     worker._draft_worker.init_cuda_graphs()
            >>> # 退出后自动恢复
        """
        sa = self.server_args
        dw = self._draft_worker
        backup = (
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            dw.speculative_num_steps,
            dw.speculative_num_draft_tokens,
            dw.draft_attn_backend,
            dw.draft_extend_attn_backend,
            dw.draft_runner.draft_attn_backend,
            dw.draft_runner.attn_backend,
            dw.cuda_graph_runner,
            dw.cuda_graph_runner_for_draft_extend,
            sa.speculative_num_steps,
            sa.speculative_num_draft_tokens,
            sa.cuda_graph_bs_decode,
            sa.disable_cuda_graph,
        )

        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        dw.speculative_num_steps = speculative_num_steps
        dw.speculative_num_draft_tokens = speculative_num_draft_tokens
        sa.speculative_num_steps = speculative_num_steps
        sa.speculative_num_draft_tokens = speculative_num_draft_tokens
        if cuda_graph_bs is not None:
            sa.cuda_graph_bs_decode = cuda_graph_bs
            # BS 感知的自适应投机可能将 cuda_graph_bs 裁剪为空列表
            # （例如 step=1 没有使用该 BS 范围）。对这些步数禁用 graph
            # 捕获；在 finally 中恢复，以免影响后续步数。
            if not cuda_graph_bs:
                sa.disable_cuda_graph = True
        dw._rebuild_topk1_chain_buffers()

        try:
            yield
        finally:
            (
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
                dw.speculative_num_steps,
                dw.speculative_num_draft_tokens,
                dw.draft_attn_backend,
                dw.draft_extend_attn_backend,
                dw.draft_runner.draft_attn_backend,
                dw.draft_runner.attn_backend,
                dw.cuda_graph_runner,
                dw.cuda_graph_runner_for_draft_extend,
                sa.speculative_num_steps,
                sa.speculative_num_draft_tokens,
                sa.cuda_graph_bs_decode,
                sa.disable_cuda_graph,
            ) = backup
            dw._rebuild_topk1_chain_buffers()

    def verify(self, batch: ScheduleBatch):
        """目标模型验证：一次前向验证整棵草稿树，采样并接受匹配的 token。

        参数：
            batch: 当前调度 batch，spec_info 为 EagleVerifyInput

        返回：GenerationBatchResult
            - next_token_ids:    接受的 token（shape [bs, spec_steps+1]）
            - accept_lens:       每个请求的接受长度（含 bonus）
            - new_seq_lens:      更新后的序列长度
            - next_draft_input:  下一轮草稿输入（含 bonus_tokens）

        流程：
            1. eagle_prepare_for_verify: 准备验证 ForwardBatch
            2. 目标模型前向验证整棵草稿树
            3. eagle_sample: 沿树路径贪心/拒绝采样，得到 accept_lens
            4. commit_mamba_states_after_verify: 更新 Mamba 状态
            5. fill_bonus_tokens: 提取每请求的 bonus token
            6. topk > 1 时压缩接受路径到前端

        示例：
            >>> # 假设 bs=2, topk=4, spec_steps=4, num_draft_tokens=17
            >>> result = worker.verify(batch)
            >>> # result.next_token_ids shape = [2, 5]  (spec_steps + 1)
            >>> # result.accept_lens = tensor([3, 5])  各请求接受长度
        """
        fwd_stream = torch.get_device_module(self.device).current_stream()
        verify_input: EagleVerifyInput = batch.spec_info
        record_stream_for_v2_verify(batch, verify_input, fwd_stream)

        verify_input.num_tokens_per_req = self.speculative_num_steps + 1
        bs = len(batch.seq_lens)

        # 批次 1: 目标模型验证
        # 在独立的 plan stream 中准备验证数据
        with self.plan_stream_ctx:
            verify_forward_batch, can_run_cuda_graph = eagle_prepare_for_verify(
                verify_input,
                self.req_to_token_pool,
                batch,
                self.target_worker,
            )

        # 覆盖 post-prepare 的重绑定：draft_token、plan_stream 分配的 out_cache_loc
        record_stream_each((batch.input_ids, batch.out_cache_loc), fwd_stream)

        # 修正 overlap plan 导致的 buffer 问题
        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )
            if (
                _is_npu
                and self._target_worker.model_runner.model_is_mrope
                and batch.spec_info is not None
                and getattr(batch.spec_info, "positions", None) is not None
                and not batch.forward_mode.is_idle()
            ):
                # mrope_position 依赖默认 stream 中的草稿输出，但在 plan stream
                # 中计算会导致错误。在此重新计算正确值。
                verify_forward_batch.compute_spec_mrope_positions(
                    self._target_worker.model_runner, batch
                )

            # custom_mask 和 position 等值依赖草稿输出，
            # 之前的 plan 步骤使用了错误值。在此重新运行相关计算以更新。
            self.target_worker.model_runner.attn_backend.update_verify_buffers_to_fill_after_draft(
                verify_input,
                (
                    self.target_worker.model_runner.decode_cuda_graph_runner.bs
                    if can_run_cuda_graph
                    else None
                ),
            )

        # 如果需要语法约束，在 CPU 上准备 grammar 数据
        if batch.has_grammar:
            retrieve_next_token_cpu = verify_input.retrieve_next_token.cpu()
            retrieve_next_sibling_cpu = verify_input.retrieve_next_sibling.cpu()
            draft_tokens_cpu = verify_input.draft_token.view(
                verify_input.retrieve_next_token.shape
            ).cpu()

        # 在主计算 stream 中运行目标模型验证（GPU 计算）
        # 当且仅当 cuda-graph 已运行 replay_prepare 时跳过元数据初始化——
        # eagle_prepare_for_verify 正好在那种情况下标记了 batch；
        # 非 cuda-graph 路径保持未标记，走 forward_extend 的初始化（pad 后）。
        forward_batch_output = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
        )
        logits_output = forward_batch_output.logits_output

        # 生成约束解码的 vocab mask
        vocab_mask = None
        if batch.has_grammar:
            # 为结构化输出生成 logit mask
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                verify_input,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )

            if vocab_mask is not None:
                assert verify_input.grammar is not None
                vocab_mask = vocab_mask.to(verify_input.retrieve_next_token.device)
                # 注意：否则此 vocab mask 会是上一个 extend 阶段的，
                # 应用后会产生错误结果
                batch.sampling_info.vocab_mask = None

        # 采样：沿草稿树路径贪心/拒绝采样
        maybe_detect_nan(logits_output.next_token_logits, "verify: target model logits")
        maybe_detect_inf(logits_output.next_token_logits, "verify: target model logits")
        (
            predict,
            accept_lens,
            accept_index,
        ) = eagle_sample(verify_input, batch, logits_output, vocab_mask)
        new_seq_lens = batch.seq_lens + accept_lens

        # 验证后更新混合 GDN 模型的 Mamba 状态
        commit_mamba_states_after_verify(
            self.target_worker,
            batch,
            accept_lens,
            accept_index,
            self.speculative_num_draft_tokens,
        )

        if not batch.forward_mode.is_idle():
            accept_tokens = predict[accept_index]
            bonus_tokens = torch.empty_like(accept_lens, dtype=torch.int32)
            # stride = accept_tokens 每请求宽度 = accept_index.shape[1]
            # （spec_steps + 1）；不是 num_draft_tokens，对 topk > 1 的树不对。
            fill_bonus_tokens[(bs,)](
                accept_tokens,
                accept_lens,
                bonus_tokens,
                accept_index.shape[1],
            )
        else:
            bonus_tokens = torch.empty((0,), device=self.device, dtype=torch.int32)

        # 计算对数概率（如果请求了 logprob）
        if batch.return_logprob and not batch.forward_mode.is_idle():
            compute_spec_v2_logprobs(
                batch, logits_output, predict, accept_index, self.speculative_num_steps
            )

        # topk > 1 时压缩接受路径到前端
        if not batch.forward_mode.is_idle() and self.topk > 1:
            # topk == 1 不需要处理：接受路径已经是前端的链，
            # 整个压缩操作是恒等变换。
            predict = self._finalize_accept_tree_path(
                batch, accept_index, accept_lens, predict, logits_output, bs
            )

        next_draft_input = EagleDraftInput(bonus_tokens=bonus_tokens)

        # verify_forward_batch 间接持有验证时 GPU tensor
        # （draft_token / out_cache_loc / ...），必须存活到即将到来的
        # prepare_for_draft_extend 中的 batch.input_ids 重绑定。
        # 调度器将其固定在 batch_record_buf 中维持 2 迭代窗口。
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=can_run_cuda_graph,
            speculative_num_draft_tokens=self.speculative_num_draft_tokens,
            next_draft_input=next_draft_input,
            accept_lens=accept_lens,
            new_seq_lens=new_seq_lens,
            routed_experts_output=forward_batch_output.routed_experts_output,
            indexer_topk_output=forward_batch_output.indexer_topk_output,
            extra_keep_alive_refs=[verify_forward_batch],
        )

    def _finalize_accept_tree_path(
        self,
        batch: ScheduleBatch,
        accept_index: torch.Tensor,
        accept_lens: torch.Tensor,
        predict: torch.Tensor,
        logits_output,
        bs: int,
    ) -> torch.Tensor:
        """树形草稿（topk > 1）：将接受路径移到每个请求块的前端。

        将 KV 槽位、predict、hidden_states 移到每个请求 per-req 块的连续前端，
        下游链式布局代码（draft-extend select_index、committed-KV 读取）依赖此布局。

        参数：
            batch:         当前调度 batch
            accept_index:  接受路径的节点索引 [bs, spec_steps + 1]
            accept_lens:   每请求接受长度 [bs]
            predict:       预测 token [bs * num_draft_tokens, ...]
            logits_output: logits 输出（hidden_states 会被原地修改）
            bs:            batch size

        返回：压缩后的 predict tensor
        """
        move_accept_tokens_to_target_kvcache(
            batch, accept_index, accept_lens - 1, self.token_to_kv_pool_allocator
        )
        predict = self._compact_accept_to_front(predict, accept_index, bs)
        if logits_output.hidden_states is not None:
            logits_output.hidden_states = self._compact_accept_to_front(
                logits_output.hidden_states, accept_index, bs
            )
        return predict

    def _compact_accept_to_front(
        self, x: torch.Tensor, accept_index: torch.Tensor, bs: int
    ) -> torch.Tensor:
        """将接受的树路径 gather 到每个 per-req 块的前端。

        ``x`` 是整个树的节点索引 (``[bs * num_draft_tokens, ...]``)，
        ``accept_index`` 是 ``[bs, spec_steps + 1]`` 的全局节点索引 (-1 填充)。
        填充项 clamp 到节点 0 但落在 accept_lens 之后（不会被读取）；
        尾部未接受槽位保留，作为 overshoot 释放。

        参数：
            x:            待压缩的 tensor [bs * num_draft_tokens, ...]
            accept_index: 接受路径索引 [bs, spec_steps + 1]
            bs:           batch size

        返回：压缩后的 tensor，接受路径在每请求块前端
        """
        nd = self.speculative_num_draft_tokens
        s1 = accept_index.shape[1]  # spec_steps + 1
        safe = accept_index.to(torch.int64).clamp(min=0).reshape(-1)
        gathered = x[safe]
        out = x.clone()
        out.view(bs, nd, *x.shape[1:])[:, :s1] = gathered.view(bs, s1, *x.shape[1:])
        return out

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """从磁盘更新草稿模型权重。

        参数：
            recv_req: 包含模型路径和加载格式的请求

        返回：(success: bool, message: str)
        """
        success, message = self._draft_worker.draft_runner.update_weights_from_disk(
            recv_req.model_path,
            recv_req.load_format,
            recapture_cuda_graph=recv_req.recapture_cuda_graph,
        )
        if not success:
            return success, message
        return True, "Succeeded to update model weights."

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """从 IPC 更新草稿模型权重。

        参数：
            recv_req: IPC 更新请求

        返回：(success: bool, message: str)
        """
        success, message = self._draft_worker.draft_runner.update_weights_from_ipc(
            recv_req
        )
        if not success:
            return success, message
        return True, "Succeeded to update model weights."

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        """从 tensor 更新草稿模型和目标模型权重。

        参数：
            recv_req: 包含序列化 tensor 的请求

        返回：(success: bool, message: str)

        流程：
            1. 反序列化 named_tensors
            2. 更新草稿模型权重
            3. 更新目标模型权重
        """
        monkey_patch_torch_reductions()
        named_tensors = MultiprocessingSerializer.deserialize(
            recv_req.serialized_named_tensors[self.tp_rank]
        )
        success, message = self.draft_worker.draft_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        if not success:
            return success, message

        success, message = self.target_worker.model_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        return success, message
