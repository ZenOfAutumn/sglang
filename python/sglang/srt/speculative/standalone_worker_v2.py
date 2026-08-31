import logging
from dataclasses import replace
from typing import Optional

import torch

from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.layers.moe.utils import (
    draft_model_build_scope,
    speculative_moe_backend_context,
)
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.runtime_context import (
    get_device,
    get_parallel,
    get_schedule,
    get_spec,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
)
from sglang.srt.speculative.base_spec_worker import (
    BaseSpecWorker,
    EagleDraftWorkerBase,
)
from sglang.srt.speculative.eagle_utils import default_tree_mask_mode
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker, EAGLEWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import draft_tp_context, get_plan_stream
from sglang.srt.utils import empty_context, get_bool_env_var, is_cuda

if is_cuda():
    from sgl_kernel import segment_packbits  # noqa: F401

logger = logging.getLogger(__name__)
SGLANG_RETURN_ORIGINAL_LOGPROB = get_bool_env_var("SGLANG_RETURN_ORIGINAL_LOGPROB")


class StandaloneDraftWorker(EagleDraftWorker):
    """不与目标模型共享 embedding/lm_head 的独立草稿 Worker。

    与标准 EAGLE 草稿 Worker 的区别：
    - 草稿模型使用自己的 embedding 层和 lm_head 层
    - 不从目标模型复制权重，而是独立加载草稿模型权重
    - 适用于草稿模型与目标模型架构不同的情况

    示例::

        >>> # 当草稿模型和目标模型词汇表不同时使用
        >>> draft_worker = StandaloneDraftWorker(
        ...     server_args, gpu_id=0, tp_rank=0, dp_rank=None,
        ...     moe_ep_rank=0, attn_cp_rank=0, moe_dp_rank=0,
        ...     nccl_port=29500, target_worker=target)
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        EagleDraftWorkerBase.__init__(self)

        # copy args
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.ps = ps
        self.nccl_port = nccl_port
        self.target_worker = target_worker

        # Args for easy access
        self.device = get_device().device
        self.topk = get_spec().speculative_eagle_topk
        self.speculative_num_steps = get_spec().speculative_num_steps
        self.speculative_num_draft_tokens = get_spec().speculative_num_draft_tokens
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )

        self._rebuild_topk1_chain_buffers()

        # 设置常量
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        EagleDraftInput.ALLOC_LEN_PER_DECODE = max(
            self.speculative_num_steps * self.topk, self.speculative_num_draft_tokens
        )

        # Load draft model weights only. The standalone draft is a real model
        # whose MoE gates run during construction; the scope routes their
        # fusion decision to the speculative leaf (it does not swap
        # runner_backend — the draft's forwards run outside that context).
        with empty_context(), draft_model_build_scope():
            self.draft_worker = TpModelWorker(
                server_args=server_args,
                gpu_id=gpu_id,
                # spec workers don't support pipeline parallelism
                ps=replace(ps, pp_rank=0, pp_size=1),
                nccl_port=nccl_port,
                is_draft_worker=True,
                # The draft runs at absolute target positions.
                context_length=target_worker.model_runner.model_config.context_len,
            )

        # 别名，提高可读性
        self.draft_runner = self.draft_worker.model_runner
        self.draft_tp_context = (
            draft_tp_context if get_parallel().enable_dp_attention else empty_context
        )
        self.tree_mask_mode = default_tree_mask_mode()
        self.plan_stream, self.plan_stream_ctx = get_plan_stream(self.device)
        # draft_forward reads this (set in EagleDraftWorker.__init__, skipped here).
        self.index_share_for_mtp_iteration = (
            getattr(
                self.draft_runner.model_config.hf_config,
                "index_share_for_mtp_iteration",
                False,
            )
            and self.topk == 1
        )
        self.dsa_index_topk = None
        self.seed_dsa_topk_from_draft_extend = False
        self.dsa_extend_topk_buf = None

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        """分配内存池（Standalone 模式：不共享 embedding）。

        输入:
            memory_pool_config: 内存池配置
            req_to_token_pool: 请求到 token 映射池
            token_to_kv_pool_allocator: KV cache 分配器

        输出:
            None（原地初始化各池和映射表）
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

    def init_attention_backends(self):
        with (
            self.draft_tp_context(self.draft_runner.tp_group),
            speculative_moe_backend_context(),
        ):
            super().init_attention_backends()

    def init_cuda_graphs(self):
        with (
            self.draft_tp_context(self.draft_runner.tp_group),
            speculative_moe_backend_context(),
        ):
            super().init_cuda_graphs()

    def init_lm_head(self):
        """覆盖父类方法，阻止与目标模型共享 embedding 和 lm_head。

        Standalone 模式下，草稿模型使用自己的 embedding 和 lm_head，
        因此此方法为空操作（pass）。
        """
        # 对于 standalone worker，不共享 embedding 和 lm_head
        # 草稿模型使用自己的 embedding 和 lm_head
        pass


class StandaloneWorkerV2(EAGLEWorkerV2):
    """Standalone V2 投机 Worker。

    继承 EAGLEWorkerV2，使用独立草稿模型（不与目标模型共享 embedding/lm_head）。
    适用于草稿模型架构与目标模型不同的场景。

    示例::

        >>> worker = StandaloneWorkerV2(
        ...     server_args, gpu_id=0, tp_rank=0, dp_rank=None,
        ...     moe_ep_rank=0, attn_cp_rank=0, moe_dp_rank=0,
        ...     nccl_port=29500, target_worker=target)
        >>> # worker 的 draft_worker 是 StandaloneDraftWorker
        >>> type(worker.draft_worker).__name__
        'StandaloneDraftWorker'
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        BaseSpecWorker.__init__(self)

        # Parse arguments
        self.server_args = server_args
        self.topk = get_spec().speculative_eagle_topk
        self.speculative_num_steps = get_spec().speculative_num_steps
        self.speculative_num_draft_tokens = get_spec().speculative_num_draft_tokens
        self.gpu_id = gpu_id
        self.device = get_device().device
        self._target_worker = target_worker
        self.page_size = get_schedule().page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )

        # Create our custom draft worker that doesn't share embeddings/lm_head
        self._draft_worker = StandaloneDraftWorker(
            server_args,
            gpu_id,
            ps,
            nccl_port,
            target_worker,
        )

        self._validate_vocab_compatibility(
            target_vocab_size=target_worker.model_runner.model_config.vocab_size,
            target_tokenizer=target_worker.tokenizer,
        )

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        self.plan_stream, self.plan_stream_ctx = get_plan_stream(self.device)

        # TODO: 自适应投机解码
        self.adaptive_controller: Optional[AdaptiveController] = None

    def _validate_vocab_compatibility(
        self,
        target_vocab_size: int,
        target_tokenizer,
    ) -> None:
        """Raise ValueError if the draft and target vocabularies are incompatible."""
        draft_vocab_size = self._draft_worker.draft_runner.model_config.vocab_size
        draft_tokenizer = self._draft_worker.draft_worker.tokenizer
        if target_vocab_size != draft_vocab_size:
            raise ValueError(
                f"STANDALONE speculative decoding requires the draft model to share the "
                f"same vocabulary as the target model, but got "
                f"target vocab_size={target_vocab_size} and "
                f"draft vocab_size={draft_vocab_size}. "
                f"Use a draft model with a matching vocabulary, or a speculative "
                f"algorithm that supports heterogeneous vocabularies."
            )
        if (
            target_tokenizer is not None
            and draft_tokenizer is not None
            and hasattr(target_tokenizer, "get_vocab")
            and hasattr(draft_tokenizer, "get_vocab")
            and target_tokenizer.get_vocab() != draft_tokenizer.get_vocab()
        ):
            raise ValueError(
                "STANDALONE speculative decoding requires the draft model to share the "
                "same vocabulary as the target model, but the two tokenizers have "
                "different token-to-id mappings even though their vocab sizes match. "
                "Use a draft model with a matching vocabulary, or a speculative "
                "algorithm that supports heterogeneous vocabularies."
            )
