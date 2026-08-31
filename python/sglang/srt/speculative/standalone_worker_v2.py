import contextlib
import logging
from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import speculative_moe_backend_context
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
)
from sglang.srt.speculative.eagle_utils import TreeMaskMode
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker, EAGLEWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import draft_tp_context
from sglang.srt.utils import empty_context, get_bool_env_var, is_cuda

if is_cuda():
    from sgl_kernel import segment_packbits  # noqa: F401

logger = logging.getLogger(__name__)
SGLANG_RETURN_ORIGINAL_LOGPROB = get_bool_env_var("SGLANG_RETURN_ORIGINAL_LOGPROB")


def _get_plan_stream(
    device: str,
) -> Tuple[any, contextlib.AbstractContextManager]:
    """获取调度计划流的上下文管理器。

    输入:
        device: 设备字符串，如 "cuda" 或 "cuda:0"

    输出:
        (plan_stream, plan_stream_ctx) 元组:
        - plan_stream: CUDA Stream 对象（启用时）或 None
        - plan_stream_ctx: 上下文管理器，用于进入该流

    示例::

        >>> stream, ctx = _get_plan_stream("cuda")
        >>> with ctx:
        ...     # 在 plan stream 上执行调度逻辑
        ...     pass
    """
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        plan_stream = torch.get_device_module(device).Stream()
        plan_stream_ctx = torch.get_device_module(device).stream(plan_stream)
        return plan_stream, plan_stream_ctx
    else:
        return None, contextlib.nullcontext()


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
        tp_rank: int,
        dp_rank: int,
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # 复制参数
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.nccl_port = nccl_port
        self.target_worker = target_worker
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank

        # 便捷访问参数
        self.device = server_args.device
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # 为 topk=1 链式快速路径预分配常量张量
        self._topk1_parents_prealloc = None
        self._topk1_score_indices_prealloc = None
        self._rebuild_topk1_chain_buffers()

        # 设置常量
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        EagleDraftInput.ALLOC_LEN_PER_DECODE = max(
            self.speculative_num_steps * self.topk, self.speculative_num_draft_tokens
        )

        # 仅加载草稿模型权重（不共享目标模型的 embedding/lm_head）
        with empty_context():
            self.draft_worker = TpModelWorker(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # spec worker 不支持流水线并行
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
            )

        # 别名，提高可读性
        self.draft_runner = self.draft_worker.model_runner
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

    def init_backends(self):
        """初始化后端，使用草稿 TP 上下文和投机 MoE 后端上下文。"""
        with self.draft_tp_context(
            self.draft_runner.tp_group
        ), speculative_moe_backend_context():
            super().init_backends()

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
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # 将草稿模型的上下文长度覆盖为目标模型的上下文长度
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # 创建不共享 embedding/lm_head 的独立草稿 worker
        self._draft_worker = StandaloneDraftWorker(
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

        # 哑张量（用于 CUDA Graph 捕获时的占位）
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

        # TODO: 自适应投机解码
        self.adaptive_controller: Optional[AdaptiveController] = None
