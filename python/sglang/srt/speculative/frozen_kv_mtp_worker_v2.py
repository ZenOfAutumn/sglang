# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Spec-v2 worker for Frozen-KV MTP (两层结构，类似 ``eagle_worker_v2``)。

Frozen-KV MTP (Multi-Token Prediction) 的 spec-v2 worker 实现。

核心设计理念:
    - 冻结 draft 模型以**只读**方式访问 target 模型的 KV cache，自身不拥有 KV pool
    - 因此其 "draft extend" 并非一次模型前向传播:
        1. 它从上一次 verify 的输出中选择最后接受的 token + target hidden state
           作为下一轮迭代的**种子 (seed)**
        2. 种子前向传播在下一轮 draft 开始时执行
    - 复用 EAGLEWorkerV2 的 verify / move_accept_tokens / forward 骨架
    - 仅 draft worker 和基于种子的 draft-extend 是 Frozen-KV 特有的

与 EAGLE 的关键区别:
    - EAGLE draft 拥有自己的 KV pool，需要 draft-extend 前向传播来扩展 KV
    - Frozen-KV draft 没有自己的 KV，直接读取 target 的 KV，无需扩展

典型工作流:
    1. Target prefill → 捕获 FULL hidden states
    2. Draft prefill seed (无前向传播，仅选取种子)
    3. Decode 循环:
       a. Draft forward (种子前向 + 循环推测)
       b. Target verify (验证推测 token)
       c. Draft extend for decode (从 verify 结果中选取下一轮种子)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.cuda_graph_config import cuda_graph_fully_disabled
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import EagleDraftWorkerBase
from sglang.srt.speculative.eagle_utils import (
    build_tree_kernel_efficient,
    organize_draft_results,
)
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2, _get_plan_stream
from sglang.srt.speculative.frozen_kv_mtp_info import (
    FrozenKVMTPContext,
    FrozenKVMTPDraftInput,
    FrozenKVMTPVerifyInput,
)
from sglang.srt.speculative.frozen_kv_mtp_utils import (
    expand_for_topk_draft,
    frozen_kv_target_view,
    position_for_batch,
    select_last_extend_hidden,
    set_frozen_kv_positions,
    target_kv_pool_view,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    draft_tp_context,
    fast_topk,
    select_top_k_tokens,
    spec_stage_span,
)
from sglang.srt.utils import empty_context
from sglang.srt.utils.async_probe import (
    maybe_detect_inf,
    maybe_detect_nan,
    maybe_detect_oob,
)

logger = logging.getLogger(__name__)


class FrozenKVMTPDraftWorker(EagleDraftWorkerBase, TpModelWorker):
    """Frozen-KV MTP draft worker (草稿工作器)。

    核心特性:
        - Assistant (草稿模型) 仅读取 target KV cache，不拥有自己的 KV pool
        - 复用 EAGLE 的 verify input/output 契约 (数据结构兼容)
        - 自行管理种子 (seed) 和循环 draft loop，因为没有 assistant 端的 KV 扩展

    继承关系:
        - EagleDraftWorkerBase: 提供 draft worker 的抽象接口契约
        - TpModelWorker: 提供 TP (Tensor Parallel) 模型工作器的完整实现

    与 EAGLE draft worker 的区别:
        - 无独立 KV pool (复用 target 的 req_to_token_pool)
        - 无 draft-extend 前向传播 (仅种子选择)
        - 需要绑定 frozen KV context 以实现只读 KV 访问

    示例::

        >>> # 在 scheduler 中创建 FrozenKVMTPDraftWorker
        >>> draft_worker = FrozenKVMTPDraftWorker(
        ...     server_args=server_args,
        ...     gpu_id=0,
        ...     tp_rank=0,
        ...     dp_rank=None,
        ...     moe_ep_rank=0,
        ...     attn_cp_rank=0,
        ...     moe_dp_rank=0,
        ...     nccl_port=29500,
        ...     target_worker=target_worker,
        ... )
        >>> draft_worker.init_backends()
        >>> # 之后由 FrozenKVMTPWorkerV2 调用 draft_worker.draft(batch)
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
        """初始化 Frozen-KV MTP draft worker。

        参数:
            server_args: 服务器配置参数，包含 speculative 算法配置
            gpu_id: GPU 设备 ID
            tp_rank: Tensor Parallel rank
            dp_rank: Data Parallel rank (可选)
            moe_ep_rank: MoE Expert Parallel rank
            attn_cp_rank: Attention Context Parallel rank
            moe_dp_rank: MoE Data Parallel rank
            nccl_port: NCCL 通信端口
            target_worker: 目标模型工作器，提供 KV cache 和 embedding/head

        初始化流程:
            1. 保存配置参数 (topk, num_steps, num_draft_tokens 等)
            2. 验证 speculative_algorithm 为 FROZEN_KV_MTP
            3. 复用 target 的 req_to_token_pool 和 KV allocator (只读)
            4. 创建 dummy draft_pool_config (实际不使用)
            5. 调用 TpModelWorker.__init__ 初始化模型
            6. 绑定 target 的 embedding 和 head 到 draft 模型
            7. 绑定 frozen KV context (实现只读 KV 访问)
            8. 初始化 attention backend 和 CUDA graph (设为 None，后续 init_backends 中创建)
        """
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        assert self.speculative_algorithm.is_frozen_kv_mtp(), (
            "FrozenKVMTPDraftWorker 只能在 SpeculativeAlgorithm.FROZEN_KV_MTP 下创建，"
            f"当前算法为 {self.speculative_algorithm.name}."
        )

        # Draft attention 复用 target 的 req_to_token + KV allocator (只读模式)
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # 创建一个 dummy 的 draft pool config (max_total_num_tokens=64)
        # 实际不使用，仅为保持与 TpModelWorker 初始化接口的兼容性
        target_cfg = target_worker.model_runner.memory_pool_config
        self.draft_pool_config = MemoryPoolConfig(
            max_total_num_tokens=64,  # 哑值，实际不分配 draft KV
            max_running_requests=target_cfg.max_running_requests,
        )

        self.hot_token_id = None

        with (
            empty_context()
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            # 注意: 显式调用 TpModelWorker.__init__ -- EagleDraftWorkerBase
            # 是一个没有 __init__ 的 ABC，使用 cooperative super() 会有歧义
            TpModelWorker.__init__(
                self,
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                memory_pool_config=self.draft_pool_config,
            )

        # 从 target 模型获取 embedding 和 head，绑定到 draft 模型
        # 这样 draft 模型可以复用 target 的词嵌入和输出头
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        if hasattr(self.draft_model_runner.model, "set_embed_and_head"):
            self.draft_model_runner.model.set_embed_and_head(embed, head)
        else:
            logger.debug(
                "Draft 模型 %s 未实现 set_embed_and_head; "
                "在 Frozen-KV MTP skeleton 中跳过 target-embedding 绑定。",
                type(self.draft_model_runner.model).__name__,
            )

        # 绑定 frozen KV context: 使 draft 模型能以只读方式访问 target KV cache
        self.kv_context: Optional[FrozenKVMTPContext] = None
        if hasattr(self.draft_model_runner.model, "bind_frozen_kv_context"):
            self._bind_kv_context()

        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )

        # 以下属性在 init_backends() 中初始化，这里先设为 None
        self.draft_attn_backend = None
        self.cuda_graph_runner = None
        # Frozen draft 没有 draft-extend 前向传播 (仅种子选择);
        # 保持 None 以使继承的探针 (spec_v2_attn_backends, adaptive) 类型正确
        self.draft_extend_attn_backend = None
        self.cuda_graph_runner_for_draft_extend = None

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        """分配内存池 (重写)。

        Frozen draft 不拥有自己的 KV pool，复用 target 的内存池。
        此方法仅为满足 TpModelWorker 的初始化接口要求。

        参数:
            memory_pool_config: 内存池配置 (未使用，用 self.draft_pool_config 替代)
            req_to_token_pool: 请求到 token 的映射池 (可选，默认用 target 的)
            token_to_kv_pool_allocator: token 到 KV pool 的分配器 (可选，默认用 target 的)

        示例::

            >>> draft_worker.alloc_memory_pool()
            >>> # 内部调用 TpModelWorker.alloc_memory_pool，使用 dummy pool config
        """
        # 注意: 显式调用 TpModelWorker.alloc_memory_pool --
        # EagleDraftWorkerBase 在 MRO 中优先于它，但其 alloc_memory_pool 是空操作桩
        TpModelWorker.alloc_memory_pool(
            self,
            memory_pool_config=self.draft_pool_config,
            req_to_token_pool=(
                req_to_token_pool
                if req_to_token_pool is not None
                else self.req_to_token_pool
            ),
            token_to_kv_pool_allocator=(
                token_to_kv_pool_allocator
                if token_to_kv_pool_allocator is not None
                else self.token_to_kv_pool_allocator
            ),
        )

    def init_backends(self):
        """初始化后端: attention backend + CUDA graph。

        初始化流程:
            1. 在 draft TP context 和 MoE context 下调用 TpModelWorker.init_backends
               (禁用 CUDA graph，因为 draft 自行管理)
            2. 初始化 draft attention backend (triton 或复用 target backend)
            3. 将 draft_attn_backend 设置到 draft_model_runner
            4. 初始化 CUDA graph (如果支持)

        示例::

            >>> draft_worker.init_backends()
            >>> # 之后 draft_worker.draft_attn_backend 可用
        """
        with (
            self.draft_tp_context(self.draft_model_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            TpModelWorker.init_backends(self, disable_cuda_graph=True)
            self.draft_attn_backend = self._init_draft_attn_backend()
            self.draft_model_runner.draft_attn_backend = self.draft_attn_backend
            self.init_cuda_graphs()

    @property
    def draft_model_runner(self):
        """返回 draft 模型的 runner (等同于 self.model_runner)。"""
        return self.model_runner

    @property
    def draft_runner(self):
        """draft_runner 的别名。

        用于继承的 EAGLEWorkerV2 forward/verify 骨架，
        该骨架通过 `draft_worker.draft_runner` 访问 model_runner。
        """
        return self.model_runner

    def get_attn_backend(self):  # pragma: no cover - exposed for adaptive
        """返回 draft attention backend (供 adaptive 控制器使用)。"""
        return self.draft_attn_backend

    def clear_cache_pool(self):
        """清空 cache pool (空操作)。

        Frozen draft 不拥有 KV pool，无需清空。
        """
        pass

    def _resolve_draft_backend_type(self) -> str:
        """解析 draft attention backend 类型。

        按优先级返回:
            1. speculative_draft_attention_backend (如果配置了专门的 draft backend)
            2. decode_attention_backend (解码阶段的 backend)
            3. attention_backend (通用 backend)

        返回:
            backend 类型字符串，例如 "triton", "flashinfer" 等

        示例::

            >>> backend_type = draft_worker._resolve_draft_backend_type()
            >>> # 如果 server_args.speculative_draft_attention_backend="triton"
            >>> # 则返回 "triton"
        """
        return (
            self.server_args.speculative_draft_attention_backend
            or self.server_args.decode_attention_backend
            or self.server_args.attention_backend
        )

    def _init_draft_attn_backend(self):
        """初始化 draft attention backend。

        根据 topk 值选择不同的初始化路径:
            - topk == 1: 直接复用 target 的 attn_backend (简单路径)
            - topk > 1:  必须使用 triton backend (支持变长 KV 的 topk 扩展)

        返回:
            attention backend 实例

        异常:
            ValueError: 当 topk > 1 但 backend 不是 triton 时抛出

        示例::

            >>> backend = draft_worker._init_draft_attn_backend()
            >>> # topk=1 时返回 target 的 attn_backend
            >>> # topk=8 时返回 TritonAttnBackend 实例
        """
        if self.topk == 1:
            return self.draft_model_runner.attn_backend

        backend_type = self._resolve_draft_backend_type()
        if backend_type != "triton":
            raise ValueError(
                "Frozen-KV MTP topk > 1 当前仅支持 triton attention backend，"
                f"收到 {backend_type}."
            )
        return self._init_triton_draft_attn_backend()

    def _init_triton_draft_attn_backend(self):
        """初始化 Triton attention backend (用于 topk > 1 的场景)。

        Triton backend 支持变长 KV，适合 topk 扩展后的 batch。

        返回:
            TritonAttnBackend 实例

        示例::

            >>> backend = draft_worker._init_triton_draft_attn_backend()
            >>> # 创建 TritonAttnBackend，max_bs = req_pool.size * topk
        """
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        # 最大 batch size = 请求数 × topk (每个请求扩展为 topk 个候选)
        max_bs = self.req_to_token_pool.size * self.topk
        kv_indptr_buf = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=self.draft_model_runner.device
        )
        return TritonAttnBackend(
            self.draft_model_runner,
            skip_prefill=True,
            kv_indptr_buf=kv_indptr_buf,
        )

    def _bind_kv_context(self) -> None:
        """绑定 frozen KV context 到 draft 模型。

        使 draft 模型能以只读方式访问 target 模型的 KV cache。
        如果 draft 模型未实现相关接口，则跳过绑定。

        前置条件:
            draft 模型需实现 build_frozen_kv_mtp_context 和 bind_frozen_kv_context 方法

        示例::

            >>> draft_worker._bind_kv_context()
            >>> # draft_worker.kv_context 被设置
            >>> # draft 模型可通过该 context 只读访问 target KV
        """
        draft_model = self.draft_model_runner.model
        if not hasattr(draft_model, "build_frozen_kv_mtp_context") or not hasattr(
            draft_model, "bind_frozen_kv_context"
        ):
            logger.debug(
                "Draft 模型 %s 未实现 Frozen-KV MTP context hooks; "
                "跳过 frozen-kv 绑定。",
                type(draft_model).__name__,
            )
            return

        ctx = draft_model.build_frozen_kv_mtp_context(
            target_model=self.target_worker.model_runner.model,
            target_token_to_kv_pool=self.target_worker.model_runner.token_to_kv_pool,
        )
        draft_model.bind_frozen_kv_context(ctx)
        self.kv_context = ctx

    def _frozen_kv_target_view(self, forward_batch: ForwardBatch):
        """返回 frozen KV target view 上下文管理器。

        在该上下文中，draft attention backend 以只读方式访问 target KV cache。
        用于初始化 forward metadata。

        参数:
            forward_batch: 前向传播 batch 信息

        返回:
            上下文管理器，在其中 target KV cache 被“冻结”为只读视图

        示例::

            >>> with draft_worker._frozen_kv_target_view(forward_batch):
            ...     draft_worker.draft_attn_backend.init_forward_metadata(forward_batch)
        """
        return frozen_kv_target_view(
            forward_batch, self.kv_context, self.draft_attn_backend
        )

    def _target_kv_pool_view(self, forward_batch: ForwardBatch):
        """返回 target KV pool view 上下文管理器。

        在该上下文中，draft 模型的 forward 可以读取 target 的 KV pool。
        用于种子前向传播和循环推测的前向传播。

        参数:
            forward_batch: 前向传播 batch 信息

        返回:
            上下文管理器，在其中 target KV pool 可被 draft 模型读取

        示例::

            >>> with draft_worker._target_kv_pool_view(forward_batch):
            ...     output = draft_worker.draft_model_runner.forward(forward_batch)
        """
        return target_kv_pool_view(
            forward_batch, self.kv_context, self.draft_attn_backend
        )

    def _set_positions(self, forward_batch: ForwardBatch) -> None:
        """设置 forward batch 中每个 token 的位置编码 (RoPE position)。

        Frozen-KV MTP 的特殊之处:
            - 种子迭代和循环迭代共享相同的 `seq_lens - 1` 位置
            - 因为 draft 不写入 KV，位置不会随迭代递增

        参数:
            forward_batch: 前向传播 batch 信息

        示例::

            >>> draft_worker._set_positions(forward_batch)
            >>> # forward_batch.positions 被设置为 seq_lens - 1
        """
        set_frozen_kv_positions(forward_batch, self.topk)

    def _expand_for_topk_draft(self, forward_batch: ForwardBatch) -> None:
        """将 forward batch 按 topk 扩展。

        对于 topk > 1 的场景，将每个请求的数据复制 topk 份，
        以匹配 topk 候选 token 的并行计算。

        参数:
            forward_batch: 前向传播 batch 信息

        示例::

            >>> # topk=8, batch_size=4
            >>> draft_worker._expand_for_topk_draft(forward_batch)
            >>> # forward_batch 现在有 4*8=32 个“虚拟请求”
        """
        expand_for_topk_draft(forward_batch, self.topk)

    def _position_for_batch(self, batch: ScheduleBatch) -> torch.Tensor:
        """计算 batch 中每个请求的位置编码。

        参数:
            batch: 调度 batch

        返回:
            positions: shape [batch_size] 的位置张量

        示例::

            >>> positions = draft_worker._position_for_batch(batch)
            >>> # positions = tensor([10, 20, 15], device='cuda')
        """
        return position_for_batch(batch)

    @property
    def _recurrent_hidden_size(self) -> int:
        """返回 draft 模型的 backbone hidden size (用于种子 hidden state 分配)。

        用于创建空闲种子时的 hidden_states 张量分配。

        示例::

            >>> hidden_size = draft_worker._recurrent_hidden_size
            >>> # hidden_size = 4096 (取决于模型配置)
        """
        return int(self.draft_model_runner.model.backbone_hidden_size)

    def _init_frozen_kv_metadata(self, forward_batch: ForwardBatch) -> None:
        """初始化 frozen KV 的 forward metadata。

        在 frozen KV target view 上下文中初始化 attention metadata。
        种子迭代和循环迭代共享相同的 metadata (因为位置不随迭代变化)。

        参数:
            forward_batch: 前向传播 batch 信息

        示例::

            >>> draft_worker._init_frozen_kv_metadata(forward_batch)
            >>> # forward_batch 的 seq_lens_sum 被计算
            >>> # draft_attn_backend 的 forward metadata 被初始化
            >>> # forward_batch.mark_forward_metadata_ready() 被调用
        """
        if forward_batch.forward_mode.is_idle():
            return
        if forward_batch.seq_lens_cpu is not None:
            forward_batch.seq_lens_sum = forward_batch.seq_lens_cpu.sum().item()
        else:
            forward_batch.seq_lens_sum = torch.sum(forward_batch.seq_lens).item()
        with self._frozen_kv_target_view(forward_batch):
            self.draft_attn_backend.init_forward_metadata(forward_batch)
        forward_batch.mark_forward_metadata_ready()

    def _init_frozen_kv_metadata_capture_cuda_graph(
        self, forward_batch: ForwardBatch
    ) -> None:
        """在 CUDA graph 捕获阶段初始化 frozen KV forward metadata。

        使用 init_forward_metadata_out_graph (而非 init_forward_metadata)，
        因为 CUDA graph 捕获需要特殊的 metadata 初始化方式。

        参数:
            forward_batch: 前向传播 batch 信息

        示例::

            >>> # 在 CUDA graph 捕获过程中
            >>> draft_worker._init_frozen_kv_metadata_capture_cuda_graph(forward_batch)
        """
        with self._frozen_kv_target_view(forward_batch):
            self.draft_attn_backend.init_forward_metadata_out_graph(
                forward_batch, in_capture=True
            )
        forward_batch.mark_forward_metadata_ready()

    def _init_frozen_kv_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int, seq_lens_sum: int
    ) -> None:
        """在 CUDA graph replay 阶段初始化 frozen KV forward metadata。

        使用 SimpleNamespace 构造一个精简的 forward batch 视图，
        只包含 replay 所需的最少字段。

        参数:
            forward_batch: 原始前向传播 batch 信息
            bs: 实际 batch size (可能小于 forward_batch.batch_size)
            seq_lens_sum: 序列长度总和

        示例::

            >>> # 在 CUDA graph replay 过程中
            >>> draft_worker._init_frozen_kv_metadata_replay_cuda_graph(
            ...     forward_batch, bs=4, seq_lens_sum=120
            ... )
        """
        from types import SimpleNamespace

        # 构造精简的 forward batch 视图，仅包含 replay 所需字段
        fb_view = SimpleNamespace(
            batch_size=bs,
            forward_mode=ForwardMode.DECODE,
            input_ids=getattr(forward_batch, "input_ids", None),
            req_pool_indices=forward_batch.req_pool_indices[:bs],
            seq_lens=forward_batch.seq_lens[:bs],
            seq_lens_sum=seq_lens_sum,
            seq_lens_cpu=(
                forward_batch.seq_lens_cpu[:bs]
                if forward_batch.seq_lens_cpu is not None
                else None
            ),
            encoder_lens=None,
            out_cache_loc=getattr(forward_batch, "out_cache_loc", None),
            spec_info=None,
        )
        with self._frozen_kv_target_view(forward_batch):
            self.draft_attn_backend.init_forward_metadata_out_graph(fb_view)

    def init_cuda_graphs(self) -> None:
        """初始化 CUDA graph (如果支持)。

        条件:
            - CUDA graph 未被完全禁用
            - speculative_num_steps > 1 (单步无需 CUDA graph)
            - 设备为 CUDA

        如果条件不满足，则在 eager 模式下运行 draft loop。

        示例::

            >>> draft_worker.init_cuda_graphs()
            >>> # 如果支持 CUDA graph:
            >>> #   draft_worker.cuda_graph_runner = FrozenKVMTPCudaGraphRunner(...)
            >>> # 否则:
            >>> #   draft_worker.cuda_graph_runner = None (eager 模式)
        """
        if cuda_graph_fully_disabled() or self.speculative_num_steps <= 1:
            return
        if self.target_worker.device != "cuda":
            logger.info(
                "Frozen-KV MTP draft CUDA graph 仅支持 CUDA; "
                "在 %s 上以 eager 模式运行 draft loop。",
                self.target_worker.device,
            )
            return

        from sglang.srt.speculative.frozen_kv_mtp_cuda_graph_runner import (
            FrozenKVMTPCudaGraphRunner,
        )

        logger.info("开始捕获 Frozen-KV MTP draft CUDA graph。")
        self.cuda_graph_runner = FrozenKVMTPCudaGraphRunner(self)
        logger.info("完成捕获 Frozen-KV MTP draft CUDA graph。")

    def _select_last_extend_hidden(
        self, batch: ScheduleBatch, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """从 extend (prefill) 的 hidden states 中选取最后一个 token 的 hidden state。

        用于构建 prefill 后的首轮 decode 种子。

        参数:
            batch: 调度 batch，包含 extend_lens 等信息
            hidden_states: target 模型 prefill 的所有 hidden states

        返回:
            last_hidden: shape [batch_size, hidden_size] 的张量，
                         每个请求最后一个 prompt token 的 hidden state

        示例::

            >>> # batch 有 3 个请求，extend_lens = [10, 20, 15]
            >>> # hidden_states shape = [45, 4096] (45 = 10+20+15)
            >>> last_hidden = draft_worker._select_last_extend_hidden(batch, hidden_states)
            >>> # last_hidden shape = [3, 4096]
        """
        return select_last_extend_hidden(batch, hidden_states)

    def _idle_seed(self) -> FrozenKVMTPDraftInput:
        """创建空闲种子 (idle seed)。

        当没有实际请求需要处理时，返回一个填充零的 FrozenKVMTPDraftInput。
        用于保持 batch 维度的一致性。

        返回:
            FrozenKVMTPDraftInput: 空闲种子，所有字段为零/默认值

        示例::

            >>> idle = draft_worker._idle_seed()
            >>> # idle.bonus_tokens = tensor([0], dtype=torch.int64)
            >>> # idle.hidden_states = tensor([[0, 0, ...]], dtype=torch.float16)
            >>> # idle.capture_hidden_mode = CaptureHiddenMode.LAST
        """
        return FrozenKVMTPDraftInput.create_idle_input(
            device=self.device,
            hidden_size=self._recurrent_hidden_size,
            dtype=self.model_config.dtype,
            topk=self.topk,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )

    def _build_seed_draft_input(
        self,
        last_token_ids: torch.Tensor,
        last_hidden_states: torch.Tensor,
    ) -> FrozenKVMTPDraftInput:
        """从 (bonus token, target hidden) 构建下一轮迭代的种子 FrozenKVMTPDraftInput。

        这里不执行前向传播——种子前向传播在 captured draft graph 中执行
        (参见 ``draft_forward`` 的种子迭代部分)。

        参数:
            last_token_ids: 最后接受的 token ID，shape [batch_size]
            last_hidden_states: 对应的 target hidden state，shape [batch_size, hidden_size]

        返回:
            FrozenKVMTPDraftInput: 包含种子信息的 draft input
            - bonus_tokens: 来自 last_token_ids
            - hidden_states: 来自 last_hidden_states
            - topk_p/topk_index: 零张量 (被后续种子迭代覆盖)

        示例::

            >>> last_token_ids = tensor([123, 456], dtype=torch.int64)
            >>> last_hidden_states = tensor([[0.1, ...], [0.2, ...]], dtype=torch.float16)
            >>> seed = draft_worker._build_seed_draft_input(last_token_ids, last_hidden_states)
            >>> # seed.bonus_tokens = tensor([123, 456], dtype=torch.int64)
            >>> # seed.hidden_states = last_hidden_states
            >>> # seed.topk_p = tensor([[0., ...], [0., ...]], dtype=torch.float32)
        """
        if last_token_ids.numel() == 0:
            return self._idle_seed()

        stashed = FrozenKVMTPDraftInput()
        stashed.bonus_tokens = last_token_ids.to(torch.int64)
        stashed.hidden_states = last_hidden_states
        # 分配真实形状的零张量，以便继承的 `filter_batch`/`merge_batch` 可以在
        # 迭代间对它们进行切片; 这些值会被 captured 种子迭代覆盖
        bs = last_token_ids.shape[0]
        device = last_token_ids.device
        stashed.topk_p = torch.zeros(
            (bs, self.topk), device=device, dtype=torch.float32
        )
        stashed.topk_index = torch.zeros(
            (bs, self.topk), device=device, dtype=torch.int64
        )
        stashed.capture_hidden_mode = CaptureHiddenMode.LAST
        stashed.num_tokens_per_req = 1
        stashed.num_tokens_for_logprob_per_req = 1
        return stashed

    def draft(self, batch: ScheduleBatch):
        """执行 draft 推测，生成待验证的 token 树。

        工作流程:
            1. 如果是 idle 模式，返回 idle verify input
            2. 从 batch.spec_info 获取 FrozenKVMTPDraftInput (包含种子)
            3. 设置 capture_hidden_mode、positions 等
            4. 创建 ForwardBatch 并设置 frozen KV 位置/topk 扩展
            5. 尝试 CUDA graph replay，否则执行 eager draft_forward
            6. 使用 build_tree_kernel_efficient 构建验证树
            7. 返回 FrozenKVMTPVerifyInput

        参数:
            batch: 调度 batch，包含 spec_info (FrozenKVMTPDraftInput)

        返回:
            FrozenKVMTPVerifyInput: 包含推测 token、tree mask、positions 等

        示例::

            >>> # batch.spec_info = FrozenKVMTPDraftInput(bonus_tokens=[123], ...)
            >>> verify_input = draft_worker.draft(batch)
            >>> # verify_input.draft_token shape = [num_draft_tokens]
            >>> # verify_input.custom_mask shape = [num_draft_tokens, num_draft_tokens]
            >>> # verify_input.positions shape = [num_draft_tokens]
        """
        if batch.forward_mode.is_idle():
            return FrozenKVMTPVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        spec_info = batch.spec_info
        assert isinstance(spec_info, FrozenKVMTPDraftInput)

        # 注意: 每轮迭代的记账工作 (penalty 累积, maybe_evict_swa,
        # decode_batch_idx tick) 由继承的 EagleDraftInputV2Mixin.prepare_for_decode
        # 完成 (调度器驱动，参见 ScheduleBatch.prepare_for_decode)，不在这里做——
        # 与 EAGLE v2 一致。在这里重复 evict/tick 会导致双重执行:
        # idx 时钟控制 SWA eviction 时序和 SWA prefix-lock 释放。

        spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        spec_info.num_tokens_per_req = self.topk
        spec_info.num_tokens_for_logprob_per_req = self.topk
        spec_info.positions = self._position_for_batch(batch)
        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()
        batch.return_hidden_states = False

        # 创建 ForwardBatch 并设置 frozen KV 特有的位置和 topk 扩展
        forward_batch = ForwardBatch.init_new(batch, self.draft_model_runner)
        assert forward_batch.capture_hidden_mode == CaptureHiddenMode.LAST
        self._set_positions(forward_batch)
        self._expand_for_topk_draft(forward_batch)

        # 尝试 CUDA graph replay，否则执行 eager draft_forward
        can_run_cuda_graph = self.cuda_graph_runner and self.cuda_graph_runner.can_run(
            forward_batch
        )
        if can_run_cuda_graph:
            parent_list, top_scores_index, draft_tokens = self.cuda_graph_runner.replay(
                forward_batch
            )
        else:
            forward_batch.can_run_dp_cuda_graph = False
            parent_list, top_scores_index, draft_tokens = self.draft_forward(
                forward_batch
            )

        # 使用高效的 tree 构建内核生成验证树
        # tree_mask: token 树的注意力掩码
        # position: 每个 token 的位置编码
        # retrieve_index/next_token/next_sibling: 验证后的接受路径索引
        (
            tree_mask,
            position,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            spec_info.bonus_tokens,
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
        )

        return FrozenKVMTPVerifyInput(
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
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=batch.seq_lens_sum,
            seq_lens_cpu=batch.seq_lens_cpu,
        )

    def draft_forward(self, forward_batch: ForwardBatch):
        """执行 eager 模式的 draft 前向传播 (种子迭代 + 循环推测)。

        工作流程:
            1. 初始化 forward metadata (如果需要)
            2. 种子迭代: 用 (bonus_token, target_h) 前向传播生成 iter-0 的 topk 结果
            3. 循环迭代 (speculative_num_steps 次):
               a. select_top_k_tokens 选取 topk token
               b. 前向传播生成下一轮的 logits 和 hidden states
               c. fast_topk 选取 topk 候选
            4. organize_draft_results 整理结果

        参数:
            forward_batch: 前向传播 batch 信息，包含 spec_info (FrozenKVMTPDraftInput)

        返回:
            tuple: (parent_list, top_scores_index, draft_tokens)
            - parent_list: shape [num_steps, num_draft_tokens] 的父节点索引
            - top_scores_index: shape [num_steps, num_draft_tokens] 的 topk 索引
            - draft_tokens: shape [num_draft_tokens] 的推测 token

        示例::

            >>> # forward_batch.spec_info = FrozenKVMTPDraftInput(...)
            >>> # forward_batch 包含种子 bonus_tokens 和 hidden_states
            >>> parent_list, top_scores_index, draft_tokens = draft_worker.draft_forward(forward_batch)
            >>> # parent_list shape = [num_steps, num_draft_tokens_per_step]
            >>> # draft_tokens shape = [num_draft_tokens]
        """
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, FrozenKVMTPDraftInput)

        score_list: list[torch.Tensor] = []
        token_list: list[torch.Tensor] = []
        parents_list: list[torch.Tensor] = []

        # 种子迭代和循环迭代共享相同的 `seq_lens - 1` rope 位置，
        # 因此一次 init 覆盖整个循环。即使 num_steps == 1 也必须执行。
        if forward_batch.needs_forward_metadata_init():
            self._init_frozen_kv_metadata(forward_batch)

        # 种子迭代: assistant 对 (bonus_token, target_h) 前向传播，
        # 生成 iter-0 的 (topk_p, topk_index, hidden_states)。
        # 对于 topk>1，复制到 bs*topk 以匹配 kernel 形状，然后按请求切回。
        bonus_tokens = spec_info.bonus_tokens
        target_hidden = spec_info.hidden_states
        if self.topk > 1:
            seed_input_ids = bonus_tokens.repeat_interleave(self.topk, dim=0)
            seed_prev_hidden = target_hidden.repeat_interleave(self.topk, dim=0)
        else:
            seed_input_ids = bonus_tokens
            seed_prev_hidden = target_hidden

        forward_batch.input_ids = seed_input_ids
        forward_batch.spec_info.hidden_states = seed_prev_hidden
        self._set_positions(forward_batch)

        # 种子前向传播: 在 target KV pool view 上下文中执行 draft 模型 forward
        with (
            self._target_kv_pool_view(forward_batch),
            forward_context(ForwardContext(attn_backend=self.draft_attn_backend)),
        ):
            seed_output = self.draft_model_runner.forward(forward_batch).logits_output

        # 检测种子迭代的 NaN
        maybe_detect_nan(
            seed_output.next_token_logits, "frozen_kv_mtp_draft: seed iter"
        )

        # topk > 1 时，从 bs*topk 个输出中按步长提取每请求一个结果
        if self.topk > 1:
            seed_next_logits = seed_output.next_token_logits[:: self.topk]
            seed_hidden_per_req = seed_output.hidden_states[:: self.topk]
        else:
            seed_next_logits = seed_output.next_token_logits
            seed_hidden_per_req = seed_output.hidden_states

        # softmax + fast_topk 选取 topk 候选 token
        probs = torch.softmax(seed_next_logits, dim=-1)
        topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
        # 检测 topk_index 是否越界
        maybe_detect_oob(
            topk_index,
            0,
            seed_next_logits.shape[-1],
            "frozen_kv_mtp_draft: seed topk_index OOB",
        )
        hidden_states = seed_hidden_per_req

        scores = None
        # 循环推测: speculative_num_steps 次迭代
        for i in range(self.speculative_num_steps):
            # 选取 topk token 并更新 scores/hidden_states
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            # 最后一步不需要前向传播
            if i == self.speculative_num_steps - 1:
                break

            # 设置下一轮迭代的 input_ids 和 hidden_states
            forward_batch.input_ids = input_ids
            forward_batch.spec_info.hidden_states = hidden_states
            self._set_positions(forward_batch)

            # 循环迭代前向传播: 读取 target KV 生成 logits 和 hidden states
            with (
                self._target_kv_pool_view(forward_batch),
                forward_context(ForwardContext(attn_backend=self.draft_attn_backend)),
            ):
                logits_output = self.draft_model_runner.forward(
                    forward_batch
                ).logits_output

            # 检测 NaN 和 Inf
            maybe_detect_nan(
                logits_output.next_token_logits, f"frozen_kv_mtp_draft step {i}"
            )
            maybe_detect_inf(
                logits_output.next_token_logits, f"frozen_kv_mtp_draft step {i}"
            )
            # softmax + fast_topk 选取下一轮 topk 候选
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            # 检测 topk_index 越界
            maybe_detect_oob(
                topk_index,
                0,
                logits_output.next_token_logits.shape[-1],
                "frozen_kv_mtp_draft: topk_index OOB",
            )
            hidden_states = logits_output.hidden_states

        # 整理 draft 结果为统一格式
        return organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )

    def draft_extend(self):
        """draft_extend 空操作 (EagleDraftWorkerBase 契约)。

        Frozen draft 没有 draft-KV extend 前向传播;
        orchestrator 直接调用 `_draft_extend_for_{prefill,decode}`。
        """
        pass

    def _draft_extend_for_prefill(
        self,
        batch: ScheduleBatch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        mm_input_embeds: Optional[torch.Tensor] = None,
    ) -> FrozenKVMTPDraftInput:
        """为 prefill 后的首轮 decode 构建种子。

        Frozen draft 不写入 KV (读取 target KV)，因此不像 EAGLE 那样执行
        draft-extend 前向传播: 只需选取最后一个 prompt 的 hidden + bonus token
        并暂存为种子。

        参数:
            batch: 调度 batch
            target_hidden_states: target 模型 prefill 的所有 hidden states
            next_token_ids: target 生成的首个 decode token ID (bonus token)
            mm_input_embeds: 多模态输入嵌入 (未使用，frozen seed 不需要)

        返回:
            FrozenKVMTPDraftInput: 包含种子的 draft input

        示例::

            >>> # batch 是 prefill batch，extend_lens = [10, 20]
            >>> # target_hidden_states shape = [30, 4096]
            >>> # next_token_ids = tensor([123, 456])
            >>> seed = draft_worker._draft_extend_for_prefill(
            ...     batch, target_hidden_states, next_token_ids
            ... )
            >>> # seed.bonus_tokens = tensor([123, 456], dtype=torch.int64)
            >>> # seed.hidden_states shape = [2, 4096]
        """
        del mm_input_embeds  # frozen seed 不需要 input embeds
        if batch.forward_mode.is_idle():
            return self._idle_seed()
        last_hidden = self._select_last_extend_hidden(batch, target_hidden_states)
        return self._build_seed_draft_input(next_token_ids, last_hidden)

    def _draft_extend_for_decode(self, batch: ScheduleBatch, batch_result) -> None:
        """Frozen 的 'draft extend' (无前向传播)。

        从 verify 输出中提取最后接受 token 的 target hidden state，
        暂存为下一轮迭代的种子。

        将 verify 的 `EagleDraftInput` 替换为 `FrozenKVMTPDraftInput`，
        以便下一轮 draft 能通过 FROZEN_KV_MTP attn-backend 断言。

        参数:
            batch: 调度 batch
            batch_result: verify 的输出结果，包含:
                - logits_output.hidden_states: verify 的 hidden states
                - accept_lens: 每请求接受的 token 数
                - next_draft_input.bonus_tokens: bonus token

        副作用:
            修改 batch_result.next_draft_input 为 FrozenKVMTPDraftInput

        示例::

            >>> # batch_result.accept_lens = tensor([3, 5])
            >>> # batch_result.next_draft_input.bonus_tokens = tensor([789, 012])
            >>> # num_draft_tokens = 8
            >>> # select_index = [0*8+3-1, 1*8+5-1] = [2, 12]
            >>> # last_hidden = batch_result.logits_output.hidden_states[[2, 12]]
            >>> draft_worker._draft_extend_for_decode(batch, batch_result)
            >>> # batch_result.next_draft_input 现在是 FrozenKVMTPDraftInput
        """
        if batch.forward_mode.is_idle():
            batch_result.next_draft_input = self._idle_seed()
            return

        bs = len(batch.seq_lens)
        # 与 EAGLE 在 draft-extend 输出上使用的相同的 per-req select_index:
        # 每个请求的 num_draft_tokens 宽度块中最后一个接受节点 (accept_lens - 1)。
        # Verify 已将接受路径压缩到前部 (topk > 1) / 它就是前部链 (topk == 1)。
        select_index = (
            torch.arange(bs, device=self.device) * self.speculative_num_draft_tokens
            + batch_result.accept_lens
            - 1
        )
        last_hidden = batch_result.logits_output.hidden_states[select_index]
        bonus_tokens = batch_result.next_draft_input.bonus_tokens
        batch_result.next_draft_input = self._build_seed_draft_input(
            bonus_tokens, last_hidden
        )


class FrozenKVMTPWorkerV2(EAGLEWorkerV2):
    """Spec-v2 (overlap) 编排器 for Frozen-KV MTP。

    复用 ``EAGLEWorkerV2`` 的 verify / ``move_accept_tokens`` / forward 骨架;
    只有 draft worker 和基于种子的 draft-extend 是 Frozen-KV 特有的。

    与 EAGLEWorkerV2 的区别:
        - 不调用 EAGLEWorkerV2.__init__ (那会创建 EagleDraftWorker 及其 KV pool)
        - Frozen draft 不拥有 KV，因此镜像相关设置并创建 FrozenKVMTPDraftWorker
        - 不支持 adaptive speculative decoding

    示例::

        >>> worker = FrozenKVMTPWorkerV2(
        ...     server_args=server_args,
        ...     gpu_id=0,
        ...     tp_rank=0,
        ...     dp_rank=None,
        ...     moe_ep_rank=0,
        ...     attn_cp_rank=0,
        ...     moe_dp_rank=0,
        ...     nccl_port=29500,
        ...     target_worker=target_worker,
        ... )
        >>> # worker.forward_batch_generation(batch) 执行 overlap 调度
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
        # 注意: 有意不调用 EAGLEWorkerV2.__init__ -- 那会创建
        # EagleDraftWorker (带有自己的 draft KV pool)。Frozen draft 不拥有 KV，
        # 因此我们镜像相关设置并创建 FrozenKVMTPDraftWorker。
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

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        # 将 draft 上下文长度匹配到 target (assistant 读取 target KV)
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # 创建 Frozen-KV MTP draft worker
        self._draft_worker = FrozenKVMTPDraftWorker(
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

        # Frozen MTP 尚未接入 adaptive 控制器
        assert (
            not server_args.speculative_adaptive
        ), "Frozen-KV MTP 尚不支持 adaptive speculative decoding。"
        self.adaptive_controller = None

        # 一些哑张量 (与 EAGLEWorkerV2 初始化保持一致)
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    @property
    def spec_v2_attn_backends(self) -> tuple:
        """返回 spec-v2 使用的 attention backends 元组。

        Frozen draft 不涉及 draft-extend backend; 只有 target + draft。

        返回:
            tuple: (target_attn_backend, draft_attn_backend)

        示例::

            >>> backends = worker.spec_v2_attn_backends
            >>> # backends[0] = target 模型的 attn_backend
            >>> # backends[1] = draft 模型的 attn_backend
        """
        # Frozen draft 不涉及 draft-extend backend; 只有 target + draft
        return (
            self._target_worker.model_runner.attn_backend,
            self._draft_worker.draft_attn_backend,
        )

    def forward_batch_generation(self, batch: ScheduleBatch, on_publish=None):
        """执行 forward batch 生成 (overlap 调度)。

        镜像 EAGLEWorkerV2.forward_batch_generation; 唯一的 frozen 特有变化
        是 idle draft-input (FrozenKVMTPDraftInput + recurrent hidden size)。
        draft / 基于种子的 draft-extend 钩子在 FrozenKVMTPDraftWorker 中。

        工作流程:
            - Extend (prefill) 模式:
                1. Target prefill (捕获 FULL hidden)
                2. Publish new_seq_lens
                3. Draft prefill seed (无前向传播)
            - Decode 模式:
                1. Draft forward (种子前向 + 循环推测)
                2. Target verify (验证推测 token)
                3. Publish new_seq_lens
                4. Draft extend for decode (从 verify 结果中选取下一轮种子)

        参数:
            batch: 调度 batch
            on_publish: 可选的回调函数，在 target/verify 完成后调用
                        用于发布 new_seq_lens 作为 overlap fence

        返回:
            batch_output: 包含 logits_output、next_token_ids、next_draft_input 等

        示例::

            >>> # prefill 模式
            >>> batch.forward_mode = ForwardMode.EXTEND
            >>> output = worker.forward_batch_generation(batch)
            >>> # output.next_draft_input = FrozenKVMTPDraftInput (种子)
            >>>
            >>> # decode 模式
            >>> batch.forward_mode = ForwardMode.DECODE
            >>> batch.spec_info = FrozenKVMTPDraftInput(...)
            >>> output = worker.forward_batch_generation(batch)
            >>> # output 包含 verify 结果和下一轮种子
        """
        # Mirrors EAGLEWorkerV2.forward_batch_generation; the only frozen-specific
        # change is the idle draft-input (FrozenKVMTPDraftInput + recurrent hidden
        # size). The draft / seed-based draft-extend hooks are FrozenKVMTPDraftWorker's.
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            # Target prefill (frozen 从不独立运行 -> 捕获 FULL hidden)
            batch.capture_hidden_mode = CaptureHiddenMode.FULL
            batch_output = self.target_worker.forward_batch_generation(batch)

            # Spec_v2 约定: batch.seq_lens = 本次迭代 token 之前的长度
            batch_output.new_seq_lens = batch.seq_lens
            # 在 draft-extend 之前发布，以便 fence 在 target 端
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)

            # Draft prefill seed (无前向传播)
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
            self.activate_step_by_batch(batch.seq_lens.shape[0])

            if batch.spec_info is None:
                batch.spec_info = self.draft_worker._idle_seed()
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
                spec_stage_span("draft"),
            ):
                verify_input = self.draft_worker.draft(batch)
            assert verify_input.is_verify_input()
            batch.spec_info = verify_input
            # 执行 target verify (复用 EAGLEWorkerV2.verify)
            batch_output = self.verify(batch)
            # 在 draft-extend 之前发布，以便 fence 在 verify 端
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
                spec_stage_span("draft_extend"),
            ):
                # Frozen draft extend for decode: 从 verify 结果中选取下一轮种子
                self.draft_worker._draft_extend_for_decode(batch, batch_output)

            return batch_output
