# Copyright 2023-2024 SGLang Team
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

"""多层 EAGLE V2 投机解码 Worker 模块。

本模块实现多层 EAGLE（Multi-Layer EAGLE）投机解码的核心逻辑，
包括草稿模型的多步前向、目标验证、草稿扩展等。

多层 EAGLE 与标准 EAGLE 的区别：
    - 草稿模型有多个 transformer 层，每层独立运行并产生候选 token
    - 支持 chain-style MTP（链式多 token 预测）
    - 使用专用的 Triton kernel 管理隐藏状态池和 input_ids 旋转

主要类：
    - MultiLayerEagleDraftWorker: 草稿模型 worker，管理多层草稿 runner
    - MultiLayerEagleWorkerV2:    顶层 worker，协调目标验证与草稿扩展
"""

import contextlib
import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.graph_runner.multi_layer_eagle_draft_extend_npu_graph_runner import (
    MultiLayerEagleMultiStepDraftExtendNpuGraphRunner,
)
from sglang.srt.layers.moe.utils import speculative_moe_backend_context
from sglang.srt.layers.utils.logprob import compute_spec_v2_logprobs
from sglang.srt.managers.io_struct import (
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromIPCReqInput,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, EagleDraftWorkerBase
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_info import (
    EagleDraftExtendInput,
    EagleDraftInput,
    EagleVerifyInput,
)
from sglang.srt.speculative.eagle_info_v2 import fill_bonus_tokens
from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    build_tree_kernel_efficient,
    eagle_prepare_for_verify,
    eagle_sample,
)
from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (
    MultiLayerEagleMultiStepDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.multi_layer_eagle_utils import (
    assign_hidden_states_pool_triton,
    rotate_input_ids_triton,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    draft_tp_context,
    record_stream_each,
    record_stream_for_v2_verify,
    select_top_k_tokens,
)
from sglang.srt.utils import is_npu
from sglang.srt.utils.async_probe import (
    maybe_detect_inf,
    maybe_detect_nan,
    maybe_detect_oob,
)
from sglang.srt.utils.common import empty_context, fast_topk

_is_npu = is_npu()

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner, ModelRunnerOutput


logger = logging.getLogger(__name__)


def _get_plan_stream(
    device: str,
) -> Tuple[any, contextlib.AbstractContextManager]:
    """获取 overlap 调度所需的 plan stream。

    当启用 SGLANG_ENABLE_OVERLAP_PLAN_STREAM 时，创建独立的 CUDA stream
    用于 prepare 阶段，使 prepare 和前向计算可以重叠执行，从而隐藏
    元数据准备的开销。

    参数：
        device: 设备字符串（如 'cuda'、'npu'）

    返回：(plan_stream, plan_stream_ctx)
        - plan_stream:     CUDA stream 对象（未启用时为 None）
        - plan_stream_ctx: 对应的 stream 上下文管理器（未启用时为 nullcontext）

    示例：
        >>> plan_stream, ctx = _get_plan_stream('cuda')
        >>> with ctx:
        ...     # 在 plan stream 上执行准备操作
        ...     prepare_for_verify(...)
        >>> # 主流等待 plan stream 完成
        >>> torch.cuda.current_stream().wait_stream(plan_stream)
    """
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        plan_stream = torch.get_device_module(device).Stream()
        plan_stream_ctx = torch.get_device_module(device).stream(plan_stream)
        return plan_stream, plan_stream_ctx
    else:
        return None, contextlib.nullcontext()


class MultiLayerEagleDraftWorker(EagleDraftWorkerBase):
    """多层 EAGLE 草稿模型 Worker。

    管理多个草稿模型 runner（每步一个），执行多步草稿推理。
    支持链式 MTP（每步输出隐藏状态传递给下一步）和非链式模式。

    核心属性：
        draft_runner_list:           草稿 runner 列表（每步一个）
        chain_mtp_hidden_states:     是否为链式 MTP 模式
        req_to_hidden_states_pool:   隐藏状态池（用于 KV cache 回滚）
        tree_mask_mode:              树掩码模式
        plan_stream:                 overlap 调度的 plan stream

    示例：
        >>> worker = MultiLayerEagleDraftWorker(server_args, 0, 0, 0, ...)
        >>> # 每轮 decode：
        >>> parent_list, topk_idx, draft_tokens = worker.draft_forward(batch)
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
        """初始化多层 EAGLE 草稿 worker。

        参数：
            server_args: 服务器参数
            gpu_id:      GPU ID
            tp_rank:     张量并行 rank
            dp_rank:     数据并行 rank
            moe_ep_rank: MoE 专家并行 rank
            attn_cp_rank: 注意力上下文并行 rank
            moe_dp_rank: MoE 数据并行 rank
            nccl_port:   NCCL 通信端口
            target_worker: 目标模型 worker（用于共享 embed/head）
        """
        # 复制参数
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.nccl_port = nccl_port
        self.target_worker = target_worker
        self.draft_extend_attn_backend_list = []
        self.model_config = target_worker.model_config

        # 方便访问的参数
        self.device = server_args.device
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        assert self.speculative_num_draft_tokens == self.speculative_num_steps + 1, (
            "多层 EAGLE 要求 speculative_num_draft_tokens == "
            "speculative_num_steps + 1，"
            f"得到 {self.speculative_num_draft_tokens} 和 {self.speculative_num_steps}"
        )
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # 设置常量
        EagleDraftInput.ALLOC_LEN_PER_DECODE = max(
            self.speculative_num_steps * self.topk, self.speculative_num_draft_tokens
        )

        # 仅加载草稿模型权重
        with empty_context(), speculative_moe_backend_context():
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
                is_multi_layer_eagle=True,
            )

        # 别名，提高可读性
        self.draft_runner_list: List[ModelRunner] = self.draft_worker.model_runner_list
        # 匹配 `EagleDraftWorker.draft_runner`，使 `_draft_runner_of(self)`
        # 能用于 EagleDraftInput 的 shape classmethod
        self.draft_runner: ModelRunner = self.draft_runner_list[0]

        # 链式 MTP：每步将其输出隐藏状态传递给下一步。
        # 非链式：每步使用目标模型的隐藏状态。
        draft_arch = self.draft_worker.model_config.hf_config.architectures[0]
        self.chain_mtp_hidden_states = draft_arch in ["Step3p5MTP"]
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
        """分配草稿模型 KV cache 池（由 scheduler 调用）。

        同时初始化隐藏状态回滚缓冲区 req_to_hidden_states_pool，
        用于在 KV cache 回滚时恢复草稿模型的中间隐藏状态。

        参数：
            memory_pool_config:         内存池配置
            req_to_token_pool:          请求→token 映射池
            token_to_kv_pool_allocator: KV cache 池分配器
        """
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.draft_worker.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        self.init_lm_head()

        # KV cache 回滚缓冲区；大小与 req_to_token 对应（按 req_pool_idx 索引）
        self.req_to_hidden_states_pool = torch.empty(
            (
                self.req_to_token_pool.req_to_token.shape[0],
                self.speculative_num_steps - 1,
                self.model_config.hidden_size,
            ),
            dtype=self.model_config.dtype,
            device=self.device,
        )

    def init_backends(self):
        """初始化所有草稿 runner 的后端（注意力、采样器等）。

        在 draft_tp_context（数据并行注意力时为 tp_group）和
        speculative_moe_backend_context 下调用父类 init_backends。
        """
        with self.draft_tp_context(
            self.draft_runner_list[0].tp_group
        ), speculative_moe_backend_context():
            super().init_backends()

    def mtp_model_runner(self, step: int):
        """获取指定步数的草稿 runner。

        参数：
            step: 步数索引（0 ~ num_steps-1）

        返回：对应步数的 ModelRunner

        示例：
            >>> worker.mtp_model_runner(0)  # 第 0 步的 runner
            <ModelRunner ...>
            >>> worker.mtp_model_runner(2)  # 第 2 步的 runner
            <ModelRunner ...>
        """
        return self.draft_runner_list[step]

    def init_lm_head(self):
        """共享目标模型的 embedding 和 lm_head 到所有草稿 runner。

        草稿模型不维护独立的 embedding / lm_head，而是复用目标模型的权重，
        以节省显存并保证 token 映射一致。

        示例：
            >>> worker.init_lm_head()
            >>> # 此后所有 draft_runner_list[i].model 的 embed/head
            >>> # 均指向 target_worker 的 embed/head
        """
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        # 共享 embedding 和 lm_head
        for i in range(self.speculative_num_steps):
            self.draft_runner_list[i].model.set_embed_and_head(embed, head)

    def init_attention_backend(self):
        """为每步草稿 runner 创建注意力后端。

        每步草稿 runner 使用独立的 DraftBackendFactory 创建 extend 阶段
        的注意力后端（如 FlashInfer / TritonAttnBackend 等），
        并将结果设置到 runner.attn_backend。

        示例：
            >>> worker.init_attention_backend()
            >>> # draft_extend_attn_backend_list 现在包含 num_steps 个后端
            >>> len(worker.draft_extend_attn_backend_list)
            4  # speculative_num_steps=4 时
        """
        # 创建注意力后端
        self.draft_extend_attn_backend_list = []
        for step in range(self.speculative_num_steps):
            draft_backend_factory = DraftBackendFactory(
                self.server_args,
                self.draft_runner_list[step],
                self.topk,
                self.speculative_num_steps,
            )
            self.draft_extend_attn_backend_list.append(
                draft_backend_factory.create_draft_extend_backend()
            )
            if self.draft_extend_attn_backend_list[-1] is not None:
                self.draft_runner_list[step].attn_backend = (
                    self.draft_extend_attn_backend_list[-1]
                )

    def init_cuda_graphs(self):
        """捕获 CUDA Graph（或 NPU Graph）。

        为草稿扩展阶段创建专用的 graph runner，支持 CUDA 和 NPU 平台。
        当 CUDA Graph 被禁用时（check_cuda_graph_backend 返回 DISABLED），
        跳过 graph 捕获。

        示例：
            >>> worker.init_cuda_graphs()
            >>> # CUDA 平台：
            >>> type(worker.cuda_graph_runner_for_draft_extend)
            <class 'MultiLayerEagleMultiStepDraftExtendCudaGraphRunner'>
            >>> # NPU 平台：
            <class 'MultiLayerEagleMultiStepDraftExtendNpuGraphRunner'>
        """
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED):
            return

        if not _is_npu:
            self.cuda_graph_runner_for_draft_extend = (
                MultiLayerEagleMultiStepDraftExtendCudaGraphRunner(self)
            )
        else:
            self.cuda_graph_runner_for_draft_extend = (
                MultiLayerEagleMultiStepDraftExtendNpuGraphRunner(self)
            )

    def reset_cuda_graph_buffers(self, forward_batch, batch_result):
        """重置 CUDA Graph 的内部缓冲区。

        在每轮草稿扩展前调用，确保 graph 的静态缓冲区与当前批次状态一致。

        参数：
            forward_batch: 前向批次
            batch_result:  上一阶段（验证）的结果
        """
        if self.cuda_graph_runner_for_draft_extend:
            self.cuda_graph_runner_for_draft_extend.reset_buffers(
                forward_batch, batch_result
            )

    def draft(self, batch: ScheduleBatch):
        """执行草稿推理，生成候选 token 和树掩码。

        参数：
            batch: 调度批次，含 spec_info（EagleDraftInput）

        返回：EagleVerifyInput，含 draft_token / custom_mask / positions / retrieve_index 等
        """
        draft_input: EagleDraftInput = batch.spec_info
        forward_batch, can_cuda_graph = self.prepare_for_draft(
            draft_input,
            self.req_to_token_pool,
            batch,
            self.cuda_graph_runner,
            self.draft_runner_list[0],
            self.topk,
            self.speculative_num_steps,
        )

        # 运行草稿推理
        parent_list, top_scores_index, draft_tokens = self.draft_forward(forward_batch)

        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        # 构建树掩码
        # 直接写入 cuda graph buffer 供验证注意力使用
        tree_mask_buf, position_buf = (
            self.target_worker.model_runner.attn_backend.get_verify_buffers_to_fill_after_draft()
        )
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
            batch.seq_lens_sum,
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
        """执行多步草稿前向，生成候选 token 和分数。

        参数：
            forward_batch: 前向批次，含 spec_info（topk_p / topk_index / hidden_states）

        返回：(parent_list, top_scores_index, draft_tokens)
            - parent_list:       每个候选的父节点索引 [bs, num_candidates]
            - top_scores_index:   最高分候选的索引 [bs, num_draft_tokens-1]
            - draft_tokens:       草稿 token [bs, num_draft_tokens-1]
        """
        # 解析参数
        spec_info: EagleDraftInput = forward_batch.spec_info
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")

        # 返回值容器
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # 多步前向
        scores = None
        _, hidden_states, scores, tree_info = select_top_k_tokens(
            0, topk_p, topk_index, hidden_states, scores, self.topk
        )
        if self.speculative_num_steps == 1:
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])
        else:
            for i in range(self.speculative_num_steps):
                score_list.append(tree_info[0][:, :, i].unsqueeze(-1))
                token_index = tree_info[1][:, i].unsqueeze(-1)
                token_list.append(token_index)
                if i == 0:
                    parents_list.append(tree_info[2])
                else:
                    parents_list.append(
                        torch.full(
                            (tree_info[2].size(0), 1),
                            i,
                            dtype=torch.long,
                            device="cuda",
                        )
                    )

        # 整理结果
        score_list = torch.cat(score_list, dim=1).flatten(
            1
        )  # b, n, topk; n= 1 + (num_steps-1) * self.topk
        ss_token_list = torch.cat(
            token_list, dim=1
        )  # b, (self.topk + (num_steps-1) * self.topk)
        top_scores = torch.topk(
            score_list, self.speculative_num_draft_tokens - 1, dim=-1
        )
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values
        maybe_detect_oob(
            top_scores_index,
            0,
            ss_token_list.shape[1],
            "draft_forward: top_scores_index OOB for gather on ss_token_list",
        )
        draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

        if len(parents_list) > 1:
            parent_list = torch.cat(parents_list[:-1], dim=1)
        else:
            batch_size = parents_list[0].shape[0]
            parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

        return parent_list, top_scores_index, draft_tokens

    def draft_extend(self):
        """草稿扩展占位方法（V2 中逻辑由 _draft_extend_for_decode / _draft_extend_for_prefill 实现）。

        保留以兼容父类接口。
        """
        pass

    def _draft_extend_for_prefill(
        self,
        batch: ScheduleBatch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
    ):
        """在 prefill 阶段运行草稿模型 extend，正确填充 KV cache。

        参数：
            batch:                调度批次
            target_hidden_states: 目标模型前向的隐藏状态
            next_token_ids:       目标前向生成的下一 token id

        返回：EagleDraftInput，含多步 topk_p / topk_index / hidden_states / bonus_tokens
        """
        # 草稿 embed 会无条件 clamp（为容忍多模态 pad 哨兵），
        # 所以先探测 next_token_ids——否则损坏的 id 会被 clamp 掉而非报错。
        maybe_detect_oob(
            next_token_ids,
            0,
            self.model_config.vocab_size,
            "draft_extend_for_prefill: next_token_ids before draft embed",
        )

        # 草稿扩展 spec_info；仅携带 hidden_states 和 shape 信息
        extend_input = EagleDraftExtendInput(
            hidden_states=target_hidden_states,
            # draft 模式与 decode 模式相同，每请求 1 个 token
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )
        batch.spec_info = extend_input

        # 链式 MTP 需要 FULL 以获取所有 token 的隐藏状态；
        # 非链式仅需 LAST（目标模型的隐藏状态）。
        # STANDALONE 全程跳过隐藏状态。
        if self.speculative_algorithm.is_standalone():
            draft_capture_hidden_mode = CaptureHiddenMode.NULL
        elif self.chain_mtp_hidden_states:
            draft_capture_hidden_mode = CaptureHiddenMode.FULL
        else:
            draft_capture_hidden_mode = CaptureHiddenMode.LAST

        # 运行前向
        batch.capture_hidden_mode = draft_capture_hidden_mode
        batch.return_hidden_states_before_norm = True
        forward_batch = ForwardBatch.init_new(batch, self.draft_runner_list[0])

        # 构造 input_ids
        # TODO: 与 PR #26329 相同的 chunked-prefill 链分歧
        if not batch.forward_mode.is_idle():
            rotate_input_ids_triton(
                forward_batch.input_ids,
                forward_batch.extend_start_loc,
                forward_batch.extend_seq_lens,
                next_token_ids,
            )

        topk_p_list = []
        topk_index_list = []
        for step in range(self.speculative_num_steps):
            output: ModelRunnerOutput = self.draft_runner_list[step].forward(
                forward_batch
            )
            maybe_detect_nan(
                output.logits_output.next_token_logits,
                f"draft_extend_for_prefill step {step}",
            )
            maybe_detect_inf(
                output.logits_output.next_token_logits,
                f"draft_extend_for_prefill step {step}",
            )
            probs = torch.softmax(output.logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            topk_p_list.append(topk_p)
            topk_index_list.append(topk_index)
            # 链式：使用本步输出 hidden_states 作为下一步输入
            if (
                self.chain_mtp_hidden_states
                and step < self.speculative_num_steps - 1
                and output.logits_output.hidden_states is not None
            ):
                forward_batch.spec_info.hidden_states = (
                    output.logits_output.hidden_states
                )
            if forward_batch.extend_seq_lens is not None:
                rotate_input_ids_triton(
                    forward_batch.input_ids,
                    forward_batch.extend_start_loc,
                    forward_batch.extend_seq_lens,
                    topk_index,
                )
        next_draft_input = EagleDraftInput(
            topk_p=torch.cat(topk_p_list, dim=1),
            topk_index=torch.cat(topk_index_list, dim=1),
            # Chain-style left the last step's hidden_states on the extend
            # input; non-chain keeps the target hidden states.
            hidden_states=extend_input.hidden_states,
            bonus_tokens=next_token_ids,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )

        # 更新 req_to_hidden_states_pool（用于 KV Cache 回滚）
        if forward_batch.extend_seq_lens is not None:
            assign_hidden_states_pool_triton(
                target_hidden_states,
                forward_batch.req_pool_indices,
                self.req_to_hidden_states_pool,
                self.speculative_num_steps - 1,
                forward_batch.batch_size,
                forward_batch.extend_seq_lens,
                forward_batch.extend_start_loc,
            )
        return next_draft_input

    def _draft_extend_for_decode(
        self, batch: ScheduleBatch, batch_result: GenerationBatchResult
    ):
        """在 decode 阶段运行草稿扩展，为下一轮验证准备 topk 和隐藏状态。

        参数：
            batch:        调度批次
            batch_result: 验证阶段的结果（含 accept_lens / hidden_states 等）
        """
        # 批次 2：草稿扩展
        draft_extend_input = EagleDraftExtendInput(
            hidden_states=batch_result.logits_output.hidden_states,
            num_tokens_per_req=self.speculative_num_steps + 1,
            num_tokens_for_logprob_per_req=1,
        )

        # 在独立 stream 中准备草稿扩展
        # 注意这里使用 batch_result.next_token_ids 作为 input_ids
        with self.plan_stream_ctx:
            forward_batch = self.prepare_for_draft_extend(
                draft_extend_input,
                batch,
                batch_result.next_token_ids,
                self.speculative_num_draft_tokens,
                self.draft_runner_list[0],
                self.cuda_graph_runner_for_draft_extend,
            )
            forward_batch.return_hidden_states_before_norm = True

        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )
        # 在主流上运行草稿扩展
        can_cuda_graph = (
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run(forward_batch)
        )
        ret_topk_p_list = []
        ret_topk_index_list = []
        next_token_ids_backup = batch_result.next_token_ids.clone()

        if can_cuda_graph:
            self.reset_cuda_graph_buffers(forward_batch, batch_result)
        else:
            logger.warning_once(
                "can't use cuda graph for draft extend! may have correctness issue!"
            )
            select_index = (
                torch.arange(len(batch.seq_lens), device=self.device)
                * self.speculative_num_draft_tokens
                + batch_result.accept_lens
                - 1
            )
            # NOTE: 此非 graph 路径在无预计划的情况下运行每步前向（见上方警告）。
            # 标记 batch 使前向路径继续跳过元数据初始化——保留既有行为。
            # 在 NPU 且 --disable-cuda-graph 时，让每个 draft runner 在
            # forward_extend（post-pad）中初始化自己的元数据
            if not _is_npu or can_cuda_graph:
                forward_batch.mark_forward_metadata_ready()

        for step in range(self.speculative_num_steps):
            # log_info_on_rank0(logger, f"step: {step}, forward_batch.input_ids: {forward_batch.input_ids}")
            if can_cuda_graph:
                draft_logits_output = (
                    self.cuda_graph_runner_for_draft_extend.get_runner(step).replay(
                        forward_batch, init_state=(step == 0)
                    )
                )
                ret_topk_p, ret_topk_index = (
                    draft_logits_output.topk_p,
                    draft_logits_output.topk_index,
                )
            else:
                # Skip relies on the unconditional mark above (pre-existing
                # no-pre-plan behavior preserved verbatim).
                draft_logits_output = self.draft_runner_list[step].forward(
                    forward_batch
                )
                probs = torch.softmax(
                    draft_logits_output.logits_output.next_token_logits[select_index],
                    dim=-1,
                )
                ret_topk_p, ret_topk_index = fast_topk(probs, self.topk, dim=-1)
                # 链式：使用本步输出 hidden_states 作为下一步输入
                if (
                    self.chain_mtp_hidden_states
                    and step < self.speculative_num_steps - 1
                    and draft_logits_output.logits_output.hidden_states is not None
                ):
                    forward_batch.spec_info.hidden_states = (
                        draft_logits_output.logits_output.hidden_states
                    )
                if forward_batch.extend_seq_lens is not None:
                    rotate_input_ids_triton(
                        forward_batch.input_ids,
                        forward_batch.extend_start_loc,
                        forward_batch.extend_seq_lens,
                        ret_topk_index,
                        select_index,
                    )
            ret_topk_p_list.append(ret_topk_p)
            ret_topk_index_list.append(ret_topk_index)

        # 更新 req_to_hidden_states_pool（用于 KV Cache 回滚）
        if (
            forward_batch.extend_seq_lens is not None
            and self.cuda_graph_runner_for_draft_extend is not None
        ):
            if can_cuda_graph:
                last_runner = self.cuda_graph_runner_for_draft_extend.get_last_runner()
                hidden_states = last_runner.buffers.hidden_states
                req_pool_indices = last_runner.buffers.req_pool_indices
                extend_seq_lens = last_runner.buffers.extend_seq_lens
                extend_start_loc = last_runner.buffers.extend_start_loc
            else:
                hidden_states = draft_logits_output.logits_output.hidden_states
                req_pool_indices = forward_batch.req_pool_indices
                extend_seq_lens = forward_batch.extend_seq_lens
                extend_start_loc = forward_batch.extend_start_loc
            assign_hidden_states_pool_triton(
                hidden_states,
                req_pool_indices,
                self.req_to_hidden_states_pool,
                self.speculative_num_steps - 1,
                forward_batch.batch_size,
                extend_seq_lens,
                extend_start_loc,
            )

        # 重组 spec_info 供下一批次使用
        # draft_logits_output.next_token_logits = draft_logits_output.next_token_logits[
        #     select_index
        # ]
        # draft_logits_output.hidden_states = draft_logits_output.hidden_states[
        #     select_index
        # ]
        batch_result.next_token_ids = next_token_ids_backup
        # 构造返回值
        next_draft_input = batch_result.next_draft_input
        (
            next_draft_input.topk_p,
            next_draft_input.topk_index,
            next_draft_input.hidden_states,
        ) = (
            torch.cat(ret_topk_p_list, dim=1).clone(),
            torch.cat(ret_topk_index_list, dim=1).clone(),
            None,
        )


class MultiLayerEagleWorkerV2(BaseSpecWorker):
    """多层 EAGLE V2 顶层投机解码 Worker。

    继承 BaseSpecWorker，协调目标模型验证与草稿模型扩展。
    管理一个 MultiLayerEagleDraftWorker 实例，
    在每轮 decode 中执行「草稿→验证→草稿扩展」循环。

    核心属性：
        _target_worker: 目标模型 worker
        _draft_worker:  多层草稿模型 worker
        topk:           每步每请求候选数
        speculative_num_steps:      草稿推理步数
        speculative_num_draft_tokens: 草稿 token 总数（含 bonus）

    示例：
        >>> worker = MultiLayerEagleWorkerV2(server_args, 0, 0, 0, ...)
        >>> result = worker.forward_batch_generation(batch)
        >>> # result.next_token_ids = 验证接受的 token
        >>> # result.accept_lens = 各请求接受长度
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
        """初始化多层 EAGLE V2 worker。

        参数：
            server_args:   服务器参数
            gpu_id:        GPU ID
            tp_rank:       张量并行 rank
            dp_rank:       数据并行 rank（可选）
            moe_ep_rank:   MoE 专家并行 rank
            attn_cp_rank:  注意力上下文并行 rank
            moe_dp_rank:   MoE 数据并行 rank
            nccl_port:     NCCL 通信端口
            target_worker: 目标模型 worker
        """
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

        self._draft_worker = MultiLayerEagleDraftWorker(
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

        # 一些占位张量
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        """分配 KV cache 内存池（委托给草稿 worker）。

        参数：
            memory_pool_config:         内存池配置
            req_to_token_pool:          请求→token 映射池
            token_to_kv_pool_allocator: KV cache 池分配器
        """
        self._draft_worker.alloc_memory_pool(
            memory_pool_config, req_to_token_pool, token_to_kv_pool_allocator
        )
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

    def init_backends(self):
        """初始化草稿 worker 的后端（注意力、采样器等）。"""
        self._draft_worker.init_backends()

    @property
    def target_worker(self):
        """返回目标模型 worker。"""
        return self._target_worker

    @property
    def draft_worker(self):
        """返回草稿模型 worker（MultiLayerEagleDraftWorker）。"""
        return self._draft_worker

    @property
    def spec_v2_attn_backends(self) -> tuple:
        """返回所有 spec_v2 相关的注意力后端元组。

        包含目标模型注意力后端 + 每步草稿 runner 的注意力后端。
        用于统一管理 verify 和 draft_extend 阶段的注意力后端。

        返回：(target_attn_backend, draft_attn_backend_0, draft_attn_backend_1, ...)
        """
        return (
            self._target_worker.model_runner.attn_backend,
            *(
                backend or runner.attn_backend
                for backend, runner in zip(
                    self._draft_worker.draft_extend_attn_backend_list,
                    self._draft_worker.draft_runner_list,
                )
            ),
        )

    def clear_cache_pool(self):
        """清空 KV cache 池（空实现）。

        分配器和 KV cache 池与目标模型共享，由 scheduler 统一清理，
        所以此处无需操作。
        """
        # allocator and kv cache pool are shared with target worker, which are cleared in scheduler
        pass

    def forward_batch_generation(self, batch: ScheduleBatch, on_publish=None):
        """处理一个前向批次（prefill 或 decode）。

        prefill 阶段：目标模型 prefill → 草稿扩展填充 KV cache。
        decode 阶段：草稿推理 → 目标验证 → 草稿扩展准备下一轮。

        参数：
            batch:      调度批次
            on_publish: 回调函数，在 seq_lens 发布时调用（用于 overlap 同步）

        返回：GenerationBatchResult，含 next_token_ids / accept_lens / next_draft_input 等
        """
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            # 目标 prefill
            target_capture_mode = (
                CaptureHiddenMode.NULL
                if self.speculative_algorithm.is_standalone()
                else CaptureHiddenMode.FULL
            )
            batch.capture_hidden_mode = target_capture_mode
            batch_output = self.target_worker.forward_batch_generation(batch)

            # spec_v2 约定：batch.seq_lens = 本次迭代 token 之前的长度。
            # Extend 处理了 L 个 prompt token；下一轮验证迭代期望相同的 L。
            batch_output.new_seq_lens = batch.seq_lens
            # 在 draft_extend 之前发布，使同步点在 target-end
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)

            # 链式 MTP 需要 FULL 以获取所有 token 隐藏状态；
            # 非链式仅需 LAST（目标模型隐藏状态）
            batch_output.next_draft_input = self.draft_worker._draft_extend_for_prefill(
                batch,
                batch_output.logits_output.hidden_states,
                batch_output.next_token_ids,
            )
            return batch_output
        else:
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
                    topk=self.topk * self.speculative_num_steps,
                    capture_hidden_mode=capture_mode,
                )
            verify_input: EagleVerifyInput = self.draft_worker.draft(batch)
            assert verify_input.is_verify_input()
            batch.spec_info = verify_input
            batch_output = self.verify(batch)
            # 在 draft_extend 之前发布，使同步点在 verify-end
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)
            self.draft_worker._draft_extend_for_decode(batch, batch_output)
            return batch_output

    def verify(
        self,
        batch: ScheduleBatch,
    ):
        """执行目标模型验证前向并采样接受的 token。

        参数：
            batch: 调度批次，含 spec_info（EagleVerifyInput）

        返回：GenerationBatchResult，含 next_token_ids / accept_lens / new_seq_lens 等
        """
        fwd_stream = torch.get_device_module(self.device).current_stream()
        verify_input: EagleVerifyInput = batch.spec_info
        record_stream_for_v2_verify(batch, verify_input, fwd_stream)

        bs = len(batch.seq_lens)

        # 批次 1：目标验证
        # 在独立 stream 中准备目标验证
        with self.plan_stream_ctx:
            verify_forward_batch, can_run_cuda_graph = eagle_prepare_for_verify(
                verify_input,
                self.req_to_token_pool,
                batch,
                self.target_worker,
            )

        # 覆盖 prepare 后的重绑定：draft_token、plan_stream 分配的 out_cache_loc
        record_stream_each((batch.input_ids, batch.out_cache_loc), fwd_stream)

        # 修正 overlap plan 导致的一些 buffer 问题
        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )

            # custom_mask 和 position 等值依赖 draft 的输出，
            # 所以前一步 plan 使用了错误值。这里需要重新运行相关计算来更新。
            self.target_worker.model_runner.attn_backend.update_verify_buffers_to_fill_after_draft(
                verify_input,
                (
                    self.target_worker.model_runner.decode_cuda_graph_runner.bs
                    if can_run_cuda_graph
                    else None
                ),
            )
        # NOTE: 这里无条件跳过元数据初始化，尽管 eagle_prepare_for_verify
        # 仅在 cuda-graph replay_prepare 运行时才 plan。
        # eagle_worker_v2 对非 graph 路径重新初始化（post-pad）；
        # 本 worker 未采用该修复，保留其原有行为。
        # 在 NPU 且 --disable-cuda-graph 时，非 graph 验证需要在
        # forward_extend（post-pad）中初始化元数据
        if not _is_npu or can_run_cuda_graph:
            verify_forward_batch.mark_forward_metadata_ready()
        # 在主流上运行目标验证
        forward_batch_output = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
        )
        logits_output = forward_batch_output.logits_output

        # 采样
        maybe_detect_nan(logits_output.next_token_logits, "verify: target model logits")
        maybe_detect_inf(logits_output.next_token_logits, "verify: target model logits")
        (
            predict,
            accept_lens,
            accept_index,
        ) = eagle_sample(verify_input, batch, logits_output)
        new_seq_lens = batch.seq_lens + accept_lens

        if not batch.forward_mode.is_idle():
            accept_tokens = predict[accept_index]
            bonus_tokens = torch.empty_like(accept_lens, dtype=torch.int32)
            # stride = accept_tokens 每请求宽度 = accept_index.shape[1]
            fill_bonus_tokens[(bs,)](
                accept_tokens,
                accept_lens,
                bonus_tokens,
                accept_index.shape[1],
            )
        else:
            bonus_tokens = torch.empty((0,), device=self.device, dtype=torch.int32)

        if batch.return_logprob and not batch.forward_mode.is_idle():
            compute_spec_v2_logprobs(
                batch, logits_output, predict, accept_index, self.speculative_num_steps
            )

        next_draft_input = EagleDraftInput(bonus_tokens=bonus_tokens)
        # verify_forward_batch 传递性地持有验证时 GPU 张量，
        # 这些张量必须在即将发生的 batch.input_ids 重绑定之前存活。
        # scheduler 通过 extra_keep_alive_refs 在 batch_record_buf 中固定它。
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

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """从磁盘更新所有草稿 runner 的模型权重。

        遍历每步草稿 runner，依次从磁盘加载新权重。
        任一步骤失败则立即返回。

        参数：
            recv_req: 含 model_path / load_format / recapture_cuda_graph

        返回：(success, message)

        示例：
            >>> result = worker.update_weights_from_disk(recv_req)
            >>> result
            (True, 'Succeeded to update model weights.')
        """
        for i in range(self.speculative_num_steps):
            success, message = self._draft_worker.draft_runner_list[
                i
            ].update_weights_from_disk(
                recv_req.model_path,
                recv_req.load_format,
                recapture_cuda_graph=recv_req.recapture_cuda_graph,
            )
            if not success:
                return success, message
        return True, "Succeeded to update model weights."

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """通过 IPC 更新所有草稿 runner 的模型权重。

        遍历每步草稿 runner，依次通过 IPC 接收新权重。
        任一步骤失败则立即返回。

        参数：
            recv_req: IPC 权重更新请求

        返回：(success, message)

        示例：
            >>> result = worker.update_weights_from_ipc(recv_req)
            >>> result
            (True, 'Succeeded to update model weights.')
        """
        for i in range(self.speculative_num_steps):
            success, message = self._draft_worker.draft_runner_list[
                i
            ].update_weights_from_ipc(recv_req)
            if not success:
                return success, message
        return True, "Succeeded to update model weights."
