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
"""张量并行（Tensor Parallel）工作节点。

该模块定义了调度器与底层模型运行器（ModelRunner）之间的桥梁：
TpModelWorker 负责持有 ModelRunner、tokenizer/processor、显存池等，
并对外提供前向生成、权重更新、LoRA 加载等能力。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.distributed import get_pp_group, get_world_group
from sglang.srt.managers.io_struct import (
    DestroyWeightsUpdateGroupReqInput,
    GetWeightsByNameReqInput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsUpdateGroupReqInput,
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterReqInput,
    SendWeightsToRemoteInstanceReqInput,
    UnloadLoRAAdapterReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import MultiprocessingSerializer, broadcast_pyobj, set_random_seed
from sglang.srt.utils.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.model_runner_kv_cache_mixin import MemoryPoolConfig

logger = logging.getLogger(__name__)


class BaseTpWorker(ABC):
    """TP 工作节点的抽象基类，封装了所有 worker 共有的、对 ModelRunner 的代理操作。

    子类需实现 forward_batch_generation 与 model_runner 两个抽象成员。
    本类提供权重更新、LoRA 加载、显存池查询、embedding 前向等通用能力。
    """

    @abstractmethod
    def forward_batch_generation(self, forward_batch: ForwardBatch):
        """执行一次生成前向（由子类实现）。"""
        pass

    @property
    @abstractmethod
    def model_runner(self) -> "ModelRunner":
        """返回底层的模型运行器（由子类实现）。"""
        pass

    @property
    def sliding_window_size(self) -> Optional[int]:
        """滑动窗口注意力（SWA）的窗口大小，无 SWA 时为 None。"""
        return self.model_runner.sliding_window_size

    @property
    def is_hybrid_swa(self) -> bool:
        """是否为混合 SWA 模型（部分层用全局注意力、部分层用滑动窗口）。"""
        return self.model_runner.is_hybrid_swa

    def get_tokens_per_layer_info(self):
        """返回全局层与 SWA 层各自可容纳的最大 token 总数。"""
        return (
            self.model_runner.full_max_total_num_tokens,
            self.model_runner.swa_max_total_num_tokens,
        )

    def get_pad_input_ids_func(self):
        """获取模型的 pad_input_ids 函数（多模态模型用于插入占位 token），无则返回 None。"""
        return getattr(self.model_runner.model, "pad_input_ids", None)

    def get_memory_pool(self):
        """返回显存池：请求->token 映射池 与 token->KV 缓存分配器。"""
        return (
            self.model_runner.req_to_token_pool,
            self.model_runner.token_to_kv_pool_allocator,
        )

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """从磁盘加载并热更新模型权重。"""
        success, message = self.model_runner.update_weights_from_disk(
            recv_req.model_path,
            recv_req.load_format,
            recapture_cuda_graph=recv_req.recapture_cuda_graph,
        )
        return success, message

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        """初始化权重更新的分布式通信组（用于从训练端在线同步权重）。"""
        success, message = self.model_runner.init_weights_update_group(
            recv_req.master_address,
            recv_req.master_port,
            recv_req.rank_offset,
            recv_req.world_size,
            recv_req.group_name,
            recv_req.backend,
        )
        return success, message

    def destroy_weights_update_group(self, recv_req: DestroyWeightsUpdateGroupReqInput):
        """销毁权重更新的分布式通信组。"""
        success, message = self.model_runner.destroy_weights_update_group(
            recv_req.group_name,
        )
        return success, message

    def init_weights_send_group_for_remote_instance(
        self, recv_req: InitWeightsSendGroupForRemoteInstanceReqInput
    ):
        """初始化向远端实例发送权重的通信组。"""
        success, message = (
            self.model_runner.init_weights_send_group_for_remote_instance(
                recv_req.master_address,
                recv_req.ports,
                recv_req.group_rank,
                recv_req.world_size,
                recv_req.group_name,
                recv_req.backend,
            )
        )
        return success, message

    def send_weights_to_remote_instance(
        self, recv_req: SendWeightsToRemoteInstanceReqInput
    ):
        """将本实例的权重发送到远端实例。"""
        success, message = self.model_runner.send_weights_to_remote_instance(
            recv_req.master_address,
            recv_req.ports,
            recv_req.group_name,
        )
        return success, message

    def update_weights_from_distributed(
        self, recv_req: UpdateWeightsFromDistributedReqInput
    ):
        """通过分布式通信组（如 NCCL）接收并更新权重。"""
        success, message = self.model_runner.update_weights_from_distributed(
            recv_req.names,
            recv_req.dtypes,
            recv_req.shapes,
            recv_req.group_name,
            recv_req.load_format,
        )
        return success, message

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        """直接用传入的张量（按 tp_rank 切分）更新权重。"""
        # 打补丁以支持跨进程传递张量的序列化/反序列化
        monkey_patch_torch_reductions()
        success, message = self.model_runner.update_weights_from_tensor(
            named_tensors=MultiprocessingSerializer.deserialize(
                recv_req.serialized_named_tensors[self.tp_rank]
            ),
            load_format=recv_req.load_format,
        )
        return success, message

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update weights from IPC for checkpoint-engine integration."""
        success, message = self.model_runner.update_weights_from_ipc(recv_req)
        return success, message

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        """按参数名读取权重（可截断），用于调试/校验。"""
        parameter = self.model_runner.get_weights_by_name(
            recv_req.name, recv_req.truncate_size
        )
        return parameter

    def load_lora_adapter(self, recv_req: LoadLoRAAdapterReqInput):
        """从磁盘路径加载 LoRA 适配器。"""
        result = self.model_runner.load_lora_adapter(recv_req.to_ref())
        return result

    def unload_lora_adapter(self, recv_req: UnloadLoRAAdapterReqInput):
        """卸载已加载的 LoRA 适配器。"""
        result = self.model_runner.unload_lora_adapter(recv_req.to_ref())
        return result

    def load_lora_adapter_from_tensors(
        self, recv_req: LoadLoRAAdapterFromTensorsReqInput
    ):
        """直接从张量加载 LoRA 适配器（支持扁平化 bucket 或普通序列化两种格式）。"""
        # LoRA 代码内部通过 slice_lora_a_weights / slice_lora_b_weights 自行处理 TP 切分
        # （参见 lora/layers.py:46-49、mem_pool.py:437-440）。
        if recv_req.load_format == "flattened_bucket":
            flattened_data = MultiprocessingSerializer.deserialize(
                recv_req.serialized_tensors
            )
            bucket = FlattenedTensorBucket(
                flattened_tensor=flattened_data["flattened_tensor"],
                metadata=flattened_data["metadata"],
            )
            tensors = dict(bucket.reconstruct_tensors())
        else:
            tensors = MultiprocessingSerializer.deserialize(recv_req.serialized_tensors)
        result = self.model_runner.load_lora_adapter_from_tensors(
            recv_req.to_ref(),
            tensors,
            recv_req.config_dict,
            recv_req.added_tokens_config,
        )
        return result

    def forward_batch_embedding(self, model_worker_batch: ModelWorkerBatch):
        """执行 embedding 任务的前向，返回各序列的 embedding 向量。"""
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        logits_output = self.model_runner.forward(forward_batch).logits_output
        embeddings = logits_output.embeddings
        return embeddings


class TpModelWorker(BaseTpWorker):
    """张量并行的模型工作节点。

    每个 TP rank 对应一个 TpModelWorker，它创建并持有 ModelRunner，
    负责实际的模型前向与采样，同时管理 tokenizer、随机种子、显存池与各项容量上限。
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
        memory_pool_config: Optional[MemoryPoolConfig] = None,
        is_multi_layer_eagle: bool = False,
    ):
        # 解析参数
        self.server_args = server_args
        self.tp_size = server_args.tp_size  # 张量并行大小
        self.ep_size = server_args.ep_size  # 专家并行（MoE EP）大小
        self.pp_size = server_args.pp_size  # 流水线并行大小
        self.tp_rank = tp_rank  # 当前 TP rank
        self.moe_ep_rank = moe_ep_rank  # 当前 MoE 专家并行 rank
        self.pp_rank = pp_rank  # 当前流水线 rank
        self.dp_rank = dp_rank  # 当前数据并行 rank
        self.gpu_id = gpu_id  # 本 worker 使用的 GPU 编号
        self.nccl_port = nccl_port  # NCCL 通信端口
        self.is_draft_worker = is_draft_worker  # 是否为投机解码的草稿（draft）worker
        self.is_multi_layer_eagle = is_multi_layer_eagle  # 是否为多层 EAGLE 投机解码
        self.req_to_token_pool = req_to_token_pool  # 请求->token 映射池（可与主 worker 共享）
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator  # token->KV 缓存分配器
        self.memory_pool_config = memory_pool_config  # 显存池配置
        self.attn_cp_rank = attn_cp_rank  # 注意力上下文并行 rank
        self.moe_dp_rank = moe_dp_rank  # MoE 数据并行 rank

        # MTP（多 token 预测）的模型运行器列表
        self.model_runner_list: List[ModelRunner] = []

        self._init_model_config()  # 初始化模型配置
        self._init_model_runner()  # 初始化主模型运行器

        if is_multi_layer_eagle:
            # 多层 EAGLE 需为每一步创建一个草稿模型运行器
            self._init_multi_layer_eagle_model_runners()

        self._init_dllm_algorithm()  # 初始化扩散式 LLM 算法（如启用）

        if server_args.skip_tokenizer_init:
            # 跳过 tokenizer 初始化（调用方自行处理分词）
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
        self.device = self.model_runner.device  # 计算设备（如 cuda）

        # 初始化 NCCL 通信组
        self.pp_group = get_pp_group()  # 流水线并行通信组
        self.world_group = get_world_group()  # 全局通信组

        # 探测/记录各类 token 与请求数量上限
        self.max_total_num_tokens = self.model_runner.max_total_num_tokens  # KV 缓存可容纳的最大 token 总数
        self.max_prefill_tokens = server_args.max_prefill_tokens  # 单次 prefill 的最大 token 数
        self.max_running_requests = self.model_runner.max_running_requests  # 最大同时运行请求数
        assert self.max_running_requests > 0, "max_running_request is zero"
        self.max_queued_requests = server_args.max_queued_requests  # 最大排队请求数
        assert (
            self.max_queued_requests is None or self.max_queued_requests >= 1
        ), "If configured, max_queued_requests must be at least 1 for any work to be scheduled."
        # 单个请求的最大总长度（受上下文长度与显存池大小双重限制）
        self.max_req_len = min(
            self.model_config.context_len - 1,
            self.model_runner.max_token_pool_size - 1,
        )
        # 单个请求的最大输入长度（预留 5 个 token 作为余量）
        self.max_req_input_len = self.max_req_len - 5
        assert (
            self.max_req_len > 0 and self.max_req_input_len > 0
        ), "Memory pool size is too small"

        # 在各 TP worker 间同步随机种子，以保证采样一致性
        self.random_seed = broadcast_pyobj(
            [server_args.random_seed],
            self.tp_size * self.pp_rank + tp_rank,
            self.world_group.cpu_group,
            src=self.world_group.ranks[0],
        )[0]
        set_random_seed(self.random_seed)

        self.enable_overlap = not server_args.disable_overlap_schedule  # 是否启用 overlap 调度
        self.enable_spec = server_args.speculative_algorithm is not None  # 是否启用投机解码
        self.hicache_layer_transfer_counter = None  # HiCache 逐层传输计数器

    def _init_model_config(self):
        """根据 server_args 初始化模型配置（草稿 worker 用投机草稿模型路径）。"""
        from sglang.srt.configs.model_config import ModelConfig

        self.model_config = ModelConfig.from_server_args(
            self.server_args,
            model_path=(
                self.server_args.model_path
                if not self.is_draft_worker
                else self.server_args.speculative_draft_model_path
            ),
            model_revision=(
                self.server_args.revision
                if not self.is_draft_worker
                else self.server_args.speculative_draft_model_revision
            ),
            is_draft_model=self.is_draft_worker,
        )

    def _init_model_runner(self):
        """创建主模型运行器 ModelRunner，传入各并行 rank 与显存池配置。"""
        from sglang.srt.model_executor.model_runner import ModelRunner

        self._model_runner = ModelRunner(
            model_config=self.model_config,
            mem_fraction_static=self.server_args.mem_fraction_static,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            moe_ep_rank=self.moe_ep_rank,
            moe_ep_size=self.ep_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            nccl_port=self.nccl_port,
            dp_rank=self.dp_rank,
            server_args=self.server_args,
            is_draft_worker=self.is_draft_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            memory_pool_config=self.memory_pool_config,
            draft_model_idx=0 if self.is_multi_layer_eagle else None,
        )

    def _init_multi_layer_eagle_model_runners(self):
        """为多层 EAGLE 投机解码创建多个草稿模型运行器（每一投机步一个）。"""
        from sglang.srt.model_executor.model_runner import ModelRunner

        self.model_runner_list.append(self.model_runner)
        for i in range(1, self.server_args.speculative_num_steps):
            self.model_runner_list.append(
                ModelRunner(
                    model_config=self.model_config,
                    mem_fraction_static=self.server_args.mem_fraction_static,
                    gpu_id=self.gpu_id,
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                    moe_ep_rank=self.moe_ep_rank,
                    moe_ep_size=self.ep_size,
                    pp_rank=self.pp_rank,
                    pp_size=self.pp_size,
                    nccl_port=self.nccl_port,
                    dp_rank=self.dp_rank,
                    server_args=self.server_args,
                    is_draft_worker=self.is_draft_worker,
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    memory_pool_config=self.memory_pool_config,
                    draft_model_idx=i,
                )
            )

    def _init_dllm_algorithm(self):
        """初始化扩散式 LLM（diffusion LLM）算法，未配置时为 None。"""
        from sglang.srt.dllm.algorithm.base import DllmAlgorithm

        if self.server_args.dllm_algorithm is not None:
            self.dllm_algorithm = DllmAlgorithm.from_server_args(self.server_args)
        else:
            self.dllm_algorithm = None

    @property
    def model_runner(self) -> "ModelRunner":
        """返回主模型运行器。"""
        return self._model_runner

    def register_hicache_layer_transfer_counter(self, counter: LayerDoneCounter):
        """注册 HiCache 的逐层传输计数器（用于同步分层 KV 加载进度）。"""
        self.hicache_layer_transfer_counter = counter

    def set_hicache_consumer(self, consumer_index: int):
        """设置当前 HiCache 消费者索引。"""
        if self.hicache_layer_transfer_counter is not None:
            self.hicache_layer_transfer_counter.set_consumer(consumer_index)

    def register_hisparse_coordinator(self, coordinator):
        """注册 HiSparse 协调器到模型运行器。"""
        self.model_runner.hisparse_coordinator = coordinator

    def get_worker_info(self):
        """返回 worker 的各项容量/配置信息，供调度器初始化使用。"""
        return (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_queued_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            self.model_runner.forward_stream,
            self.model_runner.req_to_token_pool.size,
            self.model_runner.req_to_token_pool.max_context_len,
            self.model_runner.token_to_kv_pool.size,
        )

    def is_dllm(self):
        """是否启用了扩散式 LLM 算法。"""
        return self.dllm_algorithm is not None

    def _forward_batch_generation_dllm(
        self, forward_batch: ForwardBatch
    ) -> GenerationBatchResult:
        """扩散式 LLM 的生成前向：委托给 dllm_algorithm 辐代去噪生成 token。"""
        logits_output, next_token_ids, can_run_cuda_graph = self.dllm_algorithm.run(
            self.model_runner, forward_batch
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        forward_batch: Optional[ForwardBatch] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        is_verify: bool = False,
        skip_attn_backend_init=False,
    ) -> GenerationBatchResult:
        """执行一次生成前向：构造 ForwardBatch → 模型前向 → （最后一个 PP rank）采样出下一个 token。

        返回 GenerationBatchResult，包含 logits、next_token_ids、是否走了 CUDA Graph 等。
        """
        # FIXME(lsyin): 或许可从 forward_batch_generation 中移除 skip_attn_backend_init，
        #               这需要把准备 replay 始终放在本函数中

        # 从 model worker batch 构造 forward batch
        if model_worker_batch is not None:
            # 将 HiCache 的消费者索引更新到当前运行批次
            self.set_hicache_consumer(model_worker_batch.hicache_consumer_index)

            forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        else:
            # FIXME(lsyin): 统一 forward_batch 的接口
            assert forward_batch is not None

        if self.is_dllm():
            # 扩散式 LLM 走独立的生成路径
            return self._forward_batch_generation_dllm(forward_batch)

        if self.pp_group.is_last_rank:
            # 最后一个流水线 rank：负责产出 logits 并采样
            out = self.model_runner.forward(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
                skip_attn_backend_init=skip_attn_backend_init,
            )
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            batch_result = GenerationBatchResult(
                logits_output=logits_output,
                can_run_cuda_graph=can_run_cuda_graph,
                expert_distribution_metrics=out.expert_distribution_metrics,
            )

            if is_verify:
                # 投机解码的目标验证前向：跳过采样，直接返回 logits
                return batch_result

            if (
                self.enable_overlap
                and not self.enable_spec
                and model_worker_batch.sampling_info.grammars is not None
            ):
                # overlap 调度 + 含 grammar 约束时，将采样延迟到 grammar 准备好之后再执行
                def sample_batch_func():
                    batch_result.next_token_ids = self.model_runner.sample(
                        logits_output, forward_batch
                    )
                    return batch_result

                batch_result.delay_sample_func = sample_batch_func
                return batch_result

            if not model_worker_batch.is_prefill_only:
                # 普通请求：采样出下一个 token id
                batch_result.next_token_ids = self.model_runner.sample(
                    logits_output, forward_batch
                )
            else:
                # 仅 prefill 请求：在 CPU 上创建占位 token id
                # 大小应与批次大小（序列数）一致，而非 total tokens
                batch_result.next_token_ids = torch.zeros(
                    len(model_worker_batch.seq_lens),
                    dtype=torch.long,
                    device=model_worker_batch.input_ids.device,
                )
                if (
                    model_worker_batch.return_logprob
                    and logits_output.next_token_logits is not None
                ):
                    # 注：不做完整采样，仅计算 logprob
                    self.model_runner.compute_logprobs_only(
                        logits_output, model_worker_batch
                    )

            return batch_result
        else:
            # 非最后一个流水线 rank：只返回中间隐藏状态代理张量，传给下一个 rank
            out = self.model_runner.forward(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
                skip_attn_backend_init=skip_attn_backend_init,
            )
            pp_proxy_tensors, can_run_cuda_graph = out.logits_output, out.can_run_graph
            return GenerationBatchResult(
                pp_hidden_states_proxy_tensors=pp_proxy_tensors,
                can_run_cuda_graph=can_run_cuda_graph,
                expert_distribution_metrics=out.expert_distribution_metrics,
            )

    def forward_batch_split_prefill(self, batch: ScheduleBatch):
        """拆分 prefill 的前向：首次（split_index==0）构造并缓存 forward batch，后续复用。"""
        if batch.split_index == 0:
            model_worker_batch = batch.get_model_worker_batch()
            forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
            batch.split_forward_batch = forward_batch
            batch.seq_lens_cpu_cache = model_worker_batch.seq_lens_cpu
        else:
            model_worker_batch = batch.get_model_worker_batch(batch.seq_lens_cpu_cache)

        out = self.model_runner.forward(
            batch.split_forward_batch, split_forward_count=batch.split_forward_count
        )
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        if logits_output:
            next_token_ids = self.model_runner.sample(logits_output, model_worker_batch)
        else:
            next_token_ids = None
        batch_result = GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=can_run_cuda_graph,
            expert_distribution_metrics=out.expert_distribution_metrics,
        )
        batch_result.next_token_ids = next_token_ids
        return batch_result
