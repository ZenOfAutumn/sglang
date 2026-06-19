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
"""A tensor parallel worker.

中译：张量并行（tensor parallel, TP）worker。它是 Scheduler 与底层 ModelRunner 之间的
      薄封装层，核心职责是：
      1) 把调度层的 ScheduleBatch 转换为执行层的 ForwardBatch（ForwardBatch.init_new）；
      2) 驱动 ModelRunner.forward 跑前向，再做采样（sample）得到 next_token_ids；
      3) 把结果打包成 GenerationBatchResult 返回给调度器；
      4) 对接 KV 池 / 显存分配器（ReqToTokenPool、TokenToKVPoolAllocator）；
      5) 转发各类权重更新与 LoRA 适配器加载/卸载请求到 ModelRunner。
      每个 TP rank 对应一个本进程内的 worker，多个 rank 通过 NCCL 通信组协作。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Tuple

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
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
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
    from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig

logger = logging.getLogger(__name__)


class BaseTpWorker(ABC):
    """中译：TP worker 抽象基类。定义子类必须实现的两个接口（forward_batch_generation
    与 model_runner 属性），并把大量「转发到 ModelRunner」的通用方法（取内存池、更新权重、
    LoRA 加载等）集中在此处实现，供子类复用。"""

    @abstractmethod
    def forward_batch_generation(self, forward_batch: ForwardBatch):
        # 中译：子类必须实现的前向生成入口（跑一个批次得到下一 token）。
        pass

    @property
    @abstractmethod
    def model_runner(self) -> ModelRunner:
        # 中译：子类必须提供底层 ModelRunner 实例（真正执行模型前向的对象）。
        pass

    @property
    def sliding_window_size(self) -> Optional[int]:
        # 中译：滑动窗口注意力（SWA）的窗口大小，None 表示非滑窗模型。
        return self.model_runner.sliding_window_size

    @property
    def is_hybrid_swa(self) -> bool:
        # 中译：是否为「混合滑窗」模型（部分层全注意力、部分层滑窗）。
        return self.model_runner.is_hybrid_swa

    def get_tokens_per_layer_info(self):
        # 中译：返回每层可缓存的 token 上限信息——全注意力层与滑窗层各自的 max_total_num_tokens。
        return (
            self.model_runner.full_max_total_num_tokens,
            self.model_runner.swa_max_total_num_tokens,
        )

    def get_pad_input_ids_func(self):
        # 中译：取模型自定义的 pad_input_ids 函数（多模态模型常用其插入占位 token），没有则返回 None。
        return getattr(self.model_runner.model, "pad_input_ids", None)

    def get_memory_pool(self) -> Tuple[ReqToTokenPool, BaseTokenToKVPoolAllocator]:
        # 中译：返回两级内存池——请求到 token 的映射池，以及 token 到 KV cache 的分配器。
        return (
            self.model_runner.req_to_token_pool,
            self.model_runner.token_to_kv_pool_allocator,
        )

    # 中译：以下一组 update_weights_* / init_weights_* / *_lora_* 方法都是「转发器」：
    #       把 Scheduler 收到的请求拆包，调用 ModelRunner 上对应的实现，再回传 (success, message)。
    #       常见于在线权重热更新（RLHF / 训练-推理共置）与 LoRA 适配器的动态加载/卸载。

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        # 中译：从磁盘重新加载权重（如训练产出新 checkpoint 后热更新）。
        success, message = self.model_runner.update_weights_from_disk(
            recv_req.model_path,
            recv_req.load_format,
            recapture_cuda_graph=recv_req.recapture_cuda_graph,
        )
        return success, message

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        # 中译：初始化「权重更新用」的分布式通信组（如训练进程通过 NCCL 向推理进程广播权重前的建组）。
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
        # 中译：销毁之前建立的权重更新通信组，释放资源。
        success, message = self.model_runner.destroy_weights_update_group(
            recv_req.group_name,
        )
        return success, message

    def init_weights_send_group_for_remote_instance(
        self, recv_req: InitWeightsSendGroupForRemoteInstanceReqInput
    ):
        # 中译：为「向远端实例发送权重」建立通信组（如分离式部署中把权重推送到另一推理实例）。
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
        # 中译：通过上面建立的发送组，把本实例权重实际推送到远端实例。
        success, message = self.model_runner.send_weights_to_remote_instance(
            recv_req.master_address,
            recv_req.ports,
            recv_req.group_name,
        )
        return success, message

    def update_weights_from_distributed(
        self, recv_req: UpdateWeightsFromDistributedReqInput
    ):
        # 中译：从分布式通信组接收并更新权重（按名称/dtype/shape 逐张量同步，常用于在线 RLHF 训推同步）。
        success, message = self.model_runner.update_weights_from_distributed(
            recv_req.names,
            recv_req.dtypes,
            recv_req.shapes,
            recv_req.group_name,
            recv_req.load_format,
        )
        return success, message

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        # 中译：直接用「序列化后的张量」更新权重。

        # 中译：先打补丁，让 torch 的张量序列化（reduction）支持跨进程共享内存/CUDA IPC 句柄传递。
        monkey_patch_torch_reductions()
        # 中译：每个 TP rank 只反序列化属于自己分片的那一份张量（按 self.tp_rank 索引）。
        success, message = self.model_runner.update_weights_from_tensor(
            named_tensors=MultiprocessingSerializer.deserialize(
                recv_req.serialized_named_tensors[self.tp_rank]
            ),
            load_format=recv_req.load_format,
        )
        return success, message

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update weights from IPC for checkpoint-engine integration.

        中译：通过 IPC（进程间共享 CUDA 显存句柄）更新权重，用于对接 checkpoint-engine，
              避免权重在进程间反复拷贝/落盘。
        """
        success, message = self.model_runner.update_weights_from_ipc(recv_req)
        return success, message

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        # 中译：按参数名取出（可选截断的）权重张量，多用于调试/校验权重是否更新成功。
        parameter = self.model_runner.get_weights_by_name(
            recv_req.name, recv_req.truncate_size
        )
        return parameter

    def load_lora_adapter(self, recv_req: LoadLoRAAdapterReqInput):
        # 中译：从磁盘动态加载一个 LoRA 适配器（运行时挂载，无需重启服务）。
        result = self.model_runner.load_lora_adapter(recv_req.to_ref())
        return result

    def unload_lora_adapter(self, recv_req: UnloadLoRAAdapterReqInput):
        # 中译：卸载已加载的 LoRA 适配器，释放其显存。
        result = self.model_runner.unload_lora_adapter(recv_req.to_ref())
        return result

    def load_lora_adapter_from_tensors(
        self, recv_req: LoadLoRAAdapterFromTensorsReqInput
    ):
        # 中译：直接用内存中的张量（而非磁盘文件）加载 LoRA 适配器。
        # The LoRA code handles TP sharding internally using slice_lora_a_weights
        # and slice_lora_b_weights methods (see lora/layers.py:46-49, mem_pool.py:437-440).
        # 中译：LoRA 代码会在内部用 slice_lora_a/b_weights 自行处理 TP 分片，这里无需手动切分。
        if recv_req.load_format == "flattened_bucket":
            # 中译：flattened_bucket 格式——多个张量被打包进一个扁平张量+元数据，需先解包重建。
            flattened_data = MultiprocessingSerializer.deserialize(
                recv_req.serialized_tensors
            )
            bucket = FlattenedTensorBucket(
                flattened_tensor=flattened_data["flattened_tensor"],
                metadata=flattened_data["metadata"],
            )
            tensors = dict(bucket.reconstruct_tensors())
        else:
            # 中译：普通格式——直接反序列化为 {名称: 张量} 字典。
            tensors = MultiprocessingSerializer.deserialize(recv_req.serialized_tensors)
        result = self.model_runner.load_lora_adapter_from_tensors(
            recv_req.to_ref(),
            tensors,
            recv_req.config_dict,
            recv_req.added_tokens_config,
        )
        return result

    def forward_batch_embedding(self, batch: ScheduleBatch):
        # 中译：embedding 模型的前向入口。同样先把 ScheduleBatch 转成 ForwardBatch，再跑前向；
        #       但无需采样，直接取池化后的输出（logits_output 此处即 EmbeddingPoolerOutput）。
        forward_batch = ForwardBatch.init_new(batch, self.model_runner)
        output = self.model_runner.forward(forward_batch).logits_output
        return output  # Returns EmbeddingPoolerOutput


class TpModelWorker(BaseTpWorker):
    """A tensor parallel model worker.

    中译：张量并行模型 worker 的具体实现（生成类模型）。负责持有 ModelRunner、分词器/处理器、
          NCCL 通信组与随机种子等运行时状态，并实现 forward_batch_generation 等前向接口。
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
        # Parse args
        # 中译：解析并暂存各类并行维度的 size 与本 worker 所处的 rank。
        #       tp=张量并行、ep=专家并行(MoE)、pp=流水并行、dp=数据并行、cp=上下文并行。
        self.server_args = server_args
        self.tp_size = server_args.tp_size
        self.ep_size = server_args.ep_size
        self.pp_size = server_args.pp_size
        self.tp_rank = tp_rank
        self.moe_ep_rank = moe_ep_rank
        self.pp_rank = pp_rank
        self.dp_rank = dp_rank
        self.gpu_id = gpu_id
        self.nccl_port = nccl_port
        # 中译：is_draft_worker 标识本 worker 是否为投机解码（speculative）中的 draft 模型 worker。
        self.is_draft_worker = is_draft_worker
        # 中译：is_multi_layer_eagle 表示多层 EAGLE 投机解码（每个 draft 步用一个独立 ModelRunner）。
        self.is_multi_layer_eagle = is_multi_layer_eagle
        # 中译：两级内存池可由外部传入（如 draft worker 复用 target worker 的池），也可后续自行分配。
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank
        # Draft worker: target's resolved MemoryPoolConfig (forwarded to ModelRunner).
        # 中译：draft worker 复用 target（主模型）已确定的内存池配置，直接转发给 ModelRunner。
        self.memory_pool_config = memory_pool_config

        # MTP model runners
        # 中译：MTP（multi-token prediction，如多层 EAGLE）会用到多个 ModelRunner，存于此列表；
        #       常规情况下该列表为空，只用单个 self._model_runner。
        self.model_runner_list: List[ModelRunner] = []

        # 中译：依次构建模型配置与 ModelRunner（真正加载权重、占显存的重操作在此发生）。
        self._init_model_config()
        self._init_model_runner()

        if is_multi_layer_eagle:
            # 中译：多层 EAGLE 时为每个投机步额外创建一个 ModelRunner。
            self._init_multi_layer_eagle_model_runners()

        # 中译：初始化扩散式 LLM（dLLM）算法（若未启用则为 None）。
        self._init_dllm_algorithm()

        if server_args.skip_tokenizer_init:
            # 中译：跳过分词器初始化时（调用方自行传入 token id），分词器与处理器均置 None。
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                # 中译：多模态模型用 processor（同时处理文本与图像/音频），再从中取出文本分词器。
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                    tokenizer_backend=server_args.tokenizer_backend,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                # 中译：纯文本模型只需加载分词器。
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                    tokenizer_backend=server_args.tokenizer_backend,
                )
        self.device = self.model_runner.device

        # Init nccl groups
        # 中译：取得 NCCL 通信组——pp_group（流水并行组）与 world_group（全体进程组）。
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()

        # Sync random seed across TP workers
        # 中译：跨所有 TP/PP worker 同步随机种子。由 0 号 rank 广播，保证各 rank 采样行为一致
        #       （否则同一请求在不同 rank 上采样不同会导致 KV 不一致）。
        self.random_seed = broadcast_pyobj(
            [server_args.random_seed],
            self.tp_size * self.pp_rank + tp_rank,
            self.world_group.cpu_group,
            src=self.world_group.ranks[0],
        )[0]
        set_random_seed(self.random_seed)

        # 中译：是否启用「重叠调度」（overlap schedule，把 CPU 调度与 GPU 前向重叠以提吞吐）。
        self.enable_overlap = not server_args.disable_overlap_schedule
        # 中译：是否启用投机解码。
        self.enable_spec = server_args.speculative_algorithm is not None
        # 中译：分层 KV cache 传输计数器（hierarchical cache 用），默认未注册。
        self.hicache_layer_transfer_counter = None

    def alloc_memory_pool(
        self,
        memory_pool_config: Optional[MemoryPoolConfig] = None,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
    ):
        """Allocate KV cache pools only (no backends or cuda graphs).

        中译：仅分配 KV cache 内存池（不初始化注意力后端、不捕获 cuda graph）。
              把池分配与后端初始化拆成两步，便于在确定显存预算后统一规划池大小。
        """
        if req_to_token_pool is not None:
            self.req_to_token_pool = req_to_token_pool
            self.model_runner.req_to_token_pool = req_to_token_pool
        if token_to_kv_pool_allocator is not None:
            self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
            self.model_runner.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.model_runner.alloc_memory_pool(memory_pool_config)
        # 中译：多 ModelRunner（多层 EAGLE）时，其余 runner 共享同一套内存池，避免重复占显存。
        for mr in self.model_runner_list[1:]:
            mr.req_to_token_pool = self.req_to_token_pool
            mr.token_to_kv_pool_allocator = self.token_to_kv_pool_allocator
            mr.alloc_memory_pool(memory_pool_config)

        # Validation
        # 中译：校验池规模合理——可并发请求数须大于 0，单请求最大长度受上下文长度与池大小双重约束。
        assert self.model_runner.max_running_requests > 0, "max_running_request is zero"
        max_req_len = min(
            self.model_config.context_len - 1,
            self.model_runner.max_token_pool_size - 1,
        )
        assert max_req_len > 0, "Memory pool size is too small"

    def init_backends(self, disable_cuda_graph: bool = False):
        """Initialize attention backends and capture cuda graphs.

        中译：初始化注意力后端并捕获 cuda graph（alloc_memory_pool 之后的第二步）。
        """
        self.model_runner.init_backends(disable_cuda_graph=disable_cuda_graph)
        # 中译：同样对多层 EAGLE 的其余 runner 逐一初始化后端。
        for mr in self.model_runner_list[1:]:
            mr.init_backends(disable_cuda_graph=disable_cuda_graph)

    def _init_model_config(self):
        # 中译：构建 ModelConfig。draft worker 用投机 draft 模型的路径/版本，否则用主模型的。
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
        # 中译：创建主 ModelRunner，传入全部并行 rank/size 与内存池等。这是真正加载权重的重操作。
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
        # 中译：多层 EAGLE：先把主 runner 作为第 0 层放入列表，再为剩余每个投机步各建一个 runner
        #       （draft_model_idx=i 区分不同层）。它们共享内存池，仅模型层不同。
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
        # 中译：若配置了 dLLM（扩散式语言模型）算法则按 server_args 构建，否则置 None。
        from sglang.srt.dllm.algorithm.base import DllmAlgorithm

        if self.server_args.dllm_algorithm is not None:
            self.dllm_algorithm = DllmAlgorithm.from_server_args(self.server_args)
        else:
            self.dllm_algorithm = None

    @property
    def model_runner(self) -> ModelRunner:
        # 中译：实现基类抽象属性，暴露主 ModelRunner。
        return self._model_runner

    def register_hicache_layer_transfer_counter(self, counter: LayerDoneCounter):
        # 中译：注册分层 KV cache 传输的「逐层完成计数器」，供分层缓存场景使用。
        self.hicache_layer_transfer_counter = counter

    def set_hicache_consumer(self, consumer_index: int):
        # 中译：把分层缓存计数器的消费者索引切换到当前正在运行的批次。
        if self.hicache_layer_transfer_counter is not None:
            self.hicache_layer_transfer_counter.set_consumer(consumer_index)

    def register_hisparse_coordinator(self, coordinator):
        # 中译：注册分层稀疏注意力（hisparse）协调器到 ModelRunner。
        self.model_runner.hisparse_coordinator = coordinator

    def get_worker_info(self):
        # 中译：汇总并返回本 worker 的关键运行参数（供调度器初始化容量、限流等），如可缓存 token 总数、
        #       最大 prefill token、最大并发请求、随机种子、前向 stream、各内存池尺寸等。
        max_req_len = min(
            self.model_config.context_len - 1,
            self.model_runner.max_token_pool_size - 1,
        )
        return (
            self.model_runner.max_total_num_tokens,
            self.server_args.max_prefill_tokens,
            self.model_runner.max_running_requests,
            self.server_args.max_queued_requests,
            max_req_len,
            max_req_len - 5,
            self.random_seed,
            self.device,
            self.model_runner.forward_stream,
            self.model_runner.req_to_token_pool.size,
            self.model_runner.req_to_token_pool.max_context_len,
            self.model_runner.token_to_kv_pool.size,
        )

    def is_dllm(self):
        # 中译：是否运行在 dLLM（扩散式语言模型）模式。
        return self.dllm_algorithm is not None

    def _forward_batch_generation_dllm(
        self, forward_batch: ForwardBatch
    ) -> GenerationBatchResult:
        # 中译：dLLM 模式下的前向——由 dllm_algorithm 自行编排多步去噪并完成采样，直接产出 next_token_ids。
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
        batch: Optional[ScheduleBatch],
        forward_batch: Optional[ForwardBatch] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        is_verify: bool = False,
        skip_attn_backend_init: Optional[bool] = None,  # deprecated
    ) -> GenerationBatchResult:
        # 中译：生成类模型前向的核心入口。可传入 ScheduleBatch（调度层批次）或已构造好的 ForwardBatch。
        #       流程：ScheduleBatch → ForwardBatch → ModelRunner.forward → sample → GenerationBatchResult。
        # Get forward batch from schedule batch
        # 中译：关键转换点——把调度层的 ScheduleBatch 转为执行层的 ForwardBatch。
        if batch is not None:
            # update the consumer index of hicache to the running batch
            # 中译：先把分层缓存的消费者索引切到本批次（保证 KV 传输计数对应当前请求）。
            self.set_hicache_consumer(batch.hicache_consumer_index)

            # 中译：ForwardBatch.init_new 据 ScheduleBatch 与 ModelRunner 状态构建本次前向所需的全部张量
            #       （input_ids、positions、KV 索引、采样信息等）。
            forward_batch = ForwardBatch.init_new(batch, self.model_runner)
        else:
            # FIXME(lsyin): unify the interface of forward_batch
            # 中译：未传 ScheduleBatch 时，调用方必须已直接给出 forward_batch。
            assert forward_batch is not None

        # Deprecated kwarg: pre-planners mark the batch themselves now.
        # 中译：兼容已废弃的 skip_attn_backend_init 参数（如今由预规划器自行在 batch 上标记）。
        forward_batch.apply_deprecated_skip_attn_backend_init(skip_attn_backend_init)

        if self.is_dllm():
            # 中译：dLLM 模式走专门分支。
            return self._forward_batch_generation_dllm(forward_batch)

        # 中译：流水并行（PP）下只有「最后一个 rank」持有词表头、能算 logits 并采样；
        #       其余 rank 只产出隐藏状态代理张量（PPProxyTensors）传给下一段流水。
        if self.pp_group.is_last_rank:
            # 中译：跑模型前向。pp_proxy_tensors 为上游 PP rank 传来的隐藏状态（非首段时）。
            out = self.model_runner.forward(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            batch_result = GenerationBatchResult(
                logits_output=logits_output,
                can_run_cuda_graph=can_run_cuda_graph,
                expert_distribution_metrics=out.expert_distribution_metrics,
                routed_experts_output=out.routed_experts_output,
                indexer_topk_output=out.indexer_topk_output,
            )

            if is_verify:
                # Skip sampling; spec_v2 worker fires its own publish post-verify.
                # 中译：投机解码的 verify 阶段——这里只返回 logits，不在此采样；
                #       spec_v2 worker 会在验证完成后自行发布结果。
                return batch_result

            # 中译：特例——重叠调度 + 非投机 + 使用了语法约束（grammar）解码时，把采样延后执行。
            #       原因：grammar 采样需在 CPU 侧与 GPU 结果配合，封成 delay_sample_func 交由调度器择机调用，
            #       以维持「CPU 调度与 GPU 前向重叠」的流水。
            if (
                self.enable_overlap
                and not self.enable_spec
                and forward_batch.sampling_info.grammars is not None
            ):

                def sample_batch_func():
                    batch_result.next_token_ids = self.model_runner.sample(
                        logits_output, forward_batch
                    )
                    return batch_result

                batch_result.delay_sample_func = sample_batch_func
                return batch_result

            if not forward_batch.is_prefill_only:
                # For normal requests, sample the next token ids.
                # 中译：常规请求——基于 logits 采样得到下一个 token id。
                batch_result.next_token_ids = self.model_runner.sample(
                    logits_output, forward_batch
                )
            else:
                # For prefill-only requests, create dummy token IDs on CPU
                # The size should match the batch size (number of sequences), not total tokens
                # 中译：仅 prefill 的请求（如只取 logprob、不续写）无需采样，造一组占位 token id；
                #       长度应等于序列条数（batch size），而非 token 总数。
                batch_result.next_token_ids = torch.zeros(
                    len(forward_batch.seq_lens),
                    dtype=torch.long,
                    device=forward_batch.input_ids.device,
                )
                if (
                    forward_batch.return_logprob
                    and logits_output.next_token_logits is not None
                ):
                    # NOTE: Compute logprobs without full sampling
                    # 中译：需要 logprob 时只计算 logprob，不做完整采样（省去采样开销）。
                    self.model_runner.compute_logprobs_only(
                        logits_output, forward_batch
                    )

            return batch_result
        else:
            # 中译：非最后 PP rank——只跑前向得到隐藏状态，作为代理张量传给下一段流水，不采样。
            out = self.model_runner.forward(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            pp_proxy_tensors, can_run_cuda_graph = out.logits_output, out.can_run_graph
            return GenerationBatchResult(
                pp_hidden_states_proxy_tensors=pp_proxy_tensors,
                can_run_cuda_graph=can_run_cuda_graph,
                expert_distribution_metrics=out.expert_distribution_metrics,
            )

    def forward_batch_split_prefill(self, batch: ScheduleBatch):
        # 中译：分块 prefill（split prefill）——把一次超长 prefill 切成多块、分多次前向以控制显存峰值。
        #       仅在第 0 块时构建 ForwardBatch 并缓存到 batch 上，后续块复用同一个 forward_batch。
        if batch.split_index == 0:
            forward_batch = ForwardBatch.init_new(batch, self.model_runner)
            batch.split_forward_batch = forward_batch

        out = self.model_runner.forward(
            batch.split_forward_batch, split_forward_count=batch.split_forward_count
        )
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        # 中译：只有在分块走到最后、产出 logits 时才采样；中间块没有 logits，next_token_ids 为 None。
        if logits_output:
            next_token_ids = self.model_runner.sample(
                logits_output, batch.split_forward_batch
            )
        else:
            next_token_ids = None
        batch_result = GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=can_run_cuda_graph,
            expert_distribution_metrics=out.expert_distribution_metrics,
        )
        batch_result.next_token_ids = next_token_ids
        return batch_result
