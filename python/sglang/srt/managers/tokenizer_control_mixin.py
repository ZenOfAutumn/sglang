from __future__ import annotations

# 中译：本模块定义 TokenizerControlMixin —— TokenizerManager 的「控制面」混入类。
#       它把所有与调度器（Scheduler）交互的「管理/控制」操作（更新权重、刷新缓存、
#       加载 LoRA、性能分析、查询内部状态、HiCache 存储挂载等）从 TokenizerManager
#       主类中拆分出来。这些操作的共同点：都通过 FanOutCommunicator「扇出」到各个
#       数据并行（DP）rank 的调度器并汇总结果，区别于按请求 id（rid）复用的推理数据面。

import asyncio
import hashlib
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import fastapi

from sglang.srt.managers.communicator import FanOutCommunicator
from sglang.srt.managers.io_struct import (
    AddExternalCorpusReqInput,
    AddExternalCorpusReqOutput,
    AttachHiCacheStorageReqInput,
    AttachHiCacheStorageReqOutput,
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    ClearHiCacheReqInput,
    ClearHiCacheReqOutput,
    CloseSessionReqInput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    DetachHiCacheStorageReqInput,
    DetachHiCacheStorageReqOutput,
    DumperControlReqInput,
    DumperControlReqOutput,
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    ExpertDistributionReqType,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetLoadsReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsSendGroupForRemoteInstanceReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    ListExternalCorporaReqInput,
    ListExternalCorporaReqOutput,
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterFromTensorsReqOutput,
    LoadLoRAAdapterReqInput,
    LoadLoRAAdapterReqOutput,
    LoRAUpdateOutput,
    OpenSessionReqInput,
    ProfileReq,
    ProfileReqOutput,
    ProfileReqType,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    RemoveExternalCorpusReqInput,
    RemoveExternalCorpusReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    SendWeightsToRemoteInstanceReqInput,
    SendWeightsToRemoteInstanceReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    SlowDownReqInput,
    SlowDownReqOutput,
    UnloadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)
from sglang.srt.managers.load_snapshot import LoadSnapshot
from sglang.srt.server_args import LoRARef, ServerArgs
from sglang.srt.utils import get_bool_env_var
from sglang.utils import TypeBasedDispatcher

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)

# Declarative spec: (attr_name_prefix, response_type[, mode])
# Each entry creates self.{prefix}_communicator and registers
# response_type -> communicator.handle_recv in the dispatch table.
# 中译：声明式配置表，每项为 (属性名前缀, 响应类型[, 工作模式])。
#       init_communicators 会据此为每个前缀创建 self.{prefix}_communicator 通信器，
#       并在分发表里登记「响应类型 -> communicator.handle_recv」的路由。
#       mode 缺省为 "queueing"（排队），少数如 get_loads 为 "watching"（监听）。
_COMMUNICATOR_SPECS = [
    ("init_weights_update_group", InitWeightsUpdateGroupReqOutput),
    ("destroy_weights_update_group", DestroyWeightsUpdateGroupReqOutput),
    ("update_weights_from_distributed", UpdateWeightsFromDistributedReqOutput),
    (
        "init_weights_send_group_for_remote_instance",
        InitWeightsSendGroupForRemoteInstanceReqOutput,
    ),
    ("send_weights_to_remote_instance", SendWeightsToRemoteInstanceReqOutput),
    ("update_weights_from_tensor", UpdateWeightsFromTensorReqOutput),
    ("update_weights_from_ipc", UpdateWeightsFromIPCReqOutput),
    ("get_weights_by_name", GetWeightsByNameReqOutput),
    ("release_memory_occupation", ReleaseMemoryOccupationReqOutput),
    ("resume_memory_occupation", ResumeMemoryOccupationReqOutput),
    ("check_weights", CheckWeightsReqOutput),
    ("slow_down", SlowDownReqOutput),
    ("flush_cache", FlushCacheReqOutput),
    ("add_external_corpus", AddExternalCorpusReqOutput),
    ("remove_external_corpus", RemoveExternalCorpusReqOutput),
    ("list_external_corpora", ListExternalCorporaReqOutput),
    ("clear_hicache_storage", ClearHiCacheReqOutput),
    ("attach_hicache_storage", AttachHiCacheStorageReqOutput),
    ("detach_hicache_storage", DetachHiCacheStorageReqOutput),
    ("profile", ProfileReqOutput),
    ("get_internal_state", GetInternalStateReqOutput),
    ("set_internal_state", SetInternalStateReqOutput),
    ("expert_distribution", ExpertDistributionReqOutput),
    ("update_lora_adapter", LoRAUpdateOutput),
    ("get_loads", GetLoadsReqOutput, "watching"),
    ("dumper_control", DumperControlReqOutput),
]


class TokenizerControlMixin:
    """Mixin for TokenizerManager's control-plane operations (weights, cache, lora,
    profile, internal state, etc.) -- everything that talks to the scheduler via
    FanOutCommunicator, as opposed to data-plane inference requests multiplexed by rid.

    中译：TokenizerManager 的「控制面」操作混入类（权重更新、缓存、LoRA、性能分析、
          内部状态等）—— 即所有通过 FanOutCommunicator 与调度器通信的操作，
          区别于按请求 id（rid）多路复用的「数据面」推理请求。
          作为 mixin，所有方法的 self 实际类型都是 TokenizerManager（故签名标注
          self: TokenizerManager），方法依赖 TokenizerManager 提供的属性
          （如 send_to_scheduler、server_args、各种锁、lora_registry 等）。
    """

    def init_communicators(self: TokenizerManager, server_args: ServerArgs):
        """根据 _COMMUNICATOR_SPECS 批量创建各控制操作的扇出通信器并注册分发。

        中译：遍历配置表，为每个前缀创建一个 FanOutCommunicator（扇出到 dp_size 个
              调度器），挂到 self.{prefix}_communicator 上；同时把「响应类型 ->
              通信器的 handle_recv」追加进结果分发器 _result_dispatcher，使调度器返回
              的结果能被正确路由回对应通信器。在 TokenizerManager 初始化时调用一次。
        """
        dispatch_pairs = []
        for spec in _COMMUNICATOR_SPECS:
            name, resp_type = spec[0], spec[1]
            mode = spec[2] if len(spec) > 2 else "queueing"
            comm = FanOutCommunicator(self.send_to_scheduler, server_args.dp_size, mode)
            setattr(self, f"{name}_communicator", comm)
            dispatch_pairs.append((resp_type, comm.handle_recv))
        self._result_dispatcher += TypeBasedDispatcher(dispatch_pairs)

    async def add_external_corpus(
        self: TokenizerManager, obj: AddExternalCorpusReqInput
    ) -> AddExternalCorpusReqOutput:
        """添加外部语料库（仅用于 NGRAM 投机解码）。

        中译：把外部文本/文件预编码为 token 块（token_chunks）后扇出给各调度器加载，
              供 NGRAM 投机解码作为草稿来源。
              - 参数 obj：可携带 file_path（文件路径）或 documents（文档列表）二选一，
                以及可选的 corpus_id；token 总量受 speculative_ngram_external_corpus_max_tokens 限制。
              - 副作用：会在本进程用 tokenizer 编码文本，并把 file_path/documents 清空后再发送
                （只发 token，不发原始文本/路径）；超限时截断并在返回消息中标注。
              - 返回：包含 success、corpus_id、已加载 token 数与消息的输出对象。
        """
        self.auto_create_handle_loop()
        # 中译：仅在启用 NGRAM 投机解码时才支持外部语料库，否则直接返回失败。
        if self.server_args.speculative_algorithm != "NGRAM":
            return AddExternalCorpusReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        truncated = False  # 中译：标记是否因超过 token 上限而被截断。
        try:
            # 中译：未指定语料库 id 时随机生成一个。
            if not obj.corpus_id:
                import uuid

                obj.corpus_id = uuid.uuid4().hex
            # 中译：分支一——从文件读取，按 max_tokens 上限切成 token 块。
            if obj.file_path is not None:
                from sglang.srt.speculative.cpp_ngram.external_corpus import (
                    iter_external_corpus_chunks,
                )

                max_tokens = (
                    self.server_args.speculative_ngram_external_corpus_max_tokens
                )
                obj.token_chunks = list(
                    iter_external_corpus_chunks(
                        obj.file_path, self.tokenizer, max_tokens
                    )
                )
            # 中译：分支二——从内存中的文档列表逐条编码，文档间插入分隔 token，累计到上限即截断。
            elif obj.documents is not None:
                from sglang.srt.speculative.cpp_ngram.external_corpus import (
                    SEPARATOR_TOKEN,
                )

                max_tokens = (
                    self.server_args.speculative_ngram_external_corpus_max_tokens
                )
                token_chunks = []
                total_tokens = 0
                has_prev = False  # 中译：标记前面是否已有文档（用于决定是否插入分隔符）。
                for doc in obj.documents:
                    if not doc:
                        continue
                    token_ids = list(
                        self.tokenizer.encode(doc, add_special_tokens=False)
                    )
                    if not token_ids:
                        continue
                    # 中译：非首篇文档前补一个分隔 token，避免相邻文档的 n-gram 串接。
                    if has_prev:
                        token_ids = [SEPARATOR_TOKEN] + token_ids
                    # 中译：加上本篇会超过上限，则停止并标记截断。
                    if total_tokens + len(token_ids) > max_tokens:
                        truncated = True
                        break
                    token_chunks.append(token_ids)
                    total_tokens += len(token_ids)
                    has_prev = True
                obj.token_chunks = token_chunks
            else:
                return AddExternalCorpusReqOutput(
                    success=False,
                    message="Either file_path or documents must be provided.",
                )
            # 中译：清空原始文件路径与文档，只把已编码好的 token_chunks 扇出给各调度器。
            obj.file_path = None
            obj.documents = None
            results = await self.add_external_corpus_communicator(obj)
            # 中译：合并各 DP rank 的结果（全成功才算成功）。
            all_success, all_message = FanOutCommunicator.merge_results(results)
            if truncated and all_success:
                all_message += f" (truncated: exceeded {max_tokens} token limit)"
            return AddExternalCorpusReqOutput(
                success=all_success,
                corpus_id=results[0].corpus_id if all_success else "",
                message=all_message,
                loaded_token_count=results[0].loaded_token_count if all_success else 0,
            )
        except Exception as e:
            return AddExternalCorpusReqOutput(success=False, message=str(e))

    async def remove_external_corpus(
        self: TokenizerManager, corpus_id: str
    ) -> RemoveExternalCorpusReqOutput:
        """按 corpus_id 移除已加载的外部语料库（仅 NGRAM 投机解码）。

        中译：扇出移除请求到各调度器并合并结果。corpus_id 为待删除语料库标识，
              返回成功标志与合并后的消息。未启用 NGRAM 时直接返回失败。
        """
        self.auto_create_handle_loop()
        if self.server_args.speculative_algorithm != "NGRAM":
            return RemoveExternalCorpusReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        results = await self.remove_external_corpus_communicator(
            RemoveExternalCorpusReqInput(corpus_id=corpus_id)
        )
        all_success, all_message = FanOutCommunicator.merge_results(results)
        return RemoveExternalCorpusReqOutput(success=all_success, message=all_message)

    async def list_external_corpora(
        self: TokenizerManager,
    ) -> ListExternalCorporaReqOutput:
        """列出当前已加载的外部语料库及其 token 数（仅 NGRAM 投机解码）。

        中译：扇出查询并合并结果。由于各 DP rank 加载的语料集合相同，token 计数
              直接取第一个 rank 的结果。未启用 NGRAM 时返回失败。
        """
        self.auto_create_handle_loop()
        if self.server_args.speculative_algorithm != "NGRAM":
            return ListExternalCorporaReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        results = await self.list_external_corpora_communicator(
            ListExternalCorporaReqInput()
        )
        all_success, all_message = FanOutCommunicator.merge_results(results)
        # Merge corpus token counts from all DP ranks (each rank loads the same set).
        # 中译：合并各 DP rank 的语料 token 计数（各 rank 加载的是同一套，取第一个即可）。
        corpus_token_counts = results[0].corpus_token_counts if all_success else {}
        return ListExternalCorporaReqOutput(
            success=all_success,
            corpus_token_counts=corpus_token_counts,
            message=all_message,
        )

    async def flush_cache(
        self: TokenizerManager, timeout_s: Optional[float] = None
    ) -> FlushCacheReqOutput:
        """清空 KV/前缀缓存（RadixCache）。

        中译：扇出刷新缓存请求；timeout_s 为可选的等待超时（秒）。取首个 rank 的结果返回。
        """
        self.auto_create_handle_loop()
        return (
            await self.flush_cache_communicator(FlushCacheReqInput(timeout_s=timeout_s))
        )[0]

    async def clear_hicache_storage(self: TokenizerManager) -> ClearHiCacheReqOutput:
        """Clear the hierarchical cache storage.

        中译：清空分层缓存（HiCache）的存储后端。委托给调度器执行实际清理。
        """
        self.auto_create_handle_loop()
        # Delegate to the scheduler to handle HiCacheStorage clearing
        # 中译：委托调度器处理 HiCacheStorage 的清空。
        return (await self.clear_hicache_storage_communicator(ClearHiCacheReqInput()))[
            0
        ]

    async def attach_hicache_storage(
        self: TokenizerManager,
        hicache_storage_backend: str,
        hicache_storage_backend_extra_config_json: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> AttachHiCacheStorageReqOutput:
        """Attach (enable) HiCache storage backend at runtime.

        中译：运行时挂载（启用）HiCache 存储后端。
              - 参数：存储后端名称及其额外配置 json、预取策略、写策略（均可选）。
              - 副作用：成功后会同步更新 tokenizer 侧的 server_args，保持与调度器侧一致。
              - 返回：含 success 与 message 的输出对象（失败时暂不做部分回滚，见 TODO）。
        """
        self.auto_create_handle_loop()
        results = await self.attach_hicache_storage_communicator(
            AttachHiCacheStorageReqInput(
                hicache_storage_backend=hicache_storage_backend,
                hicache_storage_backend_extra_config_json=hicache_storage_backend_extra_config_json,
                hicache_storage_prefetch_policy=hicache_storage_prefetch_policy,
                hicache_write_policy=hicache_write_policy,
            )
        )

        all_success, all_message = FanOutCommunicator.merge_results(results)
        out = AttachHiCacheStorageReqOutput(success=all_success, message=all_message)
        # TODO: partial rollback if failed
        # 中译：TODO——若部分 rank 失败，目前不做回滚。
        if all_success:
            # Keep tokenizer side server_info consistent with scheduler side.
            # 中译：保持 tokenizer 侧的 server_args 与调度器侧一致（仅在全成功时更新）。
            self.server_args.hicache_storage_backend = hicache_storage_backend
            if hicache_storage_backend_extra_config_json is not None:
                self.server_args.hicache_storage_backend_extra_config = (
                    hicache_storage_backend_extra_config_json
                )
            if hicache_storage_prefetch_policy is not None:
                self.server_args.hicache_storage_prefetch_policy = (
                    hicache_storage_prefetch_policy
                )
            if hicache_write_policy is not None:
                self.server_args.hicache_write_policy = hicache_write_policy
        return out

    async def detach_hicache_storage(
        self: TokenizerManager,
    ) -> DetachHiCacheStorageReqOutput:
        """Detach (disable) HiCache storage backend at runtime.

        中译：运行时卸载（禁用）HiCache 存储后端。成功后把 tokenizer 侧的存储后端配置清空，
              与调度器侧保持一致。
        """
        self.auto_create_handle_loop()
        results = await self.detach_hicache_storage_communicator(
            DetachHiCacheStorageReqInput()
        )

        all_success, all_message = FanOutCommunicator.merge_results(results)
        out = DetachHiCacheStorageReqOutput(success=all_success, message=all_message)
        # TODO: partial rollback if failed
        # 中译：TODO——若部分 rank 失败，目前不做回滚。
        if all_success:
            self.server_args.hicache_storage_backend = None
            self.server_args.hicache_storage_backend_extra_config = None
        return out

    async def start_profile(
        self: TokenizerManager,
        output_dir: Optional[str] = None,
        start_step: Optional[int] = None,
        num_steps: Optional[int] = None,
        activities: Optional[List[str]] = None,
        with_stack: Optional[bool] = None,
        record_shapes: Optional[bool] = None,
        profile_by_stage: bool = False,
        merge_profiles: bool = False,
        profile_prefix: Optional[str] = None,
        profile_stages: Optional[List[str]] = None,
    ):
        """启动性能分析（profiling）。

        中译：组装 START_PROFILE 请求并扇出到各调度器开始采样。
              - 关键参数：output_dir 输出目录、start_step/num_steps 起始步与步数、
                activities 采样活动类别、with_stack 是否记录调用栈、record_shapes 是否记录张量形状、
                profile_by_stage/profile_stages 按阶段分别采样、merge_profiles 合并多个 profile。
              - with_stack/record_shapes：命令行未显式给出时由环境变量
                SGLANG_PROFILE_WITH_STACK / SGLANG_PROFILE_RECORD_SHAPES 决定（默认开）。
              - profile_id 用当前时间戳标识本次采样。
        """
        self.auto_create_handle_loop()
        # 中译：with_stack 的最终取值——只要参数或环境变量任一为 False 则关闭，否则开启。
        env_with_stack: bool = get_bool_env_var("SGLANG_PROFILE_WITH_STACK", "true")
        with_stack = False if with_stack is False or env_with_stack is False else True
        env_record_shapes: bool = get_bool_env_var(
            "SGLANG_PROFILE_RECORD_SHAPES", "true"
        )
        # 中译：record_shapes 同理——参数未显式禁用且环境变量开启时才记录张量形状。
        record_shapes = (record_shapes is not False) and env_record_shapes
        req = ProfileReq(
            type=ProfileReqType.START_PROFILE,
            output_dir=output_dir,
            start_step=start_step,
            num_steps=num_steps,
            activities=activities,
            with_stack=with_stack,
            record_shapes=record_shapes,
            profile_by_stage=profile_by_stage,
            profile_id=str(time.time()),
            merge_profiles=merge_profiles,
            profile_prefix=profile_prefix,
            profile_stages=profile_stages,
        )
        return await self._execute_profile(req)

    async def stop_profile(self: TokenizerManager):
        """停止性能分析。中译：发送 STOP_PROFILE 请求，结束采样并落盘。"""
        self.auto_create_handle_loop()
        req = ProfileReq(type=ProfileReqType.STOP_PROFILE)
        return await self._execute_profile(req)

    async def _execute_profile(self: TokenizerManager, req: ProfileReq):
        """执行一次 profile 请求并校验结果。

        中译：扇出 profile 请求，取首个结果；若失败则抛 RuntimeError（携带错误消息）。
              start_profile/stop_profile 共用此内部方法。
        """
        result = (await self.profile_communicator(req))[0]
        if not result.success:
            raise RuntimeError(result.message)
        return result

    async def start_expert_distribution_record(self: TokenizerManager):
        """开始记录 MoE 专家分布（用于负载均衡分析）。中译：发送 START_RECORD 动作。"""
        self.auto_create_handle_loop()
        req = ExpertDistributionReq(action=ExpertDistributionReqType.START_RECORD)
        await self.expert_distribution_communicator(req)

    async def stop_expert_distribution_record(self: TokenizerManager):
        """停止记录专家分布。中译：发送 STOP_RECORD 动作。"""
        self.auto_create_handle_loop()
        req = ExpertDistributionReq(action=ExpertDistributionReqType.STOP_RECORD)
        await self.expert_distribution_communicator(req)

    async def dump_expert_distribution_record(self: TokenizerManager):
        """导出已记录的专家分布数据。中译：发送 DUMP_RECORD 动作，落盘记录结果。"""
        self.auto_create_handle_loop()
        req = ExpertDistributionReq(action=ExpertDistributionReqType.DUMP_RECORD)
        await self.expert_distribution_communicator(req)

    async def init_weights_update_group(
        self: TokenizerManager,
        obj: InitWeightsUpdateGroupReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        """初始化「分布式权重更新」的通信组。

        中译：为后续从其他训练进程分布式更新权重做准备（建立 process group）。
              要求 dp_size==1 或启用了 dp attention。返回 (是否成功, 消息)。
        """
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for update weights from distributed"

        results = await self.init_weights_update_group_communicator(obj)
        return FanOutCommunicator.merge_results(results)

    async def destroy_weights_update_group(
        self: TokenizerManager,
        obj: DestroyWeightsUpdateGroupReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        """销毁「分布式权重更新」通信组。

        中译：与 init_weights_update_group 相对，释放之前建立的 process group。
              返回 (是否成功, 消息)。
        """
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for destroy parameter update group"

        results = await self.destroy_weights_update_group_communicator(obj)
        return FanOutCommunicator.merge_results(results)

    async def update_weights_from_distributed(
        self: TokenizerManager,
        obj: UpdateWeightsFromDistributedReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        """从分布式通信组接收并热更新模型权重。

        中译：在线（不停机）从外部训练进程拉取新权重。
              - 若请求要求中止所有在途请求（abort_all_requests），先全部中止。
              - 加锁策略：若引擎已暂停（is_pause），在 is_pause_cond 条件锁下更新，
                防止与「取消暂停」竞争；否则取模型更新写锁（model_update_lock.writer_lock）
                独占更新，避免与推理读取并发。
              - 成功且带 weight_version 时，更新权重版本号并在消息中标注。
        """
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for update weights from distributed"

        # 中译：按需中止所有在途请求，避免它们读到「半更新」的权重。
        if obj.abort_all_requests:
            self.abort_request(abort_all=True)

        # Hold is_pause_cond while updating to prevent unpause from racing.
        # 中译：已暂停时持有 is_pause_cond 期间更新，防止「取消暂停」与更新竞争。
        async with self.is_pause_cond:
            is_paused = self.is_pause
            if is_paused:
                results = await self.update_weights_from_distributed_communicator(obj)

        # 中译：未暂停时取写锁独占更新（与推理读路径互斥）。
        if not is_paused:
            async with self.model_update_lock.writer_lock:
                results = await self.update_weights_from_distributed_communicator(obj)

        success, message = FanOutCommunicator.merge_results(results)
        if success and obj.weight_version is not None:
            self._update_weight_version_if_provided(obj.weight_version)
            message += f" Weight version updated to {obj.weight_version}."

        return success, message

    async def init_weights_send_group_for_remote_instance(
        self: TokenizerManager,
        obj: InitWeightsSendGroupForRemoteInstanceReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        """初始化「向远端实例发送权重」的通信组（PD 分离/权重迁移场景）。

        中译：为把本实例权重发送到另一个远端实例建立通信组。目前仅支持 dp_size==1（见 TODO）。
              返回 (是否成功, 消息)。
        """
        self.auto_create_handle_loop()
        # TODO: support DP
        assert (
            self.server_args.dp_size == 1
        ), "dp_size must be 1 for init_weights_send_group_for_remote_instance"
        result = (
            await self.init_weights_send_group_for_remote_instance_communicator(obj)
        )[0]
        return result.success, result.message

    async def send_weights_to_remote_instance(
        self: TokenizerManager,
        obj: SendWeightsToRemoteInstanceReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        """把本实例权重发送到远端实例。

        中译：在 init_weights_send_group_for_remote_instance 建好组后执行实际发送。
              目前仅支持 dp_size==1。返回 (是否成功, 消息)。
        """
        self.auto_create_handle_loop()
        # TODO: support DP
        assert (
            self.server_args.dp_size == 1
        ), "dp_size must be 1 for send_weights_to_remote_instance"
        result = (await self.send_weights_to_remote_instance_communicator(obj))[0]
        return result.success, result.message

    async def update_weights_from_tensor(
        self: TokenizerManager,
        obj: UpdateWeightsFromTensorReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        """直接用内存中的张量热更新权重。

        中译：从调用方传入的张量（已序列化在 obj 中）就地更新模型权重，
              加锁/中止策略与 update_weights_from_distributed 相同。
              成功且带 weight_version 时更新版本号。返回 (是否成功, 消息)。
        """
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for update weights from tensor"

        # 中译：按需先中止所有在途请求。
        if obj.abort_all_requests:
            self.abort_request(abort_all=True)

        # 中译：已暂停时在条件锁下更新；未暂停时取写锁独占更新。
        async with self.is_pause_cond:
            is_paused = self.is_pause
            if is_paused:
                results = await self.update_weights_from_tensor_communicator(obj)

        if not is_paused:
            async with self.model_update_lock.writer_lock:
                results = await self.update_weights_from_tensor_communicator(obj)

        success, message = FanOutCommunicator.merge_results(results)
        if success and obj.weight_version is not None:
            self._update_weight_version_if_provided(obj.weight_version)
            message += f" Weight version updated to {obj.weight_version}."

        return success, message

    async def update_weights_from_ipc(
        self: TokenizerManager,
        obj: UpdateWeightsFromIPCReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        """Update weights via IPC for checkpoint-engine integration.

        中译：通过进程间通信（IPC）热更新权重，用于对接 checkpoint-engine。
              加锁策略同上（已暂停走条件锁，否则走写锁）。整个过程包在 try 中，
              出错则记录日志并返回失败消息。成功且带 weight_version 时更新版本号。
        """
        self.auto_create_handle_loop()
        try:
            # For now, we only support single data parallel instance
            # 中译：目前仅支持单 DP 实例（或启用 dp attention）。
            assert (
                self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
            ), "dp_size must be 1 or dp attention must be enabled for update weights from IPC"
            logger.info("Starting IPC weight update")

            async with self.is_pause_cond:
                is_paused = self.is_pause
                if is_paused:
                    result = (await self.update_weights_from_ipc_communicator(obj))[0]
                    success, message = result.success, result.message

            if not is_paused:
                async with self.model_update_lock.writer_lock:
                    result = (await self.update_weights_from_ipc_communicator(obj))[0]
                    success, message = result.success, result.message
        except Exception as e:
            error_msg = f"IPC weight update failed: {str(e)}"
            logger.error(error_msg)
            success, message = False, error_msg

        if success and obj.weight_version is not None:
            self._update_weight_version_if_provided(obj.weight_version)
            message += f" Weight version updated to {obj.weight_version}."

        return success, message

    async def _unload_lora_adapter_locked(
        self: TokenizerManager,
        obj: UnloadLoRAAdapterReqInput,
    ) -> UnloadLoRAAdapterReqOutput:
        """卸载某个 LoRA 适配器（要求调用方已持有 lora_update_lock）。

        中译：内部加锁版卸载逻辑。
              - 先从 lora_registry 注销，阻止新请求再使用该适配器；
              - 等待所有正在使用该适配器的在途请求结束（wait_for_unload）后，
                才真正命令后端进程卸载，避免卸载到正被使用的权重。
              - 前置断言：必须已持有 lora_update_lock。
        """
        assert (
            self.lora_update_lock.locked()
        ), "self.lora_update_lock must be locked in order for self._unload_lora_adapter_locked() to be called"

        # Unregister the LoRA adapter from the registry to stop new requests for this adapter
        # from being started.
        # 中译：先从注册表注销，阻止该适配器的新请求被启动。
        lora_id = await self.lora_registry.unregister(obj.lora_name)
        obj.lora_id = lora_id

        # Initiate the actual unloading operation at the backend processes only after all
        # ongoing requests using this LoRA adapter are finished.
        # 中译：等待所有仍在使用该适配器的在途请求完成后，再让后端进程真正卸载。
        await self.lora_registry.wait_for_unload(lora_id)
        result = (await self.update_lora_adapter_communicator(obj))[0]

        return result

    async def load_lora_adapter(
        self: TokenizerManager,
        obj: LoadLoRAAdapterReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> LoadLoRAAdapterReqOutput:
        """从路径动态加载一个 LoRA 适配器。

        中译：运行时按 lora_name/lora_path 加载 LoRA 适配器。
              - 前置条件：必须启用 LoRA（--enable-lora）且 dp_size==1。
              - 流程：在 lora_update_lock 下生成唯一 LoRARef → 命令后端加载 →
                加载成功才注册到 registry 并写入 lora_ref_cache。
              - 容量控制：若设置了 max_loaded_loras，超出时按 LRU 淘汰未固定（unpinned）的适配器。
              - 出错（ValueError）时返回带 error_message 的失败输出而不抛出。
        """
        self.auto_create_handle_loop()

        try:
            if not self.server_args.enable_lora:
                raise ValueError(
                    "LoRA is not enabled. Please set `--enable-lora` to enable LoRA."
                )

            # TODO (lifuhuang): Remove this after we verify that dynamic lora loading works
            # with dp_size > 1.
            assert (
                self.server_args.dp_size == 1
            ), "dp_size must be 1 for dynamic lora loading"
            logger.info(
                "Start load Lora adapter. Lora name=%s, path=%s",
                obj.lora_name,
                obj.lora_path,
            )

            # 中译：整个加载在 lora_update_lock 下串行执行，避免并发加载/卸载相互干扰。
            async with self.lora_update_lock:
                # Generate new uniquely identifiable LoRARef object.
                # 中译：生成全局唯一可识别的 LoRARef（含自动分配的 lora_id）。
                new_adapter = LoRARef(
                    lora_name=obj.lora_name,
                    lora_path=obj.lora_path,
                    pinned=obj.pinned,
                )

                # Trigger the actual loading operation at the backend processes.
                # 中译：命令后端进程执行实际加载。
                obj.lora_id = new_adapter.lora_id
                result = (await self.update_lora_adapter_communicator(obj))[0]

                # Register the LoRA adapter only after loading is successful.
                # 中译：仅在后端加载成功后才注册到 registry 并缓存引用。
                if result.success:
                    await self.lora_registry.register(new_adapter)
                    self.lora_ref_cache[obj.lora_name] = new_adapter

                # 中译：若配置了最大同时加载数，循环淘汰最久未用（且未 pin）的适配器至不超限。
                if self.server_args.max_loaded_loras is not None:
                    while (
                        self.lora_registry.num_registered_loras
                        > self.server_args.max_loaded_loras
                    ):
                        lru_lora_name = await self.lora_registry.lru_lora_name(
                            exclude_pinned=True
                        )
                        if lru_lora_name is None:
                            raise ValueError(
                                "Didn't find any LoRA adapters when trying to evict LRU LoRA adapter. "
                                f"LoRA registry is: {self.lora_registry._registry}"
                            )

                        logger.info(
                            f"Unloading least recently used LoRA adapter '{lru_lora_name}' "
                            f"(current number of adapters: {self.lora_registry.num_registered_loras}, "
                            f"max allowed: {self.server_args.max_loaded_loras})"
                        )

                        unload_result = await self._unload_lora_adapter_locked(
                            UnloadLoRAAdapterReqInput(lora_name=lru_lora_name)
                        )
                        if not unload_result.success:
                            raise ValueError(
                                f"Error while unloading LRU LoRA adapter '{lru_lora_name}': "
                                f"{unload_result.error_message}"
                            )
                        del result.loaded_adapters[lru_lora_name]

                return result
        except ValueError as e:
            return LoadLoRAAdapterReqOutput(
                success=False,
                error_message=str(e),
            )

    async def load_lora_adapter_from_tensors(
        self: TokenizerManager,
        obj: LoadLoRAAdapterFromTensorsReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> LoadLoRAAdapterFromTensorsReqOutput:
        """直接从内存张量加载 LoRA 适配器（路径占位为 "__tensor__"）。

        中译：与 load_lora_adapter 流程一致，区别在于权重来自传入的张量而非磁盘路径，
              故 lora_path 用占位符 "__tensor__"。同样有 enable_lora/dp_size==1 前置条件、
              加载成功后注册、以及 max_loaded_loras 的 LRU 淘汰。
        """
        self.auto_create_handle_loop()

        try:
            if not self.server_args.enable_lora:
                raise ValueError(
                    "LoRA is not enabled. Please set `--enable-lora` to enable LoRA."
                )

            assert (
                self.server_args.dp_size == 1
            ), "dp_size must be 1 for dynamic lora loading"
            logger.info(
                "Start load Lora adapter from tensors. Lora name=%s",
                obj.lora_name,
            )

            async with self.lora_update_lock:
                # 中译：路径占位为 "__tensor__"，表示权重来自内存张量而非文件。
                new_adapter = LoRARef(
                    lora_name=obj.lora_name,
                    lora_path="__tensor__",
                    pinned=obj.pinned,
                )
                obj.lora_id = new_adapter.lora_id
                result = (await self.update_lora_adapter_communicator(obj))[0]

                # 中译：加载成功后注册并缓存引用。
                if result.success:
                    await self.lora_registry.register(new_adapter)
                    self.lora_ref_cache[obj.lora_name] = new_adapter
                # 中译：超出最大加载数时，按 LRU 淘汰未 pin 的适配器至不超限。
                if self.server_args.max_loaded_loras is not None:
                    while (
                        self.lora_registry.num_registered_loras
                        > self.server_args.max_loaded_loras
                    ):
                        lru_lora_name = await self.lora_registry.lru_lora_name(
                            exclude_pinned=True
                        )
                        if lru_lora_name is None:
                            raise ValueError(
                                "Didn't find any LoRA adapters when trying to evict LRU LoRA adapter. "
                                f"LoRA registry is: {self.lora_registry._registry}"
                            )

                        logger.info(
                            f"Unloading least recently used LoRA adapter '{lru_lora_name}' "
                            f"(current number of adapters: {self.lora_registry.num_registered_loras}, "
                            f"max allowed: {self.server_args.max_loaded_loras})"
                        )

                        unload_result = await self._unload_lora_adapter_locked(
                            UnloadLoRAAdapterReqInput(lora_name=lru_lora_name)
                        )
                        if not unload_result.success:
                            raise ValueError(
                                f"Error while unloading LRU LoRA adapter '{lru_lora_name}': "
                                f"{unload_result.error_message}"
                            )
                        del result.loaded_adapters[lru_lora_name]

                return result
        except ValueError as e:
            return LoadLoRAAdapterFromTensorsReqOutput(
                success=False,
                error_message=str(e),
            )

    async def unload_lora_adapter(
        self: TokenizerManager,
        obj: UnloadLoRAAdapterReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> UnloadLoRAAdapterReqOutput:
        """卸载指定名称的 LoRA 适配器（对外入口）。

        中译：校验 enable_lora、lora_name 非空、dp_size==1 后，在 lora_update_lock 下
              委托 _unload_lora_adapter_locked 完成实际卸载。出错返回失败输出。
        """
        self.auto_create_handle_loop()

        try:
            if not self.server_args.enable_lora:
                raise ValueError(
                    "LoRA is not enabled. Please set `--enable-lora` to enable LoRA."
                )

            assert (
                obj.lora_name is not None
            ), "lora_name must be provided to unload LoRA adapter"

            # TODO (lifuhuang): Remove this after we verify that dynamic lora loading works
            # with dp_size > 1.
            assert (
                self.server_args.dp_size == 1
            ), "dp_size must be 1 for dynamic lora loading"
            logger.info(
                "Start unload Lora adapter. Lora name=%s",
                obj.lora_name,
            )

            async with self.lora_update_lock:
                return await self._unload_lora_adapter_locked(obj)
        except ValueError as e:
            return UnloadLoRAAdapterReqOutput(success=False, error_message=str(e))

    async def get_weights_by_name(
        self: TokenizerManager,
        obj: GetWeightsByNameReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        """按名称取回某个权重参数（用于调试/校验）。

        中译：扇出查询各调度器，收集每个 rank 的参数。dp_size==1 时返回单个参数，
              否则返回各 rank 的参数列表。
        """
        self.auto_create_handle_loop()
        results = await self.get_weights_by_name_communicator(obj)
        all_parameters = [r.parameter for r in results]
        if self.server_args.dp_size == 1:
            return all_parameters[0]
        else:
            return all_parameters

    async def release_memory_occupation(
        self: TokenizerManager,
        obj: ReleaseMemoryOccupationReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        """释放显存占用（如把权重/KV 缓存卸到 CPU，腾出显存）。

        中译：扇出释放请求给各调度器，常用于训推一体场景下临时让出显存给训练侧。
        """
        self.auto_create_handle_loop()
        await self.release_memory_occupation_communicator(obj)

    async def resume_memory_occupation(
        self: TokenizerManager,
        obj: ResumeMemoryOccupationReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        """恢复之前释放的显存占用（与 release_memory_occupation 相对）。

        中译：扇出恢复请求，把权重/KV 缓存重新加载回显存。
        """
        self.auto_create_handle_loop()
        await self.resume_memory_occupation_communicator(obj)

    async def check_weights(
        self: TokenizerManager,
        obj: CheckWeightsReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str, Optional[List[Dict]], Optional[str]]:
        """校验各 GPU 上的权重一致性（计算并汇总校验和）。

        中译：扇出校验请求，汇总各 rank 返回的 per-GPU 校验和；再对所有 rank 的
              校验和做一次 sha256 得到「整引擎校验和」（per_engine_checksum）。
              返回 (是否成功, 消息, 各 rank 明细列表, 整引擎校验和)。
        """
        self.auto_create_handle_loop()
        results = await self.check_weights_communicator(obj)
        success, message = FanOutCommunicator.merge_results(results)
        ranks: Optional[List[Dict]] = None
        per_engine_checksum: Optional[str] = None
        # 中译：仅当有 rank 返回了 payload（校验明细）时才汇总。
        if any(r.payload is not None for r in results):
            ranks = []
            for r in results:
                # 中译：payload 可能是列表（多 GPU）或单条，统一展开收集。
                if isinstance(r.payload, list):
                    ranks.extend(r.payload)
                else:
                    ranks.append(r.payload)
            # 中译：把各 GPU 校验和按顺序喂入 sha256，得到整引擎级别的总校验和。
            h = hashlib.sha256()
            for rank in ranks:
                h.update(rank["per_gpu_checksum"].encode())
            per_engine_checksum = h.hexdigest()
        return success, message, ranks, per_engine_checksum

    async def slow_down(
        self: TokenizerManager,
        obj: SlowDownReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        """人为放缓调度器处理速度（多用于测试/限流）。

        中译：扇出 slow_down 请求，让调度器在每步之间引入延迟。
        """
        self.auto_create_handle_loop()
        await self.slow_down_communicator(obj)

    async def get_internal_state(self: TokenizerManager) -> List[Dict[Any, Any]]:
        """获取各 DP rank 调度器的内部状态快照。

        中译：扇出查询，返回每个 DP rank 的 internal_state 字典组成的列表。
        """
        self.auto_create_handle_loop()
        req = GetInternalStateReq()
        responses: List[GetInternalStateReqOutput] = (
            await self.get_internal_state_communicator(req)
        )
        # Many DP ranks
        # 中译：可能有多个 DP rank，逐个取其 internal_state。
        return [res.internal_state for res in responses]

    async def set_internal_state(
        self: TokenizerManager, obj: SetInternalStateReq
    ) -> List[bool]:
        """设置各 DP rank 调度器的内部状态。

        中译：扇出设置请求，返回每个 rank 是否更新成功（updated）的布尔列表。
        """
        self.auto_create_handle_loop()
        responses: List[SetInternalStateReqOutput] = (
            await self.set_internal_state_communicator(obj)
        )
        return [res.updated for res in responses]

    async def dumper_control(
        self: TokenizerManager, obj: DumperControlReqInput
    ) -> List[DumperControlReqOutput]:
        """控制调试用的 dumper（中间张量转储）开关/配置。

        中译：扇出控制请求，返回各 rank 的 DumperControlReqOutput 列表。
        """
        self.auto_create_handle_loop()
        return await self.dumper_control_communicator(obj)

    async def get_loads(
        self: TokenizerManager,
        include: Optional[List[str]] = None,
        dp_rank: Optional[int] = None,
    ) -> List[LoadSnapshot]:
        """
        Get load snapshots for /v1/loads endpoint.

        Args:
            include: List of sections to include. Options: core, memory, spec, lora, disagg, queues, all
            dp_rank: Optional filter for specific DP rank

        Returns:
            List of LoadSnapshot, one per scheduler (filtered by dp_rank if specified)

        中译：为 /v1/loads 接口读取负载快照（每个调度器一个）。
              不同于其他方法，本方法不走 FanOutCommunicator，而是直接读共享内存里的
              load_snapshot_reader（无需往返调度器，开销低）。
              - include：要包含的统计分区（core/memory/spec/lora/disagg/queues/all）。
              - dp_rank：可选，只取指定 DP rank（越界则返回空列表）。
        """
        self.auto_create_handle_loop()
        # 中译：dp_rank 越界时直接返回空列表。
        if dp_rank is not None and (dp_rank < 0 or dp_rank >= self.server_args.dp_size):
            return []

        reader = self.load_snapshot_reader
        # 中译：指定 rank 则只读该 rank（可能为 None），否则读全部。
        if dp_rank is not None:
            load = reader.read(dp_rank)
            results = [load] if load is not None else []
        else:
            results = reader.read_all()

        return results

    async def open_session(
        self: TokenizerManager,
        obj: OpenSessionReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        """打开一个会话（session），用于跨多次请求保持状态（如多轮对话）。

        中译：与上面的扇出型控制操作不同，会话不经 FanOutCommunicator，而是直接发给调度器，
              并用 asyncio.Future 等待调度器回执 session_id。
              - streaming 会话需启用 --enable-streaming-session，否则报错。
              - session_id 未提供则随机生成；若已存在则返回 None（避免重复打开）。
              - 通过 session_futures[session_id] 注册 future 等待结果，finally 中清理该 future。
        """
        self.auto_create_handle_loop()
        # 中译：流式会话必须显式启用，否则拒绝。
        if obj.streaming:
            if not self.server_args.enable_streaming_session:
                raise ValueError(
                    "Streaming sessions are disabled. "
                    "Please relaunch with --enable-streaming-session."
                )

        # 中译：未指定 session_id 则随机生成；若该 id 已在等待中则视为重复，返回 None。
        if obj.session_id is None:
            obj.session_id = uuid.uuid4().hex
        elif obj.session_id in self.session_futures:
            return None

        # 中译：注册 future，发请求给调度器，await 等待其回执；无论成败都清理 future。
        future = asyncio.Future()
        self.session_futures[obj.session_id] = future
        self.send_to_scheduler.send_pyobj(obj)

        try:
            return await future
        finally:
            self.session_futures.pop(obj.session_id, None)

    async def close_session(
        self: TokenizerManager,
        obj: CloseSessionReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        """关闭会话。中译：直接把关闭请求发给调度器，无需等待回执。"""
        await self.send_to_scheduler.send_pyobj(obj)

    def _update_weight_version_if_provided(
        self: TokenizerManager, weight_version: Optional[str]
    ) -> None:
        """Update weight version if provided.

        中译：若提供了权重版本号，则更新到 server_args.weight_version。
              供各权重更新方法在更新成功后调用，使版本号与实际权重保持同步。
        """
        if weight_version is not None:
            self.server_args.weight_version = weight_version
