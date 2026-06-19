# 中译：本模块是 Scheduler 侧的「权重热更新（online weight update）」管理器。
#       它把 Scheduler 收到的各类「更新权重 / 内存占用」请求，转交给底层 worker 执行，
#       并在更新后按需 flush KV cache、做分布式 barrier 同步、上报耗时指标。
#       支持四种权重来源：disk（磁盘）、distributed（分布式广播）、tensor（直接传张量）、
#       ipc（进程间共享，用于 checkpoint-engine 集成）。
#       此外还负责显存的释放/恢复（release/resume memory occupation），用于「让出显存」
#       场景（如训练-推理共卡），按 weights / kv_cache / cuda_graph 三类标签分别 pause/resume。
from __future__ import annotations

import hashlib
import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import torch

from sglang.srt.constants import (
    GPU_MEMORY_ALL_TYPES,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)

logger = logging.getLogger(__name__)


def _get_draft_model_runner(draft_worker):
    # 中译：从「投机解码的 draft worker」中取出其 ModelRunner。
    #       不同投机实现把 runner 挂在不同属性上，故按已知约定逐个尝试，取不到返回 None。
    # DFlash / FrozenKVMTP workers expose draft_model_runner directly
    # 中译：DFlash / FrozenKVMTP 这类 worker 直接暴露 draft_model_runner 属性。
    runner = getattr(draft_worker, "draft_model_runner", None)
    if runner is not None:
        return runner
    # EAGLEWorkerV2: _draft_worker.draft_runner
    # 中译：EAGLEWorkerV2 则是嵌套在 _draft_worker.draft_runner 上。
    inner = getattr(draft_worker, "_draft_worker", None)
    if inner is not None:
        runner = getattr(inner, "draft_runner", None)
        if runner is not None:
            return runner
    return None


def _merge_checksum_payloads(target: Dict, draft: Dict) -> Dict:
    # 中译：合并 target（主模型）与 draft（投机草稿模型）的权重校验和（checksum）载荷。
    #       draft 的每个权重名加 "draft." 前缀以避免与主模型重名，
    #       再按名字排序后整体哈希（sha256）得到本 GPU 的总校验和 per_gpu_checksum。
    merged_checksums = dict(target["checksums"])
    for name, chk in draft["checksums"].items():
        merged_checksums[f"draft.{name}"] = chk
    h = hashlib.sha256()
    # 中译：排序后再喂入哈希，确保结果与遍历顺序无关、可复现。
    for name in sorted(merged_checksums):
        h.update(name.encode())
        h.update(merged_checksums[name].encode())
    target["checksums"] = merged_checksums
    target["per_gpu_checksum"] = h.hexdigest()
    return target


@dataclass(kw_only=True, slots=True)
class SchedulerWeightUpdaterManager:
    # 中译：Scheduler 侧权重热更新管理器。本身不持有模型，而是聚合若干「协作对象」：
    #   tp_worker —— 张量并行 worker（持有 model_runner，真正执行权重更新）。
    #   draft_worker —— 投机解码草稿模型 worker（可能为 None）。
    #   tp_cpu_group —— TP 组的 CPU 通信组，用于 barrier / all_gather 同步。
    #   memory_saver_adapter —— 显存「让出/恢复」适配器，按标签 pause/resume。
    #   flush_cache / is_fully_idle —— 由 Scheduler 注入的回调（清缓存 / 判断是否完全空闲）。
    #   scheduler —— 反向引用 Scheduler，用于在 PD 分离模式下操作各传输队列。
    #   offload_tags —— 当前已被「让出（offload）」的显存类型集合。
    #   stashed_model_static_state —— 让出 weights 时暂存的模型静态缓冲（buffers），恢复时回填。
    tp_worker: Any
    draft_worker: Any
    tp_cpu_group: Any
    memory_saver_adapter: Any
    flush_cache: Callable[..., bool]
    is_fully_idle: Callable[..., bool]
    scheduler: Optional[Any] = None
    metrics_collector: Optional[Any] = None
    offload_tags: set = field(default_factory=set)
    stashed_model_static_state: Any = None

    @contextmanager
    def _observe_weight_load(self, source: str) -> Iterator[None]:
        # Edge-trigger weight_load_duration_seconds at the end of each
        # update_weights_from_* call. Engine is paused during the update so
        # the periodic log_stats path can't carry this.
        # `source` distinguishes disk vs distributed vs tensor vs ipc.
        # 中译：上下文管理器，用于「边沿触发」上报权重加载耗时指标。
        #       更新权重期间引擎是暂停的，周期性的 log_stats 无法覆盖这段，故在此显式计时上报。
        #       source 区分来源（disk / distributed / tensor / ipc）。
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.metrics_collector is not None:
                self.metrics_collector.observe_weight_load(
                    time.perf_counter() - t0, source
                )

    def flush_cache_after_weight_update(self, recv_req) -> None:
        # 中译：权重更新后按需清空（radix/KV）缓存——因为旧权重算出的缓存已失效。
        #       torch_empty_cache 决定是否同时归还 PyTorch 显存分配器的缓存块。
        if recv_req.flush_cache:
            flush_cache_success = self.flush_cache(
                empty_cache=recv_req.torch_empty_cache
            )
            assert flush_cache_success, "Cache flush failed after updating weights"

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """In-place update of the weights from disk.

        中译：从磁盘原地（in-place）更新权重。先更新主模型，成功再更新草稿模型；
              只要主模型成功就 flush 缓存（tp_success 记录主模型结果，避免被草稿结果覆盖）。
        """
        with self._observe_weight_load("disk"):
            success, message = self.tp_worker.update_weights_from_disk(recv_req)
            # 中译：单独记下主模型是否成功，后续 flush 判断以它为准。
            tp_success = success
            if success and self.draft_worker is not None:
                success, message = self.draft_worker.update_weights_from_disk(recv_req)
            if tp_success:
                self.flush_cache_after_weight_update(recv_req)
            if not success:
                logger.error(message)
            return UpdateWeightFromDiskReqOutput(success, message, 0)

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        """Initialize the online model parameter update group.

        中译：初始化「在线权重更新通信组」（用于 distributed 方式广播权重前的握手建组）。
        """
        success, message = self.tp_worker.init_weights_update_group(recv_req)
        return InitWeightsUpdateGroupReqOutput(success, message)

    def destroy_weights_update_group(
        self,
        recv_req: DestroyWeightsUpdateGroupReqInput,
    ):
        """Destroy the online model parameter update group.

        中译：销毁上面建立的在线权重更新通信组，释放相关资源。
        """
        success, message = self.tp_worker.destroy_weights_update_group(recv_req)
        return DestroyWeightsUpdateGroupReqOutput(success, message)

    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter.

        中译：通过分布式通信组接收并更新权重（训练端广播、推理端接收）。
        """
        with self._observe_weight_load("distributed"):
            success, message = self.tp_worker.update_weights_from_distributed(recv_req)
            if success:
                self.flush_cache_after_weight_update(recv_req)
            else:
                logger.error(message)
            return UpdateWeightsFromDistributedReqOutput(success, message)

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        """Update the online model parameter from tensors.

        中译：直接用传入的张量更新权重。
        """
        with self._observe_weight_load("tensor"):
            # 中译：选择更新目标——明确禁用草稿模型时只更新主模型；否则优先更新草稿模型（无则回落到主模型）。
            if recv_req.disable_draft_model:
                worker = self.tp_worker
            else:
                worker = self.draft_worker or self.tp_worker
            success, message = worker.update_weights_from_tensor(recv_req)
            if success:
                self.flush_cache_after_weight_update(recv_req)
            else:
                logger.error(message)
            # 中译：所有 TP rank 在此对齐，确保更新完成后再继续，避免 rank 间权重不一致。
            torch.distributed.barrier(group=self.tp_cpu_group)
            return UpdateWeightsFromTensorReqOutput(success, message)

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update the online model parameter from IPC for checkpoint-engine integration.

        中译：通过 IPC（进程间共享内存）更新权重，用于 checkpoint-engine 集成场景。
              同 disk 路径：主模型成功即 flush；末尾做 TP barrier 同步。
        """
        with self._observe_weight_load("ipc"):
            success, message = self.tp_worker.update_weights_from_ipc(recv_req)
            tp_success = success
            if success and self.draft_worker is not None:
                success, message = self.draft_worker.update_weights_from_ipc(recv_req)
            if tp_success:
                self.flush_cache_after_weight_update(recv_req)
            if not success:
                logger.error(message)
            torch.distributed.barrier(group=self.tp_cpu_group)
            return UpdateWeightsFromIPCReqOutput(success, message)

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        # 中译：按名字读取某个权重张量（调试/校验用）。
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return GetWeightsByNameReqOutput(parameter)

    def release_memory_occupation(self, recv_req: ReleaseMemoryOccupationReqInput):
        # 中译：释放（让出）GPU 显存占用，把指定标签的显存 pause 掉。仅允许在引擎完全空闲时调用，
        #       否则正在进行的请求会因显存被收走而出错。
        assert (
            self.is_fully_idle()
        ), "release_memory_occupation should be called only when server is idle."

        tags = recv_req.tags

        # 中译：未指定标签时默认释放所有类型（weights / kv_cache / cuda_graph）。
        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        # 中译：记录这些标签已被让出，供其他逻辑查询当前 offload 状态。
        for tag in tags:
            self.offload_tags.add(tag)

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            # 中译：释放 KV cache 前，PD 分离模式下需先让各传输/预分配队列释放它们持有的显存。
            scheduler = self.scheduler
            if scheduler is not None:
                if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
                    # 中译：Decode 节点：释放「传输队列」和「预分配队列」。
                    for queue_name in (
                        "disagg_decode_transfer_queue",
                        "disagg_decode_prealloc_queue",
                    ):
                        queue = getattr(scheduler, queue_name, None)
                        if queue is not None:
                            queue.release_memory_occupation()
                elif scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
                    # 中译：Prefill 节点：释放「bootstrap 队列」。
                    queue = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
                    if queue is not None:
                        queue.release_memory_occupation()
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
            self.flush_cache()

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            # 中译：让出权重显存前，先把模型的静态缓冲（buffers，如旋转位置编码缓存等）导出暂存，
            #       因为这些 buffer 不会被重新加载，resume 时需原样回填。
            self.stashed_model_static_state = _export_static_state(
                self.tp_worker.model_runner.model
            )
            torch.distributed.barrier(self.tp_cpu_group)
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_WEIGHTS)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_CUDA_GRAPH)

        # 中译：同步设备，确保所有 pause/释放操作真正完成后再返回。
        torch.get_device_module().synchronize()

        return ReleaseMemoryOccupationReqOutput()

    def resume_memory_occupation(self, recv_req: ResumeMemoryOccupationReqInput):
        # 中译：恢复之前让出的显存占用，是 release_memory_occupation 的逆操作。
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        # 中译：从「已让出」集合中移除这些标签（恢复后不再处于 offload 状态）。
        for tag in tags:
            self.offload_tags.remove(tag)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_WEIGHTS)
            torch.distributed.barrier(self.tp_cpu_group)
            # 中译：把之前暂存的静态缓冲回填到模型，并删除暂存引用以释放内存。
            _import_static_state(
                self.tp_worker.model_runner.model,
                self.stashed_model_static_state,
            )
            del self.stashed_model_static_state

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)
            # 中译：恢复 KV cache 后，PD 分离模式下也要让相应队列恢复其显存占用（与 release 对称）。
            scheduler = self.scheduler
            if scheduler is not None:
                if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
                    for queue_name in (
                        "disagg_decode_transfer_queue",
                        "disagg_decode_prealloc_queue",
                    ):
                        queue = getattr(scheduler, queue_name, None)
                        if queue is not None:
                            queue.resume_memory_occupation()
                elif scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
                    queue = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
                    if queue is not None:
                        queue.resume_memory_occupation()

        return ResumeMemoryOccupationReqOutput()

    def check_weights(self, recv_req: CheckWeightsReqInput):
        # 中译：校验权重——对每个权重计算校验和并按 TP rank 汇总，用于验证多机/多卡间权重一致性。
        try:
            payload = self.tp_worker.model_runner.check_weights(action=recv_req.action)

            # 中译：若有草稿模型，也校验其权重，并把两者的校验和合并（draft 名加前缀）。
            if self.draft_worker is not None:
                draft_runner = _get_draft_model_runner(self.draft_worker)
                if draft_runner is not None:
                    draft_payload = draft_runner.check_weights(action=recv_req.action)
                    if payload is not None and draft_payload is not None:
                        payload = _merge_checksum_payloads(payload, draft_payload)

            # 中译：TP > 1 时把各 rank 的校验和 all_gather 汇总成一个列表，便于上层比对所有 rank 是否一致。
            tp_size = torch.distributed.get_world_size(group=self.tp_cpu_group)
            if tp_size > 1 and payload is not None:
                all_payloads = [None] * tp_size
                torch.distributed.all_gather_object(
                    all_payloads, payload, group=self.tp_cpu_group
                )
                payload = all_payloads
            return CheckWeightsReqOutput(
                success=True, message="Success.", payload=payload
            )
        except Exception as e:
            # 中译：校验过程出错不应让进程崩溃，记录警告并返回失败结果即可。
            logger.warning(f"check_weights see error: {e}")
            traceback.print_exc()
            return CheckWeightsReqOutput(success=False, message=f"{e}")

    def save_remote_model(self, params):
        # 中译：把当前权重保存到远端（URL）。有草稿模型时必须同时提供 draft_url。
        url = params["url"]

        self.tp_worker.model_runner.save_remote_model(url)

        if self.draft_worker is not None:
            draft_url = params.get("draft_url", None)
            assert (
                draft_url is not None
            ), "draft_url must be provided when draft model is enabled"
            self.draft_worker.model_runner.save_remote_model(draft_url)

    def save_sharded_model(self, params):
        # 中译：把权重按分片（sharded）格式保存到本地路径，便于后续分片加载。
        self.tp_worker.model_runner.save_sharded_model(
            path=params["path"],
            pattern=params["pattern"],
            max_size=params["max_size"],
        )


def _export_static_state(model):
    # 中译：导出模型的「静态缓冲（buffers）」快照——detach + clone 出一份副本暂存。
    #       buffers 是非参数但需持久的张量（如位置编码缓存），让出显存时需保留、恢复时回填。
    return dict(
        buffers=[
            (name, buffer.detach().clone()) for name, buffer in model.named_buffers()
        ]
    )


def _import_static_state(model, static_params):
    # 中译：把之前导出的静态缓冲回填到模型对应 buffer 中。
    #       inference_mode 下用 [...] 原地拷贝，不触发梯度记录、也不替换张量对象本身。
    with torch.inference_mode():
        self_named_buffers = dict(model.named_buffers())
        for name, tensor in static_params["buffers"]:
            self_named_buffers[name][...] = tensor
