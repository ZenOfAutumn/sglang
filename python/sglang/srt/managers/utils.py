# 中译：调度层（managers）通用工具模块。集中放置若干供 Scheduler / TpWorker 共用的小工具：
#       - GenerationBatchResult / EmbeddingBatchResult：前向结果的容器（含 overlap 调度下的 CPU 拷贝语义）；
#       - validate_input_length：输入长度校验与（可选）自动截断；
#       - get_logprob_dict_from_result / get_logprob_from_pp_outputs：从前向结果中提取 logprob；
#       - is_health_check_generate_req：识别健康检查请求。

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Union

import torch

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.eplb.expert_distribution import ExpertDistributionMetrics
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.state_capturer.base import TopkCaptureOutput

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.speculative.eagle_info import EagleDraftInput


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GenerationBatchResult:
    """中译：一次生成前向（forward）的结果容器。TpWorker 跑完 forward+sample 后填充本结构，
    交给调度器的输出处理流程。字段大体分几类：核心输出（logits/next_token_ids）、
    投机解码统计、PP/overlap 调度所需的张量与同步事件、logprob 起止信息、各类观测指标。

    重叠调度（overlap scheduling）语义要点：GPU 前向与 CPU 输出处理在不同 stream 上并行，
    因此结果张量先以 non_blocking 方式异步拷回 CPU（见 copy_to_cpu），并用 copy_done 这个
    CUDA event 标记拷贝完成；CPU 侧消费前需等待该 event，避免读到尚未拷完的数据。"""

    logits_output: Optional[LogitsProcessorOutput] = None
    pp_hidden_states_proxy_tensors: Optional[PPProxyTensors] = None
    next_token_ids: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None
    num_correct_drafts: int = 0  # no bonus included  # 中译：投机解码被接受的 draft token 数（不含 bonus token）
    num_correct_drafts_per_req_cpu: Optional[List[int]] = None  # 中译：每请求的接受数（已在 CPU 上）
    can_run_cuda_graph: bool = False  # 中译：本批次是否走了 cuda graph 回放路径

    # PP skip output comm: True when output send/recv was skipped and
    # next_token_ids are placeholder zeros. Used by process_batch_result_prefill
    # to validate that skipped output is never consumed.
    # 中译：流水并行下「跳过输出通信」标记——为 True 时 next_token_ids 是占位的全 0，
    #       供 process_batch_result_prefill 断言这种被跳过的输出绝不会被真正消费。
    skipped_output_comm: bool = False

    # For output processing
    # 中译：输出处理用——每请求的 extend（prefill 续填）长度，及 logprob 计算起点。
    extend_input_len_per_req: Optional[List[int]] = None
    extend_logprob_start_len_per_req: Optional[List[int]] = None

    # For overlap scheduling
    # 中译：重叠调度用——copy_done：CPU 异步拷贝完成的 CUDA event；
    #       delay_sample_func：延后执行的采样闭包；future_indices：结果在 future 缓冲中的索引；
    #       speculative_num_draft_tokens：本步投机 draft token 数。
    copy_done: Optional[torch.cuda.Event] = None
    delay_sample_func: Optional[callable] = None
    future_indices: Optional[torch.Tensor] = None
    speculative_num_draft_tokens: Optional[int] = None

    # FIXME(lsyin): maybe move to a better place?
    # sync path: forward stream -> output processor
    # 中译：同步路径——投机解码每请求实际接受的长度，从前向 stream 传给输出处理器。
    accept_lens: Optional[torch.Tensor] = None

    # Next-iter seq_lens; published via on_publish.
    # 中译：下一轮迭代的序列长度，通过 on_publish 发布。
    new_seq_lens: Optional[torch.Tensor] = None

    # relay path: forward stream -> next step forward
    # 中译：接力路径——本步的 EAGLE draft 输入，直接传给下一步前向（无需经过 CPU）。
    next_draft_input: Optional[EagleDraftInput] = None

    # Refs the worker wants scheduler to keep alive for the same 2-iter window
    # as batch_record_buf. Used for cross-stream tensor lifetime (e.g. a spec
    # V2 verify ForwardBatch whose tensors must outlive mid-iter SB rebinds).
    # 中译：worker 希望调度器额外「保活」的引用，存活窗口与 batch_record_buf 同为 2 个迭代。
    #       用于跨 stream 的张量生命周期管理（例如 spec V2 的 verify ForwardBatch，其张量必须活过
    #       迭代中途的 ScheduleBatch 重绑定，否则会被提前释放）。
    extra_keep_alive_refs: Optional[List[Any]] = None

    # Routed experts: pending async D2H for overlap scheduling
    # 中译：MoE 路由专家信息——重叠调度下待完成的异步 D2H（device→host）拷贝。
    routed_experts_output: Optional[TopkCaptureOutput] = None
    indexer_topk_output: Optional[TopkCaptureOutput] = None

    # metrics
    # 中译：观测指标——MoE 专家负载分布。
    expert_distribution_metrics: Optional[ExpertDistributionMetrics] = None

    # Forward pass metrics (FPM) — GPU-accurate timing via CUDA events
    # 中译：前向耗时指标（FPM）——用 CUDA event 在 GPU 上精确计时（起止两个事件之差即前向用时）。
    fpm_start_event: Optional[torch.cuda.Event] = None
    fpm_end_event: Optional[torch.cuda.Event] = None

    def copy_to_cpu(self, return_logprob: bool, return_hidden_states: bool = True):
        """Copy tensors to CPU in overlap scheduling.
        Only the tensors which are needed for processing results are copied,
        e.g., next_token_ids, logits outputs

        中译：重叠调度下把结果张量异步拷回 CPU。只拷「输出处理真正需要」的张量
              （如 next_token_ids、各类 logprob、hidden_states），以减少 D2H 开销。
              全部用 non_blocking=True 异步发起，最后用 self.copy_done.record() 在前向 stream 上
              打一个完成标记；CPU 侧消费前须等待该 event，确保数据已拷完。
        """
        if return_logprob:
            if self.logits_output.next_token_logprobs is not None:
                self.logits_output.next_token_logprobs = (
                    self.logits_output.next_token_logprobs.to("cpu", non_blocking=True)
                )
            if self.logits_output.input_token_logprobs is not None:
                self.logits_output.input_token_logprobs = (
                    self.logits_output.input_token_logprobs.to("cpu", non_blocking=True)
                )
            if self.logits_output.next_token_top_logprobs_val is not None:
                self.logits_output.next_token_top_logprobs_val = [
                    v.to("cpu", non_blocking=True) if torch.is_tensor(v) else v
                    for v in self.logits_output.next_token_top_logprobs_val
                ]
            if self.logits_output.next_token_top_logprobs_idx is not None:
                self.logits_output.next_token_top_logprobs_idx = [
                    x.to("cpu", non_blocking=True) if torch.is_tensor(x) else x
                    for x in self.logits_output.next_token_top_logprobs_idx
                ]
            if self.logits_output.next_token_token_ids_logprobs_val is not None:
                self.logits_output.next_token_token_ids_logprobs_val = [
                    v.to("cpu", non_blocking=True) if torch.is_tensor(v) else v
                    for v in self.logits_output.next_token_token_ids_logprobs_val
                ]
        if return_hidden_states and self.logits_output.hidden_states is not None:
            self.logits_output.hidden_states = self.logits_output.hidden_states.to(
                "cpu", non_blocking=True
            )
        # 中译：next_token_ids 是每轮都要回 CPU 的核心输出，无条件异步拷贝。
        self.next_token_ids = self.next_token_ids.to("cpu", non_blocking=True)

        if self.accept_lens is not None:
            self.accept_lens = self.accept_lens.to("cpu", non_blocking=True)

        if self.routed_experts_output is not None:
            self.routed_experts_output.copy_to_cpu()

        if self.indexer_topk_output is not None:
            self.indexer_topk_output.copy_to_cpu()

        if (x := self.expert_distribution_metrics) is not None:
            x.copy_to_cpu()

        # 中译：在当前（前向）stream 上记录拷贝完成事件；消费方等待它即可安全读取 CPU 数据。
        self.copy_done.record()

    @classmethod
    def from_pp_proxy(
        cls, logits_output, next_pp_outputs: PPProxyTensors, can_run_cuda_graph
    ):
        # 中译：从流水并行最后一段传回的代理张量（PPProxyTensors）构造结果对象。
        # TODO(lsyin): refactor PP and avoid using dict
        # 中译：当前 PP 输出用 dict 承载，此处按 key 取出各字段；待重构后可去掉 dict。
        proxy_dict = next_pp_outputs.tensors
        return cls(
            logits_output=logits_output,
            pp_hidden_states_proxy_tensors=None,
            next_token_ids=next_pp_outputs["next_token_ids"],
            extend_input_len_per_req=proxy_dict.get("extend_input_len_per_req", None),
            extend_logprob_start_len_per_req=proxy_dict.get(
                "extend_logprob_start_len_per_req", None
            ),
            can_run_cuda_graph=can_run_cuda_graph,
        )


def validate_input_length(
    req: Req, max_req_input_len: int, allow_auto_truncate: bool
) -> Optional[str]:
    """Validate and potentially truncate input length.

    Args:
        req: The request containing input_ids to validate
        max_req_input_len: Maximum allowed input length
        allow_auto_truncate: Whether to truncate long inputs

    Returns:
        Error message if validation fails, None if successful

    中译：校验请求输入长度，必要时截断。输入超过上限时：若允许自动截断则就地截短 origin_input_ids
          并返回 None（成功）；否则返回错误信息字符串，由调用方据此拒绝该请求。
    """
    if len(req.origin_input_ids) >= max_req_input_len:
        if allow_auto_truncate:
            # 中译：超长但允许截断——告警并把输入截到 max_req_input_len（注意是就地修改 req）。
            logger.warning(
                "Request length is longer than the KV cache pool size or "
                "the max context length. Truncated. "
                f"{len(req.origin_input_ids)=}, {max_req_input_len=}."
            )
            req.origin_input_ids = req.origin_input_ids[:max_req_input_len]
            return None
        else:
            # 中译：超长且不允许截断——返回错误信息，提示改用更短输入或开启 --allow-auto-truncate。
            error_msg = (
                f"Input length ({len(req.origin_input_ids)} tokens) exceeds "
                f"the maximum allowed length ({max_req_input_len} tokens). "
                f"Use a shorter input or enable --allow-auto-truncate."
            )
            return error_msg

    return None


def get_logprob_dict_from_result(result: GenerationBatchResult) -> dict:
    # 中译：从前向结果中抽取与 logprob 相关的全部字段，打平成一个 dict（含输入/输出 token 的
    #       logprob 值与下标、top-k logprob、指定 token id 的 logprob 等），便于后续统一处理/序列化。

    logits_output = result.logits_output
    assert logits_output is not None

    return {
        "extend_input_len_per_req": result.extend_input_len_per_req,
        "extend_logprob_start_len_per_req": result.extend_logprob_start_len_per_req,
        "next_token_logprobs": result.logits_output.next_token_logprobs,
        "next_token_top_logprobs_val": result.logits_output.next_token_top_logprobs_val,
        "next_token_top_logprobs_idx": result.logits_output.next_token_top_logprobs_idx,
        "next_token_token_ids_logprobs_val": result.logits_output.next_token_token_ids_logprobs_val,
        "next_token_token_ids_logprobs_idx": result.logits_output.next_token_token_ids_logprobs_idx,
        "input_token_logprobs": result.logits_output.input_token_logprobs,
        "input_top_logprobs_val": result.logits_output.input_top_logprobs_val,
        "input_top_logprobs_idx": result.logits_output.input_top_logprobs_idx,
        "input_token_ids_logprobs_val": result.logits_output.input_token_ids_logprobs_val,
        "input_token_ids_logprobs_idx": result.logits_output.input_token_ids_logprobs_idx,
    }


def get_logprob_from_pp_outputs(
    next_pp_outputs: PPProxyTensors,
) -> tuple[LogitsProcessorOutput, list[int], list[int]]:
    # 中译：流水并行下，从最后一段 PP 传回的代理张量中重建 LogitsProcessorOutput 及 logprob 相关信息。
    logits_output = LogitsProcessorOutput(
        # Do not send logits and hidden states because they are large
        # 中译：不回传原始 logits 与 hidden_states——它们体积大、跨 PP 传输代价高，故置 None。
        next_token_logits=None,
        hidden_states=None,
        next_token_logprobs=next_pp_outputs["next_token_logprobs"],
        next_token_top_logprobs_val=next_pp_outputs["next_token_top_logprobs_val"],
        next_token_top_logprobs_idx=next_pp_outputs["next_token_top_logprobs_idx"],
        next_token_token_ids_logprobs_val=next_pp_outputs[
            "next_token_token_ids_logprobs_val"
        ],
        next_token_token_ids_logprobs_idx=next_pp_outputs[
            "next_token_token_ids_logprobs_idx"
        ],
        input_token_logprobs=next_pp_outputs["input_token_logprobs"],
        input_top_logprobs_val=next_pp_outputs["input_top_logprobs_val"],
        input_top_logprobs_idx=next_pp_outputs["input_top_logprobs_idx"],
        input_token_ids_logprobs_val=next_pp_outputs["input_token_ids_logprobs_val"],
        input_token_ids_logprobs_idx=next_pp_outputs["input_token_ids_logprobs_idx"],
    )
    extend_input_len_per_req = next_pp_outputs["extend_input_len_per_req"]
    extend_logprob_start_len_per_req = next_pp_outputs[
        "extend_logprob_start_len_per_req"
    ]

    return logits_output, extend_input_len_per_req, extend_logprob_start_len_per_req


@dataclass
class EmbeddingBatchResult:
    """Result from an embedding/classification forward pass.

    Attributes:
        embeddings: Model output — pooled embeddings or classification logits.
        pooled_hidden_states: Raw hidden states before the task head.  Present
            only when the batch contained ``return_pooled_hidden_states=True``
            requests.  Tensor (uniform shapes) or list of tensors (MIS).
        copy_done: CUDA event recorded after the async CPU copy completes.

    中译：embedding / 分类前向的结果容器（与 GenerationBatchResult 对应，但无需采样）。
          embeddings：池化后的嵌入向量或分类 logits；pooled_hidden_states：任务头之前的原始
          隐藏状态（仅当批次含 return_pooled_hidden_states=True 的请求时存在；统一形状时为单个
          张量，MIS 多形状时为张量列表）；copy_done：异步拷回 CPU 完成后记录的 CUDA event。
    """

    embeddings: torch.Tensor
    pooled_hidden_states: Optional[torch.Tensor] = None
    copy_done: Optional[torch.cuda.Event] = None

    @property
    def can_run_cuda_graph(self) -> bool:
        # 中译：embedding 路径不走 cuda graph，恒为 False（与生成路径接口对齐）。
        return False

    def copy_to_cpu(self):
        """Copy embeddings and pooled hidden states to CPU for overlap scheduling.

        中译：重叠调度下把 embeddings 与 pooled_hidden_states 异步拷回 CPU。
              embeddings 可能是单个张量，也可能是张量列表（MIS），分别处理；空列表直接返回。
        """
        if isinstance(self.embeddings, torch.Tensor):
            self.copy_done = torch.get_device_module(self.embeddings.device).Event()
            self.embeddings = self.embeddings.to("cpu", non_blocking=True)
        else:
            assert isinstance(self.embeddings, list)
            if len(self.embeddings) == 0:
                return

            self.copy_done = torch.get_device_module(self.embeddings[0].device).Event()
            self.embeddings = [
                emb.to("cpu", non_blocking=True) for emb in self.embeddings
            ]

        # 中译：若存在任务头前的隐藏状态，同样异步拷回 CPU（列表/单张量两种形态分别处理）。
        if self.pooled_hidden_states is not None:
            if isinstance(self.pooled_hidden_states, list):
                self.pooled_hidden_states = [
                    t.to("cpu", non_blocking=True) for t in self.pooled_hidden_states
                ]
            else:
                self.pooled_hidden_states = self.pooled_hidden_states.to(
                    "cpu", non_blocking=True
                )

        # 中译：记录拷贝完成事件，供消费方等待后安全读取。
        self.copy_done.record()


def is_health_check_generate_req(recv_req):
    # 中译：判断收到的生成请求是否为健康检查请求（按 rid 前缀识别，健康检查不计入正常推理统计）。
    rid = getattr(recv_req, "rid", None)
    return rid is not None and rid.startswith(HEALTH_CHECK_RID_PREFIX)
