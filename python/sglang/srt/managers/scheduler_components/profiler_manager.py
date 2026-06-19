"""中译：调度器侧的「性能分析（profiling）」管理模块。

封装 SchedulerProfilerManager，负责响应 /start_profile、/stop_profile 等控制请求，
驱动 PyTorch Profiler（CPU/GPU 活动、显存历史、CUDA Profiler）以及 ROCm 的 RPD profiler，
按「forward 步数」或「按阶段（prefill/decode）」两种方式控制采集的起止，并在结束后
导出 chrome trace（可选合并多 rank 的 trace）。当设置环境变量 SGLANG_PROFILE_V2 时，
全部委托给新的 ProfileManager 实现（本类只做转发）。
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
)

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import ProfileReq, ProfileReqOutput, ProfileReqType
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import is_npu
from sglang.srt.utils.profile_merger import ProfileMerger
from sglang.srt.utils.torch_npu_patch_utils import apply_torch_npu_patches

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

_is_npu = is_npu()
# 中译：在昇腾 NPU 平台上，torch.profiler 的若干符号需要替换为 torch_npu 的对应实现，
#       这里用 monkey-patch 把 profiler.profile 及 ProfilerActivity.CUDA/CPU 映射到 NPU 版本，
#       从而让后续代码无需区分平台即可调用统一的 torch.profiler 接口。
if _is_npu:
    import torch_npu

    patches = [
        ["profiler.profile", torch_npu.profiler.profile],
        ["profiler.ProfilerActivity.CUDA", torch_npu.profiler.ProfilerActivity.NPU],
        ["profiler.ProfilerActivity.CPU", torch_npu.profiler.ProfilerActivity.CPU],
    ]
    apply_torch_npu_patches(torch_npu, patches)

logger = logging.getLogger(__name__)


from sglang.srt.utils.profile_utils import ProfileManager


@dataclass(kw_only=True)
class SchedulerProfilerManager:
    """中译：调度器的性能分析管理器。

    持有 profiling 所需的全部运行时状态与目标计数，被 Scheduler 组合使用。
    字段：
    - ps：parallel state，包含 tp/dp/pp/moe_ep 的 rank 与 size，用于决定哪个 rank
      负责落盘、做 barrier 同步、生成 trace 文件名等。
    - dp_tp_cpu_group：用于 torch.distributed.barrier 的 CPU 通信组。
    - get_forward_ct：回调，返回当前已执行的 forward 次数（步数计），用于判断起止时机。
    """

    ps: Any
    dp_tp_cpu_group: Any
    get_forward_ct: Callable[[], int]

    def __post_init__(self) -> None:
        # 中译：若开启 PROFILE_V2，则改用新的 ProfileManager 实现，本类其余状态不再初始化、
        #       所有方法都提前 return 到 _profile_manager；否则初始化下面这套 v1 状态机。
        if envs.SGLANG_PROFILE_V2.get():
            self._profile_manager = ProfileManager(
                ps=self.ps,
                cpu_group=self.dp_tp_cpu_group,
            )
            return

        # 中译：torch profiler 实例与输出目录、采集活动类型（CPU/GPU/MEM 等）、本次 profile 的标识。
        self.torch_profiler = None
        self.torch_profiler_output_dir: Optional[Path] = None
        self.profiler_activities: Optional[List[str]] = None
        self.profile_id: Optional[str] = None

        # 中译：按「forward 步数」控制采集时的起止目标——到达 start_ct 时开始，达到 target_ct 时停止。
        self.profiler_start_forward_ct: Optional[int] = None
        self.profiler_target_forward_ct: Optional[int] = None

        # 中译：按「阶段」控制采集时，分别统计 prefill/decode 的已执行步数与目标步数。
        self.profiler_prefill_ct: Optional[int] = None
        self.profiler_decode_ct: Optional[int] = None
        self.profiler_target_prefill_ct: Optional[int] = None
        self.profiler_target_decode_ct: Optional[int] = None

        # 中译：profile_by_stage——是否按阶段分别采集；profile_in_progress——当前是否正在采集中；
        #       merge_profiles——结束后是否合并多 rank 的 trace 文件。
        self.profile_by_stage: bool = False
        self.profile_in_progress: bool = False
        self.merge_profiles = False

        # For ROCM
        # 中译：ROCm（AMD GPU）平台专用的 RPD profiler 句柄，CUDA/NPU 平台保持为 None。
        self.rpd_profiler = None

    def _init_profile(
        self,
        output_dir: Optional[str],
        start_step: Optional[int],
        num_steps: Optional[int],
        activities: Optional[List[str]],
        with_stack: Optional[bool],
        record_shapes: Optional[bool],
        profile_by_stage: bool,
        profile_id: str,
        merge_profiles: bool = False,
        profile_prefix: str = "",
        profile_stages: Optional[List[str]] = None,
    ) -> ProfileReqOutput:
        # 中译：解析并保存本次 profiling 的配置（输出目录、起始步、步数、活动类型、是否记录调用栈/形状、
        #       是否按阶段、profile_id 等），但不立即开始采集；实际开始由 _start_profile 触发。
        if envs.SGLANG_PROFILE_V2.get():
            return self._profile_manager.configure(
                output_dir=output_dir,
                start_step=start_step,
                num_steps=num_steps,
                activities=activities,
                with_stack=with_stack,
                record_shapes=record_shapes,
                profile_by_stage=profile_by_stage,
                profile_id=profile_id,
                merge_profiles=merge_profiles,
                profile_prefix=profile_prefix,
                profile_stages=profile_stages,
            )

        # 中译：若已有采集在进行中，拒绝重复配置，提示先调用 /stop_profile。
        if self.profile_in_progress:
            return ProfileReqOutput(
                success=False,
                message="Profiling is already in progress. Call /stop_profile first.",
            )

        self.profile_by_stage = profile_by_stage
        self.merge_profiles = merge_profiles

        # 中译：输出目录缺省取环境变量 SGLANG_TORCH_PROFILER_DIR（再兜底 /tmp）；
        #       活动类型缺省同时采集 CPU 与 GPU。
        if output_dir is None:
            output_dir = os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp")
        if activities is None:
            activities = ["CPU", "GPU"]

        self.torch_profiler_output_dir = Path(output_dir).expanduser()
        self.torch_profiler_with_stack = with_stack
        self.torch_profiler_record_shapes = record_shapes
        self.profiler_activities = activities
        self.profile_id = profile_id
        self.profile_prefix = profile_prefix

        # 中译：指定了起始步时，起点取 max(start_step, 当前 forward 计数+1)，确保不早于当前步。
        if start_step:
            self.profiler_start_forward_ct = max(start_step, self.get_forward_ct() + 1)

        # 中译：指定了采集步数 num_steps 时，依据三种模式计算「停止」目标：
        if num_steps:
            if self.profile_by_stage:
                # 中译：按阶段——prefill/decode 各自从 0 计数，分别采集 num_steps 步。
                self.profiler_prefill_ct = 0
                self.profiler_decode_ct = 0
                self.profiler_target_prefill_ct = num_steps
                self.profiler_target_decode_ct = num_steps
            elif start_step:
                # 中译：指定了起始步——目标步 = 起始步 + num_steps。
                self.profiler_target_forward_ct = (
                    self.profiler_start_forward_ct + num_steps
                )
            else:
                # 中译：未指定起始步——立即从当前步起算，采集 num_steps 步。
                self.profiler_target_forward_ct = self.get_forward_ct() + num_steps
            # The caller will be notified when reaching profiler_target_forward_ct
            # 中译：到达 profiler_target_forward_ct 时会通知调用方（采集自动结束）。
        else:
            # 中译：未指定步数——表示手动控制（由显式的 stop 请求结束），无自动停止目标。
            self.profiler_target_forward_ct = None

        return ProfileReqOutput(success=True, message="Succeeded")

    def _start_profile(
        self, stage: Optional[ForwardMode] = None
    ) -> ProfileReqOutput | None:
        """中译：真正启动采集器。

        根据 profiler_activities 选择启用对应后端：
        torch profiler（CPU/GPU）、显存历史记录（MEM）、CUDA Profiler、或 ROCm 的 RPD。
        stage 仅在按阶段采集时传入，用于日志区分当前是 prefill 还是 decode 阶段。
        """
        if envs.SGLANG_PROFILE_V2.get():
            return self._profile_manager.manual_start()

        stage_str = f" for {stage.name}" if stage else ""
        logger.info(
            f"Profiling starts{stage_str}. Traces will be saved to: {self.torch_profiler_output_dir} (with profile id: {self.profile_id})",
        )

        activities = self.profiler_activities
        with_stack = self.torch_profiler_with_stack
        record_shapes = self.torch_profiler_record_shapes

        # 中译：把字符串形式的活动名（"CPU"/"GPU"/"XPU"）映射为 torch.profiler 的枚举；
        #       XPU（Intel GPU）仅在当前 torch 版本支持时加入。MEM/CUDA_PROFILER/RPD 不在此表，
        #       它们走下面各自独立的分支处理。
        activity_map = {
            "CPU": torch.profiler.ProfilerActivity.CPU,
            "GPU": torch.profiler.ProfilerActivity.CUDA,
        }
        if hasattr(torch.profiler.ProfilerActivity, "XPU"):
            activity_map["XPU"] = torch.profiler.ProfilerActivity.XPU
        torchprof_activities = [
            activity_map[a] for a in activities if a in activity_map
        ]

        if "RPD" in activities:  # for ROCM
            # 中译：ROCm 平台——使用 RPD tracer 采集。rank 0 负责创建/重置 trace.rpd 的 sqlite 库结构，
            #       barrier 后各 rank 同步开始；rangePush 标记采集区间。
            from rpdTracerControl import rpdTracerControl

            rpdTracerControl.skipCreate()

            self.rpd_profile_path = os.path.join(
                self.torch_profiler_output_dir,
                "rpd-" + str(time.time()) + f"-TP-{self.ps.tp_rank}" + ".trace.json.gz",
            )

            if self.ps.tp_rank == 0:
                import sqlite3

                from rocpd.schema import RocpdSchema

                if os.path.exists("trace.rpd"):
                    os.unlink("trace.rpd")
                schema = RocpdSchema()
                connection = sqlite3.connect("trace.rpd")
                schema.writeSchema(connection)
                connection.commit()
                del connection
            torch.distributed.barrier(self.dp_tp_cpu_group)

            self.rpd_profiler = rpdTracerControl()
            self.rpd_profiler.setPythonTrace(True)
            self.rpd_profiler.start()
            self.rpd_profiler.rangePush("", "rpd profile range", "")
            self.profile_in_progress = True
        elif torchprof_activities:
            # 中译：常规 CUDA/CPU/XPU 路径——构造 torch.profiler.profile。
            #       with_stack 缺省 True（记录调用栈），record_shapes 缺省 False；
            #       NPU 平台额外提供 tensorboard trace handler 与昇腾专用的 experimental_config。
            self.torch_profiler = torch.profiler.profile(
                activities=torchprof_activities,
                with_stack=with_stack if with_stack is not None else True,
                record_shapes=record_shapes if record_shapes is not None else False,
                on_trace_ready=(
                    None
                    if not _is_npu
                    else torch_npu.profiler.tensorboard_trace_handler(
                        str(self.torch_profiler_output_dir)
                    )
                ),
                experimental_config=(
                    None
                    if not _is_npu
                    else torch_npu.profiler._ExperimentalConfig(
                        export_type=torch_npu.profiler.ExportType.Text,
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        msprof_tx=False,
                        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                        l2_cache=False,
                        op_attr=False,
                        data_simplification=False,
                        record_op_args=False,
                        gc_detect_threshold=None,
                    )
                ),
            )
            self.torch_profiler.start()
            self.profile_in_progress = True

        # 中译：MEM——开启 CUDA 显存分配历史记录，停止时再 dump 出快照用于分析显存占用。
        if "MEM" in activities:
            torch.cuda.memory._record_memory_history(max_entries=100000)
            self.profile_in_progress = True

        # 中译：CUDA_PROFILER——调用 cudaProfilerStart（配合 nsys 等外部工具采集），
        #       仅在持有 base_gpu_id 的进程上触发，避免多进程重复启动。
        if "CUDA_PROFILER" in activities:
            if self.ps.gpu_id == get_global_server_args().base_gpu_id:
                torch.cuda.cudart().cudaProfilerStart()
            self.profile_in_progress = True

        return ProfileReqOutput(success=True, message="Succeeded")

    def _merge_profile_traces(self) -> str:
        """中译：把多个 rank 各自导出的 chrome trace 合并为一个文件，返回追加到结果消息里的说明串。

        仅当开启 merge_profiles 时执行，且只在「全局唯一」的那个 rank（各并行维度均为 0 号）上做合并，
        避免多个进程重复合并。失败时返回错误说明而不抛出，以免影响 stop 流程。
        """
        if not self.merge_profiles:
            return ""

        # 中译：只有所有并行维度（tp/dp/pp/moe_ep）都是 rank 0 的进程才负责合并；其余直接返回空串。
        if self.ps.tp_rank != 0:
            return ""
        if self.ps.dp_size > 1 and self.ps.dp_rank != 0:
            return ""
        if self.ps.pp_size > 1 and self.ps.pp_rank != 0:
            return ""
        if self.ps.moe_ep_size > 1 and self.ps.moe_ep_rank != 0:
            return ""

        try:
            logger.info("Starting profile merge...")
            merger = ProfileMerger(self.torch_profiler_output_dir, self.profile_id)
            merged_path = merger.merge_chrome_traces()

            summary = merger.get_merge_summary()
            merge_message = (
                f" Merged trace: {merged_path} "
                f"(Events: {summary.get('total_events', '?')}, "
                f"Files: {summary.get('total_files', '?')})"
            )

            logger.info(f"Profile merge completed: {merged_path}")
        except Exception as e:
            logger.error(f"Failed to merge profiles: {e}", exc_info=True)
            return f" Merge failed: {e!s}"
        else:
            return merge_message

    def _stop_profile(
        self, stage: Optional[ForwardMode] = None
    ) -> ProfileReqOutput | None:
        """中译：停止采集并落盘。

        依次停止 torch profiler（导出 chrome trace）、RPD profiler、显存历史快照与 CUDA Profiler，
        必要时合并多 rank trace，最后清空运行状态以便下一轮采集。
        """
        if envs.SGLANG_PROFILE_V2.get():
            return self._profile_manager.manual_stop()

        # 中译：当前没有采集在进行时拒绝停止，提示先 /start_profile。
        if not self.profile_in_progress:
            return ProfileReqOutput(
                success=False,
                message="Profiling is not in progress. Call /start_profile first.",
            )

        # 中译：确保输出目录存在（递归创建，已存在不报错）。
        self.torch_profiler_output_dir.mkdir(parents=True, exist_ok=True)

        if self.profile_prefix:
            stage_prefix = self.profile_prefix + "-"
        else:
            stage_prefix = ""

        stage_suffix = f"-{stage.name}" if stage else ""
        logger.info("Stop profiling" + stage_suffix + "...")
        if self.torch_profiler is not None:
            self.torch_profiler.stop()
            # 中译：NPU 平台在 start 时已通过 tensorboard_trace_handler 自动落盘，这里无需手动导出；
            #       非 NPU 才手动 export chrome trace。
            if not _is_npu:
                # Build filename with only non-zero ranks to maintain backward compatibility
                # 中译：文件名包含 profile_id 与 TP rank；其余并行维度仅在启用（size>1）时追加，
                #       以保持与历史文件名的向后兼容。
                filename_parts = [self.profile_id, f"TP-{self.ps.tp_rank}"]

                # Only add other ranks if parallelism is enabled (size > 1)
                if self.ps.dp_size > 1:
                    filename_parts.append(f"DP-{self.ps.dp_rank}")
                if self.ps.pp_size > 1:
                    filename_parts.append(f"PP-{self.ps.pp_rank}")
                if self.ps.moe_ep_size > 1:
                    filename_parts.append(f"EP-{self.ps.moe_ep_rank}")

                filename = (
                    stage_prefix
                    + "-".join(filename_parts)
                    + stage_suffix
                    + ".trace.json.gz"
                )

                self.torch_profiler.export_chrome_trace(
                    os.path.join(self.torch_profiler_output_dir, filename)
                )
            # 中译：barrier 确保所有 rank 都完成导出后再继续，避免后续合并读到不完整文件。
            torch.distributed.barrier(self.dp_tp_cpu_group)

        if self.rpd_profiler is not None:
            # 中译：停止 RPD 采集并 flush；rank 0 把 trace.rpd 转换为 chrome trace 格式后清理句柄。
            self.rpd_profiler.rangePop()
            self.rpd_profiler.stop()
            self.rpd_profiler.flush()

            torch.distributed.barrier(self.dp_tp_cpu_group)
            if self.ps.tp_rank == 0:
                from sglang.srt.utils.rpd_utils import rpd_to_chrome_trace

                rpd_to_chrome_trace("trace.rpd", self.rpd_profile_path)
            self.rpd_profiler = None
            self.rpd_profile_path = None

        # 中译：MEM——把显存分配历史 dump 成 pickle 快照，并关闭历史记录。
        if self.profiler_activities is not None and "MEM" in self.profiler_activities:
            memory_profile_path = os.path.join(
                self.torch_profiler_output_dir,
                str(time.time())
                + f"-TP-{self.ps.tp_rank}-memory"
                + stage_suffix
                + ".pickle",
            )
            torch.cuda.memory._dump_snapshot(memory_profile_path)
            torch.cuda.memory._record_memory_history(enabled=None)

        # 中译：CUDA_PROFILER——与 start 对称，仅在 base_gpu_id 进程上调用 cudaProfilerStop。
        if "CUDA_PROFILER" in self.profiler_activities:
            if self.ps.gpu_id == get_global_server_args().base_gpu_id:
                torch.cuda.cudart().cudaProfilerStop()

        # 中译：按需合并多 rank 的 trace，得到追加到结果消息里的说明串。
        merge_message = self._merge_profile_traces()

        logger.info(
            "Profiling done. Traces are saved to: %s%s",
            self.torch_profiler_output_dir,
            merge_message,
        )
        # 中译：清空运行状态，使采集器回到「空闲」可重新配置/启动的状态。
        self.torch_profiler = None
        self.profile_in_progress = False
        self.profiler_start_forward_ct = None

        return ProfileReqOutput(success=True, message=f"Succeeded.{merge_message}")

    def _profile_batch_predicate(self, batch: ScheduleBatch):
        """中译：每个 batch 执行前由调度器调用的「采集起止判定」钩子。

        根据当前 batch 的 forward_mode（prefill/decode/idle）与已配置的步数目标，
        自动决定是否在此刻 start/stop 采集，从而实现「自动按步数/按阶段」采集。
        """
        if envs.SGLANG_PROFILE_V2.get():
            self._profile_manager.step(forward_mode=batch.forward_mode)
            return

        # 中译：按阶段模式——prefill 与 decode 各自独立计数与起止。
        if self.profile_by_stage:
            if batch.forward_mode.is_prefill():
                # 中译：首次进入 prefill（计数为 0）时开始采集；每个 prefill batch 计数 +1，
                #       超过目标步数则停止（阶段标记为 EXTEND）。
                if self.profiler_prefill_ct == 0:
                    self._start_profile(batch.forward_mode)
                self.profiler_prefill_ct += 1
                if self.profiler_prefill_ct > self.profiler_target_prefill_ct:
                    if self.profile_in_progress:
                        self._stop_profile(stage=ForwardMode.EXTEND)
            elif batch.forward_mode.is_decode():
                if self.profiler_decode_ct == 0:
                    if self.profile_in_progress:
                        # force trace flush
                        # 中译：首个 decode batch 到来时，若 prefill 阶段采集仍在进行，
                        #       先强制停止以 flush prefill 的 trace，再为 decode 单独开始一段采集。
                        self._stop_profile(stage=ForwardMode.EXTEND)
                    self._start_profile(batch.forward_mode)
                self.profiler_decode_ct += 1
                if self.profiler_decode_ct > self.profiler_target_decode_ct:
                    if self.profile_in_progress:
                        self._stop_profile(stage=ForwardMode.DECODE)
            elif batch.forward_mode.is_idle():
                # 中译：空转（idle）batch 不计入采集步数，跳过。
                pass
            else:
                raise RuntimeError(f"unsupported profile stage: {batch.forward_mode}")
        else:
            # Check profiler
            # 中译：按步数模式——到达目标步则停止；恰好等于起始步则开始。
            #       注意 stop 的判定写在 start 之前，保证「先结束上一轮、再开启下一轮」的顺序正确。
            if (
                self.profiler_target_forward_ct
                and self.profiler_target_forward_ct <= self.get_forward_ct()
            ):
                self._stop_profile()
            if (
                self.profiler_start_forward_ct
                and self.profiler_start_forward_ct == self.get_forward_ct()
            ):
                self._start_profile()

    def _profile(self, recv_req: ProfileReq):
        """中译：处理外部 ProfileReq 请求的入口（START_PROFILE / STOP_PROFILE）。

        对于 START：若是「按阶段」或指定了起始步，则只做配置（_init_profile），实际采集交给
        _profile_batch_predicate 在跑到对应步时自动触发；否则配置后立即 _start_profile。
        对于 STOP：直接 _stop_profile。
        """
        if recv_req.type == ProfileReqType.START_PROFILE:
            # 中译：按阶段 / 指定起始步——延迟启动，只返回配置结果，后续由 predicate 自动起止。
            if recv_req.profile_by_stage or recv_req.start_step:
                return self._init_profile(
                    recv_req.output_dir,
                    recv_req.start_step,
                    recv_req.num_steps,
                    recv_req.activities,
                    recv_req.with_stack,
                    recv_req.record_shapes,
                    recv_req.profile_by_stage,
                    recv_req.profile_id,
                    recv_req.merge_profiles,
                    recv_req.profile_prefix,
                    recv_req.profile_stages,
                )
            else:
                # 中译：未指定起始步且非按阶段——先完成配置，再立即开始采集。
                self._init_profile(
                    recv_req.output_dir,
                    recv_req.start_step,
                    recv_req.num_steps,
                    recv_req.activities,
                    recv_req.with_stack,
                    recv_req.record_shapes,
                    recv_req.profile_by_stage,
                    recv_req.profile_id,
                    recv_req.merge_profiles,
                    recv_req.profile_prefix,
                )
                return self._start_profile()
        else:
            # 中译：STOP_PROFILE——停止当前采集并落盘。
            return self._stop_profile()
