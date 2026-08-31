from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from sglang.srt.speculative.adaptive_spec_params import AdaptiveSpeculativeParams

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
    from sglang.srt.model_executor.runner import DecodeCudaGraphRunner
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
        EAGLEDraftExtendCudaGraphRunner,
    )


@dataclass
class SpecRuntimeState:
    """绑定到特定投机解码配置的完整运行时资源集合。

    每轮解码执行三个阶段——草稿、验证、扩展——每个阶段都有
    依赖于当前配置形状的资源（注意力后端和 CUDA Graph）。
    切换自适应步数意味着原子性地交换整个状态。

    示例::

        >>> state = SpecRuntimeState(
        ...     speculative_num_steps=3,
        ...     speculative_num_draft_tokens=4,
        ...     draft_attn_backend=draft_attn,
        ...     cuda_graph_runner=draft_graph,
        ...     target_attn_backend=target_attn,
        ...     target_graph_runner=target_graph,
        ...     draft_extend_attn_backend=extend_attn,
        ...     cuda_graph_runner_for_draft_extend=extend_graph,
        ... )
    """

    # -- 配置（决定所有阶段的形状）--
    speculative_num_steps: int
    speculative_num_draft_tokens: int

    # -- 草稿阶段：草稿模型多步自回归生成 --
    draft_attn_backend: "AttentionBackend | None"
    cuda_graph_runner: "EAGLEDraftCudaGraphRunner | None"

    # -- 验证阶段：目标模型一次性树形验证 --
    target_attn_backend: "AttentionBackend"
    target_graph_runner: "DecodeCudaGraphRunner | CPUGraphRunner | None"

    # -- 扩展阶段：验证后草稿模型的 KV cache 追赶 --
    draft_extend_attn_backend: "AttentionBackend | None"
    cuda_graph_runner_for_draft_extend: "EAGLEDraftExtendCudaGraphRunner | None"


class AdaptiveSpecWorker(Protocol):
    """Worker 必须实现的协议，以使用 AdaptiveController。

    Worker 需要实现两个方法:
      - ``build_adaptive_runtime_state(steps, draft_tokens)`` → 构建运行时状态
      - ``apply_runtime_state(state)`` → 将状态应用到 worker
    """

    speculative_num_steps: int

    def build_adaptive_runtime_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: list[int] | None = None,
    ) -> SpecRuntimeState: ...

    def apply_runtime_state(self, state: SpecRuntimeState) -> None: ...


class AdaptiveController:
    """自适应决策和运行时状态切换的门面对象。

    与任何实现了 AdaptiveSpecWorker 协议的 worker 配合使用:
      - ``build_adaptive_runtime_state(steps, draft_tokens)`` → 构建运行时状态
      - ``apply_runtime_state(state)`` → 应用状态到 worker

    Worker 只需要:
      1. 启动时调用 ``register()`` 注册初始状态，然后调用 ``init_states()``
      2. 每次验证完成后调用 ``on_verify_complete(num_correct_drafts_per_req)``

    示例::

        >>> controller = AdaptiveController(worker, config_path=None)
        >>> controller.register(initial_state)
        >>> controller.init_states(cuda_graph_bs=[1, 2, 4, 8])
        >>> # 每轮验证后:
        >>> controller.on_verify_complete([3, 2, 4], batch_size=3)
        >>> # 自动切换步数和对应的运行时状态
    """

    def __init__(self, worker: AdaptiveSpecWorker, config_path: str | None = None):
        """
        输入:
            worker: 实现了 AdaptiveSpecWorker 协议的 worker
            config_path: 自适应配置文件路径（None 时使用默认配置）
        """
        self.worker = worker
        self.params = AdaptiveSpeculativeParams(
            initial_steps=worker.speculative_num_steps,
            cfg_path=config_path,
        )
        self._states: dict[int, SpecRuntimeState] = {}

    @property
    def candidate_steps(self) -> list[int]:
        """所有候选步数列表。"""
        return self.params.candidate_steps

    def register(self, state: SpecRuntimeState, steps: int | None = None) -> None:
        """注册一个预构建的运行时状态。

        输入:
            state: 运行时状态
            steps: 对应的步数（None 时从 state.speculative_num_steps 推断）
        """
        key = steps if steps is not None else state.speculative_num_steps
        self._states[key] = state

    def init_states(self, cuda_graph_bs: list[int] | None = None) -> None:
        """为所有候选步数构建并注册运行时状态。

        输入:
            cuda_graph_bs: CUDA Graph 支持的 batch size 列表
        """
        self.params.set_cuda_graph_bs(cuda_graph_bs)

        for steps in self.candidate_steps:
            if steps in self._states:
                continue

            pruned_bs = self.params.cuda_graph_bs_for_step(steps)
            state = self.worker.build_adaptive_runtime_state(
                speculative_num_steps=steps,
                speculative_num_draft_tokens=steps + 1,
                cuda_graph_bs=pruned_bs,
            )
            self._states[steps] = state

        # 从初始步数开始
        self._activate(self.worker.speculative_num_steps)

    def activate_step_by_batch(self, batch_size: int) -> None:
        """根据当前批次大小激活最优步数。

        输入:
            batch_size: 当前批次大小

        示例::

            >>> controller.activate_step_by_batch(batch_size=1)
            >>> # bs=1 时可能激活 7 步草稿
            >>> controller.activate_step_by_batch(batch_size=32)
            >>> # bs=32 时可能激活 1 步草稿
        """
        target = self.params.get_steps_for_batch(batch_size)
        if target != self.worker.speculative_num_steps:
            self._activate(target)

    def on_verify_complete(
        self, num_correct_drafts_per_req: list[int], batch_size: int
    ) -> None:
        """馈送验证结果，若 EMA 判定需要切换则切换运行时状态。

        输入:
            num_correct_drafts_per_req: 各请求接受的草稿数列表
            batch_size: 当前批次大小
        """
        new_step = self.params.on_verify_complete(
            num_correct_drafts_per_req, batch_size
        )
        if new_step is not None:
            self._activate(new_step)

    def _activate(self, speculative_num_steps: int) -> None:
        """激活指定步数对应的运行时状态。

        输入:
            speculative_num_steps: 目标步数

        异常:
            ValueError: 当未注册对应步数的状态时
        """
        state = self._states.get(speculative_num_steps)
        if state is None:
            raise ValueError(
                f"缺少 steps={speculative_num_steps} 的自适应运行时状态"
            )
        self.worker.apply_runtime_state(state)
