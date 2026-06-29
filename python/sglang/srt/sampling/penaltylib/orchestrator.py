from __future__ import annotations

import abc
import weakref
from typing import TYPE_CHECKING, Optional, Set, Type

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


class BatchedPenalizerOrchestrator:
    """惩罚器（penalizer）的批量编排器。

    负责统一管理一个批次（batch）内的所有采样惩罚项（如重复惩罚、频率惩罚、
    存在惩罚等），并在解码过程中把它们整体作用到 logits 上。

    设计要点：
    - 通过弱引用（weakref）持有 ScheduleBatch，避免与 batch 形成循环引用导致
      内存无法及时回收。
    - 采用「按需准备」（lazy prepare）策略：只有当某个惩罚器在当前批次中确实
      被请求需要时，才会真正分配其内部张量，从而节省显存。
    - 同时支持加性惩罚（直接加到 logits 上）和乘性/缩放惩罚（按比例缩放 logits）
      两类惩罚器。
    """

    def __init__(
        self,
        vocab_size: int,
        batch: ScheduleBatch,
        penalizers: Set[Type[_BatchedPenalizer]],
    ):
        self.vocab_size = vocab_size
        # 用弱引用持有 batch，防止编排器与 batch 互相强引用形成循环引用
        self._batch_ref = weakref.ref(batch)
        self.device = batch.device
        # 为传入的每个惩罚器类型实例化一个对象，建立「类型 -> 实例」的映射
        self.penalizers = {Penalizer: Penalizer(self) for Penalizer in penalizers}

        # 遍历所有惩罚器，对当前批次中确实需要的惩罚器执行准备（分配张量等），
        # 只要有任意一个惩罚器被需要，整个编排器就标记为「需要生效」
        is_required = False
        for penalizer in self.penalizers.values():
            pen_is_required = penalizer.prepare_if_required()
            is_required |= pen_is_required
        self.is_required = is_required

    @property
    def batch(self) -> ScheduleBatch | None:
        # 通过弱引用取回 batch；若已被回收则返回 None
        return self._batch_ref()

    @batch.setter
    def batch(self, value: Optional[ScheduleBatch]):
        if value is None:
            # 显式置空时，用一个始终返回 None 的可调用对象替代弱引用，
            # 保证 self.batch 的访问行为（调用 self._batch_ref()）依然一致
            self._batch_ref = lambda: None
        else:
            self._batch_ref = weakref.ref(value)

    def reqs(self):
        return self.batch.reqs

    def cumulate_output_tokens(self, output_ids: torch.Tensor):
        """把新生成的输出 token 喂给所有惩罚器。

        解码每生成一步，都需要让各惩罚器累计已出现的 token（例如更新词频
        统计、记录出现过的 token 等），以便下一步计算惩罚值。

        Args:
            output_ids (torch.Tensor): 本步生成的输出 token。
        """
        for penalizer in self.penalizers.values():
            penalizer.cumulate_output_tokens(output_ids=output_ids)

    def apply(self, logits: torch.Tensor, repeat: Optional[int] = None):
        """就地（in-place）把所有惩罚器作用到 logits 上。

        Args:
            logits: 待施加惩罚的 logits 张量。
            repeat: 若设置（用于投机解码 speculative decoding），则每个请求的
                惩罚值会通过 repeat_interleave 扩展，以匹配 draft token 的布局。
                具体做法：加性惩罚先写入一个全零张量，扩展后再加到 logits 上；
                缩放惩罚先累乘汇总，扩展后再直接应用。
        """
        if repeat is None:
            # 常规路径：每个请求与一行 logits 一一对应，逐个惩罚器直接应用即可
            for penalizer in self.penalizers.values():
                penalizer.apply(logits)
        else:
            # 投机解码路径：一个请求对应 repeat 行 logits（draft token），
            # 因此需要先按「每请求」计算惩罚，再扩展到「每 draft token」。

            # 加性惩罚：先写入一个 [bs, vocab] 的全零张量，扩展后再加到 logits 上
            bs = logits.shape[0] // repeat
            additive = torch.zeros(
                (bs, logits.shape[1]), dtype=torch.float32, device=logits.device
            )
            self.accumulate_additive_penalties(additive)
            logits.add_(torch.repeat_interleave(additive, repeat, dim=0))

            # 缩放惩罚：先把所有乘性惩罚累乘成一个张量，扩展后再直接应用
            accumulated = self.accumulate_scaling_penalties()
            if accumulated is not None:
                from sglang.srt.sampling.penaltylib.repetition_penalty import (
                    apply_scaling_penalties,
                )

                expanded = torch.repeat_interleave(accumulated, repeat, dim=0)
                apply_scaling_penalties(logits, expanded)

    def accumulate_additive_penalties(self, logits: torch.Tensor):
        """仅应用加性（非乘性）惩罚器。"""
        for penalizer in self.penalizers.values():
            if not penalizer.is_multiplicative:
                penalizer.apply(logits)

    def accumulate_scaling_penalties(self) -> Optional[torch.Tensor]:
        """把所有乘性惩罚张量累乘汇总成一个；若没有生效的乘性惩罚则返回 None。"""
        result = None
        for penalizer in self.penalizers.values():
            # 跳过未准备好的、以及非乘性的惩罚器
            if not penalizer._is_prepared or not penalizer.is_multiplicative:
                continue
            if result is None:
                # 第一个乘性惩罚：克隆一份作为累乘起点，避免就地修改原张量
                result = penalizer.get_scaling_penalties().clone()
            else:
                result *= penalizer.get_scaling_penalties()
        return result

    def filter(self, keep_indices: torch.Tensor):
        """根据要保留的索引过滤各惩罚器。

        当批次中部分请求完成（finish）被移除时，需要把惩罚器内部张量裁剪到
        仅保留 keep_indices 对应的行。

        Args:
            keep_indices (torch.Tensor): 批次中需要保留的请求索引张量。
        """
        if not self.is_required:
            return

        if len(keep_indices) == 0:
            # 批次中已无请求，直接彻底释放编排器持有的全部资源
            self.release()
            return

        # 过滤后重新评估整体是否仍需要生效：对仍被需要的惩罚器执行过滤，
        # 对不再被需要的惩罚器执行 teardown 以释放其张量
        is_required = False
        for penalizer in self.penalizers.values():
            tmp_is_required = penalizer.is_required()
            is_required |= tmp_is_required
            if tmp_is_required:
                penalizer.filter(keep_indices=keep_indices)
            else:
                penalizer.teardown()
        self.is_required = is_required

    # 资源管理辅助方法
    def release(self) -> None:
        """释放所有惩罚器并断开引用，使 GC 能够尽快回收相关资源。"""
        for penalizer in self.penalizers.values():
            penalizer.teardown()
        self.penalizers.clear()
        # 断开对 ScheduleBatch 的引用
        self._batch_ref = None
        self.is_required = False

    # 上下文管理器支持：可用 with 语句确保退出时自动释放资源
    def __enter__(self) -> BatchedPenalizerOrchestrator:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def merge(self, their: BatchedPenalizerOrchestrator):
        """把另一个编排器的惩罚器合并到当前编排器中。

        注意：本函数**必须**在 self.batch.reqs 被更新（过滤）**之前**调用。
        每个尚未准备的惩罚器都需要先完成准备（创建张量等）才能合并，而这一步
        依赖合并前的原始 batch.reqs（即尚未与其他 batch.reqs 合并的状态）。

        Args:
            their (BatchedPenalizerOrchestrator): 要被合并进来的另一个编排器。
        """
        # 两边都不需要惩罚则无需合并
        if not self.is_required and not their.is_required:
            return

        # 只要有一方需要，合并结果就一定需要生效
        self.is_required = True
        for penalizer, their_penalizer in their.penalizers.items():
            self.penalizers[penalizer].merge(their_penalizer)


class _BatchedPenalizer(abc.ABC):
    """批量惩罚器的抽象基类。

    定义了惩罚器的通用生命周期与接口。具体的惩罚逻辑（如重复惩罚、频率惩罚、
    存在惩罚等）由子类通过实现下划线前缀的钩子方法（_prepare、_apply 等）来完成。

    采用「公开方法 + 受保护钩子」的模板方法模式：公开方法（prepare、apply 等）
    统一处理「是否已准备」等通用状态判断，再委托给子类实现的钩子方法。

    类属性：
        is_multiplicative: 标记该惩罚器是加性（False）还是乘性/缩放（True）。
            编排器据此决定走加性累加路径还是乘性累乘路径。
    """

    is_multiplicative: bool = False

    def __init__(self, orchestrator: BatchedPenalizerOrchestrator):
        # 用弱引用持有编排器，避免与编排器形成循环引用
        self._orchestrator_ref: weakref.ReferenceType[BatchedPenalizerOrchestrator] = (
            weakref.ref(orchestrator)
        )
        # 标记内部张量是否已分配；未准备时各操作会直接跳过
        self._is_prepared = False

    @property
    def orchestrator(self) -> BatchedPenalizerOrchestrator:
        orch: Optional[BatchedPenalizerOrchestrator] = self._orchestrator_ref()
        # 正常情况下不应发生（惩罚器生命周期短于编排器），但仍需优雅处理
        if orch is None:
            raise RuntimeError(
                "BatchedPenalizerOrchestrator has been garbage-collected"
            )
        return orch

    def is_prepared(self) -> bool:
        return self._is_prepared

    def is_required(self) -> bool:
        return self._is_required()

    def prepare(self):
        # 幂等：仅在尚未准备时才执行准备，避免重复分配张量
        if not self._is_prepared:
            self._prepare()
            self._is_prepared = True

    def prepare_if_required(self):
        # 仅当该惩罚器在当前批次中确实被需要时才准备，返回是否被需要
        if self._is_required():
            self.prepare()
            return True
        else:
            return False

    def teardown(self):
        self._teardown()
        self._is_prepared = False

    def cumulate_output_tokens(self, output_ids: torch.Tensor):
        # 未准备的惩罚器无需累计 token
        if not self._is_prepared:
            return

        self._cumulate_output_tokens(output_ids=output_ids)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        # 未准备的惩罚器不施加任何惩罚
        if not self._is_prepared:
            return

        self._apply(logits=logits)

    def filter(self, keep_indices: torch.Tensor):
        # 未准备的惩罚器没有需要过滤的张量
        if not self._is_prepared:
            return

        self._filter(keep_indices=keep_indices)

    def merge(self, their: _BatchedPenalizer):
        # 两边都未准备则无需合并
        if not self._is_prepared and not their._is_prepared:
            return

        # 合并前确保双方都已准备（创建好张量），否则无法对齐合并
        self.prepare()
        their.prepare()
        self._merge(their)

    @abc.abstractmethod
    def _is_required(self) -> bool:
        """判断该惩罚器是否需要被准备（即当前批次中是否有请求用到它）。"""
        pass

    @abc.abstractmethod
    def _prepare(self):
        """准备惩罚器，通常在这里初始化其内部张量。"""
        pass

    @abc.abstractmethod
    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        """累计输出 token。编排器会调用本函数把输出 token 喂给惩罚器。"""
        pass

    @abc.abstractmethod
    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        """把惩罚作用到 logits 上。惩罚器可在需要时就地修改 logits。"""
        pass

    def get_scaling_penalties(self) -> torch.Tensor:
        """返回乘性惩罚器累计的缩放惩罚张量。

        仅当 is_multiplicative 为 True 时有意义，子类应重写本方法。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _filter(self, keep_indices: torch.Tensor):
        """根据批次中要保留的索引，过滤惩罚器的张量或底层数据。"""
        pass

    @abc.abstractmethod
    def _merge(self, their: _BatchedPenalizer):
        """把当前惩罚器与另一个惩罚器合并。"""
        pass

    @abc.abstractmethod
    def _teardown(self):
        """拆除惩罚器，释放其占用的资源。"""
        pass
