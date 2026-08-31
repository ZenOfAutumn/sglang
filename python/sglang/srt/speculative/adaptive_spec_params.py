"""自适应投机解码参数管理。

根据运行时观测到的接受长度（acceptance length）动态调整 ``speculative_num_steps``，
在命中率高时增加步数以提升吞吐，在命中率低时减少步数以降低浪费。

核心公式: ``target_steps = clamp(round(ema_accept_len) + 1, min_steps, max_steps)``
- 探测比观测接受长度多一步的草稿数
- EMA 平滑防止振荡
- 每 ``update_interval`` 个 batch 更新一次以保持稳定性
- 不同 batch_size 可使用不同的候选步数集合
"""

from __future__ import annotations

import bisect
import json
import logging
import math
from functools import cached_property
from typing import TYPE_CHECKING

from sglang.srt.arg_groups.overrides import (
    resolved_view,
    resolving_view,
)
from sglang.srt.utils import log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

DEFAULT_ADAPTIVE_CONFIG: dict[str, dict] = {
    "1": {
        "candidate_steps": [1, 3, 7],   # bs=1 时候选步数：1/3/7
        "up_hysteresis": 0.0,           # 升级滞后量
        "down_hysteresis": -0.25,       # 降级滞后量（负值 = 更快降级）
        "ceiling_coeff": 0,             # EMA 上限系数（0 = 不限）
    },
    "8": {
        "candidate_steps": [0, 1, 3],
        "up_hysteresis": 0.0,
        "down_hysteresis": 0.0,
        "ceiling_coeff": 0,
    },
    "32": {
        "candidate_steps": [0, 1],
        "up_hysteresis": 0.0,
        "down_hysteresis": 0.0,
        "ceiling_coeff": 0,
    },
    "64": {
        "candidate_steps": [0],
        "up_hysteresis": 0.0,
        "down_hysteresis": 0.0,
        "ceiling_coeff": 0,
    },
}


def adaptive_unsupported_reason(server_args: ServerArgs) -> str | None:
    """Return why adaptive spec cannot run under the given server args, or None if supported."""

    cfg = resolving_view(server_args)

    if cfg.speculative_algorithm not in ("EAGLE", "EAGLE3"):
        return (
            f"speculative_algorithm={cfg.speculative_algorithm} "
            "(only EAGLE/EAGLE3 are supported)"
        )
    if cfg.speculative_eagle_topk is not None and cfg.speculative_eagle_topk != 1:
        return (
            f"speculative_eagle_topk={cfg.speculative_eagle_topk} "
            "(only topk=1 is supported)"
        )
    if resolved_view(server_args).enable_dp_attention:
        return (
            "enable_dp_attention=True 不支持 "
            "(自适应层级决策无法跨 DP rank 同步)"
        )
    if resolved_view(server_args).enable_multi_layer_eagle:
        return (
            "enable_multi_layer_eagle=True 不支持 "
            "(MultiLayerEagleWorkerV2 未实现自适应)"
        )
    if cfg.enable_two_batch_overlap:
        return (
            "enable_two_batch_overlap=True 不支持 "
            "(自适应状态切换会丢弃 TboAttnBackend 包装器)"
        )
    if cfg.enable_pdmux:
        return (
            "enable_pdmux=True 不支持 "
            "(自适应状态切换不会更新 decode_attn_backend_group)"
        )
    return None


def _load_adaptive_config(
    cfg_path: str | None,
) -> tuple[dict, dict[int, dict]]:
    """加载并校验自适应配置。

    输入:
        cfg_path: 配置文件路径（None 时使用默认配置）

    输出:
        (cfg, bs_entries) 元组:
        - cfg: 原始配置字典
        - bs_entries: {batch_size: entry} 的整型键字典

    异常:
        ValueError: 当配置缺少 candidate_steps 或没有有效的 BS 键时
    """
    if cfg_path is not None:
        with open(cfg_path) as f:
            cfg = json.load(f)
    else:
        cfg = DEFAULT_ADAPTIVE_CONFIG

    bs_entries: dict[int, dict] = {}
    for key, entry in cfg.items():
        if not key.isdigit():
            continue

        steps = entry.get("candidate_steps")
        if (
            not isinstance(steps, list)
            or not steps
            or not all(isinstance(s, int) and s >= 0 for s in steps)
        ):
            raise ValueError(
                f"BS {key}: candidate_steps must be a list of non-negative ints, "
                f"got {steps!r}"
            )
        bs_entries[int(key)] = entry

    if not bs_entries:
        raise ValueError(
            "speculative_adaptive_config 必须包含至少一个整数字符串 BS 键，"
            '例如 {"1": {"candidate_steps": [1,3,7]}}。'
            f"得到键: {list(cfg.keys())}"
        )
    return cfg, bs_entries


def resolve_candidate_steps_from_config(
    cfg_path: str | None = None,
) -> list[int]:
    """获取所有 BS 槽位候选步数的并集，用于确定运行时缓冲区大小。

    输入:
        cfg_path: 配置文件路径（None 时使用默认配置）

    输出:
        排序后的候选步数列表

    示例::

        >>> resolve_candidate_steps_from_config()
        [1, 3, 7]  # 默认配置中所有 BS 槽位的步数并集
    """
    _, bs_entries = _load_adaptive_config(cfg_path)
    all_steps: set[int] = set()
    for entry in bs_entries.values():
        all_steps.update(entry["candidate_steps"])
    return sorted(all_steps)


class AdaptiveStepSlot:
    """通过 EMA 跟踪接受率并自适应调整 ``num_steps``。

    核心思想：如果草稿持续被接受，尝试更多步数；
    如果草稿早期就被拒绝，减少步数以避免浪费。

    公式: ``target_steps = clamp(round(ema_accept_len) + 1, min_steps, max_steps)``
    - 探测比观测接受长度多一步的草稿数
    - EMA 平滑防止振荡
    - 每 ``update_interval`` 个 batch 才更新一次以保持稳定性
    - 不同 batch_size 可使用不同的候选步数集合

    示例::

        >>> slot = AdaptiveStepSlot(initial_steps=3, cfg={
        ...     "candidate_steps": [1, 3, 7],
        ...     "ema_alpha": 0.2, "update_interval": 5, "warmup_batches": 10,
        ...     "down_hysteresis": -0.25, "up_hysteresis": 0.0, "ceiling_coeff": 0,
        ... })
        >>> slot.current_steps
        3
        >>> # 模拟多轮高接受率
        >>> for _ in range(15):
        ...     slot.update([6, 7, 5])  # 平均接受 6 个草稿
        >>> slot.current_steps  # 步数可能升到 7
        7
    """

    def __init__(self, initial_steps: int, cfg: dict):
        """
        输入:
            initial_steps: 初始步数
            cfg: 配置字典，包含 candidate_steps、ema_alpha 等
        """
        candidates = sorted(set(cfg["candidate_steps"]))
        assert len(candidates) >= 1, "candidate_steps 至少要有 1 个值"
        self.candidate_steps = candidates

        self.ema_alpha = cfg.get("ema_alpha", 0.2)
        self.update_interval = cfg.get("update_interval", 5)
        self.warmup_batches = cfg.get("warmup_batches", 10)
        self.down_hysteresis = cfg.get("down_hysteresis", -0.25)
        self.up_hysteresis = cfg.get("up_hysteresis", 0.0)
        self.ceiling_coeff = cfg.get("ceiling_coeff", 0)

        if initial_steps in self.candidate_steps:
            self.current_steps = initial_steps
        else:
            self.current_steps = self.candidate_steps[len(self.candidate_steps) // 2]

        # EMA 初始值为 current_steps - 1（中性起点）
        self.ema_accept_len = float(self.current_steps - 1)
        self._batch_count = 0

    def update(self, num_correct_drafts_per_req: list[int]) -> bool:
        """用观测到的接受长度更新 EMA，返回参数是否变化。

        输入:
            num_correct_drafts_per_req: 各请求上一轮验证接受的草稿 token 数列表
                例 [2, 0, 5] — 3 个请求分别猜对 2/0/5 个

        输出:
            bool: 如果步数发生变化返回 True，否则 False

        示例::

            >>> slot.update([3, 2, 4])  # 平均接受 3 个
            False  # 预热期内不更新
        """
        if not num_correct_drafts_per_req:
            return False

        if self.current_steps > 0:
            batch_avg = sum(num_correct_drafts_per_req) / len(
                num_correct_drafts_per_req
            )
            self.ema_accept_len = (
                1 - self.ema_alpha
            ) * self.ema_accept_len + self.ema_alpha * batch_avg

        self._batch_count += 1
        if self._batch_count <= self.warmup_batches:
            return False

        if (self._batch_count - self.warmup_batches) % self.update_interval != 0:
            return False

        return self._recompute_params()

    def _recompute_params(self) -> bool:
        """根据 EMA 重新计算步数，返回是否变化。"""
        old_steps = self.current_steps
        current_idx = self.candidate_steps.index(old_steps)
        old_idx = current_idx

        # Probe the smallest positive step after a zero-step nospec interval.
        if old_steps == 0:
            current_idx = min(current_idx + 1, len(self.candidate_steps) - 1)
            target = self.candidate_steps[current_idx]
            if target > 0 and self.ema_accept_len < 0:
                # A slot initialized at steps=0 has no draft acceptance history;
                # start the first positive-step probe from that step's neutral EMA.
                self.ema_accept_len = float(target - 1)
            return self._apply_target_steps(old_steps, target)

        # TODO: 考虑限制步数变化幅度以避免过冲
        # 降级：EMA 低于阈值时减少步数
        while current_idx > 0:
            prev_step = self.candidate_steps[current_idx - 1]
            # A zero-step candidate disables drafting. Treat zero accepted drafts
            # as low enough to reach it when it is the floor candidate.
            drop_threshold = 0.5 if prev_step == 0 else prev_step - 0.5
            drop_threshold += self.down_hysteresis
            if self.ema_accept_len <= drop_threshold:
                current_idx -= 1
            else:
                break

        moved_down = current_idx < old_idx
        if not moved_down:
            while current_idx < len(self.candidate_steps) - 1:
                current_step = self.candidate_steps[current_idx]
                rise_threshold = current_step - 0.5 + self.up_hysteresis
                if self.ema_accept_len > rise_threshold:
                    current_idx += 1
                else:
                    break

        target = self.candidate_steps[current_idx]
        # EMA 上限：只限制降级——不阻止升级，
        # 使系统可以探索更高步数并让 EMA 追赶
        if self.ceiling_coeff > 0:
            ceiling = max(1, math.ceil(self.ema_accept_len * self.ceiling_coeff))
            if target > ceiling and target <= old_steps:
                while current_idx > 0 and self.candidate_steps[current_idx] > ceiling:
                    current_idx -= 1
                target = self.candidate_steps[current_idx]

        return self._apply_target_steps(old_steps, target)

    def _apply_target_steps(self, old_steps: int, target: int) -> bool:
        if target != old_steps:
            self.current_steps = target
            log_info_on_rank0(
                logger,
                f"自适应投机参数更新: steps {old_steps} -> {target} "
                f"(ema_accept_len={self.ema_accept_len:.2f})",
            )
            return True
        return False


class AdaptiveSpeculativeParams:
    """将 ``batch_size`` 路由到正确的 per-BS 槽位。

    每个槽位是一个针对特定 batch_size 的自适应步数选择配置。

    示例::

        >>> params = AdaptiveSpeculativeParams(initial_steps=3)
        >>> params.get_steps_for_batch(batch_size=1)
        3
        >>> params.get_steps_for_batch(batch_size=32)
        1
    """

    def __init__(
        self,
        initial_steps: int,
        cfg_path: str | None = None,
    ):
        """
        输入:
            initial_steps: 初始步数
            cfg_path: 自定义配置文件路径（None 时使用默认配置）
        """
        cfg, bs_entries = _load_adaptive_config(cfg_path)
        self._bs_list: list[int] = sorted(bs_entries)
        self._slots: dict[int, AdaptiveStepSlot] = {}
        self._cuda_graph_bs: list[int] | None = None

        for bs, entry in sorted(bs_entries.items()):
            self._slots[bs] = AdaptiveStepSlot(
                initial_steps=initial_steps,
                cfg={**cfg, **entry},
            )

        first_slot = self._slots[self._bs_list[0]]
        log_info_on_rank0(
            logger,
            f"AdaptiveSpeculativeParams 初始化: "
            f"steps={first_slot.current_steps}, "
            f"candidate_steps={first_slot.candidate_steps}",
        )

    @cached_property
    def candidate_steps(self) -> list[int]:
        """所有 BS 槽位候选步数的并集。"""
        return sorted({s for p in self._slots.values() for s in p.candidate_steps})

    def set_cuda_graph_bs(self, cuda_graph_bs: list[int] | None) -> None:
        """设置 CUDA Graph 支持的 batch size 列表。

        输入:
            cuda_graph_bs: CUDA Graph 捕获的 batch size 列表，或 None
        """
        self._cuda_graph_bs = sorted(cuda_graph_bs) if cuda_graph_bs else None

    def get_steps_for_batch(self, batch_size: int) -> int:
        """获取指定 batch_size 的当前步数。

        输入:
            batch_size: 当前批次大小

        输出:
            当前应使用的草稿步数

        示例::

            >>> params.get_steps_for_batch(batch_size=1)
            3
        """
        return self._route(batch_size).current_steps

    def on_verify_complete(
        self, num_correct_drafts_per_req: list[int], batch_size: int
    ) -> int | None:
        """将验证结果馈送给匹配的 BS 槽位的 EMA。

        输入:
            num_correct_drafts_per_req: 各请求接受的草稿数列表
            batch_size: 当前批次大小

        输出:
            新步数（如果发生切换），否则 None

        示例::

            >>> new_step = params.on_verify_complete([3, 2, 4], batch_size=1)
            >>> if new_step is not None:
            ...     print(f"步数切换到 {new_step}")
        """
        params = self._route(batch_size)
        if params.update(num_correct_drafts_per_req):
            return params.current_steps
        return None

    def cuda_graph_bs_for_step(self, step: int) -> list[int] | None:
        """返回能在运行时达到指定 step 的 CUDA Graph batch size 值。

        输入:
            step: 目标步数

        输出:
            batch size 列表，或 None（CUDA Graph 禁用时）

        示例::

            >>> params.cuda_graph_bs_for_step(step=7)
            [1, 2]  # bs=1 和 bs=2 的槽位支持 step=7
        """
        if self._cuda_graph_bs is None:
            return None
        return [
            v
            for v in self._cuda_graph_bs
            if step in self._slots[self._find_closest_bs(v)].candidate_steps
        ]

    def _route(self, batch_size: int) -> AdaptiveStepSlot:
        """将 batch_size 映射到正确的槽位（经 CUDA Graph padding 后）。"""
        return self._slots[
            self._find_closest_bs(self._pad_to_cuda_graph_bs(batch_size))
        ]

    def _pad_to_cuda_graph_bs(self, batch_size: int) -> int:
        """将 batch_size 向上 pad 到最近的 CUDA Graph batch size。"""
        if self._cuda_graph_bs is None:
            return batch_size
        idx = bisect.bisect_left(self._cuda_graph_bs, batch_size)
        return (
            self._cuda_graph_bs[idx] if idx < len(self._cuda_graph_bs) else batch_size
        )

    def _find_closest_bs(self, target: int) -> int:
        """找到不超过 target 的最大 BS 槽位。"""
        idx = bisect.bisect_right(self._bs_list, target) - 1
        return self._bs_list[max(0, idx)]
