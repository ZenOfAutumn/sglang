"""中译：「新 token 比率（new token ratio）」跟踪器。

调度器在做显存预算估计时，并不知道每个请求最终会生成多少 token，因此用一个介于
(min, init] 的系数来「预估单个请求未来还会新增多少 token」（相对其声明的 max_new_tokens）。
该系数越大表示越保守（预留更多显存、更不容易因显存不足而回退/抢占）。

运行时它会随着调度步数逐步「衰减（decay）」——从 init 线性下降到 min，意味着系统越跑越乐观、
逐渐释放预留余量以提升吞吐；一旦发生回退（retract，因显存不足把请求踢回等待队列），就
reset 回 init 重新保守。estimate_new_token_ratio_after_retract 则用实际已解码情况现场重估。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


@dataclass(slots=True, kw_only=True)
class NewTokenRatioTracker:
    """中译：保存并演进 new token ratio 的当前值。

    - init：初始（最保守）比率。
    - min：衰减下限（最乐观）比率。
    - decay：每个衰减步从 current 中扣减的量（线性衰减）。
    - current：当前生效的比率，被调度器用于预算估计。
    """

    init: float
    min: float
    decay: float
    current: float

    @classmethod
    def from_server_args(cls, server_args: ServerArgs) -> NewTokenRatioTracker:
        """中译：从 ServerArgs / 环境变量构造跟踪器并算好衰减步长。"""
        # 中译：初始比率 = 基准比率 × 调度保守度（schedule_conservativeness），并夹到 <=1.0；
        #       conservativeness 越大越保守（预留越多）。
        init = min(
            envs.SGLANG_INIT_NEW_TOKEN_RATIO.get()
            * server_args.schedule_conservativeness,
            1.0,
        )
        # 中译：下限 = init × 下限系数（<1），同样夹到 <=1.0。
        min_ratio = min(
            init * envs.SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR.get(),
            1.0,
        )
        # 中译：把 (init - min) 的差额平摊到约定的衰减步数上，得到每步线性扣减量。
        decay = (init - min_ratio) / envs.SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS.get()
        return cls(init=init, min=min_ratio, decay=decay, current=init)

    def decay_step(self) -> None:
        # 中译：执行一次衰减——current 下降一个 decay，但不低于 min（越跑越乐观）。
        self.current = max(self.current - self.decay, self.min)

    def reset(self) -> None:
        # 中译：重置回最保守的 init（通常在发生 retract 后调用）。
        self.current = self.init

    @staticmethod
    def estimate_new_token_ratio_after_retract(reqs: Sequence[Req]) -> float:
        """中译：发生回退（retract）后，依据这批请求的实际进度现场重估一个新比率。

        思路：把「已解码 token 总数 + 给每个请求预留的额外解码步」当作分子，
        「声明的 max_new_tokens 总数」当作分母，得到一个更贴合现状的预估占比，并夹到 <=1.0。
        """
        # 中译：分子第一项——这批请求目前已经实际生成的 token 总数。
        total_decoded_tokens = sum(len(r.output_ids) for r in reqs)
        # 中译：分母——这批请求声明的最大新增 token 总数。
        total_max_new_tokens = sum(r.sampling_params.max_new_tokens for r in reqs)

        # 中译：分子再加上「每个请求预留 RETRACT_DECODE_STEPS 步」的余量，分母 +1 避免除零。
        new_estimate_ratio = (
            total_decoded_tokens + envs.SGLANG_RETRACT_DECODE_STEPS.get() * len(reqs)
        ) / (
            total_max_new_tokens + 1
        )  # avoid zero division
        new_estimate_ratio = min(1.0, new_estimate_ratio)
        return new_estimate_ratio
