import torch

from sglang.srt.sampling.penaltylib.orchestrator import _BatchedPenalizer
from sglang.srt.utils import get_compiler_backend, is_npu

_is_npu = is_npu()


# 使用 torch.compile 将此函数编译为融合 kernel，减少逐元素操作的启动开销。
# dynamic=True 表示支持动态 shape（batch 大小、序列长度会变），避免反复重新编译；
# NPU 上 torch.compile 支持不完善，故 disable=_is_npu 退化为 eager 执行。
@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def apply_scaling_penalties(logits, scaling_penalties):
    """对 logits 施加乘法型缩放惩罚（in-place 原地修改）。

    repetition_penalty 是 HuggingFace 风格的「乘法型」惩罚，与 frequency/presence
    这类「加法型」惩罚不同。对正负 logit 采用对称处理：
    - logit < 0：乘以惩罚系数。penalty > 1 时结果更负，token 更不可能被采样；
    - logit >= 0：除以惩罚系数。penalty > 1 时结果变小，同样降低被采样概率。

    之所以分正负讨论：如果对负 logit 也做除法，penalty > 1 反而会让它变大（更接近 0），
    起不到抑制作用。乘/除的对称设计保证无论 logit 正负，penalty > 1 始终抑制、
    penalty < 1 始终鼓励该 token。

    Args:
        logits: [bs, vocab_size]，待修改的 logits，原地更新。
        scaling_penalties: [bs, vocab_size]，每个 token 当前应乘/除的缩放系数，
            未出现过的 token 为 1.0（不影响）。
    """
    logits[:] = torch.where(
        logits < 0,
        logits * scaling_penalties,
        logits / scaling_penalties,
    )


class BatchedRepetitionPenalizer(_BatchedPenalizer):
    """重复惩罚器：根据 token 是否已在生成结果中出现来惩罚它。

    实现的是 HuggingFace 风格的 repetition_penalty —— 只关心某个 token「是否出现过」，
    而不关心「出现了几次」（后者是 frequency_penalty 的职责）。因此同一个 token 重复
    多次，缩放系数也只是该请求设定的固定值，不会叠加。

    惩罚以乘法形式作用于 logits（见 apply_scaling_penalties），故
    is_multiplicative = True，由 orchestrator 与其他乘法型惩罚累乘后统一施加。

    张量示例（bs=2, vocab_size=5）：
        假设两个请求的 repetition_penalty 分别为 1.2 和 1.0（第二个请求不惩罚）。

        1) _prepare() 后：
           repetition_penalties = [[1.2],      # [bs, 1]
                                   [1.0]]
           cumulated_repetition_penalties =    # [bs, vocab_size]，初始全 1.0
               [[1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0]]

        2) 某 decode step，两请求分别生成 token id = 2 和 0：
           output_ids = [2, 0]  ->  unsqueeze(1) -> [[2], [0]]
           scatter_ 把每行 output_ids 指向的列赋值为该行的惩罚系数：
           cumulated_repetition_penalties =
               [[1.0, 1.0, 1.2, 1.0, 1.0],     # 第 0 行第 2 列被写成 1.2
                [1.0, 1.0, 1.0, 1.0, 1.0]]     # 第 1 行第 0 列写成 1.0（无变化）

        3) 若下一步请求 0 再次生成 token id = 2，仍是「赋值」1.2 而非累乘，
           系数保持 1.2 不变（这正是 HF repetition_penalty「只看是否出现过」的语义）。

        4) _apply() 对 logits 施加（以请求 0 为例，logits[0] = [0.5, -0.5, 2.0, 1.0, -1.0]）：
           - 列 2 logit=2.0 >= 0 -> 2.0 / 1.2 ≈ 1.667（被抑制）
           - 其余列系数为 1.0，logit 不变
           结果 logits[0] ≈ [0.5, -0.5, 1.667, 1.0, -1.0]
           （若被惩罚位置的 logit 为负，则改用乘法，例如 -1.0 * 1.2 = -1.2，同样更不可能被采样）
    """

    # 标记为乘法型惩罚。orchestrator 据此把它与其他乘法型惩罚的系数矩阵相乘，
    # 而把加法型（frequency/presence）惩罚累加，两类分别处理。
    is_multiplicative: bool = True

    def _is_required(self) -> bool:
        # 只要 batch 中存在任一请求设置了非默认的 repetition_penalty，就需要启用本惩罚器。
        # 全部为 1.0（默认值，表示不惩罚）时返回 False，从而完全跳过张量分配与计算，零开销。
        return any(
            req.sampling_params.repetition_penalty != 1.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        # cumulated_repetition_penalties: [bs, vocab_size]，记录每个 token 当前应施加的
        # 缩放系数，初始全为 1.0（即「未出现过、不惩罚」）。随着生成推进，已出现 token 的
        # 对应位置会被写成该请求的惩罚值。
        self.cumulated_repetition_penalties = torch.ones(
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size),
            dtype=torch.float32,
            device=self.orchestrator.device,
        )
        # repetition_penalties: [bs, 1]，每个请求各自的惩罚系数，unsqueeze_ 增加一维以便
        # 后续 scatter_ 时按行广播到对应 token 位置。
        self.repetition_penalties = (
            torch.tensor(
                data=[
                    req.sampling_params.repetition_penalty
                    for req in self.orchestrator.reqs()
                ],
                dtype=torch.float32,
                device=self.orchestrator.device,
            )
        ).unsqueeze_(1)

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        # 每个 decode step 把最新生成的 token 标记为「已出现」。
        # scatter_ 按 output_ids 给出的列索引，将 cumulated 矩阵对应位置「赋值」为该请求的
        # 惩罚系数 —— 注意是赋值而非累乘，所以同一 token 重复出现，系数保持不变（HF 语义）。
        self.cumulated_repetition_penalties.scatter_(
            dim=1,
            index=output_ids.unsqueeze(1),
            src=self.repetition_penalties,
        )

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        # 非重叠模式下的直接施加路径：用累积的缩放系数原地修改 logits。
        apply_scaling_penalties(logits, self.cumulated_repetition_penalties)
        return logits

    def get_scaling_penalties(self) -> torch.Tensor:
        # 供 orchestrator 在重叠模式 / 投机解码下取出本惩罚器的缩放系数矩阵，
        # 与其他乘法型惩罚累乘后再统一施加（见 orchestrator.accumulate_scaling_penalties）。
        return self.cumulated_repetition_penalties

    def _filter(self, keep_indices: torch.Tensor):
        # 当 batch 中部分请求完成被移除时，按保留的行索引裁剪两个张量，保持与 batch 对齐。
        self.repetition_penalties = self.repetition_penalties[keep_indices]
        self.cumulated_repetition_penalties = self.cumulated_repetition_penalties[
            keep_indices
        ]

    def _merge(self, their: "BatchedRepetitionPenalizer"):
        # 两个 batch 合并时，沿 batch 维（dim=0）拼接各自的惩罚系数与累积矩阵。
        self.repetition_penalties = torch.cat(
            [self.repetition_penalties, their.repetition_penalties], dim=0
        )
        self.cumulated_repetition_penalties = torch.cat(
            [self.cumulated_repetition_penalties, their.cumulated_repetition_penalties],
            dim=0,
        )

    def _teardown(self) -> None:
        # 释放张量引用，便于 GC 及时回收显存。
        for name in ("repetition_penalties", "cumulated_repetition_penalties"):
            if hasattr(self, name):
                delattr(self, name)
