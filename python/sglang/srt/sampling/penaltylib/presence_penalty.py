import torch

from sglang.srt.sampling.penaltylib.orchestrator import _BatchedPenalizer


class BatchedPresencePenalizer(_BatchedPenalizer):
    """存在惩罚器：根据 token 是否在输出中「出现过」来惩罚它。

    与 frequency_penalty（看出现几次，惩罚随频次累加）不同，presence_penalty
    只关心 token「是否出现过」：只要出现过一次就施加一个固定惩罚，出现更多
    次也不会加重（这一点与 repetition_penalty 类似，但 presence 是加法型、
    repetition 是乘法型）。通过 scatter_（赋值）实现“出现过即打上固定惩罚”。

    惩罚以「加法」形式作用于 logits（直接减去累积惩罚），属加法型惩罚，
    故未覆盖 is_multiplicative（基类默认 False），由 orchestrator 与其他加法型
    惩罚（frequency/min_new_tokens）累加后统一施加。
    """

    def _is_required(self) -> bool:
        # 只要 batch 中存在任一请求设置了非默认的 presence_penalty（默认 0.0 表示不惩罚），
        # 就需要启用本惩罚器；全为 0.0 时返回 False，完全跳过张量分配与计算，零开销。
        return any(
            req.sampling_params.presence_penalty != 0.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        # cumulated_presence_penalties: [bs, vocab_size]，记录每个 token 当前的累积惩罚量，
        # 初始全为 0.0（加法型惩罚的「零元」，即不影响 logits）。
        self.cumulated_presence_penalties = torch.zeros(
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size),
            dtype=torch.float32,
            device=self.orchestrator.device,
        )

        # presence_penalties: [bs, 1]，每个请求各自的惩罚系数，unsqueeze_ 增加一维
        # 以便后续 scatter_ 时按行广播到对应 token 位置。
        self.presence_penalties = (
            torch.tensor(
                data=[
                    req.sampling_params.presence_penalty
                    for req in self.orchestrator.reqs()
                ],
                dtype=torch.float32,
                device=self.orchestrator.device,
            )
        ).unsqueeze_(1)

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        # 每个 decode step 把最新生成的 token 标记为「已出现」。
        # scatter_ 按 output_ids 给出的列索引，将对应位置「赋值」为该请求的惩罚系数 ——
        # 是赋值而非累加，所以同一 token 重复出现，惩罚也保持不变（“是否出现”语义）。
        self.cumulated_presence_penalties.scatter_(
            dim=1,
            index=output_ids.unsqueeze(1),
            src=self.presence_penalties,
        )

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        # 加法型惩罚：直接从 logits 中减去累积惩罚量（原地修改）。
        logits.sub_(self.cumulated_presence_penalties)

    def _filter(self, keep_indices: torch.Tensor):
        # 当 batch 中部分请求完成被移除时，按保留的行索引裁剪两个张量，保持与 batch 对齐。
        self.presence_penalties = self.presence_penalties[keep_indices]
        self.cumulated_presence_penalties = self.cumulated_presence_penalties[
            keep_indices
        ]

    def _merge(self, their: "BatchedPresencePenalizer"):
        # 两个 batch 合并时，沿 batch 维（dim=0）拼接各自的惩罚系数与累积矩阵。
        self.presence_penalties = torch.cat(
            [self.presence_penalties, their.presence_penalties], dim=0
        )
        self.cumulated_presence_penalties = torch.cat(
            [self.cumulated_presence_penalties, their.cumulated_presence_penalties],
            dim=0,
        )

    def _teardown(self) -> None:
        # 释放张量引用，便于 GC 及时回收显存。
        for name in ("presence_penalties", "cumulated_presence_penalties"):
            if hasattr(self, name):
                delattr(self, name)
