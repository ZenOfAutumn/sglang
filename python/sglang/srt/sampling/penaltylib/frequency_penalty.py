import torch

from sglang.srt.sampling.penaltylib.orchestrator import _BatchedPenalizer


class BatchedFrequencyPenalizer(_BatchedPenalizer):
    """频率惩罚器：根据 token 在输出中出现的「频次」来惩罚它。

    与 repetition_penalty（只看是否出现过）不同，frequency_penalty 关心 token
    出现了几次：出现得越频繁，惩罚越大。这通过 scatter_add_（累加）实现 —— 每出现
    一次就把该请求的惩罚值累加一次到对应 token 上。

    惩罚以「加法」形式作用于 logits（直接减去累积惩罚），属于加法型惩罚，
    故未覆盖 is_multiplicative（基类默认 False），由 orchestrator 与其他加法型
    惩罚（presence/min_new_tokens）累加后统一施加。
    """

    def _is_required(self) -> bool:
        # 只要 batch 中存在任一请求设置了非默认的 frequency_penalty（默认 0.0 表示不惩罚），
        # 就需要启用本惩罚器；全为 0.0 时返回 False，完全跳过张量分配与计算，零开销。
        return any(
            req.sampling_params.frequency_penalty != 0.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        # cumulated_frequency_penalties: [bs, vocab_size]，记录每个 token 当前累积的惩罚量，
        # 初始全为 0.0（加法型惩罚的「零元」，即不影响 logits）。
        self.cumulated_frequency_penalties = torch.zeros(
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size),
            dtype=torch.float32,
            device=self.orchestrator.device,
        )

        # frequency_penalties: [bs, 1]，每个请求各自的单次惩罚系数，unsqueeze_ 增加一维
        # 以便后续 scatter_add_ 时按行广播到对应 token 位置。
        self.frequency_penalties = (
            torch.tensor(
                data=[
                    req.sampling_params.frequency_penalty
                    for req in self.orchestrator.reqs()
                ],
                dtype=torch.float32,
                device=self.orchestrator.device,
            )
        ).unsqueeze_(1)

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        # 每个 decode step 把最新生成的 token 计入频次。
        # scatter_add_ 按 output_ids 给出的列索引，将该请求的惩罚值「累加」到对应位置 ——
        # 注意是累加而非赋值，所以同一 token 出现越多次，累积惩罚越大（频次语义）。
        self.cumulated_frequency_penalties.scatter_add_(
            dim=1,
            index=output_ids.unsqueeze(1),
            src=self.frequency_penalties,
        )

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        # 加法型惩罚：直接从 logits 中减去累积惩罚量（原地修改）。
        # 惩罚值越大，对应 token 的 logit 越小，越不容易被采样。
        logits.sub_(self.cumulated_frequency_penalties)

    def _filter(self, keep_indices: torch.Tensor):
        # 当 batch 中部分请求完成被移除时，按保留的行索引裁剪两个张量，保持与 batch 对齐。
        self.frequency_penalties = self.frequency_penalties[keep_indices]
        self.cumulated_frequency_penalties = self.cumulated_frequency_penalties[
            keep_indices
        ]

    def _merge(self, their: "BatchedFrequencyPenalizer"):
        # 两个 batch 合并时，沿 batch 维（dim=0）拼接各自的惩罚系数与累积矩阵。
        self.frequency_penalties = torch.cat(
            [self.frequency_penalties, their.frequency_penalties], dim=0
        )
        self.cumulated_frequency_penalties = torch.cat(
            [self.cumulated_frequency_penalties, their.cumulated_frequency_penalties],
            dim=0,
        )

    def _teardown(self) -> None:
        # 释放张量引用，便于 GC 及时回收显存。
        for name in ("frequency_penalties", "cumulated_frequency_penalties"):
            if hasattr(self, name):
                delattr(self, name)
