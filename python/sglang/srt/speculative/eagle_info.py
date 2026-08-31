import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.environ import envs
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.eagle_info_v2 import EagleDraftInputV2Mixin
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType

logger = logging.getLogger(__name__)


def _draft_runner_of(worker):
    """跨 worker 形状获取草稿 model_runner 的访问器。

    V2 草稿 worker（``EagleDraftWorker`` 及其子类）通过 ``draft_runner``
    暴露草稿 model_runner；对于直接运行草稿模型的 worker，回退到 ``model_runner``。

    输入:
        worker: 投机 worker 实例

    输出:
        草稿模型的 ModelRunner

    示例::

        >>> runner = _draft_runner_of(eagle_worker)
        >>> runner.model_config.spec_hidden_size
        2048
    """
    return (
        worker.draft_runner if hasattr(worker, "draft_runner") else worker.model_runner
    )


@dataclass
class EagleVerifyInput(SpecInput):
    """EAGLE 验证阶段的输入数据结构。

    包含草稿树的所有信息，供目标模型一次前向验证整棵树。

    以 bs=2、draft_token_num=4 为例（草稿树形如::

        root → [child0, child1, child2, child3]
               每个 child 可能有自己的子节点，形成树结构

    各字段含义::

        draft_token: 拍平的候选 token id，shape [bs*draft_token_num] = [8]
            tensor([464, 4062, 3105, 7586, 318, 257, 1332, 13])
        custom_mask: 树形注意力 mask（bool 拍平），每个 query 只能 attend 祖先链
        positions: 各节点绝对位置，shape [8]
            tensor([100, 101, 101, 102, 50, 51, 52, 52])
        retrieve_index: 各节点在拍平数组中的全局下标，shape [bs, draft_token_num]
        retrieve_next_token: 左孩子下标，-1 表无子
        retrieve_next_sibling: 右兄弟下标，-1 表无兄弟
        spec_steps: 草稿步数（树的深度 = spec_steps + 1）
        topk: 每层分支数
        draft_token_num: 每个请求每轮验证的 token 数（含 root）
        capture_hidden_mode: hidden state 捕获模式
        seq_lens_sum: 序列长度总和
        seq_lens_cpu: CPU 侧的序列长度
        grammar: 结构化输出约束对象（可选）
        num_tokens_per_req: 每请求 token 数（-1 自动从 draft_token_num 推断）
    """

    draft_token: torch.Tensor
    custom_mask: torch.Tensor
    positions: torch.Tensor
    retrieve_index: torch.Tensor
    retrieve_next_token: torch.Tensor
    retrieve_next_sibling: torch.Tensor
    retrieve_cum_len: torch.Tensor
    spec_steps: int
    topk: int
    draft_token_num: int
    capture_hidden_mode: CaptureHiddenMode
    seq_lens_sum: int
    seq_lens_cpu: torch.Tensor
    grammar: BaseGrammarObject = None

    # padding 用的形状信息
    num_tokens_per_req: int = -1  # -1 时自动从 draft_token_num 填充

    def __post_init__(self):
        super().__init__(SpecInputType.EAGLE_VERIFY)
        if self.num_tokens_per_req < 0:
            self.num_tokens_per_req = self.draft_token_num

    @property
    def max_tree_depth(self) -> int:
        """验证树的最大深度（含根节点）。

        约束 accept_index 的行宽。EAGLE 树的深度由草稿循环步数决定。
        其他树形的算法会覆盖此属性。

        示例::

            >>> # spec_steps=5 时
            >>> verify_input.max_tree_depth
            6  # 5 步草稿 + 1 个根节点
        """
        return self.spec_steps + 1

    @property
    def tree_topk(self) -> int:
        """传递给树验证 kernel 的分支因子；-1 表示不规则树（无固定每层分支数）。"""
        return self.topk

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        """返回 (显存预留系数, 调度预留系数)，两者均等于 draft_token_num。

        示例::

            >>> # draft_token_num=8 时
            >>> verify_input.get_spec_adjust_token_coefficient()
            (8, 8)
        """
        return self.draft_token_num, self.draft_token_num

    @classmethod
    def create_idle_input(cls, topk: int, spec_steps: int, num_verify_tokens: int):
        """创建空闲（无请求）状态的验证输入。

        输入:
            topk: 分支数
            spec_steps: 草稿步数
            num_verify_tokens: 验证 token 数

        输出:
            EagleVerifyInput，所有张量长度为 0

        示例::

            >>> idle = EagleVerifyInput.create_idle_input(topk=4, spec_steps=5, num_verify_tokens=8)
            >>> idle.draft_token.shape
            torch.Size([0])
        """
        return cls(
            draft_token=torch.empty((0,), dtype=torch.long, device="cuda"),
            custom_mask=torch.full((0,), True, dtype=torch.bool, device="cuda"),
            positions=torch.empty((0,), dtype=torch.int64, device="cuda"),
            retrieve_index=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device="cuda"
            ),
            retrieve_next_token=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device="cuda"
            ),
            retrieve_next_sibling=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device="cuda"
            ),
            retrieve_cum_len=None,
            topk=topk,
            draft_token_num=num_verify_tokens,
            spec_steps=spec_steps,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=0,
            seq_lens_cpu=torch.empty((0,), dtype=torch.int64),
        )

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        """为树形验证的这一次目标模型前向构造注意力参数。

        验证阶段本质是一次 prefill：一次性喂入 bs*draft_token_num 个 query token，
        每个 query 都要 attend 到「历史 KV + 树内祖先」。

        输入:
            req_pool_indices: 各请求在 req_to_token_pool 中的槽位号，shape [bs]
            paged_kernel_lens: 各请求验证前的 KV 长度，shape [bs]
            paged_kernel_lens_sum: 上者之和
            req_to_token: 请求槽位→KV 页索引的映射表

        输出:
            (kv_indices, cum_kv_seq_len, qo_indptr, custom_mask) 元组

        示例::

            >>> # bs=2, draft_token_num=4, paged_kernel_lens=[100,50]
            >>> kv_idx, cum_kv, qo_ptr, mask = verify_input.generate_attn_arg_prefill(
            ...     req_pool_indices, paged_kernel_lens, 150, req_to_token)
            >>> # kv_indices 长度 = 150 + 4*2 = 158
            >>> # qo_indptr = [0, 4, 8]
        """
        device = req_pool_indices.device
        batch_size = len(req_pool_indices)
        qo_indptr = torch.arange(
            0,
            (1 + batch_size) * self.draft_token_num,
            step=self.draft_token_num,
            dtype=torch.int32,
            device=device,
        )
        cum_kv_seq_len = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device=device
        )

        # 每个请求的 KV 长度加上本轮草稿树节点数
        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.empty(
            paged_kernel_lens_sum + self.draft_token_num * batch_size,
            dtype=torch.int32,
            device=device,
        )
        create_flashinfer_kv_indices_triton[(batch_size,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        mask_numel = (
            paged_kernel_lens_sum * self.draft_token_num
            + (self.draft_token_num**2) * batch_size
        )
        if self.custom_mask.numel() < mask_numel:
            # FIXME(attn): CUDA graph 下 custom mask padding 的临时修复
            self.custom_mask = torch.cat(
                [
                    self.custom_mask,
                    torch.full(
                        (mask_numel - self.custom_mask.numel(),),
                        True,
                        dtype=torch.bool,
                        device=device,
                    ),
                ],
                dim=0,
            )

        return kv_indices, cum_kv_seq_len, qo_indptr, self.custom_mask


@dataclass
class EagleDraftInput(SpecInput, EagleDraftInputV2Mixin):
    """EAGLE 草稿前向的输入数据结构。

    包含草稿模型前向所需的 top-k 概率/索引、hidden states、bonus token 等。
    注意：空闲 stub 请用 ``create_idle_input``，不要直接用构造器，
    因为 ``filter_batch`` / ``merge_batch`` 会无条件切片/拼接
    ``topk_p`` / ``topk_index`` / ``hidden_states`` / ``bonus_tokens``。

    以 bs=2、topk=4 为例::

        topk_p: shape (2, 4) — 每个请求的 top-4 概率
            tensor([[0.3, 0.2, 0.15, 0.1],
                    [0.4, 0.25, 0.1, 0.05]])
        topk_index: shape (2, 4) — 每个请求的 top-4 token id
            tensor([[464, 4062, 3105, 7586],
                    [318, 257, 1332, 13]])
        hidden_states: shape (2, hidden_size) — 每个请求一个 hidden state
            # STANDALONE 时为 None
        bonus_tokens: shape (2,) — 每个请求的 bonus token（上一轮目标模型的预测）
    """

    # shape: (b, topk)
    topk_p: torch.Tensor = None
    topk_index: torch.Tensor = None
    # shape: (b, hidden_size) — 每个请求一个 hidden，被草稿前向消费。
    # 当草稿算法不读取 hidden_states 时为 None（如 STANDALONE — 普通 LLM 草稿）。
    hidden_states: Optional[torch.Tensor] = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # 每请求的 bonus token（每条接受链末尾的 "+1" 目标预测）；
    # worker 在 extend 后将其复制到此处供下一轮草稿使用。
    bonus_tokens: torch.Tensor = None

    # shape: (b + 1,)
    kv_indptr: torch.Tensor = None
    kv_indices: torch.Tensor = None

    num_tokens_per_req: int = -1
    num_tokens_for_logprob_per_req: int = -1

    # 仅 V2 overlap worker 使用：req_pool_indices 作为缓冲槽位键。
    future_indices: Optional[torch.Tensor] = None

    def __post_init__(self):
        super().__init__(SpecInputType.EAGLE_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        """返回 (num_tokens_per_req, num_tokens_for_logprob_per_req)。

        示例::

            >>> # topk=4 时
            >>> draft_input.get_spec_adjust_token_coefficient()
            (4, 4)
        """
        return self.num_tokens_per_req, self.num_tokens_for_logprob_per_req

    @classmethod
    def hidden_size_for(cls, worker) -> Optional[int]:
        """解码阶段 ``hidden_states`` 的宽度：草稿自链输出
        （草稿模型通过 ``capture_for_decode`` 和草稿循环写回自己的最后 hidden）。
        当草稿架构不消费此字段时返回 None（如 STANDALONE）。

        输入:
            worker: 投机 worker 实例

        输出:
            hidden_size 整数或 None
        """
        if worker.speculative_algorithm.is_standalone():
            return None
        return _draft_runner_of(worker).model_config.spec_hidden_size

    @classmethod
    def dtype_for(cls, worker) -> Optional[torch.dtype]:
        """获取草稿 hidden_states 的数据类型。STANDALONE 返回 None。"""
        if worker.speculative_algorithm.is_standalone():
            return None
        return _draft_runner_of(worker).model_config.dtype

    @classmethod
    def create_idle_input(
        cls,
        device: torch.device,
        hidden_size: Optional[int],
        dtype: Optional[torch.dtype],
        topk: int,
        capture_hidden_mode: CaptureHiddenMode,
    ):
        """创建空闲（无请求）状态的草稿输入。

        输入:
            device: GPU 设备
            hidden_size: hidden state 维度（None 时不创建 hidden_states）
            dtype: 数据类型
            topk: 分支数
            capture_hidden_mode: hidden 捕获模式

        输出:
            EagleDraftInput，所有张量第一维为 0

        示例::

            >>> idle = EagleDraftInput.create_idle_input(
            ...     torch.device("cuda"), 2048, torch.float16, 4, CaptureHiddenMode.FULL)
            >>> idle.topk_p.shape
            torch.Size([0, 4])
        """
        return cls(
            bonus_tokens=torch.empty((0,), device=device, dtype=torch.int32),
            hidden_states=(
                torch.empty((0, hidden_size), device=device, dtype=dtype)
                if hidden_size is not None
                else None
            ),
            topk_p=torch.empty((0, topk), device=device, dtype=torch.float32),
            topk_index=torch.empty((0, topk), device=device, dtype=torch.int64),
            capture_hidden_mode=capture_hidden_mode,
        )

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        """请求退出 batch 时，按 ``new_indices`` 过滤各张量。

        输入:
            new_indices: 保留下来的请求在原 batch 中的下标，shape [新bs]
            has_been_filtered: 是否已被上游过滤（True 时做截断而非索引）

        示例::

            >>> # 原 bs=3，req1 已完成
            >>> new_indices = tensor([0, 2])
            >>> draft_input.filter_batch(new_indices)
            >>> # topk_p 从 (3, 4) 变为 (2, 4)
        """
        if self.future_indices is not None:
            self.future_indices = self.future_indices[new_indices]
            return

        strict_check = envs.SGLANG_SPEC_ENABLE_STRICT_FILTER_CHECK.get()
        if has_been_filtered:
            # 在 eagle_utils.py:verify 中，已经通过 `unfinished_index` 过滤了 batch，
            # 因此调度器中不需要再次过滤
            error_msg = f"new_indices 长度: {len(new_indices)} != topk_p 长度: {len(self.topk_p)}，不应发生"
            if len(new_indices) != len(self.topk_p):
                if strict_check:
                    raise ValueError(error_msg)
                else:
                    logger.warning(error_msg)

            self.topk_p = self.topk_p[: len(new_indices)]
            self.topk_index = self.topk_index[: len(new_indices)]
            if self.hidden_states is not None:
                self.hidden_states = self.hidden_states[: len(new_indices)]
            self.bonus_tokens = self.bonus_tokens[: len(new_indices)]
        else:
            # 某些情况下（如 draft_extend），尚未通过 `unfinished_index` 过滤
            self.topk_p = self.topk_p[new_indices]
            self.topk_index = self.topk_index[new_indices]
            if self.hidden_states is not None:
                self.hidden_states = self.hidden_states[new_indices]
            self.bonus_tokens = self.bonus_tokens[new_indices]

    def merge_batch(self, spec_info: "EagleDraftInput"):
        """新请求加入 batch 时，拼接另一个 EagleDraftInput 的各张量。

        输入:
            spec_info: 待并入的另一批草稿输入

        示例::

            >>> # self 有 2 个请求，spec_info 有 1 个
            >>> draft_input.merge_batch(other_input)
            >>> # topk_p 从 (2, 4) 变为 (3, 4)
        """
        if self.future_indices is not None:
            assert spec_info.future_indices is not None
            self.future_indices = torch.cat(
                [self.future_indices, spec_info.future_indices]
            )
            return

        # 通过 `topk_index` 长度检测空闲 stub（空闲输入所有字段 shape[0] == 0）。
        # 不要用 `hidden_states is None`：STANDALONE 的所有非空闲输入也是 None。
        if len(self.topk_index) == 0:
            self.hidden_states = spec_info.hidden_states
            self.bonus_tokens = spec_info.bonus_tokens
            self.topk_p = spec_info.topk_p
            self.topk_index = spec_info.topk_index
            return
        if len(spec_info.topk_index) == 0:
            return
        if self.hidden_states is not None and spec_info.hidden_states is not None:
            self.hidden_states = torch.cat(
                [self.hidden_states, spec_info.hidden_states], axis=0
            )
        self.bonus_tokens = torch.cat(
            [self.bonus_tokens, spec_info.bonus_tokens], axis=0
        )
        self.topk_p = torch.cat([self.topk_p, spec_info.topk_p])
        self.topk_index = torch.cat([self.topk_index, spec_info.topk_index])


@dataclass
class EagleDraftExtendInput(SpecInput):
    """草稿扩展前向的输入（目标 prefill/verify 后的草稿 KV 补齐阶段）。

    由 worker 的 ``_draft_extend_for_*`` 方法安装到 ``batch.spec_info`` 上
    （也被草稿扩展 CUDA Graph 捕获合成使用），然后在下一轮草稿迭代中
    被替换为新的 ``EagleDraftInput``。

    以 bs=2、num_accept_tokens=[3, 5] 为例::

        hidden_states: shape (3+5, hidden_size) = (8, 2048)
            # 或 None（STANDALONE）
        num_correct_drafts: shape (2,) = tensor([2, 4])
            # 接受的草稿 token 数（不含 bonus）
        num_accept_tokens: shape (2,) = tensor([3, 5])
            # 总接受 token 数 = num_correct_drafts + 1（含 bonus）
        input_ids: shape (8,) — 拍平的接受 token
        seq_lens: shape (2,) — 接受后的序列长度
    """

    # 目标模型的 hidden states 供草稿扩展使用；草稿不读 hidden 时为 None（如 STANDALONE）。
    # shape: decode (bs * num_draft_tokens, hidden), prefill (extend_num_tokens, hidden)
    hidden_states: Optional[torch.Tensor] = None

    # 每请求的接受计数。``num_accept_tokens = num_correct_drafts + 1``。
    # 两者都保留以供 CUDA Graph 缓冲区索引。
    num_correct_drafts: torch.Tensor = None
    num_accept_tokens: torch.Tensor = None
    # CPU 视图，扩展前向期间由注意力后端读取。
    num_accept_tokens_cpu: List[int] = None

    # 每请求的批次状态切片，用于草稿扩展前向:
    #   - input_ids:        拍平的接受 token
    #   - seq_lens / _cpu:  每请求序列长度（接受后）
    #   - req_pool_indices: 每请求的 KV 池槽位
    input_ids: torch.Tensor = None
    seq_lens: torch.Tensor = None
    seq_lens_cpu: torch.Tensor = None
    req_pool_indices: torch.Tensor = None

    #   - positions: shape ``[total_accepted]``
    #   - bonus_tokens: shape ``[bs]``；扩展后读取以填充下一轮迭代的
    #     ``EagleDraftInput.bonus_tokens``
    positions: Optional[torch.Tensor] = None
    bonus_tokens: Optional[torch.Tensor] = None

    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.LAST
    num_tokens_per_req: int = -1
    num_tokens_for_logprob_per_req: int = 1

    def __post_init__(self):
        super().__init__(SpecInputType.EAGLE_DRAFT_EXTEND)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        """返回 (num_tokens_per_req, num_tokens_for_logprob_per_req)。"""
        return self.num_tokens_per_req, self.num_tokens_for_logprob_per_req

    @classmethod
    def hidden_size_for(cls, worker) -> Optional[int]:
        """扩展阶段 ``hidden_states`` 的宽度：目标的 ``spec_hidden_size``，
        EAGLE-3 aux 模式下扩展为 ``num_aux * target_hidden``。
        当草稿架构不消费此字段时返回 None（如 STANDALONE）。

        输入:
            worker: 投机 worker 实例

        输出:
            hidden_size 整数或 None

        示例::

            >>> # 普通 EAGLE: 返回 spec_hidden_size（如 2048）
            >>> # EAGLE-3 aux 模式: 返回 target_hidden * num_aux（如 4096 * 3 = 12288）
            >>> # STANDALONE: 返回 None
        """
        if worker.speculative_algorithm.is_standalone():
            return None
        target_cfg = worker.target_worker.model_runner.model_config
        if not (
            worker.speculative_algorithm.is_eagle3()
            and worker.eagle_use_aux_hidden_state
        ):
            return target_cfg.spec_hidden_size

        hf_config = target_cfg.hf_config

        # `num_aux` 解析优先级: 显式属性 > eagle_config layer_ids > 默认 3
        num_aux = getattr(hf_config, "num_aux_hidden_states", None)
        if num_aux is None:
            eagle_config = getattr(hf_config, "eagle_config", None) or {}
            layer_ids = eagle_config.get("eagle_aux_hidden_state_layer_ids")
            num_aux = len(layer_ids) if layer_ids else 3

        target_hidden = getattr(hf_config, "target_hidden_size", target_cfg.hidden_size)
        return target_hidden * num_aux

    @classmethod
    def dtype_for(cls, worker) -> Optional[torch.dtype]:
        """获取扩展阶段 hidden_states 的数据类型。STANDALONE 返回 None。"""
        if worker.speculative_algorithm.is_standalone():
            return None
        return worker.target_worker.model_runner.model_config.dtype

    @classmethod
    def create_idle_input(
        cls,
        device: torch.device,
        hidden_size: Optional[int],
        dtype: Optional[torch.dtype],
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.LAST,
    ) -> "EagleDraftExtendInput":
        """创建空闲（无请求）状态的扩展输入。

        输入:
            device: GPU 设备
            hidden_size: hidden state 维度（None 时不创建）
            dtype: 数据类型
            capture_hidden_mode: hidden 捕获模式

        输出:
            EagleDraftExtendInput，所有张量第一维为 0
        """
        return cls(
            hidden_states=(
                torch.empty((0, hidden_size), device=device, dtype=dtype)
                if hidden_size is not None
                else None
            ),
            num_correct_drafts=torch.empty((0,), device=device, dtype=torch.int32),
            num_accept_tokens=torch.empty((0,), device=device, dtype=torch.int32),
            num_accept_tokens_cpu=[],
            input_ids=torch.empty((0,), device=device, dtype=torch.long),
            seq_lens=torch.empty((0,), device=device, dtype=torch.int64),
            seq_lens_cpu=torch.empty((0,), dtype=torch.int64),
            req_pool_indices=torch.empty((0,), device=device, dtype=torch.int64),
            capture_hidden_mode=capture_hidden_mode,
        )

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: Optional[int],
        req_to_token: torch.Tensor,
    ):
        """为草稿扩展前向构造注意力参数。

        输入:
            req_pool_indices: 各请求的 KV 池槽位，shape [bs]
            paged_kernel_lens: 各请求的 KV 长度，shape [bs]
            paged_kernel_lens_sum: KV 总长度（None 时自动计算）
            req_to_token: 请求槽位→KV 页索引映射表

        输出:
            (kv_indices, cum_kv_seq_len, qo_indptr, None) 元组
            注意：扩展阶段不需要 custom_mask，返回 None

        示例::

            >>> # bs=2, num_accept_tokens=[3, 5], paged_kernel_lens=[103, 105]
            >>> kv_idx, cum_kv, qo_ptr, _ = ext_input.generate_attn_arg_prefill(
            ...     req_pool_indices, paged_kernel_lens, 208, req_to_token)
            >>> # qo_indptr = [0, 3, 8]
            >>> # cum_kv_seq_len = [0, 103, 208]
        """
        device = req_pool_indices.device
        bs = self.num_correct_drafts.numel()
        qo_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        qo_indptr[1:] = torch.cumsum(self.num_accept_tokens, dim=0)
        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        if paged_kernel_lens_sum is None:
            paged_kernel_lens_sum = cum_kv_seq_len[-1]

        kv_indices = torch.empty(
            paged_kernel_lens_sum, dtype=torch.int32, device=device
        )

        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        return kv_indices, cum_kv_seq_len, qo_indptr, None
