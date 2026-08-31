"""DFLASH 投机解码工具函数模块。

本模块提供 DFLASH 算法的配置解析、KV cache 预算计算、验证采样、
草稿层选择等工具函数，被 dflash_worker_v2.py 等模块调用。

主要功能：
    - 配置解析：parse_dflash_draft_config / DFlashDraftConfig
    - KV cache 预算：scale_kv_cell_size_per_token_for_dflash
    - 验证采样：compute_dflash_correct_drafts_and_bonus / compute_dflash_sampling_correct_drafts_and_bonus
    - 草稿层选择：build_target_layer_ids
    - 请求校验：validate_dflash_request
    - logit 调整：apply_dflash_verify_logits_adjustments
    - prefill 批处理延迟：should_delay_dflash_prefill_for_batching
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.utils import is_cuda, is_musa

# DFLASH 默认的 mask token 字符串
DEFAULT_DFLASH_MASK_TOKEN = "<|MASK|>"

logger = logging.getLogger(__name__)

# 是否可以使用 sgl_kernel 的采样验证（仅 CUDA/MUSA）
_DFLASH_SAMPLING_VERIFY_AVAILABLE = False
# 缓存链式验证所需的 buffer，按 (device_index, draft_token_num) 索引
_DFLASH_CHAIN_VERIFY_BUFFERS: dict[tuple[Optional[int], int], dict[str, Any]] = {}
# 这些注意力后端不需要 custom_mask
_DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS = frozenset(
    {
        "FlashInferAttnBackend",
        "FlashInferMLAAttnBackend",
        "FlashAttentionBackend",
        "TritonAttnBackend",
        "TRTLLMHAAttnBackend",
        "TRTLLMMLABackend",
    }
)


if is_cuda() or is_musa():
    try:
        from sgl_kernel import (
            top_k_renorm_prob,
            top_p_renorm_prob,
            tree_speculative_sampling_target_only,
        )

        _DFLASH_SAMPLING_VERIFY_AVAILABLE = True
    except Exception:
        top_k_renorm_prob = None
        top_p_renorm_prob = None
        tree_speculative_sampling_target_only = None
else:
    top_k_renorm_prob = None
    top_p_renorm_prob = None
    tree_speculative_sampling_target_only = None


def is_dflash_sampling_verify_available() -> bool:
    """检查当前环境是否支持 DFLASH 非贪心采样验证。

    返回：True 表示 sgl_kernel 的采样验证 kernel 可用
    """
    return _DFLASH_SAMPLING_VERIFY_AVAILABLE


def resolve_dflash_prefill_refill_target(max_running_requests: int) -> int:
    """确定 DFlash prefill 前等待多少个空闲请求槽位再填充。

    小 batch 时逐个填充，大 batch 时批量填充以提高吞吐。
    可通过 SGLANG_DFLASH_PREFILL_REFILL_TARGET 环境变量覆盖。

    参数：
        max_running_requests: 最大并发请求数

    返回：prefill 填充目标请求数

    示例：
        >>> resolve_dflash_prefill_refill_target(4)   # 小 batch -> 1
        1
        >>> resolve_dflash_prefill_refill_target(32)  # 大 batch -> 4
        4
    """
    override = envs.SGLANG_DFLASH_PREFILL_REFILL_TARGET.get()
    if override is not None:
        return override

    max_running_requests = max(0, int(max_running_requests))
    if max_running_requests < 8:
        return 1
    return min(4, max(2, (max_running_requests + 5) // 6))


def should_delay_dflash_prefill_for_batching(
    *,
    running_bs: int,
    num_allocatable_reqs: int,
    max_running_requests: int,
    prefill_refill_target: int,
) -> bool:
    """判断是否应延迟 DFlash prefill 以积累更多请求进行批处理。

    参数：
        running_bs:            当前运行中的 batch size
        num_allocatable_reqs:  可分配的请求数
        max_running_requests:  最大并发请求数
        prefill_refill_target: prefill 填充目标

    返回：True 表示应延迟 prefill

    示例：
        >>> should_delay_dflash_prefill_for_batching(
        ...     running_bs=10, num_allocatable_reqs=1, max_running_requests=32,
        ...     prefill_refill_target=4)
        True  # 只有 1 个可分配请求，少于目标 4，应延迟
    """
    if running_bs <= 0:
        return False

    target_prefill_bs = int(prefill_refill_target)
    if target_prefill_bs <= 1:
        return False

    target_prefill_bs = min(target_prefill_bs, int(max_running_requests))
    return int(num_allocatable_reqs) < target_prefill_bs


def scale_kv_cell_size_per_token_for_dflash(
    *,
    target_cell_size_per_token: int,
    target_num_layers: int,
    draft_num_layers: int,
    draft_cell_size_per_token: Optional[int] = None,
) -> int:
    """计算 DFLASH 目标+草稿 KV 池合并后的每 token 字节预算。

    DFLASH 运行独立的草稿 runner，拥有自己的 KV 池。目标 runner 的
    token 容量必须容纳两个池的总量。

    参数：
        target_cell_size_per_token: 目标模型每 token KV 字节数
        target_num_layers:          目标模型层数
        draft_num_layers:           草稿模型层数
        draft_cell_size_per_token:  草稿模型每 token KV 字节数（可选）

    返回：合并后的每 token 字节预算

    示例：
        >>> # 目标 32 层 × 128B/token，草稿 4 层
        >>> scale_kv_cell_size_per_token_for_dflash(
        ...     target_cell_size_per_token=128, target_num_layers=32,
        ...     draft_num_layers=4)
        144  # 128 * 36 / 32 = 144
    """
    if target_cell_size_per_token <= 0:
        raise ValueError(
            "target_cell_size_per_token must be positive, "
            f"got {target_cell_size_per_token}."
        )

    if draft_cell_size_per_token is not None:
        draft_cell_size_per_token = int(draft_cell_size_per_token)
        if draft_cell_size_per_token <= 0:
            raise ValueError(
                "draft_cell_size_per_token must be positive when provided, "
                f"got {draft_cell_size_per_token}."
            )
        return int(target_cell_size_per_token) + int(draft_cell_size_per_token)

    if target_num_layers <= 0 or draft_num_layers <= 0:
        return int(target_cell_size_per_token)

    total_layers = int(target_num_layers) + int(draft_num_layers)
    return (
        int(target_cell_size_per_token) * int(total_layers) + int(target_num_layers) - 1
    ) // int(target_num_layers)


def resolve_dflash_verify_mask_policy(attn_backend: Any) -> tuple[str, bool]:
    """确定 DFLASH 验证时是否需要 custom mask。

    遍历 attn_backend 的 full_attn_backend 链（最多 4 层），
    判断后端是否在跳过列表中。

    参数：
        attn_backend: 注意力后端实例

    返回：(backend_name, needs_custom_mask)
        - backend_name:       后端类名
        - needs_custom_mask:  True 表示需要 custom mask
    """
    backend = attn_backend
    for _ in range(4):
        full_backend = getattr(backend, "full_attn_backend", None)
        if full_backend is None:
            break
        backend = full_backend
    backend_name = type(backend).__name__
    return backend_name, (backend_name not in _DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS)


def apply_dflash_verify_logits_adjustments(
    *,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    draft_token_num: int,
) -> None:
    """对 DFlash 验证的 logits 原地应用采样时调整。

    保持 v1 和 v2 验证语义一致，同时让 overlap 调度使用更便宜的
    预计算 acc_linear_penalties 路径，避免每步分配
    [bs * draft_token_num, vocab] 的惩罚张量。

    参数：
        next_token_logits: 验证 logits [bs * draft_token_num, vocab]
        sampling_info:     采样信息对象
        draft_token_num:   每请求草稿 token 数
    """
    if sampling_info is None:
        return
    if next_token_logits.ndim != 2:
        raise ValueError(
            "next_token_logits must be 2D, "
            f"got shape={tuple(next_token_logits.shape)}."
        )
    if draft_token_num <= 0:
        raise ValueError(f"draft_token_num must be positive, got {draft_token_num}.")

    bs = len(sampling_info)
    if next_token_logits.shape[0] != bs * draft_token_num:
        raise ValueError(
            "next_token_logits row count mismatch for DFlash verify adjustments. "
            f"Expected {bs * draft_token_num}, got {next_token_logits.shape[0]}."
        )

    if sampling_info.has_custom_logit_processor:
        apply_custom_logit_processor(
            next_token_logits,
            sampling_info,
            num_tokens_in_batch=draft_token_num,
        )

    acc_linear_penalties = getattr(sampling_info, "acc_linear_penalties", None)
    penalizer = getattr(sampling_info, "penalizer_orchestrator", None)
    vocab_mask = getattr(sampling_info, "vocab_mask", None)
    logit_bias = getattr(sampling_info, "logit_bias", None)

    logits_3d: Optional[torch.Tensor] = None

    def get_logits_3d() -> torch.Tensor:
        nonlocal logits_3d
        if logits_3d is None:
            logits_3d = next_token_logits.reshape(bs, draft_token_num, -1)
        return logits_3d

    # 仅在需要实时 penalizer 应用或 vocab mask 时使用密集回退。
    # overlap 调度中常见路径是 acc_linear_penalties，可以直接在验证 block
    # 上广播，无需物化重复缓冲区。
    if (
        penalizer is not None and penalizer.is_required and acc_linear_penalties is None
    ) or vocab_mask is not None:
        linear_penalty = torch.zeros(
            (bs, next_token_logits.shape[1]),
            dtype=torch.float32,
            device=next_token_logits.device,
        )
        sampling_info.apply_logits_bias(linear_penalty)
        get_logits_3d().add_(
            linear_penalty[:, None, :].to(dtype=next_token_logits.dtype)
        )
        return

    if acc_linear_penalties is not None:
        if (
            acc_linear_penalties.device != next_token_logits.device
            or acc_linear_penalties.dtype != next_token_logits.dtype
        ):
            acc_linear_penalties = acc_linear_penalties.to(
                device=next_token_logits.device,
                dtype=next_token_logits.dtype,
            )
        get_logits_3d().add_(acc_linear_penalties[:, None, :])

    if logit_bias is not None:
        if (
            logit_bias.device != next_token_logits.device
            or logit_bias.dtype != next_token_logits.dtype
        ):
            logit_bias = logit_bias.to(
                device=next_token_logits.device,
                dtype=next_token_logits.dtype,
            )
        get_logits_3d().add_(logit_bias[:, None, :])


def _get_or_create_chain_verify_buffers(
    *,
    bs: int,
    draft_token_num: int,
    device: torch.device,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """获取或创建链式验证所需的缓冲区（带缓存）。

    按 (device_index, draft_token_num) 缓存 buffer，按需增长容量。
    DFLASH 验证是线性的（topk=1），所以 retrieve_next_sibling 全为 -1。

    参数：
        bs:               batch size
        draft_token_num:  每请求草稿 token 数
        device:           目标设备

    返回：(retrieve_index, retrieve_next_token, retrieve_next_sibling,
          predicts, accept_index, accept_token_num)
    """
    key = (device.index, int(draft_token_num))
    cached = _DFLASH_CHAIN_VERIFY_BUFFERS.get(key)
    cap_bs = 0 if cached is None else int(cached["cap_bs"])
    if cap_bs < bs:
        new_cap = max(int(bs), cap_bs * 2 if cap_bs > 0 else int(bs))
        retrieve_index = torch.arange(
            new_cap * draft_token_num, dtype=torch.int64, device=device
        ).view(new_cap, draft_token_num)
        row_next = torch.arange(
            1, draft_token_num + 1, dtype=torch.int64, device=device
        )
        row_next[-1] = -1
        retrieve_next_token = row_next.unsqueeze(0).expand(new_cap, -1).clone()
        retrieve_next_sibling = torch.full(
            (new_cap, draft_token_num), -1, dtype=torch.int64, device=device
        )
        predicts = torch.empty(
            (new_cap * draft_token_num,), dtype=torch.int32, device=device
        )
        accept_index = torch.empty(
            (new_cap, draft_token_num), dtype=torch.int32, device=device
        )
        accept_token_num = torch.empty((new_cap,), dtype=torch.int32, device=device)
        cached = {
            "cap_bs": int(new_cap),
            "retrieve_index": retrieve_index,
            "retrieve_next_token": retrieve_next_token,
            "retrieve_next_sibling": retrieve_next_sibling,
            "predicts": predicts,
            "accept_index": accept_index,
            "accept_token_num": accept_token_num,
        }
        _DFLASH_CHAIN_VERIFY_BUFFERS[key] = cached

    assert cached is not None
    retrieve_index = cached["retrieve_index"][:bs]
    retrieve_next_token = cached["retrieve_next_token"][:bs]
    retrieve_next_sibling = cached["retrieve_next_sibling"][:bs]
    predicts = cached["predicts"][: bs * draft_token_num]
    accept_index = cached["accept_index"][:bs]
    accept_token_num = cached["accept_token_num"][:bs]
    return (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        predicts,
        accept_index,
        accept_token_num,
    )


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> List[int]:
    """选择用于构建 DFlash 上下文特征的目标层索引。

    DFlash 使用每个选定目标层之后的隐藏状态（HF 风格）。
    SGLang 捕获「layer i 之前」，所以 model hook 通常 +1 映射到捕获点。

    参数：
        num_target_layers: 目标模型 transformer 层数
        num_draft_layers:  草稿模型层数

    返回：长度为 num_draft_layers 的 0-based 目标层索引列表

    示例：
        >>> build_target_layer_ids(32, 4)  # 32 层目标，4 层草稿
        [1, 11, 21, 28]  # 均匀分布在 [1, 28] 范围内
    """
    if num_target_layers <= 0:
        raise ValueError(
            f"num_target_layers must be positive, got {num_target_layers}."
        )
    if num_draft_layers <= 0:
        raise ValueError(f"num_draft_layers must be positive, got {num_draft_layers}.")

    if num_draft_layers == 1:
        return [num_target_layers // 2]

    start = 1
    end = num_target_layers - 3
    if end < start:
        raise ValueError(
            "DFlash layer selection requires num_target_layers >= 4. "
            f"Got num_target_layers={num_target_layers}."
        )

    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def get_dflash_layer_types(config: Any) -> Optional[Sequence[str]]:
    """从模型配置中获取 layer_types（如 attention / sliding_attention）。

    参数：
        config: 模型配置对象或字典

    返回：layer_types 序列或 None
    """
    text_config = _get_text_config(config)
    layer_types = _cfg_get(text_config, "layer_types", _cfg_get(config, "layer_types"))
    if layer_types is None:
        return None
    if isinstance(layer_types, str) or not isinstance(layer_types, Sequence):
        raise ValueError(
            "DFLASH config.layer_types must be a sequence of attention type strings."
        )
    return layer_types


def get_dflash_attention_sliding_window_size(config: Any) -> Optional[int]:
    """获取 DFLASH 滑动注意力窗口大小。

    HF 的 sliding_window 包含当前 token，SGLang 存储 window_left，
    所以返回 sliding_window - 1。

    参数：
        config: 模型配置

    返回：窗口大小（int）或 None（无滑动注意力层时）
    """
    layer_types = get_dflash_layer_types(config)
    if layer_types is None or "sliding_attention" not in layer_types:
        return None

    text_config = _get_text_config(config)
    sliding_window = _cfg_get(
        text_config, "sliding_window", _cfg_get(config, "sliding_window")
    )
    if sliding_window is None:
        raise ValueError(
            "DFLASH sliding_attention layers require config.sliding_window."
        )

    # HF sliding windows include the current token; SGLang stores window_left.
    return int(sliding_window) - 1


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    """兼容 dict 和对象的配置字段获取。"""
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _get_text_config(config: Any) -> Any:
    """从模型配置中提取 text_config（处理多模态模型的嵌套配置）。"""
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get("text_config", config)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return text_config
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            resolved = get_text_config()
            if resolved is not None:
                return resolved
        except TypeError:
            pass
    return config


def _get_dflash_config(config: Any) -> dict:
    """从模型配置中提取 dflash_config 子配置。"""
    if isinstance(config, dict):
        cfg = config.get("dflash_config", None)
    else:
        cfg = getattr(config, "dflash_config", None)
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg

    try:
        return dict(cfg)
    except Exception:
        return {}


def _parse_optional_int(
    value: Any,
    *,
    field_name: str,
    min_value: Optional[int] = None,
) -> Optional[int]:
    """解析可选的整数值，带范围校验。"""
    if value is None:
        return None
    try:
        parsed = int(value)
    except Exception as e:
        raise ValueError(f"Invalid {field_name}={value!r}.") from e
    if min_value is not None and parsed < int(min_value):
        comparator = "positive" if int(min_value) == 1 else f">= {int(min_value)}"
        raise ValueError(f"{field_name} must be {comparator}, got {parsed}.")
    return parsed


@dataclass(frozen=True)
class DFlashDraftConfig:
    """DFLASH 草稿模型配置。

    属性：
        num_hidden_layers:   草稿模型隐藏层数
        num_target_layers:   目标模型层数
        block_size:          DFLASH 验证 block 大小
        target_layer_ids:    选用的目标层索引列表（可选，未配置时自动均匀选择）
        mask_token:          mask token 字符串，默认 "<|MASK|>"
        mask_token_id:       mask token 的 id（可选）

    示例：
        >>> cfg = DFlashDraftConfig(
        ...     num_hidden_layers=4, num_target_layers=32, block_size=5,
        ...     target_layer_ids=None, mask_token="<|MASK|>", mask_token_id=None)
        >>> cfg.resolve_target_layer_ids(target_num_layers=32)
        [1, 11, 21, 28]
        >>> cfg.resolve_block_size(default=4)
        5
    """

    num_hidden_layers: Optional[int]
    num_target_layers: Optional[int]
    block_size: Optional[int]
    target_layer_ids: Optional[List[int]]
    mask_token: str
    mask_token_id: Optional[int]

    def require_num_layers(self) -> int:
        """返回草稿模型层数，如果未配置则报错。"""
        if self.num_hidden_layers is None:
            raise ValueError(
                "DFLASH requires draft num_hidden_layers in config. "
                "Got config without num_hidden_layers."
            )
        return int(self.num_hidden_layers)

    def resolve_block_size(self, *, default: Optional[int] = None) -> Optional[int]:
        """解析 block_size，使用配置值或默认值。"""
        return self.block_size if self.block_size is not None else default

    def resolve_target_layer_ids(
        self,
        *,
        target_num_layers: int,
        draft_num_layers: Optional[int] = None,
    ) -> List[int]:
        """解析目标层索引，未配置时自动均匀选择。

        参数：
            target_num_layers: 目标模型层数
            draft_num_layers:  草稿模型层数（可选，默认从配置获取）

        返回：目标层索引列表
        """
        target_num_layers = int(target_num_layers)
        if target_num_layers <= 0:
            raise ValueError(
                f"target_num_layers must be positive, got {target_num_layers}."
            )

        if self.target_layer_ids is None:
            if draft_num_layers is None:
                draft_num_layers = self.require_num_layers()
            return build_target_layer_ids(target_num_layers, int(draft_num_layers))

        resolved = list(self.target_layer_ids)
        if len(resolved) <= 0:
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be non-empty. "
                f"Got len(target_layer_ids)={len(resolved)}."
            )
        for idx, val in enumerate(resolved):
            if val < 0 or val >= target_num_layers:
                raise ValueError(
                    "DFLASH target_layer_ids contains an out-of-range layer id. "
                    f"target_layer_ids[{idx}]={val}, target_num_layers={target_num_layers}."
                )
        return resolved


def parse_dflash_draft_config(*, draft_hf_config: Any) -> DFlashDraftConfig:
    """从 HF 配置/字典解析并校验 DFLASH 草稿配置字段。

    参数：
        draft_hf_config: 草稿模型的 HF 配置对象或字典

    返回：DFlashDraftConfig 实例
    """
    dflash_cfg = _get_dflash_config(draft_hf_config)
    draft_text_config = _get_text_config(draft_hf_config)

    num_hidden_layers = _parse_optional_int(
        _cfg_get(draft_text_config, "num_hidden_layers", None),
        field_name="DFLASH draft num_hidden_layers",
        min_value=1,
    )
    raw_num_target_layers = dflash_cfg.get(
        "num_target_layers",
        _cfg_get(draft_hf_config, "num_target_layers", None),
    )
    num_target_layers = _parse_optional_int(
        raw_num_target_layers,
        field_name="DFLASH draft num_target_layers",
        min_value=1,
    )

    # 保留对当前 checkpoint 中 block_size 在顶层时的支持
    raw_block_size = dflash_cfg.get(
        "block_size",
        _cfg_get(draft_hf_config, "block_size", None),
    )
    block_size = _parse_optional_int(
        raw_block_size,
        field_name="DFLASH block_size",
        min_value=1,
    )

    layer_ids = dflash_cfg.get(
        "target_layer_ids",
        _cfg_get(draft_hf_config, "target_layer_ids", None),
    )
    parsed_target_layer_ids: Optional[List[int]]
    if layer_ids is None:
        parsed_target_layer_ids = None
    else:
        if not isinstance(layer_ids, (list, tuple)):
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be a list of ints, "
                f"got type={type(layer_ids).__name__}."
            )
        parsed_target_layer_ids = [int(x) for x in layer_ids]
        if len(parsed_target_layer_ids) <= 0:
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be non-empty. "
                f"Got len(target_layer_ids)={len(parsed_target_layer_ids)}."
            )

    mask_token = dflash_cfg.get("mask_token", None)
    if mask_token is None:
        mask_token = DEFAULT_DFLASH_MASK_TOKEN
    if not isinstance(mask_token, str) or not mask_token:
        raise ValueError(
            "DFLASH dflash_config.mask_token must be a non-empty string, "
            f"got {mask_token!r}."
        )

    mask_token_id = dflash_cfg.get("mask_token_id", None)
    if mask_token_id is not None:
        if not isinstance(mask_token_id, Integral) or isinstance(mask_token_id, bool):
            raise ValueError(
                "DFLASH dflash_config.mask_token_id must be an integer, "
                f"got {mask_token_id!r} (type={type(mask_token_id).__name__})."
            )
        mask_token_id = int(mask_token_id)
        if mask_token_id < 0:
            raise ValueError(
                "DFLASH dflash_config.mask_token_id must be non-negative, "
                f"got {mask_token_id}."
            )

    return DFlashDraftConfig(
        num_hidden_layers=num_hidden_layers,
        num_target_layers=num_target_layers,
        block_size=block_size,
        target_layer_ids=parsed_target_layer_ids,
        mask_token=mask_token,
        mask_token_id=mask_token_id,
    )


def can_dflash_slice_qkv_weight(qkv_proj: Any) -> Tuple[bool, str]:
    """检查 DFlash 是否可以从融合 QKV 线性层中切分 KV 权重。

    参数：
        qkv_proj: QKV 投影层对象

    返回：(can_slice, reason) can_slice 为 True 时 reason 为空
    """
    quant_method = getattr(qkv_proj, "quant_method", None)
    if not isinstance(quant_method, UnquantizedLinearMethod):
        return (
            False,
            "quantized qkv_proj is not supported for this path "
            f"(quant_method={type(quant_method).__name__})",
        )
    if not hasattr(qkv_proj, "weight"):
        return False, "qkv weight tensor is missing"
    return True, ""


def can_dflash_use_fused_qkv_proj(qkv_proj: Any) -> Tuple[bool, str]:
    """检查 QKV 层是否适用于 DFlash 融合 KV 物化。

    参数：
        qkv_proj: QKV 投影层对象

    返回：(eligible, reason) eligible 为 True 时 reason 为空
    """
    eligible, reason = can_dflash_slice_qkv_weight(qkv_proj)
    if not eligible:
        return False, reason
    if getattr(qkv_proj, "bias", None) is not None:
        return False, "qkv bias is not supported for fused KV path"
    return True, ""


def compute_dflash_correct_drafts_and_bonus(
    *,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算 DFlash 贪心验证的接受长度和 bonus token。

    规则：从 candidates[:, 1:] 与 target_predict[:, :-1] 逐位比较，
    连续匹配的长度即为接受长度。bonus 为 target_predict[correct_len]。

    参数：
        candidates:      草稿提议的 token id [bs, block_size]，
                         candidates[:, 0] 是当前 token
        target_predict:  目标模型预测的 token id [bs, block_size]

    返回：(correct_len, bonus)
        - correct_len: int32 [bs]，接受的草稿 token 数（不含当前 token 和 bonus）
        - bonus:      int64 [bs]，bonus token

    示例：
        >>> candidates = torch.tensor([[5, 10, 20, 30]])  # 草稿提议
        >>> target = torch.tensor([[10, 20, 25, 35]])      # 目标预测
        >>> correct, bonus = compute_dflash_correct_drafts_and_bonus(
        ...     candidates=candidates, target_predict=target)
        >>> # correct = tensor([2], dtype=int32)  # 10==10, 20==20, 30!=25
        >>> # bonus = tensor([25])  # target_predict[0, 2] = 25
    """
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape={tuple(candidates.shape)}")
    if target_predict.shape != candidates.shape:
        raise ValueError(
            "target_predict must have the same shape as candidates. "
            f"candidates.shape={tuple(candidates.shape)}, target_predict.shape={tuple(target_predict.shape)}"
        )

    bs, block_size = candidates.shape
    if bs <= 0:
        raise ValueError(f"batch size must be positive, got {bs}.")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    matches = candidates[:, 1:] == target_predict[:, :-1]
    correct_len = matches.to(torch.int32).cumprod(dim=1).sum(dim=1)
    bonus = target_predict[torch.arange(bs, device=target_predict.device), correct_len]
    return correct_len, bonus.to(torch.int64)


def compute_dflash_sampling_correct_drafts_and_bonus(
    *,
    candidates: torch.Tensor,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    max_top_k: Optional[int] = None,
    uniform_top_k_value: Optional[int] = None,
    threshold_single: Optional[float] = None,
    threshold_acc: Optional[float] = None,
    uniform_samples: Optional[torch.Tensor] = None,
    uniform_samples_for_final_sampling: Optional[torch.Tensor] = None,
    use_sparse_topk: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算 DFlash 非贪心采样验证的接受长度和 bonus token。

    这是投机解码 target-only 验证的链式特化变体：
      - DFlash 提议是线性的（topk == 1），每个验证层最多一个候选
      - 当候选在某层被拒绝时，最终 token 从 relu(q - p) 采样，
        其中 p 只包含被拒候选的质量

    参数：
        candidates:      草稿提议的 token id [bs, draft_token_num]
        next_token_logits: 目标模型 logits [bs * draft_token_num, vocab]
        sampling_info:   采样信息对象
        max_top_k:       最大 top_k 值
        uniform_top_k_value: 统一 top_k 值
        threshold_single: 单步接受阈值
        threshold_acc:    累积接受阈值
        uniform_samples:  均匀随机采样 [bs, draft_token_num]
        uniform_samples_for_final_sampling: 最终采样随机数 [bs]
        use_sparse_topk:  是否使用稀疏 topk 优化

    返回：(correct_len, bonus)
        - correct_len: int32 [bs]
        - bonus:      int64 [bs]
    """
    if not _DFLASH_SAMPLING_VERIFY_AVAILABLE:
        raise RuntimeError(
            "DFLASH non-greedy verification is unavailable on this build/device."
        )
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape={tuple(candidates.shape)}")
    if next_token_logits.ndim != 2:
        raise ValueError(
            "next_token_logits must be 2D, "
            f"got shape={tuple(next_token_logits.shape)}."
        )

    bs, draft_token_num = candidates.shape
    if bs <= 0:
        raise ValueError(f"batch size must be positive, got {bs}.")
    if draft_token_num <= 0:
        raise ValueError(f"draft_token_num must be positive, got {draft_token_num}.")
    if next_token_logits.shape[0] != bs * draft_token_num:
        raise ValueError(
            "next_token_logits row count mismatch. "
            f"Expected {bs * draft_token_num}, got {next_token_logits.shape[0]}."
        )
    if candidates.device != next_token_logits.device:
        raise ValueError(
            "candidates and next_token_logits must be on the same device, "
            f"got {candidates.device} and {next_token_logits.device}."
        )

    if threshold_single is None:
        from sglang.srt.server_args import get_global_server_args

        threshold_single = get_global_server_args().speculative_accept_threshold_single
    if threshold_acc is None:
        from sglang.srt.server_args import get_global_server_args

        threshold_acc = get_global_server_args().speculative_accept_threshold_acc
    threshold_single = float(threshold_single)
    threshold_acc = max(float(threshold_acc), 1e-9)

    device = next_token_logits.device

    if uniform_samples is None:
        uniform_samples = torch.rand(
            (bs, draft_token_num), dtype=torch.float32, device=device
        )
    else:
        if uniform_samples.shape != (bs, draft_token_num):
            raise ValueError(
                "uniform_samples shape mismatch. "
                f"Expected {(bs, draft_token_num)}, got {tuple(uniform_samples.shape)}."
            )
        uniform_samples = uniform_samples.to(device=device, dtype=torch.float32)

    if uniform_samples_for_final_sampling is None:
        uniform_samples_for_final_sampling = torch.rand(
            (bs,), dtype=torch.float32, device=device
        )
    else:
        if uniform_samples_for_final_sampling.shape != (bs,):
            raise ValueError(
                "uniform_samples_for_final_sampling shape mismatch. "
                f"Expected {(bs,)}, got {tuple(uniform_samples_for_final_sampling.shape)}."
            )
        uniform_samples_for_final_sampling = uniform_samples_for_final_sampling.to(
            device=device,
            dtype=torch.float32,
        )

    need_top_k = bool(getattr(sampling_info, "need_top_k_sampling", True))
    need_top_p = bool(getattr(sampling_info, "need_top_p_sampling", False))
    # 构建所有验证行的目标分布
    expanded_temperature = torch.repeat_interleave(
        sampling_info.temperatures, draft_token_num, dim=0
    )
    scaled_logits = next_token_logits / expanded_temperature
    sparse_topk_applied = False

    if use_sparse_topk and need_top_k:
        repeated_top_ks = torch.repeat_interleave(
            sampling_info.top_ks, draft_token_num, dim=0
        ).to(dtype=torch.int64)
        vocab_size = int(scaled_logits.shape[-1])
        repeated_top_ks.clamp_(min=1, max=vocab_size)
        if max_top_k is None:
            max_top_k = int(repeated_top_ks.max().item())
        else:
            max_top_k = int(max_top_k)
        if max_top_k < 1:
            max_top_k = 1
        elif max_top_k > vocab_size:
            max_top_k = vocab_size

        # top-k/top-p 的稀疏精确路径（top-k-first 语义），然后 scatter 到密集
        if 0 < max_top_k < vocab_size:
            topk_logits, topk_indices = torch.topk(scaled_logits, k=max_top_k, dim=-1)
            if uniform_top_k_value is None or int(uniform_top_k_value) != max_top_k:
                ranks = torch.arange(max_top_k, device=device, dtype=torch.int64)[
                    None, :
                ]
                valid = ranks < repeated_top_ks.unsqueeze(1)
                topk_logits = topk_logits.masked_fill(~valid, float("-inf"))

            topk_probs = F.softmax(topk_logits, dim=-1)
            if need_top_p:
                repeated_top_ps = torch.repeat_interleave(
                    sampling_info.top_ps, draft_token_num, dim=0
                )
                topk_probs = top_p_renorm_prob(topk_probs, repeated_top_ps)

            target_probs = torch.zeros_like(scaled_logits, dtype=topk_probs.dtype)
            target_probs.scatter_(1, topk_indices, topk_probs)
            sparse_topk_applied = True

    if not sparse_topk_applied:
        target_probs = F.softmax(scaled_logits, dim=-1)
        if need_top_k:
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(sampling_info.top_ks, draft_token_num, dim=0),
            )
        if need_top_p:
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(sampling_info.top_ps, draft_token_num, dim=0),
            )
    target_probs = target_probs.view(bs, draft_token_num, -1).contiguous()
    draft_probs = torch.zeros_like(target_probs)

    (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        predicts,
        accept_index,
        accept_token_num,
    ) = _get_or_create_chain_verify_buffers(
        bs=bs,
        draft_token_num=draft_token_num,
        device=device,
    )
    candidates_i64 = (
        candidates if candidates.dtype == torch.int64 else candidates.to(torch.int64)
    )
    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates_i64,
        # kwarg 左侧保留为 `retrive_*` 以匹配 sgl_kernel op schema
        retrive_index=retrieve_index,
        retrive_next_token=retrieve_next_token,
        retrive_next_sibling=retrieve_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )

    correct_len = accept_token_num
    row_ids = torch.arange(bs, dtype=torch.long, device=device)
    accept_pos = accept_index[row_ids, correct_len.to(torch.long)].to(torch.long)
    bonus = predicts[accept_pos].to(torch.int64)
    return correct_len, bonus


def validate_dflash_request(req: Req, enable_overlap: bool) -> Optional[str]:
    """校验 DFLASH 请求是否支持所请求的特性。

    参数：
        req:            请求对象
        enable_overlap: 是否启用 overlap 调度

    返回：错误信息字符串（不支持时），或 None（支持时）
    """
    if req.return_logprob:
        return "DFLASH speculative decoding does not support return_logprob yet."

    if enable_overlap and req.return_hidden_states:
        return "DFLASH speculative decoding does not support return_hidden_states yet."

    if (
        req.sampling_params.json_schema is not None
        or req.sampling_params.regex is not None
        or req.sampling_params.ebnf is not None
        or req.sampling_params.structural_tag is not None
    ):
        return (
            "DFLASH speculative decoding does not support "
            "grammar-constrained decoding yet."
        )

    return None
