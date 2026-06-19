# 中译：本模块集中封装 Scheduler 进程的「进程间通信（IPC）通道」——一组 ZMQ 套接字。
#       Scheduler 通过这些通道与上下游进程交互：
#         - recv_from_tokenizer：从 TokenizerManager 接收待处理请求（PULL）。
#         - recv_from_rpc：接收带状态控制的 RPC 命令（DEALER）。
#         - send_to_tokenizer / send_to_detokenizer：把结果发往 Tokenizer / Detokenizer（PUSH）。
#         - send_metrics_from_scheduler：上报指标（PUSH，按需开启）。
#       这些通道只在 rank 0（is_rank_zero）上真正建立；其余 rank 用空占位对象，
#       因为只有 rank 0 与外部进程通信。
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import zmq

from sglang.srt.managers.scheduler_components.output_sender import SenderWrapper
from sglang.srt.server_args import PortArgs
from sglang.srt.utils.network import get_zmq_socket

if TYPE_CHECKING:
    from sglang.test.scripted_runtime.tokenizer_recv_proxy import (
        ScriptedTokenizerRecvProxy,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class SchedulerIpcChannels:
    # 中译：Scheduler 用到的全部 IPC 通道的聚合体（frozen 不可变）。
    #       recv_from_tokenizer 在脚本化运行时下会被包成 ScriptedTokenizerRecvProxy，故为 Union 类型。
    recv_from_tokenizer: Union[zmq.Socket, "ScriptedTokenizerRecvProxy"]
    recv_from_rpc: Optional[zmq.Socket]
    send_to_tokenizer: SenderWrapper
    send_to_detokenizer: SenderWrapper
    send_metrics_from_scheduler: Optional[zmq.Socket]

    @classmethod
    def create(
        cls,
        *,
        port_args: PortArgs,
        is_rank_zero: bool,
        skip_tokenizer_init: bool,
        metrics_enabled: bool,
        enable_scripted_runtime: bool,
    ) -> "SchedulerIpcChannels":
        # 中译：工厂方法——根据是否为 rank 0、是否跳过分词器、是否启用指标等条件，创建相应套接字。
        context = zmq.Context(2)

        if is_rank_zero:
            # 中译：只有 rank 0 与外部进程通信，故仅在此真正建立各套接字。
            recv_from_tokenizer = get_zmq_socket(
                context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )
            if enable_scripted_runtime:
                # 中译：脚本化运行时下，用代理包裹接收端，以便测试拦截/注入来自 tokenizer 的消息。
                from sglang.test.scripted_runtime.tokenizer_recv_proxy import (
                    ScriptedTokenizerRecvProxy,
                )

                recv_from_tokenizer = ScriptedTokenizerRecvProxy(
                    underlying=recv_from_tokenizer
                )
            recv_from_rpc = get_zmq_socket(
                context, zmq.DEALER, port_args.rpc_ipc_name, False
            )

            send_to_tokenizer_raw = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_ipc_name, False
            )
            if skip_tokenizer_init:
                # Directly send to the TokenizerManager
                # 中译：跳过分词器初始化时，没有 Detokenizer 进程，结果直接发回 TokenizerManager
                #       （故这里复用 tokenizer 的 IPC 名）。
                send_to_detokenizer_raw = get_zmq_socket(
                    context, zmq.PUSH, port_args.tokenizer_ipc_name, False
                )
            else:
                # Send to the DetokenizerManager
                # 中译：常规情况下，结果发往 DetokenizerManager 做反向解码。
                send_to_detokenizer_raw = get_zmq_socket(
                    context, zmq.PUSH, port_args.detokenizer_ipc_name, False
                )

            # 中译：用 SenderWrapper 包裹原始 PUSH 套接字（统一发送接口、便于多 worker 路由等）。
            send_to_tokenizer = SenderWrapper(send_to_tokenizer_raw)
            send_to_detokenizer = SenderWrapper(send_to_detokenizer_raw)
        else:
            # 中译：非 rank 0：不建实际套接字，接收端置 None，发送端用包裹 None 的空 SenderWrapper（发送即无操作）。
            recv_from_tokenizer = None
            recv_from_rpc = None
            send_to_tokenizer = SenderWrapper(None)
            send_to_detokenizer = SenderWrapper(None)

        # 中译：仅在启用指标采集时创建指标上报通道。
        if metrics_enabled:
            send_metrics_from_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.metrics_ipc_name, False
            )
        else:
            send_metrics_from_scheduler = None

        return cls(
            recv_from_tokenizer=recv_from_tokenizer,
            recv_from_rpc=recv_from_rpc,
            send_to_tokenizer=send_to_tokenizer,
            send_to_detokenizer=send_to_detokenizer,
            send_metrics_from_scheduler=send_metrics_from_scheduler,
        )
