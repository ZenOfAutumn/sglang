from typing import Optional, Union

import zmq

from sglang.srt.managers.io_struct import BaseBatchReq, BaseReq


class SenderWrapper:
    """中译：对一个 ZMQ 套接字的轻量发送封装。

    职责：把调度器产出的输出对象通过 ZMQ 发往下游（如 detokenizer / tokenizer）；
    并在多 HTTP worker 场景下，自动把请求来源的回传地址（http_worker_ipc）从收到的
    请求对象透传到输出对象上，确保响应能被路由回正确的 HTTP worker。
    """

    def __init__(self, socket: zmq.Socket):
        # 中译：保存底层套接字；允许为 None（此时 send_output 直接空操作）。
        self.socket = socket

    def send_output(
        self,
        output: Union[BaseReq, BaseBatchReq],
        recv_obj: Optional[Union[BaseReq, BaseBatchReq]] = None,
    ):
        # 中译：套接字未配置（None）时直接返回，相当于丢弃发送（如某些进程不需要该通道）。
        if self.socket is None:
            return

        if (
            isinstance(recv_obj, BaseReq)
            and recv_obj.http_worker_ipc is not None
            and output.http_worker_ipc is None
        ):
            # handle communicator reqs for multi-http worker case
            # 中译：多 HTTP worker 场景——若输出本身没带回传地址，就从原始请求继承，
            #       使响应能路由回发起该请求的那个 HTTP worker。
            output.http_worker_ipc = recv_obj.http_worker_ipc

        # 中译：以 pickle 形式（send_pyobj）把输出对象发送出去。
        self.socket.send_pyobj(output)
