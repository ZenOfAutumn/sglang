"""启动推理服务（inference server）。

本模块是 SGLang 的命令行启动入口之一，既支持
``python -m sglang.launch_server`` 这种传统方式，也通过 ``sglang serve`` 间接调用。
它负责解析命令行参数，并根据参数（gRPC 开关、是否为 encoder 专用进程等）
选择并拉起对应的服务实现。

整体流程：
    1. 屏蔽无关紧要的告警噪音（``suppress_noisy_warnings``）；
    2. 加载插件（``load_plugins``）；
    3. 解析命令行参数得到 ``ServerArgs``（``prepare_server_args``）；
    4. 调用 ``run_server`` 根据参数分发到具体服务；
    5. 无论正常结束还是抛异常，都在 ``finally`` 中清理子进程树。
"""

import asyncio
import os
import sys
import warnings

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.plugins import load_plugins
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import suppress_noisy_warnings

# 在导入阶段就抑制掉一批已知的、无意义的告警，避免刷屏干扰日志观察。
suppress_noisy_warnings()


def run_server(server_args):
    """根据 gRPC 相关开关以及 ``server_args.encoder_only`` 选择并启动对应的服务。

    该函数是一个"分发器"（dispatcher）：它本身不实现任何服务逻辑，
    只负责判断当前进程应该以哪种形态运行，然后 import 对应的实现并启动。

    参数：
        server_args (ServerArgs)：
            由 ``prepare_server_args`` 解析命令行后得到的原始参数对象。
            注意：此处传入的仍是"未生效（raw）"的参数，
            所以在函数开头必须先调用 ``resolve_once()`` 完成参数解析/生效。

    返回：
        None：
            函数会阻塞在具体的服务实现中（直到该服务退出）。
            不会返回有意义的返回值，只会在启动失败时抛出异常。

    输入/输出示例：
        >>> from sglang.srt.server_args import prepare_server_args
        >>> server_args = prepare_server_args(["--model-path", "meta-llama/Llama-3-8B"])
        >>> run_server(server_args)   # doctest: +SKIP
        # 阻塞运行，日志：INFO: Started server process ...；Ctrl+C 后返回 None
    """
    # 下面分发所依赖的标志位是由"参数解析（resolution）"决定的
    # （例如 ``--grpc-mode`` 会被折叠合并进 ``smg_grpc_mode``），
    # 而 ``prepare_server_args`` 返回的是未经解析的原始输入，因此这里先解析一次。
    server_args.resolve_once()
    # 获取"已解析视图"：即所有 override / 默认值 / 别名折叠都生效后的配置对象。
    cfg = resolving_view(server_args)

    if cfg.encoder_only:
        # 分支一：encoder 分离部署（encoder disaggregation）场景，
        # 当前进程只承担 encoder 的推理工作，不跑完整的 LLM 服务。
        if cfg.smg_grpc_mode or cfg.grpc_mode:
            from sglang.srt.disaggregation.encoder.grpc_server import (
                serve_grpc_encoder,
            )

            # 以 gRPC 协议暴露 encoder 服务（异步实现，用 asyncio.run 驱动）。
            asyncio.run(serve_grpc_encoder(server_args))
        else:
            from sglang.srt.disaggregation.encoder.http_server import launch_server

            # 以 HTTP 协议暴露 encoder 服务。
            launch_server(server_args)
    elif cfg.smg_grpc_mode:
        # 分支二：旧版 SMG gRPC 服务
        # （由 ``--smg-grpc-mode`` 触发，或被废弃的 ``--grpc-mode`` 经
        # ``__post_init__`` 折叠进 ``smg_grpc_mode``）。
        # 注意：原生 Rust 版 gRPC 服务是另一条独立路径，由 ``--grpc-port`` 开启，
        # 它会与下方默认的 HTTP 服务同时启动，而不是走这里的分支。
        from sglang.srt.entrypoints.grpc_server import serve_grpc

        asyncio.run(serve_grpc(server_args))
    elif cfg.use_ray:
        # 分支三：Ray 模式，即使用 Ray 作为后端的 HTTP 服务。
        try:
            from sglang.srt.ray.http_server import launch_server
        except ImportError:
            # Ray 属于可选依赖，未安装时给出明确的安装指引。
            raise ImportError(
                "Ray is required for --use-ray mode. "
                "Install it with: pip install 'sglang[ray]'"
            )

        launch_server(server_args)
    else:
        # 分支四：默认模式，普通的 HTTP 服务（最常用的启动方式）。
        from sglang.srt.entrypoints.http_server import launch_server

        launch_server(server_args)


if __name__ == "__main__":
    # 提示用户：旧的 ``python -m`` 启动方式仍可用，但推荐使用 ``sglang serve``。
    warnings.warn(
        "'python -m sglang.launch_server' is still supported, but "
        "'sglang serve' is the recommended entrypoint.\n"
        "  Example: sglang serve --model-path <model> [options]",
        UserWarning,
        stacklevel=1,
    )

    # 加载用户插件，允许在正式启动前对框架行为做扩展/打补丁。
    load_plugins()

    # 解析命令行参数（``sys.argv[1:]`` 即去掉脚本名后的全部参数），
    # 得到 ServerArgs 对象；此时参数尚未 resolve。
    server_args = prepare_server_args(sys.argv[1:])

    try:
        # 启动服务（内部会阻塞直到服务退出）。
        run_server(server_args)
    finally:
        # 无论正常退出还是异常退出，都清理本进程派生的子进程，
        # 避免残留子进程占用显存/端口。``include_parent=False`` 表示不杀当前进程自身。
        kill_process_tree(os.getpid(), include_parent=False)
