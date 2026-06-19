"""
Copyright 2023-2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Configure the logging settings of a server.

Usage:
python3 -m sglang.srt.managers.configure_logging --url http://localhost:30000

中译：一个命令行小工具，用于在运行时（不重启服务）动态调整某个 SGLang 服务实例的
      日志相关设置。其原理是向目标服务的 HTTP 端点 /configure_logging 发送一个
      POST 请求，由服务端解析并热更新日志级别、是否记录请求、请求转储（dump）策略等。
      用法见上方 Usage。
"""

import argparse

import requests

if __name__ == "__main__":
    # 中译：解析命令行参数，构造请求体并 POST 到目标服务的 /configure_logging 端点。
    parser = argparse.ArgumentParser()
    # 中译：--url 目标服务地址（默认本机 30000 端口）。
    parser.add_argument("--url", type=str, default="http://localhost:30000")
    # 中译：--log-level 运行时日志级别；不传则保持服务端原有级别。
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set runtime log level",
    )
    # 中译：--log-requests 是否记录每个请求的内容（开关）。
    parser.add_argument("--log-requests", action="store_true")
    # 中译：--log-requests-level 请求日志的详细级别（数值越大越详细）。
    parser.add_argument("--log-requests-level", type=int, default=3)
    # 中译：--dump-requests-folder 请求转储（落盘）目录。
    parser.add_argument(
        "--dump-requests-folder", type=str, default="/tmp/sglang_request_dump"
    )
    # 中译：--dump-requests-threshold 累计多少条请求后触发一次转储落盘。
    parser.add_argument("--dump-requests-threshold", type=int, default=1000)
    parser.add_argument(
        "--dump-requests-exclude-meta-keys",
        type=str,
        default=None,
        help=(
            "Comma-separated meta_info keys to strip from each dumped request "
            "(e.g. 'routed_experts,hidden_states'). Pass an empty string to "
            "keep all keys. If not set, the server default is used."
        ),
    )
    args = parser.parse_args()

    # 中译：组装要发送给服务端的 JSON 请求体（payload）。
    payload = {
        "log_requests": args.log_requests,
        "log_requests_level": args.log_requests_level,  # Log full requests
        "dump_requests_folder": args.dump_requests_folder,
        "dump_requests_threshold": args.dump_requests_threshold,
        "log_level": args.log_level,
    }
    # 中译：仅当显式传入了 exclude-meta-keys 时才加入 payload；
    #       将逗号分隔的字符串拆分、去空白、过滤空项后作为列表传递。
    if args.dump_requests_exclude_meta_keys is not None:
        payload["dump_requests_exclude_meta_keys"] = [
            k.strip()
            for k in args.dump_requests_exclude_meta_keys.split(",")
            if k.strip()
        ]

    # 中译：发送 POST 请求并断言返回 200，确保配置已成功热更新。
    response = requests.post(args.url + "/configure_logging", json=payload)
    assert response.status_code == 200
