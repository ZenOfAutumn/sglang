# 使用 Docker 进行开发的指南

## 在远程主机上设置 VSCode
（可选 - 如果你打算在本地运行 sglang 开发容器，可以跳过此步骤）

1. 在远程主机上，从 [Https://code.visualstudio.com/docs/?dv=linux64cli](https://code.visualstudio.com/download) 下载 `code`，并在 shell 中运行 `code tunnel`。

示例
```bash
wget https://vscode.download.prss.microsoft.com/dbazure/download/stable/fabdb6a30b49f79a7aba0f2ad9df9b399473380f/vscode_cli_alpine_x64_cli.tar.gz
tar xf vscode_cli_alpine_x64_cli.tar.gz

# https://code.visualstudio.com/docs/remote/tunnels
./code tunnel
```

2. 在你的本地机器上，在 VSCode 中按 F1 并选择 "Remote Tunnels: Connect to Tunnel"。

## 设置 Docker 容器

### 选项 1. 从 VSCode 自动使用默认的开发容器
sglang 仓库根目录下有一个 `.devcontainer` 文件夹，允许 VSCode 自动在开发容器中启动。你可以在 VSCode 官方文档 [Developing inside a Container](https://code.visualstudio.com/docs/devcontainers/containers) 中了解更多关于此 VSCode 扩展的信息。
![image](https://github.com/user-attachments/assets/6a245da8-2d4d-4ea8-8db1-5a05b3a66f6d)
(*图 1：来自 VSCode 官方文档 [Developing inside a Container](https://code.visualstudio.com/docs/devcontainers/containers) 的示意图。*)

要启用此功能，你只需要：
1. 启动 Visual Studio Code 并安装 [VSCode dev container extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)。
2. 按 F1，输入并选择 "Dev Container: Open Folder in Container。
3. 输入你机器上的 `sglang` 本地仓库路径并按回车。

首次在开发容器中打开它可能需要较长时间，因为需要进行 docker pull 和构建。一旦成功，你应该会在左下角的状态栏看到显示你正处于开发容器中：

![image](https://github.com/user-attachments/assets/650bba0b-c023-455f-91f9-ab357340106b)

现在当你在 VSCode 终端中运行 `sglang.launch_server` 或使用 F5 开始调试时，sglang server 将在开发容器中启动，并自动应用你所有的本地更改：

![image](https://github.com/user-attachments/assets/748c85ba-7f8c-465e-8599-2bf7a8dde895)


### 选项 2. 手动启动容器（进阶）

下面的启动命令是 SGLang 团队内部开发的一个示例。你可以**根据需要修改或添加目录映射**，特别是对于模型权重下载，以防止不同的 Docker 容器重复下载。

❗️ **关于 RDMA 的说明**

    1. RDMA 需要 `--network host` 和 `--privileged`。如果你不需要 RDMA，可以移除它们，但保留它们也没有坏处。因此，我们在下面的命令中默认启用这两个标志。
    2. 如果你使用 RoCE，你可能需要设置 `NCCL_IB_GID_INDEX`，例如：`export NCCL_IB_GID_INDEX=3`。

```bash
# Change the name to yours
docker run -itd --shm-size 32g --gpus all -v <volumes-to-mount> --ipc=host --network=host --privileged --name sglang_dev lmsysorg/sglang:dev /bin/zsh
docker exec -it sglang_dev /bin/zsh
```
一些有用的挂载卷包括：
1. **Huggingface 模型缓存**：挂载模型缓存可以避免每次 docker 重启时重新下载。Linux 上的默认位置是 `~/.cache/huggingface/`。
2. **SGLang 仓库**：SGLang 本地仓库中的代码更改将自动同步到 .devcontainer。

示例 1：仅挂载本地缓存文件夹 `/opt/dlami/nvme/.cache` 而不挂载 SGLang 仓库。当你更愿意手动将本地代码更改传输到开发容器时使用此方式。
```bash
docker run -itd --shm-size 32g --gpus all -v /opt/dlami/nvme/.cache:/root/.cache --ipc=host --network=host --privileged --name sglang_zhyncs lmsysorg/sglang:dev /bin/zsh
docker exec -it sglang_zhyncs /bin/zsh
```
示例 2：同时挂载 HuggingFace 缓存和本地 SGLang 仓库。由于 SGLang 在开发镜像中以可编辑模式安装，本地代码更改将自动同步到开发容器。
```bash
docker run -itd --shm-size 32g --gpus all -v $HOME/.cache/huggingface/:/root/.cache/huggingface -v $HOME/src/sglang:/sgl-workspace/sglang --ipc=host --network=host --privileged --name sglang_zhyncs lmsysorg/sglang:dev /bin/zsh
docker exec -it sglang_zhyncs /bin/zsh
```
## 使用 VSCode 调试器调试 SGLang
1. （如果不存在则创建）在 VSCode 中打开 `launch.json`。
2. 添加以下配置并保存。请注意，你可以根据需要编辑脚本以应用不同的参数或调试不同的程序（例如 benchmark 脚本）。
     ```JSON
       {
          "version": "0.2.0",
          "configurations": [
              {
                  "name": "Python Debugger: launch_server",
                  "type": "debugpy",
                  "request": "launch",
                  "module": "sglang.launch_server",
                  "console": "integratedTerminal",
                  "args": [
                      "--model-path", "meta-llama/Llama-3.2-1B",
                      "--host", "0.0.0.0",
                      "--port", "30000",
                      "--trust-remote-code",
                  ],
                  "justMyCode": false
              }
          ]
      }
    ```

3. 按 "F5" 开始。即使程序运行在远程 SSH/Tunnel 主机 + 开发容器上，VSCode 调试器也会确保程序在断点处暂停。

## Profile

```bash
# Change batch size, input, output and add `disable-cuda-graph` (for easier analysis)
# e.g. DeepSeek V3
nsys profile -o deepseek_v3 python3 -m sglang.bench_one_batch --batch-size 1 --input 128 --output 256 --model deepseek-ai/DeepSeek-V3 --trust-remote-code --tp 8 --disable-cuda-graph
```

## 评估

```bash
# e.g. gsm8k 8 shot
python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --num-shots 8
```
