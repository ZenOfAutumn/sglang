# 为 GitHub Actions 设置自托管 Runner

## 添加一个 Runner

### 步骤 1：启动一个 docker 容器。

**你可以挂载一个文件夹用于共享的 huggingface 模型权重缓存。**
下面的命令以 `/tmp/huggingface` 为例。

```
docker pull nvidia/cuda:12.9.1-devel-ubuntu22.04
# Nvidia
docker run --shm-size 128g -it -v /tmp/huggingface:/hf_home --gpus all nvidia/cuda:12.9.1-devel-ubuntu22.04 /bin/bash
# AMD
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --shm-size 128g -it -v /tmp/huggingface:/hf_home lmsysorg/sglang:v0.5.8-rocm700-mi30x /bin/bash
# AMD just the last 2 GPUs
docker run --rm --device=/dev/kfd --device=/dev/dri/renderD176 --device=/dev/dri/renderD184 --group-add video --shm-size 128g -it -v /tmp/huggingface:/hf_home lmsysorg/sglang:v0.5.8-rocm700-mi30x /bin/bash
```

### 步骤 2：通过 `config.sh` 配置 runner

在容器内运行以下命令。

```
apt update && apt install -y curl python3-pip git
pip install --upgrade pip
export RUNNER_ALLOW_RUNASROOT=1
```

然后按照 https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/adding-self-hosted-runners 运行 `config.sh`

**注意事项**
- 不需要指定 runner group
- 给它起一个名字（例如 `test-sgl-gpu-0`）以及一些标签（例如 `1-gpu-h100`）。这些标签之后可以在 Github Settings 中编辑。
- 不需要更改工作文件夹。

### 步骤 3：通过 `run.sh` 运行 runner

- 设置环境变量
```
export HF_HOME=/hf_home
export SGLANG_IS_IN_CI=true
export HF_TOKEN=hf_xxx
export OPENAI_API_KEY=sk-xxx
export CUDA_VISIBLE_DEVICES=0
```

- 让它持续运行
```
while true; do ./run.sh; echo "Restarting..."; sleep 2; done
```
