# 使用来自 ModelScope 的模型

要使用来自 [ModelScope](https://www.modelscope.cn) 的模型，请设置环境变量 `SGLANG_USE_MODELSCOPE`。

```bash
export SGLANG_USE_MODELSCOPE=true
```

我们以 [Qwen2-7B-Instruct](https://www.modelscope.cn/models/qwen/qwen2-7b-instruct) 为例。

启动服务器：
```bash
python -m sglang.launch_server --model-path qwen/Qwen2-7B-Instruct --port 30000
```

或者通过 docker 启动：

```bash
docker run --gpus all \
    -p 30000:30000 \
    -v ~/.cache/modelscope:/root/.cache/modelscope \
    --env "SGLANG_USE_MODELSCOPE=true" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 30000
```

请注意，modelscope 使用与 huggingface 不同的缓存目录。你可能需要手动设置缓存目录以避免磁盘空间耗尽。
