# 为 torch.compile 启用缓存

SGLang 使用 torch.compile 的 `max-autotune-no-cudagraphs` 模式。自动调优可能会很慢。
如果你想在许多不同的机器上部署模型,可以将 torch.compile 缓存分发到这些机器上,从而跳过编译步骤。

这是基于 https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html


1. 通过设置 TORCHINDUCTOR_CACHE_DIR 并运行模型一次来生成缓存。
```
TORCHINDUCTOR_CACHE_DIR=/root/inductor_root_cache python3 -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct --enable-torch-compile
```
2. 将缓存文件夹复制到其他机器,并使用 `TORCHINDUCTOR_CACHE_DIR` 启动服务器。
