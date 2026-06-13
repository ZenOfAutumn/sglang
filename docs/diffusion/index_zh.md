# SGLang Diffusion

SGLang Diffusion 是一个面向图像和视频生成的高性能推理框架。它提供原生的 SGLang pipeline、diffusers 后端支持、一个 OpenAI 兼容的服务端,以及一套基于预编译 `sgl-kernel` 算子和针对关键推理路径的 JIT kernel 构建的优化 kernel 栈。

## 主要特性

- 广泛的模型支持,涵盖 Wan、Hunyuan、Qwen-Image、FLUX、Z-Image、GLM-Image 等
- 借助 `sgl-kernel`、JIT kernel、调度器改进以及缓存加速实现快速推理
- 多种接口:`sglang generate`、`sglang serve` 以及 OpenAI 兼容 API
- 多平台支持,涵盖 NVIDIA、AMD、Ascend、Apple Silicon 和 Moore Threads

## 快速开始

```bash
uv pip install "sglang[diffusion]" --prerelease=allow
```

```bash
sglang generate --model-path Qwen/Qwen-Image \
  --prompt "A beautiful sunset over the mountains" \
  --save-output
```

```bash
sglang serve --model-path Qwen/Qwen-Image --port 30010
```

## 从这里开始

- [Installation](installation.md):安装 SGLang Diffusion 及平台依赖
- [Compatibility Matrix](compatibility_matrix.md):查看模型和优化支持情况
- [CLI](api/cli.md):运行一次性生成任务或启动一个持久化服务端
- [OpenAI-Compatible API](api/openai_api.md):向 HTTP 服务端发送图像和视频请求
- [Attention Backends](performance/attention_backends.md):为你的模型和硬件选择最佳后端
- [Caching Acceleration](performance/cache/index.md):使用 Cache-DiT 或 TeaCache 来降低去噪开销
- [Quantization](quantization.md):加载量化后的 transformer 检查点
- [Contributing](contributing.md):贡献流程、添加新模型以及 CI 性能基线

## 附加文档

- [Post-Processing](api/post_processing.md):帧插值与超分辨率
- [Performance Overview](performance/index.md):attention、缓存与性能分析概览
- [Environment Variables](environment_variables.md):平台、缓存、存储和调试相关配置
- [Support New Models](support_new_models.md):新 diffusion pipeline 的实现指南
- [CI Performance](ci_perf.md):性能基线生成

## 参考资料

- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [Cache-DiT](https://github.com/vipshop/cache-dit)
- [FastVideo](https://github.com/hao-ai-lab/FastVideo)
- [xDiT](https://github.com/xdit-project/xDiT)
- [Diffusers](https://github.com/huggingface/diffusers)
