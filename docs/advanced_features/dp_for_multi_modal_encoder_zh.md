# SGLang 中面向多模态编码器的 DP

一个典型的 VLM 架构涉及两个主要组件:一个多模态编码器和一个文本解码器。

大多数 VLM 使用 Vision Transformer(ViT)作为其多模态编码器,它负责处理视觉数据、提取特征(物体、颜色、纹理等),并将其转换为模型能够理解的格式。

文本解码器基于 LLM。它处理文本数据,并基于编码后的视觉特征生成输出。

然而,由于 ViT 的大小相比语言解码器非常小,
从 TP 中获得的收益相对较少。另一方面,由于在每一层之后都要执行 all-reduce,TP 会带来显著的通信开销。

将 ViT 置于数据并行(data parallel)中,同时保持 LLM 处于张量并行(tensor parallel),能够持续降低 TTFT 并提升端到端吞吐量。在这种混合布局中,视觉前端变得并行且轻量,而稀缺的互联带宽和集合通信操作则被保留给 LLM。

数据并行将整个模型复制到多个 GPU 组上,并行处理不同批次的请求。

## 命令示例
你可以通过设置 `mm-enable-dp-encoder` 来启用批次级的 DP,例如:
```
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-VL-7B-Instruct \
    --tp 2 \
    --mm-enable-dp-encoder
```

## 已知支持的模型
- Qwen2.5-VL (<https://github.com/sgl-project/sglang/pull/13126>)
- Qwen3-VL (<https://github.com/sgl-project/sglang/pull/13724>)
- InternVL (<https://github.com/sgl-project/sglang/pull/13925>)
- GLM-4.5V & GLM-4.6V (<https://github.com/sgl-project/sglang/pull/14097>)
