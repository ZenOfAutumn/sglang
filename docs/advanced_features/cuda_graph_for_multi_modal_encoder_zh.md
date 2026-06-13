# SGLang 中面向多模态编码器的 CUDA Graph

## 动机

在多模态推理服务中,视觉编码器(ViT / Vision Transformer)通常具有几个特征性的特点:

层数多、算子碎片化:每一层都包含 LN、QKV 投影、注意力、MLP、残差连接等,导致极其频繁的 kernel 启动。

服务端"小批次 / 低延迟"很常见:批次大小非常小(在批次"展平(flatten)"后有时看起来像是 1),因此 kernel 启动开销占据了端到端延迟的很大一部分。

输入 token 数量(patch 数量)频繁变化:不同的图像/视频分辨率以及不同的批次组成导致不同的序列长度 S——而这恰恰是 CUDA Graph 最大的障碍(形状不稳定)。

CUDA Graph 的价值:它将一长串具有固定形状和固定内存地址的 GPU kernel 捕获到一个图中;之后,对于相同的形状,它可以直接重放该图,从而大幅减少启动开销并使 GPU 调度更加紧凑。

这促使我们寻求一种为 ViT 启用 CUDA Graph 的功能,以提升 ViT 的性能。

## 设计与限制

新的启用 CUDA Graph 的 ViT 逻辑构建在 ViTCudaGraphRunner 之上。这个 runner 将 vision transformer 中的"blocks + merger + deepstack merger(可选)"部分捕获到一个 CUDA graph 中,并对相同的形状进行重放。更多细节请参见以下设计考量和限制。

### 用动态输入适配 CUDA Graph 的静态约束

可变的序列长度 S 在 ViT 中非常常见。而 CUDA Graph 要求固定的形状。解决方案是按 S 构建一个图缓存(例如,graph_key = S)。第一次遇到新的 S 时,捕获一个图;之后,重放它。

如果存在许多不同的 S 值,我们需要增加显存(VRAM)使用量,即为许多图准备的图私有内存池。

### 稳定的地址

所有"类参数(parameter-like)"的东西都变成静态缓冲区:

- block_input / block_ws / block_output
- cu_full_len / cu_window_len 及其 kk 变体
- sin_cos_ws

通过这种方式来满足底层要求:在重放期间,不允许交换张量,只能修改张量内容。

### 注意力后端参数
注意力后端参数在图内是固定的:

TritonAttn 期望 [cu_seqlens, cu_seqlens_kk, max_len]
FA3 期望 [cu_seqlens, max_len]

max_len 被冻结为一个 int 常量。
cu_seqlens 在 create_graph() 期间被缓存到一个 dict 中,其内容在后续的重放期间不会被更新。

对于相同的 graph_key = S,你不仅要求输入形状匹配,还要求 cu_seqlens(以及 window seqlens)中的分段模式完全相同。否则,注意力会错误地分割序列。

### Rotary 缓冲区管理
当 seq_len 增加时,该功能会重新分配一个更大的 sin_cos_ws。
max_content_len 用于确保所分配 rotary 缓冲区的最大大小。


## 命令示例
你可以通过设置环境变量 `SGLANG_VIT_ENABLE_CUDA_GRAPH=1` 为 ViT 启用 CUDA Graph,例如:
```
SGLANG_VIT_ENABLE_CUDA_GRAPH=1 \
python3 -m sglang.launch_server \
  --model Qwen/Qwen3-VL-8B-Instruct
```
或者,你可以通过同时设置环境变量 `SGLANG_VIT_ENABLE_CUDA_GRAPH=1` 和设置 `--enable-piecewise-cuda-graph`,将 ViT 的 CUDA Graph 与分段 CUDA Graph(Piecewise CUDA Graph)功能一起运行,例如:
```
SGLANG_VIT_ENABLE_CUDA_GRAPH=1 \
python3 -m sglang.launch_server \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --piecewise-cuda-graph-max-tokens 4096 \
  --enable-piecewise-cuda-graph \
  --piecewise-cuda-graph-compiler eager
```

## 已知支持的模型
- Qwen2.5-VL (https://github.com/sgl-project/sglang/pull/14422)
- Qwen3-VL (https://github.com/sgl-project/sglang/pull/15320)
