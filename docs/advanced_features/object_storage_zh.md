# 从对象存储加载模型(Loading Models from Object Storage)

SGLang 支持直接从对象存储(S3 和 Google Cloud Storage)加载模型,而无需完整的本地下载。此功能使用 `runai_streamer` 加载格式,直接从云存储流式传输模型权重,显著减少启动时间和本地存储需求。

## 概述

从对象存储加载模型时,SGLang 采用两阶段方法:

1. **元数据下载**(一次性,在进程启动前):配置文件和 tokenizer 文件被下载到本地缓存
2. **权重流式传输**(惰性,在模型加载期间):模型权重按需直接从对象存储流式传输

## 支持的存储后端

1. **Amazon S3**:`s3://bucket-name/path/to/model/`
2. **Google Cloud Storage**:`gs://bucket-name/path/to/model/`
3. **Azure Blob**:`az://some-azure-container/path/`
4. **S3 兼容存储**:`s3://bucket-name/path/to/model/`

## 快速开始

### 基本用法

只需提供一个对象存储 URI 作为模型路径:

```bash
# S3
python -m sglang.launch_server \
  --model-path s3://my-bucket/models/llama-3-8b/ \
  --load-format runai_streamer

# Google Cloud Storage
python -m sglang.launch_server \
  --model-path gs://my-bucket/models/llama-3-8b/ \
  --load-format runai_streamer
```

**注意**:当使用对象存储 URI 时,`--load-format runai_streamer` 会被自动检测,因此你可以省略它:

```bash
python -m sglang.launch_server \
  --model-path s3://my-bucket/models/llama-3-8b/
```

### 配合张量并行

```bash
python -m sglang.launch_server \
  --model-path gs://my-bucket/models/llama-70b/ \
  --tp 4 \
  --model-loader-extra-config '{"distributed": true}'
```

## 配置

### 加载格式

`runai_streamer` 加载格式专为对象存储、SSD 和共享文件系统设计

```bash
python -m sglang.launch_server \
  --model-path s3://bucket/model/ \
  --load-format runai_streamer
```

### 扩展配置参数

使用 `--model-loader-extra-config` 以 JSON 字符串形式传递额外配置:

```bash
python -m sglang.launch_server \
  --model-path s3://bucket/model/ \
  --model-loader-extra-config '{
    "distributed": true,
    "concurrency": 8,
    "memory_limit": 2147483648
  }'
```

#### 可用参数

| 参数 | 类型 | 描述 | 默认值 |
|-----------|------|-------------|---------|
| `distributed` | bool | 为多 GPU 配置启用分布式流式传输。对于对象存储路径和 cuda 类设备会自动设置为 `true`。 | 自动检测 |
| `concurrency` | int | 并发下载流的数量。更高的值可以提升大模型的吞吐量。 | 4 |
| `memory_limit` | int | 流式传输缓冲区的内存限制(以字节为单位)。 | 取决于系统 |


## 性能考量

### 分布式流式传输

对于多 GPU 配置,启用分布式流式传输以在进程之间并行化权重加载:

```bash
python -m sglang.launch_server \
  --model-path s3://bucket/model/ \
  --tp 8 \
  --model-loader-extra-config '{"distributed": true}'
```

## 限制

- **支持的格式**:目前仅支持 `.safetensors` 权重格式(推荐格式)
- **支持的设备**:分布式流式传输在 cuda 类设备上受支持。否则回退到非分布式流式传输

## 另请参阅

- [Runai model streamer 文档](https://github.com/run-ai/runai-model-streamer)
