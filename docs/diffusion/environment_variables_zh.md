# 环境变量

## Apple MPS

| Environment Variable | Default | Description                                                  |
|----------------------|---------|--------------------------------------------------------------|
| `SGLANG_USE_MLX`     | not set | 设置为 `1` 可在 MPS 上为 norm 运算启用 MLX 融合 Metal kernel |

## 缓存加速

这些变量用于配置 Diffusion Transformer(DiT)模型的缓存加速。
SGLang 支持多种缓存策略 —— 概览参见 [caching documentation](performance/cache/index.md)。

### Cache-DiT 配置

详细配置参见 [cache-dit documentation](performance/cache/cache_dit.md)。

| Environment Variable                | Default | Description                              |
|-------------------------------------|---------|------------------------------------------|
| `SGLANG_CACHE_DIT_ENABLED`          | false   | 启用 Cache-DiT 加速            |
| `SGLANG_CACHE_DIT_FN`               | 1       | 始终计算的前 N 个 block         |
| `SGLANG_CACHE_DIT_BN`               | 0       | 始终计算的后 N 个 block          |
| `SGLANG_CACHE_DIT_WARMUP`           | 4       | 开始缓存前的预热步数              |
| `SGLANG_CACHE_DIT_RDT`              | 0.24    | 残差差异阈值            |
| `SGLANG_CACHE_DIT_MC`               | 3       | 最大连续缓存步数              |
| `SGLANG_CACHE_DIT_TAYLORSEER`       | false   | 启用 TaylorSeer 校准器             |
| `SGLANG_CACHE_DIT_TS_ORDER`         | 1       | TaylorSeer 阶数(1 或 2)                |
| `SGLANG_CACHE_DIT_SCM_PRESET`       | none    | SCM 预设(none/slow/medium/fast/ultra) |
| `SGLANG_CACHE_DIT_SCM_POLICY`       | dynamic | SCM 缓存策略                       |
| `SGLANG_CACHE_DIT_SCM_COMPUTE_BINS` | not set | 自定义 SCM compute bins                  |
| `SGLANG_CACHE_DIT_SCM_CACHE_BINS`   | not set | 自定义 SCM cache bins                  |

## 云存储

这些变量用于配置兼容 S3 的云存储,以便自动上传生成的图像和视频。

| Environment Variable            | Default | Description                                            |
|---------------------------------|---------|--------------------------------------------------------|
| `SGLANG_CLOUD_STORAGE_TYPE`     | not set | 设置为 `s3` 以启用云存储                    |
| `SGLANG_S3_BUCKET_NAME`         | not set | S3 bucket 的名称                              |
| `SGLANG_S3_ENDPOINT_URL`        | not set | 自定义 endpoint URL(用于 MinIO、OSS 等)             |
| `SGLANG_S3_REGION_NAME`         | us-east-1 | AWS region 名称                                      |
| `SGLANG_S3_ACCESS_KEY_ID`       | not set | AWS Access Key ID                                      |
| `SGLANG_S3_SECRET_ACCESS_KEY`   | not set | AWS Secret Access Key                                  |

## CUDA 崩溃调试

这些变量在 diffusion CUDA kernel 调用边界处启用 kernel API 日志记录以及可选的输入/输出 dump。它们在排查诸如非法内存访问、设备端 assert 或自定义 kernel 中形状不匹配等 CUDA 崩溃时非常有用。

| Environment Variable | Default | Description |
|----------------------|---------|-------------|
| `SGLANG_KERNEL_API_LOGLEVEL` | `0` | 控制崩溃调试 kernel API 日志。`1` 记录 API 名称,`3` 记录张量元数据,`5` 增加张量统计信息,`10` 还会写入 dump 快照。 |
| `SGLANG_KERNEL_API_LOGDEST` | `stdout` | 崩溃调试 kernel API 日志的目标位置。可使用 `stdout`、`stderr` 或文件路径。`%i` 会被替换为进程的 PID。 |
| `SGLANG_KERNEL_API_DUMP_DIR` | `sglang_kernel_api_dumps` | level-10 kernel API dump 的输出目录。`%i` 会被替换为进程的 PID。 |
| `SGLANG_KERNEL_API_DUMP_INCLUDE` | not set | 以逗号分隔的通配符模式,用于指定要包含在 level-10 dump 中的 kernel API 名称。 |
| `SGLANG_KERNEL_API_DUMP_EXCLUDE` | not set | 以逗号分隔的通配符模式,用于指定要从 level-10 dump 中排除的 kernel API 名称。 |
