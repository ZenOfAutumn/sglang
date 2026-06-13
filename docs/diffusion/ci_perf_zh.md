# CI 性能

## 性能基线生成脚本

`python/sglang/multimodal_gen/test/scripts/gen_perf_baselines.py` 会启动一个本地 diffusion 服务端,为选定的测试用例发起请求,从性能日志中汇总各阶段(stage)、去噪步(denoise-step)和端到端(E2E)的耗时,并将结果写回 `perf_baselines.json` 的 `scenarios` 部分。

### 用法

更新单个用例:

```bash
python python/sglang/multimodal_gen/test/scripts/gen_perf_baselines.py --case qwen_image_t2i
```

通过正则表达式选择:

```bash
python python/sglang/multimodal_gen/test/scripts/gen_perf_baselines.py --match 'qwen_image_.*'
```

运行基线文件 `scenarios` 中的所有键:

```bash
python python/sglang/multimodal_gen/test/scripts/gen_perf_baselines.py --all-from-baseline
```

指定输入/输出路径和超时:

```bash
python python/sglang/multimodal_gen/test/scripts/gen_perf_baselines.py --baseline python/sglang/multimodal_gen/test/server/perf_baselines.json --out /tmp/perf_baselines.json --timeout 600
```
