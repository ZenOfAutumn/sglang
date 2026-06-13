# SGLang 性能仪表盘

一个基于 Web 的仪表盘，用于可视化 SGLang nightly 测试的性能指标。

## 功能特性

- **性能趋势**：查看吞吐量、延迟和 TTFT 随时间的变化趋势
- **模型对比**：比较不同模型和配置之间的性能
- **筛选**：按 GPU 配置、模型、变体和批大小进行筛选
- **交互式图表**：缩放、平移并悬停以查看详细指标
- **运行历史**：查看最近的基准测试运行，并附带指向 GitHub Actions 的链接

## 快速开始

### 选项一：使用本地服务器运行（推荐）

获取来自 GitHub Actions artifacts 的实时数据：

```bash
# Install requirements
pip install requests

# Run the server
python server.py --fetch-on-start

# Visit http://localhost:8000
```

该服务器提供：
- 自动从 GitHub 获取指标
- 缓存以减少 API 调用
- 供前端使用的 `/api/metrics` 端点

### 选项二：手动获取数据

使用 fetch 脚本下载指标数据：

```bash
# Fetch last 30 days of metrics
python fetch_metrics.py --output metrics_data.json

# Fetch a specific run
python fetch_metrics.py --run-id 21338741812 --output single_run.json

# Fetch only scheduled (nightly) runs
python fetch_metrics.py --scheduled-only --days 7
```

## GitHub Token

要从 GitHub 下载 artifacts，你需要进行身份验证：

1. **使用 `gh` CLI**（推荐）：
   ```bash
   gh auth login
   ```

2. **使用环境变量**：
   ```bash
   export GITHUB_TOKEN=your_token_here
   ```

如果没有 token，仪表盘将显示运行的元数据，但不会显示详细的基准测试结果。

## 数据结构

指标 JSON 具有如下结构：

```json
{
  "run_id": "21338741812",
  "run_date": "2026-01-25T22:24:02.090218+00:00",
  "commit_sha": "5cdb391...",
  "branch": "main",
  "results": [
    {
      "gpu_config": "8-gpu-h200",
      "partition": 0,
      "model": "deepseek-ai/DeepSeek-V3.1",
      "variant": "TP8+MTP",
      "benchmarks": [
        {
          "batch_size": 1,
          "input_len": 4096,
          "output_len": 512,
          "latency_ms": 2400.72,
          "input_throughput": 21408.64,
          "output_throughput": 231.74,
          "overall_throughput": 1919.43,
          "ttft_ms": 191.32,
          "acc_length": 3.19
        }
      ]
    }
  ]
}
```

## 部署

### GitHub Pages

该仪表盘可部署到 GitHub Pages 以供公开访问：

1. 将仪表盘文件复制到 `docs/performance_dashboard/`
2. 在仓库设置中启用 GitHub Pages
3. 设置一个 GitHub Action 以定期更新指标数据

### 自托管

对于带有实时数据的自托管部署：

1. 设置一个运行 `server.py` 的服务器
2. 配置 cron job 或 systemd timer 以刷新数据
3. 可选地置于 nginx/caddy 之后以提供 SSL

## 指标说明

- **Overall Throughput**：每秒处理的总 token 数（input + output）
- **Input Throughput**：每秒处理的输入 token 数（prefill 速度）
- **Output Throughput**：每秒生成的输出 token 数（decode 速度）
- **Latency**：完成请求的端到端时间
- **TTFT**：Time to First Token（首 token 时间）——直到第一个输出 token 的时间
- **Acc Length**：speculative decoding 的接受长度（MTP 变体）

## 贡献

要添加对新指标或可视化的支持：

1. 如果数据收集需要更改，请更新 `fetch_metrics.py`
2. 修改 `app.js` 以添加新的图表类型或筛选器
3. 更新 `index.html` 以进行 UI 更改

## 故障排查

**未显示数据**
- 检查浏览器控制台是否有错误
- 验证 GitHub API 是否可访问
- 尝试使用 `server.py --fetch-on-start` 运行

**API 速率限制**
- 使用 GitHub token 以获得更高的限制
- 服务器会将数据缓存 5 分钟

**图表未渲染**
- 确保 Chart.js 正从 CDN 加载
- 检查控制台中是否有 JavaScript 错误
