> 来源: <https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/prefix-aware-routing.html>
> 抓取时间: 2026-09-01 ｜ 格式: 网页转 Markdown
> 所属项目: <https://github.com/vllm-project/production-stack>

# 前缀感知路由 — vLLM Production Stack

本教程演示如何在 vLLM Production Stack 中使用前缀感知路由（prefix aware routing）。前缀感知路由确保具有相同提示词前缀的后续请求被路由到同一个实例，从而最大化 KV 缓存利用率并提升性能。

## 前置条件

* 已完成以下教程：
  * Prerequisite（前置准备）
  * Quick Start（快速开始）
* 一个支持 GPU 的 Kubernetes 环境
* 对 Kubernetes 和 Helm 有基本了解

## 第 1 步：部署启用前缀感知路由的服务

我们将使用预定义的配置文件 values-18-prefix-aware.yaml，它会启动两个启用了前缀感知路由的 vLLM 实例。

1. 使用该配置部署 Helm chart：

```bash
helm install vllm helm/ -f tutorials/assets/values-18-prefix-aware.yaml
```

等待部署完成：

```bash
kubectl get pods -w
```

## 第 2 步：端口转发

将 router 服务端口转发到本地机器：

```bash
kubectl port-forward svc/vllm-router-service 30080:80
```

## 第 3 步：测试前缀感知路由

首先，向 router 发送一个请求：

```bash
curl http://localhost:30080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "prompt": "What is the capital of France?",
  "max_tokens": 100
}'
```

然后，发送另一个具有相同提示词前缀的请求：

```bash
curl http://localhost:30080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "prompt": "What is the capital of France? And what is its population?",
  "max_tokens": 100
}'
```

你应该会观察到第二个请求被路由到了与第一个请求相同的实例。这是因为前缀感知 router 检测到第二个请求与第一个请求共享前缀，于是将其路由到同一个实例以最大化 KV 缓存利用率。

具体来说，你应该会看到类似下面的日志：

```
[2025-06-03 06:16:28,963] LMCache DEBUG: Scheduled to load 5 tokens for request cmpl-306538839e87480ca5604ecc5f75c847-0 (vllm_v1_adapter.py:299:lmcache.integration.vllm.vllm_v1_adapter)
[2025-06-03 06:16:28,966] LMCache DEBUG: Retrieved 6 out of 6 out of total 6 tokens (cache_engine.py:330:lmcache.experimental.cache_engine)
```

## 第 4 步：清理

清理本次部署：

```bash
helm uninstall vllm
```

## 结论

在本教程中，我们演示了如何：

1. 部署启用前缀感知路由的 vLLM Production Stack
2. 设置端口转发以访问 router
3. 测试前缀感知路由功能

前缀感知路由特性通过确保具有共享前缀的请求被路由到同一个实例，最大化 KV 缓存利用率，从而帮助提升性能。

---

*注: 该页面为 vLLM Production Stack 文档站的 Use Cases 教程页;同目录下还有 KV Cache Aware Routing / Load Aware Routing / Sharing KV Cache Across Instances 等相关页面。*
