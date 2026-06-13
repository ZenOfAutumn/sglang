# 贡献指南

欢迎来到 **SGLang**！我们非常感谢你有兴趣参与贡献。本指南简明地概述了如何设置你的环境、运行测试、构建文档以及提交 Pull Request（PR）。无论你是修复一个小 bug 还是开发一个重要功能，我们都鼓励遵循这些步骤以获得顺畅的贡献体验。

## 从源码安装 SGLang

### Fork 并克隆仓库

**注意**：新贡献者**没有**向 SGLang 官方仓库推送的写入权限。请在你的 GitHub 账户下 fork 该仓库，然后将你的 fork 克隆到本地。

```bash
git clone https://github.com/<your_user_name>/sglang.git
```

### 从源码构建

参考 [Install SGLang from Source](../get_started/install.md#method-2-from-source)。

## 使用 pre-commit 格式化代码

我们使用 [pre-commit](https://pre-commit.com/) 来保持一致的代码风格检查。在推送你的更改之前，请运行：

```bash
pip3 install pre-commit
pre-commit install
pre-commit run --all-files
```

- **`pre-commit run --all-files`** 手动运行所有已配置的检查，并在可能的情况下应用修复。如果第一次失败，请重新运行以确保 lint 错误被完全解决。在创建 Pull Request **之前**，请确保你的代码通过所有检查。
- **不要**直接提交到 `main` 分支。始终创建一个新分支（例如 `feature/my-new-feature`），推送你的更改，然后从该分支提交 PR。
- 使用 lychee 进行的链接检查在 **CI 中是强制执行的**。默认情况下，它不会阻塞本地提交。
- 要手动运行本地链接检查，请使用：`pre-commit run --hook-stage manual lychee --all-files`。

## 运行并添加单元测试

如果你添加了新功能或修复了 bug，请添加相应的单元测试以确保覆盖率并防止回归。

### 单元测试（不需要服务器）

单元测试位于 [`test/registered/unit/`](https://github.com/sgl-project/sglang/tree/main/test/registered/unit)，其组织结构与 `python/sglang/srt/` 源码树相对应。这些测试**无需**启动服务器或加载真实模型权重即可验证组件逻辑。
SGLang 使用 Python 内置的 [unittest](https://docs.python.org/3/library/unittest.html) 框架，并以 [pytest](https://docs.pytest.org/) 作为测试运行器。

**何时添加单元测试：** 如果你修改了 `python/sglang/srt/` 下的某个文件，请检查 `test/registered/unit/` 中是否存在相应的测试，并为你的更改添加覆盖。例如：

```
srt/mem_cache/radix_cache.py   →  unit/mem_cache/test_radix_cache.py
srt/sampling/sampling_params.py →  unit/sampling/test_sampling_params.py
```

**在本地运行单元测试：**

```bash
pytest test/registered/unit/ -v                # all unit tests
pytest test/registered/unit/mem_cache/ -v      # one module
```

**带覆盖率运行：**

```bash
pytest test/registered/unit/ --cov --cov-config=.coveragerc -v
```

关于 CI 注册、测试结构和示例的约定，请参阅 [`test/registered/unit/README.md`](https://github.com/sgl-project/sglang/tree/main/test/registered/unit/README.md)。

### E2E 测试（需要服务器）

对于需要启动服务器的测试，请参阅 [`test/registered/README.md`](https://github.com/sgl-project/sglang/tree/main/test/registered/README.md) 以获得关于放置你的测试位置的指导。

有关运行测试并将其集成到 CI 的详细说明，请参阅 [test/README.md](https://github.com/sgl-project/sglang/tree/main/test/README.md)。

## 编写文档

我们建议新贡献者从编写文档开始，这有助于你快速理解 SGLang 代码库。
更多详情，请参阅 [docs/README.md](https://github.com/sgl-project/sglang/tree/main/docs/README.md)。

## 测试准确率
如果你的代码更改了模型输出，请运行准确率测试。一个快速的合理性检查是 few-shot GSM8K。

```
# Launch a server
python3 -m sglang.launch_server --model Qwen/Qwen2-7B-Instruct

# Evaluate
python3 -m sglang.test.few_shot_gsm8k --num-questions 200
```

请注意，上述脚本主要是一个合理性检查，而不是严格的准确率或速度测试。
由于批处理和推理引擎的非确定性本质，该测试的准确率可能存在显著的方差（1%–5%）。
此外，不要依赖该脚本输出的 "Latency/Output throughput"，因为它不是一个正规的速度测试。

如今对于最先进的模型来说，GSM8K 太简单了。请尝试你自己的更具挑战性的准确率测试。
你可以在以下位置找到更多的准确率评估示例：
- [test_eval_accuracy_large.py](https://github.com/sgl-project/sglang/blob/main/test/registered/eval/test_eval_accuracy_large.py)
- [test_gpt_oss_1gpu.py](https://github.com/sgl-project/sglang/blob/main/test/registered/core/test_gpt_oss_1gpu.py)

## Benchmark 速度
参考 [Benchmark and Profiling](../developer_guide/benchmark_and_profiling.md)。

## 请求审查以合并
你可以遵循 [MAINTAINER.md](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md) 中描述的 pull request 合并流程。
你需要与 Merge Oncall、Codeowner 以及其他审查者合作以获得他们的批准。
然后你的 PR 就可以被合并了。

## 如何触发 CI 测试

我们有大量开放的 PR 但 CI 机器有限，因此只有顶级和受信任的贡献者才有权限触发 CI 测试。
拥有权限的用户列在 [CI_PERMISSIONS.json](https://github.com/sgl-project/sglang/blob/main/.github/CI_PERMISSIONS.json) 中。

**PR 作者**始终可以在自己的 PR 上使用 `/rerun-failed-ci`，即使他们未列在 `CI_PERMISSIONS.json` 中。

要使 CI 在 pull request 上运行，它必须带有 "run-ci" 标签。授权用户可以通过在 PR 上评论以下命令之一来添加标签或重新运行失败的测试：

- `/tag-run-ci-label`：添加 "run-ci" 标签。之后的每次提交都会触发 CI。
- `/rerun-failed-ci`：重新运行最近一次提交中失败或不稳定（flaky）的测试。
- `/tag-and-rerun-ci`：一个同时执行 `/tag-run-ci-label` 和 `/rerun-failed-ci` 的命令。
- `/rerun-stage <stage-name>`：重新运行特定的测试 stage，而无需等待其依赖项。当你想快速验证某个特定测试失败的修复，而不是等待约 30 分钟让前置 stage 完成时，这很有用。

如果你有权限，[Slash Command Handler](https://github.com/sgl-project/sglang/actions/workflows/slash-command-handler.yml) 将运行你的命令并对你的评论作出 👍 反应。反应出现可能需要几分钟。这是一个使用[示例](https://github.com/sgl-project/sglang/pull/14253#issuecomment-3599509302)。

为避免用过多的 `/rerun-failed-ci` 评论刷屏 PR，你也可以通过编辑现有评论并添加任意后缀（例如 `/rerun-failed-ci try again`）来触发该命令。

重新运行单个测试 stage 的示例：`/rerun-stage unit-test-backend-4-gpu`。

如果你没有权限并且你不是 PR 作者，请让维护者为你触发 CI。

### CI 速率限制

由于 CI 调度和资源有限，更高优先级的 PR 可能会抢占正在运行的作业。在这种情况下，你可能需要重新运行测试。
我们应用 CI 速率限制来防止滥用并确保对我们 CI 资源的公平使用。

每个 CI 工作流在其工作流配置文件中都定义了一个默认限制。例如，在 [pr-gate.yml](https://github.com/sgl-project/sglang/blob/main/.github/workflows/pr-gate.yml) 中，默认冷却期为 120 分钟，每个工作流可以通过 `cool-down-minutes` 输入参数覆盖它：

```yaml
cool-down-minutes:
  description: "Default cooldown period in minutes; 0 disables rate limiting"
  type: number
  default: 120
```

列在 [CI_PERMISSIONS.json](https://github.com/sgl-project/sglang/blob/main/.github/CI_PERMISSIONS.json) 中的用户可能有一个每用户的冷却间隔。实际操作中，我们取工作流默认窗口和用户特定间隔两者中的最小值。

## 代码风格指导
- 避免代码重复。如果同一段代码（超过五行）多次出现，请将其提取为一个共享函数。
- 最小化设备同步。尽可能减少昂贵的 CPU-GPU 同步操作，例如 `tensor.item()` 或 `tensor.cpu()`。使用向量化代码。
- 优先考虑极致效率。SGLang 是一个运行时，你的大部分代码都在每个请求的关键路径上运行。尽可能优化所有微小的开销，尤其是在模型前向代码中。
  - 一个常见的模式是在模型前向传递中进行一些运行时检查（例如[这个](https://github.com/sgl-project/sglang/blob/f1b0eda55c2c4838e8ab90a0fac7fb1e3d7064ab/python/sglang/srt/models/deepseek_v2.py#L486-L491)）。这些检查对每一层很可能都是相同的。请尽可能将结果缓存为单个布尔值。
- 让函数尽可能纯粹。避免就地修改参数。
- 保持文件简洁。如果一个文件超过 2,000 行代码，将其拆分为多个较小的文件。（例如 `scheduler.py`、`scheduler_output_processor_mixin.py`）
- 保持测试运行快速。
  - 如果单个测试文件运行时间超过 500 秒，将其拆分为多个较小的文件（例如 `test_eagle_infer_a.py`、`test_eagle_infer_b.py`）。
  - 如果 github 工作流中的单个作业运行时间超过 30 分钟，将其拆分为更小的作业/步骤。
  - 在你的单元测试中复用服务器启动，以使测试运行更快。
- 切勿使用 `pickle.loads()`、`pickle.load()` 或 `recv_pyobj()` 来反序列化不受信任或从网络接收的数据。Python 的 [pickle 模块并不安全](https://docs.python.org/3/library/pickle.html) —— 它在反序列化期间可能执行任意代码。请使用安全的序列化格式，例如 [msgpack](https://github.com/jcrist/msgspec) 或 JSON。
- 在支持新硬件或功能时，请遵循以下准则：
  - 不要大幅更改现有代码。
  - 始终优先用新文件为你的新硬件引入特定组件（例如 `allocator_ascend.py`）。
  - 如果你为新功能编写多个 if/else 块，请确保公共路径（例如 NVIDIA 硬件或现有代码路径）是第一个分支。

## 如何更新 sgl-kernel
由于 sglang 和 `sglang-kernel`（之前的 `sgl-kernel`）发行版是独立的 Python 包，我们当前的 GitHub CI 基础设施不支持在同一个 pull request（PR）中更新一个 kernel 并立即使用它。
要在 `sgl-kernel/` 源码树中添加新 kernel 或修改现有 kernel，你必须使用多个 PR。

请遵循以下步骤：

1. 提交一个 PR 来更新 sgl-kernel 源代码，但不在 sglang python 包中使用它（例如 [#8884](https://github.com/sgl-project/sglang/pull/8884/files)）。
2. 提升 kernel 包的版本（例如 [#9220](https://github.com/sgl-project/sglang/pull/9220/files)）。
   - 一旦合并，这将触发 `sglang-kernel` wheel 到 PyPI 的自动发布。
   - 如果不紧急，你可以等待其他人发布 wheel。新版本通常会在一周内发布。
3. 应用更改：
   - 更新 `sglang/python/pyproject.toml` 中的 `sglang-kernel` 版本以使用修改后的 kernel。
   - 更新 sglang 中相关的调用代码以使用新 kernel。

## 给新手的提示

如果你想贡献但还没有具体的想法，可以挑选标记为 [“good first issue” 或 “help wanted”](https://github.com/sgl-project/sglang/issues?q=is%3Aissue+label%3A%22good+first+issue%22%2C%22help+wanted%22) 的 issue。这些任务通常复杂度较低，是了解代码库的绝佳入门。

也可以查看以下材料作为入门指南：
- [Mini-SGLang](https://github.com/sgl-project/mini-sglang)，快速了解 sglang 的结构。
- [Code Walk-through](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main/sglang/code-walk-through)，更深入地了解 SGLang 的工作流程。
- [GTC-2026 Training Lab](https://drive.google.com/file/d/1mwOZEtipNLJzrflCTodj34KhuOZEoEw5/view?usp=drive_link)，关于如何在已启动的 SGLang 实例上进行优化、benchmark 或 profiling 的动手实践。

如果你有任何问题或想发起讨论，请随时在我们的 [Slack channel](https://slack.sglang.io) 中提问。

感谢你对 SGLang 的关注。祝编码愉快！
