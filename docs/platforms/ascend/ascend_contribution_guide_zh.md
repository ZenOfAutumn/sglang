# 贡献指南

欢迎来到 **SGLang**!我们非常感谢你有兴趣参与贡献。本指南简要介绍了如何搭建环境、运行测试、构建文档以及提交 Pull Request(PR)。无论你是修复一个小 bug 还是开发一项重要功能,我们都鼓励你遵循这些步骤,以获得顺畅的贡献体验。

## 从源码安装 SGLang

### 准备环境

在贡献之前,请确保你的环境已正确搭建。按照[安装指南](ascend_npu.md)中的步骤安装必要的依赖。我们推荐[使用 docker](ascend_npu.md#method-2-using-docker-image) 来构建环境。

### Fork 并克隆仓库

**注意**:新贡献者**没有**推送到 SGLang 官方仓库的写权限。请在你的 GitHub 账号下 fork 该仓库,然后将你的 fork 克隆到本地。

```bash
git clone https://github.com/<your_user_name>/sglang.git
# if you are using docker, the environment is already set up.
cd sglang
export PYTHONPATH=$PWD/python:$PYTHONPATH
```

## 使用 pre-commit 格式化代码

我们使用 [pre-commit](https://pre-commit.com/) 来维护一致的代码风格检查。在推送你的更改之前,请运行:

```bash
pip3 install pre-commit
pre-commit install
pre-commit run --all-files
```

- **`pre-commit run --all-files`** 会手动运行所有已配置的检查,并在可能的情况下应用修复。如果第一次失败,请重新运行以确保 lint 错误完全解决。在创建 Pull Request **之前**,请确保你的代码通过了所有检查。
- **不要**直接向 `main` 分支提交。务必创建一个新分支(例如 `feature/my-new-feature`),推送你的更改,并从该分支发起 PR。

## 运行并添加单元测试

如果你新增了功能或修复了 bug,请添加相应的单元测试,以确保覆盖率并防止回归。
SGLang 使用 Python 内置的 [unittest](https://docs.python.org/3/library/unittest.html) 框架。
关于运行测试以及将其集成到 CI 的详细说明,请参阅 [test/README.md](https://github.com/sgl-project/sglang/tree/main/test/README.md)。

如果你需要使用 ```python/sglang/test/ascend/test_ascend_utils.py`` 列表中没有的模型,请按以下步骤操作:
1. 注册账号并将你的模型上传到 [modelscope](https://modelscope.cn/models)。
2. 确保你的模型已在 CI 服务器上预缓存,并位于 "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/{your_model_repo}/{your_model}" 路径下。
如果不是这种情况,请在 CI 服务器上使用以下命令:
  ```bash
  modelscope download
  --model {your_model_repo}/{your_model}
  --local_dir /data/ascend-ci-share-pkking-sglang/modelscope/hub/models/{your_model_repo}/{your_model}
  ```
  > 注意:如果你没有 CI 服务器的访问权限,请联系维护者(zl19940307@163.com)下载你的模型。
4. 将模型添加到 ```python/sglang/test/ascend/test_ascend_utils.py```(在 docker 中使用 ```"/root/.cache/modelscope/hub/models/{your_model_repo}/{your_model}"``` 路径)。

## 编写文档

我们建议新贡献者从编写文档开始,这有助于你快速了解 SGLang 代码库。
更多细节请参阅 [docs/README.md](https://github.com/sgl-project/sglang/tree/main/docs/README.md)。

## 测试准确性
如果你的代码改变了模型输出,请运行准确性测试。一个快速的健全性检查是 few-shot GSM8K。

```
# Launch a server
python3 -m sglang.launch_server --model Qwen/Qwen2-7B-Instruct

# Evaluate
python3 -m sglang.test.few_shot_gsm8k --num-questions 200
```

请注意,上述脚本主要是一个健全性检查,而非严格的准确性或速度测试。
由于 batching 以及推理引擎的非确定性本质,该测试在准确性上可能存在显著的方差(1%–5%)。
此外,不要依赖该脚本输出的 "Latency/Output throughput",因为它并不是一个合适的速度测试。

如今 GSM8K 对最先进的模型来说太简单了。请尝试你自己的更具挑战性的准确性测试。
你可以在以下文件中找到更多准确性评测示例:
- [test_eval_accuracy_large.py](https://github.com/sgl-project/sglang/blob/main/test/registered/eval/test_eval_accuracy_large.py)
- [test_moe_eval_accuracy_large.py](https://github.com/sgl-project/sglang/blob/main/test/registered/eval/test_moe_eval_accuracy_large.py)

## 测试速度
请参阅 [Benchmark and Profiling](../../developer_guide/benchmark_and_profiling.md)。

## 请求合并评审
你可以遵循 [MAINTAINER.md](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md) 中描述的 pull request 合并流程。
你需要与 Merge Oncall、Codeowner 以及其他评审者协作以获得他们的批准。
然后你的 PR 才能被合并。

## 如何触发 CI 测试

我们有大量待处理的 PR,但 CI 机器有限,因此只有顶级且受信任的贡献者才有权限触发 CI 测试。
拥有权限的用户列在 [CI_PERMISSIONS.json](https://github.com/sgl-project/sglang/blob/main/.github/CI_PERMISSIONS.json) 中

要让 CI 在某个 pull request 上运行,该 PR 必须带有 "run-ci" 标签。被授权的用户可以添加该标签,或通过在 PR 上评论以下命令之一来重新运行失败的测试:

- `/tag-run-ci-label`:添加 "run-ci" 标签。之后的每次提交都将触发 CI。
- `/rerun-failed-ci`:重新运行最近一次提交中失败或不稳定(flaky)的测试。
- `/tag-and-rerun-ci`:一个命令同时执行 `/tag-run-ci-label` 和 `/rerun-failed-ci`。
- `/rerun-stage <stage-name>`:重新运行特定测试阶段,而无需等待其依赖项。当你想快速验证某个特定测试失败的修复,而不愿等待前置阶段完成约 30 分钟时,这非常有用。

如果你有权限,[Slash Command Handler](https://github.com/sgl-project/sglang/actions/workflows/slash-command-handler.yml) 将运行你的命令,并对你的评论作出 👍 的反应。该反应可能需要几分钟才会出现。这里有一个使用[示例](https://github.com/sgl-project/sglang/pull/14253#issuecomment-3599509302)。

为避免用过多的 `/rerun-failed-ci` 评论刷屏某个 PR,你也可以通过编辑一条已有评论并添加任意后缀(例如 `/rerun-failed-ci try again`)来触发该命令。

重新运行单个测试阶段的示例:`/rerun-stage unit-test-backend-4-gpu`。

如果你没有权限,请联系维护者为你触发 CI。

### CI 速率限制

由于 CI 调度和资源有限,优先级更高的 PR 可能会抢占正在运行的任务。在这种情况下,你可能需要重新运行测试。

我们采用 CI 速率限制来防止滥用,并确保公平使用我们的 CI 资源。

每个 CI 工作流在其工作流配置文件中都定义了默认限制。例如,在 [pr-gate.yml](https://github.com/sgl-project/sglang/blob/main/.github/workflows/pr-gate.yml) 中,默认冷却时间为 120 分钟,每个工作流都可以通过 `cool-down-minutes` 输入参数来覆盖它:

```yaml
cool-down-minutes:
  description: "Default cooldown period in minutes; 0 disables rate limiting"
  type: number
  default: 120
```

列在 [CI_PERMISSIONS.json](https://github.com/sgl-project/sglang/blob/main/.github/CI_PERMISSIONS.json) 中的用户可能拥有按用户设定的冷却间隔。实际操作中,我们取工作流默认窗口与用户特定间隔中的最小值。

## 代码风格指南
- 避免代码重复。如果同一段代码(超过五行)出现多次,请将其提取为一个共享函数。
- 尽量减少设备同步。尽可能减少昂贵的 CPU-GPU 同步操作,例如 `tensor.item()` 或 `tensor.cpu()`。使用向量化代码。
- 优先追求极致效率。SGLang 是一个 runtime,你的大部分代码都运行在每个请求的关键路径上。请尽可能优化所有微小开销,尤其是在模型前向(forward)代码中。
  - 一个常见模式是在模型前向传递中进行一些运行时检查(例如[这个](https://github.com/sgl-project/sglang/blob/f1b0eda55c2c4838e8ab90a0fac7fb1e3d7064ab/python/sglang/srt/models/deepseek_v2.py#L486-L491))。这些检查对每一层很可能都是相同的。请尽可能将结果缓存为单个布尔值。
- 尽量使函数保持纯函数。避免对参数进行原地(in-place)修改。
- 保持文件简洁。如果一个文件超过 2,000 行代码,请将其拆分为多个较小的文件。(例如 `scheduler.py`、`scheduler_output_processor_mixin.py`)
- 保持测试运行快速。
  - 如果单个测试文件运行时间超过 500 秒,请将其拆分为多个较小的文件(例如 `test_eagle_infer_a.py`、`test_eagle_infer_b.py`)。
  - 如果 github 工作流中的单个 job 运行时间超过 30 分钟,请将其拆分为更小的 jobs/steps。
  - 在你的单元测试中复用服务启动,以加快测试运行速度。
- 在支持新硬件或新功能时,请遵循以下准则:
  - 不要大幅改动现有代码。
  - 始终优先用新文件来为你的新硬件引入特定组件(例如 `allocator_ascend.py`)。
  - 如果你为新功能编写多个 if/else 代码块,请确保公共路径(例如 NVIDIA 硬件或现有代码路径)是第一个分支。

## 如何更新 sgl-kernel
由于 sglang 和 sgl-kernel 是独立的 Python 包,我们当前的 GitHub CI 基础设施不支持在同一个 pull request(PR)内更新一个 kernel 并立即使用它。
要在 `sgl-kernel/` 源码树中添加新 kernel 或修改现有 kernel,你必须使用多个 PR。

请按以下步骤操作:

1. 提交一个 PR 来更新 sgl-kernel 源码,而不在 sglang python 包中使用它(例如 [#8884](https://github.com/sgl-project/sglang/pull/8884/files))。
2. 提升 kernel 包的版本(例如 [#9220](https://github.com/sgl-project/sglang/pull/9220/files))。
   - 一旦合并,这将触发 `sglang-kernel` wheel 自动发布到 PyPI。
   - 如果不紧急,你可以等待其他人发布 wheel。通常一周内会发布一个新版本。
3. 应用更改:
   - 更新 `sglang/python/pyproject.toml` 中的 `sglang-kernel` 版本,以使用修改后的 kernels。
   - 更新 sglang 中相关的调用方代码,以使用新的 kernel。

## 如何更新 sgl-kernel-npu

Sgl-kernel-npu 是 Ascend NPU 的 kernel 包,维护在 [sgl-kernel-npu](https://github.com/sgl-project/sgl-kernel-npu) 仓库中。如果你想添加新 kernel 并在 sglang 中使用它,请遵循[贡献指南](https://github.com/sgl-project/sgl-kernel-npu/blob/main/docs/developer_guide/contribution_guide.md)中的步骤。

## 给新手的提示

如果你想贡献但没有具体的想法,可以选择标记为[“good first issue” 或 “help wanted”](https://github.com/sgl-project/sglang/issues?q=is%3Aissue+label%3A%22good+first+issue%22%2C%22help+wanted%22) 的 issue。这些任务通常复杂度较低,是了解代码库的绝佳入门。也可以查看这个[代码导览](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main/sglang/code-walk-through),以更深入地了解 SGLang 的工作流程。

如果你有任何问题或想发起讨论,欢迎随时在我们的 [Slack 频道](https://slack.sglang.io)提问。

感谢你对 SGLang 的关注。祝编码愉快!
