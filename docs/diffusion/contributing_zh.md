# 为 SGLang Diffusion 做贡献

本指南概述了为 SGLang Diffusion 模块(`sglang.multimodal_gen`)做贡献的要求。

## 贡献者指南

- [Support New Models](support_new_models.md):添加新 diffusion pipeline 的实现指南
- [CI Performance](ci_perf.md):更新和重新生成性能基线

```{toctree}
:maxdepth: 1

support_new_models
ci_perf
```

## 关于 AI 辅助("Vibe Coding")PR

我们欢迎 vibe-coded 的 PR —— 我们评判的是代码质量,而非它是如何产生的。所有 PR 的标准是一致的:

- **不要过度注释。** 如果名称已经说明了一切,就跳过 docstring。
- **不要过度捕获异常。** 不要去防御那些在实践中几乎不会发生的错误。
- **提交前先测试。** AI 生成的代码可能存在细微错误 —— 请端到端地验证其正确性。

## 提交信息约定

我们遵循一种结构化的提交信息格式,以保持干净的历史记录。

**格式:**
```text
[diffusion] <scope>: <subject>
```

**示例:**
- `[diffusion] cli: add --perf-dump-path argument`
- `[diffusion] scheduler: fix deadlock in batch processing`
- `[diffusion] model: support Stable Diffusion 3.5`

**规则:**
- **前缀**:始终以 `[diffusion]` 开头。
- **Scope**(可选):`cli`、`scheduler`、`model`、`pipeline`、`docs` 等。
- **Subject**:使用祈使语气,简短清晰(例如使用 "add feature" 而非 "added feature")。

## 性能报告

对于影响 **latency**、**throughput** 或 **memory usage** 的 PR,你**应当**提供一份性能对比报告。

### 如何生成报告

1.  **基线(Baseline)**:运行基准测试(针对单个生成任务)
    ```bash
    $ sglang generate --model-path <model> --prompt "A benchmark prompt" --perf-dump-path baseline.json
    ```

2.  **新版(New)**:运行相同的基准测试,不修改任何 server_args 或 sampling_params
    ```bash
    $ sglang generate --model-path <model> --prompt "A benchmark prompt" --perf-dump-path new.json
    ```

3.  **对比(Compare)**:运行对比脚本,它会向控制台打印一个 Markdown 表格
    ```bash
    $ python python/sglang/multimodal_gen/benchmarks/compare_perf.py baseline.json new.json [new2.json ...]
    ### Performance Comparison Report
    ...
    ```
4. **粘贴(Paste)**:将该表格粘贴到 PR 描述中

## 基于 CI 的变更保护

考虑向 `pr-test` 或 `nightly-test` 套件添加测试,以保护你的改动,尤其是对于以下类型的 PR:

- 支持一个新模型
    - 为这个新模型在 `testcase_configs.py` 中添加一个测试用例
- 支持或修复重要特性
- 显著提升性能

请运行相应的测试用例,然后在适用的情况下,按照控制台中的说明更新/添加基线到 `perf_baselines.json`。

参见 [test](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/test) 获取示例
