# SGLang 文档

这是 SGLang 项目（https://github.com/sgl-project/sglang）的文档网站。

我们建议新贡献者从编写文档开始，这有助于你快速理解 SGLang 代码库。
大多数文档文件位于 `docs/` 目录下。

## 文档工作流

### 安装依赖

**Linux：**
```bash
apt-get update && apt-get install -y pandoc parallel retry
pip install -r requirements.txt
```

**macOS：**
```bash
brew install pandoc parallel retry
pip install -r requirements.txt
```

### 更新文档

在 `docs/` 下相应的子目录中更新你的 Jupyter notebook。如果你新增了文件，记得同步更新 `index.rst`（或相关的 `.rst` 文件）。

- **`pre-commit run --all-files`** 会手动运行所有已配置的检查，并在可能时自动修复问题。如果第一次运行失败，请再次运行以确保 lint 错误已完全解决。在创建 Pull Request **之前**，请确保你的代码通过所有检查。

```bash
# 1) 编译所有 Jupyter notebook
make compile  # 这一步可能耗时较长（10 分钟以上）。如果你能确保新增的文件是正确的，可以考虑跳过此步骤。
make html

# 2) 在本地编译并预览文档，支持自动构建
# 当文件发生变化时会自动重新构建文档
# 在浏览器中打开显示的端口即可查看文档
bash serve.sh

# 2a) 提供文档服务的其他方式
# 直接使用 make serve
make serve
# 使用自定义端口
PORT=8080 make serve

# 3) 清理 notebook 输出
# nbstripout 会移除 notebook 输出，以保持你的 PR 干净整洁
pip install nbstripout
find . -name '*.ipynb' -exec nbstripout {} \;

# 4) 执行 pre-commit 检查并创建 PR
# 这些检查通过后，推送你的更改并在你的分支上发起 PR
pre-commit run --all-files
```

## 文档风格指南

- 对于常用功能，我们更倾向于使用 **Jupyter Notebook** 而非 Markdown，这样所有示例都可以由我们的文档 CI 流水线执行和验证。对于复杂特性（例如分布式部署），则更推荐使用 Markdown。
- 编写交互式 Jupyter notebook 时请留意文档的执行时间。每个交互式 notebook 都会针对每次提交被运行和编译，以确保它们可正常运行，因此应用一些技巧来减少文档编译时间非常重要：
  - 大多数情况下使用小模型（例如 `qwen/qwen2.5-0.5b-instruct`）以减少服务器启动时间。
  - 尽可能复用已启动的服务器以减少服务器启动时间。
- 不要使用绝对链接（例如 `https://docs.sglang.io/get_started/install.html`）。请始终优先使用相对链接（例如 `../get_started/install.md`）。
- 参考现有示例来学习如何启动服务器、发送查询以及其他常见写法。

## 文档构建、部署与 CI

SGLang 文档流水线基于 **Sphinx**，支持将 Jupyter notebook（`.ipynb`）渲染为 HTML/Markdown 以供网页展示。详细逻辑可参见 [Makefile](./Makefile)。

### Notebook 执行（`make compile`）

`make compile` 目标负责在渲染之前执行 notebook：

* 查找 `docs/` 下的所有 `.ipynb` 文件（排除 `_build/`）
* 使用 GNU Parallel 并行执行 notebook，并采用相对较小的 `--mem-fraction-static`
* 用 `retry` 包装执行过程，以减少偶发性失败
* 通过 `jupyter nbconvert --execute --inplace` 执行 notebook
* 在 `logs/timing.log` 中记录执行耗时

这一步确保在渲染之前，主分支的每次提交中 notebook 都包含最新的输出。

### 网页渲染（`make html`）

编译完成后，Sphinx 构建网站：

* 读取 Markdown、reStructuredText 和 Jupyter notebook
* 将它们渲染为 HTML 页面
* 将网站输出到：

```
docs/_build/html/
```

该目录是在线文档托管的来源。

### Markdown 导出（`make markdown`）

为支持下游使用方，我们新增了一个 **新的 Makefile 目标**：

```bash
make markdown
```

该目标：

* **不会修改** `make compile`
* 扫描所有 `.ipynb` 文件（排除 `_build/`）
* 使用 `jupyter nbconvert --to markdown` 将 notebook 直接转换为 Markdown
* 将 Markdown 产物写入现有的构建目录：

```
docs/_build/html/markdown/<relative-path>.md
```

示例：

```
docs/advanced_features/lora.ipynb
→ docs/_build/html/markdown/advanced_features/lora.md
```

### CI 执行

在我们的 [CI](https://github.com/sgl-project/sglang/blob/main/.github/workflows/release-docs.yml) 中，文档流水线首先获取所有已执行的结果，然后通过以下方式渲染 HTML 和 Markdown：

```bash
make compile    # 执行 notebook（确保输出是最新的）
make html       # 像往常一样构建网站
make markdown   # 将 markdown 产物导出到 _build/html/markdown
```

随后，编译后的结果会被强制推送到 [sgl-project.io](https://github.com/sgl-project/sgl-project.github.io) 进行渲染。换句话说，sgl-project.io 是仅推送（push-only）的。所有 SGLang 文档的改动都应直接在 SGLang 主仓库中进行，然后再推送到 sgl-project.io。
