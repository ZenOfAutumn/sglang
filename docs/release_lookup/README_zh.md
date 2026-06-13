# SGLang Release 查找工具

此工具允许用户查找包含特定 PR 或 commit 的最早 release。
它完全在浏览器中运行，使用从 git 历史生成的静态 JSON 索引。

## 用法

1. **生成索引**：
   运行 Python 脚本，从你的本地 git 仓库生成 `release_index.json` 文件。

   ```bash
   python3 generate_index.py --output release_index.json
   ```

   该脚本会：
   - 查找所有匹配 `v*` 和 `gateway-v*` 的 tag。
   - 按创建日期对它们排序。
   - 遍历历史，找出哪个 release 首次引入了每个 commit 和 PR。
   - 从 commit 消息中提取 PR 编号。

2. **打开工具**：
   在你的浏览器中打开 `index.html`。

   ```bash
   # You can open it directly if your browser supports local file fetch (Firefox usually does),
   # or serve it locally:
   python3 -m http.server
   # Then go to http://localhost:8000/index.html
   ```

## 文件

- `index.html`：查找工具的 UI。
- `generate_index.py`：用于构建索引的脚本。
- `release_index.json`：UI 使用的索引文件。

## 逻辑

该工具基于 tag 创建日期确定“最早的 release”。它从最旧到最新遍历各个 tag。任何从某个 tag 可达（而从前一个 tag 不可达）的 commit 都会被分配给该 release。
