# 搭载 Metal 的 Apple Silicon

本文档介绍如何使用 [Metal](https://developer.apple.com/metal/) 在 Apple Silicon 上运行 SGLang。如果你遇到问题或有疑问，请 [open an issue](https://github.com/sgl-project/sglang/issues)。

## 安装 SGLang

你可以使用下列方法之一安装 SGLang。

### 从源码安装

```bash
# Use the default branch
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Install sglang python package
pip install --upgrade pip
rm -f python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
uv pip install -e "python[all_mps]"
```
