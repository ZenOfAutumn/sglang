# 摩尔线程 GPU

本文档介绍如何在摩尔线程（Moore Threads）GPU 上运行 SGLang。如果你遇到问题或有疑问，请 [open an issue](https://github.com/sgl-project/sglang/issues)。

## 安装 SGLang

你可以使用下列方法之一安装 SGLang。

### 从源码安装

```bash
# Use the default branch
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Compile sgl-kernel
pip install --upgrade pip
cd sgl-kernel
python setup_musa.py install

# Install sglang python package
cd ..
rm -f python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_musa]"
```
