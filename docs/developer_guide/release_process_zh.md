# PyPI 软件包发布流程

## 更新代码中的版本号
更新 `python/pyproject.toml` 和 `python/sglang/__init__.py` 中的软件包版本号。

## 上传 PyPI 软件包

```
pip install build twine
```

```
cd python
bash upload_pypi.sh
```

## 在 GitHub 上发布 Release
在 https://github.com/sgl-project/sglang/releases/new 创建一个新的 release。
