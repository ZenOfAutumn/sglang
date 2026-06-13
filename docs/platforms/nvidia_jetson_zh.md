# NVIDIA Jetson Orin

## 前置条件

开始之前，请确保满足以下条件：

- [**NVIDIA Jetson AGX Orin Devkit**](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) 已安装 **JetPack 6.1** 或更高版本。
- 已安装 **CUDA Toolkit** 和 **cuDNN**。
- 确认 Jetson AGX Orin 处于**高性能模式（high-performance mode）**：
```bash
sudo nvpmodel -m 0
```
* * * * *
## 使用 Jetson Containers 安装并运行 SGLang
克隆 jetson-containers 的 GitHub 仓库：
```
git clone https://github.com/dusty-nv/jetson-containers.git
```
运行安装脚本：
```
bash jetson-containers/install.sh
```
构建容器镜像：
```
jetson-containers build sglang
```
运行容器：
```
jetson-containers run $(autotag sglang)
```
你也可以使用以下命令手动运行容器：
```
docker run --runtime nvidia -it --rm --network=host IMAGE_NAME
```
* * * * *

运行推理
-----------------------------------------

启动服务器：
```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --device cuda \
  --dtype half \
  --attention-backend flashinfer \
  --mem-fraction-static 0.8 \
  --context-length 8192
```
之所以进行量化并限制上下文长度（`--dtype half --context-length 8192`），是由于 [Nvidia jetson kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) 的算力资源有限。详细解释参见 [Server Arguments](../advanced_features/server_arguments.md)。

启动引擎后，参考 [Chat completions](https://docs.sglang.io/basic_usage/openai_api_completions.html#Usage) 测试其可用性。
* * * * *
使用 TorchAO 运行量化
-------------------------------------
建议在 NVIDIA Jetson Orin 上使用 TorchAO。
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --device cuda \
    --dtype bfloat16 \
    --attention-backend flashinfer \
    --mem-fraction-static 0.8 \
    --context-length 8192 \
    --torchao-config int4wo-128
```
这会启用 TorchAO 的 int4 weight-only 量化，分组大小为 128。使用 `--torchao-config int4wo-128` 同样是为了提升内存效率。


* * * * *
使用 XGrammar 进行结构化输出
-------------------------------
请参考 [SGLang doc structured output](../advanced_features/structured_outputs.ipynb)。
* * * * *

感谢 [Nurgaliyev Shakhizat](https://github.com/shahizat)、[Dustin Franklin](https://github.com/dusty-nv) 和 [Johnny Núñez Cano](https://github.com/johnnynunez) 提供的支持。

参考资料
----------
-   [NVIDIA Jetson AGX Orin Documentation](https://developer.nvidia.com/embedded/jetson-agx-orin)
