# Ascend 上的量化

要加载已量化的模型,只需加载模型权重和配置即可。同样地,如果模型已离线量化,则在启动引擎时无需添加 `--quantization` 参数。量化方法将从下载的 `quant_model_description.json` 或 `config.json` 配置中自动解析。

SGLang 支持 **mix-bits** 量化(根据 `quant_model_description'.json` 中指定的量化类型,独立定义并加载每一层)。[MoE 的进阶 mix-bits](https://github.com/sgl-project/sglang/pull/17361) 正在进行中,将为 w13(up-gate)层和 w2(down)层添加独立的量化判定)。

[ModelSlim on Ascend support](https://github.com/sgl-project/sglang/pull/14504)
| Quantization scheme                                       | Layer type               |               A2 Supported               |               A3 Supported               |               A5 Supported                 |             Diffusion models               |
|-----------------------------------------------------------|--------------------------|:----------------------------------------:|:----------------------------------------:|:------------------------------------------:|:------------------------------------------:|
| W4A4 dynamic                                              | Linear                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>**  | **<span style="color: green;">√</span>**   |
| W8A8 static                                               | Linear                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>**  | **<span style="color: green;">√</span>**   |
| W8A8 dynamic                                              | Linear                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>**  | **<span style="color: green;">√</span>**   |
| [MXFP8](https://github.com/sgl-project/sglang/pull/20922) | Linear                   | **<span style="color: red;">x</span>**   | **<span style="color: red;">x</span>**   | **<span style="color: blue;">WIP</span>**  | **<span style="color: blue;">WIP</span>**  |
| W4A4 dynamic                                              | MoE                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>**  | **<span style="color: red;">x</span>**     |
| W4A8 dynamic                                              | MoE                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>**  | **<span style="color: red;">x</span>**     |
| W8A8 dynamic                                              | MoE                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>**  | **<span style="color: red;">x</span>**     |
| [MXFP8](https://github.com/sgl-project/sglang/pull/20922) | MoE                      | **<span style="color: red;">x</span>**   | **<span style="color: red;">x</span>**   | **<span style="color: blue;">WIP</span>**  | **<span style="color: red;">x</span>**     |

[AWQ on Ascend support](https://github.com/sgl-project/sglang/pull/10158):
| Quantization scheme            | Layer type               |               A2 Supported               |               A3 Supported               |               A5 Supported                 |
|--------------------------------|--------------------------|:----------------------------------------:|:----------------------------------------:|:------------------------------------------:|
| W4A16                          | Linear                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>**  |
| W8A16                          | Linear                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>**  |
| W4A16                          | MoE                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>**  |

GPTQ on Ascend support
| Quantization scheme                                                        | Layer type               |               A2 Supported               |               A3 Supported               |               A5 Supported                |
|----------------------------------------------------------------------------|--------------------------|:----------------------------------------:|:----------------------------------------:|:-----------------------------------------:|
| [W4A16](https://github.com/sgl-project/sglang/pull/15203)                  | Linear                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>** |
| [W8A16](https://github.com/sgl-project/sglang/pull/15203)                  | Linear                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>** |
| [W4A16 MOE](https://github.com/sgl-project/sglang/pull/16364)              | MoE                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>** |
| [W8A16 MOE](https://github.com/sgl-project/sglang/pull/16364)              | MoE                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>** |

[Auto-round on Ascend support](https://github.com/sgl-project/sglang/pull/16699)
| Quantization scheme            | Layer type               |               A2 Supported               |               A3 Supported               |               A5 Supported                |
|--------------------------------|--------------------------|:----------------------------------------:|:----------------------------------------:|:-----------------------------------------:|
| W4A16                          | Linear                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>** |
| W8A16                          | Linear                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>** |
| W4A16                          | MoE                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>** |
| W8A16                          | MoE                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>** |

Compressed-tensors (LLM Compressor) on Ascend support:
| Quantization scheme                                                                           | Layer type               |               A2 Supported               |               A3 Supported               |               A5 Supported                |
|-----------------------------------------------------------------------------------------------|--------------------------|:----------------------------------------:|:----------------------------------------:|:-----------------------------------------:|
| [W8A8 dynamic](https://github.com/sgl-project/sglang/pull/14504)                              | Linear                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>** |
| [W4A8 dynamic with/without activation clip](https://github.com/sgl-project/sglang/pull/14736) | MoE                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>** |
| [W4A16 MOE](https://github.com/sgl-project/sglang/pull/12759)                                 | MoE                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>** |
| [W8A8 dynamic](https://github.com/sgl-project/sglang/pull/14504)                              | MoE                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>** |

[GGUF on Ascend support](https://github.com/sgl-project/sglang/pull/17883)

进行中
