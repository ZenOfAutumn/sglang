# 故障排查与常见问题

## 故障排查

本页列出了常见错误及解决它们的技巧。

### CUDA 显存不足(Out of Memory)
如果你遇到显存不足(OOM)错误,可以调整以下参数:

- 如果 OOM 发生在 prefill 阶段,尝试将 `--chunked-prefill-size` 减小到 `4096` 或 `2048`。这能节省内存,但会降低长 prompt 的 prefill 速度。
- 如果 OOM 发生在解码(decoding)阶段,尝试降低 `--max-running-requests`。
- 你也可以将 `--mem-fraction-static` 减小到更小的值,例如 0.8 或 0.7。这会减少 KV cache 内存池的内存占用,有助于防止 prefill 和解码阶段的 OOM 错误。然而,它会限制最大并发量并降低峰值吞吐量。
- 另一种常见的 OOM 情况是为长 prompt 请求输入的 logprobs,因为这需要大量内存。要解决这个问题,在采样参数中设置 `logprob_start_len`,以仅包含必要的部分。如果你确实需要长 prompt 的输入 logprobs,可以尝试减小 `--mem-fraction-static`。

### CUDA 错误:遇到非法内存访问(Illegal Memory Access Encountered)
这个错误可能由内核错误或显存不足问题导致:
- 如果是内核错误,解决起来可能比较困难。请在 GitHub 上提交一个 issue。
- 如果是显存不足问题,有时它会以这个错误的形式报告,而不是 "Out of Memory"。请参阅上面的章节获取避免 OOM 问题的指导。

### 服务器卡住
- 如果服务器在初始化或运行期间卡住,可能是内存问题(显存不足)、网络问题(nccl 错误),或者 sglang 中的其他 bug。
    - 如果是显存不足,你可能会看到在初始化期间或初始化后立即 `avail mem` 非常低。在这种情况下,
      你可以尝试减小 `--mem-fraction-static`、减小 `--cuda-graph-max-bs`,或减小 `--chunked-prefill-size`。
- 其他 bug,请在 GitHub 上提交 issue。


## 常见问题

### 即使温度为 0,结果也不是确定性的

你可能会注意到,当你两次发送相同的请求时,引擎返回的结果会略有不同,即使温度被设置为 0。

根据我们的初步调查,这种不确定性源于两个因素:动态批处理(dynamic batching)和前缀缓存(prefix caching)。粗略地说,动态批处理约占不确定性的 95%,而前缀缓存占剩余部分。服务器在底层运行动态批处理。不同的批大小会导致 PyTorch/CuBLAS 调度到不同的 CUDA 内核,这可能导致细微的数值差异。这种差异在许多层之间累积,导致批大小变化时产生不确定的输出。类似地,当启用前缀缓存时,它也可能调度到不同的内核。即使计算在数学上是等价的,不同内核实现产生的微小数值差异也会导致最终的不确定输出。

要在当前代码中获得更具确定性的输出,你可以添加 `--disable-radix-cache` 并且每次只发送一个请求。在此设置下,结果将大体上是确定性的。

**更新**:
最近,我们还引入了确定性模式,你可以使用 `--enable-deterministic-inference` 启用它。
请在这篇博文中查找更多细节:https://lmsys.org/blog/2025-09-22-sglang-deterministic/
