# HiCache 系统设计与优化

本文档全面概述了 SGLang HiCache,涵盖其系统架构、工作流和关键组件。它还详细介绍了配置参数、优化技术,以及与各类 L3 存储后端的集成,可作为用户和开发者理解并调优 HiCache 以实现高效 LLM 推理的完整参考。

## 为什么需要 HiCache?它是什么?

在大语言模型推理中,prefill 阶段通常很耗时:输入序列需要首先被转换为 Key-Value 缓存(KV cache)以供后续解码。当多个请求共享相同的前缀时,该前缀的 KV 缓存是相同的。通过缓存和复用这些共享的 KV 缓存,可以避免冗余计算。为解决这一问题,SGLang 引入了 RadixAttention,它利用空闲的 GPU 内存来缓存和复用前缀 KV 缓存;以及 **HiCache**,它将这一思想扩展到主机内存和分布式存储。

受现代 CPU 经典三级缓存设计的启发,HiCache 将 GPU 内存组织为 L1、主机内存组织为 L2、分布式存储组织为 L3。这种分层结构使 HiCache 能够充分利用 GPU 和 CPU 的"空闲"存储空间,同时集成 Mooncake、3FS、NIXL 和 AIBrix KVCache 等分布式缓存系统,实现全局 KV 缓存存储和调度。因此,HiCache 在保持强劲读取性能的同时,显著扩展了 KV 缓存容量——尤其是在 multi-QA 和长上下文推理这类 KV 缓存复用频繁的工作负载中。详细的基准测试结果请参见 [这篇博客](https://lmsys.org/blog/2025-09-10-sglang-hicache/)。


## 系统设计

### 整体架构

在许多现代 CPU 架构中,小而快的 L1 和 L2 缓存为每个核心私有,从而能够快速访问最热的数据;而更大的 L3 缓存则由所有核心共享,以显著减少缓存内的冗余。类似地,在 HiCache 中,L1 和 L2 KV 缓存为每个推理实例私有,而 L3 KV 缓存则在集群内所有推理实例之间共享。

### HiRadixTree:HiCache 中的元数据组织

对于 KV 缓存数据的组织,HiCache 在 RadixAttention 引入的 RadixTree 结构之上构建,并提出了 HiRadixTree。在 RadixAttention 中,RadixTree 的每个节点对应于 GPU 内存中一段连续 token 的 KV 缓存。从根节点到叶节点的一条路径表示一个请求的前缀,多个请求之间共享的前缀可以复用相同的节点,从而避免冗余存储。

HiRadixTree 扩展了这一思想:每个节点对应一段连续 token 的 KV 缓存,并记录该 KV 缓存的存储位置——无论是在本地 GPU 内存、CPU 内存、L3 存储,还是这些层级中的多个。如果存储在本地,HiRadixTree 会维护精确的元数据,包括确切的存储地址。然而,为了减少开销,HiRadixTree 不存储也不持续同步 L3 KV 缓存的元数据。相反,在访问 L3 数据时,它会实时查询后端以获取所需的元数据,例如数据是否存在以及它驻留在哪台服务器和哪个位置。

### 整体工作流

HiCache 的工作流主要涉及三个关键操作:**本地匹配(local match)**、**预取(prefetch)** 和 **回写(write-back)**。当系统收到新请求时,首先在本地 L1 和 L2 缓存中搜索匹配的 KV 缓存。对于本地未找到的部分,它尝试从 L3 预取。预取之后,所有所需的 KV 缓存被加载到 GPU 中进行计算。一旦 prefill 计算完成,系统会考虑将新生成的数据存储到 L2 或 L3。

![HiCache Workflow](https://lmsys.org/images/blog/hicache/hicache_overview.png)

### 本地匹配

本地匹配是 HiCache 工作流的第一步,在此步骤中,传入的请求 token 会与 HiRadixTree 进行匹配,以定位本地内存层级(L1 GPU 内存和 L2 主机内存)中已缓存的 KV 数据。

匹配算法从根节点开始遍历 HiRadixTree,沿着与 token 序列前缀匹配的子节点向下走。在每个节点处,传入的 token 序列会与该节点存储的 token 序列进行比较。当 `page_size > 1` 时,匹配以 page 粒度进行,以优化内存访问模式。如果匹配在某个节点存储序列的中间终止,该节点会被自动拆分以创建精确的边界,从而提升未来匹配的效率。

该算法返回请求的一个连续前缀,其中前一部分驻留在 L1,后一部分驻留在 L2。

由于该过程只需遍历本地 HiRadixTree,不涉及任何实际的数据复制,因此本地匹配极其快速。

### 从 L3 预取

数据预取是 HiCache 的核心优化技术之一,旨在主动地将 KV 缓存从 L3 存储加载到本地 L2 内存,从而降低后续操作的访问延迟。

**预取触发条件**:
在本地匹配之后,对于在 L1 或 L2 中未找到的部分,系统查询 L3 以获取下一段连续匹配 KV 缓存的元数据。如果 L3 中命中缓存的长度超过某个阈值(默认:256 个 token,可配置),则触发预取操作。

**预取策略**:HiCache 提供三种不同的预取终止策略,以满足不同场景的需求:
- **best_effort**:当 GPU 可以执行 prefill 计算时立即终止,没有等待时间,适用于对延迟极其敏感的场景。
- **wait_complete**:必须等待所有预取操作完成,适用于需要高缓存命中率的场景。
- **timeout**:在指定时间后或完成时终止,平衡延迟和缓存命中率的需求。

预取停止后,已经取到的数据会与本地数据一起用于 prefill 计算。

对于 **timeout** 策略,HiCache 引入了两个配置参数,以支持对预取超时条件进行细粒度控制:

* `prefetch_timeout_base`:基础超时,表示与 token 数量无关的开销(例如调度和同步)。
* `prefetch_timeout_per_ki_token`:每千个 token 的增量超时。

超时计算为:

```
timeout = prefetch_timeout_base + prefetch_timeout_per_ki_token * num_token_to_fetch / 1024
```

### 数据回写

回写机制负责将频繁访问的 KV 缓存从 L1 移动到 L2 和 L3,从而实现更大、更长期的存储以及跨实例的缓存共享。

**可配置的回写策略**:HiCache 支持三种回写策略:

* **write_through**:每次访问都会立即回写到下一层。当带宽充足时,此策略提供最强的缓存收益。
* **write_through_selective**:仅在访问频率超过阈值后才回写数据。此策略只备份热数据,从而减少 I/O 开销。
* **write_back**:仅当数据从上层被逐出时才回写到下一层。此策略缓解了存储压力,适用于存储容量有限但必须最大化内存利用率的场景。

**跨实例共享**:当数据从 L2 回写到 L3 时,只传输 L3 中尚不存在的数据。存储在 L3 中的 KV 缓存随后可以在集群中所有 SGLang 实例之间共享(取决于 L3 后端的实现),在相同的内存预算内显著提升缓存命中率。

### 多 Rank 同步

在多 GPU 并行计算期间,例如张量并行(TP),HiCache 必须确保不同 rank 之间的状态一致。因此,关键的计算步骤需要使用 `all_reduce` 进行状态同步。

例如,在预取期间,使用 `all_reduce(op=min)` 来确保所有 rank 获得相同数量的 L3 命中,从而防止对是否达到预取阈值产生不一致的判断。类似地,在预取完成或终止后,再次需要 `all_reduce(op=min)` 来保证各 rank 对成功取回的 KV 缓存前缀长度达成共识。

### 数据传输优化

**零拷贝数据传输**:预取和回写都涉及大量的数据移动。最小化数据拷贝的次数可以显著提升系统性能。HiCache 支持在从 L2 内存向 L3 后端传输数据时直接传递内存地址和大小。

**"面向批次"的数据组织**:数据读写的粒度对性能有重大影响。为此,HiCache L3 以 **page** 为粒度存储和传输 KV 缓存数据,并在现有的 `layer first` 方案之外支持不同的数据布局,包括 `page first` 和 `page first direct`。在 `page first` 和 `page first direct` 布局下,属于同一 page 的所有 KV 缓存数据被放置在连续的内存中,使其能够作为单个对象通过零拷贝传输传递给 L3。

![HiCache L2 MEM layout](https://lmsys.org/images/blog/hicache/hicache_layout.png)

然而,由于 GPU 的 KV 计算天然是逐层进行的,GPU 本质上以 `layer first` 布局运作。当从 L2 向 GPU 传输 `page first` 数据时,数据必须以每层一个 token 的粒度传输。`page first direct` 布局通过将一个 page 内某给定层的所有 token 组织在一起来缓解这一问题,从而允许从 L2 到 GPU 的传输以 page-layer 级别聚合。

**CPU 到 GPU 的传输优化**:在 HiCache 中,将数据从 CPU 内存移动到 GPU 与从 L3 预取数据到 L2 同样对性能至关重要。HiCache 为此过程采用了多项优化:

* **计算-传输重叠**:在 prefill 阶段,当从 CPU 向 GPU 传输数据时,HiCache 通过在计算第 N 层的同时并发加载第 N+1 层的 KV 缓存来实现层间重叠。这有效地隐藏了数据传输延迟。
* **GPU 辅助的 I/O kernel**:在 `cudaMemcpyAsync` 之上,HiCache 实现了一组专门针对 CPU 与 GPU 之间 KV 缓存传输优化的 GPU 辅助 I/O kernel。与基线方法相比,这些 kernel 实现了高达 3 倍的传输速度。

**针对 MLA 的回写优化**:对于多 TP 下的 MHA(Multi-Head Attention)模型,每个 rank 持有一个 token 的 KV 数据的 `1/tp_size`。相比之下,对于 MLA(Multi-Layer Attention)模型,所有 rank 都为每个 token 持有完整且相同的 KV 数据。HiCache 为 MLA 包含了一项专门优化:只有一个 rank 发起回写操作,确保数据不会在各 rank 之间冗余存储。

### 与 PD 分离部署模式的集成

SGLang 通过 Mooncake TransferEngine 支持 PD(Prefill-Decode)分离部署模式(详见 [此文档](https://docs.sglang.io/advanced_features/pd_disaggregation.html))。在 PD 分离部署模式下,HiCache 可以在 prefill 节点和 decode 节点上同时启用,以优化 prefill 性能。如果在 decode 节点上启用,decode 输出也将被回写到 L3。

### 统一接口与丰富的 L3 存储后端

HiCache 将对 L3 后端的所有读、写、查询操作封装在 `class HiCacheStorage(ABC)` 中,暴露出一组简单一致的接口。这种设计支持广泛的 L3 存储后端,并允许用户选择最适合其特定用例的后端。

- **Mooncake**:Mooncake 是一个用于 LLM 推理的高性能缓存系统,它利用 RDMA 和多网卡资源实现零拷贝、超快速的数据传输。在[这里](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache/storage/mooncake_store)试用 Mooncake。

- **DeepSeek 3FS(HF3FS)**:HF3FS 是一个 Kubernetes 原生的分布式存储解决方案,采用基于 operator 的部署。在[这里](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache/storage/hf3fs)试用 HF3FS。

- **NIXL**:NIXL 提供了一个统一的 API 来访问各种存储插件,包括但不限于 DeepSeek 的 3FS、GPU Direct Storage(GDS)和兼容 Amazon S3 的对象存储。在[这里](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache/storage/nixl)试用 NIXL。

- **AIBrix KVCache**:AIBrix KVCache 是一个生产就绪的 KVCache 卸载框架,它实现了高效的内存分级和低开销的跨引擎复用。在[这里](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache/storage/aibrix_kvcache)试用 AIBrix KVCache。

- **HiCacheFile**:一个用于演示目的的简单的基于文件的存储后端。

特别地,**LMCache** 是一个面向企业级 LLM 推理的高效 KV 缓存层,为 HiCache 提供了一种替代方案。在[这里](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache/storage/lmcache)试用 LMCache。

## 相关参数

- **`--enable-hierarchical-cache`**:启用分级缓存功能。这是使用 HiCache 的前提。

- **`--hicache-ratio HICACHE_RATIO`**:主机 KV 缓存内存池大小与设备池大小的比值。例如,值为 2 表示主机内存池是设备内存池的两倍大。此参数的值必须大于 1,因为当前实现要求为 KV 缓存分配的主机内存大于为 KV 缓存分配的设备内存。

- **`--hicache-size HICACHE_SIZE`**:主机 KV 缓存内存池的大小,单位为 GB。如果设置了此参数,它会覆盖 `hicache-ratio`。例如,`--hicache-size 30` 为**每个 rank** 分配 30GB(1GB = 1e9 字节)的主机内存池。如果有 8 个 rank,则总内存大小为 240GB。与 `hicache-ratio` 一样,此参数的值必须大于为 KV 缓存分配的设备内存大小。

**注意**:`--hicache-ratio` 和 `--hicache-size` 是两个关键参数。一般而言,更大的 HiCache 大小会带来更高的缓存命中率,从而提升 prefill 性能。然而,缓存大小与命中率之间的关系并非线性。一旦大多数可复用的 KV 数据——尤其是热 token——已被缓存,进一步增大大小可能只带来微小的性能提升。用户可以根据其工作负载特性和性能要求来设置这些参数。

- **`--page-size PAGE_SIZE`**:每个 page 的 token 数量。此参数决定了 KV 缓存存储和检索的粒度。更大的 page 大小会减少元数据开销并提升存储后端的 I/O 效率,但当只有部分 page 与存储的 KV 缓存匹配时,可能会降低缓存命中率。对于具有长公共前缀的工作负载,更大的 page 可以提升性能;而前缀更多样的工作负载可能从更小的 page 中受益。关于 page 粒度如何影响 I/O 性能,请参见 [数据传输优化](#data-transfer-optimization)。

- **`--hicache-storage-prefetch-policy {best_effort,wait_complete,timeout}`**:控制何时停止从存储预取。详见 [从 L3 预取](#prefetch-from-l3)。
  - `best_effort`:在不阻塞的情况下尽可能多地预取
  - `wait_complete`:在继续之前等待预取完成
  - `timeout`:在指定时间后或完成时终止(推荐用于生产环境,因为设置合适的超时有助于系统满足所需的 SLO)

- **`--hicache-write-policy {write_back,write_through,write_through_selective}`**:控制数据如何从更快的内存层级写入更慢的内存层级。详见 [数据回写](#data-write-back)。
  - `write_through`:立即将数据写入所有层级(最强的缓存收益)
  - `write_through_selective`:使用命中计数跟踪,只备份频繁访问的数据
  - `write_back`:仅在需要逐出时才将数据回写到更慢的层级(减少 I/O 负载)

- **`--hicache-io-backend {direct,kernel}`**:选择用于 CPU 和 GPU 之间 KV 缓存传输的 I/O 后端。详见 [数据传输优化](#data-transfer-optimization)。
  - `direct`:标准的 CUDA 内存拷贝操作
  - `kernel`:GPU 辅助的 I/O kernel(推荐,性能更好)

- **`--hicache-mem-layout {layer_first,page_first,page_first_direct}`**:主机内存池的内存布局。详见 [数据传输优化](#data-transfer-optimization)。
  - `layer_first`:兼容 GPU 计算 kernel(GPU 内存的默认值)
  - `page_first`:针对 I/O 效率优化
  - `page_first_direct`:将一个 page 内某给定层的所有 token 组织在一起,从而允许从 L2 到 GPU 的传输以 page-layer 级别聚合

- **`--hicache-storage-backend {file,mooncake,hf3fs,nixl,aibrix,dynamic}`**:选择 L3 层级的存储后端。内置后端:file、mooncake、hf3fs、nixl、aibrix。对于 dynamic 后端,使用 --hicache-storage-backend-extra-config 来指定:`backend_name`(自定义名称)、`module_path`(Python 模块路径)、`class_name`(后端类名)。可用的后端请参见 [统一接口与丰富的 L3 存储后端](#unified-interfaces-and-rich-l3-storage-backends)。

- **`--enable-lmcache`**:使用 LMCache 作为替代的分级缓存解决方案。

- **`--hicache-storage-backend-extra-config HICACHE_STORAGE_BACKEND_EXTRA_CONFIG`**:额外配置可以是
  - 一个包含存储后端额外配置的 JSON 字符串,例如 `--hicache-storage-backend-extra-config '{"prefetch_threshold":512, "prefetch_timeout_base": 0.5, "prefetch_timeout_per_ki_token": 0.25}' `,或
  - 一个指定存储后端额外配置的 TOML 或 JSON 或 YAML 文件(为了与 JSON 字符串输入区分,在文件名前加一个 `@`),例如 `--hicache-storage-backend-extra-config "@config.toml"`,其中 `config.toml` 是包含复杂配置的配置文件。当配置由许多或复杂的键值对组成时,这会很有用(例如,NIXL 后端的配置可能很复杂,因此更倾向于使用配置文件)。
