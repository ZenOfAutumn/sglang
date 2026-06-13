# 后训练集成

SGLang 已成为现代 LLM 训练框架事实上的推理后端,为业界众多最先进的模型提供支持。从 GLM-4.6 到 Qwen3,领先的模型在强化学习和后训练工作流中都利用了 SGLang 的高性能推理。

是什么让 SGLang 对后训练如此重要?

- 开箱即用的 Refit 功能:为 colocate 或 disaggregate 提供多样化的方法
- 易于推迟生成:支持部分 rollout 和专门的 rollout 控制
- 细粒度的引擎休眠与唤醒:助力火力全开的 rollout 与训练
- 训练与服务对齐:确保训练和服务中的性能一致性
- 负载均衡路由器:面向高吞吐 rollout 的缓存感知负载均衡
- 确定性推理:确保 rollout 与训练之间零 KL 散度

这些能力,加上在各大主流框架中的原生集成支持,使 SGLang 成为现代 LLM/VLM 后训练的基础设施支柱。我们也在这份幻灯片中分享了我们的最新工作,[Optimizing Large-Scale RL with SGLang](https://gamma.app/docs/Optimizing-RL-with-SGLang-y0kqgj877k34779)。

## 采用情况

- [**Miles**](https://github.com/radixark/miles):面向大型 MoE 模型的企业级 RL 框架,具备 SGLang 原生 rollout、推测式训练和生产级稳定性
- [**slime**](https://github.com/THUDM/slime):结合 Megatron 和 SGLang 的后训练框架,用于训练 GLM-4.6
- [**AReaL**](https://github.com/inclusionAI/AReaL):全异步 RL 系统,借助 SGLang 后端实现连续 rollout 生成,获得 2.77 倍加速
- [**ROLL**](https://github.com/alibaba/ROLL):ROLL 是一个高效且易用的 RL 库,专为利用大规模 GPU 资源的大语言模型而设计
- [**verl**](https://github.com/volcengine/verl):全栈 RLHF 框架,支持 PPO、GRPO 和 ReMax,并具备模块化的 SGLang 集成
- [**Unsloth**](https://docs.unsloth.ai/basics/inference-and-deployment/sglang-guide):借助优化内核实现 2 倍更快的微调,并可与 SGLang 推理无缝部署
- [**LLaMA Factory**](https://github.com/hiyouga/LLaMA-Factory):统一框架,可使用 LoRA、QLoRA 和全量微调方法训练 100+ 种 LLM
- [**Tunix**](https://github.com/google/tunix):Google 的 JAX 原生库,用于 LLM 后训练,支持 SFT、DPO、PPO 和 GRPO
- [**RL2**](https://github.com/ChenmienTan/RL2):Ray Less Reinforcement Learning,一个简洁的大语言模型后训练库


## 合作

由于设计合作伙伴的隐私原因,我们无法列出采用 SGLang 进行后训练的公司。不过,如果你有兴趣,我们很乐意与你分享细节,并信赖来自美国和中国 10 多家顶级公司及前沿实验室的选择。如果你有兴趣将 SGLang 与你的训练框架集成,或需要技术支持,我们随时为你提供帮助!请通过 **rl_team@lmsys.org** 联系我们,以洽谈合作、获取集成指导以及定制功能开发。
