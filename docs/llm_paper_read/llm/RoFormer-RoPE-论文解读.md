# RoFormer 论文解读：Rotary Position Embedding（RoPE，旋转位置编码）

> 论文：RoFormer: Enhanced Transformer with Rotary Position Embedding（Jianlin Su 苏剑林 等，追一科技，arXiv:2104.09864v5，2023-11）
> 本地 PDF：`llm/RoFormer-Enhanced-Transformer-with-Rotary-Position-Embedding.pdf`
> 官方实现：https://github.com/ZhuiyiTechnology/roformer ｜ HF：`transformers.RoFormerModel`
> 配图目录：`llm/images/roformer/`（`paper_*` 为论文原图，其余由 numpy/matplotlib 按论文公式复现）

---

## 0. 一句话总结

RoPE 的做法是：**对 Query/Key 按绝对位置 $m$ 做旋转**（每 2 维一组，第 $i$ 组转 $m\theta_i$ 角度），这样旋转后的内积 $\langle R_m q, R_n k\rangle = q^\top R_{n-m} k$ **只跟相对位置 $n-m$ 有关**。所以它是**写法上用绝对位置、效果上是相对位置**的位置编码，而且是**乘法**（旋转）而不是**加法**。它有三个性质：

1. 不需要可学习参数，长度可变；
2. 内积会随相对距离增大而**长程衰减**；
3. 能和**线性注意力**兼容。

后来 LLaMA、Qwen、DeepSeek、GLM、Mistral 等几乎所有主流 LLM 都采用了 RoPE（或其变体）。

![PE 演进](images/roformer/pe_evolution.png)

---

## 1. 背景：为什么需要位置编码

### 1.1 Self-Attention 本身不区分位置

标准注意力：

$$
a_{m,n}=\frac{\exp\left(\frac{q_m^\top k_n}{\sqrt d}\right)}{\sum_{j=1}^{N}\exp\left(\frac{q_m^\top k_j}{\sqrt d}\right)},\qquad o_m=\sum_{n=1}^{N}a_{m,n}v_n
$$

如果 $q,k,v$ 只由词向量 $x$ 计算，那么把输入序列打乱后，每个 token 的输出也只是跟着换了位置（置换等变），模型看不出词的顺序。所以需要通过

$$
q_m=f_q(x_m,m),\quad k_n=f_k(x_n,n),\quad v_n=f_v(x_n,n)
$$

把位置信息注入进去。问题就变成：**怎么选 $f$**。

### 1.2 已有方案：都是"加法"

| 流派 | 代表 | 形式 | 问题 |
|---|---|---|---|
| 绝对·可学习 | BERT / GPT-2 / ALBERT / ELECTRA | $f_t(x_i,i)=W_t(x_i+p_i)$，$p_i$ 是可训练参数 | 长度上限固定为 $L$，无法外推 |
| 绝对·Sinusoidal | Vaswani 2017 | $p_{i,2t}=\sin(i/10000^{2t/d})$，$p_{i,2t+1}=\cos(\cdot)$ | 加到 embedding 上后，相对位置只能间接学到 |
| 相对·Shaw | Shaw 2018 | $f_k=W_k(x_n+\tilde p^k_{r})$，$r=\text{clip}(m-n)$ | 需要 $N\times N$ 的相对位置表 |
| 相对·XL | Transformer-XL | 把 $q^\top k$ 展开成 4 项，$p_n\to\tilde p_{m-n}$，$p_m\to u,v$ | 项数多，实现复杂 |
| 相对·T5 | T5 | $q^\top k=x^\top W_q^\top W_k x+b_{i,j}$ | 只是一个标量偏置 |
| 相对·DeBERTa | He 2020 | 保留中间两项 content-position 交叉项 | 同上 |

这些方法的共同点（论文 §2 的核心观察）是：**都从 $W(x+p)$ 这种加法分解出发**，把 $q_m^\top k_n$ 展开成

$$
q_m^\top k_n = x_m^\top W_q^\top W_k x_n + x_m^\top W_q^\top W_k p_n + p_m^\top W_q^\top W_k x_n + p_m^\top W_q^\top W_k p_n
$$

再去改其中的某几项。这样做有两个缺点：

- 位置信息被**加**进了表征，和内容混在一起；
- 相对位置是通过修改注意力矩阵的元素来实现的，必须**显式算出 $N\times N$ 的矩阵**，所以**线性注意力用不了**（线性注意力本来就是要避免算出这个矩阵）。

---

## 2. 核心思想：从约束反推 $f$

RoPE 不从"加法分解"出发，而是直接提出一个**约束**，然后求解满足约束的 $f$：

$$
\boxed{\langle f_q(x_m,m),\ f_k(x_n,n)\rangle = g(x_m,x_n,m-n)}
\tag{11}
$$

意思是：**内积只能通过相对位置 $m-n$ 依赖位置**。

### 2.1 二维情形的推导（§3.4.1）

取 $d=2$，把 2 维向量看成复数，用极坐标表示：

$$
f_q(x_q,m)=R_q(x_q,m)\,e^{i\Theta_q(x_q,m)},\quad
f_k(x_k,n)=R_k(x_k,n)\,e^{i\Theta_k(x_k,n)}
$$

复数内积 $\langle a,b\rangle=\mathrm{Re}[a b^*]$，代入约束 (11)，模长和辐角分开写：

$$
R_q(x_q,m)\,R_k(x_k,n)=R_g(x_q,x_k,n-m),\qquad
\Theta_k(x_k,n)-\Theta_q(x_q,m)=\Theta_g(x_q,x_k,n-m)
$$

再加上初始条件 $f_q(x_q,0)=q,\ f_k(x_k,0)=k$（位置 0 处等于不带位置的原向量）：

1. **令 $m=n$**：得到 $R_q(x_q,m)R_k(x_k,m)=\|q\|\|k\|$，最简单的解是**模长和位置无关**：$R_q=\|q\|,\ R_k=\|k\|$。
2. **令 $m=n$ 看辐角**：$\Theta_q(x_q,m)-\theta_q=\Theta_k(x_k,m)-\theta_k$，说明辐角的增量和内容无关，只和位置有关，记为 $\phi(m)$。
3. **令 $n=m+1$**：$\phi(m+1)-\phi(m)=$ 常数，所以 $\phi(m)$ 是**等差数列**：$\phi(m)=m\theta+\gamma$。

取 $\gamma=0$，$q=W_qx_m$，$k=W_kx_n$，得到最终解：

$$
f_q(x_m,m)=(W_qx_m)\,e^{im\theta},\qquad f_k(x_n,n)=(W_kx_n)\,e^{in\theta}
$$

$$
g(x_m,x_n,m-n)=\mathrm{Re}\left[(W_qx_m)(W_kx_n)^*\,e^{i(m-n)\theta}\right]
\tag{12}
$$

**几何含义**：把 $q$ 转 $m\theta$，把 $k$ 转 $n\theta$，两者的**夹角差**只取决于 $m-n$，所以内积只依赖相对位置。写成实矩阵就是一个 2D 旋转：

$$
f_{\{q,k\}}(x_m,m)=\begin{pmatrix}\cos m\theta & -\sin m\theta\\ \sin m\theta & \cos m\theta\end{pmatrix}
W_{\{q,k\}}\,x_m
\tag{13}
$$

下图是数值验证：同一对 $(q,k)$，分别放在 $(m,n)=(2,0)$ 和 $(6,4)$，绝对位置不同、相对距离都是 2，**内积完全相同（0.9425）**：

![2D rotation](images/roformer/rope_2d_rotation.png)

> 注意这个推导的**唯一性**：在"模长不随位置变"这个最简单的选择下，旋转是满足约束 (11) 的**必然**形式，而不是拍脑袋设计的。

### 2.2 推广到 $d$ 维（§3.2.2）

利用内积的线性性，把 $d$ 维（$d$ 为偶数）拆成 $d/2$ 个 2 维子空间，每个子空间用不同的频率 $\theta_i$ 独立旋转：

$$
f_{\{q,k\}}(x_m,m)=R^d_{\Theta,m}\,W_{\{q,k\}}\,x_m
\tag{14}
$$

$$
R^d_{\Theta,m}=\begin{pmatrix}
\cos m\theta_1 & -\sin m\theta_1 & & & & \\
\sin m\theta_1 & \cos m\theta_1 & & & & \\
& & \cos m\theta_2 & -\sin m\theta_2 & & \\
& & \sin m\theta_2 & \cos m\theta_2 & & \\
& & & & \ddots & \\
& & & & & \begin{smallmatrix}\cos m\theta_{d/2} & -\sin m\theta_{d/2}\\ \sin m\theta_{d/2} & \cos m\theta_{d/2}\end{smallmatrix}
\end{pmatrix}
\tag{15}
$$

$$
\Theta=\left\{\theta_i=10000^{-2(i-1)/d},\ i\in[1,2,\dots,d/2]\right\}
$$

代入注意力：

$$
q_m^\top k_n=(R^d_{\Theta,m}W_qx_m)^\top(R^d_{\Theta,n}W_kx_n)=x_m^\top W_q^\top\,R^d_{\Theta,n-m}\,W_kx_n
\tag{16}
$$

其中 $R^d_{\Theta,n-m}=(R^d_{\Theta,m})^\top R^d_{\Theta,n}$。关键性质：

- **正交矩阵**：$R^\top R=I$，不改变向量模长，数值稳定；
- **乘法**：位置信息通过矩阵乘注入，不是加法；
- **相对性由群结构保证**：旋转满足 $R_m^\top R_n=R_{n-m}$。

论文原图 Figure 1 展示了这一过程（上半部分是 $d=2$ 的旋转，下半部分是每个 token 的 $d/2$ 组分量各自按频率 $\theta_1\ldots\theta_{d/2}$ 旋转）：

![Paper Figure 1](images/roformer/paper_fig1_rope_impl.png)

### 2.3 "绝对位置写法、相对位置效果"的直观验证

让序列每个位置都放**同一个 token**，只改变位置，然后看注意力得分矩阵 $A_{m,n}=q_m^\top k_n$：

- **RoPE**：$A$ 是严格的 **Toeplitz 矩阵**（每条对角线上的值都一样），数值上对角线方差 $\approx 2\times10^{-15}$；
- **加法 Sinusoidal 绝对 PE**：交叉项 $x^\top W p$、$p^\top W p$ 破坏了对角结构，对角线方差 $\approx 4.36$。

![Toeplitz vs abs](images/roformer/rope_toeplitz_vs_abs.png)

---

## 3. 多频率设计：$\theta_i$ 的含义

$\theta_i=10000^{-2i/d}$ 和 Sinusoidal PE 的频率完全一样，是一组**几何级数**的频率：

| 维度组 $i$ | 频率 $\theta_i$ | 波长 $\lambda_i=2\pi/\theta_i$ | 作用 |
|---|---|---|---|
| 小 $i$（前面的维度） | $\approx 1$ rad/token | $\approx 6$ tokens | 高频，区分相邻 token，对应**局部**位置 |
| 大 $i$（后面的维度） | $\approx 10^{-4}$ | $\approx 5\times10^4$ tokens | 低频，转得很慢，对应**长程**位置 |

可以类比时钟的秒针、分针、时针：多个不同转速的指针组合起来，就能唯一确定一个很大范围内的位置。

![frequencies](images/roformer/rope_frequencies.png)

> **工程含义**：当 $d=128$、base=10000 时，大约从 $i\approx45$ 开始，波长超过 4096。也就是说，**训练长度为 4K 时，最后十几组低频维度连一个完整周期都没转完**，模型对这些维度的大角度没有见过。这正是后来 **PI / NTK-aware / YaRN** 等长度外推方法要解决的问题（见 §7）。

---

## 4. 高效实现（§3.4.2）

式 (15) 是稀疏的分块对角矩阵，直接做矩阵乘会浪费大量计算。等价的逐元素写法：

$$
R^d_{\Theta,m}x=
\begin{pmatrix}x_1\\x_2\\x_3\\x_4\\\vdots\\x_{d-1}\\x_d\end{pmatrix}\otimes
\begin{pmatrix}\cos m\theta_1\\\cos m\theta_1\\\cos m\theta_2\\\cos m\theta_2\\\vdots\\\cos m\theta_{d/2}\\\cos m\theta_{d/2}\end{pmatrix}+
\begin{pmatrix}-x_2\\x_1\\-x_4\\x_3\\\vdots\\-x_d\\x_{d-1}\end{pmatrix}\otimes
\begin{pmatrix}\sin m\theta_1\\\sin m\theta_1\\\sin m\theta_2\\\sin m\theta_2\\\vdots\\\sin m\theta_{d/2}\\\sin m\theta_{d/2}\end{pmatrix}
\tag{34}
$$

计算量是 $O(d)$：2 次逐元素乘加上一次 shuffle。$\cos/\sin$ 只和位置有关，可以**预先算好缓存**。

### 4.1 两种维度配对方式（工程中非常重要）

论文式 (34) 是**相邻两维配对** $(x_1,x_2),(x_3,x_4),\ldots$，即 **GPT-J / interleave 风格**。而 GPT-NeoX、HF LLaMA 采用的是**前后两半配对** $(x_i,x_{i+d/2})$，即 **NeoX / rotate_half 风格**。两者在数学上等价（只是对维度做了一个固定的置换，可以吸收进 $W_q,W_k$），但**权重不能混用**：转换 checkpoint 时必须对 $W_q,W_k$ 的行做相应的重排。

![pairing](images/roformer/rope_pairing_styles.png)

sglang 中这两种风格由 `is_neox_style` 控制：

```36:62:/home/sankuai/sglang/python/sglang/srt/layers/rotary_embedding/utils.py
def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)
```

$\theta_i$ 和 cos/sin 缓存的计算与论文完全一致（`inv_freq` 就是 $\theta_i=\text{base}^{-2i/d}$）：

```153:182:/home/sankuai/sglang/python/sglang/srt/layers/rotary_embedding/base.py
    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        """Compute the inverse frequency."""
        # ... existing code ...
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(
                    0, self.rotary_dim, 2, dtype=torch.float, device=init_device
                )
                / self.rotary_dim
            )
        )
        # ... existing code ...
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache
```

一个最小 PyTorch 参考实现（interleave 风格，和论文一致）：

```python
import torch

def rope(x: torch.Tensor, pos: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """x: [seq, heads, d], pos: [seq]"""
    d = x.shape[-1]
    theta = base ** (-torch.arange(0, d, 2, device=x.device).float() / d)   # [d/2]
    ang = pos[:, None].float() * theta[None, :]                              # [seq, d/2]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]                  # broadcast heads
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x2 * cos + x1 * sin
    return out

# q, k 分别旋转；v 不加位置
q, k = rope(q, pos), rope(k, pos)
```

> **只旋转 Q/K，不旋转 V**：位置信息只需要进入注意力权重，V 保持原样。这和 Transformer-XL 之后"值项去掉位置"的做法一致。

---

## 5. RoPE 的性质

### 5.1 长程衰减（§3.3、§3.4.3）

把 $q,k$ 按两维一组写成复数，内积为：

$$
(R^d_{\Theta,m}W_qx_m)^\top(R^d_{\Theta,n}W_kx_n)=\mathrm{Re}\left[\sum_{i=0}^{d/2-1}q_{[2i:2i+1]}\,k^*_{[2i:2i+1]}\,e^{i(m-n)\theta_i}\right]
\tag{35}
$$

令 $h_i=q_{[2i:2i+1]}k^*_{[2i:2i+1]}$，$S_j=\sum_{i=0}^{j-1}e^{i(m-n)\theta_i}$，$h_{d/2}=0$，$S_0=0$，用 **Abel 变换**（分部求和）：

$$
\sum_{i=0}^{d/2-1}h_i\,e^{i(m-n)\theta_i}=\sum_{i=0}^{d/2-1}h_i(S_{i+1}-S_i)=-\sum_{i=0}^{d/2-1}S_{i+1}(h_{i+1}-h_i)
\tag{36}
$$

于是得到上界：

$$
\left|\sum_{i=0}^{d/2-1}q_{[2i:2i+1]}k^*_{[2i:2i+1]}e^{i(m-n)\theta_i}\right|
\le\left(\max_i|h_{i+1}-h_i|\right)\sum_{i=0}^{d/2-1}|S_{i+1}|
\tag{37}
$$

当 $\theta_i=10000^{-2i/d}$ 时，$\frac{1}{d/2}\sum_{i=1}^{d/2}|S_i|$ 会随相对距离 $|m-n|$ 增大而减小。直观理解：距离越远，各频率的相位 $e^{i(m-n)\theta_i}$ 越"散"，求和时互相抵消得越多。

论文原图 Figure 2：

![Paper Figure 2](images/roformer/paper_fig2_long_term_decay.png)

我用 $d=128$ 复现并扩展到 1024 距离，同时对比了不同 base（右图是 $q=k=\mathbf 1$ 时归一化后的实际内积）：

![decay](images/roformer/rope_long_term_decay.png)

几个值得注意的细节：

- 衰减**并不单调**，曲线有明显震荡；它衰减的是**上界**，不是内积本身；
- **base 越大衰减越慢**（绿线 base=500000，LLaMA-3 用的就是这个值），远距离 token 的注意力保留得更多，这是加大 base 支持长上下文的直观原因之一；
- base 太小（红线 1000）时，远距离内积会在 0 附近来回变号，长程信息基本被"抹掉"。

### 5.2 与线性注意力兼容（§3.3）

注意力的一般形式：

$$
\mathrm{Attention}(Q,K,V)_m=\frac{\sum_{n=1}^N\mathrm{sim}(q_m,k_n)\,v_n}{\sum_{n=1}^N\mathrm{sim}(q_m,k_n)}
\tag{17}
$$

线性注意力（Katharopoulos 2020）取 $\mathrm{sim}(q,k)=\phi(q)^\top\varphi(k)$，利用结合律先算 $\sum_n\varphi(k_n)v_n^\top$，复杂度从 $O(N^2)$ 降到 $O(N)$：

$$
\mathrm{Attention}(Q,K,V)_m=\frac{\sum_{n=1}^N\phi(q_m)^\top\varphi(k_n)\,v_n}{\sum_{n=1}^N\phi(q_m)^\top\varphi(k_n)}
\tag{18}
$$

加法式的相对位置编码要修改 $N\times N$ 矩阵的每个元素，线性注意力里根本不显式算出这个矩阵，所以**用不了**。而 RoPE 只是对每个向量**单独**做旋转，可以直接作用在 $\phi,\varphi$ 的输出上：

$$
\mathrm{Attention}(Q,K,V)_m=\frac{\sum_{n=1}^N\left(R^d_{\Theta,m}\phi(q_m)\right)^\top\left(R^d_{\Theta,n}\varphi(k_n)\right)v_n}{\sum_{n=1}^N\phi(q_m)^\top\varphi(k_n)}
\tag{19}
$$

注意：**分母不加旋转**，否则旋转后可能出现负数甚至零，导致除零。代价是权重不再是严格归一化的概率分布，论文认为这仍然能表示 value 的相对重要性。

> 这一点在今天仍然重要：Mamba / 线性注意力 / KDA 等混合架构中，RoPE 依旧是最容易和它们组合的位置编码。

### 5.3 序列长度灵活

RoPE 没有可学习的位置参数，$\cos/\sin$ 对任意 $m$ 都能直接计算，**理论上**不受最大长度限制。但"能算"不等于"能外推"：训练时没见过的低频大角度仍然会让效果下降。这一点论文没有深入讨论，是后续大量工作的出发点。

---

## 6. 实验

所有实验都在 2 台 4×V100 服务器上完成，规模较小（2021 年的条件）。

### 6.1 机器翻译 WMT14 En-De（§4.1）

基于 fairseq 的 Transformer-base，仅替换位置编码：

| 模型 | BLEU |
|---|---|
| Transformer-base | 27.3 |
| **RoFormer** | **27.5** |

### 6.2 预训练 MLM（§4.2）与 Performer（§4.4）

- **左图**：BERT-base vs RoFormer（把 BERT 的位置编码换成 RoPE），BookCorpus + Wikipedia，seq 512，bs 64，100k 步。**RoFormer 收敛更快，loss 更低**。
- **右图**：12 层 char-level Performer（线性注意力），Enwik8，seq 1024。**加 RoPE 后收敛明显更快，loss 更低**，验证了 §5.2 的兼容性。

![Paper Figure 3](images/roformer/paper_fig3_pretrain_loss.png)

### 6.3 GLUE 微调（§4.3）

| 模型 | MRPC | SST-2 | QNLI | STS-B | QQP | MNLI (m/mm) |
|---|---|---|---|---|---|---|
| BERT | 88.9 | **93.5** | **90.5** | 85.8 | 71.2 | **84.6/83.4** |
| RoFormer | **89.5** | 90.7 | 88.0 | **87.0** | **86.4** | 80.2/79.8 |

论文说"6 个任务中 3 个显著超过 BERT"。但要客观看待：**SST-2、QNLI、MNLI 三个任务 RoFormer 明显更差**（MNLI 低约 4 个点）；QQP 从 71.2 提升到 86.4 这么大的幅度也不太正常，很可能是 baseline 设置的问题。所以 GLUE 结果只能说**有好有坏**，不能作为 RoPE 更优的强证据。

### 6.4 中文长文本（§4.5）

在 WoBERT（词粒度中文 BERT）的基础上把绝对位置编码换成 RoPE，用约 34GB 中文语料分 6 个阶段预训练（不同阶段使用不同的 max_len 和 batch size）：

| 阶段 | max_len | bs | steps | loss | acc |
|---|---|---|---|---|---|
| 1 | 512 | 256 | 200k | 1.73 | 65.0% |
| 2 | 1536 | 256 | 12.5k | 1.61 | 66.8% |
| 3 | 256 | 256 | 120k | 1.75 | 64.6% |
| 4 | 128 | 512 | 80k | 1.83 | 63.4% |
| 5 | 1536 | 256 | 10k | 1.58 | 67.4% |
| 6 | 512 | 512 | 30k | 1.66 | 66.2% |

下游任务是 **CAIL2019-SCM**（法律相似案例匹配，三元组，文本大多超过 512 字）：

| 模型 | Validation | Test |
|---|---|---|
| BERT-512 | 64.13% | 67.77% |
| WoBERT-512 | 64.07% | 68.10% |
| RoFormer-512 | 64.13% | 68.29% |
| **RoFormer-1024** | **66.07%** | **69.79%** |

在相同的 512 长度下，RoFormer 和 WoBERT 差不多；**放开到 1024 后提升 1.5 个点**。这说明它的主要价值在于**能够处理更长的输入**（绝对位置编码 BERT 受 512 个可学习位置向量限制，没法直接用 1024）。

![experiments](images/roformer/rope_experiments.png)

---

## 7. 论文之后：RoPE 的演进与工程实践

RoFormer 原文的实验规模不大，但 RoPE 的影响主要体现在后来的 LLM 中。下面是和推理系统（如 sglang）相关的主要演进：

| 变体 | 核心改动 | 代表模型 | sglang 实现 |
|---|---|---|---|
| 原版 RoPE | $\theta_i=10000^{-2i/d}$ | LLaMA-1/2、GPT-J、GPT-NeoX | `RotaryEmbedding` |
| 增大 base | base 改为 $5\times10^5$、$10^6$ 等 | LLaMA-3、Qwen2 | `rope_theta` 参数 |
| Linear / PI | 位置插值 $m\to m/s$ | 早期长上下文微调 | `LinearScalingRotaryEmbedding` |
| NTK-aware / Dynamic NTK | 按长度放大 base，高频基本不变、低频插值 | CodeLLaMA 等 | `DynamicNTKScalingRotaryEmbedding` |
| YaRN | 分频段插值 + attention 温度 | Qwen、DeepSeek | `YaRNScalingRotaryEmbedding` / `DeepseekScalingRotaryEmbedding` |
| LLaMA-3 scaling | 按波长分段缩放 | LLaMA-3.1 | `Llama3RotaryEmbedding` |
| LongRoPE | 搜索每个维度的缩放因子 | Phi-3 | `Phi3LongRoPEScaledRotaryEmbedding` |
| M-RoPE | 把维度分给 时间/高/宽 3 个轴 | Qwen2-VL / Qwen2.5-VL | `MRotaryEmbedding` |
| Partial / 解耦 RoPE | 只对部分维度加 RoPE | DeepSeek MLA（`qk_rope_head_dim=64`）、GLM（rotary 只作用于一半维度） | `rotary_dim < head_size` |

### 7.1 为什么 MLA 必须把 RoPE 解耦

MLA 希望把 $W^{UK}$ **吸收**进 $W_q$，这样 decode 时可以直接在低秩 latent $c^{KV}$ 上算注意力。但 RoPE 会在 $q$ 和 $k$ 之间插入一个依赖位置的矩阵：

$$
q_m^\top k_n=(R_mW_qx_m)^\top(R_nW^{UK}c_n)=x_m^\top W_q^\top R_{n-m}W^{UK}c_n
$$

$R_{n-m}$ 夹在中间，而且随 $(m,n)$ 变化，所以 $W_q^\top$ 和 $W^{UK}$ **无法合并成一个固定矩阵**。DeepSeek 的解决方法是：把 head 拆成 **nope 部分**（不加位置，可以吸收）和 **rope 部分**（单独一小段维度 $d_h^R$ 加 RoPE，所有 head 共享一个 $k^R$）。详见 `docs/learn/arch/MLA.md`。

### 7.2 推理系统中与 RoPE 相关的问题

- **KV Cache 中存的是已经旋转过的 K**：位置一旦确定就不能改。Prefix cache 复用的前提是**同一前缀的位置编号完全相同**；StreamingLLM、PD 分离传 KV、跨请求复用不同偏移的片段时，都需要特别注意这一点。
- **cos/sin 缓存长度**：超过 `max_position_embeddings` 时需要扩容（sglang 的 `_ensure_cos_sin_cache_length`）。
- **融合算子**：sglang / FlashInfer 把 RoPE 和写 KV cache（`fused_set_kv_buffer_arg`）融合到一个 kernel，减少一次显存读写。
- **数值精度**：位置很大时，$m\theta_i$ 在 bf16 下精度不够，cos/sin 一般用 fp32 计算后再转换类型（HF 实现中 CPU/GPU 计算的细微差异也来源于此）。

---

## 8. 论文的局限与勘误

**作者承认的局限（§4.5.5）**：

1. 没有解释为什么 RoPE **收敛更快**；
2. 长程衰减性质其他位置编码也有，**没有解释 RoPE 为什么在长文本上更好**；
3. 预训练需要较多硬件资源。

**我补充的几点**：

- 实验规模小，GLUE 结果有好有坏（见 §6.3），说服力有限。RoPE 真正被认可是在 GPT-J / PaLM / LLaMA 等大模型上的大规模实践；
- 长程衰减只是**上界**的衰减，并且是在 $q,k$ 内容"平均"的情况下才成立；实际训练出来的模型中，某些头会利用低频维度保持对远距离 token 的关注（例如 attention sink）；
- 论文没有讨论**外推**：没见过的位置角度会导致效果崩溃，这是 PI/NTK/YaRN 系列工作的出发点。

**原文中的笔误**（阅读时注意）：

- 式 (4) 中 $\sin(k/10000^{2t/d})$ 的 $k$ 应为位置 $i$；
- 式 (15) 下方写 $\theta_i=10000^{-2(i-1)/d}$（$i$ 从 1 开始），§3.3 又写成 $10000^{-2i/d}$（$i$ 从 0 开始），两者只是下标起点不同；
- 式 (16) 最右边的 $x^\top$ 应为 $x_m^\top$，$W_q$ 应带转置；
- 式 (32) 中 $q=W_qx_n$ 应为 $W_qx_m$；
- §4.3 表 2 标题中的 "GLEU" 应为 "GLUE"。

---

## 9. 速记

| 问题 | 答案 |
|---|---|
| 全称 | Rotary Position Embedding，旋转位置编码 |
| 一句话 | 对 Q/K 按绝对位置旋转，内积只依赖相对位置 |
| 核心公式 | $q_m^\top k_n=x_m^\top W_q^\top R^d_{\Theta,n-m}W_kx_n$ |
| 频率 | $\theta_i=\text{base}^{-2i/d}$，base 默认 10000 |
| 作用对象 | 只作用于 Q 和 K，不作用于 V；每层注意力都要做 |
| 加法还是乘法 | 乘法（正交旋转，保持模长） |
| 参数量 | 0 |
| 三个性质 | 长度灵活；长程衰减；兼容线性注意力 |
| 实现风格 | GPT-J interleave（论文） vs NeoX rotate_half（LLaMA/HF），对应 sglang 的 `is_neox_style` |
| 后续演进 | 增大 base / PI / NTK / YaRN / LongRoPE / M-RoPE / MLA 解耦 RoPE |
