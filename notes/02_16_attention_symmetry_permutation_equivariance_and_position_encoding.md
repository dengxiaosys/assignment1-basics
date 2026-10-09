# Attention 对称性、排列等变性与位置编码

## 0. 本文要回答的问题

学习 self-attention 时，下面三句话很容易混在一起：

1. “没有位置编码，attention matrix 是对称的。”
2. “没有位置编码，self-attention 不知道 token 顺序。”
3. “Transformer 必须加入位置编码。”

第一句通常是错的，第二句需要限定为不含 causal mask 的完整 self-attention，第三句对标准双向 Transformer 成立，但对带 causal mask 的 decoder-only Transformer 不是不可违背的数学定理。

混淆的根源是把三种不同性质都叫成了“对称”：

- score 矩阵是否满足 $S=S^\top$；
- attention probability 是否满足 $P=P^\top$；
- 整个 self-attention 层是否对 token 排列保持等变。

本文将这三个问题分开，并解释位置编码和 causal mask 各自负责什么。

配套阅读：

- [02_08 位置编码与 RoPE 原理](./02_08_rope_explained.md)
- [02_11 Scaled Dot-Product Attention 实现](./02_11_attention_implementation_notes.md)
- [Assignment 1 handout：RoPE 与 NoPE](./00_01_cs336_assignment1_basics_extracted.md#L1515-L1521)

## 1. 先区分 $S$、$P$、$O$ 和 FlashAttention accumulator

对单个 attention head，设第 $i$ 个 token 的输入向量为列向量 $x_i\in\mathbb R^{d_{\mathrm{model}}}$。query、key、value 由不同线性变换得到：

$$q_i=W_Qx_i,\qquad k_i=W_Kx_i,\qquad v_i=W_Vx_i$$

raw attention score 为：

$$S_{ij}=\frac{q_i^\top k_j}{\sqrt{d_k}}$$

对 $S$ 的每一行执行 softmax，得到 attention probability：

$$P_{ij}=\frac{\exp(S_{ij})}{\sum_c\exp(S_{ic})}$$

输出向量为：

$$o_i=\sum_jP_{ij}v_j$$

因此：

| 符号 | 含义 | shape |
|---|---|---|
| $S$ | softmax 前的 score 矩阵 | $(N_q,N_k)$ |
| $P$ | 逐行 softmax 后的 attention probability | $(N_q,N_k)$ |
| $O$ | attention 输出 | $(N_q,d_v)$ |

有些材料会把 attention weight 写成 $A$，此时 $A$ 通常就是这里的 $P$。但在 FlashAttention 推导中，$A_i$ 还可能表示未归一化输出 accumulator，其 shape 是 $(B_q,d_v)$，不是方阵，不能讨论矩阵对称性。阅读公式时必须先确认作者对 $A$ 的定义。

## 2. 没有位置编码，score 矩阵也通常不对称

矩阵 $S$ 对称要求：

$$S_{ij}=S_{ji}$$

而实际 attention 中：

$$S_{ij}=\frac{q_i^\top k_j}{\sqrt{d_k}},\qquad S_{ji}=\frac{q_j^\top k_i}{\sqrt{d_k}}$$

因为 $W_Q$ 和 $W_K$ 是两套独立参数，通常有 $q_i\ne k_i$，因而没有理由要求：

$$q_i^\top k_j=q_j^\top k_i$$

所以“不加入位置编码”并不能推出 $S$ 对称。

将投影展开后可以看得更清楚：

$$S_{ij}=\frac{x_i^\top W_Q^\top W_Kx_j}{\sqrt{d_k}}$$

要让所有输入下的 $S$ 都对称，需要 $W_Q^\top W_K$ 具有相应的对称性；标准 Transformer 并没有施加这个约束。

### 2.1 什么时候 $S$ 会对称

如果人为令 $Q=K$，也就是对 query 和 key 使用完全相同的向量，那么：

$$S_{ij}=\frac{q_i^\top q_j}{\sqrt{d_k}}=\frac{q_j^\top q_i}{\sqrt{d_k}}=S_{ji}$$

此时 $S$ 是 Gram matrix，天然对称。

但这仍然和位置编码没有必然关系。即使加入绝对位置向量 $p_i$，只要 query 和 key 仍由同一变换得到：

$$q_i=k_i=W(x_i+p_i)$$

那么 $S=QQ^\top/\sqrt{d_k}$ 依旧对称。反过来，即使没有任何位置编码，只要 $W_Q\ne W_K$，$S$ 通常就不对称。

## 3. 即使 $S$ 对称，逐行 softmax 后的 $P$ 也通常不对称

softmax 是逐行归一化：

$$P_{ij}=\frac{\exp(S_{ij})}{\sum_c\exp(S_{ic})}$$

即使 $S_{ij}=S_{ji}$，第 $i$ 行和第 $j$ 行的分母通常不同，所以：

$$P_{ij}\ne P_{ji}$$

例如：

$$S=\begin{bmatrix}1&0\\0&2\end{bmatrix}$$

$S$ 是对称矩阵，但逐行 softmax 后：

$$P\approx\begin{bmatrix}0.731&0.269\\0.119&0.881\end{bmatrix}$$

其中 $P_{12}\ne P_{21}$。

因此需要记住：

> score 对称不能推出 attention probability 对称。

## 4. 没有位置编码真正带来的是排列等变性

### 4.1 什么是排列等变

考虑不含 causal mask、所有 token 可以互相访问的完整 self-attention。设 $\sigma$ 是对 token 位置的任意排列，将原序列重排后，第 $i$ 个新位置放置原来的第 $\sigma(i)$ 个 token。

没有位置编码时，线性投影只依赖 token 内容，因此：

$$q'_i=q_{\sigma(i)},\qquad k'_i=k_{\sigma(i)},\qquad v'_i=v_{\sigma(i)}$$

重排后的 score 满足：

$$S'_{ij}=S_{\sigma(i),\sigma(j)}$$

逐行 softmax 后同样有：

$$P'_{ij}=P_{\sigma(i),\sigma(j)}$$

最终输出只是跟着 token 一起重排：

$$o'_i=o_{\sigma(i)}$$

这叫做排列等变：

> 输入怎样排列，输出就做同样的排列。

它不等于“score 矩阵对称”，也不等于“每个排列得到完全相同的输出序列”。

### 4.2 等变与不变的区别

- **排列等变**：重排输入后，输出按相同方式重排；
- **排列不变**：无论怎样重排输入，最终输出完全相同。

不带位置编码的 full self-attention 层是排列等变的。如果后面再使用对所有 token 对称的 sum/mean pooling，得到的序列级结果才会进一步变成排列不变。

因此，更准确的表述不是“attention matrix 会对称”，而是：

> 不含位置机制的完整 self-attention 没有固定的序列坐标系，无法仅根据位置区分不同排列。

## 5. 为什么语言模型需要位置信息

内容 embedding 主要回答：

> 这个 token 是什么？

序列模型还必须回答：

- 这个 token 位于第几个位置；
- 哪个 token 在前，哪个在后；
- 两个 token 相距多远；
- 同一个 token 在不同位置是否承担不同作用。

“狗咬人”和“人咬狗”包含相同的 token 集合，但顺序改变了语义。对于无 causal mask 的完整 self-attention，如果没有任何位置机制，模型结构本身只会随 token 一起重排，不能把“第一个位置”和“第三个位置”赋予稳定、不同的意义。

### 5.1 绝对位置编码

绝对位置编码把位置向量 $p_i$ 注入 token 表示：

$$\widetilde x_i=x_i+p_i$$

即使两个位置使用相同 token，满足 $x_i=x_j$，通常仍有：

$$x_i+p_i\ne x_j+p_j$$

模型因此可以区分同一内容出现在不同绝对位置的情况。

### 5.2 RoPE

RoPE 不把位置向量加到输入，而是在每一层旋转 query 和 key：

$$\widetilde q_i=R_iq_i,\qquad \widetilde k_j=R_jk_j$$

注意力点积变成：

$$\widetilde q_i^\top\widetilde k_j=q_i^\top R_i^\top R_jk_j=q_i^\top R_{j-i}k_j$$

它以绝对位置 $i,j$ 执行旋转，但最终 score 可以直接感知相对位移 $j-i$。详细推导见 [02_08 位置编码与 RoPE 原理](./02_08_rope_explained.md)。

## 6. Causal mask 与位置编码不是一回事

Causal mask 规定：

$$M_{ij}=\begin{cases}0,&j\le i\\-\infty,&j>i\end{cases}$$

它回答的是：

> 第 $i$ 个 query 能不能访问第 $j$ 个 key？

显式位置编码回答的则是：

> 当前 token 位于哪里，两个 token 的相对距离是多少？

二者的职责可以概括为：

| 机制 | 注入的信息 | 主要目的 |
|---|---|---|
| Causal mask | $j\le i$ 的先后约束 | 禁止看到未来，保证自回归因果性 |
| 绝对位置编码 | token 的绝对索引 $i$ | 区分不同位置 |
| RoPE、relative bias、ALiBi | 相对位置或距离 $j-i$ | 让 attention score 感知相对关系 |

## 7. 为什么 NoPE causal Transformer 仍然能够工作

前面关于排列等变性的结论针对的是不含 causal mask 的完整 self-attention。加入 causal mask 后，任意排列不再保持相同的可见关系：

- 位置 0 只能看见 1 个 token；
- 位置 1 可以看见 2 个 token；
- 位置 $i$ 可以看见长度为 $i+1$ 的前缀。

因此 causal mask 本身已经把序列索引写进了计算图。不同位置拥有不同大小、不同内容的可见前缀，模型可以利用这种结构隐式推断顺序、绝对位置或相对位置。

这解释了 NoPE decoder-only Transformer 为什么并非完全“看不见位置”，也解释了 Assignment 1 为什么要求比较 RoPE 与 NoPE。相关研究表明，NoPE causal Transformer 可以完成语言建模，并可能表现出不错的长度外推能力。

但这不意味着显式位置编码没有价值。RoPE 等机制可以：

- 更直接地把相对距离写入 attention score；
- 提供清晰、可控的位置归纳偏置；
- 减少模型仅靠多层网络自行推断位置的负担；
- 配合 scaling 方法扩展长上下文。

所以“Transformer 必须加入位置编码”应分场景理解：

| 场景 | 是否必须显式加入位置编码 |
|---|---|
| 无 causal mask 的双向/full self-attention | 通常必须，否则结构保持排列等变，缺乏固定顺序信息 |
| 带 causal mask 的 decoder-only Transformer | 数学上不一定必须，NoPE 可以工作 |
| 主流生产 LLM | 通常仍使用 RoPE、ALiBi 等显式位置机制 |

## 8. Causal mask 对 score 和 probability 的影响

加 mask 前，$S$ 是完整的稠密方阵，但通常不对称。加 causal mask 后：

$$\widetilde S_{ij}=S_{ij}+M_{ij}$$

所有未来位置满足：

$$\widetilde S_{ij}=-\infty,\qquad j>i$$

逐行 softmax 后：

$$P_{ij}=0,\qquad j>i$$

所以 causal attention probability 具有下三角结构，显然通常不对称。

完整关系是：

| 对象 | 一般性质 |
|---|---|
| 无位置编码的 raw score $S=QK^\top/\sqrt{d_k}$ | 完整稠密，但通常不对称 |
| 特殊情形 $Q=K$ 的 raw score | 对称 Gram matrix |
| 对称 $S$ 经过逐行 softmax 后的 $P$ | 通常不对称 |
| 加 causal mask 后的 $P$ | 上三角为 0，通常不对称 |
| 无位置机制的 full self-attention 层 | 对 token 排列等变 |
| 带 causal mask 的 self-attention 层 | causal 结构已经打破任意排列等变 |

## 9. 三个反例快速排除混淆

### 9.1 无位置编码不代表 $S$ 对称

只要 $W_Q\ne W_K$，一般就有：

$$q_i^\top k_j\ne q_j^\top k_i$$

### 9.2 加入位置编码不代表 $S$ 不对称

如果加入位置后仍令 $Q=K$：

$$S=QQ^\top/\sqrt{d_k}$$

那么 $S$ 仍然对称。

### 9.3 $S$ 对称不代表 $P$ 对称

即使 $S_{ij}=S_{ji}$，逐行 softmax 的两个分母通常不同，因此 $P_{ij}\ne P_{ji}$。

## 10. 计算复杂度与可并行性

设序列长度为 $N$，单头维度为 $d_k$。

- 计算 $Q,K,V$ 的投影成本取决于模型维度和投影矩阵；attention score 的核心成本为 $O(N^2d_k)$。
- 绝对位置向量相加的成本为 $O(Nd_{\mathrm{model}})$。
- RoPE 对 Q/K 的旋转成本为 $O(Nd_k)$ 每个 head，低于 attention 的二次复杂度。
- 训练时已知完整输入序列，causal mask 不会迫使各位置串行计算；所有 token 的 Q/K/V、score 和 masked softmax 仍可并行。
- 自回归推理时，第 $t$ 个 token 依赖前面已生成的 token，时间维生成过程必须串行；这来自生成依赖，而不是位置编码本身。

## 11. 最终结论

1. 没有位置编码，不代表 score 矩阵对称。
2. $S$ 是否对称主要取决于 Q/K 的关系，而不是是否存在位置编码。
3. 即使 $S$ 对称，逐行 softmax 后的 $P$ 通常也不对称。
4. 无位置机制的完整 self-attention 的关键性质是排列等变。
5. Causal mask 通过前缀可见性提供隐式顺序信息，所以 NoPE causal LM 可以工作。
6. 显式位置编码仍能更直接、稳定地向模型提供绝对位置或相对距离。
7. “位置编码必须存在”对双向 full attention 是合理工程结论，对 causal decoder 则不是绝对数学定理。
