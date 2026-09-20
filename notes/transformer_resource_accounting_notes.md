# Transformer 资源核算（FLOPs / 参数量）：原理、方法与逐题解答

## 0. 本文目标

回答 CS336 assignment1 的 `transformer_accounting` 问题：一个 Transformer LM 有多少参数、前向一次要多少 FLOPs、哪些部件最耗算力、模型规模与上下文长度如何改变各部件占比。同时把背后的**核算方法**讲清楚，作为理解"模型为什么这么贵"的基础。

对应 handout：[cs336_assignment1_basics_extracted.md](./cs336_assignment1_basics_extracted.md) 的 `#### Resource accounting` 与 `Problem (transformer_accounting)`。

> 说明：这是**分析题**，本文给出推导、公式与数值结论；数值由脚本按"我们的作业架构"精确计算（SwiGLU 三矩阵 FFN、RoPE、RMSNorm、无 bias）。与真实 GPT-2 的差异见 §5。

---

## 1. 核算方法：为什么只数矩阵乘

**核心事实**：Transformer 前向的绝大多数 FLOPs 来自矩阵乘法。逐元素操作（RMSNorm、SiLU、残差加、softmax 的 exp）相对可忽略。所以核算分两步：写出所有矩阵乘 → 按规则换算 FLOPs。

**矩阵乘 FLOPs 规则**：$A\in\mathbb{R}^{m\times n}$、$B\in\mathbb{R}^{n\times p}$，则 $AB$ 需要 $2mnp$ FLOPs。

原因：输出有 $m\times p$ 个元素，每个是长度 $n$ 的点积（$n$ 次乘 + $n$ 次加 = $2n$ FLOPs），合计 $2mnp$。这个"**2 × 输入维 × 输出元素数**"的口诀是全部核算的基石。

---

## 2. 参数量与 FLOPs 的通用公式

设 `vocab_size = V`、`context_length = seq`、`num_layers = L`、`d_model = d`、`d_ff = f`。我们的架构里 $d_k = d/\text{num\_heads}$，QKVO 四个投影都是 $d\times d$。

### 2.1 参数量

| 部件 | 参数量 |
|---|---|
| token 嵌入 | $V d$ |
| 每层：QKVO 四投影 | $4 d^2$ |
| 每层：FFN（SwiGLU 三矩阵 $W_1,W_3\in\mathbb{R}^{f\times d}$，$W_2\in\mathbb{R}^{d\times f}$） | $3 d f$ |
| 每层：两个 RMSNorm 增益 | $2 d$ |
| 末端 ln_final | $d$ |
| lm_head | $d V$ |

总计：$\;V d + L(4d^2 + 3df + 2d) + d + dV$。

### 2.2 前向 FLOPs（矩阵乘，输入 seq 个 token）

| 部件 | 矩阵乘 | FLOPs |
|---|---|---|
| QKVO 投影（每层） | 4 个 $(seq,d)\times(d,d)$ | $8\,seq\,d^2$ |
| 注意力打分 + 加权（每层） | $QK^\top$ 与 $\cdot V$ | $4\,seq^2 d$ |
| FFN（每层，SwiGLU 三矩阵） | $2(seq,d)(d,f)+1(seq,f)(f,d)$ | $6\,seq\,d f$ |
| lm_head（一次） | $(seq,d)\times(d,V)$ | $2\,seq\,d V$ |

总计：$\;L\,(8\,seq\,d^2 + 4\,seq^2 d + 6\,seq\,d f) + 2\,seq\,d V$。

> 注意两类项的**量纲**：投影和 FFN 随 $seq$ **线性**增长（含 $d^2$、$df$），注意力打分随 $seq^2$ **平方**增长（含 $seq^2 d$）。这个区别是后面 (e) 的关键。

---

## 3. 逐题解答

### (a) GPT-2 XL 的参数量与加载显存

配置：$V{=}50257,\ seq{=}1024,\ L{=}48,\ d{=}1600,\ \text{heads}{=}25,\ f{=}4288$。

- **可训练参数：约 1.64B**（1,640,452,800）。
- **单精度加载显存**：$1.64\text{B}\times 4\,\text{bytes} \approx 6.11\ \text{GiB}$。

（仅"加载模型权重"的显存；训练时还要加梯度、优化器状态、激活，实际远不止此。）

### (b) 前向所需的矩阵乘与总 FLOPs

矩阵乘清单（每层）：Q/K/V/O 四个投影、$QK^\top$、$\text{attn}\cdot V$、FFN 的 $W_1/W_3/W_2$；最后 lm_head 一次。

- **GPT-2 XL 前向总 FLOPs ≈ 3.52 TFLOPs**（3,516,769,894,400）。

分部件：

| 部件 | FLOPs | 占比 |
|---|---:|---:|
| FFN | 2.02 T | **57.5%** |
| QKVO 投影 | 1.01 T | 28.6% |
| 注意力（$QK^\top,\cdot V$） | 0.32 T | 9.2% |
| lm_head | 0.16 T | 4.7% |

### (c) 哪部分最耗 FLOPs

**FFN 最贵**（约 57.5%），其次是 QKVO 投影（约 28.6%）。即在 `seq=1024` 这个尺度下，**线性层（FFN + 投影）合计约 86% 的 FLOPs**，而 $seq^2$ 的注意力打分只占约 9%。

### (d) 四个规模的分解与趋势

各模型前向 FLOPs 占比（`seq=1024`）：

| 部件 | small | medium | large | XL |
|---|---:|---:|---:|---:|
| FFN | 39.8% | 50.1% | 54.3% | **57.5%** |
| QKVO 投影 | 19.9% | 24.8% | 27.3% | 28.6% |
| 注意力 | 13.3% | 12.4% | 10.9% | 9.2% |
| lm_head | 27.1% | 12.7% | 7.4% | 4.7% |
| 总 FLOPs | 0.29 T | 0.83 T | 1.77 T | 3.52 T |

（参数量：small 0.16B、medium 0.41B、large 0.83B、XL 1.64B。）

**趋势**：随模型变大（$d$、$L$ 增加），**FFN 与投影的占比上升、lm_head 与注意力的占比下降**。原因：FFN/投影 $\propto L d^2$（随深度和宽度平方涨），而 lm_head $\propto dV$（$V$ 固定、只随 $d$ 线性涨，被摊薄），注意力打分 $\propto L\,seq^2 d$（$seq$ 固定时相对 $d^2$ 项也被摊薄）。所以大模型的算力越来越集中在"逐位置的线性变换"上。

### (e) GPT-2 XL 把 context 增到 16,384

- **总 FLOPs 从 3.52 T 暴增到约 133.6 T**（约 38 倍，而 seq 只增大 16 倍——超线性）。
- 各部件占比剧变：

| 部件 | seq=1024 | seq=16384 |
|---|---:|---:|
| 注意力（$seq^2$） | 9.2% | **61.7%** |
| FFN | 57.5% | 24.2% |
| QKVO 投影 | 28.6% | 12.1% |
| lm_head | 4.7% | 2.0% |

**结论**：长上下文下，$O(seq^2)$ 的注意力打分从"次要"跃升为**主导项**（超过六成）。这正是长序列如此昂贵、以及 FlashAttention / 稀疏注意力 / 线性注意力等优化聚焦于此的根本原因。

---

## 4. 关键直觉

1. **FLOPs 口诀**：矩阵乘 $2mnp$ = 2 × 输入维 × 输出元素数；数模型只需数矩阵乘。
2. **两种增长量纲**：线性层（投影/FFN）$\propto seq \cdot d^2$；注意力打分 $\propto seq^2 \cdot d$。谁主导取决于 $seq$ 与 $d$ 的相对大小。
3. **常规上下文（seq≈d 量级）**：FFN + 投影主导（约 85%+），注意力占比小。
4. **长上下文（seq≫d）**：注意力的 $seq^2$ 项主导，成为瓶颈。
5. **模型越大**：算力越集中于逐位置线性变换；固定词表的 lm_head 被摊薄。

---

## 5. 与真实 GPT-2 的差异（避免混淆）

上面的数按**本作业架构**算，和原版 GPT-2 有别，故参数量不完全等于公开数字：

- **FFN 用 SwiGLU 三矩阵**（$3df$），原 GPT-2 是 GELU 两矩阵（$2\times 4d^2$）；
- **位置编码用 RoPE**（无参数），原 GPT-2 是可学习绝对位置嵌入（有 $seq\cdot d$ 参数）；
- **归一化用 RMSNorm**（仅增益 $d$），原版 LayerNorm 还有偏置；
- **无 bias**、**lm_head 独立**（未与嵌入权重共享）。

所以本文的 XL≈1.64B 是"作业架构下"的结果，用于理解**相对占比与趋势**，而非复现 GPT-2 的确切参数。

---

## 参考

- Handout：[cs336_assignment1_basics_extracted.md](./cs336_assignment1_basics_extracted.md)（Resource accounting 一节）
- 组件参数来源：[transformer_lm_implementation_notes.md](./transformer_lm_implementation_notes.md)、[multihead_attention_implementation_notes.md](./multihead_attention_implementation_notes.md)、[swiglu_implementation_notes.md](./swiglu_implementation_notes.md)
- 原始文献：Vaswani et al., *Attention Is All You Need*, 2017；Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2), 2019。
