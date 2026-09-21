# AdamW 训练资源核算：显存、FLOPs 与训练时长

## 0. 本文目标

回答 CS336 assignment1 的 `adamw_accounting` 问题：用 AdamW 训练一个 Transformer LM 需要多少**显存**（参数/激活/梯度/优化器状态四块）、一步多少 **FLOPs**、以及在单张 H100 上训练 GPT-2 XL 要多久。并把训练显存构成、fwd/bwd 计算比、MFU 等背景讲清。

对应 handout：[cs336_assignment1_basics_extracted.md](./cs336_assignment1_basics_extracted.md) 的 `Problem (adamw_accounting)`。前置：[FLOPs 核算笔记](./transformer_resource_accounting_notes.md)（矩阵乘 $2mnp$、前向 FLOPs 公式）、[AdamW 笔记](./adamw_implementation_notes.md)（优化器状态 $m,v$）。

> 说明：分析题。数值按脚本精确计算，`float32`（4 字节/元素），$d_{ff}=\frac{8}{3}d_{model}$（本题不取 64 倍数）。

---

## 1. 背景：训练显存的四大块

推理只需存参数，**训练**要同时容纳四类张量（全 float32，每元素 4 字节）：

| 类别 | 是什么 | 大小（元素数） |
|---|---|---|
| **参数（Parameters）** | 模型权重 $\theta$ | $P$ |
| **梯度（Gradients）** | 每个参数一个 $\partial L/\partial\theta$ | $P$（与参数一一对应） |
| **优化器状态（Optimizer state）** | AdamW 的一阶矩 $m$ + 二阶矩 $v$ | $2P$（每参数两个，见 [AdamW 笔记](./adamw_implementation_notes.md)） |
| **激活（Activations）** | 前向各层的中间结果，反向要用 | 正比于 `batch_size × 各层输出规模` |

前三块都**只依赖参数量 $P$**（$1+1+2=4$ 倍 $P$），与 batch 无关；**只有激活随 batch 线性增长**。这是理解"最大 batch"的关键。

---

## 2. (a) 各部分显存的代数表达式

记 `batch_size=B`、`vocab_size=V`、`context_length=s`、`num_layers=L`、`d_model=d`、`num_heads=H`、$d_{ff}=\frac{8}{3}d$。

### 2.1 参数量 $P$

$$ P = \underbrace{Vd}_{\text{嵌入}} + L\underbrace{(4d^2 + 3d\,d_{ff} + 2d)}_{\text{每层:QKVO+FFN三矩阵+2 RMSNorm}} + \underbrace{d}_{\text{final norm}} + \underbrace{dV}_{\text{lm\_head}} $$

### 2.2 梯度与优化器状态

- 梯度：$P$（每参数一个）；
- 优化器状态：$2P$（AdamW 每参数存 $m$ 和 $v$）。

### 2.3 激活（按 handout 指定的组件，逐层，乘 $B$）

每个 Transformer 层：

$$ \underbrace{2sd}_{\text{2×RMSNorm}} + \underbrace{3sd}_{\text{QKV投影}} + \underbrace{2Hs^2}_{QK^\top+\text{softmax}} + \underbrace{sd}_{\text{加权和}} + \underbrace{sd}_{\text{输出投影}} + \underbrace{4s\,d_{ff}}_{W_1,W_3,\text{SiLU},\text{逐元素积}} + \underbrace{sd}_{W_2} $$

> **FFN（SwiGLU）的激活到底算了哪些？门×值的乘积算了吗？** 算了——它就是上式 $4s\,d_{ff}$ 里的"逐元素积"那一项。把 SwiGLU 的五个组件拆开看（回顾 [SwiGLU 笔记](./swiglu_implementation_notes.md)：$\mathrm{FFN}(x)=W_2(\mathrm{SiLU}(W_1x)\odot W_3x)$）：
>
> | FFN 组件 | 输出形状 | 激活元素数 |
> |---|---|---|
> | $W_1 x$（门支路升维） | $(s, d_{ff})$ | $s\,d_{ff}$ |
> | $W_3 x$（值支路升维） | $(s, d_{ff})$ | $s\,d_{ff}$ |
> | $\mathrm{SiLU}(W_1x)$（门激活） | $(s, d_{ff})$ | $s\,d_{ff}$ |
> | $\mathrm{SiLU}(W_1x)\odot W_3x$（**门×值 逐元素积**） | $(s, d_{ff})$ | $s\,d_{ff}$ |
> | $W_2(\dots)$（降维回 $d$） | $(s, d)$ | $s\,d$ |
>
> 前四项都在 $d_{ff}$ 维、各 $s\,d_{ff}$，合计 $4s\,d_{ff}$（就是公式里那一项，**含你问的门值乘积**）；$W_2$ 的输出在 $d$ 维，单独是 $sd$。这正好对应 handout 列的 FFN 组件（$W_1$、$W_2$、SiLU、逐元素积、$W_3$）。所以门×值乘积没有漏，只是被并进了 $4s\,d_{ff}$。

再加末端：final RMSNorm $sd$ + 输出 logits $sV$ + 交叉熵 $sV$。总激活元素数：

$$ A = B\Big[L\big(8sd + 2Hs^2 + 4s\,d_{ff}\big) + sd + 2sV\Big] $$

### 2.4 总显存

> 记号约定（避免混淆）：本文 **$B$ 一律指 batch_size**；**字节**一律写"字节"、不缩写；十亿写作 $\times10^9$。

元素总数 = 前三块 $4P$ + 激活 $A$。每个元素是 float32、占 **4 字节**，所以：

$$ \text{显存(字节)} = \big(\underbrace{4P}_{\text{params+grads+opt}} + \underbrace{A}_{\text{激活}}\big)\times 4 $$

其中 batch_size $B$ 只出现在激活 $A=B[\dots]$ 里；前三块 $4P$ 与 $B$ 无关，激活 $\propto B$。

---

## 3. (b) GPT-2 XL 实例化与最大 batch

代入 $V{=}50257,\ s{=}1024,\ L{=}48,\ d{=}1600,\ H{=}25,\ d_{ff}{=}\frac83\cdot1600{\approx}4266$：

- **参数量** $P \approx 1.635\times10^9$；
- **参数+梯度+优化器状态** $=4P$ 个元素 $\times 4\,\text{字节} \approx$ **24.37 GiB**（$6.09+6.09+12.18$）；
- **激活** $\approx (4.089\times10^9)\times B$ 个元素 $\times 4\,\text{字节} \approx$ **15.23 GiB × $B$**（$B$=batch_size）。

于是总显存（GiB）关于 batch 的表达式：

$$ \boxed{\text{Memory} \approx 15.23\,B + 24.37\ \text{GiB}} $$

**80GB 内的最大 batch**：$15.23B + 24.37 \le 80 \Rightarrow B \le 3.65$，即**最大 batch_size = 3**。

> 直观结论：对 XL 这种规模，**激活是显存大头**（每个 batch 就要 15 GiB），固定开销（$4P$）24 GiB 也不小。这解释了为何大模型训练要靠梯度检查点（重算激活省显存）、混合精度、多卡切分等手段。

---

## 4. (c) 一步 AdamW 的 FLOPs

先约定基准：下面的 fwd/bwd 都指**单条序列**（$s$ 个 token）的 FLOPs；一个 batch 有 $B$ 条**相互独立**的序列，各自走完整前向和反向、计算量相同，所以整个 batch 就是**单序列结果 × $B$**。因此 **fwd、bwd 都与 $B$ 成正比**（这是"直接乘 $B$"的依据）。

一步 = 前向 + 反向 + 优化器更新，逐项分开列：

- **前向（每序列）**：矩阵乘为主（见 [FLOPs 笔记](./transformer_resource_accounting_notes.md)），
  $$ \text{fwd} = L(8sd^2 + 4s^2d + 6s\,d_{ff}\,d) + 2sdV $$
  GPT-2 XL 约 **3.51 TFLOPs/序列**。
- **反向（每序列）**：约为前向的 **2 倍**（Kaplan/Hoffmann 惯例，理由见 §6），单独就是
  $$ \text{bwd} \approx 2\,\text{fwd} \approx 7.01\ \text{TFLOPs/序列}。 $$
- **优化器更新（AdamW 本身）**：逐元素的 $m,v$ 滑动平均与参数更新，约 $O(P)$，**与 $s$、$B$ 无关**，相对上面的 T 量级**可忽略**。

把三者合起来（这一步才做加总）：

$$ \text{fwd} + \text{bwd} \approx 3\,\text{fwd} \approx 10.52\ \text{TFLOPs/序列} $$

$$ \text{FLOPs/step} \approx (\text{fwd}+\text{bwd})\times B \approx 10.52\text{T}\times B \quad(\text{再加可忽略的 }O(P)\text{ 优化器项}) $$

其中 $B$=batch_size：一个训练步处理 $B$ 条序列，故乘 $B$。

> **反向是对标量 loss 调用的，为什么还 ∝ B？** 这是最容易困惑的一点。关键：**`loss.backward()` 的成本不取决于"loss 是不是标量"，而取决于"产生这个 loss 的计算图有多大"**——而这张图正是 $B$ 条序列的完整前向，规模 $\propto B$。
>
> **1. 标量只是反向的"起点种子"。** loss 是把 $B\times s$ 个 token 的交叉熵**平均**成的一个数：$\text{loss}=\frac{1}{Bs}\sum_i \ell_i$。反向从 $\frac{\partial \text{loss}}{\partial \text{loss}}=1$ 出发，但**下一步**就是 $\frac{\partial \text{loss}}{\partial \text{logits}}$——它的形状和 logits 一样是 $(B, s, V)$，**已经是 $B$ 规模的张量了**。从这里再往回传，每一层的激活梯度都和该层前向激活同形状（都含 $B$ 维）。所以"标量"只在第一步，之后全程是 $B$ 大小的梯度张量。
>
> **2. 反向要"逐操作地微分"整张前向图。** 前向为了算出这个标量，做了 $B$ 条序列 × 每层的全部矩阵乘（$\propto B$）。反向按链式法则**把每个前向操作都对应地求一次导**，遍历同一张图一遍。以一个矩阵乘 $Y=XW$ 为例，反向要算两个同规模的矩阵乘：$\frac{\partial L}{\partial X}=\frac{\partial L}{\partial Y}W^\top$、$\frac{\partial L}{\partial W}=X^\top\frac{\partial L}{\partial Y}$。这里 $X$、$\frac{\partial L}{\partial Y}$ 都带 $B$ 维，所以这两个反向矩阵乘也 $\propto B$——这正是"反向约 2× 前向"的来源（两个 matmul vs 前向一个）。
>
> **3. 平均只改梯度的"大小"，不改"要算多少个"。** $\frac{1}{Bs}$ 这个系数只是把所有梯度整体缩小，**不减少需要计算的梯度数量**：每条序列、每个 token、每个参数仍各自有一份梯度要沿图回传。$B$ 越大，图越大，要回传的中间梯度越多，FLOPs 越多。
>
> **一句话**：loss 是标量，但它是 $B$ 条序列"汇总"出来的；反向要把这个汇总**摊回**到每条序列的每个中间量上，工作量自然 $\propto B$。可以类比前向——前向做了 $B$ 份工作才得到这个标量，反向就要对这 $B$ 份工作各求一次导。

---

## 5. (d) 单张 H100 训练 GPT-2 XL 要多久

条件：400K 步、batch 1024、H100 峰值 495 TFLOP/s、**MFU 50%**、bwd=2×fwd。

- 每步 FLOPs：$10.52\text{T} \times 1024 \approx 1.077\times10^{16}$；
- 总 FLOPs：$\times\,400\text{K} \approx 4.31\times10^{21}$；
- 有效算力：$495\text{T}\times0.5 = 247.5\text{TFLOP/s}$；
- 时间：$\dfrac{4.31\times10^{21}}{2.475\times10^{14}} \approx 1.74\times10^{7}\text{s}$。

$$ \boxed{\approx 4836\ \text{小时} \approx 202\ \text{天} \approx 0.55\ \text{年}} $$

**结论**：单卡训 GPT-2 XL 到这个步数要约**半年**——直观说明了为什么大模型必须**多卡/集群并行**。

---

## 6. 背景补充：几个关键概念

- **为什么优化器状态是 $2P$**：AdamW 每个参数要存一阶矩 $m$ 和二阶矩 $v$（见 [AdamW 笔记](./adamw_implementation_notes.md)）。所以 Adam 类训练的固定显存是 $4P$（params+grads+$m$+$v$），比 SGD（无状态，$2P$）翻倍——这是自适应优化器的显存代价。
- **为什么 bwd ≈ 2×fwd**：反向要对每个矩阵乘算"对输入的梯度"和"对权重的梯度"两个矩阵乘，各约等于一次前向 matmul，故约 2 倍（Kaplan 2020、Hoffmann 2022 的通用近似）。合计一步约 3×fwd。
- **MFU（Model FLOPs Utilization）**：实测吞吐 / 硬件理论峰值。50% 已是相当好的工程水平——受访存带宽、通信、kernel 效率等限制，很难跑满峰值。核算训练时间必须乘上 MFU，否则会严重低估。
- **激活显存为何常是瓶颈**：它 $\propto B\times L\times(sd + s^2H)$，长上下文时 $s^2$ 项还会爆炸（见 [FLOPs 笔记](./transformer_resource_accounting_notes.md) 的长上下文分析）。这催生了梯度检查点、FlashAttention（不显式存 $s\times s$ 注意力矩阵）等技术。

---

## 7. 小结

1. **训练显存四块**：params $P$ + grads $P$ + AdamW 状态 $2P$ + 激活（$\propto B$）。前三块 $=4P$ 与 batch 无关。
2. **(a)** 给出各块代数式；总元素数 $= 4P + A$，显存 $=(4P+A)\times4\,\text{字节}$。
3. **(b)** GPT-2 XL：$\approx 15.23B + 24.37$ GiB，80GB 内**最大 batch=3**。
4. **(c)** 一步 $\approx 3\times$前向 $\approx 10.52\text{T}\times B$ FLOPs；AdamW 逐元素部分 $O(P)$ 可忽略。
5. **(d)** 400K 步、batch 1024、50% MFU：单 H100 约 **4836 小时（≈202 天）**。
6. **洞见**：优化器状态翻倍显存、激活是大头、单卡训大模型要以月计——并行与省显存技术的动机所在。

---

## 参考

- Handout：[cs336_assignment1_basics_extracted.md](./cs336_assignment1_basics_extracted.md)（adamw_accounting）
- 前置：[transformer_resource_accounting_notes.md](./transformer_resource_accounting_notes.md)（FLOPs 核算）、[adamw_implementation_notes.md](./adamw_implementation_notes.md)（优化器状态）
- 文献：Kaplan et al. 2020、Hoffmann et al. 2022（fwd/bwd 比例、scaling）；Chowdhery et al. 2022（MFU）。
