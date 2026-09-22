# 实现交叉熵损失（Cross-Entropy）：原理、数值稳定与验证

## 0. 本文目标

记录 CS336 assignment1 里**交叉熵损失** `cross_entropy` 的实现思路：它算什么、为什么等价于"负对数似然"、为什么必须用 log-sum-exp 做数值稳定、adapter 怎么接到测试。这是从"模型输出 logits"迈向"能训练"的第一步——有了损失才能反向传播。

对应实际代码：
- 实现：[cs336_basics/nn_utils.py](../cs336_basics/nn_utils.py) 的 `cross_entropy`
- 接线：[tests/adapters.py](../tests/adapters.py) 的 `run_cross_entropy`
- 测试：[tests/test_nn_utils.py](../tests/test_nn_utils.py) 的 `test_cross_entropy`

依赖前置：[softmax 笔记](./softmax_implementation_notes.md)（同款减最大值稳定技巧）、[TransformerLM 笔记](./transformer_lm_implementation_notes.md)（logits 的来源）。

---

## 1. 背景：交叉熵在语言模型里是什么

语言模型为每个位置输出一个词表上的分布 $p_\theta(x_{i+1}\mid x_{1:i})$。训练目标是让模型**给正确的下一个 token 尽量高的概率**。handout 的损失（式 16）就是所有位置的**负对数似然平均**：

$$ \ell(\theta;D)=\frac{1}{|D|\,m}\sum_{x\in D}\sum_{i=1}^{m}-\log p_\theta(x_{i+1}\mid x_{1:i}) $$

其中 $D$ 是训练集（工程上是一个 batch，$|D|$=batch 里序列数）、$m$ 是每条序列的预测位置数。**双重求和 = 在 batch 维和 seq 维上都遍历**，前面的 $\frac{1}{|D|m}$ 把它变成"每个 token 的平均损失"。

而单个位置的概率由 softmax 给出（式 17）：

$$ p(x_{i+1}\mid x_{1:i})=\mathrm{softmax}(o_i)[x_{i+1}]=\frac{\exp(o_i[x_{i+1}])}{\sum_a \exp(o_i[a])} $$

$o_i$ 是该位置的 logits 向量。所以**交叉熵 = 先 softmax 成概率、取正确类的概率、再取负对数**。

---

## 2. 从定义到可稳定计算的公式

### 2.1 展开负对数似然

对单个样本，设 logits 为 $o$、目标类为 $t$：

$$ -\log p(t) = -\log \frac{\exp(o[t])}{\sum_a \exp(o[a])} = -o[t] + \log\sum_a \exp(o[a]) $$

右边第二项 $\log\sum_a \exp(o[a])$ 就是 **log-sum-exp（LSE）**。所以：

$$ \text{loss} = \mathrm{LSE}(o) - o[t] $$

这个形式很关键——它**不先显式算出 softmax 概率**，而是直接用 logits 算，既省一步、又为数值稳定铺路。

### 2.2 为什么必须数值稳定（减最大值）

朴素地算 $\exp(o[a])$ 会**溢出**：logits 到几十上百时 $\exp$ 就变 `inf`（float32 里 $e^{89}$ 已溢出）。测试专门用 `1000×` 放大的 logits 卡这一点。

解法和 [softmax 笔记](./softmax_implementation_notes.md) 完全一样——利用 LSE 的**平移不变性**。令 $c=\max_a o[a]$：

$$ \mathrm{LSE}(o) = c + \log\sum_a \exp(o[a]-c) $$

减去最大值后，指数里最大是 $e^0=1$，绝不上溢；结果又与原式相等。对应实现：

```python
x_max = inputs.max(dim=-1, keepdim=True).values
shifted = inputs - x_max                              # 每行减最大值
log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1))
target_logit = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
loss = log_sum_exp - target_logit                     # = LSE(o) - o[t]
return loss.mean()
```

注意 `target_logit` 用的是 `shifted`（也减了 max）——因为 loss 里 $-o[t]$ 和 LSE 里都减同一个 $c$，一致抵消，结果不变。

### 2.3 逐行拆解

- **`inputs.max(dim=-1)`**：沿词表维取每个样本的最大 logit，用于稳定；
- **`gather(-1, targets)`**：按目标索引，从每行取出"正确类"的 logit。`targets` 形状 `(N,)`，`unsqueeze` 成 `(N,1)` 做索引，再 `squeeze` 回 `(N,)`；这是"按样本取对应类"的标准张量写法；
- **`loss.mean()`**：对所有样本（即展平后的 batch×seq）取平均，对应式 16 的 $\frac{1}{|D|m}$。

---

## 3. 接口与 adapter

### 3.1 契约

- `cross_entropy(inputs, targets)`：`inputs (N, vocab)` 是 logits，`targets (N,)` 是正确类索引，返回**标量**平均损失。
- 这里的 `N` 是"展平后的样本数"——语言模型里就是 `batch × seq`。测试正是先把 `(batch, seq, vocab)` 用 `.view(-1, vocab)` 压成 `(N, vocab)`、`targets` 压成 `(N,)` 再调用。

### 3.2 adapter

`run_cross_entropy` 一步转发到 `cross_entropy(inputs, targets)`。测试和 PyTorch 的 `F.cross_entropy` 对比（`atol=1e-4`），并加测 `1000×` 溢出场景。

---

## 4. 为什么测试能过

用 LSE−目标 logit 的稳定形式，与 `F.cross_entropy` 数学等价、数值稳定，普通和溢出两个断言都满足。链路：

```text
test_cross_entropy → run_cross_entropy → cross_entropy(减max → LSE − 目标logit → mean)
                  → 与 F.cross_entropy 比对（含 1000× 溢出） → PASS
```

运行：

```sh
uv run pytest -k test_cross_entropy
```

预期 `1 passed`。

---

## 5. 延伸：困惑度（Perplexity）

交叉熵用于**训练**，而**评估**语言模型时通常还要报告**困惑度（perplexity）**。它和交叉熵只差一个指数——理解了交叉熵，困惑度是顺带的。

**定义**：对一条长度 $m$、逐位置交叉熵损失为 $\ell_1,\dots,\ell_m$ 的序列，

$$ \mathrm{perplexity} = \exp\!\left(\frac{1}{m}\sum_{i=1}^{m}\ell_i\right) $$

即"**先对各位置的交叉熵取平均，再取 exp**"。因为我们的 `cross_entropy` 已经返回平均损失，困惑度就是 `exp(平均交叉熵)`，一行 `torch.exp(loss)` 即可。

**怎么理解这个数**：困惑度可解释为"模型在每一步平均在多少个候选 token 之间**犹豫**"。

- 完美预测（正确 token 概率=1）→ 交叉熵 0 → 困惑度 $e^0 = 1$（毫不犹豫）；
- 在词表 $V$ 个 token 上完全均匀瞎猜 → 每步交叉熵 $\ln V$ → 困惑度 $e^{\ln V}=V$（在全部 $V$ 个里平均犹豫）；
- 所以困惑度范围是 $[1, V]$，**越低越好**，数值上等于"等效的均匀候选数"。

**为什么用它而非直接看交叉熵**：两者单调等价（$\mathrm{ppl}=e^{\text{loss}}$），但困惑度有直观的"候选数"含义、且是语言建模领域的历史习惯，便于横向比较不同模型。训练时监控 loss，评估/汇报时换算成 perplexity。

> 数值提醒：算困惑度要用**自然对数**下的交叉熵（`loss` 用 $\ln$），与 $\exp$ 配套；我们的实现正是自然对数，直接 `exp(loss)` 即可。若在很长序列上评估，应按 §1 的方式对所有位置的 $\ell_i$ 求平均再 exp，而不是对每段分别 exp 再平均。

---

## 6. 小结

1. **是什么**：交叉熵 = 负对数似然，$\text{loss}=-\log p(\text{目标类})$，$p$ 由 logits 经 softmax 得到。
2. **等价形式**：$\text{loss}=\mathrm{LSE}(o)-o[t]$，直接用 logits 算，不必先显式 softmax。
3. **必须稳定**：`exp` 易溢出，减每行最大值（LSE 平移不变）后最大指数为 $e^0=1$，绝不上溢；测试用 `1000×` 卡这一点。
4. **gather 取目标**：`gather(-1, targets)` 按样本取正确类 logit。
5. **平均维度**：`mean()` 对 batch×seq 展平后取平均，对应式 16 的 $\frac{1}{|D|m}$。

有了损失，就可以反向传播、配合优化器训练了——下一步是 AdamW、梯度裁剪与学习率调度。

---

## 参考

- 本仓库实现：[cs336_basics/nn_utils.py](../cs336_basics/nn_utils.py)
- 适配层：[tests/adapters.py](../tests/adapters.py)
- 测试：[tests/test_nn_utils.py](../tests/test_nn_utils.py)
- 依赖：[softmax_implementation_notes.md](./softmax_implementation_notes.md)（同款 log-sum-exp 稳定）、[transformer_lm_implementation_notes.md](./transformer_lm_implementation_notes.md)（logits 来源）
- Handout：[cs336_assignment1_basics_extracted.md](./cs336_assignment1_basics_extracted.md)（式 16、17）
