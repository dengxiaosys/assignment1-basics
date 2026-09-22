# 实现缩放点积注意力（Scaled Dot-Product Attention）：原理、掩码与验证

## 0. 本文目标

记录 CS336 assignment1 里 **缩放点积注意力（SDPA）** 的实现思路：公式怎么来、为什么要除以 $\sqrt{d_k}$、布尔 mask 怎么用、adapter 怎么接到测试。这是注意力机制的**核心运算**，多头注意力就是在它外面套投影和拆头。

对应实际代码：
- 实现：[cs336_basics/model.py](../cs336_basics/model.py) 的 `scaled_dot_product_attention`
- 接线：[tests/adapters.py](../tests/adapters.py) 的 `run_scaled_dot_product_attention`
- 测试：[tests/test_model.py](../tests/test_model.py) 的 `test_scaled_dot_product_attention`、`test_4d_scaled_dot_product_attention`

依赖前置：[softmax 笔记](./softmax_implementation_notes.md)（本函数直接调用它）。

---

## 1. 背景：注意力在算什么

注意力的一句话直觉：**每个 query 去和所有 key 比相似度，按相似度对 value 加权求和**。得到的输出是"综合了相关 token 内容"的新表示。

公式（Vaswani et al. 2017）：

$$ \mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V $$

三个输入的角色：

- **Q（query）**：`(..., queries, d_k)`，"我想找什么"；
- **K（key）**：`(..., keys, d_k)`，"我有什么特征可被检索"；
- **V（value）**：`(..., keys, d_v)`，"被检索到时我贡献什么内容"。

分三步：

1. **打分**：$QK^\top$ 得 `(..., queries, keys)`，第 $(i,j)$ 元是 query $i$ 与 key $j$ 的点积相似度；
2. **缩放 + softmax**：除以 $\sqrt{d_k}$ 后沿 key 维 softmax，变成每个 query 的一组注意力权重（对所有 key 求和为 1）；
3. **加权求和**：权重乘 $V$，得 `(..., queries, d_v)`。

---

## 2. 逐个决定的理由

### 2.1 接口与契约

来自 `run_scaled_dot_product_attention` / 测试：

- `scaled_dot_product_attention(Q, K, V, mask=None)`；
- `Q (..., queries, d_k)`、`K (..., keys, d_k)`、`V (..., keys, d_v)`，输出 `(..., queries, d_v)`；
- `mask` 是布尔 `(..., queries, keys)`，可选；
- 前置的 `...` 是任意批量维（batch、head 都塞在这里）——所以同一份实现既能过 3D 测试，也能过 4D 多头测试（`test_4d_...` 把形状 reshape 成 `(batch, head, seq, d)`）。无可学习参数。

### 2.2 为什么除以 $\sqrt{d_k}$

这是"scaled"的由来，也是最容易被追问的点。若 $q,k$ 各维独立、均值 0 方差 1，则点积 $q\cdot k=\sum_{l=1}^{d_k} q_l k_l$ 的**方差正比于 $d_k$**、标准差约 $\sqrt{d_k}$。$d_k$ 越大，打分的数值波动越大。

问题在于：**softmax 对输入尺度极敏感**。分数一大，softmax 会趋近 one-hot（几乎全押一个 key），梯度也随之变得极小（饱和区），训练困难。除以 $\sqrt{d_k}$ 把分数方差拉回约 1，softmax 保持"温度适中"，梯度健康。这就是实现里：

```python
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
```

`K.transpose(-2, -1)` 只换最后两维（`keys` 与 `d_k`），使 `(..., keys, d_k)` 变 `(..., d_k, keys)`，与 `Q` 相乘得 `(..., queries, keys)`；用 `transpose(-2,-1)` 而非 `.T` 是为了对任意前置批量维安全（同 [linear 笔记](./linear_implementation_notes.md) 的转置讨论）。

### 2.3 布尔 mask：True 保留、False 屏蔽

CS336 约定：mask 里 **`True` 表示该 (query, key) 对参与注意力，`False` 表示屏蔽**。屏蔽的做法是把对应分数置为 $-\infty$：

```python
if mask is not None:
    scores = scores.masked_fill(~mask, float("-inf"))
```

- 为什么置 $-\infty$：softmax 里 $e^{-\infty}=0$，被屏蔽的 key 权重变成 0，等于"看不见"它；
- `~mask` 取反，因为 `masked_fill(m, val)` 是把 **m 为 True** 的位置填 `val`——我们要填的是"要屏蔽"的位置，即原 mask 为 False 处；
- 这套机制正是**因果掩码**（每个位置只能看自己和之前）和 **padding 掩码** 的统一实现。本测试用随机布尔 mask 验证一般性。

> 数值细节：某个 query 若整行 key 都被屏蔽，会得到全 $-\infty$，softmax 出 `nan`。本作业的随机 mask 不会构造这种退化行，故无需特殊处理；真实因果掩码也保证每个位置至少能看到自己，不会整行屏蔽。

### 2.4 复用 softmax、是否手写 backward

- **复用**：直接调 [2.2 节的 `softmax`](./softmax_implementation_notes.md)（沿 `dim=-1`，即 key 维），它自带减最大值的数值稳定；
- **backward**：矩阵乘、缩放、`masked_fill`、softmax 全是可微算子，autograd 自动求导，无需手写。无可学习参数。

---

## 3. adapter 怎么接到测试

`run_scaled_dot_product_attention(Q, K, V, mask)` 一步转发到 `scaled_dot_product_attention(Q, K, V, mask)`。无参数、无需 `load_state_dict`。

两个测试：

- `test_scaled_dot_product_attention`：3D 输入 `(batch, seq, d)` + 随机 mask，输出比对快照（`atol=1e-5`）；
- `test_4d_scaled_dot_product_attention`：把 Q/K/V/mask reshape 成 `(batch, head, seq, d)` 的 4D，验证实现对**任意前置批量维**都成立——靠的就是 2.1 里 `...` 的写法和 `transpose(-2,-1)` 的维度安全。

---

## 4. 为什么测试能过

打分、缩放、掩码、softmax、加权求和都与参考一致，且 `...`/`transpose(-2,-1)` 保证 3D、4D 通吃。链路：

```text
test_sdpa → run_scaled_dot_product_attention
         → scaled_dot_product_attention(QK^T/√d_k → mask 置 -inf → softmax → @V)
         → numpy_snapshot 对照 _snapshots → PASS
```

运行：

```sh
uv run pytest -k scaled_dot_product
```

预期 `2 passed`。

---

## 5. 小结

1. **公式**：$\mathrm{softmax}(QK^\top/\sqrt{d_k})V$——打分、缩放归一、加权求和三步。
2. **为什么缩放**：点积方差 $\propto d_k$，不缩放会让 softmax 饱和、梯度消失；除以 $\sqrt{d_k}$ 稳住尺度。
3. **布尔 mask**：True 保留、False 屏蔽；`masked_fill(~mask, -inf)` 让被屏蔽 key 的权重 softmax 后归 0（因果/padding 掩码统一机制）。
4. **任意批量维**：用 `...` 和 `transpose(-2,-1)`，同一实现通吃 3D 与 4D 多头。
5. **复用 softmax、无参数、backward 自动**：直接调数值稳定的 softmax，全可微。

---

## 参考

- 本仓库实现：[cs336_basics/model.py](../cs336_basics/model.py)
- 适配层：[tests/adapters.py](../tests/adapters.py)
- 测试：[tests/test_model.py](../tests/test_model.py)
- 依赖：[softmax_implementation_notes.md](./softmax_implementation_notes.md)
- 相关：[rope_explained.md](./rope_explained.md)（RoPE 在注意力打分前作用于 Q/K）、[linear_implementation_notes.md](./linear_implementation_notes.md)（转置的维度安全）
- 原始文献：Vaswani et al., *Attention Is All You Need*, 2017。
