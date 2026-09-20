# 实现 Softmax：原理、数值稳定与验证

## 0. 本文目标

记录 CS336 assignment1 里 **softmax** 的实现思路：它算什么、为什么必须"减最大值"做数值稳定、在指定维度上如何用 `keepdim` 广播、adapter 怎么接到测试。

对应实际代码：
- 实现：[cs336_basics/nn.py](../cs336_basics/nn.py) 的 `softmax`
- 接线：[tests/adapters.py](../tests/adapters.py) 的 `run_softmax`
- 测试：[tests/test_nn_utils.py](../tests/test_nn_utils.py) 的 `test_softmax_matches_pytorch`

---

## 1. 背景：softmax 是什么

softmax 把一组任意实数（logits）变成一个**概率分布**：每项非负、且沿指定维度求和为 1。定义在向量 $x=(x_1,\dots,x_n)$ 上：

$$ \mathrm{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j} e^{x_j}} $$

它在 LLM 里无处不在：注意力打分归一化（把 $QK^\top/\sqrt d$ 变成注意力权重）、输出层把 logits 变成词表概率。直觉上它是"软化的 argmax"——最大的项拿到最多概率，但不是非 0 即 1，而是平滑分配。

**一个关键性质：平移不变。** 给所有输入同时加一个常数 $c$，结果不变：

$$ \mathrm{softmax}(x+c)_i = \frac{e^{x_i+c}}{\sum_j e^{x_j+c}} = \frac{e^c e^{x_i}}{e^c\sum_j e^{x_j}} = \mathrm{softmax}(x)_i $$

这个性质正是下面数值稳定技巧的理论依据。

---

## 2. 逐个决定的理由

### 2.1 接口与契约

来自 `run_softmax` / 测试：

- `softmax(x, dim)`：对**任意形状**的 `x`，在指定的 `dim` 上做归一化，输出与输入同形状；
- 无可学习参数（纯函数）。

### 2.2 为什么必须"减最大值"（数值稳定）

朴素实现 `exp(x) / exp(x).sum()` 有致命问题：**`exp` 很容易溢出**。float32 里 $e^{89}$ 就超出上限变成 `inf`，而注意力 logits 或大 embedding 值完全可能到几十上百。一旦 `exp` 得到 `inf`，`inf/inf` 就是 `nan`，整个前向坏掉。

利用 1 节的平移不变性，先把每个（沿 `dim` 的）切片**减去它的最大值**再取 exp：

$$ \mathrm{softmax}(x)_i = \frac{e^{x_i - \max_k x_k}}{\sum_j e^{x_j - \max_k x_k}} $$

这样指数里最大的项变成 $e^0=1$，其余都是 $e^{\le 0}\in(0,1]$，**绝不会上溢**；结果又因平移不变与原式完全相等。这就是实现里这几行的原因：

```python
x_max = x.max(dim=dim, keepdim=True).values
x_exp = torch.exp(x - x_max)
return x_exp / x_exp.sum(dim=dim, keepdim=True)
```

测试专门验证了这一点——`test_softmax_matches_pytorch` 不仅比对普通输入，还断言 **`softmax(x + 100)` 要和 `softmax(x)` 相等**：

```python
expected = F.softmax(x, dim=-1)
# 加 100 后仍要相等 —— 逼你必须做减最大值稳定化
run_softmax(x + 100, dim=-1) ≈ expected
```

朴素实现在 `x+100` 上会溢出成 `nan`，直接挂掉；减最大值的版本才能过。

### 2.3 `dim` 与 `keepdim` 的广播细节

- **`dim`**：沿哪个维度归一化就传哪个。注意力里通常是最后一维（每个 query 对所有 key 的分数之和为 1）；
- **`keepdim=True`**：`max`/`sum` 会把 `dim` 维**压掉**，比如 `(3, 5)` 沿 `dim=-1` 求和得 `(3,)`。加 `keepdim=True` 保留成 `(3, 1)`，才能和原 `(3, 5)` 做广播相减/相除。不保留就会形状不匹配或错误广播。

`x.max(dim=...)` 返回一个具名元组 `(values, indices)`，我们只要 `.values`。

### 2.4 是否手写 backward

不需要。`max`、`exp`、`sum`、除法都是可微算子（`max` 沿该维对最大元素传梯度），autograd 自动求导。减最大值这步在数学上不改变函数值，也**不影响梯度**（它对输入的贡献会被分子分母抵消）。

---

## 3. adapter 怎么接到测试

`run_softmax(in_features, dim)` 一步：转发到 `softmax(in_features, dim)`。无参数、无需 `load_state_dict`。

`test_softmax_matches_pytorch` 拿它和 `F.softmax(x, dim=-1)` 对比（`atol=1e-5`），并加测 `x+100` 的溢出场景。

---

## 4. 为什么测试能过

减最大值的实现与 `F.softmax` 数学等价、数值稳定，两个断言（普通 + 溢出）都满足。链路：

```text
test_softmax → run_softmax → softmax(减最大值 → exp → 归一化)
            → 与 F.softmax 逐元素比对（含 x+100 溢出用例） → PASS
```

运行：

```sh
uv run pytest -k test_softmax
```

预期 `1 passed`。

---

## 5. 小结

1. **是什么**：把 logits 变成概率分布，$\mathrm{softmax}(x)_i = e^{x_i}/\sum_j e^{x_j}$，沿指定维求和为 1。
2. **平移不变**：加常数不改变结果——数值稳定的理论依据。
3. **必减最大值**：`exp` 易溢出，减去该维 max 后最大指数为 $e^0=1$，绝不上溢，且结果不变；测试用 `x+100` 专门卡这一点。
4. **keepdim 广播**：`max`/`sum` 用 `keepdim=True` 保留维度，才能与原张量广播。
5. **无参数、backward 自动**：全可微算子；减最大值不影响梯度。

---

## 参考

- 本仓库实现：[cs336_basics/nn.py](../cs336_basics/nn.py)
- 适配层：[tests/adapters.py](../tests/adapters.py)
- 测试：[tests/test_nn_utils.py](../tests/test_nn_utils.py)
- 配套：softmax 是注意力的一环，后续 `scaled_dot_product_attention` 会直接用它。
