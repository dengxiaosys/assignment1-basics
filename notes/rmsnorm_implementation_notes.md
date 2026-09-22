# 实现 RMSNorm：思路、数值细节与验证

## 0. 本文目标

记录 CS336 assignment1 里 `RMSNorm` 的实现思路，重点讲**它算什么、为什么要在 float32 上算、weight 是什么**。结构与 [linear_implementation_notes.md](./linear_implementation_notes.md)、[embedding_implementation_notes.md](./embedding_implementation_notes.md) 一致。

对应实际代码：
- 实现：[cs336_basics/model.py](../cs336_basics/model.py) 的 `RMSNorm`
- 接线：[tests/adapters.py](../tests/adapters.py) 的 `run_rmsnorm`
- 测试：[tests/test_model.py](../tests/test_model.py) 的 `test_rmsnorm`

RMSNorm 的原理、与 LayerNorm 的区别、以及"增益 $g_i$ 为何每块独立"已在 [prenorm_vs_postnorm_explained.md](./prenorm_vs_postnorm_explained.md) 第 6 节讲过，本文不重复推导，聚焦实现。

---

## 1. RMSNorm 算什么

对输入向量 $x$ 的**最后一维**（`d_model`）做归一化：**只按均方根缩放，不减均值、不加 bias**。

$$ y_i = g_i \cdot \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \epsilon}} $$

- 分母是该向量的**均方根**（root mean square），$\epsilon$ 防止除零；
- $g_i$ 是**可学习的逐通道增益**（就是 `weight`，形状 `(d_model,)`），归一化后再逐通道放大/缩小；
- 只在**特征维**上归一化，与 batch/序列位置无关——所以支持任意前置批量维。

与 LayerNorm 的区别：LayerNorm 还要**减均值**、通常带 **bias**；RMSNorm 把这两样都去掉，更省、实践中效果相当甚至更好（详见 prenorm 笔记）。

---

## 2. 逐个决定的理由

### 2.1 接口与形状

契约（来自 `run_rmsnorm` / `test_rmsnorm`）：

- `RMSNorm(d_model, eps=1e-5, device=None, dtype=None)`；
- `weight` 形状 `(d_model,)`，即逐通道增益 $g$；
- `forward(x)`：输入 `(..., d_model)` → 输出同形状。

测试取的参考权重是 `layers.1.ln1.weight`——正是某个 RMSNorm 块的 $g$，形状 `(d_model,)`，印证了"每个 norm 块各有一份独立 $g$"。

### 2.2 weight 初始化为全 1

`self.weight = nn.Parameter(torch.ones(d_model))`。初始为 **1** 而非随机，是因为 RMSNorm 的 $g$ 是"在归一化之上做微调"的增益——初始设 1 表示"先不缩放、保持归一化后的尺度"，让训练从一个中性起点开始。这也是 PyTorch/各框架 norm 层的通行默认。（本测试会用外部参考权重覆盖它，故初始化不影响测试结果。）

### 2.3 为什么在 float32 上算（关键数值细节）

```python
in_dtype = x.dtype
x = x.to(torch.float32)
rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
y = x / rms * self.weight
return y.to(in_dtype)
```

核心是 `x.pow(2)`——**平方会放大数值范围**。若输入是 fp16/bf16，平方求和很容易**溢出或损失精度**（bf16 尾数只有 7 位，`x^2` 累加误差明显）。所以标准做法是：**先升到 float32 做平方、求均值、开方、缩放，最后再转回原 dtype**。这样既数值稳定，又不改变对外的精度契约。

`test_rmsnorm` 的输入是 float32，这一步在这里不改变数值；但对混合精度训练（bf16 激活）这是**必要**的正确性保证——现代 LLM 的 RMSNorm 都这么写。

### 2.4 `mean(dim=-1, keepdim=True)` 的两个细节

- `dim=-1`：沿**最后一维**（特征维）求均方——每个 token 各自归一化，互不影响；
- `keepdim=True`：保留被归约的维度为 1（形状从 `(..., d)` 变 `(..., 1)`），这样 `x / rms` 能正确**广播**回 `(..., d)`。若不 keepdim 会形状不匹配。

最后 `* self.weight`：`weight` 形状 `(d,)` 自动广播到最后一维，实现逐通道增益。

### 2.5 是否手写 backward

不需要。`pow`、`mean`、`sqrt`、除法、乘法都是可微算子，autograd 自动求导（同 [linear 笔记 2.6](./linear_implementation_notes.md)）。梯度会同时回流到输入 `x` 和增益 `weight`。

---

## 3. adapter 怎么接到测试

`run_rmsnorm(d_model, eps, weights, in_features)` 三步（同一套路）：

1. 构造 `RMSNorm(d_model, eps=eps, device=weights.device, dtype=weights.dtype)`；
2. `load_state_dict({"weight": weights})` 装入参考增益 $g$；
3. 返回 `rmsnorm(in_features)`。

`test_rmsnorm` 用 `in_embeddings`（float32 随机张量）作输入，`layers.1.ln1.weight` 作参考权重，输出与 `_snapshots/test_rmsnorm.npz` 比对（`atol=1e-4`）。

---

## 4. 为什么测试能过

权重来自外部、公式是标准 RMSNorm、且在 float32 上计算，输出与参考在容差内一致。链路：

```text
test_rmsnorm → run_rmsnorm(装入参考 g) → RMSNorm.forward（float32 均方根缩放）
            → numpy_snapshot 对照 _snapshots/test_rmsnorm.npz → PASS
```

运行：

```sh
uv run pytest -k test_rmsnorm
```

预期 `1 passed`。

---

## 5. 小结

1. **算什么**：$y_i = g_i \cdot x_i / \sqrt{\text{mean}(x^2)+\epsilon}$，只按均方根缩放、无均值中心化、无 bias。
2. **weight = 逐通道增益 $g$**，形状 `(d_model,)`，初始化为全 1，每个 norm 块独立一份。
3. **float32 计算**：`x^2` 会放大范围，低精度易溢出/掉精度，故升 float32 算完再转回——混合精度下的必要正确性保证。
4. **形状细节**：`mean(dim=-1, keepdim=True)` 保证只归一化特征维且能正确广播。
5. **backward 自动**：全用可微算子，autograd 处理。

---

## 参考

- 本仓库实现：[cs336_basics/model.py](../cs336_basics/model.py)
- 适配层：[tests/adapters.py](../tests/adapters.py)
- 测试：[tests/test_model.py](../tests/test_model.py)
- 原理背景：[prenorm_vs_postnorm_explained.md](./prenorm_vs_postnorm_explained.md)（RMSNorm 与 LayerNorm、$g_i$ 独立性）
- 配套：[linear_implementation_notes.md](./linear_implementation_notes.md)、[embedding_implementation_notes.md](./embedding_implementation_notes.md)
