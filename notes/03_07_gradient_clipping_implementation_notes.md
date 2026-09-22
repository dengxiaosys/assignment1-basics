# 实现梯度裁剪（Gradient Clipping）：原理与验证

## 0. 本文目标

记录 CS336 assignment1 里**梯度裁剪** `gradient_clipping` 的实现：它解决什么问题、为什么按"全局范数"裁剪、公式与实现细节、adapter 怎么接到测试。

对应实际代码：
- 实现：[cs336_basics/optimizer.py](../cs336_basics/optimizer.py) 的 `gradient_clipping`
- 接线：[tests/adapters.py](../tests/adapters.py) 的 `run_gradient_clipping`
- 测试：[tests/test_nn_utils.py](../tests/test_nn_utils.py) 的 `test_gradient_clipping`

前置：[学习率调参笔记](./03_03_learning_rate_tuning_notes.md)（梯度过大导致发散）、[优化器 API 笔记](./03_02_pytorch_optimizer_api_notes.md)（训练四拍：backward 后、step 前做裁剪）。

---

## 1. 背景：为什么要裁剪梯度

训练中偶尔会出现**梯度爆炸**——某个 batch 的梯度突然特别大（数据异常、数值不稳、深层网络累积等），一步大更新就可能把参数推到坏区域，导致 loss 变 NaN 或发散（见 [lr 笔记](./03_03_learning_rate_tuning_notes.md)）。

**梯度裁剪**是一道"保险丝"：**限制梯度的整体幅度不超过一个阈值**，异常大的梯度被按比例缩回安全范围，正常的梯度不受影响。它让训练对偶发的大梯度更鲁棒，是 LLM 训练的标准操作（常配合 AdamW 使用）。

**在训练循环里的位置**：`backward`（算出梯度）之后、`optimizer.step()`（用梯度更新）之前——先把梯度裁好，再让优化器迈步。

---

## 2. 原理：按"全局 L2 范数"裁剪

关键设计：不是逐个参数各自裁，而是把**所有参数的梯度看成一个大向量**，按它的总 L2 范数裁剪。

设所有参数梯度为 $g_1,\dots,g_n$，**全局范数**：

$$ \|g\| = \sqrt{\sum_i \|g_i\|_2^2} $$

裁剪规则：给定阈值 $M$（`max_l2_norm`），

- 若 $\|g\| \le M$：**不动**，梯度保持原样；
- 若 $\|g\| > M$：所有梯度**统一乘同一个缩放因子** $\dfrac{M}{\|g\|+\epsilon}$，使裁剪后总范数约为 $M$。

$$ g_i \leftarrow g_i \cdot \frac{M}{\|g\|+\epsilon} \quad(\text{当 }\|g\|>M) $$

$\epsilon$（取 $10^{-6}$）是数值稳定项，避免除零。

**为什么用全局范数而非逐参数裁**：逐参数裁会改变各参数梯度之间的**相对比例**，等于扭曲了梯度的"方向"；全局按同一因子缩放则**只改整体步长、不改方向**——仍朝原来的下坡方向走，只是步子收小。这保留了梯度携带的方向信息，是 PyTorch `clip_grad_norm_` 的做法。

---

## 3. 实现与关键细节

```python
grads = [p.grad for p in parameters if p.grad is not None]   # 跳过无梯度参数
if not grads:
    return
total_norm = torch.sqrt(sum((g ** 2).sum() for g in grads))  # 全局 L2 范数
if total_norm > max_l2_norm:
    scale = max_l2_norm / (total_norm + eps)
    for g in grads:
        g.mul_(scale)                                        # 就地缩放
```

- **跳过 `p.grad is None`**：有些参数被冻结（`requires_grad_(False)`）或没参与前向，没有梯度，必须跳过——测试专门冻结了一个参数验证这点；
- **全局范数**：把每个梯度的平方和加总再开方，是"所有梯度拼成一个向量"的 L2 范数；
- **就地修改 `g.mul_(scale)`**：契约要求原地改 `parameter.grad`（优化器随后就用这些 grad），不能返回新张量；
- **只在超过阈值时缩放**：没超过就不动，保证正常梯度无损。

对齐 PyTorch：本实现与 `torch.nn.utils.clip_grad_norm_` 行为一致，测试正是拿两者的裁剪结果逐元素比对（`atol=1e-5`）。

### 3.1 时间/空间复杂度与性能分析

设模型总参数量为 $P$（梯度元素数与之相同）。

**时间复杂度 $O(P)$**：算全局范数要把每个梯度元素平方并求和（遍历一遍 $P$ 个元素），缩放又遍历一遍——都是逐元素线性操作，共 $O(P)$。**没有矩阵乘**，只有逐元素乘加与规约。

**空间复杂度 $O(1)$ 额外开销**：全部就地进行——范数是标量累加、缩放用 `g.mul_()` 原地改，不额外分配与 $P$ 同量级的张量。梯度本身占的 $O(P)$ 显存是训练本就有的（见 [AdamW 核算笔记](./03_05_adamw_accounting_notes.md) 的显存四块），裁剪不额外增加。

**会是性能瓶颈吗？一般不会。** 对照一个训练步的成本（见 [AdamW 核算笔记](./03_05_adamw_accounting_notes.md)）：前向+反向的矩阵乘是 $O(B\cdot s\cdot P)$ 量级（每个 token 都要过全部参数），高出裁剪的 $O(P)$ **好几个数量级**（差了 $B\cdot s$ 倍，即 batch×序列长度）。所以裁剪相对 matmul 几乎免费，和 AdamW 的逐元素更新是同一量级、同样可忽略。

**什么情况下可能变得不可忽略**：

1. **分布式训练的通信**：多卡时全局范数需要跨卡 **all-reduce**（每张卡先算本地平方和，再规约出全局范数）。此时瓶颈不是计算，而是这一次**通信/同步**——它引入一个 barrier，卡越多越明显。这是大规模训练里梯度裁剪真正的开销来源。
2. **访存带宽受限**：$O(P)$ 逐元素操作是**访存密集型**（算术强度低，每读一个元素只做几次运算）。当模型极大、且这步没和别的操作融合时，它受显存带宽而非算力限制；不过相对整步仍是小头。
3. **实现不当**：若用 Python 循环逐参数、或频繁做 GPU→CPU 同步（如反复 `.item()` 取范数）、或每步无谓地新建张量，可能引入额外开销——本实现用张量化的平方和 + 就地缩放规避了这些。

**一句话**：单卡下梯度裁剪是 $O(P)$ 时间、$O(1)$ 额外空间，相对矩阵乘可忽略，**基本不是瓶颈**；只有在**多卡分布式**里，它的全局范数 all-reduce 通信才可能成为需要关注的同步开销。

---

## 4. adapter 与测试

`run_gradient_clipping(parameters, max_l2_norm)` 转发到 `gradient_clipping`。

`test_gradient_clipping` 造 6 个随机张量、**冻结其中一个**，分别用 PyTorch 的 `clip_grad_norm_` 和我们的实现裁剪，比对两者对每个（有梯度的）参数的裁剪结果一致。

运行：

```sh
uv run pytest -k test_gradient_clipping
```

预期 `1 passed`。

---

## 5. 小结

1. **动机**：防梯度爆炸，一步过大更新导致发散/NaN；是训练的"保险丝"。
2. **全局范数裁剪**：按所有梯度拼成的总 L2 范数 $\|g\|$ 裁——超过阈值 $M$ 才整体乘 $\frac{M}{\|g\|+\epsilon}$。
3. **只缩放不改方向**：全局同因子缩放保留梯度方向，优于逐参数裁。
4. **细节**：跳过 `grad is None`（含冻结参数）、就地 `mul_`、$\epsilon$ 防除零。
5. **位置**：训练循环里 `backward` 之后、`step` 之前。

---

## 参考

- 本仓库实现：[cs336_basics/optimizer.py](../cs336_basics/optimizer.py)
- 适配层：[tests/adapters.py](../tests/adapters.py)
- 测试：[tests/test_nn_utils.py](../tests/test_nn_utils.py)
- 前置：[03_03_learning_rate_tuning_notes.md](./03_03_learning_rate_tuning_notes.md)、[03_02_pytorch_optimizer_api_notes.md](./03_02_pytorch_optimizer_api_notes.md)
- PyTorch：`torch.nn.utils.clip_grad_norm_`
