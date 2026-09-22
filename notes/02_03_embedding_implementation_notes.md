# 实现 Embedding（词嵌入查表层）：思路、接线与验证

## 0. 本文目标

记录 CS336 assignment1 里 `Embedding` 层的实现思路——讲清楚**每个决定为什么这么做**：它本质是什么、weight 形状怎么定、`forward` 为何只是"按行索引"、adapter 怎么接到测试。结构与配套的 [02_02_linear_implementation_notes.md](./02_02_linear_implementation_notes.md) 保持一致。

对应实际代码：
- 实现：[cs336_basics/model.py](../cs336_basics/model.py) 的 `Embedding`
- 接线：[tests/adapters.py](../tests/adapters.py) 的 `run_embedding`
- 测试：[tests/test_model.py](../tests/test_model.py) 的 `test_embedding`

背景见 [02_01_nn_module_and_linear_explained.md](./02_01_nn_module_and_linear_explained.md)、[00_03_pytest_fixtures_explained.md](./00_03_pytest_fixtures_explained.md)。

---

## 1. Embedding 本质：一张"可学习的查找表"

语言模型的输入是**离散的 token id**（整数），但网络要吃**连续向量**。Embedding 就是这层转换：给每个 token id 分配一个 `d_model` 维向量，训练中这些向量被学习。

它其实就是一张矩阵 $E \in \mathbb{R}^{V \times d}$（$V$=词表大小，$d$=向量维度）：**第 $i$ 行就是 id 为 $i$ 的 token 的向量**。所谓"embedding lookup"，就是**按行取**：给定 id，返回对应行。

$$ \text{Embedding}(i) = E[i, :] $$

对一批 ids，就按各自的行取出、堆成对应形状。

> 和 Linear 的关系：Linear 是"矩阵乘"，Embedding 是"按行索引"。其实 embedding lookup 等价于"用 one-hot 向量乘以 $E$"——但直接索引远比构造 one-hot 再矩阵乘高效，所以实现上就用索引。

---

## 2. 逐个决定的理由

### 2.1 接口与形状

契约（来自 `run_embedding` / `test_embedding`）：

- `Embedding(num_embeddings, embedding_dim, device=None, dtype=None)`——即 `(vocab_size, d_model)`；
- `weight` 形状 `(num_embeddings, embedding_dim)`，**第 i 行 = id i 的向量**；
- `forward(token_ids)`：`token_ids` 是**任意形状的整型张量**，输出形状是 `token_ids.shape + (embedding_dim,)`。

形状约定与官方 `nn.Embedding`、以及 `state_dict` 里的 `token_embeddings.weight` 一致，便于加载/比对参考权重。

### 2.2 为什么 weight 用 `nn.Parameter`

和 Linear 同理：只有 `nn.Parameter` 才会被 `nn.Module` 登记为可学习参数，进入 `parameters()`/`state_dict()`、被优化器更新、被 `load_state_dict` 加载。Embedding 表是**要训练**的（词向量在学习中不断更新），所以必须是 Parameter。

### 2.3 `forward` 为什么就是 `self.weight[token_ids]`

PyTorch 的**高级索引**：用一个整型张量去索引矩阵的第 0 维，会**按元素取对应行、并保持索引张量的形状**。所以：

- `token_ids` 形状 `(batch, seq)` → `self.weight[token_ids]` 形状 `(batch, seq, d_model)`；
- `token_ids` 是标量/一维/任意维都成立，输出自动是 `token_ids.shape + (d_model,)`。

这一步天然可微：梯度只会回流到**被取用的那些行**（用到的 id 才有梯度，没用到的行梯度为 0），这正是 embedding 训练的预期行为。也因此不需要手写 backward——autograd 对索引操作有支持（见 [linear 笔记 2.6](./02_02_linear_implementation_notes.md)）。

### 2.4 初始化（只影响独立使用，不影响本测试）

词向量常用**标准正态**初始化（本实现用 `trunc_normal_`、`std=1.0`、截断 ±3）。与 Linear 不同，这里没有"扇入/扇出"意义上的 Xavier 方差——因为 embedding 不是矩阵乘、没有 fan_in 的求和放大问题，每个向量是**独立取用**的，用标准正态即可。`test_embedding` 会用外部参考权重覆盖它，故初始化不影响该测试结果。

---

## 3. adapter 怎么接到测试

`run_embedding(vocab_size, d_model, weights, token_ids)` 三步（与 `run_linear` 同一套路）：

1. 构造 `Embedding(vocab_size, d_model, device=weights.device, dtype=weights.dtype)`；
2. `load_state_dict({"weight": weights})` 把参考权重装进去（键 `"weight"` 必须与 `self.weight` 同名，`strict=True` 会校验形状/键名）；
3. 返回 `embedding(token_ids)`（用 `embedding(x)` 触发 `__call__`）。

`test_embedding` 用 `in_indices`（一批随机 token id，来自 conftest 的 `torch.randint(0, 10_000, (batch, seq))`）作为输入，用官方 `token_embeddings.weight` 作参考权重，输出与 `_snapshots/test_embedding.npz` 逐元素比对。

---

## 4. 为什么测试能过

权重来自外部、`forward` 就是标准的按行索引，输出自然与快照一致。链路：

```text
test_embedding → run_embedding(装入参考 weights) → Embedding.forward = weight[token_ids]
              → numpy_snapshot 对照 _snapshots/test_embedding.npz → PASS
```

运行：

```sh
uv run pytest -k test_embedding
```

预期 `1 passed`。

---

## 5. 与 Linear 的对照小结

| | Linear | Embedding |
|---|---|---|
| 本质 | 矩阵乘 $y = xW^\top$ | 按行查表 $E[\text{id}]$ |
| weight 形状 | `(d_out, d_in)` | `(vocab_size, d_model)` |
| 输入 | 连续向量 `(..., d_in)` | 整型 id `(...)` |
| 输出 | `(..., d_out)` | `(..., d_model)` |
| forward | `x @ weight.transpose(-2,-1)` | `weight[token_ids]` |
| 初始化 | Xavier（含 fan_in/out） | 标准正态 |
| 梯度 | 全 weight | 只回流到用到的行 |
| 是否手写 backward | 否（autograd） | 否（autograd） |

一句话：**两者都是"一张可学习的权重表 + 一个 forward"，区别只在 forward 是"矩阵乘"还是"按行取"。**

---

## 参考

- 本仓库实现：[cs336_basics/model.py](../cs336_basics/model.py)
- 适配层：[tests/adapters.py](../tests/adapters.py)
- 测试：[tests/test_model.py](../tests/test_model.py)
- 配套：[02_02_linear_implementation_notes.md](./02_02_linear_implementation_notes.md)、[02_01_nn_module_and_linear_explained.md](./02_01_nn_module_and_linear_explained.md)
