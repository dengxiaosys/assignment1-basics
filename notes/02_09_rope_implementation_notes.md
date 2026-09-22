# 实现 RoPE（旋转位置编码）：思路、缓存与验证

## 0. 本文目标

记录 CS336 assignment1 里 **RoPE**（`RotaryPositionalEmbedding`）的实现思路：频率怎么算、cos/sin 为什么预缓存成 buffer、"相邻配对"怎么旋转、adapter 怎么接到测试。

对应实际代码：
- 实现：[cs336_basics/model.py](../cs336_basics/model.py) 的 `RotaryPositionalEmbedding`
- 接线：[tests/adapters.py](../tests/adapters.py) 的 `run_rope`
- 测试：[tests/test_model.py](../tests/test_model.py) 的 `test_rope`

RoPE 的**原理与背景**（为什么旋转能编码相对位置、在 LLM 中的位置、多频率、长上下文扩展、与其它位置编码对比）已在 [02_08_rope_explained.md](./02_08_rope_explained.md) 详述并配图，本文聚焦实现。

---

## 1. 原理速览（够用即可）

RoPE 把向量按**相邻二维对**分组，每对在自己的平面里旋转一个角度 = **位置 × 该对频率**。设位置 $m$、第 $i$ 对的频率 $\theta_i$，对 $(x_{2i}, x_{2i+1})$ 旋转：

$$ \begin{bmatrix} x'_{2i} \\ x'_{2i+1} \end{bmatrix} = \begin{bmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{bmatrix} \begin{bmatrix} x_{2i} \\ x_{2i+1} \end{bmatrix} $$

频率按维度递减：

$$ \theta_i = \text{base}^{-2i/d}, \qquad i = 0,1,\dots,\tfrac{d}{2}-1 $$

其中 base 就是参数 `theta`（测试用 10000）。旋转后 $\langle R_m q, R_n k\rangle$ 只依赖 $m-n$——这就是"相对位置"的来源（推导见 rope 笔记）。

---

## 2. 逐个决定的理由

### 2.1 接口与契约

来自 `run_rope` / `test_rope`：

- `RotaryPositionalEmbedding(theta, d_k, max_seq_len, device=None)`；
- `forward(x, token_positions)`：`x` 形状 `(..., seq, d_k)`，`token_positions` 形状 `(..., seq)`，输出同 `x` 形状；
- 只作用于 Q/K（本测试直接喂 `in_embeddings` 当作 Q/K 的输入），**不含可学习参数**。

### 2.2 频率与角度表怎么算

```python
inv_freq = theta ** (-torch.arange(0, d_k, 2).float() / d_k)   # (d_k/2,)
pos = torch.arange(max_seq_len).float()                        # (max_seq_len,)
angles = torch.outer(pos, inv_freq)                            # (max_seq_len, d_k/2)
```

- `arange(0, d_k, 2)` 取 $0,2,4,\dots$，除以 `d_k` 即 $2i/d$，`theta ** (-...)` 得每对频率 $\theta_i$；
- `outer(pos, inv_freq)` 是**外积**：第 `(m, i)` 个元素就是 $m\cdot\theta_i$，即"位置 m、第 i 对"的旋转角。一次性把所有位置 × 所有对的角度算成一张表。

### 2.3 为什么把 cos/sin 预缓存成 buffer

```python
self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
self.register_buffer("sin_cached", torch.sin(angles), persistent=False)
```

- **预缓存**：角度只依赖位置和频率，与输入内容无关，所以可以在 `__init__` 里一次算好 cos/sin，forward 时按位置**查表**即可，省去每次重算三角函数；
- **用 `register_buffer` 而非 `nn.Parameter`**：cos/sin 是**不可学习**的常量——RoPE 没有可训练参数。buffer 会随模型 `.to(device)`、进 `state_dict`，但**不被优化器更新**（见 [nn_module 笔记](./02_01_nn_module_and_linear_explained.md) 的登记规则）；
- **`persistent=False`**：这些缓存可由 `theta/d_k/max_seq_len` 完全重建，不必写进 `state_dict`，避免和外部权重加载冲突、也减小 checkpoint。

> 呼应 [prenorm 笔记](./02_07_prenorm_vs_postnorm_explained.md) 的一个对照：RMSNorm 的增益 $g$ 是**可学习、每块独立**的 `Parameter`；RoPE 的频率/角度是**不可学习、全局共享**的 `buffer`。可学习性与共享范围恰好相反。

### 2.4 `forward`：按位置查表 + 相邻配对旋转

```python
cos = self.cos_cached[token_positions]   # (..., seq, d_k/2)
sin = self.sin_cached[token_positions]
x_even = x[..., 0::2]                     # 偶数位分量 x_0, x_2, ...
x_odd  = x[..., 1::2]                     # 奇数位分量 x_1, x_3, ...
out_even = x_even * cos - x_odd * sin
out_odd  = x_even * sin + x_odd * cos
out = torch.empty_like(x)
out[..., 0::2] = out_even
out[..., 1::2] = out_odd
```

- **`cos_cached[token_positions]`**：用整型位置张量做高级索引，按每个 token 的位置取出对应角度行（同 Embedding 的按行查表思路）；
- **`0::2` / `1::2`**：CS336 用**相邻配对**——把维度切成 $(x_0,x_1),(x_2,x_3),\dots$，偶数索引是每对的第一分量、奇数索引是第二分量；
- 两行 `out_even/out_odd` 正是 2.1 节旋转矩阵的逐元素展开；
- **交错拼回**：旋转后再按 `0::2`/`1::2` 写回，恢复原始维度顺序。

> 注意配对约定要与推理端一致：有的实现用"折半配对"（前半 $x_{0..d/2}$ 配后半），角度排布不同。CS336 快照是按相邻配对生成的，所以这里必须用相邻配对，否则数值对不上。

### 2.5 是否手写 backward / 有无参数

无可学习参数（只有 buffer），也不需要手写 backward——`cos/sin` 是常量，`乘/加/索引` 都是可微算子，梯度会正常穿过 RoPE 回到输入 Q/K。

### 2.6 三个常见疑问：可学习性、跨层 theta、base 怎么定

这三问围绕"RoPE 到底有没有参数、参数怎么定"，一起讲清楚。

**Q1：它没有任何可学习参数吗？**

对，完全没有。RoPE 里只有 `cos_cached`、`sin_cached` 两个 buffer，由 `theta`、`d_k`、`max_seq_len` **确定性**算出，不是 `nn.Parameter`，不进优化器、不被梯度更新（见 2.3、2.5）。`model.parameters()` 里数不到 RoPE 的任何一项——它是**纯函数式**的位置变换。对照：Embedding/Linear 的 weight、RMSNorm 的增益 $g$ 都是可学习 `Parameter`，而 RoPE 是"零参数"层。

**Q2：不同层的 theta 通常会不同吗？**

通常**不会**——绝大多数主流模型（LLaMA、GPT-NeoX、Qwen 等）**所有层共享同一个 theta（base）**。RoPE 虽然在每一层都重新施加旋转（见 [rope 笔记 4.2](./02_08_rope_explained.md)），但每层用的是**同一套频率公式、同一个 base**，即"每层都转，但转法一样"。

- **为什么不用层间不同**：位置的物理含义（第几个 token、相距多远）不随层变化，用同一套频率最自然、也最省参数与调参成本；
- **长上下文扩展**（PI / NTK / YaRN）会调整频率，但通常是**对所有层做同一种调整**，而不是让层与层之间用不同 base；
- 少数研究探索过 per-layer / per-head 的频率分配，但**不是主流**，标准实现就是全模型一个 theta。

**Q3：theta（base）怎么定？**

它是一个**人为设定的超参数**，不是学出来的：

- **默认 10000**：沿袭原始 Transformer 正弦位置编码里的 base，RoFormer 直接继承，成了事实默认值（本测试也用 10000）；
- **它控制什么**：base 决定各二维对旋转"波长"的跨度。由 $\theta_i=\text{base}^{-2i/d}$，$i=0$ 的对转最快（在位置上的周期约 $2\pi$），$i=d/2-1$ 的对转最慢（周期约 $2\pi\cdot\text{base}$）。base 越大，最慢的对波长越长，越能区分很远的位置；
- **长上下文常把 base 调大**：为支持更长序列，很多模型把 base 从 10000 提高，例如 CodeLlama、Qwen 用 1,000,000，Llama 3 用 500,000。base 变大 → 旋转更慢 → 高位对在长距离上仍不"转满一圈"、不易混淆，利于长程外推；
- **要点**：base 是设计者拍定、再按长度需求经验调整的超参数，全模型一致；改它属于"架构/推理配置"，而非训练中被优化的参数。（具体数值以各模型官方配置为准。）

---

## 3. adapter 怎么接到测试

`run_rope(d_k, theta, max_seq_len, in_query_or_key, token_positions)` 两步：

1. 构造 `RotaryPositionalEmbedding(theta, d_k, max_seq_len, device=in_query_or_key.device)`；
2. 返回 `rope(in_query_or_key, token_positions)`。

不需要 `load_state_dict`——RoPE 没有可学习权重，cos/sin 由构造参数完全决定。

`test_rope` 用 `in_embeddings`（`(batch, seq, d_model)`）当输入、`pos_ids = arange(n_queries)` 当位置、`theta=10000`，输出与 `_snapshots/test_rope.npz` 比对（`atol=1e-5`）。

---

## 4. 为什么测试能过

频率公式、相邻配对、旋转展开都与参考实现一致，且无外部权重需加载。链路：

```text
test_rope → run_rope → RotaryPositionalEmbedding.forward（查 cos/sin 表 + 逐对旋转）
         → numpy_snapshot 对照 _snapshots/test_rope.npz → PASS
```

运行：

```sh
uv run pytest -k test_rope
```

预期 `1 passed`。

---

## 5. 小结

1. **频率**：$\theta_i=\text{base}^{-2i/d}$，`outer(pos, inv_freq)` 得角度表。
2. **缓存**：cos/sin 预算好、`register_buffer(persistent=False)`——不可学习、可重建、不进 checkpoint。
3. **相邻配对旋转**：`0::2`/`1::2` 取偶奇分量，按旋转矩阵展开，再交错写回。
4. **无参数、backward 自动**：只有 buffer，梯度正常穿过。
5. **配对约定要一致**：相邻 vs 折半会改变数值，必须与参考/推理端对齐。

---

## 参考

- 本仓库实现：[cs336_basics/model.py](../cs336_basics/model.py)
- 适配层：[tests/adapters.py](../tests/adapters.py)
- 测试：[tests/test_model.py](../tests/test_model.py)
- 原理背景：[02_08_rope_explained.md](./02_08_rope_explained.md)（旋转编码相对位置、多频率、长上下文扩展、与其它位置编码对比）
- 配套：[02_01_nn_module_and_linear_explained.md](./02_01_nn_module_and_linear_explained.md)、[02_04_rmsnorm_implementation_notes.md](./02_04_rmsnorm_implementation_notes.md)
