# 实现 Transformer Block（Pre-Norm）：原理、组装与验证

## 0. 本文目标

记录 CS336 assignment1 里 **Transformer block** 的实现思路：它如何把前面实现的 RMSNorm、MHA、SwiGLU、RoPE 组装成一个完整的层，为什么用 **Pre-Norm** 和残差连接、adapter 怎么接到测试。这是"把零件拼成整机"的一步。

对应实际代码：
- 实现：[cs336_basics/model.py](../cs336_basics/model.py) 的 `TransformerBlock`（及本轮补上的 `run_multihead_self_attention_with_rope`）
- 接线：[tests/adapters.py](../tests/adapters.py) 的 `run_transformer_block`
- 测试：[tests/test_model.py](../tests/test_model.py) 的 `test_transformer_block`

原理背景：Pre-Norm vs Post-Norm 已在 [02_07_prenorm_vs_postnorm_explained.md](./02_07_prenorm_vs_postnorm_explained.md) 详述并配图，本文聚焦组装。

---

## 1. 背景：一个 block 由什么组成

现代 decoder-only Transformer 的每个 block 有**两个子层**，各自包"归一化 + 残差"：

1. **注意力子层**：多头自注意力（[MHA](./02_12_multihead_attention_implementation_notes.md)，含 [RoPE](./02_09_rope_implementation_notes.md)）；
2. **前馈子层**：[SwiGLU](./02_06_swiglu_implementation_notes.md) FFN。

把 $L$ 个这样的 block 叠起来，加上词嵌入和末端输出，就是完整的语言模型。所以 block 是"承上启下"的组装单元——它不引入新算子，只是把已实现的组件按正确结构接起来。

---

## 2. 逐个决定的理由

### 2.1 Pre-Norm 结构与残差

本 block 用 **Pre-Norm**：归一化放在**子层之前**，残差**绕过**归一化直连。公式：

$$ y = x + \mathrm{MHA}(\mathrm{RMSNorm}_1(x)) $$

$$ \mathrm{out} = y + \mathrm{FFN}(\mathrm{RMSNorm}_2(y)) $$

对应代码：

```python
x = x + self.attn(self.ln1(x), token_positions=..., rope=self.rope)
x = x + self.ffn(self.ln2(x))
```

- **为什么 Pre-Norm**：残差路径上没有归一化，梯度可以从顶层几乎无损地直达底层（残差是恒等映射），训练更稳、能堆更深、对学习率和 warmup 更鲁棒。这正是现代 LLM（LLaMA、GPT-NeoX 等）的标准选择，详细梯度分析见 [prenorm 笔记](./02_07_prenorm_vs_postnorm_explained.md)。
- **两个独立的 RMSNorm**：`ln1`、`ln2` 各有自己的可学习增益 $g$（呼应 prenorm 笔记里"每块独立"的结论），不能共用。

### 2.2 组件复用：block 只负责"接线"

`__init__` 里建五个子模块，全是前面实现好的：

```python
self.ln1  = RMSNorm(d_model)
self.attn = MultiHeadSelfAttention(d_model, num_heads)
self.ln2  = RMSNorm(d_model)
self.ffn  = SwiGLU(d_model, d_ff)
self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
```

- 它们都是子 `Module`，参数自动登记进 block（`nn.Module` 树）；
- block 自身**不引入任何新参数或新算子**，`forward` 只是按 Pre-Norm 结构把它们串起来——这是"组装层"的典型形态。

### 2.3 RoPE 从哪来、怎么传

契约要求 block **使用 RoPE**，且 RoPE 的维度是**头维度** `d_model // num_heads`（不是 d_model）。做法：block 自己持有一个 `RotaryPositionalEmbedding`，在调用注意力时把它和 `token_positions` 一起传进去：

```python
self.attn(self.ln1(x), token_positions=token_positions, rope=self.rope)
```

MHA 的 `forward` 里预留了 `rope` 钩子——拆头后对每个头的 Q/K 施加旋转（见 [MHA 笔记](./02_12_multihead_attention_implementation_notes.md) 与 [RoPE 笔记](./02_09_rope_implementation_notes.md)）。`token_positions` 缺省时用 `arange(seq)`。

### 2.4 权重键与 `load_state_dict` 直接匹配

这是本实现的一个便利点。block 的子模块命名刻意与官方 `weights` 字典的键**完全对齐**：

| weights 键 | block 里的子模块 |
|---|---|
| `attn.{q,k,v,output}_proj.weight` | `self.attn` 的四个投影 |
| `ln1.weight` / `ln2.weight` | `self.ln1` / `self.ln2` |
| `ffn.{w1,w2,w3}.weight` | `self.ffn` 的三个 Linear |

所以 adapter 里一句 `block.load_state_dict(weights)` 就能全部装入，无需手动改名。RoPE 的 `cos/sin` 是 `persistent=False` 的 buffer，不进 `state_dict`，因此不会与这些键冲突（`strict=True` 也能通过）。

### 2.5 是否手写 backward

不需要。block 只是把可微子模块按加法（残差）和函数调用组合起来，autograd 自动求导。

---

## 3. adapter 怎么接到测试

`run_transformer_block` 三步：

1. 构造 `TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta, ...)`；
2. `block.load_state_dict(weights)` 直接装入官方权重；
3. 返回 `block(in_features)`。

`test_transformer_block` 从 `ts_state_dict` 取 `layers.0.` 前缀的权重（去掉前缀后正好是上表的键）、`in_embeddings` 作输入，输出与快照比对（`atol` 由 numpy_snapshot 默认）。

> 本轮还顺带补上了之前跳过的 `run_multihead_self_attention_with_rope`（`test_multihead_self_attention_with_rope` 现已 PASSED）——因为 block 依赖"带 RoPE 的 MHA"，两者一起实现更顺。

---

## 4. 为什么测试能过

Pre-Norm 结构、两处残差、RoPE 施加、权重命名都与参考一致。链路：

```text
test_transformer_block → run_transformer_block（load_state_dict 装权重）
  → TransformerBlock.forward:
      x = x + MHA(RMSNorm1(x), rope)     # 注意力子层
      x = x + FFN(RMSNorm2(x))           # 前馈子层
  → numpy_snapshot 对照 _snapshots → PASS
```

运行：

```sh
uv run pytest -k test_transformer_block
```

预期 `1 passed`。

---

## 5. 小结

1. **两个子层**：注意力（MHA+RoPE）+ 前馈（SwiGLU），各包"归一化 + 残差"。
2. **Pre-Norm**：`x + Sublayer(RMSNorm(x))`，归一化在子层前、残差绕过它 → 训练稳、可堆深（详见 prenorm 笔记）。
3. **纯组装**：复用 RMSNorm/MHA/SwiGLU/RoPE 五个子模块，block 不引入新参数/新算子。
4. **两个独立 RMSNorm**：`ln1`/`ln2` 各有自己的增益 $g$。
5. **命名对齐**：子模块键与官方 `weights` 一致，`load_state_dict(weights)` 一步装入；RoPE buffer 不进 state_dict 故不冲突。

---

## 参考

- 本仓库实现：[cs336_basics/model.py](../cs336_basics/model.py)
- 适配层：[tests/adapters.py](../tests/adapters.py)
- 测试：[tests/test_model.py](../tests/test_model.py)
- 原理背景：[02_07_prenorm_vs_postnorm_explained.md](./02_07_prenorm_vs_postnorm_explained.md)（Pre-Norm 为何流行）
- 组件笔记：[02_12_multihead_attention_implementation_notes.md](./02_12_multihead_attention_implementation_notes.md)、[02_06_swiglu_implementation_notes.md](./02_06_swiglu_implementation_notes.md)、[02_04_rmsnorm_implementation_notes.md](./02_04_rmsnorm_implementation_notes.md)、[02_09_rope_implementation_notes.md](./02_09_rope_implementation_notes.md)
- 原始文献：Vaswani et al., *Attention Is All You Need*, 2017。
