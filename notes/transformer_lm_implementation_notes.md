# 实现 Transformer 语言模型（TransformerLM）：原理、组装与验证

## 0. 本文目标

记录 CS336 assignment1 里**完整语言模型** `TransformerLM` 的实现思路：它如何把词嵌入、$N$ 个 Transformer block、末端归一化和输出头组装成一个 decoder-only 语言模型、输出的 logits 是什么、adapter 怎么接到测试。这是整个 assignment1 建模部分的**收官**——从此前所有零件到一个能出词表分布的模型。

对应实际代码：
- 实现：[cs336_basics/nn.py](../cs336_basics/nn.py) 的 `TransformerLM`
- 接线：[tests/adapters.py](../tests/adapters.py) 的 `run_transformer_lm`
- 测试：[tests/test_model.py](../tests/test_model.py) 的 `test_transformer_lm`、`test_transformer_lm_truncated_input`

组件背景见：[transformer_block 笔记](./transformer_block_implementation_notes.md)（block 本身）、[embedding 笔记](./embedding_implementation_notes.md)、[linear 笔记](./linear_implementation_notes.md)。

---

## 1. 背景：一个语言模型的整体结构

decoder-only Transformer 语言模型做的事：给定一串 token id，为每个位置预测"下一个 token"的分布。整体是一条直链：

$$ \text{token ids} \to \text{Embedding} \to \underbrace{\text{Block} \times N}_{\text{堆叠}} \to \text{RMSNorm} \to \text{Linear} \to \text{logits} $$

- **token_embeddings**：把整型 id 查表成 `d_model` 维向量；
- **N × TransformerBlock**：每个 block 含注意力 + 前馈（见 [block 笔记](./transformer_block_implementation_notes.md)），逐层精炼表示；
- **ln_final（RMSNorm）**：最后一层归一化，稳定输出尺度；
- **lm_head（Linear）**：把 `d_model` 投影到 `vocab_size`，得到每个 token 的**未归一化 logits**。

输出形状 `(.., seq, vocab_size)`：序列里每个位置都给出对整个词表的打分。注意**输出的是 logits，不是概率**——softmax 留给损失函数或采样时再做。

---

## 2. 逐个决定的理由

### 2.1 接口与契约

来自 `run_transformer_lm` / 测试：

- 超参：`vocab_size`、`context_length`、`d_model`、`num_layers`、`num_heads`、`d_ff`、`rope_theta`；
- 输入 `in_indices (batch, seq)` 是**整型 token id**；
- 输出 `(batch, seq, vocab_size)` 的 logits。
- 权重键：`token_embeddings.weight`、`layers.{i}.*`（每个 block 的全部权重）、`ln_final.weight`、`lm_head.weight`。

### 2.2 用 `nn.ModuleList` 堆叠 N 层

```python
self.layers = nn.ModuleList([
    TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, ...)
    for _ in range(num_layers)
])
```

- **为什么用 `ModuleList`**：它是专门用来存"子模块列表"的容器——列表里的每个 block 都会被正确登记为 `TransformerLM` 的子模块，参数进 `parameters()`/`state_dict()`。普通 Python `list` 不会被 `nn.Module` 登记（是 [nn_module 笔记](./nn_module_and_linear_explained.md) 里说的常见 bug）；
- **键名自然对齐**：`ModuleList` 的 `state_dict` 键就是 `layers.0.xxx`、`layers.1.xxx`……正好匹配官方权重的 `layers.{i}.` 命名。

`forward` 里就是简单地按顺序过每一层：

```python
x = self.token_embeddings(token_ids)
for layer in self.layers:
    x = layer(x)
x = self.ln_final(x)
return self.lm_head(x)
```

### 2.3 命名对齐，`load_state_dict` 一步装入

四类子模块的命名刻意与官方权重键完全一致：

| 权重键 | 模块 |
|---|---|
| `token_embeddings.weight` | `self.token_embeddings`（Embedding） |
| `layers.{i}.attn/ln1/ffn/ln2.*` | `self.layers[i]`（TransformerBlock） |
| `ln_final.weight` | `self.ln_final`（RMSNorm） |
| `lm_head.weight` | `self.lm_head`（Linear） |

所以 adapter 里 `lm.load_state_dict(weights)` 一步装入全部权重。各 block 内部的 RoPE `cos/sin` 是 `persistent=False` buffer、不进 `state_dict`，因此不会与 `strict=True` 冲突。

### 2.4 lm_head 与嵌入是否共享权重

本实现里 `token_embeddings` 和 `lm_head` 是**两套独立权重**（测试的参考权重也分别提供 `token_embeddings.weight` 与 `lm_head.weight`）。有些模型会做 **weight tying**（输入嵌入和输出投影共享同一矩阵）以省参数，但本作业不绑定——按契约各自独立即可。

### 2.5 为什么输出 logits 而非概率

`lm_head` 之后**不接 softmax**。原因：

1. 训练时用交叉熵损失，它内部自带 log-softmax（数值更稳），直接吃 logits；
2. 推理采样时可能要先做温度缩放、top-k/top-p 等，也在 logits 上操作。

所以模型只负责产出 logits，把"变概率"留给下游，最灵活也最稳。

### 2.6 是否手写 backward

不需要。整条链全是已实现的可微子模块（Embedding、Block、RMSNorm、Linear）串联，autograd 自动求导。

---

## 3. adapter 怎么接到测试

`run_transformer_lm` 三步：

1. 构造 `TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, ...)`；
2. `lm.load_state_dict(weights)` 装入全部参考权重；
3. 返回 `lm(in_indices)`。

两个测试：

- `test_transformer_lm`：标准输入，输出与快照比对（`atol=1e-4, rtol=1e-2`）；
- `test_transformer_lm_truncated_input`：输入序列被截短，验证模型对**变长序列**都成立（因为位置、掩码都按当次 `seq` 动态生成）。

---

## 4. 为什么测试能过

结构（嵌入→N 层→归一化→输出头）、命名、logits 输出都与参考一致。链路：

```text
test_transformer_lm → run_transformer_lm（load_state_dict 装全部权重）
  → TransformerLM.forward:
      embed → [block]*N → ln_final → lm_head
  → logits (batch, seq, vocab) → numpy_snapshot 对照 → PASS
```

运行：

```sh
uv run pytest -k test_transformer_lm
```

预期 `2 passed`（含 truncated_input）。

---

## 5. 小结

1. **整体结构**：`Embedding → N×TransformerBlock → RMSNorm → Linear`，输出 `(.., seq, vocab_size)` 的 logits。
2. **ModuleList 堆叠**：用 `nn.ModuleList` 存 N 个 block，参数正确登记、键名自然是 `layers.{i}.`。
3. **命名对齐**：四类子模块键与官方权重一致，`load_state_dict(weights)` 一步装入；RoPE buffer 不进 state_dict 故不冲突。
4. **输出 logits 非概率**：softmax 留给损失/采样，数值更稳、更灵活。
5. **纯组装、backward 自动**：不引入新算子，全可微子模块串联。

至此 assignment1 的建模部分（Linear、Embedding、RMSNorm、SwiGLU、RoPE、softmax、SDPA、MHA、TransformerBlock、TransformerLM）全部实现完毕，可以从 token id 一路算到词表 logits。

---

## 参考

- 本仓库实现：[cs336_basics/nn.py](../cs336_basics/nn.py)
- 适配层：[tests/adapters.py](../tests/adapters.py)
- 测试：[tests/test_model.py](../tests/test_model.py)
- 组件笔记：[transformer_block_implementation_notes.md](./transformer_block_implementation_notes.md)、[embedding_implementation_notes.md](./embedding_implementation_notes.md)、[linear_implementation_notes.md](./linear_implementation_notes.md)
- 设计原则：[nn_module_and_linear_explained.md](./nn_module_and_linear_explained.md)（ModuleList/登记）
- 原始文献：Vaswani et al., *Attention Is All You Need*, 2017。
