# 基于 checkpoint 的推理与自回归生成：原理、脚本与验证

## 0. 本文目标

训练完成后，checkpoint 保存的是某个参数状态下的语言模型；推理（inference）则是在**不计算梯度、不更新参数**的前提下，加载这份权重，根据 prompt 自回归地产生后续 token。本笔记记录新实现的推理脚本、checkpoint 在推理时如何加载、temperature / top-p 采样，以及当前实现的限制。

对应实际代码：

- 推理脚本：[cs336_basics/generate.py](../cs336_basics/generate.py)
- checkpoint 加载：[cs336_basics/nn.py](../cs336_basics/nn.py) 的 `load_checkpoint`
- BPE 编解码：[cs336_basics/bpe.py](../cs336_basics/bpe.py) 的 `Tokenizer`
- 训练入口：[cs336_basics/train.py](../cs336_basics/train.py)

背景：[checkpoint 笔记](./checkpointing_implementation_notes.md)、[Tokenizer 笔记](./bpe/tokenizer_implementation_notes.md)、handout §6 `decoding`。

---

## 1. 从 checkpoint 到文本的完整链路

```text
checkpoint.pt ──> TransformerLM 权重
                              │
prompt 文本 ──> BPE encode ──> prompt token ids
                              │
                    循环：前向 -> 最后位置 logits -> 采样下一个 id
                              │
                   BPE decode <── 新生成 token ids
                              │
                            completion 文本
```

语言模型在第 $t$ 步接收上下文 token $x_{1:t}$，输出最后位置的 logits 列向量 $v_t \in \mathbb{R}^{V}$，其中 $V$ 是词表大小。对 logits 做采样，得到 $x_{t+1}$；再把它追加到上下文，重复直到生成 EOS 或达到上限。

---

## 2. 推理时怎样加载 checkpoint

### 2.1 checkpoint 仍然需要先构造同构模型

当前 checkpoint 结构是：

```python
{
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "iteration": iteration,
}
```

它保存了**张量数据**，但不保存 `TransformerLM` 的网络结构代码。因此加载时必须先用**与训练完全一致**的以下超参构造模型：

- `vocab_size`
- `context_length`
- `d_model`
- `num_layers`
- `num_heads`
- `d_ff`
- `rope_theta`

若结构不匹配，`model.load_state_dict(...)` 会因键或张量形状不匹配而报错。这是保护机制，而不是脚本的问题。

当前 `ckpt/ts_valid.pt` 经检查对应的结构是：`vocab_size=10000`、`context_length=128`、`d_model=256`、`num_layers=4`、`num_heads=8`、`d_ff=1024`、`rope_theta=10000`。

> 当前 checkpoint 也不保存 BPE 的 vocab/merges 路径；推理必须使用**训练语料时的同一套 BPE 文件**，否则 token id 的语义会变，权重没有意义。未来可把模型配置、BPE 路径等元信息也写进 checkpoint，减少命令行参数；本实现优先兼容已经存在的 checkpoint。

### 2.2 推理不加载 AdamW 状态

原来的 `load_checkpoint(src, model, optimizer)` 服务于续训，必须恢复 AdamW 的 $m$、$v$、$t$。但推理完全不调用 `optimizer.step()`，加载这些状态既耗时又占 GPU 显存。

现在接口扩展为：

```python
load_checkpoint(src, model, optimizer=None, map_location=device)
```

- **续训**：传入 optimizer，行为与原来相同，恢复 model + optimizer + iteration；
- **推理**：`optimizer=None`，只恢复 model；
- **`map_location`**：支持在 B 的 CPU 上保存、在 C 的 GPU 上加载，例如 `map_location="cuda"`。

这让同一个 checkpoint 接口同时适用于训练恢复和轻量推理。

---

## 3. `cs336-generate` 的命令行接口

`pyproject.toml` 注册了：

```text
cs336-generate = cs336_basics.generate:main
```

因此既可运行：

```sh
uv run python -m cs336_basics.generate --help
```

也可运行：

```sh
uv run cs336-generate --help
```

### 3.1 用当前 `ts_valid.pt` 的可直接运行命令

```sh
uv run cs336-generate \
  --checkpoint ckpt/ts_valid.pt \
  --vocab bpe_out/tinystories/vocab.json \
  --merges bpe_out/tinystories/merges.txt \
  --special-tokens "<|endoftext|>" \
  --prompt "Once upon a time" \
  --vocab-size 10000 \
  --context-length 128 \
  --d-model 256 \
  --num-layers 4 \
  --num-heads 8 \
  --d-ff 1024 \
  --max-new-tokens 128 \
  --temperature 0.8 \
  --top-p 0.9 \
  --device cpu
```

在 GPU C 上运行时，把最后一项改为 `--device cuda`；注意 C 必须有独立的、包含 CUDA 版 PyTorch 的环境，不能复用 B 的 `.venv`。跨机开发的挂载/环境约束见 [跨机开发笔记](./cross_machine_dev_B_to_C_gpu_guide.md)。

---

## 4. 采样：temperature 与 top-p

### 4.1 Temperature

若 temperature 为 $\tau > 0$，脚本从 $q_i = \operatorname{softmax}(v_t/\tau)_i$ 采样。

- `--temperature 0`：不采样，直接取最大 logit（贪心 / argmax）；同一 checkpoint 和 prompt 下结果确定；
- $0 < \tau < 1$：分布更尖锐，更保守、更稳定；
- $\tau = 1$：原始 softmax 分布；
- $\tau > 1$：分布更平，更多样但更容易跑偏。

### 4.2 Nucleus / top-p

先按概率从高到低排序，取累计概率首次达到 `top_p` 的最小 token 集合，再在这个集合内重新归一化采样。

- `--top-p 1.0`：不截断，保留全词表；
- `--top-p 0.9`：只在覆盖 90% 概率质量的高概率 token 内采样；
- 参数必须在 $(0,1]$。

这正是 handout §6 要求的 temperature scaling + nucleus sampling。脚本用 `--seed` 控制 PyTorch 随机数，便于复现同一采样轨迹。

---

## 5. 上下文窗口与 EOS

### 5.1 上下文截断

RoPE 只预计算到 `context_length`。生成变长后，脚本每步取最近窗口：

```python
context_ids = all_ids[-context_length:]
next_logits = model(context)[0, -1]
```

这保证输入长度不超过模型训练时支持的最大长度。代价是生成超过窗口后，模型不再看得到更早的 token——这就是固定上下文 Transformer 的正常行为。

### 5.2 EOS 停止

`--eos-token` 默认是 `<|endoftext|>`。脚本编码它得到 EOS id；每步采样后若命中此 id，立即停止。EOS 是控制标记，因此最终输出会移除这个 token，不把字符串 `<|endoftext|>` 显示为用户可见的续写。

---

## 6. 实测验证

使用现有 `ckpt/ts_valid.pt`（iteration=1000）在 CPU 上进行贪心生成：

```text
prompt: Once upon a time
completion: , there was a little girl named Lily. She loved to play with her friends and play with her
```

这验证了完整推理路径：

```text
checkpoint -> model weights -> prompt encode -> autoregressive forward -> token sampling -> decode
```

1000 步、且在 validation 数据上训练的模型会明显重复，生成质量有限；这属于训练规模与数据划分的问题，不是推理脚本错误。使用完整 `ts_train.npy`、GPU、更长训练和合理的采样超参后，质量才有意义。

---

## 7. 复杂度与当前限制

当前实现没有 KV cache。假设每步窗口长度至多为 $C$、生成 $G$ 个新 token、模型层数为 $L$、模型维度为 $d$，每一步都要重新执行整段上下文的 attention，因此注意力主项约为 $O(G L C^2 d)$。

- **优点**：实现直接、容易验证，适合 assignment1；
- **代价**：每生成一个 token 都重复计算已有上下文，长生成效率较低；
- **后续优化**：KV cache 会把历史 K/V 缓存下来，每步只算新增 token，显著降低解码成本；这是后续系统/推理课程的自然延伸。

---

## 8. 小结

1. `cs336-generate` 已把已有 checkpoint 用于实际自回归推理。
2. 推理只加载 model 权重，跳过 optimizer 状态，并支持 CPU/GPU 间的 `map_location`。
3. 当前 checkpoint 没有结构/BPE 元信息，所以模型超参和 BPE 路径必须显式传入。
4. 脚本支持 prompt、最大生成 token 数、temperature、top-p、EOS 停止和固定上下文截断。
5. 已用 `ts_valid.pt` 真实生成验证；全量训练后可用同一脚本在 GPU C 上生成。

---

## 参考

- 推理实现：[cs336_basics/generate.py](../cs336_basics/generate.py)
- checkpoint 实现：[cs336_basics/nn.py](../cs336_basics/nn.py)
- BPE：[cs336_basics/bpe.py](../cs336_basics/bpe.py)
- 相关笔记：[checkpointing_implementation_notes.md](./checkpointing_implementation_notes.md)、[tokenizer_implementation_notes.md](./bpe/tokenizer_implementation_notes.md)、[cross_machine_dev_B_to_C_gpu_guide.md](./cross_machine_dev_B_to_C_gpu_guide.md)
