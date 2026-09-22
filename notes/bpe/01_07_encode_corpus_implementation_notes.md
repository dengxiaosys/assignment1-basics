# 把语料编码成 token 数组（encode_corpus）：原理与用途

## 0. 本文目标

记录打通"文本 → 训练"链路的**最后一环**：用训练好的 BPE 把原始文本语料编码成**一维 token-id 数组并存盘（.npy）**，供 [训练脚本](../03_10_training_loop_implementation_notes.md) 的 `load_tokens` 加载。这一步之前是缺口——[train.py](../../cs336_basics/train.py) 要的是 token 数组，而磁盘上只有 `.txt`。

对应实际代码：
- 脚本：[cs336_basics/encode_corpus.py](../../cs336_basics/encode_corpus.py)
- 依赖：[Tokenizer](../../cs336_basics/bpe.py)（`from_files` + `encode_iterable`）、上游产物 `bpe_out/tinystories/{vocab.json,merges.txt}`
- 下游：[train.py](../../cs336_basics/train.py) 的 `load_tokens` / `get_batch`

前置：[Tokenizer 实现](./01_06_tokenizer_implementation_notes.md)、[BPE 训练实验](./01_05_train_bpe_tinystories_experiment_notes.md)、[数据加载](../03_08_data_loading_implementation_notes.md)。

---

## 1. 它填的是哪个缺口

完整训练链路：

```text
.txt 语料 ──[本步: encode_corpus]──> token .npy ──> load_tokens(memmap) ──> get_batch ──> train.py
```

- 训练时模型吃的是**整数 token id**，不是文本。所以必须先把语料**一次性编码**成 id 序列存盘，训练时再随机采样（见 [数据加载笔记](../03_08_data_loading_implementation_notes.md)）。
- 为什么**预先编码到磁盘**、而不是训练时现编：BPE 编码不便宜（正则预分词 + 逐预 token 合并），若每步现编会拖慢训练；预生成一次、之后训练多轮反复读，摊销掉编码成本。这也是 nanoGPT 等实现的通用做法。

---

## 2. 关键设计

### 2.1 流式编码：`encode_iterable`，内存与文件大小无关

```python
tokenizer = Tokenizer.from_files(vocab, merges, special_tokens)
ids = []
with open(input_path, encoding="utf-8") as f:
    for token_id in tokenizer.encode_iterable(f):   # 逐行流式产出，不整体读文件
        ids.append(token_id)
```

- 用 `encode_iterable`（见 [Tokenizer 笔记](./01_06_tokenizer_implementation_notes.md) §5）**逐行**编码，避免把 2.1 GB 文本一次性读进内存。
- 注意：产出的 `ids` 列表本身会驻留内存（最终要存成一个数组）。TinyStories train 约 5 亿 token 量级，`list[int]` 占内存不小，但本机内存充足；若语料再大到装不下，可改成**分块写入**（`np.memmap` 边编边写），本步未做这层优化。

### 2.2 存成 `uint16` 的 .npy

```python
arr = np.array(ids, dtype=np.uint16)
np.save(output_path, arr)
```

- **为什么 `uint16`**：TinyStories 词表 10000 < 65536，一个 token id 用 **2 字节**足够。相比 `int64`（8 字节）省 **4 倍**磁盘和内存带宽。若词表 ≥ 65536 需改用 `uint32`。
- **为什么 `.npy`**：自带 dtype/shape 头，`load_tokens` 用 `np.load(mmap_mode="r")` 可**内存映射**读取——训练时按需从磁盘取用到的片段，不必全量载入（见 [数据加载笔记](../03_08_data_loading_implementation_notes.md) §2.4）。与 train.py 的数据约定一致。

### 2.3 顺带报告压缩率与吞吐

脚本打印 `compression_ratio = 原始字节数 / token 数`（bytes/token）和 `throughput = 字节/秒`——正好是 `tokenizer_experiments` 那道题要的指标（见 [非编码问题](./01_03_bpe_conceptual_questions_notes.md) §6），编码时免费得到。

---

## 3. 用法与实测

```sh
# valid（小，先验证）
uv run python -m cs336_basics.encode_corpus \
    --vocab bpe_out/tinystories/vocab.json \
    --merges bpe_out/tinystories/merges.txt \
    --special-tokens "<|endoftext|>" \
    --input data/TinyStoriesV2-GPT4-valid.txt \
    --output data/ts_valid.npy

# train（大，同样命令换 input/output）
```

**valid 实测**：22.5 MB → **5,461,210 个 token**，耗时 24 s，**压缩率 4.12 bytes/token**，吞吐 ≈935 KB/s。压缩率 ~4 与预期一致（GPT-2 系分词器常见 3–5）。

> train（2.1 GB）单进程流式编码约需几十分钟（吞吐 ~0.9 MB/s）。这是一次性成本，产物 `ts_train.npy` 之后训练反复复用。若嫌慢，可按 [Tokenizer 笔记](./01_06_tokenizer_implementation_notes.md) §7.4 的思路做**分块 + 多进程**并行编码（按文档边界切，各块编码后拼接）——本步为简单起见用单进程流式。

---

## 4. 验证：链路真的通了

编码出的 `.npy` 已通过端到端验证：

1. **`load_tokens` 读取**：`np.load(mmap_mode="r")` 得到 shape=(5461210,)、dtype=uint16、id∈[10, 9999]（符合 vocab=10000）。
2. **端到端 mini 训练**：用 `ts_valid.npy` 当训练数据、小模型（d_model=128, 2 层）跑 30 步，`data_loaded → model_built → 训练 → val 评估 → done` 全通，loss 从 9.20 降到 8.94。

这证明 **文本 → encode_corpus → token .npy → load_tokens → get_batch → train.py** 整条链路接通，可以真正在 TinyStories 上启动训练。

---

## 5. 现在能做什么 / 还差什么

**能做**：链路已通，可 `uv run cs336-train --train-path data/ts_train.npy --val-path data/ts_valid.npy --dtype-tokens uint16 --vocab-size 10000 ...` 启动训练（CPU 上先用小模型 + 少步数验证 loss 下降；完整训练需 GPU）。

**仍待做**（不阻塞训练本身）：
- **超参定标**：train.py 默认超参是占位值，TinyStories 该用多大模型/多少步/多大 lr，按 handout 的 `learning_rate` / `batch_size_experiment` 等实验题设定。
- **文本生成（decoding）**：handout §6 的 `decoding` 题——训完后给 prompt 自回归采样（temperature/top-p）生成文本，再 `Tokenizer.decode` 转回字符串。尚未实现，训练之后再做。

---

## 6. 小结

1. **作用**：把 `.txt` 语料用训练好的 BPE 编码成一维 token-id 数组存 `.npy`，填上"文本→训练"链路缺的一环。
2. **流式**：`encode_iterable` 逐行编码，读文件不占大内存。
3. **uint16 + .npy**：vocab<65536 用 2 字节/token，`.npy` 支持 memmap，与 train.py 数据约定一致。
4. **顺带**：报告压缩率（valid 实测 4.12 bytes/token）与吞吐，正好服务 `tokenizer_experiments`。
5. **已验证**：load_tokens 读取 + 端到端 mini 训练全通，链路打通。

---

## 参考

- 脚本：[cs336_basics/encode_corpus.py](../../cs336_basics/encode_corpus.py)
- 依赖：[Tokenizer](../../cs336_basics/bpe.py)、[BPE 训练实验](./01_05_train_bpe_tinystories_experiment_notes.md)（产物来源）
- 下游：[train.py](../../cs336_basics/train.py)、[训练脚本笔记](../03_10_training_loop_implementation_notes.md)、[数据加载笔记](../03_08_data_loading_implementation_notes.md)
- 相关：[Tokenizer 实现](./01_06_tokenizer_implementation_notes.md)（encode_iterable、复杂度/并行）、[非编码问题](./01_03_bpe_conceptual_questions_notes.md)（压缩率）
