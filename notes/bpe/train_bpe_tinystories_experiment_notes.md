# 在 TinyStories 上训练 BPE（train_bpe_tinystories）：实验、结果与原理

## 0. 本文目标

记录 CS336 assignment1 §2.5 Problem `train_bpe_tinystories` 的**实验实现与结果**：在 TinyStories 全量 train（2.1 GB）上训练一个 vocab_size=10000 的字节级 BPE，序列化产物，并报告耗时/内存/最长 token、做 profile。区别于 [train_bpe 实现笔记](./train_bpe_implementation_notes.md)（讲算法本身），本文聚焦**把算法跑到真实大语料上**要解决的工程问题——尤其是**并行预分词**——以及从实测数据里能读出什么。

对应实际代码：
- 训练算法（含并行开关）：[cs336_basics/bpe.py](../../cs336_basics/bpe.py) 的 `train_bpe(..., num_processes=)`、`find_chunk_boundaries`、`_pretoken_counts_parallel`
- 实验脚本：[cs336_basics/train_bpe_experiment.py](../../cs336_basics/train_bpe_experiment.py)
- 产物：`bpe_out/tinystories/vocab.json`、`bpe_out/tinystories/merges.txt`

前置：[train_bpe 实现笔记](./train_bpe_implementation_notes.md)、[非编码问题](./bpe_conceptual_questions_notes.md) §4（本文的结论也回填到那里）。

---

## 1. 实验设置与如何复现

```sh
uv run python -m cs336_basics.train_bpe_experiment \
    --input data/TinyStoriesV2-GPT4-train.txt \
    --vocab-size 10000 --num-processes 16 \
    --out-dir bpe_out/tinystories
```

- **语料**：TinyStories train，2,227,753,162 字节（≈2.1 GB），约 4.4 亿词、1560 万行。
- **词表**：10000（= 256 字节 + 1 个 `<|endoftext|>` + 9743 个合并）。
- **并行度**：16 进程（本机 56 核，取 16 已足够）。
- **机器**：56 核、约 65 GB 可用内存。

脚本做四件事：计时训练、测峰值内存、序列化 vocab/merges、报告最长 token。

---

## 2. 实测结果

| 指标 | 值 |
|---|---|
| 训练耗时 | **≈155 秒（2.6 分钟）** |
| 主进程峰值 RSS | ≈0.13 GB（子进程另计，总量远低于 30 GB 上限） |
| 最终词表大小 | 10000（9743 merges） |
| 去重后预 token 数 | ≈59,933 |
| **最长 token** | **`b' accomplishment'`（15 字节）** |

**最长的若干 token**（均含前导空格）：

```text
b' accomplishment'  (15)
b' disappointment'  (15)
b' responsibility'  (15)
b' understanding'   (14)
b' compassionate'   (14)
b' Unfortunately'   (14)
b' neighbourhood'   (14)
b' determination'   (14)
b' encouragement'   (14)
```

**这合理吗？非常合理。** TinyStories 是用 LLM 合成的、面向 3–4 岁儿童的教育故事，用词高度集中且重复，像 `accomplishment`、`understanding`、`Unfortunately`、`determination` 这些"教养/情感"主题的完整长单词高频出现，于是 BPE 把它们整体合并成单个 token。前导空格来自 GPT-2 预分词正则——它把词首空格并入词里（`" accomplishment"` 而非 `"accomplishment"`），所以词表里的完整词大多带一个前导空格。这直接印证了 BPE 的本质：**把语料里高频的字节串压成一个 token**（见 [train_bpe 笔记](./train_bpe_implementation_notes.md) §1）。

---

## 3. 关键工程点：并行预分词（否则跑不动 2.1 GB）

单元测试里的 `train_bpe`（`num_processes=1`）走的是"整体 `f.read()` + 单进程预分词"的简单路径——对 corpus.en（KB 级）没问题，但对 **2.1 GB 的 TinyStories 会非常慢**（预分词要用复杂正则单线程扫过全部文本）。所以本实验加了**可选并行预分词**。

### 3.1 怎么切块：对齐到 special token 边界

```python
def find_chunk_boundaries(file, desired_num_chunks, split_special_token):
    # 先按文件大小等分出初始边界，再把每个边界"往后挪"到最近的一个 <|endoftext|>
    ...
```

- 先按字节大小把文件等分成 N 块，再把每个切点**移动到最近的 `<|endoftext|>` 处**。
- **为什么必须对齐到 special token**：这样切块**绝不会把一个文档/预 token 劈成两半**。因为合并本来就不跨文档边界（见 [数据加载笔记](../data_loading_implementation_notes.md) 的"文档边界"讨论），在 `<|endoftext|>` 处切，各块独立预分词的计数**加起来和整体预分词完全相同**——分块是无损的。

### 3.2 怎么并行：多进程各数各的，再合并 Counter

```python
def _pretoken_counts_parallel(input_path, special_tokens, num_processes):
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    tasks = [(path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with mp.Pool(num_processes) as pool:
        for c in pool.map(_count_chunk, tasks):
            total.update(c)          # Counter 相加
    return total
```

- 每个进程读自己那段字节、`_pretoken_counts` 出一个"预 token → 频次"的 `Counter`，最后主进程把所有 `Counter` **相加**。
- **正确性保证**：预分词计数是**可加**的统计量，且切点不破坏预 token，所以并行结果与单进程**逐条一致**。我在小文件上验证过：`num_processes=1` 与 `=8` 训出的 merges 完全相同、vocab 值集合相同（valid 上 10.6s → 1.8s，~6× 加速）。
- **接口设计**：`train_bpe` 加了 `num_processes=1` 默认参数——单测和小文件走原路径（行为不变），大语料显式传 `>1` 走并行。两条路径结果一致。

---

## 4. profile：瓶颈在哪，以及并行如何改变它

分段计时（并行 16 进程）：

| 阶段 | 耗时 | 占比 |
|---|---|---|
| 并行预分词 | 64.3 s | 44% |
| 合并循环 | 82.0 s | 56% |

**读出的关键结论——瓶颈会转移**：

- **并行之前**：预分词是**绝对瓶颈**。它要用 GPT-2 那个带 `\p{L}`/`\p{N}`/负向先行的复杂正则 `re.finditer` **扫过 2.1 GB 全文本**，是一次昂贵的全量字符串遍历；而合并循环只在压缩后的"预 token → 频次"表（去重后仅 ~6 万条）上做。所以 handout 明确建议"并行化预分词"。
- **并行之后**：预分词被 16 进程分摊，占比降到 44%；**纯 Python、单线程的合并循环**（9743 轮，每轮 `max` 选最优对 + 增量更新，见 [train_bpe 笔记](./train_bpe_implementation_notes.md) §4）反倒成了略大的一头（56%）。
- **启示**：优化要**跟着瓶颈走**。第一步优化预分词（并行）收益最大；并行后若想再快，得优化合并循环——例如用**堆/优先队列**维护"当前最大频次对"（省去每轮 `max` 的线性扫描），或把合并核心用 Rust/C++ 重写（handout 提到的 PyO3/nanobind 路线）。对本作业 2.6 分钟已达标，无需再优化。

> 为什么合并循环即便"增量更新"过仍占一半：增量更新省掉的是"每轮重扫全语料重建 pair 计数"，但**每轮仍要 `max` 遍历一次当前所有相邻对**（pair 种类可能上万），9743 轮累积起来就不小。堆能把这块的每轮成本从 O(pairs) 降到 O(log)。

---

## 5. 产物与后续用途

序列化到 `bpe_out/tinystories/`：

- **`vocab.json`**（≈161 KB）：`{token_str: id}`，用 GPT-2 的"字节→可打印字符"映射（`gpt2_bytes_to_unicode_safe`）把 bytes 编码成可读、可往返的字符串。
- **`merges.txt`**（≈83 KB）：每行 `tok1 tok2`，按创建顺序。

> **别被 `vocab.json` 里的 `Ġ` / `Ċ` 吓到**：打开会看到 `Ġnamed`、`Ġfriends`、`ĠLily` 这类字符，它们**不是非英文 token**——`Ġ` 是**空格**（`0x20`）的可打印替身、`Ċ` 是**换行**（`0x0A`）的替身。即磁盘上的 `"Ġnamed"` 就是真实 token `b' named'`（带前导空格的 named）。这是 GPT-2 为把"含空白字节的字节级 token"无损存成文本而定义的可逆编码，加载时会还原成真实字节。详见 [Tokenizer 笔记](./tokenizer_implementation_notes.md) §2 的说明。（本文 §2 展示最长 token 用的是 `b' accomplishment'` 这种**真实字节形式**，在 vocab.json 里则显示为 `Ġaccomplishment`。）

这套产物可被 [Tokenizer.from_files](../../cs336_basics/bpe.py) 直接加载（已验证能正常 encode/decode），供接下来的实验使用：

- `tokenizer_experiments`：用它算 TinyStories 的压缩率（bytes/token）、跨域编码等（见 [非编码问题](./bpe_conceptual_questions_notes.md) §6）；
- 用 [训练脚本](../training_loop_implementation_notes.md) 把 TinyStories 编码成 token 数组，真正训练一个语言模型。

---

## 6. 小结

1. **结果**：2.1 GB TinyStories、vocab 10000，**≈155 秒**训完，峰值内存很低；**最长 token `b' accomplishment'`（15 字节）**，全是高频完整长词，符合"BPE 压缩高频串 + 语料同质"的预期。
2. **能跑动大语料的关键**：**并行预分词**——按 `<|endoftext|>` 边界无损切块、多进程各数各的再合并 `Counter`；`train_bpe` 用 `num_processes` 开关兼容单测路径，两路结果一致。
3. **profile**：并行后预分词 44% / 合并 56%——**瓶颈从预分词转移到合并循环**，印证"优化跟着瓶颈走"，也说明并行是达标（<2 分钟量级）的关键。
4. **产物**：`vocab.json` + `merges.txt`，可被 `Tokenizer.from_files` 加载，供压缩率实验与 LM 训练。

---

## 参考

- 训练算法与并行：[cs336_basics/bpe.py](../../cs336_basics/bpe.py)
- 实验脚本：[cs336_basics/train_bpe_experiment.py](../../cs336_basics/train_bpe_experiment.py)
- 相关笔记：[train_bpe 实现](./train_bpe_implementation_notes.md)、[Tokenizer 实现](./tokenizer_implementation_notes.md)、[非编码问题](./bpe_conceptual_questions_notes.md)、[数据加载](../data_loading_implementation_notes.md)
- handout：[cs336_assignment1_basics_extracted.md](../cs336_assignment1_basics_extracted.md) §2.5（`train_bpe_tinystories`）
