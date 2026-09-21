# 实现 BPE 分词器训练（train_bpe）：原理与验证

## 0. 本文目标

记录 CS336 assignment1 §2.4 **BPE 分词器训练**的实现：怎么从一份原始文本语料，学出一套"词表（vocab）+ 合并规则（merges）"。核心是三步——词表初始化、预分词、迭代合并——外加两个决定"结果是否与参考完全一致"的关键约定（并列打破规则、special token 边界），以及一个决定"能否通过速度测试"的工程点（增量更新）。

对应实际代码：
- 实现：[cs336_basics/bpe.py](../../cs336_basics/bpe.py) 的 `train_bpe`
- 接线：[tests/adapters.py](../../tests/adapters.py) 的 `run_train_bpe`
- 测试：[tests/test_train_bpe.py](../../tests/test_train_bpe.py)（`test_train_bpe` 正确性、`test_train_bpe_speed` 速度、`test_train_bpe_special_tokens`）

前置：[BPE 分词器原理](./bpe_tokenizer.md)、[手工复算例子](./byte_level_bpe_worked_example.md)、[非编码问题详解](./bpe_conceptual_questions_notes.md)。

---

## 1. 背景：BPE 训练在学什么

BPE（Byte-Pair Encoding）训练的产物是两样东西：

- **`vocab: dict[int, bytes]`**：token id → token 字节串。初始是 256 个单字节 + special tokens，训练中不断加入"合并出的新 token"。
- **`merges: list[tuple[bytes, bytes]]`**：**有序**的合并规则列表，每条 `(A, B)` 表示"把相邻的 A、B 合成 AB"。顺序即创建顺序——**编码时要按这个顺序复现合并**（见后续 Tokenizer）。

训练的本质：**在语料上反复找"最高频的相邻字节对"，把它合并成一个新 token**，从而把高频字符串（如 `the`）压成单个 token。合并越多，词表越大、序列越短——这正是子词分词"用词表换序列长度"的核心（见 [BPE 原理](./bpe_tokenizer.md)）。

---

## 2. 三步流程

### 2.1 词表初始化

```python
vocab = {i: bytes([i]) for i in range(256)}   # 256 个字节
for tok in special_tokens:
    vocab[len(vocab)] = tok.encode("utf-8")    # 追加 special tokens
```

- **先放 256 个字节**：字节级 BPE 的底座，保证**任何文本都能表示**（无 OOV，见 [非编码问题](./bpe_conceptual_questions_notes.md) §1.2）。
- **再放 special tokens**（如 `<|endoftext|>`）：它们有固定 id，**永不被拆分、也不参与合并**。
- **最终词表大小** = 256 + special 数 + 合并次数。所以合并次数 = `vocab_size - 256 - len(special_tokens)`。

### 2.2 预分词（pre-tokenization）

不是直接在整段文本上数字节对，而是先切成"预 token"再统计。两层切分：

**第一层：按 special tokens 切成硬边界段。**

```python
pattern = "|".join(re.escape(tok) for tok in special_tokens)
segments = re.split(pattern, text)
```

- 用 special tokens 把文本**切开**，special token 本身**不进入**后续统计。
- 目的：**合并绝不跨越文档边界**。`<|endoftext|>` 分隔不同文档，跨它合并没有语义意义（见 [数据加载笔记](../data_loading_implementation_notes.md) 里"文档边界"的讨论）。这也是 `test_train_bpe_special_tokens` 验证的——训练出的词表里除 special 外不含 `<|` 之类的碎片。

**第二层：段内用 GPT-2 正则抽预 token。**

```python
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
for match in re.finditer(PAT, seg):
    b = match.group().encode("utf-8")
    counts[tuple(bytes([x]) for x in b)] += 1   # {(b'l',b'o',b'w'): 5, ...}
```

- 这个正则来自 GPT-2（tiktoken#234），把文本切成"词 / 空格+词 / 数字 / 标点 / 空白"等**粗粒度片段**，需要 `regex` 包（支持 `\p{L}` 字母类、`\p{N}` 数字类）。
- **为什么要预分词**：(1) 避免跨词合并出 `dog!`/`dog.` 这种只差标点的怪 token；(2) **大幅提速**——把语料压成"预 token → 频次"表，统计对 `(t,e)` 时直接加该词频次，而非在原文逐处扫描（见 [非编码问题](./bpe_conceptual_questions_notes.md) §4）。
- **每个预 token 表示成"单字节 bytes 对象的元组"**，如 `low`→`(b'l', b'o', b'w')`。注意 Python 没有单独的 byte 类型，单个字节也是 `bytes` 对象。

### 2.3 迭代合并

反复做：数所有相邻对的频次 → 取最高频对 → 合并 → 记录 → 更新词表，直到词表填满。

```python
best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
new_token = best_pair[0] + best_pair[1]
merges.append(best_pair)
vocab[len(vocab)] = new_token
# ...把所有含 best_pair 的预 token 里的该对合并成 new_token...
```

---

## 3. 两个决定"正确性"的约定

### 3.1 并列打破：取字典序更大的对

多个对频次相同时，**必须取字典序（lexicographically）更大的那个**（handout 明确要求）。实现靠 `max` 的复合 key：

```python
best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
```

- key 是元组 `(频次, pair本身)`：先比频次，频次相同再比 `pair`（`bytes` 元组按字典序比较）。`max` 于是先选频次最高、并列时选 pair 更大者。
- **为什么这条如此关键**：`test_train_bpe` 要求学出的 `merges` **逐条精确等于**参考实现。BPE 训练是确定性算法，只要 tie-break 规则一致，结果就唯一。规则错了（比如取更小的、或按插入顺序），会在某个并列点分叉，导致后续所有 merge 全错。handout 给的例子：`('e','s')` 和 `('s','t')` 并列时取 `('s','t')`（字典序更大）。

### 3.2 special token 不参与合并

§2.2 已说：先按 special tokens 切段、special token 不进统计。这保证它们**始终是单个 token**——语言模型靠 `<|endoftext|>` 判断"文档结束/停止生成"，绝不能被拆碎或和别的字节合并。

---

## 4. 决定"速度"的工程点：增量更新

朴素实现（我最初的版本）每轮都**全量重扫**所有预 token 来重建 pair 计数、并重建整个词表字典——复杂度 O(合并次数 × 语料规模)。在 corpus.en 上跑 244 轮合并要约 2.7s，**超过 `test_train_bpe_speed` 的 1.5s 阈值**。

优化为**增量更新**：合并一个对后，只有**含该对的预 token**会变，所以只更新它们。

```python
# 维护两个索引：
pair_counts: Counter                       # 每个相邻对的加权频次
pair_to_words: dict[pair, set[int]]        # 每个对出现在哪些词（下标）

# 每轮：
best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
for wi in pair_to_words[best_pair]:        # 只遍历受影响的词
    # 1) 从 pair_counts 里撤销这个词的所有旧相邻对
    # 2) 在词内把 best_pair 合并成 new_token
    # 3) 登记新词的所有相邻对
```

- **核心思想**：pair 计数是"局部可增量维护"的——合并只影响 best_pair 所在词的相邻关系，其余词纹丝不动。用 `pair_to_words` 倒排索引直接定位受影响的词，避免每轮扫全量。
- 效果：整套 3 个测试从"超时"降到 **0.69s（正确性+速度两项）/ 3.73s（含 special snapshot）**，稳过阈值。
- **正确性不变**：增量更新只是"换一种方式维护同样的 pair 计数"，取 best_pair 的逻辑和结果与全扫版完全一致（`test_train_bpe` 仍逐条匹配参考）。

> 背景：这是 BPE 高效实现的通用套路。更极致的实现（如 Rust/C++）还会用堆/优先队列维护最大频次对；但对本作业的规模，Python + 倒排索引已足够快。handout 也提到大语料训练要靠 `multiprocessing` 并行**预分词**（那才是 GB 级语料的主瓶颈，见 [非编码问题](./bpe_conceptual_questions_notes.md) §4b）。

### 4.1 增量更新的核心循环逐行拆解（带实测例子）

这一节把 [bpe.py](../../cs336_basics/bpe.py) 里"处理受影响词"的那段（`affected = ...` 到登记新对）讲透。先看代码：

```python
affected = list(pair_to_words[best_pair])   # ① 哪些词含 best_pair
for wi in affected:
    word = words[wi]
    f = freqs[wi]                            # 这个词的频次（权重）
    # ② 撤销旧词的所有相邻对贡献
    for a, b in zip(word[:-1], word[1:]):
        pair_counts[(a, b)] -= f
        pair_to_words[(a, b)].discard(wi)
    # ③ 在词内把所有 best_pair 合并成 new_token
    merged = []
    i = 0
    n = len(word)
    while i < n:
        if i < n - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
            merged.append(new_token)
            i += 2
        else:
            merged.append(word[i])
            i += 1
    words[wi] = merged
    # ④ 登记新词的所有相邻对贡献
    for a, b in zip(merged[:-1], merged[1:]):
        pair_counts[(a, b)] += f
        pair_to_words[(a, b)].add(wi)
```

**先建立数据结构的直观**。用 handout 的经典语料 `{low:5, lower:2, widest:3, newest:6}`，预分词后：

- `words`（每个词是字节 token 的 list，用下标 0/1/2/3 标识）：
  - `words[0]=[l,o,w]` freq=5、`words[1]=[l,o,w,e,r]` freq=2、`words[2]=[w,i,d,e,s,t]` freq=3、`words[3]=[n,e,w,e,s,t]` freq=6
- `pair_counts`（相邻对 → 加权频次）初始为：
  ```text
  {l|o:7, o|w:7, w|e:8, e|r:2, w|i:3, i|d:3, d|e:3, e|s:9, s|t:9, n|e:6, e|w:6}
  ```
- `pair_to_words`（相邻对 → 含它的词下标集合），例如 `s|t → {2, 3}`（widest、newest 都含 `s,t`）。

本轮 `best_pair`：`e|s` 和 `s|t` 都是 9 并列，按 §3.1 取字典序更大的 **`s|t`**（`b's' > b'e'`）。`new_token = b'st'`。

**① 只取受影响的词**（第 97 行）。`pair_to_words[('s','t')]` = `{2, 3}`，所以只需处理 `words[2]`(widest)、`words[3]`(newest)——**其余词完全不碰**，这正是增量更新省时的关键。`list(...)` 是因为循环里要改 `pair_to_words`，先固化一份下标快照，避免"边遍历边改集合"。

**② 撤销旧贡献**（第 101–104 行）。处理 `words[2]=[w,i,d,e,s,t]`(freq=3) 时，把它当前**每个相邻对**从 `pair_counts` 里各减去 `f=3`，并从倒排索引里移除 `wi=2`。实测撤销后：

```text
s|t: 9→6   e|s: 9→6   w|i: 3→0   i|d: 3→0   d|e: 3→0   (这些都是 widest 贡献的)
```

> 为什么要先"整词撤销"再重算？因为合并 `s,t` 会改变这个词里 `s,t` **周围**的相邻关系（`e|s` 消失、新出现 `e|st`）。与其精细地判断"哪几个对变了"，不如**把这个词的旧贡献全撤掉、合并后按新词全登记一遍**——简单且不会漏。别的词没被动过，不必重算。

**③ 词内合并**（第 105–116 行）。用 `while` 扫一遍：遇到 `word[i]==b's' and word[i+1]==b't'` 就把这两个合成 `b'st'` 并 `i+=2`（**跳过已消费的两个**），否则原样保留、`i+=1`。`widest` 于是 `[w,i,d,e,s,t] → [w,i,d,e,st]`。用 `i+=2` 而非 `+=1` 是为了**不重复消费**——一个位置被合并后不能再作为下一对的左元素。

**④ 登记新贡献**（第 117–120 行）。对合并后的 `[w,i,d,e,st]` 的每个相邻对各加回 `f=3`、并把 `wi=2` 加进倒排。实测新增了 `e|st:3`（原来没有这个对），`d|e` 等又加回 3：

```text
… e|st: 0→3 (新对), d|e/i|d/w|i: 回到 3, s|t: 仍是 6（还差 newest 没处理）…
```

对 `words[3]=[n,e,w,e,s,t]`(freq=6) 重复 ②③④ 后，`newest → [n,e,w,e,st]`，`e|st` 再 +6 变成 **9**。

**收尾**（第 123–124 行）：`del pair_counts[best_pair]` + `pair_to_words.pop(best_pair)`。此时 `s|t` 的计数应已被减到 0（两个含它的词都合并掉了），显式删除是**保险**——确保这个已消灭的对不会再被 `max` 选中。本轮结束时的 `pair_counts`（实测）：

```text
{l|o:7, o|w:7, w|e:8, e|r:2, w|i:3, i|d:3, d|e:3, n|e:6, e|w:6, e|st:9}
```

下一轮 `max` 会选出 `e|st`（=9）——与 handout "第二轮合并 `e st`" 完全吻合，印证了增量维护的 `pair_counts` 和全量重算等价。

**一句话**：这段做的是"**局部差量维护**"——用倒排索引锁定含 best_pair 的少数词，对每个词"整词旧对全减 → 合并 → 整词新对全加"，把每轮成本从"全语料"压到"受影响词"，同时保证 `pair_counts` 始终精确、结果与全扫版一致。

---

## 5. adapter 与测试

`run_train_bpe` 延迟导入并转发到 `train_bpe`。三个测试各验证一面：

| 测试 | 验证什么 | 通过依赖 |
|---|---|---|
| `test_train_bpe` | 学出的 `merges` **逐条**等于参考、`vocab` 键值集合匹配（vocab_size=500） | §3.1 并列打破规则正确 |
| `test_train_bpe_speed` | corpus.en 上训练 < 1.5s | §4 增量更新 |
| `test_train_bpe_special_tokens` | special token 不被合并（词表里除它外不含 `<|`）、快照匹配 | §3.2 special 边界切分 |

运行：

```sh
uv run pytest tests/test_train_bpe.py
```

预期 `3 passed`。

> 测试怎么比对 merges：参考文件用 GPT-2 的"字节→可打印字符"映射（`gpt2_bytes_to_unicode`）把字节编码成可读字符存盘；测试再用其逆映射解回 `bytes`，与我们输出的 `bytes` 元组逐条比较。所以我们只要输出**原始 bytes** 的 merges 即可，无需关心那层可读编码。

---

## 6. 小结

1. **产物**：`vocab`（id→bytes）+ 有序 `merges`（合并规则）。
2. **三步**：初始化（256 字节 + special）→ 预分词（special 切段 + GPT-2 正则抽预 token，表示成字节元组并计数）→ 迭代合并（取最高频相邻对，合并到词表填满）。
3. **正确性两约定**：并列取**字典序更大**的对（`max` 复合 key）；special token **不参与合并**（切段隔离）。
4. **速度关键**：**增量更新**——用 `pair_to_words` 倒排索引只更新受影响的词，避免每轮全扫；把 2.7s 降到亚秒级。
5. **测试三面**：正确性（逐条匹配参考）、速度（<1.5s）、special（不被合并）。

至此 BPE 训练完成。下一步实现 `Tokenizer` 类（encode/decode），它会**加载**这里训练出的 vocab/merges，把文本编码成 token id 序列（供 [训练脚本](../training_loop_implementation_notes.md) 使用）。

---

## 参考

- 实现：[cs336_basics/bpe.py](../../cs336_basics/bpe.py)
- 适配层：[tests/adapters.py](../../tests/adapters.py)
- 测试：[tests/test_train_bpe.py](../../tests/test_train_bpe.py)
- 相关笔记：[BPE 原理](./bpe_tokenizer.md)、[手工复算例子](./byte_level_bpe_worked_example.md)、[非编码问题](./bpe_conceptual_questions_notes.md)
- handout：[cs336_assignment1_basics_extracted.md](../cs336_assignment1_basics_extracted.md) §2.4（`train_bpe`）
- GPT-2 预分词正则：`openai/tiktoken#234`
