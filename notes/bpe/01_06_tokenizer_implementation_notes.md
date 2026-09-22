# 实现 BPE 分词器的编码/解码（Tokenizer 类）：原理与验证

## 0. 本文目标

记录 CS336 assignment1 §2.6 **Tokenizer** 的实现：加载训练好的 vocab/merges 后，怎么把文本 `encode` 成 token id、把 id `decode` 回文本。这是 [BPE 训练](./01_04_train_bpe_implementation_notes.md) 的下游——训练学出"词表+合并规则"，Tokenizer 则用它们做实际的编解码，是模型训练/推理的数据入口。

对应实际代码：
- 实现：[cs336_basics/bpe.py](../../cs336_basics/bpe.py) 的 `Tokenizer` 类
- 接线：[tests/adapters.py](../../tests/adapters.py) 的 `get_tokenizer`
- 测试：[tests/test_tokenizer.py](../../tests/test_tokenizer.py)（24 个，逐字节对拍 tiktoken 的 GPT-2）

前置：[BPE 训练](./01_04_train_bpe_implementation_notes.md)（vocab/merges 怎么来）、[BPE 原理](./01_01_bpe_tokenizer.md)、[非编码问题](./01_03_bpe_conceptual_questions_notes.md)（Unicode/UTF-8）。

---

## 1. 大图：encode 是训练的"镜像"

训练时我们把语料合并出 merges；编码时对**新文本**复现同样的合并过程，就能得到一致的 token 切分。encode 四步：

```text
文本
 └─(1) 按 special tokens 切分（special 段单独成 token，不参与合并）
      └─(2) 普通段用 GPT-2 正则预分词成"预 token"
           └─(3) 每个预 token 拆成单字节，按 merges 的创建顺序反复合并
                └─(4) 查 vocab，把每个最终 token 映射成整数 id
```

decode 则是逆过程：id → 查 vocab 得字节 → 拼接 → UTF-8 解码回字符串。

**关键正确性要求**：测试要求 `encode` 的输出**逐 id 等于 tiktoken 的 GPT-2 分词器**。这意味着上面每一步都得和 GPT-2 一致——尤其是预分词正则（同一个 `PAT`）和"按 merges 顺序合并"这条规则。

---

## 2. 构造：`__init__` 存了哪些东西

```python
def __init__(self, vocab, merges, special_tokens=None):
    self.vocab = vocab                                  # id -> bytes
    self.byte_to_id = {b: i for i, b in vocab.items()}  # bytes -> id（反向表）
    self.merges = merges
    self.merge_rank = {pair: r for r, pair in enumerate(merges)}  # pair -> 优先级
    self.special_tokens = special_tokens or []
```

- **`vocab`（id→bytes）**：decode 时用（id 查字节）。
- **`byte_to_id`（bytes→id）**：encode 时用（合并后的 token 字节查 id）。存反向表避免每次线性搜索。
- **`merge_rank`（pair→序号）**：把 merges 列表转成"pair → 它的创建次序"。这是 encode 合并时的**优先级依据**（见 §3.2）——越早创建的合并优先级越高。用 dict 查 rank 是 O(1)。
- **`special_tokens`**：编码时要特殊处理的字符串列表。

> `from_files` 类方法（handout 也要求）：从磁盘的 vocab.json / merges.txt 反序列化构造，用 GPT-2 的"字节↔可打印字符"映射解析（与训练输出/测试存储格式一致）。

> **关于磁盘里的 `Ġ` / `Ċ`（重要，别被吓到）**：打开 `vocab.json` / `merges.txt` 会看到大量 `Ġne`、`Ġnamed`、`Ġfriends` 这类"奇怪"字符——它们**不是非英文 token**，而是 GPT-2 用来**可视化空白字节**的替身符号。BPE 的 token 本质是**任意字节序列**，可能含空格、换行等不可打印/破坏文本格式的字节，无法直接安全地写进文本文件。于是 GPT-2 定义了一个**可逆映射**（我们代码里的 `gpt2_bytes_to_unicode_safe`）：把这些字节平移到一段可打印码点——**空格 `0x20` → `Ġ`（U+0120）**、**换行 `0x0A` → `Ċ`**，而字母/数字等可打印 ASCII 保持原样。所以：
> - 磁盘上的 `"Ġnamed"` = 真实 token **`b' named'`**（带前导空格的 named）；
> - `from_files` 加载时用逆映射把 `Ġ` 还原成空格字节，内存里就是正常的 `b' named'`。
>
> 这纯粹是**序列化的可读/可往返编码**，与 tiktoken、官方 `gpt2_vocab.json`、测试用的存储格式完全一致，不影响 token 的实际内容。

---

## 3. encode 的两个核心难点

### 3.1 special token 优先切分（含重叠处理）

special token（如 `<|endoftext|>`）**必须整体保留为单个 token，绝不能被拆或参与合并**。做法是先把它们从文本里"挖"出来：

```python
specials = sorted(self.special_tokens, key=len, reverse=True)   # 长的在前！
pattern = "(" + "|".join(re.escape(s) for s in specials) + ")"
for segment in re.split(pattern, text):
    if segment in self.special_tokens:
        ids.append(self.byte_to_id[segment.encode("utf-8")])    # 直接查 id
    else:
        ids.extend(self._encode_chunk(segment))                 # 普通段照常编码
```

两个要点：

- **用捕获组 `( ... )` 切分**：`re.split` 带捕获组时，**分隔符本身也会保留在结果里**，于是 special 段和普通段交替出现，普通段照常走 BPE、special 段直接查 id。
- **按长度降序排序（最关键）**：如果 special tokens 里同时有 `<|endoftext|>` 和 `<|endoftext|><|endoftext|>`，遇到连着两个的情形，**必须优先匹配更长的那个**。正则的 `|` 是最左最长中"最左优先"，把长的排在前面才能让它先被匹配。这正是 `test_overlapping_special_tokens` 验证的：`<|endoftext|><|endoftext|>` 要被识别成**一个** token，而非两个。

### 3.2 按 merges 顺序合并（encode 与训练一致的核心）

拿到一个预 token 的字节序列后，要复现训练时学到的合并。规则：**每一步，在当前所有相邻对里，选"创建时间最早（rank 最小）"的那个可合并对来合并，重复直到没有可合并对**。

```python
def _apply_merges(self, token_bytes):
    parts = [bytes([x]) for x in token_bytes]     # 先拆成单字节
    while len(parts) >= 2:
        # 找 rank 最小（最早创建）的可合并相邻对
        best_rank, best_i = None, -1
        for i in range(len(parts) - 1):
            r = self.merge_rank.get((parts[i], parts[i + 1]))
            if r is not None and (best_rank is None or r < best_rank):
                best_rank, best_i = r, i
        if best_rank is None:
            break                                  # 没有可合并的了
        parts[best_i : best_i + 2] = [parts[best_i] + parts[best_i + 1]]
    return parts
```

**为什么是"rank 最小优先"而不是"从左到右扫一遍"**：BPE 的合并是**有先后依赖**的——晚期的合并（如 `th`+`e`→`the`）建立在早期合并（`t`+`h`→`th`）之上。必须严格按训练时的创建顺序应用，才能复现出和训练一致（也和 GPT-2 一致）的切分。如果只是"从左到右遇到能合并就合并"，会得到错误的、与参考不符的结果。

- 用 `merge_rank` 把"哪个 pair 更该先合"变成比较整数，简单可靠。
- `parts[best_i:best_i+2] = [merged]` 是**切片赋值**：把相邻两个元素原地替换成合并后的一个，列表长度减 1。
- 循环直到某轮找不到任何在 `merge_rank` 里的相邻对为止。

> 复杂度：每个预 token 每轮 O(len) 找最优对、最多合并 len 次，单预 token 是 O(len²)。预 token 通常很短（一个词量级），所以整体很快；这也是"预分词把长文本切成短预 token"的又一好处。

### 3.3 拼起来：`_encode_chunk`

```python
def _encode_chunk(self, text):          # text 不含 special token
    ids = []
    for match in re.finditer(PAT, text):        # 同一个 GPT-2 正则
        token_bytes = match.group().encode("utf-8")
        for part in self._apply_merges(token_bytes):
            ids.append(self.byte_to_id[part])
    return ids
```

用**和训练时同一个 `PAT`** 预分词，保证切法一致；每个预 token 合并后查 `byte_to_id`。

---

## 4. decode：查表 + 容错解码

```python
def decode(self, ids):
    data = b"".join(self.vocab[i] for i in ids)     # id -> bytes 拼接
    return data.decode("utf-8", errors="replace")   # 非法字节 -> U+FFFD
```

- **先拼字节再解码**：不能逐 id 单独 decode——一个多字节 UTF-8 字符可能**跨多个 token**（比如 emoji 的几个字节被分到不同 token），必须先把所有字节拼成一个 `bytes` 再整体 `decode`（这正是 [非编码问题](./01_03_bpe_conceptual_questions_notes.md) §3b "逐字节 decode 是错的" 的实践对应）。
- **`errors="replace"`**：用户可能传入任意 id 序列，拼出的字节未必是合法 UTF-8；handout 要求此时用官方替换字符 **U+FFFD（�）** 兜底，而非抛异常。

---

## 5. encode_iterable：流式、常量内存

```python
def encode_iterable(self, iterable):
    for chunk in iterable:          # 如文件句柄，逐行
        yield from self.encode(chunk)
```

- **为什么需要它**：`encode` 要先拿到完整字符串，处理超大文件（几 GB）会 OOM。`encode_iterable` 接收一个**可迭代对象**（如打开的文件，逐行产出），**一次只编码一块、`yield` 出 id 就丢**，内存占用与文件大小无关。
- **测试怎么验证**：`test_encode_iterable_memory_usage` 把内存限制到 **1MB** 去编码一个 5MB 文件——只有真正流式才能过。对照的 `test_encode_memory_usage` 用整体 `encode` 读 5MB，被标 `xfail`（预期超内存失败），正好反衬出 `encode_iterable` 的价值。
- **`yield from`**：把 `self.encode(chunk)` 返回的 id 列表逐个转交出去，使整个方法成为一个生成器。

> 注意：按行迭代之所以安全，是因为文档用 `<|endoftext|>` 等分隔、且预 token 不跨行边界，逐行编码与整体编码结果一致。

---

## 6. adapter 与测试

`get_tokenizer(vocab, merges, special_tokens)` 直接构造 `Tokenizer`。24 个测试分几类：

| 类别 | 例子 | 验证 |
|---|---|---|
| **round-trip** | empty / 单字符 / unicode / 含 special | `decode(encode(x)) == x` |
| **对拍 tiktoken** | ascii / unicode / address / german / tinystories | `encode` 输出**逐 id 等于** GPT-2 |
| **special token** | 保留、重叠 | 不被拆、重叠取更长（§3.1） |
| **流式/内存** | encode_iterable + 1MB 限制 | 常量内存处理 5MB（§5） |

运行：

```sh
uv run pytest tests/test_tokenizer.py
```

预期 **24 passed, 1 xfailed**（xfail 是故意的内存对照，见 §5）。

> 全套 assignment1 测试此时应为 `47 passed, 1 xfailed`——Tokenizer 是最后一块。

---

## 7. 各环节的时间复杂度与可并行性

记 $N$ = 文本字节数，$L$ = 单个预 token 的字节长度（一个词量级，视作小常数），$M$ = merges 条数。

### 7.1 encode 逐环节

| 环节 | 时间复杂度 | 说明 |
|---|---|---|
| special 切分（`re.split`） | $O(N)$ | 正则扫一遍全文，把 special token 段和普通段分开 |
| 预分词（`re.finditer(PAT)`） | $O(N)$ | GPT-2 正则扫一遍普通段；实际常数较大（复杂正则） |
| 单个预 token 合并（`_apply_merges`） | $O(L^2)$ | 每轮 $O(L)$ 线性扫找 rank 最小的可合并对，最多合并 $L$ 轮 |
| 全部预 token 合并 | $O(N \cdot L)$ | 预 token 总长约 $N$，每个摊 $O(L)$；$L$ 是小常数，故近似 $O(N)$ |
| 查 `byte_to_id` | $O(1)$/token | dict 查表 |

**合起来：encode 对长度 $N$ 的文本近似 $O(N)$**（把 $L$ 当常数）。真正的常数瓶颈是预分词的复杂正则，与训练时一致。

> `_apply_merges` 的 $O(L^2)$ 值得说明：它每轮都**重新线性扫描**当前序列找 rank 最小的可合并对（`merge_rank.get`），合并一次序列缩短 1，最多 $L$ 轮，故 $O(L^2)$。因为 $L$ 只是一个词的字节数（通常 < 20），$L^2$ 很小，不构成瓶颈。若要对超长"预 token"提速，可改用**优先队列/双向链表**维护候选对（类似训练里的增量思路，见 [train_bpe 笔记](./01_04_train_bpe_implementation_notes.md) §4），把单预 token 降到 $O(L\log L)$——但对自然语言不必要。

### 7.2 decode

- **时间 $O(T + B)$**：$T$ 个 id 各查一次 vocab（$O(1)$）、拼接得 $B$ 字节，再一次性 `bytes.decode`（$O(B)$）。整体线性。
- 注意 decode 必须**先拼全部字节再解码**（§4），所以它对一次 `decode(ids)` 调用是 $O(B)$ 空间；要流式解码需自己按安全边界切分，本实现未做（decode 通常针对已成段的输出，不是瓶颈）。

### 7.3 encode_iterable

- **时间**：与把整个文件喂给 `encode` 相同（所有块的 encode 之和，线性于总字节）。
- **空间 $O(1)$（与文件大小无关）**：一次只持有当前块及其产出的 id，`yield` 后即释放。这正是它相对 `encode` 的核心优势——`test_encode_iterable_memory_usage` 用 1MB 限制处理 5MB 文件即验证此点（§5）。

### 7.4 可并行性

**encode 高度可并行，且并行方式与训练预分词同理**：

- **可并行的维度**：文本可按 **special token / 文档边界切块**（同 [train_bpe 实验笔记](./01_05_train_bpe_tinystories_experiment_notes.md) §3 的 `find_chunk_boundaries` 思路），各块**独立** encode 后按顺序拼接 id 即可——因为 encode **不跨预 token 边界**，切块无损。这对"把整个语料一次性编码成 token 数组"（LM 训练前的数据准备）尤其有用，可用 `multiprocessing` 把几 GB 语料的编码从小时级压到分钟级。
- **预 token 之间也天然独立**：`_encode_chunk` 里每个预 token 的合并互不影响，理论上可并行；但单块内预 token 很多、每个又很轻，进程/线程开销通常不划算，**按大块并行**才是实际做法。
- **不可并行的部分**：**单个预 token 内部的合并是严格有序的**（后面的合并依赖前面的结果，见 §3.2），这一步无法并行——但它只是 $O(L)$ 的小工作，无需并行。
- **decode 可并行**：可按 id 分段各自查表拼字节，最后拼接再解码；但要小心**不能在多字节 UTF-8 字符中间切分**，否则各段 `decode` 会产生 U+FFFD。稳妥做法是仅并行"id→字节"的查表，最后统一 `decode`。实践中 decode 很少是瓶颈，一般不并行。
- **encode_iterable 与并行的关系**：它是**流式（省内存）**方案，天然是串行逐块；若追求**吞吐**则用"分块 + 多进程"。两者目标不同——一个省内存、一个抢速度，可按需选择（真要兼顾就多进程 + 每进程内部流式）。

> 一句话：**encode/decode 都是线性时间；encode 可按文档边界大块并行（与训练预分词同理），单预 token 合并这步有序不可并行但极轻；encode_iterable 提供的是常量内存的串行流式，与"并行抢吞吐"是两条正交的优化路线。**

---

## 8. 小结

1. **encode 是训练的镜像**：special 切分 → GPT-2 正则预分词 → 按 merges 顺序合并 → 查 vocab 得 id。
2. **special token**：用带捕获组的 `re.split` 挖出、直接查 id；**按长度降序**保证重叠时优先匹配更长者。
3. **合并按 rank**：`merge_rank` 记录创建顺序，每步合并 **rank 最小**的可合并对——这是与训练/GPT-2 一致的关键，不能简单从左到右。
4. **decode**：先拼所有字节再整体 UTF-8 解码，非法字节用 `errors="replace"`（U+FFFD）。
5. **encode_iterable**：逐块 `yield`，常量内存处理超大文件（1MB 限制过 5MB 文件）。
6. **验证**：24 个测试逐 id 对拍 tiktoken GPT-2，含 round-trip、special、流式内存三类，全通过。

至此 §2 BPE 全部完成（训练 + 编解码）。配合前面的 [训练脚本](../03_10_training_loop_implementation_notes.md)，就能"文本 →（BPE）→ token 数组 →（get_batch）→ 训练 LM"跑通全链路。

---

## 参考

- 实现：[cs336_basics/bpe.py](../../cs336_basics/bpe.py)（`Tokenizer`）
- 适配层：[tests/adapters.py](../../tests/adapters.py)（`get_tokenizer`）
- 测试：[tests/test_tokenizer.py](../../tests/test_tokenizer.py)
- 相关笔记：[BPE 训练](./01_04_train_bpe_implementation_notes.md)、[BPE 原理](./01_01_bpe_tokenizer.md)、[非编码问题](./01_03_bpe_conceptual_questions_notes.md)、[训练脚本](../03_10_training_loop_implementation_notes.md)
- handout：[00_01_cs336_assignment1_basics_extracted.md](../00_01_cs336_assignment1_basics_extracted.md) §2.6（`tokenizer`）
