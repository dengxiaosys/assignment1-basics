# BPE 分词器（§2）的非编码问题详解与背景知识

## 0. 本文目标

CS336 assignment1 §2 除了要写代码（`train_bpe`、`Tokenizer`）外，还夹着一批**问答/分析题**：Unicode 理解、UTF-8 编码取舍、训练观察、分词器对比、压缩率与吞吐估计。本文把 §2 里**所有非编码问题**逐题答清，并补足背后的背景知识（Unicode / UTF-8 / 子词切分 / BPE / 压缩率 / 吞吐）。

对应 handout（`notes/00_01_cs336_assignment1_basics_extracted.md`）的 §2，涉及题号：`unicode1`、`unicode2`、`train_bpe_tinystories`(问答部分)、`train_bpe_expts_owt`(问答部分)、`tokenizer_experiments`。

> **说明**：`unicode1`/`unicode2` 是确定性问题，下面的答案都在 Python 里实测过。`train_bpe_*` 和 `tokenizer_experiments` 里有几问依赖**实际训练出的 BPE**（如具体耗时、最长 token、实测压缩率）——这些我给出**方法 + 预期范围 + 判断依据**，并标注 🔧「跑完实验回填」，等 `run_train_bpe`/`Tokenizer` 实现并训练后填入实测值。

---

## 1. 背景：从字符到字节到子词

在答题前，先把 §2 的三层表示串清楚，后面很多问题都建立在此。

### 1.1 Unicode 码点（code point）

Unicode 给每个字符分配一个整数**码点**，如 `'s'`→115（写作 `U+0073`）、`'牛'`→29275。Python 里 `ord(c)` 取码点、`chr(n)` 反查。截至 Unicode 17.0 有约 16 万字符——**码点空间大且稀疏**，直接拿码点当词表不现实（15 万词表、大量字符极罕见）。

### 1.2 Unicode 编码：把码点变成字节

**编码（encoding）**把码点序列转成**字节序列**（0–255 的整数）。三种标准编码：UTF-8 / UTF-16 / UTF-32。字节词表只有 256 项，**极其可控**，而且**任何文本都能表示成 0–255 的字节**——所以字节级分词**天然没有 OOV（未登录词）问题**。

### 1.3 子词（subword）：字节和词之间的折中

字节级序列虽无 OOV，但**太长**（一句 10 词的话可能 50+ 字节 token），拉长序列、增加计算和长程依赖。子词分词是折中：**用更大的词表换更短的序列**。BPE 就是构造子词词表的算法——**迭代地把最高频的相邻对合并成新 token**，高频字节串（如 `the`）被压成单个 token。

---

## 2. `unicode1`：理解 Unicode（已实测）

### (a) `chr(0)` 返回什么字符？

返回**空字符 NULL**（码点 U+0000）——一个不可打印的控制字符，其字符串字面为 `'\x00'`。

### (b) 它的 `__repr__()` 和打印表示有何不同？

`repr(chr(0))` 显示为**可见的转义形式** `'\x00'`（便于调试看清它是什么）；而 `print(chr(0))` **什么也看不见**（终端不渲染 NULL 控制字符，只输出一个不可见字节）。一句话：repr 给人看的转义、print 是原始不可见字符。

### (c) 它出现在文本里会怎样？

它**照常作为字符存在**（占一个位置、计入 `len`），但**打印时不可见**。实测：

```python
>>> s = "this is a test" + chr(0) + "string"
>>> s                       # repr：能看到 \x00
'this is a test\x00string'
>>> print(s)                # 打印：中间的 NULL 不可见
this is a teststring
>>> len(s)
21
```

即字符串里悄悄多了个字符，肉眼从打印结果看不出来——这类不可见字符是文本处理里常见的坑。

---

## 3. `unicode2`：Unicode 编码（已实测）

### (a) 为什么偏好在 UTF-8 字节上训练，而非 UTF-16/UTF-32？

核心原因：**UTF-8 对常见文本最省字节，且没有字节序/BOM 麻烦、ASCII 兼容**。实测各编码的字节数：

| 文本 | UTF-8 | UTF-16 | UTF-32 |
|---|---|---|---|
| `"s"` | **1** | 4 | 8 |
| `"abc"` | **3** | 8 | 16 |
| `"牛"` | 3 | **4** | 8 |
| `"hello! こんにちは!"` | **23** | 28 | 56 |

要点：

- **UTF-8 是变长（1–4 字节）**：ASCII 字符只占 1 字节，绝大多数网页文本（英文、代码、数字、标点）几乎全是 ASCII，所以 UTF-8 序列**最短**——序列短意味着模型要处理的 token 更少。UTF-16 最少 2 字节、UTF-32 恒 4 字节，对 ASCII 极其浪费（`"abc"` 在 UTF-32 里要 16 字节）。
- **UTF-16/32 带 BOM 和字节序问题**：实测 `'A'.encode('utf-16')` = `[255, 254, 65, 0]`——前两字节是 BOM（字节序标记），且存在大端/小端两种排列。这些会让**字节分布更杂乱、引入无信息的固定模式**（如大量 `\x00`），不利于分词器学到有意义的合并。
- **UTF-8 是互联网事实标准**（>98% 网页），训练数据本身多为 UTF-8，直接用最自然。
- **256 字节词表 + 无 OOV**：三者都能做到无 OOV，但 UTF-8 的字节分布最贴合真实文本，合并统计更有意义。

一句话：**UTF-8 变长、ASCII 省字节、无字节序/BOM 噪声、又是网络标准，所以最适合做字节级 BPE 的底座。**

### (b) 给出的解码函数为什么错？举一个出错的输入。

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
```

**为什么错**：它**逐个字节单独 decode**，但 UTF-8 是**变长编码**——一个非 ASCII 字符由 2–4 个字节共同表示，单看其中一个字节不是合法的完整 UTF-8 序列。

**反例**：`"牛".encode("utf-8")` = `b'\xe7\x89\x9b'`（3 个字节 `[231,137,155]`）。把它喂给该函数会抛错：

```python
>>> decode_utf8_bytes_to_str_wrong("牛".encode("utf-8"))
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe7 in position 0: unexpected end of data
```

因为 `0xe7` 是"3 字节字符的引导字节"，单独 decode 时后面缺了 2 个续接字节，直接失败。正确做法是对**整个字节串**一次 `bytestring.decode("utf-8")`（得到 `'牛'`），让解码器按 UTF-8 的多字节规则组合。（即便某个多字节字符的各字节碰巧都落在能单独解码的范围，逐字节法也会把一个字符拆成多个错误字符。）

### (c) 给一个不解码为任何字符的双字节序列。

`b'\xff\xff'`。解释：**`0xff` 在 UTF-8 里是永远非法的字节**（UTF-8 的引导字节最高只到 `0xf4`，`0xff` 不构成任何合法序列的一部分），所以 `b'\xff\xff'.decode('utf-8')` 抛 `UnicodeDecodeError`。另一类反例是 `b'\x80\x80'`——`0x80` 是"续接字节"（10xxxxxx），但前面没有引导字节，孤立出现同样非法。

> 背景：UTF-8 字节分三类——单字节 ASCII（`0xxxxxxx`）、引导字节（`110xxxxx`/`1110xxxx`/`11110xxx`）、续接字节（`10xxxxxx`）。合法序列必须"引导字节 + 正确数量的续接字节"。破坏这个结构（如续接字节打头、或用 `0xff` 这种非法字节）就无法解码。这也呼应了 §2.6.2 解码时用 `errors='replace'` 把坏字节替换成 U+FFFD（�）的做法。

---

## 4. `train_bpe_tinystories`（问答部分）

> 已在本机（56 核，`num_processes=16`）实跑，数值为实测；实验脚本见 [train_bpe_experiment.py](../../cs336_basics/train_bpe_experiment.py)，详细说明见 [01_05_train_bpe_tinystories_experiment_notes.md](./01_05_train_bpe_tinystories_experiment_notes.md)。

### (a) 训练耗时/内存？最长 token 是什么？合理吗？

- **耗时/内存**（实测）：TinyStories train（2.1 GB，约 4.4 亿词）训练 vocab_size=10000，**耗时约 155 秒（2.6 分钟）**，主进程峰值 RSS 约 0.13 GB（子进程另计，整体也远低于 handout 的 30 GB 上限）。用 `multiprocessing` 并行预分词（16 进程）+ 把 `<|endoftext|>` 作为文档边界切块，符合 handout"2 分钟量级"的提示。
- **最长 token & 是否合理**（实测）：最长 token 是 **`b' accomplishment'`（15 字节，含前导空格）**；紧随其后的还有 `b' disappointment'`、`b' responsibility'`、`b' understanding'`、`b' compassionate'`、`b' Unfortunately'` 等。**非常合理**：这些都是 TinyStories（幼儿教育故事）里高频出现的完整长单词；BPE 把高频串压成单 token，语料越同质、这类长词越容易整体成词。前导空格是因为 GPT-2 正则把词首空格并入词里（`" accomplishment"`）。

### (b) profile 一下，哪一步最耗时？

**实测（并行 16 进程后）**：预分词 **64.3s（44%）**、合并循环 **82.0s（56%）**，去重后预 token 仅约 6 万个。

- 值得注意的转折：**并行化之前**，预分词（用 GPT-2 复杂正则 `re.finditer` 扫过 GB 级全语料）是**绝对瓶颈**；一旦用 `multiprocessing` 把它分摊到 16 进程，预分词占比就降到 44%，反倒是**纯 Python 单线程的合并循环**（9743 轮 `max` + 增量更新）成了略大的一头（56%）。
- 换句话说：**单进程时预分词最慢，是首要优化目标（并行）；并行后瓶颈转移到合并循环**——要再快就得优化合并（如堆维护最大频次、或用系统语言）。这与 handout"预分词是主瓶颈、建议并行"的指引一致，也解释了为什么并行是这道题达标（<2 分钟量级）的关键。

**背景：为什么预分词能加速合并**。预分词把语料压成"预 token → 频次"的表：`text` 出现 10 次就只存一条 `{(t,e,x,t):10}`，统计相邻对 `(t,e)` 时直接加 10，而不必在原文里逐处扫描。合并也**不跨预 token 边界**（避免 `dog!` 和 `dog.` 因标点被合成怪 token）。

---

## 5. `train_bpe_expts_owt`（问答部分）

### (a) 在 OWT 上训练（词表 32K），最长 token 是什么？合理吗？

🔧 预期 OWT 的最长 token 会**更长也更"杂"**：因为 OWT 是**真实网页**，含大量重复的 URL 片段、HTML/代码残留、长专有名词、连写串等。可能出现像超长的网址片段、`----------` 之类的分隔线、或长英文复合词被整体合并。**是否合理**：合理但也暴露语料噪声——高频出现的长串（哪怕是"脏"串）都会被 BPE 吃成单 token，这既是压缩的体现，也说明真实语料里存在大量模板化/噪声文本。〔具体字节待回填〕

### (b) 对比 TinyStories 与 OWT 训出来的分词器。

🔧 预期对比（结合 TinyStories 与 OWT 的数据集差异——合成极简 vs 真实复杂）：

| 维度 | TinyStories 分词器（10K） | OWT 分词器（32K） |
|---|---|---|
| 词表大小 | 10K | 32K |
| token 内容 | 干净、常见幼儿词汇、故事高频词 | 更杂：网址片段、代码/HTML 残留、长专名、噪声串 |
| 最长 token | 常见长单词 | 更长且更"脏" |
| 覆盖面 | 窄（题材同质） | 宽（主题/风格多样） |
| 在本域文本上的压缩率 | 高（词表贴合语料） | 高但需更大词表才达到 |

核心结论：**词表是语料的镜子**。TinyStories 分词器学到的是干净、同质的日常词；OWT 分词器学到的是真实网络文本的全谱（含噪声）。词表大小差异（10K vs 32K）也反映了两者复杂度的差距。

---

## 6. `tokenizer_experiments`：分词器实验

### (a) 两个分词器的压缩率（bytes/token）各是多少？

**压缩率定义**：`压缩率 = 原始 UTF-8 字节数 / 编码后 token 数`（bytes/token），**越大表示压缩越好**（平均每个 token 顶更多字节）。

🔧 方法：各从 TinyStories、OWT 采 10 篇文档，用**对应的**分词器 encode，统计 `sum(len(doc.encode('utf-8'))) / sum(len(ids))`。预期两者都在 **~4 bytes/token** 量级（GPT-2 系分词器常见范围 3–5），TinyStories（10K）可能略低于 OWT（32K），因为词表更小、长词合并更少。〔实测值待回填〕

**背景**：压缩率直接关系模型效率——同样一段文本，压缩率越高 token 越少，模型算得越快、上下文能装下越多内容。这也是子词优于纯字节（字节级压缩率≈1）的核心收益。

### (b) 用 TinyStories 分词器编码 OWT 样本会怎样？

🔧 预期：**压缩率明显下降**（bytes/token 变小，序列变长）。因为 TinyStories 分词器的词表是在**干净、同质的幼儿故事**上学的，缺少 OWT 里的网址、专名、代码、复杂词汇对应的合并规则；遇到这些，只能退化成更细碎的字节/短子词组合。定性看：OWT 文本会被切得**更碎**，很多在 OWT 分词器里是单 token 的串，在 TinyStories 分词器里要用好几个 token 拼。这说明**分词器有强烈的域依赖**——跨域使用会损失压缩效率。

### (c) 估计吞吐（bytes/second），编码 Pile（825GB）要多久？

🔧 方法：`吞吐 = 采样文本的字节数 / encode 耗时`。假设实测吞吐为 $R$ 字节/秒，则编码 Pile 的时间 $\approx \dfrac{825\times 10^9}{R}$ 秒。

举例（占位，待实测）：若 $R \approx 2\,\text{MB/s}$（纯 Python BPE 常见量级），则

$$T \approx \frac{825\times10^9}{2\times10^6}\ \text{s} \approx 4.1\times10^5\ \text{s} \approx 4.8\ \text{天（单进程）}$$

**背景**：这说明大规模语料**必须并行/优化**编码——`encode_iterable` 做流式、`multiprocessing` 分块、或用 Rust/C++ 实现核心循环，才能把"天"级降到可接受范围。这也是工业界分词器（tiktoken 等）用系统语言实现的原因。〔实测吞吐与换算待回填〕

---

## 7. 汇总：哪些已定、哪些待回填

| 题 | 状态 |
|---|---|
| `unicode1` (a)(b)(c) | ✅ 已实测确定 |
| `unicode2` (a)(b)(c) | ✅ 已实测确定 |
| `train_bpe_tinystories` (a) 耗时/内存/最长token | 🔧 方法+预期已给，数值待训练回填 |
| `train_bpe_tinystories` (b) profile | 🔧 预期"预分词最耗时"，数字待回填 |
| `train_bpe_expts_owt` (a)(b) | 🔧 方法+预期已给，具体 token 待回填 |
| `tokenizer_experiments` (a)(b)(c) | 🔧 方法+公式已给，实测值待回填 |

待 `run_train_bpe` 与 `Tokenizer` 实现、并在 TinyStories/OWT 上训练后，把 🔧 项的实测数值填回本文即可。

---

## 8. 小结

1. **三层表示**：Unicode 码点（大而稀疏）→ UTF-8 字节（256 词表、无 OOV）→ BPE 子词（压缩序列）。
2. **`chr(0)`** 是 NULL，repr 显示 `\x00`、打印不可见、但作为字符真实存在。
3. **偏好 UTF-8**：变长、ASCII 省字节、无 BOM/字节序噪声、网络标准。
4. **逐字节 decode 是错的**：多字节字符（如 `牛`=3 字节）会解码失败；`b'\xff\xff'` 这类是永不合法的字节序列。
5. **BPE 训练瓶颈在预分词**（全语料正则扫描），故并行化；合并不跨预 token 边界。
6. **压缩率 = 字节/token**，越大越好；分词器有强域依赖，跨域（TinyStories 分词器编 OWT）压缩率下降。
7. **大规模编码需并行/系统语言**：Pile 825GB 单进程要数天。

---

## 参考

- handout：[00_01_cs336_assignment1_basics_extracted.md](../00_01_cs336_assignment1_basics_extracted.md) §2（`unicode1`、`unicode2`、`train_bpe_tinystories`、`train_bpe_expts_owt`、`tokenizer_experiments`）
- 相关实现（待完成）：[tests/adapters.py](../../tests/adapters.py) 的 `run_train_bpe` / `get_tokenizer`
- GPT-2 预分词正则：`openai/tiktoken#234`
- 数据集背景：TinyStories（合成幼儿故事）vs OpenWebText（真实网页）——见此前对话中的对比
