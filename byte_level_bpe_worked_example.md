# Byte-level BPE 全流程：一个可手工复算的完整例子

## 1. 目标与范围

本文通过一个规模较小但结构完整的语料，逐步推导 byte-level BPE 的训练、编码与解码过程。示例刻意同时包含：

- 重复出现的英文词；
- 一个由多个 UTF-8 字节构成的中文字符；
- 需要原样保留的空格；
- 表示文档边界的 special token；
- pair 频率并列时的确定性 tie-break；
- 不能单独解码为 Unicode 字符的中间 byte token；
- 训练语料中未出现过的字符。

全文不依赖程序执行，每个计数和 merge 均可手工复算。

## 2. 问题设定

### 2.1 两个训练文档

定义两个文档：

| 文档 | 原始文本 |
|---|---|
| $D_1$ | `low low lower 牛` |
| $D_2$ | `low lower 牛 牛` |

使用 special token `<|endoftext|>` 分隔文档，则序列化语料为：

```text
low low lower 牛<|endoftext|>low lower 牛 牛
```

这里的 `<|endoftext|>` 具有控制语义，不是普通文本。训练时，它满足以下约束：

1. 作为一个完整 token 加入词表；
2. 不参与普通 BPE pair 频率统计；
3. 左右两侧构成硬边界；
4. 任何 merge 都不能跨越该边界。

### 2.2 目标词表大小

设目标词表大小为：

$$ V_{\text{target}}=263. $$

Byte-level BPE 的基础词表包含 256 个单字节 token，再加入一个 special token，因此 merge 开始前的词表大小为：

$$ V_0=256+1=257. $$

每轮 merge 恰好产生一个新 token，所以需要执行：

$$ M=V_{\text{target}}-V_0=263-257=6 $$

轮 merge。

### 2.3 为手工推导定义的 pre-tokenizer

为了把重点集中在 byte-level BPE 的统计和 merge 过程上，本文采用一个简化但可逆的 pre-tokenizer：

1. 连续英文字母构成一个 pre-token；
2. 每个汉字构成一个 pre-token；
3. 连续空格作为独立 pre-token；
4. `<|endoftext|>` 在 pre-tokenization 之前被识别并隔离；
5. BPE merge 不能跨越 pre-token 边界。

因此，空格不会被丢弃，而是作为字节 `0x20` 保留。这个简化规则与 CS336 作业使用的 GPT-2 风格正则不完全相同；作业正则通常会把部分前导空格附着到后续文本。两者只会改变 pre-token 及频率统计，不会改变后续 BPE 算法的基本原理。

## 3. 阶段一：文档分段与 pre-token 计数

### 3.1 各文档的 pre-token 序列

$D_1$ 被切分为：

$$ P(D_1)=(\texttt{low},\texttt{ },\texttt{low},\texttt{ },\texttt{lower},\texttt{ },\texttt{牛}). $$

$D_2$ 被切分为：

$$ P(D_2)=(\texttt{low},\texttt{ },\texttt{lower},\texttt{ },\texttt{牛},\texttt{ },\texttt{牛}). $$

其中，$\texttt{ }$ 表示一个空格 pre-token。

### 3.2 频率表

把两个文档的 pre-token 计数相加：

| Pre-token $p$ | 出现次数 $c(p)$ |
|---|---:|
| `low` | 3 |
| `lower` | 2 |
| `牛` | 3 |
| 空格 ` ` | 6 |

Special token `<|endoftext|>` 不进入该频率表，因为它不参与 merge 统计。

这一频率表是 pre-tokenization 的关键收益。训练算法不需要分别保存 3 份 `low` 和 3 份 `牛`；只需保存每种 pre-token 的当前表示及其出现次数。

## 4. 阶段二：转换为 UTF-8 字节

### 4.1 ASCII pre-token

英文字符和空格属于 ASCII，其 UTF-8 编码均为单字节：

| 字符 | 十六进制字节 | 十进制字节 |
|---|---:|---:|
| 空格 | `20` | 32 |
| `e` | `65` | 101 |
| `l` | `6C` | 108 |
| `o` | `6F` | 111 |
| `r` | `72` | 114 |
| `w` | `77` | 119 |

因此：

$$ \operatorname{UTF8}(\texttt{low})=(\texttt{6C},\texttt{6F},\texttt{77}), $$

$$ \operatorname{UTF8}(\texttt{lower})=(\texttt{6C},\texttt{6F},\texttt{77},\texttt{65},\texttt{72}). $$

### 4.2 中文 pre-token

字符 `牛` 的 Unicode code point 是 U+725B，其 UTF-8 编码为：

$$ \operatorname{UTF8}(\texttt{牛})=(\texttt{E7},\texttt{89},\texttt{9B}). $$

对应十进制字节序列：

$$ (231,137,155). $$

需要强调：`牛` 是一个 Unicode code point，但在 byte-level BPE 的初始状态中，它由三个独立的 byte token 表示。

### 4.3 初始 pre-token 状态

设 $S_0(p)$ 表示 pre-token $p$ 在任何 merge 发生之前的 token 序列：

| Pre-token $p$ | 频率 $c(p)$ | 初始状态 $S_0(p)$ |
|---|---:|---|
| `low` | 3 | `[6C, 6F, 77]` |
| `lower` | 2 | `[6C, 6F, 77, 65, 72]` |
| `牛` | 3 | `[E7, 89, 9B]` |
| 空格 ` ` | 6 | `[20]` |

空格 pre-token 的长度为 1，因此它内部没有相邻 pair，不会对 pair 频率产生贡献。

## 5. 阶段三：初始化词表

本文采用最直接的 ID 约定：

$$ \operatorname{id}(b)=b,\qquad b\in\{0,\ldots,255\}. $$

也就是说，单字节 token 的 ID 等于该字节的无符号整数值。例如：

| Token bytes | Token ID |
|---|---:|
| `20` | 32 |
| `65` | 101 |
| `6C` | 108 |
| `6F` | 111 |
| `72` | 114 |
| `77` | 119 |
| `89` | 137 |
| `9B` | 155 |
| `E7` | 231 |

再令：

$$ \operatorname{id}(\texttt{<|endoftext|>})=256. $$

初始词表为：

$$ \mathcal{V}_0=\{[00],[01],\ldots,[\mathrm{FF}]\}\cup\{\texttt{<|endoftext|>}\}. $$

其中，方括号表示一个字节串 token，而不是 Unicode 字符。

## 6. 阶段四：定义 pair 频率与 tie-break

### 6.1 Pair 频率

设 $\operatorname{occ}_{S_r(p)}(a,b)$ 表示 pair $(a,b)$ 在第 $r$ 轮开始时的 pre-token 序列 $S_r(p)$ 中出现的次数，则全局频率为：

$$ f_r(a,b)=\sum_{p}c(p)\operatorname{occ}_{S_r(p)}(a,b). $$

算法在第 $r$ 轮选择：

$$ (a_r,b_r)=\arg\max_{(a,b)} f_r(a,b). $$

随后把每个 pre-token 内所有不重叠的相邻 $(a_r,b_r)$ 替换成新 token $a_rb_r$。

### 6.2 确定性 tie-break

本文遵循 CS336 作业规则：如果多个 pair 具有相同最高频率，则选择字节串 pair 中字典序最大的一个。

对于两个 pair：

$$ (a,b)\quad\text{与}\quad(c,d), $$

先比较 $a$ 与 $c$ 的字节字典序；只有当二者相同时，才比较 $b$ 与 $d$。

Tie-break 不影响 BPE 的基本可逆性，但会改变 merge 顺序、最终词表和 token ID，因此必须固定。

## 7. 阶段五：逐轮训练 BPE

为消除“当前状态”“当前统计”和“本轮产物”之间的歧义，以下每轮均严格采用同一顺序：

1. 读取上一轮输出状态 $S_{k-1}(p)$ 及固定乘数 $c(p)$；
2. 计算加权 pair 频率 $f_{k-1}(a,b)$；
3. 选择最高频 pair，并在并列时执行 tie-break；
4. 创建新 token、分配 ID，并立即写入 merge 列表；
5. 在所有 pre-token 中执行该 merge，得到新状态 $S_k(p)$。

第 $k$ 轮产生的 merge rank 为 $k-1$。Pre-token 频率 $c(p)$ 在所有轮次中保持不变；发生变化的只有其内部 token 序列 $S_k(p)$。

### 7.1 第 1 轮：由 $S_0$ 生成 token `ow`

#### 7.1.1 输入状态

| Pre-token $p$ | 固定乘数 $c(p)$ | 第 1 轮输入 $S_0(p)$ |
|---|---:|---|
| `low` | 3 | `[6C, 6F, 77]` |
| `lower` | 2 | `[6C, 6F, 77, 65, 72]` |
| `牛` | 3 | `[E7, 89, 9B]` |
| 空格 | 6 | `[20]` |

#### 7.1.2 加权 pair 统计

每个单元格均表示“单个 pre-token 内的 pair 出现次数 $\times$ 该 pre-token 的固定乘数”：

| Pair | 来自 `low`，$c=3$ | 来自 `lower`，$c=2$ | 来自 `牛`，$c=3$ | 全局频率 $f_0$ |
|---|---:|---:|---:|---:|
| `(6C, 6F)`，即 `(l, o)` | $1\times3=3$ | $1\times2=2$ | 0 | 5 |
| `(6F, 77)`，即 `(o, w)` | $1\times3=3$ | $1\times2=2$ | 0 | 5 |
| `(77, 65)`，即 `(w, e)` | 0 | $1\times2=2$ | 0 | 2 |
| `(65, 72)`，即 `(e, r)` | 0 | $1\times2=2$ | 0 | 2 |
| `(E7, 89)` | 0 | 0 | $1\times3=3$ | 3 |
| `(89, 9B)` | 0 | 0 | $1\times3=3$ | 3 |

空格状态 `[20]` 长度为 1，不包含 pair，因此即使 $c(\texttt{空格})=6$，其贡献仍为 $0\times6=0$。

#### 7.1.3 选择、登记并执行 merge

最高频率为 5，并列 pair 是 `(l,o)` 与 `(o,w)`。由于 `o` 的字节 `6F` 在字典序上大于 `l` 的字节 `6C`，选择：

$$ m_0=(\texttt{6F},\texttt{77}). $$

创建新 token：

$$ \texttt{6F}\Vert\texttt{77}=\texttt{6F77}=\texttt{ow},\qquad \operatorname{id}(\texttt{ow})=257. $$

在修改 pre-token 状态之前，先把本轮训练产物写入 merge 列表：

$$ \mathcal{M}^{(1)}=[m_0]=[(\texttt{6F},\texttt{77})]. $$

随后，在所有 pre-token 内把不重叠的 `(6F, 77)` 替换为 `ow`。

#### 7.1.4 输出状态

| Pre-token $p$ | 固定乘数 $c(p)$ | 第 1 轮输出 $S_1(p)$ |
|---|---:|---|
| `low` | 3 | `[6C, ow]` |
| `lower` | 2 | `[6C, ow, 65, 72]` |
| `牛` | 3 | `[E7, 89, 9B]` |
| 空格 | 6 | `[20]` |

本轮至此结束。第 2 轮只能以该表中的 $S_1(p)$ 为输入重新统计 pair。

### 7.2 第 2 轮：由 $S_1$ 生成 token `low`

#### 7.2.1 输入状态与加权统计

第 2 轮输入就是上一轮输出 $S_1$。逐个 pre-token 展开：

| 来源 pre-token | 当前局部 pair | 单个 pre-token 内出现次数 | 固定乘数 $c(p)$ | 全局贡献 |
|---|---|---:|---:|---:|
| `low` | `(6C, ow)` | 1 | 3 | $1\times3=3$ |
| `lower` | `(6C, ow)` | 1 | 2 | $1\times2=2$ |
| `lower` | `(ow, 65)` | 1 | 2 | $1\times2=2$ |
| `lower` | `(65, 72)` | 1 | 2 | $1\times2=2$ |
| `牛` | `(E7, 89)` | 1 | 3 | $1\times3=3$ |
| `牛` | `(89, 9B)` | 1 | 3 | $1\times3=3$ |
| 空格 | 无 pair | 0 | 6 | $0\times6=0$ |

聚合相同 pair：

| Pair | 全局频率 $f_1$ |
|---|---:|
| `(6C, ow)` | $3+2=5$ |
| `(ow, 65)` | 2 |
| `(65, 72)` | 2 |
| `(E7, 89)` | 3 |
| `(89, 9B)` | 3 |

#### 7.2.2 选择、登记并执行 merge

唯一最高频 pair 是 `(6C,ow)`：

$$ m_1=(\texttt{6C},\texttt{ow}). $$

创建新 token 并分配 ID：

$$ \texttt{6C}\Vert\texttt{ow}=\texttt{low},\qquad \operatorname{id}(\texttt{low})=258. $$

立即追加 merge 记录：

$$ \mathcal{M}^{(2)}=[m_0,m_1]. $$

#### 7.2.3 输出状态

| Pre-token $p$ | 固定乘数 $c(p)$ | 第 2 轮输出 $S_2(p)$ |
|---|---:|---|
| `low` | 3 | `[low]` |
| `lower` | 2 | `[low, 65, 72]` |
| `牛` | 3 | `[E7, 89, 9B]` |
| 空格 | 6 | `[20]` |

此时，`low` 已成为单个 token，不再贡献内部 pair。

### 7.3 第 3 轮：由 $S_2$ 生成中间 byte token `[E7 89]`

#### 7.3.1 加权 pair 统计

| Pair | 来源与乘法 | 全局频率 $f_2$ |
|---|---|---:|
| `(low, 65)` | `lower`：$1\times2$ | 2 |
| `(65, 72)` | `lower`：$1\times2$ | 2 |
| `(E7, 89)` | `牛`：$1\times3$ | 3 |
| `(89, 9B)` | `牛`：$1\times3$ | 3 |

`low` 与空格在 $S_2$ 中均为单 token 状态，不再贡献 pair；`lower` 和 `牛` 产生的全部 pair 已在表中列出。

#### 7.3.2 选择、登记并执行 merge

最高频率为 3，`(E7,89)` 与 `(89,9B)` 并列。比较第一个 token 时，`E7` 大于 `89`，因此选择：

$$ m_2=(\texttt{E7},\texttt{89}). $$

创建新 token 并分配 ID：

$$ \texttt{E7}\Vert\texttt{89}=[\texttt{E7 89}],\qquad \operatorname{id}([\texttt{E7 89}])=259. $$

立即追加 merge 记录：

$$ \mathcal{M}^{(3)}=[m_0,m_1,m_2]. $$

#### 7.3.3 输出状态

| Pre-token $p$ | 固定乘数 $c(p)$ | 第 3 轮输出 $S_3(p)$ |
|---|---:|---|
| `low` | 3 | `[low]` |
| `lower` | 2 | `[low, 65, 72]` |
| `牛` | 3 | `[E789, 9B]` |
| 空格 | 6 | `[20]` |

Token `[E7 89]` 是合法的 byte-level BPE token，但不是完整 UTF-8 字符。单独解码这两个字节会得到“不完整字节序列”错误；byte-level BPE 只要求 token 是字节串，不要求每个 token 都能独立解码成 Unicode。

### 7.4 第 4 轮：由 $S_3$ 生成完整 token `牛`

#### 7.4.1 加权 pair 统计

| Pair | 来源与乘法 | 全局频率 $f_3$ |
|---|---|---:|
| `(low, 65)` | `lower`：$1\times2$ | 2 |
| `(65, 72)` | `lower`：$1\times2$ | 2 |
| `(E789, 9B)` | `牛`：$1\times3$ | 3 |

#### 7.4.2 选择、登记并执行 merge

选择唯一最高频 pair：

$$ m_3=(\texttt{E789},\texttt{9B}). $$

创建新 token：

$$ \texttt{E789}\Vert\texttt{9B}=\texttt{E7899B}=\texttt{牛},\qquad \operatorname{id}(\texttt{牛})=260. $$

立即追加 merge 记录：

$$ \mathcal{M}^{(4)}=[m_0,m_1,m_2,m_3]. $$

#### 7.4.3 输出状态

| Pre-token $p$ | 固定乘数 $c(p)$ | 第 4 轮输出 $S_4(p)$ |
|---|---:|---|
| `low` | 3 | `[low]` |
| `lower` | 2 | `[low, 65, 72]` |
| `牛` | 3 | `[牛]` |
| 空格 | 6 | `[20]` |

### 7.5 第 5 轮：由 $S_4$ 生成 token `lowe`

#### 7.5.1 加权 pair 统计

此时只有 `lower` 仍包含两个以上 token：

| Pair | 来源与乘法 | 全局频率 $f_4$ |
|---|---|---:|
| `(low, 65)`，即 `(low, e)` | `lower`：$1\times2$ | 2 |
| `(65, 72)`，即 `(e, r)` | `lower`：$1\times2$ | 2 |

#### 7.5.2 选择、登记并执行 merge

两个 pair 频率相同。比较第一个 token 的字节串时，`low` 的首字节 `6C` 大于 `e` 的字节 `65`，因此选择：

$$ m_4=(\texttt{low},\texttt{65}). $$

创建新 token 并分配 ID：

$$ \texttt{low}\Vert\texttt{65}=\texttt{lowe},\qquad \operatorname{id}(\texttt{lowe})=261. $$

立即追加 merge 记录：

$$ \mathcal{M}^{(5)}=[m_0,m_1,m_2,m_3,m_4]. $$

#### 7.5.3 输出状态

| Pre-token $p$ | 固定乘数 $c(p)$ | 第 5 轮输出 $S_5(p)$ |
|---|---:|---|
| `low` | 3 | `[low]` |
| `lower` | 2 | `[lowe, 72]` |
| `牛` | 3 | `[牛]` |
| 空格 | 6 | `[20]` |

### 7.6 第 6 轮：由 $S_5$ 生成 token `lower`

#### 7.6.1 加权 pair 统计

唯一剩余 pair 是：

$$ (\texttt{lowe},\texttt{72}),\qquad f_5(\texttt{lowe},\texttt{72})=1\times c(\texttt{lower})=1\times2=2. $$

#### 7.6.2 选择、登记并执行 merge

选择：

$$ m_5=(\texttt{lowe},\texttt{72}). $$

创建新 token 并分配 ID：

$$ \texttt{lowe}\Vert\texttt{72}=\texttt{lower},\qquad \operatorname{id}(\texttt{lower})=262. $$

立即追加 merge 记录：

$$ \mathcal{M}^{(6)}=[m_0,m_1,m_2,m_3,m_4,m_5]. $$

#### 7.6.3 最终状态

| Pre-token $p$ | 固定乘数 $c(p)$ | 第 6 轮输出 $S_6(p)$ |
|---|---:|---|
| `low` | 3 | `[low]` |
| `lower` | 2 | `[lower]` |
| `牛` | 3 | `[牛]` |
| 空格 | 6 | `[20]` |

已经完成 6 轮 merge，词表大小达到 263，训练结束。

## 8. 训练产物

### 8.1 Merge 列表

Merge 列表必须按创建顺序保存：

| Merge rank | 左 token | 右 token | 新 token | 新 ID |
|---:|---|---|---|---:|
| 0 | `6F` (`o`) | `77` (`w`) | `ow` | 257 |
| 1 | `6C` (`l`) | `ow` | `low` | 258 |
| 2 | `E7` | `89` | `E7 89` | 259 |
| 3 | `E7 89` | `9B` | `牛` | 260 |
| 4 | `low` | `65` (`e`) | `lowe` | 261 |
| 5 | `lowe` | `72` (`r`) | `lower` | 262 |

记为：

$$ \mathcal{M}=\big[(\texttt{o},\texttt{w}),(\texttt{l},\texttt{ow}),(\texttt{E7},\texttt{89}),(\texttt{E789},\texttt{9B}),(\texttt{low},\texttt{e}),(\texttt{lowe},\texttt{r})\big]. $$

### 8.2 新增词表项

最终词表为：

$$ \mathcal{V}=\mathcal{V}_0\cup\{\texttt{ow},\texttt{low},[\texttt{E7 89}],\texttt{牛},\texttt{lowe},\texttt{lower}\}. $$

需要同时保存 `vocab` 和 `merges`：

- `vocab` 决定每个 token 字节串对应哪个整数 ID；
- `merges` 决定编码时哪些相邻 token 可以合并，以及合并优先级。

只有词表而没有 merge rank，不能唯一复现训练时定义的编码过程。

## 9. 阶段六：用训练好的 BPE 编码新文本

考虑训练结束后输入：

```text
lower 牛 low
```

### 9.1 Pre-tokenization

按照同一 pre-tokenizer，得到：

$$ (\texttt{lower},\texttt{ },\texttt{牛},\texttt{ },\texttt{low}). $$

### 9.2 转换为基础字节 token

| Pre-token | 初始字节 token |
|---|---|
| `lower` | `[6C, 6F, 77, 65, 72]` |
| 空格 | `[20]` |
| `牛` | `[E7, 89, 9B]` |
| 空格 | `[20]` |
| `low` | `[6C, 6F, 77]` |

### 9.3 按 merge rank 重放

编码阶段不重新统计输入中的 pair 频率，而是严格按照训练产物中的 merge rank 重放。

这里使用 $m_i$ 表示训练阶段学到的第 $i$ 条 merge 规则：

| Merge rank | 规则 |
|---:|---|
| $m_0$ | `(o, w) → ow` |
| $m_1$ | `(l, ow) → low` |
| $m_2$ | `(E7, 89) → E789` |
| $m_3$ | `(E789, 9B) → 牛` |
| $m_4$ | `(low, e) → lowe` |
| $m_5$ | `(lowe, r) → lower` |

记号 $X\xrightarrow{m_i}Y$ 表示：规则 $m_i$ 的左右 token 在当前序列 $X$ 中相邻出现，因此执行替换后得到 $Y$。如果规则 $m_i$ 在当前序列中没有匹配项，则序列保持不变。

#### 9.3.1 展开编码 `lower`

`lower` 的 UTF-8 字节都是 ASCII 字节，因此初始 token 序列可以写成：

$$ [l,o,w,e,r]=[\texttt{6C},\texttt{6F},\texttt{77},\texttt{65},\texttt{72}]. $$

依次检查全部 6 条 merge 规则：

| 检查顺序 | Merge 规则 | 检查前状态 | 是否匹配 | 检查后状态 |
|---:|---|---|---|---|
| 1 | $m_0:(o,w)\rightarrow ow$ | `[l, o, w, e, r]` | 是，`o` 与 `w` 相邻 | `[l, ow, e, r]` |
| 2 | $m_1:(l,ow)\rightarrow low$ | `[l, ow, e, r]` | 是，`l` 与 `ow` 相邻 | `[low, e, r]` |
| 3 | $m_2:(E7,89)\rightarrow E789$ | `[low, e, r]` | 否，不含 `E7`、`89` | `[low, e, r]` |
| 4 | $m_3:(E789,9B)\rightarrow 牛$ | `[low, e, r]` | 否，不含 `E789`、`9B` | `[low, e, r]` |
| 5 | $m_4:(low,e)\rightarrow lowe$ | `[low, e, r]` | 是，`low` 与 `e` 相邻 | `[lowe, r]` |
| 6 | $m_5:(lowe,r)\rightarrow lower$ | `[lowe, r]` | 是，`lowe` 与 `r` 相邻 | `[lower]` |

因此，完整变化过程是：

$$ [l,o,w,e,r]\xrightarrow{m_0}[l,ow,e,r]\xrightarrow{m_1}[low,e,r]\xrightarrow{m_4}[lowe,r]\xrightarrow{m_5}[lower]. $$

这个紧凑公式只画出实际改变序列的 merge，所以没有显示 $m_2$ 和 $m_3$。它们并非没有被检查，而是对 `[low,e,r]` 不匹配，因此属于 no-op。

最终 `[lower]` 是长度为 1 的 token 序列，其中 `lower` 对应词表 ID 262。

#### 9.3.2 编码 `牛`

`牛` 的初始状态是 `[E7,89,9B]`。规则 $m_0$ 和 $m_1$ 不匹配；$m_2$ 与 $m_3$ 依次匹配：

$$ [E7,89,9B]\xrightarrow{m_2}[E789,9B]\xrightarrow{m_3}[E7899B]. $$

得到单个 token `[牛]`，对应词表 ID 260。后续 $m_4$ 和 $m_5$ 不匹配。

#### 9.3.3 编码 `low`

`low` 的初始状态是 `[l,o,w]`。前两条规则依次匹配：

$$ [l,o,w]\xrightarrow{m_0}[l,ow]\xrightarrow{m_1}[low]. $$

得到单个 token `[low]`，对应词表 ID 258。规则 $m_2$ 至 $m_5$ 均不匹配。

空格 pre-token `[20]` 不发生 merge。

### 9.4 映射到 ID

最终 token 序列为：

$$ [\texttt{lower},\texttt{ },\texttt{牛},\texttt{ },\texttt{low}]. $$

对应 ID：

$$ [262,32,260,32,258]. $$

这说明同一个最终 token 可能来自不同数量的原始字节：

| 最终 token | 原始字节数 |
|---|---:|
| `lower` | 5 |
| 空格 | 1 |
| `牛` | 3 |
| `low` | 3 |

## 10. 阶段七：解码

给定 ID 序列：

$$ [262,32,260,32,258], $$

首先查词表：

| ID | Token bytes |
|---:|---|
| 262 | `6C 6F 77 65 72` |
| 32 | `20` |
| 260 | `E7 89 9B` |
| 32 | `20` |
| 258 | `6C 6F 77` |

把所有 token bytes 拼接：

$$ \texttt{6C 6F 77 65 72 20 E7 89 9B 20 6C 6F 77}. $$

最后对完整字节流执行一次 UTF-8 解码：

```text
lower 牛 low
```

于是：

$$ \operatorname{decode}([262,32,260,32,258])=\texttt{lower 牛 low}. $$

### 10.1 为什么不能逐 token 解码

ID 259 对应字节串 `E7 89`。该字节串只是 `牛` 的 UTF-8 前缀，不能单独解码为完整字符。

但是：

$$ \operatorname{bytes}(259)\Vert\operatorname{bytes}(155)=\texttt{E7 89}\Vert\texttt{9B}=\texttt{E7 89 9B}, $$

拼接后可以正确解码为 `牛`。

因此，正确顺序必须是：

1. ID 查表得到字节串；
2. 拼接所有字节串；
3. 对完整字节流执行一次 UTF-8 decode。

## 11. Special token 如何进入最终序列

训练语料：

```text
low low lower 牛<|endoftext|>low lower 牛 牛
```

编码后为：

$$ [258,32,258,32,262,32,260,\boxed{256},258,32,262,32,260,32,260]. $$

其中，ID 256 是完整的 `<|endoftext|>`，不会被拆成 `<`、`|`、`end` 等普通 token。

文档边界左右也不会产生 merge。例如，不会学习：

$$ (\texttt{牛},\texttt{low}) $$

这一跨文档 pair。

## 12. 对训练语料的压缩效果

忽略 special token，仅统计两个文档的普通文本。

$D_1$ 的 UTF-8 字节数为：

$$ 3+1+3+1+5+1+3=17. $$

$D_2$ 的 UTF-8 字节数为：

$$ 3+1+5+1+3+1+3=17. $$

总字节数：

$$ B=17+17=34. $$

在任何 merge 发生之前，纯字节表示需要 34 个 token。训练完成后：

- $D_1$ 编码为 7 个普通 token；
- $D_2$ 编码为 7 个普通 token；
- 总计 $N=14$ 个普通 token。

因此压缩率为：

$$ R_{\text{bytes/token}}=\frac{B}{N}=\frac{34}{14}\approx2.43. $$

相对于纯字节序列，token 数减少比例为：

$$ 1-\frac{14}{34}\approx58.8\%. $$

该压缩来自高频片段 `low`、`lower` 和 `牛` 被提升为独立 token。

## 13. 对未见字符串的编码

### 13.1 已知前缀加未见后缀

考虑训练语料中未出现的 `lows`：

$$ [l,o,w,s]\xrightarrow{r=0}[l,ow,s]\xrightarrow{r=1}[low,s]. $$

因此：

$$ \operatorname{encode}(\texttt{lows})=[258,115]. $$

虽然完整单词 `lows` 未在训练语料中出现，但已学习的高频前缀 `low` 仍可复用。

### 13.2 完全未见的 Unicode 字符

考虑训练语料中从未出现的汉字 `新`。其 UTF-8 编码是：

$$ \operatorname{UTF8}(\texttt{新})=(\texttt{E6},\texttt{96},\texttt{B0}). $$

本例的 merge 列表没有任何规则适用于这些 pair，因此：

$$ \operatorname{encode}(\texttt{新})=[230,150,176]. $$

仍然可以无损解码，因为 256 个基础 byte token 覆盖所有可能字节。这正是 byte-level BPE 不需要 `<unk>` 的原因。

## 14. 训练阶段与编码阶段的根本区别

| 阶段 | Pair 的选择依据 | 是否更新词表 | 输出 |
|---|---|---|---|
| BPE 训练 | 整个训练语料中的加权 pair 频率 | 是 | `vocab` 与有序 `merges` |
| 文本编码 | 训练得到的固定 merge rank | 否 | token ID 序列 |

训练阶段计算：

$$ \arg\max_{(a,b)}f_r(a,b). $$

编码阶段不计算新的 $\arg\max$。它只判断当前相邻 pair 是否存在于 merge 表，并按照既定 rank 应用。

如果编码时重新根据输入频率选择 pair，则同一字符串可能因所在 batch 或上下文不同而得到不同 tokenization，模型输入接口将失去确定性。

## 15. 全流程汇总

| 阶段 | 本例中的结果 |
|---|---|
| 原始文档 | `low low lower 牛` 与 `low lower 牛 牛` |
| Special token | `<|endoftext|>`，ID 256 |
| Pre-token 频率 | `low: 3`，`lower: 2`，`牛: 3`，空格 `: 6` |
| 初始普通词表 | 256 个单字节 token |
| Merge 轮数 | 6 |
| 新 token | `ow`、`low`、`E7 89`、`牛`、`lowe`、`lower` |
| 最终词表大小 | 263 |
| 新文本 | `lower 牛 low` |
| 编码结果 | `[262, 32, 260, 32, 258]` |
| 解码结果 | `lower 牛 low` |
| 未见字符 `新` | 回退为 `[230, 150, 176]` |

## 16. 最核心的六条结论

1. Byte-level BPE 的基础符号是字节，不是 Unicode 字符。
2. Pre-tokenization 决定允许在哪些局部范围内统计和执行 merge。
3. 训练阶段按语料加权频率学习 merge，编码阶段只重放固定 merge rank。
4. 一个合法 token 可以是不完整的 UTF-8 字节片段，因此解码前必须先拼接全部 token bytes。
5. 高频字节串会逐步变成较长 token，从而缩短序列。
6. 任何未见字符串仍能回退到 256 个基础字节，不会产生 OOV。

## 17. Byte-level BPE 的数学形式化描述

### 17.1 输入对象

设训练语料由 $N$ 个 Unicode 文档组成：

$$ \mathcal{D}=(d_1,d_2,\ldots,d_N). $$

设 $\mathcal{S}$ 为 special token 集合，$\mathcal{P}_{\mathcal{S}}$ 为边界感知的 pre-tokenizer。$\mathcal{P}_{\mathcal{S}}$ 先识别 special token，把它们视为硬边界，再对各普通文本区间执行 pre-tokenization；special token 本身不进入普通 pre-token 频率统计。

整个语料产生的普通 pre-token 多重集为：

$$ \mathcal{C}=\biguplus_{n=1}^{N}\mathcal{P}_{\mathcal{S}}(d_n). $$

不同 pre-token 的集合是该多重集的支撑集：

$$ \mathcal{U}=\operatorname{supp}(\mathcal{C}). $$

对任意 $p\in\mathcal{U}$，定义其语料频率：

$$ c(p)=\operatorname{mult}_{\mathcal{C}}(p). $$

因此，训练阶段只需维护每种不同 pre-token 的当前状态和固定乘数 $c(p)$，而不必分别维护其每一次出现。

### 17.2 字节字母表、token 与初始词表

定义字节值域：

$$ \mathbb{B}=\{0,1,\ldots,255\}. $$

UTF-8 编码函数记为：

$$ \mathcal{E}:\operatorname{Unicode}^{*}\rightarrow\mathbb{B}^{*}. $$

对每个字节 $b\in\mathbb{B}$，定义一个原子 byte token $\langle b\rangle$。所有原子 byte token 构成：

$$ \mathcal{A}=\{\langle b\rangle:b\in\mathbb{B}\}. $$

对每个 special token $s\in\mathcal{S}$，定义一个不可拆分的原子 special token $\langle s\rangle_{\mathcal{S}}$。初始词表为：

$$ \mathcal{V}_0=\mathcal{A}\cup\{\langle s\rangle_{\mathcal{S}}:s\in\mathcal{S}\}. $$

若所有 special token 互不重复，且均不与单字节 token 重合，则：

$$ |\mathcal{V}_0|=256+|\mathcal{S}|. $$

定义 token 到底层字节串的映射 $\beta$：

$$ \beta(\langle b\rangle)=(b),\qquad \beta(\langle s\rangle_{\mathcal{S}})=\mathcal{E}(s). $$

若 token 序列为 $Z=(z_1,\ldots,z_\ell)$，定义其底层字节串拼接：

$$ \beta^{*}(Z)=\beta(z_1)\Vert\beta(z_2)\Vert\cdots\Vert\beta(z_\ell). $$

其中 $\Vert$ 表示字节串连接。

### 17.3 Pre-token 的初始状态

对任意普通 pre-token $p\in\mathcal{U}$，设：

$$ \mathcal{E}(p)=(b_1,b_2,\ldots,b_{\ell_p}). $$

其初始 token 状态定义为：

$$ Z_0(p)=(\langle b_1\rangle,\langle b_2\rangle,\ldots,\langle b_{\ell_p}\rangle). $$

由定义立即得到初始字节保持性质：

$$ \beta^{*}(Z_0(p))=\mathcal{E}(p). $$

### 17.4 相邻 pair 的加权频率

对 token 序列 $Z=(z_1,\ldots,z_\ell)$，定义 pair $(u,v)$ 的相邻出现次数：

$$ \operatorname{occ}_{Z}(u,v)=\sum_{j=1}^{\ell-1}\mathbf{1}[z_j=u\land z_{j+1}=v]. $$

这里的 $\mathbf{1}[\cdot]$ 是示性函数。第 $k$ 轮开始时，pair $(u,v)$ 的全局加权频率为：

$$ f_k(u,v)=\sum_{p\in\mathcal{U}}c(p)\operatorname{occ}_{Z_k(p)}(u,v). $$

第 $k$ 轮的候选 pair 集合为：

$$ \mathcal{Q}_k=\{(u,v):f_k(u,v)>0\}. $$

频率统计不会跨 pre-token 边界，也不会跨 special token 边界。因此，$f_k$ 只包含各 $Z_k(p)$ 内部的相邻关系。

### 17.5 Pair 选择与确定性 tie-break

若 $\mathcal{Q}_k\neq\varnothing$，先定义最高频候选集合：

$$ \mathcal{H}_k=\underset{(u,v)\in\mathcal{Q}_k}{\arg\max}\ f_k(u,v). $$

设 $\succ_{\mathrm{lex}}$ 为基于 token 底层字节串的字典序，并按 pair 的第一分量、第二分量依次比较。第 $k$ 条 merge 规则定义为：

$$ m_k=(u_k,v_k)=\max_{\succ_{\mathrm{lex}}}\mathcal{H}_k. $$

这一定义把“最高频优先”和“并列时选择字典序最大 pair”组合为一个确定性选择规则。

### 17.6 新 token、词表与 merge 列表更新

选中 $m_k=(u_k,v_k)$ 后，创建新 token $w_k$，其底层字节串为：

$$ \beta(w_k)=\beta(u_k)\Vert\beta(v_k). $$

词表更新为：

$$ \mathcal{V}_{k+1}=\mathcal{V}_k\cup\{w_k\}. $$

第 $k$ 个新 token 可分配下一个未使用 ID。若初始 token ID 占据 $0,\ldots,|\mathcal{V}_0|-1$，则一种自然分配是：

$$ \iota(w_k)=|\mathcal{V}_0|+k. $$

增量分配完成后，最终 token-ID 映射是一个双射：

$$ \iota:\mathcal{V}_K\xrightarrow{\sim}\{0,1,\ldots,|\mathcal{V}_K|-1\}. $$

前 $k+1$ 条 merge 构成有序 merge 列表：

$$ \mathcal{M}^{(k+1)}=(m_0,m_1,\ldots,m_k). $$

Merge rank 定义为：

$$ \rho(m_k)=k. $$

### 17.7 非重叠 merge 状态转移

定义 $\mathcal{R}_{(u,v)}$ 为从左到右执行的非重叠替换算子：它把 token 序列中每个尚未参与替换的相邻 $(u,v)$ 替换为新 token $uv$。

第 $k$ 轮对所有普通 pre-token 同步更新：

$$ Z_{k+1}(p)=\mathcal{R}_{m_k}(Z_k(p)),\qquad p\in\mathcal{U}. $$

Pre-token 频率 $c(p)$ 不随 merge 改变；发生变化的只有内部 token 序列 $Z_k(p)$。

设 $\operatorname{nocc}_{Z}(u,v)$ 为从左到右可执行的非重叠替换次数，则加权语料 token 长度为：

$$ L_k=\sum_{p\in\mathcal{U}}c(p)|Z_k(p)|. $$

一次 merge 后：

$$ L_{k+1}=L_k-\sum_{p\in\mathcal{U}}c(p)\operatorname{nocc}_{Z_k(p)}(u_k,v_k). $$

只要被选 pair 至少出现一次，就有：

$$ L_{k+1}<L_k. $$

### 17.8 终止条件与训练输出

给定满足 $V_{\mathrm{target}}\ge|\mathcal{V}_0|$ 的目标词表大小 $V_{\mathrm{target}}$，允许的最大 merge 数为：

$$ K_{\max}=V_{\mathrm{target}}-|\mathcal{V}_0|. $$

训练在以下任一条件成立时停止：

1. 已执行 $K_{\max}$ 次 merge；
2. 当前候选 pair 集合为空，即 $\mathcal{Q}_k=\varnothing$。

实际执行的 merge 数记为 $K$，满足：

$$ 0\le K\le K_{\max}. $$

训练输出为最终词表和有序 merge 列表：

$$ \operatorname{TrainBPE}(\mathcal{D},\mathcal{S},\mathcal{P}_{\mathcal{S}},V_{\mathrm{target}})=(\mathcal{V}_K,\mathcal{M}^{(K)},\iota). $$

### 17.9 编码的形式化定义

给定待编码 Unicode 字符串 $x$，边界感知分段器首先把它分成普通 pre-token 与原子 special token 的有序序列：

$$ \mathcal{G}(x)=(g_1,g_2,\ldots,g_q). $$

若 $g_j$ 是普通 pre-token，设 $\mathcal{E}(g_j)=(b_1,\ldots,b_\ell)$，定义：

$$ Y_0(g_j)=(\langle b_1\rangle,\ldots,\langle b_\ell\rangle). $$

然后按固定 merge rank 依次应用训练得到的规则：

$$ Y_{k+1}(g_j)=\mathcal{R}_{m_k}(Y_k(g_j)),\qquad k=0,1,\ldots,K-1. $$

若 $g_j$ 是 special token $s$，则始终保持原子状态：

$$ Y_K(g_j)=(\langle s\rangle_{\mathcal{S}}). $$

把所有分段的最终 token 序列按原顺序拼接：

$$ T(x)=Y_K(g_1)\Vert Y_K(g_2)\Vert\cdots\Vert Y_K(g_q). $$

若：

$$ T(x)=(t_1,t_2,\ldots,t_h), $$

则编码结果为：

$$ \operatorname{Encode}(x)=(\iota(t_1),\iota(t_2),\ldots,\iota(t_h)). $$

编码阶段不重新计算 $f_k$，也不重新执行 $\arg\max$；它只重放训练阶段已经固定的 $m_0,\ldots,m_{K-1}$。

### 17.10 解码的形式化定义

给定 token ID 序列：

$$ I=(i_1,i_2,\ldots,i_h), $$

先通过逆词表映射获得 token：

$$ t_j=\iota^{-1}(i_j). $$

这一步与 UTF-8 解码不是同一种操作。$i_j$ 只是词表中的整数索引，其数值本身没有固定的字节语义；$t_j$ 才是该索引指向的词表 token。两者之间的类型关系是：

$$ i_j\in\{0,\ldots,|\mathcal{V}_K|-1\}\xrightarrow{\ \iota^{-1}\ }t_j\in\mathcal{V}_K\xrightarrow{\ \beta\ }\beta(t_j)\in\mathbb{B}^{+}. $$

必须先执行 $\iota^{-1}$，原因包括：

1. Merge 生成的 token ID 可以大于 255，例如本例中 ID 262 对应 `lower`，262 不是合法单字节值；
2. Token ID 只是人为编号，即使把整个词表重新编号，只要 $\iota$ 与 $\iota^{-1}$ 同步更新，tokenizer 的文本语义仍不变；
3. 同一个 token 可能对应多个字节，例如 `lower` 对应 `6C 6F 77 65 72`，不能从整数 262 的数值直接推导出来。

例如：

| ID $i_j$ | 逆词表查询 $t_j=\iota^{-1}(i_j)$ | 底层字节串 $\beta(t_j)$ |
|---:|---|---|
| 262 | `lower` | `6C 6F 77 65 72` |
| 32 | 空格 token | `20` |
| 260 | `牛` | `E7 89 9B` |

因此，ID 序列 `[262, 32, 260]` 不能直接作为 UTF-8 字节 `[262, 32, 260]` 解码；它必须先查词表，变成 token 序列 `[lower, 空格, 牛]`，再转换为字节串。

再拼接所有 token 的底层字节串：

$$ B(I)=\beta(t_1)\Vert\beta(t_2)\Vert\cdots\Vert\beta(t_h). $$

这里还不能对每个 $\beta(t_j)$ 分别执行 UTF-8 解码，因为单个 byte-level token 不一定构成完整 Unicode 字符。例如，ID 259 对应 `E7 89`，只有与后续字节 `9B` 拼接为 `E7 89 9B` 后才能解码为 `牛`。

最后执行一次 UTF-8 解码：

$$ \operatorname{Decode}(I)=\mathcal{E}^{-1}(B(I)). $$

对任意由该 tokenizer 合法编码得到的 Unicode 字符串 $x$，有：

$$ \operatorname{Decode}(\operatorname{Encode}(x))=x. $$

若输入 ID 序列不是合法编码结果，$B(I)$ 可能不是合法 UTF-8 字节流。此时，解码器必须显式选择 strict error 或 replacement character 策略。

### 17.11 正确性不变量

#### 17.11.1 字节保持不变量

对任意 $p\in\mathcal{U}$ 和任意训练轮次 $k$：

$$ \beta^{*}(Z_k(p))=\mathcal{E}(p). $$

证明依据是每次 merge 仅用字节串拼接 $\beta(u_k)\Vert\beta(v_k)$ 替换原来的两个相邻字节串，不会增加、删除或重排字节。

#### 17.11.2 边界不变量

任何 $\mathcal{R}_{m_k}$ 都只作用于单个 $Z_k(p)$ 内部，因此不存在：

- 跨 pre-token merge；
- 跨文档 merge；
- 跨 special token merge。

#### 17.11.3 词表闭包不变量

每个普通 token 的底层表示都是有限字节串：

$$ \forall t\in\mathcal{V}_k\setminus\{\langle s\rangle_{\mathcal{S}}:s\in\mathcal{S}\},\qquad \beta(t)\in\mathbb{B}^{+}. $$

#### 17.11.4 无 OOV 性质

由于 $\mathcal{A}$ 包含全部 256 个单字节 token，任意 UTF-8 字节串都可由 $\mathcal{A}$ 中元素组成。因此，对任意 Unicode 字符串 $x$：

$$ \operatorname{Encode}(x)\ \text{必然存在}. $$

#### 17.11.5 确定性

若以下对象固定：

- 语料 $\mathcal{D}$；
- special token 集合 $\mathcal{S}$；
- pre-tokenizer $\mathcal{P}_{\mathcal{S}}$；
- UTF-8 编码 $\mathcal{E}$；
- tie-break 顺序 $\succ_{\mathrm{lex}}$；
- 目标词表大小 $V_{\mathrm{target}}$；

则每轮 $m_k$、最终词表 $\mathcal{V}_K$、merge 列表 $\mathcal{M}^{(K)}$ 和任意输入的编码结果均唯一确定。

### 17.12 朴素复杂度

定义第 $k$ 轮所有不同 pre-token 状态的未加权总长度：

$$ \widetilde{L}_k=\sum_{p\in\mathcal{U}}|Z_k(p)|. $$

若每轮都完整扫描所有不同 pre-token 来重新统计 pair，则第 $k$ 轮时间复杂度为：

$$ O(\widetilde{L}_k). $$

执行 $K$ 轮的总时间复杂度为：

$$ O\left(\sum_{k=0}^{K-1}\widetilde{L}_k\right)\subseteq O(K\widetilde{L}_0). $$

编码一个初始长度为 $n_{\mathrm{enc}}$ 的 pre-token 时，若对每条 merge 都完整扫描当前序列，朴素上界为：

$$ O(Kn_{\mathrm{enc}}). $$

增量 pair 索引、优先队列、邻接链表和 pre-token 编码缓存可以降低实际开销，但不改变上述训练与编码的数学定义。

## 18. 符号表

| 符号 | 类型或定义域 | 含义 |
|---|---|---|
| $\mathcal{D}$ | Unicode 文档序列 | 完整训练语料 |
| $N$ | 非负整数 | 训练文档数量 |
| $d_n$ | Unicode 字符串 | 第 $n$ 个训练文档 |
| $n$ | $\{1,\ldots,N\}$ 中的整数 | 文档索引 |
| $k$ | 非负整数 | Merge 轮次或 merge rank 索引 |
| $j$ | 非负整数 | Token、字节或分段在序列中的位置索引 |
| $\ell,\ell_p$ | 非负整数 | 一般序列长度，以及 pre-token $p$ 的 UTF-8 字节长度 |
| $q$ | 非负整数 | 输入 $x$ 经边界感知分段后的分段数 |
| $h$ | 非负整数 | 编码结果或待解码 ID 序列中的 token 数 |
| $n_{\mathrm{enc}}$ | 非负整数 | 待编码 pre-token 的初始 byte token 数 |
| $\operatorname{Unicode}^{*}$ | Unicode 字符串集合 | 所有有限 Unicode code point 序列 |
| $\mathcal{S}$ | 字符串集合 | Special token 集合 |
| $s$ | $\mathcal{S}$ 中元素 | 一个 special token |
| $\mathcal{P}_{\mathcal{S}}$ | 文本到 pre-token 多重集的映射 | 识别 special token 硬边界的 pre-tokenizer |
| $\biguplus$ | 多重集运算 | 多重集并集，保留元素重复次数 |
| $\mathcal{C}$ | Pre-token 多重集 | 整个语料产生的普通 pre-token 及其重复实例 |
| $\operatorname{supp}(\mathcal{C})$ | 集合 | 多重集 $\mathcal{C}$ 中出现过的不同元素 |
| $\mathcal{U}$ | Pre-token 集合 | 所有不同普通 pre-token 的集合 |
| $p$ | $\mathcal{U}$ 中元素 | 一个普通 pre-token |
| $c(p)$ | 正整数 | Pre-token $p$ 在语料中的固定出现次数 |
| $\operatorname{mult}_{\mathcal{C}}(p)$ | 非负整数 | $p$ 在多重集 $\mathcal{C}$ 中的重数 |
| $\mathbb{B}$ | $\{0,\ldots,255\}$ | 所有可能的字节值 |
| $\mathbb{B}^{*}$ | 有限字节串集合 | 包含空串的所有有限字节序列 |
| $\mathbb{B}^{+}$ | 非空有限字节串集合 | 所有至少包含一个字节的序列 |
| $\mathcal{E}$ | $\operatorname{Unicode}^{*}\to\mathbb{B}^{*}$ | UTF-8 编码函数 |
| $\mathcal{E}^{-1}$ | 合法 UTF-8 字节串到 Unicode 字符串 | UTF-8 解码函数 |
| $b,b_j$ | $\mathbb{B}$ 中元素 | 单个字节值 |
| $\langle b\rangle$ | 原子 token | 表示字节 $b$ 的基础 byte token |
| $\mathcal{A}$ | Token 集合 | 256 个基础 byte token 的集合 |
| $\langle s\rangle_{\mathcal{S}}$ | 原子 token | 不可被普通 BPE 拆分的 special token |
| $\mathcal{V}_k$ | Token 集合 | 完成 $k$ 轮 merge 后的词表 |
| $\mathcal{V}_0$ | Token 集合 | 基础 byte token 与 special token 构成的初始词表 |
| $V_{\mathrm{target}}$ | 正整数 | 目标最大词表大小 |
| $\beta(t)$ | 非空字节串 | Token $t$ 对应的底层字节串 |
| $\beta^{*}(Z)$ | 字节串 | Token 序列 $Z$ 中所有底层字节串的顺序拼接 |
| $\Vert$ | 连接运算 | 字节串或 token 序列的有序拼接 |
| $Z_k(p)$ | Token 序列 | 第 $k$ 轮开始或结束时 pre-token $p$ 的表示 |
| $Z_0(p)$ | 基础 byte token 序列 | $p$ 在任何 merge 之前的初始表示 |
| $\lvert Z\rvert$ | 非负整数 | Token 序列 $Z$ 的长度 |
| $u,v$ | Token | 一个候选相邻 pair 的左右 token |
| $\operatorname{occ}_{Z}(u,v)$ | 非负整数 | Pair $(u,v)$ 在 $Z$ 中的相邻出现次数 |
| $\operatorname{nocc}_{Z}(u,v)$ | 非负整数 | Pair $(u,v)$ 在左到右非重叠替换中的实际替换次数 |
| $\mathbf{1}[\cdot]$ | $\{0,1\}$ | 条件成立时为 1，否则为 0 的示性函数 |
| $f_k(u,v)$ | 非负整数 | 第 $k$ 轮 pair $(u,v)$ 的全局加权频率 |
| $\mathcal{Q}_k$ | Pair 集合 | 第 $k$ 轮所有正频率候选 pair |
| $\mathcal{H}_k$ | Pair 集合 | 第 $k$ 轮具有最高频率的候选 pair |
| $\succ_{\mathrm{lex}}$ | 全序关系 | 基于 token 底层字节串的 pair 字典序 |
| $m_k=(u_k,v_k)$ | Token pair | 第 $k$ 条被选中的 merge 规则 |
| $w_k$ | Token | 合并 $u_k$ 与 $v_k$ 得到的新 token |
| $\mathcal{R}_{m_k}$ | Token 序列到 token 序列的映射 | 从左到右执行规则 $m_k$ 的非重叠替换算子 |
| $\mathcal{M}^{(k)}$ | 有序 pair 序列 | 前 $k$ 条 merge 规则组成的列表 |
| $\rho(m_k)$ | 非负整数 | Merge 规则 $m_k$ 的 rank，等于 $k$ |
| $\iota$ | $\mathcal{V}_K$ 到连续整数集合的双射 | 词表中的 token ID 映射 |
| $\iota^{-1}$ | 整数到 token 的映射 | Token ID 的逆词表查询 |
| $L_k$ | 非负整数 | 按 $c(p)$ 加权的第 $k$ 轮语料 token 总长度 |
| $\widetilde{L}_k$ | 非负整数 | 所有不同 pre-token 状态的未加权总长度 |
| $K_{\max}$ | 非负整数 | 由目标词表大小允许的最大 merge 数 |
| $K$ | 非负整数 | 实际执行的 merge 数 |
| $\operatorname{TrainBPE}$ | 训练输入到三元组的映射 | 输出最终词表、merge 列表和 ID 映射的训练过程 |
| $x$ | Unicode 字符串 | 待编码的输入文本 |
| $\mathcal{G}(x)$ | 有序分段序列 | 输入 $x$ 经 special token 识别和 pre-tokenization 后的分段 |
| $g_j$ | 普通 pre-token 或原子 special token | $\mathcal{G}(x)$ 的第 $j$ 个分段 |
| $Y_k(g_j)$ | Token 序列 | 编码阶段对分段 $g_j$ 应用前 $k$ 条 merge 后的状态 |
| $T(x)$ | Token 序列 | 输入 $x$ 的所有分段完成 merge 后的拼接结果 |
| $t_j$ | $\mathcal{V}_K$ 中元素 | 最终 token 序列中的第 $j$ 个 token |
| $\operatorname{Encode}(x)$ | 整数序列 | 输入 Unicode 字符串 $x$ 的 token ID 编码 |
| $I=(i_1,\ldots,i_h)$ | 整数序列 | 待解码的 token ID 序列 |
| $i_j$ | 非负整数 | ID 序列中的第 $j$ 个 token ID |
| $B(I)$ | 字节串 | ID 序列 $I$ 经查表后所得 token bytes 的拼接 |
| $\operatorname{Decode}(I)$ | Unicode 字符串 | 对 $B(I)$ 执行 UTF-8 解码的结果 |
