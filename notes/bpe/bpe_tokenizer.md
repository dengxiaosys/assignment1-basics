# Tokenizer 技术谱系：BPE 的历史背景、设计空间与主流方案

## 文档定位

本文讨论 tokenizer 的历史背景、系统作用、设计空间，以及当前主流算法与工程生态之间的差异。重点不是再次推导 BPE 的每一轮 merge，而是回答以下问题：

1. 为什么语言模型需要 tokenizer；
2. BPE 在怎样的历史背景下产生；
3. BPE 实际解决了什么问题，又没有解决什么问题；
4. BPE、WordPiece、Unigram、SentencePiece 和 tiktoken 分别属于什么概念层级；
5. 当前常见 tokenizer 家族各自适用于哪些模型与数据；
6. 词表大小、语言覆盖、序列长度和计算成本之间如何权衡。

Byte-level BPE 的逐轮训练、merge rank、编码、解码及数学形式化推导已经集中到：

- [Byte-level BPE 全流程：一个可手工复算的完整例子](./byte_level_bpe_worked_example.md)

本文不再重复该算例中的操作细节。

## 1. Tokenizer 不是普通的文本预处理

### 1.1 从字符串空间到离散状态空间

语言模型通常接收整数序列，而不是自然语言字符串。Tokenizer 定义映射：

$$ \operatorname{Encode}:\mathcal{X}\rightarrow\{0,1,\ldots,V-1\}^{*}, $$

其中 $\mathcal{X}$ 是可接受的文本空间，$V$ 是词表大小。

解码器定义反向映射：

$$ \operatorname{Decode}:\{0,1,\ldots,V-1\}^{*}\rightarrow\mathcal{X}. $$

对于无损 tokenizer，合法输入应满足：

$$ \operatorname{Decode}(\operatorname{Encode}(x))=x. $$

因此，tokenizer 实际上规定了模型的离散输入空间。模型 embedding、输出 projection、训练数据、推理服务、上下文长度与计费单位都建立在该空间之上。

### 1.2 Tokenizer 是模型协议的一部分

一个可复现的 tokenizer 不只包含“词表”。完整定义通常包括：

- Unicode normalization；
- pre-tokenization 规则；
- 基础符号集合；
- 子词学习算法；
- merge rank 或 token score；
- token 到 ID 的映射；
- special token 集合及其优先级；
- post-processing；
- padding、truncation 和 chat template 约定；
- 非法字节与未知输入的回退策略。

只复制 `vocab.json` 而遗漏 normalization、pre-tokenizer 或 merges，通常无法复现相同的 token ID 序列。

### 1.3 Tokenizer 同时影响参数和计算

设词表大小为 $V$，模型维度为 $d$，token 序列长度为 $n$。

若输入 embedding 与输出 projection 不共享参数，两部分参数量近似为：

$$ P_{\mathrm{vocab}}\approx 2Vd. $$

若二者共享权重，则近似为：

$$ P_{\mathrm{vocab}}\approx Vd. $$

输出 projection 的主要计算量近似为：

$$ C_{\mathrm{output}}=O(nVd). $$

Self-attention 的主要计算量近似为：

$$ C_{\mathrm{attention}}=O(n^2d). $$

增大词表 $V$ 往往可以缩短序列 $n$，但会增加 embedding 与输出层成本；减小词表则会产生相反影响。Tokenizer 设计本质上是离散表示与系统成本之间的联合优化。

## 2. 开放词表问题

### 2.1 为什么词级建模不可持续

早期自然语言处理系统经常把完整单词作为离散单位。该方案直观且序列较短，但自然语言词表并不封闭：

- 人名、地名和机构名持续出现；
- 产品型号、日期、URL 和哈希几乎没有上限；
- 拼写错误和网络语言形成大量变体；
- 形态丰富语言会为同一词根产生大量词形；
- 代码标识符可任意组合；
- 多语言系统面对庞大的联合词表。

若词不在词表中，传统方案通常映射为 `<unk>`。多个不同字符串因此坍缩为同一个符号，信息在进入模型之前已经不可逆丢失。

### 2.2 Zipf 分布与长尾浪费

自然语言词频近似服从 Zipf 分布。少量词非常常见，大量词只出现一次或数次。

完整保留所有低频词会导致：

- 词表和参数规模过大；
- 长尾 embedding 训练不足；
- 输出层在大量低频类别上浪费计算；
- 新词仍然无法覆盖。

完全丢弃长尾词又会产生严重 OOV。合理方案应当让高频字符串使用较大单位，让低频字符串回退到较小单位。

### 2.3 从闭集分类转向可组合表示

子词 tokenization 的核心思想不是“找到语言的真正单词”，而是把开放字符串空间表示为有限符号集合的组合闭包：

$$ x=t_1\Vert t_2\Vert\cdots\Vert t_n. $$

高频字符串可以成为单个 token，低频字符串则由多个较小 token 构成。BPE、WordPiece 和 Unigram 都属于这一思路，但其词表学习目标不同。

## 3. 粒度选择的基本矛盾

![Tokenization 粒度之间的系统权衡](../images/tokenization_granularity_tradeoff.svg)

图 1：从词级到字节级，基础词表逐渐缩小，覆盖能力增强，但序列通常变长。

| 粒度 | 基础单位 | 主要优点 | 主要缺点 |
|---|---|---|---|
| 词级 | 完整单词 | 序列短，语义直观 | OOV 严重，词表巨大 |
| 子词级 | 高频字符串片段 | 词表与序列长度较均衡 | 分词依赖语料，边界不等于语义 |
| 字符级 | Unicode code point 或 grapheme | 对词形和拼写变化稳健 | 序列长，多语言字符集合仍大 |
| 字节级 | 0–255 的字节值 | 无 OOV，基础词表固定为 256 | 序列最长，模型承担更多组合学习 |

子词 tokenizer 之所以长期占据主流，是因为它在词表大小、序列长度和开放输入覆盖之间提供了工程上可接受的折中。

## 4. Tokenizer 技术的历史脉络

### 4.1 词典与规则分词时代

传统 NLP 系统通常依赖：

- 空白和标点规则；
- 语言专用分词器；
- 人工词典；
- stemming 与 lemmatization；
- 固定 OOV token。

这些方法适合封闭任务，但难以统一处理多语言、噪声文本、代码和开放领域语料。

### 4.2 原始 Byte Pair Encoding

Philip Gage 在 1994 年提出 Byte Pair Encoding，目标是数据压缩。算法反复寻找最常见的相邻 byte pair，并用新符号替换该 pair。

原始 BPE 关注存储压缩，不理解单词、词根或语义。它提供的是一种贪心字典构造机制：

> 高频局部模式值得分配独立符号。

### 4.3 WordPiece

WordPiece 最早用于日语和语音搜索系统，后来因 BERT 系列而广泛传播。

它与 BPE 都逐步构造子词词表，但通常不直接按原始 pair 频率选择 merge，而是使用更接近语言模型似然或符号关联强度的准则。编码阶段常采用 longest-match-first。

### 4.4 NLP 中的 BPE

Sennrich、Haddow 和 Birch 在 2016 年把 BPE 用于神经机器翻译中的稀有词问题。完整单词被拆成可复用的子词，翻译模型不再需要把所有低频词映射为 `<unk>`。

这一工作确立了现代子词建模的基本范式：

1. 用有限词表覆盖开放字符串；
2. 用高频片段压缩序列；
3. 通过组合处理未见词。

### 4.5 Unigram Language Model

Unigram tokenizer 不从小词表逐步合并，而是从较大的候选词表开始，为 token 分配概率，再反复删除对语料似然贡献较小的候选项。

分词被视为隐变量。对字符串 $x$，最优切分为：

$$ z^{*}=\underset{z\in\mathcal{Z}(x)}{\arg\max}\sum_{t\in z}\log p(t). $$

这种概率化定义允许同一字符串存在多条合法切分路径，也支持 subword regularization。

### 4.6 SentencePiece

SentencePiece 把 normalization、空白表示、子词训练和解码封装成统一系统。它可以使用 BPE，也可以使用 Unigram。

因此：

> SentencePiece 是 tokenizer 框架和模型格式，不是与 BPE 并列的单一分词算法。

SentencePiece 直接从原始句子训练，用 `▁` 一类 meta-symbol 显式表示空格，适合没有天然空格边界的语言。

### 4.7 Byte-level BPE

GPT-2 进一步普及了 byte-level BPE。其基础符号不是 Unicode 字符，而是 UTF-8 字节。

这带来两项关键性质：

- 任意输入都可回退到 256 个基础 byte token；
- 不需要 `<unk>` 才能处理未见 Unicode 字符。

代价是单个 token 可能只是某个 Unicode 字符的部分字节，token 边界与字符边界不再一致。

### 4.8 当代 tokenizer

当代 tokenizer 已经从“文本切分器”扩展为模型输入协议。除普通文本外，它还负责表达：

- system、user、assistant 角色；
- tool call 与 tool result；
- fill-in-the-middle；
- 图像、音频和视频占位符；
- 文档边界；
- reasoning 或控制模式；
- padding、BOS、EOS 和 generation boundary。

Tokenizer 的 special token 与 chat template 因而具有接口语义，不能只从字符串压缩角度理解。

## 5. BPE 为什么长期流行

### 5.1 确定性与可部署性

固定 pre-tokenizer、词表和 merge rank 后，BPE 编码是确定性的。它不依赖神经网络推理，也不需要复杂动态规划，适合高吞吐数据管线和在线服务。

### 5.2 可控词表预算

每执行一次 merge，词表恰好增加一个 token。因此，目标词表大小可以直接控制。

### 5.3 兼顾覆盖与压缩

若以字符或字节为基础符号，BPE 既保留回退路径，又能把高频片段合并为较长 token。

### 5.4 与语言模型目标弱耦合

Tokenizer 可以在训练语言模型之前独立构建，并复用于不同模型规模。对于需要重复训练大量模型的工程体系，这种解耦具有明显价值。

### 5.5 工程生态成熟

BPE 已经拥有：

- 高性能 Rust、C++ 实现；
- 成熟的序列化格式；
- 流式编码和缓存；
- GPU/CPU 数据管线支持；
- 大量兼容的预训练模型。

迁移成本和生态惯性进一步强化了其主流地位。

## 6. BPE 没有解决什么

### 6.1 不保证语言学边界

BPE token 是频率驱动的字节串或字符片段，不一定对应词根、词缀、汉字或完整单词。

### 6.2 不保证全局最优

BPE 每轮执行局部贪心选择。早期 merge 会改变后续候选空间，无法保证最终词表在全局意义上具有最优压缩率或最低语言模型 loss。

### 6.3 不直接优化下游模型质量

BPE 通常优化局部频率与序列压缩，而不是验证集 loss、推理延迟或任务准确率。压缩率更高不必然意味着模型效果更好。

### 6.4 不能消除语料偏置

高资源语言和高频领域会获得更多完整 token。低资源语言可能被切得更碎，承担更高 token 成本和更短有效上下文。

### 6.5 对 normalization 与 pre-tokenization 敏感

相同 BPE 算法配合不同 Unicode normalization、数字规则、空格策略或正则表达式，会生成完全不同的词表。

## 7. BPE 的主要变体

| 变体 | 基础单位 | 主要特征 | 主要风险 |
|---|---|---|---|
| 字符级 BPE | Unicode 字符 | Token 更接近可见字符 | 字符集合大，仍可能 OOV |
| Byte-level BPE | UTF-8 字节 | 256 项基础词表，无 OOV | 序列可能更长，token 可跨字符边界 |
| SentencePiece BPE | 原始 Unicode 与空格 meta-symbol | 不依赖外部词分割 | Normalization 与模型格式绑定 |
| BPE-Dropout | BPE merge 加随机丢弃 | 产生多种切分，增强鲁棒性 | 训练不再完全确定，调参更复杂 |
| BPE + byte fallback | 子词为主，字节为回退 | 常见文本紧凑，任意输入可表示 | 罕见字符会突然膨胀为多个 token |
| Domain-adaptive BPE | 特定领域语料 | 对代码、医学、法律等压缩率高 | 跨领域泛化和兼容性较差 |

完整 merge 算例见 [Byte-level BPE 全流程](./byte_level_bpe_worked_example.md)。

## 8. 需要区分的三个概念层级

Tokenizer 讨论中最常见的混淆，是把算法、实现框架和模型专用资产放在同一层比较。

| 层级 | 回答的问题 | 示例 |
|---|---|---|
| 算法家族 | 词表如何学习、字符串如何切分 | BPE、WordPiece、Unigram、WordLevel |
| 实现框架 | 如何训练、序列化和高性能执行 | SentencePiece、tiktoken、Hugging Face Tokenizers |
| 模型专用资产 | 某个模型实际采用哪些规则和 ID | `tokenizer.json`、`tokenizer.model`、`tekken.json`、chat template |

### 8.1 SentencePiece 不是 Unigram 的同义词

SentencePiece 同时实现 BPE 和 Unigram。一个模型使用 SentencePiece，并不能推出它一定使用 Unigram。

### 8.2 tiktoken 不是新的统计目标

tiktoken 是面向 BPE 的高性能实现和 encoding 生态。`cl100k_base`、`o200k_base` 等是具体 encoding 资产，不是与 BPE 并列的新算法。

### 8.3 Hugging Face Tokenizers 不是单一 tokenizer

Hugging Face Tokenizers 是 Rust 实现的通用流水线，支持 BPE、WordPiece、Unigram 和 WordLevel，并提供 normalization、pre-tokenization、alignment、padding、truncation 与 post-processing。

### 8.4 模型名称也不是 tokenizer 算法

“BERT tokenizer”“GPT tokenizer”“Mistral tokenizer”通常指特定模型资产与协议。其底层仍需进一步说明是 WordPiece、BPE、Unigram，还是其他方案。

## 9. 当前主流 tokenizer 家族

### 9.1 Byte-level BPE

#### 核心机制

Byte-level BPE 从 UTF-8 字节出发，通过有序 merge 把高频字节串提升为 token。

#### 代表生态

- OpenAI `tiktoken` encodings；
- Mistral Tekken；
- 多种现代 decoder-only LLM 的 BPE 词表；
- Hugging Face ByteLevel + BPE 流水线。

#### 优势

- 任意 Unicode 输入均可表示；
- 不需要传统 `<unk>`；
- 编码确定、可逆；
- 对代码、URL、控制字符和噪声文本具有统一回退路径；
- 高性能实现成熟；
- 易于固定词表预算。

#### 劣势

- 罕见语言和复杂 emoji 可能产生较长序列；
- Token 可能是不完整 UTF-8 片段；
- 词表受训练语料语言比例显著影响；
- 固定 merge 只有一种默认切分；
- 贪心频率目标不等价于下游最优。

#### 适用场景

开放域 decoder-only LLM、多语言与代码混合语料、需要严格无 OOV 和高吞吐推理的系统。

### 9.2 SentencePiece BPE

#### 核心机制

使用 SentencePiece 处理原始 Unicode 文本，以 BPE 学习子词。空格通常映射为可见 meta-symbol，因此可逆性不依赖外部空格规则。

#### 优势

- 不依赖语言专用分词器；
- 中文、日文等无空格语言可直接训练；
- 模型文件可封装 normalization、词表和切分信息；
- 支持 BPE-Dropout；
- C++ 实现成熟，跨语言绑定丰富。

#### 劣势

- SentencePiece normalization 可能改变原始 code point 序列；
- 若未启用 byte fallback，字符覆盖配置不当仍可能产生 `<unk>`；
- 与 JSON-based tokenizer 生态互转时容易遗漏 normalization 或 special token 语义；
- 具体模型到底使用 BPE 还是 Unigram，需要查看配置，不能只看文件后缀。

#### 适用场景

多语言生成模型、机器翻译、需要从原始句子训练且希望减少外部 pre-tokenization 依赖的系统。

### 9.3 WordPiece

#### 核心机制

WordPiece 构造子词词表，并在编码时通常执行 longest-match-first。连续子词常使用 `##` 一类前缀标记。

#### 代表生态

- BERT；
- mBERT；
- DistilBERT；
- 大量兼容 BERT 接口的 encoder checkpoint。

#### 优势

- 在 encoder 模型生态中成熟稳定；
- 子词边界与词内 continuation 关系明确；
- 词表和实现格式简单；
- 对自然语言分类、检索和序列标注具有大量历史兼容资产。

#### 劣势

- 传统实现依赖词级 pre-tokenization；
- 单词无法拆分时可能整体变为 `[UNK]`；
- 对拼写错误、长标识符、URL 和混合代码不如 byte-level 回退稳健；
- 通常只提供单一确定性切分；
- 多语言联合词表容易出现语言间容量不均衡。

#### 适用场景

BERT 系 encoder、需要复用成熟 checkpoint 和 `vocab.txt` 生态的任务。

### 9.4 Unigram Language Model

#### 核心机制

Unigram 从较大候选词表出发，为 token 分配概率，通过 EM 与剪枝逐渐缩小词表。编码可通过 Viterbi 寻找最高概率路径，也可采样其他路径。

#### 代表生态

- SentencePiece Unigram；
- T5、mT5 等 SentencePiece 模型；
- 机器翻译与多语言预训练体系。

#### 优势

- 具有明确概率模型；
- 同一字符串可以存在多条合法切分；
- 支持 subword regularization；
- 剪枝过程能够撤销不理想候选；
- 对数据增强和噪声鲁棒性研究更友好。

#### 劣势

- 训练比 BPE 更复杂；
- 依赖初始候选词表质量；
- 编码通常需要动态规划；
- 概率、采样温度和 n-best 配置增加复现复杂度；
- 若基础字符覆盖或 byte fallback 配置不足，仍可能出现未知 token。

#### 适用场景

机器翻译、多语言模型、希望通过随机子词切分增强泛化的训练体系。

### 9.5 WordLevel

#### 核心机制

直接把 pre-token 映射到固定 ID。

#### 优势

- 编码逻辑简单；
- 在小型封闭词典中序列最短；
- Token 语义直观；
- 可与结构化命令、标签集合或专业术语表对齐。

#### 劣势

- 开放文本 OOV 严重；
- 词表规模大；
- 形态变化和拼写噪声导致碎片化或 `<unk>`；
- 不适合通用生成模型。

#### 适用场景

封闭命令集、有限实体集合、传统检索特征、严格受控领域。

### 9.6 纯字节与 token-free 模型

#### 核心机制

不学习子词词表，直接以 256 个字节值作为模型输入；部分架构在模型内部执行局部下采样或层次化建模。

#### 代表方向

- ByT5；
- MegaByte；
- 字节级语言模型；
- 字符或字节下采样架构。

#### 优势

- 完全消除 tokenizer OOV；
- 不存在静态子词词表的语言容量分配；
- 对拼写、噪声和任意字节内容鲁棒；
- embedding 与输出词表很小；
- 文本表示规则极其稳定。

#### 劣势

- 序列显著增长；
- Attention 和激活成本增加；
- 模型必须自行学习 byte-to-character 和 character-to-word 层次；
- 固定 token context 可容纳的语义内容减少；
- 现有 LLM 训练与推理基础设施主要围绕子词优化。

#### 适用场景

研究 tokenizer 偏置、强噪声鲁棒性、超多语言覆盖，以及具备专用层次化架构的模型。

### 9.7 Byte fallback 混合方案

#### 核心机制

常见文本优先使用字符或子词 token；当字符不在主词表中时，回退为对应 UTF-8 字节。

#### 优势

- 常见文本保持较短序列；
- 任意 Unicode 输入仍可表示；
- 不要求所有 merge 都直接在原始字节层训练；
- 易于给现有 SentencePiece 或 Unigram 方案补充开放词表能力。

#### 劣势

- 罕见字符会突然展开为多个 token；
- 主词表 token 与 byte fallback token 共存，协议更复杂；
- 不同库的 byte token 命名和序列化格式可能不兼容；
- 训练语料中未见语言仍可能具有很高 fertility。

#### 适用场景

希望保留字符级或 Unigram 词表，同时要求严格无 OOV 的多语言系统。

## 10. 当前主流工程框架

### 10.1 tiktoken

[tiktoken](https://github.com/openai/tiktoken) 是 OpenAI 开源的高性能 BPE tokenizer。官方仓库提供 `cl100k_base`、`o200k_base` 等 encoding，并支持按模型名称选择 encoding。

#### 工程优势

- Rust 核心实现，吞吐高；
- Mergeable ranks 与 regex pre-tokenizer 结构清晰；
- 对 byte-level BPE 支持直接；
- 可注册自定义 encoding；
- 适合 token 计数、在线服务和大规模数据预处理。

#### 局限

- 主要围绕 BPE，不是多算法研究框架；
- Encoding 行为高度依赖 regex 和 special token 配置；
- 自定义 special token 若与已有 encoding 混用，必须严格版本化；
- 与 SentencePiece 模型并非直接等价。

### 10.2 SentencePiece

[SentencePiece](https://github.com/google/sentencepiece) 是面向神经文本模型的 C++ tokenizer 框架，支持 BPE 与 Unigram，可直接从原始句子训练。

#### 工程优势

- 语言无关；
- 空白可逆表示；
- `.model` 文件自包含；
- 支持 normalization；
- 支持 subword regularization 与 BPE-Dropout；
- 多语言和机器翻译生态成熟。

#### 局限

- 模型内部 normalization 容易被忽视；
- 将 `.model` 转换为其他格式时可能产生行为差异；
- Special token 与 chat template 通常还需要模型侧额外配置；
- “使用 SentencePiece”不足以说明具体算法。

### 10.3 Hugging Face Tokenizers

[Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/index) 是 Rust 实现的通用 tokenizer 流水线，也是 Transformers 中大量 fast tokenizer 的底层。

官方模型 API同时支持：

- BPE；
- WordPiece；
- Unigram；
- WordLevel；
- byte fallback；
- BPE dropout；
- token 与原文 offset alignment。

#### 工程优势

- 多算法统一接口；
- 训练和编码速度高；
- normalization、pre-tokenization、model、post-processing 分层明确；
- 支持 offset mapping；
- 与 Hugging Face Hub 和 Transformers 集成紧密。

#### 局限

- 配置自由度高，也意味着两个名称相同的 tokenizer 可能行为不同；
- `tokenizer.json`、Python wrapper、chat template 和模型配置必须协同版本化；
- Slow tokenizer 与 fast tokenizer 的边界行为需要测试；
- 跨库转换仍可能在 normalization、added token 和 byte fallback 上出现差异。

### 10.4 Mistral Tekken

Mistral 的官方文档说明，其 tokenizer 体系从 SentencePiece 迁移到基于 tiktoken 的 [Tekken](https://mistralai.github.io/mistral-common/usage/tokenizers/)，并将 tokenizer 配置保存为 `tekken.json`。

Tekken 体现了一个重要趋势：

> 模型 tokenizer 不再只是基础 BPE，而是 raw tokenizer、instruction protocol、tool call、FIM 和多模态 special token 的组合系统。

#### 工程优势

- 基于 tiktoken 的高性能实现；
- 针对多语言效率设计；
- Tokenizer 版本与模型请求协议绑定；
- 支持聊天、工具调用和图像控制 token。

#### 局限

- 模型专用协议更强，跨模型复用更困难；
- Special token 不能简单当普通字符串 encode；
- `tekken.json` 与通用 SentencePiece 格式不兼容；
- Serving 系统需要显式支持 Mistral 的 instruction tokenizer 层。

## 11. 主流方案横向比较

| 方案 | 词表学习原则 | 编码策略 | OOV | 多种切分 | 典型优势 | 典型劣势 |
|---|---|---|---|---|---|---|
| Byte-level BPE | 最高频 byte pair 贪心合并 | 固定 merge rank | 无 | 默认无 | 通用、可逆、高吞吐 | 语料偏置、罕见文本较碎 |
| 字符级 BPE | 最高频字符 pair 合并 | 固定 merge rank | 取决于字符覆盖 | 默认无 | Token 更接近字符 | Unicode 词表大，仍需 fallback |
| WordPiece | 似然或关联准则构词表 | Longest-match-first | 常有 `[UNK]` | 无 | BERT 生态成熟 | 噪声和开放输入较脆弱 |
| Unigram | 概率模型与迭代剪枝 | Viterbi 或采样 | 取决于 fallback | 有 | 概率化、支持正则化 | 训练和编码更复杂 |
| SentencePiece BPE | BPE + 原始文本框架 | 固定 merge | 取决于字符覆盖/fallback | 可用 BPE-Dropout | 语言无关、自包含 | Normalization 与格式迁移复杂 |
| WordLevel | 固定词典 | 精确查表 | 高 | 无 | 简单、序列短 | 不适合开放世界 |
| 纯字节 | 无学习词表 | 每字节一个 token | 无 | 无 | 极强覆盖和鲁棒性 | 序列长、模型计算重 |
| 子词 + byte fallback | 主子词模型 + 字节回退 | 主路径失败时展开字节 | 无 | 取决于主模型 | 平衡常见文本与开放覆盖 | 稀有文本 token 数突增 |

## 12. 选择 tokenizer 时真正需要比较的维度

### 12.1 覆盖能力

需要确认：

- 任意 Unicode 是否可编码；
- 非法字节如何处理；
- 是否存在 `<unk>`；
- 罕见脚本是否退化为逐字节；
- emoji 与组合字符是否可逆。

### 12.2 压缩率

常用指标为 bytes/token：

$$ R_{\mathrm{compression}}=\frac{\text{UTF-8 byte count}}{\text{token count}}. $$

该值越高，固定 token context 中通常能容纳更多原始文本。但压缩率不应只报告全局均值。

### 12.3 Fertility

对有空格词边界的语言，可以计算：

$$ F=\frac{\text{token count}}{\text{word count}}. $$

Fertility 越高，说明同一单词被切得越碎。跨语言比较时还应同时报告 characters/token 与 bytes/token。

### 12.4 多语言公平性

设同一语义内容在语言 $a$ 与语言 $b$ 中分别需要 $n_a,n_b$ 个 token，可定义相对 token 成本：

$$ \Gamma_{a,b}=\frac{n_a}{n_b}. $$

若 $\Gamma_{a,b}$ 长期显著偏离 1，则两种语言在有效上下文、推理费用和截断概率上承担不同成本。

### 12.5 领域适配

代码、数学、医学和法律文本具有不同高频模式。通用 tokenizer 可能把领域符号切得过碎；领域 tokenizer 又可能降低普通文本兼容性。

### 12.6 吞吐与延迟

训练语料离线编码关注 bytes/s；在线推理还需关注：

- 单请求固定开销；
- 小字符串延迟；
- 批量编码效率；
- streaming 边界；
- 多线程扩展；
- 内存与缓存。

### 12.7 协议兼容

需要验证：

- BOS/EOS/PAD ID；
- chat template；
- tool call token；
- FIM token；
- multimodal placeholder；
- added token；
- 模型 embedding size；
- serving runtime 的 tokenizer 版本。

## 13. 不同场景下的选择

### 13.1 通用 decoder-only LLM

优先需求通常是：

- 无 OOV；
- 多语言和代码统一覆盖；
- 高吞吐；
- 可逆；
- 大规模 serving 生态成熟。

Byte-level BPE 或带 byte fallback 的子词 tokenizer 通常更合适。

### 13.2 BERT 系 encoder

如果目标是复用 BERT、mBERT 或 DistilBERT checkpoint，WordPiece 是模型定义的一部分，不应为了理论偏好随意更换。

重新训练 tokenizer 会改变 embedding 行语义，原 checkpoint 无法直接兼容。

### 13.3 多语言模型

需要重点比较：

- 各语言 bytes/token；
- 平行语料的 token 数比；
- 低资源语言 byte fallback 比例；
- 各脚本在词表中的容量；
- normalization 是否改变语言特有字符。

Unigram、SentencePiece BPE、byte-level BPE 和 byte fallback 都可用于多语言，但最终公平性主要取决于训练语料、词表预算和字符覆盖配置。

### 13.4 代码模型

代码 tokenizer 需要单独关注：

- 缩进和换行；
- 运算符与标点；
- 长标识符；
- 数字与十六进制常量；
- 多语言源码；
- 自然语言注释；
- FIM special token。

Byte-level BPE 通常具有较好开放覆盖，但 regex pre-tokenizer 若错误拆分缩进或操作符，仍会显著降低效率。

### 13.5 封闭领域系统

命令识别、有限标签或固定结构协议可以使用 WordLevel 或人工词表。此时开放文本能力并非首要目标，短序列和可解释 ID 可能更重要。

### 13.6 噪声与安全敏感输入

面对拼写扰动、控制字符、同形异码和恶意 Unicode，需关注：

- normalization 策略；
- byte fallback；
- round-trip；
- special token 注入；
- 不可见字符；
- tokenizer 与模型服务之间的字符串处理差异。

Byte-level 覆盖只能保证“可表示”，不能自动解决 Unicode 安全问题。

## 14. 当代 tokenizer 的发展趋势

### 14.1 词表规模增大

许多现代 LLM 使用六位数规模词表。较大词表有助于多语言、代码和常见短语压缩，但会增加 embedding、LM head 与训练稀疏性成本。

### 14.2 Byte fallback 成为常见保险

严格无 OOV 已逐渐成为通用模型的基本要求。即使主算法不是 byte-level BPE，也经常加入 byte fallback。

### 14.3 Tokenizer 与训练语料共同设计

Tokenizer 不再只在随机语料子集上训练。更成熟的流程会控制：

- 语言采样比例；
- 代码比例；
- 去重；
- 数字和标点分布；
- 特殊领域；
- 隐私和长字符串污染。

### 14.4 Tokenizer 与对话协议融合

Chat、tool use、FIM 和 multimodal 输入需要大量控制 token。Tokenizer 资产因此与模型协议、模板和 serving 实现更紧密地绑定。

### 14.5 性能实现专业化

tiktoken、Hugging Face Tokenizers、SentencePiece 和 Tekken 都把吞吐、缓存、序列化与跨语言绑定视为核心能力。算法相同并不意味着工程性能相同。

### 14.6 Token-free 研究持续存在

ByT5、MegaByte 等工作试图把分词偏置移入模型内部。该方向理论上更统一，但更长序列和更高计算成本使其尚未全面替代子词 tokenizer。

## 15. 一个实用决策框架

### 第一步：确认兼容性约束

若已有预训练 checkpoint，通常必须使用原 tokenizer。Tokenizer 不是可独立替换的前端插件。

### 第二步：确认覆盖目标

- 通用开放文本：要求 byte-level 或 byte fallback；
- 封闭词典：WordLevel 可能足够；
- 传统 BERT 兼容：WordPiece；
- 多切分正则化：Unigram；
- 高吞吐 GPT 风格服务：tiktoken-compatible BPE。

### 第三步：确定评估语料

评估集必须覆盖真实输入分布，而不是只使用英语新闻文本。多语言、代码、数字、URL、emoji 和控制字符需要单独统计。

### 第四步：联合比较 $V$ 与 $n$

Tokenizer 选择不能只比较 token 数，也不能只比较词表大小。至少需要同时测量：

- 平均序列长度；
- 词表大小；
- embedding/LM head 参数；
- attention 计算；
- 编码吞吐；
- 下游验证 loss。

### 第五步：冻结完整协议

发布 tokenizer 时应共同版本化：

- normalization；
- pre-tokenizer；
- vocab；
- merges 或 token scores；
- special token；
- post-processor；
- chat template；
- 测试向量。

## 16. 结论

BPE 的历史价值在于把开放词表问题转化为有限符号的可组合表示问题。它以简单、确定、可控的贪心压缩机制，在词级模型与字节级模型之间建立了长期有效的工程折中。

但“当前使用哪种 tokenizer”不能只回答 BPE、WordPiece 或 Unigram。完整答案至少需要包含三个层次：

1. **算法家族**：词表如何学习、切分如何求解；
2. **实现框架**：normalization、pre-tokenization、序列化和性能如何实现；
3. **模型协议**：special token、chat template 和多模态边界如何定义。

Byte-level BPE 仍是通用 decoder-only LLM 的重要主流方案；WordPiece 仍广泛存在于 BERT 系 encoder；Unigram 与 SentencePiece 在多语言和机器翻译生态中保持重要地位；byte fallback 与纯字节模型则分别代表工程保险和更彻底的开放输入路线。

不存在脱离语料、模型架构、硬件与应用协议的“最佳 tokenizer”。合理选择应建立在覆盖率、压缩率、语言公平性、吞吐、下游质量与兼容性共同测量之上。

## 参考资料

1. Philip Gage, “A New Algorithm for Data Compression,” *C Users Journal*, 1994.
2. Rico Sennrich, Barry Haddow, Alexandra Birch, “Neural Machine Translation of Rare Words with Subword Units,” ACL 2016.
3. Mike Schuster, Kaisuke Nakajima, “Japanese and Korean Voice Search,” ICASSP 2012.
4. Taku Kudo, “Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates,” ACL 2018.
5. Taku Kudo, John Richardson, “SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing,” EMNLP 2018.
6. Alec Radford et al., *Language Models are Unsupervised Multitask Learners*, 2019.
7. Ivan Provilkov, Dmitrii Emelianenko, Elena Voita, “BPE-Dropout: Simple and Effective Subword Regularization,” ACL 2020.
8. Linting Xue et al., “ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models,” TACL 2022.
9. [OpenAI tiktoken](https://github.com/openai/tiktoken).
10. [SentencePiece](https://github.com/google/sentencepiece).
11. [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/index).
12. [Hugging Face Tokenizer Models API](https://huggingface.co/docs/tokenizers/api/models).
13. [Mistral Tokenizers and Tekken](https://mistralai.github.io/mistral-common/usage/tokenizers/).
