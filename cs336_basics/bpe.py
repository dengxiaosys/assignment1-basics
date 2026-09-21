"""字节级 BPE 分词器训练。

实现 handout §2.4 的 BPE 训练三步：
  1. 词表初始化：256 个字节 + 用户给的 special tokens。
  2. 预分词：用 special tokens 切成硬边界段（段间不合并），段内用 GPT-2 正则
     抽预 token，统计 {预token 的字节元组: 频次}。
  3. 迭代合并：统计相邻字节对频次，取最高频对（并列时取字典序更大的对），
     合并并记录，直到词表达到 vocab_size。

关键约定（决定结果是否与参考一致）：
  - 并列打破：`max` 取 **字典序更大** 的 pair（handout 明确要求）。
  - 合并不跨预 token 边界，也不跨 special token 边界。
"""

import os
import regex as re
from collections import Counter, defaultdict

# GPT-2 预分词正则（取自 tiktoken#234），需要 `regex` 包支持 \p{L}\p{N}。
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _pretoken_counts(text: str, special_tokens: list[str]) -> Counter:
    """把文本预分词成 {字节元组: 频次}。

    先在 special tokens 处切开（这些位置是硬边界，绝不参与合并统计），
    再对每一段用 GPT-2 正则抽预 token，把每个预 token 表示成单字节的元组。
    """
    counts: Counter = Counter()
    # 按 special tokens 切分；special token 本身不进入合并统计。
    if special_tokens:
        # 用 | 连接并转义，切出来的片段里不含 special token。
        pattern = "|".join(re.escape(tok) for tok in special_tokens)
        segments = re.split(pattern, text)
    else:
        segments = [text]

    for seg in segments:
        for match in re.finditer(PAT, seg):
            piece = match.group()
            b = piece.encode("utf-8")
            # 每个预 token 初始为"单字节 bytes 对象"的元组，如 (b'l', b'o', b'w')
            counts[tuple(bytes([x]) for x in b)] += 1
    return counts


def _merge_word(word: list[bytes], pair: tuple[bytes, bytes], new_token: bytes) -> list[bytes]:
    """把一个词里所有相邻的 `pair` 合并成 `new_token`，返回新的 token 列表。

    从左到右扫描：命中 pair 就替换成 new_token 并跳过两个位置（i += 2，避免重复
    消费已合并的元素），否则原样保留（i += 1）。
    """
    merged: list[bytes] = []
    i = 0
    n = len(word)
    while i < n:
        if i < n - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            merged.append(new_token)
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return merged


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """训练字节级 BPE，返回 (vocab, merges)。

    vocab: {token_id: token_bytes}
    merges: [(token1_bytes, token2_bytes), ...] 按创建顺序。
    """
    # ---- 1) 初始词表：256 字节 + special tokens ----
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")

    # ---- 2) 预分词计数 ----
    with open(input_path, encoding="utf-8") as f:
        text = f.read()
    word_counts = dict(_pretoken_counts(text, special_tokens))

    # ---- 3) 迭代合并，直到达到目标词表大小（增量更新，避免每轮全扫）----
    # 用 list 存所有词，便于用下标做倒排索引。
    words: list[list[bytes]] = [list(w) for w in word_counts.keys()]
    freqs: list[int] = list(word_counts.values())

    # pair_counts: 每个相邻对的加权频次；pair_to_words: 该对出现在哪些词（下标集合）。
    pair_counts: Counter = Counter()
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for wi, word in enumerate(words):
        f = freqs[wi]
        for a, b in zip(word[:-1], word[1:]):
            pair_counts[(a, b)] += f
            pair_to_words[(a, b)].add(wi)

    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)

    for _ in range(num_merges):
        if not pair_counts:
            break
        # 取最高频对；并列时取字典序更大的对（handout 要求）。
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        if pair_counts[best_pair] <= 0:
            break
        new_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[len(vocab)] = new_token

        # 只处理包含 best_pair 的词：先撤销它们对 pair_counts 的贡献，
        # 合并后再重新登记新词的相邻对。
        affected = list(pair_to_words[best_pair])
        for wi in affected:
            word = words[wi]
            f = freqs[wi]

            # 撤销旧词的所有相邻对贡献
            for a, b in zip(word[:-1], word[1:]):
                pair_counts[(a, b)] -= f
                pair_to_words[(a, b)].discard(wi)
                
            # 合并该词里所有 best_pair
            merged = _merge_word(word, best_pair, new_token)
            words[wi] = merged
            
            # 登记新词的相邻对贡献
            for a, b in zip(merged[:-1], merged[1:]):
                pair_counts[(a, b)] += f
                pair_to_words[(a, b)].add(wi)

        # best_pair 已被消除
        del pair_counts[best_pair]
        pair_to_words.pop(best_pair, None)

    return vocab, merges
