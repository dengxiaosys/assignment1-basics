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
from collections.abc import Iterable, Iterator
from typing import BinaryIO

# GPT-2 预分词正则（取自 tiktoken#234），需要 `regex` 包支持 \p{L}\p{N}。
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def gpt2_bytes_to_unicode_safe() -> dict[int, str]:
    """GPT-2 的"字节(0-255) → 可打印 Unicode 字符"可逆映射。

    用于把 bytes 词表/merges 序列化成人类可读、可往返的字符串（与 tiktoken/测试格式一致）。
    可打印字节保留原字符，不可打印字节平移到 256+ 区间的可打印码点。
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """把文件切成若干可独立处理的块，块边界对齐到 split_special_token（如 b"<|endoftext|>"）。

    先按文件大小等分给出初始边界，再把每个边界往后挪到最近的一个 special token 处，
    保证不会把一个"预 token"或文档劈成两半。返回去重排序后的字节偏移列表（可能少于
    desired_num_chunks）。改写自 handout 的 pretokenization_example.find_chunk_boundaries。
    """
    assert isinstance(split_special_token, bytes)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    boundaries[-1] = file_size

    mini_chunk_size = 4096
    for bi in range(1, len(boundaries) - 1):
        pos = boundaries[bi]
        file.seek(pos)
        while True:
            mini = file.read(mini_chunk_size)
            if mini == b"":  # EOF
                boundaries[bi] = file_size
                break
            found = mini.find(split_special_token)
            if found != -1:
                boundaries[bi] = pos + found
                break
            pos += mini_chunk_size
    return sorted(set(boundaries))


def _count_chunk(args: tuple[str, int, int, list[str]]) -> Counter:
    """（供多进程调用）读取文件 [start, end) 区间、预分词并返回计数。"""
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    return _pretoken_counts(text, special_tokens)


def _pretoken_counts_parallel(input_path: str | os.PathLike, special_tokens: list[str], num_processes: int) -> Counter:
    """并行预分词：按 special token 边界切块，多进程各自计数后合并。

    切块对齐到 special token（默认用第一个，通常是 <|endoftext|>），确保分块统计与
    整体统计结果完全一致（合并从不跨文档边界）。
    """
    import multiprocessing as mp

    split_tok = (special_tokens[0] if special_tokens else "<|endoftext|>").encode("utf-8")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_tok)

    tasks = [
        (str(input_path), start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    total: Counter = Counter()
    with mp.Pool(num_processes) as pool:
        for c in pool.map(_count_chunk, tasks):
            total.update(c)
    return total


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
    num_processes: int = 1,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """训练字节级 BPE，返回 (vocab, merges)。

    vocab: {token_id: token_bytes}
    merges: [(token1_bytes, token2_bytes), ...] 按创建顺序。
    num_processes: >1 时并行预分词（按 special token 边界切块），用于大语料加速；
        =1 时整体读入单进程处理（小文件/单测路径）。两种路径结果一致。
    """
    # ---- 1) 初始词表：256 字节 + special tokens ----
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")

    # ---- 2) 预分词计数（可选并行）----
    if num_processes > 1:
        word_counts = dict(_pretoken_counts_parallel(input_path, special_tokens, num_processes))
    else:
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


class Tokenizer:
    """BPE 分词器：加载训练好的 vocab/merges，做 encode（文本→id）与 decode（id→文本）。

    编码流程与训练时镜像：special token 切分 → GPT-2 正则预分词 → 按 merges 的
    创建顺序在每个预 token 内应用合并 → 查 vocab 得到 id。
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab                                  # id -> bytes
        self.byte_to_id = {b: i for i, b in vocab.items()}  # bytes -> id（decode/查表用）
        self.merges = merges
        # merge 的优先级：越早创建 rank 越小，编码时优先应用 rank 最小的可合并对。
        self.merge_rank = {pair: r for r, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """从序列化的 vocab（json: {token_str: id}）与 merges（每行 "tok1 tok2"）构造。

        采用 GPT-2 的"字节↔可打印字符"映射来解析磁盘格式（与 tests 的存储格式一致）。
        """
        import json

        byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode_safe().items()}
        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = {
            idx: bytes([byte_decoder[c] for c in tok])
            for tok, idx in raw_vocab.items()
        }
        merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line and len(line.split(" ")) == 2:
                    a, b = line.split(" ")
                    merges.append(
                        (bytes([byte_decoder[c] for c in a]), bytes([byte_decoder[c] for c in b]))
                    )
        return cls(vocab, merges, special_tokens)

    def _apply_merges(self, token_bytes: bytes) -> list[bytes]:
        """把一个预 token（bytes）拆成单字节，按 merges 优先级反复合并，返回最终 token 列表。"""
        parts = [bytes([x]) for x in token_bytes]
        while len(parts) >= 2:
            # 找当前所有相邻对里 rank 最小（最早创建）的一个可合并对
            best_rank = None
            best_i = -1
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                r = self.merge_rank.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_i = i
            if best_rank is None:
                break  # 没有可合并的对了
            # 合并 best_i 处的这一对
            parts[best_i : best_i + 2] = [parts[best_i] + parts[best_i + 1]]
        return parts

    def _encode_chunk(self, text: str) -> list[int]:
        """对不含 special token 的一段文本编码（GPT-2 正则预分词 + 逐预 token 合并）。"""
        ids: list[int] = []
        for match in re.finditer(PAT, text):
            token_bytes = match.group().encode("utf-8")
            for part in self._apply_merges(token_bytes):
                ids.append(self.byte_to_id[part])
        return ids

    def encode(self, text: str) -> list[int]:
        """把文本编码成 token id 列表。"""
        if not self.special_tokens:
            return self._encode_chunk(text)

        # 按长度降序排序：保证重叠时优先匹配更长的 special token（如
        # "<|endoftext|><|endoftext|>" 先于 "<|endoftext|>"）。
        specials = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "(" + "|".join(re.escape(s) for s in specials) + ")"
        ids: list[int] = []
        for segment in re.split(pattern, text):
            if segment == "":
                continue
            if segment in self.special_tokens:
                ids.append(self.byte_to_id[segment.encode("utf-8")])
            else:
                ids.extend(self._encode_chunk(segment))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """惰性编码：逐块（如文件按行）产出 id，内存占用与文件大小无关。"""
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        """把 token id 列表解码回文本；非法字节用 U+FFFD 替换。"""
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")
