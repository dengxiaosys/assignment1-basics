"""在 TinyStories（或任意语料）上训练 BPE 并报告实验指标。

对应 handout Problem `train_bpe_tinystories`：
  - 训练一个 vocab_size=10000 的字节级 BPE，加入 <|endoftext|> special token；
  - 序列化 vocab / merges 到磁盘（供后续 tokenizer 实验加载）；
  - 报告训练耗时、峰值内存、最长 token（并判断是否合理）；
  - 简单 profile：预分词 vs 合并各花多少时间。

用法：
    uv run python -m cs336_basics.train_bpe_experiment \
        --input data/TinyStoriesV2-GPT4-train.txt \
        --vocab-size 10000 --num-processes 8 \
        --out-dir bpe_out/tinystories
"""

import argparse
import json
import os
import time

from cs336_basics.bpe import gpt2_bytes_to_unicode_safe, train_bpe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a byte-level BPE and report metrics.")
    p.add_argument("--input", required=True, help="训练语料路径（纯文本）")
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    p.add_argument("--num-processes", type=int, default=1, help=">1 时并行预分词")
    p.add_argument("--out-dir", default="bpe_out", help="vocab/merges 输出目录")
    return p.parse_args()


def peak_rss_bytes() -> int:
    """当前进程的峰值常驻内存（字节）。Linux 上用 resource.ru_maxrss(KB)。"""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def serialize(vocab, merges, out_dir: str) -> tuple[str, str]:
    """把 vocab / merges 存盘（vocab.json + merges.txt）。

    用 GPT-2 的"字节→可打印字符"映射把 bytes 编码成可读字符串，便于人工检视，
    也与测试/tiktoken 的存储格式一致。
    """
    os.makedirs(out_dir, exist_ok=True)
    b2u = gpt2_bytes_to_unicode_safe()

    def enc(bs: bytes) -> str:
        return "".join(b2u[x] for x in bs)

    vocab_path = os.path.join(out_dir, "vocab.json")
    merges_path = os.path.join(out_dir, "merges.txt")
    # vocab.json: {token_str: id}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({enc(b): i for i, b in vocab.items()}, f, ensure_ascii=False)
    # merges.txt: 每行 "tok1 tok2"
    with open(merges_path, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{enc(a)} {enc(b)}\n")
    return vocab_path, merges_path


def main() -> None:
    args = parse_args()
    corpus_bytes = os.path.getsize(args.input)
    print(f"bpe_train_start: input={args.input} size_bytes={corpus_bytes} "
          f"vocab_size={args.vocab_size} num_processes={args.num_processes}")

    t0 = time.time()
    vocab, merges = train_bpe(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        num_processes=args.num_processes,
    )
    elapsed = time.time() - t0

    # 最长 token（排除 special tokens，看的是学出来的合并 token）
    special_bytes = {s.encode("utf-8") for s in args.special_tokens}
    longest = max((b for b in vocab.values() if b not in special_bytes), key=len)

    vocab_path, merges_path = serialize(vocab, merges, args.out_dir)

    print(f"bpe_train_done: elapsed_sec={elapsed:.2f} peak_rss_gb={peak_rss_bytes() / 1e9:.2f}")
    print(f"vocab_final_size={len(vocab)} num_merges={len(merges)}")
    print(f"longest_token_len={len(longest)} longest_token_repr={longest!r}")
    print(f"serialized: vocab={vocab_path} merges={merges_path}")


if __name__ == "__main__":
    main()
