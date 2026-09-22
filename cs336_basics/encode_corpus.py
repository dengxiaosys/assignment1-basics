"""把文本语料用训练好的 BPE 编码成 token id 数组（.npy），供 train.py 加载。

打通 "文本 .txt → token .npy → get_batch → 训练" 链路里缺的一环：train.py 的
load_tokens 需要一维 token-id 数组（.npy/.bin），而磁盘上只有原始文本。本脚本
用 Tokenizer.encode_iterable 流式编码（内存占用与文件大小无关），把结果存成
uint16 的 .npy（vocab < 65536 时 2 字节/token 足够）。

用法：
    uv run python -m cs336_basics.encode_corpus \
        --vocab bpe_out/tinystories/vocab.json \
        --merges bpe_out/tinystories/merges.txt \
        --special-tokens "<|endoftext|>" \
        --input data/TinyStoriesV2-GPT4-train.txt \
        --output data/ts_train.npy
"""

import argparse
import os
import time

import numpy as np

from cs336_basics.bpe import Tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Encode a text corpus into a token-id .npy using a trained BPE.")
    p.add_argument("--vocab", required=True, help="BPE vocab.json 路径")
    p.add_argument("--merges", required=True, help="BPE merges.txt 路径")
    p.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    p.add_argument("--input", required=True, help="待编码的文本文件（.txt）")
    p.add_argument("--output", required=True, help="输出的 token-id 数组（.npy）")
    p.add_argument("--dtype", default="uint16", help="token id 的存储 dtype（vocab<65536 用 uint16）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    corpus_bytes = os.path.getsize(args.input)
    print(f"encode_start: input={args.input} size_bytes={corpus_bytes} "
          f"vocab={args.vocab} dtype={args.dtype}")

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_tokens)

    # 流式编码：逐行喂给 encode_iterable，边产出边收集，避免把整份语料读进内存。
    t0 = time.time()
    ids: list[int] = []
    with open(args.input, encoding="utf-8") as f:
        for token_id in tokenizer.encode_iterable(f):
            ids.append(token_id)
    elapsed = time.time() - t0

    arr = np.array(ids, dtype=np.dtype(args.dtype))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, arr)

    num_tokens = len(arr)
    ratio = corpus_bytes / num_tokens if num_tokens else float("nan")
    throughput = corpus_bytes / elapsed if elapsed else float("nan")
    print(f"encode_done: num_tokens={num_tokens} elapsed_sec={elapsed:.2f}")
    print(f"compression_ratio_bytes_per_token={ratio:.3f}")
    print(f"throughput_bytes_per_sec={throughput:.0f}")
    print(f"saved: output={args.output} dtype={arr.dtype} shape={arr.shape}")


if __name__ == "__main__":
    main()
