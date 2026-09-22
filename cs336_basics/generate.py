"""从 TransformerLM checkpoint 自回归生成文本。

已有 checkpoint 只保存 model / optimizer / iteration，不保存模型结构和 tokenizer 路径；
因此本脚本要求调用方显式传入训练时的模型超参及对应的 BPE vocab/merges。

示例（对应本地 ts_valid.pt 的训练配置）：
    uv run python -m cs336_basics.generate \
        --checkpoint ckpt/ts_valid.pt \
        --vocab bpe_out/tinystories/vocab.json \
        --merges bpe_out/tinystories/merges.txt \
        --prompt "Once upon a time" \
        --vocab-size 10000 --context-length 128 --d-model 256 \
        --num-layers 4 --num-heads 8 --d-ff 1024 --device cpu
"""

import argparse

import torch

from cs336_basics.bpe import Tokenizer
from cs336_basics.nn import TransformerLM, load_checkpoint


def parse_args() -> argparse.Namespace:
    """解析 checkpoint、tokenizer、模型结构和采样超参。"""
    parser = argparse.ArgumentParser(description="Generate text from a TransformerLM checkpoint.")

    # --- checkpoint / tokenizer ---
    parser.add_argument("--checkpoint", required=True, help="要加载的 .pt checkpoint")
    parser.add_argument("--vocab", required=True, help="对应 BPE 的 vocab.json")
    parser.add_argument("--merges", required=True, help="对应 BPE 的 merges.txt")
    parser.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    parser.add_argument("--eos-token", default="<|endoftext|>", help="生成到此 special token 时停止")

    # --- 模型结构：必须与 checkpoint 训练时完全一致 ---
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--d-ff", type=int, required=True)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # --- prompt / 采样 ---
    parser.add_argument("--prompt", default="<|endoftext|>", help="用于续写的文本前缀")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0, help="0 表示贪心 argmax")
    parser.add_argument("--top-p", type=float, default=1.0, help="nucleus sampling 阈值，(0, 1]")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    """从下一 token 的 logits 采样一个 id，支持 temperature 与 nucleus（top-p）采样。"""
    if temperature == 0:
        return int(torch.argmax(logits).item())

    probabilities = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_ids = torch.sort(probabilities, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # 保留累计概率首次达到 top_p 的 token，以及它之前的所有 token。
        cutoff = int(
            torch.searchsorted(
                cumulative_probs,
                torch.tensor(top_p, device=probabilities.device),
                right=False,
            ).item()
        ) + 1
        candidate_probs = sorted_probs[:cutoff]
        candidate_ids = sorted_ids[:cutoff]
        sampled_index = torch.multinomial(candidate_probs / candidate_probs.sum(), num_samples=1)
        return int(candidate_ids[sampled_index].item())

    return int(torch.multinomial(probabilities, num_samples=1).item())


@torch.inference_mode()
def generate(
    model: TransformerLM,
    prompt_ids: list[int],
    eos_id: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
    context_length: int,
) -> list[int]:
    """从 prompt_ids 自回归采样，并返回只含新生成内容的 token id。"""
    all_ids = prompt_ids.copy()
    completion_ids: list[int] = []

    for _ in range(max_new_tokens):
        # 模型的 RoPE 表只预计算到 context_length；长提示/长生成时只保留最近窗口。
        context_ids = all_ids[-context_length:]
        context = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)
        next_logits = model(context)[0, -1]
        next_id = sample_next_token(next_logits, temperature, top_p)

        all_ids.append(next_id)
        completion_ids.append(next_id)
        if next_id == eos_id:
            break

    return completion_ids


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be >= 1")
    if args.temperature < 0:
        raise ValueError("--temperature must be >= 0")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")

    torch.manual_seed(args.seed)
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_tokens)
    eos_bytes = args.eos_token.encode("utf-8")
    if eos_bytes not in tokenizer.byte_to_id:
        raise ValueError(f"eos_token_missing={args.eos_token!r}")

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise ValueError("prompt_must_encode_to_at_least_one_token=True")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    )
    iteration = load_checkpoint(args.checkpoint, model, optimizer=None, map_location=args.device)
    model.eval()

    print(
        f"generation_loaded=checkpoint:{args.checkpoint} iteration:{iteration} "
        f"device:{args.device} prompt_tokens:{len(prompt_ids)}"
    )
    completion_ids = generate(
        model=model,
        prompt_ids=prompt_ids,
        eos_id=tokenizer.byte_to_id[eos_bytes],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
        context_length=args.context_length,
    )
    # 停止用的 EOS 是控制标记，不作为用户可见的续写文本输出。
    if completion_ids and completion_ids[-1] == tokenizer.byte_to_id[eos_bytes]:
        completion_ids.pop()

    print(f"generation_completion_tokens={len(completion_ids)}")
    print(f"generation_output={tokenizer.decode(completion_ids)}")


if __name__ == "__main__":
    main()
