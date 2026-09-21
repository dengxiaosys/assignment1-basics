"""端到端训练脚本：把 assignment1 实现的所有组件串成一个可配置的训练循环。

覆盖 handout `training_together` 的四点交付要求：
  1. 模型/优化器超参可通过命令行配置（argparse）。
  2. 用 np.memmap 内存高效地加载大 train/val 数据（token id 的二进制 .npy/.bin）。
  3. checkpoint 序列化到用户指定路径（支持从 checkpoint 恢复续训）。
  4. 周期性地把 train/val 表现打到控制台（可选简单的 jsonl 日志文件）。

数据约定：train/val 是一维 uint16/int32 的 token id 序列，存成 .npy 或原始 .bin。
用法示例（小规模冒烟）：
    uv run python -m cs336_basics.train \
        --train-path data/ts_train.npy --val-path data/ts_val.npy \
        --vocab-size 10000 --context-length 128 --d-model 128 --num-layers 2 \
        --num-heads 4 --d-ff 512 --batch-size 16 --total-iters 100 \
        --eval-interval 50 --checkpoint-out out/ckpt.pt
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from cs336_basics.nn import (
    AdamW,
    TransformerLM,
    cross_entropy,
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    """所有模型/优化器/训练超参都做成命令行参数（交付点 1）。"""
    p = argparse.ArgumentParser(description="Train a Transformer LM (CS336 assignment1).")

    # --- 数据 ---
    p.add_argument("--train-path", type=str, required=True, help="1D token id 数组（.npy 或 .bin）")
    p.add_argument("--val-path", type=str, default=None, help="验证集 token id 数组（可选）")
    p.add_argument("--dtype-tokens", type=str, default="uint16", help="token 数组的 dtype（.bin 时用）")

    # --- 模型结构 ---
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--rope-theta", type=float, default=10000.0)

    # --- 优化器 / LR 调度 ---
    p.add_argument("--lr-max", type=float, default=3e-4, help="峰值学习率")
    p.add_argument("--lr-min", type=float, default=3e-5, help="余弦退火底值")
    p.add_argument("--warmup-iters", type=int, default=200)
    p.add_argument("--cosine-iters", type=int, default=None, help="余弦周期末步；默认=total-iters")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=1.0, help="全局 L2 范数裁剪阈值；<=0 关闭")

    # --- 训练流程 ---
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--total-iters", type=int, default=5000)
    p.add_argument("--eval-interval", type=int, default=200, help="每多少步做一次 val 评估+日志")
    p.add_argument("--eval-batches", type=int, default=20, help="每次评估平均多少个 batch")
    p.add_argument("--log-interval", type=int, default=20, help="每多少步打一次 train loss")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)

    # --- checkpoint ---
    p.add_argument("--checkpoint-out", type=str, default=None, help="checkpoint 保存路径")
    p.add_argument("--checkpoint-interval", type=int, default=1000, help="每多少步存一次 checkpoint")
    p.add_argument("--resume-from", type=str, default=None, help="从该 checkpoint 恢复续训")
    p.add_argument("--log-file", type=str, default=None, help="可选：把指标追加写入的 jsonl 文件")

    return p.parse_args()


def load_tokens(path: str, dtype: str) -> np.ndarray:
    """用 np.memmap 内存高效加载 token 数组（交付点 2）。

    - .npy：np.load(mmap_mode="r") 直接返回磁盘映射，不载入内存；
    - 其它（原始 .bin）：np.memmap 按给定 dtype 映射。
    两种情况都只在 get_batch 切片时按需从磁盘读取用到的那几段。
    """
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    return np.memmap(path, dtype=np.dtype(dtype), mode="r")


@torch.no_grad()
def evaluate(model, data, args) -> float:
    """在 val 上取若干 batch，返回平均交叉熵损失。"""
    model.eval()
    losses = []
    for _ in range(args.eval_batches):
        x, y = get_batch(data, args.batch_size, args.context_length, args.device)
        logits = model(x)  # (B, seq, vocab)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def log_metrics(record: dict, log_file: str | None) -> None:
    """打到控制台；若指定了 log_file 则同时追加一行 jsonl。"""
    msg = "  ".join(f"{k}={v}" for k, v in record.items())
    print(f"train_log: {msg}")
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        with open(log_file, "a") as f:
            f.write(json.dumps(record) + "\n")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.cosine_iters is None:
        args.cosine_iters = args.total_iters

    # --- 数据（memmap）---
    train_data = load_tokens(args.train_path, args.dtype_tokens)
    val_data = load_tokens(args.val_path, args.dtype_tokens) if args.val_path else None
    print(f"data_loaded: train_tokens={len(train_data)} "
          f"val_tokens={len(val_data) if val_data is not None else 0}")

    # --- 模型 ---
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
    model.to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model_built: num_params={n_params}")

    # --- 优化器 ---
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # --- 从 checkpoint 恢复（交付点 3）---
    start_iter = 0
    if args.resume_from is not None and os.path.exists(args.resume_from):
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        print(f"resumed: from={args.resume_from} start_iter={start_iter}")

    # --- 训练循环 ---
    model.train()
    t0 = time.time()
    for it in range(start_iter, args.total_iters):
        # 1) 按当前步计算学习率（cosine + warmup），写回每个 param group
        lr = get_lr_cosine_schedule(it, args.lr_max, args.lr_min, args.warmup_iters, args.cosine_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # 2) 采一个 batch，前向 + 反向
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        logits = model(x)  # (B, seq, vocab)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()

        # 3) 梯度裁剪（可选）后更新
        if args.grad_clip and args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        # 4) 周期性日志（交付点 4）
        if it % args.log_interval == 0:
            log_metrics({"iter": it, "train_loss": round(loss.item(), 4),
                         "lr": round(lr, 6), "sec": round(time.time() - t0, 1)}, args.log_file)

        if val_data is not None and args.eval_interval > 0 and it > 0 and it % args.eval_interval == 0:
            val_loss = evaluate(model, val_data, args)
            log_metrics({"iter": it, "val_loss": round(val_loss, 4)}, args.log_file)

        # 5) 周期性保存 checkpoint
        if args.checkpoint_out and args.checkpoint_interval > 0 and it > 0 \
                and it % args.checkpoint_interval == 0:
            os.makedirs(os.path.dirname(args.checkpoint_out) or ".", exist_ok=True)
            save_checkpoint(model, optimizer, it, args.checkpoint_out)
            print(f"checkpoint_saved: iter={it} path={args.checkpoint_out}")

    # --- 收尾：存最终 checkpoint ---
    if args.checkpoint_out:
        os.makedirs(os.path.dirname(args.checkpoint_out) or ".", exist_ok=True)
        save_checkpoint(model, optimizer, args.total_iters, args.checkpoint_out)
        print(f"checkpoint_saved: iter={args.total_iters} path={args.checkpoint_out} (final)")
    print("training_done: total_sec=" + str(round(time.time() - t0, 1)))


if __name__ == "__main__":
    main()
