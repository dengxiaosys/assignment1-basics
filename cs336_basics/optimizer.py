"""优化器、学习率调度与梯度处理。"""

import math

import torch


class AdamW(torch.optim.Optimizer):
    """按 handout Algorithm 1 实现的 AdamW。"""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                m, v = state["m"], state["v"]
                t = state["t"] + 1
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data.mul_(1 - lr * weight_decay)
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-lr_t)
                state["t"] = t
        return loss


def get_lr_cosine_schedule(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters):
    """计算带线性 warmup 的余弦退火学习率。"""
    if it < warmup_iters:
        return it / warmup_iters * max_lr
    if it <= cosine_cycle_iters:
        coeff = 0.5 * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi))
        return min_lr + coeff * (max_lr - min_lr)
    return min_lr


def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    """按所有梯度的全局 L2 范数就地裁剪。"""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm = torch.sqrt(sum((g**2).sum() for g in grads))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for grad in grads:
            grad.mul_(scale)


__all__ = ["AdamW", "get_lr_cosine_schedule", "gradient_clipping"]
