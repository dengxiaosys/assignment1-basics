"""无状态的神经网络数值函数。"""

import torch
from torch import Tensor


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """沿指定维度计算数值稳定的 softmax。"""
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """用稳定的 log-sum-exp 形式计算平均交叉熵。"""
    x_max = inputs.max(dim=-1, keepdim=True).values
    shifted = inputs - x_max
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1))
    target_logit = shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (log_sum_exp - target_logit).mean()
