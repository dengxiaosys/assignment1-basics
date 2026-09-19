import math

import torch
from torch import Tensor, nn


class Linear(nn.Module):
    """无 bias 的线性层，遵循 nn.Linear 接口。

    数学上（列向量约定）为 y = W x；在 PyTorch 中特征位于最后一维，
    因此实现为 y = x W^T，其中 weight 形状为 (d_out, d_in)。

    uv run pytest -k test_linear
    """

    def __init__(self, d_in: int, d_out: int, device=None, dtype=None):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.weight = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 截断正态初始化，方差 2/(d_in+d_out)，截断在 ±3σ。
        std = math.sqrt(2.0 / (self.d_in + self.d_out))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: Tensor) -> Tensor:
        # 对最后一维做 x @ W^T，保留任意前置批量维。
        return x @ self.weight.transpose(-2, -1)
