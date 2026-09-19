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


class Embedding(nn.Module):
    """查表式词嵌入，遵循 nn.Embedding 接口（无 padding_idx 等附加特性）。

    weight 形状为 (num_embeddings, embedding_dim)：第 i 行是 id=i 的向量。
    forward 用整型 token_ids 按行索引，输出形状为 token_ids.shape + (embedding_dim,)。
    
    uv run pytest -k test_embedding
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 词向量常用标准正态初始化（截断在 ±3）。
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        # 按行查表：等价于 self.weight[token_ids]，保留 token_ids 的任意形状。
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """RMSNorm：只做均方根缩放，无均值中心化、无 bias。

    对最后一维 (d_model) 归一化：
        y_i = g_i * x_i / sqrt(mean(x^2) + eps)
    其中 g（weight，形状 (d_model,)）是每个 RMSNorm 块独立的逐通道增益。
    为数值稳定，均方根在 float32 上计算，再转回原 dtype。
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # 沿最后一维求均方根，保持维度以便广播。
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = x / rms * self.weight
        return y.to(in_dtype)
