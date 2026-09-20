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


def silu(x: Tensor) -> Tensor:
    """SiLU / Swish: x * sigmoid(x)。逐元素、无参数。"""
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """SwiGLU 前馈网络（无 bias）。

    FFN(x) = W2 ( SiLU(W1 x) ⊙ (W3 x) )
    其中 W1、W3 形状 (d_ff, d_model)（升维），W2 形状 (d_model, d_ff)（降维），
    ⊙ 为逐元素相乘。W1 为门支路（过 SiLU），W3 为值支路。
    """

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    """RoPE：对 Q/K 按位置旋转，只作用于相邻二维对。

    频率 theta_i = base^(-2i/d)，i=0..d/2-1（base 即 theta，常取 10000）。
    对每对 (x_{2i}, x_{2i+1})，在位置 pos 处旋转角 pos*theta_i：
        x'_{2i}   = x_{2i} cos - x_{2i+1} sin
        x'_{2i+1} = x_{2i} sin + x_{2i+1} cos
    cos/sin 预计算并登记为 buffer（不可训练、随模型走）。
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, "d_k 必须为偶数（按二维对分组）"
        self.d_k = d_k
        # 每对的频率：(d_k/2,)
        inv_freq = theta ** (-torch.arange(0, d_k, 2, device=device).float() / d_k)
        # 位置 × 频率 -> 角度表 (max_seq_len, d_k/2)
        pos = torch.arange(max_seq_len, device=device).float()
        angles = torch.outer(pos, inv_freq)
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        # 按位置取角度：cos/sin 形状 (..., seq, d_k/2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        # 拆成相邻二维对的偶/奇分量
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        # 逐对旋转
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        # 交错拼回原顺序 (..., d_k)
        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out


def softmax(x: Tensor, dim: int) -> Tensor:
    # 数值稳定：先减去该维最大值，避免 exp 溢出（softmax 平移不变）。
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None
) -> Tensor:
    """缩放点积注意力：softmax(QK^T / sqrt(d_k) + mask_bias) V。

    Q: (..., queries, d_k)  K: (..., keys, d_k)  V: (..., keys, d_v)
    mask: (..., queries, keys) 布尔，True=参与注意力，False=屏蔽（置 -inf）。
    """
    d_k = Q.shape[-1]
    # 打分并缩放：(..., queries, keys)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        # False 的位置置 -inf，softmax 后权重趋于 0。
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = softmax(scores, dim=-1)
    return attn @ V
