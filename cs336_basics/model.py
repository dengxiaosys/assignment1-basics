"""Transformer 语言模型及其构建模块。"""

import math

import torch
from torch import Tensor, nn

from cs336_basics.nn_utils import softmax


class Linear(nn.Module):
    """无 bias 的线性层，遵循 nn.Linear 接口。

    数学上（列向量约定）为 y = W x；在 PyTorch 中特征位于最后一维，
    因此实现为 y = x W^T，其中 weight 形状为 (d_out, d_in)。
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
        # 按行查表，保留 token_ids 的任意形状。
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """RMSNorm：只做均方根缩放，无均值中心化、无 bias。"""

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = x / rms * self.weight
        return y.to(in_dtype)


def silu(x: Tensor) -> Tensor:
    """SiLU / Swish: x * sigmoid(x)。逐元素、无参数。"""
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """SwiGLU 前馈网络（无 bias）。"""

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
    """RoPE：对 Q/K 按位置旋转，只作用于相邻二维对。"""

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, "d_k 必须为偶数（按二维对分组）"
        self.d_k = d_k
        inv_freq = theta ** (-torch.arange(0, d_k, 2, device=device).float() / d_k)
        pos = torch.arange(max_seq_len, device=device).float()
        angles = torch.outer(pos, inv_freq)
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """缩放点积注意力：softmax(QK^T / sqrt(d_k) + mask_bias) V。"""
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = softmax(scores, dim=-1)
    return attn @ V


class MultiHeadSelfAttention(nn.Module):
    """因果多头自注意力，可选对每个头的 Q/K 应用 RoPE。"""

    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None, rope=None) -> Tensor:
        *batch, seq, _ = x.shape
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        def split_heads(t: Tensor) -> Tensor:
            return t.reshape(*batch, seq, self.num_heads, self.head_dim).transpose(-3, -2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)
        if rope is not None:
            Q = rope(Q, token_positions)
            K = rope(K, token_positions)
        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=x.device))
        out = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(-3, -2).reshape(*batch, seq, self.d_model)
        return self.output_proj(out)


class TransformerBlock(nn.Module):
    """含 RoPE 的 Pre-Norm Transformer block。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device=device)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if token_positions is None:
            seq = x.shape[-2]
            token_positions = torch.arange(seq, device=x.device)
        x = x + self.attn(self.ln1(x), token_positions=token_positions, rope=self.rope)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """Decoder-only Transformer 语言模型，输出未归一化 logits。"""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: Tensor) -> Tensor:
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)


__all__ = [
    "Embedding",
    "Linear",
    "MultiHeadSelfAttention",
    "RMSNorm",
    "RotaryPositionalEmbedding",
    "SwiGLU",
    "TransformerBlock",
    "TransformerLM",
    "scaled_dot_product_attention",
    "silu",
]
