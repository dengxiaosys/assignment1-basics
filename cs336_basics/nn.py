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


class MultiHeadSelfAttention(nn.Module):
    """因果多头自注意力（Vaswani et al. 2017, §3.2.2）。

    QKV 用单个大矩阵一次投影，再拆成 num_heads 个头并行做 SDPA，
    最后合头并过输出投影。可选传入 rope 对每个头的 Q/K 施加旋转。
    """

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
        # 单次大矩阵投影，(..., seq, d_model)
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 拆头：(..., seq, d_model) -> (..., num_heads, seq, head_dim)
        def split_heads(t: Tensor) -> Tensor:
            return t.reshape(*batch, seq, self.num_heads, self.head_dim).transpose(-3, -2)
        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)
        # 可选 RoPE：只转 Q/K（逐头）
        if rope is not None:
            Q = rope(Q, token_positions)
            K = rope(K, token_positions)
        # 因果掩码：下三角 True，表示只能看自己和之前
        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=x.device))
        # SDPA：(..., num_heads, seq, head_dim)
        out = scaled_dot_product_attention(Q, K, V, mask)
        # 合头：(..., num_heads, seq, head_dim) -> (..., seq, d_model)
        out = out.transpose(-3, -2).reshape(*batch, seq, self.d_model)
        return self.output_proj(out)


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer block（含 RoPE）。

    y = x + MHA(RMSNorm(x))          # 注意力子层
    out = y + FFN(RMSNorm(y))        # 前馈子层
    归一化放在子层之前（Pre-Norm），残差绕过归一化直连。
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 max_seq_len: int, theta: float, device=None, dtype=None):
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
        # Pre-Norm：先归一化再进子层，残差直连原始输入
        x = x + self.attn(self.ln1(x), token_positions=token_positions, rope=self.rope)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """Decoder-only Transformer 语言模型。

    token_embeddings -> N x TransformerBlock -> ln_final (RMSNorm) -> lm_head (Linear)
    输出未归一化的 logits (.., seq, vocab_size)。
    """

    def __init__(self, vocab_size: int, context_length: int, d_model: int,
                 num_layers: int, num_heads: int, d_ff: int, rope_theta: float,
                 device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta,
                             device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: Tensor) -> Tensor:
        x = self.token_embeddings(token_ids)   # (.., seq, d_model)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)                  # (.., seq, vocab_size)


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """平均交叉熵损失。

    inputs: (N, vocab) 未归一化 logits；targets: (N,) 正确类别索引。
    用 log-sum-exp 稳定：loss_i = -logit[target] + logsumexp(logits)。
    """
    # 数值稳定：减去每行最大值（不改变 softmax/交叉熵的值）。
    x_max = inputs.max(dim=-1, keepdim=True).values
    shifted = inputs - x_max
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1))  # (N,)
    # 取每个样本目标类的 logit（同样减了 max）
    target_logit = shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # (N,)
    loss = log_sum_exp - target_logit  # 每样本 -log p
    return loss.mean()


class AdamW(torch.optim.Optimizer):
    """AdamW（Loshchilov & Hutter 2019），按 handout Algorithm 1 实现。

    与 Adam 的区别：权重衰减 lambda 与梯度更新解耦——直接对参数做 theta <- theta - lr*lambda*theta，
    而非把 L2 正则加进梯度。每个参数维护一阶矩 m、二阶矩 v 和步数 t。
    """

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
                # 初始化状态
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                m, v = state["m"], state["v"]
                t = state["t"] + 1
                # 一阶/二阶矩估计
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # 偏差校正后的学习率
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                # 解耦权重衰减
                p.data.mul_(1 - lr * weight_decay)
                # 矩调整的参数更新
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-lr_t)
                state["t"] = t
        return loss


def get_lr_cosine_schedule(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters):
    """带线性 warmup 的余弦退火学习率调度。

    - warmup (it < T_w):       lr = it/T_w * max_lr
    - cosine (T_w<=it<=T_c):   lr = min_lr + 0.5*(1+cos((it-T_w)/(T_c-T_w)*pi))*(max_lr-min_lr)
    - post   (it > T_c):       lr = min_lr
    """
    if it < warmup_iters:
        return it / warmup_iters * max_lr
    if it <= cosine_cycle_iters:
        coeff = 0.5 * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi))
        return min_lr + coeff * (max_lr - min_lr)
    return min_lr
