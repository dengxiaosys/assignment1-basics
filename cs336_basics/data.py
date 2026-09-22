"""语言模型训练数据的批采样。"""

import numpy as np
import torch


def get_batch(dataset, batch_size, context_length, device):
    """从一维 token 数组随机采样语言建模的输入与右移标签。"""
    max_start = len(dataset) - context_length
    starts = np.random.randint(0, max_start, size=batch_size)
    x = np.stack([dataset[i : i + context_length] for i in starts])
    y = np.stack([dataset[i + 1 : i + 1 + context_length] for i in starts])
    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    return x, y
