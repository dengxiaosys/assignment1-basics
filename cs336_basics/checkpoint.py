"""训练 checkpoint 的保存与恢复。"""

import torch


def save_checkpoint(model, optimizer, iteration, out):
    """保存模型、优化器状态与已完成的迭代数。"""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer=None, map_location=None):
    """恢复模型，按需恢复优化器，并返回保存时的迭代数。"""
    checkpoint = torch.load(src, map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
