# `cs336_basics` 代码组织与模块职责

## 1. 目标

Assignment 1 最初按测试顺序实现，模型、优化器、数据和 checkpoint 曾集中在 `nn.py`。随着功能完整，这个文件同时承担多种职责，不利于 Assignment 2 复用和后续维护。现在代码已按职责完成拆分，**真实实现只存在一份，旧 `nn.py` 已删除**。

## 2. 规范模块

| 模块 | 职责 | 主要 interface |
|---|---|---|
| `model.py` | Transformer 模型及构建模块 | `Linear`、`Embedding`、`RMSNorm`、`SwiGLU`、RoPE、Attention、`TransformerBlock`、`TransformerLM` |
| `nn_utils.py` | 无状态数值函数 | `softmax`、`cross_entropy` |
| `optimizer.py` | 参数更新与训练数值控制 | `AdamW`、余弦 LR、梯度裁剪 |
| `data.py` | 语言模型批采样 | `get_batch` |
| `checkpoint.py` | 持久化训练状态 | `save_checkpoint`、`load_checkpoint` |
| `bpe.py` | BPE 训练与文本编解码 | `train_bpe`、`Tokenizer` |
| `train.py` | 组合以上模块完成训练 | CLI `cs336-train` |
| `generate.py` | 加载 checkpoint 并自回归生成 | CLI `cs336-generate` |
| `encode_corpus.py` | 把文本预编码成 token 数组 | 语料预处理 CLI |

依赖方向保持单向：

```text
nn_utils <- model

model + nn_utils + optimizer + data + checkpoint <- train
model + checkpoint + bpe <- generate
bpe <- encode_corpus
```

底层模块互不依赖，组合只发生在脚本层，因此没有循环导入。

## 3. 为什么不保留 `nn.py`

项目内调用已全部迁移到规范模块：

```python
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.data import get_batch
```

继续保留 `nn.py` 即使只做重导出，也会形成第二个公共入口：调用者仍需判断“应该从 `nn` 还是职责模块导入”，让旧结构长期存在。既然 Assignment 1 adapters、训练/生成脚本和 Assignment 2 均已迁移，就直接删除旧入口，使错误导入尽早失败，并确保规范模块是唯一 interface。

## 4. 对 Assignment 2 的意义

Assignment 2 把 `cs336_basics.model` 当作稳定 seam，例如：

```python
from cs336_basics.model import Embedding, Linear, RMSNorm
```

现在 `model.py` 已是模型的真实实现位置，而不是转发到一个混合职责文件；对 `scaled_dot_product_attention` 的 monkey-patch 也会直接作用于 `MultiHeadSelfAttention` 前向使用的同一模块全局符号。

需要单独注意：Assignment 2 handout 部分示例使用 `RotaryEmbedding` 和接收 `positional_encoder` 的 `TransformerBlock`，其构造 interface 与本仓库 Assignment 1 实现不同。这是**接口差异**，不是文件组织问题；做到对应实验时应写显式 adapter，而不能只做名称别名。

## 5. 维护约定

1. 模型结构只写进 `model.py`。
2. 无状态损失/归一化函数写进 `nn_utils.py`。
3. 优化器、LR 和梯度处理写进 `optimizer.py`。
4. 数据采样和 checkpoint 不放进模型模块。
5. 调用方直接依赖规范模块，不重新引入 `nn.py` 聚合入口。
