# 组装完整训练脚本（training loop）：原理与设计

## 0. 本文目标

CS336 assignment1 到这里，所有零件都造好了：模型、损失、优化器、学习率调度、梯度裁剪、数据批采样、checkpoint。本题（handout `training_together`）要求把它们**拼成一个能真正训练的脚本**。本文讲清这个训练循环的骨架、handout 的四点交付要求各自怎么落地、以及每个设计决定背后的原因。它是前面一系列"单组件笔记"的**总集篇**。

对应实际代码：
- 训练脚本：[cs336_basics/train.py](../cs336_basics/train.py)
- 入口：`pyproject.toml` 的 `[project.scripts]` → `cs336-train`
- handout：`training_together`（§5.3 Training loop）

前置（本文会反复引用）：[数据加载](./03_08_data_loading_implementation_notes.md)、[交叉熵](./03_01_cross_entropy_implementation_notes.md)、[AdamW](./03_04_adamw_implementation_notes.md)、[LR 调度](./03_06_lr_schedule_implementation_notes.md)、[梯度裁剪](./03_07_gradient_clipping_implementation_notes.md)、[checkpoint](./03_09_checkpointing_implementation_notes.md)、[优化器 API](./03_02_pytorch_optimizer_api_notes.md)。

---

## 1. 大图：一个训练循环长什么样

训练脚本的核心就是一个 for 循环，每一步（iteration）做固定几件事。用伪代码勾勒：

```text
搭好 model / optimizer（按命令行超参）
可选：从 checkpoint 恢复 -> 得到 start_iter
for it in range(start_iter, total_iters):
    lr = 调度(it)            # 按当前步算学习率，写回 optimizer
    x, y = get_batch(...)    # 采一个 batch
    logits = model(x)
    loss = cross_entropy(logits, y)
    optimizer.zero_grad(); loss.backward()
    gradient_clipping(...)   # 裁剪梯度（可选）
    optimizer.step()         # 迈一步
    周期性：打 train/val 日志、存 checkpoint
```

这正是 [优化器 API 笔记](./03_02_pytorch_optimizer_api_notes.md) 里"一个 step = zero_grad → forward → backward → step"的放大版，外面再套上**学习率调度、梯度裁剪、日志、checkpoint** 四件外围事务。下面按 handout 的四点交付要求逐一说明。

---

## 2. 交付点 1：超参可配置（argparse）

handout 强调"要方便地用不同超参启动训练（例如做成命令行参数），因为后面要反复调"。所以脚本用 `argparse` 把**所有**会影响训练的量都暴露成命令行选项，分四组：

- **数据**：`--train-path` / `--val-path` / `--dtype-tokens`；
- **模型结构**：`--vocab-size` / `--context-length` / `--d-model` / `--num-layers` / `--num-heads` / `--d-ff` / `--rope-theta`（对应 [TransformerLM 笔记](./02_14_transformer_lm_implementation_notes.md) 的构造签名）；
- **优化器 / LR**：`--lr-max` / `--lr-min` / `--warmup-iters` / `--cosine-iters` / `--weight-decay` / `--beta1` / `--beta2` / `--eps` / `--grad-clip`；
- **流程 / checkpoint**：`--batch-size` / `--total-iters` / `--eval-interval` / `--log-interval` / `--device` / `--seed` / `--checkpoint-out` / `--checkpoint-interval` / `--resume-from` / `--log-file`。

**为什么全做成参数而不写死**：语言模型训练是一门"调参的实证科学"，后续作业要系统地扫 lr、batch、层数、上下文长度看它们如何影响 loss。把超参外置，就能用 shell 脚本批量起不同配置的 run，而不必改代码。`--seed` 固定随机性便于对比；`--device` 让同一脚本在 CPU（本地调试）和 GPU（真训）间无缝切换。

> 小细节：`--cosine-iters` 默认等于 `--total-iters`（余弦周期铺满整个训练），这是最常用的设置；需要时可单独指定。约束 $T_w < T_c$ 见 [LR 调度笔记](./03_06_lr_schedule_implementation_notes.md)。

---

## 3. 交付点 2：用 `np.memmap` 内存高效加载大数据

编码后的语料可能几 GB 到几十 GB（我们的 OWT train 解压后 12G），**一次性 `np.load` 进内存不现实**。解决办法是**内存映射**：

```python
def load_tokens(path, dtype):
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r")   # .npy：带 header，直接映射
    return np.memmap(path, dtype=dtype, mode="r")  # 原始 .bin：按 dtype 映射
```

- **memmap 的本质**：把磁盘文件"假装"成一个 numpy 数组，但**不把数据读进内存**；只有当你切片访问某几段时，操作系统才按页把那部分从磁盘调入。所以 `get_batch` 每步只随机切 `batch_size × context_length` 个 token，**内存占用与数据集总大小无关**。
- **为什么能直接对接 `get_batch`**：见 [数据加载笔记](./03_08_data_loading_implementation_notes.md) §2.4——`get_batch` 的切片逻辑对普通数组和 memmap 完全一样，实现不用改。这就是当初把它写成"纯切片"的回报。
- **两种格式**：`.npy` 自带 dtype/shape 头，`np.load(mmap_mode="r")` 最省心；若是 tokenizer 直接写出的原始 `.bin`，就用 `np.memmap` 并显式给 `dtype`（token id 常用 `uint16`，vocab < 65536 时足够）。

---

## 4. 交付点 3：checkpoint 到指定路径 + 恢复续训

复用 [checkpoint 笔记](./03_09_checkpointing_implementation_notes.md) 的 `save_checkpoint` / `load_checkpoint`。脚本在三处用到：

1. **恢复**：若给了 `--resume-from` 且文件存在，`start_iter = load_checkpoint(...)`，循环从这一步接着跑——model 权重、optimizer 的 $m,v,t$、迭代数全部还原（这正是"为什么要存优化器状态"的实战意义）。
2. **周期保存**：每 `--checkpoint-interval` 步存一次到 `--checkpoint-out`，防止机器中断丢进度。
3. **收尾保存**：训练结束再存一次最终状态。

**iteration 在这里的双重作用**：它既是 checkpoint 里存的续训位置，又是**学习率调度的输入**（`get_lr_cosine_schedule(it, ...)`）。因为我们的 LR 是无状态纯函数（见 [checkpoint 笔记](./03_09_checkpointing_implementation_notes.md) §4.1），恢复时只要拿回 `it` 就能把学习率接续到正确位置，不必单独存 scheduler。

---

## 5. 交付点 4：周期性记录 train/val 表现

两类日志，节奏不同：

- **train loss**：每 `--log-interval` 步打一次当前 batch 的训练损失和学习率（便宜，几乎无开销），用来看训练是否在降、有没有发散。
- **val loss**：每 `--eval-interval` 步，切到 `model.eval()`、在 `torch.no_grad()` 下取 `--eval-batches` 个验证 batch 求平均损失，再切回 `model.train()`。这更贵，所以间隔更大。它衡量**泛化**，是判断过拟合、选 checkpoint 的依据。

```python
@torch.no_grad()
def evaluate(model, data, args):
    model.eval()
    losses = [cross_entropy(model(x).view(-1, V), y.view(-1)).item()
              for x, y in (get_batch(data, ...) for _ in range(args.eval_batches))]
    model.train()
    return mean(losses)
```

- **为什么评估要 `no_grad` + `eval()`**：`no_grad` 不建计算图，省显存和时间；`eval()` 切换 dropout/BN 等模块到推理行为（本模型没有 dropout，但这是规范）。评估后**务必切回 `train()`**，否则后续训练行为异常。
- **落盘**：日志除了打到控制台，还可用 `--log-file` 追加成 jsonl，方便事后画 loss 曲线；handout 也提到可接 Weights & Biases 这类外部服务，这里保持轻量用 jsonl。

**形状对接**：模型输出 logits 是 `(B, seq, vocab)`，而 [交叉熵](./03_01_cross_entropy_implementation_notes.md) 期望 `(N, vocab)` + `(N,)`。所以 `logits.view(-1, vocab)`、`y.view(-1)` 把 batch 和 seq 两维摊平成 $N = B\times\text{seq}$ 个独立的下一 token 预测——这与交叉熵笔记里"在 batch 和 sequence 两维上求和取平均"完全一致。

---

## 6. 每一步的顺序为什么是这样

训练步内部顺序不是随意的，几个关键点：

1. **先算 lr 再 step**：学习率要在这一步更新前就写进 `optimizer.param_groups`，`step` 才会用新 lr。我们的 AdamW 从 `group["lr"]` 读学习率（见 [AdamW 笔记](./03_04_adamw_implementation_notes.md)），所以循环里直接改 `group["lr"] = lr` 即可。
2. **`zero_grad` 在 `backward` 前**：梯度是累加的，不清零会掺入上一步（见 [优化器 API 笔记](./03_02_pytorch_optimizer_api_notes.md) §7.1）。
3. **裁剪在 `backward` 之后、`step` 之前**：梯度裁剪要作用在"已算出、未使用"的梯度上（见 [梯度裁剪笔记](./03_07_gradient_clipping_implementation_notes.md)）。顺序错了就没意义。
4. **`optimizer.step()` 最后迈步**：用（可能被裁剪过的）梯度更新参数。

---

## 7. 冒烟测试：怎么确认脚本是对的

本题没有 pytest（它是"写脚本"的交付题），但可以用一个**小规模冒烟测试**自证跑通：造一小段随机 token 的 `.npy`，用极小模型跑几十步，检查：

- 数据能 memmap 加载、模型能建、loss 能算并反传；
- train/val 日志按间隔打印；
- checkpoint 能存；再用 `--resume-from` 重跑，确认打印 `resumed: start_iter=...` 且从该步接续。

（随机数据上 loss 不会明显下降，这正常——冒烟测试只验证"管道通不通"，不验证"学得好不好"。真正的收敛要在 TinyStories/OWT 上、有 BPE 之后才谈。）

示例命令（小模型、CPU、40 步）：

```sh
uv run cs336-train \
  --train-path <train.npy> --val-path <val.npy> \
  --vocab-size 256 --context-length 32 --d-model 64 --num-layers 2 \
  --num-heads 4 --d-ff 128 --batch-size 8 --total-iters 40 \
  --warmup-iters 5 --eval-interval 20 --checkpoint-out out/ckpt.pt --device cpu
```

---

## 8. 小结

1. **训练循环 = 内层 `zero_grad→forward→backward→(clip)→step`，外层套 LR 调度、日志、checkpoint**。
2. **交付点 1（超参可配）**：argparse 暴露所有模型/优化器/流程超参，便于批量扫参。
3. **交付点 2（memmap）**：`np.load(mmap_mode="r")` / `np.memmap` 让内存占用与数据规模脱钩，无缝对接 `get_batch`。
4. **交付点 3（checkpoint）**：周期+收尾保存；`--resume-from` 恢复 model/optimizer/iteration，LR 靠 iteration 续接。
5. **交付点 4（日志）**：train loss 高频打、val loss 低频评估（`eval()`+`no_grad`，评估后切回 `train()`）。
6. **形状对接**：logits `(B,seq,vocab)` 摊平成 `(B·seq, vocab)` 喂交叉熵，与"batch×seq 两维取平均"一致。
7. **顺序不可乱**：先设 lr、清零在反传前、裁剪在反传后 step 前。

至此 assignment1 的训练侧组件全部打通，配合后面的 BPE tokenizer（把语料编码成 token 数组）即可在 TinyStories/OWT 上真正开训。

---

## 参考

- 训练脚本：[cs336_basics/train.py](../cs336_basics/train.py)
- 各组件笔记：[数据加载](./03_08_data_loading_implementation_notes.md)、[交叉熵](./03_01_cross_entropy_implementation_notes.md)、[AdamW](./03_04_adamw_implementation_notes.md)、[LR 调度](./03_06_lr_schedule_implementation_notes.md)、[梯度裁剪](./03_07_gradient_clipping_implementation_notes.md)、[checkpoint](./03_09_checkpointing_implementation_notes.md)、[TransformerLM](./02_14_transformer_lm_implementation_notes.md)、[优化器 API](./03_02_pytorch_optimizer_api_notes.md)
- handout：`training_together`（§5.3 Training loop）
