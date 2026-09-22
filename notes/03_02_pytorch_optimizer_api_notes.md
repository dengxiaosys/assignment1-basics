# 在 PyTorch 中实现一个优化器要满足什么条件

## 0. 本文目标

在动手写 AdamW 之前，先讲清楚一件事：**在 PyTorch 里自定义一个优化器，需要遵守哪些约定（契约）？** 本文系统说明 `torch.optim.Optimizer` 的接口要求、`param_groups` 与 `state` 两大机制、`step` 的写法规范，以及优化器如何嵌入训练循环。理解这套契约后，AdamW 只是"在 `step` 里换一套更新公式"。

对应背景：handout [00_01_cs336_assignment1_basics_extracted.md](./00_01_cs336_assignment1_basics_extracted.md) 的 `### 4.2 The SGD Optimizer`；接下来要实现的 [tests/adapters.py](../tests/adapters.py) 的 `get_adamw_cls`。

> 说明：本文讲的是**优化器 API 的通用规范**，用 handout 自带的 SGD 作合法示例；不含作业要交的 AdamW 实现。

---

## 1. 大图：优化器在训练里扮演什么角色

一个训练步固定四拍：

```python
opt.zero_grad()   # 1. 清空上一步的梯度
loss = ...        # 2. 前向：算损失
loss.backward()   # 3. 反向：autograd 把梯度写进每个 p.grad
opt.step()        # 4. 优化器：根据 p.grad 更新 p
```

**优化器只负责第 4 拍**：拿到 autograd 已经算好的梯度 `p.grad`，按某种规则（SGD、Adam、AdamW…）**原地修改参数** `p`。它不碰前向、不碰反向，只做"给定梯度，如何走一步"。

所以自定义优化器的本质就是：**继承基类、在 `step` 里写你的更新公式**。

---

## 2. 硬性契约：继承 `torch.optim.Optimizer` 并实现两个方法

PyTorch 要求自定义优化器**子类化 `torch.optim.Optimizer`**，并至少实现两个方法。

### 2.1 `__init__(self, params, ...)`

职责：接收待优化参数，把**默认超参数**打包成字典交给基类。

```python
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}          # 超参数放进 defaults 字典
        super().__init__(params, defaults)   # 必须调用基类构造器
```

要点：

- **`params`**：待优化的参数集合，通常是 `model.parameters()`。它可能是"一组参数"，也可能是"多个参数组"（见 §3）；
- **`defaults`**：一个字典，键是你为超参数起的名字（如 `"lr"`），值是默认值。**必须把它传给 `super().__init__(params, defaults)`**——基类据此把参数组织成 `param_groups`；
- 可以在这里做**合法性校验**（如 `lr < 0` 报错），这是良好实践；
- AdamW 的话，这里还会接收 `betas`、`eps`、`weight_decay` 等，一并放进 `defaults`。

### 2.2 `step(self, closure=None)`

职责：用当前梯度**做一次参数更新**。它在 `loss.backward()` 之后被调用，此时每个参数的 `p.grad` 已就绪。

```python
def step(self, closure=None):
    loss = None if closure is None else closure()
    for group in self.param_groups:        # 遍历每个参数组
        lr = group["lr"]                   # 取该组的超参数
        for p in group["params"]:          # 遍历组内每个参数
            if p.grad is None:             # 没有梯度就跳过
                continue
            state = self.state[p]          # 取该参数的持久状态
            t = state.get("t", 0)
            grad = p.grad.data
            p.data -= lr / math.sqrt(t + 1) * grad   # 原地更新
            state["t"] = t + 1             # 写回状态
    return loss
```

这段浓缩了几乎所有关键规范，下面逐条拆。

---

## 3. 机制一：`param_groups`（参数组）

基类会把你 `__init__` 传入的参数整理成 `self.param_groups`——**一个列表，每个元素是一个字典**，含 `"params"`（该组的参数列表）和这组的所有超参数（如 `"lr"`）。

- 若用户只传一个普通的参数集合（如 `model.parameters()`），基类会创建**单个组**，用 `defaults` 里的超参数；
- 若用户想给不同部分设不同超参（如"嵌入层用小 lr、其余用大 lr"），可以传入**多个组**，每组各自的超参覆盖默认值。

所以 `step` 里**必须先遍历 `param_groups`、再遍历组内 `params`**（两层循环），并从 `group[...]` 里读超参——不能把 lr 写死。这让同一个优化器天然支持"分组超参数"。

---

## 4. 机制二：`self.state`（每参数的持久状态）

有状态的优化器（Adam、AdamW、带动量的 SGD）需要**为每个参数记住一些跨步骤的量**（如动量、二阶矩估计、步数）。基类提供 `self.state`：

- 它是一个字典，**键是 `nn.Parameter` 对象，值是"该参数的状态字典"**；
- 第一次访问 `self.state[p]` 会得到空字典，你往里存需要的东西（示例里存 `"t"`）；
- 下一步 `step` 再访问同一个 `p`，就能读回上次存的值。

对 AdamW，这里会存一阶矩 `m`、二阶矩 `v` 和步数 `t`——这正是"矩估计"要持久化的地方。**状态与参数一一绑定**，所以多个参数各有独立状态，互不干扰。

> 这也解释了为什么 checkpoint 要单独保存 `optimizer.state_dict()`：模型权重在 `model` 里，但动量/矩估计在优化器的 `self.state` 里，二者都要存才能无缝续训。

---

## 5. `step` 的三条硬规范

### 5.1 原地修改参数

必须**原地**改 `p.data`（`p.data -= ...` 或 `p.data.add_(...)`），不能 `p = p - ...`（那只是重绑局部变量，不影响真正的参数张量）。用 `p.data` 或包在 `torch.no_grad()` 里，是为了**绕开 autograd**——参数更新本身不应被记录进计算图。

### 5.2 跳过无梯度的参数

`if p.grad is None: continue`。有些参数这一步可能没参与前向（没有梯度），必须跳过，否则会对 `None` 操作报错。

### 5.3 兼容 `closure` 参数

`step(self, closure=None)` 要接受一个可选的 `closure`。某些优化器（如 LBFGS）需要用它重算损失；我们的 SGD/AdamW 用不到，但**为符合 API 仍要保留这个参数并在开头处理**（`loss = closure() if closure else None`），最后把 `loss` 返回。

---

## 6. 基类免费给你的东西

继承 `torch.optim.Optimizer` 后，下面这些**不用自己写**：

- **`zero_grad()`**：把所有参数的 `.grad` 清零（训练循环每步开头调用）；
- **`state_dict()` / `load_state_dict()`**：导出/恢复优化器状态（`param_groups` 超参 + `self.state` 里的动量等），checkpoint 直接用；
- **`param_groups` 的组织**：你只管在 `__init__` 传 `defaults`，分组逻辑基类包办；
- **`add_param_group()`**：训练中动态加参数组。

这与 `nn.Module` 的哲学一致（见 [nn_module 笔记](./02_01_nn_module_and_linear_explained.md)）：**遵守约定登记 → 换来一整套现成服务**。

---

## 7. 一个可运行的训练循环（handout 示例）

```python
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1)
for t in range(100):
    opt.zero_grad()          # 清梯度
    loss = (weights**2).mean()
    loss.backward()          # 算梯度 -> weights.grad
    opt.step()               # 按规则更新 weights
```

这就是优化器的完整生命周期：构造（传参数+超参）→ 循环里 `zero_grad → backward → step`。

### 7.1 什么叫"run a step of the optimizer"

handout 说训练循环每轮"compute the loss and **run a step of the optimizer**"。这里的 **"a step" 就是调用一次 `opt.step()`**——**用当前这一批数据算出的梯度，把所有参数朝"减小损失"的方向挪动一次**。一次 step = 一次参数更新。

**"step（步）"这个词的思维范式**：把训练想象成在高维的"损失地形（loss landscape）"上**下山**。

- 你站在当前参数点 $\theta_t$，四周是损失曲面；
- 反向传播算出的梯度 $\nabla L$ 指向**上坡最陡**的方向，那么它的反方向就是**下坡最陡**；
- **一个 step 就是朝下坡方向迈一小步**：$\theta_{t+1} = \theta_t - \alpha\,\nabla L$（SGD 的形式），$\alpha$（学习率）控制步子多大；
- 迭代很多 step，参数一步步走向损失更低的谷底。

所以"run a step"不是"跑一段代码"的泛指，而是特指**优化器在参数空间里迈出的这一步**。训练 $N$ 个 iteration，就是走 $N$ 步。

**一步内部到底发生了什么**（把四拍串起来）：

```python
opt.zero_grad()   # 1. 清掉上一步的梯度（否则会累加，见下）
loss = f(batch)   # 2. 前向：在这批数据上算损失（你在损失地形上的"高度"）
loss.backward()   # 3. 反向：autograd 算出梯度，填进每个 p.grad（"最陡上坡方向"）
opt.step()        # 4. 优化器迈步：按规则用 p.grad 更新每个 p（朝下坡挪一步）
```

关键点：**梯度（第 3 拍）和更新（第 4 拍）是分开的**。autograd 只负责"算出该往哪个方向、多陡"，而"具体怎么迈这一步"——用不用动量、要不要自适应缩放、要不要权重衰减——完全由优化器的 `step` 决定。SGD、Adam、AdamW 的差别就在这"迈步规则"上，梯度来源是一样的。

**几个常被追问的点**：

- **为什么每步先 `zero_grad`**：PyTorch 的梯度是**累加**的（`backward` 会把新梯度加到 `.grad` 上，不是覆盖）。若不清零，这一步会掺入上一步的旧梯度，方向就错了。清零保证"每步只用当前 batch 的梯度"。
- **"step" vs "epoch"**：一个 **step = 一次参数更新 = 处理一个 batch**；一个 **epoch = 完整过一遍训练集 = 很多个 step**。语言模型训练常按 step 计（如"训练 100k steps"），因为数据量极大、未必按 epoch 组织。
- **step 与"学习率调度"的关系**：学习率 $\alpha$ 可以随 step 变化（warmup、cosine 衰减等，见后续 LR schedule）。所以很多实现会记录"当前是第几步"（示例 SGD 就把 `t` 存进 `self.state`），据此调整每一步的步长。
- **batch 让每步的方向是"估计值"**：损失和梯度是在**采样的一个 batch** 上算的，是全量梯度的噪声估计。所以每一步不保证严格下降，但大量步平均下来朝谷底走——这正是"随机（Stochastic）梯度下降"里"随机"的含义。

一句话：**"run a step of the optimizer" = 调一次 `opt.step()`，用当前梯度让参数在损失地形上朝下坡迈一步**；训练就是把这一步重复成千上万次。

### 7.2 对 `loss.backward()` 的前提条件与范式

第 3 拍 `loss.backward()` 不是随便对任何张量都能调的。它要求 loss 处在一个**可微的计算图**里。下面列清前提条件和使用范式。

**前提条件（不满足会报错或拿不到梯度）**：

1. **loss 必须是标量**（单个数），否则要显式传 `gradient` 参数。语言模型里我们对逐 token 损失取了 `.mean()`（见 [交叉熵笔记](./03_01_cross_entropy_implementation_notes.md)），结果是标量，所以能直接 `loss.backward()`。若对非标量张量 `t` 调用，需写 `t.backward(gradient=...)` 指定上游梯度——因为"对向量求导"要先知道对它每个分量的权重。
2. **计算图里必须有 `requires_grad=True` 的张量**。模型参数（`nn.Parameter`）默认 `requires_grad=True`；loss 必须是**从这些参数经过一串可微运算算出来的**。如果所有输入都不需要梯度，loss 就没有 `grad_fn`，`backward` 会报错 "element 0 ... does not require grad and does not have a grad_fn"。
3. **前向过程必须在 autograd 追踪下进行**。默认就是追踪的；但如果前向被包在 `torch.no_grad()` 里、或中途对张量调了 `.detach()`，那条路径就断开了计算图，loss 拿不到通向参数的梯度路径，backward 无效。
4. **用到的中间量没有被原地操作破坏**。autograd 反向时需要某些前向的中间结果；若你用原地操作（`x.add_()` 等）覆盖了它们，backward 可能报 "a variable needed for gradient computation has been modified by an inplace operation"。

**要遵循的范式**：

- **先 `zero_grad` 再 `backward`**：梯度是**累加**的（§7.1），不清零会掺入上一步的旧梯度。标准顺序永远是 `zero_grad → 前向 → backward → step`。
- **一个图默认只能 backward 一次**：`backward` 后计算图被释放。想对同一次前向再算一次梯度，要 `loss.backward(retain_graph=True)`（少用，通常重新前向即可）。
- **梯度只落在"叶子且 requires_grad"的张量上**：参数会收到 `.grad`；中间张量默认**不保留** `.grad`（需要的话对它调 `.retain_grad()`）。所以 `opt.step()` 里遍历的正是这些参数的 `.grad`。
- **backward 阶段不要包 `no_grad`**：`no_grad` 用在**参数更新**时（`step` 内改 `p.data`）以避免把更新记进图；而前向和 backward 需要 autograd，不能禁用。
- **backward 只算梯度、不改参数**：它只把梯度写进 `.grad`，"怎么用梯度走一步"是 optimizer 的事（§7.1 的梯度/更新分离）。

一句话：**能对 loss 调 `backward` 的前提是——它是一个标量、且由带 `requires_grad` 的参数经未被切断的可微计算图算出来的**；范式则是 `zero_grad → forward → backward → step` 且一图一次、梯度只落参数。

---

## 8. 小结：实现优化器的检查清单

1. **继承** `torch.optim.Optimizer`。
2. **`__init__(self, params, ...)`**：校验超参 → 打包成 `defaults` 字典 → `super().__init__(params, defaults)`。
3. **`step(self, closure=None)`**：处理 closure → 两层循环（`param_groups` → `params`）→ 跳过 `p.grad is None` → 从 `self.state[p]` 读写持久状态 → **原地**更新 `p.data` → 返回 loss。
4. **`param_groups`**：超参从 `group[...]` 取，别写死，天然支持分组。
5. **`self.state`**：每参数的动量/矩估计/步数存这里，与参数一一绑定，也是 checkpoint 要保存的。
6. **原地更新 + 跳空梯度 + 兼容 closure**：三条不能漏。

掌握这套契约后，AdamW 就是"在 `step` 里把更新公式换成带一阶/二阶矩估计和解耦权重衰减的版本"——机制不变，只换算法。

---

## 参考

- Handout：[00_01_cs336_assignment1_basics_extracted.md](./00_01_cs336_assignment1_basics_extracted.md)（4.2 The SGD Optimizer、Algorithm 1 AdamW）
- 适配层：[tests/adapters.py](../tests/adapters.py)（`get_adamw_cls`）
- 设计哲学对照：[02_01_nn_module_and_linear_explained.md](./02_01_nn_module_and_linear_explained.md)（"登记换服务"）
- PyTorch 文档：`torch.optim.Optimizer`（`https://pytorch.org/docs/stable/optim.html`）
