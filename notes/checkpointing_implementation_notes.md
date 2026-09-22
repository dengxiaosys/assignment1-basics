# 实现检查点保存与加载（checkpointing）：原理与验证

## 0. 本文目标

记录 CS336 assignment1 里 **checkpoint** 的实现：训练到一半怎么把"整个训练现场"存到磁盘、之后怎么原样恢复继续训。核心就两个函数——`save_checkpoint` 和 `load_checkpoint`——但背后牵涉到一个重要问题：**要恢复训练，到底需要存哪些东西？** 本文讲清这三样（模型权重、优化器状态、迭代数）为什么缺一不可，以及 `torch.save` / `state_dict` 的机制。

对应实际代码：
- 实现：[cs336_basics/checkpoint.py](../cs336_basics/checkpoint.py) 的 `save_checkpoint` / `load_checkpoint`
- 接线：[tests/adapters.py](../tests/adapters.py) 的 `run_save_checkpoint` / `run_load_checkpoint`
- 测试：[tests/test_serialization.py](../tests/test_serialization.py) 的 `test_checkpointing`

前置：[优化器 API 笔记](./pytorch_optimizer_api_notes.md)（`self.state`、`state_dict()`）、[AdamW 笔记](./adamw_implementation_notes.md)（一阶/二阶矩是"状态"）。

---

## 1. 背景：为什么训练需要 checkpoint

语言模型训练动辄几小时到几周。中途可能遇到：机器抢占（spot 实例被回收）、崩溃、想换超参从某步续训、或单纯想保存一个"训练到 N 步"的快照做评估。**checkpoint 就是把训练现场冻结成磁盘上的一个文件，之后能原样解冻、无缝接着跑。**

关键词是**无缝**：恢复后的下一步，必须和"没中断、直接跑到这一步的下一步"**完全一样**。这就要求我们想清楚——一个训练步依赖哪些"隐藏状态"？只存模型权重够不够？

---

## 2. 核心问题：恢复训练需要存哪三样

一个训练步是 `zero_grad → forward → backward → step`（见 [优化器 API 笔记](./pytorch_optimizer_api_notes.md)）。要让"下一步"可复现，必须恢复这一步开始前的全部持久状态：

### 2.1 模型权重 `model.state_dict()`——显然要存

参数就是模型本身，不存就白训了。`state_dict()` 返回一个**有序字典**，键是参数名（如 `fc1.weight`），值是张量。

### 2.2 优化器状态 `optimizer.state_dict()`——最容易被漏掉的一样

**这是初学者最常忘的**。以为"存了权重就行"，但 AdamW 这类**有状态优化器**在 `self.state` 里为每个参数维护了跨步累积的量：一阶矩 $m$、二阶矩 $v$、步数 $t$（见 [AdamW 笔记](./adamw_implementation_notes.md)）。这些**不在 `model` 里，在 `optimizer` 里**。

为什么必须存？AdamW 的更新依赖偏差校正 $\hat m = m/(1-\beta_1^t)$、$\hat v = v/(1-\beta_2^t)$，以及自适应缩放 $\propto m/\sqrt{v}$。如果恢复时把 $m, v, t$ 全清零，相当于优化器"失忆"从头开始——动量丢了、自适应学习率的分母重置、偏差校正的 $t$ 归零，恢复后的更新方向和步长都和中断前对不上，训练会抖动甚至变差。

`optimizer.state_dict()` 里含两部分：`"state"`（每个参数的 $m,v,t$）和 `"param_groups"`（lr、betas、weight_decay 等超参）。

### 2.3 迭代数 `iteration`——一个不起眼但必要的标量

已经训了多少步。为什么要存？

- **学习率调度**：cosine schedule 的当前 lr 是 `t` 的函数（见 [LR schedule 笔记](./lr_schedule_implementation_notes.md)），续训必须知道"现在是第几步"才能接着算对 lr；
- **循环控制**：知道从第几步继续、还要跑到几步；
- **日志/评估节奏**：按步数触发。

它只是个普通 Python `int`，`torch.save` 能直接序列化。

> **一句话**：模型权重 = "学到了什么"，优化器状态 = "正以什么惯性在学"，迭代数 = "学到第几步了"。三者共同构成完整训练现场，缺一都无法真正无缝续训。

---

## 3. 实现：`torch.save` + `state_dict` 的对称配合

### 3.1 保存

```python
def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)
```

- **打成一个 dict 再存**：把三样装进一个字典，`torch.save` 一次写完。这样文件自描述、加载时按 key 取，比存三个文件清爽；
- **`torch.save`**：PyTorch 的序列化入口，底层用 `pickle` + zip 容器，能存张量（含其 dtype/device/形状）、嵌套 dict、Python 标量等。它接受**路径或文件对象**（`out` 的类型注解 `str | os.PathLike | BinaryIO | IO[bytes]` 就是这两类），所以既能 `torch.save(ckpt, "a.pt")` 也能传一个打开的 file/BytesIO；
- **存的是 `state_dict()` 而不是 `model` 本身**：只存"状态字典"（纯数据），不存整个对象。这样加载时不依赖类定义的 pickle 细节，跨代码改动更稳，是官方推荐做法。

### 3.2 加载

```python
def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
```

- **和保存严格对称**：`torch.load` 读回那个 dict，再按 key 把状态灌回去；
- **`load_state_dict` 是就地恢复**：注意它**修改传入的 `model`/`optimizer` 对象**（把张量值拷进已有参数），**不返回新对象**。所以调用方要先**构造好结构相同的空 model 和 optimizer**（测试里就是新建了 `new_model` / `new_optimizer`），再把状态灌进去。这也是为什么 `run_load_checkpoint` 的签名是"传入 model/optimizer、返回 int"；
- **返回 `iteration`**：唯一需要"返回"的是那个标量（它没有就地恢复的载体），调用方拿它继续训练循环。

### 3.3 为什么 model 和 optimizer 要"先建好再灌"

`state_dict` 是**纯数据**（张量值 + 超参），不含"网络结构""优化算法"这些**代码逻辑**。结构由你 `_TestNet(...)` / `AdamW(...)` 的构造决定。所以恢复分两步：**先用代码搭好同构的骨架，再用 `load_state_dict` 把数据填进骨架**。骨架不匹配（层数/形状对不上）会直接报错——这也是一种保护。

---

## 4. 真的只需要 model 和 optimizer 吗？——够用边界与业界范式

这是个好问题。分两个层面回答。

### 4.1 对本作业：存这三样是**够且正确**的

`test_checkpointing` 只验证 model 权重、optimizer 状态、iteration 三者能否恢复（见 §6）。而且这里的学习率调度是**无状态的纯函数** `get_lr_cosine_schedule(t, ...)`（见 [LR schedule 笔记](./lr_schedule_implementation_notes.md)）——它的"当前 lr"完全由传入的 `t` 决定，不在任何对象里藏状态。所以只要存了 `iteration`，续训时把它当 `t` 传进去就能算对 lr，**不需要单独存 scheduler**。对 CS336 的规模（单卡、fp32、数据可从头随机采），这三样确实覆盖了"下一步可复现"的全部持久状态。

### 4.2 对工业级"完全可复现续训"：通常还要多存几样

严格来说，"存 model + optimizer 就够"是**在若干隐含前提成立时**才对。一旦追求**位级别可复现**（恢复后每一步的数值和不中断时逐位相同）或用了更复杂的训练栈，下面这些也属于"训练现场"：

| 还需要存的东西 | 为什么 | 不存的后果 |
|---|---|---|
| **LR scheduler 状态**（若 scheduler 有内部状态） | PyTorch 的 `torch.optim.lr_scheduler.*` 是**有状态对象**（记 `last_epoch` 等），不像我们这个纯函数 | 恢复后 lr 从头算，学习率曲线错位 |
| **RNG 随机数状态**（CPU/CUDA/numpy/python） | dropout、数据 shuffle、数据增强、参数噪声都依赖 RNG；不恢复种子状态，恢复后的"随机"和原轨迹分叉 | 无法**位级复现**；dropout mask、采样顺序都变 |
| **AMP 混合精度的 `GradScaler` 状态** | 混合精度训练里 `GradScaler` 维护动态 loss scale，本身是跨步状态 | 恢复后 scale 重置，可能触发一轮 inf/nan 跳步 |
| **数据加载进度**（epoch、样本指针 / sampler 状态） | 要精确"接着上次的数据"往下喂，而非从头 | 恢复后数据顺序变，可能重复/跳过样本 |
| **训练元信息**（当前 loss/best metric、超参、代码/config 版本、`torch`/CUDA 版本） | 便于比较、早停、审计、排查"换环境后结果变了" | 难以复盘和对齐 |

**为什么本作业能省掉这些**：LR 是纯函数（靠 iteration 重算）、没用 dropout 之类强 RNG 依赖的复现要求、fp32 无 GradScaler、数据是每步独立随机采（`get_batch` 无"进度"概念，见 [数据加载笔记](./data_loading_implementation_notes.md)）。所以这些项要么不存在、要么可由 `iteration` 间接恢复。

### 4.3 业界范式确实如此吗？

是的，**"至少存 model + optimizer(+iteration/step)"是公认的最小集**，几乎所有训练框架都这么做；差别只在"为了多严格的可复现，额外再存多少"：

- **nanoGPT**（Karpathy）：checkpoint dict 存 `model` / `optimizer` / `model_args` / `iter_num` / `best_val_loss` / `config`——正是"三样 + 元信息"。
- **HuggingFace `Trainer`**：`save_model` 存权重，另存 `optimizer.pt`、`scheduler.pt`、`scaler.pt`、`rng_state.pth`（含 python/numpy/cpu/cuda 的 RNG）、`trainer_state.json`（step/epoch/日志）——把 §4.2 那张表基本存全了。
- **PyTorch Lightning**：`state_dict` 里含 model、optimizers、lr_schedulers、`global_step`/`epoch`、以及可选的 RNG、AMP scaler、回调状态。
- **大规模分布式**：用 `torch.distributed.checkpoint`(DCP) 做**分片**保存（每个 rank 存自己那片、异步、可重切分），因为单文件存不下也太慢。

一句话：**你的"三样"是所有人的公共底座；工业框架只是在其上按"想多可复现 / 用了多复杂的栈"再往上叠 RNG、scheduler、scaler、数据进度、元信息。**

---

## 5. PyTorch 里和 checkpoint 相关的接口全景

围绕"存/取训练状态"，PyTorch 提供了一组接口，适用场景各不相同。

### 5.1 序列化底座：`torch.save` / `torch.load`

- **`torch.save(obj, f)`**：把任意可 pickle 的对象（张量、含张量的嵌套 dict、Python 标量…）写到路径或文件对象。**适用**：几乎所有保存场景的最外层调用。
- **`torch.load(f, map_location=..., weights_only=...)`**：读回。两个关键参数：
  - **`map_location`**：把张量重定向到指定设备。**适用**：在 GPU 上存的 checkpoint 想在 CPU（或不同 GPU）上加载，用 `map_location="cpu"` 或 `map_location={"cuda:0":"cuda:1"}`，避免"CUDA device 不存在"报错。这是跨机/跨卡恢复的必备。
  - **`weights_only`**：新版本 PyTorch 出于**安全**默认按"只允许加载张量等安全类型"处理（`torch.load` 用 pickle，加载不可信文件可能执行任意代码）。**适用**：加载**别人给的**权重要保持 `weights_only=True`；若你的 checkpoint 里存了自定义类实例而报错，再显式 `weights_only=False`（仅限你信任的文件）。

### 5.2 状态导出/导入：`state_dict()` / `load_state_dict()`

这是**推荐的保存粒度**——只存"状态数据"而非整个对象（见 §3.1）。几乎每个"有状态"的训练组件都实现了这对方法：

| 对象 | `state_dict()` 里有什么 | 适用场景 |
|---|---|---|
| **`nn.Module`** | 参数 + 持久 buffer（如 BN 的 running_mean） | 存/取模型权重——最常用 |
| **`torch.optim.Optimizer`** | `state`（每参数的 $m,v,t$）+ `param_groups`（超参） | 存/取优化器状态——续训必备 |
| **`lr_scheduler.*`** | `last_epoch` 等调度内部状态 | 用了**有状态** scheduler 时（本作业的纯函数不需要） |
| **`torch.amp.GradScaler`** | 动态 loss scale 等 | 混合精度训练续训 |

- **`nn.Module.load_state_dict(sd, strict=True, assign=False)`**：
  - **`strict`**：默认要求键**完全对齐**；置 `False` 可**部分加载**。**适用**：迁移学习/改了网络结构，只想灌入能匹配上的那部分权重。
  - **`assign`**：是否直接把张量对象赋进去（而非拷贝值）。**适用**：配合 meta device / 特殊初始化的高级场景。

### 5.3 随机数状态：`get_rng_state` / `set_rng_state`（及 CUDA、numpy、python 版本）

- `torch.get_rng_state()` / `torch.set_rng_state(s)`：CPU 端 RNG。
- `torch.cuda.get_rng_state_all()` / `set_rng_state_all()`：所有 GPU 的 RNG。
- 还要配合 `numpy.random.get_state()` 和 python `random.getstate()`。
- **适用**：追求**位级可复现**、或训练里有 dropout / 随机数据增强 / 采样时。本作业没有这类严格复现要求，故未存。

### 5.4 分布式：`torch.distributed.checkpoint`（DCP）

- `dcp.save` / `dcp.load`：为多卡/多机训练做**分片 checkpoint**——每个 rank 存自己持有的那部分张量，支持异步、支持加载时**重新切分**（换卡数也能恢复）。
- **适用**：模型大到单机存不下、或用 FSDP/张量并行时。单卡作业用不到，属于 assignment2+ 的范畴。

### 5.5 选型小结

- **单卡、只要能续训** → `torch.save({model/optimizer/iteration})` + `torch.load`（**本作业**）。
- **跨设备加载** → 加 `map_location`。
- **加载他人权重** → 保持 `weights_only=True`。
- **迁移学习/改结构** → `load_state_dict(strict=False)`。
- **要位级复现 / 有 dropout / 随机增强** → 额外再存 RNG 状态。
- **有状态 scheduler / AMP** → 额外再存它们的 `state_dict()`。
- **大模型分布式** → `torch.distributed.checkpoint`。

---

## 6. adapter 与测试

`run_save_checkpoint` / `run_load_checkpoint` 直接转发到实现。[test_checkpointing](../tests/test_serialization.py) 的流程完整验证了"无缝恢复"：

1. 建一个三层 MLP `_TestNet` + AdamW，**跑 10 个真实训练步**（让优化器的 $m,v,t$ 积累起非平凡状态）；
2. `run_save_checkpoint(model, optimizer, iteration=10, out=path)` 存盘；
3. **新建** `new_model` / `new_optimizer`（全新随机初始化、优化器状态为空）；
4. `loaded = run_load_checkpoint(path, new_model, new_optimizer)`，断言 `loaded == 10`（迭代数恢复对）；
5. 断言两个 model 的 `state_dict` **键集合相同**、**每个张量数值 `assert_allclose`**（权重恢复对）；
6. 断言两个 optimizer 的 `state_dict` **键相同、`state` 里的 $m,v,t$ 数值相同、`param_groups` 超参相同**（优化器状态恢复对）。

第 6 步正是"为什么必须存优化器状态"的验证：如果 `save_checkpoint` 漏存 `optimizer.state_dict()`，`new_optimizer` 的 $m,v$ 全是空/零，这一步的 `allclose` 会失败。

运行：

```sh
uv run pytest -k test_checkpointing
```

预期 `1 passed`。

---

## 7. 小结

1. **要存哪三样**：模型权重（`model.state_dict()`）、优化器状态（`optimizer.state_dict()`，含 $m,v,t$）、迭代数（`iteration`）。缺优化器状态就无法无缝续训。
2. **只存这三样够吗**：对本作业够且正确（LR 是无状态纯函数、无 dropout/AMP、数据每步独立随机采）；工业级完全可复现通常还会再存 RNG、有状态 scheduler、GradScaler、数据进度、元信息（§4）。"至少 model+optimizer(+step)"是业界公认最小集。
3. **怎么存**：装进一个 dict，`torch.save` 一次写；存 `state_dict()`（纯数据）而非整个对象。
4. **怎么恢复**：`torch.load` 读回 → **先建好同构的 model/optimizer** → `load_state_dict` **就地**灌数据 → 返回 `iteration`。
5. **对称性**：save/load 一一对应；`load_state_dict` 改传入对象、不返回新对象，只有 `iteration` 靠返回值传出。
6. **相关接口**：底座 `torch.save`/`torch.load`（`map_location` 跨设备、`weights_only` 安全）；粒度 `state_dict`/`load_state_dict`（`strict` 部分加载）；进阶 RNG 状态、DCP 分布式（§5）。

---

## 参考

- 本仓库实现：[cs336_basics/checkpoint.py](../cs336_basics/checkpoint.py)
- 适配层：[tests/adapters.py](../tests/adapters.py)
- 测试：[tests/test_serialization.py](../tests/test_serialization.py)
- 前置：[pytorch_optimizer_api_notes.md](./pytorch_optimizer_api_notes.md)（`self.state` / `state_dict`）、[adamw_implementation_notes.md](./adamw_implementation_notes.md)（$m,v,t$ 是状态）、[lr_schedule_implementation_notes.md](./lr_schedule_implementation_notes.md)（lr 依赖迭代数）
- PyTorch 文档：`torch.save` / `torch.load` / `nn.Module.load_state_dict`（`https://pytorch.org/docs/stable/notes/serialization.html`）
