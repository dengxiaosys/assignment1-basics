# 实现无 bias 的 Linear：思路、接线与验证

## 0. 本文目标

记录 CS336 assignment1 里 `Linear` 层的实现思路——不是罗列代码，而是讲清楚**每个决定为什么这么做**：模块怎么搭、权重形状怎么定、`forward` 怎么算、adapter 怎么把它接到测试、以及"装权重"那一步的几种写法与取舍。

对应的实际代码在本仓库：
- 实现：[cs336_basics/nn.py](../cs336_basics/nn.py)
- 接线：[tests/adapters.py](../tests/adapters.py) 的 `run_linear`
- 测试：[tests/test_model.py](../tests/test_model.py) 的 `test_linear`

配套背景见同目录 [nn_module_and_linear_explained.md](./nn_module_and_linear_explained.md)（讲 `nn.Module`/`nn.Linear` 原理）与 [pytest_fixtures_explained.md](./pytest_fixtures_explained.md)（讲测试怎么跑）。

> 记号约定（仓库规则 R1/R2）：数学上以列向量记 $y = Wx$；PyTorch 中特征在最后一维，实现为 $y = x W^\top$。下文两种写法都会出现，涉及张量形状时按框架布局。

---

## 1. 需求拆解

deliverable 一句话：**实现一个继承 `nn.Module`、遵循 `nn.Linear` 接口、但没有 bias 的 `Linear`。** 拆成可执行的约束：

1. 继承 `nn.Module`，`__init__` 先 `super().__init__()`；
2. 只有一个可学习参数 `weight`，**无 bias**；
3. 接口风格 `Linear(d_in, d_out, device=None, dtype=None)`；
4. `weight` 形状 `(d_out, d_in)`；
5. `forward` 支持任意前置批量维：`(..., d_in) → (..., d_out)`。

---

## 2. 逐个决定的理由

### 2.1 为什么权重要用 `nn.Parameter`

只有用 `nn.Parameter` 包起来的张量，才会被 `nn.Module` 登记为可学习参数，从而进入 `parameters()` / `state_dict()`、被优化器更新、被 `load_state_dict` 加载。直接 `self.weight = torch.empty(...)`（普通张量）不会被登记——这是最常见的坑。

### 2.2 为什么形状是 `(d_out, d_in)`

三条理由（详见配套背景笔记）：

1. 每个输出通道对应权重矩阵的**一行**，语义直观；
2. 与官方 `state_dict` 的形状/命名一致，便于加载/比对参考权重（本测试正是塞进官方风格权重）；
3. `forward` 对最后一维做 $x W^\top$，天然支持任意批量维。

数学上（列向量）就是 $y = Wx$，$W \in \mathbb{R}^{d_{out}\times d_{in}}$。

### 2.3 为什么不声明 bias

需求明确不要 bias；现代 LLM 线性层也大多去 bias（后接的 norm 能吸收偏移、省参数且通常无损）。所以既不声明 bias 参数，`forward` 里也不加。

### 2.4 `forward` 怎么写

对最后一维做矩阵乘、保留批量维即可：`x @ weight.transpose(-2, -1)`。用 `einsum("... i, o i -> ... o", x, weight)` 等价。数学上就是 $y = Wx$（列向量视角），PyTorch 里是 $y = xW^\top$。

**为什么要转置。** `weight` 存成 `(d_out, d_in)`，而 `x` 的最后一维是 `d_in`。矩阵乘 `x @ M` 的规则是"`x` 最后一维 == `M` 倒数第二维"，所以要把 `weight` 从 `(d_out, d_in)` 转成 `(d_in, d_out)`，才能：

$$ \underbrace{x}_{(\dots,\, d_{in})} \;@\; \underbrace{W^\top}_{(d_{in},\, d_{out})} \;=\; \underbrace{y}_{(\dots,\, d_{out})} $$

这正是 $y = x W^\top$（等价于列向量的 $y = Wx$）。

**为什么用 `transpose(-2, -1)` 而不是 `.T`。**

- `transpose(-2, -1)` 只交换**最后两维**，用负索引，无论张量前面还有多少维都语义明确、可移植；
- `.T` 只在二维上等价，对 `>2` 维会**反转所有维度**，语义不同，且新版 PyTorch 对高维用 `.T` 会告警。这里 `weight` 恒为二维，`.T` 也能对，但 `transpose(-2, -1)` 是更稳健的习惯写法；
- 等价替代：`self.weight.mT`（"最后两维转置"的简写）、`einsum(...)`，或直接 `torch.nn.functional.linear(x, self.weight)`（内部就是 $xW^\top$，连转置都省了）。

### 2.5 初始化（只影响独立使用，不影响本测试）

`reset_parameters` 用截断正态、方差 $2/(d_{in}+d_{out})$、截断在 $\pm 3\sigma$，作为独立使用时的合理默认。这行是：

```python
std = math.sqrt(2.0 / (self.d_in + self.d_out))
nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
```

**为什么方差取 $2/(d_{in}+d_{out})$（Xavier/Glorot）。** 目标是让信号"不放大也不缩小"地穿过很多层，同时兼顾前向与反向：

- 只保前向激活方差稳定，理想是 $\mathrm{Var}(W)=1/d_{in}$（一个输出是 $d_{in}$ 项之和，要除掉扇入）；
- 只保反向梯度方差稳定，理想是 $1/d_{out}$（扇出）；
- 两者通常不等，Glorot 取折中 $\mathrm{Var}(W)=\dfrac{2}{d_{in}+d_{out}}$，于是 $\text{std}=\sqrt{\dfrac{2}{d_{in}+d_{out}}}$。

初值尺度不当会让深层前向激活或反向梯度指数级爆炸/消失，这就是要精心选方差的原因。

**为什么用截断正态而不是普通正态。** 普通正态偶尔采到 $4\sigma$–$5\sigma$ 的离群值 → 个别过大的初始权重 → 训练早期不稳。`trunc_normal_` 把采样截断在 $[-3\sigma, 3\sigma]$（参数 `a=-3*std, b=3*std`）砍掉尾部尖峰，而正态 $\pm3\sigma$ 内已覆盖约 99.7% 概率质量，几乎不损表达力却更稳。`mean=0.0` 是要零均值（无偏）权重，配合无 bias 层，初始输出期望为 0。

**要点**：这只在"独立使用/真正训练"时生效；`test_linear` 会用外部参考权重覆盖它，所以初始化方案**不影响那个测试的数值结果**。若要严格对齐 handout 指定初始化，按 PDF 公式改 `std`（有些实现用 $1/\sqrt{d_{in}}$ 之类）即可。

**为什么用正态分布而不是其它分布。** 先强调一点：真正决定成败的是**方差（尺度）选对**，而非分布形状本身。选正态有几个自然理由：

1. **中心极限定理**：每个输出是大量 $w_i x_i$ 的求和，独立随机量之和趋于正态。既然前向传播天然在"造正态"，用正态初始化让每层分布自洽，最利于分析与稳定。
2. **两参数即定 + 最大熵**：正态由均值、方差唯一确定，而初始化的诉求恰好是"零均值 + 控方差"。在"给定均值方差"约束下，正态是**最大熵**分布——最不引入额外偏见。
3. **各向同性**：正态圆对称，不偏袒任何方向，符合"初始权重无方向偏好"。

**其它分布也可用（只要方差匹配）：**

- **均匀分布**：非常常见。PyTorch `nn.Linear` 默认就是 Kaiming **均匀**（`kaiming_uniform_`），也有 `xavier_uniform_`。把方差调到同样的 $2/(d_{in}+d_{out})$，均匀与正态效果接近，且均匀有界、无长尾。
- **截断正态**：即本实现所用，是正态的稳健版。
- **正交初始化**（`nn.init.orthogonal_`）：让权重矩阵正交，对很深或 RNN 类模型有利于保范数。
- **全 0 / 常数：不行**。权重全相同会导致神经元对称、学到相同内容（对称性无法打破）——这正是初始化必须随机的根本原因。

一句话：**正态是"假设最少、最自然"的默认，但成败取决于方差；均匀等有界分布只要方差匹配也同样好用。**

**正态分布的别名（同一分布的不同叫法）。**

| 别名 | 说明 |
|---|---|
| 正态分布（Normal） | 最标准的名字 |
| 高斯分布（Gaussian） | ML/工程最常用；`torch.randn` 的 "n" 即 normal，代码里常说 "Gaussian init" |
| 钟形曲线（Bell curve） | 因形状得名的通俗叫法 |
| 标准正态 / Z 分布 | 特指 $\mu=0,\sigma=1$ 的那一个（对应 Z-score） |
| 误差分布（law of errors） | 早期因描述测量误差而得名 |

易混提醒：**"标准正态" ≠ "正态"**——前者特指 $\mu=0,\sigma=1$，后者是任意 $\mu,\sigma$ 的一族。本实现用的是 $\mu=0$、$\sigma=\sqrt{2/(d_{in}+d_{out})}$ 的正态再做截断，并非标准正态。中文里"正态分布"与"高斯分布"完全等价、可互换，论文/代码中 Gaussian 更常见。

### 2.6 这个类具备 backward 能力吗

**具备——虽然我们没写任何 `backward`。** 原因是 PyTorch 的 **autograd 自动求导**，而非因为继承了 `nn.Module`。要分清两件事：

- `nn.Module` 只负责**参数登记与组织**（`parameters()`、`state_dict()` 等），它**不负责求导**；
- 能不能反向传播，取决于 `forward` 里用的算子**是否可微、是否被 autograd 追踪**。

本实现满足自动求导的两个条件：

1. **`self.weight` 是 `nn.Parameter`**，默认 `requires_grad=True`，autograd 会追踪一切用到它的运算；
2. **`forward` 只用了 `@`（矩阵乘）和 `transpose`** 这些 PyTorch 内建**可微算子**。前向时 autograd 自动把每步记入**计算图**（每个输出张量的 `grad_fn` 记录了它是怎么算出来的、反向如何求导）。

于是对最终 loss 调用 `.backward()` 时，autograd 沿计算图反向自动求导，把梯度累加到 `self.weight.grad`。**你只需写对 forward，backward 由 autograd 依计算图自动生成**——这正是"自动微分"的意义。

**什么时候才需要手写 `backward`**：只有绕过 autograd 时——用了非 PyTorch 算子（手写 CUDA/Triton kernel、numpy 运算），或继承 `torch.autograd.Function` 自定义算子。本 `Linear` 全程用标准可微算子，因此**不需要也不应**手写 backward。

**一个能自证的小检查**：

```python
lin = Linear(4, 3)
x = torch.randn(2, 4)
lin(x).sum().backward()
print(lin.weight.grad is not None)   # True → 梯度确实回传到了 weight
```

`weight.grad` 非空即证明反向链路是通的。

---

## 3. adapter 怎么把模块接到测试

`run_linear(d_in, d_out, weights, in_features)` 是"作业适配层"：测试给一份**参考权重** `weights` 和输入 `in_features`，要求用这份权重算出结果。思路三步：

1. 构造 `Linear(d_in, d_out, device=weights.device, dtype=weights.dtype)`——device/dtype 跟随传入权重，避免设备/精度不匹配；
2. **用传入的 `weights` 覆盖模块自带的随机权重**（见下节）；
3. 返回 `linear(in_features)`（注意用 `linear(x)` 触发 `__call__`，而非 `linear.forward(x)`）。

---

## 4. "装权重"这一步的几种写法与取舍

核心矛盾：模块 `__init__` 会随机初始化 `weight`，但测试要用它给的参考权重，所以必须覆盖。

| 写法 | 代码要点 | 优点 | 缺点 |
|---|---|---|---|
| **A. `load_state_dict`（当前采用）** | `linear.load_state_dict({"weight": weights})` | 走官方路径，**校验键名与形状**；让测试真正覆盖模块的 `__init__`/`forward` | 键名必须与参数名严格一致（默认 `strict=True`） |
| B. 直接赋值 | `linear.weight.data = weights` 或 `with torch.no_grad(): linear.weight.copy_(weights)` | 直观简单 | 绕过键名/形状校验，写错不易发现 |
| C. 不建模块 | `return in_features @ weights.transpose(-2, -1)` | 最简，数值等价 | 没真正测到你的 `Linear` 类 |

选 A 的理由：**顺带体检**（键名/形状对不对），且确保测试覆盖到 `Linear` 本身而非 adapter 里另写的等价计算。代价只是多一行。

一个关键点：`{"weight": weights}` 的键 `"weight"` 必须与 `nn.py` 里的属性名 `self.weight` 一致；若命名成 `self.w`，这里就得写 `{"w": weights}`，否则 `strict=True` 会报键不匹配。

---

## 5. 为什么测试能过

`test_linear` 取官方 `ts_state_dict` 里的 `layers.0.ffn.w1.weight` 作为参考权重，经 `run_linear` 用**我们的 `Linear`** 前向，再用 `numpy_snapshot` 与 `tests/_snapshots/test_linear.npz` 逐元素比对（容差 `rtol=1e-4, atol=1e-2`）。链路：

```text
test_linear → run_linear(装入参考 weights) → Linear.forward = x @ Wᵀ
           → numpy_snapshot 对照 _snapshots/test_linear.npz → PASS
```

因为权重来自外部、`forward` 是标准 $xW^\top$，数值自然与快照一致。

运行：

```sh
uv run pytest -k test_linear
```

预期 `1 passed`。

---

## 6. 小结

1. **模块三要素**：`super().__init__()` → `self.weight = nn.Parameter(torch.empty(d_out, d_in, ...))` → `forward` 做 $xW^\top$。
2. **无 bias**：不声明、不相加。
3. **形状 (d_out, d_in)**：对应"每个输出一行"、兼容官方权重、支持批量维。
4. **adapter**：建模块 → 装参考权重（`load_state_dict` 最稳）→ `linear(x)`。
5. **初始化不影响本测试**：外部权重会覆盖它；要对齐 handout 就改 `std`。

---

## 参考

- 本仓库实现：[cs336_basics/nn.py](../cs336_basics/nn.py)
- 适配层：[tests/adapters.py](../tests/adapters.py)
- 测试：[tests/test_model.py](../tests/test_model.py)
- 背景：[nn_module_and_linear_explained.md](./nn_module_and_linear_explained.md)、[pytest_fixtures_explained.md](./pytest_fixtures_explained.md)
