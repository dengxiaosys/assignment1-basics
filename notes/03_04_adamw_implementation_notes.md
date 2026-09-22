# 实现 AdamW 优化器：算法、原理与验证

## 0. 本文目标

记录 CS336 assignment1 里 **AdamW** 优化器的实现思路：它在 SGD 之上加了什么、为什么用一阶/二阶矩估计、什么是"偏差校正"、AdamW 相比 Adam 的关键改动（**解耦权重衰减**）是什么、adapter 怎么接到测试。

对应实际代码：
- 实现：[cs336_basics/optimizer.py](../cs336_basics/optimizer.py) 的 `AdamW`
- 接线：[tests/adapters.py](../tests/adapters.py) 的 `get_adamw_cls`
- 测试：[tests/test_optimizer.py](../tests/test_optimizer.py) 的 `test_adamw`

前置：[优化器 API 笔记](./03_02_pytorch_optimizer_api_notes.md)（`step`/`state`/`param_groups` 契约）、[学习率调参笔记](./03_03_learning_rate_tuning_notes.md)（步长直觉）。本文只在 §2 讲清算法后按 handout Algorithm 1 落地。

---

## 1. 背景：优化器的演进（SGD → Momentum → AdaGrad → RMSProp → Adam → AdamW）

AdamW 不是凭空出现的，它是一条"每一步都在补上一个前任缺陷"的演进链的终点。逐个讲清楚每个优化器**解决了什么、公式是什么、又留下什么问题**，AdamW 就水到渠成。统一记号：$\theta$ 是参数，$g_t=\nabla_\theta L$ 是第 $t$ 步梯度，$\alpha$ 是学习率。

### 1.1 SGD（随机梯度下降）

**更新**：

$$ \theta \leftarrow \theta - \alpha\, g_t $$

**思想**：朝当前梯度的反方向（下坡）迈一步（见 [优化器 API 笔记](./03_02_pytorch_optimizer_api_notes.md) 的"损失地形下山"）。

**痛点**：

1. **所有参数共用一个 $\alpha$**：但不同参数的梯度尺度差异可能很大，一个学习率很难同时合适；
2. **只看当前梯度**：梯度噪声大时方向抖动，在"狭长峡谷"地形里会来回震荡、收敛慢；
3. **对学习率极敏感**：太小慢、太大发散（见 [lr 笔记](./03_03_learning_rate_tuning_notes.md) 的实测）。

### 1.2 Momentum（动量 SGD）

**更新**（引入速度 $m$）：

$$ m \leftarrow \beta m + g_t, \qquad \theta \leftarrow \theta - \alpha\, m $$

**解决了什么**：把历史梯度做指数滑动平均，像"带惯性地下山"——一致的方向被累积加速、来回抖动的分量相互抵消。**缓解 SGD 的震荡、加快收敛**。

**$\beta$ 是超参数，怎么定**：$\beta$ 控制"记忆多久的历史梯度"，是**人为设定、非学习**的超参，取值在 $[0,1)$：

- **直觉**：指数滑动平均的"有效记忆窗口"约为 $\frac{1}{1-\beta}$ 步。$\beta=0.9$ → 大致平均最近 10 步；$\beta=0.99$ → 约 100 步。$\beta$ 越大越平滑、越"重惯性"，但对梯度变化的响应也越滞后。
- **常用值**：动量 SGD 里 $\beta$ 常取 **0.9**（最常见），有时 0.95 / 0.99。
- **权衡**：太小（接近 0）退化回普通 SGD、失去平滑；太大则惯性过强，可能冲过最优点或响应迟钝。0.9 是久经验证的默认。
- 到了 Adam（§1.5），一阶矩的 $\beta_1$ 就扮演这里的 $\beta$ 角色，默认也是 0.9；二阶矩另有 $\beta_2$（见 §1.5、§2）。

**仍留的问题**：学习率还是所有参数共用一个，没有"自适应"。

> **补充：什么是"指数滑动平均（EMA）"？** 上面的 $m$、以及后面 AdaGrad/RMSProp/Adam 用到的 $v$，本质都是对某个序列做 EMA，这里一次讲清。
>
> **递推公式**（观测 $g_t$，平均 $m_t$，初始 $m_0=0$）：
>
> $$ m_t = \beta\, m_{t-1} + (1-\beta)\, g_t $$
>
> $\beta\in[0,1)$ 是衰减系数，越大越"重历史、轻当前"。（动量里常写成 $m_t=\beta m_{t-1}+g_t$，是没乘 $1-\beta$ 的未归一化版；Adam 用的是上面带 $(1-\beta)$ 的标准 EMA，两者只差一个常数缩放。）
>
> **展开成显式加权和**（看清"指数衰减"）：
>
> $$ m_t = (1-\beta)\sum_{i=1}^{t} \beta^{\,t-i}\, g_i $$
>
> 第 $i$ 步观测对当前的权重是 $(1-\beta)\beta^{\,t-i}$——离现在越远、权重按 $\beta$ 的幂次指数衰减；所有权重和为 $1$，故是不改变尺度的加权平均。
>
> **有效记忆窗口** $\approx \dfrac{1}{1-\beta}$ 步：$\beta{=}0.9$≈10 步、$0.99$≈100 步、$0.999$≈1000 步。
>
> **偏差校正的来源**：因 $m_0=0$，早期 $\mathbb{E}[m_t]\approx(1-\beta^t)\,\mathbb{E}[g]$ 偏小，故除以 $1-\beta^t$ 校正：$\hat m_t = \dfrac{m_t}{1-\beta^{\,t}}$——这正是 Adam/AdamW 偏差校正（§2.1）的由来。Adam 的一阶矩 $m$、二阶矩 $v$ 就是分别对 $g$ 和 $g^2$ 各做一次 EMA。

### 1.3 AdaGrad

**更新**（累积梯度平方 $G_t=\sum_{i\le t} g_i^2$）：

$$ \theta \leftarrow \theta - \frac{\alpha}{\sqrt{G_t}+\epsilon}\, g_t $$

**解决了什么**：**每个参数自适应学习率**——历史梯度大的参数分母大、步子自动变小，反之变大。对稀疏特征尤其友好。

**仍留的问题**：$G_t$ 是**单调累加、只增不减**的，训练越久分母越大，**有效学习率不断衰减到几乎为 0**，后期几乎不再更新——"学习率过早枯竭"。

### 1.4 RMSProp

**更新**（把累加换成指数滑动平均 $v$）：

$$ v \leftarrow \beta v + (1-\beta) g_t^2, \qquad \theta \leftarrow \theta - \frac{\alpha}{\sqrt{v}+\epsilon}\, g_t $$

**解决了什么**：把 AdaGrad 的"无限累加"改成"**滑动平均**"——$v$ 只记住最近一段的梯度平方、会遗忘远古历史，于是分母不再无限增长，**修好了 AdaGrad 学习率枯竭**的问题。保留了"每参数自适应"。

**仍留的问题**：只有二阶矩的自适应缩放，**没有动量**（没利用 1.2 的方向平滑）。

### 1.5 Adam = Momentum + RMSProp

**思想**：把前两条线合并——**同时**用一阶矩 $m$（动量、方向平滑）和二阶矩 $v$（自适应缩放）。

$$ m \leftarrow \beta_1 m + (1-\beta_1) g_t \quad\text{（一阶矩：动量）} $$

$$ v \leftarrow \beta_2 v + (1-\beta_2) g_t^2 \quad\text{（二阶矩：自适应缩放）} $$

$$ \hat m = \frac{m}{1-\beta_1^t},\quad \hat v = \frac{v}{1-\beta_2^t} \quad\text{（偏差校正，见 §2.1）} $$

$$ \theta \leftarrow \theta - \alpha\,\frac{\hat m}{\sqrt{\hat v}+\epsilon} $$

**解决了什么**：兼得 Momentum 的方向平滑与 RMSProp 的每参数自适应，还加了**偏差校正**修正 $m,v$ 初始为 0 的早期偏差。**对学习率不那么敏感、收敛快而稳**，成为深度学习默认优化器。

**仍留的问题**：**权重衰减（L2 正则）与 Adam 结合得不干净**——见下。

### 1.6 AdamW = Adam + 解耦权重衰减

**问题的由来**：传统做法是把 L2 正则 $\lambda\theta$ 加进梯度 $g_t$。但在 Adam 里，这个 $\lambda\theta$ 也会被 $\frac{1}{\sqrt{\hat v}+\epsilon}$ 缩放——于是梯度大的参数，其权重衰减反而被削弱，正则强度和梯度大小**耦合**在一起，效果不可控。

**AdamW 的改动**：把权重衰减**从梯度里拿出来**，单独作用在参数上：

$$ \theta \leftarrow \theta - \alpha\lambda\theta \quad\text{（解耦权重衰减，不进梯度）} $$

$$ \theta \leftarrow \theta - \alpha_t\,\frac{m}{\sqrt{v}+\epsilon} \quad\text{（照常的 Adam 更新）} $$

**解决了什么**：权重衰减不再被 $\sqrt v$ 缩放，对所有参数**力度一致、可控**，正则化/泛化更好。这就是名字里 "W"（Weight decay）的由来，也是现代 LLM（LLaMA、GPT 等）普遍用 AdamW 而非 Adam 的原因。

### 1.7 一张演进表

| 优化器 | 关键机制 | 修好了前任的什么 | 仍留的问题 |
|---|---|---|---|
| SGD | $\theta-\alpha g$ | —— | 共用 lr、震荡、对 lr 敏感 |
| Momentum | 一阶矩 $m$（动量） | 震荡、收敛慢 | 仍共用一个 lr |
| AdaGrad | 累积 $g^2$ 自适应 | 每参数自适应 lr | lr 单调枯竭 |
| RMSProp | $g^2$ 滑动平均 | lr 枯竭 | 无动量 |
| Adam | $m$ + $v$ + 偏差校正 | 合并动量与自适应 | 权重衰减耦合 |
| **AdamW** | Adam + **解耦权重衰减** | 权重衰减耦合 | （当前主流） |

一句话主线：**Momentum 加"方向记忆"、AdaGrad/RMSProp 加"每参数自适应步长"、Adam 合二为一、AdamW 再把权重衰减做干净。** 下面按 handout Algorithm 1 落地 AdamW。

---

## 2. AdamW 算法（handout Algorithm 1）

每个参数维护一阶矩 $m$、二阶矩 $v$（初始为 0）和步数 $t$。每步（$g$ 是当前梯度）：

$$ m \leftarrow \beta_1 m + (1-\beta_1) g $$

$$ v \leftarrow \beta_2 v + (1-\beta_2) g^2 $$

$$ \alpha_t \leftarrow \alpha\,\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t} $$

$$ \theta \leftarrow \theta - \alpha\lambda\theta \quad\text{（解耦权重衰减）} $$

$$ \theta \leftarrow \theta - \alpha_t\,\frac{m}{\sqrt{v}+\epsilon} \quad\text{（矩调整的更新）} $$

超参：学习率 $\alpha$、矩衰减 $(\beta_1,\beta_2)$（常 $(0.9,0.999)$，大模型常用 $(0.9,0.95)$）、数值稳定 $\epsilon\approx10^{-8}$、权重衰减 $\lambda$。

### 2.1 为什么要"偏差校正" $\alpha_t$

$m,v$ 初始化为 0，前几步会**偏向 0**（还没积累够历史）。第 7 步的 $\alpha_t=\alpha\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}$ 就是**偏差校正**：早期 $t$ 小，$1-\beta_1^t$、$1-\beta_2^t$ 都远小于 1，$\alpha_t$ 被放大以补偿"矩偏小"；随 $t$ 增大，$\beta^t\to0$，$\alpha_t\to\alpha$，校正逐渐消失。这让训练**初期就有合理步长**，不至于因矩估计偏小而走得过慢。

### 2.2 AdamW 的核心：解耦权重衰减

- **传统 Adam + L2**：把 $\lambda\theta$ 加进梯度 $g$，于是它也被 $\frac{1}{\sqrt{v}+\epsilon}$ 缩放——梯度大的参数，其权重衰减反而被削弱，正则强度和参数耦合，不干净。
- **AdamW**：权重衰减单独一步 $\theta \leftarrow \theta - \alpha\lambda\theta$（等价 $\theta\leftarrow(1-\alpha\lambda)\theta$），**不进梯度、不被 $\sqrt{v}$ 缩放**。这样"把参数往 0 拉"的力度对所有参数一致、可控，正则化效果更好。这正是 AdamW 优于 Adam 的关键。

---

## 3. 实现映射（代码怎么对应算法）

关键片段（完整见 [optimizer.py](../cs336_basics/optimizer.py)）：

```python
m.mul_(beta1).add_(grad, alpha=1 - beta1)          # m ← β1 m + (1-β1) g
v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) # v ← β2 v + (1-β2) g²
lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t) # 偏差校正学习率 α_t
p.data.mul_(1 - lr * weight_decay)                  # 解耦权重衰减 θ←(1-αλ)θ
p.data.addcdiv_(m, v.sqrt().add_(eps), value=-lr_t) # θ ← θ - α_t m/(√v+ε)
```

- **遵循优化器契约**（见 [优化器 API 笔记](./03_02_pytorch_optimizer_api_notes.md)）：继承 `torch.optim.Optimizer`，`__init__` 把 `lr/betas/eps/weight_decay` 放进 `defaults`；`step` 两层遍历 `param_groups → params`、跳过 `p.grad is None`、状态存 `self.state[p]`、**原地**更新 `p.data`。
- **状态初始化**：`if len(state)==0` 时建 `m=v=zeros_like(p)`、`t=0`；这就是"AdamW 是 stateful、每参数存两个矩"的体现，也是 checkpoint 要保存 optimizer 状态的原因。
- **原地算子**：`mul_/add_/addcmul_/addcdiv_` 都原地更新 `m`、`v`、`p.data`，省显存；`v.sqrt()` 返回新张量，`.add_(eps)` 只改这个临时结果、不污染 `v`。

---

## 4. adapter 与测试

`get_adamw_cls()` 直接返回 `AdamW` **类**（不是实例）——测试会自己用 `lr=1e-3, weight_decay=0.01, betas=(0.9,0.999), eps=1e-8` 实例化、跑 1000 步、比对最终权重。

`test_adamw` 的判定较宽松：**先与 PyTorch 的 `torch.optim.AdamW` 对比**（`atol=1e-4`），匹配即通过；否则再比对参考快照。因为"权重衰减在哪一步施加"有几种数学等价但浮点略异的写法，测试同时接受两种参考。我们的实现与 PyTorch AdamW 在容差内一致，直接通过。

运行：

```sh
uv run pytest -k test_adamw
```

预期 `1 passed`。

---

## 5. 小结

1. **Adam = 动量（一阶矩 $m$）+ 自适应缩放（二阶矩 $v$，$1/\sqrt v$）**：平滑噪声 + 每参数自适应有效学习率。
2. **偏差校正 $\alpha_t$**：补偿 $m,v$ 初始为 0 的早期偏差，随 $t$ 增大趋于 $\alpha$。
3. **AdamW 的 "W"**：权重衰减**解耦**——单独一步 $\theta\leftarrow(1-\alpha\lambda)\theta$，不进梯度、不被 $\sqrt v$ 缩放，正则更干净。
4. **stateful**：每参数存 $m,v,t$，用 `self.state[p]`；这也是 checkpoint 要存 optimizer 状态的原因。
5. **契约照旧**：继承 Optimizer、`defaults` 传超参、`step` 里原地更新、跳空梯度。

有了损失（交叉熵）和优化器（AdamW），训练三要素还差"学习率调度"和"梯度裁剪"——接下来实现。

---

## 参考

- 本仓库实现：[cs336_basics/optimizer.py](../cs336_basics/optimizer.py)
- 适配层：[tests/adapters.py](../tests/adapters.py)
- 测试：[tests/test_optimizer.py](../tests/test_optimizer.py)
- 前置：[03_02_pytorch_optimizer_api_notes.md](./03_02_pytorch_optimizer_api_notes.md)、[03_03_learning_rate_tuning_notes.md](./03_03_learning_rate_tuning_notes.md)
- Handout：[00_01_cs336_assignment1_basics_extracted.md](./00_01_cs336_assignment1_basics_extracted.md)（4.3 AdamW、Algorithm 1）
- 原始文献：Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW), 2019；Kingma & Ba, *Adam*, 2015。
