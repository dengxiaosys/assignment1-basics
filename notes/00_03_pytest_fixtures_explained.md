# pytest fixture 与依赖注入：`uv run pytest -k test_linear` 背后的机制

## 0. 本文目标

这份讲义解释一个初次接触会觉得"像魔法"的现象：

```python
def test_linear(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff):
    ...
```

这个测试函数**声明了 5 个参数，却从不自己传值**，运行时它们却都被自动填好了。本文回答：

1. pytest 凭什么"自动加载"这些参数？
2. 为什么不用 `import` 就能找到它们？
3. `d_model` 这种"参数又带参数"的写法是怎么解析的？
4. 这套机制在本仓库（CS336 assignment1）的 `tests/conftest.py` 里是如何具体体现的。

本讲义与同目录其它 CS336 笔记同属一套学习笔记。所有代码引用都对应本仓库 [tests/conftest.py](../tests/conftest.py) 与 [tests/test_model.py](../tests/test_model.py) 的真实内容。

> 说明：本文是工具/框架机制讲解，不含任何作业实现代码。

---

## 1. 一句话原理

**pytest 把测试函数的每个参数名，当作一个"要注入的 fixture 名"；它去查找同名的 `@pytest.fixture` 函数，调用它，把返回值作为该参数传进来。**

所以"自动加载"不是魔法，而是一条极简单的约定：

$$ \text{测试函数的参数名} \;=\; \text{fixture 的名字} $$

名字对上了，就注入；名字对不上，就报错 `fixture '...' not found`。

这套"框架负责创建并传入依赖、使用者只需声明需要什么"的模式，就是**依赖注入（Dependency Injection）**。测试函数只"声明需求"（参数名），不"负责构造"（怎么来的交给 fixture）。

---

## 2. 什么是 fixture

fixture 是用 `@pytest.fixture` 装饰的函数，用来**为测试准备"前置数据或前置环境"**，并把它交给测试使用。

本仓库的一个最小例子（[conftest.py](../tests/conftest.py)）：

```python
@pytest.fixture
def d_ff():
    return 128
```

它声明了一个名为 `d_ff` 的 fixture，返回 `128`。于是任何测试只要在参数里写 `d_ff`，就会拿到 `128`：

```python
def test_linear(..., d_ff):   # d_ff 这个参数 == 128
    ...
```

fixture 的返回值可以是任何东西：一个整数、一个张量、一个打开的文件、一个数据库连接、一个比对器对象等。

---

## 3. 为什么不用 import 就能找到：`conftest.py` 的特殊地位

普通模块里的函数，别的文件要用必须 `import`。但 `test_model.py` 并没有 `from conftest import d_ff`，为什么能直接用？

因为文件名叫 **`conftest.py`**——这是 pytest 约定的**特殊文件**：

- pytest 在收集测试前会**自动发现并加载**每个目录下的 `conftest.py`；
- 其中定义的 fixture **自动对该目录及其所有子目录下的测试可见**，无需任何 import；
- 因此 `conftest.py` 是"本目录测试共享的 fixture 仓库"。

这就是"自动"的另一半来源：**名字匹配**解决"注入谁"，**conftest.py 自动加载**解决"从哪找到"。

> 对比：如果 fixture 定义在某个普通测试文件里，它默认只对**那个文件**可见；要跨文件共享，就放进 `conftest.py`。

---

## 4. fixture 可以依赖 fixture：递归解析

fixture 自己也可以带参数，那些参数又是别的 fixture。这让依赖能层层组合。

看本仓库最典型的一条（[conftest.py:242](../tests/conftest.py)）：

```python
@pytest.fixture
def d_model(n_heads, d_head):
    return n_heads * d_head
```

`d_model` 依赖 `n_heads` 和 `d_head`，而这两个也都是 fixture：

```python
@pytest.fixture
def n_heads():
    return 4

@pytest.fixture
def d_head():
    return 16
```

于是当某个测试要 `d_model` 时，pytest 会**递归**地先解析 `n_heads`（得 4）、`d_head`（得 16），再算 `d_model = 4 * 16 = 64`。

再往上一层，`in_embeddings` 又依赖 `d_model`（[conftest.py:270](../tests/conftest.py)）：

```python
@pytest.fixture
def in_embeddings(batch_size, n_queries, d_model):
    torch.manual_seed(4)
    return torch.randn(batch_size, n_queries, d_model)
```

这样就形成一棵**依赖树**，pytest 按拓扑顺序自底向上求值。

---

## 5. 完整走一遍 `test_linear`

本仓库 [test_model.py](../tests/test_model.py) 的测试签名是：

```python
def test_linear(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff):
    ...
```

pytest 在运行这条用例前的解析过程（对照真实 conftest 的返回值）：

```text
test_linear 需要: [numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff]

解析依赖树（自底向上）：
  n_heads()      -> 4
  d_head()       -> 16
  d_model(n_heads, d_head) -> 4 * 16 = 64
  batch_size()   -> 4
  n_queries()    -> 12
  in_embeddings(batch_size, n_queries, d_model)
                 -> torch.randn(4, 12, 64)   # 固定 seed=4，可复现
  d_ff()         -> 128
  ts_state_dict(request)  -> 从 fixtures/ts_tests 读参考权重和 config
  numpy_snapshot(request) -> 构造快照比对器（默认名 = 当前测试名）

全部就绪 -> 调用 test_linear(numpy_snapshot=..., ts_state_dict=...,
                               in_embeddings=<tensor>, d_model=64, d_ff=128)
```

测试体里再用这些值去调用作业适配层 `run_linear(...)`，最后用 `numpy_snapshot.assert_match(output)` 与 `tests/_snapshots/` 里的参考数组比对。

> 注意固定随机种子：`in_embeddings` 里的 `torch.manual_seed(4)` 保证每次生成的随机输入**完全一致**，这样快照比对才有意义（否则输入一变，参考输出就对不上）。本仓库为 `q`/`k`/`v`/`in_embeddings`/`mask`/`in_indices` 各用了不同的固定种子。

---

## 6. 特殊内置 fixture：`request`

`numpy_snapshot(request)` 和 `ts_state_dict(request)` 里的 `request` 不是 conftest 定义的，而是 pytest **内置**的特殊 fixture。它携带"当前测试的上下文"，常用字段：

- `request.node.name`：当前测试的名字（[conftest.py:185](../tests/conftest.py) 用它作为快照的默认名，从而定位 `_snapshots/` 里对应的参考文件）；
- `request.config.getoption(...)`：读取命令行选项（[conftest.py:181](../tests/conftest.py) 用它读 `--snapshot-exact`，决定是否精确比对）。

所以 `numpy_snapshot` 之所以"知道当前测试叫什么、该跟哪个快照比"，正是靠 `request` 拿到的上下文。

---

## 7. 快照文件路径是怎么拼出来的

第 5 节末尾提到，比对会去找 `tests/_snapshots/<测试名>.npz`。这个路径**不是在某个配置文件里单独指定的**，而是由三个来源拼起来的——目录、后缀、文件名主体各有出处。

### 7.1 目录 `tests/_snapshots/`：来自默认参数

写死在 `NumpySnapshot.__init__` 的默认参数里（[conftest.py:24-32](../tests/conftest.py)）：

```python
def __init__(
    self,
    snapshot_dir: str = "tests/_snapshots",   # ← 默认目录
    ...
):
    self.snapshot_dir = Path(snapshot_dir)
```

而 `numpy_snapshot` fixture 构造它时（[conftest.py:184-186](../tests/conftest.py)）**没有传 `snapshot_dir`**，所以就用了这个默认值。换句话说，目录是"默认值"，不是显式配置的。

### 7.2 后缀 `.npz`：硬编码

在 `_get_snapshot_path` 里直接拼死（[conftest.py:37-39](../tests/conftest.py)）：

```python
def _get_snapshot_path(self, test_name: str) -> Path:
    return self.snapshot_dir / f"{test_name}.npz"   # ← 目录 + 测试名 + .npz
```

后缀 `.npz` 对应 numpy 的压缩数组格式（`np.load` 读的就是它）。对比同文件的 `Snapshot` 类用的是 `.pkl`（[conftest.py:110-111](../tests/conftest.py)）——后缀由用哪个比对器类决定，而非配置项。

### 7.3 文件名主体 `<测试名>`：默认取测试名，可覆盖

- **默认**：fixture 里设了 `default_test_name=request.node.name`（[conftest.py:185](../tests/conftest.py)），即当前测试函数名。`assert_match` 在没收到显式名字时启用这个默认值（[conftest.py:61-63](../tests/conftest.py)）：

```python
if test_name is DEFAULT:
    assert self.default_test_name is not None, ...
    test_name = self.default_test_name   # ← 没显式传就用测试名
```

- **可覆盖**：调用 `assert_match(actual, test_name="my_special_test")` 显式传名时，就用你传的（见 [conftest.py:41-47](../tests/conftest.py) 的签名，及文件末尾注释示例）。

### 7.4 汇总

| 路径部分 | 值 | 来源 |
|---|---|---|
| 目录 | `tests/_snapshots/` | `NumpySnapshot.__init__` 的**默认参数**（fixture 未覆盖） |
| 后缀 | `.npz` | `_get_snapshot_path` **硬编码**（因为用 numpy 数组比对器） |
| 文件名主体 | 当前测试名 | fixture 设 `default_test_name=request.node.name`，**可在 `assert_match` 传 `test_name` 覆盖** |

所以对 `test_linear` 这类没传 `test_name` 的用例，快照文件就是 `tests/_snapshots/test_linear.npz`——`ls tests/_snapshots/` 可以看到这些 `.npz` 与测试名一一对应。

> 记忆要点：**目录=默认值、后缀=硬编码、名字=默认取测试名（可覆盖）**。三者拼成完整路径，没有任何一处是在 `pytest.ini`/`pyproject.toml` 里配置的。

---

## 8. fixture 的生命周期与 scope（补充）

fixture 默认 `scope="function"`：**每个测试函数都会重新执行一次 fixture**，用完即弃。这保证测试之间互不污染。

其它常见 scope（本仓库大多用默认）：

| scope | 含义 | 何时创建 |
|---|---|---|
| `function`（默认） | 每个测试各来一份 | 每条用例前 |
| `class` | 一个测试类共享 | 类中第一个用例前 |
| `module` | 一个文件共享 | 文件中第一个用例前 |
| `session` | 整轮 pytest 共享一份 | 整个运行只建一次 |

若某个准备开销很大（如加载大模型权重），可用更大的 scope 避免重复。此外，fixture 里如果用 `yield` 而非 `return`，`yield` 之后的代码会作为**清理逻辑**在测试结束后执行（类似 setup/teardown）。

---

## 9. 和 `-k` 过滤、`uv run` 的关系

回到完整命令 `uv run pytest -k test_linear`：

1. **`uv run`**：在项目 `.venv` 里执行后面的命令（缺则据 `pyproject.toml`/`uv.lock` 自动建好），与 fixture 机制无关，只负责"在对的环境里跑"。
2. **`pytest` 收集**：扫描 `tests/`，加载各级 `conftest.py`（登记所有 fixture）。
3. **`-k test_linear`**：按名字**子串**过滤，只保留匹配的用例。注意 `-k` 只影响"选哪些测试跑"，**不影响 fixture 注入逻辑**——被选中的测试照样按第 1–6 节的规则解析依赖。
4. 对每条选中的用例：解析其参数对应的 fixture 依赖树 → 求值 → 注入 → 运行 → 断言。

---

## 10. 排错与自查小抄

- **`fixture 'xxx' not found`**：参数名和 fixture 名没对上（拼写、大小写），或该 fixture 不在可见的 `conftest.py`/本文件里。因为是**按名字**匹配，名字必须完全一致。
- **想看某个测试实际用到哪些 fixture、来自哪里**：

```sh
uv run pytest --fixtures-per-test tests/test_model.py::test_linear
```

- **想列出所有可用 fixture**：`uv run pytest --fixtures`
- **快照对不上**：多半是输入未固定随机种子，或实现输出与参考不一致；先确认 fixture 的 `manual_seed` 没被改动。

---

## 11. 一页纸总结

1. **原理**：pytest 按"参数名 = fixture 名"做依赖注入——看测试要哪些参数，就去找同名 `@pytest.fixture` 调用并注入返回值。
2. **为什么自动**：`conftest.py` 是特殊文件，其 fixture 对同目录及子目录测试**免 import 自动可见**。
3. **递归**：fixture 能依赖 fixture，形成依赖树，自底向上求值（如 `d_model = n_heads * d_head`）。
4. **request**：内置 fixture，携带测试上下文，快照器靠它拿测试名、读命令行选项。
5. **可复现**：输入类 fixture 用固定 `manual_seed`，保证与 `_snapshots/` 参考输出可比对。
6. **`-k` 只筛选**测试、不改注入逻辑；`uv run` 只负责跑在对的环境里。

---

## 参考

- pytest 官方文档：Fixtures（`https://docs.pytest.org/en/stable/how-to/fixtures.html`）
- pytest 官方文档：conftest.py 与 fixture 可见性
- 本仓库：[tests/conftest.py](../tests/conftest.py)、[tests/test_model.py](../tests/test_model.py)
