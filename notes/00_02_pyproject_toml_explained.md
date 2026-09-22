# `pyproject.toml` 逐项详解：这份配置每一行都在做什么

## 0. 本文目标

CS336 assignment1 用 `uv` 管理环境，项目的一切元数据、依赖、构建方式、工具配置都集中在 [pyproject.toml](../pyproject.toml)。本文**逐个表（table）、逐个键**解释它的作用，并补足背后的背景知识：什么是 `pyproject.toml`、`[project]` 与 `[tool.*]` 的分工、版本约束符号（`>=` / `~=`）的含义、以及 `uv` / `ruff` / `pytest` 各自读哪一段。

> 说明：`pyproject.toml` 是 Python 官方推荐的项目配置文件（PEP 518/621），用 [TOML](https://toml.io) 格式。它取代了过去零散的 `setup.py`、`setup.cfg`、`requirements.txt`、各种 `.ini`，把"项目是什么、依赖什么、怎么构建、各工具怎么配"统一到一个文件。

---

## 1. 全局结构：两类表

整个文件由若干 `[表名]` 段组成，分两类：

- **`[project]` 与 `[build-system]`**：**标准表**（PEP 621/518 定义），任何构建工具都认。描述"项目本身"和"用什么来构建它"。
- **`[tool.xxx]`**：**工具私有表**，每个工具（`uv`、`ruff`、`pytest`）只读自己那一段。这是 PEP 518 约定的"工具命名空间"，互不干扰。

下面按文件顺序逐段讲。

---

## 2. `[project]`：项目核心元数据（PEP 621 标准）

```toml
[project]
name = "cs336_basics"
version = "26.0.0"
description = "CS336 Assignment 1"
readme = "README.md"
requires-python = ">=3.12,<3.14"
dependencies = [ ... ]
```

| 键 | 作用 |
|---|---|
| `name` | **包名** `cs336_basics`。安装后 `import cs336_basics`、`pip show cs336_basics` 都用它。 |
| `version` | **版本号** `26.0.0`。遵循语义化版本（主.次.补丁）；这里是课程约定的年份式版本。 |
| `description` | 一句话**项目描述**，显示在包索引/`pip show` 里。 |
| `readme` | **长描述来源文件**，指向 `README.md`。发布时其内容作为项目主页正文。 |
| `requires-python` | **Python 版本约束** `>=3.12,<3.14`：要求 ≥3.12 且 <3.14（即 3.12 或 3.13）。`uv` 据此挑选/创建解释器；不满足会拒绝安装。 |
| `dependencies` | **运行期依赖列表**（详见 §3）。 |

**背景**：`[project]` 是**跨工具通用**的标准。换成 poetry、hatch、pip 也都读这一段——这正是 PEP 621 的意义：元数据标准化，不再和构建工具绑死。

---

## 3. `dependencies`：依赖与版本约束符号

```toml
dependencies = [
    "einops>=0.8",      # tensor operations
    "einx>=0.4",        # more general tensor operations
    "jaxtyping>=0.3",   # type hints for pytorch/jax/numpy
    "numpy>=2.4",
    "psutil>=7",
    "pytest>=9.0",
    "pytest-timeout",
    "regex>=2026.3.32", # more powerful regex than the builtin `re` module
    "tiktoken>=0.12.0",
    "torch~=2.11.0",
    "tqdm>=4.67",
    "wandb>=0.25",
    "ty>=0.0.26",
    "ruff>=0.15.8",
]
```

### 3.1 先看版本约束符号（关键背景）

- **`>=X`**：不低于 X 的任意版本。例：`numpy>=2.4` 允许 2.4、2.5、3.0……只要 ≥2.4。最宽松。
- **`~=X.Y.Z`**（兼容版本，"compatible release"）：允许最后一位增长、但**锁住前面**。`torch~=2.11.0` 等价于 `>=2.11.0, <2.12.0`——补丁能升（2.11.1、2.11.2），但不会跳到 2.12。用于**既要修复又怕破坏 API** 的库（torch 小版本间常有行为变化，故锁得紧）。
- **无符号**（如 `pytest-timeout`）：**任意版本**，装最新兼容的即可（该插件稳定、无需约束）。

> 记忆：`>=` 只设下限；`~=` 设一个"补丁级"上限；不写则完全不限。约束越紧越可复现，越松越易升级——权衡而已。

### 3.2 各依赖是干什么的

| 依赖 | 用途 |
|---|---|
| `einops` / `einx` | 张量重排/爱因斯坦记号操作，写形状变换更清晰。 |
| `jaxtyping` | 给张量加**带形状的类型注解**（如 `Float[Tensor, "batch seq"]`），adapters.py 里大量用到。 |
| `numpy` | 数值数组基础（数据加载、`get_batch` 的 memmap 等）。 |
| `psutil` | 读进程/系统资源（内存、CPU），用于 profile/监控。 |
| `pytest` / `pytest-timeout` | **测试框架**及其超时插件（防止个别测试卡死）。 |
| `regex` | 比内置 `re` 更强的正则（支持 `\p{L}` 等），BPE 预分词的 GPT-2 正则需要它。 |
| `tiktoken` | OpenAI 的 BPE 分词器库，作参考/对拍。 |
| `torch` | **核心**：PyTorch，所有模型/优化器/自动微分。 |
| `tqdm` | 进度条。 |
| `wandb` | Weights & Biases，实验指标记录（训练循环里可选接入）。 |
| `ty` | 类型检查器（astral 家的 type checker）。 |
| `ruff` | 极快的 linter + formatter（见 §8）。 |

> 注意：这里把 `pytest`、`ruff`、`ty` 也放进了 `dependencies`（而非单独的 dev 依赖组）。严格说它们是开发工具，但课程为简化统一放主依赖，`uv run` 时全都可用——对作业够用。

---

## 4. `[build-system]`：用什么来"构建"这个项目

```toml
[build-system]
requires = ["uv_build>=0.11.0,<0.12.0"]
build-backend = "uv_build"
```

| 键 | 作用 |
|---|---|
| `requires` | **构建时**需要的工具（构建后端本身），这里用 `uv_build`，版本锁在 0.11.x。 |
| `build-backend` | 指定**构建后端**为 `uv_build`——即由 uv 自己的后端把源码打成可安装的包（wheel/sdist）。 |

**背景**：PEP 517/518 把"构建"抽象成"前端 + 后端"。前端（pip、uv）负责调度，**后端**负责真正把源码变成包。常见后端有 `setuptools.build_meta`、`hatchling`、`flit_core`，这里选 `uv_build`（uv 原生后端，快且与 uv 工作流一致）。`requires` 保证构建环境里有对的后端版本——**它是构建期依赖，和 §3 的运行期 `dependencies` 是两回事**。

---

## 5. `[project.scripts]`：命令行入口

```toml
[project.scripts]
cs336-train = "cs336_basics.train:main"
```

- 定义一个**控制台脚本**：安装后终端里可直接敲 `cs336-train`，它会调用 `cs336_basics/train.py` 里的 `main` 函数（`模块路径:函数名` 格式）。
- 这就是之前 `uv run cs336-train --help` 能用的原因——本项目自己注册的训练入口。
- **背景**：`[project.scripts]` 是 PEP 621 标准键，构建时会生成一个可执行包装器放进环境的 `bin/`，`PATH` 里可见。

---

## 6. `[tool.uv.build-backend]`：告诉 uv 包在哪

```toml
[tool.uv.build-backend]
module-name = "cs336_basics"
module-root = ""
```

| 键 | 作用 |
|---|---|
| `module-name` | 要打包的**顶层模块名** `cs336_basics`。 |
| `module-root` | 模块所在的**根目录**。`""`（空串）表示**包就在项目根下**（即 `./cs336_basics/`），而非常见的 `src/` 布局（那种会写 `module-root = "src"`）。 |

**背景（已核实本项目布局）**：项目根直接就有 `cs336_basics/` 目录，没有 `src/` 中间层——所以 `module-root = ""`。这两行让 `uv_build` 后端准确定位到要打包的代码。若哪天改成 src-layout，把包挪进 `src/` 并设 `module-root = "src"` 即可。

---

## 7. `[tool.uv]`：uv 自身的行为

```toml
[tool.uv]
package = true
python-preference = "managed"
```

| 键 | 作用 |
|---|---|
| `package = true` | 明确告诉 uv **这是一个可安装的包**（要走构建后端、可 `import`、可注册脚本），而不是一个"只装依赖不建包"的纯应用工程。 |
| `python-preference = "managed"` | 解释器偏好：**优先使用 uv 自己下载管理的 Python**，而非系统里的。好处是版本可控、可复现，避免"系统 Python 被污染/版本不符"的麻烦。 |

**背景**：`uv` 会据此在需要时自动拉取符合 `requires-python`（§2）的解释器、建 `.venv`、按 `uv.lock` 装依赖——这就是 README 里"`uv run` 会自动解好环境"的机制来源。

---

## 8. 工具配置：pytest 与 ruff

### 8.1 `[tool.pytest.ini_options]`：测试运行选项

```toml
[tool.pytest.ini_options]
log_cli = true
log_cli_level = "WARNING"
addopts = "-s"
```

| 键 | 作用 |
|---|---|
| `log_cli = true` | 让测试运行时**把日志实时打到控制台**（而非只在失败摘要里）。 |
| `log_cli_level = "WARNING"` | 控制台日志级别阈值：只显示 WARNING 及以上，过滤掉 INFO/DEBUG 噪声。 |
| `addopts = "-s"` | 每次 `pytest` 默认追加的参数。`-s` = 不捕获 stdout，**让 `print` 直接显示**（调试时能看到打印）。 |

### 8.2 `[tool.ruff]` 及其子表：代码风格

```toml
[tool.ruff]
line-length = 120

[tool.ruff.lint.extend-per-file-ignores]
"__init__.py" = ["E402", "F401", "F403", "E501"]

[tool.ruff.lint]
extend-select = ["UP"]
ignore = ["F722"]
```

| 键 | 作用 |
|---|---|
| `line-length = 120` | 每行**最大 120 字符**（默认 88），超出会被 lint 提示。 |
| `extend-per-file-ignores` 里 `"__init__.py" = [...]` | 对 **`__init__.py` 单独放宽**若干规则：`E402`（import 不在文件顶部）、`F401`（导入未使用）、`F403`（`import *`）、`E501`（行过长）。因为 `__init__.py` 常故意做"汇总导出"，这些告警属误伤。 |
| `[tool.ruff.lint]` `extend-select = ["UP"]` | **额外启用** `pyupgrade`（`UP`）规则集：提示把旧写法升级成新语法（如老式类型注解 → 新式）。 |
| `[tool.ruff.lint]` `ignore = ["F722"]` | **全局忽略** `F722`（"语法错误的 forward 注解"）。因为 `jaxtyping` 的形状注解如 `Float[Tensor, "batch seq"]` 里的字符串会被误判为非法注解，需关掉。 |

**背景**：`ruff` 一个工具同时干 flake8（lint）+ isort（排序）+ 部分 pyupgrade 的活，极快。规则用**字母+数字编码**（`E`=pycodestyle、`F`=pyflakes、`UP`=pyupgrade…）。`select`/`ignore` 控制启用哪些、忽略哪些，`per-file-ignores` 做**按文件**的精细豁免。

---

## 9. 小结：谁读哪一段

| 表 | 谁读 | 管什么 |
|---|---|---|
| `[project]` | 所有工具（标准） | 名称/版本/描述/Python 约束/运行依赖 |
| `[build-system]` | 构建前端（pip/uv） | 用哪个后端、构建期依赖 |
| `[project.scripts]` | 构建后端 | 生成命令行入口 `cs336-train` |
| `[tool.uv.build-backend]` | uv 构建后端 | 包名与包根目录（本项目非 src-layout） |
| `[tool.uv]` | uv | 当作包处理、用托管 Python |
| `[tool.pytest.ini_options]` | pytest | 日志与默认参数 |
| `[tool.ruff]` / `[tool.ruff.lint.*]` | ruff | 行宽、规则启用/忽略、按文件豁免 |

一句话：**`[project]`/`[build-system]` 是标准元数据与构建声明，`[project.scripts]` 造入口，`[tool.*]` 各归各的工具**。理解这套划分后，以后加依赖、改行宽、加命令行工具就都知道该动哪一段了。

---

## 参考

- 本项目配置：[pyproject.toml](../pyproject.toml)
- 训练入口：[cs336_basics/train.py](../cs336_basics/train.py)（`cs336-train` 指向的 `main`）
- 标准：PEP 621（`[project]`）、PEP 517/518（`[build-system]`）
- 工具文档：`uv`（`https://docs.astral.sh/uv/`）、`ruff`（`https://docs.astral.sh/ruff/`）、`pytest`（`https://docs.pytest.org/`）
