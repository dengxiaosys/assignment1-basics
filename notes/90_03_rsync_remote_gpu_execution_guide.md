# 从 B 调用 C 的 GPU：rsync 增量部署与 SSH 远程执行

## 0. 适用场景

网络拓扑：

```text
Mac A ──可访问──> Linux B（开发机、代码与 code agent）
Mac A ──可访问──> Linux C（GPU 机）
Linux B ──不可直接访问──> Linux C
```

真实目标不是“让 C 永久挂载 B 的文件系统”，而是：

1. 在 B 修改代码；
2. 用很低的成本把修改送到 C；
3. 从 B 启动、观察和管理 C 上的 GPU Python 任务；
4. A 或隧道断开后，已经启动的训练尽量继续运行。

对此，推荐：

```text
反向 SSH 隧道（解决 B→C 可达性）
                +
rsync（运行前增量部署代码到 C 本地）
                +
SSH / tmux（在 C 本地环境执行）
```

相较 sshfs，这套方案不提供“每次保存后 C 的目录立即变化”，而是在**每次运行前同步一次**。但 rsync 只传变化内容，通常只需不到一秒，并且更适合长时间 GPU 训练。

---

## 1. 为什么它比 sshfs 更适合训练

| 维度 | rsync + SSH | sshfs |
|---|---|---|
| C 执行代码的位置 | C 本地磁盘 | B 的远程文件系统 |
| 每次修改后 | 运行前增量同步 | 立即可见 |
| Python import 延迟 | 本地，低 | 经过 A，受网络影响 |
| A 断线后 | tmux 中任务可继续，代码已在 C | 挂载失效；后续文件读取可能失败 |
| 环境隔离 | C 本地 `.venv` 很自然 | 容易误用挂载目录中的 B `.venv` |
| 大量小文件访问 | 快 | SSH/FUSE 往返开销明显 |
| 复杂度 | 一条隧道 + 一条同步命令 | 双向隧道 + FUSE 挂载 |

**关键判断**：训练进程需要的是“启动时拿到一份确定的代码快照”，而不是运行过程中持续读取正在变化的源码。因此本地部署比实时远程挂载更稳定。

---

## 2. 第一步：由 A 建立 B→C 的反向隧道

在 A（Mac）执行：

```bash
ssh -fN \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -R 127.0.0.1:2222:192.168.71.6:22 \
  online1
```

形成的数据路径：

```text
B:127.0.0.1:2222 -> A -> C:192.168.71.6:22
```

详细原理见 [反向 SSH 隧道原理](./90_02_reverse_ssh_tunnel_explained.md)。

在 B 的 `~/.ssh/config` 中配置：

```sshconfig
Host cuda-via-a
    HostName 127.0.0.1
    Port 2222
    User dengxiao
```

在 B 验证：

```bash
ssh cuda-via-a 'hostname; nvidia-smi'
```

看到 C 的主机名与 GPU 信息，说明控制通道可用。

---

## 3. 第二步：准备 C 的本地目录

在 B 执行：

```bash
ssh cuda-via-a '
  mkdir -p /home/dengxiao/work/cs336/assignments
  mkdir -p /home/dengxiao/data/cs336
  mkdir -p /home/dengxiao/checkpoints/cs336
  mkdir -p /home/dengxiao/logs/cs336
  mkdir -p /home/dengxiao/.venvs
'
```

目录职责：

| 目录 | 内容 |
|---|---|
| `~/work/cs336/assignments/` | rsync 过来的代码快照 |
| `~/data/cs336/` | C 本地训练数据 |
| `~/checkpoints/cs336/` | C 本地 checkpoint |
| `~/logs/cs336/` | C 本地训练日志 |
| `~/.venvs/cs336-systems` | C 本地 Python/CUDA 环境 |

代码、数据、环境和运行产物明确分开。

---

## 4. 第三步：从 B 增量同步 Assignment 1 和 2

Assignment 2 的 `pyproject.toml` 使用相对路径：

```toml
cs336-basics = { path = "../assignment1-basics", editable = true }
```

因此必须让 C 同时拥有两个兄弟目录：

```text
assignments/
├── assignment1-basics/
└── assignment2-systems/
```

### 4.1 第一次先 dry-run

在 B 执行：

```bash
cd /home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments

rsync -azn --delete --itemize-changes \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='data/' \
  --exclude='ckpt/' \
  --exclude='bpe_out/' \
  --exclude='logs/' \
  --exclude='outputs/' \
  --exclude='profiles/' \
  --exclude='wandb/' \
  assignment1-basics/ \
  cuda-via-a:/home/dengxiao/work/cs336/assignments/assignment1-basics/

rsync -azn --delete --itemize-changes \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='data/' \
  --exclude='ckpt/' \
  --exclude='logs/' \
  --exclude='outputs/' \
  --exclude='profiles/' \
  --exclude='wandb/' \
  assignment2-systems/ \
  cuda-via-a:/home/dengxiao/work/cs336/assignments/assignment2-systems/
```

`-n` 表示 dry-run，只展示将执行的操作，不实际修改 C。确认目标路径正确后再去掉 `-n`。

### 4.2 正式同步

```bash
cd /home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments

rsync -az --delete --itemize-changes \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='data/' \
  --exclude='ckpt/' \
  --exclude='bpe_out/' \
  --exclude='logs/' \
  --exclude='outputs/' \
  --exclude='profiles/' \
  --exclude='wandb/' \
  assignment1-basics/ \
  cuda-via-a:/home/dengxiao/work/cs336/assignments/assignment1-basics/

rsync -az --delete --itemize-changes \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='data/' \
  --exclude='ckpt/' \
  --exclude='logs/' \
  --exclude='outputs/' \
  --exclude='profiles/' \
  --exclude='wandb/' \
  assignment2-systems/ \
  cuda-via-a:/home/dengxiao/work/cs336/assignments/assignment2-systems/
```

参数含义：

| 参数 | 作用 |
|---|---|
| `-a` | 递归同步并保留常规文件属性 |
| `-z` | 传输时压缩，适合源码文本 |
| `--delete` | 删除 C 中已经不在 B 源目录里的代码文件 |
| `--itemize-changes` | 明确列出新增、修改、删除了什么 |
| `--exclude` | 不传环境、缓存、数据、checkpoint 等大文件 |

> `--delete` 只能对这里明确的项目目标目录使用。第一次必须先运行 dry-run，绝不能把目标误写成 `/home/dengxiao/` 等上级目录。

### 4.3 rsync 是单向同步，C 的文件不会回到 B

本命令的方向由参数位置决定：

```text
B 上的 source/  ──rsync──>  C 上的 destination/
```

因此：

- C 上生成的日志、缓存、checkpoint、临时文件**不会反向传回 B**；
- C 若修改一份源代码，该修改也不会自动回到 B，并可能在下次同步时被 B 的版本覆盖；
- `--delete` 只会修改**目标 C**：C 中存在、B 中不存在且未被排除的文件，会在下次同步时被删除；
- 被 `--exclude` 排除的文件默认也受到删除保护；只有额外使用 `--delete-excluded` 才会删除它们，本文命令没有使用该选项；
- rsync 不会修改 B 的源目录。

最稳妥的约定是把 C 的部署目录视为**只读、可随时重建的代码快照**，所有运行产物放在它外部：

```text
/home/dengxiao/work/cs336/assignments/   # rsync 管理的代码，禁止存运行产物
/home/dengxiao/data/cs336/               # 数据
/home/dengxiao/checkpoints/cs336/        # checkpoint
/home/dengxiao/logs/cs336/               # 日志、W&B、profile
/home/dengxiao/tmp/cs336/                # 临时文件
```

启动任务时显式指定这些路径，例如：

```bash
export TMPDIR=/home/dengxiao/tmp/cs336
export WANDB_DIR=/home/dengxiao/logs/cs336/wandb
export TORCH_EXTENSIONS_DIR=/home/dengxiao/.cache/torch_extensions
export TRITON_CACHE_DIR=/home/dengxiao/.cache/triton
```

这样即使使用 `--delete`，也只会让 C 的代码快照严格匹配 B，不会碰到数据和运行产物。

之后每次修改代码再执行，相同文件会跳过，只传差异。

---

## 5. 第四步：在 C 创建独立 GPU 环境

首次同步后，在 B 执行：

```bash
ssh cuda-via-a
```

进入 C 后：

```bash
cd /home/dengxiao/work/cs336/assignments/assignment2-systems

export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/cs336-systems"
export UV_CACHE_DIR="$HOME/.cache/uv"

uv sync --locked
uv run python -c 'import torch; print(f"cuda_available={torch.cuda.is_available()}")'
```

要求输出：

```text
cuda_available=True
```

原则：

- 不复制 B 的 `.venv`；
- 不让 uv 在同步目录里创建环境；
- CUDA 版 PyTorch 安装在 C 本地 `~/.venvs/cs336-systems`；
- uv 缓存也保存在 C 本地。

若后续 `pyproject.toml` 或 `uv.lock` 变化，重新运行 `uv sync --locked` 即可；依赖未变化时成本很低。

---

## 6. 第五步：从 B 执行 C 上的 Python

简单验证：

```bash
ssh cuda-via-a '
  cd /home/dengxiao/work/cs336/assignments/assignment2-systems &&
  UV_PROJECT_ENVIRONMENT="$HOME/.venvs/cs336-systems" \
  uv run python -c "import torch; print(f\"device={torch.cuda.get_device_name(0)}\")"
'
```

运行测试：

```bash
ssh cuda-via-a '
  cd /home/dengxiao/work/cs336/assignments/assignment2-systems &&
  UV_PROJECT_ENVIRONMENT="$HOME/.venvs/cs336-systems" \
  uv run pytest tests/test_flash_attention.py
'
```

每次运行的固定流程是：

```text
B 修改代码 -> rsync 增量同步 -> ssh C 执行
```

---

## 7. 长任务用 tmux

从 B 进入 C 的 tmux：

```bash
ssh -t cuda-via-a 'tmux new -A -s cs336'
```

在 tmux 内执行：

```bash
cd /home/dengxiao/work/cs336/assignments/assignment2-systems
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/cs336-systems"
uv run python your_training_script.py \
  > /home/dengxiao/logs/cs336/train.log 2>&1
```

按 `Ctrl-b`，再按 `d`，可从 tmux 分离。之后在 B 查看日志：

```bash
ssh cuda-via-a 'tail -f /home/dengxiao/logs/cs336/train.log'
```

A 或隧道断开后：

- C 本地代码已经同步完成；
- Python 环境、数据和 checkpoint 都在 C 本地；
- tmux 中的训练继续运行；
- 恢复隧道后可重新查看。

这是本方案相对 sshfs 的核心稳定性优势。

---

## 8. 可选：在 B 包装为一个命令

可以在 B 创建 `run_on_cuda.sh`，把同步与执行组合起来：

```bash
#!/usr/bin/env bash
set -euo pipefail

ASSIGNMENTS=/home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments
REMOTE=cuda-via-a
REMOTE_ROOT=/home/dengxiao/work/cs336/assignments
REMOTE_ENV=/home/dengxiao/.venvs/cs336-systems

sync_project() {
  local name=$1
  rsync -az --delete \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='.pytest_cache/' \
    --exclude='.ruff_cache/' \
    --exclude='data/' \
    --exclude='ckpt/' \
    --exclude='bpe_out/' \
    --exclude='logs/' \
    --exclude='outputs/' \
    --exclude='profiles/' \
    --exclude='wandb/' \
    "$ASSIGNMENTS/$name/" \
    "$REMOTE:$REMOTE_ROOT/$name/"
}

sync_project assignment1-basics
sync_project assignment2-systems

printf -v command '%q ' "$@"
ssh "$REMOTE" \
  "cd '$REMOTE_ROOT/assignment2-systems' && \
   UV_PROJECT_ENVIRONMENT='$REMOTE_ENV' uv run $command"
```

调用：

```bash
chmod +x run_on_cuda.sh
./run_on_cuda.sh python -c 'import torch; print(torch.cuda.get_device_name(0))'
./run_on_cuda.sh pytest tests/test_flash_attention.py
```

`printf '%q'` 会安全转义传入的参数，避免简单字符串拼接破坏空格和引号。

> 该脚本适合短命令和测试。长时间训练仍建议通过 tmux 启动，避免 SSH 会话中断影响前台进程。

---

## 9. 何时仍应使用 sshfs

只有下面情况更适合 sshfs：

- 必须在 C 上交互式浏览 B 的完整目录；
- C 上的工具必须持续读取 B 刚保存的非 Python 文件；
- 不希望 C 保留任何源码副本；
- 能接受 A 持续在线以及远程文件系统延迟。

若目标只是“B 写代码、C 跑 Python/GPU”，优先使用本文的 rsync + SSH。

---

## 10. 常见问题

### 10.1 rsync 报连接失败

先验证隧道：

```bash
ssh cuda-via-a 'hostname'
```

若失败，按 [反向 SSH 隧道原理](./90_02_reverse_ssh_tunnel_explained.md) §8 排查监听、A→C 可达性和认证。

### 10.2 C 仍加载旧代码

检查同步输出是否包含目标文件：

```bash
rsync -azn --delete --itemize-changes ...
```

同时确认 Python 导入路径：

```bash
ssh cuda-via-a '
  cd /home/dengxiao/work/cs336/assignments/assignment2-systems &&
  UV_PROJECT_ENVIRONMENT="$HOME/.venvs/cs336-systems" \
  uv run python -c "import cs336_basics; print(f\"module_path={cs336_basics.__file__}\")"
'
```

### 10.3 修改代码后，正在运行的训练没有变化

这是正常现象。Python 进程启动时已经把模块加载进内存；同步更新磁盘文件不会热替换运行中的类和函数。需要停止旧进程并重新启动。

### 10.4 是否每次都要 `uv sync`

不需要。只有 `pyproject.toml` 或 `uv.lock` 改变时才需同步环境。普通 `.py` 文件变化只需 rsync 后重启 Python 命令。

---

## 11. 小结

1. A 只负责提供 B→C 的反向 SSH 通路。
2. B 用 rsync 把 Assignment 1/2 的代码差量部署到 C 本地。
3. C 使用本地源码、本地 CUDA 环境、本地数据和本地 checkpoint。
4. B 通过 SSH 或 tmux 启动和管理 C 的任务。
5. 该方案只需一条隧道，且 A 断线后已启动训练仍能继续，比 sshfs 更适合 GPU 训练。
