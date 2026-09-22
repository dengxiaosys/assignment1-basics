# 跨机开发：在 B 上改代码、马上在 C（带 GPU）上运行

## 0. 背景与网络拓扑

**场景**：Mac A 用 VSCode Remote 连接 Linux B 做开发（本仓库、code agent 都在 B 上）。A 还能连另一台 Linux C（C 有 GPU，但不具备跑 code agent 的条件）。**B 和 C 之间无法互相访问**。

```text
        ┌─────────┐
        │  Mac A  │   ← 唯一能同时到达 B 和 C 的节点（枢纽）
        └────┬────┘
       ssh  / \  ssh
           /   \
   ┌──────┴─┐ ┌─┴──────┐
   │Linux B │ │Linux C │
   │(开发/  │ │(GPU/   │
   │ CPU)   │ │ 训练)  │
   └────────┘ └────────┘
     B 和 C 互不可达 ✗
```

**目标**：在 B 上（通过 VSCode）改代码后，**马上能在 C 上用 GPU 运行**，且**不经过复杂、耗时的代码同步**。

**核心约束（决定一切方案的形状）**：B↔C 不通，但 A 同时连着两者 → **任何 B 与 C 之间的通路都必须借道 A 中转**。思路就是"用 A 把 C 引到 B 面前"。

> 其它候选方案（供对比，本文只详记方案一）：
> - **方案三 rsync over A**：在 A 上 B→A→C 单向增量推送，简单可靠，但每次改完要手动/定时触发一次。
> - **方案四 git 借道 A 中转**：A 上建裸仓库，B push / C pull，有版本记录，但要 commit，非"改完即生效"。
> - 方案一（下面）最贴合"改完马上就能跑、零显式同步"。

---

## 1. 方案一：C 经 A 反向隧道，用 sshfs 挂载 B 的代码目录

**一句话**：让 C 通过 `sshfs` **挂载 B 上的代码目录**——C 看到的就是 B 的实时文件。你在 B 上（VSCode）改完保存，C 立刻看到新代码，直接运行即可，**没有"同步"这一步**。

难点只在于 B、C 不通，所以要靠 A 建一条**反向 SSH 隧道**，把 B 的 SSH 端口"经 A 送到 C"。

### 1.1 步骤①：在 A 上建反向隧道（把 B 的 22 端口暴露到 C 的 localhost:2222）

在 **A** 上执行（占位符 `userX@X_host` 改成真实值）：

```bash
# A 连到 C，并在 C 上开一个本地端口 2222，转发（经 A）回 B 的 22 端口
ssh -R 2222:B_host:22 userC@C_host
# 保持这个会话开着；或用 -fN 放后台：
# ssh -fN -R 2222:B_host:22 userC@C_host
```

含义：`-R 2222:B_host:22` = "在远端（C）监听 2222，任何到 C:localhost:2222 的连接，都经由这条 SSH（A→C）转发到 `B_host:22`"。于是**在 C 上访问 `localhost:2222` 就等于访问 B 的 SSH**。

> 前提：A 能 ssh 到 B（用于解析 `B_host:22`）和 C。若 A→B 需要密钥/跳板，先确保 A 能直接 `ssh userB@B_host` 成功。

### 1.2 步骤②：在 C 上用 sshfs 挂载 B 的代码目录

在 **C** 上执行（通过隧道，`localhost:2222` 就是 B）：

```bash
mkdir -p ~/mnt_B_code
sshfs -p 2222 \
  userB@localhost:/home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment1-basics \
  ~/mnt_B_code \
  -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3
```

挂载后，`~/mnt_B_code` 就是 B 上那个目录的**实时视图**。B 改完保存 → C 立即可见。

### 1.3 步骤③：在 C 上运行（代码来自 B，数据/环境在 C 本地）

```bash
cd ~/mnt_B_code
# 用 C 本地的 GPU 环境跑 B 的实时代码
python -m cs336_basics.train --train-path /data_on_C/ts_train.npy --device cuda ...
```

---

## 2. 关键注意事项（否则会踩坑）

### 2.1 代码挂 B，但数据 / checkpoint / 环境要放 C 本地

- **数据（大文件）放 C 本地**：`ts_train.npy` 等经网络读会慢。训练数据应在 **C 本地磁盘**，只有**代码**挂载自 B。即"**代码来自 B、数据在 C 本地**"。
- **checkpoint 写 C 本地**：`--checkpoint-out` 指向 C 本地路径，避免大文件反复走网络写回 B。
- **Python 环境必须在 C 上独立建**：B 是 CPU 环境，C 要 **GPU 版 PyTorch**。`.venv` 含平台相关二进制，**绝不能跨机共享**。在 C 上单独 `uv sync` / 建 venv。
  - 因此挂载时**最好只挂代码子目录**（如上面挂到 `assignment1-basics`，在 C 上另建 `.venv`），或挂载后在 C 用独立环境运行，忽略 B 的 `.venv`。

### 2.2 隧道保活

- 裸 `ssh -R` 断线后隧道就没了。长时间训练建议用 **autossh**（若可装）自动重连：
  ```bash
  autossh -M 0 -fN -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -R 2222:B_host:22 userC@C_host
  ```
- sshfs 侧加 `-o reconnect,ServerAliveInterval=15`（上面已加）能在短暂抖动后自动重连。

### 2.3 C 上 sshd 需允许端口转发

- 反向隧道要求 **C 的 sshd** 配置 `AllowTcpForwarding yes`（多数默认开启）。若 2222 端口连不通 B，先查这一项。
- 端口 2222 可换任意未占用端口，两处保持一致即可。

### 2.4 卸载

```bash
# 在 C 上
fusermount3 -u ~/mnt_B_code    # 或 fusermount -u
```

---

## 3. 备选：方案三（A 上 rsync 中转，手动/定时）

若某天隧道不便用，退而求其次——在 **A** 上把 B 的代码增量推到 C（A 是唯一能同时够到 B、C 的机器）：

```bash
# 在 A 上：先从 B 拉到 A，再从 A 推到 C（--delete 保持一致，排除 .venv）
rsync -az --delete --exclude '.venv' --exclude '__pycache__' \
  userB@B_host:/path/.../assignment1-basics/cs336_basics/ /tmp/relay/
rsync -az --delete /tmp/relay/ userC@C_host:~/code/cs336_basics/
```

土办法自动化（A 上每 2 秒同步一次）：

```bash
while true; do
  rsync -az --delete --exclude '.venv' userB@B_host:/path/src/ /tmp/relay/ 2>/dev/null
  rsync -az --delete /tmp/relay/ userC@C_host:~/code/src/ 2>/dev/null
  sleep 2
done
```

对比方案一：rsync 是"推一次生效一次"，方案一是"改完即生效"。追求零同步延迟用方案一。

---

## 4. 待办清单（改天回来试）

- [ ] 填入真实 `userB@B_host`、`userC@C_host`、端口。
- [ ] 确认 A 能分别 `ssh` 到 B 和 C。
- [ ] 在 A 上建反向隧道（§1.1），确认 C 上 `ssh -p 2222 userB@localhost` 能连上 B。
- [ ] 在 C 上 `sshfs` 挂载（§1.2），`ls ~/mnt_B_code` 看到 B 的代码。
- [ ] 在 C 上 `uv sync` 建 GPU 环境（**不要**用 B 的 `.venv`）。
- [ ] 把训练数据 `ts_train.npy` 放到 C 本地，`--checkpoint-out` 指向 C 本地。
- [ ] （可选）用 `autossh` 给隧道保活。

---

## 5. 小结

- **根本约束**：B↔C 不通，A 是唯一枢纽 → 一切通路借道 A。
- **方案一（推荐）**：A 建反向隧道把 B 的 SSH 送到 C → C 用 sshfs 挂载 B 的代码目录 → **B 改完即生效，C 直接用 GPU 跑，零显式同步**。
- **铁律**：**代码来自 B，数据/checkpoint/Python 环境在 C 本地**（`.venv` 绝不跨机共享）。
- **备选**：A 上 rsync 中转（手动/定时）、A 做中转 git（有版本管理）。



## 6. 实际主机定义

以下别名配置位于 A（Mac）的 `~/.ssh/config`。

### 6.1 B：开发机 `online1`

```sshconfig
Host online1
    HostName 10.37.102.220
    User dengxiao.cs
    Port 16101
    GSSAPIAuthentication yes
    GSSAPIDelegateCredentials no
```

### 6.2 C：GPU 机 `cuda`

```sshconfig
Host cuda
    HostName 192.168.71.6
    User dengxiao
```

---

## 7. 从 B 操作 C：由 A 提供反向 SSH 隧道

这是一套方案：A 建立反向隧道，使 B 通过 A 连接 C。

### 7.1 在 A 上建立隧道

```bash
ssh -fN -R 2222:192.168.71.6:22 online1
```

### 7.2 在 B 上连接 C

```bash
ssh -p 2222 dengxiao@localhost
```

### 7.3 可选：在 B 上配置 SSH 别名

将以下内容加入 B 的 `~/.ssh/config`：

```sshconfig
Host cuda-via-a
    HostName localhost
    Port 2222
    User dengxiao
```

以后可简写为：

```bash
ssh cuda-via-a
```

别名只是简化命令，不是另一套方案。隧道在 A 断网、休眠或重启后需要
重新建立。