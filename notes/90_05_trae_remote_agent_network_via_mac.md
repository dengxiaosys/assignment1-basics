# Trae Remote Agent 通过 Mac A 网络访问模型服务

## 0. 结论与日常命令

Trae 客户端虽然运行在 Mac A，但 Remote SSH 连接 Linux C 后，Trae Server 和 `ai-agent` 实际运行在 C。Agent 的模型请求默认从 C 出网，不会自动复用 A 的网络。

当前稳定方案是：

```text
C 上的 ai-agent
    │ HTTP CONNECT
    ▼
C:127.0.0.1:17891
    │ SSH RemoteForward
    ▼
A:127.0.0.1:17900
    │ gost HTTP proxy
    ▼
A 的网络出口
    │ HTTPS / WebSocket
    ▼
模型服务
```

推荐方式只要求在 A 运行一个控制脚本：

```bash
ctl="$HOME/.local/bin/trae_remote/a_trae_proxy.sh"

"$ctl" start
"$ctl" verify
"$ctl" stop
```

其中：

- `start`：启动或复用代理与隧道，配置 C，并按需重启 Trae Server；
- `verify`：验证 A、C、Trae API 和 `ai-agent` 的完整状态；
- `stop`：撤销 C 的代理配置，再停止隧道和 `gost`。

这些操作是幂等的。重复执行同一个动作不会反复修改 C 的 `~/.profile`，也不会无意义地重启 Trae。

---

## 1. 架构与职责

### 1.1 A 上的组件

| 组件 | 位置 | 作用 |
|---|---|---|
| `a_trae_proxy.sh` | `$HOME/.local/bin/trae_remote/` | 唯一的日常控制入口 |
| `c_configure_trae_proxy.sh` | 与 A 脚本同目录 | 每次操作前上传到 C |
| `gost` | `127.0.0.1:17900` | 提供 HTTP CONNECT 代理 |
| SSH ControlMaster | A 到 C | 维护反向端口转发 |

SSH 反向转发的核心参数为：

```text
-R 127.0.0.1:17891:127.0.0.1:17900
```

含义是：

| 字段 | 所在机器 | 含义 |
|---|---|---|
| 第一个 `127.0.0.1:17891` | C | 只在 C 的 loopback 上监听 |
| 第二个 `127.0.0.1:17900` | A | 连接 A 本机的 `gost` |
| `cuda` | A 的 SSH config | 指向 C 的主机别名 |

第二个 `127.0.0.1` 属于发起 SSH 的 A，不属于 C。更完整的原理见 [反向 SSH 隧道原理](./90_02_reverse_ssh_tunnel_explained.md)。

### 1.2 C 上的组件

A 脚本每次执行时，都会先将同目录下的 C helper 原子上传到：

```text
$HOME/.local/bin/c_configure_trae_proxy.sh
```

C helper 负责：

1. 验证 `127.0.0.1:17891` 和 Trae API；
2. 幂等管理 `~/.profile` 中的代理配置块；
3. 检查运行中 `ai-agent` 的实际环境；
4. 仅在环境需要变化时重启 Trae Server；
5. 在修改 `~/.profile` 前创建备份。

C helper 是内部实现，不需要在 C 手工调用。

### 1.3 `start` 的执行顺序

在 A 执行 `start` 后：

1. 将最新版 C helper 上传到 C；
2. 启动或复用 A 的 `gost`；
3. 启动或复用 SSH ControlMaster；
4. 在 C 建立 `127.0.0.1:17891`；
5. 验证 A 和 C 的代理出口以及 Trae API；
6. 自动读取 C 所见的 A 来源 IP；
7. 在 C 的 `~/.profile` 中写入受 marker 管理的代理块；
8. 若运行中的 `ai-agent` 尚未继承代理，则重启 Trae Server。

### 1.4 `stop` 的执行顺序

在 A 执行 `stop` 后：

1. 再次上传最新版 C helper；
2. 从 C 的 `~/.profile` 删除受管理的代理块；
3. 若运行中的 `ai-agent` 仍继承该代理，则重启 Trae Server；
4. 关闭 SSH ControlMaster 和反向转发；
5. 停止脚本管理的 `gost`。

因此，`stop` 后 C 不会遗留一个指向失效端口的代理配置。

---

## 2. 固定配置与依赖

### 2.1 当前固定值

| 项目 | 值 |
|---|---|
| A 的 SSH host alias | `cuda` |
| A 的当前局域网地址 | `192.168.71.121` |
| C 的当前局域网地址 | `192.168.71.6` |
| A 的 HTTP 代理 | `127.0.0.1:17900` |
| C 的代理入口 | `127.0.0.1:17891` |
| A 的脚本目录 | `$HOME/.local/bin/trae_remote/` |
| C helper 的目标路径 | `$HOME/.local/bin/c_configure_trae_proxy.sh` |

A 来源 IP 默认由脚本通过 `SSH_CLIENT` 自动探测。A 的局域网 IP 改变后，重新执行 `start` 即可更新 C。

### 2.2 A 的依赖

```text
bash
curl
gost
lsof
scp
ssh
```

确认：

```bash
command -v bash curl gost lsof scp ssh
gost -V
```

### 2.3 C 的依赖

```text
awk
bash
curl
grep
pgrep
ps
ss
```

这些工具只由 C helper 使用，不需要在 C 安装代理软件。

### 2.4 SSH host alias

A 的 `~/.ssh/config` 至少需要：

```sshconfig
Host cuda
    HostName 192.168.71.6
    User dengxiao
```

验证：

```bash
ssh -o ClearAllForwardings=yes cuda 'hostname'
```

---

## 3. 首次安装

### 3.1 将两个脚本复制到 A

仓库中的源文件：

- [A 端控制脚本](./scripts/trae_remote_agent_proxy/a_trae_proxy.sh)
- [C 端 helper](./scripts/trae_remote_agent_proxy/c_configure_trae_proxy.sh)

若 A 通过 `online1` 访问当前开发机 B，在 A 执行：

```bash
install_dir="$HOME/.local/bin/trae_remote"
mkdir -p "$install_dir"

scp \
  online1:/home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment1-basics/notes/scripts/trae_remote_agent_proxy/a_trae_proxy.sh \
  online1:/home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment1-basics/notes/scripts/trae_remote_agent_proxy/c_configure_trae_proxy.sh \
  "$install_dir/"

chmod 0755 "$install_dir/"*.sh
```

两个脚本必须保留在同一目录，因为 A 脚本通过自身目录定位 C helper。

### 3.2 一次性清理旧手工隧道

早期手工执行的 `ssh -fN -R ...` 没有 control socket，A 脚本无法安全接管。首次切换到脚本前，在 A 检查：

```bash
pgrep -af 'ssh.*17891.*17900.*cuda'
```

若存在旧隧道，核对命令行后只终止对应 PID：

```bash
kill <旧隧道 PID>
```

不要使用 `pkill ssh`，否则会终止 A 上其它 SSH 会话。

若 `17900` 已由手工启动的 `gost` 监听：

```bash
lsof -nP -iTCP@127.0.0.1:17900 -sTCP:LISTEN
```

确认进程确实是专用 `gost` 后，可以将其 PID 交给脚本管理：

```bash
printf '%s\n' '<gost PID>' >"$HOME/.trae-gost.pid"
```

也可以先终止该进程，再让 `start` 创建新的 `gost`。

### 3.3 首次启动

在 A 的普通终端执行：

```bash
"$HOME/.local/bin/trae_remote/a_trae_proxy.sh" start
```

关键输出应包含：

```text
c_helper_status=deployed
gost_status=started
ssh_tunnel_status=started
trae_api_root_http_code=404
profile_status=configured
```

若组件已经存在，也可能看到：

```text
gost_status=already_running
ssh_tunnel_status=already_running
profile_status=already_configured
```

如果脚本更新了 `ai-agent` 的环境，它会重启 Trae Server，Remote 窗口可能暂时断开。这是预期行为。

重新连接 `cuda` 后，在 A 执行：

```bash
"$HOME/.local/bin/trae_remote/a_trae_proxy.sh" verify
```

然后在 Trae Remote 窗口发送：

```text
只回复 OK
```

收到正常回答才表示端到端链路已经完成验证。

---

## 4. 日常操作

所有命令都在 A 的普通终端执行：

```bash
ctl="$HOME/.local/bin/trae_remote/a_trae_proxy.sh"
```

| 目的 | 命令 |
|---|---|
| 启动或修复链路 | `"$ctl" start` |
| 查看状态 | `"$ctl" status` |
| 完整网络验收 | `"$ctl" verify` |
| 重建 A 侧代理和隧道 | `"$ctl" restart` |
| 完整停止并撤销 C 配置 | `"$ctl" stop` |

### 4.1 什么时候执行 `start`

以下情况直接重跑 `start`：

- A 刚开机；
- A 睡眠后 SSH 隧道断开；
- A 的局域网 IP 变化；
- C 重启；
- C 上的 helper 丢失；
- C 的 `~/.profile` 配置被删除；
- Trae Agent 再次报告网络错误。

`start` 会在每次运行前上传最新版 C helper，因此不需要单独维护 C 上的脚本。

### 4.2 幂等性

重复执行 `start` 时：

- 已运行且 PID 匹配的 `gost` 会被复用；
- 已运行且 control socket 有效的 SSH 隧道会被复用；
- 内容完全一致的 `~/.profile` 不会重写，也不会新增备份；
- 已继承正确代理的 `ai-agent` 不会重启。

重复执行 `stop` 时：

- 已删除的配置不会再次修改；
- 已停止的隧道和 `gost` 不会再次终止；
- 未继承受管理代理的 `ai-agent` 不会重启。

### 4.3 `status` 与 `verify` 的区别

`status` 检查进程、listener、C 配置和 `ai-agent` 环境。

`verify` 还会实际访问：

```text
https://api.ipify.org
https://api.trae.com.cn/
```

Trae API 根路径预期返回 HTTP `404`。该状态表示 DNS、TCP、TLS 和 HTTP 均已成功，只是根路径没有业务路由。

若尚未打开 Trae Remote 窗口，C 上可能没有 `ai-agent`，此时 `status` 或 `verify` 会报告 `ai_agent_pid=not_running`。先连接 `cuda`，再执行验证。

### 4.4 日志与状态文件

脚本不记录运行日志：

- `gost` 的标准输出和错误输出进入 `/dev/null`；
- SSH 不使用 `-E`；
- 状态只打印到当前终端。

A 仅保留两个管理状态：

```text
$HOME/.trae-gost.pid
$HOME/.local/state/trae-agent-proxy/ssh-control.sock
```

C 在真实修改 `~/.profile` 前会创建：

```text
$HOME/.profile.backup-before-trae-proxy-<timestamp>
```

这些文件是配置备份，不是运行日志。重复同一状态操作不会继续创建备份。

---

## 5. C 上的代理配置

`start` 会在 C 的 `~/.profile` 中维护以下 marker 块，`stop` 会完整删除它：

```bash
# >>> trae-remote-agent-proxy >>>
# Route Trae Remote agent traffic through the HTTP tunnel on Mac A.
case "${SSH_CLIENT:-}" in
    192.168.71.121\ *)
        _trae_proxy="http://127.0.0.1:17891"

        export HTTP_PROXY="$_trae_proxy"
        export HTTPS_PROXY="$_trae_proxy"
        export ALL_PROXY="$_trae_proxy"
        export http_proxy="$_trae_proxy"
        export https_proxy="$_trae_proxy"
        export all_proxy="$_trae_proxy"

        export NO_PROXY="127.0.0.1,localhost,::1,192.168.71.0/24"
        export no_proxy="$NO_PROXY"
        unset _trae_proxy
        ;;
esac
# <<< trae-remote-agent-proxy <<<
```

`case` 条件确保只有来自 A 的 SSH 登录环境继承代理。B 通过 `cuda-via-a` 登录 C 时，不会自动走该代理。

同时设置大小写变量是为了覆盖不同网络库：

| 变量 | 典型使用方 |
|---|---|
| `HTTP_PROXY` / `HTTPS_PROXY` | Rust、Go、部分原生网络库 |
| `http_proxy` / `https_proxy` | `curl`、Python、Unix 工具 |
| `ALL_PROXY` / `all_proxy` | 通用回退 |
| `NO_PROXY` / `no_proxy` | loopback 与局域网绕过 |

Trae Server 只在启动时继承登录环境，因此配置变化后需要重启远端 Server。C helper 会自动判断是否需要重启。

---

## 6. 验证与排障

### 6.1 首选验证

在 A 执行：

```bash
"$HOME/.local/bin/trae_remote/a_trae_proxy.sh" verify
```

正常结果至少包括：

```text
gost_status=running
ssh_tunnel_status=running
remote_listener=127.0.0.1:17891
trae_api_root_http_code=404
profile_status=configured
ai_agent_proxy_status=active
```

### 6.2 分层检查

检查 A 的 HTTP 代理：

```bash
curl -sS \
  --proxy http://127.0.0.1:17900 \
  https://api.ipify.org
```

检查 C 的 listener：

```bash
ssh -o ClearAllForwardings=yes cuda \
  'ss -ltn | grep 127.0.0.1:17891'
```

检查 C 经 A 访问 Trae API：

```bash
ssh -o ClearAllForwardings=yes cuda \
  'curl -sS \
    --proxy http://127.0.0.1:17891 \
    -o /dev/null \
    -w "http_code=%{http_code} total=%{time_total}\n" \
    https://api.trae.com.cn/'
```

检查 `ai-agent` 的实际环境：

```bash
ssh -o ClearAllForwardings=yes cuda '
  pid="$(pgrep -f "/modules/ai-agent/ai-agent$" | head -n 1)"
  tr "\0" "\n" <"/proc/$pid/environ" |
    grep -iE "^(HTTP_PROXY|HTTPS_PROXY|ALL_PROXY|NO_PROXY)="
'
```

### 6.3 常见故障

| 现象 | 原因与处理 |
|---|---|
| `C port 17891 is occupied by an unmanaged tunnel` | A 仍有旧手工 `ssh -R`；按 3.2 只终止对应 PID |
| `17900 is already used by an unmanaged process` | 用 `lsof` 确认占用者；终止它或将正确 PID 写入 `~/.trae-gost.pid` |
| `remote_listener=missing` | SSH ControlMaster 已失效；执行 `restart` |
| `ai_agent_pid=not_running` | Trae Remote 尚未连接 C；连接后重新 `verify` |
| `ai_agent_proxy_status=stale_or_missing` | 执行 `start`，helper 会按需重启 Trae Server |
| `curl` 成功但 Agent 失败 | 检查 Agent 环境；必要时执行 `start` 后重新连接 Remote 窗口 |
| A 睡眠后请求失败 | 直接重跑 `start` |
| A 与 C 查询到的公网 IP 不同 | 企业网络可能使用 NAT 出口池；只要两者均属于 A 的出口且请求成功即可 |

---

## 7. 性能、并发与安全

对于大小为 $B$ 字节的请求，代理链路只做流式转发，处理量为 $O(B)$，除网络栈缓冲区外不需要随请求总量增长的额外内存。

并发特征：

- `gost` 可以并发处理多个 HTTP CONNECT；
- SSH 在一条加密连接上复用多个转发 channel；
- Agent 不需要为每个请求建立新的 SSH 登录；
- 主要延迟仍来自 A 的网络 RTT 和模型服务。

安全边界：

- A 的 `17900` 和 C 的 `17891` 都只绑定 `127.0.0.1`；
- 不使用 `0.0.0.0`，不会向局域网开放代理；
- C 只对来自 A 的登录环境注入代理；
- `NO_PROXY` 保证 loopback 和当前局域网不进入代理；
- SSH 断开后，C 的 `17891` 自动消失；
- A 脚本只停止 PID/control socket 与自身状态匹配的进程。

---

## 附录 A：方案探索记录

本附录记录为什么最终使用 HTTP CONNECT。它不属于日常操作步骤。

### A.1 请求实际从 C 发出

Remote SSH 模式下观察到：

```text
~/.trae-cn-server/.../modules/ai-agent/ai-agent
~/.trae-cn-server/.../modules/ai-agent/bin/agent-tool-host
```

`ai-agent` 初始环境没有代理变量，因此模型请求最初直接使用 C 的网络。

### A.2 远程动态 SOCKS 可以被 `curl` 使用

最初在 A 建立 OpenSSH 远程动态代理：

```bash
ssh -fN \
  -R 127.0.0.1:17890 \
  cuda
```

C 上的以下请求可以成功：

```bash
curl --proxy socks5h://127.0.0.1:17890 \
  https://api.trae.com.cn/
```

这证明 SSH 隧道和 A 的网络出口都正常。

### A.3 Trae/Cronet 不接受该 SOCKS 配置

将 Agent 环境设置为 `socks5h://127.0.0.1:17890` 后，`ai-agent` 确实连接了该端口，但请求返回：

```text
Cronet Error: code=11 / internal_code=-324
```

连接观测显示：

- `curl` 会完成 OpenSSH 能识别的 SOCKS5 握手；
- Agent 连接到达端口后立即失败，没有完成 SOCKS5 握手。

因此问题不是“Agent 没走代理”，而是代理协议不匹配。Trae/Cronet 读取 `HTTP_PROXY` 时需要 HTTP CONNECT 代理。

### A.4 最终改为 A 上的 HTTP 代理

中间方案曾在 C 增加 HTTP 到 SOCKS 的转换层，证明 HTTP CONNECT 可以工作，但会让 C 多维护一个代理进程。

最终链路改为：

```text
C HTTP CONNECT
    -> SSH fixed RemoteForward
    -> A gost HTTP proxy
    -> A network
```

C 只保留环境变量和 SSH listener，不安装代理软件。

### A.5 公网 IP 可能变化

测试中出现过：

```text
139.177.225.237
139.177.225.253
```

这两个地址来自同一出口池。验证时不要求每次返回完全相同的 IP，应关注请求是否成功以及出口网络归属。

---

## 附录 B：从零复现清单

本附录是命令优先的完整恢复步骤。除 Trae Remote 窗口内的最终测试外，所有命令都在 A 执行。

### B.1 检查依赖和 SSH

```bash
command -v bash curl gost lsof scp ssh
gost -V
ssh -o ClearAllForwardings=yes cuda 'hostname'
```

### B.2 安装脚本

```bash
install_dir="$HOME/.local/bin/trae_remote"
mkdir -p "$install_dir"

scp \
  online1:/home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment1-basics/notes/scripts/trae_remote_agent_proxy/a_trae_proxy.sh \
  online1:/home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment1-basics/notes/scripts/trae_remote_agent_proxy/c_configure_trae_proxy.sh \
  "$install_dir/"

chmod 0755 "$install_dir/"*.sh
```

### B.3 清理旧手工隧道

```bash
pgrep -af 'ssh.*17891.*17900.*cuda'
lsof -nP -iTCP@127.0.0.1:17900 -sTCP:LISTEN
```

只处理核对无误的旧专用进程，不使用宽泛的 `pkill ssh`。

### B.4 启动

```bash
"$HOME/.local/bin/trae_remote/a_trae_proxy.sh" start
```

若 Remote 窗口断开，重新连接 `cuda`。

### B.5 验收

```bash
"$HOME/.local/bin/trae_remote/a_trae_proxy.sh" verify
```

然后在 Trae Remote 窗口发送：

```text
只回复 OK
```

### B.6 日后使用

```bash
"$HOME/.local/bin/trae_remote/a_trae_proxy.sh" start
"$HOME/.local/bin/trae_remote/a_trae_proxy.sh" status
"$HOME/.local/bin/trae_remote/a_trae_proxy.sh" verify
"$HOME/.local/bin/trae_remote/a_trae_proxy.sh" stop
```

---

## 附录 C：完全不使用脚本时的最短操作

该方式不提供 PID/control socket 管理和 C 配置幂等性，只适合临时恢复或理解底层命令。长期使用应采用正文中的 A 端控制脚本。

### C.1 C 已有代理配置时

如果 C 的 `~/.profile` 已包含第 5 节的 marker 块，只需在 A 执行：

```bash
nohup gost -L http://127.0.0.1:17900 \
  >/dev/null 2>&1 &
echo $! >"$HOME/.trae-gost.pid"

ssh -fN \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -R 127.0.0.1:17891:127.0.0.1:17900 \
  cuda
```

然后验证：

```bash
ssh -o ClearAllForwardings=yes cuda \
  'curl -sS \
    --proxy http://127.0.0.1:17891 \
    -o /dev/null \
    -w "http_code=%{http_code}\n" \
    https://api.trae.com.cn/'
```

预期：

```text
http_code=404
```

### C.2 C 尚无代理配置时

先在 A 查询 C 所见的来源 IP：

```bash
ssh cuda 'printf "%s\n" "${SSH_CLIENT%% *}"'
```

将第 5 节的 marker 块追加到 C 的 `~/.profile`，并将示例 IP 替换为实际输出。然后在 A 的命令面板执行：

```text
Remote-SSH: Kill Remote Server on Host...
```

选择 `cuda` 并重新连接。

### C.3 手工停止

在 A 查找专用隧道和 `gost`：

```bash
pgrep -af 'ssh.*17891.*17900.*cuda'
lsof -nP -iTCP@127.0.0.1:17900 -sTCP:LISTEN
```

核对后只终止对应 PID。若还要恢复 C 的直接网络环境，需要删除第 5 节的 marker 块并重启 Trae Server。
