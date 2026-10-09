# Trae Remote Agent 通过 Mac A 网络访问模型服务

## 0. 适用场景与结论

网络拓扑：

```text
Mac A（Trae 客户端、可访问模型服务）
    │
    │ Remote SSH
    ▼
Linux C（GPU 机、Trae Server 与 ai-agent）
```

Trae 在 A 上显示窗口，但连接 C 后，远端 Agent 实际运行在 C：

```text
~/.trae-cn-server/.../modules/ai-agent/ai-agent
```

因此，Agent 的模型请求默认从 C 发起，不会自动复用 A 的网络出口。本文给出的稳定方案是：

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

该方案已完成以下实际验证：

- C 上的 Trae `ai-agent` 继承了 HTTP 代理环境变量；
- C 经 `127.0.0.1:17891` 可以访问 `api.trae.com.cn`；
- Trae API 根路径完成 TLS 后返回预期的 HTTP `404`；
- C 经代理查询到的是 A 所在网络的公网出口池；
- A 上 Trae Remote 窗口中的 Agent 请求能够正常完成。

---

## 1. 最终架构

### 1.1 A 上的两个长期进程

A 需要保持两个进程：

| 进程 | 作用 |
|---|---|
| `gost` | 在 A 的 `127.0.0.1:17900` 提供 HTTP CONNECT 代理 |
| `ssh -R` | 在 C 的 `127.0.0.1:17891` 建立入口，并转发到 A 的 `17900` |

SSH 固定反向转发的核心参数是：

```text
-R 127.0.0.1:17891:127.0.0.1:17900
```

它表示：

| 字段 | 位置 | 含义 |
|---|---|---|
| 第一个 `127.0.0.1:17891` | C | 只在 C 的 loopback 上监听 |
| 第二个 `127.0.0.1:17900` | A | 由 A 上的 SSH 客户端连接本地 `gost` |
| `cuda` | SSH 目标 | A 的 SSH config 中指向 C 的别名 |

目标地址 `127.0.0.1:17900` 属于 A，不属于 C。更完整的 `ssh -R` 原理见 [反向 SSH 隧道原理](./90_02_reverse_ssh_tunnel_explained.md)。

### 1.2 C 上的持久配置

C 的 `~/.profile` 中保存 Trae 代理环境：

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

这里通过 `SSH_CLIENT` 限制配置只对来自 A 的连接生效。B 通过 `cuda-via-a` 登录 C 时，不会自动继承这组代理变量。

### 1.3 为什么同时设置大小写变量

不同网络库读取的变量不完全一致：

| 变量 | 常见消费者 |
|---|---|
| `HTTP_PROXY` / `HTTPS_PROXY` | Rust、Go、部分原生网络库 |
| `http_proxy` / `https_proxy` | `curl`、Python 与部分 Unix 工具 |
| `ALL_PROXY` / `all_proxy` | 未区分 HTTP/HTTPS 时的通用回退 |
| `NO_PROXY` / `no_proxy` | 本机与局域网绕过代理 |

Trae Server 会从启动它的登录 shell 继承环境，随后将环境继续传给 `ai-agent`。修改 `~/.profile` 后必须重启远端 Trae Server，仅重新加载窗口不够。

---

## 2. 固定端口与依赖

本文固定使用：

| 项目 | 值 |
|---|---|
| A 的 SSH host alias | `cuda` |
| A 的局域网地址 | `192.168.71.121` |
| C 的局域网地址 | `192.168.71.6` |
| A 的 `gost` HTTP 端口 | `127.0.0.1:17900` |
| C 暴露给 Trae 的 HTTP 端口 | `127.0.0.1:17891` |

A 需要：

```text
bash
curl
gost
lsof
ssh
```

C 需要：

```text
bash
curl
pgrep
ss
```

所有监听都绑定到 `127.0.0.1`，不会向 A 或 C 的局域网暴露开放代理。

---

## 3. 固定脚本

脚本目录：

- [A 端代理与隧道管理脚本](./scripts/trae_remote_agent_proxy/a_trae_proxy.sh)
- [C 端 Trae 代理配置脚本](./scripts/trae_remote_agent_proxy/c_configure_trae_proxy.sh)

### 3.1 A 端脚本

`a_trae_proxy.sh` 支持：

```bash
./a_trae_proxy.sh start
./a_trae_proxy.sh verify
./a_trae_proxy.sh status
./a_trae_proxy.sh stop
./a_trae_proxy.sh restart
```

它负责：

1. 启动或复用 A 上的 `gost`；
2. 通过 SSH ControlMaster 建立专用反向隧道；
3. 检查 C 的 `17891` 是否监听；
4. 从 A 和 C 分别查询公网出口；
5. 检查 `api.trae.com.cn` 的 TLS 与 HTTP 响应；
6. 使用 PID 文件和 SSH control socket 精确停止自身管理的进程。

可覆盖参数：

```bash
CUDA_HOST=cuda
A_HTTP_PORT=17900
C_HTTP_PORT=17891
```

例如：

```bash
CUDA_HOST=cuda A_HTTP_PORT=17900 C_HTTP_PORT=17891 \
  ./a_trae_proxy.sh start
```

### 3.2 C 端脚本

`c_configure_trae_proxy.sh` 支持：

```bash
./c_configure_trae_proxy.sh apply
./c_configure_trae_proxy.sh apply-and-restart
./c_configure_trae_proxy.sh verify
./c_configure_trae_proxy.sh status
./c_configure_trae_proxy.sh restart-trae
./c_configure_trae_proxy.sh remove
./c_configure_trae_proxy.sh remove-and-restart
```

它负责：

1. 在修改配置前验证 `C:17891` 和 Trae API；
2. 检查 `.profile` marker 是否成对且没有重复；
3. 删除旧 marker 块后写入新配置，因此可幂等执行；
4. 每次修改前创建带时间戳的 `.profile` 备份；
5. 检查 shell 语法；
6. 精确终止 Trae Server 进程组；
7. 验证运行中的 `ai-agent` 是否继承正确代理。

可覆盖参数：

```bash
A_SOURCE_IP=192.168.71.121
C_HTTP_PORT=17891
PROFILE_PATH="$HOME/.profile"
NO_PROXY_VALUE="127.0.0.1,localhost,::1,192.168.71.0/24"
```

`restart-trae` 应从独立 SSH session 执行。脚本会拒绝在与 Trae Server 相同的进程组中终止自身。

---

## 4. 首次部署

### 4.1 将 A 脚本安装到 A

若 A 能通过别名 `online1` 访问 B，可在 A 执行：

```bash
mkdir -p "$HOME/.local/bin"

scp \
  online1:/home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment1-basics/notes/scripts/trae_remote_agent_proxy/a_trae_proxy.sh \
  "$HOME/.local/bin/a_trae_proxy.sh"

chmod 0755 "$HOME/.local/bin/a_trae_proxy.sh"
```

确认 `gost` 已安装：

```bash
command -v gost
gost -V
```

### 4.2 将 C 脚本安装到 C

在 B 执行：

```bash
scp \
  /home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment1-basics/notes/scripts/trae_remote_agent_proxy/c_configure_trae_proxy.sh \
  cuda-via-a:/home/dengxiao/.local/bin/c_configure_trae_proxy.sh

ssh cuda-via-a \
  'chmod 0755 "$HOME/.local/bin/c_configure_trae_proxy.sh"'
```

### 4.3 启动 A 的代理与隧道

在 A 执行：

```bash
"$HOME/.local/bin/a_trae_proxy.sh" start
```

验收输出应包括：

```text
gost_status=started
ssh_tunnel_status=started
a_proxy_egress_ip=...
c_proxy_egress_ip=...
trae_api_root_http_code=404
```

A 与 C 的公网 IP 不要求逐次完全相同。企业网络可能按连接从同一 NAT 地址池选择不同出口。

### 4.4 配置 C 并重启 Trae Server

在 A 的普通终端执行：

```bash
ssh cuda \
  'A_SOURCE_IP=192.168.71.121 \
   C_HTTP_PORT=17891 \
   "$HOME/.local/bin/c_configure_trae_proxy.sh" apply-and-restart'
```

该命令会中断当前连接 C 的 Trae Remote 窗口。重新连接 `cuda` 后，在 A 执行：

```bash
ssh cuda \
  '"$HOME/.local/bin/c_configure_trae_proxy.sh" verify'
```

验收输出应包括：

```text
proxy_listener=127.0.0.1:17891
profile_status=configured
ai_agent_proxy_status=active
trae_api_root_http_code=404
```

最后在 Trae Remote 窗口发送一个最小 Agent 请求：

```text
只回复 OK
```

收到正常回答才算完成端到端验收。

---

## 5. 日常使用

### 5.1 启动顺序

每次 A 重启、睡眠导致 SSH 连接断开，或手动停止代理后，在 A 执行：

```bash
"$HOME/.local/bin/a_trae_proxy.sh" start
```

然后再用 Trae Remote 连接 `cuda`。

C 的 `.profile` 是持久配置，日常不需要重复执行 `apply`。

### 5.2 状态检查

在 A 执行：

```bash
"$HOME/.local/bin/a_trae_proxy.sh" status
"$HOME/.local/bin/a_trae_proxy.sh" verify
```

在 C 或通过 A 执行：

```bash
ssh cuda \
  '"$HOME/.local/bin/c_configure_trae_proxy.sh" status'
```

### 5.3 停止

先关闭连接 C 的 Trae Remote 窗口，再在 A 执行：

```bash
"$HOME/.local/bin/a_trae_proxy.sh" stop
```

这会停止脚本管理的 SSH 隧道和 `gost`，但保留 C 的 `.profile` 配置。

若需要同时移除 C 的配置，在 A 执行：

```bash
ssh cuda \
  '"$HOME/.local/bin/c_configure_trae_proxy.sh" remove-and-restart'
```

---

## 6. 验证与排障

### 6.1 A 的 HTTP 代理

```bash
curl -sS \
  --proxy http://127.0.0.1:17900 \
  https://api.ipify.org
```

### 6.2 C 的反向转发入口

```bash
ssh -o ClearAllForwardings=yes cuda \
  'ss -ltn | grep 127.0.0.1:17891'
```

### 6.3 C 经 A 访问 Trae API

```bash
ssh -o ClearAllForwardings=yes cuda \
  'curl -sS \
     --proxy http://127.0.0.1:17891 \
     -o /dev/null \
     -w "http_code=%{http_code} total=%{time_total}\n" \
     https://api.trae.com.cn/'
```

预期：

```text
http_code=404
```

这里的 `404` 只表示 API 根路径没有业务路由；TLS 和 HTTP 已成功完成。

### 6.4 `ai-agent` 的实际环境

在 C 执行：

```bash
pid="$(
  pgrep -f '/modules/ai-agent/ai-agent$' |
    head -n 1
)"

tr '\0' '\n' <"/proc/$pid/environ" |
  grep -iE '^(HTTP_PROXY|HTTPS_PROXY|ALL_PROXY|NO_PROXY)='
```

预期：

```text
HTTP_PROXY=http://127.0.0.1:17891
HTTPS_PROXY=http://127.0.0.1:17891
ALL_PROXY=http://127.0.0.1:17891
```

### 6.5 常见故障

| 现象 | 检查重点 |
|---|---|
| A 脚本报告 `gost_status=stopped` | `gost` 是否安装，`17900` 是否被其它进程占用 |
| SSH 报 remote forward failure | C 的 `17891` 是否已有旧隧道或残留进程 |
| C 上没有 `17891` listener | A 的 SSH master 是否存活 |
| `curl` 经代理成功，但 Agent 失败 | 是否已完全重启 Trae Server |
| `ai_agent_proxy_status=stale_or_missing` | 旧 `ai-agent` 是否仍在运行 |
| A 的 IP 变化后不再注入代理 | 使用新的 `A_SOURCE_IP` 重跑 C 脚本 |
| A 睡眠后 Agent 请求失败 | 重启 A 脚本管理的隧道 |

---

## 7. 性能、并发与安全边界

对大小为 $B$ 字节的一次请求，代理链路各层只做流式转发，数据处理量为 $O(B)$；不会产生与请求历史长度平方相关的额外工作。

并发特征：

- `gost` 可以同时处理多个 HTTP CONNECT；
- SSH 在同一条加密连接上复用多个转发 channel；
- Agent 的并发请求不需要为每个请求重新创建 SSH 登录；
- A 的上行带宽、网络 RTT 和模型服务本身仍是主要延迟来源。

额外路径包括两个 loopback TCP hop 和一条 A↔C SSH 加密通道。相较模型推理时间，这部分通常只增加小量固定延迟。

安全边界：

- A 和 C 都只监听 `127.0.0.1`；
- 不使用 `0.0.0.0`，不向局域网开放代理；
- C 只对来自 A 的 Trae 启动环境注入代理；
- `NO_PROXY` 保证本机 IPC、loopback 与当前局域网不进入代理；
- SSH 断开后 C 的 `17891` 自动消失，不留下独立公网代理。

---

## 附录 A：探索与排障记录

本附录只记录形成最终方案的过程，不属于日常操作步骤。

### A.1 确认请求实际从 C 发出

C 上观察到以下远端进程：

```text
~/.trae-cn-server/.../modules/ai-agent/ai-agent
~/.trae-cn-server/.../modules/ai-agent/bin/agent-tool-host
```

`ai-agent` 初始环境中没有 `HTTP_PROXY`、`HTTPS_PROXY` 或 `ALL_PROXY`，因此最初的请求直接使用 C 的网络。

### A.2 第一版：SSH 远程动态 SOCKS

第一版在 A 使用：

```bash
ssh -fN \
  -R 127.0.0.1:17890 \
  cuda
```

省略 `-R` 的目标地址时，OpenSSH 在 C 提供动态 SOCKS 服务，出口位于 A。使用 `curl --proxy socks5h://127.0.0.1:17890` 可以成功访问公网和 `api.trae.com.cn`。

随后将 Trae 环境设为：

```text
HTTP_PROXY=socks5h://127.0.0.1:17890
HTTPS_PROXY=socks5h://127.0.0.1:17890
ALL_PROXY=socks5h://127.0.0.1:17890
```

`ai-agent` 确实连接了 `17890`，但 Agent 请求返回：

```text
Request service failed ... (996)
Cronet Error: code=11 / internal_code=-324
```

### A.3 用连接和 SSH 日志区分“未走代理”与“协议错误”

在 Agent 请求期间，C 明确出现：

```text
ai-agent -> 127.0.0.1:17890
```

因此“Agent 完全没有走代理”被排除。

OpenSSH debug 日志中，Agent 连接只出现：

```text
read failed
write failed
```

同一端口上的 `curl SOCKS5` 对照请求则出现：

```text
decode socks5
socks5 auth done
dynamic request: socks5 host api.trae.com.cn port 443 command 1
```

这证明：

1. SSH SOCKS 隧道本身正常；
2. Agent 连接到达了 SOCKS 端口；
3. Agent/Cronet 没有发送 OpenSSH 能识别的 SOCKS5 握手；
4. `HTTP_PROXY` 的消费方实际需要 HTTP CONNECT 代理。

### A.4 中间验证：C 上 HTTP 到 SOCKS 的转换层

曾在 C 临时启动 HTTP → SOCKS 转换：

```text
C HTTP proxy -> C SSH SOCKS -> A network
```

该方式证明增加协议转换后 HTTP CONNECT 可以工作，但会在 C 引入额外常驻工具和服务。最终方案将 HTTP 代理直接放在 A：

```text
C HTTP CONNECT -> SSH fixed RemoteForward -> A gost
```

这样 C 只保留环境变量，不需要安装或维护代理软件。

### A.5 公网 IP 不完全相同

测试中 A 与 C 经代理访问 `api.ipify.org` 时，出现过：

```text
139.177.225.237
139.177.225.253
```

多次请求会在两个地址间变化，说明 A 所在网络使用多出口 NAT 池。验证时应比较网络归属和可达性，不应要求每次返回完全相同的单个 IP。

---

## 附录 B：从零复现操作清单

本附录用于 C 重装、A 更换或端口配置丢失后的完整恢复。

### B.1 在 A 确认 SSH 与 `gost`

```bash
ssh cuda 'hostname'
command -v gost
gost -V
```

若 `cuda` 尚未配置，在 A 的 `~/.ssh/config` 中加入：

```sshconfig
Host cuda
    HostName 192.168.71.6
    User dengxiao
```

### B.2 在 A 安装固定脚本

```bash
mkdir -p "$HOME/.local/bin"

scp \
  online1:/home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment1-basics/notes/scripts/trae_remote_agent_proxy/a_trae_proxy.sh \
  "$HOME/.local/bin/a_trae_proxy.sh"

chmod 0755 "$HOME/.local/bin/a_trae_proxy.sh"
```

### B.3 在 B 将 C 脚本部署到 C

```bash
scp \
  /home/dengxiao.cs/code_repos/institutionalized/stanford_cs336/assignments/assignment1-basics/notes/scripts/trae_remote_agent_proxy/c_configure_trae_proxy.sh \
  cuda-via-a:/home/dengxiao/.local/bin/c_configure_trae_proxy.sh

ssh cuda-via-a \
  'chmod 0755 "$HOME/.local/bin/c_configure_trae_proxy.sh"'
```

### B.4 在 A 启动代理与隧道

若此前曾手工执行 `ssh -fN -R ...` 建立 `17891`，先在 A 找到旧隧道：

```bash
pgrep -af 'ssh.*17891.*17900.*cuda'
```

核对命令行无误后，只终止对应 PID：

```bash
kill <确认后的旧隧道 PID>
```

不要用宽泛的 `pkill ssh`，否则会终止 A 上其它 SSH 会话。

若 A 上的 `gost` 由本文早期命令启动，且 `~/.trae-gost.pid` 仍指向该进程，脚本会直接复用它。

随后执行：

```bash
"$HOME/.local/bin/a_trae_proxy.sh" start
```

不要在 `start` 成功前打开连接 C 的 Trae Remote 窗口。

### B.5 确认 A 的来源 IP

在 A 执行：

```bash
ssh cuda 'printf "SSH_CLIENT=%s\n" "$SSH_CLIENT"'
```

取输出中的第一个地址。例如：

```text
SSH_CLIENT=192.168.71.121 52311 22
```

则：

```text
A_SOURCE_IP=192.168.71.121
```

### B.6 在 A 触发 C 的一次性配置

```bash
ssh cuda \
  'A_SOURCE_IP=192.168.71.121 \
   C_HTTP_PORT=17891 \
   "$HOME/.local/bin/c_configure_trae_proxy.sh" apply'
```

若 C 上已有 Trae Server，从 A 的独立终端执行：

```bash
ssh cuda \
  '"$HOME/.local/bin/c_configure_trae_proxy.sh" restart-trae'
```

### B.7 重新连接 Trae 并验收

在 A 用 Trae Remote 重新连接 `cuda`，然后执行：

```bash
ssh cuda \
  '"$HOME/.local/bin/c_configure_trae_proxy.sh" verify'
```

最后发送：

```text
只回复 OK
```

### B.8 以后每天需要做什么

A 未重启且隧道仍在时：

```bash
"$HOME/.local/bin/a_trae_proxy.sh" status
```

A 重启、睡眠断线或代理已停止时：

```bash
"$HOME/.local/bin/a_trae_proxy.sh" start
```

C 正常保留 `.profile` 时：

```text
不需要执行任何配置命令
```

C 重装或 `.profile` 配置丢失时：

```bash
"$HOME/.local/bin/c_configure_trae_proxy.sh" apply
```

### B.9 完整停止和清理

在 A 停止代理与隧道：

```bash
"$HOME/.local/bin/a_trae_proxy.sh" stop
```

在 C 删除 Trae 代理配置：

```bash
"$HOME/.local/bin/c_configure_trae_proxy.sh" remove
```

脚本创建的 `.profile.backup-before-trae-proxy-*` 备份不会自动删除，确认长期运行稳定后再人工清理旧备份。
