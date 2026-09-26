# Assignment 2 Profile 远程 GPU 运行方案

## 0. 范围与状态

本文只规划在 `cuda-via-a` 上学习和运行 Assignment 2 的 benchmark 与 Nsight Systems profile，不部署训练数据、checkpoint、W&B、分布式训练或完整 Assignment 2 环境。

当前已经完成最小源码同步、cu126 环境安装、CUDA kernel 验证、small GPU benchmark，以及固定脚本形式的 Nsight Systems profile。所有 C 上的 shell 操作均在现有 tmux session `cs336` 中执行；唯一例外是 B 发起的 rsync 传输进程，因为它必须通过 SSH 把 B 的文件发送到 C。

相关背景：

- [反向 SSH 隧道原理](./90_02_reverse_ssh_tunnel_explained.md)
- [rsync 增量部署与 SSH 远程执行](./90_03_rsync_remote_gpu_execution_guide.md)

---

## 1. 当前资源结论

### 1.1 C 的实际状态

2026-09-25 在 `cs336` tmux session 中只读检查得到：

| 项目 | 当前值 |
|---|---|
| GPU | NVIDIA GeForce GTX 1060 |
| 显存 | 6144 MiB |
| 驱动 | 570.211.01 |
| 根分区 | 116 GiB |
| 已使用 | 102 GiB |
| 可用 | 8.6 GiB |
| Conda base | 28 GiB，Python 3.13.9，无 PyTorch |
| uv 缓存 | 已清理；执行前为 4.7 GiB |
| Profile 环境 | 5.9 GiB，PyTorch 2.11.0+cu126 |
| Nsight Systems | `/usr/local/cuda-12.8/bin/nsys`，2024.6.2 |

### 1.2 真正的磁盘风险

本地测得：

| 内容 | 大小 |
|---|---:|
| Assignment 1 完整目录 | 20 GiB |
| Assignment 1 `.venv` | 4.7 GiB |
| Assignment 1 数据 | 15 GiB |
| Assignment 2 完整目录 | 618 MiB |
| Assignment 2 `.venv` | 4.9 GiB |
| Profile 所需白名单文件 | 66,114 bytes，约 64.6 KiB |
| 现有 CPU benchmark JSON | 约 148 KiB |

结论：

1. 源码同步不是磁盘瓶颈，白名单源码不足 1 MiB。
2. 不能同步 `.venv`、数据、checkpoint 或已有 benchmark 产物。
3. 真正需要预算的是 PyTorch/CUDA 依赖，约 4-6 GiB，以及 `.nsys-rep`。
4. C 原有 4.7 GiB uv 缓存是 CUDA 13.0 构建，对 GTX 1060 无用，现已清理。
5. 28 GiB Conda base 暂不删除，因为可能包含其它用途的软件。

---

## 2. GPU 兼容性边界

GTX 1060 属于 Pascal，compute capability 为 6.1。当前项目锁定的 PyTorch 2.11 CUDA 13.0 构建不能在该 GPU 上运行，因为 CUDA 13 已移除 Pascal 支持。

Profile-only 环境应改用：

```text
PyTorch 2.11.0 + CUDA 12.6
```

PyTorch 2.11 的 CUDA 12.6 构建仍包含 Pascal 支持，参考 [PyTorch 2.11 CUDA 架构支持说明](https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290)。

限制：

- 本方案不执行 Triton kernel、`torch.compile` CUDA 后端或 BF16 实验；
- GTX 1060 没有 Tensor Core，不能代表课程使用的现代 GPU 性能；
- `small + full + B=4 + S=512` 虽能运行，但实测 reserved 显存为 5.258 GiB，余量过小；
- 本方案用于学习 Nsight Systems、CUDA timeline、kernel 分布和 CPU/GPU 异步关系；
- 缩小模型得到的数字不能写成 handout 的正式 benchmark 结果。

---

## 3. 精简后的远端目录

C 只需要：

```text
/home/dengxiao/
├── work/cs336-profile/assignments/
│   ├── assignment1-basics/
│   └── assignment2-systems/
├── .venvs/cs336-profile-cu126/
└── var/cs336-profile/
    ├── profiles/
    ├── results/
    ├── logs/
    └── tmp/
```

不创建：

- `data/`；
- `checkpoints/`；
- `wandb/`；
- 每次运行的完整源码快照；
- 第二套完整 Assignment 2 开发环境。

短 profile 运行期间不执行 rsync，因此没有必要为每次运行复制源码。

---

## 4. 反向 SSH 与 tmux

反向隧道继续沿用：

```text
B:127.0.0.1:2222 -> A -> C:192.168.71.6:22
```

B 已配置：

```sshconfig
Host cuda-via-a
    HostName localhost
    Port 2222
    User dengxiao
```

所有 C 上的安装、检查和运行命令都在现有 session 中执行：

```bash
ssh -t cuda-via-a 'tmux attach -t cs336'
```

推荐在该 session 内新建窗口：

```text
profile-env     环境安装和检查
profile-run     运行 benchmark/nsys
profile-monitor nvidia-smi 和日志
```

在 `profile-env` 窗口创建最小目录：

```bash
mkdir -p "$HOME/work/cs336-profile/assignments/assignment1-basics"
mkdir -p "$HOME/work/cs336-profile/assignments/assignment2-systems"
mkdir -p "$HOME/var/cs336-profile/profiles"
mkdir -p "$HOME/var/cs336-profile/results"
mkdir -p "$HOME/var/cs336-profile/logs"
mkdir -p "$HOME/var/cs336-profile/tmp"
```

仅监控时：

```bash
ssh -t cuda-via-a 'tmux attach -r -t cs336'
```

---

## 5. 最小源码同步

已实现独立同步工具：

- [sync_to_cuda.sh](../../assignment2-systems/remote_sync/sync_to_cuda.sh)
- [使用说明](../../assignment2-systems/remote_sync/README.md)

实际操作应优先调用该脚本。下面保留展开后的 rsync 命令，用于审核其同步边界。

### 5.1 必要文件

Assignment 1：

```text
assignment1-basics/
└── cs336_basics/
```

Assignment 2：

```text
assignment2-systems/
├── cs336_systems/
├── benchmark.py
└── remote_profile/
    ├── run_small_benchmark.sh
    └── run_small_profile.sh
```

其中：

- `cs336_basics/` 提供 Transformer、loss 和 AdamW；
- `cs336_systems/` 与 `benchmark.py` 提供 profile 入口；
- `remote_profile/run_small_benchmark.sh` 只保存固定 benchmark 命令；
- `remote_profile/run_small_profile.sh` 只保存固定 Nsight Systems 采集命令；
- 两个项目都通过 `PYTHONPATH` 直接导入，不需要远端安装为 Python package；
- `pyproject.toml`、`uv.lock` 和 `README.md` 均不参与本次远端运行；
- 不同步 Assignment 2 内部重复的 `cs336-basics/`。

### 5.2 默认 dry-run

在 B 的 Assignment 2 根目录执行：

```bash
./remote_sync/sync_to_cuda.sh
```

默认只展示变化，不修改 C。第一次运行时，如果远端目标目录尚不存在，脚本会基于本地空目录展示完整白名单，并提示先在 `cs336` tmux session 中创建目标目录。

### 5.3 正式同步

查看预览、人工确认后同步：

```bash
./remote_sync/sync_to_cuda.sh --apply
```

非交互调用必须显式添加：

```bash
./remote_sync/sync_to_cuda.sh --apply --yes
```

脚本具有以下保护：

- 目标根目录必须以 `/work/cs336-profile/assignments` 结尾；
- 同步前检查本地源文件、SSH 和远端 rsync；
- 本地文件锁阻止两个同步进程并发执行；
- 只允许两个 Python package 和 `benchmark.py`；
- 使用内容校验，不依赖两台机器的时间戳；
- 使用延迟更新和延迟删除；
- `--delete-excluded` 只作用于专用代码目录；
- 正式同步后再次 dry-run，无变化才报告幂等检查通过。

本地验证的白名单总量为 66,114 bytes，约 64.6 KiB。rsync 是唯一不位于 tmux 中的 C 端进程，因为传输必须由 B 主动发起；所有环境和实验命令仍在 `cs336` session 中执行。

---

## 6. Profile-only 环境

### 6.1 为什么不执行 `uv sync --locked`

当前 `uv.lock` 指向 PyTorch 2.11 CUDA 13.0，并会安装 pytest、Ruff、ty、W&B、pandas、matplotlib 等 profile 不需要的包。直接执行会：

1. 安装 GTX 1060 无法运行的 CUDA 13.0 PyTorch；
2. 浪费额外磁盘空间；
3. 无法满足本次仅运行 profile 的目标。

因此本次创建独立的最小环境，并通过 `PYTHONPATH` 直接加载同步后的源码。

### 6.2 安装前磁盘处理

C 原有 uv 缓存中的主要内容是 `torch==2.11.0+cu130` 和 `triton==3.6.0`。它们不能用于 GTX 1060，现已清理。

在用户确认缓存不再用于其它任务后，可在 `cs336` session 的 `profile-env` 窗口执行：

```bash
uv cache clean
df -h "$HOME"
```

预计可释放约 4.7 GiB。该命令只清除 uv 下载/构建缓存，不删除 Conda base 或已有虚拟环境，但以后需要相同包时必须重新下载。

### 6.3 创建 CUDA 12.6 环境

仍在 `profile-env` 窗口执行：

```bash
set -euo pipefail

PROFILE_ENV="$HOME/.venvs/cs336-profile-cu126"

mkdir -p "$HOME/.venvs"
mkdir -p "$HOME/var/cs336-profile/profiles"
mkdir -p "$HOME/var/cs336-profile/results"
mkdir -p "$HOME/var/cs336-profile/logs"
mkdir -p "$HOME/var/cs336-profile/tmp"

uv venv --python 3.13 "$PROFILE_ENV"

UV_NO_CACHE=1 uv pip install \
  --python "$PROFILE_ENV/bin/python" \
  --index-url https://download.pytorch.org/whl/cu126 \
  "torch==2.11.0"
```

`UV_NO_CACHE=1` 避免安装完成后额外保留一份下载缓存。环境本身预计占约 4-6 GiB，安装前后都应使用 `df -h "$HOME"` 检查实际剩余空间。

保留至少 5 GiB 空闲空间作为 profile、临时文件和系统运行余量。若安装后不足 5 GiB，停止实验并先处理磁盘，不继续生成 trace。

### 6.4 验证 Pascal 支持

```bash
PROFILE_ENV="$HOME/.venvs/cs336-profile-cu126"

"$PROFILE_ENV/bin/python" -c '
import torch
print(f"torch_version={torch.__version__}")
print(f"torch_cuda_version={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_arch_list={torch.cuda.get_arch_list()}")
print(f"gpu_name={torch.cuda.get_device_name(0)}")
print(f"compute_capability={torch.cuda.get_device_capability(0)}")
result = (torch.ones(1, device="cuda") + 1).item()
print(f"cuda_smoke_result={result}")
'
```

验收条件：

- `torch_cuda_version=12.6`；
- `cuda_available=True`；
- `cuda_arch_list` 包含 Pascal 可用代码，例如 `sm_60` 或 `sm_61`；
- `compute_capability=(6, 1)`。
- `cuda_smoke_result=2.0`，证明实际 CUDA kernel 可以启动。

---

## 7. 运行环境变量

在 `cs336` session 的 `profile-run` 窗口设置：

```bash
export PROFILE_ENV="$HOME/.venvs/cs336-profile-cu126"
export PROFILE_ROOT="$HOME/var/cs336-profile"
export ASSIGNMENTS="$HOME/work/cs336-profile/assignments"
export PYTHONPATH="$ASSIGNMENTS/assignment1-basics:$ASSIGNMENTS/assignment2-systems"
export TMPDIR="$PROFILE_ROOT/tmp"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0

cd "$ASSIGNMENTS/assignment2-systems"
```

确认导入位置：

```bash
"$PROFILE_ENV/bin/python" -c '
import cs336_basics
import cs336_systems
print(f"cs336_basics_path={cs336_basics.__file__}")
print(f"cs336_systems_path={cs336_systems.__file__}")
'
```

两个路径都必须来自 `~/work/cs336-profile/assignments/`。

---

## 8. 先做最小 CUDA 冒烟

固定命令保存在 [run_small_benchmark.sh](../../assignment2-systems/remote_profile/run_small_benchmark.sh)。在 C 的 Assignment 2 目录直接执行：

```bash
./remote_profile/run_small_benchmark.sh
```

该命令使用 `full`、FP32、课程完整 `small` preset、`B=2`、`S=512`、5 次 warmup 和 10 次 measurement。

验收：

1. 命令正常退出；
2. JSON 中存在 `forward`、`loss`、`backward`、`optimizer` 和 `total`；
3. `nvidia-smi` 能观察到进程；
4. 没有 CUDA 架构不兼容错误。

2026-09-25 使用上述直接命令实际运行成功，结果如下：

| 阶段 | 均值 ± 总体标准差 |
|---|---:|
| forward | 136.559 ± 0.271 ms |
| loss | 1.721 ± 0.021 ms |
| backward | 286.656 ± 1.080 ms |
| optimizer | 67.655 ± 0.060 ms |
| total | 492.616 ± 1.084 ms |

最终配置使用完整的 128,625,408 参数 `small` 模型、`B=2` 和 `S=512`，同配置探测到的峰值 allocated/reserved CUDA 显存分别为 3.255/3.453 GiB。结果保存在 C 的 `~/var/cs336-profile/results/small_b2_s512_full.json`。PyTorch 因最小环境没有安装 NumPy 而输出一次非致命警告，不影响本次纯 PyTorch benchmark。

---

## 9. Profile 代码准备

当前脚本只有外层 `benchmark_measurement` NVTX range。它可以排除 warmup，但不方便直接区分 forward、loss、backward 和 optimizer。

开始正式 profile 前，建议在 B 上为四个阶段增加 NVTX range：

```text
benchmark_measurement
├── forward
├── loss
├── backward
└── optimizer
```

后续分析 self-attention 时，再为以下子阶段增加 range：

```text
scaled_dot_product_attention
├── attention_scores
├── softmax
└── attention_value_matmul
```

这些修改在 B 完成和测试，然后通过第 5 节的白名单 rsync 更新到 C。不要直接在 C 修改。

---

## 10. Nsight Systems 采集

### 10.1 已完成的基础 profile

按照 handout 的基础调用形式，在 `cs336:profile-run` 中执行了：

```bash
PYTHONPATH="$HOME/work/cs336-profile/assignments/assignment1-basics:$HOME/work/cs336-profile/assignments/assignment2-systems" \
CUDA_VISIBLE_DEVICES=0 \
nsys profile \
  --force-overwrite=true \
  --output="$HOME/var/cs336-profile/profiles/basic/small_b2_s512_full" \
  -- "$HOME/.venvs/cs336-profile-cu126/bin/python" benchmark.py \
    --model-size small --mode full --device cuda --dtype float32 \
    --batch-size 2 --context-length 512 \
    --warmup-steps 5 --measurement-steps 10 --seed 0 \
    --output-json "$HOME/var/cs336-profile/results/small_b2_s512_nsys.json"
```

执行成功，profile 下的完整 step 为 `494.815 ± 0.912 ms`。这次基础验证产生的旧文件随后已清理，由第 10.3 节的固定脚本产物取代。

```text
~/var/cs336-profile/profiles/basic/small_b2_s512_full.nsys-rep
~/var/cs336-profile/profiles/basic/small_b2_s512_full.sqlite
```

`.nsys-rep` 为 3.2 MiB，运行 `nsys stats` 后整个目录为 22 MiB。CUDA kernel 汇总中累计耗时最高的三个 kernel 均为 SGEMM：

| Kernel | GPU 时间占比 |
|---|---:|
| `sgemm_128x128x8_NN_vec` | 16.4% |
| `sgemm_128x128x8_NT_vec` | 15.8% |
| `sgemm_128x128x8_TN_vec` | 15.0% |

基础命令会采集模型初始化、warmup 和全部 10 个 measurement step。它适合确认工具链可用，但不适合作为最终的精简 trace。

### 10.2 为什么精简 profile 只测量一个 step

普通 benchmark 使用 10 个样本计算均值和标准差；Nsight Systems 的目的则是观察一次执行的 timeline 和 kernel 分布。捕获 10 个相同步骤只会扩大 `.nsys-rep`，不增加结构信息。

因此 profile 使用：

```text
warmup_steps=5
measurement_steps=1
```

### 10.3 固定采集脚本

固定命令保存在 [run_small_profile.sh](../../assignment2-systems/remote_profile/run_small_profile.sh)。在 `profile-run` 窗口执行：

```bash
./remote_profile/run_small_profile.sh
```

该脚本保持 handout 的基本形式：

- 每次只保留一个 measurement step；
- 固定写入 `~/var/cs336-profile/profiles/small_b2_s512/`；
- 运行前删除同名 `.nsys-rep` 和 `.sqlite`，避免旧报告干扰；
- 使用 `--force-overwrite=true`，避免重复生成带后缀的文件。

由于不使用 NVTX capture-range，该版本会同时记录初始化、5 个 warmup step 和 1 个 measurement step。它优先保证在当前 Nsight Systems 2024.6.2 上稳定产出报告；后续完成阶段级 NVTX 标注后，再单独处理精确截取。

2026-09-25 实际运行和 `nsys stats` 解析均成功：

| 项目 | 结果 |
|---|---:|
| profile 下的完整 step | 495.851 ms |
| `.nsys-rep` | 1.8 MiB |
| `.nsys-rep`、SQLite 和 JSON 合计 | 13 MiB |
| SGEMM 三个主要 kernel 的累计占比 | 16.2%、15.9%、14.9% |

当前产物位于：

```text
~/var/cs336-profile/profiles/small_b2_s512/profile.nsys-rep
~/var/cs336-profile/profiles/small_b2_s512/profile.sqlite
~/var/cs336-profile/profiles/small_b2_s512/result.json
```

### 10.4 查看结果

先在 C 上查看摘要：

```bash
PROFILE_DIR="$HOME/var/cs336-profile/profiles/small_b2_s512"
nsys stats "$PROFILE_DIR/profile.nsys-rep"
du -sh "$PROFILE_DIR"
df -h "$HOME"
```

只有需要 GUI 时，才把单个最终 `.nsys-rep` 显式复制到源码树之外。不要建立 C 到 B 的目录级反向同步。

---

## 11. 显存范围

推荐依次尝试：

| 层级 | 配置 | 目的 |
|---|---|---|
| 1 | `small`，`B=1, S=128` | 验证完整 profile |
| 2 | `small`，`B=1, S=256` | 观察序列长度影响 |
| 3 | `small`，`B=2, S=512` | 日常 benchmark/profile 的目标配置 |

每次升级前执行：

```bash
nvidia-smi
```

发生 OOM 后停止扩大配置并记录边界。`small + full + B=4 + S=512` 已验证可运行，但 5.258 GiB reserved 显存对 6 GiB GPU 来说过于接近上限，不作为日常固定配置。

---

## 12. 磁盘预算与清理策略

### 12.1 预计新增占用

| 项目 | 预计占用 |
|---|---:|
| 白名单源码 | 小于 1 MiB |
| PyTorch 2.11 CUDA 12.6 最小环境 | 约 4-6 GiB |
| 单个精简 `.nsys-rep` | 通常几十到数百 MiB |
| JSON 与文本日志 | 通常小于 10 MiB |

在清理 4.7 GiB 无用 cu130 缓存后，预计磁盘可用空间约从 15 GiB 增至接近 20 GiB；安装 profile 环境后预计仍保留约 14-16 GiB，实际值以 `df` 为准。

### 12.2 保留策略

1. 同一配置只保留最后一个有效 `.nsys-rep`；
2. profile 失败时先检查文件，再删除明显不完整的运行目录；
3. 不自动导出 SQLite，只有需要 SQL 分析时才生成；
4. 每轮 profile 后执行 `du -sh "$PROFILE_ROOT/profiles"/*`；
5. 始终保留至少 5 GiB 根分区空间；
6. 不自动删除 28 GiB Conda base；
7. 不把 profile 文件放入 rsync 代码目录。

### 12.3 将单个报告复制到 B

Assignment 2 根目录下的 `profile_artifacts/` 已加入 `.gitignore`，用于承接显式下载的报告，不参与源码同步或 Git 追踪。

在 B 的 Assignment 2 根目录执行：

```bash
mkdir -p profile_artifacts
scp \
  cuda-via-a:~/var/cs336-profile/profiles/small_b2_s512/profile.nsys-rep \
  profile_artifacts/small_b2_s512.nsys-rep
```

2026-09-25 已完成复制：

```text
size=1.8 MiB
sha256=b5d679fb97c7b6fd24e2193e42e9a5616d52b4f26f82f7712299464d90b388d4
```

---

## 13. 每次迭代流程

```text
1. B 修改 profile 或 NVTX 代码
2. B 完成本地测试
3. B 运行两条白名单 rsync dry-run
4. B 审核后正式同步
5. C 的 cs336 tmux session 中执行 CUDA 冒烟
6. 创建唯一 RUN_ID
7. 使用 5 次 warmup、1 次 measurement 运行 nsys
8. 在 C 上运行 nsys stats
9. 记录结论并检查磁盘
10. 只保留有分析价值的 profile
```

同步期间不要运行 profile；profile 期间不要同步源码。

---

## 14. 实施检查清单

实施前：

- [x] `cuda-via-a` 可通过反向 SSH 隧道访问；
- [x] C 已有 tmux session `cs336`；
- [x] C 已安装 Nsight Systems 2024.6.2；
- [x] 已确认 C 是 GTX 1060 6 GiB；
- [x] 已确认 C 只有 15 GiB 可用磁盘；
- [x] 已确认旧 uv 缓存为不兼容的 PyTorch cu130；
- [x] 用户批准并完成 C 的 uv 缓存清理；
- [x] 已创建 5.9 GiB 的 cu126 环境；
- [x] 已完成 rsync dry-run、正式同步和幂等检查；
- [x] 已将固定 benchmark 命令保存为 shell 脚本；
- [x] 已将精简 Nsight Systems 命令保存为 shell 脚本；
- [x] 直接 Python 命令已在 C 上运行通过；
- [ ] B 上补充阶段级 NVTX range；
- [x] CUDA kernel 和微型完整训练 benchmark 通过；
- [x] 基础 `nsys profile -- python benchmark.py` 运行和统计解析通过；
- [x] 固定 profile 脚本运行并通过 `nsys stats` 解析；
- [ ] 阶段级 NVTX 精确截取 profile 通过。

后续只剩阶段级 NVTX 标注和精简 profile，执行前仍需用户明确批准。
