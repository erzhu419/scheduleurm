# scheduleurm

[English README](README.md) · 中文

**多资源（CPU + RAM + VRAM）调度器，跨异构节点跑 ML 训练任务。** 以 [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skill 形态发布，但底层 `scheduler.py` 是一个单文件 Python 脚本，可被任意 LLM agent 或人工驱动。

为真实 ML 研究场景而设计：几十个跑数小时的训练同时排队、GPU/CPU 任务混合、本地+远程节点混合、内存峰值不确定、进程会被 kill 然后需要 resume、外部启动的进程也要追踪。

## 它实际能解决的问题

| 你今天的痛点 | scheduleurm 怎么解决 |
|---|---|
| 盯 `nvidia-smi` 决定还能不能再塞一个 | `status` 直接显示每节点剩余 CPU/RAM/VRAM；`dispatch` 贪心填充空位 |
| 两个 seed 同时跑覆盖了对方的 `--out_dir` | 同 signature 在 dispatch 时去重（含 launching 窗口的竞态保护） |
| 14 小时训练因为主机没 swap 被 OOM 杀掉 | 放置前强制 RAM 余量（WSL local 25%，远程 10%） |
| 一张卡 100% 利用率另一张闲着 | best-fit 优先暖卡；RL 平台期适用的 1/3 VRAM 打包规则；util ≥90% 饱和守卫 |
| 被抢占的任务静默从 step 0 重启而不是 resume | `--ckpt-dir` + `--resume-flag` 在重派时自动注入 `<flag> <ckpt_path>` |
| 外部直接 nohup 起的进程不在调度器视野里 | watcher 每 60s 自动 adopt 外部 GPU + CPU 进程 |
| 一个失败 sibling 把整组估计值拉到 5GB | history 用**最近 10 个采样的 p80** —— 单个离群点不再钉死估计 |
| 重跑同一个 config 还要手工猜 `--vram` / `--cpu` | 按 `--signature` 自动从历史填资源估计 |
| `Bash run_in_background` 训练完了不通知 | `wait-for --signature 'X/*'` 阻塞直到全部终态；包在后台 bash 里就能唤醒 |

## 集群模型

一个节点用 5 个数描述：

```python
NODES = {
    "local":     {"host": None,       "cpu_cores": 12, "ram_mb": 56*1024,  "ram_headroom_frac": 0.25, "max_vram_per_task": None, "max_concurrent_running": 10},
    "remote-A":  {"host": "remote-A", "cpu_cores": 12, "ram_mb": 200*1024, "ram_headroom_frac": 0.10, "max_vram_per_task": None, "max_concurrent_running": None},
    "remote-B":  {"host": "remote-B", "cpu_cores": 12, "ram_mb": 200*1024, "ram_headroom_frac": 0.10, "max_vram_per_task": None, "max_concurrent_running": None},
}
```

- `host=None` 是本地；其他值是 `~/.ssh/config` 里的 SSH 别名（必须免密）。
- `cpu_cores` / `ram_mb` 是**可调度**预算，已经扣了 OS 保留量。（例：16 物理核 → 本地 12 可调度，剩下留给 OS/IO。）
- `ram_headroom_frac` —— RAM 必须保留的余量比例。WSL2 上要更高（OOM 会冻整机）。
- `max_vram_per_task` —— `None` 时自动从探测到的 GPU `total_mb` 推导；填数字则封顶（如 WSL 4060 8GB 把单任务封到 4GB 让两个能共享）。
- `max_concurrent_running` —— CPU/RAM 记账之上的兜底（防止任务低估 RAM 而堆爆）。

GPU 在 dispatch 时通过 `nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu` 自动探测。CPU/RAM 来自 `/proc/loadavg` + `/proc/meminfo`（远程节点走 SSH）。

改 `skill/scheduler.py` 顶部的 `NODES` 字典匹配你的集群，然后重跑 `install.sh`。完整旋钮列表见 [docs/CONFIGURATION.md](docs/CONFIGURATION.md)。

## 安装

依赖：Python 3.10+，每个 GPU 节点上有 `nvidia-smi`，远程节点免密 SSH，可选 `systemd --user` 跑 watcher 守护。

```bash
git clone https://github.com/erzhu419/scheduleurm.git
cd scheduleurm
./install.sh                  # COPY 模式：把 skill 拷到 ~/.claude/skills/scheduler/
# 或者
./install.sh --link           # LINK 模式：symlink ~/.claude/skills/scheduler -> clone/skill
                              #            clone 里改了立即生效，不用重装
                              #            （推荐：会 git pull / 自己改的人）
# 或者
./install.sh --no-systemd     # 跳过 watcher unit；自己手动跑 watch
                              # （可以和 --link 组合）
```

**COPY vs LINK 一句话对比**：COPY 是"装一次就忘"，LINK 是"在 clone 里直接开发，不用拷"。`--link` 之后 `git pull` 直接更新 live skill；`systemctl --user restart scheduler` 让 watcher 加载新版 scheduler.py。LINK 模式下别挪/删 clone —— symlink 会断。

验证：
```bash
python3 ~/.claude/skills/scheduler/scheduler.py status
systemctl --user status scheduler   # 没传 --no-systemd 才有
python3 ~/.claude/skills/scheduler/test_regression.py    # 290+ 回归测试
```

卸载：`./uninstall.sh`（保留状态） · `./uninstall.sh --purge-state`（连同 queue/history/log 一起删）。

## 三种用法

### 在 Claude Code 里当 skill 用

安装后 skill 自动被发现。直接说人话：

> 跑这个脚本
>
> GPU 还空吗
>
> 跑这 6 个 ablation seeds
>
> 取消 t0042
>
> 跑完通知我

Skill 把意图翻译成 `submit` / `dispatch` / `status` / `wait-for` 调用。完整决策规则见 [`skill/SKILL.md`](skill/SKILL.md)。

### 在任何 MCP 客户端里（ChatGPT Desktop / Cursor / Cline / …）

[`skill/integrations/scheduler_mcp.py`](skill/integrations/scheduler_mcp.py) 是 MCP server，通过 stdio JSON-RPC 暴露 8 个工具（`submit_task`, `dispatch`, `status`, `show_task`, `cancel_task`, `history`, `queue_dump`, `task_log`）。每客户端配置见 [`skill/integrations/README.md`](skill/integrations/README.md)。

### 直接命令行

```bash
sch=~/.claude/skills/scheduler/scheduler.py

# 提交训练；资源估计从历史自动填（或回落到默认）
python3 $sch submit \
  --description "RE-SAC b1 multi-gpu" \
  --cwd /path/to/repo \
  --signature "RE-SAC/b1/multi-gpu" \
  --git-repo /path/to/repo \
  --ckpt-dir /path/to/repo/ckpts \
  --resume-flag '--resume_from' \
  --cmd "/abs/path/to/python -u train.py --seed 42"

# 贪心地把队列里所有能放下的塞满空位
python3 $sch dispatch

# 实时状态
python3 $sch status

# 等一批跑完（全部到达终态时 exit 0）
python3 $sch wait-for --signature 'RE-SAC/b1/*' --poll 30

# 取消队列中的任务；running 需要 --force
python3 $sch cancel t0042
```

完整子命令：`python3 scheduler.py --help`。

## 跟 Slurm 共存（Phase 2）

如果目标节点装了 `sbatch` 和 `squeue`，scheduleurm 会**自动通过 slurm 跑**——生成一份 `sbatch`
脚本（带 `--gres=gpu:1`、`--mem`、`--cpus-per-task`、`--time` 是历史 EWMA × 3、body 是你的
`--cmd`），通过 `sbatch -` 用 stdin 提交（脚本不落盘到节点），靠 `squeue` 跟踪生死，靠 `scancel`
取消。检测是按节点的，进程生命周期内缓存。

| 目标节点 | 你拿到什么 |
|---|---|
| 装了 slurm | scheduleurm 生成 sbatch，slurm 处理跨用户排队 + cgroup 隔离 + walltime。scheduleurm 仍然做 signature 去重、history 估计、resume 注入 |
| 没装 slurm | scheduleurm 直接 `ssh + nohup + setsid`，跟之前完全一样 |
| 混合集群 | 按节点判断 —— A 节点走 slurm，B 节点走 ssh+nohup，路由自动处理 |

scheduleurm 在 slurm 节点上**仍然**owns 的（因为 slurm 不做这些）：每 signature 的 p80 历史
估计、ckpt resume flag 自动注入、跨任务 `--ckpt-dir` 冲突检测、env-deploy（docker/conda）
包装、MCP/skill UI、外部启动进程的自动 adopt。

slurm **接管**的：跨用户队列排序、cgroup 内存/CPU 上限、walltime 强制、`--gres` GPU 绑定。
通过 `sstat`/`sacct` 抓 peak VRAM/RAM 在 v1 没启用 —— slurm 已经强制 declared 上限，所以
peak ≈ declared。

类层次：
- `Backend`（ABC）—— `launch` / `kill` / `batch_probe`
- `LocalBackend` —— 当前的 `ssh + nohup` 路径
- `SlurmBackend` —— `sbatch` / `scancel` / `squeue`
- `HybridBackend` —— 按节点路由；`_BACKEND` 实际就是这个

Phase 3（计划中）会加 `MultiUserLocalBackend`，处理"节点没 slurm **且**多个 scheduleurm 用户
同时用"的场景（`/tmp/scheduleurm/` 协作共享状态）。

## 架构（一屏看完）

```
                     ┌────────────────────────────────────────────────┐
                     │  scheduler.py — 单 Python 模块                   │
                     │                                                 │
   user / agent →    │   submit  → queue.json (atomic, fcntl-locked)   │
                     │   dispatch → pick_placement(NODES, history) →   │
                     │              probe_node (ssh 或 local) →        │
                     │              wrap_cmd_docker / inject -u →      │
                     │              ssh node 'cmd' 或 local Popen      │
                     │   watch    → 60s 循环：                          │
                     │              - dispatch                         │
                     │              - check_running (peak VRAM/RAM)    │
                     │              - diagnose_terminal (4 条规则)      │
                     │              - eviction (mem ≥1/3 AND util ≥90%)│
                     │              - 自动 adopt 外部进程               │
                     └────────────────────────────────────────────────┘
                                          ↕
                              ~/.claude/scheduler/
                                  queue.json          ← 活跃任务
                                  vram_history.json   ← 每 sig 的 p80 采样
                                  escalations.jsonl   ← heal session 收件箱
                                  logs/watcher.log    ← 滚动 JSONL 事件
```

**状态分离**：`skill/` 只有代码。运行时状态全在 `~/.claude/scheduler/`。`install.sh` 永远不动状态 —— 重跑只升级代码。

**关键不变量**（全部强制 + 回归测试覆盖）：
1. 同一 `--signature` 不能同时占两个 `running`/`launching` 槽（dispatch 时竞态保护）。
2. 外部杀死但有可用 ckpt 的任务，重派时自动注入 `--resume-flag`（永不从 step 0 重启）。
3. 状态原子写：tmp → fsync → `os.replace`（SIGKILL 也不会留下半写 queue.json）。
4. `kill` 永远走进程组 + 信号升级（SIGTERM → 等 10s → SIGKILL）；docker 任务先 `docker stop`。
5. 单次大块分配 = OOM 在 5min 内可见。慢泄漏靠 `peak_ram_mb` 单调追踪 + dispatch 后驱逐覆盖。

更深入的设计见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 资源估计：p80 历史的工作方式

提交带 `--signature X` 的任务时：
1. 显式传了 `--cpu`/`--ram-mb`/`--vram` → 用这些。
2. 否则查 `vram_history[X]` → 用**最近 10 次峰值的 p80**。
3. 否则级联：同 description-key 的兄弟 → 同前缀历史 → 项目中位数 → 默认值。

任务终结时 watcher 把 `peak_vram_mb` / `peak_ram_mb` 折进 `vram_history[X]["{vram,ram}_samples"]`，重算 p80 写回 `vram_mb`/`ram_mb`，最多保留 10 个采样。**单个离群点不会钉死后续估计** —— 这是 p80 之前最常踩的坑。

p80 之前的旧记录（单值）会自动迁移：下次写入时把已有值作为第一个采样种入 samples 列表。

## 常见请求 → 动作

| 说法 | 动作 |
|---|---|
| "跑这个" | `submit` + `dispatch`；汇报 node:GPU + log 路径 + resume_from |
| "跑这 N 个 seed" | 用同一 `--signature` 前缀全部 submit；一次 `dispatch` |
| "GPU 还空吗" / "status" | `status`；高亮 1/3 以下的 GPU |
| "重新分配" / "rebalance" | `dispatch`（watcher 每 60s 自动跑一次） |
| "取消 t0042" | queued: `cancel`. running: 确认 + `cancel --force` |
| "清空队列" | `clear-queue`（dry-run）→ `clear-queue --confirm`（running 永远不动） |
| "看看 t0042" | `show t0042` |
| "跑完通知我" | 把 `wait-for --signature 'X/*'` 包到 `Bash run_in_background` |
| "外面那个加进来" | watcher 每 60s 自动 adopt；边缘情况手动 `adopt` |
| "看看资源画像" | `history` |

## 硬约束（设计上拒绝做这些）

- 永不自动 `cancel --force` 正在运行的任务，必须用户确认。
- `dispatch`/`clear-queue`/`rebalance` 永不动 running 任务。
- 永不在持有状态锁时跑 image push 或 env rsync（会卡 watcher）。
- 永不静默从 step 0 重启 —— 没设 `--resume-flag` 是用户的选择；设了就走 resume 路径。
- 永不杀有可用 ckpt 的任务（驱逐路径选最年轻的，从不选 `peak_vram>0` 且有近期 ckpt 的）。

## 项目结构

```
scheduleurm/
├── README.md / README_CN.md
├── LICENSE
├── install.sh             # 幂等：拷贝 skill，安装 systemd 单元
├── uninstall.sh           # 删 skill（默认保留状态，--purge-state 全删）
├── skill/                 # 真正的源 —— install.sh 拷贝到 Claude Code 的目录
│   ├── SKILL.md
│   ├── scheduler.py
│   ├── env_deploy.py      # docker / conda 环境投递（按 --env-spec）
│   ├── tui.py             # tui-top：实时集群视图（top 风格刷新）
│   ├── test_regression.py # 290+ 回归测试（每个对应一个已知 footgun）
│   ├── test_hook.sh       # Claude Code 的 PostToolUse 钩子脚本
│   └── integrations/
│       ├── scheduler_mcp.py    # MCP server 包装（8 个工具）
│       └── README.md
├── systemd/
│   └── scheduler.service  # user 单元；install.sh 改写路径并启用
└── docs/
    ├── ARCHITECTURE.md    # 更深入的设计笔记
    ├── CONFIGURATION.md   # NODES 字典、环境变量、余量调参
    └── FOOTGUNS.md        # 产生这些回归测试的真实事故
```

## 回归测试

`skill/test_regression.py` 有 290+ 检查，每条对应一个真打过脸的 bug。例如：

- OOM 模式假阳性（裸 "Killed" 匹配到我们自己 diag 文本 → 丢了 50h 计算）
- 测试状态泄漏到 live queue.json（用 `ast.walk` 写哨兵测试防未来回归）
- watcher 重启后留下的 stale `launching` WAL 状态（验证恢复路径）
- 镜像 digest 漂移在 launch path 绕过推送检查（P1 bypass）
- p80 抗离群 + 旧单值记录迁移

任何非 trivial 的 scheduler 编辑前都先跑：
```bash
python3 skill/test_regression.py
```

如果接到 Claude Code，`PostToolUse` 钩子（`test_hook.sh`）会在 Edit/Write scheduler 文件后自动跑。

## 名字为啥叫 "scheduleurm"

`scheduler` + `slurm`。它**不是** slurm —— slurm 是给 HPC 集群批量队列用的；这个工具给的是更乱的现实场景：几台个人开发机，希望有一个工具能扛住 `Ctrl-C`、手工 `kill -9`、主机重启、外部派生的子进程，不丢任何状态。

## 状态 / 范围

- 实际在用：WSL2 local + 2 个远程 SSH 节点，24/7 后台 watcher，每月几千个任务。
- 测试过：NVIDIA GPU（3060/3080Ti/4060），仅 Linux，仅单用户。
- 没测过：Mac/Windows 原生、多用户、K8s/SLURM/RayCluster 集成。
- 不是目标：企业级调度器特性（队列、按组的优先级、公平性、计费）。它就是个单用户工具。

## 许可证

MIT —— 见 [LICENSE](LICENSE)。

## 致谢

跟 Claude Opus + Codex review 一起迭代设计。每个回归测试背后都有故事；[`docs/FOOTGUNS.md`](docs/FOOTGUNS.md) 就是这些故事的 changelog。
