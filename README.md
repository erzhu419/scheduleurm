# scheduleurm

[中文 README](README_CN.md) · English

**Multi-resource (CPU + RAM + VRAM) job scheduler for ML training across heterogeneous nodes.** Ships as a [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skill, but the underlying `scheduler.py` is a plain Python script you can drive from any LLM agent (or by hand).

Built for the real shape of ML research: dozens of multi-hour training runs, mixed GPU/CPU jobs, mixed local/remote nodes, peak-memory uncertainty, processes that get killed and need to resume, externally-launched jobs that need to be tracked retroactively.

## What it actually does

| Problem you have today | What scheduleurm gives you |
|---|---|
| Eyeballing `nvidia-smi` to decide if there's room for one more run | `status` prints free CPU/RAM/VRAM per node; `dispatch` greedily fills capacity |
| Two seeds clobbering each other's `--out_dir` because you forgot one was running | Same-signature dedup at dispatch (race-guarded across launching window too) |
| 14h of training lost to OOM because the host has no swap left | RAM headroom enforced before placement (25% on WSL local; 10% on remote) |
| GPU utilization at 100% on one card while another sits idle | Best-fit warm-first placement; 1/3 VRAM packing rule for RL plateaus; util ≥90% saturation guard |
| Pre-empted task silently restarts from step 0 instead of resuming | `--ckpt-dir` + `--resume-flag` injects `<flag> <ckpt_path>` on re-dispatch |
| Child started outside the scheduler doesn't show up anywhere | Watcher auto-adopts external GPU + CPU procs every 60s |
| Same task being recommended `5GB RAM` because one bad sibling did | History uses **p80 of last 10 samples** — single outliers don't pin estimates |
| Re-running a config takes manual `--vram` / `--cpu` guessing | Auto-fills resource estimates from per-`--signature` history |
| `Bash run_in_background` doesn't notify when training finishes | `wait-for --signature 'X/*'` blocks until terminal — wrap in bg bash for a wakeup |

## Cluster model

A node is described by 5 numbers:

```python
NODES = {
    "local":     {"host": None,       "cpu_cores": 12, "ram_mb": 56*1024,  "ram_headroom_frac": 0.25, "max_vram_per_task": None, "max_concurrent_running": 10},
    "remote-A":  {"host": "remote-A", "cpu_cores": 12, "ram_mb": 200*1024, "ram_headroom_frac": 0.10, "max_vram_per_task": None, "max_concurrent_running": None},
    "remote-B":  {"host": "remote-B", "cpu_cores": 12, "ram_mb": 200*1024, "ram_headroom_frac": 0.10, "max_vram_per_task": None, "max_concurrent_running": None},
}
```

- `host=None` means local; otherwise an SSH alias from `~/.ssh/config` (passwordless required).
- `cpu_cores` / `ram_mb` are the **schedulable** budget, already net of OS reservation. (E.g. 16 physical cores → 12 schedulable on local; rest reserved for OS/IO.)
- `ram_headroom_frac` — fraction of RAM kept unallocated as buffer. Higher on WSL2 (OOM freezes the host).
- `max_vram_per_task` — `None` auto-derives from probed GPU `total_mb`; set a number to cap (e.g. WSL local 4060 8GB caps individual tasks at 4GB so two can share).
- `max_concurrent_running` — defense-in-depth above CPU/RAM bookkeeping (catches under-declared RAM tasks).

GPUs are auto-probed at dispatch via `nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu`. CPU/RAM is read from `/proc/loadavg` + `/proc/meminfo` (over SSH for remotes).

Edit the `NODES` dict at the top of `skill/scheduler.py` to match your cluster, then re-run `install.sh`. (See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the full set of knobs.)

## Install

Requires: Python 3.10+, `nvidia-smi` on each GPU node, passwordless SSH for remotes, optional `systemd --user` for the watcher daemon.

```bash
git clone https://github.com/erzhu419/scheduleurm.git
cd scheduleurm
./install.sh                  # COPY mode: cp skill files to ~/.claude/skills/scheduler/
# OR
./install.sh --link           # LINK mode: symlink ~/.claude/skills/scheduler -> clone/skill
                              #            edits in the clone are picked up immediately
                              #            (recommended if you plan to git pull / hack on it)
# OR
./install.sh --no-systemd     # skip watcher unit; you'll run watch by hand
                              # (combinable with --link)
```

**COPY vs LINK in one line:** COPY is "install once, forget"; LINK is "develop here, no copy step". After `--link`, `git pull` updates the live skill; `systemctl --user restart scheduler` picks up scheduler.py changes in the running watcher. Don't move/delete the clone in LINK mode — the symlink would break.

Verify:
```bash
python3 ~/.claude/skills/scheduler/scheduler.py status
systemctl --user status scheduler   # if --no-systemd was NOT passed
python3 ~/.claude/skills/scheduler/test_regression.py    # 290+ regression checks
```

Uninstall: `./uninstall.sh` (preserves state) · `./uninstall.sh --purge-state` (wipes queue/history/logs).

## Use it (3 ways)

### From Claude Code as a skill

After install, the skill is auto-discovered. Just say what you want:

> 跑这个脚本 (run this script)
>
> GPU 还空吗 (any GPU room left?)
>
> 跑这 6 个 ablation seeds (run these 6 ablation seeds)
>
> 取消 t0042 (cancel t0042)
>
> 跑完通知我 (wake me when done)

The skill translates intent → `submit` / `dispatch` / `status` / `wait-for` calls. See [`skill/SKILL.md`](skill/SKILL.md) for the full decision rules.

### From any MCP-capable client (ChatGPT Desktop / Cursor / Cline / …)

The MCP wrapper at [`skill/integrations/scheduler_mcp.py`](skill/integrations/scheduler_mcp.py) exposes 8 tools (`submit_task`, `dispatch`, `status`, `show_task`, `cancel_task`, `history`, `queue_dump`, `task_log`) over stdio JSON-RPC. Per-client config in [`skill/integrations/README.md`](skill/integrations/README.md).

### Directly from the CLI

```bash
sch=~/.claude/skills/scheduler/scheduler.py

# Submit a training run; resource estimates auto-fill from history (or fall back to defaults)
python3 $sch submit \
  --description "RE-SAC b1 multi-gpu" \
  --cwd /path/to/repo \
  --signature "RE-SAC/b1/multi-gpu" \
  --git-repo /path/to/repo \
  --ckpt-dir /path/to/repo/ckpts \
  --resume-flag '--resume_from' \
  --cmd "/abs/path/to/python -u train.py --seed 42"

# Greedy-fill capacity across all queued tasks
python3 $sch dispatch

# Live status
python3 $sch status

# Wait for a batch (exit 0 when all in-signature reach terminal)
python3 $sch wait-for --signature 'RE-SAC/b1/*' --poll 30

# Cancel a queued task; running needs --force
python3 $sch cancel t0042
```

Full subcommand list: `python3 scheduler.py --help`.

## Slurm coexistence (Phase 2)

If a target node has `sbatch` and `squeue` installed, scheduleurm **automatically routes
through slurm** — generating an `sbatch` script (with `--gres=gpu:1`, `--mem`, `--cpus-per-task`,
`--time` from history EWMA × 3, and your task's `--cmd` as body), submitting it via
`sbatch -` (script piped through stdin so nothing lands on the node's filesystem), tracking
liveness via `squeue`, and killing via `scancel`. Detection is per-node and cached for the
process lifetime.

| Target node | What you get |
|---|---|
| Has slurm | scheduleurm generates sbatch, slurm handles cross-user contention + cgroup isolation + walltime. scheduleurm still does signature dedup, history-based estimation, resume injection. |
| No slurm | scheduleurm runs `ssh + nohup + setsid` directly; everything as before |
| Mixed cluster | Per-node — node A can be slurm, node B can be ssh+nohup, scheduleurm routes correctly |

What scheduleurm keeps owning even on slurm nodes (because slurm doesn't): per-signature p80
history estimation, automatic resume-from-checkpoint flag injection, cross-task `--ckpt-dir`
conflict detection, env-deploy (docker/conda) wrapping, MCP/skill UI, auto-adoption of
externally-launched processes.

What slurm owns when present: queue ordering across users, cgroup-based memory/CPU caps,
walltime enforcement, GPU pinning via `--gres`. Peak VRAM/RAM tracking via `sstat`/`sacct`
isn't enabled in v1 — slurm enforces declared limits, so peak ≈ declared in practice.

The class hierarchy:
- `Backend` (ABC) — `launch` / `kill` / `batch_probe`
- `LocalBackend` — current `ssh + nohup` path
- `SlurmBackend` — `sbatch` / `scancel` / `squeue`
- `HybridBackend` — per-node routing; this is what `_BACKEND` actually is

Phase 3 (planned) will add `MultiUserLocalBackend` for the case where a node has *no* slurm
**and** multiple scheduleurm users contend (cooperative shared state at `/tmp/scheduleurm/`).

## Architecture (one screen)

```
                     ┌────────────────────────────────────────────────┐
                     │  scheduler.py — single Python module            │
                     │                                                 │
   user / agent →    │   submit  → queue.json (atomic, fcntl-locked)   │
                     │   dispatch → pick_placement(NODES, history) →   │
                     │              probe_node (ssh or local) →        │
                     │              wrap_cmd_docker / inject -u →      │
                     │              ssh node 'cmd' OR local Popen      │
                     │   watch    → 60s loop:                          │
                     │              - dispatch                         │
                     │              - check_running (peak VRAM/RAM)    │
                     │              - diagnose_terminal (4 rules)      │
                     │              - eviction (mem ≥ 1/3 AND util ≥90%) │
                     │              - auto-adopt external procs        │
                     └────────────────────────────────────────────────┘
                                          ↕
                              ~/.claude/scheduler/
                                  queue.json          ← live tasks
                                  vram_history.json   ← per-sig p80 samples
                                  escalations.jsonl   ← heal session inbox
                                  logs/watcher.log    ← rotated, JSONL events
```

**State separation:** the `skill/` dir contains code only. All runtime state lives in `~/.claude/scheduler/`. `install.sh` never touches state — re-running upgrades code only.

**Key invariants** (all enforced + regression-tested):
1. Same `--signature` cannot be in two `running`/`launching` slots simultaneously (race-guard at dispatch).
2. Tasks killed externally with a usable checkpoint requeue with `--resume-flag` injected (never restart from step 0).
3. Atomic state writes: tmp file → fsync → `os.replace` (no half-written queue.json after SIGKILL).
4. `kill` always uses process group + signal escalation (SIGTERM → wait 10s → SIGKILL); docker tasks `docker stop` first.
5. Single oversized allocation = OOM in <5min (visible). Slow leak protection comes from `peak_ram_mb` upward tracking + post-dispatch eviction.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the deeper version.

## Resource estimation: how p80 history works

When you submit a task with `--signature X`:
1. If `--cpu`/`--ram-mb`/`--vram` explicitly passed → use them.
2. Else look up `vram_history[X]` → use the **p80 of the last 10 peak samples**.
3. Else cascade: same-description-key siblings → same-prefix history → project median → defaults.

When a task finishes, watcher folds `peak_vram_mb` and `peak_ram_mb` into `vram_history[X]["{vram,ram}_samples"]`, recomputes `vram_mb`/`ram_mb` as the p80, and caps the samples list at 10. **One outlier doesn't pin all future estimates** — that was the headline bug in this category before p80.

Legacy single-value records (from before p80) auto-migrate: the existing value is seeded as the first sample on the next write.

## Common request → action

| Phrase | Action |
|---|---|
| "跑这个" / "run this" | `submit` + `dispatch`; report node:GPU + log path + resume_from |
| "跑这 N 个" / "run these N seeds" | submit all N with same `--signature` prefix; ONE `dispatch` |
| "GPU 还空吗" / "status" | `status`; highlight GPUs under 1/3 used |
| "重新分配" / "rebalance" | `dispatch` (watcher does this automatically every 60s) |
| "取消 t0042" | queued: `cancel`. running: confirm + `cancel --force` |
| "清空队列" / "clear queue" | `clear-queue` (dry-run) → `clear-queue --confirm` (running tasks NEVER touched) |
| "看看 t0042" | `show t0042` |
| "跑完通知我" | wrap `wait-for --signature 'X/*'` in `Bash run_in_background` |
| "外面那个跑的也加进来" | watcher auto-adopts every 60s; manual `adopt` for edge cases |
| "看看资源画像" | `history` |

## Hard constraints (the design refuses to do these)

- Never auto-`cancel --force` running tasks. User must confirm.
- Never touch running tasks during `dispatch`/`clear-queue`/`rebalance`.
- Never run image push or env rsync while holding the state lock (would block watcher).
- Never silently restart a task from step 0 — if no `--resume-flag` was set, it's by user choice; if set, the resume path is exercised.
- Never kill a task that has a usable checkpoint (the eviction path picks the youngest, never one with `peak_vram>0` AND a recent ckpt).

## Project layout

```
scheduleurm/
├── README.md / README_CN.md
├── LICENSE
├── install.sh             # idempotent: copies skill, installs systemd unit
├── uninstall.sh           # removes skill (state preserved unless --purge-state)
├── skill/                 # source of truth — what install.sh copies into Claude Code
│   ├── SKILL.md
│   ├── scheduler.py
│   ├── env_deploy.py      # docker / conda env delivery (per --env-spec)
│   ├── tui.py             # tui-top: live cluster view (top-style refresh)
│   ├── test_regression.py # 290+ regression checks (one per known footgun)
│   ├── test_hook.sh       # PostToolUse hook script for Claude Code
│   └── integrations/
│       ├── scheduler_mcp.py    # MCP server wrapper (8 tools)
│       └── README.md
├── systemd/
│   └── scheduler.service  # user unit; install.sh rewrites paths and enables
└── docs/
    ├── ARCHITECTURE.md    # deeper design notes
    ├── CONFIGURATION.md   # NODES dict, env vars, headroom tuning
    └── FOOTGUNS.md        # the war-stories that produced regression tests
```

## Regression tests

`skill/test_regression.py` has 290+ checks, each tied to a real bug that hit production. Examples:

- OOM-pattern false-positives (bare "Killed" matched our own diag text → 50h compute lost)
- Test-state leak into live queue.json (sentinel `ast.walk` test catches future regressions)
- Stale `launching` WAL state after watcher restart (recovery path verified)
- Image digest drift skipping the launch-path push check (P1 bypass)
- p80 outlier resistance + legacy single-value migration

Run before any non-trivial scheduler edit:
```bash
python3 skill/test_regression.py
```

A `PostToolUse` hook (`test_hook.sh`) auto-runs the suite after Edit/Write to scheduler files when integrated with Claude Code.

## Why "scheduleurm"?

`scheduler` + `slurm`. It is not slurm — slurm is for HPC clusters with batch queues; this is for the messier reality of a few personal-dev nodes where you want one tool that survives `Ctrl-C`, manual `kill -9`, host reboots, and externally-spawned children, without losing track of what's running where.

## Status / scope

- Active use: WSL2 local + 2 remote SSH nodes, 24/7 background watcher, several thousand tasks/month.
- Tested with: NVIDIA GPUs (3060/3080Ti/4060), Linux only, single-user only.
- Not tested: Mac/Windows native, multi-user, K8s/SLURM/RayCluster integration.
- Not a goal: enterprise scheduler features (queues, priorities by group, fairness, accounting). It's deliberately a single-user tool.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Designed iteratively with Claude Opus + Codex review. Every regression test has a story behind it; the [`docs/FOOTGUNS.md`](docs/FOOTGUNS.md) file is the changelog of those stories.
