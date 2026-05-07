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

## Installing slurm on cluster nodes (`install-slurm` subcommand)

scheduleurm ships a tool that installs slurm + munge on a target node from source,
with graceful 3-tier fallback. Use it once per node; the tool detects existing
installs and is idempotent.

```bash
# Install on local + every NODES entry. Default tag: slurm-23-11-9-1 (LTS).
scheduleurm install-slurm

# Single node:
scheduleurm install-slurm --node jtl110gpu --sudo-pass <password>

# Different version:
scheduleurm install-slurm --tag slurm-24-05-0-1
```

**3-tier fallback chain (per node, in order):**

1. **Tier 1 — github clone on the node**: ssh in, `git clone --depth 1 -b <tag> https://github.com/SchedMD/slurm.git`, build with `./configure && make -j && sudo make install`. Requires github reach from the node.
2. **Tier 2 — rsync from local cache**: if tier 1 fails (corp network, air-gapped node), the local box clones once into `~/.cache/scheduleurm/slurm-src/`, then rsyncs to the node and runs the same build script with `--source-dir`. Requires github reach from the local box only.
3. **Tier 3 — LocalBackend fallback**: if both fail, the tool reports `no-local-cache` / `failed-rsync` / etc. The node continues to work via `LocalBackend` (ssh+nohup+setsid) — no slurm needed for scheduleurm to function. You can rerun the install later when network or sudo issues are resolved.

What the install script does on success:
- Installs build deps via apt: `build-essential autoconf libtool libmunge-dev libnl-3-dev libssl-dev …`
- Builds slurm from source to `/usr/local`
- Generates `/etc/munge/munge.key` if missing, starts munge daemon
- Creates `slurm` user, runtime dirs (`/var/spool/slurmctld`, `/var/log/slurm`, …)
- Writes a sensible default `/etc/slurm/slurm.conf` based on detected CPUs/RAM/GPUs (uses `proctrack/linuxproc` to avoid cgroup version issues)
- Auto-detects `/dev/nvidia[0-9]` and writes `gres.conf`
- Installs systemd units, starts `slurmctld` + `slurmd`, runs `sinfo` to verify

What it doesn't do (out of scope; manual if you need them):
- Multi-node cluster setup (cross-node munge key sync, `ControlMachine` config)
- LDAP / AD user federation
- Slurmdbd accounting database

After install, restart the watcher so `HybridBackend` re-detects the node:
```bash
systemctl --user restart scheduler
```

## Load-balanced migration (Phase 3.0)

When you submit `--preferred-node A` and node A has a long backlog while node B is nearly free, scheduleurm proactively migrates one queued task per dispatch cycle from A to B — **only when all of these hold**:

1. **A is heavily loaded**: A's `eta_load` (sum of remaining-seconds across all in-flight tasks pinned there) > `MIGRATION_LOAD_RATIO × B's eta_load` (default ratio = 2.0)
2. **B is genuinely free**: B's `eta_load` < `MIGRATION_FREE_THRESHOLD_S` (default 10 min). Migrating between two loaded nodes just shifts work, doesn't balance it.
3. **The task is portable**: `--preferred-node A`, no `--require-node`, not `auto_adopted`, ETA ≥ `MIGRATION_MIN_TASK_ETA_S` (300 s — short tasks don't recoup staging cost).
4. **Staging succeeds**: cwd is rsync'd to B if missing; `--ckpt-dir` is rsync'd if ≤ `MIGRATION_MAX_CKPT_SIZE_MB` (default 2048 MB); `python` from cmd is executable on B.

If any check fails → no migration this cycle, task stays on A, the reason is stashed in `task['last_migration_skip']` for inspection.

ETA is the load-imbalance signal, not task count (one 30-hour job ≠ thirty 1-hour jobs). ETA is computed every watcher cycle from log tails:

| Tier | Source | Accuracy | Trigger |
|---|---|---|---|
| 0 | `tqdm` bracket `[<elapsed><<remaining>, <rate>]` — its smoothed-rate estimate | Best | When script uses tqdm |
| 1 | Rate from `(current, total)` parsed in tail (e.g. `[Epoch 22/200]`, `Iter 1234/5000`) | Good | When tail has both numbers |
| 2 | Rate from `Iter N` in tail + `--max_iters N` / `--n_epochs N` / `--max_steps N` in cmd | Decent | When tail only has current |
| 3 | History EWMA − elapsed | Coarse | Fallback for new signatures |
| — | 0 (unknown) | None | No signal at all |

For best ETA accuracy, **wrap the training loop with tqdm**. The skill auto-checks scripts at submit time and proposes adding `from tqdm import tqdm; for x in tqdm(loop): ...` if missing — your loop's existing per-iter logging stays intact, tqdm just adds the bar.

Tunables (all env-var overridable; no code edit; takes effect after watcher restart):

| Var | Default | Meaning |
|---|---|---|
| `SCHEDULEURM_MIGRATION_LOAD_RATIO` | `2.0` | Source load must exceed this × target load |
| `SCHEDULEURM_MIGRATION_FREE_THRESHOLD_S` | `600` | Target's eta_load must be under this many seconds |
| `SCHEDULEURM_MIGRATION_MAX_PER_DISPATCH` | `1` | Cap on migrations per 60s dispatch cycle |
| `SCHEDULEURM_MIGRATION_MIN_TASK_ETA_S` | `300` | Tasks shorter than this aren't migrated (staging cost > savings) |
| `SCHEDULEURM_MIGRATION_MAX_CKPT_SIZE_MB` | `2048` | Reject migration if ckpt > this size (rsync would take too long) |
| `SCHEDULEURM_MIGRATION_MAX_CWD_SIZE_MB` | `1024` | Reject migration if cwd > this size (excludes .git/__pycache__/*.pyc, mirroring rsync excludes) |
| `SCHEDULEURM_MIGRATION_COOLDOWN_S` | `1800` | Same task cannot be migrated again within this many seconds (anti-oscillation) |
| `SCHEDULEURM_MIGRATION_MIN_SOURCE_LOAD_S` | `600` | Absolute floor on source eta_load — load_ratio alone allows trivial imbalances (target=0s, source=2s, ratio=2x → would migrate a 600s task to save 2s) |

Migration emits a `task_migrated` event in `~/.claude/scheduler/logs/watcher.log` (JSONL, also pushed to Feishu when configured), prints a `MIGRATE from → to (eta=...s)` line on `scheduler dispatch`, and rewrites the task's `preferred_node` + sets `last_block_reason` so it's visible in `scheduler status` too. Sibling `task_preempted` events surface the same way for high-prio preemption.

`scheduler status` now shows per-node `eta_load` so you can see imbalance directly:

```
=== nodes ===
  local       GPU0=...  cpu=...  ram_free=...  eta_load=8.0d
  jtl110gpu   GPU0=..., GPU1=...  cpu=...  ram_free=...  eta_load=1.2d
  jtl110gpu2  GPU0=..., GPU1=...  cpu=...  ram_free=...  eta_load=1.9d
```

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

**Slurm nodes bypass scheduleurm's local capacity gate.** scheduleurm's normal `dispatch`
runs `probe_node` and refuses placement when CPU/RAM/VRAM doesn't fit instantly — that's the
right thing for `LocalBackend` (we ARE the placement decider). For slurm nodes it would be
catastrophically wrong: the login node usually has no GPU, and busy clusters are exactly
when slurm's queue earns its keep. So slurm-detected nodes short-circuit `pick_placement` —
scheduleurm hands the task off via `sbatch` and slurm queues it. (Local nodes still get the
instant-fit gate; if both local and slurm can take a task, local wins because it starts now.)

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
