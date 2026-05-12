# scheduleurm

[中文 README](README_CN.md) · English

**Multi-resource (CPU + RAM + VRAM) job scheduler for ML training across heterogeneous nodes.** Ships as a [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skill, but the underlying `scheduler.py` is a plain Python script you can drive from any LLM agent (or by hand).

Built for the real shape of ML research: dozens of multi-hour training runs, mixed GPU/CPU jobs, mixed local/remote nodes, peak-memory uncertainty, processes that get killed and need to resume, externally-launched jobs that need to be tracked retroactively.

## What it actually does

| Problem you have today | What scheduleurm gives you |
|---|---|
| Eyeballing `nvidia-smi` to decide if there's room for one more run | `status` prints free CPU/RAM/VRAM per node; `dispatch` greedily fills capacity |
| Duplicate runs clobbering `--out_dir` / `--ckpt-dir` because you forgot one was running | Same run-identity dedup at dispatch (race-guarded across launching window too); active `--ckpt-dir` is globally exclusive by default |
| 14h of training lost to OOM because the host has no swap left | RAM headroom enforced before placement (fixed 2GB on WSL local; 10% on remote) |
| GPU utilization at 100% but VRAM still low | RL nodes can ignore util and pack by the 1/3+grace VRAM rule; strict nodes can keep a util saturation guard |
| Pre-empted task silently restarts from step 0 instead of resuming | `--ckpt-dir` + `--resume-flag` injects `<flag> <ckpt_path>` on re-dispatch |
| Child started outside the scheduler doesn't show up anywhere | Watcher auto-adopts external GPU + CPU procs every 60s |
| Same task being recommended `5GB RAM` because one bad sibling did | History uses **p80 of last 10 samples** — single outliers don't pin estimates |
| Re-running a config takes manual `--vram` / `--cpu` guessing | Auto-fills resource estimates from per-`--signature` history |
| `Bash run_in_background` doesn't notify when training finishes | `wait-for --signature 'X/*'` blocks until terminal — wrap in bg bash for a wakeup |

## Cluster model

A node is described by a few resource knobs:

```python
NODES = {
    "local":     {"host": None,       "cpu_cores": 16, "ram_mb": 56*1024,  "ram_headroom_mb": 2048, "ram_headroom_frac": 0.20, "max_vram_per_task": None, "max_concurrent_running": 10, "gpu_util_saturation_pct": None},
    "remote-A":  {"host": "remote-A", "cpu_cores": 12, "ram_mb": 200*1024, "ram_headroom_frac": 0.10, "max_vram_per_task": None, "max_concurrent_running": None},
    "remote-B":  {"host": "remote-B", "cpu_cores": 12, "ram_mb": 200*1024, "ram_headroom_frac": 0.10, "max_vram_per_task": None, "max_concurrent_running": None},
}
```

- `host=None` means local; otherwise an SSH alias from `~/.ssh/config` (passwordless required).
- `cpu_cores` / `ram_mb` are the **schedulable** budget. Use physical cores, not hyperthreads, unless you deliberately want to oversubscribe CPU.
- `ram_headroom_mb` / `ram_headroom_frac` — RAM kept unallocated as buffer. Fixed MB wins when set; remotes usually use the fraction.
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

> Run this script
>
> Any GPU room left?
>
> Launch these 6 ablation seeds
>
> Cancel t0042
>
> Wake me when it's done

Trigger surface (the skill description routes any of these to the scheduler):

- **Submit / launch a job** — "run a job", "submit a job", "launch a job", "train", "eval", "inference", "run this script", "run this python", "run a sweep", "data prep", "kick off X", "fire off X", "queue up X", "schedule X", "dispatch X", "deploy X", "send to GPU", "put on jtl110gpu"
- **Status / inspect** — "GPU free?", "any free RAM?", "what's running?", "which node has room?", "node status", "show queue", "show jobs", "show tasks", "how many tasks running"
- **Job control** — "cancel job", "kill job", "stop job", "clear queue", "forget X", "rebalance", "redispatch", "reassign"

(See [`README_CN.md`](README_CN.md) for the Chinese trigger phrases — the skill matches both.)

The skill translates intent → `submit` / `dispatch` / `status` / `wait-for` calls. See [`skill/SKILL.md`](skill/SKILL.md) for the full decision rules.

### From any MCP-capable client (ChatGPT Desktop / Cursor / Cline / …)

The MCP wrapper at [`skill/integrations/scheduler_mcp.py`](skill/integrations/scheduler_mcp.py) exposes 9 tools (`submit_task`, `dispatch`, `status`, `doctor`, `show_task`, `cancel_task`, `history`, `queue_dump`, `task_log`) over stdio JSON-RPC. Per-client config in [`skill/integrations/README.md`](skill/integrations/README.md).

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

The same progress signal is also folded into `~/.claude/scheduler/runtime_history.json`.
For later Slurm submissions with the same cmd/cwd/env parameters, scheduleurm uses
`p80(projected total runtime) × 1.2` as `--time` (10-minute floor, 7-day cap) before
falling back to the legacy signature EWMA × 3 / 24h default. The exact runtime key
intentionally ignores signature/project/description labels so a harmless signature
rename does not lose local timing history.

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
| `SCHEDULEURM_STAGING_TTL_S` | `600` | TTL on _STAGING_CACHE / _STAGED_TASKS entries; older than this forces re-rsync to pick up content edits while the task waited in queue |
| `SCHEDULEURM_STAGING_FAIL_COOLDOWN_S` | `3600` | Per-(task,target) cooldown after a failed staging attempt — prevents the first 2 doomed candidates from permanently starving the rest of the queue |

Migration emits a `task_migrated` event in `~/.claude/scheduler/logs/watcher.log` (JSONL, also pushed to Feishu when configured), prints a `MIGRATE from → to (eta=...s)` line on `scheduler dispatch`, and rewrites the task's `preferred_node` + sets `last_block_reason` so it's visible in `scheduler status` too. Sibling `task_preempted` events surface the same way for high-prio preemption.

`scheduler status` now shows per-node `eta_load` so you can see imbalance directly:

```
=== nodes ===
  local       GPU0=...  cpu=...  ram_free=...  eta_load=8.0d
  jtl110gpu   GPU0=..., GPU1=...  cpu=...  ram_free=...  eta_load=1.2d
  jtl110gpu2  GPU0=..., GPU1=...  cpu=...  ram_free=...  eta_load=1.9d
```

## Slurm coexistence (Phase 2)

scheduleurm defaults to its own LocalBackend placement for small jobs even if a target
node has `sbatch` and `squeue` installed. Slurm is opt-in or hardware-aware: set
`NODES["node"]["slurm_backend"] = "slurm"` to force a real shared cluster through Slurm,
set `slurm_gpu_backend` / `slurm_cpu_backend` per resource bucket, or submit a task with
explicit `--slurm-partition/account/qos` fields. Without a force setting, cluster-class
nodes (default: >=128 schedulable CPU cores or >=8 GPUs) route only LLM, multi-GPU,
large-VRAM, or large-CPU jobs through Slurm; small one-GPU jobs keep scheduleurm packing.
In Slurm mode scheduleurm generates an
`sbatch` script (with `--gres=gpu:1`, `--mem`, `--cpus-per-task`, `--time` from exact
runtime history first, then history EWMA × 3, and your task's `--cmd` as body), submits it
via stdin, tracks liveness via `squeue`, and kills via `scancel`.

| Target node | What you get |
|---|---|
| Default, including small nodes that merely have slurm installed | scheduleurm runs `ssh + nohup + setsid`; with `enable_claims=True` per node it adds atomic cross-scheduler / cross-user resource exclusion via `/tmp/scheduleurm/claims.json + flock` |
| Forced Slurm node, explicit Slurm task, or hardware-aware large task on a large Slurm-capable node | scheduleurm generates sbatch, slurm handles cross-user queueing + cgroup isolation + walltime. scheduleurm still does signature dedup, history-based estimation, resume injection. |
| Mixed deployment | Per-node/per-task — node A can be Slurm opt-in, node B can be scheduleurm local, and existing `slurm_job_id` tasks keep being tracked by SlurmBackend |

What scheduleurm keeps owning even on slurm nodes (because slurm doesn't): per-signature p80
history estimation, automatic resume-from-checkpoint flag injection, cross-task `--ckpt-dir`
conflict detection, env-deploy (docker/conda) wrapping, MCP/skill UI, auto-adoption of
externally-launched processes.

**Slurm-routed tasks bypass scheduleurm's local capacity gate.** scheduleurm's normal `dispatch`
runs `probe_node` and refuses placement when CPU/RAM/VRAM doesn't fit instantly — that's the
right thing for `LocalBackend` (we ARE the placement decider). For slurm nodes it would be
catastrophically wrong: the login node usually has no GPU, and busy clusters are exactly
when slurm's queue earns its keep. So opt-in Slurm routes short-circuit `pick_placement` —
scheduleurm hands the task off via `sbatch` and slurm queues it. Default-local nodes still
get the instant-fit gate; if both local and Slurm can take a task, local wins because it
starts now.

What slurm owns when present: queue ordering across users, cgroup-based memory/CPU caps,
walltime enforcement, GPU pinning via `--gres`. Peak VRAM/RAM tracking via `sstat`/`sacct`
isn't enabled in v1 — slurm enforces declared limits, so peak ≈ declared in practice.

For small personal nodes where Slurm is installed but GPU sharing is desired, do nothing:
the default is LocalBackend placement/VRAM packing. On large nodes, default/`auto` Slurm
routing only catches heavyweight jobs; use `slurm_backend="local"` to force packing or
`slurm_backend="slurm"` to force Slurm for every future launch on that node. Per-bucket
overrides are `NODES["node"]["slurm_gpu_backend"] = "slurm"` / `slurm_cpu_backend = "slurm"`.
Tasks with explicit Slurm fields route only to Slurm-capable nodes; scheduleurm will not
silently ignore those fields by launching them locally.
If that node should pack despite `nvidia-smi` showing 100% util, also set
`NODES["node"]["gpu_util_saturation_pct"] = None` so placement relies on
VRAM/1⁄3/RAM/CPU checks rather than util.

The class hierarchy:

- `Backend` (ABC) — `launch(task, node_state=None)` / `kill` / `batch_probe`
- `LocalBackend` — `ssh + nohup` path; calls `_ClaimManager.claim()` first when the node has `enable_claims=True` (Phase 3.2 / 3.4)
- `SlurmBackend` — `sbatch` / `scancel` / `squeue`
- `HybridBackend` — per-node routing; this is what `_BACKEND` actually is

**Phase 3.2 / 3.4 cross-scheduler claims (shipped)**:

When a node has `NODES["x"]["enable_claims"] = True`, all scheduleurm
instances on that machine (different state dirs / different OS users)
share `/tmp/scheduleurm/claims.json` mutated under `flock`. Each user
deploys their own `_claims_${USER}.py` (sticky `/tmp/scheduleurm` would
otherwise block cross-user overwrite); the shared `claims.json` and
`claims.lock` are mode 0666; the script writes IN-PLACE under flock
(no `os.rename` — that fails cross-user in sticky dirs).

`LocalBackend.launch` does an atomic capacity check before `ssh+nohup`.
The check enforces the same placement policy as local `_gpu_fits`:
total CPU/RAM/VRAM caps, per-task VRAM cap, VRAM margin, and the 1/3+grace
packing rule. Util saturation isn't replicated cross-scheduler (no shared
util reading). Nodes with `gpu_util_saturation_pct=None` skip that gate and
rely on VRAM/CPU/RAM; strict nodes still gate on util in `pick_placement`
before `claim` is invoked.

Failed claims split: capacity conflicts return `CLAIM_RACE:` (task back
to queue, no fail-count increment, retry next cycle) while transport /
setup errors return `CLAIM_ERROR:` (real launch failure, counts toward
`MAX_LAUNCH_RETRY` so a chronically-broken node escalates). Watcher
periodically `renew_many`s our live claims and `gc_stale`s any expired-
plus-dead ones from any scheduler. PIDs owned by other users count as
alive (`PermissionError` from `kill(pid,0)` means the proc exists).
`scheduler_id` is a persistent UUID stored in `STATE_DIR/claim_owner_id`
so a scheduleurm restart still matches its own pre-restart claims.

`probe_all` subtracts pending (pre-PID) claims from the node's free
resources, so a competing scheduler's `pick_placement` sees the
launch-race window as occupied even before the host process shows up
in `ps`/`nvidia-smi`.

The same shared file also carries an `intents` queue for launch admission:
each claim attempt first registers a FIFO ticket, then may claim only if it
does not delay an older intent that could run now. Smaller/disjoint work can
still backfill behind an older task that is currently blocked by resources.
Intent tickets have their own short TTL (`claim_intent_ttl_s`, default 180s)
so a dead scheduler does not pin the queue. If an older intent waits past
`claim_fifo_strict_after_s` (default 1800s), younger work that would consume
that older task's future slot is blocked even when the older task cannot run
right now; disjoint work can still backfill.

The remote claim script also does a best-effort live resource recheck while
holding the claims lock (`claim_live_check=True` by default): manual/non-
scheduleurm CPU load, RAM pressure, and `nvidia-smi` GPU memory are treated as
synthetic external claims. This closes the probe→claim race with outside users;
if `/proc`/`nvidia-smi` is unavailable, it falls back to claims-only behavior.
Use `scheduler claims [--node NODE]` to inspect both active claims and FIFO
intents; `status`, `why`, and `tui` surface intent counts/head tickets too.

Not a slurm replacement — there is no fairshare accounting, quota system, or
preemption across users. The claims layer provides atomic over-commit
prevention plus FIFO-with-backfill launch admission. If you need the rest of
slurm's feature set, install slurm and opt that node into
`slurm_backend="slurm"`; the layers compose (a node can have slurm installed
AND `enable_claims` for scheduleurm-local launches that bypass slurm's queue).

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
                     │              - eviction (mem ≥ 1/3+grace)          │
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
1. Same run identity cannot be in two `running`/`launching` slots simultaneously; broad signatures can run in parallel when `cmd`/`cwd`/env/result identity differs; the same `--ckpt-dir` cannot be shared by active tasks unless explicitly overridden.
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
| "run this" / "submit this" / "launch this" | `submit` + `dispatch`; report node:GPU + log path + resume_from |
| "run these N seeds" / "launch this sweep" / "kick off N seeds" | submit all N with same `--signature` prefix; ONE `dispatch` |
| "GPU free?" / "status" / "what's running?" / "node status" / "show queue" | `status`; highlight GPUs under the 1/3+grace freeze line |
| "rebalance" / "redispatch" / "reassign" | `dispatch` (watcher does this automatically every 60s) |
| "cancel t0042" / "kill t0042" / "stop t0042" | queued: `cancel`. running: confirm + `cancel --force` |
| "clear queue" / "wipe the queue" | `clear-queue` (dry-run) → `clear-queue --confirm` (running tasks NEVER touched) |
| "show t0042" / "details on t0042" | `show t0042` |
| "wake me when done" / "notify me when finished" | wrap `wait-for --signature 'X/*'` in `Bash run_in_background` |
| "adopt the one running outside" / "take over an external job" | watcher auto-adopts every 60s; manual `adopt` for edge cases |
| "show resource history" / "show usage profile" | `history` |

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
│       ├── scheduler_mcp.py    # MCP server wrapper (9 tools)
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
