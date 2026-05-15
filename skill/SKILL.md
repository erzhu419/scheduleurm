---
name: scheduler
description: Multi-resource (CPU + RAM + VRAM) scheduler across local 4060 + jtl110gpu (2x 3080Ti) + jtl110gpu2 (2x 3080Ti) + jtl110cpu/jtl110cpu2 (Windows CPU-only 128 physical cores each). Use whenever the user wants to launch ANY computation that consumes meaningful resources — GPU training, CPU-only training, data preprocessing, batch evaluation. Trigger phrases (English / 中文 — both fire equally, examples not exhaustive) — RUN A JOB / SUBMIT A JOB / LAUNCH A JOB / TRAIN / EVAL / INFERENCE / RUN THIS SCRIPT / RUN THIS PYTHON / RUN THIS EVAL / RUN A SWEEP / DATA PREP / N_WORKERS / --device cpu / multi-worker / multi-seed / "kick off X" / "fire off X" / "queue up X" / "schedule X" / "dispatch X" / "deploy X" / "send to GPU" / "put on jtl110gpu" / 跑训练 / 跑评估 / 跑推理 / 跑这个脚本 / 跑这个 python / 跑这个评估 / 跑 X / 提交任务 / 派活 / 派任务 / 部署 / 在 GPU 上跑 / 在 jtl110gpu 跑. STATUS QUERIES — "GPU free?" / "any free RAM?" / "what's running?" / "which node has room?" / "node status" / "show queue" / "show jobs" / "show tasks" / "how many tasks running" / GPU 还空吗 / 显存还够吗 / 哪个节点空 / 现在跑啥呢 / 节点状态 / 看看队列 / 看看任务. JOB CONTROL — "cancel job" / "kill job" / "stop job" / "clear queue" / "forget X" / "rebalance" / "redispatch" / "reassign" / 取消任务 / 杀掉任务 / 停止任务 / 清空队列 / 重新分配 / 重新派发 — also fire whenever a node frees up and queued work should be re-routed. CPU-only tasks must use --vram 0. Handles 1/3-VRAM packing rule, CPU/RAM constraints, WSL OOM defense on local, git-sync precheck, checkpoint resume, Windows CPU-node auto pinning, and auto-discovery of externally-launched tasks (GPU and Linux CPU).
argument-hint: "[submit | submit-cpu-batch | cpu-plan | dispatch | status | doctor | profile-local | results | cancel | forget | clear-queue | show | history | adopt]"
allowed-tools: Bash(*), Read
---

# scheduler — multi-resource job dispatch

You are the front-end for `~/.claude/skills/scheduler/scheduler.py`. The user does NOT want to remember its argv — they tell you what to run, you translate. **You also make judgment calls about how aggressively to pack work**; the scheduler computes "what fits", but YOU decide what to actually launch.

## Pre-flight: handle heal-blockers FIRST (you, not the user)

**Before answering any scheduler request** (submit / dispatch / status / cancel / etc.), do TWO checks. **Critical principle: heal sessions hand problems UP to YOU, not directly to the user. You are the first responder. Only escalate to the user when you've genuinely tried and are stuck.**

### Check 1: heal session left a blocker for YOU to handle

```bash
test -s ~/.claude/scheduler/HEAL_NEEDS_CLAUDE.md && \
  echo "=== HEAL NEEDS CLAUDE (parent claude inbox) ===" && cat ~/.claude/scheduler/HEAL_NEEDS_CLAUDE.md
test -s ~/.claude/scheduler/HEAL_NEEDS_USER.md && \
  echo "=== HEAL NEEDS USER (escalated past Claude) ===" && cat ~/.claude/scheduler/HEAL_NEEDS_USER.md
```

If `HEAL_NEEDS_CLAUDE.md` has content:
1. Read each entry. Each one says what failed, what the heal session tried, and "Action I'd take if I had broader context".
2. **Try to act on it yourself** using full conversation context + scheduler tooling: symlink envs, rsync code, edit queue.json (within state lock), run dispatch, submit a clone, etc. Cheap fixes (symlinks, missing path resolution, env-name mismatches) you should fix in seconds; for bigger work spawn an Agent.
3. **Mark resolved**: append a `resolved` line to `escalations.jsonl` for the relevant task_ids and DELETE the corresponding entry from `HEAL_NEEDS_CLAUDE.md`. If file is empty after deletion, remove the file.
4. **Reset HEAL_STATE if you cleared the inbox**: if `HEAL_NEEDS_CLAUDE.md` is now gone AND HEAL_STATE.status == "awaiting_claude", set `HEAL_STATE.status = "completed"`, `HEAL_STATE.stuck_on = null`, refresh `timestamp`. Otherwise the next heal session sees a stale "awaiting_claude" with 0 pending and falls through Init's branches.
5. **Only if you genuinely cannot decide** (truly destructive, truly ambiguous, needs user's stratgic input): MOVE the entry from `HEAL_NEEDS_CLAUDE.md` to `HEAL_NEEDS_USER.md` and surface it to the user with a tight question (1-3 options).
6. Then continue with the user's original request.

If `HEAL_NEEDS_USER.md` has content (post-Claude escalation): surface to user before proceeding. Don't auto-fix; you've already tried.

### Check 2: pending escalations the watcher logged

```bash
test -s ~/.claude/scheduler/escalations.jsonl && \
  python3 -c "import json,sys; recs={}; [recs.update({(r:=json.loads(l)).get('task_id'):r}) for l in open('$HOME/.claude/scheduler/escalations.jsonl') if l.strip()]; pending=[r for r in recs.values() if r.get('status')=='pending']; print(f'PENDING_ESCALATIONS={len(pending)}'); [print(f'  {r[\"task_id\"]} {r[\"category\"]} on {r[\"node\"]}: {r[\"signature\"]}') for r in pending]"
```

If `PENDING_ESCALATIONS` > 0 → invoke `/scheduler-heal` FIRST (via the Skill tool), then continue with the user's request afterward. The user explicitly does NOT want to be reminded each time — heal runs autonomously whenever there's a pending escalation and the user touches the scheduler in any way. One exception: if the user is asking ONLY a read-only question like `status`, you can answer first AND mention the pending escalations + auto-trigger `/scheduler-heal` after answering. For any write-side operation (submit/dispatch/cancel), heal first.

If 0 pending and HEAL_NEEDS_USER.md empty/absent → proceed normally.

### Check 3: queue invariant doctor (automatic cleanup)

Run this read-only check before any scheduler request where placement/order/results matter:

```bash
python ~/.claude/skills/scheduler/scheduler.py doctor --json
```

If it reports only safe queued-task fixes and the user is asking for a write-side operation
(`submit`, `dispatch`, `rebalance`, adding evals, moving tasks), apply them before continuing:

```bash
python ~/.claude/skills/scheduler/scheduler.py doctor --fix --json
```

The doctor currently auto-repairs only narrow invariants that have already cost time:
- SimpleSAC eval tasks get `wait_for_files` gates when their checkpoint can be inferred, so eval cannot dispatch before train produces the model.
- SimpleSAC train tasks that produce checkpoints awaited by queued evals are promoted to `high`.
- SimpleSAC snapshot/per-policy data tasks are forced to `require_node=local` unless explicitly submitted with `--allow-remote-large-data`.
- Queued retries/preempted tasks with stale live `eta_seconds` / progress markers are scrubbed and reseeded from local-test/runtime history, so ETA load does not inherit a killed process's remaining-time estimate.

It never rewrites running tasks. If it reports a running-task warning, surface the warning and ask only if killing/requeuing is actually needed.

## Cluster (schedulable budgets, already net of OS reservation)

| node | cpu | ram | gpus | task vram cap | RAM headroom |
|---|---|---|---|---|---|
| `local` (WSL2) | 16 cores | 56 GB | 1× 4060 8GB | auto (= GPU total_mb) | **25%** (WSL OOM defense) |
| `jtl110gpu` | 12 cores | auto-probed | 2× 3080Ti 12GB | unlimited | 10% |
| `jtl110gpu2` | 12 cores | auto-probed | 2× 3080Ti 12GB | unlimited | 10% |
| `jtl110cpu` (Windows) | 128 phys (256 logical) | 512 GB | none — CPU-only | n/a (vram=0 required) | 10% |
| `jtl110cpu2` (Windows) | 128 phys (256 logical) | 512 GB | none — CPU-only | n/a (vram=0 required) | 10% |

Note: local has 16 physical cores / 32 logical threads; scheduleurm budgets the 16 physical cores by default. WSL OOM freezes the host — the fixed local RAM headroom is non-negotiable.

`jtl110cpu` and `jtl110cpu2` are schedulable by `scheduler.py` for CPU-only work
(`--vram 0`). They are Windows/OpenSSH nodes at
`tf290q6n.zjz-service.cn:22945` and `tf290q6n.zjz-service.cn:23565`,
workspace paths map from
`/home/erzhu419/mine_code/<project>/...` to `F:\<project>\...`, and Python runs via
`F:\python\python.exe`. The scheduler refuses GPU tasks there, probes RAM via
PowerShell/.NET and CPU load from Python process CPU-time deltas (with Windows counters only
as fallback), launches through a Windows wrapper, and records logs under
`F:\.scheduleurm\logs\`.

Windows CPU-node self-check when TUI/status looks wrong:
- First run `python ~/.claude/skills/scheduler/scheduler.py status`; both `jtl110cpu`
  and `jtl110cpu2` should appear as `cpu=X/128` with realistic load.
- If a node says `Permission denied`, it is an SSH key-auth problem. Install the current
  `~/.ssh/id_ed25519.pub` on that Windows host; **never** put the password in
  `scheduler.py`, queue records, or this skill.
- If a node shows `128/128` while Python work is known to be running, treat it as a
  Windows probe bug. The scheduler should use Python process CPU-time deltas, not only
  `Get-Counter`, because Windows performance counters can fail under non-admin SSH.

**Windows Processor Group pinning — applies to ANY Windows host with >64 logical CPUs**, not
just jtl110cpu. Windows uses 64-bit affinity masks → splits CPUs into Processor Groups of
≤64 logical each; a process is locked to one group at startup; mp.Pool inherits master's
group. Without explicit cross-group pinning, every worker fights over one group's 64 logical
CPUs (regardless of how many physical cores the machine has), running at ~50% per-worker.
For scheduler-launched Windows tasks, the wrapper periodically pins child worker processes
to unique physical cores across processor groups, so future high-parallel CPU eval jobs do
not need project-level edits for pinning. `scripts/jtl110cpu_pin_block.py` remains a manual
helper for processes launched outside scheduler.py.

CPU batch worker sizing is built into scheduler.py. For M independent CPU items, it probes
all CPU-labor nodes and uses **currently free physical cores**, not full node capacity. If
`jtl110cpu` has `n1/128` free and `jtl110cpu2` has `n2/128` free, the planner uses
`N=n1+n2`, splits M proportionally into `m1,m2`, then computes each shard independently:
`e_i = ceil(m_i / n_i)` waves and `workers_i = ceil(m_i / e_i)`. Example: 901 items on
one fully-free 128-physical-core node → 8 waves, 113 workers, final wave 110 items. Do not
hand-compute this in future turns; use:

```bash
python ~/.claude/skills/scheduler/scheduler.py cpu-plan --items 901
```

On Windows CPU nodes, a single Python multiprocessing process must stay below the Windows
wait-handle limit. Scheduler caps one process at 60 workers and automatically splits a
larger node shard into multiple same-node shards so total workers still match the formula as
closely as possible. When this creates multiple shards on the same node, include
`{shard_index}` or `{start}-{end}` in output filenames; otherwise concurrent shards can
write the same CSV.

When a logical item expands into multiple independent CPU units, make scheduler plan over
the expanded work-item count instead of the logical count. Checkpoint eval is the common
case: if there are 39 checkpoints and 10 episodes per checkpoint, use
`--items 39 --item-multiplier 10` so the planner sees 390 independent items. The eval
script should interpret `[start,end)` as flattened work-item indices, e.g.
`ckpt_idx = item_idx // item_multiplier` and `episode_idx = item_idx % item_multiplier`.
Do not size workers from checkpoint count alone when each checkpoint runs many episodes.
`submit-cpu-batch` shards are soft-preferred to their planned CPU node and constrained to
the CPU-labor node set, not hard-pinned. If one CPU node is down before launch or disappears
long enough during a running shard, the shard is eligible to reroute/requeue onto another
CPU-labor node, but only after the scheduler can prove the old launch is not still alive.
If the Windows/OpenSSH/proc probe is unknown, or a terminal retry parent still has live or
unknown launch artifacts for the same run identity, scheduler must fail closed and block the
duplicate dispatch rather than start the same shard twice. For Windows staging problems
caused by irrelevant external symlinks, pass
repeatable `--stage-exclude <relative-path-or-glob>` instead of hand-editing the project
tree. Only exclude data that the job definitely does not read; for SUMO evals, do not
exclude small runtime assets such as `env/sumo_env/initialize_obj`, because missing files
there produce fast all-error CSVs.

By default this plans across all CPU-labor nodes (`jtl110cpu,jtl110cpu2`) from live
`free_cpu` values, skipping full/down nodes. Add `--use-total-cores` only for offline
what-if planning. To submit a generic sharded CPU batch, use `submit-cpu-batch` with a
command template:

```bash
python ~/.claude/skills/scheduler/scheduler.py submit-cpu-batch \
  --items 901 \
  --item-multiplier 1 \
  --stage-exclude 'env/calibrated_env/_line_envs/*/data' \
  --stage-exclude 'data/datasets_v2' \
  --cmd-template 'python eval.py --start {start} --end {end} --workers {workers}' \
  --cwd /home/erzhu419/mine_code/PROJECT \
  --signature 'PROJECT/eval/{node}' \
  --description 'PROJECT eval {node} [{start},{end})'
```

Template placeholders include `{start}`, `{end}`, `{items}`, `{workers}`, `{node}`,
`{total_items}`, `{logical_items}`, `{item_multiplier}`, `{shard_index}`, and
`{num_shards}`. The scheduler also exports
`SCHEDULEURM_CPU_*` env vars for scripts that prefer reading shard/worker metadata from
the environment. Multi-node templates must include shard placeholders unless you pass
`--allow-env-only-shard`, which is only safe when the script reads those env vars.
When the user says "用 scheduler 派" for a large CPU-parallel/eval/checkpoint batch,
inspect the script or file list to infer M, then use `submit-cpu-batch` so all available
CPU-labor nodes are used automatically.

Background watcher (`scheduler.service` systemd user unit) runs `dispatch` every 60s and **auto-adopts** any externally-launched user-owned process (BOTH GPU compute apps AND CPU-burning python procs ≥50% CPU under `/home/erzhu419/<project>/`). Notifications go to `~/.claude/scheduler/logs/watcher.log` (and Feishu if `~/.claude/feishu.json` is configured).

## What the scheduler enforces (you don't need to re-check these)

- **VRAM 1/3+grace packing rule**: GPU already past the 1/3+grace freeze line will not accept more tasks (RL plateau heuristic). Single big task on an empty card is the exception.
- **Optional GPU compute-saturation guard (util ≥ 85%)**: strict nodes may block packing when an occupied GPU is compute-saturated. Local/jtl110gpu/jtl110gpu2 set `gpu_util_saturation_pct=None`, so RL packing is governed by VRAM/CPU/RAM rather than util.
- **CPU constraint**: total declared `cpu_cores` of tasks running on a node must not exceed budget. CPU-saturated node is auto-skipped — won't pile on.
- **RAM constraint**: free RAM minus task's request must remain above headroom (25% local, 10% remote). A positive `ram_mb` in `NODES` is an explicit cap; `ram_mb=0`/unset means use the probed `MemTotal`, which is the default for the GPU servers.
- **Per-task VRAM cap**: auto-derived from probed GPU `total_mb` (was hardcoded 4GB AMD-era cap; now respects whatever NVIDIA card nvidia-smi reports as GPU0). 1/3 packing rule still applies on top.
- **Auto-learn**: every task records peak VRAM + peak RAM under its `--signature`. Re-runs reuse history — no manual estimation after first run.

## Env delivery: how the task's runtime reaches the target node

Two strategies, picked at submit time via `--env-spec`:

| `--env-spec` | Effect | When to use |
|---|---|---|
| `none` (default) | cmd already references absolute conda python path; env must exist on target. Failure → `ENV_MISSING` escalation. | Back-compat path; dedicated nodes with pre-built conda envs. |
| `docker:IMAGE[:TAG]` | Launch wraps cmd in `docker run --gpus device=N --rm -v $cwd:$cwd -w $cwd --memory ${ram_mb}m --cpus ${cpu_cores} --name sched-tXXXX $image bash -c "<inner>"`. First time on a target, scheduler does `docker save \| ssh node docker load` to push image (one-time, ~minutes for ML). Image digest is compared on every preload — if local rebuilt, remote gets re-pushed. Requires user has docker daemon access on target (`docker info` works). | Heterogeneous nodes; env mismatches; "I don't want to install conda on every box." |
| `conda:/abs/path/to/env` | Pre-dispatch: scheduler rsync's local conda env to target at the SAME absolute path (idempotent — incremental sync after first push). Cmd uses absolute python path as-is; once env exists on target, cmd resolves naturally. **Requires the same absolute path layout across nodes** (e.g., all nodes use `/home/$USER/.conda/envs/*` — symlink if conda installed elsewhere). | Quick way to flip a multi-node experiment without docker; small env diffs across nodes. |
| `auto` | Probes target: if `docker info` works AND `--image` is set → docker. Else falls back to `none`. | Mixed cluster (some nodes have docker, others only conda). |

Pair with `--image MYPROJ:latest` when using `docker` or `auto`. Image must exist locally (`docker images`) so scheduler can save+ssh+load it.

**Doesn't auto-install docker** if the target lacks it — admin step done once per node. If sudo access is available on the target, `sudo apt install docker.io && sudo usermod -aG docker $USER && newgrp docker` (one-time).

## Submit-time guards (REFUSED by default unless overridden)

These fire at `submit` time and must be addressed by the submitter — you cannot dispatch around them. Each has an explicit override flag for legitimate edge cases.

| Guard | Triggers when | Override |
|---|---|---|
| **CPU-training refusal** | `_task_looks_like_training(cmd, desc)` AND `--vram 0` | `--allow-cpu-training` (REQUIRES `--cpu-training-justification "<≥30 char reason>"` — not a reflex bypass; default for training is GPU) |
| **Missing-ckpt refusal** | training-shaped cmd without `--ckpt-dir` | `--allow-no-ckpt` (debug runs / one-shot evals only) |
| **Resume-not-wired refusal** | training-shaped cmd, has `--ckpt-dir`, BUT cmd has no `--resume`-style flag AND submit didn't pass `--resume-flag` | Either add `--resume_from <path>` to cmd, pass `--resume-flag '--resume_from'` (scheduler appends on relaunch), or `--allow-no-resume` |
| **Cross-sig ckpt-dir conflict** | `--ckpt-dir` matches an active queued/running task with a DIFFERENT signature | Cancel/wait the existing one, OR use a different `--ckpt-dir`, OR `--allow-shared-ckpt-dir` (rare; concurrent writers will still corrupt unless coordinated) |

The justification + override flags are persisted on the task record so future-you can audit why a CPU/no-ckpt/no-resume task was permitted.

## Reboot recovery (automatic — but know the contract)

When the local box reboots:

1. **systemd auto-restarts the watcher** (`scheduler.service` is `enabled`).
2. On startup, watcher uses `/proc/uptime` to detect a recent reboot and fires a `post_reboot_triage` event in `~/.claude/scheduler/logs/watcher.log` listing `local_running_pre_reboot` task ids — explicit audit trail, no need to grep.
3. **First dispatch cycle (≤ 60s after watcher start):**
   - **Local-pinned tasks marked `running`**: stale PIDs detected via `kill -0`, `_diagnose_terminal()` runs over the log tail.
     - If log shows training markers + no success marker → flagged `is_crash=True` ("training markers present but no success marker after Xs") → `_requeue_after_crash` creates a fresh retry task with `parent_id` linkage (retries up to 3, then escalates).
     - Tasks with `--resume-flag '<flag>'` set at submit time: scheduler's `find_resume()` locates the latest ckpt under `--ckpt-dir` (filtered by extension whitelist `pt|pth|pkl|ckpt|bin|safetensors|npy|npz|h5|hdf5|tar`) and appends `<flag> <ckpt_path>` to the relaunch cmd.
     - Tasks with `--allow-no-resume`: relaunch from step 0.
   - **Remote-node tasks (jtl110gpu / jtl110gpu2)**: typically survive local reboot because they're launched via `ssh + setsid bash -c` so the SSH disconnect doesn't propagate SIGHUP to the remote process. Watcher reconnects via SSH, sees the remote PID still alive, leaves task in `running`. No action needed.
4. **Queued tasks**: untouched. Watcher resumes dispatching them as resources free.
5. **You don't need to manually trigger anything.** If the user asks "did everything recover?", check `post_reboot_triage` event in the watcher log + run `status` to see current task states. Look for `parent_id` chains — those are auto-requeues that fired during recovery.

**Footgun history (now fixed, in case you see legacy bugs in old logs):**
- ❌ Pre-fix: `_diagnose_terminal` on mid-training kill returned `"ambiguous; assumed normal"` → tasks falsely marked `done`, no auto-requeue, 50h of progress silently lost. Fixed by the "training markers + no success marker → crash" rule.
- ❌ Pre-fix: `find_resume()` glob `*` matched `train_log.csv` → injected as `--resume_from`, `torch.load(csv)` → `EOFError`. Fixed by extension whitelist.
- ❌ Pre-fix: clean no-op exits (eval `--skip_existing` finds nothing to do, exits in 25s) flagged as crash. Fixed by adding `"Running 0 checkpoints"`, `"Nothing to "`, `"no checkpoints to"` to `SUCCESS_PATTERNS`.
- ❌ Pre-fix: `_classify_failure` had bare `"Killed"` in `OOM_PATTERNS`, matching innocent English like `"task killed mid-training"` in our own diagnose reason → mid-training kills false-classified as OOM → `_requeue_after_crash` escalated instead of retried → 4 wsrl/s1024 tasks (50h compute) never re-queued. Fixed by tightening to kernel-format strings: `"Killed process"`, `"oom-kill"`, `"oom_reaper"`.
- ❌ Pre-fix: cmds without `-u` had python stdout fully buffered → SIGKILL'd processes left 0-byte logs → diagnose's `"log only 0B"` rule false-flagged completed training (AWAC s123/s789 saved final.pt, marked failed). Fixed by `_inject_python_u()` at launch time, idempotent.
- ❌ Pre-fix: cross-session submissions with DIFFERENT signatures but SAME `--ckpt-dir` produced concurrent writers to one path → corrupt ckpt (3 wsrl/s1024 procs, 14h lost). Fixed by submit-time ckpt-dir conflict guard (different sig + same active ckpt-dir → REFUSED).

## YOUR job — decision-making, not just plumbing

The scheduler reports "max-fill is K". **You decide whether to launch K, fewer, or stage them.**

1. **Don't ask "want X-way?"** — you have all the info. Ask only if there's genuine ambiguity (unusual resource demand, conflicting preferences).
2. **Concurrency = min(N, max-fit-K)** — for N=9 tasks with K=5, launch 5 now, queue 4. Don't pick 4 just because it divides evenly. The user's law: "5 路跑 2 批 比 4 路跑 3 批好" — **prefer fewer batches**.
3. **WSL local restraint**: even if `dispatch` says local fits, prefer remote when both fit. Local is for small/short jobs or fallback. Watch loadavg in `status` — if local already > 12 (out of 16 schedulable physical cores), don't add to it.
4. **CPU-saturated remote → route around it**: if jtl110gpu's CPU is already at limit and user wants to run a CPU-heavy thing there, **proactively suggest** pulling artifacts (ckpt, data) to local instead. Don't blindly try to dispatch and let it sit blocked. Don't wait for the user to figure it out.
5. **Submit + dispatch as one motion** — for a batch of similar tasks, submit them all, then ONE `dispatch`. Watcher picks up stragglers as resources free. Don't ask before each.
6. **Trust history**: if signature has been seen before, scheduler auto-fills cpu/ram/vram from peak history. Pass `--cpu N --ram-mb M --vram V` only when you have a specific reason.

## Task spec — extracting from user requests

When the user says "跑这 9 个 ablation":

| field | how to derive |
|---|---|
| `--cmd` | the actual shell command, no `cd`/`ssh`/`nohup` wrapper |
| `--cwd` | absolute path on target node (e.g. `/home/erzhu419/<repo>`) |
| `--signature` | stable id like `<Project>/<config>` — **same signature for re-runs** so history accumulates |
| `--description` | one-line human description |
| `--cpu N` | only if user-specified or non-default (history fills in) |
| `--ram-mb M` | same |
| `--vram V` | use `0` for CPU-only tasks; let history decide otherwise |
| `--git-repo` | for sync check (refuses launch if local dirty or remote out of sync) |
| `--ckpt-dir` | abs path on target for resume detection |
| `--resume-flag` | flag the script accepts (e.g. `--resume_from`); scheduler appends `<flag> <ckpt_path>` to cmd on re-dispatch when `find_resume()` locates a ckpt. Empty default = no injection. Pair with `--ckpt-dir`. Required for auto-resume to actually take effect. |
| `--result-dir` | **Phase 3.5: opt-in auto result pull-back.** Abs path on TARGET node containing the experiment results (logs / final models / metrics). On task completion (status=done), watcher pulls this dir to local once: rsync for Linux nodes, SSH+tar for Windows CPU nodes. Intermediate ckpts should stay in `--ckpt-dir` (NOT auto-synced; migration / eval pulls them on demand). Set this when you want results back without manual `scp`. |
| `--local-result-dir` | Optional override for where on local to land the rsync. Defaults to mirroring the target path. Use when local home differs, or you want to collect runs under one dir. |
| `--test-log` | Local preflight/test log containing tqdm/progress output. Use whenever the code was already run locally before submit; scheduler records runtime history immediately so ETA/walltime start from the local profile. |
| `--test-peak-vram-mb` / `--test-peak-ram-mb` / `--test-cpu` | Peak resource measurements from local preflight. Pass these when available so first dispatch is sized from measured data instead of broad project defaults. |
| `--priority` | `low/normal/high` — only if user signals urgency |
| `--preferred-node` | **soft pin** — try this node first; if it's full / throttled / down, scheduler picks any other fitting node automatically. Use this for "I'd prefer X" intent. |
| `--require-node` | **hard pin** — task ONLY runs on this node, never falls back. Use ONLY when the task has node-specific dependencies that genuinely can't be moved (libsumo only on local; non-portable C extension; node-local data files; in-place state at a specific path). When in doubt, use `--preferred-node` instead — it preserves the user's preference but lets the scheduler load-balance when one node is overloaded relative to another. |
| `--env KEY=VAL ...` | env vars (CUDA_VISIBLE_DEVICES is set automatically) |

### Pinning rule of thumb (very important — bad pin choices waste cluster capacity)

When the user says "跑在 X 上" / "用 jtl110gpu2 跑这个" / "在 local 跑":
- **Default to `--preferred-node`** unless one of these is true:
  - The task uses a library that's only on that node (e.g. SUMO/libsumo: only on `local`)
  - Resume-from-ckpt and the ckpt only exists on that node (and is large enough that copying it is meaningful)
  - User explicitly says "必须 / 一定要 / 只能 / strictly" / similar hard-pin language
- **`--require-node` is the hammer**, not the default. Hard pin means the task waits forever if the node is full, even when another node is idle — exactly the resource waste scheduleurm exists to prevent. With `--preferred-node`, a node tied up with long jobs lets new work flow to its less-loaded peer.

If you previously submitted a batch with `--require-node` and now realize they should have been soft-pinned, edit the queued tasks (e.g. via a small `python3 -c '...'` script under `state_lock`) to swap `require_node` → `preferred_node` and clear `node`/`gpu_idx` so dispatch re-decides.

### Pre-submit: ensure the training loop uses `tqdm` (Phase 3.0 ETA accuracy)

Phase 3.0's load-balanced migration makes per-node load decisions based on `eta_seconds`
of each in-flight task. The watcher computes `eta_seconds` by tailing the task's log;
the most accurate signal is **tqdm's own pre-computed `remaining` field** (e.g.
`[00:42<03:21, 12.34it/s]` — `03:21` is tqdm's smoothed-rate estimate of remaining time,
which adapts to warmup vs steady-state better than any rate computation we can do
externally).

When user says "跑这个 ..." / "submit this", **before** issuing the `submit` call:

1. **Checkpoint contract first for training.** Training code must save enough state to resume at the exact epoch/iteration: model params, optimizer state, replay/buffer state when present, total step/epoch/iteration counters, RNG state when stochasticity matters, and a load path that starts the next loop at `saved_iter + 1` or equivalent. If the scheduler refuses with a checkpoint-contract error, do not bypass with `--allow-no-resume` for real experiments. Use Codex/Claude to edit the training code, add full-state save/load, then submit again. `--allow-no-resume` is only for short debug runs where restart-from-zero is explicitly acceptable.
2. **Identify the training entry script** from the cmd (the `.py` file, after stripping
   `python -u -m foo.bar` / `python script.py` / `conda run ... python ...` etc).
3. **Read the script** and look at its top-level training/eval loops:
   - `for ... in range(n_iters):` / `for epoch in range(n_epochs):` / `for step in range(...):`
   - `while ... :` outer loop (if it's the main one)
4. **Check if `tqdm` wraps the iterable**. Look for `from tqdm import tqdm` / `from tqdm.auto import tqdm` AND a wrap site like `for x in tqdm(loop, total=N):` or `pbar = tqdm(total=N)`.
5. **If missing**, propose adding it via the Edit tool BEFORE running submit:
   - Add `from tqdm import tqdm` (or `from tqdm.auto import tqdm` if mixed env) to imports
   - Wrap the outermost training loop: `for i in tqdm(range(n_iters), desc="<task>"):` or
     equivalent. Preserve any existing per-iteration logging (so RE-SAC's `Iter N | Reward
     ...` print still happens, tqdm just adds the progress bar on top).
   - One sentence diff explanation; user reviews + approves.
6. **Skip auto-add** when:
   - Script already prints something parseable as `Iter N` AND has a `--max_iters N` flag
     in cmd (parser's tier-2 fallback handles this)
   - User explicitly says "no tqdm" / "I have my own progress" / etc.
   - Multi-process / distributed training — auto-adding tqdm to ranks > 0 spams stdout.
     Detect via `torch.distributed`, `mpi4py`, `accelerate.launch`, etc. Skip those.
7. **Run/ingest the local preflight profile before submit.** The intended path is:
   - The code is first run locally outside scheduleurm and writes tqdm/progress to a log.
   - Pass that log to submit via `--test-log /path/to/local_test.log`.
   - If you need scheduleurm to collect the profile directly, run `python ~/.claude/skills/scheduler/scheduler.py profile-local --cmd ... --cwd ... --signature ... --log-path ...`; this does not enqueue the job, it just records local peak RAM/VRAM/CPU and runtime history.
   - If the local test also measured peaks, pass `--test-peak-vram-mb`, `--test-peak-ram-mb`, and `--test-cpu`.
8. **After tqdm + local profile are in place**, proceed to `submit`. The watcher will pick up tqdm's
   `[elapsed<remaining, rate]` output and use the `remaining` directly. Submit-time `--test-log`
   gives the scheduler an ETA before the real experiment even launches.

If no local test log exists, inspect runtime history and choose the closest prior run by code path,
major flags, dataset/env, and loop length. If there is no defensible close match, treat ETA as
unknown rather than inventing one.

This is one-time per script — once tqdm is in, future submits are unchanged.

If you're unsure whether a loop is THE training loop or a sub-loop, ask the user
("which is the training loop you want me to wrap with tqdm?"). Don't guess and modify
random for-loops; that's worse than no progress bar.

Example (RE-SAC b1):
```bash
python ~/.claude/skills/scheduler/scheduler.py submit \
  --description "RE-SAC b1 multi-gpu queue" \
  --cwd "/home/erzhu419/RE-SAC" \
  --signature "RE-SAC/b1/per-job-2800" \
  --git-repo "/home/erzhu419/RE-SAC" \
  --cmd "/home/huiwei/anaconda3/bin/conda run --no-capture-output -n resac-jax python -u -m jax_experiments.multi_gpu_scheduler --queue b1 --per-job-vram 2800 --gpu-reserve 1500 --ram-reserve 4000 --cpu-reserve 2 --per-gpu-cap 2" \
  --env "PYTHONPATH=/home/erzhu419/RE-SAC" "XLA_FLAGS=--xla_gpu_enable_triton_gemm=false" "XLA_PYTHON_CLIENT_PREALLOCATE=false"

python ~/.claude/skills/scheduler/scheduler.py dispatch
```

## Common request → action

### "跑这个 / run this"
1. `submit` (with all fields above)
2. `dispatch`
3. Report node:GPU, log path, resume_from if found

### "跑这 N 个 / run these N tasks"
1. Submit all N (one per task, same `--signature` if same config)
2. ONE `dispatch` — greedy-fills capacity, queues the rest
3. Report how many launched / queued / why queued (which constraint blocked)

### "GPU 还空吗 / 现在跑啥呢 / status"
```bash
python ~/.claude/skills/scheduler/scheduler.py status
```
Show node telemetry (loadavg, free CPU, free RAM, per-GPU used/total) + tasks. Highlight GPUs under the 1/3+grace freeze line as "still acceptable".

### "队列是不是又乱了 / eval 有没有早跑 / 自动整改"
```bash
python ~/.claude/skills/scheduler/scheduler.py doctor
python ~/.claude/skills/scheduler/scheduler.py doctor --fix
```
Use dry-run first for read-only questions. Use `--fix` directly before dispatch/submit flows when the fixes are queued-task-only.

### "重新分配 / rebalance"
```bash
python ~/.claude/skills/scheduler/scheduler.py dispatch
```
Watcher does this every 60s automatically — only run manually if user wants it now.

### "取消 t0007 / kill 这个任务"
- Queued: instant `cancel t0007`
- Running: ALWAYS confirm with user first, then `cancel t0007 --force`

### "清空队列 / clear queue"
1. Dry-run: `clear-queue` (no flag) shows what would die
2. Confirm with user
3. `clear-queue --confirm` — running tasks NEVER touched

### "看看 t0007"
```bash
python ~/.claude/skills/scheduler/scheduler.py show t0007
```

### "跑完通知我 / wake me when done"

Scheduler tasks completing do NOT fire a Claude `task-notification` directly. To get woken up when a batch finishes, wrap `wait-for` in `Bash run_in_background` — the bash exit IS a notification:

```bash
# Wait for everything matching a signature glob
python ~/.claude/skills/scheduler/scheduler.py wait-for \
  --signature 'H2Oplus/*' --poll 30 --timeout 14400

# Wait for a specific list of task IDs
python ~/.claude/skills/scheduler/scheduler.py wait-for \
  --task-id t0099 t0100 t0101 --poll 30
```

Submit-launch-then-arm pattern (use this whenever you fire-and-forget a batch you'll want to follow up on):

1. `submit` all tasks (same `--signature` per logical batch)
2. `dispatch`
3. **In ONE `Bash run_in_background` call**: `python ~/.claude/skills/scheduler/scheduler.py wait-for --signature '<batch_glob>'`

When the bash exits (all tasks reached terminal state, or timeout), Claude gets a `task-notification` and resumes — closing the loop without `/loop` or cron.

Exit codes: `0` = all terminal, `1` = timeout with work still pending, `2` = no matches ever found.

### "外面那个跑的也加进来 / adopt"
Watcher already auto-adopts every 60s (both GPU and CPU). Manual adopt only needed for edge cases:
1. ssh + nvidia-smi compute-apps + `readlink /proc/<pid>/cwd` — group by project
2. `adopt --node X --gpu N --pid <p1> <p2> ... --signature <Project>/<config> --description "..."`
3. Script refuses multi-project bundles by default — split per project

### "撤销刚才那个 adopt / forget t0007"
```bash
python ~/.claude/skills/scheduler/scheduler.py forget t0007
```
Removes record only — never touches processes.

### "记一下显存历史 / 看看资源画像"
```bash
python ~/.claude/skills/scheduler/scheduler.py history
```
Auto-tracked: peak VRAM + peak RAM + cpu_cores per signature.

### "让 t0099 / 这几个先跑 / bump priority" (Phase 3.1)
```bash
python ~/.claude/skills/scheduler/scheduler.py priority t0099 high
```
Queued-only. Re-sorts the queue by `(priority, submitted_at)`; high-prio tasks dispatch first when GPUs free up. Do NOT use this on running tasks (rejected). For RUNNING tasks the answer is `cancel --force` + resubmit, OR ride out the natural completion order.

### "t0099 估算太高 / GPU 永远塞不下 / 改一下" (Phase 3.1)
```bash
python ~/.claude/skills/scheduler/scheduler.py edit t0099 --vram-mb 2000
# also supports --ram-mb, --cpu, --description, --preferred-node, --require-node
```
Queued-only. Use this when history-based estimate is poisoned by a single bad sample (typical sign: sibling signatures show 1-2GB but this task estimates >5GB). Pair with `history --drop <sig>` so the next run doesn't inherit the bad value again.

### "为什么 t0099 一直不上 / why" (Phase 3.1)
```bash
python ~/.claude/skills/scheduler/scheduler.py why t0099
```
Synthesizes:
- Task header (status / priority / preferred / est)
- `last_block_reason` from the most recent dispatch attempt
- Own + sibling history (so user can see if their est_vram_mb is an outlier)
- Per-node fit analysis: probes every node and explains FITS / blocked / 1/3-rule / util-saturation / RAM-headroom rejection per GPU

The first thing to run when a queued task seems stuck. Far better than parsing `last_block_reason` strings by hand.

### "和别的 scheduler / 别的用户共用节点 / 抢资源" (Phase 3.2)

非 slurm 节点上多个 scheduleurm 实例（不同 state dir、不同 OS user，或两者都不同）想共用同一台机器时，set `NODES["x"]["enable_claims"] = True` 在 scheduler.py 的 NODES 配置里。然后：

- `LocalBackend.launch` 在 `ssh+nohup` 之前会去 `/tmp/scheduleurm/claims.json`（节点本地）拿 flock，做 atomic CPU/RAM/VRAM capacity check。输了的 scheduler 收到 `CLAIM_RACE:` 信号，dispatch 把任务回到队列等下个 cycle，**不**计 `launch_fail_count`。
- `probe_all` 会把所有 pending claim（已 claim 但还没拿到 PID 的）扣进 free 资源里 —— 对方 scheduler 的 `pick_placement` 直接看到资源被占。
- watcher 每 cycle 一次 `renew_many + gc_stale`，崩溃的 scheduler 留下的 claim 会按 TTL（默认 1h）过期 + 死 PID 检测自动清理。

什么时候不开：单用户 / 单 scheduler 配置不需要，开了反而每次 launch 多一次 ssh + flock。所以是 per-node opt-in。

slurm 节点不需要开 —— slurm 自己有 gres + cgroup 处理这个。

### "history 里这个 sig 有个 9GB 的离群值 / 清掉 / 改" (Phase 3.1)
```bash
# Drop entire entry: next runs of this sig start fresh from real measurements
python ~/.claude/skills/scheduler/scheduler.py history --drop 'RE-SAC/b2/ns_tqc_Hopper-v2_16'

# Or set a specific value (resets vram_samples to single-element list):
python ~/.claude/skills/scheduler/scheduler.py history --set 'RE-SAC/b2/ns_tqc_Hopper-v2_16' --vram-mb 1500
```
Use when one bad run got recorded as the peak (e.g. crash-spike before OOM kill, runaway leak). Drop is preferred over set — let the next clean run establish a real peak.

## Hard constraints (don't violate)

- **NEVER ask "want X-way concurrency?"** — decide based on `status` + heuristics. Ask only if genuinely ambiguous.
- **NEVER overpack local** — even if scheduler says it fits, prefer remote when remote also fits. Watch local loadavg actively.
- **NEVER auto-`--force cancel`** — running tasks need user confirmation.
- **NEVER touch running tasks during dispatch / clear-queue / rebalance**.
- **NEVER re-implement** placement / fit / 1/3 / sync-check in your own bash — call the script.
- **NEVER fabricate node availability** — always probe via `status` or `dispatch`.
- **CPU-saturated remote → route around it**: if user says "run on jtl110gpu" but its CPU is at limit, proactively suggest local with artifact pull. Don't wait for them to ask.

## Service management

```bash
systemctl --user status  scheduler   # is the watcher up?
systemctl --user restart scheduler   # after editing scheduler.py
journalctl --user -u scheduler -f    # systemd's view
tail -f ~/.claude/scheduler/logs/watcher.log  # JSONL events
```
