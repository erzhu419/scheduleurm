# Architecture

## Why a single Python file

The whole scheduler is one ~3700-line `scheduler.py` plus a few helpers (`env_deploy.py`, `tui.py`, `scheduler_mcp.py`). No framework, no DB, no broker, no task DAG runner. The reasons:

1. **State must survive any failure mode.** A K8s/Ray/Airflow-style scheduler with multiple coordinated processes has too many places to lose state during a crash. One process holding one fcntl lock writing one atomic JSON file is provably correct.
2. **Recovery has to be fast.** When the watcher restarts (power loss, host reboot, manual restart after a code edit), the whole state is one `json.load` away. No catalog rebuild, no sync.
3. **Debugging by reading.** When something looks wrong, you read `scheduler.py` top-to-bottom and find every code path. No "the framework decided to retry this for you".

Single-file constraint forces every feature to justify its complexity — features that don't earn their lines get cut.

## Lifecycle of a task

```
submit  →  queued  →  launching  →  running  →  done
                       (WAL)         │           │
                                     ├→ failed → (auto-retry up to 3x)
                                     │           │
                                     │           └→ escalated (heal session)
                                     ├→ cancelled
                                     └→ evicted → (re-queued with peak in history)
```

- **queued**: in `queue.json`, awaiting `dispatch`. Resource estimates filled at submit.
- **launching**: WAL state. Scheduler decided placement, persisted state with `status="launching"`, but `ssh + nohup` hasn't returned. If the scheduler dies here, watcher startup recovery reverts stale launching to queued. Closes the orphan window.
- **running**: ssh launch succeeded, `remote_pids` populated, watcher polls every 60s.
- **done**: all PIDs dead, success marker found in log tail, peak metrics folded into history.
- **failed**: PIDs dead, failure pattern detected in log tail. Routed to `_classify_failure` → `OOM` / `ENV_MISSING` / `DISK_FULL` etc.; some classes auto-retry, some escalate.
- **cancelled**: user-initiated kill; never auto-retried.
- **evicted**: post-dispatch eviction killed it because GPU mem ≥ 1/3 AND util ≥ 90% (real contention). Peak gets recorded; task re-queues with a more accurate estimate.

## State machine guarantees

| Invariant | Why | How enforced |
|---|---|---|
| Same signature never in two `running` slots | Two procs writing same `--out_dir` corrupts ckpts | `running_sigs` set built before each dispatch pass; `launching` included so cross-cycle WAL window is covered |
| State writes are atomic | SIGKILL during write must not corrupt `queue.json` | tmp file + fsync + `os.replace` in `save_state` |
| Killed tasks always requeue if checkpoint exists | Otherwise N hours of training lost on each kill | `_requeue_after_crash` checks `find_resume()`; only escalates if no resume path |
| Watcher restart recovers all state | Power loss / manual restart shouldn't lose tasks | Recovery scan at startup: stale `launching` → queued; running PIDs probed for liveness |
| `peak_ram_mb` only goes up | Tracking peak, not current | `max(task["peak_ram_mb"], current)` per probe |
| OOM never auto-retries | Throwing more compute at a real OOM = waste | `_classify_failure` → escalate, not retry |

## The dispatch loop (one tick)

```
1. Acquire state lock (fcntl LOCK_EX on .lock)
2. Load state, history
3. Refresh resource estimates for queued tasks (cascade lookup)
4. Build running_sigs (signatures of all running + launching tasks)
5. For each queued task in priority order:
     a. Skip if signature in running_sigs (race-guard)
     b. Skip if cross-sig --ckpt-dir conflict with running task
     c. Skip if all node candidates fail probe (CPU/RAM/VRAM/cap/util/1/3-rule)
     d. pick_placement(): best-fit (smallest leftover) + warm-first scoring
     e. Set status="launching", save_state immediately (WAL)
     f. Release lock briefly (push docker image / rsync conda env if needed)
     g. ssh + nohup the cmd; capture remote PID list
     h. Re-acquire lock; set status="running", remote_pids=[...], save_state
6. Release lock
```

The "release lock for image push" step matters: image push can take minutes, and the watcher must not hold the lock that long (other tasks would all back up). The launching WAL state lets us release the lock safely — if anything goes wrong during the push, recovery sees a stale launching record and reverts it.

## The watch loop (one 60s cycle)

```
1. dispatch()                              # try to fill queue → running
2. check_running()                         # poll PIDs + nvidia-smi; update peaks
3. for each task that just transitioned to terminal:
     diagnose_terminal()                   # 4 rules: success-marker / OOM / mid-train-kill / startup-crash
     if failed: _classify_failure()        # ENV_MISSING / OOM / DISK_FULL / APP_BUG / UNKNOWN
     if escalation-class: append to escalations.jsonl + fire heal session
     elif retry-class: _requeue_after_crash() (caps at MAX_AUTO_RETRY)
     fold peak into history
4. enforce_post_dispatch_thresholds()      # eviction check (mem ≥1/3 AND util ≥90%)
5. auto_adopt_external()                   # nvidia-smi compute-apps + cgroup CPU procs not in queue
6. archive terminal tasks > 7 days old
7. rotate watcher.log if > 5MB
```

Cycle time at steady state: ~1-3s (most time spent in `nvidia-smi` over SSH for remotes). Lock is held for the dispatch and check_running phases; released during ssh probes and heal-fire.

## Failure classification (the key safety mechanism)

`_classify_failure` decides retry vs escalate:

| Pattern in tail | Category | Action |
|---|---|---|
| `No space left on device`, `[Errno 28]`, `ENOSPC`, `disk full`, `Disk quota exceeded` | DISK_FULL | escalate (no retry — disk won't free itself) |
| `CUDA out of memory`, `out of memory`, `MemoryError`, `Killed process`, `oom-kill`, `oom_reaper` | OOM | escalate (raising vram/ram requires user thought) |
| `没有那个文件或目录`, `no such file or directory`, `command not found` | ENV_MISSING | escalate (need to deploy env) |
| `ModuleNotFoundError`, `ImportError` | PYTHON_IMPORT | escalate (need to install pkg) |
| `Traceback` + lifetime > 60s | APP_BUG | retry up to MAX_AUTO_RETRY (might be transient) |
| anything else | UNKNOWN | retry up to MAX_AUTO_RETRY |

The `Killed process` (kernel OOM-killer line, NOT bare `Killed`) is critical. The previous version used bare `"Killed"` which matched our own diag text "task killed mid-training" → false-classified as OOM → escalation skipped retry → 50h compute lost. Now the patterns are tightened to actual kernel log strings only.

DISK_FULL is checked **before** OOM because some `OSError: [Errno 28]` messages could plausibly contain "memory" elsewhere in the tail. Order matters.

## Eviction (the post-dispatch safety valve)

After dispatch, for each GPU we check:
- `mem_used / mem_total >= 1/3` — VRAM pressure exists
- `util >= 90%` — compute is saturated

**Both** must be true to evict. (Either alone is fine: 100% util on 20% mem = healthy single big task; 50% mem at 30% util = packed but not contended.)

The youngest task on the GPU is evicted (re-queued with current peak folded into history). Tasks within their warmup window (180s since launch) are protected — gives them a chance to allocate their full footprint before being measured against the 1/3 rule.

Single-task-on-GPU is a design exception: never evict a task that's the only user of its GPU, even at 100%/100%. That's not contention, that's correct utilization.

## Auto-adoption (external proc tracking)

Every 60s, watcher enumerates:
1. `nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory` — GPU compute apps
2. CPU procs under `/home/$USER/<project>/` with CPU% ≥ 50% — long-running CPU work

Procs not associated with a tracked task get adopted: a synthetic task record is added with `signature=<project>/auto-adopted/p<pid>`, `node=<detected>`, `gpu_idx=<from nvidia-smi>`, `remote_pids=[pid]`. From that point, the proc is tracked normally — peak metrics accumulate, terminal detection works, history gets folded.

This lets you `nohup` a script the old way and still see it in `status`. Over time, the auto-adopted history teaches the scheduler how big these jobs are, so when you re-launch through `submit` later the estimates are reasonable.

## What it deliberately doesn't do

- **Multi-user fairness** — single-user only. No queue priorities by user, no quota.
- **Cluster-wide DAG** — task A → task B chaining isn't here. Use bash with `wait-for`, or chain in your launcher script.
- **Hot-reload of NODES** — edit `scheduler.py`, re-run `install.sh`, watcher restart picks it up. No live config reload (would complicate the lock semantics).
- **Cross-node tensor sharing** — every task is a single-node operation. Multi-node training is the user's responsibility (NCCL/MPI/torchrun).
- **Auto-scale / spawn nodes** — `NODES` is fixed. If you need elasticity, add nodes statically and let `dispatch` ignore offline ones (probe failure = skip).
- **Replace SLURM** — if you have a real HPC cluster, use SLURM. This is for the messy single-user case where SLURM is overkill.
