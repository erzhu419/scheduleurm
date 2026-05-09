# Configuration

All scheduler-wide knobs live at the top of `skill/scheduler.py`. Re-run `install.sh` after editing — the script copies your edits into `~/.claude/skills/scheduler/` and restarts the systemd unit.

## NODES — your cluster

```python
NODES = {
    "local":     {"host": None,       "cpu_cores": 12, "ram_mb": 56*1024,  "ram_headroom_mb": 2048, "ram_headroom_frac": 0.20, "max_vram_per_task": None, "max_concurrent_running": 10},
    "remote-A":  {"host": "remote-A", "cpu_cores": 12, "ram_mb": 200*1024, "ram_headroom_frac": 0.10, "max_vram_per_task": None, "max_concurrent_running": None},
}
```

| Field | Meaning | Tuning notes |
|---|---|---|
| `host` | `None` for local; SSH alias from `~/.ssh/config` for remote | Must be passwordless: `ssh -o BatchMode=yes <host> true` exits 0 |
| `cpu_cores` | Schedulable CPU budget (already net of OS reservation) | 16 physical cores → 12 schedulable (4 reserved). HT logical count is misleading — use physical |
| `ram_mb` | Schedulable RAM in MB | Don't include OS headroom here — that's `ram_headroom_mb` / `ram_headroom_frac`'s job |
| `ram_headroom_mb` | Fixed RAM MB to keep unallocated | Wins over `ram_headroom_frac`. Current local WSL setting: `2048` |
| `ram_headroom_frac` | Fraction of RAM to keep unallocated | Used when `ram_headroom_mb` is unset. Bare-metal Linux / remotes: `0.10` |
| `max_vram_per_task` | Cap individual task VRAM | `None` auto-derives from probed `nvidia-smi total_mb`. Set a number to cap (WSL local 4060: 4096 lets two tasks share an 8GB card) |
| `max_concurrent_running` | Hard cap on tracked running tasks | Defense-in-depth above CPU/RAM bookkeeping. WSL local: 8-10. Remote with 200GB RAM: `None` |
| `slurm_backend` | `"local"` default, `"slurm"`, or `"auto"` | Default ignores Slurm even if installed. `"slurm"` forces all future launches on that node through SlurmBackend. `"auto"` preserves old detect-and-use behavior |
| `slurm_gpu_backend` / `slurm_cpu_backend` | `"local"` default, `"slurm"`, or `"auto"` | Per-resource opt-in. Use `"slurm"` only for real shared clusters; leave default for small nodes that need scheduleurm VRAM/RAM packing |
| `gpu_util_saturation_pct` | Integer percent or `None` | Per-node override for the GPU util placement gate. `None` ignores util for packing and relies on VRAM/1⁄3/RAM/CPU checks |
| `enable_claims` | `False` by default | Recommended for default-local shared nodes; adds cross-scheduler local resource claims and FIFO-with-backfill launch admission |
| `claim_ttl_s` | `3600` default | Lifetime of an active resource claim before GC may remove it if its PID is also dead |
| `claim_intent_ttl_s` | `180` default | Lifetime of a queued FIFO intent ticket. Keep above watcher interval; lower means faster recovery from dead schedulers |
| `claim_fifo_strict_after_s` | `1800` default | Aging guard for shared intents. After this wait, younger work that would consume the older task's future slot is blocked; disjoint backfill remains allowed |
| `claim_live_check` | `True` default | While holding the remote claims lock, treat live non-scheduleurm CPU/RAM/GPU usage as synthetic external claims. Set `False` if `nvidia-smi`/`/proc` is slow or misleading |
| `claim_live_check_timeout_s` | `3` default | Per-claim timeout for the remote live `nvidia-smi` check |

## Resource defaults (per task, when no history exists)

```python
DEFAULT_VRAM_MB = 512    # was 4096 → 1024 → 512; optimistic-low by design
DEFAULT_RAM_MB  = 4096
DEFAULT_CPU_CORES = 1
```

The default VRAM is deliberately **low**. If a task actually needs more, the post-dispatch eviction mechanism kills the youngest task on that GPU and re-queues it with the observed peak folded into history. Better to find out by trying than to lock a small task out of the 1/3 packing rule for hours.

## Packing rules

```python
VRAM_MARGIN_MB = 500              # headroom on a GPU after placing a task
ONE_THIRD_PACK_RULE = True        # don't add to a GPU already past 1/3 used (RL plateau heuristic)
GPU_UTIL_SATURATION_PCT = 85      # if an occupied GPU is past this util%, don't pack more
```

The 1/3 rule is specific to RL workloads where peak VRAM is hit at random plateaus during training, not at startup. Pure supervised training — set `ONE_THIRD_PACK_RULE = False` and the placer fills based on margin only.

## History accumulation

```python
HISTORY_MAX_ENTRIES   = 500   # cap on total signatures tracked (LRU eviction by last_seen)
HISTORY_SAMPLES_PER_SIG = 10  # rolling window per signature
HISTORY_PERCENTILE    = 80    # p80 of last 10 samples → estimate
```

p80 was chosen empirically: lower (e.g. p50) under-allocates and triggers eviction churn; higher (p95) reproduces the "single outlier pins everything" bug p80 was meant to fix.

## Failure handling

```python
MAX_AUTO_RETRY      = 3   # auto-requeue cap after crash
MAX_LAUNCH_RETRY    = 3   # cap on launch failures (cwd missing, ssh timeout, etc.)
LAUNCHING_RESET_S   = 60  # WAL launch marker age before reverting to queued
```

Failure categories that **never** auto-retry (escalate to heal session instead): `ENV_MISSING`, `PYTHON_IMPORT`, `OOM`, `DISK_FULL`. These are not transient — retrying just burns more compute. Other categories (`APP_BUG`, `UNKNOWN`) retry up to `MAX_AUTO_RETRY`.

## Environment variables

| Var | Effect | Default |
|---|---|---|
| `CLAUDE_BIN` | Path to `claude` CLI for the heal-session spawn path | `/home/erzhu419/.nvm/.../bin/claude` (you'll want to override) |
| `SCHEDULEURM_SKILL_DIR` | Where `install.sh` puts the skill | `~/.claude/skills/scheduler` |

The heal-session spawn requires Claude Code installed and `/scheduler-heal` skill registered. If you're not using Claude Code at all, the heal path is a no-op (failures escalate to log-only).

## Optional: per-cluster nodes file

If you don't want to edit `scheduler.py` directly (e.g. shared deployment), you can switch `NODES` to load from `~/.claude/scheduler/nodes.json` by adding at the top of `scheduler.py`:

```python
import json
_NODES_FILE = STATE_DIR / "nodes.json"
if _NODES_FILE.exists():
    NODES = json.loads(_NODES_FILE.read_text())
```

Sample `nodes.json`:
```json
{
  "local":    {"host": null,       "cpu_cores": 12, "ram_mb": 57344, "ram_headroom_mb": 2048, "ram_headroom_frac": 0.20, "max_vram_per_task": null, "max_concurrent_running": 10},
  "remote-A": {"host": "remote-A", "cpu_cores": 12, "ram_mb": 204800, "ram_headroom_frac": 0.10, "max_vram_per_task": null, "max_concurrent_running": null}
}
```

This isn't wired in by default to keep the single-file deployment story clean — but it's two lines if you want it.

## State files (read-only context — don't hand-edit while watcher runs)

```
~/.claude/scheduler/
├── queue.json              # all tasks (queued + launching + running + recent terminal)
├── vram_history.json       # per-signature p80 samples
├── runtime_history.json    # exact cmd/cwd/env runtime samples for Slurm walltime/ETA
├── escalations.jsonl       # heal session inbox (append-only)
├── queue_archive.jsonl     # terminal tasks > 7 days old
├── .lock                   # fcntl exclusive lock; held during all state mutations
├── .heal_fire.lock         # debounce marker for heal session spawning
└── logs/
    ├── watcher.log         # JSONL events; rotated at 5MB, kept 3 generations
    └── heal_fires.log      # heal session spawn audit trail
```

If you must edit `queue.json` (e.g. to fix an inflated estimate without re-running), do it under the lock:

```python
import fcntl, json, tempfile, os
LOCK = os.path.expanduser("~/.claude/scheduler/.lock")
QFILE = os.path.expanduser("~/.claude/scheduler/queue.json")
with open(LOCK, "a+") as lf:
    fcntl.flock(lf, fcntl.LOCK_EX)
    q = json.load(open(QFILE))
    # ... mutate q ...
    fd, tmp = tempfile.mkstemp(prefix="queue.", dir=os.path.dirname(QFILE))
    with os.fdopen(fd, "w") as f:
        json.dump(q, f, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, QFILE)
    fcntl.flock(lf, fcntl.LOCK_UN)
```

The lock pattern is non-negotiable — half-written `queue.json` files are recoverable from `queue.json.corrupt-<ts>` backups, but it's still ugly.
