#!/usr/bin/env python3
"""Multi-resource scheduler across local (4060 8GB / 16c / 64GB) + jtl110gpu (2x 3080Ti 12GB / 12c / 200GB) + jtl110gpu2 (same).

Resource model: each task declares cpu_cores, ram_mb, vram_mb (or vram=0 for CPU-only). Placement requires
ALL three to fit on the chosen node + GPU. Per-task resource needs are auto-learned from history (peak
VRAM and peak RAM, cores user-declared) so re-runs of the same signature use accurate budgets.

Subcommands:
  submit    Add a task to the queue (no launch yet).
  dispatch  Probe nodes, pick placements for queued tasks, launch what fits. Same call doubles as rebalance.
  status    Show node telemetry + task table. Updates running-task health and peak VRAM/RAM.
  cancel    Cancel a queued/launching task; with --force, kill a running one.
  forget    Drop a task record (never touches processes — for fixing wrong adopts).
  clear-queue   Cancel ALL queued tasks (running tasks untouched). Requires --confirm.
  record-vram   Manually record peak VRAM for a signature (auto-tracked too via status).
  history   Show recorded peak VRAM / RAM per signature.
  show      Print one task's full record (incl. log path, resume_from, etc.).
  adopt     Register externally-launched PIDs as a tracked task.
  watch     Background daemon: probe + dispatch every --interval s; auto-adopt external GPU tasks.

State lives in ~/.claude/scheduler/{queue.json, vram_history.json, logs/}. Inter-process safe via flock.
"""
import argparse
import fcntl
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

# Optional sibling module for docker / conda env deployment. Loaded lazily; if missing,
# all env-spec branches collapse to the legacy "none" path (assume conda env on target).
try:
    import importlib.util as _ilu
    _ed_spec = _ilu.spec_from_file_location("env_deploy", str(Path(__file__).parent / "env_deploy.py"))
    if _ed_spec and _ed_spec.loader:
        env_deploy = _ilu.module_from_spec(_ed_spec)
        _ed_spec.loader.exec_module(env_deploy)
    else:
        env_deploy = None
except Exception:
    env_deploy = None

# ---------- node inventory ----------
# cpu_cores / ram_mb are the SCHEDULABLE budget (already net of OS/IO reservation), not physical totals.
# Local is WSL2: 16 physical cores → 12 schedulable, 64GB RAM → 56GB schedulable, plus 25% headroom on top
# of that to defend against WSL OOM (which freezes the host). Remote nodes are dedicated boxes — looser.
# max_concurrent_running: hard cap on number of scheduler-tracked running tasks per node.
# Defense layer above CPU/RAM budgets — for SUMO/RL workloads, declared cpu/ram routinely
# under-counts (libsumo + multi-step env explodes during sim phase). On WSL local in particular
# 9+ tasks → CPU thrashing + OOM-killer fires on sibling processes. Set conservative on local;
# remote nodes have more headroom.
NODES = {
    # max_concurrent_running: loose cap, defense-in-depth above CPU/RAM. Real OOM protection
    # comes from the RSS upward-tracking in _batch_check_running; this cap only catches
    # runaway scenarios (e.g. dozens of tasks declared cpu=1 ram=500MB but each really uses 4GB).
    # local=10 covers the observed steady-state of 8-9 GPU+CPU tasks with one slot of headroom;
    # remote=None means no cap (200GB RAM, 12 cores schedulable, RAM headroom is the only bound).
    # max_vram_per_task: None = auto-derive from probed GPU total_mb at runtime (set in probe_node).
    # Was hardcoded 4096 from AMD 610 era; after NVIDIA 4060 (8188MB) became the default GPU,
    # the static cap silently blocked single-task allocations >4GB even though physically OK.
    "local":      {"host": None,         "cpu_cores": 12, "ram_mb": 56 * 1024,  "ram_headroom_frac": 0.20, "max_vram_per_task": None, "max_concurrent_running": 10},
    "jtl110gpu":  {"host": "jtl110gpu",  "cpu_cores": 12, "ram_mb": 200 * 1024, "ram_headroom_frac": 0.10, "max_vram_per_task": None, "max_concurrent_running": None},
    "jtl110gpu2": {"host": "jtl110gpu2", "cpu_cores": 12, "ram_mb": 200 * 1024, "ram_headroom_frac": 0.10, "max_vram_per_task": None, "max_concurrent_running": None},
}

STATE_DIR = Path.home() / ".claude" / "scheduler"
QUEUE_FILE = STATE_DIR / "queue.json"
VRAM_FILE = STATE_DIR / "vram_history.json"  # holds {sig: {vram, ram, cpu}} (back-compat: int = vram only)
LOCK_FILE = STATE_DIR / ".lock"
LOG_DIR = STATE_DIR / "logs"

VRAM_MARGIN_MB = 500       # headroom on a GPU after placing a task
RAM_HEADROOM_FRAC = 0.10   # keep 10% of node RAM unallocated as buffer for OS/other procs
ONE_THIRD_PACK_RULE = True # don't add to a GPU already past 1/3 used (RL plateau heuristic)
GPU_UTIL_SATURATION_PCT = 85  # if an occupied GPU is past this compute util, don't pack more (would just contend)
DEFAULT_VRAM_MB = 512      # est for unknown signatures (no history, no siblings).
                           # Optimistic-low by design: if a task actually needs more, the
                           # post-dispatch eviction mechanism (_enforce_post_dispatch_thresholds)
                           # kills the youngest task and re-queues it with the observed peak
                           # folded into history. Better to find out by trying than to keep
                           # a small task locked out of the 1/3 packing rule for hours.
                           # Was 4096 → 1024 → 512. User passes --vram N when known larger.
DEFAULT_RAM_MB = 4096      # ditto for RAM
DEFAULT_CPU_CORES = 1      # most ML jobs are single-process at the dispatch level (workers are forked)

ARCHIVE_FILE = STATE_DIR / "queue_archive.jsonl"
ARCHIVE_AGE_DAYS = 7        # terminal tasks older than this move from queue.json into archive
WATCHER_LOG_MAX_MB = 5      # rotate watcher.log when it exceeds this size
WATCHER_LOG_GENERATIONS = 3 # keep .log + .log.1 + .log.2 + .log.3 (oldest dropped on next rotation)
MAX_AUTO_RETRY = 3          # auto-requeue cap after crash (parent.retry_count + 1 > this → give up)
MAX_LAUNCH_RETRY = 3        # launch-failure cap (cwd missing, ssh timeout, etc.) before terminal failed + heal
LAUNCHING_RESET_S = 60      # stale WAL launch marker age before reverting to queued
ESCALATIONS_FILE = STATE_DIR / "escalations.jsonl"  # /scheduler-heal reads this; watcher appends

# Phase 2.16: cap on OUR slurm-managed tasks in PENDING state per node before scheduleurm
# holds the rest in its own queue. Default 1 — slurm gets at most one lookahead slot to
# bridge GPU swaps; the rest stay queued in scheduleurm and dispatch to whichever node
# frees up next. Avoids "all 10 tasks sbatch'd to node A's queue while node B sits idle"
# in a multi-slurm-node cluster.
#
# Tuning options (all need watcher restart to pick up new value):
#   1. Edit this constant in scheduler.py
#   2. Set env SCHEDULEURM_SLURM_MAX_PENDING_PER_NODE=N (recommended for no-code-edit override)
#   3. Per-node: NODES["nodename"]["max_slurm_pending"] = N (overrides this default for that node)
#
# Picking 0 means "never let scheduleurm have a pending task on any slurm node" — strict
# pull-on-demand. Risks GPU idle gaps during slurm's transitions but maximizes spread.
SLURM_MAX_PENDING_PER_NODE = int(os.environ.get("SCHEDULEURM_SLURM_MAX_PENDING_PER_NODE", "1"))

# Failure classification — drives whether to retry or escalate to /scheduler-heal.
ENV_MISSING_PATTERNS = ("没有那个文件或目录", "no such file or directory", "command not found", "未找到命令")
PYTHON_IMPORT_PATTERNS = ("ModuleNotFoundError", "ImportError")
DISK_FULL_PATTERNS = (
    "No space left on device",
    "[Errno 28]",       # OSError: [Errno 28] No space left on device
    "ENOSPC",
    "disk full",
    "Disk quota exceeded",
)
OOM_PATTERNS = (
    "CUDA out of memory",
    "out of memory",
    "MemoryError",
    "Killed process",     # kernel OOM-killer message format ("Killed process N (cmd) total-vm:...")
    "oom-kill",           # kernel oom-kill log line
    "oom_reaper",         # kernel oom-reaper log line
    # Was: bare "Killed" — but that substring matches innocent English like "task killed
    # mid-training" in our own diagnose reason text → false-classified mid-training kills as
    # OOM → _requeue_after_crash skipped → 4 wsrl/s1024 tasks (50h compute) never re-queued.
)

# ---------- state I/O with locking ----------
@contextmanager
def state_lock():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.touch()
    f = open(LOCK_FILE, "r+")
    fcntl.flock(f, fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()

def _load_json(p, default):
    """Load JSON or return default if file missing. RAISES on parse error — refusing to
    silently overwrite a corrupt state with `default`. If the file exists but is unreadable,
    we'd rather the caller see the exception (logged by watcher's iteration_error handler)
    than have save_state subsequently flush an empty `default` and lose everything."""
    if not p.exists(): return default
    text = p.read_text()
    if not text.strip():
        # zero-byte file: either an interrupted prior write or genuinely empty. Quarantine
        # for postmortem and treat as missing so watcher proceeds with `default`.
        try: p.rename(p.with_suffix(p.suffix + f".empty-{int(time.time())}"))
        except Exception: pass
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Quarantine the corrupt file so we don't keep raising; raise once so watcher logs it.
        try: p.rename(p.with_suffix(p.suffix + f".corrupt-{int(time.time())}"))
        except Exception: pass
        raise RuntimeError(f"corrupt JSON at {p} ({e}); quarantined; restart watcher to start fresh") from e

def _atomic_write_json(p, obj):
    """Write JSON via tmp + fsync + os.replace so a SIGKILL or power loss mid-write leaves
    EITHER the old file intact OR the new file fully on disk. Without this, naked write_text
    on a 1-2MB queue.json can produce a truncated/corrupt file → silent state loss."""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)

def load_state(): return _load_json(QUEUE_FILE, {"tasks": [], "next_id": 1})
def save_state(s): _atomic_write_json(QUEUE_FILE, s)
def load_history(): return _load_json(VRAM_FILE, {})
def save_history(h): _atomic_write_json(VRAM_FILE, h)

def archive_terminal_tasks(state, age_days=ARCHIVE_AGE_DAYS):
    """Move done/failed/cancelled/forgotten tasks older than age_days from queue.json into the archive
    JSONL. Caller must hold state_lock and is responsible for save_state(state) after this returns."""
    cutoff = time.time() - age_days * 86400
    keep, archived = [], []
    for t in state["tasks"]:
        if t.get("status") in ("done", "failed", "cancelled", "forgotten"):
            ts = t.get("finished_at") or t.get("submitted_at") or 0
            if ts and ts < cutoff:
                archived.append(t); continue
        keep.append(t)
    if archived:
        try:
            ARCHIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(ARCHIVE_FILE, "a") as f:
                for t in archived:
                    f.write(json.dumps(t, default=str) + "\n")
            state["tasks"] = keep
        except Exception:
            pass  # archive failure should never break the watcher; tasks just stay in queue.json
    return len(archived)

def maybe_rotate_log(path: Path, max_mb: int, generations: int):
    """If path is over max_mb, shift .log -> .log.1 -> .log.2 -> ... up to generations, dropping oldest."""
    try:
        if not path.exists() or path.stat().st_size < max_mb * 1024 * 1024:
            return
    except Exception:
        return
    base = str(path)
    for i in range(generations - 1, 0, -1):
        src, dst = Path(f"{base}.{i}"), Path(f"{base}.{i+1}")
        if src.exists():
            try: src.rename(dst)
            except Exception: pass
    try:
        path.rename(Path(f"{base}.1"))
    except Exception:
        pass

def append_heal_inbox(filepath, entry):
    """Append a Phase D entry to a heal inbox file (HEAL_NEEDS_CLAUDE.md / HEAL_NEEDS_USER.md).
    Rotates the file at 1MB to prevent unbounded growth if entries pile up unresolved.
    Used by scheduler-heal Phase D — keeps the heal SKILL one-liner instead of inlining rotation."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    maybe_rotate_log(p, max_mb=1, generations=3)
    with open(p, "a") as f:
        f.write(entry)

def history_get(sig):
    """Look up history for sig. Returns dict with optional vram_mb/ram_mb/cpu_cores keys, or None.
    Backward compat: legacy entries are bare ints (= vram_mb only)."""
    if not sig: return None
    raw = load_history().get(sig)
    if raw is None: return None
    if isinstance(raw, int): return {"vram_mb": raw}
    return raw

def _effective_est_vram(task, state, history):
    """Best-effort VRAM estimate for a task — used when own signature has no history yet.
    Priority cascade (most specific → most general):
      1. exact history[sig].vram_mb (prior runs of same task)
      2. peak_vram_mb of running/done siblings: same project + same description-first-phrase
         (e.g. "P2 retrain: Online SAC seed=42" → key "P2 retrain")
      3. history of sibling signatures sharing top-2 path prefix (e.g. offline-sumo/retrain-v2/*)
      4. ANY sibling under same project with peak_vram_mb > 100 (project-level fallback —
         catches the 'novel signature with no desc/prefix match' case where the cascade
         used to fall through to DEFAULT despite the project having usable real peaks)
      5. ANY history entry under same project with vram_mb > 0
      6. task's stored est_vram_mb (whatever submit chose)
      7. DEFAULT_VRAM_MB
    Returns int MB. Median of candidates protects against one-off outliers."""
    sig = task.get("signature") or ""
    h_self = history.get(sig)
    if isinstance(h_self, int): h_self = {"vram_mb": h_self}
    if isinstance(h_self, dict) and h_self.get("vram_mb"):
        return int(h_self["vram_mb"])

    project = task.get("project")
    desc_key = (task.get("description") or "").split(":")[0].strip().lower()
    candidates = []
    self_id = task.get("id")
    # Step 2: same project + same desc-key sibling peaks
    for t in state.get("tasks", []):
        if t.get("id") == self_id: continue
        if t.get("project") != project: continue
        td = (t.get("description") or "").split(":")[0].strip().lower()
        if desc_key and td == desc_key:
            peak = t.get("peak_vram_mb", 0)
            if peak > 100:
                candidates.append(int(peak))
    # Step 3: same top-2 prefix history
    if not candidates and sig:
        prefix = "/".join(sig.split("/")[:2])
        for sig2, hdata in history.items():
            if sig2 == sig: continue
            if "/".join(sig2.split("/")[:2]) != prefix: continue
            if isinstance(hdata, dict):
                v = hdata.get("vram_mb", 0)
                if v > 0: candidates.append(int(v))
            elif isinstance(hdata, int) and hdata > 0:
                candidates.append(int(hdata))
    # Step 4: project-level peak from queue (any sibling, any desc) — addresses the
    # 'novel sig, novel desc' gap where steps 2/3 are too strict to match.
    if not candidates and project:
        for t in state.get("tasks", []):
            if t.get("id") == self_id: continue
            if t.get("project") != project: continue
            peak = t.get("peak_vram_mb", 0)
            if peak > 100:
                candidates.append(int(peak))
    # Step 5: project-level history (any sig under same project)
    if not candidates and project:
        for sig2, hdata in history.items():
            if not sig2.startswith(project + "/"): continue
            if isinstance(hdata, dict):
                v = hdata.get("vram_mb", 0)
                if v > 0: candidates.append(int(v))
            elif isinstance(hdata, int) and hdata > 0:
                candidates.append(int(hdata))
    if candidates:
        candidates.sort()
        return candidates[len(candidates) // 2]  # median: robust to one-off outliers
    stored = task.get("est_vram_mb")
    if stored: return int(stored)
    return DEFAULT_VRAM_MB

def _effective_est_ram(task, state, history):
    """Best-effort RAM estimate mirroring _effective_est_vram. Used only to LOWER a queued
    task's stored/default RAM when it has no exact own RAM history yet. This prevents a stale
    high default from pinning a job in the queue forever even though sibling runs show it is
    small. Exact own history remains authoritative."""
    sig = task.get("signature") or ""
    h_self = history.get(sig)
    if isinstance(h_self, int):
        h_self = {}
    if isinstance(h_self, dict) and h_self.get("ram_mb"):
        return int(h_self["ram_mb"])

    project = task.get("project")
    desc_key = (task.get("description") or "").split(":")[0].strip().lower()
    candidates = []
    self_id = task.get("id")
    for t in state.get("tasks", []):
        if t.get("id") == self_id: continue
        if t.get("project") != project: continue
        td = (t.get("description") or "").split(":")[0].strip().lower()
        if desc_key and td == desc_key:
            peak = t.get("peak_ram_mb", 0)
            if peak > 100:
                candidates.append(int(peak))
    if not candidates and sig:
        prefix = "/".join(sig.split("/")[:2])
        for sig2, hdata in history.items():
            if sig2 == sig: continue
            if "/".join(sig2.split("/")[:2]) != prefix: continue
            if isinstance(hdata, dict):
                r = hdata.get("ram_mb", 0)
                if r > 0: candidates.append(int(r))
    if not candidates and project:
        for t in state.get("tasks", []):
            if t.get("id") == self_id: continue
            if t.get("project") != project: continue
            peak = t.get("peak_ram_mb", 0)
            if peak > 100:
                candidates.append(int(peak))
    if not candidates and project:
        for sig2, hdata in history.items():
            if not sig2.startswith(project + "/"): continue
            if isinstance(hdata, dict):
                r = hdata.get("ram_mb", 0)
                if r > 0: candidates.append(int(r))
    if candidates:
        candidates.sort()
        return candidates[len(candidates) // 2]
    stored = task.get("ram_mb")
    if stored: return int(stored)
    return DEFAULT_RAM_MB

HISTORY_MAX_ENTRIES = 500  # cap per item 25 (vram_history.json bloat)
HISTORY_SAMPLES_PER_SIG = 10
HISTORY_PERCENTILE = 80   # p80 of last N samples → estimate

def _percentile(samples, p):
    """p-th percentile (0..100) of samples list. Returns 0 for empty list."""
    if not samples: return 0
    s = sorted(samples)
    if len(s) == 1: return s[0]
    rank = (p / 100.0) * (len(s) - 1)
    lo = int(rank); hi = min(lo + 1, len(s) - 1)
    return int(s[lo] + (rank - lo) * (s[hi] - s[lo]))

def history_record(sig, peak_vram_mb=0, peak_ram_mb=0, cpu_cores=0, duration_s=0):
    """Fold new peak samples + EWMA duration into history for sig.

    SCORING POLICY (changed from max → p80 of last 10 samples):
    Old behavior: `cur["ram_mb"] = max(cur["ram_mb"], peak_ram_mb)` — a single anomalous
    peak (e.g. one bad run with full replay buffer + 10× ensemble = 5GB) pinned all future
    estimates at 5GB even though typical runs only use 1-2GB. Result: queued tasks with
    realistic 2GB needs were blocked because scheduler pretended they need 5GB → idle
    capacity, queue stalled.
    New behavior: keep last 10 samples per sig, use p80 as estimate. One outlier influences
    estimate <20% of the time; subsequent normal runs pull estimate back down. Cost: roughly
    20% of placements now risk peak > estimate → OOM-via-eviction recovery (already in
    place via _enforce_post_dispatch_thresholds), AND history capture path then folds the
    real peak in. Net: faster convergence to true workload size.

    Migration: if a record has the legacy `ram_mb` but no `ram_samples`, the existing
    value is treated as one sample seeding the new array. Subsequent records blend with it.

    Duration EWMA, LRU truncation, legacy int auto-migration unchanged."""
    if not sig: return
    h = load_history()
    cur = h.get(sig)
    if isinstance(cur, int): cur = {"vram_mb": cur}
    elif cur is None: cur = {}

    def _fold(field_name, samples_field, new_sample):
        """Append new_sample to samples list (capped at HISTORY_SAMPLES_PER_SIG), set
        field_name to p80 of the list. Migrates legacy field_name (no samples list) by
        seeding with the existing value so we don't lose all prior info."""
        samples = cur.get(samples_field) or []
        # Migration: if legacy `field_name` exists but no samples list, seed with it so
        # the first new sample blends with the legacy value rather than dropping it.
        if not samples and cur.get(field_name):
            samples = [int(cur[field_name])]
        samples.append(int(new_sample))
        samples = samples[-HISTORY_SAMPLES_PER_SIG:]
        cur[samples_field] = samples
        cur[field_name] = _percentile(samples, HISTORY_PERCENTILE)

    if peak_vram_mb > 0: _fold("vram_mb", "vram_samples", peak_vram_mb)
    if peak_ram_mb > 0:  _fold("ram_mb",  "ram_samples",  peak_ram_mb)
    if cpu_cores > 0:    cur["cpu_cores"] = max(cur.get("cpu_cores", 0), cpu_cores)
    if duration_s > 0:
        prev = cur.get("dur_s_ewma", 0)
        cur["dur_s_ewma"] = int(0.7 * prev + 0.3 * duration_s) if prev else int(duration_s)
        cur["dur_s_runs"] = cur.get("dur_s_runs", 0) + 1
    cur["last_seen"] = int(time.time())
    h[sig] = cur
    if len(h) > HISTORY_MAX_ENTRIES:
        # Sort by last_seen DESC; keep newest. Entries lacking last_seen (legacy) get 0 → evict first.
        kept = sorted(h.items(), key=lambda kv: -(kv[1].get("last_seen", 0) if isinstance(kv[1], dict) else 0))
        h = dict(kept[:HISTORY_MAX_ENTRIES])
    save_history(h)

# ---------- ssh helpers ----------
def run_on(node, shell_cmd, timeout=15, check=True):
    """Run shell_cmd on a node. Returns (returncode, stdout, stderr)."""
    host = NODES[node]["host"]
    if host is None:
        proc = subprocess.run(["bash", "-lc", shell_cmd], capture_output=True, timeout=timeout, text=True)
    else:
        # ServerAlive*: detect dead ControlMaster sockets fast (item 9 — half-dead persistent
        # connection used to take the full subprocess timeout to surface). 5s × 3 missed pings
        # = 15s before ssh declares dead; under our 15s default `timeout=`, this means a
        # broken master surfaces as ssh failure not subprocess timeout.
        proc = subprocess.run(
            ["ssh",
             "-o", "ConnectTimeout=5",
             "-o", "ServerAliveInterval=5",
             "-o", "ServerAliveCountMax=3",
             "-o", "BatchMode=yes",
             host, f"bash -lc {shlex.quote(shell_cmd)}"],
            capture_output=True, timeout=timeout, text=True,
        )
    if check and proc.returncode != 0:
        raise RuntimeError(f"[{node}] cmd failed (rc={proc.returncode}): {proc.stderr.strip()[:300]}")
    return proc.returncode, proc.stdout, proc.stderr

# ---------- node probe ----------
def probe_node(name):
    # /proc/meminfo is locale-independent (free -m's "Mem:" header is localized on some hosts).
    # util_pct is averaged over 3 samples (initial + 2 follow-ups, 100ms apart) because
    # nvidia-smi's `utilization.gpu` is a single-instant "any-kernel-active%" reading that
    # whipsaws between 0 and 100 on bursty RL workloads (SUMO sim ↔ train alternation).
    # 3 samples × 100ms ≈ 200ms extra latency per node; nodes probe in parallel so wall
    # cost is +200ms total.
    #
    # On `local` (WSL2), the WSL nvidia-smi sees only CUDA contexts opened FROM WSL — Windows
    # native GPU activity (browsers, compositor, games) is invisible. Result: util reads ~30%
    # when Task Manager / `nvidia-smi.exe` show ~50%. We use the Windows-side `nvidia-smi.exe`
    # (via WSL interop) when available so scheduler decisions reflect TRUE remaining headroom,
    # not just WSL-side usage. Fall back to plain `nvidia-smi` if the .exe isn't present.
    nvsmi = "nvidia-smi"
    if name == "local":
        win_nvsmi = "/mnt/c/WINDOWS/system32/nvidia-smi.exe"
        if Path(win_nvsmi).exists():
            nvsmi = win_nvsmi
    cmd = (f"{nvsmi} --query-gpu=index,memory.used,memory.total,memory.free,utilization.gpu "
           "--format=csv,noheader,nounits 2>/dev/null; echo '===SEP==='; "
           "awk '/^MemTotal:/{t=int($2/1024)} /^MemAvailable:/{a=int($2/1024)} END{print a; print t}' /proc/meminfo; echo '===SEP==='; "
           "nproc; echo '===SEP==='; "
           "awk '{print $1}' /proc/loadavg; echo '===SEP==='; "
           f"for i in 1 2; do sleep 0.1; "
           f"{nvsmi} --query-gpu=index,utilization.gpu --format=csv,noheader,nounits 2>/dev/null; "
           "echo '---SAMPLE---'; done")
    try:
        rc, out, _ = run_on(name, cmd, timeout=15, check=False)
        if rc != 0:
            return {"name": name, "alive": False, "error": "ssh/cmd failed"}
    except Exception as e:
        return {"name": name, "alive": False, "error": str(e)[:120]}
    parts = out.split("===SEP===")
    gpus = []
    for line in parts[0].strip().splitlines():
        bits = [x.strip() for x in line.split(",")]
        if len(bits) < 5: continue
        gpus.append({"idx": int(bits[0]), "used_mb": int(bits[1]), "total_mb": int(bits[2]),
                     "free_mb": int(bits[3]), "util_pct": int(bits[4])})
    # Fold extra util samples (parts[4]) into a 3-sample average per-GPU.
    if len(parts) > 4:
        extra = {}  # idx → list of util pct
        for chunk in parts[4].split("---SAMPLE---"):
            for line in chunk.strip().splitlines():
                bits = [x.strip() for x in line.split(",")]
                if len(bits) < 2: continue
                try:
                    extra.setdefault(int(bits[0]), []).append(int(bits[1]))
                except ValueError:
                    continue
        for g in gpus:
            samples = [g["util_pct"]] + extra.get(g["idx"], [])
            if len(samples) > 1:
                g["util_pct"] = sum(samples) // len(samples)
    # parts[1] has two ints in fixed order: MemAvailable then MemTotal (awk END block prints
    # `a` first, `t` second). Earlier version relied on /proc/meminfo line order, which is
    # actually MemTotal first → silently swapped free with total → free_ram massively over-
    # reported, total_ram under-reported → headroom guard ineffective. The explicit END block
    # decouples emission order from /proc/meminfo layout.
    mem_lines = [int(x) for x in parts[1].strip().split() if x.isdigit()] if len(parts) > 1 else []
    free_ram = mem_lines[0] if len(mem_lines) >= 1 else 0
    probed_total_ram = mem_lines[1] if len(mem_lines) >= 2 else 0
    cores = int(parts[2].strip()) if len(parts) > 2 and parts[2].strip().isdigit() else 0
    try:
        loadavg = float(parts[3].strip()) if len(parts) > 3 else 0.0
    except ValueError:
        loadavg = 0.0
    info = NODES.get(name, {})
    # Use min(configured, probed) so an over-declared config can't inflate the headroom denominator
    # past what the machine actually has. (Bug seen on WSL local: configured 56GB vs actual 30GB
    # produced a 14GB headroom that blocked all packing once a single 2GB task launched.)
    declared = info.get("ram_mb", 0)
    if declared and probed_total_ram:
        total_ram = min(declared, probed_total_ram)
    elif probed_total_ram:
        total_ram = probed_total_ram
    elif declared:
        total_ram = declared
    else:
        total_ram = free_ram + 1
    # cpu_cores / ram_mb in NODES are schedulable budgets, not necessarily physical totals.
    # If a remote box has more physical RAM than the configured budget, do not let MemAvailable
    # inflate placement capacity past that budget.
    sched_free_ram = min(free_ram, total_ram) if total_ram else free_ram
    total_cpu = info.get("cpu_cores", cores) or cores or 1
    free_cpu = max(0, total_cpu - int(round(loadavg)))
    return {"name": name, "alive": True, "gpus": gpus,
            "free_ram_mb": sched_free_ram, "actual_free_ram_mb": free_ram, "total_ram_mb": total_ram,
            "free_cpu": free_cpu, "total_cpu": total_cpu, "loadavg": loadavg,
            "cores": cores}

def probe_all():
    with ThreadPoolExecutor(max_workers=len(NODES)) as ex:
        return list(ex.map(probe_node, NODES.keys()))

# ---------- running-task health + peak VRAM tracking ----------
def _task_pids(task):
    """Return list of remote PIDs for this task. Tolerates legacy single 'remote_pid' field."""
    pids = task.get("remote_pids")
    if pids: return list(pids)
    p = task.get("remote_pid")
    return [p] if p else []

def _task_process_groups(task):
    """Return process-group ids that are safe to target for this task.

    Scheduler-launched tasks use `setsid`, so their recorded root PID is the PGID. Watcher
    auto-adopted tasks record `process_group` when discovered. Manual legacy adopts may not
    have a PGID; for those we only kill explicit PIDs to avoid overreaching into a user's shell
    process group."""
    groups = set()
    pg = task.get("process_group")
    try:
        if pg and int(pg) > 1:
            groups.add(int(pg))
    except (TypeError, ValueError):
        pass
    if not task.get("auto_adopted"):
        for p in _task_pids(task):
            try:
                if int(p) > 1:
                    groups.add(int(p))
            except (TypeError, ValueError):
                continue
    return sorted(groups)

def _kill_task_processes(task, timeout=15):
    """Thin wrapper: delegates to the active Backend. See Backend.kill."""
    return _BACKEND.kill(task, timeout=timeout)

def check_running(task):
    """Returns 'alive' if backend reports the task's tracked artifact is alive,
    'dead' if backend reports terminal OR has no tracking info for the task, 'unknown'
    on probe failure. Folds VRAM/RAM into peak trackers when alive.

    Thin wrapper around Backend.batch_probe over a single-task state. Backend decides
    which artifact to consult (PIDs for LocalBackend, slurm_job_id for SlurmBackend).

    Phase 2.9 P2 fix: previously had `if not _task_pids(task): return "dead"` early-
    return. That bypassed the backend entirely, so slurm tasks (which SlurmBackend.launch
    sets remote_pids=[] for, tracking via slurm_job_id instead) were always reported
    dead even when squeue would say RUNNING. The check_running helper's semantics were
    broken even though the main _batch_check_running path was unaffected (it routes
    via _BACKEND.batch_probe directly without the early-return).

    Now we let the backend decide. Each backend's batch_probe already filters tasks
    lacking its tracking artifact: LocalBackend skips on `not pids`, SlurmBackend
    skips on `not jid`. A task with neither falls through to `if not res: return "dead"`
    naturally. No extra ssh cost — both backends short-circuit when by_node is empty.
    """
    fake_state = {"tasks": [task]}
    res = _BACKEND.batch_probe(fake_state).get(task["id"])
    if not res:
        return "dead"
    if res["state"] == "unknown":
        return "unknown"
    if res["state"] == "dead":
        return "dead"
    # alive: fold deltas into peak trackers (max-tracking — peak only goes up)
    if res["vram_mb"] > 0:
        task["peak_vram_mb"] = max(task.get("peak_vram_mb", 0), res["vram_mb"])
    if res["ram_mb"] > 0:
        task["peak_ram_mb"] = max(task.get("peak_ram_mb", 0), res["ram_mb"])
    task["alive_pids"] = res["alive_pids"]
    return "alive"

CRASH_PATTERNS = [
    "Traceback (most recent call",
    "Error:", "Exception:",
    "Killed", "Segmentation fault",
    "out of memory", "OOM", "CUDA out of memory",
    "ModuleNotFoundError", "ImportError", "FileNotFoundError",
    "ConnectionError", "AssertionError", "RuntimeError",
]
SUCCESS_PATTERNS = [  # if any of these is in the log tail, the task definitely finished normally
    "Training complete",
    "DONE",
    " complete!",
    "Final model saved",
    "Saved final",
    "Eval complete",
    "Saved: ",  # generic JSON dump line many of our scripts end with
    "Saving final",
    "[Done]",
    # Clean no-op exits: when a script with --skip_existing / resume logic finds nothing left to
    # do, it exits in seconds. These were getting false-flagged as crashes (no traceback, no
    # success marker, lifetime < SHORT_LIVE) and auto-requeued forever. eval_checkpoints_parallel.py
    # specifically prints "Running 0 checkpoints" when --skip_existing eats everything.
    "Running 0 checkpoints",
    "Nothing to ",          # "Nothing to evaluate", "Nothing to do", etc.
    "no checkpoints to",    # case-insensitive partial match
    "All ckpts already",
]
# TRAINING_MARKERS: if log tail contains ANY of these, training did at least START (i.e. the task
# made it past env/init/load phase). Used to catch tasks killed mid-init (e.g. OOM before model
# loaded) — they have no error pattern, lifetime > SHORT_LIVE, but never wrote a training marker.
TRAINING_MARKERS = [
    "Epoch ", "[Epoch", "epoch ",
    "step=", "[Step", "Step ",
    "iter=", "iteration ",
    "loss=", "loss ",
    "Train: ", "[Train",
    "val_", "eval_",
]
EARLY_DEATH_SECONDS = 120   # any task dying within this window is treated as crash regardless of log
SHORT_LIVE_SECONDS = 600    # task < this AND no success marker AND no error trace → suspected crash

def _diagnose_terminal(task):
    """Inspect a just-finished task to classify normal-exit vs crash. Returns dict with is_crash + reason + log tail.
    For auto-adopted tasks (which scheduler didn't launch and has no log_path), we cannot reliably diagnose
    — refuse to call them crashed just because their log is invisible to us. Skip diagnosis entirely."""
    log_path = task.get("log_path")
    started = task.get("started_at") or 0
    finished = task.get("finished_at") or time.time()
    lifetime = max(0, finished - started)
    # Auto-adopted task without a scheduler-owned log = blind spot. Don't pretend to diagnose it.
    if task.get("auto_adopted") or not log_path:
        return {"is_crash": False, "reason": "auto-adopted (no scheduler log; cannot diagnose)",
                "tail": "(no log)", "lifetime_s": int(lifetime), "log_size": 0,
                "log_path": log_path, "success_marker": None}
    tail_text, log_size = "", 0
    if log_path:
        try:
            node = task.get("node")
            if node and NODES.get(node, {}).get("host") is None:
                lp = Path(log_path)
                if lp.exists():
                    log_size = lp.stat().st_size
                    with open(lp, "rb") as f:
                        f.seek(max(0, log_size - 4096))
                        tail_text = f.read().decode("utf-8", errors="replace")
            elif node:
                rc, out, _ = run_on(node,
                    f"tail -c 4096 {shlex.quote(log_path)} 2>/dev/null; echo '___SZ___'; wc -c < {shlex.quote(log_path)} 2>/dev/null",
                    timeout=10, check=False)
                if rc == 0 and out:
                    if "___SZ___" in out:
                        body, _, sz = out.rpartition("___SZ___")
                        tail_text = body
                        try: log_size = int(sz.strip())
                        except ValueError: pass
                    else:
                        tail_text = out
        except Exception:
            pass
    # If the user's cmd has its own stdout/stderr redirect (`> file`, `2>&1`, `&>`, etc.), the
    # wrapper log written by launch() will be 0 bytes — bash's inner redirect overrides the
    # outer one. Try to extract the user's actual log path from the cmd and read THAT file
    # instead, so size + tail reflect real training output. If extraction fails, mark the
    # log-based heuristics untrusted so we don't false-positive a successful run.
    cmd_str = task.get("cmd") or ""
    cmd_has_own_redirect = bool(re.search(r"(?<!\d)(?:&>|>>|2>&1|>&|>)\s*[^\s|;&)]+", cmd_str)) \
                           or "2>&1" in cmd_str
    log_trusted = True
    if cmd_has_own_redirect and log_size == 0:
        # Try to recover the real log path. Match the LAST `>` redirect (POSIX semantics: last
        # one wins), allowing optional `2>&1` after it. Skip `2>&1` itself as a target.
        m = re.search(r"(?:^|\s)(?:&>|>)\s*([^\s|;&)<>]+)", cmd_str)
        real_log_path = m.group(1) if m else None
        if real_log_path:
            try:
                node = task.get("node")
                if node and NODES.get(node, {}).get("host") is None:
                    rp = Path(real_log_path)
                    if rp.exists():
                        log_size = rp.stat().st_size
                        with open(rp, "rb") as f:
                            f.seek(max(0, log_size - 4096))
                            tail_text = f.read().decode("utf-8", errors="replace")
                elif node:
                    rc, out, _ = run_on(node,
                        f"tail -c 4096 {shlex.quote(real_log_path)} 2>/dev/null; echo '___SZ___'; wc -c < {shlex.quote(real_log_path)} 2>/dev/null",
                        timeout=10, check=False)
                    if rc == 0 and out and "___SZ___" in out:
                        body, _, sz = out.rpartition("___SZ___")
                        tail_text = body
                        try: log_size = int(sz.strip())
                        except ValueError: pass
            except Exception:
                pass
        # Whether or not we recovered the real log, mark log_trusted=False so the size-based
        # heuristic (step 5) and stuck-in-init heuristic (step 6) don't fire on a wrapper log
        # that is empty by design.
        log_trusted = (log_size > 0)
    err_matched = [p for p in CRASH_PATTERNS if p in tail_text]
    success_matched = [p for p in SUCCESS_PATTERNS if p in tail_text]
    reasons = []
    is_crash = False
    # Decision tree (ordered):
    # (1) explicit success marker → not a crash, even if lifetime short
    # (2) explicit error pattern → crash (overrides success — though shouldn't co-occur)
    # (3) very early death (< EARLY_DEATH_SECONDS) → suspect crash regardless
    # (4) short-lived (< SHORT_LIVE_SECONDS) AND no success marker → suspect crash
    # (5) log very small AND lifetime > 60s → suspect crash
    # (else) treat as normal completion
    if err_matched:
        is_crash = True
        reasons.append("err_pattern: " + ", ".join(err_matched[:3]))
    elif success_matched:
        # Don't override; trust the success marker
        pass
    elif 0 < lifetime < EARLY_DEATH_SECONDS:
        is_crash = True
        reasons.append(f"died after only {lifetime:.0f}s with no success marker")
    elif 0 < lifetime < SHORT_LIVE_SECONDS:
        is_crash = True
        reasons.append(f"finished in {lifetime:.0f}s with no success marker (expected DONE/Saved/complete)")
    elif log_trusted and log_size < 500 and lifetime > 60:
        is_crash = True
        reasons.append(f"log only {log_size}B after {lifetime:.0f}s (very suspicious)")
        # Item 26: 0-byte log + non-trivial lifetime is the disk-full signature when the
        # script's own write to log was rejected. Probe `df` on the log dir; if any
        # mount > 95%, retag as DISK_FULL so _classify_failure routes to the correct
        # escalation category (DISK_FULL, not generic UNKNOWN). Best-effort; don't
        # crash diagnose if probe fails.
        try:
            log_path = task.get("log_path") or ""
            log_dir = log_path.rsplit("/", 1)[0] if "/" in log_path else "/tmp"
            rc_df, out_df, _ = run_on(
                task.get("node") or "local",
                f"df -P {shlex.quote(log_dir)} | tail -n +2 | awk '{{print $5}}' | tr -d %",
                timeout=5, check=False,
            )
            if rc_df == 0:
                pct = int((out_df.strip() or "0").splitlines()[-1])
                if pct >= 95:
                    reasons.append(f"DISK_FULL: {log_dir} at {pct}% on {task.get('node')}")
        except Exception:
            pass  # diagnostic only; don't fail diagnose because probe couldn't reach node
    else:
        # (6) "stuck-in-init" check: long-lived task but never wrote a training marker AND
        # has peak_vram_mb=0 → almost certainly killed before training started. This is the
        # OOM-victim signature: SUMO sim init wrote some lines, then a child got OOM-killed
        # silently (no traceback in log), parent died too. Lifetime > SHORT_LIVE so the
        # earlier branches missed it; success_pattern absent so we know it didn't complete.
        # Only fires when log_trusted: a cmd with its own redirect leaves the wrapper log
        # empty by design, which would false-positive every long run.
        has_training = any(m in tail_text for m in TRAINING_MARKERS)
        peak_v = int(task.get("peak_vram_mb") or 0)
        if log_trusted and not has_training and peak_v == 0 and lifetime > SHORT_LIVE_SECONDS:
            is_crash = True
            reasons.append(
                f"never entered training: lifetime {lifetime:.0f}s but peak_vram=0 and no "
                f"training markers in log tail (likely OOM/silent-kill mid-init)"
            )
        elif log_trusted and not has_training and peak_v > 0:
            # Codex follow-up: log tail lacks training markers (rotated out, custom log format,
            # or markers never matched our list), BUT peak_vram>0 proves the task DID hit the
            # GPU and ran real work. No success marker means it didn't finish cleanly → must
            # be requeued. Without this branch the task fell through to "ambiguous; assumed
            # normal" → marked done → no requeue → silent work loss for any non-standard
            # logger format.
            is_crash = True
            reasons.append(
                f"GPU work observed (peak_vram={peak_v}MB) but no success marker after "
                f"{lifetime:.0f}s — kill mid-execution (training markers may have been "
                f"rotated out of tail or use non-standard format); auto-requeue will "
                f"resume from latest ckpt if --resume-flag is set"
            )
        elif log_trusted and has_training:
            # WSRL 05-04 footgun fix: log shows the task was actively training (Epoch/step/iter
            # markers present) but no success marker means it didn't finish cleanly. SIGKILL from
            # host reboot / external kill / OOM-killer leaves no traceback in the log, so the
            # CRASH_PATTERNS check above can't catch it. Without this rule, a mid-training kill
            # falls through to "ambiguous; assumed normal" → status=done → no auto-requeue → 50h
            # of work silently lost. Cost asymmetry favors flagging: false-positive is one
            # extra auto-requeue (retry budget caps it at 3); false-negative loses real progress.
            is_crash = True
            reasons.append(
                f"training markers present but no success marker after {lifetime:.0f}s — "
                f"task killed mid-training (likely SIGKILL/OOM/host reboot); "
                f"auto-requeue will resume from latest ckpt if --resume-flag is set"
            )
    return {"is_crash": is_crash, "reason": "; ".join(reasons) or "normal exit (success marker found)" if success_matched else "; ".join(reasons) or "ambiguous; assumed normal",
            "tail": tail_text[-600:].strip() if tail_text else "(no log)",
            "lifetime_s": int(lifetime), "log_size": log_size, "log_path": log_path,
            "success_marker": success_matched[0] if success_matched else None}

def _detect_oom_kills_local(state):
    """Ex-post OOM detector. WSL/Linux kernel logs OOM kills to /var/log/syslog. When the
    OOM-killer fires, it picks the largest memory hog — often NOT the scheduler-launched
    task itself, but a sibling process. Result: scheduler's task ends "ambiguously" (no
    traceback, no success marker), gets marked done by the heuristic, and silently lost.

    This function scans recent syslog OOM events and flips any LOCAL `done` task whose
    finished_at temporally overlaps with an OOM kill (and which never entered training)
    to `failed`, so the next watcher tick auto-requeues it via _requeue_after_crash.
    Only operates on local — remote nodes don't surface their syslog to us."""
    import re as _re, datetime as _dt, subprocess as _sp
    syslog = "/var/log/syslog"
    if not Path(syslog).exists():
        return []
    try:
        # tail -5000 covers a few hours of normal events; OOM events are sparse so this is fine.
        out = _sp.check_output(["tail", "-5000", syslog], text=True, errors="replace", timeout=5)
    except Exception:
        return []
    now = time.time()
    cutoff = now - 1800  # only care about OOM kills in last 30 min
    pat = _re.compile(r"^(\w+\s+\d+\s+\d+:\d+:\d+).*Out of memory: Killed process")
    oom_times = []
    year = _dt.datetime.now().year
    for line in out.splitlines():
        m = pat.match(line)
        if not m:
            continue
        try:
            t = _dt.datetime.strptime(f"{year} {m.group(1)}", "%Y %b %d %H:%M:%S")
            ts = t.timestamp()
        except Exception:
            continue
        if ts >= cutoff:
            oom_times.append(ts)
    if not oom_times:
        return []
    # Find local `done` tasks that overlap an OOM event AND look incomplete.
    flipped = []
    for t in state.get("tasks", []):
        if t.get("node") != "local":
            continue
        if t.get("status") != "done":
            continue
        if t.get("auto_adopted"):
            continue  # adopted tasks have no scheduler log; skip ex-post
        sta = t.get("started_at") or 0
        fin = t.get("finished_at") or 0
        if fin < cutoff:
            continue
        # OOM overlap: any event in [start - 30s, finish + 60s] window
        if not any(sta - 30 <= oom_t <= fin + 60 for oom_t in oom_times):
            continue
        # Incomplete signature: never loaded model on GPU AND no success marker recorded
        diag = t.get("_diagnosis") or {}
        if diag.get("success_marker"):
            continue
        if int(t.get("peak_vram_mb") or 0) > 100:
            continue  # task got into training, OOM kill probably hit a different process
        # Already flipped on a prior watcher tick? avoid double-handling
        if t.get("_oom_flipped"):
            continue
        t["status"] = "failed"
        t["_oom_flipped"] = True
        t["last_block_reason"] = (f"ex-post OOM: terminated within local OOM-kill window without "
                                   f"entering training (peak_vram={t.get('peak_vram_mb')}MB)")
        diag = dict(diag)
        diag["is_crash"] = True
        diag["reason"] = (diag.get("reason") or "") + " | ex-post: syslog OOM in window"
        t["_diagnosis"] = diag
        flipped.append(t)
    return flipped

def _classify_failure(diag):
    """Categorize a crash diag for routing: ENV_MISSING / PYTHON_IMPORT / OOM / APP_BUG / UNKNOWN.
    Looked at by _requeue_after_crash to decide retry-vs-escalate, and by pick_placement to skip nodes
    where this signature already failed for environment reasons."""
    if not diag or not diag.get("is_crash"):
        return "NORMAL"
    tail = diag.get("tail", "") or ""
    reason = diag.get("reason", "") or ""
    haystack = (tail + " " + reason).lower()
    for p in ENV_MISSING_PATTERNS:
        if p.lower() in haystack:
            return "ENV_MISSING"
    for p in PYTHON_IMPORT_PATTERNS:
        if p.lower() in haystack:
            return "PYTHON_IMPORT"
    # Disk-full check before OOM: OSError errno 28 is unambiguous, while OOM_PATTERNS could
    # accidentally match. Codex P1: distinguishing DISK_FULL from generic crash gives the user
    # actionable diagnosis ("clean /tmp" vs "tune memory").
    for p in DISK_FULL_PATTERNS:
        if p.lower() in haystack:
            return "DISK_FULL"
    for p in OOM_PATTERNS:
        if p.lower() in haystack:
            return "OOM"
    if "Traceback" in tail and diag.get("lifetime_s", 0) > 60:
        return "APP_BUG"
    return "UNKNOWN"

_HEAL_FIRE_LOCK = STATE_DIR / ".heal_fire.lock"  # debounce marker; mtime = last fire ts
_HEAL_DEBOUNCE_S = 90  # don't refire within this window — heal skill itself handles batching
_CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "/home/erzhu419/.nvm/versions/node/v22.17.1/bin/claude")
# CLI is `#!/usr/bin/env node`; system node is too old (v12) and shebang lookup picks it.
# Spawn with absolute node binary + cli.js path to bypass shebang env-lookup entirely.
_NODE_BIN = "/home/erzhu419/.nvm/versions/node/v22.17.1/bin/node"
_CLAUDE_CLI_JS = "/home/erzhu419/.nvm/versions/node/v22.17.1/lib/node_modules/@anthropic-ai/claude-code/cli.js"
_HEAL_FIRE_LOG = LOG_DIR / "heal_fires.log"

def _fire_heal_session():
    """Spawn a headless `claude -p /scheduler-heal` session, fire-and-forget.
    Debounced: at most one fire per _HEAL_DEBOUNCE_S window.

    Uses a STRIPPED env (no inherited LD_LIBRARY_PATH / shell rc) because the user's
    interactive shell sets LD_LIBRARY_PATH to anaconda's libs (libtinfo.so.6) which makes
    Node die on cli.js startup. With a minimal env containing just HOME + nvm-bin PATH,
    claude starts cleanly. Verified equivalent to `env -i HOME=... PATH=nvm:/usr/bin claude ...`."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        now = time.time()
        if _HEAL_FIRE_LOCK.exists():
            try:
                last = _HEAL_FIRE_LOCK.stat().st_mtime
                if now - last < _HEAL_DEBOUNCE_S:
                    return False  # debounced
            except Exception:
                pass
        _HEAL_FIRE_LOCK.touch()
        os.utime(_HEAL_FIRE_LOCK, (now, now))
        log_fh = open(_HEAL_FIRE_LOG, "ab")
        log_fh.write(f"\n=== fire @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n".encode())
        log_fh.flush()
        nvm_bin = os.path.dirname(_CLAUDE_BIN)  # /home/erzhu419/.nvm/.../bin
        clean_env = {
            "HOME": str(Path.home()),
            "USER": os.environ.get("USER", "erzhu419"),
            "PATH": f"{nvm_bin}:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "CLAUDE_HEAL_FIRE": "1",
        }
        # Skip shebang env-lookup: invoke node directly with cli.js path. The CLI's shebang is
        # #!/usr/bin/env node which resolves /usr/bin/node (system v12, too old for ESM ?.). We
        # need the nvm node v22 explicitly.
        subprocess.Popen(
            [_NODE_BIN, _CLAUDE_CLI_JS, "-p", "/scheduler-heal", "--dangerously-skip-permissions"],
            stdin=subprocess.DEVNULL, stdout=log_fh, stderr=log_fh,
            start_new_session=True, cwd=str(Path.home()),
            env=clean_env,
        )
        log_fh.close()
        return True
    except Exception as e:
        try:
            with open(_HEAL_FIRE_LOG, "a") as f:
                f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] _fire_heal_session FAILED: {e}\n")
        except Exception:
            pass
        return False

def _write_escalation(task, category, diag):
    """Append a structured escalation. /scheduler-heal skill reads escalations.jsonl and
    appends a new line with status=resolved when fixed (records are append-only).
    Also fires a headless claude -p /scheduler-heal session (debounced) so escalations get
    auto-handled even if no Claude conversation is currently open."""
    rec = {
        "ts": time.time(),
        "task_id": task["id"],
        "signature": task.get("signature", ""),
        "project": task.get("project", ""),
        "category": category,
        "node": task.get("node"),
        "gpu_idx": task.get("gpu_idx"),
        "reason": diag.get("reason", ""),
        "tail": (diag.get("tail", "") or "")[-500:],
        "log_path": diag.get("log_path"),
        "retry_count": task.get("retry_count", 0),
        "cmd": (task.get("cmd", "") or "")[:200],
        "cwd": task.get("cwd", ""),
        "status": "pending",
    }
    ESCALATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ESCALATIONS_FILE, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _fire_heal_session()

def _blocked_nodes_for_task(task):
    """Read pending escalations and return the set of nodes where THIS task's env is broken.
    Match on (signature OR cwd OR project) so sibling tasks of the same project family don't
    keep re-dispatching to a node we already know lacks the project's shared env. (Earlier
    version only matched exact signature, which let e.g. rlpd_050/s789 retry jtl110gpu2 even
    though rlpd_050/s42 had just established that the env was missing there.)"""
    if not ESCALATIONS_FILE.exists():
        return set()
    sig = task.get("signature") or ""
    cwd = task.get("cwd") or ""
    project = task.get("project") or ""
    latest = {}
    try:
        for line in ESCALATIONS_FILE.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            tid = rec.get("task_id")
            if tid:
                latest[tid] = rec
    except Exception:
        return set()
    blocked = set()
    for rec in latest.values():
        if rec.get("status") != "pending":
            continue
        if rec.get("category") not in ("ENV_MISSING", "PYTHON_IMPORT"):
            continue
        if not rec.get("node"):
            continue
        # Match on ANY of: same signature / same cwd / same project. cwd is the strongest
        # indicator (same conda env / sumo build); project is a fallback when cwd metadata is
        # missing (e.g. older tasks). Sig match preserves prior precise behavior.
        same_sig = sig and rec.get("signature") == sig
        same_cwd = cwd and rec.get("cwd") == cwd
        same_proj = project and rec.get("project") == project
        if same_sig or same_cwd or same_proj:
            blocked.add(rec["node"])
    return blocked

def _blocked_nodes_for_signature(sig):
    """Back-compat shim — old callers can keep passing a string. Wraps the new task-based check."""
    return _blocked_nodes_for_task({"signature": sig})

def _launch_failed_nodes_for_task(task):
    """Nodes that recently failed to launch this specific task. Soft block only: placement
    tries other nodes first, but may retry these if no clean candidate exists so the retry cap
    can still fire instead of leaving the task queued forever."""
    raw = task.get("launch_failed_nodes") or {}
    if isinstance(raw, dict):
        return {n for n in raw.keys() if n in NODES}
    if isinstance(raw, list):
        return {n for n in raw if n in NODES}
    return set()

def _requeue_after_crash(parent, state):
    """Clone a crashed scheduler-owned task back into the queue so it gets re-dispatched.
    Returns the new task id, or None if ineligible (no real cmd captured, retry cap reached).
    Preserves submitted_at so the re-queue sorts to the head of its priority class.

    HARD-FAIL categories (ENV_MISSING / PYTHON_IMPORT / OOM): write an escalation INSTEAD of retry.
    SOFT-FAIL categories (APP_BUG / UNKNOWN): retry up to MAX_AUTO_RETRY, then escalate as APP_BUG_CAP.

    Auto-adopted tasks are eligible IF we captured a real cmdline at adopt time (`task.cmd`
    != placeholder). Otherwise we have nothing to relaunch and must skip."""
    cmd = parent.get("cmd") or ""
    if cmd.startswith("(auto-adopted"):
        return None  # No real cmd captured (legacy adopt or /proc/<pid>/cmdline was unreadable)
    # Dedup guard: refuse to requeue if an active (queued/running) task with the same signature
    # AND same cmd already exists. Without this guard, multiple failure paths converging on the
    # same target (e.g. ENV_MISSING crash → requeue + manual force-requeue) produce duplicate
    # active instances all running the same training, wasting GPU. Cancelled/done/failed don't
    # block — only live duplicates do.
    sig = parent.get("signature") or ""
    if sig:
        for existing in state.get("tasks", []):
            if existing.get("id") == parent.get("id"): continue
            if existing.get("status") not in ("queued", "running", "launching"): continue
            if existing.get("signature") != sig: continue
            if existing.get("cmd") != cmd: continue
            # Found a live duplicate — link parent's requeued_as to it instead of creating a new one
            return existing["id"]
    diag = parent.get("_diagnosis") or {}
    category = _classify_failure(diag)
    parent["failure_category"] = category
    if category in ("ENV_MISSING", "PYTHON_IMPORT", "OOM", "DISK_FULL"):
        _write_escalation(parent, category, diag)
        return None
    retry_n = parent.get("retry_count", 0) + 1
    if retry_n > MAX_AUTO_RETRY:
        _write_escalation(parent, "APP_BUG_CAP", diag)
        return None
    new_id = f"t{state['next_id']:04d}"
    state["next_id"] += 1
    new_task = {**parent}
    new_task.update({
        "id": new_id,
        "status": "queued",
        "node": None,
        "gpu_idx": None,
        "remote_pids": [],
        "log_path": None,
        "started_at": None,
        "finished_at": None,
        "peak_vram_mb": 0,
        "peak_ram_mb": 0,
        "alive_pids": [],
        "resume_from": None,
        # Backend launch artifacts must not survive into a retry clone. A crashed
        # Slurm task carrying its old slurm_job_id would be routed back to
        # SlurmBackend even after dispatch picks a local node, and docker container
        # handles from the parent are stale for the new task id.
        "slurm_job_id": None,
        "slurm_state": None,
        "container_name": None,
        "container_main_pid": None,
        # A requeued task is scheduler-owned even if the parent was auto-adopted with a
        # captured real cmdline. Do not inherit auto_adopted/adopted, otherwise terminal
        # diagnosis will skip scheduler logs and preemption will treat it as external.
        "adopted": False,
        "auto_adopted": False,
        "process_group": None,
        "_diagnosis": None,
        "notified_launch": False,
        "notified_done": False,
        "retry_count": retry_n,
        "parent_id": parent["id"],
        "last_block_reason": f"auto-requeue (retry {retry_n}/{MAX_AUTO_RETRY}) after {parent['id']} crashed",
    })
    state["tasks"].append(new_task)
    return new_id

def _batch_check_running(state):
    """One round-trip per node to check liveness + VRAM + RSS for ALL running tasks. Probe
    is delegated to _BACKEND.batch_probe (Phase 1 abstraction); this function keeps all the
    policy logic: transition to done/failed, diagnose terminal tasks, fold history, upward-
    track ram_mb / cpu_cores estimates from observed peaks. Sets t['_diagnosis'] for caller
    to surface as event."""
    import math as _math

    probe_results = _BACKEND.batch_probe(state)
    if not probe_results:
        return

    for t in state["tasks"]:
        if t["status"] != "running": continue
        res = probe_results.get(t["id"])
        if res is None or res["state"] == "unknown":
            # ssh failed for this node OR task wasn't probed — leave state alone.
            continue
        if res["state"] == "dead":
            t["status"] = "done"
            t["finished_at"] = time.time()
            # Slurm reports terminal states directly. Trust COMPLETED as a clean exit and
            # treat explicit Slurm failure states as crashes instead of trying to infer
            # everything from logs (logs can be missing/rotated while Slurm still knows
            # the job outcome). For LocalBackend / absent-from-squeue cases, fall back
            # to the existing log heuristic.
            terminal_ok = res.get("terminal_ok")
            backend_state = res.get("backend_state")
            terminal_reason = res.get("terminal_reason") or (
                f"slurm terminal state {backend_state}" if backend_state else ""
            )
            if terminal_ok is True:
                diag = {
                    "is_crash": False,
                    "reason": terminal_reason or "slurm terminal state COMPLETED",
                    "tail": "(slurm reported COMPLETED; log not required for success)",
                    "lifetime_s": int(max(0, t["finished_at"] - (t.get("started_at") or t["finished_at"]))),
                    "log_size": 0,
                    "log_path": t.get("log_path"),
                    "success_marker": f"SLURM_{backend_state or 'COMPLETED'}",
                }
            else:
                # Diagnose: did it complete normally, or crash? Tail log + check patterns + lifetime.
                diag = _diagnose_terminal(t)
                if terminal_ok is False:
                    diag = dict(diag)
                    diag["is_crash"] = True
                    reason = terminal_reason or "slurm terminal state indicates failure"
                    if backend_state == "OUT_OF_MEMORY":
                        reason += " (out of memory)"
                    prior = diag.get("reason") or ""
                    if prior and prior != "ambiguous; assumed normal":
                        reason = f"{reason}; {prior}"
                    diag["reason"] = reason
            t["_diagnosis"] = diag  # caller (watcher) reads + emits task_crashed if needed
            # Belt-and-suspenders: never let heuristics flip an auto-adopted task to "failed".
            # Adopted tasks have no scheduler-owned log so heuristics like "log 0B after Ns"
            # systematically misfire on them. _diagnose_terminal already early-returns False
            # for adopted; this guard is in case that path is ever bypassed in a refactor.
            if diag["is_crash"] and not t.get("auto_adopted"):
                t["status"] = "failed"  # distinguish in queue.json
                t["last_block_reason"] = diag["reason"]
                new_id = _requeue_after_crash(t, state)
                if new_id:
                    t["requeued_as"] = new_id
            # Universal "incomplete" check: lifetime far below historical EWMA → likely killed/crashed,
            # mark + auto-requeue (unless user explicitly cancelled, in which case status was already
            # set to 'cancelled' before this code runs and we never get here). Works for both
            # scheduler-launched AND auto-adopted tasks (now that adopt captures cmdline).
            elif (not diag.get("is_crash")
                  and terminal_ok is not True
                  and t.get("status") == "done"  # not already failed/cancelled
                  and t.get("started_at") and t.get("finished_at")):
                sig = t.get("signature") or ""
                h = history_get(sig) or {}
                expected = int(h.get("dur_s_ewma", 0))
                runs = int(h.get("dur_s_runs", 0))
                lifetime = max(0, t["finished_at"] - t["started_at"])
                # Need at least 2 historical samples before trusting the EWMA threshold.
                if expected > 0 and runs >= 2 and lifetime < 0.5 * expected:
                    t["status"] = "failed"
                    t["last_block_reason"] = (f"incomplete: ran {int(lifetime)}s "
                                               f"({int(100*lifetime/expected)}% of EWMA {expected}s); "
                                               f"likely killed/crashed (not user-cancelled)")
                    cmd_real = bool(t.get("cmd")) and not t["cmd"].startswith("(auto-adopted")
                    if cmd_real:
                        new_id = _requeue_after_crash(t, state)
                        if new_id:
                            t["requeued_as"] = new_id
            duration_s = 0
            if t.get("started_at") and t.get("finished_at"):
                duration_s = max(0, int(t["finished_at"] - t["started_at"]))
            history_record(
                t.get("signature"),
                peak_vram_mb=t.get("peak_vram_mb", 0),
                peak_ram_mb=t.get("peak_ram_mb", 0),
                cpu_cores=t.get("cpu_cores", 0),
                duration_s=duration_s,
            )
            continue
        # alive: fold deltas, upward-track ram_mb / cpu_cores estimates
        t["alive_pids"] = res["alive_pids"]
        total_vram = res["vram_mb"]
        if total_vram > 0:
            t["peak_vram_mb"] = max(t.get("peak_vram_mb", 0), total_vram)
        total_ram = res["ram_mb"]
        if total_ram > 0:
            t["peak_ram_mb"] = max(t.get("peak_ram_mb", 0), total_ram)
            # B: upward-track ram_mb when actual RSS exceeds declared budget. Closes the
            # gap where a task submits with ram_mb=8000 but really uses 15GB; without this,
            # next dispatch's RAM headroom check trusts the (lying) declared 8GB and packs
            # more tasks until WSL OOM. Apply to ALL tasks (scheduler-launched + adopted).
            # Down-direction is handled separately (lower-only) to avoid noise from transient
            # GC dips. Always-bump-up is correct since OOM blast-radius is asymmetric.
            declared_ram = t.get("ram_mb", DEFAULT_RAM_MB)
            # Add 10% slack so we don't constantly bump on minor fluctuations
            if total_ram > declared_ram * 1.1:
                t["ram_mb"] = total_ram
        # Refresh cpu_cores based on real %CPU. For ALL tasks now (was auto_adopted only):
        # SUMO/RL retrains often declare cpu_cores=2 at submit but actually use 7-8 cores
        # (libsumo + multi-step env). Without upward tracking, dispatch packs too many.
        total_pcpu = res["pcpu"]
        if total_pcpu > 0:
            new_cpu = max(1, _math.ceil(total_pcpu / 100.0))
            cur_cpu = t.get("cpu_cores", DEFAULT_CPU_CORES)
            if t.get("auto_adopted"):
                # Adopted: lower-only (legacy len(pids) overcount fix).
                if new_cpu < cur_cpu:
                    t["cpu_cores"] = new_cpu
            else:
                # Scheduler-launched: upward tracking when sustained, lower only on big over-count
                # (mirrors the _refresh_adopted_resources pattern from earlier).
                if new_cpu > cur_cpu:
                    t["cpu_cores"] = new_cpu
                elif cur_cpu > new_cpu * 2.5:
                    t["cpu_cores"] = new_cpu

def update_running_tasks(state):
    """Batched version — one ssh per node instead of one per task. See _batch_check_running.
    Phase 3.0.1: also refreshes per-task eta_seconds from log-tail progress parsing."""
    _batch_check_running(state)
    _refresh_eta_from_logs(state)


def _refresh_eta_from_logs(state):
    """Phase 3.0.1: parse log tails for tqdm/epoch/iter progress markers, compute
    rate-based remaining-seconds, write task['eta_seconds']. Falls back to
    (history_ewma - elapsed) when no progress signal is found in the tail.

    Used by Phase 3.0's load-balanced migration trigger: per-node load = sum of
    eta_seconds of in-flight tasks. Without live ETA, a node whose tasks are 90%
    done would falsely look as loaded as one that just started.

    Runs ONE ssh per node (tails all running tasks' logs there in a single shell
    cmd). Failure-tolerant: ssh fail / log missing / no parse → eta uses EWMA
    fallback or 0. Never raises out of this function.
    """
    try:
        from . import eta_tracker  # type: ignore
    except Exception:
        # Module loaded as a flat script; import via spec the same way env_deploy is
        try:
            import importlib.util as _ilu  # type: ignore
            spec = _ilu.spec_from_file_location(
                "eta_tracker", str(Path(__file__).parent / "eta_tracker.py")
            )
            if spec and spec.loader:
                eta_tracker = _ilu.module_from_spec(spec)
                spec.loader.exec_module(eta_tracker)
            else:
                return
        except Exception:
            return

    by_node = {}  # node -> [(task, log_path), ...]
    pure_ewma = []  # tasks without log_path; just compute fallback
    for t in state.get("tasks", []):
        if t.get("status") != "running":
            continue
        log_path = t.get("log_path")
        if not log_path or not t.get("node"):
            pure_ewma.append(t)
            continue
        by_node.setdefault(t["node"], []).append((t, log_path))

    # No-log tasks: just use EWMA fallback
    now = time.time()
    for t in pure_ewma:
        sig = t.get("signature") or ""
        h = history_get(sig) or {}
        ewma = int(h.get("dur_s_ewma", 0))
        elapsed = max(0, now - (t.get("started_at") or now))
        t["eta_seconds"] = eta_tracker.compute_eta_seconds(
            "", elapsed_s=elapsed, fallback_ewma_s=ewma, cmd=t.get("cmd"),
        )

    if not by_node:
        return

    import re as _re

    def _probe(node):
        entries = by_node[node]
        # Build a single ssh cmd that tails each task's log with a marker separator.
        # `tail -c 4096` returns at most 4KB which is plenty for the tqdm/epoch
        # patterns we need. `2>/dev/null` swallows missing-log errors. The trailing
        # `; true` keeps overall rc=0 even if some tails fail.
        parts = []
        for (t, log_path) in entries:
            tid = t["id"]
            parts.append(f"echo '===ETA_LOG_{tid}==='")
            parts.append(f"tail -c 4096 {shlex.quote(log_path)} 2>/dev/null")
        cmd = "; ".join(parts) + "; true"
        try:
            rc, out, _ = run_on(node, cmd, timeout=30, check=False)
            return out if rc == 0 else None
        except Exception:
            return None

    nodes_list = list(by_node.keys())
    with ThreadPoolExecutor(max_workers=max(1, len(nodes_list))) as ex:
        outputs = dict(zip(nodes_list, ex.map(_probe, nodes_list)))

    marker_re = _re.compile(r'===ETA_LOG_(\S+?)===')
    for node, out in outputs.items():
        if out is None:
            # ssh failed — leave eta_seconds untouched (or fallback if missing)
            for t, _ in by_node[node]:
                if t.get("eta_seconds") is None:
                    sig = t.get("signature") or ""
                    h = history_get(sig) or {}
                    ewma = int(h.get("dur_s_ewma", 0))
                    elapsed = max(0, now - (t.get("started_at") or now))
                    t["eta_seconds"] = eta_tracker.compute_eta_seconds(
                        "", elapsed_s=elapsed, fallback_ewma_s=ewma, cmd=t.get("cmd"),
                    )
            continue
        # Split on markers: ['header', 'tid1', 'tail1', 'tid2', 'tail2', ...]
        chunks = marker_re.split(out)
        if len(chunks) >= 3:
            it = iter(chunks[1:])
            log_by_tid = dict(zip(it, it))
        else:
            log_by_tid = {}
        for (t, _log_path) in by_node[node]:
            tid = t["id"]
            tail_text = log_by_tid.get(tid, "")
            sig = t.get("signature") or ""
            h = history_get(sig) or {}
            ewma = int(h.get("dur_s_ewma", 0))
            elapsed = max(0, now - (t.get("started_at") or now))
            t["eta_seconds"] = eta_tracker.compute_eta_seconds(
                tail_text, elapsed_s=elapsed, fallback_ewma_s=ewma, cmd=t.get("cmd"),
            )

# ---------- placement ----------
def _node_resources_ok(task, node_state, node_info):
    """CPU + RAM + concurrency check at node level (independent of which GPU). Returns (ok, reason).
    `node_state['running_count']` is set by _do_dispatch before pick_placement loop and incremented
    in-loop as new launches happen — caller is responsible for keeping it current."""
    # Hard cap on concurrent tasks per node (Fix A): defense against under-declared CPU/RAM
    # for SUMO/RL workloads. Even if cpu/ram math says "fits", refuse if we're at the cap.
    cap = node_info.get("max_concurrent_running")
    cur_running = node_state.get("running_count", 0)
    if cap is not None and cur_running >= cap:
        return False, f"concurrency cap: {cur_running}/{cap} tasks already running on {node_state.get('name','?')}"
    needed_cpu = task.get("cpu_cores", DEFAULT_CPU_CORES)
    if node_state.get("free_cpu", 0) < needed_cpu:
        return False, f"cpu: need {needed_cpu}, free {node_state.get('free_cpu', 0)}/{node_state.get('total_cpu', '?')}"
    needed_ram = task.get("ram_mb", DEFAULT_RAM_MB)
    frac = node_info.get("ram_headroom_frac", RAM_HEADROOM_FRAC)
    # Use the probed total when available so headroom tracks reality, not config; the config
    # value is only used as an upper bound (already enforced in probe_node via min()).
    total_for_headroom = node_state.get("total_ram_mb") or node_info.get("ram_mb", 0)
    headroom = int(total_for_headroom * frac)
    if node_state.get("free_ram_mb", 0) - needed_ram < headroom:
        return False, f"ram: need {needed_ram}MB, free {node_state.get('free_ram_mb', 0)}MB (headroom {headroom}MB)"
    return True, "ok"

def _gpu_fits(task, gpu, node_info):
    """VRAM + compute-saturation check on a specific GPU (1/3 packing rule + util-cap + per-task cap + margin).

    1/3 rule has a small-task exemption: a task whose est ≤ DEFAULT_VRAM_MB is allowed onto a
    GPU that's past 1/3 used, as long as the GPU's compute util is also low (<util-saturation).
    Rationale: the 1/3 rule exists to stop a NEW BIG task from compounding mem pressure. A 512MB
    novel task on a card that's at 3GB but 6% compute is not the case 1/3 was designed to block —
    locking it out leaves GPU truly idle while a queue piles up. If actual peak is too high,
    _enforce_post_dispatch_thresholds evicts the youngest cleanly. For larger tasks (>DEFAULT)
    the 1/3 rule still applies — they have higher OOM blast radius."""
    cap = node_info.get("max_vram_per_task")
    if cap is not None and task["est_vram_mb"] > cap:
        return False
    if ONE_THIRD_PACK_RULE:
        third = gpu["total_mb"] // 3
        if gpu["used_mb"] >= third and gpu["used_mb"] > 100:
            small_task = task["est_vram_mb"] <= DEFAULT_VRAM_MB
            util = gpu.get("util_pct", 0)
            chip_idle = util < GPU_UTIL_SATURATION_PCT - 20  # well below saturation, e.g. <65%
            if not (small_task and chip_idle):
                return False
            # else: small task, idle chip — allow stacking past 1/3 mem
    # Compute saturation: if there's already a task on this GPU and it's pinning the chip,
    # don't pack more — the new task would just steal cycles and slow everyone down.
    # The "occupied" guard (>100MB) avoids blocking on a transient util spike on an empty GPU.
    if gpu["used_mb"] > 100 and gpu.get("util_pct", 0) >= GPU_UTIL_SATURATION_PCT:
        return False
    if gpu["free_mb"] < task["est_vram_mb"] + VRAM_MARGIN_MB:
        return False
    return True

def pick_placement(task, nodes):
    """Pick (node, gpu_idx) given current per-node free resources. gpu_idx=None for CPU-only tasks.
    preferred_node is a SOFT preference: try it first; if it can't fit, fall back to any other node
    that satisfies all constraints. This prevents tasks from getting stuck when their preferred node
    is full but other nodes have headroom.

    Also consults pending escalations: if THIS signature has a pending ENV_MISSING/PYTHON_IMPORT
    escalation on a node, that node is excluded from candidates here so we don't keep redispatching
    the task to a node we already know can't run it. /scheduler-heal resolves the escalation when
    the env is fixed, freeing that node again."""
    cpu_only = task.get("est_vram_mb", DEFAULT_VRAM_MB) <= 0
    preferred = task.get("preferred_node")
    require = task.get("require_node")  # HARD pin — never falls back
    blocked = _blocked_nodes_for_task(task)
    launch_failed = _launch_failed_nodes_for_task(task)
    if blocked:
        nodes = [n for n in nodes if n["name"] not in blocked]

    def _candidates_for_node(n):
        """Return list of (score, name, gpu_idx) candidates this node can offer (may be empty)."""
        if not n["alive"]: return []
        # Phase 2.3 P1 fix: slurm-managed nodes defer to slurm's own queue — no local
        # capacity check. Without this, login nodes with no GPU (probe gpus=[]) or busy
        # nodes never emit a candidate and the task stays queued in scheduleurm forever,
        # never reaching sbatch. Score uses 9999 in primary key so any local-fitting
        # candidate (whose primary keys are 0 or 1) ranks ahead — slurm only wins if no
        # local node fits OR the task explicitly requires/prefers a slurm node.
        # Phase 2.16: but throttle when this slurm node already has SLURM_MAX_PENDING_PER_NODE
        # of OUR tasks pending — keeping the rest in scheduleurm's queue lets them spread
        # to whichever slurm node frees up next, instead of piling on one host.
        if not _BACKEND.requires_local_capacity_check(n["name"]):
            pending = n.get("slurm_pending_count", 0)
            cap = _slurm_max_pending_for_node(n["name"])
            if pending >= cap:
                return []  # throttled — let pending drain or another node pick up
            return [((9999,), n["name"], None)]
        node_info = NODES[n["name"]]
        ok, _why = _node_resources_ok(task, n, node_info)
        if not ok: return []
        out = []
        if cpu_only:
            score = (0, -n["free_cpu"], -n["free_ram_mb"])
            out.append((score, n["name"], None))
        else:
            for g in n["gpus"]:
                if not _gpu_fits(task, g, node_info): continue
                # Best-fit (Codex follow-up): prior worst-fit (`-g["free_mb"]`) preferred the
                # GPU with MOST free VRAM → spread small tasks across empty cards, leaving no
                # contiguous space for a future big task. New scoring picks the smallest
                # fragment that still fits:
                #   primary key:   warm_first  → 0 if card already has another task, else 1.
                #                  (Empty cards get deprioritized so they're saved for big
                #                   tasks that wouldn't fit on a warm card past the 1/3 rule.)
                #   secondary key: leftover after placement, ascending — the tightest fit wins.
                fits_remaining = g["free_mb"] - (task.get("est_vram_mb") or 0)
                warm_first = 0 if g["used_mb"] >= 200 else 1
                score = (warm_first, fits_remaining)
                out.append((score, n["name"], g["idx"]))
        return out

    def _search(search_nodes):
        # 0) Hard pin — refuse to place anywhere except `require` node.
        if require:
            for n in search_nodes:
                if n["name"] == require:
                    cands = _candidates_for_node(n)
                    if cands:
                        cands.sort()
                        return cands[0][1], cands[0][2]
                    return None  # required node not ready — wait, do not fall back
            return None

        # 1) Try preferred node first (if specified and alive).
        if preferred:
            for n in search_nodes:
                if n["name"] == preferred:
                    cands = _candidates_for_node(n)
                    if cands:
                        cands.sort()
                        return cands[0][1], cands[0][2]
                    # preferred is alive but full — fall through to fallback search

        # 2) Fallback: scan all nodes (excluding the already-tried preferred if it failed).
        cands = []
        for n in search_nodes:
            if preferred and n["name"] == preferred:
                continue  # already tried above
            cands.extend(_candidates_for_node(n))
        if not cands: return None
        cands.sort()
        return cands[0][1], cands[0][2]

    # Launch-failed nodes are a soft block for unpinned tasks: prefer fresh nodes, but fall
    # back to retrying failed nodes if every viable node has already failed.
    if launch_failed and not require:
        fresh_nodes = [n for n in nodes if n["name"] not in launch_failed]
        placement = _search(fresh_nodes)
        if placement:
            return placement
    return _search(nodes)

# Back-compat alias — pre-resource-model code calls fits() directly.
def fits(task, gpu, node_info):
    return _gpu_fits(task, gpu, node_info)

# ---------- pre-launch checks ----------
def precheck_git(task):
    """Verify git repo is clean locally and synced with target node. Returns (ok, reason).
    Local-dirty is a WARNING (ok=True, reason starts with 'warn:') — the dispatcher logs it
    but launches anyway. Reasons that BLOCK (ok=False): git tooling error, repo missing on
    remote, or hash mismatch with remote (those produce broken/un-reproducible runs).
    The dirty warning is the user's responsibility to clear if reproducibility matters."""
    repo = task.get("git_repo")
    if not repo:
        return True, "no git check requested"
    try:
        local_hash = subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"], text=True, timeout=5
        ).strip()
        local_dirty = bool(subprocess.check_output(
            ["git", "-C", repo, "status", "--porcelain"], text=True, timeout=5
        ).strip())
    except Exception as e:
        return False, f"local git check failed: {e}"
    warnings = []
    if local_dirty:
        warnings.append(f"warn: local repo {repo} is dirty (uncommitted changes — run will use working tree state, not the committed commit)")
    if NODES[task["node"]]["host"] is None:
        # Local node — no remote sync to verify
        return True, "; ".join(warnings) or "ok"
    try:
        rc, out, _ = run_on(task["node"], f"git -C {shlex.quote(repo)} rev-parse HEAD", timeout=10, check=False)
        if rc != 0:
            return False, f"remote repo {repo} missing on {task['node']} (run git clone first)"
        remote_hash = out.strip()
        if remote_hash != local_hash:
            # Hash mismatch is still a BLOCK — diff between local committed and remote committed
            # means the script the user EDITED isn't what the remote will run. Distinct from
            # local-dirty (where uncommitted edits exist but committed-vs-remote is still aligned).
            return False, f"git mismatch: local={local_hash[:8]} {task['node']}={remote_hash[:8]} — push & pull first"
    except Exception as e:
        return False, f"remote git check failed: {e}"
    return True, "; ".join(warnings) or "ok"

CKPT_EXTS = ("pt", "pth", "pkl", "ckpt", "bin", "safetensors", "npy", "npz", "h5", "hdf5", "tar")
RESUME_SAFE_NAME_RE = re.compile(
    r"(checkpoint|ckpt|resume|state|snapshot|epoch|step|iter|iteration)",
    re.IGNORECASE,
)
RESUME_UNSAFE_NAME_RE = re.compile(
    r"(^|[_\-.])(model_?final|final_?model|final|best|buffer|replay|rollout|metrics?|results?|eval|train_?log)([_\-.]|$)",
    re.IGNORECASE,
)

def _resume_candidate_is_safe(path: str, explicit_glob: bool = False) -> bool:
    """Default resume selection should pick training-state checkpoints, not output artifacts.

    A restrictive user glob is treated as explicit intent, so it can still select project-specific
    names such as `model_final.pt` if the user asks for that exact file."""
    base = os.path.basename(path or "")
    ext = base.rsplit(".", 1)[-1].lower() if "." in base else ""
    if ext not in CKPT_EXTS:
        return False
    if explicit_glob:
        return True
    if RESUME_UNSAFE_NAME_RE.search(base):
        return False
    return bool(RESUME_SAFE_NAME_RE.search(base))

def find_resume(task):
    """Latest checkpoint in task['ckpt_dir'] on target node, by mtime. None if no dir / no files.

    Filters results by extension whitelist of known torch/tf/jax/numpy ckpt formats. Without
    this, default glob `*` matched ANY file (e.g. train_log.csv) which then got passed as
    --resume_from <path> → torch.load() blew up with EOFError.

    With the default glob, also require checkpoint-looking names and skip common output artifacts
    (`model_final.pt`, `buffer*.pkl`, metrics/results/eval dumps). Those files can have checkpoint
    extensions but often lack optimizer/RNG/replay state and are not safe training resume targets.
    A non-default --ckpt-glob is treated as explicit user intent and only uses the extension filter."""
    ckpt_dir = task.get("ckpt_dir")
    if not ckpt_dir: return None
    pattern = task.get("ckpt_glob", "*") or "*"
    explicit_glob = pattern != "*"
    script = r'''
import glob, os, re, sys
ckpt_dir, pattern, explicit = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
exts = set(sys.argv[4].split(","))
safe_re = re.compile(sys.argv[5], re.I)
unsafe_re = re.compile(sys.argv[6], re.I)
def ok(path):
    base = os.path.basename(path)
    ext = base.rsplit(".", 1)[-1].lower() if "." in base else ""
    if ext not in exts:
        return False
    if explicit:
        return True
    if unsafe_re.search(base):
        return False
    return bool(safe_re.search(base))
paths = [p for p in glob.glob(os.path.join(ckpt_dir, pattern)) if os.path.isfile(p) and ok(p)]
paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
if paths:
    print(paths[0])
'''
    cmd = (
        f"python3 - {shlex.quote(ckpt_dir)} {shlex.quote(pattern)} "
        f"{'1' if explicit_glob else '0'} {shlex.quote(','.join(CKPT_EXTS))} "
        f"{shlex.quote(RESUME_SAFE_NAME_RE.pattern)} {shlex.quote(RESUME_UNSAFE_NAME_RE.pattern)}"
        f" <<'PY'\n{script}\nPY"
    )
    try:
        rc, out, _ = run_on(task["node"], cmd, timeout=10, check=False)
    except Exception:
        return None
    if rc != 0:
        return None
    out = out.strip()
    return out or None

# ---------- launch ----------
_PYTHON_TOKEN_RE = re.compile(
    r'(^|[\s;&|`(])'                  # word boundary: start, whitespace, or shell separator
    r'((?:[A-Za-z0-9_./~-]+/)?'        # optional path prefix
    r'(?:python|python3)(?:\d+\.\d+)?)' # python / python3 / python3.11
    r'(\s+)'                           # whitespace AFTER the python token
)

def _inject_python_u(cmd: str) -> str:
    """Insert `-u` after every python invocation token that doesn't already have one.

    Idempotent: `python -u foo.py` stays as-is; `python -uXY foo.py` (any -u-prefixed flag) also
    stays as-is. Handles wrappers like `conda run -n env python script.py` and bare `python -m mod`.
    Doesn't try to parse complex shell pipelines beyond the first python on each side of `;|&`.
    """
    if not cmd:
        return cmd

    def _has_u_already(rest: str) -> bool:
        # rest starts right after the whitespace following the python token. Look at the first
        # word: if it's `-u` exactly, or starts with `-u` (e.g. `-u`, `-uX`), don't inject again.
        # Don't confuse with `--user` (which starts with `--u`) — we only catch single-dash `-u`.
        first = rest.lstrip().split(None, 1)[0] if rest.strip() else ""
        return first == "-u" or (first.startswith("-u") and not first.startswith("--"))

    out_parts = []
    last_end = 0
    for m in _PYTHON_TOKEN_RE.finditer(cmd):
        start, end = m.span()
        rest_after_match = cmd[end:]
        if _has_u_already(rest_after_match):
            continue
        # Splice in `-u ` after the trailing whitespace
        out_parts.append(cmd[last_end:end])
        out_parts.append("-u ")
        last_end = end
    if not out_parts:
        return cmd  # nothing to change
    out_parts.append(cmd[last_end:])
    return "".join(out_parts)

def _maybe_wrap_docker(task: dict, inner: str, cwd: str,
                       gpu_runtime_env: Optional[str] = None) -> tuple[str, Optional[str]]:
    """If task's env_spec resolves to docker, wrap inner cmd in `docker run ...`.

    Returns (wrapped_or_inner, error_or_None). When error is non-None, caller (launch())
    must fail the launch with that message — used for explicit `docker:IMAGE` requests
    where docker is unavailable / image push fails. For `auto` mode, returns (inner, None)
    on docker absence (graceful fallback to host env).

    Strategy resolution:
      - 'none' (default): return (inner, None) unchanged.
      - 'docker:IMAGE' (explicit): MUST succeed. Probes daemon, pushes image if needed,
        wraps. Any failure → (inner, error_msg) so launch fails fast. Codex review
        caught the silent host-fallback footgun.
      - 'docker' (no image, but image field set): same as docker:IMAGE.
      - 'auto': probe target. If docker daemon accessible AND image is set, use docker;
        else fall back to 'none' silently (graceful per design).

    Side effect: when docker is used, sets task["container_name"] so cancel/kill paths
    can `docker kill <name>` instead of fighting with containerd-shim PID isolation.
    """
    if env_deploy is None:
        # Module load failed entirely. Treat explicit docker as fatal, auto as graceful.
        spec = (task.get("env_spec") or "none").lower()
        if spec.startswith("docker") and spec != "auto":
            return (inner, "env_deploy module not loadable; cannot honor --env-spec docker")
        return (inner, None)
    spec = task.get("env_spec") or "none"
    image = task.get("image") or ""
    try:
        kind, spec_image = env_deploy.parse_env_spec(spec)
    except ValueError as e:
        return (inner, f"invalid env_spec {spec!r}: {e}")
    if kind == "none":
        return (inner, None)
    if kind == "conda":
        # Conda path: no cmd-wrapping needed — user's cmd already references absolute python
        # path that should now exist on target (preload rsync'd it). Bare `inner` is correct.
        # If env didn't preload (e.g., target down), launch will fail with ENV_MISSING and
        # heal flow takes over — no different from legacy `none` behavior.
        return (inner, None)
    chosen_image = spec_image or image
    node = task.get("node")
    node_host = NODES.get(node, {}).get("host") if node else None
    explicit = (kind == "docker")
    if not chosen_image:
        if explicit:
            return (inner, "--env-spec docker requires --image (or 'docker:IMAGE' inline)")
        return (inner, None)  # auto without image → graceful fallback
    if not env_deploy.has_docker(run_on, node, timeout=8):
        if explicit:
            return (inner, f"--env-spec docker requested but `docker info` failed on {node}")
        return (inner, None)  # auto → graceful fallback
    # Image presence + digest check at launch. Codex P1 follow-up: previously this only
    # checked tag presence, not digest. If preload had failed (network blip, ssh down)
    # but remote had a stale tag from a prior push, launch would silently run STALE image
    # against newer local code. Now: fetch local digest, pass to has_image so drift is
    # caught here even if preload missed. Push synchronously on drift (push_image inside
    # lock window — same scope as the existing missing-image branch; preload already
    # absorbed the bulk of this in the common case so launch-side push is rare).
    if node_host:
        local_digest = env_deploy.get_image_digest(run_on, "local", chosen_image)
        if not env_deploy.has_image(run_on, node, chosen_image, local_digest=local_digest):
            ok, msg = env_deploy.push_image(node_host, chosen_image, timeout_s=1800)
            if not ok:
                err = (f"docker image push of {chosen_image} to {node} failed (drift or missing): "
                       f"{msg[:200]}")
                if explicit:
                    return (inner, err)
                return (inner, None)  # auto → graceful fallback even on push failure
    container_name = f"sched-{task.get('id') or 'unknown'}"
    task["container_name"] = container_name
    wrapped = env_deploy.wrap_cmd_docker(
        inner=inner,
        image=chosen_image,
        cwd=cwd,
        gpu_idx=task.get("gpu_idx"),
        extra_env=task.get("extra_env") or {},
        container_name=container_name,
        memory_mb=task.get("ram_mb"),
        cpus=task.get("cpu_cores"),
        # Phase 2.6: SlurmBackend passes "CUDA_VISIBLE_DEVICES" so docker pins to
        # whatever GPU slurm allocated at runtime (rather than scheduleurm's
        # gpu_idx, which is None for slurm-routed tasks per Phase 2.3).
        gpu_runtime_env=gpu_runtime_env,
    )
    return (wrapped, None)


# ---------- backend abstraction (Phase 1) ----------
# Backend isolates the "how a task gets started/killed/probed on a node" decisions
# from scheduler policy (placement, history, dedup, diagnose). Single backend in
# Phase 1 (LocalBackend = ssh + nohup). Phase 2 will add SlurmBackend (sbatch/squeue
# /scancel) and Phase 3 will add MultiUserLocalBackend (cooperative shared state).
# The top-level functions launch() / _kill_task_processes() / check_running() /
# _batch_check_running() are kept as thin wrappers so existing call sites and tests
# don't need to change — only the bodies move.

class Backend:
    """Abstract: how we make a task run on a node.

    Implementations must be stateless (or hold only ephemeral state) — the source-
    of-truth task records live in queue.json, accessed under state_lock.
    """
    name = "abstract"

    def requires_local_capacity_check(self, node: str) -> bool:
        """Should pick_placement gate this node on local CPU/RAM/VRAM availability?

        - LocalBackend → True: scheduleurm IS the placement decider, so we MUST verify
          the task fits before launching (otherwise it OOMs the host or contends).
        - SlurmBackend → False: slurm has its own queue. The login node may have no
          GPU at all (gpus=[] from probe), or all GPUs may be busy with other slurm
          users — neither of which prevents slurm from accepting and queueing the
          job. Scheduler must NOT gate slurm submissions on instant local capacity,
          or jobs would get stuck queued in scheduleurm forever, never reaching sbatch.

        Default True (safe). Phase 2.3 fix.
        """
        return True

    def launch(self, task: dict) -> tuple[bool, str]:
        """Submit task to its assigned node.

        Mutates task in-place: sets status='running', started_at, log_path,
        peak_*_mb=0, plus backend-specific tracking handles (remote_pids /
        container_main_pid for Local; slurm_job_id for Slurm).

        Returns (ok, msg). On failure task should be re-queueable: caller will
        flip status back to 'queued' and retry/escalate per failure category.
        """
        raise NotImplementedError

    def kill(self, task: dict, timeout: int = 15) -> tuple[bool, str]:
        """Best-effort terminate. SIGTERM first, then SIGKILL. Idempotent — calling
        on an already-dead task should still return ok. Returns (ok, msg)."""
        raise NotImplementedError

    def batch_probe(self, state: dict) -> dict:
        """One round-trip per node, batch-probe liveness + current resource use for
        ALL tasks on this backend's nodes that are in status='running'.

        Returns: {task_id: {
            'state': 'alive' | 'dead' | 'unknown',
            'alive_pids': list[int],   # PIDs verified alive (incl. expanded descendants); [] OK
            'vram_mb': int,            # current VRAM total across alive PIDs; 0 if unknown
            'ram_mb':  int,            # current RSS total in MB; 0 if unknown
            'pcpu':    float,          # current %cpu total; 0.0 if unknown
        }}.
        Caller folds the deltas into peak_*_mb (max-tracking), updates alive_pids,
        and runs transition + diagnose + history-fold policy.
        """
        raise NotImplementedError


class LocalBackend(Backend):
    """ssh + nohup + setsid pattern. Tracks tasks by host-visible PIDs.

    For docker-wrapped tasks, the captured PID is the actual container main proc
    (resolved via `docker inspect --format {{.State.Pid}}` after launch) — not
    the docker-run client PID, which would be detached from the real proc tree
    by containerd-shim.
    """
    name = "local"

    def launch(self, task: dict) -> tuple[bool, str]:
        """Start the task via nohup, capture remote PID. Mutates task in-place. Returns (ok, msg)."""
        log_path = f"{STATE_DIR}/logs/{task['id']}.log" if NODES[task["node"]]["host"] is None \
                   else f"/tmp/sched_{task['id']}.log"
        cwd = task["cwd"]
        inner = task["cmd"]
        # `-u` injection for python invocations: without unbuffered stdout, python's full-buffering
        # mode (when stdout is a file) holds output in a 4KB block until process exit OR explicit
        # flush. If the process is SIGKILL'd (eviction, OOM, watcher kill) BEFORE the buffer fills,
        # the log ends up 0 bytes — which trips the diagnose's "log only 0B → crash" rule even when
        # the process actually saved its model and was about to print "Final model saved". Footgun:
        # AWAC s123/s789 trained 100k steps + saved awac_final.pt, were marked failed solely because
        # log buffer never flushed. Auto-inject -u so this can't bite again.
        inner = _inject_python_u(inner)
        # Resume injection: if find_resume() located a checkpoint AND submit declared --resume-flag,
        # append `<flag> <ckpt_path>` to the cmd. Without this the resume_from metadata is dead —
        # the script never sees the path. Empty resume_flag (default) opts out: cmd unchanged.
        resume_path = task.get("resume_from")
        resume_flag = task.get("resume_flag") or ""
        if resume_path and resume_flag:
            inner = f"{inner} {resume_flag} {shlex.quote(resume_path)}"
        # Env-deploy wrapping (docker). Done AFTER -u + resume injection so the inner shell cmd
        # is fully assembled before being wrapped in `docker run`. For env_spec='none' this is a
        # no-op; for 'auto', probes target and falls back to 'none' if docker isn't accessible;
        # for explicit 'docker' returns an error that fails the launch fast (per Codex review,
        # silent host fallback was unsafe).
        inner, docker_err = _maybe_wrap_docker(task, inner, cwd)
        if docker_err:
            return False, docker_err
        # Pre-flight: cwd must exist on target node. Skips a wasted launch + 2-3 retry cycles +
        # eventual ENV_MISSING escalation when a node simply doesn't have the repo synced.
        try:
            rc_cwd, _, err_cwd = run_on(task["node"], f"test -d {shlex.quote(cwd)}", timeout=10, check=False)
        except Exception as e:
            rc_cwd, err_cwd = 1, str(e)
        if rc_cwd != 0:
            return False, f"cwd missing on {task['node']}: {cwd}"
        # Set CUDA_VISIBLE_DEVICES. For GPU tasks: pin to assigned GPU (will appear as device 0 inside).
        # For CPU-only tasks (gpu_idx=None): set empty string so CUDA truly sees no GPUs — the literal
        # string "None" is NOT a valid CUDA value and would not disable GPU access.
        gpu_idx = task.get("gpu_idx")
        if gpu_idx is None:
            env_prefix = 'export CUDA_VISIBLE_DEVICES=""; '
        else:
            env_prefix = f"export CUDA_VISIBLE_DEVICES={gpu_idx}; "
        if task.get("extra_env"):
            for k, v in task["extra_env"].items():
                env_prefix += f"export {k}={shlex.quote(v)}; "
        # setsid creates a new session + process group leader, so `cancel --force`'s `kill -- -<pid>`
        # reliably catches every worker child. The </dev/null is so the launched process doesn't
        # inherit ssh's stdin pipe (which would otherwise keep the ssh-side bash alive).
        full = (f"mkdir -p {shlex.quote(os.path.dirname(log_path))}; "
                f"cd {shlex.quote(cwd)} && {env_prefix} "
                f"setsid bash -c {shlex.quote(inner)} > {shlex.quote(log_path)} 2>&1 < /dev/null & echo PID=$!")
        try:
            rc, out, err = run_on(task["node"], full, timeout=20, check=False)
            if rc != 0:
                return False, f"launch rc={rc}: {err.strip()[:200]}"
            pid = None
            for line in out.splitlines():
                line = line.strip()
                if line.startswith("PID="):
                    try: pid = int(line[4:])
                    except ValueError: pass
            if not pid:
                return False, f"could not parse PID from launch output: {out[:200]}"
            task["remote_pids"] = [pid]
            task["process_group"] = pid
            task["log_path"] = log_path
            task["status"] = "running"
            task["started_at"] = time.time()
            task["peak_vram_mb"] = 0
            task["peak_ram_mb"] = 0
            # Docker container PID resolution. The PID we just captured is the host-side bash
            # that ran `docker run` — NOT the python proc inside the container. With containerd-shim
            # the container's actual proc tree is detached, so liveness checks (`kill -0`),
            # peak-RAM tracking via `ps -o rss`, and adopt-dedup based on `remote_pids` all break.
            # Codex review caught this. Fix: ask docker for the container's main PID, append it
            # to remote_pids so the existing tracking machinery sees the right host-visible
            # process. nvidia-smi compute-apps already returns container-proc PIDs as host PIDs
            # via nvidia-container-toolkit, so once remote_pids has the right value, peak_vram
            # tracking + auto-adopt-dedup both light up correctly.
            cname = task.get("container_name")
            if cname:
                container_pid = None
                for _ in range(6):  # ~3s total: container start can take a moment
                    try:
                        rc2, out2, _ = run_on(
                            task["node"],
                            f"docker inspect --format '{{{{.State.Pid}}}}' {shlex.quote(cname)} 2>/dev/null",
                            timeout=4, check=False,
                        )
                    except Exception:
                        rc2, out2 = 1, ""
                    s = out2.strip() if rc2 == 0 else ""
                    if s.isdigit() and int(s) > 0:
                        container_pid = int(s)
                        break
                    time.sleep(0.5)
                if container_pid:
                    # Replace the docker-run bash PID with the actual container PID so liveness +
                    # peak tracking lock onto the real proc. Keep process_group=launcher_pid so
                    # `kill -- -<pgid>` still catches the docker client too if needed.
                    task["remote_pids"] = [container_pid]
                    task["container_main_pid"] = container_pid
            return True, f"pid={pid}" + (f" container={cname}@{task.get('container_main_pid')}" if cname else "")
        except Exception as e:
            return False, f"launch exception: {e}"

    def kill(self, task: dict, timeout: int = 15) -> tuple[bool, str]:
        """Best-effort terminate for a tracked task. SIGTERM first, then SIGKILL for both
        process groups and explicit PIDs. For docker-wrapped tasks, also `docker kill <name>`
        BEFORE the host PID kills — host `kill <docker run pid>` doesn't reliably stop the
        container because containerd-shim isolates the actual proc tree. Codex review caught
        this. Returns (ok, msg); callers still decide state changes."""
        node = task.get("node")
        if not node:
            return False, "no node"
        pids = [int(p) for p in _task_pids(task) if p]
        pgids = _task_process_groups(task)
        container = task.get("container_name")
        if not pids and not pgids and not container:
            return False, "no pids"
        parts = []
        if container:
            # SIGTERM first via `docker stop` (10s grace), then SIGKILL via `docker kill`.
            # Both are no-ops if the container already exited; `|| true` keeps the pipeline alive.
            parts.append(f"docker stop -t 5 {shlex.quote(container)} 2>/dev/null || true")
        parts += [f"kill -- -{g} 2>/dev/null" for g in pgids]
        parts += [f"kill {p} 2>/dev/null" for p in pids]
        parts.append("sleep 1")
        if container:
            parts.append(f"docker kill {shlex.quote(container)} 2>/dev/null || true")
        parts += [f"kill -9 -- -{g} 2>/dev/null" for g in pgids]
        parts += [f"kill -9 {p} 2>/dev/null" for p in pids]
        parts.append("true")
        cmd = "; ".join(parts)
        try:
            run_on(node, cmd, timeout=timeout, check=False)
            bits = []
            if container: bits.append(f"container={container}")
            if pgids: bits.append(f"pgids={pgids}")
            if pids: bits.append(f"pids={pids}")
            return True, " ".join(bits)
        except Exception as e:
            return False, str(e)[:200]

    def batch_probe(self, state: dict) -> dict:
        """ONE ssh per node to probe liveness + VRAM + RSS + %CPU for ALL tasks on this
        backend's nodes. Returns {task_id: probe_dict} with backend-agnostic shape (see
        Backend.batch_probe docstring).

        For LocalBackend, the per-node ssh runs `kill -0` + `awk` against /proc/<pid>/status
        for liveness (zombie-aware), `nvidia-smi --query-compute-apps` for VRAM, and `ps -eo
        pid,ppid,rss,pcpu` for the rest of the process tree. Descendant expansion via ppid_of
        catches setsid-wrapped workers that don't appear in the recorded remote_pids list.
        """
        by_node = {}        # node -> [(task, pids), ...]
        pids_per_node = {}  # node -> set of all pids across tasks on this node
        for t in state["tasks"]:
            if t["status"] != "running": continue
            node = t.get("node")
            if not node: continue
            pids = _task_pids(t)
            if not pids: continue
            by_node.setdefault(node, []).append((t, pids))
            pids_per_node.setdefault(node, set()).update(pids)
        results: dict = {}
        if not by_node:
            return results

        def _probe(node):
            all_pids = sorted(pids_per_node[node])
            # Zombie guard (Codex P1): kill -0 returns 0 for zombies too. Augment with /proc state
            # check: only count ALIVE if State is NOT Z (zombie) or X (dead).
            pid_checks = "; ".join(
                f"kill -0 {p} 2>/dev/null && "
                f"awk '/^State:/{{s=$2}} END{{if(s!=\"Z\" && s!=\"X\") print \"A{p}\"}}' "
                f"/proc/{p}/status 2>/dev/null"
                for p in all_pids
            )
            cmd = (f"({pid_checks}; true); echo '===VRAM==='; "
                   f"nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null; "
                   f"echo '===PSALL==='; "
                   f"ps -eo pid=,ppid=,rss=,pcpu= 2>/dev/null; true")
            try:
                rc, out, _ = run_on(node, cmd, timeout=30, check=False)
                return out if rc == 0 else None
            except Exception:
                return None

        nodes_list = list(by_node.keys())
        with ThreadPoolExecutor(max_workers=len(nodes_list)) as ex:
            outputs = dict(zip(nodes_list, ex.map(_probe, nodes_list)))

        for node, out in outputs.items():
            if out is None:
                # ssh failed for this node — emit 'unknown' for every task on it so caller leaves
                # state alone. Without this, a transient ssh blip would silently treat tasks as dead.
                for t, _ in by_node[node]:
                    results[t["id"]] = {"state": "unknown", "alive_pids": [],
                                        "vram_mb": 0, "ram_mb": 0, "pcpu": 0.0}
                continue
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            vram_sep = lines.index("===VRAM===") if "===VRAM===" in lines else len(lines)
            ps_sep = lines.index("===PSALL===") if "===PSALL===" in lines else len(lines)
            alive_roots = {int(l[1:]) for l in lines[:vram_sep] if l.startswith("A") and l[1:].isdigit()}
            vram_per_pid = {}
            for pl in lines[vram_sep+1:ps_sep]:
                parts = [x.strip() for x in pl.split(",")]
                if len(parts) < 2: continue
                try: ppid_, mb = int(parts[0]), int(parts[1])
                except ValueError: continue
                vram_per_pid[ppid_] = vram_per_pid.get(ppid_, 0) + mb
            rss_per_pid, pcpu_per_pid, ppid_of = {}, {}, {}
            for rl in lines[ps_sep+1:]:
                bits = rl.split()
                if len(bits) < 4: continue
                try:
                    p_, parent_, rss_kb = int(bits[0]), int(bits[1]), int(bits[2])
                    pc = float(bits[3])
                except ValueError: continue
                ppid_of[p_] = parent_
                rss_per_pid[p_] = rss_kb // 1024
                pcpu_per_pid[p_] = pc

            for t, pids in by_node[node]:
                pid_set = set(pids)
                # Descendant expansion: setsid bash wrapper means the recorded PID is the
                # process-group leader; the real Python proc + workers are descendants. Count
                # the whole live tree for liveness + resources, otherwise RAM/CPU/VRAM are badly
                # under-reported and child workers look external.
                expanded_pid_set = pid_set | _descendants_of(pid_set, ppid_of)
                this_alive = expanded_pid_set & (alive_roots | set(rss_per_pid))
                if not this_alive:
                    results[t["id"]] = {"state": "dead", "alive_pids": [],
                                        "vram_mb": 0, "ram_mb": 0, "pcpu": 0.0}
                    continue
                results[t["id"]] = {
                    "state": "alive",
                    "alive_pids": sorted(this_alive),
                    "vram_mb": sum(vram_per_pid.get(p, 0) for p in this_alive),
                    "ram_mb":  sum(rss_per_pid.get(p, 0) for p in this_alive),
                    "pcpu":    sum(pcpu_per_pid.get(p, 0.0) for p in this_alive),
                }
        return results


class SlurmBackend(Backend):
    """Submit via `sbatch` and let slurm own placement, queueing, and resource isolation.

    Use case: target node is part of a real slurm cluster (or has slurm installed and
    pointed at localhost). Slurm handles cross-user contention, fairness, GPU pinning
    via cgroups — capabilities scheduleurm's LocalBackend doesn't have.

    What scheduleurm STILL owns (not deferred to slurm): signature dedup, p80 history
    estimation, resume injection, env-deploy (docker/conda) wrapping, skill/MCP UI.

    What slurm owns: actual queueing across users, cgroup-based memory/CPU limits,
    GPU enumeration via gres, walltime enforcement, scancel signaling.

    Peak metrics tracking: NOT implemented in v1 — slurm enforces declared limits via
    cgroups, so peak ≈ declared in practice. If sstat is available we could poll for
    finer-grained tracking, but that requires the slurm accounting plugin to be on,
    which many simple installs lack. v1 keeps it simple: liveness-only probe.
    """
    name = "slurm"

    def requires_local_capacity_check(self, node: str) -> bool:
        """Slurm has its own queue — scheduler must not gate on local capacity here.
        See Backend.requires_local_capacity_check docstring."""
        return False

    # Walltime defaults: slurm needs --time, and "no limit" is partition-default which
    # might be only 1 hour on some clusters. We pick a generous floor + EWMA-derived
    # ceiling so well-characterized signatures get a tight bound and unknowns don't
    # get killed for being underestimated.
    DEFAULT_WALLTIME_S = 24 * 3600       # for unknown signatures: 24h
    EWMA_WALLTIME_MULT = 3.0             # known sig: 3× historical EWMA
    MIN_WALLTIME_S = 3600                # never less than 1h regardless of EWMA
    MAX_WALLTIME_S = 7 * 24 * 3600       # never more than 7d

    @staticmethod
    def _walltime_for(task: dict) -> int:
        """Pick --time= value in seconds based on history EWMA, clamped to [MIN, MAX]."""
        sig = task.get("signature") or ""
        h = history_get(sig) or {}
        ewma = int(h.get("dur_s_ewma", 0))
        if ewma > 0:
            t = int(ewma * SlurmBackend.EWMA_WALLTIME_MULT)
        else:
            t = SlurmBackend.DEFAULT_WALLTIME_S
        return max(SlurmBackend.MIN_WALLTIME_S, min(SlurmBackend.MAX_WALLTIME_S, t))

    @staticmethod
    def _format_walltime(secs: int) -> str:
        """Format seconds as slurm's HH:MM:SS or D-HH:MM:SS."""
        days, rem = divmod(secs, 86400)
        hours, rem = divmod(rem, 3600)
        mins, _ = divmod(rem, 60)
        if days > 0:
            return f"{days}-{hours:02d}:{mins:02d}:00"
        return f"{hours:02d}:{mins:02d}:00"

    def _build_sbatch_script(self, task: dict, inner_cmd: str, log_path: str) -> str:
        """Build the sbatch script as a string. Streamed via stdin to `sbatch /dev/stdin`."""
        lines = ["#!/bin/bash"]
        lines.append(f"#SBATCH --job-name=scheduleurm-{task['id']}")
        lines.append(f"#SBATCH --output={log_path}")
        lines.append(f"#SBATCH --error={log_path}")
        cpu = int(task.get("cpu_cores") or DEFAULT_CPU_CORES)
        if cpu > 0:
            lines.append(f"#SBATCH --cpus-per-task={cpu}")
        ram = int(task.get("ram_mb") or DEFAULT_RAM_MB)
        if ram > 0:
            lines.append(f"#SBATCH --mem={ram}M")
        vram = int(task.get("est_vram_mb") or 0)
        if vram > 0:
            # Request 1 GPU; slurm's gres pinning sets CUDA_VISIBLE_DEVICES for us. We don't
            # request `gpu:N` for >1 GPU because scheduleurm tasks are single-GPU by design;
            # multi-GPU is the user's launcher's responsibility.
            lines.append("#SBATCH --gres=gpu:1")
        lines.append(f"#SBATCH --time={self._format_walltime(self._walltime_for(task))}")
        # Optional slurm-specific fields if user set them on the task
        for slurm_field, sbatch_flag in (
            ("slurm_partition", "--partition"),
            ("slurm_account",   "--account"),
            ("slurm_qos",       "--qos"),
        ):
            v = task.get(slurm_field)
            if v:
                lines.append(f"#SBATCH {sbatch_flag}={v}")
        # Body: cd cwd, then the (already _inject_python_u + _maybe_wrap_docker'd) inner cmd.
        # Note: slurm does its OWN CUDA_VISIBLE_DEVICES via gres binding, so we don't export
        # it here (and explicitly NOT inheriting it from the launching shell).
        lines.append("")
        if task.get("extra_env"):
            for k, v in task["extra_env"].items():
                lines.append(f"export {k}={shlex.quote(v)}")
        # Phase 2.5 P1 fix: cd must be fatal-on-failure. A bare `cd path` followed by
        # the inner cmd silently continues from $HOME (or wherever) if the compute node
        # doesn't see this path (NFS stale handle, cwd not propagated to compute, etc.).
        # The job appears to "run" but produces no output, hits no checkpoints, and
        # diagnose has no signal — log just has whatever bash printed about cd. Match
        # LocalBackend's `cd ... && cmd` semantics with an explicit guard that also
        # leaves a parseable error in the log so diagnose can route to ENV_MISSING.
        cwd_q = shlex.quote(task['cwd'])
        lines.append(
            f"cd {cwd_q} || {{ "
            f"echo \"scheduleurm: cwd not accessible on compute node: {task['cwd']}\" >&2; "
            f"exit 1; "
            f"}}"
        )
        lines.append(inner_cmd)
        return "\n".join(lines) + "\n"

    def launch(self, task: dict) -> tuple[bool, str]:
        # Phase 2.4 P1 fix: slurm-managed jobs run on a compute node (slurm-chosen),
        # but scheduler tails via the login node. /tmp is per-node-local on virtually
        # every cluster — writing to /tmp/sched_<id>.log on compute-N would leave the
        # login node tailing a non-existent path → diagnose sees 0 bytes → false-
        # classified as crash → wasteful re-queue.
        # Use a path that's guaranteed shared: under the user's cwd, which IS on a
        # shared FS (otherwise slurm couldn't run their code from there). The pre-flight
        # `test -d cwd` already passes; we add `mkdir -p .scheduleurm` to it so sbatch's
        # --output directive can write there.
        cwd = task["cwd"]
        log_path = (f"{STATE_DIR}/logs/{task['id']}.log"
                    if NODES[task["node"]]["host"] is None
                    else f"{cwd}/.scheduleurm/{task['id']}.log")
        inner = task["cmd"]
        inner = _inject_python_u(inner)
        resume_path = task.get("resume_from")
        resume_flag = task.get("resume_flag") or ""
        if resume_path and resume_flag:
            inner = f"{inner} {resume_flag} {shlex.quote(resume_path)}"
        # Env-deploy wrapping (docker) — Phase 2.6 P1 fix: pass gpu_runtime_env so the
        # docker `--gpus` arg is `device=$CUDA_VISIBLE_DEVICES` (literal, expanded by
        # bash at sbatch runtime) rather than `device=N` with a stale scheduleurm-picked
        # gpu_idx. Slurm's gres allocator decides the GPU at job start; the env var it
        # sets is the source of truth. Without this, GPU tasks get either:
        #   - no `--gpus` flag (gpu_idx=None per Phase 2.3 → CPU-only inside container,
        #     CUDA init fails or silently uses host-leaked GPUs), or
        #   - `--gpus device=0` while slurm allocated GPU 1 → wrong card, contention,
        #     potential hard fail if slurm's cgroup blocks GPU 0 access.
        # CPU-only slurm tasks (est_vram_mb=0): leave gpu_runtime_env=None so wrapper
        # emits no --gpus at all (existing behavior).
        gpu_runtime_env = "CUDA_VISIBLE_DEVICES" if int(task.get("est_vram_mb") or 0) > 0 else None
        inner, docker_err = _maybe_wrap_docker(task, inner, cwd, gpu_runtime_env=gpu_runtime_env)
        if docker_err:
            return False, docker_err

        # Pre-flight: cwd must exist on target node (same check as LocalBackend).
        # Pre-flight: cwd must exist on target node + ensure log dir exists. The mkdir
        # is on the LOGIN node here, but since cwd is on shared FS, the directory will
        # also be visible from the compute node when sbatch's --output= writes to it.
        # Without `mkdir -p`, sbatch's first write would fail because the parent dir
        # doesn't exist yet (slurm doesn't autocreate output parent dirs).
        log_dir = os.path.dirname(log_path)
        try:
            rc_cwd, _, _ = run_on(
                task["node"],
                f"test -d {shlex.quote(cwd)} && mkdir -p {shlex.quote(log_dir)}",
                timeout=10, check=False,
            )
        except Exception as e:
            rc_cwd = 1
        if rc_cwd != 0:
            return False, f"cwd missing or log_dir uncreatable on {task['node']}: {cwd}"

        script = self._build_sbatch_script(task, inner, log_path)
        # Pipe script via stdin to `sbatch /dev/stdin`. Phase 2.10 P1 fix: empirically
        # `sbatch -` (the "argv shorthand for stdin" form) is rejected on Ubuntu 24.04 /
        # slurm 23.11.4 with "Unable to open file -" — the package's argv parser doesn't
        # treat `-` as a stdin sentinel. `/dev/stdin` is the kernel-level stdin pipe and
        # works universally across slurm versions because slurm just opens it as a path.
        # No file is left on the compute node either way (it's the same pipe).
        host = NODES[task["node"]]["host"]
        try:
            if host is None:
                proc = subprocess.run(
                    ["sbatch", "/dev/stdin"], input=script,
                    capture_output=True, text=True, timeout=30,
                )
            else:
                proc = subprocess.run(
                    ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
                     host, "sbatch /dev/stdin"],
                    input=script, capture_output=True, text=True, timeout=30,
                )
            if proc.returncode != 0:
                return False, f"sbatch rc={proc.returncode}: {proc.stderr.strip()[:200]}"
            # sbatch stdout: "Submitted batch job 12345"
            job_id = None
            for tok in proc.stdout.split():
                if tok.isdigit():
                    job_id = int(tok)
                    break
            if not job_id:
                return False, f"could not parse job id from sbatch output: {proc.stdout[:200]}"
            task["slurm_job_id"] = job_id
            task["log_path"] = log_path
            task["status"] = "running"
            task["started_at"] = time.time()
            task["peak_vram_mb"] = 0
            task["peak_ram_mb"] = 0
            # Empty for slurm tasks — there's no host-visible PID for scheduleurm to track.
            # Liveness goes via squeue. _task_pids returns [] which is fine — kill/probe paths
            # use slurm_job_id directly.
            task["remote_pids"] = []
            return True, f"slurm_job_id={job_id}"
        except subprocess.TimeoutExpired:
            return False, "sbatch timeout"
        except Exception as e:
            return False, f"sbatch exception: {e}"

    def kill(self, task: dict, timeout: int = 15) -> tuple[bool, str]:
        job_id = task.get("slurm_job_id")
        if not job_id:
            return False, "no slurm_job_id"
        # `scancel <id>` is async (returns immediately, signals slurmctld). KillWait grace
        # period is set in slurm.conf (default 30s SIGTERM → SIGKILL). We don't escalate
        # ourselves — slurm handles it.
        try:
            run_on(task["node"], f"scancel {int(job_id)}", timeout=timeout, check=False)
            return True, f"scancel {job_id}"
        except Exception as e:
            return False, str(e)[:200]

    @staticmethod
    def _parse_size_to_mb(s: str):
        """Parse slurm-formatted memory string ('512000K' / '1G' / '800M' / '2T' / bare digits)
        into MB. Returns None on parse failure (so caller can skip silently).

        Slurm reports memory with K/M/G/T suffix. Bare numbers (no suffix) are KiB by sstat
        convention. We convert everything to MB (binary), rounded down to int."""
        s = (s or "").strip()
        if not s:
            return None
        unit = ""
        if s[-1].upper() in ("K", "M", "G", "T"):
            unit = s[-1].upper()
            num_str = s[:-1]
        else:
            num_str = s
        try:
            val = float(num_str)
        except ValueError:
            return None
        # Bare digits = KiB; suffixed = corresponding binary unit. All → MB.
        factor_to_mb = {"": 1.0 / 1024, "K": 1.0 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0 * 1024}.get(unit)
        if factor_to_mb is None:
            return None
        return int(val * factor_to_mb)

    def _query_sstat_peaks(self, node: str, job_ids: list) -> dict:
        """Query `sstat` for MaxRSS of running slurm jobs on `node`. Returns
        {job_id: ram_mb} for jobs where sstat returned parseable data. Missing entries
        mean the caller should skip peak-update for that task — sstat may legitimately
        be unavailable (accounting plugin off, recent submission not yet flushed, etc.)
        and we degrade silently rather than re-classify the task.

        Why sstat (not sacct): sstat reports LIVE peaks for running jobs and is cheaper.
        sacct is for completed jobs and lives in the accounting DB; v2.1 doesn't pull
        from it (one final sample missed at transition is acceptable; mid-run sampling
        every 60s captures the true peak well enough for p80 history)."""
        if not job_ids:
            return {}
        ids = ",".join(str(j) for j in job_ids)
        # -a includes ALL step records (.batch, .extern, .0, ...). Without -a, sstat shows
        # only the "main" step and MaxRSS comes back empty for batch jobs since their data
        # lives in <jid>.batch. Verified empirically on slurm 23.11.4 / Ubuntu 24.04 — bare
        # `sstat -j 4` returns nothing, `sstat -a -j 4` returns `4.batch|975.50M`.
        # -P pipe-delim, --noheader, format=JobID|MaxRSS. 2>/dev/null swallows errors when
        # sstat exists but accounting isn't configured (e.g. JobAcctGatherType=none).
        cmd = f"sstat -a -j {ids} -P --noheader --format=JobID,MaxRSS 2>/dev/null"
        try:
            rc, out, _ = run_on(node, cmd, timeout=10, check=False)
        except Exception:
            return {}
        if rc != 0 or not out.strip():
            return {}
        peaks: dict = {}
        for line in out.splitlines():
            bits = line.strip().split("|")
            if len(bits) < 2:
                continue
            # JobID can be "12345.batch" / "12345.extern" / "12345.0" (per-step records);
            # we want the max across all steps, keyed by base job id.
            base = bits[0].split(".")[0]
            try:
                jid = int(base)
            except ValueError:
                continue
            mb = self._parse_size_to_mb(bits[1])
            if mb is None:
                continue
            peaks[jid] = max(peaks.get(jid, 0), mb)
        return peaks

    def batch_probe(self, state: dict) -> dict:
        """One squeue per node for liveness, then a best-effort sstat for live RAM peaks
        (Phase 2.1). VRAM/CPU still not tracked: VRAM lacks a portable slurm field across
        versions; CPU is enforced by slurm cgroup so tracking adds little value. Liveness
        map: RUNNING/PENDING/CONFIGURING/COMPLETING → alive; everything else → dead. If
        sstat fails (no accounting plugin / sstat not in PATH / parse error), ram_mb
        stays 0 and the caller's max-tracking is a no-op — same as v1 behavior."""
        by_node: dict = {}
        for t in state["tasks"]:
            if t["status"] != "running": continue
            jid = t.get("slurm_job_id")
            if not jid: continue
            by_node.setdefault(t["node"], []).append(t)
        results: dict = {}
        if not by_node:
            return results

        ALIVE_STATES = {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "RESIZING",
                        "REQUEUED", "SUSPENDED"}
        SUCCESS_STATES = {"COMPLETED"}
        FAILURE_STATES = {"FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY",
                          "CANCELLED", "PREEMPTED", "BOOT_FAIL", "DEADLINE",
                          "REVOKED", "SPECIAL_EXIT"}

        def _probe(node):
            ids_list = [t["slurm_job_id"] for t in by_node[node]]
            ids = ",".join(str(i) for i in ids_list)
            # squeue: liveness. `-h` no header, `-j` filter, `-t all` include finished
            # (squeue's default hides COMPLETED). `-o "%i %T"` minimal columns.
            cmd = f"squeue -h -j {ids} -t all -o '%i %T' 2>/dev/null"
            try:
                rc, out, _ = run_on(node, cmd, timeout=15, check=False)
                squeue_out = out if rc == 0 else None
            except Exception:
                squeue_out = None
            # sstat: live RAM peaks (Phase 2.1). Best-effort; failure → empty dict.
            # Same ssh round-trip would be ideal but sstat exits non-zero on jobs without
            # accounting data, polluting our error signal — so we keep them separate.
            sstat_peaks = self._query_sstat_peaks(node, ids_list) if squeue_out is not None else {}
            return (squeue_out, sstat_peaks)

        nodes_list = list(by_node.keys())
        with ThreadPoolExecutor(max_workers=len(nodes_list)) as ex:
            outputs = dict(zip(nodes_list, ex.map(_probe, nodes_list)))

        for node, (out, sstat_peaks) in outputs.items():
            if out is None:
                # ssh / squeue failed: emit 'unknown' so policy leaves state alone (don't
                # mistakenly transition tasks to dead because of a transient ssh blip).
                for t in by_node[node]:
                    results[t["id"]] = {"state": "unknown", "alive_pids": [],
                                        "vram_mb": 0, "ram_mb": 0, "pcpu": 0.0}
                continue
            seen = {}  # job_id -> state-string
            for line in out.splitlines():
                bits = line.strip().split()
                if len(bits) < 2: continue
                try:
                    seen[int(bits[0])] = bits[1].upper()
                except ValueError: continue
            for t in by_node[node]:
                jid = t["slurm_job_id"]
                slurm_state = seen.get(jid)
                if slurm_state is None:
                    # squeue doesn't know this job. Either it ran + got purged from accounting
                    # (slurm.conf MinJobAge), OR our job_id is stale. Treat as dead — the
                    # diagnose path will tail the log and decide success vs failure.
                    state_norm = "dead"
                    terminal_ok = None
                    terminal_reason = "slurm job absent from squeue; falling back to log diagnosis"
                elif slurm_state in ALIVE_STATES:
                    state_norm = "alive"
                    terminal_ok = None
                    terminal_reason = None
                    t["slurm_state"] = slurm_state
                elif slurm_state in SUCCESS_STATES:
                    state_norm = "dead"
                    terminal_ok = True
                    terminal_reason = f"slurm terminal state {slurm_state}"
                    t["slurm_state"] = slurm_state
                else:
                    # Unknown terminal-ish states are safer as failures than "maybe done":
                    # if Slurm did not say the job is alive or COMPLETED, scheduleurm should
                    # retry/escalate rather than silently drop work.
                    state_norm = "dead"
                    terminal_ok = False
                    known = slurm_state in FAILURE_STATES
                    label = slurm_state if known else f"UNRECOGNIZED:{slurm_state}"
                    terminal_reason = f"slurm terminal state {label}"
                    t["slurm_state"] = slurm_state
                # sstat peak (Phase 2.1): only fold for ALIVE tasks. Dead-state probes don't
                # fold peaks because the policy code calls history_record on transition with
                # whatever peak_ram_mb is at that point — sstat won't return useful data for
                # already-finished jobs anyway (that's sacct's domain).
                ram_mb = sstat_peaks.get(jid, 0) if state_norm == "alive" else 0
                results[t["id"]] = {"state": state_norm, "alive_pids": [],
                                    "vram_mb": 0, "ram_mb": ram_mb, "pcpu": 0.0,
                                    "backend_state": slurm_state,
                                    "terminal_ok": terminal_ok,
                                    "terminal_reason": terminal_reason}
        return results


class HybridBackend(Backend):
    """Per-node routing: slurm-detected nodes → SlurmBackend, else LocalBackend.

    Detection runs once per node per process lifetime (cached). Probes via
    `command -v sbatch` over ssh; ssh failure or missing tools default to local.
    Restarting the watcher re-runs detection.

    Why not pick at startup? NODES is static config but slurm presence is a runtime
    fact (a node can have slurm installed without scheduleurm config knowing). The
    cache keeps the perf cost to one ssh round-trip per node per process.
    """
    name = "hybrid"

    def __init__(self):
        self._local = LocalBackend()
        self._slurm = SlurmBackend()
        self._cache: dict = {}  # node_name -> 'slurm' | 'local'

    def requires_local_capacity_check(self, node: str) -> bool:
        """Per-node delegation: slurm-detected node → False (slurm queues); else True
        (LocalBackend's instant-capacity gate). Used by pick_placement to skip the
        CPU/RAM/VRAM-fits check on slurm nodes (Phase 2.3 P1 fix)."""
        return self._backend_for(node).requires_local_capacity_check(node)

    def _kind_for(self, node: str) -> str:
        """Return 'slurm' or 'local' for this node, cached after the FIRST DEFINITIVE
        answer. Phase 2.7 P1 fix: only cache definitive results (probe ssh succeeded
        AND we got either HAS_SLURM or NO_SLURM marker). Any failure mode (ssh
        exception, non-zero rc, missing marker due to bashrc spam, etc.) leaves the
        cache untouched, so the next call re-probes — a single ssh blip can no longer
        permanently mis-route a node to LocalBackend until watcher restart.

        Failure mode for THIS call: returns 'local' (the safe-loud default — at least
        the launch path will surface an error if slurm IS expected). One cycle of
        fallback is the worst case; next dispatch cycle re-probes and self-heals.
        """
        if node in self._cache:
            return self._cache[node]
        # Probe must ALWAYS emit a marker so we can distinguish:
        # - NO_SLURM: tools absent, definitively local-capable.
        # - HAS_SLURM: tools exist AND the controller answers a cheap squeue probe.
        # - SLURM_UNUSABLE: tools exist but controller/account/path is broken right now.
        #
        # The last case deliberately routes to SlurmBackend for this attempt but is not
        # cached. Falling back to LocalBackend on a Slurm node can launch jobs on a login
        # node and bypass cluster policy; a loud sbatch/squeue failure is safer.
        cmd = ("if command -v sbatch >/dev/null 2>&1 && "
               "command -v squeue >/dev/null 2>&1; then "
               "if squeue -h >/dev/null 2>&1; then echo HAS_SLURM; "
               "else echo SLURM_UNUSABLE; fi; "
               "else echo NO_SLURM; fi")
        try:
            rc, out, _ = run_on(node, cmd, timeout=5, check=False)
        except Exception:
            return "local"  # ssh broke — DON'T cache; next call re-probes
        if rc != 0:
            return "local"  # rc!=0 means our probe itself failed; don't cache
        if "HAS_SLURM" in out:
            self._cache[node] = "slurm"
            return "slurm"
        if "NO_SLURM" in out:
            self._cache[node] = "local"
            return "local"
        if "SLURM_UNUSABLE" in out:
            return "slurm"  # tools present; don't cache because controller/account may recover
        # Ambiguous output (output mangled by remote bashrc, MOTD, etc.) — don't cache.
        return "local"

    def _backend_for(self, node: str) -> Backend:
        return self._slurm if self._kind_for(node) == "slurm" else self._local

    def _backend_for_task(self, task: dict) -> Backend:
        """Route by what the task ACTUALLY has, not by the (mutable) per-node cache.

        Phase 2.8 P1 fix: previously, a node's cache flipping from 'local' → 'slurm'
        (e.g. Phase 2.7's re-probe finding slurm after a transient blip) caused
        already-running LocalBackend tasks (those have remote_pids, no slurm_job_id)
        to be re-routed to SlurmBackend.batch_probe, which skips them on
        `if not jid: continue` → tasks become forever-stuck zombies (never probed,
        never transitioned to terminal). Same hazard for kill — SlurmBackend.kill
        returns "no slurm_job_id" without doing anything.

        New rule: launch artifacts on the task itself are the source of truth.
        - slurm_job_id present → SlurmBackend (it launched the task)
        - remote_pids present → LocalBackend (it launched the task)
        - neither → queued, use per-node cache for the upcoming launch
        """
        if task.get("slurm_job_id"):
            return self._slurm
        if task.get("remote_pids"):
            return self._local
        node = task.get("node")
        if not node:
            return self._local  # no node yet (queued task): launch path will re-route
        return self._backend_for(node)

    def launch(self, task: dict) -> tuple[bool, str]:
        return self._backend_for_task(task).launch(task)

    def kill(self, task: dict, timeout: int = 15) -> tuple[bool, str]:
        return self._backend_for_task(task).kill(task, timeout=timeout)

    def batch_probe(self, state: dict) -> dict:
        """Split tasks per backend, probe each, merge results. Two ssh round-trips per node
        in the worst case (one local probe + one slurm probe), but each backend's batch_probe
        skips nodes with no relevant tasks so common case is one round-trip per node."""
        # Synthesize per-backend state subsets so each backend only sees its own tasks.
        local_tasks, slurm_tasks = [], []
        for t in state["tasks"]:
            if t["status"] != "running":
                continue
            if self._backend_for_task(t) is self._slurm:
                slurm_tasks.append(t)
            else:
                local_tasks.append(t)
        merged: dict = {}
        if local_tasks:
            merged.update(self._local.batch_probe({"tasks": local_tasks}))
        if slurm_tasks:
            merged.update(self._slurm.batch_probe({"tasks": slurm_tasks}))
        return merged


# Singleton: Phase 2 routes per-node via HybridBackend (slurm-detected → SlurmBackend,
# else LocalBackend). Phase 3 will swap in MultiUserLocalBackend at this assignment.
# Tests reference _BACKEND directly to verify backend identity and to swap in fakes.
_BACKEND: Backend = HybridBackend()


def launch(task):
    """Thin wrapper: delegates to the active Backend. See Backend.launch."""
    return _BACKEND.launch(task)

# ---------- subcommands ----------
def _cmd_looks_like_training(cmd: str) -> bool:
    """Heuristic: cmd invokes a training entry-point. Catches `train_*.py`, `/train.py`, `trainer.py`,
    and a few common framework patterns. False positives are acceptable — the user can override
    with --allow-cpu-training."""
    lower = (cmd or "").lower()
    return any(p in lower for p in (
        "train_", "/train.py", " train.py", "trainer.py",
        "/main_train", "run_train", "do_train",
        "h2o+_bus_main.py",
    ))

def _cmd_explicitly_cpu(cmd: str) -> bool:
    """Recognize app-level CPU flags. This is advisory only: scheduler-level CPU training
    still requires --allow-cpu-training so a stray `--device cpu` cannot silently consume
    the CPU partition for a training batch."""
    lower = (cmd or "").lower()
    norm = lower.replace("=", " ")
    cuda_empty = bool(re.search(
        r"(?:^|\s)(?:export\s+)?cuda_visible_devices\s*=\s*(?:\"\"|''|-1|;|$|\s)",
        lower,
    ))
    return any(p in norm for p in (
        "--device cpu", "--cpu-only", "--no-cuda", "--no-gpu", "--use-cpu",
        "device cpu",   # after = → space normalization
    )) or cuda_empty

def _task_looks_like_training(cmd: str, description: str = "") -> bool:
    """Broader task-level training heuristic used only when vram=0 would make a task CPU-only.
    False positives are intentionally acceptable: pass --allow-cpu-training for legitimate
    CPU training. False negatives are riskier because they silently bypass GPU scheduling."""
    if _cmd_looks_like_training(cmd):
        return True
    lower_desc = (description or "").lower()
    return any(p in lower_desc for p in (
        "baseline:", "train", "training",
        " iql", "iql ", "awac", "td3", "rlpd", "wsrl", "sac", "bc on",
    ))

def _cpu_training_policy_reason(cmd: str, description: str = "",
                                allow_cpu_training: bool = False, est_vram_mb: int = 0):
    """Return a refusal/block reason when a training-shaped task is marked CPU-only.

    Scheduler semantics are resource-first: vram=0 means CPU partition. App-level flags like
    `--device cpu` are not enough to prove intent because this exact footgun submitted a whole
    training batch to CPU. Legit CPU training must say so at scheduler level with
    --allow-cpu-training."""
    if not _task_looks_like_training(cmd, description):
        return None
    try:
        vram = int(est_vram_mb or 0)
    except Exception:
        vram = 0
    explicit_cpu = _cmd_explicitly_cpu(cmd)
    if allow_cpu_training:
        if explicit_cpu and vram > 0:
            return ("training-looking task has an explicit CPU device flag but scheduler vram>0 "
                    "would reserve a GPU while the program runs on CPU. Use --vram 0 with "
                    "--allow-cpu-training, or remove the CPU flag for GPU training.")
        return None
    if explicit_cpu and vram > 0:
        return ("training-looking task explicitly requests CPU in the inner cmd, but scheduler "
                "would reserve a GPU because vram>0. Remove/replace the CPU flag for GPU "
                "training, or use --vram 0 with --allow-cpu-training if CPU training is intentional.")
    if explicit_cpu:
        return ("training-looking task has vram=0 and an explicit CPU device flag; "
                "scheduler would run it CPU-only. Resubmit with --vram <MB> and a GPU device flag, "
                "or pass --allow-cpu-training if CPU training is intentional.")
    if vram > 0:
        return None
    return ("training-looking task has vram=0, so scheduler would run it CPU-only. "
            "Resubmit with --vram <MB>, or pass --allow-cpu-training if CPU training is intentional.")

def _cmd_has_resume_flag(cmd: str) -> bool:
    """Detect if cmd already contains a resume-style flag. Match common variants."""
    if not cmd:
        return False
    norm = cmd.replace("=", " ")
    return any(p in norm for p in (
        "--resume ", "--resume_from ", "--resume-from ",
        "--load_ckpt ", "--load-ckpt ", "--load_from ", "--load-from ",
        "--init_from ", "--init-from ",
        "--ckpt_path ", "--ckpt-path ",
        "--restore ", "--restore_from ", "--restore-from ",
    )) or norm.rstrip().endswith("--resume")

def _resume_capability_reason(cmd: str, ckpt_dir, resume_flag, allow_no_resume: bool):
    """Return a refusal reason when a training-looking cmd has --ckpt-dir set but no resume
    capability wired up. Without resume, an evicted/crashed/rebooted task relaunches at step 0
    even though ckpts exist on disk — silently throwing away hours of progress. Footgun: WSRL
    on 05-04 ran 50h to epoch 100, watcher false-marked done at reboot, would have requeued
    from step 0 had it run again. Submit-time fail-fast forces explicit decision."""
    if allow_no_resume:
        return None
    if not _cmd_looks_like_training(cmd):
        return None
    if not ckpt_dir:
        return None  # caught by the ckpt-dir guard upstream
    if _cmd_has_resume_flag(cmd):
        return None
    if resume_flag:  # scheduler will append '<flag> <ckpt_path>' on relaunch
        return None
    return ("training-looking task has --ckpt-dir but no resume flag in cmd nor --resume-flag at submit. "
            "On crash/eviction/reboot, relaunch starts from step 0 even though ckpts exist. "
            "Either: (a) add --resume / --resume_from <path> to the cmd, "
            "(b) pass --resume-flag '--resume_from' at submit (scheduler appends '<flag> <ckpt>' on relaunch), "
            "or (c) pass --allow-no-resume to override (e.g. script genuinely cannot resume — must accept replay loss).")

def _queued_cpu_training_block_reason(task):
    if task.get("status") != "queued":
        return None
    if task.get("auto_adopted") or task.get("adopted"):
        return None
    return _cpu_training_policy_reason(
        task.get("cmd", ""),
        task.get("description", ""),
        bool(task.get("allow_cpu_training", False)),
        int(task.get("est_vram_mb") or 0),
    )

def cmd_submit(args):
    # Pre-flight: refuse training-shaped cmd with vram=0 unless the scheduler-level override
    # is present. `--device cpu` in the inner command is not enough: that flag is precisely how
    # a GPU-intended training batch can accidentally land on CPU.
    cpu_training_reason = None
    submit_vram_for_policy = args.vram if args.vram is not None else DEFAULT_VRAM_MB
    cpu_training_reason = _cpu_training_policy_reason(
        args.cmd, args.description, bool(getattr(args, "allow_cpu_training", False)),
        submit_vram_for_policy,
    )
    if cpu_training_reason:
        print(f"REFUSED: cmd looks like training but scheduler CPU/GPU policy is inconsistent.", file=sys.stderr)
        print(f"  cmd: {args.cmd[:120]}", file=sys.stderr)
        print(f"  reason: {cpu_training_reason}", file=sys.stderr)
        print(f"  Either:", file=sys.stderr)
        print(f"    (a) submit with --vram <N> and remove/replace CPU device flags  (use GPU)", file=sys.stderr)
        print(f"    (b) pass --allow-cpu-training AND --cpu-training-justification '<reason>'", file=sys.stderr)
        sys.exit(2)
    # Friction layer: if --allow-cpu-training is set, require a non-trivial justification.
    # Default policy is GPU for training; CPU training should be the exception with a written
    # reason (e.g. "tiny baseline; GPU saturation acceptable for fairness"). Without this check,
    # `--allow-cpu-training` becomes a reflex bypass — exactly the bug that put 6 H2O+ R3
    # baselines onto CPU when they belonged on GPU.
    MIN_JUSTIFICATION_LEN = 30
    if (bool(getattr(args, "allow_cpu_training", False))
            and _task_looks_like_training(args.cmd, args.description)):
        just = (getattr(args, "cpu_training_justification", "") or "").strip()
        if len(just) < MIN_JUSTIFICATION_LEN:
            print(f"REFUSED: --allow-cpu-training requires --cpu-training-justification "
                  f"with ≥{MIN_JUSTIFICATION_LEN} chars of explanation.", file=sys.stderr)
            print(f"  cmd: {args.cmd[:120]}", file=sys.stderr)
            print(f"  Reason: training tasks should default to GPU. CPU training is the exception, "
                  f"not the rule. Document why GPU isn't right HERE so it's auditable later.", file=sys.stderr)
            print(f"  Example: --cpu-training-justification "
                  f"'tiny MLP, GPU saturated by other priority work, completion in <30 min'",
                  file=sys.stderr)
            print(f"  Currently passed: {just[:60]!r} ({len(just)} chars)", file=sys.stderr)
            sys.exit(2)
    # Pre-flight: refuse training-shaped cmd without --ckpt-dir. Without it, an evicted/crashed
    # task restarts from step 0 every relaunch — silently throwing away hours of GPU time. The
    # scheduler can't auto-add ckpt for the user (path is project-specific) so we fail-fast at
    # submit and force an explicit decision. The eviction loop that bit t1029/t1030 (12+ relaunches
    # losing ~1h of progress because cmd had no ckpt) is exactly what this guards against.
    if (_cmd_looks_like_training(args.cmd)
            and not args.ckpt_dir
            and not getattr(args, "allow_no_ckpt", False)):
        print(f"REFUSED: cmd looks like training but --ckpt-dir is not set.", file=sys.stderr)
        print(f"  cmd: {args.cmd[:120]}", file=sys.stderr)
        print(f"  Without ckpt-dir, eviction/crash loses ALL progress (relaunches start at step 0).", file=sys.stderr)
        print(f"  Either:", file=sys.stderr)
        print(f"    (a) submit with --ckpt-dir <abs-path-on-target> and --resume-flag '--resume_from'  (recommended)", file=sys.stderr)
        print(f"    (b) pass --allow-no-ckpt  (override; OK for short debug runs / one-shot evals)", file=sys.stderr)
        sys.exit(2)
    # Pre-flight: refuse training-shaped cmd that has --ckpt-dir but never wires resume into the
    # actual cmd. The user clearly intends to save ckpts (set --ckpt-dir) but the cmd lacks any
    # --resume / --resume_from / --load_ckpt flag AND submit didn't pass --resume-flag for the
    # scheduler to inject one on relaunch. Result: ckpts get written but never read — relaunch
    # always starts at step 0. This is the WSRL 05-04 footgun: 50h training to epoch 100, ckpts
    # on disk, but cmd had no resume → reboot would have re-run from 0.
    resume_reason = _resume_capability_reason(
        args.cmd, args.ckpt_dir, args.resume_flag,
        bool(getattr(args, "allow_no_resume", False))
    )
    if resume_reason:
        print(f"REFUSED: cmd looks like training but resume is not wired up.", file=sys.stderr)
        print(f"  cmd: {args.cmd[:120]}", file=sys.stderr)
        print(f"  reason: {resume_reason}", file=sys.stderr)
        sys.exit(2)
    with state_lock():
        state = load_state()
        sig = args.signature
        if not getattr(args, "allow_duplicate", False):
            for existing in state["tasks"]:
                if (existing.get("signature") == sig
                        and existing.get("cmd") == args.cmd
                        and existing.get("status") in ("queued", "running", "launching")):
                    print(f"DUPLICATE: {existing['id']} ({existing['status']}) has identical signature+cmd")
                    print(f"  signature: {sig}")
                    print(f"  cmd: {args.cmd[:120]}")
                    print(f"  pass --allow-duplicate to override")
                    sys.exit(2)
        # ckpt_dir cross-signature conflict: refuse if a queued/running task with a DIFFERENT
        # signature is already targeting the same --ckpt-dir. This is the cross-session footgun
        # that produced the 3 wsrl/s1024 procs (different sig labels, same out_dir) writing the
        # same checkpoint_epoch50.pt concurrently → corrupt ckpt → 14h lost. Same-sig duplicates
        # are caught by the block above; --allow-shared-ckpt-dir overrides for deliberate
        # multi-instance scenarios (e.g. distributed training reading same warm-start ckpt).
        if (args.ckpt_dir
                and not getattr(args, "allow_shared_ckpt_dir", False)
                and not getattr(args, "allow_duplicate", False)):
            for existing in state["tasks"]:
                if existing.get("status") not in ("queued", "running", "launching"): continue
                if existing.get("ckpt_dir") != args.ckpt_dir: continue
                if existing.get("signature") == sig: continue  # same-sig: handled above or legit retry
                print(f"REFUSED: --ckpt-dir already in use by an active task with a different signature.",
                      file=sys.stderr)
                print(f"  conflicting task: {existing['id']} ({existing['status']}) sig={existing.get('signature','')!r}",
                      file=sys.stderr)
                print(f"  this submit:      sig={sig!r}", file=sys.stderr)
                print(f"  shared ckpt-dir:  {args.ckpt_dir}", file=sys.stderr)
                print(f"  reason: concurrent procs writing the same ckpt path corrupt each other",
                      file=sys.stderr)
                print(f"  Either:", file=sys.stderr)
                print(f"    (a) cancel/wait for the existing task, OR", file=sys.stderr)
                print(f"    (b) point this submit to a different --ckpt-dir, OR", file=sys.stderr)
                print(f"    (c) pass --allow-shared-ckpt-dir if you know what you're doing",
                      file=sys.stderr)
                sys.exit(2)
        hist = history_get(sig) or {}
        if args.vram is not None:
            est_vram = args.vram
        elif hist.get("vram_mb"):
            est_vram = hist["vram_mb"]
        else:
            # No own history — borrow from sibling running/done tasks (same project + desc class).
            # Saves placement decisions from the "default 3500MB" pessimism on the first batch.
            full_hist = load_history()
            probe_task = {
                "id": None, "signature": sig, "project": args.project or _project_from_path(args.cwd),
                "description": args.description, "est_vram_mb": 0,
            }
            est_vram = _effective_est_vram(probe_task, state, full_hist)
        # RAM mirrors VRAM cascade: explicit → own-sig history → sibling/project peaks (median) → DEFAULT.
        # Without the cascade, novel-sig tasks land in queue at the inflated DEFAULT_RAM_MB and
        # block placement until the next dispatch tick refreshes them. The cascade returns sane
        # values immediately at submit so users see realistic numbers in TUI/status from t=0.
        if args.ram_mb is not None:
            ram_mb = args.ram_mb
        elif hist.get("ram_mb"):
            ram_mb = hist["ram_mb"]
        else:
            # full_hist may already be loaded by the VRAM cascade above; load on demand if not.
            ram_full_hist = locals().get("full_hist") or load_history()
            probe_task_for_ram = {
                "id": None, "signature": sig,
                "project": args.project or _project_from_path(args.cwd),
                "description": args.description, "ram_mb": DEFAULT_RAM_MB,
            }
            ram_mb = _effective_est_ram(probe_task_for_ram, state, ram_full_hist)
        cpu_cores = args.cpu if args.cpu is not None else hist.get("cpu_cores", DEFAULT_CPU_CORES)
        project = args.project or _project_from_path(args.cwd) or (sig.split("/", 1)[0] if "/" in sig else sig)
        task = {
            "id": f"t{state['next_id']:04d}",
            "status": "queued",
            "description": args.description,
            "project": project,
            "cmd": args.cmd,
            "cwd": args.cwd,
            "signature": sig,
            "est_vram_mb": int(est_vram),
            "ram_mb": int(ram_mb),
            "cpu_cores": int(cpu_cores),
            "priority": args.priority,
            "preferred_node": args.preferred_node,
            "require_node": args.require_node,
            "git_repo": args.git_repo,
            "ckpt_dir": args.ckpt_dir,
            "ckpt_glob": args.ckpt_glob,
            "resume_flag": args.resume_flag or "",
            "extra_env": _parse_env(args.env),
            "allow_cpu_training": bool(getattr(args, "allow_cpu_training", False)),
            "cpu_training_justification": (getattr(args, "cpu_training_justification", "") or "").strip(),
            # Env deployment strategy (see env_deploy.py). 'none' = legacy assume-env-on-target;
            # 'docker:IMAGE' = wrap cmd in docker run; 'auto' = probe target, prefer docker if
            # available. `image` is the docker image to use when env_spec resolves to docker.
            "env_spec": getattr(args, "env_spec", None) or "none",
            "image": getattr(args, "image", None) or "",
            "slurm_partition": getattr(args, "slurm_partition", None) or "",
            "slurm_account": getattr(args, "slurm_account", None) or "",
            "slurm_qos": getattr(args, "slurm_qos", None) or "",
            "node": None,
            "gpu_idx": None,
            "remote_pids": [],
            "log_path": None,
            "submitted_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "peak_vram_mb": 0,
            "resume_from": None,
        }
        state["tasks"].append(task)
        state["next_id"] += 1
        save_state(state)
    sources = []
    if hist.get("vram_mb") and args.vram is None: sources.append("vram=hist")
    elif args.vram is not None: sources.append("vram=explicit")
    if hist.get("ram_mb") and args.ram_mb is None: sources.append("ram=hist")
    elif args.ram_mb is not None: sources.append("ram=explicit")
    if hist.get("cpu_cores") and args.cpu is None: sources.append("cpu=hist")
    elif args.cpu is not None: sources.append("cpu=explicit")
    src = ",".join(sources) or "all-defaults"
    print(f"submitted {task['id']}  cpu={cpu_cores} ram={ram_mb}MB vram={est_vram}MB  prio={args.priority}  ({src})  {args.description[:50]}")
    print(f"  run `dispatch` to launch (resource-aware: respects 1/3 VRAM rule, CPU/RAM headroom).")

def _project_from_path(path):
    """Heuristic: project name = last path component. Strips trailing /; returns '' if path is None/empty."""
    if not path: return ""
    return os.path.basename(path.rstrip("/")) or ""

def _project_from_pid(node, pid):
    """Read /proc/<pid>/cwd on the node via readlink — gives the actual working dir of the running process."""
    try:
        rc, out, _ = run_on(node, f"readlink /proc/{pid}/cwd 2>/dev/null", timeout=8, check=False)
        return _project_from_path(out.strip()) if rc == 0 else ""
    except Exception:
        return ""

def _parse_env(pairs):
    if not pairs: return {}
    out = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--env expects KEY=VALUE, got {p!r}")
        k, v = p.split("=", 1)
        out[k] = v
    return out

STARTUP_FLOOR_MB = 500   # minimum reservation per running task with peak=0 (still loading model)
EVICT_TASK_MIN_AGE_S = 180  # don't evict a task within this many seconds of launch (give JAX/torch model loading + warmup a chance — JAX in particular spikes util to 100% during first iter compile)


def _slurm_max_pending_for_node(node_name: str) -> int:
    """Per-node override > global default. NODES[name]["max_slurm_pending"] takes precedence
    over SLURM_MAX_PENDING_PER_NODE when set. See the constant's docstring for tuning options."""
    return NODES.get(node_name, {}).get("max_slurm_pending", SLURM_MAX_PENDING_PER_NODE)


# Slurm states that count as "still pending in slurm's queue" for throttle accounting.
# RUNNING / COMPLETING / done aren't pending — they're consuming a real slot. None /
# empty string means "just-submitted, watcher hasn't probed slurm_state yet" — treat as
# pending until proven otherwise (next watcher cycle clears the ambiguity).
_SLURM_PENDING_LIKE = frozenset({None, "", "PENDING", "CONFIGURING", "REQUEUED", "SUSPENDED"})


def _count_slurm_pending_per_node(state: dict) -> dict:
    """Phase 2.16: count OUR slurm-managed tasks that are PENDING (or just-submitted with
    no slurm_state probed yet) per node. Used by pick_placement to throttle dispatch to
    a slurm node that already has pending tasks waiting — better to hold them in
    scheduleurm's own queue and dispatch to a node that can actually run NOW.

    The 'just-submitted' case (slurm_state=None right after sbatch) is treated as pending;
    one watcher cycle (60s default) refreshes the state to PENDING/RUNNING/etc."""
    counts: dict = {}
    for t in state.get("tasks", []):
        if t.get("status") != "running":
            continue
        if not _is_slurm_managed(t):
            continue
        node = t.get("node")
        if not node:
            continue
        if t.get("slurm_state") in _SLURM_PENDING_LIKE:
            counts[node] = counts.get(node, 0) + 1
    return counts


# Phase 3.0.3: load-balance migration tunables.
# All tunable via env var (no restart needed beyond watcher reload) for parity with
# the Phase 2.16 throttle. Defaults err on the SIDE OF NOT MIGRATING (high ratio,
# low free-threshold) so casual imbalance doesn't thrash the queue.
MIGRATION_LOAD_RATIO = float(os.environ.get("SCHEDULEURM_MIGRATION_LOAD_RATIO", "2.0"))
# Source load must be > RATIO × target load before migration kicks in.

MIGRATION_FREE_THRESHOLD_S = int(os.environ.get("SCHEDULEURM_MIGRATION_FREE_THRESHOLD_S", "600"))
# Target node's eta_load must be UNDER this many seconds (≈10 min default) for it
# to count as "almost free". Otherwise both nodes are loaded and migrating between
# them just shifts work, not balances it.

MIGRATION_MAX_PER_DISPATCH = int(os.environ.get("SCHEDULEURM_MIGRATION_MAX_PER_DISPATCH", "1"))
# At most N migrations per dispatch cycle. User-spec'd to 1 — let things settle a
# minute before another decision. Avoid stampeding the rsync staging path too.

MIGRATION_MIN_TASK_ETA_S = int(os.environ.get("SCHEDULEURM_MIGRATION_MIN_TASK_ETA_S", "300"))
# Don't migrate a task whose ETA is < this many seconds — it'll finish where it is
# faster than the rsync+launch round-trip would take.


def compute_node_load_seconds(state: dict) -> dict:
    """Phase 3.0.2: per-node load = sum of eta_seconds of in-flight tasks pinned to
    that node. Used by Phase 3.0.3 migration trigger to detect imbalance — when
    load(A) >> load(B), tasks with preferred_node=A may be moved to B.

    What counts as "in-flight on node N":
      - status=running on N (regardless of slurm_state — PENDING in slurm queue
        still consumes a slot eventually)
      - status=queued with require_node=N OR preferred_node=N (pinned future load)

    What we exclude:
      - status=launching (transient WAL state; recovers in ≤60s)
      - auto_adopted tasks (we don't migrate user-managed external work)
      - tasks with eta_seconds=0 (unknown ETA — neutral, don't assume)

    Returns: {node_name: total_eta_seconds}. Only nodes that appear in NODES are
    keyed (filters out stale node references).
    """
    loads: dict = {n: 0 for n in NODES}
    for t in state.get("tasks", []):
        if t.get("auto_adopted"):
            continue
        eta = int(t.get("eta_seconds") or 0)
        if eta <= 0:
            continue
        status = t.get("status")
        if status == "running":
            node = t.get("node")
            if node in loads:
                loads[node] += eta
        elif status == "queued":
            # pinned queue load: prefer require, else preferred (require dominates)
            pin = t.get("require_node") or t.get("preferred_node")
            if pin and pin in loads:
                loads[pin] += eta
    return loads


MIGRATION_MAX_CKPT_SIZE_MB = int(os.environ.get("SCHEDULEURM_MIGRATION_MAX_CKPT_SIZE_MB", "2048"))
# Hard cap on ckpt rsync size during migration. Larger ckpts → migration aborts; the
# task stays on source. Rationale: rsync of a 5+GB ckpt takes minutes, often longer
# than just letting source's queue drain naturally. User-spec'd default is 2GB.

# Cache for staged paths so we don't redundantly rsync. Key: (source_node, target_node, path)
# Reset on watcher restart (in-memory only) — that's fine, rsync's delta algorithm makes
# re-runs of unchanged paths trivial (~1s for unchanged).
_STAGING_CACHE: set = set()


def _stage_for_migration(task: dict, target_node: str,
                         max_ckpt_mb: int = MIGRATION_MAX_CKPT_SIZE_MB) -> tuple:
    """Phase 3.0.4: rsync code + ckpt + verify env before migration.

    Steps:
      1. Identify the source node = current `preferred_node` (or running `node`).
         Skip rsync if source==target.
      2. cwd: if missing on target, rsync source:cwd → target:cwd (`-az --partial`,
         idempotent; ~1s for already-synced). If still missing after rsync, fail.
      3. ckpt_dir: if set AND exists on source AND size ≤ max_ckpt_mb, rsync to
         target. Bigger → fail (don't migrate; task stays on source).
      4. env: extract `python` abs path from cmd (e.g. /home/u/conda/envs/X/bin/python);
         verify `ssh target test -x <path>`. Missing → fail (env-deploy is the user's
         responsibility; auto-rsync of a multi-GB conda env isn't this layer's job).

    Returns (ok, msg). ok=True only when target is now ready to launch the task.
    On failure msg explains why so callers can log it.

    Side effect on success: caches (source, target, cwd) in _STAGING_CACHE so
    subsequent migration attempts in the same process skip redundant rsync.
    """
    cwd = task.get("cwd")
    if not cwd:
        return (False, "no cwd on task")

    # Determine source for rsync
    source_node = task.get("preferred_node") or task.get("node")
    if source_node and source_node == target_node:
        return (True, "source==target; nothing to stage")

    # Step 2: cwd
    cwd_key = (source_node, target_node, cwd)
    if cwd_key not in _STAGING_CACHE:
        try:
            rc, _, _ = run_on(target_node, f"test -d {shlex.quote(cwd)}",
                              timeout=5, check=False)
        except Exception as e:
            return (False, f"target ssh failed: {str(e)[:120]}")
        if rc != 0:
            # cwd missing → rsync from source. Source must be NODES-keyed and have host.
            src_info = NODES.get(source_node or "", {})
            src_host = src_info.get("host")
            tgt_info = NODES.get(target_node, {})
            tgt_host = tgt_info.get("host")
            # Build src/dst rsync paths
            src_path = (f"{src_host}:" if src_host else "") + cwd.rstrip("/") + "/"
            dst_path = (f"{tgt_host}:" if tgt_host else "") + cwd.rstrip("/") + "/"
            mkdir_cmd = f"mkdir -p {shlex.quote(cwd)}"
            try:
                run_on(target_node, mkdir_cmd, timeout=10, check=False)
            except Exception:
                pass
            # rsync only works directly when one side is local. If both src and tgt are
            # remote (different hosts), we can't directly do remote→remote without ssh
            # tunneling. Use --rsync-path workaround OR, simpler: pull source to local
            # then push to target in two hops. For now: only support source-side OR
            # target-side being local (typical scheduleurm: local is usually one of the
            # two). Bail on remote→remote pairs.
            if src_host and tgt_host:
                return (False,
                        f"cwd missing on {target_node} and rsync remote→remote not "
                        f"yet supported (src={src_host}, tgt={tgt_host}); user must "
                        f"sync code manually OR via shared NFS")
            try:
                import subprocess as _sp
                r = _sp.run(["rsync", "-az", "--partial",
                             "--exclude=.git", "--exclude=__pycache__",
                             "--exclude=*.pyc", src_path, dst_path],
                            capture_output=True, text=True, timeout=600)
                if r.returncode != 0:
                    return (False, f"rsync cwd failed rc={r.returncode}: {r.stderr.strip()[:200]}")
            except _sp.TimeoutExpired:
                return (False, "rsync cwd timeout (>10min)")
            except Exception as e:
                return (False, f"rsync cwd exception: {e}")
            # Verify after rsync
            try:
                rc2, _, _ = run_on(target_node, f"test -d {shlex.quote(cwd)}",
                                   timeout=5, check=False)
                if rc2 != 0:
                    return (False, "cwd still missing after rsync")
            except Exception as e:
                return (False, f"post-rsync ssh check failed: {e}")
        _STAGING_CACHE.add(cwd_key)

    # Step 3: ckpt_dir (if set)
    ckpt_dir = task.get("ckpt_dir")
    if ckpt_dir:
        # Size check: only on source side (where ckpt actually lives)
        size_mb = 0
        if source_node:
            try:
                rc, out, _ = run_on(source_node,
                                    f"du -sm {shlex.quote(ckpt_dir)} 2>/dev/null | "
                                    f"awk '{{print $1}}'",
                                    timeout=15, check=False)
                if rc == 0 and out.strip().isdigit():
                    size_mb = int(out.strip())
            except Exception:
                size_mb = 0
        if size_mb > max_ckpt_mb:
            return (False,
                    f"ckpt_dir {ckpt_dir} is {size_mb}MB > max {max_ckpt_mb}MB; "
                    f"migration aborted (rsync would take too long)")

        # Stage ckpt (same two-host limitation as cwd: rsync needs one side local).
        # Phase 3.0.6 P1 fix: prior code had _STAGING_CACHE.add(ckpt_key) OUTSIDE the
        # rsync branch — when remote→remote was skipped (no rsync attempted) the cache
        # was still populated and the function returned success. Migration would then
        # commit and the resume task would start on target without a checkpoint —
        # silently restarting from step 0. Now we reject remote→remote (same as cwd
        # does) and only cache after a verified rsync.
        ckpt_key = (source_node, target_node, ckpt_dir)
        if size_mb > 0 and ckpt_key not in _STAGING_CACHE:
            src_info = NODES.get(source_node or "", {})
            src_host = src_info.get("host")
            tgt_info = NODES.get(target_node, {})
            tgt_host = tgt_info.get("host")
            if src_host and tgt_host:
                # Remote→remote: refuse (parity with cwd). The migrated task would
                # otherwise resume-from-zero on target — worse than not migrating.
                return (False,
                        f"ckpt_dir {ckpt_dir} ({size_mb}MB) needs rsync but "
                        f"remote→remote not supported (src={src_host}, tgt={tgt_host}); "
                        f"task would lose its checkpoint on migration. User must "
                        f"sync ckpt manually OR via shared NFS")
            src_path = (f"{src_host}:" if src_host else "") + ckpt_dir.rstrip("/") + "/"
            dst_path = (f"{tgt_host}:" if tgt_host else "") + ckpt_dir.rstrip("/") + "/"
            try:
                run_on(target_node, f"mkdir -p {shlex.quote(ckpt_dir)}",
                       timeout=10, check=False)
                import subprocess as _sp
                r = _sp.run(["rsync", "-az", "--partial",
                             src_path, dst_path],
                            capture_output=True, text=True, timeout=600)
                if r.returncode != 0:
                    return (False, f"rsync ckpt failed rc={r.returncode}: "
                                   f"{r.stderr.strip()[:200]}")
            except _sp.TimeoutExpired:
                return (False, "rsync ckpt timeout (>10min)")
            except Exception as e:
                return (False, f"rsync ckpt exception: {e}")
            # Verify rsync put files at the target — if the dir is empty, treat as
            # failure (don't migrate a resume task to an empty ckpt path).
            try:
                rc2, out2, _ = run_on(target_node,
                                      f"ls -1 {shlex.quote(ckpt_dir)} 2>/dev/null | head -1",
                                      timeout=5, check=False)
                if rc2 != 0 or not (out2 or "").strip():
                    return (False, f"ckpt_dir {ckpt_dir} appears empty on {target_node} "
                                   f"after rsync (rc={rc2}); migration aborted")
            except Exception as e:
                return (False, f"post-rsync ckpt check failed: {e}")
            # Only cache on successful + verified rsync
            _STAGING_CACHE.add(ckpt_key)

    # Step 4: env probe — extract python path from cmd
    cmd_str = task.get("cmd") or ""
    py_path = None
    py_match = re.search(r'(/[\w./-]+/python\d*(?:\.\d+)?)\b', cmd_str)
    if py_match:
        py_path = py_match.group(1)
    if py_path:
        try:
            rc, _, _ = run_on(target_node, f"test -x {shlex.quote(py_path)}",
                              timeout=5, check=False)
            if rc != 0:
                return (False,
                        f"python at {py_path} not executable on {target_node}; "
                        f"deploy the conda env first (env_spec=conda:... if available)")
        except Exception as e:
            return (False, f"env probe failed: {e}")

    return (True, f"staged (cwd{' + ckpt' if ckpt_dir else ''}{' + env' if py_path else ''})")


def _can_migrate_to(task: dict, target_node: str, timeout_s: int = 5) -> bool:
    """Phase 3.0.5: fast-path lookup of _STAGED_TASKS only — NO ssh/rsync inside the
    state_lock that callers hold. Staging itself runs outside the lock via
    _stage_migration_candidates_outside_lock(); _consider_migration uses this fn to
    confirm staging completed before committing the preferred_node rewrite.

    Pre-Phase-3.0.5: this called _stage_for_migration directly, which would block
    cmd_submit / cancel / status / watcher for up to 600s during ckpt rsync inside
    the global lock. Now it's a pure dict membership check — milliseconds."""
    return (task.get("id"), target_node) in _STAGED_TASKS


# Process-local staging cache — populated by _stage_migration_candidates_outside_lock,
# read by _can_migrate_to inside the state_lock. Key: (task_id, target_node).
# Value: timestamp. Reset on watcher restart (acceptable — re-staging is cheap thanks
# to rsync's delta algorithm, ~1s for unchanged paths).
_STAGED_TASKS: dict = {}
_STAGED_TASKS_MAX = 100  # bound memory; LRU evicts oldest when full


def _identify_migration_candidates(state: dict, nodes: list,
                                    max_candidates: int = 2) -> list:
    """Pure logic: return up to `max_candidates` (task_dict_copy, target_node) pairs
    that pass all migration GATES (load-imbalance, soft-pin, ETA-min, target alive).

    Does NOT do staging or any I/O. Safe to call inside or outside state_lock.
    Returns deep-enough copies of task fields so the outside-lock caller can read
    cwd / ckpt_dir / cmd / preferred_node without re-acquiring the lock.

    Used by _stage_migration_candidates_outside_lock to pick rsync targets.
    Inside-lock _consider_migration re-runs the same gates plus _STAGED_TASKS
    membership to actually commit.
    """
    loads = compute_node_load_seconds(state)
    if not loads or len(loads) < 2:
        return []
    alive_names = {n["name"] for n in nodes if n.get("alive")}
    candidates_loads = [(name, load) for name, load in loads.items() if name in alive_names]
    if len(candidates_loads) < 2:
        return []
    candidates_loads.sort(key=lambda kv: kv[1])
    target_name, target_load = candidates_loads[0]
    source_name, source_load = candidates_loads[-1]

    if source_name == target_name:
        return []
    if target_load >= MIGRATION_FREE_THRESHOLD_S:
        return []
    if source_load < MIGRATION_LOAD_RATIO * max(target_load, 1):
        return []

    candidates = []
    for t in state.get("tasks", []):
        if t.get("status") != "queued":
            continue
        if t.get("require_node"):
            continue
        if t.get("auto_adopted"):
            continue
        if t.get("preferred_node") != source_name:
            continue
        eta = int(t.get("eta_seconds") or 0)
        if eta > 0 and eta < MIGRATION_MIN_TASK_ETA_S:
            continue
        # Snapshot fields needed by _stage_for_migration (cwd, ckpt_dir, cmd,
        # preferred_node, signature, id) so the outside-lock caller doesn't need
        # to hold a reference into state["tasks"].
        candidates.append({
            "id": t["id"],
            "cwd": t.get("cwd"),
            "ckpt_dir": t.get("ckpt_dir"),
            "cmd": t.get("cmd"),
            "preferred_node": t.get("preferred_node"),
            "signature": t.get("signature"),
            "eta_seconds": eta,
        })
    candidates.sort(key=lambda t: int(t.get("eta_seconds") or 0))
    # Pair each candidate with the target_name decision computed above.
    return [(c, target_name) for c in candidates[:max_candidates]]


def _stage_migration_candidates_outside_lock(max_candidates: int = 2):
    """Phase 3.0.5 P1 fix: identify migration candidates inside a SHORT lock,
    release lock, run rsync staging OUTSIDE the lock (which can take minutes for
    multi-GB ckpts), update _STAGED_TASKS on success.

    Called from cmd_dispatch and watcher iteration BEFORE acquiring the main
    state_lock. Mirrors _preload_docker_images_outside_lock's pattern from
    Phase 1 — slow I/O moves out of the global lock so submit/cancel/status/
    watcher don't get stalled for minutes during staging.

    Failure tolerant: any exception in identify or stage paths just leaves the
    cache as-is; next cycle retries. Logs to watcher.log via notify() but never
    propagates exceptions to the caller.
    """
    try:
        with state_lock():
            state = load_state()
            nodes = probe_all()
            snapshot = _identify_migration_candidates(state, nodes,
                                                      max_candidates=max_candidates)
        # lock released here — slow rsync runs WITHOUT blocking submit/cancel/etc
    except Exception as e:
        try:
            notify("migration_snapshot_error", {"error": str(e)[:200]},
                   feishu_enabled=False)
        except Exception:
            pass
        return

    if not snapshot:
        return

    # LRU eviction if cache is full
    if len(_STAGED_TASKS) >= _STAGED_TASKS_MAX:
        # Drop oldest 25% to make room
        sorted_keys = sorted(_STAGED_TASKS.items(), key=lambda kv: kv[1])
        for k, _ in sorted_keys[:_STAGED_TASKS_MAX // 4]:
            _STAGED_TASKS.pop(k, None)

    for task_snapshot, target in snapshot:
        cache_key = (task_snapshot["id"], target)
        # Skip already-staged in this process
        if cache_key in _STAGED_TASKS:
            continue
        try:
            ok, msg = _stage_for_migration(task_snapshot, target)
            if ok:
                _STAGED_TASKS[cache_key] = time.time()
            else:
                # Don't cache failures; next cycle retries (transient ssh blips, etc).
                # Log so user can see persistent failures in watcher.log.
                try:
                    notify("migration_staging_skip",
                           {"task_id": task_snapshot["id"],
                            "target": target,
                            "reason": msg[:200]},
                           feishu_enabled=False)
                except Exception:
                    pass
        except Exception as e:
            try:
                notify("migration_staging_error",
                       {"task_id": task_snapshot["id"],
                        "target": target,
                        "error": str(e)[:200]},
                       feishu_enabled=False)
            except Exception:
                pass


def _consider_migration(state: dict, nodes: list, loads: Optional[dict] = None) -> list:
    """Phase 3.0.3: when one node has a much heavier eta_load than another,
    re-pin a queued task with preferred_node=overloaded to the lighter node.

    Strategy:
      1. Compute per-node load (or use cached `loads` arg).
      2. Find heaviest + lightest nodes. If gap < MIGRATION_LOAD_RATIO or
         lightest is itself loaded > MIGRATION_FREE_THRESHOLD_S → no migration.
      3. Find queued candidates with preferred_node=heaviest, no require_node
         (hard pins respected), eta_seconds ≥ MIGRATION_MIN_TASK_ETA_S.
      4. Sort by eta ascending — cheapest task moves first.
      5. For each candidate, run _can_migrate_to. Skip on fail.
      6. On first success: rewrite preferred_node = target, log reason, return
         that task_id. Hard cap MIGRATION_MAX_PER_DISPATCH per cycle.

    Returns list of migrated task IDs (current default cap=1 → 0 or 1 entry).

    Does NOT mutate scheduler.py call sites' "nodes" view of CPU/RAM/VRAM —
    migration only changes the task's preferred_node, leaving real placement to
    pick_placement on the next iteration.
    """
    if loads is None:
        loads = compute_node_load_seconds(state)
    if not loads or len(loads) < 2:
        return []

    # Order nodes by load. Skip nodes that are alive=False — can't migrate to dead box.
    alive_names = {n["name"] for n in nodes if n.get("alive")}
    candidates_loads = [(name, load) for name, load in loads.items() if name in alive_names]
    if len(candidates_loads) < 2:
        return []
    candidates_loads.sort(key=lambda kv: kv[1])
    target_name, target_load = candidates_loads[0]
    source_name, source_load = candidates_loads[-1]

    if source_name == target_name:
        return []
    if target_load >= MIGRATION_FREE_THRESHOLD_S:
        return []  # target not really free
    if source_load < MIGRATION_LOAD_RATIO * max(target_load, 1):
        return []  # not imbalanced enough

    # Find candidates: queued tasks with soft pin to source, no hard pin
    candidates = []
    for t in state.get("tasks", []):
        if t.get("status") != "queued":
            continue
        if t.get("require_node"):
            continue  # hard pin — respected
        if t.get("auto_adopted"):
            continue  # not ours to move
        if t.get("preferred_node") != source_name:
            continue
        eta = int(t.get("eta_seconds") or 0)
        if eta > 0 and eta < MIGRATION_MIN_TASK_ETA_S:
            continue  # task will finish before staging completes
        candidates.append(t)

    if not candidates:
        return []

    # Smallest ETA first — cheapest staging if Phase 3.0.4's rsync cost is proportional,
    # AND quickest to actually start running on target.
    candidates.sort(key=lambda t: int(t.get("eta_seconds") or 0))

    migrated = []
    for cand in candidates:
        if len(migrated) >= MIGRATION_MAX_PER_DISPATCH:
            break
        if not _can_migrate_to(cand, target_name):
            continue
        old_pref = cand.get("preferred_node")
        cand["preferred_node"] = target_name
        cand["migrated_from"] = old_pref
        cand["migrated_at"] = time.time()
        cand["last_block_reason"] = (
            f"migrated: preferred {old_pref}(load={int(source_load)}s) → "
            f"{target_name}(load={int(target_load)}s) for load balance"
        )
        migrated.append(cand["id"])
    return migrated


def _is_slurm_managed(task: dict) -> bool:
    """True iff this task's placement is owned by slurm (NOT scheduleurm).

    Phase 2.12 P2 defensive helper. scheduleurm's eviction / preemption /
    inflight-vram-reservation must skip slurm-routed tasks because slurm has
    its own queue + cgroup + walltime mechanism that owns the task once
    sbatch returned. Without an explicit guard, the only reason these paths
    didn't touch slurm tasks today is the implicit `gpu_idx == g["idx"]`
    filter (slurm tasks have gpu_idx=None) — fragile under refactor: the
    moment any path sets gpu_idx for a slurm task (e.g. for cosmetic display),
    eviction would start scancel'ing slurm-managed work.

    Source of truth: slurm_job_id presence. SlurmBackend.launch sets it; nothing
    else does. A task with slurm_job_id is in slurm's queue regardless of
    scheduleurm's view of node-level resources.
    """
    return bool(task.get("slurm_job_id"))

def _reserve_inflight_vram(state, nodes):
    """Reserve a small floor for running tasks whose model hasn't yet loaded onto GPU
    (peak_vram_mb still tiny). This prevents over-packing during the multi-minute SUMO sim
    phase before model upload, while letting nvidia-smi's real `used_mb` drive packing
    decisions once a task is actually using the GPU.

    Old v1 reserved full declared VRAM (3500MB) — too conservative when actual usage was 300MB.
    Old v2 reserved flat STARTUP_FLOOR_MB (500MB) — over-reserved for tasks whose sibling history
    showed peak ~334MB, blocking the 1/3 packing rule on local even when nvidia-smi reported the
    GPU was barely used.
    Current: reserve min(est_vram_mb, STARTUP_FLOOR_MB). Since est_vram_mb is sibling-aware
    (refreshed in _do_dispatch), small repeat workloads (WSRL retrain @ 334MB) reserve their
    actual size; unknown big tasks (Online SAC @ est=3500) cap at the floor so they don't
    monopolize the card before peak is observed. Once peak_vram_mb >= 100, nvidia-smi takes over."""
    by_node = {n["name"]: n for n in nodes}
    for t in state.get("tasks", []):
        if t.get("status") != "running":
            continue
        if _is_slurm_managed(t):
            continue  # Phase 2.12: slurm allocates its own VRAM via cgroup; we don't reserve here
        if t.get("gpu_idx") is None:
            continue
        n = by_node.get(t.get("node"))
        if not n or not n.get("alive"):
            continue
        peak = int(t.get("peak_vram_mb") or 0)
        if peak >= 100:
            # Real usage detected — already in nvidia-smi's used_mb, no extra reservation needed.
            continue
        est = int(t.get("est_vram_mb") or 0)
        reserve = min(est, STARTUP_FLOOR_MB) if est > 0 else STARTUP_FLOOR_MB
        if reserve < 100:  # never reserve below 100MB — tiny tasks would let too many stack
            reserve = 100
        for g in n["gpus"]:
            if g["idx"] != t["gpu_idx"]:
                continue
            g["used_mb"] = min(g["total_mb"], g["used_mb"] + reserve)
            g["free_mb"] = max(0, g["free_mb"] - reserve)
            break

def _signature_batch_key(signature):
    """Group tasks into batches by top-2 signature path components.
    e.g. 'offline-sumo/retrain-v2/online_sac/s42' → 'offline-sumo/retrain-v2'.
    A batch is "done" when no sibling sharing this prefix is queued/running anymore.
    Single-component signatures use the full string as the key."""
    if not signature:
        return ""
    parts = signature.split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else signature

def _detect_batch_completions(state, transitioned_ids):
    """If a `signature_batch_key` group went from "had non-terminal task" → "all terminal" this iter,
    return its summary so we notify once. Re-arms when a NEW task (later finished_at) of the same
    prefix later transitions, so a fresh batch of the same project family fires again."""
    if not transitioned_ids:
        return []
    transitioned_set = set(transitioned_ids)
    # Prefixes of just-terminated tasks — only these can newly complete a batch.
    candidate_prefixes = set()
    for t in state["tasks"]:
        if t["id"] in transitioned_set:
            candidate_prefixes.add(_signature_batch_key(t.get("signature") or ""))
    completed = []
    state.setdefault("_batch_notify_ts", {})
    for prefix in candidate_prefixes:
        if not prefix:
            continue
        sibs = [t for t in state["tasks"]
                if _signature_batch_key(t.get("signature") or "") == prefix
                and not t.get("auto_adopted")]
        if not sibs:
            continue
        active = [t for t in sibs if t.get("status") in ("queued", "running", "launching")]
        if active:
            continue  # batch still in flight
        last_ts = state["_batch_notify_ts"].get(prefix, 0)
        latest_finish = max((t.get("finished_at") or 0) for t in sibs)
        if latest_finish <= last_ts:
            continue  # already notified for this fully-terminal state
        from collections import Counter
        counts = Counter(t.get("status") for t in sibs)
        completed.append({
            "prefix": prefix,
            "total": len(sibs),
            "done": counts.get("done", 0),
            "failed": counts.get("failed", 0),
            "cancelled": counts.get("cancelled", 0),
            "task_ids": [t["id"] for t in sibs],
            "earliest_submit": min((t.get("submitted_at") or 0) for t in sibs),
            "latest_finish": latest_finish,
        })
        state["_batch_notify_ts"][prefix] = latest_finish
    return completed

def _enforce_post_dispatch_thresholds(state, nodes):
    """Rollback companion to the relaxed _reserve_inflight_vram. After tasks have had time to
    settle, if a GPU is now over BOTH design thresholds (mem ≥ 1/3 AND util ≥ saturation) AND
    multiple scheduler-owned tasks are sharing it, kill the most-recently-started one and put
    it back in the queue. Lets dispatch be optimistic ("trust observed usage") while preventing
    actual OOMs when the gamble fails. Returns the list of evicted task ids.

    AND (not OR) is critical: a co-located task pair where the elder pins util to 100% but mem
    is still 20% (typical JAX BAPR ablations: t1029+t1030 on jtl110gpu2:GPU0, mem=2.5/12GB,
    util=100% from t1029's hot training loop) is NOT a real threshold breach — the OOM blast
    radius is mem, and mem is fine. Evicting the youngest there just thrashes (it gets relaunched
    next dispatch, runs 5min, gets evicted again, ad infinitum, losing all progress per cycle).
    Util saturation alone is already handled at placement time (won't pack a NEW task onto a
    util-saturated GPU); using it again to evict an existing task is double jeopardy."""
    now = time.time()
    evicted = []
    for n in nodes:
        if not n.get("alive"):
            continue
        for g in n["gpus"]:
            third = g["total_mb"] // 3
            occupied = g["used_mb"] > 100
            mem_over = occupied and g["used_mb"] >= third
            util_over = occupied and g.get("util_pct", 0) >= GPU_UTIL_SATURATION_PCT
            # AND, not OR — see docstring. Only evict when BOTH mem and util are over (= real
            # contention with OOM risk). Util-only spikes are normal during JAX/torch warmup or
            # an elder task's training loop; they don't justify evicting the younger task.
            if not (mem_over and util_over):
                continue
            tasks_here = [
                t for t in state["tasks"]
                if t.get("status") == "running"
                and t.get("node") == n["name"]
                and t.get("gpu_idx") == g["idx"]
                and t.get("started_at")
                and not t.get("auto_adopted")  # never evict externally-launched tasks
                and not _is_slurm_managed(t)   # Phase 2.12: slurm owns its placement; never scancel here
            ]
            if len(tasks_here) < 2:
                continue  # single big task on the GPU — design exception, leave it
            tasks_here.sort(key=lambda t: t.get("started_at", 0), reverse=True)
            youngest = tasks_here[0]
            age = now - (youngest.get("started_at") or now)
            if age < EVICT_TASK_MIN_AGE_S:
                continue  # not had time to settle; might be ramping up
            _kill_task_processes(youngest, timeout=15)
            youngest["status"] = "queued"
            for k in ("node", "gpu_idx", "process_group", "log_path", "started_at", "finished_at", "_diagnosis"):
                youngest[k] = None
            youngest["remote_pids"] = []
            youngest["alive_pids"] = []
            youngest["peak_vram_mb"] = 0
            youngest["peak_ram_mb"] = 0
            youngest["notified_done"] = False
            youngest["last_block_reason"] = (
                f"evicted from {n['name']}:GPU{g['idx']} after threshold breach "
                f"(mem={g['used_mb']}/{g['total_mb']}MB util={g.get('util_pct','?')}%)"
            )
            evicted.append(youngest["id"])
    return evicted

PREEMPT_QUEUE_WAIT_MIN = 5      # high-prio waits > N min before we consider preempting
PREEMPT_VICTIM_MIN_AGE_MIN = 10 # don't evict a task younger than this (let it settle)
PREEMPT_VICTIM_MAX_AGE_MIN = 240 # don't evict a task that's been stable too long (probably load-bearing)
PREEMPT_MAX_VICTIMS_PER_DISPATCH = 3  # cap chain-evictions; 1 round shouldn't kill an entire node

def _evict_to_queue(victim, state, reason):
    """Send a running task back to queue WITHOUT incrementing retry_count or marking crash.
    Used by preemption — task didn't fail, we just made room for higher priority. Kills its
    PIDs, resets running fields, sets last_block_reason for visibility."""
    _kill_task_processes(victim, timeout=15)
    victim["status"] = "queued"
    for k in ("node", "gpu_idx", "process_group", "log_path", "started_at", "finished_at", "_diagnosis"):
        victim[k] = None
    victim["remote_pids"] = []
    victim["alive_pids"] = []
    victim["peak_vram_mb"] = 0
    victim["peak_ram_mb"] = 0
    victim["notified_done"] = False
    victim["last_block_reason"] = reason

def _preempt_for_high_priority(state, nodes):
    """Resolve starvation: a high-prio task waiting > PREEMPT_QUEUE_WAIT_MIN may evict younger
    normal-prio scheduler-launched tasks on its require_node, freeing CPU/RAM/cap for the high
    task on next dispatch. Sufficiency-aware: keeps evicting on the same node (newest first)
    until the freed CPU/RAM covers hi's requirement OR no more eligible victims OR cap of
    PREEMPT_MAX_VICTIMS_PER_DISPATCH is hit. Without this loop, a single eviction freeing
    cpu=2 wouldn't cover a hi needing cpu=6 — hi would wait another 60s for the next dispatch
    to evict the next victim, taking 30+ minutes to make room serially.

    Victim must be in age window [10min, 240min]: not too fresh (let it settle) and not too
    old (preserve long-stable load-bearing tasks). Adopted tasks are never evicted — user may
    have intentional pinning we don't see. Returns list of {id, node, cpu_freed, ram_freed}."""
    now = time.time()
    queued_high = [t for t in state.get("tasks", [])
                   if t.get("status") == "queued"
                   and t.get("priority") == "high"
                   and t.get("submitted_at")
                   and (now - t["submitted_at"]) > PREEMPT_QUEUE_WAIT_MIN * 60]
    if not queued_high:
        return []
    queued_high.sort(key=lambda t: t.get("submitted_at", 0))  # oldest first
    evicted = []  # list of {id, node, cpu_freed, ram_freed}
    nodes_done = set()  # nodes we've already preempted on this dispatch — don't double-target
    for hi in queued_high:
        if len(evicted) >= PREEMPT_MAX_VICTIMS_PER_DISPATCH:
            break
        node_pin = hi.get("require_node") or hi.get("preferred_node")
        if not node_pin or node_pin in nodes_done:
            continue
        cpu_need = hi.get("cpu_cores") or DEFAULT_CPU_CORES
        ram_need = hi.get("ram_mb") or DEFAULT_RAM_MB
        cpu_acc = ram_acc = 0
        wait_min = int((now - hi["submitted_at"]) / 60)
        for _ in range(PREEMPT_MAX_VICTIMS_PER_DISPATCH - len(evicted)):
            if cpu_acc >= cpu_need and ram_acc >= ram_need:
                break  # already enough freed for hi to fit
            victims = [t for t in state["tasks"]
                       if t.get("status") == "running"
                       and t.get("node") == node_pin
                       and t.get("priority") == "normal"
                       and not t.get("auto_adopted")
                       and not _is_slurm_managed(t)  # Phase 2.12: slurm owns these; don't preempt
                       and t.get("started_at")
                       and (now - t["started_at"]) > PREEMPT_VICTIM_MIN_AGE_MIN * 60
                       and (now - t["started_at"]) < PREEMPT_VICTIM_MAX_AGE_MIN * 60]
            if not victims:
                break  # no more eligible
            victims.sort(key=lambda t: (t.get("started_at", 0)), reverse=True)  # newest first
            victim = victims[0]
            cpu_freed = victim.get("cpu_cores") or DEFAULT_CPU_CORES
            ram_freed = victim.get("ram_mb") or DEFAULT_RAM_MB
            _evict_to_queue(victim, state,
                            f"preempted by {hi['id']} (high-prio waited {wait_min}min)")
            evicted.append({"id": victim["id"], "node": node_pin,
                            "cpu_freed": cpu_freed, "ram_freed": ram_freed})
            cpu_acc += cpu_freed
            ram_acc += ram_freed
        nodes_done.add(node_pin)
    return evicted

def _do_dispatch(state, nodes):
    """Place every fittable queued task. Mutates state and nodes in place. Returns event list.
    Caller is responsible for state_lock and save_state."""
    events = []
    prio = {"high": 0, "normal": 1, "low": 2}
    # Phase 3.0.3: load-balance pass — re-pin a queued soft-pinned task from an
    # overloaded node to a near-empty one, BEFORE the placement loop. Hard pins
    # (require_node) are never touched. Capped at MIGRATION_MAX_PER_DISPATCH
    # (default 1) so we don't churn the queue. This runs first so the placement
    # loop sees the new preferred_node assignment.
    migrated = _consider_migration(state, nodes)
    if migrated:
        for tid in migrated:
            events.append({"type": "migrated", "task_id": tid})
    # Preemption pass: free a slot for starved high-prio tasks (one eviction max per dispatch).
    preempted = _preempt_for_high_priority(state, nodes)
    # Initialize per-node running task count (for max_concurrent_running cap in _node_resources_ok).
    from collections import Counter as _Counter
    running_per_node = _Counter(t.get("node") for t in state["tasks"]
                                 if t.get("status") == "running" and t.get("node"))
    # Phase 2.16: count OUR slurm-pending tasks per node. pick_placement throttles further
    # dispatch to a node that already has SLURM_MAX_PENDING_PER_NODE pending — keeps tasks
    # in scheduleurm's queue so they can spread to whichever slurm node frees up next,
    # rather than piling on one host's slurm queue.
    slurm_pending_per_node = _count_slurm_pending_per_node(state)
    for n in nodes:
        n["running_count"] = running_per_node.get(n["name"], 0)
        n["slurm_pending_count"] = slurm_pending_per_node.get(n["name"], 0)
    # Apply freed resources from preemption to the local probe view so the high-prio task
    # actually fits on this dispatch round (otherwise probe lags by 60s and high keeps waiting).
    for ev in preempted:
        for n in nodes:
            if n["name"] == ev["node"]:
                n["free_cpu"] = n.get("free_cpu", 0) + ev["cpu_freed"]
                n["free_ram_mb"] = n.get("free_ram_mb", 0) + ev["ram_freed"]
                n["running_count"] = max(0, n.get("running_count", 0) - 1)
                break
        events.append({"type": "preempted", "task_id": ev["id"], "freed_node": ev["node"],
                        "cpu_freed": ev["cpu_freed"], "ram_freed": ev["ram_freed"]})
    # Refresh est_vram_mb for queued tasks based on sibling observations. Tasks submitted with
    # the 3500MB default get re-estimated against currently-running siblings, so placement
    # decisions reflect actual workload rather than a one-size-fits-all guess.
    history_cache = load_history()
    for t in state["tasks"]:
        if t.get("status") != "queued":
            continue
        sig = t.get("signature") or ""
        h = history_cache.get(sig)
        if isinstance(h, int): h = {"vram_mb": h}
        if isinstance(h, dict) and h.get("vram_mb"):
            # OWN-signature history: this is real, trust it both up and down.
            new_est = int(h["vram_mb"])
            if new_est != t.get("est_vram_mb"):
                t["est_vram_mb"] = new_est
        else:
            # NO own history yet: cascade is a guess from siblings/project. Only allow it to
            # LOWER the stored est, never raise it. Reason: if a guess says 4096 but user / prior
            # state said 512, raising hurts schedulability and the eviction mechanism would
            # immediately catch a real OOM. Symmetric trust would let one giant sibling pollute
            # all small siblings' est upward forever.
            new_est = _effective_est_vram(t, state, history_cache)
            cur = t.get("est_vram_mb") or 0
            if new_est and 0 < new_est < cur:
                t["est_vram_mb"] = new_est
        if isinstance(h, dict) and h.get("ram_mb"):
            new_ram = int(h["ram_mb"])
            if new_ram != t.get("ram_mb"):
                t["ram_mb"] = new_ram
        else:
            new_ram = _effective_est_ram(t, state, history_cache)
            cur_ram = t.get("ram_mb") or 0
            if new_ram and 0 < new_ram < cur_ram:
                t["ram_mb"] = new_ram
    # Race-condition guard: precompute set of signatures that already have a running task.
    # Without this, multi-session re-submissions of the same signature dispatch concurrently
    # and clobber each other's --out_dir / --ckpt-dir (3 procs writing same checkpoint_epoch50.pt
    # → 14h × 3 of compute fights for one shared file). Empty signature = exempt (those are
    # auto-adopted / one-off tasks where multi-instance is OK).
    # Include 'launching' (WAL window between status=queued and ssh-success → status=running)
    # so a second queued task with the same signature can't slip through during the brief
    # launching window. Without this, two dispatch cycles overlapping the WAL window could
    # both succeed → 2 instances of same sig running. Invariant: same signature can be in
    # AT MOST ONE active state (queued is the only allowed plural — but dedup at dispatch).
    running_sigs = {(t.get("signature") or "")
                    for t in state["tasks"]
                    if t.get("status") in ("running", "launching") and t.get("signature")}
    queued = sorted(
        [t for t in state["tasks"] if t["status"] == "queued"],
        key=lambda t: (prio.get(t["priority"], 1), t["submitted_at"])
    )
    for t in queued:
        sig = t.get("signature") or ""
        if sig and sig in running_sigs:
            reason = (f"signature {sig!r} already has a running task; refusing to dispatch a "
                      f"second instance (would clobber shared --out_dir/--ckpt-dir).")
            t["last_block_reason"] = reason
            events.append({"type": "blocked", "task_id": t["id"], "task": t, "reason": reason})
            continue
        cpu_training_block = _queued_cpu_training_block_reason(t)
        if cpu_training_block:
            t["last_block_reason"] = cpu_training_block
            events.append({"type": "blocked", "task_id": t["id"], "task": t, "reason": cpu_training_block})
            continue
        placement = pick_placement(t, nodes)
        if placement is None:
            # Build a precise reason by re-checking each candidate node. Helps user see e.g.
            # "GPU 1/3 locked + cpu insufficient on require_node" instead of generic "no fit".
            blocked = _blocked_nodes_for_task(t)
            require = t.get("require_node")
            prefer = t.get("preferred_node")
            reasons = []
            for n in nodes:
                if not n.get("alive"):
                    reasons.append(f"{n['name']}=DOWN"); continue
                if n["name"] in (blocked or set()):
                    reasons.append(f"{n['name']}=blocklisted"); continue
                # Phase 2.3 P1: slurm nodes don't go through local capacity gate, so showing
                # "GPU0=1/3 mem locked" is misleading (we never probed slurm-side capacity).
                # If a slurm node is alive + not blocklisted but pick_placement still returned
                # None, the only legitimate reason is require_node mismatch — surface that.
                if not _BACKEND.requires_local_capacity_check(n["name"]):
                    pending = n.get("slurm_pending_count", 0)
                    cap = _slurm_max_pending_for_node(n["name"])
                    if pending >= cap:
                        reasons.append(f"{n['name']}=slurm({pending}/{cap} pending; throttled)")
                    elif require and require != n["name"]:
                        reasons.append(f"{n['name']}=slurm(require!={require})")
                    else:
                        reasons.append(f"{n['name']}=slurm(deferred but require/prefer mismatch)")
                    continue
                ok_node, why_node = _node_resources_ok(t, n, NODES[n["name"]])
                if not ok_node:
                    reasons.append(f"{n['name']}={why_node}"); continue
                if (t.get("est_vram_mb") or 0) > 0:
                    gpu_reasons = []
                    for g in n["gpus"]:
                        if not _gpu_fits(t, g, NODES[n["name"]]):
                            third = g["total_mb"] // 3
                            sub = []
                            if g["used_mb"] >= third and g["used_mb"] > 100:
                                sub.append(f"1/3 mem ({g['used_mb']}>={third}MB)")
                            if g["used_mb"] > 100 and g.get("util_pct", 0) >= GPU_UTIL_SATURATION_PCT:
                                sub.append(f"util {g['util_pct']}%")
                            if g["free_mb"] < (t.get("est_vram_mb") or 0) + VRAM_MARGIN_MB:
                                sub.append(f"free<est+margin")
                            cap = NODES[n["name"]].get("max_vram_per_task")
                            if cap and t.get("est_vram_mb", 0) > cap:
                                sub.append(f"per-task cap {cap}")
                            gpu_reasons.append(f"GPU{g['idx']}=" + "&".join(sub or ["?"]))
                    if gpu_reasons:
                        reasons.append(f"{n['name']}=node-ok-but-" + "/".join(gpu_reasons))
                    else:
                        reasons.append(f"{n['name']}=fits?(unexpected)")
            pin = f"require={require} " if require else (f"prefer={prefer} " if prefer else "")
            t["last_block_reason"] = f"no fit ({pin}prio={t.get('priority','normal')}): " + " | ".join(reasons[:3])
            events.append({"type": "no_fit", "task_id": t["id"], "task": t})
            continue
        t["node"], t["gpu_idx"] = placement
        ok, why = precheck_git(t)
        if not ok:
            t["last_block_reason"] = why
            t["node"] = None; t["gpu_idx"] = None
            events.append({"type": "blocked", "task_id": t["id"], "task": t, "reason": why})
            continue
        # Warning path: precheck passed (ok=True) but reason starts with 'warn:' — log it on
        # the task so user sees it in status, but proceed with launch.
        if why and why.startswith("warn:"):
            t["last_block_reason"] = why
            events.append({"type": "git_warn", "task_id": t["id"], "task": t, "reason": why})
        resume = find_resume(t)
        if resume:
            t["resume_from"] = resume
            events.append({"type": "resume_found", "task_id": t["id"], "resume_from": resume})
        # Item 5 follow-up (WAL): persist "launching" status BEFORE ssh so a SIGKILL during
        # the ssh window leaves a forensics breadcrumb. Watcher startup (item 5 recovery)
        # scans for stale `launching` tasks > LAUNCHING_RESET_S old and reverts them to
        # queued — coupled with auto-adopt picking up any orphan GPU procs, this closes the
        # bulk of the orphan window. Fully closing it would require remote-side proc-scan
        # by signature, deferred.
        t["status"] = "launching"
        t["launching_started_at"] = time.time()
        try:
            save_state(state)
        except Exception:
            pass
        ok, msg = launch(t)
        if not ok:
            # Don't terminate the task — return it to the queue so dispatch can try a different
            # node next cycle. Common failure modes (ssh timeout, cwd missing on the picked
            # fallback node) are node-specific and recoverable on a different node. After
            # MAX_LAUNCH_RETRY consecutive failures, give up and escalate via heal so the user
            # gets a real diagnosis instead of an indefinite retry loop.
            attempted_node = t.get("node")
            t["launch_fail_count"] = (t.get("launch_fail_count") or 0) + 1
            if attempted_node:
                failed_nodes = t.setdefault("launch_failed_nodes", {})
                if not isinstance(failed_nodes, dict):
                    failed_nodes = {}
                    t["launch_failed_nodes"] = failed_nodes
                failed_nodes[attempted_node] = {
                    "ts": time.time(),
                    "attempt": t["launch_fail_count"],
                    "error": msg[:300],
                }
            t["last_block_reason"] = f"launch attempt {t['launch_fail_count']}/{MAX_LAUNCH_RETRY}: {msg}"
            t["node"] = None; t["gpu_idx"] = None
            if t["launch_fail_count"] >= MAX_LAUNCH_RETRY:
                t["status"] = "failed"
                # Best-effort heal escalation; failure to write must not crash dispatch.
                try: _write_escalation(t, "LAUNCH_FAIL_CAP", {"reason": msg, "tail": msg})
                except Exception: pass
                events.append({"type": "launch_failed_terminal", "task_id": t["id"], "task": t, "error": msg})
            else:
                t["status"] = "queued"  # back to queue, picker will try a different node
                events.append({"type": "launch_failed_retry", "task_id": t["id"], "task": t, "error": msg})
            continue
        t.pop("launch_fail_count", None)
        t.pop("launch_failed_nodes", None)
        events.append({"type": "launched", "task_id": t["id"], "task": t, "msg": msg})
        # Codex P0: durable persistence after each successful launch, NOT just at end of
        # dispatch. Window between ssh-launch returning and end-of-loop save_state was a
        # potential orphan source: scheduler SIGKILL'd here → remote process running, but
        # queue.json never updated to reflect status=running + remote_pids → next watcher
        # can't track or cancel it. save_state per-launch reduces orphan window from
        # "rest of loop iteration" to "syscall between launch return and disk fsync".
        try:
            save_state(state)
        except Exception as _e:
            notify("save_state_after_launch_failed",
                   {"id": t["id"], "error": str(_e)[:200]}, feishu_enabled=False)
        if sig:
            # Treat a just-launched task as running for the rest of this dispatch pass.
            # Otherwise two queued tasks with the same signature can both launch in one
            # iteration even though the precomputed running_sigs set was empty at loop start.
            running_sigs.add(sig)
        # Reflect placement in our local probe so subsequent iterations of this same dispatch
        # see the resources as already consumed (CPU + RAM at node level, VRAM at GPU level).
        for n in nodes:
            if n["name"] != t["node"]: continue
            n["free_cpu"] = max(0, n.get("free_cpu", 0) - t.get("cpu_cores", DEFAULT_CPU_CORES))
            n["free_ram_mb"] = max(0, n.get("free_ram_mb", 0) - t.get("ram_mb", DEFAULT_RAM_MB))
            n["running_count"] = n.get("running_count", 0) + 1  # for max_concurrent_running cap
            # Phase 2.16: bump slurm pending count too so pick_placement in subsequent loop
            # iterations sees this just-sbatched task and respects the per-node cap.
            if _is_slurm_managed(t):
                n["slurm_pending_count"] = n.get("slurm_pending_count", 0) + 1
            if t.get("gpu_idx") is None: continue  # CPU-only task
            for g in n["gpus"]:
                if g["idx"] != t["gpu_idx"]: continue
                g["used_mb"] += t["est_vram_mb"]
                g["free_mb"] = max(0, g["free_mb"] - t["est_vram_mb"])
    return events, len(queued)

def _print_node_summary(nodes):
    print("=== nodes ===")
    for n in nodes:
        if not n["alive"]:
            print(f"  {n['name']:11s} DOWN ({n.get('error','?')})"); continue
        gpu_parts = []
        for g in n["gpus"]:
            mem_pct = int(round(100 * g["used_mb"] / max(g["total_mb"], 1)))
            gpu_parts.append(f"GPU{g['idx']}={g['used_mb']}/{g['total_mb']}MB(mem:{mem_pct}%, util:{g['util_pct']}%)")
        gpu_str = ", ".join(gpu_parts)
        load = n.get("loadavg", 0)
        cpu_str = f"cpu={n.get('free_cpu', '?')}/{n.get('total_cpu', '?')}(load {load:.1f})"
        ram_str = f"ram_free={n['free_ram_mb']}MB"
        print(f"  {n['name']:11s} {gpu_str}  {cpu_str}  {ram_str}")

def _format_task_location(task):
    if not task.get("node"):
        return "-"
    if task.get("slurm_job_id"):
        kind = "SLURM-GPU" if int(task.get("est_vram_mb") or 0) > 0 else "SLURM-CPU"
        state = task.get("slurm_state")
        state_part = f":{state}" if state else ""
        return f"{task['node']}:{kind}#{task['slurm_job_id']}{state_part}"
    if task.get("gpu_idx") is None:
        return f"{task['node']}:CPU"
    return f"{task['node']}:GPU{task['gpu_idx']}"

_SLURM_ALIVE_STATES = {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING",
                        "RESIZING", "REQUEUED", "SUSPENDED"}


def _try_recover_orphan_slurm_job(task: dict, node: str) -> bool:
    """Phase 2.15 P2: look for an orphan slurm job named `scheduleurm-<task_id>`
    on `node`. If found in an alive state, adopt it onto the task (set
    slurm_job_id, transition to running) so we don't double-submit when the
    next dispatch tries to launch the still-queued task again.

    The orphan window: SlurmBackend.launch persists status='launching' BEFORE
    sbatch (WAL). If sbatch returns success but the scheduler process dies
    before status='running' + slurm_job_id can be flushed, slurm has the job
    (running 24h walltime by default) but scheduleurm forgot. Without this
    recovery, watcher startup reverts launching → queued, next dispatch sees
    a fresh queued task and sbatches AGAIN — slurm now has two copies running
    the same workload.

    Returns True iff orphan was found + adopted (caller should NOT revert).
    """
    job_name = f"scheduleurm-{task['id']}"
    # squeue -h: no header. -n NAME: filter by name. -t all: include finished.
    # -o "%i %T": "<jobid> <state>" per line.
    cmd = f"squeue -h -n {shlex.quote(job_name)} -t all -o '%i %T' 2>/dev/null"
    try:
        rc, out, _ = run_on(node, cmd, timeout=10, check=False)
    except Exception:
        return False
    if rc != 0 or not out.strip():
        return False
    # Pick the first matching alive job (defensive: should be at most one)
    for line in out.splitlines():
        bits = line.strip().split()
        if len(bits) < 2 or not bits[0].isdigit():
            continue
        jid = int(bits[0])
        slurm_state = bits[1].upper()
        if slurm_state not in _SLURM_ALIVE_STATES:
            # Job is already terminal in slurm — safer to let the revert path handle:
            # the task goes back to queued, next dispatch creates a fresh job, the
            # terminal slurm record stays as a forensic breadcrumb but doesn't
            # interfere.
            continue
        task["slurm_job_id"] = jid
        task["status"] = "running"
        task["remote_pids"] = []  # slurm-managed: no host PIDs to track
        task["started_at"] = task.get("launching_started_at") or time.time()
        task["peak_vram_mb"] = 0
        task["peak_ram_mb"] = 0
        task.pop("launching_started_at", None)
        task["last_block_reason"] = (
            f"WAL recovery: adopted orphan slurm job {jid} on {node} "
            f"(state={slurm_state}); avoids double-submit"
        )
        return True
    return False


def recover_stale_launching_tasks(state, now: Optional[float] = None, reset_s: int = LAUNCHING_RESET_S) -> int:
    """Revert stale WAL launch markers to queued under state_lock.

    A task is set to status='launching' immediately before the ssh/docker launch call. If
    that scheduler process dies before launch() flips it to running, the task would
    otherwise be invisible to dispatch forever. Keeping this recovery in one helper makes
    dispatch, watcher, status, and wait-for share the same active-state invariant.

    Phase 2.15 P2: for slurm-routed launching tasks, BEFORE reverting we check
    whether sbatch actually succeeded (orphan slurm job named scheduleurm-<id>
    present in squeue). If yes, adopt the orphan onto the task — prevents
    double-submission when the next dispatch tries to re-launch.

    Returns count of tasks reverted (NOT including those recovered as orphans).
    """
    now = time.time() if now is None else now
    reverted = 0
    for t in state.get("tasks", []):
        if t.get("status") != "launching":
            continue
        age = now - (t.get("launching_started_at") or now)
        if age < reset_s:
            continue
        # Phase 2.15 P2: try slurm-orphan recovery first. The check is per-task:
        # only slurm-routed nodes have orphans to recover.
        node = t.get("node")
        if node and not _BACKEND.requires_local_capacity_check(node):
            if _try_recover_orphan_slurm_job(t, node):
                continue  # adopted; do not revert
        t["status"] = "queued"
        t["last_block_reason"] = (
            f"WAL recovery: was 'launching' for {max(0, age):.0f}s, reverted to queued"
        )
        t.pop("launching_started_at", None)
        reverted += 1
    return reverted

def cmd_wait_for(args):
    """Block until all matching tasks reach a terminal state (done/failed/cancelled), then exit.

    Match by --signature (fnmatch glob over task signatures) and/or --task-id (one or more).
    Combine: tasks must match at least one of the criteria.

    Polls every --poll seconds (default 30). Times out after --timeout seconds (default 14400 = 4 h).

    Exit codes:
        0 — all matched tasks reached terminal state
        1 — timeout while at least one task still running/launching/queued
        2 — no matching tasks ever found before timeout

    Designed to be wrapped in `Bash run_in_background` so its exit fires a task-notification
    that wakes the parent Claude session for the next orchestration step.
    """
    import fnmatch as _fnmatch
    deadline = time.time() + args.timeout if args.timeout > 0 else float("inf")
    seen_ids = set()
    last_print = 0.0
    while True:
        with state_lock():
            state = load_state()
            recover_stale_launching_tasks(state)
            update_running_tasks(state)
            save_state(state)
        # Build the candidate set: any task matching ANY of the supplied filters.
        matches = []
        sig_glob = args.signature
        ids = set(args.task_ids) if args.task_ids else set()
        for t in state["tasks"]:
            hit = False
            if ids and t["id"] in ids:
                hit = True
            elif sig_glob and _fnmatch.fnmatch(t.get("signature", ""), sig_glob):
                hit = True
            if hit:
                matches.append(t)
        if not matches:
            if seen_ids:
                # Task records were forgotten/purged after we'd seen them — treat as done.
                print(f"[wait-for] all {len(seen_ids)} previously-matched tasks gone — assumed terminal")
                return 0
            if time.time() > deadline:
                print("[wait-for] timeout: no matching tasks ever found")
                return 2
            time.sleep(args.poll)
            continue
        seen_ids.update(t["id"] for t in matches)
        terminal_states = ("done", "failed", "cancelled")
        terminal = [t for t in matches if t["status"] in terminal_states]
        running = [t for t in matches if t["status"] == "running"]
        launching = [t for t in matches if t["status"] == "launching"]
        queued = [t for t in matches if t["status"] == "queued"]
        if len(terminal) == len(matches):
            done_n = sum(1 for t in terminal if t["status"] == "done")
            fail_n = sum(1 for t in terminal if t["status"] == "failed")
            canc_n = sum(1 for t in terminal if t["status"] == "cancelled")
            tag = sig_glob or f"{len(ids)} ids"
            print(f"[wait-for {tag}] all {len(matches)} terminal: {done_n} done, {fail_n} failed, {canc_n} cancelled")
            return 0
        if time.time() > deadline:
            tag = sig_glob or f"{len(ids)} ids"
            print(f"[wait-for {tag}] timeout: {len(running)} running, {len(launching)} launching, {len(queued)} queued, {len(terminal)} terminal")
            return 1
        # Periodic progress line every ~5 minutes (no spam, but enough to confirm the wait is alive).
        now = time.time()
        if args.verbose and now - last_print >= 300:
            tag = sig_glob or f"{len(ids)} ids"
            print(f"[wait-for {tag}] {len(running)} running, {len(launching)} launching, {len(queued)} queued, {len(terminal)}/{len(matches)} terminal")
            last_print = now
        time.sleep(args.poll)


def _preload_docker_images_outside_lock():
    """Walk queued tasks, preload required envs (docker images / conda envs) to candidate
    nodes BEFORE the state_lock-protected dispatch loop runs. Without this, an env push
    (docker save: 30min, conda rsync: similar) inside `with state_lock():` blocks
    status/cancel/watcher iterations for the entire window.

    Reads queue.json WITHOUT state_lock — survey-only, then env_deploy.push_image /
    push_conda_env per (node, env). Docker is push-on-missing/drift; conda is always
    incremental rsync so local env changes propagate even when remote python already works.
    Multiple pushes serial here but OUTSIDE the lock — concurrent status/cancel works.

    Codex P0a: use `spec_image or image_field` so tasks with `docker:IMAGE` inline (no
    separate `--image` flag) still get preloaded.
    """
    if env_deploy is None:
        return
    try:
        state = load_state()
    except Exception:
        return
    # Build (node, kind, payload) tuples for everything that needs preload
    needed_docker: set[tuple[str, str]] = set()  # (node, image)
    needed_conda: set[tuple[str, str]] = set()   # (node, env_path)
    for t in state.get("tasks", []):
        if t.get("status") != "queued": continue
        spec = t.get("env_spec") or "none"
        if spec == "none": continue
        try:
            kind, spec_payload = env_deploy.parse_env_spec(spec)
        except ValueError:
            continue
        require = t.get("require_node")
        candidates = [require] if require else list(NODES.keys())
        if kind == "docker" or (kind == "auto" and (spec_payload or t.get("image"))):
            chosen = spec_payload or (t.get("image") or "")
            if not chosen: continue
            for n in candidates:
                needed_docker.add((n, chosen))
        elif kind == "conda":
            if not spec_payload: continue
            # Skip non-existent local source — caller's mistake, eventual launch failure is
            # diagnosed through the normal ENV_MISSING path.
            if not Path(spec_payload).is_absolute(): continue
            if not Path(spec_payload).is_dir(): continue
            for n in candidates:
                if NODES.get(n, {}).get("host") is None: continue  # local: nothing to push
                needed_conda.add((n, spec_payload))
    # ---- docker preload ----
    local_digests: dict[str, Optional[str]] = {}
    for _, image in needed_docker:
        if image not in local_digests:
            local_digests[image] = env_deploy.get_image_digest(run_on, "local", image)
    for node, image in needed_docker:
        try:
            if not env_deploy.has_docker(run_on, node, timeout=8):
                continue
            if env_deploy.has_image(run_on, node, image, local_digest=local_digests.get(image)):
                continue
            node_host = NODES.get(node, {}).get("host")
            ok, msg = env_deploy.push_image(node_host, image, timeout_s=1800)
            ev = "preload_image_ok" if ok else "preload_image_failed"
            notify(ev, {"node": node, "image": image, "msg": msg[:200] if not ok else ""},
                   feishu_enabled=False)
        except Exception as e:
            notify("preload_image_error",
                   {"node": node, "image": image, "error": str(e)[:200]},
                   feishu_enabled=False)
    # ---- conda preload (rsync local env to remote at same absolute path) ----
    for node, env_path in needed_conda:
        try:
            node_host = NODES.get(node, {}).get("host")
            ok, msg = env_deploy.push_conda_env(node_host, env_path, env_path, timeout_s=3600)
            ev = "preload_conda_ok" if ok else "preload_conda_failed"
            notify(ev, {"node": node, "env_path": env_path, "msg": msg[:300] if not ok else ""},
                   feishu_enabled=False)
        except Exception as e:
            notify("preload_conda_error",
                   {"node": node, "env_path": env_path, "error": str(e)[:200]},
                   feishu_enabled=False)


def cmd_dispatch(args):
    recovered_launching = 0
    # Quick pre-flight recovery: make stale WAL launch markers queued BEFORE preload scans.
    # The expensive env push/sync remains outside state_lock; this short lock only rewrites
    # stale launching tasks so conda/docker preload sees them in the same dispatch cycle.
    try:
        with state_lock():
            state = load_state()
            recovered_launching = recover_stale_launching_tasks(state)
            if recovered_launching:
                save_state(state)
    except Exception as _e:
        notify("launching_recovery_error", {"error": str(_e)[:200]}, feishu_enabled=False)
    # Pre-flight: push docker images / rsync conda envs for queued tasks BEFORE the main
    # dispatch state_lock. Image push (`docker save | ssh node docker load`) and conda
    # rsync can take minutes and would otherwise block status/cancel/watcher iterations.
    # Now: scan queue for envs that need preload, push/sync outside the lock; dispatch sees
    # them already present and normally never blocks on env delivery itself.
    try:
        _preload_docker_images_outside_lock()
    except Exception as _e:
        notify("preload_error", {"error": str(_e)[:200]}, feishu_enabled=False)
    # Phase 3.0.5 P1 fix: stage migration candidates (rsync cwd/ckpt) BEFORE the main
    # state_lock. Without this, _do_dispatch's _consider_migration would call
    # _stage_for_migration inside the lock — a 5GB ckpt rsync (timeout=600s) would
    # block submit/cancel/status/watcher for up to 10 min. Now staging side-effects
    # land in the process-local _STAGED_TASKS cache; _consider_migration inside the
    # lock is a fast dict lookup (microseconds).
    try:
        _stage_migration_candidates_outside_lock()
    except Exception as _e:
        notify("migration_staging_error_outer", {"error": str(_e)[:200]},
               feishu_enabled=False)
    with state_lock():
        state = load_state()
        recovered_launching += recover_stale_launching_tasks(state)
        update_running_tasks(state)
        nodes = probe_all()
        _reserve_inflight_vram(state, nodes)
        _print_node_summary(nodes)
        events, qcount = _do_dispatch(state, nodes)
        save_state(state)
    if recovered_launching:
        notify("launching_state_recovered", {"reverted_count": recovered_launching},
               feishu_enabled=False)
    if qcount == 0:
        print("=== dispatch === (nothing queued)")
        return
    print(f"=== dispatch === ({qcount} queued)")
    for ev in events:
        tid = ev["task_id"]
        if ev["type"] == "no_fit":
            t = ev["task"]
            need = t.get("est_vram_mb", 0)
            kind = "no node fits CPU/RAM right now" if need <= 0 else f"no GPU fits {need}MB right now"
            print(f"  [{tid}] WAIT  {kind} ({t['description'][:50]})")
        elif ev["type"] == "blocked":
            print(f"  [{tid}] BLOCK {ev['reason']}")
        elif ev["type"] == "resume_found":
            print(f"  [{tid}] resume_from={ev['resume_from']}")
        elif ev["type"] == "launched":
            t = ev["task"]
            print(f"  [{tid}] LAUNCH on {_format_task_location(t)}  {ev['msg']}  log={t['log_path']}")
        elif ev["type"] == "launch_failed_retry":
            t = ev["task"]
            print(f"  [{tid}] RETRY launch failed ({t.get('launch_fail_count', '?')}/{MAX_LAUNCH_RETRY}): {ev['error'][:200]}")
        elif ev["type"] == "launch_failed_terminal":
            print(f"  [{tid}] FAIL  launch failed permanently: {ev['error'][:200]}")

# ---------- watcher (background daemon) ----------
WATCHER_LOG = STATE_DIR / "logs" / "watcher.log"
WATCHER_STATE = STATE_DIR / ".watcher_state.json"
FEISHU_CONFIG = Path.home() / ".claude" / "feishu.json"

def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False

def _load_feishu_cfg():
    """Return dict with 'webhook_url' if push mode is configured, else None."""
    if not FEISHU_CONFIG.exists():
        return None
    try:
        cfg = json.loads(FEISHU_CONFIG.read_text())
    except Exception:
        return None
    if cfg.get("mode") != "push":
        return None
    if not cfg.get("webhook_url"):
        return None
    return cfg

def _send_feishu(webhook_url, text):
    """POST a Feishu text message. Best-effort: log+swallow errors. Timeout 5s."""
    payload = json.dumps({"msg_type": "text", "content": {"text": text}}).encode()
    req = urllib.request.Request(webhook_url, data=payload,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            r.read()
    except Exception as e:
        # Don't crash the watcher on Feishu hiccups — just record.
        try:
            with open(WATCHER_LOG, "a") as f:
                f.write(json.dumps({"ts": time.time(), "type": "feishu_send_failed", "error": str(e)[:200]}) + "\n")
        except Exception:
            pass

def _format_feishu(event_type, payload):
    """Render an event as a single-line text Feishu push. Keep it concise."""
    if event_type == "task_done":
        t = payload
        runtime_min = (t.get("finished_at", time.time()) - t.get("started_at", time.time())) / 60
        return (f"[scheduler] ✅ {t['id']} {t.get('project','?')} done — "
                f"{runtime_min:.1f}m, peak {t.get('peak_vram_mb', 0)}MB on {_format_task_location(t)} | "
                f"{t.get('description','')[:60]}")
    if event_type == "task_launched":
        t = payload
        handle = f"slurm_job_id={t['slurm_job_id']}" if t.get("slurm_job_id") else \
                 f"pid={(t.get('remote_pids') or ['?'])[0]}"
        return (f"[scheduler] 🚀 {t['id']} {t.get('project','?')} launched on "
                f"{_format_task_location(t)} {handle} | {t.get('description','')[:60]}")
    if event_type == "task_auto_adopted":
        t = payload
        return (f"[scheduler] 👁 auto-adopted {t['id']} {t.get('project','?')} on "
                f"{_format_task_location(t)} ({len(t.get('remote_pids', []))} procs, {t.get('peak_vram_mb', 0)}MB) — "
                f"launched outside scheduler, now tracked")
    if event_type == "heartbeat":
        return (f"[scheduler] ⏱ heartbeat — running:{payload['running']} "
                f"launching:{payload.get('launching', 0)} queued:{payload['queued']} | "
                + " ; ".join(payload["nodes"]))
    if event_type == "watcher_started":
        return f"[scheduler] watcher started (pid={payload['pid']}, interval={payload['interval']}s, heartbeat={payload['heartbeat']}s)"
    if event_type == "watcher_stopped":
        return f"[scheduler] watcher stopped (pid={payload['pid']})"
    if event_type == "task_blocked":
        return f"[scheduler] ⚠ {payload['task_id']} blocked: {payload['reason']}"
    if event_type == "task_launch_retry":
        return (f"[scheduler] ⚠ {payload['task_id']} launch failed "
                f"({payload.get('attempt','?')}/{MAX_LAUNCH_RETRY}); will retry/fallback: "
                f"{payload['error'][:120]}")
    if event_type == "task_failed":
        return f"[scheduler] ❌ {payload['task_id']} launch failed: {payload['error'][:120]}"
    if event_type == "task_crashed":
        p = payload
        tail_short = p.get("tail", "").replace("\n", " | ")[-200:]
        return (f"[scheduler] 💥 {p['id']} {p.get('project','?')} CRASHED on {p.get('node')}:GPU{p.get('gpu_idx')} — "
                f"died after {p.get('lifetime_s', 0)}s, log {p.get('log_size', 0)}B. "
                f"reason: {p.get('reason','?')[:120]}. tail: {tail_short}")
    if event_type == "heal_awaiting_claude":
        p = payload
        return (f"[scheduler] 🤖 heal needs Claude — {p.get('prefix','?')} ({p.get('category','?')}) on {p.get('node','?')}: "
                f"{p.get('question','need decision')[:200]} | inbox: ~/.claude/scheduler/HEAL_NEEDS_CLAUDE.md")
    if event_type == "heal_awaiting_user":
        p = payload
        return (f"[scheduler] 🆘 heal AWAITING_USER (Claude couldn't decide) — {p.get('prefix','?')} ({p.get('category','?')}) on {p.get('node','?')}: "
                f"{p.get('question','need decision')[:200]} | see ~/.claude/scheduler/HEAL_NEEDS_USER.md")
    if event_type == "batch_complete":
        b = payload
        elapsed_min = (b.get("latest_finish", 0) - b.get("earliest_submit", 0)) / 60
        bits = []
        if b.get("done"): bits.append(f"✅{b['done']}")
        if b.get("failed"): bits.append(f"💥{b['failed']}")
        if b.get("cancelled"): bits.append(f"🚫{b['cancelled']}")
        return (f"[scheduler] 🏁 batch '{b['prefix']}' complete — {b['total']} tasks ({' '.join(bits)}) "
                f"in {elapsed_min:.1f}m | ids: {','.join(b.get('task_ids', [])[:5])}"
                + ("..." if len(b.get('task_ids', [])) > 5 else ""))
    return f"[scheduler] {event_type}: {json.dumps(payload)[:200]}"

def notify(event_type, payload, feishu_enabled=True):
    """Always log to watcher.log (JSONL, with size-based rotation). Optionally push to Feishu if config + enabled."""
    line = json.dumps({"ts": time.time(), "type": event_type, "payload": payload}, default=str)
    try:
        WATCHER_LOG.parent.mkdir(parents=True, exist_ok=True)
        maybe_rotate_log(WATCHER_LOG, WATCHER_LOG_MAX_MB, WATCHER_LOG_GENERATIONS)
        with open(WATCHER_LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass
    if not feishu_enabled:
        return
    cfg = _load_feishu_cfg()
    if not cfg:
        return
    _send_feishu(cfg["webhook_url"], _format_feishu(event_type, payload))

def _node_processes(name):
    """One ssh call per node — fetch every GPU compute app's (pid, gpu_idx, used_mb, owner, cwd, rss_mb).
    Used by the watcher to discover externally-launched tasks. Returns [] on failure."""
    # cmdline capture: read /proc/<pid>/cmdline (NUL-separated args), translate to spaces,
    # cap at 4KB. Used by adopt to record the real launch command so a crashed adopted task
    # can be auto-requeued instead of disappearing silently. Field 6 (cmdline) may contain
    # `|` chars in args — Python parser uses maxsplit=5 to keep cmdline intact.
    # Slurm-detection (Phase 2.2): grep /proc/<pid>/environ for SLURM_JOB_ID (modern) or
    # SLURM_JOBID (legacy). PIDs that are slurm-managed get sl=1 and the caller skips them
    # to prevent double-tracking — when scheduleurm submits via SlurmBackend, the task is
    # already tracked by slurm_job_id; auto-adopt would duplicate the same workload.
    # Reading environ requires owning the process; slurm jobs run as the submitting user.
    cmd = (
        "nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader,nounits 2>/dev/null; "
        "echo '===BUS==='; "
        "nvidia-smi --query-gpu=index,gpu_bus_id --format=csv,noheader,nounits 2>/dev/null; "
        "echo '===META==='; "
        "for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null); do "
        "  o=$(ps -o user= -p \"$p\" 2>/dev/null | tr -d ' '); "
        "  c=$(readlink /proc/$p/cwd 2>/dev/null); "
        "  r=$(awk '/^VmRSS:/ {print $2}' /proc/$p/status 2>/dev/null); "
        "  pc=$(ps -o pcpu= -p \"$p\" 2>/dev/null | tr -d ' '); "
        "  pg=$(ps -o pgid= -p \"$p\" 2>/dev/null | tr -d ' '); "
        "  sl=0; grep -aqE 'SLURM_JOB_ID=|SLURM_JOBID=' /proc/$p/environ 2>/dev/null && sl=1; "
        "  cl=$(head -c 4096 /proc/$p/cmdline 2>/dev/null | tr '\\0' ' ' | sed 's/[[:space:]]*$//'); "
        "  echo \"${p}|${o}|${c}|${r}|${pc}|${pg}|${sl}|${cl}\"; "
        "done"
    )
    try:
        rc, out, _ = run_on(name, cmd, timeout=20, check=False)
        if rc != 0: return []
    except Exception:
        return []
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    bus_sep = lines.index("===BUS===") if "===BUS===" in lines else 0
    meta_sep = lines.index("===META===") if "===META===" in lines else len(lines)
    proc_lines = lines[:bus_sep]
    gpu_lines = lines[bus_sep+1:meta_sep]
    meta_lines = lines[meta_sep+1:]
    bus_to_idx = {}
    for gl in gpu_lines:
        parts = [x.strip() for x in gl.split(",")]
        if len(parts) >= 2:
            try: bus_to_idx[parts[1]] = int(parts[0])
            except ValueError: continue
    pid_meta = {}
    for ml in meta_lines:
        # maxsplit=7 keeps the cmdline (last field) intact even if it contains `|` chars.
        # Phase 2.2: extra `sl` field between pgid and cmdline indicates slurm-managed PID.
        bits = ml.split("|", 7)
        if len(bits) >= 3:
            owner = bits[1].strip()
            cwd = bits[2].strip()
            rss_kb = 0
            if len(bits) >= 4 and bits[3].strip().isdigit():
                rss_kb = int(bits[3].strip())
            try:
                pcpu = float(bits[4].strip()) if len(bits) >= 5 and bits[4].strip() else 0.0
            except ValueError:
                pcpu = 0.0
            pgid = None
            if len(bits) >= 6 and bits[5].strip().isdigit():
                pgid = int(bits[5].strip())
            is_slurm = (len(bits) >= 7 and bits[6].strip() == "1")
            cmdline = bits[7].strip() if len(bits) >= 8 else ""
            try: pid_meta[int(bits[0])] = (owner, cwd, rss_kb, pcpu, pgid, is_slurm, cmdline)
            except ValueError: continue
    out_list = []
    for pl in proc_lines:
        parts = [x.strip() for x in pl.split(",")]
        if len(parts) < 3: continue
        try:
            pid = int(parts[1]); used = int(parts[2])
        except ValueError: continue
        gpu_idx = bus_to_idx.get(parts[0])
        if gpu_idx is None: continue
        owner, cwd, rss_kb, pcpu, pgid, is_slurm, cmdline = pid_meta.get(
            pid, ("", "", 0, 0.0, None, False, "")
        )
        out_list.append({"node": name, "pid": pid, "gpu_idx": gpu_idx, "used_mb": used,
                          "owner": owner, "cwd": cwd, "rss_mb": rss_kb // 1024,
                          "pcpu": pcpu, "pgid": pgid or pid, "cmdline": cmdline,
                          "is_slurm": is_slurm})
    return out_list

def _node_ppid_map(name):
    """Build {pid: ppid} for the entire node. Used by _reconcile_external_tasks to avoid
    adopting a child of an already-tracked PID as a separate phantom task. Without this, a
    scheduler-launched bash wrapper (tracked PID) and its python child (different PID) get
    counted as TWO tasks, doubling resource accounting and eating concurrency cap slots."""
    try:
        rc, out, _ = run_on(name, "ps -eo pid=,ppid=", timeout=15, check=False)
        if rc != 0: return {}
    except Exception:
        return {}
    ppid_of = {}
    for line in out.splitlines():
        bits = line.split()
        if len(bits) != 2: continue
        try: ppid_of[int(bits[0])] = int(bits[1])
        except ValueError: continue
    return ppid_of

_DESCENDANTS_CAP = 500  # protect probe from a runaway fork-bomb (item 21)

def _descendants_of(roots, ppid_of):
    """Return set of all PIDs whose ancestor chain (via ppid_of) hits any pid in `roots`.
    Excludes the roots themselves. PIDs without a chain to a root are skipped.

    Capped at _DESCENDANTS_CAP entries — a buggy script forking thousands of children
    (e.g. PyTorch DataLoader leak, gunicorn explosion) would otherwise blow up the BFS
    output, the killing cmd line, AND ssh stdout buffer. Hitting the cap is itself a
    diagnostic signal: caller should treat it as "process tree is sick, kill everything".
    Codex item 21."""
    if not roots or not ppid_of:
        return set()
    descendants = set()
    # Walk every PID upward; if the chain hits a root, mark every step on the path.
    cache = {}  # pid → True/False (under a root)
    for pid in ppid_of:
        if len(descendants) >= _DESCENDANTS_CAP:
            break  # cap hit; stop scanning
        if pid in roots: continue
        path = []
        cur = pid
        guard = 0
        while cur and cur != 1 and guard < 64:
            if cur in cache:
                under = cache[cur]
                break
            if cur in roots:
                under = True
                break
            path.append(cur)
            cur = ppid_of.get(cur)
            guard += 1
        else:
            under = False
        for n in path:
            cache[n] = under
            if under: descendants.add(n)
            if len(descendants) >= _DESCENDANTS_CAP:
                break
    return descendants

def _node_cpu_processes(name):
    """Find user-owned CPU-burning python processes NOT in nvidia-smi compute-apps. Catches CPU-only
    workloads (eval scripts, multi-worker batches, etc.) that the GPU-only probe misses.
    Threshold: pcpu >= 50% (half a sustained core). Returns same shape as _node_processes plus pcpu."""
    import getpass
    me = getpass.getuser()
    # cmdline appended for adopt-time cmd capture; see _node_processes for rationale.
    # Phase 2.2: emit `sl` flag (1 = slurm-managed) so caller skips slurm-owned procs.
    script = (
        f"PIDS=$(ps -eo pid,user,pcpu,cmd --no-headers 2>/dev/null | "
        f"awk -v u={me} '$2==u && $4~/python/ && $3+0>=50 {{print $1}}'); "
        f"for p in $PIDS; do "
        f"  pcpu=$(ps -o pcpu= -p $p 2>/dev/null | tr -d ' '); "
        f"  rss=$(ps -o rss= -p $p 2>/dev/null | tr -d ' '); "
        f"  pg=$(ps -o pgid= -p $p 2>/dev/null | tr -d ' '); "
        f"  cwd=$(readlink /proc/$p/cwd 2>/dev/null); "
        f"  sl=0; grep -aqE 'SLURM_JOB_ID=|SLURM_JOBID=' /proc/$p/environ 2>/dev/null && sl=1; "
        f"  cl=$(head -c 4096 /proc/$p/cmdline 2>/dev/null | tr '\\0' ' ' | sed 's/[[:space:]]*$//'); "
        f"  echo \"${{p}}|${{pcpu}}|${{rss}}|${{pg}}|${{cwd}}|${{sl}}|${{cl}}\"; "
        f"done"
    )
    try:
        rc, out, _ = run_on(name, script, timeout=20, check=False)
        if rc != 0: return []
    except Exception:
        return []
    out_list = []
    for line in out.splitlines():
        # maxsplit=6 keeps cmdline (last field) intact even if it contains `|` chars.
        # Phase 2.2: extra `sl` field after cwd indicates slurm-managed PID.
        bits = line.strip().split("|", 6)
        if len(bits) < 5: continue
        try:
            pid = int(bits[0])
            pcpu = float(bits[1]) if bits[1] else 0.0
            rss_kb = int(bits[2]) if bits[2].isdigit() else 0
            pgid = int(bits[3]) if bits[3].isdigit() else pid
        except ValueError:
            continue
        cwd = bits[4].strip()
        is_slurm = (len(bits) >= 6 and bits[5].strip() == "1")
        cmdline = bits[6].strip() if len(bits) >= 7 else ""
        out_list.append({"node": name, "pid": pid, "owner": me, "rss_mb": rss_kb // 1024,
                          "cwd": cwd, "pcpu": pcpu, "gpu_idx": None, "used_mb": 0,
                          "is_cpu_only": True, "pgid": pgid, "cmdline": cmdline,
                          "is_slurm": is_slurm})
    return out_list

def _refresh_adopted_resources(state, gpu_proc_lists, cpu_proc_lists):
    """For already-running auto_adopted tasks, re-estimate cpu_cores from current %CPU
    measurements. Fixes the original sin where len(pids) overestimated multi-worker tasks
    (e.g. 28 SUMO workers averaging 30% CPU each → was 28, should be ~9).
    Also refreshes ram_mb from current RSS sum (downward only — never inflate)."""
    import math as _math
    # Build pid → (pcpu, rss_mb, pgid) map across ALL probed nodes for fast lookup.
    pid_stats = {}
    for plist in gpu_proc_lists + cpu_proc_lists:
        for p in plist:
            key = (p["node"], p["pid"])
            pid_stats[key] = (p.get("pcpu", 0.0), p.get("rss_mb", 0), p.get("pgid"))
    for t in state["tasks"]:
        if t.get("status") != "running" or not t.get("auto_adopted"): continue
        node = t.get("node")
        pids = _task_pids(t)
        if not node or not pids: continue
        # Aggregate current pcpu + rss across this task's still-visible PIDs.
        sum_pcpu, sum_rss = 0.0, 0
        pgids_seen = set()
        seen = 0
        for p in pids:
            if (node, p) in pid_stats:
                pc, rs, pg = pid_stats[(node, p)]
                sum_pcpu += pc; sum_rss += rs; seen += 1
                if pg:
                    pgids_seen.add(pg)
        if seen == 0: continue  # task's PIDs aren't visible to probes — leave untouched
        if len(pgids_seen) == 1 and not t.get("process_group"):
            t["process_group"] = next(iter(pgids_seen))
        new_cpu = max(1, _math.ceil(sum_pcpu / 100.0)) if sum_pcpu > 0 else None
        # Track upward when the task ramps (initial pcpu often low during model-load / SUMO sim init,
        # then jumps when training kicks in — historical lower-only kept the under-estimate forever).
        # Allow downward only if current value is wildly inflated (>2.5× observed) — that's the
        # signature of an initial len(pids) over-count, not a transient dip.
        if new_cpu is not None:
            cur = t.get("cpu_cores", new_cpu)
            if new_cpu > cur:
                t["cpu_cores"] = new_cpu
            elif cur > new_cpu * 2.5:
                t["cpu_cores"] = new_cpu
            # otherwise keep current — protects against transient I/O / save-checkpoint dips
        if sum_rss > 0 and sum_rss < t.get("ram_mb", 10**9):
            t["ram_mb"] = sum_rss

def _reconcile_external_tasks(state):
    """Find external GPU AND CPU processes not in any tracked running task; auto-adopt them grouped
    by (node, gpu_or_None, project). Also refresh cpu_cores estimates of existing adopted tasks.
    Mutates state. Returns newly-adopted task records."""
    import getpass
    me = getpass.getuser()
    home_root = f"/home/{me}/"
    home_basename = me  # so a cwd of just /home/erzhu419 resolves to project=erzhu419 — we'll skip that
    tracked = set()
    for t in state["tasks"]:
        if t["status"] != "running" or not t.get("node"): continue
        for p in _task_pids(t):
            tracked.add((t["node"], int(p)))
    # Probe GPU compute-apps + CPU-burning python procs + PPID maps in parallel across nodes.
    # PPID map lets us reject a candidate whose ancestor is an already-tracked PID — fixes the
    # bash-wrapper-and-its-python-child being counted as two separate tasks.
    with ThreadPoolExecutor(max_workers=len(NODES) * 3) as ex:
        gpu_proc_lists = list(ex.map(_node_processes, NODES.keys()))
        cpu_proc_lists = list(ex.map(_node_cpu_processes, NODES.keys()))
        ppid_maps = list(ex.map(_node_ppid_map, NODES.keys()))
    ppid_by_node = dict(zip(NODES.keys(), ppid_maps))
    # Build per-node set of descendants of already-tracked PIDs. Any candidate PID in this set
    # is a child of a task we already know about → skip adoption.
    tracked_pids_by_node = {}
    scheduler_roots_by_node = {}
    for (n, p) in tracked:
        tracked_pids_by_node.setdefault(n, set()).add(p)
    for t in state["tasks"]:
        if t.get("status") != "running" or t.get("auto_adopted") or not t.get("node"):
            continue
        for p in _task_pids(t):
            scheduler_roots_by_node.setdefault(t["node"], set()).add(int(p))
    descendants_by_node = {n: _descendants_of(tracked_pids_by_node.get(n, set()),
                                                ppid_by_node.get(n, {}))
                           for n in NODES.keys()}
    scheduler_descendants_by_node = {n: _descendants_of(scheduler_roots_by_node.get(n, set()),
                                                        ppid_by_node.get(n, {}))
                                     for n in NODES.keys()}
    # Clean up phantom adopts created by older watcher versions: if an auto-adopted task's
    # PIDs are actually children of a scheduler-owned root PID, it is not an external task.
    # Mark it forgotten instead of killing anything.
    for t in state["tasks"]:
        if t.get("status") != "running" or not t.get("auto_adopted") or not t.get("node"):
            continue
        pids = set(int(p) for p in _task_pids(t))
        if pids and pids.issubset(scheduler_descendants_by_node.get(t["node"], set())):
            t["status"] = "forgotten"
            t["finished_at"] = time.time()
            t["last_block_reason"] = "auto-forgotten: duplicate child process of a scheduler-launched task"
    # Refresh resource estimates on already-adopted running tasks (lower-only) so the original
    # len(pids) overestimate self-corrects. Cheap because we already have the proc data.
    _refresh_adopted_resources(state, gpu_proc_lists, cpu_proc_lists)
    all_procs = []
    for procs in gpu_proc_lists: all_procs.extend(procs)
    # CPU procs: exclude any PID already counted as GPU compute-app on the same node.
    gpu_pids_per_node = {n: {p["pid"] for p in plist} for n, plist in zip(NODES.keys(), gpu_proc_lists)}
    for procs in cpu_proc_lists:
        for p in procs:
            if p["pid"] in gpu_pids_per_node.get(p["node"], set()): continue
            all_procs.append(p)
    # Filter to ours + new + project-shaped cwd. Reject children of already-tracked PIDs
    # (the scheduler-launched bash wrapper + its python child were being adopted as TWO tasks).
    # Phase 2.2: also reject slurm-managed PIDs (SLURM_JOB_ID set in /proc/<pid>/environ).
    # Reason: when scheduleurm submits via SlurmBackend, the actual user proc is launched by
    # slurmstepd not by scheduleurm — nvidia-smi sees its PID. Without this filter, the same
    # workload would be tracked twice (once via slurm_job_id, once as auto-adopted). Even
    # if user submits to slurm OUTSIDE scheduleurm, we still skip — they can submit through
    # scheduleurm if they want it tracked; otherwise, hands off.
    candidates = []
    for p in all_procs:
        if (p["node"], p["pid"]) in tracked: continue
        if p["pid"] in descendants_by_node.get(p["node"], set()): continue
        if p.get("is_slurm"): continue  # Phase 2.2: don't shadow slurm-managed work
        if p["owner"] != me: continue
        if not p["cwd"] or not p["cwd"].startswith(home_root): continue
        project = _project_from_path(p["cwd"])
        if not project or project == home_basename: continue
        candidates.append((p, project))
    if not candidates:
        return []
    # Group by (node, gpu_idx, project, pgid). The pgid split matters: independent experiments
    # from the same project can share a GPU or all be CPU-only; grouping only by project folded
    # them into one fake task.
    groups = {}
    for p, project in candidates:
        key = (p["node"], p["gpu_idx"], project, p.get("pgid") or p["pid"])
        groups.setdefault(key, []).append(p)
    # Adopt each group as one task.
    adopted = []
    for (node, gpu_idx, project, pgid), procs in groups.items():
        pids = sorted(p["pid"] for p in procs)
        sum_vram = sum(p["used_mb"] for p in procs)
        sum_rss = sum(p.get("rss_mb", 0) for p in procs)
        sum_pcpu = sum(p.get("pcpu", 0.0) for p in procs)
        cwd_sample = procs[0]["cwd"]
        # Capture cmdline from /proc — lets the watcher auto-requeue the task if it later
        # ends incomplete (lifetime < 50% of historical EWMA AND status != cancelled).
        # Pick the longest non-empty cmdline among grouped PIDs (multi-worker tasks may have
        # one master with full args + N children with truncated args; longest is the master).
        cmdlines = [p.get("cmdline", "") for p in procs if p.get("cmdline")]
        captured_cmd = max(cmdlines, key=len) if cmdlines else ""
        # cpu_cores estimate: sum of actual %CPU across all PIDs / 100, ceil. This is much more
        # accurate than len(pids) because multi-worker jobs (e.g. SUMO eval with 14 workers)
        # rarely peg every worker — they share GIL/IO and average to a fraction. History wins if seen.
        # Disambiguate auto-adopted sigs by smallest PID in the group: stable across watcher
        # restarts (PIDs survive), unique per process tree, and avoids the "all RE-SAC adopts
        # collapsed to one sig" effect that made the queue look like duplicates.
        sig = f"{project}/auto-adopted/p{min(pids)}"
        hist = history_get(sig) or {}
        import math as _math
        measured_cpu = max(1, _math.ceil(sum_pcpu / 100.0)) if sum_pcpu > 0 else len(pids)
        cpu_cores = hist.get("cpu_cores") or measured_cpu
        ram_mb = sum_rss or hist.get("ram_mb") or DEFAULT_RAM_MB
        # CPU-only group: gpu_idx is None and procs report sum_vram=0 → tag est_vram_mb=0 honestly,
        # not the GPU-default fallback (which would mis-classify the task on display).
        is_cpu_only_group = gpu_idx is None
        est_vram_for_record = 0 if is_cpu_only_group else (sum_vram or DEFAULT_VRAM_MB)
        desc_loc = f"{node}:CPU-only" if is_cpu_only_group else f"{node}:GPU{gpu_idx}"
        task = {
            "id": f"t{state['next_id']:04d}",
            "status": "running",
            "description": f"auto-adopted: {project} on {desc_loc} ({len(pids)} procs)",
            "project": project,
            # Real cmd if we could read /proc/<pid>/cmdline; placeholder otherwise. The placeholder
            # blocks auto-requeue downstream (we can't relaunch what we don't know).
            "cmd": captured_cmd or "(auto-adopted by watcher — cmdline not captured)",
            "cwd": cwd_sample,
            "signature": sig,
            "process_group": pgid,
            "est_vram_mb": est_vram_for_record,
            "ram_mb": ram_mb,
            "cpu_cores": cpu_cores,
            "priority": "normal",
            "preferred_node": None,
            "git_repo": None,
            "ckpt_dir": None,
            "ckpt_glob": "*",
            "resume_flag": "",
            "extra_env": {},
            "node": node,
            "gpu_idx": gpu_idx,
            "remote_pids": pids,
            "log_path": None,
            "submitted_at": time.time(),
            "started_at": time.time(),
            "finished_at": None,
            "peak_vram_mb": sum_vram,
            "peak_ram_mb": sum_rss,
            "resume_from": None,
            "adopted": True,
            "auto_adopted": True,
            # don't ever fire a "launched" event for these — they were already running.
            "notified_launch": True,
        }
        state["tasks"].append(task)
        state["next_id"] += 1
        adopted.append(task)
    return adopted

def _build_heartbeat_payload(state, nodes):
    running = sum(1 for t in state["tasks"] if t["status"] == "running")
    launching = sum(1 for t in state["tasks"] if t["status"] == "launching")
    queued = sum(1 for t in state["tasks"] if t["status"] == "queued")
    node_strs = []
    for n in nodes:
        if not n["alive"]:
            node_strs.append(f"{n['name']}:DOWN"); continue
        gpu_brief = "/".join(f"{g['used_mb']}MB" for g in n["gpus"])
        node_strs.append(f"{n['name']}:{gpu_brief}")
    return {"running": running, "launching": launching, "queued": queued, "nodes": node_strs}

def _smoke_test_envs():
    """One-shot probe at watcher startup: verify the python interpreters referenced by current
    queued + launching + running tasks can actually launch on their target nodes. Catches symlink rot
    (~/.conda/envs/X → removed real env), missing remote installs, and stale python paths
    BEFORE they trigger an ENV_MISSING storm. Logs warnings only — does not block startup.

    Codex review fixes:
      - skip docker tasks (host conda path doesn't matter when running in container)
      - recognize `conda run -n <env> python ...` (probe `which python` inside that env)
      - probe ONLY the resolved target node (require_node), not every node, to avoid
        false-positives where one node lacks an env that's not relevant to that task
    """
    import re as _re
    try:
        state = load_state()
    except Exception as e:
        notify("env_smoke_test_error", {"error": f"could not load state: {str(e)[:100]}"}, feishu_enabled=False)
        return
    seen = set()
    failures = []
    for t in state.get("tasks", []):
        if t.get("status") not in ("queued", "launching", "running"):
            continue
        if t.get("auto_adopted"):
            continue  # adopted tasks have synthetic cmd; nothing to probe
        # Skip docker tasks: their python lives in the container image, not on host
        spec = (t.get("env_spec") or "none").lower()
        if spec.startswith("docker"):
            continue
        cmd = t.get("cmd", "") or ""
        # Two cmd shapes to handle:
        #   (a) absolute path: /home/user/conda/envs/X/bin/python ... → probe that path directly
        #   (b) `conda run -n <env> python ...` → probe via `conda run -n <env> which python`
        probes: list[tuple[str, str]] = []  # (probe_cmd, label)
        m_abs = _re.search(r"(/[\w/.\-]+/python[\d.]*)\b", cmd)
        if m_abs:
            py = m_abs.group(1)
            probes.append((f"{shlex.quote(py)} -c 'print(\"ok\")'", py))
        m_conda = _re.search(r"\bconda\s+run\s+(?:--no-capture-output\s+)?-n\s+(\S+)", cmd)
        if m_conda:
            envname = m_conda.group(1)
            probes.append((
                f"conda run -n {shlex.quote(envname)} python -c 'print(\"ok\")' 2>&1",
                f"conda:{envname}",
            ))
        if not probes:
            continue
        # Probe target node only. For queued-without-placement tasks, prefer require_node;
        # if neither set, skip (don't blast every node — was generating false positives for
        # nodes that legitimately don't have the env).
        target = t.get("node") or t.get("require_node") or t.get("preferred_node")
        if not target:
            continue
        for probe_cmd, label in probes:
            key = (target, label)
            if key in seen:
                continue
            seen.add(key)
            try:
                rc, out, err = run_on(target, probe_cmd, timeout=15, check=False)
            except Exception as e:
                rc, out, err = 1, "", str(e)
            if rc != 0:
                failures.append({"node": target, "env": label, "err": (err or out or '?')[:200]})
    if failures:
        notify("env_smoke_test_failed", {"count": len(failures), "details": failures})
    else:
        notify("env_smoke_test_passed", {"checked": len(seen)}, feishu_enabled=False)

REBOOT_DETECT_WINDOW_S = 600  # uptime < 10 min ⇒ recently booted

def _post_reboot_triage_announce():
    """If local /proc/uptime indicates a recent reboot, emit a post_reboot_triage event with
    a count of local-pinned 'running' tasks (which are guaranteed to have stale PIDs). The
    actual requeue happens in the first _watch_iteration via update_running_tasks → diagnose
    → auto-requeue; this function only adds an explicit, audit-friendly log line."""
    try:
        with open('/proc/uptime') as f:
            uptime_s = float(f.read().split()[0])
    except Exception:
        return
    if uptime_s > REBOOT_DETECT_WINDOW_S:
        return  # not a fresh boot
    try:
        with state_lock():
            state = load_state()
        affected = [t for t in state['tasks']
                    if t.get('status') == 'running'
                    and (t.get('node') or t.get('assigned_node')) == 'local']
    except Exception:
        affected = []
    notify("post_reboot_triage", {
        "uptime_s": int(uptime_s),
        "local_running_pre_reboot": len(affected),
        "affected_ids": [t['id'] for t in affected[:20]],
        "note": ("local box rebooted; all listed tasks have stale PIDs and will be flagged "
                 "as crashed in the first dispatch cycle, then auto-requeued. Tasks with "
                 "--resume-flag set resume from latest ckpt; --allow-no-resume tasks restart "
                 "from step 0. Remote-node tasks are unaffected (setsid kept them alive).")
    }, feishu_enabled=False)

def cmd_watch(args):
    """Background daemon: every --interval s, update running tasks + dispatch queue.
    Notifications fire on task done, dispatch launches, and every --heartbeat s. No duplicates."""
    # Refuse to start a second watcher.
    if WATCHER_STATE.exists():
        try:
            existing = json.loads(WATCHER_STATE.read_text())
            if existing.get("pid") and _pid_alive(existing["pid"]):
                sys.exit(f"another watcher is already running (pid={existing['pid']}). "
                         f"Stop it first: kill {existing['pid']}")
        except Exception:
            pass  # stale state file, overwrite below

    stop_flag = {"stop": False}
    def _signal(sig, frame): stop_flag["stop"] = True
    signal.signal(signal.SIGTERM, _signal)
    signal.signal(signal.SIGINT, _signal)

    me = {"pid": os.getpid(), "started_at": time.time(), "last_heartbeat_ts": 0,
          "interval": args.interval, "heartbeat": args.heartbeat}
    WATCHER_STATE.parent.mkdir(parents=True, exist_ok=True)
    WATCHER_STATE.write_text(json.dumps(me))
    notify("watcher_started", me)
    # One-shot env health check: probe python interpreters referenced by current tasks.
    # Catches conda env corruption / removed symlinks before they trigger ENV_MISSING storms.
    try:
        _smoke_test_envs()
    except Exception as e:
        notify("env_smoke_test_error", {"error": str(e)[:200]}, feishu_enabled=False)
    # Stale launching-state recovery (item 5): if scheduler was SIGKILL'd between WAL write
    # and launch's status=running flip, queue.json has tasks stuck in 'launching'. Revert
    # them to queued so dispatch can retry. The auto-adopt machinery picks up any orphan
    # GPU procs that did get spawned. > 60s threshold filters out tasks legitimately
    # mid-launch RIGHT NOW (a normal launch's WAL window is sub-second).
    try:
        with state_lock():
            state = load_state()
            n_reverted = recover_stale_launching_tasks(state)
            if n_reverted > 0:
                save_state(state)
        if n_reverted > 0:
            notify("launching_state_recovered", {"reverted_count": n_reverted}, feishu_enabled=False)
    except Exception as _e:
        notify("launching_recovery_error", {"error": str(_e)[:200]}, feishu_enabled=False)
    # Post-reboot triage: if local box rebooted recently, all local-pinned running tasks have
    # stale PIDs guaranteed dead. The first _watch_iteration's update_running_tasks will detect
    # this anyway, but emit an explicit signal so the user sees "scheduler detected reboot,
    # processing N affected tasks" instead of having to grep watcher.log for the requeue events.
    try:
        _post_reboot_triage_announce()
    except Exception as e:
        notify("post_reboot_triage_error", {"error": str(e)[:200]}, feishu_enabled=False)

    try:
        while not stop_flag["stop"]:
            try:
                _watch_iteration(args)
            except Exception as e:
                # Never crash the loop — log and continue.
                notify("iteration_error", {"error": str(e)[:300]}, feishu_enabled=False)
            # Sleep in 1s chunks so signals break out promptly.
            for _ in range(args.interval):
                if stop_flag["stop"]: break
                time.sleep(1)
    finally:
        notify("watcher_stopped", {"pid": me["pid"]})
        try: WATCHER_STATE.unlink()
        except Exception: pass

def _watch_iteration(args):
    """One probe + dedup-aware notify cycle. Logs all events; pushes only un-notified ones to Feishu."""
    nodes = None
    recovered_launching_count = 0
    # Pre-flight: same as cmd_dispatch. First make stale launching tasks queued so the env
    # preload scan sees them, then do slow docker/conda delivery outside state_lock.
    try:
        with state_lock():
            state = load_state()
            recovered_launching_count = recover_stale_launching_tasks(state)
            if recovered_launching_count:
                save_state(state)
    except Exception as _e:
        notify("launching_recovery_error", {"error": str(_e)[:200]}, feishu_enabled=False)
    try:
        _preload_docker_images_outside_lock()
    except Exception as _e:
        notify("preload_error", {"error": str(_e)[:200]}, feishu_enabled=False)
    # Phase 3.0.5 P1: stage migration outside the main lock (see cmd_dispatch comment)
    try:
        _stage_migration_candidates_outside_lock()
    except Exception as _e:
        notify("migration_staging_error_outer", {"error": str(_e)[:200]},
               feishu_enabled=False)
    with state_lock():
        state = load_state()
        recovered_launching_count += recover_stale_launching_tasks(state)
        # 1. Detect transitions running → done. update_running_tasks marks status; we record which IDs
        #    transitioned so we only notify the freshly-done ones.
        pre_status = {t["id"]: t["status"] for t in state["tasks"]}
        update_running_tasks(state)
        # Ex-post OOM detection: scan local syslog for kernel OOM kills overlapping local
        # task termination windows. Flips affected `done` → `failed` so the requeue loop
        # below picks them up. Catches the silent-loss scenario where OOM kills a sibling
        # process and our task dies "ambiguously" with no traceback.
        oom_flipped = _detect_oom_kills_local(state)
        for t in (oom_flipped or []):
            new_id = _requeue_after_crash(t, state)
            if new_id:
                t["requeued_as"] = new_id
            notify("task_oom_requeue", {
                "id": t["id"], "requeued_as": t.get("requeued_as"),
                "lifetime_s": max(0, (t.get("finished_at") or 0) - (t.get("started_at") or 0)),
                "description": t.get("description", "")[:80],
            }, feishu_enabled=False)
        newly_done, newly_crashed = [], []
        for t in state["tasks"]:
            prev = pre_status.get(t["id"])
            if prev != "running": continue
            if t["status"] == "failed" and not t.get("notified_done"):
                newly_crashed.append(t)
                t["notified_done"] = True
            elif t["status"] == "done" and not t.get("notified_done"):
                newly_done.append(t)
                t["notified_done"] = True
        # 2. Probe nodes and run dispatch (mutates state + nodes; returns events).
        nodes = probe_all()
        # Pre-dispatch: if any GPU is over threshold from earlier dispatches, evict the youngest
        # task on it back to queue. This is the rollback companion to optimistic packing — we
        # let _reserve_inflight_vram allow stacking based on observed peak, but if the gamble
        # results in actual threshold breach, kill the latest one cleanly.
        evicted = _enforce_post_dispatch_thresholds(state, nodes)
        if evicted:
            # GPU memory will take a moment to release; re-probe so dispatch sees freed state.
            nodes = probe_all()
        _reserve_inflight_vram(state, nodes)
        events, _ = _do_dispatch(state, nodes)
        # 3. Mark dispatch launches as notified so a restart doesn't re-fire.
        for ev in events:
            if ev["type"] == "launched":
                ev["task"]["notified_launch"] = True
        # 4. Auto-adopt any GPU process not yet tracked (covers tasks launched
        #    outside the scheduler — direct ssh, other Claude conversations, etc.).
        auto_adopted = _reconcile_external_tasks(state)
        # 5. Archive terminal tasks older than ARCHIVE_AGE_DAYS so queue.json doesn't grow
        #    unbounded. Cheap when nothing to do (just a list scan of state["tasks"]).
        archived_count = archive_terminal_tasks(state)
        # 6. Batch completions — detect when ALL sibling tasks of a 2-level signature prefix
        #    reach terminal state. Fires once per batch (re-arms when a later task of same prefix
        #    transitions, so re-runs of the same project family also notify).
        batch_completions = _detect_batch_completions(state, [t["id"] for t in newly_done + newly_crashed])
        save_state(state)

    # 4. Emit notifications outside the lock so Feishu I/O doesn't hold up other invocations.
    if recovered_launching_count:
        notify("launching_state_recovered", {"reverted_count": recovered_launching_count},
               feishu_enabled=False)
    for t in newly_done:
        notify("task_done", t)
    for t in newly_crashed:
        diag = t.get("_diagnosis", {})
        notify("task_crashed", {
            "id": t["id"],
            "project": t.get("project"),
            "description": t.get("description",""),
            "node": t.get("node"),
            "gpu_idx": t.get("gpu_idx"),
            "lifetime_s": diag.get("lifetime_s", 0),
            "log_size": diag.get("log_size", 0),
            "log_path": diag.get("log_path"),
            "reason": diag.get("reason", "?"),
            "tail": diag.get("tail", ""),
        })
    for ev in events:
        if ev["type"] == "launched":
            notify("task_launched", ev["task"])
        elif ev["type"] == "blocked":
            notify("task_blocked", {"task_id": ev["task_id"], "reason": ev["reason"]})
        elif ev["type"] == "launch_failed_retry":
            notify("task_launch_retry", {
                "task_id": ev["task_id"],
                "error": ev["error"],
                "attempt": ev["task"].get("launch_fail_count"),
            })
        elif ev["type"] == "launch_failed_terminal":
            notify("task_failed", {"task_id": ev["task_id"], "error": ev["error"]})
        # no_fit / resume_found are not surfaced — too noisy for Feishu, but they're in the JSONL log.
    for t in auto_adopted:
        notify("task_auto_adopted", t)
    for batch in batch_completions:
        notify("batch_complete", batch)
    if archived_count:
        notify("archived_terminal_tasks", {"count": archived_count, "age_days": ARCHIVE_AGE_DAYS},
               feishu_enabled=False)

    # 5. Heartbeat — independent timer. Snapshot only, no event recap (those got their own notifications).
    me = json.loads(WATCHER_STATE.read_text()) if WATCHER_STATE.exists() else {}
    now = time.time()
    if now - me.get("last_heartbeat_ts", 0) >= args.heartbeat:
        with state_lock():
            state = load_state()
        payload = _build_heartbeat_payload(state, nodes)
        notify("heartbeat", payload)
        me["last_heartbeat_ts"] = now
        try: WATCHER_STATE.write_text(json.dumps(me))
        except Exception: pass

def cmd_status(args):
    with state_lock():
        state = load_state()
        recover_stale_launching_tasks(state)
        update_running_tasks(state)
        save_state(state)
    if args.json:
        print(json.dumps({"tasks": state["tasks"]}, indent=2))
        return
    print("=== nodes ===")
    node_loads = compute_node_load_seconds(state)
    for n in probe_all():
        if not n["alive"]:
            print(f"  {n['name']:11s} DOWN ({n.get('error','?')})"); continue
        gpu_parts = []
        for g in n["gpus"]:
            mem_pct = int(round(100 * g["used_mb"] / max(g["total_mb"], 1)))
            gpu_parts.append(f"GPU{g['idx']}={g['used_mb']}/{g['total_mb']}MB(mem:{mem_pct}%, util:{g['util_pct']}%)")
        gpu_str = ", ".join(gpu_parts)
        load = n.get("loadavg", 0)
        cpu_str = f"cpu={n.get('free_cpu', '?')}/{n.get('total_cpu', '?')}(load {load:.1f})"
        # Phase 3.0.2: show node ETA-load (sum of in-flight task ETAs) so the user
        # sees imbalance directly. Migration trigger (Phase 3.0.3) acts on this metric.
        etaload = node_loads.get(n["name"], 0)
        if etaload <= 0:
            etaload_str = ""
        elif etaload < 3600:
            etaload_str = f"  eta_load={etaload/60:.0f}m"
        elif etaload < 86400:
            etaload_str = f"  eta_load={etaload/3600:.1f}h"
        else:
            etaload_str = f"  eta_load={etaload/86400:.1f}d"
        print(f"  {n['name']:11s} {gpu_str}  {cpu_str}  ram_free={n['free_ram_mb']}MB{etaload_str}")
    print("\n=== tasks ===")
    show_done = args.all
    rows = [t for t in state["tasks"] if show_done or t["status"] in ("queued", "launching", "running")]
    if not rows:
        print("  (no active tasks; pass --all to see history)")
    for t in rows:
        loc = _format_task_location(t)
        peak = f"peak={t['peak_vram_mb']}MB" if t.get("peak_vram_mb") else f"~{t['est_vram_mb']}MB"
        # Mirror peak_vram column with a peak_ram one — same fallback to declared.
        pram = f"R={t['peak_ram_mb']}MB" if t.get("peak_ram_mb") else f"~{t.get('ram_mb', 0)}MB"
        runtime = ""
        if t.get("started_at"):
            end = t.get("finished_at") or time.time()
            runtime = f" {(end - t['started_at'])/60:.1f}m"
        proj = (t.get("project") or "?")[:14]
        print(f"  [{t['id']}] {t['status']:9s} {loc:20s} {proj:14s} {peak:14s} {pram:13s}{runtime}  {t['description'][:55]}")

def cmd_show(args):
    state = load_state()
    for t in state["tasks"]:
        if t["id"] == args.id:
            print(json.dumps(t, indent=2))
            if t.get("log_path"):
                print(f"\n# tail log: ssh {NODES[t['node']]['host'] or 'local'} tail -f {t['log_path']}")
            return
    sys.exit(f"task {args.id} not found")

def cmd_cancel(args):
    with state_lock():
        state = load_state()
        recover_stale_launching_tasks(state)
        for t in state["tasks"]:
            if t["id"] != args.id: continue
            if t["status"] in ("queued", "launching"):
                prev = t["status"]
                t["status"] = "cancelled"
                t["finished_at"] = time.time()
                t.pop("launching_started_at", None)
                save_state(state)
                print(f"cancelled {prev} task {args.id}")
                return
            if t["status"] == "running":
                if not args.force:
                    sys.exit(f"task {args.id} is RUNNING — pass --force to kill it (will not affect other tasks)")
                pids = _task_pids(t)
                ok, kill_msg = _kill_task_processes(t, timeout=15)
                t["status"] = "cancelled"
                t["finished_at"] = time.time()
                save_state(state)
                suffix = kill_msg if ok else f"kill warning: {kill_msg}"
                print(f"killed pids={pids} on {t['node']} and cancelled {args.id} ({suffix})")
                return
            sys.exit(f"task {args.id} is in state {t['status']!r} — nothing to do")
        sys.exit(f"task {args.id} not found")

def cmd_forget(args):
    """Drop a task record from tracking. NEVER touches processes.
    Use to undo a wrong `adopt`, or when external PIDs have died but tracking is stale."""
    with state_lock():
        state = load_state()
        for t in state["tasks"]:
            if t["id"] != args.id: continue
            prev = t["status"]
            t["status"] = "forgotten"
            t["finished_at"] = time.time()
            save_state(state)
            print(f"forgot {args.id} (was {prev}). No processes were touched.")
            return
        sys.exit(f"task {args.id} not found")

def cmd_clear_queue(args):
    with state_lock():
        state = load_state()
        ids = [t["id"] for t in state["tasks"] if t["status"] == "queued"]
        if not args.confirm:
            print(f"would cancel {len(ids)} queued tasks: {ids}")
            print("running tasks would NOT be touched. Re-run with --confirm to apply.")
            return
        for t in state["tasks"]:
            if t["status"] == "queued":
                t["status"] = "cancelled"
        save_state(state)
        print(f"cancelled {len(ids)} queued tasks (running tasks untouched)")

def cmd_rebalance_pending(args):
    """Pull all currently-pending slurm tasks back into scheduleurm's queue so they
    re-distribute under the current policy (e.g. after changing
    SLURM_MAX_PENDING_PER_NODE).

    Acts only on tasks with status='running' AND slurm_job_id set AND slurm_state in
    {None, '', PENDING, CONFIGURING, REQUEUED, SUSPENDED}. RUNNING / COMPLETING tasks
    are NEVER touched (they have allocated GPUs and may be mid-training). LocalBackend
    tasks (no slurm_job_id) are also untouched.

    For each candidate: `scancel <jid>` on the task's node (best-effort — orphan
    cleanup if it fails; orphan job times out at slurm walltime), then clear slurm
    fields and revert status='queued'. Next dispatch cycle re-places under the
    current throttle, spreading pending across slurm nodes that have free cap.

    Use case: after editing SLURM_MAX_PENDING_PER_NODE / NODES[name].max_slurm_pending
    or during a policy migration, to re-distribute already-sbatched-but-not-yet-running
    tasks. Safe to run anytime — RUNNING tasks won't be killed.
    """
    with state_lock():
        state = load_state()
        candidates = []
        for t in state["tasks"]:
            if t.get("status") != "running":
                continue
            if not _is_slurm_managed(t):
                continue
            if t.get("slurm_state") not in _SLURM_PENDING_LIKE:
                continue
            candidates.append(t)

        if not candidates:
            print("no slurm-pending tasks to rebalance")
            return

        if not args.yes:
            print(f"would rebalance {len(candidates)} task(s) (scancel + revert to queued):")
            for t in candidates[:15]:
                print(f"  {t['id']}: jid={t['slurm_job_id']} on {t['node']}  "
                      f"state={t.get('slurm_state') or 'NEW'}  sig={t.get('signature')}")
            if len(candidates) > 15:
                print(f"  ... and {len(candidates) - 15} more")
            print("RUNNING / COMPLETING tasks are NOT touched.")
            print("Re-run with --yes to proceed.")
            return

        rebalanced = 0
        scancel_failed = []
        for t in candidates:
            jid = int(t["slurm_job_id"])
            node = t["node"]
            # scancel — best effort. If it fails (ssh blip, jid already done), proceed
            # with requeue anyway. An orphan slurm job will time out at walltime (24h
            # default); not ideal but recoverable. Worst case the user runs `squeue`
            # and scancels manually.
            try:
                rc, _, err = run_on(node, f"scancel {jid}", timeout=10, check=False)
                if rc != 0:
                    scancel_failed.append((t["id"], jid, err.strip()[:100]))
            except Exception as e:
                scancel_failed.append((t["id"], jid, str(e)[:100]))
            # Reset slurm-related fields, return to queued. Don't clear ckpt_dir /
            # resume_flag / signature / cmd — those drive resume injection on next launch.
            for k in ("slurm_job_id", "slurm_state", "started_at", "finished_at",
                      "log_path", "_diagnosis", "process_group", "launching_started_at"):
                t[k] = None
            t["remote_pids"] = []
            t["alive_pids"] = []
            t["status"] = "queued"
            t["last_block_reason"] = (
                f"rebalance-pending: scancelled slurm job {jid} on {node}; "
                f"will re-dispatch under current policy"
            )
            rebalanced += 1

        save_state(state)
        print(f"rebalanced {rebalanced} task(s) — back to queued for re-dispatch")
        if scancel_failed:
            print(f"\nWARN: {len(scancel_failed)} scancel(s) failed (jobs may linger in slurm queue):")
            for tid, jid, err in scancel_failed[:5]:
                print(f"  {tid} jid={jid}: {err}")


def cmd_install_slurm(args):
    """Install slurm on one or more nodes via 3-tier fallback chain.

    Tier 1 (preferred): ssh node, run install_slurm_node.sh with --tag (script does git
            clone on the node itself + source build).
    Tier 2 (fallback): if local has slurm-src cache (or we clone it), rsync to node, then
            ssh + run script with --source-dir.
    Tier 3 (final fallback): give up. Node will continue to use LocalBackend (ssh+nohup);
            scheduleurm operates fine without slurm on that node.

    Usage:
        scheduler.py install-slurm                    # all nodes (default)
        scheduler.py install-slurm --node jtl110gpu   # single node
        scheduler.py install-slurm --tag slurm-23.11.10-1
        scheduler.py install-slurm --sudo-pass cshw2406  # for ssh+sudo on remotes

    Side effects: creates ~/.cache/scheduleurm/slurm-src/ as the local source cache.
    """
    import shutil
    SRC_TAG = args.tag or "slurm-23-11-9-1"  # SchedMD uses dashes in tag names, not dots
    SCRIPT = Path(__file__).resolve().parent / "scripts" / "install_slurm_node.sh"
    if not SCRIPT.exists():
        print(f"ERROR: install script missing at {SCRIPT}")
        return 2

    LOCAL_CACHE = Path.home() / ".cache" / "scheduleurm" / "slurm-src"

    targets = [args.node] if args.node else list(NODES.keys())

    def _ensure_local_cache():
        """Tier-2 prerequisite: clone slurm source on the local box for rsync to nodes
        that can't reach github. Idempotent — re-uses existing checkout if it matches the
        requested tag."""
        if LOCAL_CACHE.exists():
            try:
                cur_tag = subprocess.check_output(
                    ["git", "-C", str(LOCAL_CACHE), "describe", "--tags", "--exact-match"],
                    stderr=subprocess.DEVNULL, timeout=5
                ).decode().strip()
                if cur_tag == SRC_TAG:
                    print(f"  local source cache already at {SRC_TAG}: {LOCAL_CACHE}")
                    return True
                print(f"  local cache at {cur_tag}, refreshing to {SRC_TAG}")
            except Exception:
                pass
            shutil.rmtree(LOCAL_CACHE, ignore_errors=True)
        LOCAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"  cloning {SRC_TAG} from github → {LOCAL_CACHE} (one-time)")
        try:
            r = subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", SRC_TAG,
                 "https://github.com/SchedMD/slurm.git", str(LOCAL_CACHE)],
                capture_output=True, text=True, timeout=600,
            )
            if r.returncode != 0:
                print(f"  github clone FAILED: {r.stderr.strip()[:300]}")
                return False
        except Exception as e:
            print(f"  github clone exception: {e}")
            return False
        return True

    def _try_tier1_github_on_node(node, host):
        """ssh node, run script with --tag — script does git clone there."""
        print(f"  [tier 1] github clone on {node}")
        cmd = (f"bash -s -- --tag {shlex.quote(SRC_TAG)}"
               + (f" --sudo-pass=-" if args.sudo_pass else ""))
        # Stdin: optional sudo password (1 line) followed by script content
        # When --sudo-pass=- is set, the script reads stdin for pass first, then executes.
        # But 'bash -s' itself also reads from stdin. We need a different approach:
        # ssh ... 'cat > /tmp/sched-slurm.sh; bash /tmp/sched-slurm.sh ARGS' < script
        # Then sudo pass goes via env var instead. Simplest: env-var.
        env_prefix = f"SUDO_PASS={shlex.quote(args.sudo_pass)} " if args.sudo_pass else ""
        # Stage script + run, passing sudo via --sudo-pass arg literally
        sudo_arg = f"--sudo-pass {shlex.quote(args.sudo_pass)}" if args.sudo_pass else ""
        full = (
            f"cat > /tmp/sched-install-slurm.sh && chmod +x /tmp/sched-install-slurm.sh && "
            f"/tmp/sched-install-slurm.sh --tag {shlex.quote(SRC_TAG)} {sudo_arg}"
        )
        try:
            with open(SCRIPT, "rb") as f:
                script_bytes = f.read()
            proc = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
                 host, full],
                input=script_bytes, capture_output=True, timeout=2400,
            )
            stdout = proc.stdout.decode(errors="replace")
            stderr = proc.stderr.decode(errors="replace")
            print((stdout + stderr)[-2000:])
            if proc.returncode == 0:
                return ("source-installed", "")
            if proc.returncode == 2:
                return ("already-installed", "")
            if proc.returncode == 3:
                return ("source-acquisition-failed", stderr[-500:])
            return (f"failed-rc{proc.returncode}", stderr[-500:])
        except subprocess.TimeoutExpired:
            return ("timeout", "ssh timeout (40 min)")
        except Exception as e:
            return ("ssh-error", str(e)[:200])

    def _try_tier2_rsync(node, host):
        """rsync local-cache slurm src to node + ssh + run script with --source-dir."""
        print(f"  [tier 2] rsync local cache {LOCAL_CACHE} → {host}:/tmp/sched-slurm-src/")
        try:
            r = subprocess.run(
                ["rsync", "-az", "--delete",
                 str(LOCAL_CACHE) + "/",
                 f"{host}:/tmp/sched-slurm-src/"],
                capture_output=True, text=True, timeout=900,
            )
            if r.returncode != 0:
                return ("rsync-failed", r.stderr.strip()[:300])
        except subprocess.TimeoutExpired:
            return ("rsync-timeout", "rsync >15min")
        except Exception as e:
            return ("rsync-error", str(e)[:200])

        sudo_arg = f"--sudo-pass {shlex.quote(args.sudo_pass)}" if args.sudo_pass else ""
        full = (
            f"cat > /tmp/sched-install-slurm.sh && chmod +x /tmp/sched-install-slurm.sh && "
            f"/tmp/sched-install-slurm.sh --source-dir /tmp/sched-slurm-src {sudo_arg}"
        )
        try:
            with open(SCRIPT, "rb") as f:
                script_bytes = f.read()
            proc = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
                 host, full],
                input=script_bytes, capture_output=True, timeout=2400,
            )
            stdout = proc.stdout.decode(errors="replace")
            stderr = proc.stderr.decode(errors="replace")
            print((stdout + stderr)[-2000:])
            if proc.returncode == 0:
                return ("rsync-installed", "")
            if proc.returncode == 2:
                return ("already-installed", "")
            return (f"failed-rc{proc.returncode}", stderr[-500:])
        except subprocess.TimeoutExpired:
            return ("timeout", "ssh build timeout")
        except Exception as e:
            return ("ssh-error", str(e)[:200])

    def _run_local():
        """Local box: just exec the script directly (no ssh)."""
        sudo_arg = ["--sudo-pass", args.sudo_pass] if args.sudo_pass else []
        try:
            r = subprocess.run(
                [str(SCRIPT), "--tag", SRC_TAG] + sudo_arg,
                timeout=2400,
            )
            if r.returncode == 0:
                return ("source-installed", "")
            if r.returncode == 2:
                return ("already-installed", "")
            if r.returncode == 3:
                return ("github-unreachable-locally", "")
            return (f"failed-rc{r.returncode}", "")
        except subprocess.TimeoutExpired:
            return ("timeout", "local build >40min")
        except Exception as e:
            return ("error", str(e)[:200])

    results = {}
    for node in targets:
        info = NODES.get(node, {})
        host = info.get("host")
        print(f"\n========== {node} ({'local' if host is None else host}) ==========")
        if host is None:
            # Local — just run the script
            outcome, detail = _run_local()
            results[node] = outcome
            print(f"  → {outcome}")
            continue

        # Remote: tier 1 → tier 2 → fail
        outcome, detail = _try_tier1_github_on_node(node, host)
        if outcome in ("source-installed", "already-installed"):
            results[node] = outcome
            print(f"  → {outcome} (tier 1)")
            continue
        print(f"  tier 1 outcome: {outcome} — falling back")
        if not _ensure_local_cache():
            results[node] = "no-local-cache"
            print(f"  → cannot establish local cache; tier 3 = LocalBackend fallback")
            continue
        outcome2, detail2 = _try_tier2_rsync(node, host)
        if outcome2 in ("rsync-installed", "already-installed"):
            results[node] = outcome2
            print(f"  → {outcome2} (tier 2)")
            continue
        results[node] = f"failed-{outcome2}"
        print(f"  tier 2 outcome: {outcome2}; tier 3 = LocalBackend fallback (no slurm install)")

    print("\n========== summary ==========")
    for node, outcome in results.items():
        print(f"  {node}: {outcome}")

    # Reset HybridBackend cache so next dispatch re-probes (picks up new slurm if installed)
    if hasattr(_BACKEND, "_cache"):
        for n in results:
            _BACKEND._cache.pop(n, None)
        print("\n  HybridBackend slurm-detect cache cleared for these nodes; restart watcher to take effect.")


def cmd_adopt(args):
    """Register externally-launched process(es) as a tracked task. Verifies PIDs alive and on the claimed GPU."""
    pids = list(args.pids)
    if not pids:
        sys.exit("--pid requires at least one PID")
    # Probe: per-PID liveness, then compute-apps + gpu index→bus mapping for VRAM/GPU verification.
    # Zombie guard same as check_running (Codex P1)
    pid_checks = "; ".join(
        f"kill -0 {p} 2>/dev/null && "
        f"awk '/^State:/{{s=$2}} END{{if(s!=\"Z\" && s!=\"X\") print \"ALIVE_{p}\"}}' "
        f"/proc/{p}/status 2>/dev/null"
        for p in pids
    )
    probe = (f"({pid_checks}; true); echo '===PROC==='; "
             f"nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader,nounits 2>/dev/null; "
             f"echo '===GPU==='; "
             f"nvidia-smi --query-gpu=index,gpu_bus_id --format=csv,noheader,nounits 2>/dev/null")
    try:
        rc, out, err = run_on(args.node, probe, timeout=15, check=False)
    except Exception as e:
        sys.exit(f"failed to probe {args.node}: {e}")
    if rc != 0:
        sys.exit(f"probe rc={rc}: {err.strip()[:300]}")
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    proc_sep = lines.index("===PROC===") if "===PROC===" in lines else len(lines)
    gpu_sep = lines.index("===GPU===") if "===GPU===" in lines else len(lines)
    alive_pids = {int(l.split("_", 1)[1]) for l in lines[:proc_sep] if l.startswith("ALIVE_")}
    dead = set(pids) - alive_pids
    if dead:
        sys.exit(f"these PIDs are not alive on {args.node}: {sorted(dead)}")
    proc_lines = lines[proc_sep+1:gpu_sep]
    gpu_lines = lines[gpu_sep+1:]
    bus_to_idx = {}
    for gl in gpu_lines:
        parts = [x.strip() for x in gl.split(",")]
        if len(parts) >= 2:
            try: bus_to_idx[parts[1]] = int(parts[0])
            except ValueError: continue
    pid_set = set(pids)
    sum_vram = 0
    detected_gpus = set()
    for pl in proc_lines:
        parts = [x.strip() for x in pl.split(",")]
        if len(parts) < 3: continue
        try:
            ppid = int(parts[1]); mb = int(parts[2])
        except ValueError: continue
        if ppid not in pid_set: continue
        sum_vram += mb
        bus = parts[0]
        if bus in bus_to_idx:
            detected_gpus.add(bus_to_idx[bus])
    if detected_gpus and args.gpu not in detected_gpus:
        actual = sorted(detected_gpus)
        if len(actual) == 1:
            print(f"NOTE: detected your PIDs are on GPU{actual[0]} (you said GPU{args.gpu}). Using GPU{actual[0]}.")
            args.gpu = actual[0]
        else:
            print(f"WARNING: PIDs span GPUs {actual} (multi-GPU job). Tagging as GPU{args.gpu}, but VRAM tracking still sums all of them.")
    if sum_vram == 0 and not detected_gpus:
        print(f"WARNING: PIDs alive but not visible as nvidia-smi compute apps. Adopting with --gpu {args.gpu}; VRAM will be 0 until they allocate.")
    # Defensive check: read /proc/<pid>/cwd for ALL pids; if they span multiple projects, refuse.
    # (Lesson from a real bug where 11 PIDs on one GPU were 2 different projects bundled by mistake.)
    cwd_per_pid = {}
    for p in pids:
        try:
            rc, o, _ = run_on(args.node, f"readlink /proc/{p}/cwd 2>/dev/null", timeout=5, check=False)
            cwd_per_pid[p] = o.strip() if rc == 0 else ""
        except Exception:
            cwd_per_pid[p] = ""
    distinct_projects = {_project_from_path(c) for c in cwd_per_pid.values() if c}
    if len(distinct_projects) > 1 and not args.allow_multi_project:
        groups = {}
        for p, c in cwd_per_pid.items():
            groups.setdefault(_project_from_path(c) or "?", []).append(p)
        msg_lines = ["adopt refused — these PIDs belong to different projects:"]
        for proj, ps in groups.items():
            msg_lines.append(f"  {proj}: pids={ps}")
        msg_lines.append("Run a separate `adopt` per project, or pass --allow-multi-project to override.")
        sys.exit("\n".join(msg_lines))

    est = args.est_vram or sum_vram or DEFAULT_VRAM_MB
    # Derive project: --project > --cwd basename > /proc/<first_pid>/cwd basename > signature prefix
    project = (args.project
               or _project_from_path(args.cwd)
               or _project_from_pid(args.node, pids[0])
               or (args.signature.split("/", 1)[0] if "/" in args.signature else args.signature))
    hist = history_get(args.signature) or {}
    # Cores default = number of distinct PIDs (multi-worker is the common case for adopted tasks).
    cpu_cores = hist.get("cpu_cores") or len(pids)
    ram_mb = hist.get("ram_mb") or DEFAULT_RAM_MB
    with state_lock():
        state = load_state()
        task = {
            "id": f"t{state['next_id']:04d}",
            "status": "running",
            "description": args.description,
            "project": project,
            # Use the same "(auto-adopted" prefix the watcher emits, so the requeue guard at
            # _requeue_after_crash treats both manual and auto adopts uniformly (no real cmd
            # captured → no relaunch). Without the prefix, a crashed manual-adopted task would
            # try to re-launch the placeholder string as bash and fail every retry cycle.
            "cmd": "(auto-adopted manually — launched outside scheduler)",
            "cwd": args.cwd or "(unknown)",
            "signature": args.signature,
            "est_vram_mb": int(est),
            "ram_mb": ram_mb,
            "cpu_cores": cpu_cores,
            "priority": "normal",
            "preferred_node": None,
            "git_repo": None,
            "ckpt_dir": args.ckpt_dir,
            "ckpt_glob": "*",
            "resume_flag": "",
            "extra_env": {},
            "node": args.node,
            "gpu_idx": args.gpu,
            "remote_pids": pids,
            "log_path": args.log_path,
            "submitted_at": time.time(),
            "started_at": time.time(),
            "finished_at": None,
            "peak_vram_mb": sum_vram,
            "resume_from": None,
            "adopted": True,
            # Set auto_adopted=True so preemption / requeue guards (which all check this flag,
            # not `adopted`) treat manual adopts the same as watcher-discovered adopts.
            # Without this, manual-adopted tasks were eligible for high-prio eviction.
            "auto_adopted": True,
            "notified_launch": True,
        }
        state["tasks"].append(task)
        state["next_id"] += 1
        save_state(state)
    print(f"adopted {task['id']}: {len(pids)} pid(s)={pids} on {args.node}:GPU{args.gpu}  "
          f"vram_now={sum_vram}MB est={est}MB sig={args.signature}")
    print(f"  task is marked 'done' only when ALL {len(pids)} PIDs have exited.")

def cmd_record_vram(args):
    with state_lock():
        history_record(args.signature, peak_vram_mb=args.peak_vram_mb)
    after = history_get(args.signature) or {}
    print(f"recorded {args.signature} → vram={after.get('vram_mb', 0)}MB ram={after.get('ram_mb', 0)}MB cpu={after.get('cpu_cores', 0)}")

def cmd_tui(args):
    """Launch the interactive textual TUI. Defined in tui.py to keep import cost off the hot path."""
    from tui import main as tui_main
    tui_main()

def cmd_history(args):
    h = load_history()
    if not h:
        print("(no resource history yet — runs will record automatically as they finish)")
        return
    rows = []
    for sig, raw in h.items():
        if isinstance(raw, int): raw = {"vram_mb": raw}
        rows.append((sig, raw.get("vram_mb", 0), raw.get("ram_mb", 0), raw.get("cpu_cores", 0)))
    rows.sort(key=lambda r: -r[1])
    print(f"  {'signature':<40s} {'vram':>8s} {'ram':>10s} {'cpu':>5s}")
    for sig, vram, ram, cpu in rows:
        print(f"  {sig:<40s} {vram:>6}MB {ram:>8}MB {cpu:>5}")

# ---------- arg parsing ----------
def main():
    p = argparse.ArgumentParser(prog="scheduler")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit", help="Add a task to the queue")
    s.add_argument("--description", required=True)
    s.add_argument("--cmd", required=True, help="Shell command to run (will be wrapped with cd cwd && env)")
    s.add_argument("--cwd", required=True, help="Working directory on target node")
    s.add_argument("--signature", required=True, help="Stable id like 'RE-SAC/b1' for VRAM history lookup")
    s.add_argument("--vram", type=int, help="Override est VRAM in MB (else from history; else %d). Use 0 for CPU-only tasks." % DEFAULT_VRAM_MB)
    s.add_argument("--ram-mb", type=int, dest="ram_mb", help="Override est RAM in MB (else from history; else %d)" % DEFAULT_RAM_MB)
    s.add_argument("--cpu", type=int, help="CPU cores needed (else from history; else %d)" % DEFAULT_CPU_CORES)
    s.add_argument("--priority", choices=["low", "normal", "high"], default="normal")
    s.add_argument("--project", help="Project name (else derived from cwd basename)")
    s.add_argument("--preferred-node", choices=list(NODES.keys()), help="SOFT preference: try this node first, fall back if full")
    s.add_argument("--require-node", dest="require_node", choices=list(NODES.keys()), help="HARD pin: only place on this node, never fall back. Use when the cmd has node-specific paths/env that won't work elsewhere.")
    s.add_argument("--git-repo", help="Local + remote path of git repo to sync-check before launch")
    s.add_argument("--ckpt-dir", help="Checkpoint directory on TARGET node, for resume detection")
    s.add_argument("--ckpt-glob", default="*", help="Glob within ckpt-dir (default '*')")
    s.add_argument("--resume-flag", dest="resume_flag", default="",
                   help="If set (e.g. '--resume_from'), launcher appends '<flag> <ckpt_path>' to cmd "
                        "when find_resume() locates a checkpoint. Empty (default) = no injection. "
                        "Pair with --ckpt-dir; the script must accept this flag.")
    s.add_argument("--env", nargs="*", help="Extra env vars KEY=VALUE")
    s.add_argument("--allow-cpu-training", dest="allow_cpu_training", action="store_true",
                   help="Override the 'training cmd + --vram 0' refusal. Use when you really do "
                        "want to train on CPU; required even if the cmd contains --device cpu. "
                        "MUST be paired with --cpu-training-justification.")
    s.add_argument("--cpu-training-justification", dest="cpu_training_justification", default="",
                   help="Required when --allow-cpu-training is set: ≥30 chars explaining why "
                        "this training task should run on CPU rather than GPU. Stored on the "
                        "task record for audit (e.g. 'tiny MLP, GPU saturated, runs <30 min').")
    s.add_argument("--allow-no-ckpt", dest="allow_no_ckpt", action="store_true",
                   help="Override the 'training cmd without --ckpt-dir' refusal. Use for short "
                        "debug runs or one-shot evals where losing progress on relaunch is fine.")
    s.add_argument("--allow-no-resume", dest="allow_no_resume", action="store_true",
                   help="Override the 'training cmd without resume capability' refusal. Required "
                        "when the script genuinely cannot resume (no --resume_from arg in script, "
                        "or ckpt missing optimizer/buffer/RNG state). Acknowledges that any "
                        "crash/reboot/eviction will lose all progress.")
    s.add_argument("--env-spec", dest="env_spec", default="none",
                   help="Environment delivery strategy: 'none' (default; cmd assumes env on target), "
                        "'docker:IMAGE[:TAG]' (wrap launch in docker run, push image to remote on first use), "
                        "'conda:/abs/path/to/env' (rsync local conda env to target at same path before "
                        "dispatch; cmd's absolute python path resolves on the synced env), "
                        "'auto' (probe target docker access; falls back to 'none' if no docker AND no --image).")
    s.add_argument("--image", dest="image", default="",
                   help="Docker image to use when --env-spec includes docker. Required for --env-spec auto "
                        "to enable docker fallback. Image must exist locally (`docker images`) so scheduler "
                        "can save+ssh+load it to remotes.")
    s.add_argument("--allow-shared-ckpt-dir", dest="allow_shared_ckpt_dir", action="store_true",
                   help="Override the cross-signature ckpt-dir conflict refusal. Use only when "
                        "multiple tasks deliberately share a ckpt directory (e.g. distributed "
                        "training reading a shared warm-start ckpt) — concurrent writers will "
                        "still corrupt each other unless coordinated.")
    s.add_argument("--allow-duplicate", dest="allow_duplicate", action="store_true",
                   help="Allow submission even when (signature, cmd) matches an existing queued/launching/running task")
    s.add_argument("--slurm-partition", dest="slurm_partition", default="",
                   help="Optional Slurm partition to pass as #SBATCH --partition when routed to SlurmBackend")
    s.add_argument("--slurm-account", dest="slurm_account", default="",
                   help="Optional Slurm account to pass as #SBATCH --account when routed to SlurmBackend")
    s.add_argument("--slurm-qos", dest="slurm_qos", default="",
                   help="Optional Slurm QoS to pass as #SBATCH --qos when routed to SlurmBackend")
    s.set_defaults(func=cmd_submit)

    sub.add_parser("dispatch", help="Probe nodes & launch what fits (also rebalances queue)").set_defaults(func=cmd_dispatch)

    s = sub.add_parser("wait-for", help="Block until matching tasks reach terminal state; exit fires a task-notification when wrapped in Bash run_in_background. Match by --signature glob or --task-id list (or both).")
    s.add_argument("--signature", help="fnmatch glob over task signatures (e.g. 'H2Oplus/multiseed_*')")
    s.add_argument("--task-id", dest="task_ids", nargs="*", default=[], help="One or more explicit task IDs (e.g. t0099 t0100)")
    s.add_argument("--poll", type=int, default=30, help="Poll interval in seconds (default 30)")
    s.add_argument("--timeout", type=int, default=14400, help="Max seconds to wait (default 14400 = 4h; 0 = no timeout)")
    s.add_argument("--verbose", action="store_true", help="Print a progress line every 5 minutes")
    s.set_defaults(func=cmd_wait_for)

    s = sub.add_parser("watch", help="Background daemon: probe + dispatch every --interval s; notify on done/launch/heartbeat")
    s.add_argument("--interval", type=int, default=60, help="Probe + dispatch every N seconds (default 60)")
    s.add_argument("--heartbeat", type=int, default=3600, help="Push a state-snapshot heartbeat every N seconds (default 3600 = 1h)")
    s.set_defaults(func=cmd_watch)

    s = sub.add_parser("status", help="Show node + task state")
    s.add_argument("--all", action="store_true", help="Include done/failed/cancelled tasks")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("show", help="Show one task's full record + how to tail logs")
    s.add_argument("id"); s.set_defaults(func=cmd_show)

    s = sub.add_parser("cancel", help="Cancel a task (queued/launching: instant; running: needs --force)")
    s.add_argument("id"); s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_cancel)

    s = sub.add_parser("forget", help="Drop a task record from tracking (NEVER touches processes — for fixing wrong adopts)")
    s.add_argument("id"); s.set_defaults(func=cmd_forget)

    s = sub.add_parser("clear-queue", help="Cancel ALL queued tasks (running tasks untouched)")
    s.add_argument("--confirm", action="store_true")
    s.set_defaults(func=cmd_clear_queue)

    s = sub.add_parser("rebalance-pending", help="Pull slurm-PENDING tasks back into scheduleurm queue for re-dispatch (RUNNING tasks untouched)")
    s.add_argument("--yes", action="store_true", help="Apply (without --yes prints dry-run plan)")
    s.set_defaults(func=cmd_rebalance_pending)

    s = sub.add_parser("install-slurm", help="Install slurm + munge on a node from source (3-tier fallback: github → rsync local → LocalBackend)")
    s.add_argument("--node", help="Single node to install on (default: all nodes in NODES)")
    s.add_argument("--tag", help="Slurm git tag (default: slurm-23.11.10-1)")
    s.add_argument("--sudo-pass", help="Sudo password for the target node (use for non-passwordless sudo)")
    s.set_defaults(func=cmd_install_slurm)

    s = sub.add_parser("adopt", help="Register externally-launched PIDs as a tracked task")
    s.add_argument("--node", required=True, choices=list(NODES.keys()))
    s.add_argument("--gpu", required=True, type=int, help="GPU index on the node")
    s.add_argument("--pid", dest="pids", required=True, nargs="+", type=int, help="One or more PIDs (multi-worker tasks)")
    s.add_argument("--description", required=True)
    s.add_argument("--signature", required=True, help="For VRAM history; reuse same id across runs of same config")
    s.add_argument("--project", help="Project name (else read /proc/<pid>/cwd basename on the node)")
    s.add_argument("--cwd", help="(optional) working dir on the node, for documentation only")
    s.add_argument("--ckpt-dir", help="(optional) abs path to ckpt dir on the node")
    s.add_argument("--log-path", help="(optional) abs path to existing log file on the node")
    s.add_argument("--est-vram", type=int, help="Override est VRAM (else uses current sum)")
    s.add_argument("--allow-multi-project", action="store_true", help="Override the safety check that refuses adopting PIDs spanning different projects")
    s.set_defaults(func=cmd_adopt)

    s = sub.add_parser("record-vram", help="Manually record peak VRAM for a signature")
    s.add_argument("signature"); s.add_argument("peak_vram_mb", type=int)
    s.set_defaults(func=cmd_record_vram)

    sub.add_parser("history", help="Show VRAM history per signature").set_defaults(func=cmd_history)
    sub.add_parser("tui", help="Interactive TUI: sortable + filterable + auto-refresh task table").set_defaults(func=cmd_tui)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
