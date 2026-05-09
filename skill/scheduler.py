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
import hashlib
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
# Local is WSL2: 16 physical cores → 12 schedulable. RAM headroom can be a fixed
# MB value (`ram_headroom_mb`) or a fraction (`ram_headroom_frac`); fixed MB wins.
# Remote nodes are dedicated boxes — looser fractional headroom is usually enough.
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
    "local":      {"host": None,         "cpu_cores": 12, "ram_mb": 56 * 1024,  "ram_headroom_mb": 2048, "ram_headroom_frac": 0.20, "max_vram_per_task": None, "max_concurrent_running": 10},
    "jtl110gpu":  {"host": "jtl110gpu",  "cpu_cores": 12, "ram_mb": 200 * 1024, "ram_headroom_frac": 0.10, "max_vram_per_task": None, "max_concurrent_running": None, "enable_claims": True},
    # Slurm may be installed on this small node, but the default policy is still
    # scheduleurm-managed local placement so one GPU can hold multiple small jobs.
    # Explicit task Slurm fields or per-node slurm_*_backend="slurm" still opt in.
    "jtl110gpu2": {"host": "jtl110gpu2", "cpu_cores": 12, "ram_mb": 200 * 1024, "ram_headroom_frac": 0.10, "max_vram_per_task": None, "max_concurrent_running": None, "enable_claims": True, "gpu_util_saturation_pct": None},
}

STATE_DIR = Path.home() / ".claude" / "scheduler"
QUEUE_FILE = STATE_DIR / "queue.json"
VRAM_FILE = STATE_DIR / "vram_history.json"  # holds {sig: {vram, ram, cpu}} (back-compat: int = vram only)
RUNTIME_FILE = STATE_DIR / "runtime_history.json"  # exact-parameter walltime/unit-time history
LOCK_FILE = STATE_DIR / ".lock"
LOG_DIR = STATE_DIR / "logs"

VRAM_MARGIN_MB = 500       # headroom on a GPU after placing a task
RAM_HEADROOM_FRAC = 0.10   # keep 10% of node RAM unallocated as buffer for OS/other procs
ONE_THIRD_PACK_RULE = True # don't add to a GPU already past 1/3 used (RL plateau heuristic)
GPU_UTIL_SATURATION_PCT = 85  # if an occupied GPU is past this compute util, don't pack more (would just contend)
DEFAULT_VRAM_MB = 512      # est for unknown signatures (no history, no siblings).
GPU_EMPTY_USED_MB = 200    # treat driver/runtime noise below this as an empty GPU
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
# holds the rest in its own queue. GPU default stays 1 — slurm gets at most one lookahead
# GPU slot to bridge GPU swaps; the rest stay queued in scheduleurm and dispatch to
# whichever node frees up next. CPU-only tasks use a separate, higher default because
# they don't request gres=gpu and should not sit idle behind a pending GPU job.
#
# Tuning options (all need watcher restart to pick up new value):
#   1. Edit this constant in scheduler.py
#   2. Set env SCHEDULEURM_SLURM_MAX_PENDING_GPU_PER_NODE=N / ...CPU...=N
#      (legacy SCHEDULEURM_SLURM_MAX_PENDING_PER_NODE still controls the GPU default)
#   3. Per-node: NODES["nodename"]["max_slurm_pending_gpu"] = N or
#      ["max_slurm_pending_cpu"] = N. Legacy ["max_slurm_pending"] applies to both.
#
# Picking 0 means "never let scheduleurm have a pending task on any slurm node" — strict
# pull-on-demand. Risks GPU idle gaps during slurm's transitions but maximizes spread.
SLURM_MAX_PENDING_PER_NODE = int(os.environ.get("SCHEDULEURM_SLURM_MAX_PENDING_PER_NODE", "1"))
SLURM_MAX_PENDING_GPU_PER_NODE = int(os.environ.get(
    "SCHEDULEURM_SLURM_MAX_PENDING_GPU_PER_NODE",
    str(SLURM_MAX_PENDING_PER_NODE),
))
SLURM_MAX_PENDING_CPU_PER_NODE = int(os.environ.get(
    "SCHEDULEURM_SLURM_MAX_PENDING_CPU_PER_NODE",
    "6",
))

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
# Invalid-CLI-flag patterns: argparse / absl.flags / click / typer reject the cmd before any
# real work. Retrying is pointless because the cmd will keep failing identically. Caught here
# so the user gets a clear escalation instead of a wasted 3× retry → APP_BUG_CAP.
INVALID_FLAG_PATTERNS = (
    "FATAL Flags parsing error",                # absl.flags
    "Unknown command line flag",                # absl.flags
    "argparse.ArgumentError",                   # argparse error class
    "error: unrecognized arguments",            # argparse default error
    "error: argument",                          # argparse "expected one argument" / "invalid choice"
    "error: the following arguments are required",  # argparse missing required
    "Error: Got unexpected extra argument",     # click
    "Error: No such option",                    # click
    "Error: Missing option",                    # click
    "No such command",                          # click multi-command
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
def load_runtime_history(): return _load_json(RUNTIME_FILE, {})
def save_runtime_history(h): _atomic_write_json(RUNTIME_FILE, h)

# ---------- Phase 3.2.0: cross-scheduler / cross-user resource claims ----------
# Goal: stop two scheduleurm instances (different users OR different state dirs)
# from racing on the same non-slurm node and over-committing CPU/RAM/VRAM. Slurm
# solves this with its central daemon; without slurm, schedulers had no shared
# view — both saw the GPU as free, both ssh'd, both launched.
#
# Mechanism: per-node claims file at /tmp/scheduleurm/claims.json under flock,
# updated atomically by every scheduleurm via a small Python script the layer
# deploys to /tmp/scheduleurm/_claims.py on first use. claim() does a single
# ssh that:
#   1. (re-)deploys the script idempotently
#   2. flock -x -w N /tmp/scheduleurm/claims.lock python3 _claims.py <op> ...
#   3. read JSON, GC stale claims/intents, register this launch intent,
#      FIFO/backfill-check vs. older intents, capacity-check vs. fresh claims,
#      append-or-reject
# All schedulers see the same file under the same lock. Stale entries
# (expired TTL with dead PID for claims; expired intent TTL for intents)
# get GC'd in every op.
#
# Per-node opt-in (NODES["x"]["enable_claims"] = True) so single-user setups
# don't pay the ssh-flock cost. TTL is configurable per node; watcher renews
# living claims and runs gc_stale on a schedule.

CLAIMS_DIR_REMOTE = "/tmp/scheduleurm"
CLAIMS_FILE_REMOTE = CLAIMS_DIR_REMOTE + "/claims.json"
CLAIMS_LOCK_REMOTE = CLAIMS_DIR_REMOTE + "/claims.lock"
# Phase 3.4.0: per-user script path. Sticky bit on /tmp/scheduleurm prevents
# user A from overwriting user B's script. Each user maintains their own copy
# at /tmp/scheduleurm/_claims_${USER}.py — they all read/write the same shared
# claims.json + claims.lock under flock. Setup + op cmds resolve $USER on
# the remote shell so this works regardless of who's ssh'ing in.
CLAIMS_SCRIPT_REMOTE_TMPL = CLAIMS_DIR_REMOTE + "/_claims_${USER}.py"
CLAIM_LOCK_TIMEOUT_S = int(os.environ.get("SCHEDULEURM_CLAIM_LOCK_TIMEOUT_S", "30"))
CLAIM_TTL_S = int(os.environ.get("SCHEDULEURM_CLAIM_TTL_S", "3600"))
CLAIM_INTENT_TTL_S = int(os.environ.get("SCHEDULEURM_CLAIM_INTENT_TTL_S", "180"))
CLAIM_FIFO_STRICT_AFTER_S = int(os.environ.get("SCHEDULEURM_CLAIM_FIFO_STRICT_AFTER_S", "1800"))
CLAIM_LIVE_CHECK = os.environ.get("SCHEDULEURM_CLAIM_LIVE_CHECK", "1").lower() not in ("0", "false", "no", "off")

# Pure-Python script deployed to /tmp/scheduleurm/_claims.py on each node that
# enables claims. Reads/writes claims.json under the caller's flock. Operations:
#   claim       <record_json> <capacity_json>   → registers intent, then {ok}|{ok:false, conflict}
#   release     <{scheduler_id, task_id}_json>  → {ok, removed}
#   update_pid  <{scheduler_id, task_id, pid}>  → {ok, updated}
#   renew       <{scheduler_id, task_id, expires_at}> → {ok, renewed}
#   gc                                          → {ok, removed}
#   list                                        → {ok, claims, removed_stale}
_CLAIMS_REMOTE_SCRIPT = '''#!/usr/bin/env python3
"""scheduleurm remote claims daemon — deployed by _ClaimManager.

Caller wraps invocations in flock so the file mutates atomically. Designed
to work across OS users sharing the same node:
  - claims.json is opened r+w with mode 0666 (set + fchmod); writes happen
    in-place via truncate+write under flock — never via tmp+rename, because
    rename(my_tmp, other_users_file) fails in a sticky directory like /tmp.
  - alive() returns True on PermissionError (kill(pid, 0) → EPERM means
    the process exists but is owned by another user); returning False would
    let one user's GC drop another user's still-running claim and let the
    GPU get over-committed.
"""
import errno, json, math, os, subprocess, sys, time

CLAIMS_FILE = "/tmp/scheduleurm/claims.json"

# Phase 3.4.0: ensure new files are 0666 so any OS user sharing the node
# can update the same claims.json under flock.
os.umask(0)

def alive(pid):
    """True iff the PID exists on this node (regardless of owner).

    Phase 3.4.1 P0 fix: PermissionError from os.kill(pid, 0) means EPERM —
    the process exists but is owned by a different user. Returning False
    here treated other-user claims' PIDs as dead → one user's GC pass would
    drop another user's still-running claim → over-commit. Cross-OS-user
    correctness needs PermissionError → alive=True.
    """
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists, owned by another user
    except ValueError:
        return False
    except OSError as e:
        # ESRCH = no such process; anything else (EPERM / EINVAL / ...) means
        # the process exists or the kernel rejected the probe — treat as alive
        # so we never drop a live claim on a benign error.
        return e.errno != errno.ESRCH

def _open_shared_rw():
    """Open claims.json for in-place read+write under flock. Creates with
    0666 if missing; chmods to 0666 even when it existed (so a pre-3.4.0
    file with 0644 / 0600 from a prior writer becomes shareable). Returns
    fd (caller closes)."""
    fd = os.open(CLAIMS_FILE, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        os.fchmod(fd, 0o666)
    except (PermissionError, OSError):
        # Not the owner; mode stays as-is. Sticky dir lets us still truncate
        # and write through the fd we already hold open.
        pass
    return fd

def _read_fd(fd):
    """Read entire content of fd (already at offset 0 or rewind here)."""
    os.lseek(fd, 0, os.SEEK_SET)
    chunks = []
    while True:
        b = os.read(fd, 65536)
        if not b:
            break
        chunks.append(b)
    return b"".join(chunks).decode("utf-8", errors="replace")

def _write_fd(fd, text):
    """Truncate fd to 0 and write text. In-place rewrite — no rename needed."""
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    os.write(fd, text.encode("utf-8"))
    try:
        os.fsync(fd)
    except OSError:
        pass

def load_from_fd(fd):
    """Phase 3.4.7 P2: distinguish 'bootstrap empty' from 'corrupt empty'.
    Setup writes a default `{"version":1,"claims":[]}` so claims.json is
    NEVER 0 bytes during normal operation. A 0-byte file therefore means
    a writer crashed between ftruncate(0) and the subsequent write — we
    must not silently treat that as 'no claims', or the next op would
    over-commit by re-issuing every running task's resources. Same for
    JSON parse failures: surface as ParseError so the caller can return
    a transport-error result; CLAIM_ERROR routes to the regular launch-
    fail path (3.4.3) so the operator sees an actionable failure rather
    than silent data loss.
    """
    text = _read_fd(fd)
    if not text.strip():
        raise RuntimeError(
            "claims.json is empty — likely a partial write (post-crash). "
            "Manual recovery: inspect /tmp/scheduleurm/claims.json on the "
            "node, or `rm` it ONLY if no scheduleurm tasks are running."
        )
    try:
        data = json.loads(text)
    except Exception as e:
        raise RuntimeError(
            "claims.json failed to parse (%s) — likely a partial write. "
            "Manual recovery as above." % str(e)[:120])
    if not isinstance(data, dict) or "claims" not in data:
        raise RuntimeError("claims.json missing required schema")
    return data

def save_to_fd(fd, data):
    _write_fd(fd, json.dumps(data))

def is_stale(c, now):
    """Stale = TTL expired AND (no pid OR pid dead). A live pid past TTL is
    treated as orphan-but-still-using-resources; live scheduler should renew."""
    if c.get("expires_at", 0) >= now:
        return False
    pid = c.get("pid")
    if pid and alive(pid):
        return False
    return True

def gc_claims(claims, now):
    return [c for c in claims if not is_stale(c, now)]

def record_key(c):
    return (c.get("scheduler_id"), c.get("task_id"))

def gc_intents(intents, now):
    """Drop abandoned queue intents. Intents are pre-launch tickets, so a
    dead scheduler has no PID to probe; expiry alone is the recovery signal."""
    return [i for i in intents if float(i.get("expires_at", 0) or 0) >= now]

def upsert_intent(intents, payload, now):
    """Insert/refresh this task's shared queue intent while preserving its
    original intent_at timestamp. That timestamp is the cross-scheduler FIFO
    order; refreshing retries must not jump the task to the back."""
    key = record_key(payload)
    out = []
    found = None
    for i in intents:
        if record_key(i) == key:
            found = i
        else:
            out.append(i)
    intent = dict(payload)
    intent["intent_at"] = float((found or {}).get("intent_at") or now)
    intent["expires_at"] = float(payload.get("intent_expires_at") or payload.get("expires_at") or now + 180)
    intent["pid"] = None
    out.append(intent)
    out.sort(key=lambda x: (float(x.get("intent_at", 0) or 0), str(x.get("scheduler_id")), str(x.get("task_id"))))
    return out

def capacity_conflicts(payload, active_claims, capacity):
    """Return conflicts if payload cannot fit next to active_claims.

    Shared by capacity rejection and FIFO/backfill checks so the remote
    arbiter uses exactly one resource model.
    """
    used_cpu = sum(c.get("cpu_cores", 0) for c in active_claims)
    used_ram = sum(c.get("ram_mb", 0) for c in active_claims)
    per_gpu = {}
    for c in active_claims:
        g = c.get("gpu_idx")
        if g is not None:
            per_gpu[str(g)] = per_gpu.get(str(g), 0) + c.get("vram_mb", 0)
    cpu_need = payload.get("cpu_cores", 0)
    ram_need = payload.get("ram_mb", 0)
    gpu_idx = payload.get("gpu_idx")
    vram_need = payload.get("vram_mb", 0)
    cpu_cap = capacity.get("cpu_cores", 0)
    ram_cap = capacity.get("ram_mb", 0)
    gpu_caps = capacity.get("gpu_vram_mb", {}) or {}
    # Phase 3.4.4: cross-scheduler enforcement of local placement policy.
    max_per_task = capacity.get("max_vram_per_task")
    vram_margin = int(capacity.get("vram_margin_mb", 0) or 0)
    third_rule = bool(capacity.get("third_pack_rule"))
    default_vram = int(capacity.get("default_vram_mb", 0) or 0)
    conflicts = []
    if used_cpu + cpu_need > cpu_cap:
        conflicts.append("cpu: need %d + claimed %d > cap %d" % (cpu_need, used_cpu, cpu_cap))
    if used_ram + ram_need > ram_cap:
        conflicts.append("ram: need %dMB + claimed %dMB > cap %dMB" % (ram_need, used_ram, ram_cap))
    if gpu_idx is not None:
        gkey = str(gpu_idx)
        gcap = int(gpu_caps.get(gkey, 0))
        gused = per_gpu.get(gkey, 0)
        # Per-task VRAM cap (e.g. local enforces ≤ 4GB per task).
        if max_per_task is not None and vram_need > int(max_per_task):
            conflicts.append("gpu%s: need %dMB > per-task cap %dMB" % (
                gkey, vram_need, int(max_per_task)))
        # Total cap (raw VRAM math).
        if gused + vram_need > gcap:
            conflicts.append("gpu%s: need %dMB + claimed %dMB > cap %dMB" % (
                gkey, vram_need, gused, gcap))
        # VRAM margin: leave at least margin MB free after this claim.
        if gcap > 0 and (gcap - gused - vram_need) < vram_margin:
            conflicts.append("gpu%s: post-claim free %dMB < margin %dMB" % (
                gkey, gcap - gused - vram_need, vram_margin))
        # 1/3 packing rule. Phase 3.4.6 P1 fix: match local _gpu_fits
        # semantics — block ONLY when the EXISTING claimed VRAM is past
        # 1/3, not when (existing + this claim) would land past 1/3.
        # The rule is "don't STACK new tasks onto a card already past 1/3",
        # not "no single task may exceed 1/3 of the card".
        if third_rule and gcap > 0:
            third = gcap // 3
            if gused >= third and gused > 100:
                # Small-task exemption: allow stacking when this task is
                # ≤ default size (matches _gpu_fits's small_task branch).
                # Util-saturation half of that exemption is enforced by
                # local pick_placement before claim is even invoked.
                if vram_need > default_vram:
                    conflicts.append(
                        "gpu%s: existing claimed %dMB ≥ 1/3 of %dMB "
                        "(packing rule)" % (gkey, gused, gcap))
    return conflicts

def _read_live_snapshot(capacity):
    """Best-effort live resource view while the claims lock is held.

    Tests may pass capacity["live_snapshot"]; production reads /proc and
    nvidia-smi on the target node. Failure means "no extra external usage"
    rather than a claim-script crash.
    """
    snap = capacity.get("live_snapshot")
    if isinstance(snap, dict):
        return snap
    out = {"loadavg": None, "mem_available_mb": None, "gpu_used_mb": {}}
    try:
        with open("/proc/loadavg") as f:
            out["loadavg"] = float((f.read().split() or ["0"])[0])
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    bits = line.split()
                    if len(bits) >= 2:
                        out["mem_available_mb"] = int(int(bits[1]) / 1024)
                    break
    except Exception:
        pass
    try:
        raw = subprocess.check_output(
            "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null",
            shell=True, timeout=float(capacity.get("live_check_timeout_s", 3) or 3),
        ).decode("utf-8", "ignore")
        used = {}
        for line in raw.splitlines():
            bits = [b.strip() for b in line.split(",")]
            if len(bits) < 2:
                continue
            try:
                used[str(int(bits[0]))] = int(bits[1])
            except Exception:
                continue
        out["gpu_used_mb"] = used
    except Exception:
        pass
    return out

def live_external_claims(active_claims, capacity):
    """Represent non-scheduleurm/manual live usage as synthetic claims."""
    if not capacity.get("live_check"):
        return []
    snap = _read_live_snapshot(capacity)
    externals = []
    running_claims = [c for c in active_claims if c.get("pid")]
    claimed_cpu = sum(int(c.get("cpu_cores") or 0) for c in running_claims)
    claimed_ram = sum(int(c.get("ram_mb") or 0) for c in running_claims)
    claimed_gpu = {}
    for c in running_claims:
        g = c.get("gpu_idx")
        if g is not None:
            claimed_gpu[str(g)] = claimed_gpu.get(str(g), 0) + int(c.get("vram_mb") or 0)

    ext_cpu = 0
    try:
        if snap.get("loadavg") is not None:
            ext_cpu = max(0, int(math.ceil(float(snap.get("loadavg")))) - claimed_cpu)
    except Exception:
        pass
    ext_ram = 0
    try:
        avail = snap.get("mem_available_mb")
        cap_ram = int(capacity.get("ram_mb", 0) or 0)
        if avail is not None and cap_ram > 0:
            live_used = max(0, cap_ram - int(avail))
            ext_ram = max(0, live_used - claimed_ram)
    except Exception:
        pass
    if ext_cpu or ext_ram:
        externals.append({
            "scheduler_id": "__external__", "task_id": "__cpu_ram_live__",
            "gpu_idx": None, "vram_mb": 0,
            "cpu_cores": ext_cpu, "ram_mb": ext_ram,
            "pid": -1,
        })

    gpu_used = snap.get("gpu_used_mb") or {}
    if isinstance(gpu_used, dict):
        for g, used in gpu_used.items():
            try:
                ext_vram = max(0, int(used) - int(claimed_gpu.get(str(g), 0)))
            except Exception:
                continue
            if ext_vram <= 0:
                continue
            externals.append({
                "scheduler_id": "__external__", "task_id": "__gpu%s_live__" % str(g),
                "gpu_idx": int(g), "vram_mb": ext_vram,
                "cpu_cores": 0, "ram_mb": 0,
                "pid": -1,
            })
    return externals

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "missing op"}))
        return
    op = sys.argv[1]
    payload = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    capacity = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    os.makedirs("/tmp/scheduleurm", exist_ok=True)
    fd = _open_shared_rw()
    try:
        data = load_from_fd(fd)
    except RuntimeError as e:
        # Phase 3.4.7 P2: parse / empty failure means a writer crashed
        # mid-truncate. Don't silently treat as no claims — return an
        # error so caller's claim() routes to CLAIM_ERROR (3.4.3) and
        # the operator sees the corrupted-state message.
        try:
            os.close(fd)
        except OSError:
            pass
        print(json.dumps({"ok": False,
                            "error": "claims_corrupt: " + str(e)}))
        return
    claims = data.get("claims", [])
    intents = data.get("intents", [])
    now = time.time()
    fresh = gc_claims(claims, now)
    fresh_intents = gc_intents(intents, now)
    if op == "claim":
        fresh_intents = upsert_intent(fresh_intents, payload, now)
        key = record_key(payload)
        external = live_external_claims(fresh, capacity)
        active = fresh + external
        strict_after = float(capacity.get("fifo_strict_after_s", 0) or 0)
        # FIFO-with-backfill: older intents get priority only when the
        # current claim would delay them. If an older task cannot fit now,
        # or if both older and current fit together, this task may backfill.
        for older in fresh_intents:
            if record_key(older) == key:
                break
            if strict_after > 0:
                try:
                    older_wait = now - float(older.get("intent_at", now) or now)
                except Exception:
                    older_wait = 0
                older_can_ever_fit = not capacity_conflicts(older, [], capacity)
                older_fits_empty_after_current = not capacity_conflicts(older, [payload], capacity)
                if older_wait >= strict_after and older_can_ever_fit and not older_fits_empty_after_current:
                    data["claims"] = fresh
                    data["intents"] = fresh_intents
                    save_to_fd(fd, data)
                    print(json.dumps({
                        "ok": False,
                        "conflict": "fifo-strict: older intent %s/%s waited %.0fs" % (
                            older.get("scheduler_id"), older.get("task_id"), older_wait),
                        "claims_seen": len(fresh),
                        "intents_seen": len(fresh_intents),
                        "external_claims_seen": len(external),
                    }))
                    return
            older_fits_now = not capacity_conflicts(older, active, capacity)
            if not older_fits_now:
                continue
            older_fits_after_current = not capacity_conflicts(older, active + [payload], capacity)
            if not older_fits_after_current:
                data["claims"] = fresh
                data["intents"] = fresh_intents
                save_to_fd(fd, data)
                print(json.dumps({
                    "ok": False,
                    "conflict": "fifo: older intent %s/%s has priority" % (
                        older.get("scheduler_id"), older.get("task_id")),
                    "claims_seen": len(fresh),
                    "intents_seen": len(fresh_intents),
                    "external_claims_seen": len(external),
                }))
                return
        conflicts = capacity_conflicts(payload, active, capacity)
        if conflicts:
            data["claims"] = fresh
            data["intents"] = fresh_intents
            save_to_fd(fd, data)
            print(json.dumps({"ok": False, "conflict": "; ".join(conflicts),
                              "claims_seen": len(fresh),
                              "external_claims_seen": len(external)}))
            return
        fresh.append(payload)
        data["claims"] = fresh
        data["intents"] = [i for i in fresh_intents if record_key(i) != key]
        save_to_fd(fd, data)
        print(json.dumps({"ok": True}))
    elif op == "release":
        sid = payload.get("scheduler_id")
        tid = payload.get("task_id")
        kept = [c for c in fresh if not (c.get("scheduler_id") == sid and c.get("task_id") == tid)]
        kept_intents = [i for i in fresh_intents if not (i.get("scheduler_id") == sid and i.get("task_id") == tid)]
        data["claims"] = kept
        data["intents"] = kept_intents
        save_to_fd(fd, data)
        print(json.dumps({"ok": True, "removed": len(fresh) - len(kept),
                          "removed_intents": len(fresh_intents) - len(kept_intents)}))
    elif op == "update_pid":
        sid = payload.get("scheduler_id")
        tid = payload.get("task_id")
        pid = payload.get("pid")
        updated = 0
        for c in fresh:
            if c.get("scheduler_id") == sid and c.get("task_id") == tid:
                c["pid"] = int(pid) if pid else None
                updated += 1
        data["claims"] = fresh
        data["intents"] = fresh_intents
        save_to_fd(fd, data)
        print(json.dumps({"ok": True, "updated": updated}))
    elif op == "renew":
        sid = payload.get("scheduler_id")
        tid = payload.get("task_id")
        new_exp = payload.get("expires_at")
        renewed = 0
        for c in fresh:
            if c.get("scheduler_id") == sid and c.get("task_id") == tid:
                if new_exp:
                    c["expires_at"] = float(new_exp)
                renewed += 1
        data["claims"] = fresh
        data["intents"] = fresh_intents
        save_to_fd(fd, data)
        print(json.dumps({"ok": True, "renewed": renewed}))
    elif op == "renew_many":
        # Bulk renew for one scheduler. Watcher sends our running task ids
        # every cycle; the same call also implicitly GC's stale entries
        # because gc_claims already ran above.
        sid = payload.get("scheduler_id")
        tids = set(payload.get("task_ids") or [])
        new_exp = payload.get("expires_at")
        renewed = 0
        for c in fresh:
            if c.get("scheduler_id") == sid and c.get("task_id") in tids:
                if new_exp:
                    c["expires_at"] = float(new_exp)
                renewed += 1
        data["claims"] = fresh
        data["intents"] = fresh_intents
        save_to_fd(fd, data)
        print(json.dumps({"ok": True, "renewed": renewed,
                          "removed_stale": len(claims) - len(fresh),
                          "removed_stale_intents": len(intents) - len(fresh_intents)}))
    elif op == "gc":
        data["claims"] = fresh
        data["intents"] = fresh_intents
        save_to_fd(fd, data)
        print(json.dumps({"ok": True, "removed": len(claims) - len(fresh),
                          "removed_intents": len(intents) - len(fresh_intents)}))
    elif op == "list":
        if len(fresh) != len(claims) or len(fresh_intents) != len(intents):
            data["claims"] = fresh
            data["intents"] = fresh_intents
            save_to_fd(fd, data)
        print(json.dumps({"ok": True, "claims": fresh, "intents": fresh_intents,
                          "removed_stale": len(claims) - len(fresh),
                          "removed_stale_intents": len(intents) - len(fresh_intents)}))
    else:
        print(json.dumps({"ok": False, "error": "unknown op " + repr(op)}))
    try:
        os.close(fd)
    except OSError:
        pass

main()
'''


def _claims_setup_cmd():
    """Shell snippet that ensures the remote claims script + lock + claims
    file are usable by the calling OS user. Each user gets their own
    `_claims_${USER}.py` (sticky /tmp/scheduleurm prevents cross-user
    overwrite). claims.json + claims.lock stay shared, mode 0666 so any
    user can read+write them under flock.

    Phase 3.4.7 P2: claims.json is initialized to a non-empty default
    `{"version":1,"claims":[]}` if absent / 0 bytes, so the script can
    treat any 0-byte read as "writer crashed mid-truncate, return error"
    (instead of silently treating it as 'no claims').

    Phase 3.4.8 P3: per-user script is deployed via atomic tmp+rename
    within the user's own sticky-dir slot. Two concurrent same-user
    `claim` ops would otherwise both `cat >` the same path simultaneously
    — second writer truncates while first is still mid-write, leaving
    a partial python file that `python3` errors on. Atomic mv prevents
    half-deployed scripts.

    Quoted heredoc body means script content goes through verbatim —
    no shell expansion in the Python source.
    """
    dir_q = shlex.quote(CLAIMS_DIR_REMOTE)
    lock_q = shlex.quote(CLAIMS_LOCK_REMOTE)
    file_q = shlex.quote(CLAIMS_FILE_REMOTE)
    # Atomic per-user script deploy: write to a $$-suffixed tmp, then mv into
    # place. Same user owns both, same dir → rename succeeds despite sticky.
    return (
        f"umask 0; "
        f"mkdir -p {dir_q} && chmod 1777 {dir_q} 2>/dev/null; "
        # Lock file shared 0666 so any user can flock it.
        f"touch {lock_q} 2>/dev/null && chmod 0666 {lock_q} 2>/dev/null; "
        # Claims file: bootstrap ONLY when missing (Phase 3.4.9 P1 fix).
        # A 0-byte file means a writer crashed between ftruncate(0) and
        # the subsequent write — it must NOT be silently re-initialized
        # to empty (that would over-commit by re-issuing every running
        # task's resources on the next dispatch). Letting the 0-byte
        # state propagate makes load_from_fd raise → main() returns
        # claims_corrupt → operator sees an actionable error.
        f"if [ ! -e {file_q} ]; then "
        f"  printf '%s' '{{\"version\":1,\"claims\":[]}}' > {file_q} 2>/dev/null; "
        f"fi; "
        f"chmod 0666 {file_q} 2>/dev/null; "
        # Per-user script via atomic tmp+rename.
        f"SCRIPT_PATH={CLAIMS_DIR_REMOTE}/_claims_${{USER:-anon}}.py; "
        f"TMP_PATH=\"${{SCRIPT_PATH}}.tmp.$$\"; "
        f"cat > \"$TMP_PATH\" <<'PYEOF'\n"
        f"{_CLAIMS_REMOTE_SCRIPT}\n"
        f"PYEOF\n"
        f"chmod 0666 \"$TMP_PATH\" 2>/dev/null && "
        f"mv \"$TMP_PATH\" \"$SCRIPT_PATH\""
    )


def _claims_remote_op(node, op, payload, capacity=None, timeout_s=30):
    """Run a claims op on `node`. Returns parsed JSON result dict, or
    {"ok": False, "error": "..."} on transport / parse failure. Idempotently
    deploys the remote script before each call (ssh round-trip dominates;
    heredoc bytes are negligible)."""
    payload_arg = shlex.quote(json.dumps(payload))
    capacity_arg = shlex.quote(json.dumps(capacity or {}))
    # Per-user script path (Phase 3.4.0). $USER expanded by the remote shell.
    op_cmd = (
        f"SCRIPT_PATH={CLAIMS_DIR_REMOTE}/_claims_${{USER:-anon}}.py; "
        f"flock -x -w {CLAIM_LOCK_TIMEOUT_S} {shlex.quote(CLAIMS_LOCK_REMOTE)} "
        f"python3 \"$SCRIPT_PATH\" {shlex.quote(op)} "
        f"{payload_arg} {capacity_arg}"
    )
    full = _claims_setup_cmd() + "\n" + op_cmd
    try:
        rc, out, err = run_on(node, full, timeout=timeout_s, check=False)
    except Exception as e:
        return {"ok": False, "error": f"ssh exception: {str(e)[:200]}"}
    if rc != 0:
        return {"ok": False, "error": f"rc={rc}: {(err or '').strip()[:200]}"}
    out = (out or "").strip()
    if not out:
        return {"ok": False, "error": "empty output"}
    try:
        # Last line is the JSON result (heredoc setup may have emitted lines)
        return json.loads(out.splitlines()[-1])
    except Exception as e:
        return {"ok": False, "error": f"parse error: {e}; out={out[:200]!r}"}


class _ClaimManager:
    """Phase 3.2.0: cross-scheduler resource claims via remote flock + JSON.

    Per-node opt-in. When NODES[name]["enable_claims"] is True, every
    LocalBackend launch on that node atomically claims its CPU/RAM/VRAM
    against ALL schedulers' claims; conflicts cause the launch to retry on
    the next dispatch cycle. release/update_pid/renew/gc_stale tend the
    claim across the task's lifecycle. Watcher periodically gc_stale's so
    crashed schedulers don't pin resources beyond their TTL.
    """

    @classmethod
    def enabled_for(cls, node: str) -> bool:
        return bool(NODES.get(node, {}).get("enable_claims"))

    @classmethod
    def scheduler_id(cls) -> str:
        """Phase 3.4.2 P1 fix: PERSISTENT id keyed to the state dir, not
        the running PID. Pre-fix this returned `<host>:<pid>`, so a
        scheduler restart (watcher service restart, manual `dispatch`
        from a fresh shell) generated a new id and could no longer
        match its own pre-restart claims for release()/renew_many() —
        those would sit until TTL expired even though the same
        scheduler is alive and running.

        Stored at STATE_DIR/claim_owner_id (random UUID once, then
        cached in-process). Each scheduleurm install (different
        STATE_DIR / different state file) gets its own id; same
        install across restarts reuses the same id."""
        cached = getattr(cls, "_cached_owner_id", None)
        if cached:
            return cached
        owner_file = STATE_DIR / "claim_owner_id"
        try:
            if owner_file.exists():
                v = owner_file.read_text().strip()
                if v:
                    cls._cached_owner_id = v
                    return v
        except Exception:
            pass
        # Generate a fresh persistent id. Prefix with hostname for
        # human-readable forensics, append a random hex suffix for
        # cross-install uniqueness on the same machine.
        try:
            host = os.uname().nodename
        except Exception:
            host = "unknown"
        try:
            import uuid as _uuid
            new_id = f"{host}:{_uuid.uuid4().hex[:12]}"
        except Exception:
            # Last-resort fallback; better than nothing if uuid is missing.
            new_id = f"{host}:{os.getpid()}-{int(time.time())}"
        try:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            owner_file.write_text(new_id)
        except Exception:
            # Persistence failed — id will not survive restart, but the
            # claims layer still works for the lifetime of this process.
            pass
        cls._cached_owner_id = new_id
        return new_id

    @classmethod
    def _ttl_for(cls, node: str) -> int:
        return int(NODES.get(node, {}).get("claim_ttl_s", CLAIM_TTL_S))

    @classmethod
    def _intent_ttl_for(cls, node: str) -> int:
        return int(NODES.get(node, {}).get("claim_intent_ttl_s", CLAIM_INTENT_TTL_S))

    @classmethod
    def _fifo_strict_after_for(cls, node: str) -> int:
        return int(NODES.get(node, {}).get("claim_fifo_strict_after_s", CLAIM_FIFO_STRICT_AFTER_S))

    @classmethod
    def _live_check_for(cls, node: str) -> bool:
        return bool(NODES.get(node, {}).get("claim_live_check", CLAIM_LIVE_CHECK))

    @classmethod
    def _build_capacity(cls, node: str, node_state: Optional[dict] = None) -> dict:
        """Build the capacity payload sent with every claim op.

        Phase 3.4.4 P2 fix: payload now carries the local placement policy
        (per-task VRAM cap, VRAM margin, 1/3 packing rule, small-task
        exemption threshold). Without these, two schedulers with stale
        probes could BOTH pass `_gpu_fits` locally on a fresh GPU and BOTH
        succeed at claim() — then the second-running task would violate
        the 1/3 rule even though the first scheduler's claim was meant to
        prevent that. Util saturation isn't replicated (no shared util
        reading); local pick_placement still gates on it before claim is
        ever invoked.
        """
        info = NODES.get(node, {})
        cap = {
            "cpu_cores": int(info.get("cpu_cores", 0)),
            "ram_mb": int(info.get("ram_mb", 0)),
            "gpu_vram_mb": {},
            # Policy fields for cross-scheduler enforcement.
            "max_vram_per_task": info.get("max_vram_per_task"),
            "vram_margin_mb": int(VRAM_MARGIN_MB),
            "third_pack_rule": bool(ONE_THIRD_PACK_RULE),
            "default_vram_mb": int(DEFAULT_VRAM_MB),
            "fifo_strict_after_s": cls._fifo_strict_after_for(node),
            "live_check": cls._live_check_for(node),
            "live_check_timeout_s": int(info.get("claim_live_check_timeout_s", 3)),
        }
        if node_state and node_state.get("gpus"):
            for g in node_state["gpus"]:
                cap["gpu_vram_mb"][str(g["idx"])] = int(g["total_mb"])
        return cap

    @classmethod
    def claim(cls, node: str, task: dict, gpu_idx: Optional[int],
              node_state: Optional[dict] = None) -> tuple:
        """Atomically claim resources for `task` on `node`.

        Phase 3.4.3 P1 fix: distinguishes a capacity CONFLICT (legitimate
        contention, retry) from a transport / setup ERROR (ssh failure,
        flock failure, missing python3, parse error — these need to count
        as launch failures so MAX_LAUNCH_RETRY → APP_BUG_CAP escalation
        eventually fires; otherwise the task loops in queue forever).

        Returns a 3-tuple (ok, info, kind):
          (True,  claim_record, "ok")        — claim succeeded
          (False, conflict_msg, "conflict")  — capacity rejection (retry)
          (False, error_msg,    "error")     — transport/setup failure
        Disabled-node fast path returns (True, {...}, "ok").
        """
        if not cls.enabled_for(node):
            return (True, {"reason": "claims disabled for this node"}, "ok")
        now = time.time()
        ttl = cls._ttl_for(node)
        intent_ttl = cls._intent_ttl_for(node)
        try:
            owner = os.environ.get("USER") or os.getlogin() or "?"
        except Exception:
            owner = "?"
        record = {
            "owner": owner,
            "scheduler_id": cls.scheduler_id(),
            "task_id": task["id"],
            "gpu_idx": gpu_idx,
            "vram_mb": int(task.get("est_vram_mb") or 0),
            "cpu_cores": int(task.get("cpu_cores") or DEFAULT_CPU_CORES),
            "ram_mb": int(task.get("ram_mb") or DEFAULT_RAM_MB),
            "claimed_at": now,
            "expires_at": now + ttl,
            "intent_expires_at": now + intent_ttl,
            "pid": None,
        }
        capacity = cls._build_capacity(node, node_state)
        result = _claims_remote_op(node, "claim", record, capacity)
        if result.get("ok"):
            return (True, record, "ok")
        # The remote script puts capacity rejections in `conflict` and
        # leaves `error` empty. Transport / setup failures (rc != 0, ssh
        # exception, parse error) come back via _claims_remote_op with
        # `error` set and `conflict` absent. Distinguishing them is the
        # whole point of this fix.
        if result.get("conflict"):
            return (False, result["conflict"], "conflict")
        return (False, result.get("error") or "claim transport failed", "error")

    @classmethod
    def release(cls, node: str, task_id: str) -> bool:
        if not cls.enabled_for(node):
            return True
        result = _claims_remote_op(node, "release", {
            "scheduler_id": cls.scheduler_id(),
            "task_id": task_id,
        })
        return bool(result.get("ok"))

    @classmethod
    def update_pid(cls, node: str, task_id: str, pid: Optional[int]) -> bool:
        if not cls.enabled_for(node):
            return True
        result = _claims_remote_op(node, "update_pid", {
            "scheduler_id": cls.scheduler_id(),
            "task_id": task_id,
            "pid": int(pid) if pid else None,
        })
        return bool(result.get("ok"))

    @classmethod
    def renew(cls, node: str, task_id: str) -> bool:
        if not cls.enabled_for(node):
            return True
        new_exp = time.time() + cls._ttl_for(node)
        result = _claims_remote_op(node, "renew", {
            "scheduler_id": cls.scheduler_id(),
            "task_id": task_id,
            "expires_at": new_exp,
        })
        return bool(result.get("ok"))

    @classmethod
    def renew_many(cls, node: str, task_ids: list) -> int:
        """Bulk renew all our claims on `node` matching any of `task_ids`.
        Returns count renewed; -1 on error; 0 when claims disabled. Used by
        the watcher every cycle so live claims don't expire."""
        if not cls.enabled_for(node):
            return 0
        if not task_ids:
            return 0
        new_exp = time.time() + cls._ttl_for(node)
        result = _claims_remote_op(node, "renew_many", {
            "scheduler_id": cls.scheduler_id(),
            "task_ids": list(task_ids),
            "expires_at": new_exp,
        })
        if result.get("ok"):
            return int(result.get("renewed", 0))
        return -1

    @classmethod
    def gc_stale(cls, node: str) -> int:
        """Returns count of removed claims; -1 on error; 0 if disabled."""
        if not cls.enabled_for(node):
            return 0
        result = _claims_remote_op(node, "gc", {})
        if result.get("ok"):
            return int(result.get("removed", 0))
        return -1

    @classmethod
    def enumerate(cls, node: str) -> list:
        """Returns list of claim dicts on `node`; [] when disabled or on error."""
        return list(cls.snapshot(node).get("claims") or [])

    @classmethod
    def enumerate_intents(cls, node: str) -> list:
        """Returns list of FIFO intent dicts on `node`; [] when disabled/error."""
        return list(cls.snapshot(node).get("intents") or [])

    @classmethod
    def snapshot(cls, node: str) -> dict:
        """Returns {ok, claims, intents, ...} for node; never raises."""
        if not cls.enabled_for(node):
            return {"ok": True, "claims": [], "intents": [], "disabled": True}
        try:
            result = _claims_remote_op(node, "list", {})
        except Exception as e:
            return {"ok": False, "claims": [], "intents": [], "error": str(e)[:200]}
        if result.get("ok"):
            result["claims"] = list(result.get("claims", []))
            result["intents"] = list(result.get("intents", []))
            return result
        return {"ok": False, "claims": [], "intents": [],
                "error": result.get("error") or result.get("conflict") or "claims list failed"}


def _claim_intent_nodes(task: dict) -> list:
    nodes = []
    for key in ("claim_intent_node",):
        n = task.get(key)
        if n and n not in nodes:
            nodes.append(n)
    raw = task.get("claim_intent_nodes") or []
    if isinstance(raw, str):
        raw = [raw]
    if isinstance(raw, list):
        for n in raw:
            if n and n not in nodes:
                nodes.append(n)
    return nodes


def _remember_claim_intent(task: dict, node: Optional[str]) -> None:
    if not node or not _ClaimManager.enabled_for(node):
        return
    nodes = _claim_intent_nodes(task)
    if node not in nodes:
        nodes.append(node)
    task["claim_intent_nodes"] = nodes
    task["claim_intent_at"] = time.time()


def _clear_claim_intent_markers(task: dict) -> None:
    task.pop("claim_intent_node", None)
    task.pop("claim_intent_nodes", None)
    task.pop("claim_intent_at", None)


def _release_task_claims_and_intents(task: dict, extra_nodes=None,
                                     exclude_nodes=None, clear_markers: bool = True) -> int:
    """Best-effort release of this scheduler's claim and pending FIFO intents.

    release() removes both a real claim and a pre-launch intent on the remote
    claims file. `claim_intent_nodes` exists because CLAIM_RACE clears
    task["node"] before the task returns to queued.
    """
    exclude = set(exclude_nodes or [])
    nodes = []
    if task.get("node"):
        nodes.append(task.get("node"))
    nodes.extend(_claim_intent_nodes(task))
    if extra_nodes:
        nodes.extend(extra_nodes if isinstance(extra_nodes, (list, tuple, set)) else [extra_nodes])
    seen = set()
    released = 0
    released_nodes = set()
    for node in nodes:
        if not node or node in seen or node in exclude:
            continue
        seen.add(node)
        if not _ClaimManager.enabled_for(node):
            continue
        try:
            if _ClaimManager.release(node, task["id"]):
                released += 1
                released_nodes.add(node)
        except Exception:
            pass
    if clear_markers:
        _clear_claim_intent_markers(task)
    elif released_nodes:
        kept = [n for n in _claim_intent_nodes(task) if n not in released_nodes]
        if kept:
            task["claim_intent_nodes"] = kept
        else:
            _clear_claim_intent_markers(task)
    return released


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
RUNTIME_HISTORY_MAX_ENTRIES = 1000
RUNTIME_HISTORY_SAMPLES_PER_KEY = 10
RUNTIME_HISTORY_PERCENTILE = 80
RUNTIME_WALLTIME_MULT = 1.20
RUNTIME_MIN_WALLTIME_S = 10 * 60  # progress-derived walltime can be shorter than legacy 1h

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
def _probe_windows_host_extras(timeout_s: float = 4.0) -> dict:
    """Phase 3.3: query Windows host metrics via WSL → PowerShell interop.

    The WSL2 `local` node sees only its OWN memory pool (the VM's, capped
    by .wslconfig — typically 30GB on a 64GB host) and only NVML's
    "any-kernel-active" GPU utilization (averaged across the sample window,
    much lower than Task Manager's per-engine Compute utilization for
    bursty RL workloads). Users comparing TUI vs. Task Manager see large
    apparent discrepancies even though both numbers are "correct" for
    their respective measurement model.

    This helper queries Windows-side ground truth so the TUI can show it
    alongside the WSL view:
      host_free_ram_mb         — Win32_OperatingSystem.FreePhysicalMemory
      host_total_ram_mb        — Win32_OperatingSystem.TotalVisibleMemorySize
      gpu_compute_util_pct     — max DXGI Compute engine utilization
                                 (the metric Task Manager's GPU widget uses)

    Best-effort: any failure (powershell.exe missing, interop slow, query
    error) returns an empty dict so callers can fall back gracefully.
    Total worst-case latency: ~timeout_s; in practice 200-800ms.
    """
    out = {}
    pwsh = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    if not Path(pwsh).exists():
        # 22H2+ paths can also live here; otherwise abort silently.
        alt = "/mnt/c/WINDOWS/system32/WindowsPowerShell/v1.0/powershell.exe"
        if Path(alt).exists():
            pwsh = alt
        else:
            return out
    # Combine all queries into one PowerShell invocation to amortize the
    # WSL→Windows interop startup cost (~200ms each spawn).
    script = (
        "$os = Get-CimInstance Win32_OperatingSystem; "
        "$free = [math]::Round($os.FreePhysicalMemory / 1024); "  # KB → MB
        "$tot  = [math]::Round($os.TotalVisibleMemorySize / 1024); "
        "$gpu = (Get-Counter '\\GPU Engine(*engtype_Compute*)\\Utilization "
        "Percentage' -ErrorAction SilentlyContinue).CounterSamples "
        "| Measure-Object -Maximum CookedValue; "
        "$gpu_pct = if ($gpu.Maximum) { [math]::Round($gpu.Maximum) } else { 0 }; "
        "Write-Output \"$free|$tot|$gpu_pct\""
    )
    try:
        r = subprocess.run(
            [pwsh, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=timeout_s,
        )
        if r.returncode != 0:
            return out
        line = (r.stdout or "").strip().splitlines()[-1] if r.stdout else ""
        bits = line.split("|")
        if len(bits) >= 3:
            try:
                out["host_free_ram_mb"] = int(bits[0])
                out["host_total_ram_mb"] = int(bits[1])
                out["gpu_compute_util_pct"] = int(bits[2])
            except ValueError:
                pass
    except Exception:
        pass
    return out


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
    result = {"name": name, "alive": True, "gpus": gpus,
              "free_ram_mb": sched_free_ram, "actual_free_ram_mb": free_ram, "total_ram_mb": total_ram,
              "free_cpu": free_cpu, "total_cpu": total_cpu, "loadavg": loadavg,
              "cores": cores}
    # Phase 3.3: for WSL2 `local`, fold Windows-host metrics so the TUI can
    # show numbers that match what the user sees in Task Manager. These are
    # *display* values only; dispatch placement still uses the WSL view (the
    # only correct number for "can a new task fit inside the WSL VM").
    if name == "local":
        extras = _probe_windows_host_extras()
        if extras:
            result["host_free_ram_mb"] = extras.get("host_free_ram_mb")
            result["host_total_ram_mb"] = extras.get("host_total_ram_mb")
            cu = extras.get("gpu_compute_util_pct")
            if cu is not None:
                # Single-GPU on local; if multi, attach to all and let the
                # display layer decide.
                for g in result["gpus"]:
                    g["util_pct_compute"] = cu
    return result

def probe_all():
    with ThreadPoolExecutor(max_workers=len(NODES)) as ex:
        nodes = list(ex.map(probe_node, NODES.keys()))
    # Phase 3.2.2 P1: fold cross-scheduler "pending" claims into the probe.
    # The race window: scheduler A calls _ClaimManager.claim() (record pid=null),
    # then runs the slow ssh+nohup launch. Until launch returns and we
    # update_pid, A's process isn't visible to nvidia-smi / ps — so a
    # concurrent scheduler B's probe_all sees the GPU as free and pick_placement
    # picks the same slot. claim() catches the conflict at the atomic-update
    # layer, but we'd rather B not even consider that slot. Solution: subtract
    # all *pending* claims (pid is null) from the probe view here, so
    # _node_resources_ok / _gpu_fits already see the resource as occupied.
    # claims with a real pid are not added — those processes are already
    # visible to ps/nvidia-smi, double-counting would be wrong.
    return _fold_claims_into_probe(nodes)


def _fold_claims_into_probe(nodes: list) -> list:
    """Fold shared claim budgets into the local probe view.

    Pending claims (pid=null) are not visible to ps/nvidia-smi yet, so they
    must consume capacity outright. Live claims have a pid and usually show up
    in the normal probe, but the probe reports *actual* usage while claims
    enforce scheduler *budget* usage. Use the larger of the observed usage and
    the claimed budget so pick_placement and the remote claim arbiter make the
    same decision. This prevents a loop where the picker keeps choosing GPU0
    because actual VRAM is low, while the claim layer rejects GPU0 because its
    estimated/budgeted claims are already past the packing rule.
    """
    for n in nodes:
        if not n.get("alive"):
            continue
        name = n.get("name")
        if not name or not _ClaimManager.enabled_for(name):
            continue
        try:
            snap = _ClaimManager.snapshot(name)
            claims = list(snap.get("claims") or [])
            n["claim_intents"] = list(snap.get("intents") or [])
            n["claim_snapshot_error"] = snap.get("error") if not snap.get("ok", True) else None
        except Exception as e:
            try:
                notify("claims_probe_fold_error",
                       {"node": name, "error": str(e)[:200]},
                       feishu_enabled=False)
            except Exception:
                pass
            continue
        pending = [c for c in claims if not c.get("pid")]
        active = [c for c in claims if c.get("pid")]
        n["pending_claims"] = pending  # surfaced in status / why for debug
        n["active_claims"] = active

        active_cpu = sum(int(c.get("cpu_cores") or 0) for c in active)
        active_ram = sum(int(c.get("ram_mb") or 0) for c in active)
        pending_cpu = sum(int(c.get("cpu_cores") or 0) for c in pending)
        pending_ram = sum(int(c.get("ram_mb") or 0) for c in pending)
        per_gpu_active = {}
        per_gpu_pending = {}
        for c in active:
            g = c.get("gpu_idx")
            if g is None:
                continue
            per_gpu_active[int(g)] = per_gpu_active.get(int(g), 0) + int(c.get("vram_mb") or 0)
        for c in pending:
            g = c.get("gpu_idx")
            if g is None:
                continue
            per_gpu_pending[int(g)] = per_gpu_pending.get(int(g), 0) + int(c.get("vram_mb") or 0)

        if active_cpu or pending_cpu:
            total_cpu = int(n.get("total_cpu") or n.get("cores") or 0)
            if total_cpu > 0:
                observed_used_cpu = max(0, total_cpu - int(n.get("free_cpu") or 0))
                budget_used_cpu = max(observed_used_cpu, active_cpu) + pending_cpu
                n["free_cpu"] = max(0, total_cpu - budget_used_cpu)

        if active_ram or pending_ram:
            total_ram = int(n.get("total_ram_mb") or 0)
            if total_ram > 0:
                observed_used_ram = max(0, total_ram - int(n.get("free_ram_mb") or 0))
                budget_used_ram = max(observed_used_ram, active_ram) + pending_ram
                n["free_ram_mb"] = max(0, total_ram - budget_used_ram)

        for g in n.get("gpus") or []:
            active_claimed = per_gpu_active.get(int(g["idx"]), 0)
            pending_claimed = per_gpu_pending.get(int(g["idx"]), 0)
            if active_claimed or pending_claimed:
                total = int(g.get("total_mb") or 0)
                used = max(int(g.get("used_mb") or 0), active_claimed) + pending_claimed
                g["used_mb"] = used
                if total > 0:
                    g["free_mb"] = min(int(g.get("free_mb") or 0), max(0, total - used))
    return nodes

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


def _set_current_usage(task, vram_mb=0, ram_mb=0, pcpu=0.0):
    """Store current live usage separately from peak/history estimates."""
    task["current_vram_mb"] = int(vram_mb or 0)
    task["current_ram_mb"] = int(ram_mb or 0)
    task["current_pcpu"] = float(pcpu or 0.0)


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
        _set_current_usage(task, 0, 0, 0.0)
        return "dead"
    # alive: fold deltas into peak trackers (max-tracking — peak only goes up)
    _set_current_usage(task, res.get("vram_mb", 0), res.get("ram_mb", 0), res.get("pcpu", 0.0))
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

def _fetch_log_tail(task) -> tuple[str, int]:
    """Phase 3.0.35: shared tail fetcher used by COMPLETED-log scan and
    terminal-orphan diagnosis. Returns (tail_text, log_size).

    Returns ("", 0) on any error — caller decides what to do with empty
    result. Mirrors _diagnose_terminal's primary tail-read path (NOT the
    user-redirect recovery path; that's _diagnose_terminal-only).
    """
    log_path = task.get("log_path")
    if not log_path or task.get("auto_adopted"):
        return ("", 0)
    try:
        node = task.get("node")
        if node and NODES.get(node, {}).get("host") is None:
            lp = Path(log_path)
            if not lp.exists():
                return ("", 0)
            sz = lp.stat().st_size
            with open(lp, "rb") as f:
                f.seek(max(0, sz - 4096))
                return (f.read().decode("utf-8", errors="replace"), sz)
        elif node:
            rc, out, _ = run_on(
                node,
                f"tail -c 4096 {shlex.quote(log_path)} 2>/dev/null; "
                f"echo '___SZ___'; wc -c < {shlex.quote(log_path)} 2>/dev/null",
                timeout=10, check=False,
            )
            if rc != 0 or not out:
                return ("", 0)
            if "___SZ___" in out:
                body, _, sz_str = out.rpartition("___SZ___")
                try:
                    return (body, int(sz_str.strip()))
                except ValueError:
                    return (body, 0)
            return (out, 0)
    except Exception:
        return ("", 0)
    return ("", 0)


def _scan_full_log_for_success(task) -> list:
    """Phase 3.4.15 P0 fix: scan the ENTIRE log file for SUCCESS_PATTERNS,
    not just the 4KB tail.

    Why this exists: `_diagnose_terminal`'s tail-only scan misses the
    success marker when verbose post-train output (wandb sync, mlflow
    flush, lightning trainer summary, etc.) writes more than `tail-c
    4096` bytes AFTER the training script's "Training complete" line.
    Real-world incident: H2Oplus WSRL training prints "Training complete!"
    then wandb dumps ~5KB of metrics + "Synced 5 W&B file(s)" before
    process exit. Tail captured only wandb's verbose section → success
    marker missed → 18 successful runs misclassified as `failed` →
    cascade of pointless heal retries.

    Returns the list of matched SUCCESS_PATTERNS. Empty list = no
    success marker anywhere in the log (or we couldn't read the log).

    Cheap: one grep pass per terminal task. For multi-MB logs the cost
    is one ssh + linear file scan, ~tens of ms; happens at most once
    per task lifetime.
    """
    log_path = task.get("log_path")
    if not log_path or task.get("auto_adopted"):
        return []
    node = task.get("node")
    try:
        if node and NODES.get(node, {}).get("host") is None:
            # Local: stream-scan the whole file in 256KB chunks. Avoids
            # holding a multi-MB log in RAM if the run had a huge wandb /
            # tensorboard verbose tail. First-match wins per pattern; we
            # return ALL patterns observed (most logs hit only one).
            lp = Path(log_path)
            if not lp.exists():
                return []
            seen = set()
            with open(lp, "rb") as f:
                buf = b""
                while True:
                    chunk = f.read(262144)  # 256KB
                    if not chunk:
                        break
                    buf = buf[-256:] + chunk  # carry-over for cross-boundary matches
                    text = buf.decode("utf-8", errors="replace")
                    for p in SUCCESS_PATTERNS:
                        if p not in seen and p in text:
                            seen.add(p)
                    if len(seen) == len(SUCCESS_PATTERNS):
                        break  # all patterns found; stop early
            return list(seen)
        elif node:
            # Remote: one grep -F over ssh. -F = literal strings (no regex
            # surprises in patterns like "Saved: " or "[Done]"). -m 1 stops
            # at first match per pattern alternative — but grep with multiple
            # -e treats them as OR, so -m 1 stops at first overall match
            # (sufficient: we just need to know if ANY success pattern
            # exists in the file). Output is the matched line; we re-scan
            # it locally to identify which pattern(s) hit.
            grep_es = " ".join(f"-e {shlex.quote(p)}" for p in SUCCESS_PATTERNS)
            rc, out, _ = run_on(
                node,
                f"grep -F -m 1 {grep_es} {shlex.quote(log_path)} 2>/dev/null || true",
                timeout=15, check=False,
            )
            if rc != 0 or not out:
                return []
            matched_text = out.strip()
            if not matched_text:
                return []
            return [p for p in SUCCESS_PATTERNS if p in matched_text]
    except Exception:
        return []
    return []


def _scan_completed_log_for_crash(task) -> tuple[bool, str]:
    """Phase 3.0.30 P2: lightweight crash-pattern scan for slurm COMPLETED.

    Pre-fix the slurm terminal_ok=True path trusted slurm's exit code
    unconditionally and never read the log. But pipelines like
    `python train.py | tee out.log` exit rc=0 (tee succeeds) when the LEFT
    side traceback'd — without `set -o pipefail`, slurm sees COMPLETED.
    Scan the log tail for CRASH_PATTERNS only; do NOT apply the lifetime /
    training-marker / success-pattern heuristics from _diagnose_terminal —
    those are reserved for the no-slurm-signal path. Reviewer's request:
    "只做明确错误模式覆盖".

    Returns (matched: bool, reason: str). On tail-fetch failure returns
    (False, "") — be conservative; don't add false crashes when we can't
    read the log.
    """
    tail_text, _log_size = _fetch_log_tail(task)
    if not tail_text:
        return (False, "")
    matched = [p for p in CRASH_PATTERNS if p in tail_text]
    if matched:
        return (True,
                f"slurm reported COMPLETED but log tail contains crash "
                f"pattern(s): {', '.join(matched[:3])}")
    return (False, "")


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
    # Phase 3.4.15 P0 fix: complement tail scan with whole-log scan.
    # Verbose post-train flush (wandb / mlflow / lightning summary etc.)
    # commonly writes 4-10KB after the script's success marker, pushing
    # the marker out of the 4KB tail window. Without this, e.g. H2Oplus
    # WSRL prints "Training complete!" then wandb dumps ~5KB and exits
    # — tail caught only the wandb section, scheduler false-classified
    # 18 successful runs as `failed`. Whole-log grep is one ssh round-
    # trip per terminal task; cheap relative to the cost of mis-routing
    # a successful run through requeue + heal + redundant retry.
    if log_trusted and not success_matched:
        full_log_success = _scan_full_log_for_success(task)
        if full_log_success:
            success_matched = full_log_success
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
    """Categorize a crash diag for routing: ENV_MISSING / PYTHON_IMPORT / INVALID_FLAG / OOM / APP_BUG / UNKNOWN.
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
    # INVALID_FLAG before OOM: argparse / absl rejection happens at startup before any allocation,
    # so the tail will not contain memory-pressure noise. Retry of an invalid-flag cmd is wasted CPU
    # — the cmd will fail identically every time. Surface immediately as an escalation.
    for p in INVALID_FLAG_PATTERNS:
        if p.lower() in haystack:
            return "INVALID_FLAG"
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
    # Dedup guard: refuse to requeue if an active task with the same run identity
    # already exists. This is intentionally broader than PID/task-id and narrower
    # than signature-only: broad family signatures must not block independent
    # ablations, but a true duplicate retry should not double-launch.
    parent_key = _task_run_identity(parent)
    if parent_key:
        if _has_user_cancelled_retry_descendant(parent, state, parent_key):
            parent["last_block_reason"] = (
                "not auto-requeued: a retry descendant for this exact run "
                "identity was cancelled by the user"
            )
            return None
        for existing in state.get("tasks", []):
            if existing.get("id") == parent.get("id"): continue
            if existing.get("status") not in ("queued", "running", "launching"): continue
            if _task_run_identity(existing) != parent_key: continue
            # Found a live duplicate — link parent's requeued_as to it instead of creating a new one
            return existing["id"]
    diag = parent.get("_diagnosis") or {}
    category = _classify_failure(diag)
    parent["failure_category"] = category
    if category in ("ENV_MISSING", "PYTHON_IMPORT", "INVALID_FLAG", "OOM", "DISK_FULL"):
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
        "current_vram_mb": 0,
        "current_ram_mb": 0,
        "current_pcpu": 0.0,
        "alive_pids": [],
        "resume_from": None,
        # Backend launch artifacts must not survive into a retry clone. A crashed
        # Slurm task carrying its old slurm_job_id would be routed back to
        # SlurmBackend even after dispatch picks a local node, and docker container
        # handles from the parent are stale for the new task id.
        "slurm_job_id": None,
        "slurm_state": None,
        # Phase 3.0.29 P2 fix: clear actual_started_at too (Phase 3.0.9 stamp).
        # Pre-fix the retry clone inherited the parent's real-compute timestamp
        # so _effective_elapsed_s would report stale elapsed time as soon as
        # SlurmBackend.batch_probe assigned a slurm_job_id again, corrupting
        # eta_load / migration decisions. The retry hasn't started running on
        # slurm yet, so this must be None.
        "actual_started_at": None,
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
    for k in ("requeued_as", "cancelled_at", "cancelled_by_user", "cancel_reason",
              "claim_intent_node", "claim_intent_nodes", "claim_intent_at"):
        new_task.pop(k, None)
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
            _set_current_usage(t, 0, 0, 0.0)
            t["status"] = "done"
            t["finished_at"] = time.time()
            # Phase 3.2.1: free the cross-scheduler claim now that the task
            # is terminal. Best-effort — failure to release just leaves the
            # claim to expire via TTL + GC. No-op when claims disabled for
            # this node.
            try:
                _release_task_claims_and_intents(t)
            except Exception:
                pass
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
            if res.get("terminal_cancelled"):
                # Slurm CANCELLED means somebody explicitly cancelled/scancelled
                # the job. Treat it as an operator cancellation, not a crash,
                # otherwise a manual move/cancel creates a fresh retry clone.
                _mark_user_cancelled(
                    t,
                    terminal_reason or "slurm terminal state CANCELLED; treated as user/admin cancel",
                )
                diag = {
                    "is_crash": False,
                    "reason": terminal_reason or "slurm terminal state CANCELLED",
                    "tail": "(slurm reported CANCELLED; not auto-requeued)",
                    "lifetime_s": int(max(0, t["finished_at"] - (t.get("started_at") or t["finished_at"]))),
                    "log_size": 0,
                    "log_path": t.get("log_path"),
                    "success_marker": "SLURM_CANCELLED",
                }
            elif terminal_ok is True:
                # Phase 3.0.30 P2 fix: trust-but-verify. A slurm COMPLETED can
                # mask a hidden Python crash if cmd is `python ... | tee log`
                # without `set -o pipefail` — tee returns 0, slurm sees rc=0.
                # Scan the log tail for explicit crash patterns only (no
                # lifetime / marker heuristics; those are for the no-slurm-
                # signal path).
                crash_matched, crash_reason = _scan_completed_log_for_crash(t)
                if crash_matched:
                    diag = {
                        "is_crash": True,
                        "reason": crash_reason,
                        "tail": "(see log_path; pattern detected in tail)",
                        "lifetime_s": int(max(0, t["finished_at"] - (t.get("started_at") or t["finished_at"]))),
                        "log_size": 0,
                        "log_path": t.get("log_path"),
                        "success_marker": None,
                    }
                else:
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
            runtime_history_record(t, duration_s=duration_s)
            continue
        # alive: fold deltas, upward-track ram_mb / cpu_cores estimates
        _remember_last_placement(t)
        t["alive_pids"] = res["alive_pids"]
        total_vram = res["vram_mb"]
        _set_current_usage(t, total_vram, res.get("ram_mb", 0), res.get("pcpu", 0.0))
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


def _effective_elapsed_s(task: dict) -> float:
    """Phase 3.0.9 P2: return seconds since the task ACTUALLY started compute, not
    since launch returned. Critical for slurm tasks that may PEND for hours before
    slurm allocates resources.

    LocalBackend tasks: started_at IS the compute start (launch path runs the cmd
    inline and captures PIDs synchronously) → return now - started_at.

    SlurmBackend tasks: started_at is sbatch return time, which may be hours
    before slurm actually starts running the job. Use actual_started_at, set
    once by batch_probe when slurm_state first transitions to RUNNING. Until
    then (PENDING/CONFIGURING/None) elapsed = 0 — the task hasn't started, so
    ETA shouldn't decay and node load shouldn't shrink.

    Without this, _refresh_eta_from_logs's EWMA fallback decays a slurm-PENDING
    task's eta_seconds to 0 just from sitting in slurm's queue — corrupts
    eta_load and tricks Phase 3.0.3 migration into routing tasks toward the
    "free" node that's actually loaded but pending."""
    now = time.time()
    started_at = task.get("started_at")
    if not started_at:
        return 0.0
    # Slurm task: prefer actual_started_at; fall back to "still pending → elapsed=0"
    if task.get("slurm_job_id"):
        actual = task.get("actual_started_at")
        if actual:
            return max(0.0, now - actual)
        # No RUNNING observation yet — task is PENDING (or just-sbatched, not yet
        # probed). Treat as elapsed=0 so ETA = full EWMA, load doesn't shrink.
        return 0.0
    # LocalBackend / non-slurm task: started_at == compute start
    return max(0.0, now - started_at)


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
        ewma = _runtime_total_history_s(t) or int(h.get("dur_s_ewma", 0))
        elapsed = _effective_elapsed_s(t)
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
                    ewma = _runtime_total_history_s(t) or int(h.get("dur_s_ewma", 0))
                    elapsed = _effective_elapsed_s(t)
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
            ewma = _runtime_total_history_s(t) or int(h.get("dur_s_ewma", 0))
            elapsed = _effective_elapsed_s(t)
            t["eta_seconds"] = eta_tracker.compute_eta_seconds(
                tail_text, elapsed_s=elapsed, fallback_ewma_s=ewma, cmd=t.get("cmd"),
            )
            projection = eta_tracker.runtime_projection(
                tail_text, elapsed_s=elapsed, cmd=t.get("cmd"))
            _apply_runtime_projection(t, projection)

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
    fixed_headroom = node_info.get("ram_headroom_mb")
    if fixed_headroom is not None:
        headroom = max(0, int(fixed_headroom))
    else:
        frac = node_info.get("ram_headroom_frac", RAM_HEADROOM_FRAC)
        # Use the probed total when available so headroom tracks reality, not config; the config
        # value is only used as an upper bound (already enforced in probe_node via min()).
        total_for_headroom = node_state.get("total_ram_mb") or node_info.get("ram_mb", 0)
        headroom = int(total_for_headroom * frac)
    if node_state.get("free_ram_mb", 0) - needed_ram < headroom:
        return False, f"ram: need {needed_ram}MB, free {node_state.get('free_ram_mb', 0)}MB (headroom {headroom}MB)"
    return True, "ok"

def _node_gpu_util_limit(node_info):
    util_limit = (node_info or {}).get("gpu_util_saturation_pct", GPU_UTIL_SATURATION_PCT)
    if isinstance(util_limit, str) and util_limit.lower() in ("", "none", "off", "false", "ignore"):
        return None
    if util_limit is None:
        return None
    return int(util_limit)


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
            util_limit = _node_gpu_util_limit(node_info)
            util = gpu.get("util_pct", 0)
            chip_idle = util_limit is None or util < util_limit - 20  # well below saturation, e.g. <65%
            if not (small_task and chip_idle):
                return False
            # else: small task, idle chip — allow stacking past 1/3 mem
    # Compute saturation: if there's already a task on this GPU and it's pinning the chip,
    # don't pack more — the new task would just steal cycles and slow everyone down.
    # The "occupied" guard (>100MB) avoids blocking on a transient util spike on an empty GPU.
    util_limit = _node_gpu_util_limit(node_info)
    if util_limit is not None and gpu["used_mb"] > 100 and gpu.get("util_pct", 0) >= util_limit:
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
    # Phase 3.0.15 P1 fix: a task that was migrated has its cwd/ckpt staged ONLY
    # on staged_node. Without promoting that to a hard pin, pick_placement's
    # fallback path could land the task on a third node where staging never ran
    # → resume task silently restarts from step 0. User-explicit require_node
    # still wins over the migration pin (operator override beats auto-balance).
    staged = task.get("staged_node")
    if staged and not require:
        require = staged
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
        # Phase 2.16/3.4.13: throttle when this slurm node's matching bucket
        # (CPU-only vs GPU-using) already has its pending cap filled. Keeping
        # the rest in scheduleurm's queue lets tasks spread to whichever slurm
        # node frees up next, instead of piling on one host.
        if not _requires_local_capacity_check(n["name"], task):
            # Phase 3.4.13 P1 fix: throttle by resource bucket, not total.
            # CPU-only and GPU-using tasks request different slurm gres
            # (no --gpus vs --gpus 1) so they have ZERO real conflict in
            # slurm's scheduler. Pre-fix's single combined count meant a
            # pending GPU training job would block a CPU-only eval task
            # from joining the queue, even though slurm could trivially
            # run them in parallel.
            split = n.get("slurm_pending_split") or {"cpu": 0, "gpu": 0}
            bucket = _slurm_pending_bucket_for_task(task)
            pending = int(split.get(bucket) or 0)
            cap = _slurm_max_pending_for_node(n["name"], bucket)
            if pending >= cap:
                return []  # throttled — let this bucket drain
            return [((9999,), n["name"], None)]
        if _task_requests_slurm(task):
            # The user supplied Slurm-only fields, so do not silently discard
            # them by launching through LocalBackend on a non-Slurm/default-local
            # node. If no Slurm-capable/opt-in node exists, the task stays queued
            # with an explicit reason instead of running with the wrong policy.
            return []
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
                # Empty-first placement: if a node has a genuinely idle GPU, use it before
                # stacking onto a warm card. This is a micro-server policy, not a datacenter
                # bin-packing policy: keeping one card busy while another sits empty slows the
                # user's batch down more than it preserves a hypothetical future fragment.
                # If all fitting cards are warm, keep best-fit to avoid needless spreading.
                fits_remaining = g["free_mb"] - (task.get("est_vram_mb") or 0)
                occupied = 1 if g["used_mb"] >= GPU_EMPTY_USED_MB else 0
                score = (occupied, fits_remaining)
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
    # Phase 3.0.27: pre-resolve node / node_host so the conda branch below can
    # consult _conda_sync_ok() without tripping over the docker branch's later
    # binding.
    node = task.get("node")
    node_host = NODES.get(node, {}).get("host") if node else None
    if kind == "conda":
        # Conda path: no cmd-wrapping needed — user's cmd already references absolute python
        # path that should now exist on target (preload rsync'd it). Bare `inner` is correct.
        # If env didn't preload (e.g., target down), launch will fail with ENV_MISSING and
        # heal flow takes over — no different from legacy `none` behavior.
        #
        # Phase 3.0.22 P2 fix: but if user passed `conda:/abs/path` and the local
        # source path doesn't exist, preload silently skipped the rsync (line in
        # _preload_env_outside_lock guards `Path.is_dir()`). Launching anyway
        # would let a stale remote env at the same path silently run — same
        # blast-radius shape as the docker stale-tag P1. Fail-fast at launch
        # so the user sees the misconfiguration before compute is wasted.
        if spec_image and Path(spec_image).is_absolute() and not Path(spec_image).is_dir():
            return (inner,
                    f"--env-spec conda:{spec_image} but the env path does not "
                    f"exist locally — preload skipped (nothing to rsync); "
                    f"launching now would risk running a stale remote env at "
                    f"the same path. Create/deploy the env locally first, "
                    f"then resubmit.")
        # Phase 3.0.27 P1 fix: even when local path is fine, the latest
        # push_conda_env to this node may have failed (ssh blip, target disk
        # full, rsync timeout). Pre-fix, launch trusted preload to have
        # synced — but preload's failure was only logged, never gated. If
        # the remote happened to have a stale env at the same path (a prior
        # sync that succeeded), the launch silently used it.
        # Skip the check for local nodes (no host) and for non-absolute
        # specs (conda activate <name>); both bypass the rsync path entirely.
        if (node_host and spec_image
                and Path(spec_image).is_absolute()
                and not _conda_sync_ok(node, spec_image)):
            return (inner,
                    f"--env-spec conda:{spec_image} but the latest sync to "
                    f"{node} did not succeed (or has not run yet) — refusing "
                    f"to launch; would risk running a stale remote env. Wait "
                    f"for the next dispatch cycle's preload, or check "
                    f"~/.claude/scheduler/logs/watcher.log for "
                    f"preload_conda_failed events to diagnose.")
        return (inner, None)
    chosen_image = spec_image or image
    # node / node_host already resolved above the conda branch (Phase 3.0.27).
    explicit = (kind == "docker")
    if not chosen_image:
        if explicit:
            return (inner, "--env-spec docker requires --image (or 'docker:IMAGE' inline)")
        return (inner, None)  # auto without image → graceful fallback
    if not env_deploy.has_docker(run_on, node, timeout=8):
        if explicit:
            return (inner, f"--env-spec docker requested but `docker info` failed on {node}")
        return (inner, None)  # auto → graceful fallback
    # Image presence + digest check at launch. Phase 3.0.34 P1 fix: the local
    # digest probe used to be gated by `if node_host:` — local nodes silently
    # bypassed both the explicit fail-fast (3.0.21) and the auto fallback
    # (3.0.26). Help text says "Image must exist locally" but the launch path
    # didn't enforce it: a local docker `run` against a missing image would
    # implicitly try to pull, run a non-expected tag, or fail with a confusing
    # error. Hoist the local-digest probe out of the node_host gate so the
    # same fail-fast / fallback policy applies regardless of node locality.
    local_digest = env_deploy.get_image_digest(run_on, "local", chosen_image)
    # Phase 3.0.21 + 3.0.34 P1 fix: explicit `docker:IMAGE` with no local
    # digest must fail-fast on BOTH local and remote nodes. has_image() with
    # local_digest=None falls back to tag-presence (the legacy fast path);
    # without a local digest there's no drift detection AND no proof the
    # image we'd run on local is the one the user intends. Build/pull
    # locally first.
    if explicit and local_digest is None:
        return (inner,
                f"--env-spec docker:{chosen_image} but image not present "
                f"locally (no digest available); refusing to launch — would "
                f"risk running a stale or unintended image. Build/pull the "
                f"image locally first, then resubmit.")
    # Phase 3.0.26 + 3.0.34 P1 fix: same staleness risk for `auto` mode on
    # both local and remote. Without a local digest, freshness can't be
    # verified — fall back to bare cmd (kind=none equivalent).
    if not explicit and local_digest is None:
        return (inner, None)
    if node_host:
        if not env_deploy.has_image(run_on, node, chosen_image, local_digest=local_digest):
            # Phase 3.0.31 P3 fix: do NOT push synchronously here. push_image
            # takes up to 1800s, and _maybe_wrap_docker runs inside the
            # dispatch state_lock — a single missing image could starve
            # submit/status/cancel/watcher for 30 min. Both cmd_dispatch and
            # _watch_iteration already call _preload_docker_images_outside_lock
            # BEFORE acquiring this lock; preload is the right place for the
            # transfer. If we reach this branch, preload either failed or
            # hasn't run for this image yet — bail fast and let the next
            # dispatch cycle's preload retry. cmd_dispatch / watcher.log
            # surface the preload failure directly, so the user can diagnose
            # without reading launch-time errors.
            err = (f"docker image {chosen_image} not present (or digest "
                   f"drift) on {node} at launch — preload not yet "
                   f"successful. Will retry next dispatch cycle; if this "
                   f"keeps happening, check ~/.claude/scheduler/logs/"
                   f"watcher.log for preload_image_failed events.")
            if explicit:
                return (inner, err)
            return (inner, None)  # auto → graceful fallback (no docker wrap)
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

    def requires_local_capacity_check(self, node: str, task: Optional[dict] = None) -> bool:
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

    def launch(self, task: dict, node_state: Optional[dict] = None) -> tuple[bool, str]:
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

    def launch(self, task: dict, node_state: Optional[dict] = None) -> tuple[bool, str]:
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
        # Phase 3.2.1 P1 fix: cross-scheduler claim. When the node has
        # enable_claims=True, atomically reserve CPU/RAM/VRAM in
        # /tmp/scheduleurm/claims.json BEFORE ssh+nohup. If another
        # scheduleurm (different state dir / user) already claimed the
        # resource, conflict → return CLAIM_RACE sentinel so dispatch
        # treats it as contention (revert to queued, no fail-count
        # increment) rather than a real launch failure.
        claim_record = None
        if _ClaimManager.enabled_for(task["node"]):
            original_gpu = task.get("gpu_idx")
            gpu_attempts = [original_gpu]
            # If the picked GPU loses a race, try other GPUs on the same node
            # before punting the task back to the next watcher tick. This keeps
            # a busy GPU0 claim from starving an otherwise-free GPU1.
            if original_gpu is not None and node_state:
                try:
                    node_info = NODES[task["node"]]
                    for g in node_state.get("gpus") or []:
                        gi = g.get("idx")
                        if gi == original_gpu or gi in gpu_attempts:
                            continue
                        if _gpu_fits(task, g, node_info):
                            gpu_attempts.append(gi)
                except Exception:
                    pass
            conflicts = []
            for gi in gpu_attempts:
                task["gpu_idx"] = gi
                ok_claim, info, kind = _ClaimManager.claim(
                    task["node"], task, gi, node_state)
                if ok_claim:
                    claim_record = info
                    break
                # Phase 3.4.3 P1 fix: distinguish CAPACITY conflict from
                # TRANSPORT error. Conflict → CLAIM_RACE: dispatch treats
                # as contention (queued, no fail count increment, retry
                # next cycle). Transport error → CLAIM_ERROR: dispatch
                # treats as a real launch failure so MAX_LAUNCH_RETRY +
                # escalation can fire (otherwise a node with permission /
                # python3 / flock issues would loop tasks forever).
                if kind != "conflict":
                    task["gpu_idx"] = original_gpu
                    return False, f"CLAIM_ERROR: {info}"
                conflicts.append(f"gpu{gi}: {info}" if gi is not None else str(info))
            if not claim_record:
                task["gpu_idx"] = original_gpu
                return False, f"CLAIM_RACE: {'; '.join(conflicts) or 'claim conflict'}"

        def _release_and_fail(msg):
            """Release the claim if we hold one, then return failure."""
            if claim_record:
                try:
                    _release_task_claims_and_intents(task)
                except Exception:
                    pass
            return False, msg

        # Set CUDA_VISIBLE_DEVICES. For GPU tasks: pin to assigned GPU (will appear as device 0 inside).
        # For CPU-only tasks (gpu_idx=None): set empty string so CUDA truly sees no GPUs — the literal
        # string "None" is NOT a valid CUDA value and would not disable GPU access.
        gpu_idx = task.get("gpu_idx")
        if gpu_idx is None:
            env_prefix = 'export CUDA_VISIBLE_DEVICES=""; '
        else:
            env_prefix = f"export CUDA_VISIBLE_DEVICES={gpu_idx}; "
        # Phase 3.0.28 P1 fix: inject SCHEDULEURM_TASK_ID so WAL orphan recovery
        # can identify a launched-but-not-yet-saved process. If scheduler dies
        # between LocalBackend.launch returning success and save_state flushing
        # status=running + remote_pids, the orphan still has this marker in
        # /proc/<pid>/environ — _try_recover_orphan_local_task scans for it on
        # the candidate node and adopts the orphan instead of reverting +
        # double-launching the task.
        env_prefix += f"export SCHEDULEURM_TASK_ID={shlex.quote(task.get('id') or '')}; "
        # Phase 3.0.23 P2 fix: filter extra_env at the export site so legacy
        # state.json entries with invalid / reserved keys can't break the
        # export shell or override CUDA_VISIBLE_DEVICES set above.
        for k, v in _safe_extra_env_items(task.get("extra_env")):
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
                return _release_and_fail(f"launch rc={rc}: {err.strip()[:200]}")
            pid = None
            for line in out.splitlines():
                line = line.strip()
                if line.startswith("PID="):
                    try: pid = int(line[4:])
                    except ValueError: pass
            if not pid:
                return _release_and_fail(
                    f"could not parse PID from launch output: {out[:200]}")
            task["remote_pids"] = [pid]
            task["process_group"] = pid
            task["log_path"] = log_path
            task["status"] = "running"
            task["started_at"] = time.time()
            _remember_last_placement(task)
            task["peak_vram_mb"] = 0
            task["peak_ram_mb"] = 0
            _set_current_usage(task, 0, 0, 0.0)
            # Phase 3.2.1: persist PID into the claim so other schedulers
            # see liveness; gc_stale won't touch a claim with a live PID.
            if claim_record:
                try:
                    _ClaimManager.update_pid(task["node"], task["id"], pid)
                except Exception:
                    pass
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
                    # Phase 3.2.1: refresh the claim's PID to track the
                    # container's actual main proc, not the bash launcher
                    # (which exits as soon as `docker run` detaches).
                    if claim_record:
                        try:
                            _ClaimManager.update_pid(task["node"], task["id"], container_pid)
                        except Exception:
                            pass
            return True, f"pid={pid}" + (f" container={cname}@{task.get('container_main_pid')}" if cname else "")
        except Exception as e:
            return _release_and_fail(f"launch exception: {e}")

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
                   # Phase 3.0.25 P1 fix: include `stat=` so the parser can
                   # filter Z (zombie) / X (dead) processes out of rss_per_pid.
                   # Without this, zombies still appeared in the `ps` output
                   # after the per-PID `/proc/<pid>/status` Z/X check, and the
                   # `set(rss_per_pid)` union below silently re-marked them as
                   # alive — a task with all descendants reaped to zombies
                   # could stay status=running forever.
                   f"ps -eo pid=,ppid=,rss=,pcpu=,stat= 2>/dev/null; true")
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
                # Phase 3.0.25 P1 fix: require the `stat=` column we now request
                # and skip Z (zombie) / X (dead). Pre-fix this loop took 4
                # columns and unconditionally added every PID to rss_per_pid;
                # zombies then re-entered `this_alive` via the `set(rss_per_pid)`
                # union, defeating the per-root /proc/status Z/X filter above.
                if len(bits) < 5: continue
                try:
                    p_, parent_, rss_kb = int(bits[0]), int(bits[1]), int(bits[2])
                    pc = float(bits[3])
                except ValueError: continue
                stat = bits[4]
                if stat and stat[0] in ("Z", "X"):
                    continue  # zombie / dead — do NOT count as alive
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

    def requires_local_capacity_check(self, node: str, task: Optional[dict] = None) -> bool:
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
        """Pick --time= value in seconds.

        Priority:
          1. exact-parameter runtime history from tqdm/progress/duration × 1.2
          2. legacy per-signature duration EWMA × 3
          3. unknown fallback 24h
        """
        runtime_wall = _runtime_walltime_for_task(task)
        if runtime_wall > 0:
            return max(RUNTIME_MIN_WALLTIME_S, min(SlurmBackend.MAX_WALLTIME_S, runtime_wall))
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
        vram = int(task.get("est_vram_mb", DEFAULT_VRAM_MB) or 0)
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
        # Phase 3.0.23 P2 fix: filter extra_env via _safe_extra_env_items so any
        # legacy state.json entry with an invalid / reserved key (e.g. one that
        # would override slurm's gres-bound CUDA_VISIBLE_DEVICES) can't break
        # the sbatch script. submit-time validation is the primary gate.
        for k, v in _safe_extra_env_items(task.get("extra_env")):
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

    def launch(self, task: dict, node_state: Optional[dict] = None) -> tuple[bool, str]:
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
        gpu_runtime_env = "CUDA_VISIBLE_DEVICES" if int(task.get("est_vram_mb", DEFAULT_VRAM_MB) or 0) > 0 else None
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
            _remember_last_placement(task)
            # Phase 3.0.29 P2 fix: clear actual_started_at on (re)launch. The
            # 3.0.9 stamp is set ONLY when batch_probe first observes
            # slurm_state=RUNNING. Pre-fix it could survive into a relaunched
            # task (if the parent state was reused, or the same task object got
            # recycled via rebalance-pending), making _effective_elapsed_s
            # report stale elapsed seconds while the new job sits PENDING.
            task["actual_started_at"] = None
            task["peak_vram_mb"] = 0
            task["peak_ram_mb"] = 0
            _set_current_usage(task, 0, 0, 0.0)
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
                          "PREEMPTED", "BOOT_FAIL", "DEADLINE",
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
                terminal_cancelled = False
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
                    # Phase 3.0.9 P2: record the actual compute-start time the FIRST
                    # time we observe RUNNING. Until this is set, _effective_elapsed_s
                    # returns 0 for slurm tasks so ETA / load stay at "full ewma" while
                    # PENDING. Without this, a job pending in slurm's queue for an hour
                    # silently decays its ETA to 0 and migration sees a fake "free" node.
                    if slurm_state == "RUNNING" and not t.get("actual_started_at"):
                        t["actual_started_at"] = time.time()
                elif slurm_state in SUCCESS_STATES:
                    state_norm = "dead"
                    terminal_ok = True
                    terminal_cancelled = False
                    terminal_reason = f"slurm terminal state {slurm_state}"
                    t["slurm_state"] = slurm_state
                elif slurm_state == "CANCELLED":
                    state_norm = "dead"
                    terminal_ok = False
                    terminal_cancelled = True
                    terminal_reason = "slurm terminal state CANCELLED; treated as user/admin cancel"
                    t["slurm_state"] = slurm_state
                else:
                    # Unknown terminal-ish states are safer as failures than "maybe done":
                    # if Slurm did not say the job is alive or COMPLETED, scheduleurm should
                    # retry/escalate rather than silently drop work.
                    state_norm = "dead"
                    terminal_ok = False
                    terminal_cancelled = False
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
                                    "terminal_cancelled": terminal_cancelled,
                                    "terminal_reason": terminal_reason}
        return results


def _task_requests_slurm(task: Optional[dict]) -> bool:
    """True when the user supplied Slurm-only task options.

    These options should never be silently ignored by LocalBackend. They only
    become launchable on nodes that are configured or probed as Slurm-capable.
    """
    if not task:
        return False
    return bool(task.get("slurm_partition") or task.get("slurm_account") or task.get("slurm_qos"))


def _slurm_mode_enabled(value) -> bool:
    return str(value or "").strip().lower() in ("slurm", "true", "1", "yes", "on")


def _slurm_mode_auto(value) -> bool:
    return str(value or "").strip().lower() == "auto"


class HybridBackend(Backend):
    """Per-node routing: default LocalBackend; SlurmBackend is opt-in.

    Slurm detection is still available, but it is a capability probe, not the
    default policy. That matters for small personal nodes with Slurm installed:
    scheduleurm should keep doing its own VRAM/RAM/CPU packing unless the node
    or the task explicitly asks for Slurm.
    """
    name = "hybrid"

    def __init__(self):
        self._local = LocalBackend()
        self._slurm = SlurmBackend()
        self._cache: dict = {}  # node_name -> 'slurm' | 'local'

    def _node_wants_slurm(self, node: str, task: Optional[dict] = None) -> bool:
        """Policy decision: should this future launch use SlurmBackend?

        Defaults are deliberately local. Operators can opt in globally per node
        (`slurm_backend="slurm"`), per resource bucket
        (`slurm_gpu_backend="slurm"` / `slurm_cpu_backend="slurm"`), or by using
        explicit task Slurm fields. `auto` preserves the old "probe and use Slurm
        if installed" behavior for nodes that really are shared clusters.
        """
        info = NODES.get(node, {}) or {}
        if _slurm_mode_enabled(info.get("slurm_backend")):
            return True
        if _slurm_mode_auto(info.get("slurm_backend")):
            return self._kind_for(node) == "slurm"

        is_gpu_task = int((task or {}).get("est_vram_mb", DEFAULT_VRAM_MB) or 0) > 0
        bucket_key = "slurm_gpu_backend" if is_gpu_task else "slurm_cpu_backend"
        if _slurm_mode_enabled(info.get(bucket_key)):
            return True
        if _slurm_mode_auto(info.get(bucket_key)):
            return self._kind_for(node) == "slurm"

        # Explicit task Slurm knobs mean "send me to a Slurm-capable node", but
        # keep non-Slurm local nodes out of the candidate set in pick_placement.
        if _task_requests_slurm(task):
            return self._kind_for(node) == "slurm"
        return False

    def requires_local_capacity_check(self, node: str, task: Optional[dict] = None) -> bool:
        """True for scheduleurm-managed placement; False for Slurm opt-in nodes."""
        return not self._node_wants_slurm(node, task)

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
        # The last case deliberately reports "slurm" for this capability check but is
        # not cached. Policy still has to opt in via _node_wants_slurm before any launch
        # uses SlurmBackend.
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

    def _backend_for(self, node: str, task: Optional[dict] = None) -> Backend:
        return self._slurm if self._node_wants_slurm(node, task) else self._local

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
        - neither → queued, use opt-in Slurm policy for the upcoming launch
        """
        if task.get("slurm_job_id"):
            return self._slurm
        if task.get("remote_pids"):
            return self._local
        node = task.get("node")
        if not node:
            return self._local  # no node yet (queued task): launch path will re-route
        return self._backend_for(node, task)

    def launch(self, task: dict, node_state: Optional[dict] = None) -> tuple[bool, str]:
        return self._backend_for_task(task).launch(task, node_state=node_state)

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


# Singleton: HybridBackend defaults to scheduleurm LocalBackend placement; SlurmBackend
# is used only for explicit task Slurm fields or per-node opt-in config.
# Tests reference _BACKEND directly to verify backend identity and to swap in fakes.
_BACKEND: Backend = HybridBackend()


def _requires_local_capacity_check(node: str, task: Optional[dict] = None) -> bool:
    """Compatibility wrapper for tests/plugins with older backend stubs."""
    try:
        return _BACKEND.requires_local_capacity_check(node, task=task)
    except TypeError:
        return _BACKEND.requires_local_capacity_check(node)


def launch(task, node_state=None):
    """Thin wrapper: delegates to the active Backend. See Backend.launch.
    node_state (a probe_all entry) lets backends that need it (LocalBackend
    when claims are enabled) build a capacity payload without a second probe."""
    return _BACKEND.launch(task, node_state=node_state)

# ---------- subcommands ----------
def _cmd_looks_like_training(cmd: str) -> bool:
    """Heuristic: cmd invokes a training entry-point. Catches `train_*.py`, `/train.py`, `trainer.py`,
    and a few common framework patterns. False positives are acceptable — the user can override
    with --allow-cpu-training."""
    lower = (cmd or "").lower()
    if re.search(r"\bpython(?:\d(?:\.\d+)?)?\b[^\n;]*\s-m\s+[\w.]*train\b", lower):
        return True
    if re.search(r"\bpython(?:\d(?:\.\d+)?)?\b[^\n;]*\b[\w./-]*train[\w./-]*\.py\b", lower):
        return True
    return any(p in lower for p in (
        "train_", "/train.py", " train.py", "trainer.py",
        "/main_train", "run_train", "do_train",
        "h2o+_bus_main.py",
    ))

def _safe_read_text(path: str, max_bytes: int = 512 * 1024) -> str:
    try:
        with open(path, "rb") as f:
            return f.read(max_bytes).decode("utf-8", errors="replace")
    except Exception:
        return ""

def _script_invocation_from_cmd(cmd: str, cwd: str = ""):
    """Return (script_path, script_args) for simple `bash script.sh ...` commands.

    This is intentionally conservative: it only opens a local wrapper file when
    the command shape is clear. The scheduler still launches the original cmd;
    this helper is only for submit-time safety policy.
    """
    try:
        toks = shlex.split(cmd or "")
    except Exception:
        return (None, [])
    if not toks:
        return (None, [])
    first = os.path.basename(toks[0])
    script_i = None
    if first in ("bash", "sh", "zsh", "dash"):
        i = 1
        while i < len(toks):
            tok = toks[i]
            if tok in ("-c", "-lc"):
                return (None, [])
            if tok.startswith("-"):
                i += 1
                continue
            script_i = i
            break
    elif first == "env" and len(toks) >= 3 and os.path.basename(toks[1]) in ("bash", "sh", "zsh", "dash"):
        script_i = 2
    elif toks[0].endswith(".sh"):
        script_i = 0
    if script_i is None or script_i >= len(toks):
        return (None, [])
    script = toks[script_i]
    if not os.path.isabs(script):
        script = os.path.abspath(os.path.join(cwd or os.getcwd(), script))
    if not os.path.exists(script) or not os.path.isfile(script):
        return (None, [])
    return (script, toks[script_i + 1:])

def _submit_policy_text(cmd: str, cwd: str = "") -> str:
    """Text used by submit-time policy checks.

    The launched command remains unchanged, but policy must inspect wrappers such
    as `bash run_seed.sh ...`; otherwise a training job can hide its actual
    `python -m ...train --resume` behind a one-line shell entrypoint.
    """
    parts = [cmd or ""]
    script, _ = _script_invocation_from_cmd(cmd, cwd)
    if script:
        text = _safe_read_text(script)
        if text:
            parts.append(f"\n# scheduleurm-inspected-wrapper: {script}\n{text}")
    return "\n".join(parts)

def _shell_split_statements(text: str):
    for line in (text or "").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        for part in line.split(";"):
            part = part.strip()
            if part:
                yield part

def _shell_expand_simple(value: str, env: dict) -> str:
    value = (value or "").strip()
    if ((value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))):
        value = value[1:-1]
    m = re.fullmatch(r"\$(\d+)", value)
    if m:
        return env.get(m.group(1), "")
    m = re.fullmatch(r"\$\{(\d+):-([^}]*)\}", value)
    if m:
        return env.get(m.group(1), "") or m.group(2)

    def repl(mo):
        name = mo.group(1)
        suffix = mo.group(2)
        val = env.get(name, "")
        if suffix and val.endswith(suffix):
            return val[:-len(suffix)]
        return val

    value = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?:%([^}]+))?\}", repl, value)
    value = re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", lambda mo: env.get(mo.group(1), ""), value)
    return value

def _simple_shell_env_from_script(script_text: str, script_args: list) -> dict:
    env = {str(i + 1): str(v) for i, v in enumerate(script_args or [])}
    for stmt in _shell_split_statements(script_text):
        if stmt.startswith(("export ", "local ")):
            stmt = stmt.split(None, 1)[1].strip()
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", stmt)
        if not m:
            continue
        name, val = m.group(1), m.group(2).strip()
        # Command substitution makes the value host/runtime dependent; leave it
        # unresolved so we do not invent a false checkpoint path.
        if "$(" in val or "`" in val:
            continue
        env[name] = _shell_expand_simple(val, env)
    return env

def _extract_script_cd_dir(script_text: str, script_path: str = "", cwd: str = "") -> str:
    for stmt in _shell_split_statements(script_text):
        m = re.match(r"^cd\s+(.+)$", stmt)
        if not m:
            continue
        target = _shell_expand_simple(m.group(1).strip(), {
            "HOME": str(Path.home()),
            "0": script_path or "",
        })
        if target in ('"$(dirname "$0")"', "'$(dirname \"$0\")'") and script_path:
            return os.path.dirname(script_path)
        if target.startswith("$(dirname"):
            return os.path.dirname(script_path) if script_path else (cwd or os.getcwd())
        if not os.path.isabs(target):
            base = os.path.dirname(script_path) if script_path else (cwd or os.getcwd())
            target = os.path.abspath(os.path.join(base, target))
        return target
    return cwd or (os.path.dirname(script_path) if script_path else os.getcwd())

def _arg_value(tokens: list, name: str):
    prefix = name + "="
    for i, tok in enumerate(tokens or []):
        if tok == name and i + 1 < len(tokens):
            return tokens[i + 1]
        if tok.startswith(prefix):
            return tok[len(prefix):]
    return None

def _abs_under(base: str, path: str) -> str:
    if not path:
        return ""
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(base or os.getcwd(), path))

def _infer_direct_jax_train_ckpt(cmd: str, cwd: str = ""):
    try:
        toks = shlex.split(cmd or "")
    except Exception:
        return None
    save_root = _arg_value(toks, "--save_root") or _arg_value(toks, "--save-root")
    run_name = _arg_value(toks, "--run_name") or _arg_value(toks, "--run-name")
    has_resume = _cmd_has_resume_flag(cmd)
    if save_root and run_name:
        base = cwd or os.getcwd()
        return {
            "ckpt_dir": os.path.join(_abs_under(base, save_root), run_name, "checkpoints"),
            "resume_managed_by_cmd": bool(has_resume),
            "source": "python-args",
        }
    return None

def _infer_wrapper_ckpt(cmd: str, cwd: str = ""):
    script, script_args = _script_invocation_from_cmd(cmd, cwd)
    if not script:
        return None
    text = _safe_read_text(script)
    if not text:
        return None
    env = _simple_shell_env_from_script(text, script_args)
    script_cwd = _extract_script_cd_dir(text, script, cwd)

    save_root = None
    m = re.search(r"--save[_-]root\s+([^\s\\]+)", text)
    if m:
        save_root = _shell_expand_simple(m.group(1), env)
    run_name = env.get("RUN_NAME")
    if not run_name:
        m = re.search(r"--run[_-]name\s+([^\s\\]+)", text)
        if m:
            run_name = _shell_expand_simple(m.group(1), env)
    if not (save_root and run_name):
        return None
    if "$" in save_root or "$" in run_name or not run_name.strip():
        return None
    return {
        "ckpt_dir": os.path.join(_abs_under(script_cwd, save_root), run_name, "checkpoints"),
        "resume_managed_by_cmd": _cmd_has_resume_flag(text),
        "source": f"wrapper:{os.path.basename(script)}",
    }

def _infer_checkpoint_from_submit(cmd: str, cwd: str = ""):
    return _infer_direct_jax_train_ckpt(cmd, cwd) or _infer_wrapper_ckpt(cmd, cwd)

def _candidate_training_source_paths(cmd: str, cwd: str = ""):
    paths = []
    script, _ = _script_invocation_from_cmd(cmd, cwd)
    script_text = _safe_read_text(script) if script else ""
    policy = _submit_policy_text(cmd, cwd)
    bases = []
    for base in (cwd, _extract_script_cd_dir(script_text, script, cwd) if script_text else "", os.path.dirname(script) if script else ""):
        if base and base not in bases:
            bases.append(base)
    if not bases:
        bases = [os.getcwd()]

    for mod in re.findall(r"-m\s+([A-Za-z_][\w.]*train[A-Za-z0-9_.]*)", policy):
        rel = mod.replace(".", os.sep) + ".py"
        for base in bases:
            p = os.path.join(base, rel)
            if os.path.exists(p):
                paths.append(os.path.normpath(p))
    for py in re.findall(r"(?<![A-Za-z0-9_./-])([A-Za-z0-9_./-]*train[A-Za-z0-9_./-]*\.py)", policy):
        for base in bases:
            p = py if os.path.isabs(py) else os.path.join(base, py)
            if os.path.exists(p):
                paths.append(os.path.normpath(p))

    # If train.py imports a sibling checkpoint module, include it in the contract
    # source so save/load details are checked too.
    expanded = []
    for p in paths:
        if p not in expanded:
            expanded.append(p)
        src = _safe_read_text(p)
        if "jax_experiments.common.checkpoint" in src:
            for base in bases:
                cp = os.path.join(base, "jax_experiments", "common", "checkpoint.py")
                if os.path.exists(cp) and os.path.normpath(cp) not in expanded:
                    expanded.append(os.path.normpath(cp))
    return expanded

def _checkpoint_contract_reason(cmd: str, cwd: str, ckpt_dir, resume_flag, allow_no_resume: bool):
    """Static submit-time check that resume is not just syntactically wired.

    This cannot prove bitwise determinism for every framework, but it catches the
    dangerous class where a command has a `--resume`-looking flag yet the code does
    not save enough state to continue from the saved iteration.
    """
    policy_cmd = _submit_policy_text(cmd, cwd)
    if allow_no_resume or not _cmd_looks_like_training(policy_cmd) or not ckpt_dir:
        return None
    if not (_cmd_has_resume_flag(policy_cmd) or resume_flag):
        return None  # _resume_capability_reason reports the clearer wiring error.
    paths = _candidate_training_source_paths(cmd, cwd)
    if not paths:
        return ("could not inspect local training source for checkpoint contract. "
                "Submit with a local cwd/wrapper the scheduler can read, or pass "
                "--allow-no-resume to explicitly accept non-resumable relaunches.")
    src = "\n".join(_safe_read_text(p) for p in paths)
    lower = src.lower()
    checks = [
        ("save_checkpoint call", "save_checkpoint(" in src or "torch.save(" in src or ".save_checkpoint(" in src),
        ("load_checkpoint call", "load_checkpoint(" in src or "torch.load(" in src or ".load_checkpoint(" in src),
        ("checkpoint existence gate", "has_checkpoint(" in src or "os.path.exists" in src or "path.exists" in lower),
        ("iteration/start_iter state", bool(re.search(r"start_(?:iteration|iter)|['\"]iteration['\"]|global_step|epoch", src))),
        ("resume advances past saved iter", bool(re.search(r"iteration['\"]\]\s*\+\s*1|start_(?:iteration|iter).*range\(", src, re.S))),
        ("model parameters", "params.pkl" in src or "state_dict" in lower or "nnx.state" in src or "flax" in lower),
        ("optimizer state", "opt_state" in lower or "optimizer" in lower),
        ("replay/buffer state", "replay_buffer" in lower and ("to_numpy" in src or "from_numpy" in src or ".npz" in src or "pickle" in lower)),
        ("total step/count state", "total_steps" in lower or "global_step" in lower),
    ]
    missing = [name for name, ok in checks if not ok]
    if missing:
        return ("checkpoint contract is incomplete or unverifiable in inspected source "
                f"({', '.join(os.path.basename(p) for p in paths)}); missing: "
                f"{', '.join(missing)}. Add full-state save/load or pass "
                "--allow-no-resume to explicitly accept restart-from-zero risk.")
    return None

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


def _conflict_path_key(path):
    """Normalize declared local/remote paths for equality checks without probing FS."""
    if not path:
        return ""
    return os.path.normpath(os.path.expanduser(str(path).rstrip("/")))


def _same_declared_path(a, b):
    ka = _conflict_path_key(a)
    kb = _conflict_path_key(b)
    return bool(ka and kb and ka == kb)


def _active_or_unsynced_result_status(task):
    """True while a task can still write/sync into declared result directories."""
    st = task.get("status")
    if st in ("queued", "launching", "running"):
        return True
    if st == "done" and task.get("result_dir") and not task.get("result_synced_at"):
        return True
    return False


def _task_run_identity(task):
    """Identity used only for duplicate-run suppression.

    Broad signatures are allowed: many independent BAPR/ablation jobs may share
    one family-level signature. The key therefore includes the fields that make
    a launch materially different, while intentionally excluding scheduling
    knobs (priority/resources/preferred node) so resubmitting the same run with a
    different estimate does not double-launch it.
    """
    sig = task.get("signature") or ""
    if not sig:
        return None  # empty signatures are auto-adopted / one-offs; legacy exempt
    extra_env = task.get("extra_env") or {}
    if not isinstance(extra_env, dict):
        extra_env = {}
    env_key = json.dumps(extra_env, sort_keys=True, separators=(",", ":"))
    return (
        sig,
        task.get("cmd") or "",
        _conflict_path_key(task.get("cwd")),
        task.get("env_spec") or "none",
        task.get("image") or "",
        env_key,
        _conflict_path_key(task.get("ckpt_dir")),
        task.get("ckpt_glob") or "*",
        task.get("resume_flag") or "",
        _conflict_path_key(task.get("result_dir")),
        _conflict_path_key(task.get("local_result_dir")),
        task.get("slurm_partition") or "",
        task.get("slurm_account") or "",
        task.get("slurm_qos") or "",
    )


def _task_is_descendant_of(task: dict, ancestor_id: str, by_id: dict) -> bool:
    """Return True if task's parent_id chain reaches ancestor_id."""
    if not ancestor_id:
        return False
    seen = set()
    cur = task
    while cur:
        pid = cur.get("parent_id")
        if not pid:
            return False
        if pid == ancestor_id:
            return True
        if pid in seen:
            return False
        seen.add(pid)
        cur = by_id.get(pid)
    return False


def _mark_user_cancelled(task: dict, reason: str = "user cancel") -> None:
    now = time.time()
    task["status"] = "cancelled"
    task["finished_at"] = now
    task["cancelled_at"] = now
    task["cancelled_by_user"] = True
    task["cancel_reason"] = reason
    task.pop("launching_started_at", None)
    task["last_block_reason"] = reason


def _remember_last_placement(task: dict) -> None:
    """Keep enough placement history to audit/recover stale launch artifacts.

    Several recovery paths temporarily clear task["node"] before re-placement.
    If a queued record still carries remote_pids/log_path from an older launch,
    last_node lets us probe or diagnose that older process instead of blindly
    launching the same task id again.
    """
    if task.get("node"):
        task["last_node"] = task.get("node")
    if task.get("gpu_idx") is not None:
        task["last_gpu_idx"] = task.get("gpu_idx")


def _queued_launch_artifacts(task: dict) -> list[str]:
    artifacts = []
    if _task_pids(task):
        artifacts.append("remote_pids")
    if task.get("process_group"):
        artifacts.append("process_group")
    if task.get("started_at") and not task.get("finished_at"):
        artifacts.append("started_at")
    if task.get("log_path") and task.get("started_at"):
        artifacts.append("log_path")
    if task.get("slurm_job_id"):
        artifacts.append("slurm_job_id")
    return artifacts


def _reconcile_queued_launch_artifacts_before_dispatch(task: dict, state: dict) -> Optional[dict]:
    """Prevent same-id relaunch when a queued record still has old launch state.

    A well-formed queued task has no live launch artifacts. Preemption/migration
    explicitly kill and clear remote_pids/process_group/started_at/log_path before
    returning a task to the queue. If those fields are still present, dispatching
    the same record would overwrite PID/log metadata and can run the same task id
    twice. Instead:
      * alive old artifact -> adopt back to running;
      * dead old artifact -> classify terminal and, if crashed, create a retry
        clone with a new task id via _requeue_after_crash;
      * unknown/no-node -> block instead of launching.
    """
    if task.get("status") != "queued":
        return None
    artifacts = _queued_launch_artifacts(task)
    if not artifacts:
        return None

    node = task.get("node") or task.get("last_node")
    if node and not task.get("node"):
        task["node"] = node
    if node:
        task["last_node"] = node
    if task.get("gpu_idx") is not None:
        task["last_gpu_idx"] = task.get("gpu_idx")

    if node and (_task_pids(task) or task.get("slurm_job_id")):
        probe_task = dict(task)
        probe_task["status"] = "running"
        probe_task["node"] = node
        try:
            res = _BACKEND.batch_probe({"tasks": [probe_task]}).get(task.get("id"))
        except Exception:
            res = {"state": "unknown"}
        if not res or res.get("state") == "unknown":
            reason = (
                "queued task still has launch artifacts "
                f"({','.join(artifacts)}) but liveness probe is unknown; "
                "not relaunching same task id"
            )
            task["last_block_reason"] = reason
            return {"type": "blocked", "task_id": task.get("id"), "task": task,
                    "reason": reason}
        if res.get("state") == "alive":
            task["status"] = "running"
            task["node"] = node
            task["alive_pids"] = res.get("alive_pids") or []
            _set_current_usage(task, res.get("vram_mb", 0), res.get("ram_mb", 0), res.get("pcpu", 0.0))
            if res.get("vram_mb", 0) > 0:
                task["peak_vram_mb"] = max(task.get("peak_vram_mb", 0), res.get("vram_mb", 0))
            if res.get("ram_mb", 0) > 0:
                task["peak_ram_mb"] = max(task.get("peak_ram_mb", 0), res.get("ram_mb", 0))
            task["last_block_reason"] = (
                "recovered queued task with live launch artifacts; adopted as running "
                "to avoid same-id relaunch"
            )
            return {"type": "queued_artifact_adopted", "task_id": task.get("id"),
                    "task": task, "reason": task["last_block_reason"]}

    if not node:
        reason = (
            "queued task still has launch artifacts "
            f"({','.join(artifacts)}) but no node/last_node to probe; "
            "not relaunching same task id"
        )
        task["last_block_reason"] = reason
        return {"type": "blocked", "task_id": task.get("id"), "task": task,
                "reason": reason}

    # The old artifact is terminal/dead. Finalize this task id; if it was a
    # crash, _requeue_after_crash will create a fresh queued retry id.
    task["node"] = node
    task["finished_at"] = task.get("finished_at") or time.time()
    task["started_at"] = task.get("started_at") or task["finished_at"]
    task["remote_pids"] = []
    task["alive_pids"] = []
    _set_current_usage(task, 0, 0, 0.0)
    try:
        _release_task_claims_and_intents(task)
    except Exception:
        pass
    diag = _diagnose_terminal(task)
    task["_diagnosis"] = diag
    if diag.get("is_crash") and not task.get("auto_adopted"):
        task["status"] = "failed"
        task["last_block_reason"] = (
            "queued task carried dead launch artifacts; finalized as failed "
            f"instead of relaunching same id: {diag.get('reason', '')}"
        )
        new_id = _requeue_after_crash(task, state)
        if new_id:
            task["requeued_as"] = new_id
        return {"type": "queued_artifact_finalized", "task_id": task.get("id"),
                "task": task, "requeued_as": task.get("requeued_as"),
                "reason": task["last_block_reason"]}
    task["status"] = "done"
    task["last_block_reason"] = (
        "queued task carried dead launch artifacts; finalized as done "
        "instead of relaunching same id"
    )
    return {"type": "queued_artifact_finalized", "task_id": task.get("id"),
            "task": task, "reason": task["last_block_reason"]}


def _cancel_related_queued_retries(state: dict, task: dict, reason: str) -> int:
    """Cancel queued/launching retry duplicates for the same exact run.

    A retry clone represents the same work as its parent. If the operator
    cancels one queued retry, any other queued/launching record with the same
    run identity must not remain dispatchable, otherwise the parent can be
    launched again after the user explicitly stopped the retry.
    """
    key = _task_run_identity(task)
    if not key:
        return 0
    cancelled = 0
    target_id = task.get("id")
    for other in state.get("tasks", []):
        if other is task or other.get("id") == target_id:
            continue
        if other.get("status") not in ("queued", "launching"):
            continue
        if _task_run_identity(other) != key:
            continue
        try:
            _release_task_claims_and_intents(other)
        except Exception:
            pass
        _mark_user_cancelled(
            other,
            f"{reason}; cancelling duplicate retry for same run identity as {target_id}",
        )
        other["node"] = None
        other["gpu_idx"] = None
        other["remote_pids"] = []
        cancelled += 1
    return cancelled


def _has_user_cancelled_retry_descendant(parent: dict, state: dict, parent_key=None) -> bool:
    """True if a retry descendant of parent was cancelled by the operator."""
    parent_id = parent.get("id")
    if not parent_id:
        return False
    key = parent_key if parent_key is not None else _task_run_identity(parent)
    if not key:
        return False
    by_id = {t.get("id"): t for t in state.get("tasks", []) if t.get("id")}
    for t in state.get("tasks", []):
        if t.get("status") != "cancelled":
            continue
        if _task_run_identity(t) != key:
            continue
        if t.get("parent_id") == parent_id or _task_is_descendant_of(t, parent_id, by_id):
            return True
    return False


def reconcile_requeue_lineage_invariants(state: dict) -> int:
    """Repair impossible queued parents that have already spawned a retry.

    Invariant: once a crashed task has `requeued_as=<child>`, the parent is a
    terminal audit record. It must never become dispatchable again. This guard
    catches stale/manual transitions and old queue state before dispatch can
    launch the parent as a duplicate of its retry child.
    """
    by_id = {t.get("id"): t for t in state.get("tasks", []) if t.get("id")}
    changed = 0
    for t in state.get("tasks", []):
        if t.get("status") not in ("queued", "launching"):
            continue
        child_id = t.get("requeued_as")
        if not child_id:
            continue
        child = by_id.get(child_id)
        if not child:
            continue
        try:
            _release_task_claims_and_intents(t)
        except Exception:
            pass
        t["node"] = None
        t["gpu_idx"] = None
        t["remote_pids"] = []
        t.pop("launching_started_at", None)
        if child.get("status") == "cancelled":
            _mark_user_cancelled(
                t,
                f"superseded by retry {child_id}, which was cancelled; parent will not dispatch",
            )
        else:
            t["status"] = "failed"
            t["finished_at"] = t.get("finished_at") or time.time()
            t["last_block_reason"] = (
                f"superseded by retry {child_id}; parent kept terminal to avoid duplicate dispatch"
            )
        changed += 1
    return changed


def _task_runtime_payload(task: dict) -> dict:
    """Fields that make runtime/walltime materially different.

    This is stricter than signature-only so broad family signatures do not
    smear a 5-minute eval onto a multi-hour training run. It intentionally
    excludes resource estimates and placement knobs; those do not change the
    code path being timed.
    """
    extra_env = task.get("extra_env") or {}
    if not isinstance(extra_env, dict):
        extra_env = {}
    return {
        "signature": task.get("signature") or "",
        "project": task.get("project") or "",
        "description": task.get("description") or "",
        "cmd": task.get("cmd") or "",
        "cwd": _conflict_path_key(task.get("cwd")),
        "env_spec": task.get("env_spec") or "none",
        "image": task.get("image") or "",
        "extra_env": extra_env,
    }


def _task_runtime_keys(task: dict) -> list:
    payload = _task_runtime_payload(task)
    # Exact runtime identity is parameter-based, not signature-based. Signatures
    # are often broad family labels or user-added v2 suffixes; letting them into
    # the exact hash would lose otherwise-valid local timing history.
    exact_payload = {
        "cmd": payload.get("cmd") or "",
        "cwd": payload.get("cwd") or "",
        "env_spec": payload.get("env_spec") or "none",
        "image": payload.get("image") or "",
        "extra_env": payload.get("extra_env") or {},
    }
    keys = []
    if exact_payload["cmd"]:
        raw = json.dumps(exact_payload, sort_keys=True, separators=(",", ":"))
        exact = "exact:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()
        keys.append((exact, "exact", payload))
    sig = payload.get("signature")
    if sig:
        keys.append(("sig:" + sig, "signature", payload))
    return keys


def _runtime_history_best(task: dict):
    h = load_runtime_history()
    for key, kind, _payload in _task_runtime_keys(task):
        rec = h.get(key)
        if isinstance(rec, dict) and int(rec.get("total_s") or 0) > 0:
            return rec, key, kind
    return None, None, None


def _runtime_total_history_s(task: dict) -> int:
    rec, _key, _kind = _runtime_history_best(task)
    return int(rec.get("total_s") or 0) if rec else 0


def _runtime_walltime_for_task(task: dict) -> int:
    """Return progress/duration-derived Slurm walltime, or 0 if no runtime history.

    Runtime history is deliberately tighter than legacy duration EWMA: it is
    keyed by exact cmd/cwd/env parameters first, then by signature fallback. The
    exact key ignores signature/project/description so label-only changes do not
    lose otherwise-valid local timing history. The estimate is p80(total
    runtime) × 1.2, with a small 10-minute floor so short evals do not become
    24h jobs and block Slurm backfill.
    """
    total_s = _runtime_total_history_s(task)
    if total_s <= 0:
        return 0
    return int(max(RUNTIME_MIN_WALLTIME_S, total_s * RUNTIME_WALLTIME_MULT))


def _apply_runtime_projection(task: dict, projection: Optional[dict]):
    if not projection:
        return
    total_s = int(projection.get("total_s") or 0)
    if total_s <= 0:
        return
    task["runtime_total_s_est"] = total_s
    task["runtime_est_source"] = projection.get("source") or "progress"
    task["runtime_progress_at"] = int(time.time())
    if projection.get("eta_s") is not None:
        task["runtime_eta_s_est"] = int(projection.get("eta_s") or 0)
    if projection.get("current") is not None:
        task["runtime_current_unit"] = int(projection.get("current") or 0)
    if projection.get("total_units") is not None:
        task["runtime_total_units"] = int(projection.get("total_units") or 0)
    if projection.get("unit_s") is not None:
        task["runtime_unit_s_est"] = float(projection.get("unit_s") or 0.0)


def runtime_history_record(task: dict, duration_s: int = 0):
    """Fold exact-parameter runtime samples into runtime_history.json.

    The sample source is, in priority order:
      1. tqdm/progress projection captured while the task was running
      2. terminal duration_s after clean completion

    We record both an exact parameter key (cmd+cwd+env/image, independent of
    signature/project/description labels) and a signature fallback. Exact wins
    at lookup; signature only helps when the user intentionally uses stable
    signatures for the same parameterized command family.
    """
    if not task:
        return
    total_s = int(task.get("runtime_total_s_est") or 0)
    source = task.get("runtime_est_source") or ""
    if total_s <= 0 and duration_s > 0:
        total_s = int(duration_s)
        source = "duration"
    if total_s <= 0:
        return
    unit_s = float(task.get("runtime_unit_s_est") or 0.0)
    total_units = int(task.get("runtime_total_units") or 0)

    h = load_runtime_history()
    now = int(time.time())
    for key, kind, payload in _task_runtime_keys(task):
        cur = h.get(key)
        if not isinstance(cur, dict):
            cur = {}
        samples = cur.get("total_s_samples") or []
        samples.append(int(total_s))
        samples = samples[-RUNTIME_HISTORY_SAMPLES_PER_KEY:]
        cur["total_s_samples"] = samples
        cur["total_s"] = _percentile(samples, RUNTIME_HISTORY_PERCENTILE)
        cur["walltime_s"] = int(max(RUNTIME_MIN_WALLTIME_S, cur["total_s"] * RUNTIME_WALLTIME_MULT))
        if unit_s > 0:
            us = cur.get("unit_s_samples") or []
            us.append(float(unit_s))
            us = us[-RUNTIME_HISTORY_SAMPLES_PER_KEY:]
            cur["unit_s_samples"] = us
            cur["unit_s"] = float(_percentile([int(x * 1000) for x in us], RUNTIME_HISTORY_PERCENTILE)) / 1000.0
        if total_units > 0:
            cur["total_units"] = total_units
        cur["source"] = source or cur.get("source") or "duration"
        cur["kind"] = kind
        cur["signature"] = payload.get("signature") or ""
        cur["project"] = payload.get("project") or ""
        cur["description"] = payload.get("description") or ""
        cur["cmd"] = (payload.get("cmd") or "")[:500]
        cur["runs"] = int(cur.get("runs") or 0) + 1
        cur["last_seen"] = now
        h[key] = cur
    if len(h) > RUNTIME_HISTORY_MAX_ENTRIES:
        kept = sorted(h.items(), key=lambda kv: -(kv[1].get("last_seen", 0) if isinstance(kv[1], dict) else 0))
        h = dict(kept[:RUNTIME_HISTORY_MAX_ENTRIES])
    save_runtime_history(h)


def _queued_cpu_training_block_reason(task):
    if task.get("status") != "queued":
        return None
    if task.get("auto_adopted") or task.get("adopted"):
        return None
    return _cpu_training_policy_reason(
        _submit_policy_text(task.get("cmd", ""), task.get("cwd", "")),
        task.get("description", ""),
        bool(task.get("allow_cpu_training", False)),
        int(task.get("est_vram_mb") or 0),
    )

def cmd_submit(args):
    raw_submit_cmd = args.cmd
    policy_cmd = _submit_policy_text(args.cmd, args.cwd)
    inferred_checkpoint = _infer_checkpoint_from_submit(args.cmd, args.cwd)
    inferred_ckpt_dir = ""
    inferred_resume_managed = False
    inferred_ckpt_source = ""
    if inferred_checkpoint:
        inferred_ckpt_dir = inferred_checkpoint.get("ckpt_dir") or ""
        inferred_resume_managed = bool(inferred_checkpoint.get("resume_managed_by_cmd"))
        inferred_ckpt_source = inferred_checkpoint.get("source") or ""
    ckpt_dir_was_inferred = False
    if not args.ckpt_dir and inferred_ckpt_dir:
        args.ckpt_dir = inferred_ckpt_dir
        ckpt_dir_was_inferred = True

    # Pre-flight: refuse training-shaped cmd with vram=0 unless the scheduler-level override
    # is present. `--device cpu` in the inner command is not enough: that flag is precisely how
    # a GPU-intended training batch can accidentally land on CPU.
    cpu_training_reason = None
    submit_vram_for_policy = args.vram if args.vram is not None else DEFAULT_VRAM_MB
    cpu_training_reason = _cpu_training_policy_reason(
        policy_cmd, args.description, bool(getattr(args, "allow_cpu_training", False)),
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
            and _task_looks_like_training(policy_cmd, args.description)):
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
    if (_cmd_looks_like_training(policy_cmd)
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
        policy_cmd, args.ckpt_dir, args.resume_flag,
        bool(getattr(args, "allow_no_resume", False))
    )
    if resume_reason:
        print(f"REFUSED: cmd looks like training but resume is not wired up.", file=sys.stderr)
        print(f"  cmd: {args.cmd[:120]}", file=sys.stderr)
        print(f"  reason: {resume_reason}", file=sys.stderr)
        sys.exit(2)
    contract_reason = _checkpoint_contract_reason(
        raw_submit_cmd, args.cwd, args.ckpt_dir, args.resume_flag,
        bool(getattr(args, "allow_no_resume", False))
    )
    if contract_reason:
        print("REFUSED: training checkpoint/resume contract is not verifiable.", file=sys.stderr)
        print(f"  cmd: {raw_submit_cmd[:120]}", file=sys.stderr)
        print(f"  reason: {contract_reason}", file=sys.stderr)
        print("  If this is a short/debug run, pass --allow-no-resume. Otherwise fix the "
              "training code so checkpoint save/load includes iteration, model params, "
              "optimizer state, replay/buffer state, and total step counters.", file=sys.stderr)
        sys.exit(2)
    if args.ckpt_dir and _same_declared_path(args.ckpt_dir, args.cwd):
        print("REFUSED: --ckpt-dir must not equal --cwd.", file=sys.stderr)
        print(f"  cwd:      {args.cwd}", file=sys.stderr)
        print(f"  ckpt-dir: {args.ckpt_dir}", file=sys.stderr)
        print("  reason: launch staging uses rsync --delete for cwd; if ckpts live at cwd root, "
              "remote checkpoint/output files can be deleted during code sync.", file=sys.stderr)
        print("  Put checkpoints in a dedicated subdirectory, e.g. --ckpt-dir <cwd>/checkpoints/<run>.",
              file=sys.stderr)
        sys.exit(2)
    if getattr(args, "result_dir", None) and _same_declared_path(args.result_dir, args.cwd):
        print("REFUSED: --result-dir must not equal --cwd.", file=sys.stderr)
        print(f"  cwd:        {args.cwd}", file=sys.stderr)
        print(f"  result-dir: {args.result_dir}", file=sys.stderr)
        print("  reason: launch staging uses rsync --delete for cwd; result files written at cwd root "
              "are indistinguishable from stale code files.", file=sys.stderr)
        print("  Put results in a dedicated subdirectory, e.g. --result-dir <cwd>/results/<run>.",
              file=sys.stderr)
        sys.exit(2)
    extra_env = _parse_env(args.env)
    with state_lock():
        state = load_state()
        sig = args.signature
        if not getattr(args, "allow_duplicate", False):
            submit_identity = _task_run_identity({
                "signature": sig,
                "cmd": args.cmd,
                "cwd": args.cwd,
                "extra_env": extra_env,
                "env_spec": getattr(args, "env_spec", None) or "none",
                "image": getattr(args, "image", None) or "",
                "ckpt_dir": args.ckpt_dir,
                "ckpt_glob": args.ckpt_glob,
                "resume_flag": args.resume_flag or "",
                "result_dir": getattr(args, "result_dir", None) or None,
                "local_result_dir": getattr(args, "local_result_dir", None) or None,
                "slurm_partition": getattr(args, "slurm_partition", None) or "",
                "slurm_account": getattr(args, "slurm_account", None) or "",
                "slurm_qos": getattr(args, "slurm_qos", None) or "",
            })
            for existing in state["tasks"]:
                if (existing.get("status") in ("queued", "running", "launching")
                        and _task_run_identity(existing) == submit_identity):
                    print(f"DUPLICATE: {existing['id']} ({existing['status']}) has identical run identity")
                    print(f"  signature: {sig}")
                    print(f"  cmd: {args.cmd[:120]}")
                    print(f"  cwd: {args.cwd}")
                    print(f"  pass --allow-duplicate to override")
                    sys.exit(2)
        # ckpt_dir conflict: refuse if any active task already targets the same
        # --ckpt-dir. This must be stronger than run-identity dedup: even two
        # otherwise-distinct tasks corrupt each other if they write one ckpt dir.
        if (args.ckpt_dir
                and not getattr(args, "allow_shared_ckpt_dir", False)):
            for existing in state["tasks"]:
                if existing.get("status") not in ("queued", "running", "launching"): continue
                if not _same_declared_path(existing.get("ckpt_dir"), args.ckpt_dir): continue
                print(f"REFUSED: --ckpt-dir already in use by an active task.",
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
        submit_local_result_dir = ((getattr(args, "local_result_dir", None) or None)
                                   or (getattr(args, "result_dir", None) or None))
        if (submit_local_result_dir
                and not getattr(args, "allow_shared_result_dir", False)):
            for existing in state["tasks"]:
                if not _active_or_unsynced_result_status(existing): continue
                existing_dst = existing.get("local_result_dir") or existing.get("result_dir")
                if not _same_declared_path(existing_dst, submit_local_result_dir): continue
                print("REFUSED: --local-result-dir destination already in use by an unfinished/unsynced task.",
                      file=sys.stderr)
                print(f"  conflicting task: {existing['id']} ({existing.get('status')}) sig={existing.get('signature','')!r}",
                      file=sys.stderr)
                print(f"  destination:      {submit_local_result_dir}", file=sys.stderr)
                print("  reason: result sync rsyncs remote result_dir contents into this directory; "
                      "two tasks sharing it can merge or overwrite files.", file=sys.stderr)
                print("  Use a per-task subdirectory, or pass --allow-shared-result-dir if the "
                      "layout is intentionally collision-free.", file=sys.stderr)
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
            "ckpt_dir_inferred": bool(ckpt_dir_was_inferred),
            "ckpt_inferred_source": inferred_ckpt_source if ckpt_dir_was_inferred else "",
            "ckpt_glob": args.ckpt_glob,
            "resume_flag": args.resume_flag or "",
            "resume_managed_by_cmd": bool(inferred_resume_managed and not args.resume_flag),
            # Phase 3.5: auto-pull experiment results back to local on
            # task completion. Opt-in (no field set → no sync). Mirror
            # the remote path locally unless --local-result-dir overrides.
            # Intentionally separate from ckpt_dir: ckpts stay on the
            # node where they were produced (migration / eval flows
            # already pull them on demand). result_dir should point to
            # logs / final saved models / metrics — small files the
            # user wants on their own box.
            "result_dir": getattr(args, "result_dir", None) or None,
            "local_result_dir": getattr(args, "local_result_dir", None) or None,
            "result_synced_at": None,
            "result_sync_error": None,
            "result_sync_attempts": 0,
            # Phase 3.4.11 P2: claim marker for concurrent-rsync prevention.
            # Set by _sync_completed_results_outside_lock under state_lock
            # before rsync; cleared on commit phase. Stale claims older
            # than RESULT_SYNC_TIMEOUT_S + grace are reclaimable.
            "result_syncing_at": None,
            "extra_env": extra_env,
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
            "peak_ram_mb": 0,
            "current_vram_mb": 0,
            "current_ram_mb": 0,
            "current_pcpu": 0.0,
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

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Phase 3.0.23 P2 fix: env keys reserved by scheduleurm. Users must not be able
# to override these via --env because they're set by the launch path itself
# (LocalBackend / SlurmBackend) to honor scheduling decisions like GPU pinning.
# Letting --env override `CUDA_VISIBLE_DEVICES` would let a user-submitted task
# read a different GPU than the one it was scheduled onto, breaking VRAM
# accounting + the 1/3 packing rule + everything downstream.
_RESERVED_ENV_KEYS = frozenset({
    "CUDA_VISIBLE_DEVICES",
    # Phase 3.0.28: SCHEDULEURM_TASK_ID is injected by LocalBackend.launch so
    # the WAL orphan-recovery scanner can find a launched-but-not-yet-saved
    # process via /proc/<pid>/environ. User overriding this would defeat the
    # invariant ("no double-launch after scheduler crash mid-launch").
    "SCHEDULEURM_TASK_ID",
})


def _safe_extra_env_items(extra_env):
    """Yield (k, v) from extra_env, skipping any key that fails 3.0.23 validation
    (invalid POSIX shape OR reserved). Defensive layer at launch sites to
    protect against tasks persisted in state.json BEFORE 3.0.23 (which may
    have invalid keys baked in) — submit-time validation is the primary
    gate, this is belt-and-suspenders."""
    for k, v in (extra_env or {}).items():
        if not _ENV_KEY_RE.match(k):
            continue
        if k in _RESERVED_ENV_KEYS:
            continue
        yield (k, v)


def _parse_env(pairs):
    if not pairs: return {}
    out = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--env expects KEY=VALUE, got {p!r}")
        k, v = p.split("=", 1)
        # Phase 3.0.23 P2 fix: validate the key shape AND reject reserved names.
        # Pre-fix: any string was accepted, which let users smuggle in keys with
        # spaces / quotes (broken `export` line in launch shell) or override
        # reserved keys like CUDA_VISIBLE_DEVICES (broke GPU pinning silently).
        if not _ENV_KEY_RE.match(k):
            raise SystemExit(
                f"--env key {k!r} is not a valid POSIX env var name "
                f"(must match ^[A-Za-z_][A-Za-z0-9_]*$); refusing to launch "
                f"with a key that would break the export shell line.")
        if k in _RESERVED_ENV_KEYS:
            raise SystemExit(
                f"--env key {k!r} is reserved by scheduleurm (set by the "
                f"launch path to honor GPU pinning / scheduling decisions); "
                f"user override would break VRAM accounting + the 1/3 "
                f"packing rule.")
        out[k] = v
    return out

STARTUP_FLOOR_MB = 500   # minimum reservation per running task with peak=0 (still loading model)
EVICT_TASK_MIN_AGE_S = 180  # don't evict a task within this many seconds of launch (give JAX/torch model loading + warmup a chance — JAX in particular spikes util to 100% during first iter compile)


def _slurm_max_pending_for_node(node_name: str, bucket: str = None) -> int:
    """Per-node override > bucket global default.

    `bucket` is "cpu" or "gpu". Legacy callers/config still work:
    NODES[name]["max_slurm_pending"] applies to both buckets, and the
    one-argument call returns the GPU/legacy cap.
    """
    info = NODES.get(node_name, {}) or {}
    if bucket in ("cpu", "gpu"):
        specific = info.get(f"max_slurm_pending_{bucket}")
        if specific is not None:
            return int(specific)
    legacy = info.get("max_slurm_pending")
    if legacy is not None:
        return int(legacy)
    if bucket == "cpu":
        return SLURM_MAX_PENDING_CPU_PER_NODE
    if bucket == "gpu":
        return SLURM_MAX_PENDING_GPU_PER_NODE
    return SLURM_MAX_PENDING_PER_NODE


# Slurm states that count as "still pending in slurm's queue" for throttle accounting.
# RUNNING / COMPLETING / done aren't pending — they're consuming a real slot. None /
# empty string means "just-submitted, watcher hasn't probed slurm_state yet" — treat as
# pending until proven otherwise (next watcher cycle clears the ambiguity).
_SLURM_PENDING_LIKE = frozenset({None, "", "PENDING", "CONFIGURING", "REQUEUED", "SUSPENDED"})


def _count_slurm_pending_per_node(state: dict) -> dict:
    """Phase 2.16 + Phase 3.4.13: count OUR slurm-managed tasks that are
    PENDING (or just-submitted with no slurm_state probed yet) per node,
    SPLIT by resource type. Used by pick_placement to throttle dispatch
    independently for CPU-only and GPU-using tasks: a CPU-only eval task
    shouldn't be blocked by a pending GPU training job, since slurm itself
    has no resource conflict between them (different gres requests).

    Returns a dict mapping node_name → {"cpu": int, "gpu": int}. Both
    keys always present; missing nodes default to {"cpu": 0, "gpu": 0}.
    Resource type is inferred from `est_vram_mb`: > 0 → GPU, 0 → CPU.

    The 'just-submitted' case (slurm_state=None right after sbatch) is
    treated as pending; one watcher cycle (60s default) refreshes the
    state to PENDING/RUNNING/etc.
    """
    counts: dict = {}
    for t in state.get("tasks", []):
        if t.get("status") != "running":
            continue
        if not _is_slurm_managed(t):
            continue
        node = t.get("node")
        if not node:
            continue
        if t.get("slurm_state") not in _SLURM_PENDING_LIKE:
            continue
        bucket = _slurm_pending_bucket_for_task(t)
        per_node = counts.setdefault(node, {"cpu": 0, "gpu": 0})
        per_node[bucket] += 1
    return counts


def _slurm_pending_bucket_for_task(task: dict) -> str:
    """Return 'cpu' or 'gpu' to indicate which throttle pool this task
    belongs to. Mirrors the bucket logic in _count_slurm_pending_per_node
    so dispatch and accounting agree on the classification."""
    est = int(task.get("est_vram_mb", DEFAULT_VRAM_MB) or 0)
    return "gpu" if est > 0 else "cpu"


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

MIGRATION_COOLDOWN_S = int(os.environ.get("SCHEDULEURM_MIGRATION_COOLDOWN_S", "1800"))
# Phase 3.0.12 P3 fix: minimum seconds between two migrations of the same task.
# Without this, oscillating loads (A becomes heavy → B; then B becomes heavy → A;
# then A again …) can ping-pong the same task repeatedly, costing one rsync per
# dispatch cycle. 30 min is long enough that real load shifts have settled before
# the next migration is considered, short enough that genuine imbalance still gets
# rebalanced within an hour.

MIGRATION_MIN_SOURCE_LOAD_S = int(os.environ.get("SCHEDULEURM_MIGRATION_MIN_SOURCE_LOAD_S", "600"))
# Phase 3.0.14 P4 fix: don't migrate when the "overloaded" node only has a few
# seconds of work. The pre-fix LOAD_RATIO=2x check was satisfied by trivial
# imbalances (target=0s, source=2s → ratio=2.0 → migrate a 600s task to save 2s).
# A real "overloaded" node should hold at least 10 minutes of pinned ETA. Below
# that, the rsync staging cost exceeds the saving — let it drain naturally.

MIGRATION_MAX_CWD_SIZE_MB = int(os.environ.get("SCHEDULEURM_MIGRATION_MAX_CWD_SIZE_MB", "1024"))
# Phase 3.0.14 P4 fix: cap the size of cwd that we'll rsync during migration
# staging. MIGRATION_MAX_CKPT_SIZE_MB only bounded ckpt; cwd was unbounded so a
# monorepo cwd (5GB+) could blow through the 600s rsync timeout and starve the
# staging path. Default 1GB excludes .git/__pycache__/*.pyc (mirroring rsync's
# excludes), which catches typical code dirs (≤100MB) with comfortable headroom.

LAUNCH_MAX_CWD_SIZE_MB = int(os.environ.get("SCHEDULEURM_LAUNCH_MAX_CWD_SIZE_MB", "2048"))
# Phase 3.4.10 P1 fix: cap for first-launch cwd staging (separate from
# MIGRATION_MAX_CWD_SIZE_MB because semantics differ). When a queued task
# is about to launch on a non-local target and the local cwd > this cap,
# we PIN it to local instead of risking a slow / starved rsync. User-spec'd
# default 2GB matches the rule "依赖 > 2GB 就坚持本地跑". Cap is applied
# pre-rsync via local du; if the local working tree is huge (large data /
# checkpoints inlined), the dispatch reroutes back to local rather than
# attempting a multi-minute transfer that would starve other dispatches.


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

STAGING_FAIL_COOLDOWN_S = int(os.environ.get("SCHEDULEURM_STAGING_FAIL_COOLDOWN_S", "3600"))
# Phase 3.0.19 P3 fix: per-(task,target) cooldown after a failed staging attempt.
# Pre-fix, _identify_migration_candidates always returned the same first
# max_candidates=2 entries (sorted by eta). If both had permanent failures
# (ckpt > cap, env missing on target, remote→remote refuse), candidates 3+
# never got a chance — permanent starvation. Now: a failed staging attempt
# tags the (task_id, target) pair for STAGING_FAIL_COOLDOWN_S seconds; the
# identify pass skips tagged pairs so subsequent candidates are exposed.
# Recovery is automatic: cooldown expires, candidate becomes eligible again.
# 1 hour balances quick recovery for fixable failures (env-missing the user
# patches) vs. avoiding rsync churn for permanent ones (oversized ckpt).

STAGING_TTL_S = int(os.environ.get("SCHEDULEURM_STAGING_TTL_S", "600"))
# Phase 3.0.17 P2 fix: TTL on staging caches. Pre-fix _STAGING_CACHE was a plain
# set — once a (src,tgt,path) was added, it survived until process restart, so a
# user editing code in cwd OR placing a fresher ckpt would silently get
# overridden by stale staged content on the next migration. Now entries are
# tagged with their staging timestamp; lookups treat anything older than
# STAGING_TTL_S as a miss and force a re-rsync. rsync's delta algorithm keeps
# re-rsync cheap (~1s for unchanged content), so the default 10 min TTL is the
# right tradeoff: short enough to pick up content edits within a session,
# long enough to avoid needless ssh round-trips.

# Cache for staged paths so we don't redundantly rsync. dict key: (source_node,
# target_node, path); value: timestamp of last successful rsync (TTL-checked).
# Reset on watcher restart (in-memory only).
_STAGING_CACHE: dict = {}


def _staging_cache_hit(key) -> bool:
    """Return True iff the cache has a non-stale entry for `key`. Removes stale
    entries opportunistically so the next call falls through to re-rsync."""
    ts = _STAGING_CACHE.get(key)
    if ts is None:
        return False
    if (time.time() - ts) > STAGING_TTL_S:
        _STAGING_CACHE.pop(key, None)
        return False
    return True


# Phase 3.0.19: per-(task_id, target) failure tag with TTL. Populated when
# _stage_migration_candidates_outside_lock observes a staging failure;
# consulted by _identify_migration_candidates so doomed candidates don't
# permanently block later eligible ones from getting their staging slot.
_STAGING_FAILED: dict = {}
_STAGING_FAILED_MAX = 200  # bound memory; LRU evicts oldest when full


def _staging_recently_failed(task_id, target) -> bool:
    """Return True iff (task_id, target) had a recent staging failure within
    STAGING_FAIL_COOLDOWN_S. Pops expired entries opportunistically."""
    key = (task_id, target)
    ts = _STAGING_FAILED.get(key)
    if ts is None:
        return False
    if (time.time() - ts) > STAGING_FAIL_COOLDOWN_S:
        _STAGING_FAILED.pop(key, None)
        return False
    return True


def _record_staging_failure(task_id, target):
    """Mark (task_id, target) as recently-failed. LRU evicts oldest 25% when full."""
    if len(_STAGING_FAILED) >= _STAGING_FAILED_MAX:
        sorted_keys = sorted(_STAGING_FAILED.items(), key=lambda kv: kv[1])
        for k, _ in sorted_keys[:_STAGING_FAILED_MAX // 4]:
            _STAGING_FAILED.pop(k, None)
    _STAGING_FAILED[(task_id, target)] = time.time()


# Phase 3.0.27: conda preload sync-success markers with TTL. Successful
# push_conda_env(node, env_path) records a timestamp here; failed sync clears
# it. Launch (_maybe_wrap_docker) refuses to wrap an explicit conda task if
# the latest sync attempt to its target node didn't succeed — stops the
# silent "stale remote env runs" failure mode that the local-path check
# alone (3.0.22) couldn't catch (sync may fail even when local path is fine).
# Reuses STAGING_TTL_S for the staleness window; same semantic.
_CONDA_SYNC_OK: dict = {}


def _record_conda_sync_ok(node, env_path):
    _CONDA_SYNC_OK[(node, env_path)] = time.time()


def _record_conda_sync_failed(node, env_path):
    """Drop any prior success marker so the next launch fails fast. Failed
    sync is ALWAYS treated as authoritative — it overrides any earlier
    success because the remote env may now be stale relative to local."""
    _CONDA_SYNC_OK.pop((node, env_path), None)


def _conda_sync_ok(node, env_path) -> bool:
    """True iff the latest preload of (node, env_path) succeeded within
    STAGING_TTL_S. Pops expired entries opportunistically."""
    ts = _CONDA_SYNC_OK.get((node, env_path))
    if ts is None:
        return False
    if (time.time() - ts) > STAGING_TTL_S:
        _CONDA_SYNC_OK.pop((node, env_path), None)
        return False
    return True


# Phase 3.4.11 P1 fix: cap-exceeded cache. Populated by _stage_cwd_for_launch
# when local cwd > LAUNCH_MAX_CWD_SIZE_MB; consulted by _stage_cwd_check (the
# fast inside-lock probe) so dispatch can pin the task to local without
# re-running `du`. TTL'd via STAGING_TTL_S so a user shrinking cwd recovers
# automatically on the next outside-lock pre-staging pass.
_STAGING_CAP_EXCEEDED: dict = {}

# Phase 3.4.12 P1-2 fix: rsync-failure cache. Populated by
# _stage_launch_candidates_outside_lock when rsync to a (target, cwd) fails
# (transport / disk full / permission). Consulted by _stage_cwd_check so
# dispatch can route the task through the existing launch_failed_nodes
# pipeline (count + retry on different node + heal escalation after MAX),
# rather than looping forever in "needs_stage". TTL via STAGING_FAIL_COOLDOWN_S
# so transient ssh blips and user-fixable issues recover automatically.
# Value: (timestamp, last_error_msg) so dispatch can surface the reason.
_STAGING_FAILS: dict = {}


def _stage_cwd_check(target_node: str, cwd: str):
    """Phase 3.4.11 P1 fix + 3.4.12 P1-2: FAST inside-lock probe of staging state.
    Returns one of:
      "ready"        — target == local, OR _STAGING_CACHE has fresh entry
      "cap_exceeded" — _STAGING_CAP_EXCEEDED has fresh entry
      "stage_failed" — _STAGING_FAILS has fresh entry (rsync failed last try)
      "needs_stage"  — cache miss; dispatch must defer this cycle and let
                       the next outside-lock pass run _stage_cwd_for_launch

    NEVER does ssh / rsync / du. Constant-time dict lookup + TTL check.
    Caller (inside _do_dispatch, under state_lock) routes:
      "ready"        → proceed with launch
      "cap_exceeded" → set require_node=local + revert queued
      "stage_failed" → bump launch_fail_count + record launch_failed_nodes
                       on this target, then revert queued so pick_placement
                       avoids it next cycle
      "needs_stage"  → emit launch_stage_deferred event, leave queued
    """
    if NODES.get(target_node, {}).get("host") is None:
        return "ready"  # local target, nothing to sync
    cwd_key = ("local", target_node, cwd)
    if _staging_cache_hit(cwd_key):
        return "ready"
    ts = _STAGING_CAP_EXCEEDED.get(cwd_key)
    if ts is not None:
        if (time.time() - ts) > STAGING_TTL_S:
            _STAGING_CAP_EXCEEDED.pop(cwd_key, None)
        else:
            return "cap_exceeded"
    fail = _STAGING_FAILS.get(cwd_key)
    if fail is not None:
        fail_ts = fail[0] if isinstance(fail, tuple) else fail
        if (time.time() - fail_ts) > STAGING_FAIL_COOLDOWN_S:
            _STAGING_FAILS.pop(cwd_key, None)
        else:
            return "stage_failed"
    return "needs_stage"


def _stage_failure_reason(target_node: str, cwd: str) -> str:
    """Return the cached rsync failure message for (target, cwd), or empty
    string if not in _STAGING_FAILS. Used by dispatch to surface the
    underlying error in last_block_reason / launch_failed_nodes."""
    cwd_key = ("local", target_node, cwd)
    fail = _STAGING_FAILS.get(cwd_key)
    if fail is None:
        return ""
    return fail[1] if isinstance(fail, tuple) and len(fail) > 1 else "unknown"


def _stage_cwd_for_launch(task: dict, target_node: str,
                          extra_excludes: list = None) -> tuple:
    """Phase 3.4.10 P1 fix: pre-launch sync of cwd from LOCAL (source-of-truth)
    to a non-local target_node. Mirrors `_stage_for_migration`'s rsync semantics
    but with source pinned to local — `local working tree` is authoritative,
    not whatever happens to live on the chosen target.

    extra_excludes (Phase 3.4.12 P1 fix): additional --exclude paths
    passed by the outside-lock orchestrator to protect ANY task's
    ckpt_dir / result_dir that lives under this cwd. Without dynamic
    protection, --delete would wipe `runs/exp1/`, `outputs/seed1/`,
    `checkpoints/`, `<arbitrary>/` — only the hard-coded `results/ logs/
    experiment_output/ archive*/` excludes survived. Caller computes
    these relative to cwd before invoking.

    Pre-fix gap: dispatch's first-launch path only did `test -d cwd` on target
    (LocalBackend.launch line ~2918). If target had a stale clone, an old
    snapshot, or even a same-named stub from an unrelated project, the test
    passed and launch proceeded with WRONG code → ENV_MISSING failure → cwd
    blocklisted forever for that node. Migration's `_stage_for_migration`
    only triggers when a task moves between nodes, never on first launch.
    Whole new nodes therefore stayed permanently un-synced.

    Returns:
      (True, msg)              — target ready (synced or already cached)
      (False, "CAP_EXCEEDED:") — cwd > LAUNCH_MAX_CWD_SIZE_MB; caller pins local
      (False, msg)             — rsync transport failure; caller treats as launch fail

    Skip rules:
      - target == local       — same machine, no-op
      - cwd not on local      — local can't be source; bail (caller treats as fail)
      - cache hit within TTL  — skip rsync (delta would be cheap anyway, but
                                cache lookup is free)
    """
    cwd = task.get("cwd")
    if not cwd:
        return (False, "no cwd on task")
    # Skip if target is local (source==target).
    if NODES.get(target_node, {}).get("host") is None:
        return (True, "target is local; nothing to sync")
    # Source must be local; if cwd doesn't exist locally we can't be the
    # source-of-truth, so don't try to rsync from nothing.
    if not Path(cwd).exists():
        return (False,
                f"cwd {cwd} does not exist on local; can't seed target {target_node}")

    cwd_key = ("local", target_node, cwd)
    if _staging_cache_hit(cwd_key):
        return (True, "cache hit (already synced within TTL)")

    # Probe local cwd size with the same excludes the rsync below uses,
    # otherwise the cap check would over-count (e.g. .git can be 500MB+
    # while the actual code transfer is <50MB).
    cwd_size_mb = 0
    try:
        du_args = ["du", "-sm",
                   "--exclude=.git", "--exclude=__pycache__", "--exclude=*.pyc",
                   "--exclude=results", "--exclude=results_*",
                   "--exclude=logs", "--exclude=logs_*",
                   "--exclude=experiment_output",
                   "--exclude=archive*", "--exclude=*.tar.gz"]
        # Phase 3.4.12 P1: extend du excludes to match the rsync excludes
        # so the cap check doesn't over-count bytes the rsync would skip
        # anyway (otherwise a 5GB results/ outside the rsync set would
        # falsely trigger CAP_EXCEEDED).
        if extra_excludes:
            for ex in extra_excludes:
                if ex:
                    # du --exclude doesn't want trailing slash like rsync does
                    du_args.append(f"--exclude={ex.rstrip('/')}")
        du_args.append(cwd)
        r = subprocess.run(
            du_args,
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            head = (r.stdout or "").strip().split()
            if head and head[0].isdigit():
                cwd_size_mb = int(head[0])
    except Exception:
        cwd_size_mb = 0

    if cwd_size_mb > LAUNCH_MAX_CWD_SIZE_MB:
        # Phase 3.4.11 P1 fix: cache the cap-exceeded determination so the
        # inside-lock fast probe can return without re-running du. TTL'd
        # via STAGING_TTL_S; if user shrinks cwd, TTL expiry forces a
        # fresh check on the next outside-lock pre-staging cycle.
        _STAGING_CAP_EXCEEDED[cwd_key] = time.time()
        return (False,
                f"CAP_EXCEEDED: cwd {cwd} is {cwd_size_mb}MB > "
                f"{LAUNCH_MAX_CWD_SIZE_MB}MB cap; pin to local instead of "
                f"transferring to {target_node}")

    tgt_host = NODES.get(target_node, {}).get("host")
    if not tgt_host:
        return (False, f"target node {target_node} has no host")

    # Ensure target dir exists (mkdir -p is idempotent; cheap).
    try:
        run_on(target_node, f"mkdir -p {shlex.quote(cwd)}", timeout=10, check=False)
    except Exception:
        pass  # if mkdir fails the rsync below will too — surface the rsync error

    src_path = cwd.rstrip("/") + "/"
    dst_path = f"{tgt_host}:{cwd.rstrip('/')}/"
    # Phase 3.4.11 P2 fix: --delete enforces "local is source-of-truth"
    # semantics. Without it, files that were renamed/deleted on local
    # remain on remote, so a stale code path could still execute (e.g.
    # an old `train_v1.py` that local replaced with `train_v2.py`).
    # The exclude list applies to BOTH transfer and delete passes, so
    # excluded paths on remote are NEVER removed — they live outside
    # the rsync set.
    rsync_args = ["rsync", "-az", "--partial", "--delete",
                  "--exclude=.git/", "--exclude=__pycache__/", "--exclude=*.pyc",
                  "--exclude=results/", "--exclude=results_*/",
                  "--exclude=logs/", "--exclude=logs_*/",
                  "--exclude=experiment_output/",
                  "--exclude=archive*/", "--exclude=*.tar.gz"]
    # Phase 3.4.12 P1 fix: dynamic --exclude for any task's ckpt_dir /
    # result_dir that lands under this cwd. Caller computes relative
    # paths and passes them here so we never wipe a sibling task's
    # output dir just because the dir name doesn't match a hard-coded
    # exclude. Each entry is appended verbatim with the `--exclude=`
    # prefix; trailing slash on dirs ensures rsync treats them as dirs.
    if extra_excludes:
        for ex in extra_excludes:
            if ex:
                rsync_args.extend([f"--exclude={ex}"])
    rsync_args.extend([src_path, dst_path])
    try:
        r = subprocess.run(
            rsync_args,
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0:
            return (False,
                    f"rsync rc={r.returncode}: {(r.stderr or '').strip()[:200]}")
    except subprocess.TimeoutExpired:
        return (False, "rsync timeout (>600s)")
    except Exception as e:
        return (False, f"rsync exception: {str(e)[:200]}")

    _STAGING_CACHE[cwd_key] = time.time()
    # Cap was previously cached but cwd has been re-checked successfully —
    # clear any stale CAP_EXCEEDED entry so stage_cwd_check returns "ready".
    _STAGING_CAP_EXCEEDED.pop(cwd_key, None)
    # Same for any prior rsync failure record — success supersedes.
    _STAGING_FAILS.pop(cwd_key, None)
    return (True, f"synced ({cwd_size_mb}MB)")


def _stage_launch_candidates_outside_lock():
    """Phase 3.4.11 P1 fix: pre-launch cwd staging OUTSIDE the global state_lock.

    Pre-fix: _stage_cwd_for_launch (added in Phase 3.4.10) was called
    synchronously from inside _do_dispatch, which itself runs under
    state_lock from cmd_dispatch / _watch_iteration. A 600s rsync would
    therefore block submit/cancel/status/watcher for up to 10 min — exactly
    the foot-gun migration staging avoided in Phase 3.0.5.

    Now: this helper runs BEFORE the main lock (mirrors
    _stage_migration_candidates_outside_lock). It scans queued tasks under
    a SHORT lock, releases, runs rsync per (target, cwd) outside the lock,
    populates _STAGING_CACHE / _STAGING_CAP_EXCEEDED. _do_dispatch's
    launch site uses _stage_cwd_check (constant-time lookup), never rsync.

    Target prediction: for each queued task, the candidate target nodes are
      - the explicit pin (require_node or preferred_node) if set, OR
      - every non-local node in NODES (so any pick_placement choice gets
        cache hit)
    rsync to already-cached (target, cwd) pairs short-circuits via
    `_staging_cache_hit`, so the per-cycle cost is bounded:
      worst: O(queued_tasks × non_local_nodes) ssh round-trips on cold start
      typical: O(0) once caches are warm (10min TTL)
    """
    try:
        with state_lock():
            state = load_state()
        # Phase 3.4.12 P1: build cwd → set of relative paths under cwd that
        # belong to ANY task's ckpt_dir / result_dir / local_result_dir.
        # These paths must NOT be deleted by --delete, even if they don't
        # exist locally yet. Scan ALL tasks regardless of status — even a
        # `failed` task may have left a recoverable ckpt the user wants.
        protected_under_cwd: dict = {}  # cwd_str -> set[rel_path]
        for t in state.get("tasks", []):
            t_cwd = t.get("cwd")
            if not t_cwd:
                continue
            t_cwd_norm = t_cwd.rstrip("/")
            for field in ("ckpt_dir", "result_dir", "local_result_dir"):
                d = t.get(field)
                if not d:
                    continue
                d_norm = str(d).rstrip("/")
                # Only protect if the path is genuinely under t_cwd.
                # commonpath handles trailing slashes / relative bits.
                try:
                    if (d_norm == t_cwd_norm
                            or os.path.commonpath([d_norm, t_cwd_norm]) == t_cwd_norm):
                        rel = os.path.relpath(d_norm, t_cwd_norm)
                        if rel and rel != "." and not rel.startswith(".."):
                            # rsync expects trailing slash for dirs to
                            # match dir-or-content semantics consistently.
                            protected_under_cwd.setdefault(t_cwd_norm, set()).add(
                                rel.rstrip("/") + "/")
                except (ValueError, OSError):
                    # commonpath raises on different drives etc; just skip.
                    pass

        # Phase 3.4.12 P2-1 fix: split require vs. preferred. require_node is
        # a hard pin → single target. preferred_node is SOFT and pick_placement
        # falls back to other nodes; if we only stage to preferred, dispatch
        # may end up choosing a different node and stay in needs_stage forever.
        # So preferred → stage to ALL non-local nodes (preferred + fallbacks).
        candidates: set = set()
        for t in state.get("tasks", []):
            if t.get("status") != "queued":
                continue
            cwd = t.get("cwd")
            if not cwd:
                continue
            require = t.get("require_node")
            if require:
                tgts = [require]
            else:
                # No hard pin (preferred OR nothing) → stage to every node
                # pick_placement might pick. Filtered to non-local below.
                tgts = list(NODES.keys())
            for tn in tgts:
                if NODES.get(tn, {}).get("host") is None:
                    continue
                cwd_key = ("local", tn, cwd)
                if _staging_cache_hit(cwd_key):
                    continue
                cap_ts = _STAGING_CAP_EXCEEDED.get(cwd_key)
                if cap_ts is not None and (time.time() - cap_ts) <= STAGING_TTL_S:
                    continue
                candidates.add((tn, cwd))
    except Exception as e:
        try:
            notify("launch_staging_snapshot_error", {"error": str(e)[:200]},
                   feishu_enabled=False)
        except Exception:
            pass
        return

    if not candidates:
        return

    for tn, cwd in candidates:
        cwd_norm = cwd.rstrip("/")
        extra = sorted(protected_under_cwd.get(cwd_norm, set()))
        try:
            ok, msg = _stage_cwd_for_launch({"cwd": cwd}, tn,
                                            extra_excludes=extra)
            if not ok:
                if msg.startswith("CAP_EXCEEDED:"):
                    # Cap is a routing decision (handled by dispatch via
                    # _STAGING_CAP_EXCEEDED cache); not a transport fail.
                    continue
                # Phase 3.4.12 P1-2 fix: rsync transport failure must
                # eventually escalate, not loop forever in needs_stage.
                # Stamp _STAGING_FAILS so _stage_cwd_check returns
                # "stage_failed" → dispatch routes to launch_failed_nodes
                # / launch_fail_count. Cooldown via STAGING_FAIL_COOLDOWN_S
                # so transient ssh blips recover automatically.
                _STAGING_FAILS[("local", tn, cwd)] = (time.time(), msg[:200])
                try:
                    notify("launch_staging_failed",
                           {"target": tn, "cwd": cwd, "reason": msg[:200]},
                           feishu_enabled=False)
                except Exception:
                    pass
        except Exception as e:
            _STAGING_FAILS[("local", tn, cwd)] = (
                time.time(), f"exception: {str(e)[:150]}")
            try:
                notify("launch_staging_exception",
                       {"target": tn, "cwd": cwd, "error": str(e)[:200]},
                       feishu_enabled=False)
            except Exception:
                pass


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
    # Phase 3.0.20 P1 fix: on cache miss, ALWAYS rsync — even if `test -d cwd`
    # says the dir already exists on target. Pre-fix: cache miss + dir present
    # → skip rsync entirely + populate the cache. If target had a stale clone
    # of the same path (older user setup, sibling task, manual ssh), the
    # migrated task ran old code with no warning. rsync's delta algorithm
    # makes "already in sync" cheap (~1s) so always running it is the safer
    # default than trusting `test -d` as a freshness proxy.
    cwd_key = (source_node, target_node, cwd)
    if not _staging_cache_hit(cwd_key):
        # Source must be NODES-keyed; both sides resolve to local-or-host.
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
                    f"cwd rsync remote→remote not yet supported "
                    f"(src={src_host}, tgt={tgt_host}); user must sync code "
                    f"manually OR via shared NFS")
        # Phase 3.0.14 P4 fix: cap cwd size before rsync, mirror the ckpt cap.
        # Excludes match the rsync excludes below so the size is the actual
        # transfer size — unbounded cwd was a 600s timeout / starvation risk.
        cwd_size_mb = 0
        if source_node:
            try:
                rc_du, out_du, _ = run_on(
                    source_node,
                    f"du -sm --exclude=.git --exclude=__pycache__ "
                    f"--exclude='*.pyc' {shlex.quote(cwd)} 2>/dev/null | "
                    f"awk '{{print $1}}'",
                    timeout=15, check=False,
                )
                if rc_du == 0 and out_du.strip().isdigit():
                    cwd_size_mb = int(out_du.strip())
            except Exception:
                cwd_size_mb = 0
        if cwd_size_mb > MIGRATION_MAX_CWD_SIZE_MB:
            return (False,
                    f"cwd {cwd} is {cwd_size_mb}MB > max "
                    f"{MIGRATION_MAX_CWD_SIZE_MB}MB; migration aborted "
                    f"(rsync would risk hitting the 600s timeout)")
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
        _STAGING_CACHE[cwd_key] = time.time()

    # Step 3: ckpt_dir (if set)
    ckpt_dir = task.get("ckpt_dir")
    if ckpt_dir:
        # Phase 3.0.16 P1 fix: ckpt staging must be fail-closed when source-side
        # state is unknown. Pre-fix: a `du -sm ckpt_dir` failure (ssh blip,
        # permission, awk parse) silently set size_mb=0; the rsync gate
        # `if size_mb > 0` then skipped the actual transfer and the function
        # still returned success. Resume task launched on target with no ckpt →
        # silent step-0 restart. Same blast-radius shape as 3.0.6 / 3.0.15.
        #
        # Now: explicit existence test on source first.
        #   - source has no source_node OR ckpt_dir absent on source → no ckpt
        #     to stage (legitimate first-launch case); return success-so-far,
        #     skip rsync.
        #   - ckpt_dir exists on source → size MUST be determinable.
        #     du failure = unknown size = fail-closed (don't migrate without
        #     transferring the ckpt we know is there).
        src_present = False
        if source_node:
            try:
                rc_test, _, _ = run_on(
                    source_node, f"test -d {shlex.quote(ckpt_dir)}",
                    timeout=5, check=False,
                )
                src_present = (rc_test == 0)
            except Exception as e:
                return (False,
                        f"ckpt_dir {ckpt_dir} reachability check on {source_node} "
                        f"failed: {str(e)[:120]}; fail-closed (would otherwise "
                        f"migrate without rsyncing the ckpt → silent step-0 restart)")

        if not src_present:
            # No ckpt to stage. Skip directly to env probe — equivalent to a
            # first-launch task that hasn't created its ckpt yet.
            ckpt_dir = None  # signal "nothing to do" to the rest of step 3
        else:
            size_mb = -1  # sentinel: "unknown"
            try:
                rc, out, _ = run_on(source_node,
                                    f"du -sm {shlex.quote(ckpt_dir)} 2>/dev/null | "
                                    f"awk '{{print $1}}'",
                                    timeout=15, check=False)
                if rc == 0 and out.strip().isdigit():
                    size_mb = int(out.strip())
            except Exception:
                pass
            if size_mb < 0:
                return (False,
                        f"ckpt_dir {ckpt_dir} exists on {source_node} but size "
                        f"probe failed; fail-closed (would otherwise migrate "
                        f"without rsyncing the ckpt → silent step-0 restart)")
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
        ckpt_key = (source_node, target_node, ckpt_dir) if ckpt_dir else None
        if ckpt_dir and size_mb > 0 and not _staging_cache_hit(ckpt_key):
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
            _STAGING_CACHE[ckpt_key] = time.time()

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
    the global lock. Now it's a pure dict membership check — milliseconds.

    Phase 3.0.17 P2 fix: TTL on the cache. A staging entry older than
    STAGING_TTL_S is treated as a miss so the next dispatch cycle re-stages
    against current source content — protects against silent reuse of stale
    staged code/ckpt while the task waits in the queue."""
    key = (task.get("id"), target_node)
    ts = _STAGED_TASKS.get(key)
    if ts is None:
        return False
    if (time.time() - ts) > STAGING_TTL_S:
        _STAGED_TASKS.pop(key, None)
        return False
    return True


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
    # Phase 3.0.14 P4 fix: even a satisfied load-ratio is meaningless when the
    # "overloaded" source actually holds only a few seconds of work — rsync cost
    # would exceed any saving. Require a minimum absolute load on source.
    if source_load < MIGRATION_MIN_SOURCE_LOAD_S:
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
        # Phase 3.0.8 P2 fix: eta=0 (unknown / no signal yet) was previously skipping
        # this filter and getting migrated FIRST due to the ascending-by-eta sort. But
        # "unknown ETA" means we can't reason about whether migration is worth its
        # rsync cost — conservative answer is "don't migrate, wait until we have a
        # rate signal". Treat eta=0 the same as eta-too-short.
        if eta < MIGRATION_MIN_TASK_ETA_S:
            continue
        # Phase 3.0.11 P2 fix: target is the lightest-load node, but if THIS task
        # has env_missing/python_import escalation pending against target, OR has
        # already failed to launch on target, pick_placement would exclude target
        # at dispatch time and fall back to another node — possibly one where the
        # ckpt isn't staged. Result: silent resume-from-step-0. Skip such candidates
        # entirely so we don't burn an rsync staging a doomed target.
        if target_name in _blocked_nodes_for_task(t):
            continue
        if target_name in _launch_failed_nodes_for_task(t):
            continue
        # Phase 3.0.12 P3 fix: cooldown gate — a task that just migrated must wait
        # MIGRATION_COOLDOWN_S before another migration. Stops oscillation from
        # ping-ponging the same task across nodes when load metrics fluctuate.
        last_mig_at = float(t.get("migrated_at") or 0)
        if last_mig_at and (time.time() - last_mig_at) < MIGRATION_COOLDOWN_S:
            continue
        # Phase 3.0.19 P3 fix: skip candidates whose (id, target) had a recent
        # staging failure. Without this, the same first 2 doomed candidates
        # (ckpt > cap, env missing, etc.) get re-picked every dispatch and the
        # rest of the queue starves. Failures TTL out via STAGING_FAIL_COOLDOWN_S.
        if _staging_recently_failed(t["id"], target_name):
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


# Phase 3.5: auto-pull results to local on task completion. Uses an outside-
# lock pattern so the rsync (potentially minutes for big result dirs) doesn't
# stall submit/cancel/status. Watcher and cmd_dispatch both invoke
# _sync_completed_results_outside_lock(); first scans state under a short
# lock for done-tasks-with-result_dir-not-yet-synced, releases, runs rsync
# per candidate, re-acquires briefly to commit success/failure markers.
RESULT_SYNC_MAX_ATTEMPTS = int(os.environ.get("SCHEDULEURM_RESULT_SYNC_MAX_ATTEMPTS", "5"))
RESULT_SYNC_TIMEOUT_S = int(os.environ.get("SCHEDULEURM_RESULT_SYNC_TIMEOUT_S", "1800"))


def _sync_one_result(candidate: dict) -> tuple:
    """rsync remote `result_dir` → local `local_result_dir`. Trailing slash
    on source means "contents", so dst structure mirrors source. Returns
    (ok, msg). NEVER pulls ckpts — those live in `ckpt_dir` which is a
    SEPARATE field, intentionally excluded from this path."""
    src = f"{candidate['host']}:{candidate['result_dir'].rstrip('/')}/"
    dst = candidate['local_result_dir'].rstrip('/') + "/"
    try:
        Path(dst).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return (False, f"mkdir local target failed: {str(e)[:120]}")
    try:
        r = subprocess.run(
            ["rsync", "-az", "--partial", src, dst],
            capture_output=True, text=True, timeout=RESULT_SYNC_TIMEOUT_S,
        )
        if r.returncode != 0:
            return (False, f"rsync rc={r.returncode}: {r.stderr.strip()[:200]}")
        return (True, "ok")
    except subprocess.TimeoutExpired:
        return (False, f"rsync timeout (>{RESULT_SYNC_TIMEOUT_S}s)")
    except Exception as e:
        return (False, f"rsync exception: {str(e)[:200]}")


# Phase 3.4.11 P2 fix: stale-marker grace window. If a session's rsync
# died (host crash, SIGKILL during transfer, etc.) the result_syncing_at
# field would never get cleared. Treat any marker older than
# RESULT_SYNC_TIMEOUT_S + this grace as orphaned so another session can
# reclaim. Grace is 10 min on top of the 30 min timeout = 40 min worst case
# before a stuck task becomes eligible again.
RESULT_SYNC_STALE_GRACE_S = int(os.environ.get("SCHEDULEURM_RESULT_SYNC_STALE_GRACE_S", "600"))


def _sync_completed_results_outside_lock():
    """Phase 3.5: pull results back to local for tasks transitioned to
    status='done' with result_dir set. Three phases (short lock → rsync
    OUTSIDE lock → short lock to commit) mirror _stage_migration_*.

    Skip rules:
      - status != "done"           — only on completion
      - result_dir not set          — opt-in feature
      - result_synced_at already set — one-shot, no re-sync
      - attempts ≥ MAX_ATTEMPTS     — give up (cap defends against
        chronically broken nodes that would otherwise hammer rsync
        every cycle indefinitely)
      - local node (host=None)     — files already here
      - result_syncing_at fresh    — Phase 3.4.11 P2 fix: another session
        (watcher OR ad-hoc dispatch) is currently rsync'ing this task.
        Concurrent rsyncs to the same local dst would race and corrupt
        the result tree. Marker is reclaimed after RESULT_SYNC_TIMEOUT_S
        + RESULT_SYNC_STALE_GRACE_S so a dead-process leak self-heals.

    Migration mid-run + eval flows are NOT triggered here. Migration's
    own staging path (Phase 3.0.4) handles ckpt+cwd between nodes; eval
    submissions reference ckpt_dir directly. result_dir is intentionally
    a separate field so the user explicitly chooses what comes back.
    """
    stale_threshold = RESULT_SYNC_TIMEOUT_S + RESULT_SYNC_STALE_GRACE_S
    candidates = []
    try:
        # Phase 3.4.11 P2 fix: claim each candidate atomically by writing
        # `result_syncing_at` under the lock. save_state at the end of this
        # block makes the claims visible to any concurrent dispatch /
        # watcher invocation that runs the snapshot phase before our rsync
        # finishes.
        with state_lock():
            state = load_state()
            now = time.time()
            for t in state.get("tasks", []):
                if t.get("status") != "done":
                    continue
                rd = t.get("result_dir")
                if not rd:
                    continue
                if t.get("result_synced_at"):
                    continue
                if int(t.get("result_sync_attempts") or 0) >= RESULT_SYNC_MAX_ATTEMPTS:
                    continue
                node = t.get("node")
                if not node:
                    continue
                host = NODES.get(node, {}).get("host")
                if not host:
                    continue  # local node — already on this box
                # Concurrent-rsync guard: skip if another session is mid-rsync.
                syncing_at = t.get("result_syncing_at")
                if syncing_at:
                    age = now - float(syncing_at)
                    if age < stale_threshold:
                        continue  # another worker has it; back off
                    # Stale claim — caller died. Log + reclaim below.
                    try:
                        notify("result_sync_claim_reclaimed", {
                            "task_id": t["id"], "stale_age_s": int(age),
                        }, feishu_enabled=False)
                    except Exception:
                        pass
                # Claim the task atomically.
                t["result_syncing_at"] = now
                candidates.append({
                    "id": t["id"],
                    "host": host,
                    "result_dir": rd,
                    "local_result_dir": t.get("local_result_dir") or rd,
                })
            if candidates:
                save_state(state)
    except Exception as e:
        try:
            notify("result_sync_snapshot_error", {"error": str(e)[:200]},
                   feishu_enabled=False)
        except Exception:
            pass
        return

    if not candidates:
        return

    results = []
    for c in candidates:
        ok, msg = _sync_one_result(c)
        results.append((c["id"], ok, msg))
        try:
            notify("result_sync_done" if ok else "result_sync_failed", {
                "task_id": c["id"], "result_dir": c["result_dir"],
                "local_result_dir": c["local_result_dir"], "msg": msg[:200],
            }, feishu_enabled=False)
        except Exception:
            pass

    try:
        with state_lock():
            state = load_state()
            by_id = {t["id"]: t for t in state["tasks"]}
            for tid, ok, msg in results:
                t = by_id.get(tid)
                if not t:
                    continue
                # Defensive: only commit if task is still in `done` state.
                # If it transitioned away (cancelled, forgotten, requeued)
                # leave the claim cleared and skip the result mutations.
                # Always clear result_syncing_at — we either succeeded or
                # bumped attempts; either way our claim is released.
                t.pop("result_syncing_at", None)
                if t.get("status") != "done":
                    continue
                if ok:
                    t["result_synced_at"] = time.time()
                    t["result_sync_error"] = None
                else:
                    t["result_sync_error"] = msg[:300]
                    t["result_sync_attempts"] = int(t.get("result_sync_attempts") or 0) + 1
            save_state(state)
    except Exception as e:
        try:
            notify("result_sync_commit_error", {"error": str(e)[:200]},
                   feishu_enabled=False)
        except Exception:
            pass


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
        # Phase 3.0.18 P2 fix: probe_all() does ssh + nvidia-smi to every node and
        # can take seconds on a slow / multi-host cluster. Pre-fix it ran INSIDE
        # state_lock here, blocking submit / cancel / status while the watcher
        # (which calls into this every 60s) probed. Now: snapshot state under a
        # short lock, release, probe outside. Identify reads only — safe to run
        # without the lock. Drift is acceptable: the actual migration commit
        # happens later in _do_dispatch under a fresh lock + fresh probe via
        # _consider_migration, which re-runs every gate.
        with state_lock():
            state = load_state()
        nodes = probe_all()
        snapshot = _identify_migration_candidates(state, nodes,
                                                  max_candidates=max_candidates)
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
                # Phase 3.0.19 P3 fix: tag (task,target) as recently-failed so
                # the next identify pass skips it and exposes the next
                # candidate. Failure tag TTLs out via STAGING_FAIL_COOLDOWN_S
                # so transient ssh blips and user-fixable issues (env missing
                # patched, ckpt shrunk) recover automatically.
                _record_staging_failure(task_snapshot["id"], target)
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
            # Same treatment for thrown exceptions — tag and let cooldown drive
            # recovery instead of trying the doomed candidate again next cycle.
            _record_staging_failure(task_snapshot["id"], target)
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
    # Phase 3.0.14 P4 fix: absolute floor on source load. Mirrors the identify-
    # side gate so a trivially-loaded source (target=0s, source=2s, ratio=2.0
    # passes) doesn't trigger migration of a 600s task to save 2s of source load.
    if source_load < MIGRATION_MIN_SOURCE_LOAD_S:
        return []

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
        # Phase 3.0.8 P2 fix: eta=0 (unknown / no signal yet) was previously skipping
        # this filter and getting migrated FIRST due to the ascending-by-eta sort. But
        # "unknown ETA" means we can't reason about whether migration is worth its
        # rsync cost — conservative answer is "don't migrate, wait until we have a
        # rate signal". Treat eta=0 the same as eta-too-short.
        if eta < MIGRATION_MIN_TASK_ETA_S:
            continue  # task will finish before staging completes
        # Phase 3.0.11 P2 fix: defensive recheck of target eligibility. Staging ran
        # OUTSIDE the lock, so a launch failure or env-missing escalation could have
        # been recorded for THIS task on target between identify and consider. Without
        # this, we'd commit migration to a target pick_placement will then exclude →
        # task falls back to a node where the ckpt isn't staged → silent restart.
        if target_name in _blocked_nodes_for_task(t):
            continue
        if target_name in _launch_failed_nodes_for_task(t):
            continue
        # Phase 3.0.12 P3 fix: cooldown gate (mirrors the identify-side gate). Don't
        # commit a second migration of the same task within MIGRATION_COOLDOWN_S.
        last_mig_at = float(t.get("migrated_at") or 0)
        if last_mig_at and (time.time() - last_mig_at) < MIGRATION_COOLDOWN_S:
            continue
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
        # Phase 3.0.15 P1 fix: staged_node pins this task to the staged target
        # for ALL future placement decisions (until launch / terminal). Without
        # this, pick_placement's preferred_node-with-fallback would happily land
        # the task on a third node where cwd/ckpt was never staged → silent
        # step-0 restart. preferred_node alone is "soft preference"; staged_node
        # is "hard pin while ckpt only lives here".
        cand["staged_node"] = target_name
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


def _counts_against_node_concurrency(task: dict) -> bool:
    """Whether a task should consume a node's max_concurrent_running slot.

    Slurm PENDING/CONFIGURING jobs are already accounted by the slurm pending
    throttle, but they do not consume local CPU/RAM yet. Counting them here
    artificially blocks LocalBackend CPU-only work on small local-slurm setups.
    Slurm RUNNING/COMPLETING jobs still count because they have an allocation.
    """
    if task.get("status") != "running":
        return False
    if not task.get("node"):
        return False
    if _is_slurm_managed(task) and task.get("slurm_state") in _SLURM_PENDING_LIKE:
        return False
    return True


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
            _set_current_usage(youngest, 0, 0, 0.0)
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
    # Phase 3.2.1: release the cross-scheduler claim BEFORE clearing the
    # node — release() needs task["node"] still set to know where to ssh.
    try:
        _release_task_claims_and_intents(victim)
    except Exception:
        pass
    _kill_task_processes(victim, timeout=15)
    victim["status"] = "queued"
    for k in ("node", "gpu_idx", "process_group", "log_path", "started_at", "finished_at", "_diagnosis"):
        victim[k] = None
    victim["remote_pids"] = []
    victim["alive_pids"] = []
    victim["peak_vram_mb"] = 0
    victim["peak_ram_mb"] = 0
    _set_current_usage(victim, 0, 0, 0.0)
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
    repaired = reconcile_requeue_lineage_invariants(state)
    if repaired:
        events.append({
            "type": "lineage_repaired",
            "count": repaired,
            "reason": "queued requeue parents were made terminal before dispatch",
        })
    # Phase 3.0.3: load-balance pass — re-pin a queued soft-pinned task from an
    # overloaded node to a near-empty one, BEFORE the placement loop. Hard pins
    # (require_node) are never touched. Capped at MIGRATION_MAX_PER_DISPATCH
    # (default 1) so we don't churn the queue. This runs first so the placement
    # loop sees the new preferred_node assignment.
    migrated = _consider_migration(state, nodes)
    if migrated:
        # Phase 3.0.10 P3 fix: enrich payload with from/to pin + eta + reason so
        # cmd_dispatch can print and the watcher's notify loop can log/Feishu
        # without a second state lookup. Without these fields the README's
        # "visible in watcher.log / journalctl" claim was empty.
        by_id = {t["id"]: t for t in state.get("tasks", [])}
        for tid in migrated:
            cand = by_id.get(tid, {})
            events.append({
                "type": "migrated",
                "task_id": tid,
                "from_node": cand.get("migrated_from"),
                "to_node": cand.get("preferred_node"),
                "eta_seconds": int(cand.get("eta_seconds") or 0),
                "reason": cand.get("last_block_reason", ""),
            })
    # Preemption pass: free a slot for starved high-prio tasks (one eviction max per dispatch).
    preempted = _preempt_for_high_priority(state, nodes)
    # Initialize per-node running task count (for max_concurrent_running cap in _node_resources_ok).
    from collections import Counter as _Counter
    running_per_node = _Counter(t.get("node") for t in state["tasks"]
                                 if _counts_against_node_concurrency(t))
    # Phase 2.16/3.4.13: count OUR slurm-pending tasks per node and split by
    # CPU/GPU bucket. pick_placement throttles further dispatch only when the
    # matching bucket is full, so CPU-only work can proceed behind pending GPU jobs.
    slurm_pending_per_node = _count_slurm_pending_per_node(state)
    for n in nodes:
        n["running_count"] = running_per_node.get(n["name"], 0)
        # Phase 3.4.13 P1 fix: store split (cpu, gpu) on the node dict for
        # pick_placement to consult. Keep the legacy `slurm_pending_count`
        # field as the SUM (cpu + gpu) for backwards compat — surfaces in
        # status output / show / `why` strings still expecting one number.
        split = slurm_pending_per_node.get(n["name"]) or {"cpu": 0, "gpu": 0}
        n["slurm_pending_split"] = split
        n["slurm_pending_count"] = int(split.get("cpu", 0)) + int(split.get("gpu", 0))
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
    # Race-condition guard: precompute run identities that already have an
    # active launch. This is deliberately NOT signature-only. A broad signature
    # may cover many BAPR ablations across local + micro-servers; only identical
    # run identities are blocked. Include 'launching' for the WAL window between
    # queued and running.
    running_keys = {key
                    for t in state["tasks"]
                    if t.get("status") in ("running", "launching")
                    for key in [_task_run_identity(t)]
                    if key}
    queued = sorted(
        [t for t in state["tasks"] if t["status"] == "queued"],
        key=lambda t: (prio.get(t["priority"], 1), t["submitted_at"])
    )
    for t in queued:
        artifact_event = _reconcile_queued_launch_artifacts_before_dispatch(t, state)
        if artifact_event:
            events.append(artifact_event)
            if t.get("status") in ("running", "launching"):
                key = _task_run_identity(t)
                if key:
                    running_keys.add(key)
            continue
        sig = t.get("signature") or ""
        run_key = _task_run_identity(t)
        if run_key and run_key in running_keys:
            reason = (f"run identity already has a running/launching task; "
                      f"refusing to dispatch a duplicate. Broad signatures are allowed: "
                      f"different cmd/cwd/env/result identities with signature {sig!r} "
                      f"can still run in parallel.")
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
                if not _requires_local_capacity_check(n["name"], t):
                    # Phase 3.4.13 P1 fix: report the bucket the task would
                    # have used, so the user sees `slurm(cpu 0/1 pending,
                    # gpu 1/1 pending; throttled gpu)` instead of an opaque
                    # combined number that obscures which pool is full.
                    split = n.get("slurm_pending_split") or {"cpu": 0, "gpu": 0}
                    bucket = _slurm_pending_bucket_for_task(t)
                    cap = _slurm_max_pending_for_node(n["name"], bucket)
                    cpu_cap = _slurm_max_pending_for_node(n["name"], "cpu")
                    gpu_cap = _slurm_max_pending_for_node(n["name"], "gpu")
                    pending = int(split.get(bucket) or 0)
                    if pending >= cap:
                        reasons.append(
                            f"{n['name']}=slurm("
                            f"cpu {int(split.get('cpu') or 0)}/{cpu_cap}, "
                            f"gpu {int(split.get('gpu') or 0)}/{gpu_cap} "
                            f"pending; throttled {bucket})")
                    elif require and require != n["name"]:
                        reasons.append(f"{n['name']}=slurm(require!={require})")
                    else:
                        reasons.append(f"{n['name']}=slurm(deferred but require/prefer mismatch)")
                    continue
                if _task_requests_slurm(t):
                    reasons.append(f"{n['name']}=not-slurm-route")
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
                            util_limit = _node_gpu_util_limit(NODES[n["name"]])
                            if util_limit is not None and g["used_mb"] > 100 and g.get("util_pct", 0) >= util_limit:
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
        # Clear stale FIFO intents from prior CLAIM_RACE attempts on other
        # nodes. Keep the current node's intent, if any, so the remote upsert
        # preserves the original FIFO timestamp for this retry.
        _release_task_claims_and_intents(t, exclude_nodes={t["node"]}, clear_markers=False)
        ok, why = precheck_git(t)
        if not ok:
            t["last_block_reason"] = why
            _release_task_claims_and_intents(t)
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
        # Phase 3.4.10 P1 fix: pre-launch cwd staging from LOCAL source-of-truth.
        # Without this, dispatch's `test -d cwd` on target accepted any directory
        # with the right name (including stale clones / unrelated stubs from
        # other projects), then launch executed the wrong code → silent crash
        # → ENV_MISSING blocklist for that node. The first-launch path was the
        # only one without rsync coverage; migration already had it.
        # Two failure modes handled:
        #   - CAP_EXCEEDED (cwd > LAUNCH_MAX_CWD_SIZE_MB): pin to local, requeue
        #     without bumping launch_fail_count — it's a routing decision, not
        #     a launch error.
        #   - rsync transport error: bump launch_fail_count, requeue (matches
        #     the existing behaviour for cwd-missing failures).
        # Phase 3.4.11 P1 fix: cache-only probe — never block lock on rsync.
        # Pre-launch staging (rsync local→target) runs in
        # _stage_launch_candidates_outside_lock BEFORE this dispatch ever
        # acquires state_lock. Here we just consult the cache state:
        #   "ready"        → proceed with launch
        #   "cap_exceeded" → cwd > 2GB; re-route to local (no fail-count bump)
        #   "stage_failed" → rsync transport failure; treat as launch fail
        #                    on this target (3.4.12 P1-2)
        #   "needs_stage"  → cache miss; defer this cycle, next outside-lock
        #                    pass will rsync, the cycle after launches
        target = t.get("node")
        cwd_for_stage = t.get("cwd")
        if target and cwd_for_stage and NODES.get(target, {}).get("host"):
            stage_state = _stage_cwd_check(target, cwd_for_stage)
            if stage_state == "cap_exceeded":
                t["status"] = "queued"
                t["require_node"] = "local"
                t["last_block_reason"] = (
                    f"launch staging: cwd > {LAUNCH_MAX_CWD_SIZE_MB}MB cap "
                    f"for {target}; pinned require_node=local"
                )
                _release_task_claims_and_intents(t, extra_nodes=[target])
                t["node"] = None
                t["gpu_idx"] = None
                t.pop("launching_started_at", None)
                events.append({
                    "type": "launch_capped",
                    "task_id": t["id"],
                    "task": t,
                    "reason": f"cwd > {LAUNCH_MAX_CWD_SIZE_MB}MB",
                })
                continue
            if stage_state == "stage_failed":
                # Phase 3.4.12 P1-2: rsync to this target keeps failing.
                # Route through the existing launch-failure pipeline so
                # pick_placement avoids this node next cycle, and after
                # MAX_LAUNCH_RETRY total launch failures the task gets
                # heal-escalated. This breaks the "needs_stage forever"
                # loop the original outside-lock split could create when
                # rsync was permanently broken (auth, disk, network).
                fail_msg = (_stage_failure_reason(target, cwd_for_stage)
                            or "rsync to target failed")
                attempted = target
                t["launch_fail_count"] = (t.get("launch_fail_count") or 0) + 1
                failed = t.setdefault("launch_failed_nodes", {})
                if not isinstance(failed, dict):
                    failed = {}
                    t["launch_failed_nodes"] = failed
                failed[attempted] = {
                    "ts": time.time(),
                    "attempt": t["launch_fail_count"],
                    "error": f"stage_cwd: {fail_msg[:280]}",
                }
                t["last_block_reason"] = (
                    f"launch stage attempt {t['launch_fail_count']}/{MAX_LAUNCH_RETRY}: "
                    f"rsync to {target} failed: {fail_msg[:200]}"
                )
                _release_task_claims_and_intents(t, extra_nodes=[target])
                t["node"] = None
                t["gpu_idx"] = None
                t.pop("launching_started_at", None)
                if t["launch_fail_count"] >= MAX_LAUNCH_RETRY:
                    t["status"] = "failed"
                    try:
                        _write_escalation(t, "LAUNCH_FAIL_CAP",
                                          {"reason": fail_msg, "tail": fail_msg})
                    except Exception:
                        pass
                    events.append({"type": "launch_failed_terminal",
                                    "task_id": t["id"], "task": t,
                                    "error": fail_msg})
                else:
                    t["status"] = "queued"
                    events.append({"type": "launch_failed_retry",
                                    "task_id": t["id"], "task": t,
                                    "error": fail_msg})
                continue
            if stage_state == "needs_stage":
                # Defer this cycle. Don't bump launch_fail_count — staging
                # hasn't been attempted yet (it's an outside-lock concern).
                t["status"] = "queued"
                t["last_block_reason"] = (
                    f"launch staging: cwd not yet rsynced to {target}; "
                    f"will retry next dispatch cycle"
                )
                _release_task_claims_and_intents(t, extra_nodes=[target])
                t["node"] = None
                t["gpu_idx"] = None
                t.pop("launching_started_at", None)
                events.append({
                    "type": "launch_stage_deferred",
                    "task_id": t["id"],
                    "task": t,
                    "reason": f"awaiting outside-lock rsync to {target}",
                })
                continue
            # stage_state == "ready" — fall through to launch
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
        # Phase 3.2.1: pass the picked node's probe state so LocalBackend
        # can build a capacity payload for cross-scheduler claim() without a
        # second ssh round-trip.
        picked_state = next((n for n in nodes if n.get("name") == t.get("node")), None)
        ok, msg = launch(t, node_state=picked_state)
        if not ok:
            # Phase 3.2.1: claim race is contention, not a real launch failure.
            # If LocalBackend.launch returned the CLAIM_RACE sentinel, the
            # cross-scheduler claim was rejected (some other scheduleurm
            # claimed the resource first). Revert to queued WITHOUT incrementing
            # launch_fail_count or recording launch_failed_nodes — otherwise
            # legitimate contention would eventually hit MAX_LAUNCH_RETRY +
            # APP_BUG_CAP escalation. Retry naturally on the next dispatch
            # cycle when capacity opens up or another scheduler releases.
            if isinstance(msg, str) and msg.startswith("CLAIM_RACE:"):
                attempted_node = t.get("node")
                _remember_claim_intent(t, attempted_node)
                t["status"] = "queued"
                t["last_block_reason"] = msg
                t["node"] = None
                t["gpu_idx"] = None
                t.pop("launching_started_at", None)
                events.append({"type": "claim_race", "task_id": t["id"],
                                "task": t, "reason": msg})
                continue
            # Don't terminate the task — return it to the queue so dispatch can try a different
            # node next cycle. Common failure modes (ssh timeout, cwd missing on the picked
            # fallback node) are node-specific and recoverable on a different node. After
            # MAX_LAUNCH_RETRY consecutive failures, give up and escalate via heal so the user
            # gets a real diagnosis instead of an indefinite retry loop.
            attempted_node = t.get("node")
            if attempted_node:
                _release_task_claims_and_intents(t, extra_nodes=[attempted_node], clear_markers=False)
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
        _clear_claim_intent_markers(t)
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
        launched_key = _task_run_identity(t)
        if launched_key:
            # Treat a just-launched task as running for the rest of this dispatch pass.
            running_keys.add(launched_key)
        # Reflect placement in our local probe so subsequent iterations of this same dispatch
        # see the resources as already consumed (CPU + RAM at node level, VRAM at GPU level).
        for n in nodes:
            if n["name"] != t["node"]: continue
            n["free_cpu"] = max(0, n.get("free_cpu", 0) - t.get("cpu_cores", DEFAULT_CPU_CORES))
            n["free_ram_mb"] = max(0, n.get("free_ram_mb", 0) - t.get("ram_mb", DEFAULT_RAM_MB))
            n["running_count"] = n.get("running_count", 0) + 1  # for max_concurrent_running cap
            # Phase 2.16 + 3.4.13: bump slurm pending count too so pick_placement
            # in subsequent loop iterations sees this just-sbatched task and
            # respects the per-node cap. Increments the correct bucket
            # (cpu/gpu) so a CPU-only sbatch doesn't fake-out the GPU pool's
            # accounting and vice-versa.
            if _is_slurm_managed(t):
                bucket = _slurm_pending_bucket_for_task(t)
                split = n.setdefault("slurm_pending_split", {"cpu": 0, "gpu": 0})
                split[bucket] = int(split.get(bucket) or 0) + 1
                n["slurm_pending_count"] = (
                    int(split.get("cpu") or 0) + int(split.get("gpu") or 0))
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
        claim_str = _format_node_claim_summary(n)
        print(f"  {n['name']:11s} {gpu_str}  {cpu_str}  {ram_str}{claim_str}")

def _format_task_location(task):
    if not task.get("node"):
        return "-"
    if task.get("slurm_job_id"):
        kind = "SLURM-GPU" if _slurm_pending_bucket_for_task(task) == "gpu" else "SLURM-CPU"
        state = task.get("slurm_state")
        state_part = f":{state}" if state else ""
        return f"{task['node']}:{kind}#{task['slurm_job_id']}{state_part}"
    if task.get("gpu_idx") is None:
        return f"{task['node']}:CPU"
    return f"{task['node']}:GPU{task['gpu_idx']}"


def _claim_wait_s(c: dict) -> int:
    try:
        ts = float(c.get("intent_at") or c.get("claimed_at") or 0)
    except Exception:
        ts = 0
    return max(0, int(time.time() - ts)) if ts else 0


def _format_claim_record(c: dict) -> str:
    gpu = c.get("gpu_idx")
    loc = "CPU" if gpu is None else f"GPU{gpu}"
    return (f"{c.get('scheduler_id','?')}/{c.get('task_id','?')}:{loc} "
            f"vram={int(c.get('vram_mb') or 0)}MB "
            f"cpu={int(c.get('cpu_cores') or 0)} "
            f"ram={int(c.get('ram_mb') or 0)}MB "
            f"age={_claim_wait_s(c)}s")


def _format_node_claim_summary(n: dict) -> str:
    pending = n.get("pending_claims") or []
    active = n.get("active_claims") or []
    intents = n.get("claim_intents") or []
    err = n.get("claim_snapshot_error")
    parts = []
    if active:
        parts.append(f"active_claims={len(active)}")
    if pending:
        parts.append(f"pending_claims={len(pending)}")
    if intents:
        head = sorted(intents, key=lambda c: (float(c.get("intent_at", 0) or 0),
                                             str(c.get("scheduler_id")),
                                             str(c.get("task_id"))))[0]
        parts.append(f"intents={len(intents)} head={head.get('task_id','?')}@{_claim_wait_s(head)}s")
    if err:
        parts.append(f"claims_error={err[:60]}")
    return ("  claims(" + ", ".join(parts) + ")") if parts else ""


def _format_claim_intent_hint_for_task(task: dict, node: str, snap: dict) -> str:
    intents = list(snap.get("intents") or [])
    if not intents:
        return ""
    intents.sort(key=lambda c: (float(c.get("intent_at", 0) or 0),
                                str(c.get("scheduler_id")),
                                str(c.get("task_id"))))
    sid = _ClaimManager.scheduler_id()
    key = (sid, task.get("id"))
    pos = None
    for i, c in enumerate(intents):
        if (c.get("scheduler_id"), c.get("task_id")) == key:
            pos = i + 1
            break
    head = intents[0]
    if pos is not None:
        return (f"claim-intent: position {pos}/{len(intents)}; "
                f"head={_format_claim_record(head)}")
    return f"claim-intents: {len(intents)} queued; head={_format_claim_record(head)}"

_SLURM_ALIVE_STATES = {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING",
                        "RESIZING", "REQUEUED", "SUSPENDED"}


def _try_recover_orphan_slurm_job(task: dict, node: str, state: Optional[dict] = None) -> bool:
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
            # Phase 3.0.33 P1 fix: terminal slurm orphan was previously skipped
            # ("let the revert path handle it"), but the revert path goes back
            # to queued → next dispatch sbatches AGAIN. The slurm job already
            # ran (and possibly succeeded). Rerunning the same task wastes
            # compute and breaks the "task never runs twice" invariant.
            # Now: adopt the terminal record + classify done/failed using the
            # 3.0.30 log-scan semantics for COMPLETED. Reconstruct log_path
            # the same way SlurmBackend.launch did so _scan_completed_log_for_
            # crash can read it.
            task["slurm_job_id"] = jid
            task["slurm_state"] = slurm_state
            task["finished_at"] = time.time()
            task["started_at"] = task.get("launching_started_at") or task["finished_at"]
            task.pop("launching_started_at", None)
            task["remote_pids"] = []
            task["peak_vram_mb"] = 0
            task["peak_ram_mb"] = 0
            _set_current_usage(task, 0, 0, 0.0)
            cwd = task.get("cwd") or ""
            if NODES.get(node, {}).get("host") is None:
                task["log_path"] = f"{STATE_DIR}/logs/{task['id']}.log"
            elif cwd:
                task["log_path"] = f"{cwd}/.scheduleurm/{task['id']}.log"
            # Phase 3.0.35 P1 fix: build a proper _diagnosis BEFORE calling
            # _requeue_after_crash. Without it, _classify_failure({}) returns
            # NORMAL → soft retry — silently re-running OUT_OF_MEMORY /
            # ModuleNotFoundError / OOM-via-pipefail crashes that should
            # escalate. Fetch the actual log tail so OOM_PATTERNS /
            # ENV_MISSING_PATTERNS / PYTHON_IMPORT_PATTERNS can match.
            lifetime_s = int(max(0, task["finished_at"] - task["started_at"]))
            if slurm_state == "COMPLETED":
                crash_matched, crash_reason = _scan_completed_log_for_crash(task)
                if crash_matched:
                    task["status"] = "failed"
                    task["last_block_reason"] = (
                        f"WAL recovery: orphan slurm job {jid} on {node} "
                        f"reported COMPLETED but log shows crash: "
                        f"{crash_reason[:120]}"
                    )
                    tail, log_size = _fetch_log_tail(task)
                    task["_diagnosis"] = {
                        "is_crash": True,
                        "reason": crash_reason,
                        "tail": tail,
                        "lifetime_s": lifetime_s,
                        "log_size": log_size,
                        "log_path": task.get("log_path"),
                        "success_marker": None,
                    }
                    if state is not None:
                        new_id = _requeue_after_crash(task, state)
                        if new_id:
                            task["requeued_as"] = new_id
                else:
                    task["status"] = "done"
                    task["last_block_reason"] = (
                        f"WAL recovery: orphan slurm job {jid} on {node} "
                        f"already COMPLETED; avoids re-submit"
                    )
            elif slurm_state == "CANCELLED":
                _mark_user_cancelled(
                    task,
                    f"WAL recovery: orphan slurm job {jid} on {node} "
                    f"was CANCELLED; treated as user/admin cancel to avoid re-submit",
                )
                task["_diagnosis"] = {
                    "is_crash": False,
                    "reason": "slurm terminal state CANCELLED",
                    "tail": "(slurm reported CANCELLED; not auto-requeued)",
                    "lifetime_s": lifetime_s,
                    "log_size": 0,
                    "log_path": task.get("log_path"),
                    "success_marker": "SLURM_CANCELLED",
                }
            else:
                task["status"] = "failed"
                task["last_block_reason"] = (
                    f"WAL recovery: orphan slurm job {jid} on {node} "
                    f"terminal in state {slurm_state}; avoids re-submit"
                )
                # 3.0.35: surface the slurm terminal state in `reason` so
                # _classify_failure picks up OUT_OF_MEMORY etc., and pull tail
                # so any in-log patterns (e.g. ModuleNotFoundError) classify too.
                tail, log_size = _fetch_log_tail(task)
                reason = f"slurm terminal state {slurm_state}"
                if slurm_state == "OUT_OF_MEMORY":
                    # explicit substring so OOM_PATTERNS matches in classify
                    reason += " (out of memory)"
                task["_diagnosis"] = {
                    "is_crash": True,
                    "reason": reason,
                    "tail": tail,
                    "lifetime_s": lifetime_s,
                    "log_size": log_size,
                    "log_path": task.get("log_path"),
                    "success_marker": None,
                }
                if state is not None:
                    new_id = _requeue_after_crash(task, state)
                    if new_id:
                        task["requeued_as"] = new_id
            return True
        task["slurm_job_id"] = jid
        task["status"] = "running"
        task["remote_pids"] = []  # slurm-managed: no host PIDs to track
        task["started_at"] = task.get("launching_started_at") or time.time()
        _remember_last_placement(task)
        task["peak_vram_mb"] = 0
        task["peak_ram_mb"] = 0
        _set_current_usage(task, 0, 0, 0.0)
        task.pop("launching_started_at", None)
        task["last_block_reason"] = (
            f"WAL recovery: adopted orphan slurm job {jid} on {node} "
            f"(state={slurm_state}); avoids double-submit"
        )
        return True
    return False


def _try_recover_orphan_local_task(task: dict, node: str) -> bool:
    """Phase 3.0.28 P1 fix: LocalBackend counterpart to _try_recover_orphan_slurm_job.

    LocalBackend.launch injects SCHEDULEURM_TASK_ID=<id> into the launched
    process's environment. If scheduler died after the launch returned but
    before save_state could record remote_pids/status=running, the orphan
    process still has the marker. Walk /proc/*/environ on the candidate node;
    if a non-zombie process has the marker, adopt it onto the task record so
    we don't revert + double-launch when dispatch next runs.

    Returns True iff an orphan was found + adopted.
    """
    tid = task.get("id") or ""
    if not tid:
        return False
    # Walk every PID and grep its environ for our marker. -a forces grep into
    # binary mode so NUL-separated environ entries are scanned. The trailing
    # boundary (env var ends with \0) is captured by `\b` since SCHEDULEURM_
    # TASK_ID values won't contain leading word chars after the id (task ids
    # are tNNNN).
    cmd = (
        "for p in $(ps -eo pid= 2>/dev/null); do "
        f"  if grep -aqE 'SCHEDULEURM_TASK_ID={tid}\\b' /proc/$p/environ 2>/dev/null; then "
        "    sid=$(ps -o sid= -p $p 2>/dev/null | tr -d ' '); "
        "    pgid=$(ps -o pgid= -p $p 2>/dev/null | tr -d ' '); "
        "    state=$(awk '/^State:/ {print $2}' /proc/$p/status 2>/dev/null); "
        "    echo \"$p|$sid|$pgid|$state\"; "
        "  fi; "
        "done"
    )
    try:
        rc, out, _ = run_on(node, cmd, timeout=20, check=False)
        if rc != 0:
            return False
    except Exception:
        return False
    candidates = []  # (pid, pgid, is_session_leader)
    for line in (out or "").splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            continue
        try:
            p_, s_, pg_ = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        st = parts[3]
        if st and st[0] in ("Z", "X"):
            continue  # zombie / dead — not adoption-worthy (3.0.25 semantics)
        candidates.append((p_, pg_, p_ == s_))
    if not candidates:
        return False
    # Prefer the session leader (the setsid'd root bash); fall back to lowest
    # PID otherwise. Stable choice across re-runs.
    candidates.sort(key=lambda c: (0 if c[2] else 1, c[0]))
    pid, pgid, _is_leader = candidates[0]
    task["status"] = "running"
    task["remote_pids"] = [pid]
    task["alive_pids"] = [pid]
    task["process_group"] = pgid
    task["started_at"] = task.get("launching_started_at") or time.time()
    _remember_last_placement(task)
    task["peak_vram_mb"] = 0
    task["peak_ram_mb"] = 0
    _set_current_usage(task, 0, 0, 0.0)
    task.pop("launching_started_at", None)
    # Phase 3.4.9 P1: orphan adopt-as-running must wire the live PID into
    # the cross-scheduler claim. The pre-launch claim recorded pid=None
    # (the original launcher died); without this update the claim stays
    # pid=None forever, which (a) gets double-counted as a "pending" claim
    # in node folding (over-subtracting capacity), and (b) becomes
    # GC-eligible after TTL expiry even though the real process is alive
    # — letting another scheduler claim the same resource. Best-effort:
    # if claims are disabled or the call fails (transport), liveness will
    # at least be observable via remote_pids; next watcher reconcile pass
    # will retry.
    if _ClaimManager.enabled_for(node):
        try:
            _ClaimManager.update_pid(node, tid, pid)
        except Exception:
            pass
    # Phase 3.0.32 P1 fix: also restore the deterministic log_path used by
    # LocalBackend.launch. Pre-fix, recovery left log_path=None, so a
    # later _diagnose_terminal saw "no log_path" and short-circuited to
    # is_crash=False — effectively swallowing real crashes silently for
    # any task that went through orphan recovery.
    if NODES.get(node, {}).get("host") is None:
        task["log_path"] = f"{STATE_DIR}/logs/{tid}.log"
    else:
        task["log_path"] = f"/tmp/sched_{tid}.log"
    # Phase 3.0.32 P1 fix: docker artifact recovery. _maybe_wrap_docker
    # set container_name = "sched-{id}" before LocalBackend.launch was
    # called, but pre-fix recovery never restored it — so kill paths
    # would skip the `docker kill <name>` cleanup AND peak-resource
    # tracking would lock onto the host-side bash launcher PID instead
    # of the container's actual main proc (PID isolation via containerd-
    # shim). Same convention as launch: derive container_name from id,
    # then docker inspect for the main pid; on success, replace
    # remote_pids[0] with the container PID so the rest of the tracking
    # machinery (peak_vram, peak_ram, alive_pids) lights up correctly.
    spec = (task.get("env_spec") or "").lower()
    looks_docker = (spec.startswith("docker")
                    or (spec == "auto" and task.get("image")))
    if looks_docker:
        cname = f"sched-{tid}"
        try:
            rc_d, out_d, _ = run_on(
                node,
                f"docker inspect --format '{{{{.State.Pid}}}}' "
                f"{shlex.quote(cname)} 2>/dev/null",
                timeout=5, check=False,
            )
            if rc_d == 0:
                s = (out_d or "").strip()
                if s.isdigit() and int(s) > 0:
                    cpid = int(s)
                    task["container_name"] = cname
                    task["container_main_pid"] = cpid
                    task["remote_pids"] = [cpid]
                    task["alive_pids"] = [cpid]
                    # Phase 3.4.9 P1: refresh claim PID to the container's
                    # main proc, not the bash launcher (mirrors the same
                    # pattern in LocalBackend.launch's docker branch).
                    if _ClaimManager.enabled_for(node):
                        try:
                            _ClaimManager.update_pid(node, tid, cpid)
                        except Exception:
                            pass
        except Exception:
            pass  # docker may not be reachable; recovery proceeds with bash PID
    task["last_block_reason"] = (
        f"WAL recovery: adopted orphan local PID {pid} (pgid={pgid}) on {node} "
        f"matching SCHEDULEURM_TASK_ID={tid}; avoids double-launch"
    )
    return True


def _try_finalize_terminal_local_task(task: dict, node: str, state: dict) -> bool:
    """Phase 3.0.33 P1 fix: if the alive-orphan probe found nothing but the
    task's deterministic log_path exists with content on the candidate node,
    the orphan ran AND finished within the launch-save window. Classify it
    (done / failed) here instead of reverting → re-launching.

    Mirrors the local-vs-remote log-path formula from LocalBackend.launch.
    Calls _diagnose_terminal for the no-slurm-signal classification (full
    heuristic, since no backend signal is available); on crash, triggers
    _requeue_after_crash to mirror the running-task transition path.

    Returns True iff the task was finalized (done or failed). False means
    no log evidence — caller falls back to the revert path.
    """
    tid = task.get("id") or ""
    if not tid:
        return False
    if NODES.get(node, {}).get("host") is None:
        log_path = f"{STATE_DIR}/logs/{tid}.log"
    else:
        log_path = f"/tmp/sched_{tid}.log"

    def _probe_size(path: str) -> int:
        """Return file size on `node` (0 on missing / probe failure)."""
        try:
            if NODES.get(node, {}).get("host") is None:
                lp = Path(path)
                return lp.stat().st_size if lp.exists() else 0
            rc, out, _ = run_on(
                node, f"wc -c < {shlex.quote(path)} 2>/dev/null",
                timeout=5, check=False,
            )
            if rc != 0:
                return 0
            try:
                return int((out or "0").strip())
            except ValueError:
                return 0
        except Exception:
            return 0

    log_size = _probe_size(log_path)
    if log_size <= 0:
        # Phase 3.0.36 P2 fix: wrapper log being empty isn't proof the task
        # didn't run — many cmds redirect their own stdout / stderr (e.g.
        # `python train.py > out.log 2>&1`), making the wrapper file 0 bytes
        # by design. _diagnose_terminal already knows how to recover the real
        # log from the cmd's redirect target; before reverting, probe THAT
        # path. Without this gate the task would get re-launched as a
        # duplicate even though it had run-and-finished successfully.
        cmd_str = task.get("cmd") or ""
        cmd_has_own_redirect = bool(re.search(
            r"(?<!\d)(?:&>|>>|2>&1|>&|>)\s*[^\s|;&)]+", cmd_str)) \
            or "2>&1" in cmd_str
        real_log = None
        if cmd_has_own_redirect:
            m = re.search(r"(?:^|\s)(?:&>|>)\s*([^\s|;&)<>]+)", cmd_str)
            if m:
                real_log = m.group(1)
        if not real_log:
            return False
        real_size = _probe_size(real_log)
        if real_size <= 0:
            # Wrapper empty AND user-redirect target also empty/missing → no
            # evidence the task ever ran. Fall back to revert path.
            return False
        # User-redirect log has content: _diagnose_terminal will recover and
        # read it for classification (it has the same redirect-recovery
        # path). Continue into the finalize block; log_size stays 0 here
        # because we want to record the WRAPPER log_path on the task —
        # _diagnose_terminal probes the user redirect itself.
    # Adopt as terminal. _diagnose_terminal needs log_path / started_at /
    # finished_at to do its work.
    task["log_path"] = log_path
    task["finished_at"] = time.time()
    task["started_at"] = task.get("launching_started_at") or task["finished_at"]
    task.pop("launching_started_at", None)
    task["remote_pids"] = []
    task["alive_pids"] = []
    task["peak_vram_mb"] = 0
    task["peak_ram_mb"] = 0
    _set_current_usage(task, 0, 0, 0.0)
    diag = _diagnose_terminal(task)
    task["_diagnosis"] = diag
    if diag.get("is_crash"):
        task["status"] = "failed"
        task["last_block_reason"] = (
            f"WAL recovery: terminal orphan local task on {node}; diagnosed "
            f"crash: {(diag.get('reason') or '')[:120]}"
        )
        new_id = _requeue_after_crash(task, state)
        if new_id:
            task["requeued_as"] = new_id
    else:
        task["status"] = "done"
        task["last_block_reason"] = (
            f"WAL recovery: terminal orphan local task on {node}; diagnosed "
            f"clean exit ({(diag.get('reason') or 'no signal')[:120]})"
        )
    # Phase 3.4.9 P1: release the claim now that the task is terminal.
    # Mirrors the regular running→done/failed transition (line 2037).
    # Without this, the claim sits with pid=None until TTL GC, occupying
    # capacity that other schedulers refuse to dispatch into.
    if _ClaimManager.enabled_for(node):
        try:
            _release_task_claims_and_intents(task, extra_nodes=[node])
        except Exception:
            pass
    return True


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

    Phase 3.0.28 P1: same recovery path for LocalBackend tasks via the
    SCHEDULEURM_TASK_ID env-var marker injected at launch.

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
        # Phase 2.15 / 3.0.28 / 3.0.33: orphan recovery before revert.
        # Per-task: slurm nodes use squeue+name (alive AND terminal); local
        # nodes use the SCHEDULEURM_TASK_ID env marker via /proc/*/environ
        # for alive orphans, then the deterministic log_path for terminal
        # orphans. Adopted-as-running and adopted-as-terminal both bypass
        # the revert+requeue path that would otherwise re-launch the same
        # workload.
        node = t.get("node")
        if node and not _requires_local_capacity_check(node, t):
            if _try_recover_orphan_slurm_job(t, node, state):
                continue  # adopted; do not revert
        elif node:
            if _try_recover_orphan_local_task(t, node):
                continue  # adopted as running
            if _try_finalize_terminal_local_task(t, node, state):
                continue  # adopted as terminal (done / failed + maybe requeued)
        # Phase 3.4.9 P1: a launching task that we revert may have had a
        # cross-scheduler claim created in LocalBackend.launch (line ~2926)
        # but never released — _release_and_fail only fires when launch()
        # itself raised, not when the scheduler died mid-launch. Releasing
        # before status=queued prevents the dead claim from sitting at
        # pid=None until TTL GC, which would block our own retry next
        # cycle (claim still occupies the resource we need to reclaim).
        if node and _ClaimManager.enabled_for(node):
            try:
                _release_task_claims_and_intents(t, extra_nodes=[node])
            except Exception:
                pass
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
            # Phase 3.0.27 P1 fix: record sync result so launch can refuse to
            # wrap when this cycle's sync didn't succeed. Pre-fix, only the
            # local-path check (3.0.22) gated launch — but that check can't
            # see remote-side staleness when local-side rsync FAILS yet local
            # path still exists. A stale remote env at the same path would
            # silently run.
            if ok:
                _record_conda_sync_ok(node, env_path)
            else:
                _record_conda_sync_failed(node, env_path)
            ev = "preload_conda_ok" if ok else "preload_conda_failed"
            notify(ev, {"node": node, "env_path": env_path, "msg": msg[:300] if not ok else ""},
                   feishu_enabled=False)
        except Exception as e:
            _record_conda_sync_failed(node, env_path)
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
    # Phase 3.4.11 P1 fix: pre-launch cwd staging OUTSIDE state_lock so the
    # 600s rsync timeout never blocks submit/cancel/status/watcher. Helper
    # populates _STAGING_CACHE / _STAGING_CAP_EXCEEDED; _do_dispatch's
    # launch site uses _stage_cwd_check (cache lookup, never rsync).
    try:
        _stage_launch_candidates_outside_lock()
    except Exception as _e:
        notify("launch_staging_error_outer", {"error": str(_e)[:200]},
               feishu_enabled=False)
    # Phase 3.5: pull results back from remote nodes for tasks that
    # transitioned to status='done' since the previous dispatch. Outside-
    # lock for the same reason migration staging is — a multi-GB rsync
    # would otherwise stall every other lock holder. The helper itself
    # uses three short-lock phases (snapshot → rsync → commit markers).
    try:
        _sync_completed_results_outside_lock()
    except Exception as _e:
        notify("result_sync_error_outer", {"error": str(_e)[:200]},
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
    if qcount == 0 and not events:
        print("=== dispatch === (nothing queued)")
        return
    print(f"=== dispatch === ({qcount} queued)")
    for ev in events:
        if ev["type"] == "lineage_repaired":
            print(f"  [lineage] repaired {ev.get('count', 0)} queued parent retry record(s)")
            continue
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
        elif ev["type"] == "migrated":
            print(f"  [{tid}] MIGRATE {ev.get('from_node') or '?'} → "
                  f"{ev.get('to_node') or '?'}  (eta={ev.get('eta_seconds', 0)}s, load-balance)")
        elif ev["type"] == "preempted":
            print(f"  [{tid}] PREEMPT on {ev.get('freed_node') or '?'} "
                  f"(freed {ev.get('cpu_freed', 0)}c / {ev.get('ram_freed', 0)}MB)")
        elif ev["type"] == "claim_race":
            print(f"  [{tid}] CLAIM-RACE {ev['reason'][:200]}")

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
    if event_type == "task_migrated":
        return (f"[scheduler] 🔀 {payload['task_id']} re-pinned "
                f"{payload.get('from_node') or '?'} → {payload.get('to_node') or '?'} "
                f"(eta={payload.get('eta_seconds', 0)}s, load-balance)")
    if event_type == "task_preempted":
        return (f"[scheduler] ⚡ {payload['task_id']} preempted on "
                f"{payload.get('freed_node') or '?'} "
                f"(freed {payload.get('cpu_freed', 0)}c/{payload.get('ram_freed', 0)}MB)")
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
            "current_vram_mb": sum_vram,
            "current_ram_mb": sum_rss,
            "current_pcpu": sum_pcpu,
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
    # Phase 3.4.11 P1 fix: pre-launch cwd staging OUTSIDE state_lock (see
    # cmd_dispatch comment for rationale).
    try:
        _stage_launch_candidates_outside_lock()
    except Exception as _e:
        notify("launch_staging_error_outer", {"error": str(_e)[:200]},
               feishu_enabled=False)
    # Phase 3.5: pull results back from remote nodes for done tasks
    # opted in via --result-dir. Outside-lock so multi-GB rsync doesn't
    # stall the watcher cycle. See cmd_dispatch comment for the same
    # rationale as migration staging.
    try:
        _sync_completed_results_outside_lock()
    except Exception as _e:
        notify("result_sync_error_outer", {"error": str(_e)[:200]},
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
        elif ev["type"] == "migrated":
            # Phase 3.0.10 P3 fix: surface migrations in watcher.log + Feishu so
            # the README's visibility claim is actually true. Was previously a
            # silent state mutation invisible to operators.
            notify("task_migrated", {
                "task_id": ev["task_id"],
                "from_node": ev.get("from_node"),
                "to_node": ev.get("to_node"),
                "eta_seconds": ev.get("eta_seconds", 0),
                "reason": ev.get("reason", ""),
            })
        elif ev["type"] == "preempted":
            notify("task_preempted", {
                "task_id": ev["task_id"],
                "freed_node": ev.get("freed_node"),
                "cpu_freed": ev.get("cpu_freed", 0),
                "ram_freed": ev.get("ram_freed", 0),
            })
        elif ev["type"] == "claim_race":
            # Phase 3.2.1: cross-scheduler claim contention. Logged so the
            # user can see WHO had the conflict (multi-user setups). No
            # Feishu push — not actionable, just informational.
            notify("task_claim_race",
                   {"task_id": ev["task_id"], "reason": ev["reason"][:300]},
                   feishu_enabled=False)
        # no_fit / resume_found are not surfaced — too noisy for Feishu, but they're in the JSONL log.
    for t in auto_adopted:
        notify("task_auto_adopted", t)
    for batch in batch_completions:
        notify("batch_complete", batch)
    if archived_count:
        notify("archived_terminal_tasks", {"count": archived_count, "age_days": ARCHIVE_AGE_DAYS},
               feishu_enabled=False)

    # Phase 3.2.1: tend cross-scheduler claims on every claims-enabled node.
    # Single ssh per node:
    #   1. renew_many — bumps expires_at on ALL of our running tasks' claims
    #      so they don't expire before the next watcher cycle (TTL is 1h
    #      default, watcher cycle is 60s, so we have plenty of margin even
    #      if a few cycles get delayed).
    #   2. gc_stale runs implicitly inside renew_many's pre-op gc pass, so
    #      any expired-and-dead-pid claim from any scheduler is dropped.
    # Graceful: ssh failures don't crash the watcher; they're logged.
    try:
        with state_lock():
            cur_state = load_state()
        running_by_node = {}
        # Phase 3.4.9 P1: also collect (task_id -> live PID) per node for
        # claim pid reconcile. update_pid in launch / orphan-adopt is best-
        # effort with try/except; if those failed (transport blip, claims
        # disabled momentarily, etc.) the claim sits at pid=None forever
        # and gets double-counted in pending-claim folding (line ~1305).
        # The watcher reconciles each cycle so the worst-case window is
        # one watch interval, not the task's lifetime.
        live_pid_by_node = {}
        for t in cur_state.get("tasks", []):
            if t.get("status") != "running":
                continue
            node = t.get("node")
            if node and _ClaimManager.enabled_for(node):
                running_by_node.setdefault(node, []).append(t["id"])
                pids = t.get("remote_pids") or []
                if pids:
                    live_pid_by_node.setdefault(node, {})[t["id"]] = int(pids[0])
        for node in NODES:
            if not _ClaimManager.enabled_for(node):
                continue
            try:
                ids = running_by_node.get(node, [])
                if ids:
                    _ClaimManager.renew_many(node, ids)
                    # Reconcile claim PIDs against task remote_pids[0]. We
                    # only update when (a) we know a live PID for the task
                    # and (b) the claim record's pid is missing or differs.
                    # No-op for tasks where remote_pids is empty (slurm-
                    # routed) — those legitimately have pid=None claims
                    # and we have no host PID to record.
                    pid_map = live_pid_by_node.get(node) or {}
                    if pid_map:
                        try:
                            current_claims = _ClaimManager.enumerate(node)
                        except Exception:
                            current_claims = []
                        for c in current_claims:
                            tid = c.get("task_id")
                            want_pid = pid_map.get(tid)
                            if want_pid is None:
                                continue
                            cur_pid = c.get("pid")
                            if cur_pid == want_pid:
                                continue
                            try:
                                _ClaimManager.update_pid(node, tid, want_pid)
                            except Exception:
                                pass
                else:
                    # No running tasks of ours on this node — but other
                    # schedulers' claims may still be expiring. Run a
                    # cheap gc pass.
                    _ClaimManager.gc_stale(node)
            except Exception as e:
                notify("claims_tend_error",
                       {"node": node, "error": str(e)[:200]},
                       feishu_enabled=False)
    except Exception as e:
        notify("claims_tend_outer_error", {"error": str(e)[:200]},
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

def _format_task_vram_usage(task):
    if task.get("status") == "running":
        cur = int(task.get("current_vram_mb") or 0)
        if cur > 0:
            return f"cur={cur}MB"
    if task.get("peak_vram_mb"):
        return f"peak={task['peak_vram_mb']}MB"
    return f"~{task.get('est_vram_mb', 0)}MB"


def _format_task_ram_usage(task):
    if task.get("status") == "running":
        cur = int(task.get("current_ram_mb") or 0)
        if cur > 0:
            return f"Rcur={cur}MB"
    if task.get("peak_ram_mb"):
        return f"Rpeak={task['peak_ram_mb']}MB"
    return f"~{task.get('ram_mb', 0)}MB"


def cmd_status(args):
    with state_lock():
        state = load_state()
        recover_stale_launching_tasks(state)
        update_running_tasks(state)
        reconcile_requeue_lineage_invariants(state)
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
        claim_str = _format_node_claim_summary(n)
        print(f"  {n['name']:11s} {gpu_str}  {cpu_str}  ram_free={n['free_ram_mb']}MB{etaload_str}{claim_str}")
    print("\n=== tasks ===")
    show_done = args.all
    rows = [t for t in state["tasks"] if show_done or t["status"] in ("queued", "launching", "running")]
    if not rows:
        print("  (no active tasks; pass --all to see history)")
    for t in rows:
        loc = _format_task_location(t)
        peak = _format_task_vram_usage(t)
        pram = _format_task_ram_usage(t)
        runtime = ""
        if t.get("started_at"):
            end = t.get("finished_at") or time.time()
            runtime = f" {(end - t['started_at'])/60:.1f}m"
        proj = (t.get("project") or "?")[:14]
        print(f"  [{t['id']}] {t['status']:9s} {loc:20s} {proj:14s} {peak:14s} {pram:13s}{runtime}  {t['description'][:55]}")


def cmd_claims(args):
    nodes = [args.node] if args.node else [n for n in NODES if _ClaimManager.enabled_for(n)]
    snapshots = {}
    for node in nodes:
        snapshots[node] = _ClaimManager.snapshot(node)
    if args.json:
        print(json.dumps(snapshots, indent=2))
        return
    if not snapshots:
        print("(no claims-enabled nodes)")
        return
    for node, snap in snapshots.items():
        if not snap.get("ok"):
            print(f"=== {node} claims ERROR ===")
            print(f"  {snap.get('error') or 'unknown error'}")
            continue
        claims = list(snap.get("claims") or [])
        intents = list(snap.get("intents") or [])
        print(f"=== {node} claims ===")
        if claims:
            for c in sorted(claims, key=lambda x: (str(x.get("gpu_idx")), str(x.get("task_id")))):
                pid = c.get("pid")
                pid_s = f" pid={pid}" if pid else " pending"
                print(f"  {_format_claim_record(c)}{pid_s}")
        else:
            print("  (none)")
        print(f"=== {node} intents ===")
        if intents:
            intents.sort(key=lambda c: (float(c.get("intent_at", 0) or 0),
                                        str(c.get("scheduler_id")),
                                        str(c.get("task_id"))))
            for i, c in enumerate(intents, 1):
                print(f"  {i:02d}. {_format_claim_record(c)}")
        else:
            print("  (none)")


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
                _release_task_claims_and_intents(t)
                _mark_user_cancelled(t, "user cancel")
                related = _cancel_related_queued_retries(state, t, "user cancel")
                t.pop("launching_started_at", None)
                save_state(state)
                suffix = f" (+{related} duplicate queued retry)" if related else ""
                print(f"cancelled {prev} task {args.id}{suffix}")
                return
            if t["status"] == "running":
                if not args.force:
                    sys.exit(f"task {args.id} is RUNNING — pass --force to kill it (will not affect other tasks)")
                pids = _task_pids(t)
                ok, kill_msg = _kill_task_processes(t, timeout=15)
                # Phase 3.2.1: release the cross-scheduler claim too. Best-
                # effort; failure leaves the claim to expire via TTL + GC.
                try:
                    _release_task_claims_and_intents(t)
                except Exception:
                    pass
                _mark_user_cancelled(t, "user force-cancel")
                related = _cancel_related_queued_retries(state, t, "user force-cancel")
                save_state(state)
                suffix = kill_msg if ok else f"kill warning: {kill_msg}"
                dup_suffix = f"; also cancelled {related} duplicate queued retry" if related else ""
                print(f"killed pids={pids} on {t['node']} and cancelled {args.id} ({suffix}{dup_suffix})")
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
            _release_task_claims_and_intents(t)
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
                _release_task_claims_and_intents(t)
                _mark_user_cancelled(t, "user clear-queue")
        save_state(state)
        print(f"cancelled {len(ids)} queued tasks (running tasks untouched)")

def cmd_rebalance_pending(args):
    """Pull all currently-pending slurm tasks back into scheduleurm's queue so they
    re-distribute under the current policy (e.g. after changing per-bucket
    slurm pending caps).

    Acts only on tasks with status='running' AND slurm_job_id set AND slurm_state in
    {None, '', PENDING, CONFIGURING, REQUEUED, SUSPENDED}. RUNNING / COMPLETING tasks
    are NEVER touched (they have allocated GPUs and may be mid-training). LocalBackend
    tasks (no slurm_job_id) are also untouched.

    For each candidate: `scancel <jid>` on the task's node (best-effort — orphan
    cleanup if it fails; orphan job times out at slurm walltime), then clear slurm
    fields and revert status='queued'. Next dispatch cycle re-places under the
    current throttle, spreading pending across slurm nodes that have free cap.

    Use case: after editing SLURM_MAX_PENDING_*_PER_NODE /
    NODES[name].max_slurm_pending_* / NODES[name].max_slurm_pending
    or during a policy migration, to re-distribute already-sbatched-but-not-yet-running
    tasks. Safe to run anytime — RUNNING tasks won't be killed.

    Phase 3.0.13 P3 fix: split into 3 phases so the slow scancel+squeue ssh round-trip
    happens OUTSIDE state_lock. Pre-fix this loop held the lock for ~5s per candidate;
    a 20-task batch blocked submit/cancel/status/watcher iterations for ~100s. Now:
      Phase 1 (short lock) — load state, snapshot candidates.
      Phase 2 (NO LOCK)    — pre-check slurm state, scancel, post-verify per candidate.
      Phase 3 (short lock) — re-load state, defensive recheck (status/jid/slurm_state
                              unchanged), commit clears + requeues.
    """
    # Phase 1: identify (short lock).
    filter_ids = set(getattr(args, "task_ids", None) or [])
    with state_lock():
        state = load_state()
        candidates_snapshot = []
        for t in state["tasks"]:
            if filter_ids and t.get("id") not in filter_ids:
                continue
            if t.get("status") != "running":
                continue
            if not _is_slurm_managed(t):
                continue
            if t.get("slurm_state") not in _SLURM_PENDING_LIKE:
                continue
            candidates_snapshot.append({
                "id": t["id"],
                "jid": int(t["slurm_job_id"]),
                "node": t["node"],
                "signature": t.get("signature"),
                "slurm_state": t.get("slurm_state"),
            })

    if not candidates_snapshot:
        suffix = f" matching {sorted(filter_ids)}" if filter_ids else ""
        print(f"no slurm-pending tasks{suffix} to rebalance")
        return

    if not args.yes:
        print(f"would rebalance {len(candidates_snapshot)} task(s) (scancel + revert to queued):")
        for c in candidates_snapshot[:15]:
            print(f"  {c['id']}: jid={c['jid']} on {c['node']}  "
                  f"state={c.get('slurm_state') or 'NEW'}  sig={c.get('signature')}")
        if len(candidates_snapshot) > 15:
            print(f"  ... and {len(candidates_snapshot) - 15} more")
        print("RUNNING / COMPLETING tasks are NOT touched.")
        print("Re-run with --yes to proceed.")
        return

    # Phase 2: scancel + verify (NO LOCK held). Slow ssh ops here.
    # Phase 3.0.7 P1 invariant: only mark cancelled=True if slurm confirms terminal/gone.
    # Phase 3.0.13: pre-scancel state check — the outside-lock window widens the race
    # where slurm transitions PENDING→RUNNING; never scancel a RUNNING/COMPLETING task.
    ALIVE_STATES = _SLURM_PENDING_LIKE | {"RUNNING", "COMPLETING"}
    results = []  # [(tid, jid, cancelled, msg)]
    for c in candidates_snapshot:
        tid, jid, node = c["id"], c["jid"], c["node"]
        # Pre-check: if slurm has already moved this job, decide without scancel.
        pre_decided = False
        try:
            rc0, out0, _ = run_on(
                node,
                f"squeue -h -j {int(jid)} -t all -o '%T' 2>/dev/null",
                timeout=10, check=False,
            )
            if rc0 == 0:
                pre_state = (out0 or "").strip().splitlines()
                pre_state = pre_state[0].strip().upper() if pre_state else ""
                if pre_state in ("RUNNING", "COMPLETING"):
                    results.append((tid, jid, False,
                                    f"transitioned to {pre_state} during rebalance "
                                    f"window; left in place"))
                    pre_decided = True
                elif pre_state and pre_state not in ALIVE_STATES:
                    results.append((tid, jid, True,
                                    f"already {pre_state} (no scancel needed)"))
                    pre_decided = True
                elif not pre_state:
                    results.append((tid, jid, True,
                                    "already gone from squeue (no scancel needed)"))
                    pre_decided = True
                # else: pre_state in PENDING_LIKE → fall through to scancel
        except Exception:
            pass  # squeue check failed → still try scancel
        if pre_decided:
            continue

        cancelled = False
        scancel_msg = ""
        try:
            rc, _, err = run_on(node, f"scancel {int(jid)}", timeout=10, check=False)
            if rc != 0:
                scancel_msg = f"scancel rc={rc}: {err.strip()[:120]}"
        except Exception as e:
            scancel_msg = f"scancel exception: {str(e)[:120]}"
        # Verify regardless of scancel rc — scancel is async; slurm may take a few
        # seconds to act.
        time.sleep(1.5)
        try:
            rc2, out2, _ = run_on(
                node,
                f"squeue -h -j {int(jid)} -t all -o '%T' 2>/dev/null",
                timeout=10, check=False,
            )
            if rc2 == 0:
                state_str = (out2 or "").strip().splitlines()
                state_str = state_str[0].strip().upper() if state_str else ""
                if not state_str:
                    cancelled = True
                elif state_str not in ALIVE_STATES:
                    cancelled = True
                else:
                    cancelled = False
                    if not scancel_msg:
                        scancel_msg = f"slurm still reports state={state_str} after scancel"
            else:
                cancelled = False
                if not scancel_msg:
                    scancel_msg = f"post-scancel verify: squeue rc={rc2}"
        except Exception as e:
            cancelled = False
            if not scancel_msg:
                scancel_msg = f"post-scancel verify exception: {str(e)[:120]}"
        results.append((tid, jid, cancelled, scancel_msg or "verified cancelled"))

    # Phase 3: commit (short lock). Defensive recheck — task may have transitioned
    # during the outside-lock window. Watcher's update_running_tasks could have
    # flipped slurm_state PENDING→RUNNING, or user could have cancelled the task.
    rebalanced = 0
    skipped_scancel_failed = []
    skipped_state_changed = []
    with state_lock():
        state = load_state()
        by_id = {t["id"]: t for t in state["tasks"]}
        for tid, jid, cancelled, msg in results:
            t = by_id.get(tid)
            if not t:
                skipped_state_changed.append((tid, "task gone from state"))
                continue
            if t.get("status") != "running":
                skipped_state_changed.append((tid, f"status now {t.get('status')!r}"))
                continue
            if str(t.get("slurm_job_id") or "") != str(jid):
                skipped_state_changed.append((tid, "slurm_job_id changed"))
                continue
            if t.get("slurm_state") not in _SLURM_PENDING_LIKE:
                # Watcher saw the job transition (e.g., PENDING→RUNNING) during our
                # outside-lock window. Even if our scancel reported cancelled=True
                # we leave the task alone — operator should investigate.
                skipped_state_changed.append((tid,
                    f"slurm_state now {t.get('slurm_state')!r}"))
                continue
            if not cancelled:
                skipped_scancel_failed.append((tid, jid, msg))
                t["last_block_reason"] = (
                    f"rebalance-pending SKIPPED for jid={jid}: {msg}; "
                    f"task left in place to avoid duplicate sbatch"
                )
                continue
            old_node = t.get("node") or "?"
            # Phase 3.0.24 P3 fix: also clear `node`, `gpu_idx`,
            # `actual_started_at`. Pre-fix the requeued task kept its old node
            # pinned in state — _do_dispatch overwrites it on re-placement, but
            # in the meantime status/TUI/env-smoke probes would see a queued
            # task displayed/probed against the OLD node, which is misleading
            # at best (the whole point of rebalance-pending is that the old
            # placement is being undone). Capture the old_node into the
            # last_block_reason message before clearing so the audit trail
            # still names where the task came from.
            for k in ("slurm_job_id", "slurm_state", "started_at", "finished_at",
                      "log_path", "_diagnosis", "process_group", "launching_started_at",
                      "node", "gpu_idx", "actual_started_at"):
                t[k] = None
            t["remote_pids"] = []
            t["alive_pids"] = []
            t["status"] = "queued"
            t["last_block_reason"] = (
                f"rebalance-pending: scancelled slurm job {jid} on {old_node} "
                f"({msg}); will re-dispatch under current policy"
            )
            rebalanced += 1
        save_state(state)

    print(f"rebalanced {rebalanced} task(s) — back to queued for re-dispatch")
    if skipped_scancel_failed:
        print(f"\nWARN: {len(skipped_scancel_failed)} scancel(s) NOT verified — "
              f"those tasks LEFT IN PLACE to avoid duplicate sbatch:")
        for tid, jid, why in skipped_scancel_failed[:5]:
            print(f"  {tid} jid={jid}: {why}")
        print("Manual recovery: run `squeue -j <jid>` on each, scancel by hand, "
              "re-run rebalance-pending. Or wait for the orphan to time out at walltime.")
    if skipped_state_changed:
        print(f"\nINFO: {len(skipped_state_changed)} task(s) transitioned during "
              f"rebalance and were left untouched:")
        for tid, why in skipped_state_changed[:5]:
            print(f"  {tid}: {why}")


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
            "peak_ram_mb": sum_rss,
            "current_vram_mb": sum_vram,
            "current_ram_mb": sum_rss,
            "current_pcpu": sum_pcpu,
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

def cmd_priority(args):
    """Phase 3.1: change priority on a queued task without cancel+resubmit.

    Only queued tasks make sense — running tasks have already been placed,
    and `priority` is a queue-ordering knob. Eviction protection on running
    tasks is currently keyed on `priority == "normal"` so changing a running
    task's priority WOULD affect future eviction decisions, but that's
    rarely the user's intent and easy to do wrong; refuse there to keep
    semantics simple.
    """
    new = args.level
    with state_lock():
        state = load_state()
        for t in state["tasks"]:
            if t["id"] != args.id:
                continue
            if t.get("status") != "queued":
                sys.exit(f"task {args.id} is {t.get('status')!r}, not queued; "
                         f"priority only affects queue ordering")
            old = t.get("priority", "normal")
            if old == new:
                print(f"{args.id} priority already {new!r}")
                return
            t["priority"] = new
            save_state(state)
            print(f"{args.id}: priority {old!r} → {new!r}  "
                  f"({(t.get('description') or '')[:60]})")
            return
        sys.exit(f"task {args.id} not found")


def cmd_edit(args):
    """Phase 3.1: override resource estimates on a queued task. Use this to
    fix bad history-based estimates (e.g. an outlier sample stuck in
    history that's making a task uplaceable). Running tasks can't be edited
    — their resources are already in flight and peak tracking uses real
    measurements anyway.
    """
    if all(getattr(args, k, None) is None for k in
           ("vram_mb", "ram_mb", "cpu", "description", "preferred_node",
            "require_node")):
        sys.exit("specify at least one of --vram-mb / --ram-mb / --cpu / "
                 "--description / --preferred-node / --require-node")
    with state_lock():
        state = load_state()
        for t in state["tasks"]:
            if t["id"] != args.id:
                continue
            if t.get("status") != "queued":
                sys.exit(f"task {args.id} is {t.get('status')!r}, not queued; "
                         f"cannot edit resources of in-flight tasks. Cancel + "
                         f"resubmit if you really need to change them.")
            changes = []
            placement_changed = False
            if args.vram_mb is not None:
                changes.append(("est_vram_mb", t.get("est_vram_mb"), int(args.vram_mb)))
                t["est_vram_mb"] = int(args.vram_mb)
            if args.ram_mb is not None:
                changes.append(("ram_mb", t.get("ram_mb"), int(args.ram_mb)))
                t["ram_mb"] = int(args.ram_mb)
            if args.cpu is not None:
                changes.append(("cpu_cores", t.get("cpu_cores"), int(args.cpu)))
                t["cpu_cores"] = int(args.cpu)
            if args.description is not None:
                changes.append(("description", t.get("description"), args.description))
                t["description"] = args.description
            if args.preferred_node is not None:
                if args.preferred_node not in NODES:
                    sys.exit(f"--preferred-node {args.preferred_node!r} not in NODES "
                             f"({list(NODES.keys())})")
                changes.append(("preferred_node", t.get("preferred_node"),
                                args.preferred_node))
                t["preferred_node"] = args.preferred_node
                placement_changed = True
            if args.require_node is not None:
                if args.require_node and args.require_node not in NODES:
                    sys.exit(f"--require-node {args.require_node!r} not in NODES "
                             f"({list(NODES.keys())})")
                changes.append(("require_node", t.get("require_node"),
                                args.require_node or None))
                t["require_node"] = args.require_node or None
                placement_changed = True
            if placement_changed:
                keep = t.get("require_node") or t.get("preferred_node")
                if keep:
                    _release_task_claims_and_intents(
                        t, exclude_nodes={keep}, clear_markers=False)
                else:
                    _release_task_claims_and_intents(t)
            save_state(state)
            for field, old, new in changes:
                print(f"  {field}: {old!r} → {new!r}")
            print(f"updated {args.id}")
            return
        sys.exit(f"task {args.id} not found")


def _explain_node_fit(task: dict, node_state: dict) -> str:
    """Phase 3.1: return a one-line explanation of whether a queued task
    would fit on `node_state` (a probe_all entry), and if not, why.
    Used by `cmd_why` so users can diagnose stuck tasks without grepping
    the source. Mirrors the predicates in pick_placement / _node_resources_ok
    / _gpu_fits."""
    name = node_state["name"]
    if not node_state.get("alive"):
        return f"DOWN ({node_state.get('error', '?')})"
    if name in _blocked_nodes_for_task(task):
        return "BLOCKED: pending env_missing/python_import escalation against this signature/cwd/project"
    soft_blocked = name in _launch_failed_nodes_for_task(task)
    soft_note = "  (soft-blocked: prior launch_failed; may retry as last resort)" if soft_blocked else ""
    if not _requires_local_capacity_check(name, task):
        # slurm-routed node: gres handles GPU pinning; only throttle matters here.
        # Phase 3.4.13 P1: report split throttle per bucket so the user can see
        # cpu-only tasks aren't blocked by gpu pending and vice-versa.
        split = node_state.get("slurm_pending_split") or {"cpu": 0, "gpu": 0}
        bucket = _slurm_pending_bucket_for_task(task)
        cap = _slurm_max_pending_for_node(name, bucket)
        cpu_cap = _slurm_max_pending_for_node(name, "cpu")
        gpu_cap = _slurm_max_pending_for_node(name, "gpu")
        pending = int(split.get(bucket) or 0)
        if pending >= cap:
            return (f"slurm: {bucket} bucket throttled "
                    f"({pending}/{cap} pending; "
                    f"cpu={int(split.get('cpu') or 0)}/{cpu_cap}, "
                    f"gpu={int(split.get('gpu') or 0)}/{gpu_cap})"
                    f"{soft_note}")
        return f"slurm: would route here (gres handles GPU pinning){soft_note}"
    if _task_requests_slurm(task):
        return f"not-slurm-route: task has --slurm-* options but this node is default-local/non-slurm{soft_note}"
    node_info = NODES[name]
    ok, why = _node_resources_ok(task, node_state, node_info)
    if not ok:
        return f"node-reject: {why}{soft_note}"
    cpu_only = (task.get("est_vram_mb") or 0) <= 0
    if cpu_only:
        return (f"FITS (CPU-only): free_cpu={node_state.get('free_cpu')} "
                f"free_ram={node_state.get('free_ram_mb')}MB{soft_note}")
    fit_gpus = []
    reject_gpus = []
    est = int(task.get("est_vram_mb") or 0)
    cap_per_task = node_info.get("max_vram_per_task")
    for g in node_state.get("gpus") or []:
        if _gpu_fits(task, g, node_info):
            fit_gpus.append(f"GPU{g['idx']}(free={g['free_mb']}MB)")
            continue
        # explain rejection
        reasons = []
        if cap_per_task is not None and est > cap_per_task:
            reasons.append(f"est={est}>per-task cap={cap_per_task}")
        third = g["total_mb"] // 3
        if g["used_mb"] >= third and g["used_mb"] > 100:
            reasons.append(f"used {g['used_mb']}/{g['total_mb']}MB ≥ 1/3 (packing rule)")
        util_limit = _node_gpu_util_limit(node_info)
        if util_limit is not None and g["used_mb"] > 100 and g.get("util_pct", 0) >= util_limit:
            reasons.append(f"util={g.get('util_pct')}% ≥ {util_limit}% (compute saturation)")
        if g["free_mb"] < est + VRAM_MARGIN_MB:
            reasons.append(f"free={g['free_mb']}MB < est+margin ({est}+{VRAM_MARGIN_MB})")
        reject_gpus.append(f"GPU{g['idx']}({'; '.join(reasons) or 'unknown'})")
    if fit_gpus:
        return f"FITS: {', '.join(fit_gpus)}{soft_note}"
    if not reject_gpus:
        return f"no GPUs probed (CPU-only node?){soft_note}"
    return f"no-GPU-fit: {' | '.join(reject_gpus)}{soft_note}"


def cmd_why(args):
    """Phase 3.1: synthesize 'why is this task stuck in queue'. Prints the
    task's own block reason, sibling history (so users can see if their
    est_vram_mb is an outlier), and a per-node fit analysis explaining
    every reject (1/3 rule, util saturation, RAM headroom, etc)."""
    with state_lock():
        state = load_state()
    task = next((t for t in state["tasks"] if t["id"] == args.id), None)
    if not task:
        sys.exit(f"task {args.id} not found")
    print(f"=== why {args.id} ===")
    print(f"  status:       {task.get('status')}")
    print(f"  description:  {(task.get('description') or '')[:100]}")
    print(f"  signature:    {task.get('signature')}")
    print(f"  priority:     {task.get('priority')}")
    print(f"  preferred:    {task.get('preferred_node')!r}  require:    {task.get('require_node')!r}")
    print(f"  est:          vram={task.get('est_vram_mb')}MB ram={task.get('ram_mb')}MB cpu={task.get('cpu_cores')}")
    if task.get("status") != "queued":
        print()
        print(f"  (task is {task.get('status')!r} — `why` is for diagnosing "
              f"queued tasks that aren't getting placed)")
        return
    print()
    print("  last_block_reason:")
    lbr = task.get("last_block_reason") or "(none yet — task hasn't been considered for placement)"
    print(f"    {lbr}")
    sig = task.get("signature") or ""
    if sig:
        h = load_history()
        own = h.get(sig)
        if own is not None:
            if isinstance(own, int):
                own = {"vram_mb": own}
            print()
            print(f"  history[{sig!r}]:")
            print(f"    vram_mb={own.get('vram_mb', 0)}  ram_mb={own.get('ram_mb', 0)}  "
                  f"cpu_cores={own.get('cpu_cores', 0)}  runs={own.get('dur_s_runs', 0)}")
            samples = own.get("vram_samples") or []
            if samples:
                print(f"    vram_samples={samples}  (single-sample outliers may "
                      f"poison estimate — `history --drop {sig}` to clear)")
        prefix = "/".join(sig.split("/")[:2])
        siblings = []
        for s, v in h.items():
            if s == sig: continue
            if not s.startswith(prefix + "/"): continue
            vm = v.get("vram_mb", 0) if isinstance(v, dict) else v
            siblings.append((s, vm))
        if siblings:
            print()
            print(f"  sibling signatures under {prefix!r}/ (compare against own est={task.get('est_vram_mb')}MB):")
            for s, vm in sorted(siblings, key=lambda kv: -kv[1])[:8]:
                marker = "  ← much smaller, est may be over" if (
                    task.get("est_vram_mb") and vm > 0
                    and task["est_vram_mb"] > 2 * vm) else ""
                print(f"    {s:<55s} vram={vm}MB{marker}")
    print()
    print("  per-node fit analysis (probe_all snapshot):")
    nodes = probe_all()
    slurm_pending_per_node = _count_slurm_pending_per_node(state)
    for n in nodes:
        split = slurm_pending_per_node.get(n["name"]) or {"cpu": 0, "gpu": 0}
        n["slurm_pending_split"] = split
        n["slurm_pending_count"] = int(split.get("cpu", 0)) + int(split.get("gpu", 0))
    for n in nodes:
        print(f"    {n['name']:11s}: {_explain_node_fit(task, n)}")
        if _ClaimManager.enabled_for(n["name"]):
            snap = {
                "claims": n.get("pending_claims") or [],
                "intents": n.get("claim_intents") or [],
                "error": n.get("claim_snapshot_error"),
                "ok": not n.get("claim_snapshot_error"),
            }
            hint = _format_claim_intent_hint_for_task(task, n["name"], snap)
            if hint:
                print(f"      {hint}")
            elif snap.get("error"):
                print(f"      claim-snapshot-error: {snap['error']}")


def cmd_history(args):
    """Phase 3.1: extended with --drop and --set for cleaning poisoned
    peak-history entries (single-outlier samples that make a signature's
    tasks unplaceable forever)."""
    if getattr(args, "drop", None):
        h = load_history()
        if args.drop not in h:
            sys.exit(f"signature {args.drop!r} not in history")
        old = h.pop(args.drop)
        save_history(h)
        print(f"dropped {args.drop!r}:")
        print(f"  was: {old}")
        print(f"  next runs of this signature will accumulate fresh peaks.")
        return
    if getattr(args, "set", None):
        if all(getattr(args, k, None) is None for k in ("vram_mb", "ram_mb", "cpu")):
            sys.exit("--set requires at least one of --vram-mb / --ram-mb / --cpu")
        h = load_history()
        rec = h.get(args.set, {})
        if isinstance(rec, int):
            rec = {"vram_mb": rec}
        elif not isinstance(rec, dict):
            rec = {}
        if args.vram_mb is not None:
            rec["vram_mb"] = int(args.vram_mb)
            rec["vram_samples"] = [int(args.vram_mb)]  # reset noisy sample list
        if args.ram_mb is not None:
            rec["ram_mb"] = int(args.ram_mb)
            rec["ram_samples"] = [int(args.ram_mb)]
        if args.cpu is not None:
            rec["cpu_cores"] = int(args.cpu)
        rec["last_seen"] = int(time.time())
        h[args.set] = rec
        save_history(h)
        print(f"set {args.set!r}:")
        print(f"  {rec}")
        return
    # Default: list (existing behavior)
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
    s.add_argument("--ckpt-dir", help="Checkpoint directory on TARGET node, for resume detection. Must be a dedicated directory, not equal to --cwd.")
    s.add_argument("--result-dir", dest="result_dir",
                   help="Phase 3.5: directory on TARGET node containing the "
                        "experiment results (logs / final models / metrics). "
                        "On task completion, scheduleurm rsyncs this dir back "
                        "to local (delta sync; no ckpts unless they live here). "
                        "Set this to opt in. Must be a dedicated directory, not equal to --cwd; intermediate ckpts should stay in "
                        "--ckpt-dir which is NOT synced automatically.")
    s.add_argument("--local-result-dir", dest="local_result_dir",
                   help="Phase 3.5: where on local to land the rsync'd results. "
                        "Defaults to mirroring the remote path (same absolute "
                        "path on local as on the target). Use a per-task "
                        "subdirectory; exact sharing is refused unless "
                        "--allow-shared-result-dir is passed.")
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
                   help="Override the active-task ckpt-dir conflict refusal. Use only when "
                        "multiple tasks deliberately share a ckpt directory (e.g. distributed "
                        "training reading a shared warm-start ckpt) — concurrent writers will "
                        "still corrupt each other unless coordinated.")
    s.add_argument("--allow-shared-result-dir", dest="allow_shared_result_dir", action="store_true",
                   help="Override the local result destination conflict refusal. Use only when "
                        "multiple result syncs intentionally land in the same directory and their "
                        "file layouts cannot overwrite each other.")
    s.add_argument("--allow-duplicate", dest="allow_duplicate", action="store_true",
                   help="Allow submission even when run identity matches an existing queued/launching/running task")
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

    s = sub.add_parser("claims", help="Show remote shared claims/intents for claims-enabled nodes")
    s.add_argument("--node", choices=list(NODES.keys()), help="Limit to one node")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_claims)

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
    s.add_argument("--task-id", dest="task_ids", nargs="*", default=[],
                   help="Optional task IDs to rebalance; default is all slurm-PENDING tasks")
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

    s = sub.add_parser("history", help="Show / edit VRAM history per signature")
    s.add_argument("--drop", metavar="SIG",
                   help="Remove the history entry for SIG (next runs will start fresh)")
    s.add_argument("--set", metavar="SIG",
                   help="Set / overwrite the history entry for SIG (use with --vram-mb / --ram-mb / --cpu)")
    s.add_argument("--vram-mb", dest="vram_mb", type=int,
                   help="With --set: peak VRAM in MB to record")
    s.add_argument("--ram-mb", dest="ram_mb", type=int,
                   help="With --set: peak RAM in MB to record")
    s.add_argument("--cpu", type=int,
                   help="With --set: cpu_cores to record")
    s.set_defaults(func=cmd_history)

    s = sub.add_parser("priority", help="Change priority of a queued task (queue ordering)")
    s.add_argument("id", help="Task id (e.g. t0042)")
    s.add_argument("level", choices=["low", "normal", "high"])
    s.set_defaults(func=cmd_priority)

    s = sub.add_parser("edit", help="Override resource estimates / pin on a queued task")
    s.add_argument("id", help="Task id (e.g. t0042)")
    s.add_argument("--vram-mb", dest="vram_mb", type=int,
                   help="Override estimated peak VRAM in MB")
    s.add_argument("--ram-mb", dest="ram_mb", type=int,
                   help="Override estimated peak RAM in MB")
    s.add_argument("--cpu", type=int, help="Override CPU cores")
    s.add_argument("--description", help="Override description")
    s.add_argument("--preferred-node", dest="preferred_node",
                   help="Set / change soft preferred node (must be a known node)")
    s.add_argument("--require-node", dest="require_node",
                   help="Set / change hard pin (use empty string to clear)")
    s.set_defaults(func=cmd_edit)

    s = sub.add_parser("why", help="Diagnose why a queued task isn't being dispatched")
    s.add_argument("id", help="Task id (e.g. t0042)")
    s.set_defaults(func=cmd_why)

    sub.add_parser("tui", help="Interactive TUI: sortable + filterable + auto-refresh task table").set_defaults(func=cmd_tui)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
